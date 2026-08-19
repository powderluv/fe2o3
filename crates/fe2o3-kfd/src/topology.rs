//! Strict, read-only discovery of the reviewed KFD `gfx942` topology profile.
//!
//! Values returned by this module are contracted sysfs observations. They are
//! not authenticated device identity and grant no VM, memory, queue, or ioctl
//! authority.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Kernel-owned topology tree used by the first Linux KFD profile.
pub const DEFAULT_TOPOLOGY_ROOT: &str = "/sys/class/kfd/kfd/topology";
pub const DEFAULT_BOOT_ID_PATH: &str = "/proc/sys/kernel/random/boot_id";
pub const DEFAULT_OS_RELEASE_PATH: &str = "/proc/sys/kernel/osrelease";
pub const DEFAULT_AMDGPU_MODULE_ROOT: &str = "/sys/module/amdgpu";
pub const DEFAULT_DEVICE_CHARACTER_ROOT: &str = "/sys/dev/char";
pub const DEFAULT_SYSFS_DEVICES_ROOT: &str = "/sys/devices";

const MAX_TOPOLOGY_NODES: usize = 256;
const MAX_ROOT_ENTRIES: usize = 8;
const MAX_NODE_ENTRIES: usize = 16;
const MAX_SCALAR_BYTES: usize = 64;
const MAX_NAME_BYTES: usize = 128;
const MAX_PROPERTY_BYTES: usize = 4096;
const MAX_PROPERTY_LINES: usize = 64;
const MAX_MODULE_FIELD_BYTES: usize = 128;
const EXPECTED_AMD_VENDOR_ID: u64 = 0x1002;
const GFX942_TARGET_VERSION: u64 = 90_402;
const MIN_DRM_RENDER_MINOR: u64 = 128;
const MAX_DRM_RENDER_MINOR: u64 = 255;

/// The only GPU target admitted by the initial direct-KFD runtime profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GfxTarget {
    Gfx942,
}

impl GfxTarget {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gfx942 => "gfx942",
        }
    }

    pub const fn encoded_version(self) -> u32 {
        match self {
            Self::Gfx942 => GFX942_TARGET_VERSION as u32,
        }
    }
}

/// Strictly parsed Linux boot UUID used to separate host incarnations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BootId([u8; 16]);

impl BootId {
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BootIdParseError;

impl fmt::Display for BootIdParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("boot id must be a lowercase canonical UUID")
    }
}

impl std::error::Error for BootIdParseError {}

impl FromStr for BootId {
    type Err = BootIdParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 36
            || value.as_bytes()[8] != b'-'
            || value.as_bytes()[13] != b'-'
            || value.as_bytes()[18] != b'-'
            || value.as_bytes()[23] != b'-'
        {
            return Err(BootIdParseError);
        }
        let mut bytes = [0_u8; 16];
        let mut output = 0;
        let mut input = 0;
        while output < bytes.len() {
            if matches!(input, 8 | 13 | 18 | 23) {
                input += 1;
            }
            let high = lowercase_hex_digit(value.as_bytes()[input]).ok_or(BootIdParseError)?;
            let low = lowercase_hex_digit(value.as_bytes()[input + 1]).ok_or(BootIdParseError)?;
            bytes[output] = high << 4 | low;
            output += 1;
            input += 2;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for BootId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, byte) in self.0.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                formatter.write_str("-")?;
            }
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Strict, bounded Linux kernel release identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelRelease(String);

impl KernelRelease {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelReleaseParseError;

impl fmt::Display for KernelReleaseParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("kernel release is empty, oversized, or contains unsupported bytes")
    }
}

impl std::error::Error for KernelReleaseParseError {}

impl FromStr for KernelRelease {
    type Err = KernelReleaseParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty()
            || value.len() > MAX_MODULE_FIELD_BYTES
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-')
            })
        {
            return Err(KernelReleaseParseError);
        }
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for KernelRelease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Read-only identity fields for the loaded `amdgpu` kernel module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmdgpuModuleObservation {
    file_system_device: u64,
    inode: u64,
    version: Option<String>,
    srcversion: Option<String>,
    mes: Option<i32>,
    sched_policy: Option<i32>,
    cwsr_enable: Option<i32>,
}

impl AmdgpuModuleObservation {
    pub const fn file_system_device(&self) -> u64 {
        self.file_system_device
    }

    pub const fn inode(&self) -> u64 {
        self.inode
    }

    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    pub fn srcversion(&self) -> Option<&str> {
        self.srcversion.as_deref()
    }

    /// Raw read-only module-parameter observation.
    pub const fn mes(&self) -> Option<i32> {
        self.mes
    }

    /// Raw read-only module-parameter observation.
    pub const fn sched_policy(&self) -> Option<i32> {
        self.sched_policy
    }

    /// Raw read-only module-parameter observation.
    pub const fn cwsr_enable(&self) -> Option<i32> {
        self.cwsr_enable
    }
}

/// A strict PCI address correlated with a KFD location observation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PciAddress {
    domain: u16,
    bus: u8,
    device: u8,
    function: u8,
}

impl PciAddress {
    pub const fn domain(self) -> u16 {
        self.domain
    }

    pub const fn bus(self) -> u8 {
        self.bus
    }

    pub const fn device(self) -> u8 {
        self.device
    }

    pub const fn function(self) -> u8 {
        self.function
    }

    const fn kfd_location_id(self) -> u32 {
        (self.bus as u32) << 8 | (self.device as u32) << 3 | self.function as u32
    }
}

impl fmt::Display for PciAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputePartition {
    Spx,
    Dpx,
    Tpx,
    Qpx,
    Cpx,
}

impl ComputePartition {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Spx => "SPX",
            Self::Dpx => "DPX",
            Self::Tpx => "TPX",
            Self::Qpx => "QPX",
            Self::Cpx => "CPX",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryPartition {
    Nps1,
    Nps2,
    Nps4,
    Nps8,
}

impl MemoryPartition {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nps1 => "NPS1",
            Self::Nps2 => "NPS2",
            Self::Nps4 => "NPS4",
            Self::Nps8 => "NPS8",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PartitionProfile {
    compute: ComputePartition,
    memory: MemoryPartition,
}

/// Partition profile accepted by the initial `gfx942` runtime admission gate.
pub const V1_PARTITION_PROFILE: PartitionProfile = PartitionProfile {
    compute: ComputePartition::Spx,
    memory: MemoryPartition::Nps1,
};

impl PartitionProfile {
    pub const fn compute(self) -> ComputePartition {
        self.compute
    }

    pub const fn memory(self) -> MemoryPartition {
        self.memory
    }
}

/// DRM and PCI evidence correlated with one KFD GPU observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderNodeObservation {
    node_id: u32,
    drm_render_minor: u16,
    canonical_sysfs_path: PathBuf,
    pci_address: PciAddress,
    pci_revision: u8,
    unique_id: u64,
    partition: PartitionProfile,
}

impl RenderNodeObservation {
    pub const fn node_id(&self) -> u32 {
        self.node_id
    }

    pub const fn drm_render_minor(&self) -> u16 {
        self.drm_render_minor
    }

    pub fn canonical_sysfs_path(&self) -> &Path {
        &self.canonical_sysfs_path
    }

    pub const fn pci_address(&self) -> PciAddress {
        self.pci_address
    }

    pub const fn pci_revision(&self) -> u8 {
        self.pci_revision
    }

    pub const fn unique_id(&self) -> u64 {
        self.unique_id
    }

    pub const fn partition(&self) -> PartitionProfile {
        self.partition
    }
}

/// Platform fields reported alongside one KFD topology generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlatformObservation {
    oem: u64,
    id: u64,
    revision: u32,
}

impl PlatformObservation {
    pub const fn oem(self) -> u64 {
        self.oem
    }

    pub const fn id(self) -> u64 {
        self.id
    }

    pub const fn revision(self) -> u32 {
        self.revision
    }
}

/// Filesystem and generation provenance for a topology snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyProvenance {
    root: PathBuf,
    file_system_device: u64,
    inode: u64,
    generation: u64,
    platform: PlatformObservation,
}

impl TopologyProvenance {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn file_system_device(&self) -> u64 {
        self.file_system_device
    }

    pub const fn inode(&self) -> u64 {
        self.inode
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn platform(&self) -> PlatformObservation {
        self.platform
    }
}

/// Bounded capacity fields checked while observing a GPU node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuCapacityObservation {
    simd_count: u32,
    simd_per_cu: u32,
    array_count: u32,
    simd_arrays_per_engine: u32,
    lds_size_in_kb: u32,
    max_waves_per_simd: u32,
    compute_queue_count: u32,
    memory_bank_count: u32,
    cache_count: u32,
    io_link_count: u32,
    p2p_link_count: u32,
    wavefront_size: u32,
    xcc_count: u32,
}

impl GpuCapacityObservation {
    pub const fn simd_count(self) -> u32 {
        self.simd_count
    }

    pub const fn simd_per_cu(self) -> u32 {
        self.simd_per_cu
    }

    pub const fn array_count(self) -> u32 {
        self.array_count
    }

    pub const fn simd_arrays_per_engine(self) -> u32 {
        self.simd_arrays_per_engine
    }

    pub const fn lds_size_in_kb(self) -> u32 {
        self.lds_size_in_kb
    }

    pub const fn max_waves_per_simd(self) -> u32 {
        self.max_waves_per_simd
    }

    pub const fn compute_queue_count(self) -> u32 {
        self.compute_queue_count
    }

    pub const fn memory_bank_count(self) -> u32 {
        self.memory_bank_count
    }

    pub const fn cache_count(self) -> u32 {
        self.cache_count
    }

    pub const fn io_link_count(self) -> u32 {
        self.io_link_count
    }

    pub const fn p2p_link_count(self) -> u32 {
        self.p2p_link_count
    }

    pub const fn wavefront_size(self) -> u32 {
        self.wavefront_size
    }

    pub const fn xcc_count(self) -> u32 {
        self.xcc_count
    }
}

/// One GPU identity observation from a stable KFD topology generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuTopologyNode {
    node_id: u32,
    gpu_id: u64,
    name: String,
    target: GfxTarget,
    pci_device_id: u16,
    drm_render_minor: u16,
    unique_id: u64,
    hive_id: u64,
    location_id: u32,
    domain: u16,
    fw_version: u32,
    sdma_fw_version: u32,
    capacity: GpuCapacityObservation,
}

impl GpuTopologyNode {
    pub const fn node_id(&self) -> u32 {
        self.node_id
    }

    pub const fn gpu_id(&self) -> u64 {
        self.gpu_id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn target(&self) -> GfxTarget {
        self.target
    }

    pub const fn pci_device_id(&self) -> u16 {
        self.pci_device_id
    }

    pub const fn drm_render_minor(&self) -> u16 {
        self.drm_render_minor
    }

    pub const fn unique_id(&self) -> u64 {
        self.unique_id
    }

    pub const fn hive_id(&self) -> u64 {
        self.hive_id
    }

    pub const fn location_id(&self) -> u32 {
        self.location_id
    }

    pub const fn domain(&self) -> u16 {
        self.domain
    }

    /// Opaque firmware version reported by the KFD topology contract.
    pub const fn fw_version(&self) -> u32 {
        self.fw_version
    }

    /// Opaque SDMA firmware version reported by the KFD topology contract.
    pub const fn sdma_fw_version(&self) -> u32 {
        self.sdma_fw_version
    }

    pub const fn capacity(&self) -> GpuCapacityObservation {
        self.capacity
    }
}

/// A generation-consistent topology observation with no operational authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologySnapshot {
    provenance: TopologyProvenance,
    observed_node_count: usize,
    gpu_nodes: Vec<GpuTopologyNode>,
}

impl TopologySnapshot {
    pub fn provenance(&self) -> &TopologyProvenance {
        &self.provenance
    }

    pub const fn observed_node_count(&self) -> usize {
        self.observed_node_count
    }

    pub fn gpu_nodes(&self) -> &[GpuTopologyNode] {
        &self.gpu_nodes
    }
}

/// A topology snapshot correlated with boot, module, DRM, and PCI evidence.
///
/// Correlation is still an observation, not a device capability or proof of
/// kernel correctness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostTopologySnapshot {
    topology: TopologySnapshot,
    boot_id: BootId,
    kernel_release: KernelRelease,
    amdgpu_module: AmdgpuModuleObservation,
    render_nodes: Vec<RenderNodeObservation>,
}

impl HostTopologySnapshot {
    pub fn topology(&self) -> &TopologySnapshot {
        &self.topology
    }

    pub const fn boot_id(&self) -> BootId {
        self.boot_id
    }

    pub fn kernel_release(&self) -> &KernelRelease {
        &self.kernel_release
    }

    pub fn amdgpu_module(&self) -> &AmdgpuModuleObservation {
        &self.amdgpu_module
    }

    pub fn render_nodes(&self) -> &[RenderNodeObservation] {
        &self.render_nodes
    }
}

/// Discovers the default KFD topology without opening a device or granting
/// runtime authority.
pub fn discover_default_topology() -> Result<HostTopologySnapshot, TopologyError> {
    discover_host_topology(&DiscoveryPaths {
        topology_root: Path::new(DEFAULT_TOPOLOGY_ROOT),
        boot_id: Path::new(DEFAULT_BOOT_ID_PATH),
        os_release: Path::new(DEFAULT_OS_RELEASE_PATH),
        amdgpu_module_root: Path::new(DEFAULT_AMDGPU_MODULE_ROOT),
        device_character_root: Path::new(DEFAULT_DEVICE_CHARACTER_ROOT),
        sysfs_devices_root: Path::new(DEFAULT_SYSFS_DEVICES_ROOT),
    })
}

#[derive(Debug)]
pub enum TopologyError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Symlink(PathBuf),
    UnexpectedFileType {
        path: PathBuf,
        expected: &'static str,
    },
    ChangedDuringRead(PathBuf),
    TooManyEntries {
        path: PathBuf,
        maximum: usize,
    },
    UnexpectedEntry {
        path: PathBuf,
        name: String,
    },
    InvalidEntryName(PathBuf),
    MissingEntry(PathBuf),
    FileTooLarge {
        path: PathBuf,
        maximum: usize,
    },
    InvalidUtf8(PathBuf),
    InvalidScalar {
        path: PathBuf,
        value: String,
    },
    InvalidNodeId(String),
    MalformedPropertyLine {
        path: PathBuf,
        line: usize,
    },
    TooManyProperties {
        path: PathBuf,
        maximum: usize,
    },
    UnknownProperty {
        path: PathBuf,
        key: String,
    },
    DuplicateProperty {
        path: PathBuf,
        key: String,
    },
    MissingProperty {
        path: PathBuf,
        key: &'static str,
    },
    PropertyOutOfRange {
        path: PathBuf,
        key: String,
        value: u64,
        minimum: u64,
        maximum: u64,
    },
    UnsupportedTarget {
        node_id: u32,
        encoded: u64,
    },
    UnsupportedVendor {
        node_id: u32,
        vendor_id: u64,
    },
    DuplicateIdentity {
        field: &'static str,
        value: String,
    },
    TopologyChanged {
        before: u64,
        after: u64,
    },
    EmptyGpuSet,
    ExpectedSymlink(PathBuf),
    EscapedSysfsRoot {
        path: PathBuf,
        target: PathBuf,
    },
    InvalidBootId {
        path: PathBuf,
        value: String,
    },
    InvalidKernelRelease {
        path: PathBuf,
        value: String,
    },
    InvalidModuleField {
        path: PathBuf,
        value: String,
    },
    InvalidHexScalar {
        path: PathBuf,
        value: String,
    },
    InvalidPciAddress(String),
    RenderCorrelationMismatch {
        node_id: u32,
        field: &'static str,
        kfd: String,
        render: String,
    },
    UnsupportedPartition {
        path: PathBuf,
        value: String,
    },
}

impl fmt::Display for TopologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} {}: {source}",
                path.display()
            ),
            Self::Symlink(path) => write!(formatter, "symlink is prohibited: {}", path.display()),
            Self::UnexpectedFileType { path, expected } => {
                write!(formatter, "{} is not a {expected}", path.display())
            }
            Self::ChangedDuringRead(path) => {
                write!(
                    formatter,
                    "{} changed during topology discovery",
                    path.display()
                )
            }
            Self::TooManyEntries { path, maximum } => write!(
                formatter,
                "{} contains more than {maximum} entries",
                path.display()
            ),
            Self::UnexpectedEntry { path, name } => {
                write!(
                    formatter,
                    "unexpected topology entry {name:?} in {}",
                    path.display()
                )
            }
            Self::InvalidEntryName(path) => {
                write!(
                    formatter,
                    "topology entry is not valid UTF-8: {}",
                    path.display()
                )
            }
            Self::MissingEntry(path) => {
                write!(
                    formatter,
                    "required topology entry is missing: {}",
                    path.display()
                )
            }
            Self::FileTooLarge { path, maximum } => {
                write!(formatter, "{} exceeds {maximum} bytes", path.display())
            }
            Self::InvalidUtf8(path) => {
                write!(formatter, "{} is not valid UTF-8", path.display())
            }
            Self::InvalidScalar { path, value } => {
                write!(
                    formatter,
                    "{} has invalid decimal value {value:?}",
                    path.display()
                )
            }
            Self::InvalidNodeId(value) => write!(formatter, "invalid KFD node id {value:?}"),
            Self::MalformedPropertyLine { path, line } => write!(
                formatter,
                "{} has a malformed property at line {line}",
                path.display()
            ),
            Self::TooManyProperties { path, maximum } => write!(
                formatter,
                "{} contains more than {maximum} properties",
                path.display()
            ),
            Self::UnknownProperty { path, key } => write!(
                formatter,
                "{} contains unknown property {key:?}",
                path.display()
            ),
            Self::DuplicateProperty { path, key } => write!(
                formatter,
                "{} contains duplicate property {key:?}",
                path.display()
            ),
            Self::MissingProperty { path, key } => {
                write!(formatter, "{} is missing property {key:?}", path.display())
            }
            Self::PropertyOutOfRange {
                path,
                key,
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "{} property {key:?} value {value} is outside {minimum}..={maximum}",
                path.display()
            ),
            Self::UnsupportedTarget { node_id, encoded } => write!(
                formatter,
                "KFD node {node_id} reports unsupported gfx target {encoded}"
            ),
            Self::UnsupportedVendor { node_id, vendor_id } => write!(
                formatter,
                "KFD node {node_id} reports unsupported vendor {vendor_id:#x}"
            ),
            Self::DuplicateIdentity { field, value } => {
                write!(formatter, "duplicate GPU {field} identity {value}")
            }
            Self::TopologyChanged { before, after } => write!(
                formatter,
                "KFD topology generation changed from {before} to {after}"
            ),
            Self::EmptyGpuSet => formatter.write_str("KFD topology contains no GPU nodes"),
            Self::ExpectedSymlink(path) => {
                write!(
                    formatter,
                    "expected a kernel sysfs symlink: {}",
                    path.display()
                )
            }
            Self::EscapedSysfsRoot { path, target } => write!(
                formatter,
                "{} resolves outside the admitted sysfs root: {}",
                path.display(),
                target.display()
            ),
            Self::InvalidBootId { path, value } => write!(
                formatter,
                "{} contains invalid boot id {value:?}",
                path.display()
            ),
            Self::InvalidKernelRelease { path, value } => write!(
                formatter,
                "{} contains invalid kernel release {value:?}",
                path.display()
            ),
            Self::InvalidModuleField { path, value } => write!(
                formatter,
                "{} contains invalid module field {value:?}",
                path.display()
            ),
            Self::InvalidHexScalar { path, value } => write!(
                formatter,
                "{} contains invalid hexadecimal scalar {value:?}",
                path.display()
            ),
            Self::InvalidPciAddress(value) => {
                write!(formatter, "invalid canonical PCI address {value:?}")
            }
            Self::RenderCorrelationMismatch {
                node_id,
                field,
                kfd,
                render,
            } => write!(
                formatter,
                "KFD node {node_id} {field} mismatch: topology={kfd}, render={render}"
            ),
            Self::UnsupportedPartition { path, value } => write!(
                formatter,
                "{} reports unsupported partition mode {value:?}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TopologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            size: metadata.size(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> TopologyError {
    TopologyError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

fn inspect(path: &Path) -> Result<Metadata, TopologyError> {
    fs::symlink_metadata(path).map_err(|source| io_error("inspect", path, source))
}

fn ensure_directory(path: &Path) -> Result<FileIdentity, TopologyError> {
    let metadata = inspect(path)?;
    if metadata.file_type().is_symlink() {
        return Err(TopologyError::Symlink(path.to_path_buf()));
    }
    if !metadata.is_dir() {
        return Err(TopologyError::UnexpectedFileType {
            path: path.to_path_buf(),
            expected: "directory",
        });
    }
    Ok(FileIdentity::from_metadata(&metadata))
}

fn read_bounded_regular(path: &Path, maximum: usize) -> Result<Vec<u8>, TopologyError> {
    let before = inspect(path)?;
    if before.file_type().is_symlink() {
        return Err(TopologyError::Symlink(path.to_path_buf()));
    }
    if !before.is_file() {
        return Err(TopologyError::UnexpectedFileType {
            path: path.to_path_buf(),
            expected: "regular file",
        });
    }
    let expected = FileIdentity::from_metadata(&before);
    let mut file = File::open(path).map_err(|source| io_error("open", path, source))?;
    let opened = file
        .metadata()
        .map_err(|source| io_error("inspect opened file", path, source))?;
    if FileIdentity::from_metadata(&opened) != expected {
        return Err(TopologyError::ChangedDuringRead(path.to_path_buf()));
    }
    let mut bytes = Vec::with_capacity(maximum.min(1024));
    (&mut file)
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|source| io_error("read", path, source))?;
    if bytes.len() > maximum {
        return Err(TopologyError::FileTooLarge {
            path: path.to_path_buf(),
            maximum,
        });
    }
    let after = file
        .metadata()
        .map_err(|source| io_error("reinspect opened file", path, source))?;
    if FileIdentity::from_metadata(&after) != expected {
        return Err(TopologyError::ChangedDuringRead(path.to_path_buf()));
    }
    Ok(bytes)
}

fn read_text(path: &Path, maximum: usize) -> Result<String, TopologyError> {
    String::from_utf8(read_bounded_regular(path, maximum)?)
        .map_err(|_| TopologyError::InvalidUtf8(path.to_path_buf()))
}

fn parse_canonical_decimal(path: &Path, text: &str) -> Result<u64, TopologyError> {
    let value = text.strip_suffix('\n').unwrap_or(text);
    if value.is_empty()
        || value.starts_with('0') && value != "0"
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || text != value && text != format!("{value}\n")
    {
        return Err(TopologyError::InvalidScalar {
            path: path.to_path_buf(),
            value: text.to_owned(),
        });
    }
    value.parse().map_err(|_| TopologyError::InvalidScalar {
        path: path.to_path_buf(),
        value: value.to_owned(),
    })
}

fn read_scalar(path: &Path) -> Result<u64, TopologyError> {
    let text = read_text(path, MAX_SCALAR_BYTES)?;
    parse_canonical_decimal(path, &text)
}

fn read_name(path: &Path) -> Result<String, TopologyError> {
    let text = read_text(path, MAX_NAME_BYTES)?;
    let value = text.strip_suffix('\n').unwrap_or(&text);
    if value.bytes().any(|byte| byte.is_ascii_control())
        || text != value && text != format!("{value}\n")
    {
        return Err(TopologyError::InvalidScalar {
            path: path.to_path_buf(),
            value: text,
        });
    }
    Ok(value.to_owned())
}

fn lowercase_hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn strip_one_newline<'a>(path: &Path, text: &'a str) -> Result<&'a str, TopologyError> {
    let value = text.strip_suffix('\n').unwrap_or(text);
    if text != value && text != format!("{value}\n") {
        return Err(TopologyError::InvalidScalar {
            path: path.to_path_buf(),
            value: text.to_owned(),
        });
    }
    Ok(value)
}

fn read_boot_id(path: &Path) -> Result<BootId, TopologyError> {
    let text = read_text(path, MAX_NAME_BYTES)?;
    let value = strip_one_newline(path, &text)?;
    value.parse().map_err(|_| TopologyError::InvalidBootId {
        path: path.to_path_buf(),
        value: value.to_owned(),
    })
}

fn read_kernel_release(path: &Path) -> Result<KernelRelease, TopologyError> {
    let text = read_text(path, MAX_MODULE_FIELD_BYTES)?;
    let value = strip_one_newline(path, &text)?;
    value
        .parse()
        .map_err(|_| TopologyError::InvalidKernelRelease {
            path: path.to_path_buf(),
            value: value.to_owned(),
        })
}

fn read_optional_module_field(
    path: &Path,
    predicate: impl Fn(u8) -> bool,
) -> Result<Option<String>, TopologyError> {
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(io_error("inspect", path, source)),
    }
    let text = read_text(path, MAX_MODULE_FIELD_BYTES)?;
    let value = strip_one_newline(path, &text)?;
    if value.is_empty() || !value.bytes().all(predicate) {
        return Err(TopologyError::InvalidModuleField {
            path: path.to_path_buf(),
            value: value.to_owned(),
        });
    }
    Ok(Some(value.to_owned()))
}

fn read_optional_module_i32(path: &Path) -> Result<Option<i32>, TopologyError> {
    let Some(value) =
        read_optional_module_field(path, |byte| byte.is_ascii_digit() || byte == b'-')?
    else {
        return Ok(None);
    };
    let digits = value.strip_prefix('-').unwrap_or(&value);
    if digits.is_empty()
        || digits.starts_with('0') && digits != "0"
        || value == "-0"
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(TopologyError::InvalidModuleField {
            path: path.to_path_buf(),
            value,
        });
    }
    value
        .parse()
        .map(Some)
        .map_err(|_| TopologyError::InvalidModuleField {
            path: path.to_path_buf(),
            value,
        })
}

fn observe_amdgpu_module(path: &Path) -> Result<AmdgpuModuleObservation, TopologyError> {
    let identity = ensure_directory(path)?;
    let version = read_optional_module_field(&path.join("version"), |byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-')
    })?;
    let srcversion = read_optional_module_field(&path.join("srcversion"), |byte| {
        byte.is_ascii_digit() || matches!(byte, b'A'..=b'F')
    })?;
    let parameters = path.join("parameters");
    let mes = read_optional_module_i32(&parameters.join("mes"))?;
    let sched_policy = read_optional_module_i32(&parameters.join("sched_policy"))?;
    let cwsr_enable = read_optional_module_i32(&parameters.join("cwsr_enable"))?;
    if ensure_directory(path)? != identity {
        return Err(TopologyError::ChangedDuringRead(path.to_path_buf()));
    }
    Ok(AmdgpuModuleObservation {
        file_system_device: identity.device,
        inode: identity.inode,
        version,
        srcversion,
        mes,
        sched_policy,
        cwsr_enable,
    })
}

fn read_hex_scalar(
    path: &Path,
    required_prefix: bool,
    exact_digits: Option<usize>,
) -> Result<u64, TopologyError> {
    let text = read_text(path, MAX_SCALAR_BYTES)?;
    let value = strip_one_newline(path, &text)?;
    let digits = if required_prefix {
        value
            .strip_prefix("0x")
            .ok_or_else(|| TopologyError::InvalidHexScalar {
                path: path.to_path_buf(),
                value: value.to_owned(),
            })?
    } else {
        value
    };
    if digits.is_empty()
        || exact_digits.is_some_and(|width| digits.len() != width)
        || !digits
            .bytes()
            .all(|byte| lowercase_hex_digit(byte).is_some())
    {
        return Err(TopologyError::InvalidHexScalar {
            path: path.to_path_buf(),
            value: value.to_owned(),
        });
    }
    u64::from_str_radix(digits, 16).map_err(|_| TopologyError::InvalidHexScalar {
        path: path.to_path_buf(),
        value: value.to_owned(),
    })
}

fn parse_pci_address(value: &str) -> Result<PciAddress, TopologyError> {
    if value.len() != 12
        || value.as_bytes()[4] != b':'
        || value.as_bytes()[7] != b':'
        || value.as_bytes()[10] != b'.'
        || !value
            .bytes()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7 | 10) || lowercase_hex_digit(byte).is_some())
    {
        return Err(TopologyError::InvalidPciAddress(value.to_owned()));
    }
    let domain = u16::from_str_radix(&value[0..4], 16)
        .map_err(|_| TopologyError::InvalidPciAddress(value.to_owned()))?;
    let bus = u8::from_str_radix(&value[5..7], 16)
        .map_err(|_| TopologyError::InvalidPciAddress(value.to_owned()))?;
    let device = u8::from_str_radix(&value[8..10], 16)
        .map_err(|_| TopologyError::InvalidPciAddress(value.to_owned()))?;
    let function = u8::from_str_radix(&value[11..12], 16)
        .map_err(|_| TopologyError::InvalidPciAddress(value.to_owned()))?;
    if device > 31 || function > 7 {
        return Err(TopologyError::InvalidPciAddress(value.to_owned()));
    }
    Ok(PciAddress {
        domain,
        bus,
        device,
        function,
    })
}

fn read_compute_partition(path: &Path) -> Result<ComputePartition, TopologyError> {
    let value = read_name(path)?;
    match value.as_str() {
        "SPX" => Ok(ComputePartition::Spx),
        "DPX" => Ok(ComputePartition::Dpx),
        "TPX" => Ok(ComputePartition::Tpx),
        "QPX" => Ok(ComputePartition::Qpx),
        "CPX" => Ok(ComputePartition::Cpx),
        _ => Err(TopologyError::UnsupportedPartition {
            path: path.to_path_buf(),
            value,
        }),
    }
}

fn read_memory_partition(path: &Path) -> Result<MemoryPartition, TopologyError> {
    let value = read_name(path)?;
    match value.as_str() {
        "NPS1" => Ok(MemoryPartition::Nps1),
        "NPS2" => Ok(MemoryPartition::Nps2),
        "NPS4" => Ok(MemoryPartition::Nps4),
        "NPS8" => Ok(MemoryPartition::Nps8),
        _ => Err(TopologyError::UnsupportedPartition {
            path: path.to_path_buf(),
            value,
        }),
    }
}

fn property_range(key: &str) -> Option<(u64, u64)> {
    let range = match key {
        "array_count" => (0, 4096),
        "caches_count" => (0, 16_384),
        "capability" | "capability2" | "hive_id" | "unique_id" => (0, u64::MAX),
        "cpu_core_id_base" | "simd_id_base" | "location_id" => (0, u32::MAX as u64),
        "cpu_cores_count" | "simd_count" => (0, 65_536),
        "cu_per_simd_array" | "max_waves_per_simd" | "simd_arrays_per_engine" => (0, 4096),
        "debug_prop" => (0, u64::MAX),
        "device_id" | "domain" | "vendor_id" => (0, u16::MAX as u64),
        "drm_render_minor" => (0, u32::MAX as u64),
        "fw_version" | "sdma_fw_version" => (0, u32::MAX as u64),
        "gds_size_in_kb" | "lds_size_in_kb" => (0, 1 << 30),
        "gfx_target_version" => (0, 999_999),
        "io_links_count" | "mem_banks_count" | "p2p_links_count" => (0, 4096),
        "local_mem_size" => (0, 1 << 60),
        "max_engine_clk_ccompute" | "max_engine_clk_fcompute" => (0, 100_000_000),
        "max_slots_scratch_cu" | "num_cp_queues" | "num_gws" => (0, 1 << 20),
        "num_sdma_engines" | "num_sdma_queues_per_engine" | "num_sdma_xgmi_engines" => (0, 4096),
        "num_xcc" => (0, 64),
        "simd_per_cu" => (0, 64),
        "wave_front_size" => (0, 128),
        _ => return None,
    };
    Some(range)
}

fn parse_properties(path: &Path) -> Result<BTreeMap<String, u64>, TopologyError> {
    let text = read_text(path, MAX_PROPERTY_BYTES)?;
    if !text.ends_with('\n') {
        return Err(TopologyError::MalformedPropertyLine {
            path: path.to_path_buf(),
            line: 1,
        });
    }
    let mut properties = BTreeMap::new();
    for (index, line) in text.split_terminator('\n').enumerate() {
        let line_number = index + 1;
        if line_number > MAX_PROPERTY_LINES {
            return Err(TopologyError::TooManyProperties {
                path: path.to_path_buf(),
                maximum: MAX_PROPERTY_LINES,
            });
        }
        let mut fields = line.split(' ');
        let Some(key) = fields.next() else {
            return Err(TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: line_number,
            });
        };
        let Some(raw_value) = fields.next() else {
            return Err(TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: line_number,
            });
        };
        if fields.next().is_some()
            || key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            || raw_value.is_empty()
            || raw_value.starts_with('0') && raw_value != "0"
            || !raw_value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: line_number,
            });
        }
        let Some((minimum, maximum)) = property_range(key) else {
            return Err(TopologyError::UnknownProperty {
                path: path.to_path_buf(),
                key: key.to_owned(),
            });
        };
        let value = raw_value
            .parse::<u64>()
            .map_err(|_| TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: line_number,
            })?;
        if value < minimum || value > maximum {
            return Err(TopologyError::PropertyOutOfRange {
                path: path.to_path_buf(),
                key: key.to_owned(),
                value,
                minimum,
                maximum,
            });
        }
        if properties.insert(key.to_owned(), value).is_some() {
            return Err(TopologyError::DuplicateProperty {
                path: path.to_path_buf(),
                key: key.to_owned(),
            });
        }
    }
    Ok(properties)
}

fn required_property(
    properties: &BTreeMap<String, u64>,
    path: &Path,
    key: &'static str,
) -> Result<u64, TopologyError> {
    properties
        .get(key)
        .copied()
        .ok_or_else(|| TopologyError::MissingProperty {
            path: path.to_path_buf(),
            key,
        })
}

fn bounded_u32(
    properties: &BTreeMap<String, u64>,
    path: &Path,
    key: &'static str,
    minimum: u64,
    maximum: u64,
) -> Result<u32, TopologyError> {
    let value = required_property(properties, path, key)?;
    if value < minimum || value > maximum {
        return Err(TopologyError::PropertyOutOfRange {
            path: path.to_path_buf(),
            key: key.to_owned(),
            value,
            minimum,
            maximum,
        });
    }
    Ok(value as u32)
}

fn parse_platform(path: &Path) -> Result<PlatformObservation, TopologyError> {
    let properties = parse_named_properties(
        path,
        &[
            ("platform_id", 1, u64::MAX),
            ("platform_oem", 1, u64::MAX),
            ("platform_rev", 0, u32::MAX as u64),
        ],
    )?;
    Ok(PlatformObservation {
        oem: properties["platform_oem"],
        id: properties["platform_id"],
        revision: properties["platform_rev"] as u32,
    })
}

fn parse_named_properties(
    path: &Path,
    allowed: &[(&'static str, u64, u64)],
) -> Result<BTreeMap<String, u64>, TopologyError> {
    let text = read_text(path, MAX_PROPERTY_BYTES)?;
    if !text.ends_with('\n') {
        return Err(TopologyError::MalformedPropertyLine {
            path: path.to_path_buf(),
            line: 1,
        });
    }
    let mut result = BTreeMap::new();
    for (index, line) in text.split_terminator('\n').enumerate() {
        if index >= allowed.len() {
            return Err(TopologyError::TooManyProperties {
                path: path.to_path_buf(),
                maximum: allowed.len(),
            });
        }
        let Some((key, raw_value)) = line.split_once(' ') else {
            return Err(TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: index + 1,
            });
        };
        if key.is_empty()
            || raw_value.is_empty()
            || raw_value.contains(' ')
            || raw_value.starts_with('0') && raw_value != "0"
            || !raw_value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: index + 1,
            });
        }
        let Some((_, minimum, maximum)) = allowed.iter().find(|entry| entry.0 == key) else {
            return Err(TopologyError::UnknownProperty {
                path: path.to_path_buf(),
                key: key.to_owned(),
            });
        };
        let value = raw_value
            .parse::<u64>()
            .map_err(|_| TopologyError::MalformedPropertyLine {
                path: path.to_path_buf(),
                line: index + 1,
            })?;
        if value < *minimum || value > *maximum {
            return Err(TopologyError::PropertyOutOfRange {
                path: path.to_path_buf(),
                key: key.to_owned(),
                value,
                minimum: *minimum,
                maximum: *maximum,
            });
        }
        if result.insert(key.to_owned(), value).is_some() {
            return Err(TopologyError::DuplicateProperty {
                path: path.to_path_buf(),
                key: key.to_owned(),
            });
        }
    }
    for (key, _, _) in allowed {
        if !result.contains_key(*key) {
            return Err(TopologyError::MissingProperty {
                path: path.to_path_buf(),
                key,
            });
        }
    }
    Ok(result)
}

fn read_directory(path: &Path, maximum: usize) -> Result<Vec<(String, PathBuf)>, TopologyError> {
    ensure_directory(path)?;
    let entries = fs::read_dir(path).map_err(|source| io_error("list", path, source))?;
    let mut result = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| io_error("list entry in", path, source))?;
        if result.len() == maximum {
            return Err(TopologyError::TooManyEntries {
                path: path.to_path_buf(),
                maximum,
            });
        }
        let entry_path = entry.path();
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| TopologyError::InvalidEntryName(entry_path.clone()))?;
        result.push((name, entry_path));
    }
    result.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(result)
}

fn validate_root(root: &Path) -> Result<(), TopologyError> {
    let expected = ["generation_id", "nodes", "system_properties"];
    let mut observed = BTreeSet::new();
    for (name, path) in read_directory(root, MAX_ROOT_ENTRIES)? {
        if !expected.contains(&name.as_str()) {
            return Err(TopologyError::UnexpectedEntry {
                path: root.to_path_buf(),
                name,
            });
        }
        if name == "nodes" {
            ensure_directory(&path)?;
        } else {
            let metadata = inspect(&path)?;
            if metadata.file_type().is_symlink() {
                return Err(TopologyError::Symlink(path));
            }
            if !metadata.is_file() {
                return Err(TopologyError::UnexpectedFileType {
                    path,
                    expected: "regular file",
                });
            }
        }
        observed.insert(name);
    }
    for name in expected {
        if !observed.contains(name) {
            return Err(TopologyError::MissingEntry(root.join(name)));
        }
    }
    Ok(())
}

fn parse_node_id(name: &str) -> Result<u32, TopologyError> {
    if name.is_empty()
        || name.starts_with('0') && name != "0"
        || !name.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(TopologyError::InvalidNodeId(name.to_owned()));
    }
    let value = name
        .parse::<u32>()
        .map_err(|_| TopologyError::InvalidNodeId(name.to_owned()))?;
    if value > u16::MAX as u32 {
        return Err(TopologyError::InvalidNodeId(name.to_owned()));
    }
    Ok(value)
}

fn validate_node_entries(path: &Path) -> Result<(), TopologyError> {
    const FILES: [&str; 3] = ["gpu_id", "name", "properties"];
    const DIRECTORIES: [&str; 5] = ["caches", "io_links", "mem_banks", "p2p_links", "perf"];
    let mut files = BTreeSet::new();
    for (name, entry_path) in read_directory(path, MAX_NODE_ENTRIES)? {
        let metadata = inspect(&entry_path)?;
        if metadata.file_type().is_symlink() {
            return Err(TopologyError::Symlink(entry_path));
        }
        if FILES.contains(&name.as_str()) {
            if !metadata.is_file() {
                return Err(TopologyError::UnexpectedFileType {
                    path: entry_path,
                    expected: "regular file",
                });
            }
            files.insert(name);
        } else if DIRECTORIES.contains(&name.as_str()) {
            if !metadata.is_dir() {
                return Err(TopologyError::UnexpectedFileType {
                    path: entry_path,
                    expected: "directory",
                });
            }
        } else {
            return Err(TopologyError::UnexpectedEntry {
                path: path.to_path_buf(),
                name,
            });
        }
    }
    for name in FILES {
        if !files.contains(name) {
            return Err(TopologyError::MissingEntry(path.join(name)));
        }
    }
    Ok(())
}

fn parse_gpu_node(
    node_id: u32,
    gpu_id: u64,
    name: String,
    properties_path: &Path,
    properties: &BTreeMap<String, u64>,
) -> Result<GpuTopologyNode, TopologyError> {
    let encoded_target = required_property(properties, properties_path, "gfx_target_version")?;
    if encoded_target != GFX942_TARGET_VERSION {
        return Err(TopologyError::UnsupportedTarget {
            node_id,
            encoded: encoded_target,
        });
    }
    let vendor_id = required_property(properties, properties_path, "vendor_id")?;
    if vendor_id != EXPECTED_AMD_VENDOR_ID {
        return Err(TopologyError::UnsupportedVendor { node_id, vendor_id });
    }
    let pci_device_id = bounded_u32(properties, properties_path, "device_id", 1, u16::MAX as u64)?;
    let drm_render_minor = bounded_u32(
        properties,
        properties_path,
        "drm_render_minor",
        MIN_DRM_RENDER_MINOR,
        MAX_DRM_RENDER_MINOR,
    )?;
    let unique_id = required_property(properties, properties_path, "unique_id")?;
    if unique_id == 0 {
        return Err(TopologyError::PropertyOutOfRange {
            path: properties_path.to_path_buf(),
            key: "unique_id".to_owned(),
            value: 0,
            minimum: 1,
            maximum: u64::MAX,
        });
    }
    let hive_id = required_property(properties, properties_path, "hive_id")?;
    let location_id = bounded_u32(
        properties,
        properties_path,
        "location_id",
        1,
        u32::MAX as u64,
    )?;
    let domain = bounded_u32(properties, properties_path, "domain", 0, u16::MAX as u64)?;
    let fw_version = bounded_u32(
        properties,
        properties_path,
        "fw_version",
        0,
        u32::MAX as u64,
    )?;
    let sdma_fw_version = bounded_u32(
        properties,
        properties_path,
        "sdma_fw_version",
        0,
        u32::MAX as u64,
    )?;
    let capacity = GpuCapacityObservation {
        simd_count: bounded_u32(properties, properties_path, "simd_count", 1, 65_536)?,
        simd_per_cu: bounded_u32(properties, properties_path, "simd_per_cu", 1, 64)?,
        array_count: bounded_u32(properties, properties_path, "array_count", 1, 4096)?,
        simd_arrays_per_engine: bounded_u32(
            properties,
            properties_path,
            "simd_arrays_per_engine",
            1,
            4096,
        )?,
        lds_size_in_kb: bounded_u32(properties, properties_path, "lds_size_in_kb", 1, 1 << 30)?,
        max_waves_per_simd: bounded_u32(
            properties,
            properties_path,
            "max_waves_per_simd",
            1,
            4096,
        )?,
        compute_queue_count: bounded_u32(properties, properties_path, "num_cp_queues", 1, 1 << 20)?,
        memory_bank_count: bounded_u32(properties, properties_path, "mem_banks_count", 1, 4096)?,
        cache_count: bounded_u32(properties, properties_path, "caches_count", 1, 16_384)?,
        io_link_count: bounded_u32(properties, properties_path, "io_links_count", 1, 4096)?,
        p2p_link_count: bounded_u32(properties, properties_path, "p2p_links_count", 0, 4096)?,
        wavefront_size: bounded_u32(properties, properties_path, "wave_front_size", 64, 64)?,
        xcc_count: bounded_u32(properties, properties_path, "num_xcc", 1, 64)?,
    };
    Ok(GpuTopologyNode {
        node_id,
        gpu_id,
        name,
        target: GfxTarget::Gfx942,
        pci_device_id: pci_device_id as u16,
        drm_render_minor: drm_render_minor as u16,
        unique_id,
        hive_id,
        location_id,
        domain: domain as u16,
        fw_version,
        sdma_fw_version,
        capacity,
    })
}

struct DiscoveryPaths<'a> {
    topology_root: &'a Path,
    boot_id: &'a Path,
    os_release: &'a Path,
    amdgpu_module_root: &'a Path,
    device_character_root: &'a Path,
    sysfs_devices_root: &'a Path,
}

fn canonicalize(path: &Path) -> Result<PathBuf, TopologyError> {
    fs::canonicalize(path).map_err(|source| io_error("canonicalize", path, source))
}

fn resolve_kernel_symlink(path: &Path, admitted_root: &Path) -> Result<PathBuf, TopologyError> {
    let metadata = inspect(path)?;
    if !metadata.file_type().is_symlink() {
        return Err(TopologyError::ExpectedSymlink(path.to_path_buf()));
    }
    let target = canonicalize(path)?;
    if !target.starts_with(admitted_root) {
        return Err(TopologyError::EscapedSysfsRoot {
            path: path.to_path_buf(),
            target,
        });
    }
    Ok(target)
}

fn mismatch(
    node_id: u32,
    field: &'static str,
    kfd: impl ToString,
    render: impl ToString,
) -> TopologyError {
    TopologyError::RenderCorrelationMismatch {
        node_id,
        field,
        kfd: kfd.to_string(),
        render: render.to_string(),
    }
}

fn correlate_render_node(
    gpu: &GpuTopologyNode,
    paths: &DiscoveryPaths<'_>,
    sysfs_devices_root: &Path,
) -> Result<RenderNodeObservation, TopologyError> {
    let link = paths
        .device_character_root
        .join(format!("226:{}", gpu.drm_render_minor));
    let render_path = resolve_kernel_symlink(&link, sysfs_devices_root)?;
    let expected_render_name = format!("renderD{}", gpu.drm_render_minor);
    if render_path.file_name().and_then(|name| name.to_str()) != Some(&expected_render_name)
        || render_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some("drm")
    {
        return Err(mismatch(
            gpu.node_id,
            "render sysfs path",
            expected_render_name,
            render_path.display(),
        ));
    }
    let render_identity = ensure_directory(&render_path)?;
    let dev_path = render_path.join("dev");
    let dev = read_text(&dev_path, MAX_SCALAR_BYTES)?;
    let expected_dev = format!("226:{}\n", gpu.drm_render_minor);
    if dev != expected_dev {
        return Err(mismatch(
            gpu.node_id,
            "DRM device number",
            expected_dev.trim_end(),
            dev.trim_end(),
        ));
    }

    let pci_link = render_path.join("device");
    let pci_path = resolve_kernel_symlink(&pci_link, sysfs_devices_root)?;
    let pci_identity = ensure_directory(&pci_path)?;
    let pci_name = pci_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| TopologyError::InvalidEntryName(pci_path.clone()))?;
    let pci_address = parse_pci_address(pci_name)?;
    if pci_address.domain != gpu.domain {
        return Err(mismatch(
            gpu.node_id,
            "PCI domain",
            gpu.domain,
            pci_address.domain,
        ));
    }
    if pci_address.kfd_location_id() != gpu.location_id {
        return Err(mismatch(
            gpu.node_id,
            "PCI location",
            gpu.location_id,
            pci_address.kfd_location_id(),
        ));
    }

    let unique_id = read_hex_scalar(&pci_path.join("unique_id"), false, None)?;
    if unique_id != gpu.unique_id {
        return Err(mismatch(gpu.node_id, "unique_id", gpu.unique_id, unique_id));
    }
    let vendor_id = read_hex_scalar(&pci_path.join("vendor"), true, Some(4))?;
    if vendor_id != EXPECTED_AMD_VENDOR_ID {
        return Err(mismatch(
            gpu.node_id,
            "PCI vendor",
            EXPECTED_AMD_VENDOR_ID,
            vendor_id,
        ));
    }
    let device_id = read_hex_scalar(&pci_path.join("device"), true, Some(4))?;
    if device_id != u64::from(gpu.pci_device_id) {
        return Err(mismatch(
            gpu.node_id,
            "PCI device",
            gpu.pci_device_id,
            device_id,
        ));
    }
    let pci_revision = read_hex_scalar(&pci_path.join("revision"), true, Some(2))?;
    let partition = PartitionProfile {
        compute: read_compute_partition(&pci_path.join("current_compute_partition"))?,
        memory: read_memory_partition(&pci_path.join("current_memory_partition"))?,
    };
    if ensure_directory(&pci_path)? != pci_identity
        || ensure_directory(&render_path)? != render_identity
    {
        return Err(TopologyError::ChangedDuringRead(render_path));
    }

    Ok(RenderNodeObservation {
        node_id: gpu.node_id,
        drm_render_minor: gpu.drm_render_minor,
        canonical_sysfs_path: render_path,
        pci_address,
        pci_revision: pci_revision as u8,
        unique_id,
        partition,
    })
}

fn discover_host_topology(
    paths: &DiscoveryPaths<'_>,
) -> Result<HostTopologySnapshot, TopologyError> {
    let topology = discover_topology_at(paths.topology_root)?;
    let boot_id = read_boot_id(paths.boot_id)?;
    let kernel_release = read_kernel_release(paths.os_release)?;
    let amdgpu_module = observe_amdgpu_module(paths.amdgpu_module_root)?;
    ensure_directory(paths.device_character_root)?;
    ensure_directory(paths.sysfs_devices_root)?;
    let sysfs_devices_root = canonicalize(paths.sysfs_devices_root)?;
    let mut render_nodes = Vec::with_capacity(topology.gpu_nodes.len());
    for gpu in &topology.gpu_nodes {
        render_nodes.push(correlate_render_node(gpu, paths, &sysfs_devices_root)?);
    }
    let generation_after = read_scalar(&paths.topology_root.join("generation_id"))?;
    if generation_after != topology.provenance.generation {
        return Err(TopologyError::TopologyChanged {
            before: topology.provenance.generation,
            after: generation_after,
        });
    }
    if read_boot_id(paths.boot_id)? != boot_id {
        return Err(TopologyError::ChangedDuringRead(
            paths.boot_id.to_path_buf(),
        ));
    }
    if read_kernel_release(paths.os_release)? != kernel_release {
        return Err(TopologyError::ChangedDuringRead(
            paths.os_release.to_path_buf(),
        ));
    }
    if observe_amdgpu_module(paths.amdgpu_module_root)? != amdgpu_module {
        return Err(TopologyError::ChangedDuringRead(
            paths.amdgpu_module_root.to_path_buf(),
        ));
    }
    Ok(HostTopologySnapshot {
        topology,
        boot_id,
        kernel_release,
        amdgpu_module,
        render_nodes,
    })
}

fn discover_topology_at(root: &Path) -> Result<TopologySnapshot, TopologyError> {
    let root_identity = ensure_directory(root)?;
    validate_root(root)?;
    let generation_path = root.join("generation_id");
    let generation_before = read_scalar(&generation_path)?;
    if generation_before == 0 {
        return Err(TopologyError::InvalidScalar {
            path: generation_path,
            value: "0".to_owned(),
        });
    }
    let platform = parse_platform(&root.join("system_properties"))?;
    let nodes_path = root.join("nodes");
    let nodes_identity = ensure_directory(&nodes_path)?;
    let mut node_directories = Vec::new();
    let mut node_ids = BTreeSet::new();
    for (name, path) in read_directory(&nodes_path, MAX_TOPOLOGY_NODES)? {
        let node_id = parse_node_id(&name)?;
        if !node_ids.insert(node_id) {
            return Err(TopologyError::DuplicateIdentity {
                field: "node_id",
                value: node_id.to_string(),
            });
        }
        let identity = ensure_directory(&path)?;
        node_directories.push((node_id, path, identity));
    }
    node_directories.sort_by_key(|entry| entry.0);

    let mut gpu_nodes = Vec::new();
    let mut gpu_ids = BTreeSet::new();
    let mut unique_ids = BTreeSet::new();
    let mut render_minors = BTreeSet::new();
    let mut locations = BTreeSet::new();
    for (node_id, path, identity) in &node_directories {
        validate_node_entries(path)?;
        let gpu_id = read_scalar(&path.join("gpu_id"))?;
        let name = read_name(&path.join("name"))?;
        let properties_path = path.join("properties");
        let properties = parse_properties(&properties_path)?;
        if gpu_id != 0 {
            let node = parse_gpu_node(*node_id, gpu_id, name, &properties_path, &properties)?;
            if !gpu_ids.insert(node.gpu_id) {
                return Err(TopologyError::DuplicateIdentity {
                    field: "gpu_id",
                    value: node.gpu_id.to_string(),
                });
            }
            if !unique_ids.insert(node.unique_id) {
                return Err(TopologyError::DuplicateIdentity {
                    field: "unique_id",
                    value: node.unique_id.to_string(),
                });
            }
            if !render_minors.insert(node.drm_render_minor) {
                return Err(TopologyError::DuplicateIdentity {
                    field: "drm_render_minor",
                    value: node.drm_render_minor.to_string(),
                });
            }
            if !locations.insert((node.domain, node.location_id)) {
                return Err(TopologyError::DuplicateIdentity {
                    field: "domain/location_id",
                    value: format!("{}:{}", node.domain, node.location_id),
                });
            }
            gpu_nodes.push(node);
        }
        let after = ensure_directory(path)?;
        if after != *identity {
            return Err(TopologyError::ChangedDuringRead(path.clone()));
        }
    }
    if gpu_nodes.is_empty() {
        return Err(TopologyError::EmptyGpuSet);
    }
    if ensure_directory(&nodes_path)? != nodes_identity || ensure_directory(root)? != root_identity
    {
        return Err(TopologyError::ChangedDuringRead(root.to_path_buf()));
    }
    let generation_after = read_scalar(&root.join("generation_id"))?;
    if generation_after != generation_before {
        return Err(TopologyError::TopologyChanged {
            before: generation_before,
            after: generation_after,
        });
    }

    Ok(TopologySnapshot {
        provenance: TopologyProvenance {
            root: root.to_path_buf(),
            file_system_device: root_identity.device,
            inode: root_identity.inode,
            generation: generation_before,
            platform,
        },
        observed_node_count: node_directories.len(),
        gpu_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn valid(gpu_count: u32) -> Self {
            let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "fe2o3-kfd-topology-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            fs::write(root.join("generation_id"), "7\n").unwrap();
            fs::write(
                root.join("system_properties"),
                "platform_oem 11\nplatform_id 22\nplatform_rev 3\n",
            )
            .unwrap();
            fs::create_dir(root.join("nodes")).unwrap();
            Self::write_node(&root, 0, 0, 0);
            for index in 1..=gpu_count {
                Self::write_node(&root, index, 1000 + u64::from(index), index);
            }
            Self { root }
        }

        fn write_node(root: &Path, node_id: u32, gpu_id: u64, identity: u32) {
            let path = root.join("nodes").join(node_id.to_string());
            fs::create_dir(&path).unwrap();
            fs::write(path.join("gpu_id"), format!("{gpu_id}\n")).unwrap();
            fs::write(
                path.join("name"),
                if gpu_id == 0 { "\n" } else { "ip discovery\n" },
            )
            .unwrap();
            let properties = if gpu_id == 0 {
                "cpu_cores_count 48\ngfx_target_version 0\n".to_owned()
            } else {
                format!(
                    "simd_count 1216\nmem_banks_count 1\ncaches_count 626\n\
                     io_links_count 8\np2p_links_count 1\nwave_front_size 64\n\
                     gfx_target_version 90402\nvendor_id 4098\ndevice_id 29857\n\
                     location_id {}\ndomain 0\ndrm_render_minor {}\nhive_id 99\n\
                     unique_id {}\nfw_version 192\nsdma_fw_version 25\nnum_xcc 8\n\
                     simd_per_cu 4\narray_count 32\nsimd_arrays_per_engine 1\n\
                     lds_size_in_kb 64\nmax_waves_per_simd 8\nnum_cp_queues 24\n",
                    4096 * identity,
                    127 + identity,
                    2000 + u64::from(identity),
                )
            };
            fs::write(path.join("properties"), properties).unwrap();
        }

        fn node(&self, id: u32) -> PathBuf {
            self.root.join("nodes").join(id.to_string())
        }

        fn replace_property(&self, id: u32, old: &str, new: &str) {
            let path = self.node(id).join("properties");
            let contents = fs::read_to_string(&path).unwrap();
            assert!(contents.contains(old));
            fs::write(path, contents.replace(old, new)).unwrap();
        }

        fn discover(&self) -> Result<TopologySnapshot, TopologyError> {
            discover_topology_at(&self.root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    struct RenderFixture {
        base: PathBuf,
        boot_id: PathBuf,
        os_release: PathBuf,
        module_root: PathBuf,
        device_character_root: PathBuf,
        devices_root: PathBuf,
        pci_path: PathBuf,
        gpu: GpuTopologyNode,
    }

    impl RenderFixture {
        fn valid() -> Self {
            use std::os::unix::fs::symlink;
            let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let base = std::env::temp_dir().join(format!(
                "fe2o3-kfd-render-{}-{sequence}",
                std::process::id()
            ));
            let boot_id = base.join("boot_id");
            let os_release = base.join("osrelease");
            let module_root = base.join("module/amdgpu");
            let device_character_root = base.join("dev-char");
            let devices_root = base.join("devices");
            let pci_path = devices_root.join("pci0000:00/0000:05:00.0");
            let render_path = pci_path.join("drm/renderD128");
            fs::create_dir_all(&module_root).unwrap();
            fs::create_dir(&device_character_root).unwrap();
            fs::create_dir_all(&render_path).unwrap();
            fs::write(&boot_id, "317d0f9a-4f05-4ab0-8922-3ebfd7354c8b\n").unwrap();
            fs::write(&os_release, "6.8.0-124-generic\n").unwrap();
            fs::write(module_root.join("version"), "6.16.13\n").unwrap();
            fs::write(module_root.join("srcversion"), "A6F143BEC60C0AFC3263226\n").unwrap();
            fs::write(render_path.join("dev"), "226:128\n").unwrap();
            symlink(&pci_path, render_path.join("device")).unwrap();
            fs::write(pci_path.join("unique_id"), "1234\n").unwrap();
            fs::write(pci_path.join("vendor"), "0x1002\n").unwrap();
            fs::write(pci_path.join("device"), "0x74a1\n").unwrap();
            fs::write(pci_path.join("revision"), "0x00\n").unwrap();
            fs::write(pci_path.join("current_compute_partition"), "SPX\n").unwrap();
            fs::write(pci_path.join("current_memory_partition"), "NPS1\n").unwrap();
            symlink(&render_path, device_character_root.join("226:128")).unwrap();
            Self {
                base,
                boot_id,
                os_release,
                module_root,
                device_character_root,
                devices_root,
                pci_path,
                gpu: GpuTopologyNode {
                    node_id: 2,
                    gpu_id: 1001,
                    name: "ip discovery".to_owned(),
                    target: GfxTarget::Gfx942,
                    pci_device_id: 0x74a1,
                    drm_render_minor: 128,
                    unique_id: 0x1234,
                    hive_id: 99,
                    location_id: 0x0500,
                    domain: 0,
                    fw_version: 192,
                    sdma_fw_version: 25,
                    capacity: GpuCapacityObservation {
                        simd_count: 1216,
                        simd_per_cu: 4,
                        array_count: 32,
                        simd_arrays_per_engine: 1,
                        lds_size_in_kb: 64,
                        max_waves_per_simd: 8,
                        compute_queue_count: 24,
                        memory_bank_count: 1,
                        cache_count: 626,
                        io_link_count: 8,
                        p2p_link_count: 1,
                        wavefront_size: 64,
                        xcc_count: 8,
                    },
                },
            }
        }

        fn paths(&self) -> DiscoveryPaths<'_> {
            DiscoveryPaths {
                topology_root: &self.base,
                boot_id: &self.boot_id,
                os_release: &self.os_release,
                amdgpu_module_root: &self.module_root,
                device_character_root: &self.device_character_root,
                sysfs_devices_root: &self.devices_root,
            }
        }

        fn correlate(&self) -> Result<RenderNodeObservation, TopologyError> {
            correlate_render_node(
                &self.gpu,
                &self.paths(),
                &canonicalize(&self.devices_root).unwrap(),
            )
        }
    }

    impl Drop for RenderFixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.base).unwrap();
        }
    }

    #[test]
    fn valid_fixture_produces_sorted_bounded_observations() {
        let fixture = Fixture::valid(2);
        let snapshot = fixture.discover().unwrap();
        assert_eq!(snapshot.observed_node_count(), 3);
        assert_eq!(snapshot.provenance().generation(), 7);
        assert_eq!(snapshot.provenance().platform().oem(), 11);
        assert_eq!(snapshot.gpu_nodes().len(), 2);
        assert_eq!(snapshot.gpu_nodes()[0].node_id(), 1);
        assert_eq!(snapshot.gpu_nodes()[0].target(), GfxTarget::Gfx942);
        assert_eq!(snapshot.gpu_nodes()[0].drm_render_minor(), 128);
        assert_eq!(snapshot.gpu_nodes()[0].fw_version(), 192);
        assert_eq!(snapshot.gpu_nodes()[0].sdma_fw_version(), 25);
        assert_eq!(snapshot.gpu_nodes()[0].capacity().xcc_count(), 8);
        assert_eq!(snapshot.gpu_nodes()[1].node_id(), 2);
    }

    #[test]
    fn boot_id_parser_accepts_only_canonical_lowercase_uuid() {
        let raw = "317d0f9a-4f05-4ab0-8922-3ebfd7354c8b";
        let boot_id: BootId = raw.parse().unwrap();
        assert_eq!(boot_id.to_string(), raw);
        for invalid in [
            "317D0F9A-4f05-4ab0-8922-3ebfd7354c8b",
            "317d0f9a4f054ab089223ebfd7354c8b",
            "317d0f9a-4f05-4ab0-8922-3ebfd7354c8g",
        ] {
            assert!(invalid.parse::<BootId>().is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn kernel_release_parser_is_bounded_and_strict() {
        let release: KernelRelease = "6.8.0-124-generic".parse().unwrap();
        assert_eq!(release.as_str(), "6.8.0-124-generic");
        for invalid in ["", "6.8.0 release", "6.8.0/release"] {
            assert!(
                invalid.parse::<KernelRelease>().is_err(),
                "accepted {invalid}"
            );
        }
        assert!(
            "x".repeat(MAX_MODULE_FIELD_BYTES + 1)
                .parse::<KernelRelease>()
                .is_err()
        );
    }

    #[test]
    fn strict_module_observation_reports_optional_fields() {
        let fixture = RenderFixture::valid();
        fs::create_dir(fixture.module_root.join("parameters")).unwrap();
        fs::write(fixture.module_root.join("parameters/mes"), "0\n").unwrap();
        fs::write(fixture.module_root.join("parameters/sched_policy"), "0\n").unwrap();
        fs::write(fixture.module_root.join("parameters/cwsr_enable"), "1\n").unwrap();
        let module = observe_amdgpu_module(&fixture.module_root).unwrap();
        assert_eq!(module.version(), Some("6.16.13"));
        assert_eq!(module.srcversion(), Some("A6F143BEC60C0AFC3263226"));
        assert_eq!(module.mes(), Some(0));
        assert_eq!(module.sched_policy(), Some(0));
        assert_eq!(module.cwsr_enable(), Some(1));
        fs::remove_file(fixture.module_root.join("version")).unwrap();
        let module = observe_amdgpu_module(&fixture.module_root).unwrap();
        assert_eq!(module.version(), None);
    }

    #[test]
    fn invalid_module_srcversion_is_rejected() {
        let fixture = RenderFixture::valid();
        fs::write(fixture.module_root.join("srcversion"), "not-hex\n").unwrap();
        assert!(matches!(
            observe_amdgpu_module(&fixture.module_root),
            Err(TopologyError::InvalidModuleField { .. })
        ));
    }

    #[test]
    fn render_node_is_correlated_to_unique_id_bdf_and_partitions() {
        let fixture = RenderFixture::valid();
        let render = fixture.correlate().unwrap();
        assert_eq!(render.node_id(), 2);
        assert_eq!(render.pci_address().to_string(), "0000:05:00.0");
        assert_eq!(render.pci_revision(), 0);
        assert_eq!(render.unique_id(), 0x1234);
        assert_eq!(render.partition().compute(), ComputePartition::Spx);
        assert_eq!(render.partition().memory(), MemoryPartition::Nps1);
    }

    #[test]
    fn render_unique_id_mismatch_is_rejected() {
        let fixture = RenderFixture::valid();
        fs::write(fixture.pci_path.join("unique_id"), "1235\n").unwrap();
        assert!(matches!(
            fixture.correlate(),
            Err(TopologyError::RenderCorrelationMismatch {
                field: "unique_id",
                ..
            })
        ));
    }

    #[test]
    fn malformed_pci_revision_is_rejected() {
        let fixture = RenderFixture::valid();
        fs::write(fixture.pci_path.join("revision"), "0X00\n").unwrap();
        assert!(matches!(
            fixture.correlate(),
            Err(TopologyError::InvalidHexScalar { .. })
        ));
    }

    #[test]
    fn render_bdf_location_mismatch_is_rejected() {
        let mut fixture = RenderFixture::valid();
        fixture.gpu.location_id = 0x0600;
        assert!(matches!(
            fixture.correlate(),
            Err(TopologyError::RenderCorrelationMismatch {
                field: "PCI location",
                ..
            })
        ));
    }

    #[test]
    fn render_link_must_be_a_kernel_style_symlink() {
        let fixture = RenderFixture::valid();
        let link = fixture.device_character_root.join("226:128");
        fs::remove_file(&link).unwrap();
        fs::create_dir(link).unwrap();
        assert!(matches!(
            fixture.correlate(),
            Err(TopologyError::ExpectedSymlink(_))
        ));
    }

    #[test]
    fn render_symlink_cannot_escape_devices_root() {
        use std::os::unix::fs::symlink;
        let fixture = RenderFixture::valid();
        let link = fixture.device_character_root.join("226:128");
        fs::remove_file(&link).unwrap();
        symlink("/tmp", link).unwrap();
        assert!(matches!(
            fixture.correlate(),
            Err(TopologyError::EscapedSysfsRoot { .. })
        ));
    }

    #[test]
    fn unknown_partition_mode_is_rejected() {
        let fixture = RenderFixture::valid();
        fs::write(
            fixture.pci_path.join("current_compute_partition"),
            "FUTURE\n",
        )
        .unwrap();
        assert!(matches!(
            fixture.correlate(),
            Err(TopologyError::UnsupportedPartition { .. })
        ));
    }

    #[test]
    fn terminal_root_symlink_is_rejected() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::valid(1);
        let alias = fixture.root.with_extension("alias");
        symlink(&fixture.root, &alias).unwrap();
        let error = discover_topology_at(&alias).unwrap_err();
        fs::remove_file(alias).unwrap();
        assert!(matches!(error, TopologyError::Symlink(_)));
    }

    #[test]
    fn node_symlink_is_rejected() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::valid(1);
        let node = fixture.node(1);
        fs::remove_dir_all(&node).unwrap();
        symlink(fixture.node(0), node).unwrap();
        assert!(matches!(fixture.discover(), Err(TopologyError::Symlink(_))));
    }

    #[test]
    fn property_symlink_is_rejected() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::valid(1);
        let properties = fixture.node(1).join("properties");
        fs::remove_file(&properties).unwrap();
        symlink("/etc/passwd", properties).unwrap();
        assert!(matches!(fixture.discover(), Err(TopologyError::Symlink(_))));
    }

    #[test]
    fn non_regular_property_file_is_rejected() {
        let fixture = Fixture::valid(1);
        let properties = fixture.node(1).join("properties");
        fs::remove_file(&properties).unwrap();
        fs::create_dir(properties).unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::UnexpectedFileType { .. })
        ));
    }

    #[test]
    fn noncanonical_node_id_is_rejected() {
        let fixture = Fixture::valid(1);
        fs::create_dir(fixture.root.join("nodes/01")).unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::InvalidNodeId(value)) if value == "01"
        ));
    }

    #[test]
    fn bounded_node_count_is_enforced_before_node_reads() {
        let fixture = Fixture::valid(1);
        for id in 2..=256 {
            fs::create_dir(fixture.root.join("nodes").join(id.to_string())).unwrap();
        }
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::TooManyEntries {
                maximum: MAX_TOPOLOGY_NODES,
                ..
            })
        ));
    }

    #[test]
    fn oversized_properties_are_rejected() {
        let fixture = Fixture::valid(1);
        fs::write(
            fixture.node(1).join("properties"),
            vec![b'a'; MAX_PROPERTY_BYTES + 1],
        )
        .unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::FileTooLarge {
                maximum: MAX_PROPERTY_BYTES,
                ..
            })
        ));
    }

    #[test]
    fn unknown_property_is_rejected() {
        let fixture = Fixture::valid(1);
        let path = fixture.node(1).join("properties");
        let mut contents = fs::read_to_string(&path).unwrap();
        contents.push_str("future_security_mode 1\n");
        fs::write(path, contents).unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::UnknownProperty { key, .. }) if key == "future_security_mode"
        ));
    }

    #[test]
    fn duplicate_property_is_rejected() {
        let fixture = Fixture::valid(1);
        let path = fixture.node(1).join("properties");
        let mut contents = fs::read_to_string(&path).unwrap();
        contents.push_str("unique_id 3000\n");
        fs::write(path, contents).unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::DuplicateProperty { key, .. }) if key == "unique_id"
        ));
    }

    #[test]
    fn malformed_numbers_are_rejected() {
        let fixture = Fixture::valid(1);
        fixture.replace_property(1, "gfx_target_version 90402", "gfx_target_version -1");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::MalformedPropertyLine { .. })
        ));
    }

    #[test]
    fn unsupported_target_is_rejected() {
        let fixture = Fixture::valid(1);
        fixture.replace_property(1, "gfx_target_version 90402", "gfx_target_version 110000");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::UnsupportedTarget {
                encoded: 110000,
                ..
            })
        ));
    }

    #[test]
    fn bounded_count_field_is_enforced() {
        let fixture = Fixture::valid(1);
        fixture.replace_property(1, "caches_count 626", "caches_count 16385");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::PropertyOutOfRange { ref key, .. }) if key == "caches_count"
        ));
    }

    #[test]
    fn firmware_observations_are_required_and_bounded() {
        let fixture = Fixture::valid(1);
        fixture.replace_property(1, "fw_version 192", "fw_version 4294967296");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::PropertyOutOfRange { ref key, .. }) if key == "fw_version"
        ));

        let fixture = Fixture::valid(1);
        fixture.replace_property(1, "sdma_fw_version 25\n", "");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::MissingProperty {
                key: "sdma_fw_version",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_gpu_id_is_rejected() {
        let fixture = Fixture::valid(2);
        fs::write(fixture.node(2).join("gpu_id"), "1001\n").unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::DuplicateIdentity {
                field: "gpu_id",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_unique_id_is_rejected() {
        let fixture = Fixture::valid(2);
        fixture.replace_property(2, "unique_id 2002", "unique_id 2001");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::DuplicateIdentity {
                field: "unique_id",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_render_minor_is_rejected() {
        let fixture = Fixture::valid(2);
        fixture.replace_property(2, "drm_render_minor 129", "drm_render_minor 128");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::DuplicateIdentity {
                field: "drm_render_minor",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_domain_location_is_rejected() {
        let fixture = Fixture::valid(2);
        fixture.replace_property(2, "location_id 8192", "location_id 4096");
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::DuplicateIdentity {
                field: "domain/location_id",
                ..
            })
        ));
    }

    #[test]
    fn unknown_top_level_entry_is_rejected() {
        let fixture = Fixture::valid(1);
        fs::write(fixture.root.join("authority"), "1\n").unwrap();
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::UnexpectedEntry { name, .. }) if name == "authority"
        ));
    }

    #[test]
    fn empty_gpu_set_is_rejected() {
        let fixture = Fixture::valid(0);
        assert!(matches!(
            fixture.discover(),
            Err(TopologyError::EmptyGpuSet)
        ));
    }
}

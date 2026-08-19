//! Checked composition of KFD, topology, and DRM identity observations.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

use fe2o3_drm_uapi::{
    AMDGPU_DRM_DRIVER_VERSION, AMDGPU_FAMILY_AI, DRM_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
    DrmAmdgpuDeviceIdentityV1, DrmDriverVersion,
};
use fe2o3_kfd_uapi::{
    KFD_IOCTL_MAJOR_VERSION, KFD_IOCTL_MINOR_VERSION, KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
    KfdProcessDeviceApertures,
};
use fe2o3_runtime_model::{
    AMD_PCI_VENDOR_ID_V1, AmdgpuModuleObservationV1, CharacterDeviceProjectionV1,
    ComputePartitionObservationV1, DEVICE_PROJECTION_SCHEMA_VERSION_V1, DeviceAdmissionErrorV1,
    DeviceAdmissionProfileIdV1, DeviceAdmissionProfileV1, DeviceGenerationV1,
    DeviceIdentityStateV1, DeviceNodeV1, DeviceObservationDomainIdV1,
    DeviceProjectionCommitFenceV1, DeviceProjectionErrorV1, DeviceProjectionHistoryErrorV1,
    DeviceProjectionHistoryLinkV1, DeviceProjectionHistoryV1, DeviceProjectionRecordV1,
    DeviceProjectionSourceV1, DrmDriverNameObservationV1, DrmFamilyObservationV1, IdentityDigestV1,
    InclusiveRangeProjectionV1, InventoryDeviceProjectionV1, InventoryInputErrorV1,
    KernelReleaseObservationV1, KfdProjectionV1, ModelAdmissionStatusV1, ModelDeviceAdmissionV1,
    ModelVmAdmissionV1, PciAddressV1, ProcessApertureProjectionV1, RenderProjectionV1,
    TopologyProjectionV1, UntrustedVmObservationV1, ValidatedDeviceProjectionV1, VmIdV1,
    XnackObservationV1,
};
use rustix::fd::OwnedFd;
use sha2::{Digest, Sha256};

use crate::topology::{
    self, HostTopologySnapshot, PciAddress, TopologyError, V1_PARTITION_PROFILE,
};
use crate::{KfdAdapterError, KfdWithAdmittedUapi};

pub const ADMITTED_KERNEL_RELEASE_V1: &str = "6.8.0-124-generic";
pub const ADMITTED_AMDGPU_MODULE_VERSION_V1: &str = "6.16.13";
pub const ADMITTED_AMDGPU_MODULE_SRCVERSION_V1: &str = "A6F143BEC60C0AFC3263226";
pub const ADMITTED_KFD_FIRMWARE_VERSION_V1: u32 = 192;
pub const ADMITTED_KFD_SDMA_FIRMWARE_VERSION_V1: u32 = 25;
pub const MAX_PROCESS_APERTURES_V1: usize = 16;
pub(super) const PROCESS_APERTURE_QUERY_CAPACITY_V1: usize = MAX_PROCESS_APERTURES_V1 + 1;

/// Canonical manifest for the exact R1 device-admission profile.
///
/// This binds checked userspace criteria and named kernel contracts. Its digest
/// identifies this profile; it does not authenticate the running kernel,
/// firmware, sysfs, or hardware.
pub const DEVICE_ADMISSION_PROFILE_MANIFEST_V1: &str = concat!(
    "profile_id=fe2o3-linux-x86_64-mi300x-gfx942-xnack-minus-spx-nps1-r1\n",
    "kfd_schema_sha256=e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4\n",
    "drm_schema_sha256=800569fe9b467b389bcfc6e5d65b23d66a0386a90fc2a669fac8c83800e76d8b\n",
    "kernel_release=6.8.0-124-generic\n",
    "amdgpu_module=6.16.13\n",
    "amdgpu_srcversion=A6F143BEC60C0AFC3263226\n",
    "target=gfx942:90402,wavefront:64,simd:1216,xcc:8\n",
    "pci=vendor:1002,device:74a1,revision:00\n",
    "drm_device=chip_rev:1,external_rev:71,family:141,acceleration:1\n",
    "partition=SPX/NPS1\n",
    "firmware_observation=compute:192,sdma:25\n",
    "xnack=disabled-query,set-disabled-no-queue-barrier,disabled-query\n",
    "apertures=max:16,count-fill-count,complete-topology-inventory,page-aligned,inclusive,record-disjoint\n",
    "commit_fence=process,retained-fds,full-topology,xnack,apertures,smi-reset-events,vram-loss-counter\n",
    "currentness=contracted-composite,no-all-reset-generation,no-aba-proof\n",
    "authority=model-only,no-vm,no-memory,no-queue,no-dispatch\n",
);

/// SHA-256 of [`DEVICE_ADMISSION_PROFILE_MANIFEST_V1`].
pub const DEVICE_ADMISSION_PROFILE_SHA256_V1: &str =
    "e12ea33b259666e7928612403109640b03b0d637b893a2c15b87d17a4211c8de";

/// Typed digest bytes of [`DEVICE_ADMISSION_PROFILE_MANIFEST_V1`].
pub const DEVICE_ADMISSION_PROFILE_SHA256_BYTES_V1: [u8; 32] = [
    0xe1, 0x2e, 0xa3, 0x3b, 0x25, 0x96, 0x66, 0xe7, 0x92, 0x86, 0x12, 0x40, 0x31, 0x09, 0x64, 0x0b,
    0x03, 0xb0, 0xd6, 0x37, 0xb8, 0x93, 0xa2, 0xc1, 0x5b, 0x87, 0xd1, 0x7a, 0x42, 0x11, 0xc8, 0xde,
];

const DRM_DEVICE_MAJOR: u32 = 226;
const PAGE_BYTES: u64 = 4096;
const MI300X_DEVICE_ID: u16 = 0x74a1;
const MI300X_PCI_REVISION: u8 = 0;
const MI300X_SPX_SIMD_COUNT: u32 = 1216;
const MI300X_SPX_XCC_COUNT: u32 = 8;

static DEVICE_ADMISSION_LEASE: Mutex<()> = Mutex::new(());
static DEVICE_MODEL_HISTORY: Mutex<DeviceModelHistory> = Mutex::new(DeviceModelHistory::Empty);
static NEXT_ADMISSION_GENERATION: AtomicU64 = AtomicU64::new(1);

enum DeviceModelHistory {
    Empty,
    State {
        identities: DeviceIdentityStateV1,
        projections: DeviceProjectionHistoryV1,
    },
    Poisoned,
}

/// Stable selector for one GPU in the complete topology snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceSelector {
    UniqueId(u64),
}

/// One inclusive process virtual-address aperture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InclusiveAperture {
    base: u64,
    limit: u64,
}

impl InclusiveAperture {
    pub const fn base(self) -> u64 {
        self.base
    }

    pub const fn limit(self) -> u64 {
        self.limit
    }

    pub const fn size(self) -> u64 {
        self.limit - self.base + 1
    }

    #[cfg(test)]
    pub(super) const fn from_checked_parts_for_memory_tests(base: u64, limit: u64) -> Self {
        Self { base, limit }
    }
}

/// Checked KFD aperture observation for one topology GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessApertureObservation {
    gpu_id: u32,
    lds: InclusiveAperture,
    scratch: InclusiveAperture,
    gpuvm: InclusiveAperture,
}

impl ProcessApertureObservation {
    pub const fn gpu_id(self) -> u32 {
        self.gpu_id
    }

    pub const fn lds(self) -> InclusiveAperture {
        self.lds
    }

    pub const fn scratch(self) -> InclusiveAperture {
        self.scratch
    }

    pub const fn gpuvm(self) -> InclusiveAperture {
        self.gpuvm
    }
}

/// Retained render descriptor identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderDescriptorObservation {
    pub(super) file_system_device: u64,
    pub(super) inode: u64,
    pub(super) character_device: u64,
    pub(super) major: u32,
    pub(super) minor: u32,
}

impl RenderDescriptorObservation {
    pub const fn file_system_device(self) -> u64 {
        self.file_system_device
    }

    pub const fn inode(self) -> u64 {
        self.inode
    }

    pub const fn character_device(self) -> u64 {
        self.character_device
    }

    pub const fn major(self) -> u32 {
        self.major
    }

    pub const fn minor(self) -> u32 {
        self.minor
    }
}

/// Contracted results returned by the reviewed DRM identity/currentness requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrmIdentityObservation {
    pub(super) driver_version: DrmDriverVersion,
    pub(super) acceleration_working: u32,
    pub(super) device: DrmAmdgpuDeviceIdentityV1,
    pub(super) vram_lost_counter: u32,
}

impl DrmIdentityObservation {
    pub const fn driver_version(self) -> DrmDriverVersion {
        self.driver_version
    }

    pub const fn acceleration_working(self) -> u32 {
        self.acceleration_working
    }

    pub const fn device(self) -> DrmAmdgpuDeviceIdentityV1 {
        self.device
    }

    /// Returns the driver's wrapping VRAM-loss observation.
    ///
    /// This is not an all-reset generation. The admitted driver increments it
    /// only on reset paths that report VRAM loss.
    pub const fn vram_lost_counter(self) -> u32 {
        self.vram_lost_counter
    }
}

/// Deterministic checks committed by the no-queue R1 device token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceBindingObservation {
    topology_node_id: u32,
    kfd_gpu_id: u32,
    unique_id: u64,
    pci: PciAddress,
    render_minor: u16,
    render_descriptor: RenderDescriptorObservation,
    drm: DrmIdentityObservation,
    aperture: ProcessApertureObservation,
}

impl DeviceBindingObservation {
    pub const fn topology_node_id(&self) -> u32 {
        self.topology_node_id
    }

    pub const fn kfd_gpu_id(&self) -> u32 {
        self.kfd_gpu_id
    }

    pub const fn unique_id(&self) -> u64 {
        self.unique_id
    }

    pub const fn pci(&self) -> PciAddress {
        self.pci
    }

    pub const fn render_minor(&self) -> u16 {
        self.render_minor
    }

    pub const fn render_descriptor(&self) -> RenderDescriptorObservation {
        self.render_descriptor
    }

    pub const fn drm(&self) -> DrmIdentityObservation {
        self.drm
    }

    pub const fn aperture(&self) -> ProcessApertureObservation {
        self.aperture
    }
}

/// Checked, retained no-queue capability for the exact R1 MI300X profile.
///
/// This value is not `Clone`, `Send`, or `Sync`. It owns both descriptors and
/// the process-global fe2o3 admission lease. It establishes checked
/// correlation under named kernel/sysfs/ioctl contracts, not a proof of the
/// kernel, firmware, hardware, or concrete-to-model refinement.
pub struct CheckedGfx942XnackMinusDevice {
    pub(super) kfd: KfdWithAdmittedUapi,
    pub(super) render_fd: OwnedFd,
    render_path: PathBuf,
    pub(super) topology: HostTopologySnapshot,
    pub(super) apertures: Vec<ProcessApertureObservation>,
    pub(super) observation: DeviceBindingObservation,
    model_admission: ModelDeviceAdmissionV1,
    projection: ValidatedDeviceProjectionV1,
    projection_history: DeviceProjectionHistoryLinkV1,
    pub(super) process: ProcessIncarnationObservation,
    pub(super) reset_fence: crate::currentness::ResetEventFence,
    pub(super) currentness_poisoned: bool,
    pub(super) retire_model_on_drop: bool,
    _lease: MutexGuard<'static, ()>,
}

impl fmt::Debug for CheckedGfx942XnackMinusDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckedGfx942XnackMinusDevice")
            .field("kfd", &self.kfd)
            .field("render_path", &self.render_path)
            .field("observation", &self.observation)
            .field("process", &self.process)
            .field("reset_fence", &self.reset_fence)
            .field("currentness_poisoned", &self.currentness_poisoned)
            .field("retire_model_on_drop", &self.retire_model_on_drop)
            .finish_non_exhaustive()
    }
}

impl CheckedGfx942XnackMinusDevice {
    pub fn topology_snapshot(&self) -> &HostTopologySnapshot {
        &self.topology
    }

    pub fn process_apertures(&self) -> &[ProcessApertureObservation] {
        &self.apertures
    }

    pub const fn observation(&self) -> &DeviceBindingObservation {
        &self.observation
    }

    pub const fn model_admission(&self) -> ModelDeviceAdmissionV1 {
        self.model_admission
    }

    pub const fn projection(&self) -> &ValidatedDeviceProjectionV1 {
        &self.projection
    }

    pub const fn projection_history(&self) -> &DeviceProjectionHistoryLinkV1 {
        &self.projection_history
    }

    pub fn render_opening_path(&self) -> &Path {
        &self.render_path
    }

    pub const fn process_incarnation(&self) -> ProcessIncarnationObservation {
        self.process
    }

    /// Rechecks that this token remains in its opening process.
    pub fn check_process(&self) -> Result<(), DeviceBindingError> {
        self.kfd
            .opened
            .ensure_process(std::process::id())
            .map_err(DeviceBindingError::Kfd)?;
        let current = crate::linux::observe_process_incarnation()?;
        if current != self.process {
            return Err(DeviceBindingError::ProcessIncarnationChanged);
        }
        Ok(())
    }

    /// No raw descriptor access is exposed. This method only reports the
    /// number of retained descriptor fields.
    pub fn descriptor_count(&self) -> usize {
        let _ = &self.render_fd;
        let _ = &self.kfd.opened.fd;
        let _ = &self.reset_fence;
        3
    }

    pub(super) fn register_memory_vm_model_only(
        &mut self,
        vm_id: VmIdV1,
    ) -> Result<ModelVmAdmissionV1, DeviceBindingError> {
        // ACQUIRE_VM has already made process-lifetime state irreversible.
        self.retire_model_on_drop = false;
        let correlation = self.model_admission.correlation();
        let observation = UntrustedVmObservationV1 {
            domain_id: self.model_admission.domain_id(),
            device: self.model_admission.model_key(),
            vm_id,
            kfd_gpu_id: correlation.kfd_gpu_id(),
            render_node: correlation.render_node(),
            pci: correlation.identity().pci,
        };
        let mut history = DEVICE_MODEL_HISTORY
            .lock()
            .map_err(|_| DeviceBindingError::ModelHistoryPoisoned)?;
        let DeviceModelHistory::State {
            identities,
            projections,
        } = &*history
        else {
            *history = DeviceModelHistory::Poisoned;
            return Err(DeviceBindingError::ModelHistoryPoisoned);
        };
        let (next, vm) = identities
            .register_vm_model_only(self.model_admission, observation)
            .map_err(DeviceBindingError::ModelAdmission)?;
        *history = DeviceModelHistory::State {
            identities: next,
            projections: projections.clone(),
        };
        Ok(vm)
    }
}

impl Drop for CheckedGfx942XnackMinusDevice {
    fn drop(&mut self) {
        if !self.retire_model_on_drop {
            return;
        }
        let Ok(mut history) = DEVICE_MODEL_HISTORY.lock() else {
            return;
        };
        let DeviceModelHistory::State {
            identities,
            projections,
        } = &*history
        else {
            *history = DeviceModelHistory::Poisoned;
            return;
        };
        match identities.retire_device_model_only(self.model_admission) {
            Ok(next) => {
                *history = DeviceModelHistory::State {
                    identities: next,
                    projections: projections.clone(),
                };
            }
            Err(_) => *history = DeviceModelHistory::Poisoned,
        }
    }
}

/// Process and mount-namespace facts captured around one admission transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessIncarnationObservation {
    pub(super) pid: u32,
    pub(super) start_time_ticks: u64,
    pub(super) mount_namespace_device: u64,
    pub(super) mount_namespace_inode: u64,
}

impl ProcessIncarnationObservation {
    pub const fn pid(self) -> u32 {
        self.pid
    }

    pub const fn start_time_ticks(self) -> u64 {
        self.start_time_ticks
    }

    pub const fn mount_namespace_device(self) -> u64 {
        self.mount_namespace_device
    }

    pub const fn mount_namespace_inode(self) -> u64 {
        self.mount_namespace_inode
    }
}

pub(super) struct OpenedRender {
    pub(super) fd: OwnedFd,
    pub(super) path: PathBuf,
    pub(super) descriptor: RenderDescriptorObservation,
    pub(super) drm: DrmIdentityObservation,
}

impl KfdWithAdmittedUapi {
    /// Correlates one explicitly selected MI300X without creating a VM, memory
    /// object, mapping, queue, event, code object, or dispatch.
    pub fn bind_gfx942_xnack_minus(
        self,
        selector: DeviceSelector,
    ) -> Result<CheckedGfx942XnackMinusDevice, DeviceBindingError> {
        let lease = match DEVICE_ADMISSION_LEASE.try_lock() {
            Ok(lease) => lease,
            Err(TryLockError::WouldBlock) => return Err(DeviceBindingError::AdmissionInProgress),
            Err(TryLockError::Poisoned(_)) => {
                return Err(DeviceBindingError::AdmissionLeasePoisoned);
            }
        };
        self.opened
            .ensure_process(std::process::id())
            .map_err(DeviceBindingError::Kfd)?;

        let process_before = crate::linux::observe_process_incarnation()?;
        let kfd_node = crate::linux::validate_kfd_descriptor_and_sysfs(
            &self.opened.fd,
            self.opened.node_observation(),
        )?;
        let topology_before = topology::discover_default_topology()?;
        validate_platform_provenance(&topology_before)?;
        let (gpu, render_sysfs) = select_unique(&topology_before, selector)?;
        let kfd_gpu_id = u32::try_from(gpu.gpu_id())
            .map_err(|_| DeviceBindingError::GpuIdOutOfRange(gpu.gpu_id()))?;

        let reobserved_uapi = crate::linux::observe_uapi(&self.opened.fd)?;
        if reobserved_uapi != self.uapi.reported_version() {
            return Err(DeviceBindingError::UapiChanged);
        }

        let render = crate::linux::open_and_observe_render(gpu.drm_render_minor())?;
        validate_render_profile(gpu, render_sysfs, &render)?;

        // Subscribe before authority/token construction and the XNACK,
        // aperture, and commit observations. The pre-subscription DRM result
        // is immediately repeated after enabling the prospective stream.
        let mut reset_fence =
            crate::currentness::ResetEventFence::subscribe(&self.opened.fd, kfd_gpu_id)?;
        let drm_after_subscription = crate::linux::observe_drm_identity(&render.fd)?;
        if drm_after_subscription != render.drm {
            return Err(DeviceBindingError::ObservableCurrentnessChanged(
                "DRM identity or VRAM-loss counter across reset subscription",
            ));
        }
        if crate::linux::observe_uapi(&self.opened.fd)? != self.uapi.reported_version() {
            return Err(DeviceBindingError::UapiChanged);
        }

        crate::linux::establish_xnack_disabled_no_queue_barrier(&self.opened.fd)?;
        let apertures_before = validate_apertures(
            crate::linux::observe_process_apertures(&self.opened.fd)?,
            &topology_before,
        )?;

        if crate::linux::query_xnack_mode(&self.opened.fd)? != 0 {
            return Err(DeviceBindingError::UnsupportedXnackMode);
        }
        let apertures_after = validate_apertures(
            crate::linux::observe_process_apertures(&self.opened.fd)?,
            &topology_before,
        )?;
        if apertures_after != apertures_before {
            return Err(DeviceBindingError::AperturesChanged);
        }
        let topology_after = topology::discover_default_topology()?;
        if topology_after != topology_before {
            return Err(DeviceBindingError::TopologySnapshotChanged);
        }
        crate::linux::revalidate_descriptor(
            &self.opened.fd,
            self.opened.node_observation(),
            "KFD",
        )?;
        crate::linux::revalidate_render_descriptor(&render.fd, render.descriptor)?;
        let process_after = crate::linux::observe_process_incarnation()?;
        if process_after != process_before {
            return Err(DeviceBindingError::ProcessIncarnationChanged);
        }

        let selected_aperture = apertures_before
            .iter()
            .find(|aperture| aperture.gpu_id == kfd_gpu_id)
            .copied()
            .ok_or(DeviceBindingError::SelectedApertureMissing(kfd_gpu_id))?;
        // Keep this as the last fallible kernel observation. `model_admission`
        // commits process-local history, so no fence failure may follow it.
        reset_fence.check_clear()?;
        let (model_admission, projection, projection_history) = model_admission(
            &topology_before,
            gpu,
            render_sysfs,
            kfd_node,
            self.opened.node_observation(),
            render.descriptor,
            render.drm,
            process_before,
            &apertures_before,
        )?;

        let observation = DeviceBindingObservation {
            topology_node_id: gpu.node_id(),
            kfd_gpu_id,
            unique_id: gpu.unique_id(),
            pci: render_sysfs.pci_address(),
            render_minor: gpu.drm_render_minor(),
            render_descriptor: render.descriptor,
            drm: render.drm,
            aperture: selected_aperture,
        };

        Ok(CheckedGfx942XnackMinusDevice {
            kfd: self,
            render_fd: render.fd,
            render_path: render.path,
            topology: topology_before,
            apertures: apertures_before,
            observation,
            model_admission,
            projection,
            projection_history,
            process: process_before,
            reset_fence,
            currentness_poisoned: false,
            retire_model_on_drop: true,
            _lease: lease,
        })
    }
}

fn validate_platform_provenance(snapshot: &HostTopologySnapshot) -> Result<(), DeviceBindingError> {
    if snapshot.kernel_release().as_str() != ADMITTED_KERNEL_RELEASE_V1 {
        return Err(DeviceBindingError::UnsupportedKernelRelease(
            snapshot.kernel_release().as_str().to_owned(),
        ));
    }
    if snapshot.amdgpu_module().version() != Some(ADMITTED_AMDGPU_MODULE_VERSION_V1) {
        return Err(DeviceBindingError::UnsupportedModuleVersion(
            snapshot.amdgpu_module().version().map(str::to_owned),
        ));
    }
    if snapshot.amdgpu_module().srcversion() != Some(ADMITTED_AMDGPU_MODULE_SRCVERSION_V1) {
        return Err(DeviceBindingError::UnsupportedModuleSourceVersion(
            snapshot.amdgpu_module().srcversion().map(str::to_owned),
        ));
    }
    Ok(())
}

fn select_unique(
    snapshot: &HostTopologySnapshot,
    selector: DeviceSelector,
) -> Result<(&topology::GpuTopologyNode, &topology::RenderNodeObservation), DeviceBindingError> {
    let mut matches = snapshot
        .topology()
        .gpu_nodes()
        .iter()
        .filter(|node| match selector {
            DeviceSelector::UniqueId(unique_id) => node.unique_id() == unique_id,
        });
    let gpu = matches
        .next()
        .ok_or(DeviceBindingError::SelectorNotFound(selector))?;
    if matches.next().is_some() {
        return Err(DeviceBindingError::SelectorAmbiguous(selector));
    }
    let mut renders = snapshot
        .render_nodes()
        .iter()
        .filter(|render| render.node_id() == gpu.node_id());
    let render = renders
        .next()
        .ok_or(DeviceBindingError::RenderObservationMissing(gpu.node_id()))?;
    if renders.next().is_some() {
        return Err(DeviceBindingError::RenderObservationAmbiguous(
            gpu.node_id(),
        ));
    }
    Ok((gpu, render))
}

fn validate_render_profile(
    gpu: &topology::GpuTopologyNode,
    sysfs: &topology::RenderNodeObservation,
    render: &OpenedRender,
) -> Result<(), DeviceBindingError> {
    if gpu.pci_device_id() != MI300X_DEVICE_ID {
        return Err(DeviceBindingError::UnsupportedPciDevice(
            gpu.pci_device_id(),
        ));
    }
    if gpu.fw_version() != ADMITTED_KFD_FIRMWARE_VERSION_V1
        || gpu.sdma_fw_version() != ADMITTED_KFD_SDMA_FIRMWARE_VERSION_V1
    {
        return Err(DeviceBindingError::UnsupportedFirmware {
            compute: gpu.fw_version(),
            sdma: gpu.sdma_fw_version(),
        });
    }
    let capacity = gpu.capacity();
    if capacity.wavefront_size() != 64
        || capacity.simd_count() != MI300X_SPX_SIMD_COUNT
        || capacity.xcc_count() != MI300X_SPX_XCC_COUNT
    {
        return Err(DeviceBindingError::UnsupportedCapacity {
            wavefront: capacity.wavefront_size(),
            simds: capacity.simd_count(),
            xccs: capacity.xcc_count(),
        });
    }
    if sysfs.partition() != V1_PARTITION_PROFILE {
        return Err(DeviceBindingError::UnsupportedPartition);
    }
    validate_pci_revision(sysfs.pci_revision())?;
    if render.descriptor.major != DRM_DEVICE_MAJOR
        || render.descriptor.minor != u32::from(gpu.drm_render_minor())
    {
        return Err(DeviceBindingError::RenderDescriptorMismatch);
    }
    if render.drm.driver_version != AMDGPU_DRM_DRIVER_VERSION {
        return Err(DeviceBindingError::UnsupportedDrmVersion(
            render.drm.driver_version,
        ));
    }
    if render.drm.acceleration_working != 1 {
        return Err(DeviceBindingError::AccelerationUnavailable(
            render.drm.acceleration_working,
        ));
    }
    let device = render.drm.device;
    if device.device_id != u32::from(gpu.pci_device_id())
        || device.pci_rev != u32::from(sysfs.pci_revision())
        || device.family != AMDGPU_FAMILY_AI
        || device.chip_rev != 1
        || device.external_rev != 71
    {
        return Err(DeviceBindingError::DrmDeviceMismatch(device));
    }
    Ok(())
}

fn validate_pci_revision(revision: u8) -> Result<(), DeviceBindingError> {
    if revision != MI300X_PCI_REVISION {
        return Err(DeviceBindingError::UnsupportedPciRevision(revision));
    }
    Ok(())
}

fn checked_aperture(
    base: u64,
    limit: u64,
    name: &'static str,
    gpu_id: u32,
) -> Result<InclusiveAperture, DeviceBindingError> {
    let end = limit
        .checked_add(1)
        .ok_or(DeviceBindingError::InvalidAperture { gpu_id, name })?;
    if base > limit || !base.is_multiple_of(PAGE_BYTES) || !end.is_multiple_of(PAGE_BYTES) {
        return Err(DeviceBindingError::InvalidAperture { gpu_id, name });
    }
    Ok(InclusiveAperture { base, limit })
}

fn ranges_overlap(left: InclusiveAperture, right: InclusiveAperture) -> bool {
    left.base <= right.limit && right.base <= left.limit
}

pub(super) fn validate_apertures(
    raw: Vec<KfdProcessDeviceApertures>,
    topology: &HostTopologySnapshot,
) -> Result<Vec<ProcessApertureObservation>, DeviceBindingError> {
    if raw.is_empty() || raw.len() > MAX_PROCESS_APERTURES_V1 {
        return Err(DeviceBindingError::InvalidApertureCount(raw.len()));
    }
    let mut result: Vec<ProcessApertureObservation> = Vec::with_capacity(raw.len());
    for item in raw {
        if item.gpu_id == 0 || item.pad != 0 || result.iter().any(|old| old.gpu_id == item.gpu_id) {
            return Err(DeviceBindingError::InvalidApertureRecord(item.gpu_id));
        }
        let lds = checked_aperture(item.lds_base, item.lds_limit, "lds", item.gpu_id)?;
        let scratch = checked_aperture(
            item.scratch_base,
            item.scratch_limit,
            "scratch",
            item.gpu_id,
        )?;
        let gpuvm = checked_aperture(item.gpuvm_base, item.gpuvm_limit, "gpuvm", item.gpu_id)?;
        if ranges_overlap(lds, scratch)
            || ranges_overlap(lds, gpuvm)
            || ranges_overlap(scratch, gpuvm)
        {
            return Err(DeviceBindingError::OverlappingApertures(item.gpu_id));
        }
        result.push(ProcessApertureObservation {
            gpu_id: item.gpu_id,
            lds,
            scratch,
            gpuvm,
        });
    }
    result.sort_by_key(|item| item.gpu_id);

    let mut topology_ids = topology
        .topology()
        .gpu_nodes()
        .iter()
        .map(|node| {
            u32::try_from(node.gpu_id())
                .map_err(|_| DeviceBindingError::GpuIdOutOfRange(node.gpu_id()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    topology_ids.sort_unstable();
    let result_ids = result.iter().map(|item| item.gpu_id).collect::<Vec<_>>();
    if result_ids != topology_ids {
        return Err(DeviceBindingError::ApertureInventoryMismatch);
    }
    Ok(result)
}

fn model_domain(
    snapshot: &HostTopologySnapshot,
    process: ProcessIncarnationObservation,
) -> DeviceObservationDomainIdV1 {
    let provenance = snapshot.topology().provenance();
    let mut digest = Sha256::new();
    digest.update(b"fe2o3-device-observation-domain-v1\0");
    digest.update(snapshot.boot_id().as_bytes());
    digest.update(provenance.file_system_device().to_le_bytes());
    digest.update(provenance.inode().to_le_bytes());
    digest.update(provenance.generation().to_le_bytes());
    digest.update(process.pid.to_le_bytes());
    digest.update(process.start_time_ticks.to_le_bytes());
    digest.update(process.mount_namespace_device.to_le_bytes());
    digest.update(process.mount_namespace_inode.to_le_bytes());
    DeviceObservationDomainIdV1::from_untrusted_digest(IdentityDigestV1::from_untrusted_bytes(
        digest.finalize().into(),
    ))
}

fn model_pci(pci: PciAddress) -> PciAddressV1 {
    PciAddressV1 {
        domain: pci.domain(),
        bus: pci.bus(),
        device: pci.device(),
        function: pci.function(),
    }
}

fn model_profile() -> DeviceAdmissionProfileV1 {
    DeviceAdmissionProfileV1::gfx942_xnack_minus_spx_nps1_kfd_1_18_drm_3_64_0(
        DeviceAdmissionProfileIdV1::from_untrusted_digest(IdentityDigestV1::from_untrusted_bytes(
            DEVICE_ADMISSION_PROFILE_SHA256_BYTES_V1,
        )),
        IdentityDigestV1::from_untrusted_bytes(KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES),
        IdentityDigestV1::from_untrusted_bytes(DRM_UAPI_SCHEMA_MANIFEST_SHA256_BYTES),
    )
}

#[allow(clippy::too_many_arguments)]
fn model_admission(
    snapshot: &HostTopologySnapshot,
    gpu: &topology::GpuTopologyNode,
    sysfs: &topology::RenderNodeObservation,
    kfd_node: DeviceNodeV1,
    kfd_descriptor: crate::KfdNodeObservation,
    render_node: RenderDescriptorObservation,
    drm: DrmIdentityObservation,
    process: ProcessIncarnationObservation,
    apertures: &[ProcessApertureObservation],
) -> Result<
    (
        ModelDeviceAdmissionV1,
        ValidatedDeviceProjectionV1,
        DeviceProjectionHistoryLinkV1,
    ),
    DeviceBindingError,
> {
    let domain = model_domain(snapshot, process);
    let pci = model_pci(sysfs.pci_address());
    let profile = model_profile();
    let capacity = gpu.capacity();
    let inventory = snapshot
        .topology()
        .gpu_nodes()
        .iter()
        .map(|inventory_gpu| {
            let inventory_render = snapshot
                .render_nodes()
                .iter()
                .find(|render| render.node_id() == inventory_gpu.node_id())
                .ok_or(DeviceBindingError::RenderObservationMissing(
                    inventory_gpu.node_id(),
                ))?;
            Ok(InventoryDeviceProjectionV1 {
                topology_node_id: inventory_gpu.node_id(),
                kfd_gpu_id: u32::try_from(inventory_gpu.gpu_id())
                    .map_err(|_| DeviceBindingError::GpuIdOutOfRange(inventory_gpu.gpu_id()))?,
                gpu_unique_id: inventory_gpu.unique_id(),
                drm_render_minor: u32::from(inventory_gpu.drm_render_minor()),
                pci: model_pci(inventory_render.pci_address()),
                vendor_id: AMD_PCI_VENDOR_ID_V1,
                device_id: inventory_gpu.pci_device_id(),
                pci_revision_id: inventory_render.pci_revision(),
                target: fe2o3_runtime_model::GpuTargetObservationV1::Gfx942,
            })
        })
        .collect::<Result<Vec<_>, DeviceBindingError>>()?;
    let record = DeviceProjectionRecordV1 {
        schema_version: DEVICE_PROJECTION_SCHEMA_VERSION_V1,
        domain_id: domain,
        profile_id: profile.identity(),
        source: DeviceProjectionSourceV1 {
            boot_id: *snapshot.boot_id().as_bytes(),
            topology_file_system_device: snapshot.topology().provenance().file_system_device(),
            topology_inode: snapshot.topology().provenance().inode(),
            topology_generation: snapshot.topology().provenance().generation(),
            process_id: process.pid,
            process_start_time_ticks: process.start_time_ticks,
            mount_namespace_device: process.mount_namespace_device,
            mount_namespace_inode: process.mount_namespace_inode,
            amdgpu_module_file_system_device: snapshot.amdgpu_module().file_system_device(),
            amdgpu_module_inode: snapshot.amdgpu_module().inode(),
            kernel_release: KernelReleaseObservationV1::Linux6_8_0_124Generic,
            amdgpu_module: AmdgpuModuleObservationV1::Version6_16_13SourceA6f143bec60c0afc3263226,
        },
        kfd: KfdProjectionV1 {
            descriptor: CharacterDeviceProjectionV1 {
                file_system_device: kfd_descriptor.file_system_device(),
                inode: kfd_descriptor.inode(),
                character_device: kfd_descriptor.character_device(),
                node: kfd_node,
            },
            uapi_major: KFD_IOCTL_MAJOR_VERSION,
            uapi_minor: KFD_IOCTL_MINOR_VERSION,
            schema_identity: IdentityDigestV1::from_untrusted_bytes(
                KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
            ),
            xnack: XnackObservationV1::Disabled,
        },
        topology: TopologyProjectionV1 {
            node_id: gpu.node_id(),
            kfd_gpu_id: u32::try_from(gpu.gpu_id())
                .map_err(|_| DeviceBindingError::GpuIdOutOfRange(gpu.gpu_id()))?,
            gpu_unique_id: gpu.unique_id(),
            drm_render_minor: u32::from(gpu.drm_render_minor()),
            pci,
            vendor_id: AMD_PCI_VENDOR_ID_V1,
            device_id: gpu.pci_device_id(),
            target: fe2o3_runtime_model::GpuTargetObservationV1::Gfx942,
            compute_partition: ComputePartitionObservationV1::Spx,
            memory_partition: fe2o3_runtime_model::MemoryPartitionObservationV1::Nps1,
            firmware_version: gpu.fw_version(),
            sdma_firmware_version: gpu.sdma_fw_version(),
            wavefront_size: capacity.wavefront_size(),
            simd_count: capacity.simd_count(),
            xcc_count: capacity.xcc_count(),
        },
        inventory,
        render: RenderProjectionV1 {
            descriptor: CharacterDeviceProjectionV1 {
                file_system_device: render_node.file_system_device,
                inode: render_node.inode,
                character_device: render_node.character_device,
                node: DeviceNodeV1 {
                    major: render_node.major,
                    minor: render_node.minor,
                },
            },
            gpu_unique_id: sysfs.unique_id(),
            pci,
            vendor_id: AMD_PCI_VENDOR_ID_V1,
            device_id: u16::try_from(drm.device.device_id)
                .map_err(|_| DeviceBindingError::DrmDeviceMismatch(drm.device))?,
            pci_revision_id: u8::try_from(drm.device.pci_rev)
                .map_err(|_| DeviceBindingError::DrmDeviceMismatch(drm.device))?,
            schema_identity: IdentityDigestV1::from_untrusted_bytes(
                DRM_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
            ),
            driver_name: DrmDriverNameObservationV1::Amdgpu,
            driver_major: u32::try_from(drm.driver_version.major)
                .map_err(|_| DeviceBindingError::UnsupportedDrmVersion(drm.driver_version))?,
            driver_minor: u32::try_from(drm.driver_version.minor)
                .map_err(|_| DeviceBindingError::UnsupportedDrmVersion(drm.driver_version))?,
            driver_patch: u32::try_from(drm.driver_version.patch)
                .map_err(|_| DeviceBindingError::UnsupportedDrmVersion(drm.driver_version))?,
            acceleration_working: drm.acceleration_working == 1,
            family: DrmFamilyObservationV1::AmdgpuFamilyAi,
            family_id: drm.device.family,
            chip_revision: drm.device.chip_rev,
            external_revision: drm.device.external_rev,
            vram_lost_counter: drm.vram_lost_counter,
        },
        apertures: apertures
            .iter()
            .map(|aperture| ProcessApertureProjectionV1 {
                kfd_gpu_id: aperture.gpu_id,
                lds: InclusiveRangeProjectionV1 {
                    base: aperture.lds.base,
                    limit: aperture.lds.limit,
                },
                scratch: InclusiveRangeProjectionV1 {
                    base: aperture.scratch.base,
                    limit: aperture.scratch.limit,
                },
                gpuvm: InclusiveRangeProjectionV1 {
                    base: aperture.gpuvm.base,
                    limit: aperture.gpuvm.limit,
                },
            })
            .collect(),
        commit_fence: DeviceProjectionCommitFenceV1 {
            process_reobserved_equal: true,
            descriptors_revalidated: true,
            topology_reobserved_equal: true,
            xnack_reobserved_disabled: true,
            apertures_reobserved_equal: true,
            reset_subscription_established: true,
            reset_event_mask_enabled: true,
            reset_event_descriptor_cloexec: true,
            reset_fence_initially_clear: true,
            drm_reobserved_after_subscription_equal: true,
            reset_fence_clear_before_commit: true,
        },
    };
    let projection =
        fe2o3_runtime_model::validate_device_projection_model_only_v1(record, &profile)?;
    let correlation = projection.correlation();
    let mut history = DEVICE_MODEL_HISTORY
        .lock()
        .map_err(|_| DeviceBindingError::ModelHistoryPoisoned)?;
    let (identities, projections) = model_histories_for_domain(&history, domain)?;
    let generation = DeviceGenerationV1(next_admission_generation()?);
    // Check the projection journal first so its reviewed process-lifetime
    // capacity has the stable public `ProjectionHistoryExhausted` failure.
    let (next_projections, projection_history) = projections
        .append_model_only(projection.clone(), generation)
        .map_err(map_projection_history_error)?;
    let (next_identities, admission) = identities
        .register_device_model_only(correlation, generation)
        .map_err(DeviceBindingError::ModelAdmission)?;
    if admission.model_key() != projection_history.current() {
        return Err(DeviceBindingError::ProjectionLinkMismatch);
    }
    debug_assert!(next_identities.validate_global_invariants().is_ok());
    debug_assert!(next_projections.validate_global_invariants().is_ok());
    *history = DeviceModelHistory::State {
        identities: next_identities,
        projections: next_projections,
    };
    Ok((admission, projection, projection_history))
}

fn model_histories_for_domain(
    history: &DeviceModelHistory,
    domain: DeviceObservationDomainIdV1,
) -> Result<(DeviceIdentityStateV1, DeviceProjectionHistoryV1), DeviceBindingError> {
    match history {
        DeviceModelHistory::Empty => Ok((
            DeviceIdentityStateV1::new(domain),
            DeviceProjectionHistoryV1::new(domain),
        )),
        DeviceModelHistory::Poisoned => Err(DeviceBindingError::ModelHistoryPoisoned),
        DeviceModelHistory::State {
            identities,
            projections,
        } if identities.domain_id() == domain && projections.domain_id() == domain => {
            Ok((identities.clone(), projections.clone()))
        }
        DeviceModelHistory::State { identities, .. }
            if identities
                .devices()
                .iter()
                .any(|record| record.status == ModelAdmissionStatusV1::Active)
                || identities
                    .vms()
                    .iter()
                    .any(|record| record.status == ModelAdmissionStatusV1::Active) =>
        {
            Err(DeviceBindingError::ModelDomainChangedWithActiveHistory)
        }
        DeviceModelHistory::State { .. } => {
            Err(DeviceBindingError::ModelDomainChangedWithRetainedHistory)
        }
    }
}

fn map_projection_history_error(error: DeviceProjectionHistoryErrorV1) -> DeviceBindingError {
    match error {
        DeviceProjectionHistoryErrorV1::CapacityExceeded => {
            DeviceBindingError::ProjectionHistoryExhausted
        }
        error => DeviceBindingError::ProjectionHistory(error),
    }
}

fn next_admission_generation() -> Result<u64, DeviceBindingError> {
    NEXT_ADMISSION_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| DeviceBindingError::GenerationExhausted)
}

#[derive(Debug)]
pub enum DeviceBindingError {
    AdmissionInProgress,
    AdmissionLeasePoisoned,
    Kfd(KfdAdapterError),
    Topology(TopologyError),
    Syscall {
        operation: &'static str,
        source: rustix::io::Errno,
    },
    Io {
        operation: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    ProcessIncarnationChanged,
    KfdDescriptorMismatch,
    KfdSysfsMismatch,
    SelectorNotFound(DeviceSelector),
    SelectorAmbiguous(DeviceSelector),
    RenderObservationMissing(u32),
    RenderObservationAmbiguous(u32),
    RenderDescriptorMismatch,
    UnsupportedKernelRelease(String),
    UnsupportedModuleVersion(Option<String>),
    UnsupportedModuleSourceVersion(Option<String>),
    UnsupportedPciDevice(u16),
    UnsupportedPciRevision(u8),
    UnsupportedFirmware {
        compute: u32,
        sdma: u32,
    },
    UnsupportedCapacity {
        wavefront: u32,
        simds: u32,
        xccs: u32,
    },
    UnsupportedPartition,
    UnsupportedDrmVersion(DrmDriverVersion),
    InvalidDrmDriverName,
    AccelerationUnavailable(u32),
    DrmDeviceMismatch(DrmAmdgpuDeviceIdentityV1),
    UnsupportedXnackMode,
    XnackChanged,
    UapiChanged,
    InvalidResetEventDescriptor(u32),
    ResetEventFenceProtocol,
    WholeGpuResetObserved,
    ObservableCurrentnessChanged(&'static str),
    CurrentnessFencePoisoned,
    InvalidApertureCount(usize),
    InvalidApertureRecord(u32),
    InvalidAperture {
        gpu_id: u32,
        name: &'static str,
    },
    OverlappingApertures(u32),
    ApertureInventoryMismatch,
    AperturesChanged,
    SelectedApertureMissing(u32),
    GpuIdOutOfRange(u64),
    TopologySnapshotChanged,
    GenerationExhausted,
    ModelHistoryPoisoned,
    ModelDomainChangedWithActiveHistory,
    ModelDomainChangedWithRetainedHistory,
    ModelInventory(InventoryInputErrorV1),
    ModelCorrelation(fe2o3_runtime_model::DeviceCorrelationErrorV1),
    ModelAdmission(DeviceAdmissionErrorV1),
    Projection(DeviceProjectionErrorV1),
    ProjectionHistory(DeviceProjectionHistoryErrorV1),
    ProjectionHistoryExhausted,
    ProjectionLinkMismatch,
}

impl From<KfdAdapterError> for DeviceBindingError {
    fn from(error: KfdAdapterError) -> Self {
        Self::Kfd(error)
    }
}

impl From<TopologyError> for DeviceBindingError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

impl From<InventoryInputErrorV1> for DeviceBindingError {
    fn from(error: InventoryInputErrorV1) -> Self {
        Self::ModelInventory(error)
    }
}

impl From<fe2o3_runtime_model::DeviceCorrelationErrorV1> for DeviceBindingError {
    fn from(error: fe2o3_runtime_model::DeviceCorrelationErrorV1) -> Self {
        Self::ModelCorrelation(error)
    }
}

impl From<DeviceProjectionErrorV1> for DeviceBindingError {
    fn from(error: DeviceProjectionErrorV1) -> Self {
        Self::Projection(error)
    }
}

impl fmt::Display for DeviceBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdmissionInProgress => formatter
                .write_str("another fe2o3 device admission owns the process-wide KFD lease"),
            Self::AdmissionLeasePoisoned => {
                formatter.write_str("the process-wide KFD admission lease is poisoned")
            }
            Self::Kfd(error) => write!(formatter, "KFD admission failed: {error}"),
            Self::Topology(error) => write!(formatter, "topology admission failed: {error}"),
            Self::Syscall { operation, source } => {
                write!(formatter, "{operation} failed: {source}")
            }
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} {}: {source}",
                path.display()
            ),
            Self::ProcessIncarnationChanged => {
                formatter.write_str("process or mount namespace changed during device admission")
            }
            Self::KfdDescriptorMismatch => {
                formatter.write_str("the retained KFD descriptor identity changed")
            }
            Self::KfdSysfsMismatch => formatter
                .write_str("the retained KFD descriptor does not match the KFD sysfs identity"),
            Self::SelectorNotFound(selector) => write!(
                formatter,
                "device selector {selector:?} did not match the complete topology"
            ),
            Self::SelectorAmbiguous(selector) => write!(
                formatter,
                "device selector {selector:?} matched more than one topology node"
            ),
            Self::RenderObservationMissing(node) => write!(
                formatter,
                "topology node {node} has no correlated render observation"
            ),
            Self::RenderObservationAmbiguous(node) => write!(
                formatter,
                "topology node {node} has multiple render observations"
            ),
            Self::RenderDescriptorMismatch => formatter.write_str(
                "the retained render descriptor does not match the selected render node",
            ),
            Self::UnsupportedKernelRelease(value) => {
                write!(formatter, "unsupported kernel release {value:?}")
            }
            Self::UnsupportedModuleVersion(value) => {
                write!(formatter, "unsupported amdgpu module version {value:?}")
            }
            Self::UnsupportedModuleSourceVersion(value) => write!(
                formatter,
                "unsupported amdgpu module source version {value:?}"
            ),
            Self::UnsupportedPciDevice(value) => {
                write!(formatter, "unsupported gfx942 PCI device {value:#06x}")
            }
            Self::UnsupportedPciRevision(value) => {
                write!(formatter, "unsupported MI300X PCI revision {value:#04x}")
            }
            Self::UnsupportedFirmware { compute, sdma } => write!(
                formatter,
                "unsupported KFD firmware observations compute={compute} sdma={sdma}"
            ),
            Self::UnsupportedCapacity {
                wavefront,
                simds,
                xccs,
            } => write!(
                formatter,
                "unsupported MI300X capacity wavefront={wavefront} simds={simds} xccs={xccs}"
            ),
            Self::UnsupportedPartition => formatter
                .write_str("the selected GPU is not in the admitted SPX/NPS1 partition profile"),
            Self::UnsupportedDrmVersion(value) => {
                write!(formatter, "unsupported AMDGPU DRM version {value:?}")
            }
            Self::InvalidDrmDriverName => {
                formatter.write_str("DRM_VERSION did not return the exact amdgpu driver name")
            }
            Self::AccelerationUnavailable(value) => {
                write!(formatter, "AMDGPU acceleration status is {value}, not 1")
            }
            Self::DrmDeviceMismatch(value) => write!(
                formatter,
                "AMDGPU device identity does not match the admitted MI300X profile: {value:?}"
            ),
            Self::UnsupportedXnackMode => formatter.write_str("process XNACK mode is not disabled"),
            Self::XnackChanged => {
                formatter.write_str("process XNACK mode changed during the no-queue barrier")
            }
            Self::UapiChanged => {
                formatter.write_str("KFD UAPI observation changed during device admission")
            }
            Self::InvalidResetEventDescriptor(value) => write!(
                formatter,
                "KFD SMI_EVENTS returned invalid descriptor value {value}"
            ),
            Self::ResetEventFenceProtocol => formatter
                .write_str("KFD reset-event fence violated its admitted descriptor protocol"),
            Self::WholeGpuResetObserved => {
                formatter.write_str("KFD reported a whole-GPU reset after event-fence subscription")
            }
            Self::ObservableCurrentnessChanged(observation) => write!(
                formatter,
                "device currentness observation changed: {observation}"
            ),
            Self::CurrentnessFencePoisoned => {
                formatter.write_str("the device currentness fence is permanently poisoned")
            }
            Self::InvalidApertureCount(value) => {
                write!(formatter, "invalid process aperture count {value}")
            }
            Self::InvalidApertureRecord(gpu) => write!(
                formatter,
                "invalid or duplicate process aperture record for GPU {gpu}"
            ),
            Self::InvalidAperture { gpu_id, name } => {
                write!(formatter, "GPU {gpu_id} has an invalid {name} aperture")
            }
            Self::OverlappingApertures(gpu) => {
                write!(formatter, "GPU {gpu} process apertures overlap")
            }
            Self::ApertureInventoryMismatch => formatter
                .write_str("KFD aperture GPU IDs do not exactly match the topology inventory"),
            Self::AperturesChanged => {
                formatter.write_str("KFD process apertures changed during device admission")
            }
            Self::SelectedApertureMissing(gpu) => {
                write!(formatter, "selected GPU {gpu} has no process aperture")
            }
            Self::GpuIdOutOfRange(value) => write!(
                formatter,
                "KFD topology GPU ID {value} does not fit the UAPI u32 field"
            ),
            Self::TopologySnapshotChanged => formatter
                .write_str("the complete topology snapshot changed during device admission"),
            Self::GenerationExhausted => {
                formatter.write_str("the process-local device admission generation is exhausted")
            }
            Self::ModelHistoryPoisoned => {
                formatter.write_str("the process-local model admission history is poisoned")
            }
            Self::ModelDomainChangedWithActiveHistory => formatter.write_str(
                "the observation domain changed while model history retained active authority",
            ),
            Self::ModelDomainChangedWithRetainedHistory => formatter.write_str(
                "the observation domain changed after process-lifetime model history was retained",
            ),
            Self::ModelInventory(error) => write!(
                formatter,
                "model inventory rejected checked projections: {error:?}"
            ),
            Self::ModelCorrelation(error) => write!(
                formatter,
                "model correlation rejected checked projections: {error:?}"
            ),
            Self::ModelAdmission(error) => {
                write!(formatter, "model generation admission failed: {error:?}")
            }
            Self::Projection(error) => {
                write!(formatter, "canonical device projection rejected: {error:?}")
            }
            Self::ProjectionHistory(error) => {
                write!(formatter, "device projection history rejected: {error:?}")
            }
            Self::ProjectionHistoryExhausted => formatter.write_str(
                "the reviewed process-lifetime device projection history bound is exhausted",
            ),
            Self::ProjectionLinkMismatch => formatter
                .write_str("device identity and projection histories produced different links"),
        }
    }
}

impl std::error::Error for DeviceBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Kfd(error) => Some(error),
            Self::Topology(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_aperture(gpu_id: u32) -> KfdProcessDeviceApertures {
        KfdProcessDeviceApertures {
            lds_base: 0x1000,
            lds_limit: 0x1fff,
            scratch_base: 0x3000,
            scratch_limit: 0x3fff,
            gpuvm_base: 0x10_0000,
            gpuvm_limit: 0x1f_ffff,
            gpu_id,
            pad: 0,
        }
    }

    #[test]
    fn aperture_validation_rejects_overlap_and_noncanonical_end() {
        let raw = raw_aperture(7);
        let valid = ProcessApertureObservation {
            gpu_id: raw.gpu_id,
            lds: checked_aperture(raw.lds_base, raw.lds_limit, "lds", raw.gpu_id).unwrap(),
            scratch: checked_aperture(raw.scratch_base, raw.scratch_limit, "scratch", raw.gpu_id)
                .unwrap(),
            gpuvm: checked_aperture(raw.gpuvm_base, raw.gpuvm_limit, "gpuvm", raw.gpu_id).unwrap(),
        };
        assert_eq!(valid.gpuvm().size(), 0x10_0000);
        assert!(checked_aperture(0x1001, 0x1fff, "lds", 7).is_err());
        assert!(ranges_overlap(valid.lds(), valid.lds()));
    }

    #[test]
    fn revision_zero_profile_rejects_consistent_nonzero_revision() {
        assert!(validate_pci_revision(MI300X_PCI_REVISION).is_ok());
        assert!(matches!(
            validate_pci_revision(1),
            Err(DeviceBindingError::UnsupportedPciRevision(1))
        ));
    }

    #[test]
    fn projection_history_capacity_maps_to_named_process_lifetime_error() {
        assert!(matches!(
            map_projection_history_error(DeviceProjectionHistoryErrorV1::CapacityExceeded),
            DeviceBindingError::ProjectionHistoryExhausted
        ));
    }

    fn retired_model_history(domain: DeviceObservationDomainIdV1) -> DeviceModelHistory {
        let profile = model_profile();
        let pci = PciAddressV1 {
            domain: 0,
            bus: 5,
            device: 0,
            function: 0,
        };
        let projection = fe2o3_runtime_model::validate_device_projection_model_only_v1(
            DeviceProjectionRecordV1 {
                schema_version: DEVICE_PROJECTION_SCHEMA_VERSION_V1,
                domain_id: domain,
                profile_id: profile.identity(),
                source: DeviceProjectionSourceV1 {
                    boot_id: [1; 16],
                    topology_file_system_device: 2,
                    topology_inode: 3,
                    topology_generation: 4,
                    process_id: 5,
                    process_start_time_ticks: 6,
                    mount_namespace_device: 7,
                    mount_namespace_inode: 8,
                    amdgpu_module_file_system_device: 9,
                    amdgpu_module_inode: 10,
                    kernel_release: KernelReleaseObservationV1::Linux6_8_0_124Generic,
                    amdgpu_module:
                        AmdgpuModuleObservationV1::Version6_16_13SourceA6f143bec60c0afc3263226,
                },
                kfd: KfdProjectionV1 {
                    descriptor: CharacterDeviceProjectionV1 {
                        file_system_device: 11,
                        inode: 12,
                        character_device: 13,
                        node: DeviceNodeV1 {
                            major: 511,
                            minor: fe2o3_runtime_model::KFD_DEVICE_MINOR_V1,
                        },
                    },
                    uapi_major: fe2o3_runtime_model::KFD_UAPI_MAJOR_V1,
                    uapi_minor: fe2o3_runtime_model::KFD_UAPI_MINOR_V1,
                    schema_identity: profile.kfd_schema_identity(),
                    xnack: XnackObservationV1::Disabled,
                },
                topology: TopologyProjectionV1 {
                    node_id: 2,
                    kfd_gpu_id: 100,
                    gpu_unique_id: 200,
                    drm_render_minor: fe2o3_runtime_model::DRM_RENDER_MIN_MINOR_V1,
                    pci,
                    vendor_id: AMD_PCI_VENDOR_ID_V1,
                    device_id: fe2o3_runtime_model::MI300X_PCI_DEVICE_ID_V1,
                    target: fe2o3_runtime_model::GpuTargetObservationV1::Gfx942,
                    compute_partition: ComputePartitionObservationV1::Spx,
                    memory_partition: fe2o3_runtime_model::MemoryPartitionObservationV1::Nps1,
                    firmware_version: fe2o3_runtime_model::MI300X_KFD_FIRMWARE_VERSION_V1,
                    sdma_firmware_version: fe2o3_runtime_model::MI300X_SDMA_FIRMWARE_VERSION_V1,
                    wavefront_size: fe2o3_runtime_model::MI300X_WAVEFRONT_SIZE_V1,
                    simd_count: fe2o3_runtime_model::MI300X_SPX_SIMD_COUNT_V1,
                    xcc_count: fe2o3_runtime_model::MI300X_SPX_XCC_COUNT_V1,
                },
                inventory: vec![InventoryDeviceProjectionV1 {
                    topology_node_id: 2,
                    kfd_gpu_id: 100,
                    gpu_unique_id: 200,
                    drm_render_minor: fe2o3_runtime_model::DRM_RENDER_MIN_MINOR_V1,
                    pci,
                    vendor_id: AMD_PCI_VENDOR_ID_V1,
                    device_id: fe2o3_runtime_model::MI300X_PCI_DEVICE_ID_V1,
                    pci_revision_id: fe2o3_runtime_model::MI300X_PCI_REVISION_V1,
                    target: fe2o3_runtime_model::GpuTargetObservationV1::Gfx942,
                }],
                render: RenderProjectionV1 {
                    descriptor: CharacterDeviceProjectionV1 {
                        file_system_device: 14,
                        inode: 15,
                        character_device: 16,
                        node: DeviceNodeV1 {
                            major: fe2o3_runtime_model::DRM_DEVICE_MAJOR_V1,
                            minor: fe2o3_runtime_model::DRM_RENDER_MIN_MINOR_V1,
                        },
                    },
                    gpu_unique_id: 200,
                    pci,
                    vendor_id: AMD_PCI_VENDOR_ID_V1,
                    device_id: fe2o3_runtime_model::MI300X_PCI_DEVICE_ID_V1,
                    pci_revision_id: fe2o3_runtime_model::MI300X_PCI_REVISION_V1,
                    schema_identity: profile.drm_schema_identity(),
                    driver_name: DrmDriverNameObservationV1::Amdgpu,
                    driver_major: fe2o3_runtime_model::DRM_DRIVER_MAJOR_V1,
                    driver_minor: fe2o3_runtime_model::DRM_DRIVER_MINOR_V1,
                    driver_patch: fe2o3_runtime_model::DRM_DRIVER_PATCH_V1,
                    acceleration_working: true,
                    family: DrmFamilyObservationV1::AmdgpuFamilyAi,
                    family_id: fe2o3_runtime_model::AMDGPU_FAMILY_AI_V1,
                    chip_revision: fe2o3_runtime_model::MI300X_CHIP_REVISION_V1,
                    external_revision: fe2o3_runtime_model::MI300X_EXTERNAL_REVISION_V1,
                    vram_lost_counter: 17,
                },
                apertures: vec![ProcessApertureProjectionV1 {
                    kfd_gpu_id: 100,
                    lds: InclusiveRangeProjectionV1 {
                        base: 0x1000,
                        limit: 0x1fff,
                    },
                    scratch: InclusiveRangeProjectionV1 {
                        base: 0x3000,
                        limit: 0x3fff,
                    },
                    gpuvm: InclusiveRangeProjectionV1 {
                        base: 0x10_0000,
                        limit: 0x1f_ffff,
                    },
                }],
                commit_fence: DeviceProjectionCommitFenceV1 {
                    process_reobserved_equal: true,
                    descriptors_revalidated: true,
                    topology_reobserved_equal: true,
                    xnack_reobserved_disabled: true,
                    apertures_reobserved_equal: true,
                    reset_subscription_established: true,
                    reset_event_mask_enabled: true,
                    reset_event_descriptor_cloexec: true,
                    reset_fence_initially_clear: true,
                    drm_reobserved_after_subscription_equal: true,
                    reset_fence_clear_before_commit: true,
                },
            },
            &profile,
        )
        .unwrap();
        let generation = DeviceGenerationV1(1);
        let (identities, admission) = DeviceIdentityStateV1::new(domain)
            .register_device_model_only(projection.correlation(), generation)
            .unwrap();
        let identities = identities.retire_device_model_only(admission).unwrap();
        let (projections, _) = DeviceProjectionHistoryV1::new(domain)
            .append_model_only(projection, generation)
            .unwrap();
        DeviceModelHistory::State {
            identities,
            projections,
        }
    }

    #[test]
    fn inactive_retained_history_rejects_domain_change_without_replacement() {
        let old_domain = DeviceObservationDomainIdV1::from_untrusted_digest(
            IdentityDigestV1::from_untrusted_bytes([41; 32]),
        );
        let new_domain = DeviceObservationDomainIdV1::from_untrusted_digest(
            IdentityDigestV1::from_untrusted_bytes([42; 32]),
        );
        let state = retired_model_history(old_domain);
        assert!(matches!(
            model_histories_for_domain(&state, new_domain),
            Err(DeviceBindingError::ModelDomainChangedWithRetainedHistory)
        ));
        let DeviceModelHistory::State {
            identities,
            projections,
        } = state
        else {
            panic!("test state changed variant")
        };
        assert_eq!(identities.domain_id(), old_domain);
        assert_eq!(identities.devices().len(), 1);
        assert_eq!(
            identities.devices()[0].status,
            ModelAdmissionStatusV1::Retired
        );
        assert_eq!(projections.domain_id(), old_domain);
        assert_eq!(projections.entries().len(), 1);
    }

    #[test]
    fn profile_manifest_digest_is_frozen() {
        use sha2::{Digest, Sha256};

        let digest = Sha256::digest(DEVICE_ADMISSION_PROFILE_MANIFEST_V1);
        assert_eq!(&digest[..], &DEVICE_ADMISSION_PROFILE_SHA256_BYTES_V1);
        let digest_hex = DEVICE_ADMISSION_PROFILE_SHA256_BYTES_V1
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(DEVICE_ADMISSION_PROFILE_SHA256_V1, digest_hex);
        assert!(DEVICE_ADMISSION_PROFILE_MANIFEST_V1.contains(&format!(
            "kfd_schema_sha256={}\n",
            fe2o3_kfd_uapi::KFD_UAPI_SCHEMA_MANIFEST_SHA256
        )));
        assert!(DEVICE_ADMISSION_PROFILE_MANIFEST_V1.contains(&format!(
            "drm_schema_sha256={}\n",
            fe2o3_drm_uapi::DRM_UAPI_SCHEMA_MANIFEST_SHA256
        )));
        assert!(
            DEVICE_ADMISSION_PROFILE_MANIFEST_V1
                .contains(&format!("kernel_release={ADMITTED_KERNEL_RELEASE_V1}\n"))
        );
        assert!(DEVICE_ADMISSION_PROFILE_MANIFEST_V1.contains(&format!(
            "amdgpu_module={ADMITTED_AMDGPU_MODULE_VERSION_V1}\n"
        )));
        assert!(DEVICE_ADMISSION_PROFILE_MANIFEST_V1.contains(&format!(
            "amdgpu_srcversion={ADMITTED_AMDGPU_MODULE_SRCVERSION_V1}\n"
        )));
        assert!(DEVICE_ADMISSION_PROFILE_MANIFEST_V1.contains(&format!(
            "firmware_observation=compute:{ADMITTED_KFD_FIRMWARE_VERSION_V1},sdma:{ADMITTED_KFD_SDMA_FIRMWARE_VERSION_V1}\n"
        )));
        assert_ne!(
            DEVICE_ADMISSION_PROFILE_SHA256_BYTES_V1,
            KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES
        );
        assert_ne!(
            DEVICE_ADMISSION_PROFILE_SHA256_BYTES_V1,
            DRM_UAPI_SCHEMA_MANIFEST_SHA256_BYTES
        );
        assert_ne!(
            KFD_UAPI_SCHEMA_MANIFEST_SHA256_BYTES,
            DRM_UAPI_SCHEMA_MANIFEST_SHA256_BYTES
        );
    }
}

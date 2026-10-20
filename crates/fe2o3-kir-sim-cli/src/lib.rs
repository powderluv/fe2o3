#![forbid(unsafe_code)]

mod bundle_declared_target_v1;
pub use bundle_declared_target_v1::{AdmittedBundleTargetV1, BundleDeclaredGpuTargetV1};

mod ordered_program_inspection;
pub use ordered_program_inspection::run_ordered_program_inspector;

#[cfg(target_os = "linux")]
mod linux;
mod schema;

use std::path::Path;
use std::process::ExitCode;

use fe2o3_kernel_ir::{
    VerifiedSimulationBundleV1, VerifiedSimulationBundleV2, VerifiedSimulationBundleV3,
    VerifiedSimulationBundleV4, VerifiedSimulationBundleV5, VerifiedSimulationBundleV6,
};
use fe2o3_kir_sim::{
    AdmittedSimulationModuleV1, PersistedSimulationScheduleArtifactV1,
    PersistedSimulationScheduleBindingV1, PersistedSimulationScheduleDocumentV1,
    SimulationLimitsV1, SimulationRequestV1, SimulationTargetV1,
};

/// Exact, strictly parsed simulator inputs admitted through the standalone
/// command's hardened file boundary.
#[derive(Debug)]
pub struct AdmittedSimulationInputV1 {
    pub module: AdmittedSimulationModuleV1,
    pub request: SimulationRequestV1,
    pub simulation_limits: SimulationLimitsV1,
    simulation_target: SimulationTargetV1,
    /// Present only when admission retained a verified simulation-bundle
    /// subject. Debugger configuration identities bind this exact subject.
    simulation_bundle_subject: Option<[u8; 32]>,
    /// Complete content identity of the exact admitted canonical bundle bytes.
    simulation_bundle_identity: Option<[u8; 32]>,
    simulation_bundle_evidence: Option<AdmittedSimulationBundleEvidenceV1>,
    // Captured simulator identity; legacy bundles may retain a different production KIR.
    simulation_bundle_kir_identity: Option<fe2o3_kir_sim::SimulationKernelIrIdentityV1>,
    bundle_target_v1: Option<AdmittedBundleTargetV1>,
    pub kir_sha256: [u8; 32],
    pub request_sha256: [u8; 32],
    request_bytes: u64,
}

/// Bounded exact content references retained from a verified simulation bundle.
/// These references grant no compiler, proof, artifact, load, launch, or hardware authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmittedSimulationBundleEvidenceV1 {
    pub envelope_version: u16,
    pub envelope_identity: [u8; 32],
    pub subject_identity: [u8; 32],
    pub production_kir_version: u16,
    pub production_kir_sha256: [u8; 32],
    pub production_kir_bytes: u64,
    pub kernel_abi_identity: [u8; 32],
    pub identity_inventory_receipt_sha256: [u8; 32],
    pub identity_inventory_receipt_bytes: u64,
    pub preflight_plan_receipt_sha256: [u8; 32],
    pub preflight_plan_receipt_bytes: u64,
}

impl AdmittedSimulationInputV1 {
    pub const fn simulation_target(&self) -> SimulationTargetV1 {
        self.simulation_target
    }

    pub const fn simulation_bundle_subject(&self) -> Option<[u8; 32]> {
        self.simulation_bundle_subject
    }

    /// Complete canonical bundle identity retained during exact admission.
    pub const fn simulation_bundle_identity(&self) -> Option<[u8; 32]> {
        self.simulation_bundle_identity
    }

    /// Exact authority-free bundle and source-lineage references retained at admission.
    pub const fn simulation_bundle_evidence(&self) -> Option<AdmittedSimulationBundleEvidenceV1> {
        self.simulation_bundle_evidence
    }

    /// Exact byte length of the strictly admitted request document.
    pub const fn request_bytes(&self) -> u64 {
        self.request_bytes
    }

    /// Exact artifact/request/target/limit binding used by persisted schedules.
    /// Diagnostic V16/V17 have no persisted schedule representation and are rejected
    /// explicitly; successful CPU admission does not grant a schedule binding.
    pub fn persisted_schedule_binding(
        &self,
    ) -> Result<PersistedSimulationScheduleBindingV1, SimulationInputErrorV1> {
        let unsupported = || SimulationInputErrorV1 {
            stage: "arguments".to_owned(),
            code: "schedule_input_unsupported".to_owned(),
            message: format!(
                "persisted schedules do not represent this admitted KIR V{} input",
                self.module.identity().wire_version()
            ),
        };
        // Also guard callers that replace the public module of an otherwise
        // bundle-backed input. Bundle metadata must never relabel raw V16/V17.
        if !matches!(self.module.identity().wire_version(), 7 | 9 | 10 | 11 | 12) {
            return Err(unsupported());
        }
        // Public fields may be replaced after admission. Bundle schedules must
        // still describe the exact simulator KIR captured with that bundle.
        // Its identity can differ from the bundle's production KIR identity.
        let identity = self.module.identity();
        if self.kir_sha256 != *identity.digest()
            || self
                .simulation_bundle_kir_identity
                .is_some_and(|captured| captured != *identity)
            || self.simulation_bundle_identity.is_some()
                != self.simulation_bundle_kir_identity.is_some()
            || self.simulation_bundle_evidence.is_some_and(|evidence| {
                Some(evidence.envelope_identity) != self.simulation_bundle_identity
                    || Some(evidence.subject_identity) != self.simulation_bundle_subject
            })
        {
            return Err(SimulationInputErrorV1 {
                stage: "arguments".to_owned(),
                code: "schedule_input_unsupported".to_owned(),
                message: "persisted schedules require a consistent admitted KIR identity and retained bundle identity".to_owned(),
            });
        }
        let artifact = match (
            self.simulation_bundle_identity,
            self.simulation_bundle_subject,
        ) {
            (None, None) => match self.module.identity().wire_version() {
                7 => PersistedSimulationScheduleArtifactV1::CanonicalKirV7,
                9 => PersistedSimulationScheduleArtifactV1::CanonicalKirV9,
                10 => PersistedSimulationScheduleArtifactV1::CanonicalKirV10,
                11 => PersistedSimulationScheduleArtifactV1::CanonicalKirV11,
                12 => PersistedSimulationScheduleArtifactV1::CanonicalKirV12,
                _ => return Err(unsupported()),
            },
            (Some(bundle_sha256), Some(subject_sha256))
                if self
                    .simulation_bundle_evidence
                    .is_some_and(|evidence| evidence.envelope_version == 6) =>
            {
                PersistedSimulationScheduleArtifactV1::SimulationBundleV6 {
                    bundle_sha256,
                    subject_sha256,
                }
            }
            (Some(bundle_sha256), Some(subject_sha256))
                if self
                    .simulation_bundle_evidence
                    .is_some_and(|evidence| evidence.envelope_version == 5) =>
            {
                PersistedSimulationScheduleArtifactV1::SimulationBundleV5 {
                    bundle_sha256,
                    subject_sha256,
                }
            }
            (Some(bundle_sha256), Some(subject_sha256)) => {
                PersistedSimulationScheduleArtifactV1::SimulationBundleV1 {
                    bundle_sha256,
                    subject_sha256,
                }
            }
            _ => return Err(unsupported()),
        };
        Ok(PersistedSimulationScheduleBindingV1::new(
            artifact,
            *self.module.identity(),
            self.request_sha256,
            self.request_bytes,
            self.simulation_target,
            self.simulation_limits,
        ))
    }
}

/// Strictly decoded bundle custody plus its exact admitted simulation input.
///
/// The verified bundle remains authority-free. Keeping it intact prevents a
/// caller from substituting loose target, subject, or debug-map identities
/// after admission.
#[derive(Debug)]
pub struct AdmittedSimulationBundleInputV1 {
    input: AdmittedSimulationInputV1,
    bundle: VerifiedSimulationBundleV1,
}

impl AdmittedSimulationBundleInputV1 {
    pub fn input(&self) -> &AdmittedSimulationInputV1 {
        &self.input
    }

    pub fn bundle(&self) -> &VerifiedSimulationBundleV1 {
        &self.bundle
    }

    pub fn into_parts(self) -> (AdmittedSimulationInputV1, VerifiedSimulationBundleV1) {
        (self.input, self.bundle)
    }

    pub const fn grants_proof_authority(&self) -> bool {
        self.bundle.grants_proof_authority()
    }

    pub const fn grants_artifact_authority(&self) -> bool {
        self.bundle.grants_artifact_authority()
    }

    pub const fn grants_compiler_authority(&self) -> bool {
        self.bundle.grants_compiler_authority()
    }

    pub const fn grants_hardware_authority(&self) -> bool {
        self.bundle.grants_hardware_authority()
    }

    pub const fn grants_load_authority(&self) -> bool {
        self.bundle.grants_load_authority()
    }

    pub const fn grants_launch_authority(&self) -> bool {
        self.bundle.grants_launch_authority()
    }

    pub const fn authenticates_compiler_execution(&self) -> bool {
        self.bundle.authenticates_compiler_execution()
    }
}

/// Strict V2 envelope custody plus the exact admitted simulation input.
#[derive(Debug)]
pub struct AdmittedSimulationBundleInputV2 {
    input: AdmittedSimulationInputV1,
    bundle: VerifiedSimulationBundleV2,
}

/// Strict V3 custody plus the exact admitted simulator request.
#[derive(Debug)]
pub struct AdmittedSimulationBundleInputV3 {
    input: AdmittedSimulationInputV1,
    bundle: VerifiedSimulationBundleV3,
}

/// Strict V4 aggregate-component custody plus the exact admitted request.
#[derive(Debug)]
pub struct AdmittedSimulationBundleInputV4 {
    input: AdmittedSimulationInputV1,
    bundle: VerifiedSimulationBundleV4,
}

/// Strict V5/V10 custody plus the exact admitted request.
#[derive(Debug)]
pub struct AdmittedSimulationBundleInputV5 {
    input: AdmittedSimulationInputV1,
    bundle: VerifiedSimulationBundleV5,
}

/// Strict V6/V11 custody plus the exact admitted request.
#[derive(Debug)]
pub struct AdmittedSimulationBundleInputV6 {
    input: AdmittedSimulationInputV1,
    bundle: VerifiedSimulationBundleV6,
}

impl AdmittedSimulationBundleInputV6 {
    pub fn input(&self) -> &AdmittedSimulationInputV1 {
        &self.input
    }
    pub fn bundle(&self) -> &VerifiedSimulationBundleV6 {
        &self.bundle
    }
    pub fn into_parts(self) -> (AdmittedSimulationInputV1, VerifiedSimulationBundleV6) {
        (self.input, self.bundle)
    }
    pub const fn grants_proof_authority(&self) -> bool {
        self.bundle.grants_proof_authority()
    }
    pub const fn grants_artifact_authority(&self) -> bool {
        self.bundle.grants_artifact_authority()
    }
    pub const fn grants_compiler_authority(&self) -> bool {
        self.bundle.grants_compiler_authority()
    }
    pub const fn authenticates_compiler_execution(&self) -> bool {
        self.bundle.authenticates_compiler_execution()
    }
    pub const fn grants_hardware_authority(&self) -> bool {
        self.bundle.grants_hardware_authority()
    }
    pub const fn grants_load_authority(&self) -> bool {
        self.bundle.grants_load_authority()
    }
    pub const fn grants_launch_authority(&self) -> bool {
        self.bundle.grants_launch_authority()
    }
}

impl AdmittedSimulationBundleInputV5 {
    pub fn input(&self) -> &AdmittedSimulationInputV1 {
        &self.input
    }
    pub fn bundle(&self) -> &VerifiedSimulationBundleV5 {
        &self.bundle
    }
    pub fn into_parts(self) -> (AdmittedSimulationInputV1, VerifiedSimulationBundleV5) {
        (self.input, self.bundle)
    }
    pub const fn grants_proof_authority(&self) -> bool {
        self.bundle.grants_proof_authority()
    }
    pub const fn grants_artifact_authority(&self) -> bool {
        self.bundle.grants_artifact_authority()
    }
    pub const fn grants_compiler_authority(&self) -> bool {
        self.bundle.grants_compiler_authority()
    }
    pub const fn authenticates_compiler_execution(&self) -> bool {
        self.bundle.authenticates_compiler_execution()
    }
    pub const fn grants_hardware_authority(&self) -> bool {
        self.bundle.grants_hardware_authority()
    }
    pub const fn grants_load_authority(&self) -> bool {
        self.bundle.grants_load_authority()
    }
    pub const fn grants_launch_authority(&self) -> bool {
        self.bundle.grants_launch_authority()
    }
}

impl AdmittedSimulationBundleInputV4 {
    pub fn input(&self) -> &AdmittedSimulationInputV1 {
        &self.input
    }
    pub fn bundle(&self) -> &VerifiedSimulationBundleV4 {
        &self.bundle
    }
    pub fn into_parts(self) -> (AdmittedSimulationInputV1, VerifiedSimulationBundleV4) {
        (self.input, self.bundle)
    }
    pub const fn grants_proof_authority(&self) -> bool {
        self.bundle.grants_proof_authority()
    }
    pub const fn grants_artifact_authority(&self) -> bool {
        self.bundle.grants_artifact_authority()
    }
    pub const fn grants_compiler_authority(&self) -> bool {
        self.bundle.grants_compiler_authority()
    }
    pub const fn authenticates_compiler_execution(&self) -> bool {
        self.bundle.authenticates_compiler_execution()
    }
    pub const fn grants_hardware_authority(&self) -> bool {
        self.bundle.grants_hardware_authority()
    }
    pub const fn grants_load_authority(&self) -> bool {
        self.bundle.grants_load_authority()
    }
    pub const fn grants_launch_authority(&self) -> bool {
        self.bundle.grants_launch_authority()
    }
}

impl AdmittedSimulationBundleInputV3 {
    pub fn input(&self) -> &AdmittedSimulationInputV1 {
        &self.input
    }
    pub fn bundle(&self) -> &VerifiedSimulationBundleV3 {
        &self.bundle
    }
    pub fn into_parts(self) -> (AdmittedSimulationInputV1, VerifiedSimulationBundleV3) {
        (self.input, self.bundle)
    }
    pub const fn grants_proof_authority(&self) -> bool {
        self.bundle.grants_proof_authority()
    }
    pub const fn grants_artifact_authority(&self) -> bool {
        self.bundle.grants_artifact_authority()
    }
    pub const fn grants_compiler_authority(&self) -> bool {
        self.bundle.grants_compiler_authority()
    }
    pub const fn authenticates_compiler_execution(&self) -> bool {
        self.bundle.authenticates_compiler_execution()
    }
    pub const fn grants_hardware_authority(&self) -> bool {
        self.bundle.grants_hardware_authority()
    }
    pub const fn grants_load_authority(&self) -> bool {
        self.bundle.grants_load_authority()
    }
    pub const fn grants_launch_authority(&self) -> bool {
        self.bundle.grants_launch_authority()
    }
}

impl AdmittedSimulationBundleInputV2 {
    pub fn input(&self) -> &AdmittedSimulationInputV1 {
        &self.input
    }
    pub fn bundle(&self) -> &VerifiedSimulationBundleV2 {
        &self.bundle
    }
    pub fn into_parts(self) -> (AdmittedSimulationInputV1, VerifiedSimulationBundleV2) {
        (self.input, self.bundle)
    }
    pub const fn grants_proof_authority(&self) -> bool {
        self.bundle.grants_proof_authority()
    }
    pub const fn grants_artifact_authority(&self) -> bool {
        self.bundle.grants_artifact_authority()
    }
    pub const fn grants_compiler_authority(&self) -> bool {
        self.bundle.grants_compiler_authority()
    }
    pub const fn authenticates_compiler_execution(&self) -> bool {
        self.bundle.authenticates_compiler_execution()
    }
    pub const fn grants_hardware_authority(&self) -> bool {
        self.bundle.grants_hardware_authority()
    }
    pub const fn grants_load_authority(&self) -> bool {
        self.bundle.grants_load_authority()
    }
    pub const fn grants_launch_authority(&self) -> bool {
        self.bundle.grants_launch_authority()
    }
}

/// Bounded failure returned while securely loading debugger simulator inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationInputErrorV1 {
    pub stage: String,
    pub code: String,
    pub message: String,
}

impl std::fmt::Display for SimulationInputErrorV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "simulation input admission failed at {} ({}): {}",
            self.stage, self.code, self.message
        )
    }
}

impl std::error::Error for SimulationInputErrorV1 {}

#[cfg(not(target_os = "linux"))]
use std::io::Write as _;

/// Runs the standalone command-line boundary with the process argument vector.
pub fn main() -> ExitCode {
    #[cfg(target_os = "linux")]
    {
        linux::main()
    }
    #[cfg(not(target_os = "linux"))]
    {
        unsupported_platform()
    }
}

/// Executes an already captured exact canonical KIR V7 image with the same
/// request, result, error, and output-publication boundary as the standalone
/// CLI. The caller supplies inert compiler-custody bytes, not a path that can
/// be substituted after capture.
pub fn run_captured_kir_v7(
    canonical_kir_v7: &[u8],
    request: &Path,
    output: Option<&Path>,
) -> ExitCode {
    #[cfg(target_os = "linux")]
    {
        linux::run_captured_kir_v7(
            canonical_kir_v7,
            request.as_os_str().to_owned(),
            None,
            output.map(|path| path.as_os_str().to_owned()),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (canonical_kir_v7, request, output);
        unsupported_platform()
    }
}

/// Executes captured KIR only if a secure reread has the same length and exact
/// bytes admitted before the build. Byte-identical pathname or inode
/// replacement is content-equivalent and remains admissible.
pub fn run_captured_kir_v7_with_bound_request(
    canonical_kir_v7: &[u8],
    request: &Path,
    expected_request: SimulationRequestIdentityV1,
    output: Option<&Path>,
) -> ExitCode {
    #[cfg(target_os = "linux")]
    {
        linux::run_captured_kir_v7(
            canonical_kir_v7,
            request.as_os_str().to_owned(),
            Some(expected_request),
            output.map(|path| path.as_os_str().to_owned()),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (canonical_kir_v7, request, expected_request, output);
        unsupported_platform()
    }
}

#[cfg(not(target_os = "linux"))]
fn unsupported_platform() -> ExitCode {
    #[derive(serde::Serialize)]
    struct PlatformError {
        schema: &'static str,
        status: &'static str,
        stage: schema::Stage,
        kind: schema::ErrorKind,
        message: &'static str,
    }
    let error = PlatformError {
        schema: "fe2o3-simulation-error-v1",
        status: "error",
        stage: schema::Stage::Platform,
        kind: schema::ErrorKind::UnsupportedPlatform,
        message: "fe2o3-kir-sim requires Linux openat2, O_TMPFILE, procfs fd links, and linkat",
    };
    let mut stderr = std::io::stderr().lock();
    let _ = serde_json::to_writer(&mut stderr, &error);
    let _ = stderr.write_all(b"\n");
    ExitCode::FAILURE
}
/// Exact pre-build identity of one admitted simulation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationRequestIdentityV1 {
    sha256: [u8; 32],
    length: usize,
}

impl SimulationRequestIdentityV1 {
    /// Returns the SHA-256 of the exact strict request bytes.
    pub const fn sha256(self) -> [u8; 32] {
        self.sha256
    }

    /// Returns the exact request byte length.
    pub const fn length(self) -> usize {
        self.length
    }
}

/// Securely reads and strictly admits the request before a source build starts.
pub fn bind_request_v1(path: &Path) -> Result<SimulationRequestIdentityV1, String> {
    #[cfg(target_os = "linux")]
    {
        linux::bind_request_v1(path.as_os_str().to_owned())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err("fe2o3 simulation request binding requires Linux".to_owned())
    }
}

/// Securely captures, strictly parses, and admits one exact KIR V7 image and
/// simulation request for a debugger session. This is the same ingestion path
/// used by `fe2o3-kir-sim`; callers never parse either document themselves.
pub fn load_debug_simulation_input_v1(
    kir_v7: &Path,
    request: &Path,
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_input_v1(
            kir_v7.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v7, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation input admission requires Linux".to_owned(),
        })
    }
}

/// Strictly admits one already captured KIR V7 image and simulation request.
///
/// This is the descriptor-neutral form of [`load_debug_simulation_input_v1`].
/// Callers remain responsible for authenticating how the exact bounded bytes
/// were captured; admission, identities, and simulation semantics are shared
/// with the path-based debugger entry point.
pub fn load_debug_simulation_input_bytes_v1(
    kir_v7: &[u8],
    request: &[u8],
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_input_bytes_v1(kir_v7, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v7, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation byte admission requires Linux".to_owned(),
        })
    }
}

/// Securely admits exact canonical V16 bytes and a strict simulation request.
///
/// This diagnostic CPU route authenticates neither source nor hardware and
/// grants no compiler, artifact, launch, source-map or persisted-schedule
/// authority. Canonical admission uses a cumulative work/storage ledger;
/// simulator admission separately checks resident storage after bounded decode,
/// not as a process-wide pre-allocation or RSS guarantee.
pub fn load_debug_simulation_input_v16(
    kir_v16: &Path,
    request: &Path,
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v16::load_debug_simulation_input_v16(kir_v16, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v16, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V16 input admission requires Linux".to_owned(),
        })
    }
}

/// Admits bounded, already captured diagnostic V16 and strict request bytes.
///
/// Unlike the path route this does not authenticate a filesystem capture. Its
/// canonical ledger accounts borrowed slice extents, not unknown caller-owned
/// allocation capacities. It otherwise uses the same admission and CPU limits.
pub fn load_debug_simulation_input_bytes_v16(
    kir_v16: &[u8],
    request: &[u8],
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v16::load_debug_simulation_input_bytes_v16(kir_v16, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v16, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V16 byte admission requires Linux".to_owned(),
        })
    }
}

/// Admits exact diagnostic V18 plus a strictly parsed request on the original ledger.
///
/// The returned CPU view receipt is unreserved: reserve it immediately on this
/// same budget and retain that charge until the whole input (including any
/// debugger backend that consumes its module) is dropped. The plain input type
/// does not enforce this lifetime. Admission errors and unwind restore the entry
/// storage floor while retaining work, peak storage and first-denial history.
/// File capture and request/execution allocations have separate existing bounds;
/// this receipt is not a combined request, allocator, or RSS guarantee.
///
/// Layout tables and exact V18 identity are preserved. This is diagnostic CPU
/// input, not source custody, a compiler route, proof, launch or GPU authority.
pub fn load_debug_simulation_input_v18(
    kir_v18: &Path,
    request: &Path,
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
) -> Result<
    (
        AdmittedSimulationInputV1,
        fe2o3_kir_sim::SimulationViewStorageV18,
    ),
    SimulationInputErrorV1,
> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v18::load_debug_simulation_input_v18(kir_v18, request, budget)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v18, request, budget);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V18 input admission requires Linux".to_owned(),
        })
    }
}

/// Admits captured V18/request bytes with the same receipt and lifetime contract
/// as [load_debug_simulation_input_v18]. Borrowed extents are charged, not unknown
/// caller allocation capacities; this route makes no filesystem-capture claim.
pub fn load_debug_simulation_input_bytes_v18(
    kir_v18: &[u8],
    request: &[u8],
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
) -> Result<
    (
        AdmittedSimulationInputV1,
        fe2o3_kir_sim::SimulationViewStorageV18,
    ),
    SimulationInputErrorV1,
> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v18::load_debug_simulation_input_bytes_v18(kir_v18, request, budget)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v18, request, budget);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V18 byte admission requires Linux".to_owned(),
        })
    }
}

/// Securely admits exact canonical V19 bytes and a strict simulation request.
///
/// This diagnostic CPU route authenticates neither source nor hardware and
/// grants no compiler, artifact, launch, source-map or persisted-schedule
/// authority. Canonical admission uses a cumulative work/storage ledger;
/// simulator admission separately checks resident storage after bounded decode,
/// not as a process-wide pre-allocation or RSS guarantee.
pub fn load_debug_simulation_input_v19(
    kir_v19: &Path,
    request: &Path,
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v19::load_debug_simulation_input_v19(kir_v19, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v19, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V19 input admission requires Linux".to_owned(),
        })
    }
}

/// Admits bounded, already captured diagnostic V19 and strict request bytes.
///
/// Unlike the path route this does not authenticate a filesystem capture. Its
/// canonical ledger accounts borrowed slice extents, not unknown caller-owned
/// allocation capacities. It otherwise uses the same admission and CPU limits.
pub fn load_debug_simulation_input_bytes_v19(
    kir_v19: &[u8],
    request: &[u8],
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v19::load_debug_simulation_input_bytes_v19(kir_v19, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v19, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V19 byte admission requires Linux".to_owned(),
        })
    }
}

/// Securely admits exact canonical V17 bytes and a strict simulation request.
///
/// This diagnostic CPU route authenticates neither source nor hardware and
/// grants no compiler, artifact, launch, source-map or persisted-schedule
/// authority. Canonical admission uses a cumulative work/storage ledger;
/// simulator admission separately checks resident storage after bounded decode,
/// not as a process-wide pre-allocation or RSS guarantee.
pub fn load_debug_simulation_input_v17(
    kir_v17: &Path,
    request: &Path,
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v17::load_debug_simulation_input_v17(kir_v17, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v17, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V17 input admission requires Linux".to_owned(),
        })
    }
}

/// Admits bounded, already captured diagnostic V17 and strict request bytes.
///
/// Unlike the path route this does not authenticate a filesystem capture. Its
/// canonical ledger accounts borrowed slice extents, not unknown caller-owned
/// allocation capacities. It otherwise uses the same admission and CPU limits.
pub fn load_debug_simulation_input_bytes_v17(
    kir_v17: &[u8],
    request: &[u8],
) -> Result<AdmittedSimulationInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::diagnostic_kir_v17::load_debug_simulation_input_bytes_v17(kir_v17, request)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (kir_v17, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "diagnostic V17 byte admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and strictly admits one complete simulation bundle plus
/// its separately bounded request. The exact embedded KIR V7 and target are
/// used directly; this path never lowers, compiles, launches, or falls back.
pub fn load_debug_simulation_bundle_v1(
    bundle: &Path,
    request: &Path,
) -> Result<AdmittedSimulationBundleInputV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_bundle_v1(
            bundle.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bundle, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation bundle admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and strictly admits the separate simulation bundle V2 envelope.
pub fn load_debug_simulation_bundle_v2(
    bundle: &Path,
    request: &Path,
) -> Result<AdmittedSimulationBundleInputV2, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_bundle_v2(
            bundle.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bundle, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation bundle V2 admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and admits a V3 semantic/layout simulation bundle.
pub fn load_debug_simulation_bundle_v3(
    bundle: &Path,
    request: &Path,
) -> Result<AdmittedSimulationBundleInputV3, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_bundle_v3(
            bundle.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bundle, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation bundle V3 admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and admits a V4 aggregate-component simulation bundle.
pub fn load_debug_simulation_bundle_v4(
    bundle: &Path,
    request: &Path,
) -> Result<AdmittedSimulationBundleInputV4, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_bundle_v4(
            bundle.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bundle, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation bundle V4 admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and admits a self-contained V5 bundle through its exact
/// canonical KIR V10 body. This never compiles, launches, or falls back to hardware.
pub fn load_debug_simulation_bundle_v5(
    bundle: &Path,
    request: &Path,
) -> Result<AdmittedSimulationBundleInputV5, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_bundle_v5(
            bundle.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bundle, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation bundle V5 admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and admits a self-contained V6 bundle through its exact
/// canonical KIR V11 body. This never compiles, launches, or falls back to hardware.
pub fn load_debug_simulation_bundle_v6(
    bundle: &Path,
    request: &Path,
) -> Result<AdmittedSimulationBundleInputV6, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_bundle_v6(
            bundle.as_os_str().to_owned(),
            request.as_os_str().to_owned(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bundle, request);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger simulation bundle V6 admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures and strictly admits a canonical persisted semantic CPU
/// schedule, then binds it to an already admitted simulator input before any
/// replay execution begins.
pub fn load_debug_simulation_schedule_v1(
    path: &Path,
    input: &AdmittedSimulationInputV1,
) -> Result<PersistedSimulationScheduleDocumentV1, SimulationInputErrorV1> {
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_simulation_schedule_v1(path.as_os_str().to_owned(), input)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, input);
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "persisted simulation schedule admission requires Linux".to_owned(),
        })
    }
}

/// Securely captures an inert, regular-file debugger sidecar with a caller
/// supplied hard bound. The returned bytes have no pathname authority.
pub fn load_debug_sidecar_v1(
    path: &Path,
    maximum: usize,
) -> Result<Vec<u8>, SimulationInputErrorV1> {
    if maximum == 0 || maximum > 4 * 1024 * 1024 {
        return Err(SimulationInputErrorV1 {
            stage: "arguments".to_owned(),
            code: "invalid_command_line".to_owned(),
            message: "debug sidecar bound must be between 1 byte and 4 MiB".to_owned(),
        });
    }
    #[cfg(target_os = "linux")]
    {
        linux::load_debug_sidecar_v1(path.as_os_str().to_owned(), maximum)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err(SimulationInputErrorV1 {
            stage: "platform".to_owned(),
            code: "unsupported_platform".to_owned(),
            message: "fe2o3 debugger sidecar admission requires Linux".to_owned(),
        })
    }
}

#[cfg(target_os = "linux")]
pub use linux::physical_entry_debug_v20::{
    MAX_PHYSICAL_DEBUG_KIR_BYTES_V20, MAX_PHYSICAL_DEBUG_REQUEST_BYTES_V20,
    PhysicalEntryDebugInputErrorV20, PhysicalEntryDebugInputStorageV20, PhysicalEntryDebugInputV20,
    load_physical_entry_debug_input_v20,
};

#[cfg(target_os = "linux")]
pub use linux::physical_global_copy_debug_v21::{
    MAX_PHYSICAL_GLOBAL_COPY_DEBUG_KIR_BYTES_V21, MAX_PHYSICAL_GLOBAL_COPY_DEBUG_REQUEST_BYTES_V21,
    PhysicalGlobalCopyDebugInputErrorV21, PhysicalGlobalCopyDebugInputStorageV21,
    PhysicalGlobalCopyDebugInputV21, load_physical_global_copy_debug_input_v21,
};

#[cfg(target_os = "linux")]
pub use linux::physical_lds_exchange_debug_v22::{
    MAX_PHYSICAL_LDS_EXCHANGE_DEBUG_KIR_BYTES_V22,
    MAX_PHYSICAL_LDS_EXCHANGE_DEBUG_REQUEST_BYTES_V22, PhysicalLdsExchangeDebugInputErrorV22,
    PhysicalLdsExchangeDebugInputStorageV22, PhysicalLdsExchangeDebugInputV22,
    load_physical_lds_exchange_debug_input_v22,
};

#[cfg(test)]
mod persisted_schedule_identity_tests;

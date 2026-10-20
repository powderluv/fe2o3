//! Single admission boundary for protected rustc invocation capabilities.

use std::env;
use std::fmt;
use std::fs;
use std::os::fd::RawFd;
use std::os::unix::ffi::OsStrExt as _;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
use fe2o3_build_authority::CompilerClosureV2;
use fe2o3_compiler_closure_capability::{
    RUSTC_INVOCATION_CHILD_FD_V1, RustcInvocationCapabilityV1,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use fe2o3_process_identity::{
    CODEGEN_BACKEND_BUILD_OBSERVATION_ENV_V2, COMPILER_IMAGE_MEASUREMENT_STORAGE_V1,
    CompilerImageMeasurementErrorV1 as ImageError, CompilerImageRoleV1 as ImageRole,
    EXPECTED_COMPILER_CLOSURE_SHA256_ENV_V1, ProtectedRustcProcessValidationErrorV1,
    measure_compiler_image_sha256_v1, validate_protected_rustc_process_observation_v1,
};
use fe2o3_rustc_invocation::{CompileEnvironmentV2, RustcInvocationDescriptorV3};

#[path = "protected_reference_invocation_identity_v1.rs"]
pub(crate) mod reference_enrollment_identity;
pub(crate) use reference_enrollment_identity::Stamp as ReferenceEnrollmentInvocationStampV1;

#[cfg(test)]
const BASELINE_PROTECTED_TARGET_V1: &str = fe2o3_amd_target::PRODUCTION_GFX942_DEVICE_TARGET_V1;
const QUALIFICATION_CODEGEN_BACKEND_SHA256_ENV_V1: &str =
    "FE2O3_QUALIFICATION_CODEGEN_BACKEND_SHA256_V1";
const RUNNING_RUSTC_PATH: &str = "/proc/self/exe";

/// One retained V3 descriptor that exactly matched this rustc process.
pub(crate) struct AdmittedProtectedRustcInvocationV1 {
    capability: RustcInvocationCapabilityV1,
    proof_runtime: CompilerProofRuntimeCustody,
    reference_enrollment: Option<reference_enrollment_identity::Owner>,
}

enum CompilerProofRuntimeCustody {
    Unselected,
    Refused,
    NativeV3,
    LegacyV1(Arc<fe2o3_verifier::FunctionalRefinementVerusRuntimeLeaseV1>),
}

impl CompilerProofRuntimeCustody {
    fn select_legacy(
        &mut self,
        admit: impl FnOnce() -> Result<
            Arc<fe2o3_verifier::FunctionalRefinementVerusRuntimeLeaseV1>,
            ProtectedRustcInvocationErrorV1,
        >,
    ) -> Result<(), ProtectedRustcInvocationErrorV1> {
        if !matches!(self, Self::Unselected) {
            return Err(ProtectedRustcInvocationErrorV1::ProofRuntime(
                "compiler proof family was already selected or refused".to_owned(),
            ));
        }
        *self = Self::Refused;
        *self = Self::LegacyV1(admit()?);
        Ok(())
    }

    fn select_native(&mut self) -> Result<(), ProtectedRustcInvocationErrorV1> {
        if !matches!(self, Self::Unselected) {
            return Err(ProtectedRustcInvocationErrorV1::ProofRuntime(
                "compiler proof family was already selected or refused".to_owned(),
            ));
        }
        *self = Self::NativeV3;
        Ok(())
    }

    fn runtime(
        &self,
    ) -> Result<
        Option<&Arc<fe2o3_verifier::FunctionalRefinementVerusRuntimeLeaseV1>>,
        ProtectedRustcInvocationErrorV1,
    > {
        match self {
            Self::LegacyV1(runtime) => Ok(Some(runtime)),
            Self::NativeV3 => Ok(None),
            Self::Unselected | Self::Refused => Err(ProtectedRustcInvocationErrorV1::ProofRuntime(
                "compiler proof family is unselected or refused".to_owned(),
            )),
        }
    }

    fn revalidate(&self) -> Result<(), ProtectedRustcInvocationErrorV1> {
        match self {
            Self::LegacyV1(runtime) => runtime
                .revalidate()
                .map_err(|error| ProtectedRustcInvocationErrorV1::ProofRuntime(error.to_string())),
            Self::Refused => Err(ProtectedRustcInvocationErrorV1::ProofRuntime(
                "compiler proof admission was refused".to_owned(),
            )),
            Self::Unselected | Self::NativeV3 => Ok(()),
        }
    }
}

impl AdmittedProtectedRustcInvocationV1 {
    pub(crate) const REFERENCE_ENROLLMENT_HASH_SCRATCH: usize =
        std::mem::size_of::<sha2::Sha256>() + 32;

    pub(crate) fn reference_enrollment_requested(
        &self,
        budget: &mut Budget<'_>,
    ) -> Result<bool, crate::reference_effect_v1::ReferenceBindingErrorV1> {
        use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
        for entry in self.descriptor().compile_environment().entries() {
            budget
                .charge_work(entry.key().len().saturating_add(1))
                .map_err(|error| Error::new(error.to_string()))?;
            if entry.key() == crate::reference_enrollment_policy_v1::ENV {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn retain_reference_enrollment_identity(
        &mut self,
        budget: &mut Budget<'_>,
    ) -> Result<
        ReferenceEnrollmentInvocationStampV1,
        crate::reference_effect_v1::ReferenceBindingErrorV1,
    > {
        use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
        self.revalidate_for_publication_with_image_budget(budget)
            .map_err(|error| Error::new(error.to_string()))?;
        if self.reference_enrollment.is_none() {
            self.reference_enrollment = Some(
                reference_enrollment_identity::Owner::new(budget)
                    .map_err(|error| Error::new(error.to_string()))?,
            );
        }
        self.reference_enrollment_stamp()
    }

    pub(crate) fn reference_enrollment_stamp(
        &self,
    ) -> Result<
        ReferenceEnrollmentInvocationStampV1,
        crate::reference_effect_v1::ReferenceBindingErrorV1,
    > {
        self.reference_enrollment
            .as_ref()
            .map(|owner| owner.stamp())
            .ok_or_else(|| {
                crate::reference_effect_v1::ReferenceBindingErrorV1::new(
                    "reference enrollment invocation identity was not retained",
                )
            })
    }

    pub(crate) fn reference_enrollment_invocation_sha256(
        &self,
        budget: &mut Budget<'_>,
    ) -> Result<[u8; 32], crate::reference_effect_v1::ReferenceBindingErrorV1> {
        use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
        use sha2::{Digest as _, Sha256};
        let bytes = self.capability.canonical_bytes();
        budget
            .with_prepaid_scope(
                0,
                1,
                bytes.len().saturating_add(1),
                Self::REFERENCE_ENROLLMENT_HASH_SCRATCH,
                |_| Ok::<_, Resource>(Sha256::digest(bytes).into()),
            )
            .map_err(|error| Error::new(error.to_string()))
    }

    pub(crate) fn revalidate_reference_enrollment_identity(
        &self,
        stamp: &ReferenceEnrollmentInvocationStampV1,
        budget: &mut Budget<'_>,
    ) -> Result<(), crate::reference_effect_v1::ReferenceBindingErrorV1> {
        use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
        budget
            .charge_work(1)
            .map_err(|error| Error::new(error.to_string()))?;
        if !self
            .reference_enrollment
            .as_ref()
            .is_some_and(|owner| owner.matches(stamp))
        {
            return Err(Error::new(
                "reference enrollment original invocation changed",
            ));
        }
        self.revalidate_for_publication_with_image_budget(budget)
            .map_err(|error| Error::new(error.to_string()))
    }

    pub(crate) fn select_legacy_proof_runtime(
        &mut self,
    ) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.revalidate_for_publication()?;
        let descriptor = self.capability.descriptor();
        self.proof_runtime.select_legacy(|| {
            let runtime = fe2o3_verifier::FunctionalRefinementVerusRuntimeLeaseV1::open(
                fe2o3_verifier::PROTECTED_FUNCTIONAL_REFINEMENT_RUNTIME_ROOT_V1,
            )
            .map_err(|error| ProtectedRustcInvocationErrorV1::ProofRuntime(error.to_string()))?;
            // SAFETY: only the authenticated legacy dispatch calls this private one-shot
            // boundary. It owns the validated original invocation and consumes its proof slots.
            let runtime = unsafe {
                fe2o3_verifier::admit_inherited_compiler_proof_runtime_v1(
                    descriptor,
                    runtime,
                    Instant::now() + Duration::from_secs(30),
                )
            }
            .map_err(|error| ProtectedRustcInvocationErrorV1::ProofRuntime(error.to_string()))?;
            Ok(Arc::new(runtime))
        })
    }

    pub(crate) fn select_native_proof_runtime(
        &mut self,
    ) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.proof_runtime.select_native()
    }

    pub(crate) fn proof_runtime(
        &self,
    ) -> Result<
        Option<&Arc<fe2o3_verifier::FunctionalRefinementVerusRuntimeLeaseV1>>,
        ProtectedRustcInvocationErrorV1,
    > {
        self.proof_runtime.runtime()
    }

    /// Borrows the exact canonical descriptor retained by the sealed invocation image.
    pub(crate) fn descriptor(&self) -> &RustcInvocationDescriptorV3 {
        self.capability.descriptor()
    }

    /// Revalidates the retained image and returns its complete canonical closure.
    #[cfg(test)]
    pub(crate) fn compiler_closure(&self) -> Result<CompilerClosureV2, String> {
        self.capability.revalidate()?;
        Ok(*self.capability.descriptor().compiler_closure())
    }

    /// Consumes admission immediately before protected publication, repeats
    /// every live-process observation, and retains the sealed V3 capability.
    pub(crate) fn finish_for_publication(
        self,
    ) -> Result<FinishedProtectedRustcInvocationV3, ProtectedRustcInvocationErrorV1> {
        self.revalidate_for_publication()?;
        Ok(FinishedProtectedRustcInvocationV3 {
            capability: self.capability,
            proof_runtime: self.proof_runtime,
            reference_enrollment: self.reference_enrollment,
        })
    }

    /// Revalidates before native publication without detaching the original
    /// invocation from the complete retained source/target preparation owner.
    pub(crate) fn revalidate_for_publication(&self) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.revalidate_with(RustcProcessObservationV1::capture)
    }

    /// Meters both image streams on the original account. Complete live argv,
    /// cwd/environment capture and legacy capability admission remain migration
    /// gates; this method does not claim to bound those other observations.
    pub(crate) fn revalidate_for_publication_with_image_budget(
        &self,
        budget: &mut Budget<'_>,
    ) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.revalidate_with(|descriptor| {
            RustcProcessObservationV1::capture_with_images(descriptor, |path, role| {
                measure_image_with_budget(path, role, budget)
                    .map_err(|e| ProtectedRustcInvocationErrorV1::Observation(e.to_string()))
            })
        })
    }

    fn revalidate_with(
        &self,
        capture: impl FnOnce(
            &RustcInvocationDescriptorV3,
        )
            -> Result<RustcProcessObservationV1, ProtectedRustcInvocationErrorV1>,
    ) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.capability
            .revalidate()
            .map_err(ProtectedRustcInvocationErrorV1::RetainedCapabilityChanged)?;
        let observation = capture(self.capability.descriptor())?;
        validate_retained_capability(&self.capability, observation)?;
        self.proof_runtime.revalidate()
    }

    #[cfg(test)]
    fn finish_for_publication_with_observation(
        self,
        observation: RustcProcessObservationV1,
    ) -> Result<FinishedProtectedRustcInvocationV3, ProtectedRustcInvocationErrorV1> {
        self.capability
            .revalidate()
            .map_err(ProtectedRustcInvocationErrorV1::RetainedCapabilityChanged)?;
        self.finish_after_publication_observation(observation)
    }

    #[cfg(test)]
    fn finish_after_publication_observation(
        self,
        observation: RustcProcessObservationV1,
    ) -> Result<FinishedProtectedRustcInvocationV3, ProtectedRustcInvocationErrorV1> {
        validate_retained_capability(&self.capability, observation)?;
        self.proof_runtime.revalidate()?;
        Ok(FinishedProtectedRustcInvocationV3 {
            capability: self.capability,
            proof_runtime: self.proof_runtime,
            reference_enrollment: self.reference_enrollment,
        })
    }
}

/// Move-only custody of the exact sealed invocation after final live-process
/// remeasurement. It is private compiler authority, not a serializable receipt.
pub(crate) struct FinishedProtectedRustcInvocationV3 {
    capability: RustcInvocationCapabilityV1,
    proof_runtime: CompilerProofRuntimeCustody,
    reference_enrollment: Option<reference_enrollment_identity::Owner>,
}

impl FinishedProtectedRustcInvocationV3 {
    /// Borrows the exact canonical V3 descriptor retained by the sealed image.
    pub(crate) fn descriptor(&self) -> &RustcInvocationDescriptorV3 {
        self.capability.descriptor()
    }

    /// Repeats the complete live-process and immutable-capability admission immediately before a
    /// protected publication transition.
    pub(crate) fn revalidate_for_publication(&self) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.capability
            .revalidate()
            .map_err(ProtectedRustcInvocationErrorV1::RetainedCapabilityChanged)?;
        let observation = RustcProcessObservationV1::capture(self.capability.descriptor())?;
        validate_retained_capability(&self.capability, observation)?;
        self.proof_runtime.revalidate()
    }

    #[cfg(test)]
    fn revalidate_for_publication_with_observation(
        &self,
        observation: RustcProcessObservationV1,
    ) -> Result<(), ProtectedRustcInvocationErrorV1> {
        self.capability
            .revalidate()
            .map_err(ProtectedRustcInvocationErrorV1::RetainedCapabilityChanged)?;
        validate_retained_capability(&self.capability, observation)?;
        self.proof_runtime.revalidate()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ProtectedRustcInvocationErrorV1 {
    Capability(String),
    ProofRuntime(String),
    UnexpectedProtectedSignals {
        descriptor_present: bool,
        compiler_closure_marker_present: bool,
        backend_marker_present: bool,
        qualification_backend_marker_present: bool,
    },
    Observation(String),
    ArgumentsMismatch,
    WorkingDirectoryMismatch,
    CompileEnvironmentMismatch,
    TargetMismatch {
        found: String,
    },
    BackendPathMismatch {
        found: String,
    },
    ArtifactDirectoryPathMismatch {
        found: String,
    },
    RustcClosurePinMismatch,
    CodegenBackendClosurePinMismatch,
    RunningRustcMismatch,
    RunningCodegenBackendMismatch,
    InvalidClosedObservation {
        name: &'static str,
    },
    CompilerClosureObservationMismatch,
    CodegenBackendObservationMismatch,
    RetainedCapabilityChanged(String),
}

impl fmt::Display for ProtectedRustcInvocationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProofRuntime(detail) => write!(
                formatter,
                "protected compiler proof runtime unavailable: {detail}"
            ),
            Self::Capability(detail) => write!(
                formatter,
                "cannot admit canonical fd {RUSTC_INVOCATION_CHILD_FD_V1} as a sealed V3 capability: {detail}"
            ),
            Self::UnexpectedProtectedSignals {
                descriptor_present,
                compiler_closure_marker_present,
                backend_marker_present,
                qualification_backend_marker_present,
            } => write!(
                formatter,
                "rustc invocation signals are forbidden by this admission policy (descriptor: {descriptor_present}, compiler-closure marker: {compiler_closure_marker_present}, protected-backend marker: {backend_marker_present}, qualification-backend marker: {qualification_backend_marker_present})"
            ),
            Self::Observation(detail) => write!(
                formatter,
                "cannot observe the current rustc process: {detail}"
            ),
            Self::ArgumentsMismatch => formatter.write_str(
                "sealed V3 argv, including argv0, differs from the current UTF-8 rustc argv",
            ),
            Self::WorkingDirectoryMismatch => formatter.write_str(
                "sealed V3 working directory differs from the canonical current directory",
            ),
            Self::CompileEnvironmentMismatch => formatter.write_str(
                "sealed V3 compile environment differs from the complete current environment",
            ),
            Self::TargetMismatch { found } => write!(
                formatter,
                "sealed V3 target must be exactly {} or {}, found {found}",
                fe2o3_amd_target::PRODUCTION_GFX942_DEVICE_TARGET_V1,
                fe2o3_amd_target::PRODUCTION_GFX950_DEVICE_TARGET_V1,
            ),
            Self::BackendPathMismatch { found } => write!(
                formatter,
                "sealed V3 backend path must be the retained backend capability {}, found {found}",
                fe2o3_artifact_transaction::BROKERED_CODEGEN_BACKEND_PATH_V1,
            ),
            Self::ArtifactDirectoryPathMismatch { found } => write!(
                formatter,
                "sealed V3 artifact-directory path must be the retained directory capability {}, found {found}",
                fe2o3_artifact_transaction::BROKERED_ARTIFACT_DIRECTORY_PATH_V1,
            ),
            Self::RustcClosurePinMismatch => formatter.write_str(
                "sealed V3 rustc pin differs from the rustc role in its full compiler closure",
            ),
            Self::CodegenBackendClosurePinMismatch => formatter.write_str(
                "sealed V3 backend pin differs from the backend role in its full compiler closure",
            ),
            Self::RunningRustcMismatch => formatter
                .write_str("sealed V3 rustc pin differs from the running /proc/self/exe image"),
            Self::RunningCodegenBackendMismatch => formatter.write_str(
                "sealed V3 backend pin differs from the retained backend image loaded by rustc",
            ),
            Self::InvalidClosedObservation { name } => write!(
                formatter,
                "closed observation {name} is not one canonical lowercase nonzero SHA-256"
            ),
            Self::CompilerClosureObservationMismatch => write!(
                formatter,
                "closed observation {EXPECTED_COMPILER_CLOSURE_SHA256_ENV_V1} differs from the sealed V3 full compiler closure"
            ),
            Self::CodegenBackendObservationMismatch => write!(
                formatter,
                "closed observation {CODEGEN_BACKEND_BUILD_OBSERVATION_ENV_V2} differs from the sealed V3 backend pin"
            ),
            Self::RetainedCapabilityChanged(detail) => {
                write!(
                    formatter,
                    "retained sealed V3 capability changed during admission: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for ProtectedRustcInvocationErrorV1 {}

pub(crate) fn admit_for_production_codegen()
-> Result<Option<AdmittedProtectedRustcInvocationV1>, ProtectedRustcInvocationErrorV1> {
    admit_protected_v3_at(
        RUSTC_INVOCATION_CHILD_FD_V1,
        env::var_os(QUALIFICATION_CODEGEN_BACKEND_SHA256_ENV_V1).is_some(),
    )
}

fn admit_protected_v3_at(
    child_fd: RawFd,
    qualification_backend_marker_present: bool,
) -> Result<Option<AdmittedProtectedRustcInvocationV1>, ProtectedRustcInvocationErrorV1> {
    if qualification_backend_marker_present {
        return Err(
            ProtectedRustcInvocationErrorV1::UnexpectedProtectedSignals {
                descriptor_present: false,
                compiler_closure_marker_present: false,
                backend_marker_present: false,
                qualification_backend_marker_present: true,
            },
        );
    }
    let capability = retain_inherited_capability_at(child_fd)?;
    let observation = RustcProcessObservationV1::capture(capability.descriptor())?;
    validate_capability(capability, observation).map(Some)
}

fn retain_inherited_capability_at(
    child_fd: RawFd,
) -> Result<RustcInvocationCapabilityV1, ProtectedRustcInvocationErrorV1> {
    let admission = RustcInvocationCapabilityV1::from_inherited_at(child_fd);
    // A successful admission retains a private close-on-exec duplicate. Always close the
    // canonical inherited slot so rejected descriptor versions cannot remain for a fallback.
    // SAFETY: close accepts any integer descriptor and reports EBADF for an absent slot.
    let close_result = unsafe { libc::close(child_fd) };

    match admission {
        Ok(capability) if close_result == 0 => Ok(capability),
        Ok(_) => Err(ProtectedRustcInvocationErrorV1::Capability(format!(
            "cannot close consumed inherited descriptor {child_fd}: {}",
            std::io::Error::last_os_error()
        ))),
        Err(detail) => Err(ProtectedRustcInvocationErrorV1::Capability(detail)),
    }
}

struct RustcProcessObservationV1 {
    argv: Vec<String>,
    canonical_working_directory: String,
    compile_environment: CompileEnvironmentV2,
    running_rustc_sha256: [u8; 32],
    running_codegen_backend_sha256: [u8; 32],
}

impl RustcProcessObservationV1 {
    fn capture(
        descriptor: &RustcInvocationDescriptorV3,
    ) -> Result<Self, ProtectedRustcInvocationErrorV1> {
        Self::capture_with_images(descriptor, |path, role| {
            measure_compiler_image_sha256_v1(path, role, |_| Ok::<_, std::convert::Infallible>(()))
                .map_err(|e| ProtectedRustcInvocationErrorV1::Observation(e.to_string()))
        })
    }

    fn capture_with_images(
        descriptor: &RustcInvocationDescriptorV3,
        mut measure: impl FnMut(&Path, ImageRole) -> Result<[u8; 32], ProtectedRustcInvocationErrorV1>,
    ) -> Result<Self, ProtectedRustcInvocationErrorV1> {
        let argv = env::args_os()
            .enumerate()
            .map(|(index, value)| {
                value.into_string().map_err(|_| {
                    ProtectedRustcInvocationErrorV1::Observation(format!(
                        "argv[{index}] is not UTF-8"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let current_directory = env::current_dir().map_err(|error| {
            ProtectedRustcInvocationErrorV1::Observation(format!(
                "cannot read current directory: {error}"
            ))
        })?;
        let canonical_working_directory =
            fs::canonicalize(&current_directory).map_err(|error| {
                ProtectedRustcInvocationErrorV1::Observation(format!(
                    "cannot canonicalize current directory {}: {error}",
                    current_directory.display()
                ))
            })?;
        let canonical_working_directory = canonical_working_directory
            .into_os_string()
            .into_string()
            .map_err(|_| {
                ProtectedRustcInvocationErrorV1::Observation(
                    "canonical current directory is not UTF-8".to_owned(),
                )
            })?;
        let compile_environment = CompileEnvironmentV2::capture_current().map_err(|error| {
            ProtectedRustcInvocationErrorV1::Observation(format!(
                "cannot capture complete compile environment: {error}"
            ))
        })?;
        let running_rustc_sha256 = measure(Path::new(RUNNING_RUSTC_PATH), ImageRole::Executable)?;
        let running_codegen_backend_sha256 = measure(
            Path::new(descriptor.codegen_backend_path()),
            ImageRole::CodegenBackend,
        )?;
        Ok(Self {
            argv,
            canonical_working_directory,
            compile_environment,
            running_rustc_sha256,
            running_codegen_backend_sha256,
        })
    }
}

fn validate_capability(
    capability: RustcInvocationCapabilityV1,
    observation: RustcProcessObservationV1,
) -> Result<AdmittedProtectedRustcInvocationV1, ProtectedRustcInvocationErrorV1> {
    validate_retained_capability(&capability, observation)?;
    Ok(AdmittedProtectedRustcInvocationV1 {
        capability,
        proof_runtime: CompilerProofRuntimeCustody::Unselected,
        reference_enrollment: None,
    })
}

fn validate_retained_capability(
    capability: &RustcInvocationCapabilityV1,
    observation: RustcProcessObservationV1,
) -> Result<(), ProtectedRustcInvocationErrorV1> {
    let descriptor = capability.descriptor();
    validate_protected_rustc_process_observation_v1(
        descriptor,
        &observation.argv,
        &observation.canonical_working_directory,
        &observation.compile_environment,
        observation.running_rustc_sha256,
        observation.running_codegen_backend_sha256,
        fe2o3_artifact_transaction::BROKERED_CODEGEN_BACKEND_PATH_V1,
        fe2o3_artifact_transaction::BROKERED_ARTIFACT_DIRECTORY_PATH_V1,
    )
    .map_err(map_process_validation_error)?;

    capability
        .revalidate()
        .map_err(ProtectedRustcInvocationErrorV1::RetainedCapabilityChanged)?;
    Ok(())
}

fn map_process_validation_error(
    error: ProtectedRustcProcessValidationErrorV1,
) -> ProtectedRustcInvocationErrorV1 {
    match error {
        ProtectedRustcProcessValidationErrorV1::ArgumentsMismatch => {
            ProtectedRustcInvocationErrorV1::ArgumentsMismatch
        }
        ProtectedRustcProcessValidationErrorV1::WorkingDirectoryMismatch => {
            ProtectedRustcInvocationErrorV1::WorkingDirectoryMismatch
        }
        ProtectedRustcProcessValidationErrorV1::CompileEnvironmentMismatch => {
            ProtectedRustcInvocationErrorV1::CompileEnvironmentMismatch
        }
        ProtectedRustcProcessValidationErrorV1::TargetMismatch { found } => {
            ProtectedRustcInvocationErrorV1::TargetMismatch { found }
        }
        ProtectedRustcProcessValidationErrorV1::BackendPathMismatch { found } => {
            ProtectedRustcInvocationErrorV1::BackendPathMismatch { found }
        }
        ProtectedRustcProcessValidationErrorV1::ArtifactDirectoryPathMismatch { found } => {
            ProtectedRustcInvocationErrorV1::ArtifactDirectoryPathMismatch { found }
        }
        ProtectedRustcProcessValidationErrorV1::RustcClosurePinMismatch => {
            ProtectedRustcInvocationErrorV1::RustcClosurePinMismatch
        }
        ProtectedRustcProcessValidationErrorV1::CodegenBackendClosurePinMismatch => {
            ProtectedRustcInvocationErrorV1::CodegenBackendClosurePinMismatch
        }
        ProtectedRustcProcessValidationErrorV1::RunningRustcMismatch => {
            ProtectedRustcInvocationErrorV1::RunningRustcMismatch
        }
        ProtectedRustcProcessValidationErrorV1::RunningCodegenBackendMismatch => {
            ProtectedRustcInvocationErrorV1::RunningCodegenBackendMismatch
        }
        ProtectedRustcProcessValidationErrorV1::CompilerClosureObservationMismatch => {
            ProtectedRustcInvocationErrorV1::CompilerClosureObservationMismatch
        }
        ProtectedRustcProcessValidationErrorV1::CodegenBackendObservationMismatch => {
            ProtectedRustcInvocationErrorV1::CodegenBackendObservationMismatch
        }
        ProtectedRustcProcessValidationErrorV1::InvalidClosedObservation { name } => {
            ProtectedRustcInvocationErrorV1::InvalidClosedObservation { name }
        }
        other => ProtectedRustcInvocationErrorV1::Observation(other.to_string()),
    }
}

fn measure_image_with_budget(
    path: &Path,
    role: ImageRole,
    budget: &mut Budget<'_>,
) -> Result<[u8; 32], ImageError<Resource>> {
    budget.with_prepaid_scope(
        path.as_os_str().as_bytes().len(),
        8,
        8,
        COMPILER_IMAGE_MEASUREMENT_STORAGE_V1 + std::mem::size_of::<ImageError<Resource>>(),
        |b| measure_compiler_image_sha256_v1(path, role, |work| b.charge_work(work)),
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "protected_reference_publication_owner_tests.rs"]
mod reference_publication_owner_tests;

#[cfg(test)]
#[path = "protected_rustc_invocation/image_budget_tests.rs"]
mod image_budget_tests;

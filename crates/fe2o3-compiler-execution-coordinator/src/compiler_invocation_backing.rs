//! Retained compiler inventory and exact invocation inputs for an owning launch.
//!
//! There is no execution entrypoint or substitute approval constructor here.
//! The runtime keeps its fixed-origin checks and original Budget association.
//! Cwd remains untranslated descriptor text; an owning launch still needs an
//! independently established directory-object mapping and dynamic-loader path.
//! Backend and proc-macro Files are inert load inputs, not loaded-library proof.

use crate::compiler_invocation_staging::{
    RustcInvocationStagingErrorV1 as StagingError, StagedRustcInvocationV1 as Invocation,
};
use crate::compiler_output_directory::{CompilerOutputDirectory as Output, Error as OutputError};
use fe2o3_build_authority::COMPILER_RUNTIME_MANIFEST_MAX_ENTRIES_V1 as MAX_RUNTIME_ENTRIES;
use fe2o3_compiler_closure_capability::{
    CompilerExecutionCapabilityErrorV2 as CaptureError,
    RetainedCompilerRuntimeErrorV1 as RuntimeError,
    RetainedCompilerRuntimeInventoryTransferV1 as Sources, RetainedCompilerRuntimeV1 as Runtime,
    RustcInvocationCapabilityV1 as Capture,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use fe2o3_protected_service_spawn::{
    ProtectedServiceDescriptorBindingV1 as Binding,
    native_spawn::StagedProtectedServiceExecV2 as Stage,
};
use fe2o3_rustc_invocation::{
    MAX_COMPILE_ENVIRONMENT_ENTRIES_V2, MAX_RUSTC_ARGUMENTS_V2,
    RustcInvocationDescriptorV3 as Descriptor,
};
use std::{fmt, fs::File, mem::size_of};

const ENTRY: usize = 8;
const MEASURE_WORK: usize = ENTRY + 32;
type Result<T> = std::result::Result<T, CompilerInvocationBackingError>;

#[path = "compiler_invocation_enrollment.rs"]
mod enrollment;
pub(crate) use enrollment::OriginalInvocationEnrollment;

#[derive(Debug)]
pub(crate) enum CompilerInvocationBackingError {
    Resource(Resource),
    Capture(CaptureError),
    Runtime(RuntimeError),
    Staging(StagingError),
    Output(OutputError),
    Enrollment(fe2o3_rustc_invocation::ReferenceEnrollmentDecodeErrorV1<Resource>),
    Digest(fe2o3_rustc_invocation::DigestError),
    InvalidCount,
}

impl From<Resource> for CompilerInvocationBackingError {
    fn from(error: Resource) -> Self {
        Self::Resource(error)
    }
}

impl From<RuntimeError> for CompilerInvocationBackingError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<CaptureError> for CompilerInvocationBackingError {
    fn from(error: CaptureError) -> Self {
        Self::Capture(error)
    }
}

impl From<StagingError> for CompilerInvocationBackingError {
    fn from(error: StagingError) -> Self {
        Self::Staging(error)
    }
}
impl From<OutputError> for CompilerInvocationBackingError {
    fn from(error: OutputError) -> Self {
        Self::Output(error)
    }
}

impl fmt::Display for CompilerInvocationBackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => error.fmt(f),
            Self::Capture(error) => error.fmt(f),
            Self::Runtime(error) => error.fmt(f),
            Self::Staging(error) => error.fmt(f),
            Self::Output(error) => error.fmt(f),
            Self::Enrollment(error) => error.fmt(f),
            Self::Digest(error) => error.fmt(f),
            Self::InvalidCount => f.write_str("compiler invocation exceeds count bound"),
        }
    }
}

impl std::error::Error for CompilerInvocationBackingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resource(error) => Some(error),
            Self::Capture(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Staging(error) => Some(error),
            Self::Output(error) => Some(error),
            Self::Enrollment(error) => Some(error),
            Self::Digest(error) => Some(error),
            Self::InvalidCount => None,
        }
    }
}

/// Unreserved GROWTH over the consumed runtime, sealed capture and output owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompilerInvocationBackingCharge {
    additional: usize,
    retained: usize,
}

impl CompilerInvocationBackingCharge {
    pub(crate) const fn additional_storage(self) -> usize {
        self.additional
    }

    pub(crate) const fn retained_storage(self) -> usize {
        self.retained
    }
}

/// Move-only custody of one genuine inventory, exact inputs and its complete source set.
///
/// An owning launch can borrow these inputs and ask for contextual revalidation.
/// It cannot detach the runtime, select another source or convert this backing
/// into process authority. File accessors require exclusive caller custody of
/// flags and content through staging; every final duplicate needs validation.
pub(crate) struct CompilerInvocationBacking {
    runtime: Runtime,
    capture: Capture,
    output: Output,
    invocation: Invocation,
    sources: Sources,
    retained: usize,
}

impl CompilerInvocationBacking {
    const ENVELOPE: usize = size_of::<(Self, CompilerInvocationBackingCharge)>()
        - size_of::<Runtime>()
        - size_of::<Capture>()
        - size_of::<Output>()
        - size_of::<Invocation>()
        - size_of::<Sources>();
    /// Scalar bookkeeping and bounded retirement of inventory AND transfer set
    /// on consuming error, plus the sealed capture and staged invocation strings.
    /// Measurement, native revalidation, runtime and staging operations charge
    /// their own work on the same Budget. This is not a complete launch work quota.
    pub(crate) const LOCAL_WORK: usize = ENTRY
        + (2 * MAX_RUNTIME_ENTRIES + 32) * 1088
        + 8 * (2 * MAX_RUSTC_ARGUMENTS_V2 + 3 * MAX_COMPILE_ENVIRONMENT_ENTRIES_V2 + 2);
    /// Local frames only; nested scratch and all overlapping backing are separate.
    pub(crate) const FRAME_STORAGE: usize =
        4 * size_of::<(Self, CompilerInvocationBackingCharge)>()
            + 8 * size_of::<CompilerInvocationBackingError>()
            + 4096;

    /// Consumes the original runtime and received sealed invocation capability.
    /// All three FULL source reservations, including the received output owner,
    /// must remain prepaid on the runtime's original Budget. The invocation's
    /// native owner includes its original FD reservation plus admission growth;
    /// decoded descriptor storage alone is insufficient. Preserve those source
    /// reservations and reserve returned GROWTH before
    /// retaining the result. On failure the consumed inputs drop, but caller-owned
    /// reservations remain unchanged and must be retired by the caller. Work and
    /// denial history are never refunded. No capability or descriptor is cloned,
    /// decoded or re-encoded here. Sealed custody does not prove cargo authorship.
    pub(crate) fn prepare(
        runtime: Runtime,
        capture: Capture,
        output: Output,
        b: &mut Budget<'_>,
    ) -> Result<(Self, CompilerInvocationBackingCharge)> {
        // Fund bounded local retirement before fallible source measurement.
        b.charge_work(Self::LOCAL_WORK)?;
        let input = measure_inputs(
            runtime
                .required_retained_storage()
                .checked_add(Output::STORAGE)
                .ok_or(Resource::Arithmetic)?,
            &capture,
            b,
        )?;
        b.with_prepaid_scope(input, 0, 0, Self::FRAME_STORAGE, |b| {
            capture.revalidate_native(b)?;
            let descriptor = capture.descriptor();
            // This checks the complete closure and the runtime's own account,
            // approval and fixed origins before any new backing is retained.
            runtime.require_compiler(*descriptor.compiler_closure(), b)?;
            output.revalidate(descriptor.artifact_output_directory(), b)?;
            let (invocation, invocation_charge) = Invocation::stage(descriptor, b)?;
            b.reserve_storage(invocation_charge.additional_storage())?;
            let (sources, sources_charge) = runtime.try_clone_inventory_for_staging(b)?;
            b.reserve_storage(sources_charge.full_storage())?;
            let charge = retained_storage_for(
                input,
                invocation.retained_storage(),
                sources_charge.full_storage(),
            )?;
            b.reserve_storage(Self::ENVELOPE)?;
            let owner = Self {
                runtime,
                capture,
                output,
                invocation,
                sources,
                retained: charge.retained_storage(),
            };
            owner.check(b)?;
            Ok((owner, charge))
        })
    }

    /// Borrows the original runtime for private launch composition, not public
    /// transport. Keep this complete owner reserved and use its original Budget
    /// for runtime operations; this borrow does not detach approval or inventory.
    pub(crate) const fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Exact retained coordination data, not evidence of cargo authorship or exec.
    pub(crate) const fn descriptor(&self) -> &Descriptor {
        self.capture.descriptor()
    }

    /// Exact inert C strings; the owning native stage funds its pointer tables.
    pub(crate) const fn invocation(&self) -> &Invocation {
        &self.invocation
    }

    /// Borrow only the fixed rustc transfer for the owning native stage.
    pub(crate) fn rustc_source(&self) -> &File {
        self.sources.rustc_source()
    }

    /// Borrow only the fixed interpreter transfer for the owning native stage.
    pub(crate) fn elf_interpreter_source(&self) -> &File {
        self.sources.elf_interpreter_source()
    }

    /// Fixed approved backend input, not proof of the descriptor's load binding.
    /// A descriptor naming `/proc/./self/fd/198` requires the owning native launch
    /// to bind this source to child FD 198, preserving the exact selector. Merely
    /// retaining this File or substituting a pathname/environment is insufficient.
    pub(crate) fn codegen_backend_source(&self) -> &File {
        self.sources.codegen_backend_source()
    }

    /// Fixed approved fe2o3 proc-macro input. The owning launch must independently
    /// establish its exact rustc input binding and ELF dependency resolution.
    pub(crate) fn fe2o3_proc_macro_source(&self) -> &File {
        self.sources.fe2o3_proc_macro_source()
    }

    /// Complete borrowed files for the future owning filesystem-view stage.
    /// Includes shared libraries but does not establish their loader mappings.
    pub(crate) const fn inventory_sources(&self) -> &Sources {
        &self.sources
    }

    pub(crate) const fn retained_storage(&self) -> usize {
        self.retained
    }

    /// Exact received output at 197. This binding is not a namespace or runtime
    /// guard. The caller retains this whole owner through staging and cleanup.
    pub(crate) fn output_binding(&self, b: &mut Budget<'_>) -> Result<Binding<'_>> {
        self.revalidate(b)?;
        Ok(self.output.binding(b)?)
    }

    /// Full additional backing for all four image sources and the output FD.
    /// Reserve it while those duplicates coexist with this complete owner. Native
    /// stage structures, other bindings and pointer tables require separate charge.
    pub(crate) fn staged_sources_storage(&self) -> Result<usize> {
        self.sources
            .compiler_staging_storage()
            .checked_add(crate::native_launch::FILE_STORAGE)
            .ok_or_else(|| Resource::Arithmetic.into())
    }

    /// Recheck the original sealed capture, full closure, account and inventory.
    pub(crate) fn revalidate(&self, b: &mut Budget<'_>) -> Result<()> {
        b.with_prepaid_scope(
            self.retained,
            ENTRY,
            Self::LOCAL_WORK,
            Self::FRAME_STORAGE,
            |b| self.check(b),
        )
    }

    /// Validate the actual final Files of the owning native stage against the
    /// original runtime. All four copies require FULL additional charges while
    /// this complete owner remains reserved. The existing runtime validators
    /// reject wrong roles, different inodes and changed origins even if bytes or
    /// pathname claims match. This never imports these Files as approved sources
    /// or proves the image FD/input bindings, loading or ELF resolution. Output
    /// validation additionally reads the exact 197 entry from the actual Stage.
    pub(crate) fn validate_staged_sources(
        &self,
        stage: &Stage,
        rustc: &File,
        interpreter: &File,
        codegen_backend: &File,
        fe2o3_proc_macro: &File,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        // The image inputs are the actual Stage's borrowed files. Preserve the
        // original full-source floor even if an unsafe staging caller underquoted.
        let stage_floor = stage.retained_storage().max(self.staged_sources_storage()?);
        let floor = staged_floor(self.retained, stage_floor)?;
        b.with_prepaid_scope(floor, ENTRY, Self::LOCAL_WORK, Self::FRAME_STORAGE, |b| {
            self.check(b)?;
            self.output.validate_staged(stage, b)?;
            self.runtime.validate_rustc_exec_transfer(rustc, b)?;
            self.runtime
                .validate_elf_interpreter_exec_transfer(interpreter, b)?;
            self.runtime
                .validate_codegen_backend_load_transfer(codegen_backend, b)?;
            self.runtime
                .validate_fe2o3_proc_macro_load_transfer(fe2o3_proc_macro, b)?;
            Ok(())
        })
    }

    fn check(&self, b: &mut Budget<'_>) -> Result<()> {
        self.capture.revalidate_native(b)?;
        self.output
            .revalidate(self.descriptor().artifact_output_directory(), b)?;
        // These existing entrypoints enforce ledger identity AND Budget address.
        // A fresh budget cannot recreate the runtime's account association.
        self.runtime
            .require_compiler(*self.descriptor().compiler_closure(), b)?;
        self.runtime
            .validate_inventory_transfer(self.inventory_sources(), b)?;
        Ok(())
    }
}

// Scalar inputs here are accounting only; they cannot manufacture Backing.
fn measure_inputs(runtime: usize, capture: &Capture, b: &mut Budget<'_>) -> Result<usize> {
    b.charge_work(ENTRY)?;
    let descriptor = capture.descriptor();
    let arguments = descriptor.rustc().argv().len();
    if arguments == 0
        || arguments > MAX_RUSTC_ARGUMENTS_V2
        || descriptor.compile_environment().entries().len() > MAX_COMPILE_ENVIRONMENT_ENTRIES_V2
    {
        return Err(CompilerInvocationBackingError::InvalidCount);
    }
    let input = b.with_prepaid_scope(
        0,
        0,
        MEASURE_WORK - ENTRY,
        CompilerInvocationBacking::FRAME_STORAGE,
        |_| -> Result<usize> {
            capture
                .native_retained_storage()?
                .checked_add(runtime)
                .ok_or_else(|| Resource::Arithmetic.into())
        },
    )?;
    // Check after the measurement scope restores entry storage: its scratch must
    // not mask a missing original invocation FD, decoded owner or runtime.
    if b.storage() < input {
        return Err(Resource::Accounting.into());
    }
    Ok(input)
}

fn staged_floor(retained: usize, transfers: usize) -> Result<usize> {
    retained
        .checked_add(transfers)
        .ok_or_else(|| Resource::Arithmetic.into())
}

fn retained_storage_for(
    input: usize,
    invocation: usize,
    sources: usize,
) -> Result<CompilerInvocationBackingCharge> {
    let additional = sources
        .checked_add(invocation)
        .and_then(|n| n.checked_add(CompilerInvocationBacking::ENVELOPE))
        .ok_or(Resource::Arithmetic)?;
    Ok(CompilerInvocationBackingCharge {
        additional,
        retained: staged_floor(input, additional)?,
    })
}

#[cfg(test)]
#[path = "compiler_invocation_backing_tests.rs"]
mod tests;

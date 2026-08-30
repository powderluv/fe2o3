//! Single production-pipeline transaction shell.
//!
//! This module owns the one integration point for issue #175. It deliberately
//! contains no workload recognition. The sole semantic-MIR importer owns the
//! consuming target-authentication boundary and moves an admitted request into
//! a typed stage before the mandatory generic kernel-verification pipeline.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;

use rustc_middle::ty::TyCtxt;

use crate::artifact_transaction::{BuildAttempt, ProducerIdentity};
use crate::collector::AuthenticatedCollectedKernelClosureV1;
use crate::protected_compiler_execution::{
    AdmittedProtectedCompilerExecutionV1, ProtectedCompilerExecutionErrorV1,
};
use crate::protected_rustc_invocation::{
    AdmittedProtectedRustcInvocationV1, ProtectedRustcInvocationErrorV1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProductionDisposition {
    HostOnly,
    DeviceTransaction,
}

pub(crate) const fn disposition(device_candidate_count: usize) -> ProductionDisposition {
    if device_candidate_count == 0 {
        ProductionDisposition::HostOnly
    } else {
        ProductionDisposition::DeviceTransaction
    }
}

#[derive(Debug)]
pub(crate) enum ProductionPipelineError {
    CustomLlvmConfiguration,
    EmptyCollectedDeviceClosure,
    SemanticImport(crate::collector::ProductionSemanticImportErrorV1),
    SemanticMiddleEnd(fe2o3_pliron::ProductionSemanticMirErrorV1),
    RankedProjection(crate::production_ranked_projection_v1::ProductionRankedProjectionErrorV1),
    RankedVerification(crate::production_ranked_projection_v1::ProductionRankedVerificationErrorV1),
    MultiRootTargetNeutralLowering {
        roots: usize,
    },
    TargetNeutralLowering(fe2o3_lower_mir_kernel::ProductionSemanticKirErrorV1),
    MissingMirPlironTranslationValidation,
    SimulationKernelIrV7(fe2o3_kernel_ir::VerifiedCanonicalKernelIrErrorV7),
    SimulationBundle(fe2o3_kernel_ir::SimulationBundleErrorV1),
    SimulationDebugMap(fe2o3_kernel_ir::DebugSourceMapErrorV1),
    SimulationBundleV2(fe2o3_kernel_ir::SimulationBundleErrorV2),
    SimulationDebugMapV2(fe2o3_kernel_ir::DebugSourceMapErrorV2),
    SimulationDebugMapCorrespondence(&'static str),
    SimulationSourceLineage(fe2o3_compiler_lineage::LineageErrorV3),
    SimulationProductionKirV9,
    FormalMemoryAdmission(fe2o3_lower_mir_kernel::ProductionFormalMemoryErrorV1),
    Geometry(crate::production_geometry_v1::ProductionGeometryErrorV1),
    TargetBinding(dialect_amdgcn::ProductionTargetBindingErrorV1),
    TargetLowering(dialect_amdgcn::LoweringErrors),
    UpstreamLlvmLayoutBinding(dialect_amdgcn::ProductionLlvmLayoutBindingErrorV1),
    DescriptorEvidence(crate::compiler_descriptor::CompilerDescriptorError),
    SemanticLineage(crate::production_semantic_lineage_v3::ProductionSemanticLineageErrorV3),
    RustcLineageMismatch,
    ProtectedRustcInvocation(ProtectedRustcInvocationErrorV1),
    ProtectedCompilerExecution(ProtectedCompilerExecutionErrorV1),
    ExtractionCannotPublish,
    WorkerHandoffExtractionRequiresExtractionCustody,
    WorkerHandoff(crate::production_worker_handoff::ProductionWorkerHandoffError),
    StrictV3Publication(fe2o3_artifact_transaction::CompilerModuleHandoffErrorV3),
    CompilerExecutionSubject(fe2o3_artifact_transaction::CompilerExecutionSubjectErrorV1),
    CompilerExecutionReceiptTransport(
        fe2o3_artifact_transaction::CompilerExecutionReceiptTransportErrorV1,
    ),
    CompilerExecutionReceiptTransportBindingMismatch,
}

impl fmt::Display for ProductionPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CustomLlvmConfiguration => formatter.write_str(
                "production compilation rejects caller-selected LLVM arguments or passes before transaction construction",
            ),
            Self::EmptyCollectedDeviceClosure => formatter.write_str(
                "production compilation requires a nonempty collector-sealed device closure",
            ),
            Self::SemanticImport(error) => write!(formatter, "production compilation {error}"),
            Self::SemanticMiddleEnd(error) => {
                write!(formatter, "production compilation exact semantic middle end failed: {error}")
            }
            Self::RankedProjection(error) => {
                write!(formatter, "production compilation general kernel verification failed: {error}")
            }
            Self::RankedVerification(error) => {
                write!(formatter, "production compilation ranked verification failed: {error}")
            }
            Self::MultiRootTargetNeutralLowering { roots } => write!(
                formatter,
                "production compilation retained a verified ranked roster with {roots} kernel roots; target-neutral Kernel IR lowering remains fail-closed until it can consume the complete roster"
            ),
            Self::TargetNeutralLowering(error) => {
                write!(formatter, "production compilation target-neutral lowering failed: {error}")
            }
            Self::MissingMirPlironTranslationValidation => formatter.write_str(
                "production compilation reached target-neutral custody without independent MIR-to-PLIRON translation validation",
            ),
            Self::SimulationKernelIrV7(error) => write!(
                formatter,
                "production compilation cannot project the already-lowered module to exact simulation Kernel IR V7: {error}"
            ),
            Self::SimulationBundle(error) => {
                write!(formatter, "production compilation simulation bundle failed: {error}")
            }
            Self::SimulationDebugMap(error) => write!(
                formatter,
                "production compilation simulation debug map failed: {error}"
            ),
            Self::SimulationBundleV2(error) => write!(
                formatter,
                "production compilation simulation bundle V2 failed: {error}"
            ),
            Self::SimulationDebugMapV2(error) => write!(
                formatter,
                "production compilation simulation debug map V2 failed: {error}"
            ),
            Self::SimulationDebugMapCorrespondence(detail) => write!(
                formatter,
                "production compilation simulation debug-map correspondence failed: {detail}"
            ),
            Self::SimulationSourceLineage(error) => write!(
                formatter,
                "production compilation simulation source-lineage receipt failed: {error}"
            ),
            Self::SimulationProductionKirV9 => formatter.write_str(
                "production Kernel IR V9 is not representable by the exact V7 CPU simulator; no downgrade or hardware fallback was attempted",
            ),
            Self::FormalMemoryAdmission(error) => {
                write!(formatter, "production compilation formal memory admission failed: {error}")
            }
            Self::Geometry(error) => {
                write!(formatter, "production compilation geometry validation failed: {error}")
            }
            Self::TargetBinding(error) => {
                write!(formatter, "production compilation AMDGPU target binding failed: {error}")
            }
            Self::TargetLowering(error) => {
                write!(formatter, "production compilation AMDGPU LLVM lowering failed: {error}")
            }
            Self::UpstreamLlvmLayoutBinding(error) => {
                write!(formatter, "production compilation upstream LLVM layout binding failed: {error}")
            }
            Self::DescriptorEvidence(error) => {
                write!(formatter, "production compilation descriptor evidence failed: {error}")
            }
            Self::SemanticLineage(error) => write!(formatter, "production compilation {error}"),
            Self::RustcLineageMismatch => formatter.write_str(
                "production compilation rustc preflight plan is not bound to the retained identity inventory",
            ),
            Self::ProtectedRustcInvocation(error) => write!(
                formatter,
                "production compilation final protected rustc invocation validation failed: {error}"
            ),
            Self::ProtectedCompilerExecution(error) => write!(
                formatter,
                "production compilation protected compiler execution failed: {error}"
            ),
            Self::ExtractionCannotPublish => formatter.write_str(
                "production extraction custody cannot publish a compiler-module handoff",
            ),
            Self::WorkerHandoffExtractionRequiresExtractionCustody => formatter.write_str(
                "inert compiler-module extraction requires extraction-only custody",
            ),
            Self::WorkerHandoff(error) => {
                write!(formatter, "production compilation compiler-module handoff failed: {error}")
            }
            Self::StrictV3Publication(error) => {
                write!(formatter, "production compilation strict V3 publication failed: {error}")
            }
            Self::CompilerExecutionSubject(error) => write!(
                formatter,
                "production compilation compiler-execution subject failed: {error}"
            ),
            Self::CompilerExecutionReceiptTransport(error) => write!(
                formatter,
                "production compilation compiler-execution receipt transport failed: {error}"
            ),
            Self::CompilerExecutionReceiptTransportBindingMismatch => formatter.write_str(
                "production compilation compiler-execution receipt transport changed its exact subject or byte length",
            ),
        }
    }
}

impl std::error::Error for ProductionPipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SemanticImport(error) => Some(error),
            Self::SemanticMiddleEnd(error) => Some(error),
            Self::RankedProjection(error) => Some(error),
            Self::RankedVerification(error) => Some(error),
            Self::TargetNeutralLowering(error) => Some(error),
            Self::SimulationKernelIrV7(error) => Some(error),
            Self::SimulationBundle(error) => Some(error),
            Self::SimulationDebugMap(error) => Some(error),
            Self::SimulationBundleV2(error) => Some(error),
            Self::SimulationDebugMapV2(error) => Some(error),
            Self::SimulationSourceLineage(error) => Some(error),
            Self::FormalMemoryAdmission(error) => Some(error),
            Self::Geometry(error) => Some(error),
            Self::TargetBinding(error) => Some(error),
            Self::TargetLowering(error) => Some(error),
            Self::UpstreamLlvmLayoutBinding(error) => Some(error),
            Self::DescriptorEvidence(error) => Some(error),
            Self::SemanticLineage(error) => Some(error),
            Self::ProtectedRustcInvocation(error) => Some(error),
            Self::ProtectedCompilerExecution(error) => Some(error),
            Self::WorkerHandoff(error) => Some(error),
            Self::StrictV3Publication(error) => Some(error),
            Self::CompilerExecutionSubject(error) => Some(error),
            Self::CompilerExecutionReceiptTransport(error) => Some(error),
            Self::CustomLlvmConfiguration
            | Self::EmptyCollectedDeviceClosure
            | Self::MissingMirPlironTranslationValidation
            | Self::MultiRootTargetNeutralLowering { .. }
            | Self::RustcLineageMismatch
            | Self::SimulationProductionKirV9
            | Self::SimulationDebugMapCorrespondence(_)
            | Self::ExtractionCannotPublish
            | Self::CompilerExecutionReceiptTransportBindingMismatch
            | Self::WorkerHandoffExtractionRequiresExtractionCustody => None,
        }
    }
}

pub(crate) fn reject_custom_llvm_configuration(
    has_custom_llvm_configuration: bool,
) -> Result<(), ProductionPipelineError> {
    if has_custom_llvm_configuration {
        Err(ProductionPipelineError::CustomLlvmConfiguration)
    } else {
        Ok(())
    }
}

pub(super) struct CollectedRustStage<'tcx> {
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    typed_descriptor_roots: Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
    debug_source_capture: crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2,
    transaction: ProductionTransactionBindings,
}

struct ProductionTransactionBindings {
    producer: ProducerIdentity,
    output_dir: PathBuf,
    compiler_ffi_envelope: Option<fe2o3_compiler_ffi::CompilerFfiEnvelopeV1>,
    compiler_custody: ProductionCompilerCustody,
}

enum ProductionCompilerCustody {
    ProtectedV3 {
        invocation: Box<AdmittedProtectedRustcInvocationV1>,
        compiler_execution: Box<AdmittedProtectedCompilerExecutionV1>,
        attempt: BuildAttempt,
    },
    ExtractionOnly,
}

impl ProductionCompilerCustody {
    fn protected(
        invocation: AdmittedProtectedRustcInvocationV1,
        compiler_execution: AdmittedProtectedCompilerExecutionV1,
        attempt: BuildAttempt,
    ) -> Self {
        Self::ProtectedV3 {
            invocation: Box::new(invocation),
            compiler_execution: Box::new(compiler_execution),
            attempt,
        }
    }

    const fn extraction_only() -> Self {
        Self::ExtractionOnly
    }

    fn retained_protected_binding_count(&self) -> usize {
        match self {
            Self::ProtectedV3 { .. } => 2,
            Self::ExtractionOnly => 0,
        }
    }

    fn is_extraction_only(&self) -> bool {
        matches!(self, Self::ExtractionOnly)
    }

    fn into_publication_custody(
        self,
    ) -> Result<ProtectedProductionPublicationCustody, ProductionPipelineError> {
        match self {
            Self::ProtectedV3 {
                invocation,
                compiler_execution,
                attempt,
            } => Ok(ProtectedProductionPublicationCustody {
                attempt,
                invocation,
                compiler_execution,
            }),
            Self::ExtractionOnly => Err(ProductionPipelineError::ExtractionCannotPublish),
        }
    }
}

struct ProtectedProductionPublicationCustody {
    attempt: BuildAttempt,
    invocation: Box<AdmittedProtectedRustcInvocationV1>,
    compiler_execution: Box<AdmittedProtectedCompilerExecutionV1>,
}

struct AuthenticatedProductionBindings {
    rustc_identity_inventory: crate::collector::AuthenticatedRustcIdentityInventoryV3,
    rustc_preflight_plan: crate::collector::AuthenticatedRustcPreflightPlanV3,
    rustc_target: crate::production_target_v1::AuthenticatedProductionTargetV1,
    reference_effect_bindings: crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
    debug_source_files: Box<[fe2o3_kernel_ir::DebugSourceMapFileV1]>,
    debug_source_scopes: Box<[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2]>,
    debug_source_variables: Box<[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2]>,
    typed_descriptor_roots: Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
    transaction: ProductionTransactionBindings,
}

pub(super) struct AdmittedSemanticMirStage {
    semantic_mir: fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1,
    bindings: AuthenticatedProductionBindings,
}

pub(super) struct EquivalentSemanticMirStage {
    semantic_mir: fe2o3_pliron::ProductionSemanticMirOwnerV1,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only owner of one production compilation stage.
///
/// Its fields and stage types stay private so no caller can synthesize or
/// bypass a transition. The transaction carries no artifact, publication,
/// load, launch, or runtime authority.
pub(crate) struct ProductionCompilation<'tcx, Stage> {
    stage: Stage,
    invariant_session: PhantomData<fn(TyCtxt<'tcx>) -> TyCtxt<'tcx>>,
}

/// Move-only production stage retaining rustc identities, transaction
/// bindings, admitted semantic MIR, and the owner-held verified PLIRON graph.
pub(crate) struct RankedVerifiedProductionCompilation {
    ranked: crate::production_ranked_projection_v1::ProductionRankedSemanticProgramV1,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only production stage retaining exact semantic ownership, verified
/// Kernel IR, correspondence evidence, and the original transaction bindings.
pub(crate) struct TargetNeutralProductionCompilation {
    lowered: fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    ranked_verification: crate::production_ranked_projection_v1::AuthenticatedRankedVerificationV5,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only production stage retaining exact semantic ownership, verified
/// Kernel IR, composed formal/ranked memory evidence, and transaction bindings.
pub(crate) struct FormalMemoryAdmittedProductionCompilation {
    admitted: fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
    ranked_verification: crate::production_ranked_projection_v1::AuthenticatedRankedVerificationV5,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only production stage retaining formal admission, exact target-bound
/// Kernel IR, deterministic exact-target LLVM text, and transaction bindings.
pub(crate) struct TargetLoweredProductionCompilation {
    admitted: fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
    ranked_verification: crate::production_ranked_projection_v1::AuthenticatedRankedVerificationV5,
    target_module: fe2o3_kernel_ir::Module,
    llvm_ir: String,
    bindings: AuthenticatedProductionBindings,
}

/// Private handoff input that can only be constructed by the exact production
/// target-lowering stage. It grants no publication or artifact authority.
pub(crate) struct AuthenticatedProductionTargetModule {
    admitted: fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
    target: fe2o3_compiler_ffi::DeviceTargetV1,
    target_module: fe2o3_kernel_ir::Module,
    llvm_ir: String,
    typed_descriptor_roots: Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
    compiler_ffi_envelope: Option<fe2o3_compiler_ffi::CompilerFfiEnvelopeV1>,
}

struct PreparedProductionWorkerPublication {
    producer: ProducerIdentity,
    output_dir: PathBuf,
    attempt: BuildAttempt,
    invocation: Box<AdmittedProtectedRustcInvocationV1>,
    compiler_execution: Box<AdmittedProtectedCompilerExecutionV1>,
    semantic_lineage: crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3,
    rustc_target: crate::production_target_v1::AuthenticatedProductionTargetV1,
    prepared: crate::production_worker_handoff::PreparedProductionWorkerHandoff,
}

impl AuthenticatedProductionTargetModule {
    pub(crate) fn into_parts(
        self,
    ) -> (
        fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
        fe2o3_compiler_ffi::DeviceTargetV1,
        fe2o3_kernel_ir::Module,
        String,
        Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
        Option<fe2o3_compiler_ffi::CompilerFfiEnvelopeV1>,
    ) {
        (
            self.admitted,
            self.target,
            self.target_module,
            self.llvm_ir,
            self.typed_descriptor_roots,
            self.compiler_ffi_envelope,
        )
    }
}

impl TargetNeutralProductionCompilation {
    fn into_prepared_simulation_bundle_v1(
        self,
        compiler_execution_binding: fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1,
    ) -> Result<
        (
            fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
            AuthenticatedProductionBindings,
            fe2o3_kernel_ir::PreparedSimulationBundleV1,
        ),
        ProductionPipelineError,
    > {
        let Self {
            lowered,
            ranked_verification: _,
            bindings,
        } = self;
        lowered
            .verify_equivalence()
            .map_err(ProductionPipelineError::TargetNeutralLowering)?;
        if bindings
            .rustc_preflight_plan
            .rustc_identity_inventory_sha256()
            != bindings.rustc_identity_inventory.sha256()
        {
            return Err(ProductionPipelineError::RustcLineageMismatch);
        }
        let production_identity = lowered.canonical_kernel_ir_identity();
        let production_identity = match production_identity.version() {
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V8 => {
                fe2o3_kernel_ir::SimulationProductionKirIdentityV1::v8(
                    *production_identity.digest(),
                    production_identity.canonical_length(),
                )
                .map_err(ProductionPipelineError::SimulationBundle)?
            }
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V9 => {
                return Err(ProductionPipelineError::SimulationProductionKirV9);
            }
        };
        let canonical_v7 =
            fe2o3_kernel_ir::VerifiedCanonicalKernelIrV7::from_module(lowered.module().clone())
                .map_err(ProductionPipelineError::SimulationKernelIrV7)?;
        let inventory_receipt =
            fe2o3_compiler_lineage::InertRustcIdentityInventoryReceiptV3::from_canonical_preimage(
                bindings.rustc_identity_inventory.canonical_transcript(),
            )
            .map_err(ProductionPipelineError::SimulationSourceLineage)?;
        let preflight_receipt =
            fe2o3_compiler_lineage::InertRustcPreflightPlanReceiptV3::from_canonical_preimage(
                bindings.rustc_preflight_plan.canonical_transcript(),
            )
            .map_err(ProductionPipelineError::SimulationSourceLineage)?;
        let inventory_identity = inventory_receipt.identity();
        let preflight_identity = preflight_receipt.identity();
        let lineage = fe2o3_kernel_ir::SimulationSourceLineageV1::new(
            *inventory_identity.sha256(),
            inventory_identity.byte_len(),
            *preflight_identity.sha256(),
            preflight_identity.byte_len(),
        )
        .map_err(ProductionPipelineError::SimulationBundle)?;
        let prepared = fe2o3_kernel_ir::PreparedSimulationBundleV1::new(
            compiler_execution_binding,
            lineage,
            production_identity,
            bindings.rustc_target.profile().device_target(),
            canonical_v7,
        )
        .map_err(ProductionPipelineError::SimulationBundle)?;
        Ok((lowered, bindings, prepared))
    }

    fn into_simulation_bundle_v1(
        self,
        compiler_execution_binding: fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV1, ProductionPipelineError> {
        let (lowered, bindings, prepared) =
            self.into_prepared_simulation_bundle_v1(compiler_execution_binding)?;
        let debug_map =
            compiler_debug_source_map_v1(&lowered, &bindings.debug_source_files, &prepared)?;
        prepared
            .finalize_with_source_map(debug_map)
            .map_err(ProductionPipelineError::SimulationBundle)
    }

    fn into_simulation_bundle_v2(
        self,
        compiler_execution_binding: fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV2, ProductionPipelineError> {
        let (lowered, bindings, prepared) =
            self.into_prepared_simulation_bundle_v1(compiler_execution_binding)?;
        let debug_map = compiler_debug_source_map_v2(
            &lowered,
            &bindings.debug_source_files,
            &bindings.debug_source_scopes,
            &bindings.debug_source_variables,
            &prepared,
        )?;
        let inner = prepared
            .finalize_without_source_map()
            .map_err(ProductionPipelineError::SimulationBundle)?;
        fe2o3_kernel_ir::VerifiedSimulationBundleV2::new(inner, debug_map)
            .map_err(ProductionPipelineError::SimulationBundleV2)
    }

    fn admit_formal_memory(
        self,
    ) -> Result<FormalMemoryAdmittedProductionCompilation, ProductionPipelineError> {
        let Self {
            lowered,
            ranked_verification,
            bindings,
        } = self;
        let admitted = fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1::try_admit(lowered)
            .map_err(ProductionPipelineError::FormalMemoryAdmission)?;
        Ok(FormalMemoryAdmittedProductionCompilation {
            admitted,
            ranked_verification,
            bindings,
        })
    }
}

impl FormalMemoryAdmittedProductionCompilation {
    fn lower_production_target(
        self,
    ) -> Result<TargetLoweredProductionCompilation, ProductionPipelineError> {
        let Self {
            admitted,
            ranked_verification,
            bindings,
        } = self;
        let target_profile = bindings.rustc_target.profile();
        let semantic = admitted.semantic_kir().semantic().semantic();
        let [semantic_root] = semantic.roots() else {
            return Err(ProductionPipelineError::Geometry(
                crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
            ));
        };
        let semantic_function = semantic
            .functions()
            .get(semantic_root.index() as usize)
            .ok_or(ProductionPipelineError::Geometry(
                crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
            ))?;
        let [typed_root] = bindings.typed_descriptor_roots.as_slice() else {
            return Err(ProductionPipelineError::Geometry(
                crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
            ));
        };
        let source_launch = typed_root
            .source_launch()
            .ok_or(ProductionPipelineError::Geometry(
            crate::production_geometry_v1::ProductionGeometryErrorV1::NonExactDescriptorWorkgroup,
        ))?;
        crate::production_geometry_v1::derive_production_geometry_v1(
            admitted.semantic_kir().module(),
            semantic_function,
            source_launch,
            target_profile.device_target(),
        )
        .map_err(ProductionPipelineError::Geometry)?;

        let target_bound = dialect_amdgcn::bind_production_target_v1(
            admitted.semantic_kir().module(),
            target_profile,
        )
        .map_err(ProductionPipelineError::TargetBinding)?;
        let (target_module, kernel_id) = target_bound.into_parts();
        let lowering = match target_profile {
            fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942 => {
                dialect_amdgcn::lower_kernel_to_gfx942_xnack_minus_llvm_ir(
                    &target_module,
                    &kernel_id,
                )
            }
            fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx950 => {
                dialect_amdgcn::lower_kernel_to_gfx950_xnack_minus_llvm_ir(
                    &target_module,
                    &kernel_id,
                )
            }
        };
        let dialect_llvm_ir = lowering.map_err(ProductionPipelineError::TargetLowering)?;
        let llvm_ir = dialect_amdgcn::bind_production_upstream_llvm_layout_v1(&dialect_llvm_ir)
            .map_err(ProductionPipelineError::UpstreamLlvmLayoutBinding)?;
        Ok(TargetLoweredProductionCompilation {
            admitted,
            ranked_verification,
            target_module,
            llvm_ir,
            bindings,
        })
    }
}

impl TargetLoweredProductionCompilation {
    pub(crate) fn module(&self) -> &fe2o3_kernel_ir::Module {
        &self.target_module
    }

    pub(crate) fn target_name(&self) -> &'static str {
        self.bindings.rustc_target.profile().device_target()
    }

    pub(crate) fn llvm_ir(&self) -> &str {
        &self.llvm_ir
    }

    pub(crate) fn workgroup_size(&self) -> Option<fe2o3_kernel_ir::WorkgroupSize> {
        self.target_module
            .kernels
            .first()
            .and_then(|kernel| kernel.workgroup_size)
    }

    pub(crate) fn semantic_function_count(&self) -> usize {
        self.admitted
            .semantic_kir()
            .semantic()
            .semantic()
            .functions()
            .len()
    }

    pub(crate) fn semantic_u32_induction_checked_addition_count(&self) -> usize {
        self.ranked_verification
            .semantic_u32_induction()
            .checked_additions_examined()
    }

    pub(crate) fn semantic_u32_induction_certificate_count(&self) -> usize {
        self.ranked_verification
            .semantic_u32_induction()
            .certificates()
            .len()
    }

    pub(crate) fn correspondence_block_count(&self) -> usize {
        self.admitted.semantic_kir().correspondence().blocks().len()
    }

    pub(crate) fn formal_witness_extent(&self) -> u64 {
        self.admitted.witness_extent()
    }

    pub(crate) fn formal_allocation_count(&self) -> usize {
        self.admitted.obligations().allocations().len()
    }

    pub(crate) fn formal_access_count(&self) -> usize {
        self.admitted.obligations().accesses().len()
    }

    pub(crate) fn ranked_dynamic_index_discharge_count(&self) -> usize {
        self.admitted.ranked_discharged_reasons().len()
    }

    pub(crate) fn runtime_bounds_requirement_count(&self) -> usize {
        self.admitted.obligations().bounds_requirements().len()
    }

    pub(crate) fn runtime_alias_requirement_count(&self) -> usize {
        self.admitted
            .obligations()
            .runtime_alias_requirements()
            .len()
    }

    pub(crate) fn inter_invocation_conflict_count(&self) -> usize {
        self.admitted
            .obligations()
            .inter_invocation_conflicts()
            .len()
    }

    pub(crate) fn retained_identity_and_transaction_binding_count(&self) -> usize {
        let _ = (
            &self.bindings.rustc_identity_inventory,
            &self.bindings.rustc_preflight_plan,
            &self.bindings.typed_descriptor_roots,
            &self.bindings.transaction.producer,
            &self.bindings.transaction.output_dir,
            &self.bindings.transaction.compiler_ffi_envelope,
        );
        6 + self
            .bindings
            .transaction
            .compiler_custody
            .retained_protected_binding_count()
    }

    pub(crate) fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }

    pub(crate) fn into_inert_worker_handoff_for_extraction(
        self,
    ) -> Result<fe2o3_compiler_ffi::CompilerModuleHandoffV2, ProductionPipelineError> {
        let Self {
            admitted,
            ranked_verification: _,
            target_module,
            llvm_ir,
            bindings,
        } = self;
        let AuthenticatedProductionBindings {
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings: _,
            debug_source_files: _,
            debug_source_scopes: _,
            debug_source_variables: _,
            typed_descriptor_roots,
            transaction,
        } = bindings;
        if rustc_preflight_plan.rustc_identity_inventory_sha256()
            != rustc_identity_inventory.sha256()
        {
            return Err(ProductionPipelineError::RustcLineageMismatch);
        }
        if !transaction.compiler_custody.is_extraction_only() {
            return Err(ProductionPipelineError::WorkerHandoffExtractionRequiresExtractionCustody);
        }
        let compiler_module = AuthenticatedProductionTargetModule {
            admitted,
            target: rustc_target.device_target(),
            target_module,
            llvm_ir,
            typed_descriptor_roots,
            compiler_ffi_envelope: transaction.compiler_ffi_envelope,
        };
        let prepared =
            crate::production_worker_handoff::prepare_production_worker_handoff(compiler_module)
                .map_err(ProductionPipelineError::WorkerHandoff)?;
        let (handoff, _) = prepared
            .into_validated_parts()
            .map_err(ProductionPipelineError::WorkerHandoff)?;
        Ok(handoff)
    }

    fn prepare_worker_handoff(
        self,
    ) -> Result<PreparedProductionWorkerPublication, ProductionPipelineError> {
        eprintln!(
            "[rustc-codegen-fe2o3] production compilation lowered {} admitted semantic function(s) into verified target-neutral Kernel IR module `{}` with {} exact block correspondence record(s), then admitted composed formal/ranked memory evidence for a {}-invocation structural witness with {} allocation(s), {} formal access(es), {} ranked dynamic-index discharge(s), {} runtime bounds requirement(s), {} runtime alias requirement(s), and {} inter-invocation conflict(s), and lowered exact target-bound KIR with compiler-selected-or-retained workgroup {:?} to {} byte(s) of deterministic {} LLVM text while retaining {} identity/transaction binding(s); artifact/launch authority {}; preparing exact compiler-module handoff",
            self.semantic_function_count(),
            self.module().id,
            self.correspondence_block_count(),
            self.formal_witness_extent(),
            self.formal_allocation_count(),
            self.formal_access_count(),
            self.ranked_dynamic_index_discharge_count(),
            self.runtime_bounds_requirement_count(),
            self.runtime_alias_requirement_count(),
            self.inter_invocation_conflict_count(),
            self.workgroup_size(),
            self.llvm_ir().len(),
            self.bindings.rustc_target.profile().device_target(),
            self.retained_identity_and_transaction_binding_count(),
            self.grants_artifact_or_launch_authority(),
        );
        let Self {
            admitted,
            ranked_verification,
            target_module,
            llvm_ir,
            bindings,
        } = self;
        let AuthenticatedProductionBindings {
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings,
            debug_source_files: _,
            debug_source_scopes: _,
            debug_source_variables: _,
            typed_descriptor_roots,
            transaction,
        } = bindings;
        let ProductionTransactionBindings {
            producer,
            output_dir,
            compiler_ffi_envelope,
            compiler_custody,
        } = transaction;
        if rustc_preflight_plan.rustc_identity_inventory_sha256()
            != rustc_identity_inventory.sha256()
        {
            return Err(ProductionPipelineError::RustcLineageMismatch);
        }
        drop(reference_effect_bindings);
        let ProtectedProductionPublicationCustody {
            attempt,
            invocation,
            compiler_execution,
        } = compiler_custody.into_publication_custody()?;
        let semantic_lineage = crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3::try_prepare(
            &rustc_identity_inventory,
            &rustc_preflight_plan,
            &rustc_target,
            ranked_verification,
            &admitted,
            &target_module,
            &llvm_ir,
        )
        .map_err(ProductionPipelineError::SemanticLineage)?;
        let compiler_module = AuthenticatedProductionTargetModule {
            admitted,
            target: rustc_target.device_target(),
            target_module,
            llvm_ir,
            typed_descriptor_roots,
            compiler_ffi_envelope,
        };
        let prepared =
            crate::production_worker_handoff::prepare_production_worker_handoff(compiler_module)
                .map_err(ProductionPipelineError::WorkerHandoff)?;
        Ok(PreparedProductionWorkerPublication {
            producer,
            output_dir,
            attempt,
            invocation,
            compiler_execution,
            semantic_lineage,
            rustc_target,
            prepared,
        })
    }

    fn publish_worker_handoff(
        self,
    ) -> Result<fe2o3_artifact_transaction::InertCompilerExecutionSubjectV1, ProductionPipelineError>
    {
        let publication = self.prepare_worker_handoff()?;
        let invocation = (*publication.invocation)
            .finish_for_publication()
            .map_err(ProductionPipelineError::ProtectedRustcInvocation)?;
        let (module_handoff, compiler_descriptor_source) = publication
            .prepared
            .into_validated_parts()
            .map_err(ProductionPipelineError::WorkerHandoff)?;
        let strict_handoff = publication
            .semantic_lineage
            .finish(
                &invocation,
                publication.rustc_target.device_target(),
                &compiler_descriptor_source,
                module_handoff,
            )
            .map_err(ProductionPipelineError::SemanticLineage)?;
        invocation
            .revalidate_for_publication()
            .map_err(ProductionPipelineError::ProtectedRustcInvocation)?;
        let receipt = fe2o3_artifact_transaction::publish_compiler_module_handoff_v3(
            &publication.output_dir,
            &publication.producer,
            publication.attempt,
            &strict_handoff,
        )
        .map_err(ProductionPipelineError::StrictV3Publication)?;
        let subject =
            fe2o3_artifact_transaction::InertCompilerExecutionSubjectV1::from_publication(
                receipt,
                &strict_handoff,
            )
            .map_err(ProductionPipelineError::CompilerExecutionSubject)?;
        let carriage = (*publication.compiler_execution)
            .acquire(subject.clone())
            .map_err(ProductionPipelineError::ProtectedCompilerExecution)?;
        let transport =
            fe2o3_artifact_transaction::publish_compiler_execution_receipt_transport_v1(
                &publication.output_dir,
                &publication.producer,
                &subject,
                carriage.canonical_bytes(),
            )
            .map_err(ProductionPipelineError::CompilerExecutionReceiptTransport)?;
        if transport.subject() != subject.identity()
            || transport.length() != carriage.canonical_bytes().len()
        {
            return Err(ProductionPipelineError::CompilerExecutionReceiptTransportBindingMismatch);
        }
        Ok(subject)
    }
}

fn sole_debug_map_body_v1(
    module: &fe2o3_kernel_ir::Module,
) -> Result<(usize, &fe2o3_kernel_ir::FunctionBody), ProductionPipelineError> {
    let mut bodies = module
        .functions
        .iter()
        .enumerate()
        .filter_map(|(ordinal, function)| function.body.as_ref().map(|body| (ordinal, body)));
    let body = bodies
        .next()
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "lowered KIR has no function body",
        ))?;
    if bodies.next().is_some() {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "V1 correspondence does not distinguish multiple KIR function bodies",
        ));
    }
    Ok(body)
}

fn compiler_debug_source_map_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
    prepared: &fe2o3_kernel_ir::PreparedSimulationBundleV1,
) -> Result<fe2o3_kernel_ir::DebugSourceMapDocumentV1, ProductionPipelineError> {
    let (function_ordinal, body) = sole_debug_map_body_v1(lowered.module())?;
    let function_ordinal = u64::try_from(function_ordinal).map_err(|_| {
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "KIR function ordinal does not fit the source-map wire",
        )
    })?;
    let block_ordinals = body
        .blocks
        .iter()
        .enumerate()
        .map(|(ordinal, block)| (block.id, ordinal))
        .collect::<BTreeMap<_, _>>();
    if block_ordinals.len() != body.blocks.len() {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "KIR body has duplicate block identities",
        ));
    }

    let mut mapped = BTreeMap::new();
    let mut eliminated = BTreeSet::new();
    for span in lowered.correspondence().statement_operation_spans() {
        let source = lowered
            .semantic()
            .resolve_statement(
                span.semantic_function(),
                span.semantic_block(),
                span.statement_ordinal(),
            )
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "statement correspondence does not resolve in retained semantic MIR",
            ))?
            .source();
        insert_debug_operation_range_v1(
            function_ordinal,
            body,
            &block_ordinals,
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
            source,
            &mut mapped,
            &mut eliminated,
        )?;
    }
    for span in lowered.correspondence().terminator_operation_spans() {
        let source = lowered
            .semantic()
            .resolve_terminator(span.semantic_function(), span.semantic_block())
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "terminator correspondence does not resolve in retained semantic MIR",
            ))?
            .source();
        insert_debug_operation_range_v1(
            function_ordinal,
            body,
            &block_ordinals,
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
            source,
            &mut mapped,
            &mut eliminated,
        )?;
    }

    let mut synthetic = BTreeSet::new();
    for span in lowered.correspondence().synthetic_operation_spans() {
        let block_ordinal = debug_block_ordinal_v1(
            body,
            &block_ordinals,
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
        )?;
        for operation in span.first_operation_ordinal()
            ..span
                .first_operation_ordinal()
                .checked_add(span.operation_count())
                .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "synthetic KIR operation range overflows",
                ))?
        {
            let site = fe2o3_kernel_ir::DebugSourceMapKirSiteV1::operation(
                function_ordinal,
                block_ordinal,
                u64::from(operation),
            );
            if !synthetic.insert(site) || mapped.contains_key(&site) {
                return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "synthetic and semantic operation ranges overlap",
                ));
            }
        }
    }
    for (block_ordinal, block) in body.blocks.iter().enumerate() {
        for operation_ordinal in 0..block.operations.len() {
            let site = fe2o3_kernel_ir::DebugSourceMapKirSiteV1::operation(
                function_ordinal,
                u64::try_from(block_ordinal).map_err(|_| {
                    ProductionPipelineError::SimulationDebugMapCorrespondence(
                        "KIR block ordinal does not fit the source-map wire",
                    )
                })?,
                u64::try_from(operation_ordinal).map_err(|_| {
                    ProductionPipelineError::SimulationDebugMapCorrespondence(
                        "KIR operation ordinal does not fit the source-map wire",
                    )
                })?,
            );
            if mapped.contains_key(&site) == synthetic.contains(&site) {
                return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "KIR operation is not covered exactly once by semantic or synthetic correspondence",
                ));
            }
        }
    }

    let referenced_files = mapped
        .values()
        .chain(&eliminated)
        .map(|span| span.file_identity())
        .collect::<BTreeSet<_>>();
    let captured_files = captured_files
        .iter()
        .map(|file| (file.identity(), file))
        .collect::<BTreeMap<_, _>>();
    let files = referenced_files
        .into_iter()
        .map(|identity| {
            captured_files.get(&identity).cloned().cloned().ok_or(
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "semantic source span has no same-session rustc file observation",
                ),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sites = mapped
        .into_iter()
        .map(|(site, span)| {
            fe2o3_kernel_ir::DebugSourceMapSiteV1::new(site, vec![span])
                .map_err(ProductionPipelineError::SimulationDebugMap)
        })
        .collect::<Result<Vec<_>, _>>()?;
    fe2o3_kernel_ir::DebugSourceMapDocumentV1::new(
        prepared.debug_source_map_binding(),
        files,
        sites,
        eliminated.into_iter().collect(),
    )
    .map_err(ProductionPipelineError::SimulationDebugMap)
}

fn compiler_debug_source_map_v2(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
    captured_scopes: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2],
    captured_variables: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2],
    prepared: &fe2o3_kernel_ir::PreparedSimulationBundleV1,
) -> Result<fe2o3_kernel_ir::DebugSourceMapDocumentV2, ProductionPipelineError> {
    let base = compiler_debug_source_map_v1(lowered, captured_files, prepared)?;
    let (function_ordinal, _) = sole_debug_map_body_v1(lowered.module())?;
    let function_ordinal = u64::try_from(function_ordinal).map_err(|_| {
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "KIR function ordinal does not fit the source-map V2 wire",
        )
    })?;
    let selected_semantic_function = lowered
        .correspondence()
        .blocks()
        .first()
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "lowering correspondence has no selected semantic function",
        ))?
        .semantic_function();

    let mut parameter_by_local = BTreeMap::new();
    for binding in lowered.correspondence().parameter_bindings() {
        if binding.semantic_function() != selected_semantic_function
            || parameter_by_local
                .insert(binding.semantic_local(), binding.kernel_ir_value())
                .is_some()
        {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "KIR parameter correspondence is not unique for the selected semantic function",
            ));
        }
    }

    let selected_scope_count = captured_scopes
        .iter()
        .filter(|scope| scope.function == selected_semantic_function)
        .count();
    if selected_scope_count > fe2o3_kernel_ir::MAX_DEBUG_SOURCE_SCOPES_V2 {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "compiler source scopes exceed the bounded V2 map domain",
        ));
    }
    let mut scopes = Vec::new();
    scopes
        .try_reserve_exact(selected_scope_count)
        .map_err(|_| {
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "compiler source-scope map allocation failed",
            )
        })?;
    for scope in captured_scopes
        .iter()
        .filter(|scope| scope.function == selected_semantic_function)
    {
        scopes.push(
            fe2o3_kernel_ir::DebugSourceScopeV2::new(
                scope.identity,
                function_ordinal,
                scope.parent_identity,
                scope.depth,
                debug_source_scope_span_v2(scope.source)?,
            )
            .map_err(ProductionPipelineError::SimulationDebugMapV2)?,
        );
    }
    let scope_identities = scopes
        .iter()
        .map(|scope| scope.identity())
        .collect::<BTreeSet<_>>();

    let selected_variable_count = captured_variables
        .iter()
        .filter(|variable| variable.function == selected_semantic_function)
        .count();
    if selected_variable_count > fe2o3_kernel_ir::MAX_DEBUG_SOURCE_VARIABLES_V2 {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "compiler source variables exceed the bounded V2 map domain",
        ));
    }
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(selected_variable_count)
        .map_err(|_| {
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "compiler source-variable map allocation failed",
            )
        })?;
    for variable in captured_variables
        .iter()
        .filter(|variable| variable.function == selected_semantic_function)
    {
        if !scope_identities.contains(&variable.scope_identity) {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "compiler source variable references an unretained lexical scope",
            ));
        }
        let name = variable.name.clone().ok_or(
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "rustc source-variable name is empty, control-containing, or exceeds the V2 bound",
            ),
        )?;
        let (fallback, parameter) = match variable.class {
            crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableClassV2::Local(local) => {
                match parameter_by_local
                    .get(&local)
                    .copied()
                    .filter(|_| variable.entry_value_preserved)
                {
                    Some(value) => (
                        fe2o3_kernel_ir::DebugSourceVariableFallbackV2::NotInScope,
                        Some(value),
                    ),
                    None => (
                        fe2o3_kernel_ir::DebugSourceVariableFallbackV2::Unrepresented,
                        None,
                    ),
                }
            }
            crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableClassV2::Unrepresented => (
                fe2o3_kernel_ir::DebugSourceVariableFallbackV2::Unrepresented,
                None,
            ),
        };
        let mut emitted = fe2o3_kernel_ir::DebugSourceVariableV2::new(
            variable.identity,
            name,
            function_ordinal,
            variable.scope_identity,
            fallback,
            Vec::new(),
        )
        .map_err(ProductionPipelineError::SimulationDebugMapV2)?;
        if let Some(value) = parameter {
            emitted = emitted
                .with_function_binding(
                    fe2o3_kernel_ir::DebugSourceVariableFunctionBindingV2::new(
                        1,
                        u64::from(value.0),
                    )
                    .map_err(ProductionPipelineError::SimulationDebugMapV2)?,
                )
                .map_err(ProductionPipelineError::SimulationDebugMapV2)?;
        }
        variables.push(emitted);
    }

    let captured_files = captured_files
        .iter()
        .map(|file| (file.identity(), file.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut files = base
        .files()
        .iter()
        .map(|file| (file.identity(), file.clone()))
        .collect::<BTreeMap<_, _>>();
    for scope in &scopes {
        let identity = scope.span().file_identity();
        if !files.contains_key(&identity) {
            let file = captured_files.get(&identity).cloned().ok_or(
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "source-variable scope has no same-session rustc file observation",
                ),
            )?;
            files.insert(identity, file);
        }
    }
    let mut file_values = Vec::new();
    file_values.try_reserve_exact(files.len()).map_err(|_| {
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "compiler source-map V2 file allocation failed",
        )
    })?;
    file_values.extend(files.into_values());
    let mut sites = Vec::new();
    sites.try_reserve_exact(base.sites().len()).map_err(|_| {
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "compiler source-map V2 site allocation failed",
        )
    })?;
    sites.extend_from_slice(base.sites());
    let mut eliminated = Vec::new();
    eliminated
        .try_reserve_exact(base.eliminated().len())
        .map_err(|_| {
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "compiler source-map V2 eliminated-span allocation failed",
            )
        })?;
    eliminated.extend_from_slice(base.eliminated());
    fe2o3_kernel_ir::DebugSourceMapDocumentV2::new(
        base.binding(),
        file_values,
        sites,
        eliminated,
        scopes,
        variables,
    )
    .map_err(ProductionPipelineError::SimulationDebugMapV2)
}

fn debug_source_scope_span_v2(
    source: fe2o3_mir_model::semantic_mir_v1::SemanticSourceProvenanceV1,
) -> Result<fe2o3_kernel_ir::DebugSourceMapSpanV1, ProductionPipelineError> {
    let origin =
        source
            .call_site()
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "source-variable scope has no resolved source call site",
            ))?;
    let (byte_start, byte_end) = origin.byte_range();
    let (line, column) = origin.start_coordinate();
    fe2o3_kernel_ir::DebugSourceMapSpanV1::new_eliminated(
        *origin.file().as_bytes(),
        byte_start,
        byte_end,
        line,
        column,
    )
    .map_err(ProductionPipelineError::SimulationDebugMap)
}

#[allow(clippy::too_many_arguments)]
fn insert_debug_operation_range_v1(
    function_ordinal: u64,
    body: &fe2o3_kernel_ir::FunctionBody,
    block_ordinals: &BTreeMap<fe2o3_kernel_ir::BlockId, usize>,
    block: fe2o3_kernel_ir::BlockId,
    first_operation: u32,
    operation_count: u32,
    source: fe2o3_mir_model::semantic_mir_v1::SemanticSourceProvenanceV1,
    mapped: &mut BTreeMap<
        fe2o3_kernel_ir::DebugSourceMapKirSiteV1,
        fe2o3_kernel_ir::DebugSourceMapSpanV1,
    >,
    eliminated: &mut BTreeSet<fe2o3_kernel_ir::DebugSourceMapSpanV1>,
) -> Result<(), ProductionPipelineError> {
    // V1 intentionally resolves every macro-originated construct to rustc's
    // final source call site. Expansion-chain identity remains in semantic MIR
    // but is not serialized as a source-map span in this version.
    let origin =
        source
            .call_site()
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "semantic operation has no resolved source call site",
            ))?;
    let (byte_start, byte_end) = origin.byte_range();
    let (line, column) = origin.start_coordinate();
    if operation_count != 0 && byte_start >= byte_end {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "resolved source call-site span is empty",
        ));
    }
    let source_span = if operation_count == 0 {
        fe2o3_kernel_ir::DebugSourceMapSpanV1::new_eliminated(
            *origin.file().as_bytes(),
            byte_start,
            byte_end,
            line,
            column,
        )
    } else {
        fe2o3_kernel_ir::DebugSourceMapSpanV1::new(
            *origin.file().as_bytes(),
            byte_start,
            byte_end,
            line,
            column,
        )
    }
    .map_err(ProductionPipelineError::SimulationDebugMap)?;
    let block_ordinal = debug_block_ordinal_v1(
        body,
        block_ordinals,
        block,
        first_operation,
        operation_count,
    )?;
    if operation_count == 0 {
        eliminated.insert(source_span);
        return Ok(());
    }
    let end = first_operation.checked_add(operation_count).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "semantic KIR operation range overflows",
        ),
    )?;
    for operation in first_operation..end {
        let site = fe2o3_kernel_ir::DebugSourceMapKirSiteV1::operation(
            function_ordinal,
            block_ordinal,
            u64::from(operation),
        );
        if mapped.insert(site, source_span).is_some() {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "one KIR operation is attributed to multiple semantic constructs",
            ));
        }
    }
    Ok(())
}

fn debug_block_ordinal_v1(
    body: &fe2o3_kernel_ir::FunctionBody,
    block_ordinals: &BTreeMap<fe2o3_kernel_ir::BlockId, usize>,
    block: fe2o3_kernel_ir::BlockId,
    first_operation: u32,
    operation_count: u32,
) -> Result<u64, ProductionPipelineError> {
    let ordinal = *block_ordinals.get(&block).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "correspondence names an unknown KIR block",
        ),
    )?;
    let operation_end = usize::try_from(first_operation)
        .ok()
        .and_then(|first| {
            usize::try_from(operation_count)
                .ok()
                .and_then(|count| first.checked_add(count))
        })
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "KIR operation range does not fit this compiler host",
        ))?;
    if operation_end
        > body
            .blocks
            .get(ordinal)
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "correspondence KIR block ordinal is unavailable",
            ))?
            .operations
            .len()
    {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "correspondence KIR operation range is outside its block",
        ));
    }
    u64::try_from(ordinal).map_err(|_| {
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "KIR block ordinal does not fit the source-map wire",
        )
    })
}

impl RankedVerifiedProductionCompilation {
    pub(crate) fn ranked_roots(
        &self,
    ) -> &[crate::production_ranked_projection_v1::ProductionRankedRootProgramV1] {
        self.ranked.roots()
    }

    pub(crate) fn ranked_root_count(&self) -> usize {
        self.ranked.root_count()
    }

    pub(crate) fn ranked_ir(&self) -> &str {
        self.ranked.ranked_ir()
    }

    pub(crate) fn function_name(&self) -> &str {
        self.ranked.function_name()
    }

    pub(crate) fn semantic_function_count(&self) -> usize {
        self.ranked.semantic_function_count()
    }

    pub(crate) fn semantic_callable_count(&self) -> usize {
        self.ranked.semantic_callable_count()
    }

    pub(crate) fn bounds_are_clean(&self) -> bool {
        self.ranked.bounds_are_clean()
    }

    pub(crate) fn all_kernel_checks_are_clean(&self) -> bool {
        self.ranked.all_kernel_checks_are_clean()
    }

    pub(crate) fn retained_identity_and_transaction_binding_count(&self) -> usize {
        let _ = (
            &self.bindings.rustc_identity_inventory,
            &self.bindings.rustc_preflight_plan,
            &self.bindings.typed_descriptor_roots,
            &self.bindings.transaction.producer,
            &self.bindings.transaction.output_dir,
            &self.bindings.transaction.compiler_ffi_envelope,
        );
        6 + self
            .bindings
            .transaction
            .compiler_custody
            .retained_protected_binding_count()
    }

    pub(crate) fn grants_artifact_or_launch_authority(&self) -> bool {
        self.ranked.grants_artifact_or_launch_authority()
    }
}

impl<'tcx> ProductionCompilation<'tcx, CollectedRustStage<'tcx>> {
    /// Retains the collector-sealed closure without granting semantic authority.
    /// The next transition must authenticate every imported MIR fact.
    pub(crate) fn from_collected_device_closure(
        tcx: TyCtxt<'tcx>,
        closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
        producer: ProducerIdentity,
        output_dir: PathBuf,
        build_attempt: BuildAttempt,
        invocation: AdmittedProtectedRustcInvocationV1,
        compiler_execution: AdmittedProtectedCompilerExecutionV1,
    ) -> Result<Self, ProductionPipelineError> {
        Self::from_collected_device_closure_with_custody(
            tcx,
            closure,
            producer,
            output_dir,
            ProductionCompilerCustody::protected(invocation, compiler_execution, build_attempt),
            crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2::Disabled,
        )
    }

    pub(crate) fn from_collected_device_closure_for_extraction(
        tcx: TyCtxt<'tcx>,
        closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
        producer: ProducerIdentity,
        output_dir: PathBuf,
    ) -> Result<Self, ProductionPipelineError> {
        Self::from_collected_device_closure_with_custody(
            tcx,
            closure,
            producer,
            output_dir,
            ProductionCompilerCustody::extraction_only(),
            crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2::Disabled,
        )
    }

    pub(crate) fn from_collected_device_closure_for_simulation_v2(
        tcx: TyCtxt<'tcx>,
        closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
        producer: ProducerIdentity,
        output_dir: PathBuf,
    ) -> Result<Self, ProductionPipelineError> {
        Self::from_collected_device_closure_with_custody(
            tcx,
            closure,
            producer,
            output_dir,
            ProductionCompilerCustody::extraction_only(),
            crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2::SourceVariables,
        )
    }

    fn from_collected_device_closure_with_custody(
        tcx: TyCtxt<'tcx>,
        closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
        producer: ProducerIdentity,
        output_dir: PathBuf,
        compiler_custody: ProductionCompilerCustody,
        debug_source_capture: crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2,
    ) -> Result<Self, ProductionPipelineError> {
        if closure.function_count() == 0 {
            return Err(ProductionPipelineError::EmptyCollectedDeviceClosure);
        }
        let typed_descriptor_roots = closure
            .rederive_typed_descriptor_roots(tcx)
            .map_err(ProductionPipelineError::DescriptorEvidence)?;
        let compiler_ffi_envelope = closure.compiler_ffi_observation().cloned();
        Ok(Self {
            stage: CollectedRustStage {
                tcx,
                closure,
                typed_descriptor_roots,
                debug_source_capture,
                transaction: ProductionTransactionBindings {
                    producer,
                    output_dir,
                    compiler_ffi_envelope,
                    compiler_custody,
                },
            },
            invariant_session: PhantomData,
        })
    }

    fn import_semantic_mir(
        self,
    ) -> Result<ProductionCompilation<'tcx, AdmittedSemanticMirStage>, ProductionPipelineError>
    {
        let CollectedRustStage {
            tcx,
            closure,
            typed_descriptor_roots,
            debug_source_capture,
            transaction,
        } = self.stage;
        let crate::collector::ConstructedProductionSemanticMirV1 {
            semantic_mir,
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings,
            debug_source_files,
            debug_source_scopes,
            debug_source_variables,
        } = crate::collector::construct_production_semantic_mir_v1(
            tcx,
            closure,
            debug_source_capture,
        )
        .map_err(ProductionPipelineError::SemanticImport)?;
        Ok(ProductionCompilation {
            stage: AdmittedSemanticMirStage {
                semantic_mir,
                bindings: AuthenticatedProductionBindings {
                    rustc_identity_inventory,
                    rustc_preflight_plan,
                    rustc_target,
                    reference_effect_bindings,
                    debug_source_files,
                    debug_source_scopes,
                    debug_source_variables,
                    typed_descriptor_roots,
                    transaction,
                },
            },
            invariant_session: PhantomData,
        })
    }

    /// Consumes the only production transaction through import and verification.
    pub(crate) fn verify_general_kernel_checks(
        self,
    ) -> Result<RankedVerifiedProductionCompilation, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .verify_general_kernel_checks()
    }

    /// Consumes the sole production transaction through exact semantic MIR,
    /// formal memory admission, and exact authenticated-target LLVM lowering.
    pub(crate) fn lower_production_target(
        self,
    ) -> Result<TargetLoweredProductionCompilation, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .verify_general_kernel_checks()?
            .lower_target_neutral()?
            .admit_formal_memory()?
            .lower_production_target()
    }

    /// Consumes the sole production transaction through the same admitted
    /// source, ranked checks, and target-neutral lowering as production, then
    /// emits an inert exact-V7 simulation input. No target lowering or
    /// artifact transaction is entered.
    pub(crate) fn export_simulation_bundle_v1(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV1, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .verify_general_kernel_checks()?
            .lower_target_neutral()?
            .into_simulation_bundle_v1(
                fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
            )
    }

    /// Emits the explicit V2 simulation envelope with compiler-produced,
    /// exact-KIR-bound source-variable metadata. This remains inert and grants
    /// no compiler, proof, artifact, hardware, load, or launch authority.
    pub(crate) fn export_simulation_bundle_v2(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV2, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .verify_general_kernel_checks()?
            .lower_target_neutral()?
            .into_simulation_bundle_v2(
                fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
            )
    }

    /// Publishes the exact production compiler module into the managed,
    /// preselected attempt-scoped protocol. This grants no link, artifact, load,
    /// or launch authority.
    pub(crate) fn publish_worker_handoff(
        self,
    ) -> Result<fe2o3_artifact_transaction::InertCompilerExecutionSubjectV1, ProductionPipelineError>
    {
        self.lower_production_target()?.publish_worker_handoff()
    }

    /// Retains the original extraction milestone while consuming the same
    /// transaction and importer as the production backend.
    pub(crate) fn require_semantic_mir_import(self) -> ProductionPipelineError {
        match self.import_semantic_mir() {
            Ok(transaction) => match transaction.construct_semantic_middle_end() {
                Ok(transaction) => transaction.require_target_neutral_lowering(),
                Err(error) => error,
            },
            Err(error) => error,
        }
    }
}

impl<'tcx> ProductionCompilation<'tcx, AdmittedSemanticMirStage> {
    fn construct_semantic_middle_end(
        self,
    ) -> Result<ProductionCompilation<'tcx, EquivalentSemanticMirStage>, ProductionPipelineError>
    {
        let AdmittedSemanticMirStage {
            semantic_mir,
            bindings,
        } = self.stage;
        let semantic_mir = fe2o3_pliron::ProductionSemanticMirOwnerV1::try_new(
            semantic_mir,
            fe2o3_pliron::ProductionSemanticMirLimitsV1::default(),
        )
        .map_err(ProductionPipelineError::SemanticMiddleEnd)?;
        Ok(ProductionCompilation {
            stage: EquivalentSemanticMirStage {
                semantic_mir,
                bindings,
            },
            invariant_session: PhantomData,
        })
    }
}

impl<'tcx> ProductionCompilation<'tcx, EquivalentSemanticMirStage> {
    fn require_target_neutral_lowering(self) -> ProductionPipelineError {
        let EquivalentSemanticMirStage {
            semantic_mir,
            bindings,
        } = self.stage;
        let error =
            crate::collector::ProductionSemanticImportErrorV1::TargetNeutralLoweringPending {
                functions: semantic_mir.semantic().functions().len(),
                callables: semantic_mir.semantic().callables().len(),
                rustc_identity_inventory_sha256: bindings.rustc_identity_inventory.sha256(),
                rustc_preflight_plan_sha256: bindings.rustc_preflight_plan.sha256(),
                semantic_sha256: *semantic_mir.semantic().semantic_sha256().as_bytes(),
            };
        drop((semantic_mir, bindings));
        ProductionPipelineError::SemanticImport(error)
    }

    fn verify_general_kernel_checks(
        self,
    ) -> Result<RankedVerifiedProductionCompilation, ProductionPipelineError> {
        let EquivalentSemanticMirStage {
            semantic_mir,
            bindings,
        } = self.stage;
        crate::compiler_descriptor::validate_production_v1_semantic_ownership_evidence(
            &bindings.typed_descriptor_roots,
            semantic_mir.semantic(),
        )
        .map_err(ProductionPipelineError::DescriptorEvidence)?;
        let ranked_roots = bindings
            .typed_descriptor_roots
            .iter()
            .map(|typed_root| {
                let source_launch = typed_root.source_launch().ok_or(
                    ProductionPipelineError::Geometry(
                        crate::production_geometry_v1::ProductionGeometryErrorV1::NonExactDescriptorWorkgroup,
                    ),
                )?;
                Ok(
                    crate::production_ranked_projection_v1::ProductionRankedRootInputV1::new(
                        typed_root.logical_name(),
                        typed_root.kernel_binding_bytes(),
                        source_launch,
                    ),
                )
            })
            .collect::<Result<Vec<_>, ProductionPipelineError>>()?;
        let ranked =
            crate::production_ranked_projection_v1::project_and_verify_ranked_semantic_mir_v1(
                semantic_mir,
                &ranked_roots,
                &bindings.reference_effect_bindings,
            )
            .map_err(ProductionPipelineError::RankedProjection)?;
        Ok(RankedVerifiedProductionCompilation { ranked, bindings })
    }
}

impl RankedVerifiedProductionCompilation {
    fn lower_target_neutral(
        self,
    ) -> Result<TargetNeutralProductionCompilation, ProductionPipelineError> {
        let Self { ranked, bindings } = self;
        let roster_receipt = ranked
            .into_verified_roster_receipt()
            .map_err(ProductionPipelineError::RankedVerification)?;
        debug_assert!(!roster_receipt.grants_artifact_or_launch_authority());
        debug_assert!(roster_receipt.verify_equivalence().is_ok());
        let root_count = roster_receipt.root_count();
        if root_count != 1 {
            drop((roster_receipt, bindings));
            return Err(ProductionPipelineError::MultiRootTargetNeutralLowering {
                roots: root_count,
            });
        }
        let source_rank = roster_receipt
            .source_order_roots()
            .first()
            .map(|root| root.source_rank())
            .ok_or(ProductionPipelineError::MultiRootTargetNeutralLowering { roots: 0 })?;
        debug_assert_eq!(roster_receipt.canonical_kernel_order(), &[0]);
        debug_assert_ne!(
            roster_receipt.canonical_roster_identity().as_bytes(),
            &[0; 32],
        );
        let (receipt, ranked_verification) = roster_receipt
            .into_singleton_verified_receipt()
            .map_err(ProductionPipelineError::RankedVerification)?;
        debug_assert_eq!(
            ranked_verification.has_authenticated_functional_verification(),
            receipt
                .lowering()
                .has_retained_policy_checked_refinement_staging()
        );
        debug_assert!(ranked_verification.retained_functional_verification_is_coherent());
        let lowered =
            fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1::try_lower_after_ranked_checks(
                receipt,
                fe2o3_lower_mir_kernel::ProductionSemanticKirLimitsV1::default(),
                source_rank,
            )
            .map_err(ProductionPipelineError::TargetNeutralLowering)?;
        if lowered.mir_pliron_translation_validation().is_none() {
            return Err(ProductionPipelineError::MissingMirPlironTranslationValidation);
        }
        Ok(TargetNeutralProductionCompilation {
            lowered,
            ranked_verification,
            bindings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn debug_map_test_function(name: &str) -> fe2o3_kernel_ir::Function {
        fe2o3_kernel_ir::Function::kernel_entry(
            name,
            fe2o3_kernel_ir::Signature::new(vec![], vec![]),
            vec![],
            vec![fe2o3_kernel_ir::BasicBlock::new(fe2o3_kernel_ir::BlockId(
                0,
            ))],
        )
    }

    #[test]
    fn debug_map_body_selection_fails_closed_until_correspondence_names_functions() {
        let empty = fe2o3_kernel_ir::Module::new("empty");
        assert!(matches!(
            sole_debug_map_body_v1(&empty),
            Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "lowered KIR has no function body"
            ))
        ));

        let mut multiple = fe2o3_kernel_ir::Module::new("multiple");
        multiple.functions.push(debug_map_test_function("kernel"));
        multiple.functions.push(debug_map_test_function("helper"));
        assert!(matches!(
            sole_debug_map_body_v1(&multiple),
            Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "V1 correspondence does not distinguish multiple KIR function bodies"
            ))
        ));
    }

    #[test]
    fn host_only_and_device_dispositions_are_exact() {
        assert_eq!(disposition(0), ProductionDisposition::HostOnly);
        assert_eq!(disposition(1), ProductionDisposition::DeviceTransaction);
        assert_eq!(
            disposition(usize::MAX),
            ProductionDisposition::DeviceTransaction
        );
    }

    #[test]
    fn private_production_implementation_is_unversioned() {
        let backend = include_str!("lib.rs");
        let pipeline = include_str!("production_pipeline.rs");
        assert!(backend.contains("mod production_pipeline;"));
        for retired in [
            concat!("production_pipeline", "_v1"),
            concat!("ProductionPipelineError", "V1"),
            concat!("ProductionCompilation", "V1"),
            concat!("ProductionDisposition", "V1"),
            concat!("ProductionCompilerCustody", "V1"),
            concat!("RetainedProductionDeviceAdmission", "V1"),
        ] {
            assert!(!backend.contains(retired), "backend retains {retired}");
            assert!(!pipeline.contains(retired), "pipeline retains {retired}");
        }
    }

    #[test]
    fn custom_llvm_configuration_is_terminal_before_construction() {
        assert!(reject_custom_llvm_configuration(false).is_ok());
        assert!(matches!(
            reject_custom_llvm_configuration(true),
            Err(ProductionPipelineError::CustomLlvmConfiguration)
        ));
    }

    #[test]
    fn production_layout_binding_uses_the_measured_worker_spelling() {
        let legacy = format!(
            "target triple = \"amdgcn-amd-amdhsa\"\ntarget datalayout = \"{}\"\n\ndefine void @body() {{ ret void }}\n",
            dialect_amdgcn::GFX942_XNACK_MINUS_DATA_LAYOUT
        );
        let bound = dialect_amdgcn::bind_production_upstream_llvm_layout_v1(&legacy).unwrap();
        assert!(bound.starts_with(&format!(
            "target triple = \"amdgcn-amd-amdhsa\"\ntarget datalayout = \"{}\"\n\n",
            crate::production_target_v1::PRODUCTION_WORKER_DATA_LAYOUT_V1
        )));
        assert!(bound.contains("target datalayout = \"e-m:e-"));
        assert!(bound.ends_with("define void @body() { ret void }\n"));
        assert_eq!(bound.matches("target triple =").count(), 1);
        assert_eq!(bound.matches("target datalayout =").count(), 1);
    }

    #[test]
    fn production_layout_binding_rejects_noncanonical_headers() {
        let canonical = format!(
            "target triple = \"amdgcn-amd-amdhsa\"\ntarget datalayout = \"{}\"\n\ndefine void @body() {{ ret void }}\n",
            dialect_amdgcn::GFX942_XNACK_MINUS_DATA_LAYOUT
        );
        for hostile in [
            canonical.replacen("target triple", "source_filename", 1),
            canonical.replacen("\n\n", "\n", 1),
            format!("{canonical}target datalayout = \"e-p:64:64\"\n"),
        ] {
            assert!(dialect_amdgcn::bind_production_upstream_llvm_layout_v1(&hostile).is_err());
        }
    }

    #[test]
    fn production_target_lowering_uses_shared_replayable_transforms() {
        let source = include_str!("production_pipeline.rs");
        let transaction = source
            .split("impl FormalMemoryAdmittedProductionCompilation")
            .nth(1)
            .expect("target-lowering stage")
            .split("impl TargetLoweredProductionCompilation")
            .next()
            .expect("bounded target-lowering body");
        assert!(transaction.contains("dialect_amdgcn::bind_production_target_v1("));
        assert!(transaction.contains("dialect_amdgcn::bind_production_upstream_llvm_layout_v1("));
        assert!(!transaction.contains("required_capabilities.insert"));
    }

    #[test]
    fn worker_publication_cannot_bypass_general_pliron_checks() {
        let source = include_str!("production_pipeline.rs");
        let transaction = source
            .split("pub(crate) fn lower_production_target(")
            .nth(1)
            .expect("AMDGPU production transaction")
            .split("pub(crate) fn publish_worker_handoff(")
            .next()
            .expect("bounded transaction body");
        let verify = transaction
            .find(".verify_general_kernel_checks()?")
            .expect("mandatory general PLIRON checks");
        let lower = transaction
            .find(".lower_target_neutral()?")
            .expect("target-neutral lowering");
        assert!(verify < lower, "lowering ran before general PLIRON checks");
        assert!(
            include_str!("production_ranked_projection_v1.rs")
                .contains("prepare_reference_effect_request_v2")
        );
    }

    #[test]
    fn referenced_kernels_complete_all_functional_gates_before_kir_lowering() {
        let projection = include_str!("production_ranked_projection_v1.rs");
        let semantic = projection
            .find("derive_and_reconcile_mir_pliron_semantic_contract_v1")
            .expect("compiler-owned semantic-contract derivation");
        let parallel = projection
            .find("derive_and_require_parallel_reference_contract_v1")
            .expect("compiler-owned parallel-contract derivation");
        let aggregate = projection
            .find("authenticate_mir_pliron_contract_per_compilation_v1")
            .expect("aggregate per-compilation Verus gate");
        assert!(semantic < parallel && parallel < aggregate);

        let pipeline = include_str!("production_pipeline.rs");
        let roster = pipeline
            .find(".into_verified_roster_receipt()")
            .expect("ranked roster verification transition");
        let singleton = pipeline
            .find(".into_singleton_verified_receipt()")
            .expect("singleton ranked receipt transition");
        let lowering = pipeline
            .find("ProductionSemanticKirOwnerV1::try_lower_after_ranked_checks")
            .expect("KIR lowering transition");
        assert!(
            roster < singleton && singleton < lowering,
            "KIR lowering ran before functional verification"
        );
    }

    #[test]
    fn ranked_roster_receipt_stops_before_singleton_kir_authority() {
        let pipeline = include_str!("production_pipeline.rs");
        let roster = pipeline
            .find(".into_verified_roster_receipt()")
            .expect("ranked roster receipt transition");
        let root_count = pipeline[roster..]
            .find("let root_count = roster_receipt.root_count()")
            .map(|offset| roster + offset)
            .expect("roster cardinality gate");
        let multi_root_stop = pipeline[root_count..]
            .find("ProductionPipelineError::MultiRootTargetNeutralLowering")
            .map(|offset| root_count + offset)
            .expect("explicit pre-KIR multi-root stop");
        let singleton = pipeline[multi_root_stop..]
            .find(".into_singleton_verified_receipt()")
            .map(|offset| multi_root_stop + offset)
            .expect("singleton receipt authority");
        let kir = pipeline[singleton..]
            .find("ProductionSemanticKirOwnerV1::try_lower_after_ranked_checks")
            .map(|offset| singleton + offset)
            .expect("KIR authority transition");
        assert!(roster < root_count && root_count < multi_root_stop);
        assert!(multi_root_stop < singleton && singleton < kir);
    }

    #[test]
    fn production_publication_has_one_protected_custody_path() {
        let pipeline = include_str!("production_pipeline.rs");
        let worker = include_str!("production_worker_handoff.rs");
        let lineage = include_str!("production_semantic_lineage_v3.rs");
        for removed in [
            concat!("ProductionCompilerModule", "PublicationV1"),
            concat!("PreparedProductionCompiler", "PublicationV1"),
            concat!("ProtectedHandoff", "RequiresV2"),
            concat!("UnprotectedHandoff", "RequiresV1"),
            concat!("publish_worker_handoff", "_v3"),
            concat!("publish_prepared_production_v1", "_worker_handoff("),
            concat!("PreparedProductionV1", "WorkerHandoffV1"),
            concat!("PreparedProductionLineage", "WorkerHandoffV3"),
            concat!("prepare_production_v1", "_worker_handoff"),
        ] {
            assert!(
                !pipeline.contains(removed) && !worker.contains(removed),
                "obsolete production publication variant remains: {removed}",
            );
        }
        assert!(pipeline.contains("ProductionCompilerCustody::protected("));
        assert!(pipeline.contains("compiler_execution"));
        assert!(pipeline.contains(concat!("publish_compiler_module_handoff", "_v3")));
        assert!(pipeline.contains(concat!(
            "publish_compiler_execution_receipt_transport",
            "_v1"
        )));
        assert!(!pipeline.contains(concat!(
            "let invocation_",
            "descriptor = invocation.descriptor().clone()"
        )));
        assert!(lineage.contains("invocation_custody: &FinishedProtectedRustcInvocationV3"));
        assert!(!lineage.contains("invocation: RustcInvocationDescriptorV3"));

        let lineage_finish = pipeline
            .find(".semantic_lineage\n            .finish(\n                &invocation,")
            .expect("semantic lineage consumes live protected invocation custody");
        let final_revalidation = pipeline[lineage_finish..]
            .find("invocation\n            .revalidate_for_publication()")
            .map(|offset| lineage_finish + offset)
            .expect("protected invocation is revalidated after lineage construction");
        let durable_publication = pipeline[lineage_finish..]
            .find(concat!("publish_compiler_module_handoff", "_v3"))
            .map(|offset| lineage_finish + offset)
            .expect("strict V3 handoff publication remains present");
        let execution_subject = pipeline[lineage_finish..]
            .find("InertCompilerExecutionSubjectV1::from_publication")
            .map(|offset| lineage_finish + offset)
            .expect("strict publication derives one canonical compiler-execution subject");
        assert!(lineage_finish < final_revalidation && final_revalidation < durable_publication);
        assert!(durable_publication < execution_subject);
        let receipt_acquisition = pipeline[execution_subject..]
            .find(".acquire(subject.clone())")
            .map(|offset| execution_subject + offset)
            .expect("exact execution subject is sent to the protected issuer");
        let receipt_transport = pipeline[receipt_acquisition..]
            .find(concat!(
                "publish_compiler_execution_receipt_transport",
                "_v1"
            ))
            .map(|offset| receipt_acquisition + offset)
            .expect("issuer receipt is published beside the exact V3 handoff");
        assert!(execution_subject < receipt_acquisition && receipt_acquisition < receipt_transport);
    }

    #[test]
    fn production_module_contains_no_profile_selection_vocabulary() {
        let sources = [
            include_str!("production_pipeline.rs"),
            include_str!("collector/production_importer_v1.rs"),
            include_str!("rustc_semantic_adapter_v1.rs"),
            include_str!("rustc_semantic_plan_v1.rs"),
            include_str!("production_semantic_fn_abi_v1.rs"),
            include_str!("production_semantic_types_v1.rs"),
            include_str!("production_semantic_terminal_v1.rs"),
            include_str!("reference_effect_v1.rs"),
        ];
        for forbidden in [
            concat!("General", "Gemm"),
            concat!("Flash", "Attention"),
            concat!("Row", "Softmax"),
            concat!("Moe", "Top2"),
            concat!("export", "_name"),
            concat!("source", " substring"),
            concat!("MIR", " transcript"),
            concat!("legacy", "-v1"),
            concat!("kernel-ir", "-v1"),
            concat!("Collection", "Result"),
            concat!("target: AmdGpu", "Target"),
        ] {
            assert!(
                !sources[0].contains(forbidden),
                "production transaction contains forbidden selector term {forbidden:?}"
            );
        }

        for forbidden_importer_term in [
            concat!("General", "Gemm"),
            concat!("Flash", "Attention"),
            concat!("Row", "Softmax"),
            concat!("Moe", "Top2"),
            concat!("source", " substring"),
            concat!("MIR", " transcript"),
            concat!("legacy", "-v1"),
            concat!("kernel-ir", "-v1"),
        ] {
            assert!(
                sources
                    .iter()
                    .skip(1)
                    .all(|source| !source.contains(forbidden_importer_term)),
                "production importer contains forbidden selector term {forbidden_importer_term:?}"
            );
        }

        for forbidden_dependency in [
            concat!("mir_import", "_v2"),
            concat!("same_session", "_rustc_v1"),
            concat!("frontend_record", "_bridge"),
            concat!("semantic_type", "_adapter_v2"),
            concat!("source_", "debug"),
            concat!("semantic_", "features"),
            concat!("crate::", "collected_"),
            concat!("collected_", "general_gemm_v1"),
        ] {
            assert!(
                sources
                    .iter()
                    .skip(1)
                    .all(|source| !source.contains(forbidden_dependency)),
                "production importer depends on qualification module {forbidden_dependency:?}"
            );
        }
    }

    #[test]
    fn production_backend_authenticates_target_before_monomorphization() {
        let backend = include_str!("lib.rs");
        let codegen = backend
            .split_once("fn codegen_crate")
            .expect("codegen entry")
            .1;
        let authentication = codegen
            .find("authenticate_before_collection")
            .expect("pre-collection target authentication");
        let monomorphization = codegen
            .find("collect_and_partition_mono_items")
            .expect("rustc monomorphization");
        assert!(authentication < monomorphization);
    }

    #[test]
    fn process_isolated_extraction_uses_the_production_transaction() {
        let driver = include_str!("production_rustc_driver_v1.rs");
        for required in [
            "reject_custom_llvm_configuration",
            "ProductionCompilation::from_collected_device_closure_for_extraction",
            "require_semantic_mir_import",
        ] {
            assert!(
                driver.contains(required),
                "production extraction driver bypassed required transaction step {required:?}",
            );
        }
        for forbidden in [
            "construct_production_semantic_mir_v1",
            "require_production_semantic_import_v1",
        ] {
            assert!(
                !driver.contains(forbidden),
                "production extraction driver directly called importer entry {forbidden:?}",
            );
        }
    }
}

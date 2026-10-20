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

#[path = "production_pipeline/bf16_tile_values_v1.rs"]
mod bf16_tile_values_v1;
#[path = "production_pipeline_checked_output_artifacts_v1.rs"]
mod checked_output_artifacts_v1;
#[path = "production_pipeline_checked_output_policy4_v1.rs"]
pub(crate) mod checked_output_policy4_v1;
#[path = "production_pipeline_checked_output_policy5_v1.rs"]
pub(crate) mod checked_output_policy5_v1;
#[path = "production_pipeline_checked_output_policy6_v1.rs"]
pub(crate) mod checked_output_policy6_v1;
#[path = "production_pipeline_checked_output_policy7_v1.rs"]
pub(crate) mod checked_output_policy7_v1;
#[path = "production_pipeline_checked_output_policy8_v1.rs"]
pub(crate) mod checked_output_policy8_v1;
#[cfg(test)]
#[path = "production_pipeline_checked_output_progress_v1_tests.rs"]
pub(crate) mod checked_output_progress_v1;
#[path = "production_pipeline_conditional_generated_fields_v1.rs"]
pub(crate) mod conditional_generated_fields_v1;
#[path = "production_pipeline_erased_checked_output_policy4_v1.rs"]
pub(crate) mod erased_checked_output_policy4_v1;
#[path = "production_pipeline_erased_checked_output_policy5_v1.rs"]
pub(crate) mod erased_checked_output_policy5_v1;
#[path = "production_pipeline_erased_checked_output_policy6_v1.rs"]
pub(crate) mod erased_checked_output_policy6_v1;
#[path = "production_pipeline_fixed_checked_output_policy6_v1.rs"]
pub(crate) mod fixed_checked_output_policy6_v1;
#[path = "production_pipeline_fixed_checked_output_v1.rs"]
pub(crate) mod fixed_checked_output_v1;
#[path = "production_native_checked_output_handoff_v1.rs"]
pub(crate) mod native_checked_output_handoff_v1;
#[path = "production_pipeline_nominal_abi_v3.rs"]
mod nominal_abi_v3;
#[path = "production_pipeline_nominal_policy4_v3.rs"]
mod nominal_policy4_v3;
#[cfg(test)]
#[path = "production_pipeline_pre_ranked_observation_v1_tests.rs"]
mod pre_ranked_observation_v1;
#[path = "production_pipeline_private_cell_native_v1.rs"]
pub(crate) mod private_cell_native_v1;
#[cfg(test)]
#[path = "production_pipeline_source_abi_v1_tests.rs"]
mod source_abi_v1_tests;
#[path = "production_pipeline_source_local_order_v1.rs"]
pub(crate) mod source_local_order_v1;
#[path = "production_pipeline/tiled_region_v1.rs"]
mod tiled_region_v1;

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
    MixedSourcePublication(Box<source_owned_v29::Error>),
    MixedRuntime(fe2o3_verifier::FunctionalRefinementRuntimeErrorV1),
    CustomLlvmConfiguration,
    EmptyCollectedDeviceClosure,
    SemanticImport(crate::collector::ProductionSemanticImportErrorV1),
    SemanticMiddleEnd(fe2o3_pliron::ProductionSemanticMirErrorV1),
    SemanticSsa(fe2o3_pliron::ProductionSemanticSsaErrorV1),
    ContextHandoff(fe2o3_lower_mir_kernel::ProductionContextRootErrorV29),
    #[cfg(test)]
    PendingScopedSource(fe2o3_lower_mir_kernel::ProductionPendingScopedSourceErrorV29),
    #[cfg(test)]
    PendingScopedObservationIncomplete,
    RankedProjection(crate::production_ranked_projection_v1::ProductionRankedProjectionErrorV1),
    RankedVerification(crate::production_ranked_projection_v1::ProductionRankedVerificationErrorV1),
    TargetNeutralLowering(fe2o3_lower_mir_kernel::ProductionSemanticKirErrorV1),
    PreRankedMaterialization(fe2o3_lower_mir_kernel::ProductionPreRankedKirErrorV1),
    TiledRegionInspection(Box<fe2o3_lower_mir_kernel::ProductionTiledRegionInspectionErrorV1>),
    Bf16TileValuesInspection(Box<fe2o3_lower_mir_kernel::Bf16CallInstanceErrorV1>),
    MissingMirPlironTranslationValidation,
    SimulationKernelIrV7(fe2o3_kernel_ir::VerifiedCanonicalKernelIrErrorV7),
    SimulationBundle(fe2o3_kernel_ir::SimulationBundleErrorV1),
    SimulationDebugMap(fe2o3_kernel_ir::DebugSourceMapErrorV1),
    SimulationBundleV2(fe2o3_kernel_ir::SimulationBundleErrorV2),
    SimulationBundleV3(fe2o3_kernel_ir::SimulationBundleErrorV3),
    SimulationBundleV4(fe2o3_kernel_ir::SimulationBundleErrorV4),
    SimulationBundleV5(fe2o3_kernel_ir::SimulationBundleErrorV5),
    SimulationBundleV6(fe2o3_kernel_ir::SimulationBundleErrorV6),
    SimulationDebugMapV2(fe2o3_kernel_ir::DebugSourceMapErrorV2),
    SimulationDebugSourceCaptureUnavailable(fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1),
    SimulationDebugMapCorrespondence(&'static str),
    SemanticDebugMap(fe2o3_kernel_ir::SemanticDebugMapErrorV1),
    SemanticDebugFragment(fe2o3_kernel_ir::ProductionSemanticDebugFragmentErrorV1),
    SimulationSourceLineage(fe2o3_compiler_lineage::LineageErrorV3),
    SimulationProductionKirV9,
    SimulationProductionKirV11,
    FormalMemoryAdmission(fe2o3_lower_mir_kernel::ProductionFormalMemoryErrorV1),
    Geometry(crate::production_geometry_v1::ProductionGeometryErrorV1),
    TargetBinding(dialect_amdgcn::ProductionTargetBindingErrorV1),
    TargetOptimization(fe2o3_kernel_opt::KernelIrPlironOptimizationErrorV2),
    TargetOptimizationV3(fe2o3_kernel_opt::KernelIrPlironOptimizationErrorV3),
    PrivateBf16CanonicalInput(fe2o3_kernel_ir::CanonicalKernelIrReplayAdmissionErrorV12),
    PrivateBf16Optimization(fe2o3_kernel_opt::KernelIrCheckedOptimizationErrorV1),
    PrivateBf16OptimizationReplay(fe2o3_kernel_opt::KernelIrCheckedOptimizationReceiptErrorV1),
    PrivateBf16NativeLlvm(dialect_amdgcn::PrivateBf16NativeLlvmErrorV1),
    CheckedOutputStage(checked_output_policy4_v1::CheckedOutputStageErrorV1),
    CheckedOutputPolicy5Stage(checked_output_policy5_v1::CheckedOutputPolicy5StageErrorV1),
    CheckedOutputPolicy6Stage(checked_output_policy6_v1::CheckedOutputPolicy6StageErrorV1),
    CheckedOutputPolicy7Stage(checked_output_policy7_v1::CheckedOutputPolicy7StageErrorV1),
    CheckedOutputPolicy8Stage(checked_output_policy8_v1::CheckedOutputPolicy8StageErrorV1),
    SourceLocalOrderStage(source_local_order_v1::SourceLocalOrderStageErrorV1),
    CompleteBodyStage(Box<complete_body_vnext::CompleteBodyStageErrorVNext>),
    PhysicalEntryDiagnosticStage(Box<physical_entry_diagnostic_v20::PhysicalEntryStageErrorV20>),
    PhysicalEntryTargetStage(Box<physical_entry_target_v20::PhysicalEntryTargetStageErrorV20>),
    PhysicalLdsExchangeTargetStage(
        Box<physical_lds_exchange_target_v22::PhysicalLdsExchangeTargetStageErrorV22>,
    ),
    PhysicalGlobalCopyTargetStage(
        Box<physical_global_copy_target_v21::PhysicalGlobalCopyTargetStageErrorV21>,
    ),
    PhysicalGlobalCopyDiagnosticStage(
        Box<physical_global_copy_diagnostic_v21::PhysicalGlobalCopyStageErrorV21>,
    ),
    PhysicalLdsExchangeDiagnosticStage(
        Box<physical_lds_exchange_diagnostic_v22::PhysicalLdsExchangeStageErrorV22>,
    ),
    PrivateCellNativeStage(private_cell_native_v1::PrivateCellNativeStageErrorV1),
    LoopPreheadersNativeStage(private_cell_native_v1::LoopPreheadersNativeStageErrorV1),
    LicmNativeStage(private_cell_native_v1::LicmNativeStageErrorV1),
    InductionRefinementNativeStage(private_cell_native_v1::InductionRefinementNativeStageErrorV1),
    ScalarEmissionCapture(Box<fe2o3_lower_mir_kernel::ProductionScalarSsaEmissionErrorV1>),
    TargetKernelIrV8(fe2o3_kernel_ir::VerifiedCanonicalKernelIrErrorV8),
    TargetKernelIrV9(fe2o3_kernel_ir::VerifiedCanonicalKernelIrErrorV9),
    TargetKernelIrV11(fe2o3_kernel_ir::VerifiedCanonicalKernelIrErrorV11),
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

impl From<Box<ProductionPipelineError>> for ProductionPipelineError {
    fn from(error: Box<Self>) -> Self {
        *error
    }
}

impl fmt::Display for ProductionPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScalarEmissionCapture(error) => write!(formatter, "production scalar emission capture failed: {error}"),
            Self::CheckedOutputStage(error) => write!(formatter, "checked-output production stage failed: {error}"),
            Self::CheckedOutputPolicy5Stage(error) => write!(formatter, "checked Policy5 production stage failed: {error}"),
            Self::CheckedOutputPolicy6Stage(error) => write!(formatter, "checked Policy6 production stage failed: {error}"),
            Self::CheckedOutputPolicy7Stage(error) => write!(formatter, "checked Policy7 production stage failed: {error}"),
            Self::CheckedOutputPolicy8Stage(error) => write!(formatter, "checked Policy8 production stage failed: {error}"),
            Self::SourceLocalOrderStage(error) => write!(formatter, "source local-order stage failed: {error}"),
            Self::CompleteBodyStage(error) => write!(formatter, "complete-body source preparation failed: {error}"),
            Self::PhysicalEntryDiagnosticStage(error) => write!(formatter, "physical-entry pre-ranked diagnostic preparation failed: {error}"),
            Self::PhysicalEntryTargetStage(error) => write!(formatter, "physical-entry checked target preparation failed: {error}"),
            Self::PhysicalLdsExchangeTargetStage(error) => write!(formatter, "physical-lds-exchange checked target preparation failed: {error}"),
            Self::PhysicalGlobalCopyTargetStage(error) => write!(formatter, "physical-global-copy checked target preparation failed: {error}"),
            Self::PhysicalGlobalCopyDiagnosticStage(error) => write!(formatter, "physical-global-copy pre-ranked diagnostic preparation failed: {error}"),
            Self::PhysicalLdsExchangeDiagnosticStage(error) => write!(formatter, "physical-lds-exchange pre-ranked diagnostic preparation failed: {error}"),
            Self::PrivateCellNativeStage(error) => write!(formatter, "private-cell native stage failed: {error}"),
            Self::LoopPreheadersNativeStage(error) => write!(formatter, "loop-preheaders native stage failed: {error}"),
            Self::LicmNativeStage(error) => write!(formatter, "LICM native stage failed: {error}"),
            Self::InductionRefinementNativeStage(error) => write!(formatter, "induction-refinement native stage failed: {error}"),
            Self::CustomLlvmConfiguration => formatter.write_str(
                "production compilation rejects caller-selected LLVM arguments or passes before transaction construction",
            ),
            Self::EmptyCollectedDeviceClosure => formatter.write_str(
                "production compilation requires a nonempty collector-sealed device closure",
            ),
            Self::MixedSourcePublication(error) => write!(formatter, "mandatory mixed source publication failed: {error}"),
            Self::MixedRuntime(error) => write!(formatter, "mandatory mixed refinement runtime unavailable: {error}"),
            Self::SemanticImport(error) => write!(formatter, "production compilation {error}"),
            Self::SemanticMiddleEnd(error) => {
                write!(formatter, "production compilation exact semantic middle end failed: {error}")
            }
            Self::SemanticSsa(error) => {
                write!(formatter, "production compilation semantic SSA planning failed: {error}")
            }
            Self::ContextHandoff(error) => write!(formatter, "production compilation context root handoff failed: {error}"),
            #[cfg(test)]
            Self::PendingScopedSource(error) => write!(formatter, "production pending scoped observation failed: {error}"),
            #[cfg(test)]
            Self::PendingScopedObservationIncomplete => formatter.write_str("pending scoped observation is incomplete and grants no execution authority"),
            Self::RankedProjection(error) => {
                write!(formatter, "production compilation general kernel verification failed: {error}")
            }
            Self::RankedVerification(error) => {
                write!(formatter, "production compilation ranked verification failed: {error}")
            }
            Self::PreRankedMaterialization(error) => {
                write!(formatter, "production compilation pre-ranked materialization failed: {error}")
            }
            Self::TiledRegionInspection(error) => write!(formatter, "production pre-ranked source inspection failed: {error}"),
            Self::Bf16TileValuesInspection(error) => write!(formatter, "production pre-ranked source inspection failed: {error}"),
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
            Self::SimulationBundleV3(error) => write!(
                formatter,
                "production compilation simulation bundle V3 failed: {error}"
            ),
            Self::SimulationBundleV4(error) => write!(
                formatter,
                "production compilation simulation bundle V4 failed: {error}"
            ),
            Self::SimulationBundleV5(error) => write!(
                formatter,
                "production compilation simulation bundle V5 failed: {error}"
            ),
            Self::SimulationBundleV6(error) => write!(
                formatter,
                "production compilation simulation bundle V6 failed: {error}"
            ),
            Self::SimulationDebugMapV2(error) => write!(
                formatter,
                "production compilation simulation debug map V2 failed: {error}"
            ),
            Self::SimulationDebugSourceCaptureUnavailable(gap) => write!(
                formatter,
                "production compilation simulation bundle V2 source-variable names and observations are incomplete ({gap:?}); explicit V2 export fails closed"
            ),
            Self::SimulationDebugMapCorrespondence(detail) => write!(
                formatter,
                "production compilation simulation debug-map correspondence failed: {detail}"
            ),
            Self::SemanticDebugMap(error) => write!(
                formatter,
                "production semantic debug map construction failed: {error}"
            ),
            Self::SemanticDebugFragment(error) => write!(
                formatter,
                "production semantic debug fragment construction failed: {error}"
            ),
            Self::SimulationSourceLineage(error) => write!(
                formatter,
                "production compilation simulation source-lineage receipt failed: {error}"
            ),
            Self::SimulationProductionKirV9 => formatter.write_str(
                "production Kernel IR V9 is not representable by the exact V7 CPU simulator; no downgrade or hardware fallback was attempted",
            ),
            Self::SimulationProductionKirV11 => formatter.write_str(
                "production Kernel IR V11 is not representable by the exact V7/V10 simulation projections; no downgrade or hardware fallback was attempted",
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
            Self::TargetOptimization(error) => write!(
                formatter,
                "production compilation target-KIR optimization failed: {error}"
            ),
            Self::TargetOptimizationV3(error) => write!(
                formatter,
                "production compilation target-KIR V11 optimization failed: {error}"
            ),
            Self::PrivateBf16CanonicalInput(error) => write!(formatter, "private BF16 V12 input admission failed: {error}"),
            Self::PrivateBf16Optimization(error) => write!(formatter, "private BF16 checked V12 optimization failed: {error}"),
            Self::PrivateBf16OptimizationReplay(error) => write!(formatter, "private BF16 checked V12 replay failed: {error}"),
            Self::PrivateBf16NativeLlvm(error) => write!(formatter, "private BF16 native LLVM failed: {error}"),
            Self::TargetKernelIrV8(error) => write!(
                formatter,
                "production compilation target-bound Kernel IR V8 identity failed: {error}"
            ),
            Self::TargetKernelIrV9(error) => write!(
                formatter,
                "production compilation target-bound Kernel IR V9 identity failed: {error}"
            ),
            Self::TargetKernelIrV11(error) => write!(
                formatter,
                "production compilation target-bound Kernel IR V11 identity failed: {error}"
            ),
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
            Self::ScalarEmissionCapture(error) => Some(error.as_ref()),
            Self::MixedSourcePublication(error) => Some(error.as_ref()),
            Self::MixedRuntime(error) => Some(error),
            Self::SemanticImport(error) => Some(error),
            Self::SemanticMiddleEnd(error) => Some(error),
            Self::SemanticSsa(error) => Some(error),
            Self::ContextHandoff(error) => Some(error),
            #[cfg(test)]
            Self::PendingScopedSource(error) => Some(error),
            #[cfg(test)]
            Self::PendingScopedObservationIncomplete => None,
            Self::RankedProjection(error) => Some(error),
            Self::RankedVerification(error) => Some(error),
            Self::TargetNeutralLowering(error) => Some(error),
            Self::PreRankedMaterialization(error) => Some(error),
            Self::TiledRegionInspection(error) => Some(error.as_ref()),
            Self::Bf16TileValuesInspection(error) => Some(error.as_ref()),
            Self::SimulationKernelIrV7(error) => Some(error),
            Self::SimulationBundle(error) => Some(error),
            Self::SimulationDebugMap(error) => Some(error),
            Self::SimulationBundleV2(error) => Some(error),
            Self::SimulationBundleV3(error) => Some(error),
            Self::SimulationBundleV4(error) => Some(error),
            Self::SimulationBundleV5(error) => Some(error),
            Self::SimulationBundleV6(error) => Some(error),
            Self::SimulationDebugMapV2(error) => Some(error),
            Self::SemanticDebugMap(error) => Some(error),
            Self::SemanticDebugFragment(error) => Some(error),
            Self::SimulationSourceLineage(error) => Some(error),
            Self::FormalMemoryAdmission(error) => Some(error),
            Self::Geometry(error) => Some(error),
            Self::TargetBinding(error) => Some(error),
            Self::TargetOptimization(error) => Some(error),
            Self::TargetOptimizationV3(error) => Some(error),
            Self::PrivateBf16CanonicalInput(error) => Some(error),
            Self::PrivateBf16Optimization(error) => Some(error),
            Self::PrivateBf16OptimizationReplay(error) => Some(error),
            Self::PrivateBf16NativeLlvm(error) => Some(error),
            Self::CheckedOutputStage(error) => Some(error),
            Self::CheckedOutputPolicy5Stage(error) => Some(error),
            Self::CheckedOutputPolicy6Stage(error) => Some(error),
            Self::CheckedOutputPolicy7Stage(error) => Some(error),
            Self::CheckedOutputPolicy8Stage(error) => Some(error),
            Self::SourceLocalOrderStage(error) => Some(error),
            Self::CompleteBodyStage(error) => Some(error.as_ref()),
            Self::PhysicalEntryDiagnosticStage(error) => Some(error.as_ref()),
            Self::PhysicalEntryTargetStage(error) => Some(error.as_ref()),
            Self::PhysicalGlobalCopyTargetStage(error) => Some(error.as_ref()),
            Self::PhysicalLdsExchangeTargetStage(error) => Some(error.as_ref()),
            Self::PhysicalGlobalCopyDiagnosticStage(error) => Some(error.as_ref()),
            Self::PhysicalLdsExchangeDiagnosticStage(error) => Some(error.as_ref()),
            Self::PrivateCellNativeStage(error) => Some(error),
            Self::LoopPreheadersNativeStage(error) => Some(error),
            Self::LicmNativeStage(error) => Some(error),
            Self::InductionRefinementNativeStage(error) => Some(error),
            Self::TargetKernelIrV8(error) => Some(error),
            Self::TargetKernelIrV9(error) => Some(error),
            Self::TargetKernelIrV11(error) => Some(error),
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
            | Self::RustcLineageMismatch
            | Self::SimulationProductionKirV9
            | Self::SimulationProductionKirV11
            | Self::SimulationDebugSourceCaptureUnavailable(_)
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

impl<'tcx> ProductionCompilation<'tcx, CollectedRustStage<'tcx>> {
    #[cfg(test)]
    pub(crate) fn collected_root_count_for_test_v259(&self) -> usize {
        self.stage.closure.collected_root_count_for_test_v259()
    }

    /// Borrows the single collection's authenticated terminal observation on the
    /// existing account. No MIR query, importer selection, or authority grant.
    pub(crate) fn collected_tile_terminals_v259(
        &self,
        budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<
        &crate::collector::CollectedTileTerminalCensusV259,
        fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1,
    > {
        self.stage.closure.collected_tile_terminals_v259(budget)
    }
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
        attempt: BuildAttempt,
    },
    ExtractionOnly,
}

// The live execution session stays outside compiler stage owners. Publication
// must consume it explicitly after preparation; native sessions can therefore
// lend the original account to preparation without embedding a self-borrow.

impl ProductionCompilerCustody {
    fn protected(invocation: AdmittedProtectedRustcInvocationV1, attempt: BuildAttempt) -> Self {
        Self::ProtectedV3 {
            invocation: Box::new(invocation),
            attempt,
        }
    }

    const fn extraction_only() -> Self {
        Self::ExtractionOnly
    }

    fn retained_protected_binding_count(&self) -> usize {
        match self {
            Self::ProtectedV3 { .. } => 1,
            Self::ExtractionOnly => 0,
        }
    }

    fn is_extraction_only(&self) -> bool {
        matches!(self, Self::ExtractionOnly)
    }

    fn proof_runtime(
        &self,
    ) -> Result<
        Option<&std::sync::Arc<fe2o3_verifier::FunctionalRefinementVerusRuntimeLeaseV1>>,
        ProductionPipelineError,
    > {
        match self {
            Self::ProtectedV3 { invocation, .. } => invocation
                .proof_runtime()
                .map_err(ProductionPipelineError::ProtectedRustcInvocation),
            Self::ExtractionOnly => Ok(None),
        }
    }

    fn into_publication_custody(
        self,
    ) -> Result<ProtectedProductionPublicationCustody, ProductionPipelineError> {
        match self {
            Self::ProtectedV3 {
                invocation,
                attempt,
            } => Ok(ProtectedProductionPublicationCustody {
                attempt,
                invocation,
            }),
            Self::ExtractionOnly => Err(ProductionPipelineError::ExtractionCannotPublish),
        }
    }
}

struct ProtectedProductionPublicationCustody {
    attempt: BuildAttempt,
    invocation: Box<AdmittedProtectedRustcInvocationV1>,
}

struct AuthenticatedProductionBindings {
    context_entries: crate::collector::RetainedContextEntriesV29,
    rustc_identity_inventory: crate::collector::AuthenticatedRustcIdentityInventoryV3,
    rustc_preflight_plan: crate::collector::AuthenticatedRustcPreflightPlanV3,
    rustc_target: crate::production_target_v1::AuthenticatedProductionTargetV1,
    reference_effect_bindings: crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
    debug_source_files: Box<[fe2o3_kernel_ir::DebugSourceMapFileV1]>,
    debug_source_scopes: Box<[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2]>,
    debug_source_variables: Box<[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2]>,
    debug_capture_gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
    typed_descriptor_roots: Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
    transaction: ProductionTransactionBindings,
}

#[path = "production_pipeline/formal_envelope_preflight_v2.rs"]
mod formal_envelope_preflight_v2;

#[path = "production_pipeline/physical_target_context_v2.rs"]
mod physical_target_context_v2;

pub(super) struct AdmittedSemanticMirStage {
    semantic_mir: fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1,
    bindings: AuthenticatedProductionBindings,
}

pub(super) struct EquivalentSemanticMirStage {
    semantic_mir: fe2o3_pliron::ProductionSemanticMirOwnerV1,
    bindings: AuthenticatedProductionBindings,
}

pub(super) struct SsaSemanticMirStage {
    semantic_ssa: fe2o3_pliron::ProductionSemanticSsaOwnerV1,
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

/// Exact executable graph and source launch custody, before ranked checks.
struct MaterializedNeutralProductionCompilation {
    materialized: fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    ranked_roots: Vec<crate::production_ranked_projection_v1::ProductionRankedRootInputV1>,
    bindings: AuthenticatedProductionBindings,
}

struct PreparedMaterializationV29<M> {
    materialized: M,
    ranked_roots: Vec<crate::production_ranked_projection_v1::ProductionRankedRootInputV1>,
    bindings: AuthenticatedProductionBindings,
}

struct PreparedSsaMaterializationV29 {
    semantic_ssa: fe2o3_pliron::ProductionSemanticSsaOwnerV1,
    ranked_roots: Vec<crate::production_ranked_projection_v1::ProductionRankedRootInputV1>,
    launch: fe2o3_lower_mir_kernel::ProductionSourceLaunchRosterV1,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only production stage retaining ranked checks and the same executable graph.
pub(crate) struct RankedVerifiedProductionCompilation {
    ranked: crate::production_ranked_projection_v1::ProductionRankedSemanticProgramV1,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only production stage retaining exact semantic ownership, verified
/// Kernel IR, correspondence evidence, and the original transaction bindings.
pub(crate) struct TargetNeutralProductionCompilation {
    lowered: fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    ranked_verification:
        crate::production_ranked_projection_v1::AuthenticatedRankedVerificationRosterV1,
    bindings: AuthenticatedProductionBindings,
}

/// Move-only production stage retaining exact semantic ownership, verified
/// Kernel IR, composed formal/ranked memory evidence, and transaction bindings.
pub(crate) struct FormalMemoryAdmittedProductionCompilation {
    admitted: fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
    ranked_verification:
        crate::production_ranked_projection_v1::AuthenticatedRankedVerificationRosterV1,
    bindings: AuthenticatedProductionBindings,
}

#[path = "production_pipeline/retained_target_v30.rs"]
mod retained_target_v30;
pub(crate) use retained_target_v30::RetainedProductionTargetV30;

/// Move-only production stage retaining formal admission, exact target-bound
/// Kernel IR, deterministic exact-target LLVM text, and transaction bindings.
pub(crate) struct TargetLoweredProductionCompilation {
    admitted: fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
    ranked_verification:
        crate::production_ranked_projection_v1::AuthenticatedRankedVerificationRosterV1,
    optimized: RetainedProductionTargetV30,
    workgroups: Box<[(String, fe2o3_kernel_ir::WorkgroupSize)]>,
    llvm_ir: String,
    bindings: AuthenticatedProductionBindings,
}

/// Private handoff input that can only be constructed by the exact production
/// target-lowering stage. It grants no publication or artifact authority.
pub(crate) struct AuthenticatedProductionTargetModule {
    admitted: fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1,
    target: fe2o3_compiler_ffi::DeviceTargetV1,
    optimized: RetainedProductionTargetV30,
    llvm_ir: String,
    typed_descriptor_roots: Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
    compiler_ffi_envelope: Option<fe2o3_compiler_ffi::CompilerFfiEnvelopeV1>,
}

fn exact_target_workgroup_roster_v1(
    module: &fe2o3_kernel_ir::Module,
) -> Result<Box<[(String, fe2o3_kernel_ir::WorkgroupSize)]>, ProductionPipelineError> {
    if module.kernels.is_empty() {
        return Err(ProductionPipelineError::Geometry(
            crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
        ));
    }
    module
        .kernels
        .iter()
        .map(|kernel| {
            kernel
                .workgroup_size
                .map(|workgroup| (kernel.id.as_str().to_owned(), workgroup))
                .ok_or(ProductionPipelineError::Geometry(
                    crate::production_geometry_v1::ProductionGeometryErrorV1::NonExactDescriptorWorkgroup,
                ))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

struct PreparedProductionWorkerPublication {
    producer: ProducerIdentity,
    output_dir: PathBuf,
    attempt: BuildAttempt,
    invocation: Box<AdmittedProtectedRustcInvocationV1>,
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
        RetainedProductionTargetV30,
        String,
        Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
        Option<fe2o3_compiler_ffi::CompilerFfiEnvelopeV1>,
    ) {
        (
            self.admitted,
            self.target,
            self.optimized,
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
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V11 => {
                return Err(ProductionPipelineError::SimulationProductionKirV11);
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
        let debug_map = compiler_debug_source_map_v1(
            &lowered,
            &bindings.debug_source_files,
            prepared.debug_source_map_binding(),
        )?;
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
        require_complete_simulation_debug_source_capture_v2(bindings.debug_capture_gap)?;
        let debug_map = compiler_debug_source_map_v2(
            &lowered,
            &bindings.debug_source_files,
            &bindings.debug_source_scopes,
            &bindings.debug_source_variables,
            prepared.debug_source_map_binding(),
        )?;
        let inner = prepared
            .finalize_without_source_map()
            .map_err(ProductionPipelineError::SimulationBundle)?;
        fe2o3_kernel_ir::VerifiedSimulationBundleV2::new(inner, debug_map)
            .map_err(ProductionPipelineError::SimulationBundleV2)
    }

    fn into_prepared_simulation_bundle_v3(
        self,
        compiler_execution_binding: fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1,
    ) -> Result<
        (
            fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
            AuthenticatedProductionBindings,
            fe2o3_kernel_ir::VerifiedSimulationBundleV3,
        ),
        ProductionPipelineError,
    > {
        let (lowered, bindings, prepared) =
            self.into_prepared_simulation_bundle_v1(compiler_execution_binding)?;
        require_complete_simulation_debug_source_capture_v2(bindings.debug_capture_gap)?;
        let debug_map = compiler_debug_source_map_v2(
            &lowered,
            &bindings.debug_source_files,
            &bindings.debug_source_scopes,
            &bindings.debug_source_variables,
            prepared.debug_source_map_binding(),
        )?;
        let inner_v1 = prepared
            .finalize_without_source_map()
            .map_err(ProductionPipelineError::SimulationBundle)?;
        let inner_v2 = fe2o3_kernel_ir::VerifiedSimulationBundleV2::new(inner_v1, debug_map)
            .map_err(ProductionPipelineError::SimulationBundleV2)?;
        let semantic = lowered.semantic().semantic();
        let mut semantic_mir = Vec::new();
        semantic_mir
            .try_reserve_exact(semantic.canonical_encoding().len())
            .map_err(|_| {
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "semantic MIR bundle allocation failed",
                )
            })?;
        semantic_mir.extend_from_slice(semantic.canonical_encoding());
        let storage_map = compiler_semantic_storage_map_v1(
            &lowered,
            &bindings.debug_source_variables,
            SemanticStorageMapBindingInputV1 {
                container_identity: *inner_v2.identity().as_bytes(),
                subject_identity: *inner_v2.subject_identity(),
                canonical_kir_digest: *inner_v2.canonical_kir_v7_identity().digest(),
                canonical_kir_bytes: inner_v2.canonical_kir_v7_identity().canonical_length(),
            },
        )?;
        let bundle =
            fe2o3_kernel_ir::VerifiedSimulationBundleV3::new(inner_v2, semantic_mir, storage_map)
                .map_err(ProductionPipelineError::SimulationBundleV3)?;
        Ok((lowered, bindings, bundle))
    }

    fn into_simulation_bundle_v3(
        self,
        compiler_execution_binding: fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV3, ProductionPipelineError> {
        let (_, _, bundle) = self.into_prepared_simulation_bundle_v3(compiler_execution_binding)?;
        Ok(bundle)
    }

    fn into_simulation_bundle_v4(
        self,
        compiler_execution_binding: fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV4, ProductionPipelineError> {
        let (lowered, _, inner) =
            self.into_prepared_simulation_bundle_v3(compiler_execution_binding)?;
        let storage_map = compiler_semantic_storage_map_v2(&lowered, *inner.identity().as_bytes())?;
        fe2o3_kernel_ir::VerifiedSimulationBundleV4::new(inner, storage_map)
            .map_err(ProductionPipelineError::SimulationBundleV4)
    }

    fn into_simulation_bundle_v5(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV5, ProductionPipelineError> {
        let Self {
            lowered,
            ranked_verification: _,
            bindings,
        } = self;
        lowered
            .verify_equivalence()
            .map_err(ProductionPipelineError::TargetNeutralLowering)?;
        require_complete_simulation_debug_source_capture_v2(bindings.debug_capture_gap)?;
        if bindings
            .rustc_preflight_plan
            .rustc_identity_inventory_sha256()
            != bindings.rustc_identity_inventory.sha256()
        {
            return Err(ProductionPipelineError::RustcLineageMismatch);
        }
        let production = lowered.canonical_kernel_ir_identity();
        let production = fe2o3_kernel_ir::SimulationProductionKirIdentityV5::new(
            match production.version() {
                fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V8 => 8,
                fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V9 => 9,
                fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V11 => {
                    return Err(ProductionPipelineError::SimulationProductionKirV11);
                }
            },
            *production.digest(),
            production.canonical_length(),
        )
        .map_err(ProductionPipelineError::SimulationBundleV5)?;
        let canonical_v10 =
            fe2o3_kernel_ir::VerifiedCanonicalKernelIrV10::from_module(lowered.module().clone())
                .map_err(|error| {
                    ProductionPipelineError::SimulationBundleV5(
                        fe2o3_kernel_ir::SimulationBundleErrorV5::CanonicalKir(error),
                    )
                })?;
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
        let prepared = fe2o3_kernel_ir::PreparedSimulationBundleV5::new(
            lineage,
            production,
            bindings.rustc_target.profile().device_target(),
            canonical_v10,
        )
        .map_err(ProductionPipelineError::SimulationBundleV5)?;
        let debug_map = compiler_debug_source_map_v2(
            &lowered,
            &bindings.debug_source_files,
            &bindings.debug_source_scopes,
            &bindings.debug_source_variables,
            prepared.debug_source_map_binding(),
        )?;
        let semantic = lowered.semantic().semantic();
        let mut semantic_mir = Vec::new();
        semantic_mir
            .try_reserve_exact(semantic.canonical_encoding().len())
            .map_err(|_| {
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "semantic MIR V5 bundle allocation failed",
                )
            })?;
        semantic_mir.extend_from_slice(semantic.canonical_encoding());
        let subject = *prepared.subject_identity();
        let kir_digest = *prepared.canonical_kir_v10_digest();
        let kir_bytes = prepared.canonical_kir_v10_length();
        let legacy_storage = compiler_semantic_storage_map_v1(
            &lowered,
            &bindings.debug_source_variables,
            SemanticStorageMapBindingInputV1 {
                container_identity: subject,
                subject_identity: subject,
                canonical_kir_digest: kir_digest,
                canonical_kir_bytes: kir_bytes,
            },
        )?;
        let storage = fe2o3_kernel_ir::SemanticStorageMapV5::new(
            subject,
            semantic.wire_version().as_u16(),
            *semantic.semantic_sha256().as_bytes(),
            semantic.canonical_encoding().len() as u64,
            *semantic.target_layout_identity().as_bytes(),
            kir_digest,
            kir_bytes,
            legacy_storage.kernels().to_vec(),
            legacy_storage.variables().to_vec(),
        )
        .map_err(ProductionPipelineError::SimulationBundleV5)?;
        let legacy_aggregate = compiler_semantic_storage_map_v2(&lowered, subject)?;
        let aggregate = fe2o3_kernel_ir::SemanticAggregateStorageMapV5::new(
            subject,
            kir_digest,
            kir_bytes,
            legacy_aggregate.kernels().to_vec(),
        )
        .map_err(ProductionPipelineError::SimulationBundleV5)?;
        prepared
            .finalize(debug_map, semantic_mir, storage, aggregate)
            .map_err(ProductionPipelineError::SimulationBundleV5)
    }

    fn into_simulation_bundle_v6(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV6, ProductionPipelineError> {
        let Self {
            lowered,
            ranked_verification: _,
            bindings,
        } = self;
        lowered
            .verify_equivalence()
            .map_err(ProductionPipelineError::TargetNeutralLowering)?;
        require_complete_simulation_debug_source_capture_v2(bindings.debug_capture_gap)?;
        if bindings
            .rustc_preflight_plan
            .rustc_identity_inventory_sha256()
            != bindings.rustc_identity_inventory.sha256()
        {
            return Err(ProductionPipelineError::RustcLineageMismatch);
        }
        let canonical_v11 =
            fe2o3_kernel_ir::VerifiedCanonicalKernelIrV11::from_module(lowered.module().clone())
                .map_err(|error| {
                    ProductionPipelineError::SimulationBundleV6(
                        fe2o3_kernel_ir::SimulationBundleErrorV6::CanonicalKir(error),
                    )
                })?;
        let production = fe2o3_kernel_ir::SimulationProductionKirIdentityV6::new(
            11,
            *canonical_v11.identity().digest(),
            canonical_v11.identity().canonical_length(),
        )
        .map_err(ProductionPipelineError::SimulationBundleV6)?;
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
        let prepared = fe2o3_kernel_ir::PreparedSimulationBundleV6::new(
            lineage,
            production,
            bindings.rustc_target.profile().device_target(),
            canonical_v11,
        )
        .map_err(ProductionPipelineError::SimulationBundleV6)?;
        let debug_map = compiler_debug_source_map_v2(
            &lowered,
            &bindings.debug_source_files,
            &bindings.debug_source_scopes,
            &bindings.debug_source_variables,
            prepared.debug_source_map_binding(),
        )?;
        let semantic = lowered.semantic().semantic();
        let mut semantic_mir = Vec::new();
        semantic_mir
            .try_reserve_exact(semantic.canonical_encoding().len())
            .map_err(|_| {
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "semantic MIR V6 bundle allocation failed",
                )
            })?;
        semantic_mir.extend_from_slice(semantic.canonical_encoding());
        let subject = *prepared.subject_identity();
        let kir_digest = *prepared.canonical_kir_v11_digest();
        let kir_bytes = prepared.canonical_kir_v11_length();
        let legacy_storage = compiler_semantic_storage_map_v1(
            &lowered,
            &bindings.debug_source_variables,
            SemanticStorageMapBindingInputV1 {
                container_identity: subject,
                subject_identity: subject,
                canonical_kir_digest: kir_digest,
                canonical_kir_bytes: kir_bytes,
            },
        )?;
        let storage = fe2o3_kernel_ir::SemanticStorageMapV6::new(
            subject,
            semantic.wire_version().as_u16(),
            *semantic.semantic_sha256().as_bytes(),
            semantic.canonical_encoding().len() as u64,
            *semantic.target_layout_identity().as_bytes(),
            kir_digest,
            kir_bytes,
            legacy_storage.kernels().to_vec(),
            legacy_storage.variables().to_vec(),
        )
        .map_err(ProductionPipelineError::SimulationBundleV6)?;
        let legacy_aggregate = compiler_semantic_storage_map_v2(&lowered, subject)?;
        let aggregate = fe2o3_kernel_ir::SemanticAggregateStorageMapV6::new(
            subject,
            kir_digest,
            kir_bytes,
            legacy_aggregate.kernels().to_vec(),
        )
        .map_err(ProductionPipelineError::SimulationBundleV6)?;
        prepared
            .finalize(debug_map, semantic_mir, storage, aggregate)
            .map_err(ProductionPipelineError::SimulationBundleV6)
    }

    fn admit_formal_memory(
        self,
    ) -> Result<FormalMemoryAdmittedProductionCompilation, ProductionPipelineError> {
        self.admit_formal_memory_with_translation_budget_v1(None)
    }

    // Formal obligation derivation is unchanged and is NOT charged here.
    fn admit_formal_memory_with_translation_budget_v1(
        self,
        budget: Option<&mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>>,
    ) -> Result<FormalMemoryAdmittedProductionCompilation, ProductionPipelineError> {
        let Self {
            lowered,
            ranked_verification,
            bindings,
        } = self;
        let envelopes = formal_envelope_preflight_v2::authenticated_envelopes(&lowered, &bindings)?;
        let admitted = match budget {
            Some(budget) => fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1::
                try_admit_for_launch_envelopes_with_bounded_translation_budget_v2(lowered, &envelopes, budget),
            None => fe2o3_lower_mir_kernel::ProductionFormalMemoryOwnerV1::
                try_admit_for_launch_envelopes_v2(lowered, &envelopes),
        }.map_err(ProductionPipelineError::FormalMemoryAdmission)?;
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
        if semantic.roots().is_empty()
            || semantic.roots().len() != bindings.typed_descriptor_roots.len()
            || semantic.roots().len() != admitted.semantic_kir().module().kernels.len()
            || semantic.roots().len() != admitted.kernels().len()
        {
            return Err(ProductionPipelineError::Geometry(
                crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
            ));
        }
        let mut physical_geometry = Vec::with_capacity(semantic.roots().len());
        for (((typed_root, semantic_root), kernel), formal) in bindings
            .typed_descriptor_roots
            .iter()
            .zip(semantic.roots())
            .zip(admitted.semantic_kir().module().kernels.iter())
            .zip(admitted.kernels())
        {
            let semantic_function = semantic
                .functions()
                .get(semantic_root.index() as usize)
                .ok_or(ProductionPipelineError::Geometry(
                    crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
                ))?;
            let semantic_entry =
                semantic_function
                    .kernel_entry()
                    .ok_or(ProductionPipelineError::Geometry(
                        crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
                    ))?;
            if semantic_entry.kernel_binding_identity().as_bytes()
                != &typed_root.kernel_binding_bytes()
                || semantic_entry.export_symbol().as_bytes() != typed_root.entry_symbol().as_bytes()
                || kernel.id.as_str() != typed_root.entry_symbol()
                || kernel.entry.as_str() != typed_root.entry_symbol()
                || formal.obligations().kernel().as_str() != typed_root.entry_symbol()
            {
                return Err(ProductionPipelineError::Geometry(
                    crate::production_geometry_v1::ProductionGeometryErrorV1::KernelClosure,
                ));
            }
            let source_launch = typed_root.source_launch().ok_or(
                ProductionPipelineError::Geometry(
                    crate::production_geometry_v1::ProductionGeometryErrorV1::NonExactDescriptorWorkgroup,
                ),
            )?;
            let geometry = crate::production_geometry_v1::derive_production_geometry_v1(
                admitted.semantic_kir().module(),
                typed_root.entry_symbol(),
                semantic_function,
                source_launch,
                target_profile.device_target(),
            )
            .map_err(ProductionPipelineError::Geometry)?;
            physical_geometry.push(geometry);
        }
        let optimized = RetainedProductionTargetV30::try_lower(&admitted, target_profile)?;
        let target_module = optimized.module();
        let physical_context = physical_target_context_v2::bind(
            admitted.semantic_kir().module(),
            target_module,
            target_profile,
            &physical_geometry,
        )?;
        let workgroups = exact_target_workgroup_roster_v1(target_module)?;
        let target_kir_identity = match admitted
            .semantic_kir()
            .canonical_kernel_ir_identity()
            .version()
        {
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V8 => {
                let owner = fe2o3_kernel_ir::VerifiedCanonicalKernelIrV8::from_module(
                    target_module.clone(),
                )
                .map_err(ProductionPipelineError::TargetKernelIrV8)?;
                dialect_amdgcn::ProductionSemanticAnchorKirIdentityV1::from_v8(&owner)
            }
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V9 => {
                let owner = fe2o3_kernel_ir::VerifiedCanonicalKernelIrV9::from_module(
                    target_module.clone(),
                )
                .map_err(ProductionPipelineError::TargetKernelIrV9)?;
                dialect_amdgcn::ProductionSemanticAnchorKirIdentityV1::from_v9(&owner)
            }
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V11 => {
                let owner = fe2o3_kernel_ir::VerifiedCanonicalKernelIrV11::from_module(
                    target_module.clone(),
                )
                .map_err(ProductionPipelineError::TargetKernelIrV11)?;
                dialect_amdgcn::ProductionSemanticAnchorKirIdentityV1::from_v11(&owner)
            }
        };
        let lowering =
            dialect_amdgcn::lower_compiler_module_to_xnack_minus_llvm_ir_with_physical_launch_v2(
                target_module,
                target_kir_identity,
                &physical_context,
            );
        let dialect_llvm_ir = lowering.map_err(ProductionPipelineError::TargetLowering)?;
        let llvm_ir = dialect_amdgcn::bind_production_llvm22_worker_layout_v1(&dialect_llvm_ir)
            .map_err(ProductionPipelineError::UpstreamLlvmLayoutBinding)?;
        Ok(TargetLoweredProductionCompilation {
            admitted,
            ranked_verification,
            optimized,
            workgroups,
            llvm_ir,
            bindings,
        })
    }
}

impl TargetLoweredProductionCompilation {
    pub(crate) fn module(&self) -> &fe2o3_kernel_ir::Module {
        self.optimized.module()
    }

    pub(crate) fn target_name(&self) -> &'static str {
        self.bindings.rustc_target.profile().device_target()
    }

    pub(crate) fn canonical_kernel_ir_version(&self) -> u16 {
        match self
            .admitted
            .semantic_kir()
            .canonical_kernel_ir_identity()
            .version()
        {
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V8 => 8,
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V9 => 9,
            fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V11 => 11,
        }
    }

    pub(crate) fn guarded_store_count(&self) -> usize {
        self.optimized
            .module()
            .functions
            .iter()
            .filter_map(|function| function.body.as_ref())
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .filter(|operation| {
                matches!(
                    operation.kind,
                    fe2o3_kernel_ir::OperationKind::GuardedStore { .. }
                )
            })
            .count()
    }

    pub(crate) fn llvm_ir(&self) -> &str {
        &self.llvm_ir
    }

    pub(crate) fn workgroup_sizes(&self) -> &[(String, fe2o3_kernel_ir::WorkgroupSize)] {
        &self.workgroups
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
        self.ranked_verification.checked_additions_examined()
    }

    pub(crate) fn semantic_u32_induction_certificate_count(&self) -> usize {
        self.ranked_verification.induction_certificate_count()
    }

    pub(crate) fn correspondence_block_count(&self) -> usize {
        self.admitted.semantic_kir().correspondence().blocks().len()
    }

    pub(crate) fn formal_witness_extent(&self) -> u64 {
        self.admitted.witness_extent()
    }

    pub(crate) fn formal_allocation_count(&self) -> usize {
        self.admitted
            .kernels()
            .iter()
            .map(|kernel| kernel.obligations().allocations().len())
            .sum()
    }

    pub(crate) fn formal_access_count(&self) -> usize {
        self.admitted
            .kernels()
            .iter()
            .map(|kernel| kernel.obligations().accesses().len())
            .sum()
    }

    pub(crate) fn ranked_dynamic_index_discharge_count(&self) -> usize {
        self.admitted
            .kernels()
            .iter()
            .map(|kernel| kernel.ranked_discharged_reasons().len())
            .sum()
    }

    pub(crate) fn runtime_bounds_requirement_count(&self) -> usize {
        self.admitted
            .kernels()
            .iter()
            .map(|kernel| kernel.obligations().bounds_requirements().len())
            .sum()
    }

    pub(crate) fn runtime_alias_requirement_count(&self) -> usize {
        self.admitted
            .kernels()
            .iter()
            .map(|kernel| kernel.obligations().runtime_alias_requirements().len())
            .sum()
    }

    pub(crate) fn inter_invocation_conflict_count(&self) -> usize {
        self.admitted
            .kernels()
            .iter()
            .map(|kernel| kernel.obligations().inter_invocation_conflicts().len())
            .sum()
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

    pub(crate) fn target_optimization_pass_count(&self) -> usize {
        self.optimized.report().passes().len()
    }

    pub(crate) fn target_optimization_mutating_pass_count(&self) -> usize {
        self.optimized
            .report()
            .passes()
            .iter()
            .filter(|pass| pass.pliron().changed())
            .count()
    }

    pub(crate) const fn target_optimization_initial_epoch(&self) -> u64 {
        self.optimized.report().initial_epoch()
    }

    pub(crate) const fn target_optimization_final_epoch(&self) -> u64 {
        self.optimized.report().final_epoch()
    }

    pub(crate) fn into_inert_worker_handoff_for_extraction(
        self,
    ) -> Result<fe2o3_compiler_ffi::CompilerModuleHandoffV2, ProductionPipelineError> {
        let Self {
            admitted,
            ranked_verification: _,
            optimized,
            workgroups: _,
            llvm_ir,
            bindings,
        } = self;
        let AuthenticatedProductionBindings {
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings: _,
            context_entries: _,
            debug_source_files: _,
            debug_source_scopes: _,
            debug_source_variables: _,
            debug_capture_gap: _,
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
            optimized,
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

    pub(crate) fn into_inert_semantic_worker_handoff_for_extraction(
        self,
        invocation: fe2o3_rustc_invocation::RustcInvocationDescriptorV3,
    ) -> Result<fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV3, ProductionPipelineError>
    {
        let Self {
            admitted,
            ranked_verification,
            optimized,
            workgroups: _,
            llvm_ir,
            bindings,
        } = self;
        let AuthenticatedProductionBindings {
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings: _,
            context_entries: _,
            debug_source_files,
            debug_source_scopes,
            debug_source_variables,
            debug_capture_gap,
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
        let semantic_debug_inputs = prepare_production_semantic_debug_inputs_v1(
            admitted.semantic_kir(),
            &rustc_identity_inventory,
            &rustc_preflight_plan,
            &rustc_target,
            &debug_source_files,
            &debug_source_scopes,
            &debug_source_variables,
            debug_capture_gap,
        );
        let semantic_lineage =
            crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3::try_prepare(
                &rustc_identity_inventory,
                &rustc_preflight_plan,
                &rustc_target,
                ranked_verification,
                &admitted,
                optimized.module(),
                optimized.report(),
                &llvm_ir,
                semantic_debug_inputs,
            )
            .map_err(ProductionPipelineError::SemanticLineage)?;
        let target = rustc_target.device_target();
        let compiler_module = AuthenticatedProductionTargetModule {
            admitted,
            target,
            optimized,
            llvm_ir,
            typed_descriptor_roots,
            compiler_ffi_envelope: transaction.compiler_ffi_envelope,
        };
        let prepared =
            crate::production_worker_handoff::prepare_production_worker_handoff(compiler_module)
                .map_err(ProductionPipelineError::WorkerHandoff)?;
        let (handoff, descriptor_source) = prepared
            .into_validated_parts()
            .map_err(ProductionPipelineError::WorkerHandoff)?;
        semantic_lineage
            .finish_for_inert_extraction(invocation, target, &descriptor_source, handoff)
            .map_err(ProductionPipelineError::SemanticLineage)
    }

    fn prepare_worker_handoff(
        self,
    ) -> Result<PreparedProductionWorkerPublication, ProductionPipelineError> {
        eprintln!(
            "[rustc-codegen-fe2o3] production compilation lowered {} admitted semantic function(s) into verified target-neutral Kernel IR module `{}` with {} exact block correspondence record(s), then admitted composed formal/ranked memory evidence for a {}-invocation structural witness with {} allocation(s), {} formal access(es), {} ranked dynamic-index discharge(s), {} runtime bounds requirement(s), {} runtime alias requirement(s), and {} inter-invocation conflict(s), applied {} structurally replayed target-KIR optimization pass(es), including {} mutating pass(es), across epoch {}..={} (semantic preservation is not yet formally proved), and lowered exact target-bound KIR with ordered compiler-selected-or-retained workgroups {:?} to {} byte(s) of deterministic {} LLVM text while retaining {} identity/transaction binding(s); artifact/launch authority {}; preparing exact compiler-module handoff",
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
            self.target_optimization_pass_count(),
            self.target_optimization_mutating_pass_count(),
            self.target_optimization_initial_epoch(),
            self.target_optimization_final_epoch(),
            self.workgroup_sizes(),
            self.llvm_ir().len(),
            self.bindings.rustc_target.profile().device_target(),
            self.retained_identity_and_transaction_binding_count(),
            self.grants_artifact_or_launch_authority(),
        );
        let Self {
            admitted,
            ranked_verification,
            optimized,
            workgroups: _,
            llvm_ir,
            bindings,
        } = self;
        let AuthenticatedProductionBindings {
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings,
            context_entries: _,
            debug_source_files,
            debug_source_scopes,
            debug_source_variables,
            debug_capture_gap,
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
        let semantic_debug_inputs = prepare_production_semantic_debug_inputs_v1(
            admitted.semantic_kir(),
            &rustc_identity_inventory,
            &rustc_preflight_plan,
            &rustc_target,
            &debug_source_files,
            &debug_source_scopes,
            &debug_source_variables,
            debug_capture_gap,
        );
        drop(reference_effect_bindings);
        let ProtectedProductionPublicationCustody {
            attempt,
            invocation,
        } = compiler_custody.into_publication_custody()?;
        let semantic_lineage = crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3::try_prepare(
            &rustc_identity_inventory,
            &rustc_preflight_plan,
            &rustc_target,
            ranked_verification,
            &admitted,
            optimized.module(),
            optimized.report(),
            &llvm_ir,
            semantic_debug_inputs,
        )
        .map_err(ProductionPipelineError::SemanticLineage)?;
        let compiler_module = AuthenticatedProductionTargetModule {
            admitted,
            target: rustc_target.device_target(),
            optimized,
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
            semantic_lineage,
            rustc_target,
            prepared,
        })
    }

    fn publish_worker_handoff(
        self,
        compiler_execution: AdmittedProtectedCompilerExecutionV1,
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
        let carriage = compiler_execution
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

fn require_complete_simulation_debug_source_capture_v2(
    gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
) -> Result<(), ProductionPipelineError> {
    match gap {
        None => Ok(()),
        Some(gap) => Err(ProductionPipelineError::SimulationDebugSourceCaptureUnavailable(gap)),
    }
}

struct ExactDebugMapFunctionV1<'a> {
    function_ordinal: u64,
    body: &'a fe2o3_kernel_ir::FunctionBody,
    block_ordinals: BTreeMap<fe2o3_kernel_ir::BlockId, usize>,
}

// Closed borrowed owners, not arbitrary executable/correspondence reconstruction.
#[derive(Clone, Copy)]
enum ExactDebugSourceOwnerV1<'a> {
    Normal(&'a fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1),
    #[cfg(test)]
    Ordered(&'a fe2o3_lower_mir_kernel::ProductionOrderedProgramPreRankedKirOwnerV17),
}

impl<'a> ExactDebugSourceOwnerV1<'a> {
    fn module(self) -> &'a fe2o3_kernel_ir::Module {
        match self {
            Self::Normal(owner) => owner.module(),
            #[cfg(test)]
            Self::Ordered(owner) => owner.executable().module(),
        }
    }
    fn semantic(self) -> &'a fe2o3_pliron::ProductionSemanticMirOwnerV1 {
        match self {
            Self::Normal(owner) => owner.semantic(),
            #[cfg(test)]
            Self::Ordered(owner) => owner.semantic_ssa().source_owner(),
        }
    }
    fn correspondence(self) -> &'a fe2o3_lower_mir_kernel::SemanticKirCorrespondenceV1 {
        match self {
            Self::Normal(owner) => owner.correspondence(),
            #[cfg(test)]
            Self::Ordered(owner) => owner.correspondence(),
        }
    }
}

// Private projection parts, not a wire document, subject, receipt or authority.
struct ExactDebugSourceProjectionV1 {
    files: Vec<fe2o3_kernel_ir::DebugSourceMapFileV1>,
    sites: Vec<fe2o3_kernel_ir::DebugSourceMapSiteV1>,
    eliminated: Vec<fe2o3_kernel_ir::DebugSourceMapSpanV1>,
}

fn exact_debug_map_functions_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
) -> Result<
    BTreeMap<
        (
            fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1,
            fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1,
        ),
        ExactDebugMapFunctionV1<'_>,
    >,
    ProductionPipelineError,
> {
    exact_debug_map_functions_from_owner_v1(ExactDebugSourceOwnerV1::Normal(lowered))
}

fn exact_debug_map_functions_from_owner_v1<'a>(
    lowered: ExactDebugSourceOwnerV1<'a>,
) -> Result<
    BTreeMap<
        (
            fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1,
            fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1,
        ),
        ExactDebugMapFunctionV1<'a>,
    >,
    ProductionPipelineError,
> {
    let defined_count = lowered
        .module()
        .functions
        .iter()
        .filter(|function| function.body.is_some())
        .count();
    if defined_count == 0 || lowered.correspondence().lowered_functions().len() < defined_count {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "live correspondence does not cover every defined KIR function",
        ));
    }
    let mut layouts = BTreeMap::new();
    let mut ordinals = BTreeSet::new();
    let mut physical_functions = BTreeMap::new();
    for record in lowered.correspondence().lowered_functions() {
        let mut matches = lowered
            .module()
            .functions
            .iter()
            .enumerate()
            .filter(|(_, function)| &function.id == record.kernel_ir_function());
        let Some((ordinal, function)) = matches.next() else {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "live correspondence names an unknown KIR function",
            ));
        };
        if matches.next().is_some() || function.body.is_none() {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "live correspondence has an ambiguous KIR function owner",
            ));
        }
        let role_matches = matches!(
            (record.role(), function.role),
            (
                fe2o3_lower_mir_kernel::SemanticKirFunctionRoleV1::KernelEntry,
                fe2o3_kernel_ir::FunctionRole::KernelEntry
            ) | (
                fe2o3_lower_mir_kernel::SemanticKirFunctionRoleV1::InternalHelper,
                fe2o3_kernel_ir::FunctionRole::InternalHelper
            )
        );
        if !role_matches {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "live correspondence KIR function role differs",
            ));
        }
        if let Some((semantic_function, role)) =
            physical_functions.insert(ordinal, (record.semantic_function(), record.role()))
            && (semantic_function != record.semantic_function()
                || role != fe2o3_lower_mir_kernel::SemanticKirFunctionRoleV1::InternalHelper
                || record.role()
                    != fe2o3_lower_mir_kernel::SemanticKirFunctionRoleV1::InternalHelper)
        {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "only one exact semantic helper may share a physical KIR function",
            ));
        }
        ordinals.insert(ordinal);
        let body = function.body.as_ref().expect("body checked");
        let block_ordinals = body
            .blocks
            .iter()
            .enumerate()
            .map(|(block_ordinal, block)| (block.id, block_ordinal))
            .collect::<BTreeMap<_, _>>();
        if block_ordinals.len() != body.blocks.len() {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "KIR body has duplicate block identities",
            ));
        }
        let key = (record.correspondence_owner(), record.semantic_function());
        if layouts
            .insert(
                key,
                ExactDebugMapFunctionV1 {
                    function_ordinal: u64::try_from(ordinal).map_err(|_| {
                        ProductionPipelineError::SimulationDebugMapCorrespondence(
                            "KIR function ordinal does not fit the source-map wire",
                        )
                    })?,
                    body,
                    block_ordinals,
                },
            )
            .is_some()
        {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "live correspondence has a duplicate semantic function owner",
            ));
        }
    }
    if ordinals.len() != defined_count {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "live correspondence omits a defined KIR function",
        ));
    }
    Ok(layouts)
}

fn kernel_storage_map_body_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    selected_root: fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1,
    selected_body: fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1,
) -> Result<(usize, &fe2o3_kernel_ir::FunctionBody), ProductionPipelineError> {
    let layouts = exact_debug_map_functions_v1(lowered)?;
    let mut entries = lowered
        .correspondence()
        .lowered_functions()
        .iter()
        .filter(|record| {
            record.role() == fe2o3_lower_mir_kernel::SemanticKirFunctionRoleV1::KernelEntry
                && record.correspondence_owner() == selected_root
                && record.semantic_function() == selected_body
        });
    let entry = entries
        .next()
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "selected semantic kernel has no exact KIR function correspondence",
        ))?;
    if entries.next().is_some() {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "selected semantic kernel has ambiguous KIR function correspondence",
        ));
    }
    let layout = layouts
        .get(&(entry.correspondence_owner(), entry.semantic_function()))
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "selected semantic kernel has no exact KIR function layout",
        ))?;
    let ordinal = usize::try_from(layout.function_ordinal).map_err(|_| {
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "selected KIR function ordinal does not fit this host",
        )
    })?;
    Ok((ordinal, layout.body))
}

fn prepare_production_semantic_debug_inputs_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    rustc_identity_inventory: &crate::collector::AuthenticatedRustcIdentityInventoryV3,
    rustc_preflight_plan: &crate::collector::AuthenticatedRustcPreflightPlanV3,
    rustc_target: &crate::production_target_v1::AuthenticatedProductionTargetV1,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
    captured_scopes: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2],
    captured_variables: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2],
    capture_gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
) -> crate::production_semantic_debug_v1::ProductionSemanticDebugInputsV1 {
    if let Some(gap) = capture_gap {
        return crate::production_semantic_debug_v1::ProductionSemanticDebugInputsV1::unavailable(
            gap,
        );
    }
    match compiler_production_semantic_debug_source_map_v1(
        lowered,
        rustc_identity_inventory,
        rustc_preflight_plan,
        rustc_target,
        captured_files,
        captured_scopes,
        captured_variables,
    ) {
        Ok((source_map, canonical_kir_v7)) => {
            crate::production_semantic_debug_v1::ProductionSemanticDebugInputsV1::Available {
                source_map: Box::new(source_map),
                canonical_kir_v7,
            }
        }
        Err(
            ProductionPipelineError::SimulationProductionKirV9
            | ProductionPipelineError::SimulationProductionKirV11,
        ) => {
            crate::production_semantic_debug_v1::ProductionSemanticDebugInputsV1::unavailable(
                fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::CanonicalKirV7ProjectionUnavailable,
            )
        }
        Err(
            ProductionPipelineError::SimulationDebugMap(
                fe2o3_kernel_ir::DebugSourceMapErrorV1::InvalidLength
                | fe2o3_kernel_ir::DebugSourceMapErrorV1::ResourceLimit
                | fe2o3_kernel_ir::DebugSourceMapErrorV1::AllocationFailure
                | fe2o3_kernel_ir::DebugSourceMapErrorV1::Encoding,
            )
            | ProductionPipelineError::SimulationDebugMapV2(
                fe2o3_kernel_ir::DebugSourceMapErrorV2::InvalidLength
                | fe2o3_kernel_ir::DebugSourceMapErrorV2::ResourceLimit
                | fe2o3_kernel_ir::DebugSourceMapErrorV2::AllocationFailure
                | fe2o3_kernel_ir::DebugSourceMapErrorV2::Encoding,
            ),
        ) => crate::production_semantic_debug_v1::ProductionSemanticDebugInputsV1::unavailable(
            fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::ResourceLimit,
        ),
        Err(_) => crate::production_semantic_debug_v1::ProductionSemanticDebugInputsV1::unavailable(
            fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::SourceMapUnavailable,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn compiler_production_semantic_debug_source_map_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    rustc_identity_inventory: &crate::collector::AuthenticatedRustcIdentityInventoryV3,
    rustc_preflight_plan: &crate::collector::AuthenticatedRustcPreflightPlanV3,
    rustc_target: &crate::production_target_v1::AuthenticatedProductionTargetV1,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
    captured_scopes: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2],
    captured_variables: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2],
) -> Result<
    (
        fe2o3_kernel_ir::DebugSourceMapDocumentV2,
        fe2o3_kernel_ir::VerifiedCanonicalKernelIrV7,
    ),
    ProductionPipelineError,
> {
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
        fe2o3_lower_mir_kernel::ProductionCanonicalKernelIrVersionV1::V11 => {
            return Err(ProductionPipelineError::SimulationProductionKirV11);
        }
    };
    let canonical_kir =
        fe2o3_kernel_ir::VerifiedCanonicalKernelIrV7::from_module(lowered.module().clone())
            .map_err(ProductionPipelineError::SimulationKernelIrV7)?;
    let mut prepared_kir_bytes = Vec::new();
    prepared_kir_bytes
        .try_reserve_exact(canonical_kir.canonical_bytes().len())
        .map_err(|_| {
            ProductionPipelineError::SemanticDebugFragment(
                fe2o3_kernel_ir::ProductionSemanticDebugFragmentErrorV1::AllocationFailure,
            )
        })?;
    prepared_kir_bytes.extend_from_slice(canonical_kir.canonical_bytes());
    let prepared_kir =
        fe2o3_kernel_ir::VerifiedCanonicalKernelIrV7::from_canonical_bytes(prepared_kir_bytes)
            .map_err(ProductionPipelineError::SimulationKernelIrV7)?;
    let inventory_receipt =
        fe2o3_compiler_lineage::InertRustcIdentityInventoryReceiptV3::from_canonical_preimage(
            rustc_identity_inventory.canonical_transcript(),
        )
        .map_err(ProductionPipelineError::SimulationSourceLineage)?;
    let preflight_receipt =
        fe2o3_compiler_lineage::InertRustcPreflightPlanReceiptV3::from_canonical_preimage(
            rustc_preflight_plan.canonical_transcript(),
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
        fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
        lineage,
        production_identity,
        rustc_target.profile().device_target(),
        prepared_kir,
    )
    .map_err(ProductionPipelineError::SimulationBundle)?;
    let source_map = compiler_debug_source_map_v2(
        lowered,
        captured_files,
        captured_scopes,
        captured_variables,
        prepared.debug_source_map_binding(),
    )?;
    Ok((source_map, canonical_kir))
}

fn compiler_debug_source_map_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
    binding: fe2o3_kernel_ir::DebugSourceMapBindingV1,
) -> Result<fe2o3_kernel_ir::DebugSourceMapDocumentV1, ProductionPipelineError> {
    let projection = compiler_debug_source_projection_v1(
        ExactDebugSourceOwnerV1::Normal(lowered),
        captured_files,
    )?;
    fe2o3_kernel_ir::DebugSourceMapDocumentV1::new(
        binding,
        projection.files,
        projection.sites,
        projection.eliminated,
    )
    .map_err(ProductionPipelineError::SimulationDebugMap)
}

fn compiler_debug_source_projection_v1(
    lowered: ExactDebugSourceOwnerV1<'_>,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
) -> Result<ExactDebugSourceProjectionV1, ProductionPipelineError> {
    let function_layouts = exact_debug_map_functions_from_owner_v1(lowered)?;

    let mut mapped = BTreeMap::new();
    let mut eliminated = BTreeSet::new();
    for span in lowered.correspondence().statement_operation_spans() {
        let layout = function_layouts
            .get(&(span.correspondence_owner(), span.semantic_function()))
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "statement correspondence has no exact KIR function owner",
            ))?;
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
            layout.function_ordinal,
            layout.body,
            &layout.block_ordinals,
            (
                1,
                span.semantic_function().index(),
                span.semantic_block().index(),
                span.statement_ordinal(),
            ),
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
            source,
            &mut mapped,
            &mut eliminated,
        )?;
    }
    for span in lowered.correspondence().terminator_operation_spans() {
        let layout = function_layouts
            .get(&(span.correspondence_owner(), span.semantic_function()))
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "terminator correspondence has no exact KIR function owner",
            ))?;
        let source = lowered
            .semantic()
            .resolve_terminator(span.semantic_function(), span.semantic_block())
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "terminator correspondence does not resolve in retained semantic MIR",
            ))?
            .source();
        insert_debug_operation_range_v1(
            layout.function_ordinal,
            layout.body,
            &layout.block_ordinals,
            (
                2,
                span.semantic_function().index(),
                span.semantic_block().index(),
                0,
            ),
            span.kernel_ir_block(),
            span.first_operation_ordinal(),
            span.operation_count(),
            source,
            &mut mapped,
            &mut eliminated,
        )?;
    }

    let mut synthetic = BTreeMap::new();
    for span in lowered.correspondence().synthetic_operation_spans() {
        let layout = function_layouts
            .get(&(span.correspondence_owner(), span.semantic_function()))
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "synthetic correspondence has no exact KIR function owner",
            ))?;
        let block_ordinal = debug_block_ordinal_v1(
            layout.body,
            &layout.block_ordinals,
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
                layout.function_ordinal,
                block_ordinal,
                u64::from(operation),
            );
            let rule = match span.rule() {
                fe2o3_lower_mir_kernel::SemanticKirSyntheticOperationRuleV1::RetainedLocalStorage => {
                    3
                }
                fe2o3_lower_mir_kernel::SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage => {
                    1
                }
                fe2o3_lower_mir_kernel::SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap => {
                    2
                }
            };
            let owner = (span.semantic_function().index(), rule);
            if mapped.contains_key(&site)
                || synthetic
                    .insert(site, owner)
                    .is_some_and(|previous| previous != owner)
            {
                return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "synthetic and semantic operation ranges overlap",
                ));
            }
        }
    }
    for layout in function_layouts.values() {
        for (block_ordinal, block) in layout.body.blocks.iter().enumerate() {
            for operation_ordinal in 0..block.operations.len() {
                let site = fe2o3_kernel_ir::DebugSourceMapKirSiteV1::operation(
                    layout.function_ordinal,
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
                if mapped.contains_key(&site) == synthetic.contains_key(&site) {
                    return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                        "KIR operation is not covered exactly once by semantic or synthetic correspondence",
                    ));
                }
            }
        }
    }

    let referenced_files = mapped
        .values()
        .map(|(_, span)| span)
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
        .map(|(site, (_, span))| {
            fe2o3_kernel_ir::DebugSourceMapSiteV1::new(site, vec![span])
                .map_err(ProductionPipelineError::SimulationDebugMap)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExactDebugSourceProjectionV1 {
        files,
        sites,
        eliminated: eliminated.into_iter().collect(),
    })
}

fn compiler_debug_source_map_v2(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    captured_files: &[fe2o3_kernel_ir::DebugSourceMapFileV1],
    captured_scopes: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2],
    captured_variables: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2],
    binding: fe2o3_kernel_ir::DebugSourceMapBindingV1,
) -> Result<fe2o3_kernel_ir::DebugSourceMapDocumentV2, ProductionPipelineError> {
    let base = compiler_debug_source_map_v1(lowered, captured_files, binding)?;
    let function_layouts = exact_debug_map_functions_v1(lowered)?;
    let function_by_semantic = exact_debug_function_ordinals_v1(&function_layouts)?;

    let mut parameter_by_local = BTreeMap::new();
    for binding in lowered.correspondence().parameter_bindings() {
        if !function_layouts
            .contains_key(&(binding.correspondence_owner(), binding.semantic_function()))
        {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "KIR parameter correspondence has no exact function instance",
            ));
        }
        let key = (binding.semantic_function(), binding.semantic_local());
        if let Some(previous) = parameter_by_local.insert(key, binding.kernel_ir_value())
            && previous != binding.kernel_ir_value()
        {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "shared helper parameter correspondence differs across roots",
            ));
        }
    }

    let selected_scope_count = captured_scopes
        .iter()
        .filter(|scope| function_by_semantic.contains_key(&scope.function))
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
        .filter(|scope| function_by_semantic.contains_key(&scope.function))
    {
        let function_ordinal = *function_by_semantic.get(&scope.function).ok_or(
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "compiler source scope has no exact KIR function owner",
            ),
        )?;
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
        .filter(|variable| function_by_semantic.contains_key(&variable.function))
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
        .filter(|variable| function_by_semantic.contains_key(&variable.function))
    {
        let function_ordinal = *function_by_semantic.get(&variable.function).ok_or(
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "compiler source variable has no exact KIR function owner",
            ),
        )?;
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
                    .get(&(variable.function, local))
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
        if let std::collections::btree_map::Entry::Vacant(entry) = files.entry(identity) {
            let file = captured_files.get(&identity).cloned().ok_or(
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "source-variable scope has no same-session rustc file observation",
                ),
            )?;
            entry.insert(file);
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

#[derive(Clone, Copy)]
struct SemanticStorageMapBindingInputV1 {
    container_identity: [u8; 32],
    subject_identity: [u8; 32],
    canonical_kir_digest: [u8; 32],
    canonical_kir_bytes: u64,
}

include!("production_debug_variable_storage_v1.rs");

fn compiler_semantic_storage_map_v1(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    captured_variables: &[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2],
    binding: SemanticStorageMapBindingInputV1,
) -> Result<fe2o3_kernel_ir::SemanticStorageMapV1, ProductionPipelineError> {
    use fe2o3_mir_model::semantic_mir_v1::{SemanticAbiPassModeV1, SemanticLocalRoleV1};

    let semantic = lowered.semantic().semantic();
    let selection = semantic.select_kernel_body_v1().ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage map requires one exact semantic kernel body",
        ),
    )?;
    let function = semantic
        .functions()
        .get(selection.body().index() as usize)
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage map semantic body is absent",
        ))?;
    let (kir_function_ordinal, kir_body) =
        kernel_storage_map_body_v1(lowered, selection.root(), selection.body())?;
    let kir_function = lowered.module().functions.get(kir_function_ordinal).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage map KIR function is absent",
        ),
    )?;
    if kir_body.parameters.len() != kir_function.signature.parameters.len() {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage map KIR parameter identities and types differ in length",
        ));
    }

    let parameter_bindings = lowered
        .correspondence()
        .parameter_bindings()
        .iter()
        .copied()
        .filter(|binding| binding.semantic_function() == selection.body())
        .collect::<Vec<_>>();
    let source_types = function.abi().source_input_types();
    let ownership = function.abi().source_argument_ownership();
    if source_types.len() != ownership.len() {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage map source types and ownership differ in length",
        ));
    }
    let mut arguments = Vec::new();
    arguments
        .try_reserve_exact(source_types.len())
        .map_err(|_| {
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "typed storage argument allocation failed",
            )
        })?;
    for (source_ordinal, (&semantic_type, &source_ownership)) in
        source_types.iter().zip(ownership).enumerate()
    {
        let source_ordinal = u32::try_from(source_ordinal).map_err(|_| {
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "typed storage source ordinal does not fit the wire",
            )
        })?;
        let semantic_local = function
            .locals()
            .iter()
            .enumerate()
            .find_map(|(index, local)| {
                (local.role() == SemanticLocalRoleV1::Argument(source_ordinal)).then_some(index)
            })
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "typed storage source argument has no exact semantic local",
            ))?;
        let semantic_local = u32::try_from(semantic_local).map_err(|_| {
            ProductionPipelineError::SimulationDebugMapCorrespondence(
                "typed storage semantic local does not fit the wire",
            )
        })?;
        let abi_ignored = function
            .abi()
            .adjusted_arguments()
            .get(source_ordinal as usize)
            .is_some_and(|argument| matches!(argument.mode(), SemanticAbiPassModeV1::Ignore));
        let storage = compiler_parameter_storage_v1(
            semantic_local,
            semantic_type.index(),
            source_ownership,
            abi_ignored,
            &parameter_bindings,
            kir_body,
            &kir_function.signature.parameters,
            semantic.types(),
        )?;
        arguments.push(fe2o3_kernel_ir::SemanticArgumentStorageV1::new(
            source_ordinal,
            semantic_local,
            semantic_type.index(),
            compiler_ownership_v1(source_ownership)?,
            storage,
        ));
    }

    let variables = compiler_debug_variable_storage_v1(lowered, captured_variables)?;

    fe2o3_kernel_ir::SemanticStorageMapV1::new(
        binding.container_identity,
        binding.subject_identity,
        semantic.wire_version().as_u16(),
        *semantic.semantic_sha256().as_bytes(),
        semantic.canonical_encoding().len() as u64,
        *semantic.target_layout_identity().as_bytes(),
        binding.canonical_kir_digest,
        binding.canonical_kir_bytes,
        vec![fe2o3_kernel_ir::SemanticKernelStorageV1::new(
            selection.root().index(),
            selection.body().index(),
            u32::try_from(kir_function_ordinal).map_err(|_| {
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "typed storage KIR function ordinal does not fit the wire",
                )
            })?,
            arguments,
        )],
        variables,
    )
    .map_err(ProductionPipelineError::SimulationBundleV3)
}

fn compiler_legacy_slot_shape_v2(
    ty: &fe2o3_kernel_ir::Type,
) -> Result<(u32, u32, Option<u32>), ProductionPipelineError> {
    let mut leaf = ty;
    loop {
        leaf = match leaf {
            fe2o3_kernel_ir::Type::Pointer(pointer) => &pointer.pointee,
            fe2o3_kernel_ir::Type::Slice(slice) => &slice.element,
            fe2o3_kernel_ir::Type::StorageObject(_) => {
                return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "storage-object types require a storage-aware simulator descriptor",
                ));
            }
            fe2o3_kernel_ir::Type::Unit
            | fe2o3_kernel_ir::Type::Scalar(_)
            | fe2o3_kernel_ir::Type::Vector(_)
            | fe2o3_kernel_ir::Type::Execution(_) => break,
        };
    }
    Ok(match ty {
        fe2o3_kernel_ir::Type::Scalar(scalar) => {
            let width = match scalar {
                fe2o3_kernel_ir::ScalarType::Bool
                | fe2o3_kernel_ir::ScalarType::I8
                | fe2o3_kernel_ir::ScalarType::U8 => 1,
                fe2o3_kernel_ir::ScalarType::I16
                | fe2o3_kernel_ir::ScalarType::U16
                | fe2o3_kernel_ir::ScalarType::F16
                | fe2o3_kernel_ir::ScalarType::Bf16 => 2,
                fe2o3_kernel_ir::ScalarType::I32
                | fe2o3_kernel_ir::ScalarType::U32
                | fe2o3_kernel_ir::ScalarType::F32 => 4,
                fe2o3_kernel_ir::ScalarType::I64
                | fe2o3_kernel_ir::ScalarType::U64
                | fe2o3_kernel_ir::ScalarType::F64
                | fe2o3_kernel_ir::ScalarType::Index => 8,
                fe2o3_kernel_ir::ScalarType::I128 | fe2o3_kernel_ir::ScalarType::U128 => 16,
            };
            (width, width, None)
        }
        fe2o3_kernel_ir::Type::Pointer(_) => (8, 8, None),
        fe2o3_kernel_ir::Type::Slice(_) => (8, 8, Some(8)),
        fe2o3_kernel_ir::Type::StorageObject(_) => {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "storage-object KIR parameters have no admitted legacy simulator slot",
            ));
        }
        fe2o3_kernel_ir::Type::Execution(_) => {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "execution KIR parameters have no admitted physical simulator slot",
            ));
        }
        fe2o3_kernel_ir::Type::Vector(_) => {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "vector KIR parameters have no admitted physical simulator slot",
            ));
        }
        fe2o3_kernel_ir::Type::Unit => {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                "unit KIR parameters have no physical simulator slot",
            ));
        }
    })
}

fn compiler_semantic_storage_map_v2(
    lowered: &fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1,
    container_identity: [u8; 32],
) -> Result<fe2o3_kernel_ir::SemanticStorageMapV2, ProductionPipelineError> {
    use fe2o3_mir_model::semantic_mir_v1::{SemanticAbiPassModeV1, SemanticLocalRoleV1};

    const MAP_ERROR: &str = "aggregate storage map is not exact";
    let semantic = lowered.semantic().semantic();
    let selection = semantic.select_kernel_body_v1().ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR),
    )?;
    let function = semantic
        .functions()
        .get(selection.body().index() as usize)
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            MAP_ERROR,
        ))?;
    let (kir_function_ordinal, kir_body) =
        kernel_storage_map_body_v1(lowered, selection.root(), selection.body())?;
    let kir_function = lowered.module().functions.get(kir_function_ordinal).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR),
    )?;
    if kir_body.parameters.len() != kir_function.signature.parameters.len() {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            MAP_ERROR,
        ));
    }

    let mut slots = Vec::new();
    slots
        .try_reserve_exact(kir_function.signature.parameters.len())
        .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?;
    let mut next = 0_u32;
    let mut kernarg_alignment = 1_u32;
    for ty in &kir_function.signature.parameters {
        let (width, alignment, metadata_relative) = compiler_legacy_slot_shape_v2(ty)?;
        next = align_up_u32_v1(next, alignment).ok_or(
            ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR),
        )?;
        let value = fe2o3_kernel_ir::SemanticKernargSlotV2::new(next, width, alignment);
        let metadata = metadata_relative
            .map(|relative| {
                next.checked_add(relative)
                    .map(|offset| {
                        fe2o3_kernel_ir::SemanticKernargSlotV2::new(offset, width, alignment)
                    })
                    .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                        MAP_ERROR,
                    ))
            })
            .transpose()?;
        let total_width = metadata_relative.map_or(width, |relative| relative + width);
        next = next.checked_add(total_width).ok_or(
            ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR),
        )?;
        kernarg_alignment = kernarg_alignment.max(alignment);
        slots.push((value, metadata));
    }
    let explicit_kernarg_bytes = align_up_u32_v1(next, kernarg_alignment).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR),
    )?;

    let source_types = function.abi().source_input_types();
    let ownership = function.abi().source_argument_ownership();
    if source_types.len() != ownership.len()
        || source_types.len() != function.abi().adjusted_arguments().len()
    {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            MAP_ERROR,
        ));
    }
    let mut arguments = Vec::new();
    arguments
        .try_reserve_exact(source_types.len())
        .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?;
    for (source_ordinal, ((&semantic_type, &source_ownership), abi)) in source_types
        .iter()
        .zip(ownership)
        .zip(function.abi().adjusted_arguments())
        .enumerate()
    {
        let source_ordinal = u32::try_from(source_ordinal)
            .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?;
        let semantic_local = function
            .locals()
            .iter()
            .enumerate()
            .find_map(|(local, declaration)| {
                (declaration.role() == SemanticLocalRoleV1::Argument(source_ordinal))
                    .then_some(local)
            })
            .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
                MAP_ERROR,
            ))?;
        let semantic_local_u32 = u32::try_from(semantic_local)
            .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?;
        let direct_matches = lowered
            .correspondence()
            .parameter_bindings()
            .iter()
            .filter(|binding| {
                binding.semantic_function() == selection.body()
                    && binding.semantic_local().index() == semantic_local_u32
            })
            .copied();
        let mut direct_probe = direct_matches.clone();
        let direct = direct_probe.next();
        let direct_is_unique = direct.is_some() && direct_probe.next().is_none();
        let component_matches = lowered
            .correspondence()
            .parameter_component_bindings()
            .iter()
            .filter(|binding| {
                binding.semantic_function() == selection.body()
                    && binding.semantic_local().index() == semantic_local_u32
            });
        let component_count = component_matches.clone().count();
        let ignored_matches = lowered
            .correspondence()
            .ignored_parameter_bindings()
            .iter()
            .filter(|binding| {
                binding.semantic_function() == selection.body()
                    && binding.semantic_local().index() == semantic_local_u32
            })
            .copied();
        let mut ignored_probe = ignored_matches.clone();
        let ignored = ignored_probe.next();
        let ignored_is_unique = ignored.is_some() && ignored_probe.next().is_none();
        let storage =
            if direct_is_unique {
                let binding = direct.expect("unique direct binding is present");
                if component_count != 0 || ignored.is_some() {
                    return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                        MAP_ERROR,
                    ));
                }
                let mut retained = Vec::new();
                retained.try_reserve_exact(1).map_err(|_| {
                    ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR)
                })?;
                retained.push(compiler_component_storage_v2(
                    Vec::new(),
                    binding.kernel_ir_value(),
                    kir_body,
                    &kir_function.signature.parameters,
                    &slots,
                )?);
                fe2o3_kernel_ir::SemanticComponentStorageBindingV2::exact(retained)
            } else if direct.is_none() && component_count != 0 && ignored.is_none() {
                let mut retained = Vec::new();
                retained.try_reserve_exact(component_count).map_err(|_| {
                    ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR)
                })?;
                for component in component_matches {
                    let mut path = Vec::new();
                    path.try_reserve_exact(component.projection().len())
                        .map_err(|_| {
                            ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR)
                        })?;
                    path.extend(
                        component.projection().iter().map(|projection| {
                            match projection {
                    fe2o3_lower_mir_kernel::SemanticKirParameterProjectionV1::Field(index) => {
                        fe2o3_kernel_ir::SemanticStorageProjectionV2::Field { index: *index }
                    }
                    fe2o3_lower_mir_kernel::SemanticKirParameterProjectionV1::ArrayIndex(index) => {
                        fe2o3_kernel_ir::SemanticStorageProjectionV2::ArrayElement {
                            index: u64::from(*index),
                        }
                    }
                }
                        }),
                    );
                    retained.push(compiler_component_storage_v2(
                        path,
                        component.kernel_ir_value(),
                        kir_body,
                        &kir_function.signature.parameters,
                        &slots,
                    )?);
                }
                fe2o3_kernel_ir::SemanticComponentStorageBindingV2::exact(retained)
            } else if direct.is_none()
                && component_count == 0
                && ignored_is_unique
                && matches!(abi.mode(), SemanticAbiPassModeV1::Ignore)
            {
                fe2o3_kernel_ir::SemanticComponentStorageBindingV2::exact(Vec::new())
            } else {
                return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    MAP_ERROR,
                ));
            };
        arguments.push(fe2o3_kernel_ir::SemanticArgumentStorageV2::new(
            source_ordinal,
            semantic_local_u32,
            semantic_type.index(),
            compiler_ownership_v1(source_ownership)?,
            storage,
        ));
    }

    let mut kernels = Vec::new();
    kernels
        .try_reserve_exact(1)
        .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?;
    kernels.push(fe2o3_kernel_ir::SemanticKernelStorageV2::new(
        selection.root().index(),
        selection.body().index(),
        u32::try_from(kir_function_ordinal)
            .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?,
        explicit_kernarg_bytes,
        kernarg_alignment,
        arguments,
    ));
    fe2o3_kernel_ir::SemanticStorageMapV2::new(container_identity, kernels)
        .map_err(ProductionPipelineError::SimulationBundleV4)
}

fn compiler_component_storage_v2(
    path: Vec<fe2o3_kernel_ir::SemanticStorageProjectionV2>,
    value: fe2o3_kernel_ir::ValueId,
    body: &fe2o3_kernel_ir::FunctionBody,
    parameter_types: &[fe2o3_kernel_ir::Type],
    slots: &[(
        fe2o3_kernel_ir::SemanticKernargSlotV2,
        Option<fe2o3_kernel_ir::SemanticKernargSlotV2>,
    )],
) -> Result<fe2o3_kernel_ir::SemanticKirComponentStorageV2, ProductionPipelineError> {
    const MAP_ERROR: &str = "aggregate component has no exact KIR physical slot";
    let ordinal = body
        .parameters
        .iter()
        .position(|candidate| *candidate == value)
        .ok_or(ProductionPipelineError::SimulationDebugMapCorrespondence(
            MAP_ERROR,
        ))?;
    if let Some(ty @ (fe2o3_kernel_ir::Type::Pointer(_) | fe2o3_kernel_ir::Type::Slice(_))) =
        parameter_types.get(ordinal)
    {
        compiler_legacy_slot_shape_v2(ty)?;
    }
    let (representation, expected_metadata) = match parameter_types.get(ordinal) {
        Some(fe2o3_kernel_ir::Type::Scalar(_)) => (
            fe2o3_kernel_ir::SemanticKirComponentRepresentationV2::ScalarValue,
            false,
        ),
        Some(fe2o3_kernel_ir::Type::Pointer(_)) => (
            fe2o3_kernel_ir::SemanticKirComponentRepresentationV2::RegionPointer,
            false,
        ),
        Some(fe2o3_kernel_ir::Type::Slice(_)) => (
            fe2o3_kernel_ir::SemanticKirComponentRepresentationV2::RegionSlice,
            true,
        ),
        Some(
            fe2o3_kernel_ir::Type::Unit
            | fe2o3_kernel_ir::Type::Vector(_)
            | fe2o3_kernel_ir::Type::Execution(_)
            | fe2o3_kernel_ir::Type::StorageObject(_),
        )
        | None => {
            return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                MAP_ERROR,
            ));
        }
    };
    let (value_slot, metadata_slot) = slots.get(ordinal).copied().ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR),
    )?;
    if metadata_slot.is_some() != expected_metadata {
        return Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            MAP_ERROR,
        ));
    }
    Ok(fe2o3_kernel_ir::SemanticKirComponentStorageV2::new(
        path,
        u32::try_from(ordinal)
            .map_err(|_| ProductionPipelineError::SimulationDebugMapCorrespondence(MAP_ERROR))?,
        value.0,
        representation,
        value_slot,
        metadata_slot,
    ))
}

const fn align_up_u32_v1(value: u32, alignment: u32) -> Option<u32> {
    let mask = alignment - 1;
    match value.checked_add(mask) {
        Some(value) => Some(value & !mask),
        None => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn compiler_parameter_storage_v1(
    semantic_local: u32,
    semantic_type: u32,
    ownership: fe2o3_mir_model::semantic_mir_v1::SemanticSourceArgumentOwnershipV1,
    abi_ignored: bool,
    bindings: &[fe2o3_lower_mir_kernel::SemanticKirParameterBindingV1],
    body: &fe2o3_kernel_ir::FunctionBody,
    parameter_types: &[fe2o3_kernel_ir::Type],
    semantic_types: &[fe2o3_mir_model::semantic_mir_v1::SemanticTypeDeclV1],
) -> Result<fe2o3_kernel_ir::SemanticStorageBindingV1, ProductionPipelineError> {
    let matching = bindings
        .iter()
        .filter(|binding| binding.semantic_local().index() == semantic_local)
        .collect::<Vec<_>>();
    let [binding] = matching.as_slice() else {
        return Ok(if matching.is_empty() {
            fe2o3_kernel_ir::SemanticStorageBindingV1::Unavailable {
                reason: if abi_ignored {
                    fe2o3_kernel_ir::SemanticStorageUnavailableReasonV1::AbiIgnored
                } else {
                    fe2o3_kernel_ir::SemanticStorageUnavailableReasonV1::NoRetainedKirStorage
                },
            }
        } else {
            fe2o3_kernel_ir::SemanticStorageBindingV1::Ambiguous
        });
    };
    let Some(parameter_ordinal) = body
        .parameters
        .iter()
        .position(|value| *value == binding.kernel_ir_value())
    else {
        return Ok(fe2o3_kernel_ir::SemanticStorageBindingV1::Unavailable {
            reason: fe2o3_kernel_ir::SemanticStorageUnavailableReasonV1::NoRetainedKirStorage,
        });
    };
    let kir_type = parameter_types.get(parameter_ordinal).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage parameter type is absent",
        ),
    )?;
    let semantic = semantic_types.get(semantic_type as usize).ok_or(
        ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage semantic type is absent",
        ),
    )?;
    let representation = match (semantic.shape(), kir_type, ownership) {
        (
            fe2o3_mir_model::semantic_mir_v1::SemanticTypeShapeV1::Scalar(_)
            | fe2o3_mir_model::semantic_mir_v1::SemanticTypeShapeV1::ValidityScalar(_),
            fe2o3_kernel_ir::Type::Scalar(_),
            _,
        ) => fe2o3_kernel_ir::SemanticKirStorageRepresentationV1::Scalar,
        (
            fe2o3_mir_model::semantic_mir_v1::SemanticTypeShapeV1::Pointer(_),
            fe2o3_kernel_ir::Type::Slice(_),
            _,
        ) => fe2o3_kernel_ir::SemanticKirStorageRepresentationV1::RegionSlice,
        (_, fe2o3_kernel_ir::Type::Slice(_), ownership)
            if ownership
                != fe2o3_mir_model::semantic_mir_v1::SemanticSourceArgumentOwnershipV1::ByValue =>
        {
            fe2o3_kernel_ir::SemanticKirStorageRepresentationV1::RegionSlice
        }
        (_, fe2o3_kernel_ir::Type::Pointer(_), ownership)
            if ownership
                != fe2o3_mir_model::semantic_mir_v1::SemanticSourceArgumentOwnershipV1::ByValue =>
        {
            fe2o3_kernel_ir::SemanticKirStorageRepresentationV1::RegionPointer
        }
        _ => fe2o3_kernel_ir::SemanticKirStorageRepresentationV1::OpaqueFlattened,
    };
    Ok(
        fe2o3_kernel_ir::SemanticStorageBindingV1::ExactKirParameter {
            kir_parameter_ordinal: u32::try_from(parameter_ordinal).map_err(|_| {
                ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "typed storage KIR parameter ordinal does not fit the wire",
                )
            })?,
            kir_value_ordinal: binding.kernel_ir_value().0,
            representation,
        },
    )
}

fn compiler_ownership_v1(
    ownership: fe2o3_mir_model::semantic_mir_v1::SemanticSourceArgumentOwnershipV1,
) -> Result<fe2o3_kernel_ir::SemanticArgumentOwnershipV1, ProductionPipelineError> {
    use fe2o3_mir_model::semantic_mir_v1::SemanticSourceArgumentOwnershipV1 as Source;
    match ownership {
        Source::ByValue => Ok(fe2o3_kernel_ir::SemanticArgumentOwnershipV1::ByValue),
        Source::SharedBorrow => Ok(fe2o3_kernel_ir::SemanticArgumentOwnershipV1::SharedBorrow),
        Source::UniqueBorrow => Ok(fe2o3_kernel_ir::SemanticArgumentOwnershipV1::UniqueBorrow),
        Source::ExclusiveOwner => Ok(fe2o3_kernel_ir::SemanticArgumentOwnershipV1::ExclusiveOwner),
        Source::RawPointer => Ok(fe2o3_kernel_ir::SemanticArgumentOwnershipV1::RawPointer),
        Source::Unspecified => Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
            "typed storage map rejects unspecified source ownership",
        )),
    }
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

type ExactDebugSemanticConstructV1 = (u8, u32, u32, u32);
type ExactDebugMappedOperationsV1 = BTreeMap<
    fe2o3_kernel_ir::DebugSourceMapKirSiteV1,
    (
        ExactDebugSemanticConstructV1,
        fe2o3_kernel_ir::DebugSourceMapSpanV1,
    ),
>;

#[allow(clippy::too_many_arguments)]
fn insert_debug_operation_range_v1(
    function_ordinal: u64,
    body: &fe2o3_kernel_ir::FunctionBody,
    block_ordinals: &BTreeMap<fe2o3_kernel_ir::BlockId, usize>,
    semantic_owner: ExactDebugSemanticConstructV1,
    block: fe2o3_kernel_ir::BlockId,
    first_operation: u32,
    operation_count: u32,
    source: fe2o3_mir_model::semantic_mir_v1::SemanticSourceProvenanceV1,
    mapped: &mut ExactDebugMappedOperationsV1,
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
        if mapped
            .insert(site, (semantic_owner, source_span))
            .is_some_and(|previous| previous != (semantic_owner, source_span))
        {
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
    #[cfg(test)]
    pub(crate) fn checked_output_source_policy_v1(
        &self,
    ) -> fe2o3_lower_mir_kernel::ProductionHelperSourcePolicyV1 {
        self.ranked.materialized().helper_source_policy_v1()
    }

    pub(crate) fn ranked_roots(
        &self,
    ) -> &[crate::production_ranked_projection_v1::ProductionRankedRootProgramV1] {
        self.ranked.roots()
    }

    pub(crate) fn ranked_root_count(&self) -> usize {
        self.ranked.root_count()
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
    ) -> Result<Self, ProductionPipelineError> {
        Self::from_collected_device_closure_with_custody(
            tcx,
            closure,
            producer,
            output_dir,
            ProductionCompilerCustody::protected(invocation, build_attempt),
            crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2::SourceVariables,
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
        self.import_semantic_mir_with_nominal_v35(false)
    }

    fn import_semantic_mir_with_nominal_v35(
        self,
        nominal: bool,
    ) -> Result<ProductionCompilation<'tcx, AdmittedSemanticMirStage>, ProductionPipelineError>
    {
        self.import_semantic_mir_with_profile_v29(if nominal {
            source_owned_v29::ImportProfile::NominalV35
        } else {
            source_owned_v29::ImportProfile::Current
        })
    }

    fn import_semantic_mir_with_profile_v29(
        self,
        profile: source_owned_v29::ImportProfile,
    ) -> Result<ProductionCompilation<'tcx, AdmittedSemanticMirStage>, ProductionPipelineError>
    {
        self.import_semantic_mir_with_enrollment_v1(profile, None)
    }

    fn import_semantic_mir_with_enrollment_v1(
        self,
        profile: source_owned_v29::ImportProfile,
        enrollment: Option<(
            crate::protected_compiler_execution::native_v3::ReferenceEnrollmentPreparationV1<'_>,
            &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        )>,
    ) -> Result<ProductionCompilation<'tcx, AdmittedSemanticMirStage>, ProductionPipelineError>
    {
        let CollectedRustStage {
            tcx,
            closure,
            typed_descriptor_roots,
            debug_source_capture,
            mut transaction,
        } = self.stage;
        let constructed = match enrollment {
            Some((preparation, budget)) if preparation.is_enrolled() => {
                let nominal = match profile {
                    source_owned_v29::ImportProfile::Current => false,
                    source_owned_v29::ImportProfile::NominalV35 => true,
                    _ => return Err(
                        crate::protected_compiler_execution::native_v3::Error::Reference(
                            crate::reference_effect_v1::ReferenceBindingErrorV1::new(
                                "reference enrollment is not integrated with this import profile",
                            ),
                        )
                        .into(),
                    ),
                };
                let ProductionCompilerCustody::ProtectedV3 { invocation, .. } =
                    &mut transaction.compiler_custody
                else {
                    return Err(ProductionPipelineError::ExtractionCannotPublish);
                };
                preparation.with_invocation(invocation, budget, |loan| {
                    crate::collector::construct_production_semantic_mir_with_enrollment_v1(
                        tcx,
                        closure,
                        debug_source_capture,
                        nominal,
                        loan,
                    )
                    .map_err(ProductionPipelineError::SemanticImport)
                })?
            }
            _ => match profile {
                source_owned_v29::ImportProfile::AtomicV41 => {
                    crate::collector::construct_production_semantic_mir_atomic_v41(
                        tcx,
                        closure,
                        debug_source_capture,
                    )
                }
                source_owned_v29::ImportProfile::NominalV35 => {
                    crate::collector::construct_production_semantic_mir_nominal_v35(
                        tcx,
                        closure,
                        debug_source_capture,
                    )
                }
                source_owned_v29::ImportProfile::Current => {
                    crate::collector::construct_production_semantic_mir_v1(
                        tcx,
                        closure,
                        debug_source_capture,
                    )
                }
                source_owned_v29::ImportProfile::SourceOwnedV29 => {
                    crate::collector::construct_production_semantic_mir_source_owned_v29(
                        tcx,
                        closure,
                        debug_source_capture,
                    )
                }
            }
            .map_err(ProductionPipelineError::SemanticImport)?,
        };
        let crate::collector::ConstructedProductionSemanticMirV1 {
            semantic_mir,
            context_entries,
            rustc_identity_inventory,
            rustc_preflight_plan,
            rustc_target,
            reference_effect_bindings,
            debug_source_files,
            debug_source_scopes,
            debug_source_variables,
            debug_capture_gap,
        } = constructed;
        let typed_descriptor_roots =
            crate::compiler_descriptor::order_typed_descriptor_roots_by_semantic_v1(
                typed_descriptor_roots,
                &semantic_mir,
            )
            .map_err(ProductionPipelineError::DescriptorEvidence)?;
        Ok(ProductionCompilation {
            stage: AdmittedSemanticMirStage {
                semantic_mir,
                bindings: AuthenticatedProductionBindings {
                    context_entries,
                    rustc_identity_inventory,
                    rustc_preflight_plan,
                    rustc_target,
                    reference_effect_bindings,
                    debug_source_files,
                    debug_source_scopes,
                    debug_source_variables,
                    debug_capture_gap,
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
        self.verify_general_kernel_checks_with_enrollment_v1(None)
    }

    fn verify_general_kernel_checks_with_enrollment_v1(
        self,
        enrollment: Option<(
            crate::protected_compiler_execution::native_v3::ReferenceEnrollmentPreparationV1<'_>,
            &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        )>,
    ) -> Result<RankedVerifiedProductionCompilation, ProductionPipelineError> {
        let admitted = self.import_semantic_mir_with_enrollment_v1(
            source_owned_v29::ImportProfile::Current,
            enrollment,
        )?;
        admitted
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()
    }

    pub(crate) fn select_compiler_proof_family_v1(
        mut self,
        family: fe2o3_compiler_execution_protocol::CompilerExecutionPolicyFamilyV1,
    ) -> Result<Self, ProductionPipelineError> {
        let ProductionCompilerCustody::ProtectedV3 { invocation, .. } =
            &mut self.stage.transaction.compiler_custody
        else {
            return Err(ProductionPipelineError::ExtractionCannotPublish);
        };
        match family {
            fe2o3_compiler_execution_protocol::CompilerExecutionPolicyFamilyV1::LegacyV1 => {
                invocation.select_legacy_proof_runtime()
            }
            fe2o3_compiler_execution_protocol::CompilerExecutionPolicyFamilyV1::NativeV3 => {
                invocation.select_native_proof_runtime()
            }
        }
        .map_err(ProductionPipelineError::ProtectedRustcInvocation)?;
        Ok(self)
    }

    /// Consumes the sole production transaction through exact semantic MIR,
    /// formal memory admission, and exact authenticated-target LLVM lowering.
    pub(crate) fn lower_production_target(
        self,
        target_budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<TargetLoweredProductionCompilation, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        let ranked = admitted
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?;
        if ranked.has_direct_conditional_roots_v2() {
            return Err(ranked.conditional_production_finalizer_refusal_v5(target_budget));
        }
        ranked
            .attach_target_neutral_checks()?
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
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?
            .attach_target_neutral_checks()?
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
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?
            .attach_target_neutral_checks()?
            .into_simulation_bundle_v2(
                fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
            )
    }

    /// Emits V3 with the exact admitted semantic MIR and its independently
    /// versioned semantic-local to KIR-storage projection.
    pub(crate) fn export_simulation_bundle_v3(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV3, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?
            .attach_target_neutral_checks()?
            .into_simulation_bundle_v3(
                fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
            )
    }

    /// Emits V4 with compiler-rederived one-to-many aggregate component and
    /// physical simulator-kernarg correspondence. The KFD descriptor path is
    /// intentionally not entered.
    pub(crate) fn export_simulation_bundle_v4(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV4, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?
            .attach_target_neutral_checks()?
            .into_simulation_bundle_v4(
                fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
            )
    }

    /// Emits self-contained V5 with the exact V10 re-encoding of the same
    /// producer-owned V8/V9 module and independently bound debug/storage data.
    /// This extraction-only path grants no compiler, artifact, load, launch,
    /// proof, or hardware authority.
    pub(crate) fn export_simulation_bundle_v5(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV5, ProductionPipelineError> {
        let admitted = self.import_semantic_mir()?;
        admitted
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?
            .attach_target_neutral_checks()?
            .into_simulation_bundle_v5()
    }

    /// Exports an authority-free V6 simulation bundle whose exact same-module
    /// execution body is canonical KIR V11.
    pub(crate) fn export_simulation_bundle_v6(
        self,
    ) -> Result<fe2o3_kernel_ir::VerifiedSimulationBundleV6, ProductionPipelineError> {
        self.import_semantic_mir()?
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?
            .materialize_target_neutral()?
            .verify_general_kernel_checks()?
            .attach_target_neutral_checks()?
            .into_simulation_bundle_v6()
    }

    /// Publishes the exact production compiler module into the managed,
    /// preselected attempt-scoped protocol. This grants no link, artifact, load,
    /// or launch authority. The early-admitted session stays live outside the
    /// compiler stages and is consumed only for this exact publication.
    pub(crate) fn publish_worker_handoff(
        self,
        target_budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        compiler_execution: AdmittedProtectedCompilerExecutionV1,
    ) -> Result<fe2o3_artifact_transaction::InertCompilerExecutionSubjectV1, ProductionPipelineError>
    {
        self.publish_predicated_worker_handoff_v90(target_budget, compiler_execution)
    }

    /// Retains the original extraction milestone while consuming the same
    /// transaction and importer as the production backend.
    pub(crate) fn require_semantic_mir_import(self) -> ProductionPipelineError {
        match self.import_semantic_mir() {
            Ok(transaction) => match transaction.construct_semantic_middle_end() {
                Ok(transaction) => match transaction.construct_semantic_ssa() {
                    Ok(transaction) => transaction.require_target_neutral_lowering(),
                    Err(error) => error,
                },
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
    fn construct_semantic_ssa(
        self,
    ) -> Result<ProductionCompilation<'tcx, SsaSemanticMirStage>, ProductionPipelineError> {
        let EquivalentSemanticMirStage {
            semantic_mir,
            bindings,
        } = self.stage;
        let semantic_ssa = fe2o3_pliron::ProductionSemanticSsaOwnerV1::try_new(
            semantic_mir,
            fe2o3_pliron::ProductionSemanticSsaLimitsV1::default(),
        )
        .map_err(ProductionPipelineError::SemanticSsa)?;
        bindings
            .context_entries
            .validate_source(semantic_ssa.source_semantic())
            .map_err(|error| {
                ProductionPipelineError::SemanticImport(
                    crate::collector::ProductionSemanticImportErrorV1::BodyConstruction(Box::new(
                        error,
                    )),
                )
            })?;
        Ok(ProductionCompilation {
            stage: SsaSemanticMirStage {
                semantic_ssa,
                bindings,
            },
            invariant_session: PhantomData,
        })
    }
}

#[path = "production_context_handoff_v29.rs"]
mod context_handoff_v29;
#[cfg(test)]
pub(crate) use context_handoff_v29::{
    check_context_handoff_v29, with_projected_execution_source_v29,
};

impl<'tcx> ProductionCompilation<'tcx, SsaSemanticMirStage> {
    fn require_target_neutral_lowering(self) -> ProductionPipelineError {
        let SsaSemanticMirStage {
            semantic_ssa,
            bindings,
        } = self.stage;
        let error =
            crate::collector::ProductionSemanticImportErrorV1::TargetNeutralLoweringPending {
                functions: semantic_ssa.source_semantic().functions().len(),
                callables: semantic_ssa.source_semantic().callables().len(),
                rustc_identity_inventory_sha256: bindings.rustc_identity_inventory.sha256(),
                rustc_preflight_plan_sha256: bindings.rustc_preflight_plan.sha256(),
                semantic_sha256: *semantic_ssa.source_semantic().semantic_sha256().as_bytes(),
            };
        drop((semantic_ssa, bindings));
        ProductionPipelineError::SemanticImport(error)
    }

    fn materialize_target_neutral(
        self,
    ) -> Result<MaterializedNeutralProductionCompilation, Box<ProductionPipelineError>> {
        self.materialize_with_context_observer_v29(|_, _| Ok(()))
    }

    fn materialize_with_context_observer_v29(
        self,
        use_root: impl for<'a> FnMut(
            fe2o3_lower_mir_kernel::ProductionCheckedContextRootV29<'a>,
            &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        ) -> Result<
            (),
            fe2o3_lower_mir_kernel::ProductionContextRootErrorV29,
        >,
    ) -> Result<MaterializedNeutralProductionCompilation, Box<ProductionPipelineError>> {
        let PreparedMaterializationV29 {
            materialized,
            ranked_roots,
            bindings,
        } = self.materialize_prepared_v29(
            use_root,
            retained_materialization_phase_v1::retained_target_pipeline_v1::materialize_ordinary_owner_with_budget_v1,
        )?;
        Ok(MaterializedNeutralProductionCompilation {
            materialized,
            ranked_roots,
            bindings,
        })
    }

    // Both owning constructors cross the same descriptor/context/launch gates.
    // This private callback is not a caller-selected compiler policy.
    fn materialize_prepared_v29<M>(
        self,
        use_root: impl for<'a> FnMut(
            fe2o3_lower_mir_kernel::ProductionCheckedContextRootV29<'a>,
            &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        ) -> Result<
            (),
            fe2o3_lower_mir_kernel::ProductionContextRootErrorV29,
        >,
        materialize: impl FnOnce(
            fe2o3_pliron::ProductionSemanticSsaOwnerV1,
            fe2o3_lower_mir_kernel::ProductionSourceLaunchRosterV1,
            &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        ) -> Result<(M, usize), ProductionPipelineError>,
    ) -> Result<PreparedMaterializationV29<M>, Box<ProductionPipelineError>> {
        self.with_prepared_materialization_budget_v29(|prepared, budget| {
            materialize_prepared_with_budget_v29(prepared, budget, use_root, materialize)
        })
    }

    fn with_prepared_materialization_budget_v29<R>(
        self,
        run: impl FnOnce(
            PreparedSsaMaterializationV29,
            &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        ) -> Result<R, Box<ProductionPipelineError>>,
    ) -> Result<R, Box<ProductionPipelineError>> {
        let prepared = self.prepare_materialization_roster_v1()?;
        let resource_error = materialization_resource_error_v29;
        let work_limit = usize::try_from(crate::production_canonical_phase_policy_v1::WORK_LIMIT)
            .map_err(|_| {
            resource_error(
                fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1::Arithmetic,
            )
        })?;
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(work_limit);
        let mut budget = fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1::new(
            &mut work,
            crate::production_canonical_phase_policy_v1::STORAGE_LIMIT,
        );
        run(prepared, &mut budget)
    }

    fn prepare_materialization_inputs_v29(
        self,
        roots: impl FnOnce(
            &[crate::compiler_descriptor::TypedDescriptorRootV1],
        ) -> Result<
            Vec<crate::production_ranked_projection_v1::ProductionRankedRootInputV1>,
            ProductionPipelineError,
        >,
    ) -> Result<PreparedSsaMaterializationV29, ProductionPipelineError> {
        let SsaSemanticMirStage {
            semantic_ssa,
            bindings,
        } = self.stage;
        crate::compiler_descriptor::validate_production_v1_semantic_ownership_evidence(
            &bindings.typed_descriptor_roots,
            semantic_ssa.source_semantic(),
        )
        .map_err(ProductionPipelineError::DescriptorEvidence)?;
        let ranked_roots = roots(&bindings.typed_descriptor_roots)?;
        let launch =
            crate::production_ranked_projection_v1::source_launch_roster_for_ranked_inputs_v1(
                &semantic_ssa,
                &ranked_roots,
            )
            .map_err(crate::production_ranked_projection_v1::source_launch_projection_error_v1)
            .map_err(ProductionPipelineError::RankedProjection)?;
        Ok(PreparedSsaMaterializationV29 {
            semantic_ssa,
            ranked_roots,
            launch,
            bindings,
        })
    }
}

fn materialization_resource_error_v29(
    error: fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1,
) -> ProductionPipelineError {
    ProductionPipelineError::PreRankedMaterialization(
        fe2o3_lower_mir_kernel::ProductionPreRankedKirErrorV1::Canonical(
            fe2o3_kernel_ir::CanonicalKernelIrReplayAdmissionErrorV12::Resource(error),
        ),
    )
}

// Same genuine context and materialization path; the caller selects no policy.
// Preserve the unreserved-delta contract consumed by ordinary and guarded owners.
fn materialize_prepared_with_budget_v29<M>(
    prepared: PreparedSsaMaterializationV29,
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    use_root: impl for<'a> FnMut(
        fe2o3_lower_mir_kernel::ProductionCheckedContextRootV29<'a>,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    )
        -> Result<(), fe2o3_lower_mir_kernel::ProductionContextRootErrorV29>,
    materialize: impl FnOnce(
        fe2o3_pliron::ProductionSemanticSsaOwnerV1,
        fe2o3_lower_mir_kernel::ProductionSourceLaunchRosterV1,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<(M, usize), ProductionPipelineError>,
) -> Result<PreparedMaterializationV29<M>, Box<ProductionPipelineError>> {
    let PreparedSsaMaterializationV29 {
        semantic_ssa,
        ranked_roots,
        launch,
        bindings,
    } = prepared;
    #[cfg(test)]
    let semantic_ssa = {
        let mut ssa = semantic_ssa;
        if ssa.source_semantic().wire_version()
            == fe2o3_mir_model::semantic_mir_v1::SemanticMirWireVersionV1::V29
        {
            let receipt = ssa
                .try_capture_occurrences_with_budget_v1(budget)
                .map_err(|error| {
                    ProductionPipelineError::PreRankedMaterialization(
                        fe2o3_lower_mir_kernel::ProductionPreRankedKirErrorV1::Occurrences(error),
                    )
                })?;
            budget
                .reserve_storage(receipt.retained_storage())
                .map_err(materialization_resource_error_v29)?;
        }
        ssa
    };
    let PreparedMaterializationV29 {
        materialized: (materialized, retained_storage),
        ranked_roots,
        bindings,
    } = consume_prepared_with_budget_v29(
        PreparedSsaMaterializationV29 {
            semantic_ssa,
            ranked_roots,
            launch,
            bindings,
        },
        budget,
        use_root,
        |semantic_ssa, launch, _, budget| materialize(semantic_ssa, launch, budget),
    )?;
    // Accept graph, sealed origins and helper transfers before the next phase.
    // Inherited descriptor/launch/source-ranked allocations remain separate.
    budget
        .reserve_storage(retained_storage)
        .map_err(materialization_resource_error_v29)?;
    Ok(PreparedMaterializationV29 {
        materialized,
        ranked_roots,
        bindings,
    })
}

// This core authenticates custody, but deliberately owns no result-storage convention.
fn consume_prepared_with_budget_v29<M>(
    prepared: PreparedSsaMaterializationV29,
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    use_root: impl for<'a> FnMut(
        fe2o3_lower_mir_kernel::ProductionCheckedContextRootV29<'a>,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    )
        -> Result<(), fe2o3_lower_mir_kernel::ProductionContextRootErrorV29>,
    consume: impl FnOnce(
        fe2o3_pliron::ProductionSemanticSsaOwnerV1,
        fe2o3_lower_mir_kernel::ProductionSourceLaunchRosterV1,
        &crate::collector::RetainedContextEntriesV29,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<M, ProductionPipelineError>,
) -> Result<PreparedMaterializationV29<M>, Box<ProductionPipelineError>> {
    let PreparedSsaMaterializationV29 {
        semantic_ssa,
        ranked_roots,
        launch,
        bindings,
    } = prepared;
    context_handoff_v29::check_context_handoff_v29(
        &bindings.context_entries,
        &semantic_ssa,
        &launch,
        budget,
        use_root,
    )?;
    let materialized = consume(semantic_ssa, launch, &bindings.context_entries, budget)?;
    Ok(PreparedMaterializationV29 {
        materialized,
        ranked_roots,
        bindings,
    })
}

// Source-qualified test checkpoint only. Ordinary consumer above is unchanged.
#[cfg(test)]
fn consume_prepared_with_actual_inputs_for_test_v1<M, F>(
    prepared: PreparedSsaMaterializationV29,
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    owned_frame: &mut usize,
    use_root: impl for<'a> FnMut(
        fe2o3_lower_mir_kernel::ProductionCheckedContextRootV29<'a>,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    )
        -> Result<(), fe2o3_lower_mir_kernel::ProductionContextRootErrorV29>,
    consume: F,
) -> Result<PreparedMaterializationV29<M>, Box<ProductionPipelineError>>
where
    F: FnOnce(
        fe2o3_pliron::ProductionSemanticSsaOwnerV1,
        fe2o3_lower_mir_kernel::ProductionSourceLaunchRosterV1,
        &crate::collector::RetainedContextEntriesV29,
        &[crate::production_ranked_projection_v1::ProductionRankedRootInputV1],
        &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        &mut usize,
    ) -> Result<M, ProductionPipelineError>,
{
    // This header-only debit belongs to the outer genuine phase; its physical
    // closure/borrow frames are gone before that phase refunds it. No payload
    // or retained input is cloned, reconstructed or newly claimed by this debit.
    let frame = std::mem::size_of::<F>()
        .checked_mul(2)
        .and_then(|n| n.checked_add(4096))
        .and_then(|n| {
            std::mem::size_of::<Result<M, ProductionPipelineError>>()
                .checked_mul(2)
                .and_then(|result| n.checked_add(result))
        })
        .ok_or_else(|| {
            materialization_resource_error_v29(
                fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1::Arithmetic,
            )
        })?;
    let total = owned_frame.checked_add(frame).ok_or_else(|| {
        materialization_resource_error_v29(
            fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1::Arithmetic,
        )
    })?;
    budget
        .charge_work(frame)
        .map_err(materialization_resource_error_v29)?;
    budget
        .reserve_storage(frame)
        .map_err(materialization_resource_error_v29)?;
    *owned_frame = total;
    let PreparedSsaMaterializationV29 {
        semantic_ssa,
        ranked_roots,
        launch,
        bindings,
    } = prepared;
    context_handoff_v29::check_context_handoff_v29(
        &bindings.context_entries,
        &semantic_ssa,
        &launch,
        budget,
        use_root,
    )?;
    let materialized = consume(
        semantic_ssa,
        launch,
        &bindings.context_entries,
        &ranked_roots,
        &bindings.reference_effect_bindings,
        budget,
        owned_frame,
    )?;
    Ok(PreparedMaterializationV29 {
        materialized,
        ranked_roots,
        bindings,
    })
}

impl MaterializedNeutralProductionCompilation {
    /// Closed private continuation, not selected by verify_general_kernel_checks.
    /// Lowerer nominal tensor/effect translation must be implemented separately.
    #[allow(dead_code)]
    fn verify_private_nominal_kernel_checks_v1(
        self,
    ) -> Result<RankedVerifiedProductionCompilation, ProductionPipelineError> {
        let Self {
            materialized,
            ranked_roots,
            bindings,
        } = self;
        let ranked =
            crate::production_ranked_projection_v1::project_private_nominal_materialized_v1(
                materialized,
                &ranked_roots,
                &bindings.reference_effect_bindings,
            )
            .map_err(ProductionPipelineError::RankedProjection)?;
        Ok(RankedVerifiedProductionCompilation { ranked, bindings })
    }

    fn verify_general_kernel_checks(
        self,
    ) -> Result<RankedVerifiedProductionCompilation, ProductionPipelineError> {
        let Self {
            materialized,
            ranked_roots,
            bindings,
        } = self;
        let ranked =
            crate::production_ranked_projection_v1::project_and_verify_ranked_materialized_with_runtime_v1(
                materialized,
                &ranked_roots,
                &bindings.reference_effect_bindings,
                bindings.transaction.compiler_custody.proof_runtime()?,
            )
            .map_err(ProductionPipelineError::RankedProjection)?;
        Ok(RankedVerifiedProductionCompilation { ranked, bindings })
    }
}

impl RankedVerifiedProductionCompilation {
    fn replay_conditional_for_target_v1(self) -> Result<Self, ProductionPipelineError> {
        let Self { ranked, bindings } = self;
        let ranked = if ranked.has_conditional_roots_v1() {
            conditional_generated_fields_v1::replay_conditional_roots_v1(ranked, &bindings)
                .map_err(ProductionPipelineError::RankedVerification)?
        } else {
            ranked
        };
        Ok(Self { ranked, bindings })
    }

    fn attach_target_neutral_checks(
        self,
    ) -> Result<TargetNeutralProductionCompilation, ProductionPipelineError> {
        self.attach_target_neutral_checks_with_translation_budget_v1(None)
    }

    // Only the helper translation scan/cache/expansion selects this caller
    // ledger. Receipt conversion and the other checks keep their old scope.
    fn attach_target_neutral_checks_with_translation_budget_v1(
        self,
        budget: Option<&mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>>,
    ) -> Result<TargetNeutralProductionCompilation, ProductionPipelineError> {
        let Self { ranked, bindings } = self.replay_conditional_for_target_v1()?;
        let roster_receipt = ranked
            .into_verified_roster_receipt_with_runtime(
                bindings
                    .transaction
                    .compiler_custody
                    .proof_runtime()?
                    .map(|runtime| runtime.as_ref()),
            )
            .map_err(ProductionPipelineError::RankedVerification)?;
        debug_assert!(!roster_receipt.grants_artifact_or_launch_authority());
        debug_assert!(roster_receipt.verify_equivalence().is_ok());
        debug_assert_ne!(
            roster_receipt.canonical_roster_identity().as_bytes(),
            &[0; 32],
        );
        let (receipt, ranked_verification) = roster_receipt
            .into_module_verified_receipt()
            .map_err(ProductionPipelineError::RankedVerification)?;
        debug_assert!(ranked_verification.every_functional_verification_is_coherent());
        let lowered = match budget {
            Some(budget) =>
                fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1::
                    try_attach_materialized_ranked_checks_with_bounded_translation_budget_v1(receipt, budget),
            None =>
                fe2o3_lower_mir_kernel::ProductionSemanticKirOwnerV1::
                    try_attach_materialized_ranked_checks(receipt),
        }.map_err(ProductionPipelineError::TargetNeutralLowering)?;
        let exact_translation_roster = {
            let mut translations = lowered.mir_pliron_translation_validations();
            translations.len() == lowered.module().kernels.len()
                && translations
                    .by_ref()
                    .zip(&lowered.module().kernels)
                    .all(|((function_name, _), kernel)| function_name == kernel.id.as_str())
        };
        if !exact_translation_roster {
            return Err(ProductionPipelineError::MissingMirPlironTranslationValidation);
        }
        Ok(TargetNeutralProductionCompilation {
            lowered,
            ranked_verification,
            bindings,
        })
    }
}

pub(crate) mod complete_body_vnext;
#[path = "production_pipeline_guarded_loop_source_v1.rs"]
pub(crate) mod guarded_loop_source_v1;
#[path = "production_pipeline_loop_capture_v1.rs"]
pub(crate) mod loop_capture_v1;
pub(crate) mod physical_entry_diagnostic_v20;
pub(crate) mod physical_entry_target_v20;
pub(crate) mod physical_global_copy_diagnostic_v21;
pub(crate) mod physical_lds_exchange_diagnostic_v22;
#[path = "production_pipeline_source_owned_v29.rs"]
pub(crate) mod source_owned_v29;
pub(crate) use complete_body_vnext::AuthenticatedCompleteBodyTargetModuleV19;
pub(crate) use physical_entry_target_v20::AuthenticatedPhysicalEntryTargetModuleV20;
pub(crate) mod ordered_composition_target_v1;
pub(crate) mod ordered_composition_v1;
pub(crate) mod ordered_program_diagnostic_v32;
pub(crate) mod ordered_program_origin_v1;
pub(crate) mod ordered_region_diagnostic_v31;
#[cfg(test)]
pub(crate) mod ordered_region_qualification_v31;
#[cfg(all(test, target_os = "linux"))]
pub(crate) mod source_candidate_debug_join_v17_tests;
#[cfg(all(test, target_os = "linux"))]
pub(crate) mod source_candidate_launch_analysis_v17_tests;

#[cfg(any(test, target_os = "linux"))]
mod source_bitselect_feasibility_v1_tests;

#[cfg(test)]
mod tests {
    include!("production_pipeline_pre_ranked_routes_tests.rs");
    use super::*;

    #[test]
    fn execution_components_have_no_physical_simulator_storage() {
        use fe2o3_kernel_ir::{
            ExecutionRoleV15 as Role, FunctionBody, SemanticKernargSlotV2, Type, ValueId,
        };
        let body = FunctionBody {
            parameters: vec![ValueId(0)],
            blocks: vec![],
        };
        let slots = [(SemanticKernargSlotV2::new(0, 8, 8), None)];
        assert!(
            compiler_component_storage_v2(vec![], ValueId(0), &body, &[Type::INDEX], &slots,)
                .is_ok()
        );
        for role in [
            Role::Context,
            Role::Workgroup,
            Role::MaskedTileU32 {
                lanes: 64,
                elements: 4,
            },
            Role::LaneFragmentU32 {
                lanes: 64,
                elements: 4,
            },
        ] {
            assert!(matches!(
                compiler_component_storage_v2(
                    vec![],
                    ValueId(0),
                    &body,
                    &[Type::Execution(role)],
                    &slots,
                ),
                Err(ProductionPipelineError::SimulationDebugMapCorrespondence(
                    "aggregate component has no exact KIR physical slot"
                )),
            ));
        }
    }

    #[test]
    fn explicit_v2_simulation_export_requires_complete_source_capture() {
        assert!(require_complete_simulation_debug_source_capture_v2(None).is_ok());

        let error = require_complete_simulation_debug_source_capture_v2(Some(
            fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::SourceObservationUnrepresentable,
        ))
        .unwrap_err();
        assert!(matches!(
            error,
            ProductionPipelineError::SimulationDebugSourceCaptureUnavailable(
                fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::
                    SourceObservationUnrepresentable
            )
        ));
        assert!(error.to_string().contains("source-variable name"));

        assert!(matches!(
            require_complete_simulation_debug_source_capture_v2(Some(
                fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::ResourceLimit,
            )),
            Err(
                ProductionPipelineError::SimulationDebugSourceCaptureUnavailable(
                    fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::ResourceLimit
                )
            )
        ));
    }

    #[test]
    fn target_workgroup_roster_rejects_empty_and_missing_entries_without_omission() {
        let empty = fe2o3_kernel_ir::Module::new("empty_workgroups");
        assert!(exact_target_workgroup_roster_v1(&empty).is_err());

        let mut module = fe2o3_kernel_ir::Module::new("workgroups");
        let mut first = fe2o3_kernel_ir::Kernel::new(
            "first",
            "first",
            fe2o3_kernel_ir::LaunchDomain::D1 {
                x: fe2o3_kernel_ir::LaunchExtent::Static(64),
            },
        );
        first.workgroup_size = Some(fe2o3_kernel_ir::WorkgroupSize::new(64, 1, 1));
        let second = fe2o3_kernel_ir::Kernel::new(
            "second",
            "second",
            fe2o3_kernel_ir::LaunchDomain::D1 {
                x: fe2o3_kernel_ir::LaunchExtent::Static(64),
            },
        );
        module.kernels.extend([first, second]);
        assert!(exact_target_workgroup_roster_v1(&module).is_err());

        module.kernels[1].workgroup_size = Some(fe2o3_kernel_ir::WorkgroupSize::new(128, 1, 1));
        assert_eq!(
            exact_target_workgroup_roster_v1(&module).unwrap().as_ref(),
            &[
                (
                    "first".to_owned(),
                    fe2o3_kernel_ir::WorkgroupSize::new(64, 1, 1),
                ),
                (
                    "second".to_owned(),
                    fe2o3_kernel_ir::WorkgroupSize::new(128, 1, 1),
                ),
            ],
        );
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
    fn extraction_custody_cannot_enter_protected_publication() {
        assert!(matches!(
            ProductionCompilerCustody::extraction_only().into_publication_custody(),
            Err(ProductionPipelineError::ExtractionCannotPublish)
        ));
    }

    #[test]
    fn production_layout_binding_uses_the_measured_worker_spelling() {
        let legacy = format!(
            "target triple = \"amdgcn-amd-amdhsa\"\ntarget datalayout = \"{}\"\n\ndefine void @body() {{ ret void }}\n",
            dialect_amdgcn::GFX942_XNACK_MINUS_DATA_LAYOUT
        );
        let bound = dialect_amdgcn::bind_production_llvm22_worker_layout_v1(&legacy).unwrap();
        assert!(bound.starts_with(&format!(
            "target triple = \"amdgcn-amd-amdhsa\"\ntarget datalayout = \"{}\"\n\n",
            crate::production_target_v1::PRODUCTION_WORKER_DATA_LAYOUT_V1
        )));
        assert!(bound.contains("target datalayout = \"e-p:64:64-"));
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
            assert!(dialect_amdgcn::bind_production_llvm22_worker_layout_v1(&hostile).is_err());
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
        let retained = include_str!("production_pipeline/retained_target_v30.rs");
        assert!(retained.contains("dialect_amdgcn::bind_production_target_v1("));
        let bind = retained
            .find("dialect_amdgcn::bind_production_target_v1(")
            .unwrap();
        for entry in [
            "fe2o3_kernel_opt::optimize_production_kernel_ir_module_v2(&target_module)",
            "fe2o3_kernel_opt::optimize_production_kernel_ir_module_v3(&target_module)",
        ] {
            assert!(bind < retained.find(entry).expect("fixed optimizer entry"));
        }
        assert!(transaction.contains("dialect_amdgcn::bind_production_llvm22_worker_layout_v1("));
        let retain = transaction
            .find("RetainedProductionTargetV30::try_lower(")
            .unwrap();
        let lower = transaction
            .find("lower_compiler_module_to_xnack_minus_llvm_ir_with_physical_launch_v2(")
            .unwrap();
        assert!(retain < lower);
        assert!(!retained.contains("optimize_kernel_ir_module_v"));
        assert!(!retained.contains(".clone()"));
        let target_lowered = source
            .split_once("impl TargetLoweredProductionCompilation {")
            .expect("target-lowered implementation")
            .1
            .split_once("\nfn require_complete_simulation_debug_source_capture_v2(")
            .expect("target-lowered implementation end")
            .0;
        assert_eq!(target_lowered.matches("optimized.report(),").count(), 2);
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
        let ssa = transaction
            .find(".construct_semantic_ssa()?")
            .expect("mandatory semantic SSA custody");
        let materialize = transaction
            .find(".materialize_target_neutral()?")
            .expect("fixed pre-ranked materialization");
        let lower = transaction
            .find(".attach_target_neutral_checks()?")
            .expect("target-neutral lowering");
        assert!(
            ssa < materialize && materialize < verify && verify < lower,
            "semantic SSA, ranked verification, and lowering typestates are out of order",
        );
        let projection = include_str!("production_ranked_projection_v1.rs");
        assert!(
            projection
                .contains("include!(\"production_ranked_projection_v1/root_recipe_core_v1.rs\");")
        );
        assert!(projection.contains("verify_prepared_ranked_root_recipe_v1("));
        let recipe = include_str!("production_ranked_projection_v1/root_recipe_core_v1.rs");
        assert!(recipe.contains("prepare_reference_effect_request_v2"));
    }

    #[test]
    fn referenced_kernels_complete_all_functional_gates_before_checked_attachment() {
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
            .find(".into_verified_roster_receipt_with_runtime(")
            .expect("ranked roster verification transition");
        let module = pipeline[roster..]
            .find(".into_module_verified_receipt()")
            .map(|offset| roster + offset)
            .expect("complete ranked module receipt transition");
        let lowering = pipeline[module..]
            .find("ProductionSemanticKirOwnerV1::try_attach_materialized_ranked_checks")
            .map(|offset| module + offset)
            .expect("KIR lowering transition");
        assert!(
            roster < module && module < lowering,
            "checked attachment ran before functional verification"
        );
    }

    #[test]
    fn ranked_roster_receipt_reaches_complete_module_kir_authority() {
        let pipeline = include_str!("production_pipeline.rs");
        let roster = pipeline
            .find(".into_verified_roster_receipt_with_runtime(")
            .expect("ranked roster receipt transition");
        let module = pipeline[roster..]
            .find(".into_module_verified_receipt()")
            .map(|offset| roster + offset)
            .expect("complete module receipt authority");
        let kir = pipeline[module..]
            .find("ProductionSemanticKirOwnerV1::try_attach_materialized_ranked_checks")
            .map(|offset| module + offset)
            .expect("KIR authority transition");
        assert!(roster < module && module < kir);
        assert!(!pipeline.contains(concat!("MultiRoot", "TargetNeutralLowering")));
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
        let compiler_custody = pipeline
            .split_once("enum ProductionCompilerCustody {")
            .unwrap()
            .1
            .split_once("struct AuthenticatedProductionBindings {")
            .unwrap()
            .0;
        let prepared = pipeline
            .split_once("struct PreparedProductionWorkerPublication {")
            .unwrap()
            .1
            .split_once("impl AuthenticatedProductionTargetModule {")
            .unwrap()
            .0;
        for compiler_owner in [compiler_custody, prepared] {
            assert!(compiler_owner.contains("AdmittedProtectedRustcInvocationV1"));
            assert!(!compiler_owner.contains("AdmittedProtectedCompilerExecutionV1"));
        }
        let publication = pipeline
            .split_once("pub(crate) fn publish_worker_handoff(")
            .unwrap()
            .1
            .split_once("/// Retains the original extraction milestone")
            .unwrap()
            .0;
        assert!(publication.contains("compiler_execution: AdmittedProtectedCompilerExecutionV1"));
        assert!(
            publication.contains(
                ".publish_predicated_worker_handoff_v90(target_budget, compiler_execution)"
            )
        );
        assert!(!publication.contains("lower_production_target("));
        assert!(pipeline.contains(concat!("publish_compiler_module_handoff", "_v3")));
        assert!(pipeline.contains(concat!(
            "publish_compiler_execution_receipt_transport",
            "_v1"
        )));
        assert!(!pipeline.contains(concat!(
            "let invocation_",
            "descriptor = invocation.descriptor().clone()"
        )));
        let _: fn(
            crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3,
            &crate::protected_rustc_invocation::FinishedProtectedRustcInvocationV3,
            fe2o3_compiler_ffi::DeviceTargetV1,
            &fe2o3_compiler_ffi::CompilerDescriptorSourceV1,
            fe2o3_compiler_ffi::CompilerModuleHandoffV2,
        ) -> Result<
            fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV3,
            crate::production_semantic_lineage_v3::ProductionSemanticLineageErrorV3,
        > = crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3::finish;
        let _: fn(
            crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3,
            fe2o3_rustc_invocation::RustcInvocationDescriptorV3,
            fe2o3_compiler_ffi::DeviceTargetV1,
            &fe2o3_compiler_ffi::CompilerDescriptorSourceV1,
            fe2o3_compiler_ffi::CompilerModuleHandoffV2,
        ) -> Result<
            fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV3,
            crate::production_semantic_lineage_v3::ProductionSemanticLineageErrorV3,
        > = crate::production_semantic_lineage_v3::PreparedProductionSemanticLineageV3::finish_for_inert_extraction;
        assert!(lineage.contains("invocation_custody: &FinishedProtectedRustcInvocationV3"));
        let inert_lineage_boundary = lineage
            .find("pub(crate) fn finish_for_inert_extraction(")
            .expect("explicit inert extraction boundary");
        assert!(
            !lineage[..inert_lineage_boundary].contains("invocation: RustcInvocationDescriptorV3")
        );
        assert_eq!(
            lineage[inert_lineage_boundary..]
                .matches("invocation: RustcInvocationDescriptorV3")
                .count(),
            2,
            "raw descriptors are confined to the inert extraction boundary and its helper",
        );

        let protected_lineage = lineage
            .find("pub(crate) fn finish(")
            .expect("protected lineage finalizer remains explicit");
        let inert_lineage = lineage[protected_lineage..]
            .find("pub(crate) fn finish_for_inert_extraction(")
            .map(|offset| protected_lineage + offset)
            .expect("inert lineage finalizer remains explicit");
        let protected_lineage = &lineage[protected_lineage..inert_lineage];
        let lineage_revalidation = protected_lineage
            .find(".revalidate_for_publication()")
            .expect("protected lineage revalidates live invocation custody");
        let lineage_descriptor = protected_lineage
            .find(".descriptor().clone()")
            .expect("protected lineage derives its descriptor from live custody");
        assert!(lineage_revalidation < lineage_descriptor);
        assert!(!protected_lineage.contains("invocation: RustcInvocationDescriptorV3"));
        assert!(!protected_lineage.contains("finish_for_inert_extraction"));

        let inert_pipeline = pipeline
            .find("pub(crate) fn into_inert_semantic_worker_handoff_for_extraction(")
            .expect("authority-free extraction path remains explicit");
        let protected_prepare = pipeline[inert_pipeline..]
            .find("fn prepare_worker_handoff(")
            .map(|offset| inert_pipeline + offset)
            .expect("protected publication path follows inert extraction");
        let inert_pipeline = &pipeline[inert_pipeline..protected_prepare];
        assert!(inert_pipeline.contains("if !transaction.compiler_custody.is_extraction_only()"));
        assert!(inert_pipeline.contains(".finish_for_inert_extraction("));
        for protected_operation in [
            concat!("publish_compiler_module_handoff", "_v3"),
            "InertCompilerExecutionSubjectV1::from_publication",
            ".acquire(subject.clone())",
            concat!("publish_compiler_execution_receipt_transport", "_v1"),
        ] {
            assert!(
                !inert_pipeline.contains(protected_operation),
                "inert extraction reached protected operation: {protected_operation}",
            );
        }
        assert_eq!(
            pipeline
                .matches(concat!("publish_compiler_module_handoff", "_v3"))
                .count(),
            1,
            "production retains one durable compiler-module publication path",
        );

        let protected_pipeline = pipeline[protected_prepare..]
            .find("fn publish_worker_handoff(")
            .map(|offset| protected_prepare + offset)
            .expect("protected publication method remains explicit");
        let protected_pipeline_end = pipeline[protected_pipeline..]
            .find("\nfn require_complete_simulation_debug_source_capture_v2(")
            .map(|offset| protected_pipeline + offset)
            .expect("protected publication method remains bounded");
        let protected_pipeline = &pipeline[protected_pipeline..protected_pipeline_end];
        assert!(!protected_pipeline.contains("finish_for_inert_extraction"));
        assert!(!protected_pipeline.contains("RustcInvocationDescriptorV3"));
        let lineage_finish = protected_pipeline
            .find(".semantic_lineage\n            .finish(\n                &invocation,")
            .expect("semantic lineage consumes live protected invocation custody");
        let final_revalidation = protected_pipeline[lineage_finish..]
            .find("invocation\n            .revalidate_for_publication()")
            .map(|offset| lineage_finish + offset)
            .expect("protected invocation is revalidated after lineage construction");
        let durable_publication = protected_pipeline[lineage_finish..]
            .find(concat!("publish_compiler_module_handoff", "_v3"))
            .map(|offset| lineage_finish + offset)
            .expect("strict V3 handoff publication remains present");
        let execution_subject = protected_pipeline[lineage_finish..]
            .find("InertCompilerExecutionSubjectV1::from_publication")
            .map(|offset| lineage_finish + offset)
            .expect("strict publication derives one canonical compiler-execution subject");
        assert!(lineage_finish < final_revalidation && final_revalidation < durable_publication);
        assert!(durable_publication < execution_subject);
        let receipt_acquisition = protected_pipeline[execution_subject..]
            .find(".acquire(subject.clone())")
            .map(|offset| execution_subject + offset)
            .expect("exact execution subject is sent to the protected issuer");
        let receipt_transport = protected_pipeline[receipt_acquisition..]
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

mod physical_global_copy_target_v21;
pub(crate) use physical_global_copy_target_v21::AuthenticatedPhysicalGlobalCopyTargetModuleV21;

mod physical_lds_exchange_target_v22;
pub(crate) use physical_lds_exchange_target_v22::AuthenticatedPhysicalLdsExchangeTargetModuleV22;

/// Borrowed actual prepared-input custody. Fields and construction belong only
/// to this pipeline module; projection siblings cannot substitute equal clones.
/// Non-Clone, no readiness, no owned proof token, no mutable inputs or budget.
pub(crate) struct ActualRetainedRankedInputsV1<'a> {
    owner: &'a fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    inputs: &'a [crate::production_ranked_projection_v1::ProductionRankedRootInputV1],
    bindings: &'a crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
}
impl ActualRetainedRankedInputsV1<'_> {
    pub(crate) fn belongs_to(
        &self,
        owner: &fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    ) -> bool {
        std::ptr::eq(self.owner, owner)
    }
    pub(crate) fn inputs(
        &self,
    ) -> &[crate::production_ranked_projection_v1::ProductionRankedRootInputV1] {
        self.inputs
    }
    pub(crate) fn bindings(
        &self,
    ) -> &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1 {
        self.bindings
    }
}
/// Shared private loan of the caller's actual retained owner/inputs/bindings.
/// HRTB prevents the view or its borrows from escaping. Accepted frame credits
/// remain in the caller's original account; this factory never refunds them.
pub(crate) fn with_actual_retained_ranked_inputs_v1<R, E, F>(
    owner: &fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    inputs: &[crate::production_ranked_projection_v1::ProductionRankedRootInputV1],
    bindings: &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    owned_frame: &mut usize,
    inspect: F,
) -> Result<R, E>
where
    E: From<fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1>,
    F: for<'a> FnOnce(
        ActualRetainedRankedInputsV1<'a>,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
        &mut usize,
    ) -> Result<R, E>,
{
    use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1 as Resource;
    let mut frame = 4096usize;
    for amount in [
        std::mem::size_of::<ActualRetainedRankedInputsV1<'static>>(),
        std::mem::size_of::<F>()
            .checked_mul(2)
            .ok_or(Resource::Arithmetic)?,
        std::mem::size_of::<Result<R, E>>()
            .checked_mul(2)
            .ok_or(Resource::Arithmetic)?,
    ] {
        frame = frame.checked_add(amount).ok_or(Resource::Arithmetic)?;
    }
    let total = owned_frame.checked_add(frame).ok_or(Resource::Arithmetic)?;
    budget.charge_work(frame)?;
    budget.reserve_storage(frame)?;
    *owned_frame = total;
    inspect(
        ActualRetainedRankedInputsV1 {
            owner,
            inputs,
            bindings,
        },
        budget,
        owned_frame,
    )
}

/// Genuine observers reuse the same factory and its original error type.
/// Their callback shape remains unchanged; no replacement source loan exists.
#[cfg(test)]
fn with_actual_retained_ranked_inputs_for_test_v1<R, F>(
    owner: &fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    inputs: &[crate::production_ranked_projection_v1::ProductionRankedRootInputV1],
    bindings: &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
    budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    owned_frame: &mut usize,
    inspect: F,
) -> Result<R, fe2o3_lower_mir_kernel::Bf16CallInstanceErrorV1>
where
    F: for<'a> FnOnce(
        ActualRetainedRankedInputsV1<'a>,
        &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<R, fe2o3_lower_mir_kernel::Bf16CallInstanceErrorV1>,
{
    with_actual_retained_ranked_inputs_v1(
        owner,
        inputs,
        bindings,
        budget,
        owned_frame,
        move |actual, budget, _| inspect(actual, budget),
    )
}

#[cfg(test)]
mod storage_descriptor_profile_tests {
    use super::*;

    use fe2o3_kernel_ir::{AccessMode, AddressSpace, ScalarType, StorageLayoutIdV1, Type};
    #[test]
    fn legacy_descriptor_slots_refuse_storage_even_behind_a_pointer_or_slice() {
        let storage = Type::StorageObject(StorageLayoutIdV1(31));
        for ty in [
            storage.clone(),
            Type::pointer(
                storage.clone(),
                AddressSpace::Private,
                AccessMode::ReadWrite,
            ),
            Type::slice(storage, AddressSpace::Global, AccessMode::ReadOnly),
        ] {
            assert!(compiler_legacy_slot_shape_v2(&ty).is_err());
        }
        assert_eq!(
            compiler_legacy_slot_shape_v2(&Type::Scalar(ScalarType::U32)).unwrap(),
            (4, 4, None)
        );
        assert_eq!(
            compiler_legacy_slot_shape_v2(&Type::pointer(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Global,
                AccessMode::ReadWrite
            ))
            .unwrap(),
            (8, 8, None)
        );
        assert_eq!(
            compiler_legacy_slot_shape_v2(&Type::slice(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Global,
                AccessMode::ReadOnly
            ))
            .unwrap(),
            (8, 8, Some(8))
        );
    }
}

#[cfg(test)]
mod storage_component_profile_tests {
    use super::*;

    use fe2o3_kernel_ir::{
        AccessMode, AddressSpace, FunctionBody, ScalarType, SemanticKernargSlotV2,
        StorageLayoutIdV1, Type, ValueId,
    };
    #[test]
    fn component_map_cannot_relabel_storage_pointers_as_old_scalar_regions() {
        let body = FunctionBody {
            parameters: vec![ValueId(7)],
            blocks: vec![],
        };
        let slot = [(SemanticKernargSlotV2::new(0, 8, 8), None)];
        let storage = Type::StorageObject(StorageLayoutIdV1(0));
        for ty in [
            storage.clone(),
            Type::pointer(storage, AddressSpace::Global, AccessMode::ReadOnly),
        ] {
            assert!(
                compiler_component_storage_v2(vec![], ValueId(7), &body, &[ty], &slot).is_err()
            );
        }
        assert!(
            compiler_component_storage_v2(
                vec![],
                ValueId(7),
                &body,
                &[Type::Scalar(ScalarType::U64)],
                &slot
            )
            .is_ok()
        );
    }
}

#[path = "production_bindings_debug_retained_storage_v1.rs"]
pub(crate) mod bindings_debug_retained_storage_v1;

#[path = "production_bindings_context_retained_storage_v1.rs"]
mod context_retained_storage_v1;

#[path = "production_bindings_typed_descriptor_retained_storage_v1.rs"]
mod bindings_typed_descriptor_retained_storage_v1;

#[path = "production_bindings_reference_retained_storage_v1.rs"]
mod reference_retained_storage_v1;

#[path = "production_bindings_retained_storage_v1.rs"]
pub(crate) mod bindings_retained_storage_v1;

#[path = "production_pipeline/bf16_tile_source_promotion_v1.rs"]
mod bf16_tile_source_promotion_v1;

#[path = "production_pipeline/retained_materialization_phase_v1.rs"]
mod retained_materialization_phase_v1;

/// Private nominal formal stage. Actual obligations and sealed nominal source
/// remain unsplit with their original projection phase; transaction bindings
/// are the same moved values. The enclosing retained materialization phase
/// keeps its distinct original source account alive after this entire stage.
#[allow(dead_code)]
pub(crate) struct PrivateBf16FormalMemoryCompilationV1 {
    admitted: crate::production_ranked_projection_v1::PrivateBf16FormalMemoryV1,
    bindings: AuthenticatedProductionBindings,
}

impl RankedVerifiedProductionCompilation {
    #[allow(dead_code)]
    fn admit_private_bf16_formal_memory_v1(
        self,
        requested_return: [u8; 4],
    ) -> Result<PrivateBf16FormalMemoryCompilationV1, ProductionPipelineError> {
        let Self { ranked, bindings } = self;
        let admitted = ranked
            .into_private_bf16_formal_memory_v1(requested_return)
            .map_err(ProductionPipelineError::RankedVerification)?;
        Ok(PrivateBf16FormalMemoryCompilationV1 { admitted, bindings })
    }
}

impl PrivateBf16FormalMemoryCompilationV1 {
    #[allow(dead_code)]
    fn revalidate_private_bf16_formal_memory_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.admitted
            .revalidate_private_bf16_formal_memory_v1(requested_return)
            .map_err(ProductionPipelineError::RankedVerification)
    }
}

/// Private exact geometry and actual target binding. The first field owns the
/// bound output before the intact formal/source pair and projection phase.
/// Original authenticated bindings and the outer materialization account stay
/// unchanged. This is not optimizer, LLVM, artifact or launch admission.
#[allow(dead_code)]
pub(crate) struct PrivateBf16TargetBoundCompilationV1 {
    bound: crate::production_ranked_projection_v1::PrivateBf16TargetBoundV1,
    bindings: AuthenticatedProductionBindings,
}

impl PrivateBf16FormalMemoryCompilationV1 {
    #[allow(dead_code)]
    fn bind_private_bf16_target_v1(
        self,
        requested_return: [u8; 4],
    ) -> Result<PrivateBf16TargetBoundCompilationV1, ProductionPipelineError> {
        let Self { admitted, bindings } = self;
        let bound = admitted.bind_private_bf16_target_v1(
            requested_return,
            &bindings.typed_descriptor_roots,
            bindings.rustc_target.profile(),
        )?;
        Ok(PrivateBf16TargetBoundCompilationV1 { bound, bindings })
    }
}

impl PrivateBf16TargetBoundCompilationV1 {
    #[allow(dead_code)]
    fn revalidate_private_bf16_target_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.bound.revalidate_private_bf16_target_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }
}

/// Private checked V12 B -> O stage. This retains the original authenticated
/// bindings and intact target/formal/source owner. LLVM and ordinary admission
/// remain unavailable; O's local rewrite proof is not fresh formal admission.
#[allow(dead_code)]
pub(crate) struct PrivateBf16OptimizedCompilationV1 {
    optimized: crate::production_ranked_projection_v1::PrivateBf16OptimizedV1,
    bindings: AuthenticatedProductionBindings,
}

impl PrivateBf16TargetBoundCompilationV1 {
    #[allow(dead_code)]
    fn optimize_private_bf16_target_v1(
        self,
        requested_return: [u8; 4],
    ) -> Result<PrivateBf16OptimizedCompilationV1, ProductionPipelineError> {
        let Self { bound, bindings } = self;
        let optimized = bound.optimize_private_bf16_target_v1(
            requested_return,
            &bindings.typed_descriptor_roots,
            bindings.rustc_target.profile(),
        )?;
        Ok(PrivateBf16OptimizedCompilationV1 {
            optimized,
            bindings,
        })
    }
}

impl PrivateBf16OptimizedCompilationV1 {
    #[allow(dead_code)]
    fn revalidate_private_bf16_optimization_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.optimized.revalidate_private_bf16_optimization_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }
}

impl PrivateBf16OptimizedCompilationV1 {
    #[allow(dead_code)]
    pub(crate) fn verify_private_bf16_output_safety_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.optimized.verify_private_bf16_output_safety_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }
}

impl PrivateBf16OptimizedCompilationV1 {
    #[allow(dead_code)]
    pub(crate) fn verify_private_bf16_output_guarded_safety_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.optimized.verify_private_bf16_output_guarded_safety_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }
}

/// Inert private LLVM text retains the actual O/B/source/formal/target owner and
/// authenticated bindings. It is not a legacy TargetLowered/Connected stage and
/// cannot publish a descriptor, Worker request, object, or launch.
#[allow(dead_code)]
pub(crate) struct PrivateBf16LlvmCompilationV1 {
    llvm: crate::production_ranked_projection_v1::PrivateBf16LlvmV1,
    bindings: AuthenticatedProductionBindings,
}
impl PrivateBf16OptimizedCompilationV1 {
    #[allow(dead_code)]
    fn lower_private_bf16_llvm_v1(
        self,
        requested_return: [u8; 4],
    ) -> Result<PrivateBf16LlvmCompilationV1, ProductionPipelineError> {
        let Self {
            optimized,
            bindings,
        } = self;
        let llvm = optimized.lower_private_bf16_llvm_v1(
            requested_return,
            &bindings.typed_descriptor_roots,
            bindings.rustc_target.profile(),
        )?;
        Ok(PrivateBf16LlvmCompilationV1 { llvm, bindings })
    }
}
impl PrivateBf16LlvmCompilationV1 {
    #[allow(dead_code)]
    fn revalidate_private_bf16_llvm_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.llvm.revalidate_private_bf16_llvm_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }
}

/// Private descriptor bytes and exact descriptor-bound LLVM, retaining the
/// actual LLVM/O/source/formal owner and authenticated compiler bindings.
/// No conversion to TargetLowered, OutputOwnerV1, signed source or Worker stage.
#[allow(dead_code)]
pub(crate) struct PrivateBf16DescriptorCompilationV1 {
    descriptor: crate::production_ranked_projection_v1::PrivateBf16DescriptorV1,
    bindings: AuthenticatedProductionBindings,
}
impl PrivateBf16LlvmCompilationV1 {
    #[allow(dead_code)]
    fn prepare_private_bf16_descriptor_v1(
        self,
        requested_return: [u8; 4],
    ) -> Result<PrivateBf16DescriptorCompilationV1, ProductionPipelineError> {
        let Self { llvm, bindings } = self;
        let descriptor = llvm.prepare_private_bf16_descriptor_v1(
            requested_return,
            &bindings.typed_descriptor_roots,
            bindings.rustc_target.profile(),
        )?;
        Ok(PrivateBf16DescriptorCompilationV1 {
            descriptor,
            bindings,
        })
    }
}
impl PrivateBf16DescriptorCompilationV1 {
    #[allow(dead_code)]
    fn revalidate_private_bf16_descriptor_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.descriptor.revalidate_private_bf16_descriptor_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }

    #[allow(dead_code)]
    const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

/// Private boxed V2 handoff custody. This does not invoke a Worker, detach the
/// descriptor/source chain, or create ordinary/protected publication authority.
#[allow(dead_code)]
pub(crate) struct PrivateBf16WorkerHandoffCompilationV1 {
    handoff: crate::production_ranked_projection_v1::PrivateBf16WorkerHandoffV1,
    bindings: AuthenticatedProductionBindings,
}
impl PrivateBf16DescriptorCompilationV1 {
    #[allow(dead_code)]
    fn prepare_private_bf16_worker_handoff_v1(
        self,
        requested_return: [u8; 4],
    ) -> Result<PrivateBf16WorkerHandoffCompilationV1, ProductionPipelineError> {
        let Self {
            descriptor,
            bindings,
        } = self;
        let handoff = descriptor.prepare_private_bf16_worker_handoff_v1(
            requested_return,
            &bindings.typed_descriptor_roots,
            bindings.rustc_target.profile(),
        )?;
        Ok(PrivateBf16WorkerHandoffCompilationV1 { handoff, bindings })
    }
}
impl PrivateBf16WorkerHandoffCompilationV1 {
    #[allow(dead_code)]
    fn revalidate_private_bf16_worker_handoff_v1(
        &mut self,
        requested_return: [u8; 4],
    ) -> Result<(), ProductionPipelineError> {
        self.handoff.revalidate_private_bf16_worker_handoff_v1(
            requested_return,
            &self.bindings.typed_descriptor_roots,
            self.bindings.rustc_target.profile(),
        )
    }

    #[allow(dead_code)]
    const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

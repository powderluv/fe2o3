//! Bounded planning and result records for an external GPU-kernel verifier.
//!
//! V1 constructs canonical proof requests and executes an evidence recorder
//! through a bounded, shell-free process boundary. It measures and seals
//! recorder, claimed-verifier, and claimed-solver images, but launches only the
//! recorder. On Linux x86_64, V2 separately launches pinned solver and Verus
//! snapshots under a pidfd-owned, two-nonce, process-creation-denied controller
//! protocol with ptrace-unresumable checkpoints. It records normalized executable
//! baselines, anonymous mappings, live executable-page bytes, and runtime/security
//! state. Those checkpoint identities do not imply exclusive measured-image
//! execution between observations. Stock Verus/Z3 integration remains future
//! work, and neither path grants proof or GPU authority. The legacy planning path
//! retains caller-supplied identities for compatibility.

#[cfg(test)]
extern crate self as fe2o3_verifier;

mod artifact_record;
mod authenticated_execution;
mod authenticated_proof_binding;
mod authenticated_verus_execution_v2;
mod compiler_multi_root_proof_v1;
mod compiler_multi_root_target_lineage_v1;
mod compiler_native_conditional_handoff_v5;
mod compiler_native_conditional_policy_roster_v1;
mod native_conditional_root_policy_file_v1;
pub use compiler_native_conditional_handoff_v5::{
    CanonicalRefinedForwardingHistoryLimitsV1, CompilerConditionalNativeSemanticHandoffErrorV5,
    RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
    RecoveredCompilerConditionalNativeSemanticHandoffV5,
    recover_compiler_conditional_native_semantic_handoff_in_original_account_v5,
    recover_compiler_conditional_native_semantic_handoff_v5,
    recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_in_original_account_v5,
    recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_v5,
    recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_in_original_account_v5,
    recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_v5,
};
pub use compiler_native_conditional_policy_roster_v1::{
    InertNativeConditionalPolicyRosterV1, NativeConditionalPolicyReconstructionErrorV1,
    NativeConditionalPolicyRosterStorageV1, native_conditional_root_policy_input_storage_v2,
    reconstruct_inert_native_conditional_policy_roster_in_original_account_v1,
    reconstruct_inert_native_conditional_policy_roster_v1,
};
pub use native_conditional_root_policy_file_v1::{
    MAX_NATIVE_CONDITIONAL_ROOT_POLICY_FILE_BYTES_V1,
    encode_native_conditional_root_policy_file_v1,
    recover_native_conditional_handoff_under_policy_file_v1,
    recover_native_conditional_handoff_under_policy_file_with_cpu_mapping_v1,
    recover_native_conditional_handoff_under_policy_file_with_cpu_origins_v1,
    validate_native_conditional_root_policy_file_v1,
};
pub mod compiler_native_conditional_source_packet_v2;
pub use compiler_native_conditional_source_packet_v2::{
    NativeConditionalPacketErrorV2, NativeConditionalSourcePacketInputV2,
    NativeConditionalSourcePacketStorageV2, NativeConditionalSourceRootV2,
    encode_native_conditional_source_packet_v2, with_decoded_native_conditional_source_packet_v2,
};
mod compiler_native_source_proof_v1;
mod compiler_native_symbol_manifest_v1;
mod compiler_nominal_abi_v3;
mod compiler_proof_binding_v3;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod compiler_proof_broker_v1;
mod compiler_refined_forwarding_output_v1;
mod compiler_target_lineage_v1;
mod conditional_contract_request_v2;
mod conditional_fill_program_v1;
mod conditional_fill_refinement_v1;
mod conditional_output_evidence_v1;
mod conditional_ranked_formulas_v1;
mod mixed_native_correspondence_v60;
mod mixed_native_correspondence_v89;
mod mixed_optimizer_refinement_v26;
mod mixed_target_selection_family;
mod mixed_target_selection_v53;
mod mixed_target_selection_v89;
mod native_copy_classification_v1;
pub use mixed_optimizer_refinement_v26::{
    AggregateMemoryCfgObligationV30, AggregateMemoryCfgSubjectV30,
    ExecutedMixedComposedRefinementV29, ExecutedMixedOptimizerBlockSimulationV26,
    ExecutedMixedPureCseCfgRefinementV27, ExecutedMixedWorklistCfgRefinementV27,
    ExecutedPredicatedTypedSourceTailV90, ExecutedTypedSourceTailV50, ExpandedSupportCallKindV280,
    ExpandedSupportCallV280, ExpandedSupportCensusV280, ExpandedSupportCursorV280,
    ExpandedSupportCutV280, ExpandedSupportForwardingV288, ExpandedSupportFrameDemandV281,
    ExpandedSupportInstanceV280, ExpandedSupportModelV280, ExpandedSupportRootV280,
    ExpandedSupportRuntimeV280, InertPredicatedTypedSourceReceiptV90, InertTypedSourceReceiptV53,
    MixedOptimizerCfgSubjectV27, MixedOptimizerReconstructionRefusalV285,
    MixedOptimizerRefinementErrorV26, MixedOptimizerRefinementSubjectV26,
    MixedOptimizerRelocationCfgSubjectV28, MixedOptimizerRelocationErrorV28,
    MixedOptimizerRelocationSubjectV28, OriginalSemanticMirRefinementSubjectV30,
    OriginalSemanticMirRefinementSubjectV31, OriginalSemanticMirRefinementSubjectV36,
    PredicatedTypedSourceTailSubjectV90, PreparedMixedComposedExecutionV29,
    PreparedMixedComposedRelocationCfgRefinementV28,
    PreparedMixedFixedpointComposedRelocationCfgRefinementV29,
    PreparedMixedFixedpointRelocationCfgRefinementV29,
    PreparedMixedFixedpointRelocationExpressionsV29, PreparedMixedOptimizerRefinementV26,
    PreparedMixedPureCseCfgRefinementV27, PreparedMixedRelocationCfgRefinementV28,
    PreparedMixedRelocationExpressionsV28, PreparedMixedWorklistCfgRefinementV27,
    PreparedOriginalSemanticMirRefinementV30, PreparedOriginalSemanticMirRefinementV31,
    PreparedOriginalSemanticMirRefinementV36, PreparedPredicatedTypedSourceTailExecutionV90,
    PreparedPredicatedTypedSourceTailV90, PreparedTypedSourceTailExecutionV50,
    PreparedTypedSourceTailV50, SourceScalarReferenceInputV69, TypedSourceTailSubjectV50,
    check_inert_predicated_typed_source_receipt_v90, check_inert_typed_source_receipt_v53,
    execute_mixed_optimizer_block_simulation_v26, execute_mixed_pure_cse_cfg_refinement_v27,
    execute_mixed_worklist_cfg_refinement_v27, prepare_aggregate_memory_cfg_obligation_v30,
    prepare_mixed_fixedpoint_relocation_expressions_v29, prepare_mixed_optimizer_refinement_v26,
    prepare_mixed_pure_cse_cfg_refinement_v27, prepare_mixed_relocation_expressions_v28,
    prepare_mixed_worklist_cfg_refinement_v27, prepare_original_semantic_mir_refinement_v30,
    prepare_original_semantic_mir_refinement_v31, prepare_original_semantic_mir_refinement_v36,
    prepare_predicated_typed_source_tail_v90,
    prepare_predicated_typed_source_tail_with_references_v90, prepare_typed_source_tail_v50,
    prepare_typed_source_tail_with_references_v69, with_expanded_support_model_v280,
};
pub mod conditional_reference_v1;
mod control_flow_binding;
mod executor;
mod functional_refinement_receipt_v2;
mod functional_refinement_runtime_v1;
mod generated_verus_proof_input_v3;
mod mir_pliron_per_compilation_verus_v1;
mod mir_pliron_verus_execution_evidence_v1;
mod model;
mod monomorphization_dead_binding;
mod multi_kernel_proof;
mod persistent_freshness;
mod plan;
pub mod portable_reference_v1;
mod production_kir_to_llvm_replay_v1;
mod proof_capsule;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod proof_executor_helper_v1;
mod result;
mod retained_functional_refinement_runtime_v1;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use proof_executor_helper_v1::{
    ProofExecutorHelperErrorV1, run_inherited_proof_executor_helper_v1,
};
mod static_view_proof;

pub mod compiler_native_conditional_source_proof_v2;
pub use compiler_native_conditional_source_proof_v2::final_replay::{
    NativeConditionalFinalErrorV2, NativeConditionalFinalInputsV2,
    validate_native_conditional_source_through_f_v2,
    validate_native_conditional_source_through_f_with_cpu_mapping_v2,
    validate_native_conditional_source_through_f_with_cpu_origins_v2,
};
pub use compiler_native_conditional_source_proof_v2::{
    NativeConditionalCpuExpectationV1, NativeConditionalCpuMappingContextV1,
    NativeConditionalCpuMappingExpectationV1, NativeConditionalCpuOriginExpectationV1,
    NativeConditionalRootPolicyV2, NativeConditionalSourceProofErrorV2,
    NativeConditionalSourceStorageV2, ReplayedNativeConditionalSourceV2,
    select_native_conditional_ownership_site_v2, validate_native_conditional_source_packet_v2,
    validate_native_conditional_source_packet_with_cpu_mapping_v2,
    validate_native_conditional_source_packet_with_cpu_origins_v2,
};

pub use artifact_record::{
    ArtifactProofEvidenceV1, ArtifactRecordConversionError, ReviewedInvocationIdentityV1,
    canonical_invocation_digest, convert_to_artifact_proof_record,
};
pub use authenticated_execution::{
    AuthenticatedBindingField, AuthenticatedExecutionError, AuthenticatedRecorderOutputV1,
    AuthenticatedResultError, BoundExecutionPayloadV1, DataOperation, ExecutableMeasurementV1,
    ExecutableOperation, ExecutableRole, MAX_EXECUTABLE_BYTES, MeasuredRecorderInputsV1,
    execute_authenticated_recorder,
};
pub use authenticated_verus_execution_v2::{
    AuthenticatedVerusExecutionDependencyV2, AuthenticatedVerusExecutionErrorKindV2,
    AuthenticatedVerusExecutionErrorV2, AuthenticatedVerusExecutionInputsV2,
    AuthenticatedVerusExecutionPolicyV2, AuthenticatedVerusExecutionReceiptV2,
    AuthenticatedVerusProcessOccurrenceV2, AuthenticatedVerusToolExecutionV2,
    BoundExecutionPayloadV2, ProcessFailureV2, RuntimeClosureMeasurementV2,
    RuntimeExecutableBaselineV2, VerusExecutionRoleV2, execute_authenticated_verus_v2,
};
pub use compiler_multi_root_proof_v1::{
    CompilerMultiRootProofValidationErrorV1, ValidatedCompilerKernelIrV1,
    ValidatedCompilerMultiRootKernelIrV1, ValidatedCompilerMultiRootProofInputsV1,
    ValidatedCompilerMultiRootProofRootV1, validate_compiler_multi_root_proof_inputs_v1,
};
pub use compiler_multi_root_target_lineage_v1::{
    ValidatedCompilerMultiRootTargetLineageV1, validate_compiler_multi_root_target_lineage_v1,
};
pub use compiler_native_source_proof_v1::*;
pub use compiler_nominal_abi_v3::{
    NominalAbiStorageV3, NominalSourceAbiAgreementV3, NominalSourceAbiErrorV3,
    check_nominal_source_abi_v3,
};
pub use compiler_proof_binding_v3::{
    CompilerProofInputValidationErrorV3, CompilerProofInputValidationErrorV4,
    ConditionalCompilerProofInputValidationErrorV1, ValidatedCompilerProofInputsV3,
    ValidatedCompilerProofInputsV4, ValidatedConditionalCompilerProofInputsV1,
    VerifiedSemanticU32InductionKirAnchorV1, validate_compiler_proof_inputs_v3,
    validate_compiler_proof_inputs_v4, validate_conditional_compiler_proof_inputs_v1,
};
pub use compiler_refined_forwarding_output_v1::{
    CheckedCompilerRefinedForwardingOutputV1, CompilerRefinedForwardingOutputErrorV1,
    CompilerRefinedForwardingOutputStorageV1, RecoveredCompilerNativeSemanticHandoffStorageV4,
    RecoveredCompilerNativeSemanticHandoffV4, RecoveredCompilerRefinedForwardingOutputV1,
    RecoveredCompilerRefinedForwardingStorageV1, RefinedForwardingOriginalSourceProofV1,
    check_compiler_refined_forwarding_output_v1, recover_compiler_native_semantic_handoff_token_v4,
    recover_compiler_native_semantic_handoff_v4, recover_compiler_refined_forwarding_carrier_v1,
    recover_compiler_refined_forwarding_output_v1,
};
pub use conditional_ranked_formulas_v1::{
    ProductionConditionalFormulaErrorV1, ProductionConditionalFormulaErrorV2,
    ProductionConditionalFormulaExecutionV1, ProductionConditionalFormulaExecutionV2,
    ProductionConditionalFormulaReportV1, ProductionConditionalFormulaReportV2,
    RetainedProductionConditionalFormulaV1, RetainedProductionConditionalFormulaV2,
    execute_and_retain_conditional_ranked_formula_policy_v2,
    execute_and_retain_conditional_ranked_formula_v1,
    execute_and_retain_conditional_ranked_formula_v2,
    import_and_retain_conditional_ranked_formula_policy_v2,
    import_and_retain_conditional_ranked_formula_v2, with_conditional_ranked_formula_execution_v1,
};
// Deprecated compatibility exports. Despite their Verus-oriented names, these
// authenticate and execute only the recorder; they do not show that Verus or a
// solver ran.
#[allow(deprecated)]
pub use authenticated_execution::{
    AuthenticatedExecutionProgramsV1, AuthenticatedVerusExecutionEvidenceV1,
    execute_authenticated_verus,
};
pub use authenticated_proof_binding::{
    AUTHENTICATED_PROOF_EXECUTABLE_BINDING_DOMAIN_V1,
    AUTHENTICATED_PROOF_EXECUTABLE_BINDING_VERSION_V1, AuthenticatedExecutionFreshnessV1,
    AuthenticatedPayloadIdentityV1, AuthenticatedProofExecutableBindingError,
    AuthenticatedProofExecutableBindingV1, AuthenticatedProofExecutablePolicyV1,
    AuthenticatedProofExecutionIdentityV1,
    PERSISTENT_AUTHENTICATED_PROOF_EXECUTABLE_BINDING_DOMAIN_V1,
    PersistentlyFreshProofExecutableBindingV1, PersistentlyFreshProofExecutableIdentityV1,
    bind_authenticated_proof_executable_persistent_v1, bind_authenticated_proof_executable_v1,
};
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use compiler_proof_broker_v1::{
    COMPILER_PROOF_BROKER_CHILD_FD_V1, COMPILER_PROOF_ENDPOINT_CHILD_FD_V1,
    CompilerProofBootstrapV1, CompilerProofBrokerV1, CompilerProofSessionCancellationV1,
    PendingCompilerProofDelegationV1, PendingCompilerProofServerV1,
    SpawnedCompilerProofDelegationV1, admit_inherited_compiler_proof_runtime_v1,
};
pub use compiler_target_lineage_v1::{
    CompilerTargetLineageValidationErrorV1, ValidatedCompilerTargetLineageV1,
    validate_compiler_target_lineage_v1, validate_conditional_compiler_target_lineage_v1,
};
pub use conditional_fill_program_v1::native_v1::{
    CheckedNativeConditionalFillProgramV1, NativeConditionalFillProgramErrorV1,
    NativeConditionalFillProgramStorageV1, check_native_conditional_fill_program_v1,
};
pub use conditional_fill_program_v1::{
    CheckedConditionalFillProgramV1, ConditionalFillProgramErrorV1,
    check_conditional_fill_program_v1,
};
pub use conditional_fill_refinement_v1::{
    ConditionalFillRefinementErrorV1, NativeConditionalFillRefinementErrorV1,
    NativeConditionalFillRefinementExecutionV1, NativeConditionalFillRefinementStorageV1,
    OwnedConditionalFillRefinementExecutionV1, ProductionConditionalFillRefinementExecutionV1,
    execute_conditional_fill_refinement_v1, execute_native_conditional_fill_refinement_v1,
    execute_owned_conditional_fill_refinement_v1,
};
pub use conditional_output_evidence_v1::{
    CanonicalConditionalOutputEvidenceV1, ConditionalOutputEvidenceErrorV1,
    ConditionalOutputLocationV1, InertConditionalOutputObligationV1,
};
pub use control_flow_binding::{
    AUTHENTICATED_CONTROL_FLOW_EXECUTABLE_BINDING_DOMAIN_V1,
    AuthenticatedControlFlowExecutableBindingV1, CONTROL_FLOW_BINDING_VERSION_V1,
    CONTROL_FLOW_FUNCTIONAL_SPECIFICATION_DOMAIN_V1, CONTROL_FLOW_REQUEST_BINDING_DOMAIN_V1,
    CONTROL_FLOW_SOURCE_BINDING_DOMAIN_V1, ControlFlowBindingErrorV1, ControlFlowClaimsV1,
    ControlFlowIntegerSwitchCaseClaimV1, ControlFlowIntegerSwitchClaimV1, ControlFlowLoopClaimV1,
    ControlFlowPayloadIdentityV1, ControlFlowProofRequestBindingV1, ControlFlowSourceBindingV1,
    MAX_BOUND_CONTROL_FLOW_LOOPS_V1, MAX_BOUND_CONTROL_FLOW_SWITCHES_V1,
    PERSISTENT_AUTHENTICATED_CONTROL_FLOW_EXECUTABLE_BINDING_DOMAIN_V1,
    PersistentlyFreshAuthenticatedControlFlowExecutableBindingV1,
    bind_authenticated_control_flow_executable_v1, bind_control_flow_proof_request_v1,
    bind_persistently_fresh_authenticated_control_flow_executable_v1,
    derive_control_flow_functional_specification_digest_v1, reconcile_control_flow_source_v1,
};
pub use executor::{
    ExecutionError, ExecutionErrorKind, ExecutionLimits, ExecutionPath, ExecutionStage,
    ExecutionSuccess, MAX_CAPTURE_BYTES, OutputStream, ProcessOutput, execute_recorder,
};
pub use functional_refinement_receipt_v2::{
    FunctionalRefinementVerusExecutionErrorKindV2, FunctionalRefinementVerusExecutionErrorV2,
    InertFunctionalRefinementReceiptSignatureV2, PreparedFunctionalRefinementReceiptV2,
    RetainedImportedFunctionalRefinementReceiptV2,
    execute_and_import_ranked_functional_refinement_locally_v2,
    execute_and_retain_ranked_functional_refinement_locally_v2,
    functional_refinement_verus_toolchain_identity_v2,
    prepare_ranked_functional_refinement_receipt_v2,
};
pub use functional_refinement_runtime_v1::{
    FunctionalRefinementRuntimeErrorV1, FunctionalRefinementVerusRuntimeIdentityV1,
    FunctionalRefinementVerusRuntimeLeaseV1, PROTECTED_FUNCTIONAL_REFINEMENT_RUNTIME_ROOT_V1,
};
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use functional_refinement_runtime_v1::{
    FunctionalRefinementRuntimeResourceErrorV1, FunctionalRefinementRuntimeStorageV1,
};
pub use generated_verus_proof_input_v3::{
    CanonicalGeneratedVerusProofInputV3, GeneratedVerusProofInputErrorV3,
    GeneratedVerusProofInputIdentityV3, MAX_GENERATED_VERUS_PROOF_SOURCE_BYTES_V3,
};
pub use mir_pliron_per_compilation_verus_v1::{
    MAX_PRODUCTION_AGGREGATE_EFFECT_FORMULA_OUTPUTS_V1,
    ProductionMirPlironPerCompilationVerusErrorV1,
    ProductionMirPlironPerCompilationVerusExecutionV1,
    ProductionMirPlironPerCompilationVerusReportV1, ProductionVerusVerifiedMirPlironKernelV1,
    execute_mir_pliron_semantic_contract_per_compilation_borrowed_v1,
    execute_mir_pliron_semantic_contract_per_compilation_v1,
};
pub use mir_pliron_verus_execution_evidence_v1::{
    CanonicalProductionMirPlironVerusExecutionEvidenceV1,
    PRODUCTION_MIR_PLIRON_VERUS_EXECUTION_EVIDENCE_BYTES_V1,
    ProductionMirPlironVerusExecutionClaimsV1, ProductionMirPlironVerusExecutionEvidenceErrorV1,
    ProductionMirPlironVerusExecutionEvidenceIdentityV1,
};
pub use mixed_native_correspondence_v60::{
    MixedNativeCorrespondenceErrorV60, check_mixed_native_correspondence_v60,
};
pub use mixed_native_correspondence_v89::{
    MixedNativeCorrespondenceErrorV89, check_mixed_native_correspondence_v89,
};
pub use mixed_target_selection_v53::*;
pub use mixed_target_selection_v89::*;
pub use model::{
    AxiomPolicy, Configuration, ConfigurationEntry, CorrelationId, Digest, ExecutionTools,
    MAX_CONFIGURATION_ENTRIES, MAX_PROPERTIES, MAX_TEXT_BYTES, MAX_TRUSTED_ITEMS,
    MeasuredToolIdentity, ModelError, ProofOutcome, ProofProperty, ProofRequestV1,
    ProofTargetIdentity, Text, TrustedItem, VerificationModelIdentity,
};
pub use monomorphization_dead_binding::{
    MONOMORPHIZATION_DEAD_BINDING_DOMAIN_V1, MONOMORPHIZATION_DEAD_BINDING_VERSION_V1,
    MonomorphizationDeadBindingErrorV1, MonomorphizationDeadClaimV1,
    MonomorphizationDeadIdentityBindingV1, reconcile_monomorphization_dead_evidence_v1,
};
pub use multi_kernel_proof::{
    KernelProofAdmissionIdentityV1, KernelProofAdmissionRequestV1,
    MULTI_KERNEL_PROOF_ADMISSION_DOMAIN_V1, MULTI_KERNEL_PROOF_ADMISSION_VERSION_V1,
    MultiKernelProofAdmissionErrorV1, MultiKernelProofAdmissionV1,
    PERSISTENT_MULTI_KERNEL_PROOF_ADMISSION_DOMAIN_V1,
    PERSISTENT_MULTI_KERNEL_PROOF_ADMISSION_VERSION_V1,
    PersistentlyFreshKernelProofAdmissionIdentityV1,
    PersistentlyFreshKernelProofAdmissionRequestV1,
    PersistentlyFreshMultiKernelProofAdmissionErrorV1,
    PersistentlyFreshMultiKernelProofAdmissionV1,
};
pub use native_copy_classification_v1::{
    NativeCopyProgramClassificationV1, classify_native_copy_program_v1,
};
pub use persistent_freshness::{
    MAX_PERSISTENT_FRESHNESS_ENTRIES_V1, MAX_PERSISTENT_FRESHNESS_STATE_BYTES_V1,
    PERSISTENT_FRESHNESS_INTENT_MAGIC_V1, PERSISTENT_FRESHNESS_STATE_MAGIC_V1,
    PERSISTENT_FRESHNESS_VERSION_V1, PersistentFreshnessIdentityFieldV1,
    PersistentFreshnessIdentityV1, PersistentFreshnessIntentInspectionV1,
    PersistentFreshnessLedgerErrorV1, PersistentFreshnessLedgerFileV1,
    PersistentFreshnessLedgerOperationV1, PersistentFreshnessReceiptV1,
    PersistentFreshnessRecordErrorV1, PersistentFreshnessRecoveryV1,
    PersistentFreshnessStateInspectionV1, PersistentProofFreshnessLedgerV1,
    PersistentProofFreshnessTransactionV1, inspect_persistent_freshness_intent_v1,
    inspect_persistent_freshness_state_v1,
};
pub use plan::{
    CommandSpec, InvocationPaths, InvocationPlan, MAX_PATH_BYTES, MAX_TIMEOUT_SECONDS, PlanError,
    VerifierPolicy, build_invocation_plan,
};
pub use production_kir_to_llvm_replay_v1::{
    CompilerKirToLlvmReplayValidationErrorV1, ValidatedCompilerKirToLlvmReplayV1,
    validate_compiler_kir_to_llvm_replay_v1,
};
pub use proof_capsule::{
    MAX_PROCESS_LOCAL_PROOF_CAPSULE_RECORDS_V1, MAX_PROOF_CAPSULE_BYTES_V1,
    MAX_PROOF_CAPSULE_DEPENDENCIES_V1, MAX_PROOF_CAPSULE_FEATURES_V1,
    MAX_PROOF_CAPSULE_SEALED_RESULT_BYTES_V1, PROOF_CAPSULE_MAGIC_V1, PROOF_CAPSULE_VERSION_V1,
    ProcessLocalProofCapsuleDuplicateDetectorV1, ProofCapsuleBuildErrorV1,
    ProofCapsuleContextErrorV1, ProofCapsuleDecodeErrorV1, ProofCapsuleDependencyV1,
    ProofCapsuleExecutionV1, ProofCapsuleExpectationV1, ProofCapsuleFreshnessExpectationV1,
    ProofCapsuleFreshnessIdentityV1, ProofCapsuleIdentityFieldV1, ProofCapsulePayloadIdentityV1,
    ProofCapsulePolicyV1, ProofCapsuleResultV1, ProofCapsuleTargetV1, ProofCapsuleV1,
};
pub use result::{
    MAX_RESULT_BYTES, ProofResultV1, RecorderTermination, ResultError, parse_recorder_result,
};
pub use static_view_proof::{
    STATIC_VIEW_PROOF_EVIDENCE_DOMAIN_V1, STATIC_VIEW_PROOF_OBLIGATION_DOMAIN_V1,
    STATIC_VIEW_PROOF_REQUIRED_PROPERTIES_V1, STATIC_VIEW_PROOF_VERSION_V1,
    StaticViewLifetimeEpochClaimV1, StaticViewProofErrorV1, StaticViewProofEvidenceV1,
    StaticViewProofObligationV1, bind_static_view_proof_evidence_v1,
    derive_static_view_functional_specification_digest_v1,
};

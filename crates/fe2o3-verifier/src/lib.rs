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

mod artifact_record;
mod authenticated_execution;
mod authenticated_proof_binding;
mod authenticated_verus_execution_v2;
mod compiler_proof_binding_v3;
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
mod production_kir_to_llvm_replay_v1;
mod proof_capsule;
mod result;
mod retained_functional_refinement_runtime_v1;
mod static_view_proof;

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
pub use compiler_proof_binding_v3::{
    CompilerProofInputValidationErrorV3, CompilerProofInputValidationErrorV4,
    ValidatedCompilerProofInputsV3, ValidatedCompilerProofInputsV4,
    VerifiedSemanticU32InductionKirAnchorV1, validate_compiler_proof_inputs_v3,
    validate_compiler_proof_inputs_v4,
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
    PreparedFunctionalRefinementReceiptV2,
    execute_and_import_ranked_functional_refinement_locally_v2,
    functional_refinement_verus_toolchain_identity_v2,
    prepare_ranked_functional_refinement_receipt_v2,
};
pub use functional_refinement_runtime_v1::{
    FunctionalRefinementRuntimeErrorV1, FunctionalRefinementVerusRuntimeIdentityV1,
    FunctionalRefinementVerusRuntimeLeaseV1,
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

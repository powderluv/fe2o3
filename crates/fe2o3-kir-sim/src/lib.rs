#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod budgeted_v12;
mod budgeted_v12_observation;
mod budgeted_v18;
pub use budgeted_v12::{SimulationViewAdmissionErrorV12, SimulationViewStorageV12};
pub use budgeted_v12_observation::{
    Bf16CallCpuObservationOptionsV1, V12CpuObservationErrorV1, V12CpuObservationInputV1,
    V12CpuObservationOptionsV1, V12CpuObservationProfileErrorV1,
};
pub use budgeted_v18::{SimulationViewAdmissionErrorV18, SimulationViewStorageV18};
mod matrix_bf16_exact_v1;
mod matrix_fp4_exact_v1;
mod matrix_fp4_fp8_exact_v1;
mod matrix_fp8_exact_v1;
pub use matrix_bf16_exact_v1::{MatrixBf16NumericsV1, MatrixInputRoleV1};
mod capability;
mod complete_body_v19;
mod debug;
mod debug_allocation_lifecycle_v1;
mod debug_identity_state;
mod debug_observation;
mod debug_runtime_frames;
mod debug_runtime_origin;
mod execute;
mod explore;
mod f32_surface;
mod model;
mod ordered_program_v17;
mod ordered_region_v16;
mod physical_entry_v20;
mod physical_lds_exchange_v22;
pub use physical_lds_exchange_v22::{
    PHYSICAL_LDS_EXCHANGE_ADMISSION_WORK_V22, PhysicalLdsExchangeSimulationAdmissionErrorV22,
    PhysicalLdsExchangeSimulationStorageV22,
};
mod physical_global_copy_v21;
pub use physical_global_copy_v21::{
    PHYSICAL_GLOBAL_COPY_ADMISSION_WORK_V21, PhysicalGlobalCopySimulationAdmissionErrorV21,
    PhysicalGlobalCopySimulationStorageV21,
};
mod generic_exposure_v18;
mod preflight;
mod reduce;
mod resident;
mod schedule;
mod soft_float;
mod storage_scalar_v18;

pub use capability::{
    POINTER_CAPABILITY_ROWS_V1, SCALAR_CAPABILITY_ROWS_V1,
    SEMANTIC_CAPABILITY_MATRIX_JSON_BYTES_V1, SEMANTIC_CAPABILITY_MATRIX_SCHEMA_V1,
    SimulationCapabilityDispositionV1, SimulationCapabilityMatrixV1, SimulationCapabilityProfileV1,
    SimulationKirWireVersionV1, SimulationOperationCapabilityRowV1, SimulationOperationSurfaceV1,
    SimulationPointerCapabilityRowV1, SimulationScalarCapabilityRowV1,
    SimulationScalarOperationFamilyV1, SimulationSemanticOwnerV1,
    SimulationUnsupportedReasonCodeV1, TOP_LEVEL_CAPABILITY_ROWS_V1, semantic_capability_matrix_v1,
};
pub use debug::{
    MAX_DEBUG_ALLOCATIONS_PER_CHECKPOINT_V1, MAX_DEBUG_FRAMES_PER_CHECKPOINT_V1,
    MAX_DEBUG_MEMORY_BYTES_PER_CHECKPOINT_V1, MAX_DEBUG_VALUES_PER_CHECKPOINT_V1,
    NoopSimulationDebugSinkV1, SimulationDebugAllocationV1, SimulationDebugBarrierActionV1,
    SimulationDebugBindingV1, SimulationDebugCaptureLimitFieldV1,
    SimulationDebugCaptureLimitsErrorV1, SimulationDebugCaptureLimitsV1,
    SimulationDebugCheckpointPhaseV1, SimulationDebugCollectionV1, SimulationDebugFrameV1,
    SimulationDebugMemoryAccessV1, SimulationDebugRecordKindV1, SimulationDebugRecordV1,
    SimulationDebugScheduleV1, SimulationDebugSinkControlV1, SimulationDebugSinkV1,
    SimulationDebugSiteV1, SimulationDebugUnavailableReasonV1, SimulationDebugValueV1,
};
pub use debug_allocation_lifecycle_v1::{
    SimulationAllocationDescriptorV1, SimulationAllocationObservationUnavailableV1,
    SimulationAllocationScopeV1, SimulationAllocationStorageIdentityV1,
    SimulationAllocationTransitionKindV1, SimulationAllocationTransitionV1,
    SimulationAllocationWatermarkV1,
};
pub use debug_observation::SimulationDebugObservationContextV1;
pub use debug_runtime_frames::{
    SimulationDebugCheckpointFramesV1, SimulationDebugFrameOperationV1,
    SimulationDebugFrameOriginUnavailableV1, SimulationDebugFrameOriginV1,
    SimulationDebugFrameOriginsV1, SimulationDebugFrameParentV1,
};
pub use debug_runtime_origin::{
    SimulationDebugOperationOriginV1, SimulationDebugOriginContextV1,
    SimulationDebugOriginUnavailableV1,
};
pub use execute::{
    DivergentWaveV1, DivergentWorkgroupBarrierV1, DivergentWorkgroupBarrierV2, IncompleteWaveV1,
    MAX_ALLOCATION_REUSE_CACHED_PAYLOAD_BYTES_V1, MismatchedWaveV1, MismatchedWorkgroupBarrierV1,
    NoopSimulationEventSinkV1, ObservationExecutionOptionsV1, SimulationAbiViewV1,
    SimulationAllocationReuseErrorV1, SimulationAllocationReuseV1, SimulationConflictAssessmentV1,
    SimulationDataRaceV1, SimulationErrorV1, SimulationEventKindV1, SimulationEventSinkControlV1,
    SimulationEventSinkErrorV1, SimulationEventSinkV1, SimulationEventSiteV1, SimulationEventV1,
    SimulationExecutionErrorKindV1, SimulationExecutionErrorV1, SimulationExecutionOutcomeV1,
    SimulationExecutionV1, SimulationHappensBeforeReasonV1, SimulationMemoryConflictV1,
    SimulationObservationFailureV1, SimulationOrderedMemoryConflictV1, SimulationOutOfBoundsV2,
    SimulationRaceAssessmentV1, WorkgroupBarrierMismatchV1, WorkgroupParticipantV1,
};
pub use explore::{
    MAX_EXPLORATION_RETAINED_DECISIONS_V1, MAX_EXPLORATION_SCHEDULES_V1,
    SimulationExplorationFailureV1, SimulationExplorationRequestErrorV1,
    SimulationExplorationRequestV1, SimulationExplorationV1, SimulationExplorationWitnessV1,
};
pub use f32_surface::{F32_SCALAR_OPERATION_ROSTER_V1, F32ScalarOperationV1};
pub use model::{
    AdmittedSimulationModuleV1, BufferArgumentErrorV1, BufferArgumentV1, BufferBackingIdV1,
    BufferViewArgumentV1, DynamicWorkgroupMemoryRequestV1, EventPolicyV1, GridShapeV1,
    IndexWidthV1, ScalarBitsErrorV1, ScalarBitsV1, SharedBufferV1, SimulationAdmissionErrorV1,
    SimulationArgumentV1, SimulationInvocationV1, SimulationKernelIrIdentityV1,
    SimulationLimitsErrorV1, SimulationLimitsV1, SimulationRequestV1, SimulationSiteV1,
    SimulationTargetV1, WorkgroupShapeV1,
};
pub use preflight::{
    DynamicWorkgroupMemorySiteV1, DynamicWorkgroupMemoryUnavailableV1,
    MAX_REPORTED_UNSUPPORTED_FINDINGS_V1, MAX_REPORTED_UNSUPPORTED_IDENTIFIER_BYTES_V1,
    SimulationPlanV1, SimulationPreflightErrorV1, UnsupportedFeatureV1,
    UnsupportedSimulationReportV1, UnsupportedSimulationSiteV1,
};
pub use reduce::{
    MAX_FAILURE_REDUCTION_ATTEMPTS_V1, MAX_FAILURE_REDUCTION_RETAINED_DECISIONS_V1,
    MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1, SimulationFailureFingerprintV1,
    SimulationFailureReductionCodecErrorV1, SimulationFailureReductionCoverageV1,
    SimulationFailureReductionErrorV1, SimulationFailureReductionLimitsV1,
    SimulationFailureReductionReportV1, SimulationFailureReductionRequestErrorV1,
    SimulationFailureScheduleV1,
};
pub use schedule::{
    MAX_PERSISTED_SCHEDULE_BYTES_V1, MAX_SCHEDULE_DECISIONS_V1,
    PersistedSimulationScheduleArtifactV1, PersistedSimulationScheduleBindingV1,
    PersistedSimulationScheduleCodecErrorV1, PersistedSimulationScheduleDocumentV1,
    SimulationScheduleCoverageV1, SimulationScheduleDecisionV1, SimulationScheduleIdentityV1,
    SimulationScheduleRecordV1, SimulationScheduleReplayErrorV1, SimulationScheduleRequestV1,
};

// Typed budget-owned physical CPU observations; no serialized schema extension.
pub use execute::{
    MAX_PHYSICAL_ENTRY_DEBUG_RECORDS_V20, PhysicalDebugSymbolicV1, PhysicalEntryDebugBindingRefV20,
    PhysicalEntryDebugCaptureErrorV20, PhysicalEntryDebugCaptureStopV20,
    PhysicalEntryDebugCaptureV20, PhysicalEntryDebugOptionsV20, PhysicalEntryDebugOutcomeV20,
    PhysicalEntryDebugRecordRefV20, PhysicalEntryDebugSymbolicKindV20,
    PhysicalEntryDebugSymbolicV20, PhysicalEntryDebugUsageV20,
};

#[cfg(test)]
use fe2o3_kernel_ir as physical_global_copy_fixture_ir;
#[cfg(test)]
#[path = "../../fe2o3-kernel-ir/tests/fixtures/physical_global_copy_v21.rs"]
mod physical_global_copy_test_fixture;

mod physical_entry_budgeted_v20;
pub use physical_entry_budgeted_v20::{
    PHYSICAL_ENTRY_ADMISSION_WORK_V20, PhysicalEntrySimulationAdmissionErrorV20,
    PhysicalEntrySimulationStorageV20,
};

// Separate exact-owner global-copy CPU observations. Legacy V21 debug routes stay closed.
pub use execute::{
    MAX_PHYSICAL_GLOBAL_COPY_DEBUG_RECORDS_V21, PhysicalGlobalCopyDebugBindingRefV21,
    PhysicalGlobalCopyDebugCaptureErrorV21, PhysicalGlobalCopyDebugCaptureStopV21,
    PhysicalGlobalCopyDebugCaptureV21, PhysicalGlobalCopyDebugOptionsV21,
    PhysicalGlobalCopyDebugOutcomeV21, PhysicalGlobalCopyDebugRecordRefV21,
    PhysicalGlobalCopyDebugSymbolicKindV21, PhysicalGlobalCopyDebugSymbolicV21,
    PhysicalGlobalCopyDebugUsageV21,
};

#[cfg(test)]
use fe2o3_kernel_ir as physical_lds_exchange_fixture_ir;
#[cfg(test)]
#[path = "../../fe2o3-kernel-ir/tests/fixtures/physical_lds_exchange_v22.rs"]
mod physical_lds_exchange_test_fixture;

pub use execute::{
    MAX_PHYSICAL_LDS_EXCHANGE_DEBUG_RECORDS_V22, PhysicalLdsExchangeDebugBindingRefV22,
    PhysicalLdsExchangeDebugCaptureErrorV22, PhysicalLdsExchangeDebugCaptureStopV22,
    PhysicalLdsExchangeDebugCaptureV22, PhysicalLdsExchangeDebugOptionsV22,
    PhysicalLdsExchangeDebugOutcomeV22, PhysicalLdsExchangeDebugRecordRefV22,
    PhysicalLdsExchangeDebugSymbolicKindV22, PhysicalLdsExchangeDebugSymbolicV22,
    PhysicalLdsExchangeDebugUsageV22,
};

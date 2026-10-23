#![allow(
    clippy::result_large_err,
    reason = "exact simulator failures retain typed invocation, site, and bounded diagnostic data"
)]

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::mem::size_of;

use fe2o3_kernel_ir::{
    AccessMode, AddressSpace, AmdGpuDiagnosticOperation, Atomic, AtomicKind, Axis, BasicBlock,
    BinaryOp, BlockId, CastKind, CheckedBinaryOperator, ComparePredicate, Constant, Fence,
    Function, FunctionId, FunctionRole, Gfx950LdsTransposeFormatV1,
    Gfx950LdsTransposeOperationKindV1, Gfx950LdsTransposeOperationV1, IndexKind, IntrinsicKind,
    MatrixElement, MatrixLdsProfile, MatrixOperation, MatrixOperationKind, MemoryAccess,
    MemoryIntrinsicOperation, MemoryOrdering, Module, Operation, OperationKind,
    PointerDistanceKind, PointerDistanceUnit, ScalarType, SynchronizationScope, Terminator, Type,
    UnaryOp, ValueDef, ValueId, WaveF32ReductionKindV1, WaveOperation, WaveOperationKind,
    WaveWidth, WorkgroupBarrier, WorkgroupMemory, WorkgroupMemoryExtent,
    validate_gfx942_inline_assembly_v1,
};

use crate::debug_identity_state::{FrameIdentityState, InvocationIdentityState, OperationIdentity};
use crate::model::mask;
use crate::preflight::{supported_cast, supports_binary, supports_compare, supports_unary};
use crate::resident::{
    ResidentLedger, hash_map_capacity_bytes, partitioned_bool_vec_storage_bytes,
    partitioned_geometric_vec_bytes, reserved_bool_vec_bytes, reserved_hash_map_bytes,
    reserved_vec_bytes,
};
use crate::schedule::{
    ExecutionScheduleRequestV1, PreparedScheduleV1, ReductionScheduleSourceV1,
    SchedulePrepareErrorV1,
};
use crate::soft_float::{SoftFloatErrorV1, SoftFloatOperationV1};
use crate::{
    AdmittedSimulationModuleV1, BufferArgumentV1, BufferBackingIdV1,
    DynamicWorkgroupMemoryRequestV1, EventPolicyV1, IndexWidthV1, NoopSimulationDebugSinkV1,
    ScalarBitsV1, SharedBufferV1, SimulationArgumentV1, SimulationDebugAllocationV1,
    SimulationDebugBarrierActionV1, SimulationDebugBindingV1, SimulationDebugCaptureLimitsV1,
    SimulationDebugCheckpointPhaseV1, SimulationDebugCollectionV1, SimulationDebugFrameV1,
    SimulationDebugMemoryAccessV1, SimulationDebugRecordKindV1, SimulationDebugRecordV1,
    SimulationDebugScheduleV1, SimulationDebugSinkControlV1, SimulationDebugSinkV1,
    SimulationDebugSiteV1, SimulationDebugUnavailableReasonV1, SimulationDebugValueV1,
    SimulationInvocationV1, SimulationKernelIrIdentityV1, SimulationLimitsV1, SimulationPlanV1,
    SimulationPreflightErrorV1, SimulationRequestV1, SimulationScheduleCoverageV1,
    SimulationScheduleIdentityV1, SimulationScheduleRecordV1, SimulationScheduleReplayErrorV1,
    SimulationScheduleRequestV1, SimulationSiteV1, SimulationTargetV1,
};

#[path = "allocation_reuse_v1.rs"]
mod allocation_reuse_v1;
use allocation_reuse_v1::{AllocationCreationV1, AllocationReusePool};
pub use allocation_reuse_v1::{
    MAX_ALLOCATION_REUSE_CACHED_PAYLOAD_BYTES_V1, SimulationAllocationReuseErrorV1,
    SimulationAllocationReuseV1,
};
#[path = "execute_observed_storage.rs"]
mod observed_storage;
pub use observed_storage::ObservationExecutionOptionsV1;

#[path = "execute_access_invocation_v1.rs"]
mod access_invocation_v1;
use access_invocation_v1::AccessInvocationV1;

#[path = "execute_atomic_scope_v1.rs"]
mod atomic_scope_v1;
use atomic_scope_v1::{AtomicAccess, AtomicHistory, Relation as AtomicRelation};

#[path = "execute_alloca_v1.rs"]
mod alloca_v1;
#[path = "execute_execution_lifecycle_v18.rs"]
pub(crate) mod execution_lifecycle_v18;
#[path = "execute_generic_exposure_v18.rs"]
mod generic_exposure_v18;
#[path = "execute_incoming_capacity_v1.rs"]
mod incoming_capacity_v1;
#[path = "execute_values_v1.rs"]
mod values_v1;
pub(crate) use incoming_capacity_v1::{IncomingValueCapacityV1, incoming_value_capacity};
pub(crate) use values_v1::{FrameValuePlan, frame_value_plan, uses_legacy_frame_values};
use values_v1::{RuntimeValues, ValueLayout};

#[path = "execute_matrix_bf16_exact_v1.rs"]
mod matrix_bf16_exact_v1;
#[path = "execute_matrix_fp4_exact_v1.rs"]
mod matrix_fp4_exact_v1;
#[path = "execute_matrix_fp4_fp8_exact_v1.rs"]
mod matrix_fp4_fp8_exact_v1;
#[path = "execute_matrix_fp8_exact_v1.rs"]
mod matrix_fp8_exact_v1;
#[path = "execute_storage_scalar_v18.rs"]
mod storage_scalar_v18;
pub(crate) fn matrix_bf16_exact_resident_bytes() -> Option<usize> {
    matrix_bf16_exact_v1::resident_bytes()
}
pub(crate) fn matrix_fp4_exact_resident_bytes() -> Option<usize> {
    matrix_fp4_exact_v1::resident_bytes()
}

pub(crate) fn matrix_fp8_exact_resident_bytes() -> Option<usize> {
    matrix_fp8_exact_v1::resident_bytes()
}
pub(crate) fn matrix_fp4_fp8_exact_resident_bytes() -> Option<usize> {
    matrix_fp4_fp8_exact_v1::resident_bytes()
}

// Pure size guard for the separate V12 metered observation facade. The existing
// calculators may allocate one capacity probe at a time; this does not execute,
// construct an Engine, change a limit or grant observation authority.
pub(crate) fn v12_observation_probe_cells_fit() -> bool {
    [
        size_of::<Engine<'static, NoopSimulationEventSinkV1>>(),
        size_of::<RaceTracker>(),
        size_of::<SimulationSupplementalV1>(),
        size_of::<Allocation>(),
        size_of::<AccessFrontier>(),
        size_of::<RuntimeValue>(),
        size_of::<RuntimeFrame<'static>>(),
        size_of::<InvocationMachine<'static>>(),
        size_of::<WorkgroupParticipantV1>(),
        size_of::<FrameAllocation>(),
        size_of::<WorkgroupAllocation>(),
        size_of::<(ValueId, RuntimeValue)>(),
        size_of::<(ValueId, Type)>(),
        size_of::<(FunctionId, usize)>(),
        size_of::<(BlockId, usize)>(),
        size_of::<(u64, usize)>(),
        size_of::<(u64, Allocation)>(),
        size_of::<((u64, usize), AccessFrontier)>(),
        size_of::<(BufferBackingIdV1, u64)>(),
        size_of::<Operation>(),
        size_of::<SimulationArgumentV1>(),
        size_of::<SharedBufferV1>(),
        size_of::<SimulationExecutionV1>(),
        size_of::<SimulationPlanV1>(),
        size_of::<SimulationExecutionErrorV1>(),
        size_of::<SimulationErrorV1>(),
    ]
    .into_iter()
    .all(|bytes| bytes <= 4096)
        && size_of::<RuntimeValue>() <= 160
}
#[path = "execute_complete_body_v19.rs"]
mod complete_body_v19;
#[path = "execute_debug_physical_symbolic_v1.rs"]
mod debug_physical_symbolic_v1;
pub use debug_physical_symbolic_v1::PhysicalDebugSymbolicV1;
#[path = "execute_debug_physical_value_v20.rs"]
mod debug_physical_value_v20;
pub use debug_physical_value_v20::*;
#[path = "execute_debug_physical_capture_v20.rs"]
mod debug_physical_capture_v20;
pub use debug_physical_capture_v20::*;
#[path = "execute_debug_global_copy_value_v21.rs"]
mod debug_global_copy_value_v21;
pub use debug_global_copy_value_v21::*;
#[path = "execute_debug_global_copy_capture_v21.rs"]
mod debug_global_copy_capture_v21;
pub use debug_global_copy_capture_v21::*;
#[path = "execute_debug_lds_exchange_value_v22.rs"]
mod debug_lds_exchange_value_v22;
pub use debug_lds_exchange_value_v22::*;
#[path = "execute_debug_lds_exchange_capture_v22.rs"]
mod debug_lds_exchange_capture_v22;
pub use debug_lds_exchange_capture_v22::*;
#[path = "execute_debug_frames.rs"]
mod debug_frames;
#[path = "execute_debug_identity.rs"]
mod debug_identity;
#[path = "execute_observed_debug.rs"]
mod observed_debug;
#[path = "execute_ordered_program_v17.rs"]
mod ordered_program_v17;
#[path = "execute_ordered_region_v16.rs"]
mod ordered_region_v16;
#[path = "execute_physical_entry_collective_v20.rs"]
mod physical_entry_collective_v20;
#[path = "execute_physical_entry_state_v20.rs"]
mod physical_entry_state_v20;
#[path = "execute_physical_entry_v20.rs"]
mod physical_entry_v20;
#[cfg(test)]
#[path = "execute_physical_entry_v20_tests.rs"]
mod physical_entry_v20_tests;
#[path = "execute_physical_global_copy_pending_v21.rs"]
mod physical_global_copy_pending_v21;
#[path = "execute_physical_lds_context_v22.rs"]
mod physical_lds_context_v22;
#[path = "execute_physical_lds_exchange_v22.rs"]
mod physical_lds_exchange_v22;
pub(crate) fn physical_lds_runtime_resident_bytes() -> Option<usize> {
    physical_lds_context_v22::resident_bytes()
}
#[path = "execute_physical_global_copy_v21.rs"]
mod physical_global_copy_v21;
#[path = "execute_pointer_view_operation_v1.rs"]
mod pointer_view_operation_v1;
#[path = "execute_scalar_memory_operation_v1.rs"]
mod scalar_memory_operation_v1;
#[path = "execute_value_operation_v1.rs"]
mod value_operation_v1;

/// Ephemeral execution event kind. This is an in-process adapter, not a durable trace schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationEventKindV1 {
    /// One logical invocation has started. It may later end as failed.
    InvocationBegin,
    /// One logical invocation has ended after either success or dynamic failure.
    InvocationEnd {
        outcome: SimulationExecutionOutcomeV1,
    },
    /// Control entered the named block after its arguments were bound.
    BlockEnter,
    /// An operation is about to be evaluated.
    OperationBegin,
    /// Operation evaluation has ended; this closes the lifecycle, not a success claim.
    OperationEnd {
        outcome: SimulationExecutionOutcomeV1,
    },
    /// A terminator is about to be evaluated after consuming one step.
    Terminator,
    /// The terminator selected this target after resolving its outgoing values.
    Branch { target: BlockId },
    /// A scalar load and initialization check completed successfully.
    MemoryRead {
        allocation: u64,
        offset: usize,
        bytes: usize,
    },
    /// A store is fully prepared; successful retention guarantees its infallible commit follows.
    MemoryWrite {
        allocation: u64,
        offset: usize,
        bytes: usize,
    },
    /// One indivisible integer atomic observation under the selected CPU schedule.
    MemoryAtomic {
        allocation: u64,
        offset: usize,
        bytes: usize,
        kind: AtomicKind,
        previous: Option<ScalarBitsV1>,
        committed: Option<ScalarBitsV1>,
        compare_exchange_success: Option<bool>,
        scope: SynchronizationScope,
        ordering: MemoryOrdering,
        failure_ordering: Option<MemoryOrdering>,
    },
    /// A scoped memory-order point. It does not synchronize invocation execution.
    MemoryFence {
        memory_scope: SynchronizationScope,
        ordering: MemoryOrdering,
        /// Bits 0 through 4 represent private, workgroup, global, constant,
        /// and generic address spaces, respectively.
        address_space_mask: u8,
    },
    /// An allocation that existed before the first invocation became observable.
    AllocationPreexisting {
        allocation: u64,
        address_space: AddressSpace,
        bytes: usize,
    },
    /// An allocation was created while evaluating the operation at this event's site.
    AllocationCreated {
        allocation: u64,
        address_space: AddressSpace,
        bytes: usize,
    },
    /// An allocation's semantic lifetime ended at this event's site.
    AllocationReleased { allocation: u64 },
    /// One live in-grid invocation arrived at a convergent workgroup barrier.
    WorkgroupBarrierArrive { phase: u64 },
    /// Every live in-grid invocation arrived and the workgroup phase was released.
    WorkgroupBarrierRelease { phase: u64, participants: u32 },
    /// Call arguments were resolved and control is about to enter the callee.
    Call {
        /// Canonical module-function ordinal of the callee.
        callee_ordinal: usize,
    },
    /// Return values were resolved and the function's private allocations were released.
    Return,
}

/// Whether an observed simulator lifecycle ended normally or through a dynamic failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationExecutionOutcomeV1 {
    Completed,
    Failed,
}

/// Nonfatal control returned by an ephemeral event sink.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationEventSinkControlV1 {
    /// The event was retained and delivery should continue.
    Continue,
    /// The event was retained; permanently stop callbacks and event accounting for this run.
    Stop,
    /// The event was not retained; permanently stop callbacks and event accounting for this run.
    DropAndStop,
}

/// One bounded ephemeral simulator event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationEventV1 {
    pub invocation: SimulationInvocationV1,
    pub site: SimulationEventSiteV1,
    pub kind: SimulationEventKindV1,
}

/// Allocation-free site carried by an ephemeral event callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationEventSiteV1 {
    /// Canonical module-function ordinal. The admitted module resolves it to an ID.
    pub function_ordinal: usize,
    pub block: BlockId,
    pub operation: Option<u32>,
}

/// Bounded result of cross-invocation global-memory conflict assessment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationConflictAssessmentV1 {
    NoConflictsObserved,
    ConflictsObserved {
        conflicting_bytes: u64,
        first: SimulationMemoryConflictV1,
    },
    Incomplete {
        conflicting_bytes: u64,
        first: Option<SimulationMemoryConflictV1>,
        /// The byte-access table reached its caller-supplied record bound.
        access_record_limit_reached: bool,
        /// A later access could depend on a representative evicted from the
        /// bounded per-byte read/write frontier.
        access_frontier_incomplete: bool,
        record_limit: usize,
    },
}

/// First observed byte-level conflicting access under the deterministic schedule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationMemoryConflictV1 {
    pub allocation: u64,
    pub offset: usize,
    pub earlier: SimulationInvocationV1,
    pub later: SimulationInvocationV1,
    pub earlier_site: SimulationSiteV1,
    pub later_site: SimulationSiteV1,
}

/// Bounded data-race assessment under one deterministic CPU schedule.
///
/// This is execution evidence, not a proof that another admitted interleaving is race-free.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationRaceAssessmentV1 {
    NoRacesObserved {
        first_ordered_conflict: Option<SimulationOrderedMemoryConflictV1>,
    },
    RacesObserved {
        racing_bytes: u64,
        first: SimulationDataRaceV1,
        first_ordered_conflict: Option<SimulationOrderedMemoryConflictV1>,
    },
    Incomplete {
        racing_bytes: u64,
        first: Option<SimulationDataRaceV1>,
        first_ordered_conflict: Option<SimulationOrderedMemoryConflictV1>,
        /// The byte-access table reached its caller-supplied record bound.
        access_record_limit_reached: bool,
        /// A later access could depend on a representative evicted from the
        /// bounded per-byte read/write frontier.
        access_frontier_incomplete: bool,
        /// Atomic scope/range serialization is unproved, or release/acquire atomic
        /// or fence HB may order an observed ordinary conflict.
        atomic_or_fence_happens_before_unmodeled: bool,
        record_limit: usize,
    },
}

/// First observed byte-level conflicting access not ordered by admitted synchronization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationDataRaceV1 {
    pub conflict: SimulationMemoryConflictV1,
    pub earlier_atomic: bool,
    pub later_atomic: bool,
}

/// Why one observed conflicting access pair is not a data race.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationHappensBeforeReasonV1 {
    AtomicSerialization,
    GlobalWorkgroupBarrier,
}

/// First conflicting pair that has an exact ordering in the simulator contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationOrderedMemoryConflictV1 {
    pub conflict: SimulationMemoryConflictV1,
    pub reason: SimulationHappensBeforeReasonV1,
}

/// Adapter for forwarding ephemeral simulator events to debugger or test infrastructure.
pub trait SimulationEventSinkV1 {
    /// Records one event atomically. An error means the event was not retained.
    fn record(&mut self, event: &SimulationEventV1) -> Result<(), SimulationEventSinkErrorV1>;

    /// Records one event and optionally stops future observation without failing execution.
    ///
    /// Sinks that bound their own storage should return `Stop` with the last event they retain,
    /// or `DropAndStop` when the current event did not fit and was not retained.
    /// The default preserves the original always-continue contract.
    fn record_controlled(
        &mut self,
        event: &SimulationEventV1,
    ) -> Result<SimulationEventSinkControlV1, SimulationEventSinkErrorV1> {
        self.record(event)?;
        Ok(SimulationEventSinkControlV1::Continue)
    }
}

/// A fallible debugger/profiler event boundary failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationEventSinkErrorV1 {
    pub detail: String,
}

impl fmt::Display for SimulationEventSinkErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl Error for SimulationEventSinkErrorV1 {}

/// Event sink that discards every event.
#[derive(Default)]
pub struct NoopSimulationEventSinkV1;

impl SimulationEventSinkV1 for NoopSimulationEventSinkV1 {
    fn record(&mut self, _event: &SimulationEventV1) -> Result<(), SimulationEventSinkErrorV1> {
        Ok(())
    }
}

/// Completed deterministic execution and copied-back arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationExecutionV1 {
    identity: SimulationKernelIrIdentityV1,
    matrix_bf16_numerics: crate::MatrixBf16NumericsV1,
    dynamic_workgroup_memory: Option<DynamicWorkgroupMemoryRequestV1>,
    arguments: Vec<SimulationArgumentV1>,
    shared_buffers: Vec<SharedBufferV1>,
    invocations_executed: u64,
    workgroups_visited: u64,
    scheduled_slots_visited: u64,
    steps_executed: u64,
    events_emitted: u64,
    schedule: SimulationScheduleIdentityV1,
    schedule_transcript_identity: [u8; 32],
    schedule_coverage: SimulationScheduleCoverageV1,
    supplemental: Vec<SimulationSupplementalV1>,
    conflict_assessment: SimulationConflictAssessmentV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SimulationSupplementalV1 {
    schedule_records: Vec<SimulationScheduleRecordV1>,
    race_assessment: SimulationRaceAssessmentV1,
}

impl SimulationExecutionV1 {
    /// Actual interpreter-selected numerical semantics, not hardware authority.
    pub const fn matrix_bf16_numerics(&self) -> crate::MatrixBf16NumericsV1 {
        self.matrix_bf16_numerics
    }
    /// Returns the exact canonical KIR identity that was simulated.
    pub const fn identity(&self) -> &SimulationKernelIrIdentityV1 {
        &self.identity
    }

    /// Returns the explicit runtime-sized LDS contract bound to this execution.
    pub const fn dynamic_workgroup_memory(&self) -> Option<DynamicWorkgroupMemoryRequestV1> {
        self.dynamic_workgroup_memory
    }

    /// Returns copied-back scalar and buffer arguments in entry-ABI order.
    pub fn arguments(&self) -> &[SimulationArgumentV1] {
        &self.arguments
    }

    /// Returns a copied-back buffer by argument ordinal.
    pub fn buffer(&self, argument: usize) -> Option<&BufferArgumentV1> {
        match self.arguments.get(argument) {
            Some(SimulationArgumentV1::Buffer(buffer)) => Some(buffer),
            _ => None,
        }
    }

    /// Consumes the observation and returns all copied-back arguments.
    pub fn into_arguments(self) -> Vec<SimulationArgumentV1> {
        self.arguments
    }

    /// Consumes the observation and returns ABI arguments plus shared backings.
    pub fn into_outputs(self) -> (Vec<SimulationArgumentV1>, Vec<SharedBufferV1>) {
        (self.arguments, self.shared_buffers)
    }

    /// Returns copied-back named backing allocations.
    pub fn shared_buffers(&self) -> &[SharedBufferV1] {
        &self.shared_buffers
    }

    /// Returns one copied-back named backing allocation.
    pub fn shared_buffer(&self, id: BufferBackingIdV1) -> Option<&BufferArgumentV1> {
        self.shared_buffers
            .iter()
            .find(|shared| shared.id == id)
            .map(|shared| &shared.buffer)
    }

    pub const fn invocations_executed(&self) -> u64 {
        self.invocations_executed
    }

    pub const fn workgroups_visited(&self) -> u64 {
        self.workgroups_visited
    }

    pub const fn scheduled_slots_visited(&self) -> u64 {
        self.scheduled_slots_visited
    }

    pub const fn steps_executed(&self) -> u64 {
        self.steps_executed
    }

    pub const fn events_emitted(&self) -> u64 {
        self.events_emitted
    }

    pub const fn schedule(&self) -> SimulationScheduleIdentityV1 {
        self.schedule
    }

    /// Returns the exact identity of the realized semantic CPU ordering.
    pub const fn schedule_transcript_identity(&self) -> &[u8; 32] {
        &self.schedule_transcript_identity
    }

    /// Returns complete runnable-decision and cooperative-barrier coverage.
    pub const fn schedule_coverage(&self) -> SimulationScheduleCoverageV1 {
        self.schedule_coverage
    }

    /// Returns the bounded record when this run explicitly requested recording.
    pub fn schedule_record(&self) -> Option<&SimulationScheduleRecordV1> {
        self.supplemental
            .first()
            .and_then(|supplemental| supplemental.schedule_records.first())
    }

    pub const fn conflict_assessment(&self) -> &SimulationConflictAssessmentV1 {
        &self.conflict_assessment
    }

    /// Returns bounded race and happens-before evidence from this exact CPU schedule.
    pub fn race_assessment(&self) -> &SimulationRaceAssessmentV1 {
        &self
            .supplemental
            .first()
            .expect("successful execution retains one race assessment")
            .race_assessment
    }

    pub(crate) fn into_schedule_and_race(
        mut self,
    ) -> (
        Option<SimulationScheduleRecordV1>,
        SimulationRaceAssessmentV1,
    ) {
        let mut supplemental = self
            .supplemental
            .pop()
            .expect("successful execution retains supplemental observations");
        (
            supplemental.schedule_records.pop(),
            supplemental.race_assessment,
        )
    }

    /// CPU execution is an observation, never a proof or execution authority.
    pub const fn grants_execution_authority(&self) -> bool {
        false
    }
}

/// Preflight or dynamic execution failure.
#[allow(
    clippy::large_enum_variant,
    reason = "dynamic failures retain exact primary and bounded secondary observation diagnostics"
)]
#[derive(Debug)]
pub enum SimulationErrorV1 {
    Preflight(SimulationPreflightErrorV1),
    Execution(SimulationExecutionErrorV1),
}

impl fmt::Display for SimulationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflight(error) => error.fmt(formatter),
            Self::Execution(error) => error.fmt(formatter),
        }
    }
}

impl Error for SimulationErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Preflight(error) => Some(error),
            Self::Execution(error) => Some(error),
        }
    }
}

/// Dynamic execution error with an exact invocation and KIR site when available.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationExecutionErrorV1 {
    pub invocation: Option<SimulationInvocationV1>,
    pub site: Option<SimulationSiteV1>,
    pub kind: SimulationExecutionErrorKindV1,
    /// First failure encountered while truthfully closing observations after the primary error.
    pub observation_failure: Option<SimulationObservationFailureV1>,
}

/// A bounded secondary failure encountered while reporting or closing a primary failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationObservationFailureV1 {
    pub invocation: Option<SimulationInvocationV1>,
    pub site: Option<SimulationSiteV1>,
    pub kind: SimulationExecutionErrorKindV1,
}

impl fmt::Display for SimulationExecutionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "simulation execution failed: {:?}", self.kind)?;
        if let Some(site) = &self.site {
            write!(
                formatter,
                " at {}, {}, op {:?}",
                site.function, site.block, site.operation
            )?;
        }
        if let Some(invocation) = self.invocation {
            write!(formatter, " for global {:?}", invocation.global)?;
        }
        if let Some(observation) = &self.observation_failure {
            write!(
                formatter,
                "; observation closure also failed: {:?}",
                observation.kind
            )?;
        }
        Ok(())
    }
}

impl Error for SimulationExecutionErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            SimulationExecutionErrorKindV1::EventSinkFailure(error) => Some(error),
            _ => self
                .observation_failure
                .as_ref()
                .and_then(|failure| match &failure.kind {
                    SimulationExecutionErrorKindV1::EventSinkFailure(error) => Some(error as _),
                    _ => None,
                }),
        }
    }
}

/// Closed dynamic failure classification for the first execution profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationExecutionErrorKindV1 {
    StepLimit {
        limit: u64,
    },
    EventLimit {
        limit: u64,
    },
    CallDepthLimit {
        limit: usize,
    },
    SsaValueLimit {
        limit: usize,
    },
    AllocationLimit {
        limit: usize,
    },
    AllocationBytesLimit {
        actual: usize,
        limit: usize,
    },
    TotalBytesLimit {
        actual: usize,
        limit: usize,
    },
    AllocationFailure,
    MissingFunction(FunctionId),
    MissingBody(FunctionId),
    UnknownBlock(BlockId),
    MissingTerminator(BlockId),
    UndefinedValue(ValueId),
    RuntimeType {
        value: Option<ValueId>,
        expected: &'static str,
    },
    ResultArity {
        expected: usize,
        actual: usize,
    },
    BlockArgumentArity {
        expected: usize,
        actual: usize,
    },
    UndefinedIntegerOperation(&'static str),
    IntegerOutOfRange,
    PointerOffsetOverflow,
    StorageArrayIndexOutOfBounds {
        index: u64,
        length: u64,
    },
    PointerDistanceDifferentAllocation {
        pointer_allocation: u64,
        origin_allocation: u64,
    },
    PointerDistanceOutOfBounds,
    PointerDistanceNotDivisible {
        byte_difference: u64,
        unit_bytes: u64,
    },
    PointerDistanceOverflow,
    PointerDistanceNegativeUnsigned,
    CopyRangesOverlap {
        allocation: u64,
        source_offset: usize,
        destination_offset: usize,
        bytes: usize,
    },
    MemoryIntrinsicByteCountOverflow,
    DanglingPointer {
        allocation: u64,
    },
    AddressSpaceMismatch,
    ReadOnlyWrite,
    WriteOnlyRead,
    MisalignedAccess {
        required: u32,
        offset: usize,
    },
    OutOfBounds {
        allocation: u64,
        offset: usize,
        bytes: usize,
        allocation_bytes: usize,
    },
    UninitializedRead {
        allocation: u64,
        offset: usize,
        bytes: usize,
    },
    WorkgroupUseBeforePublish {
        allocation: u64,
        offset: usize,
        bytes: usize,
    },
    DivergentWorkgroupBarrier(DivergentWorkgroupBarrierV1),
    MismatchedWorkgroupBarrier(MismatchedWorkgroupBarrierV1),
    IncompleteWave(IncompleteWaveV1),
    DivergentWave(DivergentWaveV1),
    MismatchedWave(MismatchedWaveV1),
    UnsupportedMatrixInputDomain {
        role: crate::MatrixInputRoleV1,
        lane: u8,
        component: u8,
    },
    NonFiniteMatrixResult {
        lane: u8,
        component: u8,
        reduction: u8,
    },
    WaveShuffleSourceOutOfRange {
        source_lane: u32,
        tile_width: u32,
    },
    WorkgroupSchedulerNoProgress {
        phase: u64,
    },
    ScheduleDecisionLimit {
        actual: usize,
        limit: usize,
    },
    ScheduleResidentLimit {
        actual: usize,
        limit: usize,
    },
    ScheduleReplay(SimulationScheduleReplayErrorV1),
    ReachedUnreachable,
    InternalInvariant(&'static str),
    EventSinkFailure(SimulationEventSinkErrorV1),
}

impl SimulationExecutionErrorKindV1 {
    pub(crate) fn retained_heap_bytes(&self) -> usize {
        match self {
            Self::MissingFunction(function) | Self::MissingBody(function) => {
                function.retained_capacity_bytes()
            }
            Self::EventSinkFailure(error) => error.detail.capacity(),
            Self::StepLimit { .. }
            | Self::EventLimit { .. }
            | Self::CallDepthLimit { .. }
            | Self::SsaValueLimit { .. }
            | Self::AllocationLimit { .. }
            | Self::AllocationBytesLimit { .. }
            | Self::TotalBytesLimit { .. }
            | Self::AllocationFailure
            | Self::UnknownBlock(_)
            | Self::MissingTerminator(_)
            | Self::UndefinedValue(_)
            | Self::RuntimeType { .. }
            | Self::ResultArity { .. }
            | Self::BlockArgumentArity { .. }
            | Self::UndefinedIntegerOperation(_)
            | Self::IntegerOutOfRange
            | Self::PointerOffsetOverflow
            | Self::StorageArrayIndexOutOfBounds { .. }
            | Self::PointerDistanceDifferentAllocation { .. }
            | Self::PointerDistanceOutOfBounds
            | Self::PointerDistanceNotDivisible { .. }
            | Self::PointerDistanceOverflow
            | Self::PointerDistanceNegativeUnsigned
            | Self::CopyRangesOverlap { .. }
            | Self::MemoryIntrinsicByteCountOverflow
            | Self::DanglingPointer { .. }
            | Self::AddressSpaceMismatch
            | Self::ReadOnlyWrite
            | Self::WriteOnlyRead
            | Self::MisalignedAccess { .. }
            | Self::OutOfBounds { .. }
            | Self::UninitializedRead { .. }
            | Self::WorkgroupUseBeforePublish { .. }
            | Self::DivergentWorkgroupBarrier(_)
            | Self::MismatchedWorkgroupBarrier(_)
            | Self::IncompleteWave(_)
            | Self::DivergentWave(_)
            | Self::MismatchedWave(_)
            | Self::UnsupportedMatrixInputDomain { .. }
            | Self::NonFiniteMatrixResult { .. }
            | Self::WaveShuffleSourceOutOfRange { .. }
            | Self::WorkgroupSchedulerNoProgress { .. }
            | Self::ScheduleDecisionLimit { .. }
            | Self::ScheduleResidentLimit { .. }
            | Self::ScheduleReplay(_)
            | Self::ReachedUnreachable
            | Self::InternalInvariant(_) => 0,
        }
    }
}

/// Which part of two same-phase workgroup barrier arrivals was incompatible.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkgroupBarrierMismatchV1 {
    Site,
    Semantics,
    SiteAndSemantics,
}

/// Exact bounded detail for a participant exiting while peers wait at a barrier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DivergentWorkgroupBarrierV1 {
    pub phase: u64,
    pub waiting: WorkgroupParticipantV1,
    pub exited: WorkgroupParticipantV1,
}

/// Complete debugger-only participant inventory for a V1 barrier-divergence error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DivergentWorkgroupBarrierV2 {
    pub phase: u64,
    pub waiting: Vec<WorkgroupParticipantV1>,
    pub exited: Vec<WorkgroupParticipantV1>,
}

/// Exact bounded detail for incompatible same-phase barrier arrivals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MismatchedWorkgroupBarrierV1 {
    pub phase: u64,
    pub expected: SimulationEventSiteV1,
    pub mismatch: WorkgroupBarrierMismatchV1,
}

/// Exact active-mask evidence for a final logical wave that cannot satisfy the full-wave KIR contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IncompleteWaveV1 {
    pub width: WaveWidth,
    pub wave_in_workgroup: u64,
    pub active_mask: u64,
    pub required_mask: u64,
}

/// Exact bounded detail for a lane reaching a collective while one active peer does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivergentWaveV1 {
    pub width: WaveWidth,
    pub wave_in_workgroup: u64,
    pub nonparticipating: WorkgroupParticipantV1,
}

/// Exact bounded detail for active lanes reaching different collective operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MismatchedWaveV1 {
    pub width: WaveWidth,
    pub expected: SimulationEventSiteV1,
}

/// Allocation-free workgroup-local participant coordinate retained by diagnostics.
///
/// The enclosing execution error carries the exact workgroup and launch geometry,
/// so this local coordinate uniquely identifies the participant without duplicating
/// the full launch descriptor into every error variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkgroupParticipantV1 {
    pub local: [u32; 3],
}

/// Exact request-local ABI view retained with an out-of-bounds pointer fault.
///
/// This is logical simulator metadata. It contains neither a host/native pointer nor a GPU
/// address. Aliased arguments remain distinguishable by `argument_ordinal`; a consumer joins
/// that ordinal to the admitted request to recover the named backing identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationAbiViewV1 {
    pub argument_ordinal: u32,
    pub byte_offset: usize,
    pub byte_len: usize,
}

/// Exact debugger-only address and ABI-view evidence for a V1 OOB error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationOutOfBoundsV2 {
    pub allocation: u64,
    pub offset: usize,
    pub bytes: usize,
    pub allocation_bytes: usize,
    pub legal_lower_bound: usize,
    pub legal_upper_bound: usize,
    pub abi_view: Option<SimulationAbiViewV1>,
}

const NO_ABI_ARGUMENT_V1: u32 = u32::MAX;

impl From<SimulationInvocationV1> for WorkgroupParticipantV1 {
    fn from(invocation: SimulationInvocationV1) -> Self {
        Self {
            local: invocation.local,
        }
    }
}

impl AdmittedSimulationModuleV1 {
    /// Runs with no event delivery. Caller arguments remain unchanged on success or failure.
    pub fn simulate(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let mut sink = NoopSimulationEventSinkV1;
        self.simulate_with_sink(request, target, limits, &mut sink)
    }

    /// Runs with one explicit runtime-sized LDS segment and no event delivery.
    pub fn simulate_with_dynamic_workgroup_memory(
        &self,
        request: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let mut event_sink = NoopSimulationEventSinkV1;
        let mut debug_sink = NoopSimulationDebugSinkV1;
        let plan = self
            .preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
                schedule: None,
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            &mut debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Runs with an explicit bounded schedule recording or exact replay policy.
    ///
    /// This controls deterministic CPU semantic ordering only. It does not model
    /// GPU scheduling, timing, performance, or physical wave execution.
    pub fn simulate_scheduled(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.simulate_scheduled_with_resident_offset(request, target, limits, schedule, 0)
    }

    /// Runs a scheduled simulation bound to one explicit runtime-sized LDS segment.
    pub fn simulate_scheduled_with_dynamic_workgroup_memory(
        &self,
        request: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.simulate_scheduled_configured_with_resident_offset(
            request,
            Some(dynamic),
            target,
            limits,
            schedule,
            0,
        )
    }

    pub(crate) fn simulate_scheduled_with_resident_offset(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
        resident_offset: usize,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.simulate_scheduled_configured_with_resident_offset(
            request,
            None,
            target,
            limits,
            schedule,
            resident_offset,
        )
    }

    pub(crate) fn simulate_scheduled_configured_with_resident_offset(
        &self,
        request: &SimulationRequestV1,
        dynamic: Option<DynamicWorkgroupMemoryRequestV1>,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
        resident_offset: usize,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let plan = match dynamic {
            Some(dynamic) => {
                self.preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            }
            None => self.preflight(request, target, limits),
        }
        .map_err(SimulationErrorV1::Preflight)?;
        let mut event_sink = NoopSimulationEventSinkV1;
        let mut debug_sink = NoopSimulationDebugSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
                schedule: Some(ExecutionScheduleRequestV1::Public(schedule)),
                resident_offset,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            &mut debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "internal reduction execution keeps every exact simulator context input explicit"
    )]
    pub(crate) fn simulate_reduction_attempt_configured(
        &self,
        request: &SimulationRequestV1,
        dynamic: Option<DynamicWorkgroupMemoryRequestV1>,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        source: ReductionScheduleSourceV1<'_>,
        max_decisions: usize,
        decisions: &mut Vec<crate::SimulationScheduleDecisionV1>,
        resident_offset: usize,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let plan = match dynamic {
            Some(dynamic) => {
                self.preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            }
            None => self.preflight(request, target, limits),
        }
        .map_err(SimulationErrorV1::Preflight)?;
        let mut event_sink = NoopSimulationEventSinkV1;
        let mut debug_sink = NoopSimulationDebugSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
                schedule: Some(ExecutionScheduleRequestV1::Reduction {
                    source,
                    max_decisions,
                    decisions,
                }),
                resident_offset,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            &mut debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Runs with bounded ephemeral event delivery after complete preflight succeeds.
    pub fn simulate_with_sink(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        sink: &mut impl SimulationEventSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.simulate_with_event_policy(request, target, limits, request.events, sink)
    }

    /// Runs with event delivery explicitly enabled without cloning or changing the request.
    ///
    /// This is the adapter boundary for bounded debugger and profiler collectors.
    pub fn simulate_observed_with_sink(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        sink: &mut impl SimulationEventSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.simulate_with_event_policy(request, target, limits, EventPolicyV1::Enabled, sink)
    }

    /// Runs explicit dynamic LDS semantics with bounded event delivery enabled.
    pub fn simulate_observed_with_dynamic_workgroup_memory_and_sink(
        &self,
        request: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        sink: &mut impl SimulationEventSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let plan = self
            .preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        let mut debug_sink = NoopSimulationDebugSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: EventPolicyV1::Enabled,
                plan,
                debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
                schedule: None,
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            sink,
            &mut debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Runs the ordinary simulator while recording an independent bounded debug observation.
    ///
    /// Debug records are derived from live interpreter state and do not alter the stable
    /// [`SimulationEventV1`] stream. Stopping the debug sink only stops debug delivery; the
    /// deterministic simulation continues to completion.
    pub fn simulate_debugged_with_sink(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        capture: SimulationDebugCaptureLimitsV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.check_debug_capture_supported_v20()
            .map_err(SimulationErrorV1::Preflight)?;
        let plan = self
            .preflight(request, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        let mut event_sink = NoopSimulationEventSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: capture,
                schedule: None,
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Runs explicit dynamic LDS semantics with independent bounded debug snapshots.
    pub fn simulate_debugged_with_dynamic_workgroup_memory_and_sink(
        &self,
        request: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        capture: SimulationDebugCaptureLimitsV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.check_debug_capture_supported_v20()
            .map_err(SimulationErrorV1::Preflight)?;
        let plan = self
            .preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        let mut event_sink = NoopSimulationEventSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: capture,
                schedule: None,
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Runs a scheduled simulation with independent bounded debug snapshots.
    pub fn simulate_debugged_scheduled_with_sink(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
        capture: SimulationDebugCaptureLimitsV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.check_debug_capture_supported_v20()
            .map_err(SimulationErrorV1::Preflight)?;
        let plan = self
            .preflight(request, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        let mut event_sink = NoopSimulationEventSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: capture,
                schedule: Some(ExecutionScheduleRequestV1::Public(schedule)),
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Runs scheduled explicit dynamic LDS semantics with bounded debug snapshots.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate_debugged_scheduled_with_dynamic_workgroup_memory_and_sink(
        &self,
        request: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
        capture: SimulationDebugCaptureLimitsV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.check_debug_capture_supported_v20()
            .map_err(SimulationErrorV1::Preflight)?;
        let plan = self
            .preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        let mut event_sink = NoopSimulationEventSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: capture,
                schedule: Some(ExecutionScheduleRequestV1::Public(schedule)),
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    fn simulate_with_event_policy(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        policy: EventPolicyV1,
        sink: &mut impl SimulationEventSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let plan = self
            .preflight(request, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
        let mut debug_sink = NoopSimulationDebugSinkV1;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy,
                plan,
                debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
                schedule: None,
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            sink,
            &mut debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RuntimeValue {
    Scalar(ScalarBitsV1),
    Execution(execution_lifecycle_v18::Token),
    Pointer(PointerValue),
    StoragePointer(StoragePointerValueV18),
    Slice(SliceValue),
    PhysicalEntry(physical_entry_state_v20::Value),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PointerValue {
    exposed_generic: bool,
    allocation: u64,
    byte_offset: usize,
    element: ScalarType,
    address_space: AddressSpace,
    access: AccessMode,
    lower_bound: usize,
    upper_bound: usize,
    abi_argument_ordinal: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoragePointerValueV18 {
    layout: fe2o3_kernel_ir::StorageLayoutIdV1,
    pointer: PointerValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SliceValue {
    exposed_generic: bool,
    allocation: u64,
    elements: usize,
    element: ScalarType,
    address_space: AddressSpace,
    access: AccessMode,
    byte_offset: usize,
    byte_len: usize,
    abi_argument_ordinal: u32,
}

struct Allocation {
    address_space: AddressSpace,
    access: AccessMode,
    alignment: u32,
    bytes: Vec<u8>,
    initialized: Vec<bool>,
    workgroup_published: Vec<bool>,
    workgroup_writer: Vec<u64>,
    // Empty in the default profile; one separately charged descriptor when observed.
    observation_descriptor: Vec<crate::SimulationAllocationDescriptorV1>,
}

struct WorkgroupAllocation {
    site: CompactSite,
    id: u64,
    element: ScalarType,
    bytes: usize,
    lifecycle_observed: bool,
}

struct PreparedStore<'a> {
    bytes: &'a mut [u8],
    initialized: &'a mut [bool],
}

impl PreparedStore<'_> {
    fn commit(self, value: ScalarBitsV1) {
        let raw = value.bits().to_le_bytes();
        self.bytes.copy_from_slice(&raw[..self.bytes.len()]);
        self.initialized.fill(true);
    }
}

struct Memory {
    allocations: HashMap<u64, Allocation>,
    argument_allocations: Vec<Option<u64>>,
    shared_allocations: HashMap<BufferBackingIdV1, u64>,
    next_allocation: u64,
    allocations_created: usize,
    live_bytes: usize,
    reuse: Option<AllocationReusePool>,
}

impl Memory {
    fn new(
        argument_count: usize,
        shared_count: usize,
        limits: SimulationLimitsV1,
        allocation_reuse: Option<SimulationAllocationReuseV1>,
    ) -> Result<Self, SimulationExecutionErrorKindV1> {
        let mut allocations = HashMap::new();
        allocations
            .try_reserve(limits.max_allocations)
            .map_err(|_| SimulationExecutionErrorKindV1::AllocationFailure)?;
        let mut shared_allocations = HashMap::new();
        shared_allocations
            .try_reserve(shared_count)
            .map_err(|_| SimulationExecutionErrorKindV1::AllocationFailure)?;
        Ok(Self {
            allocations,
            argument_allocations: try_filled(argument_count, None)?,
            shared_allocations,
            next_allocation: 1,
            allocations_created: 0,
            live_bytes: 0,
            reuse: allocation_reuse.map(AllocationReusePool::new),
        })
    }

    fn validate_allocation(
        &self,
        bytes: usize,
        limits: SimulationLimitsV1,
    ) -> Result<(), SimulationExecutionErrorKindV1> {
        if self.allocations_created >= limits.max_allocations {
            return Err(SimulationExecutionErrorKindV1::AllocationLimit {
                limit: limits.max_allocations,
            });
        }
        if bytes > limits.max_allocation_bytes {
            return Err(SimulationExecutionErrorKindV1::AllocationBytesLimit {
                actual: bytes,
                limit: limits.max_allocation_bytes,
            });
        }
        let total = self.live_bytes.checked_add(bytes).ok_or(
            SimulationExecutionErrorKindV1::TotalBytesLimit {
                actual: usize::MAX,
                limit: limits.max_total_bytes,
            },
        )?;
        if total > limits.max_total_bytes {
            return Err(SimulationExecutionErrorKindV1::TotalBytesLimit {
                actual: total,
                limit: limits.max_total_bytes,
            });
        }
        Ok(())
    }

    fn load(
        &self,
        pointer: &PointerValue,
        access: MemoryAccess,
        target: SimulationTargetV1,
        invocation: SimulationInvocationV1,
    ) -> Result<ScalarBitsV1, SimulationExecutionErrorKindV1> {
        let allocation = self.allocation(pointer)?;
        let width = target.scalar_bytes(pointer.element).ok_or(
            SimulationExecutionErrorKindV1::InternalInvariant("preflighted load element"),
        )?;
        validate_access(allocation, pointer, access, width, false)?;
        let end = pointer
            .byte_offset
            .checked_add(width)
            .ok_or(SimulationExecutionErrorKindV1::PointerOffsetOverflow)?;
        if allocation.initialized[pointer.byte_offset..end]
            .iter()
            .any(|initialized| !initialized)
        {
            return Err(SimulationExecutionErrorKindV1::UninitializedRead {
                allocation: pointer.allocation,
                offset: pointer.byte_offset,
                bytes: width,
            });
        }
        if pointer.address_space == AddressSpace::Workgroup {
            let writer = invocation_local_ordinal(invocation)
                .and_then(|ordinal| ordinal.checked_add(1))
                .ok_or(SimulationExecutionErrorKindV1::InternalInvariant(
                    "workgroup invocation ordinal",
                ))?;
            if allocation.workgroup_published[pointer.byte_offset..end]
                .iter()
                .zip(&allocation.workgroup_writer[pointer.byte_offset..end])
                .any(|(published, owner)| !published && *owner != writer)
            {
                return Err(SimulationExecutionErrorKindV1::WorkgroupUseBeforePublish {
                    allocation: pointer.allocation,
                    offset: pointer.byte_offset,
                    bytes: width,
                });
            }
        }
        let mut raw = [0_u8; 16];
        raw[..width].copy_from_slice(&allocation.bytes[pointer.byte_offset..end]);
        ScalarBitsV1::new(pointer.element, u128::from_le_bytes(raw), target)
            .map_err(|_| SimulationExecutionErrorKindV1::InternalInvariant("loaded scalar bits"))
    }

    fn validate_store(
        &self,
        pointer: &PointerValue,
        access: MemoryAccess,
        value: ScalarBitsV1,
        target: SimulationTargetV1,
    ) -> Result<usize, SimulationExecutionErrorKindV1> {
        if value.ty() != pointer.element {
            return Err(SimulationExecutionErrorKindV1::RuntimeType {
                value: None,
                expected: "pointer element scalar",
            });
        }
        let width = target.scalar_bytes(pointer.element).ok_or(
            SimulationExecutionErrorKindV1::InternalInvariant("preflighted store element"),
        )?;
        let allocation = self.allocation(pointer)?;
        validate_access(allocation, pointer, access, width, true)?;
        Ok(width)
    }

    fn validate_range_read(
        &self,
        pointer: &PointerValue,
        alignment: u32,
        width: usize,
        invocation: SimulationInvocationV1,
    ) -> Result<(), SimulationExecutionErrorKindV1> {
        let allocation = self.allocation(pointer)?;
        validate_access(
            allocation,
            pointer,
            MemoryAccess::new(pointer.address_space, alignment),
            width,
            false,
        )?;
        let end = pointer
            .byte_offset
            .checked_add(width)
            .ok_or(SimulationExecutionErrorKindV1::PointerOffsetOverflow)?;
        if allocation.initialized[pointer.byte_offset..end]
            .iter()
            .any(|initialized| !initialized)
        {
            return Err(SimulationExecutionErrorKindV1::UninitializedRead {
                allocation: pointer.allocation,
                offset: pointer.byte_offset,
                bytes: width,
            });
        }
        if pointer.address_space == AddressSpace::Workgroup {
            let writer = invocation_local_ordinal(invocation)
                .and_then(|ordinal| ordinal.checked_add(1))
                .ok_or(SimulationExecutionErrorKindV1::InternalInvariant(
                    "workgroup invocation ordinal",
                ))?;
            if allocation.workgroup_published[pointer.byte_offset..end]
                .iter()
                .zip(&allocation.workgroup_writer[pointer.byte_offset..end])
                .any(|(published, owner)| !published && *owner != writer)
            {
                return Err(SimulationExecutionErrorKindV1::WorkgroupUseBeforePublish {
                    allocation: pointer.allocation,
                    offset: pointer.byte_offset,
                    bytes: width,
                });
            }
        }
        Ok(())
    }

    fn validate_range_write(
        &self,
        pointer: &PointerValue,
        alignment: u32,
        width: usize,
    ) -> Result<(), SimulationExecutionErrorKindV1> {
        let allocation = self.allocation(pointer)?;
        validate_access(
            allocation,
            pointer,
            MemoryAccess::new(pointer.address_space, alignment),
            width,
            true,
        )
    }

    fn validate_zero_byte_pointer(
        &self,
        pointer: &PointerValue,
        alignment: u32,
    ) -> Result<(), SimulationExecutionErrorKindV1> {
        let allocation = self.allocation(pointer)?;
        if alignment == 0
            || allocation.alignment < alignment
            || !pointer.byte_offset.is_multiple_of(alignment as usize)
        {
            return Err(SimulationExecutionErrorKindV1::MisalignedAccess {
                required: alignment,
                offset: pointer.byte_offset,
            });
        }
        Ok(())
    }

    fn commit_copy(
        &mut self,
        source: &PointerValue,
        destination: &PointerValue,
        width: usize,
        workgroup_writer: Option<u64>,
    ) {
        for relative in 0..width {
            let source_offset = source.byte_offset + relative;
            let destination_offset = destination.byte_offset + relative;
            let byte = self.allocations[&source.allocation].bytes[source_offset];
            let destination_allocation = self
                .allocations
                .get_mut(&destination.allocation)
                .expect("validated copy destination remains live");
            destination_allocation.bytes[destination_offset] = byte;
            destination_allocation.initialized[destination_offset] = true;
        }
        if let Some(writer) = workgroup_writer {
            let destination_end = destination.byte_offset + width;
            let destination_allocation = self
                .allocations
                .get_mut(&destination.allocation)
                .expect("validated workgroup copy destination remains live");
            destination_allocation.workgroup_published[destination.byte_offset..destination_end]
                .fill(false);
            destination_allocation.workgroup_writer[destination.byte_offset..destination_end]
                .fill(writer);
        }
    }

    fn prepare_store(
        &mut self,
        pointer: &PointerValue,
        width: usize,
    ) -> Result<PreparedStore<'_>, SimulationExecutionErrorKindV1> {
        let allocation = self.allocations.get_mut(&pointer.allocation).ok_or(
            SimulationExecutionErrorKindV1::InternalInvariant("validated allocation remained live"),
        )?;
        let end = pointer
            .byte_offset
            .checked_add(width)
            .ok_or(SimulationExecutionErrorKindV1::PointerOffsetOverflow)?;
        let bytes = allocation.bytes.get_mut(pointer.byte_offset..end).ok_or(
            SimulationExecutionErrorKindV1::InternalInvariant(
                "validated store byte range remained live",
            ),
        )?;
        let initialized = allocation
            .initialized
            .get_mut(pointer.byte_offset..end)
            .ok_or(SimulationExecutionErrorKindV1::InternalInvariant(
                "validated store initialization range remained live",
            ))?;
        Ok(PreparedStore { bytes, initialized })
    }

    fn mark_workgroup_store(
        &mut self,
        pointer: &PointerValue,
        width: usize,
        invocation: SimulationInvocationV1,
    ) -> Result<(), SimulationExecutionErrorKindV1> {
        if pointer.address_space != AddressSpace::Workgroup {
            return Ok(());
        }
        let allocation = self.allocations.get_mut(&pointer.allocation).ok_or(
            SimulationExecutionErrorKindV1::DanglingPointer {
                allocation: pointer.allocation,
            },
        )?;
        let end = pointer
            .byte_offset
            .checked_add(width)
            .ok_or(SimulationExecutionErrorKindV1::PointerOffsetOverflow)?;
        allocation.workgroup_published[pointer.byte_offset..end].fill(false);
        let writer = invocation_local_ordinal(invocation)
            .and_then(|ordinal| ordinal.checked_add(1))
            .ok_or(SimulationExecutionErrorKindV1::InternalInvariant(
                "workgroup invocation ordinal",
            ))?;
        allocation.workgroup_writer[pointer.byte_offset..end].fill(writer);
        Ok(())
    }

    fn mark_workgroup_atomic(
        &mut self,
        pointer: &PointerValue,
        width: usize,
    ) -> Result<(), SimulationExecutionErrorKindV1> {
        if pointer.address_space != AddressSpace::Workgroup {
            return Ok(());
        }
        let allocation = self.allocations.get_mut(&pointer.allocation).ok_or(
            SimulationExecutionErrorKindV1::DanglingPointer {
                allocation: pointer.allocation,
            },
        )?;
        let end = pointer
            .byte_offset
            .checked_add(width)
            .ok_or(SimulationExecutionErrorKindV1::PointerOffsetOverflow)?;
        allocation.workgroup_published[pointer.byte_offset..end].fill(true);
        allocation.workgroup_writer[pointer.byte_offset..end].fill(0);
        Ok(())
    }

    fn publish_workgroup(&mut self) {
        for allocation in self
            .allocations
            .values_mut()
            .filter(|allocation| allocation.address_space == AddressSpace::Workgroup)
        {
            for (published, initialized) in allocation
                .workgroup_published
                .iter_mut()
                .zip(&allocation.initialized)
            {
                *published |= *initialized;
            }
        }
    }

    fn allocation(
        &self,
        pointer: &PointerValue,
    ) -> Result<&Allocation, SimulationExecutionErrorKindV1> {
        let allocation = self.allocations.get(&pointer.allocation).ok_or(
            SimulationExecutionErrorKindV1::DanglingPointer {
                allocation: pointer.allocation,
            },
        )?;
        if allocation.address_space != pointer.address_space {
            return Err(SimulationExecutionErrorKindV1::AddressSpaceMismatch);
        }
        Ok(allocation)
    }
}

fn invocation_local_ordinal(invocation: SimulationInvocationV1) -> Option<u64> {
    let x = u64::from(invocation.local[0]);
    let y = u64::from(invocation.local[1]);
    let z = u64::from(invocation.local[2]);
    let width = u64::from(invocation.workgroup_size[0]);
    let plane = width.checked_mul(u64::from(invocation.workgroup_size[1]))?;
    x.checked_add(y.checked_mul(width)?)?
        .checked_add(z.checked_mul(plane)?)
}

fn validate_access(
    allocation: &Allocation,
    pointer: &PointerValue,
    access: MemoryAccess,
    width: usize,
    write: bool,
) -> Result<(), SimulationExecutionErrorKindV1> {
    if access.address_space != pointer.logical_address_space() {
        return Err(SimulationExecutionErrorKindV1::AddressSpaceMismatch);
    }
    if write
        && (!matches!(
            pointer.access,
            AccessMode::WriteOnly | AccessMode::ReadWrite
        ) || !matches!(
            allocation.access,
            AccessMode::WriteOnly | AccessMode::ReadWrite
        ) || pointer.address_space == AddressSpace::Constant)
    {
        return Err(SimulationExecutionErrorKindV1::ReadOnlyWrite);
    }
    if !write
        && (!matches!(pointer.access, AccessMode::ReadOnly | AccessMode::ReadWrite)
            || !matches!(
                allocation.access,
                AccessMode::ReadOnly | AccessMode::ReadWrite
            ))
    {
        return Err(SimulationExecutionErrorKindV1::WriteOnlyRead);
    }
    if access.alignment == 0
        || allocation.alignment < access.alignment
        || !pointer
            .byte_offset
            .is_multiple_of(access.alignment as usize)
    {
        return Err(SimulationExecutionErrorKindV1::MisalignedAccess {
            required: access.alignment,
            offset: pointer.byte_offset,
        });
    }
    let end = pointer
        .byte_offset
        .checked_add(width)
        .ok_or_else(|| out_of_bounds_error(pointer, width, allocation.bytes.len()))?;
    if pointer.byte_offset < pointer.lower_bound
        || end > pointer.upper_bound
        || end > allocation.bytes.len()
    {
        return Err(out_of_bounds_error(pointer, width, allocation.bytes.len()));
    }
    Ok(())
}

fn out_of_bounds_error(
    pointer: &PointerValue,
    bytes: usize,
    allocation_bytes: usize,
) -> SimulationExecutionErrorKindV1 {
    SimulationExecutionErrorKindV1::OutOfBounds {
        allocation: pointer.allocation,
        offset: pointer.byte_offset,
        bytes,
        allocation_bytes,
    }
}

fn out_of_bounds_detail_v2(
    pointer: &PointerValue,
    bytes: usize,
    allocation_bytes: usize,
) -> SimulationOutOfBoundsV2 {
    let abi_view = (pointer.abi_argument_ordinal != NO_ABI_ARGUMENT_V1)
        .then_some(())
        .and_then(|()| {
            pointer
                .upper_bound
                .checked_sub(pointer.lower_bound)
                .map(|byte_len| SimulationAbiViewV1 {
                    argument_ordinal: pointer.abi_argument_ordinal,
                    byte_offset: pointer.lower_bound,
                    byte_len,
                })
        });
    SimulationOutOfBoundsV2 {
        allocation: pointer.allocation,
        offset: pointer.byte_offset,
        bytes,
        allocation_bytes,
        legal_lower_bound: pointer.lower_bound,
        legal_upper_bound: pointer.upper_bound,
        abi_view,
    }
}

struct Engine<'a, S> {
    module: &'a fe2o3_kernel_ir::Module,
    physical_lds_context: Option<&'a physical_lds_context_v22::Context>,
    physical_lds_state: Option<physical_lds_context_v22::State>,
    function_module_indices: Vec<usize>,
    block_indices: Vec<HashMap<BlockId, usize>>,
    function_ssa_values: Vec<usize>,
    value_layouts: &'a [ValueLayout<'a>],
    value_plan: FrameValuePlan,
    call_targets: Vec<Vec<Vec<CallTarget>>>,
    switch_targets: Vec<Vec<SwitchLookup>>,
    target: SimulationTargetV1,
    dynamic_workgroup_memory: Option<DynamicWorkgroupMemoryRequestV1>,
    matrix_bf16_numerics: crate::MatrixBf16NumericsV1,
    limits: SimulationLimitsV1,
    policy: EventPolicyV1,
    memory: Memory,
    sink: &'a mut S,
    debug_capture: SimulationDebugCaptureLimitsV1,
    debug_sink: &'a mut dyn SimulationDebugSinkV1,
    physical_debug: Option<&'a mut debug_physical_capture_v20::State>,
    debug_origin_requested: bool,
    debug_observation_requested: bool,
    debug_frames_requested: bool,
    debug_identity_failed: bool,
    debug_origin: Option<OperationIdentity>,
    debug_records: u64,
    debug_delivery_stopped: bool,
    allocation_lifecycle_requested: bool,
    allocation_lifecycle_stopped: bool,
    schedule_identity: SimulationScheduleIdentityV1,
    schedule_decision: u64,
    steps: u64,
    events: u64,
    reserved_event_closures: u64,
    event_delivery_stopped: bool,
    invocation: Option<SimulationInvocationV1>,
    accesses: HashMap<(u64, usize), AccessFrontier>,
    conflicting_bytes: u64,
    first_conflict: Option<SimulationMemoryConflictV1>,
    conflict_incomplete: bool,
    workgroup_happens_before_epoch: u64,
    unmodeled_atomic_or_fence_happens_before: bool,
    unmodeled_atomic_serialization: bool,
    race_trackers: Vec<RaceTracker>,
    workgroup_allocations: Vec<WorkgroupAllocation>,
}

struct RaceTracker {
    racing_bytes: u64,
    first_race: Option<SimulationDataRaceV1>,
    first_ordered_conflict: Option<SimulationOrderedMemoryConflictV1>,
}

#[derive(Clone, Copy, Default)]
struct AccessFrontier {
    write: Option<LastAccess>,
    displaced_write: Option<LastAccess>,
    read: Option<LastAccess>,
    displaced_read: Option<LastAccess>,
    conflicted: bool,
    raced: bool,
    incomplete: bool,
    lost_write: bool,
    lost_writes_atomic: Option<AtomicHistory>,
    lost_read: bool,
    lost_reads_atomic: Option<AtomicHistory>,
}

#[derive(Clone, Copy)]
struct LastAccess {
    invocation: AccessInvocationV1,
    site: CompactSite,
    atomic: Option<AtomicAccess>,
    happens_before_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CompactSite {
    function: usize,
    block: BlockId,
    operation: Option<u32>,
}

enum SwitchLookup {
    None,
    Index(HashMap<u128, usize>),
    Integer(HashMap<(ScalarType, u128), usize>),
}

#[derive(Clone, Copy)]
enum CallTarget {
    NotCall,
    Internal(usize),
    Float(SoftFloatOperationV1),
    Trap,
}

fn debug_value(value: &RuntimeValue) -> Option<SimulationDebugValueV1> {
    Some(match value {
        RuntimeValue::PhysicalEntry(_)
        | RuntimeValue::StoragePointer(_)
        | RuntimeValue::Execution(_) => return None,
        RuntimeValue::Scalar(value) => SimulationDebugValueV1::Scalar(*value),
        RuntimeValue::Pointer(value) => SimulationDebugValueV1::Pointer {
            allocation: value.allocation,
            byte_offset: value.byte_offset,
            element: value.element,
            address_space: value.logical_address_space(),
            access: value.access,
            lower_bound: value.lower_bound,
            upper_bound: value.upper_bound,
        },
        RuntimeValue::Slice(value) => SimulationDebugValueV1::Slice {
            allocation: value.allocation,
            elements: value.elements,
            element: value.element,
            address_space: value.logical_address_space(),
            access: value.access,
            byte_offset: value.byte_offset,
            byte_len: value.byte_len,
        },
    })
}

fn capture_debug_stack(
    frames: &[RuntimeFrame<'_>],
    function_module_indices: &[usize],
    limits: SimulationDebugCaptureLimitsV1,
) -> SimulationDebugCollectionV1<SimulationDebugFrameV1> {
    if frames.len() > limits.max_frames_per_checkpoint() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::FrameLimit,
            required: u64::try_from(frames.len()).unwrap_or(u64::MAX),
        };
    }
    let value_count = frames.iter().try_fold(0_usize, |count, frame| {
        count.checked_add(frame.values.len())
    });
    let Some(value_count) = value_count else {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::ValueLimit,
            required: u64::MAX,
        };
    };
    if value_count > limits.max_values_per_checkpoint() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::ValueLimit,
            required: u64::try_from(value_count).unwrap_or(u64::MAX),
        };
    }
    let mut captured = Vec::new();
    if captured.try_reserve_exact(frames.len()).is_err() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::AllocationFailure,
            required: u64::try_from(frames.len()).unwrap_or(u64::MAX),
        };
    }
    for (depth, frame) in frames.iter().enumerate() {
        let mut ordered = Vec::new();
        if ordered.try_reserve_exact(frame.values.len()).is_err() {
            return SimulationDebugCollectionV1::Unavailable {
                reason: SimulationDebugUnavailableReasonV1::AllocationFailure,
                required: u64::try_from(value_count).unwrap_or(u64::MAX),
            };
        }
        ordered.extend(frame.values.ids());
        ordered.sort_unstable();
        let mut values = Vec::new();
        if values.try_reserve_exact(ordered.len()).is_err() {
            return SimulationDebugCollectionV1::Unavailable {
                reason: SimulationDebugUnavailableReasonV1::AllocationFailure,
                required: u64::try_from(value_count).unwrap_or(u64::MAX),
            };
        }
        for value in ordered {
            let observed = frame.values.get(&value);
            let Some(observed) = observed.as_ref().and_then(debug_value) else {
                // Tagged storage, execution roles and symbolic bindings have no legacy DTO.
                // Never erase their type identity or silently omit a binding.
                return SimulationDebugCollectionV1::Unavailable {
                    reason: SimulationDebugUnavailableReasonV1::NotCaptured,
                    required: u64::try_from(value_count).unwrap_or(u64::MAX),
                };
            };
            values.push(SimulationDebugBindingV1 { value, observed });
        }
        let Some(function_ordinal) = function_module_indices.get(frame.function_index).copied()
        else {
            return SimulationDebugCollectionV1::Unavailable {
                reason: SimulationDebugUnavailableReasonV1::NotCaptured,
                required: 0,
            };
        };
        captured.push(SimulationDebugFrameV1 {
            depth: u32::try_from(depth).unwrap_or(u32::MAX),
            function_ordinal,
            block: frame.current,
            next_operation: frame
                .function
                .body
                .as_ref()
                .and_then(|body| body.blocks.get(frame.current_index))
                .and_then(|block| block.operations.get(frame.operation))
                .and_then(|_| u32::try_from(frame.operation).ok()),
            values: SimulationDebugCollectionV1::Captured(values),
        });
    }
    SimulationDebugCollectionV1::Captured(captured)
}

fn capture_debug_memory(
    memory: &Memory,
    limits: SimulationDebugCaptureLimitsV1,
) -> SimulationDebugCollectionV1<SimulationDebugAllocationV1> {
    if memory.allocations.len() > limits.max_allocations_per_checkpoint() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::AllocationLimit,
            required: u64::try_from(memory.allocations.len()).unwrap_or(u64::MAX),
        };
    }
    let Some(byte_count) = memory
        .allocations
        .values()
        .try_fold(0_usize, |count, allocation| {
            count
                .checked_add(allocation.bytes.len())?
                .checked_add(allocation.initialized.len())
        })
    else {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::MemoryByteLimit,
            required: u64::MAX,
        };
    };
    if byte_count > limits.max_memory_bytes_per_checkpoint() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::MemoryByteLimit,
            required: u64::try_from(byte_count).unwrap_or(u64::MAX),
        };
    }
    let mut ordered = Vec::new();
    if ordered.try_reserve_exact(memory.allocations.len()).is_err() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::AllocationFailure,
            required: u64::try_from(memory.allocations.len()).unwrap_or(u64::MAX),
        };
    }
    ordered.extend(memory.allocations.iter());
    ordered.sort_unstable_by_key(|(id, _)| **id);
    let mut captured = Vec::new();
    if captured.try_reserve_exact(ordered.len()).is_err() {
        return SimulationDebugCollectionV1::Unavailable {
            reason: SimulationDebugUnavailableReasonV1::AllocationFailure,
            required: u64::try_from(ordered.len()).unwrap_or(u64::MAX),
        };
    }
    for (id, allocation) in ordered {
        let mut bytes = Vec::new();
        let mut initialized = Vec::new();
        if bytes.try_reserve_exact(allocation.bytes.len()).is_err()
            || initialized
                .try_reserve_exact(allocation.initialized.len())
                .is_err()
        {
            return SimulationDebugCollectionV1::Unavailable {
                reason: SimulationDebugUnavailableReasonV1::AllocationFailure,
                required: u64::try_from(byte_count).unwrap_or(u64::MAX),
            };
        }
        bytes.extend_from_slice(&allocation.bytes);
        initialized.extend(allocation.initialized.iter().copied());
        captured.push(SimulationDebugAllocationV1 {
            allocation: *id,
            address_space: allocation.address_space,
            access: allocation.access,
            alignment: allocation.alignment,
            bytes,
            initialized,
        });
    }
    SimulationDebugCollectionV1::Captured(captured)
}

fn frame_stack_resident_bytes(depth: usize) -> Option<usize> {
    // InvocationMachine::new and Call reserve exactly one before each push;
    // inactive frames retain that same bounded capacity across reset/return.
    reserved_vec_bytes::<RuntimeFrame<'static>>(depth)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conservative_execution_resident_bytes(
    admitted_resident_bytes: usize,
    request: &SimulationRequestV1,
    limits: SimulationLimitsV1,
    reserved_call_depth: usize,
    reachable_ssa_values: usize,
    incoming_capacity: IncomingValueCapacityV1,
    value_plan: FrameValuePlan,
    plan_identity_bytes: usize,
    reachable_indices_capacity: usize,
    execution_index_resident_bytes: usize,
    maximum_reachable_identifier_bytes: usize,
    workgroup_participants: usize,
    workgroup_allocation_sites: usize,
    workgroup_static_bytes: usize,
) -> Option<usize> {
    let arguments = request.arguments.len();
    let shared = request.shared_buffers.len();
    let mut resident = ResidentLedger::new(admitted_resident_bytes);
    resident.add_bytes(size_of::<Engine<'static, NoopSimulationEventSinkV1>>())?;
    // Physical values own no heap. SSA/frame/table size_of charges include them;
    // reserve two fixed result temporaries conservatively for return/binding moves.
    resident.add_product(size_of::<physical_entry_state_v20::Results>(), 2)?;
    resident.add_bytes(reserved_vec_bytes::<RaceTracker>(1)?)?;
    resident.add_bytes(size_of::<SimulationRequestV1>())?;
    resident.add_bytes(size_of::<SimulationPlanV1>())?;
    resident.add_bytes(size_of::<SimulationExecutionV1>())?;
    resident.add_bytes(reserved_vec_bytes::<SimulationSupplementalV1>(1)?)?;
    resident.add_bytes(request.kernel.retained_capacity_bytes())?;
    resident.add_bytes(plan_identity_bytes)?;
    resident.add_vec::<usize>(reachable_indices_capacity)?;
    resident.add_bytes(execution_index_resident_bytes)?;
    // At successful return the live engine can own first conflict, first race,
    // and first ordered-conflict evidence while the returned conflict and race
    // assessments own clones of all four sites: twelve function identities.
    // Dynamic primary plus bounded secondary errors require no more clones.
    resident.add_product(maximum_reachable_identifier_bytes, 12)?;
    resident.add_product(
        reserved_vec_bytes::<WorkgroupParticipantV1>(workgroup_participants)?,
        2,
    )?;

    // The borrowed request and completed result coexist with simulated memory at copy-back.
    resident.add_product(
        request.arguments.capacity(),
        size_of::<SimulationArgumentV1>(),
    )?;
    resident.add_product(
        request.shared_buffers.capacity(),
        size_of::<SharedBufferV1>(),
    )?;
    for argument in &request.arguments {
        if let SimulationArgumentV1::Buffer(buffer) = argument {
            resident.add_bytes(buffer.retained_payload_capacity_bytes()?)?;
        }
    }
    for backing in &request.shared_buffers {
        resident.add_bytes(backing.buffer.retained_payload_capacity_bytes()?)?;
    }
    resident.add_bytes(reserved_vec_bytes::<SimulationArgumentV1>(arguments)?)?;
    resident.add_bytes(reserved_vec_bytes::<SharedBufferV1>(shared)?)?;
    for argument in &request.arguments {
        if let SimulationArgumentV1::Buffer(buffer) = argument {
            resident.add_bytes(reserved_vec_bytes::<u8>(buffer.bytes().len())?)?;
            resident.add_bytes(reserved_bool_vec_bytes(buffer.initialized().len())?)?;
        }
    }
    for backing in &request.shared_buffers {
        resident.add_bytes(reserved_vec_bytes::<u8>(backing.buffer.bytes().len())?)?;
        resident.add_bytes(reserved_bool_vec_bytes(backing.buffer.initialized().len())?)?;
    }

    // Allocation payloads use exact reservation on the pinned toolchain.
    // Vec<bool> capacity counts ordinary bool elements, each charged in bytes.
    resident.add_bytes(limits.max_total_bytes)?;
    resident.add_bytes(partitioned_bool_vec_storage_bytes(
        limits.max_total_bytes,
        limits.max_allocations,
    )?)?;
    resident.add_bytes(reserved_hash_map_bytes::<u64, Allocation>(
        limits.max_allocations,
    )?)?;
    resident.add_bytes(reserved_hash_map_bytes::<BufferBackingIdV1, u64>(shared)?)?;
    resident.add_bytes(reserved_vec_bytes::<Option<u64>>(arguments)?)?;

    // The scheduler retains parameters, frame storage, SSA tables and branch/call temporaries.
    resident.add_product(reserved_vec_bytes::<RuntimeValue>(arguments)?, 2)?;
    resident.add_bytes(reserved_vec_bytes::<InvocationMachine<'static>>(
        workgroup_participants,
    )?)?;
    resident.add_bytes(reserved_vec_bytes::<(usize, ScalarBitsV1)>(
        workgroup_participants,
    )?)?;
    resident.add_product(
        workgroup_participants,
        frame_stack_resident_bytes(reserved_call_depth)?,
    )?;
    // InvocationMachine::new owns the next frame before it is moved into one reserved stack.
    resident.add_bytes(size_of::<RuntimeFrame<'static>>())?;
    resident.add_bytes(value_plan.shared_bytes)?;
    resident.add_bytes(values_v1::fixed_temporary_bytes()?)?;
    let values_per_frame = value_plan.frame_bytes()?;
    resident.add_product(
        workgroup_participants.checked_mul(reserved_call_depth)?,
        values_per_frame,
    )?;
    let incoming_per_frame = reserved_vec_bytes::<RuntimeValue>(incoming_capacity.0)?;
    resident.add_product(
        workgroup_participants.checked_mul(reserved_call_depth)?,
        incoming_per_frame,
    )?;
    resident.add_product(reserved_vec_bytes::<RuntimeValue>(reachable_ssa_values)?, 2)?;
    resident.add_bytes(partitioned_geometric_vec_bytes::<FrameAllocation>(
        limits.max_allocations,
        workgroup_participants.checked_mul(reserved_call_depth)?,
    )?)?;
    resident.add_bytes(reserved_vec_bytes::<WorkgroupAllocation>(
        workgroup_allocation_sites,
    )?)?;
    resident.add_bytes(reserved_bool_vec_bytes(workgroup_static_bytes)?)?;
    resident.add_product(workgroup_static_bytes, size_of::<u64>())?;

    resident.add_bytes(reserved_hash_map_bytes::<(u64, usize), AccessFrontier>(
        limits.max_memory_access_records,
    )?)?;
    Some(resident.bytes())
}

#[cfg(test)]
mod execution_resident_tests {
    use super::*;

    #[test]
    fn frame_stack_charge_matches_exact_growth_and_retained_capacity() {
        let mut frames = Vec::<std::mem::MaybeUninit<RuntimeFrame<'static>>>::new();
        for depth in 1..=9 {
            if frames.len() == frames.capacity() {
                frames.try_reserve_exact(1).unwrap();
            }
            frames.push(std::mem::MaybeUninit::uninit());
            assert_eq!(
                frame_stack_resident_bytes(depth).unwrap(),
                frames.capacity() * size_of::<RuntimeFrame<'static>>()
            );
        }
        let capacity = frames.capacity();
        frames.clear();
        assert_eq!(frames.capacity(), capacity);
        assert_eq!(
            frame_stack_resident_bytes(9).unwrap(),
            capacity * size_of::<RuntimeFrame<'static>>()
        );
    }

    #[test]
    fn incoming_capacity_reserves_exact_retained_vectors_per_live_frame() {
        let request = SimulationRequestV1::new("incoming", [1, 1, 1], [1, 1, 1], vec![]);
        let limits = SimulationLimitsV1::default();
        let accounted = |count| {
            conservative_execution_resident_bytes(
                0,
                &request,
                limits,
                3,
                53,
                IncomingValueCapacityV1(count),
                FrameValuePlan::legacy(53),
                0,
                0,
                0,
                0,
                7,
                0,
                0,
            )
            .unwrap()
        };
        for (small, large) in [(0, 1), (1, 17), (17, 53), (53, 127)] {
            let delta = reserved_vec_bytes::<RuntimeValue>(large).unwrap()
                - reserved_vec_bytes::<RuntimeValue>(small).unwrap();
            assert_eq!(accounted(large) - accounted(small), 7 * 3 * delta);
        }
    }

    #[test]
    fn atomic_scope_frontier_resident_accounting_uses_actual_compact_cells() {
        let request = SimulationRequestV1::new("frontier-resident", [1, 1, 1], [1, 1, 1], vec![]);
        let accounted = |records| {
            let limits = SimulationLimitsV1 {
                max_call_depth: 1,
                max_ssa_values: 1,
                max_allocations: 1,
                max_allocation_bytes: 1,
                max_total_bytes: 1,
                max_memory_access_records: records,
                ..SimulationLimitsV1::default()
            };
            conservative_execution_resident_bytes(
                0,
                &request,
                limits,
                limits.max_call_depth,
                0,
                IncomingValueCapacityV1(0),
                FrameValuePlan::legacy(0),
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            )
            .expect("bounded resident accounting")
        };
        for (small, large) in [(1, 17), (17, 257)] {
            let actual_delta = accounted(large).checked_sub(accounted(small)).unwrap();
            let charged_delta = reserved_hash_map_bytes::<(u64, usize), AccessFrontier>(large)
                .unwrap()
                .checked_sub(
                    reserved_hash_map_bytes::<(u64, usize), AccessFrontier>(small).unwrap(),
                )
                .unwrap();
            assert_eq!(actual_delta, charged_delta);
            assert!(actual_delta > 0);
        }
    }

    #[test]
    fn successful_assessment_accounts_twelve_live_function_identity_allocations() {
        let request = SimulationRequestV1::new("resident", [1, 1, 1], [1, 1, 1], vec![]);
        let limits = SimulationLimitsV1 {
            max_call_depth: 1,
            max_ssa_values: 1,
            max_allocations: 1,
            max_allocation_bytes: 1,
            max_total_bytes: 1,
            max_memory_access_records: 1,
            ..SimulationLimitsV1::default()
        };
        let accounted = |identifier_bytes| {
            conservative_execution_resident_bytes(
                0,
                &request,
                limits,
                limits.max_call_depth,
                0,
                IncomingValueCapacityV1(0),
                FrameValuePlan::legacy(0),
                0,
                0,
                0,
                identifier_bytes,
                1,
                0,
                0,
            )
            .expect("bounded resident accounting")
        };
        assert_eq!(accounted(257) - accounted(0), 12 * 257);
    }
}

impl<S: SimulationEventSinkV1> Engine<'_, S> {
    fn fail(&self, kind: SimulationExecutionErrorKindV1) -> SimulationExecutionErrorV1 {
        SimulationExecutionErrorV1 {
            invocation: self.invocation,
            site: None,
            kind,
            observation_failure: None,
        }
    }

    fn at(
        &self,
        site: CompactSite,
        kind: SimulationExecutionErrorKindV1,
    ) -> SimulationExecutionErrorV1 {
        SimulationExecutionErrorV1 {
            invocation: self.invocation,
            site: Some(self.materialize_site(site)),
            kind,
            observation_failure: None,
        }
    }

    fn memory_error_at(
        &mut self,
        site: CompactSite,
        pointer: &PointerValue,
        kind: SimulationExecutionErrorKindV1,
    ) -> SimulationExecutionErrorV1 {
        if let SimulationExecutionErrorKindV1::OutOfBounds {
            bytes,
            allocation_bytes,
            ..
        } = &kind
        {
            self.debug_sink
                .terminal_out_of_bounds_v2(out_of_bounds_detail_v2(
                    pointer,
                    *bytes,
                    *allocation_bytes,
                ));
        }
        self.at(site, kind)
    }

    fn materialize_site(&self, site: CompactSite) -> SimulationSiteV1 {
        let module_index = self.function_module_indices[site.function];
        SimulationSiteV1 {
            function: self.module.functions[module_index].id.clone(),
            block: site.block,
            operation: site.operation,
        }
    }

    fn materialize_event_site(&self, site: CompactSite) -> SimulationEventSiteV1 {
        SimulationEventSiteV1 {
            function_ordinal: self.function_module_indices[site.function],
            block: site.block,
            operation: site.operation,
        }
    }

    fn debug_checkpoint(
        &mut self,
        frames: &[RuntimeFrame<'_>],
        site: CompactSite,
        phase: SimulationDebugCheckpointPhaseV1,
    ) {
        if self.debug_delivery_stopped {
            return;
        }
        if let Some(capture) = self.physical_debug.as_mut() {
            let kind = capture.checkpoint(
                frames,
                &self.function_module_indices,
                &self.memory,
                self.debug_capture,
                phase,
            );
            if let Some(kind) = kind {
                self.deliver_debug_with_frames(site, Some(frames), kind);
            } else {
                self.debug_delivery_stopped = true;
            }
            return;
        }
        let stack = capture_debug_stack(frames, &self.function_module_indices, self.debug_capture);
        let memory = capture_debug_memory(&self.memory, self.debug_capture);
        self.deliver_debug_with_frames(
            site,
            Some(frames),
            SimulationDebugRecordKindV1::Checkpoint {
                phase,
                stack,
                memory,
            },
        );
    }

    fn debug_memory(
        &mut self,
        site: CompactSite,
        access: SimulationDebugMemoryAccessV1,
        pointer: &PointerValue,
        byte_len: usize,
        value: ScalarBitsV1,
    ) {
        self.deliver_debug(
            site,
            SimulationDebugRecordKindV1::Memory {
                access,
                allocation: pointer.allocation,
                byte_offset: pointer.byte_offset,
                byte_len,
                address_space: pointer.address_space,
                value: SimulationDebugValueV1::Scalar(value),
            },
        );
    }

    fn debug_barrier(
        &mut self,
        site: CompactSite,
        action: SimulationDebugBarrierActionV1,
        phase: u64,
        participants: u32,
    ) {
        self.deliver_debug(
            site,
            SimulationDebugRecordKindV1::WorkgroupBarrier {
                action,
                phase,
                participants,
            },
        );
    }

    fn debug_fence(&mut self, site: CompactSite, fence: &Fence) {
        self.deliver_debug(
            site,
            SimulationDebugRecordKindV1::Fence {
                memory_scope: fence.memory_scope,
                ordering: fence.semantics.ordering,
                address_space_mask: address_space_mask(&fence.semantics.address_spaces),
            },
        );
    }

    fn deliver_debug(&mut self, site: CompactSite, kind: SimulationDebugRecordKindV1) {
        self.deliver_debug_with_frames(site, None, kind);
    }

    fn deliver_debug_with_frames(
        &mut self,
        site: CompactSite,
        frames: Option<&[RuntimeFrame<'_>]>,
        kind: SimulationDebugRecordKindV1,
    ) {
        if self.debug_delivery_stopped {
            return;
        }
        let (Some(invocation), Some(operation)) = (self.invocation, site.operation) else {
            return;
        };
        let module_index = self.function_module_indices[site.function];
        let record = SimulationDebugRecordV1 {
            ordinal: self.debug_records,
            schedule: SimulationDebugScheduleV1 {
                identity: self.schedule_identity,
                decision_ordinal: self.schedule_decision,
            },
            invocation,
            site: SimulationDebugSiteV1 {
                function_ordinal: module_index,
                block: site.block,
                operation,
            },
            kind,
        };
        let control = if let Some(capture) = self.physical_debug.as_mut() {
            capture.record(record)
        } else if self.debug_observation_requested {
            let origin = self.operation_origin_context(&record);
            let watermark = self.allocation_lifecycle_watermark_v1();
            let source = debug_frames::RuntimeFrameOrigins {
                frames: frames.unwrap_or(&[]),
                function_module_indices: &self.function_module_indices,
                invocation,
            };
            let frame_context = debug_frames::context(
                &record,
                &source,
                self.debug_frames_requested,
                self.debug_identity_failed,
            );
            self.debug_sink.record_with_observation_context_v1(
                record,
                crate::SimulationDebugObservationContextV1::new(origin, frame_context, watermark),
            )
        } else if self.debug_origin_requested {
            let origin = self.operation_origin_context(&record);
            self.debug_sink
                .record_with_operation_origin_v1(record, origin)
        } else {
            self.debug_sink.record(record)
        };
        match control {
            SimulationDebugSinkControlV1::Continue => {
                if let Some(next) = self.debug_records.checked_add(1) {
                    self.debug_records = next;
                } else {
                    self.debug_delivery_stopped = true;
                }
            }
            SimulationDebugSinkControlV1::Stop => {
                self.debug_records = self.debug_records.saturating_add(1);
                self.debug_delivery_stopped = true;
            }
            SimulationDebugSinkControlV1::DropAndStop => {
                self.debug_delivery_stopped = true;
            }
        }
    }

    fn step(&mut self, site: &CompactSite) -> Result<(), SimulationExecutionErrorV1> {
        self.charge_steps(site, 1)
    }

    fn charge_steps(
        &mut self,
        site: &CompactSite,
        count: usize,
    ) -> Result<(), SimulationExecutionErrorV1> {
        let count = u64::try_from(count).unwrap_or(u64::MAX);
        let Some(steps) = self.steps.checked_add(count) else {
            return Err(self.at(
                *site,
                SimulationExecutionErrorKindV1::StepLimit {
                    limit: self.limits.max_steps,
                },
            ));
        };
        if steps > self.limits.max_steps {
            return Err(self.at(
                *site,
                SimulationExecutionErrorKindV1::StepLimit {
                    limit: self.limits.max_steps,
                },
            ));
        }
        self.steps = steps;
        Ok(())
    }

    fn event(
        &mut self,
        site: &CompactSite,
        kind: SimulationEventKindV1,
    ) -> Result<(), SimulationExecutionErrorV1> {
        if self.policy == EventPolicyV1::Disabled || self.event_delivery_stopped {
            return Ok(());
        }
        let site = self.materialize_event_site(*site);
        let function = &self.module.functions[site.function_ordinal].id;
        deliver_event(
            self.policy,
            self.limits.max_events,
            &mut self.events,
            &mut self.reserved_event_closures,
            &mut self.event_delivery_stopped,
            self.invocation,
            self.sink,
            function,
            &site,
            kind,
            false,
        )
    }

    fn call_event(
        &mut self,
        site: &CompactSite,
        callee: usize,
    ) -> Result<(), SimulationExecutionErrorV1> {
        if self.policy == EventPolicyV1::Disabled || self.event_delivery_stopped {
            return Ok(());
        }
        self.event(
            site,
            SimulationEventKindV1::Call {
                callee_ordinal: self.function_module_indices[callee],
            },
        )
    }

    fn begin_lifecycle(
        &mut self,
        site: &CompactSite,
        kind: SimulationEventKindV1,
    ) -> Result<(), SimulationExecutionErrorV1> {
        let reserved = self.reserve_event_closure(site)?;
        self.emit_reserved_begin(site, kind, reserved)
    }

    fn reserve_event_closure(
        &mut self,
        site: &CompactSite,
    ) -> Result<bool, SimulationExecutionErrorV1> {
        if self.policy == EventPolicyV1::Disabled || self.event_delivery_stopped {
            return Ok(false);
        }
        let required = self
            .events
            .checked_add(self.reserved_event_closures)
            .and_then(|events| events.checked_add(2))
            .ok_or_else(|| {
                self.at(
                    *site,
                    SimulationExecutionErrorKindV1::EventLimit {
                        limit: self.limits.max_events,
                    },
                )
            })?;
        if required > self.limits.max_events {
            return Err(self.at(
                *site,
                SimulationExecutionErrorKindV1::EventLimit {
                    limit: self.limits.max_events,
                },
            ));
        }
        self.reserved_event_closures += 1;
        Ok(true)
    }

    fn emit_reserved_begin(
        &mut self,
        site: &CompactSite,
        kind: SimulationEventKindV1,
        reserved: bool,
    ) -> Result<(), SimulationExecutionErrorV1> {
        let result = self.event(site, kind);
        if result.is_err() && reserved {
            self.reserved_event_closures = self.reserved_event_closures.saturating_sub(1);
        }
        result
    }

    fn end_lifecycle(
        &mut self,
        site: &CompactSite,
        kind: SimulationEventKindV1,
    ) -> Result<(), SimulationExecutionErrorV1> {
        if self.policy == EventPolicyV1::Disabled || self.event_delivery_stopped {
            return Ok(());
        }
        let site = self.materialize_event_site(*site);
        let function = &self.module.functions[site.function_ordinal].id;
        deliver_event(
            self.policy,
            self.limits.max_events,
            &mut self.events,
            &mut self.reserved_event_closures,
            &mut self.event_delivery_stopped,
            self.invocation,
            self.sink,
            function,
            &site,
            kind,
            true,
        )
    }

    fn cancel_event_closure(&mut self, reserved: bool) {
        if reserved {
            self.reserved_event_closures = self.reserved_event_closures.saturating_sub(1);
        }
    }

    fn observe_and_commit_store(
        &mut self,
        site: &CompactSite,
        pointer: &PointerValue,
        value: ScalarBitsV1,
        width: usize,
    ) -> Result<(), SimulationExecutionErrorV1> {
        let observed_site = (self.policy == EventPolicyV1::Enabled && !self.event_delivery_stopped)
            .then(|| self.materialize_event_site(*site));
        let prepared = match self.memory.prepare_store(pointer, width) {
            Ok(prepared) => prepared,
            Err(kind) => {
                return Err(SimulationExecutionErrorV1 {
                    invocation: self.invocation,
                    site: Some(self.materialize_site(*site)),
                    kind,
                    observation_failure: None,
                });
            }
        };
        if let Some(observed_site) = observed_site {
            let function = &self.module.functions[observed_site.function_ordinal].id;
            deliver_event(
                self.policy,
                self.limits.max_events,
                &mut self.events,
                &mut self.reserved_event_closures,
                &mut self.event_delivery_stopped,
                self.invocation,
                self.sink,
                function,
                &observed_site,
                SimulationEventKindV1::MemoryWrite {
                    allocation: pointer.allocation,
                    offset: pointer.byte_offset,
                    bytes: width,
                },
                false,
            )?;
        }
        prepared.commit(value);
        if pointer.address_space == AddressSpace::Workgroup {
            let invocation = self.invocation.ok_or_else(|| {
                self.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("workgroup store invocation"),
                )
            })?;
            self.memory
                .mark_workgroup_store(pointer, width, invocation)
                .map_err(|kind| self.at(*site, kind))?;
        }
        self.debug_memory(
            *site,
            SimulationDebugMemoryAccessV1::WriteCommitted,
            pointer,
            width,
            value,
        );
        Ok(())
    }

    fn workgroup_pointer(
        &mut self,
        site: CompactSite,
        memory: &fe2o3_kernel_ir::WorkgroupMemory,
    ) -> Result<PointerValue, SimulationExecutionErrorV1> {
        let Type::Scalar(element) = memory.element else {
            return Err(self.at(
                site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "preflighted scalar workgroup memory",
                ),
            ));
        };
        let element_bytes = self.target.scalar_bytes(element).ok_or_else(|| {
            self.at(
                site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "preflighted workgroup memory element",
                ),
            )
        })?;
        let bytes = match memory.extent {
            WorkgroupMemoryExtent::Static(elements) => usize::try_from(elements)
                .ok()
                .and_then(|elements| elements.checked_mul(element_bytes)),
            WorkgroupMemoryExtent::Dynamic => self
                .dynamic_workgroup_memory
                .and_then(|dynamic| usize::try_from(dynamic.byte_extent()).ok()),
            WorkgroupMemoryExtent::DynamicAtLeast(_) => None,
        }
        .ok_or_else(|| {
            self.at(
                site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "preflighted workgroup memory extent",
                ),
            )
        })?;
        if let Some(existing) = self
            .workgroup_allocations
            .iter()
            .find(|allocation| allocation.site == site)
        {
            if existing.element != element || existing.bytes != bytes {
                return Err(self.at(
                    site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "workgroup allocation site shape changed",
                    ),
                ));
            }
            return Ok(PointerValue {
                exposed_generic: false,
                allocation: existing.id,
                byte_offset: 0,
                element,
                address_space: AddressSpace::Workgroup,
                access: AccessMode::ReadWrite,
                lower_bound: 0,
                upper_bound: bytes,
                abi_argument_ordinal: NO_ABI_ARGUMENT_V1,
            });
        }

        self.memory
            .validate_allocation(bytes, self.limits)
            .map_err(|kind| self.at(site, kind))?;
        if self.workgroup_allocations.len() == self.workgroup_allocations.capacity()
            && self.workgroup_allocations.try_reserve_exact(1).is_err()
        {
            return Err(self.at(site, SimulationExecutionErrorKindV1::AllocationFailure));
        }
        let allocation_bytes = try_filled(bytes, 0_u8).map_err(|kind| self.at(site, kind))?;
        let initialized = try_filled(bytes, false).map_err(|kind| self.at(site, kind))?;
        let creation = self.allocation_creation_v1(site, AddressSpace::Workgroup)?;
        let reserved = self.reserve_event_closure(&site)?;
        let commit = match self.memory.allocate(
            (
                AddressSpace::Workgroup,
                AccessMode::ReadWrite,
                memory.alignment,
            ),
            (allocation_bytes, initialized),
            creation,
            self.limits,
        ) {
            Ok(commit) => commit,
            Err(kind) => {
                self.cancel_event_closure(reserved);
                return Err(self.at(site, kind));
            }
        };
        let id = commit.id;
        self.workgroup_allocations.push(WorkgroupAllocation {
            site,
            id,
            element,
            bytes,
            lifecycle_observed: reserved,
        });
        self.deliver_allocation_lifecycle_v1(commit.transition);
        self.emit_reserved_begin(
            &site,
            SimulationEventKindV1::AllocationCreated {
                allocation: id,
                address_space: AddressSpace::Workgroup,
                bytes,
            },
            reserved,
        )?;
        Ok(PointerValue {
            exposed_generic: false,
            allocation: id,
            byte_offset: 0,
            element,
            address_space: AddressSpace::Workgroup,
            access: AccessMode::ReadWrite,
            lower_bound: 0,
            upper_bound: bytes,
            abi_argument_ordinal: NO_ABI_ARGUMENT_V1,
        })
    }

    fn publish_workgroup(&mut self) {
        self.memory.publish_workgroup();
    }

    fn publish_global_happens_before(&mut self, barrier: &WorkgroupBarrier) {
        if barrier
            .semantics
            .address_spaces
            .contains(&AddressSpace::Global)
            && matches!(
                barrier.semantics.ordering,
                MemoryOrdering::AcquireRelease | MemoryOrdering::SequentiallyConsistent
            )
        {
            self.workgroup_happens_before_epoch =
                self.workgroup_happens_before_epoch.saturating_add(1);
        }
    }

    fn release_workgroup_allocations(
        &mut self,
        primary: &mut Option<&mut SimulationExecutionErrorV1>,
    ) -> Result<(), SimulationExecutionErrorV1> {
        while let Some(allocation) = self.workgroup_allocations.pop() {
            let released = self
                .memory
                .release_one(allocation.id)
                .map_err(|kind| self.at(allocation.site, kind))?;
            if !released.removed {
                return Err(self.at(
                    allocation.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "workgroup allocation remained live until release",
                    ),
                ));
            }
            self.deliver_allocation_lifecycle_v1(released.transition);
            if allocation.lifecycle_observed
                && let Err(secondary) = self.end_lifecycle(
                    &allocation.site,
                    SimulationEventKindV1::AllocationReleased {
                        allocation: allocation.id,
                    },
                )
            {
                if let Some(primary) = primary.as_deref_mut() {
                    attach_observation_failure(primary, secondary);
                } else {
                    return Err(secondary);
                }
            }
        }
        Ok(())
    }

    fn record_access(
        &mut self,
        site: &CompactSite,
        allocation: u64,
        offset: usize,
        bytes: usize,
        write: bool,
        atomic: Option<AtomicAccess>,
    ) -> Result<(), SimulationExecutionErrorV1> {
        let invocation = self.invocation.ok_or_else(|| {
            self.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant("memory access invocation"),
            )
        })?;
        // This invocation and every retained record belong to the same Engine.
        let stored_invocation = AccessInvocationV1::store(invocation);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| self.at(*site, SimulationExecutionErrorKindV1::PointerOffsetOverflow))?;
        let compact_site = *site;
        for byte in offset..end {
            let key = (allocation, byte);
            let previous = self.accesses.get(&key).copied();
            let mut frontier = previous.unwrap_or_default();
            let candidates = if write {
                [
                    frontier.write,
                    frontier.displaced_write,
                    frontier.read,
                    frontier.displaced_read,
                ]
            } else {
                [frontier.write, frontier.displaced_write, None, None]
            };
            let mut conflicting = false;
            let mut racing = false;
            for earlier_access in candidates
                .into_iter()
                .flatten()
                .filter(|earlier| earlier.invocation != stored_invocation)
            {
                conflicting = true;
                let earlier_invocation = earlier_access.invocation.restore(invocation);
                let conflict_evidence = SimulationMemoryConflictV1 {
                    allocation,
                    offset: byte,
                    earlier: earlier_invocation,
                    later: invocation,
                    earlier_site: self.materialize_site(earlier_access.site),
                    later_site: self.materialize_site(*site),
                };
                if self.first_conflict.is_none() {
                    self.first_conflict = Some(conflict_evidence.clone());
                }
                let atomic_relation = earlier_access.atomic.zip(atomic).map(|(earlier, later)| {
                    earlier.relation(earlier_invocation, later, invocation)
                });
                let ordered = if atomic_relation == Some(AtomicRelation::Serialized) {
                    Some(SimulationHappensBeforeReasonV1::AtomicSerialization)
                } else if earlier_invocation.workgroup == invocation.workgroup
                    && earlier_access.happens_before_epoch < self.workgroup_happens_before_epoch
                {
                    Some(SimulationHappensBeforeReasonV1::GlobalWorkgroupBarrier)
                } else {
                    None
                };
                let unproved_atomic =
                    ordered.is_none() && atomic_relation == Some(AtomicRelation::Unproved);
                self.unmodeled_atomic_serialization |= unproved_atomic;
                if self.race_trackers.is_empty() {
                    self.race_trackers.try_reserve_exact(1).map_err(|_| {
                        self.at(*site, SimulationExecutionErrorKindV1::AllocationFailure)
                    })?;
                    self.race_trackers.push(RaceTracker {
                        racing_bytes: 0,
                        first_race: None,
                        first_ordered_conflict: None,
                    });
                }
                let missing_tracker = self.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "recorded memory conflict race tracker",
                    ),
                );
                let race = self.race_trackers.first_mut().ok_or(missing_tracker)?;
                if let Some(reason) = ordered {
                    if race.first_ordered_conflict.is_none() {
                        race.first_ordered_conflict = Some(SimulationOrderedMemoryConflictV1 {
                            conflict: conflict_evidence,
                            reason,
                        });
                    }
                } else if !unproved_atomic {
                    racing = true;
                    if race.first_race.is_none() {
                        race.first_race = Some(SimulationDataRaceV1 {
                            conflict: conflict_evidence,
                            earlier_atomic: earlier_access.atomic.is_some(),
                            later_atomic: atomic.is_some(),
                        });
                    }
                }
            }
            if conflicting && !frontier.conflicted {
                self.conflicting_bytes = self.conflicting_bytes.saturating_add(1);
            }
            if racing && !frontier.raced {
                let missing_tracker = self.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("recorded data race tracker"),
                );
                let race = self.race_trackers.first_mut().ok_or(missing_tracker)?;
                race.racing_bytes = race.racing_bytes.saturating_add(1);
            }
            frontier.conflicted |= conflicting;
            frontier.raced |= racing;
            if frontier.raced {
                // A proven race fixes this byte's race classification and
                // unique-byte count even if older representatives were lost.
                frontier.incomplete = false;
            } else if (frontier.lost_write
                && !frontier
                    .lost_writes_atomic
                    .zip(atomic)
                    .is_some_and(|(history, access)| history.serializes_with(access, invocation)))
                || (write
                    && frontier.lost_read
                    && !frontier
                        .lost_reads_atomic
                        .zip(atomic)
                        .is_some_and(|(history, access)| {
                            history.serializes_with(access, invocation)
                        }))
            {
                // Evicted representatives are complete only when every exact
                // interval and both scope directions are still covered. Unknown
                // or mixed atomic history must not become race-free evidence.
                frontier.incomplete = true;
            }

            if previous.is_some() || self.accesses.len() < self.limits.max_memory_access_records {
                let slot = if write {
                    if let Some(earlier) = frontier.write
                        && (earlier.invocation != stored_invocation
                            || earlier.happens_before_epoch != self.workgroup_happens_before_epoch)
                        && !frontier.raced
                    {
                        if frontier.displaced_write.is_some_and(|displaced| {
                            displaced.invocation != earlier.invocation
                                || displaced.happens_before_epoch != earlier.happens_before_epoch
                        }) {
                            let displaced = frontier
                                .displaced_write
                                .expect("checked displaced write frontier");
                            let lost = displaced.atomic.and_then(|access| {
                                AtomicHistory::new(access, displaced.invocation.restore(invocation))
                            });
                            frontier.lost_writes_atomic = if frontier.lost_write {
                                frontier
                                    .lost_writes_atomic
                                    .zip(lost)
                                    .and_then(|(history, next)| history.include(next))
                            } else {
                                lost
                            };
                            frontier.lost_write = true;
                        }
                        // Keep a displaced epoch even for the same invocation:
                        // a workgroup barrier does not order another group.
                        // Ordinary replacement still compares with its atomic predecessor.
                        frontier.displaced_write = Some(earlier);
                    }
                    &mut frontier.write
                } else {
                    if let Some(earlier) = frontier.read
                        && (earlier.invocation != stored_invocation
                            || earlier.happens_before_epoch != self.workgroup_happens_before_epoch)
                        && !frontier.raced
                    {
                        if frontier.displaced_read.is_some_and(|displaced| {
                            displaced.invocation != earlier.invocation
                                || displaced.happens_before_epoch != earlier.happens_before_epoch
                        }) {
                            let displaced = frontier
                                .displaced_read
                                .expect("checked displaced read frontier");
                            let lost = displaced.atomic.and_then(|access| {
                                AtomicHistory::new(access, displaced.invocation.restore(invocation))
                            });
                            frontier.lost_reads_atomic = if frontier.lost_read {
                                frontier
                                    .lost_reads_atomic
                                    .zip(lost)
                                    .and_then(|(history, next)| history.include(next))
                            } else {
                                lost
                            };
                            frontier.lost_read = true;
                        }
                        frontier.displaced_read = Some(earlier);
                    }
                    &mut frontier.read
                };
                *slot = Some(match *slot {
                    Some(earlier)
                        if earlier.invocation == stored_invocation
                            && earlier.happens_before_epoch
                                == self.workgroup_happens_before_epoch =>
                    {
                        // Keep a concrete narrow/ordinary site when its access
                        // replaces broader evidence in this invocation/epoch.
                        let site = if earlier.atomic.is_some()
                            && (atomic.is_none()
                                || atomic.zip(earlier.atomic).is_some_and(|(next, previous)| {
                                    next.is_narrower_than(previous)
                                })) {
                            compact_site
                        } else {
                            earlier.site
                        };
                        LastAccess {
                            invocation: stored_invocation,
                            site,
                            atomic: earlier
                                .atomic
                                .zip(atomic)
                                .map(|(old, next)| old.merge(next)),
                            happens_before_epoch: earlier.happens_before_epoch,
                        }
                    }
                    _ => LastAccess {
                        invocation: stored_invocation,
                        site: compact_site,
                        atomic,
                        happens_before_epoch: self.workgroup_happens_before_epoch,
                    },
                });
                self.accesses.insert(key, frontier);
            } else {
                self.conflict_incomplete = true;
            }
        }
        Ok(())
    }

    fn conflict_assessment(&self) -> SimulationConflictAssessmentV1 {
        let access_frontier_incomplete = self.accesses.values().any(|frontier| frontier.incomplete);
        if self.conflict_incomplete || access_frontier_incomplete {
            SimulationConflictAssessmentV1::Incomplete {
                conflicting_bytes: self.conflicting_bytes,
                first: self.first_conflict.clone(),
                access_record_limit_reached: self.conflict_incomplete,
                access_frontier_incomplete,
                record_limit: self.limits.max_memory_access_records,
            }
        } else if let Some(first) = self.first_conflict.clone() {
            SimulationConflictAssessmentV1::ConflictsObserved {
                conflicting_bytes: self.conflicting_bytes,
                first,
            }
        } else {
            SimulationConflictAssessmentV1::NoConflictsObserved
        }
    }

    fn race_assessment(&self) -> SimulationRaceAssessmentV1 {
        let race = self.race_trackers.first();
        let access_frontier_incomplete = self.accesses.values().any(|frontier| frontier.incomplete);
        let synchronization_incomplete = self.unmodeled_atomic_serialization
            || (self.unmodeled_atomic_or_fence_happens_before
                && race.is_some_and(|race| race.first_race.is_some()));
        if self.conflict_incomplete || access_frontier_incomplete || synchronization_incomplete {
            return SimulationRaceAssessmentV1::Incomplete {
                racing_bytes: race.map_or(0, |race| race.racing_bytes),
                first: race.and_then(|race| race.first_race.clone()),
                first_ordered_conflict: race.and_then(|race| race.first_ordered_conflict.clone()),
                access_record_limit_reached: self.conflict_incomplete,
                access_frontier_incomplete,
                atomic_or_fence_happens_before_unmodeled: synchronization_incomplete,
                record_limit: self.limits.max_memory_access_records,
            };
        }
        let Some(race) = race else {
            return SimulationRaceAssessmentV1::NoRacesObserved {
                first_ordered_conflict: None,
            };
        };
        if let Some(first) = race.first_race.clone() {
            SimulationRaceAssessmentV1::RacesObserved {
                racing_bytes: race.racing_bytes,
                first,
                first_ordered_conflict: race.first_ordered_conflict.clone(),
            }
        } else {
            SimulationRaceAssessmentV1::NoRacesObserved {
                first_ordered_conflict: race.first_ordered_conflict.clone(),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deliver_event<S: SimulationEventSinkV1>(
    policy: EventPolicyV1,
    max_events: u64,
    events: &mut u64,
    reserved_closures: &mut u64,
    delivery_stopped: &mut bool,
    invocation: Option<SimulationInvocationV1>,
    sink: &mut S,
    function: &FunctionId,
    site: &SimulationEventSiteV1,
    kind: SimulationEventKindV1,
    closure: bool,
) -> Result<(), SimulationExecutionErrorV1> {
    if policy == EventPolicyV1::Disabled || *delivery_stopped {
        return Ok(());
    }
    if closure {
        if *reserved_closures == 0 {
            return Err(SimulationExecutionErrorV1 {
                invocation,
                site: Some(owned_event_site(function, site)),
                kind: SimulationExecutionErrorKindV1::InternalInvariant(
                    "lifecycle closure event credit",
                ),
                observation_failure: None,
            });
        }
        *reserved_closures -= 1;
    } else if events
        .checked_add(*reserved_closures)
        .is_none_or(|required| required >= max_events)
    {
        return Err(SimulationExecutionErrorV1 {
            invocation,
            site: Some(owned_event_site(function, site)),
            kind: SimulationExecutionErrorKindV1::EventLimit { limit: max_events },
            observation_failure: None,
        });
    }
    let invocation = invocation.ok_or_else(|| SimulationExecutionErrorV1 {
        invocation: None,
        site: Some(owned_event_site(function, site)),
        kind: SimulationExecutionErrorKindV1::InternalInvariant("event invocation"),
        observation_failure: None,
    })?;
    let control = match sink.record_controlled(&SimulationEventV1 {
        invocation,
        site: *site,
        kind,
    }) {
        Ok(control) => control,
        Err(error) => {
            *delivery_stopped = true;
            *reserved_closures = 0;
            return Err(SimulationExecutionErrorV1 {
                invocation: Some(invocation),
                site: Some(owned_event_site(function, site)),
                kind: SimulationExecutionErrorKindV1::EventSinkFailure(error),
                observation_failure: None,
            });
        }
    };
    match control {
        SimulationEventSinkControlV1::Continue => *events += 1,
        SimulationEventSinkControlV1::Stop => {
            *events += 1;
            *delivery_stopped = true;
            *reserved_closures = 0;
        }
        SimulationEventSinkControlV1::DropAndStop => {
            *delivery_stopped = true;
            *reserved_closures = 0;
        }
    }
    Ok(())
}

fn owned_event_site(function: &FunctionId, site: &SimulationEventSiteV1) -> SimulationSiteV1 {
    SimulationSiteV1 {
        function: function.clone(),
        block: site.block,
        operation: site.operation,
    }
}

type ExecutionIndices<'a> = (
    HashMap<&'a FunctionId, usize>,
    Vec<usize>,
    Vec<HashMap<BlockId, usize>>,
    Vec<usize>,
    Vec<Vec<Vec<CallTarget>>>,
    Vec<Vec<SwitchLookup>>,
);

fn execution_indices_resident_bytes(indices: &ExecutionIndices<'_>) -> Option<usize> {
    let (functions, module_indices, blocks, ssa_values, call_targets, switch_targets) = indices;
    let mut resident = ResidentLedger::new(0);
    resident.add_bytes(hash_map_capacity_bytes::<&FunctionId, usize>(
        functions.capacity(),
    )?)?;
    resident.add_vec::<usize>(module_indices.capacity())?;
    resident.add_vec::<HashMap<BlockId, usize>>(blocks.capacity())?;
    for function_blocks in blocks {
        resident.add_bytes(hash_map_capacity_bytes::<BlockId, usize>(
            function_blocks.capacity(),
        )?)?;
    }
    resident.add_vec::<usize>(ssa_values.capacity())?;
    resident.add_vec::<Vec<Vec<CallTarget>>>(call_targets.capacity())?;
    for function_calls in call_targets {
        resident.add_vec::<Vec<CallTarget>>(function_calls.capacity())?;
        for block_calls in function_calls {
            resident.add_vec::<CallTarget>(block_calls.capacity())?;
        }
    }
    resident.add_vec::<Vec<SwitchLookup>>(switch_targets.capacity())?;
    for function_switches in switch_targets {
        resident.add_vec::<SwitchLookup>(function_switches.capacity())?;
        for lookup in function_switches {
            match lookup {
                SwitchLookup::None => {}
                SwitchLookup::Index(map) => {
                    resident.add_bytes(hash_map_capacity_bytes::<u128, usize>(map.capacity())?)?
                }
                SwitchLookup::Integer(map) => {
                    resident.add_bytes(hash_map_capacity_bytes::<(ScalarType, u128), usize>(
                        map.capacity(),
                    )?)?
                }
            }
        }
    }
    Some(resident.bytes())
}

pub(crate) fn preflight_execution_indices_resident_bytes(
    module: &Module,
    reachable: &[usize],
    target: SimulationTargetV1,
) -> Result<Option<usize>, SimulationPreflightErrorV1> {
    let indices = build_execution_indices(module, reachable, target)
        .map_err(|_| SimulationPreflightErrorV1::AllocationFailure)?;
    Ok(execution_indices_resident_bytes(&indices))
}

fn build_execution_indices<'a>(
    module: &'a Module,
    reachable: &[usize],
    target: SimulationTargetV1,
) -> Result<ExecutionIndices<'a>, SimulationExecutionErrorV1> {
    let mut functions = HashMap::new();
    functions
        .try_reserve(reachable.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    let mut module_indices = Vec::new();
    module_indices
        .try_reserve_exact(reachable.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(reachable.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    let mut ssa_values = Vec::new();
    ssa_values
        .try_reserve_exact(reachable.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    for (function_index, module_index) in reachable.iter().copied().enumerate() {
        let function = &module.functions[module_index];
        functions.insert(&function.id, function_index);
        module_indices.push(module_index);
    }
    let mut call_targets = Vec::new();
    call_targets
        .try_reserve_exact(reachable.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    let mut switch_targets = Vec::new();
    switch_targets
        .try_reserve_exact(reachable.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    for module_index in reachable.iter().copied() {
        let function = &module.functions[module_index];
        let block_count = function.body.as_ref().map_or(0, |body| body.blocks.len());
        let mut function_blocks = HashMap::new();
        function_blocks
            .try_reserve(block_count)
            .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
        if let Some(body) = &function.body {
            for (block_index, block) in body.blocks.iter().enumerate() {
                function_blocks.insert(block.id, block_index);
            }
        }
        blocks.push(function_blocks);
        ssa_values.push(function_ssa_definition_count(function).unwrap_or(0));
        let mut function_calls = Vec::new();
        function_calls
            .try_reserve_exact(block_count)
            .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
        if let Some(body) = &function.body {
            for block in &body.blocks {
                let mut block_calls = Vec::new();
                block_calls
                    .try_reserve_exact(block.operations.len())
                    .map_err(|_| {
                        top_level_error(SimulationExecutionErrorKindV1::AllocationFailure)
                    })?;
                for operation in &block.operations {
                    let target = match &operation.kind {
                        OperationKind::Call { callee, arguments } => {
                            if let Some(operation) =
                                crate::soft_float::operation_for_call_v1(callee, arguments)
                            {
                                CallTarget::Float(operation)
                            } else if matches!(
                                AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments),
                                Some(AmdGpuDiagnosticOperation::Trap)
                            ) {
                                CallTarget::Trap
                            } else {
                                CallTarget::Internal(*functions.get(callee).ok_or_else(|| {
                                    top_level_error(
                                        SimulationExecutionErrorKindV1::MissingFunction(
                                            callee.clone(),
                                        ),
                                    )
                                })?)
                            }
                        }
                        _ => CallTarget::NotCall,
                    };
                    block_calls.push(target);
                }
                function_calls.push(block_calls);
            }
        }
        call_targets.push(function_calls);
        let mut function_switches = Vec::new();
        function_switches
            .try_reserve_exact(block_count)
            .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
        if let Some(body) = &function.body {
            for block in &body.blocks {
                let lookup = match block.terminator.as_ref() {
                    Some(Terminator::Switch { cases, .. }) => {
                        let mut lookup = HashMap::new();
                        lookup.try_reserve(cases.len()).map_err(|_| {
                            top_level_error(SimulationExecutionErrorKindV1::AllocationFailure)
                        })?;
                        for (case_index, case) in cases.iter().enumerate() {
                            lookup.insert(u128::from(case.value), case_index);
                        }
                        SwitchLookup::Index(lookup)
                    }
                    Some(Terminator::IntegerSwitch { cases, .. }) => {
                        let mut lookup = HashMap::new();
                        lookup.try_reserve(cases.len()).map_err(|_| {
                            top_level_error(SimulationExecutionErrorKindV1::AllocationFailure)
                        })?;
                        for (case_index, case) in cases.iter().enumerate() {
                            let value =
                                constant_scalar(&case.value, target).map_err(top_level_error)?;
                            lookup.insert((value.ty(), value.bits()), case_index);
                        }
                        SwitchLookup::Integer(lookup)
                    }
                    _ => SwitchLookup::None,
                };
                function_switches.push(lookup);
            }
        }
        switch_targets.push(function_switches);
    }
    Ok((
        functions,
        module_indices,
        blocks,
        ssa_values,
        call_targets,
        switch_targets,
    ))
}

struct ExecutionConfiguration<'a> {
    target: SimulationTargetV1,
    limits: SimulationLimitsV1,
    policy: EventPolicyV1,
    plan: SimulationPlanV1,
    debug_capture: SimulationDebugCaptureLimitsV1,
    schedule: Option<ExecutionScheduleRequestV1<'a>>,
    resident_offset: usize,
    allocation_reuse: Option<SimulationAllocationReuseV1>,
    matrix_bf16_numerics: crate::MatrixBf16NumericsV1,
}

// Keep one-time layout construction temporaries off the long-lived scheduler
// stack. The original plan, allocation order and errors are unchanged.
#[inline(never)]
fn prepare_frame_value_layouts<'a>(
    admitted: &'a AdmittedSimulationModuleV1,
    function_module_indices: &[usize],
) -> Result<(FrameValuePlan, Vec<ValueLayout<'a>>), SimulationExecutionErrorV1> {
    let legacy_values = uses_legacy_frame_values(admitted.identity.wire_version());
    let value_plan = frame_value_plan(&admitted.module, function_module_indices, legacy_values)
        .ok_or_else(|| {
            top_level_error(SimulationExecutionErrorKindV1::InternalInvariant(
                "preflighted typed SSA storage plan",
            ))
        })?;
    let value_layouts =
        values_v1::build_layouts(&admitted.module, function_module_indices, value_plan)
            .map_err(top_level_error)?;
    Ok((value_plan, value_layouts))
}

// Preparation errors and allocation temporaries must not remain in the live
// scheduler stack. These fields are immediately moved into the original bindings.
struct PreparedExecutionStorage<'a> {
    function_module_indices: Vec<usize>,
    block_indices: Vec<HashMap<BlockId, usize>>,
    function_ssa_values: Vec<usize>,
    call_targets: Vec<Vec<Vec<CallTarget>>>,
    switch_targets: Vec<Vec<SwitchLookup>>,
    entry_index: usize,
    entry: &'a Function,
    value_plan: FrameValuePlan,
    value_layouts: Vec<ValueLayout<'a>>,
    memory: Memory,
    accesses: HashMap<(u64, usize), AccessFrontier>,
    workgroup_allocations: Vec<WorkgroupAllocation>,
}

#[inline(never)]
fn prepare_execution_storage<'a>(
    admitted: &'a AdmittedSimulationModuleV1,
    request: &SimulationRequestV1,
    plan: &SimulationPlanV1,
    target: SimulationTargetV1,
    limits: SimulationLimitsV1,
    allocation_reuse: Option<SimulationAllocationReuseV1>,
) -> Result<PreparedExecutionStorage<'a>, SimulationExecutionErrorV1> {
    let indices =
        build_execution_indices(&admitted.module, &plan.reachable_function_indices, target)?;
    let actual_index_resident_bytes =
        execution_indices_resident_bytes(&indices).ok_or_else(|| {
            top_level_error(SimulationExecutionErrorKindV1::InternalInvariant(
                "execution index resident accounting overflow",
            ))
        })?;
    if actual_index_resident_bytes > plan.execution_index_resident_bytes {
        return Err(top_level_error(
            SimulationExecutionErrorKindV1::InternalInvariant(
                "execution index resident capacity exceeded preflight plan",
            ),
        ));
    }
    let (
        function_indices,
        function_module_indices,
        block_indices,
        function_ssa_values,
        call_targets,
        switch_targets,
    ) = indices;
    let entry_index = function_indices
        .get(&plan.kernel.entry)
        .copied()
        .ok_or_else(|| SimulationExecutionErrorV1 {
            invocation: None,
            site: None,
            kind: SimulationExecutionErrorKindV1::MissingFunction(plan.kernel.entry.clone()),
            observation_failure: None,
        })?;
    let entry_module_index = function_module_indices[entry_index];
    drop(function_indices);
    let entry = &admitted.module.functions[entry_module_index];
    let (value_plan, value_layouts) =
        prepare_frame_value_layouts(admitted, &function_module_indices)?;
    let memory = Memory::new(
        request.arguments.len(),
        request.shared_buffers.len(),
        limits,
        allocation_reuse,
    )
    .map_err(top_level_error)?;
    let mut accesses = HashMap::new();
    accesses
        .try_reserve(limits.max_memory_access_records)
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    let mut workgroup_allocations = Vec::new();
    workgroup_allocations
        .try_reserve_exact(plan.workgroup_allocation_sites)
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    Ok(PreparedExecutionStorage {
        function_module_indices,
        block_indices,
        function_ssa_values,
        call_targets,
        switch_targets,
        entry_index,
        entry,
        value_plan,
        value_layouts,
        memory,
        accesses,
        workgroup_allocations,
    })
}

fn execute(
    admitted: &AdmittedSimulationModuleV1,
    request: &SimulationRequestV1,
    configuration: ExecutionConfiguration<'_>,
    sink: &mut impl SimulationEventSinkV1,
    debug_sink: &mut impl SimulationDebugSinkV1,
) -> Result<SimulationExecutionV1, SimulationExecutionErrorV1> {
    execute_with_physical_debug(admitted, request, configuration, sink, debug_sink, None)
}

fn execute_with_physical_debug(
    admitted: &AdmittedSimulationModuleV1,
    request: &SimulationRequestV1,
    configuration: ExecutionConfiguration<'_>,
    sink: &mut impl SimulationEventSinkV1,
    debug_sink: &mut impl SimulationDebugSinkV1,
    physical_debug: Option<&mut debug_physical_capture_v20::State>,
) -> Result<SimulationExecutionV1, SimulationExecutionErrorV1> {
    if configuration.debug_capture.is_enabled()
        && (admitted.uses_physical_global_copy_v21()
            || admitted.uses_physical_entry_v20()
            || admitted.uses_physical_lds_exchange_v22())
        && !physical_debug
            .as_ref()
            .is_some_and(|capture| capture.matches(admitted))
    {
        return Err(top_level_error(
            SimulationExecutionErrorKindV1::InternalInvariant(
                "physical symbolic capture must be refused before execution",
            ),
        ));
    }
    let ExecutionConfiguration {
        target,
        limits,
        policy,
        plan,
        debug_capture,
        schedule,
        resident_offset,
        allocation_reuse,
        matrix_bf16_numerics,
    } = configuration;
    let workgroup_participants = usize::try_from(plan.workgroup[0])
        .ok()
        .and_then(|x| x.checked_mul(plan.workgroup[1] as usize))
        .and_then(|xy| xy.checked_mul(plan.workgroup[2] as usize))
        .ok_or_else(|| {
            top_level_error(SimulationExecutionErrorKindV1::InternalInvariant(
                "preflighted workgroup participant count",
            ))
        })?;
    let mut schedule = PreparedScheduleV1::prepare(
        schedule,
        admitted.identity,
        request,
        plan.dynamic_workgroup_memory,
        target,
        limits,
        &plan,
        workgroup_participants,
        resident_offset,
    )
    .map_err(|error| top_level_error(schedule_prepare_error(error)))?;
    let PreparedExecutionStorage {
        function_module_indices,
        block_indices,
        function_ssa_values,
        call_targets,
        switch_targets,
        entry_index,
        entry,
        value_plan,
        value_layouts,
        memory,
        accesses,
        workgroup_allocations,
    } = prepare_execution_storage(admitted, request, &plan, target, limits, allocation_reuse)?;
    let allocation_lifecycle_requested = debug_sink.wants_allocation_lifecycle_v1();
    let debug_observation_requested =
        debug_capture.is_enabled() && debug_sink.wants_observation_context_v1();
    let debug_frames_requested =
        debug_observation_requested && debug_sink.wants_checkpoint_frames_v1();
    let debug_origin_requested = debug_capture.is_enabled()
        && (debug_sink.wants_operation_origin_v1() || debug_frames_requested);
    // Plan resident admission includes this descriptor, barrier set and all
    // pending-write slots before their allocation. Context outlives Engine and
    // every FrameAction/BarrierArrival borrowing its barrier.
    let physical_lds_context = if admitted.uses_physical_lds_exchange_v22() {
        Some(physical_lds_context_v22::Context::new(&admitted.module)?)
    } else {
        None
    };
    let physical_lds_state = if physical_lds_context.is_some() {
        Some(physical_lds_context_v22::State::new()?)
    } else {
        None
    };
    let mut engine = Engine {
        physical_lds_context: physical_lds_context.as_ref(),
        physical_lds_state,
        module: &admitted.module,
        function_module_indices,
        block_indices,
        function_ssa_values,
        value_layouts: &value_layouts,
        value_plan,
        call_targets,
        switch_targets,
        target,
        dynamic_workgroup_memory: plan.dynamic_workgroup_memory,
        matrix_bf16_numerics,
        limits,
        policy,
        memory,
        sink,
        debug_capture,
        debug_sink,
        physical_debug,
        debug_origin_requested,
        debug_observation_requested,
        debug_frames_requested,
        debug_identity_failed: false,
        debug_origin: None,
        debug_records: 0,
        debug_delivery_stopped: !debug_capture.is_enabled(),
        allocation_lifecycle_requested,
        allocation_lifecycle_stopped: false,
        schedule_identity: schedule.identity(),
        schedule_decision: 0,
        steps: 0,
        events: 0,
        reserved_event_closures: 0,
        event_delivery_stopped: false,
        invocation: None,
        accesses,
        conflicting_bytes: 0,
        first_conflict: None,
        conflict_incomplete: false,
        workgroup_happens_before_epoch: 0,
        unmodeled_atomic_or_fence_happens_before: false,
        unmodeled_atomic_serialization: false,
        race_trackers: Vec::new(),
        workgroup_allocations,
    };
    initialize_shared_buffers(&mut engine, request)?;
    let parameters = initialize_arguments(&mut engine, entry, request)?;
    let invocation_site = entry
        .body
        .as_ref()
        .and_then(|body| body.blocks.first())
        .map(|block| terminator_site(entry_index, block))
        .ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::MissingBody(
                entry.id.clone(),
            ))
        })?;
    let mut invocations = 0_u64;
    let mut workgroups = 0_u64;
    let mut scheduled_slots = 0_u64;
    let mut machines = Vec::new();
    machines
        .try_reserve_exact(workgroup_participants)
        .map_err(|_| engine.fail(SimulationExecutionErrorKindV1::AllocationFailure))?;

    for group_z in 0..plan.workgroup_count[2] {
        for group_y in 0..plan.workgroup_count[1] {
            for group_x in 0..plan.workgroup_count[0] {
                workgroups += 1;
                schedule.begin_workgroup();
                engine.workgroup_happens_before_epoch = 0;
                machines.clear();
                for local_z in 0..plan.workgroup[2] {
                    for local_y in 0..plan.workgroup[1] {
                        for local_x in 0..plan.workgroup[0] {
                            scheduled_slots += 1;
                            let global = [
                                group_x * u64::from(plan.workgroup[0]) + u64::from(local_x),
                                group_y * u64::from(plan.workgroup[1]) + u64::from(local_y),
                                group_z * u64::from(plan.workgroup[2]) + u64::from(local_z),
                            ];
                            if global
                                .iter()
                                .zip(plan.grid)
                                .any(|(coordinate, extent)| *coordinate >= extent)
                            {
                                continue;
                            }
                            let invocation = SimulationInvocationV1 {
                                global,
                                workgroup: [group_x, group_y, group_z],
                                local: [local_x, local_y, local_z],
                                workgroup_size: plan.workgroup,
                                workgroup_count: plan.workgroup_count,
                                launch_extent: plan.grid,
                            };
                            engine.select_debug_invocation(Some(invocation));
                            machines.push(InvocationMachine::new(
                                &engine,
                                invocation,
                                entry_index,
                                entry,
                                &parameters,
                            )?);
                        }
                    }
                }

                let begun = begin_workgroup_invocations(
                    &mut engine,
                    &mut machines,
                    &invocation_site,
                    invocations == 0,
                )?;
                debug_assert_eq!(begun, machines.len());
                let execution = if schedule.uses_canonical_order() {
                    execute_cooperative_workgroup(
                        &mut engine,
                        &mut machines,
                        &invocation_site,
                        &mut invocations,
                        &mut schedule,
                        true,
                    )
                } else {
                    execute_cooperative_workgroup(
                        &mut engine,
                        &mut machines,
                        &invocation_site,
                        &mut invocations,
                        &mut schedule,
                        false,
                    )
                };
                if let Err(mut error) = execution {
                    abort_workgroup(
                        &mut engine,
                        &mut machines,
                        begun,
                        &invocation_site,
                        &mut error,
                    );
                    let mut primary = Some(&mut error);
                    if let Err(secondary) = engine.release_workgroup_allocations(&mut primary) {
                        attach_observation_failure(&mut error, secondary);
                    }
                    return Err(error);
                }
                let mut no_primary = None;
                engine.release_workgroup_allocations(&mut no_primary)?;
                if engine.reserved_event_closures != 0 {
                    return Err(
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "workgroup lifecycle credits were not closed",
                        )),
                    );
                }
            }
        }
    }
    if invocations != plan.invocations
        || workgroups != plan.workgroups
        || scheduled_slots != plan.scheduled_slots
    {
        return Err(
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "canonical scheduler count mismatch",
            )),
        );
    }
    let arguments = copy_back_arguments(&engine.memory, &request.arguments)?;
    let shared_buffers = copy_back_shared_buffers(&engine.memory, &request.shared_buffers)?;
    let conflict_assessment = engine.conflict_assessment();
    let schedule = schedule
        .finish(
            plan.workgroups,
            admitted.identity,
            request,
            plan.dynamic_workgroup_memory,
            target,
            limits,
        )
        .map_err(|error| engine.fail(SimulationExecutionErrorKindV1::ScheduleReplay(error)))?;
    let supplemental = collect_supplemental_observations(&engine, schedule.records)?;
    Ok(SimulationExecutionV1 {
        identity: admitted.identity,
        matrix_bf16_numerics,
        dynamic_workgroup_memory: plan.dynamic_workgroup_memory,
        arguments,
        shared_buffers,
        invocations_executed: invocations,
        workgroups_visited: workgroups,
        scheduled_slots_visited: scheduled_slots,
        steps_executed: engine.steps,
        events_emitted: engine.events,
        schedule: schedule.identity,
        schedule_transcript_identity: schedule.transcript_identity,
        schedule_coverage: schedule.coverage,
        supplemental,
        conflict_assessment,
    })
}

#[inline(never)]
fn collect_supplemental_observations(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    schedule_records: Vec<SimulationScheduleRecordV1>,
) -> Result<Vec<SimulationSupplementalV1>, SimulationExecutionErrorV1> {
    let mut supplemental = Vec::new();
    supplemental
        .try_reserve_exact(1)
        .map_err(|_| engine.fail(SimulationExecutionErrorKindV1::AllocationFailure))?;
    supplemental.push(SimulationSupplementalV1 {
        schedule_records,
        race_assessment: engine.race_assessment(),
    });
    Ok(supplemental)
}

fn schedule_prepare_error(error: SchedulePrepareErrorV1) -> SimulationExecutionErrorKindV1 {
    match error {
        SchedulePrepareErrorV1::DecisionLimit { actual, limit } => {
            SimulationExecutionErrorKindV1::ScheduleDecisionLimit { actual, limit }
        }
        SchedulePrepareErrorV1::ResidentLimit { actual, limit } => {
            SimulationExecutionErrorKindV1::ScheduleResidentLimit { actual, limit }
        }
        SchedulePrepareErrorV1::AllocationFailure => {
            SimulationExecutionErrorKindV1::AllocationFailure
        }
        SchedulePrepareErrorV1::Replay(error) => {
            SimulationExecutionErrorKindV1::ScheduleReplay(error)
        }
    }
}

fn begin_workgroup_invocations<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    invocation_site: &CompactSite,
    observe_preexisting: bool,
) -> Result<usize, SimulationExecutionErrorV1> {
    let mut begun = 0;
    for machine in machines.iter() {
        engine.select_debug_invocation(Some(machine.invocation));
        if let Err(mut error) =
            engine.begin_lifecycle(invocation_site, SimulationEventKindV1::InvocationBegin)
        {
            abort_workgroup(engine, machines, begun, invocation_site, &mut error);
            return Err(error);
        }
        begun += 1;
    }
    if observe_preexisting {
        let invocation = machines.first().ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "workgroup contained no live invocations",
            ))
        })?;
        engine.select_debug_invocation(Some(invocation.invocation));
        if let Err(mut error) = observe_preexisting_allocations(engine, invocation_site) {
            abort_workgroup(engine, machines, begun, invocation_site, &mut error);
            return Err(error);
        }
    }
    Ok(begun)
}

#[inline(never)]
fn execute_cooperative_workgroup<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    invocation_site: &CompactSite,
    invocations: &mut u64,
    schedule: &mut PreparedScheduleV1<'_>,
    canonical_order: bool,
) -> Result<(), SimulationExecutionErrorV1> {
    let workgroup = machines
        .first()
        .ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "workgroup contained no live invocations",
            ))
        })?
        .invocation
        .workgroup;
    let mut wave_results = Vec::new();
    wave_results
        .try_reserve_exact(machines.len())
        .map_err(|_| engine.fail(SimulationExecutionErrorKindV1::AllocationFailure))?;
    let mut phase = 0_u64;
    loop {
        if canonical_order {
            for machine in machines.iter_mut() {
                if machine.completed || machine.waiting.is_some() {
                    continue;
                }
                schedule
                    .selected(machine.invocation, phase)
                    .map_err(|error| engine.fail(schedule_prepare_error(error)))?;
                engine.schedule_decision = schedule.current_decision() - 1;
                engine.select_debug_invocation(Some(machine.invocation));
                match machine.advance_until_yield(engine, phase)? {
                    MachineYield::Complete => {
                        engine.end_lifecycle(
                            invocation_site,
                            SimulationEventKindV1::InvocationEnd {
                                outcome: SimulationExecutionOutcomeV1::Completed,
                            },
                        )?;
                        *invocations = invocations.checked_add(1).ok_or_else(|| {
                            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                                "completed invocation count overflow",
                            ))
                        })?;
                        machine.completed = true;
                    }
                    MachineYield::Barrier(arrival) => {
                        machine.waiting = Some(MachineWait::Barrier(arrival));
                    }
                    MachineYield::Wave(arrival) => {
                        machine.waiting = Some(MachineWait::Wave(arrival));
                    }
                    MachineYield::Collective(arrival) => {
                        machine.waiting = Some(MachineWait::Collective(arrival));
                    }
                }
            }
        } else {
            let order = schedule
                .take_order(
                    machines.len(),
                    |index| machines[index].invocation,
                    |index| !machines[index].completed && machines[index].waiting.is_none(),
                    workgroup,
                    phase,
                )
                .map_err(|error| {
                    engine.select_debug_invocation(None);
                    engine.fail(SimulationExecutionErrorKindV1::ScheduleReplay(error))
                })?;
            for index in order.iter().copied() {
                advance_runnable_machine(
                    engine,
                    &mut machines[index],
                    invocation_site,
                    invocations,
                    schedule,
                    phase,
                )?;
            }
            schedule.restore_order(order);
        }

        if resolve_ready_waves(engine, machines, &mut wave_results)? != 0 {
            continue;
        }
        if resolve_ready_collectives(engine, machines, schedule, &mut phase)? != 0 {
            continue;
        }
        if machines.iter().all(|machine| machine.completed) {
            return Ok(());
        }
        release_workgroup_barrier(engine, machines, schedule, &mut phase)?;
    }
}

fn advance_runnable_machine<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machine: &mut InvocationMachine<'a>,
    invocation_site: &CompactSite,
    invocations: &mut u64,
    schedule: &mut PreparedScheduleV1<'_>,
    phase: u64,
) -> Result<(), SimulationExecutionErrorV1> {
    schedule
        .selected(machine.invocation, phase)
        .map_err(|error| engine.fail(schedule_prepare_error(error)))?;
    engine.schedule_decision = schedule.current_decision() - 1;
    engine.select_debug_invocation(Some(machine.invocation));
    match machine.advance_until_yield(engine, phase)? {
        MachineYield::Complete => {
            engine.end_lifecycle(
                invocation_site,
                SimulationEventKindV1::InvocationEnd {
                    outcome: SimulationExecutionOutcomeV1::Completed,
                },
            )?;
            *invocations = invocations.checked_add(1).ok_or_else(|| {
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "completed invocation count overflow",
                ))
            })?;
            machine.completed = true;
        }
        MachineYield::Barrier(arrival) => {
            machine.waiting = Some(MachineWait::Barrier(arrival));
        }
        MachineYield::Wave(arrival) => machine.waiting = Some(MachineWait::Wave(arrival)),
        MachineYield::Collective(arrival) => {
            machine.waiting = Some(MachineWait::Collective(arrival));
        }
    }
    Ok(())
}

fn local_linear(invocation: SimulationInvocationV1) -> u64 {
    u64::from(invocation.local[0])
        + u64::from(invocation.workgroup_size[0])
            * (u64::from(invocation.local[1])
                + u64::from(invocation.workgroup_size[1]) * u64::from(invocation.local[2]))
}

fn full_wave_mask(width: WaveWidth) -> u64 {
    match width {
        WaveWidth::Wave32 => u64::from(u32::MAX),
        WaveWidth::Wave64 => u64::MAX,
    }
}

fn wave_active_mask(machines: &[InvocationMachine<'_>], start: u64, width: u64) -> u64 {
    machines.iter().fold(0_u64, |mask, machine| {
        let linear = local_linear(machine.invocation);
        if linear >= start && linear < start + width {
            mask | (1_u64 << (linear - start))
        } else {
            mask
        }
    })
}

fn wave_member_index(machines: &[InvocationMachine<'_>], start: u64, lane: u64) -> Option<usize> {
    machines
        .iter()
        .position(|machine| local_linear(machine.invocation) == start + lane)
}

fn resolve_ready_waves<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    results: &mut Vec<(usize, ScalarBitsV1)>,
) -> Result<usize, SimulationExecutionErrorV1> {
    let mut resolved = 0_usize;
    for representative in 0..machines.len() {
        let Some(MachineWait::Wave(arrival)) = machines[representative].waiting else {
            continue;
        };
        let width = u64::from(arrival.wave.width.lanes());
        let linear = local_linear(machines[representative].invocation);
        let wave_in_workgroup = linear / width;
        let start = wave_in_workgroup * width;
        let active_mask = wave_active_mask(machines, start, width);
        let required_mask = full_wave_mask(arrival.wave.width);
        if active_mask != required_mask {
            engine.select_debug_invocation(Some(machines[representative].invocation));
            return Err(engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::IncompleteWave(IncompleteWaveV1 {
                    width: arrival.wave.width,
                    wave_in_workgroup,
                    active_mask,
                    required_mask,
                }),
            ));
        }

        for lane in 0..width {
            let index = wave_member_index(machines, start, lane).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "full wave active mask member",
                    ),
                )
            })?;
            match machines[index].waiting {
                Some(MachineWait::Wave(peer))
                    if peer.site == arrival.site && peer.wave == arrival.wave => {}
                Some(MachineWait::Wave(peer)) => {
                    engine.select_debug_invocation(Some(machines[index].invocation));
                    return Err(engine.at(
                        peer.site,
                        SimulationExecutionErrorKindV1::MismatchedWave(MismatchedWaveV1 {
                            width: arrival.wave.width,
                            expected: engine.materialize_event_site(arrival.site),
                        }),
                    ));
                }
                _ => {
                    engine.select_debug_invocation(Some(machines[representative].invocation));
                    return Err(engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::DivergentWave(DivergentWaveV1 {
                            width: arrival.wave.width,
                            wave_in_workgroup,
                            nonparticipating: machines[index].invocation.into(),
                        }),
                    ));
                }
            }
        }

        let mut reduced_f32 = [ScalarBitsV1::boolean(false); 64];
        if let WaveOperationKind::ReduceF32 {
            tile_width, kind, ..
        } = arrival.wave.kind
        {
            for lane in 0..width {
                let source = wave_member_index(machines, start, lane).ok_or_else(|| {
                    engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::InternalInvariant("f32 reduction member"),
                    )
                })?;
                let Some(WaveInput::ReduceF32(value)) = machines[source].wave_input() else {
                    return Err(engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::InternalInvariant("f32 reduction input"),
                    ));
                };
                reduced_f32[lane as usize] = value;
            }
            let mut distance = 1_u32;
            while distance < tile_width {
                let previous = reduced_f32;
                for lane in 0..width {
                    let source = lane ^ u64::from(distance);
                    reduced_f32[lane as usize] = execute_wave_f32_reduction(
                        kind,
                        previous[lane as usize],
                        previous[source as usize],
                        engine.target,
                    )
                    .map_err(|error| engine.at(arrival.site, map_soft_float_error(error)))?;
                }
                distance *= 2;
            }
        }

        results.clear();
        for lane in 0..width {
            let index = wave_member_index(machines, start, lane).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant("wave result member"),
                )
            })?;
            let Some(MachineWait::Wave(peer)) = machines[index].waiting else {
                return Err(engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant("validated wave wait"),
                ));
            };
            let value = match peer.wave.kind {
                WaveOperationKind::LaneId => ScalarBitsV1::u32(lane as u32),
                WaveOperationKind::Ballot { .. } => {
                    let mut ballot = 0_u64;
                    for source_lane in 0..width {
                        let source =
                            wave_member_index(machines, start, source_lane).ok_or_else(|| {
                                engine.at(
                                    arrival.site,
                                    SimulationExecutionErrorKindV1::InternalInvariant(
                                        "ballot member",
                                    ),
                                )
                            })?;
                        let Some(WaveInput::Predicate(predicate)) = machines[source].wave_input()
                        else {
                            return Err(engine.at(
                                arrival.site,
                                SimulationExecutionErrorKindV1::InternalInvariant(
                                    "ballot predicate",
                                ),
                            ));
                        };
                        if predicate {
                            ballot |= 1_u64 << source_lane;
                        }
                    }
                    match peer.wave.width {
                        WaveWidth::Wave32 => ScalarBitsV1::u32(ballot as u32),
                        WaveWidth::Wave64 => {
                            ScalarBitsV1::new(ScalarType::U64, u128::from(ballot), engine.target)
                                .map_err(|_| {
                                    engine.at(
                                        arrival.site,
                                        SimulationExecutionErrorKindV1::InternalInvariant(
                                            "wave64 ballot scalar",
                                        ),
                                    )
                                })?
                        }
                    }
                }
                WaveOperationKind::Any { .. } | WaveOperationKind::All { .. } => {
                    let all = matches!(peer.wave.kind, WaveOperationKind::All { .. });
                    let mut aggregate = all;
                    for source_lane in 0..width {
                        let source =
                            wave_member_index(machines, start, source_lane).ok_or_else(|| {
                                engine.at(
                                    arrival.site,
                                    SimulationExecutionErrorKindV1::InternalInvariant(
                                        "vote member",
                                    ),
                                )
                            })?;
                        let Some(WaveInput::Predicate(predicate)) = machines[source].wave_input()
                        else {
                            return Err(engine.at(
                                arrival.site,
                                SimulationExecutionErrorKindV1::InternalInvariant("vote predicate"),
                            ));
                        };
                        if all {
                            aggregate &= predicate;
                        } else {
                            aggregate |= predicate;
                        }
                    }
                    ScalarBitsV1::boolean(aggregate)
                }
                WaveOperationKind::ShuffleIndex { .. } => {
                    let Some(WaveInput::Shuffle {
                        source_lane,
                        tile_width,
                        ..
                    }) = machines[index].wave_input()
                    else {
                        return Err(engine.at(
                            arrival.site,
                            SimulationExecutionErrorKindV1::InternalInvariant("shuffle input"),
                        ));
                    };
                    let tile_start = (lane / u64::from(tile_width)) * u64::from(tile_width);
                    let source =
                        wave_member_index(machines, start, tile_start + u64::from(source_lane))
                            .ok_or_else(|| {
                                engine.at(
                                    arrival.site,
                                    SimulationExecutionErrorKindV1::InternalInvariant(
                                        "shuffle source member",
                                    ),
                                )
                            })?;
                    let Some(WaveInput::Shuffle { value, .. }) = machines[source].wave_input()
                    else {
                        return Err(engine.at(
                            arrival.site,
                            SimulationExecutionErrorKindV1::InternalInvariant(
                                "shuffle source input",
                            ),
                        ));
                    };
                    value
                }
                WaveOperationKind::ReduceF32 { .. } => reduced_f32[lane as usize],
                WaveOperationKind::BroadcastF32 { .. } => {
                    let Some(WaveInput::BroadcastF32 {
                        source_lane,
                        tile_width,
                        ..
                    }) = machines[index].wave_input()
                    else {
                        return Err(engine.at(
                            arrival.site,
                            SimulationExecutionErrorKindV1::InternalInvariant(
                                "f32 broadcast input",
                            ),
                        ));
                    };
                    let tile_start = (lane / u64::from(tile_width)) * u64::from(tile_width);
                    let source =
                        wave_member_index(machines, start, tile_start + u64::from(source_lane))
                            .ok_or_else(|| {
                                engine.at(
                                    arrival.site,
                                    SimulationExecutionErrorKindV1::InternalInvariant(
                                        "f32 broadcast source member",
                                    ),
                                )
                            })?;
                    let Some(WaveInput::BroadcastF32 { value, .. }) = machines[source].wave_input()
                    else {
                        return Err(engine.at(
                            arrival.site,
                            SimulationExecutionErrorKindV1::InternalInvariant(
                                "f32 broadcast source input",
                            ),
                        ));
                    };
                    value
                }
            };
            results.push((index, value));
        }
        for (index, value) in results.iter().copied() {
            engine.select_debug_invocation(Some(machines[index].invocation));
            machines[index].complete_wave(engine, value)?;
        }
        resolved += 1;
    }
    Ok(resolved)
}

fn resolve_ready_collectives<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    schedule: &mut PreparedScheduleV1<'_>,
    phase: &mut u64,
) -> Result<usize, SimulationExecutionErrorV1> {
    let mut resolved = 0_usize;
    for representative in 0..machines.len() {
        let Some(MachineWait::Collective(arrival)) = machines[representative].waiting else {
            continue;
        };
        let matrix_exact = matrix_bf16_exact_v1::is_operation(arrival.operation);
        if matrix_exact {
            let work = matrix_bf16_exact_v1::resolution_work_with_numerics(
                machines.len(),
                engine.matrix_bf16_numerics,
            )
            .ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::StepLimit {
                        limit: engine.limits.max_steps,
                    },
                )
            })?;
            engine.charge_steps(&arrival.site, work)?;
        }
        let matrix_fp4_exact = matrix_fp4_exact_v1::is_operation(arrival.operation);
        if matrix_fp4_exact {
            let work = matrix_fp4_exact_v1::resolution_work(machines.len()).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::StepLimit {
                        limit: engine.limits.max_steps,
                    },
                )
            })?;
            engine.charge_steps(&arrival.site, work)?;
        }
        let matrix_fp8_exact = matrix_fp8_exact_v1::is_operation(arrival.operation);
        if matrix_fp8_exact {
            let work = matrix_fp8_exact_v1::resolution_work(machines.len()).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::StepLimit {
                        limit: engine.limits.max_steps,
                    },
                )
            })?;
            engine.charge_steps(&arrival.site, work)?;
        }
        let matrix_fp4_fp8_exact = matrix_fp4_fp8_exact_v1::is_operation(arrival.operation);
        if matrix_fp4_fp8_exact {
            let work =
                matrix_fp4_fp8_exact_v1::resolution_work(machines.len()).ok_or_else(|| {
                    engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::StepLimit {
                            limit: engine.limits.max_steps,
                        },
                    )
                })?;
            engine.charge_steps(&arrival.site, work)?;
        }
        let width = u64::from(arrival.width.lanes());
        let linear = local_linear(machines[representative].invocation);
        let wave_in_workgroup = linear / width;
        let start = wave_in_workgroup * width;
        let active_mask = wave_active_mask(machines, start, width);
        let required_mask = full_wave_mask(arrival.width);
        if active_mask != required_mask {
            engine.select_debug_invocation(Some(machines[representative].invocation));
            return Err(engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::IncompleteWave(IncompleteWaveV1 {
                    width: arrival.width,
                    wave_in_workgroup,
                    active_mask,
                    required_mask,
                }),
            ));
        }

        for lane in 0..width {
            let index = wave_member_index(machines, start, lane).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "full cooperative wave active mask member",
                    ),
                )
            })?;
            match machines[index].waiting {
                Some(MachineWait::Collective(peer))
                    if peer.site == arrival.site && peer.operation == arrival.operation => {}
                Some(MachineWait::Collective(peer)) => {
                    engine.select_debug_invocation(Some(machines[index].invocation));
                    return Err(engine.at(
                        peer.site,
                        SimulationExecutionErrorKindV1::MismatchedWave(MismatchedWaveV1 {
                            width: arrival.width,
                            expected: engine.materialize_event_site(arrival.site),
                        }),
                    ));
                }
                _ => {
                    engine.select_debug_invocation(Some(machines[representative].invocation));
                    return Err(engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::DivergentWave(DivergentWaveV1 {
                            width: arrival.width,
                            wave_in_workgroup,
                            nonparticipating: machines[index].invocation.into(),
                        }),
                    ));
                }
            }
        }

        if matrix_exact {
            matrix_bf16_exact_v1::resolve(engine, machines, arrival, start)?;
            resolved += 1;
            continue;
        }
        if matrix_fp4_exact {
            matrix_fp4_exact_v1::resolve(engine, machines, arrival, start)?;
            resolved += 1;
            continue;
        }
        if matrix_fp8_exact {
            matrix_fp8_exact_v1::resolve(engine, machines, arrival, start)?;
            resolved += 1;
            continue;
        }
        if matrix_fp4_fp8_exact {
            matrix_fp4_fp8_exact_v1::resolve(engine, machines, arrival, start)?;
            resolved += 1;
            continue;
        }
        if physical_entry_collective_v20::resolve(engine, machines, arrival, start)? {
            resolved += 1;
            continue;
        }

        if matches!(
            arrival.operation,
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                kind: Gfx950LdsTransposeOperationKindV1::Publish { .. },
                ..
            })
        ) {
            let participants = u32::try_from(width).map_err(|_| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "transpose publish participant count",
                    ),
                )
            })?;
            engine.select_debug_invocation(Some(machines[representative].invocation));
            engine.event(
                &arrival.site,
                SimulationEventKindV1::WorkgroupBarrierRelease {
                    phase: *phase,
                    participants,
                },
            )?;
            engine.debug_barrier(
                arrival.site,
                SimulationDebugBarrierActionV1::Release,
                *phase,
                participants,
            );
            schedule.barrier_released();
            engine.publish_workgroup();
            *phase = phase.checked_add(1).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::StepLimit {
                        limit: engine.limits.max_steps,
                    },
                )
            })?;
        }

        for lane in 0..width {
            let index = wave_member_index(machines, start, lane).ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "validated cooperative wave member",
                    ),
                )
            })?;
            let input = machines[index].collective_input().cloned().ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "cooperative operation input",
                    ),
                )
            })?;
            engine.select_debug_invocation(Some(machines[index].invocation));
            complete_collective_lane(engine, &mut machines[index], arrival, lane as u32, input)?;
        }
        resolved += 1;
    }
    Ok(resolved)
}

fn complete_collective_lane<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machine: &mut InvocationMachine<'a>,
    arrival: CollectiveArrival<'a>,
    lane: u32,
    input: CollectiveInput,
) -> Result<(), SimulationExecutionErrorV1> {
    if let Some(frame) = machine.frames.get(machine.active_depth.saturating_sub(1)) {
        engine.restore_debug_operation(frame, arrival.site);
    }
    match (arrival.operation, input) {
        (
            OperationKind::Matrix(MatrixOperation {
                kind: MatrixOperationKind::LdsLoad { profile, .. },
                ..
            }),
            CollectiveInput::MatrixLdsLoad { base },
        ) => {
            let values = execute_matrix_lds_load(engine, &arrival.site, &base, *profile, lane)?;
            let results = values.map(RuntimeValue::Scalar);
            machine.complete_collective(engine, &results)
        }
        (
            OperationKind::Matrix(MatrixOperation {
                kind: MatrixOperationKind::LdsStore { profile, .. },
                ..
            }),
            CollectiveInput::MatrixLdsStore { base, values },
        ) => {
            execute_matrix_lds_store(engine, &arrival.site, &base, *profile, lane, values)?;
            machine.complete_collective(engine, &[])
        }
        (
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                kind: Gfx950LdsTransposeOperationKindV1::Current { .. },
                ..
            }),
            CollectiveInput::TransposeCurrent { storage },
        ) => machine.complete_collective(engine, &[RuntimeValue::Pointer(storage)]),
        (
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                kind: Gfx950LdsTransposeOperationKindV1::Stage { format, .. },
                ..
            }),
            CollectiveInput::TransposeStage {
                storage,
                source,
                offset,
                rows,
                columns,
                stride,
                token_base,
                reduction_base,
            },
        ) => {
            execute_transpose_stage(
                engine,
                &arrival.site,
                *format,
                lane,
                &storage,
                &source,
                offset,
                rows,
                columns,
                stride,
                token_base,
                reduction_base,
            )?;
            machine.complete_collective(engine, &[RuntimeValue::Pointer(storage)])
        }
        (
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                kind: Gfx950LdsTransposeOperationKindV1::Publish { .. },
                ..
            }),
            CollectiveInput::TransposePublish { storage },
        ) => machine.complete_collective(engine, &[RuntimeValue::Pointer(storage)]),
        (
            OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                kind: Gfx950LdsTransposeOperationKindV1::Read { format, .. },
                ..
            }),
            CollectiveInput::TransposeRead { storage },
        ) => {
            let values = execute_transpose_read(engine, &arrival.site, *format, lane, &storage)?;
            let results = values.map(|value| RuntimeValue::Scalar(ScalarBitsV1::u32(value)));
            machine.complete_collective(engine, &results)
        }
        _ => Err(engine.at(
            arrival.site,
            SimulationExecutionErrorKindV1::InternalInvariant(
                "cooperative operation/input mismatch",
            ),
        )),
    }
}

fn release_workgroup_barrier<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    schedule: &mut PreparedScheduleV1<'_>,
    phase: &mut u64,
) -> Result<(), SimulationExecutionErrorV1> {
    for machine in machines.iter() {
        let valid = matches!(
            (machine.completed, machine.waiting),
            (false, Some(MachineWait::Barrier(_))) | (true, None)
        );
        if !valid {
            engine.select_debug_invocation(Some(machine.invocation));
            return Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "barrier resolution machine state",
                )),
            );
        }
    }
    let first_waiting = machines.iter().find_map(|machine| match machine.waiting {
        Some(MachineWait::Barrier(arrival)) => Some((machine.invocation, arrival)),
        _ => None,
    });
    let Some((representative, expected)) = first_waiting else {
        return Err(
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "barrier resolution without a waiting participant",
            )),
        );
    };
    for machine in machines.iter() {
        if let Some(MachineWait::Barrier(arrival)) = machine.waiting
            && (arrival.site != expected.site || arrival.barrier != expected.barrier)
        {
            let mismatch = match (
                arrival.site != expected.site,
                arrival.barrier != expected.barrier,
            ) {
                (true, true) => WorkgroupBarrierMismatchV1::SiteAndSemantics,
                (true, false) => WorkgroupBarrierMismatchV1::Site,
                (false, true) => WorkgroupBarrierMismatchV1::Semantics,
                (false, false) => {
                    return Err(engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::InternalInvariant(
                            "barrier mismatch classification",
                        ),
                    ));
                }
            };
            engine.select_debug_invocation(Some(machine.invocation));
            return Err(engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::MismatchedWorkgroupBarrier(
                    MismatchedWorkgroupBarrierV1 {
                        phase: *phase,
                        expected: engine.materialize_event_site(expected.site),
                        mismatch,
                    },
                ),
            ));
        }
    }
    if machines.iter().any(|machine| machine.completed) {
        let exited_representative = machines
            .iter()
            .find(|machine| machine.completed)
            .map(|machine| machine.invocation)
            .ok_or_else(|| {
                engine.at(
                    expected.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "divergent barrier exited participant",
                    ),
                )
            })?;
        if engine.debug_capture.is_enabled() {
            let mut waiting = Vec::new();
            let mut exited = Vec::new();
            waiting.try_reserve_exact(machines.len()).map_err(|_| {
                engine.at(
                    expected.site,
                    SimulationExecutionErrorKindV1::AllocationFailure,
                )
            })?;
            exited.try_reserve_exact(machines.len()).map_err(|_| {
                engine.at(
                    expected.site,
                    SimulationExecutionErrorKindV1::AllocationFailure,
                )
            })?;
            for machine in machines.iter() {
                match (machine.completed, machine.waiting) {
                    (false, Some(MachineWait::Barrier(_))) => {
                        waiting.push(machine.invocation.into());
                    }
                    (true, None) => exited.push(machine.invocation.into()),
                    _ => {
                        return Err(engine.at(
                            expected.site,
                            SimulationExecutionErrorKindV1::InternalInvariant(
                                "validated barrier resolution machine state",
                            ),
                        ));
                    }
                }
            }
            engine
                .debug_sink
                .terminal_barrier_divergence_v2(DivergentWorkgroupBarrierV2 {
                    phase: *phase,
                    waiting,
                    exited,
                });
        }
        engine.select_debug_invocation(Some(representative));
        return Err(engine.at(
            expected.site,
            SimulationExecutionErrorKindV1::DivergentWorkgroupBarrier(
                DivergentWorkgroupBarrierV1 {
                    phase: *phase,
                    waiting: representative.into(),
                    exited: exited_representative.into(),
                },
            ),
        ));
    }
    let participants = u32::try_from(machines.len()).map_err(|_| {
        engine.at(
            expected.site,
            SimulationExecutionErrorKindV1::InternalInvariant("workgroup participant count"),
        )
    })?;
    let physical_lds_release =
        physical_lds_context_v22::validate_release(engine, expected.site, participants)?;
    engine.select_debug_invocation(Some(representative));
    engine.event(
        &expected.site,
        SimulationEventKindV1::WorkgroupBarrierRelease {
            phase: *phase,
            participants,
        },
    )?;
    engine.debug_barrier(
        expected.site,
        SimulationDebugBarrierActionV1::Release,
        *phase,
        participants,
    );
    schedule.barrier_released();
    engine.publish_workgroup();
    if physical_lds_release {
        physical_lds_context_v22::mark_published(engine);
    }
    engine.publish_global_happens_before(expected.barrier);
    for machine in machines.iter_mut() {
        machine.waiting = None;
    }
    *phase = phase.checked_add(1).ok_or_else(|| {
        engine.at(
            expected.site,
            SimulationExecutionErrorKindV1::StepLimit {
                limit: engine.limits.max_steps,
            },
        )
    })?;
    Ok(())
}

fn abort_workgroup<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    begun: usize,
    invocation_site: &CompactSite,
    primary: &mut SimulationExecutionErrorV1,
) {
    for machine in machines.iter_mut().take(begun) {
        if machine.completed {
            continue;
        }
        engine.select_debug_invocation(Some(machine.invocation));
        if machine.active_depth != 0 {
            machine.frames.truncate(machine.active_depth);
            unwind_frames(engine, &mut machine.frames, primary);
            machine.active_depth = 0;
        }
        if let Err(secondary) = engine.end_lifecycle(
            invocation_site,
            SimulationEventKindV1::InvocationEnd {
                outcome: SimulationExecutionOutcomeV1::Failed,
            },
        ) {
            attach_observation_failure(primary, secondary);
        }
    }
}

fn observe_preexisting_allocations(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
) -> Result<(), SimulationExecutionErrorV1> {
    for allocation in 1..engine.memory.next_allocation {
        let facts = engine
            .memory
            .allocations
            .get(&allocation)
            .map(|allocation| (allocation.address_space, allocation.bytes.len()))
            .ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "preexisting allocation remained live",
                    ),
                )
            })?;
        engine.event(
            site,
            SimulationEventKindV1::AllocationPreexisting {
                allocation,
                address_space: facts.0,
                bytes: facts.1,
            },
        )?;
    }
    Ok(())
}

fn initialize_arguments(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    entry: &Function,
    request: &SimulationRequestV1,
) -> Result<Vec<RuntimeValue>, SimulationExecutionErrorV1> {
    let mut parameters = Vec::new();
    parameters
        .try_reserve_exact(request.arguments.len())
        .map_err(|_| engine.fail(SimulationExecutionErrorKindV1::AllocationFailure))?;
    for (index, (argument, ty)) in request
        .arguments
        .iter()
        .zip(&entry.signature.parameters)
        .enumerate()
    {
        let argument_ordinal = u32::try_from(index).map_err(|_| {
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "preflighted ABI argument ordinal",
            ))
        })?;
        let parameter = match (argument, ty) {
            (SimulationArgumentV1::Scalar(value), Type::Scalar(_)) => {
                Ok(RuntimeValue::Scalar(*value))
            }
            (SimulationArgumentV1::Buffer(buffer), Type::Slice(slice)) => {
                let Type::Scalar(element) = slice.element.as_ref() else {
                    return Err(
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted scalar slice",
                        )),
                    );
                };
                let allocation = allocate_argument(engine, index, buffer, AddressSpace::Global)?;
                Ok(RuntimeValue::Slice(SliceValue {
                    exposed_generic: false,
                    allocation,
                    elements: buffer.element_count(engine.target).map_err(|_| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted buffer target layout",
                        ))
                    })?,
                    element: *element,
                    address_space: slice.address_space,
                    access: slice.access,
                    byte_offset: 0,
                    byte_len: buffer.bytes().len(),
                    abi_argument_ordinal: argument_ordinal,
                }))
            }
            (SimulationArgumentV1::Buffer(buffer), Type::Pointer(pointer)) => {
                let Type::Scalar(element) = pointer.pointee.as_ref() else {
                    return Err(
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted scalar pointer",
                        )),
                    );
                };
                let allocation = allocate_argument(engine, index, buffer, AddressSpace::Global)?;
                Ok(RuntimeValue::Pointer(PointerValue {
                    exposed_generic: false,
                    allocation,
                    byte_offset: 0,
                    element: *element,
                    address_space: pointer.address_space,
                    access: pointer.access,
                    lower_bound: 0,
                    upper_bound: buffer.bytes().len(),
                    abi_argument_ordinal: argument_ordinal,
                }))
            }
            (SimulationArgumentV1::BufferView(view), Type::Slice(slice)) => {
                let Type::Scalar(element) = slice.element.as_ref() else {
                    return Err(
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted scalar slice view",
                        )),
                    );
                };
                let allocation = *engine
                    .memory
                    .shared_allocations
                    .get(&view.backing())
                    .ok_or_else(|| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted shared backing",
                        ))
                    })?;
                let byte_len = view.byte_len(engine.target).map_err(|_| {
                    engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                        "preflighted shared view layout",
                    ))
                })?;
                Ok(RuntimeValue::Slice(SliceValue {
                    exposed_generic: false,
                    allocation,
                    elements: view.elements(),
                    element: *element,
                    address_space: slice.address_space,
                    access: slice.access,
                    byte_offset: view.byte_offset(),
                    byte_len,
                    abi_argument_ordinal: argument_ordinal,
                }))
            }
            (SimulationArgumentV1::BufferView(view), Type::Pointer(pointer)) => {
                let Type::Scalar(element) = pointer.pointee.as_ref() else {
                    return Err(
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted scalar pointer view",
                        )),
                    );
                };
                let allocation = *engine
                    .memory
                    .shared_allocations
                    .get(&view.backing())
                    .ok_or_else(|| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted shared backing",
                        ))
                    })?;
                let upper_bound = view
                    .byte_offset()
                    .checked_add(view.byte_len(engine.target).map_err(|_| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted shared view layout",
                        ))
                    })?)
                    .ok_or_else(|| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted shared view bounds",
                        ))
                    })?;
                Ok(RuntimeValue::Pointer(PointerValue {
                    exposed_generic: false,
                    allocation,
                    byte_offset: view.byte_offset(),
                    element: *element,
                    address_space: pointer.address_space,
                    access: pointer.access,
                    lower_bound: view.byte_offset(),
                    upper_bound,
                    abi_argument_ordinal: argument_ordinal,
                }))
            }
            _ => Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "preflighted argument shape",
                )),
            ),
        }?;
        parameters.push(parameter);
    }
    Ok(parameters)
}

fn initialize_shared_buffers(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    request: &SimulationRequestV1,
) -> Result<(), SimulationExecutionErrorV1> {
    for shared in &request.shared_buffers {
        let bytes = try_clone_slice(shared.buffer.bytes()).map_err(|kind| engine.fail(kind))?;
        let initialized =
            try_clone_slice(shared.buffer.initialized()).map_err(|kind| engine.fail(kind))?;
        let commit = engine
            .memory
            .allocate(
                (
                    AddressSpace::Global,
                    shared.buffer.access(),
                    shared.buffer.alignment(),
                ),
                (bytes, initialized),
                AllocationCreationV1::dispatch(),
                engine.limits,
            )
            .map_err(|kind| engine.fail(kind))?;
        let allocation = commit.id;
        engine.deliver_allocation_lifecycle_v1(commit.transition);
        if engine
            .memory
            .shared_allocations
            .insert(shared.id, allocation)
            .is_some()
        {
            return Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "preflighted unique shared backing",
                )),
            );
        }
    }
    Ok(())
}

fn allocate_argument(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    index: usize,
    buffer: &BufferArgumentV1,
    address_space: AddressSpace,
) -> Result<u64, SimulationExecutionErrorV1> {
    engine
        .memory
        .validate_allocation(buffer.bytes().len(), engine.limits)
        .map_err(|kind| engine.fail(kind))?;
    let bytes = try_clone_slice(buffer.bytes()).map_err(|kind| engine.fail(kind))?;
    let initialized = try_clone_slice(buffer.initialized()).map_err(|kind| engine.fail(kind))?;
    let commit = engine
        .memory
        .allocate(
            (address_space, buffer.access(), buffer.alignment()),
            (bytes, initialized),
            AllocationCreationV1::dispatch(),
            engine.limits,
        )
        .map_err(|kind| engine.fail(kind))?;
    let id = commit.id;
    engine.deliver_allocation_lifecycle_v1(commit.transition);
    engine.memory.argument_allocations[index] = Some(id);
    Ok(id)
}

fn copy_back_arguments(
    memory: &Memory,
    source: &[SimulationArgumentV1],
) -> Result<Vec<SimulationArgumentV1>, SimulationExecutionErrorV1> {
    let mut arguments = Vec::new();
    arguments
        .try_reserve_exact(source.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    for (index, argument) in source.iter().enumerate() {
        let output = match argument {
            SimulationArgumentV1::Scalar(value) => SimulationArgumentV1::Scalar(*value),
            SimulationArgumentV1::Buffer(buffer) => {
                let allocation_id = memory.argument_allocations[index].ok_or_else(|| {
                    top_level_error(SimulationExecutionErrorKindV1::InternalInvariant(
                        "distinct argument allocation",
                    ))
                })?;
                let allocation = memory.allocations.get(&allocation_id).ok_or_else(|| {
                    top_level_error(SimulationExecutionErrorKindV1::DanglingPointer {
                        allocation: allocation_id,
                    })
                })?;
                SimulationArgumentV1::Buffer(buffer.with_contents(
                    try_clone_slice(&allocation.bytes).map_err(top_level_error)?,
                    try_clone_slice(&allocation.initialized).map_err(top_level_error)?,
                ))
            }
            SimulationArgumentV1::BufferView(view) => {
                SimulationArgumentV1::BufferView(view.clone())
            }
        };
        arguments.push(output);
    }
    Ok(arguments)
}

fn copy_back_shared_buffers(
    memory: &Memory,
    source: &[SharedBufferV1],
) -> Result<Vec<SharedBufferV1>, SimulationExecutionErrorV1> {
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(source.len())
        .map_err(|_| top_level_error(SimulationExecutionErrorKindV1::AllocationFailure))?;
    for shared in source {
        let allocation_id = memory.shared_allocations.get(&shared.id).ok_or_else(|| {
            top_level_error(SimulationExecutionErrorKindV1::InternalInvariant(
                "shared output allocation",
            ))
        })?;
        let allocation = memory.allocations.get(allocation_id).ok_or_else(|| {
            top_level_error(SimulationExecutionErrorKindV1::DanglingPointer {
                allocation: *allocation_id,
            })
        })?;
        let buffer = shared.buffer.with_contents(
            try_clone_slice(&allocation.bytes).map_err(top_level_error)?,
            try_clone_slice(&allocation.initialized).map_err(top_level_error)?,
        );
        outputs.push(SharedBufferV1 {
            id: shared.id,
            buffer,
        });
    }
    Ok(outputs)
}

struct RuntimeFrame<'a> {
    function_index: usize,
    function: &'a Function,
    values: RuntimeValues<'a>,
    allocations: Vec<FrameAllocation>,
    current: BlockId,
    current_index: usize,
    incoming: Vec<RuntimeValue>,
    operation: usize,
    block_entered: bool,
    active_operation: Option<CompactSite>,
    debug_identity: Option<FrameIdentityState>,
}

#[derive(Clone, Copy)]
struct FrameAllocation {
    id: u64,
    lifecycle_observed: bool,
}

enum FrameAction<'a> {
    Continue,
    Call {
        function_index: usize,
        function: &'a Function,
        site: CompactSite,
    },
    Barrier {
        site: CompactSite,
        barrier: &'a WorkgroupBarrier,
    },
    Return,
}

struct InvocationMachine<'a> {
    invocation: SimulationInvocationV1,
    frames: Vec<RuntimeFrame<'a>>,
    active_depth: usize,
    completed: bool,
    waiting: Option<MachineWait<'a>>,
    pending_wave_input: Option<WaveInput>,
    pending_collective_input: Option<CollectiveInput>,
    debug_identity: Option<InvocationIdentityState>,
}

#[derive(Clone, Copy)]
struct BarrierArrival<'a> {
    site: CompactSite,
    barrier: &'a WorkgroupBarrier,
}

#[derive(Clone, Copy)]
enum WaveInput {
    LaneId,
    Predicate(bool),
    Shuffle {
        value: ScalarBitsV1,
        source_lane: u32,
        tile_width: u32,
    },
    ReduceF32(ScalarBitsV1),
    BroadcastF32 {
        value: ScalarBitsV1,
        source_lane: u32,
        tile_width: u32,
    },
}

#[derive(Clone, Copy)]
struct WaveArrival<'a> {
    site: CompactSite,
    wave: &'a WaveOperation,
}

#[derive(Clone)]
enum CollectiveInput {
    MatrixBf16Exact(matrix_bf16_exact_v1::Input),
    MatrixFp4Exact(matrix_fp4_exact_v1::Input),
    MatrixFp8Exact(matrix_fp8_exact_v1::Input),
    MatrixFp4Fp8Exact(matrix_fp4_fp8_exact_v1::Input),
    PhysicalEntryPredicate(bool),
    MatrixLdsLoad {
        base: PointerValue,
    },
    MatrixLdsStore {
        base: PointerValue,
        values: [ScalarBitsV1; 4],
    },
    TransposeCurrent {
        storage: PointerValue,
    },
    TransposeStage {
        storage: PointerValue,
        source: SliceValue,
        offset: u64,
        rows: u64,
        columns: u64,
        stride: u64,
        token_base: u64,
        reduction_base: u64,
    },
    TransposePublish {
        storage: PointerValue,
    },
    TransposeRead {
        storage: PointerValue,
    },
}

#[derive(Clone, Copy)]
struct CollectiveArrival<'a> {
    site: CompactSite,
    operation: &'a OperationKind,
    width: WaveWidth,
}

#[derive(Clone, Copy)]
enum MachineWait<'a> {
    Barrier(BarrierArrival<'a>),
    Wave(WaveArrival<'a>),
    Collective(CollectiveArrival<'a>),
}

enum MachineYield<'a> {
    Barrier(BarrierArrival<'a>),
    Wave(WaveArrival<'a>),
    Collective(CollectiveArrival<'a>),
    Complete,
}

impl<'a> RuntimeFrame<'a> {
    fn new(
        engine: &Engine<'a, impl SimulationEventSinkV1>,
        function_index: usize,
        function: &'a Function,
        arguments: &[RuntimeValue],
    ) -> Result<Self, SimulationExecutionErrorV1> {
        let body = function.body.as_ref().ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::MissingBody(
                function.id.clone(),
            ))
        })?;
        let entry = body.blocks.first().ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::MissingBody(
                function.id.clone(),
            ))
        })?;
        if arguments.len() != body.parameters.len() {
            return Err(engine.fail(SimulationExecutionErrorKindV1::ResultArity {
                expected: body.parameters.len(),
                actual: arguments.len(),
            }));
        }
        let mut values = RuntimeValues::prepared(
            engine.value_layouts.get(function_index),
            engine.value_plan,
            engine.target,
            engine.function_ssa_values[function_index],
        )
        .map_err(|kind| engine.fail(kind))?;
        for (id, argument) in body.parameters.iter().copied().zip(arguments) {
            bind_runtime_value(engine, &mut values, id, argument.clone())?;
        }
        Ok(Self {
            function_index,
            function,
            values,
            allocations: Vec::new(),
            current: entry.id,
            current_index: 0,
            incoming: Vec::new(),
            operation: 0,
            block_entered: false,
            active_operation: None,
            debug_identity: None,
        })
    }

    fn reset(
        &mut self,
        engine: &Engine<'a, impl SimulationEventSinkV1>,
        function_index: usize,
        function: &'a Function,
        arguments: &[RuntimeValue],
    ) -> Result<(), SimulationExecutionErrorV1> {
        let body = function.body.as_ref().ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::MissingBody(
                function.id.clone(),
            ))
        })?;
        let entry = body.blocks.first().ok_or_else(|| {
            engine.fail(SimulationExecutionErrorKindV1::MissingBody(
                function.id.clone(),
            ))
        })?;
        if arguments.len() != body.parameters.len() {
            return Err(engine.fail(SimulationExecutionErrorKindV1::ResultArity {
                expected: body.parameters.len(),
                actual: arguments.len(),
            }));
        }
        if !self.allocations.is_empty() || self.active_operation.is_some() {
            return Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "completed root frame retained live state",
                )),
            );
        }
        self.values
            .reset(
                engine.value_layouts.get(function_index),
                engine.function_ssa_values[function_index],
            )
            .map_err(|kind| engine.fail(kind))?;
        for (id, argument) in body.parameters.iter().copied().zip(arguments) {
            bind_runtime_value(engine, &mut self.values, id, argument.clone())?;
        }
        self.function_index = function_index;
        self.function = function;
        self.current = entry.id;
        self.current_index = 0;
        self.incoming.clear();
        self.operation = 0;
        self.block_entered = false;
        Ok(())
    }
}

fn function_ssa_definition_count(function: &Function) -> Option<usize> {
    let body = function.body.as_ref()?;
    let mut count = body.parameters.len();
    for block in &body.blocks {
        count = count.checked_add(block.parameters.len())?;
        for operation in &block.operations {
            count = count.checked_add(operation.results.len())?;
        }
    }
    Some(count)
}

impl<'a> InvocationMachine<'a> {
    fn new(
        engine: &Engine<'a, impl SimulationEventSinkV1>,
        invocation: SimulationInvocationV1,
        function_index: usize,
        function: &'a Function,
        arguments: &[RuntimeValue],
    ) -> Result<Self, SimulationExecutionErrorV1> {
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(1)
            .map_err(|_| engine.fail(SimulationExecutionErrorKindV1::AllocationFailure))?;
        frames.push(RuntimeFrame::new(
            engine,
            function_index,
            function,
            arguments,
        )?);
        let (debug_identity, root_identity) =
            debug_identity::root_identity(engine, invocation, function_index);
        frames[0].debug_identity = root_identity;
        Ok(Self {
            invocation,
            frames,
            active_depth: 1,
            completed: false,
            waiting: None,
            pending_wave_input: None,
            pending_collective_input: None,
            debug_identity,
        })
    }

    fn advance_until_yield<S: SimulationEventSinkV1>(
        &mut self,
        engine: &mut Engine<'a, S>,
        phase: u64,
    ) -> Result<MachineYield<'a>, SimulationExecutionErrorV1> {
        if self.completed || self.active_depth == 0 || self.waiting.is_some() {
            return Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "advanced completed invocation machine",
                )),
            );
        }

        loop {
            engine.debug_origin = None;
            let checkpoint_site = self.frames.get(self.active_depth - 1).and_then(|frame| {
                if !frame.block_entered {
                    return None;
                }
                let block = frame
                    .function
                    .body
                    .as_ref()?
                    .blocks
                    .get(frame.current_index)?;
                block
                    .operations
                    .get(frame.operation)
                    .map(|_| operation_site(frame.function_index, block, frame.operation))
            });
            if let Some(site) = checkpoint_site {
                engine.begin_debug_operation(&mut self.frames[self.active_depth - 1], site);
                engine.debug_checkpoint(
                    &self.frames[..self.active_depth],
                    site,
                    SimulationDebugCheckpointPhaseV1::BeforeOperation,
                );
            }
            if self.frames[self.active_depth - 1].block_entered
                && self.frames[self.active_depth - 1]
                    .function
                    .body
                    .as_ref()
                    .and_then(|body| {
                        body.blocks
                            .get(self.frames[self.active_depth - 1].current_index)
                    })
                    .and_then(|block| {
                        block
                            .operations
                            .get(self.frames[self.active_depth - 1].operation)
                    })
                    .is_some_and(|operation| matches!(operation.kind, OperationKind::Wave(_)))
            {
                let frame = self.frames.get_mut(self.active_depth - 1).ok_or_else(|| {
                    engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                        "wave runtime frame",
                    ))
                })?;
                let arrival = advance_wave_frame(engine, frame, &mut self.pending_wave_input)?;
                engine.suspend_debug_operation(frame);
                return Ok(MachineYield::Wave(arrival));
            }
            if self.frames[self.active_depth - 1].block_entered
                && self.frames[self.active_depth - 1]
                    .function
                    .body
                    .as_ref()
                    .and_then(|body| {
                        body.blocks
                            .get(self.frames[self.active_depth - 1].current_index)
                    })
                    .and_then(|block| {
                        block
                            .operations
                            .get(self.frames[self.active_depth - 1].operation)
                    })
                    .is_some_and(|operation| {
                        matches!(
                            operation.kind,
                            OperationKind::Matrix(_) | OperationKind::Gfx950LdsTranspose(_)
                        ) || physical_entry_v20::is_collective(&operation.kind)
                            || physical_global_copy_v21::is_collective(&operation.kind)
                            || physical_lds_exchange_v22::is_collective(&operation.kind)
                    })
            {
                let frame = self.frames.get_mut(self.active_depth - 1).ok_or_else(|| {
                    engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                        "collective runtime frame",
                    ))
                })?;
                let arrival = advance_collective_frame(
                    engine,
                    frame,
                    phase,
                    &mut self.pending_collective_input,
                )?;
                engine.suspend_debug_operation(frame);
                return Ok(MachineYield::Collective(arrival));
            }
            let action = {
                let frame = self.frames.get_mut(self.active_depth - 1).ok_or_else(|| {
                    engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                        "runtime frame stack",
                    ))
                })?;
                if !frame.block_entered {
                    advance_block_entry_frame(engine, frame)
                } else if is_internal_call_frame(engine, frame) {
                    advance_internal_call_frame(engine, frame)
                } else if is_return_frame(frame) {
                    advance_return_frame(engine, frame)
                } else {
                    advance_frame(engine, frame, phase)
                }
            };
            let action = match action {
                Ok(action) => action,
                Err(mut error) => {
                    self.frames.truncate(self.active_depth);
                    unwind_frames(engine, &mut self.frames, &mut error);
                    self.active_depth = 0;
                    return Err(error);
                }
            };
            if let Some(site) = checkpoint_site
                && matches!(&action, FrameAction::Continue | FrameAction::Barrier { .. })
            {
                engine.debug_checkpoint(
                    &self.frames[..self.active_depth],
                    site,
                    SimulationDebugCheckpointPhaseV1::AfterOperation,
                );
                engine.complete_debug_operation(&mut self.frames[self.active_depth - 1]);
            }
            match action {
                FrameAction::Continue => {}
                FrameAction::Barrier { site, barrier } => {
                    return Ok(MachineYield::Barrier(BarrierArrival { site, barrier }));
                }
                FrameAction::Call {
                    function_index,
                    function,
                    site,
                } => {
                    engine.suspend_debug_operation(&mut self.frames[self.active_depth - 1]);
                    if self.active_depth == engine.limits.max_call_depth {
                        let mut error = engine.at(
                            site,
                            SimulationExecutionErrorKindV1::CallDepthLimit {
                                limit: engine.limits.max_call_depth,
                            },
                        );
                        self.frames.truncate(self.active_depth);
                        unwind_frames(engine, &mut self.frames, &mut error);
                        self.active_depth = 0;
                        return Err(error);
                    }
                    let arguments =
                        std::mem::take(&mut self.frames[self.active_depth - 1].incoming);
                    let frame_result = if self.active_depth == self.frames.len() {
                        if self.frames.len() == self.frames.capacity()
                            && self.frames.try_reserve_exact(1).is_err()
                        {
                            Err(engine.at(site, SimulationExecutionErrorKindV1::AllocationFailure))
                        } else {
                            RuntimeFrame::new(engine, function_index, function, &arguments)
                                .map(|frame| self.frames.push(frame))
                        }
                    } else {
                        self.frames[self.active_depth].reset(
                            engine,
                            function_index,
                            function,
                            &arguments,
                        )
                    };
                    self.frames[self.active_depth - 1].incoming = arguments;
                    self.frames[self.active_depth - 1].incoming.clear();
                    if let Err(mut error) = frame_result {
                        self.frames.truncate(self.active_depth);
                        unwind_frames(engine, &mut self.frames, &mut error);
                        self.active_depth = 0;
                        return Err(error);
                    }
                    self.enter_debug_frame(engine);
                    self.active_depth += 1;
                }
                FrameAction::Return => {
                    engine.retire_debug_frame(&mut self.frames[self.active_depth - 1]);
                    let returned = std::mem::take(&mut self.frames[self.active_depth - 1].incoming);
                    if self.active_depth == 1 {
                        if !returned.is_empty() {
                            let mut error =
                                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                                    "kernel returned values after verification",
                                ));
                            self.frames[0].incoming = returned;
                            unwind_frames(engine, &mut self.frames, &mut error);
                            self.active_depth = 0;
                            return Err(error);
                        }
                        self.frames[0].incoming = returned;
                        self.active_depth = 0;
                        self.completed = true;
                        return Ok(MachineYield::Complete);
                    }
                    self.active_depth -= 1;
                    let caller = self.frames.get_mut(self.active_depth - 1).ok_or_else(|| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "caller frame after nested return",
                        ))
                    })?;
                    let site = caller.active_operation.ok_or_else(|| {
                        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                            "caller operation lifecycle",
                        ))
                    })?;
                    engine.resume_debug_operation(caller, site);
                    let operation = caller
                        .function
                        .body
                        .as_ref()
                        .and_then(|body| body.blocks.get(caller.current_index))
                        .and_then(|block| block.operations.get(caller.operation))
                        .ok_or_else(|| {
                            engine.at(
                                site,
                                SimulationExecutionErrorKindV1::InternalInvariant(
                                    "suspended call operation",
                                ),
                            )
                        })?;
                    if let Err(mut error) = bind_dynamic_results(
                        engine,
                        &mut caller.values,
                        &operation.results,
                        &returned,
                        &site,
                    ) {
                        self.frames[self.active_depth].incoming = returned;
                        self.frames[self.active_depth].incoming.clear();
                        self.frames.truncate(self.active_depth);
                        unwind_frames(engine, &mut self.frames, &mut error);
                        self.active_depth = 0;
                        return Err(error);
                    }
                    caller.incoming.clear();
                    caller.active_operation = None;
                    if let Err(mut error) = engine.end_lifecycle(
                        &site,
                        SimulationEventKindV1::OperationEnd {
                            outcome: SimulationExecutionOutcomeV1::Completed,
                        },
                    ) {
                        self.frames[self.active_depth].incoming = returned;
                        self.frames[self.active_depth].incoming.clear();
                        self.frames.truncate(self.active_depth);
                        unwind_frames(engine, &mut self.frames, &mut error);
                        self.active_depth = 0;
                        return Err(error);
                    }
                    caller.operation += 1;
                    self.frames[self.active_depth].incoming = returned;
                    self.frames[self.active_depth].incoming.clear();
                    engine.debug_checkpoint(
                        &self.frames[..self.active_depth],
                        site,
                        SimulationDebugCheckpointPhaseV1::AfterOperation,
                    );
                    engine.complete_debug_operation(&mut self.frames[self.active_depth - 1]);
                }
            }
        }
    }

    fn complete_wave<S: SimulationEventSinkV1>(
        &mut self,
        engine: &mut Engine<'a, S>,
        value: ScalarBitsV1,
    ) -> Result<(), SimulationExecutionErrorV1> {
        let Some(MachineWait::Wave(arrival)) = self.waiting.take() else {
            return Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "completed a machine not waiting at a wave operation",
                )),
            );
        };
        let frame = self.frames.get_mut(self.active_depth - 1).ok_or_else(|| {
            engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::InternalInvariant("wave runtime frame"),
            )
        })?;
        engine.resume_debug_operation(frame, arrival.site);
        let operation = frame
            .function
            .body
            .as_ref()
            .and_then(|body| body.blocks.get(frame.current_index))
            .and_then(|block| block.operations.get(frame.operation))
            .ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant("suspended wave operation"),
                )
            })?;
        bind_small_results(
            engine,
            &mut frame.values,
            &operation.results,
            SmallResults::One(RuntimeValue::Scalar(value)),
            &arrival.site,
        )?;
        self.pending_wave_input = None;
        frame.active_operation = None;
        engine.end_lifecycle(
            &arrival.site,
            SimulationEventKindV1::OperationEnd {
                outcome: SimulationExecutionOutcomeV1::Completed,
            },
        )?;
        frame.operation += 1;
        engine.debug_checkpoint(
            &self.frames[..self.active_depth],
            arrival.site,
            SimulationDebugCheckpointPhaseV1::AfterOperation,
        );
        engine.complete_debug_operation(&mut self.frames[self.active_depth - 1]);
        Ok(())
    }

    fn wave_input(&self) -> Option<WaveInput> {
        self.pending_wave_input
    }

    fn collective_input(&self) -> Option<&CollectiveInput> {
        self.pending_collective_input.as_ref()
    }

    fn complete_collective<S: SimulationEventSinkV1>(
        &mut self,
        engine: &mut Engine<'a, S>,
        results: &[RuntimeValue],
    ) -> Result<(), SimulationExecutionErrorV1> {
        let Some(MachineWait::Collective(arrival)) = self.waiting.take() else {
            return Err(
                engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                    "completed a machine not waiting at a cooperative operation",
                )),
            );
        };
        let frame = self.frames.get_mut(self.active_depth - 1).ok_or_else(|| {
            engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "cooperative operation runtime frame",
                ),
            )
        })?;
        engine.resume_debug_operation(frame, arrival.site);
        let operation = frame
            .function
            .body
            .as_ref()
            .and_then(|body| body.blocks.get(frame.current_index))
            .and_then(|block| block.operations.get(frame.operation))
            .ok_or_else(|| {
                engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "suspended cooperative operation",
                    ),
                )
            })?;
        bind_dynamic_results(
            engine,
            &mut frame.values,
            &operation.results,
            results,
            &arrival.site,
        )?;
        self.pending_collective_input = None;
        frame.active_operation = None;
        engine.end_lifecycle(
            &arrival.site,
            SimulationEventKindV1::OperationEnd {
                outcome: SimulationExecutionOutcomeV1::Completed,
            },
        )?;
        frame.operation += 1;
        engine.debug_checkpoint(
            &self.frames[..self.active_depth],
            arrival.site,
            SimulationDebugCheckpointPhaseV1::AfterOperation,
        );
        engine.complete_debug_operation(&mut self.frames[self.active_depth - 1]);
        Ok(())
    }
}

#[inline(never)]
fn advance_block_entry_frame<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    let block = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(frame.current)))?;
    bind_block_arguments(
        engine,
        frame.function_index,
        block,
        &frame.incoming,
        &mut frame.values,
    )?;
    frame.incoming.clear();
    let site = terminator_site(frame.function_index, block);
    engine.event(&site, SimulationEventKindV1::BlockEnter)?;
    frame.block_entered = true;
    Ok(FrameAction::Continue)
}

fn is_internal_call_frame(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    frame: &RuntimeFrame<'_>,
) -> bool {
    if !frame.block_entered {
        return false;
    }
    let Some(block) = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
    else {
        return false;
    };
    if !matches!(
        block.operations.get(frame.operation),
        Some(Operation {
            kind: OperationKind::Call { .. },
            ..
        })
    ) {
        return false;
    }
    matches!(
        engine
            .call_targets
            .get(frame.function_index)
            .and_then(|function| function.get(frame.current_index))
            .and_then(|block| block.get(frame.operation)),
        Some(CallTarget::Internal(_))
    )
}

fn is_return_frame(frame: &RuntimeFrame<'_>) -> bool {
    let Some(block) = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
    else {
        return false;
    };
    frame.operation == block.operations.len()
        && matches!(block.terminator, Some(Terminator::Return { .. }))
}

#[inline(never)]
fn advance_return_frame<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    let block = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(frame.current)))?;
    let site = terminator_site(frame.function_index, block);
    let Some(Terminator::Return { values }) = &block.terminator else {
        return Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant("return frame dispatch"),
        ));
    };
    engine.step(&site)?;
    engine.event(&site, SimulationEventKindV1::Terminator)?;
    resolve_values_into(engine, &frame.values, values, &site, &mut frame.incoming)?;
    release_frame_allocations_observed(engine, &mut frame.allocations, &site)?;
    engine.event(&site, SimulationEventKindV1::Return)?;
    Ok(FrameAction::Return)
}

#[inline(never)]
fn advance_internal_call_frame<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    let block = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(frame.current)))?;
    let operation = block.operations.get(frame.operation).ok_or_else(|| {
        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
            "internal call operation position",
        ))
    })?;
    let OperationKind::Call { arguments, .. } = &operation.kind else {
        return Err(
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "internal call operation dispatch",
            )),
        );
    };
    let site = operation_site(frame.function_index, block, frame.operation);
    engine.step(&site)?;
    engine.begin_lifecycle(&site, SimulationEventKindV1::OperationBegin)?;
    frame.active_operation = Some(site);
    let CallTarget::Internal(callee_index) =
        engine.call_targets[frame.function_index][frame.current_index][frame.operation]
    else {
        return Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant("preflighted internal call index"),
        ));
    };
    let callee_function = &engine.module.functions[engine.function_module_indices[callee_index]];
    if callee_function.role != FunctionRole::InternalHelper {
        return Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant("preflighted internal call"),
        ));
    }
    resolve_values_into(engine, &frame.values, arguments, &site, &mut frame.incoming)?;
    engine.call_event(&site, callee_index)?;
    Ok(FrameAction::Call {
        function_index: callee_index,
        function: callee_function,
        site,
    })
}

fn advance_frame<'a, S: SimulationEventSinkV1>(
    engine: &mut Engine<'a, S>,
    frame: &mut RuntimeFrame<'a>,
    phase: u64,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    let body = frame.function.body.as_ref().ok_or_else(|| {
        engine.fail(SimulationExecutionErrorKindV1::MissingBody(
            frame.function.id.clone(),
        ))
    })?;
    let block = body
        .blocks
        .get(frame.current_index)
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(frame.current)))?;
    if !frame.block_entered {
        bind_block_arguments(
            engine,
            frame.function_index,
            block,
            &frame.incoming,
            &mut frame.values,
        )?;
        frame.incoming.clear();
        let site = terminator_site(frame.function_index, block);
        engine.event(&site, SimulationEventKindV1::BlockEnter)?;
        frame.block_entered = true;
        return Ok(FrameAction::Continue);
    }

    if let Some(operation) = block.operations.get(frame.operation) {
        let site = operation_site(frame.function_index, block, frame.operation);
        engine.step(&site)?;
        engine.begin_lifecycle(&site, SimulationEventKindV1::OperationBegin)?;
        frame.active_operation = Some(site);
        if let OperationKind::Call { arguments, .. } = &operation.kind {
            let target =
                engine.call_targets[frame.function_index][frame.current_index][frame.operation];
            let callee_index = match target {
                CallTarget::Float(_) | CallTarget::Trap => None,
                CallTarget::Internal(callee_index) => Some(callee_index),
                CallTarget::NotCall => {
                    return Err(engine.at(
                        site,
                        SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted operation call index",
                        ),
                    ));
                }
            };
            if let Some(callee_index) = callee_index {
                let callee_function =
                    &engine.module.functions[engine.function_module_indices[callee_index]];
                if callee_function.role != FunctionRole::InternalHelper {
                    return Err(engine.at(
                        site,
                        SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted internal call",
                        ),
                    ));
                }
                resolve_values_into(engine, &frame.values, arguments, &site, &mut frame.incoming)?;
                engine.call_event(&site, callee_index)?;
                return Ok(FrameAction::Call {
                    function_index: callee_index,
                    function: callee_function,
                    site,
                });
            }
        }
        let barrier = match &operation.kind {
            OperationKind::WorkgroupBarrier(barrier) => Some(barrier),
            kind if physical_lds_exchange_v22::is_barrier(kind) => {
                Some(physical_lds_context_v22::barrier(engine, site)?)
            }
            _ => None,
        };
        if let Some(barrier) = barrier {
            engine.event(
                &site,
                SimulationEventKindV1::WorkgroupBarrierArrive { phase },
            )?;
            engine.debug_barrier(site, SimulationDebugBarrierActionV1::Arrive, phase, 1);
            frame.active_operation = None;
            engine.end_lifecycle(
                &site,
                SimulationEventKindV1::OperationEnd {
                    outcome: SimulationExecutionOutcomeV1::Completed,
                },
            )?;
            frame.operation += 1;
            return Ok(FrameAction::Barrier { site, barrier });
        }
        return advance_non_control_operation(engine, frame, block, operation, site);
    }

    let site = terminator_site(frame.function_index, block);
    let terminator = block.terminator.as_ref().ok_or_else(|| {
        engine.at(
            site,
            SimulationExecutionErrorKindV1::MissingTerminator(block.id),
        )
    })?;
    engine.step(&site)?;
    engine.event(&site, SimulationEventKindV1::Terminator)?;
    if let Terminator::Return { values } = terminator {
        resolve_values_into(engine, &frame.values, values, &site, &mut frame.incoming)?;
        release_frame_allocations_observed(engine, &mut frame.allocations, &site)?;
        engine.event(&site, SimulationEventKindV1::Return)?;
        return Ok(FrameAction::Return);
    }
    advance_non_return_terminator(engine, frame, terminator, site)
}

#[inline(never)]
fn advance_wave_frame<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
    pending: &mut Option<WaveInput>,
) -> Result<WaveArrival<'a>, SimulationExecutionErrorV1> {
    let block = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(frame.current)))?;
    let operation = block.operations.get(frame.operation).ok_or_else(|| {
        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
            "wave operation position",
        ))
    })?;
    let OperationKind::Wave(wave) = &operation.kind else {
        return Err(
            engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
                "wave operation dispatch",
            )),
        );
    };
    let site = operation_site(frame.function_index, block, frame.operation);
    engine.step(&site)?;
    engine.begin_lifecycle(&site, SimulationEventKindV1::OperationBegin)?;
    frame.active_operation = Some(site);
    prepare_wave_wait(engine, frame, wave, site, pending)?;
    Ok(WaveArrival { site, wave })
}

#[inline(never)]
fn prepare_wave_wait(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    frame: &RuntimeFrame<'_>,
    wave: &WaveOperation,
    site: CompactSite,
    pending: &mut Option<WaveInput>,
) -> Result<(), SimulationExecutionErrorV1> {
    *pending = Some(match wave.kind {
        WaveOperationKind::LaneId => WaveInput::LaneId,
        WaveOperationKind::Ballot { predicate }
        | WaveOperationKind::Any { predicate }
        | WaveOperationKind::All { predicate } => {
            let predicate = scalar_value(engine, &frame.values, predicate, &site)?
                .as_bool()
                .ok_or_else(|| {
                    engine.at(
                        site,
                        SimulationExecutionErrorKindV1::RuntimeType {
                            value: Some(predicate),
                            expected: "boolean wave predicate",
                        },
                    )
                })?;
            WaveInput::Predicate(predicate)
        }
        WaveOperationKind::ShuffleIndex {
            value,
            source_lane,
            tile_width,
        } => {
            let value = scalar_value(engine, &frame.values, value, &site)?;
            let source = scalar_value(engine, &frame.values, source_lane, &site)?;
            if source.ty() != ScalarType::U32 {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::RuntimeType {
                        value: Some(source_lane),
                        expected: "u32 wave shuffle source lane",
                    },
                ));
            }
            let source_lane = source.bits() as u32;
            if source_lane >= tile_width {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::WaveShuffleSourceOutOfRange {
                        source_lane,
                        tile_width,
                    },
                ));
            }
            WaveInput::Shuffle {
                value,
                source_lane,
                tile_width,
            }
        }
        WaveOperationKind::ReduceF32 { value, .. } => {
            let value_id = value;
            let value = scalar_value(engine, &frame.values, value_id, &site)?;
            if value.ty() != ScalarType::F32 {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::RuntimeType {
                        value: Some(value_id),
                        expected: "f32 wave reduction value",
                    },
                ));
            }
            WaveInput::ReduceF32(value)
        }
        WaveOperationKind::BroadcastF32 {
            value,
            source_lane,
            tile_width,
        } => {
            let value_id = value;
            let value = scalar_value(engine, &frame.values, value_id, &site)?;
            if value.ty() != ScalarType::F32 {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::RuntimeType {
                        value: Some(value_id),
                        expected: "f32 wave broadcast value",
                    },
                ));
            }
            let source = scalar_value(engine, &frame.values, source_lane, &site)?;
            if source.ty() != ScalarType::U32 {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::RuntimeType {
                        value: Some(source_lane),
                        expected: "u32 wave broadcast source lane",
                    },
                ));
            }
            let source_lane = source.bits() as u32;
            if source_lane >= tile_width {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "verified f32 broadcast source bound",
                    ),
                ));
            }
            WaveInput::BroadcastF32 {
                value,
                source_lane,
                tile_width,
            }
        }
    });
    Ok(())
}

#[inline(never)]
fn advance_collective_frame<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
    phase: u64,
    pending: &mut Option<CollectiveInput>,
) -> Result<CollectiveArrival<'a>, SimulationExecutionErrorV1> {
    let block = frame
        .function
        .body
        .as_ref()
        .and_then(|body| body.blocks.get(frame.current_index))
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(frame.current)))?;
    let operation = block.operations.get(frame.operation).ok_or_else(|| {
        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
            "cooperative operation position",
        ))
    })?;
    let site = operation_site(frame.function_index, block, frame.operation);
    engine.step(&site)?;
    engine.begin_lifecycle(&site, SimulationEventKindV1::OperationBegin)?;
    frame.active_operation = Some(site);
    let width = prepare_collective_wait(engine, frame, &operation.kind, site, phase, pending)?;
    Ok(CollectiveArrival {
        site,
        operation: &operation.kind,
        width,
    })
}

fn prepare_collective_wait(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    frame: &RuntimeFrame<'_>,
    operation: &OperationKind,
    site: CompactSite,
    phase: u64,
    pending: &mut Option<CollectiveInput>,
) -> Result<WaveWidth, SimulationExecutionErrorV1> {
    let (width, input) = match operation {
        OperationKind::Matrix(matrix) => (
            matrix_wave_width(matrix),
            match &matrix.kind {
                MatrixOperationKind::LdsLoad { base, .. } => CollectiveInput::MatrixLdsLoad {
                    base: pointer_value(engine, &frame.values, *base, &site)?.clone(),
                },
                MatrixOperationKind::LdsStore { base, values, .. } => {
                    let mut resolved = [ScalarBitsV1::boolean(false); 4];
                    for (destination, value) in resolved.iter_mut().zip(values) {
                        *destination = scalar_value(engine, &frame.values, *value, &site)?;
                    }
                    CollectiveInput::MatrixLdsStore {
                        base: pointer_value(engine, &frame.values, *base, &site)?.clone(),
                        values: resolved,
                    }
                }
                MatrixOperationKind::MultiplyAccumulate { .. } => CollectiveInput::MatrixBf16Exact(
                    matrix_bf16_exact_v1::prepare(engine, frame, matrix, site)?,
                ),
                MatrixOperationKind::ScaledMultiplyAccumulate { .. } => {
                    if crate::matrix_fp4_fp8_exact_v1::supported(matrix) {
                        CollectiveInput::MatrixFp4Fp8Exact(matrix_fp4_fp8_exact_v1::prepare(
                            engine, frame, matrix, site,
                        )?)
                    } else if crate::matrix_fp8_exact_v1::supported(matrix) {
                        CollectiveInput::MatrixFp8Exact(matrix_fp8_exact_v1::prepare(
                            engine, frame, matrix, site,
                        )?)
                    } else {
                        CollectiveInput::MatrixFp4Exact(matrix_fp4_exact_v1::prepare(
                            engine, frame, matrix, site,
                        )?)
                    }
                }
            },
        ),
        OperationKind::Gfx950LdsTranspose(transpose) => {
            let input = match transpose.kind {
                Gfx950LdsTransposeOperationKindV1::Current { format } => {
                    let invocation = engine.invocation.ok_or_else(|| {
                        engine.at(
                            site,
                            SimulationExecutionErrorKindV1::InternalInvariant(
                                "gfx950 transpose workgroup invocation",
                            ),
                        )
                    })?;
                    let participants = invocation
                        .workgroup_size
                        .into_iter()
                        .try_fold(1_u32, u32::checked_mul)
                        .ok_or_else(|| {
                            engine.at(
                                site,
                                SimulationExecutionErrorKindV1::InternalInvariant(
                                    "preflighted gfx950 transpose workgroup size",
                                ),
                            )
                        })?;
                    let tile_bytes = format.lds_bytes();
                    let memory = WorkgroupMemory {
                        element: Type::Scalar(ScalarType::U8),
                        extent: WorkgroupMemoryExtent::Static(tile_bytes * (participants / 64)),
                        alignment: 64,
                    };
                    let mut storage = engine.workgroup_pointer(site, &memory)?;
                    let wave = invocation.local[0] / 64;
                    let wave_base = usize::try_from(wave * tile_bytes).unwrap();
                    let wave_end = usize::try_from((wave + 1) * tile_bytes).unwrap();
                    storage.byte_offset = wave_base;
                    storage.lower_bound = wave_base;
                    storage.upper_bound = wave_end;
                    CollectiveInput::TransposeCurrent { storage }
                }
                Gfx950LdsTransposeOperationKindV1::Stage {
                    storage,
                    source_slice,
                    offset,
                    rows,
                    columns,
                    stride,
                    token_base,
                    reduction_base,
                    ..
                } => CollectiveInput::TransposeStage {
                    storage: pointer_value(engine, &frame.values, storage, &site)?.clone(),
                    source: slice_value(engine, &frame.values, source_slice, &site)?.clone(),
                    offset: index_u64(engine, &frame.values, offset, &site)?,
                    rows: index_u64(engine, &frame.values, rows, &site)?,
                    columns: index_u64(engine, &frame.values, columns, &site)?,
                    stride: index_u64(engine, &frame.values, stride, &site)?,
                    token_base: index_u64(engine, &frame.values, token_base, &site)?,
                    reduction_base: index_u64(engine, &frame.values, reduction_base, &site)?,
                },
                Gfx950LdsTransposeOperationKindV1::Publish { storage, .. } => {
                    engine.event(
                        &site,
                        SimulationEventKindV1::WorkgroupBarrierArrive { phase },
                    )?;
                    engine.debug_barrier(site, SimulationDebugBarrierActionV1::Arrive, phase, 1);
                    CollectiveInput::TransposePublish {
                        storage: pointer_value(engine, &frame.values, storage, &site)?.clone(),
                    }
                }
                Gfx950LdsTransposeOperationKindV1::Read { storage, .. } => {
                    CollectiveInput::TransposeRead {
                        storage: pointer_value(engine, &frame.values, storage, &site)?.clone(),
                    }
                }
            };
            (transpose.width, input)
        }
        OperationKind::Gfx942PhysicalGlobalCopyStep(_)
            if physical_global_copy_v21::is_collective(operation) =>
        {
            (
                WaveWidth::Wave64,
                CollectiveInput::PhysicalEntryPredicate(
                    physical_global_copy_v21::comparison_input(
                        engine,
                        &frame.values,
                        operation,
                        &site,
                    )?,
                ),
            )
        }
        OperationKind::Gfx942PhysicalLdsExchangeStep(_)
            if physical_lds_exchange_v22::is_collective(operation) =>
        {
            (
                WaveWidth::Wave64,
                CollectiveInput::PhysicalEntryPredicate(
                    physical_lds_exchange_v22::comparison_input(
                        engine,
                        &frame.values,
                        operation,
                        &site,
                    )?,
                ),
            )
        }
        OperationKind::Gfx942PhysicalEntryStep(_)
            if physical_entry_v20::is_collective(operation) =>
        {
            (
                WaveWidth::Wave64,
                CollectiveInput::PhysicalEntryPredicate(physical_entry_v20::comparison_input(
                    engine,
                    &frame.values,
                    operation,
                    &site,
                )?),
            )
        }
        _ => {
            return Err(engine.at(
                site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "non-cooperative operation dispatched as collective",
                ),
            ));
        }
    };
    *pending = Some(input);
    Ok(width)
}

const fn matrix_wave_width(matrix: &MatrixOperation) -> WaveWidth {
    match matrix.kind {
        MatrixOperationKind::MultiplyAccumulate { profile, .. }
        | MatrixOperationKind::ScaledMultiplyAccumulate { profile, .. } => profile.wave_width,
        MatrixOperationKind::LdsLoad { profile, .. }
        | MatrixOperationKind::LdsStore { profile, .. } => profile.wave_width,
    }
}

fn execute_wave_f32_reduction(
    kind: WaveF32ReductionKindV1,
    lhs: ScalarBitsV1,
    rhs: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<ScalarBitsV1, SoftFloatErrorV1> {
    match kind {
        WaveF32ReductionKindV1::Sum => {
            crate::soft_float::execute_binary_v1(BinaryOp::Add, lhs, rhs, target)
        }
        WaveF32ReductionKindV1::Maximum => {
            if crate::soft_float::execute_compare_v1(ComparePredicate::LessThan, lhs, rhs)? {
                Ok(rhs)
            } else {
                Ok(lhs)
            }
        }
    }
}

#[inline(never)]
fn advance_non_return_terminator<'a>(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
    terminator: &Terminator,
    site: CompactSite,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    match terminator {
        Terminator::Branch { target, arguments } => {
            resolve_values_into(engine, &frame.values, arguments, &site, &mut frame.incoming)?;
            engine.event(&site, SimulationEventKindV1::Branch { target: *target })?;
            branch_to(engine, frame, *target)
        }
        Terminator::ConditionalBranch {
            condition,
            then_target,
            then_arguments,
            else_target,
            else_arguments,
        } => {
            let take_then = scalar_value(engine, &frame.values, *condition, &site)?
                .as_bool()
                .ok_or_else(|| {
                    engine.at(
                        site,
                        SimulationExecutionErrorKindV1::RuntimeType {
                            value: Some(*condition),
                            expected: "boolean branch condition",
                        },
                    )
                })?;
            let (target, arguments) = if take_then {
                (then_target, then_arguments)
            } else {
                (else_target, else_arguments)
            };
            resolve_values_into(engine, &frame.values, arguments, &site, &mut frame.incoming)?;
            engine.event(&site, SimulationEventKindV1::Branch { target: *target })?;
            branch_to(engine, frame, *target)
        }
        Terminator::Switch {
            selector,
            cases,
            default_target,
            default_arguments,
        } => {
            let selector = scalar_value(engine, &frame.values, *selector, &site)?.bits();
            let SwitchLookup::Index(lookup) =
                &engine.switch_targets[frame.function_index][frame.current_index]
            else {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "preflighted index switch lookup",
                    ),
                ));
            };
            let selected = lookup.get(&selector).and_then(|index| cases.get(*index));
            let (target, arguments) = selected
                .map(|case| (&case.target, &case.arguments))
                .unwrap_or((default_target, default_arguments));
            resolve_values_into(engine, &frame.values, arguments, &site, &mut frame.incoming)?;
            engine.event(&site, SimulationEventKindV1::Branch { target: *target })?;
            branch_to(engine, frame, *target)
        }
        Terminator::IntegerSwitch {
            selector,
            cases,
            default_target,
            default_arguments,
        } => {
            let selector = scalar_value(engine, &frame.values, *selector, &site)?;
            let SwitchLookup::Integer(lookup) =
                &engine.switch_targets[frame.function_index][frame.current_index]
            else {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "preflighted integer switch lookup",
                    ),
                ));
            };
            let selected = lookup
                .get(&(selector.ty(), selector.bits()))
                .and_then(|index| cases.get(*index));
            let (target, arguments) = selected
                .map(|case| (&case.target, &case.arguments))
                .unwrap_or((default_target, default_arguments));
            resolve_values_into(engine, &frame.values, arguments, &site, &mut frame.incoming)?;
            engine.event(&site, SimulationEventKindV1::Branch { target: *target })?;
            branch_to(engine, frame, *target)
        }
        Terminator::Return { .. } => Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant(
                "return reached non-return terminator evaluator",
            ),
        )),
        Terminator::Unreachable => {
            Err(engine.at(site, SimulationExecutionErrorKindV1::ReachedUnreachable))
        }
    }
}

#[inline(never)]
fn advance_non_control_operation<'a>(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
    block: &BasicBlock,
    operation: &Operation,
    site: CompactSite,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    if matches!(operation.kind, OperationKind::Execution(_)) {
        execution_lifecycle_v18::execute_and_bind(engine, frame, operation, site)?;
    } else if physical_lds_exchange_v22::is_operation(&operation.kind) {
        physical_lds_exchange_v22::execute_and_bind(engine, &mut frame.values, operation, &site)?;
    } else if physical_global_copy_v21::is_operation(&operation.kind) {
        physical_global_copy_v21::execute_and_bind(engine, &mut frame.values, operation, &site)?;
    } else if physical_entry_v20::is_operation(&operation.kind) {
        physical_entry_v20::execute_and_bind(engine, &mut frame.values, operation, &site)?;
    } else {
        let results = execute_operation(
            engine,
            frame.function_index,
            block,
            frame.operation,
            operation,
            &frame.values,
            &mut frame.allocations,
        )?;
        bind_small_results(
            engine,
            &mut frame.values,
            &operation.results,
            results,
            &site,
        )?;
    }
    frame.active_operation = None;
    engine.end_lifecycle(
        &site,
        SimulationEventKindV1::OperationEnd {
            outcome: SimulationExecutionOutcomeV1::Completed,
        },
    )?;
    frame.operation += 1;
    Ok(FrameAction::Continue)
}

fn branch_to<'a>(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    frame: &mut RuntimeFrame<'a>,
    target: BlockId,
) -> Result<FrameAction<'a>, SimulationExecutionErrorV1> {
    let target_index = engine.block_indices[frame.function_index]
        .get(&target)
        .copied()
        .ok_or_else(|| engine.fail(SimulationExecutionErrorKindV1::UnknownBlock(target)))?;
    frame.current = target;
    frame.current_index = target_index;
    frame.operation = 0;
    frame.block_entered = false;
    Ok(FrameAction::Continue)
}

fn release_frame_allocations_observed(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    allocations: &mut Vec<FrameAllocation>,
    site: &CompactSite,
) -> Result<(), SimulationExecutionErrorV1> {
    while let Some(allocation) = allocations.pop() {
        let released = engine
            .memory
            .release_one(allocation.id)
            .map_err(|kind| engine.at(*site, kind))?;
        if !released.removed {
            return Err(engine.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "frame allocation remained live until release",
                ),
            ));
        }
        engine.deliver_allocation_lifecycle_v1(released.transition);
        if allocation.lifecycle_observed {
            engine.end_lifecycle(
                site,
                SimulationEventKindV1::AllocationReleased {
                    allocation: allocation.id,
                },
            )?;
        }
    }
    Ok(())
}

fn unwind_frames(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    frames: &mut Vec<RuntimeFrame<'_>>,
    primary: &mut SimulationExecutionErrorV1,
) {
    while let Some(mut frame) = frames.pop() {
        let site = frame.active_operation.unwrap_or(CompactSite {
            function: frame.function_index,
            block: frame.current,
            operation: None,
        });
        while !frame.allocations.is_empty() {
            if let Err(secondary) =
                release_frame_allocations_observed(engine, &mut frame.allocations, &site)
            {
                attach_observation_failure(primary, secondary);
            }
        }
        if let Some(operation_site) = frame.active_operation
            && let Err(secondary) = engine.end_lifecycle(
                &operation_site,
                SimulationEventKindV1::OperationEnd {
                    outcome: SimulationExecutionOutcomeV1::Failed,
                },
            )
        {
            attach_observation_failure(primary, secondary);
        }
    }
}

fn attach_observation_failure(
    primary: &mut SimulationExecutionErrorV1,
    secondary: SimulationExecutionErrorV1,
) {
    if primary.observation_failure.is_none() {
        primary.observation_failure = Some(SimulationObservationFailureV1 {
            invocation: secondary.invocation,
            site: secondary.site,
            kind: secondary.kind,
        });
    }
}

enum SmallResults<T> {
    None,
    One(T),
    Two(T, T),
}

impl<T> SmallResults<T> {
    const fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
        }
    }
}

fn bind_small_results(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &mut RuntimeValues<'_>,
    definitions: &[ValueDef],
    results: SmallResults<RuntimeValue>,
    site: &CompactSite,
) -> Result<(), SimulationExecutionErrorV1> {
    if definitions.len() != results.len() {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::ResultArity {
                expected: definitions.len(),
                actual: results.len(),
            },
        ));
    }
    match (definitions, results) {
        ([], SmallResults::None) => Ok(()),
        ([definition], SmallResults::One(value)) => {
            bind_typed_value(engine, values, definition, value, site)
        }
        ([first, second], SmallResults::Two(first_value, second_value)) => {
            bind_typed_value(engine, values, first, first_value, site)?;
            bind_typed_value(engine, values, second, second_value, site)
        }
        _ => Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::InternalInvariant("small result arity dispatch"),
        )),
    }
}

fn bind_dynamic_results(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &mut RuntimeValues<'_>,
    definitions: &[ValueDef],
    results: &[RuntimeValue],
    site: &CompactSite,
) -> Result<(), SimulationExecutionErrorV1> {
    if definitions.len() != results.len() {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::ResultArity {
                expected: definitions.len(),
                actual: results.len(),
            },
        ));
    }
    for (definition, value) in definitions.iter().zip(results) {
        bind_typed_value(engine, values, definition, value.clone(), site)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_operation(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    function_index: usize,
    block: &BasicBlock,
    ordinal: usize,
    operation: &Operation,
    values: &RuntimeValues<'_>,
    frame_allocations: &mut Vec<FrameAllocation>,
) -> Result<SmallResults<RuntimeValue>, SimulationExecutionErrorV1> {
    // Choose the execution frame before entering either implementation. Calling the
    // assembly helper from the large non-assembly match would combine their debug
    // stack frames, even though an assembly operation needs only the small helper.
    if matches!(&operation.kind, OperationKind::InlineAssembly(_)) {
        let site = operation_site(function_index, block, ordinal);
        execute_inline_assembly(engine, values, operation, &site)
    } else if matches!(&operation.kind, OperationKind::Gfx942OrderedRegion(_)) {
        let site = operation_site(function_index, block, ordinal);
        ordered_region_v16::execute(engine, values, operation, &site)
    } else if matches!(&operation.kind, OperationKind::Gfx942OrderedProgram(_)) {
        let site = operation_site(function_index, block, ordinal);
        ordered_program_v17::execute(engine, values, operation, &site)
    } else if matches!(
        &operation.kind,
        OperationKind::Gfx942CompleteBodyDeclaration(_) | OperationKind::Gfx942CompleteBodyStep(_)
    ) {
        let site = operation_site(function_index, block, ordinal);
        complete_body_v19::execute(engine, values, operation, &site)
    } else if matches!(&operation.kind, OperationKind::Alloca { .. }) {
        let site = operation_site(function_index, block, ordinal);
        alloca_v1::execute(engine, values, operation, site, frame_allocations)
    } else {
        execute_non_assembly_operation(
            engine,
            function_index,
            block,
            ordinal,
            operation,
            values,
            frame_allocations,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn execute_non_assembly_operation(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    function_index: usize,
    block: &BasicBlock,
    ordinal: usize,
    operation: &Operation,
    values: &RuntimeValues<'_>,
    _frame_allocations: &mut Vec<FrameAllocation>,
) -> Result<SmallResults<RuntimeValue>, SimulationExecutionErrorV1> {
    let site = operation_site(function_index, block, ordinal);
    let one = |value| Ok(SmallResults::One(value));
    match &operation.kind {
        OperationKind::Constant(_)
        | OperationKind::Intrinsic(_)
        | OperationKind::Unary { .. }
        | OperationKind::Binary { .. }
        | OperationKind::Compare { .. }
        | OperationKind::Cast { .. }
        | OperationKind::Select { .. } => {
            value_operation_v1::execute(engine, values, operation, site)
        }
        OperationKind::Call {
            callee: _,
            arguments,
        } => {
            let block_index = engine.block_indices[function_index]
                .get(&block.id)
                .copied()
                .ok_or_else(|| {
                    engine.at(
                        site,
                        SimulationExecutionErrorKindV1::InternalInvariant(
                            "preflighted operation block index",
                        ),
                    )
                })?;
            match engine.call_targets[function_index][block_index][ordinal] {
                CallTarget::Float(operation) => {
                    execute_float_call(engine, values, operation, arguments, site)
                }
                CallTarget::Trap => {
                    Err(engine.at(site, SimulationExecutionErrorKindV1::ReachedUnreachable))
                }
                CallTarget::NotCall | CallTarget::Internal(_) => Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "non-leaf call reached non-recursive operation evaluator",
                    ),
                )),
            }
        }
        OperationKind::Alloca { .. } => Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant(
                "outlined private allocation dispatch",
            ),
        )),
        OperationKind::SliceLength { .. }
        | OperationKind::SliceData { .. }
        | OperationKind::GetElementPointer { .. } => {
            pointer_view_operation_v1::execute(engine, values, operation, site)
        }
        OperationKind::Load { .. }
        | OperationKind::GuardedLoad { .. }
        | OperationKind::Store { .. }
        | OperationKind::GuardedStore { .. } => {
            scalar_memory_operation_v1::execute(engine, values, operation, site)
        }
        OperationKind::WorkgroupMemory(memory) => one(RuntimeValue::Pointer(
            engine.workgroup_pointer(site, memory)?,
        )),
        OperationKind::Atomic(atomic) => execute_atomic(engine, values, atomic, &site),
        OperationKind::Fence(fence) => {
            if fence.semantics.ordering != MemoryOrdering::Relaxed {
                engine.unmodeled_atomic_or_fence_happens_before = true;
            }
            let address_space_mask = address_space_mask(&fence.semantics.address_spaces);
            engine.event(
                &site,
                SimulationEventKindV1::MemoryFence {
                    memory_scope: fence.memory_scope,
                    ordering: fence.semantics.ordering,
                    address_space_mask,
                },
            )?;
            engine.debug_fence(site, fence);
            Ok(SmallResults::None)
        }
        OperationKind::MemoryIntrinsic(intrinsic) => {
            execute_memory_intrinsic(engine, values, intrinsic, &site)
        }
        OperationKind::Storage(_) => storage_scalar_v18::execute(engine, values, operation, site),
        OperationKind::Execution(_)
        | OperationKind::Barrier(_)
        | OperationKind::WorkgroupBarrier(_)
        | OperationKind::Matrix(_)
        | OperationKind::Wave(_)
        | OperationKind::Gfx950LdsTranspose(_)
        | OperationKind::InlineAssembly(_)
        | OperationKind::Gfx942OrderedRegion(_)
        | OperationKind::Gfx942OrderedProgram(_)
        | OperationKind::Gfx942CompleteBodyDeclaration(_)
        | OperationKind::Gfx942CompleteBodyStep(_)
        | OperationKind::Gfx942PhysicalEntryDeclaration(_)
        | OperationKind::Gfx942PhysicalEntryStep(_)
        | OperationKind::Gfx942PhysicalGlobalCopyDeclaration(_)
        | OperationKind::Gfx942PhysicalGlobalCopyStep(_)
        | OperationKind::Gfx942PhysicalLdsExchangeDeclaration(_)
        | OperationKind::Gfx942PhysicalLdsExchangeStep(_)
        | OperationKind::VectorLoad(_)
        | OperationKind::VectorStore(_)
        | OperationKind::VectorLayoutConvert(_)
        | OperationKind::VerificationContract(_) => Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant(
                "unsupported operation passed preflight",
            ),
        )),
    }
}

// Keep contract-validation and error temporaries out of the dispatcher's debug-build
// frame: unrelated operations must remain executable on the bounded 256 KiB stack.
#[inline(never)]
fn execute_inline_assembly(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    operation: &Operation,
    site: &CompactSite,
) -> Result<SmallResults<RuntimeValue>, SimulationExecutionErrorV1> {
    let validated =
        validate_gfx942_inline_assembly_v1(operation, |value| match values.get(&value) {
            Some(RuntimeValue::Scalar(scalar)) => Some(scalar.ty()),
            _ => None,
        })
        .map_err(|_| {
            engine.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "integer assembly contract changed after preflight",
                ),
            )
        })?;
    if validated.instruction().constraint() != fe2o3_kernel_ir::AssemblyConstraint::Vgpr32 {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::InternalInvariant(
                "scalar-register assembly passed preflight",
            ),
        ));
    }
    let mut inputs = [0_u32; 2];
    for (index, input) in validated.inputs().iter().enumerate() {
        inputs[index] = scalar_value(engine, values, *input, site)?.bits() as u32;
    }
    let bits = validated
        .instruction()
        .evaluate_bits(&inputs[..validated.inputs().len()])
        .map_err(|_| {
            engine.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "validated integer assembly input arity",
                ),
            )
        })?;
    Ok(SmallResults::One(RuntimeValue::Scalar(
        ScalarBitsV1::new(validated.scalar_type(), u128::from(bits), engine.target).map_err(
            |_| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "validated integer assembly result bits",
                    ),
                )
            },
        )?,
    )))
}

fn execute_memory_intrinsic(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    intrinsic: &MemoryIntrinsicOperation,
    site: &CompactSite,
) -> Result<SmallResults<RuntimeValue>, SimulationExecutionErrorV1> {
    match intrinsic {
        MemoryIntrinsicOperation::PointerDistance {
            pointer,
            origin,
            kind,
            unit,
            layout,
            ..
        } => {
            let pointer_runtime = pointer_value(engine, values, *pointer, site)?.clone();
            let origin_value = pointer_value(engine, values, *origin, site)?.clone();
            if pointer_runtime.allocation != origin_value.allocation {
                return Err(engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::PointerDistanceDifferentAllocation {
                        pointer_allocation: pointer_runtime.allocation,
                        origin_allocation: origin_value.allocation,
                    },
                ));
            }
            let allocation_bytes = match engine.memory.allocation(&pointer_runtime) {
                Ok(allocation) => allocation.bytes.len(),
                Err(error) => {
                    return Err(engine.memory_error_at(*site, &pointer_runtime, error));
                }
            };
            let in_bounds_or_one_past = |value: &PointerValue| {
                value.byte_offset >= value.lower_bound
                    && value.byte_offset <= value.upper_bound
                    && value.byte_offset <= allocation_bytes
            };
            if !in_bounds_or_one_past(&pointer_runtime) || !in_bounds_or_one_past(&origin_value) {
                return Err(engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::PointerDistanceOutOfBounds,
                ));
            }
            let difference =
                (pointer_runtime.byte_offset as i128) - (origin_value.byte_offset as i128);
            let unit_bytes = match unit {
                PointerDistanceUnit::Bytes => 1_i128,
                PointerDistanceUnit::Elements => i128::from(layout.size_bytes),
            };
            if difference % unit_bytes != 0 {
                return Err(engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::PointerDistanceNotDivisible {
                        byte_difference: difference.unsigned_abs() as u64,
                        unit_bytes: unit_bytes as u64,
                    },
                ));
            }
            let distance = difference / unit_bytes;
            let (isize_min, isize_max) = match engine.target.index_width() {
                IndexWidthV1::Bits32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
                IndexWidthV1::Bits64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
            };
            if distance < isize_min || distance > isize_max {
                return Err(engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::PointerDistanceOverflow,
                ));
            }
            let result = match kind {
                PointerDistanceKind::Signed => ScalarBitsV1::new(
                    ScalarType::I64,
                    u128::from(distance as i64 as u64),
                    engine.target,
                ),
                PointerDistanceKind::Unsigned if distance < 0 => {
                    return Err(engine.at(
                        *site,
                        SimulationExecutionErrorKindV1::PointerDistanceNegativeUnsigned,
                    ));
                }
                PointerDistanceKind::Unsigned => {
                    ScalarBitsV1::index(distance as u64, engine.target)
                }
            }
            .map_err(|_| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::PointerDistanceOverflow,
                )
            })?;
            Ok(SmallResults::One(RuntimeValue::Scalar(result)))
        }
        MemoryIntrinsicOperation::VolatileLoad {
            pointer,
            address_space,
            layout,
            ..
        } => Ok(SmallResults::One(RuntimeValue::Scalar(
            execute_scalar_load(
                engine,
                values,
                *pointer,
                MemoryAccess::new(*address_space, layout.alignment_bytes),
                site,
            )?,
        ))),
        MemoryIntrinsicOperation::VolatileStore {
            pointer,
            value,
            address_space,
            layout,
            ..
        } => {
            let pointer_value = pointer_value(engine, values, *pointer, site)?.clone();
            let stored = scalar_value(engine, values, *value, site)?;
            let access = MemoryAccess::new(*address_space, layout.alignment_bytes);
            let bytes = engine
                .memory
                .validate_store(&pointer_value, access, stored, engine.target)
                .map_err(|kind| engine.memory_error_at(*site, &pointer_value, kind))?;
            if pointer_value.address_space == AddressSpace::Global {
                engine.record_access(
                    site,
                    pointer_value.allocation,
                    pointer_value.byte_offset,
                    bytes,
                    true,
                    None,
                )?;
            }
            engine.observe_and_commit_store(site, &pointer_value, stored, bytes)?;
            Ok(SmallResults::None)
        }
        MemoryIntrinsicOperation::CopyNonOverlapping {
            source,
            destination,
            count,
            layout,
            ..
        } => {
            let source = pointer_value(engine, values, *source, site)?.clone();
            let destination = pointer_value(engine, values, *destination, site)?.clone();
            let count = scalar_nonnegative_usize(
                scalar_value(engine, values, *count, site)?,
                engine.target,
            )
            .map_err(|kind| engine.at(*site, kind))?;
            let element_bytes = usize::try_from(layout.size_bytes).map_err(|_| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::MemoryIntrinsicByteCountOverflow,
                )
            })?;
            let bytes = count
                .checked_mul(element_bytes)
                .filter(|bytes| match engine.target.index_width() {
                    IndexWidthV1::Bits32 => *bytes <= u32::MAX as usize,
                    IndexWidthV1::Bits64 => true,
                })
                .ok_or_else(|| {
                    engine.at(
                        *site,
                        SimulationExecutionErrorKindV1::MemoryIntrinsicByteCountOverflow,
                    )
                })?;
            if bytes == 0 {
                engine
                    .memory
                    .validate_zero_byte_pointer(&source, layout.alignment_bytes)
                    .map_err(|kind| engine.memory_error_at(*site, &source, kind))?;
                engine
                    .memory
                    .validate_zero_byte_pointer(&destination, layout.alignment_bytes)
                    .map_err(|kind| engine.memory_error_at(*site, &destination, kind))?;
                return Ok(SmallResults::None);
            }
            let invocation = engine.invocation.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("copy invocation"),
                )
            })?;
            engine
                .memory
                .validate_range_read(&source, layout.alignment_bytes, bytes, invocation)
                .map_err(|kind| engine.memory_error_at(*site, &source, kind))?;
            engine
                .memory
                .validate_range_write(&destination, layout.alignment_bytes, bytes)
                .map_err(|kind| engine.memory_error_at(*site, &destination, kind))?;
            let source_end = source.byte_offset + bytes;
            let destination_end = destination.byte_offset + bytes;
            if source.allocation == destination.allocation
                && source.byte_offset < destination_end
                && destination.byte_offset < source_end
            {
                return Err(engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::CopyRangesOverlap {
                        allocation: source.allocation,
                        source_offset: source.byte_offset,
                        destination_offset: destination.byte_offset,
                        bytes,
                    },
                ));
            }
            let workgroup_writer = if destination.address_space == AddressSpace::Workgroup {
                Some(
                    invocation_local_ordinal(invocation)
                        .and_then(|ordinal| ordinal.checked_add(1))
                        .ok_or_else(|| {
                            engine.at(
                                *site,
                                SimulationExecutionErrorKindV1::InternalInvariant(
                                    "workgroup invocation ordinal",
                                ),
                            )
                        })?,
                )
            } else {
                None
            };
            engine.charge_steps(site, bytes)?;
            if source.address_space == AddressSpace::Global {
                engine.record_access(
                    site,
                    source.allocation,
                    source.byte_offset,
                    bytes,
                    false,
                    None,
                )?;
            }
            engine.event(
                site,
                SimulationEventKindV1::MemoryRead {
                    allocation: source.allocation,
                    offset: source.byte_offset,
                    bytes,
                },
            )?;
            if destination.address_space == AddressSpace::Global {
                engine.record_access(
                    site,
                    destination.allocation,
                    destination.byte_offset,
                    bytes,
                    true,
                    None,
                )?;
            }
            engine.event(
                site,
                SimulationEventKindV1::MemoryWrite {
                    allocation: destination.allocation,
                    offset: destination.byte_offset,
                    bytes,
                },
            )?;
            engine
                .memory
                .commit_copy(&source, &destination, bytes, workgroup_writer);
            Ok(SmallResults::None)
        }
    }
}

fn pointer_value<'a>(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &'a RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
) -> Result<&'a PointerValue, SimulationExecutionErrorV1> {
    match runtime_non_scalar_value(engine, values, id, site, "pointer")? {
        RuntimeValue::Pointer(pointer) => Ok(pointer),
        _ => Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(id),
                expected: "pointer",
            },
        )),
    }
}

fn slice_value<'a>(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &'a RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
) -> Result<&'a SliceValue, SimulationExecutionErrorV1> {
    match runtime_non_scalar_value(engine, values, id, site, "slice")? {
        RuntimeValue::Slice(slice) => Ok(slice),
        _ => Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(id),
                expected: "slice",
            },
        )),
    }
}

fn index_u64(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
) -> Result<u64, SimulationExecutionErrorV1> {
    let value = scalar_value(engine, values, id, site)?;
    if value.ty() != ScalarType::Index {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(id),
                expected: "index",
            },
        ));
    }
    u64::try_from(value.bits())
        .map_err(|_| engine.at(*site, SimulationExecutionErrorKindV1::IntegerOutOfRange))
}

#[inline(never)]
fn execute_float_call(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    operation: SoftFloatOperationV1,
    arguments: &[ValueId],
    site: CompactSite,
) -> Result<SmallResults<RuntimeValue>, SimulationExecutionErrorV1> {
    if arguments.len() > 3 {
        return Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant("bounded float operation arity"),
        ));
    }
    let mut operands = [ScalarBitsV1::boolean(false); 3];
    for (destination, value) in operands.iter_mut().zip(arguments) {
        *destination = scalar_value(engine, values, *value, &site)?;
    }
    Ok(SmallResults::One(RuntimeValue::Scalar(
        crate::soft_float::execute_compact_operation_v1(
            operation,
            &operands[..arguments.len()],
            engine.target,
        )
        .map_err(map_soft_float_error)
        .map_err(|kind| engine.at(site, kind))?,
    )))
}

fn execute_atomic(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    atomic: &Atomic,
    site: &CompactSite,
) -> Result<SmallResults<RuntimeValue>, SimulationExecutionErrorV1> {
    let RuntimeValue::Pointer(pointer) =
        runtime_non_scalar_value(engine, values, atomic.pointer, site, "atomic pointer")?
    else {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(atomic.pointer),
                expected: "atomic pointer",
            },
        ));
    };
    let pointer = pointer.clone();
    let operand = atomic
        .value
        .map(|value| scalar_value(engine, values, value, site))
        .transpose()?;
    let compare = atomic
        .compare
        .map(|value| scalar_value(engine, values, value, site))
        .transpose()?;
    let invocation = engine.invocation.ok_or_else(|| {
        engine.at(
            *site,
            SimulationExecutionErrorKindV1::InternalInvariant("atomic invocation"),
        )
    })?;
    if atomic.ordering != MemoryOrdering::Relaxed
        || atomic
            .failure_ordering
            .is_some_and(|ordering| ordering != MemoryOrdering::Relaxed)
    {
        engine.unmodeled_atomic_or_fence_happens_before = true;
    }

    let previous = if atomic.kind == AtomicKind::Store {
        None
    } else {
        Some(
            engine
                .memory
                .load(&pointer, atomic.access, engine.target, invocation)
                .map_err(|kind| engine.memory_error_at(*site, &pointer, kind))?,
        )
    };
    let (committed, compare_exchange_success) = match atomic.kind {
        AtomicKind::Load => (None, None),
        AtomicKind::Store => (operand, None),
        AtomicKind::CompareExchange => {
            let old = previous.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("atomic compare old value"),
                )
            })?;
            let expected = compare.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("atomic compare operand"),
                )
            })?;
            let success = old == expected;
            (success.then_some(operand).flatten(), Some(success))
        }
        kind => {
            let old = previous.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("atomic RMW old value"),
                )
            })?;
            let operand = operand.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("atomic RMW operand"),
                )
            })?;
            (
                Some(
                    atomic_rmw_value(kind, old, operand, engine.target)
                        .map_err(|kind| engine.at(*site, kind))?,
                ),
                None,
            )
        }
    };

    let width = if let Some(value) = committed {
        engine
            .memory
            .validate_store(&pointer, atomic.access, value, engine.target)
            .map_err(|kind| engine.memory_error_at(*site, &pointer, kind))?
    } else {
        engine.target.scalar_bytes(pointer.element).ok_or_else(|| {
            engine.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant(
                    "preflighted atomic scalar width",
                ),
            )
        })?
    };

    if pointer.address_space == AddressSpace::Global {
        engine.record_access(
            site,
            pointer.allocation,
            pointer.byte_offset,
            width,
            committed.is_some(),
            Some(AtomicAccess::new(atomic.scope, pointer.byte_offset, width)),
        )?;
    }
    engine.event(
        site,
        SimulationEventKindV1::MemoryAtomic {
            allocation: pointer.allocation,
            offset: pointer.byte_offset,
            bytes: width,
            kind: atomic.kind,
            previous,
            committed,
            compare_exchange_success,
            scope: atomic.scope,
            ordering: atomic.ordering,
            failure_ordering: atomic.failure_ordering,
        },
    )?;

    let debug_access = match (previous, committed) {
        (Some(_), Some(_)) => SimulationDebugMemoryAccessV1::AtomicReadWriteCommitted,
        (Some(_), None) => SimulationDebugMemoryAccessV1::AtomicRead,
        (None, Some(_)) => SimulationDebugMemoryAccessV1::AtomicWriteCommitted,
        (None, None) => {
            return Err(engine.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant("empty atomic effect"),
            ));
        }
    };
    let debug_value = committed.or(previous).ok_or_else(|| {
        engine.at(
            *site,
            SimulationExecutionErrorKindV1::InternalInvariant("atomic debug value"),
        )
    })?;
    if let Some(value) = committed {
        let prepared = match engine.memory.prepare_store(&pointer, width) {
            Ok(prepared) => prepared,
            Err(kind) => {
                return Err(SimulationExecutionErrorV1 {
                    invocation: engine.invocation,
                    site: Some(engine.materialize_site(*site)),
                    kind,
                    observation_failure: None,
                });
            }
        };
        prepared.commit(value);
        engine
            .memory
            .mark_workgroup_atomic(&pointer, width)
            .map_err(|kind| engine.at(*site, kind))?;
    }
    engine.debug_memory(*site, debug_access, &pointer, width, debug_value);

    Ok(match atomic.kind {
        AtomicKind::Store => SmallResults::None,
        AtomicKind::CompareExchange => {
            let old = previous.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "compare-exchange result old value",
                    ),
                )
            })?;
            let success = compare_exchange_success.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "compare-exchange result outcome",
                    ),
                )
            })?;
            SmallResults::Two(
                RuntimeValue::Scalar(old),
                RuntimeValue::Scalar(ScalarBitsV1::boolean(success)),
            )
        }
        _ => SmallResults::One(RuntimeValue::Scalar(previous.ok_or_else(|| {
            engine.at(
                *site,
                SimulationExecutionErrorKindV1::InternalInvariant("atomic result old value"),
            )
        })?)),
    })
}

fn atomic_rmw_value(
    kind: AtomicKind,
    old: ScalarBitsV1,
    operand: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<ScalarBitsV1, SimulationExecutionErrorKindV1> {
    if old.ty() != operand.ty() || !old.ty().is_integer() || old.ty() == ScalarType::Index {
        return Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "preflighted integer atomic operands",
        ));
    }
    let width = scalar_width(old, target)?;
    let bits = match kind {
        AtomicKind::Exchange => operand.bits(),
        AtomicKind::Add => old.bits().wrapping_add(operand.bits()) & mask(width),
        AtomicKind::Subtract => old.bits().wrapping_sub(operand.bits()) & mask(width),
        AtomicKind::Min if old.ty().is_signed_integer() => {
            if signed_value(old, target)? <= signed_value(operand, target)? {
                old.bits()
            } else {
                operand.bits()
            }
        }
        AtomicKind::Max if old.ty().is_signed_integer() => {
            if signed_value(old, target)? >= signed_value(operand, target)? {
                old.bits()
            } else {
                operand.bits()
            }
        }
        AtomicKind::Min => old.bits().min(operand.bits()),
        AtomicKind::Max => old.bits().max(operand.bits()),
        AtomicKind::BitAnd => old.bits() & operand.bits(),
        AtomicKind::BitOr => old.bits() | operand.bits(),
        AtomicKind::BitXor => old.bits() ^ operand.bits(),
        AtomicKind::Load | AtomicKind::Store | AtomicKind::CompareExchange => {
            return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                "non-RMW atomic dispatch",
            ));
        }
    };
    ScalarBitsV1::new(old.ty(), bits & mask(width), target)
        .map_err(|_| SimulationExecutionErrorKindV1::InternalInvariant("atomic result bits"))
}

fn address_space_mask(address_spaces: &std::collections::BTreeSet<AddressSpace>) -> u8 {
    address_spaces.iter().fold(0_u8, |mask, address_space| {
        let bit = match address_space {
            AddressSpace::Private => 0,
            AddressSpace::Workgroup => 1,
            AddressSpace::Global => 2,
            AddressSpace::Constant => 3,
            AddressSpace::Generic => 4,
        };
        mask | (1_u8 << bit)
    })
}

fn execute_scalar_load(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    pointer: ValueId,
    access: MemoryAccess,
    site: &CompactSite,
) -> Result<ScalarBitsV1, SimulationExecutionErrorV1> {
    let RuntimeValue::Pointer(pointer_value) =
        runtime_non_scalar_value(engine, values, pointer, site, "pointer")?
    else {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(pointer),
                expected: "pointer",
            },
        ));
    };
    execute_pointer_load(engine, pointer_value, access, site)
}

fn execute_pointer_load(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    pointer: &PointerValue,
    access: MemoryAccess,
    site: &CompactSite,
) -> Result<ScalarBitsV1, SimulationExecutionErrorV1> {
    let value = engine
        .memory
        .load(
            pointer,
            access,
            engine.target,
            engine.invocation.ok_or_else(|| {
                engine.at(
                    *site,
                    SimulationExecutionErrorKindV1::InternalInvariant("load invocation"),
                )
            })?,
        )
        .map_err(|kind| engine.memory_error_at(*site, pointer, kind))?;
    let bytes = engine.target.scalar_bytes(pointer.element).ok_or_else(|| {
        engine.at(
            *site,
            SimulationExecutionErrorKindV1::InternalInvariant("preflighted load element"),
        )
    })?;
    if pointer.address_space == AddressSpace::Global {
        engine.record_access(
            site,
            pointer.allocation,
            pointer.byte_offset,
            bytes,
            false,
            None,
        )?;
    }
    engine.event(
        site,
        SimulationEventKindV1::MemoryRead {
            allocation: pointer.allocation,
            offset: pointer.byte_offset,
            bytes,
        },
    )?;
    engine.debug_memory(
        *site,
        SimulationDebugMemoryAccessV1::Read,
        pointer,
        bytes,
        value,
    );
    Ok(value)
}

fn execute_pointer_store(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    pointer: &PointerValue,
    value: ScalarBitsV1,
    access: MemoryAccess,
    site: &CompactSite,
) -> Result<(), SimulationExecutionErrorV1> {
    let bytes = engine
        .memory
        .validate_store(pointer, access, value, engine.target)
        .map_err(|kind| engine.memory_error_at(*site, pointer, kind))?;
    if pointer.address_space == AddressSpace::Global {
        engine.record_access(
            site,
            pointer.allocation,
            pointer.byte_offset,
            bytes,
            true,
            None,
        )?;
    }
    engine.observe_and_commit_store(site, pointer, value, bytes)
}

fn pointer_at_byte(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    pointer: &PointerValue,
    byte_delta: usize,
    element: ScalarType,
    site: &CompactSite,
) -> Result<PointerValue, SimulationExecutionErrorV1> {
    let byte_offset = pointer
        .byte_offset
        .checked_add(byte_delta)
        .ok_or_else(|| engine.at(*site, SimulationExecutionErrorKindV1::PointerOffsetOverflow))?;
    Ok(PointerValue {
        byte_offset,
        element,
        ..pointer.clone()
    })
}

fn matrix_lds_pointer(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    base: &PointerValue,
    profile: MatrixLdsProfile,
    lane: u32,
    component: u32,
) -> Result<PointerValue, SimulationExecutionErrorV1> {
    let row = lane & 15;
    let chunk = lane >> 4;
    let column = chunk * 4 + component;
    let physical_column = column ^ ((row & 3) << 2);
    let element_index = row * 16 + physical_column;
    let element = match profile.element {
        MatrixElement::Bf16 => ScalarType::Bf16,
        MatrixElement::F32 => ScalarType::F32,
        MatrixElement::Fp4E2M1 | MatrixElement::Fp8E4M3 => ScalarType::U8,
    };
    let element_bytes = engine.target.scalar_bytes(element).ok_or_else(|| {
        engine.at(
            *site,
            SimulationExecutionErrorKindV1::InternalInvariant("verified matrix LDS element layout"),
        )
    })?;
    let byte_delta = usize::try_from(element_index)
        .ok()
        .and_then(|index| index.checked_mul(element_bytes))
        .ok_or_else(|| engine.at(*site, SimulationExecutionErrorKindV1::PointerOffsetOverflow))?;
    pointer_at_byte(engine, base, byte_delta, element, site)
}

fn execute_matrix_lds_load(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    base: &PointerValue,
    profile: MatrixLdsProfile,
    lane: u32,
) -> Result<[ScalarBitsV1; 4], SimulationExecutionErrorV1> {
    let mut values = [ScalarBitsV1::boolean(false); 4];
    for (component, value) in values.iter_mut().enumerate() {
        let pointer = matrix_lds_pointer(engine, site, base, profile, lane, component as u32)?;
        *value = execute_pointer_load(
            engine,
            &pointer,
            MemoryAccess::new(AddressSpace::Workgroup, profile.required_alignment()),
            site,
        )?;
    }
    Ok(values)
}

fn execute_matrix_lds_store(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    base: &PointerValue,
    profile: MatrixLdsProfile,
    lane: u32,
    values: [ScalarBitsV1; 4],
) -> Result<(), SimulationExecutionErrorV1> {
    for (component, value) in values.into_iter().enumerate() {
        let pointer = matrix_lds_pointer(engine, site, base, profile, lane, component as u32)?;
        execute_pointer_store(
            engine,
            &pointer,
            value,
            MemoryAccess::new(AddressSpace::Workgroup, profile.required_alignment()),
            site,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_transpose_stage(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    format: Gfx950LdsTransposeFormatV1,
    lane: u32,
    storage: &PointerValue,
    source: &SliceValue,
    offset: u64,
    rows: u64,
    columns: u64,
    stride: u64,
    token_base: u64,
    reduction_base: u64,
) -> Result<(), SimulationExecutionErrorV1> {
    let lane_in_group = lane & 15;
    let group = lane >> 4;
    match format {
        Gfx950LdsTransposeFormatV1::Fp4E2M1 => {
            let depth_base = u64::from(group * 32 + lane_in_group);
            for part in 0..2_u32 {
                let depth_delta = depth_base + u64::from(part * 16);
                for byte in 0..8_u32 {
                    let low = guarded_transpose_source_byte(
                        engine,
                        site,
                        source,
                        offset,
                        rows,
                        columns,
                        stride,
                        token_base,
                        reduction_base,
                        u64::from(byte * 2),
                        depth_delta,
                    )? & 0x0f;
                    let high = guarded_transpose_source_byte(
                        engine,
                        site,
                        source,
                        offset,
                        rows,
                        columns,
                        stride,
                        token_base,
                        reduction_base,
                        u64::from(byte * 2 + 1),
                        depth_delta,
                    )? & 0x0f;
                    store_transpose_byte(
                        engine,
                        site,
                        storage,
                        lane,
                        format,
                        part,
                        byte,
                        low | (high << 4),
                    )?;
                }
            }
        }
        Gfx950LdsTransposeFormatV1::Fp8E4M3 => {
            let token_band = u64::from((lane_in_group & 1) * 8);
            let depth_base = u64::from(group * 16 + (lane_in_group >> 1));
            for (part, depth_offset) in [0_u64, 8, 64, 72].into_iter().enumerate() {
                for byte in 0..8_u32 {
                    let value = guarded_transpose_source_byte(
                        engine,
                        site,
                        source,
                        offset,
                        rows,
                        columns,
                        stride,
                        token_base,
                        reduction_base,
                        token_band + u64::from(byte),
                        depth_base + depth_offset,
                    )?;
                    store_transpose_byte(
                        engine,
                        site,
                        storage,
                        lane,
                        format,
                        part as u32,
                        byte,
                        value,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn guarded_transpose_source_byte(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    source: &SliceValue,
    offset: u64,
    rows: u64,
    columns: u64,
    stride: u64,
    token_base: u64,
    reduction_base: u64,
    row_delta: u64,
    column_delta: u64,
) -> Result<u8, SimulationExecutionErrorV1> {
    let Some(row) = token_base.checked_add(row_delta) else {
        return Ok(0);
    };
    let Some(column) = reduction_base.checked_add(column_delta) else {
        return Ok(0);
    };
    if row >= rows || column >= columns {
        return Ok(0);
    }
    let Some(index) = row
        .checked_mul(stride)
        .and_then(|linear| linear.checked_add(column))
        .and_then(|linear| offset.checked_add(linear))
    else {
        return Ok(0);
    };
    let Ok(index) = usize::try_from(index) else {
        return Ok(0);
    };
    if index >= source.elements {
        return Ok(0);
    }
    let base = PointerValue {
        exposed_generic: source.exposed_generic,
        allocation: source.allocation,
        byte_offset: source.byte_offset,
        element: ScalarType::U8,
        address_space: source.address_space,
        access: source.access,
        lower_bound: source.byte_offset,
        upper_bound: source.byte_offset + source.byte_len,
        abi_argument_ordinal: source.abi_argument_ordinal,
    };
    let pointer = pointer_at_byte(engine, &base, index, ScalarType::U8, site)?;
    let value = execute_pointer_load(
        engine,
        &pointer,
        MemoryAccess::new(AddressSpace::Global, 1),
        site,
    )?;
    Ok(value.bits() as u8)
}

#[allow(clippy::too_many_arguments)]
fn store_transpose_byte(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    storage: &PointerValue,
    lane: u32,
    format: Gfx950LdsTransposeFormatV1,
    part: u32,
    byte: u32,
    value: u8,
) -> Result<(), SimulationExecutionErrorV1> {
    let byte_delta = usize::try_from(lane * format.lane_byte_stride() + part * 8 + byte)
        .map_err(|_| engine.at(*site, SimulationExecutionErrorKindV1::PointerOffsetOverflow))?;
    let pointer = pointer_at_byte(engine, storage, byte_delta, ScalarType::U8, site)?;
    let value = ScalarBitsV1::new(ScalarType::U8, u128::from(value), engine.target)
        .map_err(|_| engine.at(*site, SimulationExecutionErrorKindV1::IntegerOutOfRange))?;
    execute_pointer_store(
        engine,
        &pointer,
        value,
        MemoryAccess::new(AddressSpace::Workgroup, 1),
        site,
    )
}

fn load_transpose_byte(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    storage: &PointerValue,
    lane: u32,
    format: Gfx950LdsTransposeFormatV1,
    part: u32,
    byte: u32,
) -> Result<u8, SimulationExecutionErrorV1> {
    let byte_delta = usize::try_from(lane * format.lane_byte_stride() + part * 8 + byte)
        .map_err(|_| engine.at(*site, SimulationExecutionErrorKindV1::PointerOffsetOverflow))?;
    let pointer = pointer_at_byte(engine, storage, byte_delta, ScalarType::U8, site)?;
    let value = execute_pointer_load(
        engine,
        &pointer,
        MemoryAccess::new(AddressSpace::Workgroup, 1),
        site,
    )?;
    Ok(value.bits() as u8)
}

fn execute_transpose_read(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    site: &CompactSite,
    format: Gfx950LdsTransposeFormatV1,
    lane: u32,
    storage: &PointerValue,
) -> Result<[u32; 8], SimulationExecutionErrorV1> {
    let mut results = [0_u32; 8];
    let group_base = (lane / 16) * 16;
    match format {
        Gfx950LdsTransposeFormatV1::Fp8E4M3 => {
            let source_parity = (lane % 16) / 8;
            let source_byte = lane % 8;
            for part in 0..4_u32 {
                let mut packed = [0_u8; 8];
                for (component, destination) in packed.iter_mut().enumerate() {
                    let source_lane = group_base + 2 * component as u32 + source_parity;
                    *destination = load_transpose_byte(
                        engine,
                        site,
                        storage,
                        source_lane,
                        format,
                        part,
                        source_byte,
                    )?;
                }
                results[(part * 2) as usize] =
                    u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
                results[(part * 2 + 1) as usize] =
                    u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
            }
        }
        Gfx950LdsTransposeFormatV1::Fp4E2M1 => {
            let source_byte = (lane % 16) / 2;
            let shift = (lane & 1) * 4;
            for part in 0..2_u32 {
                let mut packed = [0_u8; 8];
                for (component, destination) in packed.iter_mut().enumerate() {
                    let first = load_transpose_byte(
                        engine,
                        site,
                        storage,
                        group_base + 2 * component as u32,
                        format,
                        part,
                        source_byte,
                    )?;
                    let second = load_transpose_byte(
                        engine,
                        site,
                        storage,
                        group_base + 2 * component as u32 + 1,
                        format,
                        part,
                        source_byte,
                    )?;
                    *destination = ((first >> shift) & 0x0f) | (((second >> shift) & 0x0f) << 4);
                }
                results[(part * 2) as usize] =
                    u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
                results[(part * 2 + 1) as usize] =
                    u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
            }
        }
    }
    Ok(results)
}

fn operation_site(function: usize, block: &BasicBlock, ordinal: usize) -> CompactSite {
    CompactSite {
        function,
        block: block.id,
        operation: Some(u32::try_from(ordinal).unwrap_or(u32::MAX)),
    }
}

fn terminator_site(function: usize, block: &BasicBlock) -> CompactSite {
    CompactSite {
        function,
        block: block.id,
        operation: None,
    }
}

fn bind_block_arguments(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    function: usize,
    block: &BasicBlock,
    incoming: &[RuntimeValue],
    values: &mut RuntimeValues<'_>,
) -> Result<(), SimulationExecutionErrorV1> {
    if block.parameters.len() != incoming.len() {
        return Err(
            engine.fail(SimulationExecutionErrorKindV1::BlockArgumentArity {
                expected: block.parameters.len(),
                actual: incoming.len(),
            }),
        );
    }
    for (definition, value) in block.parameters.iter().zip(incoming) {
        let site = CompactSite {
            function,
            block: block.id,
            operation: None,
        };
        bind_typed_value(engine, values, definition, value.clone(), &site)?;
    }
    Ok(())
}

fn bind_typed_value(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &mut RuntimeValues<'_>,
    definition: &ValueDef,
    value: RuntimeValue,
    site: &CompactSite,
) -> Result<(), SimulationExecutionErrorV1> {
    if runtime_type(&value) != definition.ty {
        return Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(definition.id),
                expected: "declared SSA type",
            },
        ));
    }
    bind_runtime_value(engine, values, definition.id, value)
}

fn bind_runtime_value(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &mut RuntimeValues<'_>,
    id: ValueId,
    value: RuntimeValue,
) -> Result<(), SimulationExecutionErrorV1> {
    if !values.contains_key(&id) && values.len() == engine.limits.max_ssa_values {
        return Err(engine.fail(SimulationExecutionErrorKindV1::SsaValueLimit {
            limit: engine.limits.max_ssa_values,
        }));
    }
    values.try_insert(id, value).map_err(|()| {
        engine.fail(SimulationExecutionErrorKindV1::InternalInvariant(
            "admitted SSA slot and runtime type",
        ))
    })
}

fn runtime_type(value: &RuntimeValue) -> Type {
    match value {
        RuntimeValue::Scalar(value) => Type::Scalar(value.ty()),
        RuntimeValue::Execution(value) => value.ty(),
        RuntimeValue::PhysicalEntry(value) => Type::Scalar(value.scalar_type()),
        RuntimeValue::Pointer(pointer) => Type::pointer(
            Type::Scalar(pointer.element),
            pointer.logical_address_space(),
            pointer.access,
        ),
        RuntimeValue::StoragePointer(storage) => Type::pointer(
            Type::StorageObject(storage.layout),
            storage.pointer.address_space,
            storage.pointer.access,
        ),
        RuntimeValue::Slice(slice) => Type::slice(
            Type::Scalar(slice.element),
            slice.logical_address_space(),
            slice.access,
        ),
    }
}

fn runtime_non_scalar_value<'a>(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &'a RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
    expected: &'static str,
) -> Result<&'a RuntimeValue, SimulationExecutionErrorV1> {
    // A live scalar has no borrowed full-value cell in Columns. Preserve the
    // caller's existing wrong-type error rather than reporting it undefined.
    values.get_ref(&id).ok_or_else(|| {
        let kind = if values.contains_key(&id) {
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(id),
                expected,
            }
        } else {
            SimulationExecutionErrorKindV1::UndefinedValue(id)
        };
        engine.at(*site, kind)
    })
}

fn runtime_value_ref<'a>(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &'a RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
) -> Result<&'a RuntimeValue, SimulationExecutionErrorV1> {
    values
        .get_ref(&id)
        .ok_or_else(|| engine.at(*site, SimulationExecutionErrorKindV1::UndefinedValue(id)))
}

fn runtime_value(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
) -> Result<RuntimeValue, SimulationExecutionErrorV1> {
    values
        .get(&id)
        .ok_or_else(|| engine.at(*site, SimulationExecutionErrorKindV1::UndefinedValue(id)))
}

fn scalar_value(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    id: ValueId,
    site: &CompactSite,
) -> Result<ScalarBitsV1, SimulationExecutionErrorV1> {
    match runtime_value(engine, values, id, site)? {
        RuntimeValue::Scalar(value) => Ok(value),
        _ => Err(engine.at(
            *site,
            SimulationExecutionErrorKindV1::RuntimeType {
                value: Some(id),
                expected: "scalar",
            },
        )),
    }
}

fn resolve_values_into(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    ids: &[ValueId],
    site: &CompactSite,
    resolved: &mut Vec<RuntimeValue>,
) -> Result<(), SimulationExecutionErrorV1> {
    resolved.clear();
    // Empty call/return/edge lists need no owned RuntimeValue temporaries.
    // Keep their native stack path separate even in unoptimized builds.
    if ids.is_empty() {
        return Ok(());
    }
    resolve_nonempty_values_into(engine, values, ids, site, resolved)
}

#[inline(never)]
fn resolve_nonempty_values_into(
    engine: &Engine<'_, impl SimulationEventSinkV1>,
    values: &RuntimeValues<'_>,
    ids: &[ValueId],
    site: &CompactSite,
    resolved: &mut Vec<RuntimeValue>,
) -> Result<(), SimulationExecutionErrorV1> {
    if resolved.capacity() < ids.len() {
        resolved
            .try_reserve_exact(ids.len())
            .map_err(|_| engine.at(*site, SimulationExecutionErrorKindV1::AllocationFailure))?;
    }
    for id in ids {
        resolved.push(runtime_value(engine, values, *id, site)?);
    }
    Ok(())
}

fn try_clone_slice<T: Clone>(source: &[T]) -> Result<Vec<T>, SimulationExecutionErrorKindV1> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(source.len())
        .map_err(|_| SimulationExecutionErrorKindV1::AllocationFailure)?;
    cloned.extend_from_slice(source);
    Ok(cloned)
}

fn try_filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, SimulationExecutionErrorKindV1> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| SimulationExecutionErrorKindV1::AllocationFailure)?;
    values.resize(length, value);
    Ok(values)
}

fn top_level_error(kind: SimulationExecutionErrorKindV1) -> SimulationExecutionErrorV1 {
    SimulationExecutionErrorV1 {
        invocation: None,
        site: None,
        kind,
        observation_failure: None,
    }
}

fn intrinsic_value(kind: IntrinsicKind, invocation: SimulationInvocationV1) -> u64 {
    let axis = |axis| match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    };
    match kind {
        IntrinsicKind::LaunchExtent { axis: selected } => invocation.launch_extent[axis(selected)],
        IntrinsicKind::InvocationIndex {
            kind,
            axis: selected,
        } => {
            let selected = axis(selected);
            match kind {
                IndexKind::Global => invocation.global[selected],
                IndexKind::Workgroup => invocation.workgroup[selected],
                IndexKind::Local => u64::from(invocation.local[selected]),
                IndexKind::WorkgroupSize => u64::from(invocation.workgroup_size[selected]),
                IndexKind::WorkgroupCount => invocation.workgroup_count[selected],
            }
        }
    }
}

fn constant_scalar(
    constant: &Constant,
    target: SimulationTargetV1,
) -> Result<ScalarBitsV1, SimulationExecutionErrorKindV1> {
    let (ty, bits) = match *constant {
        Constant::Bool(value) => return Ok(ScalarBitsV1::boolean(value)),
        Constant::I8(value) => (ScalarType::I8, value as u8 as u128),
        Constant::I16(value) => (ScalarType::I16, value as u16 as u128),
        Constant::I32(value) => (ScalarType::I32, value as u32 as u128),
        Constant::I64(value) => (ScalarType::I64, value as u64 as u128),
        Constant::U8(value) => (ScalarType::U8, value as u128),
        Constant::U16(value) => (ScalarType::U16, value as u128),
        Constant::U32(value) => (ScalarType::U32, value as u128),
        Constant::U64(value) => (ScalarType::U64, value as u128),
        Constant::Index(value) => (ScalarType::Index, value as u128),
        Constant::F16Bits(value) => (ScalarType::F16, value as u128),
        Constant::Bf16Bits(value) => (ScalarType::Bf16, value as u128),
        Constant::F32Bits(value) => (ScalarType::F32, value as u128),
        Constant::F64Bits(value) => (ScalarType::F64, value as u128),
    };
    ScalarBitsV1::new(ty, bits, target)
        .map_err(|_| SimulationExecutionErrorKindV1::IntegerOutOfRange)
}

fn execute_unary(
    op: UnaryOp,
    value: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<ScalarBitsV1, SimulationExecutionErrorKindV1> {
    if !supports_unary(op, value.ty()) {
        return Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "unsupported unary passed preflight",
        ));
    }
    if value.ty().is_float() {
        return crate::soft_float::execute_unary_v1(op, value, target)
            .map_err(map_soft_float_error);
    }
    if value.ty() == ScalarType::Bool {
        return match op {
            UnaryOp::Not => Ok(ScalarBitsV1::boolean(value.bits() == 0)),
            UnaryOp::Negate => Err(SimulationExecutionErrorKindV1::InternalInvariant(
                "boolean negate passed preflight",
            )),
        };
    }
    let width = scalar_width(value, target)?;
    let bits = match op {
        UnaryOp::Not => !value.bits() & mask(width),
        UnaryOp::Negate => {
            if !value.ty().is_signed_integer() {
                return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                    "unsigned negate passed preflight",
                ));
            }
            let signed = signed_value(value, target)?;
            let negated = signed.checked_neg().ok_or(
                SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
                    "signed negation overflow",
                ),
            )?;
            if !signed_in_range(negated, width) {
                return Err(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
                    "signed negation overflow",
                ));
            }
            negated as u128 & mask(width)
        }
    };
    ScalarBitsV1::new(value.ty(), bits, target)
        .map_err(|_| SimulationExecutionErrorKindV1::InternalInvariant("unary bits"))
}

fn execute_binary(
    op: BinaryOp,
    lhs: ScalarBitsV1,
    rhs: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<SmallResults<ScalarBitsV1>, SimulationExecutionErrorKindV1> {
    if !supports_binary(op, lhs.ty(), rhs.ty()) {
        return Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "unsupported binary passed preflight",
        ));
    }
    if lhs.ty().is_float() {
        return Ok(SmallResults::One(
            crate::soft_float::execute_binary_v1(op, lhs, rhs, target)
                .map_err(map_soft_float_error)?,
        ));
    }
    if lhs.ty() == ScalarType::Bool {
        let value = match op {
            BinaryOp::BitAnd => lhs.bits() & rhs.bits(),
            BinaryOp::BitOr => lhs.bits() | rhs.bits(),
            BinaryOp::BitXor => lhs.bits() ^ rhs.bits(),
            _ => {
                return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                    "boolean binary",
                ));
            }
        };
        return Ok(SmallResults::One(ScalarBitsV1::boolean(value != 0)));
    }
    if let BinaryOp::Checked(operator) = op {
        let (bits, overflow) = checked_binary(operator, lhs, rhs, target)?;
        return Ok(SmallResults::Two(
            ScalarBitsV1::new(lhs.ty(), bits, target)
                .map_err(|_| SimulationExecutionErrorKindV1::InternalInvariant("checked bits"))?,
            ScalarBitsV1::boolean(overflow),
        ));
    }
    let width = scalar_width(lhs, target)?;
    let bits = match op {
        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => {
            defined_arithmetic(op, lhs, rhs, target)?
        }
        BinaryOp::Divide | BinaryOp::Remainder => defined_division(op, lhs, rhs, target)?,
        BinaryOp::BitAnd => lhs.bits() & rhs.bits(),
        BinaryOp::BitOr => lhs.bits() | rhs.bits(),
        BinaryOp::BitXor => lhs.bits() ^ rhs.bits(),
        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
            let amount = u32::try_from(rhs.bits()).map_err(|_| {
                SimulationExecutionErrorKindV1::UndefinedIntegerOperation("shift amount")
            })?;
            if amount >= u32::from(width) {
                return Err(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
                    "shift amount is not less than integer width",
                ));
            }
            match op {
                BinaryOp::ShiftLeft => (lhs.bits() << amount) & mask(width),
                BinaryOp::ShiftRight if lhs.ty().is_signed_integer() => {
                    (signed_value(lhs, target)? >> amount) as u128 & mask(width)
                }
                BinaryOp::ShiftRight => lhs.bits() >> amount,
                _ => {
                    return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                        "shift operation",
                    ));
                }
            }
        }
        BinaryOp::Checked(_) => {
            return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                "checked binary dispatch",
            ));
        }
    };
    Ok(SmallResults::One(
        ScalarBitsV1::new(lhs.ty(), bits & mask(width), target)
            .map_err(|_| SimulationExecutionErrorKindV1::InternalInvariant("binary bits"))?,
    ))
}

fn checked_binary(
    op: CheckedBinaryOperator,
    lhs: ScalarBitsV1,
    rhs: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<(u128, bool), SimulationExecutionErrorKindV1> {
    let width = scalar_width(lhs, target)?;
    let mask = mask(width);
    let wrapped = match op {
        CheckedBinaryOperator::Add => lhs.bits().wrapping_add(rhs.bits()) & mask,
        CheckedBinaryOperator::Subtract => lhs.bits().wrapping_sub(rhs.bits()) & mask,
        CheckedBinaryOperator::Multiply => lhs.bits().wrapping_mul(rhs.bits()) & mask,
    };
    let overflow = if lhs.ty().is_signed_integer() {
        let left = signed_value(lhs, target)?;
        let right = signed_value(rhs, target)?;
        match op {
            CheckedBinaryOperator::Add => left.checked_add(right),
            CheckedBinaryOperator::Subtract => left.checked_sub(right),
            CheckedBinaryOperator::Multiply => left.checked_mul(right),
        }
        .is_none_or(|value| !signed_in_range(value, width))
    } else {
        match op {
            CheckedBinaryOperator::Add => lhs.bits().checked_add(rhs.bits()),
            CheckedBinaryOperator::Subtract => lhs.bits().checked_sub(rhs.bits()),
            CheckedBinaryOperator::Multiply => lhs.bits().checked_mul(rhs.bits()),
        }
        .is_none_or(|value| value > mask)
    };
    Ok((wrapped, overflow))
}

fn defined_arithmetic(
    op: BinaryOp,
    lhs: ScalarBitsV1,
    rhs: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<u128, SimulationExecutionErrorKindV1> {
    let width = scalar_width(lhs, target)?;
    if lhs.ty().is_signed_integer() {
        let left = signed_value(lhs, target)?;
        let right = signed_value(rhs, target)?;
        let result = match op {
            BinaryOp::Add => left.checked_add(right),
            BinaryOp::Subtract => left.checked_sub(right),
            BinaryOp::Multiply => left.checked_mul(right),
            _ => {
                return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                    "arithmetic operation",
                ));
            }
        }
        .filter(|value| signed_in_range(*value, width))
        .ok_or(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
            "core signed arithmetic overflow",
        ))?;
        Ok(result as u128 & mask(width))
    } else {
        let result = match op {
            BinaryOp::Add => lhs.bits().checked_add(rhs.bits()),
            BinaryOp::Subtract => lhs.bits().checked_sub(rhs.bits()),
            BinaryOp::Multiply => lhs.bits().checked_mul(rhs.bits()),
            _ => {
                return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                    "arithmetic operation",
                ));
            }
        }
        .filter(|value| *value <= mask(width))
        .ok_or(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
            "core unsigned arithmetic overflow",
        ))?;
        Ok(result)
    }
}

fn defined_division(
    op: BinaryOp,
    lhs: ScalarBitsV1,
    rhs: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<u128, SimulationExecutionErrorKindV1> {
    let width = scalar_width(lhs, target)?;
    if rhs.bits() == 0 {
        return Err(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
            "division by zero",
        ));
    }
    if lhs.ty().is_signed_integer() {
        let left = signed_value(lhs, target)?;
        let right = signed_value(rhs, target)?;
        // Widening hides narrower MIN / -1 and MIN % -1 overflow.
        if lhs.bits() == 1_u128 << (width - 1) && right == -1 {
            return Err(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
                "signed division overflow",
            ));
        }
        let result = match op {
            BinaryOp::Divide => left.checked_div(right),
            BinaryOp::Remainder => left.checked_rem(right),
            _ => {
                return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                    "division operation",
                ));
            }
        }
        .ok_or(SimulationExecutionErrorKindV1::UndefinedIntegerOperation(
            "signed division overflow",
        ))?;
        Ok(result as u128 & mask(width))
    } else {
        Ok(match op {
            BinaryOp::Divide => lhs.bits() / rhs.bits(),
            BinaryOp::Remainder => lhs.bits() % rhs.bits(),
            _ => {
                return Err(SimulationExecutionErrorKindV1::InternalInvariant(
                    "division operation",
                ));
            }
        })
    }
}

fn execute_compare(
    predicate: ComparePredicate,
    lhs: ScalarBitsV1,
    rhs: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<bool, SimulationExecutionErrorKindV1> {
    if !supports_compare(predicate, lhs.ty(), rhs.ty()) {
        return Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "unsupported compare passed preflight",
        ));
    }
    if lhs.ty() == ScalarType::Bool {
        return match predicate {
            ComparePredicate::Equal => Ok(lhs.bits() == rhs.bits()),
            ComparePredicate::NotEqual => Ok(lhs.bits() != rhs.bits()),
            _ => Err(SimulationExecutionErrorKindV1::InternalInvariant(
                "ordered boolean compare passed preflight",
            )),
        };
    }
    if lhs.ty().is_signed_integer() {
        compare_ordered(
            predicate,
            signed_value(lhs, target)?,
            signed_value(rhs, target)?,
        )
    } else if lhs.ty().is_integer() {
        compare_ordered(predicate, lhs.bits(), rhs.bits())
    } else if lhs.ty().is_float() {
        crate::soft_float::execute_compare_v1(predicate, lhs, rhs).map_err(map_soft_float_error)
    } else {
        Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "unsupported compare type passed preflight",
        ))
    }
}

fn compare_ordered<T: Ord>(
    predicate: ComparePredicate,
    lhs: T,
    rhs: T,
) -> Result<bool, SimulationExecutionErrorKindV1> {
    Ok(match predicate {
        ComparePredicate::Equal => lhs == rhs,
        ComparePredicate::NotEqual => lhs != rhs,
        ComparePredicate::LessThan => lhs < rhs,
        ComparePredicate::LessThanOrEqual => lhs <= rhs,
        ComparePredicate::GreaterThan => lhs > rhs,
        ComparePredicate::GreaterThanOrEqual => lhs >= rhs,
    })
}

fn execute_cast(
    kind: CastKind,
    value: ScalarBitsV1,
    to: ScalarType,
    target: SimulationTargetV1,
) -> Result<ScalarBitsV1, SimulationExecutionErrorKindV1> {
    if !supported_cast(kind, value.ty(), to, target) {
        return Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "unsupported cast passed preflight",
        ));
    }
    if matches!(
        kind,
        CastKind::FloatExtend
            | CastKind::FloatTruncate
            | CastKind::IntegerToFloat
            | CastKind::FloatToInteger
    ) {
        return crate::soft_float::execute_cast_v1(kind, value, to, target)
            .map_err(map_soft_float_error);
    }
    let from_width = scalar_width(value, target)?;
    let to_width =
        target
            .scalar_bits(to)
            .ok_or(SimulationExecutionErrorKindV1::InternalInvariant(
                "preflighted cast target",
            ))?;
    let bits = match kind {
        CastKind::RestrictPointerAccess | CastKind::PointerToGeneric | CastKind::SliceToGeneric => {
            unreachable!("pointer or slice conversion is not a scalar cast")
        }
        CastKind::Truncate => value.bits() & mask(to_width),
        CastKind::ZeroExtend | CastKind::Bitcast => value.bits(),
        CastKind::SignExtend => signed_value(value, target)? as u128 & mask(to_width),
        CastKind::FloatExtend
        | CastKind::FloatTruncate
        | CastKind::IntegerToFloat
        | CastKind::FloatToInteger => unreachable!("handled software-float cast"),
    };
    let structurally_valid = match kind {
        CastKind::RestrictPointerAccess => false,
        CastKind::Truncate => to_width < from_width,
        CastKind::ZeroExtend | CastKind::SignExtend => to_width > from_width,
        CastKind::Bitcast => to_width == from_width,
        _ => false,
    };
    if !structurally_valid {
        return Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "invalid cast passed preflight",
        ));
    }
    ScalarBitsV1::new(to, bits, target)
        .map_err(|_| SimulationExecutionErrorKindV1::IntegerOutOfRange)
}

const fn map_soft_float_error(error: SoftFloatErrorV1) -> SimulationExecutionErrorKindV1 {
    match error {
        SoftFloatErrorV1::InternalInvariant(message) => {
            SimulationExecutionErrorKindV1::InternalInvariant(message)
        }
    }
}

fn scalar_width(
    value: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<u16, SimulationExecutionErrorKindV1> {
    target
        .scalar_bits(value.ty())
        .ok_or(SimulationExecutionErrorKindV1::InternalInvariant(
            "unsupported scalar passed preflight",
        ))
}

fn signed_value(
    value: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<i128, SimulationExecutionErrorKindV1> {
    let width = scalar_width(value, target)?;
    if !value.ty().is_signed_integer() {
        return Err(SimulationExecutionErrorKindV1::RuntimeType {
            value: None,
            expected: "signed integer",
        });
    }
    if width == 128 {
        return Ok(value.bits() as i128);
    }
    let sign = 1_u128 << (width - 1);
    let extended = if value.bits() & sign == 0 {
        value.bits()
    } else {
        value.bits() | !mask(width)
    };
    Ok(extended as i128)
}

fn signed_in_range(value: i128, width: u16) -> bool {
    if width == 128 {
        true
    } else {
        let limit = 1_i128 << (width - 1);
        (-limit..limit).contains(&value)
    }
}

fn scalar_nonnegative_usize(
    value: ScalarBitsV1,
    target: SimulationTargetV1,
) -> Result<usize, SimulationExecutionErrorKindV1> {
    if value.ty().is_signed_integer() && signed_value(value, target)? < 0 {
        return Err(SimulationExecutionErrorKindV1::IntegerOutOfRange);
    }
    usize::try_from(value.bits()).map_err(|_| SimulationExecutionErrorKindV1::IntegerOutOfRange)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execute_guarded_store_for_test(predicate: bool, pointer_allocation: u64) -> Vec<u8> {
        let mut module = Module::new("guarded-store-execution-test");
        module.functions.push(Function::declaration(
            "guarded-store-execution-test-entry",
            fe2o3_kernel_ir::Signature::new(vec![], vec![]),
        ));
        let block = BasicBlock::new(BlockId(0));
        let operation = Operation::new(
            vec![],
            OperationKind::GuardedStore {
                pointer: ValueId(0),
                value: ValueId(1),
                predicate: ValueId(2),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        );
        let allocation = Allocation {
            address_space: AddressSpace::Global,
            access: AccessMode::WriteOnly,
            alignment: 4,
            bytes: 0x1122_3344_u32.to_le_bytes().to_vec(),
            initialized: vec![true; 4],
            workgroup_published: vec![],
            workgroup_writer: vec![],
            observation_descriptor: Vec::new(),
        };
        let values = RuntimeValues::from([
            (
                ValueId(0),
                RuntimeValue::Pointer(PointerValue {
                    exposed_generic: false,
                    allocation: pointer_allocation,
                    byte_offset: 0,
                    element: ScalarType::U32,
                    address_space: AddressSpace::Global,
                    access: AccessMode::WriteOnly,
                    lower_bound: 0,
                    upper_bound: 4,
                    abi_argument_ordinal: NO_ABI_ARGUMENT_V1,
                }),
            ),
            (
                ValueId(1),
                RuntimeValue::Scalar(ScalarBitsV1::u32(0x89ab_cdef)),
            ),
            (
                ValueId(2),
                RuntimeValue::Scalar(ScalarBitsV1::boolean(predicate)),
            ),
        ]);
        let mut sink = NoopSimulationEventSinkV1;
        let mut debug_sink = NoopSimulationDebugSinkV1;
        let mut engine = Engine {
            module: &module,
            physical_lds_context: None,
            physical_lds_state: None,
            function_module_indices: vec![0],
            block_indices: vec![HashMap::new()],
            function_ssa_values: vec![0],
            value_layouts: &[],
            value_plan: FrameValuePlan::legacy(0),
            call_targets: vec![vec![]],
            switch_targets: vec![vec![]],
            target: SimulationTargetV1::amdgpu_64(),
            dynamic_workgroup_memory: None,
            matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            limits: SimulationLimitsV1::default(),
            policy: EventPolicyV1::Disabled,
            memory: Memory {
                allocations: HashMap::from([(7, allocation)]),
                argument_allocations: vec![],
                shared_allocations: HashMap::new(),
                next_allocation: 8,
                allocations_created: 1,
                live_bytes: 4,
                reuse: None,
            },
            sink: &mut sink,
            debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
            debug_sink: &mut debug_sink,
            physical_debug: None,
            debug_origin_requested: false,
            debug_observation_requested: false,
            debug_frames_requested: false,
            debug_identity_failed: false,
            debug_origin: None,
            debug_records: 0,
            debug_delivery_stopped: true,
            allocation_lifecycle_requested: false,
            allocation_lifecycle_stopped: false,
            schedule_identity: SimulationScheduleIdentityV1::WorkgroupMajorLocalZyxCooperativeV1,
            schedule_decision: 0,
            steps: 0,
            events: 0,
            reserved_event_closures: 0,
            event_delivery_stopped: false,
            invocation: Some(SimulationInvocationV1 {
                global: [0, 0, 0],
                workgroup: [0, 0, 0],
                local: [0, 0, 0],
                workgroup_size: [1, 1, 1],
                workgroup_count: [1, 1, 1],
                launch_extent: [1, 1, 1],
            }),
            accesses: HashMap::new(),
            conflicting_bytes: 0,
            first_conflict: None,
            conflict_incomplete: false,
            workgroup_happens_before_epoch: 0,
            unmodeled_atomic_or_fence_happens_before: false,
            unmodeled_atomic_serialization: false,
            race_trackers: vec![],
            workgroup_allocations: vec![],
        };
        let mut frame_allocations = vec![];
        assert!(matches!(
            execute_operation(
                &mut engine,
                0,
                &block,
                0,
                &operation,
                &values,
                &mut frame_allocations,
            )
            .expect("guarded store executes"),
            SmallResults::None,
        ));
        engine
            .memory
            .allocations
            .remove(&7)
            .expect("seed allocation remains live")
            .bytes
    }

    #[test]
    fn boolean_zero_extension_preserves_bits_across_index_widths() {
        for target in [
            SimulationTargetV1::little_endian(crate::IndexWidthV1::Bits32),
            SimulationTargetV1::little_endian(crate::IndexWidthV1::Bits64),
        ] {
            for value in [false, true] {
                let fixed = execute_cast(
                    CastKind::ZeroExtend,
                    ScalarBitsV1::boolean(value),
                    ScalarType::I64,
                    target,
                )
                .unwrap();
                assert_eq!(fixed.ty(), ScalarType::I64);
                assert_eq!(fixed.bits(), value as u128);

                let index = execute_cast(
                    CastKind::ZeroExtend,
                    ScalarBitsV1::boolean(value),
                    ScalarType::Index,
                    target,
                )
                .unwrap();
                assert_eq!(index.ty(), ScalarType::Index);
                assert_eq!(index.bits(), value as u128);
                assert!(index.matches_target(target));
            }
        }
    }

    #[test]
    fn admitted_binary_results_use_fixed_inline_storage() {
        let target = SimulationTargetV1::amdgpu_64();
        let checked = execute_binary(
            BinaryOp::Checked(CheckedBinaryOperator::Add),
            ScalarBitsV1::u32(u32::MAX),
            ScalarBitsV1::u32(1),
            target,
        )
        .expect("checked add");
        assert!(matches!(checked, SmallResults::Two(_, _)));

        let ordinary = execute_binary(
            BinaryOp::Add,
            ScalarBitsV1::u32(1),
            ScalarBitsV1::u32(2),
            target,
        )
        .expect("ordinary add");
        assert!(matches!(ordinary, SmallResults::One(_)));
        assert!(
            std::mem::size_of::<SmallResults<RuntimeValue>>()
                <= 2 * std::mem::size_of::<RuntimeValue>() + std::mem::align_of::<RuntimeValue>()
        );
    }

    #[test]
    fn write_only_simulator_memory_accepts_stores_and_rejects_reads() {
        let allocation = Allocation {
            address_space: AddressSpace::Global,
            access: AccessMode::WriteOnly,
            alignment: 4,
            bytes: vec![0; 4],
            initialized: vec![true; 4],
            workgroup_published: vec![],
            workgroup_writer: vec![],
            observation_descriptor: Vec::new(),
        };
        let pointer = PointerValue {
            exposed_generic: false,
            allocation: 7,
            byte_offset: 0,
            element: ScalarType::U32,
            address_space: AddressSpace::Global,
            access: AccessMode::WriteOnly,
            lower_bound: 0,
            upper_bound: 4,
            abi_argument_ordinal: NO_ABI_ARGUMENT_V1,
        };
        assert!(out_of_bounds_detail_v2(&pointer, 8, 4).abi_view.is_none());
        let access = MemoryAccess::new(AddressSpace::Global, 4);
        assert_eq!(
            validate_access(&allocation, &pointer, access, 4, false),
            Err(SimulationExecutionErrorKindV1::WriteOnlyRead)
        );
        validate_access(&allocation, &pointer, access, 4, true).unwrap();

        let mut memory = Memory {
            allocations: HashMap::from([(7, allocation)]),
            argument_allocations: vec![],
            shared_allocations: HashMap::new(),
            next_allocation: 8,
            allocations_created: 1,
            live_bytes: 4,
            reuse: None,
        };
        let width = memory
            .validate_store(
                &pointer,
                access,
                ScalarBitsV1::u32(0x89ab_cdef),
                SimulationTargetV1::amdgpu_64(),
            )
            .unwrap();
        memory
            .prepare_store(&pointer, width)
            .unwrap()
            .commit(ScalarBitsV1::u32(0x89ab_cdef));
        assert_eq!(memory.allocations[&7].bytes, 0x89ab_cdef_u32.to_le_bytes());
    }

    #[test]
    fn guarded_store_executes_true_and_skips_false_before_pointer_validation() {
        assert_eq!(
            execute_guarded_store_for_test(false, u64::MAX),
            0x1122_3344_u32.to_le_bytes(),
        );
        assert_eq!(
            execute_guarded_store_for_test(true, 7),
            0x89ab_cdef_u32.to_le_bytes(),
        );
    }
}

#[cfg(test)]
#[path = "execute_physical_global_copy_v21_tests.rs"]
mod physical_global_copy_v21_tests;

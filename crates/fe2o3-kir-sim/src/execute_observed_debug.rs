//! Proposed owner-reviewed in-process entry; no wire/resource profile changes.
use super::*;

impl AdmittedSimulationModuleV1 {
    /// Observe events and debugger records from one scheduled interpreter execution.
    ///
    /// The caller chooses request.events exactly as on the legacy entry.
    /// Event errors/limits preserve their existing execution-failure semantics;
    /// bounded sinks should use Stop/DropAndStop when observation is full.
    /// The two sinks are independent; pairing their retained storage is the
    /// caller's responsibility. This method does not authenticate a capture.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate_debugged_scheduled_with_sinks(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
        capture: SimulationDebugCaptureLimitsV1,
        event_sink: &mut impl SimulationEventSinkV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        let plan = self
            .preflight(request, target, limits)
            .map_err(SimulationErrorV1::Preflight)?;
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
            event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }
}

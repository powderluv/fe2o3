//! Additive in-process execution configuration. The request and schedule wire
//! schemas remain unchanged; callers must bind these options in capture identity.
use super::*;

/// Explicit observation/storage choices for one interpreter execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationExecutionOptionsV1 {
    capture_limits: SimulationDebugCaptureLimitsV1,
    matrix_bf16_numerics: crate::MatrixBf16NumericsV1,
    allocation_reuse: Option<SimulationAllocationReuseV1>,
    dynamic_workgroup_memory: Option<DynamicWorkgroupMemoryRequestV1>,
}

impl ObservationExecutionOptionsV1 {
    pub const fn new(capture_limits: SimulationDebugCaptureLimitsV1) -> Self {
        Self {
            capture_limits,
            matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            allocation_reuse: None,
            dynamic_workgroup_memory: None,
        }
    }

    /// Select ordinary numerical observation, never native/proof authority.
    /// Nondefault semantics cannot be recorded/replayed by the legacy schedule ABI.
    pub const fn with_matrix_bf16_numerics(mut self, mode: crate::MatrixBf16NumericsV1) -> Self {
        self.matrix_bf16_numerics = mode;
        self
    }
    pub const fn matrix_bf16_numerics(self) -> crate::MatrixBf16NumericsV1 {
        self.matrix_bf16_numerics
    }
    pub const fn with_allocation_reuse(mut self, policy: SimulationAllocationReuseV1) -> Self {
        self.allocation_reuse = Some(policy);
        self
    }

    pub const fn with_dynamic_workgroup_memory(
        mut self,
        dynamic: DynamicWorkgroupMemoryRequestV1,
    ) -> Self {
        self.dynamic_workgroup_memory = Some(dynamic);
        self
    }

    pub const fn capture_limits(self) -> SimulationDebugCaptureLimitsV1 {
        self.capture_limits
    }
    pub const fn allocation_reuse(self) -> Option<SimulationAllocationReuseV1> {
        self.allocation_reuse
    }
    pub const fn dynamic_workgroup_memory(self) -> Option<DynamicWorkgroupMemoryRequestV1> {
        self.dynamic_workgroup_memory
    }
}

impl AdmittedSimulationModuleV1 {
    /// Preflight includes the configured worst-case retained cache payload.
    ///
    /// Engine/map widths include inline cache and descriptor-vector headers.
    /// The separately reserved descriptor heaps are charged only when enabled.
    /// The ordinary live allocation envelope covers incoming payload temporaries;
    /// the additive cache cap covers released buffers while those inputs coexist.
    pub fn preflight_with_observation_options(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        options: ObservationExecutionOptionsV1,
    ) -> Result<SimulationPlanV1, SimulationPreflightErrorV1> {
        if options.capture_limits.is_enabled() {
            self.check_debug_capture_supported_v20()?;
        }
        let mut plan = match options.dynamic_workgroup_memory {
            Some(dynamic) => {
                self.preflight_with_dynamic_workgroup_memory(request, dynamic, target, limits)
            }
            None => self.preflight(request, target, limits),
        }?;
        let extra = match options.allocation_reuse {
            Some(policy) => observed_allocation_resident_extra(
                limits.max_allocations,
                policy.max_cached_payload_bytes(),
            )
            .ok_or(SimulationPreflightErrorV1::ResourceLimit {
                resource: "resident bytes",
                actual: u64::MAX,
                limit: limits.max_resident_bytes as u64,
            })?,
            None => 0,
        };
        plan.resident_bytes = checked_observation_resident_bytes(
            plan.resident_bytes,
            extra,
            limits.max_resident_bytes,
        )?;
        Ok(plan)
    }

    /// Observe the ordinary canonical interpreter without retaining a schedule.
    pub fn simulate_debugged_with_observation_options(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        options: ObservationExecutionOptionsV1,
        event_sink: &mut impl SimulationEventSinkV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        self.check_debug_capture_supported_v20()
            .map_err(SimulationErrorV1::Preflight)?;
        let plan = self
            .preflight_with_observation_options(request, target, limits, options)
            .map_err(SimulationErrorV1::Preflight)?;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: options.capture_limits,
                schedule: None,
                resident_offset: 0,
                allocation_reuse: options.allocation_reuse,
                matrix_bf16_numerics: options.matrix_bf16_numerics,
            },
            event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }

    /// Observe the same interpreter with explicit, opt-in backing-storage reuse.
    ///
    /// Lifecycle callbacks are independent of event policy and debug capture
    /// retention. Stop/drop of that stream never changes execution or legacy
    /// debug records. This entry does not authenticate a caller's retained data.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate_debugged_scheduled_with_observation_options(
        &self,
        request: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationScheduleRequestV1<'_>,
        options: ObservationExecutionOptionsV1,
        event_sink: &mut impl SimulationEventSinkV1,
        debug_sink: &mut impl SimulationDebugSinkV1,
    ) -> Result<SimulationExecutionV1, SimulationErrorV1> {
        if options.matrix_bf16_numerics != crate::MatrixBf16NumericsV1::ExactV1 {
            return Err(SimulationErrorV1::Preflight(
                SimulationPreflightErrorV1::InvalidLaunch(
                    "observational BF16 numerics do not support persisted schedules",
                ),
            ));
        }
        self.check_debug_capture_supported_v20()
            .map_err(SimulationErrorV1::Preflight)?;
        let plan = self
            .preflight_with_observation_options(request, target, limits, options)
            .map_err(SimulationErrorV1::Preflight)?;
        execute(
            self,
            request,
            ExecutionConfiguration {
                target,
                limits,
                policy: request.events,
                plan,
                debug_capture: options.capture_limits,
                schedule: Some(ExecutionScheduleRequestV1::Public(schedule)),
                resident_offset: 0,
                allocation_reuse: options.allocation_reuse,
                matrix_bf16_numerics: options.matrix_bf16_numerics,
            },
            event_sink,
            debug_sink,
        )
        .map_err(SimulationErrorV1::Execution)
    }
}

// Every live observed allocation owns exactly one descriptor reservation.
// Include both cache cells and one incoming temporary separately, conservatively
// even though the cumulative semantic-allocation limit further restricts them.
// Descriptor heap storage is not part of the four-vector payload cache cap.
fn observed_allocation_resident_extra(
    max_allocations: usize,
    cached_payload_bytes: usize,
) -> Option<usize> {
    reserved_vec_bytes::<crate::SimulationAllocationDescriptorV1>(1)?
        .checked_mul(max_allocations.checked_add(3)?)?
        .checked_add(cached_payload_bytes)
}

fn checked_observation_resident_bytes(
    base: usize,
    cached_payload_bytes: usize,
    limit: usize,
) -> Result<usize, SimulationPreflightErrorV1> {
    let actual = base.checked_add(cached_payload_bytes).ok_or(
        SimulationPreflightErrorV1::ResourceLimit {
            resource: "resident bytes",
            actual: u64::MAX,
            limit: limit as u64,
        },
    )?;
    if actual > limit {
        return Err(SimulationPreflightErrorV1::ResourceLimit {
            resource: "resident bytes",
            actual: actual as u64,
            limit: limit as u64,
        });
    }
    Ok(actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_heaps_are_additive_checked_and_separate_from_payload_cap() {
        let one = reserved_vec_bytes::<crate::SimulationAllocationDescriptorV1>(1).unwrap();
        assert_eq!(observed_allocation_resident_extra(7, 0), Some(10 * one));
        assert_eq!(
            observed_allocation_resident_extra(7, 23),
            Some(10 * one + 23)
        );
        assert_eq!(observed_allocation_resident_extra(usize::MAX, 0), None);
        assert_eq!(observed_allocation_resident_extra(7, usize::MAX), None);
        let extra = observed_allocation_resident_extra(7, 23).unwrap();
        assert_eq!(
            checked_observation_resident_bytes(20, extra, 20 + extra).unwrap(),
            20 + extra,
        );
        assert!(checked_observation_resident_bytes(20, extra, 19 + extra).is_err());
    }

    #[test]
    fn configured_cache_is_additive_checked_and_never_saturates() {
        assert_eq!(checked_observation_resident_bytes(20, 7, 27).unwrap(), 27);
        assert!(matches!(
            checked_observation_resident_bytes(20, 7, 26),
            Err(SimulationPreflightErrorV1::ResourceLimit {
                actual: 27,
                limit: 26,
                ..
            })
        ));
        assert!(matches!(
            checked_observation_resident_bytes(usize::MAX, 1, usize::MAX),
            Err(SimulationPreflightErrorV1::ResourceLimit {
                actual: u64::MAX,
                ..
            })
        ));
    }

    #[test]
    fn observation_options_default_to_legacy_storage_and_no_dynamic_extent() {
        let options =
            ObservationExecutionOptionsV1::new(SimulationDebugCaptureLimitsV1::disabled());
        assert_eq!(options.allocation_reuse(), None);
        assert_eq!(options.dynamic_workgroup_memory(), None);
        assert!(!options.capture_limits().is_enabled());
        let policy = SimulationAllocationReuseV1::exact_private_and_workgroup(8_192).unwrap();
        assert_eq!(
            options.with_allocation_reuse(policy).allocation_reuse(),
            Some(policy)
        );
        assert_eq!(options.allocation_reuse(), None);
    }
}

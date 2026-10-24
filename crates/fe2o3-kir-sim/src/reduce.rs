use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::mem::size_of;

use fe2o3_kernel_ir::{BlockId, WaveWidth};
use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::model::TargetIdentityWireV2;
use crate::resident::reserved_vec_bytes;
use crate::schedule::{
    ReductionScheduleSourceV1, schedule_context_identity, schedule_context_identity_with_dynamic,
};
use crate::{
    AdmittedSimulationModuleV1, DynamicWorkgroupMemoryRequestV1, IndexWidthV1,
    SimulationDataRaceV1, SimulationErrorV1, SimulationExecutionErrorKindV1,
    SimulationExecutionErrorV1, SimulationInvocationV1, SimulationLimitsV1,
    SimulationRaceAssessmentV1, SimulationRequestV1, SimulationScheduleDecisionV1,
    SimulationSiteV1, SimulationTargetV1,
};

/// Hard bound on executions performed by one failure-reduction request.
pub const MAX_FAILURE_REDUCTION_ATTEMPTS_V1: usize = crate::MAX_SCHEDULE_DECISIONS_V1 + 2;
/// Hard bound on decisions retained by one reduction operation and report.
pub const MAX_FAILURE_REDUCTION_RETAINED_DECISIONS_V1: usize = crate::MAX_SCHEDULE_DECISIONS_V1 * 3;
/// Maximum canonical bytes accepted for one persisted reduction report.
pub const MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1: usize = 768 * 1024 * 1024;

const REPORT_SCHEMA_V1: &str = "fe2o3-simulation-failure-reduction-v1";
const REPORT_SCHEMA_V2: &str = "fe2o3-simulation-failure-reduction-v2";
const FINGERPRINT_DOMAIN_V1: &[u8] = b"FE2O3/KIR-SIM/FAILURE-FINGERPRINT/V1\0";
const REPRODUCER_DOMAIN_V1: &[u8] = b"FE2O3/KIR-SIM/FAILURE-REPRODUCER/V1\0";
const REPORT_DOMAIN_V1: &[u8] = b"FE2O3/KIR-SIM/FAILURE-REDUCTION-REPORT/V1\0";
const MAX_REPORT_STRING_TOKEN_BYTES_V1: usize = 16 * 1024;

/// The deterministic schedule used to discover the original failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationFailureScheduleV1 {
    Canonical,
    Seeded { seed: u64 },
}

/// Explicit resource envelope for deterministic suffix reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationFailureReductionLimitsV1 {
    max_attempts: usize,
    max_decisions_per_schedule: usize,
    max_retained_decisions: usize,
}

impl SimulationFailureReductionLimitsV1 {
    pub fn new(
        max_attempts: usize,
        max_decisions_per_schedule: usize,
        max_retained_decisions: usize,
    ) -> Result<Self, SimulationFailureReductionRequestErrorV1> {
        if max_decisions_per_schedule == 0 {
            return Err(SimulationFailureReductionRequestErrorV1::Zero(
                "max_decisions_per_schedule",
            ));
        }
        if max_decisions_per_schedule > crate::MAX_SCHEDULE_DECISIONS_V1 {
            return Err(SimulationFailureReductionRequestErrorV1::AboveHardCap(
                "max_decisions_per_schedule",
            ));
        }
        let required_attempts = max_decisions_per_schedule.checked_add(2).ok_or(
            SimulationFailureReductionRequestErrorV1::AboveHardCap("max_attempts"),
        )?;
        if max_attempts < required_attempts {
            return Err(SimulationFailureReductionRequestErrorV1::Insufficient(
                "max_attempts",
            ));
        }
        if max_attempts > MAX_FAILURE_REDUCTION_ATTEMPTS_V1 {
            return Err(SimulationFailureReductionRequestErrorV1::AboveHardCap(
                "max_attempts",
            ));
        }
        let required_retained = max_decisions_per_schedule.checked_mul(3).ok_or(
            SimulationFailureReductionRequestErrorV1::AboveHardCap("max_retained_decisions"),
        )?;
        if max_retained_decisions < required_retained {
            return Err(SimulationFailureReductionRequestErrorV1::Insufficient(
                "max_retained_decisions",
            ));
        }
        if max_retained_decisions > MAX_FAILURE_REDUCTION_RETAINED_DECISIONS_V1 {
            return Err(SimulationFailureReductionRequestErrorV1::AboveHardCap(
                "max_retained_decisions",
            ));
        }
        Ok(Self {
            max_attempts,
            max_decisions_per_schedule,
            max_retained_decisions,
        })
    }

    pub const fn max_attempts(self) -> usize {
        self.max_attempts
    }
    pub const fn max_decisions_per_schedule(self) -> usize {
        self.max_decisions_per_schedule
    }
    pub const fn max_retained_decisions(self) -> usize {
        self.max_retained_decisions
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationFailureReductionRequestErrorV1 {
    Zero(&'static str),
    AboveHardCap(&'static str),
    Insufficient(&'static str),
}

impl fmt::Display for SimulationFailureReductionRequestErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid failure-reduction request: {self:?}")
    }
}

impl Error for SimulationFailureReductionRequestErrorV1 {}

/// Stable exact failure class, semantic sites, invocation coordinates, and detail digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationFailureFingerprintV1 {
    class: String,
    primary_invocation: Option<SimulationInvocationV1>,
    primary_site: Option<SimulationSiteV1>,
    related_invocation: Option<SimulationInvocationV1>,
    related_site: Option<SimulationSiteV1>,
    detail_identity: [u8; 32],
}

impl SimulationFailureFingerprintV1 {
    pub fn class(&self) -> &str {
        &self.class
    }
    pub const fn primary_invocation(&self) -> Option<SimulationInvocationV1> {
        self.primary_invocation
    }
    pub const fn primary_site(&self) -> Option<&SimulationSiteV1> {
        self.primary_site.as_ref()
    }
    pub const fn related_invocation(&self) -> Option<SimulationInvocationV1> {
        self.related_invocation
    }
    pub const fn related_site(&self) -> Option<&SimulationSiteV1> {
        self.related_site.as_ref()
    }
    pub const fn detail_identity(&self) -> &[u8; 32] {
        &self.detail_identity
    }
}

/// Deterministic coverage evidence for the linear suffix reducer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationFailureReductionCoverageV1 {
    attempts: usize,
    matching_candidates: usize,
    rejected_candidates: usize,
    removed_decisions: usize,
    one_shorter_checked: bool,
}

impl SimulationFailureReductionCoverageV1 {
    pub const fn attempts(self) -> usize {
        self.attempts
    }
    pub const fn matching_candidates(self) -> usize {
        self.matching_candidates
    }
    pub const fn rejected_candidates(self) -> usize {
        self.rejected_candidates
    }
    pub const fn removed_decisions(self) -> usize {
        self.removed_decisions
    }
    pub const fn is_locally_minimal(self) -> bool {
        self.one_shorter_checked
    }
}

/// Bounded, identity-bound reduction result. It is simulator evidence only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationFailureReductionReportV1 {
    kir_wire_version: u16,
    kir_sha256: [u8; 32],
    kir_canonical_bytes: u64,
    context_identity: [u8; 32],
    target: SimulationTargetV1,
    simulation_limits: SimulationLimitsV1,
    reduction_limits: SimulationFailureReductionLimitsV1,
    original_schedule: SimulationFailureScheduleV1,
    original_decisions: Vec<SimulationScheduleDecisionV1>,
    fingerprint: SimulationFailureFingerprintV1,
    minimized_prefix: Vec<SimulationScheduleDecisionV1>,
    reproducer_schedule: Vec<SimulationScheduleDecisionV1>,
    coverage: SimulationFailureReductionCoverageV1,
    reproducer_identity: [u8; 32],
    report_identity: [u8; 32],
}

impl SimulationFailureReductionReportV1 {
    pub const fn kir_wire_version(&self) -> u16 {
        self.kir_wire_version
    }
    pub const fn kir_sha256(&self) -> &[u8; 32] {
        &self.kir_sha256
    }
    pub const fn kir_canonical_bytes(&self) -> u64 {
        self.kir_canonical_bytes
    }
    pub const fn context_identity(&self) -> &[u8; 32] {
        &self.context_identity
    }
    pub const fn target(&self) -> SimulationTargetV1 {
        self.target
    }
    pub const fn simulation_limits(&self) -> SimulationLimitsV1 {
        self.simulation_limits
    }
    pub const fn reduction_limits(&self) -> SimulationFailureReductionLimitsV1 {
        self.reduction_limits
    }
    pub const fn original_schedule(&self) -> SimulationFailureScheduleV1 {
        self.original_schedule
    }
    pub fn original_decisions(&self) -> &[SimulationScheduleDecisionV1] {
        &self.original_decisions
    }
    pub const fn fingerprint(&self) -> &SimulationFailureFingerprintV1 {
        &self.fingerprint
    }
    pub fn minimized_prefix(&self) -> &[SimulationScheduleDecisionV1] {
        &self.minimized_prefix
    }
    pub fn reproducer_schedule(&self) -> &[SimulationScheduleDecisionV1] {
        &self.reproducer_schedule
    }
    pub const fn coverage(&self) -> SimulationFailureReductionCoverageV1 {
        self.coverage
    }
    pub const fn reproducer_identity(&self) -> &[u8; 32] {
        &self.reproducer_identity
    }
    pub const fn report_identity(&self) -> &[u8; 32] {
        &self.report_identity
    }
    pub const fn grants_execution_authority(&self) -> bool {
        false
    }
    pub const fn predicts_hardware_timing(&self) -> bool {
        false
    }

    /// Checks that an exact retained race observation is the one fingerprinted
    /// by this already validated reduction report.
    ///
    /// This is content consistency only. It does not authenticate the report's
    /// producer or replay the simulator.
    pub fn matches_data_race(&self, race: &SimulationDataRaceV1) -> bool {
        validate_report(self).is_ok() && self.fingerprint == fingerprint_race(race)
    }

    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, SimulationFailureReductionCodecErrorV1> {
        validate_report(self)?;
        let wire = ReportEncodeWireV1::from(self);
        let mut writer = BoundedWriterV1::default();
        serde_json::to_writer(&mut writer, &wire).map_err(|_| {
            writer
                .failure
                .clone()
                .unwrap_or(SimulationFailureReductionCodecErrorV1::EncodingFailure)
        })?;
        Ok(writer.bytes)
    }

    pub fn from_canonical_bytes(
        bytes: &[u8],
    ) -> Result<Self, SimulationFailureReductionCodecErrorV1> {
        if bytes.is_empty() {
            return Err(SimulationFailureReductionCodecErrorV1::Empty);
        }
        if bytes.len() > MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1 {
            return Err(SimulationFailureReductionCodecErrorV1::ByteLimit {
                actual: bytes.len(),
                limit: MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1,
            });
        }
        validate_string_tokens(bytes)?;
        let wire: ReportWireV1 = serde_json::from_slice(bytes)
            .map_err(|_| SimulationFailureReductionCodecErrorV1::Json)?;
        let report = Self::try_from(wire)?;
        validate_report(&report)?;
        let canonical = report.to_canonical_bytes()?;
        if canonical != bytes {
            return Err(SimulationFailureReductionCodecErrorV1::NonCanonical);
        }
        Ok(report)
    }
}

#[derive(Debug)]
pub enum SimulationFailureReductionErrorV1 {
    InvalidRequest(SimulationFailureReductionRequestErrorV1),
    Simulation(Box<SimulationErrorV1>),
    OriginalScheduleDidNotFail,
    UnsupportedFailureClass(&'static str),
    ResidentLimit { actual: usize, limit: usize },
    AllocationFailure,
    ReproducerMismatch,
    InvalidReport(SimulationFailureReductionCodecErrorV1),
}

impl fmt::Display for SimulationFailureReductionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "simulation failure reduction failed: {self:?}")
    }
}

impl Error for SimulationFailureReductionErrorV1 {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationFailureReductionCodecErrorV1 {
    Empty,
    ByteLimit { actual: usize, limit: usize },
    Json,
    UnsupportedSchema,
    InvalidTarget,
    InvalidLimits,
    InvalidReductionLimits,
    UnknownFailureClass,
    DecisionLimit,
    InvalidIdentity,
    NonCanonical,
    EncodingFailure,
    AllocationFailure,
    StringTokenLimit,
}

impl fmt::Display for SimulationFailureReductionCodecErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid failure-reduction report: {self:?}")
    }
}

impl Error for SimulationFailureReductionCodecErrorV1 {}

impl AdmittedSimulationModuleV1 {
    /// Reduces a seeded or canonical failure to a deterministic locally minimal decision prefix.
    #[allow(clippy::result_large_err)]
    pub fn reduce_simulation_failure(
        &self,
        simulation: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationFailureScheduleV1,
        reduction_limits: SimulationFailureReductionLimitsV1,
    ) -> Result<SimulationFailureReductionReportV1, SimulationFailureReductionErrorV1> {
        self.reduce_simulation_failure_configured(
            simulation,
            None,
            target,
            limits,
            schedule,
            reduction_limits,
        )
    }

    /// Reduces a failure under one explicit runtime-sized LDS contract.
    #[allow(clippy::result_large_err)]
    pub fn reduce_simulation_failure_with_dynamic_workgroup_memory(
        &self,
        simulation: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationFailureScheduleV1,
        reduction_limits: SimulationFailureReductionLimitsV1,
    ) -> Result<SimulationFailureReductionReportV1, SimulationFailureReductionErrorV1> {
        self.reduce_simulation_failure_configured(
            simulation,
            Some(dynamic),
            target,
            limits,
            schedule,
            reduction_limits,
        )
    }

    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    fn reduce_simulation_failure_configured(
        &self,
        simulation: &SimulationRequestV1,
        dynamic: Option<DynamicWorkgroupMemoryRequestV1>,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        schedule: SimulationFailureScheduleV1,
        reduction_limits: SimulationFailureReductionLimitsV1,
    ) -> Result<SimulationFailureReductionReportV1, SimulationFailureReductionErrorV1> {
        let required_attempts = reduction_limits.max_decisions_per_schedule + 2;
        if reduction_limits.max_attempts < required_attempts {
            return Err(SimulationFailureReductionErrorV1::InvalidRequest(
                SimulationFailureReductionRequestErrorV1::Insufficient("max_attempts"),
            ));
        }
        let decision_bytes = reserved_vec_bytes::<SimulationScheduleDecisionV1>(
            reduction_limits.max_decisions_per_schedule,
        )
        .and_then(|bytes| bytes.checked_mul(3))
        .and_then(|bytes| bytes.checked_add(size_of::<SimulationFailureReductionReportV1>()))
        .ok_or(SimulationFailureReductionErrorV1::ResidentLimit {
            actual: usize::MAX,
            limit: limits.max_resident_bytes,
        })?;
        let original_resident_bytes = decision_bytes
            .checked_add(candidate_fingerprint_bound(self))
            .ok_or(SimulationFailureReductionErrorV1::ResidentLimit {
                actual: usize::MAX,
                limit: limits.max_resident_bytes,
            })?;
        if original_resident_bytes > limits.max_resident_bytes {
            return Err(SimulationFailureReductionErrorV1::ResidentLimit {
                actual: original_resident_bytes,
                limit: limits.max_resident_bytes,
            });
        }
        let mut original = bounded_decisions(reduction_limits.max_decisions_per_schedule)?;
        let mut scratch = bounded_decisions(reduction_limits.max_decisions_per_schedule)?;
        let source = match schedule {
            SimulationFailureScheduleV1::Canonical => ReductionScheduleSourceV1::Canonical,
            SimulationFailureScheduleV1::Seeded { seed } => ReductionScheduleSourceV1::Seeded(seed),
        };
        let original_result = self.simulate_reduction_attempt_configured(
            simulation,
            dynamic,
            target,
            limits,
            source,
            reduction_limits.max_decisions_per_schedule,
            &mut original,
            original_resident_bytes,
        );
        let fingerprint = exact_failure(original_result)?
            .ok_or(SimulationFailureReductionErrorV1::OriginalScheduleDidNotFail)?;
        let attempt_resident_bytes = decision_bytes
            .checked_add(fingerprint_retained_bytes(&fingerprint))
            .and_then(|bytes| bytes.checked_add(candidate_fingerprint_bound(self)))
            .ok_or(SimulationFailureReductionErrorV1::ResidentLimit {
                actual: usize::MAX,
                limit: limits.max_resident_bytes,
            })?;
        if attempt_resident_bytes > limits.max_resident_bytes {
            return Err(SimulationFailureReductionErrorV1::ResidentLimit {
                actual: attempt_resident_bytes,
                limit: limits.max_resident_bytes,
            });
        }
        let mut minimized = bounded_decisions(reduction_limits.max_decisions_per_schedule)?;
        minimized.extend_from_slice(&original);
        let mut attempts = 1;
        let mut matching_candidates = 0;
        let mut rejected_candidates = 0;
        let mut one_shorter_checked = minimized.is_empty();
        while let Some(removed) = minimized.pop() {
            attempts += 1;
            scratch.clear();
            let result = self.simulate_reduction_attempt_configured(
                simulation,
                dynamic,
                target,
                limits,
                ReductionScheduleSourceV1::PrefixThenCanonical(&minimized),
                reduction_limits.max_decisions_per_schedule,
                &mut scratch,
                attempt_resident_bytes,
            );
            match exact_failure(result)? {
                Some(candidate) if candidate == fingerprint => {
                    matching_candidates += 1;
                    one_shorter_checked = minimized.is_empty();
                }
                _ => {
                    rejected_candidates += 1;
                    minimized.push(removed);
                    one_shorter_checked = true;
                    break;
                }
            }
        }
        attempts += 1;
        scratch.clear();
        let replay_result = self.simulate_reduction_attempt_configured(
            simulation,
            dynamic,
            target,
            limits,
            ReductionScheduleSourceV1::PrefixThenCanonical(&minimized),
            reduction_limits.max_decisions_per_schedule,
            &mut scratch,
            attempt_resident_bytes,
        );
        if exact_failure(replay_result)?.as_ref() != Some(&fingerprint) {
            return Err(SimulationFailureReductionErrorV1::ReproducerMismatch);
        }
        let original = original.into_boxed_slice().into_vec();
        let minimized = minimized.into_boxed_slice().into_vec();
        let reproducer = scratch.into_boxed_slice().into_vec();
        let context_identity = match dynamic {
            Some(dynamic) => schedule_context_identity_with_dynamic(
                *self.identity(),
                simulation,
                dynamic,
                target,
                limits,
            ),
            None => schedule_context_identity(*self.identity(), simulation, target, limits),
        };
        let reproducer_identity = reproducer_identity(context_identity, &fingerprint, &reproducer);
        let coverage = SimulationFailureReductionCoverageV1 {
            attempts,
            matching_candidates,
            rejected_candidates,
            removed_decisions: original.len() - minimized.len(),
            one_shorter_checked,
        };
        let mut report = SimulationFailureReductionReportV1 {
            kir_wire_version: self.identity().wire_version(),
            kir_sha256: *self.identity().digest(),
            kir_canonical_bytes: self.identity().canonical_length(),
            context_identity,
            target,
            simulation_limits: limits,
            reduction_limits,
            original_schedule: schedule,
            original_decisions: original,
            fingerprint,
            minimized_prefix: minimized,
            reproducer_schedule: reproducer,
            coverage,
            reproducer_identity,
            report_identity: [0; 32],
        };
        report.report_identity = report_identity(&report);
        Ok(report)
    }

    /// Re-executes the report's completed reproducer against the exact admitted context.
    #[allow(clippy::result_large_err)]
    pub fn replay_simulation_failure_reduction(
        &self,
        simulation: &SimulationRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        report: &SimulationFailureReductionReportV1,
    ) -> Result<SimulationFailureFingerprintV1, SimulationFailureReductionErrorV1> {
        self.replay_simulation_failure_reduction_configured(
            simulation, None, target, limits, report,
        )
    }

    /// Replays a persisted reduction under one exact dynamic LDS byte extent.
    #[allow(clippy::result_large_err)]
    pub fn replay_simulation_failure_reduction_with_dynamic_workgroup_memory(
        &self,
        simulation: &SimulationRequestV1,
        dynamic: DynamicWorkgroupMemoryRequestV1,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        report: &SimulationFailureReductionReportV1,
    ) -> Result<SimulationFailureFingerprintV1, SimulationFailureReductionErrorV1> {
        self.replay_simulation_failure_reduction_configured(
            simulation,
            Some(dynamic),
            target,
            limits,
            report,
        )
    }

    #[allow(clippy::result_large_err)]
    fn replay_simulation_failure_reduction_configured(
        &self,
        simulation: &SimulationRequestV1,
        dynamic: Option<DynamicWorkgroupMemoryRequestV1>,
        target: SimulationTargetV1,
        limits: SimulationLimitsV1,
        report: &SimulationFailureReductionReportV1,
    ) -> Result<SimulationFailureFingerprintV1, SimulationFailureReductionErrorV1> {
        validate_report(report).map_err(SimulationFailureReductionErrorV1::InvalidReport)?;
        let context = match dynamic {
            Some(dynamic) => schedule_context_identity_with_dynamic(
                *self.identity(),
                simulation,
                dynamic,
                target,
                limits,
            ),
            None => schedule_context_identity(*self.identity(), simulation, target, limits),
        };
        if report.kir_wire_version != self.identity().wire_version()
            || report.kir_sha256 != *self.identity().digest()
            || report.kir_canonical_bytes != self.identity().canonical_length()
            || report.context_identity != context
            || report.target != target
            || report.simulation_limits != limits
        {
            return Err(SimulationFailureReductionErrorV1::ReproducerMismatch);
        }
        let report_bytes = report_retained_bytes(report).ok_or(
            SimulationFailureReductionErrorV1::ResidentLimit {
                actual: usize::MAX,
                limit: limits.max_resident_bytes,
            },
        )?;
        let recorder_bytes = reserved_vec_bytes::<SimulationScheduleDecisionV1>(
            report.reduction_limits.max_decisions_per_schedule,
        )
        .ok_or(SimulationFailureReductionErrorV1::AllocationFailure)?;
        let decision_bytes = report_bytes
            .checked_add(recorder_bytes)
            .and_then(|bytes| bytes.checked_add(candidate_fingerprint_bound(self)))
            .ok_or(SimulationFailureReductionErrorV1::ResidentLimit {
                actual: usize::MAX,
                limit: limits.max_resident_bytes,
            })?;
        if decision_bytes > limits.max_resident_bytes {
            return Err(SimulationFailureReductionErrorV1::ResidentLimit {
                actual: decision_bytes,
                limit: limits.max_resident_bytes,
            });
        }
        let mut decisions = bounded_decisions(report.reduction_limits.max_decisions_per_schedule)?;
        let original_source = match report.original_schedule {
            SimulationFailureScheduleV1::Canonical => ReductionScheduleSourceV1::Canonical,
            SimulationFailureScheduleV1::Seeded { seed } => ReductionScheduleSourceV1::Seeded(seed),
        };
        let original_result = self.simulate_reduction_attempt_configured(
            simulation,
            dynamic,
            target,
            limits,
            original_source,
            report.reduction_limits.max_decisions_per_schedule,
            &mut decisions,
            decision_bytes,
        );
        let original_observed = exact_failure(original_result)?
            .ok_or(SimulationFailureReductionErrorV1::ReproducerMismatch)?;
        if original_observed != report.fingerprint || decisions != report.original_decisions {
            return Err(SimulationFailureReductionErrorV1::ReproducerMismatch);
        }
        drop(original_observed);
        decisions.clear();
        let result = self.simulate_reduction_attempt_configured(
            simulation,
            dynamic,
            target,
            limits,
            ReductionScheduleSourceV1::PrefixThenCanonical(&report.reproducer_schedule),
            report.reduction_limits.max_decisions_per_schedule,
            &mut decisions,
            decision_bytes,
        );
        let observed =
            exact_failure(result)?.ok_or(SimulationFailureReductionErrorV1::ReproducerMismatch)?;
        if observed != report.fingerprint || decisions != report.reproducer_schedule {
            return Err(SimulationFailureReductionErrorV1::ReproducerMismatch);
        }
        Ok(observed)
    }
}

fn bounded_decisions(
    maximum: usize,
) -> Result<Vec<SimulationScheduleDecisionV1>, SimulationFailureReductionErrorV1> {
    let mut decisions = Vec::new();
    decisions
        .try_reserve_exact(maximum)
        .map_err(|_| SimulationFailureReductionErrorV1::AllocationFailure)?;
    Ok(decisions)
}

fn report_retained_bytes(report: &SimulationFailureReductionReportV1) -> Option<usize> {
    size_of::<SimulationFailureReductionReportV1>()
        .checked_add(
            report
                .original_decisions
                .capacity()
                .checked_mul(size_of::<SimulationScheduleDecisionV1>())?,
        )?
        .checked_add(
            report
                .minimized_prefix
                .capacity()
                .checked_mul(size_of::<SimulationScheduleDecisionV1>())?,
        )?
        .checked_add(
            report
                .reproducer_schedule
                .capacity()
                .checked_mul(size_of::<SimulationScheduleDecisionV1>())?,
        )?
        .checked_add(report.fingerprint.class.capacity())?
        .checked_add(
            report
                .fingerprint
                .primary_site
                .as_ref()
                .map_or(0, |site| site.function.retained_capacity_bytes()),
        )?
        .checked_add(
            report
                .fingerprint
                .related_site
                .as_ref()
                .map_or(0, |site| site.function.retained_capacity_bytes()),
        )
}

fn fingerprint_retained_bytes(fingerprint: &SimulationFailureFingerprintV1) -> usize {
    fingerprint.class.capacity()
        + fingerprint
            .primary_site
            .as_ref()
            .map_or(0, |site| site.function.retained_capacity_bytes())
        + fingerprint
            .related_site
            .as_ref()
            .map_or(0, |site| site.function.retained_capacity_bytes())
}

fn candidate_fingerprint_bound(module: &AdmittedSimulationModuleV1) -> usize {
    let function_bytes = module
        .module()
        .functions
        .iter()
        .map(|function| function.id.retained_capacity_bytes())
        .max()
        .unwrap_or(0);
    128_usize.saturating_add(function_bytes.saturating_mul(2))
}

fn exact_failure(
    result: Result<crate::SimulationExecutionV1, SimulationErrorV1>,
) -> Result<Option<SimulationFailureFingerprintV1>, SimulationFailureReductionErrorV1> {
    match result {
        Err(SimulationErrorV1::Preflight(error)) => {
            Err(SimulationFailureReductionErrorV1::Simulation(Box::new(
                SimulationErrorV1::Preflight(error),
            )))
        }
        Err(SimulationErrorV1::Execution(error)) => {
            if let Some(class) = unsupported_failure(&error.kind) {
                return Err(SimulationFailureReductionErrorV1::UnsupportedFailureClass(
                    class,
                ));
            }
            Ok(Some(fingerprint_execution(&error)))
        }
        Ok(execution) => match execution.race_assessment() {
            SimulationRaceAssessmentV1::RacesObserved { first, .. } => {
                Ok(Some(fingerprint_race(first)))
            }
            SimulationRaceAssessmentV1::NoRacesObserved { .. }
            | SimulationRaceAssessmentV1::Incomplete { .. } => Ok(None),
        },
    }
}

fn unsupported_failure(kind: &SimulationExecutionErrorKindV1) -> Option<&'static str> {
    match kind {
        SimulationExecutionErrorKindV1::StepLimit { .. } => Some("step_limit"),
        SimulationExecutionErrorKindV1::EventLimit { .. } => Some("event_limit"),
        SimulationExecutionErrorKindV1::CallDepthLimit { .. } => Some("call_depth_limit"),
        SimulationExecutionErrorKindV1::SsaValueLimit { .. } => Some("ssa_value_limit"),
        SimulationExecutionErrorKindV1::AllocationLimit { .. } => Some("allocation_limit"),
        SimulationExecutionErrorKindV1::AllocationBytesLimit { .. } => {
            Some("allocation_bytes_limit")
        }
        SimulationExecutionErrorKindV1::TotalBytesLimit { .. } => Some("total_bytes_limit"),
        SimulationExecutionErrorKindV1::AllocationFailure => Some("allocation_failure"),
        SimulationExecutionErrorKindV1::WorkgroupSchedulerNoProgress { .. } => {
            Some("workgroup_scheduler_no_progress")
        }
        SimulationExecutionErrorKindV1::ScheduleDecisionLimit { .. } => {
            Some("schedule_decision_limit")
        }
        SimulationExecutionErrorKindV1::ScheduleResidentLimit { .. } => {
            Some("schedule_resident_limit")
        }
        SimulationExecutionErrorKindV1::ScheduleReplay(_) => Some("schedule_replay"),
        SimulationExecutionErrorKindV1::InternalInvariant(_) => Some("internal_invariant"),
        SimulationExecutionErrorKindV1::EventSinkFailure(_) => Some("event_sink_failure"),
        _ => None,
    }
}

fn fingerprint_execution(error: &SimulationExecutionErrorV1) -> SimulationFailureFingerprintV1 {
    let class = failure_class(&error.kind);
    let mut hash = Sha256::new();
    hash.update(FINGERPRINT_DOMAIN_V1);
    hash_bytes(&mut hash, class.as_bytes());
    hash_invocation(&mut hash, error.invocation);
    hash_site(&mut hash, error.site.as_ref());
    hash_execution_detail(&mut hash, &error.kind);
    SimulationFailureFingerprintV1 {
        class: class.to_owned(),
        primary_invocation: error.invocation,
        primary_site: error.site.clone(),
        related_invocation: None,
        related_site: None,
        detail_identity: hash.finalize().into(),
    }
}

fn fingerprint_race(race: &SimulationDataRaceV1) -> SimulationFailureFingerprintV1 {
    let conflict = &race.conflict;
    let mut hash = Sha256::new();
    hash.update(FINGERPRINT_DOMAIN_V1);
    hash_bytes(&mut hash, b"data_race");
    hash.update(conflict.allocation.to_le_bytes());
    hash.update((conflict.offset as u64).to_le_bytes());
    hash_invocation(&mut hash, Some(conflict.earlier));
    hash_site(&mut hash, Some(&conflict.earlier_site));
    hash_invocation(&mut hash, Some(conflict.later));
    hash_site(&mut hash, Some(&conflict.later_site));
    hash.update([race.earlier_atomic as u8, race.later_atomic as u8]);
    SimulationFailureFingerprintV1 {
        class: "data_race".to_owned(),
        primary_invocation: Some(conflict.later),
        primary_site: Some(conflict.later_site.clone()),
        related_invocation: Some(conflict.earlier),
        related_site: Some(conflict.earlier_site.clone()),
        detail_identity: hash.finalize().into(),
    }
}

fn failure_class(kind: &SimulationExecutionErrorKindV1) -> &'static str {
    use SimulationExecutionErrorKindV1 as K;
    match kind {
        K::StepLimit { .. } => "step_limit",
        K::EventLimit { .. } => "event_limit",
        K::CallDepthLimit { .. } => "call_depth_limit",
        K::SsaValueLimit { .. } => "ssa_value_limit",
        K::AllocationLimit { .. } => "allocation_limit",
        K::AllocationBytesLimit { .. } => "allocation_bytes_limit",
        K::TotalBytesLimit { .. } => "total_bytes_limit",
        K::AllocationFailure => "allocation_failure",
        K::MissingFunction(_) => "missing_function",
        K::MissingBody(_) => "missing_body",
        K::UnknownBlock(_) => "unknown_block",
        K::MissingTerminator(_) => "missing_terminator",
        K::UndefinedValue(_) => "undefined_value",
        K::RuntimeType { .. } => "runtime_type",
        K::ResultArity { .. } => "result_arity",
        K::BlockArgumentArity { .. } => "block_argument_arity",
        K::UndefinedIntegerOperation(_) => "undefined_integer_operation",
        K::IntegerOutOfRange => "integer_out_of_range",
        K::PointerOffsetOverflow => "pointer_offset_overflow",
        K::StorageArrayIndexOutOfBounds { .. } => "storage_array_index_out_of_bounds",
        K::PointerDistanceDifferentAllocation { .. } => "pointer_distance_different_allocation",
        K::PointerDistanceOutOfBounds => "pointer_distance_out_of_bounds",
        K::PointerDistanceNotDivisible { .. } => "pointer_distance_not_divisible",
        K::PointerDistanceOverflow => "pointer_distance_overflow",
        K::PointerDistanceNegativeUnsigned => "pointer_distance_negative_unsigned",
        K::CopyRangesOverlap { .. } => "copy_ranges_overlap",
        K::MemoryIntrinsicByteCountOverflow => "memory_intrinsic_byte_count_overflow",
        K::DanglingPointer { .. } => "dangling_pointer",
        K::AddressSpaceMismatch => "address_space_mismatch",
        K::ReadOnlyWrite => "read_only_write",
        K::WriteOnlyRead => "write_only_read",
        K::MisalignedAccess { .. } => "misaligned_access",
        K::OutOfBounds { .. } => "out_of_bounds",
        K::UninitializedRead { .. } => "uninitialized_read",
        K::WorkgroupUseBeforePublish { .. } => "workgroup_use_before_publish",
        K::DivergentWorkgroupBarrier(_) => "divergent_workgroup_barrier",
        K::MismatchedWorkgroupBarrier(_) => "mismatched_workgroup_barrier",
        K::IncompleteWave(_) => "incomplete_wave",
        K::DivergentWave(_) => "divergent_wave",
        K::MismatchedWave(_) => "mismatched_wave",
        K::UnsupportedMatrixInputDomain { .. } => "unsupported_matrix_input_domain",
        K::NonFiniteMatrixResult { .. } => "non_finite_matrix_result",
        K::WaveShuffleSourceOutOfRange { .. } => "wave_shuffle_source_out_of_range",
        K::WorkgroupSchedulerNoProgress { .. } => "workgroup_scheduler_no_progress",
        K::ScheduleDecisionLimit { .. } => "schedule_decision_limit",
        K::ScheduleResidentLimit { .. } => "schedule_resident_limit",
        K::ScheduleReplay(_) => "schedule_replay",
        K::ReachedUnreachable => "reached_unreachable",
        K::InternalInvariant(_) => "internal_invariant",
        K::EventSinkFailure(_) => "event_sink_failure",
    }
}

fn hash_execution_detail(hash: &mut Sha256, kind: &SimulationExecutionErrorKindV1) {
    use SimulationExecutionErrorKindV1 as K;
    match kind {
        K::StepLimit { limit } | K::EventLimit { limit } => hash.update(limit.to_le_bytes()),
        K::CallDepthLimit { limit } | K::SsaValueLimit { limit } | K::AllocationLimit { limit } => {
            hash.update((*limit as u64).to_le_bytes())
        }
        K::AllocationBytesLimit { actual, limit }
        | K::TotalBytesLimit { actual, limit }
        | K::ScheduleDecisionLimit { actual, limit }
        | K::ScheduleResidentLimit { actual, limit } => {
            hash.update((*actual as u64).to_le_bytes());
            hash.update((*limit as u64).to_le_bytes());
        }
        K::MissingFunction(id) | K::MissingBody(id) => hash_bytes(hash, id.as_str().as_bytes()),
        K::UnknownBlock(id) | K::MissingTerminator(id) => hash.update(id.0.to_le_bytes()),
        K::UndefinedValue(id) => hash.update(id.0.to_le_bytes()),
        K::RuntimeType { value, expected } => {
            hash.update(value.map_or(u32::MAX, |value| value.0).to_le_bytes());
            hash_bytes(hash, expected.as_bytes());
        }
        K::ResultArity { expected, actual } | K::BlockArgumentArity { expected, actual } => {
            hash.update((*expected as u64).to_le_bytes());
            hash.update((*actual as u64).to_le_bytes());
        }
        K::UndefinedIntegerOperation(detail) | K::InternalInvariant(detail) => {
            hash_bytes(hash, detail.as_bytes());
        }
        K::PointerDistanceDifferentAllocation {
            pointer_allocation,
            origin_allocation,
        } => {
            hash.update(pointer_allocation.to_le_bytes());
            hash.update(origin_allocation.to_le_bytes());
        }
        K::PointerDistanceNotDivisible {
            byte_difference,
            unit_bytes,
        } => {
            hash.update(byte_difference.to_le_bytes());
            hash.update(unit_bytes.to_le_bytes());
        }
        K::StorageArrayIndexOutOfBounds { index, length } => {
            hash.update(index.to_le_bytes());
            hash.update(length.to_le_bytes());
        }
        K::CopyRangesOverlap {
            allocation,
            source_offset,
            destination_offset,
            bytes,
        } => {
            hash.update(allocation.to_le_bytes());
            hash.update((*source_offset as u64).to_le_bytes());
            hash.update((*destination_offset as u64).to_le_bytes());
            hash.update((*bytes as u64).to_le_bytes());
        }
        K::DanglingPointer { allocation } => hash.update(allocation.to_le_bytes()),
        K::MisalignedAccess { required, offset } => {
            hash.update(required.to_le_bytes());
            hash.update((*offset as u64).to_le_bytes());
        }
        K::OutOfBounds {
            allocation,
            offset,
            bytes,
            allocation_bytes,
        } => hash_memory_detail(hash, *allocation, *offset, *bytes, Some(*allocation_bytes)),
        K::UninitializedRead {
            allocation,
            offset,
            bytes,
        }
        | K::WorkgroupUseBeforePublish {
            allocation,
            offset,
            bytes,
        } => hash_memory_detail(hash, *allocation, *offset, *bytes, None),
        K::DivergentWorkgroupBarrier(detail) => {
            hash.update(detail.phase.to_le_bytes());
            hash_local(hash, detail.waiting.local);
            hash_local(hash, detail.exited.local);
        }
        K::MismatchedWorkgroupBarrier(detail) => {
            hash.update(detail.phase.to_le_bytes());
            hash_event_site(hash, detail.expected);
            hash.update([match detail.mismatch {
                crate::WorkgroupBarrierMismatchV1::Site => 0,
                crate::WorkgroupBarrierMismatchV1::Semantics => 1,
                crate::WorkgroupBarrierMismatchV1::SiteAndSemantics => 2,
            }]);
        }
        K::IncompleteWave(detail) => {
            hash.update(wave_width(detail.width).to_le_bytes());
            hash.update(detail.wave_in_workgroup.to_le_bytes());
            hash.update(detail.active_mask.to_le_bytes());
            hash.update(detail.required_mask.to_le_bytes());
        }
        K::DivergentWave(detail) => {
            hash.update(wave_width(detail.width).to_le_bytes());
            hash.update(detail.wave_in_workgroup.to_le_bytes());
            hash_local(hash, detail.nonparticipating.local);
        }
        K::MismatchedWave(detail) => {
            hash.update(wave_width(detail.width).to_le_bytes());
            hash_event_site(hash, detail.expected);
        }
        K::WaveShuffleSourceOutOfRange {
            source_lane,
            tile_width,
        } => {
            hash.update(source_lane.to_le_bytes());
            hash.update(tile_width.to_le_bytes());
        }
        K::UnsupportedMatrixInputDomain {
            role,
            lane,
            component,
        } => {
            hash.update([*role as u8, *lane, *component]);
        }
        K::NonFiniteMatrixResult {
            lane,
            component,
            reduction,
        } => hash.update([*lane, *component, *reduction]),
        K::WorkgroupSchedulerNoProgress { phase } => hash.update(phase.to_le_bytes()),
        K::EventSinkFailure(error) => hash_bytes(hash, error.detail.as_bytes()),
        K::ScheduleReplay(_) => {}
        K::AllocationFailure
        | K::IntegerOutOfRange
        | K::PointerOffsetOverflow
        | K::PointerDistanceOutOfBounds
        | K::PointerDistanceOverflow
        | K::PointerDistanceNegativeUnsigned
        | K::MemoryIntrinsicByteCountOverflow
        | K::AddressSpaceMismatch
        | K::ReadOnlyWrite
        | K::WriteOnlyRead
        | K::ReachedUnreachable => {}
    }
}

fn hash_memory_detail(
    hash: &mut Sha256,
    allocation: u64,
    offset: usize,
    bytes: usize,
    allocation_bytes: Option<usize>,
) {
    hash.update(allocation.to_le_bytes());
    hash.update((offset as u64).to_le_bytes());
    hash.update((bytes as u64).to_le_bytes());
    if let Some(allocation_bytes) = allocation_bytes {
        hash.update((allocation_bytes as u64).to_le_bytes());
    }
}

fn hash_invocation(hash: &mut Sha256, invocation: Option<SimulationInvocationV1>) {
    hash.update([invocation.is_some() as u8]);
    if let Some(invocation) = invocation {
        for value in invocation.global {
            hash.update(value.to_le_bytes());
        }
        for value in invocation.workgroup {
            hash.update(value.to_le_bytes());
        }
        for value in invocation.local {
            hash.update(value.to_le_bytes());
        }
        for value in invocation.workgroup_size {
            hash.update(value.to_le_bytes());
        }
        for value in invocation.workgroup_count {
            hash.update(value.to_le_bytes());
        }
        for value in invocation.launch_extent {
            hash.update(value.to_le_bytes());
        }
    }
}

fn hash_site(hash: &mut Sha256, site: Option<&SimulationSiteV1>) {
    hash.update([site.is_some() as u8]);
    if let Some(site) = site {
        hash_bytes(hash, site.function.as_str().as_bytes());
        hash.update(site.block.0.to_le_bytes());
        hash.update(site.operation.unwrap_or(u32::MAX).to_le_bytes());
    }
}

fn hash_event_site(hash: &mut Sha256, site: crate::SimulationEventSiteV1) {
    hash.update((site.function_ordinal as u64).to_le_bytes());
    hash.update(site.block.0.to_le_bytes());
    hash.update(site.operation.unwrap_or(u32::MAX).to_le_bytes());
}

fn hash_local(hash: &mut Sha256, local: [u32; 3]) {
    for value in local {
        hash.update(value.to_le_bytes());
    }
}

const fn wave_width(width: WaveWidth) -> u32 {
    match width {
        WaveWidth::Wave32 => 32,
        WaveWidth::Wave64 => 64,
    }
}

fn hash_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn reproducer_identity(
    context: [u8; 32],
    fingerprint: &SimulationFailureFingerprintV1,
    decisions: &[SimulationScheduleDecisionV1],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(REPRODUCER_DOMAIN_V1);
    hash.update(context);
    hash.update(fingerprint.detail_identity);
    hash_decisions(&mut hash, decisions);
    hash.finalize().into()
}

fn report_identity(report: &SimulationFailureReductionReportV1) -> [u8; 32] {
    let mut hash = Sha256::new();
    report.target.hash_profile_identity(&mut hash);
    hash.update(REPORT_DOMAIN_V1);
    hash.update(report.kir_wire_version.to_le_bytes());
    hash.update(report.kir_sha256);
    hash.update(report.kir_canonical_bytes.to_le_bytes());
    hash.update(report.context_identity);
    hash.update([match report.target.index_width() {
        IndexWidthV1::Bits32 => 32,
        IndexWidthV1::Bits64 => 64,
    }]);
    hash_limits(&mut hash, report.simulation_limits);
    hash.update((report.reduction_limits.max_attempts as u64).to_le_bytes());
    hash.update((report.reduction_limits.max_decisions_per_schedule as u64).to_le_bytes());
    hash.update((report.reduction_limits.max_retained_decisions as u64).to_le_bytes());
    match report.original_schedule {
        SimulationFailureScheduleV1::Canonical => hash.update([0]),
        SimulationFailureScheduleV1::Seeded { seed } => {
            hash.update([1]);
            hash.update(seed.to_le_bytes());
        }
    }
    hash_decisions(&mut hash, &report.original_decisions);
    hash_fingerprint(&mut hash, &report.fingerprint);
    hash_decisions(&mut hash, &report.minimized_prefix);
    hash_decisions(&mut hash, &report.reproducer_schedule);
    hash.update((report.coverage.attempts as u64).to_le_bytes());
    hash.update((report.coverage.matching_candidates as u64).to_le_bytes());
    hash.update((report.coverage.rejected_candidates as u64).to_le_bytes());
    hash.update((report.coverage.removed_decisions as u64).to_le_bytes());
    hash.update([report.coverage.one_shorter_checked as u8]);
    hash.update(report.reproducer_identity);
    hash.finalize().into()
}

fn hash_limits(hash: &mut Sha256, limits: SimulationLimitsV1) {
    hash.update((limits.max_canonical_bytes as u64).to_le_bytes());
    hash.update((limits.max_reachable_functions as u64).to_le_bytes());
    hash.update((limits.max_reachable_operations as u64).to_le_bytes());
    hash.update(limits.max_invocations.to_le_bytes());
    hash.update(limits.max_workgroups.to_le_bytes());
    hash.update(limits.max_scheduled_slots.to_le_bytes());
    hash.update(limits.max_steps.to_le_bytes());
    hash.update((limits.max_call_depth as u64).to_le_bytes());
    hash.update((limits.max_ssa_values as u64).to_le_bytes());
    hash.update((limits.max_allocations as u64).to_le_bytes());
    hash.update((limits.max_allocation_bytes as u64).to_le_bytes());
    hash.update((limits.max_total_bytes as u64).to_le_bytes());
    hash.update((limits.max_resident_bytes as u64).to_le_bytes());
    hash.update(limits.max_events.to_le_bytes());
    hash.update((limits.max_memory_access_records as u64).to_le_bytes());
}

fn hash_fingerprint(hash: &mut Sha256, fingerprint: &SimulationFailureFingerprintV1) {
    hash_bytes(hash, fingerprint.class.as_bytes());
    hash_invocation(hash, fingerprint.primary_invocation);
    hash_site(hash, fingerprint.primary_site.as_ref());
    hash_invocation(hash, fingerprint.related_invocation);
    hash_site(hash, fingerprint.related_site.as_ref());
    hash.update(fingerprint.detail_identity);
}

fn hash_decisions(hash: &mut Sha256, decisions: &[SimulationScheduleDecisionV1]) {
    hash.update((decisions.len() as u64).to_le_bytes());
    for decision in decisions {
        for coordinate in decision.workgroup() {
            hash.update(coordinate.to_le_bytes());
        }
        hash.update(decision.phase().to_le_bytes());
        for coordinate in decision.local() {
            hash.update(coordinate.to_le_bytes());
        }
    }
}

fn validate_report(
    report: &SimulationFailureReductionReportV1,
) -> Result<(), SimulationFailureReductionCodecErrorV1> {
    report
        .simulation_limits
        .validate()
        .map_err(|_| SimulationFailureReductionCodecErrorV1::InvalidLimits)?;
    SimulationFailureReductionLimitsV1::new(
        report.reduction_limits.max_attempts,
        report.reduction_limits.max_decisions_per_schedule,
        report.reduction_limits.max_retained_decisions,
    )
    .map_err(|_| SimulationFailureReductionCodecErrorV1::InvalidReductionLimits)?;
    if !is_known_failure_class(&report.fingerprint.class) {
        return Err(SimulationFailureReductionCodecErrorV1::UnknownFailureClass);
    }
    let maximum = report.reduction_limits.max_decisions_per_schedule;
    if report.original_decisions.len() > maximum
        || report.minimized_prefix.len() > maximum
        || report.reproducer_schedule.len() > maximum
        || !report
            .original_decisions
            .starts_with(&report.minimized_prefix)
        || report.coverage.removed_decisions
            != report.original_decisions.len() - report.minimized_prefix.len()
        || !report.coverage.one_shorter_checked
        || report.coverage.attempts > report.reduction_limits.max_attempts
    {
        return Err(SimulationFailureReductionCodecErrorV1::DecisionLimit);
    }
    if reproducer_identity(
        report.context_identity,
        &report.fingerprint,
        &report.reproducer_schedule,
    ) != report.reproducer_identity
        || report_identity(report) != report.report_identity
    {
        return Err(SimulationFailureReductionCodecErrorV1::InvalidIdentity);
    }
    Ok(())
}

fn is_known_failure_class(class: &str) -> bool {
    const CLASSES: &[&str] = &[
        "missing_function",
        "missing_body",
        "unknown_block",
        "missing_terminator",
        "undefined_value",
        "runtime_type",
        "result_arity",
        "block_argument_arity",
        "undefined_integer_operation",
        "integer_out_of_range",
        "pointer_offset_overflow",
        "pointer_distance_different_allocation",
        "pointer_distance_out_of_bounds",
        "pointer_distance_not_divisible",
        "pointer_distance_overflow",
        "pointer_distance_negative_unsigned",
        "copy_ranges_overlap",
        "memory_intrinsic_byte_count_overflow",
        "dangling_pointer",
        "address_space_mismatch",
        "read_only_write",
        "write_only_read",
        "misaligned_access",
        "out_of_bounds",
        "uninitialized_read",
        "workgroup_use_before_publish",
        "divergent_workgroup_barrier",
        "mismatched_workgroup_barrier",
        "incomplete_wave",
        "divergent_wave",
        "mismatched_wave",
        "wave_shuffle_source_out_of_range",
        "reached_unreachable",
        "data_race",
    ];
    CLASSES.contains(&class)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReportWireV1 {
    schema: String,
    kir_wire_version: u16,
    kir_sha256: String,
    kir_canonical_bytes: u64,
    context_sha256: String,
    index_bits: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_identity: Option<TargetIdentityWireV2>,
    simulation_limits: LimitsWireV1,
    reduction_limits: ReductionLimitsWireV1,
    original_schedule: ScheduleWireV1,
    original_decisions: BoundedDecisionWiresV1,
    fingerprint: FingerprintWireV1,
    minimized_prefix: BoundedDecisionWiresV1,
    reproducer_schedule: BoundedDecisionWiresV1,
    coverage: CoverageWireV1,
    reproducer_sha256: String,
    report_sha256: String,
    grants_execution_authority: bool,
    predicts_hardware_timing: bool,
}

#[derive(Serialize)]
struct ReportEncodeWireV1<'a> {
    schema: &'static str,
    kir_wire_version: u16,
    kir_sha256: HexWireV1<'a>,
    kir_canonical_bytes: u64,
    context_sha256: HexWireV1<'a>,
    index_bits: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_identity: Option<TargetIdentityWireV2>,
    simulation_limits: LimitsWireV1,
    reduction_limits: ReductionLimitsWireV1,
    original_schedule: ScheduleWireV1,
    original_decisions: DecisionSliceWireV1<'a>,
    fingerprint: FingerprintEncodeWireV1<'a>,
    minimized_prefix: DecisionSliceWireV1<'a>,
    reproducer_schedule: DecisionSliceWireV1<'a>,
    coverage: CoverageWireV1,
    reproducer_sha256: HexWireV1<'a>,
    report_sha256: HexWireV1<'a>,
    grants_execution_authority: bool,
    predicts_hardware_timing: bool,
}

impl<'a> From<&'a SimulationFailureReductionReportV1> for ReportEncodeWireV1<'a> {
    fn from(report: &'a SimulationFailureReductionReportV1) -> Self {
        Self {
            schema: if report.target.amd_profile().is_some() {
                REPORT_SCHEMA_V2
            } else {
                REPORT_SCHEMA_V1
            },
            kir_wire_version: report.kir_wire_version,
            kir_sha256: HexWireV1(&report.kir_sha256),
            kir_canonical_bytes: report.kir_canonical_bytes,
            context_sha256: HexWireV1(&report.context_identity),
            index_bits: match report.target.index_width() {
                IndexWidthV1::Bits32 => 32,
                IndexWidthV1::Bits64 => 64,
            },
            target_identity: report
                .target
                .amd_profile()
                .map(|_| TargetIdentityWireV2::from_target(report.target)),
            simulation_limits: report.simulation_limits.into(),
            reduction_limits: report.reduction_limits.into(),
            original_schedule: report.original_schedule.into(),
            original_decisions: DecisionSliceWireV1(&report.original_decisions),
            fingerprint: FingerprintEncodeWireV1(&report.fingerprint),
            minimized_prefix: DecisionSliceWireV1(&report.minimized_prefix),
            reproducer_schedule: DecisionSliceWireV1(&report.reproducer_schedule),
            coverage: report.coverage.into(),
            reproducer_sha256: HexWireV1(&report.reproducer_identity),
            report_sha256: HexWireV1(&report.report_identity),
            grants_execution_authority: false,
            predicts_hardware_timing: false,
        }
    }
}

struct HexWireV1<'a>(&'a [u8; 32]);

impl Serialize for HexWireV1<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = [0_u8; 64];
        for (index, byte) in self.0.iter().copied().enumerate() {
            encoded[index * 2] = DIGITS[(byte >> 4) as usize];
            encoded[index * 2 + 1] = DIGITS[(byte & 0xf) as usize];
        }
        serializer
            .serialize_str(std::str::from_utf8(&encoded).expect("lower hexadecimal is valid UTF-8"))
    }
}

struct DecisionSliceWireV1<'a>(&'a [SimulationScheduleDecisionV1]);

impl Serialize for DecisionSliceWireV1<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for decision in self.0 {
            sequence.serialize_element(&DecisionWireV1::from(*decision))?;
        }
        sequence.end()
    }
}

struct FingerprintEncodeWireV1<'a>(&'a SimulationFailureFingerprintV1);

impl Serialize for FingerprintEncodeWireV1<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct FingerprintFieldsV1<'a> {
            class: &'a str,
            primary_invocation: Option<InvocationWireV1>,
            primary_site: Option<SiteEncodeWireV1<'a>>,
            related_invocation: Option<InvocationWireV1>,
            related_site: Option<SiteEncodeWireV1<'a>>,
            detail_sha256: HexWireV1<'a>,
        }

        let value = self.0;
        FingerprintFieldsV1 {
            class: &value.class,
            primary_invocation: value.primary_invocation.map(Into::into),
            primary_site: value.primary_site.as_ref().map(SiteEncodeWireV1),
            related_invocation: value.related_invocation.map(Into::into),
            related_site: value.related_site.as_ref().map(SiteEncodeWireV1),
            detail_sha256: HexWireV1(&value.detail_identity),
        }
        .serialize(serializer)
    }
}

struct SiteEncodeWireV1<'a>(&'a SimulationSiteV1);

impl Serialize for SiteEncodeWireV1<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SiteFieldsV1<'a> {
            function: &'a str,
            block: u32,
            operation: Option<u32>,
        }

        SiteFieldsV1 {
            function: self.0.function.as_str(),
            block: self.0.block.0,
            operation: self.0.operation,
        }
        .serialize(serializer)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LimitsWireV1 {
    max_canonical_bytes: usize,
    max_reachable_functions: usize,
    max_reachable_operations: usize,
    max_invocations: u64,
    max_workgroups: u64,
    max_scheduled_slots: u64,
    max_steps: u64,
    max_call_depth: usize,
    max_ssa_values: usize,
    max_allocations: usize,
    max_allocation_bytes: usize,
    max_total_bytes: usize,
    max_resident_bytes: usize,
    max_events: u64,
    max_memory_access_records: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReductionLimitsWireV1 {
    max_attempts: usize,
    max_decisions_per_schedule: usize,
    max_retained_decisions: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ScheduleWireV1 {
    Canonical,
    Seeded { seed: u64 },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionWireV1 {
    workgroup: [u64; 3],
    phase: u64,
    local: [u32; 3],
}

struct BoundedDecisionWiresV1(Vec<SimulationScheduleDecisionV1>);

impl Serialize for BoundedDecisionWiresV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for decision in &self.0 {
            sequence.serialize_element(&DecisionWireV1::from(*decision))?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for BoundedDecisionWiresV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DecisionsVisitorV1;

        impl<'de> Visitor<'de> for DecisionsVisitorV1 {
            type Value = BoundedDecisionWiresV1;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {} schedule decisions",
                    crate::MAX_SCHEDULE_DECISIONS_V1
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if sequence
                    .size_hint()
                    .is_some_and(|hint| hint > crate::MAX_SCHEDULE_DECISIONS_V1)
                {
                    return Err(de::Error::custom("fe2o3:failure_reduction_decision_limit"));
                }
                let mut decisions = Vec::new();
                while let Some(decision) = sequence.next_element::<DecisionWireV1>()? {
                    if decisions.len() == crate::MAX_SCHEDULE_DECISIONS_V1 {
                        return Err(de::Error::custom("fe2o3:failure_reduction_decision_limit"));
                    }
                    decisions.try_reserve(1).map_err(|_| {
                        de::Error::custom("fe2o3:failure_reduction_allocation_failure")
                    })?;
                    decisions.push(SimulationScheduleDecisionV1::from(decision));
                }
                Ok(BoundedDecisionWiresV1(decisions))
            }
        }

        deserializer.deserialize_seq(DecisionsVisitorV1)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FingerprintWireV1 {
    class: String,
    primary_invocation: Option<InvocationWireV1>,
    primary_site: Option<SiteWireV1>,
    related_invocation: Option<InvocationWireV1>,
    related_site: Option<SiteWireV1>,
    detail_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InvocationWireV1 {
    global: [u64; 3],
    workgroup: [u64; 3],
    local: [u32; 3],
    workgroup_size: [u32; 3],
    workgroup_count: [u64; 3],
    launch_extent: [u64; 3],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SiteWireV1 {
    function: String,
    block: u32,
    operation: Option<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageWireV1 {
    attempts: usize,
    matching_candidates: usize,
    rejected_candidates: usize,
    removed_decisions: usize,
    one_shorter_checked: bool,
}

impl TryFrom<ReportWireV1> for SimulationFailureReductionReportV1 {
    type Error = SimulationFailureReductionCodecErrorV1;

    fn try_from(wire: ReportWireV1) -> Result<Self, Self::Error> {
        if wire.schema != REPORT_SCHEMA_V1 && wire.schema != REPORT_SCHEMA_V2 {
            return Err(Self::Error::UnsupportedSchema);
        }
        if wire.grants_execution_authority || wire.predicts_hardware_timing {
            return Err(Self::Error::InvalidIdentity);
        }
        let target = match (wire.schema.as_str(), wire.target_identity) {
            (REPORT_SCHEMA_V1, None) => match wire.index_bits {
                32 => SimulationTargetV1::little_endian(IndexWidthV1::Bits32),
                64 => SimulationTargetV1::little_endian(IndexWidthV1::Bits64),
                _ => return Err(Self::Error::InvalidTarget),
            },
            (REPORT_SCHEMA_V2, Some(identity)) => identity
                .target(wire.index_bits)
                .filter(|target| target.amd_profile().is_some())
                .ok_or(Self::Error::InvalidTarget)?,
            _ => return Err(Self::Error::InvalidTarget),
        };
        let limits: SimulationLimitsV1 = wire.simulation_limits.into();
        limits.validate().map_err(|_| Self::Error::InvalidLimits)?;
        let reduction_limits = SimulationFailureReductionLimitsV1::new(
            wire.reduction_limits.max_attempts,
            wire.reduction_limits.max_decisions_per_schedule,
            wire.reduction_limits.max_retained_decisions,
        )
        .map_err(|_| Self::Error::InvalidReductionLimits)?;
        if wire.original_decisions.0.len() > reduction_limits.max_decisions_per_schedule
            || wire.minimized_prefix.0.len() > reduction_limits.max_decisions_per_schedule
            || wire.reproducer_schedule.0.len() > reduction_limits.max_decisions_per_schedule
        {
            return Err(Self::Error::DecisionLimit);
        }
        Ok(Self {
            kir_wire_version: wire.kir_wire_version,
            kir_sha256: parse_hex(&wire.kir_sha256)?,
            kir_canonical_bytes: wire.kir_canonical_bytes,
            context_identity: parse_hex(&wire.context_sha256)?,
            target,
            simulation_limits: limits,
            reduction_limits,
            original_schedule: wire.original_schedule.into(),
            original_decisions: wire.original_decisions.0,
            fingerprint: wire.fingerprint.try_into()?,
            minimized_prefix: wire.minimized_prefix.0,
            reproducer_schedule: wire.reproducer_schedule.0,
            coverage: wire.coverage.into(),
            reproducer_identity: parse_hex(&wire.reproducer_sha256)?,
            report_identity: parse_hex(&wire.report_sha256)?,
        })
    }
}

impl From<SimulationLimitsV1> for LimitsWireV1 {
    fn from(value: SimulationLimitsV1) -> Self {
        Self {
            max_canonical_bytes: value.max_canonical_bytes,
            max_reachable_functions: value.max_reachable_functions,
            max_reachable_operations: value.max_reachable_operations,
            max_invocations: value.max_invocations,
            max_workgroups: value.max_workgroups,
            max_scheduled_slots: value.max_scheduled_slots,
            max_steps: value.max_steps,
            max_call_depth: value.max_call_depth,
            max_ssa_values: value.max_ssa_values,
            max_allocations: value.max_allocations,
            max_allocation_bytes: value.max_allocation_bytes,
            max_total_bytes: value.max_total_bytes,
            max_resident_bytes: value.max_resident_bytes,
            max_events: value.max_events,
            max_memory_access_records: value.max_memory_access_records,
        }
    }
}

impl From<LimitsWireV1> for SimulationLimitsV1 {
    fn from(value: LimitsWireV1) -> Self {
        Self {
            max_canonical_bytes: value.max_canonical_bytes,
            max_reachable_functions: value.max_reachable_functions,
            max_reachable_operations: value.max_reachable_operations,
            max_invocations: value.max_invocations,
            max_workgroups: value.max_workgroups,
            max_scheduled_slots: value.max_scheduled_slots,
            max_steps: value.max_steps,
            max_call_depth: value.max_call_depth,
            max_ssa_values: value.max_ssa_values,
            max_allocations: value.max_allocations,
            max_allocation_bytes: value.max_allocation_bytes,
            max_total_bytes: value.max_total_bytes,
            max_resident_bytes: value.max_resident_bytes,
            max_events: value.max_events,
            max_memory_access_records: value.max_memory_access_records,
        }
    }
}

impl From<SimulationFailureReductionLimitsV1> for ReductionLimitsWireV1 {
    fn from(value: SimulationFailureReductionLimitsV1) -> Self {
        Self {
            max_attempts: value.max_attempts,
            max_decisions_per_schedule: value.max_decisions_per_schedule,
            max_retained_decisions: value.max_retained_decisions,
        }
    }
}

impl From<SimulationFailureScheduleV1> for ScheduleWireV1 {
    fn from(value: SimulationFailureScheduleV1) -> Self {
        match value {
            SimulationFailureScheduleV1::Canonical => Self::Canonical,
            SimulationFailureScheduleV1::Seeded { seed } => Self::Seeded { seed },
        }
    }
}

impl From<ScheduleWireV1> for SimulationFailureScheduleV1 {
    fn from(value: ScheduleWireV1) -> Self {
        match value {
            ScheduleWireV1::Canonical => Self::Canonical,
            ScheduleWireV1::Seeded { seed } => Self::Seeded { seed },
        }
    }
}

impl From<SimulationScheduleDecisionV1> for DecisionWireV1 {
    fn from(value: SimulationScheduleDecisionV1) -> Self {
        Self {
            workgroup: value.workgroup(),
            phase: value.phase(),
            local: value.local(),
        }
    }
}

impl From<DecisionWireV1> for SimulationScheduleDecisionV1 {
    fn from(value: DecisionWireV1) -> Self {
        Self::new(value.workgroup, value.phase, value.local)
    }
}

impl TryFrom<FingerprintWireV1> for SimulationFailureFingerprintV1 {
    type Error = SimulationFailureReductionCodecErrorV1;

    fn try_from(value: FingerprintWireV1) -> Result<Self, Self::Error> {
        if !is_known_failure_class(&value.class) || value.class.len() > 128 {
            return Err(Self::Error::UnknownFailureClass);
        }
        if value
            .primary_site
            .as_ref()
            .is_some_and(|site| site.function.len() > 16 * 1024)
            || value
                .related_site
                .as_ref()
                .is_some_and(|site| site.function.len() > 16 * 1024)
        {
            return Err(Self::Error::DecisionLimit);
        }
        Ok(Self {
            class: value.class,
            primary_invocation: value.primary_invocation.map(Into::into),
            primary_site: value.primary_site.map(Into::into),
            related_invocation: value.related_invocation.map(Into::into),
            related_site: value.related_site.map(Into::into),
            detail_identity: parse_hex(&value.detail_sha256)?,
        })
    }
}

impl From<SimulationInvocationV1> for InvocationWireV1 {
    fn from(value: SimulationInvocationV1) -> Self {
        Self {
            global: value.global,
            workgroup: value.workgroup,
            local: value.local,
            workgroup_size: value.workgroup_size,
            workgroup_count: value.workgroup_count,
            launch_extent: value.launch_extent,
        }
    }
}

impl From<InvocationWireV1> for SimulationInvocationV1 {
    fn from(value: InvocationWireV1) -> Self {
        Self {
            global: value.global,
            workgroup: value.workgroup,
            local: value.local,
            workgroup_size: value.workgroup_size,
            workgroup_count: value.workgroup_count,
            launch_extent: value.launch_extent,
        }
    }
}

impl From<SiteWireV1> for SimulationSiteV1 {
    fn from(value: SiteWireV1) -> Self {
        Self {
            function: value.function.into(),
            block: BlockId(value.block),
            operation: value.operation,
        }
    }
}

impl From<SimulationFailureReductionCoverageV1> for CoverageWireV1 {
    fn from(value: SimulationFailureReductionCoverageV1) -> Self {
        Self {
            attempts: value.attempts,
            matching_candidates: value.matching_candidates,
            rejected_candidates: value.rejected_candidates,
            removed_decisions: value.removed_decisions,
            one_shorter_checked: value.one_shorter_checked,
        }
    }
}

impl From<CoverageWireV1> for SimulationFailureReductionCoverageV1 {
    fn from(value: CoverageWireV1) -> Self {
        Self {
            attempts: value.attempts,
            matching_candidates: value.matching_candidates,
            rejected_candidates: value.rejected_candidates,
            removed_decisions: value.removed_decisions,
            one_shorter_checked: value.one_shorter_checked,
        }
    }
}

fn parse_hex(value: &str) -> Result<[u8; 32], SimulationFailureReductionCodecErrorV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(SimulationFailureReductionCodecErrorV1::InvalidIdentity);
    }
    let mut output = [0; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        output[index] = digit(pair[0])
            .and_then(|high| digit(pair[1]).map(|low| (high << 4) | low))
            .ok_or(SimulationFailureReductionCodecErrorV1::InvalidIdentity)?;
    }
    Ok(output)
}

fn validate_string_tokens(bytes: &[u8]) -> Result<(), SimulationFailureReductionCodecErrorV1> {
    let mut in_string = false;
    let mut escaped = false;
    let mut encoded = 0_usize;
    for byte in bytes {
        if !in_string {
            if *byte == b'"' {
                in_string = true;
                encoded = 0;
            }
            continue;
        }
        if escaped {
            escaped = false;
            encoded += 1;
        } else if *byte == b'\\' {
            escaped = true;
            encoded += 1;
        } else if *byte == b'"' {
            in_string = false;
            continue;
        } else {
            encoded += 1;
        }
        if encoded > MAX_REPORT_STRING_TOKEN_BYTES_V1 {
            return Err(SimulationFailureReductionCodecErrorV1::StringTokenLimit);
        }
    }
    Ok(())
}

#[derive(Default)]
struct BoundedWriterV1 {
    bytes: Vec<u8>,
    failure: Option<SimulationFailureReductionCodecErrorV1>,
}

impl Write for BoundedWriterV1 {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(total) = self.bytes.len().checked_add(bytes.len()) else {
            self.failure = Some(SimulationFailureReductionCodecErrorV1::ByteLimit {
                actual: usize::MAX,
                limit: MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1,
            });
            return Err(io::Error::other(
                "failure-reduction report byte count overflow",
            ));
        };
        if total > MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1 {
            self.failure = Some(SimulationFailureReductionCodecErrorV1::ByteLimit {
                actual: total,
                limit: MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1,
            });
            return Err(io::Error::other(
                "failure-reduction report byte limit exceeded",
            ));
        }
        if total > self.bytes.capacity() {
            let desired = total
                .max(self.bytes.capacity().max(1_024).saturating_mul(2))
                .min(MAX_PERSISTED_FAILURE_REDUCTION_BYTES_V1);
            if self
                .bytes
                .try_reserve_exact(desired.saturating_sub(self.bytes.len()))
                .is_err()
            {
                self.failure = Some(SimulationFailureReductionCodecErrorV1::AllocationFailure);
                return Err(io::Error::other(
                    "cannot allocate bounded failure-reduction report bytes",
                ));
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_and_scheduler_failures_are_not_reduction_targets() {
        let failures = [
            SimulationExecutionErrorKindV1::StepLimit { limit: 1 },
            SimulationExecutionErrorKindV1::EventLimit { limit: 1 },
            SimulationExecutionErrorKindV1::CallDepthLimit { limit: 1 },
            SimulationExecutionErrorKindV1::SsaValueLimit { limit: 1 },
            SimulationExecutionErrorKindV1::AllocationLimit { limit: 1 },
            SimulationExecutionErrorKindV1::AllocationBytesLimit {
                actual: 2,
                limit: 1,
            },
            SimulationExecutionErrorKindV1::TotalBytesLimit {
                actual: 2,
                limit: 1,
            },
            SimulationExecutionErrorKindV1::AllocationFailure,
            SimulationExecutionErrorKindV1::WorkgroupSchedulerNoProgress { phase: 0 },
            SimulationExecutionErrorKindV1::ScheduleDecisionLimit {
                actual: 2,
                limit: 1,
            },
            SimulationExecutionErrorKindV1::ScheduleResidentLimit {
                actual: 2,
                limit: 1,
            },
            SimulationExecutionErrorKindV1::InternalInvariant("fixture"),
        ];
        for failure in failures {
            assert!(unsupported_failure(&failure).is_some(), "{failure:?}");
        }
    }

    #[test]
    fn reduction_limits_and_report_string_tokens_fail_closed() {
        assert!(matches!(
            SimulationFailureReductionLimitsV1::new(3, 2, 6),
            Err(SimulationFailureReductionRequestErrorV1::Insufficient(
                "max_attempts"
            ))
        ));
        assert!(matches!(
            SimulationFailureReductionLimitsV1::new(4, 2, 5),
            Err(SimulationFailureReductionRequestErrorV1::Insufficient(
                "max_retained_decisions"
            ))
        ));
        let hostile = format!("\"{}\"", "x".repeat(MAX_REPORT_STRING_TOKEN_BYTES_V1 + 1));
        assert_eq!(
            validate_string_tokens(hostile.as_bytes()),
            Err(SimulationFailureReductionCodecErrorV1::StringTokenLimit)
        );
    }
}

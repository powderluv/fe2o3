//! Typed, move-only, budget-owned V20 CPU observations.
//! Legacy debug entrypoints deliberately remain refused for physical profiles.
use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrOwnedVerificationResourceBudgetV1 as OwnedBudget,
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource, VerifiedCanonicalKernelIrModuleV20,
};
#[path = "execute_debug_physical_snapshot_v20.rs"]
mod snapshot;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Profile {
    EntryV20,
    GlobalCopyV21,
    LdsExchangeV22,
}
impl Profile {
    pub(super) fn matches(self, admitted: &AdmittedSimulationModuleV1) -> bool {
        match self {
            Self::EntryV20 => admitted.uses_physical_entry_v20(),
            Self::GlobalCopyV21 => admitted.uses_physical_global_copy_v21(),
            Self::LdsExchangeV22 => admitted.uses_physical_lds_exchange_v22(),
        }
    }
}
/// Only genuine typed immutable canonical owners can enter the common capture.
/// This is private CPU observation plumbing, never a hashes-to-owner constructor.
#[derive(Clone, Copy)]
pub(super) enum CanonicalOwner<'a> {
    Entry(&'a VerifiedCanonicalKernelIrModuleV20),
    GlobalCopy(&'a fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV21),
    LdsExchange(&'a fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV22),
}
impl CanonicalOwner<'_> {
    fn profile(self) -> Profile {
        match self {
            Self::Entry(_) => Profile::EntryV20,
            Self::GlobalCopy(_) => Profile::GlobalCopyV21,
            Self::LdsExchange(_) => Profile::LdsExchangeV22,
        }
    }
    fn identity(self) -> SimulationKernelIrIdentityV1 {
        match self {
            Self::Entry(o) => (*o.identity()).into(),
            Self::GlobalCopy(o) => (*o.identity()).into(),
            Self::LdsExchange(o) => (*o.identity()).into(),
        }
    }
    fn bytes_len(self) -> usize {
        match self {
            Self::Entry(o) => o.canonical_bytes().len(),
            Self::GlobalCopy(o) => o.canonical_bytes().len(),
            Self::LdsExchange(o) => o.canonical_bytes().len(),
        }
    }
}

pub const MAX_PHYSICAL_ENTRY_DEBUG_RECORDS_V20: usize = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalEntryDebugOptionsV20 {
    simulation: SimulationLimitsV1,
    capture: SimulationDebugCaptureLimitsV1,
    records: usize,
}
impl PhysicalEntryDebugOptionsV20 {
    pub fn new(
        simulation: SimulationLimitsV1,
        capture: SimulationDebugCaptureLimitsV1,
        records: usize,
    ) -> Option<Self> {
        (capture.is_enabled() && records <= MAX_PHYSICAL_ENTRY_DEBUG_RECORDS_V20).then_some(Self {
            simulation,
            capture,
            records,
        })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalEntryDebugCaptureErrorV20 {
    OwnerMismatch,
    NotPhysicalEntry,
    Resource(Resource),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalEntryDebugCaptureStopV20 {
    RecordLimit,
    Resource(Resource),
    SnapshotUnavailable,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalEntryDebugOutcomeV20 {
    NotStarted,
    PreflightRefused,
    Completed,
    ExecutionFailed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalEntryDebugUsageV20 {
    pub entry_storage: usize,
    pub work: usize,
    pub failed_work: Option<usize>,
    pub retained_storage: usize,
    pub peak_storage: usize,
    pub failed_storage: Option<usize>,
}

/// This capture retains the same owned cumulative ledger throughout its lifetime.
/// Read-only projections expose no cloneable transcript/record, source authority,
/// numeric GPU pointer halves, or resumable interpreter state.
///
/// ```compile_fail
/// use fe2o3_kir_sim::PhysicalEntryDebugCaptureV20;
/// fn clone_capture(value:PhysicalEntryDebugCaptureV20) { let _=value.clone(); }
/// ```
pub struct PhysicalEntryDebugCaptureV20 {
    pub(super) state: State,
    identity: SimulationKernelIrIdentityV1,
    outcome: PhysicalEntryDebugOutcomeV20,
    error: Option<PhysicalEntryDebugCaptureErrorV20>,
}
impl PhysicalEntryDebugCaptureV20 {
    pub const fn identity(&self) -> SimulationKernelIrIdentityV1 {
        self.identity
    }
    pub const fn outcome(&self) -> PhysicalEntryDebugOutcomeV20 {
        self.outcome
    }
    pub const fn error(&self) -> Option<PhysicalEntryDebugCaptureErrorV20> {
        self.error
    }
    pub const fn stop(&self) -> Option<PhysicalEntryDebugCaptureStopV20> {
        self.state.stop
    }
    pub fn len(&self) -> usize {
        self.state.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.state.records.is_empty()
    }
    pub fn record(&self, index: usize) -> Option<PhysicalEntryDebugRecordRefV20<'_>> {
        Some(PhysicalEntryDebugRecordRefV20 {
            record: self.state.records.get(index)?,
        })
    }
    pub fn usage(&self) -> PhysicalEntryDebugUsageV20 {
        PhysicalEntryDebugUsageV20 {
            entry_storage: self.state.floor,
            work: self.state.ledger.work(),
            failed_work: self.state.ledger.failed_work(),
            retained_storage: self.state.ledger.storage() - self.state.floor,
            peak_storage: self.state.ledger.peak_storage(),
            failed_storage: self.state.ledger.failed_storage(),
        }
    }
    /// Budget ordinary observation navigation, not execution or semantic proof.
    pub fn charge_navigation(&mut self, steps: usize) -> Result<(), Resource> {
        self.state
            .ledger
            .with_budget(|budget| budget.charge_work(steps))
    }
    /// Drops every retained snapshot before returning the original ledger at its
    /// original storage floor. Work, peak and first-denial history are preserved.
    pub fn into_budget(self) -> OwnedBudget {
        let Self { state, .. } = self;
        let State {
            records,
            mut ledger,
            floor,
            ..
        } = state;
        drop(records);
        let release = ledger.storage() - floor;
        ledger
            .with_budget(|budget| budget.release_storage(release))
            .expect("capture retains its original storage floor");
        ledger
    }
}

pub(super) struct State {
    records: Vec<SimulationDebugRecordV1>,
    ledger: OwnedBudget,
    floor: usize,
    limit: usize,
    profile: Profile,
    stop: Option<PhysicalEntryDebugCaptureStopV20>,
}
impl State {
    fn new(ledger: OwnedBudget, limit: usize, profile: Profile) -> Self {
        let floor = ledger.storage();
        Self {
            records: Vec::new(),
            ledger,
            floor,
            limit,
            profile,
            stop: None,
        }
    }
    pub(super) fn matches(&self, admitted: &AdmittedSimulationModuleV1) -> bool {
        self.profile.matches(admitted)
    }
    fn initialize(&mut self) -> Result<(), PhysicalEntryDebugCaptureErrorV20> {
        self.ledger.with_budget(|budget| {
            // Include the fixed optional cursor used by the move-only debugger
            // session even when this capture is inspected without that wrapper.
            budget
                .reserve_storage(
                    size_of::<PhysicalEntryDebugCaptureV20>() + size_of::<Option<usize>>(),
                )
                .map_err(PhysicalEntryDebugCaptureErrorV20::Resource)?;
            self.records = snapshot::vector(self.limit, budget)
                .map_err(PhysicalEntryDebugCaptureErrorV20::Resource)?;
            Ok(())
        })
    }
    pub(super) fn checkpoint(
        &mut self,
        frames: &[RuntimeFrame<'_>],
        indices: &[usize],
        memory: &Memory,
        limits: SimulationDebugCaptureLimitsV1,
        phase: SimulationDebugCheckpointPhaseV1,
    ) -> Option<SimulationDebugRecordKindV1> {
        if self.records.len() == self.limit {
            self.stop = Some(PhysicalEntryDebugCaptureStopV20::RecordLimit);
            return None;
        }
        let result = self.ledger.with_budget(|budget| {
            snapshot::capture(frames, indices, memory, limits, phase, self.profile, budget)
        });
        match result {
            Ok(kind) => Some(kind),
            Err(error) => {
                self.stop = Some(match error {
                    snapshot::Failure::Resource(error) => {
                        PhysicalEntryDebugCaptureStopV20::Resource(error)
                    }
                    snapshot::Failure::Unavailable => {
                        PhysicalEntryDebugCaptureStopV20::SnapshotUnavailable
                    }
                });
                None
            }
        }
    }
    pub(super) fn record(
        &mut self,
        record: SimulationDebugRecordV1,
    ) -> SimulationDebugSinkControlV1 {
        if self.records.len() == self.limit {
            self.stop = Some(PhysicalEntryDebugCaptureStopV20::RecordLimit);
            // Checkpoints reserve before growth, so this branch is reached only
            // for a fixed-size memory row (no newly retained checkpoint heap).
            return SimulationDebugSinkControlV1::DropAndStop;
        }
        if let Err(error) = self.ledger.with_budget(|budget| budget.charge_work(1)) {
            // Release a newly constructed checkpoint payload after dropping it.
            let bytes = snapshot::heap_bytes(&record.kind);
            drop(record);
            if let Some(bytes) = bytes {
                self.ledger
                    .with_budget(|budget| budget.release_storage(bytes))
                    .expect("checkpoint payload already reserved");
            }
            self.stop = Some(PhysicalEntryDebugCaptureStopV20::Resource(error));
            return SimulationDebugSinkControlV1::DropAndStop;
        }
        // Entire record-vector capacity was charged and reserved at initialization.
        self.records.push(record);
        SimulationDebugSinkControlV1::Continue
    }
}

impl AdmittedSimulationModuleV1 {
    /// Runs the unchanged Engine with typed, fully represented V20 snapshots.
    /// The input owner must be the exact admitted canonical identity/profile.
    /// Takes the caller's existing ledger without resetting work/storage/history.
    /// Its existing floor remains caller-owned; engine scratch is prepaid and
    /// released, while all retained snapshot/vector charges stay with the result.
    ///
    /// This is CPU observation only. Reverse navigation does not resume execution.
    /// V21, generic physical registers, source maps and diagnosis are not admitted.
    pub fn capture_physical_entry_debug_v20(
        &self,
        canonical: &VerifiedCanonicalKernelIrModuleV20,
        request: &SimulationRequestV1,
        options: PhysicalEntryDebugOptionsV20,
        ledger: OwnedBudget,
    ) -> PhysicalEntryDebugCaptureV20 {
        self.capture_physical_debug_owned(
            CanonicalOwner::Entry(canonical),
            request,
            options,
            ledger,
        )
    }

    pub(super) fn capture_physical_debug_owned(
        &self,
        canonical: CanonicalOwner<'_>,
        request: &SimulationRequestV1,
        options: PhysicalEntryDebugOptionsV20,
        ledger: OwnedBudget,
    ) -> PhysicalEntryDebugCaptureV20 {
        let profile = canonical.profile();
        let mut capture = PhysicalEntryDebugCaptureV20 {
            state: State::new(ledger, options.records, profile),
            identity: self.identity,
            outcome: PhysicalEntryDebugOutcomeV20::NotStarted,
            error: None,
        };
        let initial = capture.state.ledger.with_budget(|budget| {
            budget.charge_work(
                usize::try_from(self.identity.canonical_length()).unwrap_or(usize::MAX),
            )
        });
        if let Err(error) = initial {
            capture.error = Some(PhysicalEntryDebugCaptureErrorV20::Resource(error));
            return capture;
        }
        if self.identity != canonical.identity() {
            capture.error = Some(PhysicalEntryDebugCaptureErrorV20::OwnerMismatch);
            return capture;
        }
        // Admission's immutable module and identity cannot be replaced by callers.
        if !profile.matches(self) {
            capture.error = Some(PhysicalEntryDebugCaptureErrorV20::NotPhysicalEntry);
            return capture;
        }
        // Reuse the existing preflight bounds before it grows any owned graph,
        // diagnostic, or plan buffers. The immutable caller inputs remain at the
        // original floor; conservative double-counting is intentional here.
        let preliminary = capture.state.ledger.with_budget(|budget| {
            let scan = request
                .arguments
                .len()
                .checked_add(request.shared_buffers.len())
                .and_then(|n| n.checked_add(canonical.bytes_len()))
                .ok_or(Resource::Arithmetic)?;
            budget.charge_work(scan)?;
            let bytes = crate::preflight::conservative_preflight_input_bytes(
                self.admitted_resident_bytes,
                &self.module,
                request,
            )
            .and_then(|n| {
                n.checked_add(crate::preflight::conservative_preflight_scratch_bytes(
                    &self.module,
                    request,
                    options.simulation,
                )?)
            })
            .ok_or(Resource::Arithmetic)?;
            budget.reserve_storage(bytes)?;
            Ok::<usize, Resource>(bytes)
        });
        let preflight_bytes = match preliminary {
            Ok(bytes) => bytes,
            Err(error) => {
                capture.error = Some(PhysicalEntryDebugCaptureErrorV20::Resource(error));
                return capture;
            }
        };
        let plan =
            match self.preflight(request, SimulationTargetV1::amdgpu_64(), options.simulation) {
                Ok(plan) => plan,
                Err(error) => {
                    // Drop any diagnostic ownership before releasing the prepaid phase.
                    drop(error);
                    capture
                        .state
                        .ledger
                        .with_budget(|budget| budget.release_storage(preflight_bytes))
                        .expect("reserved preflight envelope");
                    capture.outcome = PhysicalEntryDebugOutcomeV20::PreflightRefused;
                    return capture;
                }
            };
        let engine_bytes = plan.resident_bytes.max(preflight_bytes);
        if let Err(error) = capture
            .state
            .ledger
            .with_budget(|budget| budget.reserve_storage(engine_bytes - preflight_bytes))
        {
            drop(plan);
            capture
                .state
                .ledger
                .with_budget(|budget| budget.release_storage(preflight_bytes))
                .expect("reserved preflight envelope");
            capture.error = Some(PhysicalEntryDebugCaptureErrorV20::Resource(error));
            return capture;
        }
        if let Err(error) = capture.state.initialize() {
            capture.error = Some(error);
            drop(plan);
            capture
                .state
                .ledger
                .with_budget(|budget| budget.release_storage(engine_bytes))
                .expect("reserved engine scratch");
            return capture;
        }
        let mut events = NoopSimulationEventSinkV1;
        let mut debug = NoopSimulationDebugSinkV1;
        let result = execute_with_physical_debug(
            self,
            request,
            ExecutionConfiguration {
                target: SimulationTargetV1::amdgpu_64(),
                limits: options.simulation,
                policy: request.events,
                plan,
                debug_capture: options.capture,
                schedule: None,
                resident_offset: 0,
                allocation_reuse: None,
                matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
            },
            &mut events,
            &mut debug,
            Some(&mut capture.state),
        );
        capture.outcome = if result.is_ok() {
            PhysicalEntryDebugOutcomeV20::Completed
        } else {
            PhysicalEntryDebugOutcomeV20::ExecutionFailed
        };
        drop(result); // Result buffers/errors coexist with the prepaid engine envelope.
        capture
            .state
            .ledger
            .with_budget(|budget| budget.release_storage(engine_bytes))
            .expect("reserved engine scratch remains separate from snapshots");
        capture
    }
}

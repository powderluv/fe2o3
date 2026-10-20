//! Actual V5 publication custody retained in the original compiler cleanup slot.
//! Connection replacement cannot release either the lease or the locked token.
use super::*;
use fe2o3_artifact_transaction::{
    ArtifactLockRetirementBarrierErrorV1 as BarrierError,
    ArtifactLockRetirementBarrierV1 as Barrier,
    CompilerModuleHandoffCurrentnessCustodyQuoteV5 as Quote,
    CompilerModuleHandoffCustodyResourcesV5 as CustodyResources,
    MAX_COMPILER_MODULE_HANDOFF_BYTES_V5,
    try_acquire_artifact_lock_retirement_barrier_v1 as retirement_barrier,
    try_recover_compiler_module_handoff_receipt_in_root_budget_v5 as try_recover,
};
use fe2o3_protected_service_spawn::{
    LateRetainedCustodyV2 as Holder, ProtectedServiceCleanupServiceV2 as Cleanup,
    cleanup_bridge::{LateRetainedBuildV2 as Build, LateRetainedPayloadV2 as Payload},
    native_spawn::{
        ProtectedServiceSpawnErrorV2 as SpawnError, RootRetainedTaskTraceV2 as Trace,
        RootTaskObservationV2 as RootObservation,
    },
};

#[path = "compiler_execution_root_publication_payload.rs"]
mod payload;
use fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1 as Enrollment;
use payload::{Acquire, Owners, Validate};

#[path = "compiler_execution_root_publication_enrollment.rs"]
mod enrollment;

#[path = "compiler_execution_root_publication_quota.rs"]
mod quota;
pub use quota::RootPublicationQuotaV3;

#[path = "compiler_execution_root_publication_trace.rs"]
mod trace_owner;

const ENTRY: usize = 8;
const LOCAL_WORK: usize = ENTRY + 64 * 1088;
// Expected::derive and join_subject visit/copy the complete admitted descriptor.
// Their work is additional to nested native-observation and artifact operations.
const OBSERVE_WORK: usize = LOCAL_WORK + NativeOccurrence::WORK;
const FRAME: usize = 16 * fe2o3_rustc_invocation::MAX_DESCRIPTOR_BYTES_V3 + 8192;
type Result<T> = std::result::Result<T, RootPublicationCustodyErrorV3>;

/// Originating-thread handle to actual, still-locked V5 publication custody.
/// The original cleanup slot independently retains all acquired owners. Dropping
/// this handle or replacing an issuer connection does NOT release the locks.
/// Failure after slot installation is terminal for this observation attempt:
/// cancel the same compiler trace and retain its original cleanup controller.
/// Pending/quarantined cancellation is not permission to release its custody.
///
/// This observes a compiler/publication association, not protected proof execution
/// or durable retirement permission. It grants no publication, load or GPU launch
/// authority. Explicit retirement is private to the authenticated root dispatcher.
///
/// ```compile_fail
/// use fe2o3_broker_authority_service::RootPublicationCustodyV3 as C;
/// fn copy(c: C) { let _ = c.clone(); }
/// ```
/// ```compile_fail
/// use fe2o3_broker_authority_service::RootPublicationCustodyV3 as C;
/// fn send<T: Send>() {} send::<C>();
/// ```
pub struct RootPublicationCustodyV3 {
    observation: NativeObservation,
    owners: Holder<Owners>,
    subject: Subject,
    identity: [u8; 32],
    retained: usize,
    validation: RootPublicationQuotaV3,
}

pub(crate) struct PreparedRootPublicationRetirementV3<'slot>(
    fe2o3_protected_service_spawn::PreparedLateRetirementV2<'slot, Owners>,
);
impl PreparedRootPublicationRetirementV3<'_> {
    pub(crate) fn commit(self) {
        self.0.commit();
    }
}

impl RootPublicationCustodyV3 {
    /// Observe through the actual trace, then fund and install custody before
    /// acquiring the publication lease or token. The same original Budget is
    /// used throughout. Returned storage is FULL and unreserved; the independent
    /// persistent cleanup charge is paid before any long-lived lock is acquired.
    /// The compiler must have completed its authenticated exec/alias closure.
    /// Use the same original Owned::with_budget view through observation and
    /// revalidation. Its whole-request cap is unchanged; artifact operations
    /// additionally enforce their local 256 MiB working-set windows.
    pub fn observe<T: Send + 'static>(
        trace: &mut Trace<'_, T>,
        cleanup: &mut Cleanup,
        b: &mut Budget<'_>,
    ) -> Result<(Self, usize)> {
        Self::observe_with_limit(trace, cleanup, MAX_COMPILER_MODULE_HANDOFF_BYTES_V5, b)
    }

    /// Same observation with an explicit inert payload ceiling. A larger durable
    /// record refuses before payload reading; this neither selects an authority
    /// provider nor changes the schema's resource limits. Use observation_quota
    /// and observation_cleanup_quota to fund the original accounts beforehand.
    pub fn observe_with_limit<T: Send + 'static>(
        trace: &mut Trace<'_, T>,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        b: &mut Budget<'_>,
    ) -> Result<(Self, usize)> {
        Self::observe_owned(trace, cleanup, maximum_handoff_bytes, b)
    }

    /// Same publication protocol through the original runtime trace. The caller
    /// retains complete stopped-task/runtime enforcement and cancellation duties;
    /// this observation grants neither compiler resume nor retirement permission.
    pub fn observe_runtime_with_limit(
        trace: &mut fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1<'_>,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        b: &mut Budget<'_>,
    ) -> Result<(Self, usize)> {
        Self::observe_owned(trace, cleanup, maximum_handoff_bytes, b)
    }

    fn observe_owned<'work>(
        trace: &mut impl trace_owner::PublicationTrace<'work>,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        b: &mut Budget<'_>,
    ) -> Result<(Self, usize)> {
        b.with_prepaid_scope(trace.retained_storage(), ENTRY, OBSERVE_WORK, FRAME, |b| {
            let bounds = quota::custody_bounds(maximum_handoff_bytes)?;
            let (observation, observed_storage) = trace.with_task_observation(b, |root, b| {
                Ok::<_, RootPublicationCustodyErrorV3>(NativeObservation::observe_from(
                    Source::Root(root),
                    b,
                )?)
            })?;
            b.reserve_storage(observed_storage)?;
            let expected =
                Expected::derive(observation.descriptor()).map_err(NativeOccurrenceError::from)?;
            let output = observation.output_dir();
            bounds
                .validate_inputs(&output, &expected.producer)
                .map_err(NativeOccurrenceError::from)?;
            let receipt = {
                // Recovery may create temporary OutputLocks on failing paths.
                // Exclude spawns through their complete rollback/destruction.
                let barrier = retirement_barrier()?;
                try_recover(
                    &output,
                    &expected.producer,
                    expected.attempt,
                    maximum_handoff_bytes,
                    &barrier,
                    b,
                )
                .map_err(NativeOccurrenceError::from)?
            };
            let quote = bounds
                .quote_currentness(&output, &expected.producer, receipt)
                .map_err(NativeOccurrenceError::from)?;
            let validation = quota::revalidation(
                quote
                    .currentness_revalidation_quota()
                    .map_err(NativeOccurrenceError::from)?,
                quote.retained_storage(),
            )?;
            let payload_storage = quote
                .retained_storage()
                .checked_add(size_of::<Owners>())
                .ok_or(Resource::Arithmetic)?;
            // SAFETY: the artifact quote covers the exact receipt's complete
            // future lease/token backing. Owners only adds those two owners and
            // retires them together under an independent spawn barrier.
            let (owners, charge) =
                unsafe { trace.reserve_late_custody::<Owners>(cleanup, payload_storage, b) }?;
            b.reserve_storage(charge.additional_storage())?;
            let payload = Owners::new(quote, b)?;
            trace.prepare_late_attachment(&owners, b)?.commit(payload);
            let mut observed = None;
            trace.with_task_observation(b, |root, b| {
                // SAFETY: the operation borrows this trace's own observation and
                // returns only inert subject bytes. Every acquired owner stays
                // in the already funded slot, including on late scope refusal.
                unsafe {
                    trace.build_late_custody(
                        &owners,
                        Acquire {
                            root,
                            observation: &observation,
                            expected: &expected,
                            receipt,
                            output: &mut observed,
                        },
                        b,
                    )
                }
            })?;
            let (subject, identity) = observed.ok_or(RootPublicationCustodyErrorV3::state(
                "publication observation produced no subject",
            ))?;
            b.reserve_storage(size_of::<(Subject, SubjectStorage)>())?;
            let retained = observed_storage
                .checked_add(owners.retained_storage())
                .and_then(|n| n.checked_add(size_of::<(Subject, SubjectStorage)>()))
                .and_then(|n| n.checked_add(size_of::<(Self, usize)>()))
                .ok_or(Resource::Arithmetic)?;
            Ok((
                Self {
                    observation,
                    owners,
                    subject,
                    identity,
                    retained,
                    validation,
                },
                retained,
            ))
        })
    }

    /// Revalidate the same original trace and locked publication. No new lock,
    /// payload reconstruction, or replacement compiler occurrence is admitted.
    pub fn revalidate<T: Send + 'static>(
        &self,
        trace: &Trace<'_, T>,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        self.revalidate_owned(trace, b)
    }

    /// Revalidate through the same advanced trace, never a replacement process
    /// or a caller-supplied observation. A retired task cannot be observed here.
    pub fn revalidate_runtime(
        &self,
        trace: &fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1<'_>,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        self.revalidate_owned(trace, b)
    }

    /// Compare the actual locked capsule with original request coordinates.
    /// None means joined original absence, never a missing request join. This
    /// establishes neither full root/source equivalence nor launch authority.
    pub(crate) fn revalidate_runtime_with_enrollment(
        &self,
        trace: &fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1<'_>,
        expected: &Option<Enrollment>,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        self.revalidate_selected(trace, Some(expected), b)
    }

    fn revalidate_owned<'work>(
        &self,
        trace: &impl trace_owner::PublicationTrace<'work>,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        self.revalidate_selected(trace, None, b)
    }

    fn revalidate_selected<'work>(
        &self,
        trace: &impl trace_owner::PublicationTrace<'work>,
        enrollment: Option<&Option<Enrollment>>,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        let floor = self
            .retained
            .checked_add(trace.retained_storage())
            .ok_or(Resource::Arithmetic)?;
        b.with_prepaid_scope(floor, ENTRY, LOCAL_WORK, FRAME, |b| {
            trace.with_task_observation(b, |root, b| {
                // SAFETY: Validate only checks the installed concrete owners;
                // it cannot remove, replace, export or release their custody.
                unsafe {
                    trace.build_late_custody(
                        &self.owners,
                        Validate {
                            root,
                            observation: &self.observation,
                            enrollment,
                        },
                        b,
                    )
                }
            })
        })
    }

    /// Inert coordinates only; consumers must separately authenticate authority.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }
    pub const fn retained_storage(&self) -> usize {
        self.retained
    }

    /// Complete work/extra peak above this owner and the full original trace.
    pub const fn revalidation_quota(&self) -> RootPublicationQuotaV3 {
        self.validation
    }

    /// # Safety
    /// The closed root dispatcher must first authenticate its measured issuer's
    /// successful durable Worker/anchor/Ready/ACK join for this exact occurrence.
    /// An inert carriage or matching digest alone never authorizes this call.
    /// The returned closure retains custody on Drop; invoke it only after all
    /// reply/tombstone preparation and accounting have succeeded.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn prepare_durable_retirement_runtime<'slot>(
        &'slot self,
        trace: &'slot fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1<'_>,
        b: &mut Budget<'_>,
    ) -> Result<Option<PreparedRootPublicationRetirementV3<'slot>>> {
        // SAFETY: this private caller supplies independent exact-occurrence
        // durable authorization; the trace checks the original slot and account.
        Ok(unsafe { trace.prepare_late_retirement(&self.owners, b) }?
            .map(PreparedRootPublicationRetirementV3))
    }
}

/// Inert refusal, never an extracted publication owner or lock.
#[derive(Debug)]
pub struct RootPublicationCustodyErrorV3(Failure);
#[derive(Debug)]
enum Failure {
    Occurrence(NativeOccurrenceError),
    Spawn(SpawnError),
    Barrier(BarrierError),
    State(&'static str),
}
impl RootPublicationCustodyErrorV3 {
    fn state(message: &'static str) -> Self {
        Self(Failure::State(message))
    }
}
impl From<NativeOccurrenceError> for RootPublicationCustodyErrorV3 {
    fn from(e: NativeOccurrenceError) -> Self {
        Self(Failure::Occurrence(e))
    }
}
impl From<NativeObservationError> for RootPublicationCustodyErrorV3 {
    fn from(e: NativeObservationError) -> Self {
        NativeOccurrenceError::from(e).into()
    }
}
impl From<Resource> for RootPublicationCustodyErrorV3 {
    fn from(e: Resource) -> Self {
        NativeOccurrenceError::from(e).into()
    }
}
impl From<SpawnError> for RootPublicationCustodyErrorV3 {
    fn from(e: SpawnError) -> Self {
        Self(Failure::Spawn(e))
    }
}
impl From<BarrierError> for RootPublicationCustodyErrorV3 {
    fn from(e: BarrierError) -> Self {
        Self(Failure::Barrier(e))
    }
}
impl std::fmt::Display for RootPublicationCustodyErrorV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Failure::Occurrence(e) => e.fmt(f),
            Failure::Spawn(e) => e.fmt(f),
            Failure::Barrier(e) => e.fmt(f),
            Failure::State(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for RootPublicationCustodyErrorV3 {}

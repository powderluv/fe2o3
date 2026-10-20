//! Composition of the concrete observation, artifact and original-slot schedules.
use super::*;
use fe2o3_artifact_transaction::{
    COMPILER_MODULE_HANDOFF_CUSTODY_COMPOSITION_SCRATCH_V5 as COMPOSITION_SCRATCH,
    COMPILER_MODULE_HANDOFF_CUSTODY_COMPOSITION_WORK_V5 as COMPOSITION_WORK,
    CompilerModuleHandoffCustodyQuotaV5 as CustodyBounds,
    CompilerModuleHandoffOperationQuotaV5 as ArtifactQuota,
    INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3 as SUBJECT_SCRATCH,
    INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3 as SUBJECT_WORK,
    compiler_module_handoff_custody_quota_for_limit_v5,
};
use fe2o3_protected_service_spawn::LateRetainedQuotaV2;
use fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1;

/// Cumulative request work and peak additional storage above retained inputs.
/// An inert bound only; it grants no authority and creates or resets no account.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootPublicationQuotaV3 {
    work: usize,
    scratch: usize,
}
impl RootPublicationQuotaV3 {
    pub const fn work(self) -> usize {
        self.work
    }
    pub const fn scratch(self) -> usize {
        self.scratch
    }
}

fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |sum, n| {
        sum.checked_add(*n)
            .ok_or_else(|| Resource::Arithmetic.into())
    })
}

pub(super) fn custody_bounds(limit: usize) -> Result<CustodyBounds> {
    Ok(compiler_module_handoff_custody_quota_for_limit_v5(limit)
        .map_err(NativeOccurrenceError::from)?)
}

fn late(bounds: CustodyBounds) -> Result<LateRetainedQuotaV2> {
    Ok(Holder::<Owners>::quota(sum(&[
        bounds.retained_storage(),
        size_of::<Owners>(),
    ])?)?)
}

impl RootPublicationCustodyV3 {
    pub(crate) fn maximum_enrollment_revalidation_quota(
        maximum_handoff_bytes: usize,
    ) -> Result<RootPublicationQuotaV3> {
        let base = Self::maximum_revalidation_quota(maximum_handoff_bytes)?;
        Ok(RootPublicationQuotaV3 {
            work: sum(&[base.work(), enrollment::WORK])?,
            scratch: sum(&[base.scratch(), enrollment::SCRATCH])?,
        })
    }

    /// Exact-slot retirement preparation bound. This is only resource planning;
    /// the closed root dispatcher must independently establish durable authority.
    pub fn maximum_retirement_quota(
        maximum_handoff_bytes: usize,
    ) -> Result<RootPublicationQuotaV3> {
        let late = late(custody_bounds(maximum_handoff_bytes)?)?;
        Ok(RootPublicationQuotaV3 {
            work: late.retirement_work(),
            scratch: Holder::<Owners>::retirement_scratch()?,
        })
    }

    /// Inert ceiling for revalidating a publication admitted under this same
    /// byte limit. Actual observation still captures its exact retained quote.
    pub fn maximum_revalidation_quota(
        maximum_handoff_bytes: usize,
    ) -> Result<RootPublicationQuotaV3> {
        let bounds = custody_bounds(maximum_handoff_bytes)?;
        revalidation(
            bounds.currentness_revalidation_quota(),
            bounds.retained_storage(),
        )
    }

    /// Complete conservative request schedule for observe_with_limit, including
    /// two live revalidations, both artifact owners and the returned handle.
    /// Invalid or unaffordable ceilings must refuse on the original Budget.
    /// The limit bounds bytes, never compiler identity or publication authority.
    pub fn observation_quota(maximum_handoff_bytes: usize) -> Result<RootPublicationQuotaV3> {
        let bounds = custody_bounds(maximum_handoff_bytes)?;
        let late = late(bounds)?;
        let recover = bounds.try_recovery_quota();
        let lease = bounds.lease_acquisition_quota();
        let token = bounds.token_acquisition_quota();
        let current = bounds.currentness_revalidation_quota();
        let (observe_work, observe_scratch) = NativeObservation::root_operation_quota(false)?;
        let (validate_work, validate_scratch) = NativeObservation::root_operation_quota(true)?;
        let subject = size_of::<(Subject, SubjectStorage)>();
        let acquire_scratch = sum(&[
            Holder::<Owners>::ATTACH_SCRATCH,
            FRAME,
            // Returned lease/token charges accumulate in the builder scope.
            bounds.retained_storage(),
            // No Holder floor is excluded: full quote overlaps AGAIN locally.
            bounds.retained_storage(),
            COMPOSITION_SCRATCH,
            subject,
            validate_scratch
                .max(lease.scratch())
                .max(token.scratch())
                .max(current.scratch())
                .max(SUBJECT_SCRATCH),
        ])?;
        let observed = NativeObservation::ROOT_RETAINED_MAX;
        let construction = sum(&[
            observed,
            late.retained_storage(),
            sum(&[RootObservation::VIEW_SCRATCH, acquire_scratch])?
                .max(Holder::<Owners>::ATTACH_SCRATCH)
                .max(sum(&[subject, size_of::<(Self, usize)>()])?),
        ])?;
        Ok(RootPublicationQuotaV3 {
            work: sum(&[
                OBSERVE_WORK,
                observe_work,
                2 * RootObservation::VIEW_WORK,
                2 * RootRuntimeTraceV1::ROOT_OBSERVATION_WORK,
                recover.work(),
                late.request_work(),
                2 * Holder::<Owners>::ATTACH_WORK,
                OBSERVE_WORK,
                validate_work,
                validate_work,
                lease.work(),
                token.work(),
                SUBJECT_WORK,
                current.work(),
                4 * COMPOSITION_WORK,
                CustodyResources::PREPARE_WORK,
            ])?,
            scratch: sum(&[
                FRAME,
                sum(&[RootObservation::VIEW_SCRATCH, observe_scratch])?
                    .max(sum(&[observed, COMPOSITION_SCRATCH, recover.scratch()])?)
                    .max(sum(&[observed, late.scratch()])?)
                    .max(construction),
            ])?,
        })
    }

    /// Additional funding for the existing cleanup service, not a new service.
    /// Keep its current pool and all existing payloads prepaid independently.
    /// Besides work()/persistent_storage(), fund retirement_work() for every
    /// planned pump turn using the cleanup service's existing finite schedule.
    pub fn observation_cleanup_quota(maximum_handoff_bytes: usize) -> Result<LateRetainedQuotaV2> {
        late(custody_bounds(maximum_handoff_bytes)?)
    }
}

pub(super) fn revalidation(
    current: ArtifactQuota,
    owners: usize,
) -> Result<RootPublicationQuotaV3> {
    let (work, scratch) = NativeObservation::root_operation_quota(true)?;
    Ok(RootPublicationQuotaV3 {
        work: sum(&[
            LOCAL_WORK,
            RootObservation::VIEW_WORK,
            RootRuntimeTraceV1::ROOT_OBSERVATION_WORK,
            Holder::<Owners>::ATTACH_WORK,
            LOCAL_WORK,
            work,
            current.work(),
            COMPOSITION_WORK,
        ])?,
        scratch: sum(&[
            FRAME,
            RootObservation::VIEW_SCRATCH,
            Holder::<Owners>::ATTACH_SCRATCH,
            FRAME,
            scratch.max(sum(&[owners, COMPOSITION_SCRATCH, current.scratch()])?),
        ])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_publication_plan_covers_whole_request_custody_overlap() {
        let q = RootPublicationCustodyV3::observation_quota(1024 * 1024).unwrap();
        let late = RootPublicationCustodyV3::observation_cleanup_quota(1024 * 1024).unwrap();
        let bounds = custody_bounds(1024 * 1024).unwrap();
        let validation = RootPublicationCustodyV3::maximum_revalidation_quota(1024 * 1024).unwrap();
        assert_eq!(
            validation,
            revalidation(
                bounds.currentness_revalidation_quota(),
                bounds.retained_storage()
            )
            .unwrap()
        );
        // The enclosing request retains its Holder while the local window pays
        // the full quoted owners again and both constructor outputs accumulate.
        // Its aggregate peak is not the artifact-local 256 MiB ceiling.
        let overlap = sum(&[
            late.retained_storage(),
            bounds.retained_storage(),
            bounds.retained_storage(),
            COMPOSITION_SCRATCH,
            bounds
                .lease_acquisition_quota()
                .scratch()
                .max(bounds.token_acquisition_quota().scratch())
                .max(bounds.currentness_revalidation_quota().scratch())
                .max(SUBJECT_SCRATCH),
        ])
        .unwrap();
        assert!(q.scratch() >= overlap);
        assert!(q.work() > late.request_work());
        assert!(q.scratch() > late.retained_storage());
        let larger = RootPublicationCustodyV3::observation_quota(2 * 1024 * 1024).unwrap();
        assert!(larger.work() > q.work());
        assert!(larger.scratch() > q.scratch());
        for length in [0, usize::MAX] {
            assert!(RootPublicationCustodyV3::observation_quota(length).is_err());
            assert!(RootPublicationCustodyV3::observation_cleanup_quota(length).is_err());
            assert!(RootPublicationCustodyV3::maximum_revalidation_quota(length).is_err());
        }
        assert!(sum(&[usize::MAX, 1]).is_err());
    }
}

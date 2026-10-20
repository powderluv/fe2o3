//! Scoped access to the original native session and admitted rustc invocation.
use super::{Error as SessionError, Policy};
use crate::protected_rustc_invocation::{
    AdmittedProtectedRustcInvocationV1 as Invocation, ReferenceEnrollmentInvocationStampV1,
    reference_enrollment_identity::{Owner, Stamp},
};
use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
use crate::reference_enrollment_policy_v1::ReferenceEnrollmentRequestV1 as Request;
use crate::rustc_semantic_plan_v1::SourceClosureWorkV1 as Work;
use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
use fe2o3_rustc_invocation::RustcInvocationDescriptorV3;
use fe2o3_verifier::portable_reference_v1::codec::ReferenceEnrollmentOriginV1 as Origin;
use std::cell::{Cell, RefCell};

#[derive(Debug)]
pub(crate) struct RetainedReferenceEnrollmentStampV1 {
    session: Stamp,
    invocation: ReferenceEnrollmentInvocationStampV1,
}

/// Compiler-local custody, not guarded-root attempt or PREPARSE approval. This
/// private, move-only projection cannot supply an inventory header or authorize
/// parser storage. Complete live closure qualification/accounting remain required.
struct ProjectedEnrollmentRequestV1 {
    stamp: RetainedReferenceEnrollmentStampV1,
    origin: Origin,
    bindings: Option<usize>,
}

/// Descriptive header copied only after the original descriptor was checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OriginalEnrollmentInventoryContextV1 {
    pub(crate) descriptor_bindings: u32,
    pub(crate) rustc_invocation_sha256: [u8; 32],
    pub(crate) native_policy_sha256: [u8; 32],
    pub(crate) policy_generation: u64,
}

type Revalidate<'a> =
    dyn FnMut(Option<&ReferenceEnrollmentInvocationStampV1>) -> Result<(), Error> + 'a;

/// No constructor or authority-bearing trait is available to the collector.
pub(crate) struct ReferenceEnrollmentLoanV1<'a> {
    descriptor: &'a RustcInvocationDescriptorV3,
    invocation: &'a Invocation,
    owner: &'a Owner,
    revalidate: RefCell<&'a mut Revalidate<'a>>,
    refused: Cell<bool>,
    requested_bindings: Cell<Option<usize>>,
    origin: Origin,
    inventory_storage_request: &'a Cell<Option<usize>>,
    inventory_storage_limit: usize,
}

/// Borrowed only within the original client's checked preparation callback.
pub(crate) struct ReferenceEnrollmentPreparationV1<'a> {
    pub(super) policy: &'a Policy,
    pub(super) owner: Option<&'a Owner>,
    pub(super) ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
}

impl ReferenceEnrollmentPreparationV1<'_> {
    pub(crate) fn is_enrolled(&self) -> bool {
        self.owner.is_some()
    }

    pub(crate) fn with_invocation<T, E>(
        &self,
        invocation: &mut Invocation,
        budget: &mut Budget<'_>,
        run: impl FnOnce(&ReferenceEnrollmentLoanV1<'_>) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<SessionError>,
    {
        if self.ledger != budget.work_ledger_identity_v1() {
            return Err(SessionError::Reference(Error::new(
                "reference enrollment original target account changed",
            ))
            .into());
        }
        let owner = self.owner.ok_or_else(|| {
            SessionError::Reference(Error::new("reference enrollment session was not retained"))
        })?;
        self.policy.revalidate(budget).map_err(SessionError::from)?;
        let original = invocation
            .retain_reference_enrollment_identity(budget)
            .map_err(SessionError::Reference)?;
        let rustc_invocation_sha256 = invocation
            .reference_enrollment_invocation_sha256(budget)
            .map_err(SessionError::Reference)?;
        let invocation = &*invocation;
        let origin = Origin {
            rustc_invocation_sha256,
            native_policy_sha256: *self.policy.policy().identity().as_bytes(),
            policy_generation: self.policy.policy().generation(),
            mapping_ordinal: 0,
        };
        let inventory_storage_request = Cell::new(None);
        let inventory_storage_limit = budget.storage_limit();
        let mut storage_floor = budget.storage();
        let mut revalidate = |retained: Option<&ReferenceEnrollmentInvocationStampV1>| {
            check_inventory_storage_floor(budget, self.ledger, storage_floor)?;
            self.policy
                .revalidate(budget)
                .map_err(|error| Error::new(error.to_string()))?;
            invocation
                .revalidate_reference_enrollment_identity(retained.unwrap_or(&original), budget)?;
            check_inventory_storage_floor(budget, self.ledger, storage_floor)?;
            if let Some(bytes) = inventory_storage_request.take() {
                reserve_inventory_storage(budget, self.ledger, &mut storage_floor, bytes)?;
            }
            check_inventory_storage_floor(budget, self.ledger, storage_floor)?;
            Ok(())
        };
        let loan = ReferenceEnrollmentLoanV1 {
            descriptor: invocation.descriptor(),
            invocation,
            owner,
            revalidate: RefCell::new(&mut revalidate),
            refused: Cell::new(false),
            requested_bindings: Cell::new(None),
            origin,
            inventory_storage_request: &inventory_storage_request,
            inventory_storage_limit,
        };
        let result = run(&loan);
        // An ignored refusal cannot be turned into a successful preparation.
        if loan.refused.get() {
            return Err(SessionError::Reference(Error::new(
                "reference enrollment loan was refused",
            ))
            .into());
        }
        let value = result?;
        loan.current(None).map_err(SessionError::Reference)?;
        Ok(value)
    }
}

fn check_inventory_storage_floor(
    budget: &Budget<'_>,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    storage_floor: usize,
) -> Result<(), Error> {
    if ledger != budget.work_ledger_identity_v1() || budget.storage() < storage_floor {
        return Err(Error::new(
            "reference enrollment original inventory account or retained storage changed",
        ));
    }
    budget
        .check_prior_denials_v1()
        .map_err(|error| Error::new(error.to_string()))
}

fn reserve_inventory_storage(
    budget: &mut Budget<'_>,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    storage_floor: &mut usize,
    bytes: usize,
) -> Result<(), Error> {
    check_inventory_storage_floor(budget, ledger, *storage_floor)?;
    budget
        .reserve_storage(bytes)
        .map_err(|error| Error::new(error.to_string()))?;
    *storage_floor = budget.storage();
    check_inventory_storage_floor(budget, ledger, *storage_floor)
}

impl ReferenceEnrollmentLoanV1<'_> {
    pub(crate) fn inventory_context(
        &self,
        work: &mut Work,
    ) -> Result<Option<OriginalEnrollmentInventoryContextV1>, Error> {
        self.charge(
            work,
            1 + std::mem::size_of::<OriginalEnrollmentInventoryContextV1>(),
        )?;
        self.current(None)?;
        let Some(count) = self.requested_bindings.get() else {
            return Ok(None);
        };
        let descriptor_bindings = self.terminal(
            u32::try_from(count).map_err(|_| Error::new("inventory descriptor count overflow")),
        )?;
        Ok(Some(OriginalEnrollmentInventoryContextV1 {
            descriptor_bindings,
            rustc_invocation_sha256: self.origin.rustc_invocation_sha256,
            native_policy_sha256: self.origin.native_policy_sha256,
            policy_generation: self.origin.policy_generation,
        }))
    }

    /// Logical allocations stay charged on opaque failure or unwind.
    pub(crate) fn reserve_inventory_storage(&self, bytes: usize) -> Result<(), Error> {
        if self.requested_bindings.get().is_none()
            || self
                .inventory_storage_request
                .replace(Some(bytes))
                .is_some()
        {
            return self.terminal(Err(Error::new(
                "inventory storage has no checked original request",
            )));
        }
        self.current(None)?;
        if self.inventory_storage_request.get().is_some() {
            return self.terminal(Err(Error::new("inventory storage debit was not consumed")));
        }
        Ok(())
    }

    pub(crate) fn inventory_storage_limit(&self) -> usize {
        self.inventory_storage_limit
    }

    fn terminal<T>(&self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.refused.set(true);
        }
        result
    }

    fn current(
        &self,
        retained: Option<&ReferenceEnrollmentInvocationStampV1>,
    ) -> Result<(), Error> {
        revalidate_live(&self.revalidate, &self.refused, retained)
    }

    fn charge(&self, work: &mut Work, amount: usize) -> Result<(), Error> {
        self.terminal(
            work.charge(amount)
                .map_err(|error| Error::new(error.to_string())),
        )
    }

    pub(crate) fn request(&self, work: &mut Work) -> Result<Option<Request>, Error> {
        let projection = self.project_request(work)?;
        self.decode_projected_request(projection, work)
    }

    fn project_request(&self, work: &mut Work) -> Result<ProjectedEnrollmentRequestV1, Error> {
        self.charge(work, std::mem::size_of::<ProjectedEnrollmentRequestV1>())?;
        let stamp = self.capture_stamp(work)?;
        let bindings = self.terminal(Request::project_binding_count(self.descriptor, work))?;
        self.revalidate_stamp(&stamp, work)?;
        Ok(ProjectedEnrollmentRequestV1 {
            stamp,
            origin: self.origin,
            bindings,
        })
    }

    fn decode_projected_request(
        &self,
        projection: ProjectedEnrollmentRequestV1,
        work: &mut Work,
    ) -> Result<Option<Request>, Error> {
        self.revalidate_stamp(&projection.stamp, work)?;
        if projection.origin != self.origin {
            return self.terminal(Err(Error::new(
                "reference enrollment projected descriptor or policy changed",
            )));
        }
        let request = self.terminal(Request::from_descriptor(self.descriptor, work))?;
        if request.as_ref().map(|request| request.bindings().len()) != projection.bindings {
            return self.terminal(Err(Error::new(
                "reference enrollment projected count differs from owned request",
            )));
        }
        self.revalidate_stamp(&projection.stamp, work)?;
        self.requested_bindings.set(projection.bindings);
        Ok(request)
    }

    pub(crate) fn capture_stamp(
        &self,
        work: &mut Work,
    ) -> Result<RetainedReferenceEnrollmentStampV1, Error> {
        self.charge(
            work,
            std::mem::size_of::<RetainedReferenceEnrollmentStampV1>(),
        )?;
        self.current(None)?;
        Ok(RetainedReferenceEnrollmentStampV1 {
            session: self.owner.stamp(),
            invocation: self.terminal(self.invocation.reference_enrollment_stamp())?,
        })
    }

    pub(crate) fn revalidate_stamp(
        &self,
        stamp: &RetainedReferenceEnrollmentStampV1,
        work: &mut Work,
    ) -> Result<(), Error> {
        self.charge(work, 1)?;
        if !self.owner.matches(&stamp.session) {
            return self.terminal(Err(Error::new(
                "reference enrollment original native session changed",
            )));
        }
        self.current(Some(&stamp.invocation))
    }

    pub(crate) fn origin(&self, ordinal: u32) -> Result<Origin, Error> {
        self.current(None)?;
        if !self
            .requested_bindings
            .get()
            .is_some_and(|count| (ordinal as usize) < count)
        {
            return self.terminal(Err(Error::new(
                "reference enrollment ordinal is not in the captured request",
            )));
        }
        Ok(Origin {
            mapping_ordinal: ordinal,
            ..self.origin
        })
    }
}

fn revalidate_live(
    revalidate: &RefCell<&mut Revalidate<'_>>,
    refused: &Cell<bool>,
    retained: Option<&ReferenceEnrollmentInvocationStampV1>,
) -> Result<(), Error> {
    if refused.get() {
        return Err(Error::new("reference enrollment loan was refused"));
    }
    let result = revalidate
        .try_borrow_mut()
        .map_err(|_| Error::new("reference enrollment revalidation is reentrant"))
        .and_then(|mut revalidate| revalidate(retained));
    if result.is_err() {
        refused.set(true);
    }
    result?;
    if refused.get() {
        return Err(Error::new("reference enrollment loan was refused"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "protected_reference_enrollment_loan_v1_flow_tests.rs"]
mod flow_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_reservations_protect_accumulated_original_storage() {
        use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as TargetWork;
        for entry_floor in [0, 31] {
            let mut work = TargetWork::new(100);
            let mut budget = Budget::new(&mut work, entry_floor + 29);
            budget.charge_work(7).unwrap();
            budget.reserve_storage(entry_floor).unwrap();
            let ledger = budget.work_ledger_identity_v1();
            let mut floor = entry_floor;
            for amount in [0, 11, 18] {
                reserve_inventory_storage(&mut budget, ledger, &mut floor, amount).unwrap();
                assert_eq!(floor, budget.storage());
            }
            assert_eq!(floor, entry_floor + 29);
            assert_eq!(budget.work(), 7);
            budget.release_storage(1).unwrap();
            assert!(budget.storage() >= entry_floor);
            assert!(check_inventory_storage_floor(&budget, ledger, floor).is_err());
            assert!(reserve_inventory_storage(&mut budget, ledger, &mut floor, 0).is_err());
            assert_eq!(floor, entry_floor + 29);
        }
    }

    #[test]
    fn inventory_one_short_and_oversized_refusals_never_refund_or_restart() {
        use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as TargetWork;
        for failed in [19, usize::MAX] {
            let mut work = TargetWork::new(100);
            let mut budget = Budget::new(&mut work, 60);
            budget.reserve_storage(31).unwrap();
            let ledger = budget.work_ledger_identity_v1();
            let mut floor = 31;
            reserve_inventory_storage(&mut budget, ledger, &mut floor, 11).unwrap();
            assert!(reserve_inventory_storage(&mut budget, ledger, &mut floor, failed).is_err());
            let denial = budget.failed_storage();
            assert_eq!(
                (budget.storage(), floor, budget.peak_storage()),
                (42, 42, 42)
            );
            assert!(reserve_inventory_storage(&mut budget, ledger, &mut floor, 0).is_err());
            assert_eq!(budget.failed_storage(), denial);
            assert_eq!(budget.storage(), 42);
        }
    }

    #[test]
    fn inventory_failed_or_unwound_callbacks_keep_successful_prefix_debits() {
        use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as TargetWork;
        let mut work = TargetWork::new(100);
        let mut budget = Budget::new(&mut work, 60);
        budget.reserve_storage(31).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let mut floor = 31;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reserve_inventory_storage(&mut budget, ledger, &mut floor, 11).unwrap();
            panic!("opaque owner failure");
        }));
        assert!(result.is_err());
        assert_eq!((budget.storage(), floor), (42, 42));
        let mut other_work = TargetWork::new(100);
        let mut other = Budget::new(&mut other_work, 60);
        other.reserve_storage(42).unwrap();
        assert!(reserve_inventory_storage(&mut other, ledger, &mut floor, 0).is_err());
        assert_eq!((other.storage(), budget.storage(), floor), (42, 42, 42));
    }

    #[test]
    fn every_currentness_check_visits_the_original_revalidator() {
        let visits = Cell::new(0);
        let refused = Cell::new(false);
        let mut check = |_: Option<&ReferenceEnrollmentInvocationStampV1>| {
            visits.set(visits.get() + 1);
            Ok(())
        };
        let revalidate = RefCell::new(&mut check as &mut Revalidate<'_>);
        revalidate_live(&revalidate, &refused, None).unwrap();
        revalidate_live(&revalidate, &refused, None).unwrap();
        assert_eq!(visits.get(), 2);
        assert!(!refused.get());
    }

    #[test]
    fn reentrant_check_refuses_without_panicking_or_retrying() {
        let visits = Cell::new(0);
        let refused = Cell::new(false);
        let mut check = |_: Option<&ReferenceEnrollmentInvocationStampV1>| {
            visits.set(visits.get() + 1);
            Ok(())
        };
        let revalidate = RefCell::new(&mut check as &mut Revalidate<'_>);
        let active = revalidate.borrow_mut();
        assert!(
            revalidate_live(&revalidate, &refused, None)
                .unwrap_err()
                .to_string()
                .contains("reentrant")
        );
        drop(active);
        assert!(revalidate_live(&revalidate, &refused, None).is_err());
        assert!(refused.get());
        assert_eq!(visits.get(), 0);
    }

    #[test]
    fn ignored_refusal_cannot_reenter_a_successful_revalidator() {
        let visits = Cell::new(0);
        let refused = Cell::new(false);
        let mut check = |_: Option<&ReferenceEnrollmentInvocationStampV1>| {
            visits.set(visits.get() + 1);
            refused.set(true);
            Ok(())
        };
        let revalidate = RefCell::new(&mut check as &mut Revalidate<'_>);
        assert!(revalidate_live(&revalidate, &refused, None).is_err());
        assert!(revalidate_live(&revalidate, &refused, None).is_err());
        assert_eq!(visits.get(), 1);
    }
}

//! Move-stable identity for one original owner, not proof authority.
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, Ordering},
};

pub(crate) struct Owner(Arc<AtomicBool>);

#[derive(Debug)]
pub(crate) struct Stamp(Weak<AtomicBool>);

impl Owner {
    // Logical bound for the pinned Arc control block and live bit, not allocator RSS.
    pub(crate) const STORAGE: usize = 4 * std::mem::size_of::<usize>();

    pub(crate) fn new(budget: &mut Budget<'_>) -> Result<Self, Resource> {
        budget.charge_work(1)?;
        budget.reserve_storage(Self::STORAGE)?;
        Ok(Self(Arc::new(AtomicBool::new(true))))
    }

    pub(crate) fn stamp(&self) -> Stamp {
        Stamp(Arc::downgrade(&self.0))
    }

    pub(crate) fn matches(&self, stamp: &Stamp) -> bool {
        self.0.load(Ordering::Acquire) && Weak::ptr_eq(&stamp.0, &Arc::downgrade(&self.0))
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        // A retained observation must not keep an original owner logically alive.
        self.0.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

    #[test]
    fn moving_the_owner_preserves_identity_but_a_fresh_owner_does_not() {
        let mut work = Work::new(16);
        let mut budget = Budget::new(&mut work, 2 * Owner::STORAGE);
        let original = Owner::new(&mut budget).unwrap();
        let stamp = original.stamp();
        let moved = original;
        assert!(moved.matches(&stamp));
        let other = Owner::new(&mut budget).unwrap();
        assert!(!other.matches(&stamp));
    }

    #[test]
    fn an_upgraded_observation_cannot_keep_a_dropped_owner_live() {
        let mut work = Work::new(16);
        let mut budget = Budget::new(&mut work, Owner::STORAGE);
        let owner = Owner::new(&mut budget).unwrap();
        let stamp = owner.stamp();
        let retained_observation = stamp.0.upgrade().unwrap();
        drop(owner);
        assert!(!retained_observation.load(Ordering::Acquire));
        drop(retained_observation);
        assert!(stamp.0.upgrade().is_none());
    }

    #[test]
    fn exact_storage_preserves_inherited_charges_and_ledger_identity() {
        let mut work = Work::new(4);
        let mut budget = Budget::new(&mut work, 17 + Owner::STORAGE);
        budget.reserve_storage(17).unwrap();
        budget.charge_work(3).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let owner = Owner::new(&mut budget).unwrap();
        assert!(budget.work_ledger_identity_v1() == ledger);
        assert_eq!(budget.storage(), 17 + Owner::STORAGE);
        assert_eq!(budget.peak_storage(), 17 + Owner::STORAGE);
        assert_eq!(budget.work(), 4);
        assert_eq!(budget.failed_work(), None);
        assert_eq!(budget.failed_storage(), None);
        drop(owner);
        // Storage belongs to the enclosing original account, never a fresh refund.
        assert_eq!(budget.storage(), 17 + Owner::STORAGE);
    }

    #[test]
    fn work_and_one_short_storage_refuse_before_constructing_an_owner() {
        let mut work = Work::new(0);
        let mut budget = Budget::new(&mut work, Owner::STORAGE);
        assert!(matches!(Owner::new(&mut budget), Err(Resource::Work(error))
            if error.actual() == 1 && error.limit() == 0));
        assert_eq!(budget.work(), 0);
        assert_eq!(budget.failed_work(), Some(1));
        assert_eq!(budget.storage(), 0);
        assert_eq!(budget.peak_storage(), 0);
        assert_eq!(budget.failed_storage(), None);

        let mut work = Work::new(1);
        let mut budget = Budget::new(&mut work, Owner::STORAGE - 1);
        assert!(
            matches!(Owner::new(&mut budget), Err(Resource::Storage(error))
            if error.actual() == Owner::STORAGE && error.limit() == Owner::STORAGE - 1)
        );
        assert_eq!(budget.work(), 1);
        assert_eq!(budget.failed_work(), None);
        assert_eq!(budget.storage(), 0);
        assert_eq!(budget.peak_storage(), 0);
        assert_eq!(budget.failed_storage(), Some(Owner::STORAGE));
    }

    #[test]
    fn work_refusal_preserves_inherited_prefix_and_first_denial() {
        for prior_denial in [false, true] {
            let mut work = Work::new(3);
            let mut budget = Budget::new(&mut work, 17 + Owner::STORAGE);
            budget.reserve_storage(17).unwrap();
            budget.charge_work(3).unwrap();
            let ledger = budget.work_ledger_identity_v1();
            if prior_denial {
                assert!(matches!(budget.charge_work(4), Err(Resource::Work(error))
                    if error.actual() == 7 && error.limit() == 3));
            }
            assert!(matches!(Owner::new(&mut budget), Err(Resource::Work(error))
                if error.actual() == 4 && error.limit() == 3));
            assert_eq!(budget.work(), 3);
            assert_eq!(budget.failed_work(), Some(if prior_denial { 7 } else { 4 }));
            assert_eq!(budget.storage(), 17);
            assert_eq!(budget.peak_storage(), 17);
            assert_eq!(budget.failed_storage(), None);
            assert!(budget.work_ledger_identity_v1() == ledger);
        }
    }

    #[test]
    fn one_short_storage_preserves_inherited_prefix_and_first_denial() {
        let total_storage = 17 + Owner::STORAGE;
        for prior_denial in [false, true] {
            let mut work = Work::new(4);
            let mut budget = Budget::new(&mut work, total_storage - 1);
            budget.reserve_storage(17).unwrap();
            budget.charge_work(3).unwrap();
            let ledger = budget.work_ledger_identity_v1();
            if prior_denial {
                assert!(
                    matches!(budget.reserve_storage(Owner::STORAGE + 9), Err(Resource::Storage(error))
                    if error.actual() == total_storage + 9 && error.limit() == total_storage - 1)
                );
            }
            assert!(
                matches!(Owner::new(&mut budget), Err(Resource::Storage(error))
                if error.actual() == total_storage && error.limit() == total_storage - 1)
            );
            assert_eq!(budget.work(), 4);
            assert_eq!(budget.failed_work(), None);
            assert_eq!(budget.storage(), 17);
            assert_eq!(budget.peak_storage(), 17);
            assert_eq!(
                budget.failed_storage(),
                Some(total_storage + if prior_denial { 9 } else { 0 })
            );
            assert!(budget.work_ledger_identity_v1() == ledger);
        }
    }

    #[test]
    fn unwinding_invalidates_observations_without_refunding_original_storage() {
        let mut work = Work::new(4);
        let storage = 17 + Owner::STORAGE;
        let mut budget = Budget::new(&mut work, storage);
        budget.reserve_storage(17).unwrap();
        budget.charge_work(3).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let owner = Owner::new(&mut budget).unwrap();
        let stamp = owner.stamp();
        let observation = stamp.0.upgrade().unwrap();
        assert!(observation.load(Ordering::Acquire));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _owner = owner;
            panic!("identity owner unwind");
        }));
        assert!(result.is_err());
        assert!(!observation.load(Ordering::Acquire));
        assert!(stamp.0.upgrade().is_some());
        assert_eq!(budget.storage(), storage);
        drop(observation);
        assert!(stamp.0.upgrade().is_none());
        drop(stamp);
        assert_eq!(budget.storage(), storage);
        assert_eq!(budget.peak_storage(), storage);
        assert_eq!(budget.work(), 4);
        assert_eq!(budget.failed_work(), None);
        assert_eq!(budget.failed_storage(), None);
        assert!(budget.work_ledger_identity_v1() == ledger);
    }
}

//! Closed-route accounting components only; no fabricated Request/import owner.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use std::{
    cell::Cell,
    error::Error as _,
    panic::{AssertUnwindSafe, catch_unwind},
};

const CAPACITY: usize = 127;
const FLOOR: usize = METADATA + CAPACITY + 19;

#[path = "cpu_mapping_tests.rs"]
mod cpu_mapping;
#[path = "cpu_origins_tests.rs"]
mod cpu_origins;
#[path = "denial_tests.rs"]
mod denials;
struct Dropped<'a>(&'a Cell<usize>);
impl Drop for Dropped<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn original_recovery_window_exact_one_short_and_terminal_custody() {
    use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
    const OUTSIDE: usize = MAX_STORAGE + FLOOR;
    let inputs = METADATA + CAPACITY;
    let peak = OUTSIDE + inputs + Budget::STORAGE_WINDOW_SCRATCH_V1 + HEADER + WORKING + 17;
    let work = Budget::STORAGE_WINDOW_WORK_V1 + 8;
    for (work_limit, storage_limit, mode) in [
        (work, peak, 0),
        (work - 1, peak, 1),
        (work, peak - 1, 2),
        (work, peak, 3),
        (work, peak, 4),
    ] {
        let mut owned = Owned::new(Work::new(work_limit), storage_limit);
        owned.with_budget(|b| {
            b.reserve_storage(OUTSIDE).unwrap();
            let ledger = b.work_ledger_identity_v1();
            let identity = b.storage_account_identity_v1();
            let drops = Cell::new(0);
            let result = catch_unwind(AssertUnwindSafe(|| {
                original_recovery(inputs, b, |b| {
                    let entry = begin_bounded(CAPACITY, b)?;
                    b.reserve_storage(17)?;
                    let owner = Dropped(&drops);
                    if mode == 3 {
                        return Err(Error::mismatch("actual original-window partial refusal"));
                    }
                    if mode == 4 {
                        panic!("actual original-window partial unwind");
                    }
                    finish(&entry, b, Ok((owner, HEADER + 17)))
                })
            }));
            match mode {
                0 => {
                    let (owner, charge) = result.unwrap().unwrap();
                    assert_eq!(charge, HEADER + 17);
                    assert_eq!(drops.get(), 0);
                    assert_eq!(b.storage(), OUTSIDE);
                    drop(owner);
                    assert_eq!(drops.get(), 1);
                    assert_eq!((b.work(), b.peak_storage()), (work, peak));
                }
                1 | 2 => {
                    assert!(matches!(result, Ok(Err(_))));
                    assert!(b.storage() > OUTSIDE);
                }
                3 | 4 => {
                    assert!(matches!(result, Ok(Err(_)) | Err(_)));
                    assert_eq!(b.storage(), peak);
                    assert_eq!(drops.get(), 1);
                }
                _ => unreachable!(),
            }
            assert_eq!(b.storage_account_identity_v1(), identity);
            assert!(b.work_ledger_identity_v1() == ledger);
            assert_eq!(b.storage_limit(), storage_limit);
        });
    }
    let mut owned = Owned::new(Work::new(work), peak);
    owned.with_budget(|b| {
        b.reserve_storage(OUTSIDE).unwrap();
        for unpaid in [OUTSIDE + 1, usize::MAX] {
            assert!(
                original_recovery::<()>(unpaid, b, |_| panic!("unpaid input entered")).is_err()
            );
            assert_eq!(b.storage(), OUTSIDE);
        }
    });
}

#[test]
fn conditional_native_recovery_component_entry_exact_short_capacity_and_work() {
    let peak = FLOOR + HEADER + WORKING;
    for (storage, work_limit, success) in [(peak, 8, true), (peak - 1, 8, false), (peak, 7, false)]
    {
        let mut work = Work::new(work_limit);
        let mut b = Budget::new(&mut work, storage);
        b.reserve_storage(FLOOR).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let result = begin(CAPACITY, &mut b);
        if success {
            let entry = result.unwrap();
            assert_eq!(b.storage(), peak);
            finish(&entry, &mut b, Ok(((), HEADER))).unwrap();
            assert_eq!(b.storage(), FLOOR);
        } else if storage < peak {
            assert!(matches!(
                result,
                Err(Error(Cause::Resource(Resource::Storage(_))))
            ));
            assert_eq!(b.failed_storage(), Some(peak));
            assert_eq!(b.storage(), FLOOR);
        } else {
            assert!(matches!(
                result,
                Err(Error(Cause::Resource(Resource::Work(_))))
            ));
            assert!(b.failed_work().is_some());
            assert_eq!(b.storage(), FLOOR);
        }
        assert!(b.work_ledger_identity_v1() == ledger);
        assert_eq!(b.work(), if work_limit == 7 { 0 } else { 8 });
    }
    for (storage_limit, floor, capacity) in [
        (MAX_STORAGE + 1, FLOOR, CAPACITY),
        (MAX_STORAGE, METADATA + CAPACITY - 1, CAPACITY),
        // A selected slice length is insufficient when full spare backing lives.
        (MAX_STORAGE, METADATA + CAPACITY, CAPACITY + 1),
    ] {
        let mut work = Work::new(100);
        let mut b = Budget::new(&mut work, storage_limit);
        b.reserve_storage(floor).unwrap();
        assert!(begin(capacity, &mut b).is_err());
        assert_eq!(b.storage(), floor);
        assert_eq!(b.work(), 8);
    }
}

#[test]
fn conditional_native_recovery_component_success_preserves_clean_account_and_floor() {
    let mut work = Work::new(100);
    let mut b = Budget::new(&mut work, MAX_STORAGE);
    b.reserve_storage(FLOOR).unwrap();
    b.charge_work(7).unwrap();
    let denials = (b.failed_work(), b.failed_storage());
    assert_eq!(denials, (None, None));
    let ledger = b.work_ledger_identity_v1();
    let drops = Cell::new(0);
    let entry = begin(CAPACITY, &mut b).unwrap();
    b.reserve_storage(17).unwrap();
    let (owner, retained) = finish(&entry, &mut b, Ok((Dropped(&drops), HEADER + 17))).unwrap();
    assert_eq!(retained, HEADER + 17);
    assert_eq!(drops.get(), 0);
    assert_eq!(b.storage(), FLOOR);
    assert_eq!(b.peak_storage(), FLOOR + HEADER + WORKING + 17);
    assert_eq!(b.work(), 15);
    assert!(b.work_ledger_identity_v1() == ledger);
    assert_eq!((b.failed_work(), b.failed_storage()), denials);
    drop(owner);
    assert_eq!(drops.get(), 1);
}

#[test]
fn conditional_native_recovery_component_late_failure_floor_foreign_and_excess_terminal() {
    for mutation in 0..5 {
        let mut work = Work::new(100);
        let mut b = Budget::new(&mut work, MAX_STORAGE);
        b.reserve_storage(FLOOR).unwrap();
        let entry = begin(CAPACITY, &mut b).unwrap();
        b.reserve_storage(17).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let result = match mutation {
            0 => Err(Error::mismatch("component-only late join refusal")),
            1 => {
                b.release_storage(18).unwrap();
                Err(Error::mismatch("overridden by original floor damage"))
            }
            2 => {
                b.release_storage(1).unwrap();
                Ok((Dropped(&drops), HEADER + 17))
            }
            3 => {
                let stored = b.storage();
                b = Budget::new(Box::leak(Box::new(Work::new(100))), MAX_STORAGE);
                b.reserve_storage(stored).unwrap();
                Ok((Dropped(&drops), HEADER + 17))
            }
            _ => {
                b.reserve_storage(1).unwrap();
                Ok((Dropped(&drops), HEADER + 17))
            }
        };
        let stored = b.storage();
        let error = match finish(&entry, &mut b, result) {
            Err(e) => e,
            Ok(_) => panic!("must refuse"),
        };
        assert!(error.source().is_none());
        if mutation == 0 {
            assert!(matches!(error, Error(Cause::Mismatch(_))));
        } else {
            assert!(matches!(error, Error(Cause::Final(_))));
        }
        assert_eq!(b.storage(), stored);
        assert_eq!(drops.get(), usize::from(mutation >= 2));
        assert_eq!(b.work_ledger_identity_v1() == ledger, mutation != 3);
    }
}

#[test]
fn conditional_native_recovery_component_unwind_drops_before_any_refund() {
    let mut work = Work::new(100);
    let mut b = Budget::new(&mut work, MAX_STORAGE);
    b.reserve_storage(FLOOR).unwrap();
    let ledger = b.work_ledger_identity_v1();
    let drops = Cell::new(0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _entry = begin(CAPACITY, &mut b).unwrap();
        b.reserve_storage(17).unwrap();
        let _owner = Dropped(&drops);
        panic!("component-only replay unwind");
    }));
    assert!(result.is_err());
    assert_eq!(drops.get(), 1);
    assert_eq!(b.storage(), FLOOR + HEADER + WORKING + 17);
    assert!(b.work_ledger_identity_v1() == ledger);
    assert_eq!(b.work(), 8);
}

#[test]
fn conditional_native_recovery_component_actual_history_f_allocation_moves_without_replay() {
    use fe2o3_kernel_ir::Module;
    use fe2o3_kernel_opt::{
        encode_refined_forwarding_history_v1,
        test_support::with_refined_forwarding_history_module_v1,
    };
    // Real twelve-owner/history producer fixture, not a conditional source or
    // strict-import success. No Request or proof receipt is manufactured here.
    with_refined_forwarding_history_module_v1(
        &Module::new("inert-final-move"),
        0,
        |inputs, floor| {
            let mut work = Work::new(1_000_000_000);
            let mut b = Budget::new(&mut work, MAX_STORAGE);
            b.reserve_storage(floor + 19).unwrap();
            let wire = encode_refined_forwarding_history_v1(inputs, &mut b).unwrap();
            let wire_storage = wire.storage().retained_storage();
            b.reserve_storage(wire_storage).unwrap();
            let frame = read_refined_forwarding_history_v1(wire.canonical_bytes(), &mut b).unwrap();
            let frame_storage = frame.storage().retained_storage();
            b.reserve_storage(frame_storage).unwrap();
            let history = materialize_refined_forwarding_history_v1(&frame, &mut b).unwrap();
            let history_storage = history.storage().retained_storage();
            b.reserve_storage(history_storage).unwrap();
            let checked = history.check_semantics(&mut b).unwrap();
            let checked_storage = checked.storage().retained_storage();
            b.reserve_storage(checked_storage).unwrap();
            let pointer = checked.output().canonical().canonical_bytes().as_ptr();
            drop(checked);
            b.release_storage(checked_storage).unwrap();
            assert!(b.reserve_storage(usize::MAX).is_err());
            let ledger = b.work_ledger_identity_v1();
            let before = (b.work(), b.storage(), b.peak_storage(), b.failed_storage());
            let (output, storage) = history.into_final_graph();
            assert_eq!(output.canonical().canonical_bytes().as_ptr(), pointer);
            assert_eq!(
                output.canonical().canonical_bytes(),
                inputs.output.canonical().canonical_bytes()
            );
            assert_eq!(
                (b.work(), b.storage(), b.peak_storage(), b.failed_storage()),
                before
            );
            b.release_storage(history_storage - storage.retained_storage())
                .unwrap();
            drop(frame);
            b.release_storage(frame_storage).unwrap();
            drop(wire);
            b.release_storage(wire_storage).unwrap();
            assert_eq!(b.storage(), floor + 19 + storage.retained_storage());
            drop(output);
            b.release_storage(storage.retained_storage()).unwrap();
            assert_eq!(b.storage(), floor + 19);
            assert_eq!(b.failed_storage(), Some(usize::MAX));
            assert!(b.work_ledger_identity_v1() == ledger);
        },
    );
}

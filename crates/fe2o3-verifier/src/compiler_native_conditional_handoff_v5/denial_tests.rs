//! Terminal resource history at actual recovery components; no native proof fixtures.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
use std::mem::size_of_val;

const WORK: usize = 100_000_000;
fn deny(b: &mut Budget<'_>, mode: u8) {
    if mode == 3 {
        assert!(b.reserve_storage(MAX_STORAGE).is_err());
    }
    if mode != 1 {
        assert!(b.charge_work(WORK + 1).is_err());
    }
    if mode == 1 || mode == 2 {
        assert!(b.reserve_storage(MAX_STORAGE).is_err());
    }
}
fn state(b: &Budget<'_>) -> (usize, usize, usize, Option<usize>, Option<usize>) {
    (
        b.work(),
        b.storage(),
        b.peak_storage(),
        b.failed_work(),
        b.failed_storage(),
    )
}
fn refused<T>(result: Result<T, Error>, mode: u8) {
    match result {
        Err(Error(Cause::Resource(Resource::Storage(_)))) if mode == 1 => {}
        Err(Error(Cause::Resource(Resource::Work(_)))) if mode != 1 => {}
        _ => panic!("original denial must be terminal with work-before-storage precedence"),
    }
}

#[test]
fn selected_entries_refuse_prior_denials_before_charges_or_content() {
    let expected = NativeConditionalCpuMappingExpectationV1 {
        rustc_invocation_sha256: [3; 32],
        native_policy_sha256: [5; 32],
        policy_generation: 7,
        enrollment_binding_count: 1,
    };
    let origins = [NativeConditionalCpuExpectationV1 {
        semantic_root: 9,
        origin: crate::NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
    }];
    for route in 0..7 {
        for mode in 0..4 {
            let mut owned = Owned::new(Work::new(WORK), MAX_STORAGE);
            owned.with_budget(|b| {
                b.reserve_storage(FLOOR + size_of_val(&expected) + size_of_val(&origins))
                    .unwrap();
                deny(b, mode);
                let before = state(b);
                let ledger = b.work_ledger_identity_v1();
                let account = b.storage_account_identity_v1();
                let result = match route {
                    0 => begin(CAPACITY, b),
                    1 => begin_bounded(CAPACITY, b),
                    2 => begin_with_origins(CAPACITY, &origins, b),
                    3 => begin_bounded_with_origins(CAPACITY, &origins, b),
                    4 => begin_selected(CAPACITY, CpuSelection::Legacy(None), b),
                    5 => begin_selected(CAPACITY, CpuSelection::Legacy(Some(&origins)), b),
                    _ => begin_selected(CAPACITY, CpuSelection::Mapping(&expected), b),
                };
                refused(result, mode);
                assert_eq!(state(b), before);
                assert!(b.work_ledger_identity_v1() == ledger);
                assert_eq!(b.storage_account_identity_v1(), account);
            });
        }
    }
}

#[test]
fn original_window_refuses_prior_denials_without_entering_operation() {
    for mode in 0..4 {
        let mut owned = Owned::new(Work::new(WORK), MAX_STORAGE);
        owned.with_budget(|b| {
            b.reserve_storage(FLOOR).unwrap();
            deny(b, mode);
            let before = state(b);
            let calls = Cell::new(0);
            refused(
                original_recovery(METADATA + CAPACITY, b, |_| {
                    calls.set(calls.get() + 1);
                    Ok(())
                }),
                mode,
            );
            assert_eq!(calls.get(), 0);
            assert_eq!(state(b), before);
        });
    }
}

#[test]
fn original_window_cannot_refund_or_return_owner_after_swallowed_denial() {
    for mode in 0..4 {
        let mut owned = Owned::new(Work::new(WORK), MAX_STORAGE);
        owned.with_budget(|b| {
            b.reserve_storage(FLOOR).unwrap();
            let inputs = METADATA + CAPACITY;
            let peak = FLOOR + inputs + Budget::STORAGE_WINDOW_SCRATCH_V1;
            let drops = Cell::new(0);
            refused(
                original_recovery(inputs, b, |b| {
                    deny(b, mode);
                    Ok(Dropped(&drops))
                }),
                mode,
            );
            assert_eq!(drops.get(), 1);
            assert_eq!((b.storage(), b.peak_storage()), (peak, peak));
        });
    }
}

#[test]
fn final_transfer_authenticates_account_before_denial_and_never_refunds() {
    for foreign in [false, true] {
        for mode in 0..4 {
            let mut work = Work::new(WORK);
            let mut b = Budget::new(&mut work, MAX_STORAGE);
            b.reserve_storage(FLOOR).unwrap();
            let entry = begin(CAPACITY, &mut b).unwrap();
            b.reserve_storage(17).unwrap();
            let mut other_work = Work::new(WORK);
            let mut other = Budget::new(&mut other_work, MAX_STORAGE);
            other.reserve_storage(b.storage()).unwrap();
            let selected = if foreign { &mut other } else { &mut b };
            deny(selected, mode);
            let before = state(selected);
            let drops = Cell::new(0);
            let result = finish(&entry, selected, Ok((Dropped(&drops), HEADER + 17)));
            if foreign {
                assert!(matches!(result, Err(Error(Cause::Final(_)))));
            } else {
                refused(result, mode);
            }
            assert_eq!(drops.get(), 1);
            assert_eq!(state(selected), before);
        }
    }
}

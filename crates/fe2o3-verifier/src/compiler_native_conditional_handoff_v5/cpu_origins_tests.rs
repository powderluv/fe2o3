//! Account components and API shapes only, never native/proof success fixtures.
use super::*;

fn expected() -> [NativeConditionalCpuExpectationV1; 1] {
    [NativeConditionalCpuExpectationV1 {
        semantic_root: 9,
        origin: crate::NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
    }]
}

#[test]
fn selected_native_entry_prepays_combined_backing_and_additional_working_header() {
    let rows = expected();
    let extra = origin_working(Some(&rows));
    assert_eq!(extra, size_of::<CpuOrigins<'_>>() + size_of::<usize>());
    assert_eq!(origin_working(None), 0);
    assert_eq!(origin_backing(None), 0);
    assert_eq!(
        origin_backing(Some(&rows)),
        size_of::<NativeConditionalCpuExpectationV1>()
    );
    // Explicit empty selection is not coalesced to the default codec route.
    assert_eq!(origin_working(Some(&[])), extra);
    let paid = CAPACITY + METADATA + std::mem::size_of_val(&rows);
    let peak = paid + HEADER + WORKING + extra;
    for (floor, limit, work_limit, mode) in [
        (paid, peak, 10, 0),
        (paid - 1, peak, 10, 1),
        (paid, peak - 1, 10, 2),
        (paid, peak, 9, 3),
        (paid, MAX_STORAGE + 1, 10, 4),
    ] {
        let mut work = Work::new(work_limit);
        let mut b = Budget::new(&mut work, limit);
        b.reserve_storage(floor).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let result = begin_with_origins(CAPACITY, &rows, &mut b);
        match mode {
            0 => {
                let entry = result.unwrap();
                assert_eq!(b.storage(), peak);
                finish_with_origin_working(&entry, extra, &mut b, Ok(((), HEADER))).unwrap();
                assert_eq!(b.storage(), paid);
            }
            1 => assert!(matches!(
                result,
                Err(Error(Cause::Resource(Resource::Accounting)))
            )),
            2 => assert!(matches!(
                result,
                Err(Error(Cause::Resource(Resource::Storage(_))))
            )),
            3 => assert!(matches!(
                result,
                Err(Error(Cause::Resource(Resource::Work(_))))
            )),
            _ => assert!(matches!(result, Err(Error(Cause::Mismatch(_))))),
        }
        assert_eq!(b.work(), if mode == 3 { 0 } else { 10 });
        assert_eq!(b.storage(), floor);
        assert!(b.work_ledger_identity_v1() == ledger);
    }
}

#[test]
fn selected_native_success_preserves_clean_account_and_original_floor() {
    let rows = expected();
    let extra = origin_working(Some(&rows));
    let floor = FLOOR + std::mem::size_of_val(&rows);
    let mut work = Work::new(100);
    let mut b = Budget::new(&mut work, MAX_STORAGE);
    b.reserve_storage(floor).unwrap();
    b.charge_work(7).unwrap();
    let denied = (b.failed_work(), b.failed_storage());
    assert_eq!(denied, (None, None));
    let ledger = b.work_ledger_identity_v1();
    let entry = begin_with_origins(CAPACITY, &rows, &mut b).unwrap();
    b.reserve_storage(17).unwrap();
    let drops = Cell::new(0);
    let (owner, retained) =
        finish_with_origin_working(&entry, extra, &mut b, Ok((Dropped(&drops), HEADER + 17)))
            .unwrap();
    assert_eq!(retained, HEADER + 17);
    assert_eq!(b.storage(), floor);
    assert_eq!(b.work(), 17);
    assert_eq!((b.failed_work(), b.failed_storage()), denied);
    assert!(b.work_ledger_identity_v1() == ledger);
    assert_eq!(drops.get(), 0);
    drop(owner);
    assert_eq!(drops.get(), 1);
}

#[test]
fn selected_native_terminal_error_damaged_floor_and_foreign_ledger_never_refund() {
    let rows = expected();
    let extra = origin_working(Some(&rows));
    for mode in 0..4 {
        let mut work = Work::new(100);
        let mut b = Budget::new(&mut work, MAX_STORAGE);
        b.reserve_storage(FLOOR + std::mem::size_of_val(&rows))
            .unwrap();
        let entry = begin_with_origins(CAPACITY, &rows, &mut b).unwrap();
        b.reserve_storage(17).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let result = match mode {
            0 => Err(Error::mismatch("inert late join refusal")),
            1 => {
                b.release_storage(18).unwrap();
                Err(Error::mismatch("overridden by damaged floor"))
            }
            2 => {
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
        let error = match finish_with_origin_working(&entry, extra, &mut b, result) {
            Err(error) => error,
            Ok(_) => panic!("must refuse"),
        };
        assert!(error.source().is_none());
        if mode == 0 {
            assert!(matches!(error, Error(Cause::Mismatch(_))));
        } else {
            assert!(matches!(error, Error(Cause::Final(_))));
        }
        assert_eq!(b.storage(), stored);
        assert_eq!(drops.get(), usize::from(mode >= 2));
        assert_eq!(b.work_ledger_identity_v1() != ledger, mode == 2);
    }
}

#[test]
fn selected_original_window_exact_one_short_error_and_unwind_accounting() {
    use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
    let rows = expected();
    let extra = origin_working(Some(&rows));
    let inputs = CAPACITY + METADATA + std::mem::size_of_val(&rows);
    let outside = MAX_STORAGE + inputs + 19;
    let peak = outside + inputs + Budget::STORAGE_WINDOW_SCRATCH_V1 + HEADER + WORKING + extra + 17;
    let work = Budget::STORAGE_WINDOW_WORK_V1 + 10;
    for (work_limit, storage_limit, mode) in [
        (work, peak, 0),
        (work - 1, peak, 1),
        (work, peak - 1, 2),
        (work, peak, 3),
        (work, peak, 4),
    ] {
        let mut owned = Owned::new(Work::new(work_limit), storage_limit);
        owned.with_budget(|b| {
            b.reserve_storage(outside).unwrap();
            let ledger = b.work_ledger_identity_v1();
            let account = b.storage_account_identity_v1();
            let drops = Cell::new(0);
            let result = catch_unwind(AssertUnwindSafe(|| {
                original_recovery(inputs, b, |b| {
                    let entry = begin_bounded_with_origins(CAPACITY, &rows, b)?;
                    b.reserve_storage(17)?;
                    let owner = Dropped(&drops);
                    if mode == 3 {
                        return Err(Error::mismatch("inert selected refusal"));
                    }
                    if mode == 4 {
                        panic!("inert selected unwind");
                    }
                    finish_with_origin_working(&entry, extra, b, Ok((owner, HEADER + 17)))
                })
            }));
            match mode {
                0 => {
                    let (owner, charge) = result.unwrap().unwrap();
                    assert_eq!(charge, HEADER + 17);
                    assert_eq!(b.storage(), outside);
                    assert_eq!((b.work(), b.peak_storage()), (work, peak));
                    assert_eq!(drops.get(), 0);
                    drop(owner);
                    assert_eq!(drops.get(), 1);
                }
                1 | 2 => {
                    assert!(matches!(result, Ok(Err(_))));
                    assert!(b.storage() > outside);
                }
                _ => {
                    assert!(matches!(result, Ok(Err(_)) | Err(_)));
                    assert_eq!(b.storage(), peak);
                    assert_eq!(drops.get(), 1);
                }
            }
            assert!(b.work_ledger_identity_v1() == ledger);
            assert_eq!(b.storage_account_identity_v1(), account);
            assert_eq!(b.storage_limit(), storage_limit);
        });
    }
}

#[allow(dead_code)]
fn selected_public_call_shapes(
    policy: &[u8],
    handoff: Handoff,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected: &[NativeConditionalCpuExpectationV1],
    limits: Limits,
    profile: Profile,
    budget: &mut Budget<'_>,
    mode: u8,
) {
    match mode {
        0 => {
            let _ = crate::recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_v5(
                handoff, accepted, expected, limits, profile, budget,
            );
        }
        1 => {
            let _ = crate::recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_in_original_account_v5(
            handoff, accepted, expected, limits, profile, budget);
        }
        _ => {
            let _ = crate::recover_native_conditional_handoff_under_policy_file_with_cpu_origins_v1(
                policy, handoff, expected, budget,
            );
        }
    }
}

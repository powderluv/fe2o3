//! Inert callback/drop fixtures for the actual nested helpers, not proof owners.
use super::*;

const WORK: usize = 1_000;
const STORAGE: usize = 1_024;
type State = (usize, usize, usize, Option<usize>, Option<usize>);

fn state(b: &Budget<'_>) -> State {
    (
        b.work(),
        b.storage(),
        b.peak_storage(),
        b.failed_work(),
        b.failed_storage(),
    )
}
fn deny(b: &mut Budget<'_>, mode: u8) {
    if mode == 3 {
        assert!(b.reserve_storage(STORAGE + 1).is_err());
    }
    if mode != 1 {
        assert!(b.charge_work(WORK + 1).is_err());
    }
    if mode == 1 || mode == 2 {
        assert!(b.reserve_storage(STORAGE + 1).is_err());
    }
}
fn refused<T>(result: Result<T, Outer>, mode: u8) {
    match result {
        Err(Outer::Resource(Resource::Storage(_))) if mode == 1 => {}
        Err(Outer::Resource(Resource::Work(_))) if mode != 1 => {}
        _ => panic!("original denial must be terminal; work history takes precedence"),
    }
}
fn scope<T>(
    transfer: bool,
    b: &mut Budget<'_>,
    run: impl FnOnce(&mut Budget<'_>) -> Result<(T, NativeConditionalSourceStorageV2), Outer>,
) -> Result<(T, NativeConditionalSourceStorageV2), Outer> {
    if transfer {
        account::transfer_using(b, run)
    } else {
        account::temporary_using(b, 13, run)
    }
}

#[test]
fn nested_entries_refuse_prior_denials_without_callback_or_new_charge() {
    for transfer in [false, true] {
        for mode in 0..4 {
            let mut work = Work::new(WORK);
            let mut b = Budget::new(&mut work, STORAGE);
            b.reserve_storage(FLOOR).unwrap();
            deny(&mut b, mode);
            let before = state(&b);
            let ledger = b.work_ledger_identity_v1();
            let address = &b as *const Budget<'_> as usize;
            let calls = Cell::new(0);
            refused(
                scope(transfer, &mut b, |_| {
                    calls.set(calls.get() + 1);
                    Ok(((), NativeConditionalSourceStorageV2(0)))
                }),
                mode,
            );
            assert_eq!(calls.get(), 0);
            assert_eq!(state(&b), before);
            assert!(b.work_ledger_identity_v1() == ledger);
            assert_eq!(&b as *const Budget<'_> as usize, address);
        }
    }
}

#[test]
fn successful_callback_cannot_hide_denial_or_release_owner_reservation() {
    for transfer in [false, true] {
        for mode in 0..4 {
            let mut work = Work::new(WORK);
            let mut b = Budget::new(&mut work, STORAGE);
            b.reserve_storage(FLOOR).unwrap();
            let ledger = b.work_ledger_identity_v1();
            let observed = Cell::new(None);
            let drops = Cell::new(0);
            refused(
                scope(transfer, &mut b, |b| {
                    b.reserve_storage(17)?;
                    deny(b, mode);
                    observed.set(Some(state(b)));
                    Ok((Provisional(&drops), NativeConditionalSourceStorageV2(17)))
                }),
                mode,
            );
            assert_eq!(drops.get(), 1);
            assert_eq!(Some(state(&b)), observed.get());
            assert_eq!(b.storage(), FLOOR + 17 + if transfer { 0 } else { 13 });
            assert!(b.work_ledger_identity_v1() == ledger);
        }
    }
}

#[test]
fn swallowed_denial_keeps_all_five_nested_account_reservations() {
    for mode in 0..4 {
        let mut work = Work::new(WORK);
        let mut b = Budget::new(&mut work, STORAGE);
        b.reserve_storage(FLOOR).unwrap();
        let observed = Cell::new(None);
        let drops = Cell::new(0);
        let result = account::transfer_using(&mut b, |b| {
            account::temporary_using(b, 13, |b| {
                account::temporary_using(b, 23, |b| {
                    account::temporary_using(b, 11, |b| {
                        account::temporary_using(b, 7, |b| {
                            b.reserve_storage(17)?;
                            deny(b, mode);
                            observed.set(Some(state(b)));
                            Ok((Provisional(&drops), NativeConditionalSourceStorageV2(17)))
                        })
                    })
                })
            })
        });
        refused(result, mode);
        assert_eq!(drops.get(), 1);
        assert_eq!(Some(state(&b)), observed.get());
        assert_eq!(b.storage(), FLOOR + 13 + 23 + 11 + 7 + 17);
        assert_eq!(b.work(), 8 + 4);
    }
}

#[test]
fn original_slot_ledger_floor_and_owner_backing_precede_post_callback_denial() {
    for transfer in [false, true] {
        for damage in 0..3 {
            if damage == 2 && !transfer {
                continue;
            }
            for mode in 0..4 {
                let mut work = Work::new(WORK);
                let mut b = Budget::new(&mut work, STORAGE);
                b.reserve_storage(FLOOR).unwrap();
                let observed = Cell::new(None);
                let drops = Cell::new(0);
                let result = scope(transfer, &mut b, |b| {
                    b.reserve_storage(17)?;
                    match damage {
                        0 => b.release_storage(18)?,
                        1 => {
                            let paid = b.storage();
                            // Deliberately replace this inert meter at its same address.
                            *b = Budget::new(Box::leak(Box::new(Work::new(WORK))), STORAGE);
                            b.reserve_storage(paid)?;
                        }
                        _ => {}
                    }
                    deny(b, mode);
                    observed.set(Some(state(b)));
                    Ok((
                        Provisional(&drops),
                        NativeConditionalSourceStorageV2(if damage == 2 { 18 } else { 17 }),
                    ))
                });
                assert!(matches!(
                    result,
                    Err(Outer::Final(Error(Cause::Resource(Resource::Accounting))))
                ));
                assert_eq!(drops.get(), 1);
                assert_eq!(Some(state(&b)), observed.get());
            }
        }
    }
}

#[test]
fn callback_error_or_unwind_retains_charges_and_original_failure() {
    for transfer in [false, true] {
        for unwind in [false, true] {
            for mode in 0..4 {
                let mut work = Work::new(WORK);
                let mut b = Budget::new(&mut work, STORAGE);
                b.reserve_storage(FLOOR).unwrap();
                let observed = Cell::new(None);
                let drops = Cell::new(0);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    scope::<()>(transfer, &mut b, |b| {
                        b.reserve_storage(17)?;
                        let _owner = Provisional(&drops);
                        deny(b, mode);
                        observed.set(Some(state(b)));
                        if unwind {
                            panic!("nested denial followed by unwind");
                        }
                        Err(Outer::Join(71, [9; 2048]))
                    })
                }));
                if unwind {
                    assert!(result.is_err());
                } else {
                    assert!(
                        matches!(result.unwrap(), Err(Outer::Join(71, bytes)) if bytes == [9; 2048])
                    );
                }
                assert_eq!(drops.get(), 1);
                assert_eq!(Some(state(&b)), observed.get());
                assert_eq!(b.storage(), FLOOR + 17 + if transfer { 0 } else { 13 });
            }
        }
    }
}

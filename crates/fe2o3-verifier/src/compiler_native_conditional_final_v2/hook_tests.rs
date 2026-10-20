//! Generic hook accounting only; no reconstructed source or proof is fabricated.
use super::*;

#[path = "denial_tests.rs"]
mod denials;

#[derive(Debug)]
enum Outer {
    Final(Error),
    Resource(Resource),
    Join(u16, [u8; 2048]),
}
impl From<Error> for Outer {
    fn from(e: Error) -> Self {
        Self::Final(e)
    }
}
impl From<Resource> for Outer {
    fn from(e: Resource) -> Self {
        Self::Resource(e)
    }
}

#[test]
fn conditional_final_hook_keeps_old_header_and_pays_concrete_callback_and_error() {
    let base = size_of::<Inputs<'static, 'static, 'static>>() + 5 * size_of::<account::Account>();
    assert_eq!(HEADER, base + size_of::<Result<Output, Error>>());
    assert_eq!(header::<Error>(0).unwrap(), HEADER);
    let captured = [3u8; 37];
    let join = move || captured;
    let bytes = std::mem::size_of_val(&join);
    assert_eq!(bytes, 37);
    let required = base + size_of::<Result<Output, Outer>>() + 37;
    assert_eq!(header::<Outer>(bytes).unwrap(), required);
    assert!(required > HEADER);
    for short in [false, true] {
        let mut work = Work::new(9);
        let mut budget = Budget::new(&mut work, FLOOR + required - usize::from(short));
        budget.reserve_storage(FLOOR).unwrap();
        let entered = Cell::new(false);
        let result = account::transfer_using::<(), Outer>(&mut budget, |b| {
            account::temporary_using(b, header::<Outer>(bytes)?, |_| {
                entered.set(true);
                Ok(((), NativeConditionalSourceStorageV2(0)))
            })
        });
        if short {
            assert!(matches!(result, Err(Outer::Resource(Resource::Storage(_)))));
            assert!(!entered.get());
            assert_eq!(budget.failed_storage(), Some(FLOOR + required));
        } else {
            result.unwrap();
            assert!(entered.get());
            assert_eq!(budget.peak_storage(), FLOOR + required);
        }
        assert_eq!(budget.work(), 9);
        assert_eq!(budget.storage(), FLOOR);
    }
}

#[test]
fn conditional_final_hook_typed_error_survives_and_late_account_failure_overrides() {
    for damage in [false, true] {
        budgeted(|b| {
            let ledger = b.work_ledger_identity_v1();
            let result = account::transfer_using::<(), Outer>(b, |b| {
                account::temporary_using(b, 13, |b| {
                    if damage {
                        b.release_storage(1)?;
                    }
                    Err(Outer::Join(71, [9; 2048]))
                })
            });
            if damage {
                assert!(matches!(
                    result,
                    Err(Outer::Final(Error(Cause::Resource(Resource::Accounting))))
                ));
            } else {
                assert!(matches!(result, Err(Outer::Join(71, bytes)) if bytes == [9; 2048]));
            }
            assert_eq!(b.storage(), FLOOR + 13 - usize::from(damage));
            assert_eq!(b.work(), 9);
            assert!(b.work_ledger_identity_v1() == ledger);
        });
    }
}

#[test]
fn conditional_final_hook_foreign_ledger_and_unwind_do_not_refund() {
    for unwind in [false, true] {
        budgeted(|b| {
            let ledger = b.work_ledger_identity_v1();
            let drops = Cell::new(0);
            let result = catch_unwind(AssertUnwindSafe(|| {
                account::transfer_using::<(), Outer>(b, |b| {
                    account::temporary_using(b, 13, |b| {
                        let _guard = Provisional(&drops);
                        if unwind {
                            panic!("component hook unwind");
                        }
                        *b = Budget::new(Box::leak(Box::new(Work::new(100))), 100);
                        b.reserve_storage(FLOOR + 13)?;
                        Err(Outer::Join(71, [9; 2048]))
                    })
                })
            }));
            assert_eq!(drops.get(), 1);
            assert_eq!(b.storage(), FLOOR + 13);
            if unwind {
                assert!(result.is_err());
                assert!(b.work_ledger_identity_v1() == ledger);
            } else {
                assert!(matches!(
                    result.unwrap(),
                    Err(Outer::Final(Error(Cause::Resource(Resource::Accounting))))
                ));
                assert!(b.work_ledger_identity_v1() != ledger);
            }
        });
    }
}

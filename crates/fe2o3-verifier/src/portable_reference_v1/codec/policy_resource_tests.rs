use super::*;

fn run<R>(
    decoding: bool,
    f: &Fixture,
    bytes: &[u8],
    budget: &mut Budget<'_>,
    consume: impl FnOnce(&mut Budget<'_>) -> R,
) -> Result<R, Error> {
    if decoding {
        with_decoded_native_cpu_policy_input_v2(bytes, budget, |_, b| consume(b))
    } else {
        with_encoded_native_cpu_policy_input_v2(input(f), budget, |_, _, b| consume(b))
    }
}

#[test]
fn exact_and_one_short_original_work_and_storage() {
    let f = fixture();
    let (bytes, _) = encoded(input(&f)).unwrap();
    for decoding in [false, true] {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.charge_work(17).unwrap();
        budget.reserve_storage(31).unwrap();
        run(decoding, &f, &bytes, &mut budget, |_| ()).unwrap();
        let (cost, peak) = (budget.work(), budget.peak_storage());
        assert_eq!(budget.storage(), 31);
        for (work_limit, storage_limit, denial) in
            [(cost, peak, 0), (cost - 1, peak, 1), (cost, peak - 1, 2)]
        {
            let mut work = Work::new(work_limit);
            let mut budget = Budget::new(&mut work, storage_limit);
            budget.charge_work(17).unwrap();
            budget.reserve_storage(31).unwrap();
            let mut called = false;
            let result = run(decoding, &f, &bytes, &mut budget, |_| {
                called = true;
            });
            match denial {
                0 => {
                    result.unwrap();
                    assert!(called);
                    assert_eq!((budget.work(), budget.peak_storage()), (cost, peak));
                }
                1 => {
                    assert!(matches!(result, Err(Error::Resource(Resource::Work(_)))));
                    assert!(!called);
                    assert!(budget.failed_work().is_some());
                }
                _ => {
                    assert!(matches!(result, Err(Error::Resource(Resource::Storage(_)))));
                    assert!(!called);
                    assert!(budget.failed_storage().is_some());
                }
            }
            assert_eq!(budget.storage(), 31);
        }
    }
}

#[test]
fn callback_success_error_unwind_preserve_additions_and_first_denials() {
    let f = fixture();
    let (bytes, _) = encoded(input(&f)).unwrap();
    for decoding in [false, true] {
        for exit in 0..3 {
            let mut work = Work::new(usize::MAX);
            let mut budget = Budget::new(&mut work, usize::MAX);
            budget.charge_work(17).unwrap();
            budget.reserve_storage(31).unwrap();
            assert!(budget.charge_work(usize::MAX).is_err());
            assert!(budget.reserve_storage(usize::MAX).is_err());
            let denials = (budget.failed_work(), budget.failed_storage());
            let result = catch_unwind(AssertUnwindSafe(|| {
                run(decoding, &f, &bytes, &mut budget, |b| {
                    b.reserve_storage(7).unwrap();
                    b.charge_work(13).unwrap();
                    match exit {
                        0 => Ok(()),
                        1 => Err("consumer rejection"),
                        _ => panic!("policy codec callback"),
                    }
                })
            }));
            match exit {
                0 => result.unwrap().unwrap().unwrap(),
                1 => assert_eq!(result.unwrap().unwrap(), Err("consumer rejection")),
                _ => assert!(result.is_err()),
            }
            assert_eq!(budget.storage(), 38);
            assert!(budget.work() > 30);
            assert_eq!((budget.failed_work(), budget.failed_storage()), denials);
        }
    }
}

#[test]
fn foreign_account_and_damaged_floor_are_not_refunded() {
    let f = fixture();
    let (bytes, _) = encoded(input(&f)).unwrap();
    for decoding in [false, true] {
        for substitute in [false, true] {
            for unwind in [false, true] {
                let mut work = Work::new(usize::MAX);
                let mut budget = Budget::new(&mut work, usize::MAX);
                budget.reserve_storage(31).unwrap();
                let account = budget.work_ledger_identity_v1();
                let mut original_work = 0;
                let mut state = None;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    run(decoding, &f, &bytes, &mut budget, |b| {
                        original_work = b.work();
                        if substitute {
                            let protected = b.storage();
                            *b =
                                Budget::new(Box::leak(Box::new(Work::new(usize::MAX))), usize::MAX);
                            b.reserve_storage(protected).unwrap();
                            b.charge_work(13).unwrap();
                            assert!(b.work_ledger_identity_v1() != account);
                        } else {
                            b.release_storage(1).unwrap();
                        }
                        state = Some((b.work(), b.storage()));
                        if unwind {
                            panic!("damaged account");
                        }
                    })
                }));
                if unwind {
                    assert!(result.is_err());
                } else {
                    assert_eq!(result.unwrap(), Err(Error::Resource(Resource::Accounting)));
                }
                assert_eq!(Some((budget.work(), budget.storage())), state);
                drop(budget);
                assert_eq!(work.work(), original_work);
            }
        }
    }
}

#[test]
fn invalid_input_preserves_caller_floor_and_prior_denials() {
    let mut f = fixture();
    let (mut bytes, _) = encoded(input(&f)).unwrap();
    f.digest[0] ^= 1;
    *bytes.last_mut().unwrap() ^= 1;
    for decoding in [false, true] {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.charge_work(17).unwrap();
        budget.reserve_storage(31).unwrap();
        assert!(budget.charge_work(usize::MAX).is_err());
        assert!(budget.reserve_storage(usize::MAX).is_err());
        let denials = (budget.failed_work(), budget.failed_storage());
        assert!(
            run(decoding, &f, &bytes, &mut budget, |_| panic!(
                "invalid input exposed"
            ))
            .is_err()
        );
        assert_eq!(budget.storage(), 31);
        assert_eq!((budget.failed_work(), budget.failed_storage()), denials);
    }
}

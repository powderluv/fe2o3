//! Inert result/accounting tests, not fabricated native Attempt admission.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use std::panic::{AssertUnwindSafe, catch_unwind};

const FLOOR: usize = 31;
const COST: usize = 7;
const PEAK: usize = FLOOR + FRAME + OriginalCompilerEnrollment::STORAGE;

fn value() -> OriginalCompilerEnrollment {
    OriginalCompilerEnrollment {
        rustc_invocation_sha256: [1; 32],
        intake_invocation_identity: [5; 32],
        invocation_bytes: 99,
        native_policy_sha256: [2; 32],
        policy_generation: 3,
        binding_count: Some(4),
    }
}

#[test]
fn scoped_result_keeps_original_account_and_storage_through_callback() {
    let mut work = Work::new(LOCAL_WORK + COST);
    let mut b = Budget::new(&mut work, PEAK);
    b.reserve_storage(FLOOR).unwrap();
    let account = RequestAccount::capture(&b);
    let result = value()
        .with_reserved(&account, FLOOR, &mut b, |v, b| {
            assert_eq!(v, &value());
            assert_eq!(b.storage(), PEAK);
            assert!(b.work_ledger_identity_v1() == account.ledger);
            assert_eq!(b as *const Budget<'_> as usize, account.address);
            b.charge_work(COST)?;
            Ok(v.binding_count)
        })
        .unwrap();
    assert_eq!(result, Some(4));
    assert_eq!(
        (b.work(), b.storage(), b.peak_storage()),
        (LOCAL_WORK + COST, FLOOR, PEAK)
    );
}

#[test]
fn insufficient_input_output_or_work_never_enters_callback() {
    for case in 0..3 {
        let floor = FLOOR - usize::from(case == 0);
        let mut work = Work::new(LOCAL_WORK - usize::from(case == 2));
        let mut b = Budget::new(&mut work, PEAK - usize::from(case == 1));
        b.reserve_storage(floor).unwrap();
        let account = RequestAccount::capture(&b);
        let result = value().with_reserved::<()>(&account, FLOOR, &mut b, |_, _| {
            panic!("unfunded output reached callback")
        });
        assert!(result.is_err());
        assert_eq!(b.storage(), floor);
        if case != 0 {
            let history = (b.work(), b.failed_work(), b.failed_storage());
            assert!(
                value()
                    .with_reserved::<()>(&account, FLOOR, &mut b, |_, _| {
                        panic!("denied account reached callback")
                    })
                    .is_err()
            );
            assert_eq!((b.work(), b.failed_work(), b.failed_storage()), history);
        }
    }
}

#[test]
fn same_address_foreign_ledger_and_moved_original_ledger_refuse_output() {
    let mut first_work = Work::new(2 * LOCAL_WORK);
    let mut second_work = Work::new(2 * LOCAL_WORK);
    let mut first = Budget::new(&mut first_work, PEAK);
    let mut second = Budget::new(&mut second_work, PEAK);
    first.reserve_storage(FLOOR).unwrap();
    second.reserve_storage(FLOOR).unwrap();
    let account = RequestAccount::capture(&first);
    std::mem::swap(&mut first, &mut second);
    for b in [&mut first, &mut second] {
        let result = value().with_reserved::<()>(&account, FLOOR, b, |_, _| {
            panic!("substituted account received output")
        });
        assert!(matches!(
            result,
            Err(Failure::Native(Error::Resource(Resource::Accounting)))
        ));
        assert_eq!(b.storage(), FLOOR);
        assert_eq!(b.work(), LOCAL_WORK);
        assert!(b.peak_storage() < PEAK);
    }
    std::mem::swap(&mut first, &mut second);
    value()
        .with_reserved(&account, FLOOR, &mut first, |_, _| Ok(()))
        .unwrap();
    assert_eq!(first.storage(), FLOOR);
}

#[test]
fn error_and_unwind_restore_storage_without_refunding_work() {
    for unwind in [false, true] {
        let mut work = Work::new(LOCAL_WORK + COST);
        let mut b = Budget::new(&mut work, PEAK);
        b.reserve_storage(FLOOR).unwrap();
        let account = RequestAccount::capture(&b);
        let result = catch_unwind(AssertUnwindSafe(|| {
            value().with_reserved::<()>(&account, FLOOR, &mut b, |_, b| {
                b.charge_work(COST)?;
                if unwind {
                    panic!("inert callback fault");
                }
                Err(Failure::Invalid("inert callback refusal"))
            })
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            assert!(matches!(
                result.unwrap(),
                Err(Failure::Invalid("inert callback refusal"))
            ));
        }
        assert_eq!(
            (b.work(), b.storage(), b.peak_storage()),
            (LOCAL_WORK + COST, FLOOR, PEAK)
        );
    }
}

#[test]
fn swallowed_callback_denials_still_refuse_and_remain_sticky() {
    for storage in [false, true] {
        let mut work = Work::new(LOCAL_WORK);
        let mut b = Budget::new(&mut work, PEAK);
        b.reserve_storage(FLOOR).unwrap();
        let account = RequestAccount::capture(&b);
        assert!(
            value()
                .with_reserved(&account, FLOOR, &mut b, |_, b| {
                    if storage {
                        assert!(b.reserve_storage(1).is_err());
                    } else {
                        assert!(b.charge_work(1).is_err());
                    }
                    Ok(())
                })
                .is_err()
        );
        let history = (b.work(), b.storage(), b.failed_work(), b.failed_storage());
        assert!(
            value()
                .with_reserved::<()>(&account, FLOOR, &mut b, |_, _| panic!("prior denial"))
                .is_err()
        );
        assert_eq!(
            (b.work(), b.storage(), b.failed_work(), b.failed_storage()),
            history
        );
        assert_eq!(b.storage(), FLOOR);
    }
}

#[test]
fn custody_quote_includes_both_original_checks_live_issuer_and_result_account() {
    let quote = NativeAttempt::<Helper>::original_enrollment_custody_quota().unwrap();
    let original = NativeAttempt::<Helper>::original_validation_quota();
    let ready = Prepared::maximum_issuer_continuity_quota::<Helper>().unwrap();
    let policy = NativeAttempt::<Helper>::original_policy_identity_quota();
    assert_eq!(
        quote.work(),
        8 + 2 * original.work() + ready.work() + policy.work() + LOCAL_WORK
    );
    assert_eq!(
        quote.scratch(),
        SCRATCH
            + 2 * original.scratch()
            + ready.scratch()
            + policy.scratch()
            + FRAME
            + OriginalCompilerEnrollment::STORAGE
    );
}

#[test]
fn callback_cannot_retire_the_borrowed_output_reservation() {
    let mut work = Work::new(LOCAL_WORK);
    let mut b = Budget::new(&mut work, PEAK);
    b.reserve_storage(FLOOR).unwrap();
    let account = RequestAccount::capture(&b);
    let result = value().with_reserved(&account, FLOOR, &mut b, |_, b| {
        b.release_storage(OriginalCompilerEnrollment::STORAGE)?;
        Ok(())
    });
    assert!(matches!(
        result,
        Err(Failure::Resource(Resource::Accounting))
    ));
    assert_eq!(
        (b.work(), b.storage(), b.peak_storage()),
        (LOCAL_WORK, FLOOR, PEAK)
    );
}

//! Actual admission/loan process-consistency tests, never proof or native-pass
//! evidence. See the fixture's boundary and the collector replay limitations.
//!
//! Full production qualification remains open: installed compiler/backend
//! provenance, registered nonempty source collection/import, foreign rustc
//! Sessions, Prepared graph retention, recipe freeze and receipt publication.
//! The publication case below deliberately stops before receipt acquisition.

use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as TargetWork;

#[path = "protected_reference_enrollment_loan_v1_flow_fixture.rs"]
mod fixture;

const STORAGE: usize = 4 * 1024 * 1024;
const ONE_REQUEST: &str =
    r#"{"version":1,"bindings":[{"kernel":"fixture::kernel","reference":"fixture::reference"}]}"#;

#[path = "protected_reference_enrollment_projected_count_tests.rs"]
mod projected_count;

#[test]
fn actual_mixed_instances_seal_original_wrapper_before_preflight() {
    let request = crate::collector::reversed_original_root_inventory_request();
    if fixture::isolated(
        concat!(
            module_path!(),
            "::actual_mixed_instances_seal_original_wrapper_before_preflight"
        ),
        Some(&request),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    crate::collector::check_original_root_inventory_flow(native, &mut invocation, true);
}

#[test]
fn actual_no_request_instances_keep_legacy_inventory_before_preflight() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::actual_no_request_instances_keep_legacy_inventory_before_preflight"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    crate::collector::check_original_root_inventory_flow(native, &mut invocation, false);
}

fn refused<T>(result: Result<T, SessionError>, reason: &str) {
    let Err(error) = result else {
        panic!("expected refusal: {reason}")
    };
    assert!(error.to_string().contains(reason), "{error}");
}

#[test]
fn admitted_no_request_survives_fresh_loan_and_publication_preparation() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::admitted_no_request_survives_fresh_loan_and_publication_preparation"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let expected = invocation.descriptor().clone();
    let mut foreign_invocation = fixture::equal_invocation(&invocation);
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    budget.reserve_storage(91).unwrap();
    budget.charge_work(19).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let policy = fixture::policy(&mut budget);
    // Keep the setup charges, the admitted policy copy, and exactly three
    // logical identity owners: session, original invocation, foreign invocation.
    // Owner charges are not allocator RSS and are not refunded by Owner::drop.
    let retained = budget.storage() + policy.retained_storage() + 3 * Owner::STORAGE;
    let client_storage =
        std::mem::size_of::<fe2o3_compiler_execution_client::CompilerExecutionClientV3<'_, '_>>();
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    source.charge(17).unwrap();
    let (native, stamp) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            assert!(loan.request(&mut source)?.is_none());
            assert!(loan.inventory_context(&mut source)?.is_none());
            loan.capture_stamp(&mut source).map_err(SessionError::from)
        })
        .unwrap();
    let (native, foreign_stamp) = native
        .with_reference_enrollment::<_, SessionError>(&mut foreign_invocation, |loan| {
            loan.unwrap()
                .capture_stamp(&mut source)
                .map_err(SessionError::from)
        })
        .unwrap();
    let mut moved = invocation;
    let published = Cell::new(false);
    let publication_account = Cell::new(None);
    let result = native.prepare_and_acquire_with_enrollment::<_, (), (), SessionError>(
        |preparation, budget| {
            assert!(preparation.is_enrolled());
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(budget.storage(), retained + client_storage);
            assert!(preparation.owner.unwrap().matches(&stamp.session));
            preparation.with_invocation(&mut moved, budget, |loan| {
                assert!(loan.requested_bindings.get().is_none());
                loan.revalidate_stamp(&stamp, &mut source)?;
                assert!(loan.request(&mut source)?.is_none());
                Ok::<_, SessionError>(())
            })?;
            moved.select_native_proof_runtime().unwrap();
            let finished = moved.finish_for_publication().unwrap();
            finished.assert_reference_owner_for_test(&stamp.invocation, &foreign_stamp.invocation);
            Ok(finished)
        },
        |finished, budget| {
            published.set(true);
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(finished.descriptor(), &expected);
            finished.revalidate_for_publication().unwrap();
            finished.assert_reference_owner_for_test(&stamp.invocation, &foreign_stamp.invocation);
            assert_eq!(budget.storage(), retained + client_storage);
            assert_eq!(
                (budget.failed_work(), budget.failed_storage()),
                (None, None)
            );
            let accepted = (budget.storage(), budget.work(), budget.peak_storage());
            // Record real nonempty denials before the deliberate publication
            // refusal. Failed charges must neither refund nor advance acceptance.
            assert!(budget.charge_work(usize::MAX).is_err());
            assert!(budget.reserve_storage(STORAGE).is_err());
            assert_eq!(
                (budget.storage(), budget.work(), budget.peak_storage()),
                accepted
            );
            assert_eq!(budget.failed_work(), Some(usize::MAX));
            assert_eq!(budget.failed_storage(), Some(accepted.0 + STORAGE));
            publication_account.set(Some((
                accepted,
                budget.work_ledger_identity_v1(),
                budget.failed_work(),
                budget.failed_storage(),
            )));
            // Stop before receipt acquisition: there is no signing service or
            // fabricated positive carriage. Full production publication is open.
            Err(SessionError::Reference(Error::new(
                "fixture publication stop",
            )))
        },
        |_, (), _| panic!("fixture must not acquire a receipt"),
    );
    refused(result, "fixture publication stop");
    assert!(published.get());
    assert!(budget.work_ledger_identity_v1() == ledger);
    let ((storage, work, peak), publication_ledger, failed_work, failed_storage) =
        publication_account
            .get()
            .expect("publication account snapshot");
    assert!(budget.work_ledger_identity_v1() == publication_ledger);
    // Only Client::RETAINED (size_of::<Client> at this pin) retires on refusal.
    // Policy/identity logical charges remain, even after their owners drop.
    assert_eq!(
        budget.storage(),
        storage.checked_sub(client_storage).unwrap()
    );
    assert_eq!(budget.storage(), retained);
    assert_eq!(budget.work(), work);
    assert_eq!(budget.peak_storage(), peak);
    assert_eq!(budget.failed_work(), failed_work);
    assert_eq!(budget.failed_storage(), failed_storage);
}

#[test]
fn original_target_ledger_mismatch_refuses_before_loan_callback() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::original_target_ledger_mismatch_refuses_before_loan_callback"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |_| Ok(()))
        .unwrap();
    let entered = Cell::new(false);
    let result = native.prepare_and_acquire_with_enrollment::<(), (), (), SessionError>(
        |preparation, budget| {
            let mut other_work = TargetWork::new(usize::MAX);
            let mut other = Budget::new(&mut other_work, STORAGE);
            other.charge_work(budget.work()).unwrap();
            other.reserve_storage(budget.storage()).unwrap();
            assert_eq!(other.work(), budget.work());
            assert_eq!(other.storage(), budget.storage());
            assert!(other.work_ledger_identity_v1() != budget.work_ledger_identity_v1());
            let before = (other.work(), other.storage());
            let result = preparation.with_invocation(&mut invocation, &mut other, |_| {
                entered.set(true);
                Ok::<_, SessionError>(())
            });
            assert_eq!((other.work(), other.storage()), before);
            result
        },
        |(), _| panic!("foreign target account reached publication"),
        |_, (), _| panic!("foreign target account reached receipt"),
    );
    refused(
        result,
        "reference enrollment original target account changed",
    );
    assert!(!entered.get());
}

#[test]
fn equal_descriptor_foreign_live_invocation_refuses_even_if_error_is_ignored() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::equal_descriptor_foreign_live_invocation_refuses_even_if_error_is_ignored"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut other = fixture::equal_invocation(&invocation);
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, stamp) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            loan.unwrap()
                .capture_stamp(&mut source)
                .map_err(SessionError::from)
        })
        .unwrap();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut other, |loan| {
        let loan = loan.unwrap();
        assert_eq!(
            loan.revalidate_stamp(&stamp, &mut source)
                .unwrap_err()
                .to_string(),
            "reference enrollment original invocation changed"
        );
        assert!(loan.capture_stamp(&mut source).is_err());
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    // The original remains live; refusal was identity substitution, not drop.
    invocation.revalidate_for_publication().unwrap();
}

#[test]
fn equal_policy_foreign_live_session_cannot_consume_the_original_stamp() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::equal_policy_foreign_live_session_cannot_consume_the_original_stamp"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, stamp) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            loan.unwrap()
                .capture_stamp(&mut source)
                .map_err(SessionError::from)
        })
        .unwrap();
    let mut other_work = TargetWork::new(usize::MAX);
    let mut other_budget = Budget::new(&mut other_work, STORAGE);
    let other_policy = fixture::policy(&mut other_budget);
    assert_eq!(policy.policy(), other_policy.policy());
    let (other, _other_server) = fixture::session(&other_policy, &mut other_budget);
    let result = other.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        let error = loan
            .unwrap()
            .revalidate_stamp(&stamp, &mut source)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "reference enrollment original native session changed"
        );
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            loan.unwrap().revalidate_stamp(&stamp, &mut source)?;
            Ok(())
        })
        .unwrap();
    drop(native);
}

#[test]
fn ignored_live_process_failure_stays_terminal_after_cwd_is_restored() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::ignored_live_process_failure_stays_terminal_after_cwd_is_restored"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        let loan = loan.unwrap();
        assert!(loan.request(&mut source)?.is_none());
        let changed = fixture::ChangedDirectory::enter();
        let error = loan.capture_stamp(&mut source).unwrap_err();
        assert!(error.to_string().contains("directory"), "{error}");
        drop(changed);
        assert_eq!(
            loan.request(&mut source).unwrap_err().to_string(),
            "reference enrollment loan was refused"
        );
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    invocation.revalidate_for_publication().unwrap();
}

#[test]
fn successful_callback_cannot_skip_postcallback_live_process_revalidation() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::successful_callback_cannot_skip_postcallback_live_process_revalidation"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut changed = None;
    let reached = Cell::new(false);
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |_| {
        reached.set(true);
        changed = Some(fixture::ChangedDirectory::enter());
        Ok(())
    });
    refused(result, "directory");
    assert!(reached.get());
    drop(changed);
    invocation.revalidate_for_publication().unwrap();
}

#[test]
fn stale_process_refuses_constructor_before_callback_entry() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::stale_process_refuses_constructor_before_callback_entry"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let entered = Cell::new(false);
    let changed = fixture::ChangedDirectory::enter();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |_| {
        entered.set(true);
        Ok(())
    });
    refused(result, "directory");
    assert!(!entered.get());
    drop(changed);
}

#[test]
fn explicit_empty_request_is_refused_and_cannot_be_ignored() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::explicit_empty_request_is_refused_and_cannot_be_ignored"
        ),
        Some(r#"{"version":1,"bindings":[]}"#),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        let loan = loan.unwrap();
        let mut source = Work::default();
        assert_eq!(
            loan.request(&mut source).unwrap_err().to_string(),
            "reference enrollment binding count outside limits"
        );
        assert!(loan.requested_bindings.get().is_none());
        assert!(loan.inventory_context(&mut source).is_err());
        assert!(loan.reserve_inventory_storage(0).is_err());
        assert!(loan.capture_stamp(&mut source).is_err());
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
}

#[test]
fn fresh_loan_must_request_before_origin_and_retains_descriptor_membership() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::fresh_loan_must_request_before_origin_and_retains_descriptor_membership"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, (stamp, origin)) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            assert_eq!(loan.request(&mut source)?.unwrap().bindings().len(), 1);
            Ok((loan.capture_stamp(&mut source)?, loan.origin(0)?))
        })
        .unwrap();
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            assert!(loan.requested_bindings.get().is_none());
            loan.revalidate_stamp(&stamp, &mut source)?;
            assert_eq!(loan.request(&mut source)?.unwrap().bindings().len(), 1);
            assert_eq!(loan.origin(0)?, origin);
            Ok(())
        })
        .unwrap();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        assert_eq!(
            loan.unwrap().origin(0).unwrap_err().to_string(),
            "reference enrollment ordinal is not in the captured request"
        );
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
}

#[test]
fn no_request_empty_collector_roster_captures_and_replays_with_a_fresh_loan() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::no_request_empty_collector_roster_captures_and_replays_with_a_fresh_loan"
        ),
        None,
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    crate::collector::check_empty_enrollment_replay(native, &mut invocation);
}

#[test]
fn original_request_header_does_not_query_an_ordinal() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::original_request_header_does_not_query_an_ordinal"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            assert_eq!(loan.request(&mut source)?.unwrap().bindings().len(), 1);
            let context = loan.inventory_context(&mut source)?.unwrap();
            assert_eq!(context.descriptor_bindings, 1);
            assert_eq!(
                context.rustc_invocation_sha256,
                loan.origin.rustc_invocation_sha256
            );
            assert_eq!(
                context.native_policy_sha256,
                loan.origin.native_policy_sha256
            );
            assert_eq!(context.policy_generation, loan.origin.policy_generation);
            loan.reserve_inventory_storage(0)?;
            assert!(!loan.refused.get());
            Ok(())
        })
        .unwrap();
    drop(native);
}

#[test]
fn actual_original_loan_keeps_multiple_inventory_debits_after_later_refusal() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::actual_original_loan_keeps_multiple_inventory_debits_after_later_refusal"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    budget.reserve_storage(31).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            loan.unwrap().request(&mut source)?;
            Ok(())
        })
        .unwrap();
    let retained = Cell::new(None);
    let result = native.prepare_and_acquire_with_enrollment::<(), (), (), SessionError>(
        |preparation, budget| {
            let before = budget.storage();
            let result = preparation.with_invocation(&mut invocation, budget, |loan| {
                loan.request(&mut source)?;
                assert_eq!(
                    loan.inventory_context(&mut source)?
                        .unwrap()
                        .descriptor_bindings,
                    1
                );
                loan.reserve_inventory_storage(11)?;
                loan.reserve_inventory_storage(17)?;
                loan.reserve_inventory_storage(0)?;
                assert!(loan.reserve_inventory_storage(usize::MAX).is_err());
                assert!(loan.reserve_inventory_storage(0).is_err());
                Ok::<_, SessionError>(())
            });
            assert!(result.is_err());
            assert_eq!(budget.storage(), before + 28);
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(budget.failed_storage(), Some(usize::MAX));
            retained.set(Some(budget.storage()));
            result
        },
        |(), _| panic!("refused loan reached publication"),
        |_, (), _| panic!("refused loan reached receipt"),
    );
    refused(result, "reference enrollment loan was refused");
    let client_storage =
        std::mem::size_of::<fe2o3_compiler_execution_client::CompilerExecutionClientV3<'_, '_>>();
    assert_eq!(budget.storage(), retained.get().unwrap() - client_storage);
    assert!(budget.work_ledger_identity_v1() == ledger);
    assert_eq!(budget.failed_storage(), Some(usize::MAX));
}

#[test]
fn original_inventory_reservation_reentrancy_is_sticky_without_a_debit() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::original_inventory_reservation_reentrancy_is_sticky_without_a_debit"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        let loan = loan.unwrap();
        loan.request(&mut source)?;
        let active = loan.revalidate.borrow_mut();
        assert!(
            loan.reserve_inventory_storage(11)
                .unwrap_err()
                .to_string()
                .contains("reentrant")
        );
        drop(active);
        assert!(loan.inventory_context(&mut source).is_err());
        assert!(loan.reserve_inventory_storage(0).is_err());
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    assert_eq!(budget.failed_storage(), None);
}

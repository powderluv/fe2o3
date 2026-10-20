//! Inert raw replay only: no publication, compiler, or currentness admission.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
use std::sync::Arc;

const PREFIX: usize = 17;
const OUTSIDE: usize = LIMIT + 97;

fn replay_owned(
    handoff: &Handoff,
    budget: &mut Budget<'_>,
) -> Result<(Subject, InertCompilerExecutionSubjectStorageV3)> {
    Subject::from_replay_evidence_in_original_account_v3(
        fields().attempt,
        CompilerModuleHandoffSlotV5::Production,
        CompilerModuleHandoffTransactionIdentityV5::from_bytes([7; 32]),
        handoff,
        budget,
    )
}

#[test]
fn replay_original_account_keeps_full_canonical_axes_and_strict_legacy_cap() {
    for handoff in [outer(), outer_variant(false, 19)] {
        let attempt = fields().attempt;
        for (attempt, transaction) in [
            (attempt, [7; 32]),
            (
                BuildAttempt::new(
                    attempt.generation() + 1,
                    attempt.session(),
                    attempt.invocation(),
                )
                .unwrap(),
                [7; 32],
            ),
            (
                BuildAttempt::new(
                    attempt.generation(),
                    crate::BuildSession::from_bytes([19; 16]),
                    attempt.invocation(),
                )
                .unwrap(),
                [7; 32],
            ),
            (
                BuildAttempt::new(
                    attempt.generation(),
                    attempt.session(),
                    crate::BuildInvocation::from_bytes([23; 32]),
                )
                .unwrap(),
                [7; 32],
            ),
            (attempt, [29; 32]),
        ] {
            let transaction = CompilerModuleHandoffTransactionIdentityV5::from_bytes(transaction);
            let mut legacy_work = Work::new(WORK);
            let mut legacy = Budget::new(&mut legacy_work, LIMIT);
            legacy
                .reserve_storage(handoff_floor(&handoff).unwrap())
                .unwrap();
            let (expected, charge) = Subject::from_replay_evidence(
                attempt,
                CompilerModuleHandoffSlotV5::Production,
                transaction,
                &handoff,
                &mut legacy,
            )
            .unwrap();
            legacy.reserve_storage(charge.retained_storage()).unwrap();

            let inputs = replay_input_floor(&handoff).unwrap();
            let total = 2 * 1024 * 1024 * 1024;
            let floor = OUTSIDE + inputs;
            let scratch = Subject::composed_replay_storage_v3(&handoff).unwrap();
            let mut owned = Owned::new(
                Work::new(PREFIX + 8 + Subject::COMPOSED_REPLAY_WORK_V3),
                total,
            );
            owned.with_budget(|budget| {
                budget.charge_work(PREFIX).unwrap();
                budget.reserve_storage(floor).unwrap();
                let ledger = budget.work_ledger_identity_v1();
                let account = budget.storage_account_identity_v1();
                assert!(matches!(
                    Subject::from_replay_evidence(
                        attempt,
                        CompilerModuleHandoffSlotV5::Production,
                        transaction,
                        &handoff,
                        budget
                    ),
                    Err(Failure::Resource(Resource::Accounting))
                ));
                let (actual, charge) = Subject::from_replay_evidence_in_original_account_v3(
                    attempt,
                    CompilerModuleHandoffSlotV5::Production,
                    transaction,
                    &handoff,
                    budget,
                )
                .unwrap();
                assert_eq!(budget.storage(), floor);
                assert_eq!(charge.retained_storage(), RETAINED);
                budget.reserve_storage(charge.retained_storage()).unwrap();
                assert_eq!(actual.canonical_bytes(), expected.canonical_bytes());
                assert_eq!(actual.identity(), expected.identity());
                assert_eq!(actual.attempt(), attempt);
                assert_eq!(actual.slot(), CompilerModuleHandoffSlotV5::Production);
                assert_eq!(actual.transaction_identity(), transaction);
                assert_eq!(
                    actual.rustc_invocation_sha256(),
                    handoff.capsule().invocation_digest().as_bytes()
                );
                inert(&actual);
                assert!(budget.work_ledger_identity_v1() == ledger);
                assert_eq!(budget.storage_account_identity_v1(), account);
                assert_eq!(budget.storage_limit(), total);
                assert_eq!(budget.peak_storage(), floor + scratch);
                assert_eq!(budget.work(), PREFIX + 8 + Subject::COMPOSED_REPLAY_WORK_V3);
                drop(actual);
                budget.release_storage(charge.retained_storage()).unwrap();
                assert_eq!(budget.storage(), floor);
            });
        }
    }
}

#[test]
fn replay_original_account_exact_and_one_short_quotes_preserve_original_state() {
    let handoff = outer();
    let inputs = replay_input_floor(&handoff).unwrap();
    let scratch = Subject::composed_replay_storage_v3(&handoff).unwrap();
    let floor = OUTSIDE + inputs;
    let peak = floor + scratch;
    for shortage in 0..3 {
        let total = peak - usize::from(shortage == 2);
        let mut owned = Owned::new(
            Work::new(PREFIX + Subject::COMPOSED_REPLAY_WORK_V3 - usize::from(shortage == 1)),
            total,
        );
        owned.with_budget(|budget| {
            budget.charge_work(PREFIX).unwrap();
            budget.reserve_storage(floor).unwrap();
            let address = budget as *const Budget<'_> as usize;
            let ledger = budget.work_ledger_identity_v1();
            let account = budget.storage_account_identity_v1();
            let result = replay_owned(&handoff, budget);
            match shortage {
                0 => {
                    let (subject, charge) = result.unwrap();
                    assert_eq!(charge.retained_storage(), RETAINED);
                    assert_eq!(budget.storage(), floor);
                    budget.reserve_storage(charge.retained_storage()).unwrap();
                    inert(&subject);
                    drop(subject);
                    budget.release_storage(charge.retained_storage()).unwrap();
                }
                1 => assert!(matches!(result, Err(Failure::Resource(Resource::Work(_))))),
                _ => assert!(matches!(
                    result,
                    Err(Failure::Resource(Resource::Storage(_)))
                )),
            }
            assert_eq!(budget as *const Budget<'_> as usize, address);
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(budget.storage_account_identity_v1(), account);
            assert_eq!(budget.storage(), floor);
            assert_eq!(budget.storage_limit(), total);
            assert_eq!(
                budget.work(),
                PREFIX
                    + if shortage == 0 {
                        Subject::COMPOSED_REPLAY_WORK_V3
                    } else {
                        Budget::STORAGE_WINDOW_WORK_V1 + 8
                    }
            );
            assert_eq!(
                budget.peak_storage(),
                if shortage == 2 { peak - SCRATCH } else { peak }
            );
            assert_eq!(
                budget.failed_work(),
                (shortage == 1).then_some(PREFIX + Subject::COMPOSED_REPLAY_WORK_V3)
            );
            assert_eq!(budget.failed_storage(), (shortage == 2).then_some(peak));
        });
    }
}

#[test]
fn replay_original_account_counts_spare_unselected_backing_and_coordinates() {
    let source = outer();
    let bytes = source.canonical_bytes();
    let mut backing = Vec::with_capacity(bytes.len() + 8192);
    backing.extend_from_slice(&[0x55; 31]);
    backing.extend_from_slice(bytes);
    backing.extend_from_slice(&[0x77; 47]);
    let capacity = backing.capacity();
    let handoff = Handoff::decode_shared_vec(Arc::new(backing), 31..31 + bytes.len()).unwrap();
    let coordinates = size_of::<(
        BuildAttempt,
        CompilerModuleHandoffSlotV5,
        CompilerModuleHandoffTransactionIdentityV5,
    )>();
    let inputs = capacity + METADATA + coordinates;
    let scratch = Subject::composed_replay_storage_v3(&handoff).unwrap();
    assert_eq!(replay_input_floor(&handoff).unwrap(), inputs);
    assert_eq!(
        scratch,
        inputs + Budget::STORAGE_WINDOW_SCRATCH_V1 + SCRATCH
    );
    for paid in [bytes.len() + METADATA + coordinates, inputs - 1, inputs] {
        let mut owned = Owned::new(
            Work::new(PREFIX + Subject::COMPOSED_REPLAY_WORK_V3),
            inputs + scratch,
        );
        owned.with_budget(|budget| {
            budget.charge_work(PREFIX).unwrap();
            budget.reserve_storage(paid).unwrap();
            let result = replay_owned(&handoff, budget);
            if paid == inputs {
                let (subject, charge) = result.unwrap();
                budget.reserve_storage(charge.retained_storage()).unwrap();
                assert_eq!(subject.outer_handoff().sha256(), source.identity().sha256());
                inert(&subject);
                drop(subject);
                budget.release_storage(charge.retained_storage()).unwrap();
                assert_eq!(budget.peak_storage(), inputs + scratch);
                assert_eq!(budget.work(), PREFIX + Subject::COMPOSED_REPLAY_WORK_V3);
            } else {
                assert!(matches!(
                    result,
                    Err(Failure::Resource(Resource::Accounting))
                ));
                assert_eq!(budget.work(), PREFIX + Budget::STORAGE_WINDOW_WORK_V1);
                assert_eq!(budget.peak_storage(), paid);
            }
            assert_eq!(budget.storage(), paid);
            assert_eq!(budget.failed_work(), None);
            assert_eq!(budget.failed_storage(), None);
        });
    }
}

#[test]
fn replay_unpaid_foreign_account_cannot_borrow_floor_but_prepaid_replay_is_inert() {
    let handoff = outer();
    let inputs = replay_input_floor(&handoff).unwrap();
    let scratch = Subject::composed_replay_storage_v3(&handoff).unwrap();
    let mut original = Owned::new(Work::new(PREFIX), OUTSIDE + inputs);
    original.with_budget(|original| {
        original.charge_work(PREFIX).unwrap();
        original.reserve_storage(OUTSIDE + inputs).unwrap();
        let original_ledger = original.work_ledger_identity_v1();
        let original_account = original.storage_account_identity_v1();
        for paid in [inputs - 1, inputs] {
            let mut foreign = Owned::new(
                Work::new(Subject::COMPOSED_REPLAY_WORK_V3),
                inputs + scratch,
            );
            foreign.with_budget(|foreign| {
                foreign.reserve_storage(paid).unwrap();
                let ledger = foreign.work_ledger_identity_v1();
                let account = foreign.storage_account_identity_v1();
                assert!(ledger != original_ledger);
                assert_ne!(account, original_account);
                let result = replay_owned(&handoff, foreign);
                if paid == inputs {
                    let (subject, charge) = result.unwrap();
                    foreign.reserve_storage(charge.retained_storage()).unwrap();
                    assert_eq!(
                        subject.outer_handoff().sha256(),
                        handoff.identity().sha256()
                    );
                    inert(&subject);
                    drop(subject);
                    foreign.release_storage(charge.retained_storage()).unwrap();
                } else {
                    assert!(matches!(
                        result,
                        Err(Failure::Resource(Resource::Accounting))
                    ));
                    assert_eq!(foreign.work(), Budget::STORAGE_WINDOW_WORK_V1);
                }
                assert_eq!(foreign.storage(), paid);
                assert!(foreign.work_ledger_identity_v1() == ledger);
                assert_eq!(foreign.storage_account_identity_v1(), account);
            });
            assert_eq!(original.storage(), OUTSIDE + inputs);
            assert_eq!(original.work(), PREFIX);
            assert_eq!(original.peak_storage(), OUTSIDE + inputs);
            assert!(original.work_ledger_identity_v1() == original_ledger);
            assert_eq!(original.storage_account_identity_v1(), original_account);
            assert_eq!(original.failed_work(), None);
            assert_eq!(original.failed_storage(), None);
        }
    });
    let mut work = Work::new(Subject::COMPOSED_REPLAY_WORK_V3);
    let mut inline = Budget::new(&mut work, inputs + scratch);
    inline.reserve_storage(inputs).unwrap();
    assert!(matches!(
        replay_owned(&handoff, &mut inline),
        Err(Failure::Resource(Resource::Accounting))
    ));
    assert_eq!(inline.storage(), inputs);
    assert_eq!(inline.peak_storage(), inputs);
    assert_eq!(inline.work(), Budget::STORAGE_WINDOW_WORK_V1);
    assert_eq!(inline.storage_account_identity_v1(), None);
}

#[test]
fn replay_original_account_keeps_first_denials_on_success_and_invalid_coordinates() {
    let handoff = outer();
    let inputs = replay_input_floor(&handoff).unwrap();
    let scratch = Subject::composed_replay_storage_v3(&handoff).unwrap();
    let floor = OUTSIDE + inputs;
    let mut owned = Owned::new(
        Work::new(PREFIX + 2 * Subject::COMPOSED_REPLAY_WORK_V3),
        floor + scratch,
    );
    owned.with_budget(|budget| {
        budget.charge_work(PREFIX).unwrap();
        budget.reserve_storage(floor).unwrap();
        assert!(budget.charge_work(usize::MAX).is_err());
        assert!(budget.reserve_storage(usize::MAX).is_err());
        let ledger = budget.work_ledger_identity_v1();
        let account = budget.storage_account_identity_v1();
        for (index, transaction) in [[7; 32], [0; 32]].into_iter().enumerate() {
            let result = Subject::from_replay_evidence_in_original_account_v3(
                fields().attempt,
                CompilerModuleHandoffSlotV5::Production,
                CompilerModuleHandoffTransactionIdentityV5::from_bytes(transaction),
                &handoff,
                budget,
            );
            if index == 0 {
                let (subject, charge) = result.unwrap();
                budget.reserve_storage(charge.retained_storage()).unwrap();
                inert(&subject);
                drop(subject);
                budget.release_storage(charge.retained_storage()).unwrap();
            } else {
                assert!(matches!(
                    result,
                    Err(Failure::Framing(
                        CompilerExecutionSubjectErrorV1::ZeroIdentity {
                            field: "V5 handoff transaction"
                        }
                    ))
                ));
            }
            assert_eq!(budget.storage(), floor);
            assert_eq!(budget.storage_limit(), floor + scratch);
            assert_eq!(budget.peak_storage(), floor + scratch);
            assert_eq!(
                budget.work(),
                PREFIX + (index + 1) * Subject::COMPOSED_REPLAY_WORK_V3
            );
            assert_eq!(budget.failed_work(), Some(usize::MAX));
            assert_eq!(budget.failed_storage(), Some(usize::MAX));
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(budget.storage_account_identity_v1(), account);
        }
    });
}

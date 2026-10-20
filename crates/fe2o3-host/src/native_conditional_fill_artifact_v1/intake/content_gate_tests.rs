//! Inert comparison/accounting and source-order regressions, not native qualification.
use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned,
    CanonicalKernelIrWorkBudgetV1 as Work,
};

#[allow(dead_code)]
#[path = "../../../../fe2o3-compiler-execution-protocol/tests/support/native_attestation_fixture.rs"]
mod fixture;

const WORK: usize = 1_000_000_000;
const STORAGE: usize = 512 * 1024 * 1024;
const POLICY_WORK: usize = 216 + 32 + 8 + 32;

fn policy(bytes: &[u8], budget: &mut Budget<'_>) -> Policy {
    budget.reserve_storage(bytes.len()).unwrap();
    let (policy, charge) = Policy::decode(bytes, budget).unwrap();
    budget.reserve_storage(charge.additional_storage()).unwrap();
    policy
}

fn subject(bytes: &[u8], budget: &mut Budget<'_>) -> Subject {
    budget.reserve_storage(bytes.len()).unwrap();
    let (subject, charge) = Subject::decode_in_original_account_v3(bytes, budget).unwrap();
    budget.reserve_storage(charge.retained_storage()).unwrap();
    subject
}

#[test]
fn exact_installed_policy_and_config_pin_pay_complete_comparison() {
    let bytes = fixture::policy_wire(3);
    let mut work = Work::new(WORK);
    let mut budget = Budget::new(&mut work, 1024 * 1024);
    let independent = policy(&bytes, &mut budget);
    let carried = policy(&bytes, &mut budget);
    let before = budget.work();
    let storage = budget.storage();
    require_same_policy(
        &independent,
        &carried,
        independent.identity().as_bytes(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(budget.work(), before + POLICY_WORK);
    assert_eq!(budget.storage(), storage);
    assert_eq!(independent.canonical_bytes().len(), 216);
}

#[test]
fn policy_rejects_resealed_generation_measurements_and_distinct_keys() {
    let bytes = fixture::policy_wire(3);
    let mut work = Work::new(WORK);
    let mut budget = Budget::new(&mut work, 1024 * 1024);
    let independent = policy(&bytes, &mut budget);
    for offset in [24, 32, 64, 72, 104, 112, 144] {
        let mut changed = bytes;
        if offset >= 112 {
            let key = ed25519_dalek::SigningKey::from_bytes(&[offset as u8; 32])
                .verifying_key()
                .to_bytes();
            changed[offset..offset + 32].copy_from_slice(&key);
        } else {
            changed[offset] ^= 1;
        }
        fixture::seal(&mut changed, "COMPILER-EXECUTION-ISSUER-POLICY", 3);
        let carried = policy(&changed, &mut budget);
        let before = budget.work();
        let storage = budget.storage();
        assert!(
            require_same_policy(
                &independent,
                &carried,
                independent.identity().as_bytes(),
                &mut budget,
            )
            .is_err(),
            "{offset}",
        );
        assert_eq!(budget.work(), before + POLICY_WORK);
        assert_eq!(budget.storage(), storage);
    }
}

#[test]
fn compiler_policy_pin_is_not_a_semantic_policy_digest() {
    let bytes = fixture::policy_wire(3);
    let mut work = Work::new(WORK);
    let mut budget = Budget::new(&mut work, 1024 * 1024);
    let independent = policy(&bytes, &mut budget);
    let semantic_digest: [u8; 32] = Sha256::digest(b"native semantic policy bytes").into();
    assert_ne!(*independent.identity().as_bytes(), semantic_digest);
    assert!(
        require_same_policy(&independent, &independent, &semantic_digest, &mut budget).is_err()
    );
}

#[test]
fn policy_strict_decode_rejects_bad_id_reserved_and_legacy_wire() {
    for case in 0..5 {
        let mut bytes = fixture::policy_wire(if case == 3 { 2 } else { 3 });
        match case {
            0 => bytes[184] ^= 1,
            1 => {
                bytes[178] = 1;
                fixture::seal(&mut bytes, "COMPILER-EXECUTION-ISSUER-POLICY", 3);
            }
            2 => {
                bytes[24..32].fill(0);
                fixture::seal(&mut bytes, "COMPILER-EXECUTION-ISSUER-POLICY", 3);
            }
            3 => {}
            4 => bytes[32] ^= 1,
            _ => unreachable!(),
        }
        let mut work = Work::new(WORK);
        let mut budget = Budget::new(&mut work, 1024 * 1024);
        budget.reserve_storage(bytes.len()).unwrap();
        assert!(Policy::decode(&bytes, &mut budget).is_err(), "{case}");
    }
}

#[test]
fn policy_comparison_exact_and_one_short_work_keep_storage_and_denials() {
    for remaining in [POLICY_WORK - 1, POLICY_WORK] {
        let bytes = fixture::policy_wire(3);
        let mut work = Work::new(WORK);
        let mut budget = Budget::new(&mut work, 1024 * 1024);
        let independent = policy(&bytes, &mut budget);
        budget
            .charge_work(WORK - budget.work() - remaining)
            .unwrap();
        let before = budget.work();
        let storage = budget.storage();
        let result = require_same_policy(
            &independent,
            &independent,
            independent.identity().as_bytes(),
            &mut budget,
        );
        assert_eq!(result.is_ok(), remaining == POLICY_WORK);
        assert_eq!(budget.storage(), storage);
        if remaining == POLICY_WORK {
            assert_eq!(budget.work(), WORK);
        } else {
            assert_eq!(budget.work(), before);
            assert_eq!(budget.failed_work(), Some(before + POLICY_WORK));
        }
    }
}

#[test]
fn raw_subject_comparison_covers_resealed_occurrence_closure_and_content_axes() {
    let bytes = fixture::subject_wire(3);
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|budget| {
        let original = subject(&bytes, budget);
        let axes = [24, 32, 48, 88, 120]
            .into_iter()
            .chain((0..6).map(|axis| 152 + axis * 32))
            .chain((0..7).flat_map(|axis| [378 + axis * 40, 410 + axis * 40]));
        for offset in axes {
            let mut changed = bytes;
            changed[offset] ^= 1;
            if (152..344).contains(&offset) {
                let mut closure = Sha256::new();
                closure.update(b"fe2o3-compiler-closure-identity-v2\0");
                closure.update(1u16.to_le_bytes());
                closure.update(&changed[152..344]);
                changed[346..378].copy_from_slice(&closure.finalize());
            }
            fixture::seal(&mut changed, "INERT-COMPILER-EXECUTION-SUBJECT", 3);
            let claimed = subject(&changed, budget);
            let before = budget.work();
            let storage = budget.storage();
            assert!(
                require_same_subject(&original, &claimed, budget).is_err(),
                "{offset}"
            );
            assert_eq!(budget.work(), before + 690);
            assert_eq!(budget.storage(), storage);
        }
    });
}

#[test]
fn raw_subject_equality_prepays_exact_work_and_does_not_refund_denial() {
    for remaining in [689, 690] {
        let bytes = fixture::subject_wire(3);
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|budget| {
            let original = subject(&bytes, budget);
            budget
                .charge_work(WORK - budget.work() - remaining)
                .unwrap();
            let before = budget.work();
            let storage = budget.storage();
            let result = require_same_subject(&original, &original, budget);
            assert_eq!(result.is_ok(), remaining == 690);
            assert_eq!(budget.storage(), storage);
            if remaining == 690 {
                assert_eq!(budget.work(), WORK);
            } else {
                assert_eq!(budget.work(), before);
                assert_eq!(budget.failed_work(), Some(before + 690));
            }
        });
    }
}

#[test]
fn paid_original_account_accepts_nonzero_floor_and_rejects_missing_owner() {
    let mut owned = Owned::new(Work::new(WORK), 4096);
    owned.with_budget(|budget| {
        budget.reserve_storage(97).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let account = budget.storage_account_identity_v1().unwrap();
        require_original_account(ledger, account, 97, budget).unwrap();
        budget.release_storage(1).unwrap();
        assert!(matches!(
            require_original_account(ledger, account, 97, budget),
            Err(Error::Resource(Resource::Accounting)),
        ));
        assert_eq!(budget.storage(), 96);
    });
}

#[test]
fn equal_prepaid_floor_cannot_substitute_another_ledger_or_account() {
    let mut original = Owned::new(Work::new(WORK), 4096);
    original.with_budget(|budget| {
        budget.reserve_storage(97).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let account = budget.storage_account_identity_v1().unwrap();
        let mut foreign = Owned::new(Work::new(WORK), 4096);
        foreign.with_budget(|other| {
            other.reserve_storage(97).unwrap();
            let other_ledger = other.work_ledger_identity_v1();
            let other_account = other.storage_account_identity_v1().unwrap();
            assert!(require_original_account(ledger, other_account, 97, other).is_err());
            assert!(require_original_account(other_ledger, account, 97, other).is_err());
            assert_eq!(other.storage(), 97);
        });
        assert_eq!(budget.storage(), 97);
    });
}

#[test]
fn account_guard_denial_is_charged_before_any_acceptance() {
    let mut owned = Owned::new(Work::new(3), 4096);
    owned.with_budget(|budget| {
        budget.reserve_storage(97).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let account = budget.storage_account_identity_v1().unwrap();
        assert!(matches!(
            require_original_account(ledger, account, 97, budget),
            Err(Error::Resource(_)),
        ));
        assert_eq!(budget.work(), 0);
        assert_eq!(budget.failed_work(), Some(4));
        assert_eq!(budget.storage(), 97);
    });
}

fn before(source: &str, first: &str, second: &str) {
    assert!(source.find(first).unwrap() < source.find(second).unwrap());
}

#[test]
fn source_order_authenticates_before_publication_custody_and_registration() {
    let intake = include_str!("../intake.rs");
    before(
        intake,
        "profile.revalidate(compiler, budget)",
        "content::authenticate(",
    );
    before(
        intake,
        "content::authenticate(",
        "Directory::admit_service_owned(",
    );
    before(intake, "content::authenticate(", "Publication::recover(");
    before(intake, "content::recover(", "Endpoint::admit_inherited(");
    before(intake, "content::recover(", "Inputs::new(");
}

#[test]
fn source_order_uses_one_authenticated_carriage_and_original_subject_coordinates() {
    let content = include_str!("content.rs")
        .split("#[cfg(test)]")
        .next()
        .unwrap();
    assert_eq!(
        content
            .matches("Carriage::decode_in_original_account_v3(")
            .count(),
        1
    );
    assert_eq!(content.matches("Handoff::decode_owned(").count(), 1);
    let authenticate = content.split("pub(super) fn authenticate").nth(1).unwrap();
    before(
        authenticate,
        "Carriage::decode_in_original_account_v3(",
        "require_same_policy(",
    );
    before(
        authenticate,
        "require_same_policy(",
        "Handoff::decode_owned(",
    );
    before(
        authenticate,
        "Handoff::decode_owned(",
        "authenticate_raw_subject(",
    );
    let raw = content
        .split("fn authenticate_raw_subject(")
        .nth(1)
        .unwrap()
        .split("fn require_same_subject(")
        .next()
        .unwrap();
    before(
        raw,
        "budget.reserve_storage(coordinates)",
        "Subject::from_replay_evidence",
    );
    assert!(raw.contains(
        "claimed.attempt(),\n        claimed.slot(),\n        claimed.transaction_identity(),"
    ));
    assert!(!raw.contains("record.attempt()"));
    before(raw, "require_same_subject(", "drop(subject)");
    before(
        raw,
        "drop(subject)",
        "budget.release_storage(charge.retained_storage())",
    );
}

#[test]
fn source_order_preserves_account_gate_semantic_recovery_and_publication_recheck() {
    let content = include_str!("content.rs");
    let recover = content
        .split("pub(super) fn recover")
        .nth(1)
        .unwrap()
        .split("fn exact(")
        .next()
        .unwrap();
    before(recover, "require_original_account(", "recover_source(");
    before(
        recover,
        "recover_source(",
        "let (finalized, charge) = replay(",
    );
    before(
        recover,
        "let (finalized, charge) = replay(",
        "derive_intent(",
    );
    before(
        recover,
        "derive_intent(",
        "Subject::from_publication_in_original_account_v3(",
    );
    before(
        recover,
        "native actual V5 compiler-execution subject",
        "drop(subject)",
    );
}

#[path = "content_gate_raw_fixture.rs"]
mod raw_fixture;

#[path = "content_enrollment_tests.rs"]
mod enrollment_tests;
#[path = "verifier_denial_tests.rs"]
mod verifier_denial_tests;

fn raw_coordinates_storage() -> usize {
    size_of::<(
        fe2o3_artifact_transaction::BuildAttempt,
        fe2o3_artifact_transaction::CompilerModuleHandoffSlotV5,
        fe2o3_artifact_transaction::CompilerModuleHandoffTransactionIdentityV5,
    )>()
}

fn raw_claim_wire(handoff: &Handoff, generation: u64) -> [u8; 690] {
    use fe2o3_artifact_transaction::{
        BuildAttempt, BuildInvocation, BuildSession, CompilerModuleHandoffSlotV5 as Slot,
        CompilerModuleHandoffTransactionIdentityV5 as Transaction,
        MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5,
    };
    // Independent legacy inert reconstruction prepares fixture bytes only.
    let mut work = Work::new(WORK);
    let mut budget = Budget::new(&mut work, MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5);
    budget
        .reserve_storage(handoff.backing_capacity() + METADATA + raw_coordinates_storage())
        .unwrap();
    let attempt = BuildAttempt::from_env_value(&format!(
        "{generation}:{}:{}",
        BuildSession::from_bytes([0x26; 16]),
        BuildInvocation::from_bytes([0x27; 32]),
    ))
    .unwrap();
    let (subject, charge) = Subject::from_replay_evidence(
        attempt,
        Slot::Production,
        Transaction::from_bytes([0x28; 32]),
        handoff,
        &mut budget,
    )
    .unwrap();
    budget.reserve_storage(charge.retained_storage()).unwrap();
    budget.reserve_storage(690).unwrap();
    let bytes = *subject.canonical_bytes();
    drop(subject);
    budget.release_storage(charge.retained_storage()).unwrap();
    bytes
}

fn raw_inputs(handoff: &Handoff, wire: &[u8], budget: &mut Budget<'_>) -> (Subject, usize) {
    budget
        .reserve_storage(97 + handoff.backing_capacity() + METADATA + wire.len())
        .unwrap();
    let (claimed, charge) = Subject::decode_in_original_account_v3(wire, budget).unwrap();
    budget.reserve_storage(charge.retained_storage()).unwrap();
    (claimed, charge.retained_storage())
}

#[test]
fn direct_raw_composition_accepts_actual_handoff_with_full_backing_and_receipt_coordinates() {
    for (spare, generation) in [(false, 9), (true, 31)] {
        let original = raw_fixture::handoff();
        let handoff = if spare {
            let bytes = original.canonical_bytes();
            let mut backing = Vec::with_capacity(bytes.len() + 8192);
            backing.extend_from_slice(&[0x55; 31]);
            backing.extend_from_slice(bytes);
            backing.extend_from_slice(&[0x77; 47]);
            let handoff =
                Handoff::decode_shared_vec(std::sync::Arc::new(backing), 31..31 + bytes.len())
                    .unwrap();
            drop(original);
            handoff
        } else {
            original
        };
        let wire = raw_claim_wire(&handoff, generation);
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|budget| {
            let (claimed, _) = raw_inputs(&handoff, &wire, budget);
            let floor = budget.storage();
            let ledger = budget.work_ledger_identity_v1();
            let account = budget.storage_account_identity_v1();
            let before = budget.work();
            let peak = budget.peak_storage();
            let reconstruction_peak = floor
                + raw_coordinates_storage()
                + Subject::composed_replay_storage_v3(&handoff).unwrap();
            authenticate_raw_subject(&handoff, &claimed, budget).unwrap();
            assert_eq!(claimed.attempt().generation(), generation);
            assert_eq!(budget.storage(), floor);
            assert_eq!(
                budget.work(),
                before + Subject::COMPOSED_REPLAY_WORK_V3 + 690
            );
            assert_eq!(budget.peak_storage(), peak.max(reconstruction_peak));
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(budget.storage_account_identity_v1(), account);
            assert_eq!(budget.failed_work(), None);
            assert_eq!(budget.failed_storage(), None);
        });
    }
}

#[test]
fn direct_raw_composition_rejects_coherently_resealed_content_and_preserves_denial_history() {
    for prior_denial in [false, true] {
        let handoff = raw_fixture::handoff();
        let mut wire = raw_claim_wire(&handoff, 9);
        wire[378] ^= 1;
        fixture::seal(&mut wire, "INERT-COMPILER-EXECUTION-SUBJECT", 3);
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|budget| {
            let (claimed, retained) = raw_inputs(&handoff, &wire, budget);
            let floor = budget.storage();
            let ledger = budget.work_ledger_identity_v1();
            let account = budget.storage_account_identity_v1();
            if prior_denial {
                assert!(budget.charge_work(WORK).is_err());
            }
            let denied = budget.failed_work();
            let error = authenticate_raw_subject(&handoff, &claimed, budget).unwrap_err();
            if !prior_denial {
                assert!(
                    matches!(error, Error::Rejected(ref reason)
                        if reason == "native raw V5 compiler-execution subject"),
                    "{error}",
                );
                assert_eq!(
                    budget.storage(),
                    floor + raw_coordinates_storage() + retained
                );
            } else {
                // This inspects primitive failure history, not a native retry policy.
                assert!(budget.storage() >= floor);
            }
            assert_eq!(budget.failed_work(), denied);
            assert_eq!(budget.failed_storage(), None);
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert_eq!(budget.storage_account_identity_v1(), account);
        });
    }
}

#[test]
fn direct_raw_composition_one_short_compare_work_retains_known_temporary_charges() {
    let handoff = raw_fixture::handoff();
    let wire = raw_claim_wire(&handoff, 9);
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|budget| {
        let (claimed, retained) = raw_inputs(&handoff, &wire, budget);
        let floor = budget.storage();
        budget
            .charge_work(WORK - budget.work() - Subject::COMPOSED_REPLAY_WORK_V3 - 689)
            .unwrap();
        assert!(matches!(
            authenticate_raw_subject(&handoff, &claimed, budget),
            Err(Error::Resource(_)),
        ));
        assert_eq!(budget.work(), WORK - 689);
        assert_eq!(budget.failed_work(), Some(WORK + 1));
        assert_eq!(
            budget.storage(),
            floor + raw_coordinates_storage() + retained
        );
    });
}

#[test]
fn direct_raw_composition_prepays_coordinates_before_reconstruction_work() {
    let handoff = raw_fixture::handoff();
    let wire = raw_claim_wire(&handoff, 9);
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|budget| {
        let (claimed, _) = raw_inputs(&handoff, &wire, budget);
        budget
            .reserve_storage(Budget::STORAGE_WINDOW_SCRATCH_V1)
            .unwrap();
        let floor = budget.storage();
        let before = budget.work();
        let result: Result<()> = budget
            .with_additional_storage_window_v1(raw_coordinates_storage() - 1, |budget| {
                authenticate_raw_subject(&handoff, &claimed, budget)
            });
        assert!(matches!(result, Err(Error::Resource(_))));
        assert_eq!(budget.work(), before + Budget::STORAGE_WINDOW_WORK_V1);
        assert_eq!(budget.storage(), floor);
        assert_eq!(
            budget.failed_storage(),
            Some(floor + raw_coordinates_storage())
        );
    });
}

#[test]
fn direct_raw_composition_nested_scratch_shortage_never_refunds_original_inputs() {
    let handoff = raw_fixture::handoff();
    let wire = raw_claim_wire(&handoff, 9);
    let scratch = Subject::composed_replay_storage_v3(&handoff).unwrap();
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|budget| {
        let (claimed, _) = raw_inputs(&handoff, &wire, budget);
        budget
            .reserve_storage(Budget::STORAGE_WINDOW_SCRATCH_V1)
            .unwrap();
        let floor = budget.storage();
        let account = budget.storage_account_identity_v1();
        let result: Result<()> = budget
            .with_additional_storage_window_v1(raw_coordinates_storage() + scratch - 1, |budget| {
                authenticate_raw_subject(&handoff, &claimed, budget)
            });
        assert!(result.is_err());
        assert_eq!(budget.storage(), floor + raw_coordinates_storage());
        assert_eq!(
            budget.failed_storage(),
            Some(floor + raw_coordinates_storage() + scratch),
        );
        assert_eq!(budget.storage_account_identity_v1(), account);
    });
}

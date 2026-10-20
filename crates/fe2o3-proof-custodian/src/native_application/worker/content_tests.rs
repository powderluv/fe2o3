//! Inert comparison/accounting only, not signature, currentness or launch evidence.
use super::*;

#[allow(dead_code)]
#[path = "../../../../fe2o3-compiler-execution-protocol/tests/support/native_attestation_fixture.rs"]
mod fixture;

fn decode(bytes: &[u8], budget: &mut Budget<'_>) -> Subject {
    budget.reserve_storage(bytes.len()).unwrap();
    let (subject, charge) = Subject::decode_in_original_account_v3(bytes, budget).unwrap();
    budget.reserve_storage(charge.retained_storage()).unwrap();
    subject
}

#[test]
fn raw_subject_comparison_covers_resealed_occurrence_closure_and_content_axes() {
    let bytes = fixture::subject_wire(3);
    let mut account = super::super::controller_account();
    account.with_budget(|budget| {
        let original = decode(&bytes, budget);
        let work = budget.work();
        require_same_subject(&original, &original, budget).unwrap();
        assert_eq!(budget.work(), work + bytes.len());
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
            let claimed = decode(&changed, budget);
            let work = budget.work();
            let storage = budget.storage();
            let error = require_same_subject(&original, &claimed, budget).unwrap_err();
            assert_eq!(error.to_string(), "native V5 subject differs", "{offset}");
            assert_eq!(budget.work(), work + bytes.len());
            assert_eq!(budget.storage(), storage);
        }
    });
}

#[test]
fn native_controller_original_account_accepts_codec_rejected_by_windowless_budget() {
    let bytes = fixture::subject_wire(3);
    let mut work = Work::new(super::super::CONTROLLER_WORK);
    let mut windowless = Budget::new(&mut work, super::super::CONTROLLER_STORAGE);
    windowless.reserve_storage(bytes.len()).unwrap();
    assert!(windowless.storage_account_identity_v1().is_none());
    assert!(Subject::decode_in_original_account_v3(&bytes, &mut windowless).is_err());
    assert_eq!(windowless.storage(), bytes.len());

    let mut account = super::super::controller_account();
    account.with_budget(|budget| {
        assert!(budget.storage_account_identity_v1().is_some());
        let subject = decode(&bytes, budget);
        assert_eq!(subject.canonical_bytes(), &bytes);
    });
}

#[test]
fn raw_subject_comparison_funds_work_before_accepting_equal_content() {
    for remaining in [fixture::SUBJECT_BYTES - 1, fixture::SUBJECT_BYTES] {
        let bytes = fixture::subject_wire(3);
        let mut account = super::super::controller_account();
        account.with_budget(|budget| {
            let subject = decode(&bytes, budget);
            budget
                .charge_work(super::super::CONTROLLER_WORK - budget.work() - remaining)
                .unwrap();
            let before = budget.work();
            let storage = budget.storage();
            let result = require_same_subject(&subject, &subject, budget);
            assert_eq!(result.is_ok(), remaining == bytes.len());
            assert_eq!(budget.storage(), storage);
            if remaining == bytes.len() {
                assert_eq!(budget.work(), super::super::CONTROLLER_WORK);
            } else {
                assert_eq!(budget.work(), before);
                assert_eq!(budget.failed_work(), Some(before + bytes.len()));
            }
        });
    }
}

//! Inert framing and content tests only, never protected compiler/proof evidence.
use super::*;
use crate::compiler_module_handoff::conditional_v5::tests::fixture::{
    Fixture, outer, outer_variant,
};
use crate::{InertCompilerExecutionSubjectV1 as V1, InertCompilerExecutionSubjectV2 as V2};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use sha2::{Digest, Sha256};

type Subject = InertCompilerExecutionSubjectV3;
const LIMIT: usize = MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5;
const WORK: usize = INERT_COMPILER_EXECUTION_SUBJECT_WORK_V3;
const SCRATCH: usize = INERT_COMPILER_EXECUTION_SUBJECT_STORAGE_V3;

fn fields() -> codec::Fields {
    super::super::tests::fields()
}
fn wire() -> [u8; 690] {
    *Subject::from_fields(fields()).unwrap().canonical_bytes()
}
fn v2_encoded() -> codec::Encoded {
    codec::Schema {
        magic: *b"F2O3CES2",
        version: 2,
        identity_domain: b"FE2O3/INERT-COMPILER-EXECUTION-SUBJECT/V2\0",
        transaction_label: "V4 handoff transaction",
        outer_label: "outer V4 handoff",
    }
    .encode(&fields())
    .unwrap()
}
fn reseal(bytes: &mut [u8; 690]) {
    let digest = SCHEMA.identity(&bytes[..658]);
    bytes[658..].copy_from_slice(&digest);
}
fn decode(bytes: &[u8]) -> Result<(Subject, InertCompilerExecutionSubjectStorageV3)> {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, LIMIT);
    budget.reserve_storage(bytes.len()).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let result = Subject::decode(bytes, &mut budget);
    assert_eq!(budget.storage(), bytes.len());
    assert_eq!(budget.work(), WORK);
    assert!(budget.work_ledger_identity_v1() == ledger);
    result
}
fn replay(
    handoff: &Handoff,
    budget: &mut Budget<'_>,
) -> Result<(Subject, InertCompilerExecutionSubjectStorageV3)> {
    Subject::from_replay_evidence(
        fields().attempt,
        CompilerModuleHandoffSlotV5::Production,
        CompilerModuleHandoffTransactionIdentityV5::from_bytes([7; 32]),
        handoff,
        budget,
    )
}
fn inert(subject: &Subject) {
    assert!(!subject.authenticates_compiler_execution());
    assert!(subject.requires_protected_execution_attestation());
    assert!(!subject.grants_compiler_authority());
    assert!(!subject.grants_publication_authority());
    assert!(!subject.grants_load_authority());
    assert!(!subject.grants_launch_authority());
}

#[test]
fn original_account_decode_keeps_outside_owners_and_strict_legacy_cap() {
    use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
    let bytes = wire();
    const OUTSIDE: usize = LIMIT + 97;
    let total = OUTSIDE + bytes.len() + Subject::COMPOSED_DECODE_SCRATCH;
    let mut account = Owned::new(Work::new(Subject::COMPOSED_DECODE_WORK + 8), total);
    account.with_budget(|budget| {
        budget.reserve_storage(OUTSIDE + bytes.len()).unwrap();
        let floor = budget.storage();
        let ledger = budget.work_ledger_identity_v1();
        let storage_account = budget.storage_account_identity_v1();
        assert!(Subject::decode(&bytes, budget).is_err());
        let (subject, charge) = Subject::decode_in_original_account_v3(&bytes, budget).unwrap();
        assert_eq!(subject.canonical_bytes(), &bytes);
        inert(&subject);
        assert_eq!(charge.retained_storage(), RETAINED);
        assert_eq!(budget.storage(), floor);
        assert_eq!(budget.storage_limit(), total);
        assert_eq!(budget.peak_storage(), total);
        assert_eq!(budget.work(), Subject::COMPOSED_DECODE_WORK + 8);
        assert!(budget.work_ledger_identity_v1() == ledger);
        assert_eq!(budget.storage_account_identity_v1(), storage_account);
    });
}

#[test]
fn original_account_decode_exact_one_short_and_bad_input_preserve_account() {
    use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
    let bytes = wire();
    for case in 0..5 {
        let quota = Subject::COMPOSED_DECODE_WORK - usize::from(case == 1);
        let floor = bytes.len() - usize::from(case == 3);
        let limit = floor + Subject::COMPOSED_DECODE_SCRATCH - usize::from(case == 2);
        let mut account = Owned::new(Work::new(quota), limit);
        account.with_budget(|budget| {
            budget.reserve_storage(floor).unwrap();
            let input = if case == 4 { &bytes[..689] } else { &bytes[..] };
            let result = Subject::decode_in_original_account_v3(input, budget);
            assert_eq!(result.is_ok(), case == 0);
            assert_eq!(budget.storage(), floor);
            assert_eq!(budget.storage_limit(), limit);
            if case == 1 {
                assert!(budget.failed_work().is_some());
            }
            if case == 2 {
                assert!(budget.failed_storage().is_some());
            }
        });
    }
    let mut work = Work::new(Subject::COMPOSED_DECODE_WORK);
    let mut inline = Budget::new(&mut work, LIMIT);
    inline.reserve_storage(bytes.len()).unwrap();
    assert!(Subject::decode_in_original_account_v3(&bytes, &mut inline).is_err());
    assert_eq!(inline.storage(), bytes.len());
    assert_eq!(inline.work(), Budget::STORAGE_WINDOW_WORK_V1);
}

#[test]
fn conditional_subject_v3_same_690_byte_fields_and_old_golden_bytes() {
    let f = fields();
    let subject = Subject::from_fields(fields()).unwrap();
    let bytes = subject.canonical_bytes();
    assert_eq!(
        &bytes[..24],
        b"F2O3CES3\x03\0\0\0\xb2\x02\0\0\0\0\0\0\0\0\0\0"
    );
    assert_eq!(&bytes[24..32], &f.attempt.generation().to_le_bytes());
    assert_eq!(&bytes[32..48], f.attempt.session().as_bytes());
    assert_eq!(&bytes[48..80], f.attempt.invocation().as_bytes());
    assert_eq!(&bytes[80..88], &[0; 8]);
    assert_eq!(&bytes[88..120], &f.transaction_identity);
    assert_eq!(&bytes[120..152], &f.rustc_invocation_sha256);
    let closure = f.compiler_closure;
    for (slot, digest) in bytes[152..344].chunks_exact(32).zip([
        closure.cargo_executable_sha256(),
        closure.cargo_binding_trampoline_sha256(),
        closure.cargo_fe2o3_binding_wrapper_sha256(),
        closure.rustc_executable_sha256(),
        closure.rustc_runtime_tree_sha256(),
        closure.codegen_backend_sha256(),
    ]) {
        assert_eq!(slot, digest);
    }
    assert_eq!(
        &bytes[344..346],
        &closure
            .cargo_binding_transition_protocol_version()
            .to_le_bytes()
    );
    assert_eq!(&bytes[346..378], &closure.identity_sha256());
    for (slot, binding) in bytes[378..658].chunks_exact(40).zip([
        f.rustc_identity_inventory,
        f.rustc_preflight_plan,
        f.semantic_capsule,
        f.final_compiler_module_commitment,
        f.compiler_module_handoff,
        f.compiler_module_pair_binding,
        f.outer_handoff,
    ]) {
        assert_eq!(&slot[..32], binding.sha256());
        assert_eq!(&slot[32..], &binding.byte_len().to_le_bytes());
    }
    let mut hash = Sha256::new();
    hash.update(b"FE2O3/INERT-COMPILER-EXECUTION-SUBJECT/V3\0");
    hash.update(658_u64.to_le_bytes());
    hash.update(&bytes[..658]);
    assert_eq!(&bytes[658..], &hash.finalize()[..]);
    assert_eq!(subject.identity().byte_len(), 690);

    let v1 = V1::from_fields(fields()).unwrap();
    let v2 = v2_encoded();
    assert_eq!(&v1.canonical_bytes()[10..658], &bytes[10..658]);
    assert_eq!(&v2.canonical_bytes[10..658], &bytes[10..658]);
    // Existing independent V1/V2 golden identities stay pinned, not regenerated.
    let hex = |text: &str| -> [u8; 32] {
        std::array::from_fn(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
    };
    assert_eq!(
        v1.identity().sha256(),
        &hex("b256d787309ea6c161d5161920fbd05fd4db0f8f5f105a0bb50825ab1ebc6485")
    );
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(v1.canonical_bytes())),
        hex("211b14d897e8b21d8234962192b0f70e0bdd379adce8c0999db3aac8a30c112e")
    );
    assert_eq!(
        v2.sha256,
        hex("e06b5af4ce60eec43e5427cd9a3551b94a7fb2dc20dbf23705f9b1c6f51d7039")
    );
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(v2.canonical_bytes)),
        hex("17f6e7f89357ebae6e652bd76327b159b9653327c4aa37134903075248e7d064")
    );
    inert(&subject);
}

#[test]
fn conditional_subject_v3_roundtrip_and_bidirectional_version_separation() {
    let original = Subject::from_fields(fields()).unwrap();
    assert_eq!(decode(original.canonical_bytes()).unwrap().0, original);
    assert!(V1::decode(original.canonical_bytes()).is_err());
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, LIMIT);
    budget.reserve_storage(3 * 690 + 2 * RETAINED).unwrap();
    assert!(V2::decode(original.canonical_bytes(), &mut budget).is_err());
    let legacy = V1::from_fields(fields()).unwrap();
    let v2_wire = v2_encoded().canonical_bytes;
    let (v2, charge) = V2::decode(&v2_wire, &mut budget).unwrap();
    budget.reserve_storage(charge.retained_storage()).unwrap();
    assert_eq!(v2.canonical_bytes(), &v2_wire);
    assert!(decode(legacy.canonical_bytes()).is_err());
    assert!(decode(&v2_wire).is_err());
    assert!(
        original
            .identity()
            .matches_canonical_bytes(original.canonical_bytes(), &mut budget)
            .unwrap()
    );
    for bytes in [legacy.canonical_bytes(), &v2_wire] {
        assert!(
            !original
                .identity()
                .matches_canonical_bytes(bytes, &mut budget)
                .unwrap()
        );
    }
    for (magic, version) in [
        (*b"F2O3CES1", 3u16),
        (*b"F2O3CES2", 3),
        (*b"F2O3CES3", 1),
        (*b"F2O3CES3", 2),
    ] {
        let mut changed = wire();
        changed[..8].copy_from_slice(&magic);
        changed[8..10].copy_from_slice(&version.to_le_bytes());
        reseal(&mut changed);
        assert!(decode(&changed).is_err());
        assert!(V1::decode(&changed).is_err());
        assert!(V2::decode(&changed, &mut budget).is_err());
    }
}

#[test]
fn conditional_subject_v3_corruption_and_resealed_malformed_fields_refuse() {
    for offset in 0..690 {
        let mut changed = wire();
        changed[offset] ^= 1;
        assert!(decode(&changed).is_err(), "byte {offset}");
    }
    for offset in [10, 12, 20, 24, 80, 81, 344, 346] {
        let mut changed = wire();
        changed[offset] ^= 1;
        if offset == 24 {
            changed[24..32].fill(0);
        }
        reseal(&mut changed);
        assert!(decode(&changed).is_err(), "resealed byte {offset}");
    }
    for offset in [88, 120, 152, 184, 216, 248, 280, 312, 658] {
        let mut changed = wire();
        changed[offset..offset + 32].fill(0);
        if offset != 658 {
            reseal(&mut changed);
        }
        assert!(decode(&changed).is_err(), "zero digest {offset}");
    }
    for i in 0..7 {
        for range in [378 + 40 * i..410 + 40 * i, 410 + 40 * i..418 + 40 * i] {
            let mut changed = wire();
            changed[range].fill(0);
            reseal(&mut changed);
            assert!(decode(&changed).is_err());
        }
    }
    for len in 0..690 {
        assert!(decode(&wire()[..len]).is_err());
    }
    assert!(decode(&[0; 691]).is_err());
}

#[test]
fn conditional_subject_v3_resealed_alternative_is_inert_not_authenticated() {
    let original = Subject::from_fields(fields()).unwrap();
    for offset in [24, 32, 48, 88, 120, 378, 418, 458, 498, 538, 578, 618] {
        let mut changed = wire();
        changed[offset] ^= 1;
        reseal(&mut changed);
        let (alternative, _) = decode(&changed).unwrap();
        assert_ne!(alternative, original);
        inert(&alternative);
    }
    for i in 0..7 {
        let mut changed = wire();
        changed[410 + 40 * i..418 + 40 * i].fill(0xff);
        reseal(&mut changed);
        let (alternative, _) = decode(&changed).unwrap();
        assert_ne!(alternative.identity(), original.identity());
        inert(&alternative);
    }
}

#[test]
fn conditional_subject_v3_typed_axes_preserve_receipt_domains_on_both_profiles() {
    for gfx950 in [false, true] {
        let handoff = outer_variant(gfx950, 7);
        let capsule = handoff.capsule();
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget
            .reserve_storage(handoff_floor(&handoff).unwrap())
            .unwrap();
        let (subject, storage) = replay(&handoff, &mut budget).unwrap();
        budget.reserve_storage(storage.0).unwrap();
        assert_eq!(
            subject.rustc_invocation_sha256(),
            capsule.invocation_digest().as_bytes()
        );
        assert_eq!(
            subject.compiler_closure(),
            *capsule.invocation().compiler_closure()
        );
        assert_eq!(subject.slot(), CompilerModuleHandoffSlotV5::Production);
        assert_eq!(subject.attempt(), fields().attempt);
        assert_eq!(subject.transaction_identity().as_bytes(), &[7; 32]);
        for (observed, digest, length) in [
            (
                subject.rustc_identity_inventory(),
                *capsule.rustc_identity_inventory().identity().sha256(),
                capsule.rustc_identity_inventory().identity().byte_len(),
            ),
            (
                subject.rustc_preflight_plan(),
                *capsule.rustc_preflight_plan().identity().sha256(),
                capsule.rustc_preflight_plan().identity().byte_len(),
            ),
            (
                subject.semantic_capsule(),
                *capsule.identity().sha256(),
                capsule.identity().byte_len(),
            ),
            (
                subject.compiler_module_handoff(),
                *handoff.module_handoff().identity().sha256(),
                handoff.module_handoff().identity().byte_len(),
            ),
            (
                subject.compiler_module_pair_binding(),
                *handoff.pair_binding_identity().sha256(),
                handoff.pair_binding_identity().byte_len(),
            ),
            (
                subject.outer_handoff(),
                *handoff.identity().sha256(),
                handoff.identity().byte_len(),
            ),
        ] {
            assert_eq!(observed.sha256(), &digest);
            assert_eq!(observed.byte_len(), length);
        }
        // Independently recompute the lineage RECEIPT domain, not the FFI trailer.
        let commitment = capsule.final_module_commitment_bytes();
        let mut hash = Sha256::new();
        hash.update(b"FE2O3/INERT-LINEAGE-CONTENT/FINAL-COMPILER-MODULE-COMMITMENT/V3\0");
        hash.update((commitment.len() as u64).to_le_bytes());
        hash.update(commitment);
        assert_eq!(
            subject.final_compiler_module_commitment().sha256(),
            &<[u8; 32]>::from(hash.finalize())
        );
        assert_eq!(
            subject.final_compiler_module_commitment().byte_len(),
            commitment.len() as u64
        );
        assert_ne!(
            &subject.final_compiler_module_commitment().sha256()[..],
            &commitment[commitment.len() - 32..]
        );
        assert_eq!(decode(subject.canonical_bytes()).unwrap().0, subject);
        inert(&subject);
    }
}

#[test]
fn conditional_subject_v3_real_publication_consumption_and_donor_refusal() {
    let f = Fixture::new();
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, LIMIT);
    f.reserve(&mut budget);
    let receipt = f.publish(&mut budget).unwrap();
    let (expected, storage) = Subject::from_publication(receipt, &f.handoff, &mut budget).unwrap();
    budget.reserve_storage(storage.0).unwrap();
    assert_eq!(expected.attempt(), receipt.attempt());
    assert_eq!(expected.slot(), receipt.slot());
    assert_eq!(
        expected.transaction_identity(),
        receipt.transaction_identity()
    );
    for alternative in [outer_variant(false, 19), outer_variant(true, 7)] {
        let paid = handoff_floor(&alternative).unwrap();
        budget.reserve_storage(paid).unwrap();
        let before = (budget.storage(), budget.work());
        assert!(matches!(
            Subject::from_publication(receipt, &alternative, &mut budget),
            Err(Failure::HandoffIdentityMismatch)
        ));
        assert_eq!(budget.storage(), before.0);
        assert_eq!(budget.work(), before.1 + WORK);
        let (different, charge) = replay(&alternative, &mut budget).unwrap();
        budget.reserve_storage(charge.0).unwrap();
        assert_ne!(different.outer_handoff(), expected.outer_handoff());
        inert(&different);
        drop(different);
        drop(alternative);
        budget.release_storage(paid + charge.0).unwrap();
    }
    let lease = f.lease(receipt, &mut budget);
    let (token, storage) = lease.acquire_current_token(&mut budget).unwrap();
    budget.reserve_storage(storage.retained_storage()).unwrap();
    let consumed =
        crate::consume_compiler_module_handoff_with_currentness_v5(&lease, token, &mut budget)
            .unwrap();
    let floor = budget.storage();
    let ledger = budget.work_ledger_identity_v1();
    let (actual, storage) = Subject::from_consumed(&consumed, &mut budget).unwrap();
    assert_eq!(budget.storage(), floor);
    assert!(budget.work_ledger_identity_v1() == ledger);
    budget.reserve_storage(storage.0).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.attempt(), consumed.receipt().attempt());
    assert_eq!(actual.slot(), consumed.receipt().slot());
    assert_eq!(
        actual.transaction_identity(),
        consumed.receipt().transaction_identity()
    );
    inert(&actual);
}

#[test]
fn original_publication_quote_covers_exact_and_one_short_resources() {
    use fe2o3_kernel_ir::CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned;
    let fixture = Fixture::new();
    let mut setup_work = Work::new(usize::MAX);
    let mut setup = Budget::new(&mut setup_work, LIMIT);
    fixture.reserve(&mut setup);
    let receipt = fixture.publish(&mut setup).unwrap();
    let inputs =
        handoff_floor(&fixture.handoff).unwrap() + size_of::<CompilerModuleHandoffReceiptV5>();
    let scratch = Subject::composed_publication_storage_v3(&fixture.handoff).unwrap();
    let outside = LIMIT + 17;
    for (work_short, storage_short) in [(false, false), (true, false), (false, true)] {
        let total = outside + inputs + scratch;
        let mut owned = Owned::new(
            Work::new(Subject::COMPOSED_PUBLICATION_WORK_V3 - usize::from(work_short)),
            total - usize::from(storage_short),
        );
        owned.with_budget(|budget| {
            budget.reserve_storage(outside + inputs).unwrap();
            let identity = budget.storage_account_identity_v1();
            let ledger = budget.work_ledger_identity_v1();
            let result =
                Subject::from_publication_in_original_account_v3(receipt, &fixture.handoff, budget);
            assert_eq!(result.is_ok(), !work_short && !storage_short);
            if let Ok((subject, _)) = result {
                assert_eq!(subject.attempt(), receipt.attempt());
                assert_eq!(budget.peak_storage(), total);
            }
            assert_eq!(budget.storage(), outside + inputs);
            assert_eq!(budget.storage_account_identity_v1(), identity);
            assert!(budget.work_ledger_identity_v1() == ledger);
        });
    }
}

#[test]
fn conditional_subject_v3_replay_occurrence_and_carrier_are_not_omitted() {
    let a = outer();
    let b = outer_variant(false, 19);
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, LIMIT);
    budget
        .reserve_storage(handoff_floor(&a).unwrap() + handoff_floor(&b).unwrap())
        .unwrap();
    let (expected, storage) = replay(&a, &mut budget).unwrap();
    budget.reserve_storage(storage.0).unwrap();
    let (changed, storage) = replay(&b, &mut budget).unwrap();
    budget.reserve_storage(storage.0).unwrap();
    assert_eq!(a.module_handoff().identity(), b.module_handoff().identity());
    assert_eq!(
        expected.final_compiler_module_commitment(),
        changed.final_compiler_module_commitment()
    );
    assert_ne!(expected.semantic_capsule(), changed.semantic_capsule());
    assert_ne!(
        expected.compiler_module_pair_binding(),
        changed.compiler_module_pair_binding()
    );
    assert_ne!(expected.outer_handoff(), changed.outer_handoff());
    for (attempt, transaction) in [
        (
            crate::BuildAttempt::new(
                fields().attempt.generation() + 1,
                fields().attempt.session(),
                fields().attempt.invocation(),
            )
            .unwrap(),
            [7; 32],
        ),
        (fields().attempt, [8; 32]),
    ] {
        let (changed, storage) = Subject::from_replay_evidence(
            attempt,
            CompilerModuleHandoffSlotV5::Production,
            CompilerModuleHandoffTransactionIdentityV5::from_bytes(transaction),
            &a,
            &mut budget,
        )
        .unwrap();
        budget.reserve_storage(storage.0).unwrap();
        assert_eq!(expected.outer_handoff(), changed.outer_handoff());
        assert_ne!(expected.identity(), changed.identity());
        inert(&changed);
        drop(changed);
        budget.release_storage(storage.0).unwrap();
    }
    assert!(matches!(
        Subject::from_replay_evidence(
            fields().attempt,
            CompilerModuleHandoffSlotV5::Production,
            CompilerModuleHandoffTransactionIdentityV5::from_bytes([0; 32]),
            &a,
            &mut budget
        ),
        Err(Failure::Framing(
            CompilerExecutionSubjectErrorV1::ZeroIdentity {
                field: "V5 handoff transaction"
            }
        ))
    ));
}

#[path = "compiler_execution_subject_v3_resources_tests.rs"]
mod resources;

#[path = "compiler_execution_subject_v3_occurrence_tests.rs"]
mod occurrence;

#[path = "compiler_execution_subject_v3_replay_original_account_tests.rs"]
mod replay_original_account;

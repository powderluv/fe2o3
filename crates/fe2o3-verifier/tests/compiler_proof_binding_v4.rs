use ed25519_dalek::{Signer as _, SigningKey};
use fe2o3_compiler_lineage::{
    InertCanonicalSemanticMirReceiptV3, InertFormalMemoryReceiptV3, InertKernelIrReceiptV3,
    InertLineageContentIdentityV3, InertMiddleEndReceiptV3, InertMirToKirCorrespondenceReceiptV3,
    InertProofBindingAssociationInputsV4, InertProofBindingAssociationV4,
    InertProofBindingReceiptV3,
};
use fe2o3_functional_proof::{
    FunctionalRefinementBindingV2, FunctionalRefinementBoundaryV2,
    FunctionalRefinementImportExpectationV2, FunctionalRefinementImportPolicyV2,
    FunctionalRefinementReceiptImporterV2, FunctionalRefinementResultV2, SafeReferenceKindV2,
    UnsignedFunctionalRefinementReceiptV2, VerusToolchainIdentityV2,
};
use fe2o3_pliron::InertProductionMiddleEndEvidenceV5;
use fe2o3_proof_contracts::DigestV1;
use fe2o3_verifier::{
    CanonicalProductionMirPlironVerusExecutionEvidenceV1, CompilerProofInputValidationErrorV3,
    CompilerProofInputValidationErrorV4, ProductionMirPlironVerusExecutionClaimsV1,
    ValidatedCompilerProofInputsV4, validate_compiler_proof_inputs_v4,
};

#[path = "../../../tests/support/compiler_proof_inputs_v3.rs"]
mod compiler_proof_inputs_v3;
use compiler_proof_inputs_v3::{
    CanonicalCompilerProofInputsV3, canonical_compiler_proof_inputs_v4,
    canonical_compiler_proof_inputs_v4_with_induction,
};

struct Receipts {
    semantic_mir: InertCanonicalSemanticMirReceiptV3,
    middle_end: InertMiddleEndReceiptV3,
    kernel_ir: InertKernelIrReceiptV3,
    correspondence: InertMirToKirCorrespondenceReceiptV3,
    formal_memory: InertFormalMemoryReceiptV3,
}

const CORRESPONDENCE_HEADER_BYTES_V4: usize = 124;
const BLOCK_RECORD_BYTES_V4: usize = 16;
const STATEMENT_RECORD_BYTES_V4: usize = 24;
const TERMINATOR_RECORD_BYTES_V4: usize = 20;
const SYNTHETIC_RECORD_BYTES_V4: usize = 16;
const PARAMETER_RECORD_BYTES_V4: usize = 12;

#[derive(Clone, Copy)]
struct CorrespondenceSectionsV4 {
    statement_start: usize,
    statement_count: usize,
    synthetic_start: usize,
    synthetic_count: usize,
    parameter_start: usize,
    parameter_count: usize,
    induction_start: usize,
}

fn receipts(seed: u8) -> Receipts {
    receipts_from(canonical_compiler_proof_inputs_v4(seed))
}

fn induction_receipts(seed: u8) -> Receipts {
    receipts_from(canonical_compiler_proof_inputs_v4_with_induction(seed))
}

fn receipts_from(inputs: CanonicalCompilerProofInputsV3) -> Receipts {
    Receipts {
        semantic_mir: InertCanonicalSemanticMirReceiptV3::from_canonical_preimage(
            inputs.semantic_mir().to_vec(),
        )
        .unwrap(),
        middle_end: InertMiddleEndReceiptV3::from_canonical_preimage(inputs.middle_end().to_vec())
            .unwrap(),
        kernel_ir: InertKernelIrReceiptV3::from_canonical_preimage(inputs.kernel_ir().to_vec())
            .unwrap(),
        correspondence: InertMirToKirCorrespondenceReceiptV3::from_canonical_preimage(
            inputs.correspondence().to_vec(),
        )
        .unwrap(),
        formal_memory: InertFormalMemoryReceiptV3::from_canonical_preimage(
            inputs.formal_memory().to_vec(),
        )
        .unwrap(),
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn correspondence_sections(bytes: &[u8]) -> CorrespondenceSectionsV4 {
    assert!(bytes.len() >= CORRESPONDENCE_HEADER_BYTES_V4);
    let block_count = u32_at(bytes, 100) as usize;
    let statement_count = u32_at(bytes, 104) as usize;
    let terminator_count = u32_at(bytes, 108) as usize;
    let synthetic_count = u32_at(bytes, 112) as usize;
    let parameter_count = u32_at(bytes, 116) as usize;
    let statement_start = CORRESPONDENCE_HEADER_BYTES_V4 + block_count * BLOCK_RECORD_BYTES_V4;
    let terminator_start = statement_start + statement_count * STATEMENT_RECORD_BYTES_V4;
    let synthetic_start = terminator_start + terminator_count * TERMINATOR_RECORD_BYTES_V4;
    let parameter_start = synthetic_start + synthetic_count * SYNTHETIC_RECORD_BYTES_V4;
    let induction_start = parameter_start + parameter_count * PARAMETER_RECORD_BYTES_V4;
    assert!(induction_start < bytes.len());
    CorrespondenceSectionsV4 {
        statement_start,
        statement_count,
        synthetic_start,
        synthetic_count,
        parameter_start,
        parameter_count,
        induction_start,
    }
}

fn statement_record_offset(
    bytes: &[u8],
    sections: CorrespondenceSectionsV4,
    function: u32,
    block: u32,
    statement: u32,
) -> usize {
    (0..sections.statement_count)
        .map(|index| sections.statement_start + index * STATEMENT_RECORD_BYTES_V4)
        .find(|offset| {
            u32_at(bytes, *offset) == function
                && u32_at(bytes, *offset + 4) == block
                && u32_at(bytes, *offset + 8) == statement
        })
        .expect("fixture contains the requested statement span")
}

fn replace_correspondence(receipts: &mut Receipts, canonical_bytes: Vec<u8>) {
    receipts.correspondence =
        InertMirToKirCorrespondenceReceiptV3::from_canonical_preimage(canonical_bytes).unwrap();
}

fn replace_formal_memory(receipts: &mut Receipts, canonical_bytes: Vec<u8>) {
    receipts.formal_memory =
        InertFormalMemoryReceiptV3::from_canonical_preimage(canonical_bytes).unwrap();
}

fn assert_structural_failure(receipts: &Receipts, expected: &'static str) {
    let evidence = signed_verus_evidence(exact_pliron_identity(receipts));
    let proof_binding = proof_binding(receipts, None, evidence.canonical_bytes());
    assert!(matches!(
        validate(&proof_binding, receipts),
        Err(CompilerProofInputValidationErrorV4::Stage(
            CompilerProofInputValidationErrorV3::StructuralCorrespondence { detail }
        )) if detail == expected
    ));
}

fn digest(seed: u8) -> DigestV1 {
    DigestV1::from_untrusted_bytes([seed; 32])
}

fn identity(sha256: &[u8; 32], byte_len: u64) -> InertLineageContentIdentityV3 {
    InertLineageContentIdentityV3::new(*sha256, byte_len).unwrap()
}

fn stage_identities(receipts: &Receipts) -> [InertLineageContentIdentityV3; 5] {
    [
        identity(
            receipts.semantic_mir.identity().sha256(),
            receipts.semantic_mir.identity().byte_len(),
        ),
        identity(
            receipts.middle_end.identity().sha256(),
            receipts.middle_end.identity().byte_len(),
        ),
        identity(
            receipts.kernel_ir.identity().sha256(),
            receipts.kernel_ir.identity().byte_len(),
        ),
        identity(
            receipts.correspondence.identity().sha256(),
            receipts.correspondence.identity().byte_len(),
        ),
        identity(
            receipts.formal_memory.identity().sha256(),
            receipts.formal_memory.identity().byte_len(),
        ),
    ]
}

fn signed_verus_evidence(
    pliron_identity: DigestV1,
) -> CanonicalProductionMirPlironVerusExecutionEvidenceV1 {
    let binding = FunctionalRefinementBindingV2::new(
        SafeReferenceKindV2::SourceAndMir,
        digest(10),
        digest(11),
        digest(12),
        digest(13),
        digest(14),
        digest(15),
    )
    .unwrap();
    let toolchain =
        VerusToolchainIdentityV2::new(digest(20), digest(21), digest(22), digest(23), digest(24))
            .unwrap();
    let signing = SigningKey::from_bytes(&[42; 32]);
    let verifying_key = signing.verifying_key().to_bytes();
    let policy = FunctionalRefinementImportPolicyV2::new(
        verifying_key,
        toolchain,
        FunctionalRefinementBoundaryV2::SafeReferenceMirToLivePliron,
    )
    .unwrap();
    let unsigned = UnsignedFunctionalRefinementReceiptV2::from_verified_execution_join(
        policy.signer_identity(),
        binding,
        toolchain,
        digest(30),
        FunctionalRefinementResultV2::Proved,
        FunctionalRefinementBoundaryV2::SafeReferenceMirToLivePliron,
    )
    .unwrap();
    let signature = signing.sign(unsigned.signing_bytes()).to_bytes();
    let wire = unsigned.attach_signature(signature);
    let mut importer = FunctionalRefinementReceiptImporterV2::new(policy, 1).unwrap();
    let imported = importer
        .import(FunctionalRefinementImportExpectationV2::new(binding), &wire)
        .unwrap();
    let claims = ProductionMirPlironVerusExecutionClaimsV1::new(
        digest(1),
        digest(2),
        pliron_identity,
        digest(4),
        digest(5),
        binding,
        imported.signer_identity(),
        toolchain,
        imported.execution_identity(),
        imported.receipt_identity().digest(),
        3,
    )
    .unwrap();
    CanonicalProductionMirPlironVerusExecutionEvidenceV1::new(claims, verifying_key, wire).unwrap()
}

fn exact_pliron_identity(receipts: &Receipts) -> DigestV1 {
    let middle_end =
        InertProductionMiddleEndEvidenceV5::decode(receipts.middle_end.canonical_preimage())
            .unwrap();
    DigestV1::from_untrusted_bytes(*middle_end.identity().sha256())
}

fn proof_binding(
    receipts: &Receipts,
    substitute: Option<usize>,
    evidence: &[u8],
) -> InertProofBindingReceiptV3 {
    let mut identities = stage_identities(receipts);
    if let Some(index) = substitute {
        identities[index] =
            InertLineageContentIdentityV3::new([0xa0 + index as u8; 32], 99).unwrap();
    }
    let association = InertProofBindingAssociationV4::new(
        InertProofBindingAssociationInputsV4::new(
            identities[0],
            identities[1],
            identities[2],
            identities[3],
            identities[4],
        ),
        evidence,
    )
    .unwrap();
    InertProofBindingReceiptV3::from_canonical_preimage(association.canonical_bytes()).unwrap()
}

fn validate(
    proof_binding: &InertProofBindingReceiptV3,
    receipts: &Receipts,
) -> Result<ValidatedCompilerProofInputsV4, CompilerProofInputValidationErrorV4> {
    validate_compiler_proof_inputs_v4(
        proof_binding,
        &receipts.semantic_mir,
        &receipts.middle_end,
        &receipts.kernel_ir,
        &receipts.correspondence,
        &receipts.formal_memory,
    )
}

#[test]
fn exact_current_inputs_reimport_the_signed_verus_receipt() {
    let receipts = receipts(0);
    let evidence = signed_verus_evidence(exact_pliron_identity(&receipts));
    let proof_binding = proof_binding(&receipts, None, evidence.canonical_bytes());
    let validated = validate(&proof_binding, &receipts).unwrap();

    assert_eq!(validated.receipt_identity(), proof_binding.identity());
    assert_eq!(
        validated.association().verus_execution_evidence(),
        evidence.canonical_bytes()
    );
    assert_eq!(
        validated.verus_execution().canonical_bytes(),
        evidence.canonical_bytes()
    );
    assert_eq!(
        validated.semantic_mir().canonical_encoding(),
        receipts.semantic_mir.canonical_preimage()
    );
    assert_eq!(
        validated.middle_end().canonical_bytes(),
        receipts.middle_end.canonical_preimage()
    );
    assert_eq!(
        validated.kernel_ir().canonical_bytes(),
        receipts.kernel_ir.canonical_preimage()
    );
    assert!(validated.has_exact_decoded_input_association());
    assert!(validated.has_lossless_mir_to_kir_correspondence());
    assert!(validated.semantic_u32_induction_kir_anchors().is_empty());
    assert!(validated.authenticates_signed_verus_receipt_under_embedded_key());
    assert!(!validated.authenticates_compiler_origin());
    assert!(!validated.establishes_llvm_or_machine_refinement());
    assert!(!validated.grants_runtime_authority());
}

#[test]
fn exact_induction_certificate_is_anchored_to_one_checked_kir_addition() {
    let receipts = induction_receipts(0);
    let evidence = signed_verus_evidence(exact_pliron_identity(&receipts));
    let proof_binding = proof_binding(&receipts, None, evidence.canonical_bytes());
    let validated = validate(&proof_binding, &receipts).unwrap();

    let [anchor] = validated.semantic_u32_induction_kir_anchors() else {
        panic!("the exact induction fixture must retain one checked-add anchor");
    };
    assert_eq!(anchor.semantic_function(), 0);
    assert_eq!(anchor.semantic_block(), 2);
    assert_eq!(anchor.semantic_statement(), 1);
    assert_eq!(anchor.kernel_ir_block(), 2);
    assert_ne!(anchor.value_result(), anchor.overflow_result());
    assert!(validated.has_lossless_mir_to_kir_correspondence());
    assert!(!validated.establishes_llvm_or_machine_refinement());
}

#[test]
fn checked_addition_cannot_be_reassigned_to_an_adjacent_nop_span() {
    let mut receipts = induction_receipts(0);
    let mut bytes = receipts.correspondence.canonical_preimage().to_vec();
    let sections = correspondence_sections(&bytes);
    let nop = statement_record_offset(&bytes, sections, 0, 2, 0);
    let checked = statement_record_offset(&bytes, sections, 0, 2, 1);
    let checked_first = u32_at(&bytes, checked + 16);
    let checked_count = u32_at(&bytes, checked + 20);
    assert_ne!(checked_count, 0);
    put_u32(&mut bytes, nop + 16, checked_first);
    put_u32(&mut bytes, nop + 20, checked_count);
    put_u32(
        &mut bytes,
        checked + 16,
        checked_first.checked_add(checked_count).unwrap(),
    );
    put_u32(&mut bytes, checked + 20, 0);
    replace_correspondence(&mut receipts, bytes);

    assert_structural_failure(
        &receipts,
        "induction statement span contains no checked KIR addition",
    );
}

#[test]
fn retained_induction_report_must_equal_deterministic_replay() {
    let mut receipts = induction_receipts(0);
    let mut bytes = receipts.correspondence.canonical_preimage().to_vec();
    let sections = correspondence_sections(&bytes);
    let work_units = sections.induction_start + 92;
    let mutated_work_units = u64_at(&bytes, work_units).checked_add(1).unwrap();
    put_u64(&mut bytes, work_units, mutated_work_units);
    replace_correspondence(&mut receipts, bytes);

    assert_structural_failure(
        &receipts,
        "retained semantic induction report differs from deterministic replay",
    );
}

#[test]
fn semantic_argument_cannot_be_rebound_to_another_kir_value() {
    let mut receipts = induction_receipts(0);
    let mut bytes = receipts.correspondence.canonical_preimage().to_vec();
    let sections = correspondence_sections(&bytes);
    assert_ne!(sections.parameter_count, 0);
    let value = sections.parameter_start + 8;
    let mutated_value = u32_at(&bytes, value).checked_add(1).unwrap();
    put_u32(&mut bytes, value, mutated_value);
    replace_correspondence(&mut receipts, bytes);

    assert_structural_failure(
        &receipts,
        "semantic argument binding names a different KIR parameter",
    );
}

#[test]
fn runtime_assert_trap_requires_exact_synthetic_custody() {
    let mut receipts = induction_receipts(0);
    let mut bytes = receipts.correspondence.canonical_preimage().to_vec();
    let sections = correspondence_sections(&bytes);
    assert_ne!(sections.synthetic_count, 0);
    let trap = (0..sections.synthetic_count)
        .map(|index| sections.synthetic_start + index * SYNTHETIC_RECORD_BYTES_V4)
        .find(|offset| u32_at(&bytes, *offset) == 2)
        .expect("fixture contains the runtime-assert trap span");
    put_u32(&mut bytes, trap + 12, 0);
    replace_correspondence(&mut receipts, bytes);

    assert_structural_failure(
        &receipts,
        "unmapped KIR block is not the canonical runtime-assert trap",
    );
}

#[test]
fn independently_well_formed_kir_custody_substitutions_fail_closed() {
    for mutate_correspondence in [true, false] {
        let mut receipts = induction_receipts(0);
        if mutate_correspondence {
            let mut bytes = receipts.correspondence.canonical_preimage().to_vec();
            bytes[64] ^= 1;
            replace_correspondence(&mut receipts, bytes);
        } else {
            let mut bytes = receipts.formal_memory.canonical_preimage().to_vec();
            bytes[32] ^= 1;
            replace_formal_memory(&mut receipts, bytes);
        }
        let evidence = signed_verus_evidence(exact_pliron_identity(&receipts));
        let proof_binding = proof_binding(&receipts, None, evidence.canonical_bytes());
        assert!(matches!(
            validate(&proof_binding, &receipts),
            Err(CompilerProofInputValidationErrorV4::Stage(
                CompilerProofInputValidationErrorV3::NestedIdentityMismatch {
                    field: "current production Kernel IR custody"
                }
            ))
        ));
    }
}

#[test]
fn every_outer_stage_identity_substitution_fails_closed() {
    let fields = [
        "semantic MIR",
        "middle end",
        "Kernel IR",
        "MIR-to-KIR correspondence",
        "formal memory",
    ];
    for (index, field) in fields.into_iter().enumerate() {
        let receipts = receipts(0);
        let evidence = signed_verus_evidence(exact_pliron_identity(&receipts));
        let proof_binding = proof_binding(&receipts, Some(index), evidence.canonical_bytes());
        assert!(matches!(
            validate(&proof_binding, &receipts),
            Err(CompilerProofInputValidationErrorV4::ProofBindingIdentityMismatch {
                field: actual
            }) if actual == field
        ));
    }
}

#[test]
fn malformed_nested_verus_evidence_fails_at_the_signed_codec() {
    let receipts = receipts(0);
    let evidence = signed_verus_evidence(exact_pliron_identity(&receipts));
    let mut malformed = evidence.canonical_bytes().to_vec();
    malformed[0] ^= 0xff;
    let proof_binding = proof_binding(&receipts, None, &malformed);
    assert!(matches!(
        validate(&proof_binding, &receipts),
        Err(CompilerProofInputValidationErrorV4::VerusEvidence(_))
    ));
}

#[test]
fn signed_receipt_for_a_different_middle_end_fails_closed() {
    let current = receipts(0);
    let other = receipts(1);
    let evidence = signed_verus_evidence(exact_pliron_identity(&other));
    let proof_binding = proof_binding(&current, None, evidence.canonical_bytes());
    assert!(matches!(
        validate(&proof_binding, &current),
        Err(CompilerProofInputValidationErrorV4::VerusMiddleEndMismatch)
    ));
}

#[test]
fn malformed_shared_stage_still_fails_through_the_single_stage_decoder() {
    let mut receipts = receipts(0);
    receipts.middle_end =
        InertMiddleEndReceiptV3::from_canonical_preimage(b"bad middle".to_vec()).unwrap();
    let evidence = signed_verus_evidence(digest(99));
    let proof_binding = proof_binding(&receipts, None, evidence.canonical_bytes());
    assert!(matches!(
        validate(&proof_binding, &receipts),
        Err(CompilerProofInputValidationErrorV4::Stage(_))
    ));
}

//! Inert codec/binding checks, never an enrollment owner, source request or proof.
use super::{fixtures::*, *};
use crate::portable_reference_v1::codec::{
    NativeCpuPolicyAssociationV2, ReferenceEnrollmentOriginV1,
    with_decoded_native_cpu_policy_input_v2,
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

fn policy_input(fixture: &Fixture) -> NativeCpuPolicyInputV2<'_> {
    let input = fixture.input();
    NativeCpuPolicyInputV2 {
        association: NativeCpuPolicyAssociationV2 {
            semantic_mir_sha256: input.association.semantic_mir_sha256,
            semantic_root: input.association.semantic_root,
            logical_kernel_name: input.association.logical_kernel_name,
            origin: ReferenceEnrollmentOriginV1 {
                rustc_invocation_sha256: [71; 32],
                native_policy_sha256: [72; 32],
                policy_generation: 3,
                mapping_ordinal: 4,
            },
        },
        kernel: input.kernel,
        reference: input.reference,
        replay: input.replay,
    }
}

fn commitment(input: NativeCpuPolicyInputV2<'_>) -> DigestV1 {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    let account = budget.work_ledger_identity_v1();
    let expected = input.association.origin;
    let hash = with_encoded_native_cpu_policy_input_v2(
        reborrow_policy(&input),
        &mut budget,
        |bytes, hash, budget| {
            with_decoded_native_cpu_policy_input_v2(bytes, budget, |decoded, _| {
                assert_eq!(decoded.commitment_v2(), hash);
                assert_eq!(decoded.input_v2().association.origin, expected);
                DigestV1::from_untrusted_bytes(hash)
            })
            .unwrap()
        },
    )
    .unwrap();
    assert_eq!(budget.storage(), 31);
    assert!(budget.work_ledger_identity_v1() == account);
    hash
}

fn rejects(expected: DigestV1, actual: DigestV1) {
    assert!(matches!(
        require_commitment(expected, actual),
        Err(Error::Subject("conditional V2 CPU input substitution"))
    ));
    let six = std::array::from_fn(|i| digest(i as u8 + 1));
    assert_ne!(
        obligation_identity_v2(six, expected),
        obligation_identity_v2(six, actual)
    );
}

#[test]
fn policy_reborrow_preserves_exact_origin_and_all_borrowed_subjects() {
    let fixture = Fixture::new();
    let input = policy_input(&fixture);
    let borrowed = reborrow_policy(&input);
    assert_eq!(borrowed.association.origin, input.association.origin);
    assert_eq!(
        borrowed.association.semantic_mir_sha256,
        input.association.semantic_mir_sha256
    );
    assert_eq!(
        borrowed.association.semantic_root,
        input.association.semantic_root
    );
    assert!(std::ptr::eq(
        borrowed.association.logical_kernel_name,
        input.association.logical_kernel_name
    ));
    assert!(std::ptr::eq(borrowed.kernel, input.kernel));
    assert!(std::ptr::eq(borrowed.reference, input.reference));
    assert!(std::ptr::eq(
        borrowed.replay.signature_preimage,
        input.replay.signature_preimage
    ));
    assert!(std::ptr::eq(
        borrowed.replay.effect_ir,
        input.replay.effect_ir
    ));
    assert!(std::ptr::eq(
        borrowed.replay.observable_output_writes,
        input.replay.observable_output_writes
    ));
    assert_eq!(
        borrowed.replay.effect_ir_sha256,
        input.replay.effect_ir_sha256
    );
    assert_eq!(commitment(borrowed), commitment(input));
}

fn assert_subject(projection: CpuCorrespondenceInput<'_>, fixture: &Fixture) {
    let expected = fixture.input();
    assert_eq!(
        projection.semantic_mir_sha256,
        expected.association.semantic_mir_sha256
    );
    assert_eq!(projection.semantic_root, expected.association.semantic_root);
    assert!(std::ptr::eq(projection.reference.kernel, expected.kernel));
    assert!(std::ptr::eq(
        projection.reference.reference,
        expected.reference
    ));
    assert!(std::ptr::eq(
        projection.reference.replay.signature_preimage,
        expected.replay.signature_preimage
    ));
    assert!(std::ptr::eq(
        projection.reference.replay.effect_ir,
        expected.replay.effect_ir
    ));
    assert!(std::ptr::eq(
        projection.reference.replay.observable_output_writes,
        expected.replay.observable_output_writes
    ));
    assert_eq!(
        projection.reference.replay.effect_ir_sha256,
        expected.replay.effect_ir_sha256
    );
}

#[test]
fn policy_projection_preserves_exact_correspondence_subject() {
    let fixture = Fixture::new();
    assert_subject(
        CpuCorrespondenceInput::from(policy_input(&fixture)),
        &fixture,
    );
}

#[test]
fn registration_projection_preserves_exact_correspondence_subject() {
    let fixture = Fixture::new();
    assert_subject(CpuCorrespondenceInput::from(fixture.input()), &fixture);
}

#[test]
fn every_policy_origin_and_association_field_changes_formula_binding() {
    let fixture = Fixture::new();
    let expected = commitment(policy_input(&fixture));
    require_commitment(expected, expected).unwrap();
    for field in 0..7 {
        let mut input = policy_input(&fixture);
        match field {
            0 => input.association.origin.rustc_invocation_sha256[0] ^= 1,
            1 => input.association.origin.native_policy_sha256[0] ^= 1,
            2 => input.association.origin.policy_generation += 1,
            3 => input.association.origin.mapping_ordinal += 1,
            4 => input.association.semantic_mir_sha256[0] ^= 1,
            5 => input.association.semantic_root += 1,
            6 => input.association.logical_kernel_name = "other",
            _ => unreachable!(),
        }
        rejects(expected, commitment(input));
    }
}

#[test]
fn policy_and_registration_commitments_are_not_interchangeable() {
    let fixture = Fixture::new();
    let policy = commitment(policy_input(&fixture));
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    let registration =
        with_encoded_native_cpu_input_v1(fixture.input(), &mut budget, |_, hash, _| {
            DigestV1::from_untrusted_bytes(hash)
        })
        .unwrap();
    rejects(registration, policy);
    rejects(policy, registration);
    assert_eq!(budget.storage(), 0);
}

#[test]
fn policy_commitment_binds_coherent_cpu_operand_changes() {
    let mut fixture = Fixture::new();
    let expected = commitment(policy_input(&fixture));
    fixture.change_operand();
    rejects(expected, commitment(policy_input(&fixture)));
}

#[test]
fn policy_commitment_binds_all_kernel_and_reference_identity_fields() {
    let expected = commitment(policy_input(&Fixture::new()));
    for reference in [false, true] {
        for field in 0..7 {
            let mut fixture = Fixture::new();
            let identity = if reference {
                &mut fixture.reference
            } else {
                &mut fixture.kernel
            };
            match field {
                0 => identity.def_path_hash[0] ^= 1,
                1 => identity.function_sha256[0] ^= 1,
                2 => identity.item_definition_sha256[0] ^= 1,
                3 => identity.monomorphization_sha256[0] ^= 1,
                4 => identity.generic_type_arguments_sha256[0] ^= 1,
                5 => identity.const_generic_arguments_sha256[0] ^= 1,
                6 => identity.rustc_mir_body_sha256[0] ^= 1,
                _ => unreachable!(),
            }
            rejects(expected, commitment(policy_input(&fixture)));
        }
    }
}

#[test]
fn policy_reborrow_does_not_hide_stale_digest_or_distinct_write_claims() {
    let mut fixture = Fixture::new();
    let stale = fixture.ir.canonical_sha256_v1();
    fixture.change_operand();
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    let account = budget.work_ledger_identity_v1();
    let mut input = policy_input(&fixture);
    input.replay.effect_ir_sha256 = stale;
    assert!(
        with_encoded_native_cpu_policy_input_v2(
            reborrow_policy(&input),
            &mut budget,
            |_, _, _| panic!("stale digest exposed"),
        )
        .is_err()
    );
    let mut other = fixture.ir.observable_output_effects.to_vec();
    other[0].rhs = crate::portable_reference_v1::ReferenceEffectExpressionV1::Constant(constant(0));
    let mut input = policy_input(&fixture);
    input.replay.observable_output_writes = &other;
    assert!(
        with_encoded_native_cpu_policy_input_v2(
            reborrow_policy(&input),
            &mut budget,
            |_, _, _| panic!("foreign writes exposed"),
        )
        .is_err()
    );
    assert_eq!(budget.storage(), 31);
    assert!(budget.work_ledger_identity_v1() == account);
}

// Typechecked only. Actual execution requires the genuine external owners.
#[allow(dead_code)]
fn policy_execute_call_shape<'request, 'binding, 'work>(
    runtime: &FunctionalRefinementVerusRuntimeLeaseV1,
    request: &Request<'request>,
    input: NativeCpuPolicyInputV2<'binding>,
    budget: &mut Budget<'work>,
    timeout_seconds: u32,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    crate::execute_and_retain_conditional_ranked_formula_policy_v2(
        runtime,
        request,
        input,
        budget,
        timeout_seconds,
    )
}

#[allow(dead_code)]
fn policy_replay_call_shape<'request, 'binding, 'work>(
    retained: &RetainedProductionConditionalFormulaV2,
    request: &Request<'request>,
    input: NativeCpuPolicyInputV2<'binding>,
    budget: &mut Budget<'work>,
    owned_capture: Box<u8>,
) -> Result<Box<u8>, Error> {
    retained.with_replayed_policy_request_v2(request, input, budget, move |_, _| Ok(owned_capture))
}

#[test]
fn decoded_policy_owner_lends_the_same_subject_and_commitment() {
    let fixture = Fixture::new();
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    let account = budget.work_ledger_identity_v1();
    with_encoded_native_cpu_policy_input_v2(
        policy_input(&fixture),
        &mut budget,
        |bytes, hash, budget| {
            with_decoded_native_cpu_policy_input_v2(bytes, budget, |decoded, budget| {
                let input = decoded.input_v2();
                let subject = CpuCorrespondenceInput::from(decoded.input_v2());
                assert_eq!(
                    subject.semantic_mir_sha256,
                    input.association.semantic_mir_sha256
                );
                assert_eq!(subject.semantic_root, input.association.semantic_root);
                assert!(std::ptr::eq(subject.reference.kernel, input.kernel));
                assert!(std::ptr::eq(subject.reference.reference, input.reference));
                assert!(std::ptr::eq(
                    subject.reference.replay.signature_preimage,
                    input.replay.signature_preimage
                ));
                assert!(std::ptr::eq(
                    subject.reference.replay.effect_ir,
                    input.replay.effect_ir
                ));
                assert!(std::ptr::eq(
                    subject.reference.replay.observable_output_writes,
                    input.replay.observable_output_writes
                ));
                assert_eq!(
                    subject.reference.replay.effect_ir_sha256,
                    input.replay.effect_ir_sha256
                );
                let reencoded = with_encoded_native_cpu_policy_input_v2(
                    reborrow_policy(&input),
                    budget,
                    |_, commitment, _| commitment,
                )
                .unwrap();
                assert_eq!(hash, decoded.commitment_v2());
                assert_eq!(hash, reencoded);
            })
            .unwrap();
        },
    )
    .unwrap();
    assert_eq!(budget.storage(), 31);
    assert!(budget.work_ledger_identity_v1() == account);
}

#[allow(dead_code)]
fn policy_decoded_import_call_shape<'request, 'work>(
    request: &Request<'request>,
    input: &DecodedNativeCpuPolicyInputV2,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'work>,
) -> Result<RetainedProductionConditionalFormulaV2, Error> {
    crate::import_and_retain_conditional_ranked_formula_policy_v2(
        request, input, signature, accepted, budget,
    )
}

#[allow(dead_code)]
fn policy_decoded_replay_call_shape<'request, 'work>(
    retained: &RetainedProductionConditionalFormulaV2,
    request: &Request<'request>,
    input: &DecodedNativeCpuPolicyInputV2,
    budget: &mut Budget<'work>,
    owned_capture: Box<u8>,
) -> Result<Box<u8>, Error> {
    retained.with_replayed_decoded_policy_request_v2(request, input, budget, move |_, _| {
        Ok(owned_capture)
    })
}

#[allow(dead_code)]
fn policy_same_visit_import_check_call_shape<'request, 'work>(
    request: &Request<'request>,
    input: &DecodedNativeCpuPolicyInputV2,
    signature: &InertFunctionalRefinementReceiptSignatureV2,
    accepted: &FunctionalRefinementImportPolicyV2,
    budget: &mut Budget<'work>,
    owned_refusal: Box<u8>,
) -> Result<
    Result<RetainedProductionConditionalFormulaV2, Box<u8>>,
    ConditionalFormulaImportCheckErrorV2,
> {
    import_and_check_conditional_ranked_formula_policy_v2(
        request,
        input,
        signature,
        accepted,
        budget,
        move |_, _| Err(owned_refusal),
    )
}

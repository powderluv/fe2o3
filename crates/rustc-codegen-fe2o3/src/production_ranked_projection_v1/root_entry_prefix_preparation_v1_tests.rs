//! Inert component controls. No fixture here can construct the pipeline loan.
use super::*;
use crate::reference_effect_v1::reference_signature_preimage_v1::*;
use crate::reference_effect_v1::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget, CanonicalKernelIrWorkBudgetV1 as Work,
};
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticExternAbiV1, SemanticFunctionSafetyV1, SemanticMutabilityV1,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn fixture(reads: bool) -> AuthenticatedReferenceEffectBindingV1 {
    let scalar = ReferenceScalarTypeV1::F32;
    let shared = ReferenceSignatureInputV1::Reference {
        region: ReferenceRegionV1::Erased,
        mutability: SemanticMutabilityV1::Immutable,
        pointee: ReferencePointeeV1::Slice(scalar),
    };
    let output = ReferenceSignatureInputV1::NominalOutput {
        carrier: ReferenceCarrierV1::DisjointSlice,
        element: scalar,
    };
    let mut kernel = if reads { vec![shared, shared] } else { vec![] };
    kernel.push(output);
    let mut reference = vec![ReferenceSignatureInputV1::Scalar(
        ReferenceScalarTypeV1::Usize,
    )];
    if reads {
        reference.extend([shared, shared]);
    }
    reference.push(ReferenceSignatureInputV1::Reference {
        region: ReferenceRegionV1::Erased,
        mutability: SemanticMutabilityV1::Mutable,
        pointee: ReferencePointeeV1::Scalar(scalar),
    });
    let signature_preimage = ReferenceLogicalSignaturePreimageV1::new(
        kernel.into_boxed_slice(),
        reference.into_boxed_slice(),
        ReferenceReturnShapeV1::Unit,
        SemanticExternAbiV1::Rust,
        SemanticFunctionSafetyV1::Safe,
        false,
    )
    .unwrap();
    let derived = signature_preimage.derive_relations_v1().unwrap();
    let relations = (0..derived.len())
        .map(|raw| derived.relation_at_raw_argument_v1(raw as u32).unwrap())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let point = || ReferenceEffectExpressionV1::PointCoordinate { axis: 0 };
    let load = |raw| ReferenceEffectExpressionV1::InputLoad {
        reference_argument: raw,
        index: Box::new(point()),
    };
    let operand = |raw: u32| {
        ReferenceOperandV1::Copy(ReferencePlaceV1 {
            local: raw + 1,
            projection: vec![
                ReferencePlaceProjectionV1::Dereference,
                ReferencePlaceProjectionV1::Index(1),
            ]
            .into_boxed_slice(),
        })
    };
    let constant = ReferenceConstantV1::Scalar {
        scalar,
        bits: 0x422a_0000,
    };
    let (value, rhs) = if reads {
        (
            ReferenceValueV1::Binary {
                operation: ReferenceBinaryOpV1::Add,
                lhs: operand(1),
                rhs: operand(2),
                checked: false,
            },
            ReferenceEffectExpressionV1::Binary {
                operation: ReferenceBinaryOpV1::Add,
                lhs: Box::new(load(1)),
                rhs: Box::new(load(2)),
                checked: false,
            },
        )
    } else {
        (
            ReferenceValueV1::Use(ReferenceOperandV1::Constant(constant.clone())),
            ReferenceEffectExpressionV1::Constant(constant),
        )
    };
    let output_argument = if reads { 2 } else { 0 };
    let write = ReferenceOutputWriteV1 {
        argument: output_argument,
        block: 0,
        statement: 0,
        coordinate: ReferenceOutputCoordinateV1::LogicalPoint(vec![point()].into_boxed_slice()),
        guard: ReferencePathPredicateV1::unconditional_v1(),
        rhs,
        value: value.clone(),
    };
    let effect_ir = ReferenceEffectIrV1 {
        argument_count: output_argument + 2,
        local_count: output_argument + 3,
        relations,
        blocks: vec![ReferenceBlockV1 {
            block: 0,
            assignments: vec![ReferenceAssignmentV1 {
                statement: 0,
                destination: ReferencePlaceV1 {
                    local: output_argument + 2,
                    projection: vec![ReferencePlaceProjectionV1::Dereference].into_boxed_slice(),
                },
                value,
            }]
            .into_boxed_slice(),
            terminator: ReferenceTerminatorV1::Return,
        }]
        .into_boxed_slice(),
        loop_summaries: Box::default(),
        observable_output_effects: vec![write.clone()].into_boxed_slice(),
    };
    let identity = ReferenceFunctionIdentityV1 {
        def_path_hash: [1; 16],
        function_sha256: [2; 32],
        item_definition_sha256: [3; 32],
        monomorphization_sha256: [4; 32],
        generic_type_arguments_sha256: [5; 32],
        const_generic_arguments_sha256: [6; 32],
        rustc_mir_body_sha256: [7; 32],
    };
    AuthenticatedReferenceEffectBindingV1 {
        origin: crate::reference_effect_v1::ReferenceBindingOriginV1::SourceRegistration(
            "inert-replay".into(),
        ),
        logical_kernel_name: "inert-replay".into(),
        kernel: identity,
        reference: identity,
        signature_preimage,
        effect_ir_sha256: effect_ir.canonical_sha256_v1(),
        effect_ir,
        observable_output_writes: vec![write].into_boxed_slice(),
    }
}

fn frozen_emit(
    output_ranks: Vec<usize>,
    initial: u32,
) -> (
    R<()>,
    Vec<ProductionRankedValueIdV1>,
    Vec<ProductionRankedOperationV1>,
    u32,
) {
    let mut next_value = initial;
    let mut values = Vec::new();
    let mut entry_operations = Vec::new();
    let result = (|| {
        for rank in output_ranks {
            for _ in 0..3 {
                let result = next_value_id(&mut next_value)?;
                values.push(result);
                entry_operations
                    .push(ProductionRankedOperationV1::SemanticConstant { result, value: 0 });
            }
            for axis in 0..rank {
                let symbol = u32::try_from(axis).map_err(|_| {
                    ProductionRankedProjectionErrorV1::Unsupported(
                        "reference-effect logical point rank does not fit the semantic symbol domain",
                    )
                })?;
                let result = next_value_id(&mut next_value)?;
                values.push(result);
                entry_operations
                    .push(ProductionRankedOperationV1::SemanticSymbol { result, symbol });
            }
        }
        Ok(())
    })();
    (result, values, entry_operations, next_value)
}

#[test]
fn prefix_ordinary_frozen_exact_source_rank_axis_order_and_overflow_partial_state() {
    for initial in [0, 81, u32::MAX - 2] {
        for ranks in [vec![], vec![0], vec![1, 0, 3]] {
            let (expected, values, operations, next) = frozen_emit(ranks.clone(), initial);
            let mut actual_values = Vec::new();
            let mut actual_operations = Vec::new();
            let mut actual_next = initial;
            let actual = emit_reference_prefix_v1(
                ranks,
                &mut actual_values,
                &mut actual_operations,
                &mut actual_next,
                &mut PreparationResourcesV1::unmetered(),
            );
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            assert_eq!(actual_values, values);
            assert_eq!(actual_operations, operations);
            assert_eq!(actual_next, next);
        }
    }
}

struct Paid {
    result: R<()>,
    operations: Vec<ProductionRankedOperationV1>,
    values: Option<Vec<ProductionRankedValueIdV1>>,
    ranks: Vec<usize>,
    next: u32,
    work: usize,
    peak: usize,
    owned: usize,
    failed_work: bool,
    failed_storage: bool,
}
fn paid(
    bindings: &[AuthenticatedReferenceEffectBindingV1],
    work_limit: usize,
    storage_limit: usize,
) -> Paid {
    let mut work = Work::new(work_limit);
    let mut budget = Budget::new(&mut work, storage_limit);
    budget.reserve_storage(1024).unwrap(); // Inert existing prefix and fixed owner headers.
    let identity = budget.work_ledger_identity_v1();
    let mut owned = 0usize;
    let mut pending = RootEntryPrefixV1::empty();
    pending
        .entry_operations
        .push(ProductionRankedOperationV1::SemanticConstant {
            result: ProductionRankedValueIdV1::new(76),
            value: 91,
        });
    pending.next_value = 77;
    let result = prepare_reference_prefix_paid_v1(
        bindings,
        &mut pending,
        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
    );
    assert!(budget.work_ledger_identity_v1() == identity);
    assert!(budget.storage() >= 1024 + owned);
    // Inert test returns ownership for exact comparison; no component refund is
    // performed while any successful or partial payload remains physically live.
    Paid {
        result,
        operations: pending.entry_operations,
        values: pending.reserved_reference_values,
        ranks: pending.rank_scratch,
        next: pending.next_value,
        work: budget.work(),
        peak: budget.peak_storage(),
        owned,
        failed_work: budget.failed_work().is_some(),
        failed_storage: budget.failed_storage().is_some(),
    }
}
#[test]
fn prefix_paid_nonempty_reference_preserves_real_prior_namespace_and_every_operation() {
    let bindings = [fixture(false)];
    let actual = paid(&bindings, usize::MAX, usize::MAX);
    assert!(actual.result.is_ok());
    let (result, values, emitted, next) = frozen_emit(vec![1], 77);
    assert!(result.is_ok());
    assert_eq!(
        actual.operations[0],
        ProductionRankedOperationV1::SemanticConstant {
            result: ProductionRankedValueIdV1::new(76),
            value: 91,
        }
    );
    assert_eq!(&actual.operations[1..], &emitted);
    assert_eq!(actual.values, Some(values));
    assert_eq!(actual.ranks, vec![1]);
    assert_eq!(actual.next, next);
    assert!(actual.owned > 0);
    assert!(!actual.failed_work && !actual.failed_storage);
}
#[test]
fn prefix_empty_reference_is_none_and_does_not_reseed_existing_allocator() {
    let actual = paid(&[], usize::MAX, usize::MAX);
    assert!(actual.result.is_ok());
    assert_eq!(actual.next, 77);
    assert_eq!(actual.operations.len(), 1);
    assert!(actual.values.is_none());
    assert!(actual.ranks.is_empty());
}
#[test]
fn prefix_reference_validation_preserves_duplicate_empty_and_coordinate_refusal_order() {
    let mut empty = fixture(false);
    empty.observable_output_writes = Box::default();
    let duplicate = paid(&[empty.clone(), fixture(false)], usize::MAX, usize::MAX);
    assert!(matches!(
        duplicate.result,
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "reference-effect projection requires exactly one authenticated kernel/reference binding"
        ))
    ));
    let none = paid(&[empty], usize::MAX, usize::MAX);
    assert!(none.result.is_err());
    assert!(none.ranks.is_empty() && none.values.is_none());
    assert_eq!(none.next, 77);
    let mut bad = fixture(false);
    let first = bad.observable_output_writes[0].clone();
    let mut second = first.clone();
    second.coordinate = ReferenceOutputCoordinateV1::SingleCoordinate;
    bad.observable_output_writes = vec![first, second].into_boxed_slice();
    let denied = paid(&[bad], usize::MAX, usize::MAX);
    assert!(matches!(
        denied.result,
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "reference-effect projection requires independently indexed logical point outputs"
        ))
    ));
    assert_eq!(denied.ranks, vec![1], "validated first rank remains OUTER");
    assert!(
        denied.values.is_none(),
        "all coordinates precede SSA allocation"
    );
    assert_eq!(denied.next, 77);
    assert_eq!(denied.operations.len(), 1);
}
#[test]
fn prefix_paid_exact_work_storage_and_one_short_keep_partial_payloads() {
    let bindings = [fixture(false)];
    let measured = paid(&bindings, usize::MAX, usize::MAX);
    let exact = paid(&bindings, measured.work, measured.peak);
    assert!(exact.result.is_ok());
    assert_eq!(exact.operations, measured.operations);
    assert_eq!(exact.values, measured.values);
    assert_eq!((exact.work, exact.peak), (measured.work, measured.peak));
    let short_work = paid(&bindings, measured.work - 1, measured.peak);
    assert!(short_work.result.is_err());
    assert!(short_work.failed_work && !short_work.failed_storage);
    assert_eq!(short_work.ranks, vec![1]);
    assert!(
        short_work
            .values
            .as_ref()
            .is_some_and(|rows| !rows.is_empty())
    );
    let short_storage = paid(&bindings, measured.work, measured.peak - 1);
    assert!(short_storage.result.is_err());
    assert!(short_storage.failed_storage && !short_storage.failed_work);
    assert_eq!(short_storage.ranks, vec![1]);
    assert!(short_storage.values.is_some());
    assert!(short_storage.owned > 0);
}
#[test]
fn prefix_no_meter_and_sticky_denial_refuse_before_reference_mutation() {
    let bindings = [fixture(false)];
    let mut pending = RootEntryPrefixV1::empty();
    assert!(
        prepare_reference_prefix_paid_v1(
            &bindings,
            &mut pending,
            &mut PreparationResourcesV1::unmetered()
        )
        .is_err()
    );
    assert!(pending.entry_operations.is_empty() && pending.rank_scratch.is_empty());
    let mut work = Work::new(0);
    let mut budget = Budget::new(&mut work, usize::MAX);
    assert!(budget.charge_work(1).is_err());
    let mut owned = 0;
    assert!(
        prepare_reference_prefix_paid_v1(
            &bindings,
            &mut pending,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .is_err()
    );
    assert_eq!(owned, 0);
    assert!(pending.entry_operations.is_empty() && pending.rank_scratch.is_empty());
}
#[test]
fn prefix_outer_owner_survives_post_preparation_error_and_panic_until_drop_then_refund() {
    for panic_after in [false, true] {
        let bindings = [fixture(false)];
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(1024).unwrap();
        let identity = budget.work_ledger_identity_v1();
        let mut owned = 0;
        let mut pending = RootEntryPrefixV1::empty();
        let result = catch_unwind(AssertUnwindSafe(|| -> R<()> {
            prepare_reference_prefix_paid_v1(
                &bindings,
                &mut pending,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            )?;
            if panic_after {
                panic!("inert prefix post-preparation panic");
            }
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "inert prefix post-preparation error",
            ))
        }));
        assert_eq!(result.is_err(), panic_after);
        assert_eq!(pending.entry_operations.len(), 4);
        assert_eq!(pending.reserved_reference_values.as_ref().unwrap().len(), 4);
        assert!(budget.work_ledger_identity_v1() == identity);
        assert!(budget.storage() >= 1024 + owned);
        drop(result);
        drop(pending);
        budget.release_storage(owned).unwrap();
        assert_eq!(budget.storage(), 1024);
    }
}

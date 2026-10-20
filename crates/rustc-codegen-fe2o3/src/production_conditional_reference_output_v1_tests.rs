//! Inert CPU IR tests only; no source, proof, runtime or launch authority.
#[path = "production_conditional_reference_composition_v1_tests.rs"]
mod composition;

use super::*;
use crate::reference_effect_v1::{
    ReferenceAssignmentV1, ReferenceBlockV1, ReferenceEffectIrV1, ReferenceFunctionIdentityV1,
    ReferenceLogicalSignaturePreimageV1, ReferencePathPredicateV1, ReferencePlaceV1,
    ReferenceUnaryOpV1,
    reference_signature_preimage_v1::{
        ReferencePointeeV1, ReferenceRegionV1, ReferenceReturnShapeV1,
    },
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticExternAbiV1, SemanticFunctionSafetyV1, SemanticMutabilityV1,
};

fn fixture(bits: u128) -> AuthenticatedReferenceEffectBindingV1 {
    typed_fixture(Scalar::U32, bits)
}

fn typed_fixture(scalar: Scalar, bits: u128) -> AuthenticatedReferenceEffectBindingV1 {
    let value = Rvalue::Use(Operand::Constant(Const::Scalar { scalar, bits }));
    let write = ReferenceOutputWriteV1 {
        argument: 0,
        block: 0,
        statement: 0,
        coordinate: Coord::LogicalPoint(
            vec![CpuExpr::PointCoordinate { axis: 0 }].into_boxed_slice(),
        ),
        guard: ReferencePathPredicateV1::unconditional_v1(),
        rhs: CpuExpr::Constant(Const::Scalar { scalar, bits }),
        value: value.clone(),
    };
    let effect_ir = ReferenceEffectIrV1 {
        argument_count: 2,
        local_count: 3,
        relations: vec![
            Rel::PointCoordinate {
                reference_argument: 0,
                axis: 0,
            },
            Rel::DisjointOutputCoordinate {
                argument: 0,
                element: scalar,
            },
        ]
        .into_boxed_slice(),
        blocks: vec![ReferenceBlockV1 {
            block: 0,
            assignments: vec![
                ReferenceAssignmentV1 {
                    statement: 0,
                    destination: ReferencePlaceV1 {
                        local: 2,
                        projection: vec![Projection::Dereference].into_boxed_slice(),
                    },
                    value,
                },
                ReferenceAssignmentV1 {
                    statement: 1,
                    destination: ReferencePlaceV1 {
                        local: 0,
                        projection: Box::default(),
                    },
                    value: Rvalue::Use(Operand::Constant(Const::ZeroSized)),
                },
            ]
            .into_boxed_slice(),
            terminator: Term::Return,
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
            "inert".into(),
        ),
        logical_kernel_name: "inert".into(),
        kernel: identity,
        reference: identity,
        signature_preimage: typed_signature(Scalar::Usize, scalar),
        effect_ir_sha256: effect_ir.canonical_sha256_v1(),
        effect_ir,
        observable_output_writes: vec![write].into_boxed_slice(),
    }
}

fn signature(point: Scalar) -> ReferenceLogicalSignaturePreimageV1 {
    typed_signature(point, Scalar::U32)
}

fn typed_signature(point: Scalar, scalar: Scalar) -> ReferenceLogicalSignaturePreimageV1 {
    ReferenceLogicalSignaturePreimageV1::new(
        vec![Input::NominalOutput {
            carrier: Carrier::DisjointSlice,
            element: scalar,
        }]
        .into_boxed_slice(),
        vec![
            Input::Scalar(point),
            Input::Reference {
                region: ReferenceRegionV1::Erased,
                mutability: SemanticMutabilityV1::Mutable,
                pointee: ReferencePointeeV1::Scalar(scalar),
            },
        ]
        .into_boxed_slice(),
        ReferenceReturnShapeV1::Unit,
        SemanticExternAbiV1::Rust,
        SemanticFunctionSafetyV1::Safe,
        false,
    )
    .unwrap()
}

#[test]
fn constant_point_effect_borrows_the_exact_output_and_supports_u32_limits() {
    for bits in [0, 1, 17, u128::from(u32::MAX)] {
        for unit_return in [false, true] {
            let mut input = fixture(bits);
            if !unit_return {
                input.effect_ir.blocks[0].assignments =
                    vec![input.effect_ir.blocks[0].assignments[0].clone()].into_boxed_slice();
                input.effect_ir_sha256 = input.effect_ir.canonical_sha256_v1();
            }
            let mut work = Work::new(100_000);
            let mut budget = Budget::new(&mut work, 0);
            let (checked, actual) = check_constant_point_effect(&input, &mut budget).unwrap();
            assert!(std::ptr::eq(checked.binding, &input));
            assert!(std::ptr::eq(
                checked.write,
                &input.observable_output_writes[0]
            ));
            assert_eq!(checked.raw_argument, 1);
            assert_eq!(actual.scalar, ConstantScalarV1::U32);
            assert_eq!(u128::from(actual.bits), bits);
            assert_eq!(budget.storage(), 0);
        }
    }
}

#[test]
fn f32_point_constants_retain_type_and_every_representation_bit() {
    for bits in [
        0,
        0x8000_0000,
        0x422a_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0123,
        0xffc0_0456,
    ] {
        let input = typed_fixture(Scalar::F32, bits);
        let mut work = Work::new(100_000);
        let mut budget = Budget::new(&mut work, 0);
        let (checked, constant) = check_constant_point_effect(&input, &mut budget).unwrap();
        assert!(std::ptr::eq(checked.binding, &input));
        assert_eq!(constant.scalar, ConstantScalarV1::F32);
        assert_eq!(u128::from(constant.bits), bits);
        assert_eq!(constant.scalar.canonical(), ScalarType::F32);
        assert_eq!(constant.scalar.ranked(), Ty::Float { bits: 32 });
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn f32_constants_reject_integer_substitution_width_and_payload_changes() {
    for change in 0..6 {
        let mut input = typed_fixture(Scalar::F32, 0x422a_0000);
        let expected = match change {
            0 => {
                input.signature_preimage = typed_signature(Scalar::Usize, Scalar::U32);
                "IR relation"
            }
            1 => {
                input.effect_ir.blocks[0].assignments[0].value =
                    Rvalue::Use(Operand::Constant(Const::Scalar {
                        scalar: Scalar::U32,
                        bits: 0x422a_0000,
                    }));
                "CPU value"
            }
            2 => {
                input.observable_output_writes[0].rhs = CpuExpr::Constant(Const::Scalar {
                    scalar: Scalar::U32,
                    bits: 0x422a_0000,
                });
                "CPU RHS"
            }
            3 => {
                input.effect_ir.observable_output_effects[0].value =
                    Rvalue::Use(Operand::Constant(Const::Scalar {
                        scalar: Scalar::F32,
                        bits: 0x422a_0001,
                    }));
                "CPU write identity/value"
            }
            4 => {
                input = typed_fixture(Scalar::F32, 1_u128 << 32);
                "CPU constant width"
            }
            5 => {
                input = typed_fixture(Scalar::F64, 0x422a_0000);
                "signature fragment"
            }
            _ => unreachable!(),
        };
        let mut work = Work::new(100_000);
        let result = check_constant_point_effect(&input, &mut Budget::new(&mut work, 0));
        assert!(
            matches!(result, Err(Error::Reference(reason)) if reason == expected),
            "change={change}"
        );
    }
}

#[test]
fn malformed_point_effects_refuse_without_hash_or_authority_substitution() {
    type Mutation = fn(&mut AuthenticatedReferenceEffectBindingV1);
    let mutations: &[(&str, Mutation)] = &[
        ("IR relation", |b| b.effect_ir.relations.swap(0, 1)),
        ("CPU block", |b| b.effect_ir.blocks[0].block = 1),
        ("statement order", |b| {
            b.effect_ir.blocks[0].assignments[1].statement = 0
        }),
        ("CPU assignment", |b| {
            b.effect_ir.blocks[0].assignments[0].destination.local = 1
        }),
        ("duplicate CPU store", |b| {
            let mut second = b.effect_ir.blocks[0].assignments[0].clone();
            second.statement = 1;
            b.effect_ir.blocks[0].assignments[1] = second;
        }),
        ("CPU write identity/value", |b| {
            b.observable_output_writes[0].argument = 1
        }),
        ("CPU guard", |b| {
            b.observable_output_writes[0].guard.clauses = Box::default()
        }),
        ("CPU write identity/value", |b| {
            b.effect_ir.observable_output_effects[0].statement = 1
        }),
        ("effect digest", |b| b.effect_ir_sha256[0] ^= 1),
    ];
    for &(expected, mutate) in mutations {
        let mut input = fixture(17);
        mutate(&mut input);
        let mut work = Work::new(100_000);
        let mut budget = Budget::new(&mut work, 0);
        let result = check_constant_point_effect(&input, &mut budget).map(|_| ());
        assert!(
            matches!(result, Err(Error::Reference(reason)) if reason == expected),
            "{expected}: {result:?}"
        );
        assert_eq!(budget.storage(), 0);
    }
    let input = fixture(u128::from(u32::MAX) + 1);
    let mut work = Work::new(100_000);
    assert!(matches!(
        check_constant_point_effect(&input, &mut Budget::new(&mut work, 0)),
        Err(Error::Reference("CPU constant width"))
    ));
}

#[test]
fn changed_signature_is_not_authorized_by_an_unchanged_effect_digest() {
    let mut input = fixture(17);
    let digest = input.effect_ir_sha256;
    input.signature_preimage = signature(Scalar::U64);
    assert_eq!(input.effect_ir.canonical_sha256_v1(), digest);
    let mut work = Work::new(100_000);
    assert!(matches!(
        check_constant_point_effect(&input, &mut Budget::new(&mut work, 0)),
        Err(Error::Reference("signature relation"))
    ));
}

#[test]
fn recursive_expression_is_refused_before_recursive_hashing() {
    let mut input = fixture(17);
    for _ in 0..32 {
        let old = std::mem::replace(
            &mut input.effect_ir.observable_output_effects[0].rhs,
            CpuExpr::PointCoordinate { axis: 0 },
        );
        input.effect_ir.observable_output_effects[0].rhs = CpuExpr::Unary {
            operation: ReferenceUnaryOpV1::Not,
            operand: Box::new(old),
        };
    }
    let mut work = Work::new(1024);
    let mut budget = Budget::new(&mut work, 0);
    assert!(matches!(
        check_constant_point_effect(&input, &mut budget),
        Err(Error::Reference("CPU RHS"))
    ));
    assert_eq!(budget.work(), 1024);
    assert_eq!(work.failed_work(), None);
}

#[test]
fn point_effect_work_is_cumulative_and_storage_is_unchanged() {
    let input = fixture(17);
    let run = |limit| {
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, 31);
        budget.reserve_storage(31).unwrap();
        budget.charge_work(17).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let result = check_constant_point_effect(&input, &mut budget).map(|_| ());
        assert_eq!(budget.storage(), 31);
        assert!(budget.work_ledger_identity_v1() == ledger);
        let used = budget.work();
        (result, used, work.failed_work())
    };
    let (result, exact, failed) = run(100_000);
    result.unwrap();
    assert_eq!(failed, None);
    run(exact).0.unwrap();
    let (short, _, failed) = run(exact - 1);
    assert!(matches!(short, Err(Error::Resource(Resource::Work(_)))));
    assert_eq!(failed, Some(exact));
}

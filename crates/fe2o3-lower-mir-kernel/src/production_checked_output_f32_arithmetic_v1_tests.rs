use super::*;
use fe2o3_kernel_ir::{FunctionRole, ScalarType, UnaryOp, VerifiedCanonicalKernelIrV12};
use fe2o3_kir_sim::{
    AdmittedSimulationModuleV1, BufferArgumentV1, ScalarBitsV1, SimulationArgumentV1,
    SimulationConflictAssessmentV1, SimulationKernelIrIdentityV1, SimulationLimitsV1,
    SimulationRaceAssessmentV1, SimulationRequestV1, SimulationTargetV1,
};

include!("production_checked_output_f32_fixture_v1_tests.rs");
#[path = "production_checked_output_exp_connected_v1_tests.rs"]
mod exp_connected;
#[path = "production_checked_output_f32_native_v1_tests.rs"]
mod native;
#[path = "production_checked_output_numeric_cast_connected_v1_tests.rs"]
mod numeric_cast_connected;

fn fp_census(module: &Module, recipe: FpRecipe, helper: bool) {
    let mut recipes = 0;
    let mut calls = 0;
    let mut stores = 0;
    for function in &module.functions {
        for operation in function
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
        {
            let selected = match operation.kind {
                OperationKind::Unary {
                    op: UnaryOp::Negate,
                    ..
                } => recipe == FpRecipe::Negate,
                OperationKind::Binary {
                    op: BinaryOp::Divide,
                    ..
                } => recipe == FpRecipe::Divide,
                _ => false,
            };
            if selected {
                recipes += 1;
                assert_eq!(function.role == FunctionRole::InternalHelper, helper);
                assert!(matches!(operation.results.as_slice(), [result] if result.ty == Type::F32));
            }
            if matches!(operation.kind, OperationKind::Call { .. }) {
                calls += 1;
            }
            if matches!(operation.kind, OperationKind::Store { .. }) {
                stores += 1;
            }
        }
    }
    assert_eq!((recipes, calls, stores), (1, usize::from(helper), 1));
}

#[test]
fn f32_negate_divide_direct_and_retained_helpers_replay_complete_policy3_and_policy4() {
    for profile in [Profile::Gfx942, Profile::Gfx950] {
        for helper in [false, true] {
            for recipe in [FpRecipe::Negate, FpRecipe::Divide] {
                with_prepared(
                    prepare(fp_receipt(32, recipe, helper), profile, None),
                    |input, budget| {
                        fp_census(
                            input.receipt.materialized.executable().module(),
                            recipe,
                            helper,
                        );
                        fp_census(input.bound.module(), recipe, helper);
                        let floor = budget.storage();
                        let admitted = AdmittedOutput::try_admit_general_v1(
                            input.receipt,
                            input.bound,
                            input.output,
                            budget,
                        )
                        .unwrap();
                        fp_census(admitted.output().module(), recipe, helper);
                        assert_eq!(admitted.kernels()[0].accesses().len(), 1);
                        admitted.verify_equivalence(budget).unwrap();
                        assert_eq!(budget.storage(), floor);
                    },
                );
                let input = fp_prepare4(recipe, helper, profile, None);
                let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
                let mut budget = AssertOriginBudgetV1::new(&mut work, STORAGE);
                budget.reserve_storage(input.floor).unwrap();
                fp_census(
                    input.checked.intermediate_policy3().owner().module(),
                    recipe,
                    helper,
                );
                let owner = crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                    input.receipt,
                    input.bound,
                    input.checked,
                    &mut budget,
                )
                .unwrap();
                fp_census(owner.output().module(), recipe, helper);
                owner.verify_equivalence(&mut budget).unwrap();
                assert_eq!(owner.kernels()[0].accesses().len(), 1);
                assert_eq!(budget.storage(), input.floor);
                assert!(!owner.grants_artifact_or_launch_authority());
                native::check_actual(&owner, profile, recipe, &mut budget);
                assert_eq!(budget.storage(), input.floor);
            }
        }
    }
}

fn fp_limits() -> SimulationLimitsV1 {
    SimulationLimitsV1 {
        max_canonical_bytes: 1 << 20,
        max_reachable_functions: 8,
        max_reachable_operations: 1024,
        max_invocations: 1,
        max_workgroups: 1,
        max_scheduled_slots: 1,
        max_steps: 1024,
        max_call_depth: 8,
        max_ssa_values: 1024,
        max_allocations: 8,
        max_allocation_bytes: 4096,
        max_total_bytes: 4096,
        max_resident_bytes: 64 << 20,
        max_events: 4096,
        max_memory_access_records: 64,
    }
}

fn fp_request(left: u32, right: u32) -> SimulationRequestV1 {
    let target = SimulationTargetV1::amdgpu_64();
    let scalar = |bits: u32| ScalarBitsV1::new(ScalarType::F32, bits.into(), target).unwrap();
    SimulationRequestV1::new(
        FP_NAME,
        [1, 1, 1],
        [1, 1, 1],
        vec![
            SimulationArgumentV1::Buffer(
                BufferArgumentV1::from_scalars(
                    AccessMode::ReadWrite,
                    4,
                    &[
                        scalar(0xdead_beef),
                        scalar(0x7fc0_0042),
                        scalar(0x8000_0000),
                    ],
                    target,
                )
                .unwrap(),
            ),
            SimulationArgumentV1::Scalar(scalar(left)),
            SimulationArgumentV1::Scalar(scalar(right)),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
enum FpExpected {
    Bits(u32),
    Nan,
}

fn fp_vectors(recipe: FpRecipe) -> Vec<(u32, u32, FpExpected)> {
    if recipe == FpRecipe::Negate {
        return [
            0,
            0x8000_0000,
            0x3f80_0000,
            0xbf80_0000,
            1,
            0x0080_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0042,
            0x7f80_0042,
        ]
        .into_iter()
        .map(|bits| (bits, 0, FpExpected::Bits(bits ^ 0x8000_0000)))
        .collect();
    }
    assert_eq!(recipe, FpRecipe::Divide);
    vec![
        (0x40c0_0000, 0x4000_0000, FpExpected::Bits(0x4040_0000)), // 6 / 2
        (0x3f80_0000, 0x4040_0000, FpExpected::Bits(0x3eaa_aaab)), // RNE 1 / 3
        (0, 0x3f80_0000, FpExpected::Bits(0)),
        (0x8000_0000, 0x3f80_0000, FpExpected::Bits(0x8000_0000)),
        (0x3f80_0000, 0, FpExpected::Bits(0x7f80_0000)),
        (0x3f80_0000, 0x8000_0000, FpExpected::Bits(0xff80_0000)),
        (0xbf80_0000, 0x7f80_0000, FpExpected::Bits(0x8000_0000)),
        (0x7f80_0000, 0xbf80_0000, FpExpected::Bits(0xff80_0000)),
        (0x7f7f_ffff, 0x3f00_0000, FpExpected::Bits(0x7f80_0000)),
        (0x0080_0000, 0x4000_0000, FpExpected::Bits(0x0040_0000)),
        (1, 0x4000_0000, FpExpected::Bits(0)), // underflow midpoint, even zero
        (3, 0x4000_0000, FpExpected::Bits(2)), // subnormal midpoint, even two
        (0, 0, FpExpected::Nan),
        (0x7f80_0000, 0x7f80_0000, FpExpected::Nan),
        (0x7fc0_0042, 0x3f80_0000, FpExpected::Nan),
        (0x7f80_0042, 0x3f80_0000, FpExpected::Nan),
    ]
}

#[test]
fn actual_policy4_f32_output_has_defined_special_values_and_deterministic_cpu_execution() {
    for profile in [Profile::Gfx942, Profile::Gfx950] {
        for helper in [false, true] {
            for recipe in [FpRecipe::Negate, FpRecipe::Divide] {
                let input = fp_prepare4(recipe, helper, profile, None);
                let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
                let mut budget = AssertOriginBudgetV1::new(&mut work, STORAGE);
                budget.reserve_storage(input.floor).unwrap();
                let owner = crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                    input.receipt,
                    input.bound,
                    input.checked,
                    &mut budget,
                )
                .unwrap();
                let output = owner.output().canonical();
                let decoded = VerifiedCanonicalKernelIrV12::from_canonical_bytes(
                    output.canonical_bytes().to_vec(),
                )
                .unwrap();
                assert_eq!(decoded.identity(), output.identity());
                assert_eq!(decoded.canonical_bytes(), output.canonical_bytes());
                let identity = SimulationKernelIrIdentityV1::from(*decoded.identity());
                let simulation =
                    AdmittedSimulationModuleV1::admit_v12(decoded, fp_limits()).unwrap();
                assert_eq!(simulation.identity(), &identity);
                assert!(!simulation.grants_execution_authority());
                for (left, right, expected) in fp_vectors(recipe) {
                    let request = fp_request(left, right);
                    let before = request.clone();
                    let execution = simulation
                        .simulate(&request, SimulationTargetV1::amdgpu_64(), fp_limits())
                        .unwrap();
                    let again = simulation
                        .simulate(&request, SimulationTargetV1::amdgpu_64(), fp_limits())
                        .unwrap();
                    assert_eq!(execution, again);
                    assert_eq!(request, before);
                    assert_eq!(execution.identity(), &identity);
                    assert_eq!(execution.invocations_executed(), 1);
                    assert_eq!(execution.arguments()[1..], request.arguments[1..]);
                    let result = execution.buffer(0).unwrap();
                    assert_eq!(result.bytes().len(), 12);
                    assert!(result.initialized().iter().all(|bit| *bit));
                    assert_eq!(&result.bytes()[4..8], &0x7fc0_0042_u32.to_le_bytes());
                    assert_eq!(&result.bytes()[8..12], &0x8000_0000_u32.to_le_bytes());
                    let bits = u32::from_le_bytes(result.bytes()[..4].try_into().unwrap());
                    match expected {
                        FpExpected::Bits(expected) => {
                            assert_eq!(bits, expected, "{recipe:?}: {left:08x}/{right:08x}")
                        }
                        // Division NaN payload/sign selection is not promoted
                        // from a software-model choice to a native guarantee.
                        FpExpected::Nan => assert!(f32::from_bits(bits).is_nan()),
                    }
                    assert!(matches!(
                        execution.conflict_assessment(),
                        SimulationConflictAssessmentV1::NoConflictsObserved
                    ));
                    assert!(matches!(
                        execution.race_assessment(),
                        SimulationRaceAssessmentV1::NoRacesObserved { .. }
                    ));
                }
                let mut one_step = fp_limits();
                one_step.max_steps = 1;
                assert!(matches!(simulation.simulate(&fp_request(0x3f80_0000, 0),
                    SimulationTargetV1::amdgpu_64(), one_step),
                    Err(fe2o3_kir_sim::SimulationErrorV1::Execution(error))
                        if matches!(error.kind, fe2o3_kir_sim::SimulationExecutionErrorKindV1::StepLimit { .. })));
                assert_eq!(budget.storage(), input.floor);
            }
        }
    }
}

fn fp_mutate_operator(module: &mut Module) {
    let operation = module
        .functions
        .iter_mut()
        .filter_map(|f| f.body.as_mut())
        .flat_map(|b| &mut b.blocks)
        .flat_map(|b| &mut b.operations)
        .find(|operation| {
            matches!(
                operation.kind,
                OperationKind::Binary {
                    op: BinaryOp::Divide,
                    ..
                }
            )
        })
        .unwrap();
    let OperationKind::Binary { op, .. } = &mut operation.kind else {
        unreachable!()
    };
    *op = BinaryOp::Multiply;
}

fn fp_mutate_operand(module: &mut Module) {
    let operation = module
        .functions
        .iter_mut()
        .filter_map(|f| f.body.as_mut())
        .flat_map(|b| &mut b.blocks)
        .flat_map(|b| &mut b.operations)
        .find(|operation| {
            matches!(
                operation.kind,
                OperationKind::Binary {
                    op: BinaryOp::Divide,
                    ..
                }
            )
        })
        .unwrap();
    let OperationKind::Binary { lhs, rhs, .. } = &mut operation.kind else {
        unreachable!()
    };
    assert_ne!(*lhs, *rhs);
    std::mem::swap(lhs, rhs);
}

fn fp_mutate_negated_operand(module: &mut Module) {
    for body in module.functions.iter_mut().filter_map(|f| f.body.as_mut()) {
        let replacement = *body.parameters.last().unwrap();
        for operation in body
            .blocks
            .iter_mut()
            .flat_map(|block| &mut block.operations)
        {
            if let OperationKind::Unary {
                op: UnaryOp::Negate,
                operand,
            } = &mut operation.kind
            {
                assert_ne!(*operand, replacement);
                *operand = replacement;
                return;
            }
        }
    }
    panic!("actual Negate operation missing");
}

#[test]
fn freshly_verified_f32_operator_operand_and_history_substitutions_fail_source_join() {
    for profile in [Profile::Gfx942, Profile::Gfx950] {
        for helper in [false, true] {
            for (recipe, mutation) in [
                (FpRecipe::Divide, fp_mutate_operator as fn(&mut Module)),
                (FpRecipe::Divide, fp_mutate_operand),
                (FpRecipe::Negate, fp_mutate_negated_operand),
            ] {
                let input = fp_prepare4(recipe, helper, profile, Some(mutation));
                let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
                let mut budget = AssertOriginBudgetV1::new(&mut work, STORAGE);
                budget.reserve_storage(input.floor).unwrap();
                let result = crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                    input.receipt,
                    input.bound,
                    input.checked,
                    &mut budget,
                );
                assert!(matches!(
                    result,
                    Err(
                        crate::ProductionCheckedOutputAdmissionErrorPolicy4V1::Admission(
                            AdmissionError::Coordinates(_)
                        )
                    )
                ));
                assert_eq!(budget.storage(), input.floor);
            }
            let original = fp_prepare4(FpRecipe::Divide, helper, profile, None);
            let other = fp_prepare4(FpRecipe::Negate, helper, profile, None);
            let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
            let mut budget = AssertOriginBudgetV1::new(&mut work, STORAGE);
            let floor = original.floor + other.floor;
            budget.reserve_storage(floor).unwrap();
            assert!(
                crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                    original.receipt,
                    original.bound,
                    other.checked,
                    &mut budget,
                )
                .is_err()
            );
            assert_eq!(budget.storage(), floor);
        }
    }
}

#[test]
fn exact_source_nan_payload_is_not_replaced_by_nan_class_equivalence() {
    fn change_nan(module: &mut Module) {
        for operation in module
            .functions
            .iter_mut()
            .filter_map(|f| f.body.as_mut())
            .flat_map(|b| &mut b.blocks)
            .flat_map(|b| &mut b.operations)
        {
            if let OperationKind::Constant(fe2o3_kernel_ir::Constant::F32Bits(bits)) =
                &mut operation.kind
            {
                assert_eq!(*bits, 0x7fc0_0042);
                *bits = 0x7fc0_0043;
                return;
            }
        }
        panic!("actual NaN source constant missing");
    }
    for profile in [Profile::Gfx942, Profile::Gfx950] {
        for helper in [false, true] {
            for mutation in [None, Some(change_nan as fn(&mut Module))] {
                let receipt =
                    fp_receipt_with_left_bits(32, FpRecipe::Divide, helper, Some(0x7fc0_0042));
                let input = fp_prepare_receipt4(receipt, profile, mutation);
                let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
                let mut budget = AssertOriginBudgetV1::new(&mut work, STORAGE);
                budget.reserve_storage(input.floor).unwrap();
                let result = crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                    input.receipt,
                    input.bound,
                    input.checked,
                    &mut budget,
                );
                if mutation.is_none() {
                    let admitted = result.unwrap();
                    admitted.verify_equivalence(&mut budget).unwrap();
                    native::check_actual_literal_nan(&admitted, profile);
                } else {
                    assert!(matches!(
                        result,
                        Err(
                            crate::ProductionCheckedOutputAdmissionErrorPolicy4V1::Admission(
                                AdmissionError::Coordinates(_)
                            )
                        )
                    ));
                }
                assert_eq!(budget.storage(), input.floor);
            }
        }
    }
}

#[test]
fn f32_extension_keeps_f64_negate_divide_and_float_remainder_refusals() {
    for (bits, recipe) in [
        (64, FpRecipe::Negate),
        (64, FpRecipe::Divide),
        (32, FpRecipe::Remainder),
        (64, FpRecipe::Remainder),
    ] {
        for helper in [false, true] {
            with_prepared(
                prepare(fp_receipt(bits, recipe, helper), Profile::Gfx942, None),
                |input, budget| {
                    let floor = budget.storage();
                    let result = AdmittedOutput::try_admit_general_v1(
                        input.receipt,
                        input.bound,
                        input.output,
                        budget,
                    );
                    assert!(matches!(
                        result,
                        Err(AdmissionError::Unsupported {
                            phase: "source",
                            detail: "total scalar/global recipe; no unchecked arithmetic",
                        })
                    ));
                    assert_eq!(budget.storage(), floor);
                },
            );
        }
    }
}

#[test]
fn f32_actual_owner_admission_preserves_exact_and_one_short_resource_failures() {
    for recipe in [FpRecipe::Negate, FpRecipe::Divide] {
        for helper in [false, true] {
            let input = fp_prepare4(recipe, helper, Profile::Gfx942, None);
            let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
            let mut budget = AssertOriginBudgetV1::new(&mut work, STORAGE);
            budget.reserve_storage(input.floor).unwrap();
            drop(
                crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                    input.receipt,
                    input.bound,
                    input.checked,
                    &mut budget,
                )
                .unwrap(),
            );
            let required_work = budget.work();
            let required_storage = budget.peak_storage();
            assert_eq!(budget.storage(), input.floor);
            assert!(required_work > 0 && required_storage > input.floor);
            for (work_limit, storage_limit, success) in [
                (required_work, required_storage, true),
                (required_work - 1, required_storage, false),
                (required_work, required_storage - 1, false),
            ] {
                let input = fp_prepare4(recipe, helper, Profile::Gfx942, None);
                let mut work = CanonicalKernelIrWorkBudgetV1::new(work_limit);
                let mut budget = AssertOriginBudgetV1::new(&mut work, storage_limit);
                budget.reserve_storage(input.floor).unwrap();
                {
                    let result = crate::ProductionCheckedOutputOwnerPolicy4V1::try_admit_v1(
                        input.receipt,
                        input.bound,
                        input.checked,
                        &mut budget,
                    );
                    assert_eq!(result.is_ok(), success, "{result:?}");
                }
                assert_eq!(budget.storage(), input.floor);
                if success {
                    assert_eq!(budget.work(), required_work);
                    assert_eq!(budget.peak_storage(), required_storage);
                } else if work_limit < required_work {
                    assert!(work.failed_work().is_some());
                } else {
                    assert!(budget.failed_storage().is_some());
                }
            }
        }
    }
}

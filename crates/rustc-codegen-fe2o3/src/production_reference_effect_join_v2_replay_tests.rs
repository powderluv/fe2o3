//! Inert MIR-replay tests. These fixtures provide no source or proof authority.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticExternAbiV1, SemanticFunctionSafetyV1, SemanticMutabilityV1,
};

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

#[test]
fn cpu_replay_fill_and_two_input_add_preserve_raw_argument_mapping() {
    for reads in [false, true] {
        let binding = fixture(reads);
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(31).unwrap();
        budget.charge_work(17).unwrap();
        let account = budget.work_ledger_identity_v1();
        binding
            .with_replayed_output_writes_v1(&mut budget, |replay, budget| {
                let writes = replay.writes.as_slice();
                assert_eq!(writes, binding.observable_output_writes.as_ref());
                assert!(!std::ptr::eq(
                    writes.as_ptr(),
                    binding.observable_output_writes.as_ptr()
                ));
                assert!(budget.work_ledger_identity_v1() == account);
                assert!(budget.storage() > 31);
                assert_eq!(writes[0].argument, if reads { 2 } else { 0 });
                budget.reserve_storage(7).unwrap();
            })
            .unwrap();
        assert_eq!(budget.storage(), 38, "only replay scratch may be released");
        assert!(budget.work() > 17);
    }
}

#[test]
fn cpu_replay_mutations_never_expose_a_correspondence() {
    for mutation in 0..7 {
        let mut binding = fixture(true);
        match mutation {
            0 => binding.effect_ir_sha256[0] ^= 1,
            1 => binding.effect_ir.relations.swap(0, 1),
            2 => {
                let ReferenceValueV1::Binary { operation, .. } =
                    &mut binding.effect_ir.blocks[0].assignments[0].value
                else {
                    unreachable!()
                };
                *operation = ReferenceBinaryOpV1::Subtract;
                binding.effect_ir_sha256 = binding.effect_ir.canonical_sha256_v1();
            }
            3 => {
                binding.effect_ir.observable_output_effects[0].argument = 1;
                binding.effect_ir_sha256 = binding.effect_ir.canonical_sha256_v1();
            }
            4 => binding.observable_output_writes[0].statement = 1,
            5 => {
                binding.effect_ir.blocks[0].terminator = ReferenceTerminatorV1::Goto { target: 0 };
                binding.effect_ir_sha256 = binding.effect_ir.canonical_sha256_v1();
            }
            6 => {
                let ReferenceValueV1::Binary {
                    lhs: ReferenceOperandV1::Copy(place),
                    ..
                } = &mut binding.effect_ir.blocks[0].assignments[0].value
                else {
                    unreachable!()
                };
                place.local = 3;
                binding.effect_ir_sha256 = binding.effect_ir.canonical_sha256_v1();
            }
            _ => unreachable!(),
        }
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(31).unwrap();
        let result = binding
            .with_replayed_output_writes_v1(&mut budget, |_, _| panic!("mutated CPU MIR exposed"));
        assert!(result.is_err(), "mutation {mutation}");
        assert_eq!(budget.storage(), 31);
    }
}

#[test]
fn cpu_replay_exact_and_one_short_use_original_work_and_storage() {
    let binding = fixture(true);
    let run = |limit, storage| {
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, storage);
        budget.charge_work(17).unwrap();
        budget.reserve_storage(31).unwrap();
        let result = binding.with_replayed_output_writes_v1(&mut budget, |_, _| ());
        assert_eq!(budget.storage(), 31);
        let observed = (result, budget.work(), budget.peak_storage());
        observed
    };
    let (result, cost, peak) = run(usize::MAX, usize::MAX);
    result.unwrap();
    run(cost, peak).0.unwrap();
    assert!(run(cost - 1, peak).0.is_err());
    assert!(run(cost, peak - 1).0.is_err());
}

#[test]
fn cpu_replay_unwind_and_nested_error_release_only_owned_scratch() {
    let binding = fixture(false);
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    let result = binding
        .with_replayed_output_writes_v1(&mut budget, |_, budget| {
            budget.reserve_storage(7).unwrap();
            Err::<(), _>("consumer refusal")
        })
        .unwrap();
    assert_eq!(result, Err("consumer refusal"));
    assert_eq!(budget.storage(), 38);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = binding
                .with_replayed_output_writes_v1(&mut budget, |_, _| panic!("consumer panic"));
        }))
        .is_err()
    );
    assert_eq!(budget.storage(), 38);
}

#[test]
fn cpu_replay_rejects_callback_account_substitution_and_floor_release() {
    let binding = fixture(false);
    for substitute in [false, true] {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(31).unwrap();
        let result = binding.with_replayed_output_writes_v1(&mut budget, |_, budget| {
            if substitute {
                *budget = Budget::new(Box::leak(Box::new(Work::new(usize::MAX))), usize::MAX);
            } else {
                budget.release_storage(1).unwrap();
            }
        });
        assert!(result.is_err());
    }
}

#[test]
fn portable_entry_and_live_binding_adapter_have_identical_results_and_costs() {
    fn observe(
        binding: &AuthenticatedReferenceEffectBindingV1,
        portable_entry: bool,
        work_limit: usize,
        storage_limit: usize,
        seed_denials: bool,
    ) -> (
        Result<
            (
                Vec<ReferenceOutputWriteV1>,
                Vec<(u32, u32, ReferenceEffectExpressionV1)>,
                Vec<ResolvedReferenceBoundsCheckV1>,
            ),
            ReferenceBindingErrorV1,
        >,
        usize,
        usize,
        usize,
        Option<usize>,
        Option<usize>,
    ) {
        let mut work = Work::new(work_limit);
        let mut budget = Budget::new(&mut work, storage_limit);
        budget.charge_work(17).unwrap();
        budget.reserve_storage(31).unwrap();
        if seed_denials {
            assert!(budget.charge_work(usize::MAX).is_err());
            assert!(budget.reserve_storage(usize::MAX).is_err());
        }
        let account = budget.work_ledger_identity_v1();
        let consume = |replay: &ReplayedCpuEffectsV1, budget: &mut Budget<'_>| {
            assert!(budget.work_ledger_identity_v1() == account);
            (
                replay.writes.clone(),
                replay
                    .values
                    .iter()
                    .map(|value| (value.block, value.statement, value.expression.clone()))
                    .collect(),
                replay.bounds.clone(),
            )
        };
        let result = if portable_entry {
            portable::with_replayed_output_writes_v1(
                ReferenceReplayInputV1 {
                    signature_preimage: &binding.signature_preimage,
                    effect_ir: &binding.effect_ir,
                    effect_ir_sha256: binding.effect_ir_sha256,
                    observable_output_writes: &binding.observable_output_writes,
                },
                &mut budget,
                consume,
            )
        } else {
            binding.with_replayed_output_writes_v1(&mut budget, consume)
        };
        (
            result,
            budget.work(),
            budget.peak_storage(),
            budget.storage(),
            budget.failed_work(),
            budget.failed_storage(),
        )
    }

    for fixture in [fixture(false), fixture(true), bounded_fixture()] {
        for mutation in 0..4 {
            let mut binding = fixture.clone();
            match mutation {
                0 => {}
                1 => binding.effect_ir_sha256[0] ^= 1,
                2 => binding.observable_output_writes[0].statement += 1,
                3 => {
                    binding.effect_ir.blocks[0].terminator =
                        ReferenceTerminatorV1::Goto { target: 0 };
                    binding.effect_ir_sha256 = binding.effect_ir.canonical_sha256_v1();
                }
                _ => unreachable!(),
            }
            let live = observe(&binding, false, usize::MAX, usize::MAX, false);
            assert_eq!(live, observe(&binding, true, usize::MAX, usize::MAX, false));
            assert_eq!(live.3, 31);
            assert_eq!(live.0.is_ok(), mutation == 0);
            let seeded = observe(&binding, false, usize::MAX, usize::MAX, true);
            assert_eq!(
                seeded,
                observe(&binding, true, usize::MAX, usize::MAX, true)
            );
            assert_eq!(
                (&seeded.0, seeded.1, seeded.2, seeded.3),
                (&live.0, live.1, live.2, live.3)
            );
            assert_eq!((seeded.4, seeded.5), (Some(usize::MAX), Some(usize::MAX)));
            if mutation == 0 {
                let (work, storage) = (live.1, live.2);
                for (work, storage) in [(work, storage), (work - 1, storage), (work, storage - 1)] {
                    assert_eq!(
                        observe(&binding, false, work, storage, false),
                        observe(&binding, true, work, storage, false),
                    );
                }
            }
        }
    }
}

fn local(local: u32) -> ReferencePlaceV1 {
    ReferencePlaceV1 {
        local,
        projection: Box::default(),
    }
}

fn copy(local_id: u32) -> ReferenceOperandV1 {
    ReferenceOperandV1::Copy(local(local_id))
}

// Recompute descriptive fixture records, never an authenticated source binding.
fn refresh(binding: &mut AuthenticatedReferenceEffectBindingV1) {
    let writes = binding
        .effect_ir
        .observable_output_writes_v1(&ReferenceExtractionWorkV1::Inspection)
        .unwrap();
    binding.effect_ir.observable_output_effects = writes.clone().into_boxed_slice();
    binding.observable_output_writes = writes.into_boxed_slice();
    binding.effect_ir_sha256 = binding.effect_ir.canonical_sha256_v1();
}

fn bounded_fixture() -> AuthenticatedReferenceEffectBindingV1 {
    let mut binding = fixture(true);
    let mut output = binding.effect_ir.blocks[0].clone();
    output.block = 2;
    let bound = |block, raw, len, condition| ReferenceBlockV1 {
        block,
        assignments: vec![
            ReferenceAssignmentV1 {
                statement: 0,
                destination: local(len),
                value: ReferenceValueV1::InputLength {
                    reference_argument: raw,
                },
            },
            ReferenceAssignmentV1 {
                statement: 1,
                destination: local(condition),
                value: ReferenceValueV1::Binary {
                    operation: ReferenceBinaryOpV1::LessThan,
                    lhs: copy(1),
                    rhs: copy(len),
                    checked: false,
                },
            },
        ]
        .into_boxed_slice(),
        terminator: ReferenceTerminatorV1::Assert {
            condition: copy(condition),
            expected: true,
            success: block + 1,
            bounds_check: Some(ReferenceBoundsCheckV1 {
                index: copy(1),
                length: copy(len),
            }),
        },
    };
    binding.effect_ir.local_count = 9;
    binding.effect_ir.blocks =
        vec![bound(0, 1, 5, 6), bound(1, 2, 7, 8), output].into_boxed_slice();
    refresh(&mut binding);
    binding
}

fn check_cpu_reads(
    binding: &AuthenticatedReferenceEffectBindingV1,
    replay: &ReplayedCpuEffectsV1,
    budget: &mut Budget<'_>,
) -> Result<(), crate::production_reference_effect_join_v2::ProductionReferenceEffectJoinErrorV2> {
    use crate::production_reference_effect_join_v2::conditional_source_v1::read_premises_v1::check_replay_v1;
    check_replay_v1(&binding.effect_ir, replay, budget, |raw, budget| {
        budget.charge_work(3).unwrap();
        assert!(raw == 1 || raw == 2, "point ABI argument is not an input");
        Ok(())
    })
}

#[test]
fn cpu_read_premises_replay_bounds_and_raw_arguments_before_consumer() {
    // This tests the consuming replay algorithm, not source or proof authority.
    for binding in [fixture(false), bounded_fixture()] {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(31).unwrap();
        let account = budget.work_ledger_identity_v1();
        binding
            .with_replayed_output_writes_v1(&mut budget, |replay, budget| {
                check_cpu_reads(&binding, replay, budget).unwrap();
                if !replay.bounds.is_empty() {
                    assert_eq!(replay.bounds.len(), 2);
                    assert_eq!(replay.values.len(), 5);
                    assert_eq!(replay.writes[0].argument, 2);
                    assert_eq!(
                        replay.bounds[0].length,
                        ReferenceEffectExpressionV1::InputLength {
                            reference_argument: 1
                        }
                    );
                    assert_eq!(
                        replay.bounds[1].length,
                        ReferenceEffectExpressionV1::InputLength {
                            reference_argument: 2
                        }
                    );
                }
                assert!(budget.work_ledger_identity_v1() == account);
                budget.reserve_storage(7).unwrap();
            })
            .unwrap();
        assert_eq!(budget.storage(), 38);
    }
}

#[test]
fn cpu_read_premises_refuse_missing_wrong_late_and_unmatched_assertions() {
    for mutation in 0..9 {
        let mut binding = bounded_fixture();
        match mutation {
            0 => binding.effect_ir.blocks[0].terminator = ReferenceTerminatorV1::Goto { target: 1 },
            1 => {
                let ReferenceTerminatorV1::Assert { expected, .. } =
                    &mut binding.effect_ir.blocks[0].terminator
                else {
                    unreachable!()
                };
                *expected = false;
            }
            2 => {
                let ReferenceValueV1::Binary { operation, .. } =
                    &mut binding.effect_ir.blocks[0].assignments[1].value
                else {
                    unreachable!()
                };
                *operation = ReferenceBinaryOpV1::LessEqual;
            }
            3 => {
                binding.effect_ir.blocks[0].assignments[0].value = ReferenceValueV1::InputLength {
                    reference_argument: 2,
                }
            }
            4 => {
                let ReferenceTerminatorV1::Assert {
                    bounds_check: Some(check),
                    ..
                } = &mut binding.effect_ir.blocks[0].terminator
                else {
                    unreachable!()
                };
                check.length = copy(7);
            }
            5 => {
                binding.effect_ir.local_count = 10;
                let ReferenceValueV1::Binary { lhs, .. } =
                    &binding.effect_ir.blocks[2].assignments[0].value
                else {
                    unreachable!()
                };
                let read = ReferenceAssignmentV1 {
                    statement: 2,
                    destination: local(9),
                    value: ReferenceValueV1::Use(lhs.clone()),
                };
                let mut assignments = binding.effect_ir.blocks[0].assignments.to_vec();
                assignments.push(read);
                binding.effect_ir.blocks[0].assignments = assignments.into_boxed_slice();
            }
            6 => {
                let ReferenceValueV1::Binary { rhs, .. } =
                    &binding.effect_ir.blocks[2].assignments[0].value
                else {
                    unreachable!()
                };
                binding.effect_ir.blocks[2].assignments[0].value =
                    ReferenceValueV1::Use(rhs.clone());
            }
            7 => {
                binding.effect_ir.local_count = 10;
                let mut store = binding.effect_ir.blocks[2].assignments[0].clone();
                store.statement = 1;
                let ReferenceValueV1::Binary {
                    lhs: ReferenceOperandV1::Copy(place),
                    ..
                } = &mut store.value
                else {
                    unreachable!()
                };
                place.projection[1] = ReferencePlaceProjectionV1::Index(9);
                binding.effect_ir.blocks[2].assignments = vec![
                    ReferenceAssignmentV1 {
                        statement: 0,
                        destination: local(9),
                        value: ReferenceValueV1::Use(ReferenceOperandV1::Constant(
                            ReferenceConstantV1::Scalar {
                                scalar: ReferenceScalarTypeV1::Usize,
                                bits: 1,
                            },
                        )),
                    },
                    store,
                ]
                .into_boxed_slice();
            }
            8 => binding.effect_ir.blocks[0].terminator = ReferenceTerminatorV1::Goto { target: 2 },
            _ => unreachable!(),
        }
        refresh(&mut binding);
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(31).unwrap();
        let result = binding
            .with_replayed_output_writes_v1(&mut budget, |replay, budget| {
                check_cpu_reads(&binding, replay, budget)
            })
            .unwrap();
        assert!(result.is_err(), "CPU bounds mutation {mutation}");
        assert_eq!(budget.storage(), 31);
    }
}

#[test]
fn cpu_read_premises_refuse_nonbounds_traps_and_branches_after_the_output() {
    for branch in [false, true] {
        let mut binding = fixture(false);
        let condition = ReferenceOperandV1::Constant(ReferenceConstantV1::Scalar {
            scalar: ReferenceScalarTypeV1::Bool,
            bits: 0,
        });
        binding.effect_ir.blocks[0].terminator = if branch {
            ReferenceTerminatorV1::Switch {
                discriminant: condition,
                values: Box::default(),
                otherwise: 1,
            }
        } else {
            ReferenceTerminatorV1::Assert {
                condition,
                expected: true,
                success: 1,
                bounds_check: None,
            }
        };
        let mut blocks = binding.effect_ir.blocks.to_vec();
        blocks.push(ReferenceBlockV1 {
            block: 1,
            assignments: Box::default(),
            terminator: ReferenceTerminatorV1::Return,
        });
        binding.effect_ir.blocks = blocks.into_boxed_slice();
        refresh(&mut binding);
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        assert!(
            binding
                .with_replayed_output_writes_v1(&mut budget, |replay, budget| {
                    check_cpu_reads(&binding, replay, budget)
                })
                .unwrap()
                .is_err()
        );
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn cpu_read_premises_cannot_treat_expression_only_reads_as_safe_cpu_mir() {
    let binding = fixture(true);
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    assert!(
        binding
            .with_replayed_output_writes_v1(&mut budget, |replay, budget| {
                check_cpu_reads(&binding, replay, budget)
            })
            .unwrap()
            .is_err()
    );
}

#[test]
fn cpu_read_premises_original_budget_exhaustion_cannot_reach_consumer() {
    use crate::production_reference_effect_join_v2::conditional_source_v1::read_premises_v1::check_replay_v1;
    let binding = bounded_fixture();
    let run = |limit, storage| {
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, storage);
        budget.charge_work(17).unwrap();
        budget.reserve_storage(31).unwrap();
        let mut consumed = false;
        let result = binding.with_replayed_output_writes_v1(&mut budget, |replay, budget| {
            check_replay_v1(&binding.effect_ir, replay, budget, |_, _| Ok(()))?;
            consumed = true;
            Ok::<(), crate::production_reference_effect_join_v2::ProductionReferenceEffectJoinErrorV2>(())
        });
        let ok = matches!(result, Ok(Ok(())));
        assert_eq!(ok, consumed);
        assert_eq!(budget.storage(), 31);
        (ok, budget.work(), budget.peak_storage())
    };
    let (ok, work, storage) = run(usize::MAX, usize::MAX);
    assert!(ok);
    assert!(run(work, storage).0);
    assert!(!run(work - 1, storage).0);
    assert!(!run(work, storage - 1).0);
}

#[test]
fn cpu_read_premises_checked_replay_keeps_error_unwind_and_account_cleanup() {
    let binding = bounded_fixture();
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    let result = binding
        .with_replayed_output_writes_v1(&mut budget, |replay, budget| {
            check_cpu_reads(&binding, replay, budget).unwrap();
            budget.reserve_storage(7).unwrap();
            Err::<(), _>("later conditional consumer refused")
        })
        .unwrap();
    assert_eq!(result, Err("later conditional consumer refused"));
    assert_eq!(budget.storage(), 38);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = binding.with_replayed_output_writes_v1(&mut budget, |replay, budget| {
                check_cpu_reads(&binding, replay, budget).unwrap();
                panic!("later conditional consumer panicked");
            });
        }))
        .is_err()
    );
    assert_eq!(budget.storage(), 38);
    for substitute in [false, true] {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(31).unwrap();
        let result = binding.with_replayed_output_writes_v1(&mut budget, |replay, budget| {
            check_cpu_reads(&binding, replay, budget).unwrap();
            if substitute {
                *budget = Budget::new(Box::leak(Box::new(Work::new(usize::MAX))), usize::MAX);
            } else {
                budget.release_storage(1).unwrap();
            }
        });
        assert!(result.is_err());
    }
}

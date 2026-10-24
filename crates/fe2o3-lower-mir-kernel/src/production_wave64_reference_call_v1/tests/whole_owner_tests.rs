fn actual_raw_shuffle_whole_owner() -> ProductionSemanticKirOwnerV1 {
    let source =
        fixture(Mutation::RawShuffle).expect("declared raw-shuffle semantic source must admit");
    assert!(source.source_semantic().callables().iter().any(|callable|matches!(callable,
        SemanticCallableDeclV1::CompilerIntrinsic {operation:SemanticCompilerIntrinsicOperationV1::Gfx942Wave64ShuffleIndex {..},..})));
    let semantic = source
        .into_source_owner()
        .expect("original SSA replay before normal owner construction");
    // Use the normal public constructor. A missing capture/transport step is
    // an actual test failure; no privately assembled owner may replace it.
    let owner =
        ProductionSemanticKirOwnerV1::try_lower(semantic, ProductionSemanticKirLimitsV1::default())
            .expect("normal source-to-whole-owner raw-shuffle construction");
    owner
        .verify_equivalence()
        .expect("untouched whole-owner original-source replay");
    assert_eq!(
        owner.module().kernels[0].workgroup_size,
        Some(WorkgroupSize::new(64, 1, 1))
    );
    assert!(
        owner
            .module()
            .functions
            .iter()
            .flat_map(|f| f.body.iter())
            .flat_map(|b| &b.blocks)
            .flat_map(|b| &b.operations)
            .any(|op| matches!(
                op.kind,
                OperationKind::Wave(WaveOperation {
                    kind: WaveOperationKind::ShuffleIndex { .. },
                    ..
                })
            ))
    );
    owner
}
#[test]
fn genuine_declared_raw_shuffle_constructs_and_replays_normal_whole_owner() {
    drop(actual_raw_shuffle_whole_owner());
}
#[derive(Clone, Copy, Debug)]
enum WholeOwnerFault {
    Lane,
    Element,
    EntryEdge,
    Result,
    Continuation,
}
#[test]
fn coherent_raw_shuffle_graph_changes_fail_original_whole_owner_replay() {
    for fault in [
        WholeOwnerFault::Lane,
        WholeOwnerFault::Element,
        WholeOwnerFault::EntryEdge,
        WholeOwnerFault::Result,
        WholeOwnerFault::Continuation,
    ] {
        // Untouched public construction and whole-owner replay must succeed
        // before any negative mutation is permitted to count as evidence.
        let mut owner = actual_raw_shuffle_whole_owner();
        let mut module = owner.module().clone();
        let root = module
            .functions
            .iter()
            .position(|f| f.role == fe2o3_kernel_ir::FunctionRole::KernelEntry)
            .unwrap();
        let helper = module
            .functions
            .iter()
            .position(|f| f.role == fe2o3_kernel_ir::FunctionRole::InternalHelper)
            .unwrap();
        match fault {
            WholeOwnerFault::Lane | WholeOwnerFault::Element => {
                let body = module.functions[root].body.as_mut().unwrap();
                let block = body.blocks.iter_mut().find(|b| b.id == BlockId(2)).unwrap();
                let alternative = block
                    .operations
                    .iter()
                    .find_map(|op| match (&fault, &op.kind) {
                        (WholeOwnerFault::Lane, OperationKind::Constant(Constant::U32(7))) => {
                            Some(op.results[0].id)
                        }
                        (
                            WholeOwnerFault::Element,
                            OperationKind::Constant(Constant::F32Bits(bits)),
                        ) if *bits == 0x40000000 => Some(op.results[0].id),
                        _ => None,
                    })
                    .expect("distinct genuine dominating source value");
                let call = block
                    .operations
                    .iter_mut()
                    .find(|op| matches!(op.kind, OperationKind::Call { .. }))
                    .unwrap();
                let OperationKind::Call { arguments, .. } = &mut call.kind else {
                    unreachable!()
                };
                let ordinal = usize::from(matches!(fault, WholeOwnerFault::Element));
                assert_ne!(arguments[ordinal], alternative);
                arguments[ordinal] = alternative;
            }
            WholeOwnerFault::EntryEdge => {
                let block = module.functions[root]
                    .body
                    .as_mut()
                    .unwrap()
                    .blocks
                    .iter_mut()
                    .find(|b| b.id == BlockId(0))
                    .unwrap();
                let Some(Terminator::Branch { target, .. }) = &mut block.terminator else {
                    panic!("source issuer normal entry edge")
                };
                assert_eq!(*target, BlockId(1));
                *target = BlockId(2);
            }
            WholeOwnerFault::Result => {
                let function = &mut module.functions[helper];
                let input = function.body.as_ref().unwrap().parameters[1];
                let block = function
                    .body
                    .as_mut()
                    .unwrap()
                    .blocks
                    .iter_mut()
                    .find(|b| b.id == BlockId(1))
                    .unwrap();
                let Some(Terminator::Return { values }) = &mut block.terminator else {
                    panic!("actual shuffle result return")
                };
                assert_eq!(values.len(), 1);
                assert_ne!(values[0], input);
                values[0] = input;
            }
            WholeOwnerFault::Continuation => {
                let block = module.functions[root]
                    .body
                    .as_mut()
                    .unwrap()
                    .blocks
                    .iter_mut()
                    .find(|b| b.id == BlockId(2))
                    .unwrap();
                let Some(Terminator::Branch { target, .. }) = &mut block.terminator else {
                    panic!("source Defined-call normal continuation")
                };
                assert_eq!(*target, BlockId(3));
                *target = BlockId(4);
            }
        }
        verify_module(&module).unwrap_or_else(|e| panic!("{fault:?} must remain typed KIR: {e:?}"));
        // Keep canonical graph bytes coherent with the changed module.
        // The immutable semantic source/SSA owner is deliberately untouched.
        owner.canonical_kernel_ir = ProductionCanonicalKernelIrV1::from_module(module.clone())
            .unwrap_or_else(|e| panic!("{fault:?} must retain canonical graph admission: {e:?}"));
        owner.module = RetainedProductionKirModuleV1::Legacy(module);
        owner.canonical_kernel_ir.revalidate().unwrap();
        assert!(
            matches!(
                owner.verify_equivalence(),
                Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch)
            ),
            "{fault:?} escaped full immutable-source owner replay"
        );
    }
}

#[test]
fn non_raw_captured_source_keeps_legacy_helper_and_parameter_accounting() {
    let owner = fixture(Mutation::None).expect("ordinary admitted source owner");
    assert!(owner.occurrence_storage().is_some());
    assert!(!super::super::ordinary_source_has_raw_wave64_shuffle_v1(
        owner.source_semantic()
    ));
    let mut work = Work::new(0);
    let mut budget = Budget::new(&mut work, 0);
    let mut closure = super::super::ReachableClosureBudgetV1::new(0);
    let plan = super::super::checked_wave64_helper_parameter_plan_v1(
        &owner,
        SemanticFunctionIdV1::from_index(0),
        SemanticFunctionIdV1::from_index(1),
        FunctionId::new("ordinary_non_raw_control"),
        0,
        &mut closure,
        &mut budget,
    )
    .expect("non-raw source must not query the nominal helper view");
    assert!(plan.is_none());
    assert_eq!(budget.work(), 0);
    assert_eq!(budget.storage(), 0);
    assert!(budget.failed_work().is_none());
    assert!(budget.failed_storage().is_none());

    let target = Function::internal_helper(
        FunctionId::new("ordinary_non_raw_control"),
        Signature::new(
            vec![Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::F32)],
            vec![Type::Scalar(ScalarType::F32)],
        ),
        vec![ValueId(1), ValueId(3)],
        vec![],
    );
    let instance = super::super::SemanticKirFunctionCorrespondenceV1 {
        correspondence_owner: SemanticFunctionIdV1::from_index(0),
        semantic_function: SemanticFunctionIdV1::from_index(1),
        kernel_ir_function: target.id.clone(),
        role: super::super::SemanticKirFunctionRoleV1::InternalHelper,
    };
    let mut direct_work = Work::new(10_000_000);
    let mut wrapped_work = Work::new(10_000_000);
    let mut direct_budget = Budget::new(&mut direct_work, 10_000_000);
    let mut wrapped_budget = Budget::new(&mut wrapped_work, 10_000_000);
    let direct = super::super::validate_parameter_correspondence_v1(
        owner.source_semantic(),
        &instance,
        &target,
        super::super::ArgumentTraceV1 {
            direct: &[],
            components: &[],
            ignored: &[],
        },
        &mut direct_budget,
    );
    let wrapped = super::super::validate_parameter_correspondence_with_wave64_v1(
        &owner,
        &instance,
        &target,
        super::super::ArgumentTraceV1 {
            direct: &[],
            components: &[],
            ignored: &[],
        },
        &mut wrapped_budget,
    );
    assert_eq!(format!("{direct:?}"), format!("{wrapped:?}"));
    assert_eq!(direct_budget.work(), wrapped_budget.work());
    assert_eq!(direct_budget.storage(), wrapped_budget.storage());
    assert_eq!(direct_budget.failed_work(), wrapped_budget.failed_work());
    assert_eq!(
        direct_budget.failed_storage(),
        wrapped_budget.failed_storage()
    );
}

#[test]
fn retained_source_input_refunds_actual_receipt_on_success_and_error() {
    let owner = fixture(Mutation::RawShuffle).expect("genuine admitted raw-shuffle source owner");
    assert!(super::super::ordinary_source_has_raw_wave64_shuffle_v1(
        owner.source_semantic()
    ));
    let receipt = owner
        .occurrence_storage()
        .expect("real captured input receipt");
    let held = receipt.retained_storage();
    assert!(held > 0);
    for refuse in [false, true] {
        let mut work = Work::new(100);
        let mut budget = Budget::new(&mut work, 11 + held + 7);
        budget.reserve_storage(11).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let mut entered = false;
        let result =
            super::super::with_wave64_retained_source_input_v1(&owner, &mut budget, |budget| {
                entered = true;
                assert_eq!(budget.storage(), 11 + held);
                budget.reserve_storage(7)?;
                // The body owns this distinct output/scratch charge. On failure
                // it drops/refunds its scratch before returning; success retains it.
                if refuse {
                    budget.release_storage(7)?;
                    return Err(super::super::ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                Ok(29u32)
            });
        assert!(entered);
        if refuse {
            assert!(matches!(
                result,
                Err(super::super::ProductionSemanticKirErrorV1::CorrespondenceMismatch)
            ));
        } else {
            assert!(matches!(result, Ok(29)));
        }
        assert_eq!(budget.storage(), 11 + if refuse { 0 } else { 7 });
        assert!(budget.work_ledger_identity_v1() == ledger);
        assert!(budget.failed_work().is_none());
        assert!(budget.failed_storage().is_none());
        assert_eq!(owner.occurrence_storage().unwrap().retained_storage(), held);
    }
}

#[test]
fn retained_source_input_one_under_refuses_before_callback_and_preserves_failure_history() {
    let owner = fixture(Mutation::RawShuffle).expect("genuine admitted raw-shuffle source owner");
    let held = owner.occurrence_storage().unwrap().retained_storage();
    assert!(held > 0);
    let mut work = Work::new(100);
    let mut budget = Budget::new(&mut work, 11 + held - 1);
    budget.reserve_storage(11).unwrap();
    let mut entered = false;
    let result = super::super::with_wave64_retained_source_input_v1(&owner, &mut budget, |_| {
        entered = true;
        Ok(())
    });
    assert!(matches!(
        result,
        Err(
            super::super::ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                Resource::Storage(_)
            )
        )
    ));
    assert!(!entered);
    assert_eq!(budget.storage(), 11);
    assert_eq!(budget.failed_storage(), Some(11 + held));
    assert!(budget.failed_work().is_none());
    drop(budget);
    let mut work = Work::new(0);
    let mut budget = Budget::new(&mut work, 11 + held);
    budget.reserve_storage(11).unwrap();
    let mut entered = false;
    let result =
        super::super::with_wave64_retained_source_input_v1(&owner, &mut budget, |budget| {
            entered = true;
            budget.charge_work(1)?;
            Ok(())
        });
    assert!(matches!(
        result,
        Err(
            super::super::ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                Resource::Work(_)
            )
        )
    ));
    assert!(entered);
    assert_eq!(budget.storage(), 11);
    assert_eq!(budget.failed_work(), Some(1));
    assert!(budget.failed_storage().is_none());
}

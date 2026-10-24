use super::production_call_instances_v1::with_production_call_instances_v1;
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1;

const LIMIT: usize = 1_000_000;
const FLOOR: usize = 29;
const ROOT: SemanticFunctionIdV1 = SemanticFunctionIdV1::from_index(0);

// This returns original source state, not a detached or resumable lowerer. The
// independent lifetimes make retaining a mutable service borrow impossible.
fn finish_then_release_services<'source, 'service, 'work>(
    instances: &ExecutionInstancesV29<'source>,
    cursor: ExecutionAvailabilityV29<'source>,
    private: &'service mut PrivateArrayLazyBudgetV1,
    budget: &'service mut ArgumentBudgetV1<'work>,
) -> Result<ExecutionAvailabilityV29<'source>, ProductionSemanticKirErrorV1> {
    let row = instances.instance(cursor.instance).unwrap();
    let semantic = instances.owner().source_semantic();
    let function = row.declaration();
    assert_eq!(row.function().index(), 1);
    assert_eq!(function.blocks().len(), 1);
    assert!(matches!(
        function.blocks()[0].terminator().kind(),
        SemanticTerminatorKindV1::Return
    ));
    let ids = BTreeMap::new();
    let signatures = BTreeMap::new();
    let returns =
        CallReturnBufferV1::for_function(function, semantic.callables(), &signatures, 0, budget)?;
    let mut lowering: SemanticFunctionLoweringV1<'source, 'service> =
        SemanticFunctionLoweringV1::new_interprocedural(
            semantic.types(),
            semantic.callables(),
            function,
            row.ssa(),
            ROOT,
            row.function(),
            ids,
            signatures,
            Vec::new(),
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
                local_bindings: Some(&[]),
            },
            None,
            Some([64, 1, 1]),
            BTreeSet::new().into(),
            1,
            false,
            256,
            PrivateArrayRecorderWorkV1::Shared(private),
            None,
            returns,
            Some(budget),
            SemanticEmissionPlacementV1::default(),
            Some(cursor),
            None,
            None,
)?;
    assert!(std::ptr::eq(lowering.function, function));
    assert!(std::ptr::eq(lowering.types, semantic.types()));
    assert!(std::ptr::eq(lowering.callables, semantic.callables()));
    assert!(matches!(
        &lowering.private_arrays.work,
        PrivateArrayRecorderWorkV1::Shared(_)
    ));
    let mut target = BasicBlock::new(BlockId(0));
    lowering.begin_block(function.entry(), &mut target)?;
    target.terminator = Some(lowering.lower_terminator(
        function.entry(),
        function.blocks()[0].terminator().kind(),
        &mut target.operations,
    )?);
    assert!(target.operations.is_empty());
    assert!(matches!(
        target.terminator.as_ref(),
        Some(Terminator::Return { values }) if values.is_empty()
    ));
    lowering.with_emission_budget_v1(|this, budget| {
        let cursor = this.execution.as_mut().unwrap();
        cursor.finish_block(budget)?;
        cursor.finish(budget)
    })?;
    let cursor = lowering.execution.take().unwrap();
    drop(lowering);
    Ok(cursor)
}

#[test]
fn actual_original_cursor_outlives_short_mutable_emission_services() {
    for prior_denial in [false, true] {
        let mut owner = ProductionSemanticSsaOwnerV1::try_new(
            resource_tests::helper_closure_semantic_owner(),
            ProductionSemanticSsaLimitsV1::default(),
        )
        .unwrap();
        let mut work = CanonicalKernelIrWorkBudgetV1::new(LIMIT);
        if prior_denial {
            assert!(work.charge_work(LIMIT + 1).is_err());
        }
        let mut budget = ArgumentBudgetV1::new(&mut work, LIMIT);
        if prior_denial {
            assert!(budget.reserve_storage(LIMIT + 1).is_err());
        }
        budget.reserve_storage(FLOOR).unwrap();
        let capture = owner
            .try_capture_occurrences_with_budget_v1(&mut budget)
            .unwrap();
        budget.reserve_storage(capture.retained_storage()).unwrap();
        let mut reached = false;
        let result = with_production_call_instances_v1(
            &owner,
            ROOT,
            &mut budget,
            |instances, budget| -> Result<(), ProductionSemanticKirErrorV1> {
                let floor = budget.storage();
                let child = instances.calls(instances.root()).unwrap()[0]
                    .child()
                    .unwrap();
                let original = instances.instance(child).unwrap();
                let cursor = ExecutionAvailabilityV29::new(instances, child, budget)?;
                let mut private = PrivateArrayLazyBudgetV1::new(1, 256);
                let cursor = finish_then_release_services(instances, cursor, &mut private, budget)?;
                // Both services are mutable again while this exact cursor lives.
                budget.reserve_storage(11)?;
                budget.release_storage(11)?;
                private.activate()?;
                private.charge_private_array_work(1)?;
                cursor.check_source(original.declaration(), original.ssa())?;
                cursor.check_ledger(budget)?;
                assert_eq!(cursor.instance, child);
                assert_eq!(cursor.visited, vec![true]);
                assert!(std::ptr::eq(cursor.function, original.declaration()));
                assert!(std::ptr::eq(cursor.ssa, original.ssa()));
                let root = instances.instance(instances.root()).unwrap();
                assert!(cursor.check_source(root.declaration(), root.ssa()).is_err());
                let cloned = original.declaration().clone();
                assert!(cursor.check_source(&cloned, original.ssa()).is_err());
                let mut other_work = CanonicalKernelIrWorkBudgetV1::new(LIMIT);
                let other = ArgumentBudgetV1::new(&mut other_work, LIMIT);
                assert!(cursor.check_ledger(&other).is_err());
                assert_eq!(other.storage(), 0);
                drop(other);
                assert_eq!(other_work.work(), 0);
                drop(cursor);
                budget.release_storage(budget.storage() - floor)?;
                assert_eq!(budget.storage(), floor);
                reached = true;
                Ok(())
            },
        );
        assert!(result.is_ok(), "{result:?}");
        assert!(reached);
        drop(owner);
        budget.release_storage(capture.retained_storage()).unwrap();
        assert_eq!(budget.storage(), FLOOR);
        assert_eq!(budget.failed_storage(), prior_denial.then_some(LIMIT + 1));
        drop(budget);
        assert!(work.work() > 0);
        assert_eq!(work.failed_work(), prior_denial.then_some(LIMIT + 1));
    }
}

// Test only the recorder's source/service ownership separation. A manually
// retained expected-place row is not a physical array access certificate.
fn retain_expected_place<'source, 'service>(
    place: &'source SemanticPlaceV1,
    work: &'service mut PrivateArrayLazyBudgetV1,
    prior_denial: bool,
) -> Result<PrivateArrayExpectedPlaceV1<'source>, ProductionSemanticKirErrorV1> {
    let mut recorder: PrivateArrayFunctionRecorderV1<'source, 'service> =
        PrivateArrayFunctionRecorderV1::new(
            PrivateArrayRecorderWorkV1::Shared(work),
            true,
            32,
            PrivateArrayPayloadV1::default(),
            SemanticEmissionPlacementV1::default(),
        );
    recorder.work.activate()?;
    recorder.expected.reserve(1, 32, &mut recorder.work)?;
    recorder.expected.rows.push(PrivateArrayExpectedPlaceV1 {
        place,
        role: fe2o3_pliron::ProductionSemanticSsaOperandRoleV1::Destination,
        initializer_component: None,
    });
    if prior_denial {
        assert!(recorder.work.charge_private_array_work(257).is_err());
    }
    let retained = recorder.expected.rows.pop().unwrap();
    drop(recorder);
    Ok(retained)
}

#[test]
fn recorder_source_place_outlives_service_without_resetting_sticky_work() {
    let owner = ProductionSemanticSsaOwnerV1::try_new(
        resource_tests::helper_closure_semantic_owner(),
        ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap();
    let semantic = owner.source_semantic();
    let SemanticTerminatorKindV1::Call(call) =
        semantic.functions()[0].blocks()[0].terminator().kind()
    else {
        panic!("original root call")
    };
    let place = call.destination().unwrap().place();
    for prior_denial in [false, true] {
        let mut work = PrivateArrayLazyBudgetV1::new(1, 256);
        let retained = retain_expected_place(place, &mut work, prior_denial).unwrap();
        assert!(std::ptr::eq(retained.place, place));
        assert_eq!(
            retained.role,
            fe2o3_pliron::ProductionSemanticSsaOperandRoleV1::Destination
        );
        assert_eq!(retained.initializer_component, None);
        let before = work.active.as_ref().unwrap().work.work();
        let first = work.active.as_ref().unwrap().first_denial;
        assert_eq!(first.is_some(), prior_denial);
        let result = work.charge_private_array_work(1);
        assert_eq!(result.is_ok(), !prior_denial);
        assert_eq!(work.active.as_ref().unwrap().first_denial, first);
        assert_eq!(
            work.active.as_ref().unwrap().work.work(),
            before + usize::from(!prior_denial)
        );
        assert!(std::ptr::eq(
            retained.place,
            call.destination().unwrap().place()
        ));
    }
}

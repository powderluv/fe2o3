use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

type Ids = BTreeMap<SemanticFunctionIdV1, FunctionId>;
type Signatures = BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>;
const WORK: usize = 1_000_000;
const STORAGE: usize = 1_000_000;
const FLOOR: usize = 19;

fn source() -> ProductionSemanticSsaOwnerV1 {
    ProductionSemanticSsaOwnerV1::try_new(
        resource_tests::helper_closure_semantic_owner(),
        ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap()
}

fn tables(ssa: &ProductionSemanticSsaOwnerV1) -> (Ids, Signatures) {
    let semantic = ssa.source_semantic();
    let mut ids = BTreeMap::new();
    let mut signatures = BTreeMap::new();
    for (index, function) in semantic.functions().iter().enumerate().skip(1) {
        let id = SemanticFunctionIdV1::from_index(index as u32);
        ids.insert(id, helper_function_id_v1(id, function));
        signatures.insert(
            id,
            LoweredFunctionSignatureV1 {
                bf16_nominal: false,
                wave64_nominal: false,
                parameter_semantic_types: vec![],
                call_arguments: vec![],
                parameter_types: vec![],
                result_types: vec![],
                result_semantic_type: function.abi().source_output_type(),
            },
        );
    }
    (ids, signatures)
}

fn boxed_type() -> Type {
    Type::pointer(
        Type::slice(
            Type::Scalar(ScalarType::U32),
            AddressSpace::Global,
            AccessMode::ReadWrite,
        ),
        AddressSpace::Private,
        AccessMode::ReadOnly,
    )
}

fn pointee(ty: &Type) -> &Type {
    match ty {
        Type::Pointer(pointer) => &pointer.pointee,
        _ => panic!("expected retained boxed type"),
    }
}

// This is constructor plumbing coverage only: unused signature rows are not
// presented as admitted calls or a new source/ABI fixture.
fn plumbing_tables(count: usize) -> (Ids, Signatures, Vec<Type>) {
    let mut ids = BTreeMap::new();
    let mut signatures = BTreeMap::new();
    for index in 0..count {
        let id = SemanticFunctionIdV1::from_index(index as u32 + 1);
        ids.insert(id, FunctionId::new(format!("read_only_{index}")));
        signatures.insert(
            id,
            LoweredFunctionSignatureV1 {
                bf16_nominal: false,
                wave64_nominal: false,
                parameter_semantic_types: vec![SemanticTypeIdV1::from_index(0)],
                call_arguments: vec![HelperCallArgumentV1 {
                    source_argument: index as u32,
                    tuple_field: Some(2),
                    component: Some(3),
                }],
                parameter_types: vec![boxed_type(), Type::Scalar(ScalarType::U64)],
                result_types: vec![boxed_type()],
                result_semantic_type: SemanticTypeIdV1::from_index(0),
            },
        );
    }
    let results = if count == 0 {
        vec![]
    } else {
        vec![boxed_type()]
    };
    (ids, signatures, results)
}

fn construct<'a>(
    ssa: &'a ProductionSemanticSsaOwnerV1,
    selected: SemanticFunctionIdV1,
    ids: impl Into<EmissionReadOnlyV1<'a, Ids>>,
    signatures: impl Into<EmissionReadOnlyV1<'a, Signatures>>,
    results: impl Into<EmissionReadOnlyV1<'a, Vec<Type>>>,
    budget: &'a mut ArgumentBudgetV1<'_>,
) -> Result<SemanticFunctionLoweringV1<'a, 'a>, ProductionSemanticKirErrorV1> {
    let semantic = ssa.source_semantic();
    let function = &semantic.functions()[selected.index() as usize];
    let signatures = signatures.into();
    let results = results.into();
    let call_returns = CallReturnBufferV1::for_function(
        function,
        semantic.callables(),
        &signatures,
        results.len(),
        budget,
    )?;
    SemanticFunctionLoweringV1::new_interprocedural(
        semantic.types(),
        semantic.callables(),
        function,
        ssa.plan_for_function(selected).unwrap(),
        semantic.roots()[0],
        selected,
        ids,
        signatures,
        results,
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
        PrivateArrayRecorderWorkV1::Owned(PrivateArrayLazyBudgetV1::new(1, 256)),
        None,
        call_returns,
        Some(budget),
        SemanticEmissionPlacementV1::default(),
        None,
        None,
        None,
)
}

#[test]
fn read_only_view_borrows_and_moves_non_clone_backing_without_copying() {
    struct NotClone(Box<u32>);
    let value = NotClone(Box::new(41));
    let pointer = &*value.0 as *const u32;
    {
        let first: EmissionReadOnlyV1<'_, NotClone> = (&value).into();
        let second: EmissionReadOnlyV1<'_, NotClone> = (&value).into();
        assert!(std::ptr::eq(&*first, &value));
        assert!(std::ptr::eq(&*second, &value));
        assert_eq!(&*first.0 as *const u32, pointer);
    }
    let owned: EmissionReadOnlyV1<'static, NotClone> = value.into();
    assert!(matches!(&owned, EmissionReadOnlyV1::Owned(_)));
    assert_eq!(&*owned.0 as *const u32, pointer);
    assert_eq!(*owned.0, 41);
}

#[test]
fn actual_constructor_borrows_empty_single_and_multiple_typed_tables() {
    let ssa = source();
    for count in [0, 1, 3] {
        let (ids, signatures, results) = plumbing_tables(count);
        let mut work = Work::new(WORK);
        let mut budget = ArgumentBudgetV1::new(&mut work, STORAGE);
        budget.reserve_storage(FLOOR).unwrap();
        let lowering = construct(
            &ssa,
            SemanticFunctionIdV1::from_index(1),
            &ids,
            &signatures,
            &results,
            &mut budget,
        )
        .unwrap();
        assert!(matches!(
            &lowering.defined_function_ids,
            EmissionReadOnlyV1::Borrowed(_)
        ));
        assert!(matches!(
            &lowering.defined_function_signatures,
            ExecutionSignatureSourceV29::Legacy(EmissionReadOnlyV1::Borrowed(_))
        ));
        let ExecutionSignatureSourceV29::Legacy(actual_signatures) =
            &lowering.defined_function_signatures
        else {
            panic!("legacy constructor changed signature ownership");
        };
        assert!(matches!(
            &lowering.result_types,
            EmissionReadOnlyV1::Borrowed(_)
        ));
        assert!(std::ptr::eq(&*lowering.defined_function_ids, &ids));
        assert!(std::ptr::eq(&**actual_signatures, &signatures));
        assert!(std::ptr::eq(&*lowering.result_types, &results));
        assert_eq!(lowering.defined_function_ids.len(), count);
        assert_eq!(actual_signatures.len(), count);
        assert_eq!(lowering.result_types.as_slice(), results.as_slice());
        for (id, signature) in &signatures {
            let actual = actual_signatures.get(id).unwrap();
            assert!(std::ptr::eq(actual, signature));
            assert_eq!(lowering.defined_function_ids[id], ids[id]);
            assert_eq!(actual.parameter_types, signature.parameter_types);
            assert!(std::ptr::eq(
                pointee(&actual.parameter_types[0]),
                pointee(&signature.parameter_types[0])
            ));
            assert_eq!(actual.call_arguments[0].tuple_field, Some(2));
            assert_eq!(actual.call_arguments[0].component, Some(3));
            assert_eq!(actual.result_semantic_type, signature.result_semantic_type);
        }
        if count != 0 {
            assert!(std::ptr::eq(
                pointee(&lowering.result_types[0]),
                pointee(&results[0])
            ));
        }
        drop(lowering);
        budget.release_storage(budget.storage() - FLOOR).unwrap();
        assert_eq!(budget.storage(), FLOOR);
        assert_eq!(ids.len(), count);
    }
}

#[test]
fn actual_constructor_keeps_owned_temporaries_and_boxed_results_alive() {
    let ssa = source();
    let mut work = Work::new(WORK);
    let mut budget = ArgumentBudgetV1::new(&mut work, STORAGE);
    budget.reserve_storage(FLOOR).unwrap();
    let (lowering, result_pointer, type_pointer, name_pointer) = {
        let (ids, signatures, results) = plumbing_tables(3);
        let key = SemanticFunctionIdV1::from_index(1);
        let result_pointer = results.as_ptr();
        let type_pointer = signatures[&key].parameter_types.as_ptr();
        let name_pointer = ids[&key].as_str().as_ptr();
        let lowering = construct(&ssa, key, ids, signatures, results, &mut budget).unwrap();
        (lowering, result_pointer, type_pointer, name_pointer)
    };
    let key = SemanticFunctionIdV1::from_index(1);
    assert!(matches!(
        &lowering.defined_function_ids,
        EmissionReadOnlyV1::Owned(_)
    ));
    assert!(matches!(
        &lowering.defined_function_signatures,
        ExecutionSignatureSourceV29::Legacy(EmissionReadOnlyV1::Owned(_))
    ));
    let ExecutionSignatureSourceV29::Legacy(actual_signatures) =
        &lowering.defined_function_signatures
    else {
        panic!("legacy constructor changed signature ownership");
    };
    assert!(matches!(
        &lowering.result_types,
        EmissionReadOnlyV1::Owned(_)
    ));
    assert_eq!(lowering.defined_function_ids.len(), 3);
    assert_eq!(lowering.defined_function_ids[&key].as_str(), "read_only_0");
    assert_eq!(
        lowering.defined_function_ids[&key].as_str().as_ptr(),
        name_pointer
    );
    assert_eq!(
        actual_signatures[&key].parameter_types.as_ptr(),
        type_pointer
    );
    assert_eq!(lowering.result_types.as_ptr(), result_pointer);
    assert_eq!(lowering.result_types[0], boxed_type());
    drop(lowering);
    budget.release_storage(budget.storage() - FLOOR).unwrap();
    assert_eq!(budget.storage(), FLOOR);
}

#[test]
fn repeated_instances_share_tables_and_preserve_cumulative_constructor_work() {
    let ssa = source();
    let (ids, signatures, results) = plumbing_tables(3);
    let mut work = Work::new(WORK);
    let mut budget = ArgumentBudgetV1::new(&mut work, STORAGE);
    budget.reserve_storage(FLOOR).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let mut expected = None;
    for iteration in 0..4 {
        let before_work = budget.work();
        let lowering = construct(
            &ssa,
            SemanticFunctionIdV1::from_index(1),
            &ids,
            &signatures,
            &results,
            &mut budget,
        )
        .unwrap();
        let ExecutionSignatureSourceV29::Legacy(actual_signatures) =
            &lowering.defined_function_signatures
        else {
            panic!("legacy constructor changed signature ownership");
        };
        assert!(std::ptr::eq(&**actual_signatures, &signatures));
        assert!(std::ptr::eq(&*lowering.result_types, &results));
        drop(lowering);
        let delta = (budget.work() - before_work, budget.storage() - FLOOR);
        assert!(delta.0 > 0);
        match expected {
            None => expected = Some(delta),
            Some(expected) => assert_eq!(delta, expected),
        }
        assert_eq!(budget.work(), delta.0 * (iteration + 1));
        assert!(budget.work_ledger_identity_v1() == ledger);
        budget.release_storage(delta.1).unwrap();
        assert_eq!(budget.storage(), FLOOR);
    }
}

#[derive(Debug, PartialEq)]
struct Emission {
    function: Result<Function, String>,
    resource: Option<ArgumentResourceV1>,
    work: usize,
    peak: usize,
    storage_before_drop_cleanup: usize,
}

// Exercise the same constructor and real block/terminator emitter with both
// input representations. The existing admitted fixture has no statements.
fn emit(
    ssa: &ProductionSemanticSsaOwnerV1,
    selected: SemanticFunctionIdV1,
    owned: bool,
    work_limit: usize,
    storage_limit: usize,
) -> Emission {
    let (ids, signatures) = tables(ssa);
    let results = Vec::new();
    let mut work = Work::new(work_limit);
    let mut budget = ArgumentBudgetV1::new(&mut work, storage_limit);
    budget.reserve_storage(FLOOR).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let function = (|| -> Result<Function, ProductionSemanticKirErrorV1> {
        let ids = if owned {
            EmissionReadOnlyV1::Owned(ids.clone())
        } else {
            EmissionReadOnlyV1::Borrowed(&ids)
        };
        let signatures = if owned {
            EmissionReadOnlyV1::Owned(signatures.clone())
        } else {
            EmissionReadOnlyV1::Borrowed(&signatures)
        };
        let result_types = if owned {
            EmissionReadOnlyV1::Owned(results.clone())
        } else {
            EmissionReadOnlyV1::Borrowed(&results)
        };
        let mut lowering = construct(ssa, selected, ids, signatures, result_types, &mut budget)?;
        let semantic = ssa.source_semantic();
        let source = &semantic.functions()[selected.index() as usize];
        let mut blocks = Vec::new();
        for block in ssa
            .plan_for_function(selected)
            .unwrap()
            .plan()
            .reverse_postorder()
        {
            let block = SemanticBlockIdV1::from_index(block.get());
            let original = &source.blocks()[block.index() as usize];
            assert!(original.statements().is_empty());
            let mut target = BasicBlock::new(lowering.kernel_block_id_v1(block)?);
            lowering.begin_block(block, &mut target)?;
            target.terminator = Some(lowering.lower_terminator(
                block,
                original.terminator().kind(),
                &mut target.operations,
            )?);
            blocks.push(target);
        }
        require_semantic_ssa_definitions_consumed_v1(
            selected.index(),
            &lowering.pending_semantic_ssa_definitions,
        )?;
        drop(lowering);
        Ok(if selected == semantic.roots()[0] {
            Function::kernel_entry(
                "placed_root",
                Signature::new(vec![], vec![]),
                vec![],
                blocks,
            )
        } else {
            Function::internal_helper(
                helper_function_id_v1(selected, source),
                Signature::new(vec![], vec![]),
                vec![],
                blocks,
            )
        })
    })();
    let resource = match &function {
        Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(error)) => Some(*error),
        _ => None,
    };
    assert!(budget.work_ledger_identity_v1() == ledger);
    let output = Emission {
        function: function.map_err(|error| format!("{error:?}")),
        resource,
        work: budget.work(),
        peak: budget.peak_storage(),
        storage_before_drop_cleanup: budget.storage(),
    };
    // Constructor/emitter backing has dropped. Final Function output remains
    // outside the unchanged legacy argument/call-buffer reservation contract.
    budget.release_storage(budget.storage() - FLOOR).unwrap();
    assert_eq!(budget.storage(), FLOOR);
    output
}

#[test]
fn borrowed_and_owned_constructors_emit_the_same_complete_production_functions() {
    let ssa = source();
    for selected in [
        SemanticFunctionIdV1::from_index(0),
        SemanticFunctionIdV1::from_index(1),
    ] {
        let borrowed = emit(&ssa, selected, false, WORK, STORAGE);
        let owned = emit(&ssa, selected, true, WORK, STORAGE);
        assert!(borrowed.function.is_ok());
        assert_eq!(borrowed, owned);
        let mut work = Work::new(WORK);
        let mut budget = ArgumentBudgetV1::new(&mut work, STORAGE);
        let actual = resource_tests::emission_placement_lowering_tests::lower_placed_function(
            &ssa,
            selected,
            SemanticEmissionPlacementV1::default(),
            &mut budget,
        )
        .unwrap();
        assert_eq!(borrowed.function.as_ref().unwrap(), &actual.function);
        drop(actual);
        budget.release_storage(budget.storage()).unwrap();
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn borrowed_and_owned_constructors_have_identical_exact_and_short_resource_results() {
    let ssa = source();
    let root = ssa.source_semantic().roots()[0];
    let baseline = emit(&ssa, root, false, WORK, STORAGE);
    assert!(baseline.function.is_ok());
    assert!(baseline.work > 0 && baseline.peak > FLOOR);
    for (index, (work, storage, succeeds)) in [
        (baseline.work, baseline.peak, true),
        (baseline.work - 1, baseline.peak, false),
        (baseline.work, baseline.peak - 1, false),
    ]
    .into_iter()
    .enumerate()
    {
        let borrowed = emit(&ssa, root, false, work, storage);
        let owned = emit(&ssa, root, true, work, storage);
        assert_eq!(borrowed, owned);
        assert_eq!(borrowed.function.is_ok(), succeeds);
        match index {
            0 => assert_eq!(borrowed.resource, None),
            1 => assert!(matches!(
                borrowed.resource,
                Some(ArgumentResourceV1::Work(_))
            )),
            2 => assert!(matches!(
                borrowed.resource,
                Some(ArgumentResourceV1::Storage(_))
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn unwinding_drops_owned_view_once_and_never_drops_borrowed_input() {
    use std::{cell::Cell, rc::Rc};
    struct Counted(Rc<Cell<usize>>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let borrowed_drops = Rc::new(Cell::new(0));
    let owned_drops = Rc::new(Cell::new(0));
    let original = Counted(Rc::clone(&borrowed_drops));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let borrowed: EmissionReadOnlyV1<'_, Counted> = (&original).into();
        let owned: EmissionReadOnlyV1<'_, Counted> = Counted(Rc::clone(&owned_drops)).into();
        assert!(std::ptr::eq(&*borrowed, &original));
        assert_eq!(owned.0.get(), 0);
        panic!("private read-only input unwind");
    }));
    assert!(caught.is_err());
    assert_eq!(owned_drops.get(), 1);
    assert_eq!(borrowed_drops.get(), 0);
    drop(original);
    assert_eq!(borrowed_drops.get(), 1);
}

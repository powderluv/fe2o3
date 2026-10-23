use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1;
use fe2o3_mir_model::semantic_mir_v1::SemanticMfmaAccumulatorDistributionV1;

fn option_availability() -> SemanticOptionAvailabilityV1 {
    use fe2o3_mir_model::semantic_mir_v1::*;
    let owner = resource_tests::helper_closure_semantic_owner();
    let base = &owner.semantic().functions()[1];
    let source = base.source();
    let unit = SemanticTypeIdV1::from_index(0);
    let option = SemanticTypeIdV1::from_index(1);
    let discriminant = SemanticTypeIdV1::from_index(2);
    let place =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let edge =
        |role, target| SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target));
    let block = |tag, statements, kind| {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([tag; 32]),
            source,
            statements,
            SemanticTerminatorV1::new(source, kind),
        )
        .unwrap()
    };
    let call = SemanticDirectCallV1::new_callable(
        SemanticCallableIdV1::from_index(0),
        vec![],
        Some(SemanticCallDestinationV1::new(
            place(1, option),
            edge(SemanticEdgeRoleV1::CallReturn, 1),
        )),
        SemanticUnwindActionV1::Unreachable,
    )
    .unwrap();
    let assignment = SemanticStatementV1::new(
        source,
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(2, discriminant),
            SemanticRvalueV1::new(
                discriminant,
                SemanticRvalueKindV1::Discriminant(place(1, option)),
            ),
        )),
    );
    let blocks = vec![
        block(220, vec![], SemanticTerminatorKindV1::Call(call)),
        block(
            221,
            vec![assignment],
            SemanticTerminatorKindV1::SwitchInt {
                discriminant: SemanticOperandV1::Copy(place(2, discriminant)),
                targets: SemanticSwitchTargetsV1::new(
                    vec![SemanticSwitchTargetV1::new(
                        0,
                        edge(SemanticEdgeRoleV1::SwitchValue, 2),
                    )],
                    edge(SemanticEdgeRoleV1::SwitchOtherwise, 3),
                )
                .unwrap(),
            },
        ),
        block(222, vec![], SemanticTerminatorKindV1::Return),
        block(223, vec![], SemanticTerminatorKindV1::Return),
    ];
    let locals = [unit, option, discriminant]
        .into_iter()
        .enumerate()
        .map(|(i, ty)| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([231 + i as u8; 32]),
                ty,
                if i == 0 {
                    SemanticLocalRoleV1::Return
                } else {
                    SemanticLocalRoleV1::Temporary
                },
                source,
            )
        })
        .collect();
    let function = SemanticFunctionDeclV1::new(
        base.identity(),
        base.role(),
        base.item_definition_identity(),
        base.monomorphization_identity(),
        base.generic_type_arguments_identity(),
        base.const_generic_arguments_identity(),
        source,
        base.abi().clone(),
        locals,
        SemanticBlockIdV1::from_index(0),
        blocks,
    )
    .unwrap();
    let local = SemanticLocalIdV1::from_index(1);
    let dominance = SemanticOptionDominanceV1::analyze(
        &function,
        &[fe2o3_mir_model::SemanticOptionProducerV1::new(
            local,
            SemanticBlockIdV1::from_index(1),
        )],
    )
    .unwrap();
    let availability = dominance.availability(local).unwrap();
    assert!(dominance.allows(availability, SemanticBlockIdV1::from_index(3)));
    assert!(!dominance.allows(availability, SemanticBlockIdV1::from_index(2)));
    availability
}

fn pointer() -> Type {
    Type::pointer(
        Type::slice(Type::F32, AddressSpace::Global, AccessMode::ReadOnly),
        AddressSpace::Workgroup,
        AccessMode::ReadWrite,
    )
}

fn value() -> SemanticValueBindingV1 {
    SemanticValueBindingV1::Value {
        id: ValueId(17),
        ty: pointer(),
    }
}

// Structural copying controls do not authenticate these shapes for execution.
fn heap_cases() -> Vec<SemanticValueBindingV1> {
    use SemanticValueBindingV1 as B;
    let operand = SemanticMfmaOperandContractV1 {
        role: SemanticMfmaOperandRoleV1::B,
        profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
        register_distribution: SemanticMfmaRegisterDistributionV1::Tile16x16,
        wave_width: 64,
    };
    let accumulator = SemanticMfmaAccumulatorContractV1 {
        profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
        distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
        wave_width: 64,
    };
    vec![
        B::Aggregate(vec![B::Unit, value()]),
        B::Enum {
            discriminant: ValueId(19),
            discriminant_ty: pointer(),
            semantic_type: SemanticTypeIdV1::from_index(21),
            variant: Some(3),
            payloads: BTreeMap::from([(1, vec![B::Unit]), (3, vec![value(), B::Unmaterialized])]),
        },
        B::DynamicLds {
            base: ValueId(23),
            base_ty: pointer(),
            len: ValueId(25),
            byte_len: ValueId(27),
            dynamic_lds: SemanticTypeIdV1::from_index(29),
            element_storage: SemanticTypeIdV1::from_index(31),
            elements: 64,
            byte_extent: 256,
            alignment: 16,
            producer_function: SemanticFunctionIdV1::from_index(33),
            producer_block: SemanticBlockIdV1::from_index(35),
        },
        B::MatrixFragment {
            values: vec![
                (ValueId(37), Type::Scalar(ScalarType::U32)),
                (ValueId(39), pointer()),
            ],
            contract: operand,
            storage_layout: SemanticMfmaStorageLayoutV1::LdsXor4,
            wave: SemanticCurrentWaveV1::new(64),
        },
        B::AccumulatorFragment {
            values: vec![(ValueId(41), Type::F32), (ValueId(43), pointer())],
            contract: accumulator,
            wave: SemanticCurrentWaveV1::new(64),
        },
        B::WorkgroupPipeline {
            storage: ValueId(45),
            pipeline: SemanticTypeIdV1::from_index(47),
            element: SemanticTypeIdV1::from_index(49),
            payload_binding: SemanticPromotedBindingV1::MatrixFragment {
                contract: operand,
                storage_layout: SemanticMfmaStorageLayoutV1::ColumnMajor,
            },
            component_types: vec![pointer(), Type::F32].into_boxed_slice(),
            packed_type: pointer(),
            buffers: 3,
            elements: 64,
            prefetch_distance: 2,
            alignment: 16,
        },
        value(),
        B::OptionPointer {
            present: ValueId(51),
            pointer: ValueId(53),
            pointer_ty: pointer(),
            availability: option_availability(),
        },
    ]
}

fn cfg_inline_capabilities() -> Vec<SemanticValueBindingV1> {
    use SemanticValueBindingV1 as B;
    let option = option_availability();
    let availability = SemanticCapabilityAvailabilityV1::EnumPayload {
        local: SemanticLocalIdV1::from_index(9),
        variant: 3,
    };
    let index_space = SemanticDisjointIndexSpaceV1::ShiftedIndex1d { offset: 7 };
    vec![
        B::MathContext,
        B::CollectiveContext,
        B::WorkgroupLdsScope,
        B::MatrixContext,
        B::WaveLane {
            value: ValueId(59),
            wave: SemanticCurrentWaveV1::new(64),
        },
        B::Gfx950LdsTransposeTile {
            storage: ValueId(61),
            format: SemanticGfx950LdsTransposeFormatV1::Fp8E4M3,
            state: SemanticGfx950LdsTransposeStateV1::Published,
        },
        B::IndexWitness {
            id: ValueId(63),
            index_space,
            disjoint: true,
            availability: None,
        },
        B::IndexWitness {
            id: ValueId(65),
            index_space,
            disjoint: false,
            availability: Some(availability),
        },
        B::IndexWitness {
            id: ValueId(67),
            index_space,
            disjoint: true,
            availability: Some(SemanticCapabilityAvailabilityV1::Option(option)),
        },
        B::OptionIndexWitness {
            present: ValueId(69),
            id: ValueId(71),
            index_space,
            disjoint: false,
            availability: option,
        },
        B::GridLeader { availability },
        B::ComponentWitness {
            raw: ValueId(73),
            index_space,
            availability,
        },
        B::OptionComponentWitness {
            present: ValueId(75),
            raw: ValueId(77),
            index_space,
            availability: option,
        },
        B::OptionGridLeader {
            present: ValueId(79),
            availability: option,
        },
    ]
}

// This checks copying only. Producer, dominance and lifecycle admission remain
// obligations of the existing source/consumer boundaries, including real Rust tests.
fn cfg_copy_observation(
    source: &SemanticValueBindingV1,
    work_limit: usize,
    storage_limit: usize,
) -> Result<(usize, usize, usize), ProductionSemanticKirErrorV1> {
    const FLOOR: usize = 37;
    let before = format!("{source:?}");
    let mut work = CanonicalKernelIrWorkBudgetV1::new(work_limit);
    let mut budget = ArgumentBudgetV1::new(&mut work, storage_limit);
    budget.reserve_storage(FLOOR).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let result = with_canonical_call_scratch_v1(&mut budget, |budget| {
        let mut nodes = 0;
        let copied = clone_execution_cfg_binding_v29(source, &mut nodes, budget)?;
        assert_eq!(format!("{copied:?}"), before);
        drop(copied);
        Ok(nodes)
    });
    assert_eq!(format!("{source:?}"), before);
    assert_eq!(budget.storage(), FLOOR);
    assert!(budget.work_ledger_identity_v1() == ledger);
    result.map(|nodes| (budget.work(), budget.peak_storage(), nodes))
}

#[test]
fn cfg_copy_preserves_compiler_issued_capabilities_and_heap_payloads() {
    for source in cfg_inline_capabilities() {
        assert_eq!(cfg_copy_observation(&source, 3, 37).unwrap(), (3, 37, 1));
        assert!(matches!(
            cfg_copy_observation(&source, 2, 37),
            Err(
                ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Work(_)
                )
            )
        ));
    }
    for source in heap_cases() {
        let (work, peak, nodes) = cfg_copy_observation(&source, 1_000_000, 1_000_000).unwrap();
        assert_eq!(
            cfg_copy_observation(&source, work, peak).unwrap(),
            (work, peak, nodes)
        );
        assert!(matches!(
            cfg_copy_observation(&source, work - 1, peak),
            Err(
                ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Work(_)
                )
            )
        ));
        assert!(matches!(
            cfg_copy_observation(&source, work, peak - 1),
            Err(
                ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Storage(_)
                )
            )
        ));
    }
    let nested = SemanticValueBindingV1::Aggregate(vec![
        SemanticValueBindingV1::Aggregate(cfg_inline_capabilities()),
        SemanticValueBindingV1::Aggregate(heap_cases()),
    ]);
    cfg_copy_observation(&nested, 1_000_000, 1_000_000).unwrap();
}

#[test]
fn cfg_optional_pointer_copy_has_exact_independent_resource_bounds() {
    let source = SemanticValueBindingV1::OptionPointer {
        present: ValueId(81),
        pointer: ValueId(83),
        pointer_ty: pointer(),
        availability: option_availability(),
    };
    // Two scratch operations, one CFG node, one binding, three type nodes.
    const WORK: usize = 2 + 1 + 1 + 3;
    let storage = 37 + 2 * std::mem::size_of::<Type>();
    assert_eq!(
        cfg_copy_observation(&source, WORK, storage).unwrap(),
        (WORK, storage, 1)
    );
    assert!(matches!(
        cfg_copy_observation(&source, WORK - 1, storage),
        Err(
            ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(ArgumentResourceV1::Work(
                _
            ))
        )
    ));
    assert!(matches!(
        cfg_copy_observation(&source, WORK, storage - 1),
        Err(
            ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                ArgumentResourceV1::Storage(_)
            )
        )
    ));
}

#[test]
fn cfg_capability_aggregate_copy_is_linear_and_keeps_component_bound() {
    for count in [16, 64, MAX_SSA_VALUE_COMPONENTS_V1 - 1] {
        let source = SemanticValueBindingV1::Aggregate(
            (0..count)
                .map(|index| SemanticValueBindingV1::IndexWitness {
                    id: ValueId(index as u32),
                    index_space: SemanticDisjointIndexSpaceV1::Index1d,
                    disjoint: true,
                    availability: None,
                })
                .collect(),
        );
        // Two scratch operations, aggregate node, Vec construction, each leaf.
        let work = 2 + 1 + 3 + count;
        let (used, peak, nodes) = cfg_copy_observation(&source, work, 1_000_000).unwrap();
        assert_eq!((used, nodes), (work, count + 1));
        assert!(peak >= 37 + count * std::mem::size_of::<SemanticValueBindingV1>());
        assert_eq!(
            cfg_copy_observation(&source, work, peak).unwrap(),
            (work, peak, nodes)
        );
    }
    let oversized = SemanticValueBindingV1::Aggregate(
        (0..MAX_SSA_VALUE_COMPONENTS_V1)
            .map(|_| SemanticValueBindingV1::MathContext)
            .collect(),
    );
    assert!(
        matches!(
            cfg_copy_observation(&oversized, 1_000_000, 1_000_000),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "execution CFG transport differs from its captured SSA state",
                ..
            })
        ),
        "the aggregate itself also consumes one structural node"
    );
    let source = &cfg_inline_capabilities()[0];
    let mut work = CanonicalKernelIrWorkBudgetV1::new(10);
    let mut budget = ArgumentBudgetV1::new(&mut work, 37);
    budget.reserve_storage(37).unwrap();
    let mut nodes = MAX_SSA_VALUE_COMPONENTS_V1;
    let error = clone_execution_cfg_binding_v29(source, &mut nodes, &mut budget).unwrap_err();
    assert!(matches!(
        error,
        ProductionSemanticKirErrorV1::Unsupported {
            function: 0,
            block: None,
            statement: None,
            detail: "execution CFG transport differs from its captured SSA state",
        }
    ));
    assert_eq!(budget.storage(), 37);
}

#[test]
fn cfg_copy_keeps_unmaterialized_and_ordinary_execution_refusals() {
    for source in [
        SemanticValueBindingV1::Unmaterialized,
        SemanticValueBindingV1::Value {
            id: ValueId(85),
            ty: Type::Execution(fe2o3_kernel_ir::ExecutionRoleV15::Workgroup),
        },
        SemanticValueBindingV1::Aggregate(vec![SemanticValueBindingV1::Unmaterialized]),
    ] {
        assert!(matches!(
            cfg_copy_observation(&source, 1000, 10000),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                function: 0,
                block: None,
                statement: None,
                detail: "execution CFG transport differs from its captured SSA state",
            })
        ));
    }
}

fn assert_copy_budget(source: &SemanticValueBindingV1) {
    const FLOOR: usize = 37;
    const WORK_FLOOR: usize = 11;
    let before = format!("{source:?}");
    let mut work = CanonicalKernelIrWorkBudgetV1::new(usize::MAX);
    let mut budget = ArgumentBudgetV1::new(&mut work, usize::MAX);
    budget.reserve_storage(FLOOR).unwrap();
    budget.charge_work(WORK_FLOOR).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let copied = emission_clone_binding_v1(source, &mut budget).unwrap();
    assert_eq!(format!("{copied:?}"), before);
    let used = budget.work();
    let peak = budget.peak_storage();
    let live = budget.storage() - FLOOR;
    assert!(used > WORK_FLOOR);
    assert!(live > 0);
    assert!(budget.work_ledger_identity_v1() == ledger);
    drop(copied);
    budget.release_storage(live).unwrap();
    assert_eq!(budget.storage(), FLOOR);
    for (work_limit, storage_limit, expected) in
        [(used, peak, 0), (used - 1, peak, 1), (used, peak - 1, 2)]
    {
        let mut work = CanonicalKernelIrWorkBudgetV1::new(work_limit);
        let mut budget = ArgumentBudgetV1::new(&mut work, storage_limit);
        budget.reserve_storage(FLOOR).unwrap();
        budget.charge_work(WORK_FLOOR).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let result = emission_clone_binding_v1(source, &mut budget);
        match (expected, result) {
            (0, Ok(copied)) => {
                assert_eq!(format!("{copied:?}"), before);
                assert_eq!(budget.work(), used);
                assert_eq!(budget.peak_storage(), peak);
                assert_eq!(budget.storage(), FLOOR + live);
                drop(copied);
                budget.release_storage(live).unwrap();
            }
            (
                1,
                Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Work(_),
                )),
            ) => {}
            (
                2,
                Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Storage(_),
                )),
            ) => {}
            (_, other) => panic!("unexpected copy result: {other:?}"),
        }
        assert_eq!(budget.storage(), FLOOR);
        assert!(budget.work_ledger_identity_v1() == ledger);
        assert!(budget.work() >= WORK_FLOOR);
        assert_eq!(format!("{source:?}"), before);
    }
}

#[test]
fn every_heap_variant_preserves_metadata_and_exact_resource_boundaries() {
    for source in heap_cases() {
        assert_copy_budget(&source);
    }
}

#[test]
fn storage_overflow_refunds_only_partial_copy_backing() {
    let source = value();
    let before = format!("{source:?}");
    let floor = usize::MAX - std::mem::size_of::<Type>();
    let mut work = CanonicalKernelIrWorkBudgetV1::new(usize::MAX);
    let mut budget = ArgumentBudgetV1::new(&mut work, usize::MAX);
    budget.reserve_storage(floor).unwrap();
    assert!(matches!(
        emission_clone_binding_v1(&source, &mut budget),
        Err(
            ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                ArgumentResourceV1::Storage(_)
            )
        )
    ));
    assert_eq!(budget.storage(), floor);
    assert_eq!(budget.peak_storage(), usize::MAX);
    assert_eq!(budget.work(), 3);
    assert_eq!(format!("{source:?}"), before);
}

#[test]
fn nested_aggregates_and_inactive_enum_payloads_remain_owned_copies() {
    let mut source = value();
    for _ in 0..64 {
        source = SemanticValueBindingV1::Aggregate(vec![SemanticValueBindingV1::Unit, source]);
    }
    assert_copy_budget(&source);
    let source = SemanticValueBindingV1::Enum {
        discriminant: ValueId(31),
        discriminant_ty: Type::Scalar(ScalarType::U32),
        semantic_type: SemanticTypeIdV1::from_index(12),
        variant: Some(0),
        payloads: BTreeMap::from([
            (0, vec![value()]),
            (1, vec![SemanticValueBindingV1::MovedExecution, source]),
        ]),
    };
    assert_copy_budget(&source);
    let mut work = CanonicalKernelIrWorkBudgetV1::new(100000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 100000);
    let copied = emission_clone_binding_v1(&source, &mut budget).unwrap();
    assert!(semantic_binding_contains_execution_v29(&copied));
    let SemanticValueBindingV1::Enum { payloads, .. } = copied else {
        panic!()
    };
    assert_eq!(payloads.len(), 2);
    assert!(matches!(
        payloads[&1][0],
        SemanticValueBindingV1::MovedExecution
    ));
}

#[test]
fn copied_aggregate_backing_is_independent_of_source_capacity() {
    let mut fields = Vec::with_capacity(16);
    fields.push(value());
    let source = SemanticValueBindingV1::Aggregate(fields);
    let mut work = CanonicalKernelIrWorkBudgetV1::new(1000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 10000);
    let copy = emission_clone_binding_v1(&source, &mut budget).unwrap();
    let SemanticValueBindingV1::Aggregate(original) = &source else {
        panic!()
    };
    let SemanticValueBindingV1::Aggregate(mut copied) = copy else {
        panic!()
    };
    assert_ne!(original.as_ptr(), copied.as_ptr());
    assert!(original.capacity() >= 16);
    assert_eq!(
        budget.storage(),
        copied.capacity() * std::mem::size_of::<SemanticValueBindingV1>()
            + 2 * std::mem::size_of::<Type>()
    );
    copied[0] = SemanticValueBindingV1::Unit;
    assert!(matches!(
        original[0],
        SemanticValueBindingV1::Value {
            id: ValueId(17),
            ..
        }
    ));
}

#[test]
fn copying_nominal_leaves_does_not_admit_ordinary_ssa_values() {
    let mut work = CanonicalKernelIrWorkBudgetV1::new(1000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 1000);
    for (source, expected) in [
        (
            SemanticValueBindingV1::MovedExecution,
            "moved execution value cannot be observed",
        ),
        (
            SemanticValueBindingV1::Value {
                id: ValueId(7),
                ty: Type::Execution(fe2o3_kernel_ir::ExecutionRoleV15::Workgroup),
            },
            "execution role requires a nominal producer binding",
        ),
    ] {
        let copied = emission_clone_binding_v1(&source, &mut budget).unwrap();
        assert_eq!(format!("{copied:?}"), format!("{source:?}"));
        assert_eq!(copied.value(), Err(expected));
    }
    assert_eq!(budget.storage(), 0);
    assert!(budget.work() > 0);
}

#[test]
fn enum_projection_moves_the_selected_backing_and_preserves_negative_precedence() {
    let fields = vec![value()];
    let backing = fields.as_ptr();
    let payloads = BTreeMap::from([(
        3,
        vec![
            SemanticValueBindingV1::Unit,
            SemanticValueBindingV1::Aggregate(fields),
        ],
    )]);
    let projected = project_enum_payload_field(3, payloads, 1).unwrap();
    let SemanticValueBindingV1::Aggregate(fields) = projected else {
        panic!()
    };
    assert_eq!(fields.as_ptr(), backing);
    assert!(matches!(
        fields[0],
        SemanticValueBindingV1::Value {
            id: ValueId(17),
            ..
        }
    ));
    assert!(matches!(
        project_enum_payload_field(5, BTreeMap::new(), 0),
        Ok(SemanticValueBindingV1::Unmaterialized),
    ));
    assert_eq!(
        project_enum_payload_field(3, BTreeMap::from([(3, vec![])]), 0).unwrap_err(),
        "enum payload field is unavailable in this block",
    );
    for selected in [0, 7] {
        let payloads = BTreeMap::from([
            (0, vec![value()]),
            (
                1,
                vec![SemanticValueBindingV1::Aggregate(vec![
                    SemanticValueBindingV1::MovedExecution,
                ])],
            ),
        ]);
        assert_eq!(
            project_enum_payload_field(selected, payloads, 0).unwrap_err(),
            "execution bindings cannot be restored from enum payloads",
        );
    }
}

#[test]
fn enum_projection_prepays_all_payload_traversal_without_storage_credit() {
    let payloads = BTreeMap::from([
        (0, vec![value()]),
        (
            1,
            vec![SemanticValueBindingV1::Aggregate(vec![
                SemanticValueBindingV1::Unit,
                SemanticValueBindingV1::Enum {
                    discriminant: ValueId(4),
                    discriminant_ty: Type::Scalar(ScalarType::U32),
                    semantic_type: SemanticTypeIdV1::from_index(5),
                    variant: Some(2),
                    payloads: BTreeMap::from([(2, vec![SemanticValueBindingV1::MovedExecution])]),
                },
            ])],
        ),
    ]);
    let before = format!("{payloads:?}");
    let mut work = CanonicalKernelIrWorkBudgetV1::new(10000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 37);
    budget.reserve_storage(37).unwrap();
    emission_prepay_enum_projection_v1(&payloads, &mut budget).unwrap();
    let used = budget.work();
    assert!(used > 0);
    for limit in [used, used - 1] {
        let mut work = CanonicalKernelIrWorkBudgetV1::new(limit);
        let mut budget = ArgumentBudgetV1::new(&mut work, 37);
        budget.reserve_storage(37).unwrap();
        let result = emission_prepay_enum_projection_v1(&payloads, &mut budget);
        if limit == used {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(
                    ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                        ArgumentResourceV1::Work(_)
                    )
                )
            ));
        }
        assert_eq!(budget.storage(), 37);
        assert_eq!(format!("{payloads:?}"), before);
    }
}

fn assert_and_drop_type(mut ty: Type, depth: usize) {
    for level in (0..depth).rev() {
        ty = match (level % 2, ty) {
            (0, Type::Pointer(pointer)) => {
                assert_eq!(pointer.address_space, AddressSpace::Global);
                assert_eq!(pointer.access, AccessMode::ReadOnly);
                *pointer.pointee
            }
            (1, Type::Slice(slice)) => {
                assert_eq!(slice.address_space, AddressSpace::Workgroup);
                assert_eq!(slice.access, AccessMode::ReadWrite);
                *slice.element
            }
            _ => panic!("copy changed pointer/slice structure"),
        };
    }
    assert!(matches!(
        ty,
        Type::Execution(fe2o3_kernel_ir::ExecutionRoleV15::Workgroup)
    ));
}

#[test]
fn deep_types_copy_without_execution_cfg_node_limits() {
    const DEPTH: usize = 4096;
    let mut ty = Type::Execution(fe2o3_kernel_ir::ExecutionRoleV15::Workgroup);
    for level in 0..DEPTH {
        ty = if level % 2 == 0 {
            Type::pointer(ty, AddressSpace::Global, AccessMode::ReadOnly)
        } else {
            Type::slice(ty, AddressSpace::Workgroup, AccessMode::ReadWrite)
        };
    }
    let source = SemanticValueBindingV1::Value { id: ValueId(7), ty };
    let mut work = CanonicalKernelIrWorkBudgetV1::new(usize::MAX);
    let mut budget = ArgumentBudgetV1::new(&mut work, usize::MAX);
    budget.reserve_storage(37).unwrap();
    let copied = emission_clone_binding_v1(&source, &mut budget).unwrap();
    let live = budget.storage() - 37;
    assert_eq!(live, DEPTH * std::mem::size_of::<Type>());
    for binding in [source, copied] {
        let SemanticValueBindingV1::Value { id, ty } = binding else {
            panic!()
        };
        assert_eq!(id, ValueId(7));
        assert_and_drop_type(ty, DEPTH);
    }
    budget.release_storage(live).unwrap();
    assert_eq!(budget.storage(), 37);
}

#[test]
fn ordinary_place_resolution_accounts_slot_lookups_and_preserves_missing_local_error() {
    let owner = resource_tests::helper_closure_semantic_owner();
    let semantic = owner.semantic();
    let function = &semantic.functions()[1];
    let place = SemanticPlaceV1::new(
        SemanticLocalIdV1::from_index(0),
        vec![],
        function.locals()[0].ty(),
    )
    .unwrap();
    // Two empty-map searches at 32 work each, two identity checks, one copy.
    const SLOT_LOOKUP_WORK: usize = 2 * (2 * 16) + 2;
    const READ_WORK: usize = SLOT_LOOKUP_WORK + 1;
    // Measure construction separately; every case below still constructs and
    // resolves on one finite, unchanged ledger.
    let (constructor_work, constructor_storage, constructor_peak) = {
        let mut baseline_work = CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut baseline_budget = ArgumentBudgetV1::new(&mut baseline_work, 16 * 1024 * 1024);
        baseline_budget.reserve_storage(37).unwrap();
        let baseline = SemanticFunctionLoweringV1::new(
            semantic.types(),
            semantic.callables(),
            function,
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
                local_bindings: None,
            },
            None,
            None,
            BTreeSet::new(),
            1,
            false,
            1000,
            &mut baseline_budget,
        )
        .unwrap();
        drop(baseline);
        (
            baseline_budget.work(),
            baseline_budget.storage() - 37,
            baseline_budget.peak_storage() - 37,
        )
    };
    eprintln!(
        "place-resolution constructor prefix: work={constructor_work} storage={constructor_storage} peak={constructor_peak}"
    );
    for (limit, has_binding) in [
        (0, true),
        (READ_WORK - 1, true),
        (READ_WORK, true),
        (SLOT_LOOKUP_WORK, false),
    ] {
        let mut work = CanonicalKernelIrWorkBudgetV1::new(constructor_work + limit);
        let mut budget = ArgumentBudgetV1::new(&mut work, 37 + constructor_peak);
        budget.reserve_storage(37).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let mut lowering = SemanticFunctionLoweringV1::new(
            semantic.types(),
            semantic.callables(),
            function,
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
                local_bindings: None,
            },
            None,
            None,
            BTreeSet::new(),
            1,
            false,
            1000,
            &mut budget,
        )
        .unwrap();
        let attached = lowering.emission_work.as_deref().unwrap();
        assert_eq!(attached.emission_service_work_v1(), Some(constructor_work));
        assert_eq!(attached.storage(), 37 + constructor_storage);
        assert!(attached.work_ledger_identity_v1() == ledger);
        lowering.locals[0] = has_binding.then_some(SemanticValueBindingV1::Unit);
        let mut operations = Vec::new();
        let result = lowering.resolve_place(
            SemanticBlockIdV1::from_index(0),
            Some(7),
            &place,
            &mut operations,
        );
        if has_binding && limit < READ_WORK {
            assert!(matches!(
                result,
                Err(
                    ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                        ArgumentResourceV1::Work(_)
                    )
                )
            ));
        } else if has_binding {
            assert!(matches!(result, Ok(SemanticValueBindingV1::Unit)));
        } else {
            assert!(matches!(
                result,
                Err(ProductionSemanticKirErrorV1::MissingLocalDefinition {
                    function: 0,
                    block: 0,
                    statement: Some(7),
                    local: 0,
                })
            ));
        }
        assert!(operations.is_empty());
        drop(lowering);
        assert_eq!(budget.storage(), 37 + constructor_storage);
        assert_eq!(budget.work(), constructor_work + limit);
    }
}

struct PanicAfterBacking<'a> {
    inner: &'a mut dyn SemanticEmissionBudgetV1,
    floor: usize,
}

impl SemanticEmissionBudgetV1 for PanicAfterBacking<'_> {
    fn work_ledger_identity_v1(&self) -> fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1 {
        self.inner.work_ledger_identity_v1()
    }
    fn charge_work(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        if self.inner.storage() > self.floor {
            panic!("binding-copy injected unwind");
        }
        self.inner.charge_work(amount)
    }
    fn reserve_storage(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        self.inner.reserve_storage(amount)
    }
    fn release_storage(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        self.inner.release_storage(amount)
    }
    fn storage(&self) -> usize {
        self.inner.storage()
    }
}

#[test]
fn unwind_releases_only_dropped_partial_backing() {
    let source = SemanticValueBindingV1::Aggregate(vec![value()]);
    let before = format!("{source:?}");
    let mut work = CanonicalKernelIrWorkBudgetV1::new(10000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 10000);
    budget.reserve_storage(37).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut injected = PanicAfterBacking {
            inner: &mut budget,
            floor: 37,
        };
        emission_clone_binding_v1(&source, &mut injected)
    }));
    let payload = result.expect_err("copy must reach injected unwind");
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"binding-copy injected unwind")
    );
    assert_eq!(budget.storage(), 37);
    assert!(budget.peak_storage() > 37);
    assert_eq!(format!("{source:?}"), before);
}

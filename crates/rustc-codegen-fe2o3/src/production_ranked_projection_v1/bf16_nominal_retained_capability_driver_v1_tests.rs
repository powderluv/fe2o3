//! Synthetic component helpers only, not full driver or genuine query acceptance.
use super::*;
use crate::production_ranked_projection_v1::{
    assertion_definition_inventory, bind_capability_read_effects_to_call_blocks_v1,
    collect_workgroup_pipeline_payloads_v1, constant_locals, local_allocation_contracts,
    local_provenance_with_scalar_inventory_v1, propagate_capability_dataflow_v1,
    workgroup_pipeline_local_owners_v1,
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_mir_model::semantic_mir_v1::*;

const SCALAR_TYPE: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(0);

const ARRAY_TYPE: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(1);

const POINTER_TYPE: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(2);

fn bytes(tag: u8) -> [u8; 32] {
    [tag; 32]
}

fn local(tag: u8, ty: SemanticTypeIdV1, role: SemanticLocalRoleV1) -> SemanticLocalDeclV1 {
    SemanticLocalDeclV1::new(
        SemanticLocalIdentityV1::from_sha256(bytes(tag)),
        ty,
        role,
        SemanticSourceProvenanceV1::unavailable(),
    )
}

fn block(
    tag: u8,
    statements: Vec<SemanticStatementV1>,
    terminator: SemanticTerminatorKindV1,
) -> SemanticBasicBlockV1 {
    SemanticBasicBlockV1::new(
        SemanticBlockIdentityV1::from_sha256(bytes(tag)),
        SemanticSourceProvenanceV1::unavailable(),
        statements,
        SemanticTerminatorV1::new(SemanticSourceProvenanceV1::unavailable(), terminator),
    )
    .unwrap()
}

fn cfg_edge(role: SemanticEdgeRoleV1, target: u32) -> SemanticControlFlowEdgeV1 {
    SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target))
}

fn projection_function(blocks: Vec<SemanticBasicBlockV1>) -> SemanticFunctionDeclV1 {
    projection_function_with_locals(
        blocks,
        vec![
            local(20, SCALAR_TYPE, SemanticLocalRoleV1::Return),
            local(21, ARRAY_TYPE, SemanticLocalRoleV1::Temporary),
            local(22, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            local(23, POINTER_TYPE, SemanticLocalRoleV1::Temporary),
        ],
    )
}

fn projection_function_with_locals(
    blocks: Vec<SemanticBasicBlockV1>,
    locals: Vec<SemanticLocalDeclV1>,
) -> SemanticFunctionDeclV1 {
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256(bytes(10)),
        SemanticLayoutIdentityV1::from_sha256(bytes(10)),
        SemanticCanonAbiV1::GpuKernel,
        SemanticExternAbiV1::GpuKernel,
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(SCALAR_TYPE, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256(bytes(11)),
        SemanticFunctionRoleV1::KernelRoot,
        SemanticItemDefinitionIdentityV1::from_sha256(bytes(12)),
        SemanticMonomorphizationIdentityV1::from_sha256(bytes(13)),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256(bytes(14)),
        SemanticConstGenericArgumentsIdentityV1::from_sha256(bytes(15)),
        SemanticSourceProvenanceV1::unavailable(),
        abi,
        locals,
        SemanticBlockIdV1::from_index(0),
        blocks,
    )
    .unwrap()
}

fn statement(kind: SemanticStatementKindV1) -> SemanticStatementV1 {
    SemanticStatementV1::new(SemanticSourceProvenanceV1::unavailable(), kind)
}

fn constant(value: u128) -> SemanticOperandV1 {
    SemanticOperandV1::Constant(SemanticConstantV1::new(
        SCALAR_TYPE,
        SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, 4).unwrap()),
    ))
}

fn compiler_intrinsic_callable(
    operation: SemanticCompilerIntrinsicOperationV1,
) -> SemanticCallableDeclV1 {
    let abi = projection_function(vec![block(116, vec![], SemanticTerminatorKindV1::Return)])
        .abi()
        .clone();
    SemanticCallableDeclV1::CompilerIntrinsic {
        binding: SemanticNonBodyCallableBindingV1::new(
            SemanticFunctionIdentityV1::from_sha256(bytes(116)),
            SemanticItemDefinitionIdentityV1::from_sha256(bytes(117)),
            SemanticMonomorphizationIdentityV1::from_sha256(bytes(118)),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256(bytes(119)),
            SemanticConstGenericArgumentsIdentityV1::from_sha256(bytes(120)),
            SemanticSourceProvenanceV1::unavailable(),
            abi,
        ),
        operation,
        operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256(bytes(121)),
    }
}

fn place(index: u32) -> SemanticPlaceV1 {
    SemanticPlaceV1::new(SemanticLocalIdV1::from_index(index), vec![], POINTER_TYPE).unwrap()
}

fn operand(index: u32) -> SemanticOperandV1 {
    SemanticOperandV1::Move(place(index))
}

fn assign(index: u32, value: SemanticRvalueKindV1) -> SemanticStatementV1 {
    statement(SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
        place(index),
        SemanticRvalueV1::new(POINTER_TYPE, value),
    )))
}

fn borrow(destination: u32, source: u32) -> SemanticStatementV1 {
    assign(
        destination,
        SemanticRvalueKindV1::Borrow {
            kind: SemanticBorrowKindV1::Mutable,
            place: place(source),
        },
    )
}

fn call(callee: u32, arguments: Vec<SemanticOperandV1>, target: u32) -> SemanticTerminatorKindV1 {
    SemanticTerminatorKindV1::Call(
        SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(callee),
            arguments,
            Some(SemanticCallDestinationV1::new(
                SemanticPlaceV1::new(SemanticLocalIdV1::from_index(5), vec![], SCALAR_TYPE)
                    .unwrap(),
                cfg_edge(SemanticEdgeRoleV1::CallReturn, target),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap(),
    )
}

fn callables() -> Vec<SemanticCallableDeclV1> {
    vec![
        compiler_intrinsic_callable(SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut {
            disjoint_slice: POINTER_TYPE,
            index_witness: SCALAR_TYPE,
            element: SCALAR_TYPE,
            raw_index: SCALAR_TYPE,
        }),
        SemanticCallableDeclV1::defined(SemanticFunctionIdV1::from_index(1)),
    ]
}

fn function(
    first: Vec<SemanticStatementV1>,
    first_terminator: SemanticTerminatorKindV1,
    second: Vec<SemanticStatementV1>,
    second_arguments: Vec<SemanticOperandV1>,
    ownership: SemanticSourceArgumentOwnershipV1,
) -> SemanticFunctionDeclV1 {
    let abi = SemanticFunctionAbiV1::new(
        SemanticAbiIdentityV1::from_sha256(bytes(210)),
        SemanticLayoutIdentityV1::from_sha256(bytes(210)),
        SemanticCanonAbiV1::GpuKernel,
        false,
        false,
        vec![
            SemanticAbiValueV1::new(
                POINTER_TYPE,
                SemanticAbiPassModeV1::Direct(SemanticAbiValueAttributesV1::plain())
            );
            2
        ],
        SemanticAbiValueV1::new(SCALAR_TYPE, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap()
    .with_source_argument_ownership(vec![
        ownership,
        SemanticSourceArgumentOwnershipV1::RawPointer,
    ])
    .unwrap();
    SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256(bytes(211)),
        SemanticFunctionRoleV1::KernelRoot,
        SemanticItemDefinitionIdentityV1::from_sha256(bytes(212)),
        SemanticMonomorphizationIdentityV1::from_sha256(bytes(213)),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256(bytes(214)),
        SemanticConstGenericArgumentsIdentityV1::from_sha256(bytes(215)),
        SemanticSourceProvenanceV1::unavailable(),
        abi,
        vec![
            local(210, SCALAR_TYPE, SemanticLocalRoleV1::Return),
            local(211, POINTER_TYPE, SemanticLocalRoleV1::Argument(0)),
            local(212, POINTER_TYPE, SemanticLocalRoleV1::Temporary),
            local(213, POINTER_TYPE, SemanticLocalRoleV1::Temporary),
            local(214, POINTER_TYPE, SemanticLocalRoleV1::Temporary),
            local(215, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            local(216, POINTER_TYPE, SemanticLocalRoleV1::Argument(1)),
            local(217, POINTER_TYPE, SemanticLocalRoleV1::Temporary),
        ],
        SemanticBlockIdV1::from_index(0),
        vec![
            block(210, first, first_terminator),
            block(211, second, call(0, second_arguments, 2)),
            block(212, vec![], SemanticTerminatorKindV1::Return),
        ],
    )
    .unwrap()
}

fn ordinary(first: Vec<SemanticStatementV1>) -> SemanticFunctionDeclV1 {
    function(
        first,
        SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 1)),
        vec![borrow(3, 2)],
        vec![operand(3), constant(0)],
        SemanticSourceArgumentOwnershipV1::ExclusiveOwner,
    )
}

fn with_parts(
    base: &SemanticFunctionDeclV1,
    abi: SemanticFunctionAbiV1,
    locals: Vec<SemanticLocalDeclV1>,
) -> SemanticFunctionDeclV1 {
    SemanticFunctionDeclV1::new(
        base.identity(),
        base.role(),
        base.item_definition_identity(),
        base.monomorphization_identity(),
        base.generic_type_arguments_identity(),
        base.const_generic_arguments_identity(),
        base.source(),
        abi,
        locals,
        base.entry(),
        base.blocks().to_vec(),
    )
    .unwrap()
}

fn projection_types() -> Vec<SemanticTypeDeclV1> {
    vec![
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(1)),
            SemanticLayoutIdentityV1::from_sha256(bytes(1)),
            SemanticTypeLayoutV1::new(Some(4), 4).unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 32,
            }),
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(2)),
            SemanticLayoutIdentityV1::from_sha256(bytes(2)),
            SemanticTypeLayoutV1::new(Some(16), 4).unwrap(),
            SemanticTypeShapeV1::Array {
                element: SCALAR_TYPE,
                length: 4,
            },
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(3)),
            SemanticLayoutIdentityV1::from_sha256(bytes(3)),
            SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
            SemanticTypeShapeV1::Pointer(
                SemanticPointerTypeV1::new(
                    SCALAR_TYPE,
                    SemanticMutabilityV1::Mutable,
                    1,
                    64,
                    SemanticPointerMetadataV1::None,
                )
                .unwrap(),
            ),
        ),
    ]
}

struct SyntheticConsumer<'b, 'w> {
    budget: &'b mut Budget<'w>,
}
impl NominalCapabilityConsumerV1 for SyntheticConsumer<'_, '_> {
    fn charge_work_v1(&mut self, amount: usize) -> Result<()> {
        self.budget.charge_work(amount).map_err(Error::Resource)
    }
    fn reserve_storage_v1(&mut self, amount: usize) -> Result<()> {
        self.budget.reserve_storage(amount).map_err(Error::Resource)
    }
    fn release_storage_v1(&mut self, amount: usize) -> Result<()> {
        self.budget.release_storage(amount).map_err(Error::Resource)
    }
    fn ledger_v1(&self) -> NominalCapabilityLedgerV1 {
        NominalCapabilityLedgerV1 {
            slot: self.budget as *const Budget<'_> as usize,
            identity: self.budget.work_ledger_identity_v1(),
            work: self.budget.work(),
            storage: self.budget.storage(),
            peak: self.budget.peak_storage(),
            denied_work: self.budget.failed_work().is_some(),
            denied_storage: self.budget.failed_storage().is_some(),
        }
    }
    fn with_nominal_call_v1(
        &mut self,
        _: usize,
        _: &SemanticDirectCallV1,
        _: SemanticSourceProvenanceV1,
        _: &mut NominalCallVisitorV1<'_>,
    ) -> Result<()> {
        Err(Error::Unavailable(
            "synthetic consumer has no actual nominal candidate",
        ))
    }
}

const LIMIT: usize = 8 * 1024 * 1024;
fn function_with_context_and_dead_local() -> (SemanticFunctionDeclV1, Vec<SemanticCallableDeclV1>) {
    let function = projection_function_with_locals(
        vec![
            block(
                1,
                vec![],
                SemanticTerminatorKindV1::Call(
                    SemanticDirectCallV1::new_callable(
                        SemanticCallableIdV1::from_index(0),
                        vec![],
                        Some(SemanticCallDestinationV1::new(
                            place(1),
                            cfg_edge(SemanticEdgeRoleV1::CallReturn, 1),
                        )),
                        SemanticUnwindActionV1::Unreachable,
                    )
                    .unwrap(),
                ),
            ),
            block(
                2,
                vec![statement(SemanticStatementKindV1::StorageDead(
                    SemanticLocalIdV1::from_index(1),
                ))],
                SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 2)),
            ),
            block(3, vec![], SemanticTerminatorKindV1::Return),
        ],
        vec![
            local(1, SCALAR_TYPE, SemanticLocalRoleV1::Return),
            local(2, POINTER_TYPE, SemanticLocalRoleV1::Temporary),
        ],
    );
    let callables = vec![compiler_intrinsic_callable(
        SemanticCompilerIntrinsicOperationV1::MatrixContextCurrent {
            context: POINTER_TYPE,
        },
    )];
    (function, callables)
}
fn allocation(origin: u64) -> AllocationContractV1 {
    AllocationContractV1 {
        allocation_origin: origin,
        noalias_class: 1,
        writable: false,
        singleton_object: false,
    }
}
#[test]
fn actual_no_pipeline_owner_scan_matches_unchanged_owner_analyzer() {
    let (function, callables) = function_with_context_and_dead_local();
    let expected = workgroup_pipeline_local_owners_v1(&callables, &function).unwrap();
    let mut pending = RetainedNominalCapabilityDriverV1::new();
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    {
        let mut consumer = SyntheticConsumer {
            budget: &mut budget,
        };
        fill_none(
            &mut pending.pipeline_owners,
            function.locals().len(),
            &mut consumer,
            &mut owned,
        )
        .unwrap();
        owner_scan(
            &function,
            &callables,
            &mut pending.pipeline_owners,
            &mut pending.owner_scan_blocks,
            &mut pending.alias_scan_locals,
            &mut consumer,
        )
        .unwrap();
    }
    assert_eq!(pending.pipeline_owners, expected);
    assert_eq!(pending.owner_scan_blocks, function.blocks().len());
    assert_eq!(pending.alias_scan_locals, function.locals().len());
    assert!(budget.work() > 0 && budget.storage() == owned);
    assert_eq!(pending.original_work, 0);
    drop(pending);
    budget.release_storage(owned).unwrap();
    assert_eq!(budget.storage(), 0);
}
#[test]
fn actual_empty_payload_scan_uses_complete_original_initial_pass_and_keeps_legacy_work() {
    let (function, callables) = function_with_context_and_dead_local();
    let types = projection_types();
    let dominance = SemanticEnumPayloadDominanceV1::analyze(&function, &types).unwrap();
    let scalar = assertion_definition_inventory(&function).unwrap();
    let provenance = local_provenance_with_scalar_inventory_v1(
        &callables,
        &types,
        &function,
        &scalar.counts,
        &scalar.address_escaped,
    )
    .unwrap();
    let allocations =
        local_allocation_contracts(&types, &function, &provenance.allocation_origins).unwrap();
    let constants = constant_locals(&function).unwrap();
    let owners = workgroup_pipeline_local_owners_v1(&callables, &function).unwrap();
    let empty = HashMap::new();
    let mut old_work = 0;
    let original_entries = propagate_capability_dataflow_v1(
        &callables,
        &function,
        &dominance,
        &allocations,
        &constants,
        &owners,
        &empty,
        function.entry().index() as usize,
        &mut old_work,
    )
    .unwrap();
    let before_payload = old_work;
    let expected = collect_workgroup_pipeline_payloads_v1(
        &callables,
        &types,
        &function,
        &dominance,
        &owners,
        &original_entries,
        &mut old_work,
    )
    .unwrap();
    assert_eq!(before_payload, old_work);
    let mut pass = RetainedCapabilityFifoPassV1::new();
    let mut payloads = HashMap::new();
    let mut blocks_seen = 0;
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    let mut actual_work = 0;
    {
        let mut consumer = SyntheticConsumer {
            budget: &mut budget,
        };
        pass.prepare_into(
            &function,
            &mut consumer,
            &mut owned,
            &mut actual_work,
            |block, scratch, _| {
                let mut state = DenseCapabilityStateV1::new(scratch);
                transfer_capability_statements_v1(&function, block, &mut state, &dominance)
                    .map_err(transfer_error)?;
                let effects = transfer_capability_terminator_v1(
                    &callables,
                    &function,
                    block,
                    &mut state,
                    &allocations,
                    &constants,
                    &owners,
                    &empty,
                    false,
                )
                .map_err(transfer_error)?;
                drop(effects);
                state.check_bounds_v1()
            },
        )
        .unwrap();
        let initial = pass.completed_for(&function, &consumer).unwrap();
        for (block, row) in original_entries.iter().enumerate() {
            assert_eq!(initial.reached[block], row.is_some());
            for local in 0..function.locals().len() {
                assert_eq!(
                    initial.entries[block * function.locals().len() + local],
                    row.as_ref().and_then(|row| row.get(&local)).copied()
                );
            }
        }
        payload_scan(
            &function,
            &callables,
            &initial,
            &owners,
            &mut payloads,
            &mut blocks_seen,
            &mut consumer,
        )
        .unwrap();
    }
    assert_eq!(payloads, expected);
    assert_eq!(actual_work, old_work);
    assert_eq!(blocks_seen, function.blocks().len());
    assert!(budget.work() > actual_work);
    drop(pass);
    drop(payloads);
    budget.release_storage(owned).unwrap();
}
#[test]
fn owner_and_payload_scans_reject_actual_pipeline_operations_without_claiming_support() {
    let (function, _) = function_with_context_and_dead_local();
    let creates = vec![compiler_intrinsic_callable(
        SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate {
            scope: POINTER_TYPE,
            pipeline: POINTER_TYPE,
            buffers: 2,
            elements: 4,
            prefetch_distance: 1,
        },
    )];
    let writes = vec![compiler_intrinsic_callable(
        SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineWrite {
            pipeline: POINTER_TYPE,
            element: SCALAR_TYPE,
        },
    )];
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owners = vec![None; function.locals().len()];
    let mut blocks = 0;
    let mut locals = 0;
    let mut consumer = SyntheticConsumer {
        budget: &mut budget,
    };
    assert!(
        owner_scan(
            &function,
            &creates,
            &mut owners,
            &mut blocks,
            &mut locals,
            &mut consumer
        )
        .is_err()
    );
    assert_eq!(blocks, 1);
    assert_eq!(locals, 0);
    let reached = vec![false; function.blocks().len()];
    let initial = CompletedFifoPassV1 {
        entries: &[],
        reached: &reached,
        visits: &[],
    };
    let mut payloads = HashMap::new();
    let mut payload_blocks = 0;
    assert!(
        payload_scan(
            &function,
            &writes,
            &initial,
            &owners,
            &mut payloads,
            &mut payload_blocks,
            &mut consumer
        )
        .is_err()
    );
    assert_eq!(payload_blocks, 1);
    assert!(payloads.is_empty());
}
#[test]
fn owner_denial_prefix_retains_actual_scan_progress_and_attached_storage() {
    let (function, callables) = function_with_context_and_dead_local();
    for limit in 0..6 {
        let mut pending = RetainedNominalCapabilityDriverV1::new();
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        // Construction outside the helper's work-limit experiment is synthetic, not driver admission.
        pending.pipeline_owners = vec![None; function.locals().len()];
        let result = owner_scan(
            &function,
            &callables,
            &mut pending.pipeline_owners,
            &mut pending.owner_scan_blocks,
            &mut pending.alias_scan_locals,
            &mut SyntheticConsumer {
                budget: &mut budget,
            },
        );
        assert!(result.is_err() && budget.failed_work().is_some());
        assert!(pending.owner_scan_blocks <= function.blocks().len());
        assert!(pending.alias_scan_locals <= function.locals().len());
        assert_eq!(pending.pipeline_owners, vec![None; function.locals().len()]);
        assert_eq!(owned, 0);
        drop(pending);
        budget.release_storage(owned).unwrap();
    }
}
#[test]
fn read_binding_full_data_matches_original_and_retains_actual_terminator_sources() {
    let (function, _) = function_with_context_and_dead_local();
    let reads = [Some(allocation(10)), None, Some(allocation(20))];
    let expected = bind_capability_read_effects_to_call_blocks_v1(&function, &reads).unwrap();
    let mut pending = RetainedNominalCapabilityDriverV1::new();
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    bind_reads(
        &function,
        &reads,
        &mut pending.bound_reads,
        &mut SyntheticConsumer {
            budget: &mut budget,
        },
        &mut owned,
    )
    .unwrap();
    assert_eq!(pending.bound_reads, expected);
    assert_eq!(
        pending.bound_reads[0].unwrap().allocation.allocation_origin,
        10
    );
    assert_eq!(
        pending.bound_reads[2].unwrap().source,
        function.blocks()[2].terminator().source()
    );
    assert_eq!(pending.bound_reads[1], None);
    drop(pending);
    budget.release_storage(owned).unwrap();
}
#[test]
fn read_binding_shape_refusal_precedes_admission_and_partial_work_keeps_rows() {
    let (function, _) = function_with_context_and_dead_local();
    let reads = [Some(allocation(10)), None, Some(allocation(20))];
    let mut work = Work::new(1);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    let mut bound = Vec::new();
    assert!(
        bind_reads(
            &function,
            &reads[..2],
            &mut bound,
            &mut SyntheticConsumer {
                budget: &mut budget
            },
            &mut owned
        )
        .is_err()
    );
    assert_eq!((budget.work(), owned, bound.len()), (0, 0, 0));
    assert!(
        bind_reads(
            &function,
            &reads,
            &mut bound,
            &mut SyntheticConsumer {
                budget: &mut budget
            },
            &mut owned
        )
        .is_err()
    );
    assert_eq!(bound.len(), 1);
    assert_eq!(bound[0].unwrap().allocation.allocation_origin, 10);
    assert!(budget.failed_work().is_some() && budget.storage() == owned && owned > 0);
    drop(bound);
    budget.release_storage(owned).unwrap();
}
#[test]
fn prepay_uses_only_actual_budget_and_preserves_original_semantic_counter() {
    let (function, _) = function_with_context_and_dead_local();
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    prepay_transfer(
        &function,
        1,
        function.locals().len(),
        &mut SyntheticConsumer {
            budget: &mut budget,
        },
    )
    .unwrap();
    assert_eq!(budget.work(), 64 + function.locals().len() + 8 + 256);
    let mut original_work = 7;
    original_charge(
        &mut SyntheticConsumer {
            budget: &mut budget,
        },
        &mut original_work,
        3,
    )
    .unwrap();
    assert_eq!(original_work, 10);
    assert_eq!(budget.work(), 64 + function.locals().len() + 8 + 256 + 3);
}
#[test]
fn exact_and_short_reservation_keeps_all_previous_driver_fields() {
    for short in [false, true] {
        let mut pending = RetainedNominalCapabilityDriverV1::new();
        let first = 2 * size_of::<Option<usize>>();
        let second = 3 * size_of::<Option<AllocationContractV1>>();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, first + second - usize::from(short));
        let mut owned = 0;
        fill_none(
            &mut pending.pipeline_owners,
            2,
            &mut SyntheticConsumer {
                budget: &mut budget,
            },
            &mut owned,
        )
        .unwrap();
        let pointer = pending.pipeline_owners.as_ptr();
        let result = fill_none(
            &mut pending.global_reads,
            3,
            &mut SyntheticConsumer {
                budget: &mut budget,
            },
            &mut owned,
        );
        assert_eq!(result.is_ok(), !short);
        assert_eq!(pending.pipeline_owners.as_ptr(), pointer);
        assert_eq!(pending.pipeline_owners, vec![None, None]);
        assert_eq!(owned, if short { first } else { first + second });
        assert_eq!(budget.storage(), owned);
        drop(pending);
        budget.release_storage(owned).unwrap();
        assert_eq!(budget.storage(), 0);
    }
}
#[test]
fn returned_error_and_panic_keep_attached_driver_payload_until_drop_then_refund() {
    for mode in [1, 2] {
        let mut pending = RetainedNominalCapabilityDriverV1::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
            fill_none(
                &mut pending.global_reads,
                3,
                &mut SyntheticConsumer {
                    budget: &mut budget,
                },
                &mut owned,
            )?;
            pending.global_reads[0] = Some(allocation(9));
            if mode == 1 {
                return Err(Error::Unavailable("synthetic retained driver error"));
            }
            std::panic::panic_any(());
        }));
        match outcome {
            Ok(result) => assert!(result.is_err()),
            Err(payload) => drop(payload),
        }
        assert_eq!(pending.global_reads[0].unwrap().allocation_origin, 9);
        assert!(budget.storage() == owned && owned > 0);
        drop(pending);
        budget.release_storage(owned).unwrap();
        assert_eq!(budget.storage(), 0);
    }
}
#[test]
fn explicit_driver_rows_include_unchanged_transfer_policy_and_checked_sum() {
    let rows = driver_frame_rows().unwrap();
    assert_eq!(rows.len(), DRIVER_FRAME_ROWS);
    assert_eq!(rows[0], 4096);
    assert_eq!(rows.into_iter().sum::<usize>(), driver_frame().unwrap());
}
#[test]
fn driver_source_order_uses_actual_query_and_retains_both_passes_without_refund() {
    let source: String = include_str!("bf16_nominal_retained_capability_driver_v1.rs")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let start = source.find("pub(insuper::super)fnprepare_into(").unwrap();
    let end = source.find("pub(insuper::super)fncompleted_for").unwrap();
    let body = &source[start..end];
    let sequence = [
        "self.prepare_profile(",
        "owner_scan(",
        "self.initial.prepare_into(",
        "payload_scan(",
        "self.repeated.prepare_into(",
        "fill_none(&mutself.layouts,",
        "forblockin0..blocks",
        "NominalCapabilityPassV1::Final",
        "bind_reads(",
        "self.phase=Phase::Complete",
    ];
    let mut at = 0;
    for marker in sequence {
        at += body[at..].find(marker).unwrap() + marker.len();
    }
    assert!(source.contains("transfer_nominal(pass,site,&mutstate,consumer,visit,run)?"));
    assert!(
        body.contains("self.run.query_visits[0]>0") && body.contains("self.run.query_visits[1]>0")
    );
    for forbidden in [
        "release_storage_v1(",
        "Budget::new",
        "Work::new",
        ".initial.take(",
        ".repeated.take(",
        "query_visits==[1,1,1]",
        "for&blockin&self.validation_order",
    ] {
        assert!(!source.contains(forbidden));
    }
}

use super::production_call_instances_v1::with_production_call_instances_v1;
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1;
use fe2o3_mir_model::semantic_mir_v1::SemanticCallableIdV1;

pub(super) const LIMIT: usize = 1_000_000;
const FLOOR: usize = 37;
const ROOT: SemanticFunctionIdV1 = SemanticFunctionIdV1::from_index(0);

#[allow(dead_code)]
struct SlotFields {
    budget: usize,
    private: usize,
    calls: Option<usize>,
    lifecycle: Option<usize>,
}
#[allow(dead_code)]
struct SourceFields {
    source: ExecutionCallSourceV29,
    instance: ProductionCallInstanceIdV1,
    function: usize,
    ssa: usize,
    references: Option<usize>,
    identities: Option<usize>,
}
#[allow(dead_code)]
struct PrivateFields {
    roots: usize,
    operations: usize,
    active: Option<(
        usize,
        usize,
        usize,
        Option<fe2o3_kernel_ir::CanonicalKernelIrWorkLimitV1>,
    )>,
}
#[allow(dead_code)]
pub(super) struct GrowthFields {
    custody: &'static source_storage_v29::SourceStorageRootCustodyViewV29<'static, 'static>,
    owned: usize,
    table_owned: usize,
}
#[allow(dead_code)]
struct SealFields {
    slots: SlotFields,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    storage: usize,
    work: usize,
    credit: usize,
    function: usize,
    types: usize,
    type_count: usize,
    callables: usize,
    callable_count: usize,
    owner: SemanticFunctionIdV1,
    semantic_function: SemanticFunctionIdV1,
    source: Option<SourceFields>,
    placement: SemanticEmissionPlacementV1,
    next_value: u32,
    emitted_operations: usize,
    max_operations: usize,
    private_state: PrivateFields,
    growth: Option<GrowthFields>,
}
#[allow(dead_code)]
struct ServiceFields {
    budget: Option<&'static mut dyn SemanticEmissionBudgetV1>,
    calls: Option<&'static mut dyn ExecutionDefinedCallConsumerV29>,
    lifecycle: Option<&'static mut dyn ExecutionLifecycleConsumerV29>,
    private: PrivateArrayRecorderWorkV1<'static>,
}
#[allow(dead_code)]
struct DetachedFields {
    lowering: SemanticFunctionLoweringV1<'static, 'static>,
    seal: SealFields,
}

fn independent_service_headers() -> usize {
    use std::mem::size_of;
    assert_eq!(size_of::<SlotFields>(), size_of::<EmissionServiceSlotsV1>());
    assert_eq!(
        size_of::<SourceFields>(),
        size_of::<EmissionServiceSourceV1>()
    );
    assert_eq!(
        size_of::<PrivateFields>(),
        size_of::<EmissionPrivateStateV1>()
    );
    assert_eq!(
        size_of::<GrowthFields>(),
        size_of::<EmissionServiceGrowthV1<'static>>()
    );
    assert_eq!(
        size_of::<SealFields>(),
        size_of::<EmissionServiceSealV1<'static>>()
    );
    assert_eq!(
        size_of::<ServiceFields>(),
        size_of::<EmissionServicesV1<'static>>()
    );
    assert_eq!(
        size_of::<DetachedFields>(),
        size_of::<DetachedEmissionStateV1<'static>>()
    );
    let result = size_of::<SealFields>()
        + size_of::<ServiceFields>()
        + size_of::<Result<(DetachedFields, ServiceFields), ProductionSemanticKirErrorV1>>();
    assert_eq!(result, emission_service_headers_v1().unwrap());
    result
}

#[test]
fn actual_detached_source_owner_keeps_growth_separate_and_denies_restored_lost_credit() {
    for growth in [false, true] {
        for lose in [false, true] {
            for ignore in [false, true] {
                let mut owner = ProductionSemanticSsaOwnerV1::try_new(
                    resource_tests::helper_closure_semantic_owner(),
                    ProductionSemanticSsaLimitsV1::default(),
                )
                .unwrap();
                let mut work = CanonicalKernelIrWorkBudgetV1::new(LIMIT);
                let mut budget = ArgumentBudgetV1::new(&mut work, LIMIT);
                budget.reserve_storage(FLOOR).unwrap();
                assert!(budget.reserve_storage(LIMIT).is_err());
                let denial = budget.failed_storage();
                let capture = owner
                    .try_capture_occurrences_with_budget_v1(&mut budget)
                    .unwrap();
                budget.reserve_storage(capture.retained_storage()).unwrap();
                let unit = owner.source_semantic().functions()[1].locals()[0].ty();
                assert_eq!(
                    owner.source_semantic().types()[unit.index() as usize]
                        .layout()
                        .size_bytes(),
                    Some(0)
                );
                // This is the actual unit source type, not a stand-in physical
                // row for a nominal object. The concrete-root test chooses it
                // explicitly; full production demand selection is tested below.
                let mut layouts =
                    source_storage_v29::SourceStorageLayoutsV29::new(&owner, &[unit], &mut budget)
                        .unwrap();
                let reached = std::cell::Cell::new(false);
                with_production_call_instances_v1(&owner, ROOT, &mut budget, |instances, budget| {
                    let table_floor = budget.storage();
                    let result = source_storage_v29::with_source_storage_root_v29(
                        &mut layouts, instances, budget, |references, root, budget| {
                            let frame_floor = budget.storage();
                            let child = instances.calls(instances.root()).unwrap()[0].child().unwrap();
                            let row = instances.instance(child).unwrap();
                            let semantic = instances.owner().source_semantic();
                            let function = row.declaration();
                            let emission = source_reference_optional_emission_v29(Some(references), budget)?.unwrap();
                            let cursor = ExecutionAvailabilityV29::new_with_references(
                                instances, child, Some(&emission), budget,
                            )?;
                            let mut private = PrivateArrayLazyBudgetV1::new(1, 256);
                            let signatures = BTreeMap::new();
                            let returns = CallReturnBufferV1::for_function(
                                function, semantic.callables(), &signatures, 0, budget,
                            )?;
                            let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
                                semantic.types(), semantic.callables(), function, row.ssa(),
                                ROOT, row.function(), BTreeMap::new(), signatures, Vec::new(),
                                SemanticParameterBindingsV1 {
                                    declarations: &[], values: &[], types: &[], local_bindings: Some(&[]),
                                },
                                None, Some([64, 1, 1]), BTreeSet::new().into(), 1, false, 256,
                                PrivateArrayRecorderWorkV1::Shared(&mut private), None, returns,
                                Some(budget), SemanticEmissionPlacementV1::default(), Some(cursor), None,
                                None,
)?;
                            let mut target = BasicBlock::new(BlockId(0));
                            lowering.begin_block(function.entry(), &mut target)?;
                            target.terminator = Some(lowering.lower_terminator(
                                function.entry(), function.blocks()[0].terminator().kind(), &mut target.operations,
                            )?);
                            lowering.with_emission_budget_v1(|this, budget| this.execution.as_mut().unwrap().finish_block(budget))?;
                            assert!(target.operations.is_empty());
                            let before = lowering.emission_work.as_deref().unwrap().storage();
                            let (detached, services) = lowering.detach_services_v1()?;
                            let seal_floor = detached.seal.storage;
                            let seal_credit = detached.seal.credit;
                            let EmissionServicesV1 { budget: borrowed_budget, calls, lifecycle, private: private_work } = services;
                            assert!(calls.is_none() && lifecycle.is_none());
                            drop((borrowed_budget, calls, lifecycle, private_work));
                            assert_eq!(budget.storage(), seal_floor);
                            assert_eq!(seal_floor, before + seal_credit);
                            let growth_start = budget.storage();
                            if growth {
                                let state = root.new_state(child, SemanticLocalIdV1::from_index(0), budget)?;
                                let path = root.root_path(unit, budget)?;
                                root.mutate(state, path, source_storage_v29::SourceStorageRootMutationV29::Initialize, budget)?;
                                let copy = root.copy_state(state, budget)?;
                                assert!(root.is_initialized(copy, path, budget)?);
                            }
                            let growth_bytes = budget.storage() - growth_start;
                            assert_eq!(growth_bytes > 0, growth);
                            if lose {
                                budget.release_storage(1)?;
                                if growth {
                                    assert!(budget.storage() >= seal_floor, "old absolute seal floor still passes");
                                }
                            }
                            let before_attach = budget.storage();
                            let result = detached.attach(EmissionServicesV1 {
                                budget: Some(budget), calls: None, lifecycle: None,
                                private: PrivateArrayRecorderWorkV1::Shared(&mut private),
                            });
                            assert_eq!(result.is_err(), lose);
                            drop(result);
                            drop(private);
                            drop(target);
                            drop(emission);
                            if lose {
                                assert_eq!(budget.storage(), before_attach, "refused attachment refunds no payload");
                                assert!(!root.permits_cleanup_refund(instances, &references.failure, 0, budget));
                                assert!(references.failure.first_error().is_none(), "cleanup denial does not select a semantic error");
                                budget.reserve_storage(1)?;
                                assert!(!root.permits_cleanup_refund(instances, &references.failure, 0, budget));
                            } else {
                                assert_eq!(budget.storage(), before + growth_bytes);
                                let own = before - frame_floor;
                                assert!(root.permits_cleanup_refund(instances, &references.failure, own, budget));
                                budget.release_storage(own)?;
                                assert_eq!(budget.storage(), frame_floor + growth_bytes);
                            }
                            reached.set(true);
                            if lose && !ignore {
                                Err(source_reference_error_v29("detached frame selected source failure").into())
                            } else {
                                Ok(())
                            }
                        },
                    );
                    assert!(reached.get(), "a caught setup failure is not detach coverage");
                    match result {
                        Ok(()) => assert!(!lose),
                        Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(ArgumentResourceV1::Accounting)) => assert!(lose && ignore),
                        Err(ProductionSemanticKirErrorV1::Unsupported { detail: "detached frame selected source failure", .. }) => assert!(lose && !ignore),
                        other => panic!("unexpected root settlement: {other:?}"),
                    }
                    if !lose { assert_eq!(budget.storage(), table_floor); }
                    assert_eq!(budget.failed_storage(), denial);
                    Ok::<_, production_call_instances_v1::ProductionCallInstanceErrorV1>(())
                }).unwrap();
                assert!(reached.get());
                let before = budget.storage();
                assert_eq!(layouts.release(&mut budget).is_err(), lose);
                if lose {
                    assert_eq!(budget.storage(), before);
                }
                assert_eq!(budget.failed_storage(), denial);
                drop(owner);
                budget.release_storage(budget.storage() - FLOOR).unwrap();
                assert_eq!(budget.storage(), FLOOR);
            }
        }
    }
}

pub(super) struct Probe {
    pub(super) result: Result<(), ProductionSemanticKirErrorV1>,
    pub(super) work: usize,
    pub(super) storage: usize,
    pub(super) failed_storage: Option<usize>,
}

// Cursor/transport fixture using the actual original admitted helper. Full root
// emission and its final source census are exercised by function_frame_tests.
pub(super) fn with_completed_helper(
    consume: impl for<'source, 'service> FnOnce(
        SemanticFunctionLoweringV1<'source, 'service>,
        ProductionCallInstanceIdV1,
        &'source SemanticPlaceV1,
    ) -> Result<(), ProductionSemanticKirErrorV1>,
) -> Probe {
    with_helper_boundary(true, consume)
}

fn with_helper_boundary(
    complete: bool,
    consume: impl for<'source, 'service> FnOnce(
        SemanticFunctionLoweringV1<'source, 'service>,
        ProductionCallInstanceIdV1,
        &'source SemanticPlaceV1,
    ) -> Result<(), ProductionSemanticKirErrorV1>,
) -> Probe {
    let mut owner = ProductionSemanticSsaOwnerV1::try_new(
        resource_tests::helper_closure_semantic_owner(),
        ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap();
    let mut work = CanonicalKernelIrWorkBudgetV1::new(LIMIT);
    assert!(work.charge_work(LIMIT + 3).is_err());
    let mut budget = ArgumentBudgetV1::new(&mut work, LIMIT);
    budget.reserve_storage(FLOOR).unwrap();
    assert!(budget.reserve_storage(LIMIT + 7).is_err());
    let capture = owner
        .try_capture_occurrences_with_budget_v1(&mut budget)
        .unwrap();
    budget.reserve_storage(capture.retained_storage()).unwrap();
    let mut reached = false;
    let probe = with_production_call_instances_v1(
        &owner,
        ROOT,
        &mut budget,
        |instances, budget| -> Result<Probe, ProductionSemanticKirErrorV1> {
            let floor = budget.storage();
            let child = instances.calls(instances.root()).unwrap()[0]
                .child()
                .unwrap();
            let row = instances.instance(child).unwrap();
            let semantic = instances.owner().source_semantic();
            let function = row.declaration();
            assert_eq!(row.function().index(), 1);
            assert_eq!(function.blocks().len(), 1);
            let cursor = ExecutionAvailabilityV29::new(instances, child, budget)?;
            let mut private = PrivateArrayLazyBudgetV1::new(1, 256);
            let signatures = BTreeMap::new();
            let returns = CallReturnBufferV1::for_function(
                function,
                semantic.callables(),
                &signatures,
                0,
                budget,
            )?;
            let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
                semantic.types(),
                semantic.callables(),
                function,
                row.ssa(),
                ROOT,
                row.function(),
                BTreeMap::new(),
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
                PrivateArrayRecorderWorkV1::Shared(&mut private),
                None,
                returns,
                Some(budget),
                SemanticEmissionPlacementV1::default(),
                Some(cursor),
                None,
                None,
)?;
            let mut target = BasicBlock::new(BlockId(0));
            lowering.begin_block(function.entry(), &mut target)?;
            target.terminator = Some(lowering.lower_terminator(
                function.entry(),
                function.blocks()[0].terminator().kind(),
                &mut target.operations,
            )?);
            assert!(target.operations.is_empty());
            if complete {
                lowering.with_emission_budget_v1(|this, budget| {
                    this.execution.as_mut().unwrap().finish_block(budget)
                })?;
            }
            let SemanticTerminatorKindV1::Call(call) =
                semantic.functions()[0].blocks()[0].terminator().kind()
            else {
                panic!()
            };
            let result = consume(
                lowering,
                instances.root(),
                call.destination().unwrap().place(),
            );
            drop(private);
            let probe = Probe {
                result,
                work: budget.work(),
                storage: budget.storage(),
                failed_storage: budget.failed_storage(),
            };
            // The test consumer has destroyed the lowerer and detached services;
            // no C2 arena or emitted output is retained by this transport fixture.
            budget.release_storage(budget.storage() - floor)?;
            reached = true;
            Ok(probe)
        },
    )
    .unwrap();
    assert!(reached);
    drop(owner);
    budget.release_storage(capture.retained_storage()).unwrap();
    assert_eq!(budget.storage(), FLOOR);
    assert_eq!(budget.failed_storage(), Some(FLOOR + LIMIT + 7));
    drop(budget);
    assert_eq!(work.failed_work(), Some(LIMIT + 3));
    probe
}

#[test]
fn actual_completed_cursor_detaches_and_rebinds_only_its_original_services() {
    let before = std::cell::Cell::new(0);
    let probe = with_completed_helper(|lowering, _, _| {
        let source = lowering.function;
        let types = lowering.types;
        let callables = lowering.callables;
        let cursor = lowering.execution.as_ref().unwrap();
        let instance = cursor.instance;
        assert_eq!(cursor.visited, vec![true]);
        let local_count = lowering.locals.len();
        before.set(lowering.emission_work.as_deref().unwrap().storage());
        let (mut detached, mut services) = lowering.detach_services_v1()?;
        assert!(detached.lowering.emission_work.is_none());
        assert!(detached.lowering.private_arrays.work.activate().is_err());
        assert!(
            detached
                .lowering
                .private_arrays
                .work
                .charge_private_array_work(0)
                .is_err()
        );
        let storage = services.budget.as_deref().unwrap().storage();
        assert_eq!(storage, before.get() + emission_service_headers_v1()?);
        services
            .budget
            .as_deref_mut()
            .unwrap()
            .reserve_storage(13)?;
        services
            .budget
            .as_deref_mut()
            .unwrap()
            .release_storage(13)?;
        let mut attached = detached.attach(services)?;
        assert!(std::ptr::eq(attached.function, source));
        assert!(std::ptr::eq(attached.types, types));
        assert!(std::ptr::eq(attached.callables, callables));
        assert_eq!(attached.locals.len(), local_count);
        assert_eq!(attached.execution.as_ref().unwrap().instance, instance);
        assert_eq!(attached.execution.as_ref().unwrap().visited, vec![true]);
        assert_eq!(
            attached.emission_work.as_deref().unwrap().storage(),
            before.get()
        );
        attached.with_emission_budget_v1(|this, budget| {
            this.execution.as_mut().unwrap().finish(budget)
        })?;
        drop(attached);
        Ok(())
    });
    assert!(probe.result.is_ok(), "{:?}", probe.result);
    assert_eq!(probe.storage, before.get());
}

#[test]
fn service_boundary_refuses_foreign_source_placement_and_private_history() {
    for fault in 0..9 {
        let reached = std::cell::Cell::new(false);
        let probe = with_completed_helper(|lowering, root, _| {
            let (mut detached, mut services) = lowering.detach_services_v1()?;
            let cloned = detached.lowering.function.clone();
            let mut foreign = PrivateArrayLazyBudgetV1::new(1, 256);
            if fault == 6 {
                let EmissionServicesV1 {
                    mut budget,
                    calls,
                    lifecycle,
                    private,
                } = services;
                assert!(calls.is_none() && lifecycle.is_none());
                drop((calls, lifecycle, private));
                let result = detached.attach(EmissionServicesV1 {
                    budget: budget
                        .as_mut()
                        .map(|row| &mut **row as &mut dyn SemanticEmissionBudgetV1),
                    calls: None,
                    lifecycle: None,
                    private: PrivateArrayRecorderWorkV1::Shared(&mut foreign),
                });
                assert!(result.is_err(), "foreign private owner accepted");
                drop(result);
                reached.set(true);
                return Ok(());
            }
            match fault {
                0 => detached.lowering.next_value += 1,
                1 => detached.lowering.emission_placement.first_value += 1,
                2 => detached.lowering.semantic_function = ROOT,
                3 => detached.lowering.types = &[],
                4 => detached.lowering.callables = &[],
                5 => detached.lowering.execution.as_mut().unwrap().instance = root,
                7 => {
                    let PrivateArrayRecorderWorkV1::Shared(private) = &mut services.private else {
                        panic!()
                    };
                    private.operations += 1;
                }
                8 => {
                    // An equal-looking declaration cannot replace the source
                    // pointer named by the inert comparison seal.
                    detached.seal.function = &cloned as *const SemanticFunctionDeclV1 as usize;
                }
                _ => unreachable!(),
            }
            let result = detached.attach(services);
            assert!(result.is_err(), "fault {fault} accepted");
            reached.set(true);
            Ok(())
        });
        assert!(probe.result.is_ok(), "fault {fault}: {:?}", probe.result);
        assert!(reached.get());
    }
}

struct UnexpectedService(std::cell::Cell<usize>);
impl UnexpectedService {
    fn refuse<T>(&self) -> Result<T, ProductionSemanticKirErrorV1> {
        self.0.set(self.0.get() + 1);
        Err(emission_service_error_v1())
    }
}
impl ExecutionDefinedCallConsumerV29 for UnexpectedService {
    fn prepare(
        &mut self,
        _: &mut SemanticFunctionLoweringV1<'_, '_>,
        _: SemanticBlockIdV1,
        _: &SemanticDirectCallV1,
        _: SemanticFunctionIdV1,
        _: &[SemanticTypeIdV1],
        _: &[HelperCallArgumentV1],
        _: Vec<Type>,
        _: &mut Vec<Operation>,
    ) -> Result<PreparedExecutionDefinedCallV1, ProductionSemanticKirErrorV1> {
        self.refuse()
    }
    fn observe_return(
        &mut self,
        _: &mut SemanticFunctionLoweringV1<'_, '_>,
        _: SemanticBlockIdV1,
        _: &[ValueId],
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.refuse()
    }
}
impl ExecutionLifecycleConsumerV29 for UnexpectedService {
    fn check_instance(
        &self,
        _: &ExecutionAvailabilityV29<'_>,
        _: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.refuse()
    }
    fn require_catalog_entry(
        &self,
        _: SemanticCallableIdV1,
        _: &SemanticCallableDeclV1,
        _: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.refuse()
    }
    fn produce(
        &mut self,
        _: &mut SemanticFunctionLoweringV1<'_, '_>,
        _: SemanticBlockIdV1,
        _: &SemanticDirectCallV1,
        _: SemanticExecutionOperationV29,
        _: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.refuse()
    }
    fn normal_return(
        &mut self,
        _: &mut SemanticFunctionLoweringV1<'_, '_>,
        _: SemanticBlockIdV1,
        _: &[Operation],
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.refuse()
    }
    fn finish(
        &mut self,
        _: &mut SemanticFunctionLoweringV1<'_, '_>,
    ) -> Result<PendingLifecycleEventsV29, ProductionSemanticKirErrorV1> {
        self.refuse()
    }
}

#[test]
fn original_service_roster_rejects_injected_call_and_lifecycle_consumers_before_using_them() {
    for lifecycle in [false, true] {
        let reached = std::cell::Cell::new(false);
        let probe = with_completed_helper(|lowering, _, _| {
            let (detached, services) = lowering.detach_services_v1()?;
            let EmissionServicesV1 {
                mut budget,
                calls,
                lifecycle: old_lifecycle,
                private,
            } = services;
            assert!(calls.is_none() && old_lifecycle.is_none());
            drop((calls, old_lifecycle));
            let before = budget.as_deref().unwrap().emission_service_work_v1();
            let mut unexpected = UnexpectedService(std::cell::Cell::new(0));
            let result = if lifecycle {
                detached.attach(EmissionServicesV1 {
                    budget: budget
                        .as_mut()
                        .map(|row| &mut **row as &mut dyn SemanticEmissionBudgetV1),
                    calls: None,
                    lifecycle: Some(&mut unexpected),
                    private,
                })
            } else {
                detached.attach(EmissionServicesV1 {
                    budget: budget
                        .as_mut()
                        .map(|row| &mut **row as &mut dyn SemanticEmissionBudgetV1),
                    calls: Some(&mut unexpected),
                    lifecycle: None,
                    private,
                })
            };
            assert!(result.is_err());
            drop(result);
            assert_eq!(unexpected.0.get(), 0);
            assert_eq!(
                budget.as_deref().unwrap().emission_service_work_v1(),
                before
            );
            reached.set(true);
            Ok(())
        });
        assert!(probe.result.is_ok(), "{:?}", probe.result);
        assert!(reached.get());
    }
}

#[test]
fn detached_private_work_preserves_original_source_buffers_and_sticky_denial() {
    let reached = std::cell::Cell::new(false);
    let probe = with_completed_helper(|mut lowering, _, place| {
        lowering.private_arrays.work.activate()?;
        lowering
            .private_arrays
            .expected
            .reserve(1, 1, &mut lowering.private_arrays.work)?;
        lowering
            .private_arrays
            .expected
            .rows
            .push(PrivateArrayExpectedPlaceV1 {
                place,
                role: fe2o3_pliron::ProductionSemanticSsaOperandRoleV1::Destination,
                initializer_component: None,
            });
        let pointer = lowering.private_arrays.expected.rows.as_ptr();
        assert!(
            lowering
                .private_arrays
                .work
                .charge_private_array_work(1000)
                .is_err()
        );
        let (mut detached, services) = lowering.detach_services_v1()?;
        assert_eq!(
            detached.lowering.private_arrays.expected.rows.as_ptr(),
            pointer
        );
        assert!(std::ptr::eq(
            detached.lowering.private_arrays.expected.rows[0].place,
            place
        ));
        assert!(
            detached
                .lowering
                .private_arrays
                .work
                .charge_private_array_work(0)
                .is_err()
        );
        let mut attached = detached.attach(services)?;
        assert_eq!(attached.private_arrays.expected.rows.as_ptr(), pointer);
        assert!(
            attached
                .private_arrays
                .work
                .charge_private_array_work(0)
                .is_err()
        );
        reached.set(true);
        Ok(())
    });
    assert!(probe.result.is_ok(), "{:?}", probe.result);
    assert!(reached.get());
}

#[test]
fn service_transition_work_is_inclusive_and_preserves_prior_denial() {
    // No lifecycle or reference-plan query is present in this fixture. Capture
    // prepays8 fixed checks and6 lazy-meter fields; attach prepays12 and6.
    for (capture, short) in [(true, false), (true, true), (false, false), (false, true)] {
        let before = std::cell::Cell::new(0);
        let floor = std::cell::Cell::new(0);
        let expected_storage = std::cell::Cell::new(0);
        let exact = if capture { 8 + 6 } else { 12 + 6 };
        let reached = std::cell::Cell::new(false);
        let probe = with_completed_helper(|mut lowering, _, _| {
            if capture {
                let budget = lowering.emission_work.as_deref_mut().unwrap();
                let spent = budget.emission_service_work_v1().unwrap();
                budget.charge_work(LIMIT - exact + usize::from(short) - spent)?;
                before.set(budget.emission_service_work_v1().unwrap());
                floor.set(budget.storage());
                let result = lowering.detach_services_v1();
                assert_eq!(result.is_ok(), !short);
                expected_storage.set(
                    floor.get()
                        + if short {
                            0
                        } else {
                            emission_service_headers_v1()?
                        },
                );
                drop(result);
            } else {
                let (detached, mut services) = lowering.detach_services_v1()?;
                let budget = services.budget.as_deref_mut().unwrap();
                let spent = budget.emission_service_work_v1().unwrap();
                budget.charge_work(LIMIT - exact + usize::from(short) - spent)?;
                before.set(budget.emission_service_work_v1().unwrap());
                floor.set(budget.storage());
                let result = detached.attach(services);
                assert_eq!(result.is_ok(), !short);
                expected_storage.set(
                    floor.get()
                        - if short {
                            0
                        } else {
                            emission_service_headers_v1()?
                        },
                );
                drop(result);
            }
            reached.set(true);
            Ok(())
        });
        assert!(probe.result.is_ok(), "{:?}", probe.result);
        assert!(reached.get());
        assert_eq!(
            probe.work - before.get(),
            if short { exact - 6 } else { exact }
        );
        assert_eq!(probe.storage, expected_storage.get());
        assert_eq!(probe.failed_storage, Some(FLOOR + LIMIT + 7));
    }
}

#[test]
fn detach_header_storage_is_inclusive_and_never_refunds_a_refused_payload() {
    for short in [false, true] {
        let before = std::cell::Cell::new(0);
        let floor = std::cell::Cell::new(0);
        let reached = std::cell::Cell::new(false);
        let probe = with_completed_helper(|mut lowering, _, _| {
            let header = independent_service_headers();
            let budget = lowering.emission_work.as_deref_mut().unwrap();
            let fill = LIMIT - header + usize::from(short);
            budget.reserve_storage(fill - budget.storage())?;
            before.set(budget.emission_service_work_v1().unwrap());
            floor.set(budget.storage());
            let result = lowering.detach_services_v1();
            assert_eq!(result.is_ok(), !short);
            drop(result);
            reached.set(true);
            Ok(())
        });
        assert!(probe.result.is_ok(), "{:?}", probe.result);
        assert!(reached.get());
        assert_eq!(probe.work - before.get(), 8 + 6);
        assert_eq!(probe.storage, if short { floor.get() } else { LIMIT });
        assert_eq!(probe.failed_storage, Some(FLOOR + LIMIT + 7));
    }
}

#[test]
fn actual_unfinished_cursor_cannot_become_a_detached_boundary() {
    let reached = std::cell::Cell::new(false);
    let floor = std::cell::Cell::new(0);
    let work = std::cell::Cell::new(0);
    let probe = with_helper_boundary(false, |lowering, _, _| {
        assert!(lowering.execution.as_ref().unwrap().block.is_some());
        let budget = lowering.emission_work.as_deref().unwrap();
        floor.set(budget.storage());
        work.set(budget.emission_service_work_v1().unwrap());
        assert!(lowering.detach_services_v1().is_err());
        reached.set(true);
        Ok(())
    });
    assert!(probe.result.is_ok(), "{:?}", probe.result);
    assert!(reached.get());
    assert_eq!(probe.storage, floor.get());
    assert_eq!(probe.work - work.get(), 8);
}

#[test]
fn completed_block_cannot_detach_an_active_memory_payload_capture() {
    for store in [false, true] {
        let reached = std::cell::Cell::new(false);
        let floor = std::cell::Cell::new(0);
        let work = std::cell::Cell::new(0);
        let probe = with_completed_helper(|mut lowering, _, _| {
            assert!(lowering.execution.as_ref().unwrap().block.is_none());
            let ty = lowering.function.locals()[0].ty();
            let site = execution_site_v29(lowering.function.entry(), None);
            let recorder = lowering.scoped_memory.as_mut().unwrap();
            assert!(recorder.frame.is_none());
            assert!(recorder.read_payload.is_none() && recorder.store_payload.is_none());
            if store {
                recorder.store_payload = Some((
                    ValueId(0),
                    ScopedMemoryStoreSourceV29::CallResult { site, ty },
                ));
            } else {
                recorder.read_payload = Some((
                    ScopedMemoryReadV29 {
                        site,
                        role: ExecutionOperandV29::ReturnValue,
                        prefix: 0,
                        ty,
                        occurrence: ScopedMemoryOccurrenceV29::Retained { event: usize::MAX },
                    },
                    true,
                ));
            }
            let budget = lowering.emission_work.as_deref().unwrap();
            floor.set(budget.storage());
            work.set(budget.emission_service_work_v1().unwrap());
            assert!(matches!(
                lowering.detach_services_v1(),
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "emission services differ from their original function frame",
                    ..
                })
            ));
            reached.set(true);
            Ok(())
        });
        assert!(probe.result.is_ok(), "store={store}: {:?}", probe.result);
        assert!(reached.get());
        assert_eq!(probe.storage, floor.get());
        assert_eq!(probe.work - work.get(), 8);
        assert_eq!(probe.failed_storage, Some(FLOOR + LIMIT + 7));
    }
}

#[test]
fn service_rebinding_rejects_a_real_foreign_ledger_before_spending_it() {
    let reached = std::cell::Cell::new(false);
    let retained = std::cell::Cell::new(0);
    let probe = with_completed_helper(|lowering, _, _| {
        let (detached, services) = lowering.detach_services_v1()?;
        let EmissionServicesV1 {
            budget,
            calls,
            lifecycle,
            private,
        } = services;
        assert!(calls.is_none() && lifecycle.is_none());
        retained.set(budget.as_deref().unwrap().storage());
        drop(budget);
        let mut other_work = CanonicalKernelIrWorkBudgetV1::new(LIMIT);
        let mut other = ArgumentBudgetV1::new(&mut other_work, LIMIT);
        other.reserve_storage(retained.get())?;
        let result = detached.attach(EmissionServicesV1 {
            budget: Some(&mut other),
            calls: None,
            lifecycle: None,
            private,
        });
        let rejected = matches!(
            &result,
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "emission services differ from their original function frame",
                ..
            })
        );
        drop(result);
        assert!(rejected);
        assert_eq!(other.storage(), retained.get());
        assert_eq!(other.work(), 0);
        reached.set(true);
        Ok(())
    });
    assert!(probe.result.is_ok(), "{:?}", probe.result);
    assert!(reached.get());
    assert_eq!(probe.storage, retained.get());
}

#[test]
fn service_rebinding_cannot_reset_an_activated_private_work_history() {
    for fault in 0..3 {
        let reached = std::cell::Cell::new(false);
        let probe = with_completed_helper(|mut lowering, _, _| {
            lowering.private_arrays.work.activate()?;
            lowering.private_arrays.work.charge_private_array_work(7)?;
            assert!(
                lowering
                    .private_arrays
                    .work
                    .charge_private_array_work(1000)
                    .is_err()
            );
            let (detached, mut services) = lowering.detach_services_v1()?;
            let PrivateArrayRecorderWorkV1::Shared(private) = &mut services.private else {
                panic!()
            };
            match fault {
                0 => private.active = None,
                1 => private.active.as_mut().unwrap().first_denial = None,
                2 => {
                    private.active.as_mut().unwrap().work = CanonicalKernelIrWorkBudgetV1::new(256)
                }
                _ => unreachable!(),
            }
            assert!(detached.attach(services).is_err());
            reached.set(true);
            Ok(())
        });
        assert!(probe.result.is_ok(), "fault {fault}: {:?}", probe.result);
        assert!(reached.get());
    }
}

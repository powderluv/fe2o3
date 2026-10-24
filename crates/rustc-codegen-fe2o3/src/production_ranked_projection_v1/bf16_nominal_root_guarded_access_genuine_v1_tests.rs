//! Actual retained-input guarded-access DATA checkpoint. Independent source/payload oracle;
//! this remains test routing only, never checked-origin/root-recipe admission.
use super::*;
use crate::production_ranked_projection_v1::{
    bf16_nominal_source_preparation_v1::with_nominal_rich_source_preparation_v1,
    canonical_assertion_facts_v1::with_nominal_canonical_facts_observation_v1,
};
use fe2o3_kernel_analysis::CanonicalKirInventoryV1;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_lower_mir_kernel::{
    Bf16NominalCallQueryErrorV1 as QueryError, CheckedBf16CallInstanceV1,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

type Q<T> = std::result::Result<T, QueryError>;
const HEADERS: usize = 32 * 1024;
const REFUSAL: &str = "actual root guarded access callback refusal";
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Observe,
    Error,
    Panic,
    Occupied,
    MissingInputs,
    DuplicateInputs,
    ChangedBinding,
    EqualInputClone,
    ForeignPendingLedger,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Observation {
    roots: usize,
    references: usize,
    reserved: usize,
    operations: usize,
    next: u32,
    locals: usize,
    assigned: usize,
    processed: usize,
    fifo: usize,
    views: usize,
    accesses: usize,
    predicates: usize,
    assembly_frame: usize,
    access_frame: usize,
}
#[derive(Default)]
struct Oracle {
    cursors: Vec<usize>,
    indices: Vec<Option<ProjectedDisjointIndexV1>>,
    fifo: Vec<usize>,
    input_shadow: Vec<ProductionRankedRootInputV1>,
    views: Vec<Option<ExpectedView>>,
    predicates: Vec<Option<ProductionRankedValueV1>>,
}
fn projection(error: Error) -> QueryError {
    match error {
        Error::CanonicalAssertions(crate::production_ranked_projection_v1::canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(error)) => QueryError::Resource(error),
        _ => QueryError::Unavailable("actual root guarded access source checkpoint refused"),
    }
}
fn expect_operation(
    actual: &[ProductionRankedOperationV1],
    cursor: &mut usize,
    expected: ProductionRankedOperationV1,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> Result<()> {
    resources.work(128)?;
    if actual.get(*cursor) != Some(&expected) {
        return Err(Error::Incomplete(
            "actual root guarded access operation differs from source oracle",
        ));
    }
    *cursor = cursor
        .checked_add(1)
        .ok_or_else(|| resource(Resource::Arithmetic))?;
    Ok(())
}
fn oracle_id(next: &mut u32) -> Result<ProductionRankedValueIdV1> {
    let result = ProductionRankedValueIdV1::new(*next);
    *next = next
        .checked_add(1)
        .ok_or(Error::Incomplete("source oracle value domain exhausted"))?;
    Ok(result)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpectedView {
    result: ProductionRankedValueIdV1,
    width: u32,
    origin: u64,
    noalias: u64,
}
fn operand_local(
    operand: &SemanticOperandV1,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> Result<usize> {
    let place = match operand {
        SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => place,
        SemanticOperandV1::Constant(_) => {
            return Err(Error::Incomplete("access oracle constant operand"));
        }
    };
    resources.work(
        place
            .projections()
            .len()
            .checked_add(16)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
    )?;
    if !place.projections().is_empty() {
        return Err(Error::Incomplete("access oracle projected operand"));
    }
    Ok(place.local().index() as usize)
}
fn expect_view_operation(
    actual: Option<&ProductionRankedOperationV1>,
    expected: ExpectedView,
) -> Result<()> {
    match actual {
        Some(ProductionRankedOperationV1::ViewInSpace {
            result,
            element_width,
            writable,
            shape,
            dynamic_extents,
            memory_space,
            allocation_origin,
            noalias_class,
        }) if *result == expected.result
            && *element_width == expected.width
            && *writable
            && shape.as_slice() == [DYNAMIC_EXTENT]
            && dynamic_extents.as_slice() == [ProductionRankedValueV1::Argument(0)]
            && *memory_space == MemorySpaceAttr::Global
            && *allocation_origin == expected.origin
            && *noalias_class == expected.noalias =>
        {
            Ok(())
        }
        _ => Err(Error::Incomplete("access oracle emitted view differs")),
    }
}
fn expect_cached_view(
    actual: Option<&ProjectedViewV1>,
    expected: Option<ExpectedView>,
) -> Result<()> {
    match (actual, expected) {
        (None, None) => Ok(()),
        (Some(actual), Some(expected))
            if actual.result == expected.result
                && actual.element_width == expected.width
                && actual.writable
                && actual.shape.as_slice() == [DYNAMIC_EXTENT]
                && actual.dynamic_extents.as_slice() == [ProductionRankedValueV1::Argument(0)]
                && actual.memory_space == MemorySpaceAttr::Global
                && actual.allocation_origin == expected.origin
                && actual.noalias_class == expected.noalias =>
        {
            Ok(())
        }
        _ => Err(Error::Incomplete("access oracle cached view differs")),
    }
}
fn expect_predicate(
    actual: Option<&GuardPredicateV1>,
    index: Option<ProductionRankedValueV1>,
) -> Result<()> {
    match (actual, index) {
        (None, None) => Ok(()),
        (Some(actual), Some(index))
            if actual.comparisons.as_slice() == [(index, ProductionRankedValueV1::Argument(0))] =>
        {
            Ok(())
        }
        _ => Err(Error::Incomplete("access oracle predicate differs")),
    }
}
fn expect_access(
    actual: Option<&GuardedRankedAccessV1>,
    view: ProductionRankedValueIdV1,
    index: ProductionRankedValueV1,
    source: SemanticSourceProvenanceV1,
    output_extent: Option<ProductionRankedOutputExtentSourceV1>,
) -> Result<()> {
    match actual {
        Some(actual)
            if actual.failure == GuardedAccessFailureV1::Trap
                && actual.view == view
                && actual.indices.as_slice() == [index]
                && actual.checked_success.is_none()
                && actual.comparisons.as_slice()
                    == [(index, ProductionRankedValueV1::Argument(0))]
                && actual.access == AccessKindAttr::Write
                && actual.memory_space == MemorySpaceAttr::Global
                && actual.source == source
                && actual.semantic_site.is_none()
                && actual.output_extent == output_extent =>
        {
            Ok(())
        }
        _ => Err(Error::Incomplete("access oracle guarded payload differs")),
    }
}

fn inspect_payload(
    actual: &ActualRootGuardedAccessesV1<'_>,
    rich: &RichNominalSourceTablesV1<'_>,
    checked: &CheckedBf16NominalCallV1<'_>,
    context: &mut NominalRecipeResourcesV1<'_, '_, '_, '_, '_, '_>,
    oracle: &mut Oracle,
    input_count: usize,
) -> Result<Observation> {
    let view = &actual.prefix;
    let locals = view.function.locals().len();
    context.with_resources(|resources| {
        resources.work(
            locals
                .checked_mul(8)
                .and_then(|n| n.checked_add(256))
                .ok_or_else(|| resource(Resource::Arithmetic))?,
        )?;
        resources.reserve(&mut oracle.cursors, locals)?;
        oracle.cursors.resize(locals, 0);
        resources.reserve(&mut oracle.indices, locals)?;
        oracle.indices.resize(locals, None);
        resources.reserve(&mut oracle.views, locals)?;
        oracle.views.resize(locals, None);
        resources.reserve(&mut oracle.predicates, locals)?;
        oracle.predicates.resize(locals, None);
        Ok(())
    })?;
    // Exact complete graph row/order and retained store/load/borrow equality are
    // established by the already independent original-source rescan in this SAME
    // loan. Only then may the independent FIFO interpreter below read its edges.
    super::super::initial_graph_v1::check_source_rows_for_prefix_for_test_v1(
        view.graph,
        rich,
        checked,
        context,
        &mut oracle.cursors,
    )?;
    context.with_resources(|resources| {
        resources.work(256)?;
        assert!(std::ptr::eq(view.function, rich.function()));
        assert_eq!(source_launch_input_v1(&view.input.source_launch), view.source_root.source_launch());
        assert_eq!(view.input.kernel_binding, view.source_root.kernel_binding());
        let mut cursor = 0usize;
        let mut next = 0u32;
        let mut reserved = 0usize;
        expect_operation(
            &view.prefix.entry_operations, &mut cursor,
            ranked_execution_layout_v1(view.source_root.layout()), resources,
        )?;
        // Independently inspect ACTUAL borrowed registration outputs; do not call
        // the shared prefix emitter, output-rank/count helper or ledger metadata.
        match view.references {
            [] => assert!(view.prefix.reserved_reference_values.is_none()),
            [binding] => {
                assert!(!binding.observable_output_writes.is_empty());
                for write in &binding.observable_output_writes {
                    resources.work(32)?;
                    let crate::reference_effect_v1::ReferenceOutputCoordinateV1::LogicalPoint(axes)
                        = &write.coordinate else {
                        return Err(Error::Incomplete("source oracle non-point reference output"));
                    };
                    for _ in 0..3 {
                        resources.work(64)?;
                        let id = oracle_id(&mut next)?;
                        assert_eq!(view.prefix.reserved_reference_values.as_ref().unwrap().get(reserved), Some(&id));
                        reserved += 1;
                        expect_operation(&view.prefix.entry_operations, &mut cursor,
                            ProductionRankedOperationV1::SemanticConstant { result: id, value: 0 }, resources)?;
                    }
                    for axis in 0..axes.len() {
                        resources.work(64)?;
                        let id = oracle_id(&mut next)?;
                        assert_eq!(view.prefix.reserved_reference_values.as_ref().unwrap().get(reserved), Some(&id));
                        reserved += 1;
                        expect_operation(&view.prefix.entry_operations, &mut cursor,
                            ProductionRankedOperationV1::SemanticSymbol {
                                result: id, symbol: u32::try_from(axis)
                                    .map_err(|_| Error::Incomplete("source oracle axis overflow"))?,
                            }, resources)?;
                    }
                }
                assert_eq!(view.prefix.reserved_reference_values.as_ref().unwrap().len(), reserved);
            }
            _ => return Err(Error::Incomplete("source oracle duplicate reference registrations")),
        }
        let callables = checked.emission().owner().semantic_ssa().source_semantic().callables();
        // Source-order seeds; no shared seed/assignment/propagation helper.
        for block in view.function.blocks() {
            resources.work(64)?;
            let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else { continue; };
            let Some(SemanticCallableDeclV1::CompilerIntrinsic {
                operation: SemanticCompilerIntrinsicOperationV1::ThreadIndex1d { .. }, ..
            }) = callables.get(call.callee().index() as usize) else { continue; };
            resources.work(128)?;
            let destination = call.destination().ok_or(Error::Incomplete("source oracle seed destination"))?;
            let place = destination.place();
            if !place.projections().is_empty() { return Err(Error::Incomplete("source oracle seed projection")); }
            let local = place.local().index() as usize;
            if rich.scalar_counts().get(local).copied() != Some(1)
                || rich.address_escaped().get(local).copied() != Some(false)
                || oracle.indices.get(local).is_none()
                || oracle.indices[local].is_some()
            { return Err(Error::Incomplete("source oracle seed custody")); }
            let value = oracle_id(&mut next)?;
            expect_operation(&view.prefix.entry_operations, &mut cursor,
                ProductionRankedOperationV1::InvocationIndex { result: value, dimension: 0, launch_extent: 0 },
                resources)?;
            oracle.indices[local] = Some(ProjectedDisjointIndexV1 {
                value: ProductionRankedValueV1::Local(value),
                mapping: SemanticDisjointIndexSpaceV1::Index1d, precondition: None, availability: None,
            });
            resources.push(&mut oracle.fifo, local)?;
        }
        let mut fifo_cursor = 0usize;
        let mut processed = 0usize;
        while let Some(source) = oracle.fifo.get(fifo_cursor).copied() {
            resources.work(128)?;
            fifo_cursor += 1;
            let input = oracle.indices[source].ok_or(Error::Incomplete("source oracle missing queued value"))?;
            assert_eq!(rich.scalar_counts()[source], 1);
            assert!(!rich.address_escaped()[source]);
            for edge in &view.graph.edges()[source] {
                resources.work(256)?;
                processed = processed.checked_add(1).ok_or_else(|| resource(Resource::Arithmetic))?;
                let auth_block = match edge.kind {
                    CapabilityEdgeKindV1::AuthenticatedEnumPayload { construction_block, .. } => construction_block,
                    CapabilityEdgeKindV1::Alias => edge.use_block,
                    _ => return Err(Error::Incomplete("source oracle outside closed index profile")),
                };
                if let Some(availability) = input.availability {
                    assert!(capability_availability_allows(
                        rich.option_dominance(), rich.enum_payload_dominance(), availability,
                        SemanticBlockIdV1::from_index(auth_block as u32),
                    ));
                }
                let projected = match edge.kind {
                    CapabilityEdgeKindV1::Alias => input,
                    CapabilityEdgeKindV1::AuthenticatedEnumPayload { availability, .. } =>
                        ProjectedDisjointIndexV1 {
                            availability: Some(CapabilityAvailabilityV1::EnumPayload(availability)), ..input
                        },
                    _ => unreachable!(),
                };
                let row = oracle.indices.get_mut(edge.destination)
                    .ok_or(Error::Incomplete("source oracle out-of-range destination"))?;
                match row {
                    None => { *row = Some(projected); resources.push(&mut oracle.fifo, edge.destination)?; }
                    Some(existing) if *existing == projected => {}
                    Some(_) => return Err(Error::Incomplete("source oracle conflicting index origins")),
                }
            }
        }

        // Independently rescan source accessors AFTER the complete prefix/index
        // interpreter. No shared normalizer, cache/emission/width/predicate helper.
        let source = checked.emission().owner().semantic_ssa().source_semantic();
        let mut accesses = 0usize;
        let mut views = 0usize;
        for (block_index, block) in view.function.blocks().iter().enumerate() {
            resources.work(64)?;
            let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else { continue; };
            let Some(SemanticCallableDeclV1::CompilerIntrinsic {
                operation: SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut { element, .. }, ..
            }) = source.callables().get(call.callee().index() as usize) else { continue; };
            resources.work(512)?;
            let index_local = operand_local(call.arguments().get(1)
                .ok_or(Error::Incomplete("access oracle missing index"))?, resources)?;
            let index = oracle.indices.get(index_local).copied().flatten()
                .ok_or(Error::Incomplete("access oracle absent index capability"))?;
            if index.mapping != SemanticDisjointIndexSpaceV1::Index1d || index.precondition.is_some() {
                return Err(Error::Incomplete("access oracle outside closed identity profile"));
            }
            if let Some(availability) = index.availability {
                if !capability_availability_allows(rich.option_dominance(), rich.enum_payload_dominance(),
                    availability, SemanticBlockIdV1::from_index(block_index as u32))
                { return Err(Error::Incomplete("access oracle unavailable index")); }
            }
            let receiver = operand_local(call.arguments().first()
                .ok_or(Error::Incomplete("access oracle missing receiver"))?, resources)?;
            let contract = rich.allocations().get(receiver).copied().flatten()
                .ok_or(Error::Incomplete("access oracle absent allocation"))?;
            if !contract.writable { return Err(Error::Incomplete("access oracle read-only allocation")); }
            let bytes = source.types().get(element.index() as usize)
                .and_then(|ty| ty.layout().size_bytes())
                .ok_or(Error::Incomplete("access oracle dynamic element layout"))?;
            let width = u32::try_from(bytes.checked_mul(8)
                .ok_or(Error::Incomplete("access oracle element overflow"))?)
                .map_err(|_| Error::Incomplete("access oracle element overflow"))?;
            if !SUPPORTED_ELEMENT_WIDTHS.contains(&width) {
                return Err(Error::Incomplete("access oracle unsupported element width"));
            }
            let slot = oracle.views.get_mut(contract.allocation_origin as usize)
                .ok_or(Error::Incomplete("access oracle invalid allocation origin"))?;
            let expected = if let Some(expected) = *slot {
                if expected.width != width || expected.origin != contract.allocation_origin
                    || expected.noalias != contract.noalias_class
                { return Err(Error::Incomplete("access oracle cache conflict")); }
                expected
            } else {
                let expected = ExpectedView { result: oracle_id(&mut next)?, width,
                    origin: contract.allocation_origin, noalias: contract.noalias_class };
                expect_view_operation(view.prefix.entry_operations.get(cursor), expected)?;
                cursor = cursor.checked_add(1).ok_or_else(|| resource(Resource::Arithmetic))?;
                *slot = Some(expected);
                views += 1;
                expected
            };
            let output_extent = match rich.allocation_provenance().get(receiver).copied().flatten() {
                Some(LocalAllocationProvenanceV1::Argument(argument)) =>
                    Some(ProductionRankedOutputExtentSourceV1::new(argument,
                        ProductionRankedValueV1::Local(expected.result),
                        ProductionRankedValueV1::Argument(0), index.value)),
                _ => None,
            };
            expect_access(actual.accesses.get(accesses), expected.result, index.value,
                block.terminator().source(), output_extent)?;
            let destination = call.destination().ok_or(Error::Incomplete("access oracle absent destination"))?.place();
            if !destination.projections().is_empty() {
                return Err(Error::Incomplete("access oracle projected destination"));
            }
            let predicate = oracle.predicates.get_mut(destination.local().index() as usize)
                .ok_or(Error::Incomplete("access oracle destination range"))?;
            if predicate.replace(index.value).is_some() {
                return Err(Error::Incomplete("access oracle duplicate predicate"));
            }
            accesses = accesses.checked_add(1).ok_or_else(|| resource(Resource::Arithmetic))?;
        }
        assert_eq!(actual.accesses.len(), accesses, "every actual access row observed");
        assert_eq!(actual.views.len(), locals, "every cache slot observed");
        let mut predicates = 0usize;
        for local in 0..locals {
            resources.work(256)?;
            expect_cached_view(actual.views[local].as_ref(), oracle.views[local])?;
            expect_predicate(view.indices.predicates[local].as_ref(), oracle.predicates[local])?;
            predicates += usize::from(oracle.predicates[local].is_some());
        }

        assert_eq!(cursor, view.prefix.entry_operations.len(), "no unobserved earlier/later emissions");
        assert_eq!(next, view.prefix.next_value, "one prefix-plus-index allocator");
        assert_eq!(view.indices.indices.len(), locals);
        assert_eq!(view.indices.grids.len(), locals);
        assert_eq!(view.indices.predicates.len(), locals);
        let mut assigned = 0usize;
        for (local, expected) in oracle.indices.iter().enumerate() {
            resources.work(128)?;
            assert_eq!(view.indices.indices[local], *expected);
            assert!(view.indices.grids[local].is_none());
            // Every actual predicate was independently checked above.
            assigned += usize::from(expected.is_some());
        }
        resources.work(oracle.fifo.len().checked_mul(32).ok_or_else(|| resource(Resource::Arithmetic))?)?;
        assert_eq!(view.indices.index_fifo, oracle.fifo);
        assert_eq!(view.indices.index_cursor, fifo_cursor);
        assert_eq!(view.indices.processed_edges, processed);
        assert!(view.indices.grid_fifo.is_empty());
        assert_eq!(view.indices.grid_cursor, 0);
        Ok(Observation {
            roots: input_count, references: view.references.len(), reserved,
            operations: cursor, next, locals, assigned, processed, fifo: fifo_cursor,
            views, accesses, predicates, assembly_frame: 0, access_frame: 0,
        })
    })
}
#[allow(clippy::too_many_arguments)]
fn run(
    owner: &ProductionPreRankedKirOwnerV1,
    source: &CheckedBf16CallInstanceV1<'_>,
    inventory: &CanonicalKirInventoryV1<'_>,
    actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
    budget: &mut Budget<'_>,
    mode: Mode,
) -> Q<Observation> {
    let _accepted_run = super::accepted_frames::begin(match mode {
        Mode::Observe => super::accepted_frames::Mode::Observe,
        Mode::Occupied => super::accepted_frames::Mode::Occupied,
        Mode::Error => super::accepted_frames::Mode::Error,
        Mode::Panic => super::accepted_frames::Mode::Panic,
        Mode::MissingInputs => super::accepted_frames::Mode::MissingInputs,
        Mode::DuplicateInputs => super::accepted_frames::Mode::DuplicateInputs,
        Mode::ChangedBinding => super::accepted_frames::Mode::ChangedBinding,
        Mode::EqualInputClone => super::accepted_frames::Mode::EqualInputClone,
        Mode::ForeignPendingLedger => super::accepted_frames::Mode::ForeignPendingLedger,
    });
    let inputs = actual_inputs.inputs();
    let bindings = actual_inputs.bindings();
    assert!(actual_inputs.belongs_to(owner));
    owner.with_bf16_nominal_entry_resources_v1(inventory, budget, |budget| {
        let slot = budget as *const Budget<'_> as usize;
        let identity = budget.work_ledger_identity_v1();
        let before = budget.storage();
        let before_work = budget.work();
        let before_peak = budget.peak_storage();
        budget.charge_work(4 * HEADERS)?;
        budget.reserve_storage(HEADERS)?;
        let mut owned = HEADERS;
        let mut pending = PendingActualRootPrefixIndicesV1::new();
        let mut oracle = Oracle::default();
        let mut entered = false;
        // Both physical variable owners exist before any nested reservation.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            owner.with_checked_bf16_nominal_call_v1(
                inventory, source.root(), source.root(), source.call_block(), source.source_call(),
                budget, |checked, budget| {
                    with_nominal_rich_source_preparation_v1(
                        owner, inventory, source.root(), source.root(), source.call_block(),
                        source.source_call(), budget, |rich, budget| {
                            with_nominal_canonical_facts_observation_v1(
                                owner, inventory, source.root(), source.root(), source.call_block(),
                                source.source_call(), budget, |facts| {
                                    with_nominal_recipe_resources_v1(facts, rich, &mut owned, |context| {
                                        if matches!(mode, Mode::MissingInputs | Mode::DuplicateInputs
                                            | Mode::ChangedBinding | Mode::EqualInputClone)
                                        {
                                            context.with_resources(|resources| {
                                                if mode == Mode::MissingInputs {
                                                    actual_selected_inputs_v1(
                                                        owner, checked, rich.function(), &inputs[..0], bindings, resources,
                                                    )?;
                                                    panic!("missing actual input accepted");
                                                }
                                                // Paid, OUTER-owned shadows test content validation only.
                                                // They cannot construct/replace the private lexical view.
                                                let count = if mode == Mode::DuplicateInputs { 2 } else { inputs.len() };
                                                resources.reserve(&mut oracle.input_shadow, count)?;
                                                for ordinal in 0..count {
                                                    resources.work(128)?;
                                                    let original = &inputs[if mode == Mode::DuplicateInputs { 0 } else { ordinal }];
                                                    let bytes = original.logical_name.len();
                                                    resources.work(bytes.checked_add(32).ok_or_else(|| resource(Resource::Arithmetic))?)?;
                                                    // Install the empty String owner before variable allocation.
                                                    oracle.input_shadow.push(ProductionRankedRootInputV1 {
                                                        logical_name: String::new(),
                                                        kernel_binding: original.kernel_binding,
                                                        source_launch: original.source_launch.clone(),
                                                    });
                                                    resources.reserve_storage(bytes)?;
                                                    let shadow = oracle.input_shadow.last_mut().unwrap();
                                                    shadow.logical_name.try_reserve_exact(bytes).map_err(|_| resource(Resource::Allocation))?;
                                                    if shadow.logical_name.capacity() != bytes { return Err(resource(Resource::Allocation)); }
                                                    shadow.logical_name.push_str(&original.logical_name);
                                                }
                                                if mode == Mode::ChangedBinding { oracle.input_shadow[0].kernel_binding[0] ^= 1; }
                                                assert!(!std::ptr::eq(inputs.as_ptr(), oracle.input_shadow.as_ptr()));
                                                let selected = actual_selected_inputs_v1(
                                                    owner, checked, rich.function(), &oracle.input_shadow, bindings, resources,
                                                )?;
                                                assert!(mode == Mode::EqualInputClone);
                                                assert!(std::ptr::eq(selected.input, &oracle.input_shadow[0]));
                                                Ok(())
                                            })?;
                                            if mode != Mode::EqualInputClone { unreachable!(); }
                                        }
                                        let accepted_stage = super::accepted_frames::enter();
                                        let observed = context.with_actual_root_guarded_accesses_v1(
                                            checked, rich, actual_inputs, &mut pending,
                                            |view, context| {
                                                entered = true;
                                                let observed = inspect_payload(
                                                    &view, rich, checked, context, &mut oracle, inputs.len(),
                                                )?;
                                                match mode {
                                                    Mode::Error => Err(Error::Incomplete(REFUSAL)),
                                                    Mode::Panic => panic!("{REFUSAL}"),
                                                    _ => Ok(observed),
                                                }
                                            },
                                        )?;
                                        drop(accepted_stage);
                                        if mode == Mode::Occupied {
                                            let refused = context.with_actual_root_guarded_accesses_v1(
                                                checked, rich, actual_inputs, &mut pending,
                                                |_, _| -> Result<()> { panic!("occupied assembly entered") },
                                            );
                                            assert!(matches!(refused, Err(Error::Incomplete(
                                                "actual root assembly cannot be replaced or retried"
                                            ))));
                                        }
                                        if mode == Mode::ForeignPendingLedger {
                                            // Inert metadata-corruption control only. The actual
                                            // source graph and prefix NEVER run on this test ledger.
                                            let mut foreign_work = Work::new(0);
                                            let foreign = Budget::new(&mut foreign_work, 0);
                                            let (slot, old_identity) = pending.ledger.unwrap();
                                            assert!(old_identity != foreign.work_ledger_identity_v1());
                                            pending.ledger = Some((slot, foreign.work_ledger_identity_v1()));
                                            let refused = context.with_actual_root_guarded_accesses_v1(
                                                checked, rich, actual_inputs, &mut pending,
                                                |_, _| -> Result<()> { panic!("foreign pending ledger entered") },
                                            );
                                            assert!(matches!(refused, Err(Error::CanonicalAssertions(
                                                crate::production_ranked_projection_v1::canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Accounting)
                                            ))));
                                        }
                                        Ok(observed)
                                    }).map_err(projection)
                                },
                            )
                        },
                    )
                },
            )
        }));
        let mut result = match outcome {
            Ok(result) => result,
            Err(payload) => { drop(payload); Err(QueryError::CallbackPanicked) }
        };
        if matches!(mode, Mode::MissingInputs | Mode::DuplicateInputs | Mode::ChangedBinding) {
            assert!(!entered);
            assert!(pending.prefix.entry_operations.is_empty());
            assert!(pending.indices.indices.is_empty());
            assert!(pending.indices.index_fifo.is_empty());
        } else {
            assert!(entered);
            assert!(pending.completed);
            assert!(pending.guarded.completed());
            assert!(pending.guarded.frame_credits > 0);
            assert!(!pending.prefix.entry_operations.is_empty());
        }
        if let Ok(observed) = &mut result {
            observed.assembly_frame = pending.frame_credits;
            observed.access_frame = pending.guarded.frame_credits;
        }
        // These are post-preparation callback-error/panic controls, not claims
        // of panic injection at every partial-emission program point.
        drop(pending);
        drop(oracle);
        let floor = before.checked_add(owned).ok_or(QueryError::Resource(Resource::Arithmetic))?;
        if slot != budget as *const Budget<'_> as usize
            || identity != budget.work_ledger_identity_v1() || budget.storage() < floor
            || budget.work() < before_work || budget.peak_storage() < before_peak
            || budget.failed_work().is_some() || budget.failed_storage().is_some()
        {
            let _ = result;
            return Err(QueryError::Resource(Resource::Accounting));
        }
        budget.release_storage(owned)?;
        assert_eq!(budget.storage(), before);
        result
    })
}
pub(crate) fn observe_actual_root_guarded_accesses_for_test_v1(
    owner: &ProductionPreRankedKirOwnerV1,
    source: &CheckedBf16CallInstanceV1<'_>,
    inventory: &CanonicalKirInventoryV1<'_>,
    actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
    budget: &mut Budget<'_>,
) -> Q<()> {
    super::accepted_frames::start(super::accepted_frames::Scope::S4);
    let work_before = budget.work();
    let observed = run(
        owner,
        source,
        inventory,
        actual_inputs,
        budget,
        Mode::Observe,
    )?;
    let occupied = run(
        owner,
        source,
        inventory,
        actual_inputs,
        budget,
        Mode::Occupied,
    )?;
    assert_eq!(observed, occupied);
    assert!(matches!(
        run(owner, source, inventory, actual_inputs, budget, Mode::Error),
        Err(QueryError::Unavailable(_))
    ));
    assert!(matches!(
        run(owner, source, inventory, actual_inputs, budget, Mode::Panic),
        Err(QueryError::CallbackPanicked)
    ));
    assert!(matches!(
        run(
            owner,
            source,
            inventory,
            actual_inputs,
            budget,
            Mode::MissingInputs
        ),
        Err(QueryError::Unavailable(_))
    ));
    assert!(matches!(
        run(
            owner,
            source,
            inventory,
            actual_inputs,
            budget,
            Mode::DuplicateInputs
        ),
        Err(QueryError::Unavailable(_))
    ));
    assert!(matches!(
        run(
            owner,
            source,
            inventory,
            actual_inputs,
            budget,
            Mode::ChangedBinding
        ),
        Err(QueryError::Unavailable(_))
    ));
    assert_eq!(
        run(
            owner,
            source,
            inventory,
            actual_inputs,
            budget,
            Mode::EqualInputClone
        )?,
        observed
    );
    assert_eq!(
        run(
            owner,
            source,
            inventory,
            actual_inputs,
            budget,
            Mode::ForeignPendingLedger
        )?,
        observed
    );
    // Direct aggregate ORIGINAL-ledger delta across Observe plus eight controls.
    let work = budget
        .work()
        .checked_sub(work_before)
        .ok_or(QueryError::Resource(Resource::Accounting))?;
    // Closed bounded summary. Exact payload equality was checked above, never
    // inferred from these counts. No logical names, paths, hashes or values print.
    eprintln!(
        "fe2o3-root-guarded-access-v1 roots={} references={} reserved={} operations={} next={} locals={} assigned={} processed={} fifo={} views={} accesses={} predicates={} assembly_frame={} access_frame={} work={}",
        observed.roots,
        observed.references,
        observed.reserved,
        observed.operations,
        observed.next,
        observed.locals,
        observed.assigned,
        observed.processed,
        observed.fifo,
        observed.views,
        observed.accesses,
        observed.predicates,
        observed.assembly_frame,
        observed.access_frame,
        work,
    );
    eprintln!(
        "fe2o3-root-guarded-access-controls-v1 occupied=pass error=pass panic=pass missing_inputs=pass duplicate_inputs=pass changed_binding=pass equal_clone_data_only=pass foreign_pending_ledger=pass"
    );
    super::accepted_frames::flush(
        super::accepted_frames::Scope::S4,
        work,
        observed.assembly_frame,
    );
    origins_s5a::observe_actual_root_reference_origins_for_test_v1(
        owner,
        source,
        inventory,
        actual_inputs,
        budget,
    )?;
    Ok(())
}
#[test]
fn actual_guarded_genuine_headers_cover_outer_owners_and_fixed_control_frames() {
    assert!(
        HEADERS
            >= 4 * size_of::<PendingActualRootPrefixIndicesV1>()
                + 4 * size_of::<Oracle>()
                + 4 * size_of::<Observation>()
                + 4 * size_of::<Budget<'static>>()
                + 4 * size_of::<Work>()
    );
}

#[test]
fn actual_guarded_operation_oracle_rejects_missing_or_changed_rows_without_advancing() {
    let operation = ProductionRankedOperationV1::SemanticConstant {
        result: ProductionRankedValueIdV1::new(91),
        value: 0,
    };
    let changed = ProductionRankedOperationV1::SemanticConstant {
        result: ProductionRankedValueIdV1::new(92),
        value: 0,
    };
    let mut cursor = 0;
    let mut resources = PreparationResourcesV1::unmetered();
    assert!(expect_operation(&[], &mut cursor, operation.clone(), &mut resources).is_err());
    assert_eq!(cursor, 0);
    assert!(expect_operation(&[changed], &mut cursor, operation.clone(), &mut resources).is_err());
    assert_eq!(cursor, 0);
    expect_operation(&[operation.clone()], &mut cursor, operation, &mut resources).unwrap();
    assert_eq!(cursor, 1);
}

fn oracle_fixture() -> (ExpectedView, ProductionRankedValueV1, GuardedRankedAccessV1) {
    let view = ExpectedView {
        result: ProductionRankedValueIdV1::new(17),
        width: 32,
        origin: 3,
        noalias: 4,
    };
    let index = ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(12));
    let extent = Some(ProductionRankedOutputExtentSourceV1::new(
        2,
        ProductionRankedValueV1::Local(view.result),
        ProductionRankedValueV1::Argument(0),
        index,
    ));
    let access = GuardedRankedAccessV1 {
        failure: GuardedAccessFailureV1::Trap,
        view: view.result,
        indices: vec![index],
        checked_success: None,
        comparisons: vec![(index, ProductionRankedValueV1::Argument(0))],
        access: AccessKindAttr::Write,
        memory_space: MemorySpaceAttr::Global,
        source: SemanticSourceProvenanceV1::unavailable(),
        semantic_site: None,
        output_extent: extent,
    };
    (view, index, access)
}
#[test]
fn actual_guarded_oracle_rejects_every_access_field_and_missing_row() {
    let (view, index, original) = oracle_fixture();
    assert!(
        expect_access(
            None,
            view.result,
            index,
            original.source,
            original.output_extent
        )
        .is_err()
    );
    expect_access(
        Some(&original),
        view.result,
        index,
        original.source,
        original.output_extent,
    )
    .unwrap();
    for field in 0..12 {
        let mut changed = original.clone();
        match field {
            11 => changed.failure = GuardedAccessFailureV1::ReturnFallback,
            0 => changed.view = ProductionRankedValueIdV1::new(99),
            1 => changed.indices[0] = ProductionRankedValueV1::Argument(3),
            2 => changed.indices.push(index),
            3 => changed.checked_success = Some(index),
            4 => changed.comparisons[0].1 = ProductionRankedValueV1::Argument(2),
            5 => changed.comparisons.push((index, index)),
            6 => changed.access = AccessKindAttr::Read,
            7 => changed.memory_space = MemorySpaceAttr::Private,
            8 => {
                changed.semantic_site = Some(ProjectedSemanticAccessSiteV1 {
                    block: 1,
                    statement: None,
                })
            }
            9 => changed.output_extent = None,
            10 => {
                use fe2o3_mir_model::semantic_mir_v1::{
                    SemanticSourceFileIdentityV1, SemanticSourceOriginV1,
                };
                let origin = SemanticSourceOriginV1::new(
                    SemanticSourceFileIdentityV1::from_sha256([7; 32]),
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                )
                .unwrap();
                changed.source = SemanticSourceProvenanceV1::new(Some(origin), Some(origin));
            }
            _ => unreachable!(),
        }
        assert!(
            expect_access(
                Some(&changed),
                view.result,
                index,
                original.source,
                original.output_extent
            )
            .is_err(),
            "access field {field}"
        );
    }
}
#[test]
fn actual_guarded_oracle_rejects_view_operation_and_cache_substitutions() {
    let (expected, _, _) = oracle_fixture();
    let original = ProjectedViewV1 {
        result: expected.result,
        element_width: expected.width,
        writable: true,
        shape: vec![DYNAMIC_EXTENT],
        dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
        memory_space: MemorySpaceAttr::Global,
        allocation_origin: expected.origin,
        noalias_class: expected.noalias,
    };
    assert!(expect_cached_view(None, Some(expected)).is_err());
    assert!(expect_cached_view(Some(&original), None).is_err());
    expect_cached_view(Some(&original), Some(expected)).unwrap();
    for field in 0..8 {
        let mut changed = original.clone();
        match field {
            0 => changed.result = ProductionRankedValueIdV1::new(99),
            1 => changed.element_width = 64,
            2 => changed.writable = false,
            3 => changed.shape[0] = 1,
            4 => changed.dynamic_extents[0] = ProductionRankedValueV1::Argument(3),
            5 => changed.memory_space = MemorySpaceAttr::Private,
            6 => changed.allocation_origin ^= 1,
            7 => changed.noalias_class ^= 1,
            _ => unreachable!(),
        }
        assert!(
            expect_cached_view(Some(&changed), Some(expected)).is_err(),
            "cache field {field}"
        );
        let operation = ProductionRankedOperationV1::ViewInSpace {
            result: changed.result,
            element_width: changed.element_width,
            writable: changed.writable,
            shape: changed.shape,
            dynamic_extents: changed.dynamic_extents,
            memory_space: changed.memory_space,
            allocation_origin: changed.allocation_origin,
            noalias_class: changed.noalias_class,
        };
        assert!(
            expect_view_operation(Some(&operation), expected).is_err(),
            "operation field {field}"
        );
    }
    assert!(expect_view_operation(None, expected).is_err());
}
#[test]
fn actual_guarded_oracle_rejects_missing_extra_and_changed_predicate_rows() {
    let (_, index, _) = oracle_fixture();
    let original = GuardPredicateV1 {
        comparisons: vec![(index, ProductionRankedValueV1::Argument(0))],
    };
    expect_predicate(Some(&original), Some(index)).unwrap();
    assert!(expect_predicate(None, Some(index)).is_err());
    assert!(expect_predicate(Some(&original), None).is_err());
    assert!(
        expect_predicate(
            Some(&GuardPredicateV1 {
                comparisons: vec![]
            }),
            Some(index)
        )
        .is_err()
    );
    assert!(
        expect_predicate(
            Some(&GuardPredicateV1 {
                comparisons: vec![(index, index)]
            }),
            Some(index)
        )
        .is_err()
    );
    assert!(
        expect_predicate(
            Some(&GuardPredicateV1 {
                comparisons: vec![
                    (index, ProductionRankedValueV1::Argument(0)),
                    (index, ProductionRankedValueV1::Argument(0))
                ]
            }),
            Some(index)
        )
        .is_err()
    );
}

#[path = "bf16_nominal_root_reference_origins_genuine_v1_tests.rs"]
mod origins_s5a;

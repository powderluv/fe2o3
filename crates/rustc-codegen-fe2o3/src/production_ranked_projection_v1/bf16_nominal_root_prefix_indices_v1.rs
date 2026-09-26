//! Integrated actual retained-input/root-prefix/index preparation. This is one
//! physical root-assembly prefix, not a separate index namespace or ready token.
//! An integrated sibling extends this assembly through actual guarded access DATA.
//! S5A joins source-call/guard/reference-origin DATA in this same owner.
//! Semantic sites/Final/full CFG assembly and normal routing remain pending.
use super::*;
use crate::production_ranked_projection_v1::root_guarded_access_preparation_v1::{
    RootGuardedAccessStorageV1, RootGuardedSourceCallV1, prepare_root_guarded_accesses_v1,
};
use crate::production_ranked_projection_v1::root_reference_origin_preparation_v1::{
    ActualRootReferenceOriginsStorageV1, UnjoinedReferenceOriginPayloadV1,
    prepare_actual_root_reference_origins_v1,
};
use crate::production_ranked_projection_v1::*;
use crate::production_ranked_projection_v1::{
    root_entry_prefix_preparation_v1::{RootEntryPrefixV1, prepare_root_entry_prefix_paid_v1},
    root_invocation_index_preparation_v1::{
        RootInvocationIndexStorageV1, prepare_root_namespace_indices_v1,
    },
};
use crate::reference_effect_v1::{
    AuthenticatedReferenceEffectBindingV1, AuthenticatedReferenceEffectBindingsV1,
};
use fe2o3_lower_mir_kernel::{CheckedBf16NominalCallV1, ProductionPreRankedKirOwnerV1};

/// Must exist physically outside rich/facts/context/graph callbacks. It owns
/// actual entry operations/SSA counter; future access/CFG preparation extends
/// this SAME assembly. No conversion from unjoined component data is provided.
pub(in crate::production_ranked_projection_v1) struct PendingActualRootPrefixIndicesV1 {
    graph: PendingNominalInitialGraphV1,
    prefix: RootEntryPrefixV1,
    indices: RootInvocationIndexStorageV1,
    guarded: RootGuardedAccessStorageV1,
    origins: ActualRootReferenceOriginsStorageV1,
    ledger: Option<(usize, CanonicalKernelIrWorkLedgerIdentityV1)>,
    started: bool,
    completed: bool,
    frame_credits: usize,
}
impl PendingActualRootPrefixIndicesV1 {
    pub(in crate::production_ranked_projection_v1) const fn new() -> Self {
        Self {
            graph: PendingNominalInitialGraphV1::new(),
            prefix: RootEntryPrefixV1::empty(),
            indices: RootInvocationIndexStorageV1::empty(),
            guarded: RootGuardedAccessStorageV1::empty(),
            origins: ActualRootReferenceOriginsStorageV1::empty(),
            ledger: None,
            started: false,
            completed: false,
            frame_credits: 0,
        }
    }
}

/// Read-only lexical access to the actual CO-OWNED data. It establishes neither
/// guarded accesses nor checked origins, ranked recipe, verification or launch.
pub(in crate::production_ranked_projection_v1) struct ActualRootPrefixIndicesV1<'a> {
    graph: &'a NominalCompleteForProfileGraphV1<'a>,
    source_root: ProductionSourceLaunchRootV1,
    function: &'a SemanticFunctionDeclV1,
    input: &'a ProductionRankedRootInputV1,
    references: &'a [AuthenticatedReferenceEffectBindingV1],
    prefix: &'a RootEntryPrefixV1,
    indices: &'a RootInvocationIndexStorageV1,
}
impl ActualRootPrefixIndicesV1<'_> {
    pub(in crate::production_ranked_projection_v1) fn function(&self) -> &SemanticFunctionDeclV1 {
        self.function
    }
    pub(in crate::production_ranked_projection_v1) fn entry_operations(
        &self,
    ) -> &[ProductionRankedOperationV1] {
        &self.prefix.entry_operations
    }
    pub(in crate::production_ranked_projection_v1) fn next_value(&self) -> u32 {
        self.prefix.next_value
    }
    pub(in crate::production_ranked_projection_v1) fn indices(
        &self,
    ) -> &[Option<ProjectedDisjointIndexV1>] {
        &self.indices.indices
    }
}

/// Module-private mutable data seam. Only the original factory can construct it.
/// No caller can replace the graph, actual inputs, namespace or predicate table.
struct ActualRootAssemblyPartsV1<'a> {
    graph: &'a NominalCompleteForProfileGraphV1<'a>,
    source_root: ProductionSourceLaunchRootV1,
    function: &'a SemanticFunctionDeclV1,
    input: &'a ProductionRankedRootInputV1,
    references: &'a [AuthenticatedReferenceEffectBindingV1],
    prefix: &'a mut RootEntryPrefixV1,
    indices: &'a mut RootInvocationIndexStorageV1,
    guarded: &'a mut RootGuardedAccessStorageV1,
    origins: &'a mut ActualRootReferenceOriginsStorageV1,
}
impl ActualRootAssemblyPartsV1<'_> {
    fn prefix_view(&self) -> ActualRootPrefixIndicesV1<'_> {
        ActualRootPrefixIndicesV1 {
            graph: self.graph,
            source_root: self.source_root,
            function: self.function,
            input: self.input,
            references: self.references,
            prefix: self.prefix,
            indices: self.indices,
        }
    }
}
/// Lexically joined actual guarded payload DATA, not checked origins or a recipe.
/// Construction remains in the same actual-input/context/complete-graph loan.
pub(in crate::production_ranked_projection_v1) struct ActualRootGuardedAccessesV1<'a> {
    prefix: ActualRootPrefixIndicesV1<'a>,
    views: &'a [Option<ProjectedViewV1>],
    accesses: &'a [GuardedRankedAccessV1],
    source_calls: &'a [RootGuardedSourceCallV1],
}
impl ActualRootGuardedAccessesV1<'_> {
    pub(in crate::production_ranked_projection_v1) fn prefix(
        &self,
    ) -> &ActualRootPrefixIndicesV1<'_> {
        &self.prefix
    }
    pub(in crate::production_ranked_projection_v1) fn accesses(&self) -> &[GuardedRankedAccessV1] {
        self.accesses
    }
    pub(in crate::production_ranked_projection_v1) fn source_calls(
        &self,
    ) -> &[RootGuardedSourceCallV1] {
        self.source_calls
    }
    pub(in crate::production_ranked_projection_v1) fn views(&self) -> &[Option<ProjectedViewV1>] {
        self.views
    }
}

/// Lexical S5A data: actual source-call associations and FIFO origins share the
/// same pending guard owner. No public CheckedReferencesV1 or ready conversion.
pub(in crate::production_ranked_projection_v1) struct ActualRootReferenceOriginsV1<'a> {
    guarded: ActualRootGuardedAccessesV1<'a>,
    origins: &'a UnjoinedReferenceOriginPayloadV1,
}
impl ActualRootReferenceOriginsV1<'_> {
    pub(in crate::production_ranked_projection_v1) fn guarded(
        &self,
    ) -> &ActualRootGuardedAccessesV1<'_> {
        &self.guarded
    }
    pub(in crate::production_ranked_projection_v1) fn origins(
        &self,
    ) -> &[Option<CheckedReferenceOriginV1>] {
        &self.origins.origins
    }
}

/// Borrowed local-contract DATA inside the actual S5A source/owner loan.
/// This is not a completed block stream, emitted access or ready capability.
/// The ledger pair is compared only while its original lexical loan is live.
pub(in crate::production_ranked_projection_v1) struct ActualRootLocalContractsV1<'a> {
    origins: ActualRootReferenceOriginsV1<'a>,
    local:
        crate::production_ranked_projection_v1::root_local_contracts_v1::BorrowedLocalContractsV1<
            'a,
        >,
    ledger: (usize, CanonicalKernelIrWorkLedgerIdentityV1),
}
impl ActualRootLocalContractsV1<'_> {
    pub(in crate::production_ranked_projection_v1) fn origins(
        &self,
    ) -> &ActualRootReferenceOriginsV1<'_> {
        &self.origins
    }
    /// Raw coordinates produce only DATA; they do not authenticate a source
    /// occurrence, assign a semantic site or manufacture an operation cursor.
    pub(in crate::production_ranked_projection_v1) fn query(
        &self,
        place: &SemanticPlaceV1,
        block: usize,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> Result<
        crate::production_ranked_projection_v1::root_local_contracts_v1::LocalContractDecisionV1,
    > {
        self.local.query_on_ledger(
            self.origins.guarded.prefix.function,
            place,
            block,
            self.ledger,
            resources,
        )
    }
}

/// One source-derived checked-reference decision inside the actual S5A loan.
/// This has no operation cursor or block-stream completion/ready authority.
pub(in crate::production_ranked_projection_v1) struct ActualRootReferenceUseV1<'a> {
    origins: ActualRootReferenceOriginsV1<'a>,
    occurrence: crate::production_ranked_projection_v1::root_checked_reference_use_preparation_v1::SourceUseOccurrenceV1<'a>,
    origin: Option<CheckedReferenceSourceV1>,
}
impl ActualRootReferenceUseV1<'_> {
    pub(in crate::production_ranked_projection_v1) fn origins(
        &self,
    ) -> &ActualRootReferenceOriginsV1<'_> {
        &self.origins
    }
    pub(in crate::production_ranked_projection_v1) fn place(&self) -> &SemanticPlaceV1 {
        self.occurrence.place
    }
    pub(in crate::production_ranked_projection_v1) fn site(&self) -> ProjectedSemanticAccessSiteV1 {
        self.occurrence.site
    }
    pub(in crate::production_ranked_projection_v1) fn ordinal(&self) -> usize {
        self.occurrence.ordinal
    }
    pub(in crate::production_ranked_projection_v1) fn access(&self) -> AccessKindAttr {
        self.occurrence.access
    }
    pub(in crate::production_ranked_projection_v1) fn origin(
        &self,
    ) -> Option<CheckedReferenceSourceV1> {
        self.origin
    }
}

/// Separate lexical query frame, prepaid before constructing the wrapper's
/// captured consumer. No runtime owner field or detached pending use container.
fn actual_use_frame<R, F>() -> Result<usize> {
    let mut frame = 4096usize;
    for amount in [
        size_of::<ActualRootReferenceUseV1<'static>>(),
        size_of::<ProjectedSemanticAccessSiteV1>(),
        size_of::<usize>(),
        size_of::<F>()
            .checked_mul(2)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
        size_of::<Result<R>>()
            .checked_mul(2)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
    ] {
        frame = frame
            .checked_add(amount)
            .ok_or_else(|| resource(Resource::Arithmetic))?;
    }
    Ok(frame)
}

struct ActualSelectedInputsV1<'a> {
    input: &'a ProductionRankedRootInputV1,
    source_root: ProductionSourceLaunchRootV1,
    references: &'a [AuthenticatedReferenceEffectBindingV1],
}
fn names_equal_v1(
    left: &str,
    right: &str,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> Result<bool> {
    let work = left
        .len()
        .checked_add(right.len())
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| resource(Resource::Arithmetic))?;
    resources.work(work)?;
    Ok(left == right)
}

/// Content validation helper only. It accepts raw borrows for inert negative
/// controls, but cannot issue a lexical input view or enter assembly. The joined
/// factory obtains these borrows only from the pipeline-owned lexical view.
/// Complete roster selection runs before prefix emission. No cloned bindings,
/// inferred empty default, caller ranks/counts, or caller namespace are accepted.
fn actual_selected_inputs_v1<'a>(
    owner: &ProductionPreRankedKirOwnerV1,
    checked: &CheckedBf16NominalCallV1<'_>,
    function: &SemanticFunctionDeclV1,
    inputs: &'a [ProductionRankedRootInputV1],
    bindings: &'a AuthenticatedReferenceEffectBindingsV1,
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> Result<ActualSelectedInputsV1<'a>> {
    resources.work(128)?;
    let source = owner.semantic_ssa().source_semantic();
    let roster = owner.source_launch();
    // The already-joined S1 profile has exactly two source functions. Use a
    // bounded stack selection table, not an unmetered partition/clone builder.
    if source.functions().len() != 2
        || inputs.is_empty()
        || inputs.len() > 2
        || inputs.len() != roster.roots().len()
        || roster.semantic_sha256() != source.semantic_sha256().as_bytes()
    {
        return Err(Error::Incomplete(
            "actual root prefix input/source roster differs",
        ));
    }
    for (index, input) in inputs.iter().enumerate() {
        resources.work(16)?;
        for earlier in &inputs[..index] {
            if names_equal_v1(&input.logical_name, &earlier.logical_name, resources)? {
                return Err(Error::Unsupported(
                    "duplicate typed logical roots in the ranked roster",
                ));
            }
        }
    }
    let mut bound = [false; 2];
    let mut selected_reference = None;
    let mut selected = None;
    for (index, (input, source_root)) in inputs.iter().zip(roster.roots()).enumerate() {
        resources.work(128)?;
        let actual = source
            .functions()
            .get(source_root.selected_root().index() as usize)
            .ok_or(Error::Incomplete("actual root prefix roster source absent"))?;
        let binding = *actual
            .kernel_entry()
            .ok_or(Error::Unsupported(
                "a semantic KernelRoot without an authenticated kernel entry",
            ))?
            .kernel_binding_identity()
            .as_bytes();
        if actual.role() != SemanticFunctionRoleV1::KernelRoot
            || source_root.semantic_root_identity() != actual.identity()
            || source_root.kernel_binding() != binding
            || input.kernel_binding != binding
            || source_root.source_launch() != source_launch_input_v1(&input.source_launch)
        {
            return Err(Error::Unsupported(
                "source launch roster root changed before ranked projection",
            ));
        }
        if source_root.selected_root() == checked.emission().root() {
            if selected.is_some() || !std::ptr::eq(actual, function) {
                return Err(Error::Incomplete(
                    "actual root prefix selected source differs",
                ));
            }
            selected = Some((index, *source_root));
        }
    }
    let (root_index, source_root) = selected.ok_or(Error::Incomplete(
        "actual root prefix checked root is absent from retained inputs",
    ))?;
    for (binding_index, binding) in bindings.as_slice().iter().enumerate() {
        resources.work(32)?;
        let mut matched = None;
        for (index, input) in inputs.iter().enumerate() {
            if names_equal_v1(&binding.logical_kernel_name, &input.logical_name, resources)? {
                matched = Some(index);
                break;
            }
        }
        let index = matched.ok_or(Error::Unsupported(
            "a reference-effect binding outside the exact typed root roster",
        ))?;
        if bound[index] {
            return Err(Error::Unsupported(
                "duplicate reference-effect bindings for one ranked root",
            ));
        }
        bound[index] = true;
        if index == root_index {
            selected_reference = Some(binding_index);
        }
    }
    let references = match selected_reference {
        Some(index) => &bindings.as_slice()[index..index + 1],
        // Borrowed empty subslice is justified by the COMPLETE actual binding
        // scan, not a manufactured AuthenticatedReferenceEffectBindings default.
        None => &bindings.as_slice()[..0],
    };
    Ok(ActualSelectedInputsV1 {
        input: &inputs[root_index],
        source_root,
        references,
    })
}

/// S1 currently excludes these producers. Recheck their actual discriminants in
/// the SAME source loan so a later whitelist edit cannot silently invalidate the
/// no-emission prefix fact. Non-Call transfer paths return default read_view.
fn require_excluded_prefix_producers_v1(
    function: &SemanticFunctionDeclV1,
    callables: &[SemanticCallableDeclV1],
    resources: &mut PreparationResourcesV1<'_, '_>,
) -> Result<()> {
    for block in function.blocks() {
        resources.work(32)?;
        let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
            continue;
        };
        resources.work(32)?;
        let callable = callables
            .get(call.callee().index() as usize)
            .ok_or(Error::Incomplete("actual prefix source callable absent"))?;
        if matches!(
            callable,
            SemanticCallableDeclV1::CompilerIntrinsic {
                operation: SemanticCompilerIntrinsicOperationV1::StridedReadView2DFromSharedSlice { .. }
                    | SemanticCompilerIntrinsicOperationV1::StridedReadView2DLoadOr { .. }
                    | SemanticCompilerIntrinsicOperationV1::DisjointBlockComponentIndex { .. },
                ..
            }
        ) {
            return Err(Error::Incomplete(
                "actual root prefix has an unconnected emitting producer",
            ));
        }
    }
    Ok(())
}

fn assembly_frame<R, F>() -> Result<usize> {
    let mut bytes = 8192usize;
    for amount in [
        size_of::<PendingActualRootPrefixIndicesV1>(),
        size_of::<ActualRootPrefixIndicesV1<'static>>(),
        size_of::<ActualSelectedInputsV1<'static>>(),
        size_of::<ActualRootAssemblyPartsV1<'static>>(),
        size_of::<ActualRootGuardedAccessesV1<'static>>(),
        size_of::<ActualRootReferenceOriginsV1<'static>>(),
        // Public wrapper, private continuation and their result transfers.
        size_of::<F>()
            .checked_mul(2)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
        size_of::<Result<R>>()
            .checked_mul(2)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
        size_of::<F>()
            .checked_mul(2)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
        size_of::<Result<R>>()
            .checked_mul(2)
            .ok_or_else(|| resource(Resource::Arithmetic))?,
    ] {
        bytes = bytes
            .checked_add(amount)
            .ok_or_else(|| resource(Resource::Arithmetic))?;
    }
    Ok(bytes)
}

impl NominalRecipeResourcesV1<'_, '_, '_, '_, '_, '_> {
    /// Preserve the existing prefix-only checkpoint and its immutable oracle view.
    pub(in crate::production_ranked_projection_v1) fn with_actual_root_prefix_indices_v1<R, F>(
        &mut self,
        checked: &CheckedBf16NominalCallV1<'_>,
        rich: &RichNominalSourceTablesV1<'_>,
        actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
        pending: &mut PendingActualRootPrefixIndicesV1,
        inspect: F,
    ) -> Result<R>
    where
        F: for<'a> FnOnce(ActualRootPrefixIndicesV1<'a>, &mut Self) -> Result<R>,
    {
        self.with_actual_root_assembly_v1(
            checked,
            rich,
            actual_inputs,
            pending,
            |parts, context| inspect(parts.prefix_view(), context),
        )
    }

    /// Same retained assembly extended through actual identity access appends.
    /// No normal production route constructs the actual input loan at this stage.
    pub(in crate::production_ranked_projection_v1) fn with_actual_root_guarded_accesses_v1<R, F>(
        &mut self,
        checked: &CheckedBf16NominalCallV1<'_>,
        rich: &RichNominalSourceTablesV1<'_>,
        actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
        pending: &mut PendingActualRootPrefixIndicesV1,
        inspect: F,
    ) -> Result<R>
    where
        F: for<'a> FnOnce(ActualRootGuardedAccessesV1<'a>, &mut Self) -> Result<R>,
    {
        self.with_actual_root_assembly_v1(
            checked,
            rich,
            actual_inputs,
            pending,
            |parts, context| {
                let owner = context.facts.owner;
                let source = owner.semantic_ssa().source_semantic();
                context.with_resources(|resources| {
                    prepare_root_guarded_accesses_v1(
                        source.types(),
                        source.callables(),
                        parts.function,
                        &parts.indices.indices,
                        rich.option_dominance(),
                        rich.enum_payload_dominance(),
                        rich.allocations(),
                        rich.allocation_provenance(),
                        &mut parts.indices.predicates,
                        parts.guarded,
                        &mut parts.prefix.entry_operations,
                        &mut parts.prefix.next_value,
                        resources,
                    )
                })?;
                if !parts.guarded.completed() {
                    return Err(Error::Incomplete(
                        "actual guarded access payload unfinished",
                    ));
                }
                inspect(
                    ActualRootGuardedAccessesV1 {
                        prefix: parts.prefix_view(),
                        views: &parts.guarded.views,
                        accesses: &parts.guarded.accesses,
                        source_calls: &parts.guarded.source_calls,
                    },
                    context,
                )
            },
        )
    }

    /// S5A extends the same physical namespace/guard owner, under the exact S1
    /// profile. Sites and later memory-use projection remain deliberately absent.
    pub(in crate::production_ranked_projection_v1) fn with_actual_root_reference_origins_v1<R, F>(
        &mut self,
        checked: &CheckedBf16NominalCallV1<'_>,
        rich: &RichNominalSourceTablesV1<'_>,
        actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
        pending: &mut PendingActualRootPrefixIndicesV1,
        inspect: F,
    ) -> Result<R>
    where
        F: for<'a> FnOnce(ActualRootReferenceOriginsV1<'a>, &mut Self) -> Result<R>,
    {
        self.with_actual_root_assembly_v1(
            checked,
            rich,
            actual_inputs,
            pending,
            |parts, context| {
                let owner = context.facts.owner;
                let source = owner.semantic_ssa().source_semantic();
                context.with_resources(|resources| {
                    prepare_root_guarded_accesses_v1(
                        source.types(),
                        source.callables(),
                        parts.function,
                        &parts.indices.indices,
                        rich.option_dominance(),
                        rich.enum_payload_dominance(),
                        rich.allocations(),
                        rich.allocation_provenance(),
                        &mut parts.indices.predicates,
                        parts.guarded,
                        &mut parts.prefix.entry_operations,
                        &mut parts.prefix.next_value,
                        resources,
                    )?;
                    prepare_actual_root_reference_origins_v1(
                        parts.function,
                        source.callables(),
                        parts.guarded,
                        parts.graph.edges(),
                        rich.option_dominance(),
                        rich.enum_payload_dominance(),
                        parts.origins,
                        resources,
                    )
                })?;
                let origins = parts.origins.payload().ok_or(Error::Incomplete(
                    "actual reference origins payload unfinished",
                ))?;
                inspect(
                    ActualRootReferenceOriginsV1 {
                        guarded: ActualRootGuardedAccessesV1 {
                            prefix: parts.prefix_view(),
                            views: &parts.guarded.views,
                            accesses: &parts.guarded.accesses,
                            source_calls: &parts.guarded.source_calls,
                        },
                        origins,
                    },
                    context,
                )
            },
        )
    }

    /// Extend the original same-owner origin loan with borrowed rich local DATA.
    /// No new source visit, ledger, owner, table clone or completed-stream flag.
    pub(in crate::production_ranked_projection_v1) fn with_actual_root_local_contracts_v1<R, F>(
        &mut self,
        checked: &CheckedBf16NominalCallV1<'_>,
        rich: &RichNominalSourceTablesV1<'_>,
        actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
        pending: &mut PendingActualRootPrefixIndicesV1,
        inspect: F,
    ) -> Result<R>
    where
        F: for<'a> FnOnce(ActualRootLocalContractsV1<'a>, &mut Self) -> Result<R>,
    {
        use crate::production_ranked_projection_v1::root_local_contracts_v1::{
            BorrowedLocalContractsV1, SourceLocalContractPartsV1, local_contract_frame_v1,
            require_local_contract_ledger_v1,
        };
        let expected = (self.state.slot, self.state.ledger);
        let owner = self.facts.owner;
        let function = self.function;
        self.with_resources(|resources| {
            require_local_contract_ledger_v1(expected, resources)?;
            resources.work(64)?;
            if !std::ptr::eq(owner, checked.emission().owner())
                || !actual_inputs.belongs_to(owner)
                || !rich.belongs_to_original_ledger_v1(expected)
            {
                return Err(resource(Resource::Accounting));
            }
            require_same_source_v1(function, rich.function())?;
            let frame =
                local_contract_frame_v1::<R, F>(size_of::<ActualRootLocalContractsV1<'static>>())?;
            resources.work(frame)?;
            resources.reserve_storage(frame)
        })?;
        // The captured consumer is constructed only after its frame was paid.
        self.with_actual_root_reference_origins_v1(
            checked,
            rich,
            actual_inputs,
            pending,
            |origins, context| {
                // Copy references to owned rows, never borrow the movable loan
                // wrapper itself while moving that wrapper into the result.
                let payload = origins.origins;
                let function = origins.guarded.prefix.function;
                let local = context.with_resources(|resources| {
                    require_local_contract_ledger_v1(expected, resources)?;
                    resources.work(64)?;
                    if !rich.belongs_to_original_ledger_v1(expected) {
                        return Err(resource(Resource::Accounting));
                    }
                    BorrowedLocalContractsV1::source_data(
                        SourceLocalContractPartsV1 {
                            function,
                            table_function: rich.function(),
                            counts: rich.scalar_counts(),
                            assignments: rich.scalar_assignments(),
                            address_escaped: rich.address_escaped(),
                            allocations: rich.allocations(),
                            allocation_provenance: rich.allocation_provenance(),
                            origins: &payload.origins,
                            option_dominance: rich.option_dominance(),
                            enum_payload_dominance: rich.enum_payload_dominance(),
                        },
                        resources,
                    )
                })?;
                inspect(
                    ActualRootLocalContractsV1 {
                        origins,
                        local,
                        ledger: expected,
                    },
                    context,
                )
            },
        )
    }

    /// Read-only source-site query under the existing actual S5A factory.
    /// The selector takes coordinates only; it obtains the place/kind/provenance
    /// from the retained source. No projected operation vector is fabricated.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::production_ranked_projection_v1) fn with_actual_root_reference_use_v1<R, F>(
        &mut self,
        checked: &CheckedBf16NominalCallV1<'_>,
        rich: &RichNominalSourceTablesV1<'_>,
        actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
        pending: &mut PendingActualRootPrefixIndicesV1,
        site: ProjectedSemanticAccessSiteV1,
        ordinal: usize,
        inspect: F,
    ) -> Result<R>
    where
        F: for<'a> FnOnce(ActualRootReferenceUseV1<'a>, &mut Self) -> Result<R>,
    {
        self.with_resources(|resources| {
            if resources.has_denial() || resources.original_ledger_v1().is_none() {
                return Err(resource(Resource::Accounting));
            }
            let frame = actual_use_frame::<R, F>()?;
            resources.work(frame)?;
            resources.reserve_storage(frame)
        })?;
        self.with_actual_root_reference_origins_v1(
            checked, rich, actual_inputs, pending,
            |origins, context| {
                let function = origins.guarded.prefix.function;
                let (occurrence, origin) = context.with_resources(|resources| {
                    let occurrence = crate::production_ranked_projection_v1::root_checked_reference_use_preparation_v1::select_source_use_v1(
                        function, site, ordinal, resources,
                    )?;
                    let origin = crate::production_ranked_projection_v1::root_checked_references_v1::origin_for_actual_borrowed_v1(
                        function, rich.function(), occurrence.place, site.block, origins.origins(),
                        rich.option_dominance(), rich.enum_payload_dominance(), resources,
                    )?;
                    resources.work(32)?;
                    if occurrence.access.is_atomic() != occurrence.atomic.is_some() {
                        return Err(Error::Unsupported(
                            "an atomic access whose ordering/scope contract is missing or attached to a non-atomic access",
                        ));
                    }
                    match origin {
                        Some(CheckedReferenceSourceV1::GuardedAccess(index)) => {
                            if occurrence.atomic.is_some() {
                                return Err(Error::Incomplete(
                                    "an atomic access through a checked disjoint reference before exact atomic capability projection",
                                ));
                            }
                            if origins.guarded.accesses.get(index).is_none() {
                                return Err(Error::Unsupported(
                                    "a checked disjoint reference whose access origin is out of range",
                                ));
                            }
                        }
                        Some(CheckedReferenceSourceV1::ProjectedSharedBorrow)
                            if occurrence.atomic.is_none() && occurrence.access == AccessKindAttr::Read => {}
                        Some(CheckedReferenceSourceV1::ProjectedSharedBorrow) =>
                            return Err(Error::Unsupported(
                                "a projected shared reference used for a non-read memory effect",
                            )),
                        None => {} // Explicitly NOT a checked access; ordinary fallback remains pending.
                    }
                    Ok((occurrence, origin))
                })?;
                inspect(ActualRootReferenceUseV1 { origins, occurrence, origin }, context)
            },
        )
    }

    /// The non-Clone lexical input view is constructed only by the owning
    /// pipeline after materialization. Equal raw slices cannot replace it.
    /// No normal production route constructs this view at this checkpoint.
    /// No root/access ready token is constructed: one source visit populates
    /// the actual physical namespace, lends it, then performs original postflights.
    #[allow(clippy::too_many_arguments)]
    fn with_actual_root_assembly_v1<R, F>(
        &mut self,
        checked: &CheckedBf16NominalCallV1<'_>,
        rich: &RichNominalSourceTablesV1<'_>,
        actual_inputs: &crate::production_pipeline::ActualRetainedRankedInputsV1<'_>,
        pending: &mut PendingActualRootPrefixIndicesV1,
        inspect: F,
    ) -> Result<R>
    where
        F: for<'a> FnOnce(ActualRootAssemblyPartsV1<'a>, &mut Self) -> Result<R>,
    {
        let facts_owner = self.facts.owner;
        self.with_resources(|resources| {
            resources.work(64)?;
            if !actual_inputs.belongs_to(facts_owner)
                || !actual_inputs.belongs_to(checked.emission().owner())
            {
                return Err(Error::Incomplete(
                    "actual input loan belongs to another retained owner",
                ));
            }
            let ledger = resources
                .original_ledger_v1()
                .ok_or_else(|| resource(Resource::Accounting))?;
            if pending.ledger.is_some_and(|saved| saved != ledger) {
                return Err(resource(Resource::Accounting));
            }
            if pending.started || pending.ledger.is_some() || pending.completed {
                return Err(Error::Incomplete(
                    "actual root assembly cannot be replaced or retried",
                ));
            }
            let frame = assembly_frame::<R, F>()?;
            resources.work(frame)?;
            resources.reserve_storage(frame)?;
            #[cfg(test)]
            accepted_frames::record(accepted_frames::Kind::Assembly, frame);
            pending.frame_credits = frame; // Actual accepted original-ledger debit.
            pending.ledger = Some(ledger);
            pending.started = true;
            Ok(())
        })?;
        let PendingActualRootPrefixIndicesV1 {
            graph,
            prefix,
            indices,
            guarded,
            origins,
            completed,
            ..
        } = pending;
        self.with_complete_for_profile_graph_v1(checked, rich, graph, |graph, context| {
            let owner = context.facts.owner;
            let source = owner.semantic_ssa().source_semantic();
            let function = graph.function();
            require_same_source_v1(function, rich.function())?;
            let selected = context.with_resources(|resources| {
                actual_selected_inputs_v1(
                    owner,
                    checked,
                    function,
                    actual_inputs.inputs(),
                    actual_inputs.bindings(),
                    resources,
                )
            })?;
            context.with_resources(|resources| {
                prepare_root_entry_prefix_paid_v1(
                    selected.source_root,
                    selected.references,
                    prefix,
                    resources,
                )?;
                // Preserve the ordinary whole-callable retired-family refusal;
                // S1 intentionally is not a substitute for this distinct check.
                let retired_work = source
                    .callables()
                    .len()
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(64))
                    .ok_or_else(|| resource(Resource::Arithmetic))?;
                resources.work(retired_work)?;
                reject_retired_production_intrinsics_v1(source.callables())?;
                require_excluded_prefix_producers_v1(function, source.callables(), resources)?;
                prepare_root_namespace_indices_v1(
                    source.callables(),
                    function,
                    rich.scalar_counts(),
                    rich.address_escaped(),
                    rich.option_dominance(),
                    rich.enum_payload_dominance(),
                    graph.edges(),
                    graph.edge_count(),
                    indices,
                    &mut prefix.entry_operations,
                    &mut prefix.next_value,
                    resources,
                )
            })?;
            *completed = true; // Prefix/index DATA only; no actual access rows yet.
            inspect(
                ActualRootAssemblyPartsV1 {
                    graph: &graph,
                    source_root: selected.source_root,
                    function,
                    input: selected.input,
                    references: selected.references,
                    prefix,
                    indices,
                    guarded,
                    origins,
                },
                context,
            )
        })
    }
}

#[cfg(test)]
#[path = "bf16_root_accepted_frame_v1_tests.rs"]
pub(in crate::production_ranked_projection_v1) mod accepted_frames;

#[cfg(test)]
#[path = "bf16_nominal_root_prefix_indices_v1_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bf16_nominal_root_prefix_indices_genuine_v1_tests.rs"]
mod genuine;
#[cfg(test)]
pub(crate) use genuine::observe_actual_root_prefix_indices_for_test_v1;

#[cfg(test)]
#[path = "bf16_nominal_root_guarded_access_genuine_v1_tests.rs"]
mod guarded_genuine_v1;
#[cfg(test)]
pub(crate) use guarded_genuine_v1::observe_actual_root_guarded_accesses_for_test_v1;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct SemanticEnumSsaFactsV1 {
    variants: BTreeMap<SsaValueV1, u32>,
    discriminants: BTreeMap<SsaValueV1, SsaValueV1>,
}

impl SemanticEnumSsaFactsV1 {
    fn meet(&self, incoming: &Self) -> Self {
        Self {
            variants: self
                .variants
                .iter()
                .filter_map(|(value, variant)| {
                    (incoming.variants.get(value) == Some(variant)).then_some((*value, *variant))
                })
                .collect(),
            discriminants: self
                .discriminants
                .iter()
                .filter_map(|(value, source)| {
                    (incoming.discriminants.get(value) == Some(source)).then_some((*value, *source))
                })
                .collect(),
        }
    }

    fn retained_entries(&self) -> usize {
        self.variants.len().saturating_add(self.discriminants.len())
    }

    fn renamed_for_edge(
        &self,
        arguments: &[SsaArgumentV1],
        target: u32,
        budget: &mut SemanticEnumAnalysisBudgetV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let argument_count = arguments.len();
        let fact_count = self.retained_entries();
        budget.charge_work(
            1_usize
                .saturating_add(argument_count.saturating_mul(2))
                .saturating_add(fact_count),
        )?;
        // Two argument indexes, carried facts, and at most two copied facts per argument.
        // These are logical entries, matching this analyzer's existing storage domain.
        let temporary_entries = fact_count.saturating_add(argument_count.saturating_mul(4));
        budget.reserve_temporary_entries(temporary_entries)?;
        let result = (|| {
            let mut destinations = BTreeSet::new();
            let mut renames = BTreeMap::new();
            for argument in arguments {
                let destination = SsaValueV1::BlockArgument {
                    block: SsaBlockIdV1::new(target),
                    variable: argument.variable(),
                };
                if !destinations.insert(destination) {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                renames
                    .entry(argument.value())
                    .and_modify(|rename| *rename = None)
                    .or_insert(Some(destination));
            }
            let rename_source = |source: &SsaValueV1| match renames.get(source) {
                Some(destination) => *destination,
                None if destinations.contains(source) => None,
                None => Some(*source),
            };
            let carry =
                |value: &SsaValueV1| !destinations.contains(value) && !renames.contains_key(value);
            let mut next = Self::default();
            for (value, variant) in &self.variants {
                if carry(value) {
                    next.variants.insert(*value, *variant);
                }
            }
            for (value, source) in &self.discriminants {
                if carry(value)
                    && let Some(source) = rename_source(source)
                {
                    next.discriminants.insert(*value, source);
                }
            }
            // Read every source from the original snapshot: phi assignment is simultaneous.
            // Duplicate sources may fan out variants; ambiguous discriminator aliases refuse.
            for argument in arguments {
                let destination = SsaValueV1::BlockArgument {
                    block: SsaBlockIdV1::new(target),
                    variable: argument.variable(),
                };
                if let Some(variant) = self.variants.get(&argument.value()) {
                    next.variants.insert(destination, *variant);
                }
                if let Some(source) = self
                    .discriminants
                    .get(&argument.value())
                    .and_then(rename_source)
                {
                    next.discriminants.insert(destination, source);
                }
            }
            Ok(next)
        })();
        let retained_entries = result.as_ref().map_or(0, Self::retained_entries);
        // The indexes have dropped; the caller keeps the output reservation until merge/drop.
        budget.replace_storage(temporary_entries, retained_entries)?;
        result
    }
}

#[derive(Clone, Copy, Debug)]
struct SemanticEnumAnalysisBudgetV1 {
    work_limit: usize,
    storage_limit: usize,
    work: usize,
    storage: usize,
}

impl SemanticEnumAnalysisBudgetV1 {
    const fn new(work_limit: usize, storage_limit: usize) -> Self {
        Self {
            work_limit,
            storage_limit,
            work: 0,
            storage: 0,
        }
    }

    fn charge_work(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        self.work =
            self.work
                .checked_add(amount)
                .ok_or(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::AnalysisWork,
                    actual: usize::MAX,
                    limit: self.work_limit,
                })?;
        enforce_limit(
            ProductionSemanticKirResourceV1::AnalysisWork,
            self.work,
            self.work_limit,
        )
    }

    fn replace_storage(
        &mut self,
        previous: usize,
        next: usize,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.storage = self.storage.checked_sub(previous).ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "promoted enum analysis storage accounting underflow",
            )
        })?;
        self.storage =
            self.storage
                .checked_add(next)
                .ok_or(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::AnalysisStorage,
                    actual: usize::MAX,
                    limit: self.storage_limit,
                })?;
        enforce_limit(
            ProductionSemanticKirResourceV1::AnalysisStorage,
            self.storage,
            self.storage_limit,
        )
    }

    fn charge_storage(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        self.storage = self.storage.checked_add(amount).ok_or(
            ProductionSemanticKirErrorV1::ResourceLimit {
                resource: ProductionSemanticKirResourceV1::AnalysisStorage,
                actual: usize::MAX,
                limit: self.storage_limit,
            },
        )?;
        enforce_limit(
            ProductionSemanticKirResourceV1::AnalysisStorage,
            self.storage,
            self.storage_limit,
        )
    }

    fn reserve_temporary_entries(
        &mut self,
        amount: usize,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let previous = self.storage;
        if let Err(error) = self.charge_storage(amount) {
            // A denied reservation has not allocated any entries; retain its error diagnostic.
            self.storage = previous;
            return Err(error);
        }
        Ok(())
    }
}

fn analyze_promoted_enum_variants_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    control_flow_ssa: &SemanticControlFlowSsaPlanV1,
    max_analysis_work: usize,
    max_analysis_storage: usize,
) -> Result<BTreeMap<(u32, SsaValueV1), u32>, ProductionSemanticKirErrorV1> {
    fn whole_local(operand: &SemanticOperandV1) -> Option<u32> {
        match operand {
            SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)
                if place.projections().is_empty() =>
            {
                Some(place.local().index())
            }
            SemanticOperandV1::Copy(_)
            | SemanticOperandV1::Move(_)
            | SemanticOperandV1::Constant(_) => None,
        }
    }

    fn record_block_entry_value_v1(
        entry_values_by_block: &mut [BTreeMap<u32, SsaValueV1>],
        block: u32,
        local: u32,
        value: SsaValueV1,
        budget: &mut SemanticEnumAnalysisBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let values = entry_values_by_block
            .get_mut(block as usize)
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        if let Some(previous) = values.get(&local) {
            if *previous != value {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        } else {
            budget.charge_storage(1)?;
            values.insert(local, value);
        }
        Ok(())
    }

    fn record_enum_value_v1(
        enum_local_by_value: &mut BTreeMap<SsaValueV1, u32>,
        value: SsaValueV1,
        local: u32,
        budget: &mut SemanticEnumAnalysisBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if let Some(previous_local) = enum_local_by_value.get(&value) {
            if *previous_local != local {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        } else {
            budget.charge_storage(1)?;
            enum_local_by_value.insert(value, local);
        }
        Ok(())
    }

    let mut budget = SemanticEnumAnalysisBudgetV1::new(max_analysis_work, max_analysis_storage);
    let mut promoted_enums = BTreeSet::new();
    for (local, promoted) in &control_flow_ssa.promoted {
        budget.charge_work(1)?;
        if promoted.transport.uses_structural_enum_transport()
            && types
                .get(promoted.semantic_type.index() as usize)
                .is_some_and(|declaration| {
                    matches!(declaration.shape(), SemanticTypeShapeV1::Enum { .. })
                })
        {
            budget.charge_storage(1)?;
            promoted_enums.insert(*local);
        }
    }
    if promoted_enums.is_empty() {
        return Ok(BTreeMap::new());
    }

    let block_count = function.blocks().len();
    let entry = function.entry().index() as usize;
    let mut incoming = vec![None::<SemanticEnumSsaFactsV1>; block_count];
    let mut queued = BTreeSet::from([function.entry().index()]);
    let mut worklist = VecDeque::from([function.entry().index()]);
    budget.charge_storage(block_count.saturating_mul(4))?;
    let mut entry_values_by_block = vec![BTreeMap::<u32, SsaValueV1>::new(); block_count];
    let mut enum_local_by_value = BTreeMap::<SsaValueV1, u32>::new();
    for (local, value) in &control_flow_ssa.entry_definitions {
        budget.charge_work(1)?;
        if control_flow_ssa.ssa_value_locals.contains(local) {
            let entry_live_in = control_flow_ssa.live_in(function.entry().index());
            budget.charge_work(entry_live_in.len())?;
            let header_value = if entry_live_in.contains(local) {
                SsaValueV1::BlockArgument {
                    block: SsaBlockIdV1::new(function.entry().index()),
                    variable: fe2o3_mir_model::SsaVariableIdV1::new(*local),
                }
            } else {
                *value
            };
            record_block_entry_value_v1(
                &mut entry_values_by_block,
                function.entry().index(),
                *local,
                header_value,
                &mut budget,
            )?;
        }
        if promoted_enums.contains(local) {
            record_enum_value_v1(&mut enum_local_by_value, *value, *local, &mut budget)?;
        }
    }
    for ((_, local), definitions) in &control_flow_ssa.definition_values {
        budget.charge_work(1)?;
        for value in definitions {
            budget.charge_work(1)?;
            if promoted_enums.contains(local) {
                record_enum_value_v1(&mut enum_local_by_value, *value, *local, &mut budget)?;
            }
        }
    }
    for definitions in control_flow_ssa.edge_definitions.values() {
        budget.charge_work(1)?;
        for definition in definitions {
            budget.charge_work(1)?;
            let local = definition.variable().get();
            if promoted_enums.contains(&local) {
                record_enum_value_v1(
                    &mut enum_local_by_value,
                    definition.value(),
                    local,
                    &mut budget,
                )?;
            }
        }
    }
    for (block, locals) in &control_flow_ssa.live_in {
        budget.charge_work(1)?;
        for local in locals {
            budget.charge_work(1)?;
            let value = SsaValueV1::BlockArgument {
                block: SsaBlockIdV1::new(*block),
                variable: fe2o3_mir_model::SsaVariableIdV1::new(*local),
            };
            record_block_entry_value_v1(
                &mut entry_values_by_block,
                *block,
                *local,
                value,
                &mut budget,
            )?;
            if promoted_enums.contains(local) {
                record_enum_value_v1(&mut enum_local_by_value, value, *local, &mut budget)?;
            }
        }
    }
    for ((block, local), value) in &control_flow_ssa.block_entry_values {
        budget.charge_work(1)?;
        record_block_entry_value_v1(
            &mut entry_values_by_block,
            *block,
            *local,
            *value,
            &mut budget,
        )?;
    }
    let maximum_block_entries = entry_values_by_block
        .iter()
        .map(BTreeMap::len)
        .max()
        .unwrap_or(0);
    budget.charge_work(block_count)?;
    budget.charge_storage(maximum_block_entries)?;
    incoming[entry] = Some(SemanticEnumSsaFactsV1::default());

    while let Some(block_index) = worklist.pop_front() {
        budget.charge_work(1)?;
        queued.remove(&block_index);
        let Some(block) = function.blocks().get(block_index as usize) else {
            return Err(unsupported(
                0,
                Some(block_index),
                None,
                "promoted enum analysis references a missing block",
            ));
        };
        let Some(mut facts) = incoming[block_index as usize].clone() else {
            continue;
        };
        let indexed_entry_values = entry_values_by_block
            .get(block_index as usize)
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        budget.charge_work(indexed_entry_values.len())?;
        let mut current_values = indexed_entry_values.clone();
        let mut definition_offsets = BTreeMap::<u32, usize>::new();
        for statement in block.statements() {
            budget.charge_work(1)?;
            let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                continue;
            };
            let destination = assignment.destination();
            if !destination.projections().is_empty() {
                continue;
            }
            let destination_local = destination.local().index();
            if !control_flow_ssa
                .ssa_value_locals
                .contains(&destination_local)
            {
                continue;
            }
            let definition_offset = definition_offsets.entry(destination_local).or_default();
            let destination_value = control_flow_ssa
                .definition_values
                .get(&(block_index, destination_local))
                .and_then(|definitions| definitions.get(*definition_offset))
                .copied()
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            *definition_offset = definition_offset.checked_add(1).ok_or_else(|| {
                unsupported(
                    0,
                    Some(block_index),
                    None,
                    "promoted enum definition ordinal overflow",
                )
            })?;

            if promoted_enums.contains(&destination_local) {
                let exact_variant = match assignment.value().kind() {
                    SemanticRvalueKindV1::Aggregate(aggregate) => match aggregate.kind() {
                        SemanticAggregateKindV1::EnumVariant(variant) => Some(*variant),
                        _ => None,
                    },
                    SemanticRvalueKindV1::Use(operand) => whole_local(operand)
                        .and_then(|source| current_values.get(&source))
                        .and_then(|source| facts.variants.get(source))
                        .copied(),
                    _ => None,
                };
                if let Some(variant) = exact_variant {
                    facts.variants.insert(destination_value, variant);
                }
            }
            if let SemanticRvalueKindV1::Discriminant(place) = assignment.value().kind()
                && place.projections().is_empty()
                && promoted_enums.contains(&place.local().index())
                && let Some(source) = current_values.get(&place.local().index()).copied()
            {
                facts.discriminants.insert(destination_value, source);
            } else if let SemanticRvalueKindV1::Use(operand) = assignment.value().kind()
                && let Some(source) = whole_local(operand)
                    .and_then(|source| current_values.get(&source))
                    .and_then(|source| facts.discriminants.get(source))
                    .copied()
            {
                facts.discriminants.insert(destination_value, source);
            }
            current_values.insert(destination_local, destination_value);
        }

        let mut refined_enum = None;
        if let SemanticTerminatorKindV1::SwitchInt {
            discriminant,
            targets,
        } = block.terminator().kind()
            && let Some(enum_value) = whole_local(discriminant)
                .and_then(|local| current_values.get(&local))
                .and_then(|value| facts.discriminants.get(value))
                .copied()
        {
            let enum_local = enum_local_by_value
                .get(&enum_value)
                .copied()
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block_index),
                        None,
                        "promoted enum discriminant provenance has no current enum local",
                    )
                })?;
            let promoted = &control_flow_ssa.promoted[&enum_local];
            let declaration = types
                .get(promoted.semantic_type.index() as usize)
                .ok_or_else(|| {
                    unsupported(
                        0,
                        Some(block_index),
                        None,
                        "promoted enum switch type is missing",
                    )
                })?;
            let SemanticTypeShapeV1::Enum { variants, .. } = declaration.shape() else {
                return Err(unsupported(
                    0,
                    Some(block_index),
                    None,
                    "promoted enum switch source is not an enum",
                ));
            };
            refined_enum = Some(enum_value);
            for (ordinal, target) in targets.values().iter().enumerate() {
                let mut edge_facts = facts.clone();
                if let Some((variant, _)) = variants
                    .iter()
                    .enumerate()
                    .find(|(_, variant)| variant.discriminant() == target.value())
                {
                    edge_facts.variants.insert(enum_value, variant as u32);
                } else {
                    edge_facts.variants.remove(&enum_value);
                }
                propagate_promoted_enum_facts_v1(
                    block_index,
                    ordinal as u32,
                    target.edge().target().index(),
                    edge_facts,
                    control_flow_ssa,
                    &mut incoming,
                    &mut queued,
                    &mut worklist,
                    &mut budget,
                )?;
            }
            let mut otherwise = variants.iter().enumerate().filter(|(_, variant)| {
                !variant.is_uninhabited()
                    && !targets
                        .values()
                        .iter()
                        .any(|target| target.value() == variant.discriminant())
            });
            let exact_otherwise = otherwise.next().map(|(variant, _)| variant as u32);
            if otherwise.next().is_some() {
                facts.variants.remove(&enum_value);
            } else if let Some(variant) = exact_otherwise {
                facts.variants.insert(enum_value, variant);
            } else {
                facts.variants.remove(&enum_value);
            }
            propagate_promoted_enum_facts_v1(
                block_index,
                targets.values().len() as u32,
                targets.otherwise().target().index(),
                facts.clone(),
                control_flow_ssa,
                &mut incoming,
                &mut queued,
                &mut worklist,
                &mut budget,
            )?;
        }
        if refined_enum.is_none() {
            let mut edge_ordinal = 0_u32;
            block
                .terminator()
                .kind()
                .try_for_each_edge::<ProductionSemanticKirErrorV1>(|edge| {
                    propagate_promoted_enum_facts_v1(
                        block_index,
                        edge_ordinal,
                        edge.target().index(),
                        facts.clone(),
                        control_flow_ssa,
                        &mut incoming,
                        &mut queued,
                        &mut worklist,
                        &mut budget,
                    )?;
                    edge_ordinal = edge_ordinal.checked_add(1).ok_or_else(|| {
                        unsupported(
                            0,
                            Some(block_index),
                            None,
                            "promoted enum edge ordinal overflow",
                        )
                    })?;
                    Ok(())
                })?;
        }
    }

    Ok(incoming
        .into_iter()
        .enumerate()
        .flat_map(|(block, facts)| {
            facts.into_iter().flat_map(move |facts| {
                facts
                    .variants
                    .into_iter()
                    .map(move |(value, variant)| ((block as u32, value), variant))
            })
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn propagate_promoted_enum_facts_v1(
    source: u32,
    edge_ordinal: u32,
    target: u32,
    facts: SemanticEnumSsaFactsV1,
    control_flow_ssa: &SemanticControlFlowSsaPlanV1,
    incoming: &mut [Option<SemanticEnumSsaFactsV1>],
    queued: &mut BTreeSet<u32>,
    worklist: &mut VecDeque<u32>,
    budget: &mut SemanticEnumAnalysisBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let arguments = control_flow_ssa
        .edge_arguments
        .get(&(source, edge_ordinal))
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let edge_facts = facts.renamed_for_edge(arguments, target, budget)?;
    let edge_storage = edge_facts.retained_entries();
    let result = (|| {
        let Some(target_incoming) = incoming.get_mut(target as usize) else {
            return Err(unsupported(
                0,
                Some(source),
                None,
                "promoted enum analysis references a missing successor",
            ));
        };
        let previous_storage = target_incoming
            .as_ref()
            .map_or(0, SemanticEnumSsaFactsV1::retained_entries);
        let next_bound = target_incoming
            .as_ref()
            .map_or(edge_storage, SemanticEnumSsaFactsV1::retained_entries);
        budget.charge_work(
            1_usize
                .saturating_add(edge_storage)
                .saturating_add(previous_storage.saturating_mul(2)),
        )?;
        budget.reserve_temporary_entries(next_bound)?;
        let next = target_incoming
            .as_ref()
            .map_or_else(|| edge_facts.clone(), |current| current.meet(&edge_facts));
        if target_incoming.as_ref() != Some(&next) {
            let retained_entries = next.retained_entries();
            let previous = target_incoming.replace(next);
            drop(previous);
            budget.replace_storage(
                previous_storage.saturating_add(next_bound),
                retained_entries,
            )?;
            if queued.insert(target) {
                worklist.push_back(target);
            }
        } else {
            drop(next);
            budget.replace_storage(next_bound, 0)?;
        }
        Ok(())
    })();
    drop(edge_facts);
    budget.replace_storage(edge_storage, 0)?;
    result
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadComponentStorageV1 {
    pointer: ValueId,
    storage: Option<CompilerEnumPointerStorageV57>,
    kernel_type: Type,
    alignment: u32,
}

#[derive(Clone, Debug)]
struct SemanticRetainedLocalSlotV1 {
    pointer: ValueId,
    semantic_type: SemanticTypeIdV1,
    storage: SemanticRetainedStorageV29,
}

#[derive(Clone, Copy, Debug, Default)]
struct SemanticBlockPrologueSpansV1 {
    retained_local_storage: u32,
    enum_payload_storage: u32,
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadFieldStorageV1 {
    semantic_type: SemanticTypeIdV1,
    exact_enum_variant: Option<u32>,
    compiler_issued_binding: Option<SemanticPromotedBindingV1>,
    components: Box<[SemanticEnumPayloadComponentStorageV1]>,
}

#[derive(Clone, Debug)]
enum SemanticEnumPayloadSourceV1 {
    AggregateOperand(SemanticPlaceV1),
    CallResult,
}

#[derive(Clone, Debug)]
struct SemanticEnumPayloadCustodyV1 {
    source_block: SemanticBlockIdV1,
    binding: SemanticValueBindingV1,
}

#[derive(Clone, Debug)]
enum SemanticEnumPayloadRestoreV1 {
    PrivateStorage(SemanticEnumPayloadFieldStorageV1),
    UniqueSource(SemanticValueBindingV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticCapabilityAvailabilityV1 {
    Option(SemanticOptionAvailabilityV1),
    EnumPayload {
        local: SemanticLocalIdV1,
        variant: u32,
    },
}

#[derive(Clone, Debug)]
enum SemanticValueBindingV1 {
    Unit,
    Unmaterialized,
    Execution(SemanticExecutionBindingV29),
    ExecutionBorrow(SemanticExecutionBorrowBindingV29),
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Checked borrowed-role consumers remain gated on source custody"
        )
    )]
    ExecutionReferent(SemanticExecutionBorrowBindingV29),
    MovedExecution,
    SourceReference(SemanticSourceReferenceBindingV29),
    SourceInactive(SemanticSourceInactiveBindingV29),
    SourceEnumTag(SemanticSourceEnumTagBindingV55),
    Aggregate(Vec<SemanticValueBindingV1>),
    Enum {
        discriminant: ValueId,
        discriminant_ty: Type,
        semantic_type: SemanticTypeIdV1,
        variant: Option<u32>,
        payloads: BTreeMap<u32, Vec<SemanticValueBindingV1>>,
    },
    MathContext,
    CollectiveContext,
    WorkgroupLdsScope,
    DynamicLds {
        base: ValueId,
        base_ty: Type,
        len: ValueId,
        byte_len: ValueId,
        dynamic_lds: SemanticTypeIdV1,
        element_storage: SemanticTypeIdV1,
        elements: u32,
        byte_extent: u64,
        alignment: u32,
        producer_function: SemanticFunctionIdV1,
        producer_block: SemanticBlockIdV1,
    },
    MatrixContext,
    WaveLane {
        value: ValueId,
        wave: SemanticCurrentWaveV1,
    },
    MatrixFragment {
        values: Vec<(ValueId, Type)>,
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
        wave: SemanticCurrentWaveV1,
    },
    AccumulatorFragment {
        values: Vec<(ValueId, Type)>,
        contract: SemanticMfmaAccumulatorContractV1,
        wave: SemanticCurrentWaveV1,
    },
    Gfx950LdsTransposeTile {
        storage: ValueId,
        format: SemanticGfx950LdsTransposeFormatV1,
        state: SemanticGfx950LdsTransposeStateV1,
    },
    WorkgroupPipeline {
        storage: ValueId,
        pipeline: SemanticTypeIdV1,
        element: SemanticTypeIdV1,
        payload_binding: SemanticPromotedBindingV1,
        component_types: Box<[Type]>,
        packed_type: Type,
        buffers: u32,
        elements: u64,
        prefetch_distance: u32,
        alignment: u32,
    },
    Value {
        id: ValueId,
        ty: Type,
    },
    OptionPointer {
        present: ValueId,
        pointer: ValueId,
        pointer_ty: Type,
        availability: SemanticOptionAvailabilityV1,
    },
    IndexWitness {
        id: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        disjoint: bool,
        availability: Option<SemanticCapabilityAvailabilityV1>,
    },
    OptionIndexWitness {
        present: ValueId,
        id: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        disjoint: bool,
        availability: SemanticOptionAvailabilityV1,
    },
    GridLeader {
        availability: SemanticCapabilityAvailabilityV1,
    },
    ComponentWitness {
        raw: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticCapabilityAvailabilityV1,
    },
    OptionComponentWitness {
        present: ValueId,
        raw: ValueId,
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticOptionAvailabilityV1,
    },
    OptionGridLeader {
        present: ValueId,
        availability: SemanticOptionAvailabilityV1,
    },
}

fn project_enum_payload_field(
    selected_variant: u32,
    mut payloads: BTreeMap<u32, Vec<SemanticValueBindingV1>>,
    field: u32,
) -> Result<SemanticValueBindingV1, &'static str> {
    if payloads
        .values()
        .flatten()
        .any(semantic_binding_contains_execution_v29)
    {
        return Err("execution bindings cannot be restored from enum payloads");
    }
    let Some(mut fields) = payloads.remove(&selected_variant) else {
        return Ok(SemanticValueBindingV1::Unmaterialized);
    };
    if field as usize >= fields.len() {
        return Err("enum payload field is unavailable in this block");
    }
    Ok(fields.swap_remove(field as usize))
}

fn semantic_binding_kind_v1(binding: &SemanticValueBindingV1) -> &'static str {
    match binding {
        SemanticValueBindingV1::Unit => "unit",
        SemanticValueBindingV1::Unmaterialized => "unmaterialized enum payload",
        SemanticValueBindingV1::Execution(_) => "nominal execution role",
        SemanticValueBindingV1::ExecutionBorrow(_) => "nominal execution borrow",
        SemanticValueBindingV1::ExecutionReferent(_) => "borrowed execution referent",
        SemanticValueBindingV1::MovedExecution => "moved execution value",
        SemanticValueBindingV1::SourceReference(_) => "checked source reference",
        SemanticValueBindingV1::SourceInactive(_) => "inactive source transport",
        SemanticValueBindingV1::SourceEnumTag(_) => "checked source enum tag transport",
        SemanticValueBindingV1::Aggregate(_) => "aggregate",
        SemanticValueBindingV1::Enum {
            variant: Some(_), ..
        } => "variant-refined enum",
        SemanticValueBindingV1::Enum { variant: None, .. } => "unrefined enum",
        SemanticValueBindingV1::MathContext => "math context",
        SemanticValueBindingV1::CollectiveContext => "collective context",
        SemanticValueBindingV1::WorkgroupLdsScope => "workgroup LDS scope",
        SemanticValueBindingV1::DynamicLds { .. } => "compiler-issued dynamic LDS",
        SemanticValueBindingV1::MatrixContext => "matrix context",
        SemanticValueBindingV1::WaveLane { .. } => "wave lane",
        SemanticValueBindingV1::MatrixFragment { .. } => "matrix fragment",
        SemanticValueBindingV1::AccumulatorFragment { .. } => "accumulator fragment",
        SemanticValueBindingV1::Gfx950LdsTransposeTile { .. } => "gfx950 LDS transpose tile",
        SemanticValueBindingV1::WorkgroupPipeline { .. } => "workgroup pipeline",
        SemanticValueBindingV1::Value { .. } => "ordinary value",
        SemanticValueBindingV1::OptionPointer { .. } => "optional pointer",
        SemanticValueBindingV1::IndexWitness { .. } => "index witness",
        SemanticValueBindingV1::OptionIndexWitness { .. } => "optional index witness",
        SemanticValueBindingV1::GridLeader { .. } => "grid leader",
        SemanticValueBindingV1::ComponentWitness { .. } => "component witness",
        SemanticValueBindingV1::OptionComponentWitness { .. } => "optional component witness",
        SemanticValueBindingV1::OptionGridLeader { .. } => "optional grid leader",
    }
}

fn semantic_binding_can_restore_from_unique_source_v1(binding: &SemanticValueBindingV1) -> bool {
    match binding {
        SemanticValueBindingV1::Unit
        | SemanticValueBindingV1::MathContext
        | SemanticValueBindingV1::CollectiveContext
        | SemanticValueBindingV1::MatrixContext
        | SemanticValueBindingV1::WaveLane { .. }
        | SemanticValueBindingV1::MatrixFragment { .. }
        | SemanticValueBindingV1::AccumulatorFragment { .. }
        | SemanticValueBindingV1::Gfx950LdsTransposeTile { .. }
        | SemanticValueBindingV1::WorkgroupPipeline { .. }
        | SemanticValueBindingV1::IndexWitness { .. }
        | SemanticValueBindingV1::GridLeader { .. }
        | SemanticValueBindingV1::ComponentWitness { .. } => true,
        SemanticValueBindingV1::Aggregate(fields) => fields
            .iter()
            .all(semantic_binding_can_restore_from_unique_source_v1),
        SemanticValueBindingV1::Value { ty, .. } => !matches!(ty, Type::Execution(_)),
        SemanticValueBindingV1::Execution(_)
        | SemanticValueBindingV1::ExecutionBorrow(_)
        | SemanticValueBindingV1::ExecutionReferent(_)
        | SemanticValueBindingV1::MovedExecution
        | SemanticValueBindingV1::SourceReference(_)
        | SemanticValueBindingV1::SourceInactive(_)
        | SemanticValueBindingV1::SourceEnumTag(_)
        | SemanticValueBindingV1::Unmaterialized
        | SemanticValueBindingV1::Enum { .. }
        | SemanticValueBindingV1::OptionPointer { .. }
        | SemanticValueBindingV1::OptionIndexWitness { .. }
        | SemanticValueBindingV1::OptionComponentWitness { .. }
        | SemanticValueBindingV1::OptionGridLeader { .. }
        | SemanticValueBindingV1::WorkgroupLdsScope
        | SemanticValueBindingV1::DynamicLds { .. } => false,
    }
}

include!("semantic_ssa_enum_reauthentication_v1.rs");

impl SemanticValueBindingV1 {
    fn value(&self) -> Result<(ValueId, Type), &'static str> {
        match self {
            Self::Value {
                ty: Type::Execution(_),
                ..
            } => Err("execution role requires a nominal producer binding"),
            Self::Value { id, ty } => Ok((*id, ty.clone())),
            Self::IndexWitness { id, .. } => Ok((*id, Type::INDEX)),
            Self::WaveLane { value, .. } => Ok((*value, Type::Scalar(ScalarType::U32))),
            Self::Unmaterialized => {
                Err("unmaterialized enum payload has no ordinary SSA representation")
            }
            Self::MovedExecution => Err("moved execution value cannot be observed"),
            Self::SourceInactive(_) => Err("inactive source transport cannot be observed"),
            Self::SourceEnumTag(_) => Err("source enum tag requires its checked original owner"),
            Self::SourceReference(_) => {
                Err("source reference requires its checked owner and projection")
            }
            Self::Execution(_) | Self::ExecutionBorrow(_) | Self::ExecutionReferent(_) => {
                Err("execution binding has no ordinary scalar representation")
            }
            Self::Unit
            | Self::Aggregate(_)
            | Self::Enum { .. }
            | Self::MathContext
            | Self::CollectiveContext
            | Self::WorkgroupLdsScope
            | Self::DynamicLds { .. }
            | Self::MatrixContext
            | Self::MatrixFragment { .. }
            | Self::AccumulatorFragment { .. }
            | Self::Gfx950LdsTransposeTile { .. }
            | Self::WorkgroupPipeline { .. }
            | Self::OptionPointer { .. }
            | Self::OptionIndexWitness { .. }
            | Self::ComponentWitness { .. }
            | Self::OptionComponentWitness { .. }
            | Self::GridLeader { .. }
            | Self::OptionGridLeader { .. } => {
                Err("aggregate or capability value requires a semantic projection")
            }
        }
    }

    fn values(&self) -> Result<Vec<(ValueId, Type)>, &'static str> {
        let mut values = Vec::new();
        self.append_values(&mut values)?;
        Ok(values)
    }

    fn append_values(&self, values: &mut Vec<(ValueId, Type)>) -> Result<(), &'static str> {
        self.visit_values_v1(&mut LegacyTransportCollectorV1(values))
    }

    fn visit_values_v1<V: SemanticTransportVisitorV1>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        use BorrowedTransportTypeV1::{Existing, Pointer};
        visitor.node()?;
        match self {
            Self::Value {
                ty: Type::Execution(_),
                ..
            } => {
                return Err(V::invalid(
                    "execution role requires a nominal producer binding",
                ));
            }
            Self::Value { id, ty } => visitor.component(*id, Existing(ty))?,
            Self::IndexWitness { id, .. } => visitor.component(*id, Existing(&Type::INDEX))?,
            Self::WaveLane { value, .. } => {
                visitor.component(*value, Existing(&Type::Scalar(ScalarType::U32)))?;
            }
            Self::Aggregate(fields) => {
                for field in fields {
                    field.visit_values_v1(visitor)?;
                }
            }
            Self::MatrixFragment {
                values: components, ..
            }
            | Self::AccumulatorFragment {
                values: components, ..
            } => {
                for (value, ty) in components {
                    visitor.component(*value, Existing(ty))?;
                }
            }
            Self::Gfx950LdsTransposeTile { storage, .. } => {
                visitor.component(
                    *storage,
                    Pointer {
                        pointee: &Type::Scalar(ScalarType::U8),
                        address_space: AddressSpace::Workgroup,
                        access: AccessMode::ReadWrite,
                    },
                )?;
            }
            Self::Enum {
                discriminant,
                discriminant_ty,
                payloads,
                ..
            } => {
                for fields in payloads.values() {
                    visitor.node()?;
                    for field in fields {
                        if semantic_transport_contains_execution_v1(field, visitor)? {
                            return Err(V::invalid(
                                "execution bindings cannot be flattened through enum payloads",
                            ));
                        }
                    }
                }
                visitor.component(*discriminant, Existing(discriminant_ty))?;
            }
            Self::Unit => {}
            Self::Unmaterialized => {
                return Err(V::invalid(
                    "unmaterialized enum payload has no ordinary SSA representation",
                ));
            }
            Self::MovedExecution => {
                return Err(V::invalid("moved execution value cannot be observed"));
            }
            Self::SourceInactive(_) => {
                return Err(V::invalid("inactive source transport cannot be observed"));
            }
            Self::SourceEnumTag(_) => {
                return Err(V::invalid("source enum tag requires checked CFG transport"));
            }
            Self::SourceReference(_) => {
                return Err(V::invalid(
                    "source reference requires checked call or CFG transport",
                ));
            }
            Self::Execution(_) | Self::ExecutionBorrow(_) | Self::ExecutionReferent(_) => {
                return Err(V::invalid(
                    "execution binding has no ordinary SSA representation",
                ));
            }
            Self::MathContext
            | Self::CollectiveContext
            | Self::WorkgroupLdsScope
            | Self::DynamicLds { .. }
            | Self::MatrixContext
            | Self::WorkgroupPipeline { .. }
            | Self::OptionPointer { .. }
            | Self::OptionIndexWitness { .. }
            | Self::ComponentWitness { .. }
            | Self::OptionComponentWitness { .. }
            | Self::GridLeader { .. }
            | Self::OptionGridLeader { .. } => {
                return Err(V::invalid(
                    "capability value has no ordinary SSA representation",
                ));
            }
        }
        Ok(())
    }
}

fn semantic_transport_contains_execution_v1<V: SemanticTransportVisitorV1>(
    binding: &SemanticValueBindingV1,
    visitor: &mut V,
) -> Result<bool, V::Error> {
    visitor.node()?;
    match binding {
        SemanticValueBindingV1::Aggregate(fields) => {
            for field in fields {
                if semantic_transport_contains_execution_v1(field, visitor)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        SemanticValueBindingV1::Enum { payloads, .. } => {
            for fields in payloads.values() {
                visitor.node()?;
                for field in fields {
                    if semantic_transport_contains_execution_v1(field, visitor)? {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }
        _ => Ok(semantic_binding_contains_execution_v29(binding)),
    }
}

fn require_components(
    block: SemanticBlockIdV1,
    values: Vec<(ValueId, Type)>,
    expected_type: Type,
    expected_count: usize,
    description: &'static str,
) -> Result<Vec<(ValueId, Type)>, ProductionSemanticKirErrorV1> {
    if values.len() != expected_count || values.iter().any(|(_, actual)| actual != &expected_type) {
        return Err(unsupported(0, Some(block.index()), None, description));
    }
    Ok(values)
}

fn require_single_u32_component(
    block: SemanticBlockIdV1,
    binding: SemanticValueBindingV1,
    description: &'static str,
) -> Result<ValueId, ProductionSemanticKirErrorV1> {
    Ok(require_components(
        block,
        binding
            .values()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?,
        Type::Scalar(ScalarType::U32),
        1,
        description,
    )?[0]
        .0)
}

fn index_and_u64_are_transport_equivalent(actual: &Type, expected: &Type) -> bool {
    matches!(
        (actual, expected),
        (
            Type::Scalar(ScalarType::Index),
            Type::Scalar(ScalarType::U64)
        ) | (
            Type::Scalar(ScalarType::U64),
            Type::Scalar(ScalarType::Index)
        )
    )
}

fn issued_reference_binding_v1(
    types: &[SemanticTypeDeclV1],
    compiler_issued: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    reference_type: SemanticTypeIdV1,
    projection: &SemanticProjectionV1,
) -> Option<SemanticPromotedBindingV1> {
    let SemanticTypeShapeV1::Pointer(reference) = types
        .get(reference_type.index() as usize)
        .map(SemanticTypeDeclV1::shape)?
    else {
        return None;
    };
    if projection.kind() != SemanticProjectionKindV1::Dereference
        || reference.kind() != SemanticPointerKindV1::Reference
        || reference.address_space() != 0
        || reference.pointer_width_bits() != 64
        || reference.metadata() != SemanticPointerMetadataV1::None
        || reference.pointee() != projection.result_type()
    {
        return None;
    }
    compiler_issued.get(&reference.pointee()).copied()
}

// Observe the arithmetic field of an already-issued lane carrier. The result
// deliberately loses the nominal capability and cannot authorize wave execution.
fn wave_lane_arithmetic_field_v1(
    types: &[SemanticTypeDeclV1],
    compiler_issued: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    carrier_type: SemanticTypeIdV1,
    projection: &SemanticProjectionV1,
    binding: &SemanticValueBindingV1,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Option<SemanticValueBindingV1>, ProductionSemanticKirErrorV1> {
    budget.charge_work(4)?;
    let SemanticValueBindingV1::WaveLane { value, wave } = binding else {
        return Ok(None);
    };
    let SemanticProjectionKindV1::Field(selected_field) = projection.kind() else {
        return Ok(None);
    };
    if wave.width != 64 {
        return Ok(None);
    }
    charge_execution_cfg_lookup_v29(compiler_issued.len(), budget)?;
    if !matches!(compiler_issued.get(&carrier_type),
        Some(SemanticPromotedBindingV1::WaveLane { wave_width: 64 }))
    {
        return Ok(None);
    }
    let Some(SemanticTypeShapeV1::Aggregate(fields)) = types
        .get(carrier_type.index() as usize).map(SemanticTypeDeclV1::shape)
    else {
        return Ok(None);
    };
    // Match the closed source ancestry relation: one ordinary u32 runtime
    // field, at most eight fields total, and only zero-sized marker siblings.
    if fields.fields().len() > 8 {
        return Ok(None);
    }
    let mut runtime_field = None;
    for (index, field_type) in fields.fields().iter().copied().enumerate() {
        budget.charge_work(2)?;
        let Some(field) = types.get(field_type.index() as usize) else {
            return Ok(None);
        };
        if field.layout().size_bytes() == Some(0) {
            continue;
        }
        if field.rust_type_kind() != fe2o3_mir_model::semantic_mir_v1::SemanticRustTypeKindV1::Ordinary
            || !matches!(field.shape(), SemanticTypeShapeV1::Scalar(
                SemanticScalarTypeV1::Integer { signed: false, bits: 32 }))
            || runtime_field.replace((index as u32, field_type)).is_some()
        {
            return Ok(None);
        }
    }
    if runtime_field != Some((selected_field, projection.result_type())) {
        return Ok(None);
    }
    Ok(Some(SemanticValueBindingV1::Value {
        id: *value,
        ty: Type::Scalar(ScalarType::U32),
    }))
}

fn wave_lane_reference_dereference_matches_v1(
    types: &[SemanticTypeDeclV1],
    compiler_issued: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    reference_type: SemanticTypeIdV1,
    projection: &SemanticProjectionV1,
    binding: &SemanticValueBindingV1,
) -> bool {
    let SemanticValueBindingV1::WaveLane { wave, .. } = binding else {
        return false;
    };
    wave.width == 64
        && matches!(
            issued_reference_binding_v1(types, compiler_issued, reference_type, projection),
            Some(SemanticPromotedBindingV1::WaveLane { wave_width })
                if wave_width == wave.width
        )
}

fn issued_capability_reference_dereference_matches_v1(
    types: &[SemanticTypeDeclV1],
    compiler_issued: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    reference_type: SemanticTypeIdV1,
    projection: &SemanticProjectionV1,
    binding: &SemanticValueBindingV1,
) -> bool {
    match binding {
        SemanticValueBindingV1::WaveLane { .. } => wave_lane_reference_dereference_matches_v1(
            types,
            compiler_issued,
            reference_type,
            projection,
            binding,
        ),
        SemanticValueBindingV1::WorkgroupLdsScope => matches!(
            issued_reference_binding_v1(types, compiler_issued, reference_type, projection),
            Some(SemanticPromotedBindingV1::WorkgroupLdsScope)
        ),
        _ => false,
    }
}

fn require_current_wave_lane(
    block: SemanticBlockIdV1,
    binding: SemanticValueBindingV1,
    expected_width: u32,
    description: &'static str,
) -> Result<(ValueId, SemanticCurrentWaveV1), ProductionSemanticKirErrorV1> {
    let SemanticValueBindingV1::WaveLane { value, wave } = binding else {
        return Err(unsupported(0, Some(block.index()), None, description));
    };
    if wave.width != expected_width {
        return Err(unsupported(0, Some(block.index()), None, description));
    }
    Ok((value, wave))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SemanticMfmaOperandBasesV1<T> {
    minor: T,
    reduction: T,
}

fn semantic_mfma_operand_bases_v1<T>(
    role: fe2o3_mir_model::semantic_mir_v1::SemanticMfmaOperandRoleV1,
    first: T,
    second: T,
) -> SemanticMfmaOperandBasesV1<T> {
    use fe2o3_mir_model::semantic_mir_v1::SemanticMfmaOperandRoleV1;

    match role {
        SemanticMfmaOperandRoleV1::A => SemanticMfmaOperandBasesV1 {
            minor: first,
            reduction: second,
        },
        SemanticMfmaOperandRoleV1::B => SemanticMfmaOperandBasesV1 {
            minor: second,
            reduction: first,
        },
    }
}

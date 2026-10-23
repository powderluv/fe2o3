//! Memory-effect control-flow correspondence and authenticated fallback expansion.

use super::*;

pub(super) fn validate_effect_control_flow_v1(
    body: &FunctionBody,
    ranked: &fe2o3_pliron::ProductionRankedKernelV1,
    locations: &[(
        SemanticAccessSiteV1,
        FunctionOperationLocation,
        u32,
        (u32, u32),
    )],
    total_fallback_reads: &BTreeSet<SemanticAccessSiteV1>,
    budget: &mut dyn CorrelationChargeV18,
) -> Result<(), ProductionMirPlironTranslationErrorV1> {
    let mut kir_events = BTreeMap::<u32, Vec<(u64, SemanticAccessSiteV1)>>::new();
    let mut ranked_events = BTreeMap::<u32, Vec<(u64, SemanticAccessSiteV1)>>::new();
    let mut kir_optional = BTreeSet::new();
    let mut seen_ranked_effects = BTreeMap::<(u32, u32), SemanticAccessSiteV1>::new();
    for (site, kir, operation_access_ordinal, ranked_location) in locations {
        budget
            .charge()
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
        if let Some(first) = seen_ranked_effects.get(ranked_location) {
            if (first.block, first.statement) != (site.block, site.statement) {
                return Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
                    first_semantic_block: first.block,
                    first_semantic_statement: first.statement,
                    second_semantic_block: site.block,
                    second_semantic_statement: site.statement,
                });
            }
            continue;
        }
        seen_ranked_effects.insert(*ranked_location, *site);
        let kir_order = (u64::try_from(kir.operation_index)
            .map_err(|_| ProductionMirPlironTranslationErrorV1::ResourceLimit)?
            << 32)
            | u64::from(*operation_access_ordinal);
        if total_fallback_reads.contains(site) {
            for _ in &body.blocks {
                budget
                    .charge()
                    .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
            }
            let operation = body
                .blocks
                .iter()
                .find(|block| block.id == kir.block)
                .and_then(|block| block.operations.get(kir.operation_index))
                .ok_or(ProductionMirPlironTranslationErrorV1::KernelShape)?;
            if matches!(operation.kind, OperationKind::GuardedLoad { .. }) {
                if !ranked_total_fallback_cfg_v1(ranked, *ranked_location, budget)
                    .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?
                {
                    return Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
                        first_semantic_block: site.block,
                        first_semantic_statement: site.statement,
                        second_semantic_block: site.block,
                        second_semantic_statement: site.statement,
                    });
                }
                kir_optional.insert((kir.block.0, kir_order));
            }
        }
        kir_events.entry(kir.block.0).or_default().push((
            (u64::try_from(kir.operation_index)
                .map_err(|_| ProductionMirPlironTranslationErrorV1::ResourceLimit)?
                << 32)
                | u64::from(*operation_access_ordinal),
            *site,
        ));
        ranked_events
            .entry(ranked_location.0)
            .or_default()
            .push((u64::from(ranked_location.1) << 32, *site));
    }
    for events in kir_events.values_mut().chain(ranked_events.values_mut()) {
        events.sort_unstable();
    }
    let kir_successors = body
        .blocks
        .iter()
        .map(|block| {
            let successors = kir_terminator_successors_v1(block.terminator.as_ref()?)?;
            Some((block.id.0, successors))
        })
        .collect::<Option<BTreeMap<_, _>>>()
        .ok_or(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
            first_semantic_block: 0,
            first_semantic_statement: None,
            second_semantic_block: 0,
            second_semantic_statement: None,
        })?;
    let ranked_successors = ranked
        .blocks()
        .iter()
        .enumerate()
        .map(|(block, contents)| {
            Some((
                u32::try_from(block).ok()?,
                ranked_terminator_successors_v1(contents.terminator()),
            ))
        })
        .collect::<Option<BTreeMap<_, _>>>()
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let kir_entry = body
        .blocks
        .first()
        .map(|block| block.id.0)
        .ok_or(ProductionMirPlironTranslationErrorV1::KernelShape)?;
    let (kir_events, kir_successors) = if kir_optional.is_empty() {
        (kir_events, kir_successors)
    } else {
        expand_optional_effects_v1(&kir_events, &kir_successors, &kir_optional, budget)
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?
    };
    let kir_flow = effect_flow_signature_v1(kir_entry, &kir_events, &kir_successors, budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    let ranked_flow = effect_flow_signature_v1(0, &ranked_events, &ranked_successors, budget)
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    if kir_flow == ranked_flow {
        return Ok(());
    }
    let differing = kir_flow
        .next_effects
        .symmetric_difference(&ranked_flow.next_effects)
        .next()
        .copied()
        .or_else(|| {
            kir_flow
                .entry_effects
                .symmetric_difference(&ranked_flow.entry_effects)
                .next()
                .copied()
                .map(|site| (site, site))
        })
        .unwrap_or((
            SemanticAccessSiteV1 {
                block: 0,
                statement: None,
                ordinal: 0,
            },
            SemanticAccessSiteV1 {
                block: 0,
                statement: None,
                ordinal: 0,
            },
        ));
    Err(ProductionMirPlironTranslationErrorV1::ControlFlowMismatch {
        first_semantic_block: differing.0.block,
        first_semantic_statement: differing.0.statement,
        second_semantic_block: differing.1.block,
        second_semantic_statement: differing.1.statement,
    })
}

pub(super) fn ranked_total_fallback_cfg_v1(
    recipe: &fe2o3_pliron::ProductionRankedKernelV1,
    location: (u32, u32),
    budget: &mut dyn CorrelationChargeV18,
) -> Option<bool> {
    use fe2o3_pliron::ProductionRankedTerminatorV1 as T;
    let a = location.0;
    let Some(g) = a.checked_sub(2) else {
        return Some(false);
    };
    let f = a.checked_add(1)?;
    let k = a.checked_add(2)?;
    let Some(access) = recipe.blocks().get(a as usize) else {
        return Some(false);
    };
    let [
        ProductionRankedOperationV1::Access {
            kind: dialect_kernel::AccessKindAttr::Read,
            view: ProductionRankedValueV1::Local(view),
            indices,
        },
    ] = access.operations()
    else {
        return Some(false);
    };
    let [row, column] = indices.as_slice() else {
        return Some(false);
    };
    if location.1 != 0 {
        return Some(false);
    }
    let mut extents = None;
    for block in recipe.blocks() {
        for operation in block.operations() {
            budget.charge()?;
            if let ProductionRankedOperationV1::ViewInSpace {
                result,
                writable: false,
                shape,
                dynamic_extents,
                memory_space: dialect_kernel::MemorySpaceAttr::Global,
                ..
            } = operation
            {
                if result == view {
                    if shape.as_slice() != [0, 0] || dynamic_extents.len() != 2 || extents.is_some()
                    {
                        return Some(false);
                    }
                    extents = Some(dynamic_extents);
                }
            }
        }
    }
    let Some(extents) = extents else {
        return Some(false);
    };
    let count = access.index_argument_count();
    for b in [g, g + 1, f, k] {
        budget.charge()?;
        let Some(block) = recipe.blocks().get(b as usize) else {
            return Some(false);
        };
        if block.index_argument_count() != count {
            return Some(false);
        }
    }
    if !recipe.blocks()[g as usize + 1].operations().is_empty()
        || !recipe.blocks()[f as usize].operations().is_empty()
    {
        return Some(false);
    }
    let identity =
        |args: &[ProductionRankedValueV1], block, budget: &mut dyn CorrelationChargeV18| {
            if args.len() != count as usize {
                return Some(false);
            }
            for (i, arg) in args.iter().enumerate() {
                budget.charge()?;
                if *arg
                    != (ProductionRankedValueV1::BlockArgument {
                        block,
                        argument: u32::try_from(i).ok()?,
                    })
                {
                    return Some(false);
                }
            }
            Some(true)
        };
    for (b, lhs, rhs, next) in [
        (g, *row, extents[0], g + 1),
        (g + 1, *column, extents[1], a),
    ] {
        budget.charge()?;
        let exact = match recipe.blocks()[b as usize].terminator() {
            T::IndexLessThan {
                lhs: l,
                rhs: r,
                true_block,
                false_block,
            } => count == 0 && *l == lhs && *r == rhs && *true_block == next && *false_block == f,
            T::IndexLessThanArgs {
                lhs: l,
                rhs: r,
                true_arguments,
                false_arguments,
                true_block,
                false_block,
            } => {
                *l == lhs
                    && *r == rhs
                    && *true_block == next
                    && *false_block == f
                    && identity(true_arguments, b, budget)?
                    && identity(false_arguments, b, budget)?
            }
            _ => false,
        };
        if !exact {
            return Some(false);
        }
    }
    for b in [a, f] {
        budget.charge()?;
        let exact = match recipe.blocks()[b as usize].terminator() {
            T::Branch { target } => count == 0 && *target == k,
            T::BranchArgs { arguments, target } => *target == k && identity(arguments, b, budget)?,
            _ => false,
        };
        if !exact {
            return Some(false);
        }
    }
    // An external jump into a guard interior could bypass a comparison or
    // manufacture different live arguments; only the canonical edges qualify.
    for (b, block) in recipe.blocks().iter().enumerate() {
        let (targets, n) = ranked_terminator_successors_v18(block.terminator());
        for target in &targets[..n] {
            budget.charge()?;
            let source = u32::try_from(b).ok()?;
            if (*target == g + 1 && source != g)
                || (*target == a && source != g + 1)
                || (*target == f && source != g && source != g + 1)
                || (*target == k && source != a && source != f)
            {
                return Some(false);
            }
        }
    }
    Some(true)
}

// Effect order observes possible memory events, not evaluation of the guard.
// Split conditional memory into an event edge and a no-memory continuation,
// just as the ranked CFG materializes a total load_or fallback. Every event
// remains present and ordinary accesses remain mandatory.
pub(super) fn expand_optional_effects_v1(
    events: &BTreeMap<u32, Vec<(u64, SemanticAccessSiteV1)>>,
    successors: &BTreeMap<u32, Vec<u32>>,
    optional: &BTreeSet<(u32, u64)>,
    budget: &mut dyn CorrelationChargeV18,
) -> Option<(
    BTreeMap<u32, Vec<(u64, SemanticAccessSiteV1)>>,
    BTreeMap<u32, Vec<u32>>,
)> {
    let mut next = successors.keys().next_back()?.checked_add(1)?;
    let mut expanded_events = BTreeMap::<u32, Vec<(u64, SemanticAccessSiteV1)>>::new();
    let mut expanded_successors = BTreeMap::new();
    for (&block, targets) in successors {
        budget.charge()?;
        let mut current = block;
        if let Some(rows) = events.get(&block) {
            for &(order, site) in rows {
                budget.charge()?;
                if optional.contains(&(block, order)) {
                    let access = next;
                    let continuation = next.checked_add(1)?;
                    next = next.checked_add(2)?;
                    budget.charge()?;
                    expanded_successors.insert(current, vec![access, continuation]);
                    expanded_events.insert(access, vec![(order, site)]);
                    expanded_successors.insert(access, vec![continuation]);
                    current = continuation;
                } else {
                    expanded_events
                        .entry(current)
                        .or_default()
                        .push((order, site));
                }
            }
        }
        let mut copied = Vec::new();
        for &target in targets {
            budget.charge()?;
            if !successors.contains_key(&target) {
                return None;
            }
            copied.push(target);
        }
        expanded_successors.insert(current, copied);
    }
    Some((expanded_events, expanded_successors))
}

fn ordinary_source_has_raw_wave64_shuffle_v1(source: &AdmittedInertSemanticMirV1) -> bool {
    source.callables().iter().any(|callable| {
        matches!(
            callable,
            SemanticCallableDeclV1::CompilerIntrinsic {
                operation: SemanticCompilerIntrinsicOperationV1::Gfx942Wave64ShuffleIndex { .. },
                ..
            }
        )
    })
}

// The actual reference holder cannot acquire a new SSA meaning after its
// authenticated definition. Conservatively close alias creation and escape;
// ordinary pointer/reference lowering remains unchanged for other functions.
// Bounded reachability over this immutable owner's original captured edges.
// Successor seeding distinguishes a later loop iteration from same-block order.
fn wave64_holder_source_reachability_v1(
    rows: &fe2o3_pliron::ProductionSemanticSsaFunctionOccurrencesV1<'_>,
    blocks: usize,
    start: usize,
    removed: Option<usize>,
    successors_only: bool,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<[bool; 128], ProductionSemanticKirErrorV1> {
    budget.charge_work(2)?;
    if blocks > 128 || start >= blocks || removed.is_some_and(|block| block >= blocks) {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    let mut reached = [false; 128];
    if !successors_only && removed != Some(start) {
        reached[start] = true;
    }
    for edge in rows.successors() {
        budget.charge_work(1)?;
        let from = edge.id().source().get() as usize;
        let to = edge.edge().target().index() as usize;
        if from >= blocks || to >= blocks {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        if successors_only && from == start && removed != Some(to) {
            reached[to] = true;
        }
    }
    for _ in 0..128 {
        let mut changed = false;
        for edge in rows.successors() {
            budget.charge_work(1)?;
            let from = edge.id().source().get() as usize;
            let to = edge.edge().target().index() as usize;
            if reached[from] && removed != Some(to) && !reached[to] {
                reached[to] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(reached)
}
fn require_ordered_wave64_holder_storage_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    local: u32,
    borrow: (SemanticBlockIdV1, u32),
    initial_live: Option<(usize, usize)>,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    budget.charge_work(4)?;
    let held = owner
        .occurrence_storage()
        .ok_or(ArgumentResourceV1::Accounting)?
        .retained_storage();
    if budget.storage() < held {
        return Err(ArgumentResourceV1::Accounting.into());
    }
    let rows = owner
        .occurrences_v1()
        .and_then(|capture| capture.function(function))
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    if !std::ptr::eq(rows.owner(), owner) {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    let body = &owner.source_semantic().functions()[function.index() as usize];
    let blocks = body.blocks().len();
    let borrow_block = borrow.0.index() as usize;
    let borrow_statement = borrow.1 as usize;
    let entry = body.entry().index() as usize;
    let before_borrow = wave64_holder_source_reachability_v1(
        &rows,
        blocks,
        entry,
        Some(borrow_block),
        false,
        budget,
    )?;
    let after_borrow =
        wave64_holder_source_reachability_v1(&rows, blocks, borrow_block, None, true, budget)?;
    if let Some((live_block, live_statement)) = initial_live {
        let bypass_live = wave64_holder_source_reachability_v1(
            &rows,
            blocks,
            entry,
            Some(live_block),
            false,
            budget,
        )?;
        if (live_block == borrow_block && live_statement >= borrow_statement)
            || (live_block != borrow_block && bypass_live[borrow_block])
            || after_borrow[live_block]
        {
            return Err(unsupported(
                function.index(),
                Some(live_block as u32),
                Some(live_statement as u32),
                "checked Wave64 holder initial storage live does not precede and dominate its unique borrow",
            ));
        }
    }
    let mut definitions = 0usize;
    for event in rows.events() {
        budget.charge_work(1)?;
        if event.site()
            == (fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Statement {
                block: fe2o3_mir_model::SsaBlockIdV1::new(borrow.0.index()),
                statement: borrow.1,
            })
            && event.role() == fe2o3_pliron::ProductionSemanticSsaEventRoleV1::DestinationDefine
            && event.event()
                == fe2o3_mir_model::SsaEventV1::Define(fe2o3_mir_model::SsaVariableIdV1::new(local))
        {
            if !event.is_reachable()
                || !event.is_promoted()
                || !matches!(event.resolved(), Some(fe2o3_mir_model::SsaResolvedEventV1::Define { variable, .. }) if variable.get() == local)
            {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            definitions += 1;
        }
        let fe2o3_mir_model::SsaEventV1::Use(variable) = event.event() else {
            continue;
        };
        if variable.get() != local {
            continue;
        }
        let (block, statement) = match event.site() {
            fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Statement { block, statement } => {
                (block.get() as usize, statement as usize)
            }
            fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Terminator { block } => {
                (block.get() as usize, usize::MAX)
            }
        };
        if block >= blocks
            || before_borrow[block]
            || (block == borrow_block && statement <= borrow_statement)
        {
            return Err(unsupported(
                function.index(),
                Some(block as u32),
                None,
                "checked Wave64 holder source read is not dominated by its unique borrow",
            ));
        }
    }
    if definitions != 1 {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    for (bi, block) in body.blocks().iter().enumerate() {
        for (si, statement) in block.statements().iter().enumerate() {
            budget.charge_work(1)?;
            if !matches!(statement.kind(), SemanticStatementKindV1::StorageDead(value) if value.index() == local)
            {
                continue;
            }
            let after_dead =
                wave64_holder_source_reachability_v1(&rows, blocks, bi, None, true, budget)?;
            if after_dead[borrow_block] || (bi == borrow_block && si < borrow_statement) {
                return Err(unsupported(
                    function.index(),
                    Some(bi as u32),
                    Some(si as u32),
                    "checked Wave64 holder storage end can reach its borrow again",
                ));
            }
            for event in rows.events() {
                budget.charge_work(1)?;
                if !matches!(event.event(), fe2o3_mir_model::SsaEventV1::Use(variable) if variable.get() == local)
                {
                    continue;
                }
                let (use_block, use_statement) = match event.site() {
                    fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Statement {
                        block,
                        statement,
                    } => (block.get() as usize, statement as usize),
                    fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Terminator { block } => {
                        (block.get() as usize, usize::MAX)
                    }
                };
                if use_block >= blocks
                    || after_dead[use_block]
                    || (use_block == bi && use_statement > si)
                {
                    return Err(unsupported(
                        function.index(),
                        Some(bi as u32),
                        Some(si as u32),
                        "checked Wave64 holder storage end can precede an original source read",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn require_unwritten_wave64_local_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    local: u32,
    original_borrow: Option<(SemanticBlockIdV1, u32)>,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let declaration = owner
        .source_semantic()
        .functions()
        .get(function.index() as usize)
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let mut writes = 0usize;
    let mut initial_live = None;
    let mut has_storage_markers = false;
    for (bi, block) in declaration.blocks().iter().enumerate() {
        for (si, statement) in block.statements().iter().enumerate() {
            budget.charge_work(2)?;
            match statement.kind() {
                SemanticStatementKindV1::Assign(assignment)
                    if assignment.destination().local().index() == local =>
                {
                    if original_borrow
                        != Some((SemanticBlockIdV1::from_index(bi as u32), si as u32))
                        || !assignment.destination().projections().is_empty()
                        || !matches!(
                            assignment.value().kind(),
                            SemanticRvalueKindV1::Borrow {
                                kind: SemanticBorrowKindV1::Shared,
                                ..
                            }
                        )
                    {
                        return Err(unsupported(
                            function.index(),
                            Some(bi as u32),
                            Some(si as u32),
                            "checked Wave64 holder has another original write",
                        ));
                    }
                    writes += 1;
                }
                SemanticStatementKindV1::StorageLive(value) if value.index() == local => {
                    if original_borrow.is_none() || initial_live.replace((bi, si)).is_some() {
                        return Err(unsupported(
                            function.index(),
                            Some(bi as u32),
                            Some(si as u32),
                            "checked Wave64 holder allocation lifetime restarted or changed for a formal",
                        ));
                    }
                    has_storage_markers = true;
                }
                SemanticStatementKindV1::StorageDead(value) if value.index() == local => {
                    if original_borrow.is_none() {
                        return Err(unsupported(
                            function.index(),
                            Some(bi as u32),
                            Some(si as u32),
                            "checked Wave64 formal has a storage lifetime change",
                        ));
                    }
                    has_storage_markers = true;
                }
                SemanticStatementKindV1::Deinitialize(place)
                | SemanticStatementKindV1::SetDiscriminant { place, .. }
                    if place.local().index() == local =>
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                SemanticStatementKindV1::Store(store)
                    if store.destination().local().index() == local =>
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                SemanticStatementKindV1::AtomicRmw(atomic)
                    if atomic.destination().local().index() == local
                        || atomic.address().local().index() == local =>
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                SemanticStatementKindV1::AtomicCompareExchange(atomic)
                    if atomic.destination().local().index() == local
                        || atomic.address().local().index() == local =>
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                _ => {}
            }
        }
        if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() {
            if call
                .destination()
                .is_some_and(|destination| destination.place().local().index() == local)
            {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        }
    }
    if writes != usize::from(original_borrow.is_some()) {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    if has_storage_markers {
        require_ordered_wave64_holder_storage_v1(
            owner,
            function,
            local,
            original_borrow.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?,
            initial_live,
            budget,
        )?;
    }
    Ok(())
}
fn closed_wave64_scalar_projection_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    place: &SemanticPlaceV1,
) -> bool {
    matches!(
        place
            .projections()
            .first()
            .map(|projection| projection.kind()),
        Some(SemanticProjectionKindV1::Dereference)
    ) && place
        .projections()
        .iter()
        .skip(1)
        .all(|projection| matches!(projection.kind(), SemanticProjectionKindV1::Field(_)))
        && matches!(
            owner
                .source_semantic()
                .types()
                .get(place.ty().index() as usize)
                .map(SemanticTypeDeclV1::shape),
            Some(SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 32 | 64
            }))
        )
}
fn require_closed_wave64_reference_holder_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    local: u32,
    original_borrow: Option<(SemanticBlockIdV1, u32)>,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    require_unwritten_wave64_local_v1(owner, function, local, original_borrow, budget)?;
    let declaration = &owner.source_semantic().functions()[function.index() as usize];
    for (bi, block) in declaration.blocks().iter().enumerate() {
        for (si, statement) in block.statements().iter().enumerate() {
            budget.charge_work(2)?;
            match statement.kind() {
                SemanticStatementKindV1::Assign(assignment) => {
                    assignment.value().kind().try_visit_operands(|operand| {
                        budget.charge_work(1)?;
                        if let SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) =
                            operand
                        {
                            if place.local().index() == local
                                && !closed_wave64_scalar_projection_v1(owner, place)
                            {
                                return Err(unsupported(
                                    function.index(),
                                    Some(bi as u32),
                                    Some(si as u32),
                                    "checked Wave64 reference alias or escape",
                                ));
                            }
                        }
                        Ok(())
                    })?;
                    match assignment.value().kind() {
                        SemanticRvalueKindV1::Borrow { place, .. }
                        | SemanticRvalueKindV1::AddressOf { place, .. }
                        | SemanticRvalueKindV1::Length(place)
                        | SemanticRvalueKindV1::Discriminant(place)
                            if place.local().index() == local =>
                        {
                            return Err(unsupported(
                                function.index(),
                                Some(bi as u32),
                                Some(si as u32),
                                "checked Wave64 reference address or reborrow escape",
                            ));
                        }
                        SemanticRvalueKindV1::Load(load)
                            if load.source().local().index() == local =>
                        {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        _ => {}
                    }
                }
                SemanticStatementKindV1::Store(store) => {
                    if let SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) =
                        store.value()
                    {
                        if place.local().index() == local {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                    }
                }
                _ => {}
            }
        }
        match block.terminator().kind() {
            SemanticTerminatorKindV1::Call(call) => {
                for (argument, operand) in call.arguments().iter().enumerate() {
                    budget.charge_work(2)?;
                    let (SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)) = operand
                    else {
                        continue;
                    };
                    if place.local().index() != local
                        || closed_wave64_scalar_projection_v1(owner, place)
                    {
                        continue;
                    }
                    if !place.projections().is_empty() {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    match owner
                        .source_semantic()
                        .callables()
                        .get(call.callee().index() as usize)
                    {
                        Some(SemanticCallableDeclV1::Defined { .. }) => {
                            let checked = budget.checked_wave64_call_v1(
                                owner,
                                function,
                                SemanticBlockIdV1::from_index(bi as u32),
                                &mut |view, _| {
                                    if !std::ptr::eq(view.owner(), owner)
                                        || !std::ptr::eq(view.source_call(), call)
                                        || argument > 1
                                    {
                                        return Err(
                                            ProductionSemanticKirErrorV1::CorrespondenceMismatch,
                                        );
                                    }
                                    Ok(())
                                },
                            )?;
                            if !checked {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            }
                        }
                        Some(SemanticCallableDeclV1::CompilerIntrinsic {
                            operation:
                                SemanticCompilerIntrinsicOperationV1::Gfx942Wave64ShuffleIndex {
                                    ..
                                },
                            ..
                        }) if argument == 0 => {}
                        _ => {
                            return Err(unsupported(
                                function.index(),
                                Some(bi as u32),
                                None,
                                "checked Wave64 reference foreign call escape",
                            ));
                        }
                    }
                }
            }
            SemanticTerminatorKindV1::TailCall(_) => {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            _ => {}
        }
    }
    Ok(())
}

impl SemanticFunctionLoweringV1<'_, '_> {
    fn lower_checked_wave64_context_borrow_v1(
        &mut self,
        block: SemanticBlockIdV1,
        statement: u32,
        result_type: SemanticTypeIdV1,
        value: &SemanticRvalueKindV1,
        expected: Wave64NominalReferenceDescriptorV1,
    ) -> Result<Option<SemanticValueBindingV1>, ProductionSemanticKirErrorV1> {
        let owner = self
            .wave64_references
            .owner
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        let mut result = None;
        self.with_emission_budget_v1(|this, budget| {
            let checked = budget.checked_wave64_context_borrow_v1(
                owner,
                this.semantic_function,
                block,
                statement,
                &mut |view, budget| {
                    if !std::ptr::eq(view.owner(), owner)
                        || !std::ptr::eq(view.source_rvalue().kind(), value)
                        || Wave64NominalReferenceDescriptorV1::from_checked_context_borrow_v1(view)?
                            != expected
                        || view.reference_type() != result_type
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    require_closed_wave64_reference_holder_v1(
                        owner,
                        this.semantic_function,
                        expected.local,
                        Some((block, statement)),
                        budget,
                    )?;
                    let (function, issuer_block, local) = view.issuer();
                    if function != this.semantic_function
                        || !matches!(
                            this.locals.get(local.index() as usize),
                            Some(Some(SemanticValueBindingV1::CollectiveContext))
                        )
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    let SemanticTerminatorKindV1::Call(original) =
                        owner.source_semantic().functions()[function.index() as usize].blocks()
                            [issuer_block.index() as usize]
                            .terminator()
                            .kind()
                    else {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    };
                    if !std::ptr::eq(original, view.source_call()) {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    result = Some(expected.reconstruct_v1(this.types, result_type, &[])?);
                    Ok(())
                },
            )?;
            if !checked || result.is_none() {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            Ok(())
        })?;
        Ok(result)
    }
}

fn wave64_original_call_argument_ssa_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    argument: u32,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<SsaValueV1, ProductionSemanticKirErrorV1> {
    let SemanticTerminatorKindV1::Call(call) = owner.source_semantic().functions()
        [function.index() as usize]
        .blocks()[block.index() as usize]
        .terminator()
        .kind()
    else {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    };
    let Some(SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)) =
        call.arguments().get(argument as usize)
    else {
        return Err(unsupported(
            function.index(),
            Some(block.index()),
            None,
            "nominal call transport requires actual original SSA operand",
        ));
    };
    if !place.projections().is_empty() {
        return Err(unsupported(
            function.index(),
            Some(block.index()),
            None,
            "nominal call projected source requires exact projected SSA binding join",
        ));
    }
    let occurrences = owner
        .occurrences_v1()
        .and_then(|rows| rows.function(function))
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let mut actual = None;
    for event in occurrences.events() {
        budget.charge_work(2)?;
        if event.site()
            == (fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Terminator {
                block: SsaBlockIdV1::new(block.index()),
            })
            && event.operand()
                == fe2o3_pliron::ProductionSemanticSsaOperandRoleV1::CallArgument(argument)
            && event.role() == fe2o3_pliron::ProductionSemanticSsaEventRoleV1::BaseUse
        {
            let Some(SsaResolvedEventV1::Use { value, .. }) = event.resolved() else {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            };
            if !event.is_promoted() || !event.is_reachable() || actual.replace(value).is_some() {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        }
    }
    actual.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)
}
fn same_wave64_original_binding_v1(
    left: &SemanticValueBindingV1,
    right: &SemanticValueBindingV1,
    budget: &mut dyn SemanticEmissionBudgetV1,
    depth: usize,
) -> Result<bool, ProductionSemanticKirErrorV1> {
    budget.charge_work(2)?;
    if depth >= 64 {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    match (left, right) {
        (SemanticValueBindingV1::Unit, SemanticValueBindingV1::Unit)
        | (SemanticValueBindingV1::CollectiveContext, SemanticValueBindingV1::CollectiveContext) => {
            Ok(true)
        }
        (
            SemanticValueBindingV1::Value { id: a, ty: at },
            SemanticValueBindingV1::Value { id: b, ty: bt },
        ) => Ok(a == b && invocation_equal_types_v1(at, bt, budget)?),
        (SemanticValueBindingV1::Aggregate(a), SemanticValueBindingV1::Aggregate(b)) => {
            if a.len() != b.len() {
                return Ok(false);
            }
            for (a, b) in a.iter().zip(b) {
                if !same_wave64_original_binding_v1(a, b, budget, depth + 1)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

// Fresh replay scopes borrow the real immutable source occurrence payload.
// Callers select the Wave64 path before entering; this private scope accounts
// the actual owner receipt, refunds only that input charge on every exit, and
// preserves body-owned output charges and all failure history.
fn with_wave64_retained_source_input_v1<'work, R>(
    owner: &ProductionSemanticSsaOwnerV1,
    budget: &mut ArgumentBudgetV1<'work>,
    body: impl FnOnce(&mut ArgumentBudgetV1<'work>) -> Result<R, ProductionSemanticKirErrorV1>,
) -> Result<R, ProductionSemanticKirErrorV1> {
    let receipt = owner
        .occurrence_storage()
        .ok_or(ArgumentResourceV1::Accounting)?;
    let bytes = receipt.retained_storage();
    let ledger = budget.work_ledger_identity_v1();
    let floor = budget.storage();
    budget.reserve_storage(bytes)?;
    let live_input = budget.storage();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(budget)));
    if budget.work_ledger_identity_v1() != ledger || budget.storage() < live_input {
        return Err(ArgumentResourceV1::Accounting.into());
    }
    let retained_output = budget
        .storage()
        .checked_sub(live_input)
        .ok_or(ArgumentResourceV1::Accounting)?;
    let output_floor = argument_sum_v1(&[floor, retained_output])?;
    let release = budget
        .storage()
        .checked_sub(output_floor)
        .ok_or(ArgumentResourceV1::Accounting)?;
    if release != bytes {
        return Err(ArgumentResourceV1::Accounting.into());
    }
    budget.release_storage(release)?;
    match result {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

// Arithmetic invariant for the existing reduction's closed source counter:
// one u32 init32, then only self >> 1. This issues no EXEC/loop authority.
fn checked_wave64_shift_counter_bound_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    local: u32,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Option<u128>, ProductionSemanticKirErrorV1> {
    budget.charge_work(4)?;
    let source = owner.source_semantic();
    let body = source
        .functions()
        .get(function.index() as usize)
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    let Some(declaration) = body.locals().get(local as usize) else {
        return Ok(None);
    };
    let ty = declaration.ty();
    if declaration.role() != SemanticLocalRoleV1::Temporary
        || source.types().get(ty.index() as usize).is_none_or(|ty| {
            ty.rust_type_kind()
                != fe2o3_mir_model::semantic_mir_v1::SemanticRustTypeKindV1::Ordinary
                || !matches!(
                    ty.shape(),
                    SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                        signed: false,
                        bits: 32
                    })
                )
        })
    {
        return Ok(None);
    }
    let mut init = None;
    let mut live = None;
    let mut definitions = 0usize;
    let mut shifts = 0usize;
    for (bi, block) in body.blocks().iter().enumerate() {
        for (si, statement) in block.statements().iter().enumerate() {
            budget.charge_work(2)?;
            match statement.kind() {
                SemanticStatementKindV1::Assign(assignment) => {
                    if assignment.destination().local().index() == local {
                        if !assignment.destination().projections().is_empty()
                            || assignment.destination().ty() != ty
                            || assignment.value().result_type() != ty
                        {
                            return Ok(None);
                        }
                        match assignment.value().kind() {
                            SemanticRvalueKindV1::Use(operand)
                                if exact_unsigned_semantic_constant_v1(
                                    source.types(),
                                    operand,
                                    ty,
                                ) == Some(32) =>
                            {
                                if init.replace((bi, si)).is_some() {
                                    return Ok(None);
                                }
                            }
                            SemanticRvalueKindV1::Binary {
                                operation: SemanticBinaryOpV1::ShiftRight,
                                left,
                                right,
                            } => {
                                if !matches!(left, SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)
                                    if place.local().index() == local && place.projections().is_empty() && place.ty() == ty)
                                    || !matches!(right, SemanticOperandV1::Constant(constant)
                                        if matches!(source.types().get(constant.ty().index() as usize).map(SemanticTypeDeclV1::shape),
                                            Some(SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer { bits: 32, .. })))
                                        && matches!(constant.value(), SemanticConstantValueV1::Scalar(value) if value.size_bytes() == 4 && value.bits() == 1))
                                {
                                    return Ok(None);
                                }
                                shifts += 1;
                            }
                            _ => return Ok(None),
                        }
                        definitions += 1;
                    }
                    match assignment.value().kind() {
                        SemanticRvalueKindV1::Borrow { place, .. }
                        | SemanticRvalueKindV1::AddressOf { place, .. }
                        | SemanticRvalueKindV1::Length(place)
                        | SemanticRvalueKindV1::Discriminant(place)
                            if place.local().index() == local =>
                        {
                            return Ok(None);
                        }
                        SemanticRvalueKindV1::Load(load)
                            if load.source().local().index() == local =>
                        {
                            return Ok(None);
                        }
                        _ => {}
                    }
                }
                SemanticStatementKindV1::StorageLive(value) if value.index() == local => {
                    if live.replace((bi, si)).is_some() {
                        return Ok(None);
                    }
                }
                SemanticStatementKindV1::Store(store)
                    if store.destination().local().index() == local =>
                {
                    return Ok(None);
                }
                SemanticStatementKindV1::Deinitialize(place)
                | SemanticStatementKindV1::SetDiscriminant { place, .. }
                    if place.local().index() == local =>
                {
                    return Ok(None);
                }
                SemanticStatementKindV1::AtomicRmw(atomic)
                    if atomic.destination().local().index() == local
                        || atomic.address().local().index() == local =>
                {
                    return Ok(None);
                }
                SemanticStatementKindV1::AtomicCompareExchange(atomic)
                    if atomic.destination().local().index() == local
                        || atomic.address().local().index() == local =>
                {
                    return Ok(None);
                }
                _ => {}
            }
        }
        if matches!(block.terminator().kind(), SemanticTerminatorKindV1::Call(call)
            if call.destination().is_some_and(|destination| destination.place().local().index() == local))
        {
            return Ok(None);
        }
    }
    let Some((init_block, init_statement)) = init else {
        return Ok(None);
    };
    if shifts == 0 {
        return Ok(None);
    }
    let held = owner
        .occurrence_storage()
        .ok_or(ArgumentResourceV1::Accounting)?
        .retained_storage();
    if budget.storage() < held {
        return Err(ArgumentResourceV1::Accounting.into());
    }
    let rows = owner
        .occurrences_v1()
        .and_then(|capture| capture.function(function))
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    if !std::ptr::eq(rows.owner(), owner) {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    let plan = owner
        .plan_for_function(function)
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
        .plan();
    budget.charge_work(plan.promoted_variables().len() + plan.entry_definitions().len())?;
    if !plan
        .promoted_variables()
        .iter()
        .any(|variable| variable.get() == local)
        || plan
            .entry_definitions()
            .iter()
            .any(|argument| argument.variable().get() == local)
    {
        return Ok(None);
    }
    let blocks = body.blocks().len();
    let entry = body.entry().index() as usize;
    let bypass_init = wave64_holder_source_reachability_v1(
        &rows,
        blocks,
        entry,
        Some(init_block),
        false,
        budget,
    )?;
    let after_init =
        wave64_holder_source_reachability_v1(&rows, blocks, init_block, None, true, budget)?;
    if after_init[init_block] {
        return Ok(None);
    }
    if let Some((live_block, live_statement)) = live {
        let bypass_live = wave64_holder_source_reachability_v1(
            &rows,
            blocks,
            entry,
            Some(live_block),
            false,
            budget,
        )?;
        if (live_block == init_block && live_statement >= init_statement)
            || (live_block != init_block && bypass_live[init_block])
            || after_init[live_block]
        {
            return Ok(None);
        }
    }
    let mut captured_definitions = 0usize;
    for event in rows.events() {
        budget.charge_work(1)?;
        if event.event().variable().get() != local {
            continue;
        }
        if !event.is_reachable() || !event.is_promoted() {
            return Ok(None);
        }
        let (block, statement) = match event.site() {
            fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Statement { block, statement } => {
                (block.get() as usize, statement as usize)
            }
            fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Terminator { block } => {
                (block.get() as usize, usize::MAX)
            }
        };
        if block >= blocks {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        match event.event() {
            fe2o3_mir_model::SsaEventV1::Define(_) => {
                if event.role() != fe2o3_pliron::ProductionSemanticSsaEventRoleV1::DestinationDefine
                    || !matches!(event.resolved(), Some(fe2o3_mir_model::SsaResolvedEventV1::Define { variable, .. }) if variable.get() == local)
                {
                    return Ok(None);
                }
                captured_definitions += 1;
            }
            fe2o3_mir_model::SsaEventV1::Use(_) => {
                if !matches!(event.resolved(), Some(fe2o3_mir_model::SsaResolvedEventV1::Use { variable, .. }) if variable.get() == local)
                    || bypass_init[block]
                    || (block == init_block && statement <= init_statement)
                {
                    return Ok(None);
                }
            }
            fe2o3_mir_model::SsaEventV1::Kill(_) => {
                let original = body.blocks()[block]
                    .statements()
                    .get(statement)
                    .map(|statement| statement.kind());
                if !matches!(original, Some(SemanticStatementKindV1::StorageLive(value) | SemanticStatementKindV1::StorageDead(value))
                    if value.index() == local)
                {
                    return Ok(None);
                }
            }
        }
    }
    if captured_definitions != definitions {
        return Ok(None);
    }
    for (bi, block) in body.blocks().iter().enumerate() {
        for (si, statement) in block.statements().iter().enumerate() {
            budget.charge_work(1)?;
            if !matches!(statement.kind(), SemanticStatementKindV1::StorageDead(value) if value.index() == local)
            {
                continue;
            }
            let after_dead =
                wave64_holder_source_reachability_v1(&rows, blocks, bi, None, true, budget)?;
            if after_dead[init_block] || (bi == init_block && si < init_statement) {
                return Ok(None);
            }
            for event in rows.events() {
                budget.charge_work(1)?;
                if !matches!(event.event(), fe2o3_mir_model::SsaEventV1::Use(variable) if variable.get() == local)
                {
                    continue;
                }
                let (use_block, use_statement) = match event.site() {
                    fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Statement {
                        block,
                        statement,
                    } => (block.get() as usize, statement as usize),
                    fe2o3_pliron::ProductionSemanticSsaOccurrenceSiteV1::Terminator { block } => {
                        (block.get() as usize, usize::MAX)
                    }
                };
                if use_block >= blocks
                    || after_dead[use_block]
                    || (use_block == bi && use_statement > si)
                {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(33))
}
impl SemanticFunctionLoweringV1<'_, '_> {
    fn seed_checked_wave64_shift_parameter_v1(
        &mut self,
        block: SemanticBlockIdV1,
        local: u32,
        source_value: SsaValueV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let Some(owner) = self.wave64_references.owner else {
            return Ok(());
        };
        self.with_emission_budget_v1(|this, budget| {
            budget.charge_work(4)?;
            let declaration = this.function.locals().get(local as usize)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            if declaration.role() != SemanticLocalRoleV1::Temporary { return Ok(()); }
            // This is an attempt selector only; the live source callback below
            // independently rederives the complete invariant before insertion.
            let mut candidate = false;
            for source_block in this.function.blocks() {
                for statement in source_block.statements() {
                    budget.charge_work(1)?;
                    if matches!(statement.kind(), SemanticStatementKindV1::Assign(assignment)
                        if assignment.destination().local().index() == local
                        && matches!(assignment.value().kind(), SemanticRvalueKindV1::Use(operand)
                            if exact_unsigned_semantic_constant_v1(this.types, operand, declaration.ty()) == Some(32))) { candidate = true; }
                }
            }
            if !candidate { return Ok(()); }
            budget.checked_wave64_helper_v1(owner, this.semantic_function, &mut |view, budget| {
                if !std::ptr::eq(view.representative().owner(), owner)
                    || !std::ptr::eq(view.representative().helper_declaration(), this.function) {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                if !matches!(view.representative().argument(0), Some(wave64_reference_call_v1::ArgumentV1::Snapshot { .. })) {
                    return Ok(());
                }
                let Some(bound) = checked_wave64_shift_counter_bound_v1(owner, this.semantic_function, local, budget)? else { return Ok(()) };
                let plan = owner.plan_for_function(this.semantic_function).ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?.plan();
                let variables = plan.transport_variables(SsaBlockIdV1::new(block.index()))
                    .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
                budget.charge_work(variables.len())?;
                if !variables.iter().any(|variable| variable.get() == local)
                    || source_value != (SsaValueV1::BlockArgument { block: SsaBlockIdV1::new(block.index()),
                        variable: fe2o3_mir_model::SsaVariableIdV1::new(local) }) { return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch); }
                charge_execution_cfg_lookup_v29(this.semantic_ssa_bindings.len(), budget)?;
                let Some(SemanticValueBindingV1::Value { id, ty }) = this.semantic_ssa_bindings.get(&source_value) else {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                };
                if *ty != Type::Scalar(ScalarType::U32)
                    || !matches!(this.locals.get(local as usize).and_then(Option::as_ref),
                        Some(SemanticValueBindingV1::Value { id: current, ty: Type::Scalar(ScalarType::U32) }) if current == id) {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let id = *id;
                charge_execution_cfg_lookup_v29(this.emitted_unsigned_exclusive_bounds.len(), budget)?;
                if !this.emitted_unsigned_exclusive_bounds.contains_key(&id) {
                    reserve_execution_cfg_map_entry_v29::<ValueId, u128>(this.emitted_unsigned_exclusive_bounds.len(), budget)?;
                }
                this.emitted_unsigned_exclusive_bounds.insert(id, bound);
                Ok(())
            })?;
            Ok(())
        })
    }
}

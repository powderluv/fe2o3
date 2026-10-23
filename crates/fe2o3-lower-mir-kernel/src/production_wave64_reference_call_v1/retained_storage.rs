impl<'a> Resolver<'a> {
    fn require_snapshot_dead_suffix(
        &self,
        function: SemanticFunctionIdV1,
        local: SemanticLocalIdV1,
        dead_block: usize,
        dead_statement: usize,
        budget: &mut Budget<'_>,
    ) -> ResultV1<()> {
        let declaration = &self.owner.source_semantic().functions()[function.index() as usize];
        if declaration.blocks().len() > 128 || dead_block >= declaration.blocks().len() {
            return refuse("retained lifetime CFG coordinate");
        }
        let mut holders = [u32::MAX; 128];
        let mut holder_count = 0;
        let mut closure = [None; 128];
        for (bi, block) in declaration.blocks().iter().enumerate() {
            for (si, statement) in block.statements().iter().enumerate() {
                budget.charge_work(1)?;
                let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                    continue;
                };
                if matches!(assignment.value().kind(), SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Shared, place
                } if place.local() == local && place.projections().is_empty())
                {
                    if !assignment.destination().projections().is_empty()
                        || holder_count == holders.len()
                    {
                        return refuse("retained lifetime shared-holder census");
                    }
                    holders[holder_count] = assignment.destination().local().index();
                    closure[holder_count] = Some((
                        function,
                        assignment.destination().local().index(),
                        Some((SemanticBlockIdV1::from_index(bi as u32), si as u32)),
                    ));
                    holder_count += 1;
                }
            }
        }
        // Inspect direct root holders and their exact original formal0 only.
        // Formal0 may read scalar fields; further whole-reference forwarding
        // fails closed. No checked_call recursion or new authority is issued.
        let mut closure_len = holder_count;
        let mut next = 0;
        while next < closure_len {
            budget.charge_work(4)?;
            let (holder_function, holder, borrow) = closure[next].ok_or(
                ReferenceErrorV1::Unavailable("retained lifetime holder task"),
            )?;
            next += 1;
            match super::require_unwritten_wave64_local_v1(
                self.owner,
                holder_function,
                holder,
                borrow,
                budget,
            ) {
                Ok(()) => {}
                Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(error)) => {
                    return Err(error.into());
                }
                Err(error) => {
                    let kind = if let ProductionSemanticKirErrorV1::Unsupported {
                        function,
                        block: Some(block),
                        statement: Some(statement),
                        ..
                    } = &error
                    {
                        self.owner
                            .source_semantic()
                            .functions()
                            .get(*function as usize)
                            .and_then(|f| f.blocks().get(*block as usize))
                            .and_then(|b| b.statements().get(*statement as usize))
                            .map(|s| match s.kind() {
                                SemanticStatementKindV1::StorageLive(_) => "StorageLive",
                                SemanticStatementKindV1::StorageDead(_) => "StorageDead",
                                SemanticStatementKindV1::Assign(_) => "Assign",
                                _ => "other",
                            })
                            .unwrap_or("absent")
                    } else {
                        "no statement coordinate"
                    };
                    eprintln!(
                        "Wave64 snapshot holder {} in function {}: {:?}; source kind {}",
                        holder,
                        holder_function.index(),
                        error,
                        kind
                    );
                    return refuse(
                        "retained snapshot holder lifetime or source write around storage end",
                    );
                }
            }
            let body = &self.owner.source_semantic().functions()[holder_function.index() as usize];
            let reference_type = body
                .locals()
                .get(holder as usize)
                .ok_or(ReferenceErrorV1::Unavailable(
                    "retained lifetime holder local",
                ))?
                .ty();
            let SemanticTypeShapeV1::Pointer(reference) = self.declaration(reference_type)?.shape()
            else {
                return refuse("retained lifetime holder is not a source reference");
            };
            if reference.kind() != SemanticPointerKindV1::Reference
                || reference.mutability() != SemanticMutabilityV1::Immutable
                || reference.address_space() != 0
                || reference.pointer_width_bits() != 64
                || reference.metadata() != SemanticPointerMetadataV1::None
                || reference.pointee() != declaration.locals()[local.index() as usize].ty()
            {
                return refuse("retained lifetime holder reference shape changed");
            }
            for event in rows(self.owner, holder_function)?.events() {
                budget.charge_work(1)?;
                if !matches!(event.event(), fe2o3_mir_model::SsaEventV1::Use(variable) if variable.get() == holder)
                {
                    continue;
                }
                let allowed = match (event.site(), event.operand()) {
                    (Site::Terminator { block }, OperandRole::CallArgument(0)) => {
                        if borrow.is_none() {
                            return refuse(
                                "retained snapshot formal whole-reference forwarding unavailable",
                            );
                        }
                        let call = call_at(
                            self.owner,
                            holder_function,
                            SemanticBlockIdV1::from_index(block.get()),
                        )
                        .ok_or(ReferenceErrorV1::Unavailable(
                            "retained lifetime holder source call",
                        ))?;
                        let Some(SemanticCallableDeclV1::Defined { function: callee }) = self
                            .owner
                            .source_semantic()
                            .callables()
                            .get(call.callee().index() as usize)
                        else {
                            return refuse(
                                "retained snapshot holder foreign call escape around storage end",
                            );
                        };
                        let callee_body =
                            &self.owner.source_semantic().functions()[callee.index() as usize];
                        let abi = callee_body.abi();
                        if callee_body.role() != SemanticFunctionRoleV1::InternalHelper
                            || callee_body.export().is_some()
                            || abi.source_input_types().len() != 3
                            || abi.source_input_types()[0] != reference_type
                            || abi.source_argument_ownership()[0]
                                != SemanticSourceArgumentOwnershipV1::SharedBorrow
                            || !self.is_f32(abi.source_output_type())
                            || !matches!(call.arguments().first(), Some(SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place))
                                if place.local().index() == holder && place.projections().is_empty() && place.ty() == reference_type)
                        {
                            return refuse(
                                "retained snapshot holder is not an exact shared source formal",
                            );
                        }
                        let mut formal = None;
                        for (index, row) in callee_body.locals().iter().enumerate() {
                            budget.charge_work(1)?;
                            if row.role() == SemanticLocalRoleV1::Argument(0) {
                                if row.ty() != reference_type {
                                    return refuse("retained lifetime formal type changed");
                                }
                                unique(&mut formal, index as u32)?;
                            }
                        }
                        let formal = formal.ok_or(ReferenceErrorV1::Unavailable(
                            "retained lifetime formal absent",
                        ))?;
                        budget.charge_work(closure_len)?;
                        if !closure[..closure_len]
                            .iter()
                            .flatten()
                            .any(|(function, local, _)| function == callee && *local == formal)
                        {
                            if closure_len == closure.len() {
                                return refuse("retained lifetime formal census");
                            }
                            closure[closure_len] = Some((*callee, formal, None));
                            closure_len += 1;
                        }
                        true
                    }
                    (Site::Statement { block, statement }, OperandRole::RvalueOperand(ordinal)) => {
                        let mut allowed = false;
                        let original = body
                            .blocks()
                            .get(block.get() as usize)
                            .and_then(|block| block.statements().get(statement as usize));
                        if let Some(SemanticStatementKindV1::Assign(assignment)) =
                            original.map(SemanticStatementV1::kind)
                        {
                            let mut index = 0u32;
                            assignment.value().kind().try_visit_operands(|operand| {
                                budget.charge_work(1)?;
                                if index == ordinal {
                                    if let SemanticOperandV1::Copy(place)
                                    | SemanticOperandV1::Move(place) = operand
                                    {
                                        allowed = place.local().index() == holder
                                            && super::closed_wave64_scalar_projection_v1(
                                                self.owner, place,
                                            );
                                    }
                                }
                                index += 1;
                                Ok::<(), ReferenceErrorV1>(())
                            })?;
                        }
                        allowed
                    }
                    _ => false,
                };
                if !allowed {
                    return refuse("retained snapshot holder alias or escape around storage end");
                }
            }
        }
        let occurrences = rows(self.owner, function)?;
        // Seed successors only. Reaching dead_block again means earlier
        // statements can execute again after the storage end.
        let mut reachable = [false; 128];
        for edge in occurrences.successors() {
            budget.charge_work(1)?;
            let from = edge.id().source().get() as usize;
            let to = edge.edge().target().index() as usize;
            if from >= declaration.blocks().len() || to >= declaration.blocks().len() {
                return refuse("retained lifetime source edge coordinate");
            }
            if from == dead_block {
                reachable[to] = true;
            }
        }
        for _ in 0..128 {
            let mut changed = false;
            for edge in occurrences.successors() {
                budget.charge_work(1)?;
                let from = edge.id().source().get() as usize;
                let to = edge.edge().target().index() as usize;
                if reachable[from] && !reachable[to] {
                    reachable[to] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for event in occurrences.events() {
            budget.charge_work(1)?;
            let fe2o3_mir_model::SsaEventV1::Use(variable) = event.event() else {
                continue;
            };
            budget.charge_work(holder_count)?;
            let holder = holders[..holder_count].contains(&variable.get());
            if variable.get() != local.index() && !holder {
                continue;
            }
            let (block, after_in_same_block) = match event.site() {
                Site::Statement { block, statement } => {
                    (block.get() as usize, statement as usize > dead_statement)
                }
                Site::Terminator { block } => (block.get() as usize, true),
            };
            if block >= declaration.blocks().len() {
                return refuse("retained lifetime source use coordinate");
            }
            // Count unpromoted and unreachable original events too. Raw/alias
            // escapes remain independently refused by the closed-holder guard.
            if reachable[block] || (block == dead_block && after_in_same_block) {
                return refuse("retained snapshot source use can follow lifetime end");
            }
        }
        Ok(())
    }

    fn retained_snapshot(
        &mut self,
        function: SemanticFunctionIdV1,
        use_site: Site,
        place: &SemanticPlaceV1,
        depth: usize,
        budget: &mut Budget<'_>,
    ) -> ResultV1<Meaning> {
        self.charge(depth, budget)?;
        if function != self.root || !place.projections().is_empty() {
            return refuse("retained snapshot requires the exact root local");
        }
        let declaration = &self.owner.source_semantic().functions()[function.index() as usize];
        let (use_block, use_statement) = match use_site {
            Site::Statement { block, statement } => (block.get() as usize, statement as usize),
            _ => return refuse("retained referent is not an original shared borrow statement"),
        };
        if declaration.blocks().len() > 128 {
            return refuse("retained constructor CFG cap");
        }
        let local = place.local();
        let mut initializer = None;
        for (bi, block) in declaration.blocks().iter().enumerate() {
            for (si, statement) in block.statements().iter().enumerate() {
                budget.charge_work(1)?;
                match statement.kind() {
                    SemanticStatementKindV1::Assign(assignment) => {
                        if assignment.destination().local() == local {
                            if !assignment.destination().projections().is_empty()
                                || !matches!(
                                    assignment.value().kind(),
                                    SemanticRvalueKindV1::Aggregate(_)
                                )
                            {
                                return refuse("retained snapshot has a nonconstructor write");
                            }
                            unique(&mut initializer, (bi, si, assignment.value()))?;
                        }
                        match assignment.value().kind() {
                            SemanticRvalueKindV1::AddressOf { place, .. }
                                if place.local() == local =>
                            {
                                return refuse("retained snapshot has raw address exposure");
                            }
                            SemanticRvalueKindV1::Borrow { kind, place }
                                if place.local() == local =>
                            {
                                if *kind != SemanticBorrowKindV1::Shared
                                    || !place.projections().is_empty()
                                {
                                    return refuse(
                                        "retained snapshot has mutable/projected exposure",
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    SemanticStatementKindV1::StorageDead(dead) if *dead == local => {
                        self.require_snapshot_dead_suffix(function, local, bi, si, budget)?;
                    }
                    SemanticStatementKindV1::Store(store)
                        if store.destination().local() == local =>
                    {
                        return refuse("retained snapshot source store");
                    }
                    SemanticStatementKindV1::Deinitialize(place) if place.local() == local => {
                        return refuse("retained snapshot deinitialization");
                    }
                    SemanticStatementKindV1::SetDiscriminant { place, .. }
                        if place.local() == local =>
                    {
                        return refuse("retained snapshot discriminant write");
                    }
                    SemanticStatementKindV1::AtomicRmw(atomic)
                        if atomic.destination().local() == local
                            || atomic.address().local() == local =>
                    {
                        return refuse("retained snapshot atomic write");
                    }
                    SemanticStatementKindV1::AtomicCompareExchange(atomic)
                        if atomic.destination().local() == local
                            || atomic.address().local() == local =>
                    {
                        return refuse("retained snapshot atomic compare exchange");
                    }
                    _ => {}
                }
            }
            if let SemanticTerminatorKindV1::Call(call) = block.terminator().kind()
                && call
                    .destination()
                    .is_some_and(|d| d.place().local() == local)
            {
                return refuse("retained snapshot is overwritten by a call");
            }
        }
        let (constructor_block, constructor_statement, value) = initializer.ok_or(
            ReferenceErrorV1::Unavailable("retained source constructor absent"),
        )?;
        // StorageLive starts a fresh allocation lifetime. Only initial entry
        // lives or same-block lives before this unique constructor may precede
        // the already-required constructor domination of every borrow.
        for (bi, block) in declaration.blocks().iter().enumerate() {
            for (si, statement) in block.statements().iter().enumerate() {
                budget.charge_work(1)?;
                if matches!(statement.kind(), SemanticStatementKindV1::StorageLive(live) if *live == local)
                    && !((bi == constructor_block && si < constructor_statement)
                        || (bi == declaration.entry().index() as usize && bi != constructor_block))
                {
                    return refuse("retained snapshot allocation lifetime restarted");
                }
            }
        }
        if constructor_block == use_block {
            if constructor_statement >= use_statement {
                return refuse("constructor does not precede shared borrow");
            }
        } else {
            // A reachability search with the constructor block removed proves
            // dominance; the source successor inventory is the owner's capture.
            let mut reachable = [false; 128];
            let entry = declaration.entry().index() as usize;
            if entry != constructor_block {
                reachable[entry] = true;
            }
            for _ in 0..128 {
                let mut changed = false;
                for edge in rows(self.owner, function)?.successors() {
                    budget.charge_work(1)?;
                    let from = edge.id().source().get() as usize;
                    let to = edge.edge().target().index() as usize;
                    if from >= 128 || to >= 128 {
                        return refuse("retained source CFG coordinate");
                    }
                    if reachable[from] && to != constructor_block && !reachable[to] {
                        reachable[to] = true;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            if reachable[use_block] {
                return refuse("constructor does not dominate shared borrow");
            }
        }
        let meaning = self.rvalue(
            function,
            Site::Statement {
                block: fe2o3_mir_model::SsaBlockIdV1::new(constructor_block as u32),
                statement: constructor_statement as u32,
            },
            value.result_type(),
            value.kind(),
            depth + 1,
            budget,
        )?;
        if !matches!(meaning.kind, Kind::Snapshot { .. }) || meaning.ty != place.ty() {
            return refuse("retained referent is not the original arithmetic snapshot");
        }
        Ok(meaning)
    }

    // A borrowed zero-state context remains address-taken source storage. Its
    // exact root intrinsic destination is an issuer, never an invented SSA key.
    fn retained_context(
        &mut self,
        function: SemanticFunctionIdV1,
        use_site: Site,
        place: &SemanticPlaceV1,
        depth: usize,
        budget: &mut Budget<'_>,
    ) -> ResultV1<Meaning> {
        self.charge(depth, budget)?;
        if function != self.root || !place.projections().is_empty() || !self.zst(place.ty()) {
            return refuse("retained context requires exact zero-state root local");
        }
        let declaration = &self.owner.source_semantic().functions()[function.index() as usize];
        let Site::Statement {
            block: use_block, ..
        } = use_site
        else {
            return refuse("retained context must be original shared borrow statement");
        };
        if declaration.blocks().len() > 128 {
            return refuse("retained context CFG cap");
        }
        let local = place.local();
        let mut issuer = None;
        for (bi, block) in declaration.blocks().iter().enumerate() {
            for statement in block.statements() {
                budget.charge_work(1)?;
                match statement.kind() {
                    SemanticStatementKindV1::Assign(assignment) => {
                        if assignment.destination().local() == local {
                            return refuse("retained context has a source assignment");
                        }
                        match assignment.value().kind() {
                            SemanticRvalueKindV1::AddressOf { place, .. }
                                if place.local() == local =>
                            {
                                return refuse("retained context has raw address exposure");
                            }
                            SemanticRvalueKindV1::Borrow { kind, place }
                                if place.local() == local =>
                            {
                                if *kind != SemanticBorrowKindV1::Shared
                                    || !place.projections().is_empty()
                                {
                                    return refuse(
                                        "retained context has mutable/projected exposure",
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    SemanticStatementKindV1::StorageDead(dead) if *dead == local => {
                        return refuse("retained context lifetime ends in source");
                    }
                    SemanticStatementKindV1::Store(store)
                        if store.destination().local() == local =>
                    {
                        return refuse("retained context source store");
                    }
                    SemanticStatementKindV1::Deinitialize(place) if place.local() == local => {
                        return refuse("retained context deinitialization");
                    }
                    SemanticStatementKindV1::SetDiscriminant { place, .. }
                        if place.local() == local =>
                    {
                        return refuse("retained context discriminant write");
                    }
                    SemanticStatementKindV1::AtomicRmw(atomic)
                        if atomic.destination().local() == local
                            || atomic.address().local() == local =>
                    {
                        return refuse("retained context atomic write");
                    }
                    SemanticStatementKindV1::AtomicCompareExchange(atomic)
                        if atomic.destination().local() == local
                            || atomic.address().local() == local =>
                    {
                        return refuse("retained context atomic compare exchange");
                    }
                    _ => {}
                }
            }
            let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
                continue;
            };
            let Some(destination) = call.destination().filter(|d| d.place().local() == local)
            else {
                continue;
            };
            if !destination.place().projections().is_empty()
                || destination.place().ty() != place.ty()
                || !call.arguments().is_empty()
                || !matches!(call.unwind(), SemanticUnwindActionV1::Unreachable)
                || !matches!(self.owner.source_semantic().callables().get(call.callee().index() as usize),
                    Some(SemanticCallableDeclV1::CompilerIntrinsic {
                        operation: SemanticCompilerIntrinsicOperationV1::CollectiveContextCurrent { context }, ..
                    }) if *context == place.ty())
            {
                return refuse("retained context has nonissuer call write");
            }
            let producer = SemanticBlockIdV1::from_index(bi as u32);
            // The captured source edge must explicitly be reachable, retained,
            // valueless and the exact normal return destination of this call.
            let mut original_edge = None;
            for edge in rows(self.owner, function)?.edge_definitions() {
                budget.charge_work(1)?;
                if edge.edge().source().get() == producer.index()
                    && edge.variable().get() == local.index()
                {
                    if edge.edge().ordinal() != 0
                        || edge.ordinal() != 0
                        || !edge.is_reachable()
                        || edge.is_promoted()
                        || edge.value().is_some()
                    {
                        return refuse("retained context original call edge mismatch");
                    }
                    unique(&mut original_edge, ())?;
                }
            }
            if original_edge.is_none() {
                return refuse("retained context original edge absent");
            }
            unique(&mut issuer, producer)?;
        }
        let producer = issuer.ok_or(ReferenceErrorV1::Unavailable(
            "retained context exact issuer absent",
        ))?;
        // In the issuer block the call terminator follows every statement.
        // Entry lives also precede any path to this dominating call. All other
        // lifetime resets fail closed; none inherit the previous origin.
        for (bi, block) in declaration.blocks().iter().enumerate() {
            for statement in block.statements() {
                budget.charge_work(1)?;
                if matches!(statement.kind(), SemanticStatementKindV1::StorageLive(live) if *live == local)
                    && bi != producer.index() as usize
                    && bi != declaration.entry().index() as usize
                {
                    return refuse("retained context allocation lifetime restarted");
                }
            }
        }
        if producer.index() == use_block.get() {
            return refuse("context call cannot precede statement in its own terminator block");
        }
        // Removing the actual call block must disconnect the borrow from entry.
        let mut reachable = [false; 128];
        let entry = declaration.entry().index() as usize;
        if entry != producer.index() as usize {
            reachable[entry] = true;
        }
        for _ in 0..128 {
            let mut changed = false;
            for edge in rows(self.owner, function)?.successors() {
                budget.charge_work(1)?;
                let from = edge.id().source().get() as usize;
                let to = edge.edge().target().index() as usize;
                if from >= 128 || to >= 128 {
                    return refuse("retained context CFG coordinate");
                }
                if reachable[from] && to != producer.index() as usize && !reachable[to] {
                    reachable[to] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if reachable[use_block.get() as usize] {
            return refuse("context issuer does not dominate shared borrow");
        }
        Ok(Meaning {
            kind: Kind::Context,
            ty: place.ty(),
            referent: None,
            origin: SourceOrigin::RootRetainedCall {
                function,
                block: producer,
                local,
                context_type: place.ty(),
            },
        })
    }
}

impl<'a> Resolver<'a> {
    fn charge(&mut self, depth: usize, budget: &mut Budget<'_>) -> ResultV1<()> {
        budget.charge_work(8)?;
        if depth >= 64 || self.visits >= 4096 {
            return refuse("reference ancestry depth/work cap");
        }
        self.visits += 1;
        Ok(())
    }
    fn shared(&self, mut meaning: Meaning, reference: SemanticTypeIdV1) -> ResultV1<Meaning> {
        let SemanticTypeShapeV1::Pointer(pointer) = self.declaration(reference)?.shape() else {
            return refuse("shared source borrow has no reference type");
        };
        if meaning.referent.is_some()
            || pointer.kind() != SemanticPointerKindV1::Reference
            || pointer.mutability() != SemanticMutabilityV1::Immutable
            || pointer.metadata() != SemanticPointerMetadataV1::None
            || pointer.address_space() != 0
            || pointer.pointer_width_bits() != 64
            || pointer.pointee() != meaning.ty
            || !matches!(
                meaning.kind,
                Kind::Context | Kind::Snapshot { .. } | Kind::LaneCarrier
            )
        {
            return refuse("borrow is not exact immutable reference to original referent");
        }
        meaning.referent = Some(meaning.ty);
        meaning.ty = reference;
        Ok(meaning)
    }
    fn project(&self, mut meaning: Meaning, place: &SemanticPlaceV1) -> ResultV1<Meaning> {
        for projection in place.projections() {
            match projection.kind() {
                SemanticProjectionKindV1::Dereference => {
                    let referent = meaning
                        .referent
                        .take()
                        .ok_or(ReferenceErrorV1::Unavailable(
                            "dereference lacks original shared borrow",
                        ))?;
                    if referent != projection.result_type() {
                        return refuse("dereference source type changed");
                    }
                    meaning.ty = referent;
                }
                SemanticProjectionKindV1::Field(field) if meaning.referent.is_none() => {
                    let SemanticTypeShapeV1::Aggregate(fields) =
                        self.declaration(meaning.ty)?.shape()
                    else {
                        return refuse("field source is not original aggregate");
                    };
                    let field_ty = *fields
                        .fields()
                        .get(field as usize)
                        .ok_or(ReferenceErrorV1::Unavailable("source field out of range"))?;
                    if field_ty != projection.result_type() {
                        return refuse("source field type changed");
                    }
                    meaning.kind = match meaning.kind {
                        Kind::Snapshot { lane_field }
                            if field == lane_field && self.is_u32(field_ty) =>
                        {
                            Kind::LaneValue
                        }
                        Kind::LaneCarrier if self.exact_lane_field(meaning.ty, field)? => {
                            Kind::LaneValue
                        }
                        _ if self.zst(field_ty) => Kind::Zst,
                        _ => return refuse("field does not carry the original physical lane"),
                    };
                    meaning.ty = field_ty;
                }
                _ => return refuse("raw, mutable, indexed or cast reference projection"),
            }
        }
        if meaning.ty != place.ty() {
            return refuse("final source projection type changed");
        }
        Ok(meaning)
    }
    fn operand(
        &mut self,
        function: SemanticFunctionIdV1,
        site: Site,
        role: OperandRole,
        operand: &SemanticOperandV1,
        depth: usize,
        budget: &mut Budget<'_>,
    ) -> ResultV1<Meaning> {
        match operand {
            SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place) => {
                let value = base_use(self.owner, function, site, role, budget)?;
                let meaning = self.resolve(function, value, depth + 1, budget)?;
                self.project(meaning, place)
            }
            SemanticOperandV1::Constant(_) if self.zst(operand.ty()) => Ok(Meaning {
                kind: Kind::Zst,
                ty: operand.ty(),
                referent: None,
                origin: SourceOrigin::ZeroState,
            }),
            _ => refuse("constant cannot manufacture lane or context ancestry"),
        }
    }
    fn resolve(
        &mut self,
        function: SemanticFunctionIdV1,
        value: SsaValueV1,
        depth: usize,
        budget: &mut Budget<'_>,
    ) -> ResultV1<Meaning> {
        self.charge(depth, budget)?;
        let declaration = self
            .owner
            .source_semantic()
            .functions()
            .get(function.index() as usize)
            .ok_or(ReferenceErrorV1::Unavailable("source function coordinate"))?;
        // Only actual compiler operations at the original root seed authority.
        if function == self.root {
            for (index, block) in declaration.blocks().iter().enumerate() {
                budget.charge_work(1)?;
                let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
                    continue;
                };
                let Some(SemanticCallableDeclV1::CompilerIntrinsic { operation, .. }) = self
                    .owner
                    .source_semantic()
                    .callables()
                    .get(call.callee().index() as usize)
                else {
                    continue;
                };
                let (kind, ty) = match operation {
                    SemanticCompilerIntrinsicOperationV1::WaveLaneCurrent {
                        lane,
                        wave_width: 64,
                    } => (Kind::LaneCarrier, *lane),
                    SemanticCompilerIntrinsicOperationV1::CollectiveContextCurrent { context } => {
                        (Kind::Context, *context)
                    }
                    _ => continue,
                };
                let block_id = SemanticBlockIdV1::from_index(index as u32);
                if result_value(self.owner, function, block_id, budget)? == Some(value) {
                    if !call.arguments().is_empty()
                        || call.destination().is_none_or(|d| d.place().ty() != ty)
                    {
                        return refuse("compiler producer source type/arity changed");
                    }
                    return Ok(Meaning {
                        kind,
                        ty,
                        referent: None,
                        origin: SourceOrigin::RootIssuedSsa { function, value },
                    });
                }
            }
        }
        let occurrences = rows(self.owner, function)?;
        // Formals are not issuers. Resolve EVERY original incoming call back to
        // the actual root producer; foreign origins and recursion fail closed.
        for entry in occurrences.entry_definitions() {
            budget.charge_work(1)?;
            if entry.value() != Some(value) {
                continue;
            }
            let EntryOrigin::Argument(argument) = entry.origin() else {
                return refuse("non-positional reference formal");
            };
            if function == self.root {
                return refuse("kernel argument cannot issue context/lane");
            }
            let mut inherited = None;
            for (fi, caller) in self.owner.source_semantic().functions().iter().enumerate() {
                for (bi, block) in caller.blocks().iter().enumerate() {
                    budget.charge_work(1)?;
                    let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
                        continue;
                    };
                    if !matches!(self.owner.source_semantic().callables().get(call.callee().index() as usize),
                        Some(SemanticCallableDeclV1::Defined { function: callee }) if *callee == function)
                    {
                        continue;
                    }
                    let operand = call.arguments().get(argument as usize).ok_or(
                        ReferenceErrorV1::Unavailable("actual caller argument absent"),
                    )?;
                    let meaning = self.operand(
                        SemanticFunctionIdV1::from_index(fi as u32),
                        site(SemanticBlockIdV1::from_index(bi as u32)),
                        OperandRole::CallArgument(argument),
                        operand,
                        depth + 1,
                        budget,
                    )?;
                    if declaration
                        .abi()
                        .source_input_types()
                        .get(argument as usize)
                        != Some(&meaning.ty)
                    {
                        return refuse("formal/caller source type mismatch");
                    }
                    match inherited {
                        Some(previous) if previous != meaning => {
                            return refuse("distinct source origins join at formal");
                        }
                        None => inherited = Some(meaning),
                        _ => {}
                    }
                }
            }
            return inherited.ok_or(ReferenceErrorV1::Unavailable(
                "formal has no original caller",
            ));
        }
        match value {
            SsaValueV1::BlockArgument { block, variable } => {
                let plan = self
                    .owner
                    .plan_for_function(function)
                    .ok_or(ReferenceErrorV1::Unavailable("actual SSA plan"))?;
                let mut inherited = None;
                for successor in occurrences.successors() {
                    budget.charge_work(1)?;
                    if successor.edge().target().index() != block.get() {
                        continue;
                    }
                    let arguments = plan
                        .plan()
                        .edge_arguments(successor.id())
                        .ok_or(ReferenceErrorV1::Unavailable("actual SSA edge arguments"))?;
                    for argument in arguments {
                        budget.charge_work(1)?;
                        if argument.variable() != variable {
                            continue;
                        }
                        if argument.value() == value {
                            continue;
                        }
                        let meaning =
                            self.resolve(function, argument.value(), depth + 1, budget)?;
                        match inherited {
                            Some(previous) if previous != meaning => {
                                return refuse("distinct source origins join at block");
                            }
                            None => inherited = Some(meaning),
                            _ => {}
                        }
                    }
                }
                inherited.ok_or(ReferenceErrorV1::Unavailable(
                    "SSA block argument has no origin",
                ))
            }
            SsaValueV1::Definition(_) => {
                let mut definition = None;
                for event in occurrences.events() {
                    budget.charge_work(1)?;
                    if matches!(event.resolved(), Some(SsaResolvedEventV1::Define { value: actual, .. }) if actual == value)
                    {
                        if !event.is_reachable() || !event.is_promoted() {
                            return refuse("unavailable source definition");
                        }
                        unique(&mut definition, event.site())?;
                    }
                }
                let Some(site @ Site::Statement { block, statement }) = definition else {
                    return refuse("non-issuer source call result is not reference ancestry");
                };
                let statement = declaration
                    .blocks()
                    .get(block.get() as usize)
                    .and_then(|b| b.statements().get(statement as usize))
                    .ok_or(ReferenceErrorV1::Unavailable("source statement coordinate"))?;
                let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                    return refuse("source definition is not assignment");
                };
                if !assignment.destination().projections().is_empty() {
                    return refuse("projected source definition");
                }
                self.rvalue(
                    function,
                    site,
                    assignment.value().result_type(),
                    assignment.value().kind(),
                    depth + 1,
                    budget,
                )
            }
        }
    }
    fn rvalue(
        &mut self,
        function: SemanticFunctionIdV1,
        site: Site,
        ty: SemanticTypeIdV1,
        rvalue: &SemanticRvalueKindV1,
        depth: usize,
        budget: &mut Budget<'_>,
    ) -> ResultV1<Meaning> {
        self.charge(depth, budget)?;
        let meaning = match rvalue {
            SemanticRvalueKindV1::Use(operand) => self.operand(
                function,
                site,
                OperandRole::RvalueOperand(0),
                operand,
                depth + 1,
                budget,
            )?,
            SemanticRvalueKindV1::Borrow {
                kind: SemanticBorrowKindV1::Shared,
                place,
            } => {
                // Elision is not storage classification: a normal retained
                // BaseUse remains in the captured syntax with no SSA resolution.
                let occurrences = rows(self.owner, function)?;
                let mut retained = occurrences.elisions().contains(&site);
                if !retained {
                    let mut original_place = None;
                    for event in occurrences.events() {
                        budget.charge_work(1)?;
                        if event.site() != site
                            || event.operand() != OperandRole::RvaluePlace
                            || event.role() != EventRole::BaseUse
                        {
                            continue;
                        }
                        if !matches!(event.event(), fe2o3_mir_model::SsaEventV1::Use(variable)
                            if variable.get() == place.local().index())
                            || !event.is_reachable()
                        {
                            return refuse("original shared borrow place occurrence mismatch");
                        }
                        if event.is_promoted() {
                            if !matches!(event.resolved(), Some(SsaResolvedEventV1::Use { variable, .. })
                                if variable.get() == place.local().index())
                            {
                                return refuse(
                                    "promoted shared borrow place lacks original SSA use",
                                );
                            }
                            unique(&mut original_place, false)?;
                        } else {
                            if event.resolved().is_some()
                                || function != self.root
                                || !place.projections().is_empty()
                            {
                                return refuse(
                                    "retained shared borrow lacks exact root-local source storage",
                                );
                            }
                            unique(&mut original_place, true)?;
                        }
                    }
                    retained = original_place.ok_or(ReferenceErrorV1::Unavailable(
                        "original shared borrow place occurrence absent",
                    ))?;
                }
                let meaning = if retained {
                    if self.zst(place.ty()) {
                        self.retained_context(function, site, place, depth + 1, budget)?
                    } else {
                        self.retained_snapshot(function, site, place, depth + 1, budget)?
                    }
                } else {
                    let value =
                        base_use(self.owner, function, site, OperandRole::RvaluePlace, budget)?;
                    let meaning = self.resolve(function, value, depth + 1, budget)?;
                    self.project(meaning, place)?
                };
                self.shared(meaning, ty)?
            }

            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Integer,
                operand,
            } => {
                let mut meaning = self.operand(
                    function,
                    site,
                    OperandRole::RvalueOperand(0),
                    operand,
                    depth + 1,
                    budget,
                )?;
                if meaning.kind != Kind::LaneValue
                    || meaning.referent.is_some()
                    || !matches!(
                        self.declaration(meaning.ty)?.shape(),
                        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                            signed: false,
                            bits: 32 | 64
                        })
                    )
                    || !matches!(
                        self.declaration(ty)?.shape(),
                        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                            signed: false,
                            bits: 32 | 64
                        })
                    )
                {
                    return refuse("integer cast is not identity transport of original lane");
                }
                meaning.ty = ty;
                meaning
            }
            SemanticRvalueKindV1::Binary {
                operation: SemanticBinaryOpV1::Remainder,
                left,
                right,
            } => {
                let meaning = self.operand(
                    function,
                    site,
                    OperandRole::RvalueOperand(0),
                    left,
                    depth + 1,
                    budget,
                )?;
                if meaning.kind != Kind::LaneValue
                    || meaning.referent.is_some()
                    || meaning.ty != ty
                    || exact_unsigned_semantic_constant_v1(
                        self.owner.source_semantic().types(),
                        right,
                        ty,
                    ) != Some(64)
                {
                    return refuse("rank remainder is not exact identity modulo Wave64");
                }
                meaning
            }
            SemanticRvalueKindV1::Aggregate(aggregate)
                if matches!(aggregate.kind(), SemanticAggregateKindV1::Aggregate) =>
            {
                let SemanticTypeShapeV1::Aggregate(fields) = self.declaration(ty)?.shape() else {
                    return refuse("source snapshot constructor is not aggregate");
                };
                if aggregate.operands().len() != fields.fields().len()
                    || aggregate.operands().len() > 8
                {
                    return refuse("source snapshot field census changed");
                }
                let mut lane = None;
                for (index, operand) in aggregate.operands().iter().enumerate() {
                    budget.charge_work(1)?;
                    let expected = fields.fields()[index];
                    if operand.ty() != expected {
                        return refuse("source snapshot field type mismatch");
                    }
                    let meaning = self.operand(
                        function,
                        site,
                        OperandRole::RvalueOperand(index as u32),
                        operand,
                        depth + 1,
                        budget,
                    )?;
                    if meaning.kind == Kind::LaneValue
                        && meaning.referent.is_none()
                        && self.is_u32(expected)
                    {
                        unique(&mut lane, (index as u32, meaning))?;
                    } else if meaning.kind != Kind::Zst || !self.zst(expected) {
                        return refuse(
                            "snapshot contains state without original current-lane ancestry",
                        );
                    }
                }
                let (lane_field, mut lane) = lane.ok_or(ReferenceErrorV1::Unavailable(
                    "snapshot has no issued lane component",
                ))?;
                lane.kind = Kind::Snapshot { lane_field };
                lane.ty = ty;
                lane
            }
            _ => return refuse("computed, mutable, raw or reconstructed reference ancestry"),
        };
        if meaning.ty != ty {
            return refuse("source assignment type mismatch");
        }
        Ok(meaning)
    }
}

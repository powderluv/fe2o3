use super::*;

pub(super) enum Output<'a> {
    Count,
    Fill(&'a mut Vec<u8>),
    Compare(&'a [u8]),
}

pub(super) fn frame(
    input: &NativeCpuInputV1<'_>,
    alternate: bool,
    length: usize,
    output: Output<'_>,
    s: &mut Scope<'_, '_>,
) -> Result<usize, Error> {
    subject_frame(
        &input.subject(),
        input.origin_v2(),
        alternate,
        length,
        output,
        s,
    )
}

pub(super) fn subject_frame(
    input: &CpuSubject<'_>,
    origin: NativeCpuOriginV2<'_>,
    alternate: bool,
    length: usize,
    output: Output<'_>,
    s: &mut Scope<'_, '_>,
) -> Result<usize, Error> {
    let signature = input.replay.signature_preimage;
    let axes = signature
        .reference_inputs()
        .len()
        .checked_sub(signature.kernel_inputs().len())
        .ok_or(Error::Wire("signature arity"))?;
    require(axes <= MAX_REFERENCE_POINT_AXES_V1, "point axes")?;
    let ir = input.replay.effect_ir;
    require(
        ir.local_count > ir.argument_count && u64::from(ir.local_count) <= HARD_MAX_LOCALS_V1,
        "local count",
    )?;
    require(!ir.blocks.is_empty(), "empty blocks")?;
    require(ir.loop_summaries.is_empty(), "loop summaries unsupported")?;
    require(
        u64::from(input.semantic_root) < HARD_MAX_FUNCTIONS_V1,
        "semantic root",
    )?;
    let mut w = Writer {
        output,
        position: 0,
        s,
        ir,
        axes,
        kernel_count: signature.kernel_inputs().len(),
        nodes: 0,
    };
    let (magic, version) = match origin {
        NativeCpuOriginV2::SourceRegistration { .. } => (MAGIC, 1u16),
        NativeCpuOriginV2::AdmittedPolicy(_) => (policy_v2::POLICY_MAGIC, 2u16),
    };
    w.put(magic)?;
    w.put(&version.to_le_bytes())?;
    w.put(&0u16.to_le_bytes())?;
    w.u32(u32::try_from(length).map_err(|_| Resource::Arithmetic)?)?;
    w.u8(64)?;
    w.put(&input.semantic_mir_sha256)?;
    w.u32(input.semantic_root)?;
    match origin {
        NativeCpuOriginV2::SourceRegistration { registration_path } => {
            w.text(registration_path)?;
        }
        NativeCpuOriginV2::AdmittedPolicy(origin) => {
            require(
                origin.mapping_ordinal < MAX_REFERENCE_ENROLLMENT_BINDINGS_V1,
                "mapping ordinal",
            )?;
            w.u8(policy_v2::ADMITTED_POLICY_TAG)?;
            w.put(&policy_v2::REFERENCE_ENROLLMENT_DOMAIN_V1.to_le_bytes())?;
            w.put(&origin.rustc_invocation_sha256)?;
            w.put(&origin.native_policy_sha256)?;
            w.u64(origin.policy_generation)?;
            w.u32(origin.mapping_ordinal)?;
        }
    }
    w.text(input.logical_kernel_name)?;
    w.identity(input.kernel)?;
    w.identity(input.reference)?;
    w.put(&[0; 4])?;
    for inputs in [signature.kernel_inputs(), signature.reference_inputs()] {
        w.count(inputs.len(), MAX_REFERENCE_SIGNATURE_INPUTS_V1)?;
        for input in inputs {
            w.signature_input(input)?;
        }
    }
    w.put(&input.replay.effect_ir_sha256)?;
    w.u32(ir.argument_count)?;
    w.u32(ir.local_count)?;
    w.count(ir.relations.len(), MAX_REFERENCE_SIGNATURE_INPUTS_V1)?;
    for relation in &ir.relations {
        w.relation(relation)?;
    }
    w.count(ir.blocks.len(), MAX_REFERENCE_BLOCKS_V1)?;
    let mut statements = 0;
    for (index, block) in ir.blocks.iter().enumerate() {
        require(block.block as usize == index, "block ordinal")?;
        w.u32(block.block)?;
        statements = add(statements, block.assignments.len())?;
        require(statements <= MAX_REFERENCE_STATEMENTS_V1, "statement limit")?;
        w.count(block.assignments.len(), MAX_REFERENCE_STATEMENTS_V1)?;
        let mut previous = None;
        for assignment in &block.assignments {
            require(
                (assignment.statement as usize) < MAX_REFERENCE_STATEMENTS_V1
                    && previous.is_none_or(|p| p < assignment.statement),
                "statement ordinal",
            )?;
            previous = Some(assignment.statement);
            w.u32(assignment.statement)?;
            w.place(&assignment.destination)?;
            w.value(&assignment.value)?;
        }
        w.terminator(&block.terminator)?;
    }
    w.u32(0)?;
    let effects = if alternate {
        input.replay.observable_output_writes
    } else {
        &ir.observable_output_effects
    };
    w.count(effects.len(), MAX_REFERENCE_STATEMENTS_V1)?;
    let mut previous = None;
    for effect in effects {
        let key = (effect.argument, effect.block, effect.statement);
        require(previous.is_none_or(|p| p < key), "effect occurrence order")?;
        previous = Some(key);
        w.effect(effect)?;
    }
    require(
        length == 0 || w.position == length,
        "frame length or effect-list mismatch",
    )?;
    Ok(w.position)
}

struct Writer<'a, 's, 'b, 'w> {
    output: Output<'a>,
    position: usize,
    s: &'s mut Scope<'b, 'w>,
    ir: &'a ReferenceEffectIrV1,
    axes: usize,
    kernel_count: usize,
    nodes: usize,
}
impl Writer<'_, '_, '_, '_> {
    fn put(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let end = add(self.position, bytes.len())?;
        require(end <= MAX_NATIVE_CPU_INPUT_BYTES_V1, "frame limit")?;
        self.s.work(add(bytes.len(), 1)?)?;
        match &mut self.output {
            Output::Count => (),
            Output::Fill(out) => {
                if end > out.capacity() {
                    return Err(Resource::Accounting.into());
                }
                out.extend_from_slice(bytes);
            }
            Output::Compare(expected) => require(
                expected.get(self.position..end) == Some(bytes),
                "noncanonical bytes or retained effects differ",
            )?,
        }
        self.position = end;
        Ok(())
    }
    fn u8(&mut self, value: u8) -> Result<(), Error> {
        self.put(&[value])
    }
    fn u32(&mut self, value: u32) -> Result<(), Error> {
        self.put(&value.to_le_bytes())
    }
    fn u64(&mut self, value: u64) -> Result<(), Error> {
        self.put(&value.to_le_bytes())
    }
    fn u128(&mut self, value: u128) -> Result<(), Error> {
        self.put(&value.to_le_bytes())
    }
    fn boolean(&mut self, value: bool) -> Result<(), Error> {
        self.u8(u8::from(value))
    }
    fn count(&mut self, count: usize, max: usize) -> Result<(), Error> {
        require(count <= max, "count limit")?;
        self.u32(u32::try_from(count).map_err(|_| Resource::Arithmetic)?)
    }
    fn text(&mut self, text: &str) -> Result<(), Error> {
        self.count(text.len(), MAX_NATIVE_CPU_INPUT_BYTES_V1)?;
        self.put(text.as_bytes())
    }
    fn scalar(&mut self, scalar: ReferenceScalarTypeV1) -> Result<(), Error> {
        self.enum_tag(SCALARS, &scalar)
    }
    fn enum_tag<T: PartialEq>(&mut self, table: &[(u8, T)], value: &T) -> Result<(), Error> {
        self.s.work(table.len())?;
        self.u8(tag(table, value)?)
    }
    fn identity(&mut self, identity: &ReferenceFunctionIdentityV1) -> Result<(), Error> {
        self.put(&identity.def_path_hash)?;
        for field in [
            &identity.function_sha256,
            &identity.item_definition_sha256,
            &identity.monomorphization_sha256,
            &identity.generic_type_arguments_sha256,
            &identity.const_generic_arguments_sha256,
            &identity.rustc_mir_body_sha256,
        ] {
            self.put(field)?;
        }
        Ok(())
    }
    fn signature_input(&mut self, input: &ReferenceSignatureInputV1) -> Result<(), Error> {
        match input {
            ReferenceSignatureInputV1::Scalar(scalar) => {
                self.u8(0)?;
                self.scalar(*scalar)
            }
            ReferenceSignatureInputV1::Reference {
                region,
                mutability,
                pointee,
            } => {
                self.u8(1)?;
                self.u8(match region {
                    ReferenceRegionV1::Erased => 0,
                    ReferenceRegionV1::Static => 1,
                })?;
                self.u8(match mutability {
                    SemanticMutabilityV1::Immutable => 0,
                    SemanticMutabilityV1::Mutable => 1,
                })?;
                match pointee {
                    ReferencePointeeV1::Scalar(scalar) => {
                        self.u8(0)?;
                        self.scalar(*scalar)
                    }
                    ReferencePointeeV1::Slice(scalar) => {
                        self.u8(1)?;
                        self.scalar(*scalar)
                    }
                }
            }
            ReferenceSignatureInputV1::NominalOutput { carrier, element } => {
                self.u8(2)?;
                self.u8(match carrier {
                    ReferenceCarrierV1::DisjointSlice => 0,
                    ReferenceCarrierV1::WriteOnlyDisjointSlice => 1,
                })?;
                self.scalar(*element)
            }
        }
    }
    fn relation(&mut self, relation: &ReferenceArgumentRelationV1) -> Result<(), Error> {
        let (tag, argument, scalar) = match relation {
            ReferenceArgumentRelationV1::ScalarInput { argument, scalar } => (0, argument, scalar),
            ReferenceArgumentRelationV1::SharedSliceInput { argument, element } => {
                (1, argument, element)
            }
            ReferenceArgumentRelationV1::DisjointOutputSlice { argument, element } => {
                (2, argument, element)
            }
            ReferenceArgumentRelationV1::DisjointOutputCoordinate { argument, element } => {
                (3, argument, element)
            }
            ReferenceArgumentRelationV1::PointCoordinate {
                reference_argument,
                axis,
            } => {
                self.u8(4)?;
                self.u32(*reference_argument)?;
                return self.u32(*axis);
            }
        };
        self.u8(tag)?;
        self.u32(*argument)?;
        self.scalar(*scalar)
    }
    fn constant_index(
        &mut self,
        offset: u64,
        minimum_length: u64,
        from_end: bool,
    ) -> Result<(), Error> {
        require(
            if from_end {
                offset > 0 && offset <= minimum_length
            } else {
                offset < minimum_length
            },
            "constant index",
        )?;
        self.u64(offset)?;
        self.u64(minimum_length)?;
        self.boolean(from_end)
    }
    fn place(&mut self, place: &ReferencePlaceV1) -> Result<(), Error> {
        require(place.local < self.ir.local_count, "place local")?;
        self.u32(place.local)?;
        self.count(place.projection.len(), DEPTH)?;
        for projection in &place.projection {
            match projection {
                ReferencePlaceProjectionV1::Dereference => self.u8(0)?,
                ReferencePlaceProjectionV1::Field(field) => {
                    require(*field <= 1, "checked-pair field")?;
                    self.u8(1)?;
                    self.u32(*field)?;
                }
                ReferencePlaceProjectionV1::Index(local) => {
                    require(*local < self.ir.local_count, "index local")?;
                    self.u8(2)?;
                    self.u32(*local)?;
                }
                ReferencePlaceProjectionV1::ConstantIndex {
                    offset,
                    minimum_length,
                    from_end,
                } => {
                    self.u8(3)?;
                    self.constant_index(*offset, *minimum_length, *from_end)?;
                }
            }
        }
        Ok(())
    }
    fn constant(&mut self, constant: &ReferenceConstantV1) -> Result<(), Error> {
        match constant {
            ReferenceConstantV1::ZeroSized => self.u8(0),
            ReferenceConstantV1::Scalar { scalar, bits } => {
                scalar_bits(*scalar, *bits)?;
                self.u8(1)?;
                self.scalar(*scalar)?;
                self.u128(*bits)
            }
        }
    }
    fn operand(&mut self, operand: &ReferenceOperandV1) -> Result<(), Error> {
        match operand {
            ReferenceOperandV1::Copy(place) => {
                self.u8(0)?;
                self.place(place)
            }
            ReferenceOperandV1::Move(place) => {
                self.u8(1)?;
                self.place(place)
            }
            ReferenceOperandV1::Constant(constant) => {
                self.u8(2)?;
                self.constant(constant)
            }
        }
    }
    fn raw_slice(&self, raw: u32, load: bool) -> Result<(), Error> {
        require(
            match self.ir.relations.get(raw as usize) {
                Some(ReferenceArgumentRelationV1::SharedSliceInput { .. }) => true,
                Some(ReferenceArgumentRelationV1::DisjointOutputSlice { .. }) => !load,
                _ => false,
            },
            "slice raw argument",
        )
    }
    fn value(&mut self, value: &ReferenceValueV1) -> Result<(), Error> {
        match value {
            ReferenceValueV1::Use(operand) => {
                self.u8(0)?;
                self.operand(operand)
            }
            ReferenceValueV1::Binary {
                operation,
                lhs,
                rhs,
                checked,
            } => {
                self.u8(1)?;
                self.enum_tag(BINARY, operation)?;
                self.boolean(*checked)?;
                self.operand(lhs)?;
                self.operand(rhs)
            }
            ReferenceValueV1::Unary { operation, operand } => {
                self.u8(2)?;
                self.enum_tag(UNARY, operation)?;
                self.operand(operand)
            }
            ReferenceValueV1::Cast {
                kind,
                source,
                target,
                operand,
            } => {
                self.u8(3)?;
                self.enum_tag(CASTS, kind)?;
                self.scalar(*source)?;
                self.scalar(*target)?;
                self.operand(operand)
            }
            ReferenceValueV1::InputLength { reference_argument } => {
                self.raw_slice(*reference_argument, false)?;
                self.u8(4)?;
                self.u32(*reference_argument)
            }
            ReferenceValueV1::SafeHelperCall { .. } => Err(Error::Wire("helper calls unsupported")),
        }
    }
    fn edge(&mut self, target: u32) -> Result<(), Error> {
        require((target as usize) < self.ir.blocks.len(), "block edge")?;
        self.u32(target)
    }
    fn terminator(&mut self, terminator: &ReferenceTerminatorV1) -> Result<(), Error> {
        match terminator {
            ReferenceTerminatorV1::Return => self.u8(0),
            ReferenceTerminatorV1::Goto { target } => {
                self.u8(1)?;
                self.edge(*target)
            }
            ReferenceTerminatorV1::Switch {
                discriminant,
                values,
                otherwise,
            } => {
                self.u8(2)?;
                self.operand(discriminant)?;
                self.count(values.len(), MAX_REFERENCE_STATEMENTS_V1)?;
                for (value, target) in values {
                    self.u128(*value)?;
                    self.edge(*target)?;
                }
                self.edge(*otherwise)
            }
            ReferenceTerminatorV1::Assert {
                condition,
                expected,
                success,
                bounds_check,
            } => {
                self.u8(3)?;
                self.operand(condition)?;
                self.boolean(*expected)?;
                self.edge(*success)?;
                self.boolean(bounds_check.is_some())?;
                if let Some(bounds) = bounds_check {
                    self.operand(&bounds.index)?;
                    self.operand(&bounds.length)?;
                }
                Ok(())
            }
        }
    }
    fn expr(&mut self, expr: &ReferenceEffectExpressionV1) -> Result<(), Error> {
        self.expression(expr, 1, &mut 0)
    }
    fn expression(
        &mut self,
        expr: &ReferenceEffectExpressionV1,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(), Error> {
        *nodes = add(*nodes, 1)?;
        self.nodes = add(self.nodes, 1)?;
        require(
            depth <= DEPTH
                && *nodes <= MAX_REFERENCE_EXPRESSION_NODES_V1
                && self.nodes <= TOTAL_NODES,
            "expression depth/node limit",
        )?;
        self.s.work(1)?;
        match expr {
            ReferenceEffectExpressionV1::PointCoordinate { axis } => {
                require((*axis as usize) < self.axes, "point axis")?;
                self.u8(0)?;
                self.u32(*axis)
            }
            ReferenceEffectExpressionV1::KernelScalarArgument { argument } => {
                let raw = add(self.axes, *argument as usize)?;
                require(
                    matches!(self.ir.relations.get(raw), Some(ReferenceArgumentRelationV1::ScalarInput { argument: a, .. }) if a == argument),
                    "scalar argument",
                )?;
                self.u8(1)?;
                self.u32(*argument)
            }
            ReferenceEffectExpressionV1::Constant(value) => {
                self.u8(2)?;
                self.constant(value)
            }
            ReferenceEffectExpressionV1::Binary {
                operation,
                lhs,
                rhs,
                checked,
            } => {
                self.u8(3)?;
                self.enum_tag(BINARY, operation)?;
                self.boolean(*checked)?;
                self.expression(lhs, depth + 1, nodes)?;
                self.expression(rhs, depth + 1, nodes)
            }
            ReferenceEffectExpressionV1::Unary { operation, operand } => {
                self.u8(4)?;
                self.enum_tag(UNARY, operation)?;
                self.expression(operand, depth + 1, nodes)
            }
            ReferenceEffectExpressionV1::Cast {
                kind,
                source,
                target,
                operand,
            } => {
                self.u8(5)?;
                self.enum_tag(CASTS, kind)?;
                self.scalar(*source)?;
                self.scalar(*target)?;
                self.expression(operand, depth + 1, nodes)
            }
            ReferenceEffectExpressionV1::InputLoad {
                reference_argument,
                index,
            } => {
                self.raw_slice(*reference_argument, true)?;
                self.u8(6)?;
                self.u32(*reference_argument)?;
                self.expression(index, depth + 1, nodes)
            }
            ReferenceEffectExpressionV1::InputLength { reference_argument } => {
                self.raw_slice(*reference_argument, false)?;
                self.u8(7)?;
                self.u32(*reference_argument)
            }
        }
    }
    fn predicate(&mut self, predicate: &ReferencePathPredicateV1) -> Result<(), Error> {
        let start = self.position;
        self.count(predicate.clauses.len(), MAX_REFERENCE_GUARD_CLAUSES_V1)?;
        let mut atoms = 0;
        for clause in &predicate.clauses {
            atoms = add(atoms, clause.atoms.len())?;
            require(atoms <= MAX_REFERENCE_GUARD_ATOMS_V1, "guard atom limit")?;
            self.count(clause.atoms.len(), MAX_REFERENCE_GUARD_ATOMS_V1)?;
            for atom in &clause.atoms {
                match atom {
                    ReferenceGuardAtomV1::SwitchValueSet {
                        discriminant,
                        values,
                        inside_set,
                    } => {
                        self.u8(0)?;
                        self.expr(discriminant)?;
                        self.count(values.len(), MAX_REFERENCE_STATEMENTS_V1)?;
                        let mut previous = None;
                        for value in values {
                            require(previous.is_none_or(|p| p < *value), "guard value order")?;
                            previous = Some(*value);
                            self.u128(*value)?;
                        }
                        self.boolean(*inside_set)?;
                    }
                    ReferenceGuardAtomV1::Assert {
                        condition,
                        expected,
                    } => {
                        self.u8(1)?;
                        self.expr(condition)?;
                        self.boolean(*expected)?;
                    }
                }
            }
        }
        // Every recursive payload is bounded before derived Ord traverses it.
        self.s.work(mul(4, self.position - start)?)?;
        require(
            predicate.clauses.windows(2).all(|v| v[0] < v[1]),
            "guard clause order",
        )?;
        require(
            predicate
                .clauses
                .iter()
                .all(|c| c.atoms.windows(2).all(|v| v[0] < v[1])),
            "guard atom order",
        )
    }
    fn effect(&mut self, effect: &ReferenceOutputWriteV1) -> Result<(), Error> {
        let start = self.position;
        require(
            (effect.argument as usize) < self.kernel_count,
            "effect argument",
        )?;
        let raw = add(self.axes, effect.argument as usize)?;
        require(
            matches!(self.ir.relations.get(raw),
            Some(ReferenceArgumentRelationV1::DisjointOutputSlice { argument, .. }
                | ReferenceArgumentRelationV1::DisjointOutputCoordinate { argument, .. })
                if *argument == effect.argument),
            "effect output relation",
        )?;
        let ir = self.ir;
        let block = ir
            .blocks
            .get(effect.block as usize)
            .ok_or(Error::Wire("effect block"))?;
        self.s.work(add(block.assignments.len(), 1)?)?;
        let assignment = block
            .assignments
            .binary_search_by_key(&effect.statement, |a| a.statement)
            .ok()
            .and_then(|index| block.assignments.get(index))
            .ok_or(Error::Wire("effect statement"))?;
        self.u32(effect.argument)?;
        self.u32(effect.block)?;
        self.u32(effect.statement)?;
        match &effect.coordinate {
            ReferenceOutputCoordinateV1::LogicalPoint(axes) => {
                require(
                    !axes.is_empty() && axes.len() == self.axes,
                    "effect point shape",
                )?;
                self.u8(0)?;
                self.count(axes.len(), MAX_REFERENCE_POINT_AXES_V1)?;
                for axis in axes {
                    self.expr(axis)?;
                }
            }
            ReferenceOutputCoordinateV1::SingleCoordinate => {
                require(self.axes == 0, "effect single coordinate")?;
                self.u8(1)?;
            }
            ReferenceOutputCoordinateV1::Dynamic(expr) => {
                self.u8(2)?;
                self.expr(expr)?;
            }
            ReferenceOutputCoordinateV1::Constant {
                offset,
                minimum_length,
                from_end,
            } => {
                self.u8(3)?;
                self.constant_index(*offset, *minimum_length, *from_end)?;
            }
        }
        self.predicate(&effect.guard)?;
        self.expr(&effect.rhs)?;
        self.value(&effect.value)?;
        self.s.work(mul(4, self.position - start)?)?;
        require(assignment.value == effect.value, "effect assignment value")
    }
}

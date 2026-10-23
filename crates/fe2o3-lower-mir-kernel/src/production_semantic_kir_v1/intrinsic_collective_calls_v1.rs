// Non-inlined collective call bodies keep unrelated intrinsic temporaries off the dispatcher stack.
impl SemanticFunctionLoweringV1<'_, '_> {
    // These requirements describe the CURRENT operation. A retained WaveLane or
    // arithmetic SubgroupTile snapshot is never persistent EXEC/epoch authority.
    // Actual entry launch and CFG convergence remain checked by AMDGPU lowering.
    #[inline(never)]
    fn lower_intrinsic_gfx942_wave64_shuffle_index_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        element: SemanticTypeIdV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 3)?;
        let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
        if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "Wave64 shuffle lacks compiler-issued collective context",
            ));
        }
        let scalar = lower_scalar_type(self.types, element)?;
        let (value, value_ty) = self
            .lower_operand(block, None, &call.arguments()[1], operations)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let induction_bound = authenticated_unsigned_operand_exclusive_bound_v1(
            self.types,
            self.function,
            &self.authenticated_loop_induction_bounds,
            block,
            &call.arguments()[2],
        );
        let (source_lane, source_ty) = self
            .lower_operand(block, None, &call.arguments()[2], operations)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let bound = self
            .emitted_unsigned_exclusive_bounds
            .get(&source_lane)
            .copied()
            .or(induction_bound);
        if !workgroup_reduction_scalar_supported_v1(&scalar)
            || value_ty != scalar
            || call
                .destination()
                .is_none_or(|destination| destination.place().ty() != element)
            || source_ty != Type::Scalar(ScalarType::U32)
            || !bound.is_some_and(|bound| (1..=64).contains(&bound))
        {
            eprintln!(
                "Wave64 shuffle rejected: function {:?}, block {}, scalar {:?}, value type {:?}, destination type {:?}, source type {:?}, source SSA {:?}, emitted bound {:?}, induction bound {:?}, source operand {:?}",
                self.semantic_function, block.index(), scalar, value_ty,
                call.destination().map(|destination| destination.place().ty()),
                source_ty, source_lane,
                self.emitted_unsigned_exclusive_bounds.get(&source_lane),
                induction_bound, &call.arguments()[2],
            );
            eprintln!("Wave64 rejected source: role {:?}, entry {:?}, ABI {:?}, locals {:?}, rank locals {:?}",
                self.function.role(), self.function.entry(), self.function.abi(),
                &self.function.locals()[..self.function.locals().len().min(16)],
                self.wave64_references.rank_locals);
            for (index, source_block) in self.function.blocks().iter().take(8).enumerate() {
                eprintln!("Wave64 rejected source block {index}: statements {:?}, terminator {:?}",
                    &source_block.statements()[..source_block.statements().len().min(8)],
                    source_block.terminator());
            }
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "Wave64 shuffle requires exact u32/i32/f32 and an authenticated u32 lane below 64",
            ));
        }
        // Full participation is an obligation, not authority manufactured by this
        // label. Unchanged native validation rejects partial waves, divergent CFG,
        // unauthenticated targets and wave operations in ordinary helper bodies.
        let bits = if scalar == Type::Scalar(ScalarType::U32) {
            value
        } else {
            self.emit_id(
                operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value,
                    to: Type::Scalar(ScalarType::U32),
                },
            )?
        };
        let shuffled = self.emit_id(
            operations,
            Type::Scalar(ScalarType::U32),
            OperationKind::Wave(WaveOperation::full(
                WaveOperationKind::ShuffleIndex {
                    value: bits,
                    source_lane,
                    tile_width: 64,
                },
                WaveWidth::Wave64,
            )),
        )?;
        if scalar == Type::Scalar(ScalarType::U32) {
            Ok(SemanticValueBindingV1::Value {
                id: shuffled,
                ty: scalar,
            })
        } else {
            self.emit(
                operations,
                scalar.clone(),
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: shuffled,
                    to: scalar,
                },
            )
        }
    }
    #[inline(never)]
    fn lower_intrinsic_subgroup_reduce_f32_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        width: &u32,
        kind: &SemanticSubgroupReductionKindV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 2)?;
            let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
            if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "subgroup reduction lacks compiler-issued collective authority",
                ));
            }
            let value = self.lower_operand(block, None, &call.arguments()[1], operations)?;
            self.lower_subgroup_reduce_f32(block, operations, value, *width, *kind)?
        })
    }

    #[inline(never)]
    fn lower_intrinsic_gfx950_subgroup_reduce_f32_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        width: &u32,
        kind: &SemanticSubgroupReductionKindV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 2)?;
            let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
            if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "gfx950 subgroup reduction lacks compiler-issued authority",
                ));
            }
            let (value, ty) = self
                .lower_operand(block, None, &call.arguments()[1], operations)?
                .value()
                .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
            if ty != Type::Scalar(ScalarType::F32)
                || *width == 0
                || !width.is_power_of_two()
                || *width > 64
            {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "gfx950 subgroup reduction type or width changed",
                ));
            }
            let kind = match kind {
                SemanticSubgroupReductionKindV1::Sum => WaveF32ReductionKindV1::Sum,
                SemanticSubgroupReductionKindV1::Maximum => WaveF32ReductionKindV1::Maximum,
            };
            self.emit(
                operations,
                Type::Scalar(ScalarType::F32),
                OperationKind::Wave(WaveOperation::full(
                    WaveOperationKind::ReduceF32 {
                        value,
                        tile_width: *width,
                        kind,
                    },
                    WaveWidth::Wave64,
                )),
            )?
        })
    }

    #[inline(never)]
    fn lower_intrinsic_subgroup_broadcast_f32_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        width: &u32,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 3)?;
            let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
            if !matches!(context, SemanticValueBindingV1::CollectiveContext) {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "gfx950 subgroup broadcast lacks compiler-issued authority",
                ));
            }
            let (value, value_ty) = self
                .lower_operand(block, None, &call.arguments()[1], operations)?
                .value()
                .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
            let authenticated_source_bound = authenticated_unsigned_operand_exclusive_bound_v1(
                self.types,
                self.function,
                &self.authenticated_loop_induction_bounds,
                block,
                &call.arguments()[2],
            );
            let (source_lane, source_ty) = self
                .lower_operand(block, None, &call.arguments()[2], operations)?
                .value()
                .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
            let authenticated_source_bound = authenticated_source_bound.filter(|bound| {
                source_ty == Type::Scalar(ScalarType::U32)
                    && authenticated_subgroup_broadcast_source_is_bounded(*bound, *width)
            });
            let source_lane = if authenticated_source_bound.is_some() {
                let mask = self
                    .emit(
                        operations,
                        Type::Scalar(ScalarType::U32),
                        OperationKind::Constant(Constant::U32(*width - 1)),
                    )?
                    .value()
                    .expect("authenticated subgroup mask has one value")
                    .0;
                self.emit(
                    operations,
                    Type::Scalar(ScalarType::U32),
                    OperationKind::Binary {
                        op: BinaryOp::BitAnd,
                        lhs: source_lane,
                        rhs: mask,
                    },
                )?
                .value()
                .expect("authenticated subgroup source mask has one value")
                .0
            } else {
                source_lane
            };
            if let Some(bound) = authenticated_source_bound {
                self.emitted_unsigned_exclusive_bounds
                    .insert(source_lane, bound);
            }
            let bounded_source =
                subgroup_broadcast_source_is_statically_bounded(operations, source_lane, *width)
                    || self
                        .emitted_u32_constants
                        .get(&source_lane)
                        .is_some_and(|lane| *lane < *width)
                    || self
                        .emitted_u32_bitand_masks
                        .get(&source_lane)
                        .is_some_and(|mask| *mask < *width)
                    || self
                        .emitted_unsigned_exclusive_bounds
                        .get(&source_lane)
                        .is_some_and(|bound| {
                            authenticated_subgroup_broadcast_source_is_bounded(*bound, *width)
                        });
            if value_ty != Type::Scalar(ScalarType::F32)
                || source_ty != Type::Scalar(ScalarType::U32)
                || *width == 0
                || !width.is_power_of_two()
                || *width > 64
                || !bounded_source
            {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "gfx950 subgroup broadcast requires f32, a valid width, and a statically bounded source lane",
                ));
            }
            self.emit(
                operations,
                Type::Scalar(ScalarType::F32),
                OperationKind::Wave(WaveOperation::full(
                    WaveOperationKind::BroadcastF32 {
                        value,
                        source_lane,
                        tile_width: *width,
                    },
                    WaveWidth::Wave64,
                )),
            )?
        })
    }
}

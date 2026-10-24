// Total fixed-width saturation. The checked overflow result is deliberately
// unused: source/ranked correspondence independently matches the predicate.
use fe2o3_mir_model::semantic_mir_v1::SemanticSaturatingIntegerOpV1;

fn saturating_integer_shape_v1(ty: &Type) -> Option<(bool, u16)> {
    let scalar = ty.as_scalar()?;
    match scalar {
        ScalarType::I8
        | ScalarType::U8
        | ScalarType::I16
        | ScalarType::U16
        | ScalarType::I32
        | ScalarType::U32
        | ScalarType::I64
        | ScalarType::U64 => Some((scalar.is_signed_integer(), scalar.bit_width()?)),
        _ => None,
    }
}

impl SemanticFunctionLoweringV1<'_, '_> {
    fn lower_saturating_integer_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        operation: SemanticSaturatingIntegerOpV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.require_call_argument_count(block, call, 2)?;
        let invalid = || {
            unsupported(
                self.semantic_function.index(),
                Some(block.index()),
                None,
                "saturating integer intrinsic requires exact non-unwinding (T, T) -> T",
            )
        };
        let destination = call.destination().ok_or_else(invalid)?;
        let semantic_ty = destination.place().ty();
        let Some(SemanticCallableDeclV1::CompilerIntrinsic {
            binding,
            operation: SemanticCompilerIntrinsicOperationV1::SaturatingInteger(actual),
            ..
        }) = self.callables.get(call.callee().index() as usize)
        else {
            return Err(invalid());
        };
        let abi = binding.abi();
        if *actual != operation
            || abi.canon_abi() != SemanticCanonAbiV1::Rust
            || abi.extern_abi() != fe2o3_mir_model::semantic_mir_v1::SemanticExternAbiV1::Rust
            || abi.c_variadic()
            || abi.can_unwind()
            || abi.source_input_types() != [semantic_ty, semantic_ty]
            || abi.source_output_type() != semantic_ty
            || abi.return_type() != semantic_ty
            || !call.variadic_argument_abis().is_empty()
            || matches!(call.unwind(), SemanticUnwindActionV1::Cleanup(_))
            || call
                .arguments()
                .iter()
                .any(|argument| argument.ty() != semantic_ty)
            || !matches!(
                self.types
                    .get(semantic_ty.index() as usize)
                    .map(SemanticTypeDeclV1::shape),
                Some(SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                    bits: 8 | 16 | 32 | 64,
                    ..
                }))
            )
        {
            return Err(invalid());
        }
        let expected = lower_scalar_type(self.types, semantic_ty)?;
        // Each actual operand is evaluated exactly once, in source order.
        let (lhs, left_ty) = self
            .lower_operand(block, None, &call.arguments()[0], operations)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        let (rhs, right_ty) = self
            .lower_operand(block, None, &call.arguments()[1], operations)?
            .value()
            .map_err(|detail| unsupported(0, Some(block.index()), None, detail))?;
        if left_ty != expected || right_ty != expected {
            return Err(unsupported(
                0,
                Some(block.index()),
                None,
                "saturating integer lowered operand types changed",
            ));
        }
        self.emit_saturating_integer_v1(operations, expected, operation, lhs, rhs)
    }

    fn emit_saturating_integer_v1(
        &mut self,
        operations: &mut Vec<Operation>,
        ty: Type,
        operation: SemanticSaturatingIntegerOpV1,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        let (signed, bits) = saturating_integer_shape_v1(&ty).ok_or_else(|| {
            unsupported(
                0,
                None,
                None,
                "saturation requires a fixed 8/16/32/64-bit integer",
            )
        })?;
        let checked = match operation {
            SemanticSaturatingIntegerOpV1::Add => CheckedBinaryOperator::Add,
            SemanticSaturatingIntegerOpV1::Subtract => CheckedBinaryOperator::Subtract,
        };
        let SemanticValueBindingV1::Aggregate(parts) =
            self.emit_checked_binary(operations, ty.clone(), checked, lhs, rhs)?
        else {
            unreachable!("checked binary emits value and overflow");
        };
        let (raw, _) = parts
            .into_iter()
            .next()
            .expect("checked value is result zero")
            .value()
            .map_err(|detail| unsupported(0, None, None, detail))?;
        let (overflow, clamp) = if signed {
            let changed = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs,
                    rhs: raw,
                },
            )?;
            let (other_lhs, other_rhs) = match operation {
                SemanticSaturatingIntegerOpV1::Add => (rhs, raw),
                SemanticSaturatingIntegerOpV1::Subtract => (lhs, rhs),
            };
            let other = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs: other_lhs,
                    rhs: other_rhs,
                },
            )?;
            let signs = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: changed,
                    rhs: other,
                },
            )?;
            let zero = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Constant(integer_constant(&ty, 0)?),
            )?;
            let overflow =
                self.emit_compare(operations, ComparePredicate::LessThan, signs, zero)?;
            let negative = self.emit_compare(operations, ComparePredicate::LessThan, lhs, zero)?;
            let sign = 1_u128 << (bits - 1);
            let minimum = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Constant(integer_constant(&ty, sign)?),
            )?;
            let maximum = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Constant(integer_constant(&ty, sign - 1)?),
            )?;
            let clamp = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Select {
                    condition: negative,
                    true_value: minimum,
                    false_value: maximum,
                },
            )?;
            (overflow, clamp)
        } else {
            let (left, right, clamp) = match operation {
                SemanticSaturatingIntegerOpV1::Add => (raw, lhs, (1_u128 << bits) - 1),
                SemanticSaturatingIntegerOpV1::Subtract => (lhs, rhs, 0),
            };
            let overflow =
                self.emit_compare(operations, ComparePredicate::LessThan, left, right)?;
            let clamp = self.emit_id(
                operations,
                ty.clone(),
                OperationKind::Constant(integer_constant(&ty, clamp)?),
            )?;
            (overflow, clamp)
        };
        let result = self.emit(
            operations,
            ty,
            OperationKind::Select {
                condition: overflow,
                true_value: clamp,
                false_value: raw,
            },
        )?;
        // For unsigned subtraction, max(lhs - rhs, 0) never exceeds lhs.
        // Record this only for the final clamp selection, never the wrapped raw result.
        if !signed
            && operation == SemanticSaturatingIntegerOpV1::Subtract
            && let Some(bound) = self.emitted_unsigned_exclusive_bounds.get(&lhs).copied()
        {
            let (value, _) = result
                .clone()
                .value()
                .map_err(|detail| unsupported(0, None, None, detail))?;
            self.emitted_unsigned_exclusive_bounds.insert(value, bound);
        }
        Ok(result)
    }
}

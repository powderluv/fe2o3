// Non-inlined matrix call bodies keep unrelated intrinsic temporaries off the dispatcher stack.
impl SemanticFunctionLoweringV1<'_, '_> {
    #[inline(never)]
    fn lower_intrinsic_wave_lane_current_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        lane: &SemanticTypeIdV1,
        wave_width: &u32,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 0)?;
            if *wave_width != 64 {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "typed MFMA lane requires the authenticated wave64 profile",
                ));
            }
            let lane_id = self.emit_results(
                operations,
                vec![Type::Scalar(ScalarType::U32)],
                OperationKind::Wave(WaveOperation::full(
                    WaveOperationKind::LaneId,
                    WaveWidth::Wave64,
                )),
            )?;
            let lane_binding = binding_from_value_defs(self.types, *lane, &lane_id)?;
            let value = require_single_u32_component(
                block,
                lane_binding,
                "typed MFMA lane has no exact u32 representation",
            )?;
            self.emitted_unsigned_exclusive_bounds.insert(value, 64);
            SemanticValueBindingV1::WaveLane {
                value,
                wave: SemanticCurrentWaveV1::new(*wave_width),
            }
        })
    }

    #[inline(never)]
    fn lower_intrinsic_f32_matrix_accumulator_zero_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        fragment: &SemanticTypeIdV1,
        contract: &SemanticMfmaAccumulatorContractV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 1)?;
            let (_, wave) = require_current_wave_lane(
                block,
                self.lower_operand(block, None, &call.arguments()[0], operations)?,
                contract.wave_width,
                "zero accumulator lane",
            )?;
            let mut values = Vec::with_capacity(4);
            for _ in 0..4 {
                let (id, ty) = self
                    .emit(
                        operations,
                        Type::Scalar(ScalarType::F32),
                        OperationKind::Constant(Constant::F32Bits(0.0_f32.to_bits())),
                    )?
                    .value()
                    .expect("emitted zero accumulator component");
                values.push((id, ty));
            }
            let _ = fragment;
            SemanticValueBindingV1::AccumulatorFragment {
                values,
                contract: *contract,
                wave,
            }
        })
    }

    #[inline(never)]
    fn lower_intrinsic_f32_matrix_accumulator_values_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        values: &SemanticTypeIdV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 1)?;
            let fragment = self.lower_operand(block, None, &call.arguments()[0], operations)?;
            let SemanticValueBindingV1::AccumulatorFragment {
                values: fragment, ..
            } = fragment
            else {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "FP32 matrix accumulator lacks typed producer metadata",
                ));
            };
            let fragment = require_components(
                block,
                fragment,
                Type::Scalar(ScalarType::F32),
                4,
                "FP32 matrix accumulator fragment",
            )?
            .into_iter()
            .map(|(id, ty)| ValueDef::new(id, ty))
            .collect::<Vec<_>>();
            binding_from_value_defs(self.types, *values, &fragment)?
        })
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn lower_intrinsic_matrix_multiply_accumulate_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        accumulator_fragment: &SemanticTypeIdV1,
        expected_lhs: &SemanticMfmaOperandContractV1,
        expected_rhs: &SemanticMfmaOperandContractV1,
        expected_accumulator: &SemanticMfmaAccumulatorContractV1,
        operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        Ok({
            self.require_call_argument_count(block, call, 4)?;
            let context = self.lower_operand(block, None, &call.arguments()[0], operations)?;
            if !matches!(context, SemanticValueBindingV1::MatrixContext) {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "matrix operation lacks compiler-issued context authority",
                ));
            }
            let lhs = self.lower_operand(block, None, &call.arguments()[1], operations)?;
            let rhs = self.lower_operand(block, None, &call.arguments()[2], operations)?;
            let accumulator = self.lower_operand(block, None, &call.arguments()[3], operations)?;
            let SemanticValueBindingV1::MatrixFragment {
                values: lhs,
                contract: lhs_contract,
                storage_layout: lhs_storage,
                wave: lhs_wave,
            } = lhs
            else {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "matrix lhs lacks an authenticated checked-load producer",
                ));
            };
            let SemanticValueBindingV1::MatrixFragment {
                values: rhs,
                contract: rhs_contract,
                storage_layout: rhs_storage,
                wave: rhs_wave,
            } = rhs
            else {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "matrix rhs lacks an authenticated checked-load producer",
                ));
            };
            let SemanticValueBindingV1::AccumulatorFragment {
                values: accumulator,
                contract: accumulator_contract,
                wave: accumulator_wave,
            } = accumulator
            else {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "matrix accumulator lacks an authenticated zero/MFMA producer",
                ));
            };
            if lhs_contract != *expected_lhs
                || rhs_contract != *expected_rhs
                || accumulator_contract != *expected_accumulator
                || lhs_wave != rhs_wave
                || lhs_wave != accumulator_wave
            {
                return Err(unsupported(
                    0,
                    Some(block.index()),
                    None,
                    "matrix operand producer contracts or wave associations differ",
                ));
            }
            if matches!(
                expected_accumulator.profile,
                SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                    | SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128
            ) {
                if lhs_storage != SemanticMfmaStorageLayoutV1::RowMajor
                    || rhs_storage != SemanticMfmaStorageLayoutV1::RowMajor
                {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "gfx950 low-precision matrix operands require checked row-major producers",
                    ));
                }
                let lhs = require_components(
                    block,
                    lhs,
                    Type::Scalar(ScalarType::U32),
                    8,
                    "gfx950 low-precision matrix lhs fragment",
                )?
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>()
                .try_into()
                .expect("eight checked gfx950 lhs dwords");
                let rhs = require_components(
                    block,
                    rhs,
                    Type::Scalar(ScalarType::U32),
                    8,
                    "gfx950 low-precision matrix rhs fragment",
                )?
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>()
                .try_into()
                .expect("eight checked gfx950 rhs dwords");
                let accumulator = require_components(
                    block,
                    accumulator,
                    Type::Scalar(ScalarType::F32),
                    4,
                    "gfx950 low-precision matrix accumulator fragment",
                )?
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>()
                .try_into()
                .expect("four checked gfx950 accumulator components");
                let matrix = if expected_accumulator.profile
                    == SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                {
                    let layout = if expected_lhs.profile
                        == SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                        && expected_rhs.profile == SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128
                    {
                        TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_fp8_e4m3_f32_m16n16k128_wave64()
                    } else {
                        TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_f32_m16n16k128_wave64()
                    };
                    MatrixOperation::scaled_multiply_accumulate_fp4_e2m1(lhs, rhs, accumulator)
                        .with_declared_tensor_layout(layout)
                } else {
                    MatrixOperation::scaled_multiply_accumulate_fp8_e4m3(lhs, rhs, accumulator)
                        .with_declared_tensor_layout(
                        TensorLayoutContractV1::gfx950_scaled_mfma_fp8_e4m3_f32_m16n16k128_wave64(),
                    )
                };
                let results = self.emit_results(
                    operations,
                    vec![Type::Scalar(ScalarType::F32); 4],
                    OperationKind::Matrix(matrix),
                )?;
                let _ = accumulator_fragment;
                SemanticValueBindingV1::AccumulatorFragment {
                    values: results
                        .into_iter()
                        .map(|value| (value.id, value.ty))
                        .collect(),
                    contract: accumulator_contract,
                    wave: accumulator_wave,
                }
            } else {
                if !matches!(
                    lhs_storage,
                    SemanticMfmaStorageLayoutV1::RowMajor | SemanticMfmaStorageLayoutV1::LdsXor4
                ) || !matches!(
                    rhs_storage,
                    SemanticMfmaStorageLayoutV1::RowMajor
                        | SemanticMfmaStorageLayoutV1::LdsXor4
                        | SemanticMfmaStorageLayoutV1::ColumnMajor
                ) {
                    return Err(unsupported(
                        0,
                        Some(block.index()),
                        None,
                        "BF16 matrix source layout is not admitted for its operand role",
                    ));
                }
                let lhs = require_components(
                    block,
                    lhs,
                    Type::Scalar(ScalarType::Bf16),
                    4,
                    "matrix lhs fragment",
                )?;
                let rhs = require_components(
                    block,
                    rhs,
                    Type::Scalar(ScalarType::Bf16),
                    4,
                    "matrix rhs fragment",
                )?;
                let accumulator = require_components(
                    block,
                    accumulator,
                    Type::Scalar(ScalarType::F32),
                    4,
                    "matrix accumulator fragment",
                )?;
                let lhs = lhs
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("four checked lhs components");
                let rhs = rhs
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("four checked rhs components");
                let accumulator = accumulator
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("four checked accumulator components");
                let mut tensor_layout =
                    TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64()
                        .with_zero_filled_predicate_inputs();
                if lhs_storage == SemanticMfmaStorageLayoutV1::LdsXor4 {
                    tensor_layout = tensor_layout.with_a_lds_xor4();
                }
                if rhs_storage == SemanticMfmaStorageLayoutV1::LdsXor4 {
                    tensor_layout = tensor_layout.with_b_lds_xor4();
                }
                let results = self.emit_results(
                    operations,
                    vec![Type::Scalar(ScalarType::F32); 4],
                    OperationKind::Matrix(
                        MatrixOperation::multiply_accumulate(lhs, rhs, accumulator)
                            .with_declared_tensor_layout(tensor_layout),
                    ),
                )?;
                let _ = accumulator_fragment;
                SemanticValueBindingV1::AccumulatorFragment {
                    values: results
                        .into_iter()
                        .map(|value| (value.id, value.ty))
                        .collect(),
                    contract: accumulator_contract,
                    wave: accumulator_wave,
                }
            }
        })
    }
}

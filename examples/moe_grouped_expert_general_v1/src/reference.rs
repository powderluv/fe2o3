#![forbid(unsafe_code)]

//! Sequential safe-Rust reference for one dynamic routed expert launch.

use fe2o3_device::Bf16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceLayoutV1 {
    pub rows_padded: u32,
    pub output_columns: u32,
    pub reduction: u32,
    pub token_stride: u32,
    pub weight_stride: u32,
    pub expert_weight_stride: u32,
    pub bias_stride: u32,
    pub output_stride: u32,
    pub expert: u32,
    pub experts: u32,
}

pub fn evaluate_reference_v1(
    tokens: &[u16],
    weights: &[u16],
    gates: &[f32],
    bias: &[f32],
    initial_output: &[f32],
    layout: ReferenceLayoutV1,
) -> Result<Vec<f32>, &'static str> {
    if layout.expert >= layout.experts {
        return Err("expert identity is outside the declared expert domain");
    }
    if layout.token_stride < layout.reduction
        || layout.weight_stride < layout.output_columns
        || layout.bias_stride < layout.output_columns
        || layout.output_stride < layout.output_columns
    {
        return Err("a physical stride is smaller than its logical extent");
    }
    let checked_len = |major: u32, stride: u32| {
        usize::try_from(major)
            .ok()
            .and_then(|major| major.checked_mul(stride as usize))
    };
    let token_len =
        checked_len(layout.rows_padded, layout.token_stride).ok_or("token extent overflow")?;
    let weight_len =
        checked_len(layout.experts, layout.expert_weight_stride).ok_or("weight extent overflow")?;
    let bias_len = checked_len(layout.experts, layout.bias_stride).ok_or("bias extent overflow")?;
    let output_len =
        checked_len(layout.rows_padded, layout.output_stride).ok_or("output extent overflow")?;
    if tokens.len() < token_len
        || weights.len() < weight_len
        || gates.len() < layout.rows_padded as usize
        || bias.len() < bias_len
        || initial_output.len() < output_len
    {
        return Err("reference input is shorter than its declared physical extent");
    }

    let mut output = initial_output.to_vec();
    for row in 0..layout.rows_padded as usize {
        for column in 0..layout.output_columns as usize {
            let mut projection = 0.0_f32;
            for depth in 0..layout.reduction as usize {
                let activation =
                    Bf16::from_bits(tokens[row * layout.token_stride as usize + depth]).to_f32();
                let weight = Bf16::from_bits(
                    weights[layout.expert as usize * layout.expert_weight_stride as usize
                        + depth * layout.weight_stride as usize
                        + column],
                )
                .to_f32();
                projection += activation * weight;
            }
            projection += bias[layout.expert as usize * layout.bias_stride as usize + column];
            output[row * layout.output_stride as usize + column] = gates[row] * projection;
        }
    }
    Ok(output)
}

/// Validates the complete finite fixture and bitwise-preserved stride padding.
pub fn validate_output_v1(
    actual: &[f32],
    expected: &[f32],
    initial_output: &[f32],
    layout: ReferenceLayoutV1,
) -> Result<f32, String> {
    let rows = layout.rows_padded as usize;
    let length = rows
        .checked_mul(layout.output_stride as usize)
        .ok_or("output extent overflow")?;
    if layout.output_columns > layout.output_stride
        || actual.len() != length
        || expected.len() != length
        || initial_output.len() != length
    {
        return Err("output validation shape mismatch".into());
    }
    let mut maximum_error = 0.0_f32;
    for row in 0..rows {
        for column in 0..layout.output_stride as usize {
            let index = row * layout.output_stride as usize + column;
            if column < layout.output_columns as usize {
                let error = (actual[index] - expected[index]).abs();
                let tolerance = 5.0e-3_f32.max(expected[index].abs() * 5.0e-3);
                if !actual[index].is_finite()
                    || !expected[index].is_finite()
                    || !error.is_finite()
                    || error > tolerance
                {
                    return Err(format!(
                        "MoE expert mismatch at ({row}, {column}): got {}, expected {}, tolerance {tolerance}",
                        actual[index], expected[index]
                    ));
                }
                maximum_error = maximum_error.max(error);
            } else if actual[index].to_bits() != initial_output[index].to_bits() {
                return Err(format!(
                    "MoE expert modified output padding at ({row}, {column})"
                ));
            }
        }
    }
    Ok(maximum_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_gate_zeroes_a_padded_route_and_preserves_columns() {
        let bits = |value| Bf16::from_f32(value).to_bits();
        let tokens = [
            bits(2.0),
            bits(3.0),
            bits(9.0),
            bits(4.0),
            bits(5.0),
            bits(8.0),
        ];
        let weights = [bits(7.0), bits(11.0), bits(13.0), bits(17.0)];
        let output = evaluate_reference_v1(
            &tokens,
            &weights,
            &[0.5, 0.0],
            &[1.0, 2.0],
            &[-7.0; 6],
            ReferenceLayoutV1 {
                rows_padded: 2,
                output_columns: 2,
                reduction: 2,
                token_stride: 3,
                weight_stride: 2,
                expert_weight_stride: 4,
                bias_stride: 2,
                output_stride: 3,
                expert: 0,
                experts: 1,
            },
        )
        .unwrap();
        assert_eq!(output, [27.0, 37.5, -7.0, 0.0, 0.0, -7.0]);
    }

    fn strided_layout(expert: u32) -> ReferenceLayoutV1 {
        ReferenceLayoutV1 {
            rows_padded: 16,
            output_columns: 3,
            reduction: 3,
            token_stride: 5,
            weight_stride: 5,
            expert_weight_stride: 17,
            bias_stride: 4,
            output_stride: 6,
            expert,
            experts: 2,
        }
    }

    #[test]
    fn nonquarter_routed_experts_keep_selection_gating_bias_and_all_padding() {
        let layout = strided_layout(0);
        let bits = |value| Bf16::from_f32(value).to_bits();
        let mut tokens = vec![0x7fc1; 16 * 5];
        for row in 0..16 {
            tokens[row * 5..row * 5 + 3].copy_from_slice(&[bits(1.125), bits(-0.625), bits(0.5)]);
        }
        let columns: [[f32; 3]; 2] = [[0.5, -0.75, 1.25], [2.0, -1.0, 0.25]];
        let biases = [[0.25, -0.5, 0.75], [-0.25, 0.5, -0.75]];
        let mut weights = vec![0x7fc1; 2 * 17];
        let mut bias = vec![f32::NAN; 2 * 4];
        for expert in 0..2 {
            for column in 0..3 {
                bias[expert * 4 + column] = biases[expert][column];
                for depth in 0..3 {
                    weights[expert * 17 + depth * 5 + column] = bits(columns[expert][column]);
                }
            }
        }
        let gates = (0..16)
            .map(|row| [0.5_f32, -0.25, 0.0][row % 3])
            .collect::<Vec<_>>();
        let initial = (0..16 * 6)
            .map(|index| 1000.0 + index as f32)
            .collect::<Vec<_>>();
        // The three activations sum exactly to one. These projections include
        // the selected expert's bias before gating, independently of the oracle.
        let projections: [[f32; 3]; 2] = [[0.75, -1.25, 2.0], [1.75, -0.5, -0.5]];
        for expert in 0..2 {
            let selected = ReferenceLayoutV1 {
                expert: expert as u32,
                ..layout
            };
            let actual =
                evaluate_reference_v1(&tokens, &weights, &gates, &bias, &initial, selected)
                    .unwrap();
            for row in 0..16 {
                for column in 0..6 {
                    let expected = if column < 3 {
                        gates[row] * projections[expert][column]
                    } else {
                        initial[row * 6 + column]
                    };
                    assert_eq!(actual[row * 6 + column].to_bits(), expected.to_bits());
                }
            }
            assert_eq!(
                validate_output_v1(&actual, &actual, &initial, selected),
                Ok(0.0)
            );
        }
    }

    #[test]
    fn cancellation_and_gated_bias_preserve_the_last_logical_output() {
        let bits = |value| Bf16::from_f32(value).to_bits();
        let layout = ReferenceLayoutV1 {
            rows_padded: 16,
            output_columns: 1,
            reduction: 3,
            token_stride: 4,
            weight_stride: 2,
            expert_weight_stride: 7,
            bias_stride: 2,
            output_stride: 3,
            expert: 1,
            experts: 2,
        };
        let mut tokens = vec![0x7fc1; 16 * 4];
        for row in 0..16 {
            tokens[row * 4..row * 4 + 3].copy_from_slice(&[bits(1.125), bits(1.125), bits(0.5)]);
        }
        tokens[15 * 4 + 2] = bits(0.75);
        let mut weights = vec![0x7fc1; 14];
        weights[7] = bits(1.0);
        weights[9] = bits(-1.0);
        weights[11] = bits(0.25);
        let output = evaluate_reference_v1(
            &tokens,
            &weights,
            &[-0.5; 16],
            &[9.0, f32::NAN, -0.0625, f32::NAN],
            &[37.0; 48],
            layout,
        )
        .unwrap();
        for row in 0..16 {
            assert_eq!(output[row * 3], if row == 15 { -0.0625 } else { -0.03125 });
            assert_eq!(&output[row * 3 + 1..row * 3 + 3], &[37.0, 37.0]);
        }
    }

    #[test]
    fn complete_output_validation_rejects_nan_infinity_and_padding_damage() {
        let layout = strided_layout(1);
        let expected = vec![1.25; 16 * 6];
        assert_eq!(
            validate_output_v1(&expected, &expected, &expected, layout),
            Ok(0.0)
        );
        for index in [0, 8 * 6 + 1, 15 * 6 + 2] {
            for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut actual = expected.clone();
                actual[index] = nonfinite;
                assert!(validate_output_v1(&actual, &expected, &expected, layout).is_err());
            }
        }
        for index in [3, 15 * 6 + 5] {
            let mut actual = expected.clone();
            actual[index] = f32::NAN;
            assert!(validate_output_v1(&actual, &expected, &expected, layout).is_err());
            actual[index] = 2.0;
            assert!(validate_output_v1(&actual, &expected, &expected, layout).is_err());
        }
    }

    #[test]
    fn output_validation_rejects_nonfinite_expectation_error_and_wrong_length() {
        let layout = strided_layout(1);
        let expected = vec![1.25; 16 * 6];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut reference = expected.clone();
            reference[0] = bad;
            assert!(validate_output_v1(&expected, &reference, &expected, layout).is_err());
        }
        let mut actual = expected.clone();
        let mut reference = expected.clone();
        actual[0] = f32::MAX;
        reference[0] = -f32::MAX;
        assert!(validate_output_v1(&actual, &reference, &expected, layout).is_err());
        assert!(validate_output_v1(&expected[..95], &expected, &expected, layout).is_err());
    }
}

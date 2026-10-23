#![forbid(unsafe_code)]

//! Sequential safe-Rust reference for dynamic masked attention.

use fe2o3_device::Bf16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceLayoutV1 {
    pub batch_heads: u32,
    pub queries: u32,
    pub query_rows_padded: u32,
    pub keys: u32,
    pub keys_padded: u32,
    pub depth: u32,
    pub value_dimension: u32,
    pub query_stride: u32,
    pub key_depth_stride: u32,
    pub key_head_stride: u32,
    pub value_stride: u32,
    pub value_head_stride: u32,
    pub mask_stride: u32,
    pub output_stride: u32,
    pub scale: f32,
}

pub fn evaluate_reference_v1(
    query: &[u16],
    key: &[u16],
    value: &[f32],
    mask: &[f32],
    initial_output: &[f32],
    layout: ReferenceLayoutV1,
) -> Result<Vec<f32>, &'static str> {
    if layout.queries > layout.query_rows_padded || layout.keys > layout.keys_padded {
        return Err("a logical attention extent exceeds its padded extent");
    }
    if layout.query_stride < layout.depth
        || layout.key_depth_stride < layout.keys_padded
        || layout.value_stride < layout.value_dimension
        || layout.mask_stride < layout.keys_padded
        || layout.output_stride < layout.value_dimension
    {
        return Err("a physical stride is smaller than its logical extent");
    }
    let output_rows = layout
        .batch_heads
        .checked_mul(layout.query_rows_padded)
        .ok_or("output row extent overflow")?;
    let checked_len = |major: u32, stride: u32| {
        usize::try_from(major)
            .ok()
            .and_then(|major| major.checked_mul(stride as usize))
    };
    let query_len = checked_len(output_rows, layout.query_stride).ok_or("query extent overflow")?;
    let key_len =
        checked_len(layout.batch_heads, layout.key_head_stride).ok_or("key extent overflow")?;
    let value_len =
        checked_len(layout.batch_heads, layout.value_head_stride).ok_or("value extent overflow")?;
    let mask_len = checked_len(output_rows, layout.mask_stride).ok_or("mask extent overflow")?;
    let output_len =
        checked_len(output_rows, layout.output_stride).ok_or("output extent overflow")?;
    if query.len() < query_len
        || key.len() < key_len
        || value.len() < value_len
        || mask.len() < mask_len
        || initial_output.len() < output_len
    {
        return Err("reference input is shorter than its declared physical extent");
    }

    let mut output = initial_output.to_vec();
    for head in 0..layout.batch_heads as usize {
        for row in 0..layout.query_rows_padded as usize {
            let global_row = head * layout.query_rows_padded as usize + row;
            let mut scores = vec![f32::NEG_INFINITY; layout.keys as usize];
            if row < layout.queries as usize {
                for key_index in 0..layout.keys as usize {
                    let mask_value = mask[global_row * layout.mask_stride as usize + key_index];
                    if !mask_value.is_finite() {
                        continue;
                    }
                    let mut score = 0.0_f32;
                    for depth in 0..layout.depth as usize {
                        let query_value = Bf16::from_bits(
                            query[global_row * layout.query_stride as usize + depth],
                        )
                        .to_f32();
                        let key_value = Bf16::from_bits(
                            key[head * layout.key_head_stride as usize
                                + depth * layout.key_depth_stride as usize
                                + key_index],
                        )
                        .to_f32();
                        score += query_value * key_value;
                    }
                    scores[key_index] = score * layout.scale + mask_value;
                }
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = if maximum == f32::NEG_INFINITY {
                0.0
            } else {
                scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .sum::<f32>()
            };
            for component in 0..layout.value_dimension as usize {
                let mut reference = 0.0_f32;
                if denominator > 0.0 {
                    for key_index in 0..layout.keys as usize {
                        reference += ((scores[key_index] - maximum).exp() / denominator)
                            * value[head * layout.value_head_stride as usize
                                + key_index * layout.value_stride as usize
                                + component];
                    }
                }
                output[global_row * layout.output_stride as usize + component] = reference;
            }
        }
    }
    Ok(output)
}

/// Checks every finite logical output and every bitwise-preserved padding slot.
/// Returns the maximum absolute error; nonfinite values cannot satisfy tolerance.
pub fn validate_output_v1(
    actual: &[f32],
    expected: &[f32],
    initial_output: &[f32],
    layout: ReferenceLayoutV1,
) -> Result<f32, String> {
    let rows = layout
        .batch_heads
        .checked_mul(layout.query_rows_padded)
        .ok_or("output row extent overflow")? as usize;
    let length = rows
        .checked_mul(layout.output_stride as usize)
        .ok_or("output physical extent overflow")?;
    if layout.value_dimension > layout.output_stride
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
            if column < layout.value_dimension as usize {
                let error = (actual[index] - expected[index]).abs();
                let tolerance = 5.0e-3_f32.max(expected[index].abs() * 5.0e-3);
                if !actual[index].is_finite()
                    || !expected[index].is_finite()
                    || !error.is_finite()
                    || error > tolerance
                {
                    return Err(format!(
                        "attention mismatch at ({row}, {column}): got {}, expected {}, tolerance {tolerance}",
                        actual[index], expected[index]
                    ));
                }
                maximum_error = maximum_error.max(error);
            } else if actual[index].to_bits() != initial_output[index].to_bits() {
                return Err(format!(
                    "attention modified output padding at ({row}, {column})"
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
    fn fully_masked_rows_use_finite_zero_and_preserve_padding() {
        let query = [Bf16::from_f32(1.0).to_bits(); 2];
        let key = [Bf16::from_f32(1.0).to_bits(); 2];
        let value = [3.0, 4.0];
        let mask = [f32::NEG_INFINITY; 2];
        let initial = [-7.0, -7.0];
        let output = evaluate_reference_v1(
            &query,
            &key,
            &value,
            &mask,
            &initial,
            ReferenceLayoutV1 {
                batch_heads: 1,
                queries: 1,
                query_rows_padded: 1,
                keys: 2,
                keys_padded: 2,
                depth: 1,
                value_dimension: 1,
                query_stride: 2,
                key_depth_stride: 2,
                key_head_stride: 2,
                value_stride: 1,
                value_head_stride: 2,
                mask_stride: 2,
                output_stride: 2,
                scale: 1.0,
            },
        )
        .unwrap();
        assert_eq!(output, [0.0, -7.0]);
    }

    #[test]
    fn finite_additive_mask_changes_the_probability_distribution() {
        let query = [Bf16::from_f32(1.0).to_bits()];
        let key = [Bf16::from_f32(1.0).to_bits(), Bf16::from_f32(1.0).to_bits()];
        let value = [2.0, 6.0];
        let mask = [0.0, (3.0_f32).ln()];
        let output = evaluate_reference_v1(
            &query,
            &key,
            &value,
            &mask,
            &[0.0],
            ReferenceLayoutV1 {
                batch_heads: 1,
                queries: 1,
                query_rows_padded: 1,
                keys: 2,
                keys_padded: 2,
                depth: 1,
                value_dimension: 1,
                query_stride: 1,
                key_depth_stride: 2,
                key_head_stride: 2,
                value_stride: 1,
                value_head_stride: 2,
                mask_stride: 2,
                output_stride: 1,
                scale: 1.0,
            },
        )
        .unwrap();
        assert!((output[0] - 5.0).abs() < 1.0e-6);
    }

    fn output_check_layout() -> ReferenceLayoutV1 {
        ReferenceLayoutV1 {
            batch_heads: 1,
            queries: 1,
            query_rows_padded: 1,
            keys: 1,
            keys_padded: 1,
            depth: 1,
            value_dimension: 2,
            query_stride: 1,
            key_depth_stride: 1,
            key_head_stride: 1,
            value_stride: 2,
            value_head_stride: 2,
            mask_stride: 1,
            output_stride: 3,
            scale: 1.0,
        }
    }

    #[test]
    fn output_validation_rejects_nan_at_each_logical_position_and_padding_damage() {
        let layout = output_check_layout();
        let expected = [1.25, -2.5, 37.0];
        assert_eq!(
            validate_output_v1(&expected, &expected, &expected, layout),
            Ok(0.0)
        );
        for index in 0..2 {
            let mut actual = expected;
            actual[index] = f32::NAN;
            assert!(validate_output_v1(&actual, &expected, &expected, layout).is_err());
        }
        let mut actual = expected;
        actual[2] = 38.0;
        assert!(validate_output_v1(&actual, &expected, &expected, layout).is_err());
        actual[2] = f32::NAN;
        assert!(validate_output_v1(&actual, &expected, &expected, layout).is_err());
    }

    #[test]
    fn output_validation_rejects_nonfinite_expected_error_and_shape() {
        let layout = output_check_layout();
        let expected = [1.25, -2.5, 37.0];
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = expected;
            bad[0] = nonfinite;
            assert!(validate_output_v1(&bad, &expected, &expected, layout).is_err());
            assert!(validate_output_v1(&expected, &bad, &expected, layout).is_err());
        }
        let actual = [f32::MAX, -2.5, 37.0];
        let opposite = [-f32::MAX, -2.5, 37.0];
        assert!(validate_output_v1(&actual, &opposite, &expected, layout).is_err());
        assert!(validate_output_v1(&expected[..2], &expected, &expected, layout).is_err());
    }

    #[test]
    fn reference_excludes_nonfinite_logical_masks_without_reading_padding() {
        let query = [Bf16::from_f32(1.0).to_bits()];
        let key = [Bf16::from_f32(1.0).to_bits(); 2];
        let mut layout = output_check_layout();
        layout.keys = 2;
        layout.keys_padded = 2;
        layout.key_depth_stride = 2;
        layout.key_head_stride = 2;
        layout.value_dimension = 1;
        layout.value_stride = 1;
        layout.value_head_stride = 2;
        layout.mask_stride = 3;
        layout.output_stride = 2;
        for excluded in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let actual = evaluate_reference_v1(
                &query,
                &key,
                &[2.0, 6.0],
                &[0.0, excluded, f32::NAN],
                &[0.0, 37.0],
                layout,
            )
            .unwrap();
            assert_eq!(actual, [2.0, 37.0]);
            let fully_masked = evaluate_reference_v1(
                &query,
                &key,
                &[2.0, 6.0],
                &[excluded, excluded, f32::NAN],
                &[0.0, 37.0],
                layout,
            )
            .unwrap();
            assert_eq!(fully_masked, [0.0, 37.0]);
        }
    }

    #[test]
    fn padded_value_rows_do_not_contribute_even_when_nonfinite() {
        let query = [Bf16::from_f32(1.0).to_bits()];
        let key = [Bf16::from_f32(1.0).to_bits(); 2];
        let mut layout = output_check_layout();
        layout.keys_padded = 2;
        layout.key_depth_stride = 2;
        layout.key_head_stride = 2;
        layout.value_dimension = 1;
        layout.value_stride = 1;
        layout.value_head_stride = 2;
        layout.mask_stride = 2;
        layout.output_stride = 2;
        for padding in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let actual = evaluate_reference_v1(
                &query,
                &key,
                &[2.0, padding],
                &[0.0, f32::NAN],
                &[0.0, 37.0],
                layout,
            )
            .unwrap();
            assert_eq!(actual, [2.0, 37.0]);
        }
    }
}

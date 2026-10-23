use crate::protocol::{Argument, Output, Request, buffer, float_scalar, floats, scalar};

pub(crate) const READ: &str = "read_only";
pub(crate) const WRITE: &str = "read_write";

pub fn kernels() -> Vec<&'static str> {
    let mut names = vec![
        "gemm_autoresearch_v1",
        "tiled_gemm_general_v1",
        "simt_gemm_general_v1",
        "row_softmax_general_v1",
        "flash_attention_general_v1",
        "moe_grouped_expert_general_v1",
    ];
    names.extend(crate::reference_basic::KERNELS);
    names.extend(crate::reference_gpt::KERNELS);
    names.extend(crate::reference_attention::KERNELS);
    names.extend(crate::reference_low_precision::KERNELS);
    names.extend(crate::reference_systems::KERNELS);
    names.sort_unstable();
    names
}

pub(crate) fn data(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i % 13) as f32 * 0.125 - 0.5).collect()
}
fn bf16(count: usize) -> Argument {
    let bits: Vec<u8> = data(count)
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    buffer("u16", READ, &bits)
}
pub(crate) fn input(count: usize) -> Argument {
    floats(&data(count), READ)
}
pub(crate) fn output(count: usize) -> Argument {
    floats(&vec![-1234.5; count], WRITE)
}

pub fn generate(kernel: &str) -> Result<Request, String> {
    if crate::reference_basic::contains(kernel) {
        return crate::reference_basic::generate(kernel);
    }
    if crate::reference_gpt::contains(kernel) {
        return crate::reference_gpt::generate(kernel);
    }
    if crate::reference_attention::contains(kernel) {
        return crate::reference_attention::generate(kernel);
    }
    if crate::reference_low_precision::contains(kernel) {
        return crate::reference_low_precision::generate(kernel);
    }
    if crate::reference_systems::contains(kernel) {
        return crate::reference_systems::generate(kernel);
    }
    let (groups, arguments) = match kernel {
        "gemm_autoresearch_v1" | "tiled_gemm_general_v1" | "simt_gemm_general_v1" => {
            let mut arguments = vec![bf16(256), bf16(256), output(256)];
            arguments.extend([16, 16, 16, 16, 16, 16].map(scalar));
            arguments.extend([float_scalar(1.0), float_scalar(0.0)]);
            (
                if kernel == "simt_gemm_general_v1" {
                    4
                } else {
                    1
                },
                arguments,
            )
        }
        "row_softmax_general_v1" => {
            let mut arguments = vec![input(128), output(128)];
            arguments.extend([2, 64, 64, 64].map(scalar));
            (2, arguments)
        }
        "flash_attention_general_v1" => {
            let mut arguments = vec![
                bf16(256),
                bf16(256),
                input(256),
                floats(&vec![0.0; 256], READ),
                output(256),
            ];
            arguments
                .extend([1, 16, 16, 16, 16, 16, 16, 16, 16, 256, 16, 256, 16, 16, 16].map(scalar));
            arguments.push(float_scalar(0.25));
            (1, arguments)
        }
        "moe_grouped_expert_general_v1" => {
            let mut arguments = vec![
                bf16(256),
                bf16(512),
                floats(&vec![0.5; 16], READ),
                input(32),
                output(256),
            ];
            arguments.extend([16, 16, 16, 16, 16, 256, 16, 16, 1, 2].map(scalar));
            (1, arguments)
        }
        _ => return Err(format!("no reviewed current reference corpus for {kernel}")),
    };
    Ok(Request::new(kernel, 64, groups, arguments))
}

pub fn evaluate(request: &Request) -> Result<Vec<Output>, String> {
    if crate::reference_basic::contains(&request.kernel) {
        return crate::reference_basic::evaluate(request);
    }
    if crate::reference_gpt::contains(&request.kernel) {
        return crate::reference_gpt::evaluate(request);
    }
    if crate::reference_attention::contains(&request.kernel) {
        return crate::reference_attention::evaluate(request);
    }
    if crate::reference_low_precision::contains(&request.kernel) {
        return crate::reference_low_precision::evaluate(request);
    }
    if crate::reference_systems::contains(&request.kernel) {
        return crate::reference_systems::evaluate(request);
    }
    let expected = generate(&request.kernel)?;
    if matches!(
        request.kernel.as_str(),
        "tiled_gemm_general_v1" | "simt_gemm_general_v1"
    ) {
        request.corpus_shape(&expected)?;
    } else {
        request.launch(
            expected.workgroup[0],
            expected.grid[0] / u64::from(expected.workgroup[0]),
            expected.arguments.len(),
        )?;
    }
    let n = |index| -> Result<u32, String> {
        let value = request.scalar(index, "u32")?;
        if value > 4096 {
            return Err("reference scalar extent bound".to_owned());
        }
        Ok(value)
    };
    let f = |index| -> Result<f32, String> {
        let value = f32::from_bits(request.scalar(index, "f32")?);
        if value.is_finite() {
            Ok(value)
        } else {
            Err("nonfinite reference scale".to_owned())
        }
    };
    let exact_count = |expected| {
        if request.arguments.len() == expected {
            Ok(())
        } else {
            Err("reference ABI argument count".to_owned())
        }
    };
    let (index, result, tolerance) = match request.kernel.as_str() {
        "gemm_autoresearch_v1" => {
            use crate::host_cpu::gemm::{ReferenceProblemV1, evaluate_reference_v1};
            exact_count(11)?;
            let result = evaluate_reference_v1(
                &request.u16s(0)?,
                &request.u16s(1)?,
                &request.f32s(2)?,
                ReferenceProblemV1 {
                    rows: n(3)?,
                    columns: n(4)?,
                    reduction: n(5)?,
                    lhs_stride: n(6)?,
                    rhs_stride: n(7)?,
                    output_stride: n(8)?,
                    product_scale: f(9)?,
                    output_scale: f(10)?,
                },
            )
            .map_err(str::to_owned)?;
            (2, result, 0.00001)
        }
        "tiled_gemm_general_v1" | "simt_gemm_general_v1" => {
            use crate::host_cpu::tiled_gemm::{ReferenceProblemV1, evaluate_reference_v1};
            exact_count(11)?;
            let result = evaluate_reference_v1(
                &request.u16s(0)?,
                &request.u16s(1)?,
                &request.bf16_gemm_initial()?,
                ReferenceProblemV1 {
                    rows: n(3)?,
                    columns: n(4)?,
                    reduction: n(5)?,
                    lhs_stride: n(6)?,
                    rhs_stride: n(7)?,
                    output_stride: n(8)?,
                    product_scale: f(9)?,
                    output_scale: f(10)?,
                },
            )
            .map_err(str::to_owned)?;
            (2, result, 0.00001)
        }
        "row_softmax_general_v1" => {
            use crate::host_cpu::row_softmax::{ReferenceLayoutV1, evaluate_reference_v1};
            exact_count(6)?;
            let result = evaluate_reference_v1(
                &request.f32s(0)?,
                &request.f32s(1)?,
                ReferenceLayoutV1 {
                    rows: n(2)?,
                    columns: n(3)?,
                    input_stride: n(4)?,
                    output_stride: n(5)?,
                },
            )
            .map_err(str::to_owned)?;
            (1, result, 0.00001)
        }
        "flash_attention_general_v1" => {
            use crate::host_cpu::flash_attention::{ReferenceLayoutV1, evaluate_reference_v1};
            exact_count(21)?;
            if n(19)?
                != n(5)?
                    .checked_mul(n(7)?)
                    .ok_or("output row extent overflow")?
            {
                return Err("attention output rows do not match request".to_owned());
            }
            let result = evaluate_reference_v1(
                &request.u16s(0)?,
                &request.u16s(1)?,
                &request.f32s(2)?,
                &request.f32s(3)?,
                &request.f32s(4)?,
                ReferenceLayoutV1 {
                    batch_heads: n(5)?,
                    queries: n(6)?,
                    query_rows_padded: n(7)?,
                    keys: n(8)?,
                    keys_padded: n(9)?,
                    depth: n(10)?,
                    value_dimension: n(11)?,
                    query_stride: n(12)?,
                    key_depth_stride: n(13)?,
                    key_head_stride: n(14)?,
                    value_stride: n(15)?,
                    value_head_stride: n(16)?,
                    mask_stride: n(17)?,
                    output_stride: n(18)?,
                    scale: f(20)?,
                },
            )
            .map_err(str::to_owned)?;
            (4, result, 0.0001)
        }
        "moe_grouped_expert_general_v1" => {
            use crate::host_cpu::moe_grouped::{ReferenceLayoutV1, evaluate_reference_v1};
            exact_count(15)?;
            let result = evaluate_reference_v1(
                &request.u16s(0)?,
                &request.u16s(1)?,
                &request.f32s(2)?,
                &request.f32s(3)?,
                &request.f32s(4)?,
                ReferenceLayoutV1 {
                    rows_padded: n(5)?,
                    output_columns: n(6)?,
                    reduction: n(7)?,
                    token_stride: n(8)?,
                    weight_stride: n(9)?,
                    expert_weight_stride: n(10)?,
                    bias_stride: n(11)?,
                    output_stride: n(12)?,
                    expert: n(13)?,
                    experts: n(14)?,
                },
            )
            .map_err(str::to_owned)?;
            (4, result, 0.00001)
        }
        _ => {
            return Err(format!(
                "no independent request adapter for {}",
                request.kernel
            ));
        }
    };
    Ok(vec![Output::f32s(index, &result, tolerance)])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_dynamic_corpus_executes_existing_references_on_exact_requests() {
        for kernel in [
            "gemm_autoresearch_v1",
            "tiled_gemm_general_v1",
            "simt_gemm_general_v1",
            "row_softmax_general_v1",
            "flash_attention_general_v1",
            "moe_grouped_expert_general_v1",
        ] {
            let request = generate(kernel).unwrap();
            request.validate().unwrap();
            crate::protocol::Response::new(&request, [1; 32], evaluate(&request).unwrap()).unwrap();
        }
    }
    #[test]
    fn reference_rejects_unknown_kernels_and_wrong_actual_inputs() {
        assert!(generate("unknown").is_err());
        let mut request = generate("row_softmax_general_v1").unwrap();
        request.arguments[0] = input(1);
        assert!(evaluate(&request).is_err());
        request.arguments.pop();
        assert!(evaluate(&request).is_err());
    }
}

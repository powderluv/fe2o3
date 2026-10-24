use std::io;

use fe2o3_core::{DeviceBuffer, GpuContext, KernelParams, LaunchConfig, launch_kernel_on_stream};
use fe2o3_device::Bf16;
use fe2o3_moe_grouped_expert_general_v1::reference::{
    ReferenceLayoutV1, evaluate_reference_v1, validate_output_v1,
};

const HSACO_ENV: &str = "FE2O3_MOE_EXPERT_HSACO";
const KERNEL: &str = fe2o3_moe_grouped_expert_general_v1::MOE_EXPERT_KERNEL_NAME_V1;

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn checked_len(rows: u32, stride: u32, name: &'static str) -> Result<usize, io::Error> {
    usize::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(stride as usize))
        .ok_or_else(|| invalid_input(name))
}

fn qualification_target_matches(requested: &str, observed: &str, wave: i32) -> bool {
    matches!(requested, "gfx942" | "gfx950") && requested == observed && wave == 64
}

fn same_f32_bits(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layout = ReferenceLayoutV1 {
        rows_padded: 32,
        output_columns: 21,
        reduction: 32,
        token_stride: 37,
        weight_stride: 25,
        expert_weight_stride: 32 * 25 + 7,
        bias_stride: 29,
        output_stride: 31,
        expert: 2,
        experts: 4,
    };
    let token_len = checked_len(
        layout.rows_padded,
        layout.token_stride,
        "routed-token extent overflow",
    )?;
    let weight_len = checked_len(
        layout.experts,
        layout.expert_weight_stride,
        "expert-weight extent overflow",
    )?;
    let bias_len = checked_len(layout.experts, layout.bias_stride, "bias extent overflow")?;
    let output_len = checked_len(
        layout.rows_padded,
        layout.output_stride,
        "routed-output extent overflow",
    )?;

    let routed_tokens = (0..token_len)
        .map(|index| Bf16::from_f32((index % 17) as f32 * 0.03125 - 0.25).to_bits())
        .collect::<Vec<_>>();
    let expert_weights = (0..weight_len)
        .map(|index| Bf16::from_f32((index % 19) as f32 * 0.025 - 0.225).to_bits())
        .collect::<Vec<_>>();
    let route_gates = (0..layout.rows_padded as usize)
        .map(|row| {
            if row >= 19 {
                0.0
            } else {
                0.25 + (row % 5) as f32 * 0.125
            }
        })
        .collect::<Vec<_>>();
    let expert_bias = (0..bias_len)
        .map(|index| (index % 11) as f32 * 0.02 - 0.1)
        .collect::<Vec<_>>();
    let mut initial_output = (0..output_len)
        .map(|index| 3000.0 + index as f32)
        .collect::<Vec<_>>();
    initial_output.extend_from_slice(&[f32::from_bits(0x7f80_0001); 8]);
    let expected = evaluate_reference_v1(
        &routed_tokens,
        &expert_weights,
        &route_gates,
        &expert_bias,
        &initial_output,
        layout,
    )
    .map_err(invalid_input)?;

    let requested = match std::env::var("FE2O3_MOE_EXPERT_TARGET") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "gfx942".into(),
        Err(error) => return Err(error.into()),
    };
    if !matches!(requested.as_str(), "gfx942" | "gfx950") {
        return Err(invalid_input("unsupported requested MoE qualification target").into());
    }
    let context = GpuContext::new(0)?;
    let observed = context.observe_target()?;
    if !qualification_target_matches(
        &requested,
        observed.target_id().processor(),
        observed.hip_default_warp_size() as i32,
    ) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "qualification requires the requested gfx942/gfx950 wave64 device",
        )
        .into());
    }
    let stream = context.create_stream()?;
    let routed_tokens_device = DeviceBuffer::from_host(&stream, &routed_tokens)?;
    let expert_weights_device = DeviceBuffer::from_host(&stream, &expert_weights)?;
    let route_gates_device = DeviceBuffer::from_host(&stream, &route_gates)?;
    let expert_bias_device = DeviceBuffer::from_host(&stream, &expert_bias)?;
    let routed_output_device = DeviceBuffer::from_host(&stream, &initial_output)?;
    let hsaco = std::env::var_os(HSACO_ENV).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("{HSACO_ENV} is not set"))
    })?;

    // SAFETY: the qualification runner compiles this exact symbol for the requested observed target. The argument
    // order and scalar widths mirror the compiler-emitted 120-byte explicit kernarg ABI, and
    // all allocations remain live through the synchronous stream completion.
    unsafe {
        let module = context.load_module_from_file_unchecked(hsaco)?;
        let function = module.load_function(KERNEL)?;
        let mut arguments = KernelParams::new();
        arguments.push(routed_tokens_device.as_device_ptr());
        arguments.push(routed_tokens_device.len());
        arguments.push(expert_weights_device.as_device_ptr());
        arguments.push(expert_weights_device.len());
        arguments.push(route_gates_device.as_device_ptr());
        arguments.push(route_gates_device.len());
        arguments.push(expert_bias_device.as_device_ptr());
        arguments.push(expert_bias_device.len());
        arguments.push(routed_output_device.as_device_ptr());
        arguments.push(routed_output_device.len());
        arguments.push(layout.rows_padded);
        arguments.push(layout.output_columns);
        arguments.push(layout.reduction);
        arguments.push(layout.token_stride);
        arguments.push(layout.weight_stride);
        arguments.push(layout.expert_weight_stride);
        arguments.push(layout.bias_stride);
        arguments.push(layout.output_stride);
        arguments.push(layout.expert);
        arguments.push(layout.experts);
        let tile_rows = layout.rows_padded / 16;
        let tile_columns = layout.output_columns.div_ceil(16);
        launch_kernel_on_stream(
            &function,
            LaunchConfig {
                grid_dim: (tile_rows * tile_columns, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            &stream,
            &mut arguments,
        )?;
        stream.synchronize()?;
    }

    if routed_tokens_device.to_host_vec(&stream)? != routed_tokens
        || expert_weights_device.to_host_vec(&stream)? != expert_weights
        || !same_f32_bits(&route_gates_device.to_host_vec(&stream)?, &route_gates)
        || !same_f32_bits(&expert_bias_device.to_host_vec(&stream)?, &expert_bias)
    {
        return Err(io::Error::other("MoE modified immutable input buffers").into());
    }
    let actual = routed_output_device.to_host_vec(&stream)?;
    if actual.len() != initial_output.len()
        || !same_f32_bits(&actual[output_len..], &initial_output[output_len..])
    {
        return Err(io::Error::other("MoE modified trailing output canaries").into());
    }
    let maximum_error = validate_output_v1(
        &actual[..output_len],
        &expected[..output_len],
        &initial_output[..output_len],
        layout,
    )
    .map_err(io::Error::other)?;
    println!(
        "PASS {KERNEL} target={requested}: expert {}, {}x{}x{}, {} workgroups, max_abs_error={maximum_error:.6}",
        layout.expert,
        layout.rows_padded,
        layout.output_columns,
        layout.reduction,
        (layout.rows_padded / 16) * layout.output_columns.div_ceil(16),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_target_must_be_supported_observed_and_wave64() {
        for target in ["gfx942", "gfx950"] {
            assert!(qualification_target_matches(target, target, 64));
            assert!(!qualification_target_matches(target, target, 32));
        }
        assert!(!qualification_target_matches("gfx942", "gfx950", 64));
        assert!(!qualification_target_matches("gfx950", "gfx942", 64));
        assert!(!qualification_target_matches("gfx1100", "gfx1100", 64));
        assert!(!qualification_target_matches("", "gfx950", 64));
    }

    #[test]
    fn immutable_floats_and_nan_canaries_are_compared_bitwise_and_completely() {
        let canary = f32::from_bits(0x7f80_0001);
        assert!(same_f32_bits(&[canary, -0.0], &[canary, -0.0]));
        assert!(!same_f32_bits(&[canary, -0.0], &[canary, 0.0]));
        assert!(!same_f32_bits(&[canary], &[f32::from_bits(0x7f80_0002)]));
        assert!(!same_f32_bits(&[canary], &[]));
        assert!(!same_f32_bits(&[], &[canary]));
        assert!(!same_f32_bits(&[1.0], &[2.0]));
    }
}

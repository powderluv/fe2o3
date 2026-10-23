use std::io;

use fe2o3_core::{DeviceBuffer, GpuContext, KernelParams, LaunchConfig, launch_kernel_on_stream};
use fe2o3_device::Bf16;
use fe2o3_flash_attention_general_v1::reference::{
    ReferenceLayoutV1, evaluate_reference_v1, validate_output_v1,
};

const HSACO_ENV: &str = "FE2O3_FLASH_ATTENTION_HSACO";
const KERNEL: &str = fe2o3_flash_attention_general_v1::FLASH_ATTENTION_KERNEL_NAME_V1;

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn qualification_target_matches(expected: &str, processor: &str, warp_size: u32) -> bool {
    matches!(expected, "gfx942" | "gfx950") && processor == expected && warp_size == 64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expected_target =
        std::env::var("FE2O3_FLASH_ATTENTION_TARGET").unwrap_or_else(|_| "gfx942".to_owned());
    if !matches!(expected_target.as_str(), "gfx942" | "gfx950") {
        return Err(invalid_input("unsupported Flash qualification target").into());
    }
    let layout = ReferenceLayoutV1 {
        batch_heads: 2,
        queries: 19,
        query_rows_padded: 32,
        keys: 21,
        keys_padded: 32,
        depth: 23,
        value_dimension: 13,
        query_stride: 27,
        key_depth_stride: 37,
        key_head_stride: 23 * 37 + 5,
        value_stride: 17,
        value_head_stride: 32 * 17 + 7,
        mask_stride: 37,
        output_stride: 19,
        scale: 0.2,
    };
    let output_rows = layout
        .batch_heads
        .checked_mul(layout.query_rows_padded)
        .ok_or_else(|| invalid_input("output row extent overflow"))?;
    let query_len = output_rows as usize * layout.query_stride as usize;
    let key_len = layout.batch_heads as usize * layout.key_head_stride as usize;
    let value_len = layout.batch_heads as usize * layout.value_head_stride as usize;
    let mask_len = output_rows as usize * layout.mask_stride as usize;
    let output_len = output_rows as usize * layout.output_stride as usize;

    let query = (0..query_len)
        .map(|index| Bf16::from_f32((index % 17) as f32 * 0.0625 - 0.5).to_bits())
        .collect::<Vec<_>>();
    let key = (0..key_len)
        .map(|index| Bf16::from_f32((index % 19) as f32 * 0.05 - 0.45).to_bits())
        .collect::<Vec<_>>();
    let value = (0..value_len)
        .map(|index| (index % 23) as f32 * 0.03125 - 0.25)
        .collect::<Vec<_>>();
    let mut mask = vec![f32::NAN; mask_len];
    for head in 0..layout.batch_heads as usize {
        for row in 0..layout.query_rows_padded as usize {
            let global_row = head * layout.query_rows_padded as usize + row;
            for key_index in 0..layout.keys as usize {
                let is_fully_masked = head == 1 && row == 3;
                let is_visible = row < layout.queries as usize && key_index <= row + 4;
                mask[global_row * layout.mask_stride as usize + key_index] =
                    if is_visible && !is_fully_masked {
                        (key_index % 5) as f32 * 0.03125 - 0.0625
                    } else {
                        f32::NEG_INFINITY
                    };
            }
        }
    }
    let initial_output = (0..output_len)
        .map(|index| 2000.0 + index as f32)
        .collect::<Vec<_>>();
    let expected = evaluate_reference_v1(&query, &key, &value, &mask, &initial_output, layout)
        .map_err(invalid_input)?;

    let context = GpuContext::new(0)?;
    let observed = context.observe_target()?;
    if !qualification_target_matches(
        &expected_target,
        observed.target_id().processor(),
        observed.hip_default_warp_size(),
    ) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "qualification requires the declared gfx942 or gfx950 wave64 device",
        )
        .into());
    }
    let stream = context.create_stream()?;
    let query_device = DeviceBuffer::from_host(&stream, &query)?;
    let key_device = DeviceBuffer::from_host(&stream, &key)?;
    let value_device = DeviceBuffer::from_host(&stream, &value)?;
    let mask_device = DeviceBuffer::from_host(&stream, &mask)?;
    let output_device = DeviceBuffer::from_host(&stream, &initial_output)?;
    let hsaco = std::env::var_os(HSACO_ENV).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("{HSACO_ENV} is not set"))
    })?;

    // SAFETY: the qualification runner compiled this exact symbol for the declared target. The argument
    // order and widths mirror the compiler-emitted 144-byte explicit kernarg ABI, and every
    // allocation remains live through synchronous stream completion.
    unsafe {
        let module = context.load_module_from_file_unchecked(hsaco)?;
        let function = module.load_function(KERNEL)?;
        let mut arguments = KernelParams::new();
        arguments.push(query_device.as_device_ptr());
        arguments.push(query_device.len());
        arguments.push(key_device.as_device_ptr());
        arguments.push(key_device.len());
        arguments.push(value_device.as_device_ptr());
        arguments.push(value_device.len());
        arguments.push(mask_device.as_device_ptr());
        arguments.push(mask_device.len());
        arguments.push(output_device.as_device_ptr());
        arguments.push(output_device.len());
        arguments.push(layout.batch_heads);
        arguments.push(layout.queries);
        arguments.push(layout.query_rows_padded);
        arguments.push(layout.keys);
        arguments.push(layout.keys_padded);
        arguments.push(layout.depth);
        arguments.push(layout.value_dimension);
        arguments.push(layout.query_stride);
        arguments.push(layout.key_depth_stride);
        arguments.push(layout.key_head_stride);
        arguments.push(layout.value_stride);
        arguments.push(layout.value_head_stride);
        arguments.push(layout.mask_stride);
        arguments.push(layout.output_stride);
        arguments.push(output_rows);
        arguments.push(layout.scale);
        let workgroups = layout.batch_heads * (layout.query_rows_padded / 16);
        launch_kernel_on_stream(
            &function,
            LaunchConfig {
                grid_dim: (workgroups, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            &stream,
            &mut arguments,
        )?;
        stream.synchronize()?;
    }

    let actual = output_device.to_host_vec(&stream)?;
    let maximum_error = validate_output_v1(&actual, &expected, &initial_output, layout)
        .map_err(io::Error::other)?;
    println!(
        "PASS {KERNEL}: {} heads, {}x{} logical attention, depth {}, value dimension {}, {} workgroups, max_abs_error={maximum_error:.6}",
        layout.batch_heads,
        layout.queries,
        layout.keys,
        layout.depth,
        layout.value_dimension,
        layout.batch_heads * (layout.query_rows_padded / 16),
    );
    Ok(())
}

#[cfg(test)]
mod target_tests {
    use super::qualification_target_matches;

    #[test]
    fn accepts_exact_supported_wave64_target() {
        for target in ["gfx942", "gfx950"] {
            assert!(qualification_target_matches(target, target, 64));
        }
    }

    #[test]
    fn rejects_wrong_processor_wave_or_unsupported_target() {
        assert!(!qualification_target_matches("gfx942", "gfx950", 64));
        assert!(!qualification_target_matches("gfx950", "gfx942", 64));
        assert!(!qualification_target_matches("gfx950", "gfx950", 32));
        assert!(!qualification_target_matches("gfx1100", "gfx1100", 64));
    }
}

//! Explicit external-HSACO numerical qualification, using the same deprecated
//! HIP surface as the checked-in general BF16/Flash examples. No Worker V3
//! publication, protected execution, proof, or artifact-currentness authority.
use fe2o3_core::{DeviceBuffer, GpuContext, KernelParams, LaunchConfig, launch_kernel_on_stream};
use fe2o3_row_softmax_general_v1::{
    reference::{ReferenceLayoutV1, evaluate_reference_v1},
    target,
};
use std::{error::Error, io};

const GUARD: usize = 8;
const CANARY: f32 = f32::from_bits(0x7fc1_abcd);
const KERNEL: &str = "row_softmax_general_v1";
// Two (pointer, usize length) pairs followed by four u32 dimensions/strides.
const EXPLICIT_KERNARG_BYTES: usize = 4 * size_of::<usize>() + 4 * size_of::<u32>();
const TARGET: &str = if cfg!(feature = "gfx950-target") {
    "gfx950"
} else {
    "gfx942"
};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
fn guarded(body: &[f32]) -> Vec<f32> {
    let mut allocation = vec![CANARY; GUARD];
    allocation.extend_from_slice(body);
    allocation.extend_from_slice(&[CANARY; GUARD]);
    allocation
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
struct Case {
    name: String,
    layout: ReferenceLayoutV1,
    input: Vec<f32>,
    initial: Vec<f32>,
}
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for rows in [0, 1, 3] {
        for columns in [1, 3, 63, 64, 65, 127, 129, 4095, 4096] {
            let layout = ReferenceLayoutV1 {
                rows,
                columns,
                input_stride: columns + 7,
                output_stride: columns + 11,
            };
            let mut input = vec![CANARY; (rows * layout.input_stride) as usize];
            for row in 0..rows as usize {
                for col in 0..columns as usize {
                    input[row * layout.input_stride as usize + col] =
                        (((row * 131 + col * 37 + 11) % 101) as f32 - 50.0) / 13.7;
                }
            }
            cases.push(Case {
                name: format!("nonquarter-{rows}x{columns}"),
                layout,
                input,
                initial: vec![CANARY; (rows * layout.output_stride) as usize],
            });
        }
    }
    let layout = ReferenceLayoutV1 {
        rows: 2,
        columns: 65,
        input_stride: 71,
        output_stride: 79,
    };
    let mut input = vec![CANARY; 2 * 71];
    for row in 0..2 {
        for col in 0..65 {
            input[row * 71 + col] = 1000.0 + col as f32 / 7.3 - row as f32 * 2000.0;
        }
    }
    cases.push(Case {
        name: "large-logits-max-subtraction".into(),
        layout,
        input,
        initial: vec![CANARY; 2 * 79],
    });
    cases
}
fn expected(case: &Case) -> Result<Vec<f32>, io::Error> {
    evaluate_reference_v1(&case.input, &case.initial, case.layout)
        .map(|x| guarded(&x))
        .map_err(invalid)
}
fn validate_output(case: &Case, actual: &[f32], expected: &[f32]) -> Result<f32, io::Error> {
    if actual.len() != expected.len() {
        return Err(invalid("output allocation length changed"));
    }
    let mut maximum_error = 0.0_f32;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let logical = index
            .checked_sub(GUARD)
            .filter(|&i| i < case.initial.len())
            .is_some_and(|i| {
                i % (case.layout.output_stride as usize) < case.layout.columns as usize
            });
        if logical {
            let error = (a - e).abs();
            if !a.is_finite()
                || !e.is_finite()
                || !error.is_finite()
                || !(0.0..=1.0).contains(&a)
                || error > 5.0e-5 * e.abs().max(1.0e-8)
            {
                return Err(invalid(format!(
                    "{} softmax mismatch at allocation slot {index}: {a} expected {e}",
                    case.name
                )));
            }
            maximum_error = maximum_error.max(error);
        } else if a.to_bits() != e.to_bits() {
            return Err(invalid(format!(
                "{} changed output padding/canary at allocation slot {index}",
                case.name
            )));
        }
    }
    for row in 0..case.layout.rows as usize {
        let start = GUARD + row * case.layout.output_stride as usize;
        let sum: f64 = actual[start..start + case.layout.columns as usize]
            .iter()
            .map(|&x| x as f64)
            .sum();
        if (sum - 1.0).abs() > 5.0e-5 {
            return Err(invalid(format!(
                "{} row {row} does not normalize",
                case.name
            )));
        }
    }
    Ok(maximum_error)
}

pub fn run() -> Result<(), Box<dyn Error>> {
    if size_of::<usize>() != 8 || EXPLICIT_KERNARG_BYTES != 48 {
        return Err(
            invalid("qualification requires the emitted 48-byte 64-bit kernarg ABI").into(),
        );
    }
    let path = std::env::var_os("FE2O3_ROW_SOFTMAX_HSACO").ok_or_else(|| {
        invalid("FE2O3_ROW_SOFTMAX_HSACO is required; use the documented qualification runner")
    })?;
    let context = GpuContext::new(0)?;
    let observed = context.observe_target()?;
    if !target::qualification_target_matches(
        TARGET,
        &observed.target_id().to_string(),
        observed.hip_default_warp_size(),
    ) {
        return Err(invalid(format!(
            "qualification requires observed {TARGET}:xnack- Wave64"
        ))
        .into());
    }
    let stream = context.create_stream()?;
    // SAFETY: this opt-in qualification program trusts the external HSACO just
    // compiled and inspected by its local runner. It grants no protected or
    // production loading authority. The runner checks symbol, 48-byte explicit
    // ABI, target, Wave64, workgroup size and exact artifact hashes before load.
    let module = unsafe { context.load_module_from_file_unchecked(path)? };
    let function = module.load_function(KERNEL)?;
    let mut dispatched = 0;
    for case in cases() {
        let expected = expected(&case)?;
        let input = guarded(&case.input);
        let initial = guarded(&case.initial);
        let input_device = DeviceBuffer::from_host(&stream, &input)?;
        let mut output_device = DeviceBuffer::from_host(&stream, &initial)?;
        let groups = target::gfx950_workgroups(
            case.layout.rows,
            case.layout.columns,
            cfg!(feature = "structured-tile"),
        )
        .ok_or_else(|| invalid("softmax qualification grid overflow"))?;
        if groups != 0 {
            if groups > observed.max_grid_dimensions()[0]
                || observed.max_threads_per_block() < 64
                || observed.max_block_dimensions()[0] < 64
            {
                return Err(invalid("softmax launch exceeds observed device limits").into());
            }
            let input_view = input_device.view(GUARD..GUARD + case.input.len())?;
            let output_view = output_device.view_mut(GUARD..GUARD + case.initial.len())?;
            let mut args = KernelParams::new();
            args.push(input_view.as_device_ptr());
            args.push(input_view.len());
            args.push(output_view.as_device_ptr());
            args.push(output_view.len());
            args.push(case.layout.rows);
            args.push(case.layout.columns);
            args.push(case.layout.input_stride);
            args.push(case.layout.output_stride);
            // SAFETY: two distinct live F32 allocations, read-only input and
            // exclusive output view; eight owned parameters match the emitted
            // 48-byte ABI. The grid follows this mode's checked tile mapping.
            // Keep module, parameters, buffers and stream alive until completion.
            let submitted = unsafe {
                launch_kernel_on_stream(
                    &function,
                    LaunchConfig {
                        grid_dim: (groups, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &stream,
                    &mut args,
                )
            };
            if let Err(error) = stream.synchronize() {
                eprintln!("qualification completion is ambiguous: {error}");
                // Do not run buffer/module destructors while work may be live.
                std::process::abort();
            }
            submitted?;
            dispatched += 1;
        }
        if !same_bits(&input_device.to_host_vec(&stream)?, &input) {
            return Err(invalid(format!("{} modified input allocation bits", case.name)).into());
        }
        let error = validate_output(&case, &output_device.to_host_vec(&stream)?, &expected)?;
        println!(
            "{} {KERNEL} case={} target={TARGET} groups={groups} max_abs_error={error:.8}",
            if groups == 0 {
                "NO DISPATCH (legacy HIP)"
            } else {
                "NUMERICAL PASS (legacy HIP)"
            },
            case.name
        );
    }
    println!(
        "NUMERICAL QUALIFICATION PASS (legacy HIP): {KERNEL} dispatched={dispatched}; production Worker V3/protected authority remains unestablished"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_strided_case_matrix_and_oracles_keep_canaries() {
        let all = cases();
        assert_eq!(all.len(), 28);
        assert!(all.iter().any(|c| c.layout.rows == 0));
        assert!(all.iter().any(|c| c.layout.columns == 4095));
        for case in &all {
            let want = expected(case).unwrap();
            assert_eq!(validate_output(case, &want, &want).unwrap(), 0.0);
        }
        assert_eq!(EXPLICIT_KERNARG_BYTES, 48);
    }
    #[test]
    fn nonfinite_logical_results_and_any_guard_write_reject() {
        let case = cases()
            .into_iter()
            .find(|c| c.layout.rows == 3 && c.layout.columns == 65)
            .unwrap();
        let want = expected(&case).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut got = want.clone();
            got[GUARD] = value;
            assert!(validate_output(&case, &got, &want).is_err());
        }
        for index in [0, GUARD + 65, want.len() - 1] {
            let mut got = want.clone();
            got[index] = 0.0;
            assert!(validate_output(&case, &got, &want).is_err());
        }
        assert!(!same_bits(&[0.0], &[-0.0]));
        assert!(same_bits(&[CANARY], &[CANARY]));
    }
}

//! Explicit external-HSACO legacy-HIP numerical qualification. This retains
//! the separate production Worker V3 boundary and creates no proof/receipt.
use fe2o3_core::{DeviceBuffer, GpuContext, KernelParams, LaunchConfig, launch_kernel_on_stream};
use fe2o3_scalar_gemm_v1::{
    harness::{HARDWARE_CASES, Shape, scalar_gemm_oracle},
    target,
};
use std::{error::Error, io};

const GUARD: usize = 8;
const CANARY: f32 = f32::from_bits(0x7fc1_abcd);
const KERNEL: &str = "scalar_gemm_v1";
// Three (pointer, usize length) pairs followed by three u32 dimensions.
// Use HIP's owned parameter list, not an invented padded repr(C) struct.
const EXPLICIT_KERNARG_BYTES: usize = 6 * size_of::<usize>() + 3 * size_of::<u32>();
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
    shape: Shape,
    a: Vec<f32>,
    b: Vec<f32>,
}
fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    // Preserve the complete retained boundary/FMA-distinguishing matrix.
    for case in HARDWARE_CASES {
        let shape = case.shape().expect("retained qualification shape");
        let (a, b) = case.inputs(shape);
        all.push(Case {
            name: case.name.into(),
            shape,
            a,
            b,
        });
    }
    // Full nonquarter values at scalar, rectangular and workgroup-tail sizes.
    for (m, n, k) in [
        (1, 1, 1),
        (2, 3, 4),
        (7, 5, 9),
        (1, 255, 7),
        (1, 256, 7),
        (1, 257, 7),
    ] {
        let shape = Shape::checked(m, n, k).unwrap();
        let a = (0..shape.a_len)
            .map(|i| ((i * 17 % 71) as f32 - 35.0) / 11.3)
            .collect();
        let b = (0..shape.b_len)
            .map(|i| ((i * 23 % 79) as f32 - 39.0) / 17.9)
            .collect();
        all.push(Case {
            name: format!("nonquarter-{m}x{n}x{k}"),
            shape,
            a,
            b,
        });
    }
    all
}
fn expected(case: &Case) -> Vec<f32> {
    guarded(&scalar_gemm_oracle(case.shape, &case.a, &case.b))
}
fn validate_output(case: &Case, actual: &[f32], expected: &[f32]) -> Result<(), io::Error> {
    if actual.len() != expected.len() {
        return Err(invalid("GEMM output allocation length changed"));
    }
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let logical = index >= GUARD && index < GUARD + case.shape.c_len;
        if logical && (!a.is_finite() || !e.is_finite()) {
            return Err(invalid(format!(
                "{} nonfinite GEMM output at {index}",
                case.name
            )));
        }
        // Strict ordered F32 recurrence includes positive zero for K=0 and
        // the retained FMA-distinguishing case, so compare every output bit.
        if a.to_bits() != e.to_bits() {
            return Err(invalid(format!(
                "{} {} at allocation slot {index}: {a} expected {e}",
                case.name,
                if logical {
                    "GEMM mismatch"
                } else {
                    "output canary modified"
                }
            )));
        }
    }
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    if size_of::<usize>() != 8 || EXPLICIT_KERNARG_BYTES != 60 {
        return Err(
            invalid("qualification requires the emitted 60-byte 64-bit kernarg ABI").into(),
        );
    }
    let path = std::env::var_os("FE2O3_SCALAR_GEMM_HSACO").ok_or_else(|| {
        invalid("FE2O3_SCALAR_GEMM_HSACO is required; use the documented qualification runner")
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
    // SAFETY: the explicit qualification opt-in trusts this external HSACO
    // just compiled and inspected by its local runner. The runner checks the
    // actual symbol/60-byte ABI/target/Wave64/256-thread metadata and hashes.
    // This is the documented qualification escape hatch, not Worker V3 load.
    let module = unsafe { context.load_module_from_file_unchecked(path)? };
    let function = module.load_function(KERNEL)?;
    let mut dispatched = 0;
    for case in cases() {
        let expected = expected(&case);
        let a = guarded(&case.a);
        let b = guarded(&case.b);
        let initial = vec![CANARY; GUARD + case.shape.c_len + GUARD];
        let a_device = DeviceBuffer::from_host(&stream, &a)?;
        let b_device = DeviceBuffer::from_host(&stream, &b)?;
        let mut c_device = DeviceBuffer::from_host(&stream, &initial)?;
        let [m, n, k] = case.shape.dimensions();
        let groups = target::gfx950_workgroups(m, n, k)?;
        if let Some(groups) = groups {
            if groups > observed.max_grid_dimensions()[0]
                || observed.max_threads_per_block() < 256
                || observed.max_block_dimensions()[0] < 256
            {
                return Err(invalid("GEMM launch exceeds observed device limits").into());
            }
            let av = a_device.view(GUARD..GUARD + case.a.len())?;
            let bv = b_device.view(GUARD..GUARD + case.b.len())?;
            let cv = c_device.view_mut(GUARD..GUARD + case.shape.c_len)?;
            let mut args = KernelParams::new();
            args.push(av.as_device_ptr());
            args.push(av.len());
            args.push(bv.as_device_ptr());
            args.push(bv.len());
            args.push(cv.as_device_ptr());
            args.push(cv.len());
            args.push(m);
            args.push(n);
            args.push(k);
            // SAFETY: disjoint live F32 allocations, exact checked extents,
            // nine owned ABI fields, original rounded 256-thread grid, and
            // no output/input alias. The module, arguments and allocations
            // remain live through synchronization even when enqueue fails.
            let submitted = unsafe {
                launch_kernel_on_stream(
                    &function,
                    LaunchConfig {
                        grid_dim: (groups, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &stream,
                    &mut args,
                )
            };
            if let Err(error) = stream.synchronize() {
                eprintln!("qualification completion is ambiguous: {error}");
                // Never release allocations/module while a launch may be live.
                std::process::abort();
            }
            submitted?;
            dispatched += 1;
        }
        if !same_bits(&a_device.to_host_vec(&stream)?, &a)
            || !same_bits(&b_device.to_host_vec(&stream)?, &b)
        {
            return Err(invalid(format!(
                "{} modified immutable input allocation bits",
                case.name
            ))
            .into());
        }
        validate_output(&case, &c_device.to_host_vec(&stream)?, &expected)?;
        println!(
            "{} {KERNEL} case={} target={TARGET} shape={m}x{n}x{k} groups={}",
            if groups.is_none() {
                "NO DISPATCH (legacy HIP)"
            } else {
                "NUMERICAL PASS (legacy HIP)"
            },
            case.name,
            groups.unwrap_or(0)
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
    fn complete_original_and_nonquarter_matrix_keeps_ordered_oracles() {
        let all = cases();
        assert_eq!(all.len(), HARDWARE_CASES.len() + 6);
        assert!(all.iter().any(|c| c.name == "fma-distinguishing"));
        assert!(all.iter().any(|c| c.name == "zero-k"));
        assert_eq!(EXPLICIT_KERNARG_BYTES, 60);
        for case in &all {
            let want = expected(case);
            validate_output(case, &want, &want).unwrap();
        }
    }
    #[test]
    fn output_guard_and_nonfinite_or_signed_zero_substitutions_reject() {
        let all = cases();
        let case = all.iter().find(|c| c.name == "zero-k").unwrap();
        let want = expected(case);
        assert_eq!(want[GUARD].to_bits(), 0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0] {
            let mut got = want.clone();
            got[GUARD] = value;
            assert!(validate_output(case, &got, &want).is_err());
        }
        for index in [0, want.len() - 1] {
            let mut got = want.clone();
            got[index] = 0.0;
            assert!(validate_output(case, &got, &want).is_err());
        }
        assert!(same_bits(&[CANARY], &[CANARY]));
        assert!(!same_bits(&[0.0], &[-0.0]));
    }
}

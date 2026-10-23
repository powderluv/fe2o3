//! Explicit external-HSACO legacy-HIP numerical qualification. This retains
//! the separate production Worker V3 boundary and creates no proof/receipt.
use fe2o3_core::{DeviceBuffer, GpuContext, KernelParams, LaunchConfig, launch_kernel_on_stream};
use fe2o3_vecadd::target;
use std::{error::Error, io};

const GUARD: usize = 8;
const CANARY: f32 = f32::from_bits(0x7fc1_abcd);
const KERNEL: &str = "vecadd";
// Three (pointer, usize length) pairs.
// Use HIP's owned parameter list, not an invented padded repr(C) struct.
const EXPLICIT_KERNARG_BYTES: usize = 6 * size_of::<usize>();
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
#[path = "qualification_fixture.rs"]
mod qualification_fixture;
use qualification_fixture::{Case, cases};

fn expected(case: &Case) -> Vec<f32> {
    guarded(&qualification_fixture::oracle(case).expect("fixed valid qualification case"))
}

fn validate_output(case: &Case, actual: &[f32], expected: &[f32]) -> Result<(), io::Error> {
    if actual.len() != expected.len() {
        return Err(invalid("vecadd allocation length changed"));
    }
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let logical = index >= GUARD && index < GUARD + case.c_len;
        // Preserve original F32 NaN-class semantics; every other value,
        // signed zero, infinity, and every canary compares exact bits.
        let equal = if logical && e.is_nan() {
            a.is_nan()
        } else {
            a.to_bits() == e.to_bits()
        };
        if !equal {
            return Err(invalid(format!(
                "{} {} at allocation slot {index}",
                case.name,
                if logical {
                    "output mismatch"
                } else {
                    "output canary modified"
                }
            )));
        }
    }
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    if size_of::<usize>() != 8 || EXPLICIT_KERNARG_BYTES != 48 {
        return Err(
            invalid("qualification requires the emitted 48-byte 64-bit kernarg ABI").into(),
        );
    }
    let path = std::env::var_os("FE2O3_VECADD_HSACO").ok_or_else(|| {
        invalid("FE2O3_VECADD_HSACO is required; use the documented qualification runner")
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
    // actual symbol/48-byte ABI/target/Wave64/256-thread metadata and hashes.
    // This is the documented qualification escape hatch, not Worker V3 load.
    let module = unsafe { context.load_module_from_file_unchecked(path)? };
    let function = module.load_function(KERNEL)?;
    let mut dispatched = 0;
    for case in cases() {
        let expected = expected(&case);
        let a = guarded(&case.a);
        let b = guarded(&case.b);
        let initial = vec![CANARY; GUARD + case.c_len + GUARD];
        let a_device = DeviceBuffer::from_host(&stream, &a)?;
        let b_device = DeviceBuffer::from_host(&stream, &b)?;
        let mut c_device = DeviceBuffer::from_host(&stream, &initial)?;
        let groups = target::workgroups(case.c_len)?;
        if let Some(groups) = groups {
            if groups > observed.max_grid_dimensions()[0]
                || observed.max_threads_per_block() < 256
                || observed.max_block_dimensions()[0] < 256
            {
                return Err(invalid("vecadd launch exceeds observed device limits").into());
            }
            let av = a_device.view(GUARD..GUARD + case.a.len())?;
            let bv = b_device.view(GUARD..GUARD + case.b.len())?;
            let cv = c_device.view_mut(GUARD..GUARD + case.c_len)?;
            let mut args = KernelParams::new();
            args.push(av.as_device_ptr());
            args.push(av.len());
            args.push(bv.as_device_ptr());
            args.push(bv.len());
            args.push(cv.as_device_ptr());
            args.push(cv.len());
            // SAFETY: disjoint live F32 allocations, exact checked extents,
            // six owned ABI fields, original rounded 256-thread grid, and
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
            "{} {KERNEL} case={} target={TARGET} length={} groups={}",
            if groups.is_none() {
                "NO DISPATCH (legacy HIP)"
            } else {
                "NUMERICAL PASS (legacy HIP)"
            },
            case.name,
            case.c_len,
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
    fn full_qualification_cases_and_48_byte_abi() {
        assert_eq!(EXPLICIT_KERNARG_BYTES, 48);
        let all = cases();
        assert_eq!(all.len(), 17);
        for c in &all {
            assert!(c.a.len() >= c.c_len && c.b.len() >= c.c_len);
            let want = expected(c);
            assert_eq!(want.len(), c.c_len + 2 * GUARD);
            validate_output(c, &want, &want).unwrap();
        }
    }
    #[test]
    fn full_oracle_rejects_tail_canary_nan_and_length_corruption() {
        let all = cases();
        let c = all.iter().find(|c| c.c_len == 257).unwrap();
        let want = expected(c);
        for index in [0, GUARD, GUARD + 256, want.len() - 1] {
            let mut got = want.clone();
            got[index] = 42.0;
            assert!(validate_output(c, &got, &want).is_err());
        }
        let mut got = want.clone();
        got[GUARD] = f32::NAN;
        assert!(validate_output(c, &got, &want).is_err());
        assert!(validate_output(c, &want[..want.len() - 1], &want).is_err());
        assert!(!same_bits(&[0.0], &[-0.0]));
        assert!(same_bits(&[CANARY], &[CANARY]));
        assert!(!same_bits(
            &[CANARY],
            &[f32::from_bits(CANARY.to_bits() ^ 1)]
        ));
    }
    #[test]
    fn original_special_values_and_signed_zero_contract() {
        let all = cases();
        let c = all
            .iter()
            .find(|c| c.name == "floating-specials-and-rounding")
            .unwrap();
        let want = expected(c);
        assert_eq!(want[GUARD].to_bits(), 0);
        assert_eq!(want[GUARD + 1].to_bits(), 0x8000_0000);
        let mut got = want.clone();
        got[GUARD + 1] = 0.0;
        assert!(validate_output(c, &got, &want).is_err());
        let mut got = want.clone();
        got[GUARD + 6] = 0.0;
        assert!(validate_output(c, &got, &want).is_err());
    }
}

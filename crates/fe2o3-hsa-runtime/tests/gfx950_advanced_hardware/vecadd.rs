//! Numerical qualification only; this grants no production Worker V3 authority.
use super::*;
#[path = "../../../../examples/vecadd/src/qualification_fixture.rs"]
mod reference;

pub(super) const CASE: AdvancedCase = AdvancedCase {
    label: "gfx950 vecadd",
    export: "vecadd",
    descriptor: "vecadd.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THREE_SLICES,
};
const GUARDS: usize = 8;
const CANARY: u32 = 0x7fc1_abcd;
const OFFSET: usize = GUARDS * 4;

fn groups(length: usize) -> Result<Option<u32>, BoxError> {
    if length == 0 {
        return Ok(None);
    }
    let rounded = length.checked_add(255).ok_or("vecadd extent overflow")? / 256;
    Ok(Some(
        u32::try_from(rounded).map_err(|_| "vecadd grid exceeds u32")?,
    ))
}
fn guarded(values: &[f32]) -> Vec<u8> {
    let mut words = vec![f32::from_bits(CANARY); GUARDS];
    words.extend_from_slice(values);
    words.extend(std::iter::repeat_n(f32::from_bits(CANARY), GUARDS));
    value_bytes(&words)
}
struct Plan {
    launch: LaunchPlan,
    expected: Vec<f32>,
    groups: Option<u32>,
}
fn plan(case: reference::Case) -> Result<Plan, BoxError> {
    // Check both complete input extents before allocating or obtaining a GPU context.
    let expected = reference::oracle(&case)?;
    let grid = groups(case.c_len)?;
    let poison: Vec<_> = expected
        .iter()
        .map(|x| f32::from_bits(!x.to_bits()))
        .collect();
    let buffers = [
        ("a", &case.a, true),
        ("b", &case.b, true),
        ("c", &poison, false),
    ]
    .into_iter()
    .map(|(name, values, immutable)| PlannedBuffer {
        name,
        initial: guarded(values),
        body_offset: OFFSET,
        elements: values.len(),
        immutable,
        expected: None,
    })
    .collect();
    let args = [case.a.len(), case.b.len(), case.c_len]
        .into_iter()
        .enumerate()
        .map(|(buffer, elements)| PlannedArg::Slice { buffer, elements })
        .collect();
    Ok(Plan {
        launch: LaunchPlan {
            label: case.name,
            buffers,
            args,
        },
        expected,
        groups: grid,
    })
}
fn verify_output(plan: &Plan, actual: &[u8]) -> Result<(), BoxError> {
    let initial = &plan.launch.buffers[2].initial;
    require(
        actual.len() == initial.len(),
        "vecadd output allocation length changed",
    )?;
    let end = actual.len() - OFFSET;
    require(
        actual[..OFFSET] == initial[..OFFSET] && actual[end..] == initial[end..],
        "vecadd output canary changed",
    )?;
    let values = decode_f32(&actual[OFFSET..end])?;
    require(
        values.len() == plan.expected.len(),
        "vecadd output extent changed",
    )?;
    for (index, (actual, expected)) in values.iter().zip(&plan.expected).enumerate() {
        require(
            if expected.is_nan() {
                actual.is_nan()
            } else {
                actual.to_bits() == expected.to_bits()
            },
            format!(
                "{} output[{index}] actual={:08x} expected={:08x}",
                plan.launch.label,
                actual.to_bits(),
                expected.to_bits()
            ),
        )?;
    }
    Ok(())
}
fn readbacks(plan: &Plan, buffers: &[ReviewedHsaHardwareTestBufferV1]) -> Result<(), BoxError> {
    for index in 0..2 {
        verify_buffer(
            &plan.launch.buffers[index],
            &buffers[index].read_after_synchronous_dispatch(),
        )?;
    }
    verify_output(plan, &buffers[2].read_after_synchronous_dispatch())
}
fn execute(
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    bytes: &[u8],
    digest: PayloadDigest,
    plan: Plan,
) -> Result<(), BoxError> {
    let buffers = plan
        .launch
        .buffers
        .iter()
        .map(|b| adapter.allocate_hardware_test_buffer(&b.initial))
        .collect::<Result<Vec<_>, _>>()?;
    let Some(grid) = plan.groups else {
        readbacks(&plan, &buffers)?;
        println!(
            "PASS {} empty: full guarded readbacks, no executable load or dispatch",
            plan.launch.label
        );
        return Ok(());
    };
    let explicit = explicit_kernarg(CASE, &plan.launch, &buffers)?;
    // SAFETY: digest-pinned immutable bytes and allocations outlive synchronous dispatch and sole unload.
    let (executable, load) = unsafe { adapter.load_executable(bytes, digest) }?;
    let identity = load.executable_object();
    let execution = (|| -> Result<(), BoxError> {
        require(
            load.finalized_digest() == digest && load.byte_len() == bytes.len() as u64,
            "vecadd load observation changed",
        )?;
        // SAFETY: strict profile inspection admitted this exact export and descriptor.
        let (kernels, resolutions) =
            unsafe { adapter.resolve_kernel_set(&executable, [CASE.export]) }?;
        require(
            kernels.len() == 1
                && resolutions.len() == 1
                && resolutions[0].executable_object() == identity,
            "vecadd resolved a substituted kernel",
        )?;
        let kernel = kernels.get(0).ok_or("vecadd runtime omitted kernel")?;
        // SAFETY: same reviewed launch boundary, checked dynamic grid, live complete slice allocations.
        unsafe {
            dispatch_with_grid(
                CASE,
                adapter,
                &executable,
                kernel,
                &resolutions[0],
                &explicit,
                grid,
            )?;
        }
        readbacks(&plan, &buffers)
    })();
    // SAFETY: completion is synchronous and this is the sole consuming unload.
    let unload = unsafe { adapter.unload_executable(executable) }?;
    require(
        unload.released() && unload.executable_object() == identity,
        "vecadd executable was not released",
    )?;
    execution?;
    println!(
        "PASS {}: full output, both full inputs and all canaries",
        plan.launch.label
    );
    Ok(())
}
pub(super) fn run() -> Result<(), BoxError> {
    require(
        std::env::var_os(PERF_OUTPUT_ENV).is_none(),
        "vecadd numerical qualification does not support profiling",
    )?;
    if let Some(value) = std::env::var_os(WORKGROUP_X_ENV) {
        require(
            value == "256",
            "vecadd requires the compiled 256-thread workgroup",
        )?;
    }
    let plans = reference::cases()
        .into_iter()
        .map(plan)
        .collect::<Result<Vec<_>, _>>()?;
    let (bytes, digest) = read_pinned_hsaco()?;
    inspect_profile(CASE, &bytes)?;
    let context = GpuContext::new(0)?;
    let mut adapter = ReviewedHsaRuntimeAdapterV1::new_gfx950(context)?;
    require(
        adapter.environment().physical_device().target().processor() == "gfx950"
            && adapter.environment().physical_device().target().xnack()
                == Some(FeatureState::Disabled),
        "vecadd requires a gfx950:xnack- physical device",
    )?;
    for plan in plans {
        execute(&mut adapter, &bytes, digest, plan)?;
    }
    println!("PASS vecadd non-authoritative numerical HSA qualification: all 17 cases");
    Ok(())
}

#[test]
fn full_case_matrix_and_checked_grid() -> Result<(), BoxError> {
    let cases = reference::cases();
    assert_eq!(cases.len(), 17);
    assert_eq!(
        cases[..15].iter().map(|c| c.c_len).collect::<Vec<_>>(),
        [
            0, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025
        ]
    );
    assert_eq!(kernarg_size(CASE.args), 48);
    assert_eq!(groups(0)?, None);
    assert_eq!(groups(1)?, Some(1));
    assert_eq!(groups(256)?, Some(1));
    assert_eq!(groups(257)?, Some(2));
    assert_eq!(groups(1025)?, Some(5));
    assert!(groups(usize::MAX).is_err());
    let special = reference::oracle(&cases[15])?;
    assert_eq!(special.len(), 13);
    assert_eq!(special[1].to_bits(), (-0.0_f32).to_bits());
    assert!(special[4].is_infinite() && special[5].is_infinite());
    assert!(special[6..9].iter().all(|v| v.is_nan()));
    assert_ne!(cases[16].a.len(), cases[16].b.len());
    for case in cases {
        let p = plan(case)?;
        verify_output(&p, &guarded(&p.expected))?;
    }
    Ok(())
}
#[test]
fn invalid_inputs_rejected_before_runtime() {
    for (a, b) in [(vec![], vec![1.0]), (vec![1.0], vec![])] {
        assert!(
            plan(reference::Case {
                name: "invalid".into(),
                c_len: 1,
                a,
                b
            })
            .is_err()
        );
    }
}
#[test]
fn full_outputs_poison_and_canary_mutations_are_rejected() -> Result<(), BoxError> {
    for case in reference::cases() {
        let p = plan(case)?;
        let good = guarded(&p.expected);
        verify_output(&p, &good)?;
        if !p.expected.is_empty() {
            assert!(verify_output(&p, &p.launch.buffers[2].initial).is_err());
        }
        for i in 0..p.expected.len() {
            let mut bad = good.clone();
            let replacement = if p.expected[i].is_nan() {
                0.0_f32.to_bits()
            } else {
                f32::NAN.to_bits()
            };
            bad[OFFSET + i * 4..OFFSET + i * 4 + 4].copy_from_slice(&replacement.to_le_bytes());
            assert!(
                verify_output(&p, &bad).is_err(),
                "{} slot {i}",
                p.launch.label
            );
        }
        for i in (0..OFFSET).chain(good.len() - OFFSET..good.len()) {
            let mut bad = good.clone();
            bad[i] ^= 1;
            assert!(verify_output(&p, &bad).is_err());
        }
        for input in &p.launch.buffers[..2] {
            verify_buffer(input, &input.initial)?;
            for i in 0..input.initial.len() {
                let mut bad = input.initial.clone();
                bad[i] ^= 1;
                assert!(verify_buffer(input, &bad).is_err());
            }
        }
    }
    Ok(())
}
#[test]
fn nan_payloads_allow_class_but_signed_zero_and_infinity_are_exact() -> Result<(), BoxError> {
    let p = plan(reference::cases().remove(15))?;
    let mut bytes = guarded(&p.expected);
    for (i, v) in p.expected.iter().enumerate() {
        if v.is_nan() {
            bytes[OFFSET + i * 4..OFFSET + i * 4 + 4]
                .copy_from_slice(&0xffc1_2345_u32.to_le_bytes());
        }
    }
    verify_output(&p, &bytes)?;
    for i in [1, 4, 5] {
        let mut bad = bytes.clone();
        let changed = p.expected[i].to_bits() ^ 0x8000_0000;
        bad[OFFSET + i * 4..OFFSET + i * 4 + 4].copy_from_slice(&changed.to_le_bytes());
        assert!(verify_output(&p, &bad).is_err());
    }
    Ok(())
}

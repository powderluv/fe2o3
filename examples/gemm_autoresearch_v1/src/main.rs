use std::io;

use fe2o3_core::{
    DeviceBuffer, Event, GpuContext, GpuFunction, KernelParams, LaunchConfig, Stream,
    launch_kernel_on_stream,
};
use fe2o3_device::Bf16;
use fe2o3_gemm_autoresearch_v1::reference::{ReferenceProblemV1, evaluate_reference_v1};

const HSACO_ENV: &str = "FE2O3_AUTORESEARCH_GEMM_HSACO";
const BENCHMARK_ENV: &str = "FE2O3_BENCHMARK";
#[cfg(not(feature = "kernel-simt-autoresearch"))]
const KERNEL: &str = "gemm_autoresearch_v1";
#[cfg(feature = "kernel-simt-autoresearch")]
const KERNEL: &str = "gemm_autoresearch_simt_v1";
const BENCHMARK_SIZES: [(u32, usize); 3] = [(256, 20), (512, 10), (1024, 3)];

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn kernel_params(
    lhs: &DeviceBuffer<u16>,
    rhs: &DeviceBuffer<u16>,
    output: &DeviceBuffer<f32>,
    problem: ReferenceProblemV1,
) -> KernelParams {
    let mut arguments = KernelParams::new();
    arguments.push(lhs.as_device_ptr());
    arguments.push(lhs.len());
    arguments.push(rhs.as_device_ptr());
    arguments.push(rhs.len());
    arguments.push(output.as_device_ptr());
    arguments.push(output.len());
    arguments.push(problem.rows);
    arguments.push(problem.columns);
    arguments.push(problem.reduction);
    arguments.push(problem.lhs_stride);
    arguments.push(problem.rhs_stride);
    arguments.push(problem.output_stride);
    arguments.push(problem.product_scale);
    arguments.push(problem.output_scale);
    arguments
}

fn launch_config(problem: ReferenceProblemV1) -> Result<LaunchConfig, io::Error> {
    Ok(LaunchConfig {
        grid_dim: (qualification_workgroups(problem)?, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    })
}

unsafe fn launch(
    function: &GpuFunction,
    stream: &Stream,
    problem: ReferenceProblemV1,
    arguments: &mut KernelParams,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: the caller retains the matching module and allocations and passes
    // the compiler-emitted ABI assembled by `kernel_params`.
    unsafe { launch_kernel_on_stream(function, launch_config(problem)?, stream, arguments)? };
    Ok(())
}

fn percentile(sorted: &[f32], percentile: usize) -> f32 {
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn benchmark(
    context: &std::sync::Arc<GpuContext>,
    function: &GpuFunction,
    stream: &Stream,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut throughputs = Vec::with_capacity(BENCHMARK_SIZES.len());
    for (size, launches_per_sample) in BENCHMARK_SIZES {
        let elements = usize::try_from(size)? * usize::try_from(size)?;
        let lhs = DeviceBuffer::from_host(stream, &vec![0_u16; elements])?;
        let rhs = DeviceBuffer::from_host(stream, &vec![0_u16; elements])?;
        let output = DeviceBuffer::from_host(stream, &vec![0.0_f32; elements])?;
        let problem = ReferenceProblemV1 {
            rows: size,
            columns: size,
            reduction: size,
            lhs_stride: size,
            rhs_stride: size,
            output_stride: size,
            product_scale: 1.0,
            output_scale: 0.0,
        };
        let mut arguments = kernel_params(&lhs, &rhs, &output, problem);
        for _ in 0..10 {
            // SAFETY: buffers, module, function, and ABI remain live through synchronization.
            unsafe { launch(function, stream, problem, &mut arguments)? };
        }
        stream.synchronize()?;

        let mut start = Event::new(context)?;
        let mut stop = Event::new(context)?;
        let mut samples = Vec::with_capacity(11);
        for _ in 0..11 {
            start.record(stream)?;
            for _ in 0..launches_per_sample {
                // SAFETY: buffers, module, function, and ABI remain live through the stop event.
                unsafe { launch(function, stream, problem, &mut arguments)? };
            }
            stop.record(stream)?;
            stop.synchronize()?;
            samples.push(stop.elapsed_time_ms_since(&start)? * 1000.0 / launches_per_sample as f32);
        }
        samples.sort_by(f32::total_cmp);
        let median_us = percentile(&samples, 50);
        let operations = 2.0_f64 * f64::from(size).powi(3);
        let gflops = operations / f64::from(median_us) / 1000.0;
        throughputs.push(gflops);
        println!(
            "AUTORESEARCH_BENCH size={size} median_us={median_us:.3} p10_us={:.3} p90_us={:.3} gflops={gflops:.2}",
            percentile(&samples, 10),
            percentile(&samples, 90),
        );
    }
    let score = throughputs
        .iter()
        .product::<f64>()
        .powf(1.0 / throughputs.len() as f64);
    println!("AUTORESEARCH_SCORE geometric_mean_gflops={score:.2}");
    Ok(())
}

fn qualification_workgroups(problem: ReferenceProblemV1) -> Result<u32, io::Error> {
    if problem.rows == 0 || problem.columns == 0 || problem.output_stride < problem.columns {
        return Err(invalid_input(
            "qualification requires a nonempty valid output",
        ));
    }
    #[cfg(feature = "kernel-simt-autoresearch")]
    let groups = {
        // SIMT indices address physical C slots, including inter-row padding.
        let extent = u64::from(problem.rows - 1) * u64::from(problem.output_stride)
            + u64::from(problem.columns);
        extent.div_ceil(64)
    };
    #[cfg(not(feature = "kernel-simt-autoresearch"))]
    let groups = u64::from(problem.rows).div_ceil(16) * u64::from(problem.columns).div_ceil(16);
    u32::try_from(groups).map_err(|_| invalid_input("qualification grid exceeds u32"))
}

fn finite_reference_error(actual: f32, expected: f32) -> Option<f32> {
    if !actual.is_finite() || !expected.is_finite() {
        return None;
    }
    let error = (actual - expected).abs();
    error.is_finite().then_some(error)
}

struct QualificationCase {
    name: &'static str,
    problem: ReferenceProblemV1,
    lhs: Vec<u16>,
    rhs: Vec<u16>,
    initial: Vec<f32>,
    exact: Option<Vec<f32>>,
}

fn qualification_cases() -> Vec<QualificationCase> {
    const CANARY: f32 = f32::from_bits(0x7f80_0001);
    let problem = ReferenceProblemV1 {
        rows: 19,
        columns: 21,
        reduction: 23,
        lhs_stride: 27,
        rhs_stride: 25,
        output_stride: 29,
        product_scale: 0.75,
        output_scale: -0.25,
    };
    let mut initial = (0..19 * 29).map(|i| 1000.0 + i as f32).collect::<Vec<_>>();
    initial.extend_from_slice(&[CANARY; 8]);
    let mut cases = vec![QualificationCase {
        name: "strided-tails",
        problem,
        lhs: (0..19 * 27)
            .map(|i| Bf16::from_f32((i % 13) as f32 * 0.125 - 0.75).to_bits())
            .collect(),
        rhs: (0..23 * 25)
            .map(|i| Bf16::from_f32((i % 11) as f32 * 0.1 - 0.5).to_bits())
            .collect(),
        initial,
        exact: None,
    }];
    for (name, lhs, rhs, bits) in [
        ("nonquarter", vec![0x3f81], vec![0x3f7f], 0x3f80_7f00),
        (
            "cancellation",
            vec![0x4281, 0x4280],
            vec![0x4281, 0xc282],
            0x3e80_0000,
        ),
    ] {
        let reduction = lhs.len() as u32;
        cases.push(QualificationCase {
            name,
            lhs,
            rhs,
            initial: vec![0.0, CANARY],
            exact: Some(vec![f32::from_bits(bits), CANARY]),
            problem: ReferenceProblemV1 {
                rows: 1,
                columns: 1,
                reduction,
                lhs_stride: reduction,
                rhs_stride: 1,
                output_stride: 1,
                product_scale: 1.0,
                output_scale: 0.0,
            },
        });
    }
    let mut lhs = vec![0_u16; 256];
    let mut rhs = vec![0_u16; 256];
    let mut exact = vec![CANARY; 264];
    for row in 0..16 {
        for depth in 0..16 {
            lhs[row * 16 + depth] = 0x3f80 + (row + depth) as u16;
        }
    }
    for depth in 0..16 {
        for column in 0..16 {
            let numerator = 128 + 2 * column - depth;
            rhs[depth * 16 + column] = if numerator >= 128 {
                0x3f80 + (numerator - 128) as u16
            } else {
                0x3f00 + 2 * (numerator - 64) as u16
            };
        }
    }
    for row in 0..16 {
        for column in 0..16 {
            let a = (128 + row) as i32;
            let b = (128 + 2 * column) as i32;
            // Exact rational sum from the maintained paired-contract oracle.
            exact[row * 16 + column] = (16 * a * b + 120 * (b - a) - 1240) as f32 / 16384.0;
        }
    }
    let mut initial = vec![CANARY; 264];
    initial[..256].fill(0.0);
    cases.push(QualificationCase {
        name: "fractional-full",
        lhs,
        rhs,
        initial,
        exact: Some(exact),
        problem: ReferenceProblemV1 {
            rows: 16,
            columns: 16,
            reduction: 16,
            lhs_stride: 16,
            rhs_stride: 16,
            output_stride: 16,
            product_scale: 1.0,
            output_scale: 0.0,
        },
    });
    cases
}

fn qualification_target_matches(requested: &str, observed: &str, wave: i32) -> bool {
    matches!(requested, "gfx942" | "gfx950") && requested == observed && wave == 64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for case in qualification_cases() {
        run_case(case)?;
    }
    if std::env::var_os(BENCHMARK_ENV).is_some() {
        let requested =
            std::env::var("FE2O3_AUTORESEARCH_GEMM_TARGET").unwrap_or_else(|_| "gfx942".into());
        let context = GpuContext::new(0)?;
        let observed = context.observe_target()?;
        if !qualification_target_matches(
            &requested,
            observed.target_id().processor(),
            observed.hip_default_warp_size() as i32,
        ) {
            return Err(invalid_input(
                "benchmark target must match requested gfx942/gfx950 wave64",
            )
            .into());
        }
        let stream = context.create_stream()?;
        let hsaco = std::env::var_os(HSACO_ENV)
            .ok_or_else(|| invalid_input("missing compiled GEMM HSACO"))?;
        // SAFETY: the runner compiled this symbol and observed target. The module
        // and every benchmark allocation remain live through event synchronization.
        let module = unsafe { context.load_module_from_file_unchecked(hsaco)? };
        let function = module.load_function(KERNEL)?;
        benchmark(&context, &function, &stream)?;
    }
    Ok(())
}

fn run_case(case: QualificationCase) -> Result<(), Box<dyn std::error::Error>> {
    let problem = case.problem;
    let lhs = case.lhs;
    let rhs = case.rhs;
    let initial_output = case.initial;
    let expected =
        evaluate_reference_v1(&lhs, &rhs, &initial_output, problem).map_err(invalid_input)?;
    if let Some(exact) = &case.exact {
        if expected.len() != exact.len()
            || !expected
                .iter()
                .zip(exact)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err(
                io::Error::other("maintained rational oracle disagrees with reference").into(),
            );
        }
    }
    let workgroups = qualification_workgroups(problem)?;

    let context = GpuContext::new(0)?;
    let observed = context.observe_target()?;
    let requested = match std::env::var("FE2O3_AUTORESEARCH_GEMM_TARGET") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "gfx942".into(),
        Err(error) => return Err(error.into()),
    };
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
    let lhs_device = DeviceBuffer::from_host(&stream, &lhs)?;
    let rhs_device = DeviceBuffer::from_host(&stream, &rhs)?;
    let output_device = DeviceBuffer::from_host(&stream, &initial_output)?;
    let hsaco = std::env::var_os(HSACO_ENV).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("{HSACO_ENV} is not set"))
    })?;

    // SAFETY: the qualification runner just compiled this exact symbol for the requested observed target. The argument
    // order and scalar widths below mirror the compiler-emitted 80-byte explicit kernarg ABI;
    // all three allocations remain live through the synchronous stream completion.
    unsafe {
        let module = context.load_module_from_file_unchecked(hsaco)?;
        let function = module.load_function(KERNEL)?;
        let mut arguments = KernelParams::new();
        arguments.push(lhs_device.as_device_ptr());
        arguments.push(lhs_device.len());
        arguments.push(rhs_device.as_device_ptr());
        arguments.push(rhs_device.len());
        arguments.push(output_device.as_device_ptr());
        arguments.push(output_device.len());
        arguments.push(problem.rows);
        arguments.push(problem.columns);
        arguments.push(problem.reduction);
        arguments.push(problem.lhs_stride);
        arguments.push(problem.rhs_stride);
        arguments.push(problem.output_stride);
        arguments.push(problem.product_scale);
        arguments.push(problem.output_scale);
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

    if lhs_device.to_host_vec(&stream)? != lhs || rhs_device.to_host_vec(&stream)? != rhs {
        return Err(io::Error::other("GEMM modified immutable BF16 inputs").into());
    }
    let actual = output_device.to_host_vec(&stream)?;
    let mut maximum_error = 0.0_f32;
    for row in 0..problem.rows as usize {
        for column in 0..problem.output_stride as usize {
            let index = row * problem.output_stride as usize + column;
            if column < problem.columns as usize {
                let error =
                    finite_reference_error(actual[index], expected[index]).ok_or_else(|| {
                        io::Error::other(format!(
                            "non-finite GEMM result/reference at ({row}, {column})"
                        ))
                    })?;
                maximum_error = maximum_error.max(error);
                let tolerance = 2.0e-3_f32.max(expected[index].abs() * 2.0e-3);
                if error > tolerance {
                    return Err(io::Error::other(format!(
                        "GEMM mismatch at ({row}, {column}): got {}, expected {}, tolerance {tolerance}",
                        actual[index], expected[index]
                    ))
                    .into());
                }
            } else if actual[index].to_bits() != initial_output[index].to_bits() {
                return Err(io::Error::other(format!(
                    "GEMM modified output padding at ({row}, {column})"
                ))
                .into());
            }
        }
    }
    for index in (problem.rows as usize * problem.output_stride as usize)..actual.len() {
        if actual[index].to_bits() != initial_output[index].to_bits() {
            return Err(io::Error::other("GEMM changed trailing output canary").into());
        }
    }
    println!(
        "PASS {KERNEL} case={} target={requested}: {}x{}x{}, {} workgroups, max_abs_error={maximum_error:.6}",
        case.name, problem.rows, problem.columns, problem.reduction, workgroups,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_workload_is_the_original_three_sizes_and_repeat_counts() {
        assert_eq!(BENCHMARK_SIZES, [(256, 20), (512, 10), (1024, 3)]);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 50), 2.0);
    }

    #[test]
    fn targets_are_closed_and_match_the_requested_architecture() {
        for target in ["gfx942", "gfx950"] {
            assert!(qualification_target_matches(target, target, 64));
            assert!(!qualification_target_matches(target, target, 32));
        }
        assert!(!qualification_target_matches("gfx942", "gfx950", 64));
        assert!(!qualification_target_matches("gfx950", "gfx942", 64));
        assert!(!qualification_target_matches("gfx1100", "gfx1100", 64));
    }

    #[test]
    fn gpu_cases_keep_the_complete_raw_bf16_rational_oracles() {
        let cases = qualification_cases();
        assert_eq!(cases.len(), 4);
        assert_eq!(cases[1].lhs, [0x3f81]);
        assert_eq!(cases[2].rhs, [0x4281, 0xc282]);
        assert_eq!(cases[3].initial.len(), 264);
        for case in cases {
            let expected =
                evaluate_reference_v1(&case.lhs, &case.rhs, &case.initial, case.problem).unwrap();
            if let Some(exact) = case.exact {
                assert_eq!(expected.len(), exact.len());
                for (a, b) in expected.iter().zip(exact) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }

    fn problem() -> ReferenceProblemV1 {
        ReferenceProblemV1 {
            rows: 19,
            columns: 21,
            reduction: 23,
            lhs_stride: 27,
            rhs_stride: 25,
            output_stride: 29,
            product_scale: 0.75,
            output_scale: -0.25,
        }
    }

    #[test]
    fn selected_symbol_and_grid_match_the_source_mapping() {
        #[cfg(feature = "kernel-simt-autoresearch")]
        {
            assert_eq!(KERNEL, "gemm_autoresearch_simt_v1");
            // Last logical output is physical slot 542: ceil(543 / 64) = 9.
            assert_eq!(qualification_workgroups(problem()).unwrap(), 9);
        }
        #[cfg(not(feature = "kernel-simt-autoresearch"))]
        {
            assert_eq!(KERNEL, "gemm_autoresearch_v1");
            assert_eq!(qualification_workgroups(problem()).unwrap(), 4);
        }
    }

    #[test]
    fn grid_rejects_empty_invalid_and_unrepresentable_outputs() {
        for candidate in [
            ReferenceProblemV1 {
                rows: 0,
                ..problem()
            },
            ReferenceProblemV1 {
                columns: 0,
                ..problem()
            },
            ReferenceProblemV1 {
                output_stride: 20,
                ..problem()
            },
            ReferenceProblemV1 {
                rows: u32::MAX,
                columns: u32::MAX,
                output_stride: u32::MAX,
                ..problem()
            },
        ] {
            assert!(qualification_workgroups(candidate).is_err());
        }
    }

    #[test]
    fn nonfinite_results_and_reference_never_pass_the_comparison() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(finite_reference_error(value, 1.0), None);
            assert_eq!(finite_reference_error(1.0, value), None);
        }
        assert_eq!(finite_reference_error(f32::MAX, -f32::MAX), None);
        assert_eq!(finite_reference_error(-3.0, 2.0), Some(5.0));
        assert_eq!(finite_reference_error(-0.0, 0.0), Some(0.0));
    }
}

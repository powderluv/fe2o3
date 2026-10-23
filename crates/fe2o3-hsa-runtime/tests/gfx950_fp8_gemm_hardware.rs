//! Non-authoritative gfx950 FP8 GEMM hardware and numerical harness.
//!
//! This ignored test consumes a caller-supplied, digest-pinned COV6 HSACO.
//! It bypasses production prerequisite authentication and grants no protected
//! evidence. Its purpose is to exercise the first Rust-produced gfx950 FP8
//! GEMM artifact through the reviewed raw HSA adapter.

#[cfg(feature = "hardware-test-hooks")]
use fe2o3_amd_target::FeatureState;
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_artifacts::{DigestAlgorithm, PayloadDigest};
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_core::GpuContext;
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_host::{
    HsaKernelResolutionObservationV1, HsaLaunchGeometryV1, ReviewedHsaExecutableLifecycleAdapterV1,
    ReviewedHsaImplicitKernargAdapterV1,
};
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_hsa_runtime::{
    ReviewedHsaExecutableV1, ReviewedHsaHardwareTestBufferV1, ReviewedHsaKernelV1,
    ReviewedHsaRuntimeAdapterV1,
};
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_hsaco::{CodeObjectVersion, ExplicitValueKind};

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy)]
enum KernelVariant {
    Tile,
    Simt,
}

#[cfg(feature = "hardware-test-hooks")]
impl KernelVariant {
    fn export(self) -> &'static str {
        match self {
            Self::Tile => "gfx950_fp8_gemm_rust",
            Self::Simt => "gfx950_fp8_gemm_simt",
        }
    }

    fn descriptor(self) -> &'static str {
        match self {
            Self::Tile => "gfx950_fp8_gemm_rust.kd",
            Self::Simt => "gfx950_fp8_gemm_simt.kd",
        }
    }
}
#[cfg(feature = "hardware-test-hooks")]
const M: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const N: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const K: usize = 128;
#[cfg(feature = "hardware-test-hooks")]
const A_ELEMENTS: usize = M * K;
#[cfg(feature = "hardware-test-hooks")]
const B_ELEMENTS: usize = K * N;
#[cfg(feature = "hardware-test-hooks")]
const C_ELEMENTS: usize = M * N;
#[cfg(feature = "hardware-test-hooks")]
const BATCHES: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const WORKGROUP_X: u32 = 256;
#[cfg(feature = "hardware-test-hooks")]
const GRID_X: u32 = 4;
#[cfg(feature = "hardware-test-hooks")]
const EXPLICIT_KERNARG_BYTES: usize = 48;
#[cfg(feature = "hardware-test-hooks")]
const COMPLETE_KERNARG_BYTES: usize = EXPLICIT_KERNARG_BYTES;
#[cfg(feature = "hardware-test-hooks")]
const HSA_KERNARG_ALIGNMENT: u64 = 16;
#[cfg(feature = "hardware-test-hooks")]
// AMDHSA metadata records the argument ABI's natural alignment. The reviewed
// HSA symbol query separately reports the 16-byte runtime allocation minimum.
const METADATA_KERNARG_ALIGNMENT: u64 = 8;
#[cfg(feature = "hardware-test-hooks")]
const CANARY_ELEMENTS: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const C_PREFIX: f32 = f32::from_bits(0x7fc0_9501);
#[cfg(feature = "hardware-test-hooks")]
const C_SUFFIX: f32 = f32::from_bits(0x7fc0_9502);
#[cfg(feature = "hardware-test-hooks")]
const C_POISON: f32 = f32::from_bits(0x7fc0_95ff);
#[cfg(feature = "hardware-test-hooks")]
const MAX_ABSOLUTE_ERROR: f32 = 1.0e-5;

#[cfg(feature = "hardware-test-hooks")]
type BoxError = Box<dyn std::error::Error>;

#[cfg(feature = "hardware-test-hooks")]
fn require(condition: bool, message: impl Into<String>) -> Result<(), BoxError> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn parse_sha256(text: &str) -> Result<[u8; 32], BoxError> {
    require(
        text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "FE2O3_GFX950_FP8_GEMM_SHA256 must be exactly 64 lowercase hex digits",
    )?;
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| "FE2O3_GFX950_FP8_GEMM_SHA256 is malformed")?;
    }
    Ok(bytes)
}

#[cfg(feature = "hardware-test-hooks")]
fn read_pinned_hsaco() -> Result<(Vec<u8>, PayloadDigest), BoxError> {
    require(
        std::env::var("FE2O3_RUN_GFX950_FP8_GEMM_HARDWARE").as_deref() == Ok("1"),
        "set FE2O3_RUN_GFX950_FP8_GEMM_HARDWARE=1 to opt into this raw hardware test",
    )?;
    let path = std::path::PathBuf::from(
        std::env::var_os("FE2O3_GFX950_FP8_GEMM_HSACO")
            .ok_or("FE2O3_GFX950_FP8_GEMM_HSACO is not set")?,
    );
    require(
        path.is_absolute(),
        "FE2O3_GFX950_FP8_GEMM_HSACO must be an absolute path",
    )?;
    let metadata = std::fs::symlink_metadata(&path)?;
    require(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "FE2O3_GFX950_FP8_GEMM_HSACO must name a regular non-symlink file",
    )?;
    require(
        std::fs::canonicalize(&path)? == path,
        "FE2O3_GFX950_FP8_GEMM_HSACO must already be canonical",
    )?;
    require(
        (1..=fe2o3_hsaco::MAX_HSACO_BYTES as u64).contains(&metadata.len()),
        "FE2O3_GFX950_FP8_GEMM_HSACO has an invalid byte length",
    )?;
    let expected = parse_sha256(
        &std::env::var("FE2O3_GFX950_FP8_GEMM_SHA256")
            .map_err(|_| "FE2O3_GFX950_FP8_GEMM_SHA256 is not set")?,
    )?;
    let bytes = std::fs::read(&path)?;
    require(
        bytes.len() as u64 == metadata.len(),
        "FE2O3_GFX950_FP8_GEMM_HSACO changed size while being read",
    )?;
    let final_metadata = std::fs::symlink_metadata(&path)?;
    require(
        final_metadata.file_type().is_file()
            && !final_metadata.file_type().is_symlink()
            && final_metadata.len() == metadata.len()
            && std::fs::canonicalize(&path)? == path,
        "FE2O3_GFX950_FP8_GEMM_HSACO changed identity while being read",
    )?;
    let digest = DigestAlgorithm::Sha256.calculate(&bytes);
    require(
        digest.bytes().as_bytes() == &expected,
        "FE2O3_GFX950_FP8_GEMM_HSACO does not match its exact SHA-256 pin",
    )?;
    Ok((bytes, digest))
}

#[cfg(feature = "hardware-test-hooks")]
fn inspect_profile(bytes: &[u8], variant: KernelVariant) -> Result<(), BoxError> {
    let bound = fe2o3_hsaco::inspect_and_bind_kernel_descriptors(bytes)?;
    let inspected = bound.inspection();
    require(
        inspected.code_object_version() == CodeObjectVersion::V6,
        "gfx950 FP8 GEMM HSACO must be code object V6",
    )?;
    require(
        inspected.target().processor() == "gfx950"
            && inspected.target().xnack() == Some(FeatureState::Disabled),
        "gfx950 FP8 GEMM HSACO must target exact gfx950:xnack-",
    )?;
    require(
        !inspected.has_printf_metadata(),
        "gfx950 FP8 GEMM HSACO must not carry printf metadata",
    )?;
    let [kernel] = inspected.kernels() else {
        return Err("gfx950 FP8 GEMM HSACO must declare exactly one kernel".into());
    };
    require(
        kernel.name() == variant.export() && kernel.symbol() == variant.descriptor(),
        "gfx950 FP8 GEMM HSACO has a substituted kernel or descriptor symbol",
    )?;
    require(
        kernel.kernarg_segment_size() == COMPLETE_KERNARG_BYTES as u64
            && kernel.kernarg_segment_alignment() == METADATA_KERNARG_ALIGNMENT
            && kernel.implicit_argument_offset().is_none()
            && kernel.implicit_argument_size() == 0,
        "gfx950 FP8 GEMM HSACO does not expose the exact 48-byte explicit kernarg",
    )?;
    require(
        kernel.required_workgroup_size() == Some([WORKGROUP_X, 1, 1])
            && kernel.max_flat_workgroup_size() == WORKGROUP_X
            && kernel.wavefront_size() == 64
            && kernel.group_segment_fixed_size() == 0
            && !kernel.uses_dynamic_stack(),
        "gfx950 FP8 GEMM HSACO does not expose the exact static WG256 profile",
    )?;
    const EXPECTED_ARGUMENTS: [(u64, u64, ExplicitValueKind); 6] = [
        (0, 8, ExplicitValueKind::GlobalBuffer),
        (8, 8, ExplicitValueKind::ByValue),
        (16, 8, ExplicitValueKind::GlobalBuffer),
        (24, 8, ExplicitValueKind::ByValue),
        (32, 8, ExplicitValueKind::GlobalBuffer),
        (40, 8, ExplicitValueKind::ByValue),
    ];
    let arguments = kernel
        .explicit_arguments()
        .iter()
        .map(|argument| (argument.offset(), argument.size(), argument.value_kind()))
        .collect::<Vec<_>>();
    require(
        arguments.as_slice() == EXPECTED_ARGUMENTS,
        "gfx950 FP8 GEMM HSACO has a substituted three-slice ABI",
    )?;
    let [binding] = bound.bindings() else {
        return Err("gfx950 FP8 GEMM HSACO must bind exactly one descriptor".into());
    };
    let descriptor = binding.descriptor();
    require(
        binding.kernel_index() == 0
            && descriptor.kernarg_size() == COMPLETE_KERNARG_BYTES as u32
            && u64::from(descriptor.group_segment_fixed_size())
                == kernel.group_segment_fixed_size()
            && u64::from(descriptor.private_segment_fixed_size())
                == kernel.private_segment_fixed_size()
            && descriptor.wavefront_size() == 64
            && !descriptor.uses_dynamic_stack(),
        "gfx950 FP8 GEMM descriptor disagrees with its metadata",
    )
}

#[cfg(feature = "hardware-test-hooks")]
fn decode_fp8_e4m3(bits: u8) -> f32 {
    let exponent = (bits >> 3) & 0xf;
    let mantissa = bits & 0x7;
    if exponent == 0xf && mantissa == 0x7 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if bits & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn deterministic_inputs() -> (Vec<u8>, Vec<u8>) {
    const CODES: [u8; 11] = [
        0x00, 0x28, 0xa8, 0x30, 0xb0, 0x38, 0xb8, 0x3c, 0xbc, 0x40, 0xc0,
    ];
    let mut lhs = Vec::with_capacity(BATCHES * A_ELEMENTS);
    let mut rhs = Vec::with_capacity(BATCHES * B_ELEMENTS);
    for batch in 0..BATCHES {
        lhs.extend((0..A_ELEMENTS).map(|index| {
            let row = index / K;
            let depth = index % K;
            CODES[(row * 3 + depth * 5 + row * depth + batch + (batch / CODES.len()) * (depth % 5))
                % CODES.len()]
        }));
        rhs.extend((0..B_ELEMENTS).map(|index| {
            let depth = index / N;
            let column = index % N;
            CODES[(depth * 7
                + column * 2
                + depth * column
                + 1
                + batch * 3
                + (batch / CODES.len()) * (column % 5))
                % CODES.len()]
        }));
    }
    (lhs, rhs)
}

#[cfg(feature = "hardware-test-hooks")]
fn cpu_reference(lhs: &[u8], rhs: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0; BATCHES * C_ELEMENTS];
    for batch in 0..BATCHES {
        let lhs_base = batch * A_ELEMENTS;
        let rhs_base = batch * B_ELEMENTS;
        let output_base = batch * C_ELEMENTS;
        for row in 0..M {
            for column in 0..N {
                let mut value = 0.0;
                for depth in 0..K {
                    value += decode_fp8_e4m3(lhs[lhs_base + row * K + depth])
                        * decode_fp8_e4m3(rhs[rhs_base + depth * N + column]);
                }
                output[output_base + row * N + column] = value;
            }
        }
    }
    output
}

#[cfg(feature = "hardware-test-hooks")]
fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns and the byte extent is exact.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn f32_values(bytes: &[u8]) -> Result<Vec<f32>, BoxError> {
    require(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "gfx950 FP8 GEMM output contains a partial f32",
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("exact f32 chunk")))
        .collect())
}

#[cfg(feature = "hardware-test-hooks")]
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "hardware-test-hooks")]
fn explicit_kernarg(addresses: [u64; 3]) -> [u8; EXPLICIT_KERNARG_BYTES] {
    let mut bytes = [0; EXPLICIT_KERNARG_BYTES];
    for (index, (address, elements)) in addresses
        .into_iter()
        .zip([
            BATCHES * A_ELEMENTS,
            BATCHES * B_ELEMENTS,
            BATCHES * C_ELEMENTS,
        ])
        .enumerate()
    {
        let offset = index * 16;
        put_u64(&mut bytes, offset, address);
        put_u64(&mut bytes, offset + 8, elements as u64);
    }
    bytes
}

#[cfg(feature = "hardware-test-hooks")]
struct RuntimeKernarg {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

#[cfg(feature = "hardware-test-hooks")]
impl RuntimeKernarg {
    fn new() -> Result<Self, BoxError> {
        let layout = std::alloc::Layout::from_size_align(
            COMPLETE_KERNARG_BYTES,
            HSA_KERNARG_ALIGNMENT as usize,
        )?;
        // SAFETY: layout is valid and this owner deallocates the result once.
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .ok_or("failed to allocate aligned gfx950 FP8 GEMM kernarg")?;
        Ok(Self { pointer, layout })
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is live and exactly layout.size() bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

#[cfg(feature = "hardware-test-hooks")]
impl Drop for RuntimeKernarg {
    fn drop(&mut self) {
        // SAFETY: this owner deallocates its exact live allocation once.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

#[cfg(feature = "hardware-test-hooks")]
unsafe fn dispatch_multigrid(
    variant: KernelVariant,
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    resolution: &HsaKernelResolutionObservationV1,
    explicit: &[u8; EXPLICIT_KERNARG_BYTES],
) -> Result<(), BoxError> {
    require(
        resolution.export_symbol() == variant.export()
            && resolution.kernarg_segment_size() == COMPLETE_KERNARG_BYTES as u64
            && resolution.kernarg_segment_alignment() == HSA_KERNARG_ALIGNMENT,
        "runtime resolution differs from the exact gfx950 FP8 GEMM ABI",
    )?;
    let geometry = HsaLaunchGeometryV1::new([GRID_X, 1, 1], [WORKGROUP_X, 1, 1], 0);
    let mut storage = RuntimeKernarg::new()?;
    let kernarg = storage.bytes_mut();
    kernarg[..EXPLICIT_KERNARG_BYTES].copy_from_slice(explicit);

    // SAFETY: metadata inspection admitted one digest-pinned COV6 image with
    // the exact explicit-only three-slice ABI. Preparation binds the exact
    // launch queue without mutating the empty implicit span. All buffers stay
    // live and dispatch completion is synchronous.
    unsafe {
        adapter.initialize_implicit_kernarg(
            executable,
            kernel,
            geometry,
            EXPLICIT_KERNARG_BYTES,
            EXPLICIT_KERNARG_BYTES,
            0,
            kernarg,
        )?;
        let mut session =
            adapter.prepare_profiled_dispatch_session(executable, kernel, geometry, kernarg)?;
        session.dispatch()?;

        if std::env::var("FE2O3_GFX950_LOWP_PERF").as_deref() == Ok("1") {
            const WARMUPS: usize = 1_000;
            const SAMPLES: usize = 3_000;
            for _ in 0..WARMUPS {
                session.dispatch()?;
            }
            let mut durations = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                durations.push(session.dispatch()?.duration_ns());
            }
            durations.sort_unstable();
            println!(
                "PERF {} warmups={WARMUPS} samples={SAMPLES} p50_ns={} p05_ns={} p95_ns={}",
                variant.export(),
                durations[SAMPLES / 2],
                durations[SAMPLES / 20],
                durations[SAMPLES * 19 / 20],
            );
        }
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn output_body_address(buffer: &ReviewedHsaHardwareTestBufferV1) -> Result<u64, BoxError> {
    require(
        buffer.byte_len()
            == (BATCHES * C_ELEMENTS + 2 * CANARY_ELEMENTS) * std::mem::size_of::<f32>(),
        "guarded gfx950 FP8 GEMM output has the wrong extent",
    )?;
    Ok(buffer.device_address(CANARY_ELEMENTS * std::mem::size_of::<f32>())?)
}

#[cfg(feature = "hardware-test-hooks")]
fn verify_output(actual: &[f32], expected: &[f32]) -> Result<(), BoxError> {
    require(
        actual.len() == BATCHES * C_ELEMENTS + 2 * CANARY_ELEMENTS,
        "guarded gfx950 FP8 GEMM output length changed",
    )?;
    let (prefix, remainder) = actual.split_at(CANARY_ELEMENTS);
    let (body, suffix) = remainder.split_at(BATCHES * C_ELEMENTS);
    require(
        prefix
            .iter()
            .all(|value| value.to_bits() == C_PREFIX.to_bits()),
        "gfx950 FP8 GEMM output prefix canary changed",
    )?;
    require(
        suffix
            .iter()
            .all(|value| value.to_bits() == C_SUFFIX.to_bits()),
        "gfx950 FP8 GEMM output suffix canary changed",
    )?;
    require(
        expected.len() == BATCHES * C_ELEMENTS,
        "gfx950 FP8 GEMM CPU reference has the wrong extent",
    )?;
    let mut maximum_absolute_error = 0.0_f32;
    for (index, (actual, expected)) in body.iter().zip(expected).enumerate() {
        let absolute_error = (actual - expected).abs();
        require(
            actual.is_finite() && expected.is_finite() && absolute_error <= MAX_ABSOLUTE_ERROR,
            format!(
                "gfx950 FP8 GEMM C[{index}] mismatch: actual={actual}, expected={expected}, tolerance={MAX_ABSOLUTE_ERROR}"
            ),
        )?;
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
    }
    println!(
        "PASS gfx950 FP8 GEMM batches={BATCHES} outputs={} max_absolute_error={maximum_absolute_error:.9e} tolerance={MAX_ABSOLUTE_ERROR:.1e}",
        BATCHES * C_ELEMENTS
    );
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn run_hardware(
    bytes: Vec<u8>,
    digest: PayloadDigest,
    variant: KernelVariant,
) -> Result<(), BoxError> {
    inspect_profile(&bytes, variant)?;
    let context = GpuContext::new(0)?;
    let mut adapter = ReviewedHsaRuntimeAdapterV1::new_gfx950(context)?;
    require(
        adapter.environment().physical_device().target().processor() == "gfx950"
            && adapter.environment().physical_device().target().xnack()
                == Some(FeatureState::Disabled),
        "gfx950 FP8 GEMM hardware test requires a gfx950:xnack- physical device",
    )?;

    let (lhs_host, rhs_host) = deterministic_inputs();
    let expected = cpu_reference(&lhs_host, &rhs_host);
    let mut output_host = Vec::with_capacity(BATCHES * C_ELEMENTS + 2 * CANARY_ELEMENTS);
    output_host.extend(std::iter::repeat_n(C_PREFIX, CANARY_ELEMENTS));
    output_host.extend(std::iter::repeat_n(C_POISON, BATCHES * C_ELEMENTS));
    output_host.extend(std::iter::repeat_n(C_SUFFIX, CANARY_ELEMENTS));
    let lhs = adapter.allocate_hardware_test_buffer(&lhs_host)?;
    let rhs = adapter.allocate_hardware_test_buffer(&rhs_host)?;
    let output = adapter.allocate_hardware_test_buffer(f32_bytes(&output_host))?;
    let explicit = explicit_kernarg([
        lhs.device_address(0)?,
        rhs.device_address(0)?,
        output_body_address(&output)?,
    ]);

    // SAFETY: immutable, digest-pinned bytes and every allocation are retained
    // until the synchronous dispatch and sole consuming unload complete.
    let (executable, load) = unsafe { adapter.load_executable(&bytes, digest) }?;
    let executable_identity = load.executable_object();
    let execution = (|| -> Result<(), BoxError> {
        require(
            load.finalized_digest() == digest && load.byte_len() == bytes.len() as u64,
            "gfx950 FP8 GEMM load observation changed",
        )?;
        // SAFETY: inspection admitted exactly this export and descriptor.
        let (kernels, resolutions) =
            unsafe { adapter.resolve_kernel_set(&executable, [variant.export()]) }?;
        let kernel = kernels.get(0).ok_or("runtime omitted gfx950 FP8 GEMM")?;
        require(
            kernels.len() == 1
                && resolutions.len() == 1
                && resolutions[0].executable_object() == executable_identity,
            "runtime resolved a substituted gfx950 FP8 GEMM kernel",
        )?;
        // SAFETY: dispatch_multigrid owns the reviewed raw launch boundary.
        unsafe {
            dispatch_multigrid(
                variant,
                &mut adapter,
                &executable,
                kernel,
                &resolutions[0],
                &explicit,
            )?;
        }
        require(
            lhs.read_after_synchronous_dispatch() == lhs_host,
            "gfx950 FP8 GEMM modified lhs",
        )?;
        require(
            rhs.read_after_synchronous_dispatch() == rhs_host,
            "gfx950 FP8 GEMM modified rhs",
        )?;
        verify_output(
            &f32_values(&output.read_after_synchronous_dispatch())?,
            &expected,
        )
    })();
    let unload = unsafe { adapter.unload_executable(executable) }?;
    require(
        unload.released() && unload.executable_object() == executable_identity,
        "reviewed HSA unload did not release the gfx950 FP8 GEMM executable",
    )?;
    execution
}

/// Runs a Rust-produced FP8 GEMM COV6 image as four WG256 workgroups.
///
/// This ignored test is non-authoritative and grants no protected evidence.
/// Invoke it with:
///
/// ```text
/// FE2O3_RUN_GFX950_FP8_GEMM_HARDWARE=1 \
/// FE2O3_GFX950_FP8_GEMM_HSACO=/absolute/canonical/gfx950-fp8-gemm.hsaco \
/// FE2O3_GFX950_FP8_GEMM_SHA256=<64-lowercase-hex-digits> \
/// cargo test -p fe2o3-hsa-runtime --features hardware-test-hooks \
///   --test gfx950_fp8_gemm_hardware \
///   gfx950_fp8_gemm_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output \
///   -- --ignored --exact --nocapture
/// ```
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- FP8 GEMM COV6 HSACO and MI350"]
fn gfx950_fp8_gemm_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output()
-> Result<(), BoxError> {
    let (bytes, digest) = read_pinned_hsaco()?;
    run_hardware(bytes, digest, KernelVariant::Tile)
}

/// Executes scalar OCP E4M3 with the same ABI, oracle, input checks and canaries.
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- FP8 SIMT GEMM COV6 HSACO and MI350"]
fn gfx950_fp8_gemm_simt_cov6_runs_multigrid_and_matches_every_cpu_reference_output()
-> Result<(), BoxError> {
    let (bytes, digest) = read_pinned_hsaco()?;
    run_hardware(bytes, digest, KernelVariant::Simt)
}

#[cfg(all(test, feature = "hardware-test-hooks"))]
mod tests {
    use super::*;

    #[test]
    fn e4m3_decoder_matches_canonical_exact_values_and_nan() {
        for (bits, expected) in [
            (0x00, 0.0_f32),
            (0x01, 1.0 / 512.0),
            (0x28, 0.25),
            (0x30, 0.5),
            (0x38, 1.0),
            (0x3c, 1.5),
            (0x40, 2.0),
            (0xb8, -1.0),
        ] {
            assert_eq!(decode_fp8_e4m3(bits).to_bits(), expected.to_bits());
        }
        assert!(decode_fp8_e4m3(0x7f).is_nan());
    }

    #[test]
    fn explicit_kernarg_is_the_exact_three_slice_cov6_prefix() {
        let bytes = explicit_kernarg([0x1111, 0x2222, 0x3333]);
        assert_eq!(bytes.len(), EXPLICIT_KERNARG_BYTES);
        for (index, (address, elements)) in [0x1111_u64, 0x2222, 0x3333]
            .into_iter()
            .zip([
                BATCHES * A_ELEMENTS,
                BATCHES * B_ELEMENTS,
                BATCHES * C_ELEMENTS,
            ])
            .enumerate()
        {
            let offset = index * 16;
            assert_eq!(&bytes[offset..offset + 8], &address.to_le_bytes());
            assert_eq!(
                &bytes[offset + 8..offset + 16],
                &(elements as u64).to_le_bytes(),
            );
        }
    }

    #[test]
    fn deterministic_fp8_oracle_is_finite_axis_varying_and_complete() {
        let (lhs, rhs) = deterministic_inputs();
        let output = cpu_reference(&lhs, &rhs);
        assert_eq!(
            (lhs.len(), rhs.len(), output.len()),
            (
                BATCHES * A_ELEMENTS,
                BATCHES * B_ELEMENTS,
                BATCHES * C_ELEMENTS,
            )
        );
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.windows(2).any(|pair| pair[0] != pair[1]));
        assert!(
            output[..N]
                .iter()
                .zip(&output[N..2 * N])
                .any(|(first, second)| first != second),
        );
    }

    #[test]
    fn output_validation_covers_every_element_and_both_canaries() {
        let (lhs, rhs) = deterministic_inputs();
        let expected = cpu_reference(&lhs, &rhs);
        let mut actual = Vec::with_capacity(BATCHES * C_ELEMENTS + 2 * CANARY_ELEMENTS);
        actual.extend(std::iter::repeat_n(C_PREFIX, CANARY_ELEMENTS));
        actual.extend_from_slice(&expected);
        actual.extend(std::iter::repeat_n(C_SUFFIX, CANARY_ELEMENTS));
        verify_output(&actual, &expected).unwrap();

        actual[CANARY_ELEMENTS + BATCHES * C_ELEMENTS - 1] += 2.0 * MAX_ABSOLUTE_ERROR;
        assert!(verify_output(&actual, &expected).is_err());
        actual[CANARY_ELEMENTS + BATCHES * C_ELEMENTS - 1] = expected[BATCHES * C_ELEMENTS - 1];
        actual[0] = 0.0;
        assert!(verify_output(&actual, &expected).is_err());
        actual[0] = C_PREFIX;
        *actual.last_mut().unwrap() = 0.0;
        assert!(verify_output(&actual, &expected).is_err());
    }
}

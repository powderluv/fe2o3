//! Non-authoritative gfx950 FP4/FP8 attention hardware and numerical harness.
//!
//! Each ignored test consumes one caller-supplied, digest-pinned COV6 HSACO.
//! This bypasses production prerequisite authentication and grants no protected
//! evidence. It exercises Rust-produced gfx950 attention through the reviewed
//! raw HSA adapter and compares every output with an independent CPU oracle.

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
const TOKENS: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const HEAD_DIMENSION: usize = 128;
#[cfg(feature = "hardware-test-hooks")]
const VALUE_COLUMNS: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const QUERY_ELEMENTS: usize = TOKENS * HEAD_DIMENSION;
#[cfg(feature = "hardware-test-hooks")]
const KEY_ELEMENTS: usize = TOKENS * HEAD_DIMENSION;
#[cfg(feature = "hardware-test-hooks")]
const VALUE_ELEMENTS: usize = TOKENS * VALUE_COLUMNS;
#[cfg(feature = "hardware-test-hooks")]
const OUTPUT_ELEMENTS: usize = TOKENS * VALUE_COLUMNS;
#[cfg(feature = "hardware-test-hooks")]
const BATCHES: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const WORKGROUP_X: u32 = 256;
#[cfg(feature = "hardware-test-hooks")]
const GRID_X: u32 = 4;
#[cfg(feature = "hardware-test-hooks")]
const EXPLICIT_KERNARG_BYTES: usize = 64;
#[cfg(feature = "hardware-test-hooks")]
const HSA_KERNARG_ALIGNMENT: u64 = 16;
#[cfg(feature = "hardware-test-hooks")]
const METADATA_KERNARG_ALIGNMENT: u64 = 8;
#[cfg(feature = "hardware-test-hooks")]
const CANARY_ELEMENTS: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const OUTPUT_PREFIX: f32 = f32::from_bits(0x7fc0_95a1);
#[cfg(feature = "hardware-test-hooks")]
const OUTPUT_SUFFIX: f32 = f32::from_bits(0x7fc0_95a2);
#[cfg(feature = "hardware-test-hooks")]
const OUTPUT_POISON: f32 = f32::from_bits(0x7fc0_95ff);
#[cfg(feature = "hardware-test-hooks")]
// Covers f32 MFMA accumulation, Wave64 reductions, and the device exp
// approximation while remaining tight enough to reject layout substitutions.
const MAX_ABSOLUTE_ERROR: f32 = 2.0e-3;
#[cfg(feature = "hardware-test-hooks")]
const MAX_RELATIVE_ERROR: f32 = 2.0e-3;

#[cfg(feature = "hardware-test-hooks")]
const RUN_ENV: &str = "FE2O3_RUN_GFX950_ATTENTION_HARDWARE";

#[cfg(feature = "hardware-test-hooks")]
type BoxError = Box<dyn std::error::Error>;

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionPrecision {
    Fp4E2M1,
    Fp8E4M3,
}

#[cfg(feature = "hardware-test-hooks")]
impl AttentionPrecision {
    fn decode(self, bits: u8) -> f32 {
        match self {
            Self::Fp4E2M1 => decode_fp4_e2m1(bits),
            Self::Fp8E4M3 => decode_fp8_e4m3(bits),
        }
    }

    fn input_codes(self) -> &'static [u8] {
        match self {
            Self::Fp4E2M1 => &[0x0, 0x1, 0x9, 0x0, 0x1, 0x9, 0x8],
            Self::Fp8E4M3 => &[0x00, 0x28, 0xa8, 0x30, 0xb0, 0x28, 0xa8],
        }
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AttentionCase {
    label: &'static str,
    export: &'static str,
    descriptor: &'static str,
    hsaco_env: &'static str,
    sha256_env: &'static str,
    static_lds_bytes: u64,
    precision: AttentionPrecision,
}

#[cfg(feature = "hardware-test-hooks")]
const FP4_CASE: AttentionCase = AttentionCase {
    label: "gfx950 FP4 attention",
    export: "gfx950_fp4_attention_rust",
    descriptor: "gfx950_fp4_attention_rust.kd",
    hsaco_env: "FE2O3_GFX950_FP4_ATTENTION_HSACO",
    sha256_env: "FE2O3_GFX950_FP4_ATTENTION_SHA256",
    static_lds_bytes: 4096,
    precision: AttentionPrecision::Fp4E2M1,
};

#[cfg(feature = "hardware-test-hooks")]
const FP8_CASE: AttentionCase = AttentionCase {
    label: "gfx950 FP8 attention",
    export: "gfx950_fp8_attention_rust",
    descriptor: "gfx950_fp8_attention_rust.kd",
    hsaco_env: "FE2O3_GFX950_FP8_ATTENTION_HSACO",
    sha256_env: "FE2O3_GFX950_FP8_ATTENTION_SHA256",
    static_lds_bytes: 8192,
    precision: AttentionPrecision::Fp8E4M3,
};

#[cfg(feature = "hardware-test-hooks")]
const FP8_SIMT_CASE: AttentionCase = AttentionCase {
    label: "gfx950 FP8 SIMT attention",
    export: "gfx950_fp8_attention_simt",
    descriptor: "gfx950_fp8_attention_simt.kd",
    static_lds_bytes: 0,
    ..FP8_CASE
};

#[cfg(feature = "hardware-test-hooks")]
const FP4_SIMT_CASE: AttentionCase = AttentionCase {
    label: "gfx950 FP4 SIMT attention",
    export: "gfx950_fp4_attention_simt",
    descriptor: "gfx950_fp4_attention_simt.kd",
    static_lds_bytes: 0,
    ..FP4_CASE
};

#[cfg(feature = "hardware-test-hooks")]
fn require(condition: bool, message: impl Into<String>) -> Result<(), BoxError> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn parse_sha256(case: AttentionCase, text: &str) -> Result<[u8; 32], BoxError> {
    require(
        text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        format!(
            "{} must be exactly 64 lowercase hex digits",
            case.sha256_env
        ),
    )?;
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("{} is malformed", case.sha256_env))?;
    }
    Ok(bytes)
}

#[cfg(feature = "hardware-test-hooks")]
fn read_pinned_hsaco(case: AttentionCase) -> Result<(Vec<u8>, PayloadDigest), BoxError> {
    require(
        std::env::var(RUN_ENV).as_deref() == Ok("1"),
        format!("set {RUN_ENV}=1 to opt into this raw hardware test"),
    )?;
    let path = std::path::PathBuf::from(
        std::env::var_os(case.hsaco_env).ok_or_else(|| format!("{} is not set", case.hsaco_env))?,
    );
    require(
        path.is_absolute(),
        format!("{} must be an absolute path", case.hsaco_env),
    )?;
    let metadata = std::fs::symlink_metadata(&path)?;
    require(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        format!("{} must name a regular non-symlink file", case.hsaco_env),
    )?;
    require(
        std::fs::canonicalize(&path)? == path,
        format!("{} must already be canonical", case.hsaco_env),
    )?;
    require(
        (1..=fe2o3_hsaco::MAX_HSACO_BYTES as u64).contains(&metadata.len()),
        format!("{} has an invalid byte length", case.hsaco_env),
    )?;
    let expected = parse_sha256(
        case,
        &std::env::var(case.sha256_env).map_err(|_| format!("{} is not set", case.sha256_env))?,
    )?;
    let bytes = std::fs::read(&path)?;
    require(
        bytes.len() as u64 == metadata.len(),
        format!("{} changed size while being read", case.hsaco_env),
    )?;
    let final_metadata = std::fs::symlink_metadata(&path)?;
    require(
        final_metadata.file_type().is_file()
            && !final_metadata.file_type().is_symlink()
            && final_metadata.len() == metadata.len()
            && std::fs::canonicalize(&path)? == path,
        format!("{} changed identity while being read", case.hsaco_env),
    )?;
    let digest = DigestAlgorithm::Sha256.calculate(&bytes);
    require(
        digest.bytes().as_bytes() == &expected,
        format!("{} does not match its exact SHA-256 pin", case.hsaco_env),
    )?;
    Ok((bytes, digest))
}

#[cfg(feature = "hardware-test-hooks")]
fn inspect_profile(case: AttentionCase, bytes: &[u8]) -> Result<(), BoxError> {
    let bound = fe2o3_hsaco::inspect_and_bind_kernel_descriptors(bytes)?;
    let inspected = bound.inspection();
    require(
        inspected.code_object_version() == CodeObjectVersion::V6,
        format!("{} HSACO must be code object V6", case.label),
    )?;
    require(
        inspected.target().processor() == "gfx950"
            && inspected.target().xnack() == Some(FeatureState::Disabled),
        format!("{} HSACO must target exact gfx950:xnack-", case.label),
    )?;
    require(
        !inspected.has_printf_metadata(),
        format!("{} HSACO must not carry printf metadata", case.label),
    )?;
    let [kernel] = inspected.kernels() else {
        return Err(format!("{} HSACO must declare exactly one kernel", case.label).into());
    };
    require(
        kernel.name() == case.export && kernel.symbol() == case.descriptor,
        format!(
            "{} HSACO has a substituted kernel or descriptor symbol",
            case.label
        ),
    )?;
    require(
        kernel.kernarg_segment_size() == EXPLICIT_KERNARG_BYTES as u64
            && kernel.kernarg_segment_alignment() == METADATA_KERNARG_ALIGNMENT
            && kernel.implicit_argument_offset().is_none()
            && kernel.implicit_argument_size() == 0,
        format!(
            "{} HSACO does not expose the exact 64-byte explicit kernarg",
            case.label
        ),
    )?;
    require(
        kernel.required_workgroup_size() == Some([WORKGROUP_X, 1, 1])
            && kernel.max_flat_workgroup_size() == WORKGROUP_X
            && kernel.wavefront_size() == 64
            && kernel.group_segment_fixed_size() == case.static_lds_bytes
            && !kernel.uses_dynamic_stack(),
        format!(
            "{} HSACO does not expose its exact static WG256/LDS profile",
            case.label
        ),
    )?;
    const EXPECTED_ARGUMENTS: [(u64, u64, ExplicitValueKind); 8] = [
        (0, 8, ExplicitValueKind::GlobalBuffer),
        (8, 8, ExplicitValueKind::ByValue),
        (16, 8, ExplicitValueKind::GlobalBuffer),
        (24, 8, ExplicitValueKind::ByValue),
        (32, 8, ExplicitValueKind::GlobalBuffer),
        (40, 8, ExplicitValueKind::ByValue),
        (48, 8, ExplicitValueKind::GlobalBuffer),
        (56, 8, ExplicitValueKind::ByValue),
    ];
    let arguments = kernel
        .explicit_arguments()
        .iter()
        .map(|argument| (argument.offset(), argument.size(), argument.value_kind()))
        .collect::<Vec<_>>();
    require(
        arguments.as_slice() == EXPECTED_ARGUMENTS,
        format!("{} HSACO has a substituted four-slice ABI", case.label),
    )?;
    let [binding] = bound.bindings() else {
        return Err(format!("{} HSACO must bind exactly one descriptor", case.label).into());
    };
    let descriptor = binding.descriptor();
    require(
        binding.kernel_index() == 0
            && descriptor.kernarg_size() == EXPLICIT_KERNARG_BYTES as u32
            && u64::from(descriptor.group_segment_fixed_size()) == case.static_lds_bytes
            && u64::from(descriptor.private_segment_fixed_size())
                == kernel.private_segment_fixed_size()
            && descriptor.wavefront_size() == 64
            && !descriptor.uses_dynamic_stack(),
        format!("{} descriptor disagrees with its metadata", case.label),
    )
}

#[cfg(feature = "hardware-test-hooks")]
fn decode_fp4_e2m1(bits: u8) -> f32 {
    let magnitude = match bits & 0x7 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        _ => unreachable!("three-bit FP4 E2M1 magnitude"),
    };
    if bits & 0x8 == 0 {
        magnitude
    } else {
        -magnitude
    }
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
struct AttentionInputs {
    query: Vec<u8>,
    key: Vec<u8>,
    value: Vec<u8>,
}

#[cfg(feature = "hardware-test-hooks")]
fn deterministic_inputs(case: AttentionCase) -> AttentionInputs {
    let codes = case.precision.input_codes();
    let mut query = Vec::with_capacity(BATCHES * QUERY_ELEMENTS);
    let mut key = Vec::with_capacity(BATCHES * KEY_ELEMENTS);
    let mut value = Vec::with_capacity(BATCHES * VALUE_ELEMENTS);
    for batch in 0..BATCHES {
        query.extend((0..QUERY_ELEMENTS).map(|index| {
            let row = index / HEAD_DIMENSION;
            let depth = index % HEAD_DIMENSION;
            codes[(row * 11
                + depth * 5
                + row * depth * 3
                + 1
                + batch
                + (batch / codes.len()) * (depth % 5))
                % codes.len()]
        }));
        key.extend((0..KEY_ELEMENTS).map(|index| {
            let token = index / HEAD_DIMENSION;
            let depth = index % HEAD_DIMENSION;
            codes[(token * 7
                + depth * 3
                + token * depth * 5
                + 2
                + batch * 2
                + (batch / codes.len()) * (depth % 5))
                % codes.len()]
        }));
        value.extend((0..VALUE_ELEMENTS).map(|index| {
            let token = index / VALUE_COLUMNS;
            let column = index % VALUE_COLUMNS;
            codes[(token * 5
                + column * 2
                + token * column * 3
                + 4
                + batch * 3
                + (batch / codes.len()) * (column % 5))
                % codes.len()]
        }));
    }
    AttentionInputs { query, key, value }
}

#[cfg(feature = "hardware-test-hooks")]
fn cpu_reference(case: AttentionCase, inputs: &AttentionInputs) -> Vec<f32> {
    let scale = (HEAD_DIMENSION as f64).sqrt().recip();
    let mut output = vec![0.0; BATCHES * OUTPUT_ELEMENTS];
    let mut scores = [0.0_f64; TOKENS];
    for batch in 0..BATCHES {
        let query_base = batch * QUERY_ELEMENTS;
        let key_base = batch * KEY_ELEMENTS;
        let value_base = batch * VALUE_ELEMENTS;
        let output_base = batch * OUTPUT_ELEMENTS;
        for row in 0..TOKENS {
            for token in 0..TOKENS {
                let mut dot = 0.0_f64;
                for depth in 0..HEAD_DIMENSION {
                    let query = case
                        .precision
                        .decode(inputs.query[query_base + row * HEAD_DIMENSION + depth]);
                    let key = case
                        .precision
                        .decode(inputs.key[key_base + token * HEAD_DIMENSION + depth]);
                    dot += f64::from(query) * f64::from(key);
                }
                scores[token] = dot * scale;
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut denominator = 0.0_f64;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            for column in 0..VALUE_COLUMNS {
                let mut result = 0.0_f64;
                for token in 0..TOKENS {
                    let value = case
                        .precision
                        .decode(inputs.value[value_base + token * VALUE_COLUMNS + column]);
                    result += (scores[token] / denominator) * f64::from(value);
                }
                output[output_base + row * VALUE_COLUMNS + column] = result as f32;
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
fn f32_values(case: AttentionCase, bytes: &[u8]) -> Result<Vec<f32>, BoxError> {
    require(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        format!("{} output contains a partial f32", case.label),
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
fn explicit_kernarg(addresses: [u64; 4]) -> [u8; EXPLICIT_KERNARG_BYTES] {
    let mut bytes = [0; EXPLICIT_KERNARG_BYTES];
    for (index, (address, elements)) in addresses
        .into_iter()
        .zip([
            BATCHES * QUERY_ELEMENTS,
            BATCHES * KEY_ELEMENTS,
            BATCHES * VALUE_ELEMENTS,
            BATCHES * OUTPUT_ELEMENTS,
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
    fn new(case: AttentionCase) -> Result<Self, BoxError> {
        let layout = std::alloc::Layout::from_size_align(
            EXPLICIT_KERNARG_BYTES,
            HSA_KERNARG_ALIGNMENT as usize,
        )?;
        // SAFETY: layout is valid and this owner deallocates the result once.
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .ok_or_else(|| format!("failed to allocate aligned {} kernarg", case.label))?;
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
    case: AttentionCase,
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    resolution: &HsaKernelResolutionObservationV1,
    explicit: &[u8; EXPLICIT_KERNARG_BYTES],
) -> Result<(), BoxError> {
    require(
        resolution.export_symbol() == case.export
            && resolution.kernarg_segment_size() == EXPLICIT_KERNARG_BYTES as u64
            && resolution.kernarg_segment_alignment() == HSA_KERNARG_ALIGNMENT
            && resolution.group_segment_size() == case.static_lds_bytes,
        format!(
            "runtime resolution differs from the exact {} ABI",
            case.label
        ),
    )?;
    let geometry = HsaLaunchGeometryV1::new([GRID_X, 1, 1], [WORKGROUP_X, 1, 1], 0);
    let mut storage = RuntimeKernarg::new(case)?;
    let kernarg = storage.bytes_mut();
    kernarg.copy_from_slice(explicit);
    let original = *explicit;

    // SAFETY: inspection admitted one digest-pinned COV6 image with the exact
    // explicit-only four-slice ABI. Preparation binds the exact launch queue
    // without mutating the empty implicit span. All buffers stay live and
    // dispatch completion is synchronous.
    unsafe {
        let initialization = adapter.initialize_implicit_kernarg(
            executable,
            kernel,
            geometry,
            EXPLICIT_KERNARG_BYTES,
            EXPLICIT_KERNARG_BYTES,
            0,
            kernarg,
        )?;
        require(
            initialization.initialized()
                && initialization.explicit_byte_len() == EXPLICIT_KERNARG_BYTES as u64
                && initialization.implicit_byte_offset() == EXPLICIT_KERNARG_BYTES as u64
                && initialization.implicit_byte_len() == 0
                && kernarg == original,
            format!(
                "{} explicit-only kernarg initialization changed its ABI",
                case.label
            ),
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
                case.export,
                durations[SAMPLES / 2],
                durations[SAMPLES / 20],
                durations[SAMPLES * 19 / 20],
            );
        }
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn output_body_address(
    case: AttentionCase,
    buffer: &ReviewedHsaHardwareTestBufferV1,
) -> Result<u64, BoxError> {
    require(
        buffer.byte_len()
            == (BATCHES * OUTPUT_ELEMENTS + 2 * CANARY_ELEMENTS) * std::mem::size_of::<f32>(),
        format!("guarded {} output has the wrong extent", case.label),
    )?;
    Ok(buffer.device_address(CANARY_ELEMENTS * std::mem::size_of::<f32>())?)
}

#[cfg(feature = "hardware-test-hooks")]
fn verify_output(case: AttentionCase, actual: &[f32], expected: &[f32]) -> Result<(), BoxError> {
    require(
        actual.len() == BATCHES * OUTPUT_ELEMENTS + 2 * CANARY_ELEMENTS,
        format!("guarded {} output length changed", case.label),
    )?;
    let (prefix, remainder) = actual.split_at(CANARY_ELEMENTS);
    let (body, suffix) = remainder.split_at(BATCHES * OUTPUT_ELEMENTS);
    require(
        prefix
            .iter()
            .all(|value| value.to_bits() == OUTPUT_PREFIX.to_bits()),
        format!("{} output prefix canary changed", case.label),
    )?;
    require(
        suffix
            .iter()
            .all(|value| value.to_bits() == OUTPUT_SUFFIX.to_bits()),
        format!("{} output suffix canary changed", case.label),
    )?;
    require(
        expected.len() == BATCHES * OUTPUT_ELEMENTS,
        format!("{} CPU reference has the wrong extent", case.label),
    )?;
    let mut maximum_absolute_error = 0.0_f32;
    for (index, (actual, expected)) in body.iter().zip(expected).enumerate() {
        let absolute_error = (actual - expected).abs();
        let tolerance = MAX_ABSOLUTE_ERROR + MAX_RELATIVE_ERROR * expected.abs();
        require(
            actual.is_finite() && expected.is_finite() && absolute_error <= tolerance,
            format!(
                "{} O[{index}] mismatch: actual={actual}, expected={expected}, tolerance={tolerance}",
                case.label
            ),
        )?;
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
    }
    println!(
        "PASS {} outputs={} max_absolute_error={maximum_absolute_error:.9e} abs_tolerance={MAX_ABSOLUTE_ERROR:.1e} rel_tolerance={MAX_RELATIVE_ERROR:.1e}",
        case.label,
        BATCHES * OUTPUT_ELEMENTS
    );
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn run_hardware(
    case: AttentionCase,
    bytes: Vec<u8>,
    digest: PayloadDigest,
) -> Result<(), BoxError> {
    inspect_profile(case, &bytes)?;
    let context = GpuContext::new(0)?;
    let mut adapter = ReviewedHsaRuntimeAdapterV1::new_gfx950(context)?;
    require(
        adapter.environment().physical_device().target().processor() == "gfx950"
            && adapter.environment().physical_device().target().xnack()
                == Some(FeatureState::Disabled),
        format!(
            "{} hardware test requires a gfx950:xnack- physical device",
            case.label
        ),
    )?;

    let inputs = deterministic_inputs(case);
    let expected = cpu_reference(case, &inputs);
    let mut output_host = Vec::with_capacity(BATCHES * OUTPUT_ELEMENTS + 2 * CANARY_ELEMENTS);
    output_host.extend(std::iter::repeat_n(OUTPUT_PREFIX, CANARY_ELEMENTS));
    output_host.extend(std::iter::repeat_n(
        OUTPUT_POISON,
        BATCHES * OUTPUT_ELEMENTS,
    ));
    output_host.extend(std::iter::repeat_n(OUTPUT_SUFFIX, CANARY_ELEMENTS));
    let query = adapter.allocate_hardware_test_buffer(&inputs.query)?;
    let key = adapter.allocate_hardware_test_buffer(&inputs.key)?;
    let value = adapter.allocate_hardware_test_buffer(&inputs.value)?;
    let output = adapter.allocate_hardware_test_buffer(f32_bytes(&output_host))?;
    let explicit = explicit_kernarg([
        query.device_address(0)?,
        key.device_address(0)?,
        value.device_address(0)?,
        output_body_address(case, &output)?,
    ]);

    // SAFETY: immutable, digest-pinned bytes and every allocation are retained
    // until the synchronous dispatch and sole consuming unload complete.
    let (executable, load) = unsafe { adapter.load_executable(&bytes, digest) }?;
    let executable_identity = load.executable_object();
    let execution = (|| -> Result<(), BoxError> {
        require(
            load.finalized_digest() == digest && load.byte_len() == bytes.len() as u64,
            format!("{} load observation changed", case.label),
        )?;
        // SAFETY: inspection admitted exactly this export and descriptor.
        let (kernels, resolutions) =
            unsafe { adapter.resolve_kernel_set(&executable, [case.export]) }?;
        let kernel = kernels
            .get(0)
            .ok_or_else(|| format!("runtime omitted {}", case.label))?;
        require(
            kernels.len() == 1
                && resolutions.len() == 1
                && resolutions[0].executable_object() == executable_identity,
            format!("runtime resolved a substituted {} kernel", case.label),
        )?;
        // SAFETY: dispatch_multigrid owns the reviewed raw launch boundary.
        unsafe {
            dispatch_multigrid(
                case,
                &mut adapter,
                &executable,
                kernel,
                &resolutions[0],
                &explicit,
            )?;
        }
        require(
            query.read_after_synchronous_dispatch() == inputs.query,
            format!("{} modified query", case.label),
        )?;
        require(
            key.read_after_synchronous_dispatch() == inputs.key,
            format!("{} modified key", case.label),
        )?;
        require(
            value.read_after_synchronous_dispatch() == inputs.value,
            format!("{} modified value", case.label),
        )?;
        verify_output(
            case,
            &f32_values(case, &output.read_after_synchronous_dispatch())?,
            &expected,
        )
    })();
    let unload = unsafe { adapter.unload_executable(executable) }?;
    require(
        unload.released() && unload.executable_object() == executable_identity,
        format!(
            "reviewed HSA unload did not release the {} executable",
            case.label
        ),
    )?;
    execution
}

/// Runs Rust-produced FP4 attention as four WG256 workgroups.
///
/// This ignored test is non-authoritative and grants no protected evidence.
/// Invoke it with:
///
/// ```text
/// FE2O3_RUN_GFX950_ATTENTION_HARDWARE=1 \
/// FE2O3_GFX950_FP4_ATTENTION_HSACO=/absolute/canonical/gfx950-fp4-attention.hsaco \
/// FE2O3_GFX950_FP4_ATTENTION_SHA256=<64-lowercase-hex-digits> \
/// cargo test -p fe2o3-hsa-runtime --features hardware-test-hooks \
///   --test gfx950_attention_hardware \
///   gfx950_fp4_attention_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output \
///   -- --ignored --exact --nocapture
/// ```
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- FP4 attention COV6 HSACO and MI350"]
fn gfx950_fp4_attention_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output()
-> Result<(), BoxError> {
    let (bytes, digest) = read_pinned_hsaco(FP4_CASE)?;
    run_hardware(FP4_CASE, bytes, digest)
}

/// Runs Rust-produced FP8 attention as four WG256 workgroups.
///
/// This ignored test is non-authoritative and grants no protected evidence.
/// Invoke it with:
///
/// ```text
/// FE2O3_RUN_GFX950_ATTENTION_HARDWARE=1 \
/// FE2O3_GFX950_FP8_ATTENTION_HSACO=/absolute/canonical/gfx950-fp8-attention.hsaco \
/// FE2O3_GFX950_FP8_ATTENTION_SHA256=<64-lowercase-hex-digits> \
/// cargo test -p fe2o3-hsa-runtime --features hardware-test-hooks \
///   --test gfx950_attention_hardware \
///   gfx950_fp8_attention_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output \
///   -- --ignored --exact --nocapture
/// ```
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- FP8 attention COV6 HSACO and MI350"]
fn gfx950_fp8_attention_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output()
-> Result<(), BoxError> {
    let (bytes, digest) = read_pinned_hsaco(FP8_CASE)?;
    run_hardware(FP8_CASE, bytes, digest)
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- FP8 SIMT attention COV6 HSACO and MI350"]
fn gfx950_fp8_attention_simt_cov6_runs_multigrid_and_matches_every_cpu_reference_output()
-> Result<(), BoxError> {
    let (bytes, digest) = read_pinned_hsaco(FP8_SIMT_CASE)?;
    run_hardware(FP8_SIMT_CASE, bytes, digest)
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- FP4 SIMT attention COV6 HSACO and MI350"]
fn gfx950_fp4_attention_simt_cov6_runs_multigrid_and_matches_every_cpu_reference_output()
-> Result<(), BoxError> {
    let (bytes, digest) = read_pinned_hsaco(FP4_SIMT_CASE)?;
    run_hardware(FP4_SIMT_CASE, bytes, digest)
}

#[cfg(all(test, feature = "hardware-test-hooks"))]
mod tests {
    use super::*;

    #[test]
    fn low_precision_decoders_match_canonical_values_and_fp8_nan() {
        for (bits, expected) in [
            (0x0, 0.0_f32),
            (0x1, 0.5),
            (0x7, 6.0),
            (0x8, -0.0),
            (0x9, -0.5),
            (0xf, -6.0),
        ] {
            assert_eq!(decode_fp4_e2m1(bits).to_bits(), expected.to_bits());
        }
        for (bits, expected) in [
            (0x00, 0.0_f32),
            (0x01, 1.0 / 512.0),
            (0x28, 0.25),
            (0x30, 0.5),
            (0x38, 1.0),
            (0x3c, 1.5),
            (0xb8, -1.0),
        ] {
            assert_eq!(decode_fp8_e4m3(bits).to_bits(), expected.to_bits());
        }
        assert!(decode_fp8_e4m3(0x7f).is_nan());
    }

    #[test]
    fn exact_cases_are_distinct_and_bind_reviewed_profiles() {
        assert_eq!(FP4_CASE.export, "gfx950_fp4_attention_rust");
        assert_eq!(FP8_CASE.export, "gfx950_fp8_attention_rust");
        assert_eq!(FP4_CASE.static_lds_bytes, 4096);
        assert_eq!(FP8_CASE.static_lds_bytes, 8192);
        assert_ne!(FP4_CASE, FP8_CASE);
        assert_ne!(FP4_CASE.hsaco_env, FP8_CASE.hsaco_env);
        assert_ne!(FP4_CASE.sha256_env, FP8_CASE.sha256_env);
    }

    #[test]
    fn explicit_kernarg_is_the_exact_four_slice_cov6_prefix() {
        let bytes = explicit_kernarg([0x1111, 0x2222, 0x3333, 0x4444]);
        for (index, (address, elements)) in [0x1111_u64, 0x2222, 0x3333, 0x4444]
            .into_iter()
            .zip([
                BATCHES * QUERY_ELEMENTS,
                BATCHES * KEY_ELEMENTS,
                BATCHES * VALUE_ELEMENTS,
                BATCHES * OUTPUT_ELEMENTS,
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
    fn stable_attention_oracles_are_finite_axis_varying_and_complete() {
        for case in [FP4_CASE, FP8_CASE] {
            let inputs = deterministic_inputs(case);
            let output = cpu_reference(case, &inputs);
            assert_eq!(inputs.query.len(), BATCHES * QUERY_ELEMENTS);
            assert_eq!(inputs.key.len(), BATCHES * KEY_ELEMENTS);
            assert_eq!(inputs.value.len(), BATCHES * VALUE_ELEMENTS);
            assert_eq!(output.len(), BATCHES * OUTPUT_ELEMENTS);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.windows(2).any(|pair| pair[0] != pair[1]));
            assert!(
                output[..VALUE_COLUMNS]
                    .iter()
                    .zip(&output[VALUE_COLUMNS..2 * VALUE_COLUMNS])
                    .any(|(first, second)| first != second),
                "{} oracle must vary across query rows",
                case.label
            );
        }
    }

    #[test]
    fn output_validation_checks_every_element_finiteness_and_both_canaries() {
        for case in [FP4_CASE, FP8_CASE] {
            let inputs = deterministic_inputs(case);
            let expected = cpu_reference(case, &inputs);
            let mut actual = Vec::with_capacity(BATCHES * OUTPUT_ELEMENTS + 2 * CANARY_ELEMENTS);
            actual.extend(std::iter::repeat_n(OUTPUT_PREFIX, CANARY_ELEMENTS));
            actual.extend_from_slice(&expected);
            actual.extend(std::iter::repeat_n(OUTPUT_SUFFIX, CANARY_ELEMENTS));
            verify_output(case, &actual, &expected).unwrap();

            let last = CANARY_ELEMENTS + BATCHES * OUTPUT_ELEMENTS - 1;
            actual[last] = f32::NAN;
            assert!(verify_output(case, &actual, &expected).is_err());
            actual[last] = expected[BATCHES * OUTPUT_ELEMENTS - 1];
            actual[0] = 0.0;
            assert!(verify_output(case, &actual, &expected).is_err());
            actual[0] = OUTPUT_PREFIX;
            *actual.last_mut().unwrap() = 0.0;
            assert!(verify_output(case, &actual, &expected).is_err());
        }
    }

    #[test]
    fn sha256_parser_rejects_noncanonical_and_truncated_pins() {
        let canonical = "ab".repeat(32);
        assert_eq!(parse_sha256(FP4_CASE, &canonical).unwrap(), [0xab; 32]);
        assert!(parse_sha256(FP4_CASE, &canonical.to_uppercase()).is_err());
        assert!(parse_sha256(FP8_CASE, &canonical[..62]).is_err());
        assert!(parse_sha256(FP8_CASE, &"gg".repeat(32)).is_err());
    }
}

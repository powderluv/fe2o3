//! Safe Rust fixed-shape gfx950 kernels.

#![allow(missing_docs)]

use fe2o3_device::{
    Blocked, DisjointSlice, Gfx950F32AccumulatorFragment, Gfx950Fp4E2M1, Gfx950Fp4MfmaAMatrix,
    Gfx950Fp4MfmaBMatrix, Gfx950Fp8E4M3, Gfx950Fp8MfmaAMatrix, Gfx950Fp8MfmaBMatrix,
    Gfx950LdsTransposeTile, Gfx950Matrix, Gfx950Subgroup, Gfx950TransposeUninitialized, Index1D,
    KernelError, KernelResult, Math, StridedReadView2D, Wave64, WaveLane, kernel, thread,
};

pub const GFX950_WORKGROUP: [u32; 3] = [64, 1, 1];
pub const GEMM_M: usize = 16;
pub const GEMM_N: usize = 16;
pub const GEMM_K: usize = 128;
pub const ATTENTION_TOKENS: usize = 16;
pub const VALUE_COLUMNS: usize = 16;
const ATTENTION_SCALE: f32 = 0.088_388_35;

fn decode_fp4_e2m1(bits: u8) -> f32 {
    let payload = bits & 0x7;
    let exponent = payload >> 1_u8;
    let mantissa = f32::from(payload & 1);
    let magnitude = if exponent == 0 {
        mantissa * 0.5
    } else {
        let scale = f32::from(1_u8 << (exponent - 1_u8));
        (1.0 + mantissa * 0.5) * scale
    };
    if bits & 0x8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-fp4-gemm"))]
#[kernel(
    typed,
    namespace = "ff22ff3610dda0a94803a8011ced229b78c77400ca63c9b929d6ecba78ed6f01",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
)]
pub fn gfx950_fp4_gemm_rust(
    lhs: &[u8],
    rhs: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    if lhs.len() < GEMM_M * GEMM_K || rhs.len() < GEMM_K * GEMM_N || output.len() < GEMM_M * GEMM_N
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let lane = WaveLane::<Wave64>::current();
    let Ok(lhs_matrix) = Gfx950Fp4MfmaAMatrix::row_major(lhs, 0, GEMM_M, GEMM_K, GEMM_K) else {
        return Err(KernelError::InvalidArgument);
    };
    let lhs = lhs_matrix.load_m16k128(&lane, 0, 0);
    let Ok(rhs_matrix) = Gfx950Fp4MfmaBMatrix::row_major(rhs, 0, GEMM_K, GEMM_N, GEMM_N) else {
        return Err(KernelError::InvalidArgument);
    };
    let rhs = rhs_matrix.load_k128n16(&lane, 0, 0);
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane);
    let values = Gfx950Matrix::current()
        .multiply_accumulate_fp4(lhs, rhs, accumulator)
        .into_values();
    let Some(output_block) = index.checked_block::<16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    if let Some(element) = output.get_block_mut(&output_block, 0) {
        *element = values[0];
    }
    if let Some(element) = output.get_block_mut(&output_block, 1) {
        *element = values[1];
    }
    if let Some(element) = output.get_block_mut(&output_block, 2) {
        *element = values[2];
    }
    if let Some(element) = output.get_block_mut(&output_block, 3) {
        *element = values[3];
    }
    Ok(())
}

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-fp8-gemm"))]
#[kernel(
    typed,
    namespace = "d67f1755b38fbdac67cec83da3ebc359f874e3fbf90fcc036471455ec117dfea",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
)]
pub fn gfx950_fp8_gemm_rust(
    lhs: &[u8],
    rhs: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    if lhs.len() < GEMM_M * GEMM_K || rhs.len() < GEMM_K * GEMM_N || output.len() < GEMM_M * GEMM_N
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let lane = WaveLane::<Wave64>::current();
    let Ok(lhs_matrix) = Gfx950Fp8MfmaAMatrix::row_major(lhs, 0, GEMM_M, GEMM_K, GEMM_K) else {
        return Err(KernelError::InvalidArgument);
    };
    let lhs = lhs_matrix.load_m16k128(&lane, 0, 0);
    let Ok(rhs_matrix) = Gfx950Fp8MfmaBMatrix::row_major(rhs, 0, GEMM_K, GEMM_N, GEMM_N) else {
        return Err(KernelError::InvalidArgument);
    };
    let rhs = rhs_matrix.load_k128n16(&lane, 0, 0);
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp8E4M3>::zero(&lane);
    let values = Gfx950Matrix::current()
        .multiply_accumulate_fp8(lhs, rhs, accumulator)
        .into_values();
    let Some(output_block) = index.checked_block::<16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    if let Some(element) = output.get_block_mut(&output_block, 0) {
        *element = values[0];
    }
    if let Some(element) = output.get_block_mut(&output_block, 1) {
        *element = values[1];
    }
    if let Some(element) = output.get_block_mut(&output_block, 2) {
        *element = values[2];
    }
    if let Some(element) = output.get_block_mut(&output_block, 3) {
        *element = values[3];
    }
    Ok(())
}

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-fp4-attention"))]
#[kernel(
    typed,
    namespace = "a9a878f0e2fc3a42ad17edf0a326a89695398bb6d7460eaf278ea3e8c53f4cf5",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
)]
pub fn gfx950_fp4_attention_rust(
    query: &[u8],
    key: &[u8],
    value: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) {
    // Invalid launch buffers abort the full wave before collective work begins.
    if query.len() < ATTENTION_TOKENS * GEMM_K
        || key.len() < ATTENTION_TOKENS * GEMM_K
        || value.len() < ATTENTION_TOKENS * VALUE_COLUMNS
        || output.len() < ATTENTION_TOKENS * VALUE_COLUMNS
    {
        fe2o3_device::trap();
    }
    let index = thread::index_1d();
    let lane_column = index.get() % 16;
    let lane = WaveLane::<Wave64>::current();
    let Ok(query_matrix) =
        Gfx950Fp4MfmaAMatrix::row_major(query, 0, ATTENTION_TOKENS, GEMM_K, GEMM_K)
    else {
        fe2o3_device::trap();
    };
    let query = query_matrix.load_m16k128(&lane, 0, 0);
    let Ok(key) = Gfx950Fp4MfmaAMatrix::row_major(key, 0, ATTENTION_TOKENS, GEMM_K, GEMM_K)
    else {
        fe2o3_device::trap();
    };
    let key = Gfx950LdsTransposeTile::<Gfx950Fp4E2M1, Gfx950TransposeUninitialized>::current(&lane)
        .stage_k_transposed(&key, 0, 0)
        .publish()
        .read_mfma_fragment();
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane);
    let scores = Gfx950Matrix::current()
        .multiply_accumulate_fp4(query, key, accumulator)
        .into_values();
    let Ok(value) = StridedReadView2D::from_shared_slice(
        value,
        0,
        ATTENTION_TOKENS,
        VALUE_COLUMNS,
        VALUE_COLUMNS,
    ) else {
        fe2o3_device::trap();
    };
    let subgroup = Gfx950Subgroup::current();
    let math = Math::current();
    let maximum0 = subgroup.reduce_max_f32::<16>(scores[0] * ATTENTION_SCALE);
    let maximum1 = subgroup.reduce_max_f32::<16>(scores[1] * ATTENTION_SCALE);
    let maximum2 = subgroup.reduce_max_f32::<16>(scores[2] * ATTENTION_SCALE);
    let maximum3 = subgroup.reduce_max_f32::<16>(scores[3] * ATTENTION_SCALE);
    let probability0 = math.exp_f32(scores[0] * ATTENTION_SCALE - maximum0);
    let probability1 = math.exp_f32(scores[1] * ATTENTION_SCALE - maximum1);
    let probability2 = math.exp_f32(scores[2] * ATTENTION_SCALE - maximum2);
    let probability3 = math.exp_f32(scores[3] * ATTENTION_SCALE - maximum3);
    let normalized0 = probability0 / subgroup.reduce_sum_f32::<16>(probability0);
    let normalized1 = probability1 / subgroup.reduce_sum_f32::<16>(probability1);
    let normalized2 = probability2 / subgroup.reduce_sum_f32::<16>(probability2);
    let normalized3 = probability3 / subgroup.reduce_sum_f32::<16>(probability3);
    let mut result0 = 0.0;
    let mut result1 = 0.0;
    let mut result2 = 0.0;
    let mut result3 = 0.0;
    let value0 = decode_fp4_e2m1(value.load_or(0, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 0) * value0;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 0) * value0;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 0) * value0;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 0) * value0;
    let value1 = decode_fp4_e2m1(value.load_or(1, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 1) * value1;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 1) * value1;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 1) * value1;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 1) * value1;
    let value2 = decode_fp4_e2m1(value.load_or(2, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 2) * value2;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 2) * value2;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 2) * value2;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 2) * value2;
    let value3 = decode_fp4_e2m1(value.load_or(3, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 3) * value3;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 3) * value3;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 3) * value3;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 3) * value3;
    let value4 = decode_fp4_e2m1(value.load_or(4, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 4) * value4;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 4) * value4;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 4) * value4;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 4) * value4;
    let value5 = decode_fp4_e2m1(value.load_or(5, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 5) * value5;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 5) * value5;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 5) * value5;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 5) * value5;
    let value6 = decode_fp4_e2m1(value.load_or(6, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 6) * value6;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 6) * value6;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 6) * value6;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 6) * value6;
    let value7 = decode_fp4_e2m1(value.load_or(7, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 7) * value7;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 7) * value7;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 7) * value7;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 7) * value7;
    let value8 = decode_fp4_e2m1(value.load_or(8, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 8) * value8;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 8) * value8;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 8) * value8;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 8) * value8;
    let value9 = decode_fp4_e2m1(value.load_or(9, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 9) * value9;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 9) * value9;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 9) * value9;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 9) * value9;
    let value10 = decode_fp4_e2m1(value.load_or(10, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 10) * value10;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 10) * value10;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 10) * value10;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 10) * value10;
    let value11 = decode_fp4_e2m1(value.load_or(11, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 11) * value11;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 11) * value11;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 11) * value11;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 11) * value11;
    let value12 = decode_fp4_e2m1(value.load_or(12, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 12) * value12;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 12) * value12;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 12) * value12;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 12) * value12;
    let value13 = decode_fp4_e2m1(value.load_or(13, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 13) * value13;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 13) * value13;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 13) * value13;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 13) * value13;
    let value14 = decode_fp4_e2m1(value.load_or(14, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 14) * value14;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 14) * value14;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 14) * value14;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 14) * value14;
    let value15 = decode_fp4_e2m1(value.load_or(15, lane_column, 0));
    result0 += subgroup.broadcast_f32::<16>(normalized0, 15) * value15;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 15) * value15;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 15) * value15;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 15) * value15;
    let Some(output_block) = index.checked_block::<16, 4>() else {
        fe2o3_device::trap();
    };
    if let Some(element) = output.get_block_mut(&output_block, 0) {
        *element = result0;
    }
    if let Some(element) = output.get_block_mut(&output_block, 1) {
        *element = result1;
    }
    if let Some(element) = output.get_block_mut(&output_block, 2) {
        *element = result2;
    }
    if let Some(element) = output.get_block_mut(&output_block, 3) {
        *element = result3;
    }
}

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-fp8-attention"))]
#[kernel(
    typed,
    namespace = "0c9610e86137831ce25b08b9ad87073ec16f459aa11aeea6806733f788bbeec1",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
)]
pub fn gfx950_fp8_attention_rust(
    query: &[u8],
    key: &[u8],
    value: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) {
    // Invalid launch buffers abort the full wave before collective work begins.
    if query.len() < ATTENTION_TOKENS * GEMM_K
        || key.len() < ATTENTION_TOKENS * GEMM_K
        || value.len() < ATTENTION_TOKENS * VALUE_COLUMNS
        || output.len() < ATTENTION_TOKENS * VALUE_COLUMNS
    {
        fe2o3_device::trap();
    }
    let index = thread::index_1d();
    let lane_column = index.get() % 16;
    let lane = WaveLane::<Wave64>::current();
    let Ok(query_matrix) =
        Gfx950Fp8MfmaAMatrix::row_major(query, 0, ATTENTION_TOKENS, GEMM_K, GEMM_K)
    else {
        fe2o3_device::trap();
    };
    let query = query_matrix.load_m16k128(&lane, 0, 0);
    let Ok(key) = Gfx950Fp8MfmaAMatrix::row_major(key, 0, ATTENTION_TOKENS, GEMM_K, GEMM_K)
    else {
        fe2o3_device::trap();
    };
    let key = Gfx950LdsTransposeTile::<Gfx950Fp8E4M3, Gfx950TransposeUninitialized>::current(&lane)
        .stage_k_transposed(&key, 0, 0)
        .publish()
        .read_mfma_fragment();
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp8E4M3>::zero(&lane);
    let scores = Gfx950Matrix::current()
        .multiply_accumulate_fp8(query, key, accumulator)
        .into_values();
    let Ok(value) = StridedReadView2D::from_shared_slice(
        value,
        0,
        ATTENTION_TOKENS,
        VALUE_COLUMNS,
        VALUE_COLUMNS,
    ) else {
        fe2o3_device::trap();
    };
    let subgroup = Gfx950Subgroup::current();
    let math = Math::current();
    let maximum0 = subgroup.reduce_max_f32::<16>(scores[0] * ATTENTION_SCALE);
    let maximum1 = subgroup.reduce_max_f32::<16>(scores[1] * ATTENTION_SCALE);
    let maximum2 = subgroup.reduce_max_f32::<16>(scores[2] * ATTENTION_SCALE);
    let maximum3 = subgroup.reduce_max_f32::<16>(scores[3] * ATTENTION_SCALE);
    let probability0 = math.exp_f32(scores[0] * ATTENTION_SCALE - maximum0);
    let probability1 = math.exp_f32(scores[1] * ATTENTION_SCALE - maximum1);
    let probability2 = math.exp_f32(scores[2] * ATTENTION_SCALE - maximum2);
    let probability3 = math.exp_f32(scores[3] * ATTENTION_SCALE - maximum3);
    let normalized0 = probability0 / subgroup.reduce_sum_f32::<16>(probability0);
    let normalized1 = probability1 / subgroup.reduce_sum_f32::<16>(probability1);
    let normalized2 = probability2 / subgroup.reduce_sum_f32::<16>(probability2);
    let normalized3 = probability3 / subgroup.reduce_sum_f32::<16>(probability3);
    let mut result0 = 0.0;
    let mut result1 = 0.0;
    let mut result2 = 0.0;
    let mut result3 = 0.0;
    let bits0 = value.load_or(0, lane_column, 0);
    let exponent0 = (bits0 >> 3_u8) & 0xf;
    let mantissa0 = bits0 & 0x7;
    let magnitude0 = if exponent0 == 0xf && mantissa0 == 0x7 {
        let nan_source = f32::from(mantissa0 - 7_u8);
        nan_source / nan_source
    } else if exponent0 == 0 {
        f32::from(mantissa0) / 512.0
    } else {
        let scale = if exponent0 < 8 {
            f32::from(1_u8 << exponent0) / 128.0
        } else {
            f32::from(1_u8 << (exponent0 - 7_u8))
        };
        (1.0 + f32::from(mantissa0) / 8.0) * scale
    };
    let value0 = if bits0 & 0x80 == 0 {
        magnitude0
    } else {
        -magnitude0
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 0) * value0;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 0) * value0;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 0) * value0;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 0) * value0;
    let bits1 = value.load_or(1, lane_column, 0);
    let exponent1 = (bits1 >> 3_u8) & 0xf;
    let mantissa1 = bits1 & 0x7;
    let magnitude1 = if exponent1 == 0xf && mantissa1 == 0x7 {
        let nan_source = f32::from(mantissa1 - 7_u8);
        nan_source / nan_source
    } else if exponent1 == 0 {
        f32::from(mantissa1) / 512.0
    } else {
        let scale = if exponent1 < 8 {
            f32::from(1_u8 << exponent1) / 128.0
        } else {
            f32::from(1_u8 << (exponent1 - 7_u8))
        };
        (1.0 + f32::from(mantissa1) / 8.0) * scale
    };
    let value1 = if bits1 & 0x80 == 0 {
        magnitude1
    } else {
        -magnitude1
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 1) * value1;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 1) * value1;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 1) * value1;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 1) * value1;
    let bits2 = value.load_or(2, lane_column, 0);
    let exponent2 = (bits2 >> 3_u8) & 0xf;
    let mantissa2 = bits2 & 0x7;
    let magnitude2 = if exponent2 == 0xf && mantissa2 == 0x7 {
        let nan_source = f32::from(mantissa2 - 7_u8);
        nan_source / nan_source
    } else if exponent2 == 0 {
        f32::from(mantissa2) / 512.0
    } else {
        let scale = if exponent2 < 8 {
            f32::from(1_u8 << exponent2) / 128.0
        } else {
            f32::from(1_u8 << (exponent2 - 7_u8))
        };
        (1.0 + f32::from(mantissa2) / 8.0) * scale
    };
    let value2 = if bits2 & 0x80 == 0 {
        magnitude2
    } else {
        -magnitude2
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 2) * value2;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 2) * value2;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 2) * value2;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 2) * value2;
    let bits3 = value.load_or(3, lane_column, 0);
    let exponent3 = (bits3 >> 3_u8) & 0xf;
    let mantissa3 = bits3 & 0x7;
    let magnitude3 = if exponent3 == 0xf && mantissa3 == 0x7 {
        let nan_source = f32::from(mantissa3 - 7_u8);
        nan_source / nan_source
    } else if exponent3 == 0 {
        f32::from(mantissa3) / 512.0
    } else {
        let scale = if exponent3 < 8 {
            f32::from(1_u8 << exponent3) / 128.0
        } else {
            f32::from(1_u8 << (exponent3 - 7_u8))
        };
        (1.0 + f32::from(mantissa3) / 8.0) * scale
    };
    let value3 = if bits3 & 0x80 == 0 {
        magnitude3
    } else {
        -magnitude3
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 3) * value3;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 3) * value3;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 3) * value3;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 3) * value3;
    let bits4 = value.load_or(4, lane_column, 0);
    let exponent4 = (bits4 >> 3_u8) & 0xf;
    let mantissa4 = bits4 & 0x7;
    let magnitude4 = if exponent4 == 0xf && mantissa4 == 0x7 {
        let nan_source = f32::from(mantissa4 - 7_u8);
        nan_source / nan_source
    } else if exponent4 == 0 {
        f32::from(mantissa4) / 512.0
    } else {
        let scale = if exponent4 < 8 {
            f32::from(1_u8 << exponent4) / 128.0
        } else {
            f32::from(1_u8 << (exponent4 - 7_u8))
        };
        (1.0 + f32::from(mantissa4) / 8.0) * scale
    };
    let value4 = if bits4 & 0x80 == 0 {
        magnitude4
    } else {
        -magnitude4
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 4) * value4;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 4) * value4;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 4) * value4;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 4) * value4;
    let bits5 = value.load_or(5, lane_column, 0);
    let exponent5 = (bits5 >> 3_u8) & 0xf;
    let mantissa5 = bits5 & 0x7;
    let magnitude5 = if exponent5 == 0xf && mantissa5 == 0x7 {
        let nan_source = f32::from(mantissa5 - 7_u8);
        nan_source / nan_source
    } else if exponent5 == 0 {
        f32::from(mantissa5) / 512.0
    } else {
        let scale = if exponent5 < 8 {
            f32::from(1_u8 << exponent5) / 128.0
        } else {
            f32::from(1_u8 << (exponent5 - 7_u8))
        };
        (1.0 + f32::from(mantissa5) / 8.0) * scale
    };
    let value5 = if bits5 & 0x80 == 0 {
        magnitude5
    } else {
        -magnitude5
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 5) * value5;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 5) * value5;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 5) * value5;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 5) * value5;
    let bits6 = value.load_or(6, lane_column, 0);
    let exponent6 = (bits6 >> 3_u8) & 0xf;
    let mantissa6 = bits6 & 0x7;
    let magnitude6 = if exponent6 == 0xf && mantissa6 == 0x7 {
        let nan_source = f32::from(mantissa6 - 7_u8);
        nan_source / nan_source
    } else if exponent6 == 0 {
        f32::from(mantissa6) / 512.0
    } else {
        let scale = if exponent6 < 8 {
            f32::from(1_u8 << exponent6) / 128.0
        } else {
            f32::from(1_u8 << (exponent6 - 7_u8))
        };
        (1.0 + f32::from(mantissa6) / 8.0) * scale
    };
    let value6 = if bits6 & 0x80 == 0 {
        magnitude6
    } else {
        -magnitude6
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 6) * value6;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 6) * value6;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 6) * value6;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 6) * value6;
    let bits7 = value.load_or(7, lane_column, 0);
    let exponent7 = (bits7 >> 3_u8) & 0xf;
    let mantissa7 = bits7 & 0x7;
    let magnitude7 = if exponent7 == 0xf && mantissa7 == 0x7 {
        let nan_source = f32::from(mantissa7 - 7_u8);
        nan_source / nan_source
    } else if exponent7 == 0 {
        f32::from(mantissa7) / 512.0
    } else {
        let scale = if exponent7 < 8 {
            f32::from(1_u8 << exponent7) / 128.0
        } else {
            f32::from(1_u8 << (exponent7 - 7_u8))
        };
        (1.0 + f32::from(mantissa7) / 8.0) * scale
    };
    let value7 = if bits7 & 0x80 == 0 {
        magnitude7
    } else {
        -magnitude7
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 7) * value7;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 7) * value7;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 7) * value7;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 7) * value7;
    let bits8 = value.load_or(8, lane_column, 0);
    let exponent8 = (bits8 >> 3_u8) & 0xf;
    let mantissa8 = bits8 & 0x7;
    let magnitude8 = if exponent8 == 0xf && mantissa8 == 0x7 {
        let nan_source = f32::from(mantissa8 - 7_u8);
        nan_source / nan_source
    } else if exponent8 == 0 {
        f32::from(mantissa8) / 512.0
    } else {
        let scale = if exponent8 < 8 {
            f32::from(1_u8 << exponent8) / 128.0
        } else {
            f32::from(1_u8 << (exponent8 - 7_u8))
        };
        (1.0 + f32::from(mantissa8) / 8.0) * scale
    };
    let value8 = if bits8 & 0x80 == 0 {
        magnitude8
    } else {
        -magnitude8
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 8) * value8;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 8) * value8;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 8) * value8;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 8) * value8;
    let bits9 = value.load_or(9, lane_column, 0);
    let exponent9 = (bits9 >> 3_u8) & 0xf;
    let mantissa9 = bits9 & 0x7;
    let magnitude9 = if exponent9 == 0xf && mantissa9 == 0x7 {
        let nan_source = f32::from(mantissa9 - 7_u8);
        nan_source / nan_source
    } else if exponent9 == 0 {
        f32::from(mantissa9) / 512.0
    } else {
        let scale = if exponent9 < 8 {
            f32::from(1_u8 << exponent9) / 128.0
        } else {
            f32::from(1_u8 << (exponent9 - 7_u8))
        };
        (1.0 + f32::from(mantissa9) / 8.0) * scale
    };
    let value9 = if bits9 & 0x80 == 0 {
        magnitude9
    } else {
        -magnitude9
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 9) * value9;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 9) * value9;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 9) * value9;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 9) * value9;
    let bits10 = value.load_or(10, lane_column, 0);
    let exponent10 = (bits10 >> 3_u8) & 0xf;
    let mantissa10 = bits10 & 0x7;
    let magnitude10 = if exponent10 == 0xf && mantissa10 == 0x7 {
        let nan_source = f32::from(mantissa10 - 7_u8);
        nan_source / nan_source
    } else if exponent10 == 0 {
        f32::from(mantissa10) / 512.0
    } else {
        let scale = if exponent10 < 8 {
            f32::from(1_u8 << exponent10) / 128.0
        } else {
            f32::from(1_u8 << (exponent10 - 7_u8))
        };
        (1.0 + f32::from(mantissa10) / 8.0) * scale
    };
    let value10 = if bits10 & 0x80 == 0 {
        magnitude10
    } else {
        -magnitude10
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 10) * value10;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 10) * value10;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 10) * value10;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 10) * value10;
    let bits11 = value.load_or(11, lane_column, 0);
    let exponent11 = (bits11 >> 3_u8) & 0xf;
    let mantissa11 = bits11 & 0x7;
    let magnitude11 = if exponent11 == 0xf && mantissa11 == 0x7 {
        let nan_source = f32::from(mantissa11 - 7_u8);
        nan_source / nan_source
    } else if exponent11 == 0 {
        f32::from(mantissa11) / 512.0
    } else {
        let scale = if exponent11 < 8 {
            f32::from(1_u8 << exponent11) / 128.0
        } else {
            f32::from(1_u8 << (exponent11 - 7_u8))
        };
        (1.0 + f32::from(mantissa11) / 8.0) * scale
    };
    let value11 = if bits11 & 0x80 == 0 {
        magnitude11
    } else {
        -magnitude11
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 11) * value11;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 11) * value11;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 11) * value11;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 11) * value11;
    let bits12 = value.load_or(12, lane_column, 0);
    let exponent12 = (bits12 >> 3_u8) & 0xf;
    let mantissa12 = bits12 & 0x7;
    let magnitude12 = if exponent12 == 0xf && mantissa12 == 0x7 {
        let nan_source = f32::from(mantissa12 - 7_u8);
        nan_source / nan_source
    } else if exponent12 == 0 {
        f32::from(mantissa12) / 512.0
    } else {
        let scale = if exponent12 < 8 {
            f32::from(1_u8 << exponent12) / 128.0
        } else {
            f32::from(1_u8 << (exponent12 - 7_u8))
        };
        (1.0 + f32::from(mantissa12) / 8.0) * scale
    };
    let value12 = if bits12 & 0x80 == 0 {
        magnitude12
    } else {
        -magnitude12
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 12) * value12;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 12) * value12;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 12) * value12;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 12) * value12;
    let bits13 = value.load_or(13, lane_column, 0);
    let exponent13 = (bits13 >> 3_u8) & 0xf;
    let mantissa13 = bits13 & 0x7;
    let magnitude13 = if exponent13 == 0xf && mantissa13 == 0x7 {
        let nan_source = f32::from(mantissa13 - 7_u8);
        nan_source / nan_source
    } else if exponent13 == 0 {
        f32::from(mantissa13) / 512.0
    } else {
        let scale = if exponent13 < 8 {
            f32::from(1_u8 << exponent13) / 128.0
        } else {
            f32::from(1_u8 << (exponent13 - 7_u8))
        };
        (1.0 + f32::from(mantissa13) / 8.0) * scale
    };
    let value13 = if bits13 & 0x80 == 0 {
        magnitude13
    } else {
        -magnitude13
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 13) * value13;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 13) * value13;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 13) * value13;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 13) * value13;
    let bits14 = value.load_or(14, lane_column, 0);
    let exponent14 = (bits14 >> 3_u8) & 0xf;
    let mantissa14 = bits14 & 0x7;
    let magnitude14 = if exponent14 == 0xf && mantissa14 == 0x7 {
        let nan_source = f32::from(mantissa14 - 7_u8);
        nan_source / nan_source
    } else if exponent14 == 0 {
        f32::from(mantissa14) / 512.0
    } else {
        let scale = if exponent14 < 8 {
            f32::from(1_u8 << exponent14) / 128.0
        } else {
            f32::from(1_u8 << (exponent14 - 7_u8))
        };
        (1.0 + f32::from(mantissa14) / 8.0) * scale
    };
    let value14 = if bits14 & 0x80 == 0 {
        magnitude14
    } else {
        -magnitude14
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 14) * value14;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 14) * value14;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 14) * value14;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 14) * value14;
    let bits15 = value.load_or(15, lane_column, 0);
    let exponent15 = (bits15 >> 3_u8) & 0xf;
    let mantissa15 = bits15 & 0x7;
    let magnitude15 = if exponent15 == 0xf && mantissa15 == 0x7 {
        let nan_source = f32::from(mantissa15 - 7_u8);
        nan_source / nan_source
    } else if exponent15 == 0 {
        f32::from(mantissa15) / 512.0
    } else {
        let scale = if exponent15 < 8 {
            f32::from(1_u8 << exponent15) / 128.0
        } else {
            f32::from(1_u8 << (exponent15 - 7_u8))
        };
        (1.0 + f32::from(mantissa15) / 8.0) * scale
    };
    let value15 = if bits15 & 0x80 == 0 {
        magnitude15
    } else {
        -magnitude15
    };
    result0 += subgroup.broadcast_f32::<16>(normalized0, 15) * value15;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 15) * value15;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 15) * value15;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 15) * value15;
    let Some(output_block) = index.checked_block::<16, 4>() else {
        fe2o3_device::trap();
    };
    if let Some(element) = output.get_block_mut(&output_block, 0) {
        *element = result0;
    }
    if let Some(element) = output.get_block_mut(&output_block, 1) {
        *element = result1;
    }
    if let Some(element) = output.get_block_mut(&output_block, 2) {
        *element = result2;
    }
    if let Some(element) = output.get_block_mut(&output_block, 3) {
        *element = result3;
    }
}

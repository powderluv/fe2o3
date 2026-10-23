//! Safe Rust fixed-shape gfx950 kernels.
//!
//! Read each entrypoint in five phases: validate the complete launch contract,
//! map the global thread to a Wave64-owned batch, build typed matrix views,
//! execute collective matrix or reduction operations, and publish through a
//! statically disjoint output view. Those explicit phases keep the safety
//! argument and the expected gfx950 ISA reviewable together.

#![allow(missing_docs)]

use fe2o3_device::{
    Blocked, DisjointSlice, Gfx950F32AccumulatorFragment, Gfx950Fp4E2M1, Gfx950Fp4MfmaAMatrix,
    Gfx950Fp4MfmaBMatrix, Gfx950Fp8E4M3, Gfx950Fp8MfmaAMatrix, Gfx950Fp8MfmaBMatrix,
    Gfx950LdsTransposeTile, Gfx950Matrix, Gfx950Subgroup, Gfx950TransposeUninitialized, Index1D,
    KernelError, KernelResult, Math, StridedReadView2D, Wave64, WaveLane, kernel, thread,
};

pub use crate::dimensions::*;
const ATTENTION_SCALE: f32 = 0.088_388_35;

fn decode_fp4_e2m1(bits: u8) -> f32 {
    match bits & 0xf {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        8 => -0.0,
        9 => -0.5,
        10 => -1.0,
        11 => -1.5,
        12 => -2.0,
        13 => -3.0,
        14 => -4.0,
        _ => -6.0,
    }
}

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-fp4-gemm"))]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
)]
/// Computes 16 independent 16x16x128 packed-E2M1 GEMM tiles.
pub fn gfx950_fp4_gemm_rust(
    lhs: &[u8],
    rhs: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    // Validate all wave-shared inputs before any lane performs matrix work.
    if lhs.len() < GFX950_BATCHES * GEMM_M * GEMM_K
        || rhs.len() < GFX950_BATCHES * GEMM_K * GEMM_N
        || output.len() < GFX950_BATCHES * GEMM_M * GEMM_N
    {
        return Err(KernelError::InvalidArgument);
    }
    // Four Wave64 tiles per WG256 and four workgroups produce 16 batches.
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane = WaveLane::<Wave64>::current();
    // Typed views encode operand layout and bounds before fragment loads.
    let Ok(lhs_matrix) = Gfx950Fp4MfmaAMatrix::row_major(
        lhs,
        batch.wrapping_mul(GEMM_M * GEMM_K),
        GEMM_M,
        GEMM_K,
        GEMM_K,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let lhs = lhs_matrix.load_m16k128(&lane, 0, 0);
    let Ok(rhs_matrix) = Gfx950Fp4MfmaBMatrix::row_major(
        rhs,
        batch.wrapping_mul(GEMM_K * GEMM_N),
        GEMM_K,
        GEMM_N,
        GEMM_N,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let rhs = rhs_matrix.load_k128n16(&lane, 0, 0);
    // One native gfx950 MFMA covers the fixed K=128 reduction.
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane);
    let values = Gfx950Matrix::current()
        .multiply_accumulate_fp4(lhs, rhs, accumulator)
        .into_values();
    // Each lane owns four accumulator elements; the blocked proof makes stores disjoint.
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
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
)]
/// Computes 16 independent 16x16x128 E4M3 GEMM tiles.
pub fn gfx950_fp8_gemm_rust(
    lhs: &[u8],
    rhs: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    // Validate the launch-wide storage contract before entering MFMA code.
    if lhs.len() < GFX950_BATCHES * GEMM_M * GEMM_K
        || rhs.len() < GFX950_BATCHES * GEMM_K * GEMM_N
        || output.len() < GFX950_BATCHES * GEMM_M * GEMM_N
    {
        return Err(KernelError::InvalidArgument);
    }
    // A global Wave64 index selects one of the 16 independent output tiles.
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane = WaveLane::<Wave64>::current();
    // Format-specific matrix views keep E4M3 layout out of pointer arithmetic.
    let Ok(lhs_matrix) = Gfx950Fp8MfmaAMatrix::row_major(
        lhs,
        batch.wrapping_mul(GEMM_M * GEMM_K),
        GEMM_M,
        GEMM_K,
        GEMM_K,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let lhs = lhs_matrix.load_m16k128(&lane, 0, 0);
    let Ok(rhs_matrix) = Gfx950Fp8MfmaBMatrix::row_major(
        rhs,
        batch.wrapping_mul(GEMM_K * GEMM_N),
        GEMM_K,
        GEMM_N,
        GEMM_N,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let rhs = rhs_matrix.load_k128n16(&lane, 0, 0);
    // Accumulate the unified low-precision MFMA result in FP32.
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp8E4M3>::zero(&lane);
    let values = Gfx950Matrix::current()
        .multiply_accumulate_fp8(lhs, rhs, accumulator)
        .into_values();
    // Convert lane-local accumulator ownership into four proven-disjoint stores.
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

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-mixed-fp4-fp8-gemm"))]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
)]
/// Computes 16 independent 16x16x128 FP4-A x FP8-B GEMM tiles.
/// This new ordinary-source example is not a native or MoE qualification.
pub fn gfx950_mixed_fp4_fp8_gemm_rust(
    lhs: &[u8],
    rhs: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    // Validate the launch-wide storage contract before entering MFMA code.
    if lhs.len() < GFX950_BATCHES * GEMM_M * GEMM_K
        || rhs.len() < GFX950_BATCHES * GEMM_K * GEMM_N
        || output.len() < GFX950_BATCHES * GEMM_M * GEMM_N
    {
        return Err(KernelError::InvalidArgument);
    }
    // A global Wave64 index selects one of the 16 independent output tiles.
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane = WaveLane::<Wave64>::current();
    // The existing typed loaders pack low-nibble E2M1 A and split-K E4M3 B.
    let Ok(lhs_matrix) = Gfx950Fp4MfmaAMatrix::row_major(
        lhs,
        batch.wrapping_mul(GEMM_M * GEMM_K),
        GEMM_M,
        GEMM_K,
        GEMM_K,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let lhs = lhs_matrix.load_m16k128(&lane, 0, 0);
    let Ok(rhs_matrix) = Gfx950Fp8MfmaBMatrix::row_major(
        rhs,
        batch.wrapping_mul(GEMM_K * GEMM_N),
        GEMM_K,
        GEMM_N,
        GEMM_N,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let rhs = rhs_matrix.load_k128n16(&lane, 0, 0);
    // Accumulate the unified low-precision MFMA result in FP32.
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane);
    let values = Gfx950Matrix::current()
        .multiply_accumulate_fp4_fp8(lhs, rhs, accumulator)
        .into_values();
    // Convert lane-local accumulator ownership into four proven-disjoint stores.
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
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
)]
/// Computes batched FP4 QK, stable softmax, and the FP32 PV projection.
pub fn gfx950_fp4_attention_rust(
    query: &[u8],
    key: &[u8],
    value: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) {
    // Invalid launch buffers abort the full wave before collective work begins.
    if query.len() < GFX950_BATCHES * ATTENTION_TOKENS * GEMM_K
        || key.len() < GFX950_BATCHES * ATTENTION_TOKENS * GEMM_K
        || value.len() < GFX950_BATCHES * ATTENTION_TOKENS * VALUE_COLUMNS
        || output.len() < GFX950_BATCHES * ATTENTION_TOKENS * VALUE_COLUMNS
    {
        fe2o3_device::trap();
    }
    // A global wave owns one head; lane modulo 16 selects the PV value column.
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane_column = index.get() % 16;
    let lane = WaveLane::<Wave64>::current();
    // Load Q directly and stage K through a wave-private LDS transpose tile.
    let Ok(query_matrix) = Gfx950Fp4MfmaAMatrix::row_major(
        query,
        0,
        GFX950_BATCHES * ATTENTION_TOKENS,
        GEMM_K,
        GEMM_K,
    ) else {
        fe2o3_device::trap();
    };
    let row_base = batch.wrapping_mul(ATTENTION_TOKENS);
    let query = query_matrix.load_m16k128(&lane, row_base, 0);
    let Ok(key) =
        Gfx950Fp4MfmaAMatrix::row_major(key, 0, GFX950_BATCHES * ATTENTION_TOKENS, GEMM_K, GEMM_K)
    else {
        fe2o3_device::trap();
    };
    let key = Gfx950LdsTransposeTile::<Gfx950Fp4E2M1, Gfx950TransposeUninitialized>::current(&lane)
        .stage_k_transposed(&key, row_base, 0)
        .publish()
        .read_mfma_fragment();
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane);
    let scores = Gfx950Matrix::current()
        .multiply_accumulate_fp4(query, key, accumulator)
        .into_values();
    let Ok(value) = StridedReadView2D::from_shared_slice(
        value,
        batch.wrapping_mul(ATTENTION_TOKENS * VALUE_COLUMNS),
        ATTENTION_TOKENS,
        VALUE_COLUMNS,
        VALUE_COLUMNS,
    ) else {
        fe2o3_device::trap();
    };
    // Subtract the row maximum before exponentiation to stabilize softmax.
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
    // Broadcast each key probability across its Wave16 row for the PV reduction.
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
    // Publish four rows per lane through the same blocked ownership used by GEMM.
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
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
)]
/// Computes batched FP8 QK, stable softmax, and the FP32 PV projection.
pub fn gfx950_fp8_attention_rust(
    query: &[u8],
    key: &[u8],
    value: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) {
    // Invalid launch buffers abort the full wave before collective work begins.
    if query.len() < GFX950_BATCHES * ATTENTION_TOKENS * GEMM_K
        || key.len() < GFX950_BATCHES * ATTENTION_TOKENS * GEMM_K
        || value.len() < GFX950_BATCHES * ATTENTION_TOKENS * VALUE_COLUMNS
        || output.len() < GFX950_BATCHES * ATTENTION_TOKENS * VALUE_COLUMNS
    {
        fe2o3_device::trap();
    }
    // A global wave owns one head; lane modulo 16 selects the PV value column.
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane_column = index.get() % 16;
    let lane = WaveLane::<Wave64>::current();
    // The E4M3 K operand uses a wave-private B8 LDS transpose tile.
    let Ok(query_matrix) = Gfx950Fp8MfmaAMatrix::row_major(
        query,
        0,
        GFX950_BATCHES * ATTENTION_TOKENS,
        GEMM_K,
        GEMM_K,
    ) else {
        fe2o3_device::trap();
    };
    let row_base = batch.wrapping_mul(ATTENTION_TOKENS);
    let query = query_matrix.load_m16k128(&lane, row_base, 0);
    let Ok(key) =
        Gfx950Fp8MfmaAMatrix::row_major(key, 0, GFX950_BATCHES * ATTENTION_TOKENS, GEMM_K, GEMM_K)
    else {
        fe2o3_device::trap();
    };
    let key = Gfx950LdsTransposeTile::<Gfx950Fp8E4M3, Gfx950TransposeUninitialized>::current(&lane)
        .stage_k_transposed(&key, row_base, 0)
        .publish()
        .read_mfma_fragment();
    let accumulator = Gfx950F32AccumulatorFragment::<Gfx950Fp8E4M3>::zero(&lane);
    let scores = Gfx950Matrix::current()
        .multiply_accumulate_fp8(query, key, accumulator)
        .into_values();
    let Ok(value) = StridedReadView2D::from_shared_slice(
        value,
        batch.wrapping_mul(ATTENTION_TOKENS * VALUE_COLUMNS),
        ATTENTION_TOKENS,
        VALUE_COLUMNS,
        VALUE_COLUMNS,
    ) else {
        fe2o3_device::trap();
    };
    // Keep maxima, exponentials, denominators, and final division in FP32.
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
    // Broadcast probabilities within each Wave16 row while accumulating PV.
    let mut result0 = 0.0;
    let mut result1 = 0.0;
    let mut result2 = 0.0;
    let mut result3 = 0.0;
    let bits0 = value.load_or(0, lane_column, 0);
    let value0 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits0);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 0) * value0;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 0) * value0;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 0) * value0;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 0) * value0;
    let bits1 = value.load_or(1, lane_column, 0);
    let value1 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits1);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 1) * value1;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 1) * value1;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 1) * value1;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 1) * value1;
    let bits2 = value.load_or(2, lane_column, 0);
    let value2 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits2);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 2) * value2;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 2) * value2;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 2) * value2;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 2) * value2;
    let bits3 = value.load_or(3, lane_column, 0);
    let value3 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits3);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 3) * value3;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 3) * value3;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 3) * value3;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 3) * value3;
    let bits4 = value.load_or(4, lane_column, 0);
    let value4 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits4);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 4) * value4;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 4) * value4;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 4) * value4;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 4) * value4;
    let bits5 = value.load_or(5, lane_column, 0);
    let value5 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits5);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 5) * value5;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 5) * value5;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 5) * value5;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 5) * value5;
    let bits6 = value.load_or(6, lane_column, 0);
    let value6 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits6);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 6) * value6;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 6) * value6;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 6) * value6;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 6) * value6;
    let bits7 = value.load_or(7, lane_column, 0);
    let value7 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits7);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 7) * value7;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 7) * value7;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 7) * value7;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 7) * value7;
    let bits8 = value.load_or(8, lane_column, 0);
    let value8 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits8);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 8) * value8;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 8) * value8;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 8) * value8;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 8) * value8;
    let bits9 = value.load_or(9, lane_column, 0);
    let value9 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits9);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 9) * value9;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 9) * value9;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 9) * value9;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 9) * value9;
    let bits10 = value.load_or(10, lane_column, 0);
    let value10 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits10);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 10) * value10;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 10) * value10;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 10) * value10;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 10) * value10;
    let bits11 = value.load_or(11, lane_column, 0);
    let value11 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits11);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 11) * value11;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 11) * value11;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 11) * value11;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 11) * value11;
    let bits12 = value.load_or(12, lane_column, 0);
    let value12 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits12);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 12) * value12;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 12) * value12;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 12) * value12;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 12) * value12;
    let bits13 = value.load_or(13, lane_column, 0);
    let value13 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits13);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 13) * value13;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 13) * value13;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 13) * value13;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 13) * value13;
    let bits14 = value.load_or(14, lane_column, 0);
    let value14 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits14);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 14) * value14;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 14) * value14;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 14) * value14;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 14) * value14;
    let bits15 = value.load_or(15, lane_column, 0);
    let value15 = crate::kernel_simt::decode_fp8_e4m3_scalar!(bits15);
    result0 += subgroup.broadcast_f32::<16>(normalized0, 15) * value15;
    result1 += subgroup.broadcast_f32::<16>(normalized1, 15) * value15;
    result2 += subgroup.broadcast_f32::<16>(normalized2, 15) * value15;
    result3 += subgroup.broadcast_f32::<16>(normalized3, 15) * value15;
    // The blocked output mapping proves the four stores owned by each lane do not race.
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

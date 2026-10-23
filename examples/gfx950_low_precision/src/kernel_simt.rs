//! Scalar Rust counterpart of the gfx950 OCP E4M3 GEMM.

#![allow(missing_docs)]

use fe2o3_device::{
    Blocked, DisjointSlice, Index1D, KernelError, KernelResult, StridedReadView2D, kernel, thread,
};

use crate::dimensions::{GEMM_K, GEMM_M, GEMM_N, GFX950_BATCHES};

// Expand the same tested arithmetic directly in barrier-checked tile kernels.
macro_rules! decode_fp8_e4m3_scalar {
    ($bits:expr) => {{
        let bits: u8 = $bits;
        let exponent = (bits >> 3) & 0xf;
        let mantissa = bits & 7;
        // Scalar arithmetic keeps bit-conversion library calls out of the decoder.
        // Every finite result and each power-of-two factor is exact in F32.
        let magnitude = if exponent == 15 && mantissa == 7 {
            // E4M3 defines a NaN class, not a widened F32 literal payload.
            // Keep the input-derived operation in the supported strict-F32 path.
            let zero = mantissa as f32 - 7.0;
            zero / zero
        } else if exponent == 0 {
            (mantissa as f32) * 0.001_953_125
        } else {
            // Compose 2^exponent with exact, branch-free binary factors.
            // Each bit is 0 or 1; these factors are 1 or 2, 4, 16, and 256.
            let scale = 0.007_812_5_f32
                * (1.0 + (exponent & 1) as f32)
                * (1.0 + ((exponent >> 1) & 1) as f32 * 3.0)
                * (1.0 + ((exponent >> 2) & 1) as f32 * 15.0)
                * (1.0 + ((exponent >> 3) & 1) as f32 * 255.0);
            (1.0 + (mantissa as f32) * 0.125) * scale
        };
        if bits & 0x80 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }};
}
pub(crate) use decode_fp8_e4m3_scalar;

/// Widens the actual OCP E4M3 byte, including signed zero, subnormals, and NaN.
///
/// The normal encodings fit exactly in F32. This is deliberately separate from
/// gfx942's E4M3-FNUZ storage format, whose bias and exceptional values differ.
#[inline(always)]
pub fn decode_fp8_e4m3_simt(bits: u8) -> f32 {
    decode_fp8_e4m3_scalar!(bits)
}

/// Computes the same 16 independent 16x16x128 E4M3 GEMMs with scalar F32 MACs.
///
/// Each lane writes the same four disjoint outputs as the tile implementation,
/// preserving its three-slice ABI, WG256/grid4 geometry, and complete output.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-simt-fp8-gemm"))]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(4, 128))
)]
pub fn gfx950_fp8_gemm_simt(
    lhs: &[u8],
    rhs: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    if lhs.len() < GFX950_BATCHES * GEMM_M * GEMM_K
        || rhs.len() < GFX950_BATCHES * GEMM_K * GEMM_N
        || output.len() < GFX950_BATCHES * GEMM_M * GEMM_N
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane = index.get() % 64;
    let column = lane % 16;
    let Ok(lhs) = StridedReadView2D::from_shared_slice(
        lhs,
        batch.wrapping_mul(GEMM_M * GEMM_K),
        GEMM_M,
        GEMM_K,
        GEMM_K,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(rhs) = StridedReadView2D::from_shared_slice(
        rhs,
        batch.wrapping_mul(GEMM_K * GEMM_N),
        GEMM_K,
        GEMM_N,
        GEMM_N,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Some(output_block) = index.checked_block::<16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    let mut component = 0_usize;
    while component < 4 {
        // Blocked<Index1D,16,4> maps this lane/component to this row and column.
        let row = (lane / 16) * 4 + component;
        let mut accumulator = 0.0_f32;
        let mut depth = 0_usize;
        while depth < GEMM_K {
            let a = decode_fp8_e4m3_simt(lhs.load_or(row, depth, 0));
            let b = decode_fp8_e4m3_simt(rhs.load_or(depth, column, 0));
            accumulator += a * b;
            depth += 1;
        }
        if component == 0 {
            if let Some(element) = output.get_block_mut(&output_block, 0) {
                *element = accumulator;
            }
        } else if component == 1 {
            if let Some(element) = output.get_block_mut(&output_block, 1) {
                *element = accumulator;
            }
        } else if component == 2 {
            if let Some(element) = output.get_block_mut(&output_block, 2) {
                *element = accumulator;
            }
        } else if let Some(element) = output.get_block_mut(&output_block, 3) {
            *element = accumulator;
        }
        component += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::decode_fp8_e4m3_simt;
    use crate::reference::decode_fp8_e4m3;

    #[test]
    fn scalar_decoder_matches_independent_ocp_reference_for_every_byte() {
        for bits in 0_u8..=255 {
            let actual = decode_fp8_e4m3_simt(bits);
            let expected = decode_fp8_e4m3(bits);
            if expected.is_nan() {
                assert!(actual.is_nan(), "encoding {bits:#04x}");
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "encoding {bits:#04x}");
            }
        }
    }
}

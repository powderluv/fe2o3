//! Scalar Rust counterpart of the gfx950 OCP E2M1 GEMM.

#![allow(missing_docs)]

use fe2o3_device::{
    Blocked, DisjointSlice, Index1D, KernelError, KernelResult, StridedReadView2D, kernel, thread,
};

use crate::dimensions::{GEMM_K, GEMM_M, GEMM_N, GFX950_BATCHES};

/// Widens the low E2M1 nibble of each storage byte, preserving signed zero.
/// The high nibble is ignored, as in the structured matrix fragment load.
pub fn decode_fp4_e2m1_simt(bits: u8) -> f32 {
    let magnitude = bits & 7;
    let value = if magnitude < 2 {
        (magnitude as f32) * 0.5
    } else {
        // Masking bounds the exponent to 126..=129; this addition never wraps.
        f32::from_bits(
            ((magnitude as u32) >> 1).wrapping_add(126) << 23 | (((magnitude as u32) & 1) << 22),
        )
    };
    if bits & 8 != 0 { -value } else { value }
}

/// Computes the same 16 independent 16x16x128 E2M1 GEMMs with scalar F32 MACs.
///
/// Each lane writes the same four disjoint outputs as the tile implementation,
/// preserving its three-slice ABI, WG256/grid4 geometry, and complete output.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-simt-fp4-gemm"))]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(4, 128))
)]
pub fn gfx950_fp4_gemm_simt(
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
            let a = decode_fp4_e2m1_simt(lhs.load_or(row, depth, 0));
            let b = decode_fp4_e2m1_simt(rhs.load_or(depth, column, 0));
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
    use super::decode_fp4_e2m1_simt;
    use crate::reference::decode_fp4_e2m1;

    #[test]
    fn scalar_decoder_matches_independent_ocp_reference_for_every_byte() {
        for bits in 0_u8..=255 {
            let actual = decode_fp4_e2m1_simt(bits);
            let expected = decode_fp4_e2m1(bits);
            assert_eq!(actual.to_bits(), expected.to_bits(), "encoding {bits:#04x}");
        }
    }
}

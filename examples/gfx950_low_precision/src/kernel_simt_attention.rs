//! Ordinary scalar E4M3 attention; no matrix or subgroup instructions.
#![allow(missing_docs)]

use crate::dimensions::{ATTENTION_TOKENS, GEMM_K, GFX950_BATCHES, VALUE_COLUMNS};
use crate::kernel_simt::decode_fp8_e4m3_simt;
use fe2o3_device::{
    Blocked, DisjointSlice, Index1D, KernelError, KernelResult, Math, StridedReadView2D, kernel,
    thread,
};

/// Same four-slice ABI, WG256/grid4, and per-lane four-output ownership as tile.
/// Scores use increasing-depth F32 MACs; max, exp, denominator and PV remain F32.
/// Two score passes avoid private score arrays and cross-lane communication.
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(4, 16, 128, 16, 128))
)]
pub fn gfx950_fp8_attention_simt(
    query: &[u8],
    key: &[u8],
    value: &[u8],
    mut output: DisjointSlice<f32, Blocked<Index1D, 16, 4>>,
) -> KernelResult {
    if query.len() < GFX950_BATCHES * ATTENTION_TOKENS * GEMM_K
        || key.len() < GFX950_BATCHES * ATTENTION_TOKENS * GEMM_K
        || value.len() < GFX950_BATCHES * ATTENTION_TOKENS * VALUE_COLUMNS
        || output.len() < GFX950_BATCHES * ATTENTION_TOKENS * VALUE_COLUMNS
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let lane = index.get() % 64;
    let column = lane % 16;
    let Ok(query) = StridedReadView2D::from_shared_slice(
        query,
        batch.wrapping_mul(ATTENTION_TOKENS * GEMM_K),
        ATTENTION_TOKENS,
        GEMM_K,
        GEMM_K,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(key) = StridedReadView2D::from_shared_slice(
        key,
        batch.wrapping_mul(ATTENTION_TOKENS * GEMM_K),
        ATTENTION_TOKENS,
        GEMM_K,
        GEMM_K,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(value) = StridedReadView2D::from_shared_slice(
        value,
        batch.wrapping_mul(ATTENTION_TOKENS * VALUE_COLUMNS),
        ATTENTION_TOKENS,
        VALUE_COLUMNS,
        VALUE_COLUMNS,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Some(output_block) = index.checked_block::<16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    let math = Math::current();
    // The identical rounded 1/sqrt(128) scale is used by the tile source.
    let scale = 0.088_388_35_f32;
    let mut component = 0_usize;
    while component < 4 {
        let row = (lane / 16) * 4 + component;
        let mut maximum = f32::NEG_INFINITY;
        let mut token = 0_usize;
        while token < ATTENTION_TOKENS {
            let mut score = 0.0_f32;
            let mut depth = 0_usize;
            while depth < GEMM_K {
                let q = decode_fp8_e4m3_simt(query.load_or(row, depth, 0));
                let k = decode_fp8_e4m3_simt(key.load_or(token, depth, 0));
                score += q * k;
                depth += 1;
            }
            let scaled = score * scale;
            if scaled > maximum {
                maximum = scaled;
            }
            token += 1;
        }
        let mut denominator = 0.0_f32;
        let mut numerator = 0.0_f32;
        let mut value_token = 0_usize;
        while value_token < ATTENTION_TOKENS {
            let mut score = 0.0_f32;
            let mut depth = 0_usize;
            while depth < GEMM_K {
                let q = decode_fp8_e4m3_simt(query.load_or(row, depth, 0));
                let k = decode_fp8_e4m3_simt(key.load_or(value_token, depth, 0));
                score += q * k;
                depth += 1;
            }
            let probability = math.exp_f32(score * scale - maximum);
            denominator += probability;
            numerator += probability * decode_fp8_e4m3_simt(value.load_or(value_token, column, 0));
            value_token += 1;
        }
        let result = numerator / denominator;
        if component == 0 {
            if let Some(element) = output.get_block_mut(&output_block, 0) {
                *element = result;
            }
        } else if component == 1 {
            if let Some(element) = output.get_block_mut(&output_block, 1) {
                *element = result;
            }
        } else if component == 2 {
            if let Some(element) = output.get_block_mut(&output_block, 2) {
                *element = result;
            }
        } else if let Some(element) = output.get_block_mut(&output_block, 3) {
            *element = result;
        }
        component += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn blocked_attention_mapping_owns_every_output_once() {
        let mut writes = [0_u8; 16 * 16 * 16];
        for index in 0..1024 {
            let batch = index / 64;
            let lane = index % 64;
            for component in 0..4 {
                let row = (lane / 16) * 4 + component;
                let column = lane % 16;
                let offset = batch * 256 + row * 16 + column;
                assert_eq!(offset, (index / 16) * 64 + component * 16 + index % 16);
                writes[offset] += 1;
            }
        }
        assert!(writes.iter().all(|count| *count == 1));
    }

    #[test]
    fn scalar_attention_keeps_real_bytes_and_stable_two_pass_math() {
        let source = include_str!("kernel_simt_attention.rs");
        let body = source.split("#[cfg(test)]").next().unwrap();
        assert_eq!(body.matches("decode_fp8_e4m3_simt(").count(), 5);
        assert_eq!(body.matches("while depth < GEMM_K").count(), 2);
        assert!(body.contains("score * scale - maximum"));
        assert!(body.contains("numerator / denominator"));
        assert_eq!(body.matches("get_block_mut").count(), 4);
        assert!(!body.contains("unsafe"));
        assert!(!body.contains("Gfx950Matrix"));
    }
}

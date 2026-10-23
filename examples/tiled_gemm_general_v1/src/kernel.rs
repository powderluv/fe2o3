//! Safe Rust qualification kernel for dynamic strided matrix multiplication.
//!
//! Its review order is deliberate: validate dynamic extents, map a wave to a
//! tile, build typed matrices, run a uniform double-buffered K pipeline, then
//! perform edge-safe stores through a tiled disjoint capability.

#![allow(missing_docs)] // Generated typed-kernel modules lack rustdoc in V1.

use fe2o3_device::{
    Bf16MfmaAFragment, Bf16MfmaAMatrix, Bf16MfmaBFragment, Bf16MfmaBMatrix, DisjointSlice,
    F32AccumulatorFragment, Index1D, KernelError, KernelResult, Matrix, Tiled2D, Wave64, WaveLane,
    WorkgroupLdsScope, WorkgroupPipeline, kernel, thread,
};

/// Exact workgroup dimensions required by the wave64 matrix profile.
pub const GENERAL_TILED_GEMM_WORKGROUP_V1: [u32; 3] = [64, 1, 1];

fn accessed_extent(rows: u32, columns: u32, stride: u32) -> u64 {
    if rows == 0 || columns == 0 {
        return 0;
    }
    ((rows - 1) as u64) * (stride as u64) + (columns as u64)
}

/// Computes `C = alpha * A * B + beta * C` for dynamic row-major matrices.
///
/// Each workgroup is one wave64 and owns one 16x16 output tile. All lanes call
/// the matrix operation uniformly; edge loads contribute BF16 zero, while the
/// checked tiled output witness suppresses stores outside logical M and N.
#[kernel(
    typed,
    launch(
        required = [64, 1, 1],
        max = [64, 1, 1],
        static_shared_memory_bytes = 2048
    ),
    control_flow(loop_bounds(4294967295))
)]
#[allow(clippy::too_many_arguments)]
pub fn tiled_gemm_general_v1(
    a: &[u16],
    b: &[u16],
    mut c: DisjointSlice<f32, Tiled2D<Index1D, 64, 16, 16, 4>>,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    alpha: f32,
    beta: f32,
) -> KernelResult {
    // Reject incompatible shapes and strides before every lane enters LDS barriers.
    let invalid_stride = (m != 0 && k != 0 && lda < k)
        || (k != 0 && n != 0 && ldb < n)
        || (m != 0 && n != 0 && ldc < n);
    let a_extent = accessed_extent(m, k, lda);
    let b_extent = accessed_extent(k, n, ldb);
    let c_extent = accessed_extent(m, n, ldc);
    if invalid_stride
        || (a.len() as u64) < a_extent
        || (b.len() as u64) < b_extent
        || (c.len() as u64) < c_extent
    {
        return Err(KernelError::InvalidArgument);
    }

    // Grid coordinates assign one Wave64 to one 16x16 output tile.
    let thread_index = thread::index_1d();
    let raw_index = thread_index.get();
    let tiles_per_row = (n as usize + 15) / 16;
    if tiles_per_row == 0 {
        return Ok(());
    }
    let tile = raw_index / 64;
    let tile_row = tile / tiles_per_row;
    let tile_column = tile % tiles_per_row;

    if tile_row > usize::MAX / 16 {
        return Err(KernelError::InvalidArgument);
    }
    let tile_row_base = tile_row * 16;
    if tile_column > usize::MAX / 16 {
        return Err(KernelError::InvalidArgument);
    }
    let tile_column_base = tile_column * 16;

    // Typed views centralize layout and padding; the output witness captures ownership.
    let output_tile = thread_index.checked_tiled_2d::<64, 16, 16, 4>();
    let Ok(a_matrix) = Bf16MfmaAMatrix::row_major(a, 0, m as usize, k as usize, lda as usize)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(b_matrix) = Bf16MfmaBMatrix::row_major(b, 0, k as usize, n as usize, ldb as usize)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let wave_lane = WaveLane::<Wave64>::current();
    let matrix = Matrix::current();
    let mut accumulator = F32AccumulatorFragment::zero(&wave_lane);
    // Two LDS stages overlap fragment production for n+1 with MFMA consumption for n.
    let phase_count = (k as usize + 15) / 16;
    let lane = raw_index % 64;
    let mut pipeline_scope = WorkgroupLdsScope::current();
    let mut lhs_pipeline =
        WorkgroupPipeline::<Bf16MfmaAFragment<'_>, 2, 64, 1>::current(&mut pipeline_scope);
    let mut rhs_pipeline =
        WorkgroupPipeline::<Bf16MfmaBFragment<'_>, 2, 64, 1>::current(&mut pipeline_scope);

    let lhs = a_matrix.load_m16k16(&wave_lane, tile_row_base, 0);
    let rhs = b_matrix.load_k16n16(&wave_lane, 0, tile_column_base);
    lhs_pipeline.stage(0);
    lhs_pipeline.write(0, lane, lhs);
    lhs_pipeline.commit(0);
    rhs_pipeline.stage(0);
    rhs_pipeline.write(0, lane, rhs);
    rhs_pipeline.commit(0);

    let mut phase_index = 0_usize;
    while phase_index < phase_count {
        let future_epoch = phase_index + 1;
        let next_phase = future_epoch * 16;
        let next_lhs = a_matrix.load_m16k16(&wave_lane, tile_row_base, next_phase);
        let next_rhs = b_matrix.load_k16n16(&wave_lane, next_phase, tile_column_base);

        lhs_pipeline.stage(future_epoch);
        lhs_pipeline.write(future_epoch, lane, next_lhs);
        lhs_pipeline.commit(future_epoch);
        rhs_pipeline.stage(future_epoch);
        rhs_pipeline.write(future_epoch, lane, next_rhs);
        rhs_pipeline.commit(future_epoch);

        lhs_pipeline.wait(phase_index);
        lhs_pipeline.consume(phase_index);
        let lhs = lhs_pipeline.read(phase_index, lane);
        rhs_pipeline.wait(phase_index);
        rhs_pipeline.consume(phase_index);
        let rhs = rhs_pipeline.read(phase_index, lane);
        accumulator = matrix.multiply_accumulate(lhs, rhs, accumulator);
        lhs_pipeline.release(phase_index);
        rhs_pipeline.release(phase_index);
        phase_index += 1;
    }
    lhs_pipeline.wait(phase_count);
    lhs_pipeline.discard(phase_count);
    lhs_pipeline.release(phase_count);
    rhs_pipeline.wait(phase_count);
    rhs_pipeline.discard(phase_count);
    rhs_pipeline.release(phase_count);

    // The tiled capability clips M/N edges while proving unique ownership of valid C.
    let values = accumulator.into_values();
    if let Some(output_tile) = output_tile {
        if let Some(output) =
            c.get_tiled_2d_mut(&output_tile, 0, m as usize, n as usize, ldc as usize)
        {
            *output = alpha * values[0] + beta * *output;
        }
        if let Some(output) =
            c.get_tiled_2d_mut(&output_tile, 1, m as usize, n as usize, ldc as usize)
        {
            *output = alpha * values[1] + beta * *output;
        }
        if let Some(output) =
            c.get_tiled_2d_mut(&output_tile, 2, m as usize, n as usize, ldc as usize)
        {
            *output = alpha * values[2] + beta * *output;
        }
        if let Some(output) =
            c.get_tiled_2d_mut(&output_tile, 3, m as usize, n as usize, ldc as usize)
        {
            *output = alpha * values[3] + beta * *output;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_extents_include_only_accessed_elements() {
        assert_eq!(accessed_extent(3, 2, 5), 12);
        assert_eq!(accessed_extent(0, 2, 5), 0);
        assert_eq!(accessed_extent(3, 0, 5), 0);
        assert_eq!(
            accessed_extent(u32::MAX, u32::MAX, u32::MAX),
            u64::from(u32::MAX) * u64::from(u32::MAX)
        );
    }
}

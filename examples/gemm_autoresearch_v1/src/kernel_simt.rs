//! SIMT source for the same dynamic strided BF16 GEMM contract as the tile body.

#![allow(missing_docs)] // Generated typed-kernel modules lack rustdoc in V1.

use fe2o3_device::{
    Bf16, DisjointSlice, Index1D, KernelError, KernelResult, StridedReadView2D, kernel, thread,
};

/// Workgroup dimensions for the scalar per-output implementation.
pub const AUTORESEARCH_SIMT_GEMM_WORKGROUP_V1: [u32; 3] = [64, 1, 1];

fn accessed_extent(rows: u32, columns: u32, stride: u32) -> u64 {
    if rows == 0 || columns == 0 {
        return 0;
    }
    ((rows - 1) as u64) * (stride as u64) + (columns as u64)
}

/// Computes `C = alpha * A * B + beta * C` with one invocation per physical C slot.
///
/// Launch over the accessed C extent, rounded up to workgroup size with checked
/// host arithmetic. Padding and trailing invocations leave storage unchanged.
#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1]),
    control_flow(loop_bounds(4294967295))
)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_autoresearch_simt_v1(
    a: &[u16],
    b: &[u16],
    mut c: DisjointSlice<f32, Index1D>,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    alpha: f32,
    beta: f32,
) -> KernelResult {
    // Validate every operand even when a different dimension makes C empty.
    let invalid_stride = (m != 0 && k != 0 && lda < k)
        || (k != 0 && n != 0 && ldb < n)
        || (m != 0 && n != 0 && ldc < n);
    if invalid_stride
        || (a.len() as u64) < accessed_extent(m, k, lda)
        || (b.len() as u64) < accessed_extent(k, n, ldb)
        || (c.len() as u64) < accessed_extent(m, n, ldc)
    {
        return Err(KernelError::InvalidArgument);
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if ldc == 0 {
        return Err(KernelError::InvalidArgument);
    }

    // Nonempty, validated C has ldc >= n > 0, so division cannot be by zero.
    let thread_index = thread::index_1d();
    let physical = thread_index.get();
    let row = physical / ldc as usize;
    let column = physical % ldc as usize;
    if row >= m as usize || column >= n as usize {
        return Ok(());
    }

    let Ok(lhs) = StridedReadView2D::from_shared_slice(a, 0, m as usize, k as usize, lda as usize)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(rhs) = StridedReadView2D::from_shared_slice(b, 0, k as usize, n as usize, ldb as usize)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let mut accumulator = 0.0_f32;
    let mut depth = 0_usize;
    while depth < k as usize {
        let lhs_value = Bf16::from_bits(lhs.load_or(row, depth, 0)).to_f32();
        let rhs_value = Bf16::from_bits(rhs.load_or(depth, column, 0)).to_f32();
        accumulator += lhs_value * rhs_value;
        depth += 1;
    }
    if let Some(output) = c.get_mut(thread_index) {
        *output = alpha * accumulator + beta * *output;
    }
    Ok(())
}

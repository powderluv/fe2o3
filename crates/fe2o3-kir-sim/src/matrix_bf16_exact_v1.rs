//! Closed numerical subdomain, not general BF16 or hardware authority.
use fe2o3_kernel_ir::{
    Convergence, MatrixMultiplyProfile, MatrixOperation, MatrixOperationKind, SynchronizationScope,
    TensorLayoutContractV1,
};

/// Numerical semantics for the existing BF16 matrix interpreter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatrixBf16NumericsV1 {
    /// Original closed integer domain; all legacy entries retain this default.
    #[default]
    ExactV1,
    /// Ordinary observation only: finite BF16 inputs widen exactly to F32, then
    /// F32 fused multiply-add rounds to nearest, ties even in ascending k=0..15
    /// order for each source matrix operation. Gradual underflow and signed zero
    /// are retained. Nonfinite inputs and intermediate results are refused.
    /// This is not native MFMA reduction/rounding equivalence or proof authority.
    FiniteSequentialFmaV1,
}
impl MatrixBf16NumericsV1 {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ExactV1 => "exact-v1",
            Self::FiniteSequentialFmaV1 => "finite-sequential-fma-v1",
        }
    }
    pub(crate) fn accepts_bf16(self, bits: u16) -> bool {
        match self {
            Self::ExactV1 => bf16_input(bits),
            Self::FiniteSequentialFmaV1 => bits & 0x7f80 != 0x7f80,
        }
    }
    pub(crate) fn accepts_accumulator(self, bits: u32) -> bool {
        match self {
            Self::ExactV1 => f32_accumulator(bits),
            Self::FiniteSequentialFmaV1 => finite_f32(bits),
        }
    }
}
pub(crate) const fn finite_f32(bits: u32) -> bool {
    bits & 0x7f80_0000 != 0x7f80_0000
}

/// Positional operand only; no user values are retained in a domain refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MatrixInputRoleV1 {
    A = 0,
    B = 1,
    Accumulator = 2,
}

pub(crate) fn supported(matrix: &MatrixOperation) -> bool {
    if !matches!(&matrix.kind, MatrixOperationKind::MultiplyAccumulate { profile, .. }
        if *profile == MatrixMultiplyProfile::bf16_f32_m16n16k16_wave64())
        || matrix.active_lanes != 64
        || matrix.convergence != Convergence::uniform(SynchronizationScope::Subgroup)
    {
        return false;
    }
    let Some(layout) = matrix.tensor_layout else {
        return false;
    };
    for a_xor in [false, true] {
        for b_xor in [false, true] {
            for zero_tail in [false, true] {
                let mut exact = TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64();
                if a_xor {
                    exact = exact.with_a_lds_xor4();
                }
                if b_xor {
                    exact = exact.with_b_lds_xor4();
                }
                if zero_tail {
                    exact = exact.with_zero_filled_predicate_inputs();
                }
                if layout == exact {
                    return true;
                }
            }
        }
    }
    false
}

/// +0 or a signed integral normal F32 within the supplied positive bound.
/// Bit-only classification; no rounding, host float, or silent coercion.
fn integral_f32(bits: u32, maximum_bits: u32) -> bool {
    if bits == 0 {
        return true;
    }
    let magnitude = bits & 0x7fff_ffff;
    if magnitude == 0 || magnitude > maximum_bits {
        return false;
    }
    let exponent = (magnitude >> 23) & 255;
    if !(127..=150).contains(&exponent) {
        return false;
    }
    let discarded = 150 - exponent;
    let fraction_mask = (1_u32 << discarded) - 1;
    magnitude & fraction_mask == 0
}

pub(crate) fn bf16_input(bits: u16) -> bool {
    integral_f32(u32::from(bits) << 16, 0x4180_0000) // |x| <= 16
}
pub(crate) fn f32_accumulator(bits: u32) -> bool {
    integral_f32(bits, 0x4980_0000) // |x| <= 2^20
}

#[cfg(test)]
#[path = "matrix_bf16_exact_v1_tests.rs"]
mod tests;

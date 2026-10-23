use super::*;
use fe2o3_kernel_ir::{TensorLdsSwizzleV1, TensorTailMaskV1, ValueId};

fn matrix() -> MatrixOperation {
    MatrixOperation::multiply_accumulate([ValueId(0); 4], [ValueId(1); 4], [ValueId(2); 4])
        .with_declared_tensor_layout(
            TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64(),
        )
}

#[test]
fn exact_layout_roster_is_closed_and_old_layoutless_operation_stays_refused() {
    let base = matrix();
    assert!(supported(&base));
    for a in [false, true] {
        for b in [false, true] {
            for tail in [false, true] {
                let mut value = base.clone();
                let mut layout = value.tensor_layout.unwrap();
                if a {
                    layout = layout.with_a_lds_xor4();
                }
                if b {
                    layout = layout.with_b_lds_xor4();
                }
                if tail {
                    layout = layout.with_zero_filled_predicate_inputs();
                }
                value.tensor_layout = Some(layout);
                assert!(supported(&value));
            }
        }
    }
    let mut missing = base.clone();
    missing.tensor_layout = None;
    assert!(!supported(&missing));
    for tail in [
        TensorTailMaskV1::Missing,
        TensorTailMaskV1::PredicateMask,
        TensorTailMaskV1::Unsupported(0),
    ] {
        let mut value = base.clone();
        value.tensor_layout.as_mut().unwrap().tail_mask = tail;
        assert!(!supported(&value));
    }
    let mut wrong = base.clone();
    wrong
        .tensor_layout
        .as_mut()
        .unwrap()
        .accumulator
        .lds_swizzle = TensorLdsSwizzleV1::Xor4;
    assert!(!supported(&wrong));
    wrong = base.clone();
    wrong.tensor_layout.as_mut().unwrap().a.shape[0] = 8;
    assert!(!supported(&wrong));
    wrong = base;
    wrong.active_lanes = 63;
    assert!(!supported(&wrong));
}

#[test]
fn domain_has_exact_integer_edges_and_no_nonfinite_fractional_or_signed_zero_inputs() {
    for bits in [0, 0x3f80, 0xbf80, 0x4180, 0xc180] {
        assert!(bf16_input(bits));
    }
    for bits in [
        0x8000, 1, 0x3f00, 0x4181, 0xc181, 0x7f80, 0xff80, 0x7fc0, 0x7f81,
    ] {
        assert!(!bf16_input(bits), "{bits:04x}");
    }
    for bits in [0, 0x3f80_0000, 0xbf80_0000, 0x4980_0000, 0xc980_0000] {
        assert!(f32_accumulator(bits));
    }
    for bits in [
        0x8000_0000,
        1,
        0x3f00_0000,
        0x4980_0001,
        0xc980_0001,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f80_0001,
    ] {
        assert!(!f32_accumulator(bits), "{bits:08x}");
    }
}

#[test]
fn finite_observation_domain_is_explicit_exhaustive_and_default_stays_exact() {
    let finite = MatrixBf16NumericsV1::FiniteSequentialFmaV1;
    assert_eq!(
        MatrixBf16NumericsV1::default(),
        MatrixBf16NumericsV1::ExactV1
    );
    assert_eq!(
        (0..=u16::MAX)
            .filter(|bits| finite.accepts_bf16(*bits))
            .count(),
        65_280
    );
    for bits in [0, 0x8000, 1, 0x8001, 0x3f81, 0x7f7f, 0xff7f] {
        assert!(finite.accepts_bf16(bits));
    }
    for bits in [0x7f80, 0xff80, 0x7fc0, 0x7f81, 0xffff] {
        assert!(!finite.accepts_bf16(bits));
    }
    for bits in [0, 0x8000_0000, 1, 0x8000_0001, 0x7f7f_ffff] {
        assert!(finite.accepts_accumulator(bits));
    }
    for bits in [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0x7fc0_0000] {
        assert!(!finite.accepts_accumulator(bits));
    }
}

#[test]
fn bf16_domain_exhaustion_matches_independent_integer_encoding_roster() {
    // Exactly33 encodings: -16..-1, +0, +1..+16.
    let mut accepted = Vec::new();
    for bits in 0..=u16::MAX {
        if bf16_input(bits) {
            accepted.push(bits);
        }
    }
    let mut expected = vec![0];
    for integer in 1_u16..=16 {
        let msb = 15 - integer.leading_zeros();
        let exponent = 127 + msb;
        let fraction = (integer - (1 << msb)) << (7 - msb);
        let bits = ((exponent as u16) << 7) | fraction;
        expected.extend([bits, bits | 0x8000]);
    }
    expected.sort_unstable();
    assert_eq!(accepted, expected);
}

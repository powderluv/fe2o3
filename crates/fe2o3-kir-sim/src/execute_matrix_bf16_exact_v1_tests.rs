use super::*;

#[test]
fn fixed_work_formula_is_exact_and_overflow_checked() {
    assert_eq!(resolution_work(64), Some(26_048));
    assert_eq!(resolution_work(1024), Some(89_408));
    assert_eq!(resolution_work(64).unwrap() + 64 * ARRIVAL_WORK, 27_584);
    assert_eq!(resolution_work(usize::MAX), None);
}

#[test]
fn finite_numerical_checks_are_additively_prepaid_without_overflow() {
    use crate::MatrixBf16NumericsV1::{ExactV1, FiniteSequentialFmaV1};
    assert_eq!(
        resolution_work_with_numerics(64, ExactV1),
        resolution_work(64)
    );
    assert_eq!(
        resolution_work_with_numerics(64, FiniteSequentialFmaV1),
        Some(26_048 + 4865)
    );
    assert_eq!(
        resolution_work_with_numerics(usize::MAX, FiniteSequentialFmaV1),
        None
    );
}

#[test]
fn compact_pending_matrix_state_preserves_shared_value_layouts() {
    assert_eq!(size_of::<Input>(), 32);
    assert_eq!(
        size_of::<Scratch>(),
        64 * size_of::<usize>() + 64 * 4 * size_of::<u32>()
    );
    assert!(size_of::<RuntimeValue>() <= 160);
    assert_eq!(size_of::<SimulationDebugValueV1>(), 176);
    assert_eq!(size_of::<SimulationDebugBindingV1>(), 192);
    // Legacy alternatives remain the largest inline payloads. No boxed wave.
    let legacy_ceiling = size_of::<(u8, PointerValue, [ScalarBitsV1; 4])>().max(size_of::<(
        u8,
        PointerValue,
        SliceValue,
        [u64; 6],
    )>());
    assert!(size_of::<CollectiveInput>() <= legacy_ceiling);
    assert!(
        resident_bytes().unwrap() >= 2 * size_of::<Scratch>() + 2 * size_of::<[RuntimeValue; 4]>()
    );
}

#![forbid(unsafe_code)]

extern crate std;

use core::{cell::RefCell, ops::Index};
use std::{panic::AssertUnwindSafe, vec, vec::Vec};

#[derive(Clone, Copy)]
struct TestIndex(usize);

impl TestIndex {
    fn get(&self) -> usize {
        self.0
    }
}

mod cpu_thread {
    pub(super) fn index_1d(linear: usize) -> super::TestIndex {
        super::TestIndex(linear)
    }
}

struct TrackedInput<'a> {
    values: &'a [f32],
    reads: RefCell<Vec<usize>>,
}

impl<'a> TrackedInput<'a> {
    fn new(values: &'a [f32]) -> Self {
        Self {
            values,
            reads: RefCell::new(Vec::new()),
        }
    }
}

impl Index<usize> for TrackedInput<'_> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.reads.borrow_mut().push(index);
        &self.values[index]
    }
}

struct TestOutput<'a> {
    values: &'a mut [f32],
    attempts: Vec<usize>,
}

impl<'a> TestOutput<'a> {
    fn new(values: &'a mut [f32]) -> Self {
        Self {
            values,
            attempts: Vec::new(),
        }
    }

    fn get_mut(&mut self, index: TestIndex) -> Option<&mut f32> {
        self.attempts.push(index.0);
        self.values.get_mut(index.0)
    }
}

fn invoke(linear: usize, a: &TrackedInput<'_>, b: &TrackedInput<'_>, output: &mut TestOutput<'_>) {
    vecadd_kernel_body!(cpu_thread, (linear), production_f32_add, a, b, output);
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_reads(a: &TrackedInput<'_>, b: &TrackedInput<'_>, expected: &[usize]) {
    assert_eq!(a.reads.borrow().as_slice(), expected);
    assert_eq!(b.reads.borrow().as_slice(), expected);
}

#[test]
fn shared_body_nonuniform_finite_inputs_and_rounding() {
    let a = TrackedInput::new(&[
        1.25,
        -3.5,
        8.0,
        -0.75,
        16_777_216.0,
        -16_777_216.0,
        0.125,
        -100.5,
    ]);
    let b = TrackedInput::new(&[2.5, 1.25, -8.0, -0.125, 1.0, -1.0, 0.25, 2.25]);
    let expected = [
        3.75,
        -2.25,
        0.0,
        -0.875,
        16_777_216.0,
        -16_777_216.0,
        0.375,
        -98.25,
    ];
    let mut values = [91.5; 8];
    let mut output = TestOutput::new(&mut values);
    for index in 0..expected.len() {
        invoke(index, &a, &b, &mut output);
    }
    assert_eq!(bits(output.values), bits(&expected));
    let active = (0..expected.len()).collect::<Vec<_>>();
    assert_reads(&a, &b, &active);
    assert_eq!(output.attempts, active);
}

#[test]
fn shared_body_floating_special_values_preserve_observed_host_semantics() {
    let a = TrackedInput::new(&[
        0.0,
        -0.0,
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        1.0,
        f32::MAX,
        -f32::MAX,
    ]);
    let b = TrackedInput::new(&[
        0.0,
        -0.0,
        0.0,
        -0.0,
        2.0,
        -2.0,
        f32::NEG_INFINITY,
        1.0,
        f32::NAN,
        f32::MAX,
        -f32::MAX,
    ]);
    // None requires a computed NaN, without requiring its sign or payload.
    let expected = [
        Some(0x0000_0000),
        Some(0x8000_0000),
        Some(0x0000_0000),
        Some(0x0000_0000),
        Some(0x7f80_0000),
        Some(0xff80_0000),
        None,
        None,
        None,
        Some(0x7f80_0000),
        Some(0xff80_0000),
    ];
    let mut values = [91.5; 11];
    let mut output = TestOutput::new(&mut values);
    for index in 0..expected.len() {
        invoke(index, &a, &b, &mut output);
    }
    for (value, expected) in output.values.iter().zip(expected) {
        match expected {
            Some(expected) => assert_eq!(value.to_bits(), expected),
            None => assert!(value.is_nan()),
        }
    }
    let active = (0..expected.len()).collect::<Vec<_>>();
    assert_reads(&a, &b, &active);
    assert_eq!(output.attempts, active);
}

#[test]
fn shared_body_output_extent_canaries_and_launch_tail() {
    const CANARY: f32 = -8192.5;
    for length in [1, 63, 64, 65] {
        let left = vec![1.25; length];
        let right = vec![2.5; length];
        let a = TrackedInput::new(&left);
        let b = TrackedInput::new(&right);
        let mut backing = vec![CANARY; length + 4];
        let launched = length.div_ceil(64) * 64 + 3;
        {
            let mut output = TestOutput::new(&mut backing[2..length + 2]);
            for index in 0..launched {
                invoke(index, &a, &b, &mut output);
            }
            assert!(
                output
                    .values
                    .iter()
                    .all(|value| value.to_bits() == 3.75_f32.to_bits())
            );
            assert_eq!(output.attempts, (0..launched).collect::<Vec<_>>());
        }
        assert_reads(&a, &b, &(0..length).collect::<Vec<_>>());
        assert_eq!(bits(&backing[..2]), vec![CANARY.to_bits(); 2]);
        assert_eq!(bits(&backing[length + 2..]), vec![CANARY.to_bits(); 2]);
    }
}

#[test]
fn shared_body_empty_and_extreme_inactive_indices_do_not_read_inputs() {
    let a = TrackedInput::new(&[]);
    let b = TrackedInput::new(&[]);
    let mut empty = [];
    let mut output = TestOutput::new(&mut empty);
    invoke(0, &a, &b, &mut output);
    invoke(usize::MAX, &a, &b, &mut output);
    assert_eq!(output.attempts, [0, usize::MAX]);
    assert_reads(&a, &b, &[]);

    let mut backing = [-77.5, 91.5, 92.5, -78.5];
    {
        let mut output = TestOutput::new(&mut backing[1..3]);
        invoke(2, &a, &b, &mut output);
        invoke(usize::MAX, &a, &b, &mut output);
        assert_eq!(output.attempts, [2, usize::MAX]);
    }
    assert_eq!(bits(&backing), bits(&[-77.5, 91.5, 92.5, -78.5]));
    assert_reads(&a, &b, &[]);
}

#[test]
fn shared_body_accepts_distinct_sufficient_input_extents() {
    let a = TrackedInput::new(&[1.25, -3.5, 800.0, 900.0]);
    let b = TrackedInput::new(&[2.5, 1.25, 1000.0]);
    let mut values = [91.5; 2];
    let mut output = TestOutput::new(&mut values);
    for index in 0..6 {
        invoke(index, &a, &b, &mut output);
    }
    assert_eq!(bits(output.values), bits(&[3.75, -2.25]));
    assert_reads(&a, &b, &[0, 1]);
    assert_eq!(output.attempts, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn shared_body_active_short_inputs_panic_without_rolling_back_prior_writes() {
    for short_left in [true, false] {
        let full_left = [1.25, -3.5];
        let full_right = [2.5, 1.25];
        let a = TrackedInput::new(if short_left {
            &full_left[..1]
        } else {
            &full_left
        });
        let b = TrackedInput::new(if short_left {
            &full_right
        } else {
            &full_right[..1]
        });
        let mut values = [91.5; 2];
        let mut output = TestOutput::new(&mut values);
        invoke(0, &a, &b, &mut output);
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| {
            invoke(1, &a, &b, &mut output);
        }));
        assert!(failed.is_err());
        assert_eq!(bits(output.values), bits(&[3.75, 91.5]));
        assert_eq!(a.reads.borrow().as_slice(), &[0, 1]);
        assert_eq!(
            b.reads.borrow().as_slice(),
            if short_left { &[0][..] } else { &[0, 1][..] }
        );
        invoke(2, &a, &b, &mut output);
        invoke(usize::MAX, &a, &b, &mut output);
        assert_eq!(output.attempts, [0, 1, 2, usize::MAX]);
        assert_eq!(a.reads.borrow().as_slice(), &[0, 1]);
        assert_eq!(
            b.reads.borrow().as_slice(),
            if short_left { &[0][..] } else { &[0, 1][..] }
        );
        assert_eq!(bits(output.values), bits(&[3.75, 91.5]));
    }
}

impl TestIndex {
    fn checked_tiled_2d<const X: usize, const Y: usize, const W: usize, const H: usize>(
        &self,
    ) -> Option<Self> {
        assert_eq!((X, Y, W, H), (64, 1, 64, 1));
        Some(*self)
    }
}
impl TestOutput<'_> {
    fn len(&self) -> usize {
        self.values.len()
    }
    fn get_tiled_2d_mut(
        &mut self,
        index: &TestIndex,
        base: usize,
        rows: usize,
        columns: usize,
        stride: usize,
    ) -> Option<&mut f32> {
        assert_eq!(
            (base, rows, columns, stride),
            (0, 1, self.values.len(), self.values.len())
        );
        self.get_mut(*index)
    }
}
fn invoke_tile(
    linear: usize,
    a: &TrackedInput<'_>,
    b: &TrackedInput<'_>,
    output: &mut TestOutput<'_>,
) {
    vecadd_tile_body!(cpu_thread, (linear), production_f32_add, a, b, output);
}

#[test]
fn both_bodies_full_nonquarter_oracle_tails_canaries_and_immutable_inputs() {
    for length in [
        0_usize, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025,
    ] {
        let left = (0..length)
            .map(|i| ((i % 29) as i32 - 14) as f32 / 7.0)
            .collect::<Vec<_>>();
        let right = (0..length)
            .map(|i| ((i % 31) as i32 - 15) as f32 / 11.0)
            .collect::<Vec<_>>();
        let before = (bits(&left), bits(&right));
        let expected = left
            .iter()
            .zip(&right)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        for body in [invoke, invoke_tile] {
            let a = TrackedInput::new(&left);
            let b = TrackedInput::new(&right);
            let guard = f32::from_bits(0x7fc1_abcd);
            let mut backing = vec![guard; length + 4];
            let launched = length.div_ceil(256) * 256 + 3;
            {
                let mut output = TestOutput::new(&mut backing[2..2 + length]);
                for i in 0..launched {
                    body(i, &a, &b, &mut output);
                }
                body(usize::MAX, &a, &b, &mut output);
                assert_eq!(bits(output.values), bits(&expected));
            }
            assert_reads(&a, &b, &(0..length).collect::<Vec<_>>());
            assert_eq!(bits(&backing[..2]), vec![guard.to_bits(); 2]);
            assert_eq!(bits(&backing[2 + length..]), vec![guard.to_bits(); 2]);
            assert_eq!((bits(&left), bits(&right)), before);
        }
    }
}
#[test]
fn tile_retains_sufficient_and_short_input_contracts() {
    let a = TrackedInput::new(&[1.0 / 7.0, -3.0 / 11.0, 99.0]);
    let b = TrackedInput::new(&[2.0 / 13.0, 1.0 / 17.0, 100.0, 200.0]);
    let mut values = [0.0; 2];
    let mut output = TestOutput::new(&mut values);
    for i in 0..4 {
        invoke_tile(i, &a, &b, &mut output);
    }
    assert_eq!(
        bits(output.values),
        bits(&[1.0 / 7.0 + 2.0 / 13.0, -3.0 / 11.0 + 1.0 / 17.0])
    );
    assert_reads(&a, &b, &[0, 1]);
    for short_left in [true, false] {
        let aa = [1.25, -3.5];
        let bb = [2.5, 1.25];
        let a = TrackedInput::new(if short_left { &aa[..1] } else { &aa });
        let b = TrackedInput::new(if short_left { &bb } else { &bb[..1] });
        let mut values = [91.5; 2];
        let mut output = TestOutput::new(&mut values);
        invoke_tile(0, &a, &b, &mut output);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| invoke_tile(1, &a, &b, &mut output)))
                .is_err()
        );
        assert_eq!(bits(output.values), bits(&[3.75, 91.5]));
        invoke_tile(usize::MAX, &a, &b, &mut output);
    }
}

#[test]
fn tile_floating_special_values_preserve_observed_host_semantics() {
    let a = TrackedInput::new(&[
        0.0,
        -0.0,
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        1.0,
        f32::MAX,
        -f32::MAX,
    ]);
    let b = TrackedInput::new(&[
        0.0,
        -0.0,
        0.0,
        -0.0,
        2.0,
        -2.0,
        f32::NEG_INFINITY,
        1.0,
        f32::NAN,
        f32::MAX,
        -f32::MAX,
    ]);
    // None requires a computed NaN, without requiring its sign or payload.
    let expected = [
        Some(0x0000_0000),
        Some(0x8000_0000),
        Some(0x0000_0000),
        Some(0x0000_0000),
        Some(0x7f80_0000),
        Some(0xff80_0000),
        None,
        None,
        None,
        Some(0x7f80_0000),
        Some(0xff80_0000),
    ];
    let mut values = [91.5; 11];
    let mut output = TestOutput::new(&mut values);
    for index in 0..expected.len() {
        invoke_tile(index, &a, &b, &mut output);
    }
    for (value, expected) in output.values.iter().zip(expected) {
        match expected {
            Some(expected) => assert_eq!(value.to_bits(), expected),
            None => assert!(value.is_nan()),
        }
    }
    let active = (0..expected.len()).collect::<Vec<_>>();
    assert_reads(&a, &b, &active);
    assert_eq!(output.attempts, active);
}

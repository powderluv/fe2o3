use fe2o3_row_softmax_general_v1::reference::{ReferenceLayoutV1, evaluate_reference_v1};
#[test]
fn full_strided_nonquarter_rows_tails_and_canaries() {
    for rows in [0_u32, 1, 3] {
        for columns in [1_u32, 3, 63, 64, 65, 129, 4096] {
            let input_stride = columns + 7;
            let output_stride = columns + 11;
            let input: Vec<f32> = (0..rows * input_stride + 5)
                .map(|i| (((i * 37 + 11) % 101) as f32 - 50.0) / 13.7)
                .collect();
            let input_bits: Vec<_> = input.iter().map(|x| x.to_bits()).collect();
            let canary = f32::from_bits(0xc53abcdf);
            let initial = vec![canary; (rows * output_stride + 9) as usize];
            let output = evaluate_reference_v1(
                &input,
                &initial,
                ReferenceLayoutV1 {
                    rows,
                    columns,
                    input_stride,
                    output_stride,
                },
            )
            .unwrap();
            for row in 0..rows as usize {
                let start = row * input_stride as usize;
                let values = &input[start..start + columns as usize];
                let maximum = values
                    .iter()
                    .map(|&x| x as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let denominator: f64 = values.iter().map(|&x| (x as f64 - maximum).exp()).sum();
                let mut sum = 0.0_f64;
                for (column, &x) in values.iter().enumerate() {
                    let got = output[row * output_stride as usize + column] as f64;
                    let expected = (x as f64 - maximum).exp() / denominator;
                    assert!((got - expected).abs() <= 2.0e-5 * expected.abs().max(1.0e-6));
                    sum += got;
                }
                assert!((sum - 1.0).abs() < 2.0e-5);
                for &x in &output[row * output_stride as usize + columns as usize
                    ..(row + 1) * output_stride as usize]
                {
                    assert_eq!(x.to_bits(), canary.to_bits());
                }
            }
            assert!(
                output[(rows * output_stride) as usize..]
                    .iter()
                    .all(|x| x.to_bits() == canary.to_bits())
            );
            assert_eq!(
                input.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                input_bits
            );
            assert!(initial.iter().all(|x| x.to_bits() == canary.to_bits()));
        }
    }
}
#[test]
fn invalid_dynamic_layouts_reject() {
    for layout in [
        ReferenceLayoutV1 {
            rows: 1,
            columns: 0,
            input_stride: 1,
            output_stride: 1,
        },
        ReferenceLayoutV1 {
            rows: 1,
            columns: 2,
            input_stride: 1,
            output_stride: 2,
        },
        ReferenceLayoutV1 {
            rows: 1,
            columns: 2,
            input_stride: 2,
            output_stride: 1,
        },
        ReferenceLayoutV1 {
            rows: 2,
            columns: 2,
            input_stride: 2,
            output_stride: 2,
        },
    ] {
        assert!(evaluate_reference_v1(&[0.1; 2], &[0.2; 2], layout).is_err());
    }
}

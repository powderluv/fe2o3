use fe2o3_scalar_gemm_v1::harness::{Shape, scalar_gemm_oracle};
#[test]
fn full_nonquarter_boundary_matrix_against_f64_dot_products() {
    for (m, n, k) in [
        (0, 257, 3),
        (7, 0, 3),
        (3, 5, 0),
        (1, 1, 1),
        (1, 255, 7),
        (1, 256, 7),
        (1, 257, 7),
        (7, 5, 9),
    ] {
        let shape = Shape::checked(m, n, k).unwrap();
        let a: Vec<f32> = (0..shape.a_len)
            .map(|i| ((i * 17 % 71) as f32 - 35.0) / 11.3)
            .collect();
        let b: Vec<f32> = (0..shape.b_len)
            .map(|i| ((i * 23 % 79) as f32 - 39.0) / 17.9)
            .collect();
        let before_a: Vec<_> = a.iter().map(|x| x.to_bits()).collect();
        let before_b: Vec<_> = b.iter().map(|x| x.to_bits()).collect();
        let output = scalar_gemm_oracle(shape, &a, &b);
        assert_eq!(output.len(), shape.c_len);
        for row in 0..shape.m {
            for col in 0..shape.n {
                let expected: f64 = (0..shape.k)
                    .map(|t| a[row * shape.k + t] as f64 * b[t * shape.n + col] as f64)
                    .sum();
                let got = output[row * shape.n + col] as f64;
                assert!((got - expected).abs() < 1.0e-5 * expected.abs().max(1.0));
                if k == 0 {
                    assert_eq!(output[row * shape.n + col].to_bits(), 0);
                }
            }
        }
        assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), before_a);
        assert_eq!(b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), before_b);
    }
}

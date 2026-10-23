#[cfg(feature = "qualification-legacy-hip")]
mod qualification;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "qualification-legacy-hip")]
    {
        qualification::run()
    }
    #[cfg(not(feature = "qualification-legacy-hip"))]
    {
        use fe2o3_scalar_gemm_v1::harness::{Shape, scalar_gemm_inputs, scalar_gemm_oracle};
        let shape = Shape::checked(2, 3, 4)?;
        let (a, b) = scalar_gemm_inputs(shape);
        let expected = scalar_gemm_oracle(shape, &a, &b);
        println!("scalar GEMM V1 CPU oracle: {expected:?}");
        println!(
            "production GPU execution requires Worker V3; the documented runner selects explicit legacy-HIP numerical qualification"
        );
        Ok(())
    }
}

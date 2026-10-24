#[cfg(feature = "qualification-legacy-hip")]
mod qualification;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "qualification-legacy-hip")]
    {
        qualification::run()
    }
    #[cfg(not(feature = "qualification-legacy-hip"))]
    {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "the production Worker V3 application verifier is not wired for fe2o3-row_softmax_general_v1; the documented runner selects explicit legacy-HIP numerical qualification",
        ).into())
    }
}

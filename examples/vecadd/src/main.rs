use fe2o3_device::{DisjointSlice, kernel, thread};

include!("vecadd_body.rs");

macro_rules! production_f32_add {
    ($lhs:expr, $rhs:expr) => {{ $lhs + $rhs }};
}

#[cfg(not(feature = "structured-tile"))]
#[kernel(typed)]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    vecadd_kernel_body!(thread, (), production_f32_add, a, b, c);
}

#[cfg(feature = "structured-tile")]
include!("vecadd_tile_body.rs");

#[cfg(feature = "structured-tile")]
#[kernel(typed)]
pub fn vecadd(
    a: &[f32],
    b: &[f32],
    mut c: DisjointSlice<f32, fe2o3_device::Tiled2D<fe2o3_device::Index1D, 64, 1, 64, 1>>,
) {
    vecadd_tile_body!(thread, (), production_f32_add, a, b, c);
}

#[cfg(feature = "qualification-legacy-hip")]
mod qualification;

#[cfg(feature = "qualification-legacy-hip")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    qualification::run()
}

#[cfg(not(feature = "qualification-legacy-hip"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "the production Worker V3 application verifier is not wired for fe2o3-vecadd",
    )
    .into())
}

#[cfg(test)]
mod tests {
    const KERNEL_SOURCE: &str = include_str!("main.rs");
    const SHARED_BODY: &str = include_str!("vecadd_body.rs");

    #[allow(dead_code)]
    fn generated_v3_arguments_typecheck<'allocation>(
        a: &'allocation [f32],
        b: &'allocation [f32],
        c: &'allocation mut [f32],
    ) {
        let a = fe2o3_host::__generated::GeneratedKfdReadSlice::new(a);
        let b = fe2o3_host::__generated::GeneratedKfdReadSlice::new(b);
        let c = fe2o3_host::__generated::GeneratedKfdReadWriteSlice::new(c);
        let _arguments: super::vecadd_gpu::Arguments<'allocation> =
            super::vecadd_gpu::Arguments::new(a, b, c);
    }

    #[test]
    fn real_kernel_expands_the_shared_body() {
        assert!(KERNEL_SOURCE.contains("include!(\"vecadd_body.rs\")"));
        assert!(
            KERNEL_SOURCE.contains("vecadd_kernel_body!(thread, (), production_f32_add, a, b, c)")
        );
        assert!(KERNEL_SOURCE.contains("macro_rules! production_f32_add"));
        assert!(KERNEL_SOURCE.contains("$lhs + $rhs"));
    }

    #[test]
    fn example_contains_only_the_worker_v3_host_contract() {
        let production_source = KERNEL_SOURCE
            .split("#[cfg(test)]")
            .next()
            .expect("example has production source");

        for required in [
            "#[kernel(typed)]",
            "production Worker V3 application verifier",
        ] {
            assert!(production_source.contains(required), "missing `{required}`");
        }
        assert!(!production_source.contains("namespace ="));

        for forbidden in [
            "#[kernel]",
            "FE2O3_HSACO_DIR",
            "load_module_from_file",
            "launch!",
            "LaunchConfig",
            "FE2O3_CODEGEN_PIPELINE",
            "qualification_worker_v2",
            "qualification-embedded-vecadd-test-only",
            "vecadd_gpu::Kernel",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "typed example retained `{forbidden}`"
            );
        }
    }

    #[test]
    fn shared_body_retains_the_verified_memory_shape() {
        for operation in [
            "let idx = $thread::index_1d",
            "let i = idx.get()",
            "if let Some(out) = $output.get_mut(idx)",
            "*out = $add!($a[i], $b[i])",
        ] {
            assert!(SHARED_BODY.contains(operation), "missing `{operation}`");
        }

        let guard = SHARED_BODY.find("if let Some(out)").unwrap();
        let first_input_access = SHARED_BODY.find("$a[i]").unwrap();
        assert!(guard < first_input_access, "input access escaped the guard");
    }
}

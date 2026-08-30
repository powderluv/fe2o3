#![doc = include_str!("../README.md")]

mod api;
mod dispatch;
mod environment;
mod lifecycle;
mod sys;
#[cfg(test)]
mod test_process_execution {
    use std::io;
    use std::process::{Command, ExitStatus};

    pub(super) fn status(command: &mut Command) -> io::Result<ExitStatus> {
        fe2o3_artifact_transaction::with_artifact_process_spawn_v1(|| command.spawn())
            .and_then(|mut child| child.wait())
    }
}
#[cfg(feature = "hardware-test-hooks")]
pub use dispatch::{
    ReviewedHsaHardwareTestBufferV1, ReviewedHsaProfiledDispatchObservationV1,
    ReviewedHsaProfiledDispatchSessionV1,
};
pub use environment::{HsaRuntimeAdapterError, ReviewedHsaRuntimeAdapterV1};
pub use lifecycle::{ReviewedHsaExecutableV1, ReviewedHsaKernelSetV1, ReviewedHsaKernelV1};

/// Whether this build found reviewed HSA and HIP headers and runtime libraries.
pub const HSA_RUNTIME_AVAILABLE: bool = cfg!(fe2o3_hsa_runtime);

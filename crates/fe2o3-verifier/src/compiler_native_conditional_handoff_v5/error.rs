use super::*;
use crate::compiler_native_conditional_source_proof_v2::final_replay::manifest::ManifestErrorV5;
use std::fmt;

/// Opaque typed terminal refusal. Its source chain is deliberately empty so
/// ordinary refund classifiers cannot reinterpret a nested refundable error.
#[derive(Debug)]
pub struct CompilerConditionalNativeSemanticHandoffErrorV5(pub(super) Cause);

#[derive(Debug)]
pub(super) enum Cause {
    Resource(Resource),
    Final(NativeConditionalFinalErrorV2),
    Source(NativeConditionalSourceProofErrorV2),
    History(fe2o3_kernel_opt::RefinedForwardingHistoryWireErrorV1),
    Catalog(fe2o3_kernel_ir::KernelIrContractCatalogErrorV1),
    Descriptor(fe2o3_kernel_descriptor::DescriptorWireErrorV5<Resource>),
    Inventory(fe2o3_compiler_lineage::RustcEnrollmentInventoryErrorV1<Resource>),
    Manifest(ManifestErrorV5),
    TargetLineage(fe2o3_compiler_lineage::ProductionTargetLineageErrorV3),
    Subject(fe2o3_compiler_lineage::NativeNeutralSubjectErrorV1),
    Utf8(std::str::Utf8Error),
    Mismatch(&'static str),
}
impl Error {
    pub(super) fn mismatch(rule: &'static str) -> Self {
        Self(Cause::Mismatch(rule))
    }
}
impl From<Resource> for Error {
    fn from(error: Resource) -> Self {
        Self(Cause::Resource(error))
    }
}
impl From<NativeConditionalFinalErrorV2> for Error {
    fn from(error: NativeConditionalFinalErrorV2) -> Self {
        Self(Cause::Final(error))
    }
}
impl From<NativeConditionalSourceProofErrorV2> for Error {
    fn from(error: NativeConditionalSourceProofErrorV2) -> Self {
        Self(Cause::Source(error))
    }
}
impl fmt::Display for Error {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(out, "conditional native V5 content: {:?}", self.0)
    }
}
impl std::error::Error for Error {}

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

mod bounded_pair;
mod capsule;
mod capsule_v4;
mod capsule_v5;
mod error;
mod mixed_middle_end_v29;
mod mixed_middle_end_v50;
mod mixed_middle_end_v90;
mod mixed_target_selection_family;
mod mixed_target_selection_v53;
mod mixed_target_selection_v89;
mod multi_root_correspondence_payload_v2;
mod multi_root_proof_roster_v2;
mod multi_root_target_lineage_v2;
mod multi_root_target_lineage_v3;
mod native_conditional_carrier_v1;
mod native_conditional_metadata_v1;
mod native_conditional_metadata_v2;
mod native_conditional_output_v1;
mod native_conditional_policy_roster_v1;
mod native_lowering_association_v1;
mod native_neutral_module_v1;
mod native_neutral_subject_v1;
mod native_output_transition_association_v1;
mod native_refined_forwarding_carrier_v1;
mod proof_binding;
mod proof_binding_v4;
mod receipt;
mod rustc_enrollment_inventory_v1;
mod semantic_to_llvm_v3;
mod target_lineage_v3;

pub use capsule::{
    INERT_PRODUCTION_SEMANTIC_CAPSULE_MAGIC_V3, INERT_PRODUCTION_SEMANTIC_CAPSULE_VERSION_V3,
    InertProductionSemanticCapsuleIdentityV3, InertProductionSemanticCapsuleV3,
    MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3,
    MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_DECODE_OWNED_BYTES_V3,
    OrderedInertSemanticLineageReceiptsV3,
};
pub use capsule_v4::*;
pub use capsule_v5::*;
pub use error::{LineageDecodeErrorV3, LineageErrorV3};
pub use mixed_middle_end_v29::*;
pub use mixed_middle_end_v50::*;
pub use mixed_middle_end_v90::*;
pub use mixed_target_selection_v53::*;
pub use mixed_target_selection_v89::*;
pub use multi_root_correspondence_payload_v2::{
    MULTI_ROOT_CORRESPONDENCE_PAYLOAD_MAGIC_V2, MULTI_ROOT_CORRESPONDENCE_PAYLOAD_POLICY_V2,
    MULTI_ROOT_CORRESPONDENCE_PAYLOAD_VERSION_V2, MultiRootCorrespondenceBlockV2,
    MultiRootCorrespondenceFunctionRoleV2, MultiRootCorrespondenceFunctionV2,
    MultiRootCorrespondenceParameterV2, MultiRootCorrespondencePayloadErrorV2,
    MultiRootCorrespondencePayloadV2, MultiRootCorrespondenceStatementV2,
    MultiRootCorrespondenceSyntheticRuleV2, MultiRootCorrespondenceSyntheticV2,
    MultiRootCorrespondenceTerminatorV2,
};
pub use multi_root_proof_roster_v2::{
    MAX_MULTI_ROOT_PROOF_ROSTER_ROOTS_V2, MULTI_ROOT_PROOF_ROSTER_POLICY_V2,
    MULTI_ROOT_PROOF_ROSTER_VERSION_V2, MultiRootCanonicalKirVersionV2,
    MultiRootNeutralKirIdentityV2, MultiRootProofRosterErrorV2, MultiRootProofRosterInputsV2,
    MultiRootProofRosterKindV2, MultiRootProofRosterRootInputV2, MultiRootProofRosterRootV2,
    MultiRootProofRosterTranscriptV2,
};
pub use multi_root_proof_roster_v2::{
    MAX_MULTI_ROOT_PROOF_ROSTER_ROOTS_V3, MULTI_ROOT_PROOF_ROSTER_POLICY_V3,
    MULTI_ROOT_PROOF_ROSTER_VERSION_V3, MultiRootProofRosterErrorV3, MultiRootProofRosterInputsV3,
    MultiRootProofRosterKindV3, MultiRootProofRosterRootInputV3, MultiRootProofRosterRootV3,
    MultiRootProofRosterTranscriptV3,
};
pub use multi_root_target_lineage_v2::{
    MAX_MULTI_ROOT_TARGET_BINDING_ROOTS_V2, MULTI_ROOT_TARGET_BINDING_MAGIC_V2,
    MULTI_ROOT_TARGET_BINDING_VERSION_V2, MultiRootTargetBindingInputsV2,
    MultiRootTargetBindingTranscriptV2, MultiRootTargetWorkgroupInputV2,
    MultiRootTargetWorkgroupV2,
};
pub use multi_root_target_lineage_v3::{
    MAX_MULTI_ROOT_TARGET_BINDING_ROOTS_V3, MULTI_ROOT_TARGET_BINDING_MAGIC_V3,
    MULTI_ROOT_TARGET_BINDING_VERSION_V3, MultiRootTargetBindingInputsV3,
    MultiRootTargetBindingTranscriptV3, MultiRootTargetWorkgroupInputV3,
    MultiRootTargetWorkgroupV3,
};
pub use native_conditional_carrier_v1::*;
pub use native_conditional_metadata_v1::*;
pub use native_conditional_metadata_v2::*;
pub use native_conditional_output_v1::*;
pub use native_conditional_policy_roster_v1::*;
pub use native_lowering_association_v1::*;
pub use native_neutral_module_v1::{
    NativeNeutralModuleErrorV1, NativeNeutralModuleRefV1, encode_native_neutral_module_v1,
};
pub use native_neutral_subject_v1::{
    InertNativeNeutralSubjectV1, NATIVE_NEUTRAL_SUBJECT_BYTES_V1, NATIVE_NEUTRAL_SUBJECT_POLICY_V1,
    NATIVE_NEUTRAL_SUBJECT_VERSION_V1, NativeNeutralSubjectErrorV1, NativeNeutralSubjectIdentityV1,
};
pub use native_output_transition_association_v1::{
    NATIVE_OUTPUT_TRANSITION_ASSOCIATION_HEADER_BYTES_V1,
    NATIVE_OUTPUT_TRANSITION_ASSOCIATION_MAGIC_V1, NATIVE_OUTPUT_TRANSITION_ASSOCIATION_POLICY_V1,
    NATIVE_OUTPUT_TRANSITION_ASSOCIATION_VERSION_V1, NativeOutputTransitionAssociationErrorV1,
    NativeOutputTransitionAssociationInputsV1, NativeOutputTransitionAssociationRefV1,
    NativeOutputTransitionRootV1, encode_native_output_transition_association_v1,
    native_output_transition_association_length_v1,
};
pub use native_refined_forwarding_carrier_v1::*;
pub use proof_binding::{
    INERT_PROOF_BINDING_ASSOCIATION_MAGIC_V3, INERT_PROOF_BINDING_ASSOCIATION_VERSION_V3,
    InertLineageContentIdentityV3, InertProofBindingAssociationErrorV3,
    InertProofBindingAssociationInputsV3, InertProofBindingAssociationV3,
    MAX_INERT_PROOF_BINDING_ASSOCIATION_BYTES_V3,
};
pub use proof_binding_v4::{
    INERT_PROOF_BINDING_ASSOCIATION_MAGIC_V4, INERT_PROOF_BINDING_ASSOCIATION_VERSION_V4,
    InertProofBindingAssociationErrorV4, InertProofBindingAssociationInputsV4,
    InertProofBindingAssociationV4, MAX_INERT_PROOF_BINDING_ASSOCIATION_BYTES_V4,
    MAX_INERT_PROOF_BINDING_VERUS_EVIDENCE_BYTES_V4,
};
pub use receipt::{
    InertAbiReceiptIdentityV3, InertAbiReceiptV3, InertAmdgpuLoweringReceiptIdentityV3,
    InertAmdgpuLoweringReceiptV3, InertCanonicalSemanticMirIdentityV3,
    InertCanonicalSemanticMirReceiptV3, InertDataLayoutReceiptIdentityV3, InertDataLayoutReceiptV3,
    InertExportManifestReceiptIdentityV3, InertExportManifestReceiptV3,
    InertFinalCompilerModuleCommitmentIdentityV3, InertFinalCompilerModuleCommitmentReceiptV3,
    InertFormalMemoryReceiptIdentityV3, InertFormalMemoryReceiptV3, InertKernelIrReceiptIdentityV3,
    InertKernelIrReceiptV3, InertMiddleEndReceiptIdentityV3, InertMiddleEndReceiptV3,
    InertMirToKirCorrespondenceReceiptIdentityV3, InertMirToKirCorrespondenceReceiptV3,
    InertProofBindingReceiptIdentityV3, InertProofBindingReceiptV3,
    InertRustcIdentityInventoryReceiptIdentityV3, InertRustcIdentityInventoryReceiptV3,
    InertRustcPreflightPlanReceiptIdentityV3, InertRustcPreflightPlanReceiptV3,
    InertSemanticToLlvmReceiptIdentityV3, InertSemanticToLlvmReceiptV3,
    InertTargetBindingReceiptIdentityV3, InertTargetBindingReceiptV3,
    MAX_CANONICAL_SEMANTIC_MIR_BYTES_V3, MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3,
};
pub use rustc_enrollment_inventory_v1::*;
pub use semantic_to_llvm_v3::{
    INERT_SEMANTIC_TO_LLVM_ASSOCIATION_MAGIC_V3, INERT_SEMANTIC_TO_LLVM_ASSOCIATION_VERSION_V3,
    InertSemanticToLlvmAssociationErrorV3, InertSemanticToLlvmAssociationInputsV3,
    InertSemanticToLlvmAssociationV3, InertSemanticToLlvmContentIdentityV3,
    MAX_INERT_SEMANTIC_TO_LLVM_ASSOCIATION_BYTES_V3,
};
pub use target_lineage_v3::{
    ASSOCIATION_ONLY_NO_REFINEMENT_PROOF_POLICY_V3, DataLayoutTranscriptInputsV3,
    DataLayoutTranscriptV3, MAX_PRODUCTION_TARGET_LINEAGE_TRANSCRIPT_BYTES_V3,
    ProductionTargetLineageErrorV3, SemanticToLlvmAssociationInputsV3,
    SemanticToLlvmAssociationTranscriptV3, TargetBindingTranscriptInputsV3,
    TargetBindingTranscriptV3, TargetLineageClaimV3, TargetLineageIdentityV3,
    canonical_semantic_target_layout_transcript_v1, derive_semantic_target_layout_identity_v1,
};

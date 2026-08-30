//! Independent validation of compiler proof inputs carried by the frozen V3 capsule envelope.
//!
//! The legacy V3 association validates exact canonical content and structural relationships only.
//! The current V4 association additionally imports the exact signed aggregate
//! MIR-to-live-PLIRON Verus receipt under its embedded key. Neither path authenticates compiler
//! origin, establishes LLVM or machine refinement, or grants runtime authority.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use fe2o3_compiler_lineage::{
    InertCanonicalSemanticMirReceiptV3, InertFormalMemoryReceiptV3, InertKernelIrReceiptV3,
    InertLineageContentIdentityV3, InertMiddleEndReceiptV3, InertMirToKirCorrespondenceReceiptV3,
    InertProofBindingAssociationErrorV3, InertProofBindingAssociationErrorV4,
    InertProofBindingAssociationV3, InertProofBindingAssociationV4,
    InertProofBindingReceiptIdentityV3, InertProofBindingReceiptV3,
};
use fe2o3_kernel_ir::{
    AddressSpace, AmdGpuDiagnosticOperation, BasicBlock, BinaryOp, CheckedBinaryOperator,
    FunctionBody, MemoryAccess, Module, OperationKind, Terminator,
    VerifiedCanonicalKernelIrErrorV5, VerifiedCanonicalKernelIrErrorV8,
    VerifiedCanonicalKernelIrV5, VerifiedCanonicalKernelIrV8,
};
use fe2o3_lower_mir_kernel::{
    InertCanonicalFormalMemoryAdmissionEvidenceV3, InertCanonicalFormalMemoryAdmissionEvidenceV4,
    InertCanonicalMirToKirCorrespondenceEvidenceV3, InertCanonicalMirToKirCorrespondenceEvidenceV4,
    MirToKirSyntheticRuleEvidenceV4, ProductionCanonicalKernelIrVersionV1,
    ProductionCorrespondenceEvidenceErrorV4, ProductionFormalMemoryEvidenceErrorV4,
    ProductionLineageEvidenceErrorV3,
};
use fe2o3_mir_model::semantic_mir_v1::{
    AdmittedInertSemanticMirV1, SemanticCheckedBinaryOpV1, SemanticFunctionDeclV1,
    SemanticFunctionIdV1, SemanticLocalRoleV1, SemanticMirDecodeErrorV1, SemanticMirLimitsV1,
    SemanticRvalueKindV1, SemanticStatementKindV1,
};
use fe2o3_mir_model::{
    InertCanonicalSemanticU32InductionEvidenceV1, SemanticU32InductionAnalysisErrorV1,
    SemanticU32InductionEvidenceErrorV1, analyze_semantic_u32_induction_no_overflow_v1,
};
use fe2o3_pliron::{InertProductionMiddleEndEvidenceV5, ProductionMiddleEndEvidenceCodecErrorV5};

use crate::{
    CanonicalProductionMirPlironVerusExecutionEvidenceV1,
    ProductionMirPlironVerusExecutionEvidenceErrorV1,
};

/// Independently decoded and cross-checked V3 compiler proof inputs.
///
/// The value owns the exact semantic MIR, middle-end evidence, verified canonical Kernel IR,
/// MIR-to-KIR correspondence, formal-memory admission, and their outer association. It is
/// intentionally non-`Clone` so a later authority-bearing join can consume the one checked
/// occurrence without changing this type's current non-authoritative contract.
///
/// ```compile_fail
/// use fe2o3_verifier::ValidatedCompilerProofInputsV3;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<ValidatedCompilerProofInputsV3>();
/// ```
#[derive(Debug)]
#[must_use = "dropping validated proof inputs abandons the exact decoded compiler evidence"]
pub struct ValidatedCompilerProofInputsV3 {
    association: InertProofBindingAssociationV3,
    receipt_identity: InertProofBindingReceiptIdentityV3,
    semantic_mir: AdmittedInertSemanticMirV1,
    middle_end: InertProductionMiddleEndEvidenceV5,
    kernel_ir: VerifiedCanonicalKernelIrV5,
    correspondence: InertCanonicalMirToKirCorrespondenceEvidenceV3,
    formal_memory: InertCanonicalFormalMemoryAdmissionEvidenceV3,
}

/// Independently decoded current compiler proof inputs, including the exact signed aggregate
/// MIR-to-live-PLIRON Verus receipt.
///
/// This value remains non-authoritative until a private host join consumes protected compiler
/// origin. It deliberately establishes no LLVM, ISA, machine, load, or launch claim.
///
/// ```compile_fail
/// use fe2o3_verifier::ValidatedCompilerProofInputsV4;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<ValidatedCompilerProofInputsV4>();
/// ```
#[derive(Debug)]
#[must_use = "dropping validated V4 proof inputs abandons the exact signed compiler evidence"]
pub struct ValidatedCompilerProofInputsV4 {
    association: InertProofBindingAssociationV4,
    receipt_identity: InertProofBindingReceiptIdentityV3,
    semantic_mir: AdmittedInertSemanticMirV1,
    middle_end: InertProductionMiddleEndEvidenceV5,
    kernel_ir: VerifiedCanonicalKernelIrV8,
    correspondence: InertCanonicalMirToKirCorrespondenceEvidenceV4,
    formal_memory: InertCanonicalFormalMemoryAdmissionEvidenceV4,
    verus_execution: CanonicalProductionMirPlironVerusExecutionEvidenceV1,
    induction_anchors: Box<[VerifiedSemanticU32InductionKirAnchorV1]>,
}

/// Exact checked KIR addition associated with one replayed semantic induction certificate.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VerifiedSemanticU32InductionKirAnchorV1 {
    semantic_function: u32,
    semantic_block: u32,
    semantic_statement: u32,
    kernel_ir_block: u32,
    kernel_ir_operation: u32,
    value_result: u32,
    overflow_result: u32,
}

impl VerifiedSemanticU32InductionKirAnchorV1 {
    /// Returns the exact semantic function index.
    pub const fn semantic_function(self) -> u32 {
        self.semantic_function
    }

    /// Returns the exact semantic block index.
    pub const fn semantic_block(self) -> u32 {
        self.semantic_block
    }

    /// Returns the exact semantic statement ordinal.
    pub const fn semantic_statement(self) -> u32 {
        self.semantic_statement
    }

    /// Returns the exact KIR block identity.
    pub const fn kernel_ir_block(self) -> u32 {
        self.kernel_ir_block
    }

    /// Returns the exact checked-add operation ordinal inside the KIR block.
    pub const fn kernel_ir_operation(self) -> u32 {
        self.kernel_ir_operation
    }

    /// Returns the checked addition's value-result SSA identity.
    pub const fn value_result(self) -> u32 {
        self.value_result
    }

    /// Returns the checked addition's overflow-result SSA identity.
    pub const fn overflow_result(self) -> u32 {
        self.overflow_result
    }
}

impl ValidatedCompilerProofInputsV3 {
    /// Returns the independently decoded association.
    pub const fn association(&self) -> &InertProofBindingAssociationV3 {
        &self.association
    }

    /// Returns the exact outer lineage-receipt identity whose preimage was decoded.
    pub const fn receipt_identity(&self) -> InertProofBindingReceiptIdentityV3 {
        self.receipt_identity
    }

    /// Returns the independently decoded exact production semantic MIR.
    pub const fn semantic_mir(&self) -> &AdmittedInertSemanticMirV1 {
        &self.semantic_mir
    }

    /// Returns the independently decoded exact V5 middle-end evidence.
    pub const fn middle_end(&self) -> &InertProductionMiddleEndEvidenceV5 {
        &self.middle_end
    }

    /// Returns the independently decoded and semantically verified exact Kernel IR V5.
    pub const fn kernel_ir(&self) -> &VerifiedCanonicalKernelIrV5 {
        &self.kernel_ir
    }

    /// Returns the independently decoded exact MIR-to-KIR correspondence.
    pub const fn correspondence(&self) -> &InertCanonicalMirToKirCorrespondenceEvidenceV3 {
        &self.correspondence
    }

    /// Returns the independently decoded exact formal-memory admission.
    pub const fn formal_memory(&self) -> &InertCanonicalFormalMemoryAdmissionEvidenceV3 {
        &self.formal_memory
    }

    /// Reports that all five exact receipt preimages were independently decoded and associated.
    pub const fn has_exact_decoded_input_association(&self) -> bool {
        true
    }

    /// Reports that retained block locators and source statement counts match decoded MIR and KIR.
    pub const fn has_structural_mir_to_kir_correspondence(&self) -> bool {
        true
    }

    /// Reports that content association alone does not authenticate Verus execution.
    pub const fn authenticates_verus_execution(&self) -> bool {
        false
    }

    /// Reports that content association alone does not establish compiler refinement.
    pub const fn establishes_compiler_refinement(&self) -> bool {
        false
    }

    /// Reports that content association alone grants no load or launch authority.
    pub const fn grants_runtime_authority(&self) -> bool {
        false
    }
}

impl ValidatedCompilerProofInputsV4 {
    /// Returns the independently decoded current association.
    pub const fn association(&self) -> &InertProofBindingAssociationV4 {
        &self.association
    }

    /// Returns the exact outer lineage-receipt identity whose preimage was decoded.
    pub const fn receipt_identity(&self) -> InertProofBindingReceiptIdentityV3 {
        self.receipt_identity
    }

    /// Returns the independently decoded exact production semantic MIR.
    pub const fn semantic_mir(&self) -> &AdmittedInertSemanticMirV1 {
        &self.semantic_mir
    }

    /// Returns the independently decoded exact V5 middle-end evidence.
    pub const fn middle_end(&self) -> &InertProductionMiddleEndEvidenceV5 {
        &self.middle_end
    }

    /// Returns the independently decoded and semantically verified exact Kernel IR V8.
    pub const fn kernel_ir(&self) -> &VerifiedCanonicalKernelIrV8 {
        &self.kernel_ir
    }

    /// Returns the independently decoded exact MIR-to-KIR correspondence.
    pub const fn correspondence(&self) -> &InertCanonicalMirToKirCorrespondenceEvidenceV4 {
        &self.correspondence
    }

    /// Returns the independently decoded exact formal-memory admission.
    pub const fn formal_memory(&self) -> &InertCanonicalFormalMemoryAdmissionEvidenceV4 {
        &self.formal_memory
    }

    /// Returns the exact canonical aggregate evidence and its imported signed receipt.
    pub const fn verus_execution(&self) -> &CanonicalProductionMirPlironVerusExecutionEvidenceV1 {
        &self.verus_execution
    }

    /// Returns every exact semantic-certificate to checked-KIR-addition anchor.
    pub fn semantic_u32_induction_kir_anchors(&self) -> &[VerifiedSemanticU32InductionKirAnchorV1] {
        &self.induction_anchors
    }

    /// Reports that all exact stage bytes and the nested Verus evidence were associated.
    pub const fn has_exact_decoded_input_association(&self) -> bool {
        true
    }

    /// Reports that retained block locators and source statement counts match decoded MIR and KIR.
    pub const fn has_structural_mir_to_kir_correspondence(&self) -> bool {
        true
    }

    /// Reports complete exact operation-span, parameter, and induction-anchor validation.
    pub const fn has_lossless_mir_to_kir_correspondence(&self) -> bool {
        true
    }

    /// Reports that the exact signed receipt was independently imported under its embedded key.
    pub const fn authenticates_signed_verus_receipt_under_embedded_key(&self) -> bool {
        self.verus_execution
            .authenticates_signed_receipt_under_embedded_key()
    }

    /// Reports that protected compiler origin remains a separate required join.
    pub const fn authenticates_compiler_origin(&self) -> bool {
        false
    }

    /// Reports that source-side proof evidence establishes no LLVM or machine refinement.
    pub const fn establishes_llvm_or_machine_refinement(&self) -> bool {
        false
    }

    /// Reports that decoded proof inputs alone grant no load or launch authority.
    pub const fn grants_runtime_authority(&self) -> bool {
        false
    }
}

/// Strictly decodes and cross-checks the compiler proof association and all five input receipts.
pub fn validate_compiler_proof_inputs_v3(
    proof_binding: &InertProofBindingReceiptV3,
    semantic_mir: &InertCanonicalSemanticMirReceiptV3,
    middle_end: &InertMiddleEndReceiptV3,
    kernel_ir: &InertKernelIrReceiptV3,
    mir_to_kir_correspondence: &InertMirToKirCorrespondenceReceiptV3,
    formal_memory: &InertFormalMemoryReceiptV3,
) -> Result<ValidatedCompilerProofInputsV3, CompilerProofInputValidationErrorV3> {
    let association = InertProofBindingAssociationV3::decode(proof_binding.canonical_preimage())
        .map_err(CompilerProofInputValidationErrorV3::ProofBindingDecode)?;
    let inputs = association.inputs();
    for (actual, expected, field) in [
        (
            inputs.semantic_mir(),
            content_identity(
                semantic_mir.identity().sha256(),
                semantic_mir.identity().byte_len(),
            )?,
            "semantic MIR",
        ),
        (
            inputs.middle_end(),
            content_identity(
                middle_end.identity().sha256(),
                middle_end.identity().byte_len(),
            )?,
            "middle end",
        ),
        (
            inputs.kernel_ir(),
            content_identity(
                kernel_ir.identity().sha256(),
                kernel_ir.identity().byte_len(),
            )?,
            "Kernel IR",
        ),
        (
            inputs.mir_to_kir_correspondence(),
            content_identity(
                mir_to_kir_correspondence.identity().sha256(),
                mir_to_kir_correspondence.identity().byte_len(),
            )?,
            "MIR-to-KIR correspondence",
        ),
        (
            inputs.formal_memory(),
            content_identity(
                formal_memory.identity().sha256(),
                formal_memory.identity().byte_len(),
            )?,
            "formal memory",
        ),
    ] {
        if actual != expected {
            return Err(
                CompilerProofInputValidationErrorV3::ProofBindingIdentityMismatch { field },
            );
        }
    }

    let decoded = decode_and_cross_check_stages_v3(
        semantic_mir,
        middle_end,
        kernel_ir,
        mir_to_kir_correspondence,
        formal_memory,
    )?;

    Ok(ValidatedCompilerProofInputsV3 {
        association,
        receipt_identity: proof_binding.identity(),
        semantic_mir: decoded.semantic_mir,
        middle_end: decoded.middle_end,
        kernel_ir: decoded.kernel_ir,
        correspondence: decoded.correspondence,
        formal_memory: decoded.formal_memory,
    })
}

/// Strictly decodes the current proof association, all five stage receipts, and the exact signed
/// aggregate MIR-to-live-PLIRON Verus execution.
pub fn validate_compiler_proof_inputs_v4(
    proof_binding: &InertProofBindingReceiptV3,
    semantic_mir: &InertCanonicalSemanticMirReceiptV3,
    middle_end: &InertMiddleEndReceiptV3,
    kernel_ir: &InertKernelIrReceiptV3,
    mir_to_kir_correspondence: &InertMirToKirCorrespondenceReceiptV3,
    formal_memory: &InertFormalMemoryReceiptV3,
) -> Result<ValidatedCompilerProofInputsV4, CompilerProofInputValidationErrorV4> {
    let association = InertProofBindingAssociationV4::decode(proof_binding.canonical_preimage())
        .map_err(CompilerProofInputValidationErrorV4::ProofBindingDecode)?;
    let inputs = association.inputs();
    for (actual, expected, field) in [
        (
            inputs.semantic_mir(),
            content_identity(
                semantic_mir.identity().sha256(),
                semantic_mir.identity().byte_len(),
            )
            .map_err(CompilerProofInputValidationErrorV4::Stage)?,
            "semantic MIR",
        ),
        (
            inputs.middle_end(),
            content_identity(
                middle_end.identity().sha256(),
                middle_end.identity().byte_len(),
            )
            .map_err(CompilerProofInputValidationErrorV4::Stage)?,
            "middle end",
        ),
        (
            inputs.kernel_ir(),
            content_identity(
                kernel_ir.identity().sha256(),
                kernel_ir.identity().byte_len(),
            )
            .map_err(CompilerProofInputValidationErrorV4::Stage)?,
            "Kernel IR",
        ),
        (
            inputs.mir_to_kir_correspondence(),
            content_identity(
                mir_to_kir_correspondence.identity().sha256(),
                mir_to_kir_correspondence.identity().byte_len(),
            )
            .map_err(CompilerProofInputValidationErrorV4::Stage)?,
            "MIR-to-KIR correspondence",
        ),
        (
            inputs.formal_memory(),
            content_identity(
                formal_memory.identity().sha256(),
                formal_memory.identity().byte_len(),
            )
            .map_err(CompilerProofInputValidationErrorV4::Stage)?,
            "formal memory",
        ),
    ] {
        if actual != expected {
            return Err(
                CompilerProofInputValidationErrorV4::ProofBindingIdentityMismatch { field },
            );
        }
    }
    let decoded = decode_and_cross_check_stages_v4(
        semantic_mir,
        middle_end,
        kernel_ir,
        mir_to_kir_correspondence,
        formal_memory,
    )
    .map_err(CompilerProofInputValidationErrorV4::Stage)?;
    let verus_execution = CanonicalProductionMirPlironVerusExecutionEvidenceV1::decode(
        association.verus_execution_evidence(),
    )
    .map_err(CompilerProofInputValidationErrorV4::VerusEvidence)?;
    if verus_execution
        .claims()
        .pliron_evidence_identity()
        .as_bytes()
        != decoded.middle_end.identity().sha256()
    {
        return Err(CompilerProofInputValidationErrorV4::VerusMiddleEndMismatch);
    }

    Ok(ValidatedCompilerProofInputsV4 {
        association,
        receipt_identity: proof_binding.identity(),
        semantic_mir: decoded.semantic_mir,
        middle_end: decoded.middle_end,
        kernel_ir: decoded.kernel_ir,
        correspondence: decoded.correspondence,
        formal_memory: decoded.formal_memory,
        verus_execution,
        induction_anchors: decoded.induction_anchors,
    })
}

struct DecodedCompilerProofStagesV3 {
    semantic_mir: AdmittedInertSemanticMirV1,
    middle_end: InertProductionMiddleEndEvidenceV5,
    kernel_ir: VerifiedCanonicalKernelIrV5,
    correspondence: InertCanonicalMirToKirCorrespondenceEvidenceV3,
    formal_memory: InertCanonicalFormalMemoryAdmissionEvidenceV3,
}

fn decode_and_cross_check_stages_v3(
    semantic_mir: &InertCanonicalSemanticMirReceiptV3,
    middle_end: &InertMiddleEndReceiptV3,
    kernel_ir: &InertKernelIrReceiptV3,
    mir_to_kir_correspondence: &InertMirToKirCorrespondenceReceiptV3,
    formal_memory: &InertFormalMemoryReceiptV3,
) -> Result<DecodedCompilerProofStagesV3, CompilerProofInputValidationErrorV3> {
    let decoded_semantic_mir = AdmittedInertSemanticMirV1::decode_current_production_canonical(
        semantic_mir.canonical_preimage(),
        SemanticMirLimitsV1::default(),
    )
    .map_err(CompilerProofInputValidationErrorV3::SemanticMirDecode)?;
    let decoded_middle_end =
        InertProductionMiddleEndEvidenceV5::decode(middle_end.canonical_preimage())
            .map_err(CompilerProofInputValidationErrorV3::MiddleEndDecode)?;
    let (decoded_kernel_ir, kernel_module) =
        VerifiedCanonicalKernelIrV5::from_canonical_bytes_with_module(
            kernel_ir.canonical_preimage().to_vec(),
        )
        .map_err(CompilerProofInputValidationErrorV3::KernelIr)?;
    let decoded_correspondence = InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(
        mir_to_kir_correspondence.canonical_preimage(),
    )
    .map_err(CompilerProofInputValidationErrorV3::CorrespondenceDecode)?;
    let decoded_formal_memory =
        InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(formal_memory.canonical_preimage())
            .map_err(CompilerProofInputValidationErrorV3::FormalMemoryDecode)?;

    let semantic_identity = decoded_semantic_mir.semantic_sha256();
    for (actual, field) in [
        (
            decoded_middle_end.source_semantic_identity(),
            "middle-end source semantic MIR",
        ),
        (
            decoded_correspondence.semantic_sha256(),
            "MIR-to-KIR correspondence semantic MIR",
        ),
    ] {
        if actual != semantic_identity.as_bytes() {
            return Err(CompilerProofInputValidationErrorV3::NestedIdentityMismatch { field });
        }
    }
    for (actual, field) in [
        (
            decoded_correspondence.canonical_kir_v5_identity(),
            "MIR-to-KIR correspondence Kernel IR",
        ),
        (
            decoded_formal_memory.canonical_kir_v5_identity(),
            "formal-memory admission Kernel IR",
        ),
    ] {
        if actual != decoded_kernel_ir.identity().digest() {
            return Err(CompilerProofInputValidationErrorV3::NestedIdentityMismatch { field });
        }
    }
    validate_structural_correspondence(
        &decoded_semantic_mir,
        &kernel_module,
        &decoded_correspondence,
    )?;
    Ok(DecodedCompilerProofStagesV3 {
        semantic_mir: decoded_semantic_mir,
        middle_end: decoded_middle_end,
        kernel_ir: decoded_kernel_ir,
        correspondence: decoded_correspondence,
        formal_memory: decoded_formal_memory,
    })
}

struct DecodedCompilerProofStagesV4 {
    semantic_mir: AdmittedInertSemanticMirV1,
    middle_end: InertProductionMiddleEndEvidenceV5,
    kernel_ir: VerifiedCanonicalKernelIrV8,
    correspondence: InertCanonicalMirToKirCorrespondenceEvidenceV4,
    formal_memory: InertCanonicalFormalMemoryAdmissionEvidenceV4,
    induction_anchors: Box<[VerifiedSemanticU32InductionKirAnchorV1]>,
}

fn decode_and_cross_check_stages_v4(
    semantic_mir: &InertCanonicalSemanticMirReceiptV3,
    middle_end: &InertMiddleEndReceiptV3,
    kernel_ir: &InertKernelIrReceiptV3,
    mir_to_kir_correspondence: &InertMirToKirCorrespondenceReceiptV3,
    formal_memory: &InertFormalMemoryReceiptV3,
) -> Result<DecodedCompilerProofStagesV4, CompilerProofInputValidationErrorV3> {
    let decoded_semantic_mir = AdmittedInertSemanticMirV1::decode_current_production_canonical(
        semantic_mir.canonical_preimage(),
        SemanticMirLimitsV1::default(),
    )
    .map_err(CompilerProofInputValidationErrorV3::SemanticMirDecode)?;
    let decoded_middle_end =
        InertProductionMiddleEndEvidenceV5::decode(middle_end.canonical_preimage())
            .map_err(CompilerProofInputValidationErrorV3::MiddleEndDecode)?;
    let (decoded_kernel_ir, kernel_module) =
        VerifiedCanonicalKernelIrV8::from_canonical_bytes_with_module(
            kernel_ir.canonical_preimage().to_vec(),
        )
        .map_err(CompilerProofInputValidationErrorV3::KernelIrV8)?;
    let decoded_correspondence = InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(
        mir_to_kir_correspondence.canonical_preimage(),
    )
    .map_err(CompilerProofInputValidationErrorV3::CorrespondenceV4Decode)?;
    let decoded_formal_memory =
        InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(formal_memory.canonical_preimage())
            .map_err(CompilerProofInputValidationErrorV3::FormalMemoryV4Decode)?;

    let semantic_identity = decoded_semantic_mir.semantic_sha256();
    for (actual, field) in [
        (
            decoded_middle_end.source_semantic_identity(),
            "middle-end source semantic MIR",
        ),
        (
            decoded_correspondence.semantic_sha256(),
            "lossless MIR-to-KIR correspondence semantic MIR",
        ),
        (
            decoded_correspondence
                .semantic_u32_induction()
                .semantic_mir_sha256(),
            "semantic induction report MIR",
        ),
    ] {
        if actual != semantic_identity.as_bytes() {
            return Err(CompilerProofInputValidationErrorV3::NestedIdentityMismatch { field });
        }
    }
    let correspondence_kir = decoded_correspondence.canonical_kernel_ir_identity();
    let formal_kir = decoded_formal_memory.canonical_kernel_ir_identity();
    if correspondence_kir != formal_kir
        || correspondence_kir.version() != ProductionCanonicalKernelIrVersionV1::V8
        || correspondence_kir.digest() != decoded_kernel_ir.identity().digest()
        || correspondence_kir.canonical_length() != decoded_kernel_ir.identity().canonical_length()
    {
        return Err(
            CompilerProofInputValidationErrorV3::NestedIdentityMismatch {
                field: "current production Kernel IR custody",
            },
        );
    }
    let induction_anchors = validate_lossless_correspondence_v4(
        &decoded_semantic_mir,
        &kernel_module,
        &decoded_correspondence,
    )?;
    Ok(DecodedCompilerProofStagesV4 {
        semantic_mir: decoded_semantic_mir,
        middle_end: decoded_middle_end,
        kernel_ir: decoded_kernel_ir,
        correspondence: decoded_correspondence,
        formal_memory: decoded_formal_memory,
        induction_anchors: induction_anchors.into_boxed_slice(),
    })
}

fn content_identity(
    sha256: &[u8; 32],
    byte_len: u64,
) -> Result<InertLineageContentIdentityV3, CompilerProofInputValidationErrorV3> {
    InertLineageContentIdentityV3::new(*sha256, byte_len)
        .map_err(CompilerProofInputValidationErrorV3::ProofBindingDecode)
}

fn validate_structural_correspondence(
    semantic_mir: &AdmittedInertSemanticMirV1,
    kernel_ir: &Module,
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV3,
) -> Result<(), CompilerProofInputValidationErrorV3> {
    let mut defined_kernel_functions = kernel_ir
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref());
    let records = correspondence.blocks();
    let mut record_offset = 0_usize;
    let mut covered_functions = 0_usize;
    while let Some(first) = records.get(record_offset) {
        let semantic_function_index = usize::try_from(first.semantic_function()).map_err(|_| {
            CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                detail: "semantic function locator does not fit this host",
            }
        })?;
        let group_start = record_offset;
        while records
            .get(record_offset)
            .is_some_and(|record| record.semantic_function() == first.semantic_function())
        {
            record_offset += 1;
        }
        let function_records = &records[group_start..record_offset];
        let semantic_function = semantic_mir
            .functions()
            .get(semantic_function_index)
            .ok_or(
                CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                    detail: "correspondence names an absent semantic function",
                },
            )?;
        let kernel_body = defined_kernel_functions.next().ok_or(
            CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                detail: "defined Kernel IR function coverage differs from correspondence records",
            },
        )?;
        if semantic_function.blocks().len() != function_records.len() {
            return Err(
                CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                    detail: "semantic block coverage differs from correspondence records",
                },
            );
        }
        if kernel_body.blocks.len() != function_records.len() {
            return Err(
                CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                    detail: "Kernel IR block coverage differs from correspondence records",
                },
            );
        }
        for record in function_records {
            let semantic_block_index = usize::try_from(record.semantic_block()).map_err(|_| {
                CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                    detail: "semantic block locator does not fit this host",
                }
            })?;
            let semantic_block = semantic_function.blocks().get(semantic_block_index).ok_or(
                CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                    detail: "correspondence names an absent semantic block",
                },
            )?;
            if usize::try_from(record.source_statement_count())
                != Ok(semantic_block.statements().len())
            {
                return Err(
                    CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                        detail: "correspondence source statement count differs from semantic MIR",
                    },
                );
            }
            if !kernel_body
                .blocks
                .iter()
                .any(|block| block.id.0 == record.kernel_ir_block())
            {
                return Err(
                    CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                        detail: "correspondence names an absent Kernel IR block",
                    },
                );
            }
        }
        covered_functions += 1;
    }
    if usize::try_from(correspondence.function_count()) != Ok(covered_functions) {
        return Err(
            CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                detail: "declared function coverage differs from correspondence records",
            },
        );
    }
    if defined_kernel_functions.next().is_some() {
        return Err(
            CompilerProofInputValidationErrorV3::StructuralCorrespondence {
                detail: "defined Kernel IR function coverage differs from correspondence records",
            },
        );
    }
    Ok(())
}

fn validate_lossless_correspondence_v4(
    semantic_mir: &AdmittedInertSemanticMirV1,
    kernel_ir: &Module,
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
) -> Result<Vec<VerifiedSemanticU32InductionKirAnchorV1>, CompilerProofInputValidationErrorV3> {
    let blocks = correspondence.blocks();
    let mut defined_kernel_functions = kernel_ir
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref());
    let mut semantic_functions = BTreeMap::<u32, &SemanticFunctionDeclV1>::new();
    let mut kernel_bodies = BTreeMap::<u32, &FunctionBody>::new();
    let mut kernel_blocks = BTreeMap::<(u32, u32), &BasicBlock>::new();
    let mut semantic_to_kir = BTreeMap::<(u32, u32), u32>::new();
    let mut record_offset = 0_usize;
    let mut covered_functions = 0_usize;

    while let Some(first) = blocks.get(record_offset) {
        let semantic_function_index = usize::try_from(first.semantic_function())
            .map_err(|_| structural_v4("semantic function locator does not fit this host"))?;
        let group_start = record_offset;
        while blocks
            .get(record_offset)
            .is_some_and(|record| record.semantic_function() == first.semantic_function())
        {
            record_offset += 1;
        }
        let function_records = &blocks[group_start..record_offset];
        let semantic_function = semantic_mir
            .functions()
            .get(semantic_function_index)
            .ok_or_else(|| structural_v4("correspondence names an absent semantic function"))?;
        let kernel_body = defined_kernel_functions.next().ok_or_else(|| {
            structural_v4("defined KIR function coverage differs from correspondence records")
        })?;
        if semantic_function.blocks().len() != function_records.len() {
            return Err(structural_v4(
                "semantic block coverage differs from correspondence records",
            ));
        }
        if semantic_functions
            .insert(first.semantic_function(), semantic_function)
            .is_some()
            || kernel_bodies
                .insert(first.semantic_function(), kernel_body)
                .is_some()
        {
            return Err(structural_v4("semantic function coverage is not unique"));
        }

        let mut body_block_ids = BTreeSet::new();
        for block in &kernel_body.blocks {
            if !body_block_ids.insert(block.id.0)
                || kernel_blocks
                    .insert((first.semantic_function(), block.id.0), block)
                    .is_some()
            {
                return Err(structural_v4("KIR block identities are not unique"));
            }
        }
        for record in function_records {
            let semantic_block_index = usize::try_from(record.semantic_block())
                .map_err(|_| structural_v4("semantic block locator does not fit this host"))?;
            let semantic_block = semantic_function
                .blocks()
                .get(semantic_block_index)
                .ok_or_else(|| structural_v4("correspondence names an absent semantic block"))?;
            if usize::try_from(record.source_statement_count())
                != Ok(semantic_block.statements().len())
            {
                return Err(structural_v4(
                    "correspondence source statement count differs from semantic MIR",
                ));
            }
            if !body_block_ids.contains(&record.kernel_ir_block()) {
                return Err(structural_v4("correspondence names an absent KIR block"));
            }
            if semantic_to_kir
                .insert(
                    (record.semantic_function(), record.semantic_block()),
                    record.kernel_ir_block(),
                )
                .is_some()
            {
                return Err(structural_v4("semantic block correspondence is not unique"));
            }
        }
        covered_functions = covered_functions
            .checked_add(1)
            .ok_or_else(|| structural_v4("covered function count overflows"))?;
    }
    if usize::try_from(correspondence.function_count()) != Ok(covered_functions)
        || defined_kernel_functions.next().is_some()
    {
        return Err(structural_v4(
            "defined KIR function coverage differs from correspondence records",
        ));
    }

    validate_exact_operation_spans_v4(
        correspondence,
        &semantic_functions,
        &kernel_bodies,
        &kernel_blocks,
        &semantic_to_kir,
    )?;
    validate_parameter_bindings_v4(correspondence, &semantic_functions, &kernel_bodies)?;
    validate_induction_replay_and_anchors_v4(
        semantic_mir,
        correspondence,
        &kernel_blocks,
        &semantic_to_kir,
    )
}

fn validate_exact_operation_spans_v4(
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
    semantic_functions: &BTreeMap<u32, &SemanticFunctionDeclV1>,
    kernel_bodies: &BTreeMap<u32, &FunctionBody>,
    kernel_blocks: &BTreeMap<(u32, u32), &BasicBlock>,
    semantic_to_kir: &BTreeMap<(u32, u32), u32>,
) -> Result<(), CompilerProofInputValidationErrorV3> {
    let expected_statement_count =
        semantic_functions
            .values()
            .try_fold(0_usize, |total, function| {
                function.blocks().iter().try_fold(total, |total, block| {
                    total.checked_add(block.statements().len())
                })
            });
    let expected_statement_count = expected_statement_count
        .ok_or_else(|| structural_v4("semantic statement count overflows"))?;
    let expected_block_count = semantic_functions
        .values()
        .try_fold(0_usize, |total, function| {
            total.checked_add(function.blocks().len())
        });
    let expected_block_count =
        expected_block_count.ok_or_else(|| structural_v4("semantic block count overflows"))?;
    if correspondence.statement_spans().len() != expected_statement_count
        || correspondence.terminator_spans().len() != expected_block_count
    {
        return Err(structural_v4(
            "exact semantic statement or terminator span coverage differs",
        ));
    }

    let synthetic_function = if correspondence.synthetic_spans().is_empty() {
        None
    } else {
        if semantic_functions.len() != 1 {
            return Err(structural_v4(
                "synthetic spans do not have unique function ownership",
            ));
        }
        semantic_functions.keys().next().copied()
    };
    let mut synthetics_by_block = BTreeMap::new();
    for span in correspondence.synthetic_spans() {
        let function = synthetic_function
            .ok_or_else(|| structural_v4("synthetic span function ownership is absent"))?;
        synthetics_by_block
            .entry((function, span.kernel_ir_block()))
            .or_insert_with(Vec::new)
            .push(span);
    }
    let mut runtime_assert_blocks = 0_usize;

    for (&function_index, semantic_function) in semantic_functions {
        let body = kernel_bodies
            .get(&function_index)
            .ok_or_else(|| structural_v4("corresponding KIR function body is absent"))?;
        let mut mapped_kir_blocks = BTreeSet::new();
        for (semantic_block_index, semantic_block) in semantic_function.blocks().iter().enumerate()
        {
            let semantic_block_index = u32::try_from(semantic_block_index)
                .map_err(|_| structural_v4("semantic block index does not fit correspondence"))?;
            let kir_block_id = *semantic_to_kir
                .get(&(function_index, semantic_block_index))
                .ok_or_else(|| structural_v4("semantic block has no KIR correspondence"))?;
            mapped_kir_blocks.insert(kir_block_id);
            let target = *kernel_blocks
                .get(&(function_index, kir_block_id))
                .ok_or_else(|| structural_v4("corresponding KIR block is absent"))?;
            let mut next_operation = 0_usize;

            if let Some(synthetic) = synthetics_by_block.remove(&(function_index, kir_block_id)) {
                if synthetic.len() != 1
                    || synthetic[0].rule() != MirToKirSyntheticRuleEvidenceV4::EnumPayloadStorage
                    || synthetic[0].first_operation() != 0
                    || synthetic[0].operation_count() == 0
                {
                    return Err(structural_v4(
                        "mapped KIR block has invalid synthetic prologue coverage",
                    ));
                }
                next_operation = checked_span_end_v4(
                    synthetic[0].first_operation(),
                    synthetic[0].operation_count(),
                    target.operations.len(),
                )?;
                if !target.operations[..next_operation].iter().all(|operation| {
                    matches!(
                        operation.kind,
                        OperationKind::Alloca {
                            address_space: AddressSpace::Private,
                            ..
                        } | OperationKind::Load {
                            access: MemoryAccess {
                                address_space: AddressSpace::Private,
                                ..
                            },
                            ..
                        }
                    )
                }) {
                    return Err(structural_v4(
                        "enum-payload synthetic span contains a non-private-storage operation",
                    ));
                }
            }

            for statement in 0..semantic_block.statements().len() {
                let statement = u32::try_from(statement).map_err(|_| {
                    structural_v4("semantic statement index does not fit correspondence")
                })?;
                let span = find_statement_span_v4(
                    correspondence,
                    function_index,
                    semantic_block_index,
                    statement,
                )?;
                if span.kernel_ir_block() != kir_block_id
                    || usize::try_from(span.first_operation()) != Ok(next_operation)
                {
                    return Err(structural_v4(
                        "semantic statement span is not contiguous in its exact KIR block",
                    ));
                }
                next_operation = checked_span_end_v4(
                    span.first_operation(),
                    span.operation_count(),
                    target.operations.len(),
                )?;
            }
            let terminator =
                find_terminator_span_v4(correspondence, function_index, semantic_block_index)?;
            if terminator.kernel_ir_block() != kir_block_id
                || usize::try_from(terminator.first_operation()) != Ok(next_operation)
            {
                return Err(structural_v4(
                    "semantic terminator span is not contiguous in its exact KIR block",
                ));
            }
            next_operation = checked_span_end_v4(
                terminator.first_operation(),
                terminator.operation_count(),
                target.operations.len(),
            )?;
            if next_operation != target.operations.len() || target.terminator.is_none() {
                return Err(structural_v4(
                    "semantic spans do not cover the complete mapped KIR block",
                ));
            }
        }

        for target in &body.blocks {
            if mapped_kir_blocks.contains(&target.id.0) {
                continue;
            }
            let synthetic = synthetics_by_block
                .remove(&(function_index, target.id.0))
                .ok_or_else(|| structural_v4("unmapped KIR block has no synthetic custody"))?;
            if synthetic.len() != 1
                || synthetic[0].rule() != MirToKirSyntheticRuleEvidenceV4::RuntimeAssertFailureTrap
                || synthetic[0].first_operation() != 0
                || synthetic[0].operation_count() != 1
                || target.operations.as_slice() != [AmdGpuDiagnosticOperation::Trap.operation(None)]
                || !matches!(target.terminator, Some(Terminator::Unreachable))
            {
                return Err(structural_v4(
                    "unmapped KIR block is not the canonical runtime-assert trap",
                ));
            }
            runtime_assert_blocks = runtime_assert_blocks
                .checked_add(1)
                .ok_or_else(|| structural_v4("runtime-assert block count overflows"))?;
        }
    }
    if !synthetics_by_block.is_empty() || runtime_assert_blocks > 1 {
        return Err(structural_v4(
            "synthetic span coverage is incomplete or noncanonical",
        ));
    }
    Ok(())
}

fn validate_parameter_bindings_v4(
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
    semantic_functions: &BTreeMap<u32, &SemanticFunctionDeclV1>,
    kernel_bodies: &BTreeMap<u32, &FunctionBody>,
) -> Result<(), CompilerProofInputValidationErrorV3> {
    let mut expected_bindings = 0_usize;
    for (&function_index, semantic_function) in semantic_functions {
        let body = kernel_bodies
            .get(&function_index)
            .ok_or_else(|| structural_v4("corresponding KIR function body is absent"))?;
        let mut arguments = semantic_function
            .locals()
            .iter()
            .enumerate()
            .filter_map(|(local, declaration)| match declaration.role() {
                SemanticLocalRoleV1::Argument(argument) => Some((argument, local)),
                SemanticLocalRoleV1::Return | SemanticLocalRoleV1::Temporary => None,
            })
            .collect::<Vec<_>>();
        arguments.sort_unstable();
        if arguments.len() != body.parameters.len()
            || arguments
                .iter()
                .enumerate()
                .any(|(expected, (actual, _))| usize::try_from(*actual) != Ok(expected))
        {
            return Err(structural_v4(
                "semantic argument roster differs from KIR parameters",
            ));
        }
        for (argument, (_, local)) in arguments.iter().enumerate() {
            let local = u32::try_from(*local).map_err(|_| {
                structural_v4("semantic argument local does not fit correspondence")
            })?;
            let binding = find_parameter_binding_v4(correspondence, function_index, local)?;
            if binding.kernel_ir_value() != body.parameters[argument].0 {
                return Err(structural_v4(
                    "semantic argument binding names a different KIR parameter",
                ));
            }
        }
        expected_bindings = expected_bindings
            .checked_add(arguments.len())
            .ok_or_else(|| structural_v4("parameter binding count overflows"))?;
    }
    if correspondence.parameter_bindings().len() != expected_bindings {
        return Err(structural_v4(
            "parameter binding coverage differs from semantic arguments",
        ));
    }
    Ok(())
}

fn validate_induction_replay_and_anchors_v4(
    semantic_mir: &AdmittedInertSemanticMirV1,
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
    kernel_blocks: &BTreeMap<(u32, u32), &BasicBlock>,
    semantic_to_kir: &BTreeMap<(u32, u32), u32>,
) -> Result<Vec<VerifiedSemanticU32InductionKirAnchorV1>, CompilerProofInputValidationErrorV3> {
    let retained = correspondence.semantic_u32_induction();
    let function = SemanticFunctionIdV1::from_index(retained.function());
    let replay = analyze_semantic_u32_induction_no_overflow_v1(semantic_mir, function)
        .map_err(CompilerProofInputValidationErrorV3::SemanticInductionAnalysis)?;
    let canonical_replay = InertCanonicalSemanticU32InductionEvidenceV1::from_report(&replay)
        .map_err(CompilerProofInputValidationErrorV3::SemanticInductionEvidence)?;
    if canonical_replay.canonical_bytes() != retained.canonical_bytes() {
        return Err(structural_v4(
            "retained semantic induction report differs from deterministic replay",
        ));
    }

    let semantic_function = semantic_mir
        .functions()
        .get(retained.function() as usize)
        .ok_or_else(|| structural_v4("semantic induction function is absent"))?;
    let mut anchors = Vec::new();
    anchors
        .try_reserve_exact(retained.certificates().len())
        .map_err(|_| structural_v4("semantic induction anchor allocation failed"))?;
    for certificate in retained.certificates() {
        let site = certificate.checked_addition();
        let semantic_block = site.block().block();
        let semantic_statement = site.statement();
        let statement = semantic_function
            .blocks()
            .get(semantic_block as usize)
            .and_then(|block| block.statements().get(semantic_statement as usize))
            .ok_or_else(|| structural_v4("semantic induction statement site is absent"))?;
        if !matches!(
            statement.kind(),
            SemanticStatementKindV1::Assign(assignment)
                if matches!(
                    assignment.value().kind(),
                    SemanticRvalueKindV1::CheckedBinary(checked)
                        if checked.operation() == SemanticCheckedBinaryOpV1::Add
                )
        ) {
            return Err(structural_v4(
                "semantic induction certificate does not name a checked addition",
            ));
        }
        let span = find_statement_span_v4(
            correspondence,
            retained.function(),
            semantic_block,
            semantic_statement,
        )?;
        let kir_block_id = *semantic_to_kir
            .get(&(retained.function(), semantic_block))
            .ok_or_else(|| structural_v4("induction statement has no KIR block mapping"))?;
        if span.kernel_ir_block() != kir_block_id {
            return Err(structural_v4(
                "induction statement span names a different KIR block",
            ));
        }
        let kir_block = *kernel_blocks
            .get(&(retained.function(), kir_block_id))
            .ok_or_else(|| structural_v4("induction KIR block is absent"))?;
        let end = checked_span_end_v4(
            span.first_operation(),
            span.operation_count(),
            kir_block.operations.len(),
        )?;
        let first = usize::try_from(span.first_operation())
            .map_err(|_| structural_v4("induction KIR operation ordinal does not fit this host"))?;
        let mut checked_add = None;
        for (relative, operation) in kir_block.operations[first..end].iter().enumerate() {
            if !matches!(
                operation.kind,
                OperationKind::Binary {
                    op: BinaryOp::Checked(CheckedBinaryOperator::Add),
                    ..
                }
            ) {
                continue;
            }
            if checked_add.is_some() {
                return Err(structural_v4(
                    "induction statement span contains multiple checked KIR additions",
                ));
            }
            let [value, overflow] = operation.results.as_slice() else {
                return Err(structural_v4(
                    "checked KIR addition does not have exact value and overflow results",
                ));
            };
            let operation = first
                .checked_add(relative)
                .and_then(|ordinal| u32::try_from(ordinal).ok())
                .ok_or_else(|| structural_v4("induction KIR operation ordinal overflows"))?;
            checked_add = Some((operation, value.id.0, overflow.id.0));
        }
        let (kernel_ir_operation, value_result, overflow_result) =
            checked_add.ok_or_else(|| {
                structural_v4("induction statement span contains no checked KIR addition")
            })?;
        anchors.push(VerifiedSemanticU32InductionKirAnchorV1 {
            semantic_function: retained.function(),
            semantic_block,
            semantic_statement,
            kernel_ir_block: kir_block_id,
            kernel_ir_operation,
            value_result,
            overflow_result,
        });
    }
    Ok(anchors)
}

fn find_statement_span_v4(
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
    function: u32,
    block: u32,
    statement: u32,
) -> Result<
    &fe2o3_lower_mir_kernel::MirToKirStatementSpanEvidenceV4,
    CompilerProofInputValidationErrorV3,
> {
    correspondence
        .statement_spans()
        .binary_search_by_key(&(function, block, statement), |span| {
            (
                span.semantic_function(),
                span.semantic_block(),
                span.statement(),
            )
        })
        .ok()
        .and_then(|index| correspondence.statement_spans().get(index))
        .ok_or_else(|| structural_v4("semantic statement has no exact KIR operation span"))
}

fn find_terminator_span_v4(
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
    function: u32,
    block: u32,
) -> Result<
    &fe2o3_lower_mir_kernel::MirToKirTerminatorSpanEvidenceV4,
    CompilerProofInputValidationErrorV3,
> {
    correspondence
        .terminator_spans()
        .binary_search_by_key(&(function, block), |span| {
            (span.semantic_function(), span.semantic_block())
        })
        .ok()
        .and_then(|index| correspondence.terminator_spans().get(index))
        .ok_or_else(|| structural_v4("semantic terminator has no exact KIR operation span"))
}

fn find_parameter_binding_v4(
    correspondence: &InertCanonicalMirToKirCorrespondenceEvidenceV4,
    function: u32,
    local: u32,
) -> Result<
    &fe2o3_lower_mir_kernel::MirToKirParameterBindingEvidenceV4,
    CompilerProofInputValidationErrorV3,
> {
    correspondence
        .parameter_bindings()
        .binary_search_by_key(&(function, local), |binding| {
            (binding.semantic_function(), binding.semantic_local())
        })
        .ok()
        .and_then(|index| correspondence.parameter_bindings().get(index))
        .ok_or_else(|| structural_v4("semantic argument has no exact KIR parameter binding"))
}

fn checked_span_end_v4(
    first: u32,
    count: u32,
    operation_len: usize,
) -> Result<usize, CompilerProofInputValidationErrorV3> {
    let first = usize::try_from(first)
        .map_err(|_| structural_v4("KIR operation span does not fit this host"))?;
    let count = usize::try_from(count)
        .map_err(|_| structural_v4("KIR operation span does not fit this host"))?;
    first
        .checked_add(count)
        .filter(|end| *end <= operation_len)
        .ok_or_else(|| structural_v4("KIR operation span is outside its exact block"))
}

const fn structural_v4(detail: &'static str) -> CompilerProofInputValidationErrorV3 {
    CompilerProofInputValidationErrorV3::StructuralCorrespondence { detail }
}

/// Failure to decode or exactly match compiler proof inputs.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompilerProofInputValidationErrorV3 {
    /// The proof-binding preimage is not the exact canonical frozen format.
    ProofBindingDecode(InertProofBindingAssociationErrorV3),
    /// A named association input differs from its independently retained receipt.
    ProofBindingIdentityMismatch {
        /// The mismatched semantic stage.
        field: &'static str,
    },
    /// Exact production semantic MIR could not be decoded and admitted.
    SemanticMirDecode(SemanticMirDecodeErrorV1),
    /// Exact V5 middle-end evidence could not be decoded.
    MiddleEndDecode(ProductionMiddleEndEvidenceCodecErrorV5),
    /// Exact canonical Kernel IR V5 could not be decoded or semantically verified.
    KernelIr(VerifiedCanonicalKernelIrErrorV5),
    /// Exact current canonical Kernel IR V8 could not be decoded or semantically verified.
    KernelIrV8(VerifiedCanonicalKernelIrErrorV8),
    /// Exact MIR-to-KIR correspondence evidence could not be decoded.
    CorrespondenceDecode(ProductionLineageEvidenceErrorV3),
    /// Exact lossless MIR-to-KIR aggregate evidence could not be decoded.
    CorrespondenceV4Decode(ProductionCorrespondenceEvidenceErrorV4),
    /// Exact formal-memory admission evidence could not be decoded.
    FormalMemoryDecode(ProductionLineageEvidenceErrorV3),
    /// Exact current formal-memory admission evidence could not be decoded.
    FormalMemoryV4Decode(ProductionFormalMemoryEvidenceErrorV4),
    /// Deterministic semantic induction replay failed on decoded MIR.
    SemanticInductionAnalysis(SemanticU32InductionAnalysisErrorV1),
    /// Replayed semantic induction evidence could not be canonicalized.
    SemanticInductionEvidence(SemanticU32InductionEvidenceErrorV1),
    /// A nested semantic or Kernel IR identity differs from its decoded owner.
    NestedIdentityMismatch {
        /// The mismatched nested identity.
        field: &'static str,
    },
    /// Structurally decoded correspondence does not match the decoded MIR and KIR.
    StructuralCorrespondence {
        /// Stable fail-closed mismatch description.
        detail: &'static str,
    },
}

impl fmt::Display for CompilerProofInputValidationErrorV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProofBindingDecode(error) => {
                write!(formatter, "cannot decode compiler proof binding: {error}")
            }
            Self::ProofBindingIdentityMismatch { field } => {
                write!(
                    formatter,
                    "compiler proof binding has substituted {field} identity"
                )
            }
            Self::SemanticMirDecode(error) => {
                write!(formatter, "cannot decode compiler semantic MIR: {error}")
            }
            Self::MiddleEndDecode(error) => {
                write!(
                    formatter,
                    "cannot decode compiler middle-end evidence: {error}"
                )
            }
            Self::KernelIr(error) => {
                write!(formatter, "cannot validate compiler Kernel IR: {error}")
            }
            Self::KernelIrV8(error) => {
                write!(
                    formatter,
                    "cannot validate current compiler Kernel IR: {error}"
                )
            }
            Self::CorrespondenceDecode(error) => {
                write!(
                    formatter,
                    "cannot decode compiler MIR-to-KIR evidence: {error}"
                )
            }
            Self::CorrespondenceV4Decode(error) => write!(
                formatter,
                "cannot decode lossless compiler MIR-to-KIR evidence: {error}"
            ),
            Self::FormalMemoryDecode(error) => {
                write!(
                    formatter,
                    "cannot decode compiler formal-memory evidence: {error}"
                )
            }
            Self::FormalMemoryV4Decode(error) => write!(
                formatter,
                "cannot decode current compiler formal-memory evidence: {error}"
            ),
            Self::SemanticInductionAnalysis(error) => write!(
                formatter,
                "cannot replay semantic induction analysis: {error}"
            ),
            Self::SemanticInductionEvidence(error) => write!(
                formatter,
                "cannot canonicalize replayed semantic induction evidence: {error}"
            ),
            Self::NestedIdentityMismatch { field } => {
                write!(
                    formatter,
                    "compiler proof inputs have substituted {field} identity"
                )
            }
            Self::StructuralCorrespondence { detail } => {
                write!(
                    formatter,
                    "compiler proof inputs have invalid structural correspondence: {detail}"
                )
            }
        }
    }
}

impl Error for CompilerProofInputValidationErrorV3 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ProofBindingDecode(error) => Some(error),
            Self::SemanticMirDecode(error) => Some(error),
            Self::MiddleEndDecode(error) => Some(error),
            Self::KernelIr(error) => Some(error),
            Self::KernelIrV8(error) => Some(error),
            Self::CorrespondenceDecode(error) | Self::FormalMemoryDecode(error) => Some(error),
            Self::CorrespondenceV4Decode(error) => Some(error),
            Self::FormalMemoryV4Decode(error) => Some(error),
            Self::SemanticInductionAnalysis(error) => Some(error),
            Self::SemanticInductionEvidence(error) => Some(error),
            Self::ProofBindingIdentityMismatch { .. }
            | Self::NestedIdentityMismatch { .. }
            | Self::StructuralCorrespondence { .. } => None,
        }
    }
}

/// Failure to decode or exactly match the current signed compiler proof inputs.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompilerProofInputValidationErrorV4 {
    /// The proof-binding preimage is not the exact canonical V4 format.
    ProofBindingDecode(InertProofBindingAssociationErrorV4),
    /// A named V4 association input differs from its independently retained receipt.
    ProofBindingIdentityMismatch {
        /// The mismatched semantic stage.
        field: &'static str,
    },
    /// One of the five shared compiler stages failed strict decoding or cross-checking.
    Stage(CompilerProofInputValidationErrorV3),
    /// The nested aggregate Verus execution failed canonical decode or signed receipt import.
    VerusEvidence(ProductionMirPlironVerusExecutionEvidenceErrorV1),
    /// The signed aggregate receipt names a different live PLIRON middle-end record.
    VerusMiddleEndMismatch,
}

impl fmt::Display for CompilerProofInputValidationErrorV4 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProofBindingDecode(error) => {
                write!(
                    formatter,
                    "cannot decode current compiler proof binding: {error}"
                )
            }
            Self::ProofBindingIdentityMismatch { field } => write!(
                formatter,
                "current compiler proof binding has substituted {field} identity"
            ),
            Self::Stage(error) => write!(formatter, "current compiler proof stage failed: {error}"),
            Self::VerusEvidence(error) => write!(
                formatter,
                "cannot validate current aggregate Verus execution: {error}"
            ),
            Self::VerusMiddleEndMismatch => formatter.write_str(
                "current aggregate Verus execution names a different middle-end PLIRON record",
            ),
        }
    }
}

impl Error for CompilerProofInputValidationErrorV4 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ProofBindingDecode(error) => Some(error),
            Self::Stage(error) => Some(error),
            Self::VerusEvidence(error) => Some(error),
            Self::ProofBindingIdentityMismatch { .. } | Self::VerusMiddleEndMismatch => None,
        }
    }
}

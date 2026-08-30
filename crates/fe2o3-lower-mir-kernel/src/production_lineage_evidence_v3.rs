//! Inert, bounded V3 lineage evidence derived from live production owners.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use fe2o3_kernel_ir::{
    FORMAL_MEMORY_OBLIGATION_POLICY_V1, FORMAL_MEMORY_OBLIGATION_RECEIPT_VERSION_V1,
    FormalMemoryReceiptErrorV1, InertCanonicalFormalMemoryObligationReceiptV1,
    VerifiedCanonicalKernelIrErrorV5, VerifiedCanonicalKernelIrV5,
};
use sha2::{Digest, Sha256};

use crate::{
    ProductionFormalMemoryErrorV1, ProductionFormalMemoryOwnerV1, ProductionSemanticKirErrorV1,
    ProductionSemanticKirOwnerV1,
};

/// Wire version for exact MIR-to-KIR correspondence lineage evidence.
pub const MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_VERSION_V3: u16 = 3;
/// Validation policy committed by MIR-to-KIR correspondence V3 evidence.
pub const MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_POLICY_V3: u16 = 1;
/// Maximum exact bytes in one correspondence evidence record.
pub const MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V3: usize = 512 * 1024;
/// Maximum semantic functions represented by correspondence evidence.
pub const MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3: usize = 1_024;
/// Maximum semantic blocks represented by correspondence evidence.
pub const MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3: usize = 16_384;

/// Wire version for exact formal-memory admission lineage evidence.
pub const FORMAL_MEMORY_ADMISSION_EVIDENCE_VERSION_V3: u16 = 3;
/// Validation policy committed by formal-memory admission V3 evidence.
pub const FORMAL_MEMORY_ADMISSION_EVIDENCE_POLICY_V3: u16 = 1;
/// Maximum exact bytes in one formal-memory admission evidence record.
///
/// This matches the V3 compiler-lineage receipt-preimage budget. The embedded
/// canonical formal-obligation receipt must fit inside this stricter envelope.
pub const MAX_FORMAL_MEMORY_ADMISSION_EVIDENCE_BYTES_V3: usize = 4 * 1024 * 1024;

const CORRESPONDENCE_MAGIC_V3: [u8; 8] = *b"FE2O3MC\0";
const FORMAL_MEMORY_MAGIC_V3: [u8; 8] = *b"FE2O3FA\0";
const FORMAL_OBLIGATION_RECEIPT_MAGIC_V1: [u8; 8] = *b"FE2O3FM\0";
const CORRESPONDENCE_IDENTITY_DOMAIN_V3: &[u8] =
    b"FE2O3/INERT-MIR-TO-KIR-CORRESPONDENCE-EVIDENCE/V3\0";
const FORMAL_MEMORY_IDENTITY_DOMAIN_V3: &[u8] =
    b"FE2O3/INERT-FORMAL-MEMORY-ADMISSION-EVIDENCE/V3\0";
const COMMON_HEADER_BYTES_V3: usize = 20;
const CORRESPONDENCE_HEADER_BYTES_V3: usize = COMMON_HEADER_BYTES_V3 + 32 + 32 + 4 + 4;
const CORRESPONDENCE_RECORD_BYTES_V3: usize = 16;
const FORMAL_MEMORY_HEADER_BYTES_V3: usize =
    COMMON_HEADER_BYTES_V3 + 32 + 32 + 8 + 2 + 2 + 4 + 4 + 4;

/// Exact completeness policy committed by formal-memory admission evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum FormalMemoryCompletenessPolicyV3 {
    /// Require complete extraction and reject every unresolved static or
    /// inter-invocation conflict before evidence construction.
    RequireCompleteConflictFree = 1,
}

/// Exact completeness status committed by formal-memory admission evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum FormalMemoryCompletenessStatusV3 {
    /// The live owner re-derived complete, conflict-free obligations.
    Complete = 1,
}

/// Identifies one exact canonical MIR-to-KIR correspondence evidence encoding.
///
/// This is a content identity only. It grants no producer, artifact, proof, or
/// launch authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MirToKirCorrespondenceEvidenceIdentityV3([u8; 32]);

impl MirToKirCorrespondenceEvidenceIdentityV3 {
    /// Returns the exact content digest.
    pub const fn digest(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Identifies one exact canonical formal-memory admission evidence encoding.
///
/// This is a content identity only. It grants no producer, artifact, proof, or
/// launch authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FormalMemoryAdmissionEvidenceIdentityV3([u8; 32]);

impl FormalMemoryAdmissionEvidenceIdentityV3 {
    /// Returns the exact content digest.
    pub const fn digest(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Canonically ordered correspondence for one semantic block.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MirToKirBlockCorrespondenceEvidenceV3 {
    semantic_function: u32,
    semantic_block: u32,
    kernel_ir_block: u32,
    source_statement_count: u32,
}

impl MirToKirBlockCorrespondenceEvidenceV3 {
    /// Returns the semantic function index.
    pub const fn semantic_function(&self) -> u32 {
        self.semantic_function
    }

    /// Returns the semantic block index.
    pub const fn semantic_block(&self) -> u32 {
        self.semantic_block
    }

    /// Returns the exact corresponding Kernel IR block index.
    pub const fn kernel_ir_block(&self) -> u32 {
        self.kernel_ir_block
    }

    /// Returns the source statement count covered by this rule.
    pub const fn source_statement_count(&self) -> u32 {
        self.source_statement_count
    }
}

/// Exact inert correspondence content derived from a live semantic-KIR owner.
///
/// Decoding proves canonical structure and content integrity only. Any caller
/// can decode or copy these bytes; this value intentionally carries no
/// authority.
#[derive(Debug, Eq, PartialEq)]
pub struct InertCanonicalMirToKirCorrespondenceEvidenceV3 {
    canonical_bytes: Vec<u8>,
    identity: MirToKirCorrespondenceEvidenceIdentityV3,
    semantic_sha256: [u8; 32],
    canonical_kir_v5_identity: [u8; 32],
    function_count: u32,
    blocks: Box<[MirToKirBlockCorrespondenceEvidenceV3]>,
}

impl InertCanonicalMirToKirCorrespondenceEvidenceV3 {
    /// Revalidates a live owner, canonicalizes its exact Kernel IR as V5, and
    /// constructs strict correspondence evidence.
    pub fn from_live_owner(
        owner: &ProductionSemanticKirOwnerV1,
    ) -> Result<Self, ProductionLineageEvidenceErrorV3> {
        owner
            .verify_equivalence()
            .map_err(ProductionLineageEvidenceErrorV3::SemanticKir)?;
        let (function_count, blocks) = exact_correspondence_from_owner(owner)?;
        let canonical_kir = VerifiedCanonicalKernelIrV5::from_module(owner.module().clone())
            .map_err(ProductionLineageEvidenceErrorV3::CanonicalKernelIr)?;
        canonical_kir
            .revalidate()
            .map_err(ProductionLineageEvidenceErrorV3::CanonicalKernelIr)?;

        let semantic_sha256 = *owner.semantic().semantic().semantic_sha256().as_bytes();
        let canonical_kir_v5_identity = *canonical_kir.identity().digest();
        require_nonzero_identity("semantic MIR SHA-256", &semantic_sha256)?;
        require_nonzero_identity(
            "canonical Kernel IR V5 identity",
            &canonical_kir_v5_identity,
        )?;

        owner
            .verify_equivalence()
            .map_err(ProductionLineageEvidenceErrorV3::SemanticKir)?;
        let canonical_bytes = encode_correspondence(
            semantic_sha256,
            canonical_kir_v5_identity,
            function_count,
            &blocks,
        )?;
        Self::decode(&canonical_bytes)
    }

    /// Strictly decodes one complete canonical V3 correspondence encoding.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProductionLineageEvidenceErrorV3> {
        preflight_total_bytes(
            EvidenceKindV3::MirToKirCorrespondence,
            bytes.len(),
            MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V3,
        )?;
        let mut reader = ReaderV3::new(bytes);
        decode_common_header(
            &mut reader,
            EvidenceKindV3::MirToKirCorrespondence,
            CORRESPONDENCE_MAGIC_V3,
            MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_VERSION_V3,
            MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_POLICY_V3,
        )?;
        let semantic_sha256 = reader.fixed::<32>()?;
        let canonical_kir_v5_identity = reader.fixed::<32>()?;
        require_nonzero_identity("semantic MIR SHA-256", &semantic_sha256)?;
        require_nonzero_identity(
            "canonical Kernel IR V5 identity",
            &canonical_kir_v5_identity,
        )?;

        let function_count = reader.u32()?;
        let block_count = reader.u32()?;
        let function_count_usize = usize::try_from(function_count).map_err(|_| {
            ProductionLineageEvidenceErrorV3::Overflow {
                field: "correspondence function count",
            }
        })?;
        let block_count_usize = usize::try_from(block_count).map_err(|_| {
            ProductionLineageEvidenceErrorV3::Overflow {
                field: "correspondence block count",
            }
        })?;
        enforce_count(
            "correspondence functions",
            function_count_usize,
            MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3,
        )?;
        enforce_count(
            "correspondence blocks",
            block_count_usize,
            MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3,
        )?;
        if function_count == 0 || block_count == 0 {
            return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "function and block counts must both be nonzero",
            ));
        }
        let record_bytes = block_count_usize
            .checked_mul(CORRESPONDENCE_RECORD_BYTES_V3)
            .ok_or(ProductionLineageEvidenceErrorV3::Overflow {
                field: "correspondence record bytes",
            })?;
        if reader.remaining() != record_bytes {
            return Err(if reader.remaining() < record_bytes {
                ProductionLineageEvidenceErrorV3::Truncated
            } else {
                ProductionLineageEvidenceErrorV3::TrailingBytes
            });
        }

        // Counts and exact remaining bytes are checked before this allocation.
        let mut blocks = Vec::with_capacity(block_count_usize);
        for _ in 0..block_count_usize {
            blocks.push(MirToKirBlockCorrespondenceEvidenceV3 {
                semantic_function: reader.u32()?,
                semantic_block: reader.u32()?,
                kernel_ir_block: reader.u32()?,
                source_statement_count: reader.u32()?,
            });
        }
        reader.finish()?;
        validate_canonical_correspondence(function_count, &blocks)?;

        let reencoded = encode_correspondence(
            semantic_sha256,
            canonical_kir_v5_identity,
            function_count,
            &blocks,
        )?;
        if reencoded != bytes {
            return Err(ProductionLineageEvidenceErrorV3::NonCanonical);
        }
        let identity = MirToKirCorrespondenceEvidenceIdentityV3(canonical_identity(
            CORRESPONDENCE_IDENTITY_DOMAIN_V3,
            &reencoded,
        ));
        require_nonzero_identity("correspondence evidence identity", identity.digest())?;
        Ok(Self {
            canonical_bytes: reencoded,
            identity,
            semantic_sha256,
            canonical_kir_v5_identity,
            function_count,
            blocks: blocks.into_boxed_slice(),
        })
    }

    /// Rechecks strict decoding, re-encoding, and the retained content identity.
    pub fn revalidate(&self) -> Result<(), ProductionLineageEvidenceErrorV3> {
        let decoded = Self::decode(&self.canonical_bytes)?;
        if decoded.identity != self.identity
            || decoded.semantic_sha256 != self.semantic_sha256
            || decoded.canonical_kir_v5_identity != self.canonical_kir_v5_identity
            || decoded.function_count != self.function_count
            || decoded.blocks != self.blocks
        {
            return Err(ProductionLineageEvidenceErrorV3::IdentityMismatch);
        }
        Ok(())
    }

    /// Returns the exact canonical V3 bytes.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    /// Consumes inert evidence and returns its exact canonical V3 bytes.
    pub fn into_canonical_bytes(self) -> Vec<u8> {
        self.canonical_bytes
    }

    /// Returns the exact inert evidence identity.
    pub const fn identity(&self) -> &MirToKirCorrespondenceEvidenceIdentityV3 {
        &self.identity
    }

    /// Returns the exact admitted semantic MIR SHA-256.
    pub const fn semantic_sha256(&self) -> &[u8; 32] {
        &self.semantic_sha256
    }

    /// Returns the identity of the exact canonical Kernel IR V5 bytes.
    pub const fn canonical_kir_v5_identity(&self) -> &[u8; 32] {
        &self.canonical_kir_v5_identity
    }

    /// Returns the number of covered semantic functions.
    pub const fn function_count(&self) -> u32 {
        self.function_count
    }

    /// Returns the complete canonically ordered block correspondence.
    pub fn blocks(&self) -> &[MirToKirBlockCorrespondenceEvidenceV3] {
        &self.blocks
    }

    /// Inert lineage content never grants artifact, proof, or launch authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl TryFrom<&[u8]> for InertCanonicalMirToKirCorrespondenceEvidenceV3 {
    type Error = ProductionLineageEvidenceErrorV3;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::decode(bytes)
    }
}

/// Exact inert formal-memory admission content derived from a live owner.
///
/// The exact canonical formal-obligation receipt is embedded without a lossy
/// projection. Decoding grants no authority and does not prove runtime
/// satisfaction of the retained obligations. A successful live formal owner
/// has no unresolved static admission failure, so V3 records that count as
/// zero; runtime alias requirements remain present in the embedded receipt.
#[derive(Debug, Eq, PartialEq)]
pub struct InertCanonicalFormalMemoryAdmissionEvidenceV3 {
    canonical_bytes: Vec<u8>,
    identity: FormalMemoryAdmissionEvidenceIdentityV3,
    canonical_kir_v5_identity: [u8; 32],
    formal_obligation_receipt_identity: [u8; 32],
    witness_extent: u64,
    completeness_policy: FormalMemoryCompletenessPolicyV3,
    completeness_status: FormalMemoryCompletenessStatusV3,
    static_conflict_count: u32,
    inter_invocation_conflict_count: u32,
    formal_obligation_receipt_offset: usize,
}

impl InertCanonicalFormalMemoryAdmissionEvidenceV3 {
    /// Revalidates a live formal-memory owner and constructs exact admission
    /// evidence bound to the same canonical Kernel IR V5 identity.
    pub fn from_live_owner(
        owner: &ProductionFormalMemoryOwnerV1,
    ) -> Result<Self, ProductionLineageEvidenceErrorV3> {
        owner
            .verify_equivalence()
            .map_err(ProductionLineageEvidenceErrorV3::FormalMemory)?;
        let canonical_kir =
            VerifiedCanonicalKernelIrV5::from_module(owner.semantic_kir().module().clone())
                .map_err(ProductionLineageEvidenceErrorV3::CanonicalKernelIr)?;
        canonical_kir
            .revalidate()
            .map_err(ProductionLineageEvidenceErrorV3::CanonicalKernelIr)?;
        let canonical_kir_v5_identity = *canonical_kir.identity().digest();
        require_nonzero_identity(
            "canonical Kernel IR V5 identity",
            &canonical_kir_v5_identity,
        )?;

        if !owner.obligations().inter_invocation_conflicts().is_empty() {
            return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
                "live owner retains inter-invocation conflicts",
            ));
        }
        let receipt =
            InertCanonicalFormalMemoryObligationReceiptV1::from_obligations(owner.obligations())
                .map_err(ProductionLineageEvidenceErrorV3::FormalObligationReceipt)?;
        receipt
            .revalidate()
            .map_err(ProductionLineageEvidenceErrorV3::FormalObligationReceipt)?;
        let receipt_identity = *receipt.identity().digest();
        require_nonzero_identity("formal-obligation receipt identity", &receipt_identity)?;

        owner
            .verify_equivalence()
            .map_err(ProductionLineageEvidenceErrorV3::FormalMemory)?;
        let canonical_bytes = encode_formal_memory_admission(
            canonical_kir_v5_identity,
            receipt_identity,
            owner.witness_invocation_count(),
            FormalMemoryCompletenessPolicyV3::RequireCompleteConflictFree,
            FormalMemoryCompletenessStatusV3::Complete,
            0,
            0,
            receipt.canonical_bytes(),
        )?;
        Self::decode(&canonical_bytes)
    }

    /// Strictly decodes one complete canonical V3 formal-admission encoding.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProductionLineageEvidenceErrorV3> {
        preflight_total_bytes(
            EvidenceKindV3::FormalMemoryAdmission,
            bytes.len(),
            MAX_FORMAL_MEMORY_ADMISSION_EVIDENCE_BYTES_V3,
        )?;
        let mut reader = ReaderV3::new(bytes);
        decode_common_header(
            &mut reader,
            EvidenceKindV3::FormalMemoryAdmission,
            FORMAL_MEMORY_MAGIC_V3,
            FORMAL_MEMORY_ADMISSION_EVIDENCE_VERSION_V3,
            FORMAL_MEMORY_ADMISSION_EVIDENCE_POLICY_V3,
        )?;
        let canonical_kir_v5_identity = reader.fixed::<32>()?;
        let formal_obligation_receipt_identity = reader.fixed::<32>()?;
        require_nonzero_identity(
            "canonical Kernel IR V5 identity",
            &canonical_kir_v5_identity,
        )?;
        require_nonzero_identity(
            "formal-obligation receipt identity",
            &formal_obligation_receipt_identity,
        )?;
        let witness_extent = reader.u64()?;
        if !is_production_witness_invocation_count(witness_extent) {
            return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
                "flattened witness invocation count does not match the production policy",
            ));
        }
        let completeness_policy = decode_completeness_policy(reader.u16()?)?;
        let completeness_status = decode_completeness_status(reader.u16()?)?;
        let static_conflict_count = reader.u32()?;
        let inter_invocation_conflict_count = reader.u32()?;
        if static_conflict_count != 0 || inter_invocation_conflict_count != 0 {
            return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
                "complete production admission must have zero conflict counts",
            ));
        }
        let receipt_len_u32 = reader.u32()?;
        let receipt_len = usize::try_from(receipt_len_u32).map_err(|_| {
            ProductionLineageEvidenceErrorV3::Overflow {
                field: "formal-obligation receipt length",
            }
        })?;
        if receipt_len == 0 {
            return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
                "formal-obligation receipt is empty",
            ));
        }
        if receipt_len
            > MAX_FORMAL_MEMORY_ADMISSION_EVIDENCE_BYTES_V3
                .saturating_sub(FORMAL_MEMORY_HEADER_BYTES_V3)
        {
            return Err(ProductionLineageEvidenceErrorV3::LimitExceeded {
                field: "embedded formal-obligation receipt bytes",
                actual: receipt_len,
                max: MAX_FORMAL_MEMORY_ADMISSION_EVIDENCE_BYTES_V3 - FORMAL_MEMORY_HEADER_BYTES_V3,
            });
        }
        if reader.remaining() != receipt_len {
            return Err(if reader.remaining() < receipt_len {
                ProductionLineageEvidenceErrorV3::Truncated
            } else {
                ProductionLineageEvidenceErrorV3::TrailingBytes
            });
        }
        let receipt_offset = reader.offset();
        let receipt_bytes = reader.take(receipt_len)?;
        reader.finish()?;

        // The outer length bound is checked before allocating nested bytes.
        let receipt = InertCanonicalFormalMemoryObligationReceiptV1::from_canonical_bytes(
            receipt_bytes.to_vec(),
        )
        .map_err(ProductionLineageEvidenceErrorV3::FormalObligationReceipt)?;
        receipt
            .revalidate()
            .map_err(ProductionLineageEvidenceErrorV3::FormalObligationReceipt)?;
        if receipt.identity().digest() != &formal_obligation_receipt_identity {
            return Err(ProductionLineageEvidenceErrorV3::NestedIdentityMismatch);
        }

        let reencoded = encode_formal_memory_admission(
            canonical_kir_v5_identity,
            formal_obligation_receipt_identity,
            witness_extent,
            completeness_policy,
            completeness_status,
            static_conflict_count,
            inter_invocation_conflict_count,
            receipt.canonical_bytes(),
        )?;
        if reencoded != bytes {
            return Err(ProductionLineageEvidenceErrorV3::NonCanonical);
        }
        let identity = FormalMemoryAdmissionEvidenceIdentityV3(canonical_identity(
            FORMAL_MEMORY_IDENTITY_DOMAIN_V3,
            &reencoded,
        ));
        require_nonzero_identity("formal-memory evidence identity", identity.digest())?;
        Ok(Self {
            canonical_bytes: reencoded,
            identity,
            canonical_kir_v5_identity,
            formal_obligation_receipt_identity,
            witness_extent,
            completeness_policy,
            completeness_status,
            static_conflict_count,
            inter_invocation_conflict_count,
            formal_obligation_receipt_offset: receipt_offset,
        })
    }

    /// Rechecks strict decoding, nested receipt validity, re-encoding, and all
    /// retained identities.
    pub fn revalidate(&self) -> Result<(), ProductionLineageEvidenceErrorV3> {
        let decoded = Self::decode(&self.canonical_bytes)?;
        if decoded.identity != self.identity
            || decoded.canonical_kir_v5_identity != self.canonical_kir_v5_identity
            || decoded.formal_obligation_receipt_identity != self.formal_obligation_receipt_identity
            || decoded.witness_extent != self.witness_extent
            || decoded.completeness_policy != self.completeness_policy
            || decoded.completeness_status != self.completeness_status
            || decoded.static_conflict_count != self.static_conflict_count
            || decoded.inter_invocation_conflict_count != self.inter_invocation_conflict_count
            || decoded.formal_obligation_receipt_offset != self.formal_obligation_receipt_offset
        {
            return Err(ProductionLineageEvidenceErrorV3::IdentityMismatch);
        }
        Ok(())
    }

    /// Returns the exact canonical V3 bytes.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    /// Consumes inert evidence and returns its exact canonical V3 bytes.
    pub fn into_canonical_bytes(self) -> Vec<u8> {
        self.canonical_bytes
    }

    /// Returns the exact inert evidence identity.
    pub const fn identity(&self) -> &FormalMemoryAdmissionEvidenceIdentityV3 {
        &self.identity
    }

    /// Returns the identity of the exact canonical Kernel IR V5 bytes.
    pub const fn canonical_kir_v5_identity(&self) -> &[u8; 32] {
        &self.canonical_kir_v5_identity
    }

    /// Returns the identity of the embedded exact formal-obligation receipt.
    pub const fn formal_obligation_receipt_identity(&self) -> &[u8; 32] {
        &self.formal_obligation_receipt_identity
    }

    /// Returns the embedded exact canonical formal-obligation receipt bytes.
    pub fn formal_obligation_receipt_bytes(&self) -> &[u8] {
        &self.canonical_bytes[self.formal_obligation_receipt_offset..]
    }

    /// Returns the exact flattened structural witness invocation count.
    pub const fn witness_extent(&self) -> u64 {
        self.witness_extent
    }

    /// Returns the completeness policy committed by this evidence.
    pub const fn completeness_policy(&self) -> FormalMemoryCompletenessPolicyV3 {
        self.completeness_policy
    }

    /// Returns the completeness status committed by this evidence.
    pub const fn completeness_status(&self) -> FormalMemoryCompletenessStatusV3 {
        self.completeness_status
    }

    /// Returns the unresolved static conflict count, which is zero for an
    /// admitted production owner.
    pub const fn static_conflict_count(&self) -> u32 {
        self.static_conflict_count
    }

    /// Returns the inter-invocation conflict count, which is zero for an
    /// admitted production owner.
    pub const fn inter_invocation_conflict_count(&self) -> u32 {
        self.inter_invocation_conflict_count
    }

    /// Inert lineage content never grants artifact, proof, or launch authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

impl TryFrom<&[u8]> for InertCanonicalFormalMemoryAdmissionEvidenceV3 {
    type Error = ProductionLineageEvidenceErrorV3;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::decode(bytes)
    }
}

/// V3 lineage evidence category used by bounded codec diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceKindV3 {
    /// MIR-to-KIR correspondence evidence.
    MirToKirCorrespondence,
    /// Formal-memory admission evidence.
    FormalMemoryAdmission,
}

/// Failure to derive, encode, or strictly decode V3 production lineage evidence.
#[derive(Debug)]
pub enum ProductionLineageEvidenceErrorV3 {
    /// The live semantic-KIR owner failed equivalence checking.
    SemanticKir(ProductionSemanticKirErrorV1),
    /// The live formal-memory owner failed equivalence checking.
    FormalMemory(ProductionFormalMemoryErrorV1),
    /// Exact Kernel IR V5 canonicalization or revalidation failed.
    CanonicalKernelIr(VerifiedCanonicalKernelIrErrorV5),
    /// The exact nested formal-obligation receipt failed validation.
    FormalObligationReceipt(FormalMemoryReceiptErrorV1),
    /// Input exceeded a hard byte bound before decoding allocation.
    TooLarge {
        /// Evidence category.
        evidence: EvidenceKindV3,
        /// Observed bytes.
        actual: usize,
        /// Maximum accepted bytes.
        max: usize,
    },
    /// A bounded count exceeded its hard cap.
    LimitExceeded {
        /// Bounded field.
        field: &'static str,
        /// Observed count.
        actual: usize,
        /// Maximum accepted count.
        max: usize,
    },
    /// A checked size calculation overflowed.
    Overflow {
        /// Size field that overflowed.
        field: &'static str,
    },
    /// Input ended before a complete field was available.
    Truncated,
    /// Bytes remained after the exact canonical value.
    TrailingBytes,
    /// The wire magic did not match the selected evidence kind.
    InvalidMagic {
        /// Evidence category.
        evidence: EvidenceKindV3,
    },
    /// The wire version is unsupported.
    UnknownVersion {
        /// Evidence category.
        evidence: EvidenceKindV3,
        /// Rejected version.
        version: u16,
    },
    /// The codec policy is unsupported.
    UnknownPolicy {
        /// Evidence category.
        evidence: EvidenceKindV3,
        /// Rejected policy.
        policy: u16,
    },
    /// Reserved flags were nonzero.
    UnsupportedFlags(u16),
    /// A reserved field was nonzero.
    ReservedNonzero,
    /// The declared total length did not equal the supplied bytes.
    InvalidLength {
        /// Declared length.
        declared: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A required content identity was all zeroes.
    ZeroIdentity {
        /// Rejected identity field.
        field: &'static str,
    },
    /// A correspondence invariant was violated.
    InvalidCorrespondence(&'static str),
    /// A formal-admission invariant was violated.
    InvalidFormalAdmission(&'static str),
    /// The formal completeness policy tag is unsupported.
    UnknownCompletenessPolicy(u16),
    /// The formal completeness status tag is unsupported.
    UnknownCompletenessStatus(u16),
    /// The nested receipt identity did not match its exact bytes.
    NestedIdentityMismatch,
    /// Decoding and structured re-encoding changed the bytes.
    NonCanonical,
    /// Retained fields or content identity changed during revalidation.
    IdentityMismatch,
}

impl fmt::Display for ProductionLineageEvidenceErrorV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticKir(error) => write!(formatter, "semantic-KIR owner failed: {error}"),
            Self::FormalMemory(error) => write!(formatter, "formal-memory owner failed: {error}"),
            Self::CanonicalKernelIr(error) => write!(formatter, "canonical KIR V5 failed: {error}"),
            Self::FormalObligationReceipt(error) => {
                write!(
                    formatter,
                    "canonical formal-obligation receipt failed: {error}"
                )
            }
            Self::TooLarge {
                evidence,
                actual,
                max,
            } => write!(formatter, "{evidence:?} bytes {actual} exceed limit {max}"),
            Self::LimitExceeded { field, actual, max } => {
                write!(formatter, "{field} count {actual} exceeds limit {max}")
            }
            Self::Overflow { field } => write!(formatter, "{field} size overflowed"),
            Self::Truncated => formatter.write_str("lineage evidence is truncated"),
            Self::TrailingBytes => formatter.write_str("lineage evidence has trailing bytes"),
            Self::InvalidMagic { evidence } => {
                write!(formatter, "invalid {evidence:?} wire magic")
            }
            Self::UnknownVersion { evidence, version } => {
                write!(formatter, "unsupported {evidence:?} version {version}")
            }
            Self::UnknownPolicy { evidence, policy } => {
                write!(formatter, "unsupported {evidence:?} policy {policy}")
            }
            Self::UnsupportedFlags(flags) => {
                write!(formatter, "unsupported lineage evidence flags {flags:#06x}")
            }
            Self::ReservedNonzero => {
                formatter.write_str("reserved lineage evidence field is nonzero")
            }
            Self::InvalidLength { declared, actual } => write!(
                formatter,
                "declared lineage evidence length {declared} does not equal supplied length {actual}",
            ),
            Self::ZeroIdentity { field } => write!(formatter, "{field} must be nonzero"),
            Self::InvalidCorrespondence(detail) => {
                write!(formatter, "invalid MIR-to-KIR correspondence: {detail}")
            }
            Self::InvalidFormalAdmission(detail) => {
                write!(formatter, "invalid formal-memory admission: {detail}")
            }
            Self::UnknownCompletenessPolicy(policy) => {
                write!(formatter, "unsupported formal completeness policy {policy}")
            }
            Self::UnknownCompletenessStatus(status) => {
                write!(formatter, "unsupported formal completeness status {status}")
            }
            Self::NestedIdentityMismatch => {
                formatter.write_str("formal-obligation receipt identity does not match exact bytes")
            }
            Self::NonCanonical => formatter.write_str("lineage evidence is not canonical"),
            Self::IdentityMismatch => formatter.write_str("lineage evidence identity mismatch"),
        }
    }
}

impl Error for ProductionLineageEvidenceErrorV3 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SemanticKir(error) => Some(error),
            Self::FormalMemory(error) => Some(error),
            Self::CanonicalKernelIr(error) => Some(error),
            Self::FormalObligationReceipt(error) => Some(error),
            Self::TooLarge { .. }
            | Self::LimitExceeded { .. }
            | Self::Overflow { .. }
            | Self::Truncated
            | Self::TrailingBytes
            | Self::InvalidMagic { .. }
            | Self::UnknownVersion { .. }
            | Self::UnknownPolicy { .. }
            | Self::UnsupportedFlags(_)
            | Self::ReservedNonzero
            | Self::InvalidLength { .. }
            | Self::ZeroIdentity { .. }
            | Self::InvalidCorrespondence(_)
            | Self::InvalidFormalAdmission(_)
            | Self::UnknownCompletenessPolicy(_)
            | Self::UnknownCompletenessStatus(_)
            | Self::NestedIdentityMismatch
            | Self::NonCanonical
            | Self::IdentityMismatch => None,
        }
    }
}

fn exact_correspondence_from_owner(
    owner: &ProductionSemanticKirOwnerV1,
) -> Result<(u32, Vec<MirToKirBlockCorrespondenceEvidenceV3>), ProductionLineageEvidenceErrorV3> {
    let semantic = owner.semantic().semantic();
    let covered_functions = owner
        .correspondence()
        .blocks()
        .iter()
        .map(|record| record.semantic_function().index())
        .collect::<BTreeSet<_>>();
    enforce_count(
        "correspondence functions",
        covered_functions.len(),
        MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3,
    )?;
    let target_functions = owner
        .module()
        .functions
        .iter()
        .filter(|function| function.body.is_some())
        .collect::<Vec<_>>();
    if covered_functions.is_empty() || target_functions.len() != covered_functions.len() {
        return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
            "retained semantic and defined Kernel IR function coverage differs",
        ));
    }
    let target_by_semantic_function = covered_functions
        .iter()
        .copied()
        .zip(target_functions)
        .collect::<BTreeMap<_, _>>();
    let function_count = u32::try_from(covered_functions.len()).map_err(|_| {
        ProductionLineageEvidenceErrorV3::Overflow {
            field: "correspondence function count",
        }
    })?;
    let expected_blocks = covered_functions
        .iter()
        .try_fold(0_usize, |total, function| {
            let function = semantic.functions().get(*function as usize).ok_or(
                ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                    "covered semantic function locator is absent",
                ),
            )?;
            total.checked_add(function.blocks().len()).ok_or(
                ProductionLineageEvidenceErrorV3::Overflow {
                    field: "correspondence block count",
                },
            )
        })?;
    enforce_count(
        "correspondence blocks",
        expected_blocks,
        MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3,
    )?;
    if expected_blocks == 0 || owner.correspondence().blocks().len() != expected_blocks {
        return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
            "semantic and retained block coverage differs",
        ));
    }

    // The owner is bounded before this allocation.
    let mut blocks = Vec::with_capacity(expected_blocks);
    for record in owner.correspondence().blocks() {
        blocks.push(MirToKirBlockCorrespondenceEvidenceV3 {
            semantic_function: record.semantic_function().index(),
            semantic_block: record.semantic_block().index(),
            kernel_ir_block: record.kernel_ir_block().0,
            source_statement_count: record.source_statement_count(),
        });
    }
    blocks.sort_unstable_by_key(|record| (record.semantic_function, record.semantic_block));
    validate_canonical_correspondence(function_count, &blocks)?;

    for record in &blocks {
        let function = semantic
            .functions()
            .get(record.semantic_function as usize)
            .ok_or(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "semantic function locator is absent",
            ))?;
        let block = function
            .blocks()
            .get(record.semantic_block as usize)
            .ok_or(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "semantic block locator is absent",
            ))?;
        if usize::try_from(record.source_statement_count) != Ok(block.statements().len()) {
            return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "source statement count differs from exact semantic MIR",
            ));
        }
        let target_function = target_by_semantic_function
            .get(&record.semantic_function)
            .ok_or(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "corresponding Kernel IR function is absent",
            ))?;
        let Some(body) = target_function.body.as_ref() else {
            return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "corresponding Kernel IR function has no body",
            ));
        };
        if !body
            .blocks
            .iter()
            .any(|block| block.id.0 == record.kernel_ir_block)
        {
            return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "corresponding Kernel IR block is absent",
            ));
        }
    }
    Ok((function_count, blocks))
}

fn validate_canonical_correspondence(
    function_count: u32,
    blocks: &[MirToKirBlockCorrespondenceEvidenceV3],
) -> Result<(), ProductionLineageEvidenceErrorV3> {
    let mut cursor = 0_usize;
    let mut previous_function = None;
    let mut covered_function_count = 0_u32;
    while let Some(first) = blocks.get(cursor) {
        let function = first.semantic_function;
        if previous_function.is_some_and(|previous| previous >= function) {
            return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "semantic function locators are not in strictly increasing canonical order",
            ));
        }
        let mut block = 0_u32;
        while let Some(record) = blocks.get(cursor) {
            if record.semantic_function != function {
                break;
            }
            if record.semantic_block != block {
                return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                    "semantic block locators are not contiguous canonical indices",
                ));
            }
            if record.kernel_ir_block != record.semantic_block {
                return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                    "current V3 policy requires exact semantic-to-Kernel-IR block identity",
                ));
            }
            cursor += 1;
            block = block
                .checked_add(1)
                .ok_or(ProductionLineageEvidenceErrorV3::Overflow {
                    field: "semantic block index",
                })?;
        }
        if block == 0 {
            return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
                "every covered semantic function must contain a block",
            ));
        }
        previous_function = Some(function);
        covered_function_count = covered_function_count.checked_add(1).ok_or(
            ProductionLineageEvidenceErrorV3::Overflow {
                field: "covered semantic function count",
            },
        )?;
    }
    if cursor != blocks.len() || covered_function_count != function_count {
        return Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(
            "covered semantic function count differs from canonical records",
        ));
    }
    Ok(())
}

fn encode_correspondence(
    semantic_sha256: [u8; 32],
    canonical_kir_v5_identity: [u8; 32],
    function_count: u32,
    blocks: &[MirToKirBlockCorrespondenceEvidenceV3],
) -> Result<Vec<u8>, ProductionLineageEvidenceErrorV3> {
    require_nonzero_identity("semantic MIR SHA-256", &semantic_sha256)?;
    require_nonzero_identity(
        "canonical Kernel IR V5 identity",
        &canonical_kir_v5_identity,
    )?;
    validate_canonical_correspondence(function_count, blocks)?;
    enforce_count(
        "correspondence functions",
        function_count as usize,
        MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3,
    )?;
    enforce_count(
        "correspondence blocks",
        blocks.len(),
        MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3,
    )?;
    let exact_size = CORRESPONDENCE_HEADER_BYTES_V3
        .checked_add(
            blocks
                .len()
                .checked_mul(CORRESPONDENCE_RECORD_BYTES_V3)
                .ok_or(ProductionLineageEvidenceErrorV3::Overflow {
                    field: "correspondence record bytes",
                })?,
        )
        .ok_or(ProductionLineageEvidenceErrorV3::Overflow {
            field: "correspondence evidence bytes",
        })?;
    preflight_total_bytes(
        EvidenceKindV3::MirToKirCorrespondence,
        exact_size,
        MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V3,
    )?;
    let declared =
        u32::try_from(exact_size).map_err(|_| ProductionLineageEvidenceErrorV3::Overflow {
            field: "correspondence evidence length",
        })?;
    let block_count =
        u32::try_from(blocks.len()).map_err(|_| ProductionLineageEvidenceErrorV3::Overflow {
            field: "correspondence block count",
        })?;

    // Every checked bound precedes this exact allocation.
    let mut bytes = Vec::with_capacity(exact_size);
    encode_common_header(
        &mut bytes,
        CORRESPONDENCE_MAGIC_V3,
        MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_VERSION_V3,
        MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_POLICY_V3,
        declared,
    );
    bytes.extend_from_slice(&semantic_sha256);
    bytes.extend_from_slice(&canonical_kir_v5_identity);
    push_u32(&mut bytes, function_count);
    push_u32(&mut bytes, block_count);
    for block in blocks {
        push_u32(&mut bytes, block.semantic_function);
        push_u32(&mut bytes, block.semantic_block);
        push_u32(&mut bytes, block.kernel_ir_block);
        push_u32(&mut bytes, block.source_statement_count);
    }
    debug_assert_eq!(bytes.len(), exact_size);
    Ok(bytes)
}

#[allow(clippy::too_many_arguments)]
fn encode_formal_memory_admission(
    canonical_kir_v5_identity: [u8; 32],
    formal_obligation_receipt_identity: [u8; 32],
    witness_extent: u64,
    completeness_policy: FormalMemoryCompletenessPolicyV3,
    completeness_status: FormalMemoryCompletenessStatusV3,
    static_conflict_count: u32,
    inter_invocation_conflict_count: u32,
    formal_obligation_receipt: &[u8],
) -> Result<Vec<u8>, ProductionLineageEvidenceErrorV3> {
    require_nonzero_identity(
        "canonical Kernel IR V5 identity",
        &canonical_kir_v5_identity,
    )?;
    require_nonzero_identity(
        "formal-obligation receipt identity",
        &formal_obligation_receipt_identity,
    )?;
    if !is_production_witness_invocation_count(witness_extent)
        || completeness_policy != FormalMemoryCompletenessPolicyV3::RequireCompleteConflictFree
        || completeness_status != FormalMemoryCompletenessStatusV3::Complete
        || static_conflict_count != 0
        || inter_invocation_conflict_count != 0
        || formal_obligation_receipt.is_empty()
    {
        return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
            "formal admission fields do not satisfy the production policy",
        ));
    }
    validate_formal_receipt_witness(formal_obligation_receipt, witness_extent)?;
    let exact_size = FORMAL_MEMORY_HEADER_BYTES_V3
        .checked_add(formal_obligation_receipt.len())
        .ok_or(ProductionLineageEvidenceErrorV3::Overflow {
            field: "formal-memory admission evidence bytes",
        })?;
    preflight_total_bytes(
        EvidenceKindV3::FormalMemoryAdmission,
        exact_size,
        MAX_FORMAL_MEMORY_ADMISSION_EVIDENCE_BYTES_V3,
    )?;
    let declared =
        u32::try_from(exact_size).map_err(|_| ProductionLineageEvidenceErrorV3::Overflow {
            field: "formal-memory admission evidence length",
        })?;
    let receipt_len = u32::try_from(formal_obligation_receipt.len()).map_err(|_| {
        ProductionLineageEvidenceErrorV3::Overflow {
            field: "formal-obligation receipt length",
        }
    })?;

    // Every checked bound precedes this exact allocation.
    let mut bytes = Vec::with_capacity(exact_size);
    encode_common_header(
        &mut bytes,
        FORMAL_MEMORY_MAGIC_V3,
        FORMAL_MEMORY_ADMISSION_EVIDENCE_VERSION_V3,
        FORMAL_MEMORY_ADMISSION_EVIDENCE_POLICY_V3,
        declared,
    );
    bytes.extend_from_slice(&canonical_kir_v5_identity);
    bytes.extend_from_slice(&formal_obligation_receipt_identity);
    push_u64(&mut bytes, witness_extent);
    push_u16(&mut bytes, completeness_policy as u16);
    push_u16(&mut bytes, completeness_status as u16);
    push_u32(&mut bytes, static_conflict_count);
    push_u32(&mut bytes, inter_invocation_conflict_count);
    push_u32(&mut bytes, receipt_len);
    bytes.extend_from_slice(formal_obligation_receipt);
    debug_assert_eq!(bytes.len(), exact_size);
    Ok(bytes)
}

fn preflight_total_bytes(
    evidence: EvidenceKindV3,
    actual: usize,
    max: usize,
) -> Result<(), ProductionLineageEvidenceErrorV3> {
    if actual > max {
        return Err(ProductionLineageEvidenceErrorV3::TooLarge {
            evidence,
            actual,
            max,
        });
    }
    Ok(())
}

fn enforce_count(
    field: &'static str,
    actual: usize,
    max: usize,
) -> Result<(), ProductionLineageEvidenceErrorV3> {
    if actual > max {
        return Err(ProductionLineageEvidenceErrorV3::LimitExceeded { field, actual, max });
    }
    Ok(())
}

fn require_nonzero_identity(
    field: &'static str,
    identity: &[u8; 32],
) -> Result<(), ProductionLineageEvidenceErrorV3> {
    if identity.iter().all(|byte| *byte == 0) {
        return Err(ProductionLineageEvidenceErrorV3::ZeroIdentity { field });
    }
    Ok(())
}

fn canonical_identity(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u32).to_le_bytes());
    digest.update(domain);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    digest.finalize().into()
}

fn encode_common_header(
    bytes: &mut Vec<u8>,
    magic: [u8; 8],
    version: u16,
    policy: u16,
    declared: u32,
) {
    bytes.extend_from_slice(&magic);
    push_u16(bytes, version);
    push_u16(bytes, policy);
    push_u16(bytes, 0);
    push_u16(bytes, 0);
    push_u32(bytes, declared);
}

fn decode_common_header(
    reader: &mut ReaderV3<'_>,
    evidence: EvidenceKindV3,
    magic: [u8; 8],
    version: u16,
    policy: u16,
) -> Result<(), ProductionLineageEvidenceErrorV3> {
    if reader.fixed::<8>()? != magic {
        return Err(ProductionLineageEvidenceErrorV3::InvalidMagic { evidence });
    }
    let actual_version = reader.u16()?;
    if actual_version != version {
        return Err(ProductionLineageEvidenceErrorV3::UnknownVersion {
            evidence,
            version: actual_version,
        });
    }
    let actual_policy = reader.u16()?;
    if actual_policy != policy {
        return Err(ProductionLineageEvidenceErrorV3::UnknownPolicy {
            evidence,
            policy: actual_policy,
        });
    }
    let flags = reader.u16()?;
    if flags != 0 {
        return Err(ProductionLineageEvidenceErrorV3::UnsupportedFlags(flags));
    }
    if reader.u16()? != 0 {
        return Err(ProductionLineageEvidenceErrorV3::ReservedNonzero);
    }
    let declared = reader.u32()? as usize;
    if declared != reader.bytes.len() {
        return Err(ProductionLineageEvidenceErrorV3::InvalidLength {
            declared,
            actual: reader.bytes.len(),
        });
    }
    Ok(())
}

fn decode_completeness_policy(
    value: u16,
) -> Result<FormalMemoryCompletenessPolicyV3, ProductionLineageEvidenceErrorV3> {
    match value {
        1 => Ok(FormalMemoryCompletenessPolicyV3::RequireCompleteConflictFree),
        other => Err(ProductionLineageEvidenceErrorV3::UnknownCompletenessPolicy(
            other,
        )),
    }
}

fn decode_completeness_status(
    value: u16,
) -> Result<FormalMemoryCompletenessStatusV3, ProductionLineageEvidenceErrorV3> {
    match value {
        1 => Ok(FormalMemoryCompletenessStatusV3::Complete),
        other => Err(ProductionLineageEvidenceErrorV3::UnknownCompletenessStatus(
            other,
        )),
    }
}

fn validate_formal_receipt_witness(
    receipt: &[u8],
    witness_extent: u64,
) -> Result<(), ProductionLineageEvidenceErrorV3> {
    let mut reader = ReaderV3::new(receipt);
    if reader.fixed::<8>()? != FORMAL_OBLIGATION_RECEIPT_MAGIC_V1 {
        return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
            "embedded formal-obligation receipt magic changed after validation",
        ));
    }
    if reader.u16()? != FORMAL_MEMORY_OBLIGATION_RECEIPT_VERSION_V1
        || reader.u16()? != FORMAL_MEMORY_OBLIGATION_POLICY_V1
        || reader.u16()? != 0
        || reader.u16()? != 0
        || reader.u32()? as usize != receipt.len()
    {
        return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
            "embedded formal-obligation receipt header changed after validation",
        ));
    }
    for _ in 0..2 {
        let text_len = reader.u32()? as usize;
        reader.take(text_len)?;
    }
    reader.u8()?;
    reader.u8()?;
    if reader.u16()? != 0 || reader.u8()? != 1 {
        return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
            "embedded formal-obligation receipt has no exact invocation witness",
        ));
    }
    let start = reader.u64()?;
    let end_exclusive = reader.u64()?;
    if start != 0 || end_exclusive != witness_extent {
        return Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(
            "embedded formal-obligation receipt invocation range differs from witness extent",
        ));
    }
    Ok(())
}

const fn is_production_witness_invocation_count(count: u64) -> bool {
    count != 0
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct ReaderV3<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ReaderV3<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    const fn offset(&self) -> usize {
        self.offset
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ProductionLineageEvidenceErrorV3> {
        let end =
            self.offset
                .checked_add(len)
                .ok_or(ProductionLineageEvidenceErrorV3::Overflow {
                    field: "decoder offset",
                })?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ProductionLineageEvidenceErrorV3::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], ProductionLineageEvidenceErrorV3> {
        self.take(N)?
            .try_into()
            .map_err(|_| ProductionLineageEvidenceErrorV3::Truncated)
    }

    fn u16(&mut self) -> Result<u16, ProductionLineageEvidenceErrorV3> {
        Ok(u16::from_le_bytes(self.fixed()?))
    }

    fn u8(&mut self) -> Result<u8, ProductionLineageEvidenceErrorV3> {
        Ok(self.fixed::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, ProductionLineageEvidenceErrorV3> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    fn u64(&mut self) -> Result<u64, ProductionLineageEvidenceErrorV3> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    fn finish(self) -> Result<(), ProductionLineageEvidenceErrorV3> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(ProductionLineageEvidenceErrorV3::TrailingBytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use fe2o3_kernel_ir::{
        ExplicitLaunchExtent1d, FormalIndexWidth, FormalMemoryObligationAnalysis,
        InertCanonicalFormalMemoryObligationReceiptV1, VerifiedCanonicalKernelIrV5,
        derive_kernel_memory_obligations,
    };
    use fe2o3_mir_model::semantic_mir_v1::*;
    use fe2o3_pliron::{ProductionSemanticMirLimitsV1, ProductionSemanticMirOwnerV1};

    use super::*;
    use crate::{
        FormalMemoryCompletenessPolicyV4, FormalMemoryCompletenessStatusV4,
        InertCanonicalFormalMemoryAdmissionEvidenceV4,
        InertCanonicalMirToKirCorrespondenceEvidenceV4,
        MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V4, MAX_MIR_TO_KIR_STATEMENT_SPANS_V4,
        PRODUCTION_FORMAL_MEMORY_WITNESS_EXTENT_V1, ProductionCorrespondenceEvidenceErrorV4,
        ProductionSemanticKirLimitsV1,
    };

    const SEMANTIC_SHA_OFFSET: usize = COMMON_HEADER_BYTES_V3;
    const KIR_IDENTITY_OFFSET: usize = SEMANTIC_SHA_OFFSET + 32;
    const FUNCTION_COUNT_OFFSET: usize = KIR_IDENTITY_OFFSET + 32;
    const BLOCK_COUNT_OFFSET: usize = FUNCTION_COUNT_OFFSET + 4;
    const FIRST_BLOCK_OFFSET: usize = BLOCK_COUNT_OFFSET + 4;

    const FORMAL_KIR_IDENTITY_OFFSET: usize = COMMON_HEADER_BYTES_V3;
    const FORMAL_RECEIPT_IDENTITY_OFFSET: usize = FORMAL_KIR_IDENTITY_OFFSET + 32;
    const FORMAL_WITNESS_OFFSET: usize = FORMAL_RECEIPT_IDENTITY_OFFSET + 32;
    const FORMAL_COMPLETENESS_POLICY_OFFSET: usize = FORMAL_WITNESS_OFFSET + 8;
    const FORMAL_COMPLETENESS_STATUS_OFFSET: usize = FORMAL_COMPLETENESS_POLICY_OFFSET + 2;
    const FORMAL_STATIC_CONFLICT_OFFSET: usize = FORMAL_COMPLETENESS_STATUS_OFFSET + 2;
    const FORMAL_INTER_INVOCATION_CONFLICT_OFFSET: usize = FORMAL_STATIC_CONFLICT_OFFSET + 4;
    const FORMAL_RECEIPT_LENGTH_OFFSET: usize = FORMAL_INTER_INVOCATION_CONFLICT_OFFSET + 4;
    const FORMAL_RECEIPT_OFFSET: usize = FORMAL_RECEIPT_LENGTH_OFFSET + 4;

    fn bytes(tag: u8) -> [u8; 32] {
        [tag; 32]
    }

    fn unit_type() -> SemanticTypeDeclV1 {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(4)),
            SemanticLayoutIdentityV1::from_sha256(bytes(4)),
            SemanticTypeLayoutV1::with_exact_rustc_layout(
                0,
                1,
                SemanticFieldsShapeV1::arbitrary(vec![], vec![]).unwrap(),
                SemanticRustcVariantsV1::Single { index: 0 },
                SemanticBackendReprV1::memory(true),
                None,
                false,
                None,
                1,
                0,
                SemanticTypeLayoutDetailsV1::None,
            )
            .unwrap(),
            SemanticTypeShapeV1::Unit,
        )
    }

    fn block(
        tag: u8,
        statements: Vec<SemanticStatementV1>,
        terminator: SemanticTerminatorKindV1,
    ) -> SemanticBasicBlockV1 {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256(bytes(tag)),
            SemanticSourceProvenanceV1::unavailable(),
            statements,
            SemanticTerminatorV1::new(SemanticSourceProvenanceV1::unavailable(), terminator),
        )
        .unwrap()
    }

    fn semantic_owner() -> ProductionSemanticMirOwnerV1 {
        let type_id = SemanticTypeIdV1::from_index(0);
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256(bytes(2)),
            SemanticLayoutIdentityV1::from_sha256(bytes(250)),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(type_id, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let statement = SemanticStatementV1::new(
            SemanticSourceProvenanceV1::unavailable(),
            SemanticStatementKindV1::Nop,
        );
        let block0 = block(10, vec![statement], SemanticTerminatorKindV1::Return);
        let block1 = block(
            11,
            vec![],
            SemanticTerminatorKindV1::Goto(SemanticControlFlowEdgeV1::new(
                SemanticEdgeRoleV1::Goto,
                SemanticBlockIdV1::from_index(0),
            )),
        );
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256(bytes(2)),
            SemanticFunctionRoleV1::KernelRoot,
            SemanticItemDefinitionIdentityV1::from_sha256(bytes(2)),
            SemanticMonomorphizationIdentityV1::from_sha256(bytes(2)),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256(bytes(2)),
            SemanticConstGenericArgumentsIdentityV1::from_sha256(bytes(2)),
            SemanticSourceProvenanceV1::unavailable(),
            abi,
            vec![SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256(bytes(3)),
                type_id,
                SemanticLocalRoleV1::Return,
                SemanticSourceProvenanceV1::unavailable(),
            )],
            SemanticBlockIdV1::from_index(1),
            vec![block0, block1],
        )
        .unwrap();
        let dimensions = SemanticWorkgroupDimensionsV1::new([64, 1, 1]).unwrap();
        let launch =
            SemanticKernelLaunchBoundsV1::new(Some(dimensions), Some(dimensions), None).unwrap();
        let contract = SemanticKernelSourceContractV1::new(Some(launch), None, None).unwrap();
        let function = function.with_kernel_entry(SemanticKernelEntryV1::new(
            SemanticLinkSymbolV1::new(b"lineage_evidence_test".to_vec()).unwrap(),
            SemanticKernelBindingIdentityV1::from_sha256(bytes(5)),
            contract,
        ));
        let admitted = InertSemanticMirRequestV1::new(
            SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256(bytes(250))),
            vec![unit_type()],
            vec![],
            vec![],
            vec![],
            vec![function],
            vec![SemanticFunctionIdV1::from_index(0)],
        )
        .unwrap()
        .admit(SemanticMirLimitsV1::default())
        .unwrap();
        ProductionSemanticMirOwnerV1::try_new(admitted, ProductionSemanticMirLimitsV1::default())
            .unwrap()
    }

    fn semantic_kir_owner() -> ProductionSemanticKirOwnerV1 {
        ProductionSemanticKirOwnerV1::try_lower(
            semantic_owner(),
            ProductionSemanticKirLimitsV1::default(),
        )
        .unwrap()
    }

    fn evidence_pair() -> (
        InertCanonicalMirToKirCorrespondenceEvidenceV3,
        InertCanonicalFormalMemoryAdmissionEvidenceV3,
    ) {
        let semantic_kir = semantic_kir_owner();
        let correspondence =
            InertCanonicalMirToKirCorrespondenceEvidenceV3::from_live_owner(&semantic_kir).unwrap();
        let formal_owner = ProductionFormalMemoryOwnerV1::try_admit(semantic_kir).unwrap();
        let formal =
            InertCanonicalFormalMemoryAdmissionEvidenceV3::from_live_owner(&formal_owner).unwrap();
        (correspondence, formal)
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn live_owners_bind_exact_semantics_kir_and_formal_receipt_without_authority() {
        let semantic_kir = semantic_kir_owner();
        let semantic_sha = *semantic_kir
            .semantic()
            .semantic()
            .semantic_sha256()
            .as_bytes();
        let correspondence =
            InertCanonicalMirToKirCorrespondenceEvidenceV3::from_live_owner(&semantic_kir).unwrap();
        assert_eq!(correspondence.semantic_sha256(), &semantic_sha);
        assert_eq!(correspondence.function_count(), 1);
        assert_eq!(correspondence.blocks().len(), 2);
        assert_eq!(correspondence.blocks()[0].semantic_block(), 0);
        assert_eq!(correspondence.blocks()[0].kernel_ir_block(), 0);
        assert_eq!(correspondence.blocks()[0].source_statement_count(), 1);
        assert_eq!(correspondence.blocks()[1].semantic_block(), 1);
        assert_eq!(correspondence.blocks()[1].kernel_ir_block(), 1);
        assert_eq!(correspondence.blocks()[1].source_statement_count(), 0);
        correspondence.revalidate().unwrap();
        assert!(!correspondence.grants_authority());

        let formal_owner = ProductionFormalMemoryOwnerV1::try_admit(semantic_kir).unwrap();
        let exact_receipt = InertCanonicalFormalMemoryObligationReceiptV1::from_obligations(
            formal_owner.obligations(),
        )
        .unwrap();
        let formal =
            InertCanonicalFormalMemoryAdmissionEvidenceV3::from_live_owner(&formal_owner).unwrap();
        assert_eq!(
            correspondence.canonical_kir_v5_identity(),
            formal.canonical_kir_v5_identity(),
        );
        assert_eq!(
            formal.formal_obligation_receipt_bytes(),
            exact_receipt.canonical_bytes(),
        );
        assert_eq!(
            formal.formal_obligation_receipt_identity(),
            exact_receipt.identity().digest(),
        );
        assert_eq!(
            formal.witness_extent(),
            PRODUCTION_FORMAL_MEMORY_WITNESS_EXTENT_V1
        );
        assert_eq!(
            formal.completeness_policy(),
            FormalMemoryCompletenessPolicyV3::RequireCompleteConflictFree,
        );
        assert_eq!(
            formal.completeness_status(),
            FormalMemoryCompletenessStatusV3::Complete,
        );
        assert_eq!(formal.static_conflict_count(), 0);
        assert_eq!(formal.inter_invocation_conflict_count(), 0);
        formal.revalidate().unwrap();
        assert!(!formal.grants_authority());
    }

    #[test]
    fn independent_live_derivations_and_strict_decodes_are_canonical() {
        let first = evidence_pair();
        let second = evidence_pair();
        assert_eq!(first.0.canonical_bytes(), second.0.canonical_bytes());
        assert_eq!(first.0.identity(), second.0.identity());
        assert_eq!(first.1.canonical_bytes(), second.1.canonical_bytes());
        assert_eq!(first.1.identity(), second.1.identity());

        let decoded_correspondence =
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(first.0.canonical_bytes())
                .unwrap();
        let decoded_formal =
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(first.1.canonical_bytes())
                .unwrap();
        assert_eq!(decoded_correspondence, first.0);
        assert_eq!(decoded_formal, first.1);
    }

    #[test]
    fn lossless_correspondence_retains_every_live_span_and_induction_report() {
        let semantic_kir = semantic_kir_owner();
        let semantic = semantic_kir.semantic().semantic();
        let induction = fe2o3_mir_model::analyze_semantic_u32_induction_no_overflow_v1(
            semantic,
            SemanticFunctionIdV1::from_index(0),
        )
        .unwrap();
        let evidence = InertCanonicalMirToKirCorrespondenceEvidenceV4::from_live_owner(
            &semantic_kir,
            &induction,
        )
        .unwrap();
        evidence.revalidate().unwrap();
        assert_eq!(evidence.statement_spans().len(), 1);
        assert_eq!(evidence.terminator_spans().len(), 2);
        assert!(evidence.synthetic_spans().is_empty());
        assert!(evidence.parameter_bindings().is_empty());
        assert_eq!(
            evidence.semantic_u32_induction().semantic_mir_sha256(),
            semantic.semantic_sha256().as_bytes()
        );
        assert_eq!(
            evidence.canonical_kernel_ir_identity(),
            semantic_kir.canonical_kernel_ir_identity()
        );
        assert!(!evidence.grants_authority());

        let canonical = evidence.canonical_bytes();
        assert_eq!(
            InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(canonical).unwrap(),
            evidence
        );
        for end in 0..canonical.len() {
            assert!(
                InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&canonical[..end]).is_err(),
                "accepted V4 truncation at {end}"
            );
        }
        let mut trailing = canonical.to_vec();
        trailing.push(0);
        assert!(InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&trailing).is_err());

        let mut unknown_kir_version = canonical.to_vec();
        put_u16(&mut unknown_kir_version, 52, 7);
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&unknown_kir_version),
            Err(ProductionCorrespondenceEvidenceErrorV4::InvalidHeader)
        ));

        let mut zero_kir_length = canonical.to_vec();
        put_u64(&mut zero_kir_length, 56, 0);
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&zero_kir_length),
            Err(ProductionCorrespondenceEvidenceErrorV4::ZeroIdentity)
        ));

        let mut oversized_statement_count = canonical.to_vec();
        put_u32(
            &mut oversized_statement_count,
            104,
            MAX_MIR_TO_KIR_STATEMENT_SPANS_V4 as u32 + 1,
        );
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&oversized_statement_count),
            Err(ProductionCorrespondenceEvidenceErrorV4::LimitExceeded)
        ));

        let block_count = u32::from_le_bytes(canonical[100..104].try_into().unwrap()) as usize;
        let statement_count = u32::from_le_bytes(canonical[104..108].try_into().unwrap()) as usize;
        let terminator_count = u32::from_le_bytes(canonical[108..112].try_into().unwrap()) as usize;
        let synthetic_count = u32::from_le_bytes(canonical[112..116].try_into().unwrap()) as usize;
        let parameter_count = u32::from_le_bytes(canonical[116..120].try_into().unwrap()) as usize;
        let induction_offset = 124
            + block_count * 16
            + statement_count * 24
            + terminator_count * 20
            + synthetic_count * 16
            + parameter_count * 12;
        let mut mismatched_semantic = canonical.to_vec();
        mismatched_semantic[induction_offset + 20] ^= 0x80;
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&mismatched_semantic),
            Err(ProductionCorrespondenceEvidenceErrorV4::NestedIdentityMismatch)
        ));

        let oversized = vec![0; MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V4 + 1];
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV4::decode(&oversized),
            Err(ProductionCorrespondenceEvidenceErrorV4::TooLarge)
        ));
    }

    #[test]
    fn current_formal_memory_evidence_binds_versioned_kir_and_exact_receipt() {
        let semantic_kir = semantic_kir_owner();
        let canonical_kir = semantic_kir.canonical_kernel_ir_identity();
        let formal_owner = ProductionFormalMemoryOwnerV1::try_admit(semantic_kir).unwrap();
        let receipt = InertCanonicalFormalMemoryObligationReceiptV1::from_obligations(
            formal_owner.obligations(),
        )
        .unwrap();
        let evidence =
            InertCanonicalFormalMemoryAdmissionEvidenceV4::from_live_owner(&formal_owner).unwrap();

        assert_eq!(evidence.canonical_kernel_ir_identity(), canonical_kir);
        assert_eq!(
            evidence.formal_obligation_receipt_identity(),
            receipt.identity().digest()
        );
        assert_eq!(
            evidence.formal_obligation_receipt_bytes(),
            receipt.canonical_bytes()
        );
        assert_eq!(
            evidence.witness_invocation_count(),
            formal_owner.witness_invocation_count()
        );
        assert_eq!(
            evidence.completeness_policy(),
            FormalMemoryCompletenessPolicyV4::RequireCompleteConflictFree
        );
        assert_eq!(
            evidence.completeness_status(),
            FormalMemoryCompletenessStatusV4::Complete
        );
        assert_eq!(evidence.static_conflict_count(), 0);
        assert_eq!(evidence.inter_invocation_conflict_count(), 0);
        assert!(!evidence.grants_authority());
        evidence.revalidate().unwrap();
        assert_eq!(
            InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(evidence.canonical_bytes())
                .unwrap(),
            evidence
        );

        for end in 0..evidence.canonical_bytes().len() {
            assert!(
                InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(
                    &evidence.canonical_bytes()[..end]
                )
                .is_err(),
                "accepted V4 formal-memory truncation at {end}"
            );
        }
        let mut trailing = evidence.canonical_bytes().to_vec();
        trailing.push(0);
        assert!(InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(&trailing).is_err());

        let mut unknown_kir_version = evidence.canonical_bytes().to_vec();
        put_u16(&mut unknown_kir_version, 20, 7);
        assert!(
            InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(&unknown_kir_version).is_err()
        );

        let mut zero_kir_length = evidence.canonical_bytes().to_vec();
        put_u64(&mut zero_kir_length, 24, 0);
        assert!(InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(&zero_kir_length).is_err());

        let mut wrong_receipt_identity = evidence.canonical_bytes().to_vec();
        wrong_receipt_identity[64] ^= 1;
        assert!(
            InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(&wrong_receipt_identity).is_err()
        );

        let mut wrong_witness = evidence.canonical_bytes().to_vec();
        put_u64(
            &mut wrong_witness,
            96,
            evidence.witness_invocation_count() + 1,
        );
        assert!(InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(&wrong_witness).is_err());

        let mut conflict = evidence.canonical_bytes().to_vec();
        put_u32(&mut conflict, 108, 1);
        assert!(InertCanonicalFormalMemoryAdmissionEvidenceV4::decode(&conflict).is_err());
    }

    #[test]
    fn correspondence_decoder_rejects_malformed_noncanonical_and_unbounded_inputs() {
        let (evidence, _) = evidence_pair();
        let canonical = evidence.canonical_bytes();
        for end in 0..canonical.len() {
            assert!(
                InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&canonical[..end]).is_err(),
                "accepted truncated prefix {end}",
            );
        }
        let mut trailing = canonical.to_vec();
        trailing.push(0);
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&trailing),
            Err(ProductionLineageEvidenceErrorV3::InvalidLength { .. })
        ));

        for offset in [SEMANTIC_SHA_OFFSET, KIR_IDENTITY_OFFSET] {
            let mut zero = canonical.to_vec();
            zero[offset..offset + 32].fill(0);
            assert!(matches!(
                InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&zero),
                Err(ProductionLineageEvidenceErrorV3::ZeroIdentity { .. })
            ));
        }

        let mut too_many_functions = canonical.to_vec();
        put_u32(
            &mut too_many_functions,
            FUNCTION_COUNT_OFFSET,
            MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3 as u32 + 1,
        );
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&too_many_functions),
            Err(ProductionLineageEvidenceErrorV3::LimitExceeded {
                field: "correspondence functions",
                ..
            })
        ));

        let mut too_many_blocks = canonical.to_vec();
        put_u32(
            &mut too_many_blocks,
            BLOCK_COUNT_OFFSET,
            MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3 as u32 + 1,
        );
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&too_many_blocks),
            Err(ProductionLineageEvidenceErrorV3::LimitExceeded {
                field: "correspondence blocks",
                ..
            })
        ));

        let mut noncanonical_order = canonical.to_vec();
        put_u32(&mut noncanonical_order, FIRST_BLOCK_OFFSET + 4, 1);
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&noncanonical_order),
            Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(_))
        ));

        let mut mismatched_target = canonical.to_vec();
        put_u32(&mut mismatched_target, FIRST_BLOCK_OFFSET + 8, 1);
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&mismatched_target),
            Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(_))
        ));

        let oversized = vec![0; MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V3 + 1];
        assert!(matches!(
            InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&oversized),
            Err(ProductionLineageEvidenceErrorV3::TooLarge {
                evidence: EvidenceKindV3::MirToKirCorrespondence,
                ..
            })
        ));
    }

    #[test]
    fn inert_correspondence_splice_changes_content_identity_without_gaining_authority() {
        let (original, _) = evidence_pair();
        let mut spliced = original.canonical_bytes().to_vec();
        spliced[SEMANTIC_SHA_OFFSET] ^= 0x80;
        let spliced = InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&spliced).unwrap();
        assert_ne!(spliced.semantic_sha256(), original.semantic_sha256());
        assert_ne!(spliced.identity(), original.identity());
        assert!(!spliced.grants_authority());
    }

    #[test]
    fn formal_decoder_rejects_malformed_policy_conflicts_and_nested_splices() {
        let (_, evidence) = evidence_pair();
        let canonical = evidence.canonical_bytes();
        for end in 0..canonical.len() {
            assert!(
                InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&canonical[..end]).is_err(),
                "accepted truncated prefix {end}",
            );
        }
        let mut trailing = canonical.to_vec();
        trailing.push(0);
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&trailing),
            Err(ProductionLineageEvidenceErrorV3::InvalidLength { .. })
        ));

        for offset in [FORMAL_KIR_IDENTITY_OFFSET, FORMAL_RECEIPT_IDENTITY_OFFSET] {
            let mut zero = canonical.to_vec();
            zero[offset..offset + 32].fill(0);
            assert!(matches!(
                InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&zero),
                Err(ProductionLineageEvidenceErrorV3::ZeroIdentity { .. })
            ));
        }

        let mut wrong_witness = canonical.to_vec();
        put_u64(
            &mut wrong_witness,
            FORMAL_WITNESS_OFFSET,
            PRODUCTION_FORMAL_MEMORY_WITNESS_EXTENT_V1 + 1,
        );
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&wrong_witness),
            Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(_))
        ));

        let mut unknown_policy = canonical.to_vec();
        put_u16(&mut unknown_policy, FORMAL_COMPLETENESS_POLICY_OFFSET, 2);
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&unknown_policy),
            Err(ProductionLineageEvidenceErrorV3::UnknownCompletenessPolicy(
                2
            ))
        ));
        let mut unknown_status = canonical.to_vec();
        put_u16(&mut unknown_status, FORMAL_COMPLETENESS_STATUS_OFFSET, 2);
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&unknown_status),
            Err(ProductionLineageEvidenceErrorV3::UnknownCompletenessStatus(
                2
            ))
        ));

        for offset in [
            FORMAL_STATIC_CONFLICT_OFFSET,
            FORMAL_INTER_INVOCATION_CONFLICT_OFFSET,
        ] {
            let mut conflicts = canonical.to_vec();
            put_u32(&mut conflicts, offset, 1);
            assert!(matches!(
                InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&conflicts),
                Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(_))
            ));
        }

        let mut mismatched_nested_identity = canonical.to_vec();
        mismatched_nested_identity[FORMAL_RECEIPT_IDENTITY_OFFSET] ^= 0x01;
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&mismatched_nested_identity),
            Err(ProductionLineageEvidenceErrorV3::NestedIdentityMismatch)
        ));

        let mut malformed_nested_receipt = canonical.to_vec();
        malformed_nested_receipt[FORMAL_RECEIPT_OFFSET] ^= 0x01;
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&malformed_nested_receipt),
            Err(ProductionLineageEvidenceErrorV3::FormalObligationReceipt(_))
        ));

        let mut impossible_nested_length = canonical.to_vec();
        put_u32(
            &mut impossible_nested_length,
            FORMAL_RECEIPT_LENGTH_OFFSET,
            u32::MAX,
        );
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&impossible_nested_length),
            Err(ProductionLineageEvidenceErrorV3::LimitExceeded { .. })
        ));

        let oversized = vec![0; MAX_FORMAL_MEMORY_ADMISSION_EVIDENCE_BYTES_V3 + 1];
        assert!(matches!(
            InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&oversized),
            Err(ProductionLineageEvidenceErrorV3::TooLarge {
                evidence: EvidenceKindV3::FormalMemoryAdmission,
                ..
            })
        ));
    }

    #[test]
    fn formal_evidence_rejects_a_valid_receipt_for_a_different_witness_extent() {
        let semantic_kir = semantic_kir_owner();
        let module = semantic_kir.module();
        let analysis = derive_kernel_memory_obligations(
            module,
            &module.kernels[0].id,
            ExplicitLaunchExtent1d::Exact(PRODUCTION_FORMAL_MEMORY_WITNESS_EXTENT_V1 + 1),
            FormalIndexWidth::Bits64,
        )
        .unwrap();
        let FormalMemoryObligationAnalysis::Complete(obligations) = analysis else {
            panic!("different exact witness should still produce complete obligations");
        };
        let receipt =
            InertCanonicalFormalMemoryObligationReceiptV1::from_obligations(&obligations).unwrap();
        let canonical_kir = VerifiedCanonicalKernelIrV5::from_module(module.clone()).unwrap();
        assert!(matches!(
            encode_formal_memory_admission(
                *canonical_kir.identity().digest(),
                *receipt.identity().digest(),
                PRODUCTION_FORMAL_MEMORY_WITNESS_EXTENT_V1,
                FormalMemoryCompletenessPolicyV3::RequireCompleteConflictFree,
                FormalMemoryCompletenessStatusV3::Complete,
                0,
                0,
                receipt.canonical_bytes(),
            ),
            Err(ProductionLineageEvidenceErrorV3::InvalidFormalAdmission(_))
        ));
    }

    #[test]
    fn inert_formal_kir_splice_changes_identity_without_gaining_authority() {
        let (_, original) = evidence_pair();
        let mut spliced = original.canonical_bytes().to_vec();
        spliced[FORMAL_KIR_IDENTITY_OFFSET] ^= 0x40;
        let spliced = InertCanonicalFormalMemoryAdmissionEvidenceV3::decode(&spliced).unwrap();
        assert_ne!(
            spliced.canonical_kir_v5_identity(),
            original.canonical_kir_v5_identity(),
        );
        assert_ne!(spliced.identity(), original.identity());
        assert!(!spliced.grants_authority());
    }

    #[test]
    fn correspondence_codec_counts_covered_functions_without_requiring_dense_semantic_ids() {
        let blocks = vec![
            MirToKirBlockCorrespondenceEvidenceV3 {
                semantic_function: 7,
                semantic_block: 0,
                kernel_ir_block: 0,
                source_statement_count: 3,
            },
            MirToKirBlockCorrespondenceEvidenceV3 {
                semantic_function: 42,
                semantic_block: 0,
                kernel_ir_block: 0,
                source_statement_count: 5,
            },
        ];
        let encoded = encode_correspondence(bytes(1), bytes(2), 2, &blocks).unwrap();
        let decoded = InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&encoded).unwrap();

        assert_eq!(decoded.function_count(), 2);
        assert_eq!(decoded.blocks(), blocks);
        assert!(matches!(
            encode_correspondence(bytes(1), bytes(2), 1, &blocks),
            Err(ProductionLineageEvidenceErrorV3::InvalidCorrespondence(_))
        ));
    }

    #[test]
    fn correspondence_codec_accepts_exact_maximum_and_rejects_one_more_record() {
        let mut blocks = Vec::with_capacity(MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3);
        let blocks_per_function =
            MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3 / MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3;
        for function in 0..MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3 as u32 {
            for block in 0..blocks_per_function as u32 {
                blocks.push(MirToKirBlockCorrespondenceEvidenceV3 {
                    semantic_function: function,
                    semantic_block: block,
                    kernel_ir_block: block,
                    source_statement_count: block,
                });
            }
        }
        let maximum = encode_correspondence(
            bytes(1),
            bytes(2),
            MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3 as u32,
            &blocks,
        )
        .unwrap();
        assert!(maximum.len() <= MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V3);
        let decoded = InertCanonicalMirToKirCorrespondenceEvidenceV3::decode(&maximum).unwrap();
        assert_eq!(
            decoded.blocks().len(),
            MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3
        );

        blocks.push(MirToKirBlockCorrespondenceEvidenceV3 {
            semantic_function: MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3 as u32 - 1,
            semantic_block: blocks_per_function as u32,
            kernel_ir_block: blocks_per_function as u32,
            source_statement_count: 0,
        });
        assert!(matches!(
            encode_correspondence(
                bytes(1),
                bytes(2),
                MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3 as u32,
                &blocks,
            ),
            Err(ProductionLineageEvidenceErrorV3::LimitExceeded {
                field: "correspondence blocks",
                ..
            })
        ));
    }
}

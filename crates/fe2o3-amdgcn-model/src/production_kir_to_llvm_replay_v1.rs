use std::error::Error;
use std::fmt;

use fe2o3_amd_target::ProductionAmdTargetProfileV1;
use fe2o3_kernel_ir::{
    KERNEL_IR_MAGIC_V1, KERNEL_IR_VERSION_V8, KERNEL_IR_VERSION_V9, KernelId, MAX_MODULE_BYTES_V1,
    MAX_TEXT_BYTES_V1, Module, VerifiedCanonicalKernelIrErrorV8, VerifiedCanonicalKernelIrErrorV9,
    VerifiedCanonicalKernelIrV8, VerifiedCanonicalKernelIrV9,
};
use sha2::{Digest, Sha256};

use crate::{
    LoweringErrors, ProductionLlvmLayoutBindingErrorV1, ProductionTargetBindingErrorV1,
    bind_production_target_v1, bind_production_upstream_llvm_layout_v1,
    lower_kernel_to_gfx942_xnack_minus_llvm_ir, lower_kernel_to_gfx950_xnack_minus_llvm_ir,
};

const EVIDENCE_MAGIC_V1: &[u8] = b"FE2O3/KIR-TO-LLVM-REPLAY/V1\0";
const EVIDENCE_VERSION_V1: u16 = 1;
const EXACT_DETERMINISTIC_REPLAY_CLAIM_V1: u8 = 1;
const RESERVED_V1: u8 = 0;
const GFX942_PROFILE_TAG_V1: u8 = 1;
const GFX950_PROFILE_TAG_V1: u8 = 2;
const KIR_V8_TAG_V1: u8 = 8;
const KIR_V9_TAG_V1: u8 = 9;
const IDENTITY_DOMAIN_V1: &[u8] = b"FE2O3/KIR-TO-LLVM-REPLAY/IDENTITY/V1\0";

/// The frozen outer lineage receipt admits at most four MiB per preimage.
pub const MAX_PRODUCTION_KIR_TO_LLVM_REPLAY_EVIDENCE_BYTES_V1: usize = 4 * 1024 * 1024;
pub const MAX_PRODUCTION_PRE_DESCRIPTOR_LLVM_BYTES_V1: usize =
    MAX_PRODUCTION_KIR_TO_LLVM_REPLAY_EVIDENCE_BYTES_V1 - 256;

/// Exact canonical Kernel IR version carried by replay evidence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProductionReplayKernelIrVersionV1 {
    V8,
    V9,
}

impl ProductionReplayKernelIrVersionV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::V8 => KIR_V8_TAG_V1,
            Self::V9 => KIR_V9_TAG_V1,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, ProductionKirToLlvmReplayErrorV1> {
        match tag {
            KIR_V8_TAG_V1 => Ok(Self::V8),
            KIR_V9_TAG_V1 => Ok(Self::V9),
            _ => Err(ProductionKirToLlvmReplayErrorV1::InvalidHeader),
        }
    }
}

/// Versioned identity of exact canonical Kernel IR bytes used by replay.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProductionReplayKernelIrIdentityV1 {
    version: ProductionReplayKernelIrVersionV1,
    sha256: [u8; 32],
    byte_len: u64,
}

impl ProductionReplayKernelIrIdentityV1 {
    pub const fn version(self) -> ProductionReplayKernelIrVersionV1 {
        self.version
    }

    pub const fn sha256(self) -> [u8; 32] {
        self.sha256
    }

    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// Content identity of one canonical replay-evidence record.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProductionKirToLlvmReplayEvidenceIdentityV1 {
    sha256: [u8; 32],
    byte_len: u64,
}

impl ProductionKirToLlvmReplayEvidenceIdentityV1 {
    pub const fn sha256(self) -> [u8; 32] {
        self.sha256
    }

    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// Inert canonical evidence for deterministic production KIR-to-LLVM replay.
///
/// Decoding establishes bounded canonical structure only. Call
/// [`validate_against_neutral_kernel_ir`](Self::validate_against_neutral_kernel_ir)
/// to independently reconstruct target-bound KIR and exact LLVM.
#[derive(Debug, Eq, PartialEq)]
pub struct CanonicalProductionKirToLlvmReplayEvidenceV1 {
    canonical_bytes: Box<[u8]>,
    identity: ProductionKirToLlvmReplayEvidenceIdentityV1,
    profile: ProductionAmdTargetProfileV1,
    neutral_kernel_ir: ProductionReplayKernelIrIdentityV1,
    target_bound_kernel_ir: ProductionReplayKernelIrIdentityV1,
    kernel_id: KernelId,
    pre_descriptor_llvm: Box<str>,
}

impl CanonicalProductionKirToLlvmReplayEvidenceV1 {
    /// Constructs evidence only after replaying and comparing every live input.
    pub fn from_live_inputs(
        neutral_kernel_ir: &[u8],
        target_bound_module: &Module,
        profile: ProductionAmdTargetProfileV1,
        pre_descriptor_llvm: &str,
    ) -> Result<Self, ProductionKirToLlvmReplayErrorV1> {
        let version = infer_kernel_ir_version(neutral_kernel_ir)?;
        let (_, neutral_module, neutral_identity) =
            decode_exact_kernel_ir(neutral_kernel_ir, version)?;
        let target_bound = bind_production_target_v1(&neutral_module, profile)
            .map_err(ProductionKirToLlvmReplayErrorV1::TargetBinding)?;
        if target_bound.module() != target_bound_module {
            return Err(ProductionKirToLlvmReplayErrorV1::LiveTargetModuleMismatch);
        }
        let kernel_id = target_bound.kernel_id().clone();
        let (_, target_identity) = canonicalize_target_module(target_bound.module(), version)?;
        let replayed_llvm = replay_llvm(target_bound.module(), &kernel_id, profile)?;
        if replayed_llvm.as_bytes() != pre_descriptor_llvm.as_bytes() {
            return Err(ProductionKirToLlvmReplayErrorV1::LlvmMismatch);
        }

        let canonical_bytes = encode_evidence(
            profile,
            neutral_identity,
            target_identity,
            &kernel_id,
            pre_descriptor_llvm,
        )?;
        let evidence = Self::decode(&canonical_bytes)?;
        let validated = evidence.validate_against_neutral_kernel_ir(neutral_kernel_ir)?;
        if validated.target_bound_module() != target_bound_module {
            return Err(ProductionKirToLlvmReplayErrorV1::LiveTargetModuleMismatch);
        }
        Ok(validated.into_evidence())
    }

    /// Strictly decodes and byte-for-byte re-encodes one bounded evidence record.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProductionKirToLlvmReplayErrorV1> {
        if bytes.len() > MAX_PRODUCTION_KIR_TO_LLVM_REPLAY_EVIDENCE_BYTES_V1 {
            return Err(ProductionKirToLlvmReplayErrorV1::TooLarge);
        }
        let mut reader = Reader::new(bytes);
        if reader.take(EVIDENCE_MAGIC_V1.len())? != EVIDENCE_MAGIC_V1
            || reader.u16()? != EVIDENCE_VERSION_V1
            || reader.u8()? != EXACT_DETERMINISTIC_REPLAY_CLAIM_V1
        {
            return Err(ProductionKirToLlvmReplayErrorV1::InvalidHeader);
        }
        let profile = decode_profile(reader.u8()?)?;
        let version = ProductionReplayKernelIrVersionV1::from_tag(reader.u8()?)?;
        if reader.u8()? != RESERVED_V1 {
            return Err(ProductionKirToLlvmReplayErrorV1::InvalidHeader);
        }
        let neutral_kernel_ir = decode_kernel_ir_identity(&mut reader, version)?;
        let target_bound_kernel_ir = decode_kernel_ir_identity(&mut reader, version)?;
        let kernel_id_length = reader.usize_u32()?;
        if kernel_id_length == 0 || kernel_id_length > MAX_TEXT_BYTES_V1 {
            return Err(ProductionKirToLlvmReplayErrorV1::InvalidLength);
        }
        let kernel_id = std::str::from_utf8(reader.take(kernel_id_length)?)
            .map_err(|_| ProductionKirToLlvmReplayErrorV1::InvalidUtf8)?;
        let llvm_length = reader.usize_u32()?;
        if llvm_length == 0 || llvm_length > MAX_PRODUCTION_PRE_DESCRIPTOR_LLVM_BYTES_V1 {
            return Err(ProductionKirToLlvmReplayErrorV1::InvalidLength);
        }
        let pre_descriptor_llvm = std::str::from_utf8(reader.take(llvm_length)?)
            .map_err(|_| ProductionKirToLlvmReplayErrorV1::InvalidUtf8)?;
        if pre_descriptor_llvm.as_bytes().contains(&0) {
            return Err(ProductionKirToLlvmReplayErrorV1::InvalidUtf8);
        }
        reader.finish()?;

        let canonical_bytes = encode_evidence(
            profile,
            neutral_kernel_ir,
            target_bound_kernel_ir,
            &KernelId::new(kernel_id),
            pre_descriptor_llvm,
        )?;
        if canonical_bytes.as_slice() != bytes {
            return Err(ProductionKirToLlvmReplayErrorV1::NonCanonical);
        }
        let identity = evidence_identity(&canonical_bytes);
        Ok(Self {
            canonical_bytes: canonical_bytes.into_boxed_slice(),
            identity,
            profile,
            neutral_kernel_ir,
            target_bound_kernel_ir,
            kernel_id: KernelId::new(kernel_id),
            pre_descriptor_llvm: pre_descriptor_llvm.to_owned().into_boxed_str(),
        })
    }

    /// Reconstructs target-bound KIR and LLVM from independently supplied neutral KIR.
    pub fn validate_against_neutral_kernel_ir(
        self,
        neutral_kernel_ir: &[u8],
    ) -> Result<ValidatedProductionKirToLlvmReplayV1, ProductionKirToLlvmReplayErrorV1> {
        let version = self.neutral_kernel_ir.version;
        let (neutral_owner, neutral_module, neutral_identity) =
            decode_exact_kernel_ir(neutral_kernel_ir, version)?;
        if neutral_identity != self.neutral_kernel_ir {
            return Err(ProductionKirToLlvmReplayErrorV1::IdentityMismatch {
                field: "neutral Kernel IR",
            });
        }

        let target_bound = bind_production_target_v1(&neutral_module, self.profile)
            .map_err(ProductionKirToLlvmReplayErrorV1::TargetBinding)?;
        if target_bound.kernel_id() != &self.kernel_id {
            return Err(ProductionKirToLlvmReplayErrorV1::KernelIdMismatch);
        }
        let (target_owner, target_identity) =
            canonicalize_target_module(target_bound.module(), version)?;
        if target_identity != self.target_bound_kernel_ir {
            return Err(ProductionKirToLlvmReplayErrorV1::IdentityMismatch {
                field: "target-bound Kernel IR",
            });
        }
        let replayed_llvm = replay_llvm(target_bound.module(), &self.kernel_id, self.profile)?;
        if replayed_llvm.as_bytes() != self.pre_descriptor_llvm.as_bytes() {
            return Err(ProductionKirToLlvmReplayErrorV1::LlvmMismatch);
        }
        let (target_bound_module, _) = target_bound.into_parts();
        Ok(ValidatedProductionKirToLlvmReplayV1 {
            evidence: self,
            neutral_owner,
            target_owner,
            target_bound_module,
        })
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub const fn identity(&self) -> ProductionKirToLlvmReplayEvidenceIdentityV1 {
        self.identity
    }

    pub const fn profile(&self) -> ProductionAmdTargetProfileV1 {
        self.profile
    }

    pub const fn neutral_kernel_ir_identity(&self) -> ProductionReplayKernelIrIdentityV1 {
        self.neutral_kernel_ir
    }

    pub const fn target_bound_kernel_ir_identity(&self) -> ProductionReplayKernelIrIdentityV1 {
        self.target_bound_kernel_ir
    }

    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    pub fn pre_descriptor_llvm(&self) -> &str {
        &self.pre_descriptor_llvm
    }

    pub const fn grants_runtime_authority(&self) -> bool {
        false
    }
}

/// Independently reconstructed exact production KIR-to-LLVM derivation.
#[derive(Debug)]
#[must_use = "dropping validated replay abandons exact KIR-to-LLVM custody"]
pub struct ValidatedProductionKirToLlvmReplayV1 {
    evidence: CanonicalProductionKirToLlvmReplayEvidenceV1,
    neutral_owner: ExactKernelIrOwnerV1,
    target_owner: ExactKernelIrOwnerV1,
    target_bound_module: Module,
}

impl ValidatedProductionKirToLlvmReplayV1 {
    pub const fn evidence(&self) -> &CanonicalProductionKirToLlvmReplayEvidenceV1 {
        &self.evidence
    }

    pub fn neutral_kernel_ir_bytes(&self) -> &[u8] {
        self.neutral_owner.canonical_bytes()
    }

    pub fn target_bound_kernel_ir_bytes(&self) -> &[u8] {
        self.target_owner.canonical_bytes()
    }

    pub const fn target_bound_module(&self) -> &Module {
        &self.target_bound_module
    }

    pub const fn has_exact_target_binding_replay(&self) -> bool {
        true
    }

    pub const fn has_exact_kir_to_llvm_replay(&self) -> bool {
        true
    }

    pub const fn establishes_formal_semantic_refinement(&self) -> bool {
        false
    }

    pub const fn grants_object_or_runtime_authority(&self) -> bool {
        false
    }

    fn into_evidence(self) -> CanonicalProductionKirToLlvmReplayEvidenceV1 {
        self.evidence
    }
}

#[derive(Debug)]
enum ExactKernelIrOwnerV1 {
    V8(VerifiedCanonicalKernelIrV8),
    V9(VerifiedCanonicalKernelIrV9),
}

impl ExactKernelIrOwnerV1 {
    fn canonical_bytes(&self) -> &[u8] {
        match self {
            Self::V8(owner) => owner.canonical_bytes(),
            Self::V9(owner) => owner.canonical_bytes(),
        }
    }

    fn identity(&self) -> ProductionReplayKernelIrIdentityV1 {
        match self {
            Self::V8(owner) => ProductionReplayKernelIrIdentityV1 {
                version: ProductionReplayKernelIrVersionV1::V8,
                sha256: *owner.identity().digest(),
                byte_len: owner.identity().canonical_length(),
            },
            Self::V9(owner) => ProductionReplayKernelIrIdentityV1 {
                version: ProductionReplayKernelIrVersionV1::V9,
                sha256: *owner.identity().digest(),
                byte_len: owner.identity().canonical_length(),
            },
        }
    }
}

fn infer_kernel_ir_version(
    bytes: &[u8],
) -> Result<ProductionReplayKernelIrVersionV1, ProductionKirToLlvmReplayErrorV1> {
    if bytes.len() > MAX_MODULE_BYTES_V1 {
        return Err(ProductionKirToLlvmReplayErrorV1::TooLarge);
    }
    if bytes.get(..KERNEL_IR_MAGIC_V1.len()) != Some(KERNEL_IR_MAGIC_V1.as_slice()) {
        return Err(ProductionKirToLlvmReplayErrorV1::InvalidKernelIrHeader);
    }
    let version_offset = KERNEL_IR_MAGIC_V1.len();
    let version_bytes = bytes
        .get(version_offset..version_offset + 2)
        .ok_or(ProductionKirToLlvmReplayErrorV1::Truncated)?;
    match u16::from_le_bytes([version_bytes[0], version_bytes[1]]) {
        KERNEL_IR_VERSION_V8 => Ok(ProductionReplayKernelIrVersionV1::V8),
        KERNEL_IR_VERSION_V9 => Ok(ProductionReplayKernelIrVersionV1::V9),
        _ => Err(ProductionKirToLlvmReplayErrorV1::InvalidKernelIrHeader),
    }
}

fn decode_exact_kernel_ir(
    bytes: &[u8],
    version: ProductionReplayKernelIrVersionV1,
) -> Result<
    (
        ExactKernelIrOwnerV1,
        Module,
        ProductionReplayKernelIrIdentityV1,
    ),
    ProductionKirToLlvmReplayErrorV1,
> {
    let (owner, module) = match version {
        ProductionReplayKernelIrVersionV1::V8 => {
            let (owner, module) =
                VerifiedCanonicalKernelIrV8::from_canonical_bytes_with_module(bytes.to_vec())
                    .map_err(ProductionKirToLlvmReplayErrorV1::KernelIrV8)?;
            (ExactKernelIrOwnerV1::V8(owner), module)
        }
        ProductionReplayKernelIrVersionV1::V9 => {
            let (owner, module) =
                VerifiedCanonicalKernelIrV9::from_canonical_bytes_with_module(bytes.to_vec())
                    .map_err(ProductionKirToLlvmReplayErrorV1::KernelIrV9)?;
            (ExactKernelIrOwnerV1::V9(owner), module)
        }
    };
    let identity = owner.identity();
    Ok((owner, module, identity))
}

fn canonicalize_target_module(
    module: &Module,
    version: ProductionReplayKernelIrVersionV1,
) -> Result<
    (ExactKernelIrOwnerV1, ProductionReplayKernelIrIdentityV1),
    ProductionKirToLlvmReplayErrorV1,
> {
    let owner = match version {
        ProductionReplayKernelIrVersionV1::V8 => ExactKernelIrOwnerV1::V8(
            VerifiedCanonicalKernelIrV8::from_module(module.clone())
                .map_err(ProductionKirToLlvmReplayErrorV1::KernelIrV8)?,
        ),
        ProductionReplayKernelIrVersionV1::V9 => ExactKernelIrOwnerV1::V9(
            VerifiedCanonicalKernelIrV9::from_module(module.clone())
                .map_err(ProductionKirToLlvmReplayErrorV1::KernelIrV9)?,
        ),
    };
    let identity = owner.identity();
    Ok((owner, identity))
}

fn replay_llvm(
    target_bound_module: &Module,
    kernel_id: &KernelId,
    profile: ProductionAmdTargetProfileV1,
) -> Result<String, ProductionKirToLlvmReplayErrorV1> {
    let dialect_llvm = match profile {
        ProductionAmdTargetProfileV1::Gfx942 => {
            lower_kernel_to_gfx942_xnack_minus_llvm_ir(target_bound_module, kernel_id)
        }
        ProductionAmdTargetProfileV1::Gfx950 => {
            lower_kernel_to_gfx950_xnack_minus_llvm_ir(target_bound_module, kernel_id)
        }
    }
    .map_err(ProductionKirToLlvmReplayErrorV1::TargetLowering)?;
    bind_production_upstream_llvm_layout_v1(&dialect_llvm)
        .map_err(ProductionKirToLlvmReplayErrorV1::LayoutBinding)
}

fn encode_evidence(
    profile: ProductionAmdTargetProfileV1,
    neutral_kernel_ir: ProductionReplayKernelIrIdentityV1,
    target_bound_kernel_ir: ProductionReplayKernelIrIdentityV1,
    kernel_id: &KernelId,
    pre_descriptor_llvm: &str,
) -> Result<Vec<u8>, ProductionKirToLlvmReplayErrorV1> {
    if neutral_kernel_ir.version != target_bound_kernel_ir.version
        || neutral_kernel_ir.byte_len == 0
        || target_bound_kernel_ir.byte_len == 0
        || neutral_kernel_ir.byte_len > MAX_MODULE_BYTES_V1 as u64
        || target_bound_kernel_ir.byte_len > MAX_MODULE_BYTES_V1 as u64
        || neutral_kernel_ir.sha256 == [0; 32]
        || target_bound_kernel_ir.sha256 == [0; 32]
        || kernel_id.as_str().is_empty()
        || kernel_id.as_str().len() > MAX_TEXT_BYTES_V1
        || pre_descriptor_llvm.is_empty()
        || pre_descriptor_llvm.len() > MAX_PRODUCTION_PRE_DESCRIPTOR_LLVM_BYTES_V1
        || pre_descriptor_llvm.as_bytes().contains(&0)
    {
        return Err(ProductionKirToLlvmReplayErrorV1::InvalidLength);
    }
    let total = EVIDENCE_MAGIC_V1
        .len()
        .checked_add(2 + 4 + 32 + 8 + 32 + 8 + 4 + 4)
        .and_then(|value| value.checked_add(kernel_id.as_str().len()))
        .and_then(|value| value.checked_add(pre_descriptor_llvm.len()))
        .ok_or(ProductionKirToLlvmReplayErrorV1::Overflow)?;
    if total > MAX_PRODUCTION_KIR_TO_LLVM_REPLAY_EVIDENCE_BYTES_V1 {
        return Err(ProductionKirToLlvmReplayErrorV1::TooLarge);
    }
    let kernel_length = u32::try_from(kernel_id.as_str().len())
        .map_err(|_| ProductionKirToLlvmReplayErrorV1::Overflow)?;
    let llvm_length = u32::try_from(pre_descriptor_llvm.len())
        .map_err(|_| ProductionKirToLlvmReplayErrorV1::Overflow)?;
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(EVIDENCE_MAGIC_V1);
    bytes.extend_from_slice(&EVIDENCE_VERSION_V1.to_le_bytes());
    bytes.push(EXACT_DETERMINISTIC_REPLAY_CLAIM_V1);
    bytes.push(encode_profile(profile));
    bytes.push(neutral_kernel_ir.version.tag());
    bytes.push(RESERVED_V1);
    encode_kernel_ir_identity(&mut bytes, neutral_kernel_ir);
    encode_kernel_ir_identity(&mut bytes, target_bound_kernel_ir);
    bytes.extend_from_slice(&kernel_length.to_le_bytes());
    bytes.extend_from_slice(kernel_id.as_str().as_bytes());
    bytes.extend_from_slice(&llvm_length.to_le_bytes());
    bytes.extend_from_slice(pre_descriptor_llvm.as_bytes());
    if bytes.len() != total {
        return Err(ProductionKirToLlvmReplayErrorV1::InvalidLength);
    }
    Ok(bytes)
}

const fn encode_profile(profile: ProductionAmdTargetProfileV1) -> u8 {
    match profile {
        ProductionAmdTargetProfileV1::Gfx942 => GFX942_PROFILE_TAG_V1,
        ProductionAmdTargetProfileV1::Gfx950 => GFX950_PROFILE_TAG_V1,
    }
}

fn decode_profile(
    tag: u8,
) -> Result<ProductionAmdTargetProfileV1, ProductionKirToLlvmReplayErrorV1> {
    match tag {
        GFX942_PROFILE_TAG_V1 => Ok(ProductionAmdTargetProfileV1::Gfx942),
        GFX950_PROFILE_TAG_V1 => Ok(ProductionAmdTargetProfileV1::Gfx950),
        _ => Err(ProductionKirToLlvmReplayErrorV1::InvalidHeader),
    }
}

fn encode_kernel_ir_identity(bytes: &mut Vec<u8>, identity: ProductionReplayKernelIrIdentityV1) {
    bytes.extend_from_slice(&identity.sha256);
    bytes.extend_from_slice(&identity.byte_len.to_le_bytes());
}

fn decode_kernel_ir_identity(
    reader: &mut Reader<'_>,
    version: ProductionReplayKernelIrVersionV1,
) -> Result<ProductionReplayKernelIrIdentityV1, ProductionKirToLlvmReplayErrorV1> {
    let sha256 = reader.fixed::<32>()?;
    let byte_len = reader.u64()?;
    if sha256 == [0; 32] || byte_len == 0 || byte_len > MAX_MODULE_BYTES_V1 as u64 {
        return Err(ProductionKirToLlvmReplayErrorV1::InvalidLength);
    }
    Ok(ProductionReplayKernelIrIdentityV1 {
        version,
        sha256,
        byte_len,
    })
}

fn evidence_identity(bytes: &[u8]) -> ProductionKirToLlvmReplayEvidenceIdentityV1 {
    let mut digest = Sha256::new();
    digest.update((IDENTITY_DOMAIN_V1.len() as u32).to_le_bytes());
    digest.update(IDENTITY_DOMAIN_V1);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    ProductionKirToLlvmReplayEvidenceIdentityV1 {
        sha256: digest.finalize().into(),
        byte_len: bytes.len() as u64,
    }
}

/// Closed failures for canonical replay evidence and independent reconstruction.
#[derive(Debug)]
pub enum ProductionKirToLlvmReplayErrorV1 {
    TooLarge,
    Overflow,
    Truncated,
    TrailingBytes,
    InvalidHeader,
    InvalidKernelIrHeader,
    InvalidLength,
    InvalidUtf8,
    NonCanonical,
    IdentityMismatch { field: &'static str },
    KernelIdMismatch,
    LiveTargetModuleMismatch,
    LlvmMismatch,
    KernelIrV8(VerifiedCanonicalKernelIrErrorV8),
    KernelIrV9(VerifiedCanonicalKernelIrErrorV9),
    TargetBinding(ProductionTargetBindingErrorV1),
    TargetLowering(LoweringErrors),
    LayoutBinding(ProductionLlvmLayoutBindingErrorV1),
}

impl fmt::Display for ProductionKirToLlvmReplayErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => formatter
                .write_str("production KIR-to-LLVM replay evidence exceeds its hard byte bound"),
            Self::Overflow => {
                formatter.write_str("production KIR-to-LLVM replay evidence length overflowed")
            }
            Self::Truncated => {
                formatter.write_str("production KIR-to-LLVM replay evidence is truncated")
            }
            Self::TrailingBytes => {
                formatter.write_str("production KIR-to-LLVM replay evidence has trailing bytes")
            }
            Self::InvalidHeader => {
                formatter.write_str("production KIR-to-LLVM replay evidence has an invalid header")
            }
            Self::InvalidKernelIrHeader => formatter.write_str(
                "production KIR-to-LLVM replay input is not exact canonical KIR V8 or V9",
            ),
            Self::InvalidLength => formatter
                .write_str("production KIR-to-LLVM replay evidence has an invalid bounded length"),
            Self::InvalidUtf8 => {
                formatter.write_str("production KIR-to-LLVM replay evidence contains invalid text")
            }
            Self::NonCanonical => {
                formatter.write_str("production KIR-to-LLVM replay evidence is not canonical")
            }
            Self::IdentityMismatch { field } => write!(
                formatter,
                "production KIR-to-LLVM replay changed the exact {field} identity"
            ),
            Self::KernelIdMismatch => formatter
                .write_str("production KIR-to-LLVM replay changed the exact kernel identity"),
            Self::LiveTargetModuleMismatch => formatter
                .write_str("live target-bound KIR differs from deterministic target replay"),
            Self::LlvmMismatch => {
                formatter.write_str("retained LLVM differs from deterministic KIR lowering replay")
            }
            Self::KernelIrV8(error) => write!(formatter, "exact KIR V8 validation failed: {error}"),
            Self::KernelIrV9(error) => write!(formatter, "exact KIR V9 validation failed: {error}"),
            Self::TargetBinding(error) => {
                write!(formatter, "target-binding replay failed: {error}")
            }
            Self::TargetLowering(error) => {
                write!(formatter, "AMDGPU lowering replay failed: {error}")
            }
            Self::LayoutBinding(error) => {
                write!(formatter, "upstream LLVM layout replay failed: {error}")
            }
        }
    }
}

impl Error for ProductionKirToLlvmReplayErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::KernelIrV8(error) => Some(error),
            Self::KernelIrV9(error) => Some(error),
            Self::TargetBinding(error) => Some(error),
            Self::TargetLowering(error) => Some(error),
            Self::LayoutBinding(error) => Some(error),
            _ => None,
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ProductionKirToLlvmReplayErrorV1> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ProductionKirToLlvmReplayErrorV1::Overflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ProductionKirToLlvmReplayErrorV1::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], ProductionKirToLlvmReplayErrorV1> {
        self.take(N)?
            .try_into()
            .map_err(|_| ProductionKirToLlvmReplayErrorV1::Truncated)
    }

    fn u8(&mut self) -> Result<u8, ProductionKirToLlvmReplayErrorV1> {
        Ok(self.fixed::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, ProductionKirToLlvmReplayErrorV1> {
        Ok(u16::from_le_bytes(self.fixed()?))
    }

    fn u32(&mut self) -> Result<u32, ProductionKirToLlvmReplayErrorV1> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    fn u64(&mut self) -> Result<u64, ProductionKirToLlvmReplayErrorV1> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    fn usize_u32(&mut self) -> Result<usize, ProductionKirToLlvmReplayErrorV1> {
        usize::try_from(self.u32()?).map_err(|_| ProductionKirToLlvmReplayErrorV1::Overflow)
    }

    fn finish(self) -> Result<(), ProductionKirToLlvmReplayErrorV1> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(ProductionKirToLlvmReplayErrorV1::TrailingBytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use fe2o3_kernel_ir::{
        BasicBlock, BlockId, Function, LaunchDomain, LaunchExtent, Module, Signature, Terminator,
        VerifiedCanonicalKernelIrV8, WorkgroupSize,
    };

    use super::*;

    fn neutral_module(name: &str) -> Module {
        let mut block = BasicBlock::new(BlockId(0));
        block.terminator = Some(Terminator::Return { values: vec![] });
        let function = Function::kernel_entry(
            format!("{name}_entry"),
            Signature::new(vec![], vec![]),
            vec![],
            vec![block],
        );
        let mut kernel = fe2o3_kernel_ir::Kernel::new(
            format!("{name}_kernel"),
            format!("{name}_entry"),
            LaunchDomain::D1 {
                x: LaunchExtent::Static(1),
            },
        );
        kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
        let mut module = Module::new(name);
        module.functions.push(function);
        module.kernels.push(kernel);
        module
    }

    fn fixture(
        name: &str,
    ) -> (
        Vec<u8>,
        Module,
        String,
        CanonicalProductionKirToLlvmReplayEvidenceV1,
    ) {
        let neutral = VerifiedCanonicalKernelIrV8::from_module(neutral_module(name)).unwrap();
        let neutral_bytes = neutral.into_canonical_bytes();
        let (_, neutral_module, _) =
            decode_exact_kernel_ir(&neutral_bytes, ProductionReplayKernelIrVersionV1::V8).unwrap();
        let target =
            bind_production_target_v1(&neutral_module, ProductionAmdTargetProfileV1::Gfx942)
                .unwrap();
        let llvm = replay_llvm(
            target.module(),
            target.kernel_id(),
            ProductionAmdTargetProfileV1::Gfx942,
        )
        .unwrap();
        let target_module = target.module().clone();
        let evidence = CanonicalProductionKirToLlvmReplayEvidenceV1::from_live_inputs(
            &neutral_bytes,
            &target_module,
            ProductionAmdTargetProfileV1::Gfx942,
            &llvm,
        )
        .unwrap();
        (neutral_bytes, target_module, llvm, evidence)
    }

    #[test]
    fn exact_replay_round_trips_and_retains_no_later_authority() {
        let (neutral, target_module, llvm, evidence) = fixture("replay");
        let identity = evidence.identity();
        let decoded =
            CanonicalProductionKirToLlvmReplayEvidenceV1::decode(evidence.canonical_bytes())
                .unwrap();
        assert_eq!(decoded.identity(), identity);
        assert_eq!(decoded.pre_descriptor_llvm(), llvm);
        assert!(!decoded.grants_runtime_authority());
        let validated = decoded
            .validate_against_neutral_kernel_ir(&neutral)
            .unwrap();
        assert_eq!(validated.target_bound_module(), &target_module);
        assert!(validated.has_exact_target_binding_replay());
        assert!(validated.has_exact_kir_to_llvm_replay());
        assert!(!validated.establishes_formal_semantic_refinement());
        assert!(!validated.grants_object_or_runtime_authority());
    }

    #[test]
    fn replay_rejects_hostile_evidence_framing_and_llvm_mutation() {
        let (neutral, _, _, evidence) = fixture("hostile");
        let canonical = evidence.canonical_bytes();
        for prefix in 0..canonical.len() {
            assert!(
                CanonicalProductionKirToLlvmReplayEvidenceV1::decode(&canonical[..prefix]).is_err()
            );
        }
        let mut trailing = canonical.to_vec();
        trailing.push(0);
        assert!(CanonicalProductionKirToLlvmReplayEvidenceV1::decode(&trailing).is_err());

        let mut mutated = canonical.to_vec();
        let llvm = evidence.pre_descriptor_llvm().as_bytes();
        let llvm_start = mutated.len() - llvm.len();
        let return_offset = llvm
            .windows(8)
            .position(|window| window == b"ret void")
            .expect("fixture LLVM contains a return");
        mutated[llvm_start + return_offset] = b'R';
        let decoded = CanonicalProductionKirToLlvmReplayEvidenceV1::decode(&mutated).unwrap();
        assert!(matches!(
            decoded.validate_against_neutral_kernel_ir(&neutral),
            Err(ProductionKirToLlvmReplayErrorV1::LlvmMismatch)
        ));
    }

    #[test]
    fn replay_rejects_cross_spliced_neutral_kir_and_live_target_module() {
        let (neutral_a, target_a, llvm_a, evidence_a) = fixture("alpha");
        let (neutral_b, _, _, _) = fixture("beta");
        assert!(matches!(
            evidence_a.validate_against_neutral_kernel_ir(&neutral_b),
            Err(ProductionKirToLlvmReplayErrorV1::IdentityMismatch {
                field: "neutral Kernel IR"
            })
        ));

        let mut hostile_target = target_a;
        hostile_target.id = "substituted".into();
        assert!(matches!(
            CanonicalProductionKirToLlvmReplayEvidenceV1::from_live_inputs(
                &neutral_a,
                &hostile_target,
                ProductionAmdTargetProfileV1::Gfx942,
                &llvm_a,
            ),
            Err(ProductionKirToLlvmReplayErrorV1::LiveTargetModuleMismatch)
        ));
    }

    #[test]
    fn replay_rejects_oversized_input_before_field_allocation() {
        let oversized = vec![0; MAX_PRODUCTION_KIR_TO_LLVM_REPLAY_EVIDENCE_BYTES_V1 + 1];
        assert!(matches!(
            CanonicalProductionKirToLlvmReplayEvidenceV1::decode(&oversized),
            Err(ProductionKirToLlvmReplayErrorV1::TooLarge)
        ));
    }
}

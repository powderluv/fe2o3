//! Lossless canonical semantic-proof and MIR-to-KIR correspondence custody.

use std::{collections::BTreeSet, error::Error, fmt};

use fe2o3_mir_model::{
    InertCanonicalSemanticU32InductionEvidenceV1, SemanticU32InductionEvidenceErrorV1,
    SemanticU32InductionNoOverflowReportV1,
};
use sha2::{Digest, Sha256};

use crate::{
    MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3, MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3,
    ProductionCanonicalKernelIrIdentityV1, ProductionCanonicalKernelIrVersionV1,
    ProductionSemanticKirOwnerV1, SemanticKirSyntheticOperationRuleV1,
};

/// Current wire version for lossless semantic-proof and MIR-to-KIR custody.
pub const MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_VERSION_V4: u16 = 4;
/// Closed validation policy for lossless correspondence custody.
pub const MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_POLICY_V4: u16 = 1;
/// Maximum aggregate bytes, matching the outer non-MIR lineage receipt budget.
pub const MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V4: usize = 4 * 1024 * 1024;
/// Maximum exact semantic statement spans.
pub const MAX_MIR_TO_KIR_STATEMENT_SPANS_V4: usize = 262_144;
/// Maximum exact synthetic spans.
pub const MAX_MIR_TO_KIR_SYNTHETIC_SPANS_V4: usize = 16_384;
/// Maximum exact parameter bindings.
pub const MAX_MIR_TO_KIR_PARAMETER_BINDINGS_V4: usize = 65_536;

const MAGIC_V4: [u8; 8] = *b"F2M2K4\0\0";
const IDENTITY_DOMAIN_V4: &[u8] = b"FE2O3/LOSSLESS-MIR-TO-KIR-CORRESPONDENCE-EVIDENCE/V4\0";
const HEADER_BYTES_V4: usize = 124;
const BLOCK_RECORD_BYTES_V4: usize = 16;
const STATEMENT_RECORD_BYTES_V4: usize = 24;
const TERMINATOR_RECORD_BYTES_V4: usize = 20;
const SYNTHETIC_RECORD_BYTES_V4: usize = 16;
const PARAMETER_RECORD_BYTES_V4: usize = 12;

/// Exact semantic block to KIR block correspondence under the current versioned KIR owner.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MirToKirBlockCorrespondenceEvidenceV4 {
    semantic_function: u32,
    semantic_block: u32,
    kernel_ir_block: u32,
    source_statement_count: u32,
}

impl MirToKirBlockCorrespondenceEvidenceV4 {
    /// Returns the semantic function index.
    pub const fn semantic_function(&self) -> u32 {
        self.semantic_function
    }

    /// Returns the semantic block index.
    pub const fn semantic_block(&self) -> u32 {
        self.semantic_block
    }

    /// Returns the exact KIR block identity.
    pub const fn kernel_ir_block(&self) -> u32 {
        self.kernel_ir_block
    }

    /// Returns the exact number of statements in the semantic block.
    pub const fn source_statement_count(&self) -> u32 {
        self.source_statement_count
    }
}

/// Exact Kernel IR operation span emitted by one semantic statement.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MirToKirStatementSpanEvidenceV4 {
    semantic_function: u32,
    semantic_block: u32,
    statement: u32,
    kernel_ir_block: u32,
    first_operation: u32,
    operation_count: u32,
}

impl MirToKirStatementSpanEvidenceV4 {
    /// Returns the semantic function index.
    pub const fn semantic_function(&self) -> u32 {
        self.semantic_function
    }

    /// Returns the semantic block index.
    pub const fn semantic_block(&self) -> u32 {
        self.semantic_block
    }

    /// Returns the semantic statement ordinal.
    pub const fn statement(&self) -> u32 {
        self.statement
    }

    /// Returns the exact Kernel IR block index.
    pub const fn kernel_ir_block(&self) -> u32 {
        self.kernel_ir_block
    }

    /// Returns the first emitted operation ordinal.
    pub const fn first_operation(&self) -> u32 {
        self.first_operation
    }

    /// Returns the exact number of emitted operations, including zero.
    pub const fn operation_count(&self) -> u32 {
        self.operation_count
    }
}

/// Exact Kernel IR operation span emitted by one semantic terminator.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MirToKirTerminatorSpanEvidenceV4 {
    semantic_function: u32,
    semantic_block: u32,
    kernel_ir_block: u32,
    first_operation: u32,
    operation_count: u32,
}

impl MirToKirTerminatorSpanEvidenceV4 {
    /// Returns the semantic function index.
    pub const fn semantic_function(&self) -> u32 {
        self.semantic_function
    }

    /// Returns the semantic block index.
    pub const fn semantic_block(&self) -> u32 {
        self.semantic_block
    }

    /// Returns the exact Kernel IR block index.
    pub const fn kernel_ir_block(&self) -> u32 {
        self.kernel_ir_block
    }

    /// Returns the first emitted operation ordinal.
    pub const fn first_operation(&self) -> u32 {
        self.first_operation
    }

    /// Returns the exact number of emitted operations.
    pub const fn operation_count(&self) -> u32 {
        self.operation_count
    }
}

/// Closed synthetic lowering rule encoded in lossless correspondence evidence.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum MirToKirSyntheticRuleEvidenceV4 {
    /// Private storage used for enum payload joins.
    EnumPayloadStorage = 1,
    /// Canonical trap in the shared runtime-assert failure block.
    RuntimeAssertFailureTrap = 2,
}

/// Exact operation span emitted by one synthetic lowering rule.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MirToKirSyntheticSpanEvidenceV4 {
    rule: MirToKirSyntheticRuleEvidenceV4,
    kernel_ir_block: u32,
    first_operation: u32,
    operation_count: u32,
}

impl MirToKirSyntheticSpanEvidenceV4 {
    /// Returns the closed synthetic rule.
    pub const fn rule(&self) -> MirToKirSyntheticRuleEvidenceV4 {
        self.rule
    }

    /// Returns the exact Kernel IR block index.
    pub const fn kernel_ir_block(&self) -> u32 {
        self.kernel_ir_block
    }

    /// Returns the first emitted operation ordinal.
    pub const fn first_operation(&self) -> u32 {
        self.first_operation
    }

    /// Returns the exact number of emitted operations.
    pub const fn operation_count(&self) -> u32 {
        self.operation_count
    }
}

/// Exact semantic argument-local to Kernel IR parameter binding.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MirToKirParameterBindingEvidenceV4 {
    semantic_function: u32,
    semantic_local: u32,
    kernel_ir_value: u32,
}

impl MirToKirParameterBindingEvidenceV4 {
    /// Returns the semantic function index.
    pub const fn semantic_function(&self) -> u32 {
        self.semantic_function
    }

    /// Returns the semantic argument-local index.
    pub const fn semantic_local(&self) -> u32 {
        self.semantic_local
    }

    /// Returns the exact Kernel IR parameter value index.
    pub const fn kernel_ir_value(&self) -> u32 {
        self.kernel_ir_value
    }
}

/// Complete canonical authority-free correspondence and induction custody.
#[derive(Debug, Eq, PartialEq)]
pub struct InertCanonicalMirToKirCorrespondenceEvidenceV4 {
    canonical_bytes: Box<[u8]>,
    identity: [u8; 32],
    semantic_sha256: [u8; 32],
    canonical_kernel_ir: ProductionCanonicalKernelIrIdentityV1,
    function_count: u32,
    blocks: Box<[MirToKirBlockCorrespondenceEvidenceV4]>,
    statements: Box<[MirToKirStatementSpanEvidenceV4]>,
    terminators: Box<[MirToKirTerminatorSpanEvidenceV4]>,
    synthetics: Box<[MirToKirSyntheticSpanEvidenceV4]>,
    parameters: Box<[MirToKirParameterBindingEvidenceV4]>,
    induction: InertCanonicalSemanticU32InductionEvidenceV1,
}

impl InertCanonicalMirToKirCorrespondenceEvidenceV4 {
    /// Revalidates one live owner and binds every correspondence field plus the exact report.
    pub fn from_live_owner(
        owner: &ProductionSemanticKirOwnerV1,
        induction_report: &SemanticU32InductionNoOverflowReportV1,
    ) -> Result<Self, ProductionCorrespondenceEvidenceErrorV4> {
        owner.verify_equivalence().map_err(|error| {
            ProductionCorrespondenceEvidenceErrorV4::LiveOwner(error.to_string())
        })?;
        let induction = InertCanonicalSemanticU32InductionEvidenceV1::from_report(induction_report)
            .map_err(ProductionCorrespondenceEvidenceErrorV4::Induction)?;
        let semantic_sha256 = *owner.semantic().semantic().semantic_sha256().as_bytes();
        if &semantic_sha256 != induction.semantic_mir_sha256() {
            return Err(ProductionCorrespondenceEvidenceErrorV4::NestedIdentityMismatch);
        }

        let correspondence = owner.correspondence();
        let covered_functions = correspondence
            .blocks()
            .iter()
            .map(|record| record.semantic_function().index())
            .collect::<BTreeSet<_>>();
        let function_count = u32::try_from(covered_functions.len())
            .map_err(|_| ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
        let mut blocks = correspondence
            .blocks()
            .iter()
            .map(|record| MirToKirBlockCorrespondenceEvidenceV4 {
                semantic_function: record.semantic_function().index(),
                semantic_block: record.semantic_block().index(),
                kernel_ir_block: record.kernel_ir_block().0,
                source_statement_count: record.source_statement_count(),
            })
            .collect::<Vec<_>>();
        let mut statements = correspondence
            .statement_operation_spans()
            .iter()
            .map(|span| MirToKirStatementSpanEvidenceV4 {
                semantic_function: span.semantic_function().index(),
                semantic_block: span.semantic_block().index(),
                statement: span.statement_ordinal(),
                kernel_ir_block: span.kernel_ir_block().0,
                first_operation: span.first_operation_ordinal(),
                operation_count: span.operation_count(),
            })
            .collect::<Vec<_>>();
        let mut terminators = correspondence
            .terminator_operation_spans()
            .iter()
            .map(|span| MirToKirTerminatorSpanEvidenceV4 {
                semantic_function: span.semantic_function().index(),
                semantic_block: span.semantic_block().index(),
                kernel_ir_block: span.kernel_ir_block().0,
                first_operation: span.first_operation_ordinal(),
                operation_count: span.operation_count(),
            })
            .collect::<Vec<_>>();
        let mut synthetics = correspondence
            .synthetic_operation_spans()
            .iter()
            .map(|span| MirToKirSyntheticSpanEvidenceV4 {
                rule: match span.rule() {
                    SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage => {
                        MirToKirSyntheticRuleEvidenceV4::EnumPayloadStorage
                    }
                    SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap => {
                        MirToKirSyntheticRuleEvidenceV4::RuntimeAssertFailureTrap
                    }
                },
                kernel_ir_block: span.kernel_ir_block().0,
                first_operation: span.first_operation_ordinal(),
                operation_count: span.operation_count(),
            })
            .collect::<Vec<_>>();
        let mut parameters = correspondence
            .parameter_bindings()
            .iter()
            .map(|binding| MirToKirParameterBindingEvidenceV4 {
                semantic_function: binding.semantic_function().index(),
                semantic_local: binding.semantic_local().index(),
                kernel_ir_value: binding.kernel_ir_value().0,
            })
            .collect::<Vec<_>>();
        statements.sort_unstable_by_key(|span| {
            (span.semantic_function, span.semantic_block, span.statement)
        });
        terminators.sort_unstable_by_key(|span| (span.semantic_function, span.semantic_block));
        synthetics.sort_unstable();
        parameters
            .sort_unstable_by_key(|binding| (binding.semantic_function, binding.semantic_local));
        blocks.sort_unstable_by_key(|record| (record.semantic_function, record.semantic_block));
        let bytes = encode(
            semantic_sha256,
            owner.canonical_kernel_ir_identity(),
            function_count,
            &blocks,
            &statements,
            &terminators,
            &synthetics,
            &parameters,
            &induction,
        )?;
        Self::decode(&bytes)
    }

    /// Strictly decodes one complete canonical V4 aggregate.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProductionCorrespondenceEvidenceErrorV4> {
        if bytes.len() > MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V4 {
            return Err(ProductionCorrespondenceEvidenceErrorV4::TooLarge);
        }
        let mut reader = ReaderV4::new(bytes);
        if reader.fixed::<8>()? != MAGIC_V4
            || reader.u16()? != MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_VERSION_V4
            || reader.u16()? != MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_POLICY_V4
            || reader.u32()? != 0
        {
            return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidHeader);
        }
        let declared = reader.usize_u32()?;
        if declared != bytes.len() {
            return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidLength);
        }
        let semantic_sha256 = reader.fixed::<32>()?;
        let kernel_ir_version = match reader.u16()? {
            8 => ProductionCanonicalKernelIrVersionV1::V8,
            9 => ProductionCanonicalKernelIrVersionV1::V9,
            _ => return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidHeader),
        };
        if reader.u16()? != 0 {
            return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidHeader);
        }
        let kernel_ir_length = reader.u64()?;
        let kernel_ir_digest = reader.fixed::<32>()?;
        if semantic_sha256 == [0; 32] || kernel_ir_digest == [0; 32] || kernel_ir_length == 0 {
            return Err(ProductionCorrespondenceEvidenceErrorV4::ZeroIdentity);
        }
        let canonical_kernel_ir = ProductionCanonicalKernelIrIdentityV1::from_canonical_parts(
            kernel_ir_version,
            kernel_ir_digest,
            kernel_ir_length,
        );
        let function_count = reader.u32()?;
        let block_count = reader.usize_u32()?;
        let statement_count = reader.usize_u32()?;
        let terminator_count = reader.usize_u32()?;
        let synthetic_count = reader.usize_u32()?;
        let parameter_count = reader.usize_u32()?;
        let induction_bytes = reader.usize_u32()?;
        validate_counts(
            usize::try_from(function_count)
                .map_err(|_| ProductionCorrespondenceEvidenceErrorV4::Overflow)?,
            block_count,
            statement_count,
            terminator_count,
            synthetic_count,
            parameter_count,
        )?;
        let record_bytes = exact_record_bytes(
            block_count,
            statement_count,
            terminator_count,
            synthetic_count,
            parameter_count,
        )?;
        let exact_remaining = record_bytes
            .checked_add(induction_bytes)
            .ok_or(ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
        if reader.remaining() != exact_remaining {
            return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidLength);
        }

        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            blocks.push(decode_block(&mut reader)?);
        }
        let mut statements = Vec::with_capacity(statement_count);
        for _ in 0..statement_count {
            statements.push(decode_statement(&mut reader)?);
        }
        let mut terminators = Vec::with_capacity(terminator_count);
        for _ in 0..terminator_count {
            terminators.push(decode_terminator(&mut reader)?);
        }
        let mut synthetics = Vec::with_capacity(synthetic_count);
        for _ in 0..synthetic_count {
            synthetics.push(decode_synthetic(&mut reader)?);
        }
        let mut parameters = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count {
            parameters.push(decode_parameter(&mut reader)?);
        }
        let induction =
            InertCanonicalSemanticU32InductionEvidenceV1::decode(reader.take(induction_bytes)?)
                .map_err(ProductionCorrespondenceEvidenceErrorV4::Induction)?;
        reader.finish()?;
        if &semantic_sha256 != induction.semantic_mir_sha256() {
            return Err(ProductionCorrespondenceEvidenceErrorV4::NestedIdentityMismatch);
        }
        validate_blocks(function_count, &blocks)?;
        validate_records(&statements, &terminators, &synthetics, &parameters)?;
        if !synthetics.is_empty() && function_count != 1 {
            return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord);
        }
        let reencoded = encode(
            semantic_sha256,
            canonical_kernel_ir,
            function_count,
            &blocks,
            &statements,
            &terminators,
            &synthetics,
            &parameters,
            &induction,
        )?;
        if reencoded != bytes {
            return Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical);
        }
        let identity = evidence_identity(&reencoded)?;
        Ok(Self {
            canonical_bytes: reencoded.into_boxed_slice(),
            identity,
            semantic_sha256,
            canonical_kernel_ir,
            function_count,
            blocks: blocks.into_boxed_slice(),
            statements: statements.into_boxed_slice(),
            terminators: terminators.into_boxed_slice(),
            synthetics: synthetics.into_boxed_slice(),
            parameters: parameters.into_boxed_slice(),
            induction,
        })
    }

    /// Re-decodes the exact retained aggregate and identity.
    pub fn revalidate(&self) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
        let decoded = Self::decode(&self.canonical_bytes)?;
        if decoded != *self {
            return Err(ProductionCorrespondenceEvidenceErrorV4::IdentityMismatch);
        }
        Ok(())
    }

    /// Returns the complete canonical aggregate bytes.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    /// Returns the exact aggregate identity.
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Returns the exact semantic MIR SHA-256.
    pub const fn semantic_sha256(&self) -> &[u8; 32] {
        &self.semantic_sha256
    }

    /// Returns the exact versioned canonical production-KIR identity.
    pub const fn canonical_kernel_ir_identity(&self) -> ProductionCanonicalKernelIrIdentityV1 {
        self.canonical_kernel_ir
    }

    /// Returns the number of covered semantic and defined KIR functions.
    pub const fn function_count(&self) -> u32 {
        self.function_count
    }

    /// Returns every exact semantic-to-KIR block record.
    pub fn blocks(&self) -> &[MirToKirBlockCorrespondenceEvidenceV4] {
        &self.blocks
    }

    /// Returns every exact semantic statement span.
    pub fn statement_spans(&self) -> &[MirToKirStatementSpanEvidenceV4] {
        &self.statements
    }

    /// Returns every exact semantic terminator span.
    pub fn terminator_spans(&self) -> &[MirToKirTerminatorSpanEvidenceV4] {
        &self.terminators
    }

    /// Returns every exact synthetic span.
    pub fn synthetic_spans(&self) -> &[MirToKirSyntheticSpanEvidenceV4] {
        &self.synthetics
    }

    /// Returns every exact semantic argument-local to KIR parameter binding.
    pub fn parameter_bindings(&self) -> &[MirToKirParameterBindingEvidenceV4] {
        &self.parameters
    }

    /// Returns the independently decoded exact semantic induction report evidence.
    pub const fn semantic_u32_induction(&self) -> &InertCanonicalSemanticU32InductionEvidenceV1 {
        &self.induction
    }

    /// Lossless correspondence custody grants no compiler or runtime authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Fail-closed lossless correspondence evidence error.
#[derive(Debug)]
pub enum ProductionCorrespondenceEvidenceErrorV4 {
    /// Live equivalence replay failed.
    LiveOwner(String),
    /// Nested semantic induction report evidence failed.
    Induction(SemanticU32InductionEvidenceErrorV1),
    /// Aggregate exceeds the outer receipt budget.
    TooLarge,
    /// Header magic, version, policy, flags, or reserved fields are invalid.
    InvalidHeader,
    /// Declared, computed, and available lengths differ.
    InvalidLength,
    /// Count or byte arithmetic overflowed.
    Overflow,
    /// A record count exceeds its fixed bound.
    LimitExceeded,
    /// Span or binding records are inconsistent.
    InvalidRecord,
    /// Nested semantic identities differ.
    NestedIdentityMismatch,
    /// Input ended before a complete field was available.
    Truncated,
    /// Decoded fields are not in their unique canonical representation.
    NonCanonical,
    /// Retained bytes and content identity changed.
    IdentityMismatch,
    /// Derived aggregate identity was the reserved all-zero value.
    ZeroIdentity,
}

impl fmt::Display for ProductionCorrespondenceEvidenceErrorV4 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LiveOwner(error) => write!(formatter, "live semantic-KIR owner failed: {error}"),
            Self::Induction(error) => {
                write!(formatter, "semantic induction evidence failed: {error}")
            }
            Self::TooLarge => formatter.write_str("lossless correspondence exceeds its byte limit"),
            Self::InvalidHeader => formatter.write_str("lossless correspondence header is invalid"),
            Self::InvalidLength => formatter.write_str("lossless correspondence length is invalid"),
            Self::Overflow => formatter.write_str("lossless correspondence arithmetic overflowed"),
            Self::LimitExceeded => {
                formatter.write_str("lossless correspondence count exceeds its limit")
            }
            Self::InvalidRecord => formatter.write_str("lossless correspondence record is invalid"),
            Self::NestedIdentityMismatch => {
                formatter.write_str("lossless correspondence nested identities differ")
            }
            Self::Truncated => formatter.write_str("lossless correspondence is truncated"),
            Self::NonCanonical => formatter.write_str("lossless correspondence is not canonical"),
            Self::IdentityMismatch => {
                formatter.write_str("lossless correspondence identity changed")
            }
            Self::ZeroIdentity => formatter.write_str("lossless correspondence identity is zero"),
        }
    }
}

impl Error for ProductionCorrespondenceEvidenceErrorV4 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Induction(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_counts(
    functions: usize,
    blocks: usize,
    statements: usize,
    terminators: usize,
    synthetics: usize,
    parameters: usize,
) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
    if functions == 0
        || functions > MAX_MIR_TO_KIR_CORRESPONDENCE_FUNCTIONS_V3
        || blocks == 0
        || blocks > MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3
        || statements > MAX_MIR_TO_KIR_STATEMENT_SPANS_V4
        || terminators > MAX_MIR_TO_KIR_CORRESPONDENCE_BLOCKS_V3
        || synthetics > MAX_MIR_TO_KIR_SYNTHETIC_SPANS_V4
        || parameters > MAX_MIR_TO_KIR_PARAMETER_BINDINGS_V4
    {
        Err(ProductionCorrespondenceEvidenceErrorV4::LimitExceeded)
    } else {
        Ok(())
    }
}

fn validate_blocks(
    function_count: u32,
    blocks: &[MirToKirBlockCorrespondenceEvidenceV4],
) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
    validate_counts(function_count as usize, blocks.len(), 0, 0, 0, 0)?;
    let mut cursor = 0_usize;
    let mut covered_functions = 0_u32;
    let mut previous_function = None;
    while let Some(first) = blocks.get(cursor) {
        if previous_function.is_some_and(|previous| previous >= first.semantic_function) {
            return Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical);
        }
        let function = first.semantic_function;
        let mut semantic_block = 0_u32;
        while let Some(record) = blocks.get(cursor) {
            if record.semantic_function != function {
                break;
            }
            if record.semantic_block != semantic_block
                || record.kernel_ir_block != record.semantic_block
            {
                return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord);
            }
            semantic_block = semantic_block
                .checked_add(1)
                .ok_or(ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
            cursor += 1;
        }
        previous_function = Some(function);
        covered_functions = covered_functions
            .checked_add(1)
            .ok_or(ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
    }
    if cursor != blocks.len() || covered_functions != function_count {
        return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord);
    }
    Ok(())
}

fn validate_records(
    statements: &[MirToKirStatementSpanEvidenceV4],
    terminators: &[MirToKirTerminatorSpanEvidenceV4],
    synthetics: &[MirToKirSyntheticSpanEvidenceV4],
    parameters: &[MirToKirParameterBindingEvidenceV4],
) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
    validate_counts(
        1,
        1,
        statements.len(),
        terminators.len(),
        synthetics.len(),
        parameters.len(),
    )?;
    if statements.windows(2).any(|window| {
        (
            window[0].semantic_function,
            window[0].semantic_block,
            window[0].statement,
        ) >= (
            window[1].semantic_function,
            window[1].semantic_block,
            window[1].statement,
        )
    }) || terminators.windows(2).any(|window| {
        (window[0].semantic_function, window[0].semantic_block)
            >= (window[1].semantic_function, window[1].semantic_block)
    }) || parameters.windows(2).any(|window| {
        (window[0].semantic_function, window[0].semantic_local)
            >= (window[1].semantic_function, window[1].semantic_local)
    }) {
        return Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical);
    }
    require_strict_order(synthetics)?;
    for (first, count) in statements
        .iter()
        .map(|span| (span.first_operation, span.operation_count))
        .chain(
            terminators
                .iter()
                .map(|span| (span.first_operation, span.operation_count)),
        )
        .chain(
            synthetics
                .iter()
                .map(|span| (span.first_operation, span.operation_count)),
        )
    {
        first
            .checked_add(count)
            .ok_or(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord)?;
    }
    Ok(())
}

fn require_strict_order<T: Ord>(
    records: &[T],
) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
    if records.windows(2).any(|window| window[0] >= window[1]) {
        Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical)
    } else {
        Ok(())
    }
}

fn exact_record_bytes(
    blocks: usize,
    statements: usize,
    terminators: usize,
    synthetics: usize,
    parameters: usize,
) -> Result<usize, ProductionCorrespondenceEvidenceErrorV4> {
    blocks
        .checked_mul(BLOCK_RECORD_BYTES_V4)
        .and_then(|bytes| {
            statements
                .checked_mul(STATEMENT_RECORD_BYTES_V4)
                .and_then(|next| bytes.checked_add(next))
        })
        .and_then(|bytes| {
            terminators
                .checked_mul(TERMINATOR_RECORD_BYTES_V4)
                .and_then(|next| bytes.checked_add(next))
        })
        .and_then(|bytes| {
            synthetics
                .checked_mul(SYNTHETIC_RECORD_BYTES_V4)
                .and_then(|next| bytes.checked_add(next))
        })
        .and_then(|bytes| {
            parameters
                .checked_mul(PARAMETER_RECORD_BYTES_V4)
                .and_then(|next| bytes.checked_add(next))
        })
        .ok_or(ProductionCorrespondenceEvidenceErrorV4::Overflow)
}

fn encode(
    semantic_sha256: [u8; 32],
    canonical_kernel_ir: ProductionCanonicalKernelIrIdentityV1,
    function_count: u32,
    blocks: &[MirToKirBlockCorrespondenceEvidenceV4],
    statements: &[MirToKirStatementSpanEvidenceV4],
    terminators: &[MirToKirTerminatorSpanEvidenceV4],
    synthetics: &[MirToKirSyntheticSpanEvidenceV4],
    parameters: &[MirToKirParameterBindingEvidenceV4],
    induction: &InertCanonicalSemanticU32InductionEvidenceV1,
) -> Result<Vec<u8>, ProductionCorrespondenceEvidenceErrorV4> {
    induction
        .revalidate()
        .map_err(ProductionCorrespondenceEvidenceErrorV4::Induction)?;
    if semantic_sha256 == [0; 32]
        || canonical_kernel_ir.digest() == &[0; 32]
        || canonical_kernel_ir.canonical_length() == 0
    {
        return Err(ProductionCorrespondenceEvidenceErrorV4::ZeroIdentity);
    }
    if &semantic_sha256 != induction.semantic_mir_sha256() {
        return Err(ProductionCorrespondenceEvidenceErrorV4::NestedIdentityMismatch);
    }
    validate_blocks(function_count, blocks)?;
    validate_records(statements, terminators, synthetics, parameters)?;
    validate_counts(
        function_count as usize,
        blocks.len(),
        statements.len(),
        terminators.len(),
        synthetics.len(),
        parameters.len(),
    )?;
    if !synthetics.is_empty() && function_count != 1 {
        return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord);
    }
    let record_bytes = exact_record_bytes(
        blocks.len(),
        statements.len(),
        terminators.len(),
        synthetics.len(),
        parameters.len(),
    )?;
    let exact_size = HEADER_BYTES_V4
        .checked_add(record_bytes)
        .and_then(|bytes| bytes.checked_add(induction.canonical_bytes().len()))
        .ok_or(ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
    if exact_size > MAX_MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_BYTES_V4 {
        return Err(ProductionCorrespondenceEvidenceErrorV4::TooLarge);
    }
    let mut bytes = Vec::with_capacity(exact_size);
    bytes.extend_from_slice(&MAGIC_V4);
    push_u16(&mut bytes, MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_VERSION_V4);
    push_u16(&mut bytes, MIR_TO_KIR_CORRESPONDENCE_EVIDENCE_POLICY_V4);
    push_u32(&mut bytes, 0);
    push_usize(&mut bytes, exact_size)?;
    bytes.extend_from_slice(&semantic_sha256);
    push_u16(
        &mut bytes,
        match canonical_kernel_ir.version() {
            ProductionCanonicalKernelIrVersionV1::V8 => 8,
            ProductionCanonicalKernelIrVersionV1::V9 => 9,
        },
    );
    push_u16(&mut bytes, 0);
    push_u64(&mut bytes, canonical_kernel_ir.canonical_length());
    bytes.extend_from_slice(canonical_kernel_ir.digest());
    push_u32(&mut bytes, function_count);
    push_usize(&mut bytes, blocks.len())?;
    push_usize(&mut bytes, statements.len())?;
    push_usize(&mut bytes, terminators.len())?;
    push_usize(&mut bytes, synthetics.len())?;
    push_usize(&mut bytes, parameters.len())?;
    push_usize(&mut bytes, induction.canonical_bytes().len())?;
    for block in blocks {
        for value in [
            block.semantic_function,
            block.semantic_block,
            block.kernel_ir_block,
            block.source_statement_count,
        ] {
            push_u32(&mut bytes, value);
        }
    }
    for span in statements {
        for value in [
            span.semantic_function,
            span.semantic_block,
            span.statement,
            span.kernel_ir_block,
            span.first_operation,
            span.operation_count,
        ] {
            push_u32(&mut bytes, value);
        }
    }
    for span in terminators {
        for value in [
            span.semantic_function,
            span.semantic_block,
            span.kernel_ir_block,
            span.first_operation,
            span.operation_count,
        ] {
            push_u32(&mut bytes, value);
        }
    }
    for span in synthetics {
        for value in [
            span.rule as u32,
            span.kernel_ir_block,
            span.first_operation,
            span.operation_count,
        ] {
            push_u32(&mut bytes, value);
        }
    }
    for binding in parameters {
        for value in [
            binding.semantic_function,
            binding.semantic_local,
            binding.kernel_ir_value,
        ] {
            push_u32(&mut bytes, value);
        }
    }
    bytes.extend_from_slice(induction.canonical_bytes());
    if bytes.len() != exact_size {
        return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidLength);
    }
    Ok(bytes)
}

fn decode_block(
    reader: &mut ReaderV4<'_>,
) -> Result<MirToKirBlockCorrespondenceEvidenceV4, ProductionCorrespondenceEvidenceErrorV4> {
    Ok(MirToKirBlockCorrespondenceEvidenceV4 {
        semantic_function: reader.u32()?,
        semantic_block: reader.u32()?,
        kernel_ir_block: reader.u32()?,
        source_statement_count: reader.u32()?,
    })
}

fn decode_statement(
    reader: &mut ReaderV4<'_>,
) -> Result<MirToKirStatementSpanEvidenceV4, ProductionCorrespondenceEvidenceErrorV4> {
    Ok(MirToKirStatementSpanEvidenceV4 {
        semantic_function: reader.u32()?,
        semantic_block: reader.u32()?,
        statement: reader.u32()?,
        kernel_ir_block: reader.u32()?,
        first_operation: reader.u32()?,
        operation_count: reader.u32()?,
    })
}

fn decode_terminator(
    reader: &mut ReaderV4<'_>,
) -> Result<MirToKirTerminatorSpanEvidenceV4, ProductionCorrespondenceEvidenceErrorV4> {
    Ok(MirToKirTerminatorSpanEvidenceV4 {
        semantic_function: reader.u32()?,
        semantic_block: reader.u32()?,
        kernel_ir_block: reader.u32()?,
        first_operation: reader.u32()?,
        operation_count: reader.u32()?,
    })
}

fn decode_synthetic(
    reader: &mut ReaderV4<'_>,
) -> Result<MirToKirSyntheticSpanEvidenceV4, ProductionCorrespondenceEvidenceErrorV4> {
    let rule = match reader.u32()? {
        1 => MirToKirSyntheticRuleEvidenceV4::EnumPayloadStorage,
        2 => MirToKirSyntheticRuleEvidenceV4::RuntimeAssertFailureTrap,
        _ => return Err(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord),
    };
    Ok(MirToKirSyntheticSpanEvidenceV4 {
        rule,
        kernel_ir_block: reader.u32()?,
        first_operation: reader.u32()?,
        operation_count: reader.u32()?,
    })
}

fn decode_parameter(
    reader: &mut ReaderV4<'_>,
) -> Result<MirToKirParameterBindingEvidenceV4, ProductionCorrespondenceEvidenceErrorV4> {
    Ok(MirToKirParameterBindingEvidenceV4 {
        semantic_function: reader.u32()?,
        semantic_local: reader.u32()?,
        kernel_ir_value: reader.u32()?,
    })
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

fn push_usize(
    bytes: &mut Vec<u8>,
    value: usize,
) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
    push_u32(
        bytes,
        u32::try_from(value).map_err(|_| ProductionCorrespondenceEvidenceErrorV4::Overflow)?,
    );
    Ok(())
}

fn evidence_identity(bytes: &[u8]) -> Result<[u8; 32], ProductionCorrespondenceEvidenceErrorV4> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
    let mut digest = Sha256::new();
    digest.update(IDENTITY_DOMAIN_V4);
    digest.update(length.to_le_bytes());
    digest.update(bytes);
    let identity = digest.finalize().into();
    if identity == [0; 32] {
        Err(ProductionCorrespondenceEvidenceErrorV4::ZeroIdentity)
    } else {
        Ok(identity)
    }
}

struct ReaderV4<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ReaderV4<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn fixed<const N: usize>(
        &mut self,
    ) -> Result<[u8; N], ProductionCorrespondenceEvidenceErrorV4> {
        self.take(N)?
            .try_into()
            .map_err(|_| ProductionCorrespondenceEvidenceErrorV4::Truncated)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ProductionCorrespondenceEvidenceErrorV4> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ProductionCorrespondenceEvidenceErrorV4::Overflow)?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(ProductionCorrespondenceEvidenceErrorV4::Truncated)?;
        self.offset = end;
        Ok(result)
    }

    fn u16(&mut self) -> Result<u16, ProductionCorrespondenceEvidenceErrorV4> {
        Ok(u16::from_le_bytes(self.fixed()?))
    }

    fn u32(&mut self) -> Result<u32, ProductionCorrespondenceEvidenceErrorV4> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    fn u64(&mut self) -> Result<u64, ProductionCorrespondenceEvidenceErrorV4> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    fn usize_u32(&mut self) -> Result<usize, ProductionCorrespondenceEvidenceErrorV4> {
        usize::try_from(self.u32()?).map_err(|_| ProductionCorrespondenceEvidenceErrorV4::Overflow)
    }

    fn finish(self) -> Result<(), ProductionCorrespondenceEvidenceErrorV4> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(ProductionCorrespondenceEvidenceErrorV4::InvalidLength)
        }
    }
}

#[cfg(test)]
mod private_tests {
    use super::*;

    #[test]
    fn source_keys_are_unique_and_spans_do_not_overflow() {
        let statement = MirToKirStatementSpanEvidenceV4 {
            semantic_function: 0,
            semantic_block: 1,
            statement: 2,
            kernel_ir_block: 1,
            first_operation: 3,
            operation_count: 4,
        };
        let mut duplicate = statement;
        duplicate.kernel_ir_block = 9;
        assert!(matches!(
            validate_records(&[statement, duplicate], &[], &[], &[]),
            Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical)
        ));

        let mut overflowing = statement;
        overflowing.first_operation = u32::MAX;
        overflowing.operation_count = 1;
        assert!(matches!(
            validate_records(&[overflowing], &[], &[], &[]),
            Err(ProductionCorrespondenceEvidenceErrorV4::InvalidRecord)
        ));

        let terminator = MirToKirTerminatorSpanEvidenceV4 {
            semantic_function: 0,
            semantic_block: 1,
            kernel_ir_block: 1,
            first_operation: 0,
            operation_count: 1,
        };
        let mut duplicate_terminator = terminator;
        duplicate_terminator.kernel_ir_block = 2;
        assert!(matches!(
            validate_records(&[], &[terminator, duplicate_terminator], &[], &[]),
            Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical)
        ));

        let parameter = MirToKirParameterBindingEvidenceV4 {
            semantic_function: 0,
            semantic_local: 1,
            kernel_ir_value: 2,
        };
        let mut duplicate_parameter = parameter;
        duplicate_parameter.kernel_ir_value = 3;
        assert!(matches!(
            validate_records(&[], &[], &[], &[parameter, duplicate_parameter]),
            Err(ProductionCorrespondenceEvidenceErrorV4::NonCanonical)
        ));
    }
}

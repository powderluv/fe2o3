//! Bounded two-pass preflight for the production rustc semantic importer.
//!
//! This module retains live rustc producers and proves that the selected raw
//! MIR is inside the first reviewed subset. It does not construct, admit, or
//! authorize canonical semantic MIR.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;

use fe2o3_mir_model::semantic_mir_v1::{
    SemanticAbiIdentityV1, SemanticAxisV1, SemanticBlockIdV1, SemanticBlockIdentityV1,
    SemanticFunctionIdV1, SemanticFunctionIdentityV1, SemanticLayoutIdentityV1, SemanticLocalIdV1,
    SemanticLocalIdentityV1, SemanticMirLimitsV1, SemanticMirResourceV1,
    SemanticSourceFileIdentityV1, SemanticSourceOriginV1, SemanticSourceProvenanceV1,
    SemanticTargetDataLayoutV1, SemanticTypeIdV1, SemanticTypeIdentityV1,
};
use rustc_abi::ExternAbi;
use rustc_middle::mir::{
    AggregateKind, AssertKind, BinOp, Body, BorrowKind, Local, MutBorrowKind,
    NonDivergingIntrinsic, Operand, Place, PlaceTy, ProjectionElem, RawPtrKind, Rvalue,
    START_BLOCK, StatementKind, TerminatorKind, UnwindAction, VarDebugInfoContents,
};
use rustc_middle::ty::layout::{LayoutCx, LayoutOf, TyAndLayout};
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{
    self, EarlyBinder, Instance, InstanceKind, Ty, TyCtxt, TyKind, TypeVisitableExt, TypingEnv,
};
use rustc_span::Span;
use rustc_target::callconv::FnAbi;

use crate::collector::{
    AuthenticatedContextEntriesV1, BoundContextEntryV29, CollectedFunctionRole,
};
use crate::production_rustc_drop_v1::{ProductionRustcDropClassV1, classify_rustc_drop_v1};
use crate::production_rustc_intrinsic_v1::{
    ProductionRustcIntrinsicOperationV1, atomic_ordering_tag_v1, atomic_scope_tag_v1,
};
use crate::production_rustc_slice_metadata_v1::{
    SliceMetadataErrorV1, SliceMetadataPlanV1, SliceMetadataRewriteV1,
};
use crate::production_safe_core_shift_v1::{NormalizedCallV1, SafeCoreShiftV1};
use crate::production_semantic_body_v1::receiver_materialization_v1::ReceiverLocalV1;
use crate::production_semantic_body_v1::receiver_reborrow_v1::{
    ReceiverReborrowErrorV1, ReceiverReborrowV1, derive_fn_receiver_reborrow_v1,
};
use crate::production_semantic_terminal_v1::{
    ProductionSemanticTerminalRuleV1, ProductionTerminalExpansionV1,
};
use crate::rustc_semantic_adapter_v1::{
    CanonicalFunctionIdentitiesV1, CanonicalSourceProvenanceV1, SemanticIdentityDigestV1,
    borrowed_rustc_mir_body_sha256_v1, canonical_function_identities_v1,
    canonical_source_provenance_and_debug_files_v1, canonical_source_provenance_v1,
    rustc_block_identity_v1, rustc_fn_abi_sha256_v1, rustc_fn_signature_sha256_v1,
    rustc_local_identity_v1, rustc_mir_body_sha256_v1, rustc_semantic_fn_abi_identity_v1,
    rustc_semantic_fn_abi_layout_identity_v1, rustc_semantic_layout_identity_v1,
    rustc_type_identity_v1, rustc_type_layout_sha256_v1,
};

#[cfg(test)]
#[path = "rustc_semantic_plan_v1/context_body_commitment_v29_tests.rs"]
mod context_body_commitment_v29_tests;

#[cfg(test)]
const PREFLIGHT_PLAN_DOMAIN_V1: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v1";
#[cfg(test)]
const PREFLIGHT_PLAN_DOMAIN_V2: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v2";
#[cfg(test)]
const PREFLIGHT_PLAN_DOMAIN_V3: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v3";
const PREFLIGHT_PLAN_DOMAIN_V4: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v4";
const PREFLIGHT_PLAN_DOMAIN_V5: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v5";
const PREFLIGHT_TYPES_DOMAIN_V5: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v5/types";
const PREFLIGHT_SOURCE_FILES_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/source-files";
const PREFLIGHT_FUNCTIONS_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/functions";
const PREFLIGHT_FUNCTION_ABIS_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/function-abis";
const PREFLIGHT_TERMINALS_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/terminals";
const PREFLIGHT_BODY_DOMAIN_V5: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v5/body";
const PREFLIGHT_ROOTS_DOMAIN_V5: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v5/roots";
const PREFLIGHT_EDGES_DOMAIN_V5: &[u8] = b"fe2o3/semantic-mir/rustc-preflight-plan/v5/edges";
const PREFLIGHT_DIRECT_CALLS_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/direct-calls";
const PREFLIGHT_TERMINAL_EXPANSIONS_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/terminal-expansions";
const PREFLIGHT_NORMALIZED_INTRINSICS_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/rustc-preflight-plan/v5/normalized-intrinsics";
const PREFLIGHT_SECTION_TYPES_V5: u8 = 0;
const PREFLIGHT_SECTION_SOURCE_FILES_V5: u8 = 1;
const PREFLIGHT_SECTION_FUNCTIONS_V5: u8 = 2;
const PREFLIGHT_SECTION_FUNCTION_ABIS_V5: u8 = 3;
const PREFLIGHT_SECTION_TERMINALS_V5: u8 = 4;
const PREFLIGHT_SECTION_BODIES_V5: u8 = 5;
const PREFLIGHT_SECTION_ROOTS_V5: u8 = 6;
const PREFLIGHT_SECTION_EDGES_V5: u8 = 7;
const PREFLIGHT_SECTION_DIRECT_CALLS_V5: u8 = 8;
const PREFLIGHT_SECTION_TERMINAL_EXPANSIONS_V5: u8 = 9;
const PREFLIGHT_SECTION_NORMALIZED_INTRINSICS_V5: u8 = 10;
const COMPILER_INTRINSIC_DEFINITION_DOMAIN_V1: &[u8] =
    b"fe2o3/semantic-mir/compiler-intrinsic-definition/v1";
const MAX_DIAGNOSTIC_COMPONENT_CHARS_V1: usize = 512;
const MAX_MACRO_EXPANSION_DEPTH_V1: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TerminalIdentitySchemaV1 {
    #[cfg(test)]
    IndependentV1,
    #[cfg(test)]
    CombinedV2,
    #[cfg_attr(not(test), allow(dead_code))]
    CombinedV3,
    CombinedV4,
    CombinedV5,
    CombinedV6,
    CombinedV7,
    CombinedV8,
    CombinedV9,
}

#[derive(Clone, Debug)]
pub(crate) struct RetainedSemanticFunctionProducerV1<'tcx> {
    pub(crate) identities: CanonicalFunctionIdentitiesV1,
    pub(crate) instance: Instance<'tcx>,
    pub(crate) role: CollectedFunctionRole,
    pub(crate) export_name: Option<String>,
    pub(crate) kernel_binding: Option<reserved_fe2o3_symbols::KernelBindingIdV1>,
    pub(crate) generated_host_contract_identity:
        Option<reserved_fe2o3_symbols::GeneratedHostContractIdV3>,
    pub(crate) frontend_contract: Option<crate::collector::AuthenticatedKernelFrontendContractV1>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RetainedSemanticTypeProducerV1<'tcx> {
    pub(crate) identity: SemanticTypeIdentityV1,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) layout: TyAndLayout<'tcx>,
    pub(crate) rustc_layout_sha256: [u8; 32],
    pub(crate) semantic_layout_identity: SemanticLayoutIdentityV1,
}

#[derive(Clone, Debug)]
pub(crate) struct RetainedSemanticFunctionAbiProducerV1<'tcx> {
    pub(crate) function: SemanticFunctionIdV1,
    pub(crate) identity: SemanticAbiIdentityV1,
    pub(crate) layout_identity: SemanticLayoutIdentityV1,
    pub(crate) extern_abi: ExternAbi,
    pub(crate) source_inputs: Box<[Ty<'tcx>]>,
    pub(crate) source_output: Ty<'tcx>,
    pub(crate) fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
    pub(crate) rustc_source_signature_sha256: [u8; 32],
    pub(crate) rustc_fn_abi_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RetainedSemanticLocalProducerV1 {
    pub(crate) identity: SemanticLocalIdentityV1,
    pub(crate) rustc_local: u32,
    pub(crate) ty: SemanticTypeIdV1,
    pub(crate) source: RetainedSemanticSourceProducerV1,
}

#[derive(Debug)]
pub(crate) struct RetainedSemanticBlockProducerV1 {
    pub(crate) identity: SemanticBlockIdentityV1,
    pub(crate) rustc_block: u32,
    pub(crate) source: RetainedSemanticSourceProducerV1,
    pub(crate) statements: Box<[RetainedSemanticSourceProducerV1]>,
    pub(crate) terminator: RetainedSemanticSourceProducerV1,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RetainedSemanticSourceProducerV1 {
    pub(crate) provenance: SemanticSourceProvenanceV1,
    expansion_chain_sha256: [u8; 32],
}

#[derive(Debug)]
struct RetainedRawBlockSourceProducerV1 {
    source: RetainedSemanticSourceProducerV1,
    statements: Box<[RetainedSemanticSourceProducerV1]>,
    terminator: RetainedSemanticSourceProducerV1,
}

#[derive(Debug)]
struct RetainedRawBodySourceProducerV1 {
    source: RetainedSemanticSourceProducerV1,
    locals: Box<[RetainedSemanticSourceProducerV1]>,
    blocks: Box<[RetainedRawBlockSourceProducerV1]>,
    debug_scopes: Box<[RetainedRawDebugSourceScopeV2]>,
    debug_variables: Box<[RetainedRawDebugSourceVariableV2]>,
    debug_capture_gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
}

#[derive(Clone, Copy, Debug)]
struct RetainedRawDebugSourceScopeV2 {
    raw_scope: u32,
    parent_raw_scope: Option<u32>,
    depth: u32,
    source: RetainedSemanticSourceProducerV1,
}

#[derive(Clone, Debug)]
struct RetainedRawDebugSourceVariableV2 {
    ordinal: u32,
    exact_name: Option<String>,
    name_sha256: [u8; 32],
    raw_scope: u32,
    class: RetainedRawDebugSourceVariableClassV2,
    entry_value_preserved: bool,
}

#[derive(Clone, Copy, Debug)]
enum RetainedRawDebugSourceVariableClassV2 {
    Local(u32),
    Unrepresented,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RetainedDebugSourceScopeV2 {
    pub(crate) identity: [u8; 32],
    pub(crate) function: SemanticFunctionIdV1,
    pub(crate) parent_identity: Option<[u8; 32]>,
    pub(crate) depth: u32,
    pub(crate) source: SemanticSourceProvenanceV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RetainedDebugSourceVariableClassV2 {
    Local(SemanticLocalIdV1),
    Unrepresented,
}

#[derive(Clone, Debug)]
pub(crate) struct RetainedDebugSourceVariableV2 {
    pub(crate) identity: [u8; 32],
    pub(crate) function: SemanticFunctionIdV1,
    pub(crate) name: Option<String>,
    pub(crate) scope_identity: [u8; 32],
    pub(crate) class: RetainedDebugSourceVariableClassV2,
    pub(crate) entry_value_preserved: bool,
}

#[derive(Debug)]
pub(crate) struct RetainedSemanticBodyProducerV1 {
    pub(crate) function: SemanticFunctionIdV1,
    pub(crate) source: RetainedSemanticSourceProducerV1,
    pub(crate) locals: Box<[RetainedSemanticLocalProducerV1]>,
    pub(crate) raw_to_semantic_locals: Box<[SemanticLocalIdV1]>,
    pub(crate) entry: SemanticBlockIdV1,
    pub(crate) blocks: Box<[RetainedSemanticBlockProducerV1]>,
    pub(crate) raw_to_semantic_blocks: Box<[SemanticBlockIdV1]>,
    pub(crate) debug_scopes: Box<[RetainedDebugSourceScopeV2]>,
    pub(crate) debug_variables: Box<[RetainedDebugSourceVariableV2]>,
    pub(crate) debug_capture_gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
}

struct CanonicalProducerTablesV1<'tcx> {
    types: Box<[RetainedSemanticTypeProducerV1<'tcx>]>,
    source_files: Box<[SemanticSourceFileIdentityV1]>,
    bodies: Box<[RetainedSemanticBodyProducerV1]>,
}

type RetainedTerminalTableV1<'tcx> = (
    Box<[RetainedSemanticTerminalProducerV1<'tcx>]>,
    BTreeMap<SemanticFunctionIdentityV1, u32>,
);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RawMirPreflightCountsV1 {
    types: u64,
    functions: u64,
    roots: u64,
    locals: u64,
    blocks: u64,
    statements: u64,
    projections: u64,
    operands: u64,
    call_arguments: u64,
    switch_targets: u64,
    validation_work: u64,
    constant_bytes: u64,
}

impl RawMirPreflightCountsV1 {
    fn slot_mut(&mut self, resource: SemanticMirResourceV1) -> Option<&mut u64> {
        match resource {
            SemanticMirResourceV1::Types => Some(&mut self.types),
            SemanticMirResourceV1::Functions => Some(&mut self.functions),
            SemanticMirResourceV1::Roots => Some(&mut self.roots),
            SemanticMirResourceV1::Locals => Some(&mut self.locals),
            SemanticMirResourceV1::Blocks => Some(&mut self.blocks),
            SemanticMirResourceV1::Statements => Some(&mut self.statements),
            SemanticMirResourceV1::Projections => Some(&mut self.projections),
            SemanticMirResourceV1::Operands => Some(&mut self.operands),
            SemanticMirResourceV1::CallArguments => Some(&mut self.call_arguments),
            SemanticMirResourceV1::SwitchTargets => Some(&mut self.switch_targets),
            SemanticMirResourceV1::ValidationWork => Some(&mut self.validation_work),
            SemanticMirResourceV1::ConstantBytes => Some(&mut self.constant_bytes),
            SemanticMirResourceV1::Callables
            | SemanticMirResourceV1::Allocations
            | SemanticMirResourceV1::Statics
            | SemanticMirResourceV1::VTables
            | SemanticMirResourceV1::Relocations
            | SemanticMirResourceV1::LinkSymbolBytes
            | SemanticMirResourceV1::CanonicalBytes => None,
        }
    }

    fn charge(
        &mut self,
        resource: SemanticMirResourceV1,
        amount: usize,
        limits: SemanticMirLimitsV1,
    ) -> Result<(), ProductionSemanticPreflightErrorV1> {
        let amount = u64::try_from(amount).unwrap_or(u64::MAX);
        let Some(slot) = self.slot_mut(resource) else {
            return Err(ProductionSemanticPreflightErrorV1::AccountingDomain { resource });
        };
        *slot =
            slot.checked_add(amount)
                .ok_or(ProductionSemanticPreflightErrorV1::LimitExceeded {
                    resource,
                    actual: u64::MAX,
                    maximum: limits.limit(resource),
                })?;
        let maximum = limits.limit(resource);
        if *slot > maximum {
            return Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
                resource,
                actual: *slot,
                maximum,
            });
        }
        Ok(())
    }

    fn digest_fields(self) -> [u64; 11] {
        // Preserve V5's fixed header. Borrowed normalization literals are not
        // emitted constants; their exact bytes and this private quota's total
        // are checked together in the normalized-intrinsics child below.
        [
            self.types,
            self.functions,
            self.roots,
            self.locals,
            self.blocks,
            self.statements,
            self.projections,
            self.operands,
            self.call_arguments,
            self.switch_targets,
            self.validation_work,
        ]
    }
}

/// Carries one validation-work budget through collection, closure reobservation,
/// and semantic preflight. Consumed when preflight resumes the raw MIR counts.
#[derive(Debug, Default)]
pub(crate) struct SourceClosureWorkV1 {
    limits: SemanticMirLimitsV1,
    counts: RawMirPreflightCountsV1,
    identity: Option<std::sync::Arc<()>>,
}

/// A process-local account comparison, never serialized admission evidence.
#[derive(Debug)]
pub(crate) struct SourceClosureWorkStampV1 {
    identity: std::sync::Weak<()>,
}

impl SourceClosureWorkV1 {
    pub(crate) fn retain_identity_v1(
        &mut self,
    ) -> Result<SourceClosureWorkStampV1, ProductionSemanticPreflightErrorV1> {
        // Charge the logical retained payload and identity operation before
        // allocation. This does not claim to measure allocator metadata.
        self.charge(
            1 + std::mem::size_of::<SourceClosureWorkStampV1>() + 2 * std::mem::size_of::<usize>(),
        )?;
        let identity = self.identity.get_or_insert_with(|| std::sync::Arc::new(()));
        Ok(SourceClosureWorkStampV1 {
            identity: std::sync::Arc::downgrade(identity),
        })
    }

    pub(crate) fn matches_identity_v1(
        &mut self,
        stamp: &SourceClosureWorkStampV1,
    ) -> Result<bool, ProductionSemanticPreflightErrorV1> {
        self.charge(1)?;
        Ok(self.identity.as_ref().is_some_and(|current| {
            stamp
                .identity
                .upgrade()
                .is_some_and(|original| std::sync::Arc::ptr_eq(current, &original))
        }))
    }

    pub(crate) fn limits(&self) -> SemanticMirLimitsV1 {
        self.limits
    }

    #[cfg(test)]
    pub(crate) fn validation_work_for_test(&self) -> u64 {
        self.counts.validation_work
    }

    pub(crate) fn charge(
        &mut self,
        amount: usize,
    ) -> Result<(), ProductionSemanticPreflightErrorV1> {
        self.counts
            .charge(SemanticMirResourceV1::ValidationWork, amount, self.limits)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CallEdgeV1 {
    caller: SemanticFunctionIdV1,
    callee: SemanticFunctionIdV1,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct DirectCallRecipeV1 {
    pub(crate) caller: SemanticFunctionIdV1,
    pub(crate) block: u32,
    pub(crate) callee: SemanticFunctionIdV1,
}

#[derive(Debug)]
pub(crate) struct RetainedSemanticTerminalProducerV1<'tcx> {
    pub(crate) identities: CanonicalFunctionIdentitiesV1,
    pub(crate) instance: Instance<'tcx>,
    pub(crate) expansion: ProductionTerminalExpansionV1,
    pub(crate) abi: RetainedSemanticFunctionAbiProducerV1<'tcx>,
    pub(crate) source: RetainedSemanticSourceProducerV1,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TerminalExpansionRecipeV1<'tcx> {
    pub(crate) caller: SemanticFunctionIdV1,
    pub(crate) block: u32,
    pub(crate) expansion: ProductionTerminalExpansionV1,
    pub(crate) arguments: u32,
    pub(crate) instance: Instance<'tcx>,
    pub(crate) identities: CanonicalFunctionIdentitiesV1,
    pub(crate) terminal: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NormalizedRustcIntrinsicRecipeV1<'tcx> {
    pub(crate) caller: SemanticFunctionIdV1,
    pub(crate) block: u32,
    pub(crate) operation: NormalizedCallV1<'tcx>,
    pub(crate) element_type: Ty<'tcx>,
    pub(crate) instance: Instance<'tcx>,
    pub(crate) identities: CanonicalFunctionIdentitiesV1,
}

#[derive(Debug)]
pub(crate) struct ProductionSemanticConstructionWorkV1 {
    limits: SemanticMirLimitsV1,
    validation_work: u64,
}

impl ProductionSemanticConstructionWorkV1 {
    pub(crate) fn into_parts(self) -> (SemanticMirLimitsV1, u64) {
        (self.limits, self.validation_work)
    }

    #[cfg(test)]
    pub(crate) fn for_test(limits: SemanticMirLimitsV1, validation_work: u64) -> Self {
        Self {
            limits,
            validation_work,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ProductionSemanticPreflightPlanV1<'tcx> {
    types: Box<[RetainedSemanticTypeProducerV1<'tcx>]>,
    functions: Box<[RetainedSemanticFunctionProducerV1<'tcx>]>,
    function_abis: Box<[RetainedSemanticFunctionAbiProducerV1<'tcx>]>,
    terminals: Box<[RetainedSemanticTerminalProducerV1<'tcx>]>,
    bodies: Box<[RetainedSemanticBodyProducerV1]>,
    roots: Box<[SemanticFunctionIdV1]>,
    direct_calls: Box<[DirectCallRecipeV1]>,
    terminal_expansions: Box<[TerminalExpansionRecipeV1<'tcx>]>,
    normalized_intrinsics: Box<[NormalizedRustcIntrinsicRecipeV1<'tcx>]>,
    debug_source_files: Box<[fe2o3_kernel_ir::DebugSourceMapFileV1]>,
    debug_capture_gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
    sha256: [u8; 32],
    canonical_transcript: Box<[u8]>,
    context_entries: BTreeMap<SemanticFunctionIdV1, BoundContextEntryV29<'tcx>>,
    construction_work: Option<ProductionSemanticConstructionWorkV1>,
}

impl<'tcx> ProductionSemanticPreflightPlanV1<'tcx> {
    pub(crate) fn take_construction_work(
        &mut self,
    ) -> Result<ProductionSemanticConstructionWorkV1, ProductionSemanticPreflightErrorV1> {
        self.construction_work
            .take()
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)
    }

    pub(crate) fn take_context_entries_v29(
        &mut self,
    ) -> BTreeMap<SemanticFunctionIdV1, BoundContextEntryV29<'tcx>> {
        std::mem::take(&mut self.context_entries)
    }

    pub(crate) fn type_producers(&self) -> &[RetainedSemanticTypeProducerV1<'tcx>] {
        &self.types
    }

    pub(crate) fn function_producers(&self) -> &[RetainedSemanticFunctionProducerV1<'tcx>] {
        &self.functions
    }

    pub(crate) fn function_abi_producers(&self) -> &[RetainedSemanticFunctionAbiProducerV1<'tcx>] {
        &self.function_abis
    }

    pub(crate) fn terminal_producers(&self) -> &[RetainedSemanticTerminalProducerV1<'tcx>] {
        &self.terminals
    }

    pub(crate) fn body_producers(&self) -> &[RetainedSemanticBodyProducerV1] {
        &self.bodies
    }

    pub(crate) fn roots(&self) -> &[SemanticFunctionIdV1] {
        &self.roots
    }

    pub(crate) fn direct_call_producers(&self) -> &[DirectCallRecipeV1] {
        &self.direct_calls
    }

    pub(crate) fn terminal_expansion_producers(&self) -> &[TerminalExpansionRecipeV1<'tcx>] {
        &self.terminal_expansions
    }

    pub(crate) fn normalized_intrinsic_producers(
        &self,
    ) -> &[NormalizedRustcIntrinsicRecipeV1<'tcx>] {
        &self.normalized_intrinsics
    }

    pub(crate) fn canonical_transcript(&self) -> &[u8] {
        &self.canonical_transcript
    }

    pub(crate) fn into_identity_transcript_and_debug_files(
        self,
    ) -> Result<
        (
            [u8; 32],
            Box<[u8]>,
            Box<[fe2o3_kernel_ir::DebugSourceMapFileV1]>,
            Box<[RetainedDebugSourceScopeV2]>,
            Box<[RetainedDebugSourceVariableV2]>,
            Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
        ),
        ProductionSemanticPreflightErrorV1,
    > {
        if !self.context_entries.is_empty() {
            return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
        }
        let scope_count = self.bodies.iter().try_fold(0_usize, |count, body| {
            count.checked_add(body.debug_scopes.len())
        });
        let variable_count = self.bodies.iter().try_fold(0_usize, |count, body| {
            count.checked_add(body.debug_variables.len())
        });
        let mut debug_capture_gap = self.debug_capture_gap;
        let (scope_count, variable_count) = match (scope_count, variable_count) {
            (Some(scope_count), Some(variable_count)) => (scope_count, variable_count),
            _ => {
                debug_capture_gap =
                    Some(fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::ResourceLimit);
                (0, 0)
            }
        };
        let mut debug_source_scopes = Vec::new();
        let mut debug_source_variables = Vec::new();
        let reserved = debug_source_scopes.try_reserve_exact(scope_count).is_ok()
            && debug_source_variables
                .try_reserve_exact(variable_count)
                .is_ok();
        if reserved && debug_capture_gap.is_none() {
            for body in self.bodies.into_vec() {
                debug_source_scopes.extend(body.debug_scopes.into_vec());
                debug_source_variables.extend(body.debug_variables.into_vec());
            }
        } else if !reserved {
            debug_capture_gap =
                Some(fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::ResourceLimit);
        }
        Ok((
            self.sha256,
            self.canonical_transcript,
            self.debug_source_files,
            debug_source_scopes.into_boxed_slice(),
            debug_source_variables.into_boxed_slice(),
            debug_capture_gap,
        ))
    }
}

#[derive(Debug)]
pub(crate) enum ProductionSemanticPreflightErrorV1 {
    LimitExceeded {
        resource: SemanticMirResourceV1,
        actual: u64,
        maximum: u64,
    },
    AccountingDomain {
        resource: SemanticMirResourceV1,
    },
    CommitmentBoundExceeded {
        scope: &'static str,
        actual: u64,
        maximum: u64,
    },
    TypeIdentityCollision,
    IdentityTableMismatch,
    TypeLayout {
        identity: SemanticTypeIdentityV1,
        detail: String,
    },
    FunctionAbi {
        function: SemanticFunctionIdV1,
        detail: String,
    },
    UnsupportedRustcMir {
        construct: String,
        function: String,
        call_chain: Box<[String]>,
        location: String,
        source: String,
    },
}

impl fmt::Display for ProductionSemanticPreflightErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded {
                resource,
                actual,
                maximum,
            } => write!(
                formatter,
                "raw rustc MIR preflight rejected {resource:?} count {actual}; maximum is {maximum}",
            ),
            Self::AccountingDomain { resource } => write!(
                formatter,
                "raw rustc MIR preflight attempted to charge non-raw resource {resource:?}",
            ),
            Self::CommitmentBoundExceeded {
                scope,
                actual,
                maximum,
            } => write!(
                formatter,
                "raw rustc MIR preflight {scope} commitment uses {actual} canonical bytes; maximum is {maximum}",
            ),
            Self::TypeIdentityCollision => formatter.write_str(
                "raw rustc MIR preflight derived one type identity for distinct normalized rustc types",
            ),
            Self::IdentityTableMismatch => formatter.write_str(
                "raw rustc MIR preflight could not assign a complete canonical producer table",
            ),
            Self::TypeLayout { identity, detail } => write!(
                formatter,
                "raw rustc MIR preflight could not obtain target layout for type producer {}: {detail}",
                crate::encode_hex(identity.as_bytes()),
            ),
            Self::FunctionAbi { function, detail } => write!(
                formatter,
                "raw rustc MIR preflight could not obtain the role-adjusted ABI for function {}: {detail}",
                function.index(),
            ),
            Self::UnsupportedRustcMir {
                construct,
                function,
                call_chain,
                location,
                source,
            } => write!(
                formatter,
                "raw rustc MIR preflight rejected {construct} in {function} at {location} ({source}); reachable call chain: {}",
                call_chain.join(" -> "),
            ),
        }
    }
}

impl std::error::Error for ProductionSemanticPreflightErrorV1 {}

#[derive(Clone, Copy)]
struct RejectionSiteV1 {
    function: SemanticFunctionIdV1,
    block: Option<u32>,
    statement: Option<u32>,
    local: Option<u32>,
    span: Span,
}

enum PendingRejectionV1 {
    Unsupported {
        construct: String,
        site: RejectionSiteV1,
    },
    Fatal(ProductionSemanticPreflightErrorV1),
}

struct BodyPreflightV1<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'a Body<'tcx>,
    function: SemanticFunctionIdV1,
    limits: SemanticMirLimitsV1,
    counts: &'a mut RawMirPreflightCountsV1,
    types: &'a mut BTreeMap<SemanticTypeIdentityV1, Ty<'tcx>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DebugSourceCaptureRequestV2 {
    Disabled,
    SourceVariables,
}

#[derive(Default)]
struct OptionalDebugCaptureBudgetV2 {
    variables: usize,
    scopes: usize,
}

impl OptionalDebugCaptureBudgetV2 {
    fn charge_variables(
        &mut self,
        amount: usize,
    ) -> Result<(), ProductionSemanticPreflightErrorV1> {
        self.variables = charge_optional_debug_count_v2(
            self.variables,
            amount,
            fe2o3_kernel_ir::MAX_DEBUG_SOURCE_VARIABLES_V2,
        )?;
        Ok(())
    }

    fn charge_scopes(&mut self, amount: usize) -> Result<(), ProductionSemanticPreflightErrorV1> {
        self.scopes = charge_optional_debug_count_v2(
            self.scopes,
            amount,
            fe2o3_kernel_ir::MAX_DEBUG_SOURCE_SCOPES_V2,
        )?;
        Ok(())
    }
}

fn charge_optional_debug_count_v2(
    current: usize,
    amount: usize,
    maximum: usize,
) -> Result<usize, ProductionSemanticPreflightErrorV1> {
    let actual = current
        .checked_add(amount)
        .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
    if actual > maximum {
        return Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::Locals,
            actual: u64::try_from(actual).unwrap_or(u64::MAX),
            maximum: maximum as u64,
        });
    }
    Ok(actual)
}

#[cfg(test)]
pub(crate) fn build_production_semantic_preflight_plan_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    functions: Box<[RetainedSemanticFunctionProducerV1<'tcx>]>,
    roots: Box<[SemanticFunctionIdV1]>,
    identity_inventory_sha256: [u8; 32],
    debug_source_capture: DebugSourceCaptureRequestV2,
    context_entries: Option<AuthenticatedContextEntriesV1<'tcx>>,
) -> Result<ProductionSemanticPreflightPlanV1<'tcx>, ProductionSemanticPreflightErrorV1> {
    build_production_semantic_preflight_plan_with_work_v1(
        tcx,
        target,
        functions,
        roots,
        identity_inventory_sha256,
        debug_source_capture,
        (context_entries, SourceClosureWorkV1::default()),
    )
}

pub(crate) fn build_production_semantic_preflight_plan_with_work_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    functions: Box<[RetainedSemanticFunctionProducerV1<'tcx>]>,
    roots: Box<[SemanticFunctionIdV1]>,
    identity_inventory_sha256: [u8; 32],
    debug_source_capture: DebugSourceCaptureRequestV2,
    (context_entries, work): (
        Option<AuthenticatedContextEntriesV1<'tcx>>,
        SourceClosureWorkV1,
    ),
) -> Result<ProductionSemanticPreflightPlanV1<'tcx>, ProductionSemanticPreflightErrorV1> {
    let SourceClosureWorkV1 {
        limits,
        mut counts,
        identity: _,
    } = work;
    counts.charge(SemanticMirResourceV1::Functions, functions.len(), limits)?;
    counts.charge(SemanticMirResourceV1::Roots, roots.len(), limits)?;

    let function_ids = functions
        .iter()
        .enumerate()
        .map(|(index, function)| {
            (
                function.identities.function(),
                SemanticFunctionIdV1::from_index(index as u32),
            )
        })
        .collect::<BTreeMap<_, _>>();

    // Structural cardinalities are charged before pass one traverses their
    // contents. Call discovery is therefore bounded by the same limits as
    // later canonical construction.
    for function in &functions {
        let body = tcx.instance_mir(function.instance.def);
        counts.charge(
            SemanticMirResourceV1::Locals,
            body.local_decls.len(),
            limits,
        )?;
        counts.charge(
            SemanticMirResourceV1::Blocks,
            body.basic_blocks.len(),
            limits,
        )?;
        for data in body.basic_blocks.iter() {
            counts.charge(
                SemanticMirResourceV1::Statements,
                data.statements.len(),
                limits,
            )?;
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::Call { args, .. } = &terminator.kind
            {
                counts.charge(SemanticMirResourceV1::CallArguments, args.len(), limits)?;
            }
        }
    }

    // Pass one resolves the complete direct-call relation and freezes typed
    // terminal recipes before any body is accepted as representable.
    let mut edges = BTreeSet::new();
    let mut direct_calls = BTreeSet::new();
    let mut terminal_expansions = Vec::new();
    let mut context_issuances = 0;
    let mut normalized_intrinsics = Vec::new();
    let mut has_safe_core_shift = false;
    let mut first_rejection = None;
    for (index, function) in functions.iter().enumerate() {
        let function_id = SemanticFunctionIdV1::from_index(index as u32);
        let body = tcx.instance_mir(function.instance.def);
        for (block, data) in body.basic_blocks.iter_enumerated() {
            let Some(terminator) = &data.terminator else {
                remember_rejection(
                    &mut first_rejection,
                    "basic block without a terminator",
                    RejectionSiteV1 {
                        function: function_id,
                        block: Some(block.index() as u32),
                        statement: None,
                        local: None,
                        span: body.span,
                    },
                );
                continue;
            };
            let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
                continue;
            };
            let site = RejectionSiteV1 {
                function: function_id,
                block: Some(block.index() as u32),
                statement: None,
                local: None,
                span: terminator.source_info.span,
            };
            match resolve_direct_call_v1(tcx, function.instance, body, func) {
                Ok(resolved) => {
                    if crate::production_core_panic_v50::is_candidate(tcx, func) {
                        match crate::production_core_panic_v50::observe(
                            tcx,
                            function.instance,
                            body,
                            block,
                            &mut |amount| {
                                counts.charge(SemanticMirResourceV1::ValidationWork, amount, limits)
                            },
                        ) {
                            Ok(Some(checked)) if checked.instance() == resolved => {
                                normalized_intrinsics.push(NormalizedRustcIntrinsicRecipeV1 {
                                    caller: function_id,
                                    block: block.index() as u32,
                                    operation: NormalizedCallV1::CorePanic(checked),
                                    element_type: tcx.types.never,
                                    instance: resolved,
                                    identities: canonical_function_identities_v1(tcx, resolved),
                                });
                            }
                            Ok(_) => remember_rejection(
                                &mut first_rejection,
                                "core panic normalized callee differs",
                                site,
                            ),
                            Err(crate::production_core_panic_v50::CorePanicErrorV50::Work(
                                error,
                            )) => {
                                return Err(error);
                            }
                            Err(crate::production_core_panic_v50::CorePanicErrorV50::Refused(
                                reason,
                            )) => {
                                remember_rejection(&mut first_rejection, reason, site);
                            }
                        }
                        continue;
                    }
                    match SafeCoreShiftV1::classify(tcx, resolved) {
                        Ok(Some(shift)) => {
                            has_safe_core_shift = true;
                            if args.len() != 2 {
                                remember_rejection(
                                    &mut first_rejection,
                                    "safe core wrapping shift call arity",
                                    site,
                                );
                                continue;
                            }
                            counts.charge(SemanticMirResourceV1::Locals, 2, limits)?;
                            counts.charge(SemanticMirResourceV1::Statements, 3, limits)?;
                            counts.charge(SemanticMirResourceV1::Operands, 5, limits)?;
                            counts.charge(SemanticMirResourceV1::ValidationWork, 1, limits)?;
                            normalized_intrinsics.push(NormalizedRustcIntrinsicRecipeV1 {
                                caller: function_id,
                                block: block.index() as u32,
                                operation: NormalizedCallV1::SafeCoreShift(shift),
                                element_type: shift.value_type(),
                                instance: resolved,
                                identities: canonical_function_identities_v1(tcx, resolved),
                            });
                            continue;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            remember_rejection(&mut first_rejection, error, site);
                            continue;
                        }
                    }
                    match crate::production_primitive_from_v1::check_primitive_from_v1(
                        tcx,
                        resolved,
                        crate::production_primitive_from_v1::PrimitiveFromStageV1::Preflight,
                        limits,
                        &mut |amount| {
                            counts.charge(SemanticMirResourceV1::ValidationWork, amount, limits)
                        },
                    ) {
                        Ok(Some(checked)) => {
                            if args.len() != 1 {
                                remember_rejection(
                                    &mut first_rejection,
                                    "checked primitive From call arity",
                                    site,
                                );
                                continue;
                            }
                            counts.charge(SemanticMirResourceV1::Statements, 1, limits)?;
                            counts.charge(SemanticMirResourceV1::Operands, 1, limits)?;
                            normalized_intrinsics.push(NormalizedRustcIntrinsicRecipeV1 {
                                caller: function_id,
                                block: block.index() as u32,
                                operation: NormalizedCallV1::CheckedPrimitiveFrom(checked),
                                element_type: checked.output_type(),
                                instance: resolved,
                                identities: canonical_function_identities_v1(tcx, resolved),
                            });
                            continue;
                        }
                        Ok(None) => {}
                        Err(crate::production_primitive_from_v1::PrimitiveFromErrorV1::Work {
                            source,
                            ..
                        }) => return Err(source),
                        Err(error) => {
                            remember_rejection(&mut first_rejection, error.to_string(), site);
                            continue;
                        }
                    }
                    match crate::production_semantic_terminal_v1::classify(tcx, resolved.def_id()) {
                        Some(ProductionSemanticTerminalRuleV1::Expand(expansion)) => {
                            context_issuances += usize::from(
                                expansion == ProductionTerminalExpansionV1::ContextIssue,
                            );
                            let Ok(arguments) = u32::try_from(args.len()) else {
                                remember_rejection(
                                    &mut first_rejection,
                                    "terminal call argument count outside u32",
                                    site,
                                );
                                continue;
                            };
                            terminal_expansions.push(TerminalExpansionRecipeV1 {
                                caller: function_id,
                                block: block.index() as u32,
                                expansion,
                                arguments,
                                instance: resolved,
                                identities: canonical_function_identities_v1(tcx, resolved),
                                terminal: u32::MAX,
                            });
                        }
                        Some(ProductionSemanticTerminalRuleV1::Reject(item)) => {
                            remember_rejection(
                                &mut first_rejection,
                                format!("reviewed terminal without production expansion: {item:?}"),
                                site,
                            );
                        }
                        None => {
                            match crate::production_rustc_intrinsic_v1::classify(tcx, resolved) {
                                Ok(Some(classification)) => {
                                    if classification.operation
                                        == ProductionRustcIntrinsicOperationV1::FabsF32
                                    {
                                        if args.len() != 1 {
                                            remember_rejection(
                                                &mut first_rejection,
                                                "fabs intrinsic with unexpected call arity",
                                                site,
                                            );
                                            continue;
                                        }
                                        terminal_expansions.push(TerminalExpansionRecipeV1 {
                                            caller: function_id,
                                            block: block.index() as u32,
                                            expansion: ProductionTerminalExpansionV1::RustcFabsF32,
                                            arguments: 1,
                                            instance: resolved,
                                            identities: canonical_function_identities_v1(
                                                tcx, resolved,
                                            ),
                                            terminal: u32::MAX,
                                        });
                                        continue;
                                    }
                                    if let ProductionRustcIntrinsicOperationV1::SaturatingInteger(
                                        operation,
                                    ) = classification.operation
                                    {
                                        if args.len() != 2 {
                                            remember_rejection(
                                                &mut first_rejection,
                                                "saturating integer intrinsic with unexpected call arity",
                                                site,
                                            );
                                            continue;
                                        }
                                        terminal_expansions.push(TerminalExpansionRecipeV1 {
                                            caller: function_id,
                                            block: block.index() as u32,
                                            expansion: ProductionTerminalExpansionV1::RustcSaturatingInteger(operation),
                                            arguments: 2,
                                            instance: resolved,
                                            identities: canonical_function_identities_v1(tcx, resolved),
                                            terminal: u32::MAX,
                                        });
                                        continue;
                                    }
                                    if args.len() != 2 {
                                        remember_rejection(
                                            &mut first_rejection,
                                            "normalized atomic intrinsic with unexpected call arity",
                                            site,
                                        );
                                        continue;
                                    }
                                    counts.charge(SemanticMirResourceV1::Statements, 1, limits)?;
                                    counts.charge(SemanticMirResourceV1::Projections, 1, limits)?;
                                    normalized_intrinsics.push(NormalizedRustcIntrinsicRecipeV1 {
                                        caller: function_id,
                                        block: block.index() as u32,
                                        operation: NormalizedCallV1::Rustc(
                                            classification.operation,
                                        ),
                                        element_type: classification.element_type,
                                        instance: resolved,
                                        identities: canonical_function_identities_v1(tcx, resolved),
                                    });
                                }
                                Ok(None) => {
                                    let identity =
                                        canonical_function_identities_v1(tcx, resolved).function();
                                    if let Some(callee) = function_ids.get(&identity).copied() {
                                        edges.insert(CallEdgeV1 {
                                            caller: function_id,
                                            callee,
                                        });
                                        direct_calls.insert(DirectCallRecipeV1 {
                                            caller: function_id,
                                            block: block.index() as u32,
                                            callee,
                                        });
                                    } else {
                                        remember_rejection(
                                            &mut first_rejection,
                                            "direct call escaped the collector-sealed function closure",
                                            site,
                                        );
                                    }
                                }
                                Err(error) => remember_rejection(
                                    &mut first_rejection,
                                    format!("unsupported rustc compiler intrinsic: {error}"),
                                    site,
                                ),
                            }
                        }
                    }
                }
                Err(construct) => remember_rejection(&mut first_rejection, construct, site),
            }
        }
    }

    if let Some(rejection) = first_rejection {
        return Err(materialize_rejection_v1(
            tcx, &functions, &roots, &edges, rejection,
        ));
    }

    if let Some(unreachable) = first_unreachable_function_v1(&roots, &edges, functions.len()) {
        let function = &functions[unreachable.index() as usize];
        return Err(materialize_rejection_v1(
            tcx,
            &functions,
            &roots,
            &edges,
            reject(
                "collector-sealed function without a root call path",
                RejectionSiteV1 {
                    function: unreachable,
                    block: None,
                    statement: None,
                    local: None,
                    span: tcx.instance_mir(function.instance.def).span,
                },
            ),
        ));
    }

    let (terminals, terminal_ids) =
        build_terminal_producers_v1(tcx, target, &terminal_expansions, &mut counts, limits)?;
    for recipe in &mut terminal_expansions {
        recipe.terminal = terminal_ids
            .get(&recipe.identities.function())
            .copied()
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
    }
    let function_abis = build_function_abi_producers_v1(tcx, target, &functions)?;

    // Pass two walks every raw MIR node once, classifies the first supported
    // subset, and charges only resources directly observed in rustc MIR.
    let mut types = BTreeMap::new();
    let mut source_producers = Vec::with_capacity(functions.len());
    let mut receiver_reborrows = Vec::with_capacity(functions.len());
    let mut source_cache = HashMap::new();
    let mut debug_source_files = BTreeMap::new();
    let mut debug_counts = RawMirPreflightCountsV1::default();
    let mut debug_budget = OptionalDebugCaptureBudgetV2::default();
    let mut debug_capture_gap = None;
    for (index, function) in functions.iter().enumerate() {
        let function_id = SemanticFunctionIdV1::from_index(index as u32);
        let body = tcx.instance_mir(function.instance.def);
        let sources = capture_body_sources_v1(
            tcx,
            function_id,
            body,
            &mut source_cache,
            &mut debug_source_files,
            &mut counts,
            &mut debug_counts,
            &mut debug_budget,
            limits,
            if debug_capture_gap.is_none() {
                debug_source_capture
            } else {
                DebugSourceCaptureRequestV2::Disabled
            },
        )
        .map_err(|rejection| {
            materialize_rejection_v1(tcx, &functions, &roots, &edges, rejection)
        })?;
        if debug_capture_gap.is_none() {
            debug_capture_gap = sources.debug_capture_gap;
        }
        source_producers.push(sources);
        let mut preflight = BodyPreflightV1 {
            tcx,
            instance: function.instance,
            body,
            function: function_id,
            limits,
            counts: &mut counts,
            types: &mut types,
        };
        if let Err(rejection) = preflight.inspect_body() {
            return Err(materialize_rejection_v1(
                tcx, &functions, &roots, &edges, rejection,
            ));
        }
        let abi = function_abis
            .get(index)
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        let abi_site = RejectionSiteV1 {
            function: function_id,
            block: None,
            statement: None,
            local: None,
            span: body.span,
        };
        for ty in abi
            .source_inputs
            .iter()
            .copied()
            .chain(std::iter::once(abi.source_output))
            .chain(abi.fn_abi.args.iter().map(|argument| argument.layout.ty))
            .chain(std::iter::once(abi.fn_abi.ret.layout.ty))
        {
            if let Err(rejection) = preflight.inspect_type(ty, abi_site) {
                return Err(materialize_rejection_v1(
                    tcx, &functions, &roots, &edges, rejection,
                ));
            }
        }
        let receiver_reborrow =
            derive_fn_receiver_reborrow_v1(tcx, function.instance, body, |amount| {
                counts.charge(SemanticMirResourceV1::ValidationWork, amount, limits)
            })
            .map_err(|error| match error {
                ReceiverReborrowErrorV1::Resource(error) => error,
                ReceiverReborrowErrorV1::Unsupported(construct) => materialize_rejection_v1(
                    tcx,
                    &functions,
                    &roots,
                    &edges,
                    reject(construct, abi_site),
                ),
            })?;
        if let Some(reborrow) = receiver_reborrow {
            let callee_identity =
                canonical_function_identities_v1(tcx, reborrow.callee()).function();
            let callee = function_ids
                .get(&callee_identity)
                .copied()
                .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            if function.role != CollectedFunctionRole::InternalHelper
                || !direct_calls.contains(&DirectCallRecipeV1 {
                    caller: function_id,
                    block: reborrow.block(),
                    callee,
                })
            {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            }
            for resource in [
                SemanticMirResourceV1::Locals,
                SemanticMirResourceV1::Statements,
                SemanticMirResourceV1::Projections,
            ] {
                counts.charge(resource, 1, limits)?;
            }
        }
        receiver_reborrows.push(receiver_reborrow);
    }

    for terminal in &terminals {
        let recipe = terminal_expansions
            .iter()
            .find(|recipe| recipe.terminal == terminal.abi.function.index())
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        let caller = functions
            .get(recipe.caller.index() as usize)
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        let body = tcx.instance_mir(caller.instance.def);
        let mut preflight = BodyPreflightV1 {
            tcx,
            instance: caller.instance,
            body,
            function: recipe.caller,
            limits,
            counts: &mut counts,
            types: &mut types,
        };
        let site = RejectionSiteV1 {
            function: recipe.caller,
            block: Some(recipe.block),
            statement: None,
            local: None,
            span: body.basic_blocks
                [rustc_middle::mir::BasicBlock::from_usize(recipe.block as usize)]
            .terminator()
            .source_info
            .span,
        };
        for ty in terminal
            .abi
            .source_inputs
            .iter()
            .copied()
            .chain(std::iter::once(terminal.abi.source_output))
            .chain(
                terminal
                    .abi
                    .fn_abi
                    .args
                    .iter()
                    .map(|argument| argument.layout.ty),
            )
            .chain(std::iter::once(terminal.abi.fn_abi.ret.layout.ty))
        {
            if let Err(rejection) = preflight.inspect_type(ty, site) {
                return Err(materialize_rejection_v1(
                    tcx, &functions, &roots, &edges, rejection,
                ));
            }
        }
    }

    let CanonicalProducerTablesV1 {
        types,
        source_files,
        bodies,
    } = build_canonical_producer_tables_v1(
        tcx,
        target,
        &functions,
        source_producers,
        types,
        receiver_reborrows,
        has_safe_core_shift.then_some(normalized_intrinsics.as_slice()),
        (&mut counts, &mut debug_counts, limits),
    )?;
    let debug_capture_gap = bodies.iter().find_map(|body| body.debug_capture_gap);

    let direct_calls = direct_calls.into_iter().collect::<Box<[_]>>();
    let terminal_expansions = terminal_expansions.into_boxed_slice();
    let normalized_intrinsics = normalized_intrinsics.into_boxed_slice();
    let context_entries = if let Some(entries) = context_entries {
        entries.bind_semantic_v29(
            &functions,
            &bodies,
            &terminal_expansions,
            &direct_calls,
            context_issuances,
            |amount| counts.charge(SemanticMirResourceV1::ValidationWork, amount, limits),
        )?
    } else {
        if context_issuances != 0 {
            return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
        }
        BTreeMap::new()
    };
    let (sha256, canonical_transcript) = preflight_plan_identity_and_transcript_v1(
        target,
        identity_inventory_sha256,
        &types,
        &source_files,
        &functions,
        &function_abis,
        &terminals,
        &bodies,
        &roots,
        &edges,
        &direct_calls,
        &terminal_expansions,
        &normalized_intrinsics,
        &context_entries,
        counts,
        tcx,
    )?;
    Ok(ProductionSemanticPreflightPlanV1 {
        types,
        functions,
        function_abis,
        terminals,
        bodies,
        roots,
        direct_calls,
        terminal_expansions,
        normalized_intrinsics,
        debug_source_files: debug_source_files.into_values().collect(),
        debug_capture_gap,
        sha256,
        canonical_transcript,
        context_entries,
        construction_work: Some(ProductionSemanticConstructionWorkV1 {
            limits,
            validation_work: counts.validation_work,
        }),
    })
}

impl<'a, 'tcx> BodyPreflightV1<'a, 'tcx> {
    fn inspect_body(&mut self) -> Result<(), PendingRejectionV1> {
        let metadata = SliceMetadataPlanV1::derive(self.tcx, self.instance, self.body, |amount| {
            self.charge(SemanticMirResourceV1::ValidationWork, amount)
        })
        .map_err(|error| match error {
            SliceMetadataErrorV1::Resource(error) => error,
            SliceMetadataErrorV1::Allocation => {
                PendingRejectionV1::Fatal(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)
            }
            SliceMetadataErrorV1::Unsupported(location) => reject(
                "fake raw pointer outside the exact shared-slice metadata pair",
                RejectionSiteV1 {
                    function: self.function,
                    block: Some(location.block.index() as u32),
                    statement: Some(location.statement_index as u32),
                    local: None,
                    span: self.body.basic_blocks[location.block]
                        .statements
                        .get(location.statement_index)
                        .map_or(self.body.span, |row| row.source_info.span),
                },
            ),
        })?;
        for (local, declaration) in self.body.local_decls.iter_enumerated() {
            let site = RejectionSiteV1 {
                function: self.function,
                block: None,
                statement: None,
                local: Some(local.index() as u32),
                span: declaration.source_info.span,
            };
            self.inspect_type(declaration.ty, site)?;
        }
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            for (statement_index, statement) in data.statements.iter().enumerate() {
                let site = RejectionSiteV1 {
                    function: self.function,
                    block: Some(block.index() as u32),
                    statement: Some(statement_index as u32),
                    local: None,
                    span: statement.source_info.span,
                };
                self.charge(
                    SemanticMirResourceV1::ValidationWork,
                    metadata.lookup_work(),
                )?;
                if metadata.at(rustc_middle::mir::Location {
                    block,
                    statement_index,
                }) == Some(SliceMetadataRewriteV1::ElideTemporary)
                {
                    self.work()?;
                    let StatementKind::Assign(assignment) = &statement.kind else {
                        unreachable!()
                    };
                    let Rvalue::RawPtr(_, place) = assignment.1 else {
                        unreachable!()
                    };
                    self.inspect_place(assignment.0, site)?;
                    self.inspect_place(place, site)?;
                } else {
                    self.inspect_statement(&statement.kind, site)?;
                }
            }
            let terminator = data.terminator.as_ref().ok_or_else(|| {
                reject(
                    "basic block without a terminator",
                    RejectionSiteV1 {
                        function: self.function,
                        block: Some(block.index() as u32),
                        statement: None,
                        local: None,
                        span: self.body.span,
                    },
                )
            })?;
            let site = RejectionSiteV1 {
                function: self.function,
                block: Some(block.index() as u32),
                statement: None,
                local: None,
                span: terminator.source_info.span,
            };
            self.inspect_terminator(&terminator.kind, site)?;
        }
        Ok(())
    }

    fn inspect_statement(
        &mut self,
        statement: &StatementKind<'tcx>,
        site: RejectionSiteV1,
    ) -> Result<(), PendingRejectionV1> {
        self.work()?;
        match statement {
            StatementKind::Assign(assignment) => {
                let (destination, value) = &**assignment;
                self.inspect_place(*destination, site)?;
                self.inspect_rvalue(value, site)
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                self.require_local(*local, site)
            }
            StatementKind::SetDiscriminant { place, .. } => self.inspect_place(**place, site),
            StatementKind::Nop => Ok(()),
            StatementKind::FakeRead(..) => Err(reject("FakeRead statement", site)),
            StatementKind::Intrinsic(intrinsic) => match intrinsic.as_ref() {
                NonDivergingIntrinsic::Assume(condition) => self.inspect_operand(condition, site),
                NonDivergingIntrinsic::CopyNonOverlapping(_) => {
                    Err(reject("copy_nonoverlapping intrinsic statement", site))
                }
            },
            StatementKind::Retag(..) => Err(reject("Retag statement", site)),
            StatementKind::PlaceMention(..) => Err(reject("PlaceMention statement", site)),
            StatementKind::AscribeUserType(..) => Err(reject("AscribeUserType statement", site)),
            StatementKind::Coverage(..) => Err(reject("Coverage statement", site)),
            StatementKind::ConstEvalCounter => Err(reject("ConstEvalCounter statement", site)),
            StatementKind::BackwardIncompatibleDropHint { .. } => {
                Err(reject("BackwardIncompatibleDropHint statement", site))
            }
        }
    }

    fn inspect_rvalue(
        &mut self,
        value: &Rvalue<'tcx>,
        site: RejectionSiteV1,
    ) -> Result<(), PendingRejectionV1> {
        self.work()?;
        match value {
            Rvalue::Use(operand) => self.inspect_operand(operand, site),
            Rvalue::Ref(_, borrow, place) => {
                match borrow {
                    BorrowKind::Shared | BorrowKind::Fake(_) => {}
                    // Two-phase activation is borrow-checker state; both forms
                    // have the same mutable-reference runtime representation.
                    BorrowKind::Mut {
                        kind: MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow,
                    } => {}
                    BorrowKind::Mut {
                        kind: MutBorrowKind::ClosureCapture,
                    } => {
                        return Err(reject("closure-capture mutable borrow rvalue", site));
                    }
                }
                self.inspect_place(*place, site)
            }
            Rvalue::Discriminant(place) | Rvalue::CopyForDeref(place) => {
                self.inspect_place(*place, site)
            }
            Rvalue::Aggregate(kind, operands) => {
                match &**kind {
                    AggregateKind::Array(_)
                    | AggregateKind::Tuple
                    | AggregateKind::Adt(..)
                    | AggregateKind::Closure(..) => {}
                    AggregateKind::CoroutineClosure(..) => {
                        return Err(reject("coroutine-closure aggregate rvalue", site));
                    }
                    AggregateKind::Coroutine(..) => {
                        return Err(reject("coroutine aggregate rvalue", site));
                    }
                    AggregateKind::RawPtr(..) => {
                        return Err(reject("raw-pointer aggregate rvalue", site));
                    }
                }
                for operand in operands {
                    self.inspect_operand(operand, site)?;
                }
                Ok(())
            }
            Rvalue::Repeat(operand, count) => {
                let count = count
                    .try_to_target_usize(self.tcx)
                    .and_then(|count| usize::try_from(count).ok())
                    .ok_or_else(|| reject("Repeat count outside host bounds", site))?;
                if count == 0 {
                    return Ok(());
                }
                self.inspect_operand(operand, site)?;
                self.charge(SemanticMirResourceV1::Operands, count - 1)
            }
            Rvalue::RawPtr(RawPtrKind::Const | RawPtrKind::Mut, place) => {
                self.inspect_place(*place, site)
            }
            Rvalue::RawPtr(RawPtrKind::FakeForPtrMetadata, _) => {
                Err(reject("fake raw pointer for metadata rvalue", site))
            }
            Rvalue::Cast(
                rustc_middle::mir::CastKind::IntToInt
                | rustc_middle::mir::CastKind::IntToFloat
                | rustc_middle::mir::CastKind::FloatToInt
                | rustc_middle::mir::CastKind::FloatToFloat
                | rustc_middle::mir::CastKind::PtrToPtr
                | rustc_middle::mir::CastKind::FnPtrToPtr
                | rustc_middle::mir::CastKind::PointerExposeProvenance
                | rustc_middle::mir::CastKind::PointerWithExposedProvenance
                | rustc_middle::mir::CastKind::Transmute,
                operand,
                _,
            ) => self.inspect_operand(operand, site),
            Rvalue::Cast(rustc_middle::mir::CastKind::PointerCoercion(..), ..) => {
                Err(reject("unsupported PointerCoercion Cast rvalue", site))
            }
            Rvalue::Cast(rustc_middle::mir::CastKind::Subtype, ..) => {
                Err(reject("unsupported Subtype Cast rvalue", site))
            }
            Rvalue::BinaryOp(
                BinOp::Add
                | BinOp::Sub
                | BinOp::Mul
                | BinOp::Div
                | BinOp::Rem
                | BinOp::BitXor
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::Shl
                | BinOp::Shr
                | BinOp::ShlUnchecked
                | BinOp::ShrUnchecked
                | BinOp::Eq
                | BinOp::Lt
                | BinOp::Le
                | BinOp::Ne
                | BinOp::Ge
                | BinOp::Gt
                | BinOp::AddWithOverflow
                | BinOp::SubWithOverflow
                | BinOp::MulWithOverflow
                | BinOp::AddUnchecked
                | BinOp::SubUnchecked
                | BinOp::MulUnchecked
                | BinOp::Offset,
                operands,
            ) => {
                self.inspect_operand(&operands.0, site)?;
                self.inspect_operand(&operands.1, site)
            }
            Rvalue::BinaryOp(operation, ..) => Err(reject(
                format!("unsupported BinaryOp rvalue {operation:?}"),
                site,
            )),
            Rvalue::UnaryOp(_, operand) => self.inspect_operand(operand, site),
            Rvalue::ThreadLocalRef(..) => Err(reject("ThreadLocalRef rvalue", site)),
            Rvalue::WrapUnsafeBinder(..) => Err(reject("WrapUnsafeBinder rvalue", site)),
        }
    }

    fn inspect_terminator(
        &mut self,
        terminator: &TerminatorKind<'tcx>,
        site: RejectionSiteV1,
    ) -> Result<(), PendingRejectionV1> {
        self.work()?;
        match terminator {
            TerminatorKind::Return | TerminatorKind::Unreachable | TerminatorKind::Goto { .. } => {
                Ok(())
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                self.inspect_operand(discr, site)?;
                self.charge(SemanticMirResourceV1::SwitchTargets, targets.iter().count())
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                unwind,
                ..
            } => {
                if crate::production_core_panic_v50::is_candidate(self.tcx, func) {
                    let block = site
                        .block
                        .ok_or_else(|| reject("core panic source block", site))?;
                    let checked = crate::production_core_panic_v50::observe(
                        self.tcx,
                        self.instance,
                        self.body,
                        rustc_middle::mir::BasicBlock::from_usize(block as usize),
                        &mut |amount| self.charge(SemanticMirResourceV1::ValidationWork, amount),
                    )
                    .map_err(|error| match error {
                        crate::production_core_panic_v50::CorePanicErrorV50::Work(error) => error,
                        crate::production_core_panic_v50::CorePanicErrorV50::Refused(reason) => {
                            reject(reason, site)
                        }
                    })?;
                    let checked =
                        checked.ok_or_else(|| reject("core panic source call differs", site))?;
                    self.charge(SemanticMirResourceV1::Operands, 2)?;
                    self.charge(SemanticMirResourceV1::ConstantBytes, checked.bytes().len())?;
                    // The checked literal has no read or move effect and is not
                    // installed as a source value. Its original local/statement
                    // declarations are still inspected by the ordinary walk.
                    return self.inspect_place(*destination, site);
                }
                if !matches!(unwind, UnwindAction::Continue | UnwindAction::Unreachable) {
                    return Err(reject("call with executable unwind edge", site));
                }
                // Pass one already authenticated this as a direct callable.
                // Its FnDef is retained by the callable owner, not as a
                // zero-sized value in the layout-reachable type catalog.
                self.charge(SemanticMirResourceV1::Operands, 1)?;
                for argument in args {
                    self.inspect_operand(&argument.node, site)?;
                }
                self.inspect_place(*destination, site)
            }
            TerminatorKind::TailCall { .. } => Err(reject("TailCall terminator", site)),
            TerminatorKind::Drop { place, unwind, .. } => {
                if !matches!(unwind, UnwindAction::Continue | UnwindAction::Unreachable) {
                    return Err(reject("drop with executable unwind edge", site));
                }
                self.inspect_place(*place, site)?;
                match classify_rustc_drop_v1(self.tcx, self.instance, self.body, *place) {
                    Ok(ProductionRustcDropClassV1::Trivial) => Ok(()),
                    Ok(ProductionRustcDropClassV1::RequiresDropGlue) => {
                        Err(reject("Drop terminator requiring drop glue", site))
                    }
                    Err(_) => Err(reject(
                        "Drop place type that failed monomorphic normalization",
                        site,
                    )),
                }
            }
            TerminatorKind::Assert {
                cond, msg, unwind, ..
            } => {
                if !matches!(unwind, UnwindAction::Continue | UnwindAction::Unreachable) {
                    return Err(reject("assert with executable unwind edge", site));
                }
                self.inspect_operand(cond, site)?;
                match msg.as_ref() {
                    AssertKind::BoundsCheck { len, index }
                    | AssertKind::Overflow(_, len, index)
                    | AssertKind::MisalignedPointerDereference {
                        required: len,
                        found: index,
                    } => {
                        self.inspect_operand(len, site)?;
                        self.inspect_operand(index, site)
                    }
                    AssertKind::DivisionByZero(operand) | AssertKind::RemainderByZero(operand) => {
                        self.inspect_operand(operand, site)
                    }
                    AssertKind::NullPointerDereference
                    | AssertKind::ResumedAfterReturn(_)
                    | AssertKind::ResumedAfterPanic(_) => Ok(()),
                    AssertKind::OverflowNeg(_)
                    | AssertKind::InvalidEnumConstruction(_)
                    | AssertKind::ResumedAfterDrop(_) => {
                        Err(reject("unsupported Assert terminator", site))
                    }
                }
            }
            TerminatorKind::UnwindResume => Err(reject("UnwindResume terminator", site)),
            TerminatorKind::UnwindTerminate(..) => Err(reject("UnwindTerminate terminator", site)),
            TerminatorKind::Yield { .. } => Err(reject("Yield terminator", site)),
            TerminatorKind::CoroutineDrop => Err(reject("CoroutineDrop terminator", site)),
            TerminatorKind::FalseEdge { .. } => Err(reject("FalseEdge terminator", site)),
            TerminatorKind::FalseUnwind { .. } => Err(reject("FalseUnwind terminator", site)),
            TerminatorKind::InlineAsm { .. } => Err(reject("InlineAsm terminator", site)),
        }
    }

    fn inspect_operand(
        &mut self,
        operand: &Operand<'tcx>,
        site: RejectionSiteV1,
    ) -> Result<(), PendingRejectionV1> {
        self.charge(SemanticMirResourceV1::Operands, 1)?;
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.inspect_place(*place, site),
            Operand::Constant(constant) => {
                let normalized = self
                    .instance
                    .try_instantiate_mir_and_normalize_erasing_regions(
                        self.tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(constant.const_),
                    )
                    .map_err(|_| reject("constant that failed monomorphic normalization", site))?;
                self.inspect_type(normalized.ty(), site)
            }
            Operand::RuntimeChecks(..) => Err(reject("RuntimeChecks operand", site)),
        }
    }

    fn inspect_place(
        &mut self,
        place: Place<'tcx>,
        site: RejectionSiteV1,
    ) -> Result<(), PendingRejectionV1> {
        self.require_local(place.local, site)?;
        self.charge(SemanticMirResourceV1::Projections, place.projection.len())?;
        let local_ty = self.body.local_decls[place.local].ty;
        self.inspect_type(local_ty, site)?;
        let mut derived = PlaceTy::from_ty(local_ty);
        for projection in place.projection {
            match projection {
                ProjectionElem::Deref
                | ProjectionElem::Field(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::Downcast(..)
                | ProjectionElem::OpaqueCast(..) => {}
                ProjectionElem::Index(local) => self.require_local(local, site)?,
                ProjectionElem::UnwrapUnsafeBinder(..) => {
                    return Err(reject("UnwrapUnsafeBinder projection", site));
                }
            }
            derived = derived.projection_ty(self.tcx, projection);
            self.inspect_type(derived.ty, site)?;
        }
        Ok(())
    }

    fn inspect_type(
        &mut self,
        raw: Ty<'tcx>,
        site: RejectionSiteV1,
    ) -> Result<(), PendingRejectionV1> {
        let mut pending = vec![raw];
        while let Some(raw) = pending.pop() {
            let ty = normalize_type_v1(self.tcx, self.instance, raw)
                .map_err(|_| reject("type that failed monomorphic normalization", site))?;
            if ty.has_param() || ty.has_escaping_bound_vars() {
                return Err(reject("non-monomorphic type", site));
            }
            let identity = rustc_type_identity_v1(self.tcx, ty);
            match self.types.entry(identity) {
                std::collections::btree_map::Entry::Occupied(existing) => {
                    if *existing.get() != ty {
                        return Err(PendingRejectionV1::Fatal(
                            ProductionSemanticPreflightErrorV1::TypeIdentityCollision,
                        ));
                    }
                    continue;
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(ty);
                }
            }
            self.charge(SemanticMirResourceV1::Types, 1)?;
            self.work()?;
            match ty.kind() {
                TyKind::Bool
                | TyKind::Char
                | TyKind::Int(_)
                | TyKind::Uint(_)
                | TyKind::Float(_)
                | TyKind::Never => {}
                TyKind::Tuple(fields) => {
                    for field in fields.iter() {
                        self.queue_type(&mut pending, field)?;
                    }
                }
                TyKind::Adt(definition, arguments) => {
                    // The semantic catalog is the exact layout-reachable type
                    // graph. Nominal-only arguments remain authenticated by
                    // the rustc type identity; substituted field types add
                    // every argument that has physical semantics.
                    let variants = definition.variants();
                    self.require_type_cardinality(variants.len())?;
                    if definition.is_enum() {
                        self.queue_type(
                            &mut pending,
                            definition.repr().discr_type().to_ty(self.tcx),
                        )?;
                    }
                    for (_, variant) in variants.iter_enumerated() {
                        self.require_type_cardinality(variant.fields.len())?;
                        self.work()?;
                        for field in &variant.fields {
                            self.work()?;
                            self.queue_type(&mut pending, field.ty(self.tcx, arguments))?;
                        }
                    }
                }
                TyKind::FnDef(..) => {}
                TyKind::Ref(_, pointee, _) | TyKind::RawPtr(pointee, _) => {
                    self.queue_type(&mut pending, *pointee)?;
                }
                TyKind::Array(element, length) => {
                    if length.has_param() || length.has_escaping_bound_vars() {
                        return Err(reject("non-monomorphic array length", site));
                    }
                    self.queue_type(&mut pending, *element)?;
                }
                TyKind::Slice(element) => self.queue_type(&mut pending, *element)?,
                TyKind::Str => {}
                TyKind::Pat(..) => return Err(reject("pattern type", site)),
                TyKind::Foreign(..) => return Err(reject("foreign type", site)),
                TyKind::FnPtr(signature, _) => {
                    let signature_types = signature.skip_binder().inputs_and_output;
                    self.require_type_cardinality(signature_types.len())?;
                    for signature_type in signature_types {
                        self.queue_type(&mut pending, signature_type)?;
                    }
                }
                TyKind::UnsafeBinder(..) => return Err(reject("unsafe-binder type", site)),
                TyKind::Dynamic(..) => return Err(reject("dynamic trait-object type", site)),
                TyKind::Closure(_, arguments) => {
                    let captures = arguments.as_closure().upvar_tys();
                    self.require_type_cardinality(captures.len())?;
                    for capture in captures {
                        self.queue_type(&mut pending, capture)?;
                    }
                }
                TyKind::CoroutineClosure(..) => {
                    return Err(reject("coroutine-closure type", site));
                }
                TyKind::Coroutine(..) => return Err(reject("coroutine type", site)),
                TyKind::CoroutineWitness(..) => {
                    return Err(reject("coroutine-witness type", site));
                }
                TyKind::Alias(..) => return Err(reject("unresolved alias type", site)),
                TyKind::Param(..) => return Err(reject("generic parameter type", site)),
                TyKind::Bound(..) => return Err(reject("bound type", site)),
                TyKind::Placeholder(..) => return Err(reject("placeholder type", site)),
                TyKind::Infer(..) => return Err(reject("inference type", site)),
                TyKind::Error(..) => return Err(reject("error type", site)),
            }
        }
        Ok(())
    }

    fn queue_type(
        &self,
        pending: &mut Vec<Ty<'tcx>>,
        ty: Ty<'tcx>,
    ) -> Result<(), PendingRejectionV1> {
        let maximum = self.limits.limit(SemanticMirResourceV1::Types);
        let pending_count = u64::try_from(pending.len()).unwrap_or(u64::MAX);
        if pending_count >= maximum {
            return Err(PendingRejectionV1::Fatal(
                ProductionSemanticPreflightErrorV1::LimitExceeded {
                    resource: SemanticMirResourceV1::Types,
                    actual: pending_count.saturating_add(1),
                    maximum,
                },
            ));
        }
        pending.push(ty);
        Ok(())
    }

    fn require_type_cardinality(&self, actual: usize) -> Result<(), PendingRejectionV1> {
        let actual = u64::try_from(actual).unwrap_or(u64::MAX);
        let maximum = self.limits.limit(SemanticMirResourceV1::Types);
        if actual > maximum {
            return Err(PendingRejectionV1::Fatal(
                ProductionSemanticPreflightErrorV1::LimitExceeded {
                    resource: SemanticMirResourceV1::Types,
                    actual,
                    maximum,
                },
            ));
        }
        Ok(())
    }

    fn require_local(&self, local: Local, site: RejectionSiteV1) -> Result<(), PendingRejectionV1> {
        if local.index() < self.body.local_decls.len() {
            Ok(())
        } else {
            Err(reject("place references a local outside the body", site))
        }
    }

    fn charge(
        &mut self,
        resource: SemanticMirResourceV1,
        amount: usize,
    ) -> Result<(), PendingRejectionV1> {
        self.counts
            .charge(resource, amount, self.limits)
            .map_err(PendingRejectionV1::Fatal)
    }

    fn work(&mut self) -> Result<(), PendingRejectionV1> {
        self.charge(SemanticMirResourceV1::ValidationWork, 1)
    }
}

fn capture_body_sources_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    function: SemanticFunctionIdV1,
    body: &Body<'tcx>,
    cache: &mut HashMap<Span, RetainedSemanticSourceProducerV1>,
    debug_source_files: &mut BTreeMap<[u8; 32], fe2o3_kernel_ir::DebugSourceMapFileV1>,
    counts: &mut RawMirPreflightCountsV1,
    debug_counts: &mut RawMirPreflightCountsV1,
    debug_budget: &mut OptionalDebugCaptureBudgetV2,
    limits: SemanticMirLimitsV1,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<RetainedRawBodySourceProducerV1, PendingRejectionV1> {
    let function_site = RejectionSiteV1 {
        function,
        block: None,
        statement: None,
        local: None,
        span: body.span,
    };
    let source = capture_source_v1(
        tcx,
        body.span,
        function_site,
        cache,
        debug_source_files,
        counts,
        limits,
    )?;

    let mut locals = Vec::with_capacity(body.local_decls.len());
    for (local, declaration) in body.local_decls.iter_enumerated() {
        let site = RejectionSiteV1 {
            function,
            block: None,
            statement: None,
            local: Some(local.index() as u32),
            span: declaration.source_info.span,
        };
        locals.push(capture_source_v1(
            tcx,
            source_span_or_body_v1(declaration.source_info.span, body.span),
            site,
            cache,
            debug_source_files,
            counts,
            limits,
        )?);
    }

    let mut blocks = Vec::with_capacity(body.basic_blocks.len());
    for (block, data) in body.basic_blocks.iter_enumerated() {
        let Some(terminator) = &data.terminator else {
            return Err(reject(
                "basic block without a terminator",
                RejectionSiteV1 {
                    function,
                    block: Some(block.index() as u32),
                    statement: None,
                    local: None,
                    span: body.span,
                },
            ));
        };
        let mut statements = Vec::with_capacity(data.statements.len());
        for (statement_index, statement) in data.statements.iter().enumerate() {
            let site = RejectionSiteV1 {
                function,
                block: Some(block.index() as u32),
                statement: Some(statement_index as u32),
                local: None,
                span: statement.source_info.span,
            };
            statements.push(capture_source_v1(
                tcx,
                source_span_or_body_v1(statement.source_info.span, body.span),
                site,
                cache,
                debug_source_files,
                counts,
                limits,
            )?);
        }
        let terminator_site = RejectionSiteV1 {
            function,
            block: Some(block.index() as u32),
            statement: None,
            local: None,
            span: terminator.source_info.span,
        };
        let terminator = capture_source_v1(
            tcx,
            source_span_or_body_v1(terminator.source_info.span, body.span),
            terminator_site,
            cache,
            debug_source_files,
            counts,
            limits,
        )?;
        // A basic block has no independent rustc span. Its canonical source is
        // exactly the first statement's producer, or the terminator's producer
        // for an empty block.
        let source = statements.first().copied().unwrap_or(terminator);
        blocks.push(RetainedRawBlockSourceProducerV1 {
            source,
            statements: statements.into_boxed_slice(),
            terminator,
        });
    }
    let (debug_scopes, debug_variables, debug_capture_gap) = match debug_source_capture {
        DebugSourceCaptureRequestV2::Disabled => (
            Box::<[RetainedRawDebugSourceScopeV2]>::default(),
            Box::<[RetainedRawDebugSourceVariableV2]>::default(),
            None,
        ),
        DebugSourceCaptureRequestV2::SourceVariables => {
            let mut debug_cache = HashMap::new();
            let mut captured_files = BTreeMap::new();
            match observe_optional_debug_capture_v2(|| {
                capture_debug_sources_v2(
                    tcx,
                    body,
                    function_site,
                    &mut debug_cache,
                    &mut captured_files,
                    debug_counts,
                    debug_budget,
                    limits,
                )
            }) {
                Ok(Ok((scopes, variables))) => {
                    let consistent = captured_files.iter().all(|(identity, file)| {
                        debug_source_files
                            .get(identity)
                            .is_none_or(|existing| existing == file)
                    });
                    if consistent {
                        debug_source_files.extend(captured_files);
                        (scopes, variables, None)
                    } else {
                        (
                            Box::<[RetainedRawDebugSourceScopeV2]>::default(),
                            Box::<[RetainedRawDebugSourceVariableV2]>::default(),
                            Some(fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::SourceObservationUnrepresentable),
                        )
                    }
                }
                Ok(Err(gap)) => (
                    Box::<[RetainedRawDebugSourceScopeV2]>::default(),
                    Box::<[RetainedRawDebugSourceVariableV2]>::default(),
                    Some(gap),
                ),
                Err(error) => return Err(error),
            }
        }
    };
    Ok(RetainedRawBodySourceProducerV1 {
        source,
        locals: locals.into_boxed_slice(),
        blocks: blocks.into_boxed_slice(),
        debug_scopes,
        debug_variables,
        debug_capture_gap,
    })
}

fn classify_optional_debug_rejection_v2(
    error: &PendingRejectionV1,
) -> Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1> {
    match error {
        PendingRejectionV1::Fatal(
            ProductionSemanticPreflightErrorV1::LimitExceeded { .. }
            | ProductionSemanticPreflightErrorV1::AccountingDomain { .. }
            | ProductionSemanticPreflightErrorV1::IdentityTableMismatch,
        ) => Some(fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::ResourceLimit),
        PendingRejectionV1::Unsupported { .. } => Some(
            fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1::SourceObservationUnrepresentable,
        ),
        PendingRejectionV1::Fatal(_) => None,
    }
}

fn observe_optional_debug_capture_v2<T>(
    capture: impl FnOnce() -> Result<T, PendingRejectionV1>,
) -> Result<Result<T, fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>, PendingRejectionV1> {
    match capture() {
        Ok(value) => Ok(Ok(value)),
        Err(error) => match classify_optional_debug_rejection_v2(&error) {
            Some(gap) => Ok(Err(gap)),
            None => Err(error),
        },
    }
}

fn exact_debug_variable_name_or_rejection_v2(
    name: &str,
    site: RejectionSiteV1,
) -> Result<String, PendingRejectionV1> {
    exact_debug_variable_name_v2(name).ok_or_else(|| {
        reject(
            "source debug variable name is not exactly representable",
            site,
        )
    })
}

#[cfg(test)]
fn with_debug_source_capture_v2<T, E>(
    request: DebugSourceCaptureRequestV2,
    capture: impl FnOnce() -> Result<T, E>,
) -> Result<Option<T>, E> {
    match request {
        DebugSourceCaptureRequestV2::Disabled => Ok(None),
        DebugSourceCaptureRequestV2::SourceVariables => capture().map(Some),
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_debug_sources_v2<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    function_site: RejectionSiteV1,
    cache: &mut HashMap<Span, RetainedSemanticSourceProducerV1>,
    debug_source_files: &mut BTreeMap<[u8; 32], fe2o3_kernel_ir::DebugSourceMapFileV1>,
    counts: &mut RawMirPreflightCountsV1,
    budget: &mut OptionalDebugCaptureBudgetV2,
    limits: SemanticMirLimitsV1,
) -> Result<
    (
        Box<[RetainedRawDebugSourceScopeV2]>,
        Box<[RetainedRawDebugSourceVariableV2]>,
    ),
    PendingRejectionV1,
> {
    budget
        .charge_variables(body.var_debug_info.len())
        .map_err(PendingRejectionV1::Fatal)?;
    counts
        .charge(
            SemanticMirResourceV1::ValidationWork,
            body.var_debug_info.len(),
            limits,
        )
        .map_err(PendingRejectionV1::Fatal)?;
    let invalidated_entry_parameters = entry_parameter_invalidations_v2(body, counts, limits)?;

    let mut referenced_scopes = BTreeSet::new();
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(body.var_debug_info.len())
        .map_err(|_| {
            PendingRejectionV1::Fatal(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)
        })?;
    for (ordinal, variable) in body.var_debug_info.iter().enumerate() {
        let ordinal = u32::try_from(ordinal)
            .map_err(|_| reject("source debug variable ordinal exceeds u32", function_site))?;
        let raw_scope = u32::try_from(variable.source_info.scope.index())
            .map_err(|_| reject("source debug scope exceeds u32", function_site))?;
        let mut current = Some(variable.source_info.scope);
        let mut ancestry = 0_usize;
        while let Some(scope) = current {
            ancestry = ancestry
                .checked_add(1)
                .ok_or_else(|| reject("source debug scope ancestry overflowed", function_site))?;
            if ancestry > fe2o3_kernel_ir::MAX_DEBUG_SOURCE_SCOPES_V2 {
                return Err(reject(
                    "source debug scope ancestry exceeds the V2 bound",
                    function_site,
                ));
            }
            let raw = u32::try_from(scope.index())
                .map_err(|_| reject("source debug scope exceeds u32", function_site))?;
            if !referenced_scopes.insert(raw) {
                break;
            }
            let data = body
                .source_scopes
                .get(scope)
                .ok_or_else(|| reject("source debug scope is missing", function_site))?;
            current = data.parent_scope;
        }
        let name = variable.name.as_str();
        let exact_name = exact_debug_variable_name_or_rejection_v2(name, function_site)?;
        let mut name_digest = SemanticIdentityDigestV1::new(b"fe2o3/debug/source-variable/name/v2");
        name_digest.field(name.as_bytes());
        let (class, entry_value_preserved) = match &variable.value {
            VarDebugInfoContents::Place(place) if place.projection.is_empty() => {
                let raw = u32::try_from(place.local.index()).map_err(|_| {
                    reject("source debug local identity exceeds u32", function_site)
                })?;
                let is_parameter = place.local.index() > 0 && place.local.index() <= body.arg_count;
                (
                    RetainedRawDebugSourceVariableClassV2::Local(raw),
                    is_parameter && !invalidated_entry_parameters[place.local.index()],
                )
            }
            VarDebugInfoContents::Place(_) | VarDebugInfoContents::Const(_) => {
                (RetainedRawDebugSourceVariableClassV2::Unrepresented, false)
            }
        };
        variables.push(RetainedRawDebugSourceVariableV2 {
            ordinal,
            exact_name: Some(exact_name),
            name_sha256: name_digest.finish(),
            raw_scope,
            class,
            entry_value_preserved,
        });
    }
    if referenced_scopes.len() > fe2o3_kernel_ir::MAX_DEBUG_SOURCE_SCOPES_V2 {
        return Err(PendingRejectionV1::Fatal(
            ProductionSemanticPreflightErrorV1::LimitExceeded {
                resource: SemanticMirResourceV1::Locals,
                actual: u64::try_from(referenced_scopes.len()).unwrap_or(u64::MAX),
                maximum: fe2o3_kernel_ir::MAX_DEBUG_SOURCE_SCOPES_V2 as u64,
            },
        ));
    }
    budget
        .charge_scopes(referenced_scopes.len())
        .map_err(PendingRejectionV1::Fatal)?;
    counts
        .charge(
            SemanticMirResourceV1::ValidationWork,
            referenced_scopes.len(),
            limits,
        )
        .map_err(PendingRejectionV1::Fatal)?;

    let mut parent_by_scope = BTreeMap::new();
    let mut source_by_scope = BTreeMap::new();
    let mut children = BTreeMap::<u32, Vec<u32>>::new();
    for raw_scope in &referenced_scopes {
        let scope = rustc_middle::mir::SourceScope::from_usize(*raw_scope as usize);
        let data = body
            .source_scopes
            .get(scope)
            .ok_or_else(|| reject("source debug scope is missing", function_site))?;
        let parent = data
            .parent_scope
            .map(|parent| u32::try_from(parent.index()))
            .transpose()
            .map_err(|_| reject("source debug parent scope exceeds u32", function_site))?;
        if parent.is_some_and(|parent| !referenced_scopes.contains(&parent)) {
            return Err(reject(
                "source debug scope ancestor was not retained",
                function_site,
            ));
        }
        if let Some(parent) = parent {
            children.entry(parent).or_default().push(*raw_scope);
        }
        parent_by_scope.insert(*raw_scope, parent);
        source_by_scope.insert(
            *raw_scope,
            capture_source_v1(
                tcx,
                source_span_or_body_v1(data.span, body.span),
                function_site,
                cache,
                debug_source_files,
                counts,
                limits,
            )?,
        );
    }
    let mut depths = BTreeMap::new();
    let mut pending = VecDeque::new();
    for (scope, parent) in &parent_by_scope {
        if parent.is_none() {
            depths.insert(*scope, 0_u32);
            pending.push_back(*scope);
        }
    }
    while let Some(scope) = pending.pop_front() {
        let depth = depths[&scope];
        for child in children.get(&scope).into_iter().flatten() {
            let child_depth = depth
                .checked_add(1)
                .ok_or_else(|| reject("source debug scope depth overflowed", function_site))?;
            if depths.insert(*child, child_depth).is_some() {
                return Err(reject(
                    "source debug scope graph is not a tree",
                    function_site,
                ));
            }
            pending.push_back(*child);
        }
    }
    if depths.len() != referenced_scopes.len() {
        return Err(reject(
            "source debug scope graph is cyclic or rootless",
            function_site,
        ));
    }
    let mut scopes = Vec::new();
    scopes
        .try_reserve_exact(referenced_scopes.len())
        .map_err(|_| {
            PendingRejectionV1::Fatal(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)
        })?;
    for raw_scope in referenced_scopes {
        scopes.push(RetainedRawDebugSourceScopeV2 {
            raw_scope,
            parent_raw_scope: parent_by_scope[&raw_scope],
            depth: depths[&raw_scope],
            source: source_by_scope[&raw_scope],
        });
    }
    Ok((scopes.into_boxed_slice(), variables.into_boxed_slice()))
}

fn entry_parameter_invalidations_v2(
    body: &Body<'_>,
    counts: &mut RawMirPreflightCountsV1,
    limits: SemanticMirLimitsV1,
) -> Result<Vec<bool>, PendingRejectionV1> {
    let work = body.basic_blocks.iter().try_fold(0_usize, |work, block| {
        work.checked_add(block.statements.len())
            .and_then(|work| work.checked_add(1))
    });
    let Some(work) = work else {
        return Err(PendingRejectionV1::Fatal(
            ProductionSemanticPreflightErrorV1::IdentityTableMismatch,
        ));
    };
    counts
        .charge(SemanticMirResourceV1::ValidationWork, work, limits)
        .map_err(PendingRejectionV1::Fatal)?;
    let mut invalidated = Vec::new();
    invalidated
        .try_reserve_exact(body.local_decls.len())
        .map_err(|_| {
            PendingRejectionV1::Fatal(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)
        })?;
    invalidated.resize(body.local_decls.len(), false);
    for block in body.basic_blocks.iter() {
        for statement in &block.statements {
            invalidate_entry_parameters_in_statement_v2(&statement.kind, &mut invalidated);
        }
        let Some(terminator) = &block.terminator else {
            invalidated.fill(true);
            continue;
        };
        invalidate_entry_parameters_in_terminator_v2(&terminator.kind, &mut invalidated);
    }
    Ok(invalidated)
}

fn invalidate_entry_parameters_in_statement_v2(
    statement: &StatementKind<'_>,
    invalidated: &mut [bool],
) {
    match statement {
        StatementKind::Assign(assignment) => {
            let (destination, value) = &**assignment;
            invalidate_place_v2(*destination, invalidated);
            invalidate_moves_in_rvalue_v2(value, invalidated);
        }
        StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
            invalidate_local_v2(*local, invalidated);
        }
        StatementKind::SetDiscriminant { place, .. } => {
            invalidate_place_v2(**place, invalidated);
        }
        StatementKind::Intrinsic(intrinsic) => match intrinsic.as_ref() {
            NonDivergingIntrinsic::Assume(condition) => {
                invalidate_move_operand_v2(condition, invalidated);
            }
            NonDivergingIntrinsic::CopyNonOverlapping(_) => invalidated.fill(true),
        },
        StatementKind::Nop => {}
        StatementKind::FakeRead(..)
        | StatementKind::Retag(..)
        | StatementKind::PlaceMention(..)
        | StatementKind::AscribeUserType(..)
        | StatementKind::Coverage(..)
        | StatementKind::ConstEvalCounter
        | StatementKind::BackwardIncompatibleDropHint { .. } => invalidated.fill(true),
    }
}

fn invalidate_entry_parameters_in_terminator_v2(
    terminator: &TerminatorKind<'_>,
    invalidated: &mut [bool],
) {
    match terminator {
        TerminatorKind::Return | TerminatorKind::Unreachable | TerminatorKind::Goto { .. } => {}
        TerminatorKind::SwitchInt { discr, .. } => {
            invalidate_move_operand_v2(discr, invalidated);
        }
        TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } => {
            invalidate_move_operand_v2(func, invalidated);
            for argument in args {
                invalidate_move_operand_v2(&argument.node, invalidated);
            }
            invalidate_place_v2(*destination, invalidated);
        }
        TerminatorKind::Drop { place, .. } => invalidate_place_v2(*place, invalidated),
        TerminatorKind::Assert { cond, msg, .. } => {
            invalidate_move_operand_v2(cond, invalidated);
            match msg.as_ref() {
                AssertKind::BoundsCheck { len, index }
                | AssertKind::Overflow(_, len, index)
                | AssertKind::MisalignedPointerDereference {
                    required: len,
                    found: index,
                } => {
                    invalidate_move_operand_v2(len, invalidated);
                    invalidate_move_operand_v2(index, invalidated);
                }
                AssertKind::DivisionByZero(operand)
                | AssertKind::RemainderByZero(operand)
                | AssertKind::OverflowNeg(operand) => {
                    invalidate_move_operand_v2(operand, invalidated);
                }
                AssertKind::NullPointerDereference
                | AssertKind::ResumedAfterReturn(_)
                | AssertKind::ResumedAfterPanic(_)
                | AssertKind::InvalidEnumConstruction(_)
                | AssertKind::ResumedAfterDrop(_) => {}
            }
        }
        TerminatorKind::TailCall { .. }
        | TerminatorKind::UnwindResume
        | TerminatorKind::UnwindTerminate(..)
        | TerminatorKind::Yield { .. }
        | TerminatorKind::CoroutineDrop
        | TerminatorKind::FalseEdge { .. }
        | TerminatorKind::FalseUnwind { .. }
        | TerminatorKind::InlineAsm { .. } => invalidated.fill(true),
    }
}

fn invalidate_moves_in_rvalue_v2(value: &Rvalue<'_>, invalidated: &mut [bool]) {
    match value {
        Rvalue::Use(operand)
        | Rvalue::Repeat(operand, _)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::UnaryOp(_, operand) => invalidate_move_operand_v2(operand, invalidated),
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                invalidate_move_operand_v2(operand, invalidated);
            }
        }
        Rvalue::BinaryOp(_, operands) => {
            invalidate_move_operand_v2(&operands.0, invalidated);
            invalidate_move_operand_v2(&operands.1, invalidated);
        }
        Rvalue::Ref(
            _,
            BorrowKind::Mut {
                kind:
                    MutBorrowKind::Default
                    | MutBorrowKind::TwoPhaseBorrow
                    | MutBorrowKind::ClosureCapture,
            },
            place,
        )
        | Rvalue::RawPtr(RawPtrKind::Mut, place) => invalidate_place_v2(*place, invalidated),
        Rvalue::Ref(_, BorrowKind::Shared | BorrowKind::Fake(_), _)
        | Rvalue::RawPtr(RawPtrKind::Const | RawPtrKind::FakeForPtrMetadata, _)
        | Rvalue::Discriminant(..)
        | Rvalue::CopyForDeref(..)
        | Rvalue::ThreadLocalRef(..) => {}
        Rvalue::WrapUnsafeBinder(..) => invalidated.fill(true),
    }
}

fn invalidate_move_operand_v2(operand: &Operand<'_>, invalidated: &mut [bool]) {
    if let Operand::Move(place) = operand {
        invalidate_place_v2(*place, invalidated);
    } else if matches!(operand, Operand::RuntimeChecks(..)) {
        invalidated.fill(true);
    }
}

fn invalidate_place_v2(place: Place<'_>, invalidated: &mut [bool]) {
    invalidate_local_v2(place.local, invalidated);
}

fn invalidate_local_v2(local: Local, invalidated: &mut [bool]) {
    if let Some(invalidated) = invalidated.get_mut(local.index()) {
        *invalidated = true;
    } else {
        invalidated.fill(true);
    }
}

fn exact_debug_variable_name_v2(name: &str) -> Option<String> {
    (!name.is_empty()
        && name.len() <= fe2o3_kernel_ir::MAX_DEBUG_SOURCE_VARIABLE_NAME_BYTES_V2
        && !name.chars().any(char::is_control))
    .then(|| name.to_owned())
}

fn source_span_or_body_v1(span: Span, body: Span) -> Span {
    if span.is_dummy() { body } else { span }
}

fn capture_source_v1(
    tcx: TyCtxt<'_>,
    span: Span,
    site: RejectionSiteV1,
    cache: &mut HashMap<Span, RetainedSemanticSourceProducerV1>,
    debug_source_files: &mut BTreeMap<[u8; 32], fe2o3_kernel_ir::DebugSourceMapFileV1>,
    counts: &mut RawMirPreflightCountsV1,
    limits: SemanticMirLimitsV1,
) -> Result<RetainedSemanticSourceProducerV1, PendingRejectionV1> {
    if let Some(source) = cache.get(&span).copied() {
        return Ok(source);
    }
    let (captured, file) =
        canonical_source_provenance_and_debug_files_v1(tcx, span, MAX_MACRO_EXPANSION_DEPTH_V1)
            .map_err(|error| reject(format!("invalid source provenance: {error}"), site))?;
    if let Some(file) = file {
        match debug_source_files.entry(file.identity()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(file);
            }
            std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &file => {}
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(reject(
                    "stable source-file identity resolved to inconsistent live-rustc metadata",
                    site,
                ));
            }
        }
    }
    counts
        .charge(
            SemanticMirResourceV1::ValidationWork,
            captured.expansion_depth().saturating_add(2),
            limits,
        )
        .map_err(PendingRejectionV1::Fatal)?;
    let source = retained_source_v1(captured);
    cache.insert(span, source);
    Ok(source)
}

const fn retained_source_v1(
    captured: CanonicalSourceProvenanceV1,
) -> RetainedSemanticSourceProducerV1 {
    RetainedSemanticSourceProducerV1 {
        provenance: captured.provenance(),
        expansion_chain_sha256: captured.expansion_chain_sha256(),
    }
}

fn build_function_abi_producers_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    functions: &[RetainedSemanticFunctionProducerV1<'tcx>],
) -> Result<Box<[RetainedSemanticFunctionAbiProducerV1<'tcx>]>, ProductionSemanticPreflightErrorV1>
{
    let mut producers = Vec::new();
    producers.try_reserve_exact(functions.len()).map_err(|_| {
        ProductionSemanticPreflightErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::Functions,
            actual: u64::try_from(functions.len()).unwrap_or(u64::MAX),
            maximum: SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::Functions),
        }
    })?;
    for (index, function) in functions.iter().enumerate() {
        let function_id = SemanticFunctionIdV1::from_index(
            u32::try_from(index)
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?,
        );
        let source_abi = source_signature_v1(tcx, function.instance)
            .map_err(|detail| ProductionSemanticPreflightErrorV1::FunctionAbi {
                function: function_id,
                detail: detail.to_owned(),
            })?
            .abi;
        let (extern_abi, use_instance_abi) = match function.role {
            CollectedFunctionRole::KernelEntry => (ExternAbi::GpuKernel, false),
            CollectedFunctionRole::DeviceFfiExport => (ExternAbi::C { unwind: false }, false),
            CollectedFunctionRole::InternalHelper => (source_abi, true),
        };
        producers.push(build_retained_fn_abi_producer_v1(
            tcx,
            target,
            function_id,
            function.instance,
            function.identities.function(),
            extern_abi,
            use_instance_abi,
        )?);
    }
    Ok(producers.into_boxed_slice())
}

fn build_terminal_producers_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    recipes: &[TerminalExpansionRecipeV1<'tcx>],
    counts: &mut RawMirPreflightCountsV1,
    limits: SemanticMirLimitsV1,
) -> Result<RetainedTerminalTableV1<'tcx>, ProductionSemanticPreflightErrorV1> {
    let mut unique = BTreeMap::new();
    for recipe in recipes {
        counts.charge(SemanticMirResourceV1::ValidationWork, 1, limits)?;
        let next_callable_count = counts
            .functions
            .checked_add(u64::try_from(unique.len()).unwrap_or(u64::MAX))
            .and_then(|count| count.checked_add(1))
            .unwrap_or(u64::MAX);
        match unique.entry(recipe.identities.function()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let maximum = limits.limit(SemanticMirResourceV1::Callables);
                if next_callable_count > maximum {
                    return Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
                        resource: SemanticMirResourceV1::Callables,
                        actual: next_callable_count,
                        maximum,
                    });
                }
                entry.insert((recipe.instance, recipe.identities, recipe.expansion));
            }
            std::collections::btree_map::Entry::Occupied(entry)
                if entry.get().0 == recipe.instance && entry.get().2 == recipe.expansion => {}
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            }
        }
    }

    let mut terminal_ids = BTreeMap::new();
    let mut terminals = Vec::new();
    terminals.try_reserve_exact(unique.len()).map_err(|_| {
        ProductionSemanticPreflightErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::Callables,
            actual: u64::try_from(unique.len()).unwrap_or(u64::MAX),
            maximum: limits.limit(SemanticMirResourceV1::Callables),
        }
    })?;
    for (index, (identity, (instance, identities, expansion))) in unique.into_iter().enumerate() {
        let terminal = u32::try_from(index)
            .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        terminal_ids.insert(identity, terminal);
        let source_abi = source_signature_v1(tcx, instance)
            .map_err(|detail| ProductionSemanticPreflightErrorV1::FunctionAbi {
                function: SemanticFunctionIdV1::from_index(terminal),
                detail: detail.to_owned(),
            })?
            .abi;
        let source_span = match instance.def {
            InstanceKind::Intrinsic(def_id) => tcx.def_span(def_id),
            _ => tcx.instance_mir(instance.def).span,
        };
        let captured =
            canonical_source_provenance_v1(tcx, source_span, MAX_MACRO_EXPANSION_DEPTH_V1)
                .map_err(|error| ProductionSemanticPreflightErrorV1::FunctionAbi {
                    function: SemanticFunctionIdV1::from_index(terminal),
                    detail: bounded_diagnostic_component_v1(&format!(
                        "invalid terminal source provenance: {error}"
                    )),
                })?;
        counts.charge(
            SemanticMirResourceV1::ValidationWork,
            captured.expansion_depth().saturating_add(2),
            limits,
        )?;
        terminals.push(RetainedSemanticTerminalProducerV1 {
            identities,
            instance,
            expansion,
            abi: build_retained_fn_abi_producer_v1(
                tcx,
                target,
                SemanticFunctionIdV1::from_index(terminal),
                instance,
                identity,
                source_abi,
                true,
            )?,
            source: retained_source_v1(captured),
        });
    }
    Ok((terminals.into_boxed_slice(), terminal_ids))
}

pub(crate) fn source_signature_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Result<ty::FnSig<'tcx>, &'static str> {
    let env = TypingEnv::fully_monomorphized();
    let signature = if tcx.def_kind(instance.def_id()) == rustc_hir::def::DefKind::Closure {
        if !matches!(instance.def, InstanceKind::Item(_)) {
            return Err("closure signature requires an ordinary closure body instance");
        }
        let closure_ty = instance.ty(tcx, env);
        let TyKind::Closure(_, arguments) = closure_ty.kind() else {
            return Err("closure body instance has no concrete environment type");
        };
        let closure = arguments.as_closure();
        let kind = closure
            .kind_ty()
            .to_opt_closure_kind()
            .ok_or("unresolved closure kind")?;
        let signature = tcx.instantiate_bound_regions_with_erased(closure.sig());
        if signature.abi != ExternAbi::RustCall
            || signature.c_variadic
            || !matches!(signature.inputs(), [tuple] if matches!(tuple.kind(), TyKind::Tuple(_)))
        {
            return Err("closure signature is not one nonvariadic RustCall argument tuple");
        }
        let receiver = tcx.closure_env_ty(closure_ty, kind, tcx.lifetimes.re_erased);
        tcx.mk_fn_sig(
            std::iter::once(receiver).chain(signature.inputs().iter().copied()),
            signature.output(),
            signature.c_variadic,
            signature.safety,
            signature.abi,
        )
    } else {
        tcx.instantiate_bound_regions_with_erased(
            tcx.fn_sig(instance.def_id())
                .instantiate(tcx, instance.args),
        )
    };
    tcx.try_normalize_erasing_regions(env, signature)
        .map_err(|_| "source ABI signature failed monomorphic normalization")
}

#[allow(clippy::too_many_arguments)]
fn build_retained_fn_abi_producer_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    function: SemanticFunctionIdV1,
    instance: Instance<'tcx>,
    function_identity: SemanticFunctionIdentityV1,
    extern_abi: ExternAbi,
    use_instance_abi: bool,
) -> Result<RetainedSemanticFunctionAbiProducerV1<'tcx>, ProductionSemanticPreflightErrorV1> {
    let typing_env = TypingEnv::fully_monomorphized();
    let source_signature = source_signature_v1(tcx, instance).map_err(|detail| {
        ProductionSemanticPreflightErrorV1::FunctionAbi {
            function,
            detail: detail.to_owned(),
        }
    })?;
    let fn_abi = if use_instance_abi {
        let query = typing_env.as_query_input((instance, ty::List::empty()));
        tcx.fn_abi_of_instance(query)
    } else {
        let promoted_signature = ty::Binder::dummy(tcx.mk_fn_sig(
            source_signature.inputs().iter().copied(),
            source_signature.output(),
            source_signature.c_variadic,
            source_signature.safety,
            extern_abi,
        ));
        let query = typing_env.as_query_input((promoted_signature, ty::List::empty()));
        tcx.fn_abi_of_fn_ptr(query)
    }
    .map_err(|error| ProductionSemanticPreflightErrorV1::FunctionAbi {
        function,
        detail: bounded_diagnostic_component_v1(&format!("{error:?}")),
    })?;
    Ok(RetainedSemanticFunctionAbiProducerV1 {
        function,
        identity: rustc_semantic_fn_abi_identity_v1(tcx, function_identity, fn_abi),
        layout_identity: rustc_semantic_fn_abi_layout_identity_v1(tcx, target, fn_abi),
        extern_abi,
        source_inputs: source_signature.inputs().to_vec().into_boxed_slice(),
        source_output: source_signature.output(),
        fn_abi,
        rustc_source_signature_sha256: rustc_fn_signature_sha256_v1(tcx, source_signature),
        rustc_fn_abi_sha256: rustc_fn_abi_sha256_v1(tcx, fn_abi),
    })
}

fn build_canonical_producer_tables_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    functions: &[RetainedSemanticFunctionProducerV1<'tcx>],
    source_producers: Vec<RetainedRawBodySourceProducerV1>,
    types: BTreeMap<SemanticTypeIdentityV1, Ty<'tcx>>,
    receiver_reborrows: Vec<Option<ReceiverReborrowV1<'tcx>>>,
    normalized_calls: Option<&[NormalizedRustcIntrinsicRecipeV1<'tcx>]>,
    (counts, debug_counts, limits): (
        &mut RawMirPreflightCountsV1,
        &mut RawMirPreflightCountsV1,
        SemanticMirLimitsV1,
    ),
) -> Result<CanonicalProducerTablesV1<'tcx>, ProductionSemanticPreflightErrorV1> {
    if source_producers.len() != functions.len() || receiver_reborrows.len() != functions.len() {
        return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
    }
    let mut type_ids = BTreeMap::new();
    let mut type_producers = Vec::with_capacity(types.len());
    let layout_cx = LayoutCx::new(tcx, TypingEnv::fully_monomorphized());
    for (index, (identity, ty)) in types.into_iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        let layout = layout_cx.layout_of(ty).map_err(|error| {
            ProductionSemanticPreflightErrorV1::TypeLayout {
                identity,
                detail: bounded_diagnostic_component_v1(&error.to_string()),
            }
        })?;
        let rustc_layout_sha256 = rustc_type_layout_sha256_v1(tcx, layout);
        let semantic_layout_identity = rustc_semantic_layout_identity_v1(tcx, target, layout);
        type_ids.insert(identity, SemanticTypeIdV1::from_index(index));
        type_producers.push(RetainedSemanticTypeProducerV1 {
            identity,
            ty,
            layout,
            rustc_layout_sha256,
            semantic_layout_identity,
        });
    }

    let mut bodies = Vec::with_capacity(functions.len());
    for (function_index, ((function, raw_sources), receiver_reborrow)) in functions
        .iter()
        .zip(source_producers)
        .zip(receiver_reborrows)
        .enumerate()
    {
        let function_id = SemanticFunctionIdV1::from_index(
            u32::try_from(function_index)
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?,
        );
        let function_identity = function.identities.function();
        let body = tcx.instance_mir(function.instance.def);
        if raw_sources.locals.len() != body.local_decls.len()
            || raw_sources.blocks.len() != body.basic_blocks.len()
        {
            return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
        }
        let mir_body_sha256 = rustc_mir_body_sha256_v1(tcx, function.instance);

        let mut locals = Vec::with_capacity(body.local_decls.len());
        for (raw_local, declaration) in body.local_decls.iter_enumerated() {
            let raw_local = u32::try_from(raw_local.index())
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            let normalized = normalize_type_v1(tcx, function.instance, declaration.ty)
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            let type_identity = rustc_type_identity_v1(tcx, normalized);
            let ty = type_ids
                .get(&type_identity)
                .copied()
                .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            locals.push(RetainedSemanticLocalProducerV1 {
                identity: rustc_local_identity_v1(function_identity, mir_body_sha256, raw_local),
                rustc_local: raw_local,
                ty,
                source: *raw_sources
                    .locals
                    .get(raw_local as usize)
                    .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?,
            });
        }
        locals.sort_unstable_by_key(|local| local.identity);
        if locals
            .windows(2)
            .any(|pair| pair[0].identity == pair[1].identity)
        {
            return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
        }
        let mut raw_to_semantic_locals = vec![None; locals.len()];
        for (semantic_index, local) in locals.iter().enumerate() {
            let semantic_index = u32::try_from(semantic_index)
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            let Some(slot) = raw_to_semantic_locals.get_mut(local.rustc_local as usize) else {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            };
            if slot
                .replace(SemanticLocalIdV1::from_index(semantic_index))
                .is_some()
            {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            }
        }
        let mut raw_to_semantic_locals = raw_to_semantic_locals
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;

        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for (raw_block, data) in body.basic_blocks.iter_enumerated() {
            let raw_block = u32::try_from(raw_block.index())
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            let raw_source = raw_sources
                .blocks
                .get(raw_block as usize)
                .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            if raw_source.statements.len() != data.statements.len() {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            }
            blocks.push(RetainedSemanticBlockProducerV1 {
                identity: rustc_block_identity_v1(function_identity, mir_body_sha256, raw_block),
                rustc_block: raw_block,
                source: raw_source.source,
                statements: raw_source.statements.clone(),
                terminator: raw_source.terminator,
            });
        }
        blocks.sort_unstable_by_key(|block| block.identity);
        if blocks
            .windows(2)
            .any(|pair| pair[0].identity == pair[1].identity)
        {
            return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
        }
        let mut raw_to_semantic_blocks = vec![None; blocks.len()];
        for (semantic_index, block) in blocks.iter().enumerate() {
            let semantic_index = u32::try_from(semantic_index)
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            let Some(slot) = raw_to_semantic_blocks.get_mut(block.rustc_block as usize) else {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            };
            if slot
                .replace(SemanticBlockIdV1::from_index(semantic_index))
                .is_some()
            {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            }
        }
        let raw_to_semantic_blocks = raw_to_semantic_blocks
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        let entry = raw_to_semantic_blocks
            .get(START_BLOCK.index())
            .copied()
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;

        let inserted = if let Some(reborrow) = receiver_reborrow {
            if !type_ids.contains_key(&rustc_type_identity_v1(
                tcx,
                reborrow.shared_receiver_type(),
            )) {
                return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
            }
            counts.charge(SemanticMirResourceV1::ValidationWork, 3, limits)?;
            counts.charge(SemanticMirResourceV1::ValidationWork, locals.len(), limits)?;
            Some(
                ReceiverLocalV1::derive(
                    function_identity,
                    rustc_block_identity_v1(function_identity, mir_body_sha256, reborrow.block()),
                    locals.iter().map(|local| local.identity),
                )
                .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?,
            )
        } else {
            None
        };
        let (debug_scopes, mut debug_variables, debug_capture_gap) =
            receiver_debug_v1::convert_receiver_debug_sources_v2(
                function_id,
                function_identity,
                &raw_sources,
                &raw_to_semantic_locals,
                &locals,
                inserted,
                (debug_counts, limits),
            );

        if let Some(inserted) = inserted {
            for local in &mut raw_to_semantic_locals {
                counts.charge(SemanticMirResourceV1::ValidationWork, 1, limits)?;
                *local = inserted
                    .remap(*local)
                    .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
            }
        }

        if let Some(normalized_calls) = normalized_calls {
            wrapping_locals_v1::remap_v1(
                function_id,
                function_identity,
                mir_body_sha256,
                &locals,
                inserted,
                normalized_calls,
                &mut raw_to_semantic_locals,
                &mut debug_variables,
                counts,
                limits,
            )?;
        }

        bodies.push(RetainedSemanticBodyProducerV1 {
            function: function_id,
            source: raw_sources.source,
            locals: locals.into_boxed_slice(),
            raw_to_semantic_locals: raw_to_semantic_locals.into_boxed_slice(),
            entry,
            blocks: blocks.into_boxed_slice(),
            raw_to_semantic_blocks: raw_to_semantic_blocks.into_boxed_slice(),
            debug_scopes,
            debug_variables,
            debug_capture_gap,
        });
    }
    let mut source_files = BTreeSet::new();
    for body in &bodies {
        remember_source_files_v1(&mut source_files, body.source.provenance);
        for local in &body.locals {
            remember_source_files_v1(&mut source_files, local.source.provenance);
        }
        for block in &body.blocks {
            remember_source_files_v1(&mut source_files, block.source.provenance);
            for statement in &block.statements {
                remember_source_files_v1(&mut source_files, statement.provenance);
            }
            remember_source_files_v1(&mut source_files, block.terminator.provenance);
        }
        for scope in &body.debug_scopes {
            remember_source_files_v1(&mut source_files, scope.source);
        }
    }
    Ok(CanonicalProducerTablesV1 {
        types: type_producers.into_boxed_slice(),
        source_files: source_files.into_iter().collect(),
        bodies: bodies.into_boxed_slice(),
    })
}

mod receiver_debug_v1;
mod wrapping_locals_v1;

fn convert_debug_sources_v2(
    function: SemanticFunctionIdV1,
    function_identity: SemanticFunctionIdentityV1,
    raw_scopes: &[RetainedRawDebugSourceScopeV2],
    raw_variables: &[RetainedRawDebugSourceVariableV2],
    raw_to_semantic_locals: &[SemanticLocalIdV1],
    locals: &[RetainedSemanticLocalProducerV1],
) -> Result<
    (
        Box<[RetainedDebugSourceScopeV2]>,
        Box<[RetainedDebugSourceVariableV2]>,
    ),
    ProductionSemanticPreflightErrorV1,
> {
    let mut scope_identity_by_raw = BTreeMap::new();
    for scope in raw_scopes {
        let mut digest =
            SemanticIdentityDigestV1::new(b"fe2o3/debug/source-scope/logical-identity/v2");
        digest.field(function_identity.as_bytes());
        digest.field(&scope.raw_scope.to_le_bytes());
        digest.field(&scope.source.expansion_chain_sha256);
        if scope_identity_by_raw
            .insert(scope.raw_scope, digest.finish())
            .is_some()
        {
            return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
        }
    }
    let mut scopes = Vec::new();
    scopes
        .try_reserve_exact(raw_scopes.len())
        .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
    for scope in raw_scopes {
        scopes.push(RetainedDebugSourceScopeV2 {
            identity: *scope_identity_by_raw
                .get(&scope.raw_scope)
                .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?,
            function,
            parent_identity: scope
                .parent_raw_scope
                .map(|parent| {
                    scope_identity_by_raw
                        .get(&parent)
                        .copied()
                        .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)
                })
                .transpose()?,
            depth: scope.depth,
            source: scope.source.provenance,
        });
    }
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(raw_variables.len())
        .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
    for variable in raw_variables {
        let class = match variable.class {
            RetainedRawDebugSourceVariableClassV2::Local(raw) => {
                RetainedDebugSourceVariableClassV2::Local(
                    raw_to_semantic_locals
                        .get(raw as usize)
                        .copied()
                        .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?,
                )
            }
            RetainedRawDebugSourceVariableClassV2::Unrepresented => {
                RetainedDebugSourceVariableClassV2::Unrepresented
            }
        };
        let scope_identity = *scope_identity_by_raw
            .get(&variable.raw_scope)
            .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
        let mut digest =
            SemanticIdentityDigestV1::new(b"fe2o3/debug/source-variable/logical-identity/v2");
        digest.field(function_identity.as_bytes());
        digest.field(&variable.ordinal.to_le_bytes());
        digest.field(&scope_identity);
        digest.field(&variable.name_sha256);
        match class {
            RetainedDebugSourceVariableClassV2::Local(local) => {
                digest.field(&[0]);
                digest.field(
                    locals
                        .get(local.index() as usize)
                        .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?
                        .identity
                        .as_bytes(),
                );
            }
            RetainedDebugSourceVariableClassV2::Unrepresented => digest.field(&[1]),
        }
        variables.push(RetainedDebugSourceVariableV2 {
            identity: digest.finish(),
            function,
            name: variable.exact_name.clone(),
            scope_identity,
            class,
            entry_value_preserved: variable.entry_value_preserved,
        });
    }
    Ok((scopes.into_boxed_slice(), variables.into_boxed_slice()))
}

fn remember_source_files_v1(
    files: &mut BTreeSet<SemanticSourceFileIdentityV1>,
    provenance: SemanticSourceProvenanceV1,
) {
    if let Some(origin) = provenance.expansion() {
        files.insert(origin.file());
    }
    if let Some(origin) = provenance.call_site() {
        files.insert(origin.file());
    }
}

fn normalize_type_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    raw: Ty<'tcx>,
) -> Result<Ty<'tcx>, &'static str> {
    instance
        .try_instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(raw),
        )
        .map_err(|_| "type failed monomorphic normalization")
}

fn resolve_direct_call_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: Instance<'tcx>,
    body: &Body<'tcx>,
    function: &Operand<'tcx>,
) -> Result<Instance<'tcx>, &'static str> {
    let raw = function.ty(body, tcx);
    let callable = caller
        .try_instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(raw),
        )
        .map_err(|_| "callable type that failed monomorphic normalization")?;
    let TyKind::FnDef(def_id, arguments) = callable.kind() else {
        return Err("indirect or non-function-definition call");
    };
    Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), *def_id, arguments)
        .map_err(|_| "direct call whose concrete rustc instance failed resolution")?
        .ok_or("direct call without a concrete rustc instance")
}

fn remember_rejection(
    first: &mut Option<PendingRejectionV1>,
    construct: impl Into<String>,
    site: RejectionSiteV1,
) {
    if first.is_none() {
        *first = Some(reject(construct, site));
    }
}

fn reject(construct: impl Into<String>, site: RejectionSiteV1) -> PendingRejectionV1 {
    PendingRejectionV1::Unsupported {
        construct: bounded_diagnostic_component_v1(&construct.into()),
        site,
    }
}

fn materialize_rejection_v1(
    tcx: TyCtxt<'_>,
    functions: &[RetainedSemanticFunctionProducerV1<'_>],
    roots: &[SemanticFunctionIdV1],
    edges: &BTreeSet<CallEdgeV1>,
    rejection: PendingRejectionV1,
) -> ProductionSemanticPreflightErrorV1 {
    let (construct, site) = match rejection {
        PendingRejectionV1::Unsupported { construct, site } => (construct, site),
        PendingRejectionV1::Fatal(error) => return error,
    };
    let function = &functions[site.function.index() as usize];
    let function_path =
        bounded_diagnostic_component_v1(&tcx.def_path_str(function.instance.def_id()));
    let call_chain = call_path_v1(roots, edges, site.function)
        .into_iter()
        .map(|id| {
            bounded_diagnostic_component_v1(
                &tcx.def_path_str(functions[id.index() as usize].instance.def_id()),
            )
        })
        .collect::<Box<[_]>>();
    ProductionSemanticPreflightErrorV1::UnsupportedRustcMir {
        construct,
        function: function_path,
        call_chain,
        location: location_diagnostic_v1(site),
        source: source_diagnostic_v1(tcx, site.span),
    }
}

fn call_path_v1(
    roots: &[SemanticFunctionIdV1],
    edges: &BTreeSet<CallEdgeV1>,
    target: SemanticFunctionIdV1,
) -> Vec<SemanticFunctionIdV1> {
    let predecessors = rooted_predecessors_v1(roots, edges);
    if !predecessors.contains_key(&target) {
        return vec![target];
    }
    let mut reverse = vec![target];
    let mut cursor = target;
    while let Some(Some(predecessor)) = predecessors.get(&cursor) {
        reverse.push(*predecessor);
        cursor = *predecessor;
    }
    reverse.reverse();
    reverse
}

fn first_unreachable_function_v1(
    roots: &[SemanticFunctionIdV1],
    edges: &BTreeSet<CallEdgeV1>,
    function_count: usize,
) -> Option<SemanticFunctionIdV1> {
    let predecessors = rooted_predecessors_v1(roots, edges);
    (0..function_count)
        .map(|index| SemanticFunctionIdV1::from_index(index as u32))
        .find(|function| !predecessors.contains_key(function))
}

fn rooted_predecessors_v1(
    roots: &[SemanticFunctionIdV1],
    edges: &BTreeSet<CallEdgeV1>,
) -> BTreeMap<SemanticFunctionIdV1, Option<SemanticFunctionIdV1>> {
    let mut adjacency = BTreeMap::<SemanticFunctionIdV1, Vec<SemanticFunctionIdV1>>::new();
    for edge in edges {
        adjacency.entry(edge.caller).or_default().push(edge.callee);
    }
    let mut predecessors = BTreeMap::new();
    let mut queue = VecDeque::new();
    for root in roots.iter().copied() {
        if predecessors.insert(root, None).is_none() {
            queue.push_back(root);
        }
    }
    while let Some(caller) = queue.pop_front() {
        for callee in adjacency.get(&caller).into_iter().flatten().copied() {
            if let std::collections::btree_map::Entry::Vacant(entry) = predecessors.entry(callee) {
                entry.insert(Some(caller));
                queue.push_back(callee);
            }
        }
    }
    predecessors
}

fn location_diagnostic_v1(site: RejectionSiteV1) -> String {
    match (site.local, site.block, site.statement) {
        (Some(local), _, _) => format!("local {local}"),
        (_, Some(block), Some(statement)) => format!("block {block}, statement {statement}"),
        (_, Some(block), None) => format!("block {block}, terminator"),
        _ => "function body".to_owned(),
    }
}

fn source_diagnostic_v1(tcx: TyCtxt<'_>, span: Span) -> String {
    let location = tcx.sess.source_map().lookup_char_pos(span.lo());
    bounded_diagnostic_component_v1(&format!(
        "{}:{}:{}",
        location
            .file
            .name
            .prefer_remapped_unconditionally()
            .to_string_lossy(),
        location.line,
        location.col.0 + 1,
    ))
}

fn bounded_diagnostic_component_v1(value: &str) -> String {
    value
        .chars()
        .take(MAX_DIAGNOSTIC_COMPONENT_CHARS_V1)
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreflightChildCommitmentV5 {
    field_count: u64,
    payload_framed_bytes: u64,
    sha256: [u8; 32],
}

struct PreflightChildCommitmentBuilderV5 {
    digest: SemanticIdentityDigestV1,
    scope: &'static str,
    field_count: u64,
    payload_framed_bytes: u64,
}

impl PreflightChildCommitmentBuilderV5 {
    fn section(domain: &'static [u8], section: u8, scope: &'static str) -> Self {
        let mut digest = SemanticIdentityDigestV1::new(domain);
        digest.field(&[section]);
        Self {
            digest,
            scope,
            field_count: 0,
            payload_framed_bytes: 0,
        }
    }

    fn body(ordinal: u64, function: SemanticFunctionIdV1) -> Self {
        let mut digest = SemanticIdentityDigestV1::new(PREFLIGHT_BODY_DOMAIN_V5);
        digest.field(&[PREFLIGHT_SECTION_BODIES_V5]);
        digest.field(&ordinal.to_le_bytes());
        digest.field(&function.index().to_le_bytes());
        Self {
            digest,
            scope: "body child",
            field_count: 0,
            payload_framed_bytes: 0,
        }
    }

    fn field(&mut self, field: &[u8]) -> Result<(), ProductionSemanticPreflightErrorV1> {
        let field_len = u64::try_from(field.len()).map_err(|_| {
            ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                scope: self.scope,
                actual: u64::MAX,
                maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_CANONICAL_BYTES_V1,
            }
        })?;
        let framed = 8_u64.checked_add(field_len).ok_or(
            ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                scope: self.scope,
                actual: u64::MAX,
                maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_CANONICAL_BYTES_V1,
            },
        )?;
        let next_bytes = self.payload_framed_bytes.checked_add(framed).ok_or(
            ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                scope: self.scope,
                actual: u64::MAX,
                maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_CANONICAL_BYTES_V1,
            },
        )?;
        if next_bytes > fe2o3_mir_model::semantic_mir_v1::HARD_MAX_CANONICAL_BYTES_V1 {
            return Err(
                ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                    scope: self.scope,
                    actual: next_bytes,
                    maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_CANONICAL_BYTES_V1,
                },
            );
        }
        self.field_count = self.field_count.checked_add(1).ok_or(
            ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                scope: self.scope,
                actual: u64::MAX,
                maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_VALIDATION_WORK_V1,
            },
        )?;
        self.payload_framed_bytes = next_bytes;
        self.digest.field(field);
        Ok(())
    }

    fn finish(self) -> PreflightChildCommitmentV5 {
        PreflightChildCommitmentV5 {
            field_count: self.field_count,
            payload_framed_bytes: self.payload_framed_bytes,
            sha256: self.digest.finish(),
        }
    }
}

struct BoundedPreflightTranscriptV5 {
    digest: SemanticIdentityDigestV1,
    framed_bytes: u64,
}

impl BoundedPreflightTranscriptV5 {
    fn new() -> Self {
        Self {
            digest: SemanticIdentityDigestV1::new_with_canonical_transcript(
                PREFLIGHT_PLAN_DOMAIN_V5,
            ),
            framed_bytes: 8 + PREFLIGHT_PLAN_DOMAIN_V5.len() as u64,
        }
    }

    fn field(&mut self, field: &[u8]) -> Result<(), ProductionSemanticPreflightErrorV1> {
        let maximum = fe2o3_compiler_lineage::MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3 as u64;
        let field_len = u64::try_from(field.len()).map_err(|_| {
            ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                scope: "V5 parent transcript",
                actual: u64::MAX,
                maximum,
            }
        })?;
        let actual = self
            .framed_bytes
            .checked_add(8)
            .and_then(|bytes| bytes.checked_add(field_len))
            .ok_or(
                ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                    scope: "V5 parent transcript",
                    actual: u64::MAX,
                    maximum,
                },
            )?;
        if actual > maximum {
            return Err(
                ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                    scope: "V5 parent transcript",
                    actual,
                    maximum,
                },
            );
        }
        self.framed_bytes = actual;
        self.digest.field(field);
        Ok(())
    }

    fn section(
        &mut self,
        section: u8,
        item_count: usize,
        commitment: PreflightChildCommitmentV5,
    ) -> Result<(), ProductionSemanticPreflightErrorV1> {
        self.field(&[section])?;
        self.field(&usize_to_u64_v5(item_count).to_le_bytes())?;
        self.field(&commitment.field_count.to_le_bytes())?;
        self.field(&commitment.payload_framed_bytes.to_le_bytes())?;
        self.field(&commitment.sha256)
    }

    fn body(
        &mut self,
        ordinal: u64,
        body: &RetainedSemanticBodyProducerV1,
        commitment: PreflightChildCommitmentV5,
    ) -> Result<(), ProductionSemanticPreflightErrorV1> {
        let statements = body
            .blocks
            .iter()
            .try_fold(0_usize, |count, block| {
                count.checked_add(block.statements.len())
            })
            .ok_or(
                ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                    scope: "body child counts",
                    actual: u64::MAX,
                    maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_VALIDATION_WORK_V1,
                },
            )?;
        let block_source_producers = body.blocks.len().checked_mul(2).ok_or(
            ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                scope: "body child counts",
                actual: u64::MAX,
                maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_VALIDATION_WORK_V1,
            },
        )?;
        let source_producers = 1_usize
            .checked_add(body.locals.len())
            .and_then(|count| count.checked_add(block_source_producers))
            .and_then(|count| count.checked_add(statements))
            .ok_or(
                ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                    scope: "body child counts",
                    actual: u64::MAX,
                    maximum: fe2o3_mir_model::semantic_mir_v1::HARD_MAX_VALIDATION_WORK_V1,
                },
            )?;
        self.field(&[PREFLIGHT_SECTION_BODIES_V5])?;
        self.field(&ordinal.to_le_bytes())?;
        self.field(&body.function.index().to_le_bytes())?;
        for count in [
            body.locals.len(),
            body.raw_to_semantic_locals.len(),
            body.blocks.len(),
            body.raw_to_semantic_blocks.len(),
            statements,
            source_producers,
        ] {
            self.field(&usize_to_u64_v5(count).to_le_bytes())?;
        }
        self.field(&commitment.field_count.to_le_bytes())?;
        self.field(&commitment.payload_framed_bytes.to_le_bytes())?;
        self.field(&commitment.sha256)
    }

    fn finish(self) -> Result<([u8; 32], Box<[u8]>), ProductionSemanticPreflightErrorV1> {
        let (sha256, transcript) = self.digest.finish_with_canonical_transcript();
        if transcript.len() as u64 != self.framed_bytes {
            return Err(
                ProductionSemanticPreflightErrorV1::CommitmentBoundExceeded {
                    scope: "V5 parent transcript accounting",
                    actual: transcript.len() as u64,
                    maximum: self.framed_bytes,
                },
            );
        }
        Ok((sha256, transcript))
    }
}

fn usize_to_u64_v5(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[allow(clippy::too_many_arguments)]
fn preflight_plan_identity_and_transcript_v1<'tcx>(
    target: SemanticTargetDataLayoutV1,
    identity_inventory_sha256: [u8; 32],
    types: &[RetainedSemanticTypeProducerV1<'tcx>],
    source_files: &[SemanticSourceFileIdentityV1],
    functions: &[RetainedSemanticFunctionProducerV1<'tcx>],
    function_abis: &[RetainedSemanticFunctionAbiProducerV1<'tcx>],
    terminals: &[RetainedSemanticTerminalProducerV1<'tcx>],
    bodies: &[RetainedSemanticBodyProducerV1],
    roots: &[SemanticFunctionIdV1],
    edges: &BTreeSet<CallEdgeV1>,
    direct_calls: &[DirectCallRecipeV1],
    terminal_expansions: &[TerminalExpansionRecipeV1<'tcx>],
    normalized_intrinsics: &[NormalizedRustcIntrinsicRecipeV1<'tcx>],
    context_entries: &BTreeMap<SemanticFunctionIdV1, BoundContextEntryV29<'tcx>>,
    counts: RawMirPreflightCountsV1,
    tcx: TyCtxt<'tcx>,
) -> Result<([u8; 32], Box<[u8]>), ProductionSemanticPreflightErrorV1> {
    let mut digest = BoundedPreflightTranscriptV5::new();
    // V5 is the parent domain. This separate field binds the exact flat-stream
    // lineage whose payload fields are committed by the V5 children below.
    digest.field(PREFLIGHT_PLAN_DOMAIN_V4)?;
    digest.field(target.identity().as_bytes())?;
    digest.field(&identity_inventory_sha256)?;
    for cardinality in [
        types.len(),
        source_files.len(),
        functions.len(),
        function_abis.len(),
        terminals.len(),
        bodies.len(),
        source_provenance_producer_count_v1(bodies),
        roots.len(),
        edges.len(),
        direct_calls.len(),
        terminal_expansions.len(),
        normalized_intrinsics.len(),
    ] {
        digest.field(&usize_to_u64_v5(cardinality).to_le_bytes())?;
    }
    for count in counts.digest_fields() {
        digest.field(&count.to_le_bytes())?;
    }
    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_TYPES_DOMAIN_V5,
        PREFLIGHT_SECTION_TYPES_V5,
        "types child",
    );
    for ty in types {
        section.field(ty.identity.as_bytes())?;
        section.field(rustc_type_identity_v1(tcx, ty.ty).as_bytes())?;
        section.field(&ty.rustc_layout_sha256)?;
        section.field(&rustc_type_layout_sha256_v1(tcx, ty.layout))?;
        section.field(ty.semantic_layout_identity.as_bytes())?;
    }
    digest.section(PREFLIGHT_SECTION_TYPES_V5, types.len(), section.finish())?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_SOURCE_FILES_DOMAIN_V5,
        PREFLIGHT_SECTION_SOURCE_FILES_V5,
        "source-files child",
    );
    for source_file in source_files {
        section.field(source_file.as_bytes())?;
    }
    digest.section(
        PREFLIGHT_SECTION_SOURCE_FILES_V5,
        source_files.len(),
        section.finish(),
    )?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_FUNCTIONS_DOMAIN_V5,
        PREFLIGHT_SECTION_FUNCTIONS_V5,
        "functions child",
    );
    for function in functions {
        section.field(function.identities.function().as_bytes())?;
        section.field(&rustc_mir_body_sha256_v1(tcx, function.instance))?;
        section.field(&[function_role_tag_v1(function.role)])?;
    }
    digest.section(
        PREFLIGHT_SECTION_FUNCTIONS_V5,
        functions.len(),
        section.finish(),
    )?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_FUNCTION_ABIS_DOMAIN_V5,
        PREFLIGHT_SECTION_FUNCTION_ABIS_V5,
        "function-ABIs child",
    );
    for abi in function_abis {
        section.field(&abi.function.index().to_le_bytes())?;
        section.field(abi.identity.as_bytes())?;
        section.field(abi.layout_identity.as_bytes())?;
        section.field(&abi.rustc_source_signature_sha256)?;
        section.field(&abi.rustc_fn_abi_sha256)?;
        section.field(&usize_to_u64_v5(abi.source_inputs.len()).to_le_bytes())?;
        for input in &abi.source_inputs {
            section.field(rustc_type_identity_v1(tcx, *input).as_bytes())?;
        }
        section.field(rustc_type_identity_v1(tcx, abi.source_output).as_bytes())?;
    }
    digest.section(
        PREFLIGHT_SECTION_FUNCTION_ABIS_V5,
        function_abis.len(),
        section.finish(),
    )?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_TERMINALS_DOMAIN_V5,
        PREFLIGHT_SECTION_TERMINALS_V5,
        "terminals child",
    );
    for terminal in terminals {
        section.field(terminal.identities.function().as_bytes())?;
        section.field(terminal.identities.item_definition().as_bytes())?;
        section.field(terminal.identities.monomorphization().as_bytes())?;
        section.field(terminal.identities.generic_type_arguments().as_bytes())?;
        section.field(terminal.identities.const_generic_arguments().as_bytes())?;
        section.field(&[terminal_expansion_tag_for_schema_v1(
            terminal.expansion,
            terminal_identity_schema_v1(terminal.expansion),
        )])?;
        section.field(&terminal.abi.rustc_source_signature_sha256)?;
        section.field(&terminal.abi.rustc_fn_abi_sha256)?;
        digest_source_producer_v1(&mut section, terminal.source)?;
    }
    digest.section(
        PREFLIGHT_SECTION_TERMINALS_V5,
        terminals.len(),
        section.finish(),
    )?;

    digest.field(&[PREFLIGHT_SECTION_BODIES_V5])?;
    digest.field(&usize_to_u64_v5(bodies.len()).to_le_bytes())?;
    for (body_ordinal, body) in bodies.iter().enumerate() {
        let body_ordinal = usize_to_u64_v5(body_ordinal);
        let mut child = PreflightChildCommitmentBuilderV5::body(body_ordinal, body.function);
        child.field(&body.function.index().to_le_bytes())?;
        digest_source_producer_v1(&mut child, body.source)?;
        for local in &body.locals {
            child.field(local.identity.as_bytes())?;
            child.field(&local.rustc_local.to_le_bytes())?;
            child.field(&local.ty.index().to_le_bytes())?;
            digest_source_producer_v1(&mut child, local.source)?;
        }
        for local in &body.raw_to_semantic_locals {
            child.field(&local.index().to_le_bytes())?;
        }
        child.field(&body.entry.index().to_le_bytes())?;
        for block in &body.blocks {
            child.field(block.identity.as_bytes())?;
            child.field(&block.rustc_block.to_le_bytes())?;
            digest_source_producer_v1(&mut child, block.source)?;
            for statement in &block.statements {
                digest_source_producer_v1(&mut child, *statement)?;
            }
            digest_source_producer_v1(&mut child, block.terminator)?;
        }
        for block in &body.raw_to_semantic_blocks {
            child.field(&block.index().to_le_bytes())?;
        }
        if let Some(context) = context_entries.get(&body.function) {
            context.commitment(|field| child.field(field))?;
        }
        digest.body(body_ordinal, body, child.finish())?;
    }

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_ROOTS_DOMAIN_V5,
        PREFLIGHT_SECTION_ROOTS_V5,
        "roots child",
    );
    for root in roots {
        section.field(&root.index().to_le_bytes())?;
    }
    digest.section(PREFLIGHT_SECTION_ROOTS_V5, roots.len(), section.finish())?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_EDGES_DOMAIN_V5,
        PREFLIGHT_SECTION_EDGES_V5,
        "edges child",
    );
    for edge in edges {
        section.field(&edge.caller.index().to_le_bytes())?;
        section.field(&edge.callee.index().to_le_bytes())?;
    }
    digest.section(PREFLIGHT_SECTION_EDGES_V5, edges.len(), section.finish())?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_DIRECT_CALLS_DOMAIN_V5,
        PREFLIGHT_SECTION_DIRECT_CALLS_V5,
        "direct-calls child",
    );
    for call in direct_calls {
        section.field(&call.caller.index().to_le_bytes())?;
        section.field(&call.block.to_le_bytes())?;
        section.field(&call.callee.index().to_le_bytes())?;
    }
    digest.section(
        PREFLIGHT_SECTION_DIRECT_CALLS_V5,
        direct_calls.len(),
        section.finish(),
    )?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_TERMINAL_EXPANSIONS_DOMAIN_V5,
        PREFLIGHT_SECTION_TERMINAL_EXPANSIONS_V5,
        "terminal-expansions child",
    );
    for recipe in terminal_expansions {
        section.field(&recipe.caller.index().to_le_bytes())?;
        section.field(&recipe.block.to_le_bytes())?;
        section.field(&[terminal_expansion_tag_for_schema_v1(
            recipe.expansion,
            terminal_identity_schema_v1(recipe.expansion),
        )])?;
        section.field(&recipe.arguments.to_le_bytes())?;
        section.field(recipe.identities.function().as_bytes())?;
        section.field(recipe.identities.item_definition().as_bytes())?;
        section.field(recipe.identities.monomorphization().as_bytes())?;
        section.field(recipe.identities.generic_type_arguments().as_bytes())?;
        section.field(recipe.identities.const_generic_arguments().as_bytes())?;
        section.field(&recipe.terminal.to_le_bytes())?;
        section.field(&terminal_definition_sha256_v1(tcx, recipe))?;
    }
    digest.section(
        PREFLIGHT_SECTION_TERMINAL_EXPANSIONS_V5,
        terminal_expansions.len(),
        section.finish(),
    )?;

    let mut section = PreflightChildCommitmentBuilderV5::section(
        PREFLIGHT_NORMALIZED_INTRINSICS_DOMAIN_V5,
        PREFLIGHT_SECTION_NORMALIZED_INTRINSICS_V5,
        "normalized-intrinsics child",
    );
    let mut committed_literal_bytes = 0_u64;
    for recipe in normalized_intrinsics {
        section.field(&recipe.caller.index().to_le_bytes())?;
        section.field(&recipe.block.to_le_bytes())?;
        section.field(&[recipe.operation.operation_tag()])?;
        match recipe.operation {
            NormalizedCallV1::Rustc(operation) => {
                let (operation, access) = operation
                    .atomic_rmw()
                    .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
                section.field(&[atomic_rmw_operation_tag_v1(operation)])?;
                section.field(&[atomic_ordering_tag_v1(access.ordering())])?;
                section.field(&[atomic_scope_tag_v1(access.scope())])?;
            }
            NormalizedCallV1::SafeCoreShift(shift) => {
                section.field(b"fe2o3/safe-core-wrapping-shift/recipe/v1")?;
                section.field(&[shift.width()])?;
                let signature = source_signature_v1(tcx, recipe.instance)
                    .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
                section.field(&rustc_fn_signature_sha256_v1(tcx, signature))?;
                section.field(&rustc_fn_abi_sha256_v1(tcx, shift.abi()))?;
            }
            NormalizedCallV1::CheckedPrimitiveFrom(checked) => {
                section.field(b"fe2o3/checked-primitive-from/recipe/v1")?;
                section.field(rustc_type_identity_v1(tcx, checked.input_type()).as_bytes())?;
                let signature = source_signature_v1(tcx, recipe.instance)
                    .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
                section.field(&rustc_fn_signature_sha256_v1(tcx, signature))?;
                section.field(&rustc_fn_abi_sha256_v1(tcx, checked.abi()))?;
            }
            NormalizedCallV1::CorePanic(checked) => {
                committed_literal_bytes =
                    committed_literal_bytes
                        .checked_add(u64::try_from(checked.bytes().len()).map_err(|_| {
                            ProductionSemanticPreflightErrorV1::IdentityTableMismatch
                        })?)
                        .ok_or(ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
                section.field(b"fe2o3/core-panic-literal/recipe/v50")?;
                section.field(rustc_type_identity_v1(tcx, checked.message_type()).as_bytes())?;
                section.field(checked.bytes())?;
                let signature = source_signature_v1(tcx, recipe.instance)
                    .map_err(|_| ProductionSemanticPreflightErrorV1::IdentityTableMismatch)?;
                section.field(&rustc_fn_signature_sha256_v1(tcx, signature))?;
                section.field(&rustc_fn_abi_sha256_v1(tcx, checked.abi()))?;
            }
        }
        section.field(rustc_type_identity_v1(tcx, recipe.element_type).as_bytes())?;
        section.field(recipe.identities.function().as_bytes())?;
        section.field(recipe.identities.item_definition().as_bytes())?;
        section.field(recipe.identities.monomorphization().as_bytes())?;
        section.field(recipe.identities.generic_type_arguments().as_bytes())?;
        section.field(recipe.identities.const_generic_arguments().as_bytes())?;
        section.field(&normalized_intrinsic_definition_sha256_v1(tcx, recipe))?;
    }
    if committed_literal_bytes != counts.constant_bytes {
        return Err(ProductionSemanticPreflightErrorV1::IdentityTableMismatch);
    }
    digest.section(
        PREFLIGHT_SECTION_NORMALIZED_INTRINSICS_V5,
        normalized_intrinsics.len(),
        section.finish(),
    )?;
    digest.finish()
}

const fn atomic_rmw_operation_tag_v1(
    operation: fe2o3_mir_model::semantic_mir_v1::SemanticAtomicRmwOpV1,
) -> u8 {
    use fe2o3_mir_model::semantic_mir_v1::SemanticAtomicRmwOpV1;
    match operation {
        SemanticAtomicRmwOpV1::Exchange => 0,
        SemanticAtomicRmwOpV1::Add => 1,
        SemanticAtomicRmwOpV1::Subtract => 2,
        SemanticAtomicRmwOpV1::BitAnd => 3,
        SemanticAtomicRmwOpV1::BitNand => 4,
        SemanticAtomicRmwOpV1::BitOr => 5,
        SemanticAtomicRmwOpV1::BitXor => 6,
        SemanticAtomicRmwOpV1::SignedMaximum => 7,
        SemanticAtomicRmwOpV1::SignedMinimum => 8,
        SemanticAtomicRmwOpV1::UnsignedMaximum => 9,
        SemanticAtomicRmwOpV1::UnsignedMinimum => 10,
    }
}

fn normalized_intrinsic_definition_sha256_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    recipe: &NormalizedRustcIntrinsicRecipeV1<'tcx>,
) -> [u8; 32] {
    match recipe.operation {
        NormalizedCallV1::Rustc(_) => {
            rustc_intrinsic_definition_sha256_v1(tcx, recipe.instance, recipe.identities)
        }
        NormalizedCallV1::SafeCoreShift(shift) => {
            borrowed_rustc_mir_body_sha256_v1(tcx, recipe.instance, shift.body())
        }
        NormalizedCallV1::CheckedPrimitiveFrom(checked) => {
            borrowed_rustc_mir_body_sha256_v1(tcx, recipe.instance, checked.body())
        }
        NormalizedCallV1::CorePanic(checked) => {
            borrowed_rustc_mir_body_sha256_v1(tcx, recipe.instance, checked.body())
        }
    }
}

fn terminal_definition_sha256_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    recipe: &TerminalExpansionRecipeV1<'tcx>,
) -> [u8; 32] {
    rustc_intrinsic_definition_sha256_v1(tcx, recipe.instance, recipe.identities)
}

fn rustc_intrinsic_definition_sha256_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    identities: CanonicalFunctionIdentitiesV1,
) -> [u8; 32] {
    let InstanceKind::Intrinsic(def_id) = instance.def else {
        return rustc_mir_body_sha256_v1(tcx, instance);
    };
    let intrinsic = tcx
        .intrinsic(def_id)
        .expect("a resolved intrinsic instance has intrinsic metadata");
    let mut digest = SemanticIdentityDigestV1::new(COMPILER_INTRINSIC_DEFINITION_DOMAIN_V1);
    digest.field(identities.function().as_bytes());
    digest.field(identities.item_definition().as_bytes());
    digest.field(intrinsic.name.as_str().as_bytes());
    digest.field(&[u8::from(intrinsic.must_be_overridden)]);
    digest.field(&[u8::from(intrinsic.const_stable)]);
    digest.finish()
}

fn source_provenance_producer_count_v1(bodies: &[RetainedSemanticBodyProducerV1]) -> usize {
    bodies
        .iter()
        .map(|body| {
            1 + body.locals.len()
                + body
                    .blocks
                    .iter()
                    .map(|block| 2 + block.statements.len())
                    .sum::<usize>()
        })
        .sum()
}

fn digest_source_producer_v1(
    digest: &mut PreflightChildCommitmentBuilderV5,
    source: RetainedSemanticSourceProducerV1,
) -> Result<(), ProductionSemanticPreflightErrorV1> {
    digest.field(&source.expansion_chain_sha256)?;
    digest_source_origin_v1(digest, source.provenance.expansion())?;
    digest_source_origin_v1(digest, source.provenance.call_site())
}

fn digest_source_origin_v1(
    digest: &mut PreflightChildCommitmentBuilderV5,
    origin: Option<SemanticSourceOriginV1>,
) -> Result<(), ProductionSemanticPreflightErrorV1> {
    let Some(origin) = origin else {
        return digest.field(&[0]);
    };
    digest.field(&[1])?;
    digest.field(origin.file().as_bytes())?;
    let (byte_start, byte_end) = origin.byte_range();
    digest.field(&byte_start.to_le_bytes())?;
    digest.field(&byte_end.to_le_bytes())?;
    let (line_start, column_start) = origin.start_coordinate();
    digest.field(&line_start.to_le_bytes())?;
    digest.field(&column_start.to_le_bytes())?;
    let (line_end, column_end) = origin.end_coordinate();
    digest.field(&line_end.to_le_bytes())?;
    digest.field(&column_end.to_le_bytes())
}

const fn function_role_tag_v1(role: CollectedFunctionRole) -> u8 {
    match role {
        CollectedFunctionRole::KernelEntry => 0,
        CollectedFunctionRole::InternalHelper => 1,
        CollectedFunctionRole::DeviceFfiExport => 2,
    }
}

#[cfg(test)]
// Frozen calculator for the independently published BF16 and pipeline V1 histories.
const fn terminal_expansion_tag_v1(expansion: ProductionTerminalExpansionV1) -> u8 {
    terminal_expansion_tag_for_schema_v1(expansion, TerminalIdentitySchemaV1::IndependentV1)
}

const fn terminal_identity_schema_v1(
    expansion: ProductionTerminalExpansionV1,
) -> TerminalIdentitySchemaV1 {
    match expansion {
        ProductionTerminalExpansionV1::Gfx942PhysicalEntryBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryStep => {
            TerminalIdentitySchemaV1::CombinedV7
        }
        ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeStep => {
            TerminalIdentitySchemaV1::CombinedV9
        }
        ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyStep => {
            TerminalIdentitySchemaV1::CombinedV8
        }
        ProductionTerminalExpansionV1::Gfx942CompleteBodyE32 => {
            TerminalIdentitySchemaV1::CombinedV6
        }
        ProductionTerminalExpansionV1::Gfx942OrderedProgramE32 => {
            TerminalIdentitySchemaV1::CombinedV5
        }
        _ => TerminalIdentitySchemaV1::CombinedV4,
    }
}

const fn terminal_expansion_tag_for_schema_v1(
    expansion: ProductionTerminalExpansionV1,
    schema: TerminalIdentitySchemaV1,
) -> u8 {
    // V9 belongs only to the three physical-lds-exchange primitives.
    if matches!(schema, TerminalIdentitySchemaV1::CombinedV9) {
        return match expansion {
            ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeBegin => 151,
            ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeLabel => 152,
            ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeStep => 153,
            _ => u8::MAX,
        };
    }
    // V8 belongs only to the three physical-global-copy primitives.
    if matches!(schema, TerminalIdentitySchemaV1::CombinedV8) {
        return match expansion {
            ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyBegin => 148,
            ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyLabel => 149,
            ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyStep => 150,
            _ => u8::MAX,
        };
    }
    // V7 belongs only to the three physical-entry primitives.
    if matches!(schema, TerminalIdentitySchemaV1::CombinedV7) {
        return match expansion {
            ProductionTerminalExpansionV1::Gfx942PhysicalEntryBegin => 145,
            ProductionTerminalExpansionV1::Gfx942PhysicalEntryLabel => 146,
            ProductionTerminalExpansionV1::Gfx942PhysicalEntryStep => 147,
            _ => u8::MAX,
        };
    }
    // V6 belongs only to terminal144. Never re-label an old terminal identity.
    if matches!(schema, TerminalIdentitySchemaV1::CombinedV6) {
        return if matches!(
            expansion,
            ProductionTerminalExpansionV1::Gfx942CompleteBodyE32
        ) {
            144
        } else {
            u8::MAX
        };
    }
    match expansion {
        ProductionTerminalExpansionV1::Gfx942PhysicalEntryBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryStep => u8::MAX,
        ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyStep => u8::MAX,
        ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeStep => u8::MAX,
        ProductionTerminalExpansionV1::Gfx942CompleteBodyE32 => u8::MAX,
        ProductionTerminalExpansionV1::Gfx942OrderedProgramE32 => {
            if matches!(schema, TerminalIdentitySchemaV1::CombinedV5) {
                134
            } else {
                u8::MAX
            }
        }
        ProductionTerminalExpansionV1::Gfx942OrderedXorAddE32 => {
            if matches!(
                schema,
                TerminalIdentitySchemaV1::CombinedV4 | TerminalIdentitySchemaV1::CombinedV5
            ) {
                133
            } else {
                u8::MAX
            }
        }
        ProductionTerminalExpansionV1::Gfx942InlineU32(operation) => {
            if matches!(
                schema,
                TerminalIdentitySchemaV1::CombinedV4 | TerminalIdentitySchemaV1::CombinedV5
            ) {
                crate::production_inline_assembly_v30::source_terminal_tag(operation)
            } else {
                u8::MAX
            }
        }
        // Draft allocation requested in #271; do not publish before acknowledgment.
        ProductionTerminalExpansionV1::ContextIssue => 122,
        ProductionTerminalExpansionV1::WorkgroupDerive => 123,
        ProductionTerminalExpansionV1::MaskedTileLoadU32 => 124,
        ProductionTerminalExpansionV1::MaskedTileIntoFragmentU32 => 125,
        ProductionTerminalExpansionV1::LaneFragmentIntoPartsU32 => 126,
        ProductionTerminalExpansionV1::ThreadIndex(SemanticAxisV1::X) => 13,
        ProductionTerminalExpansionV1::ThreadIndex(SemanticAxisV1::Y) => 14,
        ProductionTerminalExpansionV1::ThreadIndex(SemanticAxisV1::Z) => 15,
        ProductionTerminalExpansionV1::WorkgroupIndex(SemanticAxisV1::X) => 16,
        ProductionTerminalExpansionV1::WorkgroupIndex(SemanticAxisV1::Y) => 17,
        ProductionTerminalExpansionV1::WorkgroupIndex(SemanticAxisV1::Z) => 18,
        ProductionTerminalExpansionV1::WorkgroupDimension(SemanticAxisV1::X) => 19,
        ProductionTerminalExpansionV1::WorkgroupDimension(SemanticAxisV1::Y) => 20,
        ProductionTerminalExpansionV1::WorkgroupDimension(SemanticAxisV1::Z) => 21,
        ProductionTerminalExpansionV1::GridDimension(SemanticAxisV1::X) => 22,
        ProductionTerminalExpansionV1::GridDimension(SemanticAxisV1::Y) => 23,
        ProductionTerminalExpansionV1::GridDimension(SemanticAxisV1::Z) => 24,
        ProductionTerminalExpansionV1::DisjointSliceLen => 25,
        ProductionTerminalExpansionV1::ThreadIndex1d => 0,
        ProductionTerminalExpansionV1::ThreadIndexGet => 1,
        ProductionTerminalExpansionV1::DisjointSliceGetMut => 2,
        ProductionTerminalExpansionV1::ThreadIndexIntoDisjoint => 3,
        ProductionTerminalExpansionV1::ThreadIndexCheckedShift => 4,
        ProductionTerminalExpansionV1::DisjointIndexGet => 5,
        ProductionTerminalExpansionV1::DisjointIndexCheckedShift => 6,
        ProductionTerminalExpansionV1::DisjointSliceGetDisjointMut => 7,
        ProductionTerminalExpansionV1::GridLeaderCurrent => 8,
        ProductionTerminalExpansionV1::DisjointSliceGetMutExclusive => 9,
        ProductionTerminalExpansionV1::ThreadIndexCheckedBlock => 10,
        ProductionTerminalExpansionV1::DisjointSliceGetBlockMut => 11,
        ProductionTerminalExpansionV1::WorkgroupBarrier => 12,
        ProductionTerminalExpansionV1::MatrixContextCurrent => 26,
        ProductionTerminalExpansionV1::F32MatrixAccumulatorIntoValues => 29,
        ProductionTerminalExpansionV1::MatrixMultiplyAccumulate => 30,
        ProductionTerminalExpansionV1::ThreadIndexCheckedTiled2d => 31,
        ProductionTerminalExpansionV1::DisjointSliceGetTiled2dMut => 32,
        ProductionTerminalExpansionV1::CollectiveContextCurrent => 33,
        ProductionTerminalExpansionV1::SubgroupReduceSumF32 => 34,
        ProductionTerminalExpansionV1::SubgroupReduceMaxF32 => 35,
        ProductionTerminalExpansionV1::MathContextCurrent => 36,
        ProductionTerminalExpansionV1::MathF32(function) => 37 + f32_math_tag_v1(function),
        ProductionTerminalExpansionV1::ColdPath => 50,
        ProductionTerminalExpansionV1::WaveLaneCurrent => 51,
        ProductionTerminalExpansionV1::Bf16MatrixARowMajor => 52,
        ProductionTerminalExpansionV1::Bf16MatrixBRowMajor => 53,
        ProductionTerminalExpansionV1::Bf16MatrixALoadZeroFilledV2 => 54,
        ProductionTerminalExpansionV1::Bf16MatrixBLoadZeroFilledV2 => 55,
        ProductionTerminalExpansionV1::F32MatrixAccumulatorZero => 56,
        ProductionTerminalExpansionV1::StridedReadView2DFromSharedSlice => 57,
        ProductionTerminalExpansionV1::StridedReadView2DLoadOr => 58,
        ProductionTerminalExpansionV1::ThreadIndexCheckedRowStriped2d => 59,
        ProductionTerminalExpansionV1::DisjointSliceGetRowStriped2dMut => 60,
        ProductionTerminalExpansionV1::Gfx950MatrixContextCurrent => 61,
        ProductionTerminalExpansionV1::Gfx950Fp8MatrixARowMajor => 62,
        ProductionTerminalExpansionV1::Gfx950Fp8MatrixBRowMajor => 63,
        ProductionTerminalExpansionV1::Gfx950Fp8MatrixALoadM16K128 => 64,
        ProductionTerminalExpansionV1::Gfx950Fp8MatrixBLoadK128N16 => 65,
        ProductionTerminalExpansionV1::Gfx950Fp8AccumulatorZero => 66,
        ProductionTerminalExpansionV1::Gfx950Fp8AccumulatorIntoValues => 67,
        ProductionTerminalExpansionV1::Gfx950Fp8MultiplyAccumulate => 68,
        ProductionTerminalExpansionV1::Gfx950Fp4MatrixARowMajor => 69,
        ProductionTerminalExpansionV1::Gfx950Fp4MatrixBRowMajor => 70,
        ProductionTerminalExpansionV1::Gfx950Fp4MatrixALoadM16K128 => 71,
        ProductionTerminalExpansionV1::Gfx950Fp4MatrixBLoadK128N16 => 72,
        ProductionTerminalExpansionV1::Gfx950Fp4AccumulatorZero => 73,
        ProductionTerminalExpansionV1::Gfx950Fp4AccumulatorIntoValues => 74,
        ProductionTerminalExpansionV1::Gfx950Fp4MultiplyAccumulate => 75,
        ProductionTerminalExpansionV1::Gfx950SubgroupCurrent => 76,
        ProductionTerminalExpansionV1::Gfx950SubgroupReduceMaxF32 => 77,
        ProductionTerminalExpansionV1::Gfx950SubgroupReduceSumF32 => 78,
        ProductionTerminalExpansionV1::Gfx950SubgroupBroadcastF32 => 79,
        ProductionTerminalExpansionV1::Gfx950LdsTransposeTileCurrent => 80,
        ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB4 => 81,
        ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB8 => 82,
        ProductionTerminalExpansionV1::Gfx950LdsTransposePublish => 83,
        ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB4 => 84,
        ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB8 => 85,
        ProductionTerminalExpansionV1::Trap => 86,
        ProductionTerminalExpansionV1::Gfx950Fp4Fp8MultiplyAccumulate => 87,
        ProductionTerminalExpansionV1::DynamicLdsExactCurrent => 88,
        ProductionTerminalExpansionV1::WorkgroupReduceSum => 89,
        ProductionTerminalExpansionV1::DynamicLdsIntoCollectiveRawParts => 90,
        ProductionTerminalExpansionV1::WorkgroupPipelineCurrent => 91,
        ProductionTerminalExpansionV1::WorkgroupPipelineStage => 92,
        ProductionTerminalExpansionV1::WorkgroupPipelineWrite => 93,
        ProductionTerminalExpansionV1::WorkgroupPipelineCommit => 94,
        ProductionTerminalExpansionV1::WorkgroupPipelineWait => 95,
        ProductionTerminalExpansionV1::WorkgroupPipelineConsume => 96,
        ProductionTerminalExpansionV1::WorkgroupPipelineRead => 97,
        ProductionTerminalExpansionV1::WorkgroupPipelineDiscard => 98,
        ProductionTerminalExpansionV1::WorkgroupPipelineRelease => 99,
        ProductionTerminalExpansionV1::WorkgroupCollectiveContextCurrent => match schema {
            #[cfg(test)]
            TerminalIdentitySchemaV1::IndependentV1 | TerminalIdentitySchemaV1::CombinedV2 => 104,
            TerminalIdentitySchemaV1::CombinedV3
            | TerminalIdentitySchemaV1::CombinedV4
            | TerminalIdentitySchemaV1::CombinedV5 => 111,
            TerminalIdentitySchemaV1::CombinedV6
            | TerminalIdentitySchemaV1::CombinedV7
            | TerminalIdentitySchemaV1::CombinedV8
            | TerminalIdentitySchemaV1::CombinedV9 => u8::MAX,
        },
        ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum => match schema {
            #[cfg(test)]
            TerminalIdentitySchemaV1::IndependentV1 | TerminalIdentitySchemaV1::CombinedV2 => 105,
            TerminalIdentitySchemaV1::CombinedV3
            | TerminalIdentitySchemaV1::CombinedV4
            | TerminalIdentitySchemaV1::CombinedV5 => 112,
            TerminalIdentitySchemaV1::CombinedV6
            | TerminalIdentitySchemaV1::CombinedV7
            | TerminalIdentitySchemaV1::CombinedV8
            | TerminalIdentitySchemaV1::CombinedV9 => u8::MAX,
        },
        ProductionTerminalExpansionV1::RustcFabsF32 => 113,
        ProductionTerminalExpansionV1::Gfx942Wave64Shuffle(scalar) => scalar.terminal_tag(),
        ProductionTerminalExpansionV1::RustcSaturatingInteger(operation) => match operation {
            fe2o3_mir_model::semantic_mir_v1::SemanticSaturatingIntegerOpV1::Add => 127,
            fe2o3_mir_model::semantic_mir_v1::SemanticSaturatingIntegerOpV1::Subtract => 128,
        },
        ProductionTerminalExpansionV1::MemoryVolatileLoad => 115,
        ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum => match schema {
            #[cfg(test)]
            TerminalIdentitySchemaV1::IndependentV1 | TerminalIdentitySchemaV1::CombinedV2 => 113,
            TerminalIdentitySchemaV1::CombinedV3 => 113,
            TerminalIdentitySchemaV1::CombinedV4 | TerminalIdentitySchemaV1::CombinedV5 => 116,
            TerminalIdentitySchemaV1::CombinedV6
            | TerminalIdentitySchemaV1::CombinedV7
            | TerminalIdentitySchemaV1::CombinedV8
            | TerminalIdentitySchemaV1::CombinedV9 => u8::MAX,
        },
        ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum => match schema {
            #[cfg(test)]
            TerminalIdentitySchemaV1::IndependentV1 | TerminalIdentitySchemaV1::CombinedV2 => 114,
            TerminalIdentitySchemaV1::CombinedV3 => 114,
            TerminalIdentitySchemaV1::CombinedV4 | TerminalIdentitySchemaV1::CombinedV5 => 117,
            TerminalIdentitySchemaV1::CombinedV6
            | TerminalIdentitySchemaV1::CombinedV7
            | TerminalIdentitySchemaV1::CombinedV8
            | TerminalIdentitySchemaV1::CombinedV9 => u8::MAX,
        },
        ProductionTerminalExpansionV1::WorkgroupLdsScopeCurrent => 118,
        ProductionTerminalExpansionV1::DisjointBlockComponentIndex => 119,
        ProductionTerminalExpansionV1::Bf16MatrixBColumnMajor => 120,
        ProductionTerminalExpansionV1::Bf16MatrixBColumnMajorLoadZeroFilledV1 => 121,
        ProductionTerminalExpansionV1::Bf16Conversion(conversion) => {
            let base = match schema {
                #[cfg(test)]
                TerminalIdentitySchemaV1::IndependentV1 => 91,
                #[cfg(test)]
                TerminalIdentitySchemaV1::CombinedV2 => 100,
                TerminalIdentitySchemaV1::CombinedV3
                | TerminalIdentitySchemaV1::CombinedV4
                | TerminalIdentitySchemaV1::CombinedV5 => 100,
                TerminalIdentitySchemaV1::CombinedV6
                | TerminalIdentitySchemaV1::CombinedV7
                | TerminalIdentitySchemaV1::CombinedV8
                | TerminalIdentitySchemaV1::CombinedV9 => {
                    return u8::MAX;
                }
            };
            base + match conversion {
                crate::production_semantic_terminal_v1::ProductionBf16ConversionV1::FromBits => 0,
                crate::production_semantic_terminal_v1::ProductionBf16ConversionV1::ToBits => 1,
                crate::production_semantic_terminal_v1::ProductionBf16ConversionV1::FromF32RoundTiesEven => 2,
                crate::production_semantic_terminal_v1::ProductionBf16ConversionV1::ToF32 => 3,
            }
        }
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceLen => 104,
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWrite => 105,
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteDisjoint => 106,
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteExclusive => 107,
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteBlock => 108,
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteTiled2d => 109,
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteRowStriped2d => 110,
    }
}

const fn f32_math_tag_v1(function: fe2o3_kernel_ir::F32MathFunction) -> u8 {
    use fe2o3_kernel_ir::F32MathFunction as Function;
    match function {
        Function::Sqrt => 0,
        Function::FusedMultiplyAdd => 1,
        Function::Floor => 2,
        Function::Ceil => 3,
        Function::Truncate => 4,
        Function::RoundTiesEven => 5,
        Function::Sin => 6,
        Function::Cos => 7,
        Function::Exp => 8,
        Function::Exp2 => 9,
        Function::Ln => 10,
        Function::Log2 => 11,
        Function::Log10 => 12,
        Function::Abs => 77,
    }
}

#[cfg(test)]
mod tests {
    include!("rustc_semantic_plan_v1/tests.rs");
    include!("rustc_semantic_plan_v1/ordered_program_v32_tests.rs");
    include!("rustc_semantic_plan_v1/complete_body_v36_tests.rs");
    include!("rustc_semantic_plan_v1/physical_entry_v37_tests.rs");
    include!("rustc_semantic_plan_v1/physical_global_copy_v38_tests.rs");
    include!("rustc_semantic_plan_v1/physical_lds_exchange_v39_tests.rs");

    #[test]
    fn ordered_region_v31_preflight_tag_is_only_combined_v4_133() {
        let region = ProductionTerminalExpansionV1::Gfx942OrderedXorAddE32;
        assert_eq!(
            terminal_expansion_tag_for_schema_v1(region, TerminalIdentitySchemaV1::CombinedV4),
            133
        );
        for schema in [
            TerminalIdentitySchemaV1::IndependentV1,
            TerminalIdentitySchemaV1::CombinedV2,
            TerminalIdentitySchemaV1::CombinedV3,
        ] {
            assert_eq!(
                terminal_expansion_tag_for_schema_v1(region, schema),
                u8::MAX
            );
        }
    }

    mod source_closure_work_tests {
        include!("rustc_semantic_plan_v1/source_closure_work_tests.rs");
    }
}

#[cfg(test)]
#[path = "rustc_semantic_plan_debug_retained_storage_v1_tests.rs"]
mod debug_retained_storage_v1_tests;

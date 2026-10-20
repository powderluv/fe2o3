use fe2o3_artifacts::{BlockSize, Dimensions, LaunchContract};
use fe2o3_kernel_descriptor::MAX_ARGUMENTS_PER_KERNEL;
use fe2o3_rustc_front::{
    ASSEMBLY_OPERAND_ADDRESS_V1, ASSEMBLY_OPERAND_IMMEDIATE_V1, ASSEMBLY_OPERAND_SGPR_V1,
    ASSEMBLY_OPERAND_VGPR_V1, ASSEMBLY_OPTION_NOMEM_V1, ASSEMBLY_OPTION_NOSTACK_V1,
    ASSEMBLY_OPTION_PRESERVES_FLAGS_V1, ASSEMBLY_OPTION_PURE_V1, ASSEMBLY_OPTION_READONLY_V1,
    KERNEL_FRONTEND_REGISTRATION_KIND_V1, KERNEL_FRONTEND_REGISTRATION_MAGIC_V1,
    KERNEL_FRONTEND_REGISTRATION_PREFIX_V1, KERNEL_FRONTEND_REGISTRATION_VERSION_V1,
    KERNEL_RESOURCE_REGISTRATION_KIND_V1, KERNEL_RESOURCE_REGISTRATION_MAGIC_V1,
    KERNEL_RESOURCE_REGISTRATION_PREFIX_V1, KERNEL_RESOURCE_REGISTRATION_VERSION_V1,
    KernelFrontendContractV1, KernelResourceContractV1, decode_kernel_frontend_contract_v1,
    decode_kernel_resource_contract_v1,
};
use reserved_fe2o3_symbols::{
    CrateBindingIdV1, GeneratedHostContractIdV3, KernelBindingIdV1,
    TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3, derive_crate_binding_id_v1,
    derive_kernel_binding_id_v1, host_kernel_symbol_v1,
};
use rustc_ast::InlineAsmOptions;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{BlockCheckMode, ExprKind, ItemKind, Safety, UnsafeSource};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::GlobalAlloc;
use rustc_middle::mir::mono::{CodegenUnit, MonoItem};
use rustc_middle::mir::{
    AggregateKind, BasicBlock, Body, CastKind, InlineAsmMacro, InlineAsmOperand, Operand,
    RETURN_PLACE, Rvalue, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{
    EarlyBinder, Instance, InstanceKind, TyCtxt, TyKind, TypeVisitableExt, TypingEnv,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use crate::production_rustc_drop_v1::{ProductionRustcDropClassV1, classify_rustc_drop_v1};

mod kernel_context_auth_v1;
pub(crate) mod workgroup_scope_custody_v29;
pub(crate) use kernel_context_auth_v1::{
    AuthenticatedContextEntriesV1, BoundContextEntryV29, CallBoundaryV29,
    CapturedContextProducersV1, ContextRootVisitErrorV29, RetainedContextEntriesV29,
    RetainedContextEntryV29, RetainedExecutionSourceV29, capture_context_producers_v1,
};
mod closure_flow_v1;
mod inlined_source_safety_v1;
mod kernel_context_frontend_v1;
#[cfg(test)]
pub(crate) mod primitive_from_stage_tests;
mod production_importer_v1;
mod reference_custody_v1;
mod reference_enrollment_v1;

#[cfg(test)]
mod reference_loan_flow_tests;
#[cfg(test)]
pub(crate) use production_importer_v1::original_root_inventory_flow_v362_tests::{
    check_original_root_inventory_flow, reversed_original_root_inventory_request,
};
#[cfg(test)]
pub(crate) use reference_loan_flow_tests::check_empty_enrollment_replay;

#[cfg(test)]
pub(crate) use production_importer_v1::check_wave64_descriptor_mutations_v1;
#[cfg(test)]
pub(crate) mod semantic_import_observation_v1_tests;

pub(crate) use production_importer_v1::{
    AuthenticatedBf16MfmaInspectionMirV1, AuthenticatedBf16TileValuesMirV1,
    AuthenticatedOrderedCompositionMirV1, AuthenticatedRustcIdentityInventoryV3,
    AuthenticatedRustcPreflightPlanV3, ConstructedProductionSemanticMirV1,
    ProductionSemanticImportErrorV1, construct_production_semantic_mir_atomic_v41,
    construct_production_semantic_mir_bf16_inspection_v1,
    construct_production_semantic_mir_bf16_tile_values_v1,
    construct_production_semantic_mir_nominal_v35,
    construct_production_semantic_mir_ordered_composition_v1,
    construct_production_semantic_mir_source_owned_v29, construct_production_semantic_mir_v1,
    construct_production_semantic_mir_with_enrollment_v1,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TypedArgumentListV1<T> {
    arguments: Vec<T>,
}

impl<T> TypedArgumentListV1<T> {
    pub(crate) fn new(arguments: Vec<T>) -> Result<Self, TypedArgumentListError> {
        if arguments.len() > MAX_ARGUMENTS_PER_KERNEL {
            return Err(TypedArgumentListError::TooMany {
                actual: arguments.len(),
                maximum: MAX_ARGUMENTS_PER_KERNEL,
            });
        }
        Ok(Self { arguments })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.arguments
    }

    pub(crate) fn len(&self) -> usize {
        self.arguments.len()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TypedArgumentListError {
    TooMany { actual: usize, maximum: usize },
}

impl fmt::Display for TypedArgumentListError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooMany { actual, maximum } => write!(
                formatter,
                "typed kernel argument count {actual} exceeds maximum {maximum}"
            ),
        }
    }
}

/// Compiler-authoritative role attached when a function enters the closed
/// device graph. Downstream consumers must preserve it rather than reconstruct
/// it from symbol spelling or later reachability analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CollectedFunctionRole {
    KernelEntry,
    InternalHelper,
    DeviceFfiExport,
}

#[derive(Clone, Debug)]
pub struct CollectedFunction<'tcx> {
    pub instance: Instance<'tcx>,
    pub(crate) role: CollectedFunctionRole,
    pub export_name: String,
    /// Present only for registered kernel roots.
    pub(crate) logical_name: Option<String>,
    /// Present only for a generic typed V3 registration.
    pub(crate) generated_host_contract_identity: Option<GeneratedHostContractIdV3>,
    pub(crate) kernel_binding: Option<KernelBindingIdV1>,
    /// Compiler-authenticated source contract for this exact kernel root.
    pub(crate) frontend_contract: Option<AuthenticatedKernelFrontendContractV1>,
    /// Exact safe-Rust reference/effect binding for this kernel root.
    pub(crate) reference_effect_binding:
        Option<crate::reference_effect_v1::AuthenticatedReferenceEffectBindingV1>,
    /// Live reference instance retained until semantic import rederives the binding.
    reference_instance: Option<Instance<'tcx>>,
    /// Compiler-private observation derived from this exact monomorphized MIR.
    pub(crate) dead_branches: Option<crate::monomorphization_dead::CompilerDeadBranchObservationV1>,
    closure_observation: Option<Box<crate::closure_profile_v1::CompilerClosureObservationV2>>,
    kernel_context_contract: Option<kernel_context_frontend_v1::BoundContextEntryV1>,
}

/// Source-level kernel contract authenticated against one exact rustc instance.
///
/// This is compiler evidence only. It grants no code-generation, proof, load,
/// or launch authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AuthenticatedKernelFrontendContractV1 {
    registration_path: String,
    target_def_path_hash: [u8; 16],
    target_symbol: String,
    canonical_bytes: Vec<u8>,
    contract: KernelFrontendContractV1,
    resource: Option<AuthenticatedKernelResourceContractV1>,
    reachable_assembly: ReachableAssemblySummaryV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuthenticatedKernelResourceContractV1 {
    canonical_bytes: Vec<u8>,
    contract: KernelResourceContractV1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReachableAssemblySummaryV1 {
    blocks: u32,
    operand_bits: u16,
    option_bits: u16,
}

impl AuthenticatedKernelFrontendContractV1 {
    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub(crate) const fn contract(&self) -> KernelFrontendContractV1 {
        self.contract
    }

    pub(crate) const fn resource_contract(&self) -> Option<KernelResourceContractV1> {
        match &self.resource {
            Some(resource) => Some(resource.contract),
            None => None,
        }
    }

    pub(crate) fn resource_canonical_bytes(&self) -> Option<&[u8]> {
        self.resource
            .as_ref()
            .map(|resource| resource.canonical_bytes.as_slice())
    }

    pub(crate) const fn reachable_assembly(&self) -> ReachableAssemblySummaryV1 {
        self.reachable_assembly
    }

    #[cfg(test)]
    pub(crate) fn for_test(contract: KernelFrontendContractV1) -> Self {
        Self {
            registration_path: "tests::__fe2o3_kernel_frontend_contract_v1_kernel".to_owned(),
            target_def_path_hash: [0x5a; 16],
            target_symbol: "fe2o3_kernel_kernel".to_owned(),
            canonical_bytes: fe2o3_rustc_front::encode_kernel_frontend_contract_v1(contract),
            contract,
            resource: None,
            reachable_assembly: ReachableAssemblySummaryV1::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test_with_resource(
        contract: KernelFrontendContractV1,
        resource: KernelResourceContractV1,
    ) -> Self {
        let mut authenticated = Self::for_test(contract);
        authenticated.resource = Some(AuthenticatedKernelResourceContractV1 {
            canonical_bytes: fe2o3_rustc_front::encode_kernel_resource_contract_v1(resource),
            contract: resource,
        });
        authenticated
    }
}

impl ReachableAssemblySummaryV1 {
    pub(crate) const fn blocks(self) -> u32 {
        self.blocks
    }

    pub(crate) const fn operand_bits(self) -> u16 {
        self.operand_bits
    }

    pub(crate) const fn option_bits(self) -> u16 {
        self.option_bits
    }
}

#[derive(Clone, Debug, Default)]
pub struct CollectionResult<'tcx> {
    pub functions: Vec<CollectedFunction<'tcx>>,
    // Private source state retained for compiler-envelope construction.
    pub(crate) device_ffi: crate::device_ffi::DeviceFfiClosure,
    /// Inert canonical observation produced from the successfully closed graph.
    pub(crate) compiler_ffi_observation: Option<fe2o3_compiler_ffi::CompilerFfiEnvelopeV1>,
}

#[derive(Debug)]
struct AuthenticatedProductionRootV1<'tcx> {
    instance: Instance<'tcx>,
    role: CollectedFunctionRole,
    export_name: String,
}

/// Private move-only custody returned only by selector-independent production
/// collection. The semantic importer must consume this value while `TyCtxt`
/// remains live.
pub(crate) struct AuthenticatedCollectedKernelClosureV1<'tcx> {
    target: crate::production_target_v1::RetainedProductionTargetV1,
    collection: CollectionResult<'tcx>,
    roots: Box<[AuthenticatedProductionRootV1<'tcx>]>,
    context_entries: kernel_context_auth_v1::AuthenticatedContextEntriesV1<'tcx>,
    closure_flow: closure_flow_v1::AuthenticatedClosureFlowV1<'tcx>,
    terminal_census: CollectedTileTerminalCensusV259,
}

mod terminal_census_v259;
pub(crate) use terminal_census_v259::{
    CollectedTileTerminalCensusV259, CollectedTileTerminalKindV259,
};

#[path = "production_source_census_v1.rs"]
pub(crate) mod source_census_v1;

#[path = "production_ordered_origin_capture_v1.rs"]
pub(crate) mod ordered_origin_v1;

impl<'tcx> AuthenticatedCollectedKernelClosureV1<'tcx> {
    #[cfg(test)]
    pub(crate) fn collected_root_count_for_test_v259(&self) -> usize {
        self.roots.len()
    }

    pub(crate) fn collected_tile_terminals_v259(
        &self,
        budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<
        &CollectedTileTerminalCensusV259,
        fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1,
    > {
        self.terminal_census.borrow_on_account(budget)
    }

    /// Observes an exact trusted terminal in the already sealed closure.
    /// This chooses a continuation only; import repeats all source/provider,
    /// signature, FnABI and source-record checks before admitting MIR36.
    pub(crate) fn contains_complete_body_terminal_v19(&self, tcx: TyCtxt<'tcx>) -> bool {
        self.collection.functions.iter().any(|function| {
            let body = tcx.instance_mir(function.instance.def);
            body.basic_blocks.iter().any(|block| {
                let TerminatorKind::Call { func, .. } = &block.terminator().kind else {
                    return false;
                };
                let TyKind::FnDef(def_id, _) = *func.ty(body, tcx).kind() else {
                    return false;
                };
                matches!(
                    crate::production_semantic_terminal_v1::classify(tcx, def_id),
                    Some(crate::production_semantic_terminal_v1::ProductionSemanticTerminalRuleV1::Expand(
                        crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1::Gfx942CompleteBodyE32
                    ))
                )
            })
        })
    }

    /// Selects diagnostic MIR37 continuation only; import still authenticates
    /// every actual marker Instance, source occurrence, signature and FnABI.
    pub(crate) fn contains_physical_entry_terminal_v20(&self, tcx: TyCtxt<'tcx>) -> bool {
        self.collection.functions.iter().any(|function| {
            let body = tcx.instance_mir(function.instance.def);
            body.basic_blocks.iter().any(|block| {
                let TerminatorKind::Call { func, .. } = &block.terminator().kind else {
                    return false;
                };
                let TyKind::FnDef(def_id, _) = *func.ty(body, tcx).kind() else {
                    return false;
                };
                matches!(
                    crate::production_semantic_terminal_v1::classify(tcx, def_id),
                    Some(crate::production_semantic_terminal_v1::ProductionSemanticTerminalRuleV1::Expand(
                        crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1::Gfx942PhysicalEntryBegin
                    ))
                )
            })
        })
    }

    pub(crate) fn contains_physical_global_copy_terminal_v21(&self, tcx: TyCtxt<'tcx>) -> bool {
        self.collection.functions.iter().any(|function| {
            let body = tcx.instance_mir(function.instance.def);
            body.basic_blocks.iter().any(|block| {
                let TerminatorKind::Call { func, .. } = &block.terminator().kind else {
                    return false;
                };
                let TyKind::FnDef(def_id, _) = *func.ty(body, tcx).kind() else {
                    return false;
                };
                matches!(
                    crate::production_semantic_terminal_v1::classify(tcx, def_id),
                    Some(crate::production_semantic_terminal_v1::ProductionSemanticTerminalRuleV1::Expand(
                        crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyBegin
                    ))
                )
            })
        })
    }

    pub(crate) fn contains_physical_lds_exchange_terminal_v22(&self, tcx: TyCtxt<'tcx>) -> bool {
        self.collection.functions.iter().any(|function| {
            let body = tcx.instance_mir(function.instance.def);
            body.basic_blocks.iter().any(|block| {
                let TerminatorKind::Call { func, .. } = &block.terminator().kind else {
                    return false;
                };
                let TyKind::FnDef(def_id, _) = *func.ty(body, tcx).kind() else {
                    return false;
                };
                matches!(
                    crate::production_semantic_terminal_v1::classify(tcx, def_id),
                    Some(crate::production_semantic_terminal_v1::ProductionSemanticTerminalRuleV1::Expand(
                        crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeBegin
                    ))
                )
            })
        })
    }

    pub(crate) fn function_count(&self) -> usize {
        self.collection.functions.len()
    }

    pub(crate) fn compiler_ffi_observation(
        &self,
    ) -> Option<&fe2o3_compiler_ffi::CompilerFfiEnvelopeV1> {
        self.collection.compiler_ffi_observation.as_ref()
    }

    /// Re-derives typed descriptor roots while the collector-sealed rustc
    /// instances are still live. No retained identity is accepted without
    /// independently repeating the rustc layout extraction.
    pub(crate) fn rederive_typed_descriptor_roots(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> Result<
        Vec<crate::compiler_descriptor::TypedDescriptorRootV1>,
        crate::compiler_descriptor::CompilerDescriptorError,
    > {
        crate::compiler_descriptor::typed_descriptor_roots_from_production_collection(
            tcx,
            &self.collection.functions,
        )
    }
}

impl CollectedFunction<'_> {
    pub(crate) fn is_kernel_entry(&self) -> bool {
        self.role == CollectedFunctionRole::KernelEntry
    }
}

#[derive(Debug)]
enum CollectDecision {
    Collect,
    Forbidden { crate_name: String, fn_path: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectError {
    message: String,
}

impl fmt::Display for CollectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CollectError {}

pub fn count_kernels_in_cgus<'tcx>(tcx: TyCtxt<'tcx>, cgus: &[CodegenUnit<'tcx>]) -> usize {
    registration_candidates(tcx).len()
        + crate::device_ffi::count_exports_in_cgus(tcx, cgus)
            .max(crate::device_ffi::count_local_registration_candidates(tcx))
}

/// Counts every supported production root using HIR/static registration facts
/// only, before rustc monomorphization or MIR collection is entered.
pub(crate) fn count_production_roots_before_monomorphization_v1(tcx: TyCtxt<'_>) -> usize {
    registration_candidates(tcx)
        .len()
        .saturating_add(crate::device_ffi::count_local_registration_candidates(tcx))
}

pub(crate) fn collect_authenticated_kernel_closure_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    verbose: bool,
    target: crate::production_target_v1::RetainedProductionTargetV1,
    context_producers: CapturedContextProducersV1<'tcx>,
) -> Result<AuthenticatedCollectedKernelClosureV1<'tcx>, CollectError> {
    collect_authenticated_kernel_closure_with_enrollment_v1(
        tcx,
        cgus,
        verbose,
        target,
        context_producers,
        None,
    )
}

pub(crate) fn collect_authenticated_kernel_closure_with_enrollment_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    verbose: bool,
    target: crate::production_target_v1::RetainedProductionTargetV1,
    context_producers: CapturedContextProducersV1<'tcx>,
    enrollment: Option<
        &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
    >,
) -> Result<AuthenticatedCollectedKernelClosureV1<'tcx>, CollectError> {
    let (collection, context_entries, closure_flow, terminal_census) =
        collect_device_functions_with_enrollment_v1(
            tcx,
            cgus,
            verbose,
            target.canonical_name().to_owned(),
            context_producers,
            enrollment,
        )?;
    let roots = collection
        .functions
        .iter()
        .filter(|function| {
            matches!(
                function.role,
                CollectedFunctionRole::KernelEntry | CollectedFunctionRole::DeviceFfiExport
            )
        })
        .map(|function| AuthenticatedProductionRootV1 {
            instance: function.instance,
            role: function.role,
            export_name: function.export_name.clone(),
        })
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(CollectError {
            message: "production compilation collected no authenticated external device root"
                .to_owned(),
        });
    }
    Ok(AuthenticatedCollectedKernelClosureV1 {
        target,
        collection,
        roots: roots.into_boxed_slice(),
        context_entries,
        closure_flow,
        terminal_census,
    })
}

fn collect_device_functions<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    verbose: bool,
    target: String,
    context_producers: CapturedContextProducersV1<'tcx>,
) -> Result<
    (
        CollectionResult<'tcx>,
        kernel_context_auth_v1::AuthenticatedContextEntriesV1<'tcx>,
        closure_flow_v1::AuthenticatedClosureFlowV1<'tcx>,
        CollectedTileTerminalCensusV259,
    ),
    CollectError,
> {
    collect_device_functions_with_enrollment_v1(tcx, cgus, verbose, target, context_producers, None)
}

fn collect_device_functions_with_enrollment_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    verbose: bool,
    target: String,
    context_producers: CapturedContextProducersV1<'tcx>,
    enrollment: Option<
        &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
    >,
) -> Result<
    (
        CollectionResult<'tcx>,
        kernel_context_auth_v1::AuthenticatedContextEntriesV1<'tcx>,
        closure_flow_v1::AuthenticatedClosureFlowV1<'tcx>,
        CollectedTileTerminalCensusV259,
    ),
    CollectError,
> {
    let ffi_declarations =
        crate::device_ffi::collect_declarations(tcx, cgus).map_err(|error| CollectError {
            message: error.to_string(),
        })?;
    let ffi_exports = ffi_declarations
        .iter()
        .filter(|declaration| {
            declaration.contract.direction == crate::device_ffi::DeviceFfiDirection::Export
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut collector =
        DeviceCollector::new(tcx, verbose, ffi_declarations, target, context_producers);

    for declaration in ffi_exports {
        if declaration.contract.direction == crate::device_ffi::DeviceFfiDirection::Export {
            if verbose {
                eprintln!(
                    "[collector] standalone device export: {} -> {} ({})",
                    tcx.def_path_str(declaration.instance.def_id()),
                    declaration.contract.symbol,
                    declaration.contract.id.to_hex(),
                );
            }
            collector.add_device_export(declaration.instance, declaration.contract.symbol)?;
        }
    }

    let mut roots = kernel_roots(
        tcx,
        cgus,
        &collector.context_producers.declarations,
        &mut collector.closure_work,
    )
    .map_err(CollectError::from)?;
    let loan = enrollment;
    let enrollment = loan
        .map(|loan| {
            reference_enrollment_v1::bind_v1(
                tcx,
                cgus,
                &mut roots,
                loan,
                &mut collector.closure_work,
            )
            .map_err(|error| CollectError {
                message: error.to_string(),
            })
        })
        .transpose()?;
    for root in roots {
        let instance = root.target;
        let raw_name = tcx.def_path_str(instance.def_id());
        if verbose {
            eprintln!(
                "[collector] root kernel: {raw_name} -> {}",
                root.export_name
            );
        }
        collector.add_root(root)?;
    }

    collector.collect_with_enrollment_v1(enrollment, loan)
}

#[derive(Clone, Debug)]
struct RegistrationRecord<T> {
    registration_path: String,
    item_name: String,
    magic: u64,
    version: u16,
    kind: u16,
    logical_name: String,
    export_name: String,
    crate_binding: Option<CrateBindingIdV1>,
    kernel_binding: Option<KernelBindingIdV1>,
    profile_tag: Option<String>,
    generated_host_contract_identity: Option<GeneratedHostContractIdV3>,
    target_crate_name: String,
    target_symbol: String,
    target_identity: String,
    target: T,
}

#[derive(Clone, Debug)]
struct KernelRoot<T> {
    target: T,
    logical_name: String,
    export_name: String,
    generated_host_contract_identity: Option<GeneratedHostContractIdV3>,
    kernel_binding: Option<KernelBindingIdV1>,
    frontend_contract: Option<AuthenticatedKernelFrontendContractV1>,
    kernel_context_contract: Option<kernel_context_frontend_v1::BoundContextEntryV1>,
    reference_effect_binding:
        Option<crate::reference_effect_v1::AuthenticatedReferenceEffectBindingV1>,
    reference_target: Option<T>,
}

#[derive(Clone, Debug)]
struct FrontendContractRegistrationRecord<T> {
    registration_path: String,
    item_name: String,
    magic: u64,
    version: u16,
    kind: u16,
    logical_name: String,
    canonical_bytes: Vec<u8>,
    contract: KernelFrontendContractV1,
    target_symbol: String,
    target_identity: String,
    target: T,
}

#[derive(Clone, Debug)]
struct ResourceContractRegistrationRecord<T> {
    registration_path: String,
    item_name: String,
    magic: u64,
    version: u16,
    kind: u16,
    logical_name: String,
    canonical_bytes: Vec<u8>,
    contract: KernelResourceContractV1,
    target_symbol: String,
    target_identity: String,
    target: T,
}

#[derive(Clone, Debug)]
struct ReferenceBindingRegistrationRecord<T> {
    registration_path: String,
    item_name: String,
    logical_name: String,
    kernel: T,
    reference: T,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RegistrationError {
    registration_path: String,
    reason: String,
}

impl RegistrationError {
    fn new(registration_path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            registration_path: registration_path.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid fe2o3 kernel registration `{}`: {}",
            self.registration_path, self.reason
        )
    }
}

impl From<RegistrationError> for CollectError {
    fn from(error: RegistrationError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

fn kernel_roots<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    context_declarations: &[kernel_context_frontend_v1::DeclaredContextEntryV1<'tcx>],
    work: &mut crate::rustc_semantic_plan_v1::SourceClosureWorkV1,
) -> Result<Vec<KernelRoot<Instance<'tcx>>>, RegistrationError> {
    let mut functions_by_symbol = BTreeMap::new();

    for cgu in cgus {
        for (item, _data) in cgu.items() {
            let MonoItem::Fn(instance) = item else {
                continue;
            };
            if !is_fully_monomorphized(tcx, *instance) {
                continue;
            }

            let symbol = tcx.symbol_name(*instance).name.to_string();
            functions_by_symbol
                .entry(symbol)
                .or_insert_with(Vec::new)
                .push(*instance);
        }
    }
    for instances in functions_by_symbol.values_mut() {
        instances.sort_by_key(|instance| tcx.def_path_str(instance.def_id()));
    }

    let mut candidates = registration_candidates(tcx);
    candidates.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    let mut records = Vec::with_capacity(candidates.len());
    for (path, item_name, def_id, item) in candidates {
        if !matches!(item.kind, ItemKind::Static(..)) {
            return Err(RegistrationError::new(
                path,
                "the reserved registration name must identify a static item",
            ));
        }
        if tcx.is_mutable_static(def_id.to_def_id()) {
            return Err(RegistrationError::new(
                path,
                "registration statics must be immutable",
            ));
        }

        let flags = tcx.codegen_fn_attrs(def_id).flags;
        if !flags.intersects(CodegenFnAttrFlags::USED_COMPILER | CodegenFnAttrFlags::USED_LINKER) {
            return Err(RegistrationError::new(
                path,
                "registration statics must carry #[used]",
            ));
        }

        records.push(decode_registration_static(
            tcx,
            def_id,
            path,
            item_name,
            &functions_by_symbol,
        )?);
    }

    let session_crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
    let mut roots = validate_registration_records(
        records,
        session_crate_binding(tcx),
        Some(&session_crate_name),
    )?;
    let frontend_records = decode_frontend_contract_registrations(tcx, &functions_by_symbol)?;
    bind_frontend_contract_registrations(tcx, &mut roots, frontend_records)?;
    let resource_records = decode_resource_contract_registrations(tcx, &functions_by_symbol)?;
    bind_resource_contract_registrations(tcx, &mut roots, resource_records)?;
    let reference_records = decode_reference_binding_registrations(tcx)?;
    bind_reference_binding_registrations(tcx, &mut roots, reference_records, work)?;
    kernel_context_frontend_v1::bind_v1(
        tcx,
        &functions_by_symbol,
        &mut roots,
        context_declarations,
    )?;
    Ok(roots)
}

fn general_typed_launch_v3(
    frontend: Option<&AuthenticatedKernelFrontendContractV1>,
    registration_path: &str,
) -> Result<LaunchContract, RegistrationError> {
    const DEFAULT: [u32; 3] = [256, 1, 1];
    const MAX_THREADS: u32 = 256;

    let (dimensions, max_grid) = match frontend.and_then(|frontend| frontend.contract().launch()) {
        None => (DEFAULT, [u32::MAX, 1, 1]),
        Some(launch) => {
            if launch.min_workgroups_per_compute_unit().is_some() {
                return Err(RegistrationError::new(
                    registration_path,
                    "general typed V3 does not support launch occupancy constraints",
                ));
            }
            let Some(required) = launch.required() else {
                return Err(RegistrationError::new(
                    registration_path,
                    "general typed V3 explicit launch requires required dimensions",
                ));
            };
            let required = required.as_array();
            if launch
                .maximum()
                .is_some_and(|maximum| required != maximum.as_array())
            {
                return Err(RegistrationError::new(
                    registration_path,
                    "general typed V3 explicit launch requires identical required and maximum dimensions",
                ));
            }
            if required[0] == 0 || required[0] > MAX_THREADS || required[1] != 1 || required[2] != 1
            {
                return Err(RegistrationError::new(
                    registration_path,
                    "general typed V3 requires exact [N, 1, 1] launch dimensions with N in 1..=256",
                ));
            }
            (
                required,
                launch
                    .max_grid()
                    .map(|dimensions| dimensions.as_array())
                    .unwrap_or([u32::MAX, 1, 1]),
            )
        }
    };
    let resources = frontend
        .and_then(AuthenticatedKernelFrontendContractV1::resource_contract)
        .map(|resource| {
            (
                resource.static_shared_memory_bytes(),
                resource.max_dynamic_shared_memory_bytes(),
            )
        })
        .unwrap_or_default();
    LaunchContract::new(
        1,
        BlockSize::Exact(
            Dimensions::new(dimensions[0], dimensions[1], dimensions[2])
                .map_err(|error| RegistrationError::new(registration_path, error.to_string()))?,
        ),
        Dimensions::new(max_grid[0], max_grid[1], max_grid[2])
            .map_err(|error| RegistrationError::new(registration_path, error.to_string()))?,
        resources.0,
        resources.1,
    )
    .map_err(|error| RegistrationError::new(registration_path, error.to_string()))
}

pub(crate) fn rederive_general_typed_launch_for_descriptor_v1(
    frontend: Option<&AuthenticatedKernelFrontendContractV1>,
    kernel: &str,
) -> Result<LaunchContract, String> {
    general_typed_launch_v3(frontend, kernel).map_err(|error| error.to_string())
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

fn session_crate_binding(tcx: TyCtxt<'_>) -> Option<CrateBindingIdV1> {
    let metadata = &tcx.sess.opts.cg.metadata;
    if metadata.is_empty() {
        return None;
    }
    let crate_name = tcx.crate_name(LOCAL_CRATE);
    Some(derive_crate_binding_id_v1(
        crate_name.as_str(),
        metadata.iter().map(String::as_str),
    ))
}

fn registration_candidates<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> Vec<(
    String,
    String,
    rustc_hir::def_id::LocalDefId,
    &'tcx rustc_hir::Item<'tcx>,
)> {
    tcx.hir_free_items()
        .filter_map(|item_id| {
            let item = tcx.hir_item(item_id);
            let def_id = item.owner_id.def_id;
            let path = tcx.def_path_str(def_id.to_def_id());
            let item_name = final_path_segment(&path).to_string();
            item_name
                .starts_with(reserved_fe2o3_symbols::KERNEL_REGISTRATION_PREFIX)
                .then_some((path, item_name, def_id, item))
        })
        .collect()
}

fn frontend_contract_candidates<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> Vec<(
    String,
    String,
    rustc_hir::def_id::LocalDefId,
    &'tcx rustc_hir::Item<'tcx>,
)> {
    tcx.hir_free_items()
        .filter_map(|item_id| {
            let item = tcx.hir_item(item_id);
            let def_id = item.owner_id.def_id;
            let path = tcx.def_path_str(def_id.to_def_id());
            let item_name = final_path_segment(&path).to_string();
            item_name
                .starts_with(KERNEL_FRONTEND_REGISTRATION_PREFIX_V1)
                .then_some((path, item_name, def_id, item))
        })
        .collect()
}

fn resource_contract_candidates<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> Vec<(
    String,
    String,
    rustc_hir::def_id::LocalDefId,
    &'tcx rustc_hir::Item<'tcx>,
)> {
    tcx.hir_free_items()
        .filter_map(|item_id| {
            let item = tcx.hir_item(item_id);
            let def_id = item.owner_id.def_id;
            let path = tcx.def_path_str(def_id.to_def_id());
            let item_name = final_path_segment(&path).to_string();
            item_name
                .starts_with(KERNEL_RESOURCE_REGISTRATION_PREFIX_V1)
                .then_some((path, item_name, def_id, item))
        })
        .collect()
}

fn reference_binding_candidates<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> Vec<(
    String,
    String,
    rustc_hir::def_id::LocalDefId,
    &'tcx rustc_hir::Item<'tcx>,
)> {
    tcx.hir_free_items()
        .filter_map(|item_id| {
            let item = tcx.hir_item(item_id);
            let def_id = item.owner_id.def_id;
            let path = tcx.def_path_str(def_id.to_def_id());
            let item_name = final_path_segment(&path).to_string();
            item_name
                .starts_with(reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_PREFIX_V1)
                .then_some((path, item_name, def_id, item))
        })
        .collect()
}

fn decode_reference_binding_registrations<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> Result<Vec<ReferenceBindingRegistrationRecord<Instance<'tcx>>>, RegistrationError> {
    let mut candidates = reference_binding_candidates(tcx);
    candidates.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    candidates
        .into_iter()
        .map(|(path, item_name, def_id, item)| {
            decode_reference_binding_registration(tcx, def_id, path, item_name, item)
        })
        .collect()
}

fn decode_reference_binding_registration<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: rustc_hir::def_id::LocalDefId,
    registration_path: String,
    item_name: String,
    item: &rustc_hir::Item<'tcx>,
) -> Result<ReferenceBindingRegistrationRecord<Instance<'tcx>>, RegistrationError> {
    if !matches!(item.kind, ItemKind::Static(..)) {
        return Err(RegistrationError::new(
            registration_path,
            "reserved reference-binding name must identify a static item",
        ));
    }
    if tcx.is_mutable_static(def_id.to_def_id()) {
        return Err(RegistrationError::new(
            registration_path,
            "reference-binding static must be immutable",
        ));
    }
    let flags = tcx.codegen_fn_attrs(def_id).flags;
    if !flags.intersects(CodegenFnAttrFlags::USED_COMPILER | CodegenFnAttrFlags::USED_LINKER) {
        return Err(RegistrationError::new(
            registration_path,
            "reference-binding static must carry #[used]",
        ));
    }
    let registration_ty = tcx
        .try_normalize_erasing_regions(
            TypingEnv::fully_monomorphized(),
            tcx.type_of(def_id).instantiate_identity(),
        )
        .map_err(|_| {
            RegistrationError::new(
                &registration_path,
                "reference-binding type did not normalize",
            )
        })?;
    let TyKind::Tuple(types) = registration_ty.kind() else {
        return Err(RegistrationError::new(
            registration_path,
            "reference binding must use the exact V1 tuple type",
        ));
    };
    let exact_type = types.len()
        == reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_FIELD_COUNT_V1
        && types[0] == tcx.types.u64
        && types[1] == tcx.types.u16
        && types[2] == tcx.types.u16
        && is_shared_str(types[3])
        && matches!(types[4].kind(), TyKind::FnPtr(..))
        && matches!(types[5].kind(), TyKind::FnPtr(..));
    if !exact_type {
        return Err(RegistrationError::new(
            registration_path,
            "reference binding must be (u64, u16, u16, &str, kernel fn pointer, fn())",
        ));
    }
    let body = tcx.mir_for_ctfe(def_id);
    let fields = registration_tuple_fields(
        body,
        reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_FIELD_COUNT_V1,
        &registration_path,
    )?;
    let magic = registration_integer(tcx, fields[0], tcx.types.u64, "magic", &registration_path)?;
    let version =
        registration_integer(tcx, fields[1], tcx.types.u16, "version", &registration_path)?;
    let kind = registration_integer(tcx, fields[2], tcx.types.u16, "kind", &registration_path)?;
    if magic != u128::from(reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_MAGIC_V1)
        || version != u128::from(reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_VERSION_V1)
        || kind != u128::from(reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_KIND_V1)
    {
        return Err(RegistrationError::new(
            registration_path,
            "reference-binding magic, version, or kind is not canonical V1",
        ));
    }
    let logical_name = registration_string(tcx, fields[3], "logical name", &registration_path)?;
    let kernel = registration_target(tcx, body, fields[4], &registration_path)?;
    let anchor = registration_target(tcx, body, fields[5], &registration_path)?;
    let reference = reference_target_from_anchor_v1(tcx, anchor, &registration_path)?;
    Ok(ReferenceBindingRegistrationRecord {
        registration_path,
        item_name,
        logical_name,
        kernel,
        reference,
    })
}

fn reference_target_from_anchor_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    anchor: Instance<'tcx>,
    registration_path: &str,
) -> Result<Instance<'tcx>, RegistrationError> {
    if anchor.def_id().as_local().is_none() {
        return Err(RegistrationError::new(
            registration_path,
            "reference anchor must be a local generated function",
        ));
    };
    let expected_prefix = "__fe2o3_kernel_reference_anchor_v1_";
    let anchor_path = tcx.def_path_str(anchor.def_id());
    if !final_path_segment(&anchor_path).starts_with(expected_prefix) {
        return Err(RegistrationError::new(
            registration_path,
            "reference anchor does not use the compiler-reserved generated name",
        ));
    }
    let body = tcx.instance_mir(anchor.def);
    let mut candidates = body
        .local_decls
        .iter()
        .filter_map(|declaration| {
            let TyKind::FnDef(def_id, arguments) = declaration.ty.kind() else {
                return None;
            };
            Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), *def_id, arguments)
                .ok()
                .flatten()
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|instance| {
        (
            tcx.def_path_hash(instance.def_id()).0,
            tcx.symbol_name(*instance).name.to_string(),
        )
    });
    candidates.dedup();
    let [reference] = candidates.as_slice() else {
        return Err(RegistrationError::new(
            registration_path,
            format!(
                "reference anchor must name exactly one resolvable function item; found {}",
                candidates.len(),
            ),
        ));
    };
    if !is_fully_monomorphized(tcx, *reference) {
        return Err(RegistrationError::new(
            registration_path,
            "reference function is not fully monomorphized",
        ));
    }
    Ok(*reference)
}

fn bind_reference_binding_registrations<'tcx>(
    tcx: TyCtxt<'tcx>,
    roots: &mut [KernelRoot<Instance<'tcx>>],
    records: Vec<ReferenceBindingRegistrationRecord<Instance<'tcx>>>,
    work: &mut crate::rustc_semantic_plan_v1::SourceClosureWorkV1,
) -> Result<(), RegistrationError> {
    let roots_by_name = reference_registration_maps_v1(roots, &records, work)?;
    for record in records {
        let (index, record) = reference_registration_root_v1(&roots_by_name, record, work)?;
        let root = &mut roots[index];
        charge_reference_registration_work_v1(work, Some(1))?;
        if root.target != record.kernel {
            return Err(RegistrationError::new(
                record.registration_path,
                "safe Rust reference binding does not point at the exact registered kernel instance",
            ));
        }
        charge_reference_registration_work_v1(work, record.registration_path.len().checked_mul(2))?;
        let binding = crate::reference_effect_v1::authenticate_reference_binding_v1(
            tcx,
            record.registration_path.clone(),
            record.logical_name,
            record.kernel,
            record.reference,
            work,
        )
        .map_err(|error| RegistrationError::new(record.registration_path, error.to_string()))?;
        root.reference_effect_binding = Some(binding);
        root.reference_target = Some(record.reference);
    }
    Ok(())
}

fn reference_registration_work_overflow_v1() -> RegistrationError {
    RegistrationError::new(
        "<reference binding work>",
        "reference registration work overflow",
    )
}

fn charge_reference_registration_work_v1(
    work: &mut crate::rustc_semantic_plan_v1::SourceClosureWorkV1,
    amount: Option<usize>,
) -> Result<(), RegistrationError> {
    let amount = amount.ok_or_else(reference_registration_work_overflow_v1)?;
    work.charge(amount).map_err(|error| {
        // A failed debit must not clone an unbounded registration path.
        RegistrationError::new("<reference binding work>", error.to_string())
    })
}

fn reference_registration_search_work_v1(entries: usize, key_bytes: usize) -> Option<usize> {
    if entries == 0 {
        return Some(0);
    }
    // The pinned nightly-2026-04-03 std BTreeMap searches at most 11 keys per
    // node, with at most bit_length(entries) levels. No key is visited twice.
    // Recheck this envelope when changing the pinned standard library.
    let levels = usize::try_from(usize::BITS.checked_sub(entries.leading_zeros())?).ok()?;
    let comparisons = entries.min(levels.checked_mul(11)?);
    comparisons.checked_mul(key_bytes.checked_add(1)?)
}

fn reference_registration_maps_v1<T>(
    roots: &[KernelRoot<T>],
    records: &[ReferenceBindingRegistrationRecord<T>],
    work: &mut crate::rustc_semantic_plan_v1::SourceClosureWorkV1,
) -> Result<BTreeMap<String, usize>, RegistrationError> {
    // Debits cover logical entry payloads, copied string bytes and comparison
    // work, not std allocator pages/node slack or rustc query internals. Each
    // string allocation and its copy are counted separately before cloning.
    charge_reference_registration_work_v1(work, Some(1))?;
    let mut roots_by_name = BTreeMap::new();
    for (index, root) in roots.iter().enumerate() {
        let amount = root
            .logical_name
            .len()
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<(String, usize)>()))
            .and_then(|bytes| bytes.checked_add(1))
            .and_then(|bytes| {
                bytes.checked_add(reference_registration_search_work_v1(
                    roots_by_name.len(),
                    root.logical_name.len(),
                )?)
            });
        charge_reference_registration_work_v1(work, amount)?;
        roots_by_name.insert(root.logical_name.clone(), index);
    }
    let mut binding_counts = BTreeMap::new();
    for record in records {
        let amount = std::mem::size_of::<(&str, usize)>()
            .checked_add(1)
            .and_then(|bytes| {
                bytes.checked_add(reference_registration_search_work_v1(
                    binding_counts.len(),
                    record.logical_name.len(),
                )?)
            });
        charge_reference_registration_work_v1(work, amount)?;
        let count = binding_counts
            .entry(record.logical_name.as_str())
            .or_insert(0usize);
        *count = count
            .checked_add(1)
            .ok_or_else(reference_registration_work_overflow_v1)?;
    }
    charge_reference_registration_work_v1(work, Some(binding_counts.len()))?;
    if let Some((logical_name, _)) = binding_counts.iter().find(|(_, count)| **count > 1) {
        for record in records {
            charge_reference_registration_work_v1(
                work,
                record
                    .logical_name
                    .len()
                    .min(logical_name.len())
                    .checked_add(1),
            )?;
            if record.logical_name == *logical_name {
                charge_reference_registration_work_v1(
                    work,
                    record.registration_path.len().checked_mul(2),
                )?;
                return Err(RegistrationError::new(
                    &record.registration_path,
                    "duplicate safe Rust reference binding for one kernel",
                ));
            }
        }
        unreachable!("duplicate count came from one binding record");
    }
    Ok(roots_by_name)
}

fn reference_registration_root_v1<T>(
    roots_by_name: &BTreeMap<String, usize>,
    record: ReferenceBindingRegistrationRecord<T>,
    work: &mut crate::rustc_semantic_plan_v1::SourceClosureWorkV1,
) -> Result<(usize, ReferenceBindingRegistrationRecord<T>), RegistrationError> {
    let prefix = reserved_fe2o3_symbols::REFERENCE_BINDING_REGISTRATION_PREFIX_V1;
    let amount = prefix
        .len()
        .checked_add(record.logical_name.len())
        .and_then(|bytes| {
            bytes
                .checked_mul(2)
                .and_then(|payload| payload.checked_add(bytes.min(record.item_name.len())))
                .and_then(|payload| payload.checked_add(1))
        });
    charge_reference_registration_work_v1(work, amount)?;
    let expected_name = format!("{prefix}{}", record.logical_name);
    if record.item_name != expected_name {
        return Err(RegistrationError::new(
            record.registration_path,
            "reference-binding item name disagrees with its logical kernel name",
        ));
    }
    charge_reference_registration_work_v1(
        work,
        reference_registration_search_work_v1(roots_by_name.len(), record.logical_name.len()),
    )?;
    let Some(&index) = roots_by_name.get(&record.logical_name) else {
        return Err(RegistrationError::new(
            record.registration_path,
            "orphan safe Rust reference binding has no registered kernel",
        ));
    };
    Ok((index, record))
}

#[cfg(test)]
#[path = "collector/reference_registration_work_v1_tests.rs"]
mod reference_registration_work_v1_tests;

fn decode_frontend_contract_registrations<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions_by_symbol: &BTreeMap<String, Vec<Instance<'tcx>>>,
) -> Result<Vec<FrontendContractRegistrationRecord<Instance<'tcx>>>, RegistrationError> {
    let mut candidates = frontend_contract_candidates(tcx);
    candidates.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    candidates
        .into_iter()
        .map(|(path, item_name, def_id, item)| {
            decode_frontend_contract_registration(
                tcx,
                def_id,
                path,
                item_name,
                item,
                functions_by_symbol,
            )
        })
        .collect()
}

fn decode_frontend_contract_registration<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: rustc_hir::def_id::LocalDefId,
    registration_path: String,
    item_name: String,
    item: &rustc_hir::Item<'tcx>,
    functions_by_symbol: &BTreeMap<String, Vec<Instance<'tcx>>>,
) -> Result<FrontendContractRegistrationRecord<Instance<'tcx>>, RegistrationError> {
    if !matches!(item.kind, ItemKind::Static(..)) {
        return Err(RegistrationError::new(
            registration_path,
            "the reserved frontend-contract name must identify a static item",
        ));
    }
    if tcx.is_mutable_static(def_id.to_def_id()) {
        return Err(RegistrationError::new(
            registration_path,
            "frontend-contract registration statics must be immutable",
        ));
    }
    let flags = tcx.codegen_fn_attrs(def_id).flags;
    if !flags.intersects(CodegenFnAttrFlags::USED_COMPILER | CodegenFnAttrFlags::USED_LINKER) {
        return Err(RegistrationError::new(
            registration_path,
            "frontend-contract registration statics must carry #[used]",
        ));
    }

    let registration_ty = tcx.type_of(def_id).instantiate_identity();
    let TyKind::Tuple(field_types) = registration_ty.kind() else {
        return Err(RegistrationError::new(
            registration_path,
            "frontend-contract registration must use the exact V1 tuple type",
        ));
    };
    let exact_type = field_types.len() == 6
        && field_types[0] == tcx.types.u64
        && field_types[1] == tcx.types.u16
        && field_types[2] == tcx.types.u16
        && is_shared_str(field_types[3])
        && is_shared_u8_slice(field_types[4])
        && matches!(field_types[5].kind(), TyKind::FnPtr(..));
    if !exact_type {
        return Err(RegistrationError::new(
            registration_path,
            "frontend-contract registration type must be `(u64, u16, u16, &str, &[u8], fn pointer)`",
        ));
    }

    let body = tcx.mir_for_ctfe(def_id);
    let fields = registration_tuple_fields(body, 6, &registration_path)?;
    let magic = registration_integer(tcx, fields[0], tcx.types.u64, "magic", &registration_path)?;
    let version =
        registration_integer(tcx, fields[1], tcx.types.u16, "version", &registration_path)?;
    let kind = registration_integer(tcx, fields[2], tcx.types.u16, "kind", &registration_path)?;
    let logical_name = registration_string(tcx, fields[3], "logical name", &registration_path)?;
    let canonical_bytes = registration_bytes(tcx, fields[4], "contract", &registration_path)?;
    let contract = decode_kernel_frontend_contract_v1(&canonical_bytes).map_err(|error| {
        RegistrationError::new(
            &registration_path,
            format!("frontend-contract bytes are invalid: {error}"),
        )
    })?;
    let target = registration_target(tcx, body, fields[5], &registration_path)?;
    let target_symbol = tcx.symbol_name(target).name.to_string();
    let target_identity = tcx.def_path_str(target.def_id());
    let Some(cgu_targets) = functions_by_symbol.get(&target_symbol) else {
        return Err(RegistrationError::new(
            registration_path,
            format!(
                "frontend-contract target `{target_symbol}` was not monomorphized into a codegen unit"
            ),
        ));
    };
    if cgu_targets.as_slice() != [target] {
        return Err(RegistrationError::new(
            registration_path,
            format!("frontend-contract target `{target_symbol}` is ambiguous or inconsistent"),
        ));
    }

    Ok(FrontendContractRegistrationRecord {
        registration_path,
        item_name,
        magic: u64::try_from(magic)
            .map_err(|_| RegistrationError::new("frontend contract", "magic does not fit u64"))?,
        version: u16::try_from(version)
            .map_err(|_| RegistrationError::new("frontend contract", "version does not fit u16"))?,
        kind: u16::try_from(kind)
            .map_err(|_| RegistrationError::new("frontend contract", "kind does not fit u16"))?,
        logical_name,
        canonical_bytes,
        contract,
        target_symbol,
        target_identity,
        target,
    })
}

fn bind_frontend_contract_registrations<'tcx>(
    tcx: TyCtxt<'tcx>,
    roots: &mut [KernelRoot<Instance<'tcx>>],
    mut records: Vec<FrontendContractRegistrationRecord<Instance<'tcx>>>,
) -> Result<(), RegistrationError> {
    records.sort_by(|lhs, rhs| lhs.registration_path.cmp(&rhs.registration_path));
    let roots_by_name = roots
        .iter()
        .enumerate()
        .map(|(index, root)| (root.logical_name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut targets = BTreeMap::new();

    for record in records {
        let error = |reason| RegistrationError::new(record.registration_path.clone(), reason);
        if record.magic != KERNEL_FRONTEND_REGISTRATION_MAGIC_V1 {
            return Err(error(format!(
                "frontend-contract magic {:#018x} does not match {:#018x}",
                record.magic, KERNEL_FRONTEND_REGISTRATION_MAGIC_V1
            )));
        }
        if record.version != KERNEL_FRONTEND_REGISTRATION_VERSION_V1 {
            return Err(error(format!(
                "unknown frontend-contract registration version {}",
                record.version
            )));
        }
        if record.kind != KERNEL_FRONTEND_REGISTRATION_KIND_V1 {
            return Err(error(format!(
                "unknown frontend-contract registration kind {}",
                record.kind
            )));
        }
        if record.logical_name.is_empty() {
            return Err(error(
                "frontend-contract logical name must not be empty".to_owned(),
            ));
        }
        let expected_item_name = format!(
            "{KERNEL_FRONTEND_REGISTRATION_PREFIX_V1}{}",
            record.logical_name
        );
        if record.item_name != expected_item_name {
            return Err(error(format!(
                "frontend-contract item name `{}` is inconsistent with logical name `{}`",
                record.item_name, record.logical_name
            )));
        }
        let Some(&root_index) = roots_by_name.get(&record.logical_name) else {
            return Err(error(format!(
                "orphan frontend contract has no registered kernel `{}`",
                record.logical_name
            )));
        };
        let root = &mut roots[root_index];
        if root.target != record.target {
            return Err(error(format!(
                "frontend-contract target `{}` is not the exact registered kernel function",
                record.target_identity
            )));
        }
        if root.frontend_contract.is_some() {
            return Err(error(format!(
                "duplicate frontend contract for kernel `{}`",
                record.logical_name
            )));
        }
        if let Some(previous) = targets.insert(
            record.target_identity.clone(),
            record.registration_path.clone(),
        ) {
            return Err(error(format!(
                "duplicate frontend-contract target `{}`; first registered by `{previous}`",
                record.target_identity
            )));
        }
        if record.contract.unsafe_assembly().is_some()
            && tcx.fn_sig(record.target.def_id()).skip_binder().safety() != Safety::Unsafe
        {
            return Err(error(
                "unsafe-assembly contracts require an unsafe registered kernel function".to_owned(),
            ));
        }

        root.frontend_contract = Some(AuthenticatedKernelFrontendContractV1 {
            registration_path: record.registration_path,
            target_def_path_hash: tcx.def_path_hash(record.target.def_id()).0.to_le_bytes(),
            target_symbol: record.target_symbol,
            canonical_bytes: record.canonical_bytes,
            contract: record.contract,
            resource: None,
            reachable_assembly: ReachableAssemblySummaryV1::default(),
        });
    }
    Ok(())
}

fn decode_resource_contract_registrations<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions_by_symbol: &BTreeMap<String, Vec<Instance<'tcx>>>,
) -> Result<Vec<ResourceContractRegistrationRecord<Instance<'tcx>>>, RegistrationError> {
    let mut candidates = resource_contract_candidates(tcx);
    candidates.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    candidates
        .into_iter()
        .map(|(path, item_name, def_id, item)| {
            decode_resource_contract_registration(
                tcx,
                def_id,
                path,
                item_name,
                item,
                functions_by_symbol,
            )
        })
        .collect()
}

fn decode_resource_contract_registration<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: rustc_hir::def_id::LocalDefId,
    registration_path: String,
    item_name: String,
    item: &rustc_hir::Item<'tcx>,
    functions_by_symbol: &BTreeMap<String, Vec<Instance<'tcx>>>,
) -> Result<ResourceContractRegistrationRecord<Instance<'tcx>>, RegistrationError> {
    if !matches!(item.kind, ItemKind::Static(..)) {
        return Err(RegistrationError::new(
            registration_path,
            "the reserved resource-contract name must identify a static item",
        ));
    }
    if tcx.is_mutable_static(def_id.to_def_id()) {
        return Err(RegistrationError::new(
            registration_path,
            "resource-contract registration statics must be immutable",
        ));
    }
    let flags = tcx.codegen_fn_attrs(def_id).flags;
    if !flags.intersects(CodegenFnAttrFlags::USED_COMPILER | CodegenFnAttrFlags::USED_LINKER) {
        return Err(RegistrationError::new(
            registration_path,
            "resource-contract registration statics must carry #[used]",
        ));
    }

    let registration_ty = tcx.type_of(def_id).instantiate_identity();
    let TyKind::Tuple(field_types) = registration_ty.kind() else {
        return Err(RegistrationError::new(
            registration_path,
            "resource-contract registration must use the exact V1 tuple type",
        ));
    };
    let exact_type = field_types.len() == 6
        && field_types[0] == tcx.types.u64
        && field_types[1] == tcx.types.u16
        && field_types[2] == tcx.types.u16
        && is_shared_str(field_types[3])
        && is_shared_u8_slice(field_types[4])
        && matches!(field_types[5].kind(), TyKind::FnPtr(..));
    if !exact_type {
        return Err(RegistrationError::new(
            registration_path,
            "resource-contract registration type must be `(u64, u16, u16, &str, &[u8], fn pointer)`",
        ));
    }

    let body = tcx.mir_for_ctfe(def_id);
    let fields = registration_tuple_fields(body, 6, &registration_path)?;
    let magic = registration_integer(tcx, fields[0], tcx.types.u64, "magic", &registration_path)?;
    let version =
        registration_integer(tcx, fields[1], tcx.types.u16, "version", &registration_path)?;
    let kind = registration_integer(tcx, fields[2], tcx.types.u16, "kind", &registration_path)?;
    let logical_name = registration_string(tcx, fields[3], "logical name", &registration_path)?;
    let canonical_bytes =
        registration_bytes(tcx, fields[4], "resource contract", &registration_path)?;
    let contract = decode_kernel_resource_contract_v1(&canonical_bytes).map_err(|error| {
        RegistrationError::new(
            &registration_path,
            format!("resource-contract bytes are invalid: {error}"),
        )
    })?;
    let target = registration_target(tcx, body, fields[5], &registration_path)?;
    let target_symbol = tcx.symbol_name(target).name.to_string();
    let target_identity = tcx.def_path_str(target.def_id());
    let Some(cgu_targets) = functions_by_symbol.get(&target_symbol) else {
        return Err(RegistrationError::new(
            registration_path,
            format!(
                "resource-contract target `{target_symbol}` was not monomorphized into a codegen unit"
            ),
        ));
    };
    if cgu_targets.as_slice() != [target] {
        return Err(RegistrationError::new(
            registration_path,
            format!("resource-contract target `{target_symbol}` is ambiguous or inconsistent"),
        ));
    }

    Ok(ResourceContractRegistrationRecord {
        registration_path,
        item_name,
        magic: u64::try_from(magic)
            .map_err(|_| RegistrationError::new("resource contract", "magic does not fit u64"))?,
        version: u16::try_from(version)
            .map_err(|_| RegistrationError::new("resource contract", "version does not fit u16"))?,
        kind: u16::try_from(kind)
            .map_err(|_| RegistrationError::new("resource contract", "kind does not fit u16"))?,
        logical_name,
        canonical_bytes,
        contract,
        target_symbol,
        target_identity,
        target,
    })
}

fn bind_resource_contract_registrations<'tcx>(
    tcx: TyCtxt<'tcx>,
    roots: &mut [KernelRoot<Instance<'tcx>>],
    mut records: Vec<ResourceContractRegistrationRecord<Instance<'tcx>>>,
) -> Result<(), RegistrationError> {
    records.sort_by(|lhs, rhs| lhs.registration_path.cmp(&rhs.registration_path));
    let roots_by_name = roots
        .iter()
        .enumerate()
        .map(|(index, root)| (root.logical_name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut targets = BTreeMap::new();

    for record in records {
        let error = |reason| RegistrationError::new(record.registration_path.clone(), reason);
        if record.magic != KERNEL_RESOURCE_REGISTRATION_MAGIC_V1 {
            return Err(error(format!(
                "resource-contract magic {:#018x} does not match {:#018x}",
                record.magic, KERNEL_RESOURCE_REGISTRATION_MAGIC_V1
            )));
        }
        if record.version != KERNEL_RESOURCE_REGISTRATION_VERSION_V1 {
            return Err(error(format!(
                "unknown resource-contract registration version {}",
                record.version
            )));
        }
        if record.kind != KERNEL_RESOURCE_REGISTRATION_KIND_V1 {
            return Err(error(format!(
                "unknown resource-contract registration kind {}",
                record.kind
            )));
        }
        if record.logical_name.is_empty() {
            return Err(error(
                "resource-contract logical name must not be empty".to_owned(),
            ));
        }
        let expected_item_name = format!(
            "{KERNEL_RESOURCE_REGISTRATION_PREFIX_V1}{}",
            record.logical_name
        );
        if record.item_name != expected_item_name {
            return Err(error(format!(
                "resource-contract item name `{}` is inconsistent with logical name `{}`",
                record.item_name, record.logical_name
            )));
        }
        let Some(&root_index) = roots_by_name.get(&record.logical_name) else {
            return Err(error(format!(
                "orphan resource contract has no registered kernel `{}`",
                record.logical_name
            )));
        };
        let root = &mut roots[root_index];
        if root.target != record.target {
            return Err(error(format!(
                "resource-contract target `{}` is not the exact registered kernel function",
                record.target_identity
            )));
        }
        let Some(frontend) = root.frontend_contract.as_mut() else {
            return Err(error(
                "resource contract requires an authenticated frontend launch contract".to_owned(),
            ));
        };
        if frontend.resource.is_some() {
            return Err(error(format!(
                "duplicate resource contract for kernel `{}`",
                record.logical_name
            )));
        }
        if frontend.target_symbol != record.target_symbol
            || frontend.target_def_path_hash
                != tcx.def_path_hash(record.target.def_id()).0.to_le_bytes()
        {
            return Err(error(
                "resource contract disagrees with the authenticated frontend target".to_owned(),
            ));
        }
        if let Some(previous) = targets.insert(
            record.target_identity.clone(),
            record.registration_path.clone(),
        ) {
            return Err(error(format!(
                "duplicate resource-contract target `{}`; first registered by `{previous}`",
                record.target_identity
            )));
        }
        frontend.resource = Some(AuthenticatedKernelResourceContractV1 {
            canonical_bytes: record.canonical_bytes,
            contract: record.contract,
        });
    }
    Ok(())
}

fn decode_registration_static<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: rustc_hir::def_id::LocalDefId,
    registration_path: String,
    item_name: String,
    functions_by_symbol: &BTreeMap<String, Vec<Instance<'tcx>>>,
) -> Result<RegistrationRecord<Instance<'tcx>>, RegistrationError> {
    let registration_ty = tcx
        .try_normalize_erasing_regions(
            TypingEnv::fully_monomorphized(),
            tcx.type_of(def_id).instantiate_identity(),
        )
        .map_err(|_| {
            RegistrationError::new(&registration_path, "registration type did not normalize")
        })?;
    let TyKind::Tuple(fields) = registration_ty.kind() else {
        return Err(RegistrationError::new(
            registration_path,
            "registration must use an exact V1, V2, or V3 tuple type",
        ));
    };

    let is_v1 = fields.len() == reserved_fe2o3_symbols::KERNEL_REGISTRATION_V1_FIELD_COUNT
        && fields[0] == tcx.types.u64
        && fields[1] == tcx.types.u16
        && fields[2] == tcx.types.u16
        && is_shared_str(fields[3])
        && is_shared_str(fields[4])
        && matches!(fields[5].kind(), TyKind::FnPtr(..));
    let is_v2 = fields.len() == reserved_fe2o3_symbols::KERNEL_REGISTRATION_V2_FIELD_COUNT
        && fields[0] == tcx.types.u64
        && fields[1] == tcx.types.u16
        && fields[2] == tcx.types.u16
        && is_shared_str(fields[3])
        && is_shared_str(fields[4])
        && is_shared_str(fields[5])
        && is_shared_str(fields[6])
        && matches!(fields[7].kind(), TyKind::FnPtr(..));
    let is_v3 = fields.len() == reserved_fe2o3_symbols::KERNEL_REGISTRATION_V3_FIELD_COUNT
        && fields[0] == tcx.types.u64
        && fields[1] == tcx.types.u16
        && fields[2] == tcx.types.u16
        && is_shared_str(fields[3])
        && is_shared_str(fields[4])
        && is_shared_str(fields[5])
        && is_shared_str(fields[6])
        && is_shared_str(fields[7])
        && is_shared_str(fields[8])
        && matches!(fields[9].kind(), TyKind::FnPtr(..));
    if !is_v1 && !is_v2 && !is_v3 {
        return Err(RegistrationError::new(
            registration_path,
            "registration type must be the exact V1, V2, or 10-field V3 tuple",
        ));
    }

    let body = tcx.mir_for_ctfe(def_id);
    let mut aggregate = None;
    for block in body.basic_blocks.iter() {
        for statement in &block.statements {
            let Some((place, Rvalue::Aggregate(kind, fields))) = statement.kind.as_assign() else {
                continue;
            };
            if place.as_local() != Some(RETURN_PLACE) || !matches!(**kind, AggregateKind::Tuple) {
                continue;
            }
            if aggregate.replace(fields).is_some() {
                return Err(RegistrationError::new(
                    registration_path,
                    "registration initializer must contain exactly one tuple value",
                ));
            }
        }
    }
    let fields = aggregate.ok_or_else(|| {
        RegistrationError::new(
            registration_path.clone(),
            "registration initializer does not contain the required tuple value",
        )
    })?;
    let expected_fields = if is_v3 {
        reserved_fe2o3_symbols::KERNEL_REGISTRATION_V3_FIELD_COUNT
    } else if is_v2 {
        reserved_fe2o3_symbols::KERNEL_REGISTRATION_V2_FIELD_COUNT
    } else {
        reserved_fe2o3_symbols::KERNEL_REGISTRATION_V1_FIELD_COUNT
    };
    if fields.len() != expected_fields {
        return Err(RegistrationError::new(
            registration_path,
            "registration initializer has the wrong field count",
        ));
    }
    let fields = fields.iter().collect::<Vec<_>>();

    let magic = registration_integer(tcx, fields[0], tcx.types.u64, "magic", &registration_path)?;
    let version =
        registration_integer(tcx, fields[1], tcx.types.u16, "version", &registration_path)?;
    let kind = registration_integer(tcx, fields[2], tcx.types.u16, "kind", &registration_path)?;
    let logical_name = registration_string(tcx, fields[3], "logical name", &registration_path)?;
    let export_name = registration_string(tcx, fields[4], "export name", &registration_path)?;
    let crate_binding = if is_v2 || is_v3 {
        let value = registration_string(tcx, fields[5], "crate binding", &registration_path)?;
        Some(CrateBindingIdV1::from_hex(&value).map_err(|error| {
            RegistrationError::new(
                &registration_path,
                format!("invalid crate binding: {error}"),
            )
        })?)
    } else {
        None
    };
    let kernel_binding = if is_v2 || is_v3 {
        let value = registration_string(tcx, fields[6], "kernel binding", &registration_path)?;
        Some(KernelBindingIdV1::from_hex(&value).map_err(|error| {
            RegistrationError::new(
                &registration_path,
                format!("invalid kernel binding: {error}"),
            )
        })?)
    } else {
        None
    };
    let profile_tag = if is_v3 {
        Some(registration_string(
            tcx,
            fields[7],
            "profile tag",
            &registration_path,
        )?)
    } else {
        None
    };
    let generated_host_contract_identity = if is_v3 {
        let value = registration_string(
            tcx,
            fields[8],
            "generated host-contract identity",
            &registration_path,
        )?;
        Some(
            GeneratedHostContractIdV3::from_hex(&value).map_err(|error| {
                RegistrationError::new(
                    &registration_path,
                    format!("invalid generated host-contract identity: {error}"),
                )
            })?,
        )
    } else {
        None
    };
    let target_index = if is_v3 {
        9
    } else if is_v2 {
        7
    } else {
        5
    };
    let target = registration_target(tcx, body, fields[target_index], &registration_path)?;
    let target_crate_name = tcx.crate_name(target.def_id().krate).to_string();
    let target_symbol = tcx.symbol_name(target).name.to_string();
    let target_identity = tcx.def_path_str(target.def_id());
    match functions_by_symbol.get(&target_symbol) {
        None if !target.def_id().is_local() && tcx.is_mir_available(target.def_id()) => {}
        None => {
            return Err(RegistrationError::new(
                registration_path,
                format!(
                    "registered target `{target_symbol}` was not monomorphized into a codegen unit"
                ),
            ));
        }
        Some(cgu_targets) if cgu_targets.as_slice() == [target] => {}
        Some(cgu_targets) => {
            let paths = cgu_targets
                .iter()
                .map(|instance| tcx.def_path_str(instance.def_id()))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(RegistrationError::new(
                registration_path,
                format!(
                    "registered target symbol `{target_symbol}` is ambiguous or inconsistent across: {paths}"
                ),
            ));
        }
    }
    let magic = u64::try_from(magic)
        .map_err(|_| RegistrationError::new(&registration_path, "magic does not fit u64"))?;
    let version = u16::try_from(version)
        .map_err(|_| RegistrationError::new(&registration_path, "version does not fit u16"))?;
    let kind = u16::try_from(kind)
        .map_err(|_| RegistrationError::new(&registration_path, "kind does not fit u16"))?;
    if version == reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V1 && !is_v1 {
        return Err(RegistrationError::new(
            &registration_path,
            "registration version 1 requires the exact V1 tuple type",
        ));
    }
    if version == reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V2 && !is_v2 {
        return Err(RegistrationError::new(
            &registration_path,
            "registration version 2 requires the exact V2 tuple type",
        ));
    }
    if version == reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V3 && !is_v3 {
        return Err(RegistrationError::new(
            &registration_path,
            "registration version 3 requires the exact V3 tuple type",
        ));
    }

    Ok(RegistrationRecord {
        registration_path,
        item_name,
        magic,
        version,
        kind,
        logical_name,
        export_name,
        crate_binding,
        kernel_binding,
        profile_tag,
        generated_host_contract_identity,
        target_crate_name,
        target_symbol,
        target_identity,
        target,
    })
}

fn registration_integer<'tcx>(
    tcx: TyCtxt<'tcx>,
    operand: &Operand<'tcx>,
    expected_ty: rustc_middle::ty::Ty<'tcx>,
    field: &str,
    registration_path: &str,
) -> Result<u128, RegistrationError> {
    let Operand::Constant(constant) = operand else {
        return Err(RegistrationError::new(
            registration_path,
            format!("V1 {field} field must be a constant"),
        ));
    };
    if constant.const_.ty() != expected_ty {
        return Err(RegistrationError::new(
            registration_path,
            format!("V1 {field} field has the wrong type"),
        ));
    }
    constant
        .const_
        .try_eval_bits(tcx, TypingEnv::fully_monomorphized())
        .ok_or_else(|| {
            RegistrationError::new(
                registration_path,
                format!("V1 {field} field could not be evaluated"),
            )
        })
}

fn registration_string<'tcx>(
    tcx: TyCtxt<'tcx>,
    operand: &Operand<'tcx>,
    field: &str,
    registration_path: &str,
) -> Result<String, RegistrationError> {
    let Operand::Constant(constant) = operand else {
        return Err(RegistrationError::new(
            registration_path,
            format!("V1 {field} field must be a string constant"),
        ));
    };
    if !is_shared_str(constant.const_.ty()) {
        return Err(RegistrationError::new(
            registration_path,
            format!("V1 {field} field has the wrong type"),
        ));
    }
    let value = constant
        .const_
        .eval(tcx, TypingEnv::fully_monomorphized(), constant.span)
        .map_err(|_| {
            RegistrationError::new(
                registration_path,
                format!("V1 {field} field could not be evaluated"),
            )
        })?;
    let bytes = value
        .try_get_slice_bytes_for_diagnostics(tcx)
        .ok_or_else(|| {
            RegistrationError::new(
                registration_path,
                format!("V1 {field} field did not evaluate to string bytes"),
            )
        })?;
    std::str::from_utf8(bytes).map(str::to_owned).map_err(|_| {
        RegistrationError::new(registration_path, format!("V1 {field} field is not UTF-8"))
    })
}

fn registration_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    operand: &Operand<'tcx>,
    field: &str,
    registration_path: &str,
) -> Result<Vec<u8>, RegistrationError> {
    let Operand::Constant(constant) = operand else {
        return Err(RegistrationError::new(
            registration_path,
            format!("V1 {field} field must be a byte-slice constant"),
        ));
    };
    if !is_shared_u8_slice(constant.const_.ty()) {
        return Err(RegistrationError::new(
            registration_path,
            format!("V1 {field} field has the wrong type"),
        ));
    }
    let value = constant
        .const_
        .eval(tcx, TypingEnv::fully_monomorphized(), constant.span)
        .map_err(|_| {
            RegistrationError::new(
                registration_path,
                format!("V1 {field} field could not be evaluated"),
            )
        })?;
    value
        .try_get_slice_bytes_for_diagnostics(tcx)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| {
            RegistrationError::new(
                registration_path,
                format!("V1 {field} field did not evaluate to byte-slice data"),
            )
        })
}

fn registration_tuple_fields<'a, 'tcx>(
    body: &'a rustc_middle::mir::Body<'tcx>,
    expected_fields: usize,
    registration_path: &str,
) -> Result<Vec<&'a Operand<'tcx>>, RegistrationError> {
    let mut aggregate = None;
    for block in body.basic_blocks.iter() {
        for statement in &block.statements {
            let Some((place, Rvalue::Aggregate(kind, fields))) = statement.kind.as_assign() else {
                continue;
            };
            if place.as_local() != Some(RETURN_PLACE) || !matches!(**kind, AggregateKind::Tuple) {
                continue;
            }
            if aggregate.replace(fields).is_some() {
                return Err(RegistrationError::new(
                    registration_path,
                    "registration initializer must contain exactly one tuple value",
                ));
            }
        }
    }
    let fields = aggregate.ok_or_else(|| {
        RegistrationError::new(
            registration_path,
            "registration initializer does not contain the required tuple value",
        )
    })?;
    if fields.len() != expected_fields {
        return Err(RegistrationError::new(
            registration_path,
            format!(
                "registration initializer has {} fields; expected {expected_fields}",
                fields.len()
            ),
        ));
    }
    Ok(fields.iter().collect())
}

fn registration_target<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &rustc_middle::mir::Body<'tcx>,
    operand: &Operand<'tcx>,
    registration_path: &str,
) -> Result<Instance<'tcx>, RegistrationError> {
    let place = match operand {
        Operand::Copy(place) | Operand::Move(place) => place,
        Operand::Constant(constant) => {
            return constant_function_target(tcx, constant, registration_path);
        }
        Operand::RuntimeChecks(_) => {
            return Err(RegistrationError::new(
                registration_path,
                "registration target field must use one exact function pointer",
            ));
        }
    };
    let Some(target_local) = place.as_local() else {
        return Err(RegistrationError::new(
            registration_path,
            "registration target field must use an unprojected function-pointer local",
        ));
    };

    let mut target = None;
    for block in body.basic_blocks.iter() {
        for statement in &block.statements {
            let Some((place, Rvalue::Cast(cast, source, _))) = statement.kind.as_assign() else {
                continue;
            };
            if place.as_local() != Some(target_local)
                || !matches!(
                    cast,
                    CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_), _)
                )
            {
                continue;
            }
            let Operand::Constant(source) = source else {
                return Err(RegistrationError::new(
                    registration_path,
                    "registration target coercion must directly name a function item",
                ));
            };
            let TyKind::FnDef(def_id, args) = source.const_.ty().kind() else {
                return Err(RegistrationError::new(
                    registration_path,
                    "registration target coercion does not reference a function item",
                ));
            };
            let resolved =
                Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), *def_id, args)
                    .ok()
                    .flatten()
                    .ok_or_else(|| {
                        RegistrationError::new(
                            registration_path,
                            "registration target function could not be resolved",
                        )
                    })?;
            if target.replace(resolved).is_some() {
                return Err(RegistrationError::new(
                    registration_path,
                    "registration target local has multiple function definitions",
                ));
            }
        }
    }

    target.ok_or_else(|| {
        RegistrationError::new(
            registration_path,
            "registration target function association is missing",
        )
    })
}

fn constant_function_target<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: &rustc_middle::mir::ConstOperand<'tcx>,
    registration_path: &str,
) -> Result<Instance<'tcx>, RegistrationError> {
    if !matches!(constant.const_.ty().kind(), TyKind::FnPtr(..)) {
        return Err(RegistrationError::new(
            registration_path,
            "registration target constant is not a function pointer",
        ));
    }
    let value = constant
        .const_
        .eval(tcx, TypingEnv::fully_monomorphized(), constant.span)
        .map_err(|_| {
            RegistrationError::new(registration_path, "registration target did not evaluate")
        })?;
    let scalar = value.try_to_scalar().ok_or_else(|| {
        RegistrationError::new(
            registration_path,
            "registration target is not one scalar pointer",
        )
    })?;
    let pointer = scalar
        .to_pointer(&tcx)
        .discard_err()
        .ok_or_else(|| {
            RegistrationError::new(registration_path, "registration target is not a pointer")
        })?
        .into_pointer_or_addr()
        .map_err(|_| {
            RegistrationError::new(registration_path, "registration target has no provenance")
        })?;
    let (provenance, offset) = pointer.into_raw_parts();
    if offset.bytes() != 0 {
        return Err(RegistrationError::new(
            registration_path,
            "registration target has a nonzero function-pointer offset",
        ));
    }
    match tcx.global_alloc(provenance.alloc_id()) {
        GlobalAlloc::Function { instance } => Ok(instance),
        _ => Err(RegistrationError::new(
            registration_path,
            "registration target pointer does not identify a function",
        )),
    }
}

fn is_shared_str(ty: rustc_middle::ty::Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Ref(_, inner, mutability) if inner.is_str() && !mutability.is_mut())
}

fn is_shared_u8_slice(ty: rustc_middle::ty::Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        TyKind::Ref(_, inner, mutability)
            if !mutability.is_mut()
                && matches!(inner.kind(), TyKind::Slice(element) if matches!(element.kind(), TyKind::Uint(rustc_middle::ty::UintTy::U8)))
    )
}

fn validate_registration_records<T: Copy>(
    mut records: Vec<RegistrationRecord<T>>,
    expected_crate_binding: Option<CrateBindingIdV1>,
    expected_crate_name: Option<&str>,
) -> Result<Vec<KernelRoot<T>>, RegistrationError> {
    records.sort_by(|lhs, rhs| lhs.registration_path.cmp(&rhs.registration_path));

    let mut logical_names = BTreeMap::new();
    let mut export_names = BTreeMap::new();
    let mut target_identities = BTreeMap::new();
    let mut roots = Vec::with_capacity(records.len());

    for record in records {
        let error = |reason| RegistrationError::new(record.registration_path.clone(), reason);
        if record.magic != reserved_fe2o3_symbols::KERNEL_REGISTRATION_MAGIC {
            return Err(error(format!(
                "magic {:#018x} does not match registration magic {:#018x}",
                record.magic,
                reserved_fe2o3_symbols::KERNEL_REGISTRATION_MAGIC
            )));
        }
        let generated_host_contract_identity = match (record.version, record.kind) {
            (
                reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V1,
                reserved_fe2o3_symbols::KERNEL_REGISTRATION_KIND_KERNEL,
            ) => None,
            (
                reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V3,
                reserved_fe2o3_symbols::KERNEL_REGISTRATION_KIND_TYPED_GENERAL_LAYOUT_V3,
            ) => {
                let profile_tag = record
                    .profile_tag
                    .as_deref()
                    .ok_or_else(|| error("V3 registration has no profile tag".to_owned()))?;
                if profile_tag != TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3 {
                    return Err(error(format!(
                        "V3 profile tag {profile_tag:?} is not the canonical general typed profile {TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3:?}"
                    )));
                }
                Some(record.generated_host_contract_identity.ok_or_else(|| {
                    error("V3 registration has no generated host-contract identity".to_owned())
                })?)
            }
            (reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V2, _) => {
                return Err(error(
                    "registration version 2 has been removed; use the generic typed V3 contract"
                        .to_owned(),
                ));
            }
            (version, kind)
                if kind
                    == reserved_fe2o3_symbols::KERNEL_REGISTRATION_KIND_TYPED_GENERAL_LAYOUT_V3
                    && version != reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V3 =>
            {
                return Err(error(
                    "general typed registrations require version 3".to_owned(),
                ));
            }
            (reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V3, _) => {
                return Err(error(
                    "version 3 is reserved for the general typed registration kind".to_owned(),
                ));
            }
            (version, _)
                if version != reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V1
                    && version != reserved_fe2o3_symbols::KERNEL_REGISTRATION_VERSION_V3 =>
            {
                return Err(error(format!("unknown registration version {version}")));
            }
            (_, kind) => return Err(error(format!("unknown registration kind {kind}"))),
        };
        if record.logical_name.is_empty() {
            return Err(error("logical name must not be empty".to_string()));
        }
        if record.export_name.is_empty() {
            return Err(error("export name must not be empty".to_string()));
        }

        let expected_item_name = format!(
            "{}{}",
            reserved_fe2o3_symbols::KERNEL_REGISTRATION_PREFIX,
            record.logical_name
        );
        if record.item_name != expected_item_name {
            return Err(error(format!(
                "item name `{}` is inconsistent with logical name `{}`",
                record.item_name, record.logical_name
            )));
        }

        let registration_module = owner_module_path(&record.registration_path);
        let target_module = owner_module_path(&record.target_identity);
        let target_is_external =
            expected_crate_name.is_some_and(|expected| record.target_crate_name != expected);
        if generated_host_contract_identity.is_some() {
            let _expected_crate_name = expected_crate_name.ok_or_else(|| {
                error("typed registration has no rustc session crate identity".to_owned())
            })?;
            if target_is_external {
                return Err(error(
                    "cross-crate generic typed kernels are not yet supported".to_owned(),
                ));
            }
            if registration_module != target_module {
                return Err(error(format!(
                    "registered target module {target_module:?} disagrees with registration module {registration_module:?}"
                )));
            }
        }

        let kernel_binding = match generated_host_contract_identity {
            Some(_) => {
                let crate_binding = record
                    .crate_binding
                    .ok_or_else(|| error("V3 registration has no crate binding".to_owned()))?;
                let expected = expected_crate_binding.ok_or_else(|| {
                    error("V3 registration has no rustc session crate binding".to_owned())
                })?;
                if crate_binding != expected {
                    return Err(error(format!(
                        "crate binding {} disagrees with rustc session binding {}",
                        crate_binding.to_hex(),
                        expected.to_hex()
                    )));
                }
                let declared = record
                    .kernel_binding
                    .ok_or_else(|| error("V3 registration has no kernel binding".to_owned()))?;
                let expected = derive_kernel_binding_id_v1(
                    crate_binding,
                    TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3,
                    &record.logical_name,
                    &record.export_name,
                );
                if declared != expected {
                    return Err(error(format!(
                        "kernel binding {} disagrees with derived binding {}",
                        declared.to_hex(),
                        expected.to_hex()
                    )));
                }
                Some(declared)
            }
            None => {
                if record.crate_binding.is_some()
                    || record.kernel_binding.is_some()
                    || record.profile_tag.is_some()
                    || record.generated_host_contract_identity.is_some()
                {
                    return Err(error(
                        "V1 registration unexpectedly carries binding IDs".to_owned(),
                    ));
                }
                None
            }
        };

        let expected_target_symbol = match kernel_binding {
            Some(binding) => host_kernel_symbol_v1(binding),
            None => format!(
                "{}{}",
                reserved_fe2o3_symbols::KERNEL_PREFIX,
                record.export_name
            ),
        };
        if record.target_symbol != expected_target_symbol {
            return Err(error(format!(
                "target symbol `{}` is inconsistent with export name `{}`",
                record.target_symbol, record.export_name
            )));
        }

        reject_duplicate(
            &mut logical_names,
            &record.logical_name,
            &record.registration_path,
            "logical name",
        )?;
        reject_duplicate(
            &mut export_names,
            &record.export_name,
            &record.registration_path,
            "export name",
        )?;
        reject_duplicate(
            &mut target_identities,
            &record.target_identity,
            &record.registration_path,
            "target identity",
        )?;

        roots.push(KernelRoot {
            target: record.target,
            logical_name: record.logical_name,
            export_name: record.export_name,
            generated_host_contract_identity,
            kernel_binding,
            frontend_contract: None,
            kernel_context_contract: None,
            reference_effect_binding: None,
            reference_target: None,
        });
    }

    roots.sort_by(|lhs, rhs| {
        lhs.logical_name
            .cmp(&rhs.logical_name)
            .then_with(|| lhs.export_name.cmp(&rhs.export_name))
    });
    Ok(roots)
}

fn owner_module_path(path: &str) -> &str {
    path.rsplit_once("::")
        .map_or("", |(module_path, _)| module_path)
}

fn reject_duplicate(
    seen: &mut BTreeMap<String, String>,
    value: &str,
    registration_path: &str,
    field: &str,
) -> Result<(), RegistrationError> {
    if let Some(previous) = seen.insert(value.to_string(), registration_path.to_string()) {
        return Err(RegistrationError::new(
            registration_path,
            format!("duplicate {field} `{value}`; first registered by `{previous}`"),
        ));
    }
    Ok(())
}

fn final_path_segment(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

fn is_fully_monomorphized<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let generics = tcx.generics_of(instance.def_id());

    for arg in instance.args.iter() {
        if arg.has_param() || arg.has_escaping_bound_vars() {
            return false;
        }
    }

    generics.count() == 0 || !instance.args.is_empty()
}

struct DeviceCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    context_producers: CapturedContextProducersV1<'tcx>,
    seen: BTreeSet<crate::device_ffi::DeviceFfiInstanceIdentity>,
    call_chains: BTreeMap<
        crate::device_ffi::DeviceFfiInstanceIdentity,
        CallChainLink<crate::device_ffi::DeviceFfiInstanceIdentity>,
    >,
    call_edges: closure_flow_v1::CallGraphV1,
    closure_work: crate::rustc_semantic_plan_v1::SourceClosureWorkV1,
    terminal_census: CollectedTileTerminalCensusV259,
    reachable_unsafe_calls:
        BTreeMap<crate::device_ffi::DeviceFfiInstanceIdentity, BTreeSet<String>>,
    inline_assembly:
        BTreeMap<crate::device_ffi::DeviceFfiInstanceIdentity, ObservedInlineAssemblyV1>,
    used_export_names: BTreeSet<String>,
    worklist: VecDeque<CollectedFunction<'tcx>>,
    result: Vec<CollectedFunction<'tcx>>,
    ffi_declarations: Vec<crate::device_ffi::CollectedDeviceFfi<'tcx>>,
    reachable_ffi_imports: BTreeSet<reserved_fe2o3_symbols::DeviceFfiContractIdV1>,
    expected_target: String,
    inspected_blocks: usize,
    verbose: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CallChainLink<T> {
    predecessor: Option<T>,
    label: String,
}

fn reconstruct_call_chain<T: Clone + Ord>(
    links: &BTreeMap<T, CallChainLink<T>>,
    start: &T,
) -> Vec<String> {
    let mut reverse_chain = Vec::new();
    let mut cursor = Some(start.clone());
    for _ in 0..links.len() {
        let Some(identity) = cursor else {
            break;
        };
        let Some(link) = links.get(&identity) else {
            break;
        };
        reverse_chain.push(link.label.clone());
        cursor = link.predecessor.clone();
    }
    reverse_chain.reverse();
    reverse_chain
}

fn root_scoped_call_chains<T: Clone + Ord, S>(
    edges: &BTreeMap<T, BTreeMap<T, S>>,
    labels: &BTreeMap<T, String>,
    root: &T,
) -> (BTreeMap<T, CallChainLink<T>>, Vec<T>) {
    let Some(root_label) = labels.get(root) else {
        return (BTreeMap::new(), Vec::new());
    };
    let mut links = BTreeMap::from([(
        root.clone(),
        CallChainLink {
            predecessor: None,
            label: root_label.clone(),
        },
    )]);
    let mut order = Vec::new();
    let mut pending = VecDeque::from([root.clone()]);
    while let Some(caller) = pending.pop_front() {
        order.push(caller.clone());
        let Some(callees) = edges.get(&caller) else {
            continue;
        };
        for callee in callees.keys() {
            let Some(label) = labels.get(callee) else {
                continue;
            };
            if links.contains_key(callee) {
                continue;
            }
            links.insert(
                callee.clone(),
                CallChainLink {
                    predecessor: Some(caller.clone()),
                    label: label.clone(),
                },
            );
            pending.push_back(callee.clone());
        }
    }
    (links, order)
}

#[derive(Default)]
struct UserUnsafeBlockVisitor {
    first_span: Option<rustc_span::Span>,
}

impl<'tcx> Visitor<'tcx> for UserUnsafeBlockVisitor {
    fn visit_block(&mut self, block: &'tcx rustc_hir::Block<'tcx>) {
        if matches!(
            block.rules,
            BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
        ) {
            self.first_span.get_or_insert(block.span);
        }
        intravisit::walk_block(self, block);
    }

    fn visit_expr(&mut self, expression: &'tcx rustc_hir::Expr<'tcx>) {
        // A closure body is authenticated when its concrete callable instance
        // enters the collected graph, not merely when a closure value is made.
        if !matches!(expression.kind, ExprKind::Closure(_)) {
            intravisit::walk_expr(self, expression);
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ObservedInlineAssemblyV1 {
    blocks: u32,
    operand_bits: u16,
    option_sets: BTreeSet<u16>,
}

impl<'tcx> DeviceCollector<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        verbose: bool,
        ffi_declarations: Vec<crate::device_ffi::CollectedDeviceFfi<'tcx>>,
        target: String,
        context_producers: CapturedContextProducersV1<'tcx>,
    ) -> Self {
        Self {
            tcx,
            context_producers,
            seen: BTreeSet::new(),
            call_chains: BTreeMap::new(),
            call_edges: BTreeMap::new(),
            closure_work: crate::rustc_semantic_plan_v1::SourceClosureWorkV1::default(),
            terminal_census: CollectedTileTerminalCensusV259::empty(),
            reachable_unsafe_calls: BTreeMap::new(),
            inline_assembly: BTreeMap::new(),
            used_export_names: BTreeSet::new(),
            worklist: VecDeque::new(),
            result: Vec::new(),
            ffi_declarations,
            reachable_ffi_imports: BTreeSet::new(),
            expected_target: target,
            inspected_blocks: 0,
            verbose,
        }
    }

    fn mark_seen(
        &mut self,
        identity: crate::device_ffi::DeviceFfiInstanceIdentity,
    ) -> Result<bool, CollectError> {
        if self.seen.len() >= fe2o3_rustc_front::MAX_FUNCTIONS_V1 && !self.seen.contains(&identity)
        {
            return Err(CollectError {
                message: format!(
                    "device-reachable function count exceeds hard maximum {}",
                    fe2o3_rustc_front::MAX_FUNCTIONS_V1,
                ),
            });
        }
        Ok(self.seen.insert(identity))
    }

    fn charge_function_blocks(
        &mut self,
        function: &Instance<'tcx>,
        block_count: usize,
    ) -> Result<(), CollectError> {
        if block_count > fe2o3_rustc_front::MAX_BLOCKS_PER_FUNCTION_V1 {
            return Err(self.reachable_error(
                function,
                &format!(
                    "function contains {block_count} MIR blocks; hard maximum is {}",
                    fe2o3_rustc_front::MAX_BLOCKS_PER_FUNCTION_V1,
                ),
                None,
            ));
        }
        let total = self
            .inspected_blocks
            .checked_add(block_count)
            .ok_or_else(|| {
                self.reachable_error(function, "reachable MIR block accounting overflowed", None)
            })?;
        if total > fe2o3_rustc_front::MAX_TOTAL_BLOCKS_V1 {
            return Err(self.reachable_error(
                function,
                &format!(
                    "device closure contains {total} MIR blocks; hard maximum is {}",
                    fe2o3_rustc_front::MAX_TOTAL_BLOCKS_V1,
                ),
                None,
            ));
        }
        self.inspected_blocks = total;
        Ok(())
    }

    fn add_device_export(
        &mut self,
        instance: Instance<'tcx>,
        export_name: String,
    ) -> Result<(), CollectError> {
        if !self.used_export_names.insert(export_name.clone()) {
            return Err(CollectError {
                message: format!(
                    "fe2o3 device FFI export `{export_name}` has duplicate symbol ownership"
                ),
            });
        }
        let identity = self.instance_identity(instance);
        if self.mark_seen(identity.clone())? {
            self.call_chains.insert(
                identity,
                CallChainLink {
                    predecessor: None,
                    label: self.instance_label(instance),
                },
            );
            self.worklist.push_back(CollectedFunction {
                instance,
                role: CollectedFunctionRole::DeviceFfiExport,
                export_name,
                logical_name: None,
                generated_host_contract_identity: None,
                kernel_binding: None,
                frontend_contract: None,
                reference_effect_binding: None,
                reference_instance: None,
                dead_branches: None,
                closure_observation: None,
                kernel_context_contract: None,
            });
        }
        Ok(())
    }

    fn add_root(&mut self, root: KernelRoot<Instance<'tcx>>) -> Result<(), CollectError> {
        let KernelRoot {
            target: instance,
            logical_name,
            export_name,
            generated_host_contract_identity,
            kernel_binding,
            frontend_contract,
            kernel_context_contract,
            reference_effect_binding,
            reference_target: reference_instance,
        } = root;
        if !self.used_export_names.insert(export_name.clone()) {
            return Err(CollectError {
                message: format!(
                    "fe2o3 kernel export `{export_name}` conflicts with an existing kernel or device FFI symbol"
                ),
            });
        }
        let identity = self.instance_identity(instance);
        if self.mark_seen(identity.clone())? {
            self.call_chains.insert(
                identity.clone(),
                CallChainLink {
                    predecessor: None,
                    label: self.instance_label(instance),
                },
            );
            self.worklist.push_back(CollectedFunction {
                instance,
                role: CollectedFunctionRole::KernelEntry,
                export_name,
                logical_name: Some(logical_name),
                generated_host_contract_identity,
                kernel_binding,
                frontend_contract,
                reference_effect_binding,
                reference_instance,
                dead_branches: None,
                closure_observation: None,
                kernel_context_contract,
            });
        }
        Ok(())
    }

    fn collect(
        self,
    ) -> Result<
        (
            CollectionResult<'tcx>,
            kernel_context_auth_v1::AuthenticatedContextEntriesV1<'tcx>,
            closure_flow_v1::AuthenticatedClosureFlowV1<'tcx>,
            CollectedTileTerminalCensusV259,
        ),
        CollectError,
    > {
        self.collect_with_enrollment_v1(None, None)
    }

    fn collect_with_enrollment_v1(
        mut self,
        enrollment: Option<reference_enrollment_v1::RetainedEnrollmentV1<'tcx>>,
        loan: Option<
            &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
        >,
    ) -> Result<
        (
            CollectionResult<'tcx>,
            kernel_context_auth_v1::AuthenticatedContextEntriesV1<'tcx>,
            closure_flow_v1::AuthenticatedClosureFlowV1<'tcx>,
            CollectedTileTerminalCensusV259,
        ),
        CollectError,
    > {
        while let Some(mut function) = self.worklist.pop_front() {
            let def_id = function.instance.def_id();

            if !self.tcx.is_mir_available(def_id)
                && crate::closure_profile_v1::authenticate_once_shim_v1(self.tcx, function.instance)
                    .map_err(|error| {
                        self.reachable_error(&function.instance, &error.to_string(), None)
                    })?
                    .is_none()
            {
                return Err(self.reachable_error(
                    &function.instance,
                    "MIR is unavailable for a collected device function",
                    None,
                ));
            }

            let mir = self.tcx.instance_mir(function.instance.def);
            self.charge_function_blocks(&function.instance, mir.basic_blocks.len())?;
            let dead_branches =
                crate::monomorphization_dead::CompilerDeadBranchObservationV1::observe(
                    self.tcx,
                    function.instance,
                    mir,
                )
                .map_err(|error| {
                    self.reachable_error(
                        &function.instance,
                        &format!("dead-branch observation failed closed: {error}"),
                        None,
                    )
                })?;
            if self.verbose {
                let context = dead_branches.evidence().context();
                eprintln!(
                    "[collector] visiting {} ({} basic blocks, {} V1 decisions, {} policy-excluded)",
                    function.export_name,
                    mir.basic_blocks.len(),
                    dead_branches.evidence().decisions().len(),
                    dead_branches.excluded_blocks().len(),
                );
                eprintln!(
                    "[collector] V1 identity {} mir={} source={} target={}",
                    function.export_name,
                    encode_lower_hex(&context.cfg_identity()),
                    encode_lower_hex(&context.source_identity()),
                    encode_lower_hex(&context.target_identity()),
                );
            }

            for (index, block) in mir.basic_blocks.iter_enumerated() {
                if let Some(terminator) = &block.terminator {
                    self.process_terminator(&terminator.kind, mir, index, &function.instance)?;
                }
            }

            function.dead_branches = Some(dead_branches);
            self.result.push(function);
        }

        self.authenticate_production_kernel_source_safety()?;
        self.authenticate_reachable_frontend_contracts()?;
        let context_entries = kernel_context_auth_v1::authenticate_v1(&mut self)?;
        let closure_flow = closure_flow_v1::authenticate_with_enrollment_v1(
            self.tcx,
            &mut self.result,
            self.call_edges,
            self.closure_work,
            self.verbose,
            enrollment,
            loan,
        )
        .map_err(|error| CollectError {
            message: format!("bounded closure admission failed: {error}"),
        })?;

        let device_ffi = crate::device_ffi::validate_local_closure(
            self.tcx,
            &mut self.ffi_declarations,
            &self.reachable_ffi_imports,
        )
        .map_err(|error| CollectError {
            message: error.to_string(),
        })?;
        if self.verbose && !device_ffi.is_empty() {
            eprintln!(
                "[collector] validated local device FFI evidence: {} imports, {} exports, target {}, asserted code object v{}",
                device_ffi.imports.len(),
                device_ffi.exports.len(),
                device_ffi.target.as_deref().unwrap_or("<none>"),
                device_ffi
                    .code_object_version_assertion
                    .as_ref()
                    .map(|version| *version.asserted_for_consistency_check())
                    .unwrap_or_default(),
            );
        }

        let mut collection = CollectionResult {
            functions: self.result,
            device_ffi,
            compiler_ffi_observation: None,
        };
        collection.compiler_ffi_observation =
            crate::compiler_ffi_adapter::adapt_collection_v1(self.tcx, &collection).map_err(
                |error| CollectError {
                    message: format!("compiler FFI envelope construction failed: {error}"),
                },
            )?;
        Ok((
            collection,
            context_entries,
            closure_flow,
            self.terminal_census,
        ))
    }

    fn process_terminator(
        &mut self,
        terminator: &TerminatorKind<'tcx>,
        body: &Body<'tcx>,
        block: BasicBlock,
        caller: &Instance<'tcx>,
    ) -> Result<(), CollectError> {
        match terminator {
            TerminatorKind::Call { func, unwind, .. } => {
                // `Continue` has no executable cleanup target to traverse. Its
                // no-unwind obligation is discharged by walking the resolved
                // direct callee below and rejecting panic/unwind MIR there.
                if !matches!(unwind, UnwindAction::Continue | UnwindAction::Unreachable) {
                    return Err(self.reachable_error(
                        caller,
                        &format!(
                            "[FE2O3-FFI-EDGE002] direct call has an untraversed unwind edge `{unwind:?}`"
                        ),
                        None,
                    ));
                }
                self.process_call_operand(func, caller, body, block)
            }
            TerminatorKind::InlineAsm {
                asm_macro,
                operands,
                options,
                targets,
                unwind,
                ..
            } => self.process_inline_assembly(
                *asm_macro,
                operands,
                *options,
                targets.len(),
                *unwind,
                body,
                caller,
            ),
            TerminatorKind::Assert {
                unwind: UnwindAction::Continue | UnwindAction::Unreachable,
                ..
            } => Ok(()),
            TerminatorKind::Drop {
                place,
                unwind: UnwindAction::Continue | UnwindAction::Unreachable,
                ..
            } => {
                match classify_rustc_drop_v1(self.tcx, *caller, body, *place) {
                    Ok(ProductionRustcDropClassV1::Trivial) => Ok(()),
                    Ok(ProductionRustcDropClassV1::RequiresDropGlue) => Err(self.reachable_error(
                        caller,
                        "[FE2O3-FFI-EDGE001] unsupported executable MIR edge `Drop requiring drop glue`",
                        None,
                    )),
                    Err(_) => Err(self.reachable_error(
                        caller,
                        "[FE2O3-FFI-EDGE001] Drop place type failed monomorphic normalization",
                        None,
                    )),
                }
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable => Ok(()),
            unsupported => Err(self.reachable_error(
                caller,
                &format!("[FE2O3-FFI-EDGE001] unsupported executable MIR edge `{unsupported:?}`"),
                None,
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_inline_assembly(
        &mut self,
        asm_macro: InlineAsmMacro,
        operands: &[InlineAsmOperand<'tcx>],
        options: InlineAsmOptions,
        target_count: usize,
        unwind: UnwindAction,
        body: &Body<'tcx>,
        caller: &Instance<'tcx>,
    ) -> Result<(), CollectError> {
        if !matches!(asm_macro, InlineAsmMacro::Asm) {
            return Err(self.reachable_error(
                caller,
                "only local asm! is representable by the V1 kernel frontend contract",
                None,
            ));
        }
        if target_count != 1 || !matches!(unwind, UnwindAction::Unreachable) {
            return Err(self.reachable_error(
                caller,
                "inline assembly with labels, divergence, or unwind behavior is not representable by the V1 kernel frontend contract",
                None,
            ));
        }

        let supported_options = InlineAsmOptions::PURE
            | InlineAsmOptions::NOMEM
            | InlineAsmOptions::READONLY
            | InlineAsmOptions::PRESERVES_FLAGS
            | InlineAsmOptions::NOSTACK;
        let unsupported = options.bits() & !supported_options.bits();
        if unsupported != 0 {
            return Err(self.reachable_error(
                caller,
                &format!(
                    "inline assembly uses options {unsupported:#x} outside the V1 frontend contract"
                ),
                None,
            ));
        }

        let mut operand_bits = 0_u16;
        for operand in operands {
            operand_bits |= match operand {
                InlineAsmOperand::In { value, .. } => {
                    assembly_operand_bit(value.ty(body, self.tcx)).ok_or_else(|| {
                        self.reachable_error(
                            caller,
                            &format!(
                                "inline assembly input type `{}` has no V1 operand classification",
                                value.ty(body, self.tcx)
                            ),
                            None,
                        )
                    })?
                }
                InlineAsmOperand::Out {
                    place: Some(place), ..
                } => {
                    let ty = place.ty(body, self.tcx).ty;
                    assembly_operand_bit(ty).ok_or_else(|| {
                        self.reachable_error(
                            caller,
                            &format!(
                                "inline assembly output type `{ty}` has no V1 operand classification"
                            ),
                            None,
                        )
                    })?
                }
                InlineAsmOperand::InOut {
                    in_value,
                    out_place,
                    ..
                } => {
                    let input_ty = in_value.ty(body, self.tcx);
                    let input = assembly_operand_bit(input_ty).ok_or_else(|| {
                        self.reachable_error(
                            caller,
                            &format!(
                                "inline assembly inout type `{input_ty}` has no V1 operand classification"
                            ),
                            None,
                        )
                    })?;
                    if let Some(place) = out_place {
                        let output_ty = place.ty(body, self.tcx).ty;
                        let output = assembly_operand_bit(output_ty).ok_or_else(|| {
                            self.reachable_error(
                                caller,
                                &format!(
                                    "inline assembly inout output type `{output_ty}` has no V1 operand classification"
                                ),
                                None,
                            )
                        })?;
                        if input != output {
                            return Err(self.reachable_error(
                                caller,
                                "inline assembly inout types disagree on their V1 operand classification",
                                None,
                            ));
                        }
                    }
                    input
                }
                InlineAsmOperand::Const { .. } => ASSEMBLY_OPERAND_IMMEDIATE_V1,
                InlineAsmOperand::Out { place: None, .. }
                | InlineAsmOperand::SymFn { .. }
                | InlineAsmOperand::SymStatic { .. }
                | InlineAsmOperand::Label { .. } => {
                    return Err(self.reachable_error(
                        caller,
                        "inline assembly clobber-only, symbol, and label operands are not representable by the V1 frontend contract",
                        None,
                    ));
                }
            };
        }
        if operand_bits == 0 {
            return Err(self.reachable_error(
                caller,
                "inline assembly has no classifiable V1 operands",
                None,
            ));
        }

        let option_bits = frontend_assembly_option_bits(options);
        let identity = self.instance_identity(*caller);
        let observation = self.inline_assembly.entry(identity).or_default();
        observation.blocks = observation
            .blocks
            .checked_add(1)
            .ok_or_else(|| CollectError {
                message: "reachable inline assembly block count exceeds u32".to_owned(),
            })?;
        observation.operand_bits |= operand_bits;
        observation.option_sets.insert(option_bits);
        Ok(())
    }

    fn authenticate_reachable_frontend_contracts(&mut self) -> Result<(), CollectError> {
        let kernel_indices = self
            .result
            .iter()
            .enumerate()
            .filter_map(|(index, function)| function.is_kernel_entry().then_some(index))
            .collect::<Vec<_>>();

        for index in kernel_indices {
            let function = &self.result[index];
            let identity = self.instance_identity(function.instance);
            let observed = self.reachable_inline_assembly(&identity)?;
            let logical_name = function
                .logical_name
                .as_deref()
                .unwrap_or(&function.export_name);
            let authenticated = reconcile_frontend_contract(
                logical_name,
                &self.expected_target,
                function.frontend_contract.as_ref(),
                observed,
            )?;
            if let Some(contract) = &mut self.result[index].frontend_contract {
                contract.reachable_assembly = authenticated;
            }
        }
        Ok(())
    }

    fn authenticate_production_kernel_source_safety(&mut self) -> Result<(), CollectError> {
        let functions = self
            .result
            .iter()
            .map(|function| (self.instance_identity(function.instance), function))
            .collect::<BTreeMap<_, _>>();
        let labels = functions
            .iter()
            .map(|(identity, function)| (identity.clone(), self.instance_label(function.instance)))
            .collect::<BTreeMap<_, _>>();

        for root in self
            .result
            .iter()
            .filter(|function| function.is_kernel_entry())
        {
            if root
                .frontend_contract
                .as_ref()
                .and_then(|contract| contract.contract.unsafe_assembly())
                .is_some()
            {
                continue;
            }
            let logical_name = root.logical_name.as_deref().unwrap_or(&root.export_name);
            let root_identity = self.instance_identity(root.instance);
            let (links, order) = root_scoped_call_chains(&self.call_edges, &labels, &root_identity);

            for identity in order {
                let function = functions
                    .get(&identity)
                    .expect("root-scoped traversal retains only collected function labels");
                let chain = || reconstruct_call_chain(&links, &identity).join(" -> ");
                let safety = if self.tcx.def_kind(function.instance.def_id())
                    == rustc_hir::def::DefKind::Closure
                {
                    function.instance.args.as_closure().sig().safety()
                } else {
                    self.tcx
                        .fn_sig(function.instance.def_id())
                        .skip_binder()
                        .safety()
                };
                if safety == Safety::Unsafe
                    && !crate::production_rustc_intrinsic_v1::is_reviewed_core_atomic_function_v1(
                        self.tcx,
                        function.instance,
                    )
                    && !crate::trusted_device_items::is_authenticated_gfx942_wave64_scan_instance_v1(
                        self.tcx,
                        function.instance,
                        &self.expected_target,
                    )
                {
                    return Err(CollectError {
                        message: format!(
                            "ordinary production kernel `{logical_name}` reaches unsafe function instance `{}`; reachable call chain: {}",
                            self.instance_label(function.instance),
                            chain(),
                        ),
                    });
                }
                if let Some(callees) = self.reachable_unsafe_calls.get(&identity)
                    && let Some(callee) = callees.first()
                {
                    return Err(CollectError {
                        message: format!(
                            "ordinary production kernel `{logical_name}` reaches unsafe function instance `{callee}`; reachable call chain: {} -> {callee}",
                            chain(),
                        ),
                    });
                }

                inlined_source_safety_v1::audit(
                    self.tcx,
                    function.instance,
                    self.tcx.instance_mir(function.instance.def),
                    &mut self.closure_work,
                    |source, callsite, work| {
                        work.charge(1)?;
                        inlined_source_safety_v1::reviewed_external_source(
                            self.tcx,
                            source,
                            &self.expected_target,
                        )
                        .map_err(|detail| {
                            inlined_source_safety_v1::AuditError::Origin {
                                instance: source,
                                source,
                                callsite,
                                reason:
                                    inlined_source_safety_v1::OriginRefusal::ExternalAuthentication(
                                        detail,
                                    ),
                            }
                        })
                    },
                )
                .map_err(|error| CollectError {
                    message: format!(
                        "ordinary production kernel `{logical_name}` {}; reachable call chain: {}",
                        error.describe(self.tcx),
                        chain(),
                    ),
                })?;
                if crate::trusted_device_items::classify(self.tcx, function.instance.def_id())
                    .is_some_and(
                        crate::production_semantic_terminal_v1::is_traversed_reviewed_helper_v1,
                    )
                    || crate::production_rustc_intrinsic_v1::is_reviewed_device_global_mut_ptr_as_raw_v1(
                        self.tcx,
                        function.instance,
                    )
                {
                    continue;
                }
                if crate::closure_profile_v1::authenticate_once_shim_v1(self.tcx, function.instance)
                    .map_err(|error| {
                        self.reachable_error(&function.instance, &error.to_string(), None)
                    })?
                    .is_some()
                {
                    // Only generated adapter syntax has no user HIR. The
                    // actual closure body remains in this traversal.
                    continue;
                }
                let Some(local_def_id) = function.instance.def_id().as_local() else {
                    match inlined_source_safety_v1::reviewed_external_source(
                        self.tcx,
                        function.instance,
                        &self.expected_target,
                    ) {
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(detail) => {
                            return Err(CollectError {
                                message: format!(
                                    "ordinary production kernel `{logical_name}` rejected reviewed external helper `{}`: {detail}; reachable call chain: {}",
                                    self.instance_label(function.instance),
                                    chain(),
                                ),
                            });
                        }
                    }
                    return Err(CollectError {
                        message: format!(
                            "ordinary production kernel `{logical_name}` cannot authenticate the absence of user-provided unsafe blocks in external helper `{}`: cross-crate HIR is unavailable and optimized MIR does not retain unsafe-block syntax; reachable call chain: {}",
                            self.instance_label(function.instance),
                            chain(),
                        ),
                    });
                };
                let Some(body) = self.tcx.hir_maybe_body_owned_by(local_def_id) else {
                    return Err(CollectError {
                        message: format!(
                            "ordinary production kernel `{logical_name}` cannot authenticate local HIR for reachable function `{}`; reachable call chain: {}",
                            self.instance_label(function.instance),
                            chain(),
                        ),
                    });
                };
                let mut visitor = UserUnsafeBlockVisitor::default();
                visitor.visit_body(body);
                if let Some(span) = visitor.first_span {
                    return Err(CollectError {
                        message: format!(
                            "ordinary production kernel `{logical_name}` reaches a safe-signature local helper containing a user-provided unsafe block at {}; reachable call chain: {}",
                            self.tcx.sess.source_map().span_to_diagnostic_string(span),
                            chain(),
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn reachable_inline_assembly(
        &self,
        root: &crate::device_ffi::DeviceFfiInstanceIdentity,
    ) -> Result<ObservedInlineAssemblyV1, CollectError> {
        let mut visited = BTreeSet::new();
        let mut pending = vec![root.clone()];
        let mut summary = ObservedInlineAssemblyV1::default();
        while let Some(identity) = pending.pop() {
            if !visited.insert(identity.clone()) {
                continue;
            }
            if let Some(observed) = self.inline_assembly.get(&identity) {
                summary.blocks =
                    summary
                        .blocks
                        .checked_add(observed.blocks)
                        .ok_or_else(|| CollectError {
                            message: "reachable inline assembly block count exceeds u32".to_owned(),
                        })?;
                summary.operand_bits |= observed.operand_bits;
                summary
                    .option_sets
                    .extend(observed.option_sets.iter().copied());
            }
            if let Some(callees) = self.call_edges.get(&identity) {
                pending.extend(callees.keys().cloned());
            }
        }
        Ok(summary)
    }

    fn process_call_operand(
        &mut self,
        func: &Operand<'tcx>,
        caller: &Instance<'tcx>,
        body: &Body<'tcx>,
        block: BasicBlock,
    ) -> Result<(), CollectError> {
        let Operand::Constant(const_op) = func else {
            return Err(self.reachable_error(
                caller,
                "[FE2O3-FFI-CALL001] indirect function-pointer calls are not permitted in the closed device graph",
                None,
            ));
        };

        let ty = const_op.const_.ty();
        let TyKind::FnDef(def_id, args) = ty.kind() else {
            return Err(self.reachable_error(
                caller,
                &format!(
                    "[FE2O3-FFI-CALL002] device call operand has non-function-definition type `{ty}`"
                ),
                None,
            ));
        };
        if let Some(rejection) = crate::trusted_device_items::rejected_provider(self.tcx, *def_id) {
            return Err(self.reachable_error(
                caller,
                &format!(
                    "trusted-provider rejection: diagnostic item `{}` is defined by `{}` but is not bound to the reviewed `{}` compilation unit: {}",
                    rejection.marker,
                    self.tcx.def_path_str(*def_id),
                    rejection.expected_provider_crate,
                    rejection.reason,
                ),
                None,
            ));
        }
        let callee_path = self.tcx.def_path_str(*def_id);
        if crate::production_core_panic_v50::is_candidate(self.tcx, func) {
            let checked = crate::production_core_panic_v50::observe(
                self.tcx,
                *caller,
                body,
                block,
                &mut |amount| self.closure_work.charge(amount),
            );
            return match checked {
                Ok(Some(_)) => Ok(()),
                Ok(None) => Err(self.reachable_error(
                    caller,
                    "core panic candidate lost its exact original call",
                    Some(callee_path),
                )),
                Err(crate::production_core_panic_v50::CorePanicErrorV50::Work(error)) => {
                    Err(self.reachable_error(caller, &error.to_string(), Some(callee_path)))
                }
                Err(crate::production_core_panic_v50::CorePanicErrorV50::Refused(reason)) => {
                    Err(self.reachable_error(caller, reason, Some(callee_path)))
                }
            };
        }
        if callee_path.contains("::panicking::")
            || callee_path.contains("::panic_fmt")
            || callee_path.contains("::begin_panic")
            || callee_path.contains("::unwrap_failed")
            || callee_path.contains("precondition_check")
        {
            return Err(self.reachable_error(
                caller,
                "device code reaches a panic path",
                Some(callee_path),
            ));
        }

        let normalized_args = self.tcx.instantiate_and_normalize_erasing_regions(
            caller.args,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(*args),
        );
        let resolved = match Instance::try_resolve(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            *def_id,
            normalized_args,
        ) {
            Ok(Some(resolved)) => resolved,
            Ok(None) => {
                return Err(self.reachable_error(
                    caller,
                    "direct device callee could not be resolved to a concrete rustc instance",
                    Some(self.tcx.def_path_str(*def_id)),
                ));
            }
            Err(_) => {
                return Err(self.reachable_error(
                    caller,
                    "direct device callee normalization failed",
                    Some(self.tcx.def_path_str(*def_id)),
                ));
            }
        };
        let normalized_safe_core_shift =
            crate::production_safe_core_shift_v1::SafeCoreShiftV1::classify(self.tcx, resolved)
                .map_err(|error| {
                    self.reachable_error(caller, error, Some(self.instance_label(resolved)))
                })?;
        let normalized_intrinsic =
            if crate::production_semantic_terminal_v1::classify(self.tcx, resolved.def_id())
                .is_none()
            {
                crate::production_rustc_intrinsic_v1::classify(self.tcx, resolved).map_err(
                    |error| {
                        self.reachable_error(
                            caller,
                            &format!("unsupported rustc compiler intrinsic: {error}"),
                            Some(self.instance_label(resolved)),
                        )
                    },
                )?
            } else {
                None
            };
        if normalized_intrinsic.is_none()
            && self.tcx.fn_sig(*def_id).skip_binder().safety() == Safety::Unsafe
            && !crate::production_rustc_intrinsic_v1::is_reviewed_core_atomic_function_v1(
                self.tcx, resolved,
            )
            && !crate::trusted_device_items::is_authenticated_gfx942_wave64_shuffle_instance_v1(
                self.tcx,
                resolved,
                &self.expected_target,
            )
            && !crate::trusted_device_items::is_authenticated_gfx942_wave64_scan_instance_v1(
                self.tcx,
                resolved,
                &self.expected_target,
            )
        {
            let caller_identity = self.instance_identity(*caller);
            self.reachable_unsafe_calls
                .entry(caller_identity)
                .or_default()
                .insert(format!("{callee_path} [{ty}]"));
        }
        let marked_contract = crate::device_ffi::contract_assertion_for_def(self.tcx, *def_id)
            .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?;
        if marked_contract.is_some() {
            let instance = resolved;
            let contract =
                crate::device_ffi::contract_for_instance(self.tcx, instance, &self.expected_target)
                    .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?
                    .ok_or_else(|| {
                        self.reachable_error(
                            caller,
                            "resolved device FFI declaration lost its compiler marker",
                            Some(self.tcx.def_path_str(*def_id)),
                        )
                    })?;
            let declaration =
                crate::device_ffi::collected_declaration(self.tcx, instance, contract.clone());
            if let Some(existing) = self.ffi_declarations.iter().find(|entry| {
                entry.owner.def_path_hash == declaration.owner.def_path_hash
                    && entry.owner.concrete_instance_symbol
                        == declaration.owner.concrete_instance_symbol
            }) {
                if existing.contract != contract {
                    return Err(self.reachable_error(
                        caller,
                        "reachable device FFI marker disagrees with its collected declaration",
                        Some(self.tcx.def_path_str(*def_id)),
                    ));
                }
            } else {
                crate::device_ffi::enforce_contract_bound(self.ffi_declarations.len() + 1)
                    .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?;
                self.ffi_declarations.push(declaration);
            }
            if contract.direction == crate::device_ffi::DeviceFfiDirection::Import {
                self.reachable_ffi_imports.insert(contract.id);
                if self.verbose {
                    eprintln!(
                        "[collector] external device import: {} -> {} ({})",
                        self.tcx.def_path_str(instance.def_id()),
                        contract.symbol,
                        contract.id.to_hex(),
                    );
                }
                return Ok(());
            }
        }

        // Diagnostic-item terminals and the separate exact normalized-call
        // rules below require matching recipes in the sole semantic importer.
        if let Some(rule) = crate::production_semantic_terminal_v1::classify(self.tcx, *def_id) {
            self.terminal_census
                .record(rule, &mut self.closure_work)
                .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?;
            if self.verbose {
                eprintln!(
                    "[collector] stopping at trusted device item {}",
                    self.tcx.def_path_str(*def_id)
                );
            }
            return Ok(());
        }

        if normalized_safe_core_shift.is_some() {
            // Only the exact safe core Item is summarized. Nested unchecked
            // calls are not collector boundaries and retain their refusals.
            self.closure_work
                .charge(1)
                .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?;
            return Ok(());
        }

        if crate::production_primitive_from_v1::check_primitive_from_v1(
            self.tcx,
            resolved,
            crate::production_primitive_from_v1::PrimitiveFromStageV1::Collector,
            self.closure_work.limits(),
            &mut |amount| self.closure_work.charge(amount),
        )
        .map_err(|error| {
            self.reachable_error(
                caller,
                &error.to_string(),
                Some(self.instance_label(resolved)),
            )
        })?
        .is_some()
        {
            return Ok(());
        }

        if normalized_intrinsic.is_some() {
            if self.verbose {
                eprintln!(
                    "[collector] stopping at normalized rustc intrinsic {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return Ok(());
        }

        match self.should_collect_from_crate(*def_id) {
            CollectDecision::Collect => {}
            CollectDecision::Forbidden {
                crate_name,
                fn_path,
            } => {
                return Err(self.reachable_error(
                    caller,
                    &format!(
                        "device code reached forbidden crate `{crate_name}`; device-reachable functions must avoid `std`"
                    ),
                    Some(fn_path),
                ));
            }
        }

        let identity = self.instance_identity(resolved);
        let caller_identity = self.instance_identity(*caller);

        if !is_fully_monomorphized(self.tcx, resolved) {
            return Err(self.reachable_error(
                caller,
                "direct device callee did not resolve to a fully monomorphized instance",
                Some(self.instance_label(resolved)),
            ));
        }

        // Repeat the workload-neutral device registry check for the concrete
        // implementation selected by rustc, after normalized-call boundaries.
        if let Some(rule) =
            crate::production_semantic_terminal_v1::classify(self.tcx, resolved.def_id())
        {
            self.terminal_census
                .record(rule, &mut self.closure_work)
                .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?;
            if self.verbose {
                eprintln!(
                    "[collector] stopping at registered semantic terminal {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return Ok(());
        }

        closure_flow_v1::record_call_v1(
            &mut self.call_edges,
            self.tcx,
            (*caller, resolved),
            body,
            block,
            &mut self.closure_work,
        )
        .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?;
        if self.seen.contains(&identity) {
            return Ok(());
        }

        let closure_shim = crate::closure_profile_v1::authenticate_once_shim_v1(self.tcx, resolved)
            .map_err(|error| self.reachable_error(caller, &error.to_string(), None))?
            .is_some();
        if !matches!(resolved.def, InstanceKind::Item(_)) && !closure_shim {
            return Err(self.reachable_error(
                caller,
                &format!(
                    "[FE2O3-FFI-CALL003] rustc-generated callable instance `{:?}` is not traversable under the V1 device graph policy",
                    resolved.def
                ),
                Some(self.instance_label(resolved)),
            ));
        }

        // Trait calls name the trait method in MIR, so classification must be
        // repeated against the exact concrete implementation selected by
        // rustc. The classifier authenticates the implementation through the
        // diagnostic-item-marked device type and the exact lang-item trait.
        if let Some(rejection) =
            crate::trusted_device_items::rejected_provider(self.tcx, resolved.def_id())
        {
            return Err(self.reachable_error(
                caller,
                &format!(
                    "trusted-provider rejection: diagnostic item `{}` is defined by `{}` but is not bound to the reviewed `{}` compilation unit: {}",
                    rejection.marker,
                    self.tcx.def_path_str(resolved.def_id()),
                    rejection.expected_provider_crate,
                    rejection.reason,
                ),
                None,
            ));
        }

        if !closure_shim && !self.tcx.is_mir_available(resolved.def_id()) {
            return Err(self.reachable_error(
                caller,
                "MIR is unavailable for a device-reachable item; compile the dependency with encoded MIR (for example, an inline Rust definition) or keep the call out of device code",
                Some(self.instance_label(resolved)),
            ));
        }

        if !closure_shim && self.is_unreachable_body(resolved.def_id()) {
            return Err(self.reachable_error(
                caller,
                "device code reaches a panic path",
                Some(self.instance_label(resolved)),
            ));
        }

        let name = self.fqdn(resolved.def_id());
        let export_name = self.compute_export_name(&name, resolved);

        if self.verbose {
            eprintln!("[collector] callee: {name} -> {export_name}");
        }

        let inserted = self.mark_seen(identity.clone())?;
        debug_assert!(inserted);
        self.call_chains.insert(
            identity,
            CallChainLink {
                predecessor: Some(caller_identity),
                label: self.instance_label(resolved),
            },
        );
        self.worklist.push_back(CollectedFunction {
            instance: resolved,
            role: CollectedFunctionRole::InternalHelper,
            export_name,
            logical_name: None,
            generated_host_contract_identity: None,
            kernel_binding: None,
            frontend_contract: None,
            reference_effect_binding: None,
            reference_instance: None,
            dead_branches: None,
            closure_observation: None,
            kernel_context_contract: None,
        });
        Ok(())
    }

    fn instance_identity(
        &self,
        instance: Instance<'tcx>,
    ) -> crate::device_ffi::DeviceFfiInstanceIdentity {
        crate::device_ffi::stable_instance_identity(self.tcx, instance)
    }

    fn instance_label(&self, instance: Instance<'tcx>) -> String {
        format!(
            "{} [{}]",
            self.fqdn(instance.def_id()),
            self.tcx.symbol_name(instance).name
        )
    }

    fn call_chain(&self, instance: &Instance<'tcx>) -> Vec<String> {
        let identity = self.instance_identity(*instance);
        let chain = reconstruct_call_chain(&self.call_chains, &identity);
        if chain.is_empty() {
            vec![self.instance_label(*instance)]
        } else {
            chain
        }
    }

    fn reachable_error(
        &self,
        caller: &Instance<'tcx>,
        reason: &str,
        callee: Option<String>,
    ) -> CollectError {
        let mut chain = self.call_chain(caller);
        if let Some(callee) = callee {
            chain.push(callee);
        }
        CollectError {
            message: format!(
                "fe2o3 device collection rejected a reachable call: {reason}; reachable call chain: {}",
                chain.join(" -> ")
            ),
        }
    }

    fn should_collect_from_crate(&self, def_id: DefId) -> CollectDecision {
        if def_id.krate == LOCAL_CRATE {
            return CollectDecision::Collect;
        }

        let crate_name = self.tcx.crate_name(def_id.krate);
        let crate_name = crate_name.as_str();
        let path = self.tcx.def_path_str(def_id);

        if path.contains(reserved_fe2o3_symbols::KERNEL_PREFIX) {
            return CollectDecision::Collect;
        }

        if crate_name == "std" {
            return CollectDecision::Forbidden {
                crate_name: crate_name.to_string(),
                fn_path: path,
            };
        }

        CollectDecision::Collect
    }

    fn fqdn(&self, def_id: DefId) -> String {
        let path = self.tcx.def_path_str(def_id);
        if def_id.krate == LOCAL_CRATE {
            format!("{}::{}", self.tcx.crate_name(LOCAL_CRATE), path)
        } else {
            path
        }
    }

    fn compute_export_name(&mut self, name: &str, instance: Instance<'tcx>) -> String {
        let has_generic_args = !instance.args.is_empty();
        let has_invalid_chars = name
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'));

        let simple = name.replace("::", "__");
        if has_generic_args || has_invalid_chars || self.used_export_names.contains(&simple) {
            let symbol = self.tcx.symbol_name(instance).name.to_string();
            let sanitized = sanitize_symbol_name(&symbol);
            self.used_export_names.insert(sanitized.clone());
            sanitized
        } else {
            self.used_export_names.insert(simple.clone());
            simple
        }
    }

    fn is_unreachable_body(&self, def_id: DefId) -> bool {
        if !self.tcx.is_mir_available(def_id) {
            return false;
        }

        let mir = self.tcx.optimized_mir(def_id);
        if mir.basic_blocks.len() > 2 {
            return false;
        }

        for block in mir.basic_blocks.iter() {
            let Some(terminator) = &block.terminator else {
                continue;
            };
            match &terminator.kind {
                TerminatorKind::Call { func, .. } => {
                    if let Some(callee) = self.call_def_id(func) {
                        let path = self.tcx.def_path_str(callee);
                        if path.contains("::panicking::") || path.contains("::rt::panic") {
                            return true;
                        }
                    }
                }
                TerminatorKind::Unreachable => {}
                _ => return false,
            }
        }

        false
    }

    fn call_def_id(&self, func: &Operand<'tcx>) -> Option<DefId> {
        let Operand::Constant(const_op) = func else {
            return None;
        };
        let ty = const_op.const_.ty();
        if let TyKind::FnDef(def_id, _) = ty.kind() {
            Some(*def_id)
        } else {
            None
        }
    }
}

fn assembly_operand_bit(ty: rustc_middle::ty::Ty<'_>) -> Option<u16> {
    match ty.kind() {
        TyKind::Bool | TyKind::Char | TyKind::Int(_) | TyKind::Uint(_) | TyKind::FnDef(..) => {
            Some(ASSEMBLY_OPERAND_SGPR_V1)
        }
        TyKind::Float(_) => Some(ASSEMBLY_OPERAND_VGPR_V1),
        TyKind::Ref(..) | TyKind::RawPtr(..) | TyKind::FnPtr(..) => {
            Some(ASSEMBLY_OPERAND_ADDRESS_V1)
        }
        _ => None,
    }
}

fn frontend_assembly_option_bits(options: InlineAsmOptions) -> u16 {
    let mut bits = 0;
    for (option, frontend_bit) in [
        (InlineAsmOptions::NOMEM, ASSEMBLY_OPTION_NOMEM_V1),
        (InlineAsmOptions::READONLY, ASSEMBLY_OPTION_READONLY_V1),
        (InlineAsmOptions::PURE, ASSEMBLY_OPTION_PURE_V1),
        (
            InlineAsmOptions::PRESERVES_FLAGS,
            ASSEMBLY_OPTION_PRESERVES_FLAGS_V1,
        ),
        (InlineAsmOptions::NOSTACK, ASSEMBLY_OPTION_NOSTACK_V1),
    ] {
        if options.contains(option) {
            bits |= frontend_bit;
        }
    }
    bits
}

fn reconcile_frontend_contract(
    logical_name: &str,
    expected_target: &str,
    authenticated: Option<&AuthenticatedKernelFrontendContractV1>,
    observed: ObservedInlineAssemblyV1,
) -> Result<ReachableAssemblySummaryV1, CollectError> {
    let declared = authenticated.and_then(|contract| contract.contract.unsafe_assembly());
    if observed.blocks == 0 {
        if declared.is_some() {
            return Err(CollectError {
                message: format!(
                    "kernel `{logical_name}` declares unsafe assembly but no asm! block is reachable from its exact rustc instance"
                ),
            });
        }
        return Ok(ReachableAssemblySummaryV1::default());
    }

    let Some(declared) = declared else {
        return Err(CollectError {
            message: format!(
                "kernel `{logical_name}` reaches inline assembly without an authenticated unsafe_asm frontend contract"
            ),
        });
    };
    let target_arch = expected_target.split(':').next().unwrap_or(expected_target);
    if declared.target().canonical_name() != target_arch {
        return Err(CollectError {
            message: format!(
                "kernel `{logical_name}` unsafe-assembly target `{}` disagrees with compiler target `{expected_target}`",
                declared.target().canonical_name()
            ),
        });
    }
    if observed.operand_bits != declared.operand_bits() {
        return Err(CollectError {
            message: format!(
                "kernel `{logical_name}` unsafe-assembly operand declaration {:#x} disagrees with reachable MIR operands {:#x}",
                declared.operand_bits(),
                observed.operand_bits
            ),
        });
    }
    let [observed_options] = observed.option_sets.iter().copied().collect::<Vec<_>>()[..] else {
        return Err(CollectError {
            message: format!(
                "kernel `{logical_name}` reaches asm! blocks with different option sets that one V1 contract cannot represent"
            ),
        });
    };
    if observed_options != declared.option_bits() {
        return Err(CollectError {
            message: format!(
                "kernel `{logical_name}` unsafe-assembly option declaration {:#x} disagrees with reachable MIR options {observed_options:#x}",
                declared.option_bits()
            ),
        });
    }

    Ok(ReachableAssemblySummaryV1 {
        blocks: observed.blocks,
        operand_bits: observed.operand_bits,
        option_bits: observed_options,
    })
}

fn sanitize_symbol_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedKernelFrontendContractV1, CallChainLink, KernelRoot, ObservedInlineAssemblyV1,
        RegistrationError, RegistrationRecord, TypedArgumentListError, TypedArgumentListV1,
        general_typed_launch_v3, reconcile_frontend_contract, reconstruct_call_chain,
        root_scoped_call_chains, validate_registration_records as validate_records,
    };
    use fe2o3_artifacts::{
        BlockSize, DeclaredRustLayoutIdentity, DeclaredRustTypeIdentity, DigestBytes, Dimensions,
        TypeIdentity,
    };
    use fe2o3_kernel_descriptor::MAX_ARGUMENTS_PER_KERNEL;
    use fe2o3_rustc_front::{
        ASSEMBLY_OPERAND_SGPR_V1, ASSEMBLY_OPTION_NOMEM_V1, ASSEMBLY_OPTION_NOSTACK_V1,
        ASSEMBLY_OPTION_PRESERVES_FLAGS_V1, FrontendGridDimensionsV1, FrontendLaunchBoundsV1,
        FrontendUnsafeAssemblyDeclarationV1, FrontendUnsafeAssemblyTargetV1,
        FrontendWorkgroupDimensionsV1, KernelFrontendContractV1, KernelResourceContractV1,
    };
    use reserved_fe2o3_symbols::{
        GeneratedHostContractIdV3, KERNEL_PREFIX, KERNEL_REGISTRATION_KIND_KERNEL,
        KERNEL_REGISTRATION_KIND_TYPED_GENERAL_LAYOUT_V3, KERNEL_REGISTRATION_MAGIC,
        KERNEL_REGISTRATION_PREFIX, KERNEL_REGISTRATION_VERSION_V1, KERNEL_REGISTRATION_VERSION_V2,
        KERNEL_REGISTRATION_VERSION_V3, KernelBindingIdV1,
        TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3, derive_crate_binding_id_v1,
        derive_kernel_binding_id_v1, host_kernel_symbol_v1,
    };
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn predecessor_call_chains_reconstruct_without_per_node_prefix_copies() {
        let links = BTreeMap::from([
            (
                1_u8,
                CallChainLink {
                    predecessor: None,
                    label: "root".to_owned(),
                },
            ),
            (
                2,
                CallChainLink {
                    predecessor: Some(1),
                    label: "helper_a".to_owned(),
                },
            ),
            (
                3,
                CallChainLink {
                    predecessor: Some(2),
                    label: "helper_b".to_owned(),
                },
            ),
        ]);

        assert_eq!(
            reconstruct_call_chain(&links, &3),
            ["root", "helper_a", "helper_b"],
        );
        assert!(reconstruct_call_chain(&links, &9).is_empty());
    }

    #[test]
    fn source_safety_call_chains_are_reconstructed_per_root() {
        let edges = BTreeMap::from([
            (1_u8, BTreeMap::from([(3_u8, ())])),
            (2_u8, BTreeMap::from([(3_u8, ())])),
            (3_u8, BTreeMap::from([(4_u8, ())])),
        ]);
        let labels = BTreeMap::from([
            (1_u8, "ordinary_root".to_owned()),
            (2_u8, "low_level_root".to_owned()),
            (3_u8, "shared_helper".to_owned()),
            (4_u8, "unsafe_leaf".to_owned()),
        ]);

        let (ordinary, order) = root_scoped_call_chains(&edges, &labels, &1);
        assert_eq!(order, [1, 3, 4]);
        assert_eq!(
            reconstruct_call_chain(&ordinary, &4),
            ["ordinary_root", "shared_helper", "unsafe_leaf"],
        );

        let (low_level, order) = root_scoped_call_chains(&edges, &labels, &2);
        assert_eq!(order, [2, 3, 4]);
        assert_eq!(
            reconstruct_call_chain(&low_level, &4),
            ["low_level_root", "shared_helper", "unsafe_leaf"],
        );
    }

    fn type_identity(byte: u8) -> TypeIdentity {
        TypeIdentity::new(
            DeclaredRustTypeIdentity::from_untrusted_bytes(DigestBytes::from_bytes([byte; 32])),
            DeclaredRustLayoutIdentity::from_untrusted_bytes(DigestBytes::from_bytes(
                [byte.wrapping_add(1); 32],
            )),
        )
    }

    #[test]
    fn typed_argument_lists_are_owned_and_not_fixed_to_three_arguments() {
        let one = TypedArgumentListV1::new(vec![type_identity(1)]).unwrap();
        let two = TypedArgumentListV1::new(vec![type_identity(2), type_identity(3)]).unwrap();

        assert_eq!(one.len(), 1);
        assert_eq!(two.len(), 2);
        assert_ne!(one.as_slice(), two.as_slice());
    }

    #[test]
    fn typed_argument_lists_accept_empty_and_reject_oversized_collections() {
        let empty = TypedArgumentListV1::<TypeIdentity>::new(Vec::new()).unwrap();
        assert_eq!(empty.len(), 0);
        assert!(empty.as_slice().is_empty());
        assert_eq!(
            TypedArgumentListV1::new(vec![type_identity(7); MAX_ARGUMENTS_PER_KERNEL])
                .unwrap()
                .len(),
            MAX_ARGUMENTS_PER_KERNEL
        );
        assert!(matches!(
            TypedArgumentListV1::new(vec![type_identity(7); MAX_ARGUMENTS_PER_KERNEL + 1]),
            Err(TypedArgumentListError::TooMany { actual, maximum })
                if actual == MAX_ARGUMENTS_PER_KERNEL + 1
                    && maximum == MAX_ARGUMENTS_PER_KERNEL
        ));
    }

    fn validate_registration_records<T: Copy>(
        records: Vec<RegistrationRecord<T>>,
    ) -> Result<Vec<KernelRoot<T>>, RegistrationError> {
        let expected_crate_binding = records.iter().find_map(|record| record.crate_binding);
        validate_records(records, expected_crate_binding, Some("fixture"))
    }

    fn registration(
        path: &str,
        logical_name: &str,
        export_name: &str,
        target: u8,
    ) -> RegistrationRecord<u8> {
        let target_symbol = format!("{KERNEL_PREFIX}{export_name}");
        let module_path = path.rsplit_once("::").map_or("", |(module, _)| module);
        let target_identity = if module_path.is_empty() {
            target_symbol.clone()
        } else {
            format!("{module_path}::{target_symbol}")
        };
        RegistrationRecord {
            registration_path: path.to_string(),
            item_name: format!("{KERNEL_REGISTRATION_PREFIX}{logical_name}"),
            magic: KERNEL_REGISTRATION_MAGIC,
            version: KERNEL_REGISTRATION_VERSION_V1,
            kind: KERNEL_REGISTRATION_KIND_KERNEL,
            logical_name: logical_name.to_string(),
            export_name: export_name.to_string(),
            crate_binding: None,
            kernel_binding: None,
            profile_tag: None,
            generated_host_contract_identity: None,
            target_crate_name: "fixture".to_owned(),
            target_symbol,
            target_identity,
            target,
        }
    }

    fn general_typed_registration(
        path: &str,
        logical_name: &str,
        export_name: &str,
        target: u8,
    ) -> RegistrationRecord<u8> {
        let mut registration = registration(path, logical_name, export_name, target);
        let crate_binding = derive_crate_binding_id_v1("fixture", ["metadata"]);
        let kernel_binding = derive_kernel_binding_id_v1(
            crate_binding,
            TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3,
            logical_name,
            export_name,
        );
        registration.version = KERNEL_REGISTRATION_VERSION_V3;
        registration.kind = KERNEL_REGISTRATION_KIND_TYPED_GENERAL_LAYOUT_V3;
        registration.crate_binding = Some(crate_binding);
        registration.kernel_binding = Some(kernel_binding);
        registration.profile_tag = Some(TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3.to_owned());
        registration.generated_host_contract_identity =
            Some(GeneratedHostContractIdV3::from_bytes([0x63; 32]));
        registration.target_symbol = host_kernel_symbol_v1(kernel_binding);
        registration
    }

    #[test]
    fn genuine_v1_registration_becomes_a_kernel_root() {
        let roots = validate_registration_records(vec![registration(
            "crate::__fe2o3_kernel_registration_plain",
            "plain",
            "plain",
            7,
        )])
        .unwrap();

        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].target, 7);
        assert_eq!(roots[0].logical_name, "plain");
        assert_eq!(roots[0].export_name, "plain");
        assert_eq!(roots[0].generated_host_contract_identity, None);
    }

    #[test]
    fn general_v3_registration_carries_the_complete_profile_identity() {
        let registration = general_typed_registration(
            "crate::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            3,
        );
        let roots = validate_registration_records(vec![registration]).unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].target, 3);
        assert_eq!(
            roots[0]
                .generated_host_contract_identity
                .unwrap()
                .as_bytes(),
            [0x63; 32],
        );
    }

    #[test]
    fn authenticated_kernel_owner_rejects_owner_profile_and_build_mutations() {
        let registration = general_typed_registration(
            "general_genuine::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            3,
        );
        let expected_crate_binding = registration.crate_binding.unwrap();

        let error =
            validate_records(vec![registration.clone()], None, Some("fixture")).unwrap_err();
        assert!(error.reason.contains("no rustc session crate binding"));

        let error = validate_records(
            vec![registration.clone()],
            Some(expected_crate_binding),
            None,
        )
        .unwrap_err();
        assert!(error.reason.contains("no rustc session crate identity"));

        let mut wrong_crate = registration.clone();
        wrong_crate.target_crate_name = "other".to_owned();
        let error = validate_records(
            vec![wrong_crate],
            Some(expected_crate_binding),
            Some("fixture"),
        )
        .unwrap_err();
        assert!(error.reason.contains("cross-crate generic typed kernels"));

        let mut wrong_module = registration.clone();
        wrong_module.target_identity = format!("spoof::{}", wrong_module.target_symbol);
        let error = validate_registration_records(vec![wrong_module]).unwrap_err();
        assert!(
            error
                .reason
                .contains("registered target module \"spoof\" disagrees")
        );

        let mut wrong_logical = registration.clone();
        wrong_logical.logical_name = "beta".to_owned();
        let error = validate_registration_records(vec![wrong_logical]).unwrap_err();
        assert!(
            error
                .reason
                .contains("inconsistent with logical name `beta`")
        );

        let mut wrong_export = registration.clone();
        wrong_export.export_name = "beta".to_owned();
        let error = validate_registration_records(vec![wrong_export]).unwrap_err();
        assert!(error.reason.contains("disagrees with derived binding"));

        let mut wrong_profile = registration.clone();
        wrong_profile.profile_tag = Some("wrong-profile".to_owned());
        let error = validate_registration_records(vec![wrong_profile]).unwrap_err();
        assert!(
            error
                .reason
                .contains("not the canonical general typed profile")
        );

        let mut wrong_crate_binding = registration.clone();
        wrong_crate_binding.crate_binding = Some(derive_crate_binding_id_v1("other", ["metadata"]));
        let error = validate_records(
            vec![wrong_crate_binding],
            Some(expected_crate_binding),
            Some("fixture"),
        )
        .unwrap_err();
        assert!(
            error
                .reason
                .contains("disagrees with rustc session binding")
        );

        let mut wrong_kernel_binding = registration.clone();
        wrong_kernel_binding.kernel_binding = Some(KernelBindingIdV1::from_bytes([0x99; 32]));
        let error = validate_registration_records(vec![wrong_kernel_binding]).unwrap_err();
        assert!(error.reason.contains("disagrees with derived binding"));

        let mut wrong_symbol = registration.clone();
        wrong_symbol.target_symbol = "spoofed-symbol".to_owned();
        let error = validate_registration_records(vec![wrong_symbol]).unwrap_err();
        assert!(error.reason.contains("target symbol `spoofed-symbol`"));

        let mut duplicate = general_typed_registration(
            "general_genuine::__fe2o3_kernel_registration_beta",
            "beta",
            "beta",
            4,
        );
        duplicate.target_identity = registration.target_identity.clone();
        let error = validate_registration_records(vec![registration, duplicate]).unwrap_err();
        assert!(error.reason.contains("duplicate target identity"));
    }

    #[test]
    fn general_v3_rejects_profile_binding_identity_and_version_mutations() {
        let registration = general_typed_registration(
            "crate::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            3,
        );

        let mut wrong_profile = registration.clone();
        wrong_profile.profile_tag = Some("fe2o3.manifest-derived-scalar-slice.v2".to_owned());
        assert!(
            validate_registration_records(vec![wrong_profile])
                .unwrap_err()
                .reason
                .contains("not the canonical general typed profile")
        );

        let mut wrong_binding = registration.clone();
        wrong_binding.kernel_binding = Some(KernelBindingIdV1::from_bytes([0x99; 32]));
        assert!(
            validate_registration_records(vec![wrong_binding])
                .unwrap_err()
                .reason
                .contains("disagrees with derived binding")
        );

        let mut old_version = registration.clone();
        old_version.version = KERNEL_REGISTRATION_VERSION_V2;
        assert!(
            validate_registration_records(vec![old_version])
                .unwrap_err()
                .reason
                .contains("registration version 2 has been removed")
        );

        let mut old_kind = registration;
        old_kind.kind = KERNEL_REGISTRATION_KIND_TYPED_GENERAL_LAYOUT_V3 + 1;
        assert!(
            validate_registration_records(vec![old_kind])
                .unwrap_err()
                .reason
                .contains("version 3 is reserved")
        );
    }

    #[test]
    fn general_v3_roots_are_ordered_independently_of_registration_discovery() {
        let zeta = general_typed_registration(
            "crate::__fe2o3_kernel_registration_zeta",
            "zeta",
            "zeta",
            2,
        );
        let alpha = general_typed_registration(
            "crate::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            1,
        );
        let roots = validate_registration_records(vec![zeta, alpha]).unwrap();
        assert_eq!(
            roots
                .iter()
                .map(|root| root.logical_name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
    }

    #[test]
    fn generic_typed_registration_identity_and_host_symbol_fail_closed() {
        let typed = general_typed_registration(
            "crate::__fe2o3_kernel_registration_typed",
            "typed",
            "typed",
            7,
        );
        let wrong_crate = derive_crate_binding_id_v1("other", ["metadata"]);
        let error =
            validate_records(vec![typed.clone()], Some(wrong_crate), Some("fixture")).unwrap_err();
        assert!(
            error
                .reason
                .contains("disagrees with rustc session binding")
        );

        let mut wrong_kernel = typed.clone();
        wrong_kernel.kernel_binding = Some(derive_kernel_binding_id_v1(
            wrong_kernel.crate_binding.unwrap(),
            TYPED_GENERAL_RUSTC_LAYOUT_PROFILE_TAG_V3,
            "different",
            "different",
        ));
        let error = validate_registration_records(vec![wrong_kernel]).unwrap_err();
        assert!(error.reason.contains("disagrees with derived binding"));

        let mut logical_host_symbol = typed;
        logical_host_symbol.target_symbol = format!("{KERNEL_PREFIX}typed");
        let error = validate_registration_records(vec![logical_host_symbol]).unwrap_err();
        assert!(error.reason.contains("target symbol"));
        assert!(error.reason.contains("inconsistent with export name"));
    }

    #[test]
    fn kernel_prefix_spoof_without_registration_is_not_a_root() {
        let roots = validate_registration_records::<u8>(Vec::new()).unwrap();
        assert!(roots.is_empty());
    }

    #[test]
    fn general_typed_launch_accepts_required_only_exact_profiles_and_rejects_ambiguity() {
        let authenticated =
            |required: Option<[u32; 3]>, maximum: Option<[u32; 3]>, occupancy: Option<u16>| {
                let launch = FrontendLaunchBoundsV1::new(
                    required.map(|value| FrontendWorkgroupDimensionsV1::new(value).unwrap()),
                    maximum.map(|value| FrontendWorkgroupDimensionsV1::new(value).unwrap()),
                    occupancy,
                )
                .unwrap();
                AuthenticatedKernelFrontendContractV1::for_test(
                    KernelFrontendContractV1::new(Some(launch), None).unwrap(),
                )
            };

        let default = general_typed_launch_v3(None, "registration").unwrap();
        assert_eq!(
            default.block_size(),
            BlockSize::Exact(Dimensions::new(256, 1, 1).unwrap())
        );
        for dimensions in [
            [1, 1, 1],
            [3, 1, 1],
            [64, 1, 1],
            [65, 1, 1],
            [255, 1, 1],
            [256, 1, 1],
        ] {
            let frontend = authenticated(Some(dimensions), None, None);
            let launch = general_typed_launch_v3(Some(&frontend), "registration").unwrap();
            assert_eq!(
                launch.block_size(),
                BlockSize::Exact(
                    Dimensions::new(dimensions[0], dimensions[1], dimensions[2]).unwrap()
                )
            );

            let frontend = authenticated(Some(dimensions), Some(dimensions), None);
            let launch = general_typed_launch_v3(Some(&frontend), "registration").unwrap();
            assert_eq!(
                launch.block_size(),
                BlockSize::Exact(
                    Dimensions::new(dimensions[0], dimensions[1], dimensions[2]).unwrap()
                )
            );
        }

        let finite_launch = FrontendLaunchBoundsV1::new_with_max_grid(
            Some(FrontendWorkgroupDimensionsV1::new([64, 1, 1]).unwrap()),
            Some(FrontendWorkgroupDimensionsV1::new([64, 1, 1]).unwrap()),
            None,
            Some(FrontendGridDimensionsV1::new([1, 1, 1]).unwrap()),
        )
        .unwrap();
        let finite_frontend = AuthenticatedKernelFrontendContractV1::for_test(
            KernelFrontendContractV1::new(Some(finite_launch), None).unwrap(),
        );
        assert_eq!(
            general_typed_launch_v3(Some(&finite_frontend), "registration")
                .unwrap()
                .max_grid(),
            Dimensions::new(1, 1, 1).unwrap()
        );

        let resource_frontend = AuthenticatedKernelFrontendContractV1::for_test_with_resource(
            KernelFrontendContractV1::new(Some(finite_launch), None).unwrap(),
            KernelResourceContractV1::new(256, 1_024).unwrap(),
        );
        let resource_launch =
            general_typed_launch_v3(Some(&resource_frontend), "registration").unwrap();
        assert_eq!(resource_launch.static_shared_memory_bytes(), 256);
        assert_eq!(resource_launch.max_dynamic_shared_memory_bytes(), 1_024);
        assert_ne!(resource_frontend.resource_canonical_bytes(), Some(&[][..]),);

        for frontend in [
            authenticated(None, Some([64, 1, 1]), None),
            authenticated(Some([64, 1, 1]), Some([256, 1, 1]), None),
            authenticated(Some([256, 1, 1]), Some([256, 1, 1]), Some(2)),
            authenticated(Some([32, 2, 1]), None, None),
            authenticated(Some([257, 1, 1]), None, None),
        ] {
            assert!(general_typed_launch_v3(Some(&frontend), "registration").is_err());
        }
    }

    #[test]
    fn reachable_assembly_reconciliation_is_exact_and_target_bound() {
        let option_bits = ASSEMBLY_OPTION_NOMEM_V1
            | ASSEMBLY_OPTION_NOSTACK_V1
            | ASSEMBLY_OPTION_PRESERVES_FLAGS_V1;
        let contract = KernelFrontendContractV1::new(
            None,
            Some(
                FrontendUnsafeAssemblyDeclarationV1::new(
                    FrontendUnsafeAssemblyTargetV1::AmdGpuGfx942,
                    ASSEMBLY_OPERAND_SGPR_V1,
                    option_bits,
                    0,
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let authenticated = AuthenticatedKernelFrontendContractV1::for_test(contract);
        let observed = ObservedInlineAssemblyV1 {
            blocks: 2,
            operand_bits: ASSEMBLY_OPERAND_SGPR_V1,
            option_sets: BTreeSet::from([option_bits]),
        };

        let summary = reconcile_frontend_contract(
            "kernel",
            "gfx942:xnack-",
            Some(&authenticated),
            observed.clone(),
        )
        .unwrap();
        assert_eq!(summary.blocks, 2);
        assert_eq!(summary.operand_bits, ASSEMBLY_OPERAND_SGPR_V1);
        assert_eq!(summary.option_bits, option_bits);

        for (target, contract, observation, expected) in [
            (
                "gfx1100",
                Some(&authenticated),
                observed.clone(),
                "disagrees with compiler target",
            ),
            (
                "gfx942",
                None,
                observed.clone(),
                "without an authenticated unsafe_asm",
            ),
            (
                "gfx942",
                Some(&authenticated),
                ObservedInlineAssemblyV1 {
                    operand_bits: fe2o3_rustc_front::ASSEMBLY_OPERAND_VGPR_V1,
                    ..observed.clone()
                },
                "operand declaration",
            ),
            (
                "gfx942",
                Some(&authenticated),
                ObservedInlineAssemblyV1 {
                    option_sets: BTreeSet::from([option_bits, ASSEMBLY_OPTION_NOMEM_V1]),
                    ..observed
                },
                "different option sets",
            ),
        ] {
            let error =
                reconcile_frontend_contract("kernel", target, contract, observation).unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "missing `{expected}` in {error}"
            );
        }
    }

    #[test]
    fn malformed_magic_and_unknown_version_or_kind_fail_closed() {
        let base = registration("crate::__fe2o3_kernel_registration_bad", "bad", "bad", 1);

        let mut malformed = base.clone();
        malformed.magic ^= 1;
        assert!(
            validate_registration_records(vec![malformed])
                .unwrap_err()
                .reason
                .contains("does not match registration magic")
        );

        let mut unknown_version = base.clone();
        unknown_version.version = KERNEL_REGISTRATION_VERSION_V3 + 1;
        assert!(
            validate_registration_records(vec![unknown_version])
                .unwrap_err()
                .reason
                .contains("unknown registration version")
        );

        for kind in [
            0,
            KERNEL_REGISTRATION_KIND_TYPED_GENERAL_LAYOUT_V3 + 1,
            u16::MAX,
        ] {
            let mut unknown_kind = base.clone();
            unknown_kind.kind = kind;
            assert_eq!(
                validate_registration_records(vec![unknown_kind])
                    .unwrap_err()
                    .reason,
                format!("unknown registration kind {kind}")
            );
        }

        let mut obsolete_typed = base;
        obsolete_typed.version = KERNEL_REGISTRATION_VERSION_V2;
        obsolete_typed.kind = KERNEL_REGISTRATION_KIND_KERNEL;
        assert!(
            validate_registration_records(vec![obsolete_typed])
                .unwrap_err()
                .reason
                .contains("registration version 2 has been removed")
        );
    }

    #[test]
    fn duplicate_logical_and_export_names_fail_closed() {
        let logical_error = validate_registration_records(vec![
            registration(
                "crate::a::__fe2o3_kernel_registration_same",
                "same",
                "alpha",
                1,
            ),
            registration(
                "crate::b::__fe2o3_kernel_registration_same",
                "same",
                "beta",
                2,
            ),
        ])
        .unwrap_err();
        assert!(
            logical_error
                .reason
                .contains("duplicate logical name `same`")
        );

        let export_error = validate_registration_records(vec![
            registration(
                "crate::__fe2o3_kernel_registration_alpha",
                "alpha",
                "same",
                1,
            ),
            registration("crate::__fe2o3_kernel_registration_beta", "beta", "same", 2),
        ])
        .unwrap_err();
        assert!(export_error.reason.contains("duplicate export name `same`"));

        let typed_duplicate = general_typed_registration(
            "crate::b::__fe2o3_kernel_registration_same",
            "same",
            "typed",
            2,
        );
        let cross_kind_error = validate_registration_records(vec![
            registration(
                "crate::a::__fe2o3_kernel_registration_same",
                "same",
                "basic",
                1,
            ),
            typed_duplicate,
        ])
        .unwrap_err();
        assert!(
            cross_kind_error
                .reason
                .contains("duplicate logical name `same`")
        );
    }

    #[test]
    fn duplicate_target_identities_fail_closed() {
        let mut alpha = registration(
            "crate::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            1,
        );
        let mut beta = registration("crate::__fe2o3_kernel_registration_beta", "beta", "beta", 2);
        alpha.target_identity = "crate::same-target".to_string();
        beta.target_identity = "crate::same-target".to_string();

        let error = validate_registration_records(vec![alpha, beta]).unwrap_err();
        assert!(
            error
                .reason
                .contains("duplicate target identity `crate::same-target`")
        );
    }

    #[test]
    fn inconsistent_item_or_target_associations_fail_closed() {
        let mut item = registration(
            "crate::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            1,
        );
        item.item_name = format!("{KERNEL_REGISTRATION_PREFIX}beta");
        assert!(
            validate_registration_records(vec![item])
                .unwrap_err()
                .reason
                .contains("inconsistent with logical name")
        );

        let mut target = registration(
            "crate::__fe2o3_kernel_registration_alpha",
            "alpha",
            "alpha",
            1,
        );
        target.target_symbol = format!("{KERNEL_PREFIX}beta");
        assert!(
            validate_registration_records(vec![target])
                .unwrap_err()
                .reason
                .contains("inconsistent with export name")
        );
    }

    #[test]
    fn multiple_kernels_are_sorted_deterministically() {
        let roots = validate_registration_records(vec![
            registration("crate::__fe2o3_kernel_registration_zeta", "zeta", "zeta", 2),
            registration(
                "crate::__fe2o3_kernel_registration_alpha",
                "alpha",
                "alpha",
                1,
            ),
        ])
        .unwrap();

        assert_eq!(
            roots
                .iter()
                .map(|root| (root.logical_name.as_str(), root.target))
                .collect::<Vec<_>>(),
            vec![("alpha", 1), ("zeta", 2)]
        );
    }

    #[test]
    fn production_registration_surface_accepts_v1_and_generic_v3() {
        let ordinary = validate_registration_records(vec![registration(
            "fixture::__fe2o3_kernel_plain",
            "plain",
            "plain",
            1,
        )])
        .unwrap();
        let general = validate_registration_records(vec![general_typed_registration(
            "fixture::__fe2o3_kernel_general",
            "general",
            "general",
            3,
        )])
        .unwrap();

        assert!(ordinary[0].generated_host_contract_identity.is_none());
        assert!(general[0].generated_host_contract_identity.is_some());
    }

    #[test]
    fn production_entry_receives_target_without_reading_process_state() {
        let source = include_str!("collector.rs");
        let production = source
            .split_once("pub(crate) fn collect_authenticated_kernel_closure_v1")
            .unwrap()
            .1
            .split_once("fn collect_device_functions")
            .unwrap()
            .0;
        for forbidden in ["std::env", "QUALIFICATION_ORACLE_ENV", "FE2O3_TARGET"] {
            assert!(
                !production.contains(forbidden),
                "production collection entry contains hidden input {forbidden:?}"
            );
        }
        assert!(production.contains("RetainedProductionTargetV1"));
        assert!(!production.contains("authenticate_before_collection"));
    }
}

#[path = "collector/typed_argument_retained_storage_v1.rs"]
mod typed_argument_retained_storage_v1;

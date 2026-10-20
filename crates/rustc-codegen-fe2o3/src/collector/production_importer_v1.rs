//! Sole consuming boundary from production rustc collection to semantic MIR.

use std::collections::BTreeMap;
use std::fmt;

use dialect_amdgcn::DeviceValueDiagnosticItem;
use fe2o3_mir_model::semantic_mir_v1::{
    AdmittedInertSemanticMirV1, HARD_MAX_FUNCTIONS_V1, HARD_MAX_ROOTS_V1,
    InertSemanticMirRequestV1, SemanticAbiPassModeV1, SemanticBackendReprV1,
    SemanticBf16ConversionKindV1, SemanticCallableDeclV1, SemanticCallableIdV1, SemanticCanonAbiV1,
    SemanticCompilerIntrinsicIdentityV1, SemanticCompilerIntrinsicOperationV1,
    SemanticDisjointIndexSpaceV1, SemanticExternAbiV1, SemanticF32MathFunctionV1,
    SemanticFunctionAbiV1, SemanticFunctionIdV1, SemanticFunctionIdentityV1,
    SemanticFunctionRoleV1, SemanticGfx950LdsTransposeFormatV1, SemanticKernelBindingIdentityV1,
    SemanticKernelEntryV1, SemanticKernelLaunchBoundsV1, SemanticKernelResourceContractV1,
    SemanticKernelSourceContractV1, SemanticLinkSymbolV1, SemanticMfmaAccumulatorContractV1,
    SemanticMfmaAccumulatorDistributionV1, SemanticMfmaOperandContractV1,
    SemanticMfmaOperandRoleV1, SemanticMfmaProfileV1, SemanticMfmaRegisterDistributionV1,
    SemanticMfmaStorageLayoutV1, SemanticMirErrorV1, SemanticMirLimitsV1, SemanticMirResourceV1,
    SemanticNonBodyCallableBindingV1, SemanticReachableAssemblyV1, SemanticScalarTypeV1,
    SemanticSubgroupReductionKindV1, SemanticTargetDataLayoutV1, SemanticTypeDeclV1,
    SemanticTypeIdV1, SemanticTypeLayoutDetailsV1, SemanticTypeShapeV1,
    SemanticUnsafeAssemblyDeclarationV1, SemanticUnsafeAssemblyTargetV1,
    SemanticWorkgroupDimensionsV1, SemanticWorkgroupPipelineEventV1, SemanticWorkgroupScanKindV1,
    SemanticWriteOnlyDisjointWriteKindV1,
};
use rustc_middle::ty::{FloatTy, GenericArgKind, Instance, IntTy, Ty, TyCtxt, TyKind, UintTy};
use rustc_span::{Symbol, sym};

use super::{
    AuthenticatedCollectedKernelClosureV1, AuthenticatedProductionRootV1, CollectedFunctionRole,
    CollectionResult,
};
use crate::production_semantic_body_v1::{
    ExpectedFunctionCommitmentV29, ProductionSemanticBlockBindingV1, ProductionSemanticBodyErrorV1,
    ProductionSemanticBodyInputV1, ProductionSemanticBodyRequestOwnerV1,
    ProductionSemanticCallableOwnerEntryV1, ProductionSemanticDirectCallBindingV1,
    ProductionSemanticFunctionExportV1, ProductionSemanticFunctionIdentitiesV1,
    ProductionSemanticLocalBindingV1, ProductionSemanticNormalizedRustcIntrinsicRecipeV1,
    ProductionSemanticTerminalExpansionRecipeV1, ProductionSemanticTypeBindingV1,
    construct_production_semantic_body_v1,
};
use crate::production_semantic_fn_abi_v1::{
    ConstructedSemanticFunctionAbisV1, ProductionSemanticFnAbiErrorV1,
    construct_production_semantic_fn_abis_v1,
};
use crate::production_semantic_terminal_v1::{
    ProductionBf16ConversionV1, ProductionTerminalExpansionV1,
};
use crate::production_semantic_types_v1::{
    ProductionSemanticTypeErrorV1, construct_production_semantic_types_atomic_v41,
    construct_production_semantic_types_nominal_v35, construct_production_semantic_types_v1,
};
use crate::production_target_v1::ProductionTargetErrorV1;
use crate::rustc_semantic_adapter_v1::{
    SemanticIdentityDigestV1, canonical_function_identities_v1, canonical_target_layout_v1,
};
use crate::rustc_semantic_plan_v1::{
    DebugSourceCaptureRequestV2, ProductionSemanticPreflightErrorV1,
    ProductionSemanticPreflightPlanV1, RetainedSemanticFunctionProducerV1, SourceClosureWorkV1,
    build_production_semantic_preflight_plan_with_work_v1,
};
use crate::trusted_device_items::{self, TrustedDeviceItem};

#[path = "execution_terminal_descriptors_v29.rs"]
mod execution_terminals;
#[path = "production_saturating_terminal_v1.rs"]
mod saturating_terminal;
#[path = "production_wave64_shuffle_terminal_v1.rs"]
mod wave64_shuffle_terminal;
#[cfg(test)]
pub(crate) use wave64_shuffle_terminal::check_wave64_descriptor_mutations_v1;

use fe2o3_compiler_lineage::RUSTC_IDENTITY_INVENTORY_DOMAIN_V2 as IDENTITY_INVENTORY_DOMAIN_V2;
#[cfg(test)]
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V1: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v1";
#[cfg(test)]
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V2: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v2";
#[cfg(test)]
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V3: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v3";
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V4: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v4";
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V5: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v5";
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V7: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v7";
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V8: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v8";
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V9: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v9";
const PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V6: &[u8] =
    b"fe2o3/semantic-mir/production-compiler-intrinsic/v6";

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

#[derive(Debug)]
pub(crate) enum ProductionSemanticImportErrorV1 {
    Target(ProductionTargetErrorV1),
    RootCustodyMismatch,
    ContextCustody(super::CollectError),
    ClosureAdmission(crate::closure_profile_v1::ClosureProfileErrorV1),
    ReferenceCustody(crate::reference_effect_v1::ReferenceBindingErrorV1),
    LimitExceeded {
        resource: SemanticMirResourceV1,
        actual: u64,
        maximum: u64,
    },
    FunctionIdentityCollision,
    RootIdentityMismatch,
    Preflight(Box<ProductionSemanticPreflightErrorV1>),
    TypeConstruction(Box<ProductionSemanticTypeErrorV1>),
    FunctionAbiConstruction(Box<ProductionSemanticFnAbiErrorV1>),
    BodyConstruction(Box<ProductionSemanticBodyErrorV1>),
    SemanticSchema(SemanticMirErrorV1),
    LineageTranscriptTooLarge {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    TargetNeutralLoweringPending {
        functions: usize,
        callables: usize,
        rustc_identity_inventory_sha256: [u8; 32],
        rustc_preflight_plan_sha256: [u8; 32],
        semantic_sha256: [u8; 32],
    },
}

impl fmt::Display for ProductionSemanticImportErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target(error) => write!(formatter, "semantic import target rejection: {error}"),
            Self::RootCustodyMismatch => formatter.write_str(
                "semantic importer rejected collector root custody before MIR construction",
            ),
            Self::ContextCustody(error) => write!(formatter, "semantic import context custody rejected: {error}"),
            Self::ClosureAdmission(error) => write!(
                formatter, "semantic import closure admission rejected: {error}"
            ),
            Self::ReferenceCustody(error) => write!(
                formatter, "semantic import reference custody rejected: {error}"
            ),
            Self::LimitExceeded {
                resource,
                actual,
                maximum,
            } => write!(
                formatter,
                "semantic importer rejected {resource:?} count {actual} before semantic record allocation; maximum is {maximum}",
            ),
            Self::FunctionIdentityCollision => formatter.write_str(
                "semantic importer independently derived a duplicate canonical function identity",
            ),
            Self::RootIdentityMismatch => formatter.write_str(
                "semantic importer could not bind independently derived roots to unique collected functions",
            ),
            Self::Preflight(error) => write!(formatter, "semantic importer {error}"),
            Self::TypeConstruction(error) => {
                write!(formatter, "semantic importer rejected semantic type construction: {error}")
            }
            Self::FunctionAbiConstruction(error) => write!(
                formatter,
                "semantic importer rejected semantic function ABI construction: {error}",
            ),
            Self::BodyConstruction(error) => {
                write!(formatter, "semantic importer rejected semantic body construction: {error}")
            }
            Self::SemanticSchema(error) => {
                write!(formatter, "semantic importer rejected complete semantic MIR: {error}")
            }
            Self::LineageTranscriptTooLarge {
                field,
                actual,
                maximum,
            } => write!(
                formatter,
                "semantic importer {field} transcript uses {actual} bytes, exceeding the lineage receipt maximum {maximum}"
            ),
            Self::TargetNeutralLoweringPending {
                functions,
                callables,
                rustc_identity_inventory_sha256,
                rustc_preflight_plan_sha256,
                semantic_sha256,
            } => write!(
                formatter,
                "semantic importer authenticated rustc identity inventory {} and bounded preflight plan {}, then admitted one complete semantic MIR request with {functions} function(s), {callables} callable(s), and canonical identity {}; an owner-held Pliron locator graph was recursively verified for exact semantic equivalence; target-neutral lowering remains pending; no fallback or artifact emission was entered",
                crate::encode_hex(rustc_identity_inventory_sha256),
                crate::encode_hex(rustc_preflight_plan_sha256),
                crate::encode_hex(semantic_sha256),
            ),
        }
    }
}
impl std::error::Error for ProductionSemanticImportErrorV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Target(error) => Some(error),
            Self::ContextCustody(error) => Some(error),
            Self::Preflight(error) => Some(error.as_ref()),
            Self::TypeConstruction(error) => Some(error.as_ref()),
            Self::FunctionAbiConstruction(error) => Some(error.as_ref()),
            Self::BodyConstruction(error) => Some(error.as_ref()),
            Self::SemanticSchema(error) => Some(error),
            Self::ClosureAdmission(error) => Some(error),
            Self::ReferenceCustody(error) => Some(error),
            Self::RootCustodyMismatch
            | Self::LimitExceeded { .. }
            | Self::LineageTranscriptTooLarge { .. }
            | Self::FunctionIdentityCollision
            | Self::RootIdentityMismatch => None,
            Self::TargetNeutralLoweringPending { .. } => None,
        }
    }
}

#[derive(Debug)]
struct ProductionSemanticIdentityInventoryV1<'tcx> {
    functions: Box<[RetainedSemanticFunctionProducerV1<'tcx>]>,
    roots: Box<[SemanticFunctionIdV1]>,
    sha256: [u8; 32],
    canonical_transcript: Box<[u8]>,
    original_root_associations:
        Option<super::reference_custody_v1::RetainedOriginalRootAssociationsV1>,
}

/// Move-only rustc-produced identity-inventory evidence retained by the
/// production transaction. Public hashes cannot construct this owner.
pub(crate) struct AuthenticatedRustcIdentityInventoryV3 {
    sha256: [u8; 32],
    canonical_transcript: Box<[u8]>,
    original_root_associations:
        Option<super::reference_custody_v1::RetainedOriginalRootAssociationsV1>,
}

impl AuthenticatedRustcIdentityInventoryV3 {
    pub(crate) fn original_root_associations(
        &self,
    ) -> Option<&super::reference_custody_v1::RetainedOriginalRootAssociationsV1> {
        self.original_root_associations.as_ref()
    }

    pub(crate) const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub(crate) fn canonical_transcript(&self) -> &[u8] {
        &self.canonical_transcript
    }
}

/// Move-only rustc-produced preflight-plan evidence retained by the
/// production transaction. Public hashes cannot construct this owner.
pub(crate) struct AuthenticatedRustcPreflightPlanV3 {
    sha256: [u8; 32],
    rustc_identity_inventory_sha256: [u8; 32],
    canonical_transcript: Box<[u8]>,
}

impl AuthenticatedRustcPreflightPlanV3 {
    pub(crate) const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub(crate) const fn rustc_identity_inventory_sha256(&self) -> [u8; 32] {
        self.rustc_identity_inventory_sha256
    }

    pub(crate) fn canonical_transcript(&self) -> &[u8] {
        &self.canonical_transcript
    }
}

/// Same-session semantic import result transferred into the sole production
/// transaction. Keeping the custody axes named prevents accidental positional
/// substitution as the import surface grows.
pub(crate) struct ConstructedProductionSemanticMirV1 {
    pub(crate) semantic_mir: AdmittedInertSemanticMirV1,
    pub(crate) context_entries: super::RetainedContextEntriesV29,
    pub(crate) rustc_identity_inventory: AuthenticatedRustcIdentityInventoryV3,
    pub(crate) rustc_preflight_plan: AuthenticatedRustcPreflightPlanV3,
    pub(crate) rustc_target: crate::production_target_v1::AuthenticatedProductionTargetV1,
    pub(crate) reference_effect_bindings:
        crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
    pub(crate) debug_source_files: Box<[fe2o3_kernel_ir::DebugSourceMapFileV1]>,
    pub(crate) debug_source_scopes:
        Box<[crate::rustc_semantic_plan_v1::RetainedDebugSourceScopeV2]>,
    pub(crate) debug_source_variables:
        Box<[crate::rustc_semantic_plan_v1::RetainedDebugSourceVariableV2]>,
    pub(crate) debug_capture_gap: Option<fe2o3_kernel_ir::ProductionSemanticDebugProducerGapV1>,
}

pub(crate) fn construct_production_semantic_mir_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<ConstructedProductionSemanticMirV1, ProductionSemanticImportErrorV1> {
    construct_production_semantic_mir_with_nominal_v35(tcx, closure, debug_source_capture, false)
}

pub(crate) fn construct_production_semantic_mir_with_enrollment_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
    nominal: bool,
    loan: &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
) -> Result<ConstructedProductionSemanticMirV1, ProductionSemanticImportErrorV1> {
    let (constructed, completion) =
        construct_production_semantic_mir_with_policy_and_enrollment_v1(
            tcx,
            closure,
            debug_source_capture,
            nominal,
            SourceImportPolicyV1::Singleton,
            Some(loan),
        )?;
    match completion {
        SourceImportCompletionV1::Singleton => Ok(constructed),
        _ => Err(body_owner_table_mismatch_v1(
            "enrolled singleton import returned a foreign completion profile",
        )),
    }
}

pub(crate) fn construct_production_semantic_mir_nominal_v35<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<ConstructedProductionSemanticMirV1, ProductionSemanticImportErrorV1> {
    construct_production_semantic_mir_with_nominal_v35(tcx, closure, debug_source_capture, true)
}

/// Explicit original-rustc atomic storage import; this is not a default selector
/// and grants no pointer, execution, optimization or target authority.
pub(crate) fn construct_production_semantic_mir_atomic_v41<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<ConstructedProductionSemanticMirV1, ProductionSemanticImportErrorV1> {
    let (constructed, completion) = construct_production_semantic_mir_with_policy_v1(
        tcx,
        closure,
        debug_source_capture,
        false,
        SourceImportPolicyV1::AtomicV41,
    )?;
    match completion {
        SourceImportCompletionV1::AtomicV41 => Ok(constructed),
        _ => Err(body_owner_table_mismatch_v1(
            "atomic V41 import returned a foreign completion profile",
        )),
    }
}

/// The source-owned pipeline selects its exact grammar before admission.
/// This reuses the live rustc producers and all original custody checks; it
/// does not reencode an admitted legacy owner or infer a policy from a kernel.
pub(crate) fn construct_production_semantic_mir_source_owned_v29<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<ConstructedProductionSemanticMirV1, ProductionSemanticImportErrorV1> {
    let (constructed, completion) = construct_production_semantic_mir_with_policy_v1(
        tcx,
        closure,
        debug_source_capture,
        false,
        SourceImportPolicyV1::SourceOwnedV29,
    )?;
    match completion {
        SourceImportCompletionV1::SourceOwnedV29 => Ok(constructed),
        _ => Err(body_owner_table_mismatch_v1(
            "source-owned import returned a foreign completion profile",
        )),
    }
}

fn construct_production_semantic_mir_with_nominal_v35<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
    nominal: bool,
) -> Result<ConstructedProductionSemanticMirV1, ProductionSemanticImportErrorV1> {
    let (constructed, completion) = construct_production_semantic_mir_with_policy_v1(
        tcx,
        closure,
        debug_source_capture,
        nominal,
        SourceImportPolicyV1::Singleton,
    )?;
    match completion {
        SourceImportCompletionV1::Singleton => Ok(constructed),
        _ => Err(body_owner_table_mismatch_v1(
            "singleton import returned composition source custody",
        )),
    }
}

/// Private move-only custody of the explicitly selected live composition import.
/// All marker/call definitions have been authenticated and drained before this
/// value exists. Decoding MIR32 bytes cannot construct it.
pub(crate) struct AuthenticatedOrderedCompositionMirV1<'tcx> {
    constructed: ConstructedProductionSemanticMirV1,
    source_seed: crate::production_ordered_composition_source_v1::AuthenticatedOrderedCompositionSourceSeedV1<'tcx>,
}
impl<'tcx> AuthenticatedOrderedCompositionMirV1<'tcx> {
    pub(crate) fn into_parts(self) -> (
        ConstructedProductionSemanticMirV1,
        crate::production_ordered_composition_source_v1::AuthenticatedOrderedCompositionSourceSeedV1<'tcx>,
    ){
        (self.constructed, self.source_seed)
    }
}

#[derive(Clone, Copy)]
enum SourceImportPolicyV1 {
    Singleton,
    AtomicV41,
    SourceOwnedV29,
    Composition,
    Bf16Inspection,
    Bf16TileValues,
}
// Exact private completion modes: no inferred or fallback source custody.
enum SourceImportCompletionV1<'tcx> {
    Singleton,
    AtomicV41,
    SourceOwnedV29,
    Composition(crate::production_ordered_composition_source_v1::AuthenticatedOrderedCompositionSourceSeedV1<'tcx>),
    Bf16Inspection(crate::production_tiled_region_source_v1::AuthenticatedBf16MfmaSourceSeedV1<'tcx>),
    Bf16TileValues(crate::production_bf16_tile_values_source_v1::AuthenticatedBf16TileValuesSourceSeedV1<'tcx>),
}
pub(crate) struct AuthenticatedBf16MfmaInspectionMirV1<'tcx> {
    constructed: ConstructedProductionSemanticMirV1,
    source_seed: crate::production_tiled_region_source_v1::AuthenticatedBf16MfmaSourceSeedV1<'tcx>,
}
impl<'tcx> AuthenticatedBf16MfmaInspectionMirV1<'tcx> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        ConstructedProductionSemanticMirV1,
        crate::production_tiled_region_source_v1::AuthenticatedBf16MfmaSourceSeedV1<'tcx>,
    ) {
        (self.constructed, self.source_seed)
    }
}
pub(crate) fn construct_production_semantic_mir_bf16_inspection_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<AuthenticatedBf16MfmaInspectionMirV1<'tcx>, ProductionSemanticImportErrorV1> {
    let (constructed, completion) = construct_production_semantic_mir_with_policy_v1(
        tcx,
        closure,
        debug_source_capture,
        false,
        SourceImportPolicyV1::Bf16Inspection,
    )?;
    match completion {
        SourceImportCompletionV1::Bf16Inspection(source_seed) => {
            Ok(AuthenticatedBf16MfmaInspectionMirV1 {
                constructed,
                source_seed,
            })
        }
        _ => Err(body_owner_table_mismatch_v1(
            "BF16 inspection import returned foreign source custody",
        )),
    }
}

// Dedicated move-only source custody, not a permissiveness flag or P0 fallback.
pub(crate) struct AuthenticatedBf16TileValuesMirV1<'tcx> {
    constructed: ConstructedProductionSemanticMirV1,
    source_seed:
        crate::production_bf16_tile_values_source_v1::AuthenticatedBf16TileValuesSourceSeedV1<'tcx>,
}
impl<'tcx> AuthenticatedBf16TileValuesMirV1<'tcx> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        ConstructedProductionSemanticMirV1,
        crate::production_bf16_tile_values_source_v1::AuthenticatedBf16TileValuesSourceSeedV1<'tcx>,
    ) {
        (self.constructed, self.source_seed)
    }
}
pub(crate) fn construct_production_semantic_mir_bf16_tile_values_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<AuthenticatedBf16TileValuesMirV1<'tcx>, ProductionSemanticImportErrorV1> {
    let (constructed, completion) = construct_production_semantic_mir_with_policy_v1(
        tcx,
        closure,
        debug_source_capture,
        false,
        SourceImportPolicyV1::Bf16TileValues,
    )?;
    match completion {
        SourceImportCompletionV1::Bf16TileValues(source_seed) => {
            Ok(AuthenticatedBf16TileValuesMirV1 {
                constructed,
                source_seed,
            })
        }
        _ => Err(body_owner_table_mismatch_v1(
            "BF16 tile-values import returned foreign source custody",
        )),
    }
}

pub(crate) fn construct_production_semantic_mir_ordered_composition_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
) -> Result<AuthenticatedOrderedCompositionMirV1<'tcx>, ProductionSemanticImportErrorV1> {
    let (constructed, completion) = construct_production_semantic_mir_with_policy_v1(
        tcx,
        closure,
        debug_source_capture,
        false,
        SourceImportPolicyV1::Composition,
    )?;
    match completion {
        SourceImportCompletionV1::Composition(source_seed) => {
            Ok(AuthenticatedOrderedCompositionMirV1 {
                constructed,
                source_seed,
            })
        }
        _ => Err(body_owner_table_mismatch_v1(
            "composition import returned singleton source custody",
        )),
    }
}

fn construct_production_semantic_mir_with_policy_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
    nominal: bool,
    ordered_policy: SourceImportPolicyV1,
) -> Result<
    (
        ConstructedProductionSemanticMirV1,
        SourceImportCompletionV1<'tcx>,
    ),
    ProductionSemanticImportErrorV1,
> {
    construct_production_semantic_mir_with_policy_and_enrollment_v1(
        tcx,
        closure,
        debug_source_capture,
        nominal,
        ordered_policy,
        None,
    )
}

fn construct_production_semantic_mir_with_policy_and_enrollment_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    closure: AuthenticatedCollectedKernelClosureV1<'tcx>,
    debug_source_capture: DebugSourceCaptureRequestV2,
    nominal: bool,
    ordered_policy: SourceImportPolicyV1,
    loan: Option<&crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>>,
) -> Result<
    (
        ConstructedProductionSemanticMirV1,
        SourceImportCompletionV1<'tcx>,
    ),
    ProductionSemanticImportErrorV1,
> {
    let AuthenticatedCollectedKernelClosureV1 {
        target,
        collection,
        roots,
        context_entries,
        mut closure_flow,
        terminal_census: _,
    } = closure;
    let target = match target.authenticate_import_session(tcx) {
        Ok(target) => target,
        Err(error) => return Err(ProductionSemanticImportErrorV1::Target(error)),
    };
    let retained_roots = roots
        .iter()
        .map(|root| (root.instance, root.role, root.export_name.as_str()));
    let independently_observed_roots = collection
        .functions
        .iter()
        .filter(|function| {
            matches!(
                function.role,
                super::CollectedFunctionRole::KernelEntry
                    | super::CollectedFunctionRole::DeviceFfiExport
            )
        })
        .map(|function| {
            (
                function.instance,
                function.role,
                function.export_name.as_str(),
            )
        });
    if !exact_ordered_axes_match(retained_roots, independently_observed_roots) {
        return Err(ProductionSemanticImportErrorV1::RootCustodyMismatch);
    }
    let context_entries = context_entries
        .validate_for_import_v1(tcx, &collection)
        .map_err(ProductionSemanticImportErrorV1::ContextCustody)?;
    let (reference_effect_bindings, original_root_capture) = closure_flow
        .rederive_reference_bindings_and_inventory_v1(tcx, &collection, loan)
        .map_err(ProductionSemanticImportErrorV1::ReferenceCustody)?;
    let (identity_inventory, closure_work) = build_identity_inventory_v1(
        tcx,
        &target,
        &collection,
        &roots,
        closure_flow,
        original_root_capture,
        loan,
    )?;
    require_lineage_transcript_bound_v3(
        "rustc identity inventory",
        &identity_inventory.canonical_transcript,
    )?;

    let ProductionSemanticIdentityInventoryV1 {
        functions,
        roots,
        sha256: rustc_identity_inventory_sha256,
        canonical_transcript: rustc_identity_inventory_transcript,
        original_root_associations,
    } = identity_inventory;
    let mut plan = match build_production_semantic_preflight_plan_with_work_v1(
        tcx,
        canonical_target_layout_v1(target.rustc_layout()),
        functions,
        roots,
        rustc_identity_inventory_sha256,
        debug_source_capture,
        (Some(context_entries), closure_work),
    ) {
        Ok(plan) => plan,
        Err(error) => return Err(ProductionSemanticImportErrorV1::Preflight(Box::new(error))),
    };
    require_lineage_transcript_bound_v3("rustc preflight plan", plan.canonical_transcript())?;
    let constructed_types = if matches!(ordered_policy, SourceImportPolicyV1::AtomicV41) {
        construct_production_semantic_types_atomic_v41(tcx, plan.type_producers())
    } else if nominal {
        construct_production_semantic_types_nominal_v35(tcx, plan.type_producers())
    } else {
        construct_production_semantic_types_v1(tcx, plan.type_producers())
    };
    let semantic_types = match constructed_types {
        Ok(types) => types,
        Err(error) => {
            return Err(ProductionSemanticImportErrorV1::TypeConstruction(Box::new(
                error,
            )));
        }
    };
    let semantic_function_abis = match construct_production_semantic_fn_abis_v1(
        tcx,
        plan.function_abi_producers(),
        plan.type_producers(),
    ) {
        Ok(abis) => abis,
        Err(error) => {
            return Err(ProductionSemanticImportErrorV1::FunctionAbiConstruction(
                Box::new(error),
            ));
        }
    };
    let terminal_abi_producers = plan
        .terminal_producers()
        .iter()
        .map(|terminal| terminal.abi.clone())
        .collect::<Vec<_>>();
    let semantic_terminal_abis = match construct_production_semantic_fn_abis_v1(
        tcx,
        &terminal_abi_producers,
        plan.type_producers(),
    ) {
        Ok(abis) => abis,
        Err(error) => {
            return Err(ProductionSemanticImportErrorV1::FunctionAbiConstruction(
                Box::new(error),
            ));
        }
    };
    let (
        semantic_mir,
        context_entries,
        ordered_completion,
        bf16_completion,
        tile_values_completion,
    ) = construct_complete_request_v1(
        tcx,
        canonical_target_layout_v1(target.rustc_layout()),
        &mut plan,
        semantic_types.into_records(),
        semantic_function_abis,
        semantic_terminal_abis,
        ordered_policy,
    )?;
    use crate::production_bf16_tile_values_source_v1::Bf16TileValuesImportCompletionV1 as TileValues;
    use crate::production_ordered_composition_source_v1::OrderedSourceImportCompletionV1 as Ordered;
    use crate::production_tiled_region_source_v1::Bf16MfmaImportCompletionV1 as Bf16;
    let completion = match (
        ordered_policy,
        ordered_completion,
        bf16_completion,
        tile_values_completion,
    ) {
        (
            SourceImportPolicyV1::Singleton,
            Ordered::Singleton,
            Bf16::Disabled,
            TileValues::Disabled,
        ) => SourceImportCompletionV1::Singleton,
        (
            SourceImportPolicyV1::AtomicV41,
            Ordered::Singleton,
            Bf16::Disabled,
            TileValues::Disabled,
        ) => SourceImportCompletionV1::AtomicV41,
        (
            SourceImportPolicyV1::SourceOwnedV29,
            Ordered::Singleton,
            Bf16::Disabled,
            TileValues::Disabled,
        ) => SourceImportCompletionV1::SourceOwnedV29,
        (
            SourceImportPolicyV1::Composition,
            Ordered::Composition(seed),
            Bf16::Disabled,
            TileValues::Disabled,
        ) => SourceImportCompletionV1::Composition(seed),
        (
            SourceImportPolicyV1::Bf16Inspection,
            Ordered::Singleton,
            Bf16::Inspected(seed),
            TileValues::Disabled,
        ) => SourceImportCompletionV1::Bf16Inspection(seed),
        (
            SourceImportPolicyV1::Bf16TileValues,
            Ordered::Singleton,
            Bf16::Disabled,
            TileValues::Inspected(seed),
        ) => SourceImportCompletionV1::Bf16TileValues(seed),
        _ => {
            return Err(body_owner_table_mismatch_v1(
                "source import completion modes differ",
            ));
        }
    };
    #[cfg(test)]
    super::semantic_import_observation_v1_tests::observe_actual(tcx, &semantic_mir);
    let (
        rustc_preflight_plan_sha256,
        rustc_preflight_plan_transcript,
        debug_source_files,
        debug_source_scopes,
        debug_source_variables,
        debug_capture_gap,
    ) = plan
        .into_identity_transcript_and_debug_files()
        .map_err(|error| ProductionSemanticImportErrorV1::Preflight(Box::new(error)))?;
    drop(collection);
    Ok((
        ConstructedProductionSemanticMirV1 {
            semantic_mir,
            context_entries,
            rustc_identity_inventory: AuthenticatedRustcIdentityInventoryV3 {
                sha256: rustc_identity_inventory_sha256,
                canonical_transcript: rustc_identity_inventory_transcript,
                original_root_associations,
            },
            rustc_preflight_plan: AuthenticatedRustcPreflightPlanV3 {
                sha256: rustc_preflight_plan_sha256,
                rustc_identity_inventory_sha256,
                canonical_transcript: rustc_preflight_plan_transcript,
            },
            rustc_target: target,
            reference_effect_bindings,
            debug_source_files,
            debug_source_scopes,
            debug_source_variables,
            debug_capture_gap,
        },
        completion,
    ))
}

fn require_lineage_transcript_bound_v3(
    field: &'static str,
    transcript: &[u8],
) -> Result<(), ProductionSemanticImportErrorV1> {
    if transcript.len() > fe2o3_compiler_lineage::MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3 {
        Err(ProductionSemanticImportErrorV1::LineageTranscriptTooLarge {
            field,
            actual: transcript.len(),
            maximum: fe2o3_compiler_lineage::MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3,
        })
    } else {
        Ok(())
    }
}

fn construct_complete_request_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: SemanticTargetDataLayoutV1,
    plan: &mut ProductionSemanticPreflightPlanV1<'tcx>,
    types: Vec<SemanticTypeDeclV1>,
    function_abis: ConstructedSemanticFunctionAbisV1,
    terminal_abis: ConstructedSemanticFunctionAbisV1,
    ordered_policy: SourceImportPolicyV1,
) -> Result<
    (
        AdmittedInertSemanticMirV1,
        super::RetainedContextEntriesV29,
        crate::production_ordered_composition_source_v1::OrderedSourceImportCompletionV1<'tcx>,
        crate::production_tiled_region_source_v1::Bf16MfmaImportCompletionV1<'tcx>,
        crate::production_bf16_tile_values_source_v1::Bf16TileValuesImportCompletionV1<'tcx>,
    ),
    ProductionSemanticImportErrorV1,
> {
    let function_abis = function_abis.into_records();
    let terminal_abis = terminal_abis.into_records();
    if function_abis.len() != plan.function_producers().len() {
        return Err(body_owner_table_mismatch_v1(
            "function ABI producer cardinality",
        ));
    }
    if terminal_abis.len() != plan.terminal_producers().len() {
        return Err(body_owner_table_mismatch_v1(
            "terminal ABI producer cardinality",
        ));
    }
    if plan.body_producers().len() != plan.function_producers().len() {
        return Err(body_owner_table_mismatch_v1(
            "function body producer cardinality",
        ));
    }

    let type_bindings = plan
        .type_producers()
        .iter()
        .enumerate()
        .map(|(index, producer)| {
            let index = u32::try_from(index)
                .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
            Ok(ProductionSemanticTypeBindingV1::new(
                producer.ty,
                fe2o3_mir_model::semantic_mir_v1::SemanticTypeIdV1::from_index(index),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let function_count = u32::try_from(plan.function_producers().len())
        .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
    let inline_sources =
        crate::production_inline_source_occurrences_v30::InlineSourceOccurrencesV30::from_plan(
            plan,
        )
        .map_err(body_owner_table_mismatch_v1)?;
    let ordered_sources =
        crate::production_ordered_source_occurrences_v31::OrderedSourceOccurrencesV31::from_plan(
            tcx, plan,
        )
        .map_err(body_owner_table_mismatch_v1)?;
    let contains_physical_entry = plan
        .terminal_producers()
        .iter()
        .any(|terminal| crate::production_physical_entry_call_v37::is_physical(terminal.expansion));
    let contains_physical_lds_exchange = plan.terminal_producers().iter().any(|terminal| {
        crate::production_physical_lds_exchange_call_v39::is_physical(terminal.expansion)
    });
    let contains_physical_global_copy = plan.terminal_producers().iter().any(|terminal| {
        crate::production_physical_global_copy_call_v38::is_physical(terminal.expansion)
    });
    let contains_complete_body = plan
        .terminal_producers()
        .iter()
        .any(|terminal| terminal.expansion == ProductionTerminalExpansionV1::Gfx942CompleteBodyE32);
    let contains_ordered_program = plan.terminal_producers().iter().any(|terminal| {
        terminal.expansion == ProductionTerminalExpansionV1::Gfx942OrderedProgramE32
    });
    let mut context_entries = plan.take_context_entries_v29();
    let mut body_owner = build_body_request_owner_v1(
        plan,
        types.len(),
        function_count,
        !context_entries.is_empty(),
    )?
    .with_inline_sources_v30(inline_sources)
    .with_ordered_sources_v31(ordered_sources);
    match ordered_policy {
        SourceImportPolicyV1::Singleton
        | SourceImportPolicyV1::AtomicV41
        | SourceImportPolicyV1::SourceOwnedV29
        | SourceImportPolicyV1::Bf16Inspection
        | SourceImportPolicyV1::Bf16TileValues => {
            let sources = if contains_ordered_program {
                crate::production_ordered_program_source_occurrences_v32::OrderedProgramSourceOccurrencesV32::from_plan(tcx, plan)
                    .map_err(body_owner_table_mismatch_v1)?
            } else {
                Default::default()
            };
            body_owner = body_owner.with_program_sources_v32(sources);
        }
        SourceImportPolicyV1::Composition => {
            if !contains_ordered_program {
                return Err(body_owner_table_mismatch_v1(
                    "ordered composition marker roster missing",
                ));
            }
            body_owner
                .prepare_ordered_composition_sources_v1(tcx, plan, &types, &function_abis)
                .map_err(|error| {
                    ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error))
                })?;
        }
    }
    if matches!(ordered_policy, SourceImportPolicyV1::Bf16Inspection) {
        body_owner
            .prepare_bf16_inspection_source_v1(tcx, plan)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    if matches!(ordered_policy, SourceImportPolicyV1::Bf16TileValues) {
        body_owner
            .prepare_bf16_tile_values_source_v1(tcx, plan)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    if contains_physical_entry {
        body_owner
            .prepare_physical_entry_annotations_v37(tcx, plan)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    if contains_physical_lds_exchange {
        body_owner
            .prepare_physical_lds_exchange_annotations_v39(tcx, plan)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    if contains_physical_global_copy {
        body_owner
            .prepare_physical_global_copy_annotations_v38(tcx, plan)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    if contains_complete_body {
        body_owner
            .prepare_complete_body_annotation_vnext(tcx, plan)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    let mut callables = (0..function_count)
        .map(|index| SemanticCallableDeclV1::defined(SemanticFunctionIdV1::from_index(index)))
        .collect::<Vec<_>>();
    for (index, (terminal, abi)) in plan
        .terminal_producers()
        .iter()
        .zip(&terminal_abis)
        .enumerate()
    {
        let operation =
            terminal_operation_v1(tcx, terminal.instance, terminal.expansion, abi, &types)?;
        let (domain, schema) = compiler_intrinsic_identity_schema_v1(terminal.expansion);
        let mut digest = SemanticIdentityDigestV1::new(domain);
        digest.field(terminal.identities.function().as_bytes());
        digest.field(abi.identity().as_bytes());
        digest.field(&[terminal_operation_tag_for_schema_v1(
            terminal.expansion,
            schema,
        )]);
        digest.field(
            &u32::try_from(index)
                .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?
                .to_le_bytes(),
        );
        callables.push(SemanticCallableDeclV1::CompilerIntrinsic {
            binding: SemanticNonBodyCallableBindingV1::new(
                terminal.identities.function(),
                terminal.identities.item_definition(),
                terminal.identities.monomorphization(),
                terminal.identities.generic_type_arguments(),
                terminal.identities.const_generic_arguments(),
                terminal.source.provenance,
                abi.clone(),
            ),
            operation,
            operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256(digest.finish()),
        });
    }

    if !context_entries.is_empty() {
        body_owner
            .enable_workgroup_scope_custody_v29(tcx, target, &types, &callables)
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    let mut functions = Vec::new();
    functions
        .try_reserve_exact(plan.function_producers().len())
        .map_err(|_| ProductionSemanticImportErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::Functions,
            actual: u64::try_from(plan.function_producers().len()).unwrap_or(u64::MAX),
            maximum: HARD_MAX_FUNCTIONS_V1,
        })?;
    for (index, ((function, body), abi)) in plan
        .function_producers()
        .iter()
        .zip(plan.body_producers())
        .zip(function_abis)
        .enumerate()
    {
        let function_id = SemanticFunctionIdV1::from_index(
            u32::try_from(index)
                .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?,
        );
        if body.function != function_id {
            return Err(body_owner_table_mismatch_v1("function body owner ordering"));
        }
        let local_bindings = body
            .locals
            .iter()
            .map(|local| {
                Ok(ProductionSemanticLocalBindingV1::new(
                    local.rustc_local,
                    *body
                        .raw_to_semantic_locals
                        .get(local.rustc_local as usize)
                        .ok_or(ProductionSemanticImportErrorV1::RootIdentityMismatch)?,
                    local.identity,
                    local.source.provenance,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let block_bindings = body
            .blocks
            .iter()
            .enumerate()
            .map(|(semantic, block)| {
                Ok(ProductionSemanticBlockBindingV1::new(
                    block.rustc_block,
                    fe2o3_mir_model::semantic_mir_v1::SemanticBlockIdV1::from_index(
                        u32::try_from(semantic)
                            .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?,
                    ),
                    block.identity,
                    block.source.provenance,
                    block
                        .statements
                        .iter()
                        .map(|source| source.provenance)
                        .collect(),
                    block.terminator.provenance,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let direct_calls = plan
            .direct_call_producers()
            .iter()
            .filter(|call| call.caller == function_id)
            .map(|call| {
                let callee = plan
                    .function_producers()
                    .get(call.callee.index() as usize)
                    .ok_or(ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
                Ok(ProductionSemanticDirectCallBindingV1::new(
                    call.caller,
                    call.block,
                    callee.instance,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let terminal_expansions = plan
            .terminal_expansion_producers()
            .iter()
            .filter(|recipe| recipe.caller == function_id)
            .map(|recipe| {
                Ok(ProductionSemanticTerminalExpansionRecipeV1::new(
                    recipe.caller,
                    recipe.block,
                    recipe.instance,
                    recipe.expansion,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let normalized_intrinsics = plan
            .normalized_intrinsic_producers()
            .iter()
            .filter(|recipe| recipe.caller == function_id)
            .map(|recipe| {
                ProductionSemanticNormalizedRustcIntrinsicRecipeV1::new(
                    recipe.caller,
                    recipe.block,
                    recipe.instance,
                    recipe.element_type,
                    recipe.operation,
                )
            })
            .collect::<Vec<_>>();
        functions.push(
            construct_production_semantic_body_v1(
                ProductionSemanticBodyInputV1 {
                    tcx,
                    instance: function.instance,
                    body: tcx.instance_mir(function.instance.def),
                    function: function_id,
                    identities: ProductionSemanticFunctionIdentitiesV1::new(
                        function.identities.function(),
                        function.identities.item_definition(),
                        function.identities.monomorphization(),
                        function.identities.generic_type_arguments(),
                        function.identities.const_generic_arguments(),
                    ),
                    role: semantic_function_role_v1(function.role),
                    export: semantic_function_export_v1(function)?,
                    source: body.source.provenance,
                    abi,
                    type_bindings: &type_bindings,
                    local_bindings: &local_bindings,
                    block_bindings: &block_bindings,
                    entry: body.entry,
                    direct_calls: &direct_calls,
                    terminal_expansions: &terminal_expansions,
                    normalized_intrinsics: &normalized_intrinsics,
                    context_entry: context_entries.remove(&function_id),
                },
                &mut body_owner,
            )
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?,
        );
    }

    body_owner
        .require_inline_sources_consumed_v30()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    body_owner
        .require_ordered_sources_consumed_v31()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    body_owner
        .require_program_sources_consumed_v32()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    body_owner
        .require_physical_entry_annotations_consumed_v37()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    body_owner
        .require_physical_lds_exchange_annotations_consumed_v39()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    body_owner
        .require_physical_global_copy_annotations_consumed_v38()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    body_owner
        .require_complete_body_annotation_consumed_vnext()
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    if !context_entries.is_empty() {
        return Err(body_owner_table_mismatch_v1("unused context root binding"));
    }
    let contains_execution_roles = types.iter().any(|ty| {
        matches!(
            ty.rust_type_kind(),
            fe2o3_mir_model::semantic_mir_v1::SemanticRustTypeKindV1::Execution(_)
        )
    });
    let semantic = InertSemanticMirRequestV1::new_with_callables(
        target,
        types,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        functions,
        callables,
        plan.roots().to_vec(),
    )
    .and_then(|request| {
        if matches!(ordered_policy, SourceImportPolicyV1::AtomicV41) {
            // An explicit original transaction selects this exact composition.
            // Sibling authoring/physical forms refuse; no marker-driven fallback.
            request.admit_exact_v41(SemanticMirLimitsV1::default())
        } else if matches!(ordered_policy, SourceImportPolicyV1::SourceOwnedV29) {
            // Fixed pipeline context, including ordinary scalar roots. Newer
            // sibling syntax must fail exact membership, never fall back.
            request.admit_exact_v29(SemanticMirLimitsV1::default())
        } else if contains_physical_lds_exchange {
            // Only actual authenticated V39 providers select this exact sibling grammar.
            request.admit_exact_v39(SemanticMirLimitsV1::default())
        } else if contains_physical_global_copy {
            // Only the actual authenticated V38 provider roster selects this sibling schema.
            request.admit_exact_v38(SemanticMirLimitsV1::default())
        } else if contains_physical_entry {
            // Chosen only by the actual authenticated provider roster, never raw bytes.
            request.admit_exact_v37(SemanticMirLimitsV1::default())
        } else if contains_complete_body {
            // Selected only from the authenticated retained terminal producer.
            // Exact V36 membership rejects old sibling authoring/execution roles;
            // it is not an ordinal maximum or a request-controlled default.
            request.admit_exact_v36(SemanticMirLimitsV1::default())
        } else if contains_ordered_program {
            // Separate V30-derived grammar. In particular this rejects old V31
            // regions and Execution carriers, rather than choosing by version order.
            request.admit_exact_v32(SemanticMirLimitsV1::default())
        } else if contains_execution_roles {
            // This preserves source types, not execution authority. The shared
            // materializer rejects V29 before ordinary aggregate/ZST erasure.
            request.admit_exact_v29(SemanticMirLimitsV1::default())
        } else {
            request.admit_current_production(SemanticMirLimitsV1::default())
        }
    })
    .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?;
    let bf16_completion = body_owner
        .complete_bf16_inspection_source_v1(&semantic)
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    let tile_values_completion = body_owner
        .complete_bf16_tile_values_source_v1(&semantic)
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    let ordered_completion = body_owner
        .complete_ordered_source_import_v1(&semantic)
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    let context_entries = body_owner
        .seal_context_entries(&semantic)
        .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    Ok((
        semantic,
        context_entries,
        ordered_completion,
        bf16_completion,
        tile_values_completion,
    ))
}

fn build_body_request_owner_v1<'tcx>(
    plan: &mut ProductionSemanticPreflightPlanV1<'tcx>,
    type_count: usize,
    function_count: u32,
    context_transaction: bool,
) -> Result<ProductionSemanticBodyRequestOwnerV1<'tcx>, ProductionSemanticImportErrorV1> {
    let callable_count = plan
        .function_producers()
        .len()
        .checked_add(plan.terminal_producers().len())
        .ok_or(ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(callable_count).map_err(|_| {
        ProductionSemanticImportErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::Callables,
            actual: u64::try_from(callable_count).unwrap_or(u64::MAX),
            maximum: SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::Callables),
        }
    })?;
    for (index, function) in plan.function_producers().iter().enumerate() {
        let callable = u32::try_from(index)
            .map(SemanticCallableIdV1::from_index)
            .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
        entries.push(ProductionSemanticCallableOwnerEntryV1::defined(
            function.instance,
            callable,
        ));
    }

    let mut terminal_instances = Vec::new();
    terminal_instances
        .try_reserve_exact(plan.terminal_producers().len())
        .map_err(|_| ProductionSemanticImportErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::Callables,
            actual: u64::try_from(callable_count).unwrap_or(u64::MAX),
            maximum: SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::Callables),
        })?;
    terminal_instances.resize(plan.terminal_producers().len(), None);
    for recipe in plan.terminal_expansion_producers() {
        let slot = terminal_instances
            .get_mut(recipe.terminal as usize)
            .ok_or_else(|| body_owner_table_mismatch_v1("terminal callable owner index"))?;
        let observed = (recipe.instance, recipe.expansion);
        match slot {
            Some(previous) if *previous != observed => {
                return Err(body_owner_table_mismatch_v1(
                    "terminal callable owner instance",
                ));
            }
            Some(_) => {}
            None => *slot = Some(observed),
        }
    }
    for (index, (terminal, observed)) in plan
        .terminal_producers()
        .iter()
        .zip(terminal_instances)
        .enumerate()
    {
        let (instance, expansion) = observed
            .ok_or_else(|| body_owner_table_mismatch_v1("terminal callable owner completeness"))?;
        if expansion != terminal.expansion {
            return Err(body_owner_table_mismatch_v1(
                "terminal callable owner expansion",
            ));
        }
        let terminal_index = u32::try_from(index)
            .map_err(|_| ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
        let callable = function_count
            .checked_add(terminal_index)
            .map(SemanticCallableIdV1::from_index)
            .ok_or(ProductionSemanticImportErrorV1::RootIdentityMismatch)?;
        entries.push(ProductionSemanticCallableOwnerEntryV1::terminal(
            instance, expansion, callable,
        ));
    }

    let mut owner = ProductionSemanticBodyRequestOwnerV1::with_preflight_work(
        plan.take_construction_work()
            .map_err(|error| ProductionSemanticImportErrorV1::Preflight(Box::new(error)))?,
        type_count,
        &entries,
    )
    .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    if context_transaction {
        if plan.function_producers().len() != function_count as usize
            || plan.body_producers().len() != function_count as usize
        {
            return Err(body_owner_table_mismatch_v1(
                "function commitment preflight bodies",
            ));
        }
        owner
            .enable_function_commitments_v29(
                function_count as usize,
                plan.function_producers()
                    .iter()
                    .zip(plan.body_producers())
                    .map(|(function, body)| {
                        ExpectedFunctionCommitmentV29::new(
                            function.instance,
                            body.function,
                            ProductionSemanticFunctionIdentitiesV1::new(
                                function.identities.function(),
                                function.identities.item_definition(),
                                function.identities.monomorphization(),
                                function.identities.generic_type_arguments(),
                                function.identities.const_generic_arguments(),
                            ),
                            semantic_function_role_v1(function.role),
                            body.source.provenance,
                        )
                    }),
            )
            .map_err(|error| ProductionSemanticImportErrorV1::BodyConstruction(Box::new(error)))?;
    }
    Ok(owner)
}

fn body_owner_table_mismatch_v1(table: &'static str) -> ProductionSemanticImportErrorV1 {
    ProductionSemanticImportErrorV1::BodyConstruction(Box::new(
        ProductionSemanticBodyErrorV1::IdentityTableMismatch { table },
    ))
}

const fn semantic_function_role_v1(role: CollectedFunctionRole) -> SemanticFunctionRoleV1 {
    match role {
        CollectedFunctionRole::KernelEntry => SemanticFunctionRoleV1::KernelRoot,
        CollectedFunctionRole::InternalHelper => SemanticFunctionRoleV1::InternalHelper,
        CollectedFunctionRole::DeviceFfiExport => SemanticFunctionRoleV1::DeviceFfiExport,
    }
}

fn semantic_function_export_v1(
    function: &RetainedSemanticFunctionProducerV1<'_>,
) -> Result<ProductionSemanticFunctionExportV1, ProductionSemanticImportErrorV1> {
    semantic_function_export_metadata_v1(
        function.role,
        function.export_name.as_deref(),
        function.kernel_binding,
        function.generated_host_contract_identity,
        function.frontend_contract.as_ref(),
    )
}

fn semantic_function_export_metadata_v1(
    role: CollectedFunctionRole,
    export_name: Option<&str>,
    kernel_binding: Option<reserved_fe2o3_symbols::KernelBindingIdV1>,
    generated_host_contract_identity: Option<reserved_fe2o3_symbols::GeneratedHostContractIdV3>,
    frontend_contract: Option<&super::AuthenticatedKernelFrontendContractV1>,
) -> Result<ProductionSemanticFunctionExportV1, ProductionSemanticImportErrorV1> {
    match role {
        CollectedFunctionRole::InternalHelper
            if export_name.is_none()
                && kernel_binding.is_none()
                && generated_host_contract_identity.is_none()
                && frontend_contract.is_none() =>
        {
            Ok(ProductionSemanticFunctionExportV1::None)
        }
        CollectedFunctionRole::DeviceFfiExport
            if kernel_binding.is_none()
                && generated_host_contract_identity.is_none()
                && frontend_contract.is_none() =>
        {
            Ok(ProductionSemanticFunctionExportV1::DeviceFfi(
                semantic_link_symbol_v1(export_name)?,
            ))
        }
        CollectedFunctionRole::KernelEntry => {
            let binding = kernel_binding
                .ok_or_else(|| body_owner_table_mismatch_v1("kernel binding identity"))?;
            Ok(ProductionSemanticFunctionExportV1::Kernel(
                SemanticKernelEntryV1::new(
                    semantic_link_symbol_v1(export_name)?,
                    SemanticKernelBindingIdentityV1::from_sha256(binding.as_bytes()),
                    semantic_kernel_source_contract_v1(
                        frontend_contract,
                        generated_host_contract_identity,
                    )?,
                ),
            ))
        }
        CollectedFunctionRole::InternalHelper | CollectedFunctionRole::DeviceFfiExport => {
            Err(body_owner_table_mismatch_v1("function export metadata"))
        }
    }
}

fn semantic_link_symbol_v1(
    symbol: Option<&str>,
) -> Result<SemanticLinkSymbolV1, ProductionSemanticImportErrorV1> {
    SemanticLinkSymbolV1::new(
        symbol
            .ok_or_else(|| body_owner_table_mismatch_v1("function export symbol"))?
            .as_bytes()
            .to_vec(),
    )
    .map_err(ProductionSemanticImportErrorV1::SemanticSchema)
}

fn semantic_kernel_source_contract_v1(
    authenticated: Option<&super::AuthenticatedKernelFrontendContractV1>,
    generated_host_contract_identity: Option<reserved_fe2o3_symbols::GeneratedHostContractIdV3>,
) -> Result<SemanticKernelSourceContractV1, ProductionSemanticImportErrorV1> {
    let default_launch = if generated_host_contract_identity.is_some()
        && authenticated
            .and_then(|value| value.contract().launch())
            .is_none()
    {
        // Typed registration authenticates the implicit launch. Use the same
        // rule as descriptor reconstruction before exposing semantic geometry.
        let launch = super::general_typed_launch_v3(authenticated, "semantic typed launch")
            .map_err(|_| body_owner_table_mismatch_v1("authenticated typed default launch"))?;
        let fe2o3_artifacts::BlockSize::Exact(block) = launch.block_size() else {
            return Err(body_owner_table_mismatch_v1("exact typed default launch"));
        };
        let dimensions = SemanticWorkgroupDimensionsV1::new([block.x(), block.y(), block.z()])
            .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?;
        Some(
            SemanticKernelLaunchBoundsV1::new(Some(dimensions), Some(dimensions), None)
                .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?,
        )
    } else {
        None
    };
    let Some(authenticated) = authenticated else {
        return SemanticKernelSourceContractV1::new(default_launch, None, None)
            .map_err(ProductionSemanticImportErrorV1::SemanticSchema);
    };
    let frontend = authenticated.contract();
    let launch = frontend
        .launch()
        .map(|launch| {
            SemanticKernelLaunchBoundsV1::new(
                launch
                    .required()
                    .map(|dimensions| SemanticWorkgroupDimensionsV1::new(dimensions.as_array()))
                    .transpose()?,
                launch
                    .maximum()
                    .map(|dimensions| SemanticWorkgroupDimensionsV1::new(dimensions.as_array()))
                    .transpose()?,
                launch.min_workgroups_per_compute_unit(),
            )
        })
        .transpose()
        .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?
        .or(default_launch);
    let resources = authenticated
        .resource_contract()
        .map(|resources| {
            SemanticKernelResourceContractV1::new(
                resources.static_shared_memory_bytes(),
                resources.max_dynamic_shared_memory_bytes(),
            )
        })
        .transpose()
        .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?;
    let unsafe_assembly = frontend
        .unsafe_assembly()
        .map(|assembly| {
            SemanticUnsafeAssemblyDeclarationV1::new(
                match assembly.target() {
                    fe2o3_rustc_front::FrontendUnsafeAssemblyTargetV1::AmdGpuGfx942 => {
                        SemanticUnsafeAssemblyTargetV1::AmdGpuGfx942
                    }
                },
                assembly.operand_bits(),
                assembly.option_bits(),
                assembly.effect_bits(),
            )
        })
        .transpose()
        .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?;
    let reachable_assembly = unsafe_assembly
        .map(|assembly| {
            let reachable = authenticated.reachable_assembly();
            SemanticReachableAssemblyV1::new(
                reachable.blocks(),
                reachable.operand_bits(),
                reachable.option_bits(),
                assembly.effect_bits(),
            )
        })
        .transpose()
        .map_err(ProductionSemanticImportErrorV1::SemanticSchema)?;
    SemanticKernelSourceContractV1::new_with_resources(
        launch,
        resources,
        unsafe_assembly,
        reachable_assembly,
    )
    .map_err(ProductionSemanticImportErrorV1::SemanticSchema)
}

fn terminal_operation_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    expansion: crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1,
    abi: &SemanticFunctionAbiV1,
    types: &[SemanticTypeDeclV1],
) -> Result<SemanticCompilerIntrinsicOperationV1, ProductionSemanticImportErrorV1> {
    use crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1;
    let inputs = abi.source_input_types();
    let output = abi.source_output_type();
    let signature = tcx.instantiate_bound_regions_with_erased(
        tcx.fn_sig(instance.def_id())
            .instantiate(tcx, instance.args),
    );
    let rust_inputs = signature.inputs();
    let rust_output = signature.output();
    match expansion {
        ProductionTerminalExpansionV1::Gfx942PhysicalEntryBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryStep => {
            crate::production_physical_entry_terminal_v37::operation(tcx,instance,&signature,abi,types).map_err(body_owner_table_mismatch_v1)
        }
        ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeStep => {
            crate::production_physical_lds_exchange_terminal_v39::operation(tcx,instance,&signature,abi,types).map_err(body_owner_table_mismatch_v1)
        }
        ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyStep => {
            crate::production_physical_global_copy_terminal_v38::operation(tcx,instance,&signature,abi,types).map_err(body_owner_table_mismatch_v1)
        }
        ProductionTerminalExpansionV1::Gfx942CompleteBodyE32 => {
            crate::production_complete_body_terminal_v36::operation(tcx, instance, &signature, abi, types)
                .map_err(body_owner_table_mismatch_v1)
        }
        ProductionTerminalExpansionV1::Gfx942OrderedProgramE32
            if crate::production_ordered_program_v32::valid_signature(
                tcx, instance, &signature, abi, types,
            ) =>
        {
            let program = crate::production_ordered_program_v32::parse_program_consts(tcx, instance)
                .map_err(body_owner_table_mismatch_v1)?;
            Ok(SemanticCompilerIntrinsicOperationV1::Gfx942OrderedProgram(program))
        }
        ProductionTerminalExpansionV1::Gfx942OrderedXorAddE32
            if crate::production_ordered_region_v31::valid_signature(
                instance, &signature, abi, types
            ) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::Gfx942OrderedRegion(
                fe2o3_mir_model::semantic_mir_v1::SemanticGfx942OrderedRegionProfileV31::XorAddU32E32
            ))
        }
        ProductionTerminalExpansionV1::Gfx942InlineU32(operation)
            if inputs.len() == crate::production_inline_assembly_v30::input_count(operation)
                && rust_inputs.len() == inputs.len()
                && instance.args.is_empty()
                && abi.canon_abi() == SemanticCanonAbiV1::Rust
                && abi.extern_abi() == SemanticExternAbiV1::Rust
                && !abi.c_variadic()
                && rust_inputs
                    .iter()
                    .all(|ty| matches!(ty.kind(), TyKind::Uint(UintTy::U32)))
                && matches!(rust_output.kind(), TyKind::Uint(UintTy::U32))
                && inputs.iter().chain(std::iter::once(&output)).all(|ty| {
                    types.get(ty.index() as usize).is_some_and(|declaration| {
                        matches!(
                            declaration.shape(),
                            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                                signed: false,
                                bits: 32
                            })
                        )
                    })
                }) =>
        {
            crate::production_inline_assembly_v30::semantic_operation(operation)
                .map(SemanticCompilerIntrinsicOperationV1::Gfx942InlineU32)
                .map_err(ProductionSemanticImportErrorV1::SemanticSchema)
            }
        ProductionTerminalExpansionV1::ContextIssue
        | ProductionTerminalExpansionV1::WorkgroupDerive
        | ProductionTerminalExpansionV1::MaskedTileLoadU32
        | ProductionTerminalExpansionV1::MaskedTileIntoFragmentU32
        | ProductionTerminalExpansionV1::LaneFragmentIntoPartsU32 => {
            execution_terminals::construct(tcx, instance, expansion, abi, types)
                .map(SemanticCompilerIntrinsicOperationV1::Execution)
                .ok_or_else(|| body_owner_table_mismatch_v1("execution terminal descriptor"))
        }
        ProductionTerminalExpansionV1::ThreadIndex(axis)
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Uint(UintTy::U32)) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::ThreadIndex(axis))
        }
        ProductionTerminalExpansionV1::WorkgroupIndex(axis)
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Uint(UintTy::U32)) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::WorkgroupIndex(axis))
        }
        ProductionTerminalExpansionV1::WorkgroupDimension(axis)
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Uint(UintTy::U32)) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::WorkgroupDimension(
                axis,
            ))
        }
        ProductionTerminalExpansionV1::GridDimension(axis)
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Uint(UintTy::U32)) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::GridDimension(axis))
        }
        ProductionTerminalExpansionV1::WorkgroupLdsScopeCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && abi.canon_abi() == SemanticCanonAbiV1::Rust
                && abi.extern_abi() == SemanticExternAbiV1::Rust
                && !abi.c_variadic()
                && rust_is_trusted_adt_v1(
                    tcx,
                    rust_output,
                    TrustedDeviceItem::WorkgroupLdsScope,
                )
                && output == abi.return_value().ty()
                && matches!(abi.return_value().mode(), SemanticAbiPassModeV1::Ignore)
                && abi.return_value().adjusted().is_none()
                && abi.return_value().pointee_override().is_none()
                && abi.arguments().is_empty()
                && abi.hidden_arguments().is_empty()
                && semantic_exact_inhabited_aggregate_zst_v1(types, output) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::WorkgroupLdsScopeCurrent { scope: output })
        }
        ProductionTerminalExpansionV1::DynamicLdsExactCurrent
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::WorkgroupLdsScope)
                })
                && rust_dynamic_lds_scalar_element_v1(tcx, rust_output).is_some() =>
        {
            let elements = single_const_u64_v1(instance)
                .filter(|elements| *elements != 0 && *elements <= u64::from(u32::MAX))
                .ok_or_else(|| body_owner_table_mismatch_v1("exact LDS element count"))?;
            let scope = pointer_pointee_v1(types, inputs[0])?;
            let element_storage = dynamic_lds_element_storage_v1(types, output)?;
            let storage = types
                .get(element_storage.index() as usize)
                .ok_or_else(|| body_owner_table_mismatch_v1("exact LDS storage type"))?;
            let alignment = storage.layout().alignment_bytes();
            if !storage.layout().size_bytes().is_some_and(|size| size != 0)
                || alignment == 0
                || alignment > 16
                || !alignment.is_power_of_two()
            {
                return Err(body_owner_table_mismatch_v1(
                    "exact LDS storage size or alignment",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::DynamicLdsExactCurrent {
                    scope,
                    dynamic_lds: output,
                    element_storage,
                    elements,
                },
            )
        }
        ProductionTerminalExpansionV1::DynamicLdsIntoCollectiveRawParts
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_dynamic_lds_scalar_element_v1(tcx, rust_inputs[0])
                    .is_some_and(|element| rust_dynamic_lds_raw_parts_v1(rust_output, element)) =>
        {
            let dynamic_lds = inputs[0];
            let element_storage = dynamic_lds_element_storage_v1(types, dynamic_lds)?;
            let raw_pointer = tuple_field_v1(types, output, 0)?;
            let element = pointer_pointee_v1(types, raw_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::DynamicLdsIntoCollectiveRawParts {
                    dynamic_lds,
                    raw_parts: output,
                    element_storage,
                    element,
                },
            )
        }
        ProductionTerminalExpansionV1::WorkgroupPipelineCurrent
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::WorkgroupLdsScope)
                })
                && rust_workgroup_pipeline_contract_v1(tcx, rust_output).is_some() =>
        {
            let (_, buffers, elements, prefetch_distance) =
                rust_workgroup_pipeline_contract_v1(tcx, rust_output)
                    .ok_or_else(|| body_owner_table_mismatch_v1("workgroup pipeline contract"))?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate {
                    scope: pointer_pointee_v1(types, inputs[0])?,
                    pipeline: output,
                    buffers,
                    elements,
                    prefetch_distance,
                },
            )
        }
        ProductionTerminalExpansionV1::WorkgroupPipelineStage
        | ProductionTerminalExpansionV1::WorkgroupPipelineCommit
        | ProductionTerminalExpansionV1::WorkgroupPipelineWait
        | ProductionTerminalExpansionV1::WorkgroupPipelineConsume
        | ProductionTerminalExpansionV1::WorkgroupPipelineDiscard
        | ProductionTerminalExpansionV1::WorkgroupPipelineRelease
            if inputs.len() == 2
                && rust_inputs.len() == 2
                && rust_reference_pointee_v1(rust_inputs[0])
                    .and_then(|ty| rust_workgroup_pipeline_contract_v1(tcx, ty))
                    .is_some()
                && matches!(rust_inputs[1].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_output.kind(), TyKind::Tuple(fields) if fields.is_empty()) =>
        {
            let event = match expansion {
                ProductionTerminalExpansionV1::WorkgroupPipelineStage => {
                    SemanticWorkgroupPipelineEventV1::Stage
                }
                ProductionTerminalExpansionV1::WorkgroupPipelineCommit => {
                    SemanticWorkgroupPipelineEventV1::Commit
                }
                ProductionTerminalExpansionV1::WorkgroupPipelineWait => {
                    SemanticWorkgroupPipelineEventV1::Wait
                }
                ProductionTerminalExpansionV1::WorkgroupPipelineConsume => {
                    SemanticWorkgroupPipelineEventV1::Consume
                }
                ProductionTerminalExpansionV1::WorkgroupPipelineDiscard => {
                    SemanticWorkgroupPipelineEventV1::Discard
                }
                ProductionTerminalExpansionV1::WorkgroupPipelineRelease => {
                    SemanticWorkgroupPipelineEventV1::Release
                }
                _ => unreachable!("guarded pipeline event expansion"),
            };
            Ok(
                SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineEvent {
                    pipeline: pointer_pointee_v1(types, inputs[0])?,
                    event,
                },
            )
        }
        ProductionTerminalExpansionV1::WorkgroupPipelineWrite
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && rust_reference_pointee_v1(rust_inputs[0])
                    .and_then(|ty| rust_workgroup_pipeline_contract_v1(tcx, ty))
                    .is_some_and(|(element, ..)| element == rust_inputs[3])
                && matches!(rust_inputs[1].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_output.kind(), TyKind::Tuple(fields) if fields.is_empty()) =>
        {
            Ok(
                SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineWrite {
                    pipeline: pointer_pointee_v1(types, inputs[0])?,
                    element: inputs[3],
                },
            )
        }
        ProductionTerminalExpansionV1::WorkgroupPipelineRead
            if inputs.len() == 3
                && rust_inputs.len() == 3
                && rust_reference_pointee_v1(rust_inputs[0])
                    .and_then(|ty| rust_workgroup_pipeline_contract_v1(tcx, ty))
                    .is_some_and(|(element, ..)| element == rust_output)
                && matches!(rust_inputs[1].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize)) =>
        {
            Ok(
                SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineRead {
                    pipeline: pointer_pointee_v1(types, inputs[0])?,
                    element: output,
                },
            )
        }
        ProductionTerminalExpansionV1::Trap
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Never) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::Trap)
        }
        ProductionTerminalExpansionV1::WorkgroupBarrier
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Tuple(fields) if fields.is_empty()) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::WorkgroupBarrier)
        }
        ProductionTerminalExpansionV1::ColdPath
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && matches!(rust_output.kind(), TyKind::Tuple(fields) if fields.is_empty()) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::ColdPath)
        }
        ProductionTerminalExpansionV1::MathContextCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_is_trusted_adt_v1(
                    tcx,
                    rust_output,
                    TrustedDeviceItem::DeviceMath(
                        dialect_amdgcn::DeviceMathDiagnosticItem::Context,
                    ),
                ) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::MathContextCurrent { context: output })
        }
        ProductionTerminalExpansionV1::MathF32(function)
            if inputs.len() == function.arity() + 1
                && rust_inputs.len() == function.arity() + 1
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(
                        tcx,
                        ty,
                        TrustedDeviceItem::DeviceMath(
                            dialect_amdgcn::DeviceMathDiagnosticItem::Context,
                        ),
                    )
                })
                && rust_inputs[1..]
                    .iter()
                    .all(|ty| matches!(ty.kind(), TyKind::Float(FloatTy::F32)))
                && matches!(rust_output.kind(), TyKind::Float(FloatTy::F32)) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::MathF32 {
                context: pointer_pointee_v1(types, inputs[0])?,
                function: semantic_f32_math_function_v1(function),
            })
        }
        ProductionTerminalExpansionV1::RustcFabsF32
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && inputs[0] == output
                && semantic_f32_type_v1(types, output)
                && matches!(rust_inputs[0].kind(), TyKind::Float(FloatTy::F32))
                && matches!(rust_output.kind(), TyKind::Float(FloatTy::F32)) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::FabsF32)
        }
        ProductionTerminalExpansionV1::RustcSaturatingInteger(operation)
            if saturating_terminal::matches_v1(
                tcx,
                types,
                inputs,
                output,
                rust_inputs,
                rust_output,
            ) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::SaturatingInteger(
                operation,
            ))
        }
        ProductionTerminalExpansionV1::Gfx942Wave64Shuffle(scalar) => {
            wave64_shuffle_terminal::construct(tcx, instance, scalar, abi, types)
                .ok_or_else(|| body_owner_table_mismatch_v1("exact Wave64 shuffle primitive"))
        }
        ProductionTerminalExpansionV1::MemoryVolatileLoad
            if inputs.len() == 2
                && rust_inputs.len() == 2
                && rust_shared_slice_element_v1(rust_inputs[0]) == Some(rust_output)
                && matches!(rust_inputs[1].kind(), TyKind::Uint(UintTy::Usize))
                && rust_supported_volatile_load_scalar_v1(rust_output) =>
        {
            let slice = pointer_pointee_v1(types, inputs[0])?;
            let SemanticTypeShapeV1::Slice { element } = types
                .get(slice.index() as usize)
                .ok_or_else(|| body_owner_table_mismatch_v1("volatile-load slice type"))?
                .shape()
            else {
                return Err(body_owner_table_mismatch_v1("volatile-load slice type"));
            };
            if *element != output
                || !types
                    .get(inputs[1].index() as usize)
                    .is_some_and(|declaration| {
                        matches!(
                            declaration.shape(),
                            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                                signed: false,
                                bits: 64,
                            })
                        )
                    })
            {
                return Err(body_owner_table_mismatch_v1(
                    "volatile-load element or index type",
                ));
            }
            Ok(SemanticCompilerIntrinsicOperationV1::MemoryVolatileLoad { element: *element })
        }
        ProductionTerminalExpansionV1::Bf16Conversion(conversion) => {
            if inputs.len() != 1 || rust_inputs.len() != 1 {
                return Err(body_owner_table_mismatch_v1("BF16 conversion arity"));
            }
            let input = inputs[0];
            let rust_input = rust_inputs[0];
            let rust_is_bf16 = |ty| {
                rust_is_trusted_adt_v1(
                    tcx,
                    ty,
                    TrustedDeviceItem::DeviceValue(DeviceValueDiagnosticItem::Bf16),
                )
            };
            let rust_is_u16 = |ty: Ty<'tcx>| matches!(ty.kind(), TyKind::Uint(UintTy::U16));
            let rust_is_f32 = |ty: Ty<'tcx>| matches!(ty.kind(), TyKind::Float(FloatTy::F32));
            let (kind, valid) = match conversion {
                ProductionBf16ConversionV1::FromBits => (
                    SemanticBf16ConversionKindV1::FromBits,
                    rust_is_u16(rust_input)
                        && rust_is_bf16(rust_output)
                        && semantic_u16_type_v1(types, input)
                        && semantic_bf16_storage_type_v1(types, output),
                ),
                ProductionBf16ConversionV1::ToBits => (
                    SemanticBf16ConversionKindV1::ToBits,
                    rust_is_bf16(rust_input)
                        && rust_is_u16(rust_output)
                        && semantic_bf16_storage_type_v1(types, input)
                        && semantic_u16_type_v1(types, output),
                ),
                ProductionBf16ConversionV1::FromF32RoundTiesEven => (
                    SemanticBf16ConversionKindV1::FromF32RoundTiesEven,
                    rust_is_f32(rust_input)
                        && rust_is_bf16(rust_output)
                        && semantic_f32_type_v1(types, input)
                        && semantic_bf16_storage_type_v1(types, output),
                ),
                ProductionBf16ConversionV1::ToF32 => (
                    SemanticBf16ConversionKindV1::ToF32,
                    rust_is_bf16(rust_input)
                        && rust_is_f32(rust_output)
                        && semantic_bf16_storage_type_v1(types, input)
                        && semantic_f32_type_v1(types, output),
                ),
            };
            if !valid {
                return Err(body_owner_table_mismatch_v1(
                    "authenticated BF16 conversion ABI",
                ));
            }
            Ok(SemanticCompilerIntrinsicOperationV1::Bf16Conversion {
                kind,
                input,
                output,
            })
        }
        ProductionTerminalExpansionV1::CollectiveContextCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_is_trusted_adt_v1(
                    tcx,
                    rust_output,
                    TrustedDeviceItem::Gfx942CollectivesContext,
                ) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::CollectiveContextCurrent { context: output })
        }
        ProductionTerminalExpansionV1::WorkgroupCollectiveContextCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_is_trusted_adt_v1(
                    tcx,
                    rust_output,
                    TrustedDeviceItem::WorkgroupCollectivesContext,
                ) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::CollectiveContextCurrent { context: output })
        }
        ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum
            if inputs.len() == 3
                && rust_inputs.len() == 3
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::WorkgroupCollectivesContext)
                })
                && rust_dynamic_lds_uninitialized_scalar_element_v1(tcx, rust_inputs[1])
                    == Some(rust_output)
                && rust_inputs[2] == rust_output
                && matches!(
                    rust_output.kind(),
                    TyKind::Int(IntTy::I32)
                        | TyKind::Uint(UintTy::U32)
                        | TyKind::Float(FloatTy::F32)
                ) =>
        {
            let context = pointer_pointee_v1(types, inputs[0])?;
            let dynamic_lds = inputs[1];
            let element_storage = dynamic_lds_element_storage_v1(types, dynamic_lds)?;
            if inputs[2] != output {
                return Err(body_owner_table_mismatch_v1(
                    "target-neutral workgroup reduction element",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::NeutralWorkgroupReduceSum {
                    context,
                    dynamic_lds,
                    element_storage,
                    element: output,
                },
            )
        }
        ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum
        | ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum
            if inputs.len() == 3
                && rust_inputs.len() == 3
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::WorkgroupCollectivesContext)
                })
                && rust_dynamic_lds_uninitialized_scalar_element_v1(tcx, rust_inputs[1])
                    == Some(rust_output)
                && rust_inputs[2] == rust_output
                && matches!(
                    rust_output.kind(),
                    TyKind::Int(IntTy::I32)
                        | TyKind::Uint(UintTy::U32)
                        | TyKind::Float(FloatTy::F32)
                ) =>
        {
            let context = pointer_pointee_v1(types, inputs[0])?;
            let dynamic_lds = inputs[1];
            let element_storage = dynamic_lds_element_storage_v1(types, dynamic_lds)?;
            if inputs[2] != output {
                return Err(body_owner_table_mismatch_v1(
                    "target-neutral workgroup scan element",
                ));
            }
            let kind = match expansion {
                ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum => {
                    SemanticWorkgroupScanKindV1::Inclusive
                }
                ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum => {
                    SemanticWorkgroupScanKindV1::Exclusive
                }
                _ => unreachable!("match arm admits only target-neutral workgroup scans"),
            };
            Ok(
                SemanticCompilerIntrinsicOperationV1::NeutralWorkgroupScanSum {
                    context,
                    dynamic_lds,
                    element_storage,
                    element: output,
                    kind,
                },
            )
        }
        ProductionTerminalExpansionV1::WorkgroupReduceSum
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::WorkgroupGroup)
                })
                && rust_reference_pointee_v1(rust_inputs[1]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx942CollectivesContext)
                })
                && matches!(
                    rust_inputs[2].kind(),
                    TyKind::Ref(_, _, rustc_hir::Mutability::Mut)
                )
                && rust_reference_pointee_v1(rust_inputs[2])
                    .and_then(|ty| rust_workgroup_collective_scratch_element_v1(tcx, ty))
                    == Some(rust_output)
                && rust_inputs[3] == rust_output
                && matches!(
                    rust_output.kind(),
                    TyKind::Int(IntTy::I32)
                        | TyKind::Uint(UintTy::U32)
                        | TyKind::Float(FloatTy::F32)
                ) =>
        {
            let workgroup = pointer_pointee_v1(types, inputs[0])?;
            let context = pointer_pointee_v1(types, inputs[1])?;
            let scratch = pointer_pointee_v1(types, inputs[2])?;
            let base = aggregate_field_v1(types, scratch, 0)?;
            if pointer_pointee_v1(types, base)? != output || inputs[3] != output {
                return Err(body_owner_table_mismatch_v1(
                    "workgroup reduction scratch element",
                ));
            }
            Ok(SemanticCompilerIntrinsicOperationV1::WorkgroupReduceSum {
                workgroup,
                context,
                scratch,
                element: output,
            })
        }
        ProductionTerminalExpansionV1::SubgroupReduceSumF32
        | ProductionTerminalExpansionV1::SubgroupReduceMaxF32
            if inputs.len() == 2
                && rust_inputs.len() == 2
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx942CollectivesContext)
                })
                && matches!(rust_inputs[1].kind(), TyKind::Float(FloatTy::F32))
                && matches!(rust_output.kind(), TyKind::Float(FloatTy::F32)) =>
        {
            let width = single_const_u32_v1(instance)
                .ok_or_else(|| body_owner_table_mismatch_v1("subgroup reduction width"))?;
            let kind = match expansion {
                ProductionTerminalExpansionV1::SubgroupReduceSumF32 => {
                    SemanticSubgroupReductionKindV1::Sum
                }
                ProductionTerminalExpansionV1::SubgroupReduceMaxF32 => {
                    SemanticSubgroupReductionKindV1::Maximum
                }
                _ => unreachable!("matched subgroup reduction expansion"),
            };
            Ok(SemanticCompilerIntrinsicOperationV1::SubgroupReduceF32 {
                context: pointer_pointee_v1(types, inputs[0])?,
                width,
                kind,
            })
        }
        ProductionTerminalExpansionV1::Gfx950SubgroupCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_is_trusted_adt_v1(
                    tcx,
                    rust_output,
                    TrustedDeviceItem::Gfx950SubgroupContext,
                ) =>
        {
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950SubgroupContextCurrent {
                    context: output,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950SubgroupReduceSumF32
        | ProductionTerminalExpansionV1::Gfx950SubgroupReduceMaxF32)
            if inputs.len() == 2
                && rust_inputs.len() == 2
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx950SubgroupContext)
                })
                && matches!(rust_inputs[1].kind(), TyKind::Float(FloatTy::F32))
                && matches!(rust_output.kind(), TyKind::Float(FloatTy::F32)) =>
        {
            let width = single_const_u32_v1(instance)
                .filter(|width| *width != 0 && width.is_power_of_two() && *width <= 64)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 subgroup reduction width"))?;
            let kind = match expansion {
                ProductionTerminalExpansionV1::Gfx950SubgroupReduceSumF32 => {
                    SemanticSubgroupReductionKindV1::Sum
                }
                ProductionTerminalExpansionV1::Gfx950SubgroupReduceMaxF32 => {
                    SemanticSubgroupReductionKindV1::Maximum
                }
                _ => unreachable!("matched gfx950 subgroup reduction expansion"),
            };
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950SubgroupReduceF32 {
                    context: pointer_pointee_v1(types, inputs[0])?,
                    width,
                    kind,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950SubgroupBroadcastF32
            if inputs.len() == 3
                && rust_inputs.len() == 3
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx950SubgroupContext)
                })
                && matches!(rust_inputs[1].kind(), TyKind::Float(FloatTy::F32))
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::U32))
                && matches!(rust_output.kind(), TyKind::Float(FloatTy::F32)) =>
        {
            let width = single_const_u32_v1(instance)
                .filter(|width| *width != 0 && width.is_power_of_two() && *width <= 64)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 subgroup broadcast width"))?;
            Ok(SemanticCompilerIntrinsicOperationV1::SubgroupBroadcastF32 {
                context: pointer_pointee_v1(types, inputs[0])?,
                width,
            })
        }
        ProductionTerminalExpansionV1::MatrixContextCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_is_trusted_adt_v1(tcx, rust_output, TrustedDeviceItem::DeviceMatrix) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::MatrixContextCurrent { context: output })
        }
        ProductionTerminalExpansionV1::Gfx950MatrixContextCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_is_trusted_adt_v1(tcx, rust_output, TrustedDeviceItem::Gfx950Matrix) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::MatrixContextCurrent { context: output })
        }
        ProductionTerminalExpansionV1::WaveLaneCurrent
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_wave_lane64_v1(tcx, rust_output) =>
        {
            Ok(SemanticCompilerIntrinsicOperationV1::WaveLaneCurrent {
                lane: output,
                wave_width: 64,
            })
        }
        expansion @ (ProductionTerminalExpansionV1::Bf16MatrixARowMajor
        | ProductionTerminalExpansionV1::Bf16MatrixBRowMajor
        | ProductionTerminalExpansionV1::Bf16MatrixBColumnMajor)
            if inputs.len() == 5
                && rust_inputs.len() == 5
                && rust_shared_u16_slice_v1(rust_inputs[0])
                && rust_inputs[1..]
                    .iter()
                    .all(|ty| matches!(ty.kind(), TyKind::Uint(UintTy::Usize))) =>
        {
            let (rust_view, rust_error) = rust_result_payloads_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA row-major result"))?;
            let role = rust_mfma_matrix_role_v1(tcx, rust_view)
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA row-major view"))?;
            let expected_role = match expansion {
                ProductionTerminalExpansionV1::Bf16MatrixARowMajor => SemanticMfmaOperandRoleV1::A,
                ProductionTerminalExpansionV1::Bf16MatrixBRowMajor => SemanticMfmaOperandRoleV1::B,
                ProductionTerminalExpansionV1::Bf16MatrixBColumnMajor => {
                    SemanticMfmaOperandRoleV1::B
                }
                _ => unreachable!("matched BF16 checked-view expansion"),
            };
            let column_major = expansion == ProductionTerminalExpansionV1::Bf16MatrixBColumnMajor;
            if role != expected_role
                || rust_is_trusted_adt_v1(
                    tcx,
                    rust_view,
                    TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView,
                ) != column_major
                || !rust_is_trusted_adt_v1(
                    tcx,
                    rust_error,
                    TrustedDeviceItem::Bf16MfmaMatrixViewError,
                )
            {
                return Err(body_owner_table_mismatch_v1(
                    "typed MFMA row-major role or error",
                ));
            }
            let (view, error) = semantic_result_payloads_v1(types, output)?;
            if column_major {
                return Ok(
                    SemanticCompilerIntrinsicOperationV1::Bf16MatrixViewColumnMajor {
                        result: output,
                        view,
                        error,
                    },
                );
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Bf16MatrixViewRowMajor {
                    result: output,
                    view,
                    error,
                    role,
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950Fp4MatrixARowMajor
        | ProductionTerminalExpansionV1::Gfx950Fp4MatrixBRowMajor)
            if inputs.len() == 5
                && rust_inputs.len() == 5
                && rust_shared_u8_slice_v1(rust_inputs[0])
                && rust_inputs[1..]
                    .iter()
                    .all(|ty| matches!(ty.kind(), TyKind::Uint(UintTy::Usize))) =>
        {
            let (rust_view, rust_error) = rust_result_payloads_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 row-major result"))?;
            let role = rust_gfx950_fp4_matrix_role_v1(tcx, rust_view)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 row-major view"))?;
            let expected_role = match expansion {
                ProductionTerminalExpansionV1::Gfx950Fp4MatrixARowMajor => {
                    SemanticMfmaOperandRoleV1::A
                }
                ProductionTerminalExpansionV1::Gfx950Fp4MatrixBRowMajor => {
                    SemanticMfmaOperandRoleV1::B
                }
                _ => unreachable!("matched gfx950 FP4 row-major expansion"),
            };
            if role != expected_role
                || !rust_is_trusted_adt_v1(
                    tcx,
                    rust_error,
                    TrustedDeviceItem::Gfx950MfmaMatrixViewError,
                )
            {
                return Err(body_owner_table_mismatch_v1(
                    "gfx950 FP4 row-major role or error",
                ));
            }
            let (view, error) = semantic_result_payloads_v1(types, output)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950Fp4MatrixViewRowMajor {
                    result: output,
                    view,
                    error,
                    role,
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950Fp8MatrixARowMajor
        | ProductionTerminalExpansionV1::Gfx950Fp8MatrixBRowMajor)
            if inputs.len() == 5
                && rust_inputs.len() == 5
                && rust_shared_u8_slice_v1(rust_inputs[0])
                && rust_inputs[1..]
                    .iter()
                    .all(|ty| matches!(ty.kind(), TyKind::Uint(UintTy::Usize))) =>
        {
            let (rust_view, rust_error) = rust_result_payloads_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 row-major result"))?;
            let role = rust_gfx950_fp8_matrix_role_v1(tcx, rust_view)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 row-major view"))?;
            let expected_role = match expansion {
                ProductionTerminalExpansionV1::Gfx950Fp8MatrixARowMajor => {
                    SemanticMfmaOperandRoleV1::A
                }
                ProductionTerminalExpansionV1::Gfx950Fp8MatrixBRowMajor => {
                    SemanticMfmaOperandRoleV1::B
                }
                _ => unreachable!("matched gfx950 row-major expansion"),
            };
            if role != expected_role
                || !rust_is_trusted_adt_v1(
                    tcx,
                    rust_error,
                    TrustedDeviceItem::Gfx950MfmaMatrixViewError,
                )
            {
                return Err(body_owner_table_mismatch_v1(
                    "gfx950 FP8 row-major role or error",
                ));
            }
            let (view, error) = semantic_result_payloads_v1(types, output)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950Fp8MatrixViewRowMajor {
                    result: output,
                    view,
                    error,
                    role,
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                },
            )
        }
        ProductionTerminalExpansionV1::StridedReadView2DFromSharedSlice
            if inputs.len() == 5
                && rust_inputs.len() == 5
                && rust_shared_slice_element_v1(rust_inputs[0])
                    .is_some_and(rust_supported_read_view_scalar_v1)
                && rust_inputs[1..]
                    .iter()
                    .all(|ty| matches!(ty.kind(), TyKind::Uint(UintTy::Usize))) =>
        {
            let Some(rust_element) = rust_shared_slice_element_v1(rust_inputs[0]) else {
                return Err(body_owner_table_mismatch_v1(
                    "strided read view slice element",
                ));
            };
            let (rust_view, rust_error) = rust_result_payloads_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("strided read view result"))?;
            let view_arguments = rust_trusted_adt_type_arguments_v1(
                tcx,
                rust_view,
                TrustedDeviceItem::StridedReadView2D,
            )
            .ok_or_else(|| body_owner_table_mismatch_v1("strided read view type"))?;
            if view_arguments.as_slice() != [rust_element]
                || !rust_is_exact_trusted_marker_v1(
                    tcx,
                    rust_error,
                    TrustedDeviceItem::StridedReadView2DError,
                )
            {
                return Err(body_owner_table_mismatch_v1(
                    "strided read view element or error",
                ));
            }
            let (view, error) = semantic_result_payloads_v1(types, output)?;
            let slice = pointer_pointee_v1(types, inputs[0])?;
            let SemanticTypeShapeV1::Slice { element } = types
                .get(slice.index() as usize)
                .ok_or_else(|| body_owner_table_mismatch_v1("strided read view slice"))?
                .shape()
            else {
                return Err(body_owner_table_mismatch_v1("strided read view slice"));
            };
            Ok(
                SemanticCompilerIntrinsicOperationV1::StridedReadView2DFromSharedSlice {
                    result: output,
                    view,
                    error,
                    element: *element,
                },
            )
        }
        ProductionTerminalExpansionV1::StridedReadView2DLoadOr
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && matches!(rust_inputs[1].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize))
                && rust_inputs[3] == rust_output
                && rust_supported_read_view_scalar_v1(rust_output) =>
        {
            let rust_view = rust_reference_pointee_v1(rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("strided read view borrow"))?;
            let view_arguments = rust_trusted_adt_type_arguments_v1(
                tcx,
                rust_view,
                TrustedDeviceItem::StridedReadView2D,
            )
            .ok_or_else(|| body_owner_table_mismatch_v1("strided read view type"))?;
            if view_arguments.as_slice() != [rust_output] || inputs[3] != output {
                return Err(body_owner_table_mismatch_v1(
                    "strided read view load element",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::StridedReadView2DLoadOr {
                    view: pointer_pointee_v1(types, inputs[0])?,
                    element: output,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Bf16MatrixALoadZeroFilledV2
        | ProductionTerminalExpansionV1::Bf16MatrixBLoadZeroFilledV2
        | ProductionTerminalExpansionV1::Bf16MatrixBColumnMajorLoadZeroFilledV1)
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[3].kind(), TyKind::Uint(UintTy::Usize)) =>
        {
            let rust_view = rust_reference_pointee_v1(rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA load view borrow"))?;
            let role = rust_mfma_matrix_role_v1(tcx, rust_view)
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA load view"))?;
            let rust_lane = rust_reference_pointee_v1(rust_inputs[1])
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA load lane borrow"))?;
            let contract = rust_mfma_fragment_contract_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA load fragment"))?;
            let expected_role = match expansion {
                ProductionTerminalExpansionV1::Bf16MatrixALoadZeroFilledV2 => {
                    SemanticMfmaOperandRoleV1::A
                }
                ProductionTerminalExpansionV1::Bf16MatrixBLoadZeroFilledV2 => {
                    SemanticMfmaOperandRoleV1::B
                }
                ProductionTerminalExpansionV1::Bf16MatrixBColumnMajorLoadZeroFilledV1 => {
                    SemanticMfmaOperandRoleV1::B
                }
                _ => unreachable!("matched MFMA load expansion"),
            };
            let column_major =
                expansion == ProductionTerminalExpansionV1::Bf16MatrixBColumnMajorLoadZeroFilledV1;
            if role != expected_role
                || rust_is_trusted_adt_v1(
                    tcx,
                    rust_view,
                    TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView,
                ) != column_major
                || contract.role != expected_role
                || !rust_wave_lane64_v1(tcx, rust_lane)
            {
                return Err(body_owner_table_mismatch_v1("typed MFMA load contract"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoadZeroFilledV2 {
                    fragment: output,
                    view: pointer_pointee_v1(types, inputs[0])?,
                    lane: pointer_pointee_v1(types, inputs[1])?,
                    contract,
                    storage_layout: if column_major {
                        SemanticMfmaStorageLayoutV1::ColumnMajor
                    } else {
                        SemanticMfmaStorageLayoutV1::RowMajor
                    },
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950Fp4MatrixALoadM16K128
        | ProductionTerminalExpansionV1::Gfx950Fp4MatrixBLoadK128N16)
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[3].kind(), TyKind::Uint(UintTy::Usize)) =>
        {
            let rust_view = rust_reference_pointee_v1(rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 load view borrow"))?;
            let role = rust_gfx950_fp4_matrix_role_v1(tcx, rust_view)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 load view"))?;
            let rust_lane = rust_reference_pointee_v1(rust_inputs[1])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 load lane borrow"))?;
            let contract = rust_gfx950_fp4_fragment_contract_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 load fragment"))?;
            let expected_role = match expansion {
                ProductionTerminalExpansionV1::Gfx950Fp4MatrixALoadM16K128 => {
                    SemanticMfmaOperandRoleV1::A
                }
                ProductionTerminalExpansionV1::Gfx950Fp4MatrixBLoadK128N16 => {
                    SemanticMfmaOperandRoleV1::B
                }
                _ => unreachable!("matched gfx950 FP4 load expansion"),
            };
            if role != expected_role
                || contract.role != expected_role
                || !rust_wave_lane64_v1(tcx, rust_lane)
            {
                return Err(body_owner_table_mismatch_v1("gfx950 FP4 load contract"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950Fp4MatrixLoadM16K128 {
                    fragment: output,
                    view: pointer_pointee_v1(types, inputs[0])?,
                    lane: pointer_pointee_v1(types, inputs[1])?,
                    contract,
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950Fp8MatrixALoadM16K128
        | ProductionTerminalExpansionV1::Gfx950Fp8MatrixBLoadK128N16)
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[3].kind(), TyKind::Uint(UintTy::Usize)) =>
        {
            let rust_view = rust_reference_pointee_v1(rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 load view borrow"))?;
            let role = rust_gfx950_fp8_matrix_role_v1(tcx, rust_view)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 load view"))?;
            let rust_lane = rust_reference_pointee_v1(rust_inputs[1])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 load lane borrow"))?;
            let contract = rust_gfx950_fp8_fragment_contract_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 load fragment"))?;
            let expected_role = match expansion {
                ProductionTerminalExpansionV1::Gfx950Fp8MatrixALoadM16K128 => {
                    SemanticMfmaOperandRoleV1::A
                }
                ProductionTerminalExpansionV1::Gfx950Fp8MatrixBLoadK128N16 => {
                    SemanticMfmaOperandRoleV1::B
                }
                _ => unreachable!("matched gfx950 FP8 load expansion"),
            };
            if role != expected_role
                || contract.role != expected_role
                || !rust_wave_lane64_v1(tcx, rust_lane)
            {
                return Err(body_owner_table_mismatch_v1("gfx950 FP8 load contract"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950Fp8MatrixLoadM16K128 {
                    fragment: output,
                    view: pointer_pointee_v1(types, inputs[0])?,
                    lane: pointer_pointee_v1(types, inputs[1])?,
                    contract,
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950LdsTransposeTileCurrent
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0])
                    .is_some_and(|ty| rust_wave_lane64_v1(tcx, ty)) =>
        {
            let format = rust_gfx950_lds_transpose_format_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose current tile"))?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeCurrent {
                    tile: output,
                    lane: pointer_pointee_v1(types, inputs[0])?,
                    format,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB4
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB8)
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && matches!(rust_inputs[2].kind(), TyKind::Uint(UintTy::Usize))
                && matches!(rust_inputs[3].kind(), TyKind::Uint(UintTy::Usize)) =>
        {
            let input_format = rust_gfx950_lds_transpose_format_v1(tcx, rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose stage input"))?;
            let output_format = rust_gfx950_lds_transpose_format_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose stage output"))?;
            let rust_view = rust_reference_pointee_v1(rust_inputs[1])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose stage view"))?;
            let expected_format = match expansion {
                ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB4 => {
                    (rust_gfx950_fp4_matrix_role_v1(tcx, rust_view)
                        == Some(SemanticMfmaOperandRoleV1::A))
                    .then_some(SemanticGfx950LdsTransposeFormatV1::Fp4E2M1)
                }
                ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB8 => {
                    (rust_gfx950_fp8_matrix_role_v1(tcx, rust_view)
                        == Some(SemanticMfmaOperandRoleV1::A))
                    .then_some(SemanticGfx950LdsTransposeFormatV1::Fp8E4M3)
                }
                _ => unreachable!("matched gfx950 transpose stage expansion"),
            }
            .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose stage A view"))?;
            if input_format != expected_format || output_format != expected_format {
                return Err(body_owner_table_mismatch_v1(
                    "gfx950 transpose stage format transition",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeStage {
                    input_tile: inputs[0],
                    output_tile: output,
                    view: pointer_pointee_v1(types, inputs[1])?,
                    format: expected_format,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950LdsTransposePublish
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let input_format = rust_gfx950_lds_transpose_format_v1(tcx, rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose publish input"))?;
            let output_format = rust_gfx950_lds_transpose_format_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose publish output"))?;
            if input_format != output_format {
                return Err(body_owner_table_mismatch_v1(
                    "gfx950 transpose publish format transition",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposePublish {
                    input_tile: inputs[0],
                    output_tile: output,
                    format: input_format,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB4
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB8)
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let input_format = rust_gfx950_lds_transpose_format_v1(tcx, rust_inputs[0])
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose read tile"))?;
            let (expected_format, contract) = match expansion {
                ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB4 => (
                    SemanticGfx950LdsTransposeFormatV1::Fp4E2M1,
                    rust_gfx950_fp4_fragment_contract_v1(tcx, rust_output),
                ),
                ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB8 => (
                    SemanticGfx950LdsTransposeFormatV1::Fp8E4M3,
                    rust_gfx950_fp8_fragment_contract_v1(tcx, rust_output),
                ),
                _ => unreachable!("matched gfx950 transpose read expansion"),
            };
            let contract = contract
                .filter(|contract| contract.role == SemanticMfmaOperandRoleV1::B)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 transpose B fragment"))?;
            if input_format != expected_format {
                return Err(body_owner_table_mismatch_v1("gfx950 transpose read format"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeRead {
                    tile: inputs[0],
                    fragment: output,
                    contract,
                    format: expected_format,
                },
            )
        }
        ProductionTerminalExpansionV1::F32MatrixAccumulatorZero
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0])
                    .is_some_and(|ty| rust_wave_lane64_v1(tcx, ty)) =>
        {
            let contract = rust_mfma_accumulator_contract_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("typed MFMA zero accumulator"))?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorZero {
                    lane: pointer_pointee_v1(types, inputs[0])?,
                    fragment: output,
                    contract,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp4AccumulatorZero
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0])
                    .is_some_and(|ty| rust_wave_lane64_v1(tcx, ty)) =>
        {
            let contract = rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP4 zero accumulator"))?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorZero {
                    lane: pointer_pointee_v1(types, inputs[0])?,
                    fragment: output,
                    contract,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp8AccumulatorZero
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0])
                    .is_some_and(|ty| rust_wave_lane64_v1(tcx, ty)) =>
        {
            let contract = rust_gfx950_fp8_accumulator_contract_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("gfx950 FP8 zero accumulator"))?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorZero {
                    lane: pointer_pointee_v1(types, inputs[0])?,
                    fragment: output,
                    contract,
                },
            )
        }
        ProductionTerminalExpansionV1::F32MatrixAccumulatorIntoValues
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_mfma_accumulator_contract_v1(tcx, rust_inputs[0]).is_some()
                && rust_f32_array_v1(tcx, rust_output, 4) =>
        {
            Ok(
                SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorIntoValues {
                    fragment: inputs[0],
                    values: output,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp4AccumulatorIntoValues
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_inputs[0]).is_some()
                && rust_f32_array_v1(tcx, rust_output, 4) =>
        {
            Ok(
                SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorIntoValues {
                    fragment: inputs[0],
                    values: output,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp8AccumulatorIntoValues
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_gfx950_fp8_accumulator_contract_v1(tcx, rust_inputs[0]).is_some()
                && rust_f32_array_v1(tcx, rust_output, 4) =>
        {
            Ok(
                SemanticCompilerIntrinsicOperationV1::F32MatrixAccumulatorIntoValues {
                    fragment: inputs[0],
                    values: output,
                },
            )
        }
        ProductionTerminalExpansionV1::MatrixMultiplyAccumulate
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::DeviceMatrix)
                })
                && rust_mfma_fragment_contract_v1(tcx, rust_inputs[1]).is_some()
                && rust_mfma_fragment_contract_v1(tcx, rust_inputs[2]).is_some()
                && rust_mfma_accumulator_contract_v1(tcx, rust_inputs[3]).is_some()
                && rust_mfma_accumulator_contract_v1(tcx, rust_output).is_some() =>
        {
            let (Some(lhs), Some(rhs), Some(accumulator)) = (
                rust_mfma_fragment_contract_v1(tcx, rust_inputs[1]),
                rust_mfma_fragment_contract_v1(tcx, rust_inputs[2]),
                rust_mfma_accumulator_contract_v1(tcx, rust_inputs[3]),
            ) else {
                return Err(body_owner_table_mismatch_v1("typed MFMA argument contract"));
            };
            if lhs.role != SemanticMfmaOperandRoleV1::A
                || rhs.role != SemanticMfmaOperandRoleV1::B
                || Some(accumulator) != rust_mfma_accumulator_contract_v1(tcx, rust_output)
            {
                return Err(body_owner_table_mismatch_v1("typed MFMA argument contract"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                    context: pointer_pointee_v1(types, inputs[0])?,
                    lhs_fragment: inputs[1],
                    rhs_fragment: inputs[2],
                    accumulator_fragment: inputs[3],
                    lhs,
                    rhs,
                    accumulator,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp4MultiplyAccumulate
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx950Matrix)
                })
                && rust_gfx950_fp4_fragment_contract_v1(tcx, rust_inputs[1]).is_some()
                && rust_gfx950_fp4_fragment_contract_v1(tcx, rust_inputs[2]).is_some()
                && rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_inputs[3]).is_some()
                && rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_output).is_some() =>
        {
            let (Some(lhs), Some(rhs), Some(accumulator)) = (
                rust_gfx950_fp4_fragment_contract_v1(tcx, rust_inputs[1]),
                rust_gfx950_fp4_fragment_contract_v1(tcx, rust_inputs[2]),
                rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_inputs[3]),
            ) else {
                return Err(body_owner_table_mismatch_v1("gfx950 FP4 MFMA contract"));
            };
            if lhs.role != SemanticMfmaOperandRoleV1::A
                || rhs.role != SemanticMfmaOperandRoleV1::B
                || Some(accumulator) != rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_output)
            {
                return Err(body_owner_table_mismatch_v1("gfx950 FP4 MFMA contract"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                    context: pointer_pointee_v1(types, inputs[0])?,
                    lhs_fragment: inputs[1],
                    rhs_fragment: inputs[2],
                    accumulator_fragment: inputs[3],
                    lhs,
                    rhs,
                    accumulator,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp4Fp8MultiplyAccumulate
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx950Matrix)
                })
                && rust_gfx950_fp4_fragment_contract_v1(tcx, rust_inputs[1]).is_some()
                && rust_gfx950_fp8_fragment_contract_v1(tcx, rust_inputs[2]).is_some()
                && rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_inputs[3]).is_some()
                && rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_output).is_some() =>
        {
            let (Some(lhs), Some(rhs), Some(accumulator)) = (
                rust_gfx950_fp4_fragment_contract_v1(tcx, rust_inputs[1]),
                rust_gfx950_fp8_fragment_contract_v1(tcx, rust_inputs[2]),
                rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_inputs[3]),
            ) else {
                return Err(body_owner_table_mismatch_v1(
                    "gfx950 mixed FP4xFP8 MFMA contract",
                ));
            };
            if lhs.role != SemanticMfmaOperandRoleV1::A
                || rhs.role != SemanticMfmaOperandRoleV1::B
                || Some(accumulator) != rust_gfx950_fp4_accumulator_contract_v1(tcx, rust_output)
            {
                return Err(body_owner_table_mismatch_v1(
                    "gfx950 mixed FP4xFP8 MFMA contract",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                    context: pointer_pointee_v1(types, inputs[0])?,
                    lhs_fragment: inputs[1],
                    rhs_fragment: inputs[2],
                    accumulator_fragment: inputs[3],
                    lhs,
                    rhs,
                    accumulator,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx950Fp8MultiplyAccumulate
            if inputs.len() == 4
                && rust_inputs.len() == 4
                && rust_reference_pointee_v1(rust_inputs[0]).is_some_and(|ty| {
                    rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Gfx950Matrix)
                })
                && rust_gfx950_fp8_fragment_contract_v1(tcx, rust_inputs[1]).is_some()
                && rust_gfx950_fp8_fragment_contract_v1(tcx, rust_inputs[2]).is_some()
                && rust_gfx950_fp8_accumulator_contract_v1(tcx, rust_inputs[3]).is_some()
                && rust_gfx950_fp8_accumulator_contract_v1(tcx, rust_output).is_some() =>
        {
            let (Some(lhs), Some(rhs), Some(accumulator)) = (
                rust_gfx950_fp8_fragment_contract_v1(tcx, rust_inputs[1]),
                rust_gfx950_fp8_fragment_contract_v1(tcx, rust_inputs[2]),
                rust_gfx950_fp8_accumulator_contract_v1(tcx, rust_inputs[3]),
            ) else {
                return Err(body_owner_table_mismatch_v1("gfx950 FP8 MFMA contract"));
            };
            if lhs.role != SemanticMfmaOperandRoleV1::A
                || rhs.role != SemanticMfmaOperandRoleV1::B
                || Some(accumulator) != rust_gfx950_fp8_accumulator_contract_v1(tcx, rust_output)
            {
                return Err(body_owner_table_mismatch_v1("gfx950 FP8 MFMA contract"));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::MatrixMultiplyAccumulate {
                    context: pointer_pointee_v1(types, inputs[0])?,
                    lhs_fragment: inputs[1],
                    rhs_fragment: inputs[2],
                    accumulator_fragment: inputs[3],
                    lhs,
                    rhs,
                    accumulator,
                },
            )
        }
        ProductionTerminalExpansionV1::ThreadIndex1d
            if inputs.is_empty()
                && rust_inputs.is_empty()
                && rust_index_witness_space_v1(
                    tcx,
                    rust_output,
                    TrustedDeviceItem::ThreadIndex,
                ) == Some(SemanticDisjointIndexSpaceV1::Index1d) =>
        {
            let raw_index = aggregate_field_v1(types, output, 0)?;
            Ok(SemanticCompilerIntrinsicOperationV1::ThreadIndex1d {
                index_witness: output,
                raw_index,
            })
        }
        ProductionTerminalExpansionV1::ThreadIndexGet
            if inputs.len() == 1
                && rust_inputs.len() == 1
                && rust_reference_pointee_v1(rust_inputs[0])
                    .and_then(|ty| {
                        rust_index_witness_space_v1(tcx, ty, TrustedDeviceItem::ThreadIndex)
                    })
                    .is_some() =>
        {
            let index_witness = pointer_pointee_v1(types, inputs[0])?;
            Ok(SemanticCompilerIntrinsicOperationV1::ThreadIndexGet {
                index_witness,
                raw_index: output,
            })
        }
        ProductionTerminalExpansionV1::ThreadIndexIntoDisjoint
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let (Some(input_space), Some(output_space)) = (
                rust_index_witness_space_v1(tcx, rust_inputs[0], TrustedDeviceItem::ThreadIndex),
                rust_index_witness_space_v1(tcx, rust_output, TrustedDeviceItem::DisjointIndex),
            ) else {
                return Err(body_owner_table_mismatch_v1("terminal disjoint mapping"));
            };
            if input_space != output_space {
                return Err(body_owner_table_mismatch_v1("terminal disjoint mapping"));
            }
            let raw_index = aggregate_field_v1(types, inputs[0], 0)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::ThreadIndexIntoDisjoint {
                    input_witness: inputs[0],
                    output_witness: output,
                    raw_index,
                    index_space: input_space,
                },
            )
        }
        ProductionTerminalExpansionV1::ThreadIndexCheckedShift
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            checked_shift_operation_v1(
                tcx,
                rust_inputs[0],
                rust_output,
                inputs[0],
                output,
                types,
                TrustedDeviceItem::ThreadIndex,
                true,
            )
        }
        ProductionTerminalExpansionV1::ThreadIndexCheckedBlock
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let input_space =
                rust_index_witness_space_v1(tcx, rust_inputs[0], TrustedDeviceItem::ThreadIndex)
                    .ok_or_else(|| body_owner_table_mismatch_v1("terminal checked-block input"))?;
            let rust_output_block = rust_option_payload_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("terminal checked-block result"))?;
            let (output_space, lanes_per_block, elements_per_lane) =
                rust_disjoint_block_v1(tcx, rust_output_block).ok_or_else(|| {
                    body_owner_table_mismatch_v1("terminal checked-block witness")
                })?;
            if input_space != SemanticDisjointIndexSpaceV1::Index1d {
                return Err(body_owner_table_mismatch_v1(
                    "terminal checked-block input mapping",
                ));
            }
            let output_block = option_payload_v1(types, output)?;
            let raw_index = aggregate_field_v1(types, inputs[0], 0)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedBlock {
                    input_witness: inputs[0],
                    output_block,
                    raw_index,
                    input_space,
                    output_space,
                    lanes_per_block,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::ThreadIndexCheckedTiled2d
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let input_space =
                rust_index_witness_space_v1(tcx, rust_inputs[0], TrustedDeviceItem::ThreadIndex)
                    .ok_or_else(|| {
                        body_owner_table_mismatch_v1("terminal checked-tiled-2d input")
                    })?;
            let rust_output_tile = rust_option_payload_v1(tcx, rust_output)
                .ok_or_else(|| body_owner_table_mismatch_v1("terminal checked-tiled-2d result"))?;
            let (output_space, lanes_per_tile, tile_rows, tile_columns, elements_per_lane) =
                rust_disjoint_tile_2d_v1(tcx, rust_output_tile).ok_or_else(|| {
                    body_owner_table_mismatch_v1("terminal checked-tiled-2d witness")
                })?;
            if input_space != SemanticDisjointIndexSpaceV1::Index1d {
                return Err(body_owner_table_mismatch_v1(
                    "terminal checked-tiled-2d input mapping",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedTiled2d {
                    input_witness: inputs[0],
                    output_tile: option_payload_v1(types, output)?,
                    raw_index: aggregate_field_v1(types, inputs[0], 0)?,
                    input_space,
                    output_space,
                    lanes_per_tile,
                    tile_rows,
                    tile_columns,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::ThreadIndexCheckedRowStriped2d
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let input_space =
                rust_index_witness_space_v1(tcx, rust_inputs[0], TrustedDeviceItem::ThreadIndex)
                    .ok_or_else(|| {
                        body_owner_table_mismatch_v1("terminal checked-row-striped-2d input")
                    })?;
            let rust_output_stripe = rust_option_payload_v1(tcx, rust_output).ok_or_else(|| {
                body_owner_table_mismatch_v1("terminal checked-row-striped-2d result")
            })?;
            let (output_space, lanes_per_row, elements_per_lane) =
                rust_disjoint_row_stripe_2d_v1(tcx, rust_output_stripe).ok_or_else(|| {
                    body_owner_table_mismatch_v1("terminal checked-row-striped-2d witness")
                })?;
            if input_space != SemanticDisjointIndexSpaceV1::Index1d {
                return Err(body_owner_table_mismatch_v1(
                    "terminal checked-row-striped-2d input mapping",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedRowStriped2d {
                    input_witness: inputs[0],
                    output_stripe: option_payload_v1(types, output)?,
                    raw_index: aggregate_field_v1(types, inputs[0], 0)?,
                    input_space,
                    output_space,
                    lanes_per_row,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::DisjointIndexGet
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let Some(rust_witness) = rust_reference_pointee_v1(rust_inputs[0]) else {
                return Err(body_owner_table_mismatch_v1(
                    "terminal disjoint-index receiver",
                ));
            };
            let Some(index_space) =
                rust_index_witness_space_v1(tcx, rust_witness, TrustedDeviceItem::DisjointIndex)
            else {
                return Err(body_owner_table_mismatch_v1("terminal disjoint mapping"));
            };
            let index_witness = pointer_pointee_v1(types, inputs[0])?;
            Ok(SemanticCompilerIntrinsicOperationV1::DisjointIndexGet {
                index_witness,
                raw_index: output,
                index_space,
            })
        }
        ProductionTerminalExpansionV1::DisjointBlockComponentIndex
            if inputs.len() == 2 && rust_inputs.len() == 2 =>
        {
            let (receiver_is_shared, rust_block) = match *rust_inputs[0].kind() {
                TyKind::Ref(_, rust_block, mutability) => {
                    (mutability == rustc_hir::Mutability::Not, Some(rust_block))
                }
                _ => (false, None),
            };
            let (index_space, lanes_per_block, elements_per_lane) =
                require_rust_disjoint_block_component_contract_v1(
                    receiver_is_shared,
                    rust_block.and_then(|rust_block| rust_disjoint_block_v1(tcx, rust_block)),
                    matches!(rust_inputs[1].kind(), TyKind::Uint(UintTy::Usize)),
                    rust_option_payload_v1(tcx, rust_output)
                        .is_some_and(|ty| matches!(ty.kind(), TyKind::Uint(UintTy::Usize))),
                )?;
            let block_witness = pointer_pointee_v1(types, inputs[0])?;
            let raw_index = option_payload_v1(types, output)?;
            if inputs[1] != raw_index {
                return Err(body_owner_table_mismatch_v1(
                    "terminal disjoint-block raw index type",
                ));
            }
            Ok(
                SemanticCompilerIntrinsicOperationV1::DisjointBlockComponentIndex {
                    block_witness,
                    raw_index,
                    index_space,
                    lanes_per_block,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::DisjointIndexCheckedShift
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            checked_shift_operation_v1(
                tcx,
                rust_inputs[0],
                rust_output,
                inputs[0],
                output,
                types,
                TrustedDeviceItem::DisjointIndex,
                false,
            )
        }
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceLen
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let (_, index_space) = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_write_only_disjoint_slice_v1(tcx, ty))
                .ok_or_else(|| {
                    body_owner_table_mismatch_v1("terminal write-only disjoint-slice len")
                })?;
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::WriteOnlyDisjointSliceLen {
                    disjoint_slice,
                    element,
                    raw_index: output,
                    index_space,
                },
            )
        }
        expansion @ (ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWrite
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteDisjoint
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteExclusive
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteBlock
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteTiled2d
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteRowStriped2d) => {
            write_only_disjoint_operation_v1(
                tcx,
                expansion,
                inputs,
                rust_inputs,
                rust_output,
                types,
            )
        }
        ProductionTerminalExpansionV1::DisjointSliceLen
            if inputs.len() == 1 && rust_inputs.len() == 1 =>
        {
            let (_, index_space) = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty))
                .ok_or_else(|| body_owner_table_mismatch_v1("terminal disjoint-slice len"))?;
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(SemanticCompilerIntrinsicOperationV1::DisjointSliceLen {
                disjoint_slice,
                element,
                raw_index: output,
                index_space,
            })
        }
        ProductionTerminalExpansionV1::DisjointSliceGetMut
            if inputs.len() == 2 && rust_inputs.len() == 2 =>
        {
            let rust_slice = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty));
            let rust_index =
                rust_index_witness_space_v1(tcx, rust_inputs[1], TrustedDeviceItem::ThreadIndex);
            if rust_slice.map(|(_, space)| space) != rust_index {
                return Err(body_owner_table_mismatch_v1("terminal disjoint mapping"));
            }
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let index_witness = inputs[1];
            let raw_index = aggregate_field_v1(types, index_witness, 0)?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut {
                disjoint_slice,
                index_witness,
                element,
                raw_index,
            })
        }
        ProductionTerminalExpansionV1::DisjointSliceGetDisjointMut
            if inputs.len() == 2 && rust_inputs.len() == 2 =>
        {
            let rust_slice = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty));
            let rust_index =
                rust_index_witness_space_v1(tcx, rust_inputs[1], TrustedDeviceItem::DisjointIndex);
            let Some(index_space) = rust_index else {
                return Err(body_owner_table_mismatch_v1("terminal disjoint mapping"));
            };
            if rust_slice.map(|(_, space)| space) != Some(index_space) {
                return Err(body_owner_table_mismatch_v1("terminal disjoint mapping"));
            }
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let index_witness = inputs[1];
            let raw_index = aggregate_field_v1(types, index_witness, 0)?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::DisjointSliceGetDisjointMut {
                    disjoint_slice,
                    index_witness,
                    element,
                    raw_index,
                    index_space,
                },
            )
        }
        ProductionTerminalExpansionV1::GridLeaderCurrent
            if inputs.is_empty() && rust_inputs.is_empty() =>
        {
            let Some(rust_leader) = rust_option_payload_v1(tcx, rust_output) else {
                return Err(body_owner_table_mismatch_v1("terminal grid-leader result"));
            };
            if !rust_is_trusted_adt_v1(tcx, rust_leader, TrustedDeviceItem::GridLeader) {
                return Err(body_owner_table_mismatch_v1(
                    "terminal grid-leader identity",
                ));
            }
            Ok(SemanticCompilerIntrinsicOperationV1::GridLeaderCurrent {
                grid_leader: option_payload_v1(types, output)?,
            })
        }
        ProductionTerminalExpansionV1::DisjointSliceGetMutExclusive
            if inputs.len() == 3 && rust_inputs.len() == 3 =>
        {
            let rust_slice = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty));
            let rust_leader = rust_reference_pointee_v1(rust_inputs[1]);
            if rust_slice.map(|(_, space)| space)
                != Some(SemanticDisjointIndexSpaceV1::GridExclusive)
                || rust_leader.is_none_or(|ty| {
                    !rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::GridLeader)
                })
            {
                return Err(body_owner_table_mismatch_v1(
                    "terminal grid-exclusive mapping",
                ));
            }
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let grid_leader = pointer_pointee_v1(types, inputs[1])?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMutExclusive {
                    disjoint_slice,
                    grid_leader,
                    element,
                    raw_index: inputs[2],
                },
            )
        }
        ProductionTerminalExpansionV1::DisjointSliceGetBlockMut
            if inputs.len() == 3 && rust_inputs.len() == 3 =>
        {
            let rust_slice = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty));
            let rust_block = rust_reference_pointee_v1(rust_inputs[1])
                .and_then(|ty| rust_disjoint_block_v1(tcx, ty));
            let Some((index_space, lanes_per_block, elements_per_lane)) = rust_block else {
                return Err(body_owner_table_mismatch_v1(
                    "terminal blocked witness identity",
                ));
            };
            if rust_slice.map(|(_, space)| space) != Some(index_space) {
                return Err(body_owner_table_mismatch_v1(
                    "terminal blocked mapping identity",
                ));
            }
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let block_witness = pointer_pointee_v1(types, inputs[1])?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::DisjointSliceGetBlockMut {
                    disjoint_slice,
                    block_witness,
                    element,
                    raw_index: inputs[2],
                    index_space,
                    lanes_per_block,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::DisjointSliceGetTiled2dMut
            if inputs.len() == 6 && rust_inputs.len() == 6 =>
        {
            let rust_slice = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty));
            let rust_tile = rust_reference_pointee_v1(rust_inputs[1])
                .and_then(|ty| rust_disjoint_tile_2d_v1(tcx, ty));
            let Some((index_space, lanes_per_tile, tile_rows, tile_columns, elements_per_lane)) =
                rust_tile
            else {
                return Err(body_owner_table_mismatch_v1(
                    "terminal tiled-2d witness identity",
                ));
            };
            if rust_slice.map(|(_, space)| space) != Some(index_space) {
                return Err(body_owner_table_mismatch_v1(
                    "terminal tiled-2d mapping identity",
                ));
            }
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let tile_witness = pointer_pointee_v1(types, inputs[1])?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::DisjointSliceGetTiled2dMut {
                    disjoint_slice,
                    tile_witness,
                    element,
                    raw_index: inputs[2],
                    index_space,
                    lanes_per_tile,
                    tile_rows,
                    tile_columns,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::DisjointSliceGetRowStriped2dMut
            if inputs.len() == 6 && rust_inputs.len() == 6 =>
        {
            let rust_slice = rust_reference_pointee_v1(rust_inputs[0])
                .and_then(|ty| rust_disjoint_slice_v1(tcx, ty));
            let rust_stripe = rust_reference_pointee_v1(rust_inputs[1])
                .and_then(|ty| rust_disjoint_row_stripe_2d_v1(tcx, ty));
            let Some((index_space, lanes_per_row, elements_per_lane)) = rust_stripe else {
                return Err(body_owner_table_mismatch_v1(
                    "terminal row-striped-2d witness identity",
                ));
            };
            if rust_slice.map(|(_, space)| space) != Some(index_space) {
                return Err(body_owner_table_mismatch_v1(
                    "terminal row-striped-2d mapping identity",
                ));
            }
            let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
            let stripe_witness = pointer_pointee_v1(types, inputs[1])?;
            let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
            let element = pointer_pointee_v1(types, element_pointer)?;
            Ok(
                SemanticCompilerIntrinsicOperationV1::DisjointSliceGetRowStriped2dMut {
                    disjoint_slice,
                    stripe_witness,
                    element,
                    raw_index: inputs[2],
                    index_space,
                    lanes_per_row,
                    elements_per_lane,
                },
            )
        }
        ProductionTerminalExpansionV1::Gfx942OrderedProgramE32
        | ProductionTerminalExpansionV1::Gfx942OrderedXorAddE32
        | ProductionTerminalExpansionV1::Gfx942InlineU32(_)
        | ProductionTerminalExpansionV1::ThreadIndex(_)
        | ProductionTerminalExpansionV1::WorkgroupIndex(_)
        | ProductionTerminalExpansionV1::WorkgroupDimension(_)
        | ProductionTerminalExpansionV1::GridDimension(_)
        | ProductionTerminalExpansionV1::ThreadIndex1d
        | ProductionTerminalExpansionV1::ThreadIndexGet
        | ProductionTerminalExpansionV1::ThreadIndexIntoDisjoint
        | ProductionTerminalExpansionV1::ThreadIndexCheckedShift
        | ProductionTerminalExpansionV1::DisjointIndexGet
        | ProductionTerminalExpansionV1::DisjointBlockComponentIndex
        | ProductionTerminalExpansionV1::DisjointIndexCheckedShift
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceLen
        | ProductionTerminalExpansionV1::DisjointSliceLen
        | ProductionTerminalExpansionV1::DisjointSliceGetMut
        | ProductionTerminalExpansionV1::DisjointSliceGetDisjointMut
        | ProductionTerminalExpansionV1::GridLeaderCurrent
        | ProductionTerminalExpansionV1::DisjointSliceGetMutExclusive
        | ProductionTerminalExpansionV1::ThreadIndexCheckedBlock
        | ProductionTerminalExpansionV1::ThreadIndexCheckedTiled2d
        | ProductionTerminalExpansionV1::ThreadIndexCheckedRowStriped2d
        | ProductionTerminalExpansionV1::DisjointSliceGetBlockMut
        | ProductionTerminalExpansionV1::DisjointSliceGetTiled2dMut
        | ProductionTerminalExpansionV1::DisjointSliceGetRowStriped2dMut
        | ProductionTerminalExpansionV1::MathContextCurrent
        | ProductionTerminalExpansionV1::MathF32(_)
        | ProductionTerminalExpansionV1::RustcFabsF32
        | ProductionTerminalExpansionV1::RustcSaturatingInteger(_)
        | ProductionTerminalExpansionV1::WorkgroupCollectiveContextCurrent
        | ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum
        | ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum
        | ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum
        | ProductionTerminalExpansionV1::CollectiveContextCurrent
        | ProductionTerminalExpansionV1::WorkgroupReduceSum
        | ProductionTerminalExpansionV1::SubgroupReduceSumF32
        | ProductionTerminalExpansionV1::SubgroupReduceMaxF32
        | ProductionTerminalExpansionV1::Gfx950SubgroupCurrent
        | ProductionTerminalExpansionV1::Gfx950SubgroupReduceMaxF32
        | ProductionTerminalExpansionV1::Gfx950SubgroupReduceSumF32
        | ProductionTerminalExpansionV1::Gfx950SubgroupBroadcastF32
        | ProductionTerminalExpansionV1::WaveLaneCurrent
        | ProductionTerminalExpansionV1::MatrixContextCurrent
        | ProductionTerminalExpansionV1::Bf16MatrixARowMajor
        | ProductionTerminalExpansionV1::Bf16MatrixBRowMajor
        | ProductionTerminalExpansionV1::Bf16MatrixBColumnMajor
        | ProductionTerminalExpansionV1::Bf16MatrixALoadZeroFilledV2
        | ProductionTerminalExpansionV1::Bf16MatrixBLoadZeroFilledV2
        | ProductionTerminalExpansionV1::Bf16MatrixBColumnMajorLoadZeroFilledV1
        | ProductionTerminalExpansionV1::StridedReadView2DFromSharedSlice
        | ProductionTerminalExpansionV1::StridedReadView2DLoadOr
        | ProductionTerminalExpansionV1::WorkgroupLdsScopeCurrent
        | ProductionTerminalExpansionV1::DynamicLdsExactCurrent
        | ProductionTerminalExpansionV1::DynamicLdsIntoCollectiveRawParts
        | ProductionTerminalExpansionV1::WorkgroupPipelineCurrent
        | ProductionTerminalExpansionV1::WorkgroupPipelineStage
        | ProductionTerminalExpansionV1::WorkgroupPipelineWrite
        | ProductionTerminalExpansionV1::WorkgroupPipelineCommit
        | ProductionTerminalExpansionV1::WorkgroupPipelineWait
        | ProductionTerminalExpansionV1::WorkgroupPipelineConsume
        | ProductionTerminalExpansionV1::WorkgroupPipelineRead
        | ProductionTerminalExpansionV1::WorkgroupPipelineDiscard
        | ProductionTerminalExpansionV1::WorkgroupPipelineRelease
        | ProductionTerminalExpansionV1::F32MatrixAccumulatorZero
        | ProductionTerminalExpansionV1::F32MatrixAccumulatorIntoValues
        | ProductionTerminalExpansionV1::MatrixMultiplyAccumulate
        | ProductionTerminalExpansionV1::Gfx950MatrixContextCurrent
        | ProductionTerminalExpansionV1::Gfx950Fp4MatrixARowMajor
        | ProductionTerminalExpansionV1::Gfx950Fp4MatrixBRowMajor
        | ProductionTerminalExpansionV1::Gfx950Fp4MatrixALoadM16K128
        | ProductionTerminalExpansionV1::Gfx950Fp4MatrixBLoadK128N16
        | ProductionTerminalExpansionV1::Gfx950Fp4AccumulatorZero
        | ProductionTerminalExpansionV1::Gfx950Fp4AccumulatorIntoValues
        | ProductionTerminalExpansionV1::Gfx950Fp4MultiplyAccumulate
        | ProductionTerminalExpansionV1::Gfx950Fp4Fp8MultiplyAccumulate
        | ProductionTerminalExpansionV1::Gfx950Fp8MatrixARowMajor
        | ProductionTerminalExpansionV1::Gfx950Fp8MatrixBRowMajor
        | ProductionTerminalExpansionV1::Gfx950Fp8MatrixALoadM16K128
        | ProductionTerminalExpansionV1::Gfx950Fp8MatrixBLoadK128N16
        | ProductionTerminalExpansionV1::Gfx950Fp8AccumulatorZero
        | ProductionTerminalExpansionV1::Gfx950Fp8AccumulatorIntoValues
        | ProductionTerminalExpansionV1::Gfx950Fp8MultiplyAccumulate
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeTileCurrent
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB4
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeStageB8
        | ProductionTerminalExpansionV1::Gfx950LdsTransposePublish
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB4
        | ProductionTerminalExpansionV1::Gfx950LdsTransposeReadB8
        | ProductionTerminalExpansionV1::MemoryVolatileLoad
        | ProductionTerminalExpansionV1::Trap
        | ProductionTerminalExpansionV1::ColdPath
        | ProductionTerminalExpansionV1::WorkgroupBarrier => {
            Err(body_owner_table_mismatch_v1("terminal callable ABI"))
        }
    }
}

fn write_only_disjoint_operation_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    expansion: crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1,
    inputs: &[SemanticTypeIdV1],
    rust_inputs: &[Ty<'tcx>],
    rust_output: Ty<'tcx>,
    types: &[SemanticTypeDeclV1],
) -> Result<SemanticCompilerIntrinsicOperationV1, ProductionSemanticImportErrorV1> {
    use crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1;

    if !matches!(rust_output.kind(), TyKind::Bool) || inputs.is_empty() || rust_inputs.is_empty() {
        return Err(body_owner_table_mismatch_v1(
            "write-only disjoint terminal ABI",
        ));
    }
    let (rust_element, index_space) = rust_reference_pointee_v1(rust_inputs[0])
        .and_then(|ty| rust_write_only_disjoint_slice_v1(tcx, ty))
        .ok_or_else(|| body_owner_table_mismatch_v1("write-only disjoint receiver"))?;
    let disjoint_slice = pointer_pointee_v1(types, inputs[0])?;
    let element_pointer = aggregate_field_v1(types, disjoint_slice, 0)?;
    let element = pointer_pointee_v1(types, element_pointer)?;

    let (witness, raw_index, kind, value_argument) = match expansion {
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWrite
        | ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteDisjoint => {
            if inputs.len() != 3 || rust_inputs.len() != 3 || rust_inputs[2] != rust_element {
                return Err(body_owner_table_mismatch_v1(
                    "write-only direct write signature",
                ));
            }
            let disjoint =
                expansion == ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteDisjoint;
            let trusted = if disjoint {
                TrustedDeviceItem::DisjointIndex
            } else {
                TrustedDeviceItem::ThreadIndex
            };
            if rust_index_witness_space_v1(tcx, rust_inputs[1], trusted) != Some(index_space) {
                return Err(body_owner_table_mismatch_v1(
                    "write-only direct write mapping",
                ));
            }
            (
                inputs[1],
                aggregate_field_v1(types, inputs[1], 0)?,
                SemanticWriteOnlyDisjointWriteKindV1::Thread { disjoint },
                2,
            )
        }
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteExclusive => {
            if inputs.len() != 4 || rust_inputs.len() != 4 || rust_inputs[3] != rust_element {
                return Err(body_owner_table_mismatch_v1(
                    "write-only exclusive write signature",
                ));
            }
            let rust_leader = rust_reference_pointee_v1(rust_inputs[1]);
            if index_space != SemanticDisjointIndexSpaceV1::GridExclusive
                || rust_leader.is_none_or(|ty| {
                    !rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::GridLeader)
                })
            {
                return Err(body_owner_table_mismatch_v1("write-only exclusive mapping"));
            }
            (
                pointer_pointee_v1(types, inputs[1])?,
                inputs[2],
                SemanticWriteOnlyDisjointWriteKindV1::GridExclusive,
                3,
            )
        }
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteBlock => {
            if inputs.len() != 4 || rust_inputs.len() != 4 || rust_inputs[3] != rust_element {
                return Err(body_owner_table_mismatch_v1(
                    "write-only blocked write signature",
                ));
            }
            let Some((expected, lanes_per_block, elements_per_lane)) =
                rust_reference_pointee_v1(rust_inputs[1])
                    .and_then(|ty| rust_disjoint_block_v1(tcx, ty))
            else {
                return Err(body_owner_table_mismatch_v1("write-only blocked witness"));
            };
            if index_space != expected {
                return Err(body_owner_table_mismatch_v1("write-only blocked mapping"));
            }
            (
                pointer_pointee_v1(types, inputs[1])?,
                inputs[2],
                SemanticWriteOnlyDisjointWriteKindV1::Block {
                    lanes_per_block,
                    elements_per_lane,
                },
                3,
            )
        }
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteTiled2d => {
            if inputs.len() != 7 || rust_inputs.len() != 7 || rust_inputs[6] != rust_element {
                return Err(body_owner_table_mismatch_v1(
                    "write-only tiled write signature",
                ));
            }
            let Some((expected, lanes_per_tile, tile_rows, tile_columns, elements_per_lane)) =
                rust_reference_pointee_v1(rust_inputs[1])
                    .and_then(|ty| rust_disjoint_tile_2d_v1(tcx, ty))
            else {
                return Err(body_owner_table_mismatch_v1("write-only tiled witness"));
            };
            if index_space != expected {
                return Err(body_owner_table_mismatch_v1("write-only tiled mapping"));
            }
            (
                pointer_pointee_v1(types, inputs[1])?,
                inputs[2],
                SemanticWriteOnlyDisjointWriteKindV1::Tiled2d {
                    lanes_per_tile,
                    tile_rows,
                    tile_columns,
                    elements_per_lane,
                },
                6,
            )
        }
        ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteRowStriped2d => {
            if inputs.len() != 7 || rust_inputs.len() != 7 || rust_inputs[6] != rust_element {
                return Err(body_owner_table_mismatch_v1(
                    "write-only row-striped write signature",
                ));
            }
            let Some((expected, lanes_per_row, elements_per_lane)) =
                rust_reference_pointee_v1(rust_inputs[1])
                    .and_then(|ty| rust_disjoint_row_stripe_2d_v1(tcx, ty))
            else {
                return Err(body_owner_table_mismatch_v1(
                    "write-only row-striped witness",
                ));
            };
            if index_space != expected {
                return Err(body_owner_table_mismatch_v1(
                    "write-only row-striped mapping",
                ));
            }
            (
                pointer_pointee_v1(types, inputs[1])?,
                inputs[2],
                SemanticWriteOnlyDisjointWriteKindV1::RowStriped2d {
                    lanes_per_row,
                    elements_per_lane,
                },
                6,
            )
        }
        _ => {
            return Err(body_owner_table_mismatch_v1(
                "write-only disjoint terminal expansion",
            ));
        }
    };
    if inputs[value_argument] != element {
        return Err(body_owner_table_mismatch_v1(
            "write-only disjoint semantic element",
        ));
    }
    Ok(
        SemanticCompilerIntrinsicOperationV1::WriteOnlyDisjointSliceWrite {
            disjoint_slice,
            witness,
            element,
            raw_index,
            index_space,
            kind,
        },
    )
}

fn semantic_f32_math_function_v1(
    function: fe2o3_kernel_ir::F32MathFunction,
) -> SemanticF32MathFunctionV1 {
    use fe2o3_kernel_ir::F32MathFunction as Kernel;
    match function {
        Kernel::Sqrt => SemanticF32MathFunctionV1::Sqrt,
        Kernel::FusedMultiplyAdd => SemanticF32MathFunctionV1::FusedMultiplyAdd,
        Kernel::Floor => SemanticF32MathFunctionV1::Floor,
        Kernel::Ceil => SemanticF32MathFunctionV1::Ceil,
        Kernel::Truncate => SemanticF32MathFunctionV1::Truncate,
        Kernel::RoundTiesEven => SemanticF32MathFunctionV1::RoundTiesEven,
        Kernel::Sin => SemanticF32MathFunctionV1::Sin,
        Kernel::Cos => SemanticF32MathFunctionV1::Cos,
        Kernel::Exp => SemanticF32MathFunctionV1::Exp,
        Kernel::Exp2 => SemanticF32MathFunctionV1::Exp2,
        Kernel::Ln => SemanticF32MathFunctionV1::Ln,
        Kernel::Log2 => SemanticF32MathFunctionV1::Log2,
        Kernel::Log10 => SemanticF32MathFunctionV1::Log10,
        Kernel::Abs => unreachable!("rustc fabs has a distinct semantic intrinsic"),
    }
}

fn single_const_u32_v1(instance: Instance<'_>) -> Option<u32> {
    let mut values = instance
        .args
        .iter()
        .filter_map(|argument| argument.as_const())
        .filter_map(|value| value.try_to_leaf())
        .map(|value| value.to_bits(value.size()));
    let value = u32::try_from(values.next()?).ok()?;
    values.next().is_none().then_some(value)
}

fn semantic_u16_type_v1(types: &[SemanticTypeDeclV1], ty: SemanticTypeIdV1) -> bool {
    types.get(ty.index() as usize).is_some_and(|declaration| {
        matches!(
            declaration.shape(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 16,
            })
        )
    })
}

fn semantic_f32_type_v1(types: &[SemanticTypeDeclV1], ty: SemanticTypeIdV1) -> bool {
    types.get(ty.index() as usize).is_some_and(|declaration| {
        matches!(
            declaration.shape(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float { bits: 32 })
        )
    })
}

fn semantic_exact_inhabited_aggregate_zst_v1(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> bool {
    let Some(declaration) = types.get(ty.index() as usize) else {
        return false;
    };
    let SemanticTypeShapeV1::Aggregate(aggregate) = declaration.shape() else {
        return false;
    };
    let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details() else {
        return false;
    };
    declaration.layout().size_bytes() == Some(0)
        && !declaration.layout().is_uninhabited()
        && matches!(
            declaration.layout().backend_repr(),
            SemanticBackendReprV1::Memory { sized: true }
        )
        && aggregate.fields().len() == layout.field_offsets().len()
        && layout.field_offsets().iter().all(|offset| *offset == 0)
        && layout.padding().is_empty()
        && aggregate.fields().iter().all(|field| {
            types.get(field.index() as usize).is_some_and(|field| {
                field.layout().size_bytes() == Some(0) && !field.layout().is_uninhabited()
            })
        })
}

fn semantic_bf16_storage_type_v1(types: &[SemanticTypeDeclV1], ty: SemanticTypeIdV1) -> bool {
    let Some(declaration) = types.get(ty.index() as usize) else {
        return false;
    };
    let SemanticTypeShapeV1::Aggregate(aggregate) = declaration.shape() else {
        return false;
    };
    let [bits] = aggregate.fields() else {
        return false;
    };
    let Some(bits_declaration) = types.get(bits.index() as usize) else {
        return false;
    };
    let SemanticTypeLayoutDetailsV1::Aggregate(layout) = declaration.layout().details() else {
        return false;
    };
    semantic_u16_type_v1(types, *bits)
        && declaration.layout().size_bytes() == Some(2)
        && declaration.layout().alignment_bytes() == 2
        && !declaration.layout().is_uninhabited()
        && bits_declaration.layout().size_bytes() == Some(2)
        && bits_declaration.layout().alignment_bytes() == 2
        && !bits_declaration.layout().is_uninhabited()
        && declaration.layout().backend_repr() == bits_declaration.layout().backend_repr()
        && layout.field_offsets() == [0]
        && layout.padding().is_empty()
}

fn single_const_u64_v1(instance: Instance<'_>) -> Option<u64> {
    let mut values = instance
        .args
        .iter()
        .filter_map(|argument| argument.as_const())
        .filter_map(|value| value.try_to_leaf())
        .map(|value| value.to_bits(value.size()));
    let value = u64::try_from(values.next()?).ok()?;
    values.next().is_none().then_some(value)
}

fn rust_workgroup_pipeline_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(Ty<'tcx>, u32, u64, u32)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if !tcx.is_diagnostic_item(
        Symbol::intern("fe2o3_device_workgroup_pipeline_v1"),
        definition.did(),
    ) {
        return None;
    }
    let type_arguments = arguments.types().collect::<Vec<_>>();
    let [element] = type_arguments.as_slice() else {
        return None;
    };
    let constants = arguments
        .iter()
        .filter_map(|argument| argument.as_const())
        .map(|constant| constant.try_to_target_usize(tcx))
        .collect::<Option<Vec<_>>>()?;
    let [buffers, elements, prefetch_distance] = constants.as_slice() else {
        return None;
    };
    Some((
        *element,
        u32::try_from(*buffers).ok()?,
        *elements,
        u32::try_from(*prefetch_distance).ok()?,
    ))
}

fn rust_dynamic_lds_scalar_element_v1<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if !tcx.is_diagnostic_item(Symbol::intern("fe2o3_device_dynamic_lds"), definition.did()) {
        return None;
    }
    let arguments = arguments.types().collect::<Vec<_>>();
    let [element, _state] = arguments.as_slice() else {
        return None;
    };
    matches!(
        element.kind(),
        TyKind::Uint(UintTy::U8 | UintTy::U16 | UintTy::U32 | UintTy::U64 | UintTy::Usize)
            | TyKind::Int(IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64 | IntTy::Isize)
            | TyKind::Float(FloatTy::F32 | FloatTy::F64)
    )
    .then_some(*element)
}

fn rust_dynamic_lds_uninitialized_scalar_element_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<Ty<'tcx>> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if !tcx.is_diagnostic_item(Symbol::intern("fe2o3_device_dynamic_lds"), definition.did()) {
        return None;
    }
    let arguments = arguments.types().collect::<Vec<_>>();
    let [element, state] = arguments.as_slice() else {
        return None;
    };
    if !rust_is_trusted_adt_v1(tcx, *state, TrustedDeviceItem::LdsUninitialized)
        || !matches!(
            element.kind(),
            TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) | TyKind::Float(FloatTy::F32)
        )
    {
        return None;
    }
    Some(*element)
}

fn rust_dynamic_lds_raw_parts_v1<'tcx>(output: Ty<'tcx>, element: Ty<'tcx>) -> bool {
    let TyKind::Tuple(fields) = output.kind() else {
        return false;
    };
    fields.len() == 2
        && matches!(
            fields[0].kind(),
            TyKind::RawPtr(pointee, rustc_hir::Mutability::Mut) if *pointee == element
        )
        && matches!(fields[1].kind(), TyKind::Uint(UintTy::Usize))
}

#[allow(clippy::too_many_arguments)]
fn checked_shift_operation_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    rust_input: Ty<'tcx>,
    rust_output: Ty<'tcx>,
    input_witness: SemanticTypeIdV1,
    output: SemanticTypeIdV1,
    types: &[SemanticTypeDeclV1],
    input_kind: TrustedDeviceItem,
    thread_index: bool,
) -> Result<SemanticCompilerIntrinsicOperationV1, ProductionSemanticImportErrorV1> {
    let input_space = rust_index_witness_space_v1(tcx, rust_input, input_kind)
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal checked-shift input"))?;
    let rust_output_witness = rust_option_payload_v1(tcx, rust_output)
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal checked-shift result"))?;
    let output_space =
        rust_index_witness_space_v1(tcx, rust_output_witness, TrustedDeviceItem::DisjointIndex)
            .ok_or_else(|| body_owner_table_mismatch_v1("terminal checked-shift output"))?;
    let offset = match (input_space, output_space) {
        (
            SemanticDisjointIndexSpaceV1::Index1d,
            SemanticDisjointIndexSpaceV1::ShiftedIndex1d { offset },
        ) => offset,
        _ => {
            return Err(body_owner_table_mismatch_v1(
                "terminal checked-shift mapping",
            ));
        }
    };
    let output_witness = option_payload_v1(types, output)?;
    let raw_index = aggregate_field_v1(types, input_witness, 0)?;
    Ok(if thread_index {
        SemanticCompilerIntrinsicOperationV1::ThreadIndexCheckedShift {
            input_witness,
            output_witness,
            raw_index,
            input_space,
            output_space,
            offset,
        }
    } else {
        SemanticCompilerIntrinsicOperationV1::DisjointIndexCheckedShift {
            input_witness,
            output_witness,
            raw_index,
            input_space,
            output_space,
            offset,
        }
    })
}

fn rust_reference_pointee_v1(ty: Ty<'_>) -> Option<Ty<'_>> {
    match *ty.kind() {
        TyKind::Ref(_, pointee, _) => Some(pointee),
        _ => None,
    }
}

fn rust_workgroup_collective_scratch_element_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<Ty<'tcx>> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did())
        != Some(TrustedDeviceItem::WorkgroupCollectiveScratch)
    {
        return None;
    }
    let type_arguments = arguments.types().collect::<Vec<_>>();
    let [element] = type_arguments.as_slice() else {
        return None;
    };
    Some(*element)
}

fn rust_shared_u16_slice_v1(ty: Ty<'_>) -> bool {
    let Some(pointee) = rust_reference_pointee_v1(ty) else {
        return false;
    };
    matches!(*pointee.kind(), TyKind::Slice(element)
        if matches!(element.kind(), TyKind::Uint(UintTy::U16)))
}

fn rust_shared_u8_slice_v1(ty: Ty<'_>) -> bool {
    let Some(pointee) = rust_reference_pointee_v1(ty) else {
        return false;
    };
    matches!(*pointee.kind(), TyKind::Slice(element)
        if matches!(element.kind(), TyKind::Uint(UintTy::U8)))
}

fn rust_shared_slice_element_v1(ty: Ty<'_>) -> Option<Ty<'_>> {
    let TyKind::Ref(_, pointee, rustc_hir::Mutability::Not) = *ty.kind() else {
        return None;
    };
    match *pointee.kind() {
        TyKind::Slice(element) => Some(element),
        _ => None,
    }
}

fn rust_supported_read_view_scalar_v1(ty: Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        TyKind::Bool
            | TyKind::Char
            | TyKind::Int(IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64)
            | TyKind::Uint(UintTy::U8 | UintTy::U16 | UintTy::U32 | UintTy::U64)
            | TyKind::Float(FloatTy::F32 | FloatTy::F64)
    )
}

fn rust_supported_volatile_load_scalar_v1(ty: Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        TyKind::Bool
            | TyKind::Int(IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64)
            | TyKind::Uint(UintTy::U8 | UintTy::U16 | UintTy::U32 | UintTy::U64)
            | TyKind::Float(FloatTy::F32 | FloatTy::F64)
    )
}

pub(crate) fn rust_option_payload_v1<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    (tcx.is_diagnostic_item(sym::Option, definition.did()) && arguments.len() == 1)
        .then(|| arguments[0].as_type())
        .flatten()
}

fn rust_result_payloads_v1<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    (tcx.is_diagnostic_item(sym::Result, definition.did()) && arguments.len() == 2)
        .then(|| Some((arguments[0].as_type()?, arguments[1].as_type()?)))
        .flatten()
}

fn rust_is_trusted_adt_v1(tcx: TyCtxt<'_>, ty: Ty<'_>, item: TrustedDeviceItem) -> bool {
    matches!(*ty.kind(), TyKind::Adt(definition, arguments)
        if arguments
            .iter()
            .all(|argument| matches!(argument.kind(), GenericArgKind::Lifetime(_)))
            && trusted_device_items::classify(tcx, definition.did()) == Some(item))
}

fn rust_trusted_adt_type_arguments_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    item: TrustedDeviceItem,
) -> Option<Vec<Ty<'tcx>>> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    (trusted_device_items::classify(tcx, definition.did()) == Some(item))
        .then(|| arguments.types().collect())
}

fn rust_is_exact_trusted_marker_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    item: TrustedDeviceItem,
) -> bool {
    rust_trusted_adt_type_arguments_v1(tcx, ty, item).is_some_and(|arguments| arguments.is_empty())
}

fn rust_wave_lane64_v1<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::WaveLane).is_some_and(
        |arguments| {
            matches!(arguments.as_slice(), [width]
                if rust_is_exact_trusted_marker_v1(tcx, *width, TrustedDeviceItem::Wave64))
        },
    )
}

fn rust_mfma_fragment_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaOperandContractV1> {
    let arguments =
        rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::Bf16MfmaFragment)?;
    let [role, profile, distribution, width] = arguments.as_slice() else {
        return None;
    };
    let role = if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::MfmaOperandA) {
        SemanticMfmaOperandRoleV1::A
    } else if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::MfmaOperandB) {
        SemanticMfmaOperandRoleV1::B
    } else {
        return None;
    };
    rust_is_exact_trusted_marker_v1(tcx, *profile, TrustedDeviceItem::Bf16MfmaProfile)
        .then_some(())?;
    rust_is_exact_trusted_marker_v1(tcx, *distribution, TrustedDeviceItem::MfmaRegisterTile16x16)
        .then_some(())?;
    rust_is_exact_trusted_marker_v1(tcx, *width, TrustedDeviceItem::Wave64).then_some(())?;
    Some(SemanticMfmaOperandContractV1 {
        role,
        profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
        register_distribution: SemanticMfmaRegisterDistributionV1::Tile16x16,
        wave_width: 64,
    })
}

fn rust_mfma_accumulator_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaAccumulatorContractV1> {
    let arguments =
        rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::F32AccumulatorFragment)?;
    let [profile, distribution, width] = arguments.as_slice() else {
        return None;
    };
    rust_is_exact_trusted_marker_v1(tcx, *profile, TrustedDeviceItem::Bf16MfmaProfile)
        .then_some(())?;
    rust_is_exact_trusted_marker_v1(
        tcx,
        *distribution,
        TrustedDeviceItem::MfmaAccumulatorRowMajor,
    )
    .then_some(())?;
    rust_is_exact_trusted_marker_v1(tcx, *width, TrustedDeviceItem::Wave64).then_some(())?;
    Some(SemanticMfmaAccumulatorContractV1 {
        profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
        distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
        wave_width: 64,
    })
}

fn rust_mfma_matrix_role_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaOperandRoleV1> {
    if rust_is_trusted_adt_v1(tcx, ty, TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView) {
        return Some(SemanticMfmaOperandRoleV1::B);
    }
    let arguments =
        rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::Bf16MfmaMatrixView)?;
    let [role] = arguments.as_slice() else {
        return None;
    };
    if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::MfmaOperandA) {
        Some(SemanticMfmaOperandRoleV1::A)
    } else if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::MfmaOperandB) {
        Some(SemanticMfmaOperandRoleV1::B)
    } else {
        None
    }
}

fn rust_gfx950_fp8_fragment_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaOperandContractV1> {
    let arguments =
        rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::Gfx950MfmaFragment)?;
    let [format, role] = arguments.as_slice() else {
        return None;
    };
    rust_is_exact_trusted_marker_v1(tcx, *format, TrustedDeviceItem::Gfx950Fp8E4M3Format)
        .then_some(())?;
    let role = if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::Gfx950MfmaOperandA)
    {
        SemanticMfmaOperandRoleV1::A
    } else if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::Gfx950MfmaOperandB) {
        SemanticMfmaOperandRoleV1::B
    } else {
        return None;
    };
    Some(SemanticMfmaOperandContractV1 {
        role,
        profile: SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128,
        register_distribution: SemanticMfmaRegisterDistributionV1::Gfx950M16N16K128,
        wave_width: 64,
    })
}

fn rust_gfx950_fp4_fragment_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaOperandContractV1> {
    let arguments =
        rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::Gfx950MfmaFragment)?;
    let [format, role] = arguments.as_slice() else {
        return None;
    };
    rust_is_exact_trusted_marker_v1(tcx, *format, TrustedDeviceItem::Gfx950Fp4E2M1Format)
        .then_some(())?;
    let role = if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::Gfx950MfmaOperandA)
    {
        SemanticMfmaOperandRoleV1::A
    } else if rust_is_exact_trusted_marker_v1(tcx, *role, TrustedDeviceItem::Gfx950MfmaOperandB) {
        SemanticMfmaOperandRoleV1::B
    } else {
        return None;
    };
    Some(SemanticMfmaOperandContractV1 {
        role,
        profile: SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128,
        register_distribution: SemanticMfmaRegisterDistributionV1::Gfx950M16N16K128,
        wave_width: 64,
    })
}

fn rust_gfx950_fp8_accumulator_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaAccumulatorContractV1> {
    let arguments = rust_trusted_adt_type_arguments_v1(
        tcx,
        ty,
        TrustedDeviceItem::Gfx950F32AccumulatorFragment,
    )?;
    let [format] = arguments.as_slice() else {
        return None;
    };
    rust_is_exact_trusted_marker_v1(tcx, *format, TrustedDeviceItem::Gfx950Fp8E4M3Format)
        .then_some(())?;
    Some(SemanticMfmaAccumulatorContractV1 {
        profile: SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128,
        distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
        wave_width: 64,
    })
}

fn rust_gfx950_fp4_accumulator_contract_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaAccumulatorContractV1> {
    let arguments = rust_trusted_adt_type_arguments_v1(
        tcx,
        ty,
        TrustedDeviceItem::Gfx950F32AccumulatorFragment,
    )?;
    let [format] = arguments.as_slice() else {
        return None;
    };
    rust_is_exact_trusted_marker_v1(tcx, *format, TrustedDeviceItem::Gfx950Fp4E2M1Format)
        .then_some(())?;
    Some(SemanticMfmaAccumulatorContractV1 {
        profile: SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128,
        distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
        wave_width: 64,
    })
}

fn rust_gfx950_fp8_matrix_role_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaOperandRoleV1> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    let role = match trusted_device_items::classify(tcx, definition.did())? {
        TrustedDeviceItem::Gfx950MfmaMatrixAView => SemanticMfmaOperandRoleV1::A,
        TrustedDeviceItem::Gfx950MfmaMatrixBView => SemanticMfmaOperandRoleV1::B,
        _ => return None,
    };
    let mut formats = arguments.types();
    let format = formats.next()?;
    if formats.next().is_some()
        || !rust_is_exact_trusted_marker_v1(tcx, format, TrustedDeviceItem::Gfx950Fp8E4M3Format)
    {
        return None;
    }
    Some(role)
}

fn rust_gfx950_fp4_matrix_role_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticMfmaOperandRoleV1> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    let role = match trusted_device_items::classify(tcx, definition.did())? {
        TrustedDeviceItem::Gfx950MfmaMatrixAView => SemanticMfmaOperandRoleV1::A,
        TrustedDeviceItem::Gfx950MfmaMatrixBView => SemanticMfmaOperandRoleV1::B,
        _ => return None,
    };
    let mut formats = arguments.types();
    let format = formats.next()?;
    if formats.next().is_some()
        || !rust_is_exact_trusted_marker_v1(tcx, format, TrustedDeviceItem::Gfx950Fp4E2M1Format)
    {
        return None;
    }
    Some(role)
}

fn rust_gfx950_lds_transpose_format_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticGfx950LdsTransposeFormatV1> {
    let arguments =
        rust_trusted_adt_type_arguments_v1(tcx, ty, TrustedDeviceItem::Gfx950LdsTransposeTile)?;
    let [format, _state] = arguments.as_slice() else {
        return None;
    };
    if rust_is_exact_trusted_marker_v1(tcx, *format, TrustedDeviceItem::Gfx950Fp4E2M1Format) {
        Some(SemanticGfx950LdsTransposeFormatV1::Fp4E2M1)
    } else if rust_is_exact_trusted_marker_v1(tcx, *format, TrustedDeviceItem::Gfx950Fp8E4M3Format)
    {
        Some(SemanticGfx950LdsTransposeFormatV1::Fp8E4M3)
    } else {
        None
    }
}

fn rust_f32_array_v1<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, expected_length: u64) -> bool {
    let TyKind::Array(element, length) = *ty.kind() else {
        return false;
    };
    matches!(element.kind(), TyKind::Float(FloatTy::F32))
        && length.try_to_target_usize(tcx) == Some(expected_length)
}

fn rust_disjoint_slice_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(Ty<'tcx>, SemanticDisjointIndexSpaceV1)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did())
        != Some(TrustedDeviceItem::DisjointSlice)
        || arguments.len() != 2
    {
        return None;
    }
    Some((
        arguments[0].as_type()?,
        rust_disjoint_index_space_v1(tcx, arguments[1].as_type()?)?,
    ))
}

fn rust_write_only_disjoint_slice_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(Ty<'tcx>, SemanticDisjointIndexSpaceV1)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did())
        != Some(TrustedDeviceItem::WriteOnlyDisjointSlice)
        || arguments.len() != 2
    {
        return None;
    }
    Some((
        arguments[0].as_type()?,
        rust_disjoint_index_space_v1(tcx, arguments[1].as_type()?)?,
    ))
}

pub(crate) fn rust_index_witness_space_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    item: TrustedDeviceItem,
) -> Option<SemanticDisjointIndexSpaceV1> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did()) != Some(item) || arguments.len() != 1 {
        return None;
    }
    rust_disjoint_index_space_v1(tcx, arguments[0].as_type()?)
}

fn rust_disjoint_index_space_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<SemanticDisjointIndexSpaceV1> {
    if ty == trusted_index1d_type_v1(tcx)? {
        return Some(SemanticDisjointIndexSpaceV1::Index1d);
    }
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    match trusted_device_items::classify(tcx, definition.did()) {
        Some(TrustedDeviceItem::ShiftedIndexSpace) if arguments.len() == 2 => {
            let base = arguments[0].as_type()?;
            if base != trusted_index1d_type_v1(tcx)? {
                return None;
            }
            let offset = arguments[1].as_const()?.try_to_target_usize(tcx)?;
            Some(SemanticDisjointIndexSpaceV1::ShiftedIndex1d { offset })
        }
        Some(TrustedDeviceItem::GridExclusiveIndexSpace) if arguments.is_empty() => {
            Some(SemanticDisjointIndexSpaceV1::GridExclusive)
        }
        Some(TrustedDeviceItem::BlockedIndexSpace) if arguments.len() == 3 => {
            if arguments[0].as_type()? != trusted_index1d_type_v1(tcx)? {
                return None;
            }
            let lanes_per_block = arguments[1].as_const()?.try_to_target_usize(tcx)?;
            let elements_per_lane = arguments[2].as_const()?.try_to_target_usize(tcx)?;
            if lanes_per_block == 0
                || elements_per_lane == 0
                || lanes_per_block.checked_mul(elements_per_lane).is_none()
            {
                return None;
            }
            Some(SemanticDisjointIndexSpaceV1::BlockedIndex1d {
                lanes_per_block,
                elements_per_lane,
            })
        }
        Some(TrustedDeviceItem::Tiled2DIndexSpace) if arguments.len() == 5 => {
            if arguments[0].as_type()? != trusted_index1d_type_v1(tcx)? {
                return None;
            }
            let lanes_per_tile = arguments[1].as_const()?.try_to_target_usize(tcx)?;
            let tile_rows = arguments[2].as_const()?.try_to_target_usize(tcx)?;
            let tile_columns = arguments[3].as_const()?.try_to_target_usize(tcx)?;
            let elements_per_lane = arguments[4].as_const()?.try_to_target_usize(tcx)?;
            rust_tiled_2d_geometry_valid_v1(
                lanes_per_tile,
                tile_rows,
                tile_columns,
                elements_per_lane,
            )
            .then_some(SemanticDisjointIndexSpaceV1::Tiled2dIndex1d {
                lanes_per_tile,
                tile_rows,
                tile_columns,
                elements_per_lane,
            })
        }
        Some(TrustedDeviceItem::RowStriped2DIndexSpace) if arguments.len() == 3 => {
            if arguments[0].as_type()? != trusted_index1d_type_v1(tcx)? {
                return None;
            }
            let lanes_per_row = arguments[1].as_const()?.try_to_target_usize(tcx)?;
            let elements_per_lane = arguments[2].as_const()?.try_to_target_usize(tcx)?;
            rust_row_striped_2d_geometry_valid_v1(lanes_per_row, elements_per_lane).then_some(
                SemanticDisjointIndexSpaceV1::RowStriped2dIndex1d {
                    lanes_per_row,
                    elements_per_lane,
                },
            )
        }
        _ => None,
    }
}

fn rust_tiled_2d_geometry_valid_v1(
    lanes_per_tile: u64,
    tile_rows: u64,
    tile_columns: u64,
    elements_per_lane: u64,
) -> bool {
    lanes_per_tile != 0
        && tile_rows != 0
        && tile_columns != 0
        && elements_per_lane != 0
        && lanes_per_tile.is_multiple_of(tile_columns)
        && lanes_per_tile
            .checked_mul(elements_per_lane)
            .is_some_and(|extent| tile_rows.checked_mul(tile_columns) == Some(extent))
        && (lanes_per_tile / tile_columns).checked_mul(elements_per_lane) == Some(tile_rows)
}

fn rust_row_striped_2d_geometry_valid_v1(lanes_per_row: u64, elements_per_lane: u64) -> bool {
    lanes_per_row != 0
        && elements_per_lane != 0
        && (elements_per_lane - 1)
            .checked_mul(lanes_per_row)
            .and_then(|base| base.checked_add(lanes_per_row - 1))
            .is_some()
}

fn rust_disjoint_block_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(SemanticDisjointIndexSpaceV1, u64, u64)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did())
        != Some(TrustedDeviceItem::DisjointBlock)
        || arguments.len() != 3
    {
        return None;
    }
    let base = arguments[0].as_type()?;
    if base != trusted_index1d_type_v1(tcx)? {
        return None;
    }
    let lanes_per_block = arguments[1].as_const()?.try_to_target_usize(tcx)?;
    let elements_per_lane = arguments[2].as_const()?.try_to_target_usize(tcx)?;
    if lanes_per_block == 0
        || elements_per_lane == 0
        || lanes_per_block.checked_mul(elements_per_lane).is_none()
    {
        return None;
    }
    Some((
        SemanticDisjointIndexSpaceV1::BlockedIndex1d {
            lanes_per_block,
            elements_per_lane,
        },
        lanes_per_block,
        elements_per_lane,
    ))
}

fn require_rust_disjoint_block_component_contract_v1(
    receiver_is_shared: bool,
    mapping: Option<(SemanticDisjointIndexSpaceV1, u64, u64)>,
    component_is_usize: bool,
    output_is_option_usize: bool,
) -> Result<(SemanticDisjointIndexSpaceV1, u64, u64), ProductionSemanticImportErrorV1> {
    if !receiver_is_shared {
        return Err(body_owner_table_mismatch_v1(
            "terminal disjoint-block receiver",
        ));
    }
    let mapping =
        mapping.ok_or_else(|| body_owner_table_mismatch_v1("terminal disjoint-block mapping"))?;
    if !component_is_usize || !output_is_option_usize {
        return Err(body_owner_table_mismatch_v1(
            "terminal disjoint-block component or output",
        ));
    }
    Ok(mapping)
}

pub(crate) fn rust_disjoint_tile_2d_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(SemanticDisjointIndexSpaceV1, u64, u64, u64, u64)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did())
        != Some(TrustedDeviceItem::DisjointTile2D)
        || arguments.len() != 5
        || arguments[0].as_type()? != trusted_index1d_type_v1(tcx)?
    {
        return None;
    }
    let lanes_per_tile = arguments[1].as_const()?.try_to_target_usize(tcx)?;
    let tile_rows = arguments[2].as_const()?.try_to_target_usize(tcx)?;
    let tile_columns = arguments[3].as_const()?.try_to_target_usize(tcx)?;
    let elements_per_lane = arguments[4].as_const()?.try_to_target_usize(tcx)?;
    if !rust_tiled_2d_geometry_valid_v1(lanes_per_tile, tile_rows, tile_columns, elements_per_lane)
    {
        return None;
    }
    Some((
        SemanticDisjointIndexSpaceV1::Tiled2dIndex1d {
            lanes_per_tile,
            tile_rows,
            tile_columns,
            elements_per_lane,
        },
        lanes_per_tile,
        tile_rows,
        tile_columns,
        elements_per_lane,
    ))
}

fn rust_disjoint_row_stripe_2d_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(SemanticDisjointIndexSpaceV1, u64, u64)> {
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return None;
    };
    if trusted_device_items::classify(tcx, definition.did())
        != Some(TrustedDeviceItem::DisjointRowStripe2D)
        || arguments.len() != 3
        || arguments[0].as_type()? != trusted_index1d_type_v1(tcx)?
    {
        return None;
    }
    let lanes_per_row = arguments[1].as_const()?.try_to_target_usize(tcx)?;
    let elements_per_lane = arguments[2].as_const()?.try_to_target_usize(tcx)?;
    if !rust_row_striped_2d_geometry_valid_v1(lanes_per_row, elements_per_lane) {
        return None;
    }
    Some((
        SemanticDisjointIndexSpaceV1::RowStriped2dIndex1d {
            lanes_per_row,
            elements_per_lane,
        },
        lanes_per_row,
        elements_per_lane,
    ))
}

fn trusted_index1d_type_v1<'tcx>(tcx: TyCtxt<'tcx>) -> Option<Ty<'tcx>> {
    let function = trusted_device_items::definition(tcx, TrustedDeviceItem::ThreadIndex1d)?;
    let signature =
        tcx.instantiate_bound_regions_with_erased(tcx.fn_sig(function).instantiate_identity());
    let TyKind::Adt(definition, arguments) = *signature.output().kind() else {
        return None;
    };
    (trusted_device_items::classify(tcx, definition.did()) == Some(TrustedDeviceItem::ThreadIndex)
        && arguments.len() == 1)
        .then(|| arguments[0].as_type())
        .flatten()
}

fn option_payload_v1(
    types: &[SemanticTypeDeclV1],
    option: SemanticTypeIdV1,
) -> Result<SemanticTypeIdV1, ProductionSemanticImportErrorV1> {
    let declaration = types
        .get(option.index() as usize)
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal option type"))?;
    let SemanticTypeShapeV1::Enum { variants, .. } = declaration.shape() else {
        return Err(body_owner_table_mismatch_v1("terminal option type"));
    };
    variants
        .get(1)
        .and_then(|variant| variant.fields().fields().first())
        .copied()
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal option payload"))
}

fn aggregate_field_v1(
    types: &[SemanticTypeDeclV1],
    aggregate: SemanticTypeIdV1,
    field: usize,
) -> Result<SemanticTypeIdV1, ProductionSemanticImportErrorV1> {
    let declaration = types
        .get(aggregate.index() as usize)
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal aggregate type"))?;
    let SemanticTypeShapeV1::Aggregate(fields) = declaration.shape() else {
        return Err(body_owner_table_mismatch_v1("terminal aggregate type"));
    };
    fields
        .fields()
        .get(field)
        .copied()
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal aggregate field"))
}

fn tuple_field_v1(
    types: &[SemanticTypeDeclV1],
    tuple: SemanticTypeIdV1,
    field: usize,
) -> Result<SemanticTypeIdV1, ProductionSemanticImportErrorV1> {
    let declaration = types
        .get(tuple.index() as usize)
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal tuple type"))?;
    let SemanticTypeShapeV1::Tuple(fields) = declaration.shape() else {
        return Err(body_owner_table_mismatch_v1("terminal tuple type"));
    };
    fields
        .fields()
        .get(field)
        .copied()
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal tuple field"))
}

fn dynamic_lds_element_storage_v1(
    types: &[SemanticTypeDeclV1],
    dynamic_lds: SemanticTypeIdV1,
) -> Result<SemanticTypeIdV1, ProductionSemanticImportErrorV1> {
    let mut current = aggregate_field_v1(types, dynamic_lds, 0)?;
    for _ in 0..4 {
        let declaration = types
            .get(current.index() as usize)
            .ok_or_else(|| body_owner_table_mismatch_v1("exact LDS pointer wrapper"))?;
        match declaration.shape() {
            SemanticTypeShapeV1::Aggregate(wrapper) | SemanticTypeShapeV1::Union(wrapper)
                if wrapper.fields().len() == 1 =>
            {
                current = wrapper.fields()[0];
            }
            SemanticTypeShapeV1::Pointer(pointer) => return Ok(pointer.pointee()),
            _ => return Err(body_owner_table_mismatch_v1("exact LDS pointer wrapper")),
        }
    }
    Err(body_owner_table_mismatch_v1("exact LDS pointer depth"))
}

fn semantic_result_payloads_v1(
    types: &[SemanticTypeDeclV1],
    result: SemanticTypeIdV1,
) -> Result<(SemanticTypeIdV1, SemanticTypeIdV1), ProductionSemanticImportErrorV1> {
    let Some(declaration) = types.get(result.index() as usize) else {
        return Err(body_owner_table_mismatch_v1("semantic Result shape"));
    };
    let SemanticTypeShapeV1::Enum { variants, .. } = declaration.shape() else {
        return Err(body_owner_table_mismatch_v1("semantic Result shape"));
    };
    if variants.len() != 2
        || variants[0].discriminant() != 0
        || variants[0].fields().fields().len() != 1
        || variants[1].discriminant() != 1
        || variants[1].fields().fields().len() != 1
    {
        return Err(body_owner_table_mismatch_v1("semantic Result variants"));
    }
    Ok((
        variants[0].fields().fields()[0],
        variants[1].fields().fields()[0],
    ))
}

fn pointer_pointee_v1(
    types: &[SemanticTypeDeclV1],
    pointer: SemanticTypeIdV1,
) -> Result<SemanticTypeIdV1, ProductionSemanticImportErrorV1> {
    let declaration = types
        .get(pointer.index() as usize)
        .ok_or_else(|| body_owner_table_mismatch_v1("terminal pointer type"))?;
    let SemanticTypeShapeV1::Pointer(pointer) = declaration.shape() else {
        return Err(body_owner_table_mismatch_v1("terminal pointer type"));
    };
    Ok(pointer.pointee())
}

#[cfg(test)]
// Frozen calculator for the independently published BF16 and pipeline V1 histories.
const fn terminal_operation_tag_v1(
    expansion: crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1,
) -> u8 {
    terminal_operation_tag_for_schema_v1(expansion, TerminalIdentitySchemaV1::IndependentV1)
}

const fn compiler_intrinsic_identity_schema_v1(
    expansion: ProductionTerminalExpansionV1,
) -> (&'static [u8], TerminalIdentitySchemaV1) {
    match expansion {
        ProductionTerminalExpansionV1::Gfx942PhysicalEntryBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalEntryStep => (
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V7,
            TerminalIdentitySchemaV1::CombinedV7,
        ),
        ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalLdsExchangeStep => (
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V9,
            TerminalIdentitySchemaV1::CombinedV9,
        ),
        ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyBegin
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyLabel
        | ProductionTerminalExpansionV1::Gfx942PhysicalGlobalCopyStep => (
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V8,
            TerminalIdentitySchemaV1::CombinedV8,
        ),
        ProductionTerminalExpansionV1::Gfx942CompleteBodyE32 => (
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V6,
            TerminalIdentitySchemaV1::CombinedV6,
        ),
        ProductionTerminalExpansionV1::Gfx942OrderedProgramE32 => (
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V5,
            TerminalIdentitySchemaV1::CombinedV5,
        ),
        _ => (
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V4,
            TerminalIdentitySchemaV1::CombinedV4,
        ),
    }
}

const fn terminal_operation_tag_for_schema_v1(
    expansion: crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1,
    schema: TerminalIdentitySchemaV1,
) -> u8 {
    use crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1;
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
        ProductionTerminalExpansionV1::ThreadIndex(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::X,
        ) => 13,
        ProductionTerminalExpansionV1::ThreadIndex(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Y,
        ) => 14,
        ProductionTerminalExpansionV1::ThreadIndex(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Z,
        ) => 15,
        ProductionTerminalExpansionV1::WorkgroupIndex(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::X,
        ) => 16,
        ProductionTerminalExpansionV1::WorkgroupIndex(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Y,
        ) => 17,
        ProductionTerminalExpansionV1::WorkgroupIndex(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Z,
        ) => 18,
        ProductionTerminalExpansionV1::WorkgroupDimension(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::X,
        ) => 19,
        ProductionTerminalExpansionV1::WorkgroupDimension(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Y,
        ) => 20,
        ProductionTerminalExpansionV1::WorkgroupDimension(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Z,
        ) => 21,
        ProductionTerminalExpansionV1::GridDimension(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::X,
        ) => 22,
        ProductionTerminalExpansionV1::GridDimension(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Y,
        ) => 23,
        ProductionTerminalExpansionV1::GridDimension(
            fe2o3_mir_model::semantic_mir_v1::SemanticAxisV1::Z,
        ) => 24,
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
            TerminalIdentitySchemaV1::IndependentV1 | TerminalIdentitySchemaV1::CombinedV2 => 106,
            TerminalIdentitySchemaV1::CombinedV3 => 113,
            TerminalIdentitySchemaV1::CombinedV4 | TerminalIdentitySchemaV1::CombinedV5 => 116,
            TerminalIdentitySchemaV1::CombinedV6
            | TerminalIdentitySchemaV1::CombinedV7
            | TerminalIdentitySchemaV1::CombinedV8
            | TerminalIdentitySchemaV1::CombinedV9 => u8::MAX,
        },
        ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum => match schema {
            #[cfg(test)]
            TerminalIdentitySchemaV1::IndependentV1 | TerminalIdentitySchemaV1::CombinedV2 => 107,
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

fn build_identity_inventory_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    target: &crate::production_target_v1::AuthenticatedProductionTargetV1,
    collection: &CollectionResult<'tcx>,
    roots: &[AuthenticatedProductionRootV1<'tcx>],
    closure_flow: super::closure_flow_v1::AuthenticatedClosureFlowV1<'tcx>,
    original_root_capture: Option<
        super::reference_custody_v1::PendingOriginalRootAssociationsV1<'tcx>,
    >,
    loan: Option<&crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>>,
) -> Result<
    (
        ProductionSemanticIdentityInventoryV1<'tcx>,
        SourceClosureWorkV1,
    ),
    ProductionSemanticImportErrorV1,
> {
    require_count_within_limit_v1(
        SemanticMirResourceV1::Functions,
        collection.functions.len(),
        HARD_MAX_FUNCTIONS_V1,
    )?;
    require_count_within_limit_v1(SemanticMirResourceV1::Roots, roots.len(), HARD_MAX_ROOTS_V1)?;
    let mut closure_work = closure_flow
        .revalidate_for_import_v1(tcx, collection)
        .map_err(ProductionSemanticImportErrorV1::ClosureAdmission)?;

    let target = canonical_target_layout_v1(target.rustc_layout());
    let mut functions = Vec::with_capacity(collection.functions.len());
    for function in &collection.functions {
        functions.push(RetainedSemanticFunctionProducerV1 {
            identities: canonical_function_identities_v1(tcx, function.instance),
            instance: function.instance,
            role: function.role,
            export_name: matches!(
                function.role,
                CollectedFunctionRole::KernelEntry | CollectedFunctionRole::DeviceFfiExport
            )
            .then(|| function.export_name.clone()),
            kernel_binding: function.kernel_binding,
            generated_host_contract_identity: function.generated_host_contract_identity,
            frontend_contract: function.frontend_contract.clone(),
        });
    }
    functions.sort_unstable_by_key(|entry| entry.identities.function());
    if functions
        .windows(2)
        .any(|pair| pair[0].identities.function() == pair[1].identities.function())
    {
        return Err(ProductionSemanticImportErrorV1::FunctionIdentityCollision);
    }

    let mut function_ids = BTreeMap::<SemanticFunctionIdentityV1, SemanticFunctionIdV1>::new();
    for (index, function) in functions.iter().enumerate() {
        let index =
            u32::try_from(index).map_err(|_| ProductionSemanticImportErrorV1::LimitExceeded {
                resource: SemanticMirResourceV1::Functions,
                actual: u64::MAX,
                maximum: HARD_MAX_FUNCTIONS_V1,
            })?;
        function_ids.insert(
            function.identities.function(),
            SemanticFunctionIdV1::from_index(index),
        );
    }

    let mut canonical_roots = Vec::with_capacity(roots.len());
    for root in roots {
        let identity = canonical_function_identities_v1(tcx, root.instance).function();
        let Some(function_id) = function_ids.get(&identity).copied() else {
            return Err(ProductionSemanticImportErrorV1::RootIdentityMismatch);
        };
        canonical_roots.push(function_id);
    }
    canonical_roots.sort_unstable();
    if canonical_roots.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ProductionSemanticImportErrorV1::RootIdentityMismatch);
    }

    let original_root_associations = original_root_capture
        .map(|capture| {
            let loan = loan.ok_or_else(|| {
                ProductionSemanticImportErrorV1::ReferenceCustody(
                    crate::reference_effect_v1::ReferenceBindingErrorV1::new(
                        "root mapping original loan missing",
                    ),
                )
            })?;
            capture
                .seal(tcx, loan, &mut closure_work, &functions, &canonical_roots)
                .map_err(ProductionSemanticImportErrorV1::ReferenceCustody)
        })
        .transpose()?;
    let (sha256, canonical_transcript) = match &original_root_associations {
        Some(associations) => enrolled_identity_inventory_transcript_v1(
            target,
            &functions,
            &canonical_roots,
            associations,
            loan.ok_or(ProductionSemanticImportErrorV1::RootCustodyMismatch)?,
            &mut closure_work,
        )?,
        None => identity_inventory_identity_and_transcript_v1(target, &functions, &canonical_roots),
    };
    Ok((
        ProductionSemanticIdentityInventoryV1 {
            functions: functions.into_boxed_slice(),
            roots: canonical_roots.into_boxed_slice(),
            sha256,
            canonical_transcript,
            original_root_associations,
        },
        closure_work,
    ))
}

#[cfg(test)]
#[path = "original_root_inventory_flow_v362_tests.rs"]
pub(super) mod original_root_inventory_flow_v362_tests;

fn identity_inventory_identity_and_transcript_v1(
    target: SemanticTargetDataLayoutV1,
    functions: &[RetainedSemanticFunctionProducerV1<'_>],
    roots: &[SemanticFunctionIdV1],
) -> ([u8; 32], Box<[u8]>) {
    let mut digest =
        SemanticIdentityDigestV1::new_with_canonical_transcript(IDENTITY_INVENTORY_DOMAIN_V2);
    let result: Result<(), std::convert::Infallible> =
        visit_identity_inventory_fields_v1(target, functions, roots, |field| {
            digest.field(field);
            Ok(())
        });
    match result {
        Ok(()) => digest.finish_with_canonical_transcript(),
        Err(never) => match never {},
    }
}

fn visit_identity_inventory_fields_v1<E>(
    target: SemanticTargetDataLayoutV1,
    functions: &[RetainedSemanticFunctionProducerV1<'_>],
    roots: &[SemanticFunctionIdV1],
    mut field: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    field(target.identity().as_bytes())?;
    for function in functions {
        field(function.identities.function().as_bytes())?;
        field(function.identities.item_definition().as_bytes())?;
        field(function.identities.monomorphization().as_bytes())?;
        field(function.identities.generic_type_arguments().as_bytes())?;
        field(function.identities.const_generic_arguments().as_bytes())?;
        field(&[function_role_tag_v1(function.role)])?;
        match &function.export_name {
            Some(symbol) => {
                field(&[1])?;
                field(symbol.as_bytes())?;
            }
            None => field(&[0])?,
        }
        match function.kernel_binding {
            Some(binding) => {
                field(&[1])?;
                field(&binding.as_bytes())?;
            }
            None => field(&[0])?,
        }
        match function.generated_host_contract_identity {
            Some(identity) => {
                field(&[1])?;
                field(&identity.as_bytes())?;
            }
            None => field(&[0])?,
        }
        match &function.frontend_contract {
            Some(contract) => {
                field(&[1])?;
                field(contract.canonical_bytes())?;
                if let Some(bytes) = contract.resource_canonical_bytes() {
                    field(&[1])?;
                    field(bytes)?;
                }
                let reachable = contract.reachable_assembly();
                field(&reachable.blocks().to_le_bytes())?;
                field(&reachable.operand_bits().to_le_bytes())?;
                field(&reachable.option_bits().to_le_bytes())?;
            }
            None => field(&[0])?,
        }
    }
    for root in roots {
        field(&root.index().to_le_bytes())?;
    }
    Ok(())
}

fn enrolled_identity_inventory_transcript_v1(
    target: SemanticTargetDataLayoutV1,
    functions: &[RetainedSemanticFunctionProducerV1<'_>],
    roots: &[SemanticFunctionIdV1],
    associations: &super::reference_custody_v1::RetainedOriginalRootAssociationsV1,
    loan: &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
    work: &mut SourceClosureWorkV1,
) -> Result<([u8; 32], Box<[u8]>), ProductionSemanticImportErrorV1> {
    use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
    use fe2o3_compiler_lineage::{
        RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1, RustcEnrollmentInventoryInputV1,
        RustcEnrollmentInventoryLayoutV1, encode_rustc_enrollment_inventory_v1,
    };
    use sha2::{Digest, Sha256};
    let rejected = ProductionSemanticImportErrorV1::ReferenceCustody;
    let mut legacy_len = 8 + IDENTITY_INVENTORY_DOMAIN_V2.len();
    visit_identity_inventory_fields_v1(target, functions, roots, |field| {
        work.charge(
            field
                .len()
                .checked_add(8)
                .ok_or_else(|| Error::new("inventory quote overflow"))?,
        )
        .map_err(|failure| Error::new(failure.to_string()))?;
        legacy_len = legacy_len
            .checked_add(8)
            .and_then(|n| n.checked_add(field.len()))
            .ok_or_else(|| Error::new("inventory extent overflow"))?;
        if legacy_len > fe2o3_compiler_lineage::MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3 {
            return Err(Error::new(
                "legacy inventory exceeds complete wrapper ceiling",
            ));
        }
        Ok(())
    })
    .map_err(rejected)?;
    let layout =
        RustcEnrollmentInventoryLayoutV1::new::<Error>(legacy_len, associations.roots().len())
            .map_err(|failure| rejected(Error::new(format!("{failure:?}"))))?;
    let output_quote = layout
        .owned_storage_bytes::<Error>()
        .map_err(|failure| rejected(Error::new(format!("{failure:?}"))))?;
    let total_quote = legacy_len
        .checked_add(std::mem::size_of::<Vec<u8>>())
        .and_then(|n| n.checked_add(output_quote))
        .and_then(|n| n.checked_add(RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1))
        .ok_or_else(|| rejected(Error::new("inventory overlapping storage overflow")))?;
    loan.reserve_inventory_storage(total_quote)
        .map_err(rejected)?;
    let mut legacy = Vec::new();
    legacy
        .try_reserve_exact(legacy_len)
        .map_err(|_| rejected(Error::new("legacy inventory allocation failed")))?;
    loan.reserve_inventory_storage(
        legacy
            .capacity()
            .checked_sub(legacy_len)
            .ok_or_else(|| rejected(Error::new("legacy inventory capacity accounting")))?,
    )
    .map_err(rejected)?;
    let mut append = |field: &[u8]| -> Result<(), Error> {
        let next = legacy
            .len()
            .checked_add(8)
            .and_then(|n| n.checked_add(field.len()))
            .ok_or_else(|| Error::new("legacy inventory append overflow"))?;
        if next > legacy_len {
            return Err(Error::new("legacy inventory quote changed"));
        }
        work.charge(8 + field.len())
            .map_err(|failure| Error::new(failure.to_string()))?;
        legacy.extend_from_slice(&(field.len() as u64).to_le_bytes());
        legacy.extend_from_slice(field);
        Ok(())
    };
    append(IDENTITY_INVENTORY_DOMAIN_V2).map_err(rejected)?;
    visit_identity_inventory_fields_v1(target, functions, roots, &mut append).map_err(rejected)?;
    drop(append);
    if legacy.len() != legacy_len {
        return Err(rejected(Error::new("legacy inventory length changed")));
    }
    let input = RustcEnrollmentInventoryInputV1 {
        legacy_inventory: &legacy,
        header: associations.header(),
        roots: associations.roots(),
    };
    let bytes =
        encode_rustc_enrollment_inventory_v1(input, loan.inventory_storage_limit(), |amount| {
            work.charge(amount)
                .map_err(|failure| Error::new(failure.to_string()))
        })
        .map_err(|failure| rejected(Error::new(format!("{failure:?}"))))?;
    if bytes.len() != layout.encoded_len() || bytes.capacity() != layout.encoded_len() {
        return Err(rejected(Error::new(
            "enrollment wrapper capacity accounting",
        )));
    }
    work.charge(bytes.len())
        .map_err(|failure| rejected(Error::new(failure.to_string())))?;
    // The framing trailer has its own domain: preflight binds the COMPLETE bytes.
    let sha256 = Sha256::digest(&bytes).into();
    let context = loan
        .inventory_context(work)
        .map_err(rejected)?
        .ok_or_else(|| {
            rejected(Error::new(
                "enrollment wrapper original request disappeared",
            ))
        })?;
    let header = associations.header();
    if context.descriptor_bindings != header.enrollment_binding_count
        || context.rustc_invocation_sha256 != header.invocation_identity
        || context.native_policy_sha256 != header.native_policy_identity
        || context.policy_generation != header.native_policy_generation
    {
        return Err(rejected(Error::new(
            "enrollment wrapper original header changed",
        )));
    }
    Ok((sha256, bytes.into_boxed_slice()))
}

#[cfg(test)]
fn identity_inventory_generated_host_contract_v1(
    digest: &mut SemanticIdentityDigestV1,
    identity: Option<reserved_fe2o3_symbols::GeneratedHostContractIdV3>,
) {
    match identity {
        Some(identity) => {
            digest.field(&[1]);
            digest.field(&identity.as_bytes());
        }
        None => digest.field(&[0]),
    }
}

const fn function_role_tag_v1(role: CollectedFunctionRole) -> u8 {
    match role {
        CollectedFunctionRole::KernelEntry => 0,
        CollectedFunctionRole::InternalHelper => 1,
        CollectedFunctionRole::DeviceFfiExport => 2,
    }
}

fn require_count_within_limit_v1(
    resource: SemanticMirResourceV1,
    actual: usize,
    maximum: u64,
) -> Result<(), ProductionSemanticImportErrorV1> {
    let actual = u64::try_from(actual).unwrap_or(u64::MAX);
    if actual > maximum {
        Err(ProductionSemanticImportErrorV1::LimitExceeded {
            resource,
            actual,
            maximum,
        })
    } else {
        Ok(())
    }
}

fn exact_ordered_axes_match<T: PartialEq>(
    expected: impl IntoIterator<Item = T>,
    observed: impl IntoIterator<Item = T>,
) -> bool {
    expected.into_iter().eq(observed)
}

#[cfg(test)]
#[path = "production_importer_typed_default_v1_tests.rs"]
mod typed_default_tests;

#[cfg(test)]
mod tests {
    use super::*;
    include!("production_importer_ordered_program_v32_tests.rs");
    include!("production_importer_complete_body_v36_tests.rs");
    include!("production_importer_physical_entry_v37_tests.rs");
    include!("production_importer_physical_global_copy_v38_tests.rs");
    include!("production_importer_physical_lds_exchange_v39_tests.rs");

    #[test]
    fn ordered_region_v31_terminal_133_never_enters_frozen_source_histories() {
        let region = crate::production_semantic_terminal_v1::ProductionTerminalExpansionV1::Gfx942OrderedXorAddE32;
        assert_eq!(
            terminal_operation_tag_for_schema_v1(region, TerminalIdentitySchemaV1::CombinedV4),
            133
        );
        for schema in [
            TerminalIdentitySchemaV1::IndependentV1,
            TerminalIdentitySchemaV1::CombinedV2,
            TerminalIdentitySchemaV1::CombinedV3,
        ] {
            assert_eq!(
                terminal_operation_tag_for_schema_v1(region, schema),
                u8::MAX
            );
        }
    }

    #[test]
    fn tiled_geometry_requires_finite_equal_products() {
        for ([lanes, rows, columns, elements], expected) in [
            ([4, 2, 4, 2], true),
            ([8, 2, 4, 1], true),
            ([64, 16, 16, 4], true),
            ([u64::MAX, 1, u64::MAX, 1], true),
            ([1, u64::MAX, 1, u64::MAX], true),
            ([u64::MAX, u64::MAX, 1, 1], true),
            ([(1_u64 << 63) - 1, 2, (1_u64 << 63) - 1, 2], true),
            ([1_u64 << 63, 2, 1_u64 << 63, 2], false),
            ([1_u64 << 63, 4, 1_u64 << 62, 2], false),
            ([u64::MAX, 2, u64::MAX, 2], false),
            ([2, u64::MAX, 2, u64::MAX], false),
            ([u64::MAX, 1, u64::MAX, 2], false),
            ([1, u64::MAX, 2, 1], false),
            ([4, 3, 4, 2], false),
            ([3, 2, 2, 1], false),
            ([0, 2, 4, 2], false),
            ([4, 0, 4, 2], false),
            ([4, 2, 0, 2], false),
            ([4, 2, 4, 0], false),
        ] {
            assert_eq!(
                rust_tiled_2d_geometry_valid_v1(lanes, rows, columns, elements),
                expected,
                "geometry {:?}",
                [lanes, rows, columns, elements],
            );
        }
    }

    #[test]
    fn root_custody_comparison_rejects_every_sequence_substitution() {
        assert!(exact_ordered_axes_match([1, 2], [1, 2]));
        for substituted in [vec![1], vec![1, 2, 2], vec![2, 1], vec![1, 3]] {
            assert!(!exact_ordered_axes_match(vec![1, 2], substituted));
        }
    }

    #[test]
    fn count_preflight_rejects_before_semantic_record_allocation() {
        assert!(require_count_within_limit_v1(SemanticMirResourceV1::Functions, 4, 4).is_ok());
        assert!(matches!(
            require_count_within_limit_v1(SemanticMirResourceV1::Functions, 5, 4),
            Err(ProductionSemanticImportErrorV1::LimitExceeded {
                resource: SemanticMirResourceV1::Functions,
                actual: 5,
                maximum: 4,
            })
        ));
    }

    #[test]
    fn disjoint_block_component_terminal_rejects_unrepresentable_rust_signatures() {
        let mapping = (
            SemanticDisjointIndexSpaceV1::BlockedIndex1d {
                lanes_per_block: 16,
                elements_per_lane: 4,
            },
            16,
            4,
        );
        assert_eq!(
            require_rust_disjoint_block_component_contract_v1(true, Some(mapping), true, true,)
                .unwrap(),
            mapping,
        );

        let rejected_table = |error: ProductionSemanticImportErrorV1| match error {
            ProductionSemanticImportErrorV1::BodyConstruction(error) => match *error {
                ProductionSemanticBodyErrorV1::IdentityTableMismatch { table } => table,
                other => panic!("unexpected body error: {other:?}"),
            },
            other => panic!("unexpected importer error: {other:?}"),
        };
        assert_eq!(
            rejected_table(
                require_rust_disjoint_block_component_contract_v1(
                    false,
                    Some(mapping),
                    true,
                    true,
                )
                .unwrap_err(),
            ),
            "terminal disjoint-block receiver",
        );
        assert_eq!(
            rejected_table(
                require_rust_disjoint_block_component_contract_v1(true, None, true, true)
                    .unwrap_err(),
            ),
            "terminal disjoint-block mapping",
        );
        for (component_is_usize, output_is_option_usize) in
            [(false, true), (true, false), (false, false)]
        {
            assert_eq!(
                rejected_table(
                    require_rust_disjoint_block_component_contract_v1(
                        true,
                        Some(mapping),
                        component_is_usize,
                        output_is_option_usize,
                    )
                    .unwrap_err(),
                ),
                "terminal disjoint-block component or output",
            );
        }
    }

    #[test]
    fn terminal_operation_identity_preserves_independent_histories_and_versions_combined_use() {
        use crate::production_semantic_terminal_v1::{
            ProductionBf16ConversionV1, ProductionTerminalExpansionV1,
        };
        for (expansion, tag) in [
            (ProductionTerminalExpansionV1::Gfx950LdsTransposePublish, 83),
            (ProductionTerminalExpansionV1::WorkgroupLdsScopeCurrent, 118),
            (
                ProductionTerminalExpansionV1::DisjointBlockComponentIndex,
                119,
            ),
            (ProductionTerminalExpansionV1::Bf16MatrixBColumnMajor, 120),
            (
                ProductionTerminalExpansionV1::Bf16MatrixBColumnMajorLoadZeroFilledV1,
                121,
            ),
            (ProductionTerminalExpansionV1::ContextIssue, 122),
            (ProductionTerminalExpansionV1::WorkgroupDerive, 123),
            (ProductionTerminalExpansionV1::MaskedTileLoadU32, 124),
            (
                ProductionTerminalExpansionV1::MaskedTileIntoFragmentU32,
                125,
            ),
            (ProductionTerminalExpansionV1::LaneFragmentIntoPartsU32, 126),
        ] {
            assert_eq!(
                terminal_operation_tag_for_schema_v1(
                    expansion,
                    TerminalIdentitySchemaV1::CombinedV4
                ),
                tag
            );
        }

        let pipeline = [
            ProductionTerminalExpansionV1::WorkgroupPipelineCurrent,
            ProductionTerminalExpansionV1::WorkgroupPipelineStage,
            ProductionTerminalExpansionV1::WorkgroupPipelineWrite,
            ProductionTerminalExpansionV1::WorkgroupPipelineCommit,
            ProductionTerminalExpansionV1::WorkgroupPipelineWait,
            ProductionTerminalExpansionV1::WorkgroupPipelineConsume,
            ProductionTerminalExpansionV1::WorkgroupPipelineRead,
            ProductionTerminalExpansionV1::WorkgroupPipelineDiscard,
            ProductionTerminalExpansionV1::WorkgroupPipelineRelease,
        ];
        assert_eq!(
            pipeline.map(terminal_operation_tag_v1),
            [91, 92, 93, 94, 95, 96, 97, 98, 99]
        );
        let bf16 = [
            ProductionBf16ConversionV1::FromBits,
            ProductionBf16ConversionV1::ToBits,
            ProductionBf16ConversionV1::FromF32RoundTiesEven,
            ProductionBf16ConversionV1::ToF32,
        ];
        assert_eq!(
            bf16.map(|conversion| terminal_operation_tag_v1(
                ProductionTerminalExpansionV1::Bf16Conversion(conversion),
            )),
            [91, 92, 93, 94]
        );

        let combined_schema = TerminalIdentitySchemaV1::CombinedV3;
        assert_eq!(combined_schema, TerminalIdentitySchemaV1::CombinedV3);
        assert_ne!(
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V1,
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V2
        );
        assert_ne!(
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V2,
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V3
        );
        assert_ne!(
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V3,
            PRODUCTION_COMPILER_INTRINSIC_DOMAIN_V4
        );
        assert_eq!(
            pipeline.map(|expansion| {
                terminal_operation_tag_for_schema_v1(expansion, combined_schema)
            }),
            [91, 92, 93, 94, 95, 96, 97, 98, 99]
        );
        assert_eq!(
            [
                ProductionTerminalExpansionV1::RustcFabsF32,
                ProductionTerminalExpansionV1::MathF32(fe2o3_kernel_ir::F32MathFunction::Abs),
                ProductionTerminalExpansionV1::MemoryVolatileLoad,
            ]
            .map(|expansion| terminal_operation_tag_for_schema_v1(
                expansion,
                TerminalIdentitySchemaV1::CombinedV4,
            )),
            [113, 114, 115],
        );
        assert_eq!(
            bf16.map(|conversion| terminal_operation_tag_for_schema_v1(
                ProductionTerminalExpansionV1::Bf16Conversion(conversion),
                combined_schema,
            )),
            [100, 101, 102, 103]
        );
        assert_eq!(
            [
                ProductionTerminalExpansionV1::WorkgroupCollectiveContextCurrent,
                ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum,
                ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum,
                ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum,
            ]
            .map(|expansion| terminal_operation_tag_for_schema_v1(expansion, combined_schema)),
            [111, 112, 113, 114]
        );
        assert_eq!(
            [
                ProductionTerminalExpansionV1::RustcFabsF32,
                ProductionTerminalExpansionV1::MathF32(fe2o3_kernel_ir::F32MathFunction::Abs,),
                ProductionTerminalExpansionV1::MemoryVolatileLoad,
                ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum,
                ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum,
                ProductionTerminalExpansionV1::WorkgroupLdsScopeCurrent,
                ProductionTerminalExpansionV1::DisjointBlockComponentIndex,
            ]
            .map(|expansion| terminal_operation_tag_for_schema_v1(
                expansion,
                TerminalIdentitySchemaV1::CombinedV4,
            )),
            [113, 114, 115, 116, 117, 118, 119],
        );
        assert_eq!(
            [
                ProductionTerminalExpansionV1::Bf16Conversion(
                    ProductionBf16ConversionV1::FromBits,
                ),
                ProductionTerminalExpansionV1::Bf16Conversion(
                    ProductionBf16ConversionV1::ToBits,
                ),
                ProductionTerminalExpansionV1::Bf16Conversion(
                    ProductionBf16ConversionV1::FromF32RoundTiesEven,
                ),
                ProductionTerminalExpansionV1::Bf16Conversion(
                    ProductionBf16ConversionV1::ToF32,
                ),
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceLen,
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWrite,
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteDisjoint,
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteExclusive,
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteBlock,
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteTiled2d,
                ProductionTerminalExpansionV1::WriteOnlyDisjointSliceWriteRowStriped2d,
                ProductionTerminalExpansionV1::WorkgroupCollectiveContextCurrent,
                ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum,
                ProductionTerminalExpansionV1::NeutralWorkgroupInclusiveScanSum,
                ProductionTerminalExpansionV1::NeutralWorkgroupExclusiveScanSum,
            ]
            .map(|expansion| terminal_operation_tag_for_schema_v1(expansion, combined_schema)),
            [
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
            ]
        );
        assert_eq!(
            [
                ProductionTerminalExpansionV1::WorkgroupCollectiveContextCurrent,
                ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum,
            ]
            .map(|expansion| terminal_operation_tag_for_schema_v1(
                expansion,
                TerminalIdentitySchemaV1::CombinedV2,
            )),
            [104, 105]
        );
        assert_ne!(
            terminal_operation_tag_for_schema_v1(
                ProductionTerminalExpansionV1::NeutralWorkgroupReduceSum,
                combined_schema,
            ),
            terminal_operation_tag_for_schema_v1(
                ProductionTerminalExpansionV1::WorkgroupReduceSum,
                combined_schema,
            )
        );
    }
}
#[path = "production_importer_transcript_storage_v1.rs"]
mod transcript_storage_v1;

//! Semantic identities recognized by device lowering.
//!
//! Recognition starts from a rustc [`DefId`]. Diagnostic-item equality is only
//! accepted after the provider definition is anchored to a complete source
//! closure that exactly matches a reviewed canonical or Cargo-vendored
//! `fe2o3-device` materialization. The source location is not trusted. Rustc's stable
//! crate ID and crate hash are retained as same-session provenance
//! observations, but portable semantic identities bind only canonical
//! source-derived fields, including the actual materialization's source closure.
//!
//! This remains a compiler build-observation boundary, not cryptographic
//! package authentication. A publisher signature or transparency-log identity
//! must be checked before the managed build when that stronger claim is needed.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rustc_abi::ExternAbi;
use rustc_hir::Safety;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::{
    BinOp, Body, Operand, Rvalue, StatementKind, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::{FloatTy, Instance, InstanceKind, Ty, TyCtxt, TyKind, TypingEnv, UintTy};
use rustc_span::{SourceFileHash, Symbol};
use sha2::{Digest as _, Sha256};

use dialect_amdgcn::{
    DeviceMathDiagnosticItem, DeviceValueDiagnosticItem, Fe2o3DeviceDiagnosticItem,
};
use fe2o3_kernel_ir::{NarrowFloatFormat, WidenedFloatBinaryOp};
use fe2o3_rustc_invocation::CARGO_METADATA_BUILD_OBSERVATION_ENV_V2;

mod primitive_from_v1;
mod wave64_scan_provider_v1;
mod wave64_shuffle_provider_v1;
pub(crate) use primitive_from_v1::primitive_from_candidate_v1;
#[cfg(test)]
pub(crate) use wave64_scan_provider_v1::check_actual_scan_instances_v1;
pub(crate) use wave64_scan_provider_v1::is_authenticated_gfx942_wave64_scan_instance_v1;
#[cfg(test)]
pub(crate) use wave64_shuffle_provider_v1::check_actual_sealed_trait_chain_paths_v1;
pub(crate) use wave64_shuffle_provider_v1::{
    Wave64ShuffleScalarV1, is_authenticated_gfx942_wave64_shuffle_instance_v1,
    scalar_for_instance as wave64_shuffle_scalar_for_instance_v1,
};

const WORKGROUP_SYNC_PROVIDER_SOURCE_IDENTITY_DOMAIN_V1: &[u8] =
    b"FE2O3/WORKGROUP-SYNC-PROVIDER-SOURCE-IDENTITY/V1\0";
const WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1: &[u8] =
    b"FE2O3/WORKGROUP-SYNC-PROVIDER-SOURCE-CLOSURE/V1\0";
const REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1: [u8; 32] = [
    0x0e, 0x43, 0xcb, 0x1e, 0x81, 0x56, 0xa7, 0xec, 0x6b, 0xd9, 0xdf, 0xda, 0x34, 0x4b, 0x18, 0x8b,
    0x9c, 0x0a, 0x31, 0x55, 0xc7, 0x39, 0x5b, 0x9e, 0x3d, 0x6f, 0x68, 0x55, 0x16, 0xc7, 0x4d, 0x8b,
];
// The pinned Cargo-produced manifest fixture is checked with the complete source tree.
const REVIEWED_SAFE_EXECUTION_CARGO_VENDOR_SOURCE_CLOSURE_V1: [u8; 32] = [
    0xa5, 0x1c, 0x98, 0xe5, 0x08, 0x89, 0x6c, 0x3a, 0x95, 0x89, 0xc0, 0x07, 0x47, 0x1f, 0xbb, 0x46,
    0x51, 0x6c, 0xaa, 0xe2, 0x03, 0x30, 0xb1, 0xe5, 0xb2, 0x42, 0x45, 0xbf, 0x2f, 0x4a, 0xea, 0x5f,
];
const REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1: [[u8; 32]; 2] = [
    REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
    REVIEWED_SAFE_EXECUTION_CARGO_VENDOR_SOURCE_CLOSURE_V1,
];

const PROVIDER_SEMANTIC_DEFINITION_TRANSCRIPT_DOMAIN_V1: &[u8] =
    b"FE2O3/PROVIDER-SEMANTIC-DEFINITION-TRANSCRIPT/V1\0";
#[cfg(test)]
const PINNED_CORE_SEMANTIC_TERMINAL_TRANSCRIPT_DOMAIN_V1: &[u8] =
    b"FE2O3/PINNED-CORE-SEMANTIC-TERMINAL-TRANSCRIPT/V1\0";
const STRUCTURAL_LOCAL_DEFINITION_COMPONENT_DOMAIN_V1: &[u8] =
    b"FE2O3/STRUCTURAL-LOCAL-DEFINITION-COMPONENT/V1\0";
#[cfg(test)]
const REVIEWED_FE2O3_DEVICE_PACKAGE_ROOT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../fe2o3-device");
#[cfg(test)]
const REVIEWED_FE2O3_DEVICE_SOURCE_ROOT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../fe2o3-device/src");

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReviewedProviderSourceClosureV1 {
    source_root: PathBuf,
    identity: [u8; 32],
}

static WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE: OnceLock<
    Result<ReviewedProviderSourceClosureV1, String>,
> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompilerProviderObservationV1 {
    pub(crate) crate_name: String,
    pub(crate) stable_crate_id: u64,
    pub(crate) crate_hash_observation: [u8; 16],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReviewedProviderSemanticDefinitionV1 {
    /// These rustc values prove same-session crate membership only. They are
    /// intentionally excluded from `durable_semantic_identity` because Cargo
    /// can change them after an unrelated transitive feature change.
    pub(crate) provider: CompilerProviderObservationV1,
    pub(crate) canonical_definition_path: String,
    pub(crate) structural_local_definition_component: [u8; 32],
    pub(crate) cargo_metadata_build_observation: [u8; 32],
    pub(crate) source_closure_identity: [u8; 32],
    pub(crate) definition_source_identity: [u8; 32],
}

impl ReviewedProviderSemanticDefinitionV1 {
    fn validate(&self) -> Result<(), String> {
        if self.provider.crate_name != "fe2o3_device"
            || self.provider.stable_crate_id == 0
            || self.provider.crate_hash_observation == [0; 16]
            || self.canonical_definition_path.is_empty()
            || self.structural_local_definition_component == [0; 32]
            || self.cargo_metadata_build_observation == [0; 32]
            || self.source_closure_identity == [0; 32]
            || self.definition_source_identity == [0; 32]
        {
            return Err("reviewed provider semantic definition is incomplete".to_owned());
        }
        let local_definition_path = self
            .canonical_definition_path
            .strip_prefix("fe2o3_device::")
            .ok_or_else(|| "reviewed provider definition path is not canonical".to_owned())?;
        if structural_local_definition_component_v1(local_definition_path)?
            != self.structural_local_definition_component
        {
            return Err("reviewed provider structural definition component changed".to_owned());
        }
        Ok(())
    }

    pub(crate) fn durable_semantic_identity(
        &self,
        canonical_role: &str,
    ) -> Result<[u8; 32], String> {
        self.validate()?;
        if canonical_role.is_empty() {
            return Err("reviewed provider semantic definition is incomplete".to_owned());
        }

        let mut hasher = Sha256::new();
        hash_source_identity_field(
            &mut hasher,
            PROVIDER_SEMANTIC_DEFINITION_TRANSCRIPT_DOMAIN_V1,
        );
        hash_source_identity_field(&mut hasher, b"semantic-terminal");
        hash_source_identity_field(&mut hasher, canonical_role.as_bytes());
        hash_source_identity_field(&mut hasher, self.provider.crate_name.as_bytes());
        hash_source_identity_field(&mut hasher, self.canonical_definition_path.as_bytes());
        hash_source_identity_field(&mut hasher, &self.structural_local_definition_component);
        hash_source_identity_field(&mut hasher, &self.cargo_metadata_build_observation);
        hash_source_identity_field(&mut hasher, &self.source_closure_identity);
        hash_source_identity_field(&mut hasher, &self.definition_source_identity);
        Ok(hasher.finalize().into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RejectedTrustedProvider {
    pub(crate) marker: &'static str,
    pub(crate) expected_provider_crate: &'static str,
    pub(crate) reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TrustedHalfOperation {
    FromBits(NarrowFloatFormat),
    ToBits(NarrowFloatFormat),
    FromF32(NarrowFloatFormat),
    ToF32(NarrowFloatFormat),
    WidenedBinary {
        format: NarrowFloatFormat,
        op: WidenedFloatBinaryOp,
    },
    Bf16x2FusedMultiplyAdd,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum TrustedAmdGpuInlineOperation {
    VMovB32,
    VAddU32,
    VSubU32,
    VAndB32,
    VOrB32,
    VXorB32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TrustedAmdGpuDiagnosticOperation {
    Print0,
    Print1,
    Print2,
    AssertFail,
    Clock32,
    Trap,
    DebugTrap,
    ProfilingMarker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TrustedDeviceItem {
    KernelError,
    DisjointSlice,
    WriteOnlyDisjointSlice,
    DeviceGlobalMutPtr,
    WorkgroupLdsScope,
    WorkgroupLdsScopeCurrent,
    DynamicLdsExactCurrent,
    DynamicLdsIntoCollectiveRawParts,
    LdsUninitialized,
    WorkgroupPipeline,
    WorkgroupPipelineCurrent,
    WorkgroupPipelineStage,
    WorkgroupPipelineWrite,
    WorkgroupPipelineCommit,
    WorkgroupPipelineWait,
    WorkgroupPipelineConsume,
    WorkgroupPipelineRead,
    WorkgroupPipelineDiscard,
    WorkgroupPipelineRelease,
    Invocation3D,
    Invocation3DCurrent,
    ThreadIndexX,
    ThreadIndexY,
    ThreadIndexZ,
    WorkgroupIndexX,
    WorkgroupIndexY,
    WorkgroupIndexZ,
    WorkgroupDimensionX,
    WorkgroupDimensionY,
    WorkgroupDimensionZ,
    GridDimensionX,
    GridDimensionY,
    GridDimensionZ,
    ThreadIndex,
    DisjointIndex,
    ShiftedIndexSpace,
    BlockedIndexSpace,
    Tiled2DIndexSpace,
    RowStriped2DIndexSpace,
    GridExclusiveIndexSpace,
    DisjointBlock,
    DisjointTile2D,
    DisjointRowStripe2D,
    GridLeader,
    ThreadIndex1d,
    ThreadIndexGet,
    ThreadIndexIntoDisjoint,
    ThreadIndexCheckedShift,
    ThreadIndexCheckedBlock,
    ThreadIndexCheckedTiled2D,
    ThreadIndexCheckedRowStriped2D,
    DisjointIndexGet,
    DisjointIndexCheckedShift,
    DisjointBlockComponentIndex,
    GridLeaderCurrent,
    ThreadIndexOffset,
    ThreadIndexOffsetSigned,
    ThreadIndexStride,
    ThreadIndexStrideOffset,
    DisjointSliceGetMut,
    DisjointSliceGetDisjointMut,
    DisjointSliceGetMutExclusive,
    DisjointSliceGetBlockMut,
    DisjointSliceGetTiled2DMut,
    DisjointSliceGetRowStriped2DMut,
    DisjointSliceGetMutAt,
    DisjointSliceLen,
    WriteOnlyDisjointSliceLen,
    WriteOnlyDisjointSliceWrite,
    WriteOnlyDisjointSliceWriteDisjoint,
    WriteOnlyDisjointSliceWriteExclusive,
    WriteOnlyDisjointSliceWriteBlock,
    WriteOnlyDisjointSliceWriteTiled2D,
    WriteOnlyDisjointSliceWriteRowStriped2D,
    StridedReadView2D,
    StridedReadView2DError,
    StridedReadView2DFromSharedSlice,
    StridedReadView2DLoadOr,
    DeviceGlobalMutPtrU32AsAtomic,
    DeviceGlobalMutPtrI32AsAtomic,
    DeviceGlobalMutPtrU64AsAtomic,
    DeviceGlobalMutPtrI64AsAtomic,
    MemoryOffsetFrom,
    MemoryVolatileLoad,
    MemoryVolatileStore,
    MemoryCopyNonOverlapping,
    MemoryCopyOneNonOverlapping,
    WorkgroupGroup,
    WorkgroupCollectiveScratch,
    WorkgroupCollectivesContext,
    WorkgroupCollectivesCurrent,
    WorkgroupReduceSum,
    WorkgroupInclusiveScanSum,
    WorkgroupExclusiveScanSum,
    Gfx942CollectivesContext,
    Gfx942CollectivesCurrent,
    Gfx942SubgroupReduceSumF32,
    Gfx942SubgroupReduceMaxF32,
    Gfx942StaticLdsU32x256,
    Gfx942StaticLdsU32x256Type,
    Gfx942Wave64ReduceActiveU32,
    Gfx942Wave64Shuffle(Wave64ShuffleScalarV1),
    Gfx942Workgroup256ReduceActiveU32,
    Gfx942Wave64ReduceSum,
    Gfx942Wave64InclusiveScanSum,
    Gfx942Wave64ExclusiveScanSum,
    Gfx942Wave64InclusiveScanHelper,
    Gfx942WorkgroupReduceSum,
    Gfx942WorkgroupInclusiveScanSum,
    Gfx942WorkgroupExclusiveScanSum,
    Gfx942BarrierArrive,
    Gfx942BarrierWait,
    WaveLane,
    Wave64,
    WaveLaneCurrent,
    LdsTile16x16WriteMfmaBf16,
    LdsTile16x16ReadMfmaBf16,
    WorkgroupSyncthreads,
    DeviceMatrix,
    DeviceMatrixCurrent,
    Bf16MfmaProfile,
    MfmaOperandA,
    MfmaOperandB,
    MfmaRegisterTile16x16,
    MfmaLdsXor4Storage,
    MfmaAccumulatorRowMajor,
    Bf16MfmaFragment,
    F32AccumulatorFragment,
    F32AccumulatorFragmentZero,
    F32AccumulatorFragmentIntoValues,
    Bf16MfmaMatrixView,
    Bf16MfmaBColumnMajorMatrixView,
    Bf16MfmaMatrixViewError,
    Bf16MfmaMatrixARowMajor,
    Bf16MfmaMatrixBRowMajor,
    Bf16MfmaBColumnMajor,
    Bf16MfmaMatrixALoadZeroFilledV2,
    Bf16MfmaMatrixBLoadZeroFilledV2,
    Bf16MfmaBColumnMajorLoadZeroFilledV1,
    DeviceMatrixMultiplyAccumulate,
    Gfx950Fp4E2M1Format,
    Gfx950Fp8E4M3Format,
    Gfx950MfmaOperandA,
    Gfx950MfmaOperandB,
    Gfx950MfmaFragment,
    Gfx950F32AccumulatorFragment,
    Gfx950Fp4F32AccumulatorFragmentZero,
    Gfx950Fp4F32AccumulatorFragmentIntoValues,
    Gfx950F32AccumulatorFragmentZero,
    Gfx950F32AccumulatorFragmentIntoValues,
    Gfx950MfmaMatrixViewError,
    Gfx950MfmaMatrixAView,
    Gfx950MfmaMatrixBView,
    Gfx950MfmaMatrixAFp4RowMajor,
    Gfx950MfmaMatrixBFp4RowMajor,
    Gfx950MfmaMatrixARowMajor,
    Gfx950MfmaMatrixBRowMajor,
    Gfx950MfmaMatrixAFp8LoadM16K128,
    Gfx950MfmaMatrixBFp8LoadK128N16,
    Gfx950MfmaMatrixAFp4LoadM16K128,
    Gfx950MfmaMatrixBFp4LoadK128N16,
    Gfx950Matrix,
    Gfx950MatrixCurrent,
    Gfx950MatrixMultiplyAccumulateFp4,
    Gfx950MatrixMultiplyAccumulateFp4Fp8,
    Gfx950MatrixMultiplyAccumulateFp8,
    Gfx950SubgroupContext,
    Gfx950SubgroupCurrent,
    Gfx950SubgroupReduceMaxF32,
    Gfx950SubgroupReduceSumF32,
    Gfx950SubgroupBroadcastF32,
    Gfx950LdsTransposeTile,
    Gfx950LdsTransposeTileCurrent,
    Gfx950LdsTransposeStageB4,
    Gfx950LdsTransposeStageB8,
    Gfx950LdsTransposePublish,
    Gfx950LdsTransposeReadB4,
    Gfx950LdsTransposeReadB8,
    DeviceValue(DeviceValueDiagnosticItem),
    DeviceMath(DeviceMathDiagnosticItem),
    HalfOperation(TrustedHalfOperation),
    AmdGpuInline(TrustedAmdGpuInlineOperation),
    AmdGpuOrderedXorAddE32,
    AmdGpuOrderedProgramE32,
    AmdGpuCompleteBodyE32,
    AmdGpuPhysicalEntryBeginGfx942,
    AmdGpuPhysicalEntryLabelGfx942,
    AmdGpuPhysicalEntryStepGfx942,
    AmdGpuPhysicalGlobalCopyBeginGfx942,
    AmdGpuPhysicalLdsExchangeBeginGfx942,
    AmdGpuPhysicalGlobalCopyLabelGfx942,
    AmdGpuPhysicalLdsExchangeLabelGfx942,
    AmdGpuPhysicalGlobalCopyStepGfx942,
    AmdGpuPhysicalLdsExchangeStepGfx942,
    AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation),
    KernelContext,
    KernelContextIssue,
    ExecutionWorkgroupCapability,
    ExecutionWorkgroupCurrent,
    ExecutionWithWorkgroup,
    MaskedTile1D,
    LaneFragment1D,
    MaskedTile1DLoadMasked,
    MaskedTile1DIntoFragment,
    LaneFragment1DIntoParts,
}

const TRUSTED_ITEMS: &[(TrustedDeviceItem, &str, &str)] = &[
    (
        TrustedDeviceItem::KernelContextIssue,
        "fe2o3_device_kernel_context_issue_v1",
        "fe2o3_device::KernelContext::__compiler_issue",
    ),
    (
        TrustedDeviceItem::KernelContext,
        "fe2o3_device_kernel_context_v1",
        "fe2o3_device::context::KernelContext",
    ),
    (
        TrustedDeviceItem::ExecutionWorkgroupCapability,
        "fe2o3_device_workgroup_capability_v1",
        "fe2o3_device::execution::WorkgroupCapability",
    ),
    (
        TrustedDeviceItem::ExecutionWorkgroupCurrent,
        "fe2o3_device_workgroup_capability_current_v1",
        "fe2o3_device::KernelContext::__compiler_workgroup_capability_current",
    ),
    (
        TrustedDeviceItem::ExecutionWithWorkgroup,
        "fe2o3_device_with_workgroup_v1",
        "fe2o3_device::KernelContext::with_workgroup",
    ),
    (
        TrustedDeviceItem::MaskedTile1D,
        "fe2o3_device_masked_tile_1d_v1",
        "fe2o3_device::tile::MaskedTile1D",
    ),
    (
        TrustedDeviceItem::LaneFragment1D,
        "fe2o3_device_lane_fragment_1d_v1",
        "fe2o3_device::tile::LaneFragment",
    ),
    (
        TrustedDeviceItem::MaskedTile1DLoadMasked,
        "fe2o3_device_masked_tile_1d_load_masked_v1",
        "fe2o3_device::tile::MaskedTile1D::load_masked",
    ),
    (
        TrustedDeviceItem::MaskedTile1DIntoFragment,
        "fe2o3_device_masked_tile_1d_into_fragment_v1",
        "fe2o3_device::tile::MaskedTile1D::into_fragment",
    ),
    (
        TrustedDeviceItem::LaneFragment1DIntoParts,
        "fe2o3_device_lane_fragment_1d_into_parts_v1",
        "fe2o3_device::tile::LaneFragment::into_parts",
    ),
    (
        TrustedDeviceItem::KernelError,
        "fe2o3_device_kernel_error_v1",
        "fe2o3_device::KernelError",
    ),
    (
        TrustedDeviceItem::DisjointSlice,
        "fe2o3_device_disjoint_slice",
        "fe2o3_device::DisjointSlice",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSlice,
        "fe2o3_device_write_only_disjoint_slice_v1",
        "fe2o3_device::WriteOnlyDisjointSlice",
    ),
    (
        TrustedDeviceItem::StridedReadView2D,
        "fe2o3_device_strided_read_view_2d_v1",
        "fe2o3_device::StridedReadView2D",
    ),
    (
        TrustedDeviceItem::StridedReadView2DError,
        "fe2o3_device_strided_read_view_2d_error_v1",
        "fe2o3_device::StridedReadView2DError",
    ),
    (
        TrustedDeviceItem::StridedReadView2DFromSharedSlice,
        "fe2o3_device_strided_read_view_2d_from_shared_slice_v1",
        "fe2o3_device::StridedReadView2D::from_shared_slice",
    ),
    (
        TrustedDeviceItem::StridedReadView2DLoadOr,
        "fe2o3_device_strided_read_view_2d_load_or_v1",
        "fe2o3_device::StridedReadView2D::load_or",
    ),
    (
        TrustedDeviceItem::DeviceGlobalMutPtr,
        "fe2o3_device_ffi_global_mut_ptr_v1",
        "fe2o3_device::DeviceGlobalMutPtr",
    ),
    (
        TrustedDeviceItem::WorkgroupLdsScope,
        "fe2o3_device_workgroup_lds_scope",
        "fe2o3_device::WorkgroupLdsScope",
    ),
    (
        TrustedDeviceItem::WorkgroupLdsScopeCurrent,
        "fe2o3_device_workgroup_lds_scope_current",
        "fe2o3_device::WorkgroupLdsScope::current",
    ),
    (
        TrustedDeviceItem::DynamicLdsExactCurrent,
        "fe2o3_device_dynamic_lds_exact_current_v1",
        "fe2o3_device::DynamicLds::<T>::exact_current",
    ),
    (
        TrustedDeviceItem::DynamicLdsIntoCollectiveRawParts,
        "fe2o3_device_dynamic_lds_into_collective_raw_parts_v1",
        "fe2o3_device::DynamicLds::<T>::into_collective_raw_parts",
    ),
    (
        TrustedDeviceItem::LdsUninitialized,
        "fe2o3_device_lds_uninitialized_v1",
        "fe2o3_device::LdsUninitialized",
    ),
    (
        TrustedDeviceItem::WorkgroupPipeline,
        "fe2o3_device_workgroup_pipeline_v1",
        "fe2o3_device::WorkgroupPipeline",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineCurrent,
        "fe2o3_device_workgroup_pipeline_current_v1",
        "fe2o3_device::WorkgroupPipeline::current",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineStage,
        "fe2o3_device_workgroup_pipeline_stage_v1",
        "fe2o3_device::WorkgroupPipeline::stage",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineWrite,
        "fe2o3_device_workgroup_pipeline_write_v1",
        "fe2o3_device::WorkgroupPipeline::write",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineCommit,
        "fe2o3_device_workgroup_pipeline_commit_v1",
        "fe2o3_device::WorkgroupPipeline::commit",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineWait,
        "fe2o3_device_workgroup_pipeline_wait_v1",
        "fe2o3_device::WorkgroupPipeline::wait",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineConsume,
        "fe2o3_device_workgroup_pipeline_consume_v1",
        "fe2o3_device::WorkgroupPipeline::consume",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineRead,
        "fe2o3_device_workgroup_pipeline_read_v1",
        "fe2o3_device::WorkgroupPipeline::read",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineDiscard,
        "fe2o3_device_workgroup_pipeline_discard_v1",
        "fe2o3_device::WorkgroupPipeline::discard",
    ),
    (
        TrustedDeviceItem::WorkgroupPipelineRelease,
        "fe2o3_device_workgroup_pipeline_release_v1",
        "fe2o3_device::WorkgroupPipeline::release",
    ),
    (
        TrustedDeviceItem::Invocation3D,
        "fe2o3_device_invocation_3d",
        "fe2o3_device::Invocation3D",
    ),
    (
        TrustedDeviceItem::Invocation3DCurrent,
        "fe2o3_device_invocation_3d_current",
        "fe2o3_device::Invocation3D::current",
    ),
    (
        TrustedDeviceItem::ThreadIndexX,
        "fe2o3_device_thread_index_x_v1",
        "fe2o3_device::thread::thread_idx_x",
    ),
    (
        TrustedDeviceItem::ThreadIndexY,
        "fe2o3_device_thread_index_y_v1",
        "fe2o3_device::thread::thread_idx_y",
    ),
    (
        TrustedDeviceItem::ThreadIndexZ,
        "fe2o3_device_thread_index_z_v1",
        "fe2o3_device::thread::thread_idx_z",
    ),
    (
        TrustedDeviceItem::WorkgroupIndexX,
        "fe2o3_device_workgroup_index_x_v1",
        "fe2o3_device::thread::block_idx_x",
    ),
    (
        TrustedDeviceItem::WorkgroupIndexY,
        "fe2o3_device_workgroup_index_y_v1",
        "fe2o3_device::thread::block_idx_y",
    ),
    (
        TrustedDeviceItem::WorkgroupIndexZ,
        "fe2o3_device_workgroup_index_z_v1",
        "fe2o3_device::thread::block_idx_z",
    ),
    (
        TrustedDeviceItem::WorkgroupDimensionX,
        "fe2o3_device_workgroup_dimension_x_v1",
        "fe2o3_device::thread::block_dim_x",
    ),
    (
        TrustedDeviceItem::WorkgroupDimensionY,
        "fe2o3_device_workgroup_dimension_y_v1",
        "fe2o3_device::thread::block_dim_y",
    ),
    (
        TrustedDeviceItem::WorkgroupDimensionZ,
        "fe2o3_device_workgroup_dimension_z_v1",
        "fe2o3_device::thread::block_dim_z",
    ),
    (
        TrustedDeviceItem::GridDimensionX,
        "fe2o3_device_grid_dimension_x_v1",
        "fe2o3_device::thread::grid_dim_x",
    ),
    (
        TrustedDeviceItem::GridDimensionY,
        "fe2o3_device_grid_dimension_y_v1",
        "fe2o3_device::thread::grid_dim_y",
    ),
    (
        TrustedDeviceItem::GridDimensionZ,
        "fe2o3_device_grid_dimension_z_v1",
        "fe2o3_device::thread::grid_dim_z",
    ),
    (
        TrustedDeviceItem::ThreadIndex,
        "fe2o3_device_thread_index",
        "fe2o3_device::ThreadIndex",
    ),
    (
        TrustedDeviceItem::DisjointIndex,
        "fe2o3_device_disjoint_index",
        "fe2o3_device::DisjointIndex",
    ),
    (
        TrustedDeviceItem::ShiftedIndexSpace,
        "fe2o3_device_shifted_index_space",
        "fe2o3_device::Shifted",
    ),
    (
        TrustedDeviceItem::BlockedIndexSpace,
        "fe2o3_device_blocked_index_space",
        "fe2o3_device::Blocked",
    ),
    (
        TrustedDeviceItem::Tiled2DIndexSpace,
        "fe2o3_device_tiled_2d_index_space",
        "fe2o3_device::Tiled2D",
    ),
    (
        TrustedDeviceItem::RowStriped2DIndexSpace,
        "fe2o3_device_row_striped_2d_index_space",
        "fe2o3_device::RowStriped2D",
    ),
    (
        TrustedDeviceItem::GridExclusiveIndexSpace,
        "fe2o3_device_grid_exclusive_index_space",
        "fe2o3_device::GridExclusive",
    ),
    (
        TrustedDeviceItem::GridLeader,
        "fe2o3_device_grid_leader",
        "fe2o3_device::GridLeader",
    ),
    (
        TrustedDeviceItem::DisjointBlock,
        "fe2o3_device_disjoint_block",
        "fe2o3_device::DisjointBlock",
    ),
    (
        TrustedDeviceItem::DisjointTile2D,
        "fe2o3_device_disjoint_tile_2d",
        "fe2o3_device::DisjointTile2D",
    ),
    (
        TrustedDeviceItem::DisjointRowStripe2D,
        "fe2o3_device_disjoint_row_stripe_2d",
        "fe2o3_device::DisjointRowStripe2D",
    ),
    (
        TrustedDeviceItem::ThreadIndex1d,
        "fe2o3_device_thread_index_1d",
        "fe2o3_device::thread::index_1d",
    ),
    (
        TrustedDeviceItem::ThreadIndexGet,
        "fe2o3_device_thread_index_get",
        "fe2o3_device::ThreadIndex::get",
    ),
    (
        TrustedDeviceItem::ThreadIndexIntoDisjoint,
        "fe2o3_device_thread_index_into_disjoint",
        "fe2o3_device::ThreadIndex::into_disjoint",
    ),
    (
        TrustedDeviceItem::ThreadIndexCheckedShift,
        "fe2o3_device_thread_index_checked_shift",
        "fe2o3_device::ThreadIndex::checked_shift",
    ),
    (
        TrustedDeviceItem::ThreadIndexCheckedBlock,
        "fe2o3_device_thread_index_checked_block",
        "fe2o3_device::ThreadIndex::checked_block",
    ),
    (
        TrustedDeviceItem::ThreadIndexCheckedTiled2D,
        "fe2o3_device_thread_index_checked_tiled_2d",
        "fe2o3_device::ThreadIndex::checked_tiled_2d",
    ),
    (
        TrustedDeviceItem::ThreadIndexCheckedRowStriped2D,
        "fe2o3_device_thread_index_checked_row_striped_2d",
        "fe2o3_device::ThreadIndex::checked_row_striped_2d",
    ),
    (
        TrustedDeviceItem::DisjointIndexGet,
        "fe2o3_device_disjoint_index_get",
        "fe2o3_device::DisjointIndex::get",
    ),
    (
        TrustedDeviceItem::DisjointIndexCheckedShift,
        "fe2o3_device_disjoint_index_checked_shift",
        "fe2o3_device::DisjointIndex::checked_shift",
    ),
    (
        TrustedDeviceItem::DisjointBlockComponentIndex,
        "fe2o3_device_disjoint_block_component_index",
        "fe2o3_device::DisjointBlock::component_index",
    ),
    (
        TrustedDeviceItem::GridLeaderCurrent,
        "fe2o3_device_grid_leader_current",
        "fe2o3_device::thread::grid_leader",
    ),
    (
        TrustedDeviceItem::ThreadIndexOffset,
        "fe2o3_device_thread_index_offset",
        "fe2o3_device::ThreadIndex::offset",
    ),
    (
        TrustedDeviceItem::ThreadIndexOffsetSigned,
        "fe2o3_device_thread_index_offset_signed",
        "fe2o3_device::ThreadIndex::offset_signed",
    ),
    (
        TrustedDeviceItem::ThreadIndexStride,
        "fe2o3_device_thread_index_stride",
        "fe2o3_device::ThreadIndex::stride",
    ),
    (
        TrustedDeviceItem::ThreadIndexStrideOffset,
        "fe2o3_device_thread_index_stride_offset",
        "fe2o3_device::ThreadIndex::stride_offset",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetMut,
        "fe2o3_device_disjoint_slice_get_mut",
        "fe2o3_device::DisjointSlice::<T>::get_mut",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetDisjointMut,
        "fe2o3_device_disjoint_slice_get_disjoint_mut",
        "fe2o3_device::DisjointSlice::<T>::get_disjoint_mut",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetMutExclusive,
        "fe2o3_device_disjoint_slice_get_mut_exclusive",
        "fe2o3_device::DisjointSlice::<T, GridExclusive>::get_mut_exclusive",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetBlockMut,
        "fe2o3_device_disjoint_slice_get_block_mut",
        "fe2o3_device::DisjointSlice::<T, Blocked<IndexSpace, LANES_PER_BLOCK, ELEMENTS_PER_LANE>>::get_block_mut",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetTiled2DMut,
        "fe2o3_device_disjoint_slice_get_tiled_2d_mut",
        "fe2o3_device::DisjointSlice::<T, Tiled2D<IndexSpace, LANES_PER_TILE, TILE_ROWS, TILE_COLUMNS, ELEMENTS_PER_LANE>>::get_tiled_2d_mut",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetRowStriped2DMut,
        "fe2o3_device_disjoint_slice_get_row_striped_2d_mut",
        "fe2o3_device::DisjointSlice::<T, RowStriped2D<IndexSpace, LANES_PER_ROW, ELEMENTS_PER_LANE>>::get_row_striped_2d_mut",
    ),
    (
        TrustedDeviceItem::DisjointSliceGetMutAt,
        "fe2o3_device_disjoint_slice_get_mut_at",
        "fe2o3_device::DisjointSlice::<T>::get_mut_at",
    ),
    (
        TrustedDeviceItem::DisjointSliceLen,
        "fe2o3_device_disjoint_slice_len",
        "fe2o3_device::DisjointSlice::<T>::len",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceLen,
        "fe2o3_device_write_only_disjoint_slice_len_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T>::len",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceWrite,
        "fe2o3_device_write_only_disjoint_slice_write_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T>::write",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteDisjoint,
        "fe2o3_device_write_only_disjoint_slice_write_disjoint_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T>::write_disjoint",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteExclusive,
        "fe2o3_device_write_only_disjoint_slice_write_exclusive_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T, GridExclusive>::write_exclusive",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteBlock,
        "fe2o3_device_write_only_disjoint_slice_write_block_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T, Blocked<IndexSpace, LANES_PER_BLOCK, ELEMENTS_PER_LANE>>::write_block",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteTiled2D,
        "fe2o3_device_write_only_disjoint_slice_write_tiled_2d_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T, Tiled2D<IndexSpace, LANES_PER_TILE, TILE_ROWS, TILE_COLUMNS, ELEMENTS_PER_LANE>>::write_tiled_2d",
    ),
    (
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteRowStriped2D,
        "fe2o3_device_write_only_disjoint_slice_write_row_striped_2d_v1",
        "fe2o3_device::WriteOnlyDisjointSlice::<T, RowStriped2D<IndexSpace, LANES_PER_ROW, ELEMENTS_PER_LANE>>::write_row_striped_2d",
    ),
    (
        TrustedDeviceItem::DeviceGlobalMutPtrU32AsAtomic,
        "fe2o3_device_global_mut_ptr_u32_as_atomic_v1",
        "fe2o3_device::DeviceGlobalMutPtr::<u32>::as_atomic",
    ),
    (
        TrustedDeviceItem::DeviceGlobalMutPtrI32AsAtomic,
        "fe2o3_device_global_mut_ptr_i32_as_atomic_v1",
        "fe2o3_device::DeviceGlobalMutPtr::<i32>::as_atomic",
    ),
    (
        TrustedDeviceItem::DeviceGlobalMutPtrU64AsAtomic,
        "fe2o3_device_global_mut_ptr_u64_as_atomic_v1",
        "fe2o3_device::DeviceGlobalMutPtr::<u64>::as_atomic",
    ),
    (
        TrustedDeviceItem::DeviceGlobalMutPtrI64AsAtomic,
        "fe2o3_device_global_mut_ptr_i64_as_atomic_v1",
        "fe2o3_device::DeviceGlobalMutPtr::<i64>::as_atomic",
    ),
    (
        TrustedDeviceItem::MemoryOffsetFrom,
        "fe2o3_device_memory_offset_from_v1",
        "fe2o3_device::memory::offset_from",
    ),
    (
        TrustedDeviceItem::MemoryVolatileLoad,
        "fe2o3_device_memory_volatile_load_v1",
        "fe2o3_device::memory::volatile_load",
    ),
    (
        TrustedDeviceItem::MemoryVolatileStore,
        "fe2o3_device_memory_volatile_store_v1",
        "fe2o3_device::memory::volatile_store",
    ),
    (
        TrustedDeviceItem::MemoryCopyNonOverlapping,
        "fe2o3_device_memory_copy_nonoverlapping_v1",
        "fe2o3_device::memory::copy_nonoverlapping_unchecked",
    ),
    (
        TrustedDeviceItem::MemoryCopyOneNonOverlapping,
        "fe2o3_device_memory_copy_one_nonoverlapping_v1",
        "fe2o3_device::memory::copy_one_nonoverlapping",
    ),
    (
        TrustedDeviceItem::WorkgroupGroup,
        "fe2o3_device_workgroup_group_v1",
        "fe2o3_device::Workgroup",
    ),
    (
        TrustedDeviceItem::WorkgroupCollectiveScratch,
        "fe2o3_device_workgroup_collective_scratch_v1",
        "fe2o3_device::WorkgroupCollectiveScratch",
    ),
    (
        TrustedDeviceItem::WorkgroupCollectivesContext,
        "fe2o3_device_workgroup_collectives_context_v1",
        "fe2o3_device::WorkgroupCollectives",
    ),
    (
        TrustedDeviceItem::WorkgroupCollectivesCurrent,
        "fe2o3_device_workgroup_collectives_current_v1",
        "fe2o3_device::WorkgroupCollectives::current",
    ),
    (
        TrustedDeviceItem::WorkgroupReduceSum,
        "fe2o3_device_workgroup_reduce_sum_v1",
        "fe2o3_device::WorkgroupCollectives::reduce_sum_portable",
    ),
    (
        TrustedDeviceItem::WorkgroupInclusiveScanSum,
        "fe2o3_device_workgroup_inclusive_scan_sum_v1",
        "fe2o3_device::WorkgroupCollectives::inclusive_scan_sum",
    ),
    (
        TrustedDeviceItem::WorkgroupExclusiveScanSum,
        "fe2o3_device_workgroup_exclusive_scan_sum_v1",
        "fe2o3_device::WorkgroupCollectives::exclusive_scan_sum",
    ),
    (
        TrustedDeviceItem::Gfx942CollectivesContext,
        "fe2o3_device_gfx942_collectives_context_v1",
        "fe2o3_device::Gfx942Collectives",
    ),
    (
        TrustedDeviceItem::Gfx942CollectivesCurrent,
        "fe2o3_device_gfx942_collectives_current_v1",
        "fe2o3_device::Gfx942Collectives::current",
    ),
    (
        TrustedDeviceItem::Gfx942SubgroupReduceSumF32,
        "fe2o3_device_gfx942_subgroup_reduce_sum_f32_v1",
        "fe2o3_device::Gfx942Collectives::subgroup_reduce_sum_f32",
    ),
    (
        TrustedDeviceItem::Gfx942SubgroupReduceMaxF32,
        "fe2o3_device_gfx942_subgroup_reduce_max_f32_v1",
        "fe2o3_device::Gfx942Collectives::subgroup_reduce_max_f32",
    ),
    (
        TrustedDeviceItem::Gfx942StaticLdsU32x256,
        "fe2o3_device_gfx942_static_lds_u32x256_v1",
        "fe2o3_device::Gfx942Collectives::static_lds_u32x256",
    ),
    (
        TrustedDeviceItem::Gfx942StaticLdsU32x256Type,
        "fe2o3_device_gfx942_static_lds_u32x256_type_v1",
        "fe2o3_device::Gfx942StaticLdsU32x256",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64ReduceActiveU32,
        "fe2o3_device_gfx942_wave64_reduce_active_u32_v1",
        "fe2o3_device::Gfx942Collectives::wave64_reduce_sum_active_u32",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::U32),
        "fe2o3_device_gfx942_wave64_shuffle_u32_v1",
        "<u32 as fe2o3_device::Gfx942CollectiveElement>::__fe2o3_wave64_shuffle_index",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::I32),
        "fe2o3_device_gfx942_wave64_shuffle_i32_v1",
        "<i32 as fe2o3_device::Gfx942CollectiveElement>::__fe2o3_wave64_shuffle_index",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::F32),
        "fe2o3_device_gfx942_wave64_shuffle_f32_v1",
        "<f32 as fe2o3_device::Gfx942CollectiveElement>::__fe2o3_wave64_shuffle_index",
    ),
    (
        TrustedDeviceItem::Gfx942Workgroup256ReduceActiveU32,
        "fe2o3_device_gfx942_workgroup256_reduce_active_u32_v1",
        "fe2o3_device::Gfx942Collectives::workgroup256_reduce_sum_active_u32",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64ReduceSum,
        "fe2o3_device_gfx942_wave64_reduce_sum_v1",
        "fe2o3_device::SubgroupTile::<64>::reduce_sum",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64InclusiveScanSum,
        "fe2o3_device_gfx942_wave64_inclusive_scan_sum_v1",
        "fe2o3_device::SubgroupTile::<64>::inclusive_scan_sum",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64ExclusiveScanSum,
        "fe2o3_device_gfx942_wave64_exclusive_scan_sum_v1",
        "fe2o3_device::SubgroupTile::<64>::exclusive_scan_sum",
    ),
    (
        TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper,
        "fe2o3_device_gfx942_wave64_inclusive_scan_helper_v1",
        "fe2o3_device::collective::wave64_inclusive_scan",
    ),
    (
        TrustedDeviceItem::Gfx942WorkgroupReduceSum,
        "fe2o3_device_gfx942_workgroup_reduce_sum_v1",
        "fe2o3_device::Workgroup::reduce_sum",
    ),
    (
        TrustedDeviceItem::Gfx942WorkgroupInclusiveScanSum,
        "fe2o3_device_gfx942_workgroup_inclusive_scan_sum_v1",
        "fe2o3_device::Workgroup::inclusive_scan_sum",
    ),
    (
        TrustedDeviceItem::Gfx942WorkgroupExclusiveScanSum,
        "fe2o3_device_gfx942_workgroup_exclusive_scan_sum_v1",
        "fe2o3_device::Workgroup::exclusive_scan_sum",
    ),
    (
        TrustedDeviceItem::Gfx942BarrierArrive,
        "fe2o3_device_gfx942_barrier_arrive_v1",
        "fe2o3_device::sync::gfx942_barrier_arrive",
    ),
    (
        TrustedDeviceItem::Gfx942BarrierWait,
        "fe2o3_device_gfx942_barrier_wait_v1",
        "fe2o3_device::sync::gfx942_barrier_wait",
    ),
    (
        TrustedDeviceItem::WaveLane,
        "fe2o3_device_wave_lane",
        "fe2o3_device::WaveLane",
    ),
    (
        TrustedDeviceItem::Wave64,
        "fe2o3_device_wave64_width_v1",
        "fe2o3_device::Wave64",
    ),
    (
        TrustedDeviceItem::WaveLaneCurrent,
        "fe2o3_device_wave_lane_current",
        "fe2o3_device::WaveLane::current",
    ),
    (
        TrustedDeviceItem::LdsTile16x16WriteMfmaBf16,
        "fe2o3_device_lds_tile16x16_write_mfma_bf16_v1",
        "fe2o3_device::LdsTile16x16::write_mfma_fragment",
    ),
    (
        TrustedDeviceItem::LdsTile16x16ReadMfmaBf16,
        "fe2o3_device_lds_tile16x16_read_mfma_bf16_v1",
        "fe2o3_device::LdsTile16x16::read_mfma_fragment",
    ),
    (
        TrustedDeviceItem::WorkgroupSyncthreads,
        "fe2o3_device_workgroup_syncthreads_v1",
        "fe2o3_device::sync::syncthreads",
    ),
    (
        TrustedDeviceItem::DeviceMatrix,
        "fe2o3_device_matrix_context_v1",
        "fe2o3_device::DeviceMatrix",
    ),
    (
        TrustedDeviceItem::DeviceMatrixCurrent,
        "fe2o3_device_matrix_context_current_v1",
        "fe2o3_device::DeviceMatrix::current",
    ),
    (
        TrustedDeviceItem::Bf16MfmaProfile,
        "fe2o3_device_mfma_bf16_f32_m16n16k16_profile_v1",
        "fe2o3_device::Bf16F32M16N16K16",
    ),
    (
        TrustedDeviceItem::MfmaOperandA,
        "fe2o3_device_mfma_operand_a_role_v1",
        "fe2o3_device::MfmaOperandA",
    ),
    (
        TrustedDeviceItem::MfmaOperandB,
        "fe2o3_device_mfma_operand_b_role_v1",
        "fe2o3_device::MfmaOperandB",
    ),
    (
        TrustedDeviceItem::MfmaRegisterTile16x16,
        "fe2o3_device_mfma_tile16x16_register_distribution_v1",
        "fe2o3_device::MfmaRegisterTile16x16",
    ),
    (
        TrustedDeviceItem::MfmaLdsXor4Storage,
        "fe2o3_device_mfma_lds_xor4_storage_layout_v1",
        "fe2o3_device::MfmaLdsXor4",
    ),
    (
        TrustedDeviceItem::MfmaAccumulatorRowMajor,
        "fe2o3_device_mfma_accumulator_row_major_distribution_v1",
        "fe2o3_device::MfmaAccumulatorRowMajor",
    ),
    (
        TrustedDeviceItem::Bf16MfmaFragment,
        "fe2o3_device_bf16_mfma_fragment_v1",
        "fe2o3_device::Bf16MfmaFragment",
    ),
    (
        TrustedDeviceItem::F32AccumulatorFragment,
        "fe2o3_device_f32_accumulator_fragment_v1",
        "fe2o3_device::F32AccumulatorFragment",
    ),
    (
        TrustedDeviceItem::F32AccumulatorFragmentZero,
        "fe2o3_device_f32_accumulator_fragment_zero_v1",
        "fe2o3_device::F32AccumulatorFragment::zero",
    ),
    (
        TrustedDeviceItem::F32AccumulatorFragmentIntoValues,
        "fe2o3_device_f32_accumulator_fragment_into_values_v1",
        "fe2o3_device::F32AccumulatorFragment::into_values",
    ),
    (
        TrustedDeviceItem::Bf16MfmaMatrixView,
        "fe2o3_device_bf16_mfma_matrix_view_v1",
        "fe2o3_device::Bf16MfmaMatrix",
    ),
    (
        TrustedDeviceItem::Bf16MfmaMatrixViewError,
        "fe2o3_device_bf16_mfma_matrix_view_error_v1",
        "fe2o3_device::Bf16MatrixViewError",
    ),
    (
        TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView,
        "fe2o3_device_bf16_mfma_b_column_major_matrix_v1",
        "fe2o3_device::Bf16MfmaBColumnMajorMatrix",
    ),
    (
        TrustedDeviceItem::Bf16MfmaBColumnMajor,
        "fe2o3_device_bf16_mfma_b_column_major_v1",
        "fe2o3_device::Bf16MfmaBColumnMajorMatrix::column_major",
    ),
    (
        TrustedDeviceItem::Bf16MfmaBColumnMajorLoadZeroFilledV1,
        "fe2o3_device_bf16_mfma_b_column_major_load_zero_filled_v1",
        "fe2o3_device::Bf16MfmaBColumnMajorMatrix::load_k16n16",
    ),
    (
        TrustedDeviceItem::Bf16MfmaMatrixARowMajor,
        "fe2o3_device_bf16_mfma_matrix_a_row_major_v1",
        "fe2o3_device::Bf16MfmaAMatrix::row_major",
    ),
    (
        TrustedDeviceItem::Bf16MfmaMatrixBRowMajor,
        "fe2o3_device_bf16_mfma_matrix_b_row_major_v1",
        "fe2o3_device::Bf16MfmaBMatrix::row_major",
    ),
    (
        TrustedDeviceItem::Bf16MfmaMatrixALoadZeroFilledV2,
        "fe2o3_device_bf16_mfma_matrix_a_load_zero_filled_v2",
        "fe2o3_device::Bf16MfmaAMatrix::load_m16k16",
    ),
    (
        TrustedDeviceItem::Bf16MfmaMatrixBLoadZeroFilledV2,
        "fe2o3_device_bf16_mfma_matrix_b_load_zero_filled_v2",
        "fe2o3_device::Bf16MfmaBMatrix::load_k16n16",
    ),
    (
        TrustedDeviceItem::DeviceMatrixMultiplyAccumulate,
        "fe2o3_device_matrix_mfma_bf16_f32_m16n16k16_v1",
        "fe2o3_device::DeviceMatrix::multiply_accumulate",
    ),
    (
        TrustedDeviceItem::Gfx950Fp4E2M1Format,
        "fe2o3_device_gfx950_fp4_e2m1_format_v1",
        "fe2o3_device::Gfx950Fp4E2M1",
    ),
    (
        TrustedDeviceItem::Gfx950Fp8E4M3Format,
        "fe2o3_device_gfx950_fp8_e4m3_format_v1",
        "fe2o3_device::Gfx950Fp8E4M3",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaOperandA,
        "fe2o3_device_gfx950_mfma_operand_a_role_v1",
        "fe2o3_device::Gfx950MfmaOperandA",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaOperandB,
        "fe2o3_device_gfx950_mfma_operand_b_role_v1",
        "fe2o3_device::Gfx950MfmaOperandB",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaFragment,
        "fe2o3_device_gfx950_mfma_fragment_v1",
        "fe2o3_device::Gfx950MfmaFragment",
    ),
    (
        TrustedDeviceItem::Gfx950F32AccumulatorFragment,
        "fe2o3_device_gfx950_f32_accumulator_fragment_v1",
        "fe2o3_device::Gfx950F32AccumulatorFragment",
    ),
    (
        TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentZero,
        "fe2o3_device_gfx950_fp4_f32_accumulator_zero_v1",
        "fe2o3_device::Gfx950F32AccumulatorFragment<Gfx950Fp4E2M1>::zero",
    ),
    (
        TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentIntoValues,
        "fe2o3_device_gfx950_fp4_f32_accumulator_into_values_v1",
        "fe2o3_device::Gfx950F32AccumulatorFragment<Gfx950Fp4E2M1>::into_values",
    ),
    (
        TrustedDeviceItem::Gfx950F32AccumulatorFragmentZero,
        "fe2o3_device_gfx950_f32_accumulator_zero_v1",
        "fe2o3_device::Gfx950F32AccumulatorFragment::zero",
    ),
    (
        TrustedDeviceItem::Gfx950F32AccumulatorFragmentIntoValues,
        "fe2o3_device_gfx950_f32_accumulator_into_values_v1",
        "fe2o3_device::Gfx950F32AccumulatorFragment::into_values",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixViewError,
        "fe2o3_device_gfx950_mfma_matrix_view_error_v1",
        "fe2o3_device::Gfx950MatrixViewError",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixAView,
        "fe2o3_device_gfx950_mfma_matrix_a_view_v1",
        "fe2o3_device::Gfx950MfmaAMatrix",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixBView,
        "fe2o3_device_gfx950_mfma_matrix_b_view_v1",
        "fe2o3_device::Gfx950MfmaBMatrix",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixAFp4RowMajor,
        "fe2o3_device_gfx950_mfma_matrix_a_fp4_row_major_v1",
        "fe2o3_device::Gfx950Fp4MfmaAMatrix::row_major",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixBFp4RowMajor,
        "fe2o3_device_gfx950_mfma_matrix_b_fp4_row_major_v1",
        "fe2o3_device::Gfx950Fp4MfmaBMatrix::row_major",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixARowMajor,
        "fe2o3_device_gfx950_mfma_matrix_a_row_major_v1",
        "fe2o3_device::Gfx950MfmaAMatrix::row_major",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixBRowMajor,
        "fe2o3_device_gfx950_mfma_matrix_b_row_major_v1",
        "fe2o3_device::Gfx950MfmaBMatrix::row_major",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixAFp4LoadM16K128,
        "fe2o3_device_gfx950_mfma_matrix_a_fp4_load_m16k128_v1",
        "fe2o3_device::Gfx950Fp4MfmaAMatrix::load_m16k128",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixBFp4LoadK128N16,
        "fe2o3_device_gfx950_mfma_matrix_b_fp4_load_k128n16_v1",
        "fe2o3_device::Gfx950Fp4MfmaBMatrix::load_k128n16",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixAFp8LoadM16K128,
        "fe2o3_device_gfx950_mfma_matrix_a_fp8_load_m16k128_v1",
        "fe2o3_device::Gfx950Fp8MfmaAMatrix::load_m16k128",
    ),
    (
        TrustedDeviceItem::Gfx950MfmaMatrixBFp8LoadK128N16,
        "fe2o3_device_gfx950_mfma_matrix_b_fp8_load_k128n16_v1",
        "fe2o3_device::Gfx950Fp8MfmaBMatrix::load_k128n16",
    ),
    (
        TrustedDeviceItem::Gfx950Matrix,
        "fe2o3_device_gfx950_matrix_context_v1",
        "fe2o3_device::Gfx950Matrix",
    ),
    (
        TrustedDeviceItem::Gfx950MatrixCurrent,
        "fe2o3_device_gfx950_matrix_context_current_v1",
        "fe2o3_device::Gfx950Matrix::current",
    ),
    (
        TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4,
        "fe2o3_device_gfx950_mfma_fp4_f32_m16n16k128_v1",
        "fe2o3_device::Gfx950Matrix::multiply_accumulate_fp4",
    ),
    (
        TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4Fp8,
        "fe2o3_device_gfx950_mfma_fp4_fp8_f32_m16n16k128_v1",
        "fe2o3_device::Gfx950Matrix::multiply_accumulate_fp4_fp8",
    ),
    (
        TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp8,
        "fe2o3_device_gfx950_mfma_fp8_f32_m16n16k128_v1",
        "fe2o3_device::Gfx950Matrix::multiply_accumulate_fp8",
    ),
    (
        TrustedDeviceItem::Gfx950SubgroupContext,
        "fe2o3_device_gfx950_subgroup_context_v1",
        "fe2o3_device::Gfx950Subgroup",
    ),
    (
        TrustedDeviceItem::Gfx950SubgroupCurrent,
        "fe2o3_device_gfx950_subgroup_current_v1",
        "fe2o3_device::Gfx950Subgroup::current",
    ),
    (
        TrustedDeviceItem::Gfx950SubgroupReduceMaxF32,
        "fe2o3_device_gfx950_subgroup_reduce_max_f32_v1",
        "fe2o3_device::Gfx950Subgroup::reduce_max_f32",
    ),
    (
        TrustedDeviceItem::Gfx950SubgroupReduceSumF32,
        "fe2o3_device_gfx950_subgroup_reduce_sum_f32_v1",
        "fe2o3_device::Gfx950Subgroup::reduce_sum_f32",
    ),
    (
        TrustedDeviceItem::Gfx950SubgroupBroadcastF32,
        "fe2o3_device_gfx950_subgroup_broadcast_f32_v1",
        "fe2o3_device::Gfx950Subgroup::broadcast_f32",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposeTile,
        "fe2o3_device_gfx950_lds_transpose_tile_v1",
        "fe2o3_device::Gfx950LdsTransposeTile",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposeTileCurrent,
        "fe2o3_device_gfx950_lds_transpose_tile_current_v1",
        "fe2o3_device::Gfx950LdsTransposeTile::current",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposeStageB4,
        "fe2o3_device_gfx950_lds_transpose_stage_b4_v1",
        "fe2o3_device::Gfx950LdsTransposeTile<Gfx950Fp4E2M1>::stage_k_transposed",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposeStageB8,
        "fe2o3_device_gfx950_lds_transpose_stage_b8_v1",
        "fe2o3_device::Gfx950LdsTransposeTile<Gfx950Fp8E4M3>::stage_k_transposed",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposePublish,
        "fe2o3_device_gfx950_lds_transpose_publish_v1",
        "fe2o3_device::Gfx950LdsTransposeTile::publish",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposeReadB4,
        "fe2o3_device_gfx950_lds_transpose_read_b4_v1",
        "fe2o3_device::Gfx950LdsTransposeTile<Gfx950Fp4E2M1>::read_mfma_fragment",
    ),
    (
        TrustedDeviceItem::Gfx950LdsTransposeReadB8,
        "fe2o3_device_gfx950_lds_transpose_read_b8_v1",
        "fe2o3_device::Gfx950LdsTransposeTile<Gfx950Fp8E4M3>::read_mfma_fragment",
    ),
    (
        TrustedDeviceItem::AmdGpuOrderedXorAddE32,
        "fe2o3_device_amdgpu_ordered_xor_add_e32_v1",
        "fe2o3_device::diagnostics::__amdgpu_ordered_xor_add_e32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuOrderedProgramE32,
        "fe2o3_device_amdgpu_ordered_program_e32_v1",
        "fe2o3_device::diagnostics::__amdgpu_ordered_program_e32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalEntryBeginGfx942,
        "fe2o3_device_amdgpu_physical_entry_begin_gfx942_v1",
        "fe2o3_device::physical_entry_v1::__amdgpu_physical_entry_begin_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalGlobalCopyBeginGfx942,
        "fe2o3_device_amdgpu_physical_global_copy_begin_gfx942_v1",
        "fe2o3_device::physical_global_copy_v1::__amdgpu_physical_global_copy_begin_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalLdsExchangeBeginGfx942,
        "fe2o3_device_amdgpu_physical_lds_exchange_begin_gfx942_v1",
        "fe2o3_device::physical_lds_exchange_v1::__amdgpu_physical_lds_exchange_begin_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalEntryLabelGfx942,
        "fe2o3_device_amdgpu_physical_entry_label_gfx942_v1",
        "fe2o3_device::physical_entry_v1::__amdgpu_physical_entry_label_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalGlobalCopyLabelGfx942,
        "fe2o3_device_amdgpu_physical_global_copy_label_gfx942_v1",
        "fe2o3_device::physical_global_copy_v1::__amdgpu_physical_global_copy_label_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalLdsExchangeLabelGfx942,
        "fe2o3_device_amdgpu_physical_lds_exchange_label_gfx942_v1",
        "fe2o3_device::physical_lds_exchange_v1::__amdgpu_physical_lds_exchange_label_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalEntryStepGfx942,
        "fe2o3_device_amdgpu_physical_entry_step_gfx942_v1",
        "fe2o3_device::physical_entry_v1::__amdgpu_physical_entry_step_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalGlobalCopyStepGfx942,
        "fe2o3_device_amdgpu_physical_global_copy_step_gfx942_v1",
        "fe2o3_device::physical_global_copy_v1::__amdgpu_physical_global_copy_step_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuPhysicalLdsExchangeStepGfx942,
        "fe2o3_device_amdgpu_physical_lds_exchange_step_gfx942_v1",
        "fe2o3_device::physical_lds_exchange_v1::__amdgpu_physical_lds_exchange_step_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuCompleteBodyE32,
        "fe2o3_device_amdgpu_complete_body_gfx942_v1",
        "fe2o3_device::diagnostics::__amdgpu_complete_body_gfx942_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VMovB32),
        "fe2o3_device_amdgpu_v_mov_b32_v1",
        "fe2o3_device::diagnostics::__amdgpu_v_mov_b32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VAddU32),
        "fe2o3_device_amdgpu_v_add_u32_v1",
        "fe2o3_device::diagnostics::__amdgpu_v_add_u32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VSubU32),
        "fe2o3_device_amdgpu_v_sub_u32_v1",
        "fe2o3_device::diagnostics::__amdgpu_v_sub_u32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VAndB32),
        "fe2o3_device_amdgpu_v_and_b32_v1",
        "fe2o3_device::diagnostics::__amdgpu_v_and_b32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VOrB32),
        "fe2o3_device_amdgpu_v_or_b32_v1",
        "fe2o3_device::diagnostics::__amdgpu_v_or_b32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VXorB32),
        "fe2o3_device_amdgpu_v_xor_b32_v1",
        "fe2o3_device::diagnostics::__amdgpu_v_xor_b32_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Print0),
        "fe2o3_device_gpu_printf_0_v1",
        "fe2o3_device::diagnostics::__gpu_printf_0_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Print1),
        "fe2o3_device_gpu_printf_1_v1",
        "fe2o3_device::diagnostics::__gpu_printf_1_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Print2),
        "fe2o3_device_gpu_printf_2_v1",
        "fe2o3_device::diagnostics::__gpu_printf_2_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::AssertFail),
        "fe2o3_device_gpu_assert_fail_v1",
        "fe2o3_device::diagnostics::__gpu_assert_fail_v1",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Clock32),
        "fe2o3_device_clock32_v1",
        "fe2o3_device::diagnostics::clock32",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Trap),
        "fe2o3_device_trap_v1",
        "fe2o3_device::diagnostics::trap",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::DebugTrap),
        "fe2o3_device_debugtrap_v1",
        "fe2o3_device::diagnostics::debugtrap",
    ),
    (
        TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::ProfilingMarker),
        "fe2o3_device_profiling_marker_v1",
        "fe2o3_device::diagnostics::__profiling_marker_v1",
    ),
];

const HALF_MATH_DIAGNOSTIC_ITEMS: &[(&str, &str)] = &[
    ("fe2o3_device_f16_v1", "fe2o3_device::F16"),
    ("fe2o3_device_bf16_v1", "fe2o3_device::Bf16"),
    ("fe2o3_device_bf16x2_v1", "fe2o3_device::Bf16x2"),
    ("fe2o3_device_math_context_v1", "fe2o3_device::DeviceMath"),
    (
        "fe2o3_device_math_context_from_compiler_v1",
        "fe2o3_device::DeviceMath::current",
    ),
    (
        "fe2o3_device_math_sqrt_f32_v1",
        "fe2o3_device::DeviceMath::sqrt_f32",
    ),
    (
        "fe2o3_device_math_fma_f32_v1",
        "fe2o3_device::DeviceMath::mul_add_f32",
    ),
    (
        "fe2o3_device_math_floor_f32_v1",
        "fe2o3_device::DeviceMath::floor_f32",
    ),
    (
        "fe2o3_device_math_ceil_f32_v1",
        "fe2o3_device::DeviceMath::ceil_f32",
    ),
    (
        "fe2o3_device_math_trunc_f32_v1",
        "fe2o3_device::DeviceMath::trunc_f32",
    ),
    (
        "fe2o3_device_math_roundeven_f32_v1",
        "fe2o3_device::DeviceMath::round_ties_even_f32",
    ),
    (
        "fe2o3_device_math_sin_f32_v1",
        "fe2o3_device::DeviceMath::sin_f32",
    ),
    (
        "fe2o3_device_math_cos_f32_v1",
        "fe2o3_device::DeviceMath::cos_f32",
    ),
    (
        "fe2o3_device_math_exp_f32_v1",
        "fe2o3_device::DeviceMath::exp_f32",
    ),
    (
        "fe2o3_device_math_exp2_f32_v1",
        "fe2o3_device::DeviceMath::exp2_f32",
    ),
    (
        "fe2o3_device_math_log_f32_v1",
        "fe2o3_device::DeviceMath::ln_f32",
    ),
    (
        "fe2o3_device_math_log2_f32_v1",
        "fe2o3_device::DeviceMath::log2_f32",
    ),
    (
        "fe2o3_device_math_log10_f32_v1",
        "fe2o3_device::DeviceMath::log10_f32",
    ),
    (
        "fe2o3_device_math_fma_bf16x2_v1",
        "fe2o3_device::DeviceMath::mul_add_bf16x2",
    ),
];

impl TrustedDeviceItem {
    pub(crate) fn canonical_path(self) -> &'static str {
        match self {
            Self::DeviceValue(value) => half_math_path(Fe2o3DeviceDiagnosticItem::Value(value)),
            Self::DeviceMath(math) => half_math_path(Fe2o3DeviceDiagnosticItem::Math(math)),
            Self::HalfOperation(operation) => operation.canonical_path(),
            _ => TRUSTED_ITEMS
                .iter()
                .find_map(|(item, _, path)| (*item == self).then_some(*path))
                .expect("every trusted device item has one canonical path"),
        }
    }

    pub(crate) const fn expected_provider_crate(self) -> &'static str {
        "fe2o3_device"
    }
}

pub(crate) fn definition(tcx: TyCtxt<'_>, item: TrustedDeviceItem) -> Option<DefId> {
    let marker = TRUSTED_ITEMS
        .iter()
        .find_map(|(candidate, marker, _)| (*candidate == item).then_some(*marker))
        .or_else(|| match item {
            TrustedDeviceItem::DeviceValue(value) => {
                HALF_MATH_DIAGNOSTIC_ITEMS.iter().find_map(|(marker, _)| {
                    (dialect_amdgcn::recognize_fe2o3_device_diagnostic_item(marker)
                        == Some(Fe2o3DeviceDiagnosticItem::Value(value)))
                    .then_some(*marker)
                })
            }
            TrustedDeviceItem::DeviceMath(math) => {
                HALF_MATH_DIAGNOSTIC_ITEMS.iter().find_map(|(marker, _)| {
                    (dialect_amdgcn::recognize_fe2o3_device_diagnostic_item(marker)
                        == Some(Fe2o3DeviceDiagnosticItem::Math(math)))
                    .then_some(*marker)
                })
            }
            _ => None,
        })?;
    let def_id = tcx.get_diagnostic_item(Symbol::intern(marker))?;
    (classify(tcx, def_id) == Some(item)).then_some(def_id)
}

pub(crate) fn authenticated_compiler_definition_observation_v1(
    tcx: TyCtxt<'_>,
    item: TrustedDeviceItem,
) -> Result<(String, [u8; 32]), String> {
    let definition = definition(tcx, item).ok_or_else(|| {
        format!(
            "trusted provider item `{}` is unavailable",
            item.canonical_path()
        )
    })?;
    let observation = reviewed_provider_semantic_definition_v1(tcx, definition)?;
    validate_reviewed_fe2o3_device_provider_definition_v1(item, &observation)?;
    Ok((
        observation.canonical_definition_path,
        observation.source_closure_identity,
    ))
}

pub(crate) fn classify(tcx: TyCtxt<'_>, def_id: DefId) -> Option<TrustedDeviceItem> {
    let direct = TRUSTED_ITEMS
        .iter()
        .find_map(|(item, marker, _)| {
            (tcx.get_diagnostic_item(Symbol::intern(marker)) == Some(def_id)).then_some(*item)
        })
        .or_else(|| {
            HALF_MATH_DIAGNOSTIC_ITEMS.iter().find_map(|(marker, _)| {
                if tcx.get_diagnostic_item(Symbol::intern(marker)) != Some(def_id) {
                    return None;
                }
                let item = dialect_amdgcn::recognize_fe2o3_device_diagnostic_item(marker)
                    .expect("half/math registry markers must remain canonical");
                Some(match item {
                    Fe2o3DeviceDiagnosticItem::Value(value) => {
                        TrustedDeviceItem::DeviceValue(value)
                    }
                    Fe2o3DeviceDiagnosticItem::Math(math) => TrustedDeviceItem::DeviceMath(math),
                })
            })
        });
    if let Some(item) = direct {
        return provider_rule(tcx, def_id, item).is_ok().then_some(item);
    }
    classify_half_operation(tcx, def_id).map(TrustedDeviceItem::HalfOperation)
}

pub(crate) fn rejected_provider(tcx: TyCtxt<'_>, def_id: DefId) -> Option<RejectedTrustedProvider> {
    let (item, marker) = TRUSTED_ITEMS
        .iter()
        .find_map(|(item, marker, _)| {
            (tcx.get_diagnostic_item(Symbol::intern(marker)) == Some(def_id))
                .then_some((*item, *marker))
        })
        .or_else(|| {
            HALF_MATH_DIAGNOSTIC_ITEMS.iter().find_map(|(marker, _)| {
                if tcx.get_diagnostic_item(Symbol::intern(marker)) != Some(def_id) {
                    return None;
                }
                let recognized = dialect_amdgcn::recognize_fe2o3_device_diagnostic_item(marker)?;
                let item = match recognized {
                    Fe2o3DeviceDiagnosticItem::Value(value) => {
                        TrustedDeviceItem::DeviceValue(value)
                    }
                    Fe2o3DeviceDiagnosticItem::Math(math) => TrustedDeviceItem::DeviceMath(math),
                };
                Some((item, *marker))
            })
        })?;
    provider_rule(tcx, def_id, item)
        .err()
        .map(|reason| RejectedTrustedProvider {
            marker,
            expected_provider_crate: item.expected_provider_crate(),
            reason,
        })
}

fn provider_rule(tcx: TyCtxt<'_>, def_id: DefId, item: TrustedDeviceItem) -> Result<(), String> {
    let definition = reviewed_provider_semantic_definition_v1(tcx, def_id)?;
    validate_reviewed_fe2o3_device_provider_definition_v1(item, &definition)?;
    if let TrustedDeviceItem::Gfx942Wave64Shuffle(scalar) = item {
        wave64_shuffle_provider_v1::validate_definition(tcx, def_id, scalar, &definition)?;
    }
    if item == TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper {
        wave64_scan_provider_v1::validate_definition(tcx, def_id, &definition)?;
    }
    Ok(())
}

fn validate_reviewed_fe2o3_device_provider_definition_v1(
    item: TrustedDeviceItem,
    definition: &ReviewedProviderSemanticDefinitionV1,
) -> Result<(), String> {
    validate_safe_execution_provider_definition_v1(definition)?;
    if let Some(expected_definition_path) = exact_provider_compiler_definition_path_v1(item)
        && definition.canonical_definition_path != expected_definition_path
    {
        return Err(format!(
            "provider definition path is `{}`, expected `{expected_definition_path}` for `{}`",
            definition.canonical_definition_path,
            item.canonical_path()
        ));
    }
    definition.durable_semantic_identity(item.canonical_path())?;
    Ok(())
}
fn exact_provider_compiler_definition_path_v1(item: TrustedDeviceItem) -> Option<&'static str> {
    match item {
        TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper => {
            Some("fe2o3_device::collective::wave64_inclusive_scan")
        }
        TrustedDeviceItem::AmdGpuOrderedXorAddE32 => {
            Some("fe2o3_device::diagnostics::__amdgpu_ordered_xor_add_e32_v1")
        }
        TrustedDeviceItem::AmdGpuOrderedProgramE32 => {
            Some("fe2o3_device::diagnostics::__amdgpu_ordered_program_e32_v1")
        }
        TrustedDeviceItem::AmdGpuPhysicalEntryBeginGfx942 => {
            Some("fe2o3_device::physical_entry_v1::__amdgpu_physical_entry_begin_gfx942_v1")
        }
        TrustedDeviceItem::AmdGpuPhysicalGlobalCopyBeginGfx942 => Some(
            "fe2o3_device::physical_global_copy_v1::__amdgpu_physical_global_copy_begin_gfx942_v1",
        ),
        TrustedDeviceItem::AmdGpuPhysicalLdsExchangeBeginGfx942 => Some(
            "fe2o3_device::physical_lds_exchange_v1::__amdgpu_physical_lds_exchange_begin_gfx942_v1",
        ),
        TrustedDeviceItem::AmdGpuPhysicalEntryLabelGfx942 => {
            Some("fe2o3_device::physical_entry_v1::__amdgpu_physical_entry_label_gfx942_v1")
        }
        TrustedDeviceItem::AmdGpuPhysicalGlobalCopyLabelGfx942 => Some(
            "fe2o3_device::physical_global_copy_v1::__amdgpu_physical_global_copy_label_gfx942_v1",
        ),
        TrustedDeviceItem::AmdGpuPhysicalLdsExchangeLabelGfx942 => Some(
            "fe2o3_device::physical_lds_exchange_v1::__amdgpu_physical_lds_exchange_label_gfx942_v1",
        ),
        TrustedDeviceItem::AmdGpuPhysicalEntryStepGfx942 => {
            Some("fe2o3_device::physical_entry_v1::__amdgpu_physical_entry_step_gfx942_v1")
        }
        TrustedDeviceItem::AmdGpuPhysicalGlobalCopyStepGfx942 => Some(
            "fe2o3_device::physical_global_copy_v1::__amdgpu_physical_global_copy_step_gfx942_v1",
        ),
        TrustedDeviceItem::AmdGpuPhysicalLdsExchangeStepGfx942 => Some(
            "fe2o3_device::physical_lds_exchange_v1::__amdgpu_physical_lds_exchange_step_gfx942_v1",
        ),
        TrustedDeviceItem::AmdGpuCompleteBodyE32 => {
            Some("fe2o3_device::diagnostics::__amdgpu_complete_body_gfx942_v1")
        }
        TrustedDeviceItem::AmdGpuInline(operation) => Some(match operation {
            TrustedAmdGpuInlineOperation::VMovB32 => {
                "fe2o3_device::diagnostics::__amdgpu_v_mov_b32_v1"
            }
            TrustedAmdGpuInlineOperation::VAddU32 => {
                "fe2o3_device::diagnostics::__amdgpu_v_add_u32_v1"
            }
            TrustedAmdGpuInlineOperation::VSubU32 => {
                "fe2o3_device::diagnostics::__amdgpu_v_sub_u32_v1"
            }
            TrustedAmdGpuInlineOperation::VAndB32 => {
                "fe2o3_device::diagnostics::__amdgpu_v_and_b32_v1"
            }
            TrustedAmdGpuInlineOperation::VOrB32 => {
                "fe2o3_device::diagnostics::__amdgpu_v_or_b32_v1"
            }
            TrustedAmdGpuInlineOperation::VXorB32 => {
                "fe2o3_device::diagnostics::__amdgpu_v_xor_b32_v1"
            }
        }),
        TrustedDeviceItem::KernelError => Some("fe2o3_device::kernel_result::KernelError"),
        TrustedDeviceItem::DisjointSlice => Some("fe2o3_device::DisjointSlice"),
        TrustedDeviceItem::WriteOnlyDisjointSlice => Some("fe2o3_device::WriteOnlyDisjointSlice"),
        TrustedDeviceItem::StridedReadView2D => Some("fe2o3_device::views::StridedReadView2D"),
        TrustedDeviceItem::StridedReadView2DError => {
            Some("fe2o3_device::views::StridedReadView2DError")
        }
        TrustedDeviceItem::StridedReadView2DFromSharedSlice => {
            Some("fe2o3_device::views::{impl#4}::from_shared_slice")
        }
        TrustedDeviceItem::StridedReadView2DLoadOr => {
            Some("fe2o3_device::views::{impl#4}::load_or")
        }
        TrustedDeviceItem::ThreadIndex => Some("fe2o3_device::thread::ThreadIndex"),
        TrustedDeviceItem::DisjointIndex => Some("fe2o3_device::thread::DisjointIndex"),
        TrustedDeviceItem::ShiftedIndexSpace => Some("fe2o3_device::thread::Shifted"),
        TrustedDeviceItem::BlockedIndexSpace => Some("fe2o3_device::thread::Blocked"),
        TrustedDeviceItem::Tiled2DIndexSpace => Some("fe2o3_device::thread::Tiled2D"),
        TrustedDeviceItem::RowStriped2DIndexSpace => Some("fe2o3_device::thread::RowStriped2D"),
        TrustedDeviceItem::GridExclusiveIndexSpace => Some("fe2o3_device::thread::GridExclusive"),
        TrustedDeviceItem::GridLeader => Some("fe2o3_device::thread::GridLeader"),
        TrustedDeviceItem::DisjointBlock => Some("fe2o3_device::thread::DisjointBlock"),
        TrustedDeviceItem::DisjointTile2D => Some("fe2o3_device::thread::DisjointTile2D"),
        TrustedDeviceItem::DisjointRowStripe2D => Some("fe2o3_device::thread::DisjointRowStripe2D"),
        TrustedDeviceItem::ThreadIndex1d => Some("fe2o3_device::thread::index_1d"),
        TrustedDeviceItem::ThreadIndexGet => Some("fe2o3_device::thread::{impl#7}::get"),
        TrustedDeviceItem::ThreadIndexIntoDisjoint => {
            Some("fe2o3_device::thread::{impl#7}::into_disjoint")
        }
        TrustedDeviceItem::ThreadIndexCheckedShift => {
            Some("fe2o3_device::thread::{impl#7}::checked_shift")
        }
        TrustedDeviceItem::ThreadIndexCheckedBlock => {
            Some("fe2o3_device::thread::{impl#7}::checked_block")
        }
        TrustedDeviceItem::ThreadIndexCheckedTiled2D => {
            Some("fe2o3_device::thread::{impl#7}::checked_tiled_2d")
        }
        TrustedDeviceItem::ThreadIndexCheckedRowStriped2D => {
            Some("fe2o3_device::thread::{impl#7}::checked_row_striped_2d")
        }
        TrustedDeviceItem::DisjointIndexGet => Some("fe2o3_device::thread::{impl#9}::get"),
        TrustedDeviceItem::DisjointIndexCheckedShift => {
            Some("fe2o3_device::thread::{impl#9}::checked_shift")
        }
        TrustedDeviceItem::DisjointBlockComponentIndex => {
            Some("fe2o3_device::thread::{impl#11}::component_index")
        }
        TrustedDeviceItem::GridLeaderCurrent => Some("fe2o3_device::thread::grid_leader"),
        TrustedDeviceItem::DisjointSliceGetMut => Some("fe2o3_device::{impl#0}::get_mut"),
        TrustedDeviceItem::DisjointSliceGetDisjointMut => {
            Some("fe2o3_device::{impl#0}::get_disjoint_mut")
        }
        TrustedDeviceItem::DisjointSliceGetMutExclusive => {
            Some("fe2o3_device::{impl#1}::get_mut_exclusive")
        }
        TrustedDeviceItem::DisjointSliceGetBlockMut => {
            Some("fe2o3_device::{impl#2}::get_block_mut")
        }
        TrustedDeviceItem::DisjointSliceGetTiled2DMut => {
            Some("fe2o3_device::{impl#3}::get_tiled_2d_mut")
        }
        TrustedDeviceItem::DisjointSliceGetRowStriped2DMut => {
            Some("fe2o3_device::{impl#4}::get_row_striped_2d_mut")
        }
        TrustedDeviceItem::WriteOnlyDisjointSliceLen => Some("fe2o3_device::{impl#5}::len"),
        TrustedDeviceItem::WriteOnlyDisjointSliceWrite => Some("fe2o3_device::{impl#5}::write"),
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteDisjoint => {
            Some("fe2o3_device::{impl#5}::write_disjoint")
        }
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteExclusive => {
            Some("fe2o3_device::{impl#6}::write_exclusive")
        }
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteBlock => {
            Some("fe2o3_device::{impl#7}::write_block")
        }
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteTiled2D => {
            Some("fe2o3_device::{impl#8}::write_tiled_2d")
        }
        TrustedDeviceItem::WriteOnlyDisjointSliceWriteRowStriped2D => {
            Some("fe2o3_device::{impl#9}::write_row_striped_2d")
        }
        TrustedDeviceItem::DeviceGlobalMutPtrU32AsAtomic => {
            Some("fe2o3_device::atomic::{impl#0}::as_atomic")
        }
        TrustedDeviceItem::DeviceGlobalMutPtrI32AsAtomic => {
            Some("fe2o3_device::atomic::{impl#1}::as_atomic")
        }
        TrustedDeviceItem::DeviceGlobalMutPtrU64AsAtomic => {
            Some("fe2o3_device::atomic::{impl#2}::as_atomic")
        }
        TrustedDeviceItem::DeviceGlobalMutPtrI64AsAtomic => {
            Some("fe2o3_device::atomic::{impl#3}::as_atomic")
        }
        _ if safe_execution_provider_bound_item(item) => {
            Some(safe_execution_compiler_definition_path(item))
        }
        _ => None,
    }
}
fn validate_safe_execution_provider_definition_v1(
    definition: &ReviewedProviderSemanticDefinitionV1,
) -> Result<(), String> {
    definition.validate()?;
    if !REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1.contains(&definition.source_closure_identity) {
        return Err(format!(
            "safe execution provider source closure does not match a reviewed V1 materialization: {:02x?}",
            definition.source_closure_identity
        ));
    }
    Ok(())
}

fn safe_execution_compiler_definition_path(item: TrustedDeviceItem) -> &'static str {
    match item {
        TrustedDeviceItem::KernelContext => "fe2o3_device::context::KernelContext",
        TrustedDeviceItem::KernelContextIssue => {
            "fe2o3_device::context::{impl#4}::__compiler_issue"
        }
        TrustedDeviceItem::ExecutionWorkgroupCapability => {
            "fe2o3_device::execution::WorkgroupCapability"
        }
        TrustedDeviceItem::ExecutionWorkgroupCurrent => {
            "fe2o3_device::execution::{impl#2}::__compiler_workgroup_capability_current"
        }
        TrustedDeviceItem::ExecutionWithWorkgroup => {
            "fe2o3_device::execution::{impl#2}::with_workgroup"
        }
        TrustedDeviceItem::MaskedTile1D => "fe2o3_device::tile::MaskedTile1D",
        TrustedDeviceItem::LaneFragment1D => "fe2o3_device::tile::LaneFragment",
        TrustedDeviceItem::MaskedTile1DLoadMasked => "fe2o3_device::tile::{impl#0}::load_masked",
        TrustedDeviceItem::MaskedTile1DIntoFragment => {
            "fe2o3_device::tile::{impl#0}::into_fragment"
        }
        TrustedDeviceItem::LaneFragment1DIntoParts => "fe2o3_device::tile::{impl#1}::into_parts",
        TrustedDeviceItem::WorkgroupLdsScope => "fe2o3_device::lds::WorkgroupLdsScope",
        TrustedDeviceItem::WorkgroupLdsScopeCurrent => "fe2o3_device::lds::{impl#2}::current",
        TrustedDeviceItem::DynamicLdsExactCurrent => "fe2o3_device::lds::{impl#4}::exact_current",
        TrustedDeviceItem::DynamicLdsIntoCollectiveRawParts => {
            "fe2o3_device::lds::{impl#4}::into_collective_raw_parts"
        }
        TrustedDeviceItem::LdsUninitialized => "fe2o3_device::lds::LdsUninitialized",
        TrustedDeviceItem::WorkgroupPipeline => "fe2o3_device::lds::WorkgroupPipeline",
        TrustedDeviceItem::WorkgroupPipelineCurrent => "fe2o3_device::lds::{impl#10}::current",
        TrustedDeviceItem::WorkgroupPipelineStage => "fe2o3_device::lds::{impl#10}::stage",
        TrustedDeviceItem::WorkgroupPipelineWrite => "fe2o3_device::lds::{impl#10}::write",
        TrustedDeviceItem::WorkgroupPipelineCommit => "fe2o3_device::lds::{impl#10}::commit",
        TrustedDeviceItem::WorkgroupPipelineWait => "fe2o3_device::lds::{impl#10}::wait",
        TrustedDeviceItem::WorkgroupPipelineConsume => "fe2o3_device::lds::{impl#10}::consume",
        TrustedDeviceItem::WorkgroupPipelineRead => "fe2o3_device::lds::{impl#10}::read",
        TrustedDeviceItem::WorkgroupPipelineDiscard => "fe2o3_device::lds::{impl#10}::discard",
        TrustedDeviceItem::WorkgroupPipelineRelease => "fe2o3_device::lds::{impl#10}::release",
        TrustedDeviceItem::Invocation3D => "fe2o3_device::thread::Invocation3D",
        TrustedDeviceItem::Invocation3DCurrent => "fe2o3_device::thread::{impl#6}::current",
        TrustedDeviceItem::WorkgroupGroup => "fe2o3_device::group::Workgroup",
        TrustedDeviceItem::WorkgroupCollectiveScratch => {
            "fe2o3_device::collective::WorkgroupCollectiveScratch"
        }
        TrustedDeviceItem::Gfx942CollectivesContext => {
            "fe2o3_device::collective::Gfx942Collectives"
        }
        TrustedDeviceItem::WorkgroupCollectivesContext => {
            "fe2o3_device::collective::WorkgroupCollectives"
        }
        TrustedDeviceItem::WorkgroupCollectivesCurrent => {
            "fe2o3_device::collective::{impl#7}::current"
        }
        TrustedDeviceItem::WorkgroupReduceSum => {
            "fe2o3_device::collective::{impl#7}::reduce_sum_portable"
        }
        TrustedDeviceItem::WorkgroupInclusiveScanSum => {
            "fe2o3_device::collective::{impl#7}::inclusive_scan_sum"
        }
        TrustedDeviceItem::WorkgroupExclusiveScanSum => {
            "fe2o3_device::collective::{impl#7}::exclusive_scan_sum"
        }
        TrustedDeviceItem::Gfx942CollectivesCurrent => {
            "fe2o3_device::collective::{impl#0}::current"
        }
        TrustedDeviceItem::Gfx942SubgroupReduceSumF32 => {
            "fe2o3_device::collective::{impl#0}::subgroup_reduce_sum_f32"
        }
        TrustedDeviceItem::Gfx942SubgroupReduceMaxF32 => {
            "fe2o3_device::collective::{impl#0}::subgroup_reduce_max_f32"
        }
        TrustedDeviceItem::Gfx942StaticLdsU32x256 => {
            "fe2o3_device::collective::{impl#0}::static_lds_u32x256"
        }
        TrustedDeviceItem::Gfx942StaticLdsU32x256Type => {
            "fe2o3_device::collective::Gfx942StaticLdsU32x256"
        }
        TrustedDeviceItem::Gfx942Wave64ReduceActiveU32 => {
            "fe2o3_device::collective::{impl#0}::wave64_reduce_sum_active_u32"
        }
        TrustedDeviceItem::Gfx942Workgroup256ReduceActiveU32 => {
            "fe2o3_device::collective::{impl#0}::workgroup256_reduce_sum_active_u32"
        }
        TrustedDeviceItem::Gfx942Wave64ReduceSum => {
            "fe2o3_device::collective::{impl#5}::reduce_sum"
        }
        TrustedDeviceItem::Gfx942Wave64InclusiveScanSum => {
            "fe2o3_device::collective::{impl#5}::inclusive_scan_sum"
        }
        TrustedDeviceItem::Gfx942Wave64ExclusiveScanSum => {
            "fe2o3_device::collective::{impl#5}::exclusive_scan_sum"
        }
        TrustedDeviceItem::Gfx942WorkgroupReduceSum => {
            "fe2o3_device::collective::{impl#6}::reduce_sum"
        }
        TrustedDeviceItem::Gfx942WorkgroupInclusiveScanSum => {
            "fe2o3_device::collective::{impl#6}::inclusive_scan_sum"
        }
        TrustedDeviceItem::Gfx942WorkgroupExclusiveScanSum => {
            "fe2o3_device::collective::{impl#6}::exclusive_scan_sum"
        }
        TrustedDeviceItem::WaveLane => "fe2o3_device::wave::WaveLane",
        TrustedDeviceItem::Wave64 => "fe2o3_device::wave::Wave64",
        TrustedDeviceItem::WaveLaneCurrent => "fe2o3_device::wave::{impl#4}::current",
        TrustedDeviceItem::LdsTile16x16WriteMfmaBf16 => {
            "fe2o3_device::tensor::{impl#16}::write_mfma_fragment"
        }
        TrustedDeviceItem::LdsTile16x16ReadMfmaBf16 => {
            "fe2o3_device::tensor::{impl#17}::read_mfma_fragment"
        }
        TrustedDeviceItem::WorkgroupSyncthreads => "fe2o3_device::sync::syncthreads",
        TrustedDeviceItem::DeviceMatrix => "fe2o3_device::tensor::DeviceMatrix",
        TrustedDeviceItem::DeviceMatrixCurrent => "fe2o3_device::tensor::{impl#9}::current",
        TrustedDeviceItem::Bf16MfmaProfile => "fe2o3_device::tensor::Bf16F32M16N16K16",
        TrustedDeviceItem::MfmaOperandA => "fe2o3_device::tensor::MfmaOperandA",
        TrustedDeviceItem::MfmaOperandB => "fe2o3_device::tensor::MfmaOperandB",
        TrustedDeviceItem::MfmaRegisterTile16x16 => "fe2o3_device::tensor::MfmaRegisterTile16x16",
        TrustedDeviceItem::MfmaLdsXor4Storage => "fe2o3_device::tensor::MfmaLdsXor4",
        TrustedDeviceItem::MfmaAccumulatorRowMajor => {
            "fe2o3_device::tensor::MfmaAccumulatorRowMajor"
        }
        TrustedDeviceItem::Bf16MfmaFragment => "fe2o3_device::tensor::Bf16MfmaFragment",
        TrustedDeviceItem::F32AccumulatorFragment => "fe2o3_device::tensor::F32AccumulatorFragment",
        TrustedDeviceItem::F32AccumulatorFragmentZero => "fe2o3_device::tensor::{impl#4}::zero",
        TrustedDeviceItem::F32AccumulatorFragmentIntoValues => {
            "fe2o3_device::tensor::{impl#5}::into_values"
        }
        TrustedDeviceItem::Bf16MfmaMatrixView => "fe2o3_device::tensor::Bf16MfmaMatrix",
        TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView => {
            "fe2o3_device::tensor::column_major::Bf16MfmaBColumnMajorMatrix"
        }
        TrustedDeviceItem::Bf16MfmaBColumnMajor => {
            "fe2o3_device::tensor::column_major::{impl#0}::column_major"
        }
        TrustedDeviceItem::Bf16MfmaBColumnMajorLoadZeroFilledV1 => {
            "fe2o3_device::tensor::column_major::{impl#0}::load_k16n16"
        }
        TrustedDeviceItem::Bf16MfmaMatrixViewError => "fe2o3_device::tensor::Bf16MatrixViewError",
        TrustedDeviceItem::Bf16MfmaMatrixARowMajor => "fe2o3_device::tensor::{impl#7}::row_major",
        TrustedDeviceItem::Bf16MfmaMatrixBRowMajor => "fe2o3_device::tensor::{impl#8}::row_major",
        TrustedDeviceItem::Bf16MfmaMatrixALoadZeroFilledV2 => {
            "fe2o3_device::tensor::{impl#7}::load_m16k16"
        }
        TrustedDeviceItem::Bf16MfmaMatrixBLoadZeroFilledV2 => {
            "fe2o3_device::tensor::{impl#8}::load_k16n16"
        }
        TrustedDeviceItem::DeviceMatrixMultiplyAccumulate => {
            "fe2o3_device::tensor::{impl#9}::multiply_accumulate"
        }
        TrustedDeviceItem::Gfx950Fp4E2M1Format => "fe2o3_device::gfx950::Gfx950Fp4E2M1",
        TrustedDeviceItem::Gfx950Fp8E4M3Format => "fe2o3_device::gfx950::Gfx950Fp8E4M3",
        TrustedDeviceItem::Gfx950MfmaOperandA => "fe2o3_device::gfx950::Gfx950MfmaOperandA",
        TrustedDeviceItem::Gfx950MfmaOperandB => "fe2o3_device::gfx950::Gfx950MfmaOperandB",
        TrustedDeviceItem::Gfx950MfmaFragment => "fe2o3_device::gfx950::Gfx950MfmaFragment",
        TrustedDeviceItem::Gfx950F32AccumulatorFragment => {
            "fe2o3_device::gfx950::Gfx950F32AccumulatorFragment"
        }
        TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentZero => {
            "fe2o3_device::gfx950::{impl#8}::zero"
        }
        TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentIntoValues => {
            "fe2o3_device::gfx950::{impl#8}::into_values"
        }
        TrustedDeviceItem::Gfx950F32AccumulatorFragmentZero => {
            "fe2o3_device::gfx950::{impl#9}::zero"
        }
        TrustedDeviceItem::Gfx950F32AccumulatorFragmentIntoValues => {
            "fe2o3_device::gfx950::{impl#9}::into_values"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixViewError => {
            "fe2o3_device::gfx950::Gfx950MatrixViewError"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixAView => "fe2o3_device::gfx950::Gfx950MfmaAMatrix",
        TrustedDeviceItem::Gfx950MfmaMatrixBView => "fe2o3_device::gfx950::Gfx950MfmaBMatrix",
        TrustedDeviceItem::Gfx950MfmaMatrixAFp4RowMajor => {
            "fe2o3_device::gfx950::{impl#13}::row_major"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixBFp4RowMajor => {
            "fe2o3_device::gfx950::{impl#16}::row_major"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixARowMajor => {
            "fe2o3_device::gfx950::{impl#14}::row_major"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixBRowMajor => {
            "fe2o3_device::gfx950::{impl#17}::row_major"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixAFp4LoadM16K128 => {
            "fe2o3_device::gfx950::{impl#13}::load_m16k128"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixBFp4LoadK128N16 => {
            "fe2o3_device::gfx950::{impl#16}::load_k128n16"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixAFp8LoadM16K128 => {
            "fe2o3_device::gfx950::{impl#14}::load_m16k128"
        }
        TrustedDeviceItem::Gfx950MfmaMatrixBFp8LoadK128N16 => {
            "fe2o3_device::gfx950::{impl#17}::load_k128n16"
        }
        TrustedDeviceItem::Gfx950Matrix => "fe2o3_device::gfx950::Gfx950Matrix",
        TrustedDeviceItem::Gfx950MatrixCurrent => "fe2o3_device::gfx950::{impl#18}::current",
        TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4 => {
            "fe2o3_device::gfx950::{impl#18}::multiply_accumulate_fp4"
        }
        TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4Fp8 => {
            "fe2o3_device::gfx950::{impl#18}::multiply_accumulate_fp4_fp8"
        }
        TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp8 => {
            "fe2o3_device::gfx950::{impl#18}::multiply_accumulate_fp8"
        }
        TrustedDeviceItem::Gfx950SubgroupContext => "fe2o3_device::gfx950::Gfx950Subgroup",
        TrustedDeviceItem::Gfx950SubgroupCurrent => "fe2o3_device::gfx950::{impl#19}::current",
        TrustedDeviceItem::Gfx950SubgroupReduceMaxF32 => {
            "fe2o3_device::gfx950::{impl#19}::reduce_max_f32"
        }
        TrustedDeviceItem::Gfx950SubgroupReduceSumF32 => {
            "fe2o3_device::gfx950::{impl#19}::reduce_sum_f32"
        }
        TrustedDeviceItem::Gfx950SubgroupBroadcastF32 => {
            "fe2o3_device::gfx950::{impl#19}::broadcast_f32"
        }
        TrustedDeviceItem::Gfx950LdsTransposeTile => "fe2o3_device::gfx950::Gfx950LdsTransposeTile",
        TrustedDeviceItem::Gfx950LdsTransposeTileCurrent => {
            "fe2o3_device::gfx950::{impl#23}::current"
        }
        TrustedDeviceItem::Gfx950LdsTransposeStageB4 => {
            "fe2o3_device::gfx950::{impl#24}::stage_k_transposed"
        }
        TrustedDeviceItem::Gfx950LdsTransposeStageB8 => {
            "fe2o3_device::gfx950::{impl#25}::stage_k_transposed"
        }
        TrustedDeviceItem::Gfx950LdsTransposePublish => "fe2o3_device::gfx950::{impl#26}::publish",
        TrustedDeviceItem::Gfx950LdsTransposeReadB4 => {
            "fe2o3_device::gfx950::{impl#27}::read_mfma_fragment"
        }
        TrustedDeviceItem::Gfx950LdsTransposeReadB8 => {
            "fe2o3_device::gfx950::{impl#28}::read_mfma_fragment"
        }
        _ => item.canonical_path(),
    }
}

const fn safe_execution_provider_bound_item(item: TrustedDeviceItem) -> bool {
    matches!(
        item,
        TrustedDeviceItem::KernelContext
            | TrustedDeviceItem::KernelContextIssue
            | TrustedDeviceItem::ExecutionWorkgroupCapability
            | TrustedDeviceItem::ExecutionWorkgroupCurrent
            | TrustedDeviceItem::ExecutionWithWorkgroup
            | TrustedDeviceItem::MaskedTile1D
            | TrustedDeviceItem::LaneFragment1D
            | TrustedDeviceItem::MaskedTile1DLoadMasked
            | TrustedDeviceItem::MaskedTile1DIntoFragment
            | TrustedDeviceItem::LaneFragment1DIntoParts
            | TrustedDeviceItem::WorkgroupLdsScope
            | TrustedDeviceItem::WorkgroupLdsScopeCurrent
            | TrustedDeviceItem::DynamicLdsExactCurrent
            | TrustedDeviceItem::DynamicLdsIntoCollectiveRawParts
            | TrustedDeviceItem::LdsUninitialized
            | TrustedDeviceItem::WorkgroupPipeline
            | TrustedDeviceItem::WorkgroupPipelineCurrent
            | TrustedDeviceItem::WorkgroupPipelineStage
            | TrustedDeviceItem::WorkgroupPipelineWrite
            | TrustedDeviceItem::WorkgroupPipelineCommit
            | TrustedDeviceItem::WorkgroupPipelineWait
            | TrustedDeviceItem::WorkgroupPipelineConsume
            | TrustedDeviceItem::WorkgroupPipelineRead
            | TrustedDeviceItem::WorkgroupPipelineDiscard
            | TrustedDeviceItem::WorkgroupPipelineRelease
            | TrustedDeviceItem::Invocation3D
            | TrustedDeviceItem::Invocation3DCurrent
            | TrustedDeviceItem::ThreadIndexX
            | TrustedDeviceItem::ThreadIndexY
            | TrustedDeviceItem::ThreadIndexZ
            | TrustedDeviceItem::WorkgroupIndexX
            | TrustedDeviceItem::WorkgroupIndexY
            | TrustedDeviceItem::WorkgroupIndexZ
            | TrustedDeviceItem::WorkgroupDimensionX
            | TrustedDeviceItem::WorkgroupDimensionY
            | TrustedDeviceItem::WorkgroupDimensionZ
            | TrustedDeviceItem::GridDimensionX
            | TrustedDeviceItem::GridDimensionY
            | TrustedDeviceItem::GridDimensionZ
            | TrustedDeviceItem::WorkgroupGroup
            | TrustedDeviceItem::WorkgroupCollectiveScratch
            | TrustedDeviceItem::WorkgroupCollectivesContext
            | TrustedDeviceItem::WorkgroupCollectivesCurrent
            | TrustedDeviceItem::WorkgroupReduceSum
            | TrustedDeviceItem::WorkgroupInclusiveScanSum
            | TrustedDeviceItem::WorkgroupExclusiveScanSum
            | TrustedDeviceItem::Gfx942CollectivesContext
            | TrustedDeviceItem::Gfx942CollectivesCurrent
            | TrustedDeviceItem::Gfx942SubgroupReduceSumF32
            | TrustedDeviceItem::Gfx942SubgroupReduceMaxF32
            | TrustedDeviceItem::Gfx942StaticLdsU32x256
            | TrustedDeviceItem::Gfx942StaticLdsU32x256Type
            | TrustedDeviceItem::Gfx942Wave64ReduceActiveU32
            | TrustedDeviceItem::Gfx942Workgroup256ReduceActiveU32
            | TrustedDeviceItem::Gfx942Wave64ReduceSum
            | TrustedDeviceItem::Gfx942Wave64InclusiveScanSum
            | TrustedDeviceItem::Gfx942Wave64ExclusiveScanSum
            | TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper
            | TrustedDeviceItem::Gfx942WorkgroupReduceSum
            | TrustedDeviceItem::Gfx942WorkgroupInclusiveScanSum
            | TrustedDeviceItem::Gfx942WorkgroupExclusiveScanSum
            | TrustedDeviceItem::WaveLane
            | TrustedDeviceItem::Wave64
            | TrustedDeviceItem::WaveLaneCurrent
            | TrustedDeviceItem::LdsTile16x16WriteMfmaBf16
            | TrustedDeviceItem::LdsTile16x16ReadMfmaBf16
            | TrustedDeviceItem::WorkgroupSyncthreads
            | TrustedDeviceItem::DeviceMatrix
            | TrustedDeviceItem::DeviceMatrixCurrent
            | TrustedDeviceItem::Bf16MfmaProfile
            | TrustedDeviceItem::MfmaOperandA
            | TrustedDeviceItem::MfmaOperandB
            | TrustedDeviceItem::MfmaRegisterTile16x16
            | TrustedDeviceItem::MfmaLdsXor4Storage
            | TrustedDeviceItem::MfmaAccumulatorRowMajor
            | TrustedDeviceItem::Bf16MfmaFragment
            | TrustedDeviceItem::F32AccumulatorFragment
            | TrustedDeviceItem::F32AccumulatorFragmentZero
            | TrustedDeviceItem::F32AccumulatorFragmentIntoValues
            | TrustedDeviceItem::Bf16MfmaMatrixView
            | TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView
            | TrustedDeviceItem::Bf16MfmaBColumnMajor
            | TrustedDeviceItem::Bf16MfmaBColumnMajorLoadZeroFilledV1
            | TrustedDeviceItem::Bf16MfmaMatrixViewError
            | TrustedDeviceItem::Bf16MfmaMatrixARowMajor
            | TrustedDeviceItem::Bf16MfmaMatrixBRowMajor
            | TrustedDeviceItem::Bf16MfmaMatrixALoadZeroFilledV2
            | TrustedDeviceItem::Bf16MfmaMatrixBLoadZeroFilledV2
            | TrustedDeviceItem::DeviceMatrixMultiplyAccumulate
            | TrustedDeviceItem::Gfx950Fp4E2M1Format
            | TrustedDeviceItem::Gfx950Fp8E4M3Format
            | TrustedDeviceItem::Gfx950MfmaOperandA
            | TrustedDeviceItem::Gfx950MfmaOperandB
            | TrustedDeviceItem::Gfx950MfmaFragment
            | TrustedDeviceItem::Gfx950F32AccumulatorFragment
            | TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentZero
            | TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentIntoValues
            | TrustedDeviceItem::Gfx950F32AccumulatorFragmentZero
            | TrustedDeviceItem::Gfx950F32AccumulatorFragmentIntoValues
            | TrustedDeviceItem::Gfx950MfmaMatrixViewError
            | TrustedDeviceItem::Gfx950MfmaMatrixAView
            | TrustedDeviceItem::Gfx950MfmaMatrixBView
            | TrustedDeviceItem::Gfx950MfmaMatrixAFp4RowMajor
            | TrustedDeviceItem::Gfx950MfmaMatrixBFp4RowMajor
            | TrustedDeviceItem::Gfx950MfmaMatrixARowMajor
            | TrustedDeviceItem::Gfx950MfmaMatrixBRowMajor
            | TrustedDeviceItem::Gfx950MfmaMatrixAFp8LoadM16K128
            | TrustedDeviceItem::Gfx950MfmaMatrixBFp8LoadK128N16
            | TrustedDeviceItem::Gfx950MfmaMatrixAFp4LoadM16K128
            | TrustedDeviceItem::Gfx950MfmaMatrixBFp4LoadK128N16
            | TrustedDeviceItem::Gfx950Matrix
            | TrustedDeviceItem::Gfx950MatrixCurrent
            | TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4
            | TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4Fp8
            | TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp8
            | TrustedDeviceItem::Gfx950SubgroupContext
            | TrustedDeviceItem::Gfx950SubgroupCurrent
            | TrustedDeviceItem::Gfx950SubgroupReduceMaxF32
            | TrustedDeviceItem::Gfx950SubgroupReduceSumF32
            | TrustedDeviceItem::Gfx950SubgroupBroadcastF32
            | TrustedDeviceItem::Gfx950LdsTransposeTile
            | TrustedDeviceItem::Gfx950LdsTransposeTileCurrent
            | TrustedDeviceItem::Gfx950LdsTransposeStageB4
            | TrustedDeviceItem::Gfx950LdsTransposeStageB8
            | TrustedDeviceItem::Gfx950LdsTransposePublish
            | TrustedDeviceItem::Gfx950LdsTransposeReadB4
            | TrustedDeviceItem::Gfx950LdsTransposeReadB8
            | TrustedDeviceItem::Tiled2DIndexSpace
            | TrustedDeviceItem::RowStriped2DIndexSpace
            | TrustedDeviceItem::DisjointTile2D
            | TrustedDeviceItem::DisjointRowStripe2D
            | TrustedDeviceItem::ThreadIndexCheckedTiled2D
            | TrustedDeviceItem::ThreadIndexCheckedRowStriped2D
            | TrustedDeviceItem::DisjointSliceGetTiled2DMut
            | TrustedDeviceItem::DisjointSliceGetRowStriped2DMut
            | TrustedDeviceItem::StridedReadView2D
            | TrustedDeviceItem::StridedReadView2DError
            | TrustedDeviceItem::StridedReadView2DFromSharedSlice
            | TrustedDeviceItem::StridedReadView2DLoadOr
    )
}

fn named_external_provider(
    tcx: TyCtxt<'_>,
    crate_num: rustc_hir::def_id::CrateNum,
) -> Result<String, String> {
    named_external_provider_as(tcx, crate_num, "fe2o3_device")
}

fn named_external_provider_as(
    tcx: TyCtxt<'_>,
    crate_num: rustc_hir::def_id::CrateNum,
    expected_crate_name: &str,
) -> Result<String, String> {
    if crate_num == LOCAL_CRATE {
        return Err("provider is the local compilation crate".to_owned());
    }
    let crate_name = tcx.crate_name(crate_num).as_str().to_owned();
    if crate_name != expected_crate_name {
        return Err(format!("provider crate name is `{crate_name}`"));
    }
    Ok(crate_name)
}

pub(crate) fn compiler_provider_observation_v1(
    tcx: TyCtxt<'_>,
    crate_num: rustc_hir::def_id::CrateNum,
) -> CompilerProviderObservationV1 {
    CompilerProviderObservationV1 {
        crate_name: tcx.crate_name(crate_num).to_string(),
        stable_crate_id: tcx.stable_crate_id(crate_num).as_u64(),
        crate_hash_observation: tcx.crate_hash(crate_num).as_u128().to_le_bytes(),
    }
}

pub(crate) fn reviewed_provider_semantic_definition_v1(
    tcx: TyCtxt<'_>,
    provider_definition: DefId,
) -> Result<ReviewedProviderSemanticDefinitionV1, String> {
    reviewed_provider_semantic_definition_from_source_v1(
        tcx,
        provider_definition,
        WORKGROUP_SYNC_PROVIDER_SOURCE_IDENTITY_DOMAIN_V1,
        WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        &REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1,
        &WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE,
    )
}

/// Authenticates a safe-signature external helper from the exact reviewed
/// `fe2o3-device` source closure while leaving its MIR available for traversal.
pub(crate) fn authenticate_reviewed_safe_external_helper_v1(
    tcx: TyCtxt<'_>,
    provider_definition: DefId,
) -> Result<bool, String> {
    if provider_definition.krate == LOCAL_CRATE
        || tcx.crate_name(provider_definition.krate).as_str() != "fe2o3_device"
    {
        return Ok(false);
    }
    let definition = reviewed_provider_semantic_definition_v1(tcx, provider_definition)?;
    validate_safe_execution_provider_definition_v1(&definition)?;
    Ok(true)
}

/// Recognizes the pinned core scalar bit-conversion methods whose complete
/// executable semantics are retained as a semantic-MIR transmute.
pub(crate) fn authenticate_reviewed_safe_core_scalar_bitcast_helper_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    let Some(core_lang_item) = tcx.lang_items().sized_trait() else {
        return false;
    };
    if !matches!(instance.def, InstanceKind::Item(_))
        || instance.def_id().krate != core_lang_item.krate
        || tcx.crate_name(core_lang_item.krate).as_str() != "core"
        || !instance.args.is_empty()
        || !tcx.is_mir_available(instance.def_id())
    {
        return false;
    }
    let signature = tcx.instantiate_bound_regions_with_erased(
        tcx.fn_sig(instance.def_id())
            .instantiate(tcx, instance.args),
    );
    if signature.safety != Safety::Safe || signature.abi != ExternAbi::Rust || signature.c_variadic
    {
        return false;
    }
    let [input] = signature.inputs() else {
        return false;
    };
    let input = input.kind();
    let output = signature.output().kind();
    matches!(
        (tcx.def_path_str(instance.def_id()).as_str(), input, output),
        (
            "core::f32::<impl f32>::from_bits",
            TyKind::Uint(UintTy::U32),
            TyKind::Float(FloatTy::F32),
        ) | (
            "core::f32::<impl f32>::to_bits",
            TyKind::Float(FloatTy::F32),
            TyKind::Uint(UintTy::U32),
        ) | (
            "core::f64::<impl f64>::from_bits",
            TyKind::Uint(UintTy::U64),
            TyKind::Float(FloatTy::F64),
        ) | (
            "core::f64::<impl f64>::to_bits",
            TyKind::Float(FloatTy::F64),
            TyKind::Uint(UintTy::U64),
        )
    )
}

/// Recognizes the pinned safe `f32::abs` wrapper while retaining its MIR so
/// the nested exact `core::intrinsics::fabs::<f32>` call is still classified.
pub(crate) fn authenticate_reviewed_safe_core_fabs_f32_helper_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    let Some(core_lang_item) = tcx.lang_items().sized_trait() else {
        return false;
    };
    if !matches!(instance.def, InstanceKind::Item(_))
        || instance.def_id().krate != core_lang_item.krate
        || tcx.crate_name(core_lang_item.krate).as_str() != "core"
        || !instance.args.is_empty()
        || !tcx.is_mir_available(instance.def_id())
        || tcx.def_path_str(instance.def_id()) != "core::f32::<impl f32>::abs"
    {
        return false;
    }
    let signature = tcx.instantiate_bound_regions_with_erased(
        tcx.fn_sig(instance.def_id())
            .instantiate(tcx, instance.args),
    );
    signature.safety == Safety::Safe
        && signature.abi == ExternAbi::Rust
        && !signature.c_variadic
        && matches!(signature.inputs(), [input] if matches!(input.kind(), TyKind::Float(FloatTy::F32)))
        && matches!(signature.output().kind(), TyKind::Float(FloatTy::F32))
}

#[path = "trusted_device_items/core_saturating_integer_v1.rs"]
mod core_saturating_integer_v1;

#[path = "trusted_device_items/core_slice_metadata_v1.rs"]
mod core_slice_metadata_v1;
pub(crate) use core_slice_metadata_v1::authenticate_v1 as authenticate_reviewed_safe_core_slice_metadata_helper_v1;

pub(crate) fn authenticate_reviewed_safe_core_saturating_integer_helper_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    core_saturating_integer_v1::authenticate_v1(tcx, instance)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewedSafeCoreWrappingIntegerContractV1<'a> {
    item_instance: bool,
    core_identity: bool,
    generic_arguments: usize,
    mir_available: bool,
    canonical_path: &'a str,
    safe_signature: bool,
    rust_abi: bool,
    variadic: bool,
    input_count: usize,
    first_integer: Option<&'a str>,
    second_integer: Option<&'a str>,
    result_integer: Option<&'a str>,
}

fn authenticate_reviewed_safe_core_wrapping_integer_contract_v1(
    contract: ReviewedSafeCoreWrappingIntegerContractV1<'_>,
) -> bool {
    let Some(integer) = contract.first_integer else {
        return false;
    };
    if !matches!(
        integer,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
    ) {
        return false;
    }
    let Some((owner, method)) = contract.canonical_path.rsplit_once("::") else {
        return false;
    };
    let second_integer = match method {
        "wrapping_add" | "wrapping_sub" | "wrapping_mul" => integer,
        "wrapping_shl" | "wrapping_shr" => "u32",
        _ => return false,
    };
    contract.item_instance
        && contract.core_identity
        && contract.generic_arguments == 0
        && contract.mir_available
        && owner == format!("core::num::<impl {integer}>")
        && contract.safe_signature
        && contract.rust_abi
        && !contract.variadic
        && contract.input_count == 2
        && contract.second_integer == Some(second_integer)
        && contract.result_integer == Some(integer)
}

/// Admits only the safe primitive wrapping helpers in the trusted pinned core.
///
/// This uses the same core lang-item identity boundary as scalar bitcasts and
/// abs, not cryptographic authentication of a replacement sysroot. It only
/// discharges the external-HIR source-safety check: real MIR, nested calls and
/// arithmetic remain subject to ordinary collection, import and verification.
/// Shift counts use the actual primitive method's `u32` signature. This does
/// not authenticate unsafe unchecked helpers or prove their range precondition.
pub(crate) fn authenticate_reviewed_safe_core_wrapping_integer_helper_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    if !matches!(instance.def, InstanceKind::Item(_))
        || !crate::production_rustc_intrinsic_v1::is_reviewed_core_function_v1(tcx, instance)
        || !instance.args.is_empty()
        || !tcx.is_mir_available(instance.def_id())
    {
        return false;
    }
    let signature = tcx.instantiate_bound_regions_with_erased(
        tcx.fn_sig(instance.def_id())
            .instantiate(tcx, instance.args),
    );
    let integer_name = |ty: Ty<'tcx>| match ty.kind() {
        TyKind::Int(integer) => Some(integer.name_str()),
        TyKind::Uint(integer) => Some(integer.name_str()),
        _ => None,
    };
    authenticate_reviewed_safe_core_wrapping_integer_contract_v1(
        ReviewedSafeCoreWrappingIntegerContractV1 {
            item_instance: matches!(instance.def, InstanceKind::Item(_)),
            core_identity: crate::production_rustc_intrinsic_v1::is_reviewed_core_function_v1(
                tcx, instance,
            ),
            generic_arguments: instance.args.len(),
            mir_available: tcx.is_mir_available(instance.def_id()),
            canonical_path: &tcx.def_path_str(instance.def_id()),
            safe_signature: signature.safety == Safety::Safe,
            rust_abi: signature.abi == ExternAbi::Rust,
            variadic: signature.c_variadic,
            input_count: signature.inputs().len(),
            first_integer: signature.inputs().first().copied().and_then(integer_name),
            second_integer: signature.inputs().get(1).copied().and_then(integer_name),
            result_integer: integer_name(signature.output()),
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewedSafeCoreF32IsFiniteContractV1<'a> {
    item_instance: bool,
    core_lang_crate: bool,
    crate_name: &'a str,
    generic_arguments: usize,
    mir_available: bool,
    canonical_path: &'a str,
    safe_signature: bool,
    rust_abi: bool,
    variadic: bool,
    input_f32: bool,
    output_bool: bool,
    exact_reviewed_body: bool,
}

fn authenticate_reviewed_safe_core_f32_is_finite_contract_v1(
    contract: ReviewedSafeCoreF32IsFiniteContractV1<'_>,
) -> bool {
    contract.item_instance
        && contract.core_lang_crate
        && contract.crate_name == "core"
        && contract.generic_arguments == 0
        && contract.mir_available
        && contract.canonical_path == "core::f32::<impl f32>::is_finite"
        && contract.safe_signature
        && contract.rust_abi
        && !contract.variadic
        && contract.input_f32
        && contract.output_bool
        && contract.exact_reviewed_body
}

/// Recognizes only the reviewed pinned-core `f32::is_finite` implementation.
/// Its exact two-block MIR remains in the call graph, so the nested `f32::abs`
/// and `core::intrinsics::fabs::<f32>` chain is still independently checked.
pub(crate) fn authenticate_reviewed_safe_core_f32_is_finite_helper_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    let Some(core_lang_item) = tcx.lang_items().sized_trait() else {
        return false;
    };
    let crate_name = tcx.crate_name(instance.def_id().krate);
    let canonical_path = tcx.def_path_str(instance.def_id());
    // Other external items, including real closures, reach this recognizer.
    // Authenticate this exact function definition before asking rustc for fn_sig;
    // closures carry their signature in their closure args instead of fn_sig.
    if !matches!(instance.def, InstanceKind::Item(_))
        || !matches!(
            tcx.def_kind(instance.def_id()),
            rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn
        )
        || instance.def_id().krate != core_lang_item.krate
        || crate_name.as_str() != "core"
        || !instance.args.is_empty()
        || canonical_path != "core::f32::<impl f32>::is_finite"
    {
        return false;
    }
    let mir_available = tcx.is_mir_available(instance.def_id());
    let signature = tcx.instantiate_bound_regions_with_erased(
        tcx.fn_sig(instance.def_id())
            .instantiate(tcx, instance.args),
    );
    let input_f32 = matches!(signature.inputs(), [input] if matches!(input.kind(), TyKind::Float(FloatTy::F32)));
    let output_bool = matches!(signature.output().kind(), TyKind::Bool);
    let exact_reviewed_body =
        mir_available && reviewed_safe_core_f32_is_finite_body_v1(tcx, instance);
    authenticate_reviewed_safe_core_f32_is_finite_contract_v1(
        ReviewedSafeCoreF32IsFiniteContractV1 {
            item_instance: matches!(instance.def, InstanceKind::Item(_)),
            core_lang_crate: instance.def_id().krate == core_lang_item.krate,
            crate_name: crate_name.as_str(),
            generic_arguments: instance.args.len(),
            mir_available,
            canonical_path: &canonical_path,
            safe_signature: signature.safety == Safety::Safe,
            rust_abi: signature.abi == ExternAbi::Rust,
            variadic: signature.c_variadic,
            input_f32,
            output_bool,
            exact_reviewed_body,
        },
    )
}

fn reviewed_safe_core_f32_is_finite_body_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> bool {
    let body = tcx.instance_mir(instance.def);
    reviewed_safe_core_f32_is_finite_route_body_v1(tcx, body)
        || reviewed_safe_core_f32_is_finite_optimized_body_v1(tcx, body)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewedSafeCoreF32IsFiniteRouteBodyContractV1 {
    argument_count: usize,
    local_count: usize,
    block_count: usize,
    source_scope_count: usize,
    exact_local_types: bool,
    root_scope_not_inlined: bool,
    entry_has_no_statements: bool,
    exact_abs_callee: bool,
    copies_input: bool,
    writes_absolute_temporary: bool,
    comparison_target: bool,
    unwind_unreachable: bool,
    exact_less_than_infinity: bool,
    returns_result: bool,
}

fn authenticate_reviewed_safe_core_f32_is_finite_route_body_contract_v1(
    contract: ReviewedSafeCoreF32IsFiniteRouteBodyContractV1,
) -> bool {
    contract.argument_count == 1
        && contract.local_count == 3
        && contract.block_count == 2
        && contract.source_scope_count == 1
        && contract.exact_local_types
        && contract.root_scope_not_inlined
        && contract.entry_has_no_statements
        && contract.exact_abs_callee
        && contract.copies_input
        && contract.writes_absolute_temporary
        && contract.comparison_target
        && contract.unwind_unreachable
        && contract.exact_less_than_infinity
        && contract.returns_result
}

// Production and engineering extraction disable MIR inlining. Bind the exact
// retained wrapper before treating its unavailable cross-crate HIR as safe.
fn reviewed_safe_core_f32_is_finite_route_body_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> bool {
    let mut locals = body.local_decls.iter();
    let exact_local_types = matches!(locals.next(), Some(result) if matches!(result.ty.kind(), TyKind::Bool))
        && matches!(locals.next(), Some(input) if matches!(input.ty.kind(), TyKind::Float(FloatTy::F32)))
        && matches!(locals.next(), Some(absolute) if matches!(absolute.ty.kind(), TyKind::Float(FloatTy::F32)))
        && locals.next().is_none();
    let mut scopes = body.source_scopes.iter();
    let root_scope_not_inlined =
        matches!(scopes.next(), Some(scope) if scope.inlined.is_none()) && scopes.next().is_none();

    let mut blocks = body.basic_blocks.iter();
    let Some(entry) = blocks.next() else {
        return false;
    };
    let Some(comparison) = blocks.next() else {
        return false;
    };
    if blocks.next().is_some() {
        return false;
    }
    let Some(entry_terminator) = &entry.terminator else {
        return false;
    };
    let TerminatorKind::Call {
        func,
        args,
        destination,
        target,
        unwind,
        ..
    } = &entry_terminator.kind
    else {
        return false;
    };
    let Some(target) = target else {
        return false;
    };
    let Operand::Constant(function) = func else {
        return false;
    };
    let TyKind::FnDef(def_id, arguments) = function.const_.ty().kind() else {
        return false;
    };
    let Ok(Some(abs)) =
        Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), *def_id, arguments)
    else {
        return false;
    };
    let exact_abs_callee = authenticate_reviewed_safe_core_fabs_f32_helper_v1(tcx, abs);

    let [statement] = comparison.statements.as_slice() else {
        return false;
    };
    let StatementKind::Assign(assignment) = &statement.kind else {
        return false;
    };
    let (result, rvalue) = &**assignment;
    let Rvalue::BinaryOp(BinOp::Lt, operands) = rvalue else {
        return false;
    };
    let (absolute, infinity) = &**operands;
    let Operand::Constant(infinity) = infinity else {
        return false;
    };
    let exact_less_than_infinity = result.local.as_usize() == 0
        && result.projection.is_empty()
        && matches!(absolute, Operand::Move(place) if place.local.as_usize() == 2 && place.projection.is_empty())
        && matches!(infinity.const_.ty().kind(), TyKind::Float(FloatTy::F32))
        && infinity
            .const_
            .try_eval_bits(tcx, TypingEnv::fully_monomorphized())
            == Some(u128::from(f32::INFINITY.to_bits()));

    authenticate_reviewed_safe_core_f32_is_finite_route_body_contract_v1(
        ReviewedSafeCoreF32IsFiniteRouteBodyContractV1 {
            argument_count: body.arg_count,
            local_count: body.local_decls.len(),
            block_count: body.basic_blocks.len(),
            source_scope_count: body.source_scopes.len(),
            exact_local_types,
            root_scope_not_inlined,
            entry_has_no_statements: entry.statements.is_empty(),
            exact_abs_callee,
            copies_input: matches!(&args[..], [argument] if matches!(&argument.node, Operand::Copy(place) if place.local.as_usize() == 1 && place.projection.is_empty())),
            writes_absolute_temporary: destination.local.as_usize() == 2
                && destination.projection.is_empty(),
            comparison_target: target.index() == 1,
            unwind_unreachable: matches!(unwind, UnwindAction::Unreachable),
            exact_less_than_infinity,
            returns_result: matches!(
                comparison.terminator.as_ref().map(|term| &term.kind),
                Some(TerminatorKind::Return)
            ),
        },
    )
}

// An independently closed form covers opt3 extraction tests, where rustc
// inlines `f32::abs` but retains its exact core instance in the source scope.
fn reviewed_safe_core_f32_is_finite_optimized_body_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> bool {
    if body.arg_count != 1 || body.local_decls.len() != 3 || body.basic_blocks.len() != 2 {
        return false;
    }
    let mut locals = body.local_decls.iter();
    if !matches!(locals.next(), Some(result) if matches!(result.ty.kind(), TyKind::Bool))
        || !matches!(locals.next(), Some(input) if matches!(input.ty.kind(), TyKind::Float(FloatTy::F32)))
        || !matches!(locals.next(), Some(absolute) if matches!(absolute.ty.kind(), TyKind::Float(FloatTy::F32)))
        || locals.next().is_some()
    {
        return false;
    }
    let mut scopes = body.source_scopes.iter();
    let Some(root_scope) = scopes.next() else {
        return false;
    };
    let Some(abs_scope) = scopes.next() else {
        return false;
    };
    if scopes.next().is_some() || root_scope.inlined.is_some() {
        return false;
    }
    let Some((inlined_abs, _)) = abs_scope.inlined else {
        return false;
    };
    if !authenticate_reviewed_safe_core_fabs_f32_helper_v1(tcx, inlined_abs) {
        return false;
    }

    let mut blocks = body.basic_blocks.iter();
    let Some(entry) = blocks.next() else {
        return false;
    };
    if !matches!(
        entry.statements.as_slice(),
        [statement]
            if matches!(statement.kind, StatementKind::StorageLive(local) if local.as_usize() == 2)
    ) {
        return false;
    }
    let Some(entry_terminator) = &entry.terminator else {
        return false;
    };
    let TerminatorKind::Call {
        func,
        args,
        destination,
        target,
        unwind,
        ..
    } = &entry_terminator.kind
    else {
        return false;
    };
    let Some(target) = target else {
        return false;
    };
    if target.index() != 1
        || !matches!(unwind, UnwindAction::Unreachable)
        || args.len() != 1
        || !matches!(&args[0].node, Operand::Move(place) if place.local.as_usize() == 1 && place.projection.is_empty())
        || destination.local.as_usize() != 2
        || !destination.projection.is_empty()
    {
        return false;
    }
    let Operand::Constant(function) = func else {
        return false;
    };
    let TyKind::FnDef(def_id, arguments) = function.const_.ty().kind() else {
        return false;
    };
    let Ok(Some(fabs)) =
        Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), *def_id, arguments)
    else {
        return false;
    };
    if !matches!(
        crate::production_rustc_intrinsic_v1::classify(tcx, fabs),
        Ok(Some(classification))
            if classification.operation
                == crate::production_rustc_intrinsic_v1::ProductionRustcIntrinsicOperationV1::FabsF32
    ) {
        return false;
    }

    let Some(comparison) = blocks.next() else {
        return false;
    };
    let [statement, storage_dead] = comparison.statements.as_slice() else {
        return false;
    };
    if !matches!(storage_dead.kind, StatementKind::StorageDead(local) if local.as_usize() == 2) {
        return false;
    }
    let StatementKind::Assign(assignment) = &statement.kind else {
        return false;
    };
    let (destination, rvalue) = &**assignment;
    if destination.local.as_usize() != 0 || !destination.projection.is_empty() {
        return false;
    }
    let Rvalue::BinaryOp(BinOp::Lt, operands) = rvalue else {
        return false;
    };
    let (absolute, infinity) = &**operands;
    if !matches!(absolute, Operand::Move(place) if place.local.as_usize() == 2 && place.projection.is_empty())
    {
        return false;
    }
    let Operand::Constant(infinity) = infinity else {
        return false;
    };
    if !matches!(infinity.const_.ty().kind(), TyKind::Float(FloatTy::F32))
        || infinity
            .const_
            .try_eval_bits(tcx, TypingEnv::fully_monomorphized())
            != Some(u128::from(f32::INFINITY.to_bits()))
    {
        return false;
    }
    matches!(
        comparison.terminator.as_ref().map(|term| &term.kind),
        Some(TerminatorKind::Return)
    )
}

fn reviewed_provider_semantic_definition_from_source_v1(
    tcx: TyCtxt<'_>,
    provider_definition: DefId,
    definition_source_domain: &[u8],
    source_closure_domain: &[u8],
    expected_source_closures: &[[u8; 32]],
    source_closure_cache: &OnceLock<Result<ReviewedProviderSourceClosureV1, String>>,
) -> Result<ReviewedProviderSemanticDefinitionV1, String> {
    let crate_num = provider_definition.krate;
    let crate_name = named_external_provider(tcx, crate_num)?;
    let provider = compiler_provider_observation_v1(tcx, crate_num);
    let source_closure = source_closure_cache
        .get_or_init(|| {
            let source = compiled_provider_source_path_v1(tcx, provider_definition)?;
            reviewed_provider_source_closure_from_definition(
                &source,
                source_closure_domain,
                expected_source_closures,
            )
        })
        .clone()?;
    let definition_source_identity = reviewed_compiled_provider_source_identity_at_root(
        tcx,
        provider_definition,
        &source_closure.source_root,
        definition_source_domain,
    )?;
    if provider.crate_name != crate_name {
        return Err("reviewed provider crate-name observation changed within the session".into());
    }
    let structural_local_definition_path = tcx
        .def_path(provider_definition)
        .to_string_no_crate_verbose();
    let canonical_definition_path =
        canonical_compiler_definition_path(&crate_name, &structural_local_definition_path)?;
    let structural_local_definition_component =
        structural_local_definition_component_v1(&structural_local_definition_path)?;
    Ok(ReviewedProviderSemanticDefinitionV1 {
        provider,
        canonical_definition_path,
        structural_local_definition_component,
        cargo_metadata_build_observation: decode_sha256_environment(
            tcx,
            CARGO_METADATA_BUILD_OBSERVATION_ENV_V2,
        )?,
        source_closure_identity: source_closure.identity,
        definition_source_identity,
    })
}

fn canonical_compiler_definition_path(
    authenticated_crate_name: &str,
    structural_local_path: &str,
) -> Result<String, String> {
    let structural_local_path = structural_local_path.trim_start_matches("::");
    if authenticated_crate_name.is_empty() || structural_local_path.is_empty() {
        return Err("compiler definition path is incomplete".to_owned());
    }
    Ok(format!(
        "{authenticated_crate_name}::{structural_local_path}"
    ))
}

pub(crate) fn structural_local_definition_component_v1(
    structural_local_path: &str,
) -> Result<[u8; 32], String> {
    let structural_local_path = structural_local_path.trim_start_matches("::");
    if structural_local_path.is_empty() {
        return Err("structural local definition path is empty".to_owned());
    }
    let mut hasher = Sha256::new();
    hash_source_identity_field(&mut hasher, STRUCTURAL_LOCAL_DEFINITION_COMPONENT_DOMAIN_V1);
    hash_source_identity_field(&mut hasher, structural_local_path.as_bytes());
    Ok(hasher.finalize().into())
}

#[cfg(test)]
pub(crate) fn pinned_core_semantic_terminal_identity_v1(
    provider: &CompilerProviderObservationV1,
    canonical_role: &str,
    structural_local_definition_path: &str,
) -> Result<[u8; 32], String> {
    let structural_local_definition_component =
        structural_local_definition_component_v1(structural_local_definition_path)?;
    if provider.crate_name != "core"
        || provider.stable_crate_id == 0
        || provider.crate_hash_observation == [0; 16]
        || canonical_role.is_empty()
        || structural_local_definition_component == [0; 32]
    {
        return Err("pinned core semantic terminal observation is incomplete".to_owned());
    }
    let canonical_definition_path =
        canonical_compiler_definition_path(&provider.crate_name, structural_local_definition_path)?;
    let mut hasher = Sha256::new();
    hash_source_identity_field(
        &mut hasher,
        PINNED_CORE_SEMANTIC_TERMINAL_TRANSCRIPT_DOMAIN_V1,
    );
    hash_source_identity_field(&mut hasher, canonical_role.as_bytes());
    hash_source_identity_field(&mut hasher, provider.crate_name.as_bytes());
    hash_source_identity_field(&mut hasher, canonical_definition_path.as_bytes());
    hash_source_identity_field(&mut hasher, &structural_local_definition_component);
    Ok(hasher.finalize().into())
}

fn reviewed_provider_source_closure_identity(
    package_root: &Path,
    domain: &[u8],
) -> Result<[u8; 32], String> {
    if domain.is_empty() {
        return Err("reviewed provider source-closure domain is empty".to_owned());
    }
    require_directory_without_symlink(package_root, "package root")?;
    let package_root = std::fs::canonicalize(package_root).map_err(|error| {
        format!("reviewed fe2o3-device package root is unavailable to the managed build: {error}")
    })?;
    let mut files = vec![package_root.join("Cargo.toml")];
    let build_script = package_root.join("build.rs");
    match std::fs::symlink_metadata(&build_script) {
        Ok(_) => {
            require_regular_file_without_symlink(&build_script)?;
            files.push(build_script);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "reviewed fe2o3-device source `{}` cannot be inspected: {error}",
                build_script.display()
            ));
        }
    }
    require_regular_file_without_symlink(&package_root.join("Cargo.toml"))?;
    let source_root = package_root.join("src");
    require_directory_without_symlink(&source_root, "source directory")?;
    collect_reviewed_source_files(&source_root, &mut files)?;
    let mut reviewed_files = files
        .into_iter()
        .map(|file| reviewed_source_file(&package_root, &file))
        .collect::<Result<Vec<_>, _>>()?;
    sort_reviewed_source_files_by_relative_path(&mut reviewed_files);

    let mut hasher = Sha256::new();
    hasher.update(domain);
    for (relative, bytes) in reviewed_files {
        hash_source_identity_field(&mut hasher, relative.as_bytes());
        hash_source_identity_field(&mut hasher, &bytes);
    }
    Ok(hasher.finalize().into())
}

fn sort_reviewed_source_files_by_relative_path(files: &mut [(String, Vec<u8>)]) {
    files.sort_by(|left, right| left.0.cmp(&right.0));
}

fn reviewed_provider_source_closure_from_definition(
    definition_source: &Path,
    domain: &[u8],
    expected_identities: &[[u8; 32]],
) -> Result<ReviewedProviderSourceClosureV1, String> {
    if domain.is_empty() || expected_identities.is_empty() || expected_identities.contains(&[0; 32])
    {
        return Err("reviewed provider source closure policy is incomplete".to_owned());
    }
    require_regular_file_without_symlink(definition_source)?;
    let definition_source = std::fs::canonicalize(definition_source).map_err(|error| {
        format!(
            "provider source file `{}` is unavailable to the managed build: {error}",
            definition_source.display()
        )
    })?;
    let source_root = definition_source
        .ancestors()
        .skip(1)
        .find(|ancestor| ancestor.file_name() == Some(std::ffi::OsStr::new("src")))
        .ok_or_else(|| {
            format!(
                "provider source file `{}` has no Cargo source root",
                definition_source.display()
            )
        })?;
    let package_root = source_root.parent().ok_or_else(|| {
        format!(
            "provider source file `{}` has no Cargo package root",
            definition_source.display()
        )
    })?;
    let identity = reviewed_provider_source_closure_identity(package_root, domain)?;
    if !expected_identities.contains(&identity) {
        return Err(format!(
            "provider source file `{}` is not contained by an exact reviewed fe2o3-device materialization",
            definition_source.display()
        ));
    }
    Ok(ReviewedProviderSourceClosureV1 {
        source_root: source_root.to_path_buf(),
        identity,
    })
}

fn require_directory_without_symlink(path: &Path, description: &str) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        format!(
            "reviewed fe2o3-device {description} `{}` is unavailable: {error}",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "reviewed fe2o3-device {description} `{}` is not a regular directory",
            path.display()
        ));
    }
    Ok(())
}

fn require_regular_file_without_symlink(path: &Path) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        format!(
            "reviewed fe2o3-device source `{}` is unavailable: {error}",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "reviewed fe2o3-device source `{}` is not a regular file",
            path.display()
        ));
    }
    Ok(())
}

fn reviewed_source_file(package_root: &Path, file: &Path) -> Result<(String, Vec<u8>), String> {
    require_regular_file_without_symlink(file)?;
    let canonical = std::fs::canonicalize(file).map_err(|error| {
        format!(
            "reviewed fe2o3-device source `{}` is unavailable: {error}",
            file.display()
        )
    })?;
    let relative = canonical.strip_prefix(package_root).map_err(|_| {
        format!(
            "reviewed fe2o3-device source `{}` escaped its package root",
            canonical.display()
        )
    })?;
    let relative = relative.to_str().ok_or_else(|| {
        format!(
            "reviewed fe2o3-device source path `{}` is not UTF-8",
            relative.display()
        )
    })?;
    let bytes = std::fs::read(&canonical).map_err(|error| {
        format!(
            "reviewed fe2o3-device source `{}` cannot be observed: {error}",
            canonical.display()
        )
    })?;
    Ok((relative.to_owned(), bytes))
}

fn collect_reviewed_source_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(root).map_err(|error| {
        format!(
            "reviewed fe2o3-device source directory `{}` is unavailable: {error}",
            root.display()
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "reviewed fe2o3-device source directory `{}` cannot be read: {error}",
                root.display()
            )
        })?;
        let file_type = entry.file_type().map_err(|error| {
            format!(
                "reviewed fe2o3-device source `{}` cannot be inspected: {error}",
                entry.path().display()
            )
        })?;
        if file_type.is_dir() {
            collect_reviewed_source_files(&entry.path(), files)?;
        } else if file_type.is_file() {
            files.push(entry.path());
        } else {
            return Err(format!(
                "reviewed fe2o3-device source `{}` is not a regular file",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

fn hash_source_identity_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn reviewed_compiled_provider_source_identity_at_root(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    reviewed_root: &Path,
    domain: &[u8],
) -> Result<[u8; 32], String> {
    let source_path = compiled_provider_source_path_v1(tcx, def_id)?;
    reviewed_provider_source_identity_from_path(reviewed_root, &source_path, domain)
}

fn compiled_provider_source_path_v1(tcx: TyCtxt<'_>, def_id: DefId) -> Result<PathBuf, String> {
    let span = tcx.def_span(def_id);
    let source_file = tcx.sess.source_map().lookup_source_file(span.lo());
    if source_file.cnum != def_id.krate {
        return Err("provider definition source came from a different compiler crate".to_owned());
    }
    let file_name = source_file
        .name
        .prefer_local_unconditionally()
        .to_string_lossy()
        .into_owned();
    let source_path = Path::new(&file_name);
    require_regular_file_without_symlink(source_path)?;
    let source_bytes = std::fs::read(source_path).map_err(|error| {
        format!(
            "provider source file `{}` cannot be observed by the managed build: {error}",
            source_path.display()
        )
    })?;
    let source = std::str::from_utf8(&source_bytes).map_err(|_| {
        format!(
            "provider source file `{}` is not UTF-8",
            source_path.display()
        )
    })?;
    validate_compiled_provider_source_hash_v1(&source_file.src_hash, source, source_path)?;
    std::fs::canonicalize(source_path).map_err(|error| {
        format!(
            "provider source file `{}` is unavailable to the managed build: {error}",
            source_path.display()
        )
    })
}

fn validate_compiled_provider_source_hash_v1(
    compiled_hash: &SourceFileHash,
    reviewed_source: &str,
    source_path: &Path,
) -> Result<(), String> {
    if compiled_hash.matches(reviewed_source) {
        return Ok(());
    }
    Err(format!(
        "compiled provider source hash does not match reviewed bytes for `{}`",
        source_path.display()
    ))
}

fn reviewed_provider_source_identity_from_path(
    reviewed_root: &Path,
    source: &Path,
    domain: &[u8],
) -> Result<[u8; 32], String> {
    if domain.is_empty() {
        return Err("reviewed provider definition-source domain is empty".to_owned());
    }
    require_regular_file_without_symlink(source)?;
    let source = std::fs::canonicalize(source).map_err(|error| {
        format!(
            "provider source file `{}` is unavailable to the managed build: {error}",
            source.display()
        )
    })?;
    let reviewed_root = std::fs::canonicalize(reviewed_root).map_err(|error| {
        format!("reviewed provider source root is unavailable to the managed build: {error}")
    })?;
    let relative = source.strip_prefix(&reviewed_root).map_err(|_| {
        format!(
            "provider source file `{}` is outside the reviewed fe2o3-device source root",
            source.display()
        )
    })?;
    let relative = relative.to_str().ok_or_else(|| {
        format!(
            "provider source file `{}` has a non-UTF-8 reviewed path",
            source.display()
        )
    })?;
    let source_bytes = std::fs::read(&source).map_err(|error| {
        format!(
            "provider source file `{}` cannot be observed by the managed build: {error}",
            source.display()
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((relative.len() as u64).to_le_bytes());
    hasher.update(relative.as_bytes());
    hasher.update((source_bytes.len() as u64).to_le_bytes());
    hasher.update(source_bytes);
    Ok(hasher.finalize().into())
}

fn decode_sha256_environment(tcx: TyCtxt<'_>, name: &str) -> Result<[u8; 32], String> {
    if let Some(value) = tcx.sess.opts.logical_env.get(name) {
        return decode_sha256_environment_value(name, value);
    }
    let value = std::env::var(name).map_err(|_| format!("managed build omitted {name}"))?;
    decode_sha256_environment_value(name, &value)
}

fn decode_sha256_environment_value(name: &str, value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("managed build supplied malformed {name}"));
    }
    let mut digest = [0_u8; 32];
    for (output, pair) in digest.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let text = std::str::from_utf8(pair).expect("ASCII hex is UTF-8");
        *output = u8::from_str_radix(text, 16)
            .map_err(|_| format!("managed build supplied malformed {name}"))?;
    }
    Ok(digest)
}

#[cfg(test)]
#[path = "trusted_device_items/managed_session_environment_v1_tests.rs"]
mod managed_session_environment_v1_tests;
#[cfg(test)]
pub(crate) use managed_session_environment_v1_tests::observe_managed_build_value_v1;

fn half_math_path(item: Fe2o3DeviceDiagnosticItem) -> &'static str {
    HALF_MATH_DIAGNOSTIC_ITEMS
        .iter()
        .find_map(|(marker, path)| {
            (dialect_amdgcn::recognize_fe2o3_device_diagnostic_item(marker) == Some(item))
                .then_some(*path)
        })
        .expect("every trusted half/math diagnostic item has one canonical path")
}

impl TrustedHalfOperation {
    fn canonical_path(self) -> &'static str {
        use NarrowFloatFormat::{Bf16, F16};
        use WidenedFloatBinaryOp::{Add, Divide, Multiply, Subtract};
        match self {
            Self::FromBits(F16) => "fe2o3_device::F16::from_bits",
            Self::FromBits(Bf16) => "fe2o3_device::Bf16::from_bits",
            Self::ToBits(F16) => "fe2o3_device::F16::to_bits",
            Self::ToBits(Bf16) => "fe2o3_device::Bf16::to_bits",
            Self::FromF32(F16) => "fe2o3_device::F16::from_f32",
            Self::FromF32(Bf16) => "fe2o3_device::Bf16::from_f32",
            Self::ToF32(F16) => "fe2o3_device::F16::to_f32",
            Self::ToF32(Bf16) => "fe2o3_device::Bf16::to_f32",
            Self::WidenedBinary {
                format: F16,
                op: Add,
            } => "<fe2o3_device::F16 as core::ops::Add>::add",
            Self::WidenedBinary {
                format: F16,
                op: Subtract,
            } => "<fe2o3_device::F16 as core::ops::Sub>::sub",
            Self::WidenedBinary {
                format: F16,
                op: Multiply,
            } => "<fe2o3_device::F16 as core::ops::Mul>::mul",
            Self::WidenedBinary {
                format: F16,
                op: Divide,
            } => "<fe2o3_device::F16 as core::ops::Div>::div",
            Self::WidenedBinary {
                format: Bf16,
                op: Add,
            } => "<fe2o3_device::Bf16 as core::ops::Add>::add",
            Self::WidenedBinary {
                format: Bf16,
                op: Subtract,
            } => "<fe2o3_device::Bf16 as core::ops::Sub>::sub",
            Self::WidenedBinary {
                format: Bf16,
                op: Multiply,
            } => "<fe2o3_device::Bf16 as core::ops::Mul>::mul",
            Self::WidenedBinary {
                format: Bf16,
                op: Divide,
            } => "<fe2o3_device::Bf16 as core::ops::Div>::div",
            Self::Bf16x2FusedMultiplyAdd => "fe2o3_device::Bf16x2::mul_add_widened",
        }
    }
}

fn classify_half_operation(tcx: TyCtxt<'_>, def_id: DefId) -> Option<TrustedHalfOperation> {
    let associated = tcx.opt_associated_item(def_id)?;
    if !associated.is_fn() {
        return None;
    }
    let impl_id = tcx.impl_of_assoc(def_id)?;
    if impl_id.krate == LOCAL_CRATE {
        return None;
    }
    let self_ty = tcx.type_of(impl_id).instantiate_identity();
    let TyKind::Adt(adt, _) = self_ty.kind() else {
        return None;
    };
    let value = match classify(tcx, adt.did())? {
        TrustedDeviceItem::DeviceValue(value) => value,
        _ => return None,
    };
    let item_name = tcx.item_name(def_id);
    let name = item_name.as_str();

    if tcx.impl_is_of_trait(impl_id) {
        let trait_id = tcx.impl_trait_id(impl_id);
        let (lang_item, op) = match name {
            "add" => (LangItem::Add, WidenedFloatBinaryOp::Add),
            "sub" => (LangItem::Sub, WidenedFloatBinaryOp::Subtract),
            "mul" => (LangItem::Mul, WidenedFloatBinaryOp::Multiply),
            "div" => (LangItem::Div, WidenedFloatBinaryOp::Divide),
            _ => return None,
        };
        if tcx.lang_items().get(lang_item) != Some(trait_id) {
            return None;
        }
        return Some(TrustedHalfOperation::WidenedBinary {
            format: narrow_format(value)?,
            op,
        });
    }

    match (value, name) {
        (DeviceValueDiagnosticItem::F16, "from_bits") => {
            Some(TrustedHalfOperation::FromBits(NarrowFloatFormat::F16))
        }
        (DeviceValueDiagnosticItem::Bf16, "from_bits") => {
            Some(TrustedHalfOperation::FromBits(NarrowFloatFormat::Bf16))
        }
        (DeviceValueDiagnosticItem::F16, "to_bits") => {
            Some(TrustedHalfOperation::ToBits(NarrowFloatFormat::F16))
        }
        (DeviceValueDiagnosticItem::Bf16, "to_bits") => {
            Some(TrustedHalfOperation::ToBits(NarrowFloatFormat::Bf16))
        }
        (DeviceValueDiagnosticItem::F16, "from_f32") => {
            Some(TrustedHalfOperation::FromF32(NarrowFloatFormat::F16))
        }
        (DeviceValueDiagnosticItem::Bf16, "from_f32") => {
            Some(TrustedHalfOperation::FromF32(NarrowFloatFormat::Bf16))
        }
        (DeviceValueDiagnosticItem::F16, "to_f32") => {
            Some(TrustedHalfOperation::ToF32(NarrowFloatFormat::F16))
        }
        (DeviceValueDiagnosticItem::Bf16, "to_f32") => {
            Some(TrustedHalfOperation::ToF32(NarrowFloatFormat::Bf16))
        }
        (DeviceValueDiagnosticItem::Bf16x2, "mul_add_widened") => {
            Some(TrustedHalfOperation::Bf16x2FusedMultiplyAdd)
        }
        _ => None,
    }
}

const fn narrow_format(value: DeviceValueDiagnosticItem) -> Option<NarrowFloatFormat> {
    match value {
        DeviceValueDiagnosticItem::F16 => Some(NarrowFloatFormat::F16),
        DeviceValueDiagnosticItem::Bf16 => Some(NarrowFloatFormat::Bf16),
        DeviceValueDiagnosticItem::Bf16x2 => None,
    }
}

#[cfg(test)]
mod tests {
    include!("trusted_device_items/core_01_tests.rs");
    include!("trusted_device_items/generative_provider_v1_tests.rs");
    include!("trusted_device_items/wave64_shuffle_provider_v1_tests.rs");
    include!("trusted_device_items/wave64_scan_metadata_v1_tests.rs");
    include!("trusted_device_items/materialization_v1_tests.rs");

    include!("trusted_device_items/wrapping_integer_v1_tests.rs");
    include!("trusted_device_items/ordered_region_v31_tests.rs");
    include!("trusted_device_items/ordered_program_v32_tests.rs");
    include!("trusted_device_items/complete_body_v36_tests.rs");

    #[test]
    fn gfx942_inline_v30_providers_require_exact_reviewed_definition_and_source() {
        let cases = [
            (
                TrustedAmdGpuInlineOperation::VMovB32,
                "__amdgpu_v_mov_b32_v1",
            ),
            (
                TrustedAmdGpuInlineOperation::VAddU32,
                "__amdgpu_v_add_u32_v1",
            ),
            (
                TrustedAmdGpuInlineOperation::VSubU32,
                "__amdgpu_v_sub_u32_v1",
            ),
            (
                TrustedAmdGpuInlineOperation::VAndB32,
                "__amdgpu_v_and_b32_v1",
            ),
            (TrustedAmdGpuInlineOperation::VOrB32, "__amdgpu_v_or_b32_v1"),
            (
                TrustedAmdGpuInlineOperation::VXorB32,
                "__amdgpu_v_xor_b32_v1",
            ),
        ];
        for (operation, name) in cases {
            let item = TrustedDeviceItem::AmdGpuInline(operation);
            let local = format!("diagnostics::{name}");
            let exact_path = format!("fe2o3_device::{local}");
            assert_eq!(
                exact_provider_compiler_definition_path_v1(item),
                Some(exact_path.as_str())
            );
            let exact = semantic_definition(
                &local,
                super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
                [6; 32],
            );
            validate_reviewed_fe2o3_device_provider_definition_v1(item, &exact).unwrap();
            for wrong_path in [
                format!("local::{name}"),
                format!("diagnostics::lookalike::{name}"),
                "diagnostics::__amdgpu_v_mov_b32_impostor_v1".to_owned(),
            ] {
                let changed = semantic_definition(
                    &wrong_path,
                    super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
                    [6; 32],
                );
                assert!(
                    validate_reviewed_fe2o3_device_provider_definition_v1(item, &changed).is_err()
                );
            }
            for wrong_crate in ["local_marker", "fe2o3_device_lookalike"] {
                let mut changed = exact.clone();
                changed.provider.crate_name = wrong_crate.into();
                assert!(
                    validate_reviewed_fe2o3_device_provider_definition_v1(item, &changed).is_err()
                );
            }
            let mut stale = exact.clone();
            stale.source_closure_identity[0] ^= 1;
            assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &stale).is_err());
            // A different legitimate ISA marker is not this operation's provider.
            let other = if operation == TrustedAmdGpuInlineOperation::VMovB32 {
                TrustedAmdGpuInlineOperation::VAddU32
            } else {
                TrustedAmdGpuInlineOperation::VMovB32
            };
            assert!(
                validate_reviewed_fe2o3_device_provider_definition_v1(
                    TrustedDeviceItem::AmdGpuInline(other),
                    &exact
                )
                .is_err()
            );
        }
    }
    include!("trusted_device_items/wrapping_shift_source_safety_v1_tests.rs");

    #[test]
    fn exact_device_provider_rejects_same_name_path_and_source_substitution() {
        let item = TrustedDeviceItem::ThreadIndexCheckedBlock;
        let structural = exact_provider_compiler_definition_path_v1(item).unwrap();
        let local = structural.strip_prefix("fe2o3_device::").unwrap();
        let exact = semantic_definition(
            local,
            super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
            [6; 32],
        );
        validate_reviewed_fe2o3_device_provider_definition_v1(item, &exact)
            .expect("exact reviewed provider");

        let wrong_path = semantic_definition(
            "thread::{impl#99}::checked_block",
            super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
            [6; 32],
        );
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &wrong_path).is_err());
        let mut impostor = exact.clone();
        impostor.provider.crate_name = "same_name_impostor".into();
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &impostor).is_err());
        impostor = exact.clone();
        impostor.source_closure_identity[0] ^= 1;
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &impostor).is_err());
    }

    #[test]
    fn neutral_workgroup_terminals_reject_legacy_and_stale_provider_identities() {
        let neutral = [
            TrustedDeviceItem::WorkgroupGroup,
            TrustedDeviceItem::WorkgroupCollectiveScratch,
            TrustedDeviceItem::WorkgroupCollectivesContext,
            TrustedDeviceItem::WorkgroupCollectivesCurrent,
            TrustedDeviceItem::WorkgroupReduceSum,
            TrustedDeviceItem::WorkgroupInclusiveScanSum,
            TrustedDeviceItem::WorkgroupExclusiveScanSum,
        ];
        for item in neutral {
            let structural = exact_provider_compiler_definition_path_v1(item).unwrap();
            let local = structural.strip_prefix("fe2o3_device::").unwrap();
            let exact = semantic_definition(
                local,
                super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
                [6; 32],
            );
            validate_reviewed_fe2o3_device_provider_definition_v1(item, &exact).unwrap();

            let legacy = semantic_definition(
                "collective::{impl#6}::reduce_sum",
                super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
                [6; 32],
            );
            assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &legacy).is_err());

            let mut stale = exact.clone();
            stale.source_closure_identity[0] ^= 1;
            assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &stale).is_err());
        }
    }

    #[test]
    fn checked_view_capabilities_reject_type_constructor_and_load_lookalikes() {
        for item in [
            TrustedDeviceItem::StridedReadView2D,
            TrustedDeviceItem::StridedReadView2DError,
            TrustedDeviceItem::StridedReadView2DFromSharedSlice,
            TrustedDeviceItem::StridedReadView2DLoadOr,
            TrustedDeviceItem::Bf16MfmaMatrixView,
            TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView,
            TrustedDeviceItem::Bf16MfmaBColumnMajor,
            TrustedDeviceItem::Bf16MfmaBColumnMajorLoadZeroFilledV1,
            TrustedDeviceItem::Bf16MfmaMatrixViewError,
            TrustedDeviceItem::Bf16MfmaMatrixARowMajor,
            TrustedDeviceItem::Bf16MfmaMatrixBRowMajor,
            TrustedDeviceItem::Bf16MfmaMatrixALoadZeroFilledV2,
            TrustedDeviceItem::Bf16MfmaMatrixBLoadZeroFilledV2,
        ] {
            let structural = exact_provider_compiler_definition_path_v1(item).unwrap();
            let local = structural.strip_prefix("fe2o3_device::").unwrap();
            let exact = semantic_definition(
                local,
                super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
                [6; 32],
            );
            validate_reviewed_fe2o3_device_provider_definition_v1(item, &exact)
                .expect("exact checked read-view capability");

            let lookalike = semantic_definition(
                &format!("lookalike::{local}"),
                super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
                [6; 32],
            );
            assert!(
                validate_reviewed_fe2o3_device_provider_definition_v1(item, &lookalike).is_err(),
                "same-signature lookalike authenticated as {item:?}"
            );

            let mut wrong_crate = exact.clone();
            wrong_crate.provider.crate_name = "fe2o3_device_lookalike".into();
            assert!(
                validate_reviewed_fe2o3_device_provider_definition_v1(item, &wrong_crate).is_err(),
                "wrong-crate lookalike authenticated as {item:?}"
            );
        }
    }

    #[test]
    fn every_production_safety_terminal_has_an_exact_structural_path() {
        for item in [
            TrustedDeviceItem::DisjointSlice,
            TrustedDeviceItem::WriteOnlyDisjointSlice,
            TrustedDeviceItem::StridedReadView2D,
            TrustedDeviceItem::StridedReadView2DError,
            TrustedDeviceItem::StridedReadView2DFromSharedSlice,
            TrustedDeviceItem::StridedReadView2DLoadOr,
            TrustedDeviceItem::ThreadIndex,
            TrustedDeviceItem::DisjointIndex,
            TrustedDeviceItem::ShiftedIndexSpace,
            TrustedDeviceItem::BlockedIndexSpace,
            TrustedDeviceItem::Tiled2DIndexSpace,
            TrustedDeviceItem::RowStriped2DIndexSpace,
            TrustedDeviceItem::GridExclusiveIndexSpace,
            TrustedDeviceItem::GridLeader,
            TrustedDeviceItem::DisjointBlock,
            TrustedDeviceItem::DisjointTile2D,
            TrustedDeviceItem::DisjointRowStripe2D,
            TrustedDeviceItem::ThreadIndex1d,
            TrustedDeviceItem::ThreadIndexGet,
            TrustedDeviceItem::ThreadIndexIntoDisjoint,
            TrustedDeviceItem::ThreadIndexCheckedShift,
            TrustedDeviceItem::ThreadIndexCheckedBlock,
            TrustedDeviceItem::ThreadIndexCheckedTiled2D,
            TrustedDeviceItem::ThreadIndexCheckedRowStriped2D,
            TrustedDeviceItem::DisjointIndexGet,
            TrustedDeviceItem::DisjointIndexCheckedShift,
            TrustedDeviceItem::DisjointBlockComponentIndex,
            TrustedDeviceItem::GridLeaderCurrent,
            TrustedDeviceItem::DisjointSliceGetMut,
            TrustedDeviceItem::DisjointSliceGetDisjointMut,
            TrustedDeviceItem::DisjointSliceGetMutExclusive,
            TrustedDeviceItem::DisjointSliceGetBlockMut,
            TrustedDeviceItem::DisjointSliceGetTiled2DMut,
            TrustedDeviceItem::DisjointSliceGetRowStriped2DMut,
            TrustedDeviceItem::WriteOnlyDisjointSliceLen,
            TrustedDeviceItem::WriteOnlyDisjointSliceWrite,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteDisjoint,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteExclusive,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteBlock,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteTiled2D,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteRowStriped2D,
            TrustedDeviceItem::DeviceGlobalMutPtrU32AsAtomic,
            TrustedDeviceItem::DeviceGlobalMutPtrI32AsAtomic,
            TrustedDeviceItem::DeviceGlobalMutPtrU64AsAtomic,
            TrustedDeviceItem::DeviceGlobalMutPtrI64AsAtomic,
        ] {
            let path = exact_provider_compiler_definition_path_v1(item)
                .unwrap_or_else(|| panic!("{item:?} lacks exact provider authentication"));
            assert!(path.starts_with("fe2o3_device::"));
        }
    }

    #[test]
    fn safe_execution_source_closure_matches_the_reviewed_pin() {
        let closure = reviewed_provider_source_closure_identity(
            Path::new(super::REVIEWED_FE2O3_DEVICE_PACKAGE_ROOT),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        )
        .unwrap();
        assert_eq!(closure, super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1);

        let definition = semantic_definition("thread::thread_idx_x", [5; 32], [6; 32]);
        assert_eq!(
            definition
                .durable_semantic_identity("fe2o3_device::thread::thread_idx_x")
                .unwrap(),
            digest("36349edbdabe77499ba36d983bf758f7c00e982d7fbd930397042192af1e7416")
        );
    }

    #[test]
    fn source_closure_uses_canonical_relative_order_for_component_prefixes() {
        let mut forward = vec![
            ("src/group/tests.rs".to_owned(), vec![4]),
            ("src/collective.rs".to_owned(), vec![1]),
            ("src/group.rs".to_owned(), vec![3]),
            ("src/collective/tests.rs".to_owned(), vec![2]),
        ];
        let mut reversed = forward.clone();
        reversed.reverse();

        sort_reviewed_source_files_by_relative_path(&mut forward);
        sort_reviewed_source_files_by_relative_path(&mut reversed);
        assert_eq!(forward, reversed);
        assert_eq!(
            forward
                .iter()
                .map(|(relative, _)| relative.as_str())
                .collect::<Vec<_>>(),
            [
                "src/collective.rs",
                "src/collective/tests.rs",
                "src/group.rs",
                "src/group/tests.rs",
            ]
        );

        let closure = reviewed_provider_source_closure_identity(
            Path::new(super::REVIEWED_FE2O3_DEVICE_PACKAGE_ROOT),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        )
        .unwrap();
        assert_eq!(
            closure,
            digest("0e43cb1e8156a7ec6bd9dfda344b188b9c0a3155c7395b9e3d6f685516c74d8b")
        );
        assert_eq!(closure, super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1);
    }

    #[test]
    fn reviewed_bf16_conversion_terminals_remain_out_of_line() {
        let source =
            fs::read_to_string(Path::new(super::REVIEWED_FE2O3_DEVICE_SOURCE_ROOT).join("half.rs"))
                .unwrap();
        let bf16_impl = source
            .split_once("impl Bf16 {")
            .unwrap()
            .1
            .split_once("pub struct Bf16x2")
            .unwrap()
            .0;

        for signature in [
            "pub const fn from_bits(bits: u16) -> Self",
            "pub const fn to_bits(self) -> u16",
            "pub const fn from_f32(value: f32) -> Self",
            "pub const fn to_f32(self) -> f32",
        ] {
            assert!(
                bf16_impl.contains(&format!("#[inline(never)]\n    {signature}")),
                "reviewed BF16 terminal `{signature}` may inline into optimized external MIR"
            );
        }
    }

    #[test]
    fn reviewed_blocked_access_terminals_remain_out_of_line() {
        let slice_source =
            fs::read_to_string(Path::new(super::REVIEWED_FE2O3_DEVICE_SOURCE_ROOT).join("lib.rs"))
                .unwrap();
        let thread_source = fs::read_to_string(
            Path::new(super::REVIEWED_FE2O3_DEVICE_SOURCE_ROOT).join("thread.rs"),
        )
        .unwrap();
        assert!(
            thread_source.contains(
                "#[inline(never)]\n    #[rustc_diagnostic_item = \"fe2o3_device_thread_index_get\"]\n    pub fn get(&self) -> usize"
            ),
            "reviewed thread-index access terminal may inline into optimized external MIR"
        );
        assert!(
            thread_source.contains(
                "#[inline(never)]\n    #[rustc_diagnostic_item = \"fe2o3_device_thread_index_checked_block\"]\n    pub fn checked_block<"
            ),
            "reviewed blocked index terminal may inline into optimized external MIR"
        );
        assert!(
            thread_source.contains(
                "#[inline(never)]\n#[rustc_diagnostic_item = \"fe2o3_device_thread_index_1d\"]\npub fn index_1d() -> ThreadIndex"
            ),
            "reviewed thread-index producer may inline into optimized external MIR"
        );
        assert!(
            slice_source.contains(
                "#[inline(never)]\n    #[rustc_diagnostic_item = \"fe2o3_device_disjoint_slice_get_block_mut\"]\n    pub fn get_block_mut("
            ),
            "reviewed blocked access terminal may inline into optimized external MIR"
        );
    }

    #[test]
    fn reviewed_device_source_excludes_retired_exact_profile_allocators() {
        let source_root = Path::new(super::REVIEWED_FE2O3_DEVICE_SOURCE_ROOT);
        let mut files = Vec::new();
        super::collect_reviewed_source_files(source_root, &mut files).unwrap();
        let retired = [
            b"gfx942_lds_bf16_tile_pair_m16x16_v1".as_slice(),
            b"gfx942_publish_lds_bf16_tile_pair_m16x16_v1".as_slice(),
        ];

        for file in files {
            let bytes = fs::read(&file).unwrap();
            for symbol in retired {
                assert!(
                    !bytes.windows(symbol.len()).any(|window| window == symbol),
                    "retired exact-profile allocator `{}` reentered reviewed device source `{}`",
                    String::from_utf8_lossy(symbol),
                    file.display(),
                );
            }
        }
    }

    #[test]
    fn safe_execution_provider_validation_rejects_source_substitution() {
        let exact = semantic_definition(
            "wave::{impl#4}::current",
            super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1,
            [6; 32],
        );
        super::validate_safe_execution_provider_definition_v1(&exact).unwrap();

        let mut changed = exact.clone();
        changed.source_closure_identity[0] ^= 1;
        assert!(super::validate_safe_execution_provider_definition_v1(&changed).is_err());
    }

    #[test]
    fn compiled_source_hash_rejects_stale_rlib_after_source_restore() {
        let stale_compiled = SourceFileHash::new_in_memory(
            SourceFileHashAlgorithm::Sha256,
            "modified provider source",
        );
        validate_compiled_provider_source_hash_v1(
            &stale_compiled,
            "modified provider source",
            Path::new("provider.rs"),
        )
        .unwrap();
        assert!(
            validate_compiled_provider_source_hash_v1(
                &stale_compiled,
                "reviewed provider source restored on disk",
                Path::new("provider.rs"),
            )
            .is_err()
        );
    }

    #[test]
    fn complete_source_closure_binds_manifest_build_script_and_nested_sources() {
        let fixture = ProviderPackageFixture::new();
        let identity = || {
            reviewed_provider_source_closure_identity(
                &fixture.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .unwrap()
        };
        let initial = identity();

        fs::write(
            fixture.root.join("Cargo.toml"),
            b"[package]\nname='changed'\n",
        )
        .unwrap();
        let changed_manifest = identity();
        assert_ne!(changed_manifest, initial);

        fs::write(fixture.root.join("build.rs"), b"fn main() {}\n").unwrap();
        let with_build_script = identity();
        assert_ne!(with_build_script, changed_manifest);

        fs::write(
            fixture.root.join("src/nested/mod.rs"),
            b"pub fn changed() {}\n",
        )
        .unwrap();
        let changed_nested_source = identity();
        assert_ne!(changed_nested_source, with_build_script);

        fs::write(
            fixture.root.join("src/extra.rs"),
            b"pub const EXTRA: u8 = 1;\n",
        )
        .unwrap();
        assert_ne!(identity(), changed_nested_source);
    }

    #[test]
    fn exact_source_closure_authentication_is_location_independent() {
        let reviewed = ProviderPackageFixture::new();
        let identity = reviewed_provider_source_closure_identity(
            &reviewed.root,
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        )
        .unwrap();
        let located = reviewed_provider_source_closure_from_definition(
            &reviewed.definition(),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            &[identity],
        )
        .unwrap();
        assert_eq!(
            located.source_root,
            fs::canonicalize(reviewed.source_root()).unwrap()
        );
        assert_eq!(located.identity, identity);

        let mut substituted_identity = identity;
        substituted_identity[0] ^= 1;
        assert!(
            reviewed_provider_source_closure_from_definition(
                &reviewed.definition(),
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
                &[substituted_identity],
            )
            .is_err()
        );
        assert!(
            reviewed_provider_source_closure_from_definition(
                &reviewed.definition(),
                b"",
                &[identity],
            )
            .is_err()
        );
    }

    #[test]
    fn source_closure_rejects_missing_inputs_and_out_of_root_definitions() {
        let missing_manifest = ProviderPackageFixture::new();
        fs::remove_file(missing_manifest.root.join("Cargo.toml")).unwrap();
        assert!(
            reviewed_provider_source_closure_identity(
                &missing_manifest.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .is_err()
        );

        let missing_source_root = ProviderPackageFixture::new();
        fs::remove_dir_all(missing_source_root.source_root()).unwrap();
        assert!(
            reviewed_provider_source_closure_identity(
                &missing_source_root.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .is_err()
        );

        let reviewed = ProviderPackageFixture::new();
        let outside = ProviderPackageFixture::new();
        let reviewed_root = fs::canonicalize(&reviewed.root).unwrap();
        assert!(super::reviewed_source_file(&reviewed_root, &outside.definition()).is_err());
        assert!(
            reviewed_provider_source_identity_from_path(
                &reviewed.source_root(),
                &outside.definition(),
                WORKGROUP_SYNC_PROVIDER_SOURCE_IDENTITY_DOMAIN_V1,
            )
            .is_err()
        );
        assert!(reviewed_provider_source_closure_identity(&reviewed.root, b"").is_err());
        assert!(
            reviewed_provider_source_identity_from_path(
                &reviewed.source_root(),
                &reviewed.definition(),
                b"",
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn source_closure_rejects_symlinks_and_nonregular_files() {
        use std::os::fd::AsRawFd as _;
        use std::os::unix::fs::symlink;

        let manifest_link = ProviderPackageFixture::new();
        let manifest = manifest_link.root.join("Cargo.toml");
        let real_manifest = manifest_link.root.join("Cargo.real.toml");
        fs::rename(&manifest, &real_manifest).unwrap();
        symlink(&real_manifest, &manifest).unwrap();
        assert!(
            reviewed_provider_source_closure_identity(
                &manifest_link.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .is_err()
        );

        let source_link = ProviderPackageFixture::new();
        symlink(
            source_link.definition(),
            source_link.source_root().join("alias.rs"),
        )
        .unwrap();
        assert!(
            reviewed_provider_source_closure_identity(
                &source_link.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .is_err()
        );

        let source_root_link = ProviderPackageFixture::new();
        fs::rename(
            source_root_link.source_root(),
            source_root_link.root.join("real-src"),
        )
        .unwrap();
        symlink(
            source_root_link.root.join("real-src"),
            source_root_link.source_root(),
        )
        .unwrap();
        assert!(
            reviewed_provider_source_closure_identity(
                &source_root_link.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .is_err()
        );

        let nonregular = ProviderPackageFixture::new();
        let source_root = fs::File::open(nonregular.source_root()).unwrap();
        // SAFETY: the descriptor owns the fixture source directory and the relative name is a
        // static NUL-terminated C string.
        let result =
            unsafe { libc::mkfifoat(source_root.as_raw_fd(), c"provider.fifo".as_ptr(), 0o600) };
        let error = std::io::Error::last_os_error();
        assert_eq!(result, 0, "create nonregular source fixture: {error}");
        let error = reviewed_provider_source_closure_identity(
            &nonregular.root,
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        );
        let error = error.unwrap_err();
        assert!(error.contains("provider.fifo"), "{error}");
        assert!(error.contains("is not a regular file"), "{error}");
    }

    #[test]
    fn pinned_core_terminal_identity_excludes_volatile_crate_disambiguators() {
        let core = CompilerProviderObservationV1 {
            crate_name: "core".into(),
            stable_crate_id: 9,
            crate_hash_observation: [8; 16],
        };
        let identity = |provider: &CompilerProviderObservationV1, role, path| {
            pinned_core_semantic_terminal_identity_v1(provider, role, path)
        };
        let exact = identity(
            &core,
            "core::intrinsics::atomic_xadd",
            "intrinsics::atomic_xadd",
        )
        .expect("complete core terminal identity");

        let mut mutation = core.clone();
        mutation.stable_crate_id ^= 1;
        assert_eq!(
            identity(
                &mutation,
                "core::intrinsics::atomic_xadd",
                "intrinsics::atomic_xadd",
            )
            .unwrap(),
            exact
        );
        mutation = core.clone();
        mutation.crate_hash_observation[0] ^= 1;
        assert_eq!(
            identity(
                &mutation,
                "core::intrinsics::atomic_xadd",
                "intrinsics::atomic_xadd",
            )
            .unwrap(),
            exact
        );
        assert_ne!(
            identity(
                &core,
                "core::intrinsics::atomic_xsub",
                "intrinsics::atomic_xadd",
            )
            .unwrap(),
            exact
        );
        assert_ne!(
            identity(
                &core,
                "core::intrinsics::atomic_xadd",
                "intrinsics::atomic_xsub",
            )
            .unwrap(),
            exact
        );
        mutation = core;
        mutation.crate_name = "impostor_core".into();
        assert!(
            identity(
                &mutation,
                "core::intrinsics::atomic_xadd",
                "intrinsics::atomic_xadd",
            )
            .is_err()
        );
    }

    #[test]
    fn structural_definition_paths_are_canonical_and_crate_qualified() {
        assert_eq!(
            canonical_compiler_definition_path(
                "fe2o3_device",
                "::__fe2o3_kernel_device::thread_idx_x",
            )
            .unwrap(),
            "fe2o3_device::__fe2o3_kernel_device::thread_idx_x"
        );
        assert!(canonical_compiler_definition_path("", "thread_idx_x").is_err());
        assert!(canonical_compiler_definition_path("fe2o3_device", "::").is_err());
    }

    #[test]
    fn semantic_registry_is_complete_and_unique() {
        let items = [
            TrustedDeviceItem::KernelError,
            TrustedDeviceItem::KernelContext,
            TrustedDeviceItem::KernelContextIssue,
            TrustedDeviceItem::ExecutionWorkgroupCapability,
            TrustedDeviceItem::ExecutionWorkgroupCurrent,
            TrustedDeviceItem::ExecutionWithWorkgroup,
            TrustedDeviceItem::MaskedTile1D,
            TrustedDeviceItem::LaneFragment1D,
            TrustedDeviceItem::MaskedTile1DLoadMasked,
            TrustedDeviceItem::MaskedTile1DIntoFragment,
            TrustedDeviceItem::LaneFragment1DIntoParts,
            TrustedDeviceItem::DisjointSlice,
            TrustedDeviceItem::WriteOnlyDisjointSlice,
            TrustedDeviceItem::StridedReadView2D,
            TrustedDeviceItem::StridedReadView2DError,
            TrustedDeviceItem::StridedReadView2DFromSharedSlice,
            TrustedDeviceItem::StridedReadView2DLoadOr,
            TrustedDeviceItem::DeviceGlobalMutPtr,
            TrustedDeviceItem::WorkgroupLdsScope,
            TrustedDeviceItem::WorkgroupLdsScopeCurrent,
            TrustedDeviceItem::DynamicLdsExactCurrent,
            TrustedDeviceItem::DynamicLdsIntoCollectiveRawParts,
            TrustedDeviceItem::LdsUninitialized,
            TrustedDeviceItem::WorkgroupPipeline,
            TrustedDeviceItem::WorkgroupPipelineCurrent,
            TrustedDeviceItem::WorkgroupPipelineStage,
            TrustedDeviceItem::WorkgroupPipelineWrite,
            TrustedDeviceItem::WorkgroupPipelineCommit,
            TrustedDeviceItem::WorkgroupPipelineWait,
            TrustedDeviceItem::WorkgroupPipelineConsume,
            TrustedDeviceItem::WorkgroupPipelineRead,
            TrustedDeviceItem::WorkgroupPipelineDiscard,
            TrustedDeviceItem::WorkgroupPipelineRelease,
            TrustedDeviceItem::Invocation3D,
            TrustedDeviceItem::Invocation3DCurrent,
            TrustedDeviceItem::ThreadIndexX,
            TrustedDeviceItem::ThreadIndexY,
            TrustedDeviceItem::ThreadIndexZ,
            TrustedDeviceItem::WorkgroupIndexX,
            TrustedDeviceItem::WorkgroupIndexY,
            TrustedDeviceItem::WorkgroupIndexZ,
            TrustedDeviceItem::WorkgroupDimensionX,
            TrustedDeviceItem::WorkgroupDimensionY,
            TrustedDeviceItem::WorkgroupDimensionZ,
            TrustedDeviceItem::GridDimensionX,
            TrustedDeviceItem::GridDimensionY,
            TrustedDeviceItem::GridDimensionZ,
            TrustedDeviceItem::ThreadIndex,
            TrustedDeviceItem::DisjointIndex,
            TrustedDeviceItem::ShiftedIndexSpace,
            TrustedDeviceItem::BlockedIndexSpace,
            TrustedDeviceItem::Tiled2DIndexSpace,
            TrustedDeviceItem::RowStriped2DIndexSpace,
            TrustedDeviceItem::GridExclusiveIndexSpace,
            TrustedDeviceItem::GridLeader,
            TrustedDeviceItem::DisjointBlock,
            TrustedDeviceItem::DisjointTile2D,
            TrustedDeviceItem::DisjointRowStripe2D,
            TrustedDeviceItem::ThreadIndex1d,
            TrustedDeviceItem::ThreadIndexGet,
            TrustedDeviceItem::ThreadIndexIntoDisjoint,
            TrustedDeviceItem::ThreadIndexCheckedShift,
            TrustedDeviceItem::ThreadIndexCheckedBlock,
            TrustedDeviceItem::ThreadIndexCheckedTiled2D,
            TrustedDeviceItem::ThreadIndexCheckedRowStriped2D,
            TrustedDeviceItem::DisjointIndexGet,
            TrustedDeviceItem::DisjointIndexCheckedShift,
            TrustedDeviceItem::DisjointBlockComponentIndex,
            TrustedDeviceItem::GridLeaderCurrent,
            TrustedDeviceItem::ThreadIndexOffset,
            TrustedDeviceItem::ThreadIndexOffsetSigned,
            TrustedDeviceItem::ThreadIndexStride,
            TrustedDeviceItem::ThreadIndexStrideOffset,
            TrustedDeviceItem::DisjointSliceGetMut,
            TrustedDeviceItem::DisjointSliceGetDisjointMut,
            TrustedDeviceItem::DisjointSliceGetMutExclusive,
            TrustedDeviceItem::DisjointSliceGetBlockMut,
            TrustedDeviceItem::DisjointSliceGetTiled2DMut,
            TrustedDeviceItem::DisjointSliceGetRowStriped2DMut,
            TrustedDeviceItem::DisjointSliceGetMutAt,
            TrustedDeviceItem::DisjointSliceLen,
            TrustedDeviceItem::WriteOnlyDisjointSliceLen,
            TrustedDeviceItem::WriteOnlyDisjointSliceWrite,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteDisjoint,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteExclusive,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteBlock,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteTiled2D,
            TrustedDeviceItem::WriteOnlyDisjointSliceWriteRowStriped2D,
            TrustedDeviceItem::DeviceGlobalMutPtrU32AsAtomic,
            TrustedDeviceItem::DeviceGlobalMutPtrI32AsAtomic,
            TrustedDeviceItem::DeviceGlobalMutPtrU64AsAtomic,
            TrustedDeviceItem::DeviceGlobalMutPtrI64AsAtomic,
            TrustedDeviceItem::MemoryOffsetFrom,
            TrustedDeviceItem::MemoryVolatileLoad,
            TrustedDeviceItem::MemoryVolatileStore,
            TrustedDeviceItem::MemoryCopyNonOverlapping,
            TrustedDeviceItem::MemoryCopyOneNonOverlapping,
            TrustedDeviceItem::WorkgroupGroup,
            TrustedDeviceItem::WorkgroupCollectiveScratch,
            TrustedDeviceItem::WorkgroupCollectivesContext,
            TrustedDeviceItem::WorkgroupCollectivesCurrent,
            TrustedDeviceItem::WorkgroupReduceSum,
            TrustedDeviceItem::WorkgroupInclusiveScanSum,
            TrustedDeviceItem::WorkgroupExclusiveScanSum,
            TrustedDeviceItem::Gfx942CollectivesContext,
            TrustedDeviceItem::Gfx942CollectivesCurrent,
            TrustedDeviceItem::Gfx942SubgroupReduceSumF32,
            TrustedDeviceItem::Gfx942SubgroupReduceMaxF32,
            TrustedDeviceItem::Gfx942StaticLdsU32x256,
            TrustedDeviceItem::Gfx942StaticLdsU32x256Type,
            TrustedDeviceItem::Gfx942Wave64ReduceActiveU32,
            TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::U32),
            TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::I32),
            TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::F32),
            TrustedDeviceItem::Gfx942Workgroup256ReduceActiveU32,
            TrustedDeviceItem::Gfx942Wave64ReduceSum,
            TrustedDeviceItem::Gfx942Wave64InclusiveScanSum,
            TrustedDeviceItem::Gfx942Wave64ExclusiveScanSum,
            TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper,
            TrustedDeviceItem::Gfx942WorkgroupReduceSum,
            TrustedDeviceItem::Gfx942WorkgroupInclusiveScanSum,
            TrustedDeviceItem::Gfx942WorkgroupExclusiveScanSum,
            TrustedDeviceItem::Gfx942BarrierArrive,
            TrustedDeviceItem::Gfx942BarrierWait,
            TrustedDeviceItem::WaveLane,
            TrustedDeviceItem::Wave64,
            TrustedDeviceItem::WaveLaneCurrent,
            TrustedDeviceItem::LdsTile16x16WriteMfmaBf16,
            TrustedDeviceItem::LdsTile16x16ReadMfmaBf16,
            TrustedDeviceItem::WorkgroupSyncthreads,
            TrustedDeviceItem::DeviceMatrix,
            TrustedDeviceItem::DeviceMatrixCurrent,
            TrustedDeviceItem::Bf16MfmaProfile,
            TrustedDeviceItem::MfmaOperandA,
            TrustedDeviceItem::MfmaOperandB,
            TrustedDeviceItem::MfmaRegisterTile16x16,
            TrustedDeviceItem::MfmaLdsXor4Storage,
            TrustedDeviceItem::MfmaAccumulatorRowMajor,
            TrustedDeviceItem::Bf16MfmaFragment,
            TrustedDeviceItem::F32AccumulatorFragment,
            TrustedDeviceItem::F32AccumulatorFragmentZero,
            TrustedDeviceItem::F32AccumulatorFragmentIntoValues,
            TrustedDeviceItem::Bf16MfmaMatrixView,
            TrustedDeviceItem::Bf16MfmaMatrixViewError,
            TrustedDeviceItem::Bf16MfmaMatrixARowMajor,
            TrustedDeviceItem::Bf16MfmaMatrixBRowMajor,
            TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView,
            TrustedDeviceItem::Bf16MfmaBColumnMajor,
            TrustedDeviceItem::Bf16MfmaBColumnMajorLoadZeroFilledV1,
            TrustedDeviceItem::Bf16MfmaMatrixALoadZeroFilledV2,
            TrustedDeviceItem::Bf16MfmaMatrixBLoadZeroFilledV2,
            TrustedDeviceItem::DeviceMatrixMultiplyAccumulate,
            TrustedDeviceItem::Gfx950Fp4E2M1Format,
            TrustedDeviceItem::Gfx950Fp8E4M3Format,
            TrustedDeviceItem::Gfx950MfmaOperandA,
            TrustedDeviceItem::Gfx950MfmaOperandB,
            TrustedDeviceItem::Gfx950MfmaFragment,
            TrustedDeviceItem::Gfx950F32AccumulatorFragment,
            TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentZero,
            TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentIntoValues,
            TrustedDeviceItem::Gfx950F32AccumulatorFragmentZero,
            TrustedDeviceItem::Gfx950F32AccumulatorFragmentIntoValues,
            TrustedDeviceItem::Gfx950MfmaMatrixViewError,
            TrustedDeviceItem::Gfx950MfmaMatrixAView,
            TrustedDeviceItem::Gfx950MfmaMatrixBView,
            TrustedDeviceItem::Gfx950MfmaMatrixAFp4RowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixBFp4RowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixARowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixBRowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixAFp8LoadM16K128,
            TrustedDeviceItem::Gfx950MfmaMatrixBFp8LoadK128N16,
            TrustedDeviceItem::Gfx950MfmaMatrixAFp4LoadM16K128,
            TrustedDeviceItem::Gfx950MfmaMatrixBFp4LoadK128N16,
            TrustedDeviceItem::Gfx950Matrix,
            TrustedDeviceItem::Gfx950MatrixCurrent,
            TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4,
            TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4Fp8,
            TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp8,
            TrustedDeviceItem::Gfx950SubgroupContext,
            TrustedDeviceItem::Gfx950SubgroupCurrent,
            TrustedDeviceItem::Gfx950SubgroupReduceMaxF32,
            TrustedDeviceItem::Gfx950SubgroupReduceSumF32,
            TrustedDeviceItem::Gfx950SubgroupBroadcastF32,
            TrustedDeviceItem::Gfx950LdsTransposeTile,
            TrustedDeviceItem::Gfx950LdsTransposeTileCurrent,
            TrustedDeviceItem::Gfx950LdsTransposeStageB4,
            TrustedDeviceItem::Gfx950LdsTransposeStageB8,
            TrustedDeviceItem::Gfx950LdsTransposePublish,
            TrustedDeviceItem::Gfx950LdsTransposeReadB4,
            TrustedDeviceItem::Gfx950LdsTransposeReadB8,
            TrustedDeviceItem::AmdGpuOrderedXorAddE32,
            TrustedDeviceItem::AmdGpuOrderedProgramE32,
            TrustedDeviceItem::AmdGpuCompleteBodyE32,
            TrustedDeviceItem::AmdGpuPhysicalEntryBeginGfx942,
            TrustedDeviceItem::AmdGpuPhysicalEntryLabelGfx942,
            TrustedDeviceItem::AmdGpuPhysicalEntryStepGfx942,
            TrustedDeviceItem::AmdGpuPhysicalGlobalCopyBeginGfx942,
            TrustedDeviceItem::AmdGpuPhysicalGlobalCopyLabelGfx942,
            TrustedDeviceItem::AmdGpuPhysicalGlobalCopyStepGfx942,
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeBeginGfx942,
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeLabelGfx942,
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeStepGfx942,
            TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VMovB32),
            TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VAddU32),
            TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VSubU32),
            TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VAndB32),
            TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VOrB32),
            TrustedDeviceItem::AmdGpuInline(TrustedAmdGpuInlineOperation::VXorB32),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Print0),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Print1),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Print2),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::AssertFail),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Clock32),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::Trap),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::DebugTrap),
            TrustedDeviceItem::AmdGpuDiagnostic(TrustedAmdGpuDiagnosticOperation::ProfilingMarker),
        ];

        let paths = items.map(TrustedDeviceItem::canonical_path);
        assert_eq!(paths.len(), super::TRUSTED_ITEMS.len());
        for item in items {
            let matching = super::TRUSTED_ITEMS
                .iter()
                .filter(|(registered, _, _)| *registered == item)
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), 1, "registered identity {item:?}");
            assert_eq!(matching[0].2, item.canonical_path());
        }
        for (item, _, path) in super::TRUSTED_ITEMS {
            assert!(
                items.contains(item),
                "missing independent registry item {item:?}"
            );
            assert_eq!(*path, item.canonical_path());
        }
        for (index, path) in paths.iter().enumerate() {
            assert!(!path.is_empty());
            assert!(!paths[..index].contains(path));
        }

        let markers = super::TRUSTED_ITEMS
            .iter()
            .map(|(_, marker, _)| *marker)
            .collect::<Vec<_>>();
        for (index, marker) in markers.iter().enumerate() {
            assert!(!marker.is_empty());
            assert!(!markers[..index].contains(marker));
        }

        let half_math_items = [
            TrustedDeviceItem::DeviceValue(DeviceValueDiagnosticItem::F16),
            TrustedDeviceItem::DeviceValue(DeviceValueDiagnosticItem::Bf16),
            TrustedDeviceItem::DeviceValue(DeviceValueDiagnosticItem::Bf16x2),
            TrustedDeviceItem::DeviceMath(DeviceMathDiagnosticItem::Context),
            TrustedDeviceItem::DeviceMath(DeviceMathDiagnosticItem::ContextFromCompiler),
        ];
        for item in half_math_items {
            assert!(!item.canonical_path().is_empty());
        }

        for operation in [
            TrustedHalfOperation::FromBits(fe2o3_kernel_ir::NarrowFloatFormat::F16),
            TrustedHalfOperation::FromBits(fe2o3_kernel_ir::NarrowFloatFormat::Bf16),
            TrustedHalfOperation::ToBits(fe2o3_kernel_ir::NarrowFloatFormat::F16),
            TrustedHalfOperation::ToBits(fe2o3_kernel_ir::NarrowFloatFormat::Bf16),
            TrustedHalfOperation::FromF32(fe2o3_kernel_ir::NarrowFloatFormat::F16),
            TrustedHalfOperation::FromF32(fe2o3_kernel_ir::NarrowFloatFormat::Bf16),
            TrustedHalfOperation::ToF32(fe2o3_kernel_ir::NarrowFloatFormat::F16),
            TrustedHalfOperation::ToF32(fe2o3_kernel_ir::NarrowFloatFormat::Bf16),
        ] {
            assert!(
                !TrustedDeviceItem::HalfOperation(operation)
                    .canonical_path()
                    .is_empty()
            );
        }

        let markers = HALF_MATH_DIAGNOSTIC_ITEMS
            .iter()
            .map(|(marker, _)| *marker)
            .collect::<Vec<_>>();
        let paths = HALF_MATH_DIAGNOSTIC_ITEMS
            .iter()
            .map(|(_, path)| *path)
            .collect::<Vec<_>>();
        for index in 0..markers.len() {
            assert!(!markers[..index].contains(&markers[index]));
            assert!(!paths[..index].contains(&paths[index]));
        }
    }

    #[test]
    fn safe_execution_items_have_exact_structural_provider_paths() {
        let items = [
            TrustedDeviceItem::ExecutionWithWorkgroup,
            TrustedDeviceItem::WorkgroupLdsScope,
            TrustedDeviceItem::WorkgroupLdsScopeCurrent,
            TrustedDeviceItem::DynamicLdsExactCurrent,
            TrustedDeviceItem::DynamicLdsIntoCollectiveRawParts,
            TrustedDeviceItem::LdsUninitialized,
            TrustedDeviceItem::WorkgroupPipeline,
            TrustedDeviceItem::WorkgroupPipelineCurrent,
            TrustedDeviceItem::WorkgroupPipelineStage,
            TrustedDeviceItem::WorkgroupPipelineWrite,
            TrustedDeviceItem::WorkgroupPipelineCommit,
            TrustedDeviceItem::WorkgroupPipelineWait,
            TrustedDeviceItem::WorkgroupPipelineConsume,
            TrustedDeviceItem::WorkgroupPipelineRead,
            TrustedDeviceItem::WorkgroupPipelineDiscard,
            TrustedDeviceItem::WorkgroupPipelineRelease,
            TrustedDeviceItem::Invocation3D,
            TrustedDeviceItem::Invocation3DCurrent,
            TrustedDeviceItem::WorkgroupGroup,
            TrustedDeviceItem::WorkgroupCollectiveScratch,
            TrustedDeviceItem::WorkgroupCollectivesContext,
            TrustedDeviceItem::WorkgroupCollectivesCurrent,
            TrustedDeviceItem::WorkgroupReduceSum,
            TrustedDeviceItem::Gfx942CollectivesContext,
            TrustedDeviceItem::Gfx942CollectivesCurrent,
            TrustedDeviceItem::Gfx942SubgroupReduceSumF32,
            TrustedDeviceItem::Gfx942SubgroupReduceMaxF32,
            TrustedDeviceItem::Gfx942StaticLdsU32x256,
            TrustedDeviceItem::Gfx942Wave64ReduceActiveU32,
            TrustedDeviceItem::Gfx942Workgroup256ReduceActiveU32,
            TrustedDeviceItem::Gfx942Wave64ReduceSum,
            TrustedDeviceItem::Gfx942WorkgroupReduceSum,
            TrustedDeviceItem::WaveLaneCurrent,
            TrustedDeviceItem::LdsTile16x16WriteMfmaBf16,
            TrustedDeviceItem::LdsTile16x16ReadMfmaBf16,
            TrustedDeviceItem::WorkgroupSyncthreads,
            TrustedDeviceItem::DeviceMatrix,
            TrustedDeviceItem::DeviceMatrixCurrent,
            TrustedDeviceItem::Bf16MfmaMatrixALoadZeroFilledV2,
            TrustedDeviceItem::Bf16MfmaMatrixBLoadZeroFilledV2,
            TrustedDeviceItem::Bf16MfmaBColumnMajorMatrixView,
            TrustedDeviceItem::Bf16MfmaBColumnMajor,
            TrustedDeviceItem::Bf16MfmaBColumnMajorLoadZeroFilledV1,
            TrustedDeviceItem::DeviceMatrixMultiplyAccumulate,
            TrustedDeviceItem::Gfx950Fp4E2M1Format,
            TrustedDeviceItem::Gfx950Fp8E4M3Format,
            TrustedDeviceItem::Gfx950MfmaOperandA,
            TrustedDeviceItem::Gfx950MfmaOperandB,
            TrustedDeviceItem::Gfx950MfmaFragment,
            TrustedDeviceItem::Gfx950F32AccumulatorFragment,
            TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentZero,
            TrustedDeviceItem::Gfx950Fp4F32AccumulatorFragmentIntoValues,
            TrustedDeviceItem::Gfx950F32AccumulatorFragmentZero,
            TrustedDeviceItem::Gfx950F32AccumulatorFragmentIntoValues,
            TrustedDeviceItem::Gfx950MfmaMatrixViewError,
            TrustedDeviceItem::Gfx950MfmaMatrixAView,
            TrustedDeviceItem::Gfx950MfmaMatrixBView,
            TrustedDeviceItem::Gfx950MfmaMatrixAFp4RowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixBFp4RowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixARowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixBRowMajor,
            TrustedDeviceItem::Gfx950MfmaMatrixAFp8LoadM16K128,
            TrustedDeviceItem::Gfx950MfmaMatrixBFp8LoadK128N16,
            TrustedDeviceItem::Gfx950MfmaMatrixAFp4LoadM16K128,
            TrustedDeviceItem::Gfx950MfmaMatrixBFp4LoadK128N16,
            TrustedDeviceItem::Gfx950Matrix,
            TrustedDeviceItem::Gfx950MatrixCurrent,
            TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4,
            TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp4Fp8,
            TrustedDeviceItem::Gfx950MatrixMultiplyAccumulateFp8,
            TrustedDeviceItem::Gfx950SubgroupContext,
            TrustedDeviceItem::Gfx950SubgroupCurrent,
            TrustedDeviceItem::Gfx950SubgroupReduceMaxF32,
            TrustedDeviceItem::Gfx950SubgroupReduceSumF32,
            TrustedDeviceItem::Gfx950SubgroupBroadcastF32,
            TrustedDeviceItem::Gfx950LdsTransposeTile,
            TrustedDeviceItem::Gfx950LdsTransposeTileCurrent,
            TrustedDeviceItem::Gfx950LdsTransposeStageB4,
            TrustedDeviceItem::Gfx950LdsTransposeStageB8,
            TrustedDeviceItem::Gfx950LdsTransposePublish,
            TrustedDeviceItem::Gfx950LdsTransposeReadB4,
            TrustedDeviceItem::Gfx950LdsTransposeReadB8,
        ];
        let mut paths = BTreeSet::new();
        for item in items {
            assert!(safe_execution_provider_bound_item(item));
            let path = safe_execution_compiler_definition_path(item);
            assert!(path.starts_with("fe2o3_device::"));
            assert!(
                path.contains("::collective::")
                    || path.contains("::execution::")
                    || path.contains("::group::")
                    || path.contains("::lds::")
                    || path.contains("::gfx950::")
                    || path.contains("::sync::")
                    || path.contains("::tensor::")
                    || path.contains("::thread::")
                    || path.contains("::wave::"),
                "safe execution item retained only a public re-export path: {path}"
            );
            assert!(paths.insert(path), "duplicate provider DefPath: {path}");
        }
    }
}

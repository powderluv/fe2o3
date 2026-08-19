//! Read-only gfx942 AQL queue resource planning.
//!
//! This module turns bounded topology observations into exact geometry and
//! source-pinned ROCr policy observations. It performs no ioctl, allocation,
//! mmap, doorbell access, or queue operation.

use core::fmt;

use crate::topology::{ComputePartition, GfxTarget, HostTopologySnapshot, MemoryPartition};

pub const GFX942_QUEUE_PAGE_BYTES_V1: u64 = 4096;
pub const GFX942_AQL_PACKET_BYTES_V1: u32 = 64;
pub const GFX942_MIN_ROCR_RING_BYTES_V1: u32 = 4096;
pub const GFX942_MAX_ADMITTED_RING_BYTES_V1: u32 = 1 << 31;
pub const GFX942_EOP_BYTES_V1: u64 = 4096;
pub const GFX942_COUNTER_BYTES_V1: u64 = 8;
pub const GFX942_CONTROL_STACK_BYTES_PER_XCC_V1: u32 = 0x3000;
pub const GFX942_WORKGROUP_CONTEXT_BYTES_PER_XCC_V1: u32 = 0x161e000;
pub const GFX942_CONTEXT_SAVE_BYTES_PER_XCC_V1: u32 = 0x1621000;
pub const GFX942_DEBUG_BYTES_PER_XCC_V1: u32 = 0xbe00;
pub const GFX942_CONTEXT_SAVE_MAPPING_BYTES_V1: u64 = 0xb167000;
pub const GFX942_ROCR_SVM_ALIGNMENT_BYTES_V1: u64 = 2 * 1024 * 1024;

const EXPECTED_SIMD_COUNT: u32 = 1216;
const EXPECTED_SIMD_PER_CU: u32 = 4;
const EXPECTED_XCC_COUNT: u32 = 8;
const EXPECTED_ARRAY_COUNT: u32 = 32;
const EXPECTED_SIMD_ARRAYS_PER_ENGINE: u32 = 1;
const EXPECTED_LDS_KIB: u32 = 64;
const EXPECTED_MAX_WAVES_PER_SIMD: u32 = 8;
const EXPECTED_COMPUTE_QUEUE_COUNT: u32 = 24;

/// Canonical contract for read-only gfx942 queue-resource planning.
pub const GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1: &str = concat!(
    "profile=fe2o3-mi300x-gfx942-spx-nps1-topology-queue-resources-r4-v1\n",
    "device_profile_sha256=e12ea33b259666e7928612403109640b03b0d637b893a2c15b87d17a4211c8de\n",
    "device_profile_digest_role=compositional-prerequisite-identifier-only,no-device-token-or-xnack-evidence\n",
    "kfd_queue_output_schema_sha256=63753a9c0dcef0f69e0235b95b44fe6ce22cb5b0d1df6f60a971a5ed28f15904\n",
    "platform=linux-x86_64,kernel:6.8.0-124-generic,amdgpu:6.16.13,page:4096\n",
    "module_zst_sha256=e5a327a8f46459e07ee3f59cc991d16feee17103e199d39149823879b7fcff0b\n",
    "module_ko_sha256=61317154cee502ea97a74818879dff4b20abf8f074a2f4d19a94288e25d4ac3a\n",
    "module_srcversion=A6F143BEC60C0AFC3263226\n",
    "module_parameters=mes:0,sched_policy:0,cwsr_enable:1\n",
    "source.kfd_queue.c=fb4b2a5c9e6981222873bcd7aca7e9c1397cba8f1a6b33634d2a48d4427fe062\n",
    "source.kfd_process_queue_manager.c=8526e258824dbe145e4209cf0fed26463729234ba24369f39e3413e7e6e028db\n",
    "source.kfd_doorbell.c=de30437ee1ed9ccbdaf855899482c0bebb7f55adc120ac712c96cadef1a0ec6d\n",
    "source.kfd_device.c=ccf20227c5cdd5b258758f50f61bbc1008a09ea776c101f035f83963e7d23037\n",
    "source.kfd_mqd_manager_v9.c=21166e9dbe2a4c24cbcd6f9ff6193aa093230e91fbafc8b4ac4eee1465cd2c9e\n",
    "source.kfd_priv.h=f991330031c14725b2be0636ec1896ab530dc3d07d530ebd4f47efff97a82a99\n",
    "source.kfd_topology.h=0fc8804ee63263f3a6f36fe6d7a2907c98610cb9d7db3e33239775a4b315c3de\n",
    "rocr_commit=97f5574fe2fdc7bef44fb01545347912ee9f1779,tag:rocm-7.2.4\n",
    "rocr_binary_sha256=b8cdfe93d343649a35c1daf73a0a3a6840f09379ebeee9be65670461ffea43f4\n",
    "source.rocr.queues.c=b7ead541340ac996c2305b2e9660cb3176edcd61ee509d4880f02659fbb6f32b\n",
    "source.rocr.topology.c=97269f0baf231d490032fc47ea8fe9e1101232477e10f74ff15e616d8e54ad86\n",
    "source.rocr.fmm.c=a2addccabb82e0ca184eaaf722e976e254a898ccfc945d4d956c4e273e196aef\n",
    "source.rocr.memory.c=4376e4bc6980299efc0fb79cfa497d5758171980ce80b04632882537866e977a\n",
    "source.rocr.libhsakmt.h=f957d592df9541bef7d0e21b507c95f5046f2fb380da3d64525bc4770a5a1b93\n",
    "source.rocr.hsakamttypes.h=fd9e3e9a0874614e70e518ee420aacd2d171452c2755d05b2cf54b55144ec78e\n",
    "source.rocr.amd_aql_queue.cpp=291f2521e2a4758e852ed20c578aca79e379d1effe4dfd83c62e11347eef2b14\n",
    "source.rocr.amd_gpu_agent.cpp=c39d5f922e855ce57d3c1903beef325e6004431c2ee66ae000aac72a0e5999da\n",
    "source.rocr.amd_kfd_driver.cpp=c6f961251ebc0ceb3da5107964fa34bb5dacf0d3973a0e179fcb06cf5ca98cb3\n",
    "source.rocr.runtime.cpp=d54a0e36a3403c13f4af0b0fc6552dfcf24a2d42df7e36d23752cb1e00c11469\n",
    "source.rocr.amd_memory_region.cpp=37e11dd281156b80972c25cea9bd924beb0da1a2e6a2b55be0117955ea5249d3\n",
    "source.rocr.memory_region.h=5b7e6ff1ae24d61baf806b8bb33433b5462c8247555f1e5ba7ed944793072ddf\n",
    "source.rocr.amd_memory_region.h=7a28a882fc7b391079601b1ce78b612599440e52c1b0f6bba7ac38214c68b2e9\n",
    "target=gfx942:90402,SPX/NPS1,simd:1216,simd-per-cu:4,xcc:8,array:32,arrays-per-engine:1,lds-kib:64,max-waves-per-simd:8,cp-queues:24\n",
    "later_queue_authority_requirement=pair-with-live-checked-device-token-including-xnack-disabled-currentness\n",
    "ring=power-of-two:4096..2147483648,alignment:4096,packet:64,exact-mapping\n",
    "control=counter-width:8,counter-alignment:8,exact-page-mapping-per-pointer:4096\n",
    "eop=size:4096,alignment:4096,exact-mapping\n",
    "cwsr=ctl-per-xcc:12288,wg-per-xcc:23191552,ctx-per-xcc:23203840,debug-per-xcc:48640,xcc:8,mapping:186019840,kfd-min-align:4096,rocr-primary-svm-align:2097152,rocr-fallback-align:4096\n",
    "doorbell=width:8,process-slice:8192,exact-whole-slice-mmap-required,encoded-base-mask:8191-not-page-mask\n",
    "rocr_backing=ring:userptr-writable-executable-coherent:0xc4000004,control:userptr-writable-coherent:0x84000004,eop:vram-writable-executable:0xc0000001,cwsr:host-svm-host-access-gpu-exec-or-userptr-0xc4000004\n",
    "rocr_expression_scope=exact-reviewed-allocation-flags-and-svm-attribute-expressions,not-transitive-policy-implementation-closure,not-runtime-branch-attestation\n",
    "source_linkage=contracted,source-hashes-do-not-prove-loaded-binary\n",
    "authority=observation-and-planning-only,no-create,no-allocation,no-mmap,no-doorbell-store\n",
);

/// SHA-256 of GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1.
pub const GFX942_QUEUE_RESOURCE_PROFILE_SHA256_V1: &str =
    "b8317e4288e14c6d7546b53887ec2a10e1938ffba9595271d174a2a652320f4f";

/// Typed digest bytes of GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1.
pub const GFX942_QUEUE_RESOURCE_PROFILE_SHA256_BYTES_V1: [u8; 32] = [
    0xb8, 0x31, 0x7e, 0x42, 0x88, 0xe1, 0x4c, 0x6d, 0x75, 0x46, 0xb5, 0x38, 0x87, 0xec, 0x2a, 0x10,
    0xe1, 0x93, 0x8f, 0xfb, 0xa9, 0x59, 0x52, 0x71, 0xd1, 0x74, 0xa2, 0xa6, 0x52, 0x32, 0x0f, 0x4f,
];

/// Resource role names shared with the abstract queue lifecycle model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueResourceRoleV1 {
    Ring,
    Control,
    EndOfPipe,
    ContextSave,
}

/// Exact reviewed ROCr policy expression, not an admitted allocation kind.
///
/// These variants summarize the pinned expressions on the reviewed paths. They
/// are not a transitive implementation closure or evidence that a runtime
/// invocation selected those paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RocrQueueBackingPolicyV1 {
    UserptrWritableExecutableCoherent,
    UserptrWritableCoherent,
    VramWritableExecutable,
    HostSvmHostAccessGpuExecutable,
}

impl RocrQueueBackingPolicyV1 {
    /// Raw KFD flags yielded by the reviewed ALLOC_MEMORY expressions.
    ///
    /// None denotes the primary CWSR SVM-attribute path. These values are not
    /// accepted by fe2o3's current memory authority and grant no authority.
    pub const fn observed_kfd_alloc_flags(self) -> Option<u32> {
        match self {
            Self::UserptrWritableExecutableCoherent => Some(0xc400_0004),
            Self::UserptrWritableCoherent => Some(0x8400_0004),
            Self::VramWritableExecutable => Some(0xc000_0001),
            Self::HostSvmHostAccessGpuExecutable => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gfx942QueueTargetObservationV1 {
    unique_id: u64,
    gpu_id: u32,
    topology_generation: u64,
    mes: i32,
    sched_policy: i32,
    cwsr_enable: i32,
}

impl Gfx942QueueTargetObservationV1 {
    pub const fn unique_id(self) -> u64 {
        self.unique_id
    }

    pub const fn gpu_id(self) -> u32 {
        self.gpu_id
    }

    pub const fn topology_generation(self) -> u64 {
        self.topology_generation
    }

    pub const fn mes(self) -> i32 {
        self.mes
    }

    pub const fn sched_policy(self) -> i32 {
        self.sched_policy
    }

    pub const fn cwsr_enable(self) -> i32 {
        self.cwsr_enable
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingResourcePlanV1 {
    mapping_bytes: u32,
}

impl RingResourcePlanV1 {
    pub const fn role(self) -> QueueResourceRoleV1 {
        QueueResourceRoleV1::Ring
    }

    pub const fn mapping_bytes(self) -> u32 {
        self.mapping_bytes
    }

    pub const fn base_alignment_bytes(self) -> u64 {
        GFX942_QUEUE_PAGE_BYTES_V1
    }

    pub const fn packet_bytes(self) -> u32 {
        GFX942_AQL_PACKET_BYTES_V1
    }

    pub const fn rocr_backing_policy(self) -> RocrQueueBackingPolicyV1 {
        RocrQueueBackingPolicyV1::UserptrWritableExecutableCoherent
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControlResourcePlanV1;

impl ControlResourcePlanV1 {
    pub const fn role(self) -> QueueResourceRoleV1 {
        QueueResourceRoleV1::Control
    }

    /// Each rptr/wptr lookup must resolve to an exact one-page GPU mapping.
    pub const fn exact_mapping_bytes_per_pointer(self) -> u64 {
        GFX942_QUEUE_PAGE_BYTES_V1
    }

    pub const fn counter_bytes(self) -> u64 {
        GFX942_COUNTER_BYTES_V1
    }

    pub const fn counter_alignment_bytes(self) -> u64 {
        GFX942_COUNTER_BYTES_V1
    }

    pub const fn rocr_backing_policy(self) -> RocrQueueBackingPolicyV1 {
        RocrQueueBackingPolicyV1::UserptrWritableCoherent
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EndOfPipeResourcePlanV1;

impl EndOfPipeResourcePlanV1 {
    pub const fn role(self) -> QueueResourceRoleV1 {
        QueueResourceRoleV1::EndOfPipe
    }

    pub const fn mapping_bytes(self) -> u64 {
        GFX942_EOP_BYTES_V1
    }

    pub const fn base_alignment_bytes(self) -> u64 {
        GFX942_QUEUE_PAGE_BYTES_V1
    }

    pub const fn rocr_backing_policy(self) -> RocrQueueBackingPolicyV1 {
        RocrQueueBackingPolicyV1::VramWritableExecutable
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextSaveResourcePlanV1;

impl ContextSaveResourcePlanV1 {
    pub const fn role(self) -> QueueResourceRoleV1 {
        QueueResourceRoleV1::ContextSave
    }

    pub const fn control_stack_bytes_per_xcc(self) -> u32 {
        GFX942_CONTROL_STACK_BYTES_PER_XCC_V1
    }

    pub const fn context_save_bytes_per_xcc(self) -> u32 {
        GFX942_CONTEXT_SAVE_BYTES_PER_XCC_V1
    }

    pub const fn debug_bytes_per_xcc(self) -> u32 {
        GFX942_DEBUG_BYTES_PER_XCC_V1
    }

    pub const fn xcc_count(self) -> u32 {
        EXPECTED_XCC_COUNT
    }

    pub const fn mapping_bytes(self) -> u64 {
        GFX942_CONTEXT_SAVE_MAPPING_BYTES_V1
    }

    pub const fn kfd_minimum_base_alignment_bytes(self) -> u64 {
        GFX942_QUEUE_PAGE_BYTES_V1
    }

    pub const fn primary_rocr_base_alignment_bytes(self) -> u64 {
        GFX942_ROCR_SVM_ALIGNMENT_BYTES_V1
    }

    pub const fn fallback_rocr_base_alignment_bytes(self) -> u64 {
        GFX942_QUEUE_PAGE_BYTES_V1
    }

    pub const fn primary_rocr_backing_policy(self) -> RocrQueueBackingPolicyV1 {
        RocrQueueBackingPolicyV1::HostSvmHostAccessGpuExecutable
    }

    pub const fn fallback_rocr_backing_policy(self) -> RocrQueueBackingPolicyV1 {
        RocrQueueBackingPolicyV1::UserptrWritableExecutableCoherent
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DoorbellResourcePlanV1;

impl DoorbellResourcePlanV1 {
    pub const fn width_bytes(self) -> u64 {
        fe2o3_kfd_uapi::KFD_GFX942_DOORBELL_BYTES
    }

    pub const fn process_slice_bytes(self) -> u64 {
        fe2o3_kfd_uapi::KFD_GFX942_PROCESS_DOORBELL_SLICE_BYTES
    }
}

/// Complete read-only resource geometry for one selected target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct Gfx942AqlQueueResourcePlanV1 {
    target: Gfx942QueueTargetObservationV1,
    ring: RingResourcePlanV1,
}

impl Gfx942AqlQueueResourcePlanV1 {
    pub const fn target(self) -> Gfx942QueueTargetObservationV1 {
        self.target
    }

    pub const fn ring(self) -> RingResourcePlanV1 {
        self.ring
    }

    pub const fn control(self) -> ControlResourcePlanV1 {
        ControlResourcePlanV1
    }

    pub const fn end_of_pipe(self) -> EndOfPipeResourcePlanV1 {
        EndOfPipeResourcePlanV1
    }

    pub const fn context_save(self) -> ContextSaveResourcePlanV1 {
        ContextSaveResourcePlanV1
    }

    pub const fn doorbell(self) -> DoorbellResourcePlanV1 {
        DoorbellResourcePlanV1
    }

    pub const fn profile_sha256(self) -> &'static str {
        GFX942_QUEUE_RESOURCE_PROFILE_SHA256_V1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gfx942QueueResourcePlanningError {
    SelectedGpuNotFound {
        unique_id: u64,
    },
    CorrelatedRenderNotFound {
        unique_id: u64,
    },
    GpuIdOutOfRange {
        gpu_id: u64,
    },
    HostProfileMismatch {
        field: &'static str,
    },
    TargetMismatch,
    PartitionMismatch,
    CapacityMismatch {
        field: &'static str,
        expected: u32,
        observed: u32,
    },
    RingSizeUnsupported {
        bytes: u32,
    },
}

impl fmt::Display for Gfx942QueueResourcePlanningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Gfx942QueueResourcePlanningError {}

struct TargetFacts {
    unique_id: u64,
    gpu_id: u64,
    topology_generation: u64,
    target: GfxTarget,
    compute_partition: ComputePartition,
    memory_partition: MemoryPartition,
    simd_count: u32,
    simd_per_cu: u32,
    xcc_count: u32,
    array_count: u32,
    simd_arrays_per_engine: u32,
    lds_size_in_kb: u32,
    max_waves_per_simd: u32,
    compute_queue_count: u32,
}

/// Produces a resource plan from one exact read-only host topology snapshot.
///
/// The result is geometry and policy evidence only. In particular, it does not
/// prove that any allocation has the required type, flags, mapping, address,
/// ownership, lifetime, or currentness.
pub fn plan_gfx942_aql_queue_resources(
    snapshot: &HostTopologySnapshot,
    unique_id: u64,
    ring_bytes: u32,
) -> Result<Gfx942AqlQueueResourcePlanV1, Gfx942QueueResourcePlanningError> {
    if snapshot.kernel_release().as_str() != "6.8.0-124-generic" {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch {
            field: "kernel_release",
        });
    }
    if snapshot.amdgpu_module().version() != Some("6.16.13") {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch {
            field: "amdgpu_module_version",
        });
    }
    if snapshot.amdgpu_module().srcversion() != Some("A6F143BEC60C0AFC3263226") {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch {
            field: "amdgpu_module_srcversion",
        });
    }
    check_driver_parameters(
        snapshot.amdgpu_module().mes(),
        snapshot.amdgpu_module().sched_policy(),
        snapshot.amdgpu_module().cwsr_enable(),
    )?;
    if rustix::param::page_size() != GFX942_QUEUE_PAGE_BYTES_V1 as usize {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch { field: "page_size" });
    }

    let gpu = snapshot
        .topology()
        .gpu_nodes()
        .iter()
        .find(|gpu| gpu.unique_id() == unique_id)
        .ok_or(Gfx942QueueResourcePlanningError::SelectedGpuNotFound { unique_id })?;
    let render = snapshot
        .render_nodes()
        .iter()
        .find(|render| render.unique_id() == unique_id && render.node_id() == gpu.node_id())
        .ok_or(Gfx942QueueResourcePlanningError::CorrelatedRenderNotFound { unique_id })?;
    let capacity = gpu.capacity();
    plan_from_facts(
        TargetFacts {
            unique_id,
            gpu_id: gpu.gpu_id(),
            topology_generation: snapshot.topology().provenance().generation(),
            target: gpu.target(),
            compute_partition: render.partition().compute(),
            memory_partition: render.partition().memory(),
            simd_count: capacity.simd_count(),
            simd_per_cu: capacity.simd_per_cu(),
            xcc_count: capacity.xcc_count(),
            array_count: capacity.array_count(),
            simd_arrays_per_engine: capacity.simd_arrays_per_engine(),
            lds_size_in_kb: capacity.lds_size_in_kb(),
            max_waves_per_simd: capacity.max_waves_per_simd(),
            compute_queue_count: capacity.compute_queue_count(),
        },
        ring_bytes,
    )
}

fn check_driver_parameters(
    mes: Option<i32>,
    sched_policy: Option<i32>,
    cwsr_enable: Option<i32>,
) -> Result<(), Gfx942QueueResourcePlanningError> {
    if mes != Some(0) {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch {
            field: "amdgpu_mes",
        });
    }
    if sched_policy != Some(0) {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch {
            field: "amdgpu_sched_policy",
        });
    }
    if cwsr_enable != Some(1) {
        return Err(Gfx942QueueResourcePlanningError::HostProfileMismatch {
            field: "amdgpu_cwsr_enable",
        });
    }
    Ok(())
}

fn plan_from_facts(
    facts: TargetFacts,
    ring_bytes: u32,
) -> Result<Gfx942AqlQueueResourcePlanV1, Gfx942QueueResourcePlanningError> {
    if facts.target != GfxTarget::Gfx942 {
        return Err(Gfx942QueueResourcePlanningError::TargetMismatch);
    }
    if facts.compute_partition != ComputePartition::Spx
        || facts.memory_partition != MemoryPartition::Nps1
    {
        return Err(Gfx942QueueResourcePlanningError::PartitionMismatch);
    }
    check_capacity("simd_count", EXPECTED_SIMD_COUNT, facts.simd_count)?;
    check_capacity("simd_per_cu", EXPECTED_SIMD_PER_CU, facts.simd_per_cu)?;
    check_capacity("num_xcc", EXPECTED_XCC_COUNT, facts.xcc_count)?;
    check_capacity("array_count", EXPECTED_ARRAY_COUNT, facts.array_count)?;
    check_capacity(
        "simd_arrays_per_engine",
        EXPECTED_SIMD_ARRAYS_PER_ENGINE,
        facts.simd_arrays_per_engine,
    )?;
    check_capacity("lds_size_in_kb", EXPECTED_LDS_KIB, facts.lds_size_in_kb)?;
    check_capacity(
        "max_waves_per_simd",
        EXPECTED_MAX_WAVES_PER_SIMD,
        facts.max_waves_per_simd,
    )?;
    check_capacity(
        "num_cp_queues",
        EXPECTED_COMPUTE_QUEUE_COUNT,
        facts.compute_queue_count,
    )?;

    if !(GFX942_MIN_ROCR_RING_BYTES_V1..=GFX942_MAX_ADMITTED_RING_BYTES_V1).contains(&ring_bytes)
        || !ring_bytes.is_power_of_two()
    {
        return Err(Gfx942QueueResourcePlanningError::RingSizeUnsupported { bytes: ring_bytes });
    }
    let gpu_id = u32::try_from(facts.gpu_id).map_err(|_| {
        Gfx942QueueResourcePlanningError::GpuIdOutOfRange {
            gpu_id: facts.gpu_id,
        }
    })?;

    Ok(Gfx942AqlQueueResourcePlanV1 {
        target: Gfx942QueueTargetObservationV1 {
            unique_id: facts.unique_id,
            gpu_id,
            topology_generation: facts.topology_generation,
            mes: 0,
            sched_policy: 0,
            cwsr_enable: 1,
        },
        ring: RingResourcePlanV1 {
            mapping_bytes: ring_bytes,
        },
    })
}

fn check_capacity(
    field: &'static str,
    expected: u32,
    observed: u32,
) -> Result<(), Gfx942QueueResourcePlanningError> {
    if observed == expected {
        Ok(())
    } else {
        Err(Gfx942QueueResourcePlanningError::CapacityMismatch {
            field,
            expected,
            observed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    type CapacityMutation = (&'static str, fn(&mut TargetFacts));

    fn valid_facts() -> TargetFacts {
        TargetFacts {
            unique_id: 0x1234,
            gpu_id: 73,
            topology_generation: 17,
            target: GfxTarget::Gfx942,
            compute_partition: ComputePartition::Spx,
            memory_partition: MemoryPartition::Nps1,
            simd_count: 1216,
            simd_per_cu: 4,
            xcc_count: 8,
            array_count: 32,
            simd_arrays_per_engine: 1,
            lds_size_in_kb: 64,
            max_waves_per_simd: 8,
            compute_queue_count: 24,
        }
    }

    #[test]
    fn exact_gfx942_plan_matches_reviewed_formula() {
        let plan = plan_from_facts(valid_facts(), 4096).unwrap();
        assert_eq!(plan.target().unique_id(), 0x1234);
        assert_eq!(plan.target().gpu_id(), 73);
        assert_eq!(plan.target().topology_generation(), 17);
        assert_eq!(plan.target().mes(), 0);
        assert_eq!(plan.target().sched_policy(), 0);
        assert_eq!(plan.target().cwsr_enable(), 1);
        assert_eq!(plan.ring().mapping_bytes(), 4096);
        assert_eq!(plan.ring().packet_bytes(), 64);
        assert_eq!(plan.control().exact_mapping_bytes_per_pointer(), 4096);
        assert_eq!(plan.control().counter_bytes(), 8);
        assert_eq!(plan.end_of_pipe().mapping_bytes(), 4096);
        assert_eq!(plan.context_save().control_stack_bytes_per_xcc(), 0x3000);
        assert_eq!(plan.context_save().context_save_bytes_per_xcc(), 0x1621000);
        assert_eq!(plan.context_save().debug_bytes_per_xcc(), 0xbe00);
        assert_eq!(plan.context_save().mapping_bytes(), 0xb167000);
        assert_eq!(plan.context_save().kfd_minimum_base_alignment_bytes(), 4096);
        assert_eq!(
            plan.context_save().primary_rocr_base_alignment_bytes(),
            2 * 1024 * 1024
        );
        assert_eq!(
            plan.context_save().fallback_rocr_base_alignment_bytes(),
            4096
        );
        assert_eq!(plan.doorbell().width_bytes(), 8);
        assert_eq!(plan.doorbell().process_slice_bytes(), 8192);
    }

    #[test]
    fn backing_profiles_are_observations_not_current_memory_admission() {
        let plan = plan_from_facts(valid_facts(), 1 << 20).unwrap();
        assert_eq!(
            plan.ring().rocr_backing_policy().observed_kfd_alloc_flags(),
            Some(0xc400_0004)
        );
        assert_eq!(
            plan.control()
                .rocr_backing_policy()
                .observed_kfd_alloc_flags(),
            Some(0x8400_0004)
        );
        assert_eq!(
            plan.end_of_pipe()
                .rocr_backing_policy()
                .observed_kfd_alloc_flags(),
            Some(0xc000_0001)
        );
        assert_eq!(
            plan.context_save()
                .primary_rocr_backing_policy()
                .observed_kfd_alloc_flags(),
            None
        );
        for flags in [0xc400_0004, 0x8400_0004, 0xc000_0001] {
            assert!(fe2o3_kfd_uapi::admit_kfd_alloc_memory_flags(flags).is_err());
        }
    }

    #[test]
    fn hostile_ring_sizes_are_rejected() {
        for bytes in [0, 1024, 4095, 4097, 1 << 30 | 4096, u32::MAX] {
            assert_eq!(
                plan_from_facts(valid_facts(), bytes),
                Err(Gfx942QueueResourcePlanningError::RingSizeUnsupported { bytes })
            );
        }
        assert!(plan_from_facts(valid_facts(), 1 << 31).is_ok());
    }

    #[test]
    fn every_capacity_input_is_checked_exactly() {
        let mutations: [CapacityMutation; 8] = [
            ("simd_count", |facts| facts.simd_count = 1215),
            ("simd_per_cu", |facts| facts.simd_per_cu = 8),
            ("num_xcc", |facts| facts.xcc_count = 7),
            ("array_count", |facts| facts.array_count = 31),
            ("simd_arrays_per_engine", |facts| {
                facts.simd_arrays_per_engine = 2
            }),
            ("lds_size_in_kb", |facts| facts.lds_size_in_kb = 32),
            ("max_waves_per_simd", |facts| facts.max_waves_per_simd = 7),
            ("num_cp_queues", |facts| facts.compute_queue_count = 23),
        ];
        for (field, mutate) in mutations {
            let mut facts = valid_facts();
            mutate(&mut facts);
            assert!(matches!(
                plan_from_facts(facts, 4096),
                Err(Gfx942QueueResourcePlanningError::CapacityMismatch {
                    field: observed_field,
                    ..
                }) if observed_field == field
            ));
        }
    }

    #[test]
    fn selector_partition_and_gpu_id_fail_closed() {
        let mut facts = valid_facts();
        facts.compute_partition = ComputePartition::Cpx;
        assert_eq!(
            plan_from_facts(facts, 4096),
            Err(Gfx942QueueResourcePlanningError::PartitionMismatch)
        );

        let mut facts = valid_facts();
        facts.gpu_id = u64::from(u32::MAX) + 1;
        assert_eq!(
            plan_from_facts(facts, 4096),
            Err(Gfx942QueueResourcePlanningError::GpuIdOutOfRange {
                gpu_id: u64::from(u32::MAX) + 1
            })
        );
    }

    #[test]
    fn missing_or_changed_driver_parameters_fail_closed() {
        assert!(check_driver_parameters(Some(0), Some(0), Some(1)).is_ok());
        for (mes, sched_policy, cwsr_enable, field) in [
            (None, Some(0), Some(1), "amdgpu_mes"),
            (Some(1), Some(0), Some(1), "amdgpu_mes"),
            (Some(0), None, Some(1), "amdgpu_sched_policy"),
            (Some(0), Some(1), Some(1), "amdgpu_sched_policy"),
            (Some(0), Some(0), None, "amdgpu_cwsr_enable"),
            (Some(0), Some(0), Some(0), "amdgpu_cwsr_enable"),
        ] {
            assert_eq!(
                check_driver_parameters(mes, sched_policy, cwsr_enable),
                Err(Gfx942QueueResourcePlanningError::HostProfileMismatch { field })
            );
        }
    }

    #[test]
    fn profile_digest_and_composition_are_frozen() {
        let digest = Sha256::digest(GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1);
        assert_eq!(hex(&digest), GFX942_QUEUE_RESOURCE_PROFILE_SHA256_V1);
        assert_eq!(&digest[..], &GFX942_QUEUE_RESOURCE_PROFILE_SHA256_BYTES_V1);
        assert!(
            GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1
                .contains(fe2o3_kfd_uapi::KFD_GFX942_QUEUE_RESOURCE_SCHEMA_MANIFEST_SHA256)
        );
        assert_eq!(
            crate::DEVICE_ADMISSION_PROFILE_SHA256_V1,
            "e12ea33b259666e7928612403109640b03b0d637b893a2c15b87d17a4211c8de"
        );
        assert!(
            GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1
                .contains(crate::DEVICE_ADMISSION_PROFILE_SHA256_V1)
        );
        assert!(GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1.contains(
            "device_profile_digest_role=compositional-prerequisite-identifier-only,no-device-token-or-xnack-evidence"
        ));
        assert!(GFX942_QUEUE_RESOURCE_PROFILE_MANIFEST_V1.contains(
            "source.rocr.libhsakmt.h=f957d592df9541bef7d0e21b507c95f5046f2fb380da3d64525bc4770a5a1b93"
        ));
    }

    fn hex(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            use std::fmt::Write;
            write!(&mut result, "{byte:02x}").unwrap();
        }
        result
    }
}

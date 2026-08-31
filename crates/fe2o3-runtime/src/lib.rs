#![deny(unsafe_code)]
#![doc = include_str!("../README.md")]

#[allow(unsafe_code)]
mod authorized_execution;

pub use authorized_execution::{
    AuthorizedRuntimeDebugTelemetrySessionV1, Gfx942AuthorizedRuntimeCompletedBufferV1,
    Gfx942AuthorizedRuntimeDispatchResultV1, Gfx942AuthorizedRuntimeExecutionErrorV1,
    WorkerV3Gfx942ExecutionAuthorityV1, execute_authorized_gfx942_runtime_debug_target_dispatch_v1,
    execute_authorized_gfx942_runtime_dispatch_v1,
};

use core::fmt;

use fe2o3_amdhsa_loader::{
    AdmittedProfile, KernelClosureError, KernelIdentityInputsV1, PlanError,
    SelectedKernelResourceBindingV1, validate,
};
use fe2o3_aql::AqlDispatchGeometryV1;
use fe2o3_hsaco::{HiddenArgument, HiddenValueKind};
use fe2o3_kfd::{
    GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_SHA256_V1, Gfx942KfdDispatchBufferV1,
    Gfx942KfdDispatchPointerFixupV1, Gfx942KfdDispatchRequestErrorV1, Gfx942KfdDispatchRequestV1,
};
use sha2::{Digest, Sha256};

const COV6_IMPLICIT_KERNARG_BYTES_V1: usize = 256;
const DIRECT_KFD_KERNARG_ALIGNMENT_V1: u64 = 16;
const GFX942_WAVEFRONT_SIZE_V1: u32 = 64;
const GFX942_MAX_GROUP_SEGMENT_BYTES_V1: u64 = 64 * 1024;
const GFX942_RUNTIME_DISPATCH_CONTRACT_DOMAIN_V1: &[u8] =
    b"fe2o3.runtime.gfx942-dispatch-contract.v1\0";
const NON_NULL_EMPTY_SLICE_SENTINEL_DIGEST_LENGTH_V1: u64 = u64::MAX;
const NON_NULL_EMPTY_SLICE_SENTINEL_ALLOCATION_BYTES_V1: usize = 1;

#[derive(Clone, Copy)]
struct Gfx942RuntimeKernelIdentityProjectionV1 {
    object_sha256: [u8; 32],
    metadata_sha256: [u8; 32],
    descriptor_sha256: [u8; 32],
    entry_sha256: [u8; 32],
    closure_sha256: [u8; 32],
}

impl From<KernelIdentityInputsV1> for Gfx942RuntimeKernelIdentityProjectionV1 {
    fn from(identity: KernelIdentityInputsV1) -> Self {
        Self {
            object_sha256: identity.object_sha256(),
            metadata_sha256: identity.metadata_sha256(),
            descriptor_sha256: identity.descriptor_sha256(),
            entry_sha256: identity.entry_sha256(),
            closure_sha256: identity.closure_sha256(),
        }
    }
}

/// Worker V3 memory effect assigned to one address-free KFD dispatch buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Gfx942RuntimeBufferAccessV1 {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl Gfx942RuntimeBufferAccessV1 {
    const fn canonical_tag(self) -> u8 {
        match self {
            Self::ReadOnly => 1,
            Self::WriteOnly => 2,
            Self::ReadWrite => 3,
        }
    }
}

/// Owned initial bytes and declared effect for one address-free runtime buffer.
#[derive(Debug, Eq, PartialEq)]
pub struct Gfx942RuntimeDispatchBufferV1 {
    buffer: Gfx942KfdDispatchBufferV1,
    access: Gfx942RuntimeBufferAccessV1,
    kind: Gfx942RuntimeDispatchBufferKindV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gfx942RuntimeDispatchBufferKindV1 {
    LogicalBytes,
    NonNullEmptySliceSentinel,
}

pub(crate) struct Gfx942RuntimePreparedBufferPolicyV1 {
    access: Gfx942RuntimeBufferAccessV1,
    allocation_byte_length: u64,
    read_only_initial_bytes: Option<Vec<u8>>,
    kind: Gfx942RuntimeDispatchBufferKindV1,
}

impl Gfx942RuntimePreparedBufferPolicyV1 {
    pub(crate) const fn access(&self) -> Gfx942RuntimeBufferAccessV1 {
        self.access
    }

    pub(crate) const fn byte_length(&self) -> u64 {
        self.allocation_byte_length
    }

    pub(crate) fn read_only_initial_bytes(&self) -> Option<&[u8]> {
        self.read_only_initial_bytes.as_deref()
    }

    pub(crate) const fn is_non_null_empty_slice_sentinel_v1(&self) -> bool {
        matches!(
            self.kind,
            Gfx942RuntimeDispatchBufferKindV1::NonNullEmptySliceSentinel
        )
    }
}

impl Gfx942RuntimeDispatchBufferV1 {
    pub fn new(
        bytes: Vec<u8>,
        access: Gfx942RuntimeBufferAccessV1,
    ) -> Result<Self, Gfx942KfdDispatchRequestErrorV1> {
        Ok(Self {
            buffer: Gfx942KfdDispatchBufferV1::new(bytes)?,
            access,
            kind: Gfx942RuntimeDispatchBufferKindV1::LogicalBytes,
        })
    }

    /// Creates the private transport allocation used to give a zero-length generated slice a
    /// non-null device address. The allocation byte is not logical slice capacity and is always
    /// read-only so any unexpected device write fails completion validation.
    #[doc(hidden)]
    pub fn new_non_null_empty_slice_sentinel_v1() -> Result<Self, Gfx942KfdDispatchRequestErrorV1> {
        Ok(Self {
            buffer: Gfx942KfdDispatchBufferV1::new(vec![
                0;
                NON_NULL_EMPTY_SLICE_SENTINEL_ALLOCATION_BYTES_V1
            ])?,
            access: Gfx942RuntimeBufferAccessV1::ReadOnly,
            kind: Gfx942RuntimeDispatchBufferKindV1::NonNullEmptySliceSentinel,
        })
    }

    /// Returns only logical buffer bytes. A non-null empty-slice sentinel has no logical bytes.
    pub fn bytes(&self) -> &[u8] {
        match self.kind {
            Gfx942RuntimeDispatchBufferKindV1::LogicalBytes => self.buffer.bytes(),
            Gfx942RuntimeDispatchBufferKindV1::NonNullEmptySliceSentinel => &[],
        }
    }

    pub const fn access(&self) -> Gfx942RuntimeBufferAccessV1 {
        self.access
    }

    #[doc(hidden)]
    pub const fn is_non_null_empty_slice_sentinel_v1(&self) -> bool {
        matches!(
            self.kind,
            Gfx942RuntimeDispatchBufferKindV1::NonNullEmptySliceSentinel
        )
    }

    fn allocation_bytes(&self) -> &[u8] {
        self.buffer.bytes()
    }

    fn into_kfd_buffer(self) -> Gfx942KfdDispatchBufferV1 {
        self.buffer
    }
}

/// Complete caller-owned data needed before loader and ABI admission.
#[must_use]
pub struct Gfx942RuntimeDispatchInputsV1 {
    explicit_kernarg: Vec<u8>,
    buffers: Vec<Gfx942RuntimeDispatchBufferV1>,
    pointer_fixups: Vec<Gfx942KfdDispatchPointerFixupV1>,
    geometry: AqlDispatchGeometryV1,
    dynamic_group_segment_bytes: u32,
    timeout_milliseconds: u32,
}

impl Gfx942RuntimeDispatchInputsV1 {
    pub fn new(
        explicit_kernarg: Vec<u8>,
        buffers: Vec<Gfx942RuntimeDispatchBufferV1>,
        pointer_fixups: Vec<Gfx942KfdDispatchPointerFixupV1>,
        geometry: AqlDispatchGeometryV1,
        dynamic_group_segment_bytes: u32,
        timeout_milliseconds: u32,
    ) -> Self {
        Self {
            explicit_kernarg,
            buffers,
            pointer_fixups,
            geometry,
            dynamic_group_segment_bytes,
            timeout_milliseconds,
        }
    }
}

/// Loader-bound request plus exact immutable object and selected-kernel identities.
///
/// This value is not launch authority. Its only transition yields the unsafe
/// KFD mechanics request consumed later by the Worker V3 runtime gate.
#[must_use = "preparation does not execute or authorize the selected kernel"]
pub struct PreparedGfx942RuntimeDispatchV1 {
    request: Gfx942KfdDispatchRequestV1,
    buffer_policies: Vec<Gfx942RuntimePreparedBufferPolicyV1>,
    identity: KernelIdentityInputsV1,
    finalized_hsaco_length: u64,
    dispatch_contract_sha256: [u8; 32],
    kernel_name: String,
    descriptor_offset: u64,
    static_group_segment_bytes: u64,
    dynamic_group_segment_bytes: u32,
    packet_group_segment_bytes: u32,
    geometry: AqlDispatchGeometryV1,
}

impl fmt::Debug for PreparedGfx942RuntimeDispatchV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedGfx942RuntimeDispatchV1")
            .field("finalized_hsaco_length", &self.finalized_hsaco_length)
            .field("dispatch_contract_sha256", &self.dispatch_contract_sha256)
            .field("kernel_name", &self.kernel_name)
            .field("descriptor_offset", &self.descriptor_offset)
            .field(
                "static_group_segment_bytes",
                &self.static_group_segment_bytes,
            )
            .field(
                "dynamic_group_segment_bytes",
                &self.dynamic_group_segment_bytes,
            )
            .field(
                "packet_group_segment_bytes",
                &self.packet_group_segment_bytes,
            )
            .finish_non_exhaustive()
    }
}

impl PreparedGfx942RuntimeDispatchV1 {
    pub const fn identity(&self) -> KernelIdentityInputsV1 {
        self.identity
    }

    pub const fn finalized_hsaco_length(&self) -> u64 {
        self.finalized_hsaco_length
    }

    /// Canonical identity of the complete address-free invocation presented to KFD.
    ///
    /// This binds the finalized object, selected kernel closure, materialized image,
    /// descriptor offset, complete kernarg template, initial buffer bytes, pointer fixups,
    /// geometry, resource sizes, and timeout. It is descriptive until an admitting Worker V3
    /// authority independently names the same identity.
    pub const fn dispatch_contract_sha256(&self) -> [u8; 32] {
        self.dispatch_contract_sha256
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub const fn descriptor_offset(&self) -> u64 {
        self.descriptor_offset
    }

    pub const fn static_group_segment_bytes(&self) -> u64 {
        self.static_group_segment_bytes
    }

    pub const fn dynamic_group_segment_bytes(&self) -> u32 {
        self.dynamic_group_segment_bytes
    }

    pub const fn packet_group_segment_bytes(&self) -> u32 {
        self.packet_group_segment_bytes
    }

    pub(crate) const fn geometry(&self) -> AqlDispatchGeometryV1 {
        self.geometry
    }

    /// Returns the mechanics-only request. Calling its KFD execution function
    /// still requires the complete unsafe Worker V3 contract.
    pub fn into_unchecked_kfd_request(self) -> Gfx942KfdDispatchRequestV1 {
        self.request
    }

    pub(crate) fn into_authorized_execution_parts(
        self,
    ) -> (
        Gfx942KfdDispatchRequestV1,
        Vec<Gfx942RuntimePreparedBufferPolicyV1>,
    ) {
        (self.request, self.buffer_policies)
    }
}

/// Failure before native device mutation.
#[derive(Debug)]
#[non_exhaustive]
pub enum Gfx942RuntimePreparationErrorV1 {
    Envelope(PlanError),
    Kernel(KernelClosureError),
    ImageSize,
    DescriptorRange,
    UnsupportedResource(&'static str),
    WorkgroupMismatch,
    WorkgroupCountExceeded { axis: usize },
    KernargLayout,
    HiddenArgument { index: usize, detail: &'static str },
    KfdRequest(Gfx942KfdDispatchRequestErrorV1),
}

impl fmt::Display for Gfx942RuntimePreparationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Gfx942RuntimePreparationErrorV1 {}

impl From<PlanError> for Gfx942RuntimePreparationErrorV1 {
    fn from(value: PlanError) -> Self {
        Self::Envelope(value)
    }
}

impl From<KernelClosureError> for Gfx942RuntimePreparationErrorV1 {
    fn from(value: KernelClosureError) -> Self {
        Self::Kernel(value)
    }
}

impl From<Gfx942KfdDispatchRequestErrorV1> for Gfx942RuntimePreparationErrorV1 {
    fn from(value: Gfx942KfdDispatchRequestErrorV1) -> Self {
        Self::KfdRequest(value)
    }
}

/// Validates and materializes one exact COV6 kernel into an address-free KFD request.
///
/// The operation checks the complete object and selected descriptor, derives
/// resource fields from the closure, and initializes every declared hidden
/// argument. It performs no KFD operation and grants no execution authority.
pub fn prepare_gfx942_runtime_dispatch_v1(
    hsaco: &[u8],
    kernel_name: &str,
    inputs: Gfx942RuntimeDispatchInputsV1,
) -> Result<PreparedGfx942RuntimeDispatchV1, Gfx942RuntimePreparationErrorV1> {
    let closure = validate(hsaco, AdmittedProfile::Gfx942XnackOffCov6)?.bind_kernel(kernel_name)?;
    let resources = closure.resources();
    validate_resources(
        resources,
        inputs.geometry,
        inputs.dynamic_group_segment_bytes,
    )?;

    let kernel = closure.selected_kernel();
    let total_kernarg = usize::try_from(resources.kernarg_segment_size())
        .map_err(|_| Gfx942RuntimePreparationErrorV1::KernargLayout)?;
    let explicit_kernarg = inputs.explicit_kernarg.len();
    let implicit_offset = kernel
        .implicit_argument_offset()
        .map(usize::try_from)
        .transpose()
        .map_err(|_| Gfx942RuntimePreparationErrorV1::KernargLayout)?;
    let implicit_size = usize::try_from(kernel.implicit_argument_size())
        .map_err(|_| Gfx942RuntimePreparationErrorV1::KernargLayout)?;
    if implicit_offset != Some(explicit_kernarg)
        || implicit_size != COV6_IMPLICIT_KERNARG_BYTES_V1
        || explicit_kernarg
            .checked_add(implicit_size)
            .is_none_or(|bytes| bytes != total_kernarg)
    {
        return Err(Gfx942RuntimePreparationErrorV1::KernargLayout);
    }
    let mut kernarg = vec![0_u8; total_kernarg];
    kernarg[..explicit_kernarg].copy_from_slice(&inputs.explicit_kernarg);
    initialize_hidden_arguments(
        &mut kernarg,
        kernel.hidden_arguments(),
        inputs.geometry,
        inputs.dynamic_group_segment_bytes,
    )?;

    let plan = closure.envelope().plan();
    let image_bytes = usize::try_from(closure.envelope().materialization().image_len())
        .map_err(|_| Gfx942RuntimePreparationErrorV1::ImageSize)?;
    let mut image = vec![0_u8; image_bytes];
    closure
        .materialize_into(&mut image)
        .map_err(|_| Gfx942RuntimePreparationErrorV1::ImageSize)?;
    let binding = closure.selected_binding();
    let descriptor_offset = binding
        .descriptor_address()
        .checked_sub(plan.image_start())
        .ok_or(Gfx942RuntimePreparationErrorV1::DescriptorRange)?;
    if descriptor_offset
        .checked_add(64)
        .is_none_or(|end| end > plan.image_end() - plan.image_start())
    {
        return Err(Gfx942RuntimePreparationErrorV1::DescriptorRange);
    }

    let kernarg_alignment = resources
        .kernarg_segment_alignment()
        .max(DIRECT_KFD_KERNARG_ALIGNMENT_V1);
    let identity = closure.identity_inputs();
    let static_group_segment_bytes = resources.group_segment_fixed_size();
    // AQL carries the complete per-workgroup allocation, whereas the COV6
    // hidden field (when present) carries only the dynamic contribution.
    let packet_group_segment_bytes = static_group_segment_bytes
        .checked_add(u64::from(inputs.dynamic_group_segment_bytes))
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or(Gfx942RuntimePreparationErrorV1::UnsupportedResource(
            "total group segment representation",
        ))?;
    let finalized_hsaco_length =
        u64::try_from(hsaco.len()).map_err(|_| Gfx942RuntimePreparationErrorV1::ImageSize)?;
    let dispatch_contract_sha256 = derive_dispatch_contract_sha256_v1(
        finalized_hsaco_length,
        identity.into(),
        kernel_name,
        &image,
        descriptor_offset,
        &kernarg,
        kernarg_alignment,
        &inputs.buffers,
        &inputs.pointer_fixups,
        inputs.geometry,
        packet_group_segment_bytes,
        inputs.timeout_milliseconds,
    );
    let buffer_policies = inputs
        .buffers
        .iter()
        .map(|buffer| Gfx942RuntimePreparedBufferPolicyV1 {
            access: buffer.access(),
            allocation_byte_length: u64::try_from(buffer.allocation_bytes().len())
                .expect("validated runtime buffer length fits u64"),
            read_only_initial_bytes: (buffer.access() == Gfx942RuntimeBufferAccessV1::ReadOnly)
                .then(|| buffer.allocation_bytes().to_vec()),
            kind: buffer.kind,
        })
        .collect();
    let buffers = inputs
        .buffers
        .into_iter()
        .map(Gfx942RuntimeDispatchBufferV1::into_kfd_buffer)
        .collect();
    let request = Gfx942KfdDispatchRequestV1::new(
        image,
        descriptor_offset,
        kernarg,
        kernarg_alignment,
        buffers,
        inputs.pointer_fixups,
        inputs.geometry,
        0,
        packet_group_segment_bytes,
        inputs.timeout_milliseconds,
    )?;
    Ok(PreparedGfx942RuntimeDispatchV1 {
        request,
        buffer_policies,
        identity,
        finalized_hsaco_length,
        dispatch_contract_sha256,
        kernel_name: kernel_name.to_owned(),
        descriptor_offset,
        static_group_segment_bytes,
        dynamic_group_segment_bytes: inputs.dynamic_group_segment_bytes,
        packet_group_segment_bytes,
        geometry: inputs.geometry,
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_dispatch_contract_sha256_v1(
    finalized_hsaco_length: u64,
    identity: Gfx942RuntimeKernelIdentityProjectionV1,
    kernel_name: &str,
    image: &[u8],
    descriptor_offset: u64,
    kernarg: &[u8],
    kernarg_alignment: u64,
    buffers: &[Gfx942RuntimeDispatchBufferV1],
    pointer_fixups: &[Gfx942KfdDispatchPointerFixupV1],
    geometry: AqlDispatchGeometryV1,
    packet_group_segment_bytes: u32,
    timeout_milliseconds: u32,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(GFX942_RUNTIME_DISPATCH_CONTRACT_DOMAIN_V1);
    digest.update(GFX942_KFD_DISPATCH_TRANSACTION_MANIFEST_SHA256_V1.as_bytes());
    digest.update(identity.object_sha256);
    digest.update(finalized_hsaco_length.to_le_bytes());
    digest.update(identity.metadata_sha256);
    digest.update(identity.descriptor_sha256);
    digest.update(identity.entry_sha256);
    digest.update(identity.closure_sha256);
    update_length_delimited_v1(&mut digest, kernel_name.as_bytes());
    update_length_delimited_v1(&mut digest, image);
    digest.update(descriptor_offset.to_le_bytes());
    update_length_delimited_v1(&mut digest, kernarg);
    digest.update(kernarg_alignment.to_le_bytes());
    digest.update(
        u64::try_from(buffers.len())
            .expect("bounded runtime buffer count fits u64")
            .to_le_bytes(),
    );
    for buffer in buffers {
        digest.update([buffer.access().canonical_tag()]);
        if buffer.is_non_null_empty_slice_sentinel_v1() {
            digest.update(NON_NULL_EMPTY_SLICE_SENTINEL_DIGEST_LENGTH_V1.to_le_bytes());
            digest.update(buffer.allocation_bytes());
        } else {
            update_length_delimited_v1(&mut digest, buffer.bytes());
        }
    }
    digest.update(
        u64::try_from(pointer_fixups.len())
            .expect("bounded runtime pointer-fixup count fits u64")
            .to_le_bytes(),
    );
    for fixup in pointer_fixups {
        digest.update(
            u64::try_from(fixup.kernarg_offset())
                .expect("bounded kernarg offset fits u64")
                .to_le_bytes(),
        );
        digest.update(
            u64::try_from(fixup.buffer_index())
                .expect("bounded buffer index fits u64")
                .to_le_bytes(),
        );
        digest.update(
            u64::try_from(fixup.buffer_byte_offset())
                .expect("bounded buffer offset fits u64")
                .to_le_bytes(),
        );
        digest.update(fixup.required_alignment().to_le_bytes());
    }
    for dimension in geometry.grid() {
        digest.update(dimension.to_le_bytes());
    }
    for dimension in geometry.workgroup() {
        digest.update(u32::from(dimension).to_le_bytes());
    }
    digest.update(0_u32.to_le_bytes());
    digest.update(packet_group_segment_bytes.to_le_bytes());
    digest.update(timeout_milliseconds.to_le_bytes());
    digest.finalize().into()
}

fn update_length_delimited_v1(digest: &mut Sha256, bytes: &[u8]) {
    digest.update(
        u64::try_from(bytes.len())
            .expect("bounded runtime input length fits u64")
            .to_le_bytes(),
    );
    digest.update(bytes);
}

fn validate_resources(
    resources: SelectedKernelResourceBindingV1,
    geometry: AqlDispatchGeometryV1,
    dynamic_group_segment_bytes: u32,
) -> Result<(), Gfx942RuntimePreparationErrorV1> {
    if resources.wavefront_size() != GFX942_WAVEFRONT_SIZE_V1 {
        return Err(Gfx942RuntimePreparationErrorV1::UnsupportedResource(
            "non-wave64 kernel",
        ));
    }
    // The spill-count metadata records compiler allocation statistics; it is not an AQL scratch
    // allocation requirement. The descriptor's private-segment byte count is the authoritative
    // packet resource. A nonzero count remains unsupported until scratch backing is implemented.
    if resources.private_segment_fixed_size() != 0 {
        return Err(Gfx942RuntimePreparationErrorV1::UnsupportedResource(
            "private segment scratch",
        ));
    }
    if resources.cluster_dims().is_some() {
        return Err(Gfx942RuntimePreparationErrorV1::UnsupportedResource(
            "cluster launch",
        ));
    }
    let workgroup = geometry.workgroup().map(u32::from);
    if resources
        .required_workgroup_size()
        .is_some_and(|required| required != workgroup)
        || workgroup.into_iter().product::<u32>() > resources.max_flat_workgroup_size()
    {
        return Err(Gfx942RuntimePreparationErrorV1::WorkgroupMismatch);
    }
    let grid = geometry.grid();
    for (axis, maximum) in resources.max_workgroups().into_iter().enumerate() {
        let count = ceil_div_u32(grid[axis], workgroup[axis]);
        if maximum.is_some_and(|maximum| count > maximum) {
            return Err(Gfx942RuntimePreparationErrorV1::WorkgroupCountExceeded { axis });
        }
    }
    if resources
        .group_segment_fixed_size()
        .checked_add(u64::from(dynamic_group_segment_bytes))
        .is_none_or(|total| total > GFX942_MAX_GROUP_SEGMENT_BYTES_V1)
    {
        return Err(Gfx942RuntimePreparationErrorV1::UnsupportedResource(
            "group segment capacity",
        ));
    }
    let alignment = resources.kernarg_segment_alignment();
    if alignment == 0 || alignment > 4096 || !alignment.is_power_of_two() {
        return Err(Gfx942RuntimePreparationErrorV1::KernargLayout);
    }
    Ok(())
}

fn initialize_hidden_arguments(
    kernarg: &mut [u8],
    hidden: &[HiddenArgument],
    geometry: AqlDispatchGeometryV1,
    dynamic_group_segment_bytes: u32,
) -> Result<(), Gfx942RuntimePreparationErrorV1> {
    let mut observed_dynamic_lds = false;
    for (index, argument) in hidden.iter().copied().enumerate() {
        let value = hidden_value(argument.value_kind(), geometry, dynamic_group_segment_bytes)
            .map_err(|detail| Gfx942RuntimePreparationErrorV1::HiddenArgument { index, detail })?;
        let offset = usize::try_from(argument.offset()).map_err(|_| {
            Gfx942RuntimePreparationErrorV1::HiddenArgument {
                index,
                detail: "offset conversion",
            }
        })?;
        let size = usize::try_from(argument.size()).map_err(|_| {
            Gfx942RuntimePreparationErrorV1::HiddenArgument {
                index,
                detail: "size conversion",
            }
        })?;
        let end =
            offset
                .checked_add(size)
                .ok_or(Gfx942RuntimePreparationErrorV1::HiddenArgument {
                    index,
                    detail: "range overflow",
                })?;
        let destination = kernarg.get_mut(offset..end).ok_or(
            Gfx942RuntimePreparationErrorV1::HiddenArgument {
                index,
                detail: "range outside kernarg",
            },
        )?;
        if destination.len() != value.len() {
            return Err(Gfx942RuntimePreparationErrorV1::HiddenArgument {
                index,
                detail: "kind width mismatch",
            });
        }
        destination.copy_from_slice(&value);
        observed_dynamic_lds |= argument.value_kind() == HiddenValueKind::DynamicLdsSize;
    }
    if dynamic_group_segment_bytes != 0 && !observed_dynamic_lds {
        return Err(Gfx942RuntimePreparationErrorV1::HiddenArgument {
            index: hidden.len(),
            detail: "dynamic LDS requested without ABI field",
        });
    }
    Ok(())
}

fn hidden_value(
    kind: HiddenValueKind,
    geometry: AqlDispatchGeometryV1,
    dynamic_group_segment_bytes: u32,
) -> Result<Vec<u8>, &'static str> {
    let grid = geometry.grid();
    let workgroup = geometry.workgroup().map(u32::from);
    let u32_value = |value: u32| value.to_le_bytes().to_vec();
    let u16_value = |value: u16| value.to_le_bytes().to_vec();
    let u64_value = |value: u64| value.to_le_bytes().to_vec();
    match kind {
        HiddenValueKind::BlockCountX => Ok(u32_value(ceil_div_u32(grid[0], workgroup[0]))),
        HiddenValueKind::BlockCountY => Ok(u32_value(ceil_div_u32(grid[1], workgroup[1]))),
        HiddenValueKind::BlockCountZ => Ok(u32_value(ceil_div_u32(grid[2], workgroup[2]))),
        HiddenValueKind::GroupSizeX => Ok(u16_value(workgroup[0] as u16)),
        HiddenValueKind::GroupSizeY => Ok(u16_value(workgroup[1] as u16)),
        HiddenValueKind::GroupSizeZ => Ok(u16_value(workgroup[2] as u16)),
        HiddenValueKind::RemainderX => Ok(u16_value((grid[0] % workgroup[0]) as u16)),
        HiddenValueKind::RemainderY => Ok(u16_value((grid[1] % workgroup[1]) as u16)),
        HiddenValueKind::RemainderZ => Ok(u16_value((grid[2] % workgroup[2]) as u16)),
        HiddenValueKind::GlobalOffsetX
        | HiddenValueKind::GlobalOffsetY
        | HiddenValueKind::GlobalOffsetZ
        | HiddenValueKind::None
        | HiddenValueKind::PrintfBuffer
        | HiddenValueKind::HostcallBuffer
        | HiddenValueKind::HeapV1
        | HiddenValueKind::DefaultQueue
        | HiddenValueKind::CompletionAction
        | HiddenValueKind::MultigridSyncArgument
        | HiddenValueKind::QueuePointer => Ok(u64_value(0)),
        HiddenValueKind::GridDimensions => Ok(u16_value(geometry.dimensions())),
        HiddenValueKind::DynamicLdsSize => Ok(u32_value(dynamic_group_segment_bytes)),
        HiddenValueKind::PrivateBase | HiddenValueKind::SharedBase => {
            Err("gfx942 aperture ABI field is unsupported")
        }
    }
}

const fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    value / divisor + if value.is_multiple_of(divisor) { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DispatchContractCaseV1 {
        finalized_hsaco_length: u64,
        identity: Gfx942RuntimeKernelIdentityProjectionV1,
        kernel_name: String,
        image: Vec<u8>,
        descriptor_offset: u64,
        kernarg: Vec<u8>,
        kernarg_alignment: u64,
        buffers: Vec<(Gfx942RuntimeBufferAccessV1, Vec<u8>)>,
        pointer_fixups: Vec<Gfx942KfdDispatchPointerFixupV1>,
        geometry: AqlDispatchGeometryV1,
        group_segment_bytes: u32,
        timeout_milliseconds: u32,
    }

    impl DispatchContractCaseV1 {
        fn baseline() -> Self {
            Self {
                finalized_hsaco_length: 7_000,
                identity: Gfx942RuntimeKernelIdentityProjectionV1 {
                    object_sha256: [1; 32],
                    metadata_sha256: [2; 32],
                    descriptor_sha256: [3; 32],
                    entry_sha256: [4; 32],
                    closure_sha256: [5; 32],
                },
                kernel_name: "kernel_v1".to_owned(),
                image: vec![6; 128],
                descriptor_offset: 64,
                kernarg: vec![0; 32],
                kernarg_alignment: 16,
                buffers: vec![(Gfx942RuntimeBufferAccessV1::ReadOnly, vec![7; 64])],
                pointer_fixups: vec![Gfx942KfdDispatchPointerFixupV1::new(0, 0, 8, 4)],
                geometry: AqlDispatchGeometryV1::new([64, 1, 1], [64, 1, 1]).unwrap(),
                group_segment_bytes: 256,
                timeout_milliseconds: 5_000,
            }
        }

        fn identity(&self) -> [u8; 32] {
            let buffers = self
                .buffers
                .iter()
                .map(|(access, bytes)| {
                    Gfx942RuntimeDispatchBufferV1::new(bytes.clone(), *access).unwrap()
                })
                .collect::<Vec<_>>();
            derive_dispatch_contract_sha256_v1(
                self.finalized_hsaco_length,
                self.identity,
                &self.kernel_name,
                &self.image,
                self.descriptor_offset,
                &self.kernarg,
                self.kernarg_alignment,
                &buffers,
                &self.pointer_fixups,
                self.geometry,
                self.group_segment_bytes,
                self.timeout_milliseconds,
            )
        }
    }

    fn geometry() -> AqlDispatchGeometryV1 {
        AqlDispatchGeometryV1::new([130, 4, 1], [64, 2, 1]).unwrap()
    }

    #[test]
    fn empty_slice_sentinel_owns_physical_storage_but_exposes_no_logical_bytes() {
        let sentinel =
            Gfx942RuntimeDispatchBufferV1::new_non_null_empty_slice_sentinel_v1().unwrap();
        assert_eq!(sentinel.access(), Gfx942RuntimeBufferAccessV1::ReadOnly);
        assert!(sentinel.is_non_null_empty_slice_sentinel_v1());
        assert!(sentinel.bytes().is_empty());
        assert_eq!(sentinel.allocation_bytes(), &[0]);

        let ordinary =
            Gfx942RuntimeDispatchBufferV1::new(vec![0], Gfx942RuntimeBufferAccessV1::ReadOnly)
                .unwrap();
        assert!(!ordinary.is_non_null_empty_slice_sentinel_v1());
        assert_eq!(ordinary.bytes(), &[0]);
    }

    #[test]
    fn empty_slice_sentinel_kind_is_bound_into_dispatch_identity() {
        let mut case = DispatchContractCaseV1::baseline();
        case.buffers = vec![(Gfx942RuntimeBufferAccessV1::ReadOnly, vec![0])];
        case.pointer_fixups = vec![Gfx942KfdDispatchPointerFixupV1::new(0, 0, 0, 8)];
        let ordinary_identity = case.identity();
        let sentinel =
            Gfx942RuntimeDispatchBufferV1::new_non_null_empty_slice_sentinel_v1().unwrap();
        let sentinel_identity = derive_dispatch_contract_sha256_v1(
            case.finalized_hsaco_length,
            case.identity,
            &case.kernel_name,
            &case.image,
            case.descriptor_offset,
            &case.kernarg,
            case.kernarg_alignment,
            &[sentinel],
            &case.pointer_fixups,
            case.geometry,
            case.group_segment_bytes,
            case.timeout_milliseconds,
        );

        assert_ne!(ordinary_identity, sentinel_identity);
    }

    #[test]
    fn geometry_hidden_values_are_derived_without_native_addresses() {
        assert_eq!(
            hidden_value(HiddenValueKind::BlockCountX, geometry(), 256).unwrap(),
            3_u32.to_le_bytes()
        );
        assert_eq!(
            hidden_value(HiddenValueKind::BlockCountY, geometry(), 256).unwrap(),
            2_u32.to_le_bytes()
        );
        assert_eq!(
            hidden_value(HiddenValueKind::RemainderX, geometry(), 256).unwrap(),
            2_u16.to_le_bytes()
        );
        assert_eq!(
            hidden_value(HiddenValueKind::GridDimensions, geometry(), 256).unwrap(),
            2_u16.to_le_bytes()
        );
        assert_eq!(
            hidden_value(HiddenValueKind::DynamicLdsSize, geometry(), 256).unwrap(),
            256_u32.to_le_bytes()
        );
    }

    #[test]
    fn optional_runtime_pointers_are_zero_and_gfx8_apertures_reject() {
        for kind in [
            HiddenValueKind::HostcallBuffer,
            HiddenValueKind::MultigridSyncArgument,
            HiddenValueKind::HeapV1,
            HiddenValueKind::DefaultQueue,
            HiddenValueKind::CompletionAction,
            HiddenValueKind::QueuePointer,
        ] {
            assert_eq!(hidden_value(kind, geometry(), 0).unwrap(), [0; 8]);
        }
        assert!(hidden_value(HiddenValueKind::PrivateBase, geometry(), 0).is_err());
        assert!(hidden_value(HiddenValueKind::SharedBase, geometry(), 0).is_err());
    }

    #[test]
    fn ceil_division_is_exact_at_and_after_boundaries() {
        assert_eq!(ceil_div_u32(64, 64), 1);
        assert_eq!(ceil_div_u32(65, 64), 2);
        assert_eq!(ceil_div_u32(u32::MAX, u16::MAX.into()), 65_537);
    }

    #[test]
    fn dispatch_contract_is_deterministic_and_binds_every_runtime_axis() {
        let baseline = DispatchContractCaseV1::baseline();
        let expected = baseline.identity();
        assert_eq!(baseline.identity(), expected);

        let mut mutations = Vec::new();
        let mut changed = baseline.clone();
        changed.finalized_hsaco_length += 1;
        mutations.push(("finalized length", changed));
        let mut changed = baseline.clone();
        changed.identity.object_sha256[0] ^= 1;
        mutations.push(("object identity", changed));
        let mut changed = baseline.clone();
        changed.identity.metadata_sha256[0] ^= 1;
        mutations.push(("metadata identity", changed));
        let mut changed = baseline.clone();
        changed.identity.descriptor_sha256[0] ^= 1;
        mutations.push(("descriptor identity", changed));
        let mut changed = baseline.clone();
        changed.identity.entry_sha256[0] ^= 1;
        mutations.push(("entry identity", changed));
        let mut changed = baseline.clone();
        changed.identity.closure_sha256[0] ^= 1;
        mutations.push(("closure identity", changed));
        let mut changed = baseline.clone();
        changed.kernel_name.push('x');
        mutations.push(("kernel name", changed));
        let mut changed = baseline.clone();
        changed.image[0] ^= 1;
        mutations.push(("materialized image", changed));
        let mut changed = baseline.clone();
        changed.descriptor_offset += 64;
        mutations.push(("descriptor offset", changed));
        let mut changed = baseline.clone();
        changed.kernarg[31] ^= 1;
        mutations.push(("kernarg", changed));
        let mut changed = baseline.clone();
        changed.kernarg_alignment = 32;
        mutations.push(("kernarg alignment", changed));
        let mut changed = baseline.clone();
        changed.buffers[0].0 = Gfx942RuntimeBufferAccessV1::ReadWrite;
        mutations.push(("buffer access", changed));
        let mut changed = baseline.clone();
        changed.buffers[0].1[0] ^= 1;
        mutations.push(("buffer bytes", changed));
        let mut changed = baseline.clone();
        changed
            .buffers
            .push((Gfx942RuntimeBufferAccessV1::WriteOnly, vec![8; 4]));
        mutations.push(("buffer count", changed));
        let mut changed = baseline.clone();
        changed.pointer_fixups[0] = Gfx942KfdDispatchPointerFixupV1::new(8, 0, 8, 4);
        mutations.push(("pointer fixup", changed));
        let mut changed = baseline.clone();
        changed.geometry = AqlDispatchGeometryV1::new([128, 1, 1], [64, 1, 1]).unwrap();
        mutations.push(("grid geometry", changed));
        let mut changed = baseline.clone();
        changed.geometry = AqlDispatchGeometryV1::new([64, 1, 1], [32, 1, 1]).unwrap();
        mutations.push(("workgroup geometry", changed));
        let mut changed = baseline.clone();
        changed.group_segment_bytes += 4;
        mutations.push(("group segment", changed));
        let mut changed = baseline.clone();
        changed.timeout_milliseconds += 1;
        mutations.push(("timeout", changed));

        for (field, changed) in mutations {
            assert_ne!(changed.identity(), expected, "{field} was not bound");
        }
    }
}

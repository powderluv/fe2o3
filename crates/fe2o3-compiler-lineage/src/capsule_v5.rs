//! Conditional metadata/carrier framing; never an ordinary V3 base.
use crate::{
    InertFinalCompilerModuleCommitmentReceiptV3, InertNativeLoweringAssociationV1,
    InertRustcIdentityInventoryReceiptV3, InertRustcPreflightPlanReceiptV3, LineageDecodeErrorV3,
    NativeLoweringAssociationErrorV1, bounded_pair as pair,
    native_conditional_carrier_v1::*,
    native_conditional_metadata_v1::*,
    native_conditional_metadata_v2::*,
    native_conditional_output_v1::MAX_NATIVE_CONDITIONAL_STORAGE_V1,
    native_conditional_policy_roster_v1::NativeConditionalPolicyRosterRefV1,
    receipt::{ImmutableBytesV3, SharedBackingV3, conditional_metadata_receipts},
};
use fe2o3_kernel_descriptor::DeviceTargetV1;
use fe2o3_rustc_invocation::{
    InvocationDigestV3, RustcInvocationDescriptorV3, decode_descriptor_v3,
};
use sha2::{Digest, Sha256};
use std::{convert::Infallible, mem::size_of, ops::Range, sync::Arc};

/// Conditional capsule discriminator.
pub const INERT_PRODUCTION_SEMANTIC_CAPSULE_MAGIC_V5: [u8; 8] = *b"F2O3ISV5";
/// Closed conditional capsule version.
pub const INERT_PRODUCTION_SEMANTIC_CAPSULE_VERSION_V5: u16 = 5;
/// Fixed pair header length.
pub const INERT_PRODUCTION_SEMANTIC_CAPSULE_HEADER_BYTES_V5: usize = pair::HEADER;
/// Unchanged complete capsule ceiling.
pub const MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V5: usize = 160 * 1024 * 1024;
const POLICY: pair::Policy = pair::Policy {
    magic: INERT_PRODUCTION_SEMANTIC_CAPSULE_MAGIC_V5,
    version: 5,
    domain: b"FE2O3/INERT-PRODUCTION-SEMANTIC-CAPSULE/V5\0",
    first_max: MAX_NATIVE_CONDITIONAL_METADATA_BYTES_V2,
    second_max: MAX_NATIVE_CONDITIONAL_CARRIER_BYTES_V1,
    total_max: MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V5,
    storage_max: MAX_NATIVE_CONDITIONAL_STORAGE_V1,
};

/// Strict conditional framing or inherited content-codec failure.
#[derive(Debug)]
pub enum InertProductionSemanticCapsuleErrorV5<E = Infallible> {
    /// Original caller work refusal.
    Charge(E),
    /// Mandatory metadata length invalid.
    MetadataLength,
    /// Mandatory carrier length invalid.
    CarrierLength,
    /// Exact length or shared range invalid.
    Length,
    /// Checked extent overflow.
    Arithmetic,
    /// Unsupported schema/header.
    Header,
    /// Nonzero reserved bytes.
    Reserved,
    /// Terminal digest mismatch.
    Identity,
    /// Unsupported shared storage ceiling.
    StorageLimit,
    /// Nested conditional carrier failed.
    Carrier(NativeConditionalCarrierErrorV1<E>),
    /// Nested conditional metadata failed.
    Metadata(NativeConditionalMetadataErrorV2<E>),
    /// Roster does not bind the exact source carried by this capsule.
    SourceIdentity,
    /// Existing invocation/receipt content failed.
    Content(LineageDecodeErrorV3),
    /// Fixed native lowering association failed.
    Lowering(NativeLoweringAssociationErrorV1),
    /// Invocation target and lowering profile differ or are unsupported.
    Target,
}
type Error<E = Infallible> = InertProductionSemanticCapsuleErrorV5<E>;
impl<E> From<pair::Error<E>> for Error<E> {
    fn from(e: pair::Error<E>) -> Self {
        match e {
            pair::Error::Charge(e) => Self::Charge(e),
            pair::Error::FirstLength => Self::MetadataLength,
            pair::Error::SecondLength => Self::CarrierLength,
            pair::Error::Length => Self::Length,
            pair::Error::Arithmetic => Self::Arithmetic,
            pair::Error::Header => Self::Header,
            pair::Error::Reserved => Self::Reserved,
            pair::Error::Identity => Self::Identity,
            pair::Error::StorageLimit => Self::StorageLimit,
        }
    }
}

/// Disjoint complete metadata/carrier ranges for direct final-buffer encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InertProductionSemanticCapsuleLayoutV5(pair::Layout);
impl InertProductionSemanticCapsuleLayoutV5 {
    /// Checks member and aggregate extents without payload visits.
    pub fn new<E>(metadata: usize, carrier: usize) -> Result<Self, Error<E>> {
        Ok(Self(pair::Layout::new(&POLICY, metadata, carrier)?))
    }
    /// Complete canonical extent.
    pub const fn encoded_len(self) -> usize {
        self.0.encoded_len()
    }
    /// Exact metadata frame region.
    pub fn metadata_range(self) -> Range<usize> {
        self.0.first_range()
    }
    /// Exact conditional carrier region.
    pub fn carrier_range(self) -> Range<usize> {
        self.0.second_range()
    }
}

/// Exact V5 capsule identity; never interchangeable with V3/V4.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InertProductionSemanticCapsuleIdentityV5(pair::Identity);
impl InertProductionSemanticCapsuleIdentityV5 {
    /// Domain-separated digest.
    pub const fn sha256(&self) -> &[u8; 32] {
        &self.0.sha256
    }
    /// Complete capsule byte length.
    pub const fn byte_len(self) -> u64 {
        self.0.byte_len
    }
}

/// Recursively checked inert framing borrowing one original input.
pub struct InertProductionSemanticCapsuleRefV5<'a> {
    bytes: &'a [u8],
    layout: InertProductionSemanticCapsuleLayoutV5,
    metadata: NativeConditionalMetadataRefV2<'a>,
    carrier: NativeConditionalCarrierRefV1<'a>,
    identity: InertProductionSemanticCapsuleIdentityV5,
}
impl<'a> InertProductionSemanticCapsuleRefV5<'a> {
    /// Complete framed image.
    pub const fn canonical_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    /// Inert metadata fields.
    pub const fn metadata(&self) -> &NativeConditionalMetadataRefV1<'a> {
        self.metadata.metadata()
    }
    /// Complete V2 envelope containing unchanged six-field metadata and roster.
    pub const fn metadata_v2(&self) -> &NativeConditionalMetadataRefV2<'a> {
        &self.metadata
    }
    /// Inert policy material bound to this capsule's exact source bytes.
    pub const fn policy_roster(&self) -> &NativeConditionalPolicyRosterRefV1<'a> {
        self.metadata.policy_roster()
    }
    /// Conditional pair, never the ordinary carrier type.
    pub const fn carrier(&self) -> &NativeConditionalCarrierRefV1<'a> {
        &self.carrier
    }
    /// Complete identity.
    pub const fn identity(&self) -> InertProductionSemanticCapsuleIdentityV5 {
        self.identity
    }
    /// Framing grants no authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Fixed simultaneous framing/owner scratch, excluding backing and invocation
/// heap metadata. The owner size includes all three cached receipt headers;
/// their preimages share the separately paid backing. FFI quotes its inherited
/// bounded decode metadata separately.
pub const INERT_PRODUCTION_SEMANTIC_CAPSULE_WORKING_STORAGE_V5: usize =
    size_of::<InertProductionSemanticCapsuleRefV5<'static>>()
        + size_of::<InertProductionSemanticCapsuleLayoutV5>()
        + size_of::<InertProductionSemanticCapsuleV5>()
        + size_of::<Sha256>()
        + NATIVE_CONDITIONAL_CARRIER_WORKING_STORAGE_V1
        + NATIVE_CONDITIONAL_METADATA_WORKING_STORAGE_V2
        + pair::OVERHEAD
        + 256;

/// Seals this frame only; nested metadata/carrier must already be written.
/// All work and caller callback destruction occur before mutation.
pub fn seal_inert_production_semantic_capsule_v5<E>(
    layout: InertProductionSemanticCapsuleLayoutV5,
    bytes: &mut [u8],
    storage_limit: usize,
    charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<InertProductionSemanticCapsuleIdentityV5, Error<E>> {
    Ok(InertProductionSemanticCapsuleIdentityV5(pair::seal(
        &POLICY,
        layout.0,
        bytes,
        storage_limit,
        charge,
    )?))
}

/// Allocation-free recursive framing; full leaf semantic admission is separate.
pub fn read_inert_production_semantic_capsule_v5<'a, E>(
    bytes: &'a [u8],
    storage_limit: usize,
    mut charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<InertProductionSemanticCapsuleRefV5<'a>, Error<E>> {
    let (layout, identity) = pair::read(&POLICY, bytes, storage_limit, &mut charge)?;
    let metadata = read_native_conditional_metadata_v2(
        &bytes[layout.first_range()],
        storage_limit,
        &mut charge,
    )
    .map_err(Error::Metadata)?;
    let carrier = read_native_conditional_carrier_v1(
        &bytes[layout.second_range()],
        storage_limit,
        &mut charge,
    )
    .map_err(Error::Carrier)?;
    let roster = metadata.policy_roster();
    let source = carrier.source_packet();
    if roster.source_packet_len() != source.len() as u64 {
        return Err(Error::SourceIdentity);
    }
    charge(
        source
            .len()
            .checked_add(128 + 32)
            .ok_or(Error::Arithmetic)?,
    )
    .map_err(Error::Charge)?;
    let source_sha256: [u8; 32] = Sha256::digest(source).into();
    if roster.source_packet_sha256() != &source_sha256 {
        return Err(Error::SourceIdentity);
    }
    Ok(InertProductionSemanticCapsuleRefV5 {
        bytes,
        layout: InertProductionSemanticCapsuleLayoutV5(layout),
        metadata,
        carrier,
        identity: InertProductionSemanticCapsuleIdentityV5(identity),
    })
}

/// Immutable conditional content in shared custody, with no ordinary base.
/// Decoded invocation/receipt content does not authenticate original compiler custody.
///
/// ```compile_fail
/// use fe2o3_compiler_lineage::InertProductionSemanticCapsuleV5;
/// fn duplicate(v: InertProductionSemanticCapsuleV5) { let _ = v.clone(); }
/// ```
pub struct InertProductionSemanticCapsuleV5 {
    bytes: ImmutableBytesV3,
    invocation: RustcInvocationDescriptorV3,
    invocation_digest: InvocationDigestV3,
    invocation_range: Range<usize>,
    target: DeviceTargetV1,
    inventory: InertRustcIdentityInventoryReceiptV3,
    preflight: InertRustcPreflightPlanReceiptV3,
    final_commitment: InertFinalCompilerModuleCommitmentReceiptV3,
    lowering: InertNativeLoweringAssociationV1,
    layout_range: Range<usize>,
    policy_roster_range: Range<usize>,
    carrier_range: Range<usize>,
    history_range: Range<usize>,
    catalog_range: Range<usize>,
    descriptor_range: Range<usize>,
    source_range: Range<usize>,
    carrier_identity: NativeConditionalCarrierIdentityV1,
    identity: InertProductionSemanticCapsuleIdentityV5,
}
impl InertProductionSemanticCapsuleV5 {
    /// Transfers payload allocation; caller prepays full capacity and decode metadata/work.
    pub fn decode_owned(bytes: Vec<u8>) -> Result<Self, Error> {
        let length = bytes.len();
        Self::decode_shared_vec(Arc::new(bytes), 0..length)
    }
    /// Retains exact ranges in one backing; no payload-sized copies. This content
    /// decoder has no ledger: prepay full backing capacity and quoted metadata/work.
    pub fn decode_shared_vec(backing: Arc<Vec<u8>>, range: Range<usize>) -> Result<Self, Error> {
        let shared = SharedBackingV3::Vector(backing);
        let bytes = shared.as_slice().get(range.clone()).ok_or(Error::Length)?;
        let frame = read_inert_production_semantic_capsule_v5(bytes, POLICY.storage_max, |_| {
            Ok::<_, Infallible>(())
        })?;
        let invocation = decode_descriptor_v3(frame.metadata().invocation())
            .map_err(|e| Error::Content(LineageDecodeErrorV3::Invocation(e)))?;
        let invocation_digest = InvocationDigestV3::calculate(&invocation)
            .map_err(|_| Error::Content(LineageDecodeErrorV3::NonCanonical))?;
        let target = DeviceTargetV1::parse(invocation.amd_target()).map_err(|_| Error::Target)?;
        let lowering = InertNativeLoweringAssociationV1::decode(frame.metadata().native_lowering())
            .map_err(Error::Lowering)?;
        if invocation.amd_target() != lowering.inputs().profile.device_target() {
            return Err(Error::Target);
        }
        let shift = |base: usize, r: Range<usize>| base + r.start..base + r.end;
        let envelope_base = frame.layout.metadata_range().start;
        let metadata_base = envelope_base + frame.metadata.layout.metadata_range().start;
        let policy_roster_range = shift(envelope_base, frame.metadata.layout.policy_roster_range());
        let metadata = frame.metadata().layout;
        let invocation_range = shift(metadata_base, metadata.invocation_range());
        let (inventory, preflight, final_commitment) = conditional_metadata_receipts(
            shared.clone(),
            shift(
                range.start + metadata_base,
                metadata.rustc_inventory_range(),
            ),
            shift(
                range.start + metadata_base,
                metadata.rustc_preflight_range(),
            ),
            shift(
                range.start + metadata_base,
                metadata.final_module_commitment_range(),
            ),
        )
        .map_err(Error::Content)?;
        let carrier_range = frame.layout.carrier_range();
        let output_base = carrier_range.start + frame.carrier.layout.output_range().start;
        let output = frame.carrier.output().layout;
        let history_range = shift(output_base, output.history_range());
        let catalog_range = shift(output_base, output.catalog_range());
        let descriptor_range = shift(output_base, output.descriptor_range());
        let source_range = shift(carrier_range.start, frame.carrier.layout.source_range());
        let layout_range = shift(metadata_base, metadata.semantic_target_layout_range());
        let carrier_identity = frame.carrier.identity();
        let identity = frame.identity();
        Ok(Self {
            bytes: ImmutableBytesV3::from_shared(shared, range).ok_or(Error::Length)?,
            invocation,
            invocation_digest,
            invocation_range,
            target,
            inventory,
            preflight,
            final_commitment,
            lowering,
            layout_range,
            policy_roster_range,
            carrier_range,
            history_range,
            catalog_range,
            descriptor_range,
            source_range,
            carrier_identity,
            identity,
        })
    }
    /// Original canonical invocation content, not authenticated origin.
    pub const fn invocation(&self) -> &RustcInvocationDescriptorV3 {
        &self.invocation
    }
    /// Exact canonical invocation bytes borrowed from the retained capsule.
    /// This is inert content, not authenticated origin or a receipt digest.
    pub fn invocation_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.invocation_range.clone()]
    }
    /// Existing invocation digest domain.
    pub const fn invocation_digest(&self) -> InvocationDigestV3 {
        self.invocation_digest
    }
    /// Target matching invocation and native lowering profile.
    pub const fn target(&self) -> DeviceTargetV1 {
        self.target
    }
    /// Inert original inventory receipt.
    pub const fn rustc_identity_inventory(&self) -> &InertRustcIdentityInventoryReceiptV3 {
        &self.inventory
    }
    /// Inert original preflight receipt.
    pub const fn rustc_preflight_plan(&self) -> &InertRustcPreflightPlanReceiptV3 {
        &self.preflight
    }
    /// Unchanged semantic-target-layout transcript.
    pub fn semantic_target_layout_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.layout_range.clone()]
    }
    /// Complete validated inert roster in the original shared backing.
    /// Re-reading policy rows uses the separately charged roster reader.
    pub fn policy_roster_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.policy_roster_range.clone()]
    }
    /// Borrow inert layout fields from the preimage validated during decode.
    /// This allocation-free getter has no ledger: callers prepay preimage length
    /// plus176 work units and `size_of::<NativeConditionalTargetLayoutRefV1>()`
    /// plus `size_of::<[&[u8]; 6]>()` plus `size_of::<[usize; 8]>()` scratch.
    /// Keep the returned view paid while live. This establishes no live target facts.
    pub fn semantic_target_layout(&self) -> NativeConditionalTargetLayoutRefV1<'_> {
        crate::native_conditional_metadata_v1::target_layout::<Infallible>(
            self.semantic_target_layout_bytes(),
        )
        .expect("immutable conditional target layout was validated at decode")
    }
    /// Strictly decoded association, not a lowering theorem.
    pub const fn native_lowering(&self) -> &InertNativeLoweringAssociationV1 {
        &self.lowering
    }
    /// Cached inert lineage receipt over the whole final-module preimage.
    /// This getter neither hashes nor allocates; the caller prepays its two hash
    /// visits and retained header before decode. No FFI parsing or authority is added.
    pub const fn final_compiler_module_commitment(
        &self,
    ) -> &InertFinalCompilerModuleCommitmentReceiptV3 {
        &self.final_commitment
    }
    /// Compact final-module preimage; only FFI decodes this schema.
    pub fn final_module_commitment_bytes(&self) -> &[u8] {
        self.final_commitment.canonical_preimage()
    }
    /// Complete conditional carrier encoding.
    pub fn carrier_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.carrier_range.clone()]
    }
    /// Exact complete conditional pair identity.
    pub const fn carrier_identity(&self) -> NativeConditionalCarrierIdentityV1 {
        self.carrier_identity
    }
    /// Unchanged history containing F once.
    pub fn history_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.history_range.clone()]
    }
    /// Complete final catalog bytes.
    pub fn catalog_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.catalog_range.clone()]
    }
    /// Complete zero-digest descriptor V5 preimage; structural admission remains required.
    pub fn descriptor_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.descriptor_range.clone()]
    }
    /// Complete conditional source packet V2.
    pub fn source_packet_bytes(&self) -> &[u8] {
        &self.canonical_bytes()[self.source_range.clone()]
    }
    /// Canonical capsule in its original shared allocation.
    pub fn canonical_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
    /// Exact V5 capsule identity.
    pub const fn identity(&self) -> InertProductionSemanticCapsuleIdentityV5 {
        self.identity
    }
    /// Content never grants compiler/publication/load/launch authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

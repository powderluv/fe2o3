//! Inert compiler-captured root associations beside an unchanged legacy inventory.
//!
//! This codec validates framing and canonical coordinates, not their provenance.
//! Actual compiler ownership, source membership and original policy admission
//! remain consumer obligations. No public constructor authenticates these bytes.

use crate::{
    MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3, MAX_NATIVE_CONDITIONAL_STORAGE_V1, bounded_pair as pair,
};
use sha2::{Digest, Sha256};
use std::{convert::Infallible, mem::size_of};

/// Domain of the compiler's unchanged, length-prefixed legacy inventory.
pub const RUSTC_IDENTITY_INVENTORY_DOMAIN_V2: &[u8] =
    b"fe2o3/semantic-mir/rustc-identity-inventory/v2";
/// Fixed work bound for the legacy format discriminator; the caller charges it.
pub const RUSTC_IDENTITY_INVENTORY_PREFIX_WORK_V2: usize =
    3 * size_of::<u64>() + RUSTC_IDENTITY_INVENTORY_DOMAIN_V2.len() + 32;

/// Recognizes the legacy domain and first target-identity field without allocation.
///
/// This only distinguishes the original absent-enrollment format from a wrapper.
/// It does not validate the remaining transcript, source membership or authority;
/// those remain obligations of actual compiler/source recovery.
pub fn has_rustc_identity_inventory_prefix_v2(bytes: &[u8]) -> bool {
    if bytes.len() > MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3 {
        return false;
    }
    bytes
        .strip_prefix(&(RUSTC_IDENTITY_INVENTORY_DOMAIN_V2.len() as u64).to_le_bytes())
        .and_then(|bytes| bytes.strip_prefix(RUSTC_IDENTITY_INVENTORY_DOMAIN_V2))
        .and_then(|bytes| bytes.strip_prefix(&32u64.to_le_bytes()))
        .is_some_and(|bytes| bytes.len() > 32)
}

/// Closed V1 inventory wrapper discriminator, including its terminal NUL.
pub const RUSTC_ENROLLMENT_INVENTORY_MAGIC_V1: [u8; 8] = *b"F2RINV1\0";
/// Fixed association header extent, excluding the outer pair header.
pub const RUSTC_ENROLLMENT_INVENTORY_HEADER_BYTES_V1: usize = 80;
/// Fixed canonical root association extent.
pub const RUSTC_ENROLLMENT_INVENTORY_ROOT_BYTES_V1: usize = 144;
/// The entire wrapper, not each member independently, must fit this ceiling.
pub const MAX_RUSTC_ENROLLMENT_INVENTORY_BYTES_V1: usize = MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3;
const HEADER: usize = RUSTC_ENROLLMENT_INVENTORY_HEADER_BYTES_V1;
const ROW: usize = RUSTC_ENROLLMENT_INVENTORY_ROOT_BYTES_V1;
const MAX: usize = MAX_RUSTC_ENROLLMENT_INVENTORY_BYTES_V1;
// The legacy member is nonempty. This bound is derived from the aggregate cap.
const MAX_ROOTS: usize = (MAX - pair::OVERHEAD - HEADER - 1) / ROW;
const CENSUS_BYTES: usize = MAX_ROOTS.div_ceil(8);
const POLICY: pair::Policy = pair::Policy {
    magic: RUSTC_ENROLLMENT_INVENTORY_MAGIC_V1,
    version: 1,
    domain: b"FE2O3/RUSTC-ENROLLMENT-INVENTORY/V1\0",
    first_max: MAX,
    second_max: MAX,
    total_max: MAX,
    storage_max: MAX_NATIVE_CONDITIONAL_STORAGE_V1,
};

/// Inert original invocation coordinates, never admitted policy authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustcEnrollmentInventoryHeaderV1 {
    /// Complete KernelEntry subset count in canonical function-table order.
    pub kernel_count: u32,
    /// Exact count of original descriptor enrollment bindings.
    pub enrollment_binding_count: u32,
    /// Claimed original invocation identity.
    pub invocation_identity: [u8; 32],
    /// Claimed original admitted native policy identity.
    pub native_policy_identity: [u8; 32],
    /// Claimed original native policy generation.
    pub native_policy_generation: u64,
}

/// Independent original invocation/policy coordinates, not authority by construction.
///
/// Native callers obtain these from retained original owners; neither an inventory
/// header nor a CPU leaf may select these expectations. The verifier re-exports
/// this shared type so the root coordinator need not depend on proof recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConditionalCpuMappingExpectationV1 {
    /// Raw SHA256 of the canonical invocation, not the intake-domain digest.
    pub rustc_invocation_sha256: [u8; 32],
    /// Identity of the original admitted native policy.
    pub native_policy_sha256: [u8; 32],
    /// Generation of that original policy.
    pub policy_generation: u64,
    /// Count independently projected from the original descriptor.
    pub enrollment_binding_count: u32,
}

impl NativeConditionalCpuMappingExpectationV1 {
    /// Fixed work for the header copy and coordinate comparison; caller charges it.
    pub const HEADER_MATCH_WORK: usize = HEADER + size_of::<Self>();

    /// Checks only coordinates, not root membership, source identity or custody.
    ///
    /// The consumer must still decode its actual retained inventory and validate
    /// every root against original source and CPU evidence. Equal headers do not
    /// make two inventories interchangeable or authorize recovery or execution.
    pub fn matches_header(&self, header: &RustcEnrollmentInventoryHeaderV1) -> bool {
        header.invocation_identity == self.rustc_invocation_sha256
            && header.native_policy_identity == self.native_policy_sha256
            && header.native_policy_generation == self.policy_generation
            && header.enrollment_binding_count == self.enrollment_binding_count
            && self.enrollment_binding_count as usize
                <= fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BINDINGS_V1
    }
}

/// Fixed inert root coordinates. Construction does not establish compiler origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustcEnrollmentInventoryRootV1 {
    /// Semantic function-table ID; gaps are allowed, but rows strictly increase.
    pub semantic_root: u32,
    /// Exactly zero for registration or one for descriptor enrollment.
    pub origin_tag: u8,
    /// Original descriptor ordinal, or exactly zero for registration.
    pub descriptor_ordinal: u32,
    /// Byte length of the actual registered logical name, not a selector.
    pub logical_name_len: u32,
    /// Raw SHA256 of that exact logical name.
    pub logical_name_sha256: [u8; 32],
    /// Claimed original kernel-binding identity.
    pub kernel_binding: [u8; 32],
    /// Canonical actual kernel Instance identity, including substitutions.
    pub kernel_instance: [u8; 32],
    /// Canonical actual CPU reference Instance identity, including substitutions.
    pub reference_instance: [u8; 32],
}

/// Borrowed complete encoding inputs, without authority-bearing ownership.
#[derive(Clone, Copy, Debug)]
pub struct RustcEnrollmentInventoryInputV1<'a> {
    /// Complete unchanged legacy inventory transcript; opaque to this codec.
    pub legacy_inventory: &'a [u8],
    /// Original association coordinates.
    pub header: RustcEnrollmentInventoryHeaderV1,
    /// Complete canonical KernelEntry subset, never sorted by the encoder.
    pub roots: &'a [RustcEnrollmentInventoryRootV1],
}

/// Framing or accounting refusal, not a provenance-checking result.
#[derive(Debug, Eq, PartialEq)]
pub enum RustcEnrollmentInventoryErrorV1<E = Infallible> {
    /// Original caller work refusal, retained without replacement.
    Charge(E),
    /// Legacy transcript is empty or exceeds its aggregate bound.
    LegacyLength,
    /// Association member is not a bounded exact header plus fixed rows.
    AssociationLength,
    /// Exact total length differs or exceeds the aggregate ceiling.
    Length,
    /// A checked arithmetic operation overflowed.
    Arithmetic,
    /// Outer magic, version, policy or header size differs.
    Header,
    /// A reserved byte is nonzero.
    Reserved,
    /// Domain-separated framing digest differs.
    Identity,
    /// Caller storage ceiling is unsupported or below the published requirement.
    StorageLimit,
    /// Header count, row count or enrollment census differs.
    Count,
    /// Semantic root IDs are repeated or not in canonical table order.
    RootOrder,
    /// Origin tag is unsupported, or registration has a nonzero ordinal.
    Origin,
    /// Enrollment ordinal is duplicate, absent or out of range.
    EnrollmentOrdinal,
    /// Fallible exact-capacity allocation failed.
    Allocation,
}
type Error<E> = RustcEnrollmentInventoryErrorV1<E>;

impl<E> From<pair::Error<E>> for Error<E> {
    fn from(error: pair::Error<E>) -> Self {
        match error {
            pair::Error::Charge(e) => Self::Charge(e),
            pair::Error::FirstLength => Self::LegacyLength,
            pair::Error::SecondLength => Self::AssociationLength,
            pair::Error::Length => Self::Length,
            pair::Error::Arithmetic => Self::Arithmetic,
            pair::Error::Header => Self::Header,
            pair::Error::Reserved => Self::Reserved,
            pair::Error::Identity => Self::Identity,
            pair::Error::StorageLimit => Self::StorageLimit,
        }
    }
}

/// Constant-time checked aggregate extent; no inputs are scanned or allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustcEnrollmentInventoryLayoutV1 {
    pair: pair::Layout,
    roots: usize,
}
impl RustcEnrollmentInventoryLayoutV1 {
    /// Quote a complete wrapper including both members and outer framing.
    pub fn new<E>(legacy_len: usize, root_count: usize) -> Result<Self, Error<E>> {
        let association_len = root_count
            .checked_mul(ROW)
            .and_then(|n| n.checked_add(HEADER))
            .ok_or(Error::Arithmetic)?;
        let pair = pair::Layout::new(&POLICY, legacy_len, association_len)?;
        Ok(Self {
            pair,
            roots: root_count,
        })
    }
    /// Complete wrapper byte extent.
    pub const fn encoded_len(self) -> usize {
        self.pair.encoded_len()
    }
    /// Complete association row count.
    pub const fn root_count(self) -> usize {
        self.roots
    }
    /// Vec header and exact backing capacity to prepay before owned encoding.
    /// The published working storage and all input custody are additional.
    pub fn owned_storage_bytes<E>(self) -> Result<usize, Error<E>> {
        self.encoded_len()
            .checked_add(size_of::<Vec<u8>>())
            .ok_or(Error::Arithmetic)
    }
}

/// Borrowed fixed header; all values remain inert claims.
#[derive(Clone, Copy, Debug)]
pub struct RustcEnrollmentInventoryHeaderRefV1<'a> {
    bytes: &'a [u8],
}
impl<'a> RustcEnrollmentInventoryHeaderRefV1<'a> {
    /// Exact eighty original header bytes.
    pub const fn canonical_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    /// Copy the fixed inert header values without allocating.
    pub fn value(&self) -> RustcEnrollmentInventoryHeaderV1 {
        header_value(self.bytes)
    }
}

/// Borrowed fixed root association; not a compiler-authenticated root.
#[derive(Clone, Copy, Debug)]
pub struct RustcEnrollmentInventoryRootRefV1<'a> {
    bytes: &'a [u8],
}
impl<'a> RustcEnrollmentInventoryRootRefV1<'a> {
    /// Exact original 144-byte row, including its canonical zero reserved bytes.
    pub const fn canonical_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    /// Copy the fixed inert root values without allocating.
    pub fn value(&self) -> RustcEnrollmentInventoryRootV1 {
        root_value(self.bytes)
    }
}

/// Strict allocation-free borrowed inventory view, not admitted provenance.
///
/// Every association axis still requires checking against the actual original
/// compiler owner and semantic source. A coherently rehashed lie is still inert.
///
/// ```compile_fail
/// use fe2o3_compiler_lineage::{RustcEnrollmentInventoryRefV1, read_rustc_enrollment_inventory_v1};
/// fn escape(bytes: Vec<u8>) -> RustcEnrollmentInventoryRefV1<'static> {
///     read_rustc_enrollment_inventory_v1(&bytes, 0, |_| Ok::<_, ()>(())).unwrap()
/// }
/// ```
pub struct RustcEnrollmentInventoryRefV1<'a> {
    bytes: &'a [u8],
    layout: RustcEnrollmentInventoryLayoutV1,
    framing_sha256: [u8; 32],
    full_wrapper_sha256: [u8; 32],
}
impl<'a> RustcEnrollmentInventoryRefV1<'a> {
    /// Complete exact wrapper preimage, including its framing digest.
    pub const fn canonical_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    /// Complete unchanged opaque legacy inventory transcript.
    pub fn legacy_inventory(&self) -> &'a [u8] {
        &self.bytes[self.layout.pair.first_range()]
    }
    /// Original borrowed association header bytes.
    pub fn header_bytes(&self) -> &'a [u8] {
        &self.bytes[self.layout.pair.second_range()][..HEADER]
    }
    /// Original borrowed fixed header view.
    pub fn header_ref(&self) -> RustcEnrollmentInventoryHeaderRefV1<'a> {
        RustcEnrollmentInventoryHeaderRefV1 {
            bytes: self.header_bytes(),
        }
    }
    /// Copy of the original fixed header.
    pub fn header(&self) -> RustcEnrollmentInventoryHeaderV1 {
        self.header_ref().value()
    }
    /// Complete number of KernelEntry association rows.
    pub const fn root_count(&self) -> usize {
        self.layout.roots
    }
    /// Borrowed complete rows in original canonical function-table order.
    /// Consumers prepay subsequent row walks against their original work account.
    pub fn roots(&self) -> impl ExactSizeIterator<Item = RustcEnrollmentInventoryRootRefV1<'a>> {
        self.bytes[self.layout.pair.second_range()][HEADER..]
            .chunks_exact(ROW)
            .map(|bytes| RustcEnrollmentInventoryRootRefV1 { bytes })
    }
    /// Raw SHA256 of the COMPLETE encoded wrapper, for inventory/preflight joins.
    /// This is deliberately distinct from the domain-separated framing trailer.
    pub const fn full_wrapper_sha256(&self) -> &[u8; 32] {
        &self.full_wrapper_sha256
    }
    /// Domain-separated framing trailer only, never the inventory identity.
    pub const fn framing_sha256(&self) -> &[u8; 32] {
        &self.framing_sha256
    }
    /// Exact complete wrapper byte length.
    pub const fn byte_len(&self) -> u64 {
        self.bytes.len() as u64
    }
    /// Framing and canonical coordinates grant no compiler or proof authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
}

/// Logical retained/scratch accounting quote, excluding caller input custody
/// and destination Vec header/backing. Prepay it against the original account.
/// It includes the fixed ordinal bitmap, simultaneous codec values, two hash
/// states, wire temporaries and two 128-byte hash-padding/word-scratch allowances.
/// This is not a theorem about compiler-generated stack frames or process RSS.
pub const RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1: usize = CENSUS_BYTES
    + size_of::<RustcEnrollmentInventoryRefV1<'static>>()
    + size_of::<RustcEnrollmentInventoryInputV1<'static>>()
    + size_of::<RustcEnrollmentInventoryHeaderV1>()
    + size_of::<RustcEnrollmentInventoryRootV1>()
    + size_of::<RustcEnrollmentInventoryLayoutV1>()
    + 2 * size_of::<Sha256>()
    + pair::OVERHEAD
    + HEADER
    + ROW
    + 2 * 128;

/// Conservative cumulative work for the reader at the existing aggregate cap.
/// This sums pair framing/hash, canonical row census and the complete raw hash;
/// it plans original-account funding, not parser admission or a renewed budget.
pub const RUSTC_ENROLLMENT_INVENTORY_MAX_READ_WORK_V1: usize = pair::HEADER
    + MAX
    + POLICY.domain.len()
    + 8
    + 128
    + MAX_ROOTS * (ROW + size_of::<RustcEnrollmentInventoryRootV1>() + 16)
    + CENSUS_BYTES
    + HEADER
    + MAX
    + 128;

fn storage<E>(limit: usize, owned: usize) -> Result<(), Error<E>> {
    let required = RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1
        .checked_add(owned)
        .ok_or(Error::Arithmetic)?;
    if limit > MAX_NATIVE_CONDITIONAL_STORAGE_V1 || limit < required {
        return Err(Error::StorageLimit);
    }
    Ok(())
}

fn input_layout<E>(
    input: RustcEnrollmentInventoryInputV1<'_>,
) -> Result<RustcEnrollmentInventoryLayoutV1, Error<E>> {
    let layout =
        RustcEnrollmentInventoryLayoutV1::new(input.legacy_inventory.len(), input.roots.len())?;
    if usize::try_from(input.header.kernel_count).map_err(|_| Error::Arithmetic)?
        != input.roots.len()
        || input.header.enrollment_binding_count > input.header.kernel_count
    {
        return Err(Error::Count);
    }
    Ok(layout)
}

fn walk_work<E>(layout: RustcEnrollmentInventoryLayoutV1) -> Result<usize, Error<E>> {
    layout
        .roots
        .checked_mul(ROW + size_of::<RustcEnrollmentInventoryRootV1>() + 16)
        .and_then(|n| n.checked_add(CENSUS_BYTES + HEADER))
        .ok_or(Error::Arithmetic)
}
fn encode_work<E>(
    layout: RustcEnrollmentInventoryLayoutV1,
    storage_limit: usize,
) -> Result<usize, Error<E>> {
    let seal = pair::seal_work(&POLICY, layout.pair, layout.encoded_len(), storage_limit)?;
    let walk = walk_work::<E>(layout)?;
    layout
        .encoded_len()
        .checked_mul(3)
        .and_then(|n| n.checked_add(walk))
        .and_then(|n| n.checked_add(seal))
        .ok_or(Error::Arithmetic)
}

/// Encode with one fallible exact-capacity allocation and no hidden input copies.
/// Caller prepays original input custody, the layout's owned-storage quote, and
/// the published working set. The returned capacity is exactly encoded_len().
/// All callback calls and callback destruction precede allocation and writes.
/// This function never refunds a caller account on error or unwind.
pub fn encode_rustc_enrollment_inventory_v1<E>(
    input: RustcEnrollmentInventoryInputV1<'_>,
    storage_limit: usize,
    mut charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<Vec<u8>, Error<E>> {
    let layout = input_layout(input)?;
    storage(storage_limit, layout.owned_storage_bytes()?)?;
    charge(encode_work(layout, storage_limit)?).map_err(Error::Charge)?;
    drop(charge);
    validate_roots(input.header, input.roots.iter().copied().map(Ok))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(layout.encoded_len())
        .map_err(|_| Error::Allocation)?;
    if bytes.capacity() != layout.encoded_len() {
        return Err(Error::Allocation);
    }
    bytes.resize(layout.encoded_len(), 0);
    write_input(input, layout, &mut bytes, storage_limit)?;
    Ok(bytes)
}

/// Write into an already-paid exact destination, without allocation.
/// Original inputs, destination backing and fixed working storage remain caller
/// charges. Invalid inputs, work denial and callback destruction all precede
/// destination mutation, including when callback destruction unwinds.
pub fn encode_rustc_enrollment_inventory_into_v1<E>(
    input: RustcEnrollmentInventoryInputV1<'_>,
    bytes: &mut [u8],
    storage_limit: usize,
    mut charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<(), Error<E>> {
    let layout = input_layout(input)?;
    storage(storage_limit, 0)?;
    if bytes.len() != layout.encoded_len() {
        return Err(Error::Length);
    }
    charge(encode_work(layout, storage_limit)?).map_err(Error::Charge)?;
    drop(charge);
    validate_roots(input.header, input.roots.iter().copied().map(Ok))?;
    write_input(input, layout, bytes, storage_limit)
}

fn write_input<E>(
    input: RustcEnrollmentInventoryInputV1<'_>,
    layout: RustcEnrollmentInventoryLayoutV1,
    bytes: &mut [u8],
    storage_limit: usize,
) -> Result<(), Error<E>> {
    bytes[layout.pair.first_range()].copy_from_slice(input.legacy_inventory);
    let association = &mut bytes[layout.pair.second_range()];
    let header = input.header;
    association[..4].copy_from_slice(&header.kernel_count.to_le_bytes());
    association[4..8].copy_from_slice(&header.enrollment_binding_count.to_le_bytes());
    association[8..40].copy_from_slice(&header.invocation_identity);
    association[40..72].copy_from_slice(&header.native_policy_identity);
    association[72..80].copy_from_slice(&header.native_policy_generation.to_le_bytes());
    for (out, root) in association[HEADER..].chunks_exact_mut(ROW).zip(input.roots) {
        out[..4].copy_from_slice(&root.semantic_root.to_le_bytes());
        out[4] = root.origin_tag;
        out[5..8].fill(0);
        out[8..12].copy_from_slice(&root.descriptor_ordinal.to_le_bytes());
        out[12..16].copy_from_slice(&root.logical_name_len.to_le_bytes());
        out[16..48].copy_from_slice(&root.logical_name_sha256);
        out[48..80].copy_from_slice(&root.kernel_binding);
        out[80..112].copy_from_slice(&root.kernel_instance);
        out[112..144].copy_from_slice(&root.reference_instance);
    }
    pair::seal(&POLICY, layout.pair, bytes, storage_limit, |_| {
        Ok::<_, E>(())
    })?;
    Ok(())
}

/// Strict borrowed reader with linear canonical-order and ordinal-census checks.
/// The complete wrapper and its full raw SHA256 are charged before they are read.
/// Callbacks are destroyed before the resulting inert view can escape. Consumers
/// must still revalidate their original account/owner after this call returns.
pub fn read_rustc_enrollment_inventory_v1<'a, E>(
    bytes: &'a [u8],
    storage_limit: usize,
    mut charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<RustcEnrollmentInventoryRefV1<'a>, Error<E>> {
    storage(storage_limit, 0)?;
    let (pair, identity) = pair::read(&POLICY, bytes, storage_limit, &mut charge)?;
    let association = &bytes[pair.second_range()];
    if association.len() < HEADER || (association.len() - HEADER) % ROW != 0 {
        return Err(Error::AssociationLength);
    }
    let roots = (association.len() - HEADER) / ROW;
    let layout = RustcEnrollmentInventoryLayoutV1 { pair, roots };
    let work = walk_work::<E>(layout)?
        .checked_add(bytes.len())
        .and_then(|n| n.checked_add(128))
        .ok_or(Error::Arithmetic)?;
    charge(work).map_err(Error::Charge)?;
    drop(charge);
    let header = header_value(&association[..HEADER]);
    validate_roots(
        header,
        association[HEADER..].chunks_exact(ROW).map(|row| {
            if row[5..8] != [0; 3] {
                Err(Error::Reserved)
            } else {
                Ok(root_value(row))
            }
        }),
    )?;
    Ok(RustcEnrollmentInventoryRefV1 {
        bytes,
        layout,
        framing_sha256: identity.sha256,
        full_wrapper_sha256: Sha256::digest(bytes).into(),
    })
}

fn validate_roots<E>(
    header: RustcEnrollmentInventoryHeaderV1,
    roots: impl ExactSizeIterator<Item = Result<RustcEnrollmentInventoryRootV1, Error<E>>>,
) -> Result<(), Error<E>> {
    let count = roots.len();
    if count > MAX_ROOTS
        || usize::try_from(header.kernel_count).map_err(|_| Error::Arithmetic)? != count
        || header.enrollment_binding_count > header.kernel_count
    {
        return Err(Error::Count);
    }
    let enrollment_count =
        usize::try_from(header.enrollment_binding_count).map_err(|_| Error::Arithmetic)?;
    let mut census = [0_u8; CENSUS_BYTES];
    let mut previous = None;
    let mut enrolled = 0_usize;
    for root in roots {
        let root = root?;
        if previous.is_some_and(|id| root.semantic_root <= id) {
            return Err(Error::RootOrder);
        }
        previous = Some(root.semantic_root);
        match root.origin_tag {
            0 if root.descriptor_ordinal == 0 => {}
            1 => {
                let ordinal =
                    usize::try_from(root.descriptor_ordinal).map_err(|_| Error::Arithmetic)?;
                if ordinal >= enrollment_count {
                    return Err(Error::EnrollmentOrdinal);
                }
                let mask = 1_u8 << (ordinal % 8);
                let slot = &mut census[ordinal / 8];
                if *slot & mask != 0 {
                    return Err(Error::EnrollmentOrdinal);
                }
                *slot |= mask;
                enrolled += 1;
            }
            _ => return Err(Error::Origin),
        }
    }
    // Range + uniqueness + exact cardinality independently establish completeness.
    if enrolled != enrollment_count {
        return Err(Error::EnrollmentOrdinal);
    }
    Ok(())
}

fn fixed<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    bytes[at..at + N]
        .try_into()
        .expect("bounded fixed inventory row")
}
fn header_value(bytes: &[u8]) -> RustcEnrollmentInventoryHeaderV1 {
    RustcEnrollmentInventoryHeaderV1 {
        kernel_count: u32::from_le_bytes(fixed(bytes, 0)),
        enrollment_binding_count: u32::from_le_bytes(fixed(bytes, 4)),
        invocation_identity: fixed(bytes, 8),
        native_policy_identity: fixed(bytes, 40),
        native_policy_generation: u64::from_le_bytes(fixed(bytes, 72)),
    }
}
fn root_value(bytes: &[u8]) -> RustcEnrollmentInventoryRootV1 {
    RustcEnrollmentInventoryRootV1 {
        semantic_root: u32::from_le_bytes(fixed(bytes, 0)),
        origin_tag: bytes[4],
        descriptor_ordinal: u32::from_le_bytes(fixed(bytes, 8)),
        logical_name_len: u32::from_le_bytes(fixed(bytes, 12)),
        logical_name_sha256: fixed(bytes, 16),
        kernel_binding: fixed(bytes, 48),
        kernel_instance: fixed(bytes, 80),
        reference_instance: fixed(bytes, 112),
    }
}

#[cfg(test)]
#[path = "rustc_enrollment_inventory_v1_tests.rs"]
mod tests;

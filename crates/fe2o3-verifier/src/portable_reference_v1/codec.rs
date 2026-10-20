//! Canonical, bounded transport of inert CPU reference inputs, not source custody.
//!
//! The consumer must bind the association/commitment and run the existing shared
//! replay/join. Neither decoded cached effects nor content hashes authenticate
//! source. V1 deliberately excludes loop summaries and helper calls.

use super::signature::*;
use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use fe2o3_mir_model::semantic_mir_v1::{
    HARD_MAX_FUNCTIONS_V1, HARD_MAX_LOCALS_V1, SemanticExternAbiV1, SemanticFunctionSafetyV1,
    SemanticMutabilityV1,
};
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

mod decode;
mod encode;
mod policy_v2;
mod tags;
mod validate;

pub use policy_v2::*;
use tags::*;

pub const MAX_NATIVE_CPU_INPUT_BYTES_V1: usize = crate::MAX_NATIVE_COMPILER_SOURCE_PACKET_BYTES_V1;
const MAGIC: &[u8; 8] = b"F2CPU1\0\0";
const DOMAIN: &[u8] = b"fe2o3/native-cpu-input/v1\0";
// These are V1 codec-domain bounds, not new global limits on the live engine.
// Locals/signature counts reuse semantic MIR and signature maxima respectively.
const DEPTH: usize = 128;
const TOTAL_NODES: usize = 262_144;

#[derive(Clone, Copy, Debug)]
pub struct NativeCpuAssociationV1<'a> {
    pub semantic_mir_sha256: [u8; 32],
    pub semantic_root: u32,
    pub registration_path: &'a str,
    pub logical_kernel_name: &'a str,
}

pub struct NativeCpuInputV1<'a> {
    pub association: NativeCpuAssociationV1<'a>,
    pub kernel: &'a ReferenceFunctionIdentityV1,
    pub reference: &'a ReferenceFunctionIdentityV1,
    pub replay: ReferenceReplayInputV1<'a>,
}

struct CpuSubject<'a> {
    semantic_mir_sha256: [u8; 32],
    semantic_root: u32,
    logical_kernel_name: &'a str,
    kernel: &'a ReferenceFunctionIdentityV1,
    reference: &'a ReferenceFunctionIdentityV1,
    replay: &'a ReferenceReplayInputV1<'a>,
}

impl NativeCpuInputV1<'_> {
    fn subject(&self) -> CpuSubject<'_> {
        CpuSubject {
            semantic_mir_sha256: self.association.semantic_mir_sha256,
            semantic_root: self.association.semantic_root,
            logical_kernel_name: self.association.logical_kernel_name,
            kernel: self.kernel,
            reference: self.reference,
            replay: &self.replay,
        }
    }
}

struct DecodedCpuSubject {
    logical_kernel_name: String,
    kernel: ReferenceFunctionIdentityV1,
    reference: ReferenceFunctionIdentityV1,
    signature: ReferenceLogicalSignaturePreimageV1,
    effect_ir_sha256: [u8; 32],
    ir: ReferenceEffectIrV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCpuCodecErrorV1 {
    Wire(&'static str),
    Resource(Resource),
    Signature(ReferenceSignatureErrorV1),
}

impl std::fmt::Display for NativeCpuCodecErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wire(reason) => write!(f, "native CPU input V1: {reason}"),
            Self::Resource(error) => error.fmt(f),
            Self::Signature(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for NativeCpuCodecErrorV1 {}
impl From<Resource> for NativeCpuCodecErrorV1 {
    fn from(error: Resource) -> Self {
        Self::Resource(error)
    }
}
impl From<ReferenceSignatureErrorV1> for NativeCpuCodecErrorV1 {
    fn from(error: ReferenceSignatureErrorV1) -> Self {
        Self::Signature(error)
    }
}
type Error = NativeCpuCodecErrorV1;

/// Private ownership of inert records; no authenticated binding constructor.
pub struct DecodedNativeCpuInputV1 {
    semantic_mir_sha256: [u8; 32],
    semantic_root: u32,
    registration_path: String,
    logical_kernel_name: String,
    kernel: ReferenceFunctionIdentityV1,
    reference: ReferenceFunctionIdentityV1,
    signature: ReferenceLogicalSignaturePreimageV1,
    effect_ir_sha256: [u8; 32],
    ir: ReferenceEffectIrV1,
    commitment: [u8; 32],
}

impl DecodedNativeCpuInputV1 {
    pub fn input_v1(&self) -> NativeCpuInputV1<'_> {
        NativeCpuInputV1 {
            association: NativeCpuAssociationV1 {
                semantic_mir_sha256: self.semantic_mir_sha256,
                semantic_root: self.semantic_root,
                registration_path: &self.registration_path,
                logical_kernel_name: &self.logical_kernel_name,
            },
            kernel: &self.kernel,
            reference: &self.reference,
            replay: ReferenceReplayInputV1 {
                signature_preimage: &self.signature,
                effect_ir: &self.ir,
                effect_ir_sha256: self.effect_ir_sha256,
                observable_output_writes: &self.ir.observable_output_effects,
            },
        }
    }
    pub fn commitment_v1(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Encodes one effect list only after bounded comparison of both live lists.
/// The callback borrows bytes whose allocation remains charged until drop.
pub fn with_encoded_native_cpu_input_v1<R>(
    input: NativeCpuInputV1<'_>,
    budget: &mut Budget<'_>,
    consume: impl for<'wire> FnOnce(&'wire [u8], [u8; 32], &mut Budget<'_>) -> R,
) -> Result<R, Error> {
    scoped(budget, |s| {
        let length = encode::frame(&input, false, 0, encode::Output::Count, s)?;
        let alternate = encode::frame(&input, true, 0, encode::Output::Count, s)?;
        require(length == alternate, "retained effect lengths differ")?;
        validate::input(&input, length, s)?;
        let mut bytes = s.vector::<u8>(length)?;
        encode::frame(&input, false, length, encode::Output::Fill(&mut bytes), s)?;
        // The alternate list is traversed with the same depth/work checks, not
        // compared by an unbounded recursive Eq before validation.
        encode::frame(&input, true, length, encode::Output::Compare(&bytes), s)?;
        let commitment = commitment(&bytes, s)?;
        let account = s.budget.work_ledger_identity_v1();
        let floor = s.budget.storage();
        let result = consume(&bytes, commitment, s.budget);
        drop(bytes);
        intact(s.budget, account, floor)?;
        Ok(result)
    })
}

/// Decodes on the original account; owned scratch drops before its refund.
/// Only the existing borrowed replay input is exposed, with one shared list.
///
/// ```compile_fail
/// use fe2o3_verifier::portable_reference_v1::codec::*;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
/// fn escape<'a>(bytes: &[u8], budget: &mut Budget<'_>) -> &'a DecodedNativeCpuInputV1 {
///     with_decoded_native_cpu_input_v1(bytes, budget, |owner, _| owner).unwrap()
/// }
/// ```
pub fn with_decoded_native_cpu_input_v1<R>(
    bytes: &[u8],
    budget: &mut Budget<'_>,
    consume: impl for<'cpu> FnOnce(&'cpu DecodedNativeCpuInputV1, &mut Budget<'_>) -> R,
) -> Result<R, Error> {
    scoped(budget, |s| {
        require(bytes.len() <= MAX_NATIVE_CPU_INPUT_BYTES_V1, "frame limit")?;
        let mut owner = decode::frame(bytes, s)?;
        encode::frame(
            &owner.input_v1(),
            false,
            bytes.len(),
            encode::Output::Compare(bytes),
            s,
        )?;
        validate::input(&owner.input_v1(), bytes.len(), s)?;
        owner.commitment = commitment(bytes, s)?;
        let account = s.budget.work_ledger_identity_v1();
        let floor = s.budget.storage();
        let result = consume(&owner, s.budget);
        drop(owner);
        intact(s.budget, account, floor)?;
        Ok(result)
    })
}

fn commitment(bytes: &[u8], s: &mut Scope<'_, '_>) -> Result<[u8; 32], Error> {
    s.work(add(bytes.len(), DOMAIN.len())?)?;
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    hash.update(bytes);
    Ok(hash.finalize().into())
}

fn intact(
    budget: &Budget<'_>,
    account: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    floor: usize,
) -> Result<(), Error> {
    if budget.work_ledger_identity_v1() != account || budget.storage() < floor {
        return Err(Resource::Accounting.into());
    }
    Ok(())
}

struct Scope<'b, 'w> {
    budget: &'b mut Budget<'w>,
    reserved: &'b mut usize,
}
impl Scope<'_, '_> {
    fn work(&mut self, amount: usize) -> Result<(), Error> {
        self.budget.charge_work(amount).map_err(Into::into)
    }
    fn reserve(&mut self, amount: usize) -> Result<(), Error> {
        let next = add(*self.reserved, amount)?;
        self.budget.reserve_storage(amount)?;
        *self.reserved = next;
        Ok(())
    }
    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>, Error> {
        let payload = mul(count, size_of::<T>())?;
        self.work(add(payload, 1)?)?;
        self.reserve(add(payload, size_of::<Vec<T>>())?)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(count)
            .map_err(|_| Resource::Allocation)?;
        if rows.capacity() != count {
            return Err(Resource::Allocation.into());
        }
        Ok(rows)
    }
    fn boxed<T>(&mut self, value: T) -> Result<Box<T>, Error> {
        // Codec recursive nodes are non-ZST. Reject ZST explicitly rather than
        // rely on Vec's usize::MAX ZST capacity to fail the exact-capacity guard.
        if size_of::<T>() == 0 {
            return Err(Resource::Allocation.into());
        }
        let mut row = self.vector(1)?;
        row.push(value);
        let row = row.into_boxed_slice();
        // SAFETY: vector(1) prepaid and fallibly obtained capacity exactly one
        // non-ZST T from the global allocator, with alignment align_of::<T>().
        // After the single push, len == capacity == 1, so into_boxed_slice has
        // no excess capacity to shrink. Its allocation layout is exactly T's.
        // into_raw transfers the sole owner; the cast removes only slice length
        // metadata. from_raw takes that same initialized allocation once, with
        // the same allocator/layout and no surviving alias or second drop.
        Ok(unsafe { Box::from_raw(Box::into_raw(row) as *mut T) })
    }
}

fn scoped<R>(
    budget: &mut Budget<'_>,
    operation: impl FnOnce(&mut Scope<'_, '_>) -> Result<R, Error>,
) -> Result<R, Error> {
    let account = budget.work_ledger_identity_v1();
    let floor = budget.storage();
    let mut reserved = 0;
    let result = catch_unwind(AssertUnwindSafe(|| {
        operation(&mut Scope {
            budget,
            reserved: &mut reserved,
        })
    }));
    // Every owner made by operation has dropped here, including on unwind.
    let cleanup = match floor.checked_add(reserved) {
        Some(protected) if intact(budget, account, protected).is_ok() => {
            budget.release_storage(reserved).map_err(Error::from)
        }
        _ => Err(Resource::Accounting.into()),
    };
    match result {
        Ok(result) => {
            cleanup?;
            result
        }
        Err(panic) => resume_unwind(panic),
    }
}

fn require(ok: bool, reason: &'static str) -> Result<(), Error> {
    if ok { Ok(()) } else { Err(Error::Wire(reason)) }
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Resource::Arithmetic.into())
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Resource::Arithmetic.into())
}
fn scalar_bits(scalar: ReferenceScalarTypeV1, bits: u128) -> Result<(), Error> {
    use ReferenceScalarTypeV1::*;
    let width = match scalar {
        Bool => 1,
        U8 | I8 => 8,
        U16 | I16 => 16,
        U32 | I32 | F32 => 32,
        U64 | Usize | I64 | Isize | F64 => 64,
    };
    require(bits < (1u128 << width), "noncanonical scalar bits")
}

#[cfg(test)]
mod tests;

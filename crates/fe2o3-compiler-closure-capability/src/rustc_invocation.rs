use std::fs::File;
use std::os::fd::RawFd;
use std::process::Command;

use fe2o3_rustc_invocation::{
    MAX_DESCRIPTOR_BYTES_V3, RustcInvocationDescriptorV3, decode_descriptor_v3,
    encode_descriptor_v3,
};

use crate::native_capability::{
    CompilerExecutionCapabilityErrorV2 as NativeError,
    CompilerExecutionCapabilityStorageV2 as NativeStorage, envelope_overhead,
};
use crate::sealed_image::{CapabilityRole, ImageLength, SealedCapabilityImage};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use std::mem::size_of;

const ROLE: CapabilityRole = CapabilityRole {
    name: "rustc-invocation capability",
    memfd_name: "fe2o3-rustc-invocation-capability-v1",
};
const LENGTH: ImageLength = ImageLength::Bounded {
    max: MAX_DESCRIPTOR_BYTES_V3,
};

/// Reserved descriptor used to pass the canonical invocation from a wrapper into rustc.
pub const RUSTC_INVOCATION_CHILD_FD_V1: RawFd = 199;

/// An immutable file capability containing one canonical V3 rustc invocation descriptor.
pub struct RustcInvocationCapabilityV1 {
    descriptor: RustcInvocationDescriptorV3,
    // Immutable capacity accounting, not receipt provenance or approval.
    decoded_storage: usize,
    canonical_bytes: Vec<u8>,
    pub(super) image: SealedCapabilityImage,
}

impl RustcInvocationCapabilityV1 {
    /// Logical input charge for the consumed descriptor; image/decoded storage
    /// is discovered, admitted and returned as growth during native admission.
    pub const NATIVE_FILE_STORAGE: usize = size_of::<(File, NativeStorage)>();
    const NATIVE_IO_WORK: usize = 64 * 1024;
    const NATIVE_FRAME: usize = 8192 + 4 * size_of::<Self>();
    const NATIVE_DECODE_WORK_PER_BYTE: usize = 4096;
    const NATIVE_COMPARE_WORK_PER_BYTE: usize = 8;
    const NATIVE_STORAGE_PER_BYTE: usize = 64;

    /// Conservative full owner bound for capacity-checked native admission.
    pub const NATIVE_MAX_RETAINED_STORAGE: usize =
        Self::NATIVE_STORAGE_PER_BYTE * MAX_DESCRIPTOR_BYTES_V3 + Self::NATIVE_FRAME;
    /// Complete nested admission work; the input File owner is prepaid separately.
    pub const NATIVE_ADMISSION_WORK: usize =
        2 * Self::NATIVE_IO_WORK + Self::NATIVE_DECODE_WORK_PER_BYTE * MAX_DESCRIPTOR_BYTES_V3;
    /// Complete nested currentness work, excluding the retained owner floor.
    pub const NATIVE_REVALIDATION_WORK: usize =
        2 * Self::NATIVE_IO_WORK + Self::NATIVE_COMPARE_WORK_PER_BYTE * MAX_DESCRIPTOR_BYTES_V3;
    /// Peak nested operation scratch, above the input/retained owner floor.
    pub const NATIVE_OPERATION_SCRATCH: usize =
        Self::NATIVE_FRAME + Self::NATIVE_MAX_RETAINED_STORAGE;
    /// Complete native transfer cost, including exact retained-byte revalidation.
    pub const NATIVE_TRANSFER_WORK: usize = Self::NATIVE_IO_WORK + Self::NATIVE_REVALIDATION_WORK;
    pub const NATIVE_TRANSFER_SCRATCH: usize =
        Self::NATIVE_FRAME + Self::NATIVE_FILE_STORAGE + Self::NATIVE_OPERATION_SCRATCH;

    /// Duplicates the genuine sealed invocation without extracting its owner.
    /// The returned File charge is unreserved; keep the original full owner
    /// prepaid and reserve the charge before retaining the transfer. This is
    /// mechanical immutable-file custody, never capture or execution authority.
    pub fn try_clone_for_transfer_native(
        &self,
        budget: &mut Budget<'_>,
    ) -> Result<(File, NativeStorage), NativeError> {
        budget.with_prepaid_scope(
            self.native_retained_storage()?,
            8,
            Self::NATIVE_IO_WORK,
            Self::NATIVE_FRAME + Self::NATIVE_FILE_STORAGE,
            |b| {
                self.revalidate_native(b)?;
                Ok((
                    self.image.clone_fixed()?,
                    NativeStorage(Self::NATIVE_FILE_STORAGE),
                ))
            },
        )
    }

    /// Fresh metered admission of the existing canonical V3 invocation format.
    /// No V1 compiler-execution subject, service or policy owner is constructed.
    /// The frozen decoder is prepaid in full before entry: bounded field scans,
    /// UTF-8/path validation, fixed environment searches, canonical re-encodes
    /// and the decoded-capacity traversal are linear in its input. This is
    /// logical work/storage, not allocator RSS or a claim that the shared
    /// decoder's internal allocations are fallible.
    pub fn from_file_native(
        image: File,
        budget: &mut Budget<'_>,
    ) -> Result<(Self, NativeStorage), NativeError> {
        budget.with_prepaid_scope(
            Self::NATIVE_FILE_STORAGE,
            8,
            Self::NATIVE_IO_WORK,
            Self::NATIVE_FRAME,
            |budget| {
                let image = SealedCapabilityImage::from_file_bounded_native(
                    image,
                    ROLE,
                    MAX_DESCRIPTOR_BYTES_V3,
                )?;
                let length = image.native_length();
                Self::native_contents_scope(budget, length, true, |_: &mut Budget<'_>| {
                    let canonical_bytes = image.read_bounded_native()?;
                    let descriptor = decode_descriptor_v3(&canonical_bytes)
                        .map_err(|_| NativeError::Rejected("invalid canonical V3 invocation"))?;
                    let decoded_storage = descriptor
                        .retained_storage_bytes()
                        .ok_or(Resource::Arithmetic)?;
                    let admitted = Self {
                        descriptor,
                        decoded_storage,
                        canonical_bytes,
                        image,
                    };
                    let storage = admitted
                        .native_retained_storage()?
                        .checked_sub(Self::NATIVE_FILE_STORAGE)
                        .ok_or(Resource::Accounting)?;
                    Ok((admitted, NativeStorage(storage)))
                })
            },
        )
    }

    /// Rechecks original sealed-object identity and exact canonical bytes. The
    /// decoded descriptor is immutable, so equal bytes need no second decode.
    pub fn revalidate_native(&self, budget: &mut Budget<'_>) -> Result<(), NativeError> {
        budget.with_prepaid_scope(
            self.native_retained_storage()?,
            8,
            Self::NATIVE_IO_WORK,
            Self::NATIVE_FRAME,
            |budget| {
                Self::native_contents_scope(budget, self.canonical_bytes.len(), false, |_| {
                    if self.image.read_bounded_native()? != self.canonical_bytes {
                        return Err(NativeError::Rejected("retained invocation bytes changed"));
                    }
                    Ok(())
                })
            },
        )
    }

    /// Conservative full owner charge, including canonical and decoded backing.
    /// Rejects legacy-created owners whose spare capacity exceeds the native
    /// bound. Cached decoded capacity makes this check allocation-free and
    /// constant-time; neither the cache nor this check establishes provenance.
    pub fn native_retained_storage(&self) -> Result<usize, NativeError> {
        let actual = self
            .decoded_storage
            .checked_sub(size_of::<RustcInvocationDescriptorV3>())
            .and_then(|n| n.checked_add(size_of::<(Self, NativeStorage)>()))
            .and_then(|n| n.checked_add(envelope_overhead::<(Self, NativeStorage), NativeError>()))
            .and_then(|n| n.checked_add(self.canonical_bytes.capacity()))
            .and_then(|n| n.checked_add(self.image.native_length()))
            .ok_or(Resource::Arithmetic)?;
        let bound = Self::native_storage_for(self.canonical_bytes.len())?;
        if actual > bound {
            return Err(NativeError::Rejected(
                "retained invocation capacity exceeds native owner bound",
            ));
        }
        Ok(bound)
    }

    fn native_storage_for(length: usize) -> Result<usize, NativeError> {
        // Each string header consumes at least a 4-byte length in the input;
        // environment headers consume at least 6. Includes both vector stages,
        // immutable bytes, repeated encoder buffers and result envelopes.
        length
            .checked_mul(Self::NATIVE_STORAGE_PER_BYTE)
            .and_then(|n| n.checked_add(Self::NATIVE_FRAME))
            .ok_or_else(|| Resource::Arithmetic.into())
    }

    fn native_contents_scope<T>(
        budget: &mut Budget<'_>,
        length: usize,
        decode: bool,
        operation: impl FnOnce(&mut Budget<'_>) -> Result<T, NativeError>,
    ) -> Result<T, NativeError> {
        if length == 0 || length > MAX_DESCRIPTOR_BYTES_V3 {
            return Err(NativeError::Rejected(
                "invocation length exceeds canonical V3 bound",
            ));
        }
        // 4096 covers per-byte scanning/copying and the bounded V2-body/V3
        // validator/encoder passes and decoded-capacity traversal (at most one
        // argument per four bytes and one environment entry per six bytes);
        // fixed closure hashing is separately covered.
        let work = length
            .checked_mul(if decode {
                Self::NATIVE_DECODE_WORK_PER_BYTE
            } else {
                Self::NATIVE_COMPARE_WORK_PER_BYTE
            })
            .and_then(|n| n.checked_add(Self::NATIVE_IO_WORK))
            .ok_or(Resource::Arithmetic)?;
        budget.with_prepaid_scope(0, 8, work, Self::native_storage_for(length)?, operation)
    }

    /// Creates and seals the canonical encoding of one validated V3 invocation descriptor.
    pub fn create(descriptor: RustcInvocationDescriptorV3) -> Result<Self, String> {
        let decoded_storage = descriptor
            .retained_storage_bytes()
            .ok_or("rustc-invocation decoded storage overflow")?;
        let canonical_bytes = encode_descriptor_v3(&descriptor)
            .map_err(|error| format!("cannot encode rustc-invocation capability: {error}"))?;
        let image = SealedCapabilityImage::create(&canonical_bytes, ROLE, LENGTH)?;
        let admitted = Self {
            descriptor,
            decoded_storage,
            canonical_bytes,
            image,
        };
        admitted.revalidate()?;
        Ok(admitted)
    }

    /// Admits an owned close-on-exec descriptor carrying canonical V3 bytes.
    pub fn from_file(image: File) -> Result<Self, String> {
        Self::from_image(SealedCapabilityImage::from_file(image, ROLE, LENGTH)?)
    }

    /// Admits the invocation inherited at the canonical wrapper-to-rustc descriptor number.
    pub fn from_inherited_child() -> Result<Self, String> {
        Self::from_inherited_at(RUSTC_INVOCATION_CHILD_FD_V1)
    }

    /// Retains and admits an inherited, intentionally non-close-on-exec descriptor.
    pub fn from_inherited_at(child_fd: RawFd) -> Result<Self, String> {
        Self::from_image(SealedCapabilityImage::from_inherited_at(
            child_fd, ROLE, LENGTH,
        )?)
    }

    /// Returns the exact canonical descriptor carried by this capability.
    pub const fn descriptor(&self) -> &RustcInvocationDescriptorV3 {
        &self.descriptor
    }

    /// Borrows the retained canonical encoding without re-encoding or allocating.
    /// These bytes describe an invocation; they do not establish live-process custody.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    /// Revalidates file identity and transport invariants, then decodes and re-encodes exact V3 bytes.
    pub fn revalidate(&self) -> Result<(), String> {
        let bytes = self.image.read_exact_bytes()?;
        let descriptor = decode_canonical(&bytes)?;
        if bytes != self.canonical_bytes || descriptor != self.descriptor {
            return Err("rustc-invocation capability bytes changed".to_owned());
        }
        Ok(())
    }

    /// Clones the exact sealed descriptor for one broker or process-boundary transfer.
    pub fn try_clone_for_transfer(&self) -> Result<File, String> {
        self.revalidate()?;
        self.image.try_clone_for_transfer()
    }

    /// Installs this exact image at the canonical wrapper-to-rustc child descriptor.
    pub fn inherit_for_child(&self, command: &mut Command) -> Result<(), String> {
        self.inherit_for_child_at(command, RUSTC_INVOCATION_CHILD_FD_V1)
    }

    /// Installs this exact image at one unoccupied child descriptor.
    pub fn inherit_for_child_at(
        &self,
        command: &mut Command,
        child_fd: RawFd,
    ) -> Result<(), String> {
        self.revalidate()?;
        self.image.inherit_for_child_at(command, child_fd)
    }

    fn from_image(image: SealedCapabilityImage) -> Result<Self, String> {
        let canonical_bytes = image.read_exact_bytes()?;
        let descriptor = decode_canonical(&canonical_bytes)?;
        let decoded_storage = descriptor
            .retained_storage_bytes()
            .ok_or("rustc-invocation decoded storage overflow")?;
        let admitted = Self {
            descriptor,
            decoded_storage,
            canonical_bytes,
            image,
        };
        admitted.revalidate()?;
        Ok(admitted)
    }
}

fn decode_canonical(bytes: &[u8]) -> Result<RustcInvocationDescriptorV3, String> {
    let descriptor = decode_descriptor_v3(bytes)
        .map_err(|error| format!("rustc-invocation capability is not canonical V3: {error}"))?;
    let reencoded = encode_descriptor_v3(&descriptor)
        .map_err(|error| format!("cannot re-encode rustc-invocation capability: {error}"))?;
    if reencoded != bytes {
        return Err("rustc-invocation capability has noncanonical V3 bytes".to_owned());
    }
    Ok(descriptor)
}

#[cfg(test)]
#[path = "rustc_invocation_native_tests.rs"]
mod native_tests;

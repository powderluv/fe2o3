//! Descriptive policy-origin CPU records, never live enrollment authority.

use super::*;

pub(super) const POLICY_MAGIC: &[u8; 8] = b"F2CPU2\0\0";
pub(super) const POLICY_DOMAIN: &[u8] = b"fe2o3/native-cpu-input/v2\0";
pub(super) const ADMITTED_POLICY_TAG: u8 = 1;
pub(super) const REFERENCE_ENROLLMENT_DOMAIN_V1: u16 = 1;
pub const MAX_REFERENCE_ENROLLMENT_BINDINGS_V1: u32 = 256;

/// Describes ReferenceEnrollmentV1 selection in the original invocation.
/// These public bytes cannot construct a live session, stamp, or source owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceEnrollmentOriginV1 {
    pub rustc_invocation_sha256: [u8; 32],
    pub native_policy_sha256: [u8; 32],
    pub policy_generation: u64,
    pub mapping_ordinal: u32,
}

/// Inert origin description. Registration records retain the exact V1 codec;
/// policy records use the separate V2 codec and never invent a registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeCpuOriginV2<'a> {
    SourceRegistration { registration_path: &'a str },
    AdmittedPolicy(ReferenceEnrollmentOriginV1),
}

#[derive(Clone, Copy, Debug)]
pub struct NativeCpuPolicyAssociationV2<'a> {
    pub semantic_mir_sha256: [u8; 32],
    pub semantic_root: u32,
    pub logical_kernel_name: &'a str,
    pub origin: ReferenceEnrollmentOriginV1,
}

pub struct NativeCpuPolicyInputV2<'a> {
    pub association: NativeCpuPolicyAssociationV2<'a>,
    pub kernel: &'a ReferenceFunctionIdentityV1,
    pub reference: &'a ReferenceFunctionIdentityV1,
    pub replay: ReferenceReplayInputV1<'a>,
}

impl NativeCpuInputV1<'_> {
    pub fn origin_v2(&self) -> NativeCpuOriginV2<'_> {
        NativeCpuOriginV2::SourceRegistration {
            registration_path: self.association.registration_path,
        }
    }
}

impl NativeCpuPolicyInputV2<'_> {
    pub fn origin_v2(&self) -> NativeCpuOriginV2<'_> {
        NativeCpuOriginV2::AdmittedPolicy(self.association.origin)
    }

    pub(super) fn subject(&self) -> CpuSubject<'_> {
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

/// Callback-scoped, decoded data. Successful decoding is not source admission.
pub struct DecodedNativeCpuPolicyInputV2 {
    pub(super) semantic_mir_sha256: [u8; 32],
    pub(super) semantic_root: u32,
    pub(super) origin: ReferenceEnrollmentOriginV1,
    pub(super) subject: DecodedCpuSubject,
    pub(super) commitment: [u8; 32],
}

impl DecodedNativeCpuPolicyInputV2 {
    pub fn input_v2(&self) -> NativeCpuPolicyInputV2<'_> {
        NativeCpuPolicyInputV2 {
            association: NativeCpuPolicyAssociationV2 {
                semantic_mir_sha256: self.semantic_mir_sha256,
                semantic_root: self.semantic_root,
                logical_kernel_name: &self.subject.logical_kernel_name,
                origin: self.origin,
            },
            kernel: &self.subject.kernel,
            reference: &self.subject.reference,
            replay: ReferenceReplayInputV1 {
                signature_preimage: &self.subject.signature,
                effect_ir_sha256: self.subject.effect_ir_sha256,
                effect_ir: &self.subject.ir,
                observable_output_writes: &self.subject.ir.observable_output_effects,
            },
        }
    }

    pub fn commitment_v2(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Encodes descriptive policy selection and the complete existing CPU subject.
/// Admission, original-owner currentness and source resolution remain external.
pub fn with_encoded_native_cpu_policy_input_v2<R>(
    input: NativeCpuPolicyInputV2<'_>,
    budget: &mut Budget<'_>,
    consume: impl for<'cpu> FnOnce(&'cpu [u8], [u8; 32], &mut Budget<'_>) -> R,
) -> Result<R, NativeCpuCodecErrorV1> {
    scoped(budget, |s| {
        let subject = input.subject();
        let origin = input.origin_v2();
        let length = encode::subject_frame(&subject, origin, false, 0, encode::Output::Count, s)?;
        let alternate = encode::subject_frame(&subject, origin, true, 0, encode::Output::Count, s)?;
        require(length == alternate, "effect-list length")?;
        validate::subject(&subject, length, s)?;
        let mut bytes = s.vector(length)?;
        encode::subject_frame(
            &subject,
            origin,
            false,
            length,
            encode::Output::Fill(&mut bytes),
            s,
        )?;
        encode::subject_frame(
            &subject,
            origin,
            true,
            length,
            encode::Output::Compare(&bytes),
            s,
        )?;
        let commitment = policy_commitment(&bytes, s)?;
        let account = s.budget.work_ledger_identity_v1();
        let protected = s.budget.storage();
        let result = consume(&bytes, commitment, s.budget);
        drop(bytes);
        intact(s.budget, account, protected)?;
        Ok(result)
    })
}

/// Decodes only V2; it never accepts a registration V1 frame as policy selection.
/// The owned body cannot escape its original charged callback scope:
///
/// ```compile_fail
/// use fe2o3_verifier::portable_reference_v1::codec::*;
/// use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
/// fn escape<'a>(bytes: &[u8], budget: &mut Budget<'_>) -> &'a DecodedNativeCpuPolicyInputV2 {
///     with_decoded_native_cpu_policy_input_v2(bytes, budget, |owner, _| owner).unwrap()
/// }
/// ```
pub fn with_decoded_native_cpu_policy_input_v2<R>(
    bytes: &[u8],
    budget: &mut Budget<'_>,
    consume: impl for<'cpu> FnOnce(&'cpu DecodedNativeCpuPolicyInputV2, &mut Budget<'_>) -> R,
) -> Result<R, NativeCpuCodecErrorV1> {
    scoped(budget, |s| {
        require(bytes.len() <= MAX_NATIVE_CPU_INPUT_BYTES_V1, "frame limit")?;
        let mut owner = decode::policy_frame(bytes, s)?;
        let input = owner.input_v2();
        encode::subject_frame(
            &input.subject(),
            input.origin_v2(),
            false,
            bytes.len(),
            encode::Output::Compare(bytes),
            s,
        )?;
        validate::subject(&input.subject(), bytes.len(), s)?;
        owner.commitment = policy_commitment(bytes, s)?;
        let account = s.budget.work_ledger_identity_v1();
        let protected = s.budget.storage();
        let result = consume(&owner, s.budget);
        drop(owner);
        intact(s.budget, account, protected)?;
        Ok(result)
    })
}

fn policy_commitment(bytes: &[u8], s: &mut Scope<'_, '_>) -> Result<[u8; 32], Error> {
    s.work(add(POLICY_DOMAIN.len(), bytes.len())?)?;
    let mut hash = Sha256::new();
    hash.update(POLICY_DOMAIN);
    hash.update(bytes);
    Ok(hash.finalize().into())
}

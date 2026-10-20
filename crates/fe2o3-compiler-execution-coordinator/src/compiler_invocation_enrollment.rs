//! Descriptive enrollment coordinates from the original sealed compiler backing.
use super::*;
use fe2o3_rustc_invocation::{
    INVOCATION_DIGEST_DOMAIN_V3, InvocationDigestV3,
    ReferenceEnrollmentDecodeErrorV1 as DecodeError, ReferenceEnrollmentRequestV1 as Request,
};
use sha2::{Digest, Sha256};

/// Neither these bytes nor a matching inventory header confer authority.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct OriginalInvocationEnrollment {
    pub(crate) rustc_invocation_sha256: [u8; 32],
    pub(crate) intake_invocation_identity: [u8; 32],
    pub(crate) invocation_bytes: u64,
    pub(crate) binding_count: Option<u32>,
}

impl OriginalInvocationEnrollment {
    // Captured key scan, the shared decoder's documented source-work quote,
    // and both canonical-byte digests. This is not a parser storage grant.
    pub(crate) const PROJECTION_WORK: usize = ENTRY
        + MAX_COMPILE_ENVIRONMENT_ENTRIES_V2 * (fe2o3_rustc_invocation::MAX_NAME_BYTES_V2 + 1)
        + 1
        + 64 * fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BYTES_V1
        + size_of::<Request>()
        + 2 * fe2o3_rustc_invocation::MAX_DESCRIPTOR_BYTES_V3
        + INVOCATION_DIGEST_DOMAIN_V3.len()
        + size_of::<u64>()
        + 1;
    pub(crate) const PROJECTION_SCRATCH: usize = SCRATCH;
}

// Logical digest/result storage only, not an admitted native parser profile.
const SCRATCH: usize = size_of::<(
    Sha256,
    InvocationDigestV3,
    OriginalInvocationEnrollment,
    Option<usize>,
    Option<u32>,
)>();

impl CompilerInvocationBacking {
    /// Query only this retained capture after checking its sealed object, full
    /// runtime inventory, output and original Budget. No replacement descriptor,
    /// count, policy or inventory header is accepted. Caller retains the complete
    /// backing and funds the returned descriptive value while it remains live.
    /// This does not qualify the compiler's dynamic closure or PREPARSE storage.
    pub(crate) fn original_enrollment(
        &self,
        b: &mut Budget<'_>,
    ) -> Result<OriginalInvocationEnrollment> {
        b.check_prior_denials_v1()?;
        self.revalidate(b)?;
        project(&self.capture, b)
    }
}

// This private calculation alone is inert: the owning Backing performs the
// original runtime/account checks above. Never export it as an admission API.
fn project(capture: &Capture, b: &mut Budget<'_>) -> Result<OriginalInvocationEnrollment> {
    b.check_prior_denials_v1()?;
    b.with_prepaid_scope(
        capture.native_retained_storage()?,
        ENTRY,
        ENTRY,
        SCRATCH,
        |b| {
            let count = Request::project_binding_count_from_descriptor(capture.descriptor(), |n| {
                b.charge_work(n)
            })
            .map_err(|error| match error {
                DecodeError::Work(error) => CompilerInvocationBackingError::Resource(error),
                other => CompilerInvocationBackingError::Enrollment(other),
            })?;
            let binding_count = count
                .map(u32::try_from)
                .transpose()
                .map_err(|_| CompilerInvocationBackingError::InvalidCount)?;
            let bytes = capture.canonical_bytes();
            b.charge_work(
                bytes
                    .len()
                    .checked_mul(2)
                    .and_then(|n| {
                        n.checked_add(INVOCATION_DIGEST_DOMAIN_V3.len() + size_of::<u64>() + 1)
                    })
                    .ok_or(Resource::Arithmetic)?,
            )?;
            Ok(OriginalInvocationEnrollment {
                rustc_invocation_sha256: Sha256::digest(bytes).into(),
                intake_invocation_identity: InvocationDigestV3::calculate_encoded(bytes)
                    .map_err(CompilerInvocationBackingError::Digest)?
                    .into_bytes(),
                invocation_bytes: u64::try_from(bytes.len()).map_err(|_| Resource::Arithmetic)?,
                binding_count,
            })
        },
    )
}

#[cfg(test)]
#[path = "compiler_invocation_enrollment_tests.rs"]
mod tests;

//! Original-attempt enrollment query; no wire change or mapped-consumer activation.
use super::*;
use crate::proof_helper_backing::ProofHelperBackingError;
use crate::proof_helper_launch::{ManagedProofHelper as Helper, ProofHelperLaunchError as Failure};

/// Inert coordinates obtained together through original attempt custody. These
/// may describe verification expectations, never substitute for their live owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OriginalCompilerEnrollment {
    pub(crate) rustc_invocation_sha256: [u8; 32],
    pub(crate) intake_invocation_identity: [u8; 32],
    pub(crate) invocation_bytes: u64,
    pub(crate) native_policy_sha256: [u8; 32],
    pub(crate) policy_generation: u64,
    pub(crate) binding_count: Option<u32>,
}

impl OriginalCompilerEnrollment {
    pub(crate) const STORAGE: usize = size_of::<Self>();

    // The original account remains live while the value is borrowed. Neither
    // the value nor its reservation escapes this scope, even on error/unwind.
    fn with_reserved<R>(
        self,
        account: &RequestAccount,
        retained: usize,
        b: &mut Budget<'_>,
        operation: impl FnOnce(&Self, &mut Budget<'_>) -> std::result::Result<R, Failure>,
    ) -> std::result::Result<R, Failure> {
        b.check_prior_denials_v1()?;
        account
            .with(retained, b, |b| {
                Ok(b.with_prepaid_scope(0, 0, 0, Self::STORAGE, |b| {
                    let result = operation(&self, b);
                    b.check_prior_denials_v1()?;
                    result
                }))
            })
            .map_err(Failure::from)?
    }
}

const SCRATCH: usize =
    size_of::<crate::compiler_invocation_backing::OriginalInvocationEnrollment>()
        + size_of::<([u8; 32], u64)>()
        + size_of::<(OriginalCompilerEnrollment, Storage)>();

impl NativeAttempt<'_, Helper> {
    /// Borrow independently derived coordinates on the same original account.
    /// The callback funds its own work/outputs; it cannot retain this borrow.
    pub(crate) fn with_original_enrollment<R>(
        &self,
        b: &mut Budget<'_>,
        operation: impl FnOnce(
            &OriginalCompilerEnrollment,
            &mut Budget<'_>,
        ) -> std::result::Result<R, Failure>,
    ) -> std::result::Result<R, Failure> {
        self.original_enrollment(b)?
            .with_reserved(&self.account, self.retained, b, operation)
    }

    /// Original-attempt custody and scoped result only. The request schedule
    /// separately funds full helper/backing validation and count projection.
    pub(crate) fn original_enrollment_custody_quota() -> Result<Quota> {
        let original = Self::original_validation_quota();
        let ready = Prepared::maximum_issuer_continuity_quota::<Helper>()?;
        let policy = Self::original_policy_identity_quota();
        Ok(Quota {
            work: sum(&[
                8,
                original.work(),
                original.work(),
                ready.work(),
                policy.work(),
                LOCAL_WORK,
            ])?,
            scratch: sum(&[
                SCRATCH,
                original.scratch(),
                original.scratch(),
                ready.scratch(),
                policy.scratch(),
                FRAME,
                OriginalCompilerEnrollment::STORAGE,
            ])?,
        })
    }

    /// All inputs come from this same attempt: original trace/root association,
    /// retained issuer policy and the helper's original sealed compiler backing.
    /// Existing live-issuer/helper and original-account checks remain mandatory.
    /// Nested work is charged on that account; only the scoped accessor above
    /// exposes the result. No complete dynamic compiler/PREPARSE qualification,
    /// transport authority or permission to activate mapped recovery is returned.
    fn original_enrollment(
        &self,
        b: &mut Budget<'_>,
    ) -> std::result::Result<OriginalCompilerEnrollment, Failure> {
        b.check_prior_denials_v1()?;
        b.with_prepaid_scope(self.retained, 8, 8, SCRATCH, |b| {
            self.validate_original(b)?;
            self.validate_ready(b)?;
            let (native_policy_sha256, policy_generation) = self.original_policy_coordinates(b)?;
            let invocation = self.trace.with_backing(b, |helper, b| {
                helper.with_compiler(b, |compiler, b| {
                    compiler
                        .original_enrollment(b)
                        .map_err(ProofHelperBackingError::from)
                        .map_err(Failure::from)
                })
            })?;
            self.validate_original(b)?;
            Ok(OriginalCompilerEnrollment {
                rustc_invocation_sha256: invocation.rustc_invocation_sha256,
                intake_invocation_identity: invocation.intake_invocation_identity,
                invocation_bytes: invocation.invocation_bytes,
                native_policy_sha256,
                policy_generation,
                binding_count: invocation.binding_count,
            })
        })
    }
}

#[cfg(test)]
#[path = "native_root_enrollment_tests.rs"]
mod tests;

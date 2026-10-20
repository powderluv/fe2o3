//! One-shot nominal dispatch over the loader's original owned inputs.
use super::{
    AdmittedProtectedCompilerExecutionV1 as Legacy, CompilerExecutionStartupInputV1 as Startup,
    OwnedExecutionInputs, native_v3,
};
use crate::production_pipeline::{
    CollectedRustStage, ProductionCompilation, ProductionPipelineError,
};
use fe2o3_compiler_closure_capability::inspect_compiler_execution_policy_family_v1;
use fe2o3_compiler_execution_protocol::CompilerExecutionPolicyFamilyV1 as Family;
use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
use std::{fs::File, mem::size_of};

/// Either family retains the uninterrupted original TARGET account until the
/// matching publication finishes or fails. No family can be retried or cast.
pub(crate) enum Admitted<'b, 'w> {
    Legacy {
        session: Legacy,
        budget: &'b mut Budget<'w>,
    },
    Native(native_v3::Admitted<'b, 'w>),
}

impl Startup {
    pub(crate) fn admit_selected<'b, 'w>(
        &self,
        budget: &'b mut Budget<'w>,
    ) -> Result<Admitted<'b, 'w>, native_v3::Error> {
        // Consume first: quota refusal cannot reopen the original protocol slots.
        let OwnedExecutionInputs { policy, service } =
            self.take().map_err(native_v3::Error::Startup)?;
        budget
            .reserve_storage(native_v3::Admitted::INPUT_STORAGE + size_of::<Admitted<'_, '_>>())?;
        let policy = File::from(policy);
        let family = inspect_compiler_execution_policy_family_v1(&policy, budget)?;
        let inputs = OwnedExecutionInputs {
            policy: policy.into(),
            service,
        };
        match family {
            Family::LegacyV1 => Ok(Admitted::Legacy {
                session: Legacy::from_owned(inputs).map_err(native_v3::Error::Startup)?,
                budget,
            }),
            Family::NativeV3 => {
                native_v3::Admitted::from_owned(inputs, budget).map(Admitted::Native)
            }
        }
    }
}

impl<'b, 'w> Admitted<'b, 'w> {
    pub(crate) fn with_reference_enrollment<T>(
        self,
        invocation: &mut crate::protected_rustc_invocation::AdmittedProtectedRustcInvocationV1,
        run: impl FnOnce(
            Option<&native_v3::ReferenceEnrollmentLoanV1<'_>>,
        ) -> Result<T, ProductionPipelineError>,
    ) -> Result<(Self, T), ProductionPipelineError> {
        match self {
            Self::Legacy { session, budget } => {
                if invocation
                    .reference_enrollment_requested(budget)
                    .map_err(native_v3::Error::Reference)?
                {
                    return Err(native_v3::Error::Reference(
                        crate::reference_effect_v1::ReferenceBindingErrorV1::new(
                            "explicit reference enrollment requires the native compiler session",
                        ),
                    )
                    .into());
                }
                let value = run(None)?;
                Ok((Self::Legacy { session, budget }, value))
            }
            Self::Native(session) => {
                let (session, value) = session.with_reference_enrollment(invocation, run)?;
                Ok((Self::Native(session), value))
            }
        }
    }

    pub(crate) fn publish<'tcx>(
        self,
        transaction: ProductionCompilation<'tcx, CollectedRustStage<'tcx>>,
    ) -> Result<u64, ProductionPipelineError> {
        match self {
            Self::Legacy { session, budget } => transaction
                .select_compiler_proof_family_v1(Family::LegacyV1)?
                .publish_worker_handoff(budget, session)
                .map(|subject| subject.outer_handoff().byte_len()),
            Self::Native(session) => transaction
                .select_compiler_proof_family_v1(Family::NativeV3)?
                .publish_native_worker_handoff(session)
                .map(|subject| subject.outer_handoff().byte_len()),
        }
    }
}

#[cfg(test)]
#[path = "protected_compiler_execution_dispatch_v236_tests.rs"]
mod tests;

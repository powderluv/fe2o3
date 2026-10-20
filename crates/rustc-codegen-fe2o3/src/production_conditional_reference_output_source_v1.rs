//! Authenticated backend adapters to the shared content-only CPU/source join.
//! Source rederivation and the one retained formula runtime stay backend-owned.

use super::*;
use crate::reference_effect_v1::AuthenticatedReferenceEffectBindingV1 as Binding;
use crate::reference_effect_v1::ReferenceBindingOriginV1 as Origin;
use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
use fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1 as Request;
use fe2o3_verifier::NativeCompilerStagingCommitmentV1 as Staging;
use fe2o3_verifier::conditional_reference_v1 as portable;
use fe2o3_verifier::portable_reference_v1::ReferenceReplayInputV1;
use fe2o3_verifier::portable_reference_v1::codec::{
    NativeCpuAssociationV1, NativeCpuInputV1, NativeCpuPolicyAssociationV2, NativeCpuPolicyInputV2,
};
#[cfg(test)]
use portable::{ConditionalReferenceInputV1, SourceBoundCpuCorrespondenceV1};

type Error = ProductionReferenceEffectJoinErrorV2;

/// Inert, move-only bytes captured from a genuine replay, never proof authority.
/// The owning projection phase keeps the reservation until this owner drops.
pub(crate) struct ConditionalReplayTransportV2 {
    cpu_input: Vec<u8>,
    staging: Vec<Staging>,
    formula_receipt: InertFunctionalRefinementReceiptSignatureV2,
}

impl ConditionalReplayTransportV2 {
    pub(crate) fn cpu_input_v1(&self) -> &[u8] {
        &self.cpu_input
    }

    pub(crate) fn formula_receipt_v2(&self) -> &InertFunctionalRefinementReceiptSignatureV2 {
        &self.formula_receipt
    }

    pub(crate) fn staging_commitments_v1(&self) -> &[Staging] {
        &self.staging
    }

    pub(crate) fn retained_storage_v2(&self) -> usize {
        // Construction bounds capacity by the existing CPU frame limit.
        std::mem::size_of::<Self>()
            + self.cpu_input.capacity()
            + self.staging.capacity() * std::mem::size_of::<Staging>()
    }

    pub(crate) fn require_same_v2(
        &self,
        previous: &Self,
        budget: &mut Budget<'_>,
    ) -> Result<(), Error> {
        use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1 as Resource;
        let work = self
            .cpu_input
            .len()
            .checked_add(std::mem::size_of::<
                InertFunctionalRefinementReceiptSignatureV2,
            >())
            .and_then(|n| {
                n.checked_add(
                    self.staging
                        .len()
                        .checked_mul(std::mem::size_of::<Staging>())?,
                )
            })
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| capture_error(Resource::Arithmetic))?;
        budget.charge_work(work).map_err(capture_error)?;
        if self.cpu_input != previous.cpu_input
            || self.staging != previous.staging
            || self.formula_receipt != previous.formula_receipt
        {
            return Err(Error::UnsupportedReference(
                "conditional replay transport changed",
            ));
        }
        Ok(())
    }
}

fn capture_error(error: impl fmt::Display) -> Error {
    Error::ProofExecution(error.to_string())
}

// This copier carries no receipt admission. Its only production caller supplies
// the live execution's signature after comparing the complete B1 commitment.
pub(crate) fn copy_inert_transport_v2(
    cpu_input: &[u8],
    staging: impl ExactSizeIterator<Item = Staging>,
    formula_receipt: InertFunctionalRefinementReceiptSignatureV2,
    budget: &mut Budget<'_>,
) -> Result<ConditionalReplayTransportV2, Error> {
    use crate::production_ranked_projection_v1::guarded_source_progress_v1::resources;
    use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1 as Resource;
    use fe2o3_verifier::portable_reference_v1::codec::MAX_NATIVE_CPU_INPUT_BYTES_V1;
    resources::owned(
        budget,
        0,
        capture_error,
        || Error::UnsupportedReference("conditional transport copy panicked"),
        |budget| {
            let count = staging.len();
            let payload = count
                .checked_mul(std::mem::size_of::<Staging>())
                .and_then(|n| n.checked_add(cpu_input.len()))
                .ok_or_else(|| capture_error(Resource::Arithmetic))?;
            if payload > MAX_NATIVE_CPU_INPUT_BYTES_V1 {
                return Err(Error::UnsupportedReference("conditional CPU frame limit"));
            }
            let retained = payload
                .checked_add(std::mem::size_of::<ConditionalReplayTransportV2>())
                .ok_or_else(|| capture_error(Resource::Arithmetic))?;
            budget
                .charge_work(
                    retained
                        .checked_add(count)
                        .ok_or_else(|| capture_error(Resource::Arithmetic))?,
                )
                .map_err(capture_error)?;
            budget.reserve_storage(retained).map_err(capture_error)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(cpu_input.len())
                .map_err(|_| capture_error(Resource::Allocation))?;
            if bytes.capacity() != cpu_input.len() {
                return Err(capture_error(Resource::Allocation));
            }
            bytes.extend_from_slice(cpu_input);
            let mut rows = Vec::new();
            rows.try_reserve_exact(count)
                .map_err(|_| capture_error(Resource::Allocation))?;
            if rows.capacity() != count {
                return Err(capture_error(Resource::Allocation));
            }
            for row in staging {
                if rows.len() == count {
                    return Err(capture_error(Resource::Accounting));
                }
                rows.push(row);
            }
            if rows.len() != count {
                return Err(capture_error(Resource::Accounting));
            }
            Ok((
                ConditionalReplayTransportV2 {
                    cpu_input: bytes,
                    staging: rows,
                    formula_receipt,
                },
                retained,
            ))
        },
    )
}

fn staging_commitment_v2(
    row: &fe2o3_pliron::ProductionPolicyCheckedRefinementStagingV2,
) -> Staging {
    let toolchain = row.toolchain();
    Staging {
        receipt: *row.receipt_identity().digest().as_bytes(),
        effect: *row
            .binding()
            .normalized_obligation_effect_ir_hash()
            .as_bytes(),
        signer: *row.signer_identity().as_bytes(),
        execution: *row.execution_identity().as_bytes(),
        toolchain: [
            *toolchain.verus_executable().as_bytes(),
            *toolchain.verus_configuration().as_bytes(),
            *toolchain.solver_executable().as_bytes(),
            *toolchain.solver_configuration().as_bytes(),
            *toolchain.runtime_closure().as_bytes(),
        ],
    }
}

pub(crate) fn capture_replayed_cpu_formula_v2(
    request: &Request<'_>,
    binding: &Binding,
    semantic_root: u32,
    execution: &fe2o3_verifier::ProductionConditionalFormulaExecutionV2,
    budget: &mut Budget<'_>,
) -> Result<ConditionalReplayTransportV2, Error> {
    let capture = |bytes: &[u8], commitment: [u8; 32], budget: &mut Budget<'_>| {
        budget.charge_work(32).map_err(capture_error)?;
        if execution.report().cpu_input_commitment().as_bytes() != &commitment {
            return Err(Error::UnsupportedReference(
                "conditional CPU capture commitment",
            ));
        }
        let wire = execution
            .signed_receipt_wire()
            .try_into()
            .map_err(|_| Error::UnsupportedReference("conditional formula signature extent"))?;
        copy_inert_transport_v2(
            bytes,
            request
                .pliron_input()
                .retained_policy_checked_refinement_staging()
                .iter()
                .map(staging_commitment_v2),
            InertFunctionalRefinementReceiptSignatureV2::from_untrusted_parts(
                wire,
                *execution.receipt_verifying_key(),
            ),
            budget,
        )
    };
    match &binding.origin {
        Origin::SourceRegistration(_) => {
            fe2o3_verifier::portable_reference_v1::codec::with_encoded_native_cpu_input_v1(
                native_cpu_input_v1(request, binding, semantic_root)?,
                budget,
                capture,
            )
        }
        Origin::ReferenceEnrollment(_) => {
            fe2o3_verifier::portable_reference_v1::codec::with_encoded_native_cpu_policy_input_v2(
                native_cpu_policy_input_v2(request, binding, semantic_root)?,
                budget,
                capture,
            )
        }
    }
    .map_err(capture_error)?
}

#[cfg(test)]
#[path = "production_conditional_cpu_read_premises_v1.rs"]
pub(crate) mod read_premises_v1;

#[cfg(test)]
fn borrowed_input(binding: &Binding) -> ConditionalReferenceInputV1<'_> {
    ConditionalReferenceInputV1 {
        kernel: &binding.kernel,
        reference: &binding.reference,
        replay: ReferenceReplayInputV1 {
            signature_preimage: &binding.signature_preimage,
            effect_ir: &binding.effect_ir,
            effect_ir_sha256: binding.effect_ir_sha256,
            observable_output_writes: &binding.observable_output_writes,
        },
    }
}

// Borrow the authenticated source row without projecting either complete
// identity. V2 checks the exact root against the original semantic root roster.
fn native_cpu_input_v1<'a>(
    request: &Request<'_>,
    binding: &'a Binding,
    semantic_root: u32,
) -> Result<NativeCpuInputV1<'a>, Error> {
    let registration_path =
        binding
            .origin
            .source_registration_v1()
            .ok_or(Error::UnsupportedReference(
                "policy enrollment requires the V2 CPU execution path",
            ))?;
    Ok(NativeCpuInputV1 {
        association: NativeCpuAssociationV1 {
            semantic_mir_sha256: *request
                .source()
                .semantic_ssa()
                .source_semantic()
                .semantic_sha256()
                .as_bytes(),
            semantic_root,
            registration_path,
            logical_kernel_name: &binding.logical_kernel_name,
        },
        kernel: &binding.kernel,
        reference: &binding.reference,
        replay: ReferenceReplayInputV1 {
            signature_preimage: &binding.signature_preimage,
            effect_ir: &binding.effect_ir,
            effect_ir_sha256: binding.effect_ir_sha256,
            observable_output_writes: &binding.observable_output_writes,
        },
    })
}

pub(crate) fn retain_source_bound_cpu_formula_v2(
    runtime: &FunctionalRefinementVerusRuntimeLeaseV1,
    request: &Request<'_>,
    references: &AuthenticatedReferenceEffectBindingsV1,
    semantic_root: u32,
    budget: &mut Budget<'_>,
    timeout_seconds: u32,
) -> Result<fe2o3_verifier::RetainedProductionConditionalFormulaV2, Error> {
    let [binding] = references.as_slice() else {
        return Err(Error::UnsupportedReference(
            "conditional CPU join requires one authenticated binding",
        ));
    };
    match &binding.origin {
        Origin::SourceRegistration(_) => {
            fe2o3_verifier::execute_and_retain_conditional_ranked_formula_v2(
                runtime,
                request,
                native_cpu_input_v1(request, binding, semantic_root)?,
                budget,
                timeout_seconds,
            )
        }
        Origin::ReferenceEnrollment(_) => {
            fe2o3_verifier::execute_and_retain_conditional_ranked_formula_policy_v2(
                runtime,
                request,
                native_cpu_policy_input_v2(request, binding, semantic_root)?,
                budget,
                timeout_seconds,
            )
        }
    }
    .map_err(|error| Error::ProofExecution(error.to_string()))
}

pub(crate) fn replay_source_bound_cpu_formula_v2<R>(
    retained: &fe2o3_verifier::RetainedProductionConditionalFormulaV2,
    request: &Request<'_>,
    binding: &Binding,
    semantic_root: u32,
    budget: &mut Budget<'_>,
    consume: impl for<'proof> FnOnce(
        &'proof fe2o3_verifier::ProductionConditionalFormulaExecutionV2,
        &mut Budget<'_>,
    ) -> R,
) -> Result<R, Error> {
    #[cfg(test)]
    match &binding.origin {
        Origin::SourceRegistration(_) => {
            composition_tests::on_replay(request, binding, semantic_root, budget);
            formula_v2_tests::on_replay(retained, request, binding, semantic_root, budget);
        }
        Origin::ReferenceEnrollment(_) => {
            policy_v2_tests::on_replay(retained, request, binding, semantic_root, budget);
        }
    }
    match &binding.origin {
        Origin::SourceRegistration(_) => retained.with_replayed_request_v2(
            request,
            native_cpu_input_v1(request, binding, semantic_root)?,
            budget,
            |execution, budget| Ok(consume(execution, budget)),
        ),
        Origin::ReferenceEnrollment(_) => retained.with_replayed_policy_request_v2(
            request,
            native_cpu_policy_input_v2(request, binding, semantic_root)?,
            budget,
            |execution, budget| Ok(consume(execution, budget)),
        ),
    }
    .map_err(|error| Error::ProofExecution(error.to_string()))
}

fn native_cpu_policy_input_v2<'a>(
    request: &Request<'_>,
    binding: &'a Binding,
    semantic_root: u32,
) -> Result<NativeCpuPolicyInputV2<'a>, Error> {
    let Origin::ReferenceEnrollment(origin) = &binding.origin else {
        return Err(Error::UnsupportedReference(
            "source registration requires the V1 CPU execution path",
        ));
    };
    Ok(NativeCpuPolicyInputV2 {
        association: NativeCpuPolicyAssociationV2 {
            semantic_mir_sha256: *request
                .source()
                .semantic_ssa()
                .source_semantic()
                .semantic_sha256()
                .as_bytes(),
            semantic_root,
            logical_kernel_name: &binding.logical_kernel_name,
            origin: *origin,
        },
        kernel: &binding.kernel,
        reference: &binding.reference,
        replay: ReferenceReplayInputV1 {
            signature_preimage: &binding.signature_preimage,
            effect_ir: &binding.effect_ir,
            effect_ir_sha256: binding.effect_ir_sha256,
            observable_output_writes: &binding.observable_output_writes,
        },
    })
}

#[cfg(test)]
fn with_cpu_binding<R>(
    request: &Request<'_>,
    binding: &Binding,
    semantic_root: u32,
    budget: &mut Budget<'_>,
    consume: impl for<'cpu> FnOnce(
        &'cpu SourceBoundCpuCorrespondenceV1<'_>,
        &mut Budget<'_>,
    ) -> Result<R, Error>,
) -> Result<R, Error> {
    portable::with_source_bound_cpu_correspondence_v1(
        request,
        borrowed_input(binding),
        semantic_root,
        budget,
        consume,
    )
    .map_err(Error::from)?
}

pub(crate) fn subjects(binding: &Binding) -> Result<FunctionalRefinementSubjectsV2, Error> {
    portable::reference_subjects_v1(&binding.kernel, &binding.reference).map_err(Error::from)
}

#[cfg(test)]
#[path = "production_conditional_reference_genuine_composition_v1_tests.rs"]
pub(crate) mod composition_tests;

#[cfg(test)]
#[path = "production_conditional_formula_genuine_v2_tests.rs"]
pub(crate) mod formula_v2_tests;

#[cfg(test)]
#[path = "production_conditional_reference_policy_v2_tests.rs"]
pub(crate) mod policy_v2_tests;

#[cfg(test)]
#[path = "production_conditional_reference_transport_v2_tests.rs"]
mod transport_tests;

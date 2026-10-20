//! Same-session reference inputs retained by the authenticated collector closure.
//! These are not independent native-source replay or proof authority.

use super::{CollectedFunction, CollectedFunctionRole};
use crate::reference_effect_v1::{
    AuthenticatedReferenceEffectBindingV1, AuthenticatedReferenceEffectBindingsV1,
    ReferenceBindingErrorV1, ReferenceBindingOriginV1,
    authenticate_reference_binding_with_origin_v1, equivalent_bindings_v1, instantiated_signature,
};
use crate::rustc_semantic_plan_v1::SourceClosureWorkV1;
use rustc_middle::ty::{FnSig, Instance, TyCtxt};

#[path = "original_root_associations_v1.rs"]
mod original_roots;
pub(super) use original_roots::PendingOriginalRootAssociationsV1;
pub(crate) use original_roots::RetainedOriginalRootAssociationsV1;

struct ReferenceInputV1<'tcx> {
    function: usize,
    kernel: Instance<'tcx>,
    reference: Instance<'tcx>,
    kernel_signature: FnSig<'tcx>,
    reference_signature: FnSig<'tcx>,
    origin: ReferenceBindingOriginV1,
    logical_kernel_name: String,
}

pub(super) struct RetainedReferenceInputsV1<'tcx> {
    session: &'tcx rustc_session::Session,
    function_count: usize,
    inputs: Box<[ReferenceInputV1<'tcx>]>,
    enrollment: Option<super::reference_enrollment_v1::RetainedEnrollmentV1<'tcx>>,
}

fn error(reason: &'static str) -> ReferenceBindingErrorV1 {
    ReferenceBindingErrorV1::new(reason)
}

fn charge(work: &mut SourceClosureWorkV1, amount: usize) -> Result<(), ReferenceBindingErrorV1> {
    work.charge(amount)
        .map_err(|failure| ReferenceBindingErrorV1::new(failure.to_string()))
}

fn charge_storage<T>(
    work: &mut SourceClosureWorkV1,
    count: usize,
) -> Result<(), ReferenceBindingErrorV1> {
    charge(
        work,
        count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| error("reference input storage overflow"))?,
    )
}

fn charge_names(
    work: &mut SourceClosureWorkV1,
    origin: &ReferenceBindingOriginV1,
    name: &str,
) -> Result<(), ReferenceBindingErrorV1> {
    charge(
        work,
        origin
            .retained_payload_bytes_v1()
            .checked_add(name.len())
            .ok_or_else(|| error("reference name work overflow"))?,
    )
}

fn retained_bytes<'tcx>(
    functions: &[CollectedFunction<'tcx>],
    work: &mut SourceClosureWorkV1,
) -> Result<(usize, usize), ReferenceBindingErrorV1> {
    charge(work, functions.len())?;
    let mut count = 0_usize;
    let mut bytes = 0_usize;
    for function in functions {
        let binding = match (
            &function.reference_effect_binding,
            function.reference_instance,
        ) {
            (None, None) => continue,
            (Some(binding), Some(_)) if function.role == CollectedFunctionRole::KernelEntry => {
                binding
            }
            _ => {
                return Err(error(
                    "reference binding/instance presence or kernel role changed",
                ));
            }
        };
        count = count
            .checked_add(1)
            .ok_or_else(|| error("reference input count overflow"))?;
        bytes = bytes
            .checked_add(std::mem::size_of::<ReferenceInputV1<'tcx>>())
            .and_then(|bytes| bytes.checked_add(binding.origin.retained_payload_bytes_v1()))
            .and_then(|bytes| bytes.checked_add(binding.logical_kernel_name.len()))
            .ok_or_else(|| error("reference input storage overflow"))?;
    }
    Ok((count, bytes))
}

impl<'tcx> RetainedReferenceInputsV1<'tcx> {
    pub(super) fn capture(
        tcx: TyCtxt<'tcx>,
        functions: &[CollectedFunction<'tcx>],
        work: &mut SourceClosureWorkV1,
    ) -> Result<Self, ReferenceBindingErrorV1> {
        Self::capture_with_enrollment(tcx, functions, work, None, None)
    }

    pub(super) fn capture_with_enrollment(
        tcx: TyCtxt<'tcx>,
        functions: &[CollectedFunction<'tcx>],
        work: &mut SourceClosureWorkV1,
        enrollment: Option<super::reference_enrollment_v1::RetainedEnrollmentV1<'tcx>>,
        loan: Option<
            &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
        >,
    ) -> Result<Self, ReferenceBindingErrorV1> {
        match (&enrollment, loan) {
            (Some(retained), Some(loan)) => {
                retained.revalidate(tcx, loan, work)?;
                retained.validate_inputs(functions, work)?;
            }
            (None, None) => {}
            _ => {
                return Err(error(
                    "reference enrollment capture has no original live owner",
                ));
            }
        }
        let (count, bytes) = retained_bytes(functions, work)?;
        // SourceClosureWork is cumulative work, not the later canonical byte
        // ledger. Charging retained bytes here also bounds this new allocation.
        charge(work, bytes)?;
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(count)
            .map_err(|_| error("reference input allocation failed"))?;
        for (function, collected) in functions.iter().enumerate() {
            charge(work, 1)?;
            let Some(reference) = collected.reference_instance else {
                continue;
            };
            let binding = collected
                .reference_effect_binding
                .as_ref()
                .ok_or_else(|| error("retained reference has no binding"))?;
            if matches!(
                binding.origin,
                ReferenceBindingOriginV1::ReferenceEnrollment(_)
            ) && enrollment.is_none()
            {
                return Err(error("reference enrollment has no original live owner"));
            }
            charge_names(work, &binding.origin, &binding.logical_kernel_name)?;
            inputs.push(ReferenceInputV1 {
                function,
                kernel: collected.instance,
                reference,
                kernel_signature: instantiated_signature(tcx, collected.instance),
                reference_signature: instantiated_signature(tcx, reference),
                origin: binding.origin.clone(),
                logical_kernel_name: binding.logical_kernel_name.clone(),
            });
        }
        Ok(Self {
            session: tcx.sess,
            function_count: functions.len(),
            inputs: inputs.into_boxed_slice(),
            enrollment,
        })
    }

    pub(super) fn rederive(
        &self,
        tcx: TyCtxt<'tcx>,
        functions: &[CollectedFunction<'tcx>],
        work: &mut SourceClosureWorkV1,
    ) -> Result<AuthenticatedReferenceEffectBindingsV1, ReferenceBindingErrorV1> {
        self.rederive_with_enrollment(tcx, functions, work, None)
    }

    pub(super) fn rederive_with_enrollment(
        &self,
        tcx: TyCtxt<'tcx>,
        functions: &[CollectedFunction<'tcx>],
        work: &mut SourceClosureWorkV1,
        loan: Option<
            &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
        >,
    ) -> Result<AuthenticatedReferenceEffectBindingsV1, ReferenceBindingErrorV1> {
        match (&self.enrollment, loan) {
            (Some(retained), Some(loan)) => {
                retained.revalidate(tcx, loan, work)?;
                // A fresh scoped loan must establish descriptor membership
                // before origin access, even when the retained roster is empty.
                let _ = loan.request(work)?;
                retained.validate_inputs(functions, work)?;
            }
            (None, None) => {}
            _ => return Err(error("reference enrollment live owner presence changed")),
        }
        charge(work, 1)?;
        if !std::ptr::eq(self.session, tcx.sess) {
            return Err(error("reference input compiler session changed"));
        }
        if self.function_count != functions.len() {
            return Err(error("reference input function roster changed"));
        }
        charge(work, functions.len())?;
        let mut retained = self.inputs.iter().peekable();
        let mut bindings = Vec::new();
        charge_storage::<AuthenticatedReferenceEffectBindingV1>(work, self.inputs.len())?;
        bindings
            .try_reserve_exact(self.inputs.len())
            .map_err(|_| error("reference replay allocation failed"))?;
        for (index, function) in functions.iter().enumerate() {
            let expected = retained
                .peek()
                .filter(|input| input.function == index)
                .copied();
            let (Some(expected), Some(reference), Some(binding)) = (
                expected,
                function.reference_instance,
                function.reference_effect_binding.as_ref(),
            ) else {
                if expected.is_some()
                    || function.reference_instance.is_some()
                    || function.reference_effect_binding.is_some()
                {
                    return Err(error("reference binding or retained input roster changed"));
                }
                continue;
            };
            retained.next();
            // String equality and the later name clones each receive a debit.
            charge_names(work, &expected.origin, &expected.logical_kernel_name)?;
            charge(work, expected.logical_kernel_name.len())?;
            if function.role != CollectedFunctionRole::KernelEntry
                || function.instance != expected.kernel
                || reference != expected.reference
                || function.logical_name.as_deref() != Some(expected.logical_kernel_name.as_str())
                || binding.origin != expected.origin
                || binding.logical_kernel_name != expected.logical_kernel_name
            {
                return Err(error(
                    "reference input instance, role or registration changed",
                ));
            }
            charge(work, expected.kernel_signature.inputs().len())?;
            charge(work, expected.reference_signature.inputs().len())?;
            if instantiated_signature(tcx, function.instance) != expected.kernel_signature
                || instantiated_signature(tcx, reference) != expected.reference_signature
            {
                return Err(error("reference input instantiated signature changed"));
            }
            // Charge both allocation and copying before cloning owned names.
            charge_names(work, &expected.origin, &expected.logical_kernel_name)?;
            charge_names(work, &expected.origin, &expected.logical_kernel_name)?;
            if let ReferenceBindingOriginV1::ReferenceEnrollment(origin) = &expected.origin {
                charge(work, std::mem::size_of_val(origin))?;
                let loan =
                    loan.ok_or_else(|| error("reference enrollment replay has no live owner"))?;
                if loan.origin(origin.mapping_ordinal)? != *origin {
                    return Err(error("reference enrollment descriptive origin changed"));
                }
            }
            let fresh = authenticate_reference_binding_with_origin_v1(
                tcx,
                expected.origin.clone(),
                expected.logical_kernel_name.clone(),
                expected.kernel,
                expected.reference,
                work,
            )?;
            if !equivalent_bindings_v1(binding, &fresh, work)? {
                return Err(error(
                    "reference binding differs from fresh compiler extraction",
                ));
            }
            bindings.push(fresh);
        }
        if retained.next().is_some() {
            return Err(error("reference input occurrence is missing"));
        }
        Ok(AuthenticatedReferenceEffectBindingsV1::new(bindings))
    }
}

#[cfg(test)]
#[path = "reference_custody_v1_tests.rs"]
mod tests;

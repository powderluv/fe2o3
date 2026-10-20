//! Caller-selected CPU origin expectations, never enrollment authority.
use super::*;
use crate::portable_reference_v1::codec::{
    DecodedNativeCpuInputV1, DecodedNativeCpuPolicyInputV2, ReferenceEnrollmentOriginV1,
    with_decoded_native_cpu_input_v1, with_decoded_native_cpu_policy_input_v2,
};

/// Explicit codec selection supplied independently of the transported CPU leaf.
/// Matching this value does not authenticate original enrollment or source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeConditionalCpuOriginExpectationV1 {
    /// Select only the existing registration codec; not registration provenance.
    SourceRegistrationV1,
    /// Match all fields against independently retained original policy selection.
    ReferenceEnrollmentV1(ReferenceEnrollmentOriginV1),
}

/// One inert expectation in the exact source-root and accepted-policy order.
/// The caller retains and prepays this backing on its original account.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConditionalCpuExpectationV1 {
    pub semantic_root: u32,
    pub origin: NativeConditionalCpuOriginExpectationV1,
}

#[derive(Clone, Copy)]
pub(super) enum CpuReplayMode<'a> {
    RegistrationOnly,
    Expected(&'a [NativeConditionalCpuExpectationV1]),
}

impl CpuReplayMode<'_> {
    pub(super) fn backing_bytes(self) -> usize {
        match self {
            Self::RegistrationOnly => 0,
            Self::Expected(rows) => std::mem::size_of_val(rows),
        }
    }

    pub(super) fn require_backing(self, budget: &mut Budget<'_>) -> Result<(), E> {
        if let Self::Expected(rows) = self {
            budget.charge_work(1)?;
            if budget.storage() < std::mem::size_of_val(rows) {
                return Err(Resource::Accounting.into());
            }
        }
        Ok(())
    }

    pub(super) fn require_roster(
        self,
        roots: &[NativeConditionalSourceRootV2<'_>],
        accepted: &[NativeConditionalRootPolicyV2<'_>],
        budget: &mut Budget<'_>,
    ) -> Result<(), E> {
        let Self::Expected(expected) = self else {
            return Ok(());
        };
        budget.charge_work(3)?;
        if roots.is_empty() || expected.len() != roots.len() || accepted.len() != roots.len() {
            return Err(E::invalid(
                "complete independent CPU origin expectation roster",
            ));
        }
        for (index, ((row, policy), expectation)) in
            roots.iter().zip(accepted).zip(expected).enumerate()
        {
            budget.charge_work(3)?;
            if row.semantic_root != expectation.semantic_root
                || policy.semantic_root != expectation.semantic_root
            {
                return Err(E::invalid(
                    "ordered independent CPU origin expectation roots",
                ));
            }
            for previous in &expected[..index] {
                budget.charge_work(1)?;
                if previous.semantic_root == expectation.semantic_root {
                    return Err(E::invalid(
                        "duplicate independent CPU origin expectation root",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn at(
        self,
        ordinal: usize,
        semantic_root: u32,
        budget: &mut Budget<'_>,
    ) -> Result<NativeConditionalCpuOriginExpectationV1, E> {
        match self {
            Self::RegistrationOnly => {
                Ok(NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1)
            }
            Self::Expected(rows) => {
                budget.charge_work(2)?;
                let row = rows
                    .get(ordinal)
                    .ok_or_else(|| E::invalid("missing independent CPU origin expectation"))?;
                if row.semantic_root != semantic_root {
                    return Err(E::invalid(
                        "changed independent CPU origin expectation order",
                    ));
                }
                Ok(row.origin)
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum DecodedCpu<'a> {
    Registration(&'a DecodedNativeCpuInputV1),
    Policy(&'a DecodedNativeCpuPolicyInputV2),
}

pub(super) struct CpuSubjects<'a> {
    pub semantic_mir_sha256: [u8; 32],
    pub semantic_root: u32,
    pub logical_kernel_name: &'a str,
    pub kernel: &'a crate::portable_reference_v1::ReferenceFunctionIdentityV1,
    pub reference: &'a crate::portable_reference_v1::ReferenceFunctionIdentityV1,
}

impl<'a> DecodedCpu<'a> {
    pub(super) fn subjects(self) -> CpuSubjects<'a> {
        match self {
            Self::Registration(owner) => {
                let input = owner.input_v1();
                CpuSubjects {
                    semantic_mir_sha256: input.association.semantic_mir_sha256,
                    semantic_root: input.association.semantic_root,
                    logical_kernel_name: input.association.logical_kernel_name,
                    kernel: input.kernel,
                    reference: input.reference,
                }
            }
            Self::Policy(owner) => {
                let input = owner.input_v2();
                CpuSubjects {
                    semantic_mir_sha256: input.association.semantic_mir_sha256,
                    semantic_root: input.association.semantic_root,
                    logical_kernel_name: input.association.logical_kernel_name,
                    kernel: input.kernel,
                    reference: input.reference,
                }
            }
        }
    }
}

fn require_origin(
    actual: ReferenceEnrollmentOriginV1,
    expected: ReferenceEnrollmentOriginV1,
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    // Two digests, generation and ordinal, plus the fixed comparison decisions.
    budget.charge_work(32 + 32 + 8 + 4 + 4)?;
    if actual != expected {
        return Err(E::invalid("independent CPU enrollment-origin expectation"));
    }
    Ok(())
}

pub(super) fn with_decoded<R, Failure: From<E> + From<Resource>>(
    bytes: &[u8],
    expected: NativeConditionalCpuOriginExpectationV1,
    budget: &mut Budget<'_>,
    consume: impl for<'cpu> FnOnce(DecodedCpu<'cpu>, &mut Budget<'_>) -> Result<R, Failure>,
) -> Result<R, Failure> {
    match expected {
        NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1 => {
            with_decoded_native_cpu_input_v1(bytes, budget, |decoded, budget| {
                consume(DecodedCpu::Registration(decoded), budget)
            })
            .map_err(|error| E(Cause::Cpu(error)))?
        }
        NativeConditionalCpuOriginExpectationV1::ReferenceEnrollmentV1(expected) => {
            with_decoded_native_cpu_policy_input_v2(bytes, budget, |decoded, budget| {
                require_origin(decoded.input_v2().association.origin, expected, budget)?;
                consume(DecodedCpu::Policy(decoded), budget)
            })
            .map_err(|error| E(Cause::Cpu(error)))?
        }
    }
}

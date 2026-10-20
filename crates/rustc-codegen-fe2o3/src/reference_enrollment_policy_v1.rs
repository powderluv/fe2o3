//! Compiler work/error adapter for the shared inert enrollment decoder.
//! Decoding selects no Instance and grants no source, proof or native authority.

use crate::reference_effect_v1::ReferenceBindingErrorV1 as Error;
use crate::rustc_semantic_plan_v1::SourceClosureWorkV1;
pub(crate) use fe2o3_rustc_invocation::{
    REFERENCE_ENROLLMENT_ENV_V1 as ENV, ReferenceEnrollmentBindingV1,
};
use fe2o3_rustc_invocation::{
    ReferenceEnrollmentDecodeErrorV1, ReferenceEnrollmentRequestV1 as Request,
    RustcInvocationDescriptorV3,
};

#[cfg(test)]
use fe2o3_rustc_invocation::{
    CompileEnvironmentV2, MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 as MAX_BINDINGS,
    MAX_REFERENCE_ENROLLMENT_BYTES_V1 as MAX_BYTES,
    MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1 as MAX_SELECTOR_BYTES,
};

#[derive(Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct ReferenceEnrollmentRequestV1(Request);

impl ReferenceEnrollmentRequestV1 {
    /// Inert projection; admission and retained owner checks belong to the loan.
    pub(crate) fn project_binding_count(
        descriptor: &RustcInvocationDescriptorV3,
        work: &mut SourceClosureWorkV1,
    ) -> Result<Option<usize>, Error> {
        Request::project_binding_count_from_descriptor(descriptor, |amount| charge(work, amount))
            .map_err(adapt_error)
    }

    /// The descriptor is inert. Its original admitted owner must independently
    /// survive collection and replay; equal descriptor bytes cannot replace it.
    pub(crate) fn from_descriptor(
        descriptor: &RustcInvocationDescriptorV3,
        work: &mut SourceClosureWorkV1,
    ) -> Result<Option<Self>, Error> {
        Request::from_descriptor(descriptor, |amount| charge(work, amount))
            .map(|request| request.map(Self))
            .map_err(adapt_error)
    }

    pub(crate) fn bindings(&self) -> &[ReferenceEnrollmentBindingV1] {
        self.0.bindings()
    }

    #[cfg(test)]
    fn from_environment(
        environment: &CompileEnvironmentV2,
        work: &mut SourceClosureWorkV1,
    ) -> Result<Option<Self>, Error> {
        Request::from_environment(environment, |amount| charge(work, amount))
            .map(|request| request.map(Self))
            .map_err(adapt_error)
    }

    #[cfg(test)]
    fn decode(bytes: &str, work: &mut SourceClosureWorkV1) -> Result<Self, Error> {
        Request::decode(bytes, |amount| charge(work, amount))
            .map(Self)
            .map_err(adapt_error)
    }
}

fn adapt_error(error: ReferenceEnrollmentDecodeErrorV1<Error>) -> Error {
    match error {
        ReferenceEnrollmentDecodeErrorV1::Work(error) => error,
        other => Error::new(other.to_string()),
    }
}

fn charge(work: &mut SourceClosureWorkV1, amount: usize) -> Result<(), Error> {
    work.charge(amount)
        .map_err(|error| Error::new(error.to_string()))
}

#[cfg(test)]
#[path = "reference_enrollment_policy_v1_tests.rs"]
mod tests;

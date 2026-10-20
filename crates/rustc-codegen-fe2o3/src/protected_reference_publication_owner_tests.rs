//! Read-only assertions on actual publication custody; never a validator.

use super::{FinishedProtectedRustcInvocationV3, ReferenceEnrollmentInvocationStampV1};

impl FinishedProtectedRustcInvocationV3 {
    pub(crate) fn assert_reference_owner_for_test(
        &self,
        original: &ReferenceEnrollmentInvocationStampV1,
        foreign: &ReferenceEnrollmentInvocationStampV1,
    ) {
        let owner = self
            .reference_enrollment
            .as_ref()
            .expect("publication retained the original invocation owner");
        assert!(owner.matches(original));
        assert!(!owner.matches(foreign));
    }
}

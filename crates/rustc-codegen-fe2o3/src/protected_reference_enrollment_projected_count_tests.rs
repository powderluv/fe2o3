//! Original compiler custody only. These do not qualify a live root attempt,
//! parser storage or installed compiler provenance; no receipt service is used.
use super::*;

#[test]
fn original_projection_survives_fresh_loan_without_publishing_a_header_early() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::original_projection_survives_fresh_loan_without_publishing_a_header_early"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, projection) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            let projection = loan.project_request(&mut source)?;
            assert_eq!(projection.bindings, Some(1));
            assert!(loan.requested_bindings.get().is_none());
            assert!(loan.inventory_context(&mut source)?.is_none());
            Ok(projection)
        })
        .unwrap();
    let expected = projection.origin;
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            assert!(loan.inventory_context(&mut source)?.is_none());
            let request = loan
                .decode_projected_request(projection, &mut source)?
                .unwrap();
            assert_eq!(request.bindings().len(), 1);
            let context = loan.inventory_context(&mut source)?.unwrap();
            assert_eq!(context.descriptor_bindings, 1);
            assert_eq!(
                context.rustc_invocation_sha256,
                expected.rustc_invocation_sha256
            );
            assert_eq!(context.native_policy_sha256, expected.native_policy_sha256);
            assert_eq!(context.policy_generation, expected.policy_generation);
            Ok(())
        })
        .unwrap();
    drop(native);
}

#[test]
fn equal_descriptor_cannot_replace_the_projected_invocation_owner() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::equal_descriptor_cannot_replace_the_projected_invocation_owner"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut other = fixture::equal_invocation(&invocation);
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, projection) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            Ok(loan.unwrap().project_request(&mut source)?)
        })
        .unwrap();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut other, |loan| {
        assert_projection_refused(
            loan.unwrap(),
            projection,
            &mut source,
            "reference enrollment original invocation changed",
        );
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    invocation.revalidate_for_publication().unwrap();
}

#[test]
fn equal_policy_cannot_replace_the_projected_native_session() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::equal_policy_cannot_replace_the_projected_native_session"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let (native, projection) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            Ok(loan.unwrap().project_request(&mut source)?)
        })
        .unwrap();
    let mut other_work = TargetWork::new(usize::MAX);
    let mut other_budget = Budget::new(&mut other_work, STORAGE);
    let other_policy = fixture::policy(&mut other_budget);
    assert_eq!(policy.policy(), other_policy.policy());
    let (other, _other_server) = fixture::session(&other_policy, &mut other_budget);
    let result = other.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        assert_projection_refused(
            loan.unwrap(),
            projection,
            &mut source,
            "reference enrollment original native session changed",
        );
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    let (native, ()) = native
        .with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            assert_eq!(
                loan.unwrap()
                    .request(&mut source)?
                    .unwrap()
                    .bindings()
                    .len(),
                1
            );
            Ok(())
        })
        .unwrap();
    drop(native);
}

#[test]
fn projected_descriptor_policy_and_count_mismatches_stay_terminal() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::projected_descriptor_policy_and_count_mismatches_stay_terminal"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (invocation, _backend) = fixture::admit();
    // Private fault injection checks each coordinate independently of owner
    // identity. Production has no constructor or mutator for this projection.
    for case in 0..6 {
        let mut invocation = fixture::equal_invocation(&invocation);
        let mut work = TargetWork::new(usize::MAX);
        let mut budget = Budget::new(&mut work, STORAGE);
        let policy = fixture::policy(&mut budget);
        let (native, _server) = fixture::session(&policy, &mut budget);
        let mut source = Work::default();
        let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
            let loan = loan.unwrap();
            let mut projection = loan.project_request(&mut source)?;
            match case {
                0 => projection.origin.rustc_invocation_sha256[0] ^= 1,
                1 => projection.origin.native_policy_sha256[0] ^= 1,
                2 => projection.origin.policy_generation += 1,
                3 => projection.origin.mapping_ordinal += 1,
                4 => projection.bindings = None,
                5 => projection.bindings = Some(2),
                _ => unreachable!(),
            }
            let reason = if case < 4 {
                "reference enrollment projected descriptor or policy changed"
            } else {
                "reference enrollment projected count differs from owned request"
            };
            assert_projection_refused(loan, projection, &mut source, reason);
            Ok(())
        });
        refused(result, "reference enrollment loan was refused");
    }
}

#[test]
fn live_change_after_projection_refuses_before_owned_decode() {
    if fixture::isolated(
        concat!(
            module_path!(),
            "::live_change_after_projection_refuses_before_owned_decode"
        ),
        Some(ONE_REQUEST),
    ) {
        return;
    }
    let (mut invocation, _backend) = fixture::admit();
    let mut work = TargetWork::new(usize::MAX);
    let mut budget = Budget::new(&mut work, STORAGE);
    let policy = fixture::policy(&mut budget);
    let (native, _server) = fixture::session(&policy, &mut budget);
    let mut source = Work::default();
    let result = native.with_reference_enrollment::<_, SessionError>(&mut invocation, |loan| {
        let loan = loan.unwrap();
        let projection = loan.project_request(&mut source)?;
        let changed = fixture::ChangedDirectory::enter();
        let error = loan
            .decode_projected_request(projection, &mut source)
            .unwrap_err();
        assert!(error.to_string().contains("directory"), "{error}");
        drop(changed);
        assert!(loan.requested_bindings.get().is_none());
        assert!(loan.request(&mut source).is_err());
        assert!(loan.inventory_context(&mut source).is_err());
        Ok(())
    });
    refused(result, "reference enrollment loan was refused");
    invocation.revalidate_for_publication().unwrap();
}

fn assert_projection_refused(
    loan: &ReferenceEnrollmentLoanV1<'_>,
    projection: ProjectedEnrollmentRequestV1,
    source: &mut Work,
    reason: &str,
) {
    assert_eq!(
        loan.decode_projected_request(projection, source)
            .unwrap_err()
            .to_string(),
        reason
    );
    assert!(loan.requested_bindings.get().is_none());
    assert!(loan.request(source).is_err());
    assert!(loan.inventory_context(source).is_err());
    assert!(loan.reserve_inventory_storage(0).is_err());
}

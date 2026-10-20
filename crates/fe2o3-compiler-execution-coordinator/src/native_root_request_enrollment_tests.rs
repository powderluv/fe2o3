//! Inert transcript and refusal tests, not a fabricated protected issuer.
use super::*;
use ed25519_dalek::SigningKey;
use fe2o3_compiler_execution_protocol::{
    CompilerExecutionIssuerMeasurementV1 as Measurement, CompilerExecutionIssuerPolicyV3 as Policy,
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

fn transcript(nonce: u8, mask: u8, b: &mut Budget<'_>) -> Receiver {
    let (policy, charge) = Policy::new(
        7,
        Measurement::new([1; 32], 1024).unwrap(),
        Measurement::new([2; 32], 2048).unwrap(),
        SigningKey::from_bytes(&[3; 32]).verifying_key().to_bytes(),
        SigningKey::from_bytes(&[4; 32]).verifying_key().to_bytes(),
        b,
    )
    .unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (hello, charge) =
        Record::hello(&policy, [5; 32], [nonce; 32], mask, 4096, (71, 72), b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (challenge, charge) = Record::challenge(&hello, [6; 32], b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (last, charge) = Record::input(&challenge, challenge.roles().last().unwrap(), b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (ack, charge) = Record::enforcement_unavailable(&last, b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let mut receiver = Receiver::empty();
    receiver.hello = Some(hello);
    receiver.challenge = Some(challenge);
    receiver.last = Some(last);
    receiver.ack = Some(ack);
    receiver
}

fn coordinates(challenge: &Record) -> Enrollment {
    Enrollment {
        rustc_invocation_sha256: [9; 32],
        intake_invocation_identity: *challenge.invocation_identity(),
        invocation_bytes: challenge.invocation_bytes(),
        native_policy_sha256: *challenge.policy_identity(),
        policy_generation: 7,
        binding_count: Some(1),
    }
}

#[test]
fn every_stdio_mask_joins_exact_original_transcript_and_intake_domain() {
    for mask in 0..8 {
        let mut work = Work::new(usize::MAX);
        let mut b = Budget::new(&mut work, usize::MAX);
        let receiver = transcript(10, mask, &mut b);
        let before = (b.work(), b.storage());
        let challenge = original_transcript(&receiver, &mut b).unwrap();
        let value = coordinates(challenge);
        assert_ne!(
            value.rustc_invocation_sha256,
            value.intake_invocation_identity
        );
        matches_intake(&value, challenge, &mut b).unwrap();
        assert_eq!(b.work() - before.0, 3 * RECORD_WORK + COMPARE_WORK);
        assert_eq!(b.storage(), before.1);
    }
}

#[test]
fn missing_or_foreign_original_records_refuse_even_when_coordinates_match() {
    for foreign in [false, true] {
        for index in 0..4 {
            let mut work = Work::new(usize::MAX);
            let mut b = Budget::new(&mut work, usize::MAX);
            let mut receiver = transcript(10, 7, &mut b);
            let mut other = transcript(11, 7, &mut b);
            let (target, replacement) = match index {
                0 => (&mut receiver.hello, &mut other.hello),
                1 => (&mut receiver.challenge, &mut other.challenge),
                2 => (&mut receiver.last, &mut other.last),
                _ => (&mut receiver.ack, &mut other.ack),
            };
            *target = if foreign { replacement.take() } else { None };
            let storage = b.storage();
            assert!(original_transcript(&receiver, &mut b).is_err());
            assert_eq!(b.storage(), storage);
        }
    }
}

#[test]
fn policy_invocation_raw_digest_and_length_substitutions_refuse() {
    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, usize::MAX);
    let receiver = transcript(10, 0, &mut b);
    let challenge = receiver.challenge.as_ref().unwrap();
    for case in 0..4 {
        let mut value = coordinates(challenge);
        match case {
            0 => value.native_policy_sha256[0] ^= 1,
            1 => value.intake_invocation_identity[0] ^= 1,
            2 => value.intake_invocation_identity = value.rustc_invocation_sha256,
            _ => value.invocation_bytes += 1,
        }
        let before = b.work();
        assert!(matches!(
            matches_intake(&value, challenge, &mut b),
            Err(ProofHelperLaunchError::Invalid(_))
        ));
        assert_eq!(b.work() - before, COMPARE_WORK);
    }
}

#[test]
fn only_consumed_confirmed_phase_without_prior_output_can_join() {
    for state in [
        State::Received,
        State::Prepared,
        State::HelperReady,
        State::CompilerGated,
        State::Interrupting,
        State::Interrupted,
        State::Armed,
        State::AwaitingExec,
        State::FirstExec,
        State::HeldExec,
        State::ConfirmedExec,
        State::Running,
        State::RootExitHeld,
        State::PublicationObserved,
        State::AwaitingTerminal,
        State::Retiring,
        State::CompletionReady,
        State::DrainingCompletion,
        State::Completed,
        State::Failed,
        State::Cancelled,
    ] {
        assert_eq!(
            require_transition(state, State::Failed, false, false).is_ok(),
            state == State::ConfirmedExec
        );
        assert_eq!(
            require_transition(State::ConfirmedExec, state, false, false).is_ok(),
            state == State::Failed
        );
    }
    assert!(require_transition(State::ConfirmedExec, State::Failed, true, false).is_err());
    assert!(require_transition(State::ConfirmedExec, State::Failed, false, true).is_err());
}

#[test]
fn deadline_must_still_be_live_after_the_original_query() {
    assert!(require_deadline(None).is_err());
    assert!(require_deadline(Some(Instant::now())).is_err());
    assert!(require_deadline(Some(Instant::now() - Duration::from_secs(1))).is_err());
    require_deadline(Some(Instant::now() + Duration::from_secs(30))).unwrap();
}

// No Prepared, native Attempt, fake process or protected owner is constructed.
fn incomplete_request<'a>(b: &Budget<'a>) -> RootCompilerRequest<'a> {
    RootCompilerRequest {
        prepared: None,
        receiver: Arc::new(Receiver::empty()),
        backing: None,
        helper: None,
        attempt: None,
        terminal: None,
        completion: None,
        enrollment: None,
        state: State::Failed,
        ledger: b.work_ledger_identity_v1(),
        address: b as *const Budget<'_> as usize,
        reserved: b.storage(),
    }
}

#[test]
fn foreign_ledger_moved_account_or_missing_intake_never_installs_a_join() {
    let mut w1 = Work::new(usize::MAX);
    let mut w2 = Work::new(usize::MAX);
    let mut b1 = Budget::new(&mut w1, usize::MAX);
    let mut b2 = Budget::new(&mut w2, usize::MAX);
    let floor = Receiver::STORAGE + RootCompilerRequest::ENVELOPE;
    b1.reserve_storage(floor).unwrap();
    b2.reserve_storage(floor).unwrap();
    let mut request = incomplete_request(&b1);
    std::mem::swap(&mut b1, &mut b2);
    for b in [&mut b1, &mut b2] {
        assert!(matches!(
            request.join_original_enrollment(State::ConfirmedExec, b),
            Err(Error::Resource(Resource::Accounting))
        ));
        assert_eq!(b.storage(), floor);
        assert!(request.enrollment.is_none());
    }
    std::mem::swap(&mut b1, &mut b2);
    assert!(matches!(
        request.join_original_enrollment(State::ConfirmedExec, &mut b1),
        Err(Error::Invalid {
            reason: "compiler request requires complete original intake",
            ..
        })
    ));
    assert!(request.enrollment.is_none());
    assert_eq!(b1.storage(), floor);
}

#[test]
fn insufficient_join_input_scratch_or_work_and_prior_denials_refuse() {
    let floor = Receiver::STORAGE + RootCompilerRequest::ENVELOPE;
    for case in 0..4 {
        let mut work = Work::new(if case == 2 { ENTRY - 1 } else { ENTRY });
        let mut b = Budget::new(&mut work, floor + SCRATCH - usize::from(case == 1));
        b.reserve_storage(floor).unwrap();
        let mut request = incomplete_request(&b);
        if case == 0 {
            b.release_storage(1).unwrap();
        }
        if case == 3 {
            assert!(b.charge_work(ENTRY + 1).is_err());
        }
        let before = (b.work(), b.storage());
        assert!(matches!(
            request.join_original_enrollment(State::ConfirmedExec, &mut b),
            Err(Error::Resource(_))
        ));
        assert!(request.enrollment.is_none());
        assert_eq!(b.storage(), before.1);
        if case == 3 {
            assert_eq!(b.work(), before.0);
        }
    }
}

#[test]
fn startup_funds_query_and_callback_once_with_inline_result_storage() {
    let startup = RootCompilerRequest::runtime_startup_quota().unwrap();
    let query = quota::original_enrollment().unwrap();
    let policy = compiler_attempt::Attempt::original_policy_identity_quota().unwrap();
    assert!(startup.work() >= query.work() + policy.work() + WORK);
    assert!(startup.scratch() >= query.scratch() + policy.scratch() + SCRATCH);
    assert!(RootCompilerRequest::ENVELOPE >= size_of::<Option<Option<Mapping>>>());
    let one =
        crate::InheritedCompilerExecutionDeploymentV3::original_root_startup_quota(1, 1).unwrap();
    let two =
        crate::InheritedCompilerExecutionDeploymentV3::original_root_startup_quota(2, 1).unwrap();
    assert_eq!(one.request_storage(), two.request_storage());
}

#[test]
fn projection_preserves_absence_and_copies_only_original_cpu_coordinates() {
    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, usize::MAX);
    let receiver = transcript(10, 0, &mut b);
    let mut value = coordinates(receiver.challenge.as_ref().unwrap());
    b.reserve_storage(size_of::<Enrollment>() + size_of::<Option<Mapping>>())
        .unwrap();
    for count in [
        None,
        Some(1),
        Some(fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 as u32),
    ] {
        value.binding_count = count;
        let before = (b.work(), b.storage());
        let projected = mapping_expectation(&value, &mut b).unwrap();
        assert_eq!(projected.map(|p| p.enrollment_binding_count), count);
        if let Some(projected) = projected {
            assert_eq!(
                projected.rustc_invocation_sha256,
                value.rustc_invocation_sha256
            );
            assert_ne!(
                projected.rustc_invocation_sha256,
                value.intake_invocation_identity
            );
            assert_eq!(projected.native_policy_sha256, value.native_policy_sha256);
            assert_eq!(projected.policy_generation, value.policy_generation);
        }
        assert_eq!(b.work() - before.0, PROJECT_WORK);
        assert_eq!(b.storage(), before.1);
    }
}

#[test]
fn zero_or_overbound_original_bindings_never_fall_back_to_absent() {
    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, usize::MAX);
    let receiver = transcript(10, 0, &mut b);
    let mut value = coordinates(receiver.challenge.as_ref().unwrap());
    b.reserve_storage(size_of::<Enrollment>() + size_of::<Option<Mapping>>())
        .unwrap();
    for count in [
        0,
        fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 as u32 + 1,
        u32::MAX,
    ] {
        value.binding_count = Some(count);
        let before = b.work();
        assert!(matches!(
            mapping_expectation(&value, &mut b),
            Err(ProofHelperLaunchError::Invalid(_))
        ));
        assert_eq!(b.work() - before, PROJECT_WORK);
    }
}

#[test]
fn projected_expectation_rejects_inventory_policy_generation_count_and_digest_domain_changes() {
    use fe2o3_compiler_lineage::RustcEnrollmentInventoryHeaderV1 as Header;
    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, usize::MAX);
    let receiver = transcript(10, 0, &mut b);
    let value = coordinates(receiver.challenge.as_ref().unwrap());
    b.reserve_storage(size_of::<Enrollment>() + size_of::<Option<Mapping>>() + size_of::<Header>())
        .unwrap();
    let expected = mapping_expectation(&value, &mut b).unwrap().unwrap();
    let original = Header {
        kernel_count: 1,
        enrollment_binding_count: 1,
        invocation_identity: value.rustc_invocation_sha256,
        native_policy_identity: value.native_policy_sha256,
        native_policy_generation: value.policy_generation,
    };
    b.charge_work(Mapping::HEADER_MATCH_WORK).unwrap();
    assert!(expected.matches_header(&original));
    for case in 0..5 {
        let mut changed = original;
        match case {
            0 => changed.invocation_identity[0] ^= 1,
            1 => changed.invocation_identity = value.intake_invocation_identity,
            2 => changed.native_policy_identity[0] ^= 1,
            3 => changed.native_policy_generation += 1,
            _ => changed.enrollment_binding_count += 1,
        }
        b.charge_work(Mapping::HEADER_MATCH_WORK).unwrap();
        assert!(!expected.matches_header(&changed));
    }
}

#[test]
fn mapping_projection_refuses_short_work_and_swallowed_prior_denials() {
    let mut fixture_work = Work::new(usize::MAX);
    let mut fixture_budget = Budget::new(&mut fixture_work, usize::MAX);
    let receiver = transcript(10, 0, &mut fixture_budget);
    let value = coordinates(receiver.challenge.as_ref().unwrap());
    for case in 0..3 {
        let floor = size_of::<Enrollment>() + size_of::<Option<Mapping>>();
        let mut work = Work::new(PROJECT_WORK - usize::from(case == 0));
        let mut b = Budget::new(&mut work, floor);
        b.reserve_storage(floor).unwrap();
        if case == 1 {
            assert!(b.charge_work(PROJECT_WORK + 1).is_err());
        }
        if case == 2 {
            assert!(b.reserve_storage(1).is_err());
        }
        let before = (b.work(), b.storage(), b.failed_work(), b.failed_storage());
        assert!(matches!(
            mapping_expectation(&value, &mut b),
            Err(ProofHelperLaunchError::Resource(_))
        ));
        assert_eq!(b.storage(), floor);
        if case != 0 {
            assert_eq!(
                (b.work(), b.storage(), b.failed_work(), b.failed_storage()),
                before
            );
        }
    }
}

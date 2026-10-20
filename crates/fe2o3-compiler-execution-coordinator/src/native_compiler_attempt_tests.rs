//! Mechanical controls only; genuine execution belongs to the installed matrix.
use super::*;

#[test]
fn launch_consumes_actual_helper_and_preserves_original_work_lifetime() {
    let _: for<'work> unsafe fn(
        Helper,
        &Receiver,
        Credentials,
        Instant,
        &mut Cleanup,
        &mut Budget<'work>,
    ) -> AttemptResult<(Attempt<'work>, usize)> = launch;
}

#[test]
fn envelope_accounts_for_inventory_trace_stage_and_all_three_unreleased_gate_status_owners() {
    assert_eq!(
        Attempt::ENVELOPE
            + size_of::<Trace<'static>>()
            + size_of::<Executables>()
            + size_of::<Stage>()
            + 3 * size_of::<OwnedFd>(),
        size_of::<Attempt<'static>>(),
    );
    assert!(Attempt::ENVELOPE >= size_of::<usize>());
}

#[test]
fn equal_contents_cannot_replace_an_original_staged_object() {
    let original = tempfile::tempfile().unwrap();
    let other = tempfile::tempfile().unwrap();
    let duplicate = original.try_clone().unwrap();
    same_object(original.as_fd(), &duplicate).unwrap();
    assert!(matches!(
        same_object(original.as_fd(), &other),
        Err(Failure::Invalid(_))
    ));
    assert_eq!(
        original.metadata().unwrap().len(),
        other.metadata().unwrap().len()
    );
}

#[test]
fn final_validation_requires_both_typed_child_modes() {
    for (checkpoints, confinement) in [(false, false), (false, true), (true, false)] {
        assert!(require_stage_modes(checkpoints, confinement).is_err());
    }
    require_stage_modes(true, true).unwrap();
}

#[test]
fn runtime_step_quote_composes_original_owner_helper_and_controller() {
    let quote = Attempt::runtime_step_quota().unwrap();
    let trace = crate::native_v3::NativeAttempt::<Helper>::runtime_backing_quota().unwrap();
    let helper = Helper::checkpoint_access_quota().unwrap();
    assert_eq!(
        quote.work(),
        LOCAL_WORK + trace.work() + helper.work() + Controller::step_work().unwrap()
    );
    assert_eq!(
        quote.scratch(),
        FRAME + trace.scratch() + helper.scratch() + Controller::STEP_SCRATCH
    );
}

#[test]
fn completion_quote_preserves_original_issuer_and_outer_attempt_account() {
    let quote = Attempt::publication_completion_quota().unwrap();
    let inner = crate::native_v3::NativeAttempt::<Helper>::publication_completion_quota().unwrap();
    assert_eq!(quote.work(), LOCAL_WORK + inner.work());
    assert_eq!(quote.scratch(), FRAME + inner.scratch());
}

#[test]
fn enrollment_quote_composes_existing_validation_original_custody_and_projection() {
    use crate::compiler_invocation_backing::OriginalInvocationEnrollment as Projection;
    let quote = super::super::quota::original_enrollment().unwrap();
    let validation = super::super::quota::refusal().unwrap();
    let custody =
        crate::native_v3::NativeAttempt::<Helper>::original_enrollment_custody_quota().unwrap();
    assert_eq!(
        quote.work(),
        validation.work() + custody.work() + Projection::PROJECTION_WORK
    );
    assert_eq!(
        quote.scratch(),
        validation.scratch() + custody.scratch() + Projection::PROJECTION_SCRATCH
    );
}

#[test]
fn runtime_arm_and_cancellation_quotes_include_the_outer_attempt() {
    let arm = Attempt::arm_runtime_quota().unwrap();
    let inner = Trace::runtime_takeover_quota().unwrap();
    assert_eq!(arm.work(), LOCAL_WORK + inner.work());
    assert_eq!(arm.scratch(), FRAME + inner.scratch());
    let cancel = Attempt::cancellation_quota().unwrap();
    let issued = crate::native_v3::NativeAttempt::<Helper>::runtime_cancellation_quota().unwrap();
    assert_eq!(cancel.work(), LOCAL_WORK + issued.work());
    assert_eq!(cancel.scratch(), FRAME + issued.scratch());
}

#[test]
fn runtime_poll_confirmation_and_capture_have_closed_finite_quotes() {
    let poll = Attempt::runtime_poll_quota().unwrap();
    let original = Trace::gated_operation_quota().unwrap();
    assert_eq!(
        (poll.work(), poll.scratch()),
        (original.work(), original.scratch())
    );
    let confirm = Attempt::runtime_confirmation_quota().unwrap();
    let inner = Trace::runtime_confirmation_quota().unwrap();
    assert_eq!(confirm.work(), LOCAL_WORK + inner.work());
    assert_eq!(confirm.scratch(), FRAME + inner.scratch());
    let validation = super::super::quota::refusal().unwrap();
    let interrupt = super::super::quota::runtime_interrupt().unwrap();
    assert_eq!(interrupt.work(), validation.work() + original.work());
    let capture = super::super::quota::runtime_capture().unwrap();
    let trace = Trace::runtime_backing_quota().unwrap();
    assert_eq!(
        capture.work(),
        validation.work() + trace.work() + Controller::CONFINED_IMAGE_WORK
    );
    assert_eq!(
        capture.scratch(),
        validation.scratch() + trace.scratch() + Controller::CONFINED_IMAGE_SCRATCH
    );
}

#[test]
fn gate_quote_funds_original_device_custody_identity_and_every_interrupted_write() {
    use fe2o3_protected_service_spawn::native_spawn::{
        RootRuntimeTraceV1 as Runtime, RootTaskObservationV2 as View,
    };
    let quote = Attempt::runtime_gate_quota().unwrap();
    let trace = Trace::runtime_backing_quota().unwrap();
    assert_eq!(
        quote.work(),
        LOCAL_WORK
            + trace.work()
            + 2 * Runtime::OPERATION_WORK
            + Runtime::ROOT_OBSERVATION_WORK
            + View::VIEW_WORK
            + View::DEVICE_CONFINEMENT_WORK
            + View::IDENTITY_WORK
            + launch_io::MAX_GATE_ATTEMPTS * launch_io::Boundary::GateRelease.work()
    );
    assert_eq!(
        quote.scratch(),
        FRAME
            + CompilerConfinement::STORAGE
            + trace.scratch()
            + Runtime::OPERATION_SCRATCH
            + View::VIEW_SCRATCH
            + View::DEVICE_CONFINEMENT_SCRATCH
            + View::IDENTITY_SCRATCH
            + View::IDENTITY_STORAGE
    );
}

#[test]
fn released_gate_poll_and_policy_identity_keep_original_account_wrappers() {
    use fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1 as Runtime;
    let poll = Attempt::first_exec_poll_quota().unwrap();
    let trace = Trace::runtime_backing_quota().unwrap();
    assert_eq!(
        poll.work(),
        LOCAL_WORK + trace.work() + Runtime::OPERATION_WORK
    );
    assert_eq!(
        poll.scratch(),
        FRAME + trace.scratch() + Runtime::OPERATION_SCRATCH
    );
    let identity = Attempt::original_policy_identity_quota().unwrap();
    let inner = crate::native_v3::NativeAttempt::<Helper>::original_policy_identity_quota();
    assert_eq!(identity.work(), LOCAL_WORK + inner.work());
    assert_eq!(identity.scratch(), FRAME + inner.scratch());
}

#[test]
fn confinement_receipt_quotes_original_identity_and_actual_domain_not_stage_flags() {
    use fe2o3_protected_service_spawn::native_spawn::{
        RootRuntimeTraceV1 as Runtime, RootTaskObservationV2 as View,
    };
    assert_eq!(
        CompilerConfinement::VALIDATE_WORK,
        8 + Runtime::IDENTITY_COMPARISON_WORK
            + Runtime::ROOT_OBSERVATION_WORK
            + View::VIEW_WORK
            + View::DEVICE_CONFINEMENT_WORK
    );
    assert_eq!(
        CompilerConfinement::STORAGE,
        size_of::<CompilerConfinement>() + View::IDENTITY_STORAGE
    );
    assert_eq!(
        Controller::CONFINED_IMAGE_WORK,
        Controller::INITIAL_IMAGE_WORK + CompilerConfinement::VALIDATE_WORK
    );
    assert_eq!(
        Controller::CONFINED_IMAGE_SCRATCH,
        Controller::INITIAL_IMAGE_SCRATCH + CompilerConfinement::VALIDATE_SCRATCH
    );
}

#[test]
fn publication_quote_includes_fresh_held_entry_and_original_consuming_custody() {
    let maximum = 4096;
    let quote = Attempt::publication_observation_quota(maximum).unwrap();
    let held = Attempt::runtime_step_quota().unwrap();
    let inner =
        crate::native_v3::NativeAttempt::<Helper>::publication_observation_quota(maximum).unwrap();
    assert_eq!(quote.work(), held.work() + LOCAL_WORK + inner.work());
    assert_eq!(quote.scratch(), held.scratch() + FRAME + inner.scratch());
}

#[test]
fn running_publication_and_exit_quotes_reuse_original_root_custody() {
    type Issued = crate::native_v3::NativeAttempt<'static, Helper>;
    let rpc = Attempt::publication_service_quota(4096).unwrap();
    let inner = Issued::publication_service_quota(4096).unwrap();
    assert_eq!(rpc.work(), LOCAL_WORK + inner.work());
    assert_eq!(rpc.scratch(), FRAME + inner.scratch());
    let exit = Attempt::retired_publication_confirmation_quota().unwrap();
    let step = Attempt::runtime_step_quota().unwrap();
    let retired = Issued::retired_publication_quota().unwrap();
    assert_eq!(exit.work(), step.work() + LOCAL_WORK + retired.work());
    assert_eq!(exit.scratch(), step.scratch() + FRAME + retired.scratch());
}

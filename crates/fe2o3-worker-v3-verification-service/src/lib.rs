#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

#[cfg(not(target_os = "linux"))]
compile_error!("fe2o3-worker-v3-verification-service requires Linux descriptor semantics");

mod service;
mod service_v2;
mod service_v3;

pub use service_v3::{
    WorkerV3VerificationReadinessFailureV3, WorkerV3VerificationReadyServiceSessionV3,
    WorkerV3VerificationServiceErrorV3,
};

pub use service::{
    FramedWorkerV3VerificationSessionV1, RejectedWorkerV3VerificationSessionV1,
    RetainedWorkerV3VerificationPayloadV1, WorkerV3VerificationCallerV1,
    WorkerV3VerificationChallengeReplayGuardV1, WorkerV3VerificationMeasurementResolverV1,
    WorkerV3VerificationPolicyResolverV1, WorkerV3VerificationRejectionReasonV1,
    WorkerV3VerificationServiceErrorV1, WorkerV3VerificationSessionOutcomeV1,
    prepare_worker_v3_verification_receiver_v1, serve_worker_v3_verification_session_v1,
};
pub use service_v2::{
    CompletedWorkerV3VerificationSessionV2, PendingRejectedWorkerV3VerificationTerminalSessionV2,
    PendingWorkerV3VerificationCurrentRecordSessionV2,
    PendingWorkerV3VerificationTerminalSessionV2, RejectedWorkerV3VerificationBeginV2,
    WorkerV3VerificationAcceptedServiceAdmissionFailureV2,
    WorkerV3VerificationAcceptedServiceEndpointV2, WorkerV3VerificationBeginOutcomeV2,
    WorkerV3VerificationChallengeReservationProviderV2, WorkerV3VerificationCurrentRecordOutcomeV2,
    WorkerV3VerificationRejectedSendFailureV2, WorkerV3VerificationRejectionReasonV2,
    WorkerV3VerificationServiceErrorV2, WorkerV3VerificationTerminalSendFailureV2,
    begin_worker_v3_verification_accepted_session_until_v2,
    begin_worker_v3_verification_session_until_v2, begin_worker_v3_verification_session_v2,
};

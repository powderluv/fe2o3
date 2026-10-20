#![deny(unsafe_code)]
#![doc = include_str!("../README.md")]

mod protocol;
mod protocol_v2;
mod readiness_v3;

pub use readiness_v3::{
    MAX_WORKER_V3_VERIFICATION_BEGIN_BYTES_V3, WORKER_V3_VERIFICATION_BEGIN_PREFIX_BYTES_V3,
    WORKER_V3_VERIFICATION_READY_BYTES_V3, WorkerV3VerificationProtocolErrorV3,
    WorkerV3VerificationReadyFrameV3,
};

pub use protocol::{
    MAX_WORKER_V3_VERIFICATION_ENTRIES_V1, MAX_WORKER_V3_VERIFICATION_ENTRY_NAME_BYTES_V1,
    MAX_WORKER_V3_VERIFICATION_ENVELOPE_FD_BYTES_V1, MAX_WORKER_V3_VERIFICATION_HSACO_FD_BYTES_V1,
    MAX_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1, MIN_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1,
    WORKER_V3_VERIFICATION_FD_PAYLOADS_V1, WORKER_V3_VERIFICATION_REQUEST_MAGIC_V1,
    WORKER_V3_VERIFICATION_REQUEST_VERSION_V1, WORKER_V3_VERIFICATION_RESPONSE_BYTES_V1,
    WORKER_V3_VERIFICATION_RESPONSE_MAGIC_V1, WorkerV3VerificationEntryCoordinateV1,
    WorkerV3VerificationEntryIdentityFieldV1, WorkerV3VerificationEntryNameFieldV1,
    WorkerV3VerificationFdPayloadDescriptorV1, WorkerV3VerificationFdPayloadKindV1,
    WorkerV3VerificationFreshChallengeV1, WorkerV3VerificationIdentityFieldV1,
    WorkerV3VerificationMeasurementIdentityV1, WorkerV3VerificationPolicyIdentityV1,
    WorkerV3VerificationProtocolErrorV1, WorkerV3VerificationRequestIdentityV1,
    WorkerV3VerificationRequestV1, WorkerV3VerificationResponseDispositionV1,
    WorkerV3VerificationResponseIdentityV1, WorkerV3VerificationResponseV1,
    WorkerV3VerificationRosterIdentityV1, WorkerV3VerificationTranscriptIdentityV1,
};
pub use protocol_v2::{
    MAX_WORKER_V3_VERIFICATION_APPLICATION_RESPONSE_BYTES_V2,
    MAX_WORKER_V3_VERIFICATION_TERMINAL_BYTES_V2, MIN_WORKER_V3_VERIFICATION_TERMINAL_BYTES_V2,
    WORKER_V3_VERIFICATION_CHALLENGE_BYTES_V2, WORKER_V3_VERIFICATION_CURRENT_RECORD_BYTES_V2,
    WorkerV3VerificationChallengeDispositionV2, WorkerV3VerificationChallengeFrameV2,
    WorkerV3VerificationChallengeReservationV2, WorkerV3VerificationCurrentRecordFrameV2,
    WorkerV3VerificationProtocolErrorV2, WorkerV3VerificationTerminalDispositionV2,
    WorkerV3VerificationTerminalFrameV2,
};

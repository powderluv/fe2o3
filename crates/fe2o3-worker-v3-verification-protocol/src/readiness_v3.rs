//! Explicit opt-in pathname readiness framing. These bytes grant no authority.
use crate::{
    MAX_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1, WorkerV3VerificationProtocolErrorV1,
    WorkerV3VerificationRequestV1,
};
use std::fmt;

/// Exact ReadyV3 packet and BeginV3 prefix length.
pub const WORKER_V3_VERIFICATION_READY_BYTES_V3: usize = 64;
/// Exact BeginV3 prefix length.
pub const WORKER_V3_VERIFICATION_BEGIN_PREFIX_BYTES_V3: usize = 64;
/// Maximum composite BeginV3 packet size, including the nonce prefix.
pub const MAX_WORKER_V3_VERIFICATION_BEGIN_BYTES_V3: usize =
    MAX_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1 + WORKER_V3_VERIFICATION_BEGIN_PREFIX_BYTES_V3;
const READY_MAGIC: &[u8; 8] = b"F3WVRD3\0";
const BEGIN_MAGIC: &[u8; 8] = b"F3WVBG3\0";

/// Inert canonical nonce framing; transport state, not these bytes, enforces one use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerV3VerificationReadyFrameV3 {
    nonce: [u8; 32],
}
impl WorkerV3VerificationReadyFrameV3 {
    /// Constructs inert framing from a nonzero nonce. This does not establish freshness.
    pub fn new(nonce: [u8; 32]) -> Result<Self, WorkerV3VerificationProtocolErrorV3> {
        if nonce == [0; 32] {
            return Err(WorkerV3VerificationProtocolErrorV3::ZeroNonce);
        }
        Ok(Self { nonce })
    }
    /// Encodes the exact 64-byte ReadyV3 packet.
    pub fn encode_canonical(&self) -> [u8; 64] {
        self.prefix(READY_MAGIC)
    }
    /// Decodes only the canonical ReadyV3 format, never V2 or an unknown version.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, WorkerV3VerificationProtocolErrorV3> {
        Self::decode_prefix(bytes, READY_MAGIC)
    }
    /// Encodes one nonce-bound composite Begin, preserving the inner V1 request bytes.
    pub fn encode_begin(&self, request: &WorkerV3VerificationRequestV1) -> Vec<u8> {
        let mut bytes = self.prefix(BEGIN_MAGIC).to_vec();
        bytes.extend_from_slice(request.encode_canonical());
        bytes
    }
    /// Checks the canonical prefix and this connection's nonce before decoding Begin.
    pub fn decode_begin(
        &self,
        bytes: &[u8],
    ) -> Result<WorkerV3VerificationRequestV1, WorkerV3VerificationProtocolErrorV3> {
        if bytes.len() < 64 || bytes.len() > MAX_WORKER_V3_VERIFICATION_BEGIN_BYTES_V3 {
            return Err(WorkerV3VerificationProtocolErrorV3::Length);
        }
        let observed = Self::decode_prefix(&bytes[..64], BEGIN_MAGIC)?;
        if observed != *self {
            return Err(WorkerV3VerificationProtocolErrorV3::NonceMismatch);
        }
        WorkerV3VerificationRequestV1::decode_canonical(&bytes[64..])
            .map_err(WorkerV3VerificationProtocolErrorV3::Request)
    }
    /// Inert readiness never grants verification or execution authority.
    pub const fn grants_authority(&self) -> bool {
        false
    }
    fn prefix(&self, magic: &[u8; 8]) -> [u8; 64] {
        let mut bytes = [0; 64];
        bytes[..8].copy_from_slice(magic);
        bytes[8..10].copy_from_slice(&3_u16.to_le_bytes());
        bytes[16..48].copy_from_slice(&self.nonce);
        bytes
    }
    fn decode_prefix(
        bytes: &[u8],
        magic: &[u8; 8],
    ) -> Result<Self, WorkerV3VerificationProtocolErrorV3> {
        if bytes.len() != 64 {
            return Err(WorkerV3VerificationProtocolErrorV3::Length);
        }
        if &bytes[..8] != magic {
            return Err(WorkerV3VerificationProtocolErrorV3::Magic);
        }
        if bytes[8..10] != 3_u16.to_le_bytes() {
            return Err(WorkerV3VerificationProtocolErrorV3::Version);
        }
        if bytes[10..16].iter().chain(&bytes[48..64]).any(|b| *b != 0) {
            return Err(WorkerV3VerificationProtocolErrorV3::Reserved);
        }
        let mut nonce = [0; 32];
        nonce.copy_from_slice(&bytes[16..48]);
        Self::new(nonce)
    }
}
/// Closed V3 readiness framing refusals.
#[derive(Debug)]
#[non_exhaustive]
pub enum WorkerV3VerificationProtocolErrorV3 {
    /// Not the exact bounded packet or prefix size.
    Length,
    /// Wrong phase magic, including legacy packets.
    Magic,
    /// Unsupported explicit version.
    Version,
    /// A reserved byte is nonzero.
    Reserved,
    /// The nonce is all zero.
    ZeroNonce,
    /// The nonce belongs to another connection or readiness event.
    NonceMismatch,
    /// The inner request is not canonical.
    Request(WorkerV3VerificationProtocolErrorV1),
}
impl fmt::Display for WorkerV3VerificationProtocolErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V3 readiness framing refusal: {self:?}")
    }
}
impl std::error::Error for WorkerV3VerificationProtocolErrorV3 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MAX_WORKER_V3_VERIFICATION_ENTRIES_V1, MAX_WORKER_V3_VERIFICATION_ENTRY_NAME_BYTES_V1,
        MIN_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1, WorkerV3VerificationEntryCoordinateV1,
        WorkerV3VerificationFdPayloadDescriptorV1, WorkerV3VerificationFreshChallengeV1,
        WorkerV3VerificationMeasurementIdentityV1, WorkerV3VerificationPolicyIdentityV1,
        WorkerV3VerificationRosterIdentityV1,
    };

    fn request_at_bound(maximum: bool) -> WorkerV3VerificationRequestV1 {
        let count = if maximum {
            MAX_WORKER_V3_VERIFICATION_ENTRIES_V1
        } else {
            1
        };
        let entries = (0..count)
            .map(|i| {
                let ordinal = u32::try_from(i).unwrap();
                let mut lineage = [1; 32];
                lineage[..4].copy_from_slice(&ordinal.to_le_bytes());
                let mut marker = lineage;
                marker[31] = 2;
                let name = |prefix: char| {
                    if maximum {
                        let mut s = format!("{prefix}{i:08}");
                        s.push_str(
                            &"x".repeat(MAX_WORKER_V3_VERIFICATION_ENTRY_NAME_BYTES_V1 - s.len()),
                        );
                        s
                    } else {
                        prefix.to_string()
                    }
                };
                WorkerV3VerificationEntryCoordinateV1::new(
                    ordinal,
                    name('a'),
                    name('b'),
                    lineage,
                    marker,
                    [3; 32],
                )
                .unwrap()
            })
            .collect();
        WorkerV3VerificationRequestV1::new(
            WorkerV3VerificationFreshChallengeV1::new([1; 32]).unwrap(),
            WorkerV3VerificationRosterIdentityV1::new([2; 32]).unwrap(),
            WorkerV3VerificationPolicyIdentityV1::new([3; 32]).unwrap(),
            WorkerV3VerificationMeasurementIdentityV1::new([4; 32]).unwrap(),
            WorkerV3VerificationFdPayloadDescriptorV1::load_envelope_v2(1, [5; 32]).unwrap(),
            WorkerV3VerificationFdPayloadDescriptorV1::finalized_hsaco(1, [6; 32]).unwrap(),
            entries,
        )
        .unwrap()
    }

    #[test]
    fn composite_begin_exact_minimum_and_maximum_round_trip() {
        let ready = WorkerV3VerificationReadyFrameV3::new([7; 32]).unwrap();
        for maximum in [false, true] {
            let request = request_at_bound(maximum);
            let bytes = ready.encode_begin(&request);
            assert_eq!(
                bytes.len(),
                64 + if maximum {
                    MAX_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1
                } else {
                    MIN_WORKER_V3_VERIFICATION_REQUEST_BYTES_V1
                }
            );
            assert_eq!(ready.decode_begin(&bytes).unwrap(), request);
            assert_eq!(&bytes[64..], request.encode_canonical());
            assert!(!ready.decode_begin(&bytes).unwrap().grants_authority());
        }
    }

    #[test]
    fn composite_begin_rejects_all_prefix_and_inner_canonical_mutations() {
        let ready = WorkerV3VerificationReadyFrameV3::new([7; 32]).unwrap();
        let bytes = ready.encode_begin(&request_at_bound(false));
        for n in 0..bytes.len() {
            assert!(ready.decode_begin(&bytes[..n]).is_err(), "truncated at {n}");
        }
        for i in (0..16).chain(48..64) {
            let mut bad = bytes.clone();
            bad[i] ^= 1;
            assert!(ready.decode_begin(&bad).is_err(), "prefix {i}");
        }
        let mut zero = bytes.clone();
        zero[16..48].fill(0);
        assert!(matches!(
            ready.decode_begin(&zero),
            Err(WorkerV3VerificationProtocolErrorV3::ZeroNonce)
        ));
        let mut other = bytes.clone();
        other[16] ^= 1;
        assert!(matches!(
            ready.decode_begin(&other),
            Err(WorkerV3VerificationProtocolErrorV3::NonceMismatch)
        ));
        for i in [64, 72, bytes.len() - 1] {
            let mut bad = bytes.clone();
            bad[i] ^= 1;
            assert!(matches!(
                ready.decode_begin(&bad),
                Err(WorkerV3VerificationProtocolErrorV3::Request(_))
            ));
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(matches!(
            ready.decode_begin(&trailing),
            Err(WorkerV3VerificationProtocolErrorV3::Request(_))
        ));
        let mut oversized = ready.encode_begin(&request_at_bound(true));
        oversized.push(0);
        assert!(matches!(
            ready.decode_begin(&oversized),
            Err(WorkerV3VerificationProtocolErrorV3::Length)
        ));
    }
    #[test]
    fn ready_round_trip_and_non_authority() {
        let f = WorkerV3VerificationReadyFrameV3::new([7; 32]).unwrap();
        assert_eq!(
            WorkerV3VerificationReadyFrameV3::decode_canonical(&f.encode_canonical()).unwrap(),
            f
        );
        assert!(!f.grants_authority());
    }
    #[test]
    fn exact_length_magic_version_reserved_and_nonce_are_closed() {
        let bytes = WorkerV3VerificationReadyFrameV3::new([7; 32])
            .unwrap()
            .encode_canonical();
        for n in 0..64 {
            assert!(WorkerV3VerificationReadyFrameV3::decode_canonical(&bytes[..n]).is_err());
        }
        let mut long = bytes.to_vec();
        long.push(0);
        assert!(WorkerV3VerificationReadyFrameV3::decode_canonical(&long).is_err());
        for i in (0..16).chain(48..64) {
            let mut bad = bytes;
            bad[i] ^= 1;
            assert!(
                WorkerV3VerificationReadyFrameV3::decode_canonical(&bad).is_err(),
                "{i}"
            );
        }
        let mut zero = bytes;
        zero[16..48].fill(0);
        assert!(WorkerV3VerificationReadyFrameV3::decode_canonical(&zero).is_err());
        assert!(WorkerV3VerificationReadyFrameV3::new([0; 32]).is_err());
    }
}

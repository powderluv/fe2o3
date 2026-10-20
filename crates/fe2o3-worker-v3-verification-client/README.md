# fe2o3 Worker V3 verification client

This crate provides the authority-free Unix `SOCK_SEQPACKET` client transport for one Worker V3
verification framing exchange. It admits exactly two caller-owned, immutable memfd snapshots in
protocol order, binds their exact lengths and SHA-256 digests to one canonical request, reopens and
retains only read-only close-on-exec handles, sends the request and both descriptors in one
`SCM_RIGHTS` message, and accepts one exact framing-only response.

The client authenticates neither the service peer nor a verification theorem. A framing receipt
does not grant compiler, executable, load, launch, or safety authority. Protected peer identity,
fresh-challenge replay exclusion, theorem-record authentication, and host promotion remain owned by
later reviewed boundaries.

`WorkerV3VerificationClientV2` adds a strict multi-phase typestate on one connection and one
absolute deadline. `begin` sends the canonical V1 request and its two snapshots without closing the
write half. A reserved response yields two separate move-only values: a service-issued current-record
challenge and a pending client session. The challenge can be consumed directly into
`CompilerExecutionCurrentRecordChallengeV1`; the pending session privately retains the same bytes
and opaque reservation identity only for later correlation.

`WorkerV3VerificationClientV2::admit_until` accepts an exact caller-created monotonic `Instant` and
retains it unchanged for Begin, current-record submission, terminal receipt, and EOF. The compatible
`admit` timeout entrypoint computes one deadline and delegates to `admit_until`; it does not restart
the timeout after admission work.

`admit_connected_path` and `admit_connected_path_until` are explicit alternatives for an already
connected pathname service. They do not discover or connect an endpoint. They require the client
side to remain unnamed and the peer address to exactly match a caller-supplied, lexically validated
canonical absolute filesystem pathname, while the existing `admit` entrypoints continue to require
unnamed endpoints on both sides. The path variants also enable `SO_PASSCRED`, snapshot the service's
connection-time `SO_PEERCRED`, and require every response packet to carry exactly matching
kernel-stamped `SCM_CREDENTIALS`. This continuity check does not decide whether that process
identity is authorized and does not grant verification authority.

The pending session accepts only the exact fixed-size current-record verification and attestation
arrays, sends their canonical V2 frame, closes the write half, and accepts one bounded terminal
packet followed by exact peer EOF. Phase packets cannot carry descriptors or other ancillary data.
The terminal response is opaque and authority-free even when the remote application selected the
application-response disposition.
# V3 Readiness Client

## Explicit Pathname Readiness V3

Choose V3 explicitly at both ends. `WorkerV3VerificationClientV3::admit_connected_path_until`
uses the existing strict pathname/unnamed endpoint checks, enables credential receiving,
and snapshots SO_PEERCRED without sending anything. Consuming `wait_ready` requires exactly
one canonical 64-byte Ready packet with no descriptors and matching SCM_CREDENTIALS.
The returned move-only `WorkerV3VerificationReadyClientV3` may send one nonce-bound composite
Begin and then uses the existing V2 challenge/current/terminal state machine.

The original absolute Instant is never refreshed. Already-queued duplicate Ready/data/EOF
is rejected before Begin; traffic racing the queue probe is still rejected by the next
strict phase. Any readiness or Begin error closes the owned endpoint and ancillary
descriptors. There is no Hello, implicit downgrade, global replay ledger, standalone
client proof of nonce freshness, signer access or verification/execution authority.
V2 callers retain exactly their existing framing and behavior.

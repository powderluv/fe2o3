# fe2o3 Worker V3 verification service boundary

This crate owns one bounded Linux Unix `SOCK_SEQPACKET` exchange for the authority-free Worker V3
verification protocol. It receives one canonical request with exactly two ordered `SCM_RIGHTS`
descriptors, resolves the caller's policy and expected verifier measurement through injected
fail-closed interfaces, consumes the fresh challenge through an injected atomic replay guard, and
captures each exact immutable payload into a receiver-owned sealed memfd.

The receiver enables Linux `SO_PASSCRED` before receiving. The sole request packet must carry
exactly one kernel-stamped `SCM_CREDENTIALS` record matching the connection's `SO_PEERCRED`, plus
exactly one `SCM_RIGHTS` record containing the two ordered descriptors.

`prepare_worker_v3_verification_receiver_v1` must run before the peer can queue a packet: on a
listener before `listen`/`accept`, or on a private socket-pair receiver before exposing the sender.
The service rejects an endpoint without `SO_PASSCRED`, and it rejects any already-queued packet
that lacks the required kernel-stamped credentials.

The caller must shut down the connected socket's write half immediately after the sole packet. The
service requires exact EOF before it resolves policy or consumes replay state; a second packet or
ancillary object terminates the session without a response.

The boundary emits only `RequestFramed` or `RequestRejected`. Framing, descriptor custody, byte
identity, policy selection, measurement selection, and replay exclusion are not verification
theorems. This crate cannot construct protected roster evidence and grants no load or launch
authority. A later protected verifier must independently establish every compiler, executable,
layout, effect, and universal-safety property before a reviewed host promotion can grant authority.

Production resolver implementations must authenticate their policy and measurement stores. A
production replay guard must atomically persist used challenges across every service instance and
restart covered by the policy. The crate deliberately provides no permissive default implementation
for any of these injected decisions.

## V2 state machine

`begin_worker_v3_verification_session_v2` reuses V1's exact Begin request admission, policy and
measurement resolution, caller replay guard, credential checks, and receiver-owned immutable
payload copies. It deliberately does not require write EOF after Begin. It rejects data queued out
of phase, calls a required `WorkerV3VerificationChallengeReservationProviderV2`, sends the
provider's nonzero challenge and opaque reservation identity, and returns a move-only pending
current-record session. No provider default exists. Provider implementations remain responsible for
entropy, uniqueness, atomic reservation, durable replay exclusion across covered service restarts,
and expiry; the generic service cannot prove those properties.

`begin_worker_v3_verification_session_until_v2` accepts one exact caller-created monotonic
`Instant` and retains it unchanged through Begin, current-record receipt, and the terminal send. The
compatible timeout entrypoint computes the deadline once and delegates to the absolute-deadline
entrypoint. Pre-expired deadlines fail before the service receives the queued request or its
descriptors.

The pending state receives one exact fixed-size current-record frame with matching kernel-stamped
credentials and no other ancillary data, then requires client write-half EOF. It strictly decodes
the separate verification and attestation, checks nested byte equality, and correlates the Begin,
challenge, and reservation identity before a terminal application capability can exist. Malformed
or mismatched records produce only a custody-retaining rejection capability.

Only the ready terminal capability can send one bounded opaque application response. Both ready and
rejected capabilities can instead send a generic rejection, and send/construction failures retain
receiver-owned payload custody where the socket result is recoverable. The response remains opaque:
this crate does not treat application bytes, canonical compiler records, or successful transport as
theorem, currentness, load, launch, or protected-key authority. V1 remains available and unchanged
for its one-shot framing-only exchange.

## Connected pathname admission

`WorkerV3VerificationAcceptedServiceEndpointV2::admit` is the explicit V2 admission boundary for
an already accepted pathname connection. It requires the local address to exactly match a
caller-supplied, lexically validated canonical absolute filesystem pathname, an unnamed client
address, exact nonblocking close-on-exec read/write custody, and `SO_PASSCRED` inherited from a
listener prepared before `listen` and `accept`. Admission snapshots the connecting process's
`SO_PEERCRED`; Begin and current-record packets must retain exact kernel-stamped `SCM_CREDENTIALS`
continuity with that snapshot.

`begin_worker_v3_verification_accepted_session_until_v2` consumes only that admitted endpoint. It
does not discover a path, create or connect a socket, listen, or accept. The process whose identity
must be observed must therefore perform `connect` itself, and the service process whose identity
must be observed by the client must create the listener itself. Passing a connected descriptor to
another process does not update Linux's connection-time `SO_PEERCRED` snapshot; a later packet from
that process fails the per-message credential check. The existing unnamed V1 and V2 entrypoints
remain strict and unchanged.
# V3 Readiness Service

## Explicit Pathname Readiness V3

`WorkerV3VerificationReadyServiceSessionV3::prepare_until` consumes an endpoint which
already passed strict V2 admission, repeats the full endpoint/credential validation,
obtains one nonblocking kernel-random nonce, and emits Ready before accepting Begin.
It accepts no Hello or abstract peer. Its move-only private state retains the original
absolute deadline and nonce; consuming `begin` validates the composite nonce-bound request
before using the existing V2 policy, replay, reservation and current-record core.

Preparation failures are terminal. `ready_send_attempted` reports whether Ready may
have been sent; `into_control` returns only an already-shutdown cleanup descriptor,
or None if shutdown failed and it was closed. It never returns a reusable session.
Begin errors close the endpoint and all received payloads. Callers must provide exclusive
transport custody; OwnedFd alone cannot prove absence of external aliases.

This is a separate opt-in production protocol, not a fixture synchronization shortcut.
V2 remains unchanged and does not promise a readiness handshake. New V3 Rust I/O is not
covered by existing model proofs; no native, hardware, loading or key authority is granted.

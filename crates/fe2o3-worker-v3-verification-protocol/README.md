# fe2o3 Worker V3 verification protocol

This crate defines bounded canonical request and response frames for moving one exact Worker V3
roster-verification job across a process boundary. The request binds caller-owned challenge,
roster, policy, and expected verifier-measurement identities; an ordered roster; and exactly two
file-descriptor payload positions for the complete load envelope and finalized HSACO.

The frames are deliberately authority-free. A frame identity establishes byte equality only. It
does not authenticate a peer, prove challenge freshness, validate file-descriptor custody, perform
compiler or machine verification, or construct protected roster evidence. The fresh protocol
challenge is caller-generated entropy with replay exclusion; it is distinct from the host's
deterministic lineage-and-roster challenge. The response vocabulary can acknowledge framing or
reject a request; it cannot encode theorem success.

Logical and export names are carried losslessly with the same bounded canonical validation used by
the kernel descriptor. They remain untrusted duplicate coordinates: the protected service must
cross-check them against its pinned roster policy and the exact descriptor table carried by the
envelope, then recompute the roster identity over those names, marker bindings, and generated host
contracts. Generated host-contract identities may repeat because multiple distinct marker bindings
can intentionally share one ABI and effect contract.

The Unix transport owner must receive the frame and exactly two `SCM_RIGHTS` descriptors as one
authenticated exchange, reject missing or extra descriptors, acquire independent descriptor
custody, enforce the required file type and immutability policy, read exactly the declared length,
reject trailing payload bytes, and compare the SHA-256 digest. The caller owns cryptographically
fresh challenge generation and replay exclusion. A later theorem-record contract must separately
define authenticated verifier output and the only reviewed promotion into host authority.

## Multi-phase protected-verification transport V2

V2 is additive and leaves every V1 frame and semantic unchanged. Its Begin phase uses the complete
canonical V1 request and the same two ordered descriptors, including V1's caller-owned nonce and
external replay policy. After Begin admission, a protected service can return a fixed-size challenge
frame containing a distinct nonzero service-owned compiler-current-record challenge and a nonzero
opaque reservation identity. The provider, not this protocol crate, owns entropy, atomic
reservation, durable replay exclusion across restarts, and expiry.

The client later returns one fixed-size V2 frame containing the exact canonical
`CompilerExecutionCurrentRecordVerificationV3` and
`CompilerExecutionCurrentRecordAttestationV3` arrays. Canonical decoding requires exact nested
verification byte equality and requires the signed attestation to bind the service challenge. The
transport additionally correlates the Begin request identity and opaque reservation identity.

The only final frame is either an empty generic rejection or at most 64 KiB of opaque
application-owned response bytes. Those bytes are hashed and session-bound, but the protocol does
not interpret or authenticate them. No V2 frame is verification evidence by itself, and no frame
grants theorem, load, launch, currentness, key-custody, or application authority.
# V3 Readiness Codec

## Explicit Pathname Readiness V3

V3 is an opt-in bootstrap; no V2 framing or negotiation behavior changes. Ready is exactly
64 bytes: magic `F3WVRD3\\0` (8), little-endian version3 (2), zero reserved6,
nonzero per-connection nonce32, zero reserved16. Begin is a 64-byte prefix with
magic `F3WVBG3\\0` and the same layout/nonce, followed by the canonical V1 request.
It carries the same two payload descriptors. All later challenge/current/terminal frames
remain V2. Unknown versions, noncanonical reserved bytes, stale/cross-connection nonce
echoes and mixed-version packets fail closed. These inert codec values grant no authority;
they alone establish neither nonce freshness nor service authentication.

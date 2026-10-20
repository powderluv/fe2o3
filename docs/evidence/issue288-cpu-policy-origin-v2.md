# Descriptive CPU Policy-Origin V2

Status: source-only candidate on compiler base
`5cc0d45f0e5b6d7bd3820fa177e012e86ee98527`.
Rust formatting and `git diff --check` passed. Compilation, ordinary tests and
the new compile-fail example are **UNRUN** at this source checkpoint.
No source admission, functional proof, native launch or #288 demo is completed.

## Contract

M1 contract `m1-e1-interface-contract-20261008T1635Z.json`, SHA-256
`2ad30c61d1113b69115781d691423afd8e9ade556863f2c5c4df2dc7d099786f`,
assigns the verifier only a descriptive representation. Compiler owns actual
Instance resolution, live binding/custody and the compiler output adapter;
M1 owns the private original-session issuer; runtime owns device/publication
lifecycle. No decoded record can construct any of those owners.

Public types are in `fe2o3_verifier::portable_reference_v1::codec`:

- `ReferenceEnrollmentOriginV1`: invocation SHA-256, native-policy SHA-256,
  policy generation `u64`, and mapping ordinal `u32`.
- `NativeCpuOriginV2`: actual source-registration path or admitted-policy
  description. It is public inert data, not a live admission token.
- `NativeCpuPolicyAssociationV2` and `NativeCpuPolicyInputV2`: policy
  description plus semantic MIR/root, logical name, complete actual kernel and
  reference identities, signatures, effect IR and retained observable writes.
- Scoped `with_encoded_native_cpu_policy_input_v2` and
  `with_decoded_native_cpu_policy_input_v2`; decoded records expose
  `input_v2()` and the content-only `commitment_v2()`.

The origin ordinal must be below 256. This does not prove membership in the
actual request: the live issuer must check it against its retained request.
The complete request still fits the existing 4096-byte invocation environment
limit; this codec does not parse or widen that request. Zero hashes may be
represented as inert data, but cannot satisfy an authenticated live join merely
because they decode.

## Wire Boundary

V1 APIs, magic, domain, registration encoding and bounded subject grammar remain
unchanged. Policy input has no registration field, and never constructs a V1
input with an empty or synthetic registration string.

V2 uses magic `F2CPU2\0\0`, little-endian version 2, zero flags, exact total
length, and pointer-width tag 64. Its SHA-256 domain is
`fe2o3/native-cpu-input/v2\0`. After semantic MIR SHA-256 and root:

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 53 | 1 | Admitted-policy origin tag 1 |
| 54 | 2 | ReferenceEnrollmentV1 domain tag 1, little-endian |
| 56 | 32 | Full canonical V3 invocation SHA-256 |
| 88 | 32 | Native PolicyV3 identity |
| 120 | 8 | Policy generation, little-endian |
| 128 | 4 | Mapping ordinal, little-endian |
| 132 | variable | Existing logical-name, identities, signature and effect grammar |

The bounded encoder/decoder share the subject grammar, not a synthetic V1
envelope. Unknown tags, versions, flags, oversized ordinals and trailing bytes
reject. Loop/helper support is unchanged. Each codec keeps the original work
ledger and storage floor, fallible charged allocation, callback-only ownership,
and cleanup checks on ordinary return and unwind.

## Compatibility Oracle

An independent read-only worker assembled the V1 fixture from the frozen
`ir.rs`, `hash.rs`, `signature.rs`, `data.rs`, wire tags and encoder grammar.
It ran only an in-memory Node calculation on MI300X, not either Rust encoder.

- Legacy IR preimage: 194 bytes; SHA-256
  `5468a10e3a0e89f44c6c7a7d613178ce7af9e472b7b525591ee1a714c95d7875`.
- V1 wire: 702 bytes; SHA-256
  `b520565c839eb05e9e36e8ad4e67d4ae42c8aee5833c3992122d08b778619324`.
- V1 commitment:
  `6a3a9193767c0b7cac6463edc8fcfc2ac850f73c482c1dd8d0974a286866bf8a`.

The new golden test compares the full literal wire, its hash, the legacy IR
digest and commitment, then decodes that wire. It is not baseline execution
evidence until the Rust test actually runs.

## Qualification and Integration

Fourteen new ordinary tests cover the golden, canonical policy roundtrip and
shared replay, cross-version rejection, all origin/association/identity
commitments, ordinal edges, unknown tags, all truncated prefixes with original
and adjusted length headers, trailing bytes, body validation, inert zero
descriptions, exact/one-short resources, retained denials, and foreign-account
or damaged-floor cleanup across callback success, error and unwind.
The existing 30 codec tests remain, yielding an expected 44-test focused roster,
subject to comparison with the actual selected binary.

A fresh ordinary-CPU campaign is required; no previous or interrupted campaign
qualifies this source. The proposed seven stages are production check, build,
bounded metadata-only roster discovery, all codec tests, full verifier tests,
doc tests, and final check. Controls, exact source/tool/binary pins, shared Cargo
lock, CPU 2/3 limits, immutable receipts, process drainage and final audit precede
any acceptance. Proposed tooling is not a test result.

This patch deliberately does not enable V2 in production consumers:
`production_conditional_reference_output_source_v1.rs`,
`compiler_native_conditional_source_proof_v2/root.rs`, and
`conditional_ranked_formulas_v2.rs` still require owner-reviewed integration.
The last two retain their V1 decoder/commitment APIs and must not reinterpret
policy bytes as registrations. Failure at that boundary is not to be bypassed.

The earliest target remains the tutorial-supported default f32/RW fill on
gfx942. Actual source enrollment, default-f32 machine/ABI correspondence,
retained native verification, original runtime custody and device output checks
are separate unsatisfied gates. This codec does not provide external policy
revocation checks, GPU correctness, or a reason to relax any of those gates.

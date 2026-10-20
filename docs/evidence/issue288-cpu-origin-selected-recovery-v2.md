# Explicit CPU Origin Selection in Conditional Recovery

Issue: #288. Status: source candidate only. No Rust qualification, protected
execution, genuine formula import, source-through-F success, native recovery,
GPU execution or completed demo is claimed by this change.

## Contract

`NativeConditionalCpuExpectationV1` is an inert pair of semantic root and an
explicit `NativeConditionalCpuOriginExpectationV1`. The complete expectation
slice is supplied independently of transport in actual source-root and accepted
policy order. Numeric sorting of semantic IDs is not required. Missing, extra,
duplicate and reordered expectations are rejected before root recipe import.

`SourceRegistrationV1` selects only the existing V1 codec.
`ReferenceEnrollmentV1` selects only the policy V2 codec and compares both
digests, generation and mapping ordinal with the external expectation before
exposing the decoded owner to reconstruction. There is no magic-based dispatch,
cross-codec fallback or expectation derived from the incoming CPU bytes.

The expectation is not a Loan, capability or enrollment authenticator. The
original compiler/policy owner must derive and freeze it from genuine selection;
a receiver needs independently authenticated provenance for that selection.
Equal bytes do not establish unchanged original ownership or detect replacement
by an equal-valued object.

## Additive API

- `validate_native_conditional_source_packet_with_cpu_origins_v2`
- `validate_native_conditional_source_through_f_with_cpu_origins_v2`

Both accept an additional borrowed, caller-prepaid expectation slice. Existing
public entrypoints retain registration-only behavior and metering. No source
packet framing or policy-file version is changed.

Each root has one selected decode scope. Its existing source/root/name and
kernel/reference checks use that same owner; the actual lower request callback
then invokes the matching typed strict formula importer. Final replay retains
the existing same-visit forwarding and contract checks. No public raw commitment
argument, synthesized registration input or substitute Request is introduced.

The new final route adds expectation backing to the complete visible input
minimum, with checked addition, rather than checking overlapping minima. It
also charges the selection header. Source and final replay preserve their
different error/refund contracts: safe source refusals may refund temporary
storage, whereas opaque import/final failures remain terminal. The original
budget and first-denial history are retained throughout.

The source-only entry checks an expectation-roster floor, following its existing
caller-prepaid discipline; it does not verify aggregate packet and policy
backing. The added final selection header is not a proof of whole-stack byte
usage for Rust values or callbacks. Those broader claims are outside this change.

## Authored Checks, Not Execution Evidence

Twelve inert unit tests cover mixed ordered expectations, incomplete/duplicate/
reordered/wrong-root rosters, missing/rebound ordinals, exact and one-short
backing/work limits, unchanged legacy selection charges, opposite-codec refusal,
all four origin fields, direct V1 metering/subjects and unpaid public input.
They also reject empty rosters and independently changed policy roots, and
check that a damaged or replaced account overrides a refundable callback error.
A compile-only helper checks the public source and final call shapes. These
fixtures do not construct successful signed receipts, Loans or proof Requests.

Existing source and final tests remain required, followed by the new focused
tests, full verifier suite and documentation checks on the exact joined source.
Genuine same-visit import and final success require the original enrollment
owner and actual protected result; inert fixtures cannot meet those criteria.

## Remaining Integration

Native V5 recovery and exact root-policy-file consumers still need additive
opt-in entrypoints carrying these external expectations through their original
checked storage window. Their existing defaults must remain V1-only. Preserve
exact file/source identities, full root roster, runtime/account checks and the
same final relation join. Qualification of this source tranche alone does not
qualify the enclosing compiler, runtime, service or GPU path.

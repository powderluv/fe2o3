# Issue 288: Live Policy-Origin Formula Adapter V2

## Status

Source-only implementation based on codec commit
`4d63acec618f82b5dd6cfca261d704ac993c26c3`.
No Rust build, unit test, doctest, Verus execution, native-service admission or
GPU execution is claimed by this checkpoint. This does not complete a D1-D11
demo. Codec qualification and consumer qualification are separate campaigns.

The production compiler's source snapshot
`d7f5d3786cce86cdd8a90091d15239d33c69c8d8` expects the two added entrypoints.
That source compatibility is not an executed end-to-end integration.

## Deliverable

- `execute_and_retain_conditional_ranked_formula_policy_v2` takes a genuine
  existing request/runtime and a typed `NativeCpuPolicyInputV2`.
- `RetainedProductionConditionalFormulaV2::with_replayed_policy_request_v2`
  re-encodes that whole typed input before lending the existing execution.
- The input's borrow is independent of the request, runtime, retained owner and
  work ledger. Replay keeps the higher-ranked `FnOnce` execution borrow.
- The existing registration-origin entrypoints and wire codec remain strict
  V1 paths. No fake registration path or V1 envelope is created for a policy.
- A private borrowed correspondence projection shares only the source digest,
  root and exact kernel/reference/replay subject. Each origin selects its own
  complete, domain-separated codec before that projection is used.

The shared execution still performs the same retained reservation, source/CPU
correspondence, preparation, actual protected runtime execution, accepted-policy
check and retained receipt construction. Replay uses the same retained source/
graph/root check, exact CPU commitment comparison, strict signed-receipt
reimport and currentness checks before and after the continuation.

The existing formula domain and retained type are unchanged: the seventh
obligation field already commits to the entire CPU record. Distinct origins,
invocations, policy identities, generations or mapping ordinals therefore
produce different CPU commitments and obligation identities.

## Trust Boundary

`ReferenceEnrollmentOriginV1` is public descriptive data, not an authenticated
enrollment stamp. Encoding, decoding or executing a content-bound formula does
not establish original policy selection or source-origin authority.

The caller must retain and validate the actual original invocation, admitted
native policy, selection membership, session/currentness and source owner.
Outer source/final recovery must explicitly opt into the typed policy path.
No generic raw digest, public proof conversion, origin bypass, launch owner or
relaxed importer is added here.

## Authored Tests

Eight unit tests cover:

- Exact policy reborrow, including origin and pointer-preserved subjects.
- Policy and registration correspondence projections independently.
- Each of four origin fields and three common association fields changing the
  complete commitment and formula binding.
- Cross-origin commitment rejection in both directions.
- A coherent CPU operand change.
- All seven identity fields on each of the kernel and CPU reference.
- Stale effect digests and inconsistent borrowed write claims, while preserving
  the caller's accounting floor and original ledger.

The commitment rejection checks do not execute a real retained-owner replay.

Two compile-only helper functions check the external compiler call shape with
independent lifetimes and a moved, non-`Copy` callback capture. Two compile-fail
doctests reject a registration input in the policy API and an escaping execution
borrow. These are authored assertions, not test results.

## Next Qualification and Integration

First qualify the frozen codec independently. Then use a fresh, pinned consumer
campaign: production check, compile-only test build, metadata-only discovery,
focused policy and existing formula/resource tests, complete verifier suite,
doctests and final production check. Preserve exact discovered full and ignored
rosters, original budgets, failure artifacts and source/tool pins.

A separate follow-up adds typed decoded import/replay, the private same-visit
import/check continuation, and explicitly opted-in source/final recovery.
The actual compiler test hooks must branch on origin or gain a policy-specific
path; existing registration-only helpers must not be silently reused.

Genuine-owner integration must test unchanged replay and changed invocation,
policy, generation, ordinal, source, root and body rejection before consumer
exposure. It must also test callback failure, unwind, stale currentness and
foreign-account behavior using real owner construction, not fabricated proof
or request fixtures.

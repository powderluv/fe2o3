# Issue 288: Decoded Policy-Origin Formula Adapter V2

## Scope and Status

This source-only follow-up is based on live-adapter commit
`414a6ade5751645f518dfd02f710ce047312a56b`. Rust build, unit tests, doctests,
protected execution and native integration are unrun for this source.
It does not complete a D1-D11 demo or inherit qualification from another tree.

## Added Entry Points

- `import_and_retain_conditional_ranked_formula_policy_v2` accepts the actual
  callback-scoped `DecodedNativeCpuPolicyInputV2`, actual signed receipt bytes
  and an independently accepted import policy.
- `with_replayed_decoded_policy_request_v2` lends the existing retained
  execution only after strict replay against that decoded owner's input and
  complete commitment.
- The crate-private
  `import_and_check_conditional_ranked_formula_policy_v2` lends the imported
  execution during the same import visit, preserving the opaque terminal error
  type used by final source recovery.

The existing V1 entrypoints still accept only the V1 decoded owner. A private
typed sum dispatches to the corresponding decoded input/commitment pair.
No public raw-commitment argument, fallback decoder, fabricated registration or
new authority constructor is introduced.

## Preserved Checks

Both origins use the same retained storage reservation, source/CPU correspondence,
formula preparation, externally accepted boundary/toolchain check, metered
signature import, exact imported identity check and source currentness checks.
The accepted policy remains external to the receipt's embedded signing key.

The same-visit check retains the original budget address, work-ledger identity,
storage floor, pre/post continuation checks and provisional-owner destruction.
Its opaque refusal deliberately exposes no generic error source that could
reclassify a terminal callback failure as a refundable import error.

All enclosing codec, lower and source callbacks must finish before the caller
installs a returned retained owner. The origin description remains inert:
agreement with it is not proof of actual enrollment, selection membership,
original invocation custody or live session currentness.

## Authored Coverage

One new inert test checks that a decoded policy owner supplies the exact borrowed
correspondence subjects and the same commitment after canonical re-encoding.
It also checks the caller storage floor and work-ledger identity survive the
nested codec scopes. Together with the prior live tests there are nine authored
policy component tests, not nine genuine-owner replay tests.

Three compile-only helpers check decoded import, replay and same-visit check
call shapes with independent request/work lifetimes and moved callback captures.
Two compile-fail doctests reject a registration-decoded owner in policy import
and an escaping decoded-policy execution borrow. These additions are unrun.

Existing formula import, account, resource and V1 replay tests must be included
when this follow-up is qualified; source review does not replace them.

## Remaining Work

Outer source/final and NativeV5 recovery still require additive, explicitly
opted-in policy-origin routes. An externally supplied ordered per-root origin
expectation must choose the decoder; decoded bytes must not choose their own
accepted origin or fall back to another codec. Actual expected-origin provenance
must come from the original retained invocation/policy owner.

The same-visit final continuation must dispatch to this typed importer without
a second decode, import or replay. Preserve the distinct accounting contracts:
certified-safe early source failures may be refundable, while the final and
NativeV5 continuation routes retain their terminal no-refund behavior.

# E1 Early Enrollment: Compiler Integration Draft

Status at 2026-10-08 17:17 UTC: source-only integration snapshot. No Rust build,
test pass, source admission, native publication, GPU result or milestone exit is
claimed by this document. Formatting and whitespace checks pass. Do not merge
this draft as a qualified production path; the missing APIs below prevent a
standalone build of this snapshot.

The compiler checkout starts from public `5cc0d45f` and contains the descriptive
codec from formal commit `4d63acec`, applied as `e2d8d64d`. The issuer, original
invocation lifetime and publication remain M1-owned. Compiler ownership covers
actual definition resolution, closure custody, source work and the import gate.

## Flow

1. The existing native admission retains its original policy, protected rustc
   invocation and TARGET account. A private scoped loan is borrowed from that
   owner; parsing an environment value cannot construct this loan.
2. The collector receives an optional loan. The legacy entry passes `None`.
   The request is read only from the original protected descriptor's retained
   compile environment. Its complete JSON value is bounded by the existing
   4096-byte transport limit; at most 256 mappings are allowed.
3. Before resolving mappings, the collector retains the issuer's non-Clone
   stamp and a lazily allocated process-local identity on its original SOURCE
   account. Moving that account preserves identity; equal reconstructed
   counters do not. All identity operations and retained payloads are charged.
4. Selectors resolve against actual registered kernel roots and local Rust
   definitions. Generic references need one actual fully monomorphized instance
   in the current codegen-unit inventory. Missing or ambiguous instantiations
   refuse. Existing safe-local-reference, logical ABI and effect extraction are
   reused without a kernel-name, scalar-value or workload special case.
5. Enrollment retains the resolved kernel/reference instances, origin ordinal
   and logical name before collection can replace them. Capture and import
   compare the complete roster and revalidate the live issuer and original
   SOURCE account, including empty rosters. Reference custody also retains a
   borrowed original rustc Session witness; foreign sessions refuse.
6. Import repeats the actual source extraction and complete binding comparison
   before semantic recipe preparation. The old no-loan import cannot consume
   an enrolled closure. M1 must keep the original native owner live while
   borrowing a fresh loan across the admitted-invocation split and consuming
   publication. Weak identity is never authorization or owner resurrection.

## Representation

`ReferenceBindingOriginV1` distinguishes source registration from explicit
reference enrollment. The former retains the exact registration string and V1
wire encoding. The latter carries the descriptive
`codec::ReferenceEnrollmentOriginV1`: original invocation digest, native policy
digest, policy generation and mapping ordinal. No pointer or live stamp is
serialized, and decoding the V2 record does not confer admission.

Enrollment selectors use canonical rustc DefPath components, fully qualified
from the crate name. Named components carry a nonzero `#disambiguator` where
needed; anonymous components use `{namespace#disambiguator}`. Comparison walks
borrowed DefKey/Symbol data without constructing a diagnostic pretty-path
String, temporary path vector or newly interned symbol. Diagnostic aliases and
pretty-printed impl types are not selectors.

## Activation Dependencies

- M1's private original-owner loan implementation and narrow
  `ProductionCompilation` invocation/import split are not supplied by this
  compiler patch. No substitute issuer or caller-provided validator is allowed.
- Formal owns the typed V2 verifier execution, retained replay and outer-root
  binding adapters. The backend draft calls the agreed separate typed policy
  execution/replay APIs, which are not implemented in this checkout yet. The V1
  CPU adapter refuses policy origins; there is no fake registration or V1 hash.
- The new explicit importer entry covers the existing Singleton grammar with
  its nominal option. Other import profiles continue to refuse enrolled
  closures until their original-owner plumbing is explicitly integrated.
- Approved proof/runtime gates, target admission, machine-code generation,
  permitted device ownership and independent output comparison remain required.
  Component tests cannot stand in for the unchanged tutorial source-to-GPU run.

## Test Plan

Written but not executed: four SOURCE identity tests covering moves versus
equal counters, dropped accounts, exact/one-short/zero/inherited budgets and
sticky repeated debits. Thirteen actual-rustc selector and resolver tests cover
qualified/nested/renamed functions, canonical named and anonymous
disambiguators, actual nongeneric Instances, missing/nonlocal definitions,
real unique/ambiguous/absent CGU monomorphizations and cumulative work limits.
They are in a separate test module, without integer stand-ins for Instances.

The policy replay probe is separate from the existing V1 registration probes.
Its opt-in observer must wrap an exact, single-threaded genuine production child
and see one actual retained policy replay. Nine report rows cover borrowed
typed input, V1 input refusal, unchanged retained replay, four independent
origin-field mutations and callback error/unwind storage accounting. No request,
retained proof or issuer is fabricated. These rows are written but unexecuted,
and do not test live issuer/currentness or decoded policy recovery. Existing V1
probe behavior and report schemas are unchanged.

Required after the actual issuer is available:

- Unchanged and renamed local kernel/reference positives; ordinary no-request
  and annotated V1 behavior remain unchanged.
- Equal-descriptor foreign live owner, dropped owner, equal-counter foreign
  SOURCE account and foreign compiler session, including empty rosters.
- Coherently reauthenticated alternate reference before and after capture;
  missing/extra/duplicate mappings, changed ordinal/origin/name/signature.
- Unique generic instance, absent/ambiguous monomorphization, duplicate CGU
  occurrence, nonlocal/unsafe reference, ABI mismatch and annotation conflict.
- Inherited, exact, one-short, zero, overflow and repeated-refusal work;
  failures must not reach the real semantic recipe freeze boundary.
- V1 frozen wire bytes, V2 domain/version and every origin-field mutation,
  independent compiler/backend suites, affected tutorial commands and the
  supported-target end-to-end native path.

Real issuer tests must use M1's admitted path. A test-only fake loan, fabricated
policy origin or lifetime cast does not qualify live-owner custody.

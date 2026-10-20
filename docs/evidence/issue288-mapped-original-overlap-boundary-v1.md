# Mapped Original-Account Boundary Followup

This test-only followup starts from formal source commit
`0b581d59b8f66d64bf485dd9a92cf5fccb33a620`. It leaves production APIs, legacy
entrypoints, error ordering, bills and all frozen qualification trees unchanged.

## Combined Boundary

The new component test invokes the actual private `original_recovery`,
`begin_selected` and `finish_with_origin_working` functions on one original owned
account. Its initial floor exceeds the nested additional-storage cap. Complete
input overlap includes handoff-capacity metadata and mapping expectation backing;
the peak additionally includes window scratch and the full mapped decoder,
context and selection working quote.

The exact total-storage and exact work budgets succeed. A total-storage budget
one byte below that combined peak refuses the mapped working reservation, with
zero downstream content callbacks and zero created/returned inert drop probes.
An additional exact-one-short work case refuses the mapped entry charge at the
same overlap floor. Both negatives retain the original work/storage denial,
already-paid overlap, storage account, ledger, budget address and total limit.
The accounting-window closure itself is entered; that is not mislabeled as a
content callback. Success alone releases the completed temporary overlap and
working storage, returning the probe unreserved for its caller to dispose of.

The drop probe is ordinary test data, not a Handoff, Request, recovered native
owner, proof import, compiler carriage, installed-policy authority or custody
token. This does not replace a genuine authenticated source/native recovery test.

## Sticky Denial Scope

The mapped production path currently rejects sticky denials at projection entry
and after the original callback account/floor checks. It does not promise public
native-entry failfast. Every successful mapped native route reaches that
projection gate before a recovered owner escapes; earlier decoding may still
perform bounded work and retain charges before refusal.

An earlier mapped-only public-entry check could avoid that work, but would change
error precedence and observed charging. It is an optional separately reviewed
contract extension, not necessary for the accepted projection-boundary safety
claim, and is not included in this test-only change.

## Qualification

At the owner's source handoff, this was source-only authored coverage. No Cargo,
rustfmt, Rust compilation/test execution,
solver, runtime, GPU or protected-service run was performed for this followup.
Independent source review and the owner-controlled ordinary CPU qualification
remained required; no execution result was inferred from this test's source.

The later M1 integration at `d3d380baf0c1665e93cc0f74eed491580ea9b512`
compiled and executed this test as part of 1,650 passing nonignored verifier
library tests. The 48 ignored tests were not selected. See the exact source,
artifact and terminal receipts in
[the joined CPU checkpoint](issue272-m1-reference-enrollment-cpu-20261008.md#later-joined-cpu-checks).
This ordinary component result does not qualify genuine native recovery or GPU use.

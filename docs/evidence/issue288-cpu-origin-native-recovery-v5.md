# Origin-Selected Native V5 Content Recovery

Issue: #288. Source candidate only. No Rust build, unit/doc execution, genuine
formula import, original enrollment authentication, protected native service,
GPU launch or completed demo is claimed by this tranche.

## Additive Entry Points

- `recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_v5`
- `recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_in_original_account_v5`
- `recover_native_conditional_handoff_under_policy_file_with_cpu_origins_v1`

Each requires the complete, ordered, independently retained
`&[NativeConditionalCpuExpectationV1]`. Explicit registration rows are required
alongside policy rows. An empty opt-in slice is not the legacy default. Existing
entrypoints remain registration-V1-only; no CPU-byte magic sniffing or fallback
is introduced. These inert values do not authenticate original Loan custody.

The source and final consumers enforce roots and all four policy-origin fields
before the typed CPU owner enters recipe/formula import. Native V5 forwards the
same expectation slice through the original-account or ordinary-account final
adapter. Each adapter retains the existing same-visit `joins::check` callback.
There is no second decoder, proof importer or final graph reconstruction. The
existing final graph allocation moves only after all joins and account checks.

## Accounting

The new strict entry charges 10 work units (legacy: 8). Its visible prepaid
minimum adds expected-slice backing to handoff backing and decode metadata.
The original-account entry adds that slice to its existing complete input
overlap, under the same additional 256 MiB ceiling and original work/storage
account. No cap is raised and no fresh resource account is introduced.

The selection option and extent header are paid in the working reservation and
used consistently by entry, exact replay postcheck and success-only finish.
The policy-file wrapper also prepays its selection header and checks a single
combined minimum: file bytes, handoff backing, metadata and expected slice.
Failures and unwinds retain partial reservations; no blanket refund encloses
recovery. Only the established successful transfer releases known temporary
storage. Existing entrypoints add neither origin work nor origin storage.

These are visible minima and explicit logical working headers, not proofs of
all Rust stack bytes. Callers must retain and prepay actual complete allocation
backing, including spare/enclosing capacity not visible in a borrowed slice.

## Unchanged Byte and Trust Contracts

The policy file remains `F3NCRP1`, singleton gfx942, production history V1 and
boundary 6. Encoder, decoder, reserved fields, checksum, exact source length/
digest and byte-identical embedded roster checks are unchanged. Generic Native
V5 continues to accept its existing independently selected target profile.

CPU expectations are not added to the policy file. Enrollment
`native_policy_sha256` is a separate externally accepted coordinate; this code
does not assert that it equals the root-policy-file SHA-256. Installation,
protected file custody, authenticated enrollment provenance, currentness,
original compiler/session ownership and launch restrictions remain mandatory
external obligations. Returned owners still represent conditional content only.

## Authored Checks and Remaining Evidence

Four native component tests cover combined input payment, exact/one-short work
and storage, selection-header transfer, retained first denials, terminal errors,
damaged floors, foreign ledgers and original-window error/unwind behavior.
Two policy-file component tests cover additive input payment and arithmetic/work
refusals. A compile-only helper covers all three public call shapes. Existing
legacy tests, policy-byte mutations and same-final-allocation tests are retained.
The preceding source/final tranche supplies twelve inert roster/codec negatives.

All new tests are authored, not executed. Qualification must run focused tests,
the full verifier suite and docs on exact integrated source. Genuine same-visit
success additionally needs the original enrollment owners and actual signed
proof results, with one import per root, unchanged native joins and the actual
F allocation. Component fixtures and public call-shape checks cannot provide it.
Full public-wrapper mixed-origin recovery and legacy-wrapper rejection of a
genuine policy leaf remain explicit integration obligations. The lower-level
opposite-codec tests do not establish those outer end-to-end results.

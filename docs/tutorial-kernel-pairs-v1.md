# Tutorial Kernel Pair Obligations

Every intended-runnable tutorial kernel needs an explicit SIMT source and a
structured tile source, with independent per-variant/per-target qualification.
A mixed example is additional. See [issue 275](https://github.com/harsh-nod/fe2o3/issues/275).

Start with [target-specific reproduction plans](tutorial-reproduction.md) for
the existing fixture commands, exact inputs and still-UNRUN correctness stages.

Inspect the current source contracts using the existing production validator:

```sh
python3 scripts/validate-tutorial-kernel-manifest.py --emit-kernel-pairs
```

To include lexical observations from an exported tutorial runtime projection:

```sh
python3 scripts/validate-tutorial-kernel-manifest.py --emit-kernel-pairs \
  --site-inventory /absolute/path/runtime-curriculum.json
```

The optional projection must pass the existing ordered lesson/tab, exact
displayed-byte and source-metadata checks first. It does not have to retain the
old site Git HEAD when its validated display/source projection is unchanged.
Without it, displayed declarations are unknown, not guessed from current files.

## Report Contract

### Ordinary Source Stage Observations

To bind the existing corpus runner's diagnostic report to the projection:

```sh
python3 scripts/validate-tutorial-kernel-manifest.py --emit-kernel-pairs \
  --ordinary-source-report /absolute/path/ordinary-source-corpus.json
```

The report must match the raw manifest bytes and its complete fixture, compiler
input and target roster. Its raw-file SHA-256 differs from the projection's
canonical-JSON `sourceContractSha256`. Reordered cases are allowed; missing,
duplicate, foreign or substituted cases reject before any JSON output.

`ordinarySourceObservations` carries the original report with `diagnosticOnly`
true. `fixtureSelections[].ordinarySourceCaseIndex` refers to that report's
case, not an individually qualified kernel. A multi-root fixture refusal does
not identify which root failed. `stageStatus` becomes
`fixture-source-observations-bound`; without a report it remains `not-evaluated`.

Preserve recorded `callback_progress`, nested Policy4 phases, `refusal` and
progress errors as observations. Missing stages remain unobserved; an active
snapshot records incomplete observation, not a currently running process.
Do not infer internal phase failures from a broader refusal or connect these
cases to unbound tutorial variants by matching names or lesson scope.

This checks input correspondence and outcome consistency, not report provenance
or compiler execution. Even an all-pass source report leaves `qualified` false,
`qualifiedPairCount` zero, and all missing source/variant/evidence bindings
unchanged. It grants no proof, simulator, hardware or launch credit;
`--require-qualified` still refuses. Both input and encoded output are bounded
to 16 MiB. Unknown nested diagnostic data is not independently certified.

### Live Compiler Source Census

Set `FE2O3_DIAGNOSTIC_SOURCE_CENSUS_PATH_V1` to a new file path and
`FE2O3_DIAGNOSTIC_SOURCE_CENSUS_RUN_ID_V1` to a fresh caller-generated
64-digit lowercase hexadecimal ID when invoking the existing
`fe2o3-rustc-extract` wrapper. Consumers must match that expected `runId`,
arguments and extraction mode, not just find a report at the requested path.
The run ID is diagnostic correlation, not execution authentication.
This optional diagnostic observes
the same authenticated collected closure before the production transaction
consumes it. It does not select a different importer or extraction mode.

The `fe2o3-diagnostic-source-census-v1` JSON records the actual driver arguments,
working directory, extraction mode/version, run ID, selected target, canonical function/definition/instance
identities, compiler-assigned roles, and separate definition and identifier
provenance. Original byte offsets account for BOM/CRLF normalization and are
distinct from rustc's normalized offsets. Expansion and callsite anchors remain
separate. Source SHA-256 values cover original bytes checked against rustc's
recorded source hash; diagnostic paths and stable source-file identities alone
are not content identities. An identifier is available only when its source
token matches the compiled definition, including a possible `r#` prefix.
Generated entry wrappers may therefore have an unavailable identifier even
when their definition anchor is known. Their logical names are not substituted
for missing identifier provenance.

Unavailable files, mismatched bytes, missing identifiers and exhausted bounds
remain explicitly unavailable. The census permits at most 512 functions, 128
source-file observations, 4 MiB per source file, 16 MiB total source reads, 64
macro-expansion levels, 1 MiB of argument bytes and 4 MiB of encoded output.
Existing report paths are never overwritten. Paths aliasing the explicit
extraction output or wrapper crate-binding sidecar disable the report.
Protected output symlinks also disable recording, including dangling links.

`extractionSucceeded` describes only that extraction invocation. A selected
closure may still fail semantic import or a later check; observation preserves
that failure. A rustc fatal error before collection yields unavailable selection
and is re-raised unchanged. A process crash or interruption can leave no
complete report. Diagnostic
errors do not change the extraction result. Empty or incomplete files are not
reports.

This census is not a compiler-execution receipt, proof, executable artifact or
qualification. `diagnosticOnly` is true; `qualified` and
`authenticatesCompilerExecution` are false. The ordinary-source corpus adapter
below joins Cargo/lock/source-closure/default-feature contracts around its actual
callback. Generated-source body maps remain separate obligations. Neither path
changes any pending display binding or kernel count.

### Same-Run Corpus Joins

The existing ordinary-source P4 corpus runner now retains `source_census` and
`source_census_error` per case. Before Cargo capture and after the actual callback,
it runs this validator's `--emit-source-input-snapshot` mode against the exact
registered fixture or source-driver invocation. Snapshots retain the raw manifest
digest, validated package/lock/default/direct-feature contract, expanded feature
set, and all physical package source hashes and lengths. Inputs changing across
the invocation, an incomplete census, or a mismatched root roster block that case.
The runner generates its nonce internally and uses the existing create-only
recorder. The test callback requires terminal census completion when requested;
normal invocations without a requested census keep their existing behavior.

Consume the fixture corpus with the existing `--ordinary-source-report` option
and the additional source-driver corpus with `--source-driver-report`, alongside
`--emit-kernel-pairs --site-inventory`. Each supplied report must contain its whole
registered invocation roster. New census-bearing reports require the live runtime
inventory; legacy reports without census fields retain their diagnostic behavior.
`sourceCensusObservations` is added to the existing projection after the source,
runtime, and identity validators succeed. The mandatory matrix script-test gate
runs the join and CLI regression suite.

The adapter checks before/after/current input equality, independently retained
arguments/cwd/nonce against the census, fixed P4 mode, target, Cargo feature cfgs,
complete selected roots, physical hashes, and original/normalized identifier
coordinates. Runtime joins use exact whole-file or unique fragment bytes, never a
name-only match. Existing explicit variant bindings are replayed; pending variant
implementations and unavailable/expanded root identifiers remain precisely
unresolved. The adapter cannot choose SIMT, tile, or mixed implementations from
names or code shape. Negative drivers remain negative obligations.
An early aggregate limit of 4096 derived rows counts joins, origins, repeated
runtime/variant occurrences, and unresolved obligations before retaining them;
the existing 16 MiB projection encoding bound remains an additional limit.

This is legacy P4 diagnostic provenance, not expanded mixed-production compilation.
The retained census digest detects modifications after retention, not forgery or
execution authenticity. A nonce correlates the callback with its runner, not a
signed timestamp. No qualification, source-bound flags, classifications, accepted
compiler pins, source receipts, or artifact/launch authority are advanced.

### Inventory Projection

The JSON schema is `fe2o3-tutorial-kernel-pair-obligations-v2` when the manifest
contains its optional `kernelInventory` extension. Legacy manifests without
that extension retain `fe2o3-tutorial-kernel-pair-obligations-v1`. This is a derived
diagnostic view of `config/tutorial-kernel-manifest-v1.json`, not another stored
inventory, execution receipt, release gate or source of compiler authority.
The existing V2 curriculum and source-item digest domain remain unchanged. The command validates all source
contracts before producing any stdout; it cannot be combined with `--emit-matrix`.

- `fixtureSelections` preserves each fixture/symbol/target input selection and
  its existing contract digest. Complete feature, Cargo target and source-path
  identity is retained. Scope lesson links are not exact displayed bindings.
- `sourceDriverCases` preserves exact lesson/tab/case identity, source contract,
  driver and expectation. A required rejection stays a rejection, not a working
  implementation or an excuse to downgrade a required positive.
- `displayObservations` covers executable Rust kernel tabs. Optional lexical
  names preserve order and duplicates, including cfg alternatives. They do not
  establish rustc feature reachability, selected kernels or runnable behavior.
- `lessonRequirements` retains SIMT/tile and any mixed lesson obligations;
  `sourceBindingGaps` retains the existing unresolved source paths.
- `sourceContractSha256` hashes the whole input manifest encoded as sorted-key,
  compact, ASCII JSON. This content identity is not a signature or qualification.

Do not add these collections to obtain a kernel count. They can overlap, repeat
symbols under different features, and omit still-unbound kernel identities.
For the current incomplete inventory, `requiredPairCount` is null,
`qualifiedPairCount` is zero, and `inventoryComplete` and `qualified` are false.
The current projection records 61 known kernel identities and 123 variant
obligations: 22 variants have exact source associations, 101 remain pending,
and no pair is source-bound. Five displayed declarations lack exact
source/display bindings; every existing positive selection has a display binding.
Stages are not evaluated unless a
diagnostic source report is bound; binding that report does not qualify a pair.
Existing historical successes do not
acquire new source/variant/target bindings from this report.

The reductions-and-scans lesson now presents one exact current Wave64 kernel.
Its former duplicate historical source remains archived with its original
source-model identity and immutable link, rather than as a second active
instructional implementation. All 61 kernel identities and 123 mode
obligations are retained; this display migration qualifies no execution.

The LDS/barriers and MoE-routing lessons likewise present their exact current
source as the primary Kernel tab, without a duplicate current-source tab.
Their superseded historical sources and claims retain their original immutable
identities outside the active instructional tabs. The six removed display
declarations were two unbound historical entries and four ordinary helpers;
no kernel record, source binding, variant obligation or negative case is removed.

The first-fill kernel display binds the 552-byte library file at
`b9378bdbee0dd1e284a1c31b25c3614c5c2039fe`, byte-identical at reviewed
compiler `188f388b`. The unchanged CPU point reference is in the same-module
`reference.rs` companion; the whole-file source digest covers the library, not
a proof or execution result. Recorded no-GPU execution of the
historical 308-byte library at `7a536e0a001202ac0bb9d8647c5395661f8fa1ec`
keeps that historical identity; the current source association supplies no new
execution or tile-pair qualification.

The MoE routing lesson's current-source tab binds the existing top-2 fixture
through its full physical module closure, including ordinary enum/struct
declarations with checked builtin derives. This source selection is not a rustc,
simulation or GPU result, and it does not bind the archived older revision.

The CPU simulation lesson's whole-file tab 6 binds
`row_affine_sum_u32_v1` from
`examples/workgroup_sync_v1/src/kernel_row_affine_u32.rs` through the existing
source-driver contract, with `row-affine-u32-kernel` and default features disabled.
The associated production test exports Bundle V5 and compares 86 independent
oracle cases on the gfx942 and gfx950 CPU profiles, including execution and
exact persisted replay. This is a source-bound SIMT variant, not a
new native fixture or a completed pair. Tile and mixed implementations,
native artifact/generated-host admission, and direct-KFD GPU validation remain
pending for gfx942/mi300x and gfx950/mi350. The inventory does not ingest these
CPU test results as per-variant target qualification receipts.

The report is bounded to 4096 charged projection/inventory records, including
input tabs, references and scanned items, and 16 MiB of encoded JSON. Each Rust
scan is limited to 4 MiB and 4096 declarations. Exhaustion rejects without partial JSON.
Physical selection work has separate aggregate caps of 8192 source-item visits
and 8192 attribute visits. The complete 95-fixture catalog exceeds the former
4096 attribute-work capacity; the record, byte, depth and source-item bounds are
unchanged. These are traversal limits, not qualification or native-memory limits.
The tests run in the existing `scripts/tests/kernel-compile-matrix.sh` CI gate.

## Kernel Identity Extension

The sibling `kernelInventory` in the existing manifest is an explicit
`fe2o3-tutorial-kernel-identities-v1` contract. It does not replace or duplicate
the fixture inputs, source-driver contracts or curriculum source metadata.

- `kernels` assigns stable IDs to exact positive fixture/symbol selections or
  source-driver cases. Each ID has SIMT and tile variant records with a blocker
  owner, issue and reason. Source associations do not qualify execution.
  Matching names, source paths or historical displayed
  bytes alone cannot establish identity.
- `negativeCases` retains the exact required-refusal source-driver cases outside
  the positive pair roster.
- `displayItems` classifies physical function occurrences, not attribute hits.
  The coordinate is lesson ID, tab ordinal and function-name UTF-8 byte offset.
  Repeated kernel attributes on one function produce one occurrence; distinct
  declarations with the same name remain distinct. Raw-identifier offsets
  include the `r#` prefix.
- The census covers attributed functions in every Rust tab kind, including
  host and comparison tabs, plus ordinary functions in Rust kernel tabs.
  Bare kernel excerpts remain kernels; helper functions, conceptual examples
  and required negatives have explicit separate classifications. Bare-function
  intent is reviewed inventory data, not inferred from lexical syntax.
- A `source-driver-contract` binding must match an existing exact source case
  and its displayed fragment.
- A `fixture-source-contract` is an expected fixture/source contract, not
  rustc-selected execution or semantic admission. It joins an existing
  feature-specific fixture identity to one attributed function occurrence in an
  exact whole-file or excerpted kernel display. Every excerpt must have a unique
  complete-byte occurrence in the selected physical source, and fragments cannot
  overlap or repeat. Each displayed name maps to its own physical UTF-8 offset;
  matching a symbol name alone is insufficient. The validator rechecks the physical package
  closure, Cargo manifest/lock, selected source member, file/display digests and
  UTF-8 name offset. Package membership alone is insufficient.
  Selection uses the existing lexer/scanner with a restricted check of ordinary
  sibling modules and literal `feature`/`target_arch`/`test` cfg expressions
  (`all`, `any`, `not`) in the declared non-test AMDGPU library context.
  Here `test` is false, including for the inner source library selected by a
  host-side test driver. This is an expected source contract, not authentication
  of a rustc invocation or inference that every Cargo library build is non-test.
  Ordinary brace-delimited `macro_rules! name { ... }` definitions are inert
  textual items: their bodies do not contribute functions or modules and are
  not expanded or checked as Rust. Function-body macro calls remain opaque;
  they may invoke other macros or reference helpers. The complete source/Cargo
  closure remains identity-bearing, but neither resolved expansion dependencies
  nor a compiler-expanded entry roster is inferred. Kernel attributes on macro
  definitions reject. Exact unqualified literal `include!("path.rs");` calls
  may select files containing only inert macro definitions and further literal
  includes from the authenticated package source closure. Ordinary unescaped
  and raw string literals are accepted; paths are canonical and relative to the
  physical containing file, without changing its Rust module context. Included
  functions, modules, imports and inner attributes reject. Missing, repeated,
  cyclic, generated or noncanonical include paths reject. Across the selected
  closure, a definition named `include`, duplicate active macro names, and
  imports containing `include`, aliases or globs conservatively reject include
  selection, including parent-scope shadowing. This is not Rust name resolution.
  Includes share the existing record/source-byte budgets and allow at most
  32 nested include edges. No included macro token becomes a selected function.
  Unknown cfg, other item macro calls, transforming attributes,
  build-script cfg, dependency features, nested/inline/path-selected modules,
  duplicate modules or selected symbols reject a claimed binding. Such sources
  retain `pending` until an exact selection can be established. This bounded
  check does not replace rustc or qualify kernel implementations.
  Supported non-cfg forms are bare `kernel`/`inline`, parenthesized
  `allow`/`deny`/`forbid`/`warn`/`doc`/`kernel`, and `inline(always)`/`inline(never)`.
  The literal inner attribute `#![no_std]` and the existing
  `#![cfg_attr(target_arch = "amdgpu", no_std)]` form preserve source selection.
  Outer or argument-bearing `no_std` forms remain unsupported.
  Qualified attribute paths and name-value forms such as `#[doc = "..."]` are
  unsupported and reject claimed bindings.
  Conditional `cfg_attr` may apply supported attributes, including nested
  `cfg_attr` and conjunctive `cfg`, with zero or multiple attributes and a
  trailing comma. Predicates and attributes are validated even in inactive
  branches. Effective kernel attribution is evaluated separately from the
  conservative display census; inactive conditional kernels do not bind.
  Multiple active kernel attributes and inner kernel attributes reject.
  Attribute bodies are limited to 8 KiB, nesting to 32 levels and each
  `cfg_attr` to 64 child attributes. All traversals share one aggregate
  attribute-visit quota, equal in size to but separate from the existing record
  quota; inactive attributes are charged too, and cached selections are not
  revisited. Each cfg predicate admits at most 512 tokens, including punctuation;
  its byte and nesting limits remain independent. This accommodates ordinary
  mutual-exclusion feature predicates, including the existing 318-token
  advanced-attention predicate, without changing the record, source-byte or
  aggregate attribute limits. Unsupported predicates and active item macros
  still reject; accepting a library's cfg does not bind its kernel source or
  qualify a variant.

The fill fixture's SIMT variant is associated with the exact current
`examples/fill/src/lib.rs` function and its registered compiler-input identity.
The source closure and selection pins include its explicit CPU reference and
focused reference tests. Its tile variant and execution evidence remain pending.
This association does not complete a pair or the curriculum census, or rebind
the historical first-fill display to the changed file.

Eight GPT-OSS occurrences in tabs 1-6 retain these expected fixture/source
contracts: serial-router, held-fragments, interleaved-stores, and the three
materialized components, plus pipelined-attention and scalar-attention. Their
existing source/feature identities remain distinct, including the repeated
megakernel symbol. The main megakernel and performance-lab tab-zero excerpts
also bind to their exact selected physical source. Eight FP4/FP8 GEMM and
attention displays, including their performance-lab excerpts, also bind to the
registered feature-selected source. The declared runners use non-test
`cargo check --lib`; test-only declarations are excluded from this selection.
Together with fourteen systems occurrences and the four whole-file
displays described below, these associations preserve distinct source
identities. The historical GEMM lesson's source gap remains unresolved.
The two attention variants are required
positive compile obligations with separate pending-design simulation requests;
neither inherits another variant's retained KIR or execution evidence. These
registrations, together with the explicit SIMT row source above, bring known
kernel identities to 61, not completed pairs or a proven final curriculum
denominator.
The systems occurrences cover routing, expert computation, expert combination,
gradient staging, Muon update, n-gram gather and speculative verification,
including repeated performance excerpts. They bind eleven existing
feature-specific identities, not eleven completed pairs. No source bytes,
historical performance evidence or execution qualification are changed.
The full-file flash-attention, GEMM autoresearch winner, tiled GEMM and grouped
expert MoE displays bind their existing default-feature fixture identities.
Their exact physical function offsets distinguish the four kernel entries
from nine displayed helpers, including inactive host tests. All helper rows,
historical source pins, kernel bytes and pending variant obligations remain
unchanged. Whole-file byte checks and the registered Cargo/module selection
establish these display associations, not new compiler or execution results.
The scalar-GEMM simulation request and independent expectation are checked in and
registered in generic CI. Their status remains `pending-reconciliation` until
the required source-produced Bundle V7/KIR V12 simulation and oracle checks run.
Comparator tests alone do not execute or qualify either kernel variant.
Other source/display joins remain `pending`; a fixture's lesson scope is not
an exact displayed-source binding.

The typed-vecadd and LDS/barriers lessons each add a current whole-file entry
tab bound to its existing feature-selected fixture. Their earlier historical
tabs retain their original bytes and pending bindings; a current entry does not
authenticate an older wrapper or helper closure. These source associations add
neither execution receipts nor tile implementations. The displayed mixed-tile
CPU source remains pinned to a published WIP revision, not compiler main.

The V2 report's `kernelInventory` projects these records and reports known
identity, display, negative-case and pending-join counts separately.
`runtimeCensusValidated` requires the actual runtime projection and complete
occurrence reconciliation. Without it, retained offsets are declarations to
check, not observed source facts. Supplying `--site-inventory` checks this
census even when neither report nor matrix output is requested.

`inventoryComplete` additionally requires all required source/display joins,
including coverage of the positive selections.
Only then may `requiredPairCount` name a reconciled denominator. Neither flag
qualifies implementations: pending variants still require their own sources,
oracles and every production/target gate. The current known source selections
must not be presented as the final number of tutorial pairs.

This is a bounded lexical inventory, not a general Rust parser or compiler
feature-reachability proof. The runtime census excludes comments, literals and
`macro_rules!` template bodies and conservatively retains conditional kernel
attributes. Direct fixture selection additionally evaluates only the restricted
literal cfg and module forms described above against the retained Cargo feature
contract. Rust compilation remains the authority for feature reachability and
executable semantics.

### Source-Bound Variants

A variant may change from `status: "pending", source: null` to
`status: "source-bound"` with this source object (digest values are placeholders):

```json
{
  "implementationKernelId": "registered-implementation-id",
  "selection": {
    "kind": "fixture",
    "fixtureId": "registered-fixture-id",
    "kernelSymbol": "selected_kernel"
  },
  "selectionSha256": "<64 lowercase hexadecimal digits>",
  "sourcePath": "examples/package/src/kernel.rs",
  "sourceSha256": "<64 lowercase hexadecimal digits>",
  "functionUtf8Offset": 123
}
```

The variant retains `kind` and its blocker owner, issue and reason. `selection`
may instead use an existing `source-driver-case` reference with its exact
`lessonId`, `tabOrdinal` and `caseOrdinal`. It must belong to the declared
implementation ID. That ID can differ from the obligation's `kernelId`, allowing
different source/feature implementations without merging their identities.
The report exposes each registered identity's `selectionSha256`, covering its
package/lock/source-closure digests, Cargo target, features and kernel symbol.
The physical file digest and function-name UTF-8 byte offset must also match the
actual selected source. Both reference kinds use the restricted physical
Cargo/cfg/module validation described above; unsupported syntax stays pending.

This is a declared source association, not verification of SIMT/tile mode,
semantic equivalence or execution. Even two associations to the same source
provide none of those guarantees. `sourceBoundVariantCount` and
`sourceBoundPairCount` count authenticated associations; `variantBindingStatus`
is `pending`, `partial` or `source-bound`. These counts do not establish the final
curriculum denominator or qualify pairs. Complete source binding removes only
`per-kernel-variant-sources` from the report's missing bindings. Numerical
contracts and independent oracles, verified artifacts, target-matched execution
and other production evidence remain required; `per-variant-target-evidence`
remains, and `qualified` stays false with zero qualified pairs. Stages are
`not-evaluated` unless a diagnostic source report is bound. The separate runtime census and source/display join rules for
`requiredPairCount` remain unchanged.

The V2 report also derives `knownVariantObligationCount` from registered
SIMT, tile and optional mixed variants. `pendingVariantCount` counts those
without a source binding, not the number awaiting execution qualification.
`unregisteredDisplayItemCount` counts positive kernel display occurrences
without an identity; it excludes helpers and required negatives, and differs
from `pendingDisplayItemCount`, which also includes stale registered bindings.
Repeated display occurrences and identical source selections across targets do
not by themselves add identities. Distinct feature-selected or source-closure
identities retain separate obligations, even when their kernel names match.
Unregistered occurrences are not a count of additional distinct kernels.
When `runtimeCensusValidated` is false these are declared inventory counts,
not a validated live census. None of these fields establishes an exhaustive
denominator or changes qualification. Legacy V1 reports omit these fields.

## Remaining Integration

Resolve the remaining exact source/display joins and coordinate the website
consumer before treating the inventory as exhaustive. Each pair must
bind both real sources, one functional/numerical contract and independent oracle,
then retain compile, verification/artifact, CPU simulation, generated-host and
target-matched KFD evidence. Until those bindings and gates exist, this report
must not be promoted to completed-pair or all-tutorial qualification.

# M1 Reference Enrollment: CPU Checkpoint

For the later joined intake and denial controls, see the
[October 9 CPU checkpoint](issue272-m1-joined-source-cpu-20261009.md).

2026-10-08. Ordinary developer validation only, not protected compiler admission,
proof execution, GPU execution, M1 completion, or 47-kernel qualification.

## Initial Tested Source

- Commit: `a10c4b0c2711d70c6bca9304f76dc1c85d760a1d`.
- Tree: `73b92a938a9e93c2645cc2647f2d322e9d54a42c`.
- Pinned toolchain: `nightly-2026-04-03-x86_64-unknown-linux-gnu`.
- Host: MI350; CPU affinity 20/21, nice 10, 64 GiB aggregate memory ceiling,
  swap disabled, 256 tasks, one test thread, bounded systemd units.
- Source snapshots before and after each successful stage matched
  `f597431c74516ef8908eeb4a17e1dbf396ea837f2b8e56f904c3d5fa76f66d8c`.
  This is the recorded tracked-file mode/type/content snapshot, not a sealed
  whole-host or continuous isolation claim.

The joined source includes original-invocation enrollment loans and identity
checks, actual rustc source resolution/custody, original CPU-origin projection,
strict explicit-origin codecs, and the shared inert enrollment-request decoder.

## Initial Results

| Stage | Result |
| --- | --- |
| Default test-binary build, both libraries | PASS, 1.10 seconds, cached artifacts, no CLI profile override |
| All `fe2o3-rustc-invocation` library tests | 108 passed, 0 failed, 0 ignored |
| Focused `rustc-codegen-fe2o3` tests | 129 passed, 0 failed, 0 ignored, 4629 filtered |
| Non-test production-library check, both libraries | PASS, 45.99 seconds |

The compiler selection covers enrollment resolution, source custody, cumulative
work/storage boundaries, CPU-origin projection, ten isolated admission/loan flows,
native ownership refusals, source-reference obligations, and CPU effect replay.
Exact discovered names and selections are retained with the run. The other 4629
compiler tests were not executed in this checkpoint.

The build and check commands were:

```sh
cargo test --locked --offline -p rustc-codegen-fe2o3 -p fe2o3-rustc-invocation --lib --no-run --message-format=json-render-diagnostics
cargo check --locked --offline -p rustc-codegen-fe2o3 -p fe2o3-rustc-invocation --lib --message-format=json-render-diagnostics
```

The actual Cargo-produced binaries were then executed with exact discovered test
names, `--exact --test-threads=1 --format=json -Z unstable-options`. No extra
test-build profile override was supplied. Cargo's committed test profile optimizes
only `sha2@0.11.0`; debug assertions and overflow checks remain enabled.
Production profiles and the 60-second fixture / 120-second native deadlines
were not changed.

Here "default" means the committed profile without a CLI optimization override,
not a pristine Cargo environment: the builds retained `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`, and `CARGO_BUILD_JOBS=1`.
The ordinary production check used the dev profile, where SHA remained opt-level 0;
it does not prove an unoptimized production backend meets admission deadlines.

## Preserved Failures

The preceding `60c2e5bc52e10e8cefe4c5d077d0cef78d76681e` run completed with
114 passes and 9 failures: three generic-CGU fixture failures and six child
timeouts. The original logs, terminal receipt, and executable were preserved.

The generic fixture requested metadata-only output and found no concrete generic
instances. It now requests object generation but still stops after analysis;
an assertion checks that no object was emitted.

Timeout observations showed sustained CPU use and repeated full-image reads.
The longest flow performs 30 fresh image checks of a roughly 533 MB test binary.
A separately recorded test-only SHA optimization experiment passed all focused
cases without extending deadlines or caching/skipping verification. Its profile
setting was then committed and the normal build/tests repeated successfully.
This is not a GPU or compiler performance benchmark, nor a recovered stack trace
proving the precise location of every earlier timeout.

Two existing native self-exec fixtures print six plaintext child-report lines.
The initial JSON-only result parser rejected that mixed stream. The supplemental
readback retains those lines and all 123 original parent results. Later reports
explicitly validate the known child diagnostics and exact parent test-name sets;
no failing run was relabeled as passing.

The immutable supplemental report's explanation says it ignores only blank lines.
That wording is inaccurate: its separately retained `bad` list also contains the
six human-format child-report lines. Its parent results and failed-suite counts
were independently checked against the original raw stream.

## Evidence

Remote evidence root:
`mi350:/home/harmenon/fe2o3-m1-integration-20261007.WWdApdE1/evidence/`.

| Receipt under that root | SHA256 |
| --- | --- |
| `r1903-focused-tests-20261008-1912/terminal.json` | `7d6f4e7d9cae05533234b12fcbe1730b8c4c7a686f6bed5a25781bfaaa1b9e11` |
| `r1903-focused-tests-20261008-1912/result-readback.json` | `b4cfe60a348613d529177f12d8d468f11ddd8f7ed2bff84faf5f9aee69d83cbc` |
| `r1908-default-profile-testbuild-20261008-1935/terminal.json` | `8b10cf207b65782814ff82a4745a3282a6e9d91e67052f7f32e34f15d3490d88` |
| `r1909-default-focused-20261008-1936/fe2o3_rustc_invocation/terminal.json` | `037db1c5177e965639d47962f5fbf2079520f8b4aa16d7282779fa03cb55b986` |
| `r1909-default-focused-20261008-1936/rustc_codegen_fe2o3/terminal.json` | `8f253d0de08252e1b4bc4e5a3586e48eb897abe744c368f8fdb9180da019b410` |
| `r1910-production-check-20261008-1938/terminal.json` | `23307e055714da82439cb2c6b5fc6512c0b5d6b85882d4da3907e233b40c0ea7` |

Actual compiler test executable SHA256:
`379c7062201a9215979eb071151cb24f11afae960f685a7418d4ffa3ccf44b5e`.
Invocation test executable SHA256:
`71283c04e6a57b5047ae21f32be5ef359fa4bf17e0c77685fa9bc2d87bb325e2`.
Default builds reproduced the already-discovered binaries byte for byte.
CLI exits, observed process/cgroup absence, boot identity, resource checks,
artifact pins, and original source snapshots are retained in the receipts.

## Remaining Gates

The isolated loan fixtures measure their real test executable, not an independently
installed production compiler/backend. Their process-consistency successes do not
establish protected compiler provenance or successful proof publication.

The compiler capture, host intake, and mapped-consumer changes must still be
qualified together through the actual signed handoff. Current host and worker
call sites retain legacy recovery; neither selects the mapped path nor calls the
shared JSON decoder yet. Any future mapping expectation must obtain its binding
count from the original authenticated descriptor, not the mapping header itself.

Protected request parsing still needs complete input/header/error-lifetime
accounting, retained-output charge transfer, and an enforceable relation to the
qualified actual decoder execution. Conditional serde allocation analysis is not
installed-profile admission or a universal guarantee for external Cargo builds.
Existing generic callback/error types and compiler diagnostic String conversion
must not disappear from that accounting.

Genuine signed cross-process tests, broader joined compiler compatibility, the
protected default-fill path, KernelContext typed-global vecadd, and supported GPU
runs remain open. No result here completes M1 or the 47-kernel end-to-end goal.

## Later Joined CPU Checks

The initial checkpoint above is preserved. Subsequent ordinary developer runs
used the same toolchain, CPU/resource limits, committed test profile, and
source/artifact checks. These results belong to the listed revisions, not to an
unqualified combined total.

| Revision | Change And Actual Result |
| --- | --- |
| `5661e49e9001b2b204ef559f00080f43a1d61c65` | Route-guard correction; 113 invocation and 133 focused compiler tests passed; broader compiler selection passed 1,031 tests |
| `6849ca6ea395c9b8e26f1bbc7efc93e9832fec0d` | Original worker account and raw Subject join; four-library check passed; 6 worker, 24 Subject and 116 lineage tests passed |
| `33f7e3755d9ed2642c5f9e1fb5f51c62597711bc` | Formal CPU mapping joined; five-library check passed; all 1,649 nonignored verifier library tests passed |
| `522d4aea1fa41aa8fb07dec5d0ea9b64daead082` | One-pass enrollment decoder; all 120 invocation and 133 focused compiler tests passed |
| `d3d380baf0c1665e93cc0f74eed491580ea9b512` | Added combined mapping resource-boundary test; six-library production check passed; all 1,650 nonignored verifier library tests passed |

The latest verifier suite discovered 1,698 tests. Its 48 ignored tests were
explicitly excluded and are not qualified. The 17 mapping component tests,
including the new exact/storage-one-short/work-one-short case, are included in
the 1,650 passes, not additional coverage. Likewise, the earlier focused compiler
and mapping runs overlap their broader selections.

The latest production check covers `rustc-codegen-fe2o3`,
`fe2o3-rustc-invocation`, `fe2o3-compiler-lineage`,
`fe2o3-artifact-transaction`, `fe2o3-proof-custodian`, and `fe2o3-verifier`.
It passed in 66.67 seconds. The final verifier run passed in 89.32 seconds.
The 522d focused compiler run passed in 44.72 seconds with 4,627 tests filtered;
it explicitly validated the two existing self-exec child reports.

### Implemented Boundaries

The proof controller now borrows one owned resource account throughout its
lifetime. It joins the full raw Subject to the signed carriage before semantic
recovery and repeats the content join afterward. Existing currentness, deadline,
acknowledgment and quarantine rules remain unchanged. Its new ordinary tests do
not establish genuine signed cross-process execution.

Formal mapping recovery projects the authenticated inventory onto reconstructed
source and checks complete kernel membership, root order, names, bindings, and
both CPU kernel/reference identities before formula construction. Expected
coordinates remain inert inputs requiring independent authenticated provenance.
Mapped sticky-denial checks occur at projection entry and after consumption,
not at the public entry before all earlier decoding. The added boundary test
exercises real accounting primitives with an inert drop probe, not a recovered
native owner.

The shared decoder replaces the typed-plus-generic-Value double parse with one
fixed-schema pass. It rejects positional objects and duplicate/unknown fields,
uses one bounded output vector and fallible selector allocations, finishes JSON
validation before semantic refusals, and preserves strict u16 and error ordering.
Seven new regressions cover those behaviors. The public decoder remains
work-metered, not storage-prepaid; serde error allocations are not guaranteed to
return an allocation error.

### Additional Preserved Failures

The earlier broader run at a10c completed with 1,028 passes and one failed
production-route AST guard. The guard assumed five checks remained inline;
production had moved them into the enrollment-aware helper. The test-only
correction follows that delegation and checks the helper's complete ordered
stages. New negatives first validate the unmodified baseline and require their
specific refusal. The original failed receipt was not rewritten.

The first worker test build, r1918, failed before Rust compilation because the
offline cache lacked locked dependency `fastrand 2.5.0`. A separate r1919
attempt enabled the locked download and passed. Source and lockfile were
unchanged; the failed attempt remains preserved.

### Followup Evidence

Paths remain relative to the evidence root above.

| Receipt | SHA256 |
| --- | --- |
| `r1911-broader-regressions-20261008-1945/rustc_codegen_fe2o3/terminal.json` | `1d8cfee881c7e00712d795faea94f38abb19ac4b2bfe1baa155c809114b83d9f` |
| `r1915-broader-regressions-20261008-2011/rustc_codegen_fe2o3/terminal.json` | `855f4982470607f4fdffb64f301a17f97f6e96630ee7889213fa10fce6629480` |
| `r1922-worker-focused-20261008-2039/fe2o3_proof_custodian/terminal.json` | `a0889dc8036aa5e5ca9d13e026de50f9ec119042168840284738569c7dbd820b` |
| `r1922-worker-focused-20261008-2039/fe2o3_artifact_transaction/terminal.json` | `5aa8232004aa2190e3234422aaf1515bda7e1cd06ff2d6fe46306e5569d10f6c` |
| `r1922-worker-focused-20261008-2039/fe2o3_compiler_lineage/terminal.json` | `c82a816248c9d3fd3b314d8d0ee9f7d11f7cb065a51db63ba4dcd687b276bd45` |
| `r1932-closed-seed-focused-20261008-2101/fe2o3_rustc_invocation/terminal.json` | `cf2b2091c44f3370df2a158eae587dd7da95a6bca59284322f29c5ce7f98c6c0` |
| `r1932-closed-seed-focused-20261008-2101/rustc_codegen_fe2o3/terminal.json` | `38d1bb2c7b8a886281c489bc84aa5f7cde8a9c78cc570d1fbed941eb8a174623` |
| `r1933-joined-parser-check-20261008-2103/terminal.json` | `7c5b2c18b10be55917a6167792ed4f95c6cf9c02d433cdbe500b8477465badab` |
| `r1936-mapped-boundary-regressions-20261008-2109/fe2o3_verifier/terminal.json` | `2796dd7d4f6b5836486d7a7145579c4d5d81dfad609513094f7ca2bc45ac56b7` |

Latest tested production tree: `15180f3d341993978ef193cc6ad1876dea403f10`;
tracked-file snapshot:
`23eaf088e3d7ecaa5731d0db4e25a8898acc0337ce00b0926c4f71d58b4672d1`.
Verifier executable:
`575ce47e789629a97bf933993f417fb5874f9d5e3634221d8944a2d00e89aba6`.
The r1932 compiler/invocation and r1936 verifier executables are preserved beside
their receipts. All reported successful units exited with original process and
cgroup absence checked. These are point-in-time observations, not a whole-host
isolation guarantee.

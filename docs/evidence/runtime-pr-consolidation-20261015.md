# Runtime PR Consolidation - 2026-10-15

Integration base: harsh-nod main 8c27e0029db793a56974ca8974df11fa678fd39e.
All observations below use fetched exact PR heads, not their stale mergeability
labels. No superseded PR is closed by this candidate.

| PR | Observed head | Disposition |
| --- | --- | --- |
| 294 | 7b452cf60e18c78dc9ddda0895caaa262a2b2d27 | Retain current-main DATA-copy/caller behavior; do not restore the older snapshot. |
| 295 | e516fdda72ac7e4b9ea906bff9ed567b841a8f00 | Retain current-main bounded registry and DATA-copy integration; old branch is not the integration base. |
| 296 | 3a2f7e8a9990d872dd6faea9c87d8cad101cddd5 | Retain current-main private argument-error module and CI resource topology guards. |
| 297 | cb562dd76eebe89b49fb412a67741c7add662cf8 | Runtime/KFD source directories are byte-identical to current main; retain current-main cached completion/writer lookup implementation and proof tooling. |

Current main already contains runtime integration be318c355, as well as newer
compiler enrollment, verifier and simulator work. Replacing its tree with any
historical PR snapshot would discard independent maintained work.

For PR297, the remaining cached-poll support manifest difference points at older
pipeline script bytes. Its pipeline-test snapshot would delete main's
test_context_support_manifests_match_current_sources guard. Neither rollback is
selected. Keep main's actual pin bytes and guard; applicable checks must validate
them, rather than relabeling historical branch proof as a fresh main proof.

Older PR294-296 runtime snapshots also lack current cached_poll_body and
writer_lookup modules and tests. Dropping those historical snapshots does not
drop their requested behavior: the newer implementation is retained in main.

## Minimal Candidate

- Preserve all current-main production Rust, proof bodies, pins, service units,
  workflows and CI guards, except the scoped deployment-source revalidation
  repair described below.
- Include signed52c5cfff's two regression tests: loader materialization/backing
  ownership and terminal cross-target preparation refusal/retry.
- Document the actual existing static deployment-tool build and protected
  compiler/runtime/device installation interface in
  ../runtime-release-installation.md.
- Stage and qualify maintained deployment tools in private non-root scratch;
  no new installer, framework, profile, credential or capability authority.

The original selected52c5cfff CPU evidence remains scoped to its exact source.
Its 205 passing outcomes and nine existing artifact-dependent ignores are
historical applicable regression evidence, not native admission or proof of a
future integrated source. New installation-tool/source-head receipts must be
provided separately. No complete A1/A2/A3, HIP/HSA parity or native performance
claim is made.

Independent compiler-owner review and release-coordinator publication/merge are
required. Keep PR294-297 open until the consolidated candidate and disposition
are reviewed. Protected deployment remains subject to genuine compiler closure,
profile, proof-runtime, exact artifact assurance, service credential/lifetime
custody and current device admission.

## Actual Staging Failure And Repair

The first maintained deployment-tool builder stopped before artifact creation:
196 deployment tests passed, one failed and twelve were ignored. The failure was
source_path_and_directory_replacement_after_sealing_refuse, mutation zero:
same-length origin content replacement was not refused by the metadata-only
final origin check. Sealed admitted copies remained intact.

The candidate adds a bounded final origin-content hash pass bracketed by the
existing metadata/path checks. A deterministic regression models equal metadata
without depending on timestamp scheduling and checks that the sealed original
copy is unchanged. Packaging's logical work quote increases from 64 to 72 units
per code byte for the additional streaming read/hash, with no additional full
backing. Existing exact/one-short account tests remain applicable. Original
failed receipts are retained; new-source validation is a distinct attempt.

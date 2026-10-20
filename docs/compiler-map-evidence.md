# Passive Compiler Map Diagnostics

The standard-library helper `scripts/compiler_map_evidence.py` summarizes
**supplied offline observations**, not live custody or an approved compiler
closure. It never opens paths named in map records, hashes code, loads libraries,
executes a compiler, or constructs a production capability.

```sh
python3 -I -B scripts/compiler_map_evidence.py observations.json
python3 -I -B scripts/tests/compiler_map_evidence_test.py
```

The CLI emits one JSON report. Exit zero means the supplied record structure
was consistent within parser bounds, **not** that coverage or a closure is
complete. Identity conflicts, observer errors and exhausted bounds produce
nonzero exit status and an incomplete report.

## Input And Bounds

The document has `boot`, `errors` (array), `truncated` (boolean), and
`observations` (array). Each observation has `boot`, a `process` with positive
integer `pid` and `start_ticks`, `exe`, `command_ancestry`, and `mapped_files`.
Mapping records contain `path`, `map_device`, `map_inode` and `raw_lines`
(complete proc-map rows). Mount namespace fields are reported when supplied.
No filesystem lookup validates serialized identities. Legacy documents without
an explicit truncation declaration remain inspectable but incomplete.

Input is capped at 8 MiB, each map row at 8192 characters, and observations,
mapping records, segments and late hash records share a 65536-record budget.
`--max-records` may lower that budget, never raise it. Duplicate JSON fields,
nonfinite numbers, malformed/overflowing ranges and inconsistent identities
are refused or retained as unknown records. Budget exhaustion declares possible
unprocessed records. Caller CPU/memory/time containment remains necessary;
these bounds are not a production logical resource ledger.

## Classification And Custody

Objects aggregate all supplied segments and observations by recorded boot,
device and nonzero inode. Inode-zero maps remain separate unknown objects.
Aliases are preserved; conflicting same-path identities and PID reuse make
input inconsistent. Inode identity alone is not content identity and does not
exclude inode reuse. Any executable segment dominates other classifications,
including for `.tmp`, `.data`, deleted and unfamiliar paths.

A writable, non-executable map in a recorded `rust-lld` process is only a
`nonexec-writable-output-candidate`. Other non-executable maps remain
`nonexec-unqualified`; an executable segment is `observed-exec-unqualified`.
These are observational labels, never approved backend/proc-macro roles.
One writable snapshot cannot prove an object never executed. Missing or
deliberately omitted executable segments cannot be recovered by this parser.
Coverage and executable-closure completeness are therefore always false.

Metadata-only records, recorded late hash claims and claims of
live-retained-mutable custody are distinct from writer-excluded content pins.
**Every serialized claim remains unverified**, including hashes,
`stable_post_observation`, `immutable` and `writer_excluded`. JSON cannot
establish even a live descriptor. No validated content pin, approval, profile,
proof, native execution or immutable-custody authority is produced.
Rename, unlink, same-size bytes, late path lookups and copied files cannot
recover historical content or exclude writable aliases/shared mappings.

## Historical Scope And Next Boundary

The 2026-10-15 installer observations reviewed in audit 4005 contained two
vanished paths with `rw-s` maps in `rust-lld`: temporary backend and linkme
outputs. Those records do **not** demonstrate executable plugin loads.
Historical bytes and rename lineage remain unknown; 57 post-observation pins
are not an atomic runtime closure. Original installer/reviewer receipts remain
unchanged. This tool removes no unknown objects from any production roster.

Producer-owned FD capture is a separate future API: retain original identity
at output creation, establish completion and writer exclusion before content
pinning, and preserve approved source custody before loading. Fixed-origin
approval, stopped-task mmap/range checks, refinement requirements, V3 profiles
and application lifetime owners are unchanged.

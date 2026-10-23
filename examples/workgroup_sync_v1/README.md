# Workgroup synchronization V1

This standalone package defines fixed-launch workgroup reduction, scan, and
atomic kernels with CPU oracles and formal contracts. LDS collectives are
target-neutral through semantic MIR and Kernel IR, then bind independently to
the production compiler profile for gfx942 or gfx950.

## LDS publish/read reduction

`lds_publish_read_reduce_i32_v1` is ordinary attributed Rust source. Every
admitted lane publishes one `i32` to its same-index LDS slot. The public
`fe2o3-device` target-neutral workgroup reduction supplies the uniform publish
barrier, deterministic reduction barriers, and final reuse barrier. Lane zero
is the only global output writer. Admission rejects mathematical sums outside
`i32`, so the device's wrapping tree computes the exact mathematical sum.

## Inclusive and exclusive scans

Six scan families expose 18 feature-isolated ordinary attributed Rust entries:
both modes for every admitted scalar at each exact 3-, 65-, and 255-lane
workgroup extent. They use the same affine dynamic-LDS and workgroup-authority
contract. The public
`inclusive_scan_sum` and `exclusive_scan_sum` terminals order prefixes by
linear work-item rank and return one result per lane through a typed
`DisjointSlice`. Their exact compiler, CPU simulation, schedule replay, and
debug evidence contract is documented in
[`target-neutral-workgroup-scan-v1.md`](../../docs/target-neutral-workgroup-scan-v1.md).
The checked-in [`scan-u32-request.json`](scan-u32-request.json) drives the
ordinary 3-lane source through Bundle V5 and the CPU simulator with the command
documented there.

The reduction source requests `DynamicLds::<i32>::exact_current::<64>` and
passes that linear capability directly to the collective terminal. It obtains
`WorkgroupCollectives::current()` and calls
`WorkgroupCollectives::reduce_sum_portable`; neither operation names or selects a GPU
family. It cannot substitute a host/global raw pointer or expose the LDS
pointer. The production importer authenticates this source, its
launch-resource sidecar, and its complete reachable portable-MIR closure. The
generic semantic lowerer creates one aligned, epoch-branded workgroup
allocation shared by all lanes; no workload profile or prebuilt Kernel IR is
selected. V1 admits only sum over `u32`, `i32`, and `f32`, with an exact
one-dimensional workgroup in `1..=256`; scan also admits odd sizes and partial
waves while reduction remains power-of-two-only. Every lane participates in
every acquire-release barrier phase. Unsupported scalar types, operations,
geometry, provider identity, and target profiles fail before target-bound
Kernel IR is created.

## Masked wrapping row transform/reduction

The opt-in `row-affine-u32-kernel` feature selects the explicit SIMT
`row_affine_sum_u32_v1` source. A checked shared view specifies an element
offset, rows, columns and row stride. For every logical row the result is
`sum(value * scale + bias) mod 2^32`, over active columns only.

The initial contract admits 0..=128 columns and complete 64-lane workgroups,
with one group per row or extra inactive groups, 256 bytes of LDS and at least
`rows` output elements. Each lane handles columns `2 * lane` and
`2 * lane + 1`; every lane reaches the existing target-neutral reduction.
Only lane zero stores through checked row-striped output ownership. Empty rows
perform no stores; empty columns produce zero without applying bias. Trailing
output elements remain unchanged.

Invalid widths, output extents, physical geometry, insufficient groups and
invalid checked input views trap before writes. Nonempty views require
`row_stride >= columns` and only the last logical element, not trailing row
padding, must fit the input. Empty views still require `offset <= input.len()`.
The caller must supply complete groups: a rounded physical launch extent is
not evidence that a simulator request contains every lane.

`row_affine_oracle` supplies the independent u128 logical-row specification
and shared boundary corpus, including true zero-length views. Host oracle
tests are not device-kernel execution. The required source-to-SIM test uses the
existing production driver, not a second CPU kernel body. It exports the
ordinary Rust source for gfx942 and gfx950 profiles, then checks all 86 cases
with canonical and seeded schedules and persisted replay. Input bytes,
initialization masks, output canaries, invalid requests and stale replay
bindings are checked. These are CPU semantic checks, not hardware execution or
performance predictions.

Run the source-to-SIM regression with the repository's pinned nightly:

```sh
cargo test --locked -p rustc-codegen-fe2o3 \
  --test production_neutral_workgroup_reduce_driver_v1 \
  ordinary_row_affine_source_matches_oracle_and_replay -- --ignored --exact
```

This does not establish the required structured tile or mixed variant,
generated host/artifact admission, direct-KFD execution or M2 completion.
The CPU simulation lesson's whole-file tab 6 and paired inventory bind this
exact SIMT source through its production test and feature selection. Tile and
mixed source variants remain pending under
[issue #275](https://github.com/harsh-nod/fe2o3/issues/275); this source association
and the separately pinned CPU checkpoint do not complete a tutorial pair.
Hardware launch representability is a separate runtime contract. The old
`run-gfx942.sh` and protected scalar driver still select their existing
reduction profiles and do not qualify this new ABI.

Run the feature-isolated host tests through the normal binding driver:

```sh
cargo fe2o3 test --locked --manifest-path examples/workgroup_sync_v1/Cargo.toml \
  --no-default-features --features row-affine-u32-kernel --test row_affine
```

## Checked output-tile row variant

The opt-in `row-affine-u32-tile-kernel` selects
[`kernel_row_affine_u32_tile.rs`](src/kernel_row_affine_u32_tile.rs), using
`DisjointSlice<u32, Tiled2D<Index1D, 64, 1, 64, 1>>` and the matching safe
`checked_tiled_2d` / `get_tiled_2d_mut` accessors. Each 1x64 output tile has
logical width one: only lane zero owns an active output column, and its tile
row is the workgroup row. This keeps compact output storage and unchanged tails.

This is a structured **output** tile with explicit SIMT arithmetic and the
same uniform LDS collective. It does not implement the staged `MaskedTile1D`
load/transform/reduce API. All admission checks, strided input reads, wrapping
operations, empty-view behavior and collective participation match the original
source; `row_affine_source` checks their parsed AST correspondence. The variants
are feature-isolated and cannot be selected together.

The production driver exports this actual source, checks its source-map member,
and runs the same independent 86-case oracle, initialization/canary checks,
canonical/seeded schedules, persisted replay, stale bindings and invalid requests:

```sh
cargo test --locked -p rustc-codegen-fe2o3 \
  --test production_neutral_workgroup_reduce_driver_v1 \
  ordinary_row_affine_tile_source_matches_oracle_and_replay -- --ignored --exact
cargo fe2o3 test --locked --manifest-path examples/workgroup_sync_v1/Cargo.toml \
  --no-default-features --features row-affine-u32-tile-kernel --all-targets
```

This added output-ownership source does not close the full tile/mixed tutorial
obligations under #275, establish native compilation, or qualify GPU execution.
The pending qualification and historical negative roles described above remain.

The existing scalar and scan selections remain in `Cargo.toml`. For each feature
below, the feature-isolated host command is
`cargo fe2o3 test --locked --manifest-path examples/workgroup_sync_v1/Cargo.toml --no-default-features --features FEATURE --all-targets`.

| Scalar | Inclusive scan features for 3 / 65 / 255 lanes | Exclusive scan features for 3 / 65 / 255 lanes |
| --- | --- | --- |
| u32 | `lds-scan-u32-kernel` / `lds-scan-u32-65-kernel` / `lds-scan-u32-255-kernel` | `lds-scan-u32-exclusive-3-kernel` / `lds-scan-u32-exclusive-65-kernel` / `lds-scan-u32-exclusive-kernel` |
| i32 | `lds-scan-i32-inclusive-kernel` / `lds-scan-i32-inclusive-65-kernel` / `lds-scan-i32-inclusive-255-kernel` | `lds-scan-i32-3-kernel` / `lds-scan-i32-kernel` / `lds-scan-i32-255-kernel` |
| f32 | `lds-scan-f32-3-kernel` / `lds-scan-f32-65-kernel` / `lds-scan-f32-kernel` | `lds-scan-f32-exclusive-3-kernel` / `lds-scan-f32-exclusive-kernel` / `lds-scan-f32-exclusive-255-kernel` |

The original reduction features are `lds-kernel`, `lds-u32-kernel` and
`lds-f32-kernel`. The existing ignored driver tests
`ordinary_scan_sources_export_v5_and_execute_every_cpu_observation_path` and
`ordinary_integer_scan_sources_wrap_on_overflow` cover real scan export/execution
and overflow respectively. Their existing target and evidence boundaries apply.

## Scoped atomic add

The second profile admits one coherent global `u32` atomic object, relaxed
ordering, system scope, and exactly 64 eligibility declarations. Eligible
lanes add once; ineligible lanes do not touch the object. Overflow is rejected
by host admission so the final value is an exact mathematical sum.

Its ordinary attributed Rust source is compiled from `src/scoped_atomic.rs`.
The signature uses `DeviceGlobalMutPtr<u32>` to state global address space
without pretending the concurrently shared object is a Rust slice. Generated
host bindings accept only a one-element initialized `u32` device region held
under an exclusive borrow; they expose neither a raw pointer nor a launch path.
The macro registration binds global address space, mutability, pointee type,
physical pointer layout, and exclusive alias admission.

## Evidence boundary

The 2026-09-25 integration check does not qualify the broad native LLVM gate:
its first gfx942 LDS reduction fails generic formal memory admission with an
inter-invocation conflict. The coverage described below remains an acceptance
requirement, not a claim that every current gate passes. Historical native
results retain their original source/compiler identities; row CPU results
grant neither native artifact nor GPU execution authority.

The checked CPU oracles and deterministic debug/release tests are usable now.
They fail before output mutation and reject missing or divergent barriers,
stale epochs, duplicate or wrong owners, invalid lane counts, incorrect atomic
address space, ordering, scope, target, eligibility, overflow, and substituted
outputs. Verus models initialization, convergence, epoch reuse, ownership,
exact integer sums, and atomic eligibility, with expected-negative mutations.

Both kernels are ordinary attributed Rust modules with source-level typed ABIs.
They enter the same feature-independent production transaction: authenticated
rustc collection, semantic MIR, ranked PLIRON, verified Kernel IR, composed
formal/ranked memory checks, target-bound gfx942 or gfx950 LLVM,
compiler-bound handoff, measured upstream LLVM target APIs plus in-process LLD,
and COV6 inspection. There is no workload-profile selector on this route, and
the compiler handoff grants no load or launch authority.

The ignored neutral-collective rustc driver requires the pinned nightly. It
authenticates the compiler-observed provider definition identities and
recomputed complete source-closure pin, then checks semantic MIR, ranked
PLIRON, generic LDS/tree/barrier KIR, target binding, and the LLVM route for
the three reduction and six canonical scan entries on both targets. The 18-entry
Bundle V5 CPU gate independently covers every scan family at every representative
extent. The separate
protected reduction driver requires the authority launcher and measured Worker
V3 and LLVM build identities. It starts again from the immutable reduction
sources for all three scalar profiles
and checks real Worker/finalizer output for the exact
256-byte static group segment, 288-byte complete kernarg ABI, required
`64x1x1` workgroup, COV6 descriptor, and deterministic two-run HSACO. It does
not dispatch the code object or grant load/launch authority. The scoped-atomic
profile and scan HSACO qualification are outside this protected driver.

This is bounded source-to-code-object evidence, not a compiler-refinement proof
or a claim of generalized memory safety, race freedom, reduction coverage, or
GPU execution. The closed target-neutral V1 reduction contract is supported
only on the pinned gfx942 and gfx950 production compiler profiles. This is
compiler-target evidence for both targets, not gfx950 execution evidence. The
currently qualified direct-KFD execution path is gfx942 only and must enter
through Worker V3 and the pure-Rust KFD runtime. The finalizer uses no COMGR and
no shell linker; it does not shell out to `clang`, `llc`, or `ld.lld`.

## Validation

```sh
cargo fe2o3 test --locked --all-targets \
  --manifest-path examples/workgroup_sync_v1/Cargo.toml
cargo fe2o3 test --release --locked --all-targets \
  --manifest-path examples/workgroup_sync_v1/Cargo.toml
cargo fe2o3 clippy --locked --all-targets \
  --manifest-path examples/workgroup_sync_v1/Cargo.toml -- -D warnings
VERUS=/absolute/path/to/pinned/verus \
  examples/workgroup_sync_v1/run-verus.sh
```

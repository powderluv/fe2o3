# fe2o3 GPU

`fe2o3` is a developer-preview GPU programming stack for writing kernels in
Rust, compiling them through a typed intermediate representation, and building
a direct-KFD runtime for AMD GPUs.

The project is investigating a deliberately integrated model: source
semantics, compiler evidence, artifacts, runtime authority, deterministic CPU
simulation, and debugger/profiler observations all carry explicit identities
instead of being joined by filenames or convention.

> **Developer Preview**
>
> fe2o3 is under active development. It is not ready for production workloads,
> does not yet provide a supported production direct-KFD first-run experience, and makes
> no API, wire-format, or compatibility stability promise before the first
> preview release. Current direct-KFD qualification is bounded to MI300X
> `gfx942:xnack-` lanes. See [What works](#what-works) and
> [Known limitations](#known-limitations) before evaluating it.

For source-checkout experiments on MI350-family `gfx950` hardware, selected
examples provide explicit SIMT and tile compile/numerical commands. Start with
the [gfx950 engineering evaluation guide](docs/getting-started.md#gfx950-engineering-evaluation).
Those example-specific HIP/HSA or external-HSACO observations are separate from
production direct-KFD authorization, protected proofs, and a supported release.

## Typed kernel authoring

Kernel authors write `#[kernel(typed)]`; they do not choose or maintain a
namespace hash:

```rust
#[kernel(typed)]
pub fn saxpy(x: &[f32], mut y: DisjointSlice<f32>, alpha: f32) {
    let index = thread::index_1d();
    if index < x.len() && index < y.len() {
        let output = y.get_mut(index);
        *output = alpha * x[index] + *output;
    }
}
```

Run the package through `cargo-fe2o3` so the compiler can derive its binding
from Cargo's crate name and ordered rustc metadata:

```text
cargo fe2o3 check --all-targets
cargo fe2o3 clippy --all-targets -- -D warnings
cargo fe2o3 test --all-targets
cargo fe2o3 build
```

The first three commands are authority-free host workflows. `build` enters the
production GPU compiler path. Binding hashes shown in compiler evidence are
outputs for reproducibility and cross-stage identity joins, not source syntax
for an author to invent or copy. An explicit namespace remains accepted only
for compiler compatibility and adversarial fixtures; public kernels must use
the compiler-derived form.

For experimental instruction-level control, see [physical-entry source authoring](docs/physical-entry-source-v20.md). That bounded diagnostic route is separate
from normal production admission; its [qualification report](docs/physical-entry-source-qualification-20260924.md) records simulation and native inspection limits.

The project currently ships only from a source checkout and makes no crates.io
installation promise. The first developer-preview release remains blocked on
the conditions in the [release process](docs/release-process.md) and
[launch issue #267](https://github.com/harsh-nod/fe2o3/issues/267).

## Why fe2o3

- **Single-source Rust kernels.** Kernel functions use ordinary Rust syntax,
  typed kernel arguments, and explicit device APIs.
- **Direct KFD.** The production runtime direction is the Linux KFD interface;
  HIP and HSA are not fallback execution paths.
- **Typed compiler contracts.** Source, semantic MIR, Pliron, Kernel IR,
  LLVM, artifact, and runtime boundaries are represented explicitly and fail
  closed when a required association is unavailable.
- **CPU simulation without a GPU.** A deterministic Kernel IR simulator can
  execute the supported semantic subset and expose logical work-item, wave,
  workgroup, memory, atomic, fence, and barrier observations. It does not
  predict GPU performance.
- **Virtual runtime without device handles.** Bounded identity-local
  allocations, copies, queues, dependencies, dispatches, completions, and
  quiesced failure states execute admitted KIR through the same CPU simulator.
  This model-only path grants no KFD, load, launch, or hardware authority.
- **Semantic debugging and profiling.** Agent-facing JSONL protocols preserve
  provenance and distinguish declared, observed, inferred, and unavailable
  facts. An artifact-bound transformation sidecar preserves exact one-to-one,
  one-to-many, many-to-one, many-to-many, and eliminated relations without
  mistaking cardinality for an optimization observation. CPU replay supports
  reverse navigation and structured diagnosis;
  bounded ROCgdb control and rocprofv3 planning/import workflows are
  implemented. Admitted stopped-GPU state, protected real-dispatch capture, and
  ATT decoding remain incomplete.
- **Evidence-aware verification.** Formal contracts and qualification evidence
  are kept separate from compiler, publication, load, and dispatch authority.

## Kernel example

This is the default kernel selection from
[`examples/fill`](examples/fill/src/lib.rs):

```rust
use fe2o3_device::{DisjointSlice, kernel, thread};

#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1]),
)]
pub fn fill(mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let Some(value) = out.get_mut(idx) else {
        return;
    };
    *value = 42.5;
}
```

`DisjointSlice` makes the output partition explicit. `thread::index_1d()` is a
typed logical index, and the bounds check remains part of the admitted kernel
semantics. The optional `reference-proof` feature attaches an independent CPU
reference to this same body. That selection requests a protected semantic proof;
its [proof boundary](examples/fill/README.md) still requires conditional ownership
continuation. The default learning workflow does not select this feature and
does not claim compiler-proved CPU/GPU equivalence.

Namespace-free `#[kernel(typed)]` packages are wrapper-managed. Raw Cargo does
not synthesize their compiler-owned crate binding, so use `cargo fe2o3` or the
repository scripts when checking kernel examples.

## Why Write Kernels In fe2o3

The unique bet in `fe2o3` is that a kernel should ship with its intended
semantics, not just its device code. A contributor writes the GPU kernel and a
safe Rust CPU reference or oracle for the same bounded behavior, then the stack
records how mismatches are caught:

- Proof-facing examples use Verus source models and negative fixtures to reject
  wrong value, ownership, bounds, or frame behavior before promotion.
- Runnable examples compare the GPU result, state, or metadata against the safe
  CPU reference and fail the bounded runner on a mismatch.
- Promoted compiler-time claims must go through the compiler-owned
  `SafeReferenceMirToLivePliron` join, generated per-compilation Verus replay,
  and PLIRON structural reconciliation before KIR lowering.

That does not mean arbitrary Rust CPU and GPU semantics are equivalent at
compile time today. Unsupported reference MIR, tensor-component replay, finite
numerical error replay, LLVM/ISA behavior, launch, hardware execution,
performance, and full-model integration still fail closed unless a specific
evidence record says otherwise.

## CPU quick start

### Requirements

- x86-64 Linux with Bash and GNU `realpath`
- Git
- `rustup`; the repository pins `nightly-2026-04-03` and its required
  components in [`rust-toolchain.toml`](rust-toolchain.toml)
- Enough disk space for a Rust compiler workspace build

Clone the repository and run the default fill source-to-CPU quickstart. This
selection does not require the optional protected reference-proof runtime.

```console
git clone https://github.com/harsh-nod/fe2o3.git
cd fe2o3
bash scripts/quickstart.sh no-gpu
```

This command exports the actual `fill` source through the production source/MIR/KIR
stages and checks its CPU simulation against an independent expectation,
including untouched canaries. No replacement fixture or proof fallback is used.
The getting-started guide also provides an exact-KIR fixture workflow that does
not compile this source.

Bundle export and simulation do not load a GPU, silently fall back from a
hardware command, authenticate compiler execution, or establish equivalence
with GPU execution. See the [getting-started guide](docs/getting-started.md)
for the component commands, exact-KIR fixture, debugger, and cleanup behavior.

## GPU evaluation status

For compiler engineering only, `cargo fe2o3 engineering hsaco` can extract one
explicitly selected device crate and run its inert handoff twice through an
exactly measured native worker. The exact `--target` selects `gfx942:xnack-`
or `gfx950:xnack-` throughout extraction and COV6 object inspection; the
observation is rejected if the requested and emitted targets differ. It writes only a fresh
`fe2o3-engineering-v1/<content-id>/` observation with `"authority":"none"`.
This command does not contact the production compiler-execution supervisor and
its output cannot be adopted as a production generation, publication, load, or
launch owner. The command requires a versioned `cargo vendor` tree containing
both the selected device crate's dependencies and the pinned nightly rust-src
workspace's complete `-Zbuild-std=core` registry closure. Construct it with the
same pinned nightly Cargo and rustc passed to the command:

```console
toolchain=nightly-2026-04-03
sysroot="$(rustup run "$toolchain" rustc --print sysroot)"
rustup run "$toolchain" cargo vendor --locked --versioned-dirs \
  --manifest-path /absolute/device/Cargo.toml \
  --sync "$sysroot/lib/rustlib/src/rust/library/Cargo.toml" \
  /absolute/cargo-vendor
```

The command checks the pinned rust-src lock against that measured vendor before
preparing the compiler child, then retains and revalidates both trees while
Cargo runs with a fresh home, frozen resolution, no ambient configuration, and
no network. In particular, caller `RUSTC_BOOTSTRAP`, rustup selectors, and
toolchain overrides remain rejected rather than filling a missing closure.
Host build scripts and proc macros are linked through separately declared,
SHA-256-bound clang, lld, and lld argv proxy executable images. The proxy invokes
the sealed lld descriptor directly with GNU linker identity; these sealed
descriptors replace ambient `PATH` lookup and their identities are recorded in
the observation.
Run `cargo fe2o3 engineering hsaco --help` for the explicit compiler, host
linker, provider, target, COV, resource-bound, vendor-source, and Cargo-selection
arguments.

There is currently no supported copy-and-paste production direct-KFD quick start.
`cargo fe2o3 build` enters the production compiler transaction, but ordinary
example applications still lack the production Worker V3 verifier deployment
and concrete semantic-to-final-machine proof backend/artifact needed to
authorize load and dispatch. The host provides only the typed, sealed join for
such a backend; it does not synthesize proof authority. For example,
`fe2o3-vecadd` and `fe2o3-fill` intentionally fail closed before dispatch.

Hardware development and CI exercise bounded direct-KFD mechanics and exact
`gfx942:xnack-` qualification lanes on MI300X. These tests demonstrate specific
compiler/runtime properties; they are not a supported general application
workflow. Publishing a clean-checkout source-to-MI300X example is a blocker for
a future GPU-ready preview, not a claim made by this source/simulator preview.

## What works

| Area | Current developer-preview state |
| --- | --- |
| Rust kernel surface | Typed kernels, device indexing, checked buffer views, bounded scalar/control/memory subsets |
| Compiler | Source/MIR through typed Pliron and verified KIR; selected `gfx942` and `gfx950` LLVM/HSACO engineering slices, with example-specific target and evidence limits |
| CPU simulation | Deterministic execution of admitted canonical KIR V7 plus exact direct V9/V10/V11 custody, including supported helpers, barriers, workgroup memory, wave f32 collectives, memory intrinsics, pointer-access restriction, atomics, fences, floating point, and seeded schedule exploration |
| Virtual runtime | Bounded model-only allocation, copy, queue, dependency, dispatch, completion, and ambiguous-failure lifetimes over admitted KIR; generated-host and multi-device integration remain incomplete |
| CPU debugger | Work-item, logical wave, workgroup, operation, stack, SSA, allocation-relative memory, break/watch, reverse replay, and structured diagnosis over retained simulator evidence |
| Live debugger | Bounded direct-KFD observation/control; ROCgdb V4 authenticates hierarchy/relative PC and V5 adds same-stop structured register/scalar-local inspection, while the installed direct-KFD target still cannot produce a physical GPU stop and source/ISA/memory remain typed unavailable |
| Profiling | Bounded rocprofv3 dispatch/counter/PC import, exact PC-to-sparse-source/IR/ISA queries over artifact-bound characteristic evidence, authority-free admission/query of external ROCprofiler SDK 7.2.4 decoded ATT callbacks, exact supplied decoded-ATT-PC/HSACO/Characteristic correlation, complete same-domain structural comparison, replayed production optimizer decisions, and deterministic evidence-linked regression hypotheses with contradictions and next measurements; opt-in direct-KFD runtime lifecycle, host-staging, queue/stream membership, completion and evidence-cited lifecycle causality queries, and a bounded paired rocprof-wrapper host-wall comparison are also implemented. Protected real-GPU PC/ATT capture, authenticated decoder custody, raw ATT decoding, device-copy/dependency producers, observed schedule execution, causal acceptance, capture-overhead qualification, and direct-KFD cross-collector correlation remain incomplete |
| Runtime | Pure-Rust KFD/AQL foundations and bounded MI300X execution diagnostics; public application authorization is incomplete |
| Verification | Verus contracts and evidence-bearing compiler/runtime boundaries for bounded slices; not an end-to-end proof of general kernels |

The [support matrix](docs/support-matrix.md) separates implemented,
qualified, experimental, and unsupported combinations. Detailed historical
milestones are retained in the [project status archive](docs/project-status.md).

## Known limitations

- The only currently qualified production direct-KFD profile is the bounded
  MI300X `gfx942:xnack-` profile. Other AMD targets are not implied.
- An ordinary external project cannot yet compile and dispatch a general Rust
  kernel through one supported public command.
- The simulator accepts defined KIR V7/V9/V10/V11 semantic subsets. Legacy
  bundles and raw CLI inputs carry V7; authority-free Bundle V5 carries an
  exact production V8/V9 module re-encoded losslessly as V10, while the current
  Bundle V6 route carries exact V11. Unsupported types and operations fail
  closed; CPU results are not timing or performance predictions.
- CPU logical waves model semantic collectives and visualization partitions,
  not physical GPU wave scheduling or `EXEC` state.
- Live KFD debugging admits authenticated hierarchy and relative-PC evidence and
  can retain same-stop, wave-scoped register values and simple scalar locals.
  The installed direct-KFD target cannot yet produce a physical GPU stop, and
  general lane values, target memory, source stepping, and breakpoints remain
  unavailable.
- ROCgdb integration is bounded by what the installed debugger exposes and is
  not a source of fe2o3 compiler or runtime authority.
- One protected MI300X direct-KFD/rocprofv3 qualification completed with exact
  runtime dispatch evidence, but that ROCprofiler SDK 1.1.0 run produced no
  collector artifacts and therefore no cross-collector dispatch import or
  join. Supplied PC evidence can be joined exactly to sparse
  characteristic source/IR/ISA coordinates, but that archive remains a
  self-claim without producer readmission and does not enable cross-run site
  deltas. Direct-KFD runtime observations now expose bounded local lifecycle
  causality, but do not imply rocprof correlation, device timing, counters, PC
  samples, dependency edges, or copy-engine events. ATT decoding is unavailable
  without a mutation-proof decoder.
- Multi-GPU distributed kernels and communication/computation overlap are not
  a supported execution surface.
- The compiler and protocols are evolving. Do not treat crate APIs, KIR, bundle,
  debugger, profiler, receipt, or evidence formats as stable unless a document
  explicitly freezes a version.

## Architecture

The intended production flow is:

```text
Rust source
  -> semantic MIR
  -> typed Pliron
  -> verified Kernel IR
  -> AMDGPU LLVM / HSACO
  -> authenticated artifact publication
  -> direct-KFD load and dispatch
```

CPU simulation branches after verified Kernel IR and never becomes runtime or
hardware evidence. Debugger and profiler services join observations through
content identities and typed provenance rather than native paths, descriptors,
or addresses.

Start with the [architecture overview](docs/architecture-v2.md), then use the
[documentation index](docs/README.md) to find compiler, runtime, debugger,
profiler, simulator, verification, and evidence contracts.

## Repository layout

| Path | Purpose |
| --- | --- |
| `crates/fe2o3-device` | Kernel-facing Rust APIs and types |
| `crates/rustc-codegen-fe2o3` | rustc integration, production lowering, and source-to-simulator export |
| `crates/cargo-fe2o3` | Cargo orchestration, inspection, debug, and profile commands |
| `crates/fe2o3-kfd` | Direct Linux KFD boundary |
| `crates/fe2o3-kir-sim*` | Deterministic CPU simulator and CLI |
| `crates/fe2o3-sim-differential` | Generated CPU semantic differential harness and reducer; its production closure is GPU-runtime-free |
| `crates/fe2o3-sim-physical-differential` | Opt-in identity-bound simulator/direct-KFD comparison protocol |
| `crates/fe2o3-virtual-runtime*` | Authority-free virtual lifecycle and headless JSON CLI over admitted simulator inputs |
| `crates/fe2o3-debug-*` | Debug protocol, simulator debugger, and live-tool adapters |
| `examples/` | Kernel source, host-boundary, proof, and qualification examples |
| `docs/` | Architecture, contracts, evidence policy, testing, and status |
| `scripts/` | CI and qualification entry points |

## Development

Run the bounded contributor preflight before opening a pull request:

```console
cargo fmt --all -- --check
bash scripts/ci-local.sh standalone-locks
bash scripts/tests/quickstart.sh
bash scripts/quickstart.sh source-check examples/vecadd/Cargo.toml
BASE_SHA="$(git merge-base origin/main HEAD)"
HEAD_SHA="$(git rev-parse HEAD)"
bash scripts/ci-local.sh workspace-policy
bash scripts/ci-local.sh hygiene-delta "${BASE_SHA}" "${HEAD_SHA}"
```

Compiler, runtime, proof, and trust-policy changes must also run their
applicable broader lanes, including `bash scripts/ci-local.sh check` for
workspace/package check coverage and `bash scripts/ci-local.sh generic-core`
where required. Do not replace those lanes with raw
`cargo check --workspace --all-targets`: wrapper-managed typed kernels require
the sealed `cargo-fe2o3` driver. The full validation matrix adds codegen
shards, policy checks, Verus, compile-only AMDGPU checks, and hardware lanes. See
[testing](docs/testing.md) for trust boundaries and required environments.

External contributions are welcome once they satisfy the repository's
fail-closed authority and evidence boundaries. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md), the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and
[`SECURITY.md`](SECURITY.md) before submitting work. Support expectations are
documented in [`SUPPORT.md`](SUPPORT.md); maintainer and decision ownership is
documented in [`MAINTAINERS.md`](MAINTAINERS.md).

## Project resources

- [Getting started](docs/getting-started.md)
- [Support matrix](docs/support-matrix.md)
- [Documentation index](docs/README.md)
- [Debugger and profiler tutorial](https://harsh-nod.github.io/fe2o3-kernels/#/debugger/profiler-import)
- [Implementation roadmap](docs/implementation-roadmap-v2.md)
- [Current parity dashboard](docs/generated/cuda-oxide-parity-dashboard.md)
- [Issue tracker](https://github.com/harsh-nod/fe2o3/issues)

## License

Except for third-party-derived or file-specific material identified in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), fe2o3-authored source is
licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE)), or
- MIT License ([`LICENSE-MIT`](LICENSE-MIT))

at your option. Contributions are accepted under the same dual license unless
explicitly stated otherwise.

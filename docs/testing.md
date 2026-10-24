# Testing fe2o3

The test matrix separates target-independent checks, Verus proofs, code-object
compilation, and hardware execution. Every pull request should run the bounded
fork-safe preflight:

```text
cargo fmt --all -- --check
bash scripts/ci-local.sh standalone-locks
bash scripts/tests/quickstart.sh
bash scripts/quickstart.sh source-check examples/vecadd/Cargo.toml
BASE_SHA="$(git merge-base origin/main HEAD)"
HEAD_SHA="$(git rev-parse HEAD)"
bash scripts/ci-local.sh workspace-policy
bash scripts/ci-local.sh hygiene-delta "${BASE_SHA}" "${HEAD_SHA}"
```

Proof, compiler, runtime, and trust-policy changes should also run their
applicable broader lanes. GitHub reports the bounded preflight first, then runs
the full generic core and codegen shards before merge and again on protected
push. The full matrix is intentionally hour-scale; the preflight provides the
fast feedback path without weakening the required generic qualification.

Raw `cargo check --workspace --all-targets` is intentionally not the supported
workspace check for namespace-free `#[kernel(typed)]` packages. Those packages
need the sealed `cargo-fe2o3` driver to supply the compiler-owned crate binding.
Use `scripts/ci-local.sh check` for workspace check coverage, or
`cargo fe2o3 check --locked --all-targets -p <wrapper-managed-package>` for one
wrapper-managed package.

## Generic validation

The current tutorial compiler checks use the package that owns production
PLIRON analysis. Each command must execute a nonempty test selection:

```sh
cargo test --locked -p fe2o3-pliron --test production_ranked_constant_fold
cargo test --locked -p fe2o3-pliron --lib pliron_analysis_witness
cargo test --locked -p fe2o3-pliron --lib textual_pliron_lit_suite
```

The former `fe2o3-kernel-analysis --lib pliron_analysis_witness` command selects
zero tests after the analysis move; its former `--test pliron_lit` target no
longer exists. Those spellings remain meaningful only at their explicitly
pinned historical source. The current witness tests retain positive replay,
incomplete cases, mutation refusals, and resource-bound controls; the textual
suite checks the existing sorted positive and diagnostic-refusal fixtures.
None of these commands executes a GPU artifact.

### Retained source-model proof replays

Historical model commands require their stated public source, not just the
current checkout and a pinned prover. Start each block at the repository root
and use a new, absent neighboring worktree. Existing source/model guards must
remain unchanged. Keep rustup's normal environment active (`. "$HOME/.cargo/env"`).

For the bounded MoE expert model, follow
[`examples/moe_expert_v1/README.md`](../examples/moe_expert_v1/README.md): source
`e3c7018f428192c350c9c425df8e4c031c788272`, Verus
`0.2026.08.02.b677dd5`, 15 verified obligations and six intended rejections.
For causal Flash attention, follow
[`examples/flash_attention_v1/README.md`](../examples/flash_attention_v1/README.md):
source `af0fd523e3b774377a9c5192cf0511e34fa19735`, the same Verus release,
25 verified obligations and ten intended rejections. Its retained companion
`examples/flash_attention_v1/run-memory-verus.sh` must also run from that
detached source; it uses Verus `0.2026.08.09.92f466f` and checks 13 obligations
plus eight intended rejections.

The removed LDS GEMM example remains replayable at its public historical pin:

```sh
git worktree add --detach ../fe2o3-proof-gemm-5a452 5a45239aeeda3ca64cf16beb7fb1d3589e649bfe
(
  cd ../fe2o3-proof-gemm-5a452
  cargo test --locked --manifest-path examples/tiled_gemm_v1/Cargo.toml --test lds_source_refinement
  VERUS=/absolute/path/to/pinned/verus examples/tiled_gemm_v1/run-verus.sh
)
```

Use Verus `0.2026.08.02.b677dd5` for this replay. The wrapper checks all six
retained proof groups (including the 96-obligation source/model group) and all
21 intended rejections. This is bounded source/model evidence only, not proof
of current Rust kernels, numerical SIMT/tile lowering, ISA, or GPU execution.
Current workgroup, row-softmax, and Wave64 model commands remain current; they
do not need these historical worktrees and retain their own explicit scopes.

The standalone numerical reference models have a focused CPU-only lane:

```text
bash scripts/ci-local.sh host-reference
```

This also runs from `test`, `generic-core`, and `generic`. It checks formatting,
strict Clippy, all test targets in debug and release, and strict rustdoc for
RMSNorm/residual, GQA prefill, paged GQA decode, SwiGLU, logits/compact
completion, RoPE/paged KV, and the pinned Ferric B3 linear reference.
The standalone manifests are selected explicitly;
root workspace tests do not include them. All share a `host-reference` subdirectory of
the configured Cargo target root, and each test harness runs serially.
Their tracked lockfiles also participate in `standalone-locks`.

These are bounded host arithmetic and metadata models, not GPU kernels,
compiler fixtures, or a model inference runtime. Their declared FP32 schedules
and final BF16 rounding are not a bitwise Hugging Face reference: intermediate
RMSNorm and attention rounding points can differ. Identity hashes describe
those host policies and grant no compiler, artifact, or runtime authority.
Passing this lane does not qualify SIMT/tile tutorial pairs or GPU execution.

This lane does not require ROCm or a GPU:

```text
scripts/ci-local.sh generic
```

The target-independent generic SSA planner has a focused executable lit-style
corpus:

```text
cargo test --locked -p fe2o3-mir-model --test ssa_lit
```

The sorted, size-bounded `.mir` fixtures under
`crates/fe2o3-mir-model/tests/ssa-lit/` use a compact CFG/event language. The
Rust harness parses and enforces one `RUN` and `EXPECT` directive per file,
ordered `CHECK` assertions, global `CHECK-NOT` exclusions, and exact rejection
text through `CHECK-ERROR`. This is an in-repository lit-style runner, not a
claim that `llvm-lit` or LLVM `FileCheck` executes the fixtures.

Successful fixtures require exact planner replay and equality with a second
construction. The 49-file corpus covers live and dead diamond merges, loop
preheaders, two-latch and nested loops, multiway branches, duplicate and
critical edges, irreducible control flow, unreachable pruning, edge-local
normal/unwind definitions, partial initialization, undefined uses and
transports, canonical identity, input limits, and the retained-memory promotion
boundary. Focused Rust tests additionally exercise sparse storage for
nonpromotable events, a 65,536-variable by 2,048-block CFG with zero or one
promotable local, a 12,000-block linear dominance-frontier case, a 4,096 by
4,096 sparse-IDF case, and 63/64/65-variable bit-word boundaries across merges,
kills, and edge definitions. The
two-dimensional stress case would exceed the planner storage limit if block
bitsets were still indexed by every semantic variable; it checks that the
incremental block-domain storage follows the promoted-variable count instead.
Semantic adapter tests also enforce exact inclusive storage/work boundaries
and one-unit-below fail-closed behavior for the combined planner, adapter, and
certificate accounting.

The production adapter and KIR replay have focused no-GPU coverage:

```text
cargo test --locked -p fe2o3-pliron production::semantic_ssa::tests
cargo test --locked -p fe2o3-lower-mir-kernel entry_seeds_only_the_certified_implicit_workgroup_lds_scope
```

The semantic suite covers disjoint partial moves, exact field
reinitialization, maybe-moved joins and loops, rejected parent/child and union
access, exact single-consumer transparent borrows, borrow escape and multi-use,
and authenticated implicit `WorkgroupLdsScope` entry values. The lowerer suite
checks that KIR seeds only the certified ambient value. Compiler-side kernel
matrices then exercise the same plan and block-argument transport, including
typed component witnesses, through real rustc extraction.

The ignored, toolchain-qualified
`ordinary_rust_nested_control_flow_executes_after_production_ssa_lowering`
integration test compiles a kernel containing nested loops, loop-carried
mutable locals, and a multiway `match` through rustc and the production
SSA/KIR path. It then admits the compiler-produced Bundle V6 with exact KIR
V11 custody and checks simulated output for zero, odd, and even trip counts
across all 64 lanes. Bundle V5/KIR V10 remains a frozen compatibility path for
older artifacts.

These are structural SSA and executable validation tests. They do not prove
semantic refinement, promote observable storage, formally verify the compiler,
or assert that the frontend accepts all Rust MIR. The planner is general over
the semantic CFG model; unsupported Rust types, operations, storage forms, and
GPU semantics continue to fail closed.

Kernel authors use `#[kernel(typed)]` without an explicit namespace. The same
compiler-derived binding is available to every supported developer command:

```text
cargo fe2o3 check --all-targets
cargo fe2o3 clippy --all-targets -- -D warnings
cargo fe2o3 test --all-targets
cargo fe2o3 build
```

The first three commands are host-only and authority-free. Clippy pins the
selected toolchain driver and rejects `--fix`. Host tests accept `--all-targets`,
or an explicit `--lib` or `--test NAME` selection for component-only coverage.
`build` is the production GPU compilation route. Hashes recorded in evidence
are compiler outputs, not author-supplied macro arguments.

The issue #216 virtual-runtime slice has focused no-GPU coverage:

```text
cargo test --locked -p fe2o3-virtual-runtime -p fe2o3-virtual-runtime-cli
cargo test --locked -p fe2o3-sim-runtime
cargo test --locked -p fe2o3-sim-differential
cargo test --locked -p fe2o3-kernel-ir simulation_bundle_v3
cargo test --locked -p fe2o3-kernel-ir simulation_bundle_v6
cargo test --locked -p rustc-codegen-fe2o3 --test production_ranked_bounds_driver_v1 ordinary_rust_exports_and_queries_exact_v3_typed_layouts_and_regions -- --ignored --exact
cargo test --locked -p rustc-codegen-fe2o3 --test production_ranked_bounds_driver_v1 ordinary_recursive_aggregates_export_and_unsafe_shapes_fail_typed -- --ignored --exact
cargo test --locked -p rustc-codegen-fe2o3 --test production_ranked_bounds_driver_v1 ordinary_recursive_aggregates_export_and_execute_bundle_v5 -- --ignored --exact
cargo run --quiet --locked -p fe2o3-sim-differential --bin fe2o3-sim-differential -- --seed-start 0 --cases 256
cargo run --quiet --locked -p fe2o3-sim-differential --bin fe2o3-sim-differential -- semantic-run-v2 --seed 0
cargo run --quiet --locked -p fe2o3-sim-differential --bin fe2o3-sim-differential -- f32-run-v3
cargo clippy --locked -p fe2o3-virtual-runtime -p fe2o3-virtual-runtime-cli -p fe2o3-sim-runtime -p fe2o3-sim-differential --all-targets -- -D warnings
cargo doc --locked -p fe2o3-virtual-runtime -p fe2o3-virtual-runtime-cli -p fe2o3-sim-runtime -p fe2o3-sim-differential --no-deps
scripts/ci-local.sh workspace-policy
scripts/ci-local.sh runtime-policy
```

The runtime-policy lane checks the virtual-runtime package closures, scalar
differential command, and normal runtime simulator adapter. The syscall-free
roots remain independent of HIP, HSA, DRM, KFD, device nodes, and GPU libraries.
The adapter is separately checked through the pure-Rust runtime policy and ELF;
it uses the public SPI owned by `fe2o3-runtime` but invokes no KFD path and links
no GPU library. Its dedicated ELF policy permits libc `dlsym`, which
`std::thread` imports for pthread compatibility, while rejecting `dlopen`, GPU
library names, and device paths. These tests establish model and semantic CPU behavior only;
they do not increase a hardware or parity pass count.

The ignored V3 production test requires the pinned AMD target. It exports an
ordinary Rust kernel through the production rustc transaction, decodes the
exact semantic-MIR and storage-map sections, queries structured type layout and
request-region state through `fe2o3-debug`, runs the same V3 bundle in the CPU
debugger, and rejects a re-signed hostile target-layout substitution. This is
source/unit evidence only. It neither adds aggregate execution semantics to
KIR V7 nor claims GPU or compiler-authentication authority.

The generic lane also runs the focused `kir-sim-capability-matrix` test before
the broader CPU package set. It checks the stable JSON command and requires an
exact simulator owner or typed unsupported reason for every top-level KIR
operation, terminator, profile, and scalar operation/type combination. The
matrix marks these as declared tool-contract facts with `authority: none`; its
newline-terminated compact JSON is fixed at 4,746,086 bytes.
Named GPU profiles select simulator data-layout semantics and do not claim
compiler lowering, ISA support, hardware availability, or physical validation.

The generic lane translates 256 deterministic wrapping-`i32` programs from the
independent `fe2o3-differential` model into admitted KIR V7, executes them in
the simulator, and compares every lane. It then runs the fixed-size V2 semantic
family corpus over every admitted fixed-width integer type, both admitted
target `index` widths, `bool`, exact
finite additions for all four admitted float storage types, multi-block CFG and
internal-call execution, global memory, overlapping shared views, and typed
bounds/initialization/undefined-operation rejections. Stable JSON digests bind
the exact cases, KIR identities, expected bytes, observations, and rejection
dispositions. Failure records retain deterministic reduction and exact replay
metadata. The capability query lists the deliberately untested or typed
unsupported families, and the suite digest binds that capability/exclusion
contract. This is model agreement with `authority: none`, not
physical-GPU parity or performance evidence. Encoded responses fail closed
above the compiled 1 MiB limit.

The generic lane validates `examples/regression-manifest-v2.txt` against Cargo
workspace metadata and the HSACO names referenced by each example. The manifest
separates source-artifact inventory from explicit qualification authority. It
does not select production or GPU execution. `verus-vecadd` remains
ordinary-rustc-only and has no artifact qualification route.

CPU example tests are the exact manifest subset with `rustc_check=true` and
`artifact_qualification=none`. The `cpu-test-raw` and
`cpu-test-wrapper-managed` queries partition that subset using the structural
source projection: ordinary packages run with raw Cargo, while every package
containing a namespace-free typed kernel runs through the feature-free binding
wrapper using exactly:

```text
cargo fe2o3 test --locked --all-targets -p <wrapper-managed-package>
```

The generic CI partition retains mandatory `--all-targets` coverage. An explicit
library or named integration-test selection is not a substitute for that lane.
The host-test command rejects caller `--target`,
`--config`, Cargo-side `-Z`, `--doc`, and `--no-run` arguments. It also rejects
ambient runner and rustdoc selection plus configured runner, protected fe2o3,
dynamic-loader, and compiler selection. Configured rustdoc is overridden with
the disabled selection, and ambient loader variables are scrubbed. The lists are
sorted, disjoint, exhaustive, and both complete lists plus the full structural
projection are rescanned after the managed tests. This routing applies to any
kernel package; it does not encode package-name exceptions or require literal
namespaces.

Cargo-declared target roots must parse as ordinary Rust during projection.
Retained non-target `.rs` evidence, such as Verus closure fixtures, is still
bounded and revalidated; syntax outside ordinary Rust is treated as an
unresolved source edge and conservatively routes the package through the
binding wrapper.

Workspace source and configuration outside those protected selections, build
scripts, procedural macros, linkers, and test bodies are trusted and execute as
the current user. The fixed runner closes its child environment and descriptor
boundary, but it is not a sandbox. It opens and hashes Cargo's original test
executable. While Cargo's path remains stable, executing the retained original
preserves ordinary `current_exe` and `$ORIGIN` behavior and prevents
directory-entry substitution between pin and execution; the runner rechecks the
object afterward. This does not freeze same-inode writes or grant
immutable-artifact or origin authority. The protected-configuration scans before
and after Cargo are diagnostic checks for persistent changes, not an atomic
snapshot or TOCTOU proof.

This route may create ordinary Cargo host artifacts, but it has no fe2o3
backend, HSACO, publication, or artifact authority. It makes no GPU observation
or performance prediction. Because project code is trusted rather than
confined, those claims do not mean a test body is prevented from opening files,
network sockets, or device nodes.

The generic test subset runs `rustc-codegen-fe2o3` in a dedicated Cargo process.
The command-plan regression in `scripts/tests/ci-local-test-gate.sh` enforces
that separation in every generic CI run.

## Tutorial CPU References

Run one exact CPU suite declared by `config/tutorial-kernel-manifest-v1.json`:

```sh
scripts/run-tutorial-cpu-reference.sh examples/vecadd/Cargo.toml lib
scripts/run-tutorial-cpu-reference.sh examples/gfx950_advanced_attention/Cargo.toml test reference
```

Use `scripts/run-tutorial-cpu-reference.sh --batch` to observe all declared CPU
suites, up to the finite limit of 36. This covers the current 14 original suites
and 22 mode-specific additions. The total scheduler allowance is the sum of
suite deadlines: 43,200 seconds for 36 suites at 1,200 seconds each, compared
with 16,800 seconds for the previous 14-suite limit. The per-suite deadline
remains 1,200 seconds. The batch builds one private driver and gives each suite
an independent private target directory and its declared deadline. The entire
common setup cost counts against every suite's budget; earlier suite execution
does not count against later suites. Shared source, configuration and tool
inputs are checked between suites and after the batch. Failed, ignored, empty,
unavailable or invalid records prevent aggregate success. Batch observations
retain the same false qualification fields as single-suite observations.

The adapter validates the declared source contract, resolves the selected
standard Cargo test harness and default host features, builds the repository's
`cargo-fe2o3` driver, and runs its binding-aware host-test route. It never falls
back to raw Cargo tests or directly runs a discovered executable. An installed
pinned nightly and cached offline dependencies are required. Metadata,
bootstrap, execution and postflight share the declared 1,200-second deadline.
The private driver build uses at most two Cargo jobs; metadata and suite builds
remain at one job. This is a build scheduling bound, not a total thread or memory cap.

The JSON observation records the exact source/config/tool identities, commands,
selected harness, process results and individual test outcomes. Failure,
ignored tests, zero tests, filtered selections, unavailable prerequisites and
input drift cannot count as a complete successful suite. Named integration
targets may build metadata-bound companion binaries; those are not counted as
executed tests. Executable hashes are post-execution on-disk observations,
not attestations of the driver's executed image.

The ordinary binding-only host-test runner and this observer allow test images
up to 1 GiB, hashing in bounded chunks with the existing descriptor and file
identity checks. The default MoE oracle with debug information measured
909,946,760 bytes, above the separate 512 MiB tool/authority pin limit. That
shared default and transferred-descriptor limit remain unchanged. The host-test
allowance is fixed in source, grants no authority, and does not change build
profiles, assertions, overflow checks, per-suite deadlines or native limits.

Each run uses private scratch outside the checkout and removes its driver/build
artifacts. Bounded logs and `observation.json` remain in the reported evidence
directory. `FE2O3_TUTORIAL_CPU_OUTPUT_ROOT` may select an existing, canonical,
owner-private directory with mode `0700`. Source/configuration remain trusted;
before/after checks and process-group cleanup are not a hostile-code sandbox.

These are source-pinned CPU-reference observations only. All compiler,
simulation, policy, hardware and overall qualification fields remain false;
the tutorial manifest's `--require-qualified` gate is unchanged. The vecadd
library tests execute the existing shared kernel-body macro through safe host
adapters, not a copied arithmetic implementation or GPU launch.

## Kernel IR V12

The [Kernel IR V12 verification regression targets](kernel-ir-v12-verification.md#regression-targets)
cover the shared semantic engine, canonical admission, diagnostic work/storage,
and unsupported-consumer boundaries. These tests are separate from production
graph migration, tutorial-wide compilation, Verus qualification, and genuine
fresh-compiler KFD execution.

## Parity evidence validation

The generic lane runs legacy integrity checks plus signed row-evidence and
MI300X queue shell suites:

```text
scripts/ci-local.sh parity-evidence
bash scripts/tests/parity-evidence.sh
bash scripts/tests/parity-row-evidence.sh
bash scripts/tests/mi300x-evidence-queue.sh
bash scripts/tests/parity-dashboard.sh
```

The row suite runs class-specific test programs, signs test-domain result and
review fixtures, and exercises trust substitution, signature mutation, replay,
relabeling, stale source, duplicate identities, exact policy classes, queue
bypass, Complete authorization, and source-tree deltas. The queue suite uses no
GPU and makes no hardware claim. It consumes a signed test queue and verifies
canonical admission, lock hardening, and concurrent serialization.

See [Signed Parity Evidence V2](parity-signed-evidence-v2.md) for protected
trust provisioning, row-sharded commands, and the production MI300X queue.

## Comprehensive workspace tests

Run every workspace test target with the repository gate:

```text
scripts/ci-local.sh workspace-test
```

This is equivalent to the following required process boundary:

```text
cargo test --locked --workspace --all-targets \
  --exclude rustc-codegen-fe2o3 \
  --exclude fe2o3-artifact-transaction \
  --exclude <binding-managed-example>...
cargo fe2o3 test --locked --all-targets -p <binding-managed-example>...
scripts/ci-local.sh rustc-codegen-test
```

The gate computes `<binding-managed-example>` as the exact intersection of the
manifest's Rust-tested examples and the sealed source projection, then
revalidates that projection after testing. This keeps raw Cargo for ordinary
workspace packages while ensuring namespace-free typed kernels never depend on
an ambient binding.

`rustc-codegen-fe2o3` has `crate-type = ["rlib", "dylib"]`. An `--all-targets`
Cargo process can build more than one backend variant while integration tests
link the unversioned `librustc_codegen_fe2o3.so`. A later variant can replace
that file and leave an integration-test binary expecting Rust symbols from the
earlier variant. The failure then appears as an undefined dynamic symbol even
though the test passes by itself. The repository command runs the library test
target, the `fe2o3-rustc-extract` and `fe2o3-export-sim` binary targets, and every
integration target in deterministic separate Cargo
invocations. Each test therefore executes against the exact dylib produced for
its link, without changing compiler or crate behavior.

`generic-core` runs the backend library and both binary unit-test suites once;
integration shards run only their assigned integration targets. The binary
suites use default features and the production limited-debug profile. These
unit regressions do not replace source-to-simulator, proof, or GPU qualification.

`scripts/ci-local.sh backend` checks both default and all-feature package builds,
one at a time. Each build uses limited debug information and no incremental
compilation, and binds the actual Cargo JSON artifact to the workspace package
and source. The gate enforces the runtime's existing 1 GiB backend bound,
ELF64/ET_DYN shape, and a bounded metadata-only load with the selected rustc.
The compiler is selected by `RUSTC` or the workspace's `rustc` proxy and set
explicitly for Cargo as well; configured compiler wrappers are disabled for
these host backend builds. No stripping or runtime limit change is applied.

The checker reports bytes, headroom, and SHA-256 only after descriptor and path
identity plus content checks succeed before and after loading. This is a CI
artifact/loadability check, not sealed runtime custody, GPU code generation,
source qualification, or launch authority. The checker regressions and both
build/load checks run in the normal `generic-core` pipeline.

The comprehensive lane may link ROCm libraries through workspace packages. It
does not opt in to ignored GPU execution tests.

Passing this lane establishes operator-selected reviewed integrity for one
host-specific compiler/code-object profile. It does not authenticate origin,
run the GPU, establish source/Verus-to-machine refinement, prove memory safety
or race freedom, or grant publication, load, launch, or parity authority.

## External rollback crash and replay coverage

The external monotonic service and the local compiler Worker ledger exercise
opposite sides of the same restart protocol:

```text
cargo test --locked -p fe2o3-external-anchor-service --all-targets
cargo test --locked -p fe2o3-broker-authority-service --lib -- --test-threads=1
bash scripts/build-static-external-anchor-service.sh
```

The first suite interrupts the real atomic state replacement before and after
cleanup, create, write, file sync, rename, and directory sync, and interrupts
the daemon before and after receive, durable exchange, and send. Every restart
must recover exactly the prior or proposed state and exact request replay must
terminate at one proposed head. The broker suite covers the complementary local
journal orderings, queued-response recovery, exact challenge re-emission,
anchor-committed restart, and every retained-record boundary. These tests prove
deterministic same-host protocol recovery. Production rollback authority still
requires the root-managed distinct-UID anchor to be wired into the deployed
supervisor and qualified as an independently administered service.

## Retained bounded MoE evidence

The obsolete MoE V1/V2 host routes and their workload-specific hardware
launchers have been removed. Validate the remaining proof, source, and production-route-absence evidence with:

```text
python3 scripts/test-bounded-moe-docs.py
VERUS=/absolute/path/to/pinned/verus \
  ./scripts/test-moe-expert-compact-plan-verus.sh
cargo test --locked -p fe2o3-host --test production_application_handoff_ui
```

These commands do not execute MoE on the GPU. A production hardware claim must
flow through the generic Worker V3 route.
## `90b6fe3` multi-kernel checkpoint

The checkpoint at
`90b6fe31cbb1d89b82755f194ac7950c4eef4756` has focused CPU-only coverage for
the two-kernel contracts at each layer:

```text
cargo test --locked -p cargo-fe2o3 --test general_two_kernel_project
cargo test --locked -p dialect-amdgcn
cargo test --locked -p fe2o3-artifacts
cargo test --locked -p fe2o3-verifier
cargo test --locked -p fe2o3-host
cargo test --locked -p fe2o3-hsa-runtime
scripts/ci-local.sh rustc-codegen-test
```

These tests cover deterministic root/helper import, exact internal-helper call
signatures, two-kernel AMDGPU lowering, one-payload bundle validation,
per-kernel proof binding, typed host selection, distinct native HSA symbol
resolution, and borrow-enforced executable lifetime. Mutation and UI tests
reject ambiguous helpers, signature changes, payload/proof/kernel
substitutions, duplicate native identities, cloning linear selections, and
unloading while a kernel set remains live.

The real-source Worker V2 integration from this checkpoint is retired: its
backend selector no longer exists. Its retained oracle logic is exercised only
inside the feature-enabled library test binary and is not current
compile/publication or hardware evidence.

## Historical alpha/zeta Worker V2 observation

Commits `daf0b459ced07a25376670c83b1474eaebcd1a68` and
`dc9738e367c392f7716eacb8459ca73fa32abbbb` recorded compiler, raw HSA, and
fake-authority generated-safe observations for one exact two-kernel COV6
artifact. The Worker V2 host admission graph, exact alpha/zeta adapters,
hardware harness, and optional parity shard have since been deleted. Those
observations remain historical evidence and their commands are intentionally
not current test instructions.

Current feature-free host coverage uses the generic Worker V3 contract and the
direct-KFD composition boundary:

```text
cargo test --locked -p fe2o3-host
cargo test --locked -p fe2o3-host --test generated_spi_ui
cargo test --locked -p fe2o3-host --test worker_v3_verification_admission_ui
cargo test --locked -p fe2o3-host --test production_application_handoff_ui
cargo test --locked -p fe2o3-macros --test typed_kernel_fixtures
```

The deprecated HIP-buffer/HSA-lifecycle surface is retained only for explicit
qualification and differential coverage. It is absent from the default host
dependency graph and must be selected by name:

```text
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo test --locked -p fe2o3-host \
    --features qualification-legacy-hip-hsa
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo test --locked -p fe2o3-host \
    --features qualification-legacy-hip-hsa \
    --test hsa_executable_lifecycle_ui
```

With a configured ROCm development installation, compile the deprecated native
HSA/HIP qualification API and every feature-gated hardware test target without
executing them:

```text
cargo test --locked -p fe2o3-hsa-runtime --all-targets --features hardware-qualification --no-run
```

This checks only that the legacy qualification surface and its test harnesses
compile and link. It does not initialize HSA/HIP, admit a code object, or execute
a kernel, and it is not production direct-KFD qualification.

The macro fixture suite also checks the generated pure-KFD argument boundary:
safe code cannot implement its unsafe generated trait, mutable host outputs
remain exclusively borrowed, and HSA-backed and KFD-backed `Arguments`
specializations cannot satisfy each other's generated trait.

The generated KFD composition lane now covers authenticated Worker V3 typestate,
generated dispatch, completion, and the KFD runtime with a synthetic verifier.
The remaining production hardware gate must use the reviewed production
verifier through the inherited KFD application transition and must obtain its
artifact from the same build/publication transaction. A synthetic verifier,
externally injected HSACO, or selected legacy route cannot satisfy that gate.

### Archived compiler evidence controller

The checked controller and fixtures preserve the hardening and identity model
used by the historical bounded alpha/zeta Worker V2 observation. New captures
are disabled because their generator depended on a retired backend selector.
`scripts/gfx942-cov6-compiler-evidence.sh` therefore fails closed in capture
mode; its mutation self-test remains available for the archived format.

A replacement compiler-evidence controller must consume a production Worker V3
artifact and cannot grant runtime authority. New hardware qualification must
use KFD rather than the historical HSA harness.

## Verus proof coverage

Run the proof harnesses and expected-rejection mutations with the two exact
Verus installations pinned by the runtime-model and MIR/PLIRON suites:

```text
FE2O3_RUNTIME_MODEL_VERUS=/absolute/path/to/runtime-model/verus \
VERUS=/absolute/path/to/mir-pliron/verus \
  scripts/ci-local.sh verus
```

`FE2O3_RUNTIME_MODEL_VERUS` defaults to `VERUS` for snapshots whose pin-owning
harnesses use one release. The runtime-model and MIR/PLIRON harnesses verify
their executable digest and version; the fixture harnesses reuse the latter
installation. An unavailable or substituted binary is a failure rather than a
skip. The production `f32` vecadd and Verus expand the exact same control,
index, guarded memory-access, and write body. The three real-body mutations
require exactly one primary Verus error, one verification result reporting one
error, and the expected diagnostic, failed source clause, and marker. The
remaining mutations require their expected obligation marker and diagnostic.

The lane proves source-level models under the documented thread-witness
contracts. Verus uses a total arithmetic adapter for vecadd; it does not prove
that production IEEE `f32` addition, ordinary Rust, the compiler pipeline,
HSACO, HIP, or GPU execution refines that model.

## ROCm compile coverage

Install the [root requirements](../README.md#requirements) first. Before either
matrix, complete the selected examples' network-enabled locked dependency setup
in the same effective `CARGO_HOME` for the root workspace, each selected example
manifest, and the pinned nightly `rust-src` workspace. Some selected adapters
unconditionally invoke Cargo with `--locked --offline`, even when the matrix
host has network access. Root workspace fetching alone does not guarantee that
the independent example and `rust-src` locked closures are cached. Follow the
existing per-example setup, such as the
[row-softmax instructions](../examples/row_softmax_general_v1/README.md#build-and-numerical-qualification).

Set the admitted LLVM target and run the bounded production compiler gates:

```text
FE2O3_TARGET=gfx942 scripts/ci-local.sh rocm-compile
```

Compilation does not execute a kernel. The lane runs the selector-free
production extraction, general-matrix, transaction, ranked-bounds, and barrier
drivers, including their fail-closed cases, then compiles and inspects the
independent hardened G1 code-object fixture. It builds one private,
content-addressed `cargo-fe2o3` production driver for `doctor`; neither that
driver nor any backend child receives a qualification selector.

The site-backed general-kernel regression matrix compiles the current dynamic
GEMM, row-softmax, FlashAttention, grouped-expert MoE, and GEMM autoresearch
sources through the production extractor, external ROCm finalization, and a
nonempty gfx942 HSACO check:

```text
scripts/kernel-compile-matrix.sh gfx942
```

The `rocm-compile` lane runs this matrix after its focused production compiler
and code-object tests. The generic lane runs the matrix's hermetic shell test,
which substitutes fixture tools and verifies orchestration without requiring
ROCm or a GPU.

The separate gfx950 matrix covers 84 compile-only fixture selections represented
by the kernels site. Together with the 11 gfx942 selections, the compile-only
matrices cover 95 manifest-enumerated fixtures:

```text
scripts/kernel-compile-matrix.sh gfx950
```

Low precision: `kernel-fp4-gemm`, `kernel-fp8-gemm`,
`kernel-fp4-attention`, and `kernel-fp8-attention`.

Advanced attention: `kernel-kda-decode`, `kernel-kda-decode-baseline-v1`,
`kernel-kda-prefill`, `kernel-kda-prefill-baseline-v1`,
`kernel-content-sparse-attention`,
`kernel-content-sparse-attention-reciprocal-reuse-v1`,
`kernel-deepseek-sparse-attention`, `kernel-compressed-hybrid-attention`,
`kernel-compressed-hybrid-attention-division-baseline-v1`,
`kernel-attnres-aggregate`,
`kernel-attnres-aggregate-explicit-reuse-v1`, `kernel-four-branch-residual`,
`kernel-four-branch-residual-explicit-v1`, `kernel-mhc-sinkhorn-mix`, and
`kernel-mhc-sinkhorn-mix-scalar-v1`.

Advanced systems: `kernel-moe-route`, `kernel-moe-expert-rank`,
`kernel-moe-expert-rank` with `expert-serial`, `kernel-combine-expert-ranks`,
`kernel-speculative-transaction` with its canonical and
`speculative-recompute-prefix` bodies, `kernel-qwen-ngram-gather` with its
canonical and `ngram-reverse-probe` bodies, `kernel-stage-gradient-shard`, and
`kernel-muon-update` with its canonical and `muon-broadcast16` bodies.

GPT-OSS: `kernel-gpt-oss-decode`, `kernel-gpt-oss-decode-router-serial`,
`kernel-gpt-oss-decode-held-fragments`,
`kernel-gpt-oss-decode-interleaved-stores`, `kernel-gpt-oss-router-component`,
`kernel-gpt-oss-attention-component`, and `kernel-gpt-oss-expert-component`.

Compile-only mode still validates the checked-in manifest and therefore
requires an exact reviewed Clang, LLD, and device-library closure. ROCm 7.2.1
remains the default. Select the checked-in ROCm 7.2.4 closure explicitly:

```text
ROCM_PATH=/opt/rocm-7.2.4 \
FE2O3_GFX950_OCML_MANIFEST="$PWD/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.4.manifest" \
  scripts/kernel-compile-matrix.sh gfx950
```

The manifest pins the observed ROCm 7.2.4 Clang, LLD, and all nine bitcode
inputs by SHA-256. This does not require an MI350X and does not inspect or
execute any GPU. The gfx950 matrix is intentionally not in the gfx942-only
`rocm-compile` lane.

Every target and artifact directory is created under one private temporary
root and removed when the command exits. The matrix deliberately stops before
the qualification host runner, reports `hardware_observed=false`, and makes no
numerical or execution claim. It does not cover the basic Cargo regression
manifest, proof/source-model-only packages, or HIP comparators. Selected
site-backed positive variants are listed above. Negative variants remain
excluded: systems `combine-transposed` and `stage-tile4` fail compiler safety
checks, the two route feature names have no live alternate Rust bodies, and
GPT-OSS `scalar-attention` and `pipelined-attention` are absent from the
admitted current-artifact table. Neither cross-compilation target is hardware
or numerical coverage.

## Hardware smoke

Hardware execution is deliberately opt-in:

```text
FE2O3_ALLOW_GPU_SMOKE=1 FE2O3_TARGET=gfx942 \
  scripts/ci-local.sh hardware-smoke
```

Native Linux requires read/write access to `/dev/kfd`; WSL `/dev/dxg` is not a
KFD substitute. The lane measures the pure-Rust KFD identity boundary, admits
every visible `gfx942:xnack-` device, maps and releases host-visible memory on
each device, and creates, validates, and destroys one isolated compute AQL
queue per device without packet submission in those all-device checks. A
separate step selects one admitted device, executes the repository-owned vecadd
fixture through the direct-KFD runtime, publishes a bounded range-only runtime
profile, and queries its exact capture identity and loss-free completeness.
This does not claim application-level Worker V3 dispatch coverage.

Supplying both diagnostic variables extends the lane with one exact-artifact
packet in an isolated selected-device process:

```text
FE2O3_ALLOW_GPU_SMOKE=1 FE2O3_TARGET=gfx942 \
FE2O3_KFD_DIAGNOSTIC_UNIQUE_ID=0x6ced1647a296545c \
FE2O3_TEST_SOURCE_AUTH_LDS_GFX942_HSACO=/absolute/path/to/lds.hsaco \
  scripts/ci-local.sh hardware-smoke
```

This lane does not build or run every registered tutorial kernel or SIMT/tile
variant. Its individual fixture checks retain their own output comparisons;
device/queue checks are not numerical kernel results. Use the
[tutorial reproduction plans](tutorial-reproduction.md) to inspect the separate
manifest-declared inputs and correctness obligations without executing them.

The scalar-GEMM compiler lane extracts the real single-source
`#[kernel(typed)]` example through semantic MIR, ranked PLIRON, Kernel IR, and
the production gfx942 LLVM lowering:

```text
FE2O3_HIP_SYS_DISABLE=1 \
cargo test --locked -p rustc-codegen-fe2o3 \
  --test production_general_matrix_driver_v1 \
  scalar_gemm_kernel_reaches_gfx942_llvm \
  -- --ignored --exact --nocapture --test-threads=1
```

The test requires the pinned nightly `rust-src` component and AMD target. It
checks the exact compiler-derived crate binding, the generated scalar entry,
gfx942 target triple and thread-index intrinsic, separate `fmul` and `fadd`,
and absence of FMA or MFMA contraction. It does not link an HSACO, invoke the
protected verifier, or execute the GPU. The earlier externally injected,
scalar-specific hardware test was deleted with the workload-specific Worker V3
authority closure. A retained hardware claim now requires the ordinary generic
Worker V3 inherited application and KFD route.

The host crate enforces the same split. Its feature-free build exposes the
Worker V3 application, admission, verification, and private joined KFD
invocation. The HSA-backed generated migration route is available only through
the explicitly named deprecated qualification feature and cannot provide
production launch authority. Worker V2 application recovery, embedded-artifact
loading, direct HIP module/function loading, raw parameter packing, cooperative
launch, and workload-specific host adapters are absent from the feature-free
production host surface. `fe2o3-core` retains a separately named deprecated
unsafe-launch feature strictly for qualification.
`production_application_handoff_ui` compile-fails representative V2 entrypoint
and runtime imports, including the retired embedded artifact contract, raw HIP
surface, and generated vecadd `Kernel`, to guard that public API boundary. The macro fixture also
proves that every supported `#[kernel(typed)]` signature, including exact
vecadd and Scalar GEMM, emits only generic Worker V3 host code. It rejects the
retired `qualification_worker_v2` option and verifies lifetime retention,
private fields, non-cloneable arguments, hidden pointers, and one-shot dispatch
against the V3 API. The vecadd example has no embedded
execution feature; it type-checks the V3 `Arguments` surface and fails closed
before runtime dispatch until a production verifier is supplied.

The Cargo V3 vertical suite builds a dedicated V3-only static consumer. Its dependency graph contains the Worker V3 host consumer fixture and hardware test hooks, with no alternate compiler route:

```text
cargo test --locked -p cargo-fe2o3 \
  --features worker-v3-envelope-integration-test-only \
  --test worker_v3_load_envelope_vertical -- --test-threads=1
```

The suite constructs a typed V4 proof association inside the frozen V3
capsule, independently decodes all five compiler-stage preimages, reimports the
exact signed aggregate MIR-to-live-PLIRON receipt, and checks its middle-end V5
binding. Host admission reconstructs the exact finalizer derivation and the
protected verifier independently repeats that reconstruction from borrowed
canonical envelope bytes. The suite checks that the accepted move-only decision
retains matching finalizer custody and rejects a foreign compiler/finalizer
derivation even when its finalized HSACO bytes match. It also asserts that the
explicit synthetic verifier retains no production proof-input owner and cannot
report protected compiler/signature custody. On the 2026-08-30 MI300X
qualification tree, all 36 serialized cases passed, including
publication-turnover, compiler-evidence substitution, finalizer cross-splice,
seccomp, restart, load-cleanup, and descriptor-lineage failures.

## Guard tests

The native-Linux and WSL device-node selection logic has a host-only test:

```text
scripts/tests/ci-local-hardware.sh
```

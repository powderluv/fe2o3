# Getting started

This guide exercises fe2o3 without a GPU. It first runs a frozen Kernel IR V7
fixture, then exports an ordinary Rust kernel through the production
source/MIR/KIR stages and simulates the resulting bundle.

fe2o3 does not yet have a supported public production direct-KFD dispatch
command. The [GPU boundary](#gpu-boundary) describes that limitation; selected
[gfx950 engineering examples](#gfx950-engineering-evaluation) have separate
source-checkout compile and numerical evaluation commands.

## Requirements

- x86-64 Linux
- Bash and GNU `realpath`
- Git
- `rustup`
- a filesystem with enough space for a Rust compiler workspace build

The repository's [`rust-toolchain.toml`](../rust-toolchain.toml) pins
`nightly-2026-04-03`, `rust-src`, `rustc-dev`, `rustfmt`, `clippy`, and the
musl host target. Allow `rustup` to install those exact components when first
entering the checkout.

No GPU, KFD device, ROCm runtime, HIP runtime, or HSA runtime is needed for the
CPU simulator workflow.

A measured CPU-only `no-gpu` run on 2026-10-26 at
[332658a](https://github.com/harsh-nod/fe2o3/commit/332658add5b8cfb3b2ef0dd1ca736107a947e723)
used about 17.18 GB of allocated disk across the core and tutorial-site
checkouts, private Cargo cache, and fresh default build target. The pinned Rust
compiler was already installed and its storage is excluded. Peak memory was
about 10 GiB under a 16 GiB execution cap. This is one measured environment,
not a universal minimum or a GPU result; allow additional space for toolchain
installation, native products, retained outputs, and later builds.

## Clone and inspect

```console
git clone https://github.com/harsh-nod/fe2o3.git
cd fe2o3
rustc --version
cargo metadata --locked --no-deps --format-version 1 > /dev/null
```

The project currently builds from source. There is no tagged binary release or
crates.io installation promise yet.

## Inspect this host

The KFD-first doctor reports CPU onboarding, direct-KFD device/topology access,
compiler tools, and optional ROCgdb/rocprofv3 availability as separate facts:

```console
bash scripts/quickstart.sh doctor
```

The default diagnostic succeeds after printing explicit unavailable states, so
it is useful on a machine without a GPU. Closed requirement modes are available
for automation:

```console
bash scripts/quickstart.sh doctor --require-direct-kfd
bash scripts/quickstart.sh doctor --require-tools-present
bash scripts/quickstart.sh doctor --require-gfx942-and-tools-present
```

The tools-present requirements admit only executable files at the reported
paths. They do not validate versions or AMDGPU target capability; the actual
compile lane establishes those properties separately. `--require-execution`
always fails in this preview because the Worker V3
ordinary-application route is not wired. HIP and HSA are neither required nor
loaded by the doctor. ROCgdb and rocprofv3 are optional presence observations;
their reported paths, versions, and capabilities are not validated by the
doctor.

## Run from Rust source

The no-GPU quickstart exports ordinary Rust through the production source/MIR/KIR
stages and executes the resulting bundle on the CPU. It selects the default fill
with no extra features; the protected reference-proof runtime is not required:

```console
bash scripts/quickstart.sh no-gpu
```

After successful export, the script checks four exact `f32` outputs of `42.5`, two untouched trailing
canary elements, and the complete initialization state against a committed
independent expectation. It prints the simulator JSON only after the comparison
passes. The bundle and result are staged in a private temporary directory and
removed on success or failure. This does not publish HSACO, load a device,
dispatch a kernel, authenticate source execution, or establish CPU/GPU equivalence.

The same checker works with any admitted source kernel's complete output. The
explicit default fill invocation is:

```console
bash scripts/quickstart.sh simulate-source \
  --crate fe2o3_fill --request scripts/quickstart/fill-request.json \
  --expectation scripts/quickstart/fill-expectation.json \
  -- --package fe2o3-fill --lib
```

The separate `reference-proof` feature attaches the CPU reference to the same
kernel body. It requires protected proof execution and a conditional TotalView
continuation that is not yet wired. It fails closed, never falls back to the
default selection, and does not qualify the default manifest row. See the
[fill proof boundary](../examples/fill/README.md).

An expectation uses schema `fe2o3-simulation-expectation-v1` and contains exactly
`schema`, `arguments`, and `shared_buffers`. The last two fields specify the
complete returned values, including immutable inputs, buffer metadata, bytes,
and initialization bits. Scalar, buffer, view, and shared-buffer values use
the existing simulation Result V1 shapes. Comparisons are exact, including JSON
types; no floating-point tolerance or partial-output match is inferred. Generate
expectations from an independent reference or algorithm contract, not by copying
the result being tested. Without `--expectation`, `simulate-source` preserves
its existing execution-only behavior and performs no numerical comparison.

To check an already captured result without invoking a compiler:

```console
python3 -I -B scripts/check-simulation-expectation.py \
  --expectation expectation.json --result result.json
```

The checker rejects duplicate keys, malformed or non-finite JSON, non-regular
or symlink input files, oversized documents, incomplete outputs, and simulator
responses that do not explicitly report observation-only CPU execution.
Expectation files are limited to 16 MiB and result files to 64 MiB. A successful
comparison is an output observation, not compiler, proof, artifact, or launch
authorization. It does not qualify the pending SIMT/tile tutorial pairs.

## Run the exact KIR fixture

The checked-in fixture fills four `u32` elements with `17`:

```console
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo run --locked -q -p fe2o3-kir-sim-cli --bin fe2o3-kir-sim -- \
  --kir-v7 crates/fe2o3-kir-sim-cli/tutorial/fill-v1/kernel.kir \
  --request crates/fe2o3-kir-sim-cli/tutorial/fill-v1/request.json
```

A successful response is a `fe2o3-simulation-result-v1` JSON document. Check
these fields before interpreting it:

```json
{
  "status": "ok",
  "authority": "observation_only",
  "simulated": true,
  "hardware_observed": false,
  "hardware_validation": false,
  "performance_prediction": false
}
```

The full result also contains canonical KIR identity, execution and scheduling
counts, a semantic transcript identity, conflict assessment, and copied-back
argument bytes. The exact fixture result is committed at
[`expected-result.json`](../crates/fe2o3-kir-sim-cli/tutorial/fill-v1/expected-result.json).

## Exercise virtual runtime lifetimes

Run the same admitted fixture twice through persistent virtual allocations and
a serial completion dependency:

```console
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo run --locked -q -p fe2o3-virtual-runtime-cli --bin fe2o3-virtual-runtime -- \
  --kir-v7 crates/fe2o3-kir-sim-cli/tutorial/fill-v1/kernel.kir \
  --request crates/fe2o3-kir-sim-cli/tutorial/fill-v1/request.json \
  --repeat 2
```

The `fe2o3-virtual-runtime-result-v1` response identifies the model-only
runtime, admitted KIR, module, allocation, queue, both completions, the serial
dependency edge, copied bytes, and terminal cleanup states. It also states
`"simulated":true`, `"hardware_observed":false`, and
`"performance_prediction":false`.

This command starts from exact admitted KIR. For ordinary Rust, use
`bash scripts/quickstart.sh no-gpu`, or supply an exported authority-free bundle
in place of `--kir-v7 PATH` with `--bundle PATH`. Read the
[virtual runtime contract](virtual-runtime-v1.md) for limits, typed misuse and
ambiguous-completion behavior, and the remaining issue #216 criteria.

## Run the component commands

The `fe2o3-export-sim` binary reuses the production source/MIR/KIR stages under
extraction-only custody. It produces a content-addressed bundle and cannot
publish HSACO, load a device, or dispatch a kernel.

The component commands below select default
[`examples/fill`](../examples/fill/src/lib.rs), without `reference-proof`:

```console
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo build --locked -q -p rustc-codegen-fe2o3 \
  --bin fe2o3-rustc-extract
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo run --locked -q -p rustc-codegen-fe2o3 --bin fe2o3-export-sim -- \
  --crate fe2o3_fill \
  --output "$PWD/fill.fe2sim" \
  --target gfx942 \
  -- --package fe2o3-fill --lib
```

The output name must not already exist. The exporter creates it with mode
`0600` and fails on path substitution or unsupported source semantics.

Create a request for four `f32` output elements:

```console
cat > fill-request.json <<'JSON'
{"schema":"fe2o3-simulation-request-v1","kernel":"fill","grid":[4,1,1],"workgroup":[64,1,1],"arguments":[{"kind":"buffer","element":"f32","access":"read_write","alignment":4,"bytes":"0x00000000000000000000000000000000"}]}
JSON
```

Execute the bundle:

```console
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo run --locked -q -p fe2o3-kir-sim-cli --bin fe2o3-kir-sim -- \
  --bundle "$PWD/fill.fe2sim" \
  --request "$PWD/fill-request.json"
```

Each copied-back element should have the little-endian IEEE-754 bytes
`00002a42`, which represent `42.5f32`. The simulator evaluates supported
floating-point operations with pinned software IEEE semantics rather than host
`f32` arithmetic.

Remove local tutorial outputs when finished:

```console
rm -f fill.fe2sim fill-request.json
```

For bundle versions, source maps, request limits, schedule recording, seeded
schedule exploration, and typed unsupported states, read
[Source-to-simulator bundle V1](simulation-bundle-v1.md) and the
[`fe2o3-kir-sim` CLI reference](../crates/fe2o3-kir-sim-cli/README.md).

## Explore the debugger

The simulator debugger consumes an immutable transcript through a versioned
JSONL protocol. Run the complete checked-in request stream:

```console
FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
  cargo run --locked -q -p fe2o3-debug-cli --bin fe2o3-debug -- \
  sim \
  --kir-v7 crates/fe2o3-kir-sim-cli/tutorial/fill-v1/kernel.kir \
  --request crates/fe2o3-kir-sim-cli/tutorial/fill-v1/request.json \
  --protocol jsonl \
  < crates/fe2o3-debug-cli/tutorial/fill-v1/requests.jsonl
```

Responses expose dispatch, workgroup, logical wave, and work-item scopes;
operation and event navigation; SSA and allocation-relative memory; typed
availability; and immutable session revisions. Logical waves are semantic
visualization partitions, not claims about physical GPU wave state.

The [debugger CLI contract](../crates/fe2o3-debug-cli/README.md) describes
breakpoints, watchpoints, source maps, stack/variable inspection, reverse replay,
failure diagnosis, live KFD observation, ROCgdb MI, and agent-facing custody.
The [interactive tutorial](https://harsh-nod.github.io/fe2o3-kernels/#/lesson/cpu-semantic-simulation)
visualizes selected golden transcripts and their provenance.

## GPU boundary

Current production direct-KFD qualification is bounded to MI300X
`gfx942:xnack-` lanes. A
machine used for repository qualification needs Linux KFD access, the pinned
ROCm/LLVM tool environment for the selected lane, and permission to open the
relevant `/dev/kfd` and DRM render nodes.

Do not treat these commands as a public GPU quick start:

```console
FE2O3_TARGET=gfx942 bash scripts/ci-local.sh rocm-compile
FE2O3_ALLOW_GPU_SMOKE=1 FE2O3_TARGET=gfx942 \
  bash scripts/ci-local.sh hardware-smoke
```

They are developer qualification lanes with narrower claims. The ordinary
`fe2o3-fill` and `fe2o3-vecadd` applications intentionally fail before dispatch
because their production Worker V3 application verifiers are not wired. The
repository also lacks a generally deployable protected release service for an
external contributor.

A community-ready GPU quick start must compile an ordinary kernel from a clean
checkout, authenticate its current artifact and launch contract, execute it
through direct KFD, verify its output, and cleanly tear down. The separate
engineering examples below do not satisfy that production acceptance criterion.

## gfx950 engineering evaluation

These are explicit example-specific workflows, not a fallback from a denied
production command. They compile ordinary Rust sources, inspect the produced
target/ABI/ISA, and optionally compare actual GPU outputs against independent
CPU oracles. A numerical pass is not a source-to-machine refinement proof,
protected publication receipt, or general direct-KFD application authorization.
It does not close the whole-curriculum acceptance in
[#275](https://github.com/harsh-nod/fe2o3/issues/275).

Start from the repository root and follow the selected example's complete
setup before its commands. Each example documents its actual `simt` and `tile`
selections, oracle, supported shapes and remaining evidence limits:

| Example | Maintained setup and commands | Numerical host path |
| --- | --- | --- |
| BF16 dynamic strided GEMM | [Run on gfx950](../examples/tiled_gemm_general_v1/README.md#run-on-gfx950), using `run-gfx950.sh` with `simt` and `tile` | Explicit external-HSACO unsafe qualification host |
| BF16 Flash Attention | [Paired commands](../examples/flash_attention_general_v1/README.md#paired-commands) and [gfx950 setup](../examples/flash_attention_general_v1/GFX950.md), using `run-gfx950.sh` with `simt` and `tile` | Explicit external-HSACO unsafe qualification host |
| BF16 grouped routed expert | [gfx950 paired commands and setup](../examples/moe_grouped_expert_general_v1/README.md#gfx950-paired-commands), using `run-gfx950.sh` with `simt` and `tile` | Explicit external-HSACO unsafe qualification host |
| FP8 and FP4 GEMM | [Setup and Rust-to-HSACO runners](../examples/gfx950_low_precision/README.md#setup), using `run-fp8-gemm-gfx950.sh` and `run-fp4-gemm-gfx950.sh` with `simt` and `tile` | Digest-pinned HSA numerical fixtures |
| F32 vector addition | [Vecadd setup and numerical commands](../examples/vecadd/README.md), using `run-gfx950.sh` with `simt` and `tile` | Digest-pinned HSA numerical fixtures |

For these runners, `FE2O3_EXAMPLE_COMPILE_ONLY=1` suppresses GPU execution;
successful compilation and ISA checks must not be reported as a numerical
run. For actual execution, use a physical `gfx950:xnack-` Wave64 device with
the required access and visibility. An MI300X/gfx942 device, CPU simulation,
or a target string cannot substitute for that device. Do not run competing
GPU jobs on a shared device without its owner's execution lease.

The setup is path-specific. Use the pinned `nightly-2026-04-03` toolchain and
the selected runner's documented ROCm LLVM/runtime development libraries.
Where a digest-pinned tool/bitcode manifest is required, select the matching
checked-in ROCm 7.2.1 or 7.2.4 closure exactly as that README describes.
Changing only `ROCM_PATH` does not make different tool or library bytes valid.

Fetch the locked root, selected standalone example, and required nightly
`rust-src` dependencies in the same effective `CARGO_HOME` before an offline
step. A root-workspace fetch alone does not populate every example or
`-Zbuild-std=core` closure. See the existing
[compile-matrix prerequisites](testing.md#rocm-compile-coverage).

Check free persistent storage for the checkout, Cargo cache, build targets and
retained outputs before building. Available RAM does not compensate for a full
filesystem. Cold compiler/host builds have a separate resource cost from a warm
numerical run; follow the example's declared profile and prebuild instructions
instead of applying an undocumented profile or reusing a private compiler.
Changing profiles or output directories can trigger a rebuild. No universal
minimum memory/disk requirement or bounded cold-build time is established by
these example instructions.

Keep each complete command, source revision, compiler/tool identities, target,
oracle result and first failure diagnostic with its output. Historical receipts
retain their original revision and cannot qualify a changed source, variant,
target, or toolchain merely because the example name is unchanged.

## Next steps

- Inspect [target-specific tutorial reproduction plans](tutorial-reproduction.md)
  before choosing a registered SIMT/tile/U32 source or interpreting a result.
- Check the [support matrix](support-matrix.md) before selecting a target or
  tool workflow.
- Read [Architecture V2](architecture-v2.md) for compiler and runtime ownership.
- Use the [testing guide](testing.md) to select a contributor validation lane.
- Read [`CONTRIBUTING.md`](../CONTRIBUTING.md) before opening a pull request.
- Report bugs with the complete command, revision, target, toolchain versions,
  and the first fail-closed diagnostic. Do not publish private keys, native
  addresses, protected receipts, or sensitive profiler/debugger captures.

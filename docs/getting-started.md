# Getting started

This guide exercises fe2o3 without a GPU. It first runs a frozen Kernel IR V7
fixture, then exports an ordinary Rust kernel through the production
source/MIR/KIR stages and simulates the resulting bundle.

fe2o3 does not yet have a supported public source-to-GPU dispatch command. The
[GPU boundary](#gpu-boundary) explains what can be evaluated on MI300X today.

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

Current hardware qualification is bounded to MI300X `gfx942:xnack-` lanes. A
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
through direct KFD, verify its output, and cleanly tear down. Until that exists,
the absence of a GPU command in this guide is intentional.

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

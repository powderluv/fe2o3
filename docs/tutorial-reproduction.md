# Reproduce A Tutorial Selection

Start with the existing [no-GPU quickstart](getting-started.md), then inspect the
registered source selection below. There is still **no supported public
source-to-GPU dispatch command**. A compiler service installer is not a complete
verified-U32 compiler package, proof runtime, native admission or GPU launch path.

## Pin The Source And Inspect Without Executing

Use Python 3.11 or newer. These commands read and validate source contracts;
they do not invoke Cargo, a compiler, simulator, proof engine or GPU:

```sh
git rev-parse HEAD
git status --short
python3 -I -B scripts/validate-tutorial-kernel-manifest.py --require-curriculum
python3 -I -B scripts/validate-tutorial-kernel-manifest.py \
  --emit-reproduction-plan gfx942 > /tmp/tutorial-gfx942-plan.json
python3 -I -B scripts/validate-tutorial-kernel-manifest.py \
  --emit-reproduction-plan gfx950 > /tmp/tutorial-gfx950-plan.json
python3 -m json.tool /tmp/tutorial-gfx942-plan.json
```

Record the checkout commit with the plan's raw `manifestSha256`. Each fixture
retains its existing `compilerInput`: Cargo package/target, default and explicit
features, source closure, lockfile and contract digests. Those are expected
inputs, not authenticated identities of an installed compiler executable.
`installedCompiler` is deliberately null.

The plan is a derived view of
[`config/tutorial-kernel-manifest-v1.json`](../config/tutorial-kernel-manifest-v1.json),
not a second registry. It runs the same physical-input validation as the existing
matrix before emitting bounded JSON. Invalid or stale contracts fail without a
partial plan. Do not execute plan strings through `eval`.

## Select The Documented Architecture

| Selection | Host Family | Meaning |
| --- | --- | --- |
| `gfx942` | MI300X, such as `mi300x` | Only fixtures explicitly registered for gfx942 |
| `gfx950` | MI350, such as `mi350` | Only fixtures explicitly registered for gfx950 |

Do not replace a target string to manufacture another architecture's coverage.
When the same source has registered selections for both targets, inspect both
plans. Architecture-specific FP4/FP8/MFMA examples may have only a gfx950
selection. A missing row is not a pass, an automatic fallback or proof of
unsupported hardware.

The `fixtures` array retains the exact matrix selection and simulation
request/expectation identities. `declaredQualificationSuites` includes only
suites whose declared coverage intersects those fixtures, but preserves each
**whole** original suite and marks the intersecting IDs separately. A shared CPU
suite does not become architecture-specific or narrower just because a target
was selected.

## SIMT, Tile And Mixed Are Separate Obligations

Use the same validator's full identity projection:

```sh
python3 -I -B scripts/validate-tutorial-kernel-manifest.py \
  --emit-kernel-pairs > /tmp/tutorial-kernel-pairs.json
```

Read explicit source-bound variants, pending variants, negative cases and source
binding gaps. A source-bound SIMT variant does not supply a tile implementation;
a shared name, feature or lesson does not establish equivalence. Keep mixed,
formal, assembly and megakernel obligations visible. The reproduction plan
covers registered compiler fixtures, not every displayed tab or source-driver
case. Use the [pair contract](tutorial-kernel-pairs-v1.md) for those additional
identities, including the source-bound U32 row kernel.

## Compile-Only Is Not Correctness

The plan's `compileOnlyMatrix.command` names the existing
[`kernel-compile-matrix.sh`](../scripts/kernel-compile-matrix.sh) runner. It
requires the pinned Rust/ROCm tool environment described in
[testing](testing.md); gfx950 additionally checks its exact registered device
library closure. It is not a GPU smoke command.

Compilation success means the selected compile-only corpus produced its
temporary artifacts. It does not establish numerical correctness, prove
CPU/GPU equivalence, admit protected execution, or run all tutorial variants.
Proof-dependent and expected-refusal cases keep their separate prerequisites.

## Compare A Real Result

Keep `executionStatus: "UNRUN"`, `qualified: false` and
`hardwareExecuted: false` in a static plan. Never edit those into a receipt.

For the no-GPU quickstart, retain an independently specified expectation and use
the existing checker on an actual simulator result:

```sh
python3 -I -B scripts/check-simulation-expectation.py \
  --expectation scripts/quickstart/fill-expectation.json \
  --result /absolute/path/to/actual-result.json
```

Exit zero means the complete returned arguments and shared buffers match exactly,
including initialization and canaries. A nonzero exit or missing result is not a
pass. This checker cannot turn an old bundle or source revision into a current
source/compiler result.

Declared CPU-reference suites retain their exact commands in the plan. The
existing adapter emits `fe2o3-tutorial-cpu-reference-observation-v1` JSON:
require exit zero and `outcome: "passed"`, retain its input/command evidence and
all false authority fields. A host reference test is not source compilation,
simulation or hardware qualification.

The [current source-to-simulation runner](compiler/tutorial-current-qualification-v92.md)
requires its whole registered corpus and protected prerequisites; do not
reinterpret a selected plan or partial report as its completion. Its scoped
success needs exit zero, `complete: true` and
`currentCompilerSimulationPassed: true`; `qualified` remains false.

## Verified-U32 And Hardware Boundary

Existing formal U32 claims concern exact source/model obligations and premises,
not every kernel, the current compiler's generated machine code, or GPU execution.
A verified-U32 package still needs its complete pinned compiler inputs, proof
runtime, artifact/native admission and independent correctness oracle. There is
no supported full public package path to recommend from a service-only installer.

The existing [hardware smoke lane](testing.md#hardware-smoke) has its own opt-in,
target and device requirements and narrower fixture claims. It does not run all
registered SIMT/tile/U32 kernels; it is not a generic newcomer launch adapter.
The plan preserves each target's declared hardware availability and supplies no
invented hardware command. Real target-matched results remain **UNRUN/unknown**
until their own authorized invocation, source/artifact identity and complete
independent output comparison are retained.

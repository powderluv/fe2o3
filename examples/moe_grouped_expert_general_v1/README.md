# Grouped Routed Expert Compute

The structured tile default and ordinary SIMT counterpart project the same
routed BF16 tokens through one selected expert's BF16 weights, then compute
`gate[row] * (projection[row, column] + bias[expert, column])` in F32.
Tokens and weights remain genuine BF16 `u16` storage. SIMT uses direct
`Bf16::from_bits(...).to_f32()` operations and increasing-depth F32 MACs,
not a bespoke decoder or F32-only replacement input. No MFMA, subgroup or LDS
operation implements the scalar path. The tile path uses BF16 MFMA with F32 accumulation.

## gfx950 Paired Commands

Use nightly-2026-04-03 with the components in the repository's
[toolchain requirements](../../docs/getting-started.md#requirements). Cargo uses
the committed lockfiles and its normal dependency resolution; offline commands
require those inputs to be fetched first.

The tested compile configuration uses ROCm 7.2.4 LLVM tools (`clang`, `ld.lld`,
`llvm-objdump`, `llvm-readobj`). Numerical GPU runs additionally require that
installation's HIP/HSA runtimes and a gfx950 Wave64 device. `ROCM_PATH` defaults
to `/opt/rocm`; point it at the ROCm 7.2.4 installation being used.

From the repository root:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/moe_grouped_expert_general_v1/run-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/moe_grouped_expert_general_v1/run-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/moe_grouped_expert_general_v1/run-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/moe_grouped_expert_general_v1/run-gfx950.sh simt
```

The package-local runner compiles for gfx950:xnack- Wave64, verifies the actual
target and BF16 MFMA instruction for tile (and absence of matrix instructions
for SIMT), and retains LLVM, ISA, notes and the compiler crate binding.
The host rejects an unsupported or mismatched requested/observed target before
loading the module. Full runs check every logical output and stride padding,
eight trailing bitwise canaries, and all immutable input buffers.
Compilation is not hardware validation; record actual paired full-run results
separately before claiming gfx950 qualification. This public candidate has no
fresh GPU qualification; historical private-source results do not qualify it.

## Existing gfx942 Commands (GPU Validation Pending)

Use the same pinned toolchain and dependency setup above. From the repository root:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/moe_grouped_expert_general_v1/run-gfx942.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/moe_grouped_expert_general_v1/run-gfx942.sh simt
bash examples/moe_grouped_expert_general_v1/run-gfx942.sh tile
bash examples/moe_grouped_expert_general_v1/run-gfx942.sh simt
```

Omitting the selector retains tile. Closed, mutually exclusive features select
exactly one AMDGPU export. The existing common runner forwards the selected
Cargo arguments to extraction and the host executable; no runtime feature is
enabled for simulation. Target, source binding and qualification boundaries
remain in that unchanged runner.

## Contract And Validation

Both paths retain WG64 and use gfx942 or gfx950 Wave64 and the existing explicit 120-byte ABI:
five pointer/length slice pairs (80 bytes) and ten `u32` scalars (40 bytes).
Dynamic row, column, reduction, expert identity/count, physical strides,
selected-expert offsets and complete input extents use the original checks.
The scalar reduction bound remains 4,294,967,295, not the fixture's reduction.
Tiled2D ownership clips partial column tiles and preserves physical stride
padding. Zero gates still multiply the full epilogue, preserving IEEE behavior.

The original full fixture remains 32 padded routed rows, 21 columns, reduction
32, four experts with expert 2 selected, nonquarter BF16 weights, 19 active
route gates and distinct token/weight/bias/output strides. Every logical output,
including zero-gated rows, must be finite and meet the unchanged
`max(5e-3, abs(expected) * 5e-3)` tolerance; output padding is bitwise checked.
The shared host/reference validator rejects nonfinite actual/expected values
and errors, so NaN cannot pass through an unordered tolerance comparison.
Reference tests additionally cover independent exact-rational expert selection,
gated bias, signed/zero gates, cancellation, tails and padding.

The scalar reduction order differs from MFMA; unchanged numerical tolerances,
not bitwise/native/protected proof equivalence, define fixture comparison.
New source and authored tests require actual compilation and execution before
any pass claim. gfx942 GPU validation remains pending. These are target-specific qualification commands, not model serving or release.

Host tests use the repository's [supported cargo-fe2o3 workflow](../../README.md#typed-kernel-authoring):

```bash
cargo fe2o3 test --locked --manifest-path examples/moe_grouped_expert_general_v1/Cargo.toml --all-targets
cargo fe2o3 test --locked --manifest-path examples/moe_grouped_expert_general_v1/Cargo.toml --all-targets --no-default-features --features kernel-simt-moe-expert
```

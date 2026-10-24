# Dynamic BF16 Flash Attention

The existing structured tile kernel and ordinary SIMT counterpart use real BF16
`u16` Q/transposed-K storage and F32 V, additive masks, accumulators and outputs.
Both retain the 144-byte explicit ABI, Wave64/WG64 and tiled disjoint ownership.
The tile pipeline/geometry are retained; four mask score selections now exclude
nonfinite logical masks before softmax, matching the established reference.
SIMT uses the established device BF16 widening operation, two
increasing-depth score passes and max-subtracted F32 softmax/PV without MFMA,
subgroup operations or LDS. Its five loop bounds cover four owned components,
up to 4,096 logical keys and depth 1,024 in both passes, not just the fixture.

## Paired Commands

Use the repository's pinned nightly-2026-04-03 and components described in
[Getting started: requirements](../../docs/getting-started.md#requirements).
Cargo resolves the committed lockfiles using its normal dependency setup; an
offline run requires those dependencies to be fetched first. From the repository
root, the gfx950 milestone commands are:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/flash_attention_general_v1/run-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/flash_attention_general_v1/run-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/flash_attention_general_v1/run-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/flash_attention_general_v1/run-gfx950.sh simt
```

See [gfx950 compilation and numerical execution](GFX950.md). Compilation alone
does not qualify GPU execution, and this public candidate has no fresh GPU result.

The gfx942 runner remains available; **gfx942 GPU validation is pending**:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/flash_attention_general_v1/run-gfx942.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/flash_attention_general_v1/run-gfx942.sh simt
bash examples/flash_attention_general_v1/run-gfx942.sh tile
bash examples/flash_attention_general_v1/run-gfx942.sh simt
```

Omitting the selector retains tile. The local runner forwards its closed Cargo
feature selection through the unchanged common runner, including the host run,
so the extracted symbol and loaded symbol agree. Device-library linkage,
source binding, target checks and qualification execution remain in that runner.
No runtime feature is required for CPU simulation.

The existing full host fixture is unchanged: two heads, 19 logical queries,
21 logical keys, depth 23, value dimension 13, distinct physical strides,
nonquarter BF16 inputs, a fully masked row and padded queries/keys/outputs.
Every logical result must be finite and satisfy the original 5e-3 absolute/
relative tolerance; output stride padding must remain bit-identical.
The shared host validator now rejects NaN/infinite outputs, expectations and
errors rather than allowing unordered comparisons to print PASS.

## Mask Semantics And Evidence

The existing reference excludes every nonfinite logical mask, including NaN
and positive infinity. Both kernel sources now explicitly apply that policy;
physical mask padding is never visited. Tile's four score selections were corrected
before its otherwise unchanged subgroup/online softmax. Padded V rows are
replaced with zero before weighted contributions, so NaN/infinity in those
rows cannot contaminate a result through a zero weight. The current fixture
uses finite or negative-infinity logical masks (NaN in padding).
Padded queries and fully masked rows produce zero; padded keys never contribute.

The reference helper retains its pre-existing full physical-buffer checks,
including mask stride at least padded keys. Kernels retain the original checked
minimal extents and mask stride at least logical keys. The fixture satisfies
both; this patch does not silently narrow the kernel's valid input layouts.

Source tests and references are prerequisites, not kernel execution. These new
paired commands require actual compilation/numerical execution before a pass
claim. Scalar reduction order differs from MFMA/online tile reduction; unchanged
tolerances, not bitwise/native/protected-proof equivalence, define fixture
comparison. Protected Worker publication remains separate.

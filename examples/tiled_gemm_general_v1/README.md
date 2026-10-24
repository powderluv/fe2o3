# Dynamic strided GEMM

This example is an attributed safe Rust GPU kernel candidate for

```text
C = alpha * A * B + beta * C
```

`M`, `N`, `K`, `lda`, `ldb`, `ldc`, `alpha`, and `beta` are runtime values.
In the structured tile variant, each wave64 workgroup owns one 16x16 output tile and executes
`V_MFMA_F32_16X16X16_BF16` for every 16-element K phase. Checked edge loads
contribute BF16 zero; checked tiled output witnesses suppress stores outside
logical M and N. Every active output applies the dynamic alpha/beta epilogue
once.

The K loop keeps current and next MFMA operand fragments live as a two-buffer
register pipeline and performs one speculative, zero-filled prefetch. This is
distinct from the target-neutral `kernel.pipeline` PLIRON protocol, which
verifies shared workgroup-storage ring lifecycles. The source frontend does not
yet synthesize that workgroup protocol from this Rust loop.

The matrix instruction is exposed through the target-neutral `DeviceMatrix`
capability. Bounds, uniformity, convergence, ranked indexing, and disjoint
output ownership are ordinary compiler analyses shared with every other kernel;
none of those passes recognizes GEMM or grants it a special case.

## Run on gfx950

From the repository root, install the pinned toolchain and fetch the locked
dependencies. ROCm LLVM and runtime development libraries must be available;
the runner defaults to `/opt/rocm` and accepts `ROCM_PATH`.
For a strict native/GPU budget, complete the CPU prebuild below before the final
two commands; a cold release host build has a separate ordinary CPU budget.

```bash
rustup toolchain install nightly-2026-04-03 --component rust-src --component rustc-dev
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_BUILD_JOBS=2
export CARGO_INCREMENTAL=0
cargo fetch --locked
cargo fetch --locked --manifest-path examples/tiled_gemm_general_v1/Cargo.toml
bash examples/tiled_gemm_general_v1/run-gfx950.sh tile
bash examples/tiled_gemm_general_v1/run-gfx950.sh simt
```

The default is tile. Each command compiles real Rust for gfx950:xnack- Wave64,
preserves the compiler source/proof and crate-binding checks, finalizes with
ROCm clang/LLD, checks target metadata and variant ISA, and executes the
matching host symbol. Tile requires actual BF16 MFMA; SIMT rejects MFMA/WMMA.
No device-library link is needed. FE2O3_EXAMPLE_COMPILE_ONLY=1 skips GPU work.

Keep the ordinary Cargo profile settings above for the default extractor and
host builds. For a warm-cache run with a bounded native execution budget,
prebuild the ordinary CPU phase first, using the default output directories:

```bash
out=examples/tiled_gemm_general_v1/target/fe2o3-gfx950
for variant in tile simt; do
  stem=tiled_gemm_general_v1
  selection=()
  if [ "$variant" = simt ]; then
    stem=simt_gemm_general_v1
    selection=(--no-default-features --features kernel-simt-gemm-general)
  fi
  FE2O3_EXAMPLE_COMPILE_ONLY=1 \
    bash examples/tiled_gemm_general_v1/run-gfx950.sh "$variant"
  binding=$(cat "$out/$stem.binding")
  env -u FE2O3_CARGO_METADATA_BUILD_OBSERVATION_V2 \
    FE2O3_CRATE_BINDING_ID_V1="$binding" CARGO_TARGET_DIR="$PWD/target" \
    FE2O3_GENERAL_GEMM_TARGET=gfx950 FE2O3_GENERAL_GEMM_HSACO="$PWD/$out/$stem.hsaco" \
    rustup run nightly-2026-04-03 cargo build --release --locked --offline -j2 \
      --manifest-path examples/tiled_gemm_general_v1/Cargo.toml \
      --bin fe2o3-tiled-gemm-general-v1 "${selection[@]}"
done
```

The binding is the actual compiler-produced handoff checked by the runner, not
a manually supplied identity. This prebuild runs no GPU program. The full
commands still perform their normal builds and reuse Cargo's cache; cold build
memory and time are separate from the warm native/GPU run. No private frontend
hash, extractor override or absolute scratch path is required.
After prebuilding, `CARGO_NET_OFFLINE=true` can be used for a no-network run.

The host checks the requested architecture, all strided outputs, unchanged
BF16 A/B inputs, output padding and canaries. It also executes the maintained
raw nonquarter, cancellation and fractional 16x16 cases with all 256 outputs
and eight canaries against independent rational expectations. F32 accumulation,
C and alpha/beta scales and the original 2e-3 tolerance are unchanged.

These are ordinary external-HSACO numerical observations, not protected
publication or native proof. Actual command results are required before
claiming gfx950 success. gfx942 GPU execution remains pending separately;
a gfx950 run cannot qualify gfx942.

## Run on gfx942

From `examples/tiled_gemm_general_v1` in the compiler repository checkout:

```bash
./run-gfx942.sh tile
./run-gfx942.sh simt
```

Omitting the argument selects `tile`. Both variants consume BF16 operands stored
as `u16` bits and use F32 accumulation, output, and alpha/beta scales. The tile
source is `src/kernel.rs`; the ordinary Rust SIMT source is `src/kernel_simt.rs`.
SIMT assigns one invocation to each physical output slot, including checked
padding slots. Its host selects the matching symbol and launch grid with
`--no-default-features --features kernel-simt-gemm-general`; the runner passes
that same selection to kernel compilation and the host executable.

The two outputs are `tiled_gemm_general_v1.hsaco` and
`simt_gemm_general_v1.hsaco` under `target/fe2o3-gfx942/`. These commands request
gfx942 qualification; they do not establish gfx950 support or replace each
variant's required compilation, proof, and numerical checks.

The script requests the complete qualification flow:

```text
safe Rust
  -> semantic MIR
  -> ranked PLIRON verification
  -> Kernel IR
  -> formal memory admission
  -> gfx942 LLVM
  -> external ROCm clang/LLD HSACO
  -> fe2o3-core unsafe qualification launch
```

The compiler now carries the checked-tiled source capability through the
dynamic-launch race proof and reaches gfx942 LLVM qualification.
`run-gfx942.sh` passes that LLVM through external ROCm clang/LLD to produce an
HSACO and runs the numerical gfx942 qualification path. Confirming that the
disassembly contains `v_mfma_f32_16x16x16_bf16` remains required before making
performance claims.

This script deliberately uses the external-HSACO unsafe qualification path. It
does not exercise protected Worker publication or artifact-currentness
admission; those remain a separate, fail-closed pipeline.

## Safety boundary

The standalone capability API owns its host UI tests. From the repository root:

```bash
cargo test --locked --offline --manifest-path examples/tiled_gemm_general_v1/device-api/Cargo.toml --test device_api_ui
```

This runs one passing and sixteen compile-fail capability cases. It does not
compile the attributed GEMM kernel or qualify its semantic proofs or GPU execution.

The library containing the kernel uses `#![forbid(unsafe_code)]`. Ordinary Rust
slice indexing and `DisjointSlice::get_mut` remain visible to the compiler, so
generic bounds and ownership analysis can verify them. The host binary contains
the two required documented unsafe operations: loading external machine code
and launching it with an exact physical ABI.

Any resulting HSACO is qualification output. Protected release publication
and artifact-currentness admission remain a separate, fail-closed pipeline.

## CPU-Only BF16 Source Observations (V6)

From the repository root, install the pinned toolchain and fetch the locked
dependencies before the offline extraction steps:
These commands build ordinary compiler/simulator binaries, not GPU execution.
The complete quickstart sequence below still needs an integrated-checkout run.

```bash
rustup toolchain install nightly-2026-04-03 --component rust-src --component rustc-dev
export RUSTC="$(rustup which --toolchain nightly-2026-04-03 rustc)"
export CARGO="$(rustup which --toolchain nightly-2026-04-03 cargo)"
export PATH="$(dirname "$CARGO"):$PATH"
export LD_LIBRARY_PATH="$("$RUSTC" --print sysroot)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target/bf16-observations}"
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
"$CARGO" fetch --locked
"$CARGO" fetch --locked --manifest-path examples/tiled_gemm_general_v1/Cargo.toml
fixtures=examples/tiled_gemm_general_v1/simulation
mkdir -p target/bf16-results

for variant in tile simt; do
  selection=(--manifest-path examples/tiled_gemm_general_v1/Cargo.toml --lib --offline -j2)
  if [ "$variant" = simt ]; then
    selection+=(--no-default-features --features kernel-simt-gemm-general)
  fi
  for case in nonquarter cancellation fractional-full; do
    scripts/quickstart.sh simulate-source \
      --crate fe2o3_tiled_gemm_general_v1 --target gfx942 --bundle-version 6 \
      --matrix-bf16-numerics finite-sequential-fma-v1 \
      --request "$fixtures/$variant-$case.request.json" \
      --expectation "$fixtures/$variant-$case.expectation.json" \
      --output "target/bf16-results/$variant-$case.bundle-v6" \
      -- "${selection[@]}" >"target/bf16-results/$variant-$case.result.json"
  done
done
```

Each invocation exports real source using the maintained
`fe2o3-export-sim --bundle-version 6`, runs the V6 simulator, and invokes
`scripts/check-simulation-expectation.py` before publishing result stdout.
The output bundle paths must not already exist: choose a fresh output
directory on reruns. Dependencies must already be cached for offline extraction.
The numerical option changes simulation, not extraction.

Observe the retained SIMT bundles again using the default exact matrix mode,
then explicitly check full outputs without another extraction:

```bash
for case in nonquarter cancellation fractional-full; do
  "$CARGO" run --locked --offline -p fe2o3-kir-sim-cli --bin fe2o3-kir-sim -- \
    --bundle-v6 "target/bf16-results/simt-$case.bundle-v6" \
    --matrix-bf16-numerics exact-v1 \
    --request "$fixtures/simt-$case.request.json" \
    >"target/bf16-results/simt-$case.exact.result.json"
  python3 -I -B scripts/check-simulation-expectation.py \
    --expectation "$fixtures/simt-$case.expectation.json" \
    --result "target/bf16-results/simt-$case.exact.result.json"
done
```

`exact-v1` remains the default. The structured tile refuses these fractional
matrix operands under that original restricted integral contract.
`finite-sequential-fma-v1` is an explicit **ordinary observation**: finite raw
BF16 inputs, soft-float F32 fused multiply-add in increasing K order,
round-to-nearest ties-even, gradual underflow, and the source epilogue.
NaN/Inf operands and nonfinite intermediates are refused; the result reports
the selected mode. This is not native MFMA reduction equivalence, proof
qualification, protected execution, or the pending legacy V7 tutorial gate.
The scalar SIMT source retains its own F32 instruction sequence; the matrix
option does not rewrite it.

### Independent Expected Values

Expectations use exact rational arithmetic on the original raw BF16 inputs,
not simulator output. All products and partial sums in these cases are exactly
representable in binary32, so increasing-K FMA and scalar multiply/add agree:

- `nonquarter`: BF16 `3f81 * 3f7f` gives
  `(129/128)*(255/256) = 32895/32768`, F32 bits `3f807f00`.
- `cancellation`: `64.5*64.5 + 64*(-65) = 1/4`, bits `3e800000`.
- `fractional-full`: indices 0..15,
  `A[r,k]=(128+r+k)/128`, `B[k,c]=(128+2*c-k)/128`.
  Each output is
  `(16*(128+r)*(128+2*c)+120*((128+2*c)-(128+r))-1240)/16384`.
  All 256 outputs are checked, followed by eight unchanged raw `7f800001`
  canaries, which are not numerical operands.

Each small case also retains one canary. Expectations check unchanged A/B
bytes, all scalar ABI arguments, complete C bytes, and initialized-byte masks.
No tolerance can hide changed inputs, missing stores, or canary corruption.

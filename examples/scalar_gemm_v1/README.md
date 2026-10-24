# scalar_gemm_v1

This existing F32 tutorial has ordinary Rust SIMT and checked 1x64 output
memory-tile paths. The local gfx950 adapter selects the device CPU, Wave64,
XNACK-off source providers and reviewed ROCm ISA950 libraries together.
The original gfx942 source/providers and shared extraction runner remain.
Neither target is GPU-qualified until the full numerical command actually
passes on physical hardware.

## Build and numerical qualification

Start in the repository root after installing the [root requirements](../../README.md#requirements),
including the pinned nightly-2026-04-03 components. Build the production
extractor once with the normal root compiler build, then set its path and
loader directories for both target runners:

```bash
export CARGO_TARGET_DIR="$(realpath -m -- "${CARGO_TARGET_DIR:-$PWD/target}")"
rustup run nightly-2026-04-03 cargo build \
    --locked --manifest-path "$PWD/Cargo.toml" \
    -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract
export FE2O3_RUSTC_EXTRACTOR="$CARGO_TARGET_DIR/debug/fe2o3-rustc-extract"
extractor_dir="$(dirname -- "$FE2O3_RUSTC_EXTRACTOR")"
sysroot="$(rustup run nightly-2026-04-03 rustc --print sysroot)"
# Network-enabled setup must finish before the offline commands below.
rustup run nightly-2026-04-03 cargo fetch --locked --manifest-path "$PWD/Cargo.toml"
rustup run nightly-2026-04-03 cargo fetch --locked \
    --manifest-path "$sysroot/lib/rustlib/src/rust/library/Cargo.toml"
export LD_LIBRARY_PATH="$extractor_dir:$extractor_dir/deps:$sysroot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH=/opt/rocm-7.2.4
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/scalar_gemm_v1/run-gfx942.sh
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/scalar_gemm_v1/run-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/scalar_gemm_v1/run-gfx950.sh tile
```

To reuse an already built extractor, skip the build and set
`FE2O3_RUSTC_EXTRACTOR` to that executable before the loader setup above.
Keep its matching backend library alongside it or in its `deps` directory.
The gfx950 adapter checks the existing reviewed ROCm 7.2.4 OCML/tool closure.
For an installation of ROCm 7.2.1, select its matching reviewed closure from
the repository root before either gfx950 command:

```bash
export ROCM_PATH=/opt/rocm-7.2.1
export FE2O3_GFX950_OCML_MANIFEST="$PWD/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.1.manifest"
```

The network-enabled setup above populates the effective locks before offline
extraction. The root lock alone does not populate the pinned `rust-src`
dependencies, including `dlmalloc` used by `-Zbuild-std=core` resolution.

If Cargo's libgit2 cannot use the local TLS setup, the normal
`CARGO_NET_GIT_FETCH_WITH_CLI=true` option uses the installed Git client without
disabling certificate verification. No lock or toolchain pin needs changing.

With the appropriate physical GPU lease and visibility, the full commands are:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/scalar_gemm_v1/run-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/scalar_gemm_v1/run-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/scalar_gemm_v1/run-gfx942.sh
```

These commands compile the real kernel, inspect its target, Wave64, workgroup,
symbol and explicit ABI metadata, then run the real HIP kernel and compare
complete device allocations against independent CPU oracles. Compile-only
success grants no GPU numerical or proof qualification. gfx950 attempts retain
LLVM, source pins, crate binding, HSACO, notes, ISA and hashes.

The explicit `qualification-legacy-hip` example feature opts into the same
`fe2o3-core/qualification-unsafe-launch` numerical route used by the existing
general BF16/Flash examples. It uses typed buffers, checked interior views,
streams and owned `KernelParams`, with the documented unsafe module/launch
boundary for the runner's trusted external HSACO. It creates no Worker V3
publication, protected execution, proof/refinement receipt or artifact-currentness
authority. The original `fe2o3-host` dependency and default production boundary
remain. No verifier, proof, or receipt requirement is disabled or synthesized.

The host `fe2o3-core` dependency is optional: only
`qualification-legacy-hip` enables it and its native launch support. Default
and `structured-tile` CPU tests do not select HIP through this package.

All input/output allocations have leading and trailing bit canaries. Every
input bit is checked after completion; every logical output and every guard
or padding element is checked. Non-finite logical results reject. Empty
outputs skip dispatch and are reported separately. Ambiguous stream completion
aborts the qualification process before allocation/module destruction.

`structured-tile` selects genuine checked F32 output memory tiles; it adds no
matrix instructions or reduced-precision types. `gfx950-target` selects the
gfx950 source providers. The host target/launch checks reject other processors,
XNACK-on observations, non-Wave64 devices, and invalid/overflowed grids.

Run host tests through the production CLI. If it is not installed, build it
from this same checkout, then select its binary and the pinned Cargo:

```bash
# From the repository root; CARGO_TARGET_DIR is the absolute path selected above.
rustup run nightly-2026-04-03 cargo build --locked -p cargo-fe2o3 --bin cargo-fe2o3
export CARGO="$(rustup which --toolchain nightly-2026-04-03 cargo)"
export PATH="$CARGO_TARGET_DIR/debug:$(dirname -- "$CARGO"):$PATH"
```

Then run all package targets with the default Cargo profiles:

```bash
cd examples/scalar_gemm_v1
cargo fe2o3 test --locked --offline --all-targets -j2 -- --test-threads=1
cargo fe2o3 test --locked --offline --all-targets -j2 --features gfx950-target,qualification-legacy-hip -- --test-threads=1
cargo fe2o3 test --locked --offline --all-targets -j2 --features gfx950-target,structured-tile,qualification-legacy-hip -- --test-threads=1
```

The qualification fixture/validator tests are CPU-only and never initialize a
GPU. They establish neither execution nor protected qualification.

The retained rectangular oracle test uses `M=2, N=3, K=2`,
`A=[1,2,3,4]`, and `B=[5,6,7,8,9,10]`. Its complete row-major expected output
is `[21,24,27,47,54,61]`. The zero-K test requires six positive-zero outputs;
the FMA-distinguishing test also requires positive zero rather than a fused
result. These small examples do not replace the full nonquarter and boundary
matrices in the tests or numerical runner.

The tile path views flattened C as checked 1x64 memory tiles and retains the
original dynamic K recurrence, separate multiply/add, N guard and rounded
256-thread launch. The actual 60-byte explicit ABI is three `(pointer, usize
length)` pairs then three `u32` dimensions, packed through HIP's owned argument
list. It does not use the retired 64/320-byte Worker ABI assumptions. The
physical numerical matrix preserves all nine original cases, including zero
M/N/K, 255/256/257 output edges, rectangular shapes and the FMA distinguisher,
and adds six full nonquarter cases. Every logical output is compared bitwise
to the retained ordered F32 oracle, including positive zero when K is zero.

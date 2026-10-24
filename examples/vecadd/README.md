# vecadd

This existing F32 example has ordinary Rust SIMT and checked 1x64 output
memory-tile paths. `structured-tile` selects the latter. The original
`vecadd_body.rs` shared with the Verus source model is unchanged; the new
tile body does not extend that theorem or establish machine-code refinement.

Start at the repository root with the [root requirements](../../README.md#requirements),
including nightly-2026-04-03 and ROCm 7.2.4 (or the gfx950 ROCm 7.2.1
alternative below). Build the normal production
extractor, or reuse an already built matching extractor/backend pair:

```bash
CARGO_TARGET_DIR="$PWD/target" rustup run nightly-2026-04-03 cargo build \
    --locked --manifest-path "$PWD/Cargo.toml" \
    -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract
export FE2O3_RUSTC_EXTRACTOR="$PWD/target/debug/fe2o3-rustc-extract"
extractor_dir="$(dirname -- "$FE2O3_RUSTC_EXTRACTOR")"
sysroot="$(rustup run nightly-2026-04-03 rustc --print sysroot)"
export LD_LIBRARY_PATH="$extractor_dir:$extractor_dir/deps:$sysroot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH=/opt/rocm-7.2.4
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/vecadd/run-gfx942.sh
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/vecadd/run-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/vecadd/run-gfx950.sh tile
```

For an installed ROCm 7.2.1 toolchain, the gfx950 runner also accepts the
checked-in 7.2.1 tool/bitcode manifest. From the repository root, replace the
ROCm setup above with the following before running the gfx950 commands:

```bash
export ROCM_PATH=/opt/rocm-7.2.1
export FE2O3_GFX950_OCML_MANIFEST="$PWD/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.1.manifest"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$extractor_dir:$extractor_dir/deps:$sysroot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The runner still validates the exact selected manifest, clang, linker and
bitcode files. Vecadd has no OCML calls and still links no OCML libraries:
the manifest is shared tool validation, not an OCML linkage requirement.
The kernel, oracle, checks and qualification scope remain unchanged.

The gfx942 runner remains compile-only. The gfx950 runner selects gfx950,
Wave64, and XNACK-off together and checks the emitted target, kernel symbol,
256-thread metadata, secure crate binding, and actual 48-byte explicit ABI:
three `(pointer, usize length)` pairs. F32 addition has no OCML calls, so
neither target links an ISA942 OCML selector. The gfx950 runner retains LLVM,
HSACO, notes, ISA, source pins, crate binding and hashes for every attempt.
Compile-only results establish no GPU execution or protected qualification.

With an independently acquired physical gfx950 GPU lease and correct device
visibility, the numerical commands are:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/vecadd/run-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=0 bash examples/vecadd/run-gfx950.sh tile
```

The runner first validates the actual Rust-produced LLVM and HSACO, then
uses plain Cargo to build and run the existing `fe2o3-hsa-runtime`
`hardware-qualification` test fixture. SIMT selects
`gfx950_vecadd_rust_cov6_matches_cpu_reference`; tile selects
`gfx950_vecadd_tile_rust_cov6_matches_cpu_reference`. Both use the same complete
17-case independent F32 oracle and the reviewed HSA executable, buffer,
explicit-kernarg and synchronous dispatch APIs. The digest-pinned HSACO must
pass the existing strict COV6 gfx950:xnack- single-kernel inspection.

This route produces numerical observations only. It supplies no production
Worker V3 publication, protected authority, native proof or artifact-currentness
receipt. Default application execution retains its Worker V3 refusal. The
older core-based `qualification-legacy-hip` application adapter remains an
explicit partial route and is not selected by these numerical commands.
The CLI binding-only tests deliberately disable HIP; their guards remain
unchanged and are not used to build the numerical HSA executable. No GPU
numerical pass is claimed until the full commands actually run.

Every allocation has bitwise NaN canaries. Complete immutable input allocations
and all output/canary slots are read back. The full independent dense F32
oracle covers empty output, signed nonquarter inputs, 63/64/65 and 255/256/257
boundaries through length 1025, distinct sufficient input extents, rounding,
signed zero, infinities and NaNs. Expected logical NaNs compare by NaN class,
as in the original tests; other logical results, canaries and input snapshots
compare exact bits. Empty outputs report no executable load or dispatch. Dispatch completion and
executable identity/release are checked by the existing reviewed runtime.

Run the ordinary source CPU simulation and package controls through the
production CLI from the repository root, with nightly-2026-04-03 first on PATH:

```bash
cargo run --locked -p cargo-fe2o3 --bin cargo-fe2o3 -- \
  test --locked --offline -p fe2o3-vecadd --all-targets -j2 -- --test-threads=1
cargo run --locked -p cargo-fe2o3 --bin cargo-fe2o3 -- \
  test --locked --offline -p fe2o3-vecadd --all-targets -j2 \
  --features gfx950-target,qualification-legacy-hip -- --test-threads=1
cargo run --locked -p cargo-fe2o3 --bin cargo-fe2o3 -- \
  test --locked --offline -p fe2o3-vecadd --all-targets -j2 \
  --features gfx950-target,structured-tile,qualification-legacy-hip -- --test-threads=1
```

The original shared-body tests remain. Both real source macros are simulated
against the complete independent oracle, including launch tails and extreme
inactive indices, guarded input reads, active short-input panics, canaries
and immutable inputs. The host target tests reject other processors, Wave32,
XNACK-on and malformed feature strings; launch tests reject overflowing grids.
These CPU tests never initialize a GPU and confer no GPU/proof qualification.

## Publication and evidence boundary

The numerical runner selects the included Vecadd SIMT and tile tests in
the shared `gfx950_advanced_hardware` runtime fixture. Both retain the same
17-case oracle and explicit opt-in, ignored hardware-test boundary.
Compile-only commands do not run these tests or establish GPU execution.

Historical private Vecadd testing covered both selections with 34 cases,
32 dispatches and two empty cases. That observation does not qualify this
public source revision. The current 17-case matrix checks 4,434 outputs,
4,435 A elements, 4,436 B elements and canaries per selection: 8,868 logical
outputs across both selections.

On 2026-10-23, public source
[`5a03850`](https://github.com/harsh-nod/fe2o3/commit/5a03850ab2e10db0250412de03e0c269cce4c09b)
passed both full gfx950 commands with a matching reused compiler,
nightly-2026-04-03 and ROCm 7.2.1. This documentation-only descendant
preserves those kernel and runtime inputs; it has no separate GPU replay.
gfx942 GPU validation, protected execution authority and whole-site acceptance
remain pending.

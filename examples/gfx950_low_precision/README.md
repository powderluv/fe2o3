# gfx950 low-precision kernels

The primary source examples in this directory are four ordinary safe Rust
`#[kernel(typed)]` functions in `src/kernel.rs`:

- `gfx950_fp4_gemm_rust`
- `gfx950_fp8_gemm_rust`
- `gfx950_fp4_attention_rust`
- `gfx950_fp8_attention_rust`

The GEMMs use format- and role-typed gfx950 `16x16x128` scaled-MFMA fragments.
Each logical FP4 code occupies the low nibble of one input byte; the typed
fragment load owns the dense register packing required by the instruction.
The attention sources additionally use the move-only
`Gfx950LdsTransposeTile` transition
`Uninitialized -> Staged -> Published` before reading the K fragment, then
perform stable softmax and value accumulation through typed fe2o3 operations.
`src/reference.rs` is an independent CPU oracle with deterministic,
axis-varying inputs and exact OCP E2M1/E4M3 decoders.

All four Rust kernels launch four `256`-thread workgroups. Each workgroup owns
four Wave64 problems, for a total batch of 16 non-identical inputs. A GEMM wave
owns one complete `16x16` output tile. An attention wave owns one complete
16-token head and a private transpose tile: 1 KiB for FP4 or 2 KiB for FP8.
The compiler reserves 4 KiB or 8 KiB of static LDS per workgroup. All waves
execute the same stage, publication, uniform workgroup barrier, and
transpose-read lifecycle, while wave-relative LDS addresses prevent aliasing.
No output element is shared between waves or workgroups.

## Setup

From the repository root, use the pinned toolchain and fetch the locked
dependencies before offline builds. ROCm LLVM and runtime development libraries
must be installed; the runners default to `/opt/rocm` and accept `ROCM_PATH`.

```bash
rustup toolchain install nightly-2026-04-03 --component rust-src --component rustc-dev
export CARGO_BUILD_JOBS=2
export CARGO_INCREMENTAL=0
cargo fetch --locked
cargo fetch --locked --manifest-path examples/gfx950_low_precision/Cargo.toml
```

The job and incremental settings bound ordinary Cargo build parallelism.
The standalone package uses a checked-in test profile with debug information
disabled; assertions and overflow checks retain their normal test defaults.

## Host Source Checks

From the repository root, build the maintained Cargo frontend for the source
and oracle tests. The native runners do not require a separately installed
frontend; they build their own default extractor.

```bash
cargo build --locked -p cargo-fe2o3 --bin cargo-fe2o3
export PATH="$PWD/target/debug:$PATH"
cargo fe2o3 test --locked --all-targets \
  --manifest-path examples/gfx950_low_precision/Cargo.toml
```

The checked-in test profile requires no profile environment overrides; the
ordinary binding-only host-test runner has a fixed 1 GiB executable bound,
while shared tool and protected executable paths retain their 512 MiB bound.

The production rustc importer, V8/V9 Kernel IR schemas, exact gfx950 target
profile, and full-module AMDGPU lowering consume all of these device terminals.
`GFX950_RUST_TO_HSACO_LOWERING_SUPPORTED_V1` is therefore `true`.

## Production Rust to gfx950 HSACO

Each runner rebuilds the production extractor, compiles exactly one ordinary
Rust kernel for `amdgcn-amd-amdhsa`, validates the crate-binding handoff and
LLVM profile, finalizes a COV6 object, checks symbol-scoped ISA and metadata,
and executes a digest-pinned HSA numerical test:

```bash
bash examples/gfx950_low_precision/run-fp4-gemm-gfx950.sh tile
bash examples/gfx950_low_precision/run-fp4-gemm-gfx950.sh simt
bash examples/gfx950_low_precision/run-fp8-gemm-gfx950.sh tile
bash examples/gfx950_low_precision/run-fp8-gemm-gfx950.sh simt
bash examples/gfx950_low_precision/run-fp4-attention-gfx950.sh tile
bash examples/gfx950_low_precision/run-fp4-attention-gfx950.sh simt
bash examples/gfx950_low_precision/run-fp8-attention-gfx950.sh tile
bash examples/gfx950_low_precision/run-fp8-attention-gfx950.sh simt
```

The FP4 and FP8 GEMM runners default to `tile`. Their `simt` selections compile
`src/kernel_simt_fp4.rs::gfx950_fp4_gemm_simt` and
`src/kernel_simt.rs::gfx950_fp8_gemm_simt`. They decode actual OCP E2M1 low
nibbles or E4M3 bytes and accumulate scalar F32 products. Each pair uses the same
16-batch, WG256/grid4 geometry, three-slice ABI, independent CPU oracle,
immutable-input checks, and output canaries. Tile retains its scaled-MFMA
checks; SIMT requires byte loads and scalar F32 arithmetic with no MFMA.
Both selections require `gfx950:xnack-`; this does not add gfx942 support.
The historical hardware receipts below cover the four original tile kernels,
not the new SIMT counterparts.

The FP4 and FP8 attention runners also accept `tile` (the default) and `simt`.
Their SIMT selections compile
`src/kernel_simt_fp4_attention.rs::gfx950_fp4_attention_simt` and
`src/kernel_simt_attention.rs::gfx950_fp8_attention_simt`: scalar E2M1 or E4M3
decoding, F32 dot products, stable softmax, and F32 value accumulation. Each uses
the same four-slice ABI, 16 independent heads, WG256/grid4 geometry, full
CPU oracle, immutable inputs, and output canaries as tile. Tile uses 4 KiB
(FP4) or 8 KiB (FP8) of static LDS per workgroup; SIMT uses none. The FP8 value
paths use the complete E4M3 decoder, including subnormals, signed zero,
exponent-15 finite values, and NaNs. Historical receipts do not qualify these
changed pairs.

For compilation and ISA validation without GPU execution, prefix a command
with `FE2O3_EXAMPLE_COMPILE_ONLY=1`.

To compile the GEMM artifacts and build the host fixtures without GPU
execution, run the following ordinary CPU steps from the repository root:

```bash
for precision in fp8 fp4; do
  for variant in tile simt; do
    FE2O3_EXAMPLE_COMPILE_ONLY=1 \
      bash "examples/gfx950_low_precision/run-$precision-gemm-gfx950.sh" "$variant"
  done
done
cargo test --locked --offline -p fe2o3-hsa-runtime \
  --features hardware-qualification \
  --test gfx950_fp8_gemm_hardware --test gfx950_fp4_gemm_hardware --no-run
```

For the compiler prebuild, CI uses the following explicit debug profile. This
is an ordinary CPU build phase, separate from the bounded native run:

```bash
CARGO_PROFILE_DEV_DEBUG=0 cargo build --locked -p rustc-codegen-fe2o3 \
  --lib --bin fe2o3-rustc-extract
```

This optional prebuild executes no GPU tests. It uses a different dev profile
from the unprefixed runner commands above, so those commands may rebuild the
compiler with their default dev profile. This command establishes no bounded
warm-cache or default-bootstrap result. The full runner retains its normal
build steps; no private extractor or absolute scratch path is needed.
With all required inputs cached, `CARGO_NET_OFFLINE=true` selects a no-network run.

The attention runner disables Clang's implicit GPU libraries and links only an
ordered nine-file closure accepted by the shared digest validator. It defaults
to `gfx950-ocml-rocm-7.2.1.manifest`; the separately reviewed
`gfx950-ocml-rocm-7.2.4.manifest` is selected explicitly with
`FE2O3_GFX950_OCML_MANIFEST`. Both manifests pin every device library plus
Clang and LLD. The tile runner requires one scaled gfx950 MFMA, two B4 or four B8
transpose reads, one uniform publication barrier, and the exact static LDS
allocation before dispatch. The FP4 and FP8 SIMT selections instead check scalar
arithmetic and byte loads, absence of MFMA and transpose operations, and
zero static LDS.

The selected root compiler binding is portable across checkout paths.
Cargo's original ordered metadata remains a separate build observation; only
the terminal analysis session receives the package-, manifest-, feature-,
target-, and profile-bound portable token. Dependency compilations and general
Cargo wrapper invocations are not rewritten.

The strict Worker V3 provider and admission policy also pin this exact closure
and reject caller-supplied providers. A measured native protected-worker build
still requires matching LLVM/LLD development packages; this does not block the
ordinary Rust-to-HSACO runners above.

## Historical WG256/grid4 tile receipts

On 2026-09-03, all four production Rust wrappers completed extraction,
gfx950:xnack- COV6 finalization, symbol-scoped ISA inspection, and numerical
execution on physical GPU 6 of SSH host `mi350` with ROCm 7.2.1. Each launch
checked 16 non-identical inputs and all 4,096 output values, as well as
immutable inputs and guard canaries. These receipts do not include new timing
measurements.

| Kernel | Maximum absolute error | HSACO SHA-256 |
| --- | ---: | --- |
| FP4 GEMM | `0` | `2e9cc2bd178e1e1b72237cb32cc8f3e08d2d140d735520ea0147ed84fe81f93b` |
| FP8 GEMM | `0` | `75ce58c286cc6c3b199bf1144e571e8a3d6b7dc0e373a9dee0589bf67b3d1e6d` |
| FP4 attention | `1.192092896e-7` | `cc25e739a12b1a889e42f522708d59b4e626908a2b351dc051f4d3df59a92e38` |
| FP8 attention | `5.960464478e-8` | `4273c31ce4545e09e051abfcb704d1c7750d7b52ee50b01801caec5ddd2d0479` |

## HIP comparison fixture

The neighboring standalone HIP fixture contains four fixed-shape, single-wave
kernels for AMD CDNA 4 (`gfx950`):

- E2M1 FP4 `16x16x128` GEMM
- E4M3 FP8 `16x16x128` GEMM
- fused FP4 attention over 16 tokens with head dimension 128
- fused FP8 attention over 16 tokens with head dimension 128

The attention kernels keep the score tile in registers, apply softmax, and
immediately accumulate the value tile. They do not materialize the `Q*K^T`
matrix. Their K fragments pass through the gfx950 LDS transpose-read
instructions before the MFMA.

Run the compiler, ISA, and numerical checks on a gfx950 machine:

```bash
./build_and_test.sh
```

On a compiler host that does not expose a gfx950 runtime device, the codegen and
disassembly checks can still run:

```bash
./build_and_test.sh --compile-only
```

`check_isa.sh` scopes every assertion to its kernel symbol. It requires
`v_mfma_f32_16x16x128_f8f6f4` in all four kernels, FP4 format selectors
`cbsz:4 blgp:4`, `ds_read_b64_tr_b4` in FP4 attention, and
`ds_read_b64_tr_b8` in FP8 attention. A scalar-only implementation cannot pass
these checks.

The executable independently decodes the packed inputs and computes CPU GEMM
and attention references. Inputs vary across both matrix axes, including
non-uniform attention K, so an incorrect token/depth permutation changes the
softmax result. The comparison rejects NaN and infinity before applying its
error tolerance. Execution is rejected when the selected HIP device is not
`gfx950`.

## Previous single-wave Rust evidence

The following 2026-08-26 measurements predate the four-workgroup, four-wave
mapping. They establish the numerical and ISA baseline but are not evidence for
the current launch geometry. All four ordinary Rust runners passed through SSH host alias
`mi350` (remote hostname `smci350-rck-g03-b19-03`) with ROCm 7.2.1 on gfx950.
Every test checked all 256 outputs, immutable inputs, output canaries, exact
gfx950:xnack- COV6 metadata, and symbol-scoped ISA.

| Rust kernel | Compiler-derived binding | LLVM SHA-256 | HSACO SHA-256 | Required ISA | Maximum absolute error |
| --- | --- | --- | --- | --- | --- |
| `gfx950_fp4_gemm_rust` | `ff22ff3610dda0a94803a8011ced229b78c77400ca63c9b929d6ecba78ed6f01` | `b92ceef45655bb2ae131c2b09645ff8fb588299a994e9cbf84b07b7868fca115` | `f170671b0b778cda3876faee253e4ac3a092efdd9c1ebbfcfe901590ea3e4e4d` | scaled MFMA, `cbsz:4 blgp:4` | `0` |
| `gfx950_fp8_gemm_rust` | `d67f1755b38fbdac67cec83da3ebc359f874e3fbf90fcc036471455ec117dfea` | `351dbfeecec00e673e3e15557b97dc1c53006839dfb9d1a0a7b03ac6c23ae6e3` | `4c19d4a90ec71afa7621cc7f9f8d4d5af8e9dd87486536c702b8eb6dcc4c3d8f` | scaled MFMA, FP8 selectors | `0` |
| `gfx950_fp4_attention_rust` | `a9a878f0e2fc3a42ad17edf0a326a89695398bb6d7460eaf278ea3e8c53f4cf5` | `ea183b9dedc375a1e98278d1053b7a500e0e8f4efe618230479e19bc1b81ecaa` | `390b8cd9d8493ddbfb953e53c4a17cfb0cdab5074365b77b7c14bf64b6f64008` | scaled MFMA, two `ds_read_b64_tr_b4` | `2.235174179e-8` |
| `gfx950_fp8_attention_rust` | `0c9610e86137831ce25b08b9ad87073ec16f459aa11aeea6806733f788bbeec1` | `2e6a2c79a57e0a8796fe95c806fbbf3a8406bae6c55646802beb235b01d88c2b` | `5511819cf16a7119f846c6fe01de703257fd9c217b8fa7f32438bf47635c9221` | scaled MFMA, four `ds_read_b64_tr_b8` | `5.960464478e-8` |

The GEMM tolerance is `1e-5`. Attention uses `2e-3` absolute plus `2e-3`
relative tolerance to cover MFMA accumulation, Wave64 reductions, and the
device exponential approximation; the measured errors are substantially
smaller.

## HIP validation evidence

On 2026-08-26, the complete `./build_and_test.sh` path passed through SSH host
alias `mi350` (remote hostname `smci350-rck-g03-b19-03`) with ROCm 7.2.1,
HIP 7.2.53211, AMD Clang 22.0.0git, and an AMD Instinct MI350X reported as
`gfx950`. The LDS staging uses the measured hardware permutations: B4 is a
`16x16` nibble transpose per 16-lane group, while B8 is two interleaved `8x8`
byte transposes per group. FP8 register packing follows the documented CDNA 4
split: lane group `g` supplies K=`g*16..g*16+15` in v0-v3 and
K=`64+g*16..64+g*16+15` in v4-v7. All four deterministic CPU-oracle
comparisons passed:

```text
PASS FP4 GEMM       max_error=0
PASS FP8 GEMM       max_error=0
PASS FP4 attention  max_error=2.38419e-07
PASS FP8 attention  max_error=2.38419e-07
```

The tested HIP source SHA-256 was
`5ecfad224a691b61a07ef4aa16e144853bd3e8f53295a0e9c60404877356609a`.
The resulting gfx950 HSACO SHA-256 for the retained final run was
`ab39293c0f251678496cb5da026b8fb6ebbb4f6c96989ad5a2962d3ad6018379`;
the host executable SHA-256 was
`5a0ca9b3a72421824c6ff4e13c8294ac0b6922cb1a4f63f3b7619abc4c1ed45d`.
Symbol-scoped disassembly reported:

| Kernel | Required gfx950 instructions |
| --- | --- |
| `gfx950_fp4_gemm` | `v_mfma_f32_16x16x128_f8f6f4` with `cbsz:4 blgp:4` |
| `gfx950_fp8_gemm` | `v_mfma_f32_16x16x128_f8f6f4` with the default FP8 format |
| `gfx950_fp4_flash_attention` | FP4 MFMA above plus two `ds_read_b64_tr_b4` |
| `gfx950_fp8_flash_attention` | FP8 MFMA above plus four `ds_read_b64_tr_b8` |

An earlier compile-only run on SSH host `mi350-2` used ROCm 6.4.1 and produced
the same required instruction families, but that host exposed only
`AMD Radeon Graphics (gfx1036)`. The executable correctly refused that device.

### FP8 Attention SIMT Counterpart

`./run-fp8-attention-gfx950.sh` retains the tile default;
`./run-fp8-attention-gfx950.sh simt` selects the ordinary scalar Rust counterpart.
Both consume real OCP E4M3 Q/K/V bytes and produce the same sixteen
16-token, 128-depth, 16-value-column attention problems (4,096 F32 outputs).
The SIMT kernel recomputes scores in increasing depth order, then performs
max-subtracted F32 exponential normalization and PV accumulation without MFMA,
LDS, or subgroup operations. It retains WG256/grid4 and the four-slice ABI.
The hardware fixture uses the unchanged independent full-output reference,
input-immutability checks, tolerances, and prefix/suffix canaries. These are
gfx950-only paths; source presence is not compilation, hardware qualification,
native MFMA numerical equivalence, or protected proof evidence.

### FP4 Attention SIMT Counterpart

`./run-fp4-attention-gfx950.sh` retains the tile default;
`./run-fp4-attention-gfx950.sh simt` selects ordinary scalar Rust attention.
Q, K, and V each store one OCP E2M1 value in the low nibble of a byte;
the high nibble is ignored, not treated as a second packed value. The existing
SIMT decoder's independent all-256-byte test covers this representation,
including signed zero. Sixteen 16-token, 128-depth, 16-value-column problems
produce all 4,096 F32 outputs with the unchanged four-slice ABI and WG256/grid4.
The SIMT path uses increasing-depth F32 MACs, two score passes and max-subtracted
F32 exponential normalization/PV, without MFMA, LDS, or subgroup operations.
It uses the original independent full-output oracle, input-immutability checks,
tolerances and canaries. This is gfx950-only source; compilation, numerical
hardware validation, native MFMA equivalence and protected proof are separate.

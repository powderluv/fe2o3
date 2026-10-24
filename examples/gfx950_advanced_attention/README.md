# gfx950 advanced attention kernels

The ordinary attributed Rust kernels in [`src/kernel.rs`](src/kernel.rs) are
the fe2o3 source for these tutorials. [`src/reference.rs`](src/reference.rs)
contains independent safe CPU references, and
`cargo fe2o3 test --offline --all-targets` checks
their fixed-shape numerical and selection contracts. The HIP program remains
a separate compiler, ISA, and MI350 hardware-validation companion for the
older non-KDA profiles. The matrix-state KDA artifacts come from the production
Rust compiler path, but their recorded device execution uses the deprecated
HSA qualification oracle rather than the production direct-KFD runtime.

Each `run-*-gfx950.sh` entry point selects exactly one kernel feature, invokes
the production fe2o3 extractor, checks the compiler-published crate binding,
links an exact gfx950:xnack- COV6 HSACO, validates its single-kernel metadata
and symbol-scoped ISA, and runs a digest-pinned numerical test through the
deprecated HSA qualification oracle. This is an explicit verification path; it
does not grant protected artifact publication or production runtime authority.
The HIP ISA and runtime results below remain independent evidence only for the
symbols that the HIP program implements with the same semantics.

This directory is a bounded educational validation suite for AMD CDNA 4
(`gfx950`). It is not a production implementation, performance claim, model
reproduction, or general-purpose operator library.

The fixed shapes include a one-head KDA teaching profile with key dimension 16,
value dimension 16, an FP32 16-by-16 matrix state, one decode token, and eight
prefill tokens split into two four-token chunks. The other profiles use 16
channels, 16 attention tokens, and attention head dimension 128. The suite
contains:

- exact single-token Kimi Delta Attention decode with per-key decay, a
  rank-one delta update, and matrix-state output projection;
- exact two-chunk KDA prefill using the four-token WY/UT formulation, with the
  updated matrix state carried from the first chunk into the second;
- content-indexed sparse attention with top-two block selection, top-three
  token selection, and sparse normalization/value reduction over those tokens;
- DeepSeek sparse attention whose explicit top-k token list is produced at the
  Lightning Indexer boundary, masks out-of-range sentinels, and evaluates QK,
  stable softmax, and PV over only the remaining unique KV rows;
- compressed hybrid attention combining three compressed global blocks with a
  four-token sliding window;
- AttnRes four-depth softmax aggregation, four-branch gated residual mixing,
  and a four-stream mHC mixer with three Sinkhorn iterations.

Every production Rust kernel launches four `256`-thread workgroups. KDA uses
all four Wave64s in a workgroup for one matrix-state problem, producing four
independent batches per launch. Content-sparse and compressed-hybrid attention
assign one independent head to each Wave64, for 16 heads per launch. DeepSeek
sparse attention, AttnRes, and four-branch residual assign one independent
16-channel item to each Wave16 subgroup, for 64 items per launch. mHC assigns
one independent four-stream problem to each Wave64, for 16 items per launch.
Inputs vary by batch and every output span has a single wave, subgroup, or
workgroup owner.

The two transpose-based kernels give each Wave64 a private 2 KiB LDS K tile.
The compiler therefore reserves 8 KiB of static LDS for each WG256. All four
waves execute the same stage, publication, workgroup-barrier, and transpose-read
lifecycle uniformly, while wave-relative LDS addresses prevent cross-wave
aliasing.

Every output and sparse index is compared against an independently written CPU
oracle using deterministic inputs. The KDA oracle is an f64 scalar,
token-by-token implementation of the matrix recurrence; it does not reuse the
kernel's WY/UT equations. The dense-tile attention profiles use
non-uniform, exactly representable E4M3 values; the DeepSeek sparse teaching
profile uses finite FP32 values. The executable rejects non-gfx950 devices.
The content-selected and compressed-hybrid kernels use a gfx950 FP8
`v_mfma_f32_16x16x128_f8f6f4` score tile whose K operand is supplied by four
`ds_read_b64_tr_b8` LDS transpose reads. `check_isa.sh` validates those
instructions, their exact counts, and transpose-before-MFMA ordering within
each kernel symbol. At this bounded shape, the score MFMA covers all 16 tokens;
the sparse kernel applies its selected ragged set to softmax and the value
reduction. The separate DeepSeek kernel instead performs FP32 Wave16 reductions
for exactly the indexed rows; using the dense MFMA tile there
would defeat its sparse-compute contract. Each Wave16 subgroup now owns one
independent indexed head and its 16 output stores. It does not reproduce the learned
indexer or claim a production FlashMLA scheduling strategy.

For KDA, the public inputs begin after the surrounding model projections and
short convolution: `q` and `k` are L2-normalized, `alpha` is already in
`(0,1]`, and `beta` is already in `[0,1]`. The kernel applies the fixed
`1/sqrt(16)` query scale. The logical recurrence stores `S[K,V]`; production
device memory stores its transpose `H[V,K]`, allowing each 16-lane Wave16 group
to reduce one value column. Each of the four workgroups receives a distinct
state and input sequence. Its checked `Index1D` ownership model makes each
decode output physical length 256, with each logical value replicated across
its 16 key lanes; the launch ABI concatenates four such spans. Prefill likewise
uses four lanes per logical token in each 256-element chunk-output span. The
runtime verifier compares every replica and canary, while tutorials identify
how to recover the logical outputs. This
fixed profile excludes learned projections, the width-four convolution, gate
parameterization, RMS/output gating, and the output projection used by a full
Kimi Linear layer.

Run the Rust source and independent CPU-reference checks:

```bash
cargo fe2o3 test --offline --all-targets
```

Run the production Rust lowering and deprecated HSA qualification-oracle
numerical verification on a gfx950 host:

```bash
./run-kda-decode-gfx950.sh
./run-kda-chunkwise-prefill-gfx950.sh
./run-content-sparse-attention-gfx950.sh
./run-deepseek-sparse-attention-gfx950.sh
./run-compressed-hybrid-attention-gfx950.sh
./run-attnres-aggregate-gfx950.sh
./run-four-branch-residual-gfx950.sh
./run-mhc-sinkhorn-mix-gfx950.sh
```

## Explicit modes for the five attention families

Each wrapper accepts `simt` or `tile`; omitting the argument retains its
original source selection and arithmetic. Existing ablation features and their
historical positive or negative roles remain unchanged.

| Family | SIMT selection | Tile selection | Stored input and output contract |
| --- | --- | --- | --- |
| KDA decode | Original Wave16 recurrence | Checked `Tiled2D<Index1D,256,16,16,1>` state/output ownership | Raw F32; complete value-major state and every replicated output |
| KDA chunkwise prefill | Original Wave16/WY two-chunk recurrence | Checked 16x16 state and both output planes | Raw F32; all eight token outputs and final state, original rounding order |
| Content-sparse attention | Ordinary Rust E4M3 decode, scalar dot products and subgroup selection | Original checked FP8 matrix views and MFMA path | Raw FP8 E4M3 Q/K/V, F32 accumulation, all selected U32 IDs and gated F32 outputs |
| DeepSeek sparse attention | Original indexed F32/Wave16 path | Checked 16x16 ownership of all three F32 output planes | Original U32 indices, sentinel/masking/duplicate behavior, maximum and normalizer |
| Compressed-hybrid attention | Ordinary Rust E4M3 decode, scalar dot products and subgroup blend | Original checked FP8 matrix views and MFMA path | Raw FP8 E4M3 Q/K/V, F32 local/global normalization and gated output |

KDA and DeepSeek tile modes preserve their canonical arithmetic and physical
layouts. KDA is ordinary Rust SIMT with subgroup reductions and zero MFMA
instructions. The two FP8 SIMT modes retain the raw byte format; they do not
replace FP8 inputs with rounded F32 data. Their LLVM and ISA checks require
subgroup exchange and reject MFMA/transpose instructions. The original FP8
tile checks continue to require their matrix and transpose instructions.

From the repository root, compile the exact ten selections without a GPU launch:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-kda-decode-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-kda-decode-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-kda-chunkwise-prefill-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-kda-chunkwise-prefill-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-content-sparse-attention-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-content-sparse-attention-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-deepseek-sparse-attention-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-deepseek-sparse-attention-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-compressed-hybrid-attention-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-compressed-hybrid-attention-gfx950.sh tile
```

`run-kda-prefill-gfx950.sh` remains an alias of chunkwise prefill and accepts
the same arguments. The scripts use the existing nightly-2026-04-03 extraction
route and validate the reviewed ROCm closure. For an installed ROCm 7.2.4
closure, select the checked-in manifest with
`FE2O3_GFX950_OCML_MANIFEST=$PWD/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.4.manifest`.
An existing compatible extractor may be selected through
`FE2O3_RUSTC_EXTRACTOR`; otherwise the normal runner builds it from the checkout.

Both modes retain the original independent HSA oracle plans, every output
word, input immutability and canary checks. Compile-only and host-test results
do not establish GPU execution or formal refinement. The hardware receipts
below remain tied to their historical sources; they do not qualify these
newly selectable pairs. gfx950 is the current scope; gfx942 remains pending.

## Performance audit

[`PERFORMANCE_AUDIT.md`](PERFORMANCE_AUDIT.md) gives the per-kernel external
candidate audit, exact-comparability decision, GPU4 latency, ablation result,
and architecture-resource floor. The corresponding structured record is
[`performance-audit-mi350-gpu4-v1.json`](performance-audit-mi350-gpu4-v1.json).
The audit intentionally does not project these teaching shapes onto full-model
latency or call a non-comparable result state of the art.

## Current WG256/grid4 numerical qualification

On 2026-09-03, all eight production Rust wrappers completed extraction,
gfx950:xnack- COV6 finalization, symbol-scoped ISA inspection, and numerical
execution on physical GPU 6 of SSH host `mi350` with ROCm 7.2.1. Every launch
used four WG256 workgroups, exercised all disjoint non-identical problems, and
checked immutable inputs plus output canaries. These are correctness and
artifact-identity receipts, not latency measurements.

| Kernel | Maximum absolute error | HSACO SHA-256 |
| --- | ---: | --- |
| KDA decode | state `2.980232239e-8`; output `7.450580597e-9` | `11af04ea552ea1e7c2a7bcad2a3dd26222ced4ffac39148015bb90b578c3f7b0` |
| KDA chunkwise prefill | state `1.490116119e-8`; both chunks `7.450580597e-9` | `dcb9f8cc55339234e05ac814536fd8b98b2d34f8c41de9c76c6baa475edaea9c` |
| Content-sparse attention | output `5.820766091e-11`; 48 selected IDs exact | `d609377c0e56d3589f88fb0a850c39c60a3e34cac3d53ad8f3dfcf0159d02d9d` |
| DeepSeek sparse attention | output `5.215406418e-8`; maximum `1.490116119e-7`; normalizer `4.768371582e-7` | `1d55054669d735190e1747c2e16f510455ae89a623d93b1fb66c2037676f9437` |
| Compressed-hybrid attention | output `5.960464478e-8` | `314ed596839d1aa04557ca26b18f9a4a2356ad67e767a3865e40a7bc0bf6a90d` |
| AttnRes aggregate | output `4.470348358e-8` | `8622031a6d857b060b8b036e99d8e2f91068904f075fa54d7c1b88b40d6c96b3` |
| Four-branch residual | output `1.490116119e-8` | `fd77ba3f34568aa95b7b59ebe9fc71506e317d7ac6090e3439601512fd46acdd` |
| mHC Sinkhorn mix | output `6.705522537e-8` | `e441bf98aec02fc596f55e00477ab2f647dd776bb8756099c6168802b16b6a13` |

## Historical KDA MI350 performance evidence

These measurements use the preceding single-workgroup launch geometry. They
remain provenance for the earlier implementation, but they do not measure the
current four-workgroup kernels and must not be used as current throughput or
latency claims until the multigrid campaign is rerun.

[`kda-mi350-performance-v1.json`](kda-mi350-performance-v1.json) records the
replicated fixed-shape KDA campaign, raw-file SHA-256 manifests, exact source
and HSACO identities, ISA resource counts, correctness results, rejected
variants, and the optimistic HBM-only resource floor. Recreate that aggregate
from retained JSONL results with:

```bash
python3 summarize_kda_mi350.py \
  --fe2o3-directory /absolute/path/to/fe2o3-jsonl \
  --fla-directory /absolute/path/to/fla-jsonl \
  --output /absolute/path/to/new-kda-summary.json
```

The official Flash Linear Attention comparator is
`fla.ops.kda.fused_recurrent_kda` at commit
`8e84ed4a6727be082c34a3855c60623fd11411e9`. Run one T=1 or T=8 process from
an environment containing that checkout, PyTorch ROCm, Triton, and FLA with
`benchmark_fla_kda_mi350.py`. Both implementations were checked against an
independent sequential matrix-state recurrence before timing.

For FP32 B=H=1, K=V=16, the median of five process medians was 4,720 ns for the
fe2o3 code object executed through the HSA qualification oracle versus 39,345
ns for FLA (8.336x), and 11,400 ns for fe2o3 two-chunk prefill through that
oracle versus 39,113 ns for FLA (3.431x). These are the fastest measured
eligible implementations for this exact shape and semantics, not a universal
KDA state-of-the-art claim or direct-KFD runtime measurement. The fe2o3 and FLA
device-side timers use different sampling granularities, which is retained as
a protocol caveat in the evidence file.

The sparse and hybrid runners additionally require exactly four
`ds_read_b64_tr_b8` instructions before one FP8
`v_mfma_f32_16x16x128_f8f6f4`. Exponential device math uses only the reviewed
selected reviewed OCML `exp` closure shared with the low-precision examples;
ROCm 7.2.1 remains the default and the checked-in ROCm 7.2.4 manifest is an
explicit alternative. Gfx950 square root lowers to its target-native LLVM
intrinsic. Set
`FE2O3_REPO_ROOT`, `ROCM_PATH`, `RUSTUP`, `CARGO`, or the documented tool and
target-directory environment variables when validating a copied checkout.

## Previous single-workgroup compiler and qualification-oracle evidence

The correctness and artifact identities below predate WG256/grid4 batching for
the non-KDA kernels and grid4 batching for KDA. They establish historical
lowering and ISA behavior only; current multigrid evidence is generated by the
updated runners and hardware plans.

On 2026-09-01, the exact matrix-state KDA wrappers passed production Rust
lowering, COV6 linking, symbol-scoped ISA inspection, and deprecated HSA
qualification-oracle execution on SSH host `mi350`
(`smci350-rck-g03-b19-03`) with ROCm 7.2.1, using physical GPU
6 (`ROCR_VISIBLE_DEVICES=6`, `HIP_VISIBLE_DEVICES` unset). Decode maximum
absolute errors were `1.490116119e-8` for the 256-element final state and
`3.725290298e-9` for every replicated output. Chunkwise prefill errors were
`2.980232239e-8` for the final state and `7.450580597e-9` for both replicated
four-token output buffers. The checked symbols use Wave16
`ds_bpermute_b32` reductions and intentionally contain no MFMA or LDS transpose
instructions. Final source and artifact identities are pinned by the advanced
tutorial evidence records.

On 2026-08-27, the remaining production-compiler wrappers passed through the
same HSA qualification oracle on the same host.
The largest observed absolute errors were `0` for both dense-tile attention
kernels and AttnRes, `0` for four-branch residual, and `4.470348358e-8` for
mHC. Sparse token IDs were checked exactly.

On 2026-08-31, the eighth wrapper,
`run-deepseek-sparse-attention-gfx950.sh`, passed on the same host and ROCm
release using physical GPU 6 (`ROCR_VISIBLE_DEVICES=6`,
`HIP_VISIBLE_DEVICES` unset). The production-compiled kernel returned maximum
absolute errors of `2.980232239e-8` for both the 16-channel output and softmax maximum,
and `2.384185791e-7` for the softmax normalizer, against a `5e-3` finite-value
tolerance. Its measured compiler-derived binding is
`62a1ee5804a9926ebb929061195f2229630ebdaf5a13a19d17ce7ddb4fcbbbe3`;
the LLVM, COV6 HSACO, and symbol-scoped ISA SHA-256 values are respectively
`0767554b7997f42b4e2fb85271779ca29182ec241b07cc162cb9185cac41362c`,
`c5f5465c405306d6df944df4f02066f75b94295b7e91b8c8cf73bc16482ed930`,
and `fa54e785c34d2ec26e94dad04a8f63ef2a68485ad9190a0ca747999216d5237a`.
The inspected symbol contains no MFMA or transpose instructions, matching its
selected-only sparse arithmetic contract.

The sparse and hybrid Rust HSACOs each contained exactly four
`ds_read_b64_tr_b8` instructions before one
`v_mfma_f32_16x16x128_f8f6f4` with E4M3 selectors. The per-kernel
compiler-derived binding and LLVM/HSACO SHA-256 values are printed by each wrapper and pinned
by the corresponding advanced tutorial evidence record.

Build, inspect, and run the independent companion HIP validation:

```bash
./build_and_test.sh
```

Compiler and ISA validation without execution is available with:

```bash
./build_and_test.sh --compile-only
```

## HIP validation evidence

On 2026-08-26, `./build_and_test.sh` passed via SSH alias `mi350` (remote
hostname `smci350-rck-g03-b19-03`) with ROCm 7.2.1, HIP 7.2.53211, AMD Clang
22.0.0git, and a visible gfx950 device. The code object metadata target was
`amdgcn-amd-amdhsa--gfx950`. FP8 register packing follows the documented CDNA
4 split: lane group `g` supplies K=`g*16..g*16+15` in v0-v3 and
K=`64+g*16..64+g*16+15` in v4-v7; transpose-source staging produces that same
layout after the four B8 transpose reads.

The sparse IDs were exactly `[7,1,4]`. Maximum errors were `2.98023e-08` for
sparse attention, `1.67638e-07` for compressed hybrid attention, `0` for
AttnRes, `0` for the four-branch residual, and `2.98023e-08` for mHC/Sinkhorn
mixing. The HIP file's historical vector-recurrence symbols are not evidence
for the matrix-state KDA Rust kernels.

Both attention symbols contained exactly four `ds_read_b64_tr_b8` followed by
one `v_mfma_f32_16x16x128_f8f6f4`. The tested HIP source SHA-256 was
`c44b4227c0ec525a367359bdc16aff69c3086676aa61def1b653266604d1ed1d`.
The validated code-object SHA-256 was
`dcfb1e00354ac14dffae5e069138c5e212b0906133838195dd717686af26ce84`;
the host executable SHA-256 was
`d741786606e6a4d05ab2fd5f0a411bbc696a3dc6b12bce87c98eb624be39901e`.

## Historical mHC Sinkhorn performance ablation

This section retains the pre-da6 campaign verbatim for provenance. The final
da6 tutorial campaign in `perf-evidence/gfx950-advanced-ablation-evidence-v1.json`
supersedes these latency numbers; it is intentionally attributed to its own
source and artifact digests.

The production Rust mHC kernel keeps the exact 4x4 mixing matrix, four 16-channel
streams, three Sinkhorn row/column normalization iterations, and 64-output ABI.
The optimized mapping assigns one rotated matrix element to each lane in four
contiguous wave16 groups. A width-four reduction computes each row sum, one
reciprocal is reused by the four row elements, and four verifier-bounded
broadcasts compute each column sum without divergent selection. The final row
weights are broadcast once and reused for the four stream loads.

The exact machine-readable accounting, artifact identities, protocol, and bound
are in [`performance-mhc-sinkhorn-v1.json`](performance-mhc-sinkhorn-v1.json).
The published pre-optimization Rust HSACO and the candidate both passed the same
CPU oracle and guard canaries on physical GPU 6.

| Artifact | Median ROCr time | Bootstrap 95% CI | ISA instructions | SGPR / VGPR |
| --- | ---: | ---: | ---: | ---: |
| Published baseline `0a42de9c...` | 7.160 us | [7.160, 7.160] us | 1,750 | 34 / 34 |
| Distributed wave16 `f463b05e...` | 5.040 us | [5.000, 5.040] us | 457 | 22 / 12 |

The default persistent-queue protocol used five fresh processes per variant in
alternating AB/BA order. Each process used 1,000 initial warmups and 30 blocks
of 100 samples, with 20 untimed rewarm dispatches per block. Across 15,000
paired samples, the median paired speedup was **1.432x** with bootstrap 95% CI
[1.4318568, 1.432], or a **30.1676%** median latency reduction.

| Optimization | Exact static contribution |
| --- | --- |
| Distribute one matrix element per lane | `v_exp_f32` 16 to 1 (-93.75%); global dword loads 12 to 5 (-58.33%) |
| Subgroup row reduction and reciprocal reuse | expanded divide sequences and `v_rcp_f32` both 96 to 6 (-93.75%) |
| Branch-free bounded column broadcasts | scalar branches 8 to 2 (-75%), at the cost of 22 `ds_bpermute_b32` instructions |
| Combined rewrite | instructions 1,750 to 457 (-73.89%), SGPR 34 to 22, VGPR 34 to 12; measured 1.432x |

Only the combined rewrite was timed. The stage-level ISA deltas overlap, so
they are not presented as additive marginal latency speedups.

The strict whole-device resource floor uses 576 unique compulsory bytes: 64 B
of logits, 256 B of streams, and 256 B of output. Counting 616 logical FP32
algebraic operations, the MI350X inputs in
[`mi350x-bound-inputs-v1.json`](../../perf-evidence/mi350x-bound-inputs-v1.json)
give:

```text
max(576 / 8e12, 616 / 144.2e12) = 0.072 ns
```

The measured 5,040 ns median is 70,000 times that fully occupied global
roofline. This is expected for a single-wave latency tutorial: the roofline
excludes dispatch latency and does not provide a dependency-throughput bound
for the 16 logical exponentials. It is a strict resource bound, not a claimed
attainable single-wave latency.


### Explicit SIMT and checked tile mixing modes

The three F32 mixing tutorials accept an optional `simt` or `tile` argument.
Without a mode argument or environment override, the canonical selection
remains unchanged. Existing ablations remain available. SIMT uses
the existing ordinary Rust body. Tile preserves its F32 operation order and
writes through checked `DisjointSlice<f32, Tiled2D<Index1D, 64, 4, 16, 1>>`
coordinates covering the same 1,024 outputs.

From the repository root, compile each selection without launching a GPU:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-attnres-aggregate-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-attnres-aggregate-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-four-branch-residual-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-four-branch-residual-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-mhc-sinkhorn-mix-gfx950.sh simt
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash examples/gfx950_advanced_attention/run-mhc-sinkhorn-mix-gfx950.sh tile
```

Use the toolchain and ROCm prerequisites documented above. From this package,
host reference and coordinate checks run with
`cargo test --locked --offline --all-targets -j2 -- --test-threads=1`.
They do not execute the GPU tile body. Compile-only success does not establish
runtime or hardware qualification.

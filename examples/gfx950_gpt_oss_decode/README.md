# GPT-OSS-120B multi-wave layer-tile megakernel

This tutorial is a production-compiler-generated safe-Rust-to-gfx950
qualification example for a bounded piece of one GPT-OSS-120B decode layer. It
is not a whole-model megakernel and does not claim to run an entire transformer
layer in one dispatch.

The architecture contract is pinned to OpenAI's official `gpt-oss` repository
at commit
[`7b583341fe16729127f6d5b94a7b09ccae97e1a1`](https://github.com/openai/gpt-oss/tree/7b583341fe16729127f6d5b94a7b09ccae97e1a1).
The pinned
[`ModelConfig`](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py)
defines hidden size 2,880, 64 query heads, eight KV heads, head dimension 64,
128 experts, top-4 routing, and a 128-token sliding window on alternating
layers. The official
[`moe.py`](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/triton/moe.py)
is the source pin for MXFP4 expert weights.

## Exact tutorial boundary

The launch uses four workgroups of 256 threads, with four Wave64 items per
workgroup: grid `[4, 1, 1]`, workgroup `[256, 1, 1]`, and 16 useful
independent items. Each wave owns one bounded batch-1 layer tile and computes:

- the full 128-expert router dot product over all 2,880 hidden values and a
  deterministic stable top-4, with lower expert ID winning an exact tie;
- one of the eight GQA groups: eight query heads against 16 cached K/V tokens,
  with attention sinks and 16 of the 64 value columns;
- four block-scaled MXFP4 matrix instructions for reduction depths `0..128`
  and output columns `0..16` of the dynamically selected top-1 MLP1 expert.

The 16-token cache lies within the 128-token window, so the selected tile is
valid for both an even sliding-window layer and an odd full-attention layer.
Rows `8..16` of the attention matrix and rows `1..16` of the expert matrix are
canonical zero padding for native `16x16` matrix instructions. QKV projection,
RoPE, RMSNorm, the other seven GQA groups, the remaining value columns, all
four complete experts, SwiGLU, MLP2, residuals, and the other layers are outside
this bounded profile.

The kernel accepts every expert tile and selects the routed expert on device.
The host does not preselect an expert. In each wave, every lane computes two router logits,
then the wave broadcasts the two `(score, expert_id)` candidates from every
lane and maintains stable top-4 state. This preserves the full 128-way dynamic
routing decision while avoiding a lane-0 serial router. The global invocation
index determines `item = index / 64` and `lane = index % 64`. Item-major
input slices and blocked output tiles are disjoint across all 16 waves, including
different workgroups. The packed route is replicated over the 64 distinct
lane-owned positions for its item, so it also has an identity-mapped disjoint
write contract.

## Fused stages and native ISA

[`src/kernel.rs`](src/kernel.rs) contains the ordinary safe Rust
`gfx950_gpt_oss_120b_decode_megakernel_v1` kernel. It keeps router state,
attention, and one selected expert tile in the same dispatch:

1. Two router rows per lane stream through authenticated `StridedReadView2D`
   views. Bounded Wave64 broadcasts merge the 128 candidates into a stable
   top-4.
2. Four BF16 `16x16x16` MFMAs form the `16x16` QK score tile. Wave16 reductions
   implement sink-aware stable softmax, and the probabilities are consumed
   directly by the value accumulation instead of materializing them.
3. Four scale-separated MXFP4 `16x16x128` MFMAs compute the selected expert
   tile. Each block is scaled and accumulated before the next fragment is
   created, reducing live accumulator state.
4. Attention output, expert output, and the packed four seven-bit expert IDs
   are committed through disjoint output capabilities.

The compiler-qualification runner requires exactly four
`v_mfma_f32_16x16x16_bf16` sites and four
`v_mfma_f32_16x16x128_f8f6f4` sites with the FP4 selectors. K is supplied in
the depth-major layout consumed by the BF16 B fragment, so an LDS transpose is
not applicable to this fixed decode interface; the runner rejects unexpected
transpose instructions rather than claiming one was used.

[`src/reference.rs`](src/reference.rs) is an independent safe CPU oracle. It
implements BF16 and OCP E2M1 decoding, the full router and tie rule,
sink-softmax attention, block scaling, and selected-expert multiplication
without calling device helpers or GPU libraries. The 16 deterministic fixtures
are nonuniform; alternating items route to experts 127 and 0.

## Build and numerical validation

Run source and CPU-oracle tests:

```bash
cargo fe2o3 test --all-targets \
  --manifest-path examples/gfx950_gpt_oss_decode/Cargo.toml
```

Build the ordinary Rust kernel through the production compiler extractor,
semantic MIR, Kernel IR, LLVM, COV6 HSACO, and symbol-scoped ISA checks, then
run the deprecated HSA qualification-oracle numerical test on physical GPU 6:

```bash
unset HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES=6 examples/gfx950_gpt_oss_decode/run-gfx950.sh
```

The current WG256/grid-`[4, 1, 1]` compatibility campaign ran on physical GPU
7 of `ssh mi350` with `ROCR_VISIBLE_DEVICES=7`. All 16 Wave64 items owned
disjoint input and output slices. The canonical fused kernel checked 4,096
attention values with maximum absolute error `1.192092896e-7`, 4,096 exact
expert values, and 1,024 exact packed-route words. Its compiler binding is
`7194a44ee0231763c5f1e345dcb682beb0922ede14a8ce1899d41b44b2b053d0`;
the LLVM, HSACO, and symbol-scoped ISA SHA-256 digests are respectively
`ae9301289f784ed43a906b8e5c2165176deca52b0d629445726e9128b94c3e12`,
`d6b2f1b54b0398cceb751d4e4a70a42b74efe368e9fbc892283a72952013daec`,
and `f52d49e23917bc11ac4b2ea1f3d8205d200b9d6ed7365350fc1189392ee837d5`.

| Current admitted kernel | Compiler binding | HSACO SHA-256 | Numerical result |
| --- | --- | --- | --- |
| Fused canonical | `7194a44ee0231763c5f1e345dcb682beb0922ede14a8ce1899d41b44b2b053d0` | `d6b2f1b54b0398cceb751d4e4a70a42b74efe368e9fbc892283a72952013daec` | attention 4,096, max abs `1.192092896e-7`; expert 4,096 exact; route 1,024 exact |
| Serial router | `0d3dd5be58c3fc42b575a9028359d3be63f5f86dffd8261a24564c09ec8c77f7` | `26515f0bd0539030f076efaf88bf52c11c13fcb6acd13997bf822a207d559722` | attention 4,096, max abs `1.192092896e-7`; expert 4,096 exact; route 1,024 exact |
| Held fragments | `94f0b2a229e36aa690afd63d7523b549390d6b62a7829a41940ee410da3f30db` | `258f002896efe9315a84db491a273f651d1cb6320b0e705e3b5a14e50859b215` | attention 4,096, max abs `1.192092896e-7`; expert 4,096 exact; route 1,024 exact |
| Interleaved stores | `cbd2a9d218b865e59afdd868c38db0aa034933cf224bb1928b6b7873a10261e0` | `b0447014a5e5c46d2c4efc862349be399100c7357a958c5b72d1a26c77c57ac6` | attention 4,096, max abs `1.192092896e-7`; expert 4,096 exact; route 1,024 exact |
| Router component | `eab1554fa8b5ca3dd61c081c2a94fe5b3a8dacbe6e263ce5ebecbc68138f980e` | `9ca3a3db0dd64f0de7a1a56b560fe097e826c6448817f749a7117c1016b10bc6` | 1,024 packed-route words exact |
| Attention component | `7e4af27e8a08344360c28b8c9b735ef8c5aa55a0807601bac5042f6f59b951b1` | `fe61d17d8679f3dde03e9ef19cacb7e0e9a2475d69eebf4a6f2803cbc194e336` | 4,096 values, max abs `1.192092896e-7` |
| Expert component | `a5299095f8322658479d027ef414fc05add7e9079610ae41797eb31507097ed7` | `3265642593dc75200cb4e078f9d8c456705b71857c8607e7eb3f491658519cb5` | 4,096 values exact |

This is an end-to-end compatibility and numerical record, not a performance
measurement or state-of-the-art claim.

## Explicit SIMT and structured-tile source modes

All nine existing selections retain their names and default roles. Set
`FE2O3_GFX950_ADVANCED_MODE=simt` or `tile` to select their existing Rust
counterparts through the same runner; omitting the variable selects the original
canonical form. This is a bounded kernel tutorial, not a model-serving runtime.

| Selection | Ablation/role | SIMT and tile counterparts |
| --- | --- | --- |
| default / `optimized` | Fused layer tile | Both |
| `serial-router` | Serial 128-expert router control | Both |
| `held-fragments` | Keep expert partials live until publication | Both |
| `scalar-attention` | Scalar QK, original FP4 expert MFMA in canonical/tile | Both; SIMT also uses the scalar expert |
| `pipelined-attention` | Two real double-buffered LDS pipelines | Both |
| `interleaved-stores` | Paired attention/expert publication | Both |
| `materialized-router` | Router component | Both |
| `materialized-attention` | Attention component | Both |
| `materialized-expert` | Expert component | Both |

`materialized` still runs the three materialized component selections in order.
Unsupported mode values/pairings fail before compilation or dispatch. Do not
enable all Cargo features together: the two source modes are mutually exclusive.

Compile the fused pair without executing a GPU:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 FE2O3_GFX950_ADVANCED_MODE=simt \
  bash examples/gfx950_gpt_oss_decode/run-gfx950.sh
FE2O3_EXAMPLE_COMPILE_ONLY=1 FE2O3_GFX950_ADVANCED_MODE=tile \
  bash examples/gfx950_gpt_oss_decode/run-gfx950.sh
```

The ablation runner also accepts explicit `<selection>-simt` and
`<selection>-tile` aliases for each of the nine table rows, using `optimized`
for the fused default. For example, `run-ablation-gfx950.sh scalar-attention-tile`
selects only that existing implementation. A conflicting inherited mode is
rejected. The aggregate `materialized` command retains its three original roles.

The same mode variable applies to `run-ablation-gfx950.sh`, including
`scalar-attention` and `pipelined-attention`. Runtime commands require one
explicit `ROCR_VISIBLE_DEVICES` UUID or ordinal. The native compiler/proof,
binding, OCML, COV6, ABI, ISA, immutable-input, canary and full-output checks are
not optional. Pipeline source requires the separately reviewed public compiler
pipeline prerequisites; copying example sources alone does not supply them.

SIMT consumes the same raw BF16 Q/K and OCP E2M1 expert bytes, all four separately
scaled expert blocks, the full 128-way router over 2,880 hidden values, stable
top-4 tie rule and all 16 items. Its FP32 operation order and collective stages
are retained; it uses no MFMA. Tile mode uses actual checked `Tiled2D` output
authority, not a renamed scalar kernel. Canonical, serial-router,
scalar-attention and component tile outputs are row-major 16x16 per item.
Held-fragment, interleaved-store and pipelined tile outputs instead preserve the
original blocked packing via checked 4x64 tiles. Route outputs retain all 1,024
replicated packed words. Both layouts retain all 4,096 attention and 4,096 expert
values, the original independent oracle, input immutability and guards.

Run normal binding-aware host controls for the fused source pair:

```bash
cargo fe2o3 test --all-targets \
  --manifest-path examples/gfx950_gpt_oss_decode/Cargo.toml \
  --features kernel-gpt-oss-decode,kernel-gpt-oss-mode-simt
cargo fe2o3 test --all-targets \
  --manifest-path examples/gfx950_gpt_oss_decode/Cargo.toml \
  --features kernel-gpt-oss-decode,kernel-gpt-oss-mode-tile
```

The checked-in package test profile disables debug information so normal host
test images remain within the existing admission bound; assertions, overflow
checks and all source/oracle controls are unchanged. Host tests cover reference
math, selected exports, source/phase ordering and physical output geometry.
They are not GPU numerical qualification. All retained native/GPU tables in this
README describe earlier campaigns. They do not qualify this public candidate or
its newly published modes.

## Current GPU-5 performance audit

The 2026-09-05 campaign measured the current 16-item WG256/grid-`[4, 1, 1]`
artifacts on physical GPU 5 of `ssh mi350`, using ROCr HSA dispatch timestamps
after 200 warmups and five blocks of 100 samples with 20 block rewarms. Every
timed artifact passed the independent CPU oracle before timing and after each
block. The fused kernel matched 4,096 attention values with maximum absolute
error `1.192092896e-7`, 4,096 expert values exactly, and 1,024 packed routes
exactly.

| Exact current artifact | Median / p95 | Contribution to fused result |
| --- | ---: | ---: |
| Fused canonical | `1.075164 / 1.086204 ms` | retained result |
| Serial-router ablation | `64.937745 / 65.285206 ms` | lane-parallel router is `60.398x`, `98.344%` lower |
| Held-fragment ablation | `1.079923 / 1.085563 ms` | sequential MXFP4 fragments are `1.00443x`, `0.441%` lower |
| Interleaved-store ablation | `1.080123 / 1.088883 ms` | grouped disjoint stores are `1.00461x`, `0.459%` lower |
| Materialized component median sum | `1.081802 ms` | fusion is `1.00617x`, `0.614%` lower |

The materialized comparison sums the medians of three separately timed Rust
components with the identical launch geometry and output contract: router
`1.001962 ms`, attention `16.480 us`, and expert `63.360 us`. It isolates the
two removed dispatch/materialization boundaries, but is not a serving-runtime
or whole-model comparison. The current fused HSACO SHA-256 is
`62139f01f2394c208e37222b896cf5ffdda18b90116dcc3181ff330b7e9c6624`.

The current batched profile has an optimistic resource floor of
`3.019944 us`, derived as the maximum of compulsory bytes at 8 TB/s, counted
FP32 work at 144.2 TFLOP/s, and MXFP4 work at 9.2 PFLOP/s. The measured median
is `356.021x` that floor because this bounded tile includes long serial router
dependencies and cannot saturate the whole device. AITER, CK, the official
GPT-OSS Triton kernels, vLLM, and SGLang do not match every shape, precision,
layout, fusion, output, and timer axis, so they are architectural references,
not admitted external performance comparators. The shared
[`sota-performance-audit-v1.json`](../gfx950_advanced_systems/sota-performance-audit-v1.json)
records the complete current campaign and raw-record hashes; no external SOTA
or whole-model tokens/s claim is made.

The following result is historical evidence for the former single-Wave64,
grid-`[1, 1, 1]` kernel. It does not validate or measure the current
WG256/grid-`[4, 1, 1]` compatibility update. That MI350X HSA qualification
run checked all outputs, immutable inputs, output
canaries, ABI size, artifact digests, and exact `gfx950:xnack-` metadata. Its
maximum absolute attention error was `8.940696716e-8`; expert output and packed
top-4 IDs matched exactly. The retained fused HSACO SHA-256 is
`1e7d249dc0c11c412d2bf2d5c4755cc16e145fedea72046b26dc09a3d1656ad2`.
This historical oracle establishes neither a qualified direct-KFD gfx950 launch
path nor public-runtime gfx950 execution.

## Exact admitted comparator

[`gpt_oss_unfused.hip`](gpt_oss_unfused.hip) is an independent exact-semantics
comparator with three separately dispatched router, attention, and expert
kernels. It uses the same deterministic inputs, numerical oracle, dynamic
stable top-4 routing, and selected expert tile. It is an admitted comparator
for this tutorial shape, not a framework or full-model baseline.

```bash
unset HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES=6 examples/gfx950_gpt_oss_decode/run-unfused-gfx950.sh
```

Its GPU6 run had attention maximum absolute error `1.490116119e-8`, exact
expert output, and exact packed top-4 IDs. The three-kernel HSACO SHA-256 is
`4be1e6224fb8c18c93bed1fe64c38641b8c392b2cae966803c0d167444c4782a`.

## Historical optimization contribution

This table is retained from the former single-Wave64 geometry. It separates
single-process optimization smoke measurements from the
five-process comparison below. Smoke measurements used the same ROCr dispatch
timer and numerical checks, but are useful only for attribution within this
artifact.

| Candidate | Change | VGPRs | Median dispatch | Contribution |
| --- | --- | ---: | ---: | ---: |
| Fused baseline | Retain all four MXFP4 accumulator fragments until the final scale-and-sum | 352 | `1.240483 ms` | Reference |
| Fused qualification candidate | Scale and consume each MXFP4 block before constructing the next fragment | 308 | `1.065242 ms` | `14.1%` lower smoke median |
| Rejected router experiment | Replace the scalar router with padded `16x16` BF16 MFMA tiles | not retained | `2.174 ms` | `104.1%` slower than the optimized smoke result |

Sequential fragment consumption removes 44 VGPRs and shortens the lifetime of
three complete matrix results. The native-router experiment was rejected: a
batch-1 router has one useful row, while each `16x16` MFMA executes 16 rows and
requires padding and extra operand movement. Native matrix instructions alone
do not make that low-utilization shape faster.

## Historical reproducible performance result

The following command and results apply to the former single-Wave64 artifact,
not the current 16-item geometry. The campaign used 1,000 initial
warmups, 30 blocks of 100 samples, and 20 rewarm dispatches before every block:

```bash
unset HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES=6 \
  perf-evidence/run-gpt-oss-performance.sh \
  /home/harmenon/perf-runs/gpt-oss-layer-tile-$(date -u +%Y%m%dT%H%M%SZ)
```

On 2026-08-29, ROCr HSA dispatch timestamps on physical GPU 6 of
`smci350-rck-g03-b19-03` reported:

| Exact artifact | Samples | Median | Hierarchical bootstrap 95% CI | p5 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Rust fused qualification candidate | 15,000 | `1.064644 ms` | `[1.064483, 1.064844] ms` | `1.059803 / 1.069283 ms` |
| Exact HIP router + attention + expert sum | 15,000 triplets | `0.780362 ms` | `[0.780243, 0.780482] ms` | `0.778162 / 0.783123 ms` |

The exact unfused sequence is `0.7330x` the fused duration, or the fused
candidate is `1.3643x` slower. Therefore this result does **not** support a
fastest or state-of-the-art claim, even among the admitted exact artifacts.
It shows that eliminating two dispatches does not offset the fused kernel's
long dependency chain and register pressure at this one-wave batch-1 shape.

The campaign source commit is `c1383e97db732f9f1ff8105f10d5c2b5971143e1`.
[`perf-evidence/gpt-oss-layer-tile-evidence-v1.json`](../../perf-evidence/gpt-oss-layer-tile-evidence-v1.json)
pins the summaries and retained raw-record hashes. `amd-smi` observations found
no process entry with nonzero GPU memory, activity, or CU occupancy before or
after a measured wrapper; clocks were not locked.

## Historical single-item theoretical resource floor

One item in the former single-Wave64 profile has 1,509,972 unique compulsory
bytes:

| Data | Bytes |
| --- | ---: |
| Router weights, `128 * 2880 * sizeof(f32)` | 1,474,560 |
| Hidden vector | 11,520 |
| BF16 query / depth-major key | 2,048 / 2,048 |
| FP32 value tile / sinks | 1,024 / 64 |
| Four storage-expanded FP4 activation / selected-weight tiles | 8,192 / 8,192 |
| Activation / selected-weight scales | 16 / 256 |
| Attention / expert / packed-ID outputs | 1,024 / 1,024 / 4 |

This is a unique-data lower bound. It intentionally does not count cache-line
transactions or duplicate lane loads, so it is optimistic. The exact unfused
sequence adds one four-byte read of the packed route between stages.

The audited arithmetic is 737,280 FP32 router FLOPs, 32,768 executed BF16 QK
FLOPs, 8,192 FP32 value-accumulation FLOPs, and 262,144 executed MXFP4 MFMA
FLOPs. Only 4,096 of the MXFP4 FLOPs belong to the non-padding batch-1 row.
Exponential, division, comparison, packing, and control operations are stated
separately rather than misclassified as peak matrix FLOPs.

Using the deliberately optimistic whole-device inputs in
[`mi350x-bound-inputs-v1.json`](../../perf-evidence/mi350x-bound-inputs-v1.json):

```text
HBM floor   = 1,509,972 / 8e12       = 188.7465 ns
FP32 floor  =   745,472 / 144.2e12   =   5.1697 ns
MXFP4 floor =   262,144 / 9.2e15     =   0.0285 ns
T_resource  = max(above)             = 188.7465 ns
```

The bound file does not provide a BF16 peak, so omitting that term makes the
reported floor more optimistic. The qualification median is about `5,641x`
this resource floor; the comparator is about `4,134x`. These ratios do not mean
either kernel can approach the bound: `T_resource` assumes full-device
occupancy, while that measured artifact launched one dependent Wave64 workgroup and had
unmodeled scalar, transcendental, instruction-latency, and dispatch costs. No
single-workgroup latency bound is fabricated from the throughput roofline.

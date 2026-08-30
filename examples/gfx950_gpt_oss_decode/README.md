# GPT-OSS-120B batch-1 layer-tile megakernel

This tutorial is a production safe-Rust-to-gfx950 example for a bounded piece
of one GPT-OSS-120B decode layer. It is not a whole-model megakernel and does
not claim to run an entire transformer layer in one workgroup.

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

The launch is one Wave64 workgroup for batch 1 and grid `[1, 1, 1]`. It
computes:

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
The host does not preselect an expert. Each lane computes two router logits,
then the wave broadcasts the two `(score, expert_id)` candidates from every
lane and maintains stable top-4 state. This preserves the full 128-way dynamic
routing decision while avoiding a lane-0 serial router.

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
4. Attention output, expert output, and four packed seven-bit expert IDs are
   committed through disjoint output capabilities.

The production runner requires exactly four
`v_mfma_f32_16x16x16_bf16` sites and four
`v_mfma_f32_16x16x128_f8f6f4` sites with the FP4 selectors. K is supplied in
the depth-major layout consumed by the BF16 B fragment, so an LDS transpose is
not applicable to this fixed decode interface; the runner rejects unexpected
transpose instructions rather than claiming one was used.

[`src/reference.rs`](src/reference.rs) is an independent safe CPU oracle. It
implements BF16 and OCP E2M1 decoding, the full router and tie rule,
sink-softmax attention, block scaling, and selected-expert multiplication
without calling device helpers or GPU libraries. Deterministic inputs vary by
axis and route to expert 127.

## Build and numerical validation

Run source and CPU-oracle tests:

```bash
cargo test --manifest-path examples/gfx950_gpt_oss_decode/Cargo.toml
```

Build the ordinary Rust kernel through the production extractor, semantic MIR,
Kernel IR, LLVM, COV6 HSACO, symbol-scoped ISA checks, and the HSA numerical
test on physical GPU 6:

```bash
unset HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES=6 examples/gfx950_gpt_oss_decode/run-gfx950.sh
```

The validated MI350X run checked all outputs, immutable inputs, output
canaries, ABI size, artifact digests, and exact `gfx950:xnack-` metadata. Its
maximum absolute attention error was `8.940696716e-8`; expert output and packed
top-4 IDs matched exactly. The retained fused HSACO SHA-256 is
`1e7d249dc0c11c412d2bf2d5c4755cc16e145fedea72046b26dc09a3d1656ad2`.

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

## Optimization contribution

The table separates single-process optimization smoke measurements from the
five-process comparison below. Smoke measurements used the same ROCr dispatch
timer and numerical checks, but are useful only for attribution within this
artifact.

| Candidate | Change | VGPRs | Median dispatch | Contribution |
| --- | --- | ---: | ---: | ---: |
| Fused baseline | Retain all four MXFP4 accumulator fragments until the final scale-and-sum | 352 | `1.240483 ms` | Reference |
| Fused production | Scale and consume each MXFP4 block before constructing the next fragment | 308 | `1.065242 ms` | `14.1%` lower smoke median |
| Rejected router experiment | Replace the scalar router with padded `16x16` BF16 MFMA tiles | not retained | `2.174 ms` | `104.1%` slower than the optimized smoke result |

Sequential fragment consumption removes 44 VGPRs and shortens the lifetime of
three complete matrix results. The native-router experiment was rejected: a
batch-1 router has one useful row, while each `16x16` MFMA executes 16 rows and
requires padding and extra operand movement. Native matrix instructions alone
do not make that low-utilization shape faster.

## Reproducible performance result

Run the exact five-process alternating AB/BA campaign with 1,000 initial
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
| Production Rust fused | 15,000 | `1.064644 ms` | `[1.064483, 1.064844] ms` | `1.059803 / 1.069283 ms` |
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

## Theoretical resource floor

The fused profile has 1,509,972 unique compulsory bytes:

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
reported floor more optimistic. The production median is about `5,641x` this
resource floor; the comparator is about `4,134x`. These ratios do not mean
either kernel can approach the bound: `T_resource` assumes full-device
occupancy, while this tutorial launches one dependent Wave64 workgroup and has
unmodeled scalar, transcendental, instruction-latency, and dispatch costs. No
single-workgroup latency bound is fabricated from the throughput roofline.

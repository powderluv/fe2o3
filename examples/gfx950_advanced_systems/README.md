# gfx950 advanced systems kernels

The ordinary attributed Rust kernels in [`src/kernel.rs`](src/kernel.rs) are
the fe2o3 source for these tutorials. [`src/reference.rs`](src/reference.rs)
contains independent safe CPU references, and `cargo test --offline` checks
their bounded numerical and transactional contracts. The HIP program remains
a separate compiler, ISA, and MI350 hardware-validation companion.

Each `run-*-gfx950.sh` entry point selects exactly one kernel feature, invokes
the production fe2o3 extractor, checks the compiler-published crate binding,
links an exact gfx950:xnack- COV6 HSACO, validates its single-kernel metadata
and symbol-scoped ISA, and runs a digest-pinned numerical HSA test. This is an
explicit verification path; it does not grant protected artifact publication
authority. The HIP results below remain independent evidence.

The fixed-shape suite covers several systems patterns on AMD CDNA 4 (`gfx950`).
It is deliberately small enough to have independent, deterministic CPU oracles;
it is not a framework, a distributed runtime, or a production collective
implementation.

The executable covers:

- A 16-token, 128-input, 16-output fused FP4/FP8 MoE. It computes router
  logits, stable top-2 selection (lower expert ID wins exact ties), compact
  per-expert dispatch metadata, five expert tiles with a gfx950 mixed
  FP4/FP8 `16x16x128` scaled MFMA, SiLU, routed weighting, and a shared-expert
  contribution.
- Two logical rank-local expert partitions. The compact fixture copies its full
  five-expert weight tensor to each device, but each rank kernel reads only its
  assigned two-expert partition. With two visible gfx950 devices, the suite
  executes one rank per device and uses peer copies when HIP exposes
  bidirectional peer access, including a peer return and GPU combine on device
  zero. Otherwise it uses a deterministic, host-staged transport simulation.
  Rank results are reduced in fixed order and checked against independently
  decoded CPU expert GEMMs. This validates the bounded execution path, not an
  expert-parallel communication library.
- Eight four-token speculative/MTP candidates with deterministic prefix
  acceptance. Recurrent/KV-style state is committed only after full acceptance;
  all other candidates roll back to the byte-identical base state.
- A GPU-resident, 16-slot Qwen-style 3-gram hash table gather. Lookup scans in
  linear-probe order, verifies the complete key, selects higher priority on
  duplicate keys, and then the lower slot on a tie. Host-offload overlap,
  eviction, and table construction are outside this kernel's scope.
- A 4x4 Muon update from two sharded gradient contributions. GPU kernels stage
  each rank, while the host stages a deterministic rank-order reduction before
  a GPU Frobenius norm and five Newton-Schulz/polar iterations. This is
  explicitly not a GPU collective or a distributed optimizer runtime.

Every input is deterministic. The executable compares outputs, routing
metadata, accepted prefixes, transaction states, hash-table results, norms,
and optimizer updates to separately implemented CPU references. Runtime
execution rejects devices whose GCN architecture is not exactly `gfx950`.

Run the production Rust lowering and numerical verification on a gfx950 host:

```bash
./run-moe-route-gfx950.sh
./run-moe-expert-rank-gfx950.sh
./run-combine-expert-ranks-gfx950.sh
./run-speculative-transaction-gfx950.sh
./run-qwen-ngram-gather-gfx950.sh
./run-stage-gradient-shard-gfx950.sh
./run-muon-update-gfx950.sh
```

The expert runner requires exactly three mixed FP4/FP8
`v_mfma_f32_16x16x128_f8f6f4` instructions with `cbsz:4`: two rank-local
experts and the optional shared expert. Routing and expert exponential math
links only the reviewed ROCm 7.2.1 OCML `exp` closure shared with the
low-precision examples; Muon square root uses the gfx950 native LLVM intrinsic.
The per-kernel harness checks immutable inputs, output canaries, exact integer
and rollback state, and bounded floating-point tolerances against the CPU
references. Set `FE2O3_REPO_ROOT`, `ROCM_PATH`, `RUSTUP`, `CARGO`, or the
documented tool and target-directory environment variables when validating a
copied checkout.

## Production Rust optimization evidence

The 2026-08-29 optimization pass used only physical GPU 6 on SSH host `mi350`
(`ROCR_VISIBLE_DEVICES=6`, with `HIP_VISIBLE_DEVICES` unset) and ROCm 7.2.1.
Every timed kernel first passed the existing CPU oracle, immutable-input checks,
and output canaries. Timings are ROCr HSA dispatch timestamps after 200 warmups,
five blocks of 50 samples, and ten block rewarm dispatches. They are bounded,
single-process ablations, not publishable fastest-kernel claims. Exact records,
artifact digests, and the machine-readable values below are in
[`optimization-evidence-v1.json`](optimization-evidence-v1.json).

The retained-candidate subset was collected with:

```bash
ROCR_VISIBLE_DEVICES=6 \
FE2O3_GFX950_ADVANCED_PERF_WARMUPS=200 \
FE2O3_GFX950_ADVANCED_PERF_BLOCKS=5 \
FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK=50 \
FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM=10 \
  ../../perf-evidence/run-gfx950-advanced-performance.sh \
  /home/harmenon/perf-runs/systems-candidate-ablation-v2 \
  ./run-combine-expert-ranks-gfx950.sh \
  ./run-speculative-transaction-gfx950.sh \
  ./run-qwen-ngram-gather-gfx950.sh \
  ./run-stage-gradient-shard-gfx950.sh \
  ./run-muon-update-gfx950.sh
```

The same four sampling variables were passed to
`../../perf-evidence/run-published-baseline-artifact.sh` for each digest-pinned
baseline. The final branchless N-gram source was rerun after commit as campaign
`systems-ngram-retained-v4`, so its source identity does not point at a dirty
worktree.

| Kernel | Exact-semantics optimization | Baseline | Retained | Contribution |
| --- | --- | ---: | ---: | ---: |
| route | Compute each token route once per wave, broadcast choices, and reuse a packed route map for counts/dispatch | blocked | blocked | not measurable |
| expert rank | One Wave64, blocked four-element ownership, three mixed FP4/FP8 MFMAs instead of three per wave across four waves | blocked | blocked | not measurable |
| combine | Tested one-wave blocked ownership; retained the exact 256-thread mapping after proof rejection | 5.00 us | 5.00 us | 0 ns, 1.00x |
| speculative | Compute the acceptance prefix once, broadcast it to the eight state elements, and remove the duplicate acceptance path | 8.00 us | 5.68 us | -2.32 us, 1.41x |
| N-gram | Tested hash-first short-circuiting; retained the branchless full-key probe because the candidate regressed | 8.52 us | 8.52 us | 0 ns, 1.00x |
| stage | Retained the existing one-Wave64 exact copy | 5.00 us | 5.00 us | 0 ns, 1.00x |
| Muon | Distribute one matrix element per lane and exchange values with Wave64 reductions/broadcasts | 5.72 us | 5.04 us | -0.68 us, 1.13x |

The rejected N-gram hash-first variant measured 10.68 us, 2.16 us slower than
the 8.52 us branchless baseline. That result is why the final source deliberately
does not contain the superficially attractive early branch. Combine and stage
also remain unchanged: their approximately 5 us time is dispatch-latency
dominated at this one-workgroup scale.

The retained speculative artifact reduces static global instructions from 16
to 10, VALU instructions from 77 to 62, VGPRs from 27 to 17, and SGPRs from 66
to 62. Distributed Muon reduces total static instructions from 1,360 to 593,
VALU instructions from 969 to 160, branches from 21 to 3, VGPRs from 62 to 19,
and SGPRs from 58 to 22; its 86 `ds_bpermute` instructions replace redundant
per-lane 4x4 matrix evaluation. Combine remains 291 instructions, N-gram 878,
and stage 282. Neither retained measured kernel changes the floating-point
accuracy contract or enables fast math.

### Resource lower bound

For each fixed fixture, the optimistic whole-device resource floor is

```text
max(logical_bytes / 8 TB/s,
    FP32_ops / 144.2 TFLOP/s,
    mixed_FP4_FP8_ops / 4.6 PFLOP/s)
```

The mixed rate uses the lower FP8 peak rather than the 9.2 PFLOP/s MXFP4 peak.
Logical bytes count unique fixed-fixture reads and writes; the speculative
fixture counts only the proposed deltas read by its two committed candidates.
This is a cold logical-payload bound, not a prediction for a warm, persistent,
single-workgroup dispatch. It intentionally excludes queue/signal latency,
instruction dependencies, and cache effects.

| Kernel | Logical bytes | Counted operations | Resource floor | Retained / floor |
| --- | ---: | ---: | ---: | ---: |
| route | 4,880 | 16,384 FP32 routing-dot ops | 0.610 ns | unavailable |
| expert rank | 9,472 | 196,608 mixed MFMA ops | 1.184 ns | unavailable |
| combine | 3,072 | 256 FP32 adds | 0.384 ns | 13,021x |
| speculative | 896 | 64 committed-state FP32 adds | 0.112 ns | 50,714x |
| N-gram | 576 | integer/hash work | 0.072 ns | 118,333x |
| stage | 128 | copy only | 0.016 ns | 312,500x |
| Muon | 196 | 1,649 FP32 ops | 0.0245 ns | 205,714x |

The very large ratios are expected: these fixtures launch one tiny workgroup,
so a whole-device bandwidth/compute roof is much lower than the practical HSA
dispatch floor. The resource bound is useful for checking arithmetic and byte
assumptions, not for claiming that this microbenchmark can occupy all 256 CUs.

### Unresolved production lowering

Host contracts pass for all seven sources. Five production wrappers (combine,
speculative, N-gram, stage, and Muon) lower and run numerically on GPU 6. Route
and expert rank remain blocked by compiler validation; verification was not
weakened to admit them:

- Route: `/tmp/systems-route-helper.log` reports `ConstantIndex projection on
  a non-array place in block 596, statement 0` during semantic body construction.
- Expert rank: `/tmp/systems-expert-rank-fixed.log` reports
  `FE2O3-RACE-002` at block 27 op 0, with unresolved access dimension `v109`
  because the checked structured-index marker lacks a validated value contract.
  The pre-optimization source also has the independent current-main regression
  retained in `/tmp/systems-expert-helper-gated.log`: semantic MIR effect
  `<block=43, statement=None, ordinal=0>` has no ranked PLIRON counterpart.

Because expert rank does not produce a verified HSACO yet, the three source MFMA
calls are not presented as post-optimization ISA evidence. Once the structured
effect projection is fixed, its wrapper still requires exactly three
`v_mfma_f32_16x16x128_f8f6f4` instructions with `cbsz:4` before execution.

Run the independent HIP compiler, target-rejection, ISA, and numerical suite:

```bash
./build_and_test.sh
```

Run only compilation and symbol-scoped ISA validation on another compiler host:

```bash
./build_and_test.sh --compile-only
```

`check_isa.sh` requires all seven kernel symbols and checks that the fused expert
tile contains `v_mfma_f32_16x16x128_f8f6f4` with the FP4 format selector
`cbsz:4`. The build also requires compilation for `gfx942` to fail, preventing
this gfx950-only example from silently targeting an older architecture.

## Validation evidence

On 2026-08-26, `./build_and_test.sh` passed through SSH host `mi350`
(`smci350-rck-g03-b19-03`) with ROCm 7.2.1, HIP 7.2.53211, AMD Clang 22,
and eight visible AMD Instinct MI350X devices. The peer-access path selected
`mode=two-device-peer`; Muon selected its explicit
`mode=two-device-host-staged` reduction. The retained run reported zero error
for fused MoE, routing weights, speculative state, staged Muon shards, and the
Muon norm. Its largest expert-partition error was `4.76837e-07`, and its Muon
update error was `4.65661e-09`. The N-gram test reported four hits, four misses,
and deterministic duplicate-key winner `4242`.

The tested HIP source SHA-256 was
`c29a6bc2de55563abddfb50f43aaccf6077ef0b4706fbfb314266ecaa48054c5`.
The retained gfx950 HSACO SHA-256 was
`5ccc37902f9b549ac405f1096ad6df8ea58eba5dd6a08c765f5ea3148eb47d16`.
Symbol-scoped disassembly found exactly one
`v_mfma_f32_16x16x128_f8f6f4` in the fused MoE kernel, with `cbsz:4`.

# gfx950 advanced systems kernels

The ordinary attributed Rust kernels in [`src/kernel.rs`](src/kernel.rs) are
the fe2o3 source for these tutorials. [`src/reference.rs`](src/reference.rs)
contains independent safe CPU references, and
`cargo fe2o3 test --offline --all-targets` checks
their bounded numerical and transactional contracts. The HIP program remains
a separate compiler, ISA, and MI350 hardware-validation companion.

Each `run-*-gfx950.sh` entry point selects exactly one kernel feature, invokes
the production fe2o3 extractor, checks the compiler-published crate binding,
links an exact gfx950:xnack- COV6 HSACO, validates its single-kernel metadata
and symbol-scoped ISA, and runs a digest-pinned numerical HSA test. This is an
explicit verification path; it does not grant protected artifact publication
authority. The HIP results below remain independent evidence.

The fixed-shape suite covers several systems patterns on AMD CDNA 4 (`gfx950`).
Each wave-oriented launch contains 16 independent, batch-local instances: four
WG256 workgroups each execute four Wave64 owners. The pointwise expert-rank
combine instead contains four independent 256-element batches, one per WG256.
Every input and output is batch-major, so no workgroup depends on another or
writes another workgroup's elements. The suite remains deliberately small
enough to have independent, deterministic CPU oracles; it is not a framework,
a distributed runtime, or a production collective implementation.

The executable covers:

- Sixteen independent 16-token, 128-input, 16-output fused FP4/FP8 MoE
  instances, one per wave. Each computes router
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
- Sixteen independent groups of eight four-token speculative/MTP candidates
  with deterministic prefix acceptance. Recurrent/KV-style state is committed
  only after full acceptance; all other candidates roll back to the
  batch-local, byte-identical base state.
- Sixteen independent GPU-resident, 16-slot Qwen-style 3-gram hash-table
  gathers. Lookup scans in
  linear-probe order, verifies the complete key, selects higher priority on
  duplicate keys, and then the lower slot on a tie. Host-offload overlap,
  eviction, and table construction are outside this kernel's scope.
- Sixteen independent 4x4 Muon updates from two sharded gradient contributions.
  GPU kernels stage
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

## Public paired-source milestone

The explicit modes select existing ordinary Rust implementations for all seven
kernels. Route, combine, speculative transaction, N-gram, shard staging and Muon
retain their canonical SIMT bodies and add checked structured-tile selections.
Expert rank retains its canonical mixed FP4/FP8 matrix-tile body and selects the
existing raw-byte FP4/FP8 SIMT decoder with `simt`. No additional numerical
format is advertised.

| Selector | SIMT feature | Tile feature |
| --- | --- | --- |
| `moe-route` | `kernel-moe-route` | `kernel-moe-route-tile` |
| `moe-expert-rank` | `kernel-moe-expert-rank-simt` | `kernel-moe-expert-rank` |
| `combine-expert-ranks` | `kernel-combine-expert-ranks` | `kernel-combine-expert-ranks-tile` |
| `speculative-transaction` | `kernel-speculative-transaction` | `kernel-speculative-transaction-tile` |
| `qwen-ngram-gather` | `kernel-qwen-ngram-gather` | `kernel-qwen-ngram-gather-tile` |
| `stage-gradient-shard` | `kernel-stage-gradient-shard` | `kernel-stage-gradient-shard-tile` |
| `muon-update` | `kernel-muon-update` | `kernel-muon-update-tile` |

From this package directory, compile and inspect each exact mode without
launching a GPU kernel:

```bash
for selector in moe-expert-rank moe-route combine-expert-ranks \
    speculative-transaction qwen-ngram-gather stage-gradient-shard muon-update; do
    for mode in simt tile; do
        FE2O3_EXAMPLE_COMPILE_ONLY=1 bash "run-$selector-gfx950.sh" "$mode"
    done
done
```

On a host with ROCm 7.2.4, explicitly select the checked-in closure with
`FE2O3_GFX950_OCML_MANIFEST=../gfx950_low_precision/gfx950-ocml-rocm-7.2.4.manifest`.
The default remains the reviewed ROCm 7.2.1 closure. The unchanged independent
CPU contracts run with `cargo fe2o3 test --offline --all-targets`; individual
feature selections can additionally be passed using `--no-default-features
--features <feature>`.

This source milestone uses public compiler foundation
[PR #321](https://github.com/harsh-nod/fe2o3/pull/321),
commit `3283b661e901d81ea30ca83ab6dd23ccc9ec0ad8`, as an explicit
publication dependency. It does not claim that dependency was present in the
baseline main commit `0e0963df4f230bf728a7b4605677c0ff09624adf`.
The bounded compile campaign uses copied public compiler products built from
`31ed896aa91bac004f4d760ad55f11345e59ed60`, whose compiler closure matches
the dependency tree, with toolchain `nightly-2026-04-03` and the explicit
ROCm 7.2.4 manifest.

Canonical main's documented CPU command first failed its normal executable
hashing guard: the default test image was 906,756,144 bytes, over the unchanged
536,870,912-byte limit. The standalone `[profile.test] debug = 0` setting
retains assertions, overflow checking and every oracle. No `CARGO_PROFILE_*`
environment override or CLI limit increase is needed.

Fixture source bindings describe exact selections. Simulation, policy,
optimized-graph and fresh hardware qualification remain separate pending gates.
The new counterpart simulation fixtures remain `pending-design`; their source
bindings do not supply execution receipts. No GPU was launched for this
publication task. No gfx942 execution or paired-source qualification is
claimed. The historical GPU/performance campaigns below retain their original
source and artifact identities and do not qualify these new mode selections.



## Current GPU-5 performance audit

The 2026-09-05 campaign measured the current WG256/grid-`[4, 1, 1]` Rust
artifacts on physical GPU 5 of `ssh mi350` with
`ROCR_VISIBLE_DEVICES=5` and `HIP_VISIBLE_DEVICES` unset. Each series used 200
warmups, five blocks of 100 ROCr HSA dispatch timestamps, and 20 rewarm
dispatches per block. Every artifact passed its independent CPU oracle before
timing and after every block, including immutable-input and output-canary
checks.

| Kernel | Current median / p95 | Exact ablation | Measured contribution | Public comparison verdict |
| --- | ---: | --- | --- | --- |
| MoE route | `17.320 / 17.680 us` | redundant four-lane reductions: `21.120 us` | depth striping and uniform partial-sum broadcasts: `1.219x`, `17.99%` lower | Excluded: public fused-MoE paths do not match E4/top-2/K128/N16 routing semantics |
| Expert-rank GEMM | rank 0 `55.440 / 63.880 us`; rank 1 `55.621 / 63.680 us` | serial MFMA schedule: rank 0 `58.660 us`; rank 1 `53.260 us` | Rank-dependent reversal; no stable gain claimed | Excluded: AITER/CK grouped GEMMs do not match routing, shared-expert, shape, and output semantics |
| Expert combine | `5.400 / 5.600 us` | none admitted | Direct coalesced fixed-order add retained | Excluded: no public kernel has this standalone fixed-order boundary |
| Speculative transaction | `11.120 / 12.080 us` | prefix recompute: `10.960 us` | `1.46%` reversal conflicts with the earlier campaign; no stable gain claimed | Excluded: vLLM/SGLang use different state, batching, and acceptance contracts |
| Qwen N-gram gather | `12.880 / 13.920 us` | reverse probe: `15.160 us` | ascending insertion-order probing: `1.177x`, `15.04%` lower | Excluded: serving paths use different table, cache, and proposal boundaries |
| Gradient staging | shards 0/1 `5.480 / 5.480 us` | none admitted | Direct coalesced copy retained | Excluded: this is not a collective or model inference operator |
| Muon update | `6.800 / 7.040 us` | broadcast16: `6.800 us` | Median tie; no sub-resolution gain claimed | Excluded: public distributed Muon kernels use production matrix sizes |

For route, the current kernel assigns the four lanes associated with a token
to disjoint depth residues. The lanes issue one quarter of the activation and
router-weight loads, then use twelve uniform Wave64 broadcasts to merge four
FP32 partial sums. The live `ablation-route-redundant-lanes` feature restores
the former full K128 reduction in every lane and provides the exact measured
baseline. The N-gram result confirms that forward linear probing matches table
construction better than the otherwise equivalent reverse walk. The other
rows record ties, reversals, or unavailable admitted variants rather than
inventing an optimization contribution.

[`sota-performance-audit-v1.json`](sota-performance-audit-v1.json) is the
machine-readable source for timings, artifact and raw-record SHA-256 digests,
public baseline pins, model-relevance limits, and the per-operator optimistic
resource floors. The floor is
`max(unique bytes / 8 TB/s, FP32 FLOPs / 144.2 TFLOP/s, low-precision FLOPs /
the named peak)`. It deliberately omits dispatch latency, cache-line
amplification, dependencies, scalar/transcendental work, and occupancy limits;
the measured-to-floor ratios therefore diagnose these tiny teaching fixtures
rather than forecast production throughput. No external SOTA claim is made.

The expert runner requires exactly three mixed FP4/FP8
`v_mfma_f32_16x16x128_f8f6f4` instructions with `cbsz:4`: two rank-local
experts and the optional shared expert. Routing and expert exponential math
links only the selected reviewed OCML `exp` closure shared with the
low-precision examples. ROCm 7.2.1 remains the default and the checked-in ROCm
7.2.4 manifest is an explicit alternative; Muon square root uses the gfx950
native LLVM intrinsic.
The per-kernel harness checks immutable inputs, output canaries, exact integer
and rollback state, and bounded floating-point tolerances against the CPU
references. Set `FE2O3_REPO_ROOT`, `ROCM_PATH`, `RUSTUP`, `CARGO`, or the
documented tool and target-directory environment variables when validating a
copied checkout.

## Multi-workgroup source status

The Rust launch contracts require WG256 and admit at most four workgroups in
the x dimension. The corresponding production hardware harness must launch
exactly four workgroups; the CPU references concatenate all batch-local
observations in the same order. Wave collectives remain scoped to their owning Wave64: there is
no grid barrier, cross-workgroup reduction, atomic output ownership, or
dependence on workgroup scheduling order.

This topology change does not inherit or fabricate performance evidence. The
historical measurements below were collected from earlier single-workgroup
artifacts and must not be interpreted as measurements of the batched kernels.
New timings require fresh digest-pinned HSACO, correctness, ISA, and dispatch
records from the four-workgroup launch.

The 2026-09-03 compatibility rerun also exercised the four live ablation bodies
through production extraction, LLVM, HSACO, ISA inspection, and HSA at
WG256/grid4. `expert-serial`, `speculative-recompute-prefix`,
`ngram-reverse-probe`, and `muon-broadcast16` all matched the same batched CPU
oracles as their canonical kernels. This is compatibility and correctness
evidence only; the minimal one-sample runner settings are not performance
evidence. `combine-transposed` still fails closed before GPU launch because
the projection cannot prove an arithmetic-overflow assertion for its dynamic
source index. `stage-tile4` still fails closed because its subgroup broadcast
source lane is not statically bounded. The older `route-owner-only` and
`route-unpacked` feature names remain documentation-only historical
experiments. The current `route-redundant-lanes` feature selects a live
alternate Rust body and is part of the GPU-5 performance audit above.

## Live ablations and explicit execution modes

The original selectors above remain runnable. For the MoE pair, select the
same ablation together with an explicit mode:

```bash
FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=expert-serial bash run-moe-expert-rank-gfx950.sh simt
FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=expert-serial bash run-moe-expert-rank-gfx950.sh tile
FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=route-redundant-lanes bash run-moe-route-gfx950.sh simt
FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=route-redundant-lanes bash run-moe-route-gfx950.sh tile
```

Expert tile mode retains the original checked FP4/FP8 matrix views and three
MFMAs. Its serial variant loads and evaluates each expert before loading the
next. Expert SIMT mode decodes the same raw bytes and, with `expert-serial`,
completes three separate K128 F32 dot products in first/second/shared order.
All three products are computed even when shared inclusion is disabled.
Canonical SIMT keeps its original interleaved three-accumulator loop.

Route tile mode uses checked `Tiled2D` stores for every route, weight, count,
and dispatch output. Its `route-redundant-lanes` variant retains the original
full K128 dot product in every lane, instead of four cooperating K32 partials.

`speculative-recompute-prefix` has source branches in both explicit modes.
`muon-broadcast16` and `ngram-reverse-probe` select the same existing arithmetic
in their original SIMT paths and checked output-tile counterparts:

```bash
FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=muon-broadcast16 bash run-muon-update-gfx950.sh tile
FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=ngram-reverse-probe bash run-qwen-ngram-gather-gfx950.sh tile
```

Add `FE2O3_EXAMPLE_COMPILE_ONLY=1` to compile and inspect without GPU execution;
omitting the mode argument retains the original SIMT selector except for
expert rank, which defaults to its original tile selector. Muon retains
the ordered sum of 16 broadcast squares and all five Newton-Schulz iterations.
N-gram retains reverse traversal of every slot, full-key collision rejection,
highest priority, and lowest physical-slot tie breaking. Both tile paths use
checked `Tiled2D` stores with the original complete layout and output guards.
Only these exact two noncanonical tile pairs are admitted by the dispatcher;
`stage-tile4` still rejects, and the combine/route negative roles remain.
None of these mode descriptions extends the historical GPU or performance
evidence to new artifacts. Compile-only success is not GPU qualification. `combine-transposed`, `stage-tile4`,
`route-owner-only`, and `route-unpacked` retain their distinct historical or
negative roles described above.

## Historical exploratory Rust optimization evidence

This section records the 2026-08-29 exploratory campaign and its then-current
lowering state. The final da6 tutorial campaign in
`perf-evidence/gfx950-advanced-ablation-evidence-v1.json` supersedes these
latencies and production-status statements; each campaign keeps separate
source and artifact identities.

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

### Historical lowering blockers

The following diagnostics explain why the exploratory campaign did not time
route and expert rank. They are resolved in the promoted gfx950 source and are
not the status of the final compatibility campaign.

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

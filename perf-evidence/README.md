# gfx950 advanced-kernel performance evidence

This directory defines the reproducible evidence format for the 14 advanced
gfx950 kernels. It does not make a fastest or state-of-the-art claim. A claim
is admissible only after the candidate and a public, unmodified baseline pass
the same correctness oracle, workload, launch shape, machine, and protocol.

`gfx950-advanced-evidence-v1.json` pins the final 2026-08-29 candidate
campaign: 48,000 dispatch samples across all 14 wrappers, with current HSACO
identities, medians, confidence intervals, and retained raw-data hashes. Records
without an admitted exact comparator remain measurements, not fastest claims.

## Measurement contract

The hardware test first runs its existing digest-pinned, ABI-checked numerical
path. Performance collection is opt-in. The performance path then creates one
executable, queue, completion signal, kernarg allocation, and set of input and
output buffers per workload. It checks an untimed preflight, performs 1,000
initial warmups, and collects 30 blocks of 100 dispatches, with 20 unrecorded
rewarm dispatches before each block. Every timed block is admitted only after
the current repository CPU oracle and guard canaries pass.

Each duration comes from
`hsa_amd_profiling_get_dispatch_time`, after enabling queue profiling with
`hsa_amd_profiling_set_profiler_enabled`. Start/end ticks and the HSA system
timestamp frequency are retained as decimal strings. Host wall time is not a
kernel-duration substitute.

Run the complete default campaign on physical GPU 6:

```bash
cd /path/to/fe2o3
ROCR_VISIBLE_DEVICES=6 \
  perf-evidence/run-gfx950-advanced-performance.sh \
  /home/harmenon/perf-runs/advanced-$(date -u +%Y%m%dT%H%M%SZ)
```

Pass wrapper paths after the output directory to run a subset. Sampling can be
changed with `FE2O3_GFX950_ADVANCED_PERF_WARMUPS`,
`FE2O3_GFX950_ADVANCED_PERF_BLOCKS`,
`FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK`, and
`FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM`. Published tables must state
non-default values.

The exact published pre-optimization machine artifacts are retained outside
the repository in a content-addressed vault. Nine kernels had published HSACO;
the other five failed historical lowering and therefore have no fabricated
Rust baseline. Validate and time one retained artifact with:

```bash
mkdir -p /home/harmenon/perf-runs/published-baseline
FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID=published-baseline \
FE2O3_GFX950_ADVANCED_PERF_PROCESS=0 \
  perf-evidence/run-published-baseline-artifact.sh \
  gfx950_attnres_aggregate \
  /home/harmenon/perf-runs/published-baseline/samples.jsonl
```

`published-baseline-artifacts-v1.json` pins source and artifact identities,
and the runner verifies every content digest before loading the HSACO.

Analyze and validate the raw records:

```bash
python3 perf-evidence/analyze.py /home/harmenon/perf-runs/advanced-*/samples.jsonl
python3 perf-evidence/analyze.py candidate.jsonl baseline.jsonl \
  --baseline-variant baseline --candidate-variant candidate
```

The analyzer rejects duplicate IDs, false correctness fields, malformed
decimal timer fields, and timer conversions that disagree with the raw ticks.
It reports median, p5, p95, MAD, and a seeded hierarchical bootstrap 95% CI of
the median. Paired speedups match the exact kernel, workload, block, and sample.

## Rootless noise controls

Use an otherwise idle physical GPU. The runner records `amd-smi process`,
static identity/limits, and clock/power/temperature/performance-level metrics
before and after every wrapper. Do not pool a run when active GPU work, thermal
throttling, a changed performance level, ECC/RAS activity, or materially
different clocks are visible. CPU affinity and NUMA binding may be applied by
the caller with `taskset` and `numactl`; record those commands with the
campaign. Clock locking and power-cap changes require privileges and are not
assumed by this protocol.

Run candidate and baseline as alternating AB/BA pairs, not all of one followed
by all of the other. Use at least five fresh processes per variant for a
publishable comparison, setting
`FE2O3_GFX950_ADVANCED_PERF_PROCESS=0..4` for both variants. The process
index is part of every record ID and the hierarchical bootstrap. Preserve
every raw run, including rejected noisy runs, with an explicit rejection
reason.

## rocprofiler cross-check

ROCr dispatch timestamps are the primary low-overhead measurement. A separate,
non-comparable campaign can confirm kernel attribution with ROCprofiler:

```bash
mkdir -p /home/harmenon/perf-runs/rocprof-crosscheck
ROCR_VISIBLE_DEVICES=6 \
FE2O3_GFX950_ADVANCED_HIP_ORDINAL=6 \
/opt/rocm/bin/rocprofv3 --kernel-trace --stats \
  --output-format json csv \
  --output-directory /home/harmenon/perf-runs/rocprof-crosscheck \
  --output-file attnres -- \
  examples/gfx950_advanced_attention/run-attnres-aggregate-gfx950.sh
```

Do not compare profiler-instrumented durations with unprofiled records. Preserve
the ROCprofiler configuration and complete JSON/CSV outputs.

## Bounds and claims

`mi350x-bound-inputs-v1.json` contains the architecture constants. For a
workload with compulsory bytes `B` and operation counts `F_p`, the strict
global resource floor is:

```text
T_resource = max(B / 8e12,
                 F_fp32 / 144.2e12,
                 F_fp8 / 4.6e15,
                 F_mxfp4 / 9.2e15)
```

This is a whole-device, fully occupied roofline. It is not a latency bound for
the current single-workgroup tutorial shapes. A practical latency comparison
must separately measure an empty/minimal dispatch through this exact persistent
queue path and retain its raw records. Report both `T_resource` and that
measured dispatch floor; never replace the former with the latter or add them
without a justified overlap model. Instruction/latency bounds additionally
require an audited ISA instruction count and published gfx950 throughput or
dependency latency for each instruction class.

Allowed wording before those conditions are met:

> On the stated MI350X, workload, artifacts, and protocol, this candidate's
> median ROCr dispatch duration was X, with a hierarchical bootstrap 95% CI of
> [L, U], versus Y for baseline Z.

“Fastest” or “state of the art” additionally requires all material public
vendor and open-source baselines, version-pinned and tuned by their documented
procedure, across representative production shapes. These one-workgroup
tutorial workloads alone cannot support that claim.

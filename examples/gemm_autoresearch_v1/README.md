# GEMM autoresearch loop

This example adapts the single-edit-surface, fixed-evaluator pattern from
Karpathy's autoresearch project to an ordinary safe Rust GPU kernel. The agent
edits only the selected candidate: `src/kernel.rs` for TILE or
`src/kernel_simt.rs` for gfx950 SIMT. `evaluate.py` owns compilation,
correctness, timing, and the experiment ledger.

The candidate computes dynamic row-major BF16/BF16/F32 GEMM:

```text
C = alpha * A * B + beta * C
```

One wave64 workgroup owns a 16x16 output tile. The current candidate loads MFMA
fragments directly because the retained MI300X experiment found that a two-slot
LDS pipeline added cost without cross-wave reuse for this mapping. The rejected
pipeline is retained in `experiments/double_buffer.rs`.

## Run on MI300X

Choose one GPU that you are authorized to use and expose only that device:

```bash
ROCR_VISIBLE_DEVICES=5 python3 evaluate.py --label direct-mfma
```

The evaluator runs three fresh production passes. Each pass requires:

```text
safe Rust -> semantic MIR -> ranked PLIRON -> Kernel IR
          -> gfx942 LLVM -> HSACO -> launch -> CPU oracle
```

Only a passing candidate is timed. Eleven device-event samples are collected at
each of 256x256x256, 512x512x512, and 1024x1024x1024. Repeated launches amortize
the event resolution, and the scalar score is the geometric mean of throughput
across the three sizes. The final score is the median across processes.

Each production pass includes native compilation and a normal release-mode Cargo
host build. The finite default timeout is 1200 seconds per pass so a cold build is
not mistaken for a kernel failure. Set `--timeout-seconds` explicitly for a
different positive finite limit. Replicate logs are written while the runner is
active and retained on failure or timeout; timeout cleanup signals only that
runner's new process group. This does not reduce the three passes, benchmark
sizes, eleven samples, or correctness checks.

Copy `results.example.tsv` to the untracked `results.tsv` only if you want a
fresh ledger template. Build products live under `target/autoresearch-gemm-v1`.
Remove that directory when the session ends on a shared machine.

## Run on gfx950

The package-local runner selects a genuine direct-MFMA/checked-output TILE body
or a distinct scalar-per-output SIMT body. Both retain raw BF16 operands, F32
accumulation, dynamic strides, alpha/beta, and the independent full-output oracle.
Choose one authorized physical GPU and expose only that device:

```bash
ROCR_VISIBLE_DEVICES=<one-authorized-uuid> python3 evaluate.py --target gfx950 --mode tile --label direct-mfma-gfx950
ROCR_VISIBLE_DEVICES=<one-authorized-uuid> python3 evaluate.py --target gfx950 --mode simt --label scalar-gfx950
```

Both use the same three fresh production passes and eleven event samples at each
original 256/512/1024 size. The four correctness cases include strided tails,
raw-BF16 nonquarter values, cancellation, and all 256 fractional outputs, with
immutable inputs, padding/trailing canaries, and finite-result checks. The
rejected two-slot LDS experiment is not selected by either mode.

For CPU-only production compilation, run from this package directory:

```bash
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash run-gfx950.sh tile
FE2O3_EXAMPLE_COMPILE_ONLY=1 bash run-gfx950.sh simt
```

This does not claim GPU correctness or timing.
The host refuses an observed architecture/wave size different from the request;
the runner checks gfx950 wave64, the 80-byte ABI, and MFMA presence/absence.
`FE2O3_RUSTC_EXTRACTOR` may name an existing matching production extractor;
otherwise the runner builds it using the pinned toolchain. Historical gfx942
evidence below remains historical and is not relabeled as gfx950 evidence.

## CPU controls

After the normal repository toolchain and `cargo-fe2o3` setup, fetch the locked
dependencies once, then run these commands from this package directory:

```bash
cargo-fe2o3 test --locked --offline --all-targets
cargo-fe2o3 test --locked --offline --all-targets --no-default-features --features kernel-simt-autoresearch
python3 -B -m unittest discover -p test_evaluate.py -v
```

The package's test profile omits debug symbols to keep host test executables
within the existing pinned test-runner size bound. Assertions and overflow
checks remain enabled; no `CARGO_PROFILE_*` overrides are needed. These are CPU
reference, mapping, parser, and timeout controls, not GPU execution evidence.

## Evidence boundary

The checked-in MI300X record demonstrates that the tutorial and two displayed
variants compiled, lowered, launched, and passed the bounded oracle. The timing
was collected while unrelated workloads occupied every GPU, so it teaches the
decision protocol but is not a clean performance publication or a SOTA claim.

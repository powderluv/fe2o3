#!/usr/bin/env python3
"""Fixed correctness and performance evaluator for the GEMM edit surface."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
from collections import Counter
import subprocess
import sys
import time

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
CANDIDATE = EXAMPLE_DIR / "src" / "kernel.rs"
RUNNER = EXAMPLE_DIR / "run-gfx942.sh"
PASS_RE = re.compile(
    r"^PASS (?P<kernel>gemm_autoresearch(?:_simt)?_v1) case=(?P<case>[a-z-]+) "
    r"target=(?P<target>gfx942|gfx950): (?P<shape>[^,]+), .*max_abs_error=(?P<error>[0-9.eE+-]+)$",
    re.MULTILINE,
)

BENCH_RE = re.compile(
    r"^AUTORESEARCH_BENCH size=(?P<size>\d+) median_us=(?P<median>[0-9.]+) "
    r"p10_us=(?P<p10>[0-9.]+) p90_us=(?P<p90>[0-9.]+) gflops=(?P<gflops>[0-9.]+)$",
    re.MULTILINE,
)
SCORE_RE = re.compile(
    r"^AUTORESEARCH_SCORE geometric_mean_gflops=(?P<score>[0-9.]+)$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str],
               timeout: float, raw_path: Path) -> subprocess.CompletedProcess[str]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException as error:
            # Keep the session leader unreaped until both group signals are sent.
            # Its PID cannot be reused for a different process group meanwhile.
            try:
                if process.returncode is None:
                    for sig in [signal.SIGTERM, signal.SIGKILL]:
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
                        if sig == signal.SIGTERM:
                            time.sleep(1)
                    process.wait(timeout=5)
            except BaseException as cleanup_error:
                try:
                    sys.stderr.write(f"runner cleanup failed: {cleanup_error}\n")
                except BaseException:
                    pass
            if isinstance(error, subprocess.TimeoutExpired):
                raise RuntimeError(
                    f"runner timed out after {timeout:g} seconds; partial log: {raw_path}"
                ) from error
            raise
    return subprocess.CompletedProcess(command, returncode, raw_path.read_text(encoding="utf-8"))


def parse_run(output: str, *, target: str = "gfx942", mode: str = "tile") -> dict[str, object]:
    passed = list(PASS_RE.finditer(output))
    scores = list(SCORE_RE.finditer(output))
    benches = [
        {"size": int(m.group("size")), "median_us": float(m.group("median")),
         "p10_us": float(m.group("p10")), "p90_us": float(m.group("p90")),
         "gflops": float(m.group("gflops"))}
        for m in BENCH_RE.finditer(output)
    ]
    expected_kernel = "gemm_autoresearch_simt_v1" if mode == "simt" else "gemm_autoresearch_v1"
    expected_cases = Counter(["strided-tails", "nonquarter", "cancellation", "fractional-full"])
    if (Counter(m.group("case") for m in passed) != expected_cases
        or any(m.group("kernel") != expected_kernel or m.group("target") != target for m in passed)
        or len(scores) != 1 or [b["size"] for b in benches] != [256, 512, 1024]):
        raise RuntimeError("runner did not emit the complete matching correctness and benchmark record")
    errors = [float(m.group("error")) for m in passed]
    score = float(scores[0].group("score"))
    if (any(not math.isfinite(x) or x < 0 for x in errors)
        or not math.isfinite(score) or score <= 0
        or any(not all(math.isfinite(float(b[k])) and float(b[k]) > 0
                       for k in ["median_us", "p10_us", "p90_us", "gflops"])
               or not b["p10_us"] <= b["median_us"] <= b["p90_us"] for b in benches)):
        raise RuntimeError("runner emitted non-finite or invalid timing/correctness data")
    return {"correctness_shape": passed[0].group("shape"), "max_abs_error": max(errors),
            "correctness_cases": [{"name": m.group("case"), "shape": m.group("shape"),
                                    "max_abs_error": float(m.group("error"))} for m in passed],
            "benchmarks": benches, "geometric_mean_gflops": score}


def append_ledger(path: Path, record: dict[str, object], note: str) -> None:
    header = "timestamp_utc\tcandidate_sha256\tlabel\tstatus\tmedian_geomean_gflops\tnote\n"
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    fields = [
        str(record["timestamp_utc"]),
        str(record["candidate_sha256"]),
        str(record["label"]),
        str(record["status"]),
        f"{float(record['median_geometric_mean_gflops']):.2f}",
        note.replace("\t", " ").replace("\n", " "),
    ]
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write("\t".join(fields) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--target", choices=["gfx942", "gfx950"], default="gfx942")
    parser.add_argument("--mode", choices=["tile", "simt"], default="tile")
    parser.add_argument("--note", default="")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=EXAMPLE_DIR / "results" / "latest.json")
    parser.add_argument("--ledger", type=Path, default=EXAMPLE_DIR / "results.tsv")
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    if args.target == "gfx942" and args.mode != "tile":
        parser.error("the retained gfx942 runner supports tile only; use gfx950 for SIMT")
    runner = EXAMPLE_DIR / f"run-{args.target}.sh"
    candidate = CANDIDATE if args.mode == "tile" else EXAMPLE_DIR / "src" / "kernel_simt.rs"

    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        parser.error("set ROCR_VISIBLE_DEVICES to exactly one authorized GPU")

    target_dir = Path(
        os.environ.get(
            "FE2O3_AUTORESEARCH_TARGET_DIR",
            REPO_ROOT / "target" / "autoresearch-gemm-v1",
        )
    )
    artifact_dir = target_dir / "artifacts"
    temporary_dir = target_dir / "tmp"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.pop("HIP_VISIBLE_DEVICES", None)
    environment["FE2O3_AUTORESEARCH_GEMM_TARGET"] = args.target
    environment.update(
        {
            "FE2O3_BENCHMARK": "1",
            "FE2O3_ROOT_TARGET_DIR": str(target_dir),
            "FE2O3_OUTPUT_DIR": str(artifact_dir),
            "TMPDIR": str(temporary_dir),
        }
    )

    runs: list[dict[str, object]] = []
    raw_outputs: list[str] = []
    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    for replicate in range(args.replicates):
        raw_path = args.output.parent / f"{args.output.stem}.replicate-{replicate + 1}.log"
        completed = run_logged(
            ["bash", str(runner)] + ([args.mode] if args.target == "gfx950" else []),
            cwd=REPO_ROOT,
            env=environment,
            timeout=args.timeout_seconds,
            raw_path=raw_path,
        )
        raw_outputs.append(completed.stdout)
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            raise RuntimeError(f"replicate {replicate + 1} failed with status {completed.returncode}")
        run = parse_run(completed.stdout, target=args.target, mode=args.mode)
        runs.append(run)
        print(
            f"replicate={replicate + 1} correctness=pass "
            f"score_gflops={run['geometric_mean_gflops']:.2f}"
        )

    scores = [float(run["geometric_mean_gflops"]) for run in runs]
    record: dict[str, object] = {
        "schema": "fe2o3.gemm-autoresearch.mi300x.v1" if args.target == "gfx942" else "fe2o3.gemm-autoresearch.gfx950.v1",
        "timestamp_utc": timestamp,
        "label": args.label,
        "status": "pass",
        "host": socket.gethostname(),
        "target": args.target + ":xnack-",
        "mode": args.mode,
        "visible_device": visible,
        "timeout_seconds_per_replicate": args.timeout_seconds,
        "base_commit": command_output(["git", "rev-parse", "HEAD"]),
        "candidate_sha256": sha256(candidate),
        "evaluator_sha256": sha256(Path(__file__)),
        "reference_sha256": sha256(EXAMPLE_DIR / "src" / "reference.rs"),
        "replicates": runs,
        "median_geometric_mean_gflops": statistics.median(scores),
        "measurement_boundary": (
            "Device-event timing on one explicitly selected matching GPU. Re-run on an idle, clock-controlled "
            "host before making a performance or state-of-the-art claim."
        ),
        "raw_output_sha256": [
            hashlib.sha256(output.encode("utf-8")).hexdigest() for output in raw_outputs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_ledger(args.ledger, record, args.note)
    print(f"median_geometric_mean_gflops={statistics.median(scores):.2f}")
    print(f"evidence={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

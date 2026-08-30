#!/usr/bin/env python3
"""Validate and compare fused versus exact unfused GPT-OSS layer-tile records."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import analyze as common

FUSED_EXPORT = "gfx950_gpt_oss_120b_decode_megakernel_v1"
UNFUSED_EXPORTS = {
    "gpt_oss_unfused_router",
    "gpt_oss_unfused_attention",
    "gpt_oss_unfused_expert",
}
EXPECTED_PROCESSES = 5
EXPECTED_BLOCKS = 30
EXPECTED_SAMPLES_PER_BLOCK = 100


def trial_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    trial = record["trial"]
    return record["campaign_id"], trial["process"], trial["block"], trial["sample"]


def summarize(durations: dict[tuple[str, int, int, int], int], seed: int) -> dict[str, Any]:
    blocks: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for (campaign, process, block, _sample), duration in durations.items():
        blocks[(campaign, process, block)].append(float(duration))
    values = list(durations.values())
    median = float(statistics.median(values))
    low, high = common.hierarchical_bootstrap(blocks, 10_000, seed)
    processes = sorted({key[1] for key in durations})
    return {
        "samples": len(values),
        "processes": len(processes),
        "blocks": len(blocks),
        "median_ns": median,
        "p5_ns": common.percentile(values, 0.05),
        "p95_ns": common.percentile(values, 0.95),
        "mad_ns": float(statistics.median(abs(value - median) for value in values)),
        "median_bootstrap_ci95_ns": [low, high],
        "per_process_median_ns": {
            str(process): float(statistics.median(
                duration for key, duration in durations.items() if key[1] == process
            ))
            for process in processes
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    records = common.load([args.records])
    fused: dict[tuple[str, int, int, int], int] = {}
    parts: dict[tuple[str, int, int, int], dict[str, int]] = defaultdict(dict)
    hashes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        variant = record["implementation"]["variant"]
        export = record["artifact"]["kernel_export"]
        key = trial_key(record)
        duration = int(record["timer"]["duration_ns"])
        hashes[variant].add(record["artifact"]["hsaco_sha256"])
        if variant == "fused-optimized" and export == FUSED_EXPORT:
            if key in fused:
                raise ValueError(f"duplicate fused sample {key}")
            fused[key] = duration
        elif variant == "exact-unfused" and export in UNFUSED_EXPORTS:
            if export in parts[key]:
                raise ValueError(f"duplicate unfused stage {export} for {key}")
            parts[key][export] = duration
        else:
            raise ValueError(f"unexpected variant/export pair: {variant}/{export}")
    incomplete = {key: set(value) for key, value in parts.items() if set(value) != UNFUSED_EXPORTS}
    if incomplete:
        raise ValueError(f"incomplete unfused trials: {incomplete}")
    unfused = {key: sum(value.values()) for key, value in parts.items()}
    if set(fused) != set(unfused):
        raise ValueError("fused and unfused trial coordinates differ")
    expected = EXPECTED_PROCESSES * EXPECTED_BLOCKS * EXPECTED_SAMPLES_PER_BLOCK
    if len(fused) != expected:
        raise ValueError(f"expected {expected} paired trials, found {len(fused)}")
    fused_summary = summarize(fused, 950)
    unfused_summary = summarize(unfused, 951)
    fused_median = fused_summary["median_ns"]
    unfused_median = unfused_summary["median_ns"]
    result = {
        "schema": "fe2o3.gfx950.gpt-oss-layer-tile-comparison.v1",
        "claim_boundary": "batch-1 single-layer fixed-context layer tile; not a full-model claim",
        "protocol": {
            "processes_per_variant": EXPECTED_PROCESSES,
            "initial_warmups": 1000,
            "blocks": EXPECTED_BLOCKS,
            "samples_per_block": EXPECTED_SAMPLES_PER_BLOCK,
            "block_rewarm": 20,
            "order": "alternating AB/BA fresh processes",
            "timer": "ROCr HSA dispatch timestamps",
        },
        "fused": fused_summary,
        "exact_unfused_three_dispatch_sum": unfused_summary,
        "exact_unfused_over_fused": unfused_median / fused_median,
        "fused_faster": fused_median < unfused_median,
        "hsaco_sha256_by_variant": {key: sorted(value) for key, value in hashes.items()},
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

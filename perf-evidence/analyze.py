#!/usr/bin/env python3
"""Validate and summarize fe2o3 gfx950 dispatch-sample JSONL."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "fe2o3.gfx950.advanced-dispatch-sample.v1"
DECIMAL_FIELDS = (
    "start_tick",
    "end_tick",
    "duration_ticks",
    "frequency_hz",
    "duration_ns",
    "aql_packet_id",
)


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def canonical_decimal(value: Any, field: str, source: str) -> int:
    if not isinstance(value, str) or not value or not value.isascii() or not value.isdigit():
        raise ValueError(f"{source}: {field} is not an unsigned decimal string")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError(f"{source}: {field} is not canonical decimal")
    return parsed


def validate(record: dict[str, Any], source: str) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{source}: unsupported schema")
    for key in (
        "campaign_id",
        "record_id",
        "implementation",
        "artifact",
        "workload",
        "launch",
        "trial",
        "timer",
        "correctness",
        "environment",
    ):
        if key not in record:
            raise ValueError(f"{source}: missing {key}")
    correctness = record["correctness"]
    if not all(
        correctness.get(field) is True
        for field in (
            "passed",
            "preflight_dispatch_checked",
            "post_block_checked",
            "guard_canaries_checked",
        )
    ):
        raise ValueError(f"{source}: correctness interlock did not pass")
    timer = record["timer"]
    if timer.get("source") != "rocr-hsa-dispatch-timestamps":
        raise ValueError(f"{source}: unsupported timer")
    parsed = {field: canonical_decimal(timer.get(field), field, source) for field in DECIMAL_FIELDS}
    if parsed["frequency_hz"] == 0:
        raise ValueError(f"{source}: zero timestamp frequency")
    if parsed["end_tick"] < parsed["start_tick"]:
        raise ValueError(f"{source}: negative dispatch duration")
    ticks = parsed["end_tick"] - parsed["start_tick"]
    if ticks != parsed["duration_ticks"]:
        raise ValueError(f"{source}: duration_ticks disagrees with start/end")
    nanoseconds = ticks * 1_000_000_000 // parsed["frequency_hz"]
    if nanoseconds != parsed["duration_ns"]:
        raise ValueError(f"{source}: duration_ns disagrees with ticks/frequency")
    trial = record["trial"]
    for field in ("process", "block", "sample", "initial_warmups", "block_rewarm", "samples_per_block"):
        if not isinstance(trial.get(field), int) or trial[field] < 0:
            raise ValueError(f"{source}: invalid trial.{field}")
    if trial["samples_per_block"] == 0:
        raise ValueError(f"{source}: zero samples_per_block")


def load(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                source = f"{path}:{line_number}"
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}: record is not an object")
                validate(value, source)
                record_id = value["record_id"]
                if record_id in ids:
                    raise ValueError(f"{source}: duplicate record_id {record_id}")
                ids.add(record_id)
                records.append(value)
    if not records:
        raise ValueError("no records")
    return records


def operator_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["artifact"]["kernel_export"],
        record["workload"]["id"],
        record["workload"]["input_sha256"],
    )


def series_key(record: dict[str, Any]) -> tuple[str, ...]:
    return operator_key(record) + (
        json.dumps(record["launch"], sort_keys=True, separators=(",", ":")),
        record["implementation"]["id"],
        record["implementation"]["variant"],
        record["artifact"]["hsaco_sha256"],
    )


def hierarchy(record: dict[str, Any]) -> tuple[str, int, int]:
    return record["campaign_id"], record["trial"]["process"], record["trial"]["block"]


def hierarchical_bootstrap(
    groups: dict[tuple[Any, ...], list[float]],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    keys = sorted(groups)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled: list[float] = []
        for key in (rng.choice(keys) for _ in keys):
            block = groups[key]
            sampled.extend(rng.choice(block) for _ in block)
        estimates.append(float(statistics.median(sampled)))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(records: list[dict[str, Any]], repetitions: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[series_key(record)].append(record)
    result = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        durations = [int(value["timer"]["duration_ns"]) for value in values]
        blocks: dict[tuple[str, int, int], list[float]] = defaultdict(list)
        for value, duration in zip(values, durations):
            blocks[hierarchy(value)].append(float(duration))
        median = float(statistics.median(durations))
        low, high = hierarchical_bootstrap(blocks, repetitions, seed + index)
        result.append(
            {
                "kernel_export": key[0],
                "workload_id": key[1],
                "input_sha256": key[2],
                "launch": json.loads(key[3]),
                "implementation_id": key[4],
                "variant": key[5],
                "hsaco_sha256": key[6],
                "samples": len(durations),
                "processes": len({value["trial"]["process"] for value in values}),
                "blocks": len(blocks),
                "median_ns": median,
                "p5_ns": percentile(durations, 0.05),
                "p95_ns": percentile(durations, 0.95),
                "mad_ns": float(statistics.median(abs(value - median) for value in durations)),
                "median_bootstrap_ci95_ns": [low, high],
            }
        )
    return result


def paired_comparison(
    records: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_operator: dict[
        tuple[str, ...], dict[str, dict[tuple[str, int, int, int], int]]
    ] = defaultdict(lambda: defaultdict(dict))
    launches: dict[tuple[tuple[str, ...], str], set[str]] = defaultdict(set)
    for record in records:
        variant = record["implementation"]["variant"]
        if variant not in (baseline, candidate):
            continue
        key = operator_key(record)
        launches[(key, variant)].add(
            json.dumps(record["launch"], sort_keys=True, separators=(",", ":"))
        )
        pair = (
            record["campaign_id"],
            record["trial"]["process"],
            record["trial"]["block"],
            record["trial"]["sample"],
        )
        table = by_operator[key][variant]
        if pair in table:
            raise ValueError(f"duplicate pair key for {variant}: {pair}")
        table[pair] = int(record["timer"]["duration_ns"])
    output = []
    for index, (key, variants) in enumerate(sorted(by_operator.items())):
        if baseline not in variants or candidate not in variants:
            continue
        for variant in (baseline, candidate):
            if len(launches[(key, variant)]) != 1:
                raise ValueError(
                    f"{key}: {variant} has multiple launch configurations"
                )
        pairs = sorted(set(variants[baseline]) & set(variants[candidate]))
        if not pairs:
            continue
        ratios = [variants[baseline][pair] / variants[candidate][pair] for pair in pairs]
        contributions = [
            100.0 * (variants[baseline][pair] - variants[candidate][pair]) / variants[baseline][pair]
            for pair in pairs
        ]
        blocks: dict[tuple[str, int, int], list[float]] = defaultdict(list)
        for pair, ratio in zip(pairs, ratios):
            blocks[pair[:3]].append(ratio)
        low, high = hierarchical_bootstrap(blocks, repetitions, seed + 10_000 + index)
        median_speedup = float(statistics.median(ratios))
        median_reduction = float(statistics.median(contributions))
        output.append(
            {
                "kernel_export": key[0],
                "workload_id": key[1],
                "input_sha256": key[2],
                "baseline_launch": json.loads(next(iter(launches[(key, baseline)]))),
                "candidate_launch": json.loads(next(iter(launches[(key, candidate)]))),
                "baseline_variant": baseline,
                "candidate_variant": candidate,
                "paired_samples": len(pairs),
                "median_paired_speedup": median_speedup,
                "paired_speedup_bootstrap_ci95": [low, high],
                "median_latency_reduction_percent": median_reduction,
                "faster_at_ci95": low > 1.0,
                "slower_at_ci95": high < 1.0,
                "direction_indistinguishable_at_ci95": low <= 1.0 <= high,
            }
        )
    return output


def chain_contributions(
    summaries: list[dict[str, Any]], chain: list[str],
) -> list[dict[str, Any]]:
    table: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in summaries:
        table[
            (item["kernel_export"], item["workload_id"], item["input_sha256"])
        ][item["variant"]] = item
    result = []
    for key, variants in sorted(table.items()):
        if any(variant not in variants for variant in chain):
            continue
        for variant in chain:
            matching = [
                item for item in summaries
                if item["kernel_export"] == key[0]
                and item["workload_id"] == key[1]
                and item["input_sha256"] == key[2]
                and item["variant"] == variant
            ]
            if len(matching) != 1:
                raise ValueError(f"{key}: {variant} has multiple summarized series")
        baseline_ns = variants[chain[0]]["median_ns"]
        final_ns = variants[chain[-1]]["median_ns"]
        total_log_speedup = math.log(baseline_ns / final_ns)
        steps = []
        for previous, current in zip(chain, chain[1:]):
            previous_item = variants[previous]
            current_item = variants[current]
            previous_ns = previous_item["median_ns"]
            current_ns = current_item["median_ns"]
            saved = previous_ns - current_ns
            steps.append(
                {
                    "from": previous,
                    "to": current,
                    "from_hsaco_sha256": previous_item["hsaco_sha256"],
                    "to_hsaco_sha256": current_item["hsaco_sha256"],
                    "from_launch": previous_item["launch"],
                    "to_launch": current_item["launch"],
                    "artifacts_identical": (
                        previous_item["hsaco_sha256"] == current_item["hsaco_sha256"]
                    ),
                    "incremental_speedup": previous_ns / current_ns,
                    "median_ns_saved": saved,
                    "median_latency_reduction_percent": 100.0 * saved / previous_ns,
                    "cumulative_speedup": baseline_ns / current_ns,
                    "cumulative_latency_reduction_percent": (
                        100.0 * (baseline_ns - current_ns) / baseline_ns
                    ),
                    "log_speedup_contribution_percent": (
                        100.0 * math.log(previous_ns / current_ns)
                        / total_log_speedup
                        if total_log_speedup != 0 else None
                    ),
                }
            )
        result.append(
            {
                "kernel_export": key[0],
                "workload_id": key[1],
                "input_sha256": key[2],
                "baseline_variant": chain[0],
                "final_variant": chain[-1],
                "baseline_median_ns": baseline_ns,
                "final_median_ns": final_ns,
                "total_speedup": baseline_ns / final_ns,
                "total_latency_reduction_percent": (
                    100.0 * (baseline_ns - final_ns) / baseline_ns
                ),
                "steps": steps,
            }
        )
    return result


def self_test() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert math.isclose(percentile([1, 2, 3, 4, 5], 0.05), 1.2)
    low, high = hierarchical_bootstrap({(0, 0): [10.0] * 4, (0, 1): [10.0] * 4}, 100, 950)
    assert (low, high) == (10.0, 10.0)
    summaries = [
        {
            "kernel_export": "kernel",
            "workload_id": "shape",
            "input_sha256": "a" * 64,
            "variant": "baseline",
            "launch": {"workgroup": [64, 1, 1]},
            "hsaco_sha256": "b" * 64,
            "median_ns": 100.0,
        },
        {
            "kernel_export": "kernel",
            "workload_id": "shape",
            "input_sha256": "a" * 64,
            "variant": "optimized",
            "launch": {"workgroup": [128, 1, 1]},
            "hsaco_sha256": "c" * 64,
            "median_ns": 80.0,
        },
    ]
    chain = chain_contributions(summaries, ["baseline", "optimized"])
    assert len(chain) == 1
    assert chain[0]["total_speedup"] == 1.25
    assert chain[0]["steps"][0]["median_ns_saved"] == 20.0
    assert chain[0]["steps"][0]["median_latency_reduction_percent"] == 20.0
    assert chain[0]["steps"][0]["artifacts_identical"] is False
    assert chain[0]["steps"][0]["to_launch"]["workgroup"] == [128, 1, 1]
    records = []
    for variant, workgroup, durations in (
        ("baseline", 64, (100, 102)),
        ("optimized", 128, (80, 82)),
    ):
        for sample, duration in enumerate(durations):
            records.append(
                {
                    "campaign_id": "campaign",
                    "implementation": {"id": "rust", "variant": variant},
                    "artifact": {
                        "kernel_export": "kernel",
                        "hsaco_sha256": ("b" if variant == "baseline" else "c") * 64,
                    },
                    "workload": {
                        "id": "shape",
                        "input_sha256": "a" * 64,
                    },
                    "launch": {"workgroup": [workgroup, 1, 1]},
                    "trial": {
                        "process": 0,
                        "block": 0,
                        "sample": sample,
                    },
                    "timer": {"duration_ns": str(duration)},
                }
            )
    comparisons = paired_comparison(
        records, "baseline", "optimized", 100, 950
    )
    assert len(comparisons) == 1
    assert comparisons[0]["paired_samples"] == 2
    assert comparisons[0]["baseline_launch"]["workgroup"] == [64, 1, 1]
    assert comparisons[0]["candidate_launch"]["workgroup"] == [128, 1, 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="*", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=950)
    parser.add_argument("--baseline-variant")
    parser.add_argument("--candidate-variant")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="BASELINE:CANDIDATE",
        help="repeatable independent paired comparison",
    )
    parser.add_argument("--chain", help="comma-separated ordered optimization variants")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("PASS analyze.py self-test")
        return 0
    if not args.jsonl:
        parser.error("at least one JSONL path is required")
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    if bool(args.baseline_variant) != bool(args.candidate_variant):
        parser.error("baseline and candidate variants must be supplied together")
    comparisons_requested: list[tuple[str, str]] = []
    if args.baseline_variant:
        comparisons_requested.append((args.baseline_variant, args.candidate_variant))
    for value in args.compare:
        parts = value.split(":")
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            parser.error("--compare must be BASELINE:CANDIDATE with distinct variants")
        comparisons_requested.append((parts[0], parts[1]))
    if len(set(comparisons_requested)) != len(comparisons_requested):
        parser.error("paired comparisons must be unique")
    records = load(args.jsonl)
    summaries = summarize(records, args.bootstrap_repetitions, args.seed)
    result: dict[str, Any] = {
        "schema": "fe2o3.gfx950.advanced-dispatch-summary.v1",
        "source_files": [str(path) for path in args.jsonl],
        "bootstrap": {
            "method": "hierarchical process/block resampling with within-block resampling",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "series": summaries,
    }
    if comparisons_requested:
        comparisons = []
        for baseline, candidate in comparisons_requested:
            values = paired_comparison(
                records,
                baseline,
                candidate,
                args.bootstrap_repetitions,
                args.seed,
            )
            if not values:
                raise ValueError(
                    f"requested comparison has no paired samples: {baseline}:{candidate}"
                )
            comparisons.extend(values)
        result["paired_comparisons"] = comparisons
    if args.chain:
        chain = [value for value in args.chain.split(",") if value]
        if len(chain) < 2:
            parser.error("--chain needs at least two comma-separated variants")
        if len(set(chain)) != len(chain):
            parser.error("--chain variants must be unique")
        contributions = chain_contributions(summaries, chain)
        if not contributions:
            raise ValueError("optimization chain has no complete operator/workload series")
        comparisons = [
            comparison
            for previous, current in zip(chain, chain[1:])
            for comparison in paired_comparison(
                records,
                previous,
                current,
                args.bootstrap_repetitions,
                args.seed,
            )
        ]
        comparison_keys = {
            (
                item["kernel_export"],
                item["workload_id"],
                item["input_sha256"],
                item["baseline_variant"],
                item["candidate_variant"],
            )
            for item in comparisons
        }
        for item in contributions:
            for previous, current in zip(chain, chain[1:]):
                key = (
                    item["kernel_export"],
                    item["workload_id"],
                    item["input_sha256"],
                    previous,
                    current,
                )
                if key not in comparison_keys:
                    raise ValueError(f"optimization chain has no paired samples for {key}")
        result["optimization_chain_contributions"] = contributions
        result["optimization_chain_paired_comparisons"] = comparisons
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

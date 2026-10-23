#!/usr/bin/env python3
"""Actual default compilation, current V18 execution and same-request CPU references.

This versioned observation does not relabel pending legacy simulation bundles,
prove LLVM/hardware equivalence or grant publication/launch authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
gate = load("tutorial_current_cargo_gate", HERE / "qualify-tutorial-default-cargo.py")
simulation = load("tutorial_current_simulation", HERE / "tutorial_simulation_v92.py")
require = simulation.require


def expectation(manifest, case):
    require(len(case["references"]) == 1, "negative must name one exact source case")
    kind, lesson, ordinal, index = case["references"][0]
    require(kind == "source-driver-case", "negative source case required")
    matches = [row for row in manifest["curriculum"]["lessons"] if row["lessonId"] == lesson]
    require(len(matches) == 1, "negative lesson identity")
    value = matches[0]["codeTabs"][ordinal]["sourceItem"]["cases"][index]["expectation"]
    require(value["kind"] == "rejected" and value["outputArtifact"] == "absent",
            "registered exact negative expectation required")
    return value


def check_no_artifacts(directory):
    """Inspect only the new, isolated invocation output; never infer from exit status."""
    pending, count, found = [(directory, 0)], 0, []
    while pending:
        parent, depth = pending.pop()
        require(depth <= 32, "negative artifact directory depth")
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                require(count <= 65536, "negative artifact census bound")
                mode = entry.stat(follow_symlinks=False).st_mode
                require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "negative output special entry")
                if stat.S_ISDIR(mode):
                    pending.append((Path(entry.path), depth + 1))
                elif entry.name.endswith((".hsaco", ".fe2sim", ".kir", ".kir-v18")):
                    found.append(str(Path(entry.path).relative_to(directory)))
    require(not found, "negative invocation produced a compiler/simulation artifact")
    return {"entriesInspected": count, "artifacts": found}


def negative_result(manifest, case, compilation):
    expected = expectation(manifest, case)
    outcome = case.get("execution", {})
    require(case.get("sourceBefore") == case.get("sourceAfter") and case.get("sourceBefore"),
            "negative source changed or was not captured")
    require(outcome.get("status") == "cargo-failed" and type(outcome.get("exitCode")) is int
            and outcome["exitCode"] != 0 and outcome.get("logComplete") is True
            and outcome.get("directChildReaped") is True, "negative did not fail normally")
    raw, pin = simulation.read_file(compilation / case["log"])
    require(pin["sha256"] == outcome["logSha256"] and pin["bytes"] == outcome["logBytes"],
            "negative transcript changed")
    require(expected["diagnosticContains"].encode() in raw, "exact negative diagnostic absent")
    require(b"[cargo-fe2o3] production-census-v91" not in raw
            and "productionCensus" not in case, "negative emitted a successful census")
    capture = compilation / case["capturedGraphsV92"]
    require(not any(capture.iterdir()), "negative retained a forwarded production graph")
    return {"expected": expected, "log": pin, "absence": check_no_artifacts(
        compilation / case["negativeArtifactDirectoryV92"]), "passed": True}


def site_inventory(root, checkout, commit, manifest, directory, node, runner, environment, validator):
    log = directory / "inventory.json"
    outcome = runner([str(node), str(root / "scripts/export-tutorial-site-v92.mjs"),
                      str(checkout), commit, str(manifest), str(directory / "vite-cache")],
                     root, environment, log, 60, simulation.MAX_DOCUMENT)
    require(simulation.command_passed(outcome), "actual tutorial-source export failed")
    raw, pin = simulation.read_file(log)
    value = simulation.strict_json(raw)
    validator.validate_site_inventory(validator.load_manifest(manifest)["curriculum"], value)
    require(value["site"]["commit"] == commit, "site export changed commit")
    return {"site": value["site"], "inventory": pin, "execution": outcome}


def run(args, *, runner=gate.run_command, environment=None):
    environment = dict(os.environ if environment is None else environment)
    root, manifest_path = args.repo_root.resolve(strict=True), args.manifest.resolve(strict=True)
    validator, _, manifest, manifest_sha, roster = gate.load_inputs(root, manifest_path)
    require(len(roster) == 109 and sum(bool(row[2]) for row in roster.values()) == 3,
            "complete registered 109-case/3-negative corpus required")
    names = sorted({name for fixture, _, negative in roster.values() if not negative
                    for name in fixture["compilerInput"]["kernelSymbols"]})
    require(len(names) == 45, "complete current positive kernel corpus required")
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = {"schema": "fe2o3-tutorial-current-qualification-v92", "manifestSha256": manifest_sha,
              "registeredInvocations": 109, "positiveInvocations": 106, "negativeInvocations": 3,
              "complete": False, "qualified": False, "currentCompilerSimulationPassed": False,
              "legacyBundleContractsReinterpreted": False, **simulation.NO_AUTHORITY, "cases": []}
    gate.write_report(output, report)
    tools = [args.reference.resolve(strict=True), args.simulator.resolve(strict=True),
             args.node.resolve(strict=True), args.cargo_fe2o3.resolve(strict=True)]
    tool_pins = [simulation.pin_executable(path) for path in tools]
    report["tools"] = [dict(path=str(path), **pin) for path, pin in zip(tools, tool_pins)]
    for label, path, commit in [("pinnedSite", args.site_pinned, manifest["curriculum"]["site"]["commit"]),
                                 ("currentSite", args.site_current, args.site_current_commit)]:
        directory = output / label
        directory.mkdir(mode=0o700)
        report[label] = site_inventory(root, path.resolve(strict=True), commit, manifest_path,
                                       directory, tools[2], runner, environment, validator)
        gate.write_report(output, report)
    require(report["pinnedSite"]["site"] == manifest["curriculum"]["site"], "pinned site tree mismatch")
    listed = runner([str(tools[0]), "list"], root, environment, output / "reference-corpus.json",
                    120, simulation.MAX_DOCUMENT)
    require(simulation.command_passed(listed), "reference corpus executable failed")
    inventory = simulation.strict_json(simulation.read_file(output / "reference-corpus.json")[0])
    require(inventory == {"schema": "fe2o3-tutorial-reference-corpus-v92", "kernels": names,
                          "authority": False}, "reference corpus differs from actual manifest")
    compilation = output / "compilation"
    census = gate.run_census(root, manifest_path, tools[3], args.target_dir.resolve(), compilation,
                            ("gfx942", "gfx950"), args.compile_timeout, gate.MAX_LOG_BYTES,
                            runner=runner, environment=environment, production=True,
                            capture_graphs=True, isolate_negative_outputs=True)
    require(census["coversAllRegisteredInvocations"] and len(census["cases"]) == 109,
            "actual compilation census dropped invocations")
    for case, (key, (fixture, references, negative)) in zip(
            census["cases"], sorted(roster.items()), strict=True):
        require(case["id"] == key and case["fixture"] == fixture
                and case["references"] == [list(reference) for reference in references]
                and case["expectedNegative"] is negative,
                "actual compilation census changed the registered invocation")
    for ordinal, case in enumerate(census["cases"]):
        result = {"id": case["id"], "fixture": case["fixture"], "references": case["references"],
                  "expectedNegative": case["expectedNegative"], "passed": False}
        report["cases"].append(result)
        try:
            if case["expectedNegative"]:
                result["negative"] = negative_result(manifest, case, compilation)
            else:
                require(case["status"] == "production-compile-census-pass", "production compilation refused")
                directory = output / f"simulation-{ordinal:04d}"
                directory.mkdir(mode=0o700)
                result["kernels"] = []
                for index, kernel in enumerate(case["fixture"]["compilerInput"]["kernelSymbols"]):
                    request = directory / f"{index:04d}.request.json"
                    generated = runner([str(tools[0]), "generate", kernel], root, environment, request,
                                       120, simulation.MAX_DOCUMENT)
                    require(simulation.command_passed(generated), "current request generation failed")
                    require(simulation.strict_json(simulation.read_file(request)[0])["kernel"] == kernel,
                            "generated request kernel substitution")
                    observation = simulation.run_case(case, compilation / case["capturedGraphsV92"],
                        request, tools[0], tools[1], directory / f"{index:04d}", runner,
                        environment=environment, cwd=root, timeout=args.simulation_timeout)
                    result["kernels"].append({"generation": generated, "observation": observation})
            result["passed"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["error"] = str(error)[:4096]
        gate.write_report(output, report)
    require([simulation.pin_executable(path) for path in tools] == tool_pins, "qualification tools changed")
    require(validator.load_manifest(manifest_path, with_sha256=True)[1] == manifest_sha, "manifest changed")
    report["complete"] = True
    report["currentCompilerSimulationPassed"] = all(case["passed"] for case in report["cases"])
    gate.write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["repo-root", "manifest", "cargo-fe2o3", "target-dir", "output", "reference",
                 "simulator", "node", "site-pinned", "site-current"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--site-current-commit", required=True)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--simulation-timeout", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.compile_timeout <= 3600 or not 1 <= args.simulation_timeout <= 600:
        parser.error("finite command deadline exceeded")
    try:
        with gate.cli_interrupt_handlers():
            report = run(args)
        return 0 if report["currentCompilerSimulationPassed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"current tutorial qualification refused: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

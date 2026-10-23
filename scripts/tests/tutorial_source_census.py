#!/usr/bin/env python3
"""Real-source joins with protocol fixtures, never fabricated execution evidence.

The row-affine file, Cargo/lock closure, runtime bytes, and SIMT binding are real.
The JSON transport is deliberately constructed for parser/tamper coverage; the
ignored Rust corpus test exercises the actual producer and callback lifecycle.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceCensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = load("census_parent", ROOT / "scripts/validate-tutorial-kernel-manifest.py")
        cls.adapter = load("census_join", ROOT / "scripts/tutorial_source_census.py")
        cls.fixture_tests = load("census_real_fixture", ROOT / "scripts/tests/tutorial_fixture_bindings.py")
        cls.fixture_tests.RowSourceBindingTests.setUpClass()

    def component(self):
        builder = self.fixture_tests.RowSourceBindingTests()
        document, runtime, tab, kernel, _ = builder.component()
        raw_manifest = json.dumps(document, sort_keys=True).encode("utf-8")
        manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
        invocation = next(iter(self.adapter.registered_invocations(document).values()))[0]
        snapshot = self.adapter.input_snapshot(self.parent, ROOT, document, manifest_sha, invocation)
        path = ROOT / tab["sourcePath"]
        raw = path.read_bytes()
        self.assertEqual(len(raw), 2668)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "07adc0c50f24e51cb3d6c6bcb6cc1c8c6ff2a772c45ee2153435601eca2f39df")
        symbol = "row_affine_sum_u32_v1"
        self.assertEqual(raw[885:885 + len(symbol)], symbol.encode("utf-8"))
        cwd = str((ROOT / invocation["compilerInput"]["packageManifest"]).parent)
        args = ["/test-only/rustc", "--crate-name", "fe2o3_workgroup_sync_v1",
                "--crate-type", "lib", "src/lib.rs", "--target", "amdgcn-amd-amdhsa",
                "-Ctarget-cpu=gfx942", "--cfg", 'feature="row-affine-u32-kernel"']

        def span(start, end):
            origin = {"file": 0, "coordinates": {"original_start": start, "original_end": end,
                                                 "normalized_start": start, "normalized_end": end}}
            return {"status": "available", "value": {"expansion": origin, "callSite": copy.deepcopy(origin),
                    "expansionChainSha256": "9" * 64, "expansionDepth": 0}}

        census = {"schema": "fe2o3-diagnostic-source-census-v1", "diagnosticOnly": True,
                  "qualified": False, "authenticatesCompilerExecution": False,
                  "extractionSucceeded": False, "arguments": args, "workingDirectory": cwd,
                  "extractionMode": {"kind": "fixed-checked-output", "policy": 4}, "runId": "a" * 64,
                  "selection": {"status": "available", "value": {"target": "gfx942", "files": [{
                      "identity": "1" * 64, "displayPath": str(path), "compiledSourceHash": "test-protocol-fixture",
                      "originalSha256": hashlib.sha256(raw).hexdigest(), "originalBytes": len(raw), "normalizedBytes": len(raw),
                  }], "functions": [{
                      "functionIdentity": "2" * 64, "definitionIdentity": "3" * 64,
                      "monomorphizationIdentity": "4" * 64, "role": "kernel-entry", "exportName": symbol,
                      "logicalName": symbol, "definition": span(878, len(raw) - 1), "identifier": span(885, 885 + len(symbol)),
                  }]}}}
        record = {"schema": "fe2o3-tutorial-source-census-observation-v1", "diagnosticOnly": True,
                  "qualified": False, "authenticatesCompilerExecution": False, "runId": "a" * 64,
                  "arguments": copy.deepcopy(args), "workingDirectory": cwd,
                  "before": snapshot, "after": copy.deepcopy(snapshot), "census": census,
                  "censusSha256": hashlib.sha256(json.dumps(census, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()}
        item = tab["sourceItem"]
        case = item["cases"][0]
        corpus = {"schema": "fe2o3-ordinary-source-policy4-source-driver-corpus-v1",
                  "manifest_sha256": manifest_sha, "configurations": 1,
                  "strict_negative_drivers_unchanged": 0, "all_checked_output_passed": False,
                  "default_pipeline_activated": False, "grants_artifact_or_launch_authority": False,
                  "cases": [{"source": {
                      "selection": kernel["selections"][0], "source_item_contract_sha256": item["contractSha256"],
                      "original_driver": item["driver"], "original_driver_test": case["testFunction"],
                      "original_expectation": case["expectation"], "fixture": invocation,
                  }, "result": {"fixture": invocation, "source_census": record, "source_census_error": None,
                                "status": "blocked", "observation": None, "refusal": {"detail": "protocol fixture only"},
                                "compiler_artifacts": []}}]}
        self.parent.validate_site_inventory(document["curriculum"], runtime)
        projection = self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)
        return document, runtime, manifest_sha, corpus, projection

    def join(self, data):
        document, runtime, manifest_sha, corpus, projection = data
        return self.adapter.join(self.parent, ROOT, document, manifest_sha, runtime, projection, [corpus])

    @staticmethod
    def record(data):
        return data[3]["cases"][0]["result"]["source_census"]

    @staticmethod
    def refresh_transport(record):
        record["censusSha256"] = hashlib.sha256(json.dumps(
            record["census"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    def test_exact_source_span_joins_real_runtime_and_existing_simt_variant(self):
        data = self.component()
        before = copy.deepcopy(data)
        joined = self.join(data)
        self.assertEqual(data, before)
        self.assertFalse(joined["qualified"])
        self.assertFalse(joined["authenticatesCompilerExecution"])
        self.assertFalse(joined["policy"]["productionMixedCompilation"])
        self.assertEqual(len(joined["joins"]), 1)
        row = joined["joins"][0]
        self.assertEqual(row["source"]["functionUtf8Offset"], 885)
        self.assertEqual(row["runtimeOccurrences"][0]["functionUtf8Offset"], 885)
        self.assertEqual(row["registeredVariantBindings"], [{
            "kernelId": data[4]["kernelIdentities"][0]["kernelId"], "kind": "simt", "status": "source-bound"}])
        self.assertEqual({row["variant"] for row in joined["unresolved"]}, {"tile", "mixed"})
        self.assertEqual(data[4]["sourceBoundPairCount"], 0)

    def test_stale_lock_default_features_closure_and_physical_source_reject(self):
        for mutation in ("lock", "defaults", "closure", "source", "after", "manifest"):
            data = self.component()
            record = self.record(data)
            if mutation == "lock":
                record["before"]["fixture"]["compilerInput"]["cargoLockSha256"] = "0" * 64
            elif mutation == "defaults":
                record["before"]["fixture"]["compilerInput"]["defaultFeatures"] = True
            elif mutation == "closure":
                record["before"]["fixture"]["compilerInput"]["sourceClosureSha256"] = "0" * 64
            elif mutation == "source":
                record["census"]["selection"]["value"]["files"][0]["originalSha256"] = "0" * 64
            elif mutation == "after":
                record["after"]["sources"][0]["sha256"] = "0" * 64
            else:
                data[3]["manifest_sha256"] = "0" * 64
            self.refresh_transport(record)
            with self.subTest(mutation=mutation), self.assertRaises((self.adapter.CensusError, SystemExit)):
                self.join(data)

    def test_replayed_run_arguments_mode_and_cfg_reject(self):
        for mutation in ("run", "arguments", "cwd", "mode", "features", "target", "response-file"):
            data = self.component()
            record = self.record(data)
            if mutation == "run":
                record["census"]["runId"] = "b" * 64
            elif mutation == "arguments":
                record["census"]["arguments"].append("--cfg=unreviewed")
            elif mutation == "cwd":
                record["workingDirectory"] = str(ROOT)
            elif mutation == "mode":
                record["census"]["extractionMode"]["policy"] = 10
            else:
                args = record["arguments"]
                args.extend(["--cfg", 'feature="lds-kernel"'] if mutation == "features" else
                            ["-Ctarget-cpu=gfx950"] if mutation == "target" else ["@hidden.rsp"])
                record["census"]["arguments"] = copy.deepcopy(args)
            self.refresh_transport(record)
            with self.subTest(mutation=mutation), self.assertRaises(self.adapter.CensusError):
                self.join(data)

    def test_duplicate_or_wrong_root_and_tampered_coordinates_reject(self):
        for mutation in ("duplicate", "role", "symbol", "offset", "normalization", "callsite", "definition"):
            data = self.component()
            selected = self.record(data)["census"]["selection"]["value"]
            function = selected["functions"][0]
            if mutation == "duplicate":
                selected["functions"].append(copy.deepcopy(function))
            elif mutation == "role":
                function["role"] = "internal-helper"
            elif mutation == "symbol":
                function["exportName"] = "another_root"
            else:
                span = function["identifier"]["value"]
                if mutation == "offset":
                    for origin in ("expansion", "callSite"):
                        span[origin]["coordinates"]["original_start"] += 1
                elif mutation == "normalization":
                    for origin in ("expansion", "callSite"):
                        span[origin]["coordinates"]["normalized_start"] += 1
                elif mutation == "callsite":
                    span["callSite"]["coordinates"]["original_start"] += 1
                else:
                    function["definition"]["value"]["expansion"]["coordinates"]["original_end"] = 880
            self.refresh_transport(self.record(data))
            with self.subTest(mutation=mutation), self.assertRaises(self.adapter.CensusError):
                self.join(data)

    def test_unavailable_macro_body_map_does_not_guess_from_names(self):
        data = self.component()
        function = self.record(data)["census"]["selection"]["value"]["functions"][0]
        function["identifier"]["value"]["expansionDepth"] = 1
        self.refresh_transport(self.record(data))
        result = self.join(data)
        self.assertEqual(result["joins"], [])
        self.assertIn("generated-body map", result["unresolved"][0]["reason"])

    def test_missing_census_and_missing_roster_are_distinct(self):
        data = self.component()
        data[3]["cases"][0]["result"]["source_census"] = None
        self.assertIn("unavailable", self.join(data)["unresolved"][0]["reason"])
        data[3]["cases"].clear()
        with self.assertRaisesRegex(self.adapter.CensusError, "complete invocation roster"):
            self.join(data)

    def test_same_nonce_cannot_be_reused_for_another_invocation(self):
        data = self.component()
        record = self.record(data)
        fixture = data[3]["cases"][0]["result"]["fixture"]
        with self.assertRaisesRegex(self.adapter.CensusError, "replayed run ID"):
            self.adapter.check_record(ROOT, fixture, record["before"], record, {record["runId"]})

    def test_census_digest_detects_changes_to_uninterpreted_compiler_metadata(self):
        data = self.component()
        self.record(data)["census"]["selection"]["value"]["files"][0]["compiledSourceHash"] = "changed"
        with self.assertRaisesRegex(self.adapter.CensusError, "changed after runner retention"):
            self.join(data)

    def test_aggregate_derived_record_bound_is_exact_and_fail_closed(self):
        document, runtime, sha, corpus, projection = self.component()
        # One join + source + runtime occurrence + explicit SIMT binding, and
        # two still-pending variants. The byte limit is a separate outer gate.
        result = self.adapter.join(self.parent, ROOT, document, sha, runtime, projection, [corpus], max_records=6)
        self.assertEqual(len(result["joins"]), 1)
        with self.assertRaisesRegex(self.adapter.CensusError, "derived join record bound"):
            self.adapter.join(self.parent, ROOT, document, sha, runtime, projection, [corpus], max_records=5)

    def test_cli_consumes_same_run_census_in_the_existing_kernel_pair_projection(self):
        data = self.component()
        document, runtime, _, corpus, _ = data
        original = self.fixture_tests.RowSourceBindingTests.original
        fixture_id = "gfx942-tiled-gemm"
        document["compilerFixtures"] = [copy.deepcopy(next(row for row in original["compilerFixtures"] if row["fixtureId"] == fixture_id))]
        entry = copy.deepcopy(next(row for row in original["entries"] if row["lessonId"] == "gemm-tiling"))
        entry["compilerFixtureIds"] = [fixture_id]
        document["entries"] = [entry]
        # This reduced row component is linked by its real source-driver item;
        # the retained legacy entry belongs to the separate full corpus.
        reduced = document["curriculum"]["lessons"][0]
        self.assertIsNotNone(reduced["sourceBindingGap"])
        self.assertEqual(len(reduced["codeTabs"]), 1)
        self.assertIsNotNone(reduced["codeTabs"][0]["sourceItem"])
        reduced["sourceEntryIds"] = []
        reduced["sourceBindingGap"] = None
        self.assertIsNotNone(next(row for row in original["curriculum"]["lessons"]
                                 if row["lessonId"] == reduced["lessonId"])["sourceBindingGap"])
        suites = []
        for suite in original["qualification"]["suites"]:
            coverage = [row for row in suite["coverage"] if row["lessonId"] == "gemm-tiling" and fixture_id in row["fixtureIds"]]
            if coverage:
                suite = copy.deepcopy(suite)
                suite["coverage"] = [{"lessonId": "gemm-tiling", "fixtureIds": [fixture_id]}]
                suites.append(suite)
        document["qualification"]["suites"] = suites
        # Reuse the checked current-source projection, not the historical site tab.
        current = self.fixture_tests.WholeFileFixtureBindingTests
        current.setUpClass()
        current_document, current_runtime = current().document()
        lesson = next(row for row in current_document["curriculum"]["lessons"]
                      if row["lessonId"] == "gemm-tiling")
        document["curriculum"]["lessons"].append(lesson)
        runtime["lessons"].append(next(row for row in current_runtime["lessons"]
                                      if row["id"] == "gemm-tiling"))
        inventory = document["kernelInventory"]
        inventory["kernels"].append(copy.deepcopy(next(row for row in original["kernelInventory"]["kernels"]
            if row["kernelId"] == "fixture:gfx942-tiled-gemm:tiled_gemm_general_v1")))
        inventory["displayItems"].extend(copy.deepcopy([row for row in current_document["kernelInventory"]["displayItems"]
            if row["lessonId"] == "gemm-tiling" and row["tabOrdinal"] == 0]))
        with tempfile.TemporaryDirectory(prefix="fe2o3-census-cli-join-") as directory:
            directory = Path(directory)
            manifest_path, runtime_path, corpus_path = [directory / name for name in ("manifest.json", "runtime.json", "corpus.json")]
            raw = json.dumps(document).encode()
            manifest_path.write_bytes(raw)
            runtime_path.write_text(json.dumps(runtime))
            sha = hashlib.sha256(raw).hexdigest()
            corpus["manifest_sha256"] = sha
            record = self.record(data)
            record["before"]["manifestSha256"] = sha
            record["after"]["manifestSha256"] = sha
            corpus_path.write_text(json.dumps(corpus))
            command = [sys.executable, "-I", "-B", str(ROOT / "scripts/validate-tutorial-kernel-manifest.py"),
                       "--repo-root", str(ROOT), "--manifest", str(manifest_path), "--site-inventory", str(runtime_path),
                       "--emit-kernel-pairs", "--source-driver-report", str(corpus_path)]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(len(result["sourceCensusObservations"]["joins"]), 1)
            self.assertFalse(result["qualified"])
            self.assertEqual(result["qualifiedPairCount"], 0)
            self.assertEqual(result["sourceBoundPairCount"], 0)
            record["census"]["runId"] = "b" * 64
            self.refresh_transport(record)
            corpus_path.write_text(json.dumps(corpus))
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout, "")
            self.assertIn("exact run/arguments/P4 mode", completed.stderr)

    def test_cli_snapshots_registered_physical_inputs_and_rejects_caller_substitution(self):
        manifest, manifest_sha = self.parent.load_manifest(ROOT / "config/tutorial-kernel-manifest-v1.json", with_sha256=True)
        invocations = self.adapter.registered_invocations(manifest)
        fixture = invocations["source-driver-cpu-semantic-simulation-tab6-case0"][0]
        with tempfile.TemporaryDirectory(prefix="fe2o3-source-census-") as directory:
            request = Path(directory) / "request.json"
            command = [sys.executable, "-I", "-B", str(ROOT / "scripts/validate-tutorial-kernel-manifest.py"),
                       "--repo-root", str(ROOT), "--emit-source-input-snapshot", str(request)]
            request.write_text(json.dumps(fixture))
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            actual = json.loads(completed.stdout)
            expected = self.adapter.input_snapshot(self.parent, ROOT, manifest, manifest_sha, fixture)
            self.assertEqual(actual, expected)
            for field, value in (("cargoLockSha256", "0" * 64), ("features", ["lds-kernel"]),
                                 ("sourceClosureSha256", "0" * 64), ("defaultFeatures", True)):
                changed = copy.deepcopy(fixture)
                changed["compilerInput"][field] = value
                request.write_text(json.dumps(changed))
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                self.assertNotEqual(completed.returncode, 0, field)
                self.assertIn("registered source contract", completed.stderr)


if __name__ == "__main__":
    unittest.main()

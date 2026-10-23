#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("simulation_v92", ROOT / "scripts/tutorial_simulation_v92.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
spec = importlib.util.spec_from_file_location("qualification_v92", ROOT / "scripts/qualify-tutorial-current-simulation-v92.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture():
    request = {"schema": "fe2o3-simulation-request-v1", "kernel": "fill", "grid": [1, 1, 1],
               "workgroup": [1, 1, 1], "arguments": [{"kind": "buffer", "element": "u32",
               "access": "read_write", "alignment": 4, "bytes": "0x00000000"},
               {"kind": "scalar", "type": "u32", "bits": "0x00000025"}]}
    raw = json.dumps(request).encode()
    reference = {"schema": "fe2o3-tutorial-reference-result-v92", "kernel": "fill",
                 "request_sha256": list(hashlib.sha256(raw).digest()), "authority": False,
                 "outputs": [{"argument": 0, "element": "u32", "bytes": "0x25000000",
                 "absolute_tolerance": 0, "relative_tolerance": 0}]}
    graph = {"sha256": "a" * 64, "canonical_bytes": 42}
    result = {"schema": "fe2o3-simulation-result-v1", "status": "ok", "authority": "observation_only",
              "simulated": True, "hardware_observed": False, "hardware_validation": False,
              "performance_prediction": False, "kir": graph, "shared_buffers": [],
              "target_profile": {"identity": "amdgpu_gfx942_little_endian_v2", "index_bits": 64,
                                 "max_workgroup_invocations": 1024},
              "conflict_assessment": {"status": "no_conflicts_observed"},
              "schedule": {"coverage": {"complete": True}}, "arguments": [
              {"kind": "buffer", "value": {"element": "u32", "access": "read_write", "alignment": 4,
                                            "bytes": "0x25000000", "initialized": "0x0f"}},
              request["arguments"][1]]}
    return raw, reference, result, graph


class CurrentSimulation(unittest.TestCase):
    def test_actual_command_binds_cpu_to_existing_complete_production_target(self):
        import re
        source = (ROOT / "crates/fe2o3-amd-target/src/lib.rs").read_text()
        for cpu in ["gfx942", "gfx950"]:
            name = f"PRODUCTION_{cpu.upper()}_DEVICE_TARGET_V1"
            target, = re.findall(rf'pub const {name}: &str = "([^"]+)";', source)
            command = sim.simulation_command("sim", "graph", "request", "output", cpu)
            self.assertEqual(command, ["sim", "--diagnostic-kir-v18", "graph",
                                      "--diagnostic-target", target, "--request", "request",
                                      "--output", "output"])
        for cpu in ["gfx999", "gfx942:xnack+", "gfx942:xnack-", "gfx950:xnack-", "gfx942 ", None]:
            with self.assertRaises(ValueError):
                sim.simulation_command("sim", "graph", "request", "output", cpu)

    def test_same_request_complete_bytes_and_target_pass(self):
        raw, reference, result, graph = fixture()
        checked = sim.compare(raw, reference, result, graph, "gfx942")
        self.assertEqual(checked["checkedBufferBytes"], 4)
        self.assertFalse(checked["authority"])
        self.assertFalse(checked["hardwareExecuted"])

    def test_reference_and_simulator_substitutions_refuse(self):
        mutations = [lambda q, r: q.update(request_sha256=[0]*32),
            lambda q, r: q.update(kernel="other"), lambda q, r: q.update(authority=True),
            lambda q, r: q["outputs"].clear(),
            lambda q, r: q["outputs"].append(copy.deepcopy(q["outputs"][0])),
            lambda q, r: q["outputs"][0].update(absolute_tolerance=1),
            lambda q, r: r.update(hardware_observed=True),
            lambda q, r: r["kir"].update(sha256="b"*64),
            lambda q, r: r["target_profile"].update(identity="amdgpu_gfx950_little_endian_v2"),
            lambda q, r: r["schedule"]["coverage"].update(complete=False),
            lambda q, r: r["conflict_assessment"].update(status="conflict"),
            lambda q, r: r["arguments"][0]["value"].update(bytes="0x26000000"),
            lambda q, r: r["arguments"][0]["value"].update(initialized="0x07"),
            lambda q, r: r["arguments"][0]["value"].update(alignment=8),
            lambda q, r: r["arguments"][1].update(bits="0x00000026")]
        for mutate in mutations:
            raw, reference, result, graph = fixture()
            graph = copy.deepcopy(graph)
            mutate(reference, result)
            with self.assertRaises(ValueError):
                sim.compare(raw, reference, result, graph, "gfx942")

    def test_readonly_and_output_padding_remain_observed(self):
        raw, reference, result, graph = fixture()
        request = json.loads(raw)
        request["arguments"][0]["bytes"] += "efbeadde"
        result["arguments"][0]["value"].update(bytes="0x25000000efbeadde", initialized="0xff")
        reference["outputs"][0]["bytes"] += "efbeadde"
        raw = json.dumps(request).encode()
        reference["request_sha256"] = list(hashlib.sha256(raw).digest())
        sim.compare(raw, reference, result, graph, "gfx942")
        result["arguments"][0]["value"]["bytes"] = "0x2500000000000000"
        with self.assertRaises(ValueError):
            sim.compare(raw, reference, result, graph, "gfx942")

    def test_command_requires_eof_reap_and_actual_zero(self):
        success = {"status": "cargo-completed-unqualified", "exitCode": 0,
                   "logComplete": True, "directChildReaped": True}
        self.assertTrue(sim.command_passed(success))
        for key, value in [("status", "timeout"), ("exitCode", 1), ("logComplete", False),
                           ("directChildReaped", False)]:
            self.assertFalse(sim.command_passed(dict(success, **{key: value})))

    def test_special_files_and_artifact_census_refuse(self):
        import os
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            os.mkfifo(root / "fifo")
            with self.assertRaises(ValueError):
                sim.read_file(root / "fifo")
            with self.assertRaises(ValueError):
                gate.check_no_artifacts(root)
            (root / "fifo").unlink()
            (root / "ordinary.rlib").write_bytes(b"fixture")
            self.assertEqual(gate.check_no_artifacts(root)["entriesInspected"], 1)
            (root / "actual.hsaco").write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                gate.check_no_artifacts(root)

    def test_current_manifest_corpus_keeps_all_invocations_and_negatives(self):
        census = gate.gate.load_module("census_v92_test", ROOT / "scripts/tutorial_source_census.py")
        manifest = json.loads((ROOT / "config/tutorial-kernel-manifest-v1.json").read_bytes())
        rows = census.registered_invocations(manifest)
        self.assertEqual(len(rows), 109)
        self.assertEqual(sum(row[2] for row in rows.values()), 3)
        self.assertEqual(len({name for fixture, _, neg in rows.values() if not neg
                              for name in fixture["compilerInput"]["kernelSymbols"]}), 45)
        paired_families = (
            "attnres-aggregate", "compressed-hybrid-attention", "content-sparse-attention",
            "deepseek-sparse-attention", "four-branch-residual", "gpt-oss-decode",
            "gpt-oss-held-fragments", "gpt-oss-interleaved-stores", "gpt-oss-materialized-attention",
            "gpt-oss-materialized-expert", "gpt-oss-materialized-router", "gpt-oss-pipelined-attention",
            "gpt-oss-scalar-attention", "gpt-oss-serial-router", "kda-decode", "kda-prefill",
            "mhc-sinkhorn-mix", "row-softmax", "scalar-gemm",
        )
        added_ids = {f"gfx950-{family}-{mode}" for family in paired_families
                     for mode in ("simt", "tile")} | {
            "gfx950-combine-expert-ranks-tile", "gfx950-moe-expert-rank-simt",
            "gfx950-moe-route-tile", "gfx950-muon-update-tile", "gfx950-qwen-ngram-gather-tile",
            "gfx950-speculative-transaction-tile", "gfx950-stage-gradient-shard-tile",
        }
        self.assertEqual(len(added_ids), 45)
        self.assertLessEqual(added_ids, rows.keys())
        added = {key: rows[key] for key in added_ids}
        retained = {key: value for key, value in rows.items() if key not in added_ids}
        self.assertEqual(len(retained), 64)
        self.assertTrue(all(fixture["target"] == "gfx950" and not negative
                            for fixture, _, negative in added.values()))

        # Compare semantic identities, not source-derived hashes that legitimately changed.
        # The retained digest is independently captured from public main c4b2cda1.
        fields = ("packageManifest", "cargoLockPath", "sourcePaths", "cargoTarget",
                  "defaultFeatures", "features", "kernelSymbols")
        def identity_digest(selected):
            values = {key: {"fixtureId": fixture["fixtureId"], "target": fixture["target"],
                           "compilerInput": {field: fixture["compilerInput"][field] for field in fields},
                           "references": refs, "negative": negative}
                      for key, (fixture, refs, negative) in selected.items()}
            raw = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
            return hashlib.sha256(raw).hexdigest()
        retained_sha = "fa8cb94dd6686c68c4ebd288e498ea43b66b95912a8293f01c663cbe5c431081"
        self.assertEqual(identity_digest(retained), retained_sha)
        self.assertEqual(identity_digest(added),
                         "a59bfcf3f627436b2b955164458b1c0fd1008f13925207ece7ebc6c791ca73db")
        for mutation in ("identity", "feature", "reference", "negative"):
            changed = copy.deepcopy(retained)
            key = sorted(changed)[0]
            fixture, refs, negative = changed[key]
            if mutation == "identity":
                changed["unregistered"] = changed.pop(key)
            elif mutation == "feature":
                fixture["compilerInput"]["features"].append("unregistered")
            elif mutation == "reference":
                changed[key] = (fixture, [], negative)
            else:
                changed[key] = (fixture, refs, not negative)
            self.assertNotEqual(identity_digest(changed), retained_sha, mutation)
        negatives = {key: gate.expectation(manifest, {"references": refs})
                     for key, (_, refs, negative) in rows.items() if negative}
        self.assertEqual(negatives, {
            f"source-driver-cpu-semantic-simulation-tab0-case{index}": {
                "kind": "rejected", "bundleVersion": 4, "diagnosticContains": diagnostic,
                "outputArtifact": "absent"}
            for index, diagnostic in ((6, "variant-aware packing evidence"),
                                      (7, "contains a pointer or reference"),
                                      (8, "Drop requiring drop glue"))
        })


class CurrentQualificationOrchestration(unittest.TestCase):
    """Real orchestration and file checks with explicit non-executing tool doubles."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = gate.gate.load_inputs(
            ROOT, ROOT / "config/tutorial-kernel-manifest-v1.json")
        _, _, cls.manifest, _, cls.roster = cls.loaded
        cls.names = sorted({name for fixture, _, negative in cls.roster.values() if not negative
                            for name in fixture["compilerInput"]["kernelSymbols"]})

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tutorial-v92-orchestration-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        tools = {}
        for name in ["reference", "simulator", "node", "cargo_fe2o3"]:
            tools[name] = self.directory / name
            tools[name].write_bytes(b"unit-only executable placeholder; never executed\n")
            tools[name].chmod(0o700)
        for name in ["pinned", "current"]:
            (self.directory / name).mkdir()
        self.args = SimpleNamespace(
            repo_root=ROOT, manifest=ROOT / "config/tutorial-kernel-manifest-v1.json",
            output=self.directory / "output", target_dir=self.directory / "cache",
            site_pinned=self.directory / "pinned", site_current=self.directory / "current",
            site_current_commit="a" * 40, compile_timeout=600, simulation_timeout=120,
            **tools)
        self.mutate_census = lambda census: None
        self.mutate_reference = lambda reference: None
        self.simulated = []
        self.commands = []

    def runner(self, arguments, cwd, environment, log, timeout, maximum):
        self.assertEqual(cwd, ROOT)
        self.assertEqual(environment, {"UNIT_ONLY": "unchanged"})
        self.assertEqual(arguments[0], str(self.args.reference))
        self.assertEqual(timeout, 120)
        self.assertEqual(maximum, sim.MAX_DOCUMENT)
        self.commands.append(arguments)
        if arguments[1:] == ["list"]:
            value = {"schema": "fe2o3-tutorial-reference-corpus-v92",
                     "kernels": list(self.names), "authority": False}
            self.mutate_reference(value)
        else:
            self.assertEqual(arguments[1], "generate")
            self.assertIn(arguments[2], self.names)
            value = {"kernel": arguments[2], "unit_only": True}
        log.write_text(json.dumps(value))
        return {"status": "cargo-completed-unqualified", "exitCode": 0,
                "logComplete": True, "directChildReaped": True}

    def census(self, root, manifest, cargo, cache, output, targets, timeout, maximum, **options):
        self.assertEqual((root, manifest, cargo, cache, targets, timeout, maximum),
                         (ROOT, self.args.manifest, self.args.cargo_fe2o3, self.args.target_dir,
                          ("gfx942", "gfx950"), 600, gate.gate.MAX_LOG_BYTES))
        self.assertIs(options["production"], True)
        self.assertIs(options["capture_graphs"], True)
        self.assertIs(options["isolate_negative_outputs"], True)
        output.mkdir()
        cases = []
        for ordinal, (key, (fixture, references, negative)) in enumerate(sorted(self.roster.items())):
            case = {"id": key, "fixture": copy.deepcopy(fixture),
                    "references": [list(reference) for reference in references],
                    "expectedNegative": negative, "sourceBefore": {"unitOnly": key},
                    "sourceAfter": {"unitOnly": key}, "capturedGraphsV92": f"graphs-{ordinal}"}
            (output / case["capturedGraphsV92"]).mkdir()
            if negative:
                raw = gate.expectation(self.manifest, case)["diagnosticContains"].encode()
                case.update(log=f"negative-{ordinal}.log",
                            negativeArtifactDirectoryV92=f"negative-{ordinal}")
                (output / case["negativeArtifactDirectoryV92"]).mkdir()
                (output / case["log"]).write_bytes(raw)
                case["execution"] = {"status": "cargo-failed", "exitCode": 1,
                    "logComplete": True, "directChildReaped": True,
                    "logSha256": hashlib.sha256(raw).hexdigest(), "logBytes": len(raw)}
            else:
                case["status"] = "production-compile-census-pass"
            cases.append(case)
        value = {"coversAllRegisteredInvocations": True, "cases": cases}
        self.mutate_census(value)
        return value

    def simulate(self, case, graph, request, reference, simulator, output, runner, **options):
        self.assertEqual(reference, self.args.reference)
        self.assertEqual(simulator, self.args.simulator)
        self.assertEqual(options["timeout"], 120)
        self.assertTrue(graph.is_dir())
        kernel = json.loads(request.read_bytes())["kernel"]
        self.assertIn(kernel, case["fixture"]["compilerInput"]["kernelSymbols"])
        self.simulated.append((case["id"], kernel))
        return {"unit_only": True, "authority": False}

    def execute(self):
        site = {"site": self.manifest["curriculum"]["site"]}
        with patch.object(gate, "site_inventory", return_value=site), \
                patch.object(gate.gate, "load_inputs", return_value=self.loaded), \
                patch.object(gate.gate, "run_census", side_effect=self.census), \
                patch.object(gate.simulation, "run_case", side_effect=self.simulate):
            return gate.run(self.args, runner=self.runner,
                            environment={"UNIT_ONLY": "unchanged"})

    def test_complete_registered_roster_runs_once_without_claiming_authority(self):
        report = self.execute()
        expected = [(key, kernel) for key, (fixture, _, negative) in sorted(self.roster.items())
                    if not negative for kernel in fixture["compilerInput"]["kernelSymbols"]]
        self.assertEqual(self.simulated, expected)
        self.assertEqual(len(report["cases"]), 109)
        self.assertEqual((report["registeredInvocations"], report["positiveInvocations"],
                          report["negativeInvocations"]), (109, 106, 3))
        self.assertEqual(sum(case["expectedNegative"] for case in report["cases"]), 3)
        self.assertTrue(report["complete"] and report["currentCompilerSimulationPassed"])
        for name in ["qualified", "authority", "hardwareExecuted",
                     "grantsCompilerOrLaunchAuthority", "legacyBundleContractsReinterpreted"]:
            self.assertIs(report[name], False)
        self.assertEqual(json.loads((self.args.output / "report.json").read_bytes()), report)

    def test_positive_failure_is_retained_and_later_cases_still_run(self):
        def fail_first(census):
            case = next(case for case in census["cases"] if not case["expectedNegative"])
            case["status"] = "production-census-refused"
        self.mutate_census = fail_first
        report = self.execute()
        self.assertTrue(report["complete"])
        self.assertFalse(report["currentCompilerSimulationPassed"])
        failed = [case for case in report["cases"] if not case["passed"]]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error"], "production compilation refused")
        self.assertTrue(self.simulated)

    def test_negative_cannot_pass_with_incomplete_child_output(self):
        def incomplete(census):
            case = next(case for case in census["cases"] if case["expectedNegative"])
            case["execution"]["directChildReaped"] = False
        self.mutate_census = incomplete
        report = self.execute()
        self.assertTrue(report["complete"])
        self.assertFalse(report["currentCompilerSimulationPassed"])
        self.assertEqual(sum(not case["passed"] for case in report["cases"]), 1)

    def test_duplicate_or_substituted_census_cannot_keep_full_coverage(self):
        mutations = [
            lambda cases: cases.__setitem__(0, copy.deepcopy(cases[1])),
            lambda cases: cases[0].update(id="substituted"),
            lambda cases: cases[0]["fixture"].update(target="gfx999"),
            lambda cases: cases[0].update(references=[]),
            lambda cases: cases[0].update(expectedNegative=not cases[0]["expectedNegative"]),
        ]
        for ordinal, mutate in enumerate(mutations):
            with self.subTest(mutation=ordinal):
                self.args.output = self.directory / f"substitution-{ordinal}"
                self.mutate_census = lambda census: mutate(census["cases"])
                with self.assertRaisesRegex(ValueError, "changed the registered invocation"):
                    self.execute()
                self.assertEqual(self.simulated, [])
                report = json.loads((self.args.output / "report.json").read_bytes())
                self.assertFalse(report["complete"] or report["currentCompilerSimulationPassed"])

    def test_short_census_fails_before_any_simulation(self):
        self.mutate_census = lambda census: census["cases"].pop()
        with self.assertRaisesRegex(ValueError, "dropped invocations"):
            self.execute()
        self.assertEqual(self.simulated, [])

    def test_reference_roster_substitution_stops_before_compilation(self):
        self.mutate_reference = lambda reference: reference["kernels"].pop()
        with self.assertRaisesRegex(ValueError, "reference corpus differs"):
            self.execute()
        self.assertEqual(self.simulated, [])
        self.assertFalse((self.args.output / "compilation").exists())

    def test_tool_change_cannot_become_completed_success(self):
        original = self.simulate
        def change(*args, **kwargs):
            self.args.simulator.write_bytes(b"changed unit-only tool\n")
            return original(*args, **kwargs)
        self.simulate = change
        with self.assertRaisesRegex(ValueError, "qualification tools changed"):
            self.execute()
        report = json.loads((self.args.output / "report.json").read_bytes())
        self.assertFalse(report["complete"] or report["currentCompilerSimulationPassed"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Static reproduction-plan contracts; no compiler, proof, simulator or GPU run."""

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
SPEC = importlib.util.spec_from_file_location("tutorial_reproduction_plan", ROOT / "scripts/tutorial_reproduction_plan.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReproductionPlanTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = {
            name: {"fixtureId": name, "target": target,
                   "matrix": {"caseId": case, "runnerPath": "examples/retained.sh", "runnerArguments": [], "environment": []},
                   "compilerInput": {"features": features, "defaultFeatures": False},
                   "simulation": {"status": "pending-reconciliation"}}
            for name, target, case, features in [
                ("a", "gfx942", "z", ["simt"]), ("b", "gfx950", "x", ["tile"]),
                ("c", "gfx942", "a", ["reference-proof"]),
            ]
        }
        self.manifest = {"qualification": {
            "hardwareTargets": [{"target": target, "deterministicSemanticRunnerAvailability": "unavailable",
                                 "reason": "No current hardware receipt."} for target in ["gfx942", "gfx950"]],
            "suites": [{"suiteId": "cross-target", "gate": "cpu-reference", "availability": "pending",
                        "command": {"executable": "scripts/run-tutorial-cpu-reference.sh", "arguments": ["retained", "lib"]},
                        "coverage": [{"fixtureIds": ["a", "b"]}]},
                       {"suiteId": "other", "gate": "semantic-simulation", "availability": "unavailable",
                        "command": None, "coverage": [{"fixtureIds": ["b"]}]}],
        }}

    def plan(self, target="gfx942"):
        return MODULE.build_plan(self.manifest, self.fixtures, target, "a" * 64)

    def test_exact_architecture_selection_and_canonical_case_order(self):
        self.assertEqual([row["fixtureId"] for row in self.plan()["fixtures"]], ["c", "a"])
        self.assertEqual([row["fixtureId"] for row in self.plan("gfx950")["fixtures"]], ["b"])

    def test_features_and_pending_simulation_are_not_reinterpreted(self):
        plan = self.plan()
        self.assertEqual(plan["fixtures"][0], self.fixtures["c"])
        self.assertEqual(plan["fixtures"][0]["compilerInput"]["features"], ["reference-proof"])
        self.assertEqual(plan["fixtures"][0]["simulation"]["status"], "pending-reconciliation")

    def test_cross_target_suite_retains_its_whole_original_coverage(self):
        selected = self.plan()["declaredQualificationSuites"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["selectedFixtureIds"], ["a"])
        self.assertEqual(selected[0]["declaredSuite"], self.manifest["qualification"]["suites"][0])
        self.assertEqual(selected[0]["declaredSuite"]["coverage"][0]["fixtureIds"], ["a", "b"])

    def test_no_execution_installed_compiler_or_hardware_authority(self):
        plan = self.plan()
        self.assertEqual(plan["executionStatus"], "UNRUN")
        self.assertFalse(plan["qualified"])
        self.assertFalse(plan["hardwareExecuted"])
        self.assertFalse(plan["authenticatesCompilerExecution"])
        self.assertIsNone(plan["installedCompiler"])
        self.assertIsNone(plan["hardware"]["command"])
        self.assertEqual(plan["hardware"]["declaredAvailability"], self.manifest["qualification"]["hardwareTargets"][0])
        self.assertEqual(plan["compileOnlyMatrix"]["command"], ["bash", "scripts/kernel-compile-matrix.sh", "gfx942"])

    def test_inputs_unchanged_and_output_does_not_alias_them(self):
        before = copy.deepcopy((self.manifest, self.fixtures))
        plan = self.plan()
        plan["fixtures"][0]["compilerInput"]["features"].append("changed")
        plan["declaredQualificationSuites"][0]["declaredSuite"]["coverage"].clear()
        plan["hardware"]["declaredAvailability"]["reason"] = "changed"
        self.assertEqual((self.manifest, self.fixtures), before)

    def test_unknown_or_empty_target_never_falls_back(self):
        for target in ["gfx941", "gfx942:xnack-", "", None]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.plan(target)
        with self.assertRaises(ValueError):
            MODULE.build_plan(self.manifest, {}, "gfx942", "a" * 64)

    def test_missing_or_duplicate_hardware_record_is_not_invented(self):
        self.manifest["qualification"]["hardwareTargets"] = []
        with self.assertRaises(ValueError):
            self.plan()
        self.manifest["qualification"]["hardwareTargets"] = [{"target": "gfx942"}] * 2
        with self.assertRaises(ValueError):
            self.plan()

    def test_raw_manifest_digest_must_be_exact(self):
        for digest in ["", "main", "A" * 64, "a" * 63, None]:
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                MODULE.build_plan(self.manifest, self.fixtures, "gfx942", digest)


class MaintainedReproductionCliTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/validate-tutorial-kernel-manifest.py"),
                               *arguments], cwd=ROOT, text=True, capture_output=True, timeout=90)

    def test_both_targets_select_exact_current_manifest_records(self):
        payload = (ROOT / "config/tutorial-kernel-manifest-v1.json").read_bytes()
        manifest = json.loads(payload)
        for target in ["gfx942", "gfx950"]:
            with self.subTest(target=target):
                result = self.run_cli("--emit-reproduction-plan", target)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                expected = sorted((row for row in manifest["compilerFixtures"] if row["target"] == target),
                                  key=lambda row: row["matrix"]["caseId"])
                self.assertEqual(plan["fixtures"], expected)
                self.assertEqual(plan["manifestSha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(plan["executionStatus"], "UNRUN")
                self.assertFalse(plan["qualified"])

    def test_invalid_mode_combinations_emit_no_plan(self):
        for arguments, diagnostic in [
            (["--emit-reproduction-plan", "gfx941"], "invalid choice"),
            (["--emit-reproduction-plan", "gfx942", "--emit-matrix", "gfx942"], "not allowed with argument"),
            (["--emit-reproduction-plan", "gfx942", "--ordinary-source-report", "missing.json"],
             "--ordinary-source-report requires --emit-kernel-pairs"),
            (["--emit-reproduction-plan", "gfx942", "--require-qualified"],
             "qualification receipts and policy/final-graph evidence are not implemented"),
        ]:
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(diagnostic, result.stderr)

    def test_stale_physical_input_refuses_without_a_partial_plan(self):
        manifest = json.loads((ROOT / "config/tutorial-kernel-manifest-v1.json").read_text())
        manifest["compilerFixtures"][0]["compilerInput"]["packageManifestSha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stale-manifest.json"
            path.write_text(json.dumps(manifest))
            result = self.run_cli("--manifest", str(path), "--emit-reproduction-plan", "gfx942")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("packageManifestSha256 is stale", result.stderr)

    def test_newcomer_links_and_commands_remain_explicit(self):
        guide = (ROOT / "docs/tutorial-reproduction.md").read_text()
        self.assertIn("--emit-reproduction-plan gfx942", guide)
        self.assertIn("--emit-reproduction-plan gfx950", guide)
        self.assertIn("UNRUN", guide)
        self.assertIn("verified-U32", guide)
        self.assertIn("no supported public source-to-GPU dispatch command", " ".join(guide.split()))
        self.assertIn("tutorial-reproduction.md", (ROOT / "docs/getting-started.md").read_text())


if __name__ == "__main__":
    unittest.main()

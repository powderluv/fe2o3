#!/usr/bin/env python3
"""Source-contract and matrix enumeration tests; never production receipts."""

from __future__ import annotations

import copy
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/validate-tutorial-kernel-manifest.py"
MANIFEST = ROOT / "config/tutorial-kernel-manifest-v1.json"


def load_validator():
    specification = importlib.util.spec_from_file_location("tutorial_contract", CHECKER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def recovered_scope(manifest):
    added_gpt_fixtures = {
        f"gfx950-gpt-oss-{component}-{mode}"
        for component in (
            "decode", "held-fragments", "interleaved-stores", "materialized-attention",
            "materialized-expert", "materialized-router", "pipelined-attention",
            "scalar-attention", "serial-router",
        )
        for mode in ("simt", "tile")
    }
    added_fixtures = {
        'gfx950-moe-expert-rank-simt', 'gfx950-moe-route-tile', 'gfx950-combine-expert-ranks-tile', 'gfx950-speculative-transaction-tile', 'gfx950-qwen-ngram-gather-tile', 'gfx950-stage-gradient-shard-tile', 'gfx950-muon-update-tile',
        "gfx942-scalar-gemm", "gfx950-gpt-oss-pipelined-attention",
        "gfx950-gpt-oss-scalar-attention",
        "gfx950-row-softmax-simt", "gfx950-row-softmax-tile",
        "gfx950-scalar-gemm-simt", "gfx950-scalar-gemm-tile",
        "gfx950-kda-decode-simt",
        "gfx950-kda-decode-tile",
        "gfx950-kda-prefill-simt",
        "gfx950-kda-prefill-tile",
        "gfx950-content-sparse-attention-simt",
        "gfx950-content-sparse-attention-tile",
        "gfx950-deepseek-sparse-attention-simt",
        "gfx950-deepseek-sparse-attention-tile",
        "gfx950-compressed-hybrid-attention-simt",
        "gfx950-compressed-hybrid-attention-tile",
        "gfx950-attnres-aggregate-simt",
        "gfx950-attnres-aggregate-tile",
        "gfx950-four-branch-residual-simt",
        "gfx950-four-branch-residual-tile",
        "gfx950-mhc-sinkhorn-mix-simt",
        "gfx950-mhc-sinkhorn-mix-tile",
    } | added_gpt_fixtures
    fixtures = []
    for original in manifest["compilerFixtures"]:
        if original["fixtureId"] in added_fixtures:
            continue
        fixture = {key: original[key] for key in ("fixtureId", "testId", "testPath", "target", "matrix")}
        fixture["compilerInput"] = {
            key: original["compilerInput"][key]
            for key in (
                "packageManifest", "sourcePaths", "cargoTarget", "defaultFeatures",
                "features", "kernelSymbols", "cargoLockPath",
            )
        }
        fixture["simulation"] = {key: value for key, value in original["simulation"].items() if key != "status"}
        fixtures.append(fixture)
    entries = copy.deepcopy(manifest["entries"])
    for entry in entries:
        entry["compilerFixtureIds"] = [
            value for value in entry["compilerFixtureIds"] if value not in added_fixtures
        ]
    for entry in entries:
        if entry["lessonId"] == "gfx950-attnres-gr-mhc":
            entry["sourcePaths"] = [path for path in entry["sourcePaths"]
                                    if path != "examples/gfx950_advanced_attention/src/kernel_tile_mixing.rs"]
    for entry in entries:
        entry["sourcePaths"] = [
            path for path in entry["sourcePaths"] if path not in {
                "examples/gfx950_advanced_attention/src/kernel_simt_content_sparse.rs",
                "examples/gfx950_advanced_attention/src/kernel_simt_compressed_hybrid.rs",
            }
        ]
    added_paths = {
        "gfx950-gpt-oss-120b-megakernel": {
            "examples/gfx950_gpt_oss_decode/src/" + name
            for name in (
                "kernel_held_fragments.rs", "kernel_held_fragments_simt.rs",
                "kernel_held_fragments_tile.rs", "kernel_interleaved_stores.rs",
                "kernel_interleaved_stores_simt.rs", "kernel_interleaved_stores_tile.rs",
                "kernel_pipelined_attention.rs", "kernel_pipelined_attention_simt.rs",
                "kernel_pipelined_attention_tile.rs", "kernel_scalar_attention.rs",
                "kernel_scalar_attention_simt.rs", "kernel_simt.rs",
                "kernel_simt_components.rs", "kernel_tile.rs", "kernel_tile_components.rs",
            )
        },
        "gfx950-advanced-moe": {
            "examples/gfx950_advanced_systems/src/" + name
            for name in (
                "kernel_simt_moe_expert_rank.rs", "kernel_tile_moe_route.rs",
                "kernel_tile_combine_expert_ranks.rs", "kernel_tile_stage_gradient_shard.rs",
            )
        },
        "gfx950-muon-optimizer": {
            "examples/gfx950_advanced_systems/src/kernel_tile_muon_update.rs",
        },
        "gfx950-ngram-embedding-gather": {
            "examples/gfx950_advanced_systems/src/kernel_tile_qwen_ngram_gather.rs",
        },
    }
    for entry in entries:
        entry["sourcePaths"] = [path for path in entry["sourcePaths"]
                                if path not in added_paths.get(entry["lessonId"], set())]
    suites = []
    for original in manifest["qualification"]["suites"]:
        if original["suiteId"] in {
            'semantic-simulation-gfx950-moe-expert-rank-simt', 'semantic-simulation-gfx950-moe-route-tile', 'semantic-simulation-gfx950-combine-expert-ranks-tile', 'semantic-simulation-gfx950-speculative-transaction-tile', 'semantic-simulation-gfx950-qwen-ngram-gather-tile', 'semantic-simulation-gfx950-stage-gradient-shard-tile', 'semantic-simulation-gfx950-muon-update-tile',
            "cpu-reference-scalar-gemm", "semantic-simulation-scalar-gemm",
            "semantic-simulation-gfx950-gpt-oss-pipelined-attention",
            "semantic-simulation-gfx950-gpt-oss-scalar-attention",
            "cpu-reference-tiled-gemm-paired-default",
            "cpu-reference-tiled-gemm-paired-simt",
            "cpu-reference-gfx950-row-softmax-simt", "cpu-reference-gfx950-row-softmax-tile",
            "cpu-reference-gfx950-scalar-gemm-simt", "cpu-reference-gfx950-scalar-gemm-tile",
            "semantic-simulation-gfx950-row-softmax-simt", "semantic-simulation-gfx950-row-softmax-tile",
            "semantic-simulation-gfx950-scalar-gemm-simt", "semantic-simulation-gfx950-scalar-gemm-tile",
            "semantic-simulation-gfx950-attnres-aggregate-simt",
            "semantic-simulation-gfx950-attnres-aggregate-tile",
            "semantic-simulation-gfx950-four-branch-residual-simt",
            "semantic-simulation-gfx950-four-branch-residual-tile",
            "semantic-simulation-gfx950-mhc-sinkhorn-mix-simt",
            "semantic-simulation-gfx950-mhc-sinkhorn-mix-tile",
        } or original["suiteId"] in {"semantic-simulation-" + fixture for fixture in added_fixtures} or original["suiteId"] in {
            "cpu-reference-" + fixture for fixture in added_gpt_fixtures
        }:
            continue
        suite = {key: copy.deepcopy(original[key]) for key in ("suiteId", "gate", "command", "coverage")}
        for coverage in suite["coverage"]:
            coverage["fixtureIds"] = [value for value in coverage["fixtureIds"] if value not in added_fixtures]
        suites.append(suite)
    return {
        "productionContract": manifest["productionContract"],
        "fixtures": fixtures,
        "entries": entries,
        "suites": suites,
    }


class TutorialKernelSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_validator()
        cls.original = cls.validator.load_manifest(MANIFEST)

    def setUp(self):
        self.manifest = copy.deepcopy(self.original)

    def reject(self, pattern):
        with self.assertRaisesRegex(SystemExit, pattern):
            self.validator.validate_manifest(ROOT, self.manifest)

    def test_all_expected_source_contracts_match_and_are_not_receipts(self):
        with mock.patch.object(
            self.validator, "package_rust_sources", wraps=self.validator.package_rust_sources
        ) as walk, mock.patch.object(
            self.validator, "_validate_package_source_path", wraps=self.validator._validate_package_source_path
        ) as source_path, mock.patch.object(
            self.validator, "source_contains_ordinary_attributed_kernel", wraps=self.validator.source_contains_ordinary_attributed_kernel
        ) as source_scan:
            fixtures = self.validator.validate_manifest(ROOT, self.manifest)
        lessons = {entry["lessonId"]: len(entry["sourcePaths"]) for entry in self.manifest["entries"]}
        self.assertEqual(
            Counter(call.args[3] for call in source_path.call_args_list if call.args[3] in lessons),
            Counter(lessons),
        )
        self.assertEqual(source_scan.call_count, sum(lessons.values()))
        expected_packages = {
            f["compilerInput"]["packageManifest"] for f in self.manifest["compilerFixtures"]
        } | {entry["packageManifest"] for entry in self.manifest["entries"]}
        expected_packages.update(
            tab["sourceItem"]["compilerInput"]["packageManifest"]
            for lesson in self.manifest["curriculum"]["lessons"] for tab in lesson["codeTabs"]
            if tab["sourceItem"] is not None
        )
        self.assertEqual(
            Counter(call.args[1] for call in walk.call_args_list),
            Counter({package: 1 for package in expected_packages}),
        )
        self.assertEqual(len(fixtures), 95)
        self.assertEqual(sum(f["target"] == "gfx942" for f in fixtures.values()), 11)
        self.assertEqual(sum(f["target"] == "gfx950" for f in fixtures.values()), 84)
        self.assertEqual(len(self.manifest["entries"]), 25)
        self.assertEqual(len(self.manifest["qualification"]["suites"]), 131)
        self.assertTrue(all(e["classification"] == "compiler-produced" for e in self.manifest["entries"]))
        self.assertTrue(all(s["availability"] == "pending" for s in self.manifest["qualification"]["suites"]))

    def test_f32_gfx950_counterparts_match_real_runner_modes(self):
        fixtures = {row["fixtureId"]: row for row in self.manifest["compilerFixtures"]}
        suites = {row["suiteId"]: row for row in self.manifest["qualification"]["suites"]}
        for family, package in (("row-softmax", "row_softmax_general_v1"),
                                ("scalar-gemm", "scalar_gemm_v1")):
            for mode in ("simt", "tile"):
                fixture_id = f"gfx950-{family}-{mode}"
                with self.subTest(fixture=fixture_id):
                    fixture = fixtures[fixture_id]
                    features = ["gfx950-target", "qualification-legacy-hip"]
                    if mode == "tile":
                        features.append("structured-tile")
                    inputs = fixture["compilerInput"]
                    self.assertEqual(fixture["target"], "gfx950")
                    self.assertEqual(fixture["matrix"]["runnerArguments"], [mode])
                    self.assertEqual(fixture["matrix"]["runnerPath"], f"examples/{package}/run-gfx950.sh")
                    self.assertEqual(fixture["matrix"]["artifactName"], package + ".hsaco")
                    self.assertEqual(inputs["features"], features)
                    self.assertIs(inputs["defaultFeatures"], False)
                    leaf = "kernel_tile.rs" if mode == "tile" else "kernel.rs"
                    self.assertEqual(inputs["sourcePaths"], [f"examples/{package}/src/{leaf}"])
                    runner = (ROOT / fixture["matrix"]["runnerPath"]).read_text()
                    runner_features = ["gfx950-target"]
                    if mode == "tile":
                        runner_features.append("structured-tile")
                    runner_features.append("qualification-legacy-hip")
                    self.assertIn(f"{mode}) FEATURES=" + ",".join(runner_features), runner)
                    cpu = suites["cpu-reference-" + fixture_id]
                    self.assertEqual(cpu["command"]["arguments"],
                                     [inputs["packageManifest"], "lib", "--no-default-features",
                                      "--features", ",".join(features)])

    def test_f32_gfx950_counterparts_do_not_promote_retained_evidence(self):
        for fixture in self.manifest["compilerFixtures"]:
            if fixture["fixtureId"] not in {
                "gfx950-row-softmax-simt", "gfx950-row-softmax-tile",
                "gfx950-scalar-gemm-simt", "gfx950-scalar-gemm-tile",
            }:
                continue
            with self.subTest(fixture=fixture["fixtureId"]):
                simulation = fixture["simulation"]
                self.assertEqual(simulation["status"], "pending-design")
                self.assertEqual((simulation["bundleVersion"], simulation["canonicalKirVersion"]), (7, 12))
                for field in ("requestSha256", "requestBytes", "expectationSha256", "expectationBytes"):
                    self.assertIsNone(simulation[field])
                kernel_id = f"fixture:{fixture['fixtureId']}:{fixture['compilerInput']['kernelSymbols'][0]}"
                kernel = next(row for row in self.manifest["kernelInventory"]["kernels"]
                              if row["kernelId"] == kernel_id)
                self.assertEqual([row["kind"] for row in kernel["variants"]], ["simt", "tile"])
                self.assertTrue(all(row["status"] == "pending" and row["source"] is None
                                    for row in kernel["variants"]))
        self.assertEqual(self.manifest["qualification"]["status"], "pending")

    def test_original_47_source_feature_symbol_and_semantic_obligations_are_preserved(self):
        payload = json.dumps(
            recovered_scope(self.manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), "db1d9a0d5c5aca3b9c76a4713ddd3b22417e41ee24667efae11773dfccdc4215")

    def test_gemm_paired_cpu_reference_suites_keep_explicit_feature_selections(self):
        suites = {suite["suiteId"]: suite for suite in self.manifest["qualification"]["suites"]}
        original = suites["cpu-reference-tiled-gemm"]
        manifest = "examples/tiled_gemm_general_v1/Cargo.toml"
        self.assertEqual(original["command"]["arguments"], [manifest, "lib"])
        selections = {
            "cpu-reference-tiled-gemm-paired-default": [],
            "cpu-reference-tiled-gemm-paired-simt": [
                "--no-default-features", "--features", "kernel-simt-gemm-general",
            ],
        }
        for suite_id, features in selections.items():
            with self.subTest(suite=suite_id):
                suite = suites[suite_id]
                expected_command = copy.deepcopy(original["command"])
                expected_command["arguments"] = [manifest, "test", "paired_contract", *features]
                self.assertEqual(suite["command"], expected_command)
                self.assertEqual(suite["gate"], "cpu-reference")
                self.assertEqual(suite["availability"], "pending")
                self.assertEqual(suite["coverage"], original["coverage"])

    def test_full_runtime_curriculum_snapshot_preserves_pending_obligations(self):
        curriculum = self.manifest["curriculum"]
        lessons = curriculum["lessons"]
        self.assertEqual(len(lessons), 56)
        self.assertEqual(Counter(lesson["role"] for lesson in lessons), {"executable": 46, "conceptual": 10})
        self.assertEqual(sum(len(lesson["codeTabs"]) for lesson in lessons), 314)
        self.assertEqual(
            [lesson["lessonId"] for lesson in lessons if any(v["kind"] == "mixed" for v in lesson["variants"])],
            ["cpu-semantic-simulation", "reductions-scans", "gemm-tiling", "softmax-invariant"],
        )
        payload = json.dumps(curriculum, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), "34b1d6e35f3c26f9b0aef61d14f6555de11321912f8a49d91f51dc6e5cda4199")

    def test_historical_verus_result_matches_retained_acb3_counts(self):
        lesson = self.curriculum_lesson("verus-contracts")
        self.assertEqual(lesson["role"], "conceptual")
        self.assertEqual(lesson["variants"], [])
        tab = lesson["codeTabs"][3]
        self.assertEqual(tab["kind"], "result")
        self.assertEqual(tab["sourceItemStatus"], "not-applicable")
        self.assertIsNone(tab["sourceCommit"])
        self.assertIsNone(tab["sourceItem"])
        expected = (
            "source-model-verified\n\n"
            "Expected results at acb3d275: 5 verified harnesses and 24 intended proof rejections. "
            "Verus fixture run passed: 5 proof harnesses, 24 expected rejections."
        ).encode("utf-8")
        self.assertEqual(tab["displayedUtf8Bytes"], len(expected))
        self.assertEqual(tab["displayedSha256"], hashlib.sha256(expected).hexdigest())

    def test_public_grouped_expert_moe_display_keeps_physical_identity_and_history(self):
        lesson = self.curriculum_lesson("moe-expert-compute")
        tab = lesson["codeTabs"][0]
        source = (ROOT / "examples/moe_grouped_expert_general_v1/src/kernel.rs").read_bytes()
        self.assertEqual(len(source), 11320)
        self.assertEqual(tab["sourceCommit"], "d558aa97758cf553c956c5ce92178eac96bd2461")
        self.assertEqual(tab["displayedUtf8Bytes"], len(source))
        self.assertEqual(tab["displayedSha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(tab["sourceSha256"], tab["displayedSha256"])
        self.assertEqual(tab["evidenceId"], "grouped-expert-moe-current-source-v1")
        self.assertIsNone(tab["sourceItem"])
        self.assertEqual(tab["sourceItemStatus"], "pending")
        rows = [r for r in self.manifest["kernelInventory"]["displayItems"]
                if r["lessonId"] == "moe-expert-compute" and r["tabOrdinal"] == 0]
        actual = self.validator.ordinary_rust_function_items(source.decode("utf-8"))
        self.assertEqual([(r["kernelSymbol"], r["functionUtf8Offset"]) for r in rows],
                         [(r["kernelSymbol"], r["functionUtf8Offset"]) for r in actual])
        self.assertEqual(len(rows), 10)
        kernel = next(r for r in rows if r["classification"] == "kernel")
        self.assertEqual(kernel["bindingStatus"], "fixture-source-contract")
        self.assertEqual(kernel["kernelIds"],
                         ["fixture:gfx942-grouped-expert-moe:moe_grouped_expert_general_v1"])
        self.assertTrue(all(v["status"] == "pending" and v["sourceItems"] == []
                            for v in lesson["variants"]))
        self.assertEqual(lesson["codeTabs"][3]["sourceCommit"],
                         "308d8fa00fa41e098b2a1a47bbfea1bc29735464")
        self.assertEqual(lesson["codeTabs"][4]["sourceCommit"],
                         "af0fd523e3b774377a9c5192cf0511e34fa19735")

    def test_public_grouped_expert_moe_rejects_its_old_physical_kernel_offset(self):
        kernel = next(r for r in self.manifest["kernelInventory"]["displayItems"]
                      if r["lessonId"] == "moe-expert-compute" and r["classification"] == "kernel")
        self.assertEqual(kernel["functionUtf8Offset"], 1806)
        kernel["functionUtf8Offset"] = 1209
        self.reject("fixture display differs from the exact current source occurrence")

    def test_current_flash_display_preserves_every_physical_function_and_history(self):
        lesson = self.curriculum_lesson("flash-attention")
        tab = lesson["codeTabs"][0]
        source = (ROOT / "examples/flash_attention_general_v1/src/kernel.rs").read_bytes()
        self.assertEqual(tab["sourceCommit"], "055ffa8dccc70fccc7675721f98f2b096abd68e7")
        self.assertEqual(tab["displayedUtf8Bytes"], 34804)
        self.assertEqual(tab["sourceSha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(tab["displayedSha256"], tab["sourceSha256"])
        self.assertEqual(tab["evidenceId"], "flash-attention-current-source-v1")
        self.assertEqual(tab["sourceItemStatus"], "pending")
        self.assertIsNone(tab["sourceItem"])
        rows = [row for row in self.manifest["kernelInventory"]["displayItems"]
                if row["lessonId"] == "flash-attention" and row["tabOrdinal"] == 0]
        actual = self.validator.ordinary_rust_function_items(source.decode())
        self.assertEqual([(row["kernelSymbol"], row["functionUtf8Offset"]) for row in rows],
                         [(row["kernelSymbol"], row["functionUtf8Offset"]) for row in actual])
        self.assertEqual(len(rows), 23)
        kernel = next(row for row in rows if row["classification"] == "kernel")
        self.assertEqual(kernel["functionUtf8Offset"], 3320)
        self.assertEqual(kernel["bindingStatus"], "fixture-source-contract")
        self.assertTrue(all(variant["status"] == "pending" and not variant["sourceItems"]
                            for variant in lesson["variants"]))
        self.assertEqual(lesson["codeTabs"][1]["sourceCommit"], "9006001157e2c3062e44088634e467b0f8963ee0")
        kernel["functionUtf8Offset"] = 1688
        self.reject("fixture display differs from the exact current source occurrence")

    def test_current_scalar_display_registers_all_helpers_without_kernel_authority(self):
        lesson = self.curriculum_lesson("gfx950-gpt-oss-120b-megakernel")
        tab = lesson["codeTabs"][6]
        source = (ROOT / tab["sourcePath"]).read_text()
        rows = [row for row in self.manifest["kernelInventory"]["displayItems"]
                if row["lessonId"] == lesson["lessonId"] and row["tabOrdinal"] == 6]
        actual = self.validator.ordinary_rust_function_items(source)
        self.assertEqual([(row["kernelSymbol"], row["functionUtf8Offset"]) for row in rows],
                         [(row["kernelSymbol"], row["functionUtf8Offset"]) for row in actual])
        self.assertEqual(len(rows), 3)
        helpers = [row for row in rows if row["classification"] == "helper"]
        self.assertEqual([row["functionUtf8Offset"] for row in helpers], [1074, 24885])
        self.assertTrue(all(row["bindingStatus"] == "not-applicable" and not row["kernelIds"]
                            and not row["negativeCases"] for row in helpers))

    def test_current_public_gemm_display_matches_source_without_transferring_receipts(self):
        lesson = self.curriculum_lesson("gemm-tiling")
        tab = lesson["codeTabs"][0]
        source = (ROOT / "examples/tiled_gemm_general_v1/src/kernel.rs").read_bytes()
        self.assertEqual(tab["sourceCommit"], "0e0963df4f230bf728a7b4605677c0ff09624adf")
        self.assertEqual(tab["displayedUtf8Bytes"], len(source))
        self.assertEqual(tab["displayedSha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(tab["sourceSha256"], tab["displayedSha256"])
        self.assertEqual(tab["evidenceId"], "tiled-gemm-current-source-v1")
        self.assertEqual(tab["sourceItemStatus"], "pending")
        self.assertIsNone(tab["sourceItem"])
        rows = [row for row in self.manifest["kernelInventory"]["displayItems"]
                if row["lessonId"] == "gemm-tiling" and row["tabOrdinal"] == 0]
        actual = self.validator.ordinary_rust_function_items(source.decode("utf-8"))
        self.assertEqual([(row["kernelSymbol"], row["functionUtf8Offset"]) for row in rows],
                         [(row["kernelSymbol"], row["functionUtf8Offset"]) for row in actual])
        kernel = next(row for row in rows if row["classification"] == "kernel")
        self.assertEqual(kernel["bindingStatus"], "pending")
        self.assertEqual(kernel["kernelIds"], ["fixture:gfx942-tiled-gemm:tiled_gemm_general_v1"])
        self.assertTrue(all(variant["status"] == "pending" and variant["sourceItems"] == []
                            for variant in lesson["variants"]))
        self.assertEqual(lesson["codeTabs"][5]["sourceCommit"],
                         "9006001157e2c3062e44088634e467b0f8963ee0")

    def test_legacy_manifests_remain_accepted_but_required_curriculum_cannot_be_omitted(self):
        self.manifest.pop("kernelInventory", None)
        del self.manifest["curriculum"]
        self.assertEqual(len(self.validator.validate_manifest(ROOT, self.manifest)), 95)
        with tempfile.TemporaryDirectory(prefix="fe2o3-curriculum-") as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CHECKER), "--manifest", str(path), "--require-curriculum"],
                text=True, capture_output=True, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exhaustive curriculum extension is required", result.stderr)

    def validate_curriculum(self):
        self.validator.validate_curriculum(
            self.manifest["curriculum"],
            {entry["lessonId"]: entry for entry in self.manifest["entries"]},
            {fixture["fixtureId"]: fixture for fixture in self.manifest["compilerFixtures"]},
            ROOT,
        )

    def curriculum_lesson(self, lesson_id="first-fill"):
        return next(lesson for lesson in self.manifest["curriculum"]["lessons"] if lesson["lessonId"] == lesson_id)

    def test_curriculum_rejects_duplicate_unknown_and_detached_source_entries(self):
        for ids, pattern in ((["first-fill", "first-fill"], "duplicate"), (["absent"], "unknown"), (["typed-vecadd"], "detach")):
            with self.subTest(ids=ids):
                self.manifest = copy.deepcopy(self.original)
                self.curriculum_lesson()["sourceEntryIds"] = ids
                with self.assertRaisesRegex(SystemExit, pattern):
                    self.validate_curriculum()
        self.manifest = copy.deepcopy(self.original)
        self.manifest["curriculum"]["lessons"].append(copy.deepcopy(self.curriculum_lesson()))
        with self.assertRaisesRegex(SystemExit, "duplicate or invalid curriculum lessonId"):
            self.validate_curriculum()

    def test_conceptual_lessons_cannot_hide_existing_or_carry_new_runnable_obligations(self):
        self.curriculum_lesson()["role"] = "conceptual"
        with self.assertRaisesRegex(SystemExit, "cannot downgrade"):
            self.validate_curriculum()
        self.manifest = copy.deepcopy(self.original)
        self.curriculum_lesson("compiler-checks")["variants"] = copy.deepcopy(self.curriculum_lesson()["variants"])
        with self.assertRaisesRegex(SystemExit, "conceptual lesson cannot carry runnable obligations"):
            self.validate_curriculum()

    def test_variants_require_both_programming_modes_and_cannot_claim_implementation(self):
        for mutate, pattern in (
            (lambda variants: variants.pop(), "ordered SIMT/tile"),
            (lambda variants: variants.reverse(), "ordered SIMT/tile"),
            (lambda variants: variants[0].update(status="qualified"), "pending obligations"),
            (lambda variants: variants[0].update(sourceItems=[{"symbol": "invented"}]), "pending obligations"),
            (lambda variants: variants[0].pop("sourceItems"), "keys differ"),
        ):
            self.manifest = copy.deepcopy(self.original)
            mutate(self.curriculum_lesson()["variants"])
            with self.assertRaisesRegex(SystemExit, pattern):
                self.validate_curriculum()
        self.manifest = copy.deepcopy(self.original)
        for lesson in self.manifest["curriculum"]["lessons"]:
            lesson["variants"] = [variant for variant in lesson["variants"] if variant["kind"] != "mixed"]
        with self.assertRaisesRegex(SystemExit, "mixed SIMT/tile showcase"):
            self.validate_curriculum()

    def test_missing_new_source_association_remains_an_explicit_unqualified_gap(self):
        lesson = self.curriculum_lesson("gemm-autoresearch")
        lesson["sourceEntryIds"] = []
        with self.assertRaisesRegex(SystemExit, "sourceBindingGap"):
            self.validate_curriculum()
        lesson["sourceBindingGap"] = "An actual source fixture must be supplied; this is not implementation."
        self.validate_curriculum()

    def test_source_item_obligations_cannot_be_manufactured_or_dropped(self):
        for key, value in (("sourceItem", {"symbol": "fill"}), ("sourceItemStatus", "not-applicable")):
            self.manifest = copy.deepcopy(self.original)
            self.curriculum_lesson()["codeTabs"][0][key] = value
            with self.assertRaisesRegex(SystemExit, "source-item obligation|source item is a contract"):
                self.validate_curriculum()

    def test_whole_file_metadata_and_tab_ordinals_remain_required(self):
        tab = next(tab for lesson in self.manifest["curriculum"]["lessons"] for tab in lesson["codeTabs"] if tab["sourceDigestScope"] == "file")
        tab["sourceCommit"] = None
        with self.assertRaisesRegex(SystemExit, "incomplete whole-file"):
            self.validate_curriculum()
        self.manifest = copy.deepcopy(self.original)
        self.curriculum_lesson()["codeTabs"][0]["ordinal"] = True
        with self.assertRaisesRegex(SystemExit, "ordered ordinal"):
            self.validate_curriculum()

    def test_malformed_enum_values_use_controlled_diagnostics(self):
        for key, value in (("kind", []), ("language", {}), ("sourceDigestScope", [])):
            self.manifest = copy.deepcopy(self.original)
            self.curriculum_lesson()["codeTabs"][0][key] = value
            with self.assertRaisesRegex(SystemExit, "must be a nonempty string"):
                self.validate_curriculum()
        self.manifest = copy.deepcopy(self.original)
        self.curriculum_lesson()["role"] = {}
        with self.assertRaisesRegex(SystemExit, "must be a nonempty string"):
            self.validate_curriculum()

    def test_new_aliases_require_selected_source_path_overlap_even_with_gap_text(self):
        for gap in (None, "A claimed gap must not authorize unrelated source entries."):
            self.manifest = copy.deepcopy(self.original)
            lesson = self.curriculum_lesson("gfx950-fp4-gemm-performance-lab")
            lesson["sourceEntryIds"] = ["typed-vecadd"]
            lesson["sourceBindingGap"] = gap
            with self.assertRaisesRegex(SystemExit, "no matching pinned kernel source path"):
                self.validate_curriculum()

    def test_selected_fixture_paths_support_aliases_without_package_widening(self):
        self.validate_curriculum()
        entry = next(entry for entry in self.manifest["entries"] if entry["lessonId"] == "gemm-tiling")
        self.assertNotIn("examples/gemm_autoresearch_v1/src/kernel.rs", entry["sourcePaths"])
        self.assertIn("gfx942-gemm-autoresearch", entry["compilerFixtureIds"])
        lesson = self.curriculum_lesson("gfx950-fp4-gemm-performance-lab")
        lesson["codeTabs"][0]["sourcePath"] = "examples/gfx950_low_precision/src/unselected.rs"
        with self.assertRaisesRegex(SystemExit, "no matching pinned kernel source path"):
            self.validate_curriculum()

    def test_existing_and_partially_matched_sources_report_exact_pending_paths(self):
        gaps = {}
        self.validator.validate_manifest(ROOT, self.manifest, curriculum_gaps=gaps)
        self.assertEqual(gaps, {
            "cpu-semantic-simulation": ["examples/workgroup_sync_v1/src/kernel_mixed_tile_u32.rs"],
            "gemm-proof-plan": ["examples/tiled_gemm_v1/src/kernel.rs"],
        })
        self.curriculum_lesson("cpu-semantic-simulation")["sourceBindingGap"] = None
        tab = self.curriculum_lesson("cpu-semantic-simulation")["codeTabs"][0]
        tab.update(sourceItem=None, sourceItemStatus="pending")
        with self.assertRaisesRegex(SystemExit, "sourceBindingGap.*production-ranked-bounds-device"):
            self.validate_curriculum()

    def test_unjustified_source_gap_claims_are_rejected(self):
        self.curriculum_lesson("gemm-autoresearch")["sourceBindingGap"] = "Not a real gap."
        with self.assertRaisesRegex(SystemExit, "fully linked paths"):
            self.validate_curriculum()

    def test_site_inventory_rejects_non_utf8_display_or_source_fragments(self):
        for key, value in (("displayedCode", "\ud800"), ("sourceFragments", ["\ud800"])):
            curriculum, inventory = self.site_inventory_fixture()
            inventory["lessons"][0]["codeTabs"][0][key] = value
            with self.assertRaisesRegex(SystemExit, "not valid UTF-8"):
                self.validator.validate_site_inventory(curriculum, inventory)

    def site_inventory_fixture(self):
        # Independent tiny rendered strings exercise hashing without retaining the full site.
        curriculum = copy.deepcopy(self.original["curriculum"])
        inventory = {"schema": self.validator.SITE_INVENTORY_SCHEMA, "site": copy.deepcopy(curriculum["site"]), "lessons": []}
        for lesson in curriculum["lessons"]:
            actual = {"id": lesson["lessonId"], "codeTabs": []}
            for tab in lesson["codeTabs"]:
                code = f"{lesson['lessonId']}:{tab['ordinal']}\n"
                tab["displayedUtf8Bytes"] = len(code.encode("utf-8"))
                tab["displayedSha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
                tab["sourceFragmentsSha256"] = None
                if tab["sourceItem"] is not None:
                    tab.update(sourceItem=None, sourceItemStatus="pending")
                    lesson["sourceBindingGap"] = "Synthetic inventory has no source-driver contract."
                projected = {key: tab[key] for key in self.validator.CURRICULUM_TAB_FIELDS if key != "sourceFragmentsSha256"}
                projected.update(displayedCode=code, sourceFragments=None)
                actual["codeTabs"].append(projected)
            inventory["lessons"].append(actual)
        return curriculum, inventory

    def test_site_only_commits_do_not_create_a_circular_pin_dependency(self):
        curriculum, inventory = self.site_inventory_fixture()
        inventory["site"].update(commit="a" * 40, tree="b" * 40)
        self.validator.validate_site_inventory(curriculum, inventory)

    def test_site_inventory_rejects_missing_new_or_reordered_lessons(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])), lambda rows: rows.reverse()):
            curriculum, inventory = self.site_inventory_fixture()
            mutate(inventory["lessons"])
            with self.assertRaisesRegex(SystemExit, "ordered lesson coverage"):
                self.validator.validate_site_inventory(curriculum, inventory)

    def test_site_inventory_rejects_missing_new_and_reordered_tabs(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])), lambda rows: rows.reverse()):
            curriculum, inventory = self.site_inventory_fixture()
            mutate(inventory["lessons"][0]["codeTabs"])
            with self.assertRaisesRegex(SystemExit, "code-tab coverage|source/display binding"):
                self.validator.validate_site_inventory(curriculum, inventory)

    def test_site_inventory_checks_actual_rendered_bytes_and_exact_metadata(self):
        for key, value, pattern in (
            ("displayedCode", "changed\n", "displayed bytes do not match"),
            ("displayedSha256", "0" * 64, "displayed bytes do not match"),
            ("label", "renamed", "source/display binding"),
            ("sourceCommit", "a" * 40, "source/display binding"),
            ("sourcePath", "examples/absent.rs", "source/display binding"),
            ("sourceFragments", ["new fragment"], "source/display binding"),
            ("ordinal", False, "source/display binding"),
        ):
            curriculum, inventory = self.site_inventory_fixture()
            inventory["lessons"][0]["codeTabs"][0][key] = value
            with self.assertRaisesRegex(SystemExit, pattern):
                self.validator.validate_site_inventory(curriculum, inventory)

    def test_scalar_gemm_adds_compile_and_pending_oracle_obligations(self):
        fixture = next(f for f in self.manifest["compilerFixtures"] if f["fixtureId"] == "gfx942-scalar-gemm")
        self.assertEqual(fixture["compilerInput"]["kernelSymbols"], ["scalar_gemm_v1"])
        simulation = fixture["simulation"]
        self.assertEqual(simulation["status"], "pending-reconciliation")
        self.assertEqual((simulation["bundleVersion"], simulation["canonicalKirVersion"]), (7, 12))
        for kind in ("request", "expectation"):
            with self.subTest(kind=kind):
                payload = (ROOT / simulation[f"{kind}Path"]).read_bytes()
                self.assertEqual(simulation[f"{kind}Bytes"], len(payload))
                self.assertEqual(simulation[f"{kind}Sha256"], hashlib.sha256(payload).hexdigest())
        suite = next(row for row in self.manifest["qualification"]["suites"]
                     if row["suiteId"] == "semantic-simulation-scalar-gemm")
        self.assertEqual(suite["availability"], "pending")
        for lesson in ("gemm-proof-plan", "gemm-tiling"):
            entry = next(e for e in self.manifest["entries"] if e["lessonId"] == lesson)
            self.assertIn(fixture["fixtureId"], entry["compilerFixtureIds"])

    def test_six_added_gfx942_library_symbol_and_feature_contracts(self):
        expected = {
            "gfx942-fill-simulation": ("fill", "fill", "fe2o3_fill", "fill", True, []),
            "gfx942-moe-top2": (
                "moe-top2", "moe_top2_v1", "fe2o3_moe_top2_v1",
                "moe_top2_route_f32_t8_e4_k2_c4_v1", True, [],
            ),
            "gfx942-scalar-gemm": (
                "scalar-gemm", "scalar_gemm_v1", "fe2o3_scalar_gemm_v1",
                "scalar_gemm_v1", True, [],
            ),
            "gfx942-typed-vecadd-source": (
                "typed-vecadd", "vecadd", "fe2o3_vecadd", "vecadd", True, [],
            ),
            "gfx942-wave64-collectives": (
                "wave64-collectives", "wave64_collectives_v1", "fe2o3_wave64_collectives_v1",
                "wave64_collectives_v1", True, [],
            ),
            "gfx942-workgroup-collectives": (
                "workgroup-collectives", "workgroup_sync_v1", "fe2o3_workgroup_sync_v1",
                "lds_publish_read_reduce_i32_v1", False, ["lds-kernel"],
            ),
        }
        fixtures = {fixture["fixtureId"]: fixture for fixture in self.manifest["compilerFixtures"]}
        for fixture_id, (case, package, crate, symbol, defaults, features) in expected.items():
            with self.subTest(fixture_id=fixture_id):
                fixture = fixtures[fixture_id]
                self.assertEqual(fixture["target"], "gfx942")
                self.assertEqual(fixture["testId"], f"kernel-compile-matrix/gfx942/{case}")
                self.assertEqual(fixture["matrix"], {
                    "caseId": case,
                    "runnerPath": f"examples/{package}/run-gfx942.sh",
                    "artifactName": f"{package}.hsaco",
                    "runnerArguments": [],
                    "environment": [],
                })
                inputs = fixture["compilerInput"]
                self.assertEqual(inputs["packageManifest"], f"examples/{package}/Cargo.toml")
                self.assertEqual(inputs["cargoTarget"], {
                    "kind": "lib", "name": crate, "sourcePath": "src/lib.rs",
                })
                self.assertEqual(inputs["kernelSymbols"], [symbol])
                self.assertEqual(inputs["defaultFeatures"], defaults)
                self.assertEqual(inputs["features"], features)


    def test_matrix_records_are_stable_and_do_not_execute(self):
        for target, count in (("gfx942", 11), ("gfx950", 84)):
            result = subprocess.run(
                [sys.executable, str(CHECKER), "--emit-matrix", target],
                check=True, text=True, capture_output=True,
            )
            records = [line.split("|") for line in result.stdout.splitlines()]
            self.assertEqual(len(records), count)
            self.assertTrue(all(len(record) == 6 for record in records))
            self.assertEqual([r[0] for r in records], sorted(r[0] for r in records))
            expected = sorted(
                (f for f in self.manifest["compilerFixtures"] if f["target"] == target),
                key=lambda f: f["matrix"]["caseId"],
            )
            self.assertEqual(result.stdout.splitlines(), [self.validator.matrix_record(f) for f in expected])

    def test_require_qualified_always_fails_closed(self):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--require-qualified"],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("qualification receipts and policy/final-graph evidence are not implemented", result.stderr)

    def test_kernel_pair_cli_emits_only_incomplete_obligations(self):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--emit-kernel-pairs"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema"], "fe2o3-tutorial-kernel-pair-obligations-v2")
        identities = report["kernelInventory"]
        for counts in (report, identities):
            self.assertEqual(counts["knownVariantObligationCount"], 203)
            self.assertEqual(counts["pendingVariantCount"], 127)
            self.assertEqual(counts["unregisteredDisplayItemCount"], 6)
        self.assertIs(identities["runtimeCensusValidated"], False)
        self.assertEqual(identities["knownKernelIdentityCount"], 101)
        self.assertEqual(identities["negativeCaseCount"], 3)
        self.assertTrue(identities["unresolvedBindings"])
        self.assertIs(report["qualified"], False)
        self.assertIs(report["inventoryComplete"], False)
        self.assertIsNone(report["requiredPairCount"])
        self.assertEqual(report["qualifiedPairCount"], 0)
        self.assertEqual(report["requiredModes"], ["simt", "tile"])
        self.assertEqual(len(report["fixtureSelections"]), 95)
        self.assertEqual(len(report["sourceDriverCases"]), 14)
        self.assertEqual(len(report["displayObservations"]), 58)
        self.assertEqual(sum(row["sourceItemStatus"] == "pending"
                             for row in report["displayObservations"]), 56)
        self.assertTrue(all(row["lexicalKernelNames"] is None
                            for row in report["displayObservations"]))
        self.assertEqual(report["stageStatus"], "not-evaluated")
        self.assertEqual(report["variantBindingStatus"], "partial")
        self.assertEqual(report["sourceBoundVariantCount"], 76)
        self.assertEqual(report["sourceBoundPairCount"], 31)
        self.assertEqual(report["productionContract"], self.original["productionContract"])

    def kernel_pair_report(self, inventory=None):
        gaps = {}
        fixtures = self.validator.validate_manifest(ROOT, self.manifest, curriculum_gaps=gaps)
        if inventory is not None:
            self.validator.validate_site_inventory(self.manifest["curriculum"], inventory)
        return self.validator._kernel_pair_report(self.manifest, fixtures, gaps, inventory)

    def ordinary_source_report(self, raw, passed=True):
        cases = []
        for fixture in self.manifest["compilerFixtures"]:
            inputs = copy.deepcopy(fixture["compilerInput"])
            inputs.pop("contractSha256")
            cases.append({
                "fixture": {"fixtureId": fixture["fixtureId"], "target": fixture["target"],
                            "compilerInput": inputs},
                "status": "checked-output-pass" if passed else "blocked",
                "observation": {"roots": inputs["kernelSymbols"]} if passed else None,
                "refusal": None if passed else {"stage": "manifest", "detail": "synthetic refusal"},
                "compiler_artifacts": [],
            })
        return {
            "schema": "fe2o3-ordinary-source-policy4-extraction-corpus-v1",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "configurations": len(cases),
            "distinct_expected_roots": len({symbol for case in cases
                                           for symbol in case["fixture"]["compilerInput"]["kernelSymbols"]}),
            "all_checked_output_passed": passed, "default_pipeline_activated": False,
            "grants_artifact_or_launch_authority": False, "cases": cases,
        }

    def test_ordinary_source_mutations_reject_without_changing_projection(self):
        original = self.ordinary_source_report(json.dumps(self.manifest).encode())
        baseline = self.kernel_pair_report()
        cases = original["cases"]
        fixtures = {fixture["fixtureId"]: fixture for fixture in self.manifest["compilerFixtures"]}
        mutations = [
            (("schema",), "unknown"), (("manifest_sha256",), "0" * 64),
            (("configurations",), True), (("distinct_expected_roots",), 0),
            (("default_pipeline_activated",), 0), (("grants_artifact_or_launch_authority",), True),
            (("all_checked_output_passed",), False), (("all_checked_output_passed",), 1),
            (("cases",), cases[:-1]), (("cases",), cases + [cases[0]]),
            (("cases",), cases[:-1] + [cases[0]]),
            (("cases", 0, "fixture", "fixtureId"), "absent"),
            (("cases", 0, "fixture", "target"), "gfx950" if cases[0]["fixture"]["target"] == "gfx942" else "gfx942"),
            (("cases", 0, "fixture", "compilerInput", "contractSha256"), "0" * 64),
        ]
        for key, value in cases[0]["fixture"]["compilerInput"].items():
            changed = (not value if type(value) is bool else value + ["changed"] if isinstance(value, list)
                       else dict(value, name="changed") if isinstance(value, dict) else "changed")
            mutations.append((("cases", 0, "fixture", "compilerInput", key), changed))
        for key in ("status", "observation", "refusal", "compiler_artifacts"):
            mutations.append((("cases", 0), {k: v for k, v in cases[0].items() if k != key}))
        mutations.extend((("cases", 0, key), value) for key, value in (
            ("observation", None), ("refusal", {}), ("compiler_artifacts", ["unexpected.o"]),
            ("compiler_artifacts", [None]), ("status", "unknown"), ("callback_progress", []),
        ))
        for path, value in mutations:
            with self.subTest(path=path, value=value):
                report, projection = copy.deepcopy(original), copy.deepcopy(baseline)
                parent = report
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = copy.deepcopy(value)
                with self.assertRaises(SystemExit):
                    self.validator._bind_ordinary_source_report(projection, fixtures, report, original["manifest_sha256"])
                self.assertEqual(projection, baseline)

    def test_ordinary_source_cli_preserves_diagnostics_without_qualification(self):
        raw = b"\n" + json.dumps(self.manifest).encode() + b"\n"
        baseline = self.kernel_pair_report()
        with tempfile.TemporaryDirectory() as temporary:
            manifest, path = Path(temporary) / "manifest.json", Path(temporary) / "report.json"
            for mode in ("canonical", "changed-bytes", "reordered", "partial", "panic", "replay", "blocked", "allpass"):
                with self.subTest(mode=mode):
                    manifest.write_bytes(raw + (b" " if mode == "changed-bytes" else b""))
                    report = self.ordinary_source_report(raw, passed=mode != "blocked")
                    if mode == "canonical":
                        report["manifest_sha256"] = baseline["sourceContractSha256"]
                    if mode == "reordered":
                        report["cases"].reverse()
                    if mode in ("partial", "panic", "replay"):
                        report["cases"][0].update(
                            callback_progress={"outcome": "panicked" if mode == "panic" else None,
                                               "active": {"stage": "policy4", "policy4": {"unavailable": "not observed"}},
                                               "phases": [{"stage": "source-collection", "outcome": "complete"}]},
                            callback_progress_error="synthetic diagnostic error",
                        )
                        if mode != "replay":
                            report["all_checked_output_passed"] = False
                            report["cases"][0].update(
                                status="blocked", refusal={"stage": "policy4", "detail": "synthetic refusal"},
                            )
                    path.write_text(json.dumps(report), encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, "-I", "-B", str(CHECKER), "--manifest", str(manifest),
                         "--emit-kernel-pairs", "--ordinary-source-report", str(path)],
                        text=True, capture_output=True,
                    )
                    if mode in ("canonical", "changed-bytes"):
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(result.stdout, "")
                        continue
                    self.assertEqual(result.returncode, 0, result.stderr)
                    projection = json.loads(result.stdout)
                    self.assertEqual(projection.pop("ordinarySourceObservations"), {"diagnosticOnly": True, "report": report})
                    self.assertEqual(projection["stageStatus"], "fixture-source-observations-bound")
                    projection["stageStatus"] = baseline["stageStatus"]
                    for row in projection["fixtureSelections"]:
                        self.assertEqual(report["cases"][row.pop("ordinarySourceCaseIndex")]["fixture"]["fixtureId"], row["fixtureId"])
                    self.assertEqual(projection, baseline)

    def test_ordinary_source_report_requires_projection_and_cannot_qualify(self):
        for flags in ([], ["--emit-matrix", "gfx942"], ["--emit-kernel-pairs", "--require-qualified"]):
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(CHECKER), *flags, "--ordinary-source-report", "absent.json"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("requires --emit-kernel-pairs" if not flags or flags[0] == "--emit-matrix"
                          else "qualification receipts", result.stderr)

    def test_manifest_raw_digest_and_read_bounds(self):
        raw = b"{}\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            path, link = Path(temporary) / "report.json", Path(temporary) / "link.json"
            path.write_bytes(raw)
            link.symlink_to(path)
            self.assertEqual(self.validator.load_manifest(path, len(raw), with_sha256=True),
                             ({}, hashlib.sha256(raw).hexdigest()))
            for candidate, limit in ((path, len(raw) - 1), (link, 64), (Path(temporary), 64), (Path(temporary) / "absent", 64)):
                with self.subTest(candidate=candidate), self.assertRaises(SystemExit):
                    self.validator.load_manifest(candidate, limit, with_sha256=True)
            with mock.patch.object(Path, "open", mock.mock_open(read_data=raw + b" ")) as opened:
                with self.assertRaises(SystemExit):
                    self.validator.load_manifest(path, len(raw), with_sha256=True)
                opened.return_value.read.assert_called_once_with(len(raw) + 1)
            for invalid in (b'{"a":0,"a":1}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e400}'):
                path.write_bytes(invalid)
                with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                    self.validator.load_manifest(path, with_sha256=True)

    def test_kernel_pair_legacy_report_remains_v1_without_identity_extension(self):
        self.manifest.pop("kernelInventory")
        report = self.kernel_pair_report()
        self.assertEqual(report["schema"], "fe2o3-tutorial-kernel-pair-obligations-v1")
        self.assertNotIn("kernelInventory", report)
        for key in ("knownVariantObligationCount", "pendingVariantCount", "unregisteredDisplayItemCount"):
            self.assertNotIn(key, report)
        self.assertIs(report["inventoryComplete"], False)
        self.assertIsNone(report["requiredPairCount"])

    def test_kernel_identity_snapshot_retains_all_declared_roles_and_gaps(self):
        inventory = self.manifest["kernelInventory"]
        payload = json.dumps(
            inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                          "222be9a48168d8b0db745b782798e4a3e1fa2a5a173d05b01816e5a79e6bd15e")
        self.assertEqual(len(inventory["kernels"]), 101)
        self.assertEqual(Counter(row["classification"] for row in inventory["displayItems"]),
                         {"kernel": 81, "required-negative": 3, "conceptual": 26, "helper": 45})
        self.assertEqual(Counter(row["bindingStatus"] for row in inventory["displayItems"]),
                         {"pending": 8, "source-driver-contract": 14, "fixture-source-contract": 62,
                          "not-applicable": 71})
        self.assertEqual([row["caseOrdinal"] for row in inventory["negativeCases"]], [6, 7, 8])
        bound = [(row["kernelId"], variant["kind"])
                 for row in inventory["kernels"] for variant in row["variants"]
                 if variant["status"] == "source-bound"]
        self.assertEqual(bound, [('fixture:gfx942-fill-simulation:fill', 'simt'),
         ('fixture:gfx942-moe-top2:moe_top2_route_f32_t8_e4_k2_c4_v1', 'simt'),
         ('fixture:gfx942-scalar-gemm:scalar_gemm_v1', 'simt'),
         ('fixture:gfx942-typed-vecadd-source:vecadd', 'simt'),
         ('fixture:gfx942-wave64-collectives:wave64_collectives_v1', 'simt'),
         ('fixture:gfx942-workgroup-collectives:lds_publish_read_reduce_i32_v1', 'simt'),
         ('fixture:gfx950-attnres-aggregate-explicit-reuse:gfx950_attnres_aggregate', 'simt'),
         ('fixture:gfx950-attnres-aggregate:gfx950_attnres_aggregate', 'simt'),
         ('fixture:gfx950-attnres-aggregate:gfx950_attnres_aggregate', 'tile'),
         ('fixture:gfx950-combine-expert-ranks:gfx950_combine_expert_ranks_v1', 'simt'),
         ('fixture:gfx950-combine-expert-ranks:gfx950_combine_expert_ranks_v1', 'tile'),
         ('fixture:gfx950-compressed-hybrid-attention-division-baseline:gfx950_compressed_hybrid_attention',
          'simt'),
         ('fixture:gfx950-compressed-hybrid-attention:gfx950_compressed_hybrid_attention', 'simt'),
         ('fixture:gfx950-compressed-hybrid-attention:gfx950_compressed_hybrid_attention', 'tile'),
         ('fixture:gfx950-content-sparse-attention-reciprocal-reuse:gfx950_content_sparse_attention',
          'simt'),
         ('fixture:gfx950-content-sparse-attention:gfx950_content_sparse_attention', 'simt'),
         ('fixture:gfx950-content-sparse-attention:gfx950_content_sparse_attention', 'tile'),
         ('fixture:gfx950-deepseek-sparse-attention:gfx950_deepseek_sparse_attention', 'simt'),
         ('fixture:gfx950-deepseek-sparse-attention:gfx950_deepseek_sparse_attention', 'tile'),
         ('fixture:gfx950-four-branch-residual-explicit:gfx950_four_branch_residual', 'simt'),
         ('fixture:gfx950-four-branch-residual:gfx950_four_branch_residual', 'simt'),
         ('fixture:gfx950-four-branch-residual:gfx950_four_branch_residual', 'tile'),
         ('fixture:gfx950-gpt-oss-decode:gfx950_gpt_oss_120b_decode_megakernel_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-decode:gfx950_gpt_oss_120b_decode_megakernel_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-held-fragments:gfx950_gpt_oss_120b_decode_megakernel_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-held-fragments:gfx950_gpt_oss_120b_decode_megakernel_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-interleaved-stores:gfx950_gpt_oss_120b_decode_megakernel_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-interleaved-stores:gfx950_gpt_oss_120b_decode_megakernel_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-materialized-attention:gfx950_gpt_oss_120b_attention_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-materialized-attention:gfx950_gpt_oss_120b_attention_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-materialized-expert:gfx950_gpt_oss_120b_expert_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-materialized-expert:gfx950_gpt_oss_120b_expert_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-materialized-router:gfx950_gpt_oss_120b_router_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-materialized-router:gfx950_gpt_oss_120b_router_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-serial-router:gfx950_gpt_oss_120b_decode_megakernel_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-serial-router:gfx950_gpt_oss_120b_decode_megakernel_v1', 'tile'),
         ('fixture:gfx950-kda-decode-baseline:gfx950_kda_decode', 'simt'),
         ('fixture:gfx950-kda-decode:gfx950_kda_decode', 'simt'),
         ('fixture:gfx950-kda-decode:gfx950_kda_decode', 'tile'),
         ('fixture:gfx950-kda-prefill-baseline:gfx950_kda_chunkwise_prefill', 'simt'),
         ('fixture:gfx950-kda-prefill:gfx950_kda_chunkwise_prefill', 'simt'),
         ('fixture:gfx950-kda-prefill:gfx950_kda_chunkwise_prefill', 'tile'),
         ('fixture:gfx950-mhc-sinkhorn-mix-scalar:gfx950_mhc_sinkhorn_mix', 'simt'),
         ('fixture:gfx950-mhc-sinkhorn-mix:gfx950_mhc_sinkhorn_mix', 'simt'),
         ('fixture:gfx950-mhc-sinkhorn-mix:gfx950_mhc_sinkhorn_mix', 'tile'),
         ('fixture:gfx950-moe-expert-rank:gfx950_moe_expert_rank_fp4_fp8_v1', 'simt'),
         ('fixture:gfx950-moe-expert-rank:gfx950_moe_expert_rank_fp4_fp8_v1', 'tile'),
         ('fixture:gfx950-moe-route:gfx950_moe_route_fp4_t16_e4_k2_v1', 'simt'),
         ('fixture:gfx950-moe-route:gfx950_moe_route_fp4_t16_e4_k2_v1', 'tile'),
         ('fixture:gfx950-muon-update:gfx950_muon_update_4x4_v1', 'simt'),
         ('fixture:gfx950-muon-update:gfx950_muon_update_4x4_v1', 'tile'),
         ('fixture:gfx950-qwen-ngram-gather:gfx950_qwen_ngram_gather_v1', 'simt'),
         ('fixture:gfx950-qwen-ngram-gather:gfx950_qwen_ngram_gather_v1', 'tile'),
         ('fixture:gfx950-speculative-transaction:gfx950_speculative_transaction_v1', 'simt'),
         ('fixture:gfx950-speculative-transaction:gfx950_speculative_transaction_v1', 'tile'),
         ('fixture:gfx950-stage-gradient-shard:gfx950_stage_gradient_shard_v1', 'simt'),
         ('fixture:gfx950-stage-gradient-shard:gfx950_stage_gradient_shard_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-pipelined-attention:gfx950_gpt_oss_120b_decode_megakernel_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-pipelined-attention:gfx950_gpt_oss_120b_decode_megakernel_v1', 'tile'),
         ('fixture:gfx950-gpt-oss-scalar-attention:gfx950_gpt_oss_120b_decode_megakernel_v1', 'simt'),
         ('fixture:gfx950-gpt-oss-scalar-attention:gfx950_gpt_oss_120b_decode_megakernel_v1', 'tile'),
         ('source-driver:cpu-semantic-simulation:6:row_affine_sum_u32_v1', 'simt'),
         ('fixture:gfx950-moe-expert-rank-simt:gfx950_moe_expert_rank_fp4_fp8_v1', 'simt'),
         ('fixture:gfx950-moe-expert-rank-simt:gfx950_moe_expert_rank_fp4_fp8_v1', 'tile'),
         ('fixture:gfx950-moe-route-tile:gfx950_moe_route_fp4_t16_e4_k2_v1', 'simt'),
         ('fixture:gfx950-moe-route-tile:gfx950_moe_route_fp4_t16_e4_k2_v1', 'tile'),
         ('fixture:gfx950-combine-expert-ranks-tile:gfx950_combine_expert_ranks_v1', 'simt'),
         ('fixture:gfx950-combine-expert-ranks-tile:gfx950_combine_expert_ranks_v1', 'tile'),
         ('fixture:gfx950-speculative-transaction-tile:gfx950_speculative_transaction_v1', 'simt'),
         ('fixture:gfx950-speculative-transaction-tile:gfx950_speculative_transaction_v1', 'tile'),
         ('fixture:gfx950-qwen-ngram-gather-tile:gfx950_qwen_ngram_gather_v1', 'simt'),
         ('fixture:gfx950-qwen-ngram-gather-tile:gfx950_qwen_ngram_gather_v1', 'tile'),
         ('fixture:gfx950-stage-gradient-shard-tile:gfx950_stage_gradient_shard_v1', 'simt'),
         ('fixture:gfx950-stage-gradient-shard-tile:gfx950_stage_gradient_shard_v1', 'tile'),
         ('fixture:gfx950-muon-update-tile:gfx950_muon_update_4x4_v1', 'simt'),
         ('fixture:gfx950-muon-update-tile:gfx950_muon_update_4x4_v1', 'tile')])
        pending = [variant for row in inventory["kernels"] for variant in row["variants"]
                   if variant["status"] == "pending"]
        self.assertEqual(len(pending), 127)
        self.assertTrue(all(variant["source"] is None for variant in pending))
        display = {(row["lessonId"], row["tabOrdinal"], row["kernelSymbol"]): row
                   for row in inventory["displayItems"]}
        fill = display[("first-fill", 0, "fill")]
        self.assertEqual(fill["kernelIds"], ["fixture:gfx942-fill-simulation:fill"])
        self.assertEqual(fill["bindingStatus"], "fixture-source-contract")
        self.assertEqual(display[("typed-vecadd", 3, "vecadd")]["bindingStatus"], "pending")
        for row in inventory["displayItems"]:
            if row["lessonId"] == "typed-vecadd" and row["classification"] == "kernel" and row["tabOrdinal"] < 5:
                self.assertEqual(row["kernelIds"], [])

    def test_attention_sources_have_distinct_required_pending_obligations(self):
        fixtures = self.validator.validate_manifest(ROOT, self.manifest)
        lesson_id = "gfx950-gpt-oss-120b-megakernel"
        symbol = "gfx950_gpt_oss_120b_decode_megakernel_v1"
        entry = next(row for row in self.manifest["entries"] if row["lessonId"] == lesson_id)
        cpu = next(row for row in self.manifest["qualification"]["suites"]
                   if row["suiteId"] == "cpu-reference-gfx950-gpt-oss-decode")
        identities = []
        for variant, tab_ordinal in (("pipelined-attention", 5), ("scalar-attention", 6)):
            fixture_id = f"gfx950-gpt-oss-{variant}"
            fixture = fixtures[fixture_id]
            with self.subTest(variant=variant):
                selected = fixture["compilerInput"]
                self.assertEqual(selected["features"], [f"kernel-gpt-oss-decode-{variant}"])
                self.assertIs(selected["defaultFeatures"], False)
                self.assertEqual(selected["kernelSymbols"], [symbol])
                self.assertEqual(selected["sourcePaths"], [
                    f"examples/gfx950_gpt_oss_decode/src/kernel_{variant.replace('-', '_')}.rs",
                ])
                self.assertEqual(fixture["matrix"]["runnerArguments"], [variant])
                self.assertEqual(fixture["matrix"]["artifactName"], f"kernel-gpt-oss-decode-{variant}.hsaco")
                self.assertIn(fixture_id, entry["compilerFixtureIds"])
                self.assertIn("production-compile", entry["requiredGates"])
                self.assertIn(fixture_id, cpu["coverage"][0]["fixtureIds"])
                self.assertEqual(cpu["availability"], "pending")
                simulation = fixture["simulation"]
                self.assertEqual(simulation["status"], "pending-design")
                for kind in ("request", "expectation"):
                    self.assertEqual(simulation[f"{kind}Path"],
                                     f"config/tutorial-simulation-v1/{fixture_id}.{kind}.json")
                    self.assertIsNone(simulation[f"{kind}Sha256"])
                    self.assertIsNone(simulation[f"{kind}Bytes"])
                suite = next(row for row in self.manifest["qualification"]["suites"]
                             if row["suiteId"] == f"semantic-simulation-{fixture_id}")
                self.assertEqual(suite["availability"], "pending")
                self.assertEqual(suite["command"]["arguments"], ["--fixture-id", fixture_id])
                self.assertEqual(suite["coverage"], [{"lessonId": lesson_id, "fixtureIds": [fixture_id]}])
                display = next(row for row in self.manifest["kernelInventory"]["displayItems"]
                               if row["lessonId"] == lesson_id and row["tabOrdinal"] == tab_ordinal
                               and row["classification"] == "kernel")
                identity = f"fixture:{fixture_id}:{symbol}"
                self.assertEqual(display["kernelIds"], [identity])
                self.assertEqual(display["bindingStatus"], "fixture-source-contract")
                identities.append(identity)
        self.assertEqual(len(set(identities)), 2)
        report = self.kernel_pair_report()
        self.assertIs(report["qualified"], False)
        self.assertEqual(report["qualifiedPairCount"], 0)
        self.assertIsNone(report["requiredPairCount"])
        self.assertEqual(report["variantBindingStatus"], "partial")

    def test_attention_bindings_reject_same_symbol_source_substitution(self):
        lesson_id = "gfx950-gpt-oss-120b-megakernel"
        # Synthetic current-source baseline for validator hostility only. This
        # does not change or claim to refresh the pinned historical website.
        baseline = copy.deepcopy(self.original)
        lesson = next(row for row in baseline["curriculum"]["lessons"] if row["lessonId"] == lesson_id)
        tab = next(row for row in lesson["codeTabs"] if row["ordinal"] == 5)
        source = (ROOT / tab["sourcePath"]).read_text()
        encoded = source.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        tab.update(displayedUtf8Bytes=len(encoded), displayedSha256=digest, sourceSha256=digest)
        display = next(row for row in baseline["kernelInventory"]["displayItems"]
                       if row["lessonId"] == lesson_id and row["tabOrdinal"] == 5)
        occurrences = [row for row in self.validator.ordinary_rust_function_items(source)
                       if row["attributedKernel"] and row["kernelSymbol"] == display["kernelSymbol"]]
        self.assertEqual(len(occurrences), 1)
        display.update(bindingStatus="fixture-source-contract",
                       functionUtf8Offset=occurrences[0]["functionUtf8Offset"])
        self.assertIs(self.validator.validate_kernel_inventory(baseline, None, repo_root=ROOT)["inventoryComplete"], False)
        rows = [row for row in baseline["kernelInventory"]["displayItems"]
                if row["lessonId"] == lesson_id and row["tabOrdinal"] in (5, 6)
                and row["classification"] == "kernel"]
        self.assertEqual(len(rows), 2)
        for index, variant in enumerate(("pipelined-attention", "scalar-attention")):
            document = copy.deepcopy(baseline)
            display = next(row for row in document["kernelInventory"]["displayItems"]
                           if row["lessonId"] == lesson_id and row["tabOrdinal"] == 5 + index
                           and row["classification"] == "kernel")
            display["kernelIds"] = copy.deepcopy(rows[1 - index]["kernelIds"])
            with self.subTest(variant=variant, mutation="identity"), self.assertRaisesRegex(
                SystemExit, "exact selected source",
            ):
                self.validator.validate_kernel_inventory(document, None, repo_root=ROOT)
            document = copy.deepcopy(baseline)
            fixture = next(row for row in document["compilerFixtures"]
                           if row["fixtureId"] == f"gfx950-gpt-oss-{variant}")
            other = "scalar-attention" if index == 0 else "pipelined-attention"
            fixture["compilerInput"]["features"] = [f"kernel-gpt-oss-decode-{other}"]
            fixture["compilerInput"]["contractSha256"] = self.validator.fixture_input_contract_sha256(fixture)
            with self.subTest(variant=variant, mutation="feature"), self.assertRaisesRegex(
                SystemExit, "exact current source occurrence",
            ):
                self.validator.validate_kernel_inventory(document, None, repo_root=ROOT)

    def test_attention_historical_lds_snapshot_cannot_regain_exact_binding(self):
        lesson_id = "gfx950-gpt-oss-120b-megakernel"
        lesson = next(row for row in self.manifest["curriculum"]["lessons"] if row["lessonId"] == lesson_id)
        tab = next(row for row in lesson["codeTabs"] if row["ordinal"] == 5)
        display = next(row for row in self.manifest["kernelInventory"]["displayItems"]
                       if row["lessonId"] == lesson_id and row["tabOrdinal"] == 5)
        self.assertEqual(display["bindingStatus"], "fixture-source-contract")
        # Reintroduce the retained historical snapshot only in this negative
        # fixture; the live tab now displays the corrected current source.
        display.update(bindingStatus="pending", functionUtf8Offset=1508)
        tab.update(displayedUtf8Bytes=22863,
                   displayedSha256="faf9bc589658b1381e1042c71b952881488261cc85ff86c21dbb7f1f1a83a460",
                   sourceSha256="faf9bc589658b1381e1042c71b952881488261cc85ff86c21dbb7f1f1a83a460")
        self.assertEqual(tab["sourceSha256"], tab["displayedSha256"])
        source = (ROOT / tab["sourcePath"]).read_bytes()
        self.assertNotEqual(hashlib.sha256(source).hexdigest(), tab["sourceSha256"])
        before = copy.deepcopy(self.manifest["curriculum"])
        report = self.validator.validate_kernel_inventory(self.manifest, None, repo_root=ROOT)
        self.assertIs(report["inventoryComplete"], False)
        self.assertIs(report["runtimeCensusValidated"], False)
        self.assertIsNone(report["requiredPairCount"])
        self.assertTrue(any(row.get("lessonId") == lesson_id and row.get("tabOrdinal") == 5
                            for row in report["unresolvedBindings"]))
        display["bindingStatus"] = "fixture-source-contract"
        with self.assertRaisesRegex(SystemExit, "exact current source occurrence"):
            self.validator.validate_kernel_inventory(self.manifest, None, repo_root=ROOT)
        self.assertEqual(self.manifest["curriculum"], before)

    def gpt_fixture_document(self):
        """A six-binding component projection, not the live website census."""
        document = copy.deepcopy(self.original)
        rows = [row for row in document["kernelInventory"]["displayItems"]
                if row["lessonId"] == "gfx950-gpt-oss-120b-megakernel"
                and row["bindingStatus"] == "fixture-source-contract" and 1 <= row["tabOrdinal"] <= 4]
        self.assertEqual(len(rows), 6)
        ids = {identity for row in rows for identity in row["kernelIds"]}
        kernels = [row for row in document["kernelInventory"]["kernels"] if row["kernelId"] in ids]
        # This isolated display fixture intentionally excludes the paired
        # implementations, so it cannot retain their cross-kernel bindings.
        for kernel in kernels:
            for variant in kernel["variants"]:
                variant.update(status="pending", source=None)
        fixtures = {ref["fixtureId"] for row in kernels for ref in row["selections"]}
        document["compilerFixtures"] = [row for row in document["compilerFixtures"] if row["fixtureId"] in fixtures]
        lesson = next(row for row in document["curriculum"]["lessons"]
                      if row["lessonId"] == "gfx950-gpt-oss-120b-megakernel")
        lesson["codeTabs"] = lesson["codeTabs"][1:5]
        for index, tab in enumerate(lesson["codeTabs"]):
            tab["ordinal"] = index
        for row in rows:
            row["tabOrdinal"] -= 1
        document["curriculum"]["lessons"] = [lesson]
        document["kernelInventory"].update(kernels=kernels, negativeCases=[], displayItems=rows)
        runtime = {"schema": self.validator.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": [{
                       "id": lesson["lessonId"], "codeTabs": [
                           {**tab, "displayedCode": (ROOT / tab["sourcePath"]).read_bytes().decode("utf-8"),
                            "sourceFragments": None} for tab in lesson["codeTabs"]]}]}
        return document, runtime

    def test_six_real_fixture_sources_bind_exact_display_occurrences(self):
        document, runtime = self.gpt_fixture_document()
        self.validator.validate_site_inventory(document["curriculum"], runtime)
        result = self.validator.validate_kernel_inventory(document, runtime)
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(result["displayItemCount"], 6)
        self.assertTrue(all(variant["status"] == "pending" and variant["source"] is None
                            for kernel in result["kernelIdentities"] for variant in kernel["variants"]))
        without_runtime = self.validator.validate_kernel_inventory(document, None)
        self.assertFalse(without_runtime["inventoryComplete"])
        self.assertIsNone(without_runtime["requiredPairCount"])
        self.assertEqual(sum("selection" in row for row in without_runtime["unresolvedBindings"]), 6)

    def gpt_excerpt_document(self):
        """A two-display source association, not a complete curriculum census."""
        document = copy.deepcopy(self.original)
        lessons = {"gfx950-gpt-oss-120b-megakernel", "gfx950-gpt-oss-megakernel-performance-lab"}
        fixture_id = "gfx950-gpt-oss-decode"
        kernel_id = f"fixture:{fixture_id}:gfx950_gpt_oss_120b_decode_megakernel_v1"
        document["compilerFixtures"] = [row for row in document["compilerFixtures"]
                                        if row["fixtureId"] == fixture_id]
        inventory = document["kernelInventory"]
        inventory["kernels"] = [row for row in inventory["kernels"] if row["kernelId"] == kernel_id]
        for kernel in inventory["kernels"]:
            for variant in kernel["variants"]:
                variant.update(status="pending", source=None)
        inventory["negativeCases"] = []
        inventory["displayItems"] = [row for row in inventory["displayItems"]
                                     if row["lessonId"] in lessons and row["tabOrdinal"] == 0]
        self.assertEqual(len(inventory["displayItems"]), 2)
        document["curriculum"]["lessons"] = [row for row in document["curriculum"]["lessons"]
                                             if row["lessonId"] in lessons]
        runtime = {"schema": self.validator.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": []}
        for lesson in document["curriculum"]["lessons"]:
            lesson["codeTabs"] = lesson["codeTabs"][:1]
            tab = lesson["codeTabs"][0]
            physical = (ROOT / tab["sourcePath"]).read_bytes()
            excerpt = physical[1338:1338 + tab["displayedUtf8Bytes"]].decode("utf-8")
            runtime["lessons"].append({"id": lesson["lessonId"], "codeTabs": [
                {**tab, "displayedCode": excerpt, "sourceFragments": [excerpt]},
            ]})
        return document, runtime

    def test_checked_in_gpt_excerpts_bind_without_qualifying_variants(self):
        document, runtime = self.gpt_excerpt_document()
        before = copy.deepcopy((document, runtime))
        self.validator.validate_site_inventory(document["curriculum"], runtime)
        result = self.validator.validate_kernel_inventory(document, runtime, repo_root=ROOT)
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(result["displayItemCount"], 2)
        self.assertEqual(result["sourceBoundVariantCount"], 0)
        self.assertEqual(result["sourceBoundPairCount"], 0)
        missing = self.validator.validate_kernel_inventory(document, None, repo_root=ROOT)
        self.assertFalse(missing["runtimeCensusValidated"])
        self.assertFalse(missing["inventoryComplete"])
        self.assertIsNone(missing["requiredPairCount"])
        self.assertEqual((document, runtime), before)

    def test_checked_in_gpt_excerpt_rejects_stale_coordinates_and_bytes(self):
        document, runtime = self.gpt_excerpt_document()
        document["kernelInventory"]["displayItems"][0]["functionUtf8Offset"] += 1
        with self.assertRaisesRegex(SystemExit, "missing from the live function census"):
            self.validator.validate_kernel_inventory(document, runtime, repo_root=ROOT)
        document, runtime = self.gpt_excerpt_document()
        tab = runtime["lessons"][0]["codeTabs"][0]
        tab["displayedCode"] += " "
        with self.assertRaisesRegex(SystemExit, "displayed bytes do not match"):
            self.validator.validate_site_inventory(document["curriculum"], runtime)

    def bind_source_variant(self, document, root, kernel_id):
        inventory = self.validator.validate_kernel_inventory(document, None, repo_root=root)
        retained = next(row for row in inventory["kernelIdentities"] if row["kernelId"] == kernel_id)
        kernel = next(row for row in document["kernelInventory"]["kernels"] if row["kernelId"] == kernel_id)
        reference = kernel["selections"][0]
        if reference["kind"] == "fixture":
            fixture = next(row for row in document["compilerFixtures"] if row["fixtureId"] == reference["fixtureId"])
            path = fixture["compilerInput"]["sourcePaths"][0]
            symbol = reference["kernelSymbol"]
        else:
            lesson = next(row for row in document["curriculum"]["lessons"] if row["lessonId"] == reference["lessonId"])
            tab = lesson["codeTabs"][reference["tabOrdinal"]]
            path = tab["sourcePath"]
            symbol = tab["sourceItem"]["cases"][reference["caseOrdinal"]]["kernelSymbol"]
        source = (root / path).read_bytes()
        function = next(row for row in self.validator.ordinary_rust_function_items(source.decode("utf-8"))
                        if row["kernelSymbol"] == symbol and row["attributedKernel"])
        binding = {
            "implementationKernelId": kernel_id, "selection": copy.deepcopy(reference),
            "selectionSha256": retained["selectionSha256"], "sourcePath": path,
            "sourceSha256": hashlib.sha256(source).hexdigest(),
            "functionUtf8Offset": function["functionUtf8Offset"],
        }
        kernel["variants"][0].update(status="source-bound", source=binding)
        return binding

    def test_checked_in_fill_simt_source_binding_matches_current_physical_source(self):
        kernel_id = "fixture:gfx942-fill-simulation:fill"
        kernel = next(row for row in self.manifest["kernelInventory"]["kernels"]
                      if row["kernelId"] == kernel_id)
        expected = copy.deepcopy(kernel["variants"][0]["source"])
        before = copy.deepcopy(self.manifest)
        self.assertEqual(self.bind_source_variant(self.manifest, ROOT, kernel_id), expected)
        self.assertEqual(self.manifest, before)
        self.assertEqual(expected["sourcePath"], "examples/fill/src/lib.rs")
        report = self.kernel_pair_report()
        self.assertEqual(report["sourceBoundVariantCount"], 76)
        self.assertEqual(report["sourceBoundPairCount"], 31)
        self.assertEqual(report["qualifiedPairCount"], 0)
        self.assertEqual(report["stageStatus"], "not-evaluated")
        self.assertIs(report["qualified"], False)
        self.assertIs(report["inventoryComplete"], False)
        self.assertIsNone(report["requiredPairCount"])
        self.assertEqual(kernel["variants"][1]["status"], "pending")

    def test_current_fill_display_binds_source_and_historical_display_refuses(self):
        lesson = self.curriculum_lesson()
        tab = lesson["codeTabs"][0]
        self.assertEqual(tab["sourceCommit"], "7404720885b13ee77bef4c48364993b463f25a6a")
        self.assertEqual(tab["displayedUtf8Bytes"], 837)
        self.assertEqual(tab["sourceSha256"], tab["displayedSha256"])
        current = (ROOT / "examples/fill/src/lib.rs").read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(), tab["sourceSha256"])
        display = next(row for row in self.manifest["kernelInventory"]["displayItems"]
                       if row["lessonId"] == "first-fill" and row["tabOrdinal"] == 0
                       and row["kernelSymbol"] == "fill")
        self.assertEqual(display["bindingStatus"], "fixture-source-contract")
        self.validator.validate_kernel_inventory(self.manifest, None, repo_root=ROOT)
        # The previous whole-file identity still cannot bind the current source.
        stale = copy.deepcopy(self.manifest)
        stale_lesson = next(row for row in stale["curriculum"]["lessons"]
                            if row["lessonId"] == "first-fill")
        stale_lesson["codeTabs"][0].update(
            sourceCommit="7a536e0a001202ac0bb9d8647c5395661f8fa1ec",
            displayedUtf8Bytes=308,
            displayedSha256="827ea368df5dd7f429792e0f8a21df79d4d5508525061a844c190da25de54213",
            sourceSha256="827ea368df5dd7f429792e0f8a21df79d4d5508525061a844c190da25de54213",
        )
        stale["kernelInventory"]["displayItems"] = [
            row for row in stale["kernelInventory"]["displayItems"]
            if not (row["lessonId"] == "first-fill" and row["tabOrdinal"] == 0
                    and row["kernelSymbol"] == "fill_reference")
        ]
        stale_display = next(row for row in stale["kernelInventory"]["displayItems"]
                             if row["lessonId"] == "first-fill" and row["tabOrdinal"] == 0
                             and row["kernelSymbol"] == "fill")
        stale_display["functionUtf8Offset"] = 148
        with self.assertRaisesRegex(SystemExit, "fixture display differs from the exact current source occurrence"):
            self.validator.validate_kernel_inventory(stale, None, repo_root=ROOT)

    def test_fill_reference_proof_is_opt_in_not_a_replacement_manifest_selection(self):
        fill = next(row for row in self.manifest["compilerFixtures"]
                    if row["fixtureId"] == "gfx942-fill-simulation")
        inputs = fill["compilerInput"]
        self.assertEqual(inputs["features"], [])
        self.assertIs(inputs["defaultFeatures"], True)
        cargo, name, _ = self.validator.parse_package_identity(
            ROOT / inputs["packageManifest"], "fill opt-in proof")
        self.assertEqual(name, "fe2o3-fill")
        self.assertEqual(self.validator.cargo_feature_closure(cargo, [], True, "default fill"), [])
        self.assertEqual(self.validator.cargo_feature_closure(
            cargo, ["reference-proof"], True, "auxiliary proof fill"), ["reference-proof"])
        self.assertEqual(inputs["kernelSymbols"], ["fill"])
        self.assertEqual(inputs["sourcePaths"], ["examples/fill/src/lib.rs"])
        items = self.validator.ordinary_rust_function_items(
            (ROOT / inputs["sourcePaths"][0]).read_text(encoding="utf-8"))
        self.assertEqual([item["kernelSymbol"] for item in items
                          if item["attributedKernel"]], ["fill"])

    def test_real_source_bound_variant_report_does_not_qualify_or_complete_census(self):
        # A physical source association, not a new tile implementation or receipt.
        kernel_id = "fixture:gfx950-gpt-oss-serial-router-simt:gfx950_gpt_oss_120b_decode_megakernel_v1"
        foreign_id = "fixture:gfx950-gpt-oss-held-fragments-simt:gfx950_gpt_oss_120b_decode_megakernel_v1"
        binding = self.bind_source_variant(self.manifest, ROOT, kernel_id)
        report = self.kernel_pair_report()
        self.assertEqual(report["sourceBoundVariantCount"], 77)
        self.assertEqual(report["sourceBoundPairCount"], 31)
        self.assertEqual(report["variantBindingStatus"], "partial")
        self.assertIs(report["qualified"], False)
        self.assertEqual(report["qualifiedPairCount"], 0)
        self.assertEqual(report["stageStatus"], "not-evaluated")
        self.assertIs(report["inventoryComplete"], False)
        self.assertIsNone(report["requiredPairCount"])
        self.assertIn("per-kernel-variant-sources", report["missingBindings"])
        self.assertIn("per-variant-target-evidence", report["missingBindings"])
        kernel = next(row for row in report["kernelInventory"]["kernelIdentities"]
                      if row["kernelId"] == binding["implementationKernelId"])
        self.assertEqual(kernel["variants"][0]["source"], binding)
        self.assertEqual(kernel["variants"][1]["status"], "pending")
        for mutate, message in (
            (lambda value: value.update(sourceSha256="0" * 64), "exact current source occurrence"),
            (lambda value: value.update(selectionSha256="0" * 64), "selection digest is stale"),
            (lambda value: value.update(implementationKernelId=foreign_id), "does not belong"),
        ):
            changed = copy.deepcopy(self.manifest)
            row = next(row for row in changed["kernelInventory"]["kernels"]
                       if row["kernelId"] == binding["implementationKernelId"])
            mutate(row["variants"][0]["source"])
            with self.subTest(message=message), self.assertRaisesRegex(SystemExit, message):
                self.validator.validate_kernel_inventory(changed, None)

    def test_fully_source_bound_report_still_requires_execution_and_artifact_evidence(self):
        document, runtime = self.gpt_fixture_document()
        fixtures = {row["fixtureId"]: row for row in document["compilerFixtures"]}
        entry = next(row for row in document["entries"] if row["lessonId"] == "gfx950-gpt-oss-120b-megakernel")
        entry["compilerFixtureIds"] = sorted(fixtures)
        document["entries"] = [entry]
        for kernel in document["kernelInventory"]["kernels"]:
            binding = self.bind_source_variant(document, ROOT, kernel["kernelId"])
            # Deliberate report-only declarations, not a claim that these real
            # sources implement both modes or have equivalent numerical behavior.
            kernel["variants"][1].update(status="source-bound", source=copy.deepcopy(binding))
        self.validator.validate_site_inventory(document["curriculum"], runtime)
        for census in (None, runtime):
            with self.subTest(runtime_census=census is not None):
                report = self.validator._kernel_pair_report(document, fixtures, {}, census, repo_root=ROOT)
                self.assertEqual(report["variantBindingStatus"], "source-bound")
                self.assertEqual(report["sourceBoundVariantCount"], 12)
                self.assertEqual(report["sourceBoundPairCount"], 6)
                self.assertNotIn("per-kernel-variant-sources", report["missingBindings"])
                self.assertIn("per-variant-target-evidence", report["missingBindings"])
                self.assertIs(report["qualified"], False)
                self.assertEqual(report["qualifiedPairCount"], 0)
                self.assertEqual(report["stageStatus"], "not-evaluated")
                self.assertIs(report["productionContract"]["requiresFinalOptimizedGraphVerification"], True)
                self.assertEqual(report["productionContract"]["requiredPolicyVersion"], 4)
                self.assertIs(report["productionContract"]["allowsPipelineSelection"], False)
                self.assertIs(report["productionContract"]["allowsFallback"], False)
                if census is None:
                    self.assertIsNone(report["requiredPairCount"])
                    self.assertIs(report["inventoryComplete"], False)
                    self.assertIn("exhaustive-kernel-identity", report["missingBindings"])
                else:
                    # Six identities in this component fixture, not the website's denominator.
                    self.assertEqual(report["requiredPairCount"], 6)
                    self.assertIs(report["inventoryComplete"], True)
                    self.assertEqual(report["missingBindings"], ["per-variant-target-evidence"])

    def test_real_fixture_same_symbol_file_and_feature_substitutions_reject(self):
        original, _ = self.gpt_fixture_document()
        rows = original["kernelInventory"]["displayItems"]
        for index in range(3):
            document = copy.deepcopy(original)
            document["kernelInventory"]["displayItems"][index]["kernelIds"] = rows[(index + 1) % 3]["kernelIds"]
            with self.subTest(index=index), self.assertRaisesRegex(SystemExit, "exact selected source"):
                self.validator.validate_kernel_inventory(document, None)
        document = copy.deepcopy(original)
        fixture = next(row for row in document["compilerFixtures"] if row["fixtureId"] == "gfx950-gpt-oss-serial-router")
        fixture["compilerInput"]["features"] = ["kernel-gpt-oss-decode-held-fragments"]
        fixture["compilerInput"]["contractSha256"] = self.validator.fixture_input_contract_sha256(fixture)
        with self.assertRaisesRegex(SystemExit, "exact current source occurrence"):
            self.validator.validate_kernel_inventory(document, None)
        for row_index in range(6):
            document = copy.deepcopy(original)
            document["kernelInventory"]["displayItems"][row_index]["functionUtf8Offset"] += 1
            with self.subTest(row=row_index), self.assertRaisesRegex(SystemExit, "exact current source occurrence"):
                self.validator.validate_kernel_inventory(document, None)

    def test_real_fixture_physical_closure_and_module_selection_are_rechecked(self):
        original, _ = self.gpt_fixture_document()
        package = original["compilerFixtures"][0]["compilerInput"]["packageManifest"]
        package_root = (ROOT / package).parent
        with tempfile.TemporaryDirectory(prefix="fe2o3-fixture-binding-") as temporary:
            root = Path(temporary)
            paths = [path for path, _ in self.validator.package_rust_sources(ROOT, package, "test")]
            paths.extend([package_root / "Cargo.toml", package_root / "Cargo.lock"])
            for path in paths:
                copy_path = root / path.relative_to(ROOT)
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                copy_path.write_bytes(path.read_bytes())

            def refresh(document):
                sources = self.validator.package_rust_sources(root, package, "test")
                digest = self.validator.package_source_closure_sha256(root, sources)
                for fixture in document["compilerFixtures"]:
                    fixture["compilerInput"]["sourceClosureSha256"] = digest
                    fixture["compilerInput"]["contractSha256"] = self.validator.fixture_input_contract_sha256(fixture)

            self.validator.validate_kernel_inventory(original, None, repo_root=root)
            helper = root / package_root.relative_to(ROOT) / "src/reference.rs"
            helper.write_bytes(helper.read_bytes() + b"\n// changed inactive closure member\n")
            with self.assertRaisesRegex(SystemExit, "sourceClosureSha256 is stale"):
                self.validator.validate_kernel_inventory(original, None, repo_root=root)
            document = copy.deepcopy(original)
            refresh(document)
            self.validator.validate_kernel_inventory(document, None, repo_root=root)

            library = root / package_root.relative_to(ROOT) / "src/lib.rs"
            before = library.read_text(encoding="utf-8")
            for replacement, message in (("", "roster differs"),
                                         ('#[cfg(unknown)] pub mod kernel_router_serial;', "cfg predicate"),
                                         ('pub mod kernel_router_serial; pub mod kernel_router_serial;', "ambiguous")):
                library.write_text(before.replace("pub mod kernel_router_serial;", replacement), encoding="utf-8")
                altered = copy.deepcopy(document)
                refresh(altered)
                with self.subTest(replacement=replacement), self.assertRaisesRegex(SystemExit, message):
                    self.validator.validate_kernel_inventory(altered, None, repo_root=root)
            library.write_text(before, encoding="utf-8")
            bound = root / package_root.relative_to(ROOT) / "src/kernel_router_serial.rs"
            bound.write_bytes(bound.read_bytes() + b"\n// changed bound source\n")
            refresh(document)
            with self.assertRaisesRegex(SystemExit, "exact current source occurrence"):
                self.validator.validate_kernel_inventory(document, None, repo_root=root)

    def test_kernel_identity_extension_rejects_incomplete_or_false_claims_before_stdout(self):
        for mutate in (
            lambda value: value["kernels"].pop(),
            lambda value: value["negativeCases"].pop(),
            lambda value: value["displayItems"].append(copy.deepcopy(value["displayItems"][0])),
            lambda value: value["kernels"][0]["variants"][0].update(status="qualified"),
        ):
            manifest = copy.deepcopy(self.original)
            mutate(manifest["kernelInventory"])
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(CHECKER), "--manifest", str(path),
                     "--emit-kernel-pairs"], text=True, capture_output=True, check=False,
                )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("tutorial kernel manifest:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_kernel_identity_runtime_census_is_required_even_without_report(self):
        curriculum, inventory = self.site_inventory_fixture()
        self.manifest["curriculum"] = curriculum
        extension = self.manifest["kernelInventory"]
        extension["kernels"] = [
            row for row in extension["kernels"]
            if all(selection["kind"] == "fixture" for selection in row["selections"])
        ]
        extension["negativeCases"] = []
        extension["displayItems"] = []
        for lesson_id, ordinal, prefix, name, token in (
            ("first-fill", 0, "#[kernel] #[cfg_attr(any(), kernel)] fn ", "visible", "r#visible"),
            ("typed-vecadd", 3, "#[kernel] fn ", "\u03c0", "\u03c0"),
        ):
            code = prefix + token + "() {}\n"
            lesson = next(row for row in curriculum["lessons"] if row["lessonId"] == lesson_id)
            runtime = next(row for row in inventory["lessons"] if row["id"] == lesson_id)
            fields = {"displayedUtf8Bytes": len(code.encode()),
                      "displayedSha256": hashlib.sha256(code.encode()).hexdigest()}
            lesson["codeTabs"][ordinal].update(fields)
            runtime["codeTabs"][ordinal].update(fields, displayedCode=code)
            extension["displayItems"].append({
                "lessonId": lesson_id, "tabOrdinal": ordinal,
                "functionUtf8Offset": len(prefix.encode()), "kernelSymbol": name,
                "classification": "kernel", "kernelIds": [], "negativeCases": [],
                "bindingStatus": "pending", "reason": "Synthetic source binding remains pending.",
            })
        report = self.kernel_pair_report(inventory)
        self.assertEqual(report["schema"], "fe2o3-tutorial-kernel-pair-obligations-v2")
        self.assertTrue(report["kernelInventory"]["runtimeCensusValidated"])
        self.assertEqual(report["kernelInventory"]["displayItemCount"], 2)
        self.assertEqual(report["kernelInventory"]["pendingDisplayItemCount"], 2)
        self.assertFalse(report["inventoryComplete"])
        self.assertIsNone(report["requiredPairCount"])
        self.assertEqual(report["qualifiedPairCount"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            runtime_path = Path(temporary) / "runtime.json"
            runtime_path.write_text(json.dumps(inventory), encoding="utf-8")
            command = [sys.executable, "-I", "-B", str(CHECKER), "--manifest", str(path),
                       "--site-inventory", str(runtime_path)]
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            extension["displayItems"].pop()
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)

    def test_kernel_pair_report_preserves_selections_and_negative_cases(self):
        before = copy.deepcopy(self.manifest)
        report = self.kernel_pair_report()
        self.assertEqual(self.manifest, before)
        rows = report["fixtureSelections"]
        expected = [(fixture["fixtureId"], symbol)
                    for fixture in sorted(self.original["compilerFixtures"], key=lambda row: row["fixtureId"])
                    for symbol in fixture["compilerInput"]["kernelSymbols"]]
        self.assertEqual([(row["fixtureId"], row["selection"]["kernelSymbol"]) for row in rows], expected)
        attnres = [row for row in rows if row["selection"]["kernelSymbol"] == "gfx950_attnres_aggregate"]
        self.assertEqual({row["fixtureId"] for row in attnres}, {
            "gfx950-attnres-aggregate", "gfx950-attnres-aggregate-explicit-reuse",
            "gfx950-attnres-aggregate-simt", "gfx950-attnres-aggregate-tile",
        })
        ablations = [row for row in attnres if row["fixtureId"] in {
            "gfx950-attnres-aggregate", "gfx950-attnres-aggregate-explicit-reuse",
        }]
        self.assertEqual(len(ablations), 2)
        self.assertNotEqual(ablations[0]["selection"]["features"], ablations[1]["selection"]["features"])
        self.assertNotEqual(ablations[0]["compilerInputContractSha256"], ablations[1]["compilerInputContractSha256"])
        fp4 = next(row for row in rows if row["fixtureId"] == "gfx950-fp4-gemm")
        self.assertIn("gfx950-fp4-gemm-performance-lab", fp4["scopeLessonIds"])
        display = next(row for row in report["displayObservations"]
                       if row["lessonId"] == "gfx950-fp4-gemm-performance-lab")
        self.assertEqual(display["sourceItemStatus"], "pending")
        self.assertIsNone(display["lexicalKernelNames"])
        cases = report["sourceDriverCases"]
        self.assertEqual(
            [(row["lessonId"], row["tabOrdinal"], row["caseOrdinal"]) for row in cases],
            [("cpu-semantic-simulation", 0, index) for index in range(13)]
            + [("cpu-semantic-simulation", 6, 0)],
        )
        self.assertEqual(Counter(row["expectation"]["kind"] for row in cases),
                         {"verified-bundle-export": 11, "rejected": 3})
        self.assertTrue(all(row["expectation"]["outputArtifact"] == "absent"
                            for row in cases if row["expectation"]["kind"] == "rejected"))
        self.assertEqual(report["qualifiedPairCount"], 0)
        self.assertIsNone(report["requiredPairCount"])
        self.assertEqual(sum(len(row["modes"]) for row in report["lessonRequirements"]), 96)

    def test_kernel_pair_runtime_observations_are_lexical_not_compiler_evidence(self):
        self.manifest.pop("kernelInventory", None)
        curriculum, inventory = self.site_inventory_fixture()
        lesson = next(row for row in curriculum["lessons"] if row["lessonId"] == "first-fill")
        projected = next(row for row in inventory["lessons"] if row["id"] == "first-fill")
        code = ('// #[kernel] fn comment() {}\n'
                'const TEXT: &str = "#[kernel] fn string() {}";\n'
                'macro_rules! template { () => { #[kernel] fn hidden() {} }; }\n'
                '#[cfg(feature = "a")] #[kernel] fn selected() {}\n'
                '#[cfg(feature = "b")] #[kernel] fn selected() {}\n')
        fields = {"displayedUtf8Bytes": len(code.encode()),
                  "displayedSha256": hashlib.sha256(code.encode()).hexdigest()}
        lesson["codeTabs"][0].update(fields)
        projected["codeTabs"][0].update(fields, displayedCode=code)
        self.manifest["curriculum"] = curriculum
        report = self.kernel_pair_report(inventory)
        observed = next(row for row in report["displayObservations"] if row["lessonId"] == "first-fill")
        self.assertEqual(observed["lexicalKernelNames"], ["selected", "selected"])
        self.assertEqual(observed["sourceItemStatus"], "pending")
        self.assertIs(report["inventoryComplete"], False)
        self.assertEqual(report["qualifiedPairCount"], 0)
        projected["codeTabs"][0]["displayedCode"] += "changed"
        with self.assertRaisesRegex(SystemExit, "displayed bytes do not match"):
            self.kernel_pair_report(inventory)

    def test_kernel_pair_cli_rejects_stale_duplicate_missing_and_legacy_inputs(self):
        def without_curriculum():
            self.manifest.pop("kernelInventory", None)
            self.manifest.pop("curriculum")

        def legacy():
            self.manifest.pop("kernelInventory", None)
            self.manifest["curriculum"]["schema"] = self.validator.CURRICULUM_SCHEMA
            lesson = self.curriculum_lesson("cpu-semantic-simulation")
            for tab in lesson["codeTabs"]:
                if tab["sourceItem"] is not None:
                    tab.update(sourceItem=None, sourceItemStatus="pending")
            lesson["sourceBindingGap"] = "Legacy source driver is not yet contract-bound."

        for mutate, pattern in (
            (without_curriculum, "exhaustive curriculum"),
            (legacy, "V2 source-item curriculum"),
            (lambda: self.manifest["compilerFixtures"][0]["compilerInput"].update(contractSha256="0" * 64), "stale"),
            (lambda: self.manifest["compilerFixtures"].append(copy.deepcopy(self.manifest["compilerFixtures"][0])), "duplicate"),
            (lambda: self.curriculum_lesson()["variants"].pop(), "ordered SIMT/tile"),
        ):
            self.manifest = copy.deepcopy(self.original)
            mutate()
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "manifest.json"
                path.write_text(json.dumps(self.manifest), encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(CHECKER), "--manifest", str(path), "--emit-kernel-pairs"],
                    text=True, capture_output=True, check=False,
                )
            with self.subTest(pattern=pattern):
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(pattern, result.stderr)

    def test_kernel_pair_cli_cannot_mix_matrix_output_or_qualification(self):
        for flags, pattern in (
            (["--emit-matrix", "gfx942"], "not allowed with argument"),
            (["--require-qualified"], "qualification receipts"),
        ):
            result = subprocess.run(
                [sys.executable, str(CHECKER), "--emit-kernel-pairs", *flags],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn(pattern, result.stderr)

    def test_kernel_pair_projection_preserves_cross_target_origins(self):
        fixture = copy.deepcopy(self.original["compilerFixtures"][0])
        other = copy.deepcopy(fixture)
        other.update(fixtureId="second-target", target="gfx950")
        manifest = copy.deepcopy(self.original)
        manifest.pop("kernelInventory", None)
        manifest["entries"] = [{"lessonId": "first-fill", "compilerFixtureIds": [fixture["fixtureId"], other["fixtureId"]]}]
        # Exercise projection identity alone; this synthetic pair is not a validated contract.
        rows = self.validator._kernel_pair_report(
            manifest, {row["fixtureId"]: row for row in (fixture, other)}, {}, None,
        )["fixtureSelections"]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["selection"]["target"] for row in rows], ["gfx942", "gfx950"])
        for key in ("packageManifest", "cargoTarget", "defaultFeatures", "features", "sourcePaths", "kernelSymbol"):
            self.assertEqual(rows[0]["selection"][key], rows[1]["selection"][key])

    def test_kernel_pair_runtime_cli_is_bounded_and_rejects_before_output(self):
        self.manifest.pop("kernelInventory", None)
        curriculum, inventory = self.site_inventory_fixture()
        lesson = next(row for row in curriculum["lessons"] if row["lessonId"] == "first-fill")
        projected = next(row for row in inventory["lessons"] if row["id"] == "first-fill")
        code = "#[kernel] fn observed() {}\n" * 200
        fields = {"displayedUtf8Bytes": len(code.encode()), "displayedSha256": hashlib.sha256(code.encode()).hexdigest()}
        lesson["codeTabs"][0].update(fields)
        projected["codeTabs"][0].update(fields, displayedCode=code)
        inventory["site"]["unvalidatedExtra"] = "must not be copied into the report"
        self.manifest["curriculum"] = curriculum
        fixtures = self.validator.validate_manifest(ROOT, self.manifest)
        self.validator.validate_site_inventory(curriculum, inventory)
        with mock.patch.object(self.validator, "MAX_KERNEL_PAIR_RECORDS", 200):
            report = self.validator._kernel_pair_report(self.manifest, fixtures, {}, inventory)
            self.assertNotIn("unvalidatedExtra", report["runtimeProjectionSite"])
        with mock.patch.object(self.validator, "MAX_KERNEL_PAIR_RECORDS", 199):
            with self.assertRaisesRegex(SystemExit, "lexical declaration bound"):
                self.validator._kernel_pair_report(self.manifest, fixtures, {}, inventory)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            runtime = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            command = [sys.executable, str(CHECKER), "--manifest", str(path),
                       "--emit-kernel-pairs", "--site-inventory", str(runtime)]
            runtime.write_text(json.dumps(inventory), encoding="utf-8")
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIs(json.loads(result.stdout)["inventoryComplete"], False)
            projected["codeTabs"][0]["displayedCode"] += "changed"
            runtime.write_text(json.dumps(inventory), encoding="utf-8")
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("displayed bytes do not match", result.stderr)

    def test_kernel_pair_projection_is_deterministic_and_bounded(self):
        self.manifest.pop("kernelInventory", None)
        report = self.kernel_pair_report()
        encoded = self.validator._encode_kernel_pair_report(report)
        self.assertEqual(encoded, self.validator._encode_kernel_pair_report(self.kernel_pair_report()))
        expected = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        self.assertEqual(report["sourceContractSha256"], hashlib.sha256(expected).hexdigest())
        with mock.patch.object(self.validator, "MAX_SITE_INVENTORY_BYTES", len(encoded)):
            self.assertEqual(self.validator._encode_kernel_pair_report(report), encoded)
        with mock.patch.object(self.validator, "MAX_SITE_INVENTORY_BYTES", len(encoded) - 1):
            with self.assertRaisesRegex(SystemExit, "output byte bound"):
                self.validator._encode_kernel_pair_report(report)
        fixtures = {row["fixtureId"]: row for row in self.manifest["compilerFixtures"]}
        count = sum(len(report[key]) for key in ("fixtureSelections", "sourceDriverCases", "displayObservations"))
        with mock.patch.object(self.validator, "MAX_KERNEL_PAIR_RECORDS", count):
            self.assertEqual(self.validator._kernel_pair_report(self.manifest, fixtures, report["sourceBindingGaps"], None), report)
        with mock.patch.object(self.validator, "MAX_KERNEL_PAIR_RECORDS", count - 1):
            with self.assertRaisesRegex(SystemExit, "record bound"):
                self.validator._kernel_pair_report(self.manifest, fixtures, {}, None)

    def test_release_schema_and_accepted_baseline_are_rejected(self):
        self.manifest["schema"] = "fe2o3-tutorial-kernel-manifest-v1"
        self.reject("source-contract schema")
        self.manifest = copy.deepcopy(self.original)
        self.manifest["baseline"]["compilerCommit"] = "a" * 40
        self.reject("accepted compiler baseline")

    def test_production_contract_cannot_allow_fallback_or_skip_final_verification(self):
        for key, value in (
            ("allowsFallback", True), ("allowsPipelineSelection", True),
            ("requiresFinalOptimizedGraphVerification", False), ("requiredPolicyVersion", 3),
        ):
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(self.original)
                self.manifest["productionContract"][key] = value
                self.reject("production pipeline")

    def test_duplicates_dangling_scope_and_missing_simulation_are_rejected(self):
        self.manifest["compilerFixtures"].append(copy.deepcopy(self.manifest["compilerFixtures"][0]))
        self.reject("duplicate or invalid fixtureId")
        self.manifest = copy.deepcopy(self.original)
        self.manifest["entries"][0]["compilerFixtureIds"] = ["absent"]
        self.reject("no fixture|unknown fixture")
        self.manifest = copy.deepcopy(self.original)
        self.manifest["qualification"]["suites"] = [
            s for s in self.manifest["qualification"]["suites"] if s["suiteId"] != "semantic-simulation-fill"
        ]
        self.reject("missing pending semantic-simulation")

    def test_runnable_and_ordinary_source_entries_cannot_be_relabelled(self):
        self.manifest["entries"][0]["classification"] = "design-only"
        self.reject("cannot be downgraded")
        self.manifest = copy.deepcopy(self.original)
        entry = next(e for e in self.manifest["entries"] if e["lessonId"] == "gemm-proof-plan")
        entry["classification"] = "simulator-only"
        self.reject("ordinary kernel cannot be downgraded")

    def test_input_identity_and_feature_contracts_fail_closed(self):
        for key, value, pattern in (
            ("packageManifestSha256", "0" * 64, "packageManifestSha256 is stale"),
            ("cargoLockSha256", "0" * 64, "cargoLockSha256 is stale"),
            ("sourceClosureSha256", "0" * 64, "sourceClosureSha256 is stale"),
            ("kernelSymbols", ["not_an_attributed_kernel"], "not attributed"),
            ("features", ["absent-feature"], "unknown Cargo feature"),
            ("contractSha256", "0" * 64, "contractSha256 is stale"),
        ):
            with self.subTest(key=key):
                fixture = copy.deepcopy(self.original["compilerFixtures"][0])
                fixture["compilerInput"][key] = value
                with self.assertRaisesRegex(SystemExit, pattern):
                    self.validator.validate_compiler_input(ROOT, fixture, "fixture")

    def test_runner_arguments_and_environment_cannot_escape_matrix_contract(self):
        for key, value in (
            ("runnerArguments", ["x", "y"]),
            ("runnerArguments", ["x|y"]),
            ("environment", ["FE2O3_OTHER_PIPELINE=1"]),
            ("artifactName", "../input.hsaco"),
            ("runnerPath", "../outside.sh"),
        ):
            with self.subTest(key=key):
                fixture = copy.deepcopy(self.original["compilerFixtures"][0])
                fixture["matrix"][key] = value
                with self.assertRaises(SystemExit):
                    self.validator.validate_matrix(ROOT, fixture, "fixture")

    def test_pending_simulation_and_suite_cannot_claim_success(self):
        self.manifest["compilerFixtures"][0]["simulation"]["status"] = "passed"
        self.reject("explicitly pending")
        self.manifest = copy.deepcopy(self.original)
        self.manifest["qualification"]["suites"][0]["availability"] = "available"
        self.reject("not an available or passed adapter")
        self.manifest = copy.deepcopy(self.original)
        self.manifest["compilerFixtures"][0]["simulation"]["canonicalKirVersion"] = 10
        self.reject("Bundle V7 / KIR V12")

    def test_source_closure_tracks_nested_sources_but_not_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "example"
            (package / "src").mkdir(parents=True)
            (package / "target").mkdir()
            (package / "Cargo.toml").write_text('[package]\nname = "test"\nversion = "0.1.0"\n', encoding="utf-8")
            source = package / "src/lib.rs"
            source.write_text("#[kernel] fn example() {}\n", encoding="utf-8")
            paths = self.validator.package_rust_sources(root, "example/Cargo.toml", "test")
            before = self.validator.package_source_closure_sha256(root, paths)
            (package / "target/generated.rs").write_text("not an input\n", encoding="utf-8")
            self.assertEqual(before, self.validator.package_source_closure_sha256(
                root, self.validator.package_rust_sources(root, "example/Cargo.toml", "test")
            ))
            source.write_text("#[kernel] fn changed() {}\n", encoding="utf-8")
            self.assertNotEqual(before, self.validator.package_source_closure_sha256(
                root, self.validator.package_rust_sources(root, "example/Cargo.toml", "test")
            ))
            (package / "src/link.rs").symlink_to(source)
            with self.assertRaisesRegex(SystemExit, "symlink"):
                self.validator.package_rust_sources(root, "example/Cargo.toml", "test")

    def test_standalone_workspace_does_not_inherit_unrelated_parent_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Cargo.lock").write_text("parent", encoding="utf-8")
            package = root / "example"
            package.mkdir()
            (package / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "workspace root has no Cargo.lock"):
                self.validator.effective_cargo_lock(root, package, "test")

    def test_package_presence_reuses_exact_cached_attributions_including_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "example"
            (package / "src").mkdir(parents=True)
            (package / "Cargo.toml").write_text('[package]\nname = "test"\nversion = "0.1.0"\n', encoding="utf-8")
            (package / "src/a.rs").write_text('const TEXT: &str = "#[kernel] fn fake() {}";\n', encoding="utf-8")
            (package / "src/z.rs").write_text("#[kernel] fn real() {}\n", encoding="utf-8")
            sources = self.validator.package_rust_sources(root, "example/Cargo.toml", "fixture")
            names = {
                path.relative_to(package).as_posix(): self.validator.ordinary_attributed_kernel_names(text)
                for path, text in sources
            }
            self.assertEqual(names, {"src/a.rs": [], "src/z.rs": ["real"]})
            compiler_cache = {"example/Cargo.toml": {
                "packageSources": sources, "attributedNames": names,
            }}
            package_kernels = {}
            with mock.patch.object(
                self.validator, "package_rust_sources", wraps=self.validator.package_rust_sources
            ) as walk, mock.patch.object(
                self.validator, "ordinary_attributed_kernel_names", wraps=self.validator.ordinary_attributed_kernel_names
            ) as scan:
                for lesson in ("first", "second"):
                    self.assertTrue(self.validator._package_has_ordinary_kernel(
                        root, "example/Cargo.toml", lesson, compiler_cache, package_kernels
                    ))
                walk.assert_not_called()
                scan.assert_not_called()
            self.assertEqual(package_kernels, {"example/Cargo.toml": True})

    def test_fresh_package_cache_observes_changed_off_entry_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "example"
            (package / "src").mkdir(parents=True)
            (package / "Cargo.toml").write_text('[package]\nname = "test"\nversion = "0.1.0"\n', encoding="utf-8")
            (package / "src/lib.rs").write_text("fn ordinary() {}\n", encoding="utf-8")
            hidden = package / "src/hidden.rs"
            hidden.write_text("fn hidden() {}\n", encoding="utf-8")
            cache = {}
            with mock.patch.object(
                self.validator, "package_rust_sources", wraps=self.validator.package_rust_sources
            ) as walk:
                for lesson in ("first", "second"):
                    self.assertFalse(self.validator._package_has_ordinary_kernel(
                        root, "example/Cargo.toml", lesson, {}, cache
                    ))
                self.assertEqual(walk.call_count, 1)
            # validate_manifest owns fresh dictionaries on every invocation.
            # This helper-level test does not fabricate a qualified manifest.
            hidden.write_text("#[kernel] fn newly_added() {}\n", encoding="utf-8")
            with mock.patch.object(
                self.validator, "package_rust_sources", wraps=self.validator.package_rust_sources
            ) as walk:
                self.assertTrue(self.validator._package_has_ordinary_kernel(
                    root, "example/Cargo.toml", "new invocation", {}, {}
                ))
                self.assertEqual(walk.call_count, 1)

    def test_duplicate_json_keys_and_non_finite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            for text, pattern in (
                ('{"schema": 1, "schema": 2}', "duplicate JSON key"),
                ('{"schema": NaN}', "non-finite JSON constant"),
            ):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, pattern):
                    self.validator.load_manifest(path)

    def test_ordinary_kernel_scanner_ignores_comments_literals_and_macro_templates(self):
        scanner = self.validator.source_contains_ordinary_attributed_kernel
        self.assertFalse(scanner(
            "// #[kernel] fn fake() {}\n"
            'const TEXT: &str = r#"#[kernel] fn fake_two() {}"#;\n'
            "/* outer #[kernel] /* nested */ fn fake_three() {} */\n"
            "macro_rules! outer { () => { #[kernel] fn fake_four() {} }; }\n"
        ))
        self.assertTrue(scanner(
            '#[kernel(typed)] #[allow(dead_code)] pub unsafe extern "C" fn real() {}\n'
        ))
        self.assertTrue(scanner('#[cfg_attr(feature = "kernel", kernel)] fn real() {}\n'))
        with self.assertRaisesRegex(SystemExit, "unterminated .*string literal"):
            scanner('const BROKEN: &str = "unterminated')

    def test_v1_remains_strict_and_retains_its_unbound_source_gap(self):
        self.manifest["curriculum"]["schema"] = self.validator.CURRICULUM_SCHEMA
        with self.assertRaisesRegex(SystemExit, "source-item obligation"):
            self.validate_curriculum()
        lesson = self.curriculum_lesson("cpu-semantic-simulation")
        for tab in lesson["codeTabs"]:
            if tab["sourceItem"] is not None:
                tab.update(sourceItem=None, sourceItemStatus="pending")
        lesson["sourceBindingGap"] = "Legacy source driver is not yet contract-bound."
        self.validate_curriculum()
        self.manifest["curriculum"]["schema"] = "unknown"
        with self.assertRaisesRegex(SystemExit, "pending obligation schema"):
            self.validate_curriculum()

    def test_source_driver_has_complete_ordered_displayed_cases_without_qualification(self):
        tab = self.curriculum_lesson("cpu-semantic-simulation")["codeTabs"][0]
        item = tab["sourceItem"]
        self.assertEqual(len(item["cases"]), 13)
        self.assertEqual(sum(row["expectation"]["kind"] == "rejected" for row in item["cases"]), 3)
        self.assertGreater(item["sourceRanges"][0]["byteOffset"], item["sourceRanges"][1]["byteOffset"])
        self.assertEqual(self.validator.validate_source_item(ROOT, "cpu-semantic-simulation", tab, {}), tab["sourcePath"])
        self.assertEqual(tab["sourceItemStatus"], "contract-bound")
        self.assertEqual(self.manifest["curriculum"]["status"], "pending")

    def test_rehashed_source_contracts_reject_malformed_or_incomplete_coverage(self):
        mutations = [
            lambda item: item["cases"].pop(),
            lambda item: item["cases"].append(copy.deepcopy(item["cases"][0])),
            lambda item: item["cases"].reverse(),
            lambda item: item["cases"][0].update(kernelSymbol="aggregate_pair_struct"),
            lambda item: item["cases"][0].update(displayedFragmentOrdinal=1),
            lambda item: item["cases"][0].update(displayedFragmentOrdinal=True),
            lambda item: item["cases"][0].update(features=["unavailable_feature"]),
            lambda item: item["cases"][0].update(target="gfx000"),
            lambda item: item["cases"][0].update(testFunction="missing_driver_test"),
            lambda item: item["cases"][0].update(testFunction="workspace"),
            lambda item: item["cases"][0]["expectation"].update(bundleVersion=7),
            lambda item: item["cases"][0]["expectation"].update(bundleVersion=True),
            lambda item: item["cases"][0]["expectation"].update(kind="qualified"),
            lambda item: item["cases"][6]["expectation"].update(outputArtifact="present"),
            lambda item: item["sourceRanges"][0].update(byteOffset=-1),
            lambda item: item["sourceRanges"][0].update(byteLength=True),
            lambda item: item["sourceRanges"][0].update(byteLength=2**32),
            lambda item: item["sourceRanges"].append(copy.deepcopy(item["sourceRanges"][0])),
            lambda item: item["sourceRanges"].__setitem__(1, copy.deepcopy(item["sourceRanges"][0])),
            lambda item: item["compilerInput"].update(cargoLockSha256="0" * 64),
            lambda item: item["compilerInput"].update(sourceClosureSha256="0" * 64),
            lambda item: item["compilerInput"].update(packageManifestSha256="0" * 64),
            lambda item: item["driver"].update(path="other.rs"),
            lambda item: item.update(qualified=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                tab = copy.deepcopy(self.curriculum_lesson("cpu-semantic-simulation")["codeTabs"][0])
                mutate(tab["sourceItem"])
                tab["sourceItem"]["contractSha256"] = self.validator.source_item_contract_sha256("cpu-semantic-simulation", tab)
                with self.assertRaises(SystemExit):
                    self.validator.validate_source_item(ROOT, "cpu-semantic-simulation", tab, {})

    def test_source_contract_digest_binds_independent_driver_expectations(self):
        original = self.curriculum_lesson("cpu-semantic-simulation")["codeTabs"][0]
        for key, value in (
            ("features", ["aggregate_pair_struct"]),
            ("testFunction", "ordinary_rust_struct_argument_exports_exact_v4_components"),
            ("target", "gfx950"),
            ("expectation", {"kind": "verified-bundle-export", "bundleVersion": 4}),
        ):
            tab = copy.deepcopy(original)
            tab["sourceItem"]["cases"][0][key] = value
            self.assertNotEqual(
                original["sourceItem"]["contractSha256"],
                self.validator.source_item_contract_sha256("cpu-semantic-simulation", tab),
            )
        self.assertNotEqual(original["sourceItem"]["contractSha256"], self.validator.source_item_contract_sha256("other-lesson", original))

    def test_context_protocol_include_is_scoped_without_dropping_its_source(self):
        relative = (
            "crates/rustc-codegen-fe2o3/tests/fixtures/"
            "production-ranked-bounds-device"
        )
        package = ROOT / relative
        path = package / "context_vecadd.rs"
        sources = dict(self.validator.package_rust_sources(
            ROOT, relative + "/Cargo.toml", "context protocol regression"
        ))
        self.assertIn(path, sources)
        source = sources[path]
        self.assertIn(
            '#[cfg(feature = "provider_context_protocol")]\n'
            '#[macro_use]\nmod context_body {\n',
            source,
        )
        self.assertIn(
            "pub fn vecadd(_ctx: KernelContext<'_>,",
            source,
        )
        self.assertIn(
            "vecadd_kernel_body!(thread, (), production_f32_add, a, b, c);",
            source,
        )
        names = self.validator.ordinary_attributed_kernel_names(source)
        self.assertEqual(len(names), 1)
        self.assertIn("vecadd", names)
        validate = self.validator.validate_rust_source_includes
        validate(
            source, path, package, "context protocol regression",
            inactive_features=frozenset({"provider_context_protocol"}),
        )
        for inactive in (None, frozenset(), frozenset({"unrelated"})):
            with self.subTest(inactive=inactive), self.assertRaisesRegex(
                SystemExit, "non-literal include"
            ):
                validate(
                    source, path, package, "context protocol regression",
                    inactive_features=inactive,
                )

    def test_context_protocol_package_scope_retains_closure_and_selected_refusal(self):
        original = ROOT / (
            "crates/rustc-codegen-fe2o3/tests/fixtures/"
            "production-ranked-bounds-device/context_vecadd.rs"
        )
        context_source = original.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "fixture"
            (package / "src").mkdir(parents=True)
            manifest = package / "Cargo.toml"
            manifest.write_text(
                '[package]\nname = "fixture"\nversion = "0.1.0"\n'
                'build = false\n[workspace]\n[features]\n'
                'provider_context_protocol = []\n'
                'protocol_alias = ["provider_context_protocol"]\n'
                'default = ["provider_context_protocol"]\n',
                encoding="utf-8",
            )
            lock = package / "Cargo.lock"
            lock.write_text("version = 4\n", encoding="utf-8")
            (package / "src/lib.rs").write_text(
                "#[kernel] fn selected() {}\n", encoding="utf-8"
            )
            context = package / "context_vecadd.rs"
            context.write_text(context_source, encoding="utf-8")
            sources = self.validator.package_rust_sources(
                root, "fixture/Cargo.toml", "context package regression"
            )
            self.assertEqual(
                {path.relative_to(package).as_posix() for path, _ in sources},
                {"context_vecadd.rs", "src/lib.rs"},
            )
            item = {
                "packageManifest": "fixture/Cargo.toml",
                "packageManifestSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "cargoLockPath": "fixture/Cargo.lock",
                "cargoLockSha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "sourcePaths": ["fixture/src/lib.rs"],
                "sourceClosureSha256": self.validator.package_source_closure_sha256(
                    root, sources
                ),
                "cargoTarget": {
                    "kind": "lib", "name": "fixture", "sourcePath": "src/lib.rs"
                },
                "defaultFeatures": False,
                "features": [],
                "kernelSymbols": ["selected"],
            }
            validate = self.validator.validate_compiler_input_data
            result = validate(
                root, item, "context package regression",
                feature_scoped_includes=True,
            )
            self.assertEqual(result["enabledFeatures"], [])
            for features, defaults, scoped in (
                (["provider_context_protocol"], False, True),
                (["protocol_alias"], False, True),
                ([], True, True),
                ([], False, False),
            ):
                candidate = copy.deepcopy(item)
                candidate.update(features=features, defaultFeatures=defaults)
                with self.subTest(
                    features=features, defaults=defaults, scoped=scoped
                ), self.assertRaisesRegex(SystemExit, "non-literal include"):
                    validate(
                        root, candidate, "context package regression",
                        feature_scoped_includes=scoped,
                    )
            # Disabled bodies remain exact package inputs, not omitted closure files.
            context.write_text(
                context_source + "\n// disabled source is still digest-bound\n",
                encoding="utf-8",
            )
            changed = self.validator.package_rust_sources(
                root, "fixture/Cargo.toml", "context package regression"
            )
            changed_digest = self.validator.package_source_closure_sha256(root, changed)
            self.assertNotEqual(item["sourceClosureSha256"], changed_digest)
            with self.assertRaisesRegex(SystemExit, "sourceClosureSha256 is stale"):
                validate(
                    root, item, "context package regression",
                    feature_scoped_includes=True,
                )
            item["sourceClosureSha256"] = changed_digest
            validate(
                root, item, "context package regression",
                feature_scoped_includes=True,
            )

    def test_context_protocol_scope_preserves_external_and_unconditional_refusals(self):
        feature = frozenset({"provider_context_protocol"})
        source = (
            '#[cfg(feature = "provider_context_protocol")]\n'
            '#[macro_use]\nmod context_body { include!("../outside.rs"); }\n'
        )
        package = ROOT / "include-regression-package"
        args = (package / "context_vecadd.rs", package, "context include regression")
        validate = self.validator.validate_rust_source_includes
        validate(source, *args, inactive_features=feature)
        for inactive in (None, frozenset(), frozenset({"unrelated"})):
            with self.subTest(inactive=inactive), self.assertRaisesRegex(
                SystemExit, "include! path escaping its package root"
            ):
                validate(source, *args, inactive_features=inactive)
        for candidate in (
            source + 'include!(concat!("outside", ".rs"));\n',
            '#[cfg(feature = "provider_context_protocol")]\n'
            '#[macro_use]\ninclude!(concat!("outside", ".rs"));\n',
        ):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(
                SystemExit, "non-literal include"
            ):
                validate(candidate, *args, inactive_features=feature)

    def test_feature_scoped_includes_require_a_literal_false_top_level_module(self):
        source = '#[cfg(feature = "generated")] mod selected { include!(env!("SOURCE")); }'
        validate = self.validator.validate_rust_source_includes
        args = (ROOT / "source.rs", ROOT, "fixture")
        validate(source, *args, inactive_features=frozenset({"generated"}))
        validate(source.replace("mod selected", "#[allow(dead_code)] pub(crate) mod selected"), *args, inactive_features=frozenset({"generated"}))
        for candidate, features in (
            (source, None), (source, frozenset()),
            (source.replace('feature = "generated"', 'unknown'), frozenset({"generated"})),
            (source.replace('feature = "generated"', 'not(feature = "generated")'), frozenset({"generated"})),
            ("macro_rules! outer { () => {" + source + "}; }", frozenset({"generated"})),
            ("other! { " + source + " }", frozenset({"generated"})),
            (source.replace("#[cfg", "#![cfg"), frozenset({"generated"})),
            (source + ' include!(env!("OTHER"));', frozenset({"generated"})),
        ):
            with self.subTest(candidate=candidate, features=features), self.assertRaisesRegex(SystemExit, "non-literal include"):
                validate(candidate, *args, inactive_features=features)

    def test_build_scripts_cannot_inject_a_supposedly_disabled_feature(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "fixture"
            (package / "src").mkdir(parents=True)
            source = package / "src/lib.rs"
            source.write_text('#[cfg(feature = "generated")] mod selected { include!(env!("SOURCE")); }\n#[kernel] fn selected() {}\n')
            lock = package / "Cargo.lock"
            lock.write_text("version = 4\n")
            manifest = package / "Cargo.toml"
            for build, implicit, allowed in (("", False, True), ("", True, False), ('build = "builder.rs"\n', False, False), ("build = true\n", False, False), ("build = false\n", True, True)):
                manifest.write_text('[package]\nname = "fixture"\nversion = "0.1.0"\n' + build + '[workspace]\n[features]\ngenerated = []\n')
                build_path = package / "build.rs"
                build_path.unlink(missing_ok=True)
                if implicit:
                    build_path.write_text('fn main() { println!("cargo::rustc-cfg=feature=\\\"generated\\\""); }\n')
                if 'builder.rs' in build:
                    (package / "builder.rs").write_text("fn main() {}\n")
                sources = self.validator.package_rust_sources(root, "fixture/Cargo.toml", "test")
                item = {
                    "packageManifest": "fixture/Cargo.toml", "packageManifestSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "cargoLockPath": "fixture/Cargo.lock", "cargoLockSha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                    "sourcePaths": ["fixture/src/lib.rs"], "sourceClosureSha256": self.validator.package_source_closure_sha256(root, sources),
                    "cargoTarget": {"kind": "lib", "name": "fixture", "sourcePath": "src/lib.rs"},
                    "defaultFeatures": True, "features": [], "kernelSymbols": ["selected"],
                }
                with self.subTest(build=build, implicit=implicit):
                    if allowed:
                        self.validator.validate_compiler_input_data(root, item, "test", feature_scoped_includes=True)
                    else:
                        with self.assertRaisesRegex(SystemExit, "non-literal include"):
                            self.validator.validate_compiler_input_data(root, item, "test", feature_scoped_includes=True)
            for features in (
                '[features]\ngenerated = []\nfirst = ["generated"]\ndefault = ["first"]\n',
                '[features]\ngenerated = []\ndefault = ["foo/bar"]\n',
                '[features]\ngenerated = []\ndefault = ["dep:foo"]\n',
                '[features]\nfoo = []\n',
                '[features]\ngenerated = "unsupported"\n',
            ):
                manifest.write_text('[package]\nname = "fixture"\nversion = "0.1.0"\nbuild = false\n[workspace]\n' + features)
                item["packageManifestSha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
                with self.subTest(features=features), self.assertRaisesRegex(SystemExit, "non-literal include"):
                    self.validator.validate_compiler_input_data(root, item, "test", feature_scoped_includes=True)

    def test_source_driver_rejects_disabled_redirected_and_symlinked_test_targets(self):
        original = self.curriculum_lesson("cpu-semantic-simulation")["codeTabs"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "crates/test-driver"
            (package / "tests").mkdir(parents=True)
            (package / "tests/selection.rs").write_text("#[test] fn selected() {}\nfn helper() {}\n")
            path = package / "Cargo.toml"
            tab = copy.deepcopy(original)
            tab["sourceItem"]["driver"] = {"package": "test-driver", "target": "selection", "path": "crates/test-driver/tests/selection.rs"}
            for settings in (
                "autotests = false\n",
                '[[test]]\nname = "selection"\npath = "other.rs"\n',
                '[[test]]\nname = "selection"\nharness = false\n',
                '[[test]]\nname = "selection"\nrequired-features = ["disabled"]\n',
                '[[test]]\nname = "renamed"\npath = "tests/selection.rs"\n',
            ):
                path.write_text('[package]\nname = "test-driver"\nversion = "0.1.0"\n' + settings)
                with self.subTest(settings=settings), self.assertRaisesRegex(SystemExit, "driver Cargo test"):
                    self.validator.validate_source_item(root, "cpu-semantic-simulation", tab, {})
            outside = root / "outside.toml"
            outside.write_text('[package]\nname = "test-driver"\nversion = "0.1.0"\n')
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(SystemExit, "regular repository input"):
                self.validator.validate_source_item(root, "cpu-semantic-simulation", tab, {})

    def test_source_driver_requires_unconditional_top_level_ignored_tests(self):
        source = '#[test] #[ignore = "source compilation"] fn selected() {}'
        scan = self.validator.source_driver_test_names
        self.assertEqual(scan(source), ["selected"])
        self.assertEqual(scan("#[allow(dead_code)] " + source), ["selected"])
        self.assertEqual(scan('#![allow(dead_code)]\n' + source), ["selected"])
        for candidate in (
            source.replace('#[ignore = "source compilation"]', ""),
            "#[cfg(any())] " + source, source.replace("#[test]", "#[test] #[cfg(any())]"),
            "#[cfg_attr(feature = \"hidden\", cfg(any()))] " + source,
            "#[custom_attribute] " + source,
            "#![cfg(any())]\n" + source,
            "mod hidden { " + source + " }",
            "fn helper() { " + source + " }",
            "macro_rules! hidden { () => {" + source + "}; }",
        ):
            with self.subTest(candidate=candidate):
                self.assertEqual(scan(candidate), [])

    def test_source_ranges_reject_utf8_splits_and_are_bound_to_fragment_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.rs").write_text("// \u03bb\n#[kernel] fn selected() {}\n", encoding="utf-8")
            tab = {
                "sourcePath": "source.rs", "sourceDigestScope": "displayed",
                "sourceFragmentsSha256": ["0" * 64],
                "sourceItem": {"sourceRanges": [{"byteOffset": 4, "byteLength": 1}]},
            }
            with self.assertRaisesRegex(SystemExit, "not valid UTF-8"):
                self.validator.source_item_fragments(root, tab, "test")
            tab["sourceItem"]["sourceRanges"][0] = {"byteOffset": 0, "byteLength": 3}
            with self.assertRaisesRegex(SystemExit, "fragment digest"):
                self.validator.source_item_fragments(root, tab, "test")

    def whole_file_source_fixture(self, root):
        package = root / "fixture"
        (package / "src").mkdir(parents=True)
        source = "// Retained tutorial header: \u03bb.\n#[kernel]\nfn selected() {}\n".encode("utf-8")
        (package / "src/lib.rs").write_bytes(source)
        manifest = package / "Cargo.toml"
        manifest.write_text('[package]\nname = "source-fixture"\nversion = "0.1.0"\nbuild = false\n[workspace]\n')
        lock = package / "Cargo.lock"
        lock.write_text("version = 4\n")
        driver = root / "crates/test-driver"
        (driver / "tests").mkdir(parents=True)
        (driver / "Cargo.toml").write_text('[package]\nname = "test-driver"\nversion = "0.1.0"\n[workspace]\n')
        (driver / "tests/selection.rs").write_text('#[test]\n#[ignore = "source compilation"]\nfn export_selected() {}\n')
        sources = self.validator.package_rust_sources(root, "fixture/Cargo.toml", "test")
        digest = hashlib.sha256(source).hexdigest()
        tab = copy.deepcopy(self.curriculum_lesson("cpu-semantic-simulation")["codeTabs"][0])
        tab.update(
            sourcePath="fixture/src/lib.rs", sourceDigestScope="file", sourceFragmentsSha256=None,
            sourceCommit="1" * 40, sourceSha256=digest, displayedSha256=digest,
            displayedUtf8Bytes=len(source),
        )
        tab["sourceItem"] = {
            "kind": "source-driver",
            "compilerInput": {
                "packageManifest": "fixture/Cargo.toml",
                "packageManifestSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "cargoLockPath": "fixture/Cargo.lock",
                "cargoLockSha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "sourcePaths": [tab["sourcePath"]],
                "sourceClosureSha256": self.validator.package_source_closure_sha256(root, sources),
                "cargoTarget": {"kind": "lib", "name": "source_fixture", "sourcePath": "src/lib.rs"},
                "defaultFeatures": True,
            },
            "driver": {"package": "test-driver", "target": "selection", "path": "crates/test-driver/tests/selection.rs"},
            "sourceRanges": [{"byteOffset": 0, "byteLength": len(source)}],
            "cases": [{
                "features": [], "kernelSymbol": "selected", "target": "gfx942",
                "displayedFragmentOrdinal": 0, "testFunction": "export_selected",
                "expectation": {"kind": "verified-bundle-export", "bundleVersion": 5},
            }],
        }
        tab["sourceItem"]["contractSha256"] = self.validator.source_item_contract_sha256("whole-file", tab)
        return tab, source

    def test_whole_file_source_driver_retains_header_and_existing_case_scanner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tab, source = self.whole_file_source_fixture(root)
            self.assertEqual(self.validator.source_item_fragments(root, tab, "test"), [source.decode("utf-8")])
            self.assertTrue(source.startswith(b"// Retained tutorial header:"))
            self.assertGreater(len(source), len(source.decode("utf-8")))
            self.assertEqual(self.validator.validate_source_item(root, "whole-file", tab, {}), tab["sourcePath"])
            self.validator.validate_curriculum_tab(tab, 0, "whole-file", "executable", source_items=True)
            with self.assertRaisesRegex(SystemExit, "pending source-item"):
                self.validator.validate_curriculum_tab(tab, 0, "whole-file", "executable")
            self.assertEqual(tab["sourceItemStatus"], "contract-bound")

    def test_source_driver_variant_binding_revalidates_physical_closure_and_cargo_lock(self):
        with tempfile.TemporaryDirectory(prefix="fe2o3-variant-source-") as temporary:
            root = Path(temporary)
            tab, source = self.whole_file_source_fixture(root)
            self.validator.validate_source_item(root, "whole-file", tab, {})
            reference = {"kind": "source-driver-case", "lessonId": "whole-file", "tabOrdinal": 0, "caseOrdinal": 0}
            offset = source.index(b"selected")
            variants = copy.deepcopy(self.original["kernelInventory"]["kernels"][0]["variants"])
            for variant in variants:
                variant.update(status="pending", source=None)
                variant["blocker"]["reason"] = "Constructed source association is pending."
            document = {
                "compilerFixtures": [],
                "curriculum": {"schema": self.validator.CURRICULUM_SCHEMA_V2,
                               "lessons": [{"lessonId": "whole-file", "role": "executable", "codeTabs": [tab]}]},
                "kernelInventory": {
                    "schema": "fe2o3-tutorial-kernel-identities-v1",
                    "kernels": [{"kernelId": "selected", "selections": [reference],
                                 "variants": variants}],
                    "negativeCases": [], "displayItems": [{
                        "lessonId": "whole-file", "tabOrdinal": 0, "functionUtf8Offset": offset,
                        "kernelSymbol": "selected", "classification": "kernel", "kernelIds": ["selected"],
                        "negativeCases": [], "bindingStatus": "source-driver-contract",
                        "reason": "Constructed physical source contract, not a compilation receipt.",
                    }],
                },
            }
            binding = self.bind_source_variant(document, root, "selected")
            result = self.validator.validate_kernel_inventory(document, None, repo_root=root)
            self.assertEqual(result["sourceBoundVariantCount"], 1)
            self.assertEqual(result["sourceBoundPairCount"], 0)
            self.assertIsNone(result["requiredPairCount"])
            self.assertEqual(binding["functionUtf8Offset"], offset)
            self.assertGreater(offset, source.decode("utf-8").index("selected"))
            mutations = (
                (root / tab["sourcePath"], b"\n// changed source\n", "sourceClosureSha256 is stale"),
                (root / "fixture/Cargo.lock", b"\n# changed lock\n", "cargoLockSha256 is stale"),
                (root / "fixture/Cargo.toml", b"\n# changed manifest\n", "packageManifestSha256 is stale"),
            )
            for path, suffix, message in mutations:
                before = path.read_bytes()
                try:
                    path.write_bytes(before + suffix)
                    with self.subTest(path=path), self.assertRaisesRegex(SystemExit, message):
                        self.validator.validate_kernel_inventory(document, None, repo_root=root)
                finally:
                    path.write_bytes(before)
            self.assertEqual(self.validator.validate_kernel_inventory(
                document, None, repo_root=root)["sourceBoundVariantCount"], 1)

    def test_rehashed_whole_file_sources_reject_partial_or_mixed_byte_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, source = self.whole_file_source_fixture(root)
            size = len(source)
            header_end = source.index(b"\n") + 1
            mutations = [
                ("omitted header", lambda tab: tab["sourceItem"].update(sourceRanges=[{"byteOffset": header_end, "byteLength": size - header_end}])),
                ("shifted", lambda tab: tab["sourceItem"]["sourceRanges"][0].update(byteOffset=1)),
                ("truncated", lambda tab: tab["sourceItem"]["sourceRanges"][0].update(byteLength=size - 1)),
                ("long", lambda tab: tab["sourceItem"]["sourceRanges"][0].update(byteLength=size + 1)),
                ("duplicate", lambda tab: tab["sourceItem"]["sourceRanges"].append(copy.deepcopy(tab["sourceItem"]["sourceRanges"][0]))),
                ("missing range", lambda tab: tab["sourceItem"].update(sourceRanges=[])),
                ("Boolean offset", lambda tab: tab["sourceItem"]["sourceRanges"][0].update(byteOffset=False)),
                ("Boolean length", lambda tab: tab["sourceItem"]["sourceRanges"][0].update(byteLength=True)),
                ("extra range field", lambda tab: tab["sourceItem"]["sourceRanges"][0].update(unknown=0)),
                ("fragment metadata", lambda tab: tab.update(sourceFragmentsSha256=[tab["sourceSha256"]])),
                ("empty fragments", lambda tab: tab.update(sourceFragmentsSha256=[])),
                ("mixed scope", lambda tab: tab.update(sourceDigestScope="displayed")),
                ("missing commit", lambda tab: tab.update(sourceCommit=None)),
                ("invalid commit", lambda tab: tab.update(sourceCommit="1" * 39)),
                ("missing source hash", lambda tab: tab.update(sourceSha256=None)),
                ("wrong source hash", lambda tab: tab.update(sourceSha256="0" * 64)),
                ("wrong displayed hash", lambda tab: tab.update(displayedSha256="0" * 64)),
                ("both wrong hashes", lambda tab: tab.update(sourceSha256="0" * 64, displayedSha256="0" * 64)),
                ("displayed count", lambda tab: tab.update(displayedUtf8Bytes=size - 1)),
                ("character count", lambda tab: tab.update(displayedUtf8Bytes=len(source.decode("utf-8")))),
                ("Boolean count", lambda tab: tab.update(displayedUtf8Bytes=True)),
                ("missing case", lambda tab: tab["sourceItem"].update(cases=[])),
                ("wrong fragment ordinal", lambda tab: tab["sourceItem"]["cases"][0].update(displayedFragmentOrdinal=1)),
            ]
            for label, mutate in mutations:
                with self.subTest(label=label):
                    tab = copy.deepcopy(original)
                    mutate(tab)
                    tab["sourceItem"]["contractSha256"] = self.validator.source_item_contract_sha256("whole-file", tab)
                    with self.assertRaises(SystemExit):
                        self.validator.validate_source_item(root, "whole-file", tab, {})
            tab = copy.deepcopy(original)
            body_digest = hashlib.sha256(source[header_end:]).hexdigest()
            tab.update(sourceSha256=body_digest, displayedSha256=body_digest, displayedUtf8Bytes=size - header_end)
            tab["sourceItem"]["sourceRanges"] = [{"byteOffset": header_end, "byteLength": size - header_end}]
            tab["sourceItem"]["contractSha256"] = self.validator.source_item_contract_sha256("whole-file", tab)
            with self.assertRaisesRegex(SystemExit, "every source byte"):
                self.validator.validate_source_item(root, "whole-file", tab, {})

    def test_whole_file_source_rejects_invalid_utf8_even_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tab, source = self.whole_file_source_fixture(root)
            source = b"\xff" + source
            (root / tab["sourcePath"]).write_bytes(source)
            digest = hashlib.sha256(source).hexdigest()
            tab.update(sourceSha256=digest, displayedSha256=digest, displayedUtf8Bytes=len(source))
            tab["sourceItem"]["sourceRanges"] = [{"byteOffset": 0, "byteLength": len(source)}]
            tab["sourceItem"]["contractSha256"] = self.validator.source_item_contract_sha256("whole-file", tab)
            with self.assertRaisesRegex(SystemExit, "not valid UTF-8"):
                self.validator.validate_source_item(root, "whole-file", tab, {})


if __name__ == "__main__":
    unittest.main()

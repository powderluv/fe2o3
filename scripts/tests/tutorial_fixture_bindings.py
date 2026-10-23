#!/usr/bin/env python3
"""Exercise registered fixture displays through the real source-binding parent."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import runpy
import tempfile
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
# Independent physical excerpt starts and name coordinates, not a symbol search.
EXCERPTS = {
    "gfx950-fp4-gemm": (1349, 1449, 1518),
    "gfx950-fp8-gemm": (3774, 3874, 3936),
    "gfx950-fp4-attention": (8718, 8818, 8898),
    "gfx950-fp8-attention": (18512, 18612, 18692),
}
SOURCE_SHA256 = "219691cbccdd6f71ac86a7f54c27f7891e8b022484ca7efffcfc59a12e4b29af"


class FixtureBindingTests(unittest.TestCase):
    expected_displays = 8
    expected_identities = 4
    changed_source_error = "differs from its physical source"

    @classmethod
    def setUpClass(cls):
        path = ROOT / "scripts/validate-tutorial-kernel-manifest.py"
        spec = importlib.util.spec_from_file_location("fixture_binding_parent", path)
        cls.parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.parent)
        cls.original = cls.parent.load_manifest(ROOT / "config/tutorial-kernel-manifest-v1.json")

    def document(self):
        """A real-source eight-display test projection, not the full site census."""
        document = copy.deepcopy(self.original)
        lesson_ids = {name + suffix for name in EXCERPTS for suffix in ("", "-performance-lab")}
        ids = {f"fixture:{name}:{name.replace('-', '_')}_rust" for name in EXCERPTS}
        document["compilerFixtures"] = [
            row for row in document["compilerFixtures"] if row["fixtureId"] in EXCERPTS
        ]
        inventory = document["kernelInventory"]
        inventory["kernels"] = [row for row in inventory["kernels"] if row["kernelId"] in ids]
        inventory["negativeCases"] = []
        inventory["displayItems"] = [
            row for row in inventory["displayItems"] if row["lessonId"] in lesson_ids and row["tabOrdinal"] == 0
        ]
        self.assertEqual(len(inventory["kernels"]), 4)
        self.assertEqual(len(inventory["displayItems"]), 8)
        self.assertTrue(all(row["bindingStatus"] == "fixture-source-contract" for row in inventory["displayItems"]))
        document["curriculum"]["lessons"] = [
            row for row in document["curriculum"]["lessons"] if row["lessonId"] in lesson_ids
        ]
        runtime = {"schema": self.parent.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": []}
        for lesson in document["curriculum"]["lessons"]:
            lesson["codeTabs"] = lesson["codeTabs"][:1]
            tab = lesson["codeTabs"][0]
            physical = (ROOT / tab["sourcePath"]).read_bytes()
            self.assertEqual(hashlib.sha256(physical).hexdigest(), SOURCE_SHA256)
            performance = lesson["lessonId"].endswith("-performance-lab")
            name = lesson["lessonId"].removesuffix("-performance-lab")
            start = EXCERPTS[name][int(performance)]
            if name == "gfx950-fp8-attention":
                # The current display must not reuse the retained historical
                # excerpt identity, even when supplied with current bytes.
                historical_size, historical_digest, size, digest = (
                    (20521, "69650ea2502ee149949d6cbec3e909032ede026be47ca52568bda455b9d9ef2c",
                     10537, "1868eedc608691d53c7f90358540a8546a54af90a1b9bbf6f38b21a38117bb07"),
                    (20421, "0a06ba5894d9afa1504dd036f9a0c8a4b3deb242156f37d5930f047d5ec1d5f1",
                     10437, "db02c7da83c25654b6a87590caff9d84965ac91e4ebbac52cbf6ebaf8aa91001"),
                )[int(performance)]
                self.assertEqual(tab["displayedUtf8Bytes"], size)
                self.assertEqual(tab["displayedSha256"], digest)
                self.assertEqual(tab["sourceSha256"], digest)
                current = physical[start:start + size].decode("utf-8")
                self.assertEqual(hashlib.sha256(current.encode("utf-8")).hexdigest(), digest)
                historical_tab = copy.deepcopy(tab)
                historical_tab.update(displayedUtf8Bytes=historical_size,
                                      displayedSha256=historical_digest,
                                      sourceSha256=historical_digest)
                if historical_tab["sourceFragmentsSha256"] is not None:
                    historical_tab["sourceFragmentsSha256"] = [historical_digest]
                historical_live = {**historical_tab, "displayedCode": current,
                                   "sourceFragments": [current] if tab["sourceFragmentsSha256"] is not None else None}
                # Current bytes cannot acquire the retained historical display binding.
                with self.assertRaisesRegex(SystemExit, "displayed bytes do not match"):
                    self.parent.validate_site_inventory(
                        {"site": runtime["site"], "lessons": [
                            {"lessonId": lesson["lessonId"], "codeTabs": [historical_tab]},
                        ]},
                        {"schema": runtime["schema"], "site": runtime["site"], "lessons": [
                            {"id": lesson["lessonId"], "codeTabs": [historical_live]},
                        ]},
                    )
                if tab["sourceFragmentsSha256"] is not None:
                    self.assertEqual(tab["sourceFragmentsSha256"], [digest])
            excerpt = physical[start:start + tab["displayedUtf8Bytes"]].decode("utf-8")
            row = next(row for row in inventory["displayItems"] if row["lessonId"] == lesson["lessonId"])
            self.assertEqual(start + row["functionUtf8Offset"], EXCERPTS[name][2])
            runtime["lessons"].append({"id": lesson["lessonId"], "codeTabs": [
                {**tab, "displayedCode": excerpt,
                 "sourceFragments": [excerpt] if tab["sourceFragmentsSha256"] is not None else None},
            ]})
        return document, runtime

    def check(self, document, runtime):
        self.parent.validate_site_inventory(document["curriculum"], runtime)
        return self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)

    def test_real_excerpts_bind_without_qualifying_variants(self):
        document, runtime = self.document()
        before = copy.deepcopy((document, runtime))
        result = self.check(document, runtime)
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(result["displayItemCount"], self.expected_displays)
        self.assertEqual(result["knownKernelIdentityCount"], self.expected_identities)
        self.assertEqual(result["sourceBoundVariantCount"], 0)
        self.assertEqual(result["sourceBoundPairCount"], 0)
        self.assertTrue(all(variant["status"] == "pending" and variant["source"] is None
                            for kernel in result["kernelIdentities"] for variant in kernel["variants"]))
        self.assertEqual((document, runtime), before)

    def test_missing_runtime_keeps_required_pair_count_unknown(self):
        document, _ = self.document()
        result = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
        self.assertFalse(result["runtimeCensusValidated"])
        self.assertFalse(result["inventoryComplete"])
        self.assertIsNone(result["requiredPairCount"])
        self.assertEqual(result["sourceBoundPairCount"], 0)

    def test_wrong_registered_kernel_identity_rejects(self):
        document, runtime = self.document()
        row = next(row for row in document["kernelInventory"]["displayItems"]
                   if row["classification"] == "kernel")
        row["kernelIds"] = [next(kernel["kernelId"] for kernel in document["kernelInventory"]["kernels"]
                                 if kernel["kernelId"] not in row["kernelIds"])]
        with self.assertRaisesRegex(SystemExit, "different selection"):
            self.check(document, runtime)

    def test_different_compilation_feature_rejects(self):
        document, runtime = self.document()
        fixtures = document["compilerFixtures"]
        fixtures[0]["compilerInput"]["features"] = fixtures[1]["compilerInput"]["features"]
        fixtures[0]["compilerInput"]["contractSha256"] = self.parent.fixture_input_contract_sha256(fixtures[0])
        with self.assertRaisesRegex(SystemExit, "feature-selected fixture kernel roster differs"):
            self.check(document, runtime)

    def test_stale_coordinate_and_missing_occurrence_reject(self):
        for remove in (False, True):
            document, runtime = self.document()
            rows = document["kernelInventory"]["displayItems"]
            if remove:
                rows.pop()
            else:
                rows[0]["functionUtf8Offset"] += 1
            with self.subTest(remove=remove), self.assertRaisesRegex(SystemExit, "function census"):
                self.check(document, runtime)

    def test_changed_runtime_bytes_reject_even_after_display_digests_are_refreshed(self):
        document, runtime = self.document()
        tab = document["curriculum"]["lessons"][0]["codeTabs"][0]
        live = runtime["lessons"][0]["codeTabs"][0]
        live["displayedCode"] += "\n// not present in the selected physical source\n"
        payload = live["displayedCode"].encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        for entry in (tab, live):
            entry.update(displayedUtf8Bytes=len(payload), displayedSha256=digest, sourceSha256=digest)
            if entry["sourceFragmentsSha256"] is not None:
                entry["sourceFragmentsSha256"] = [digest]
        if live["sourceFragments"] is not None:
            live["sourceFragments"] = [live["displayedCode"]]
        self.parent.validate_site_inventory(document["curriculum"], runtime)
        with self.assertRaisesRegex(SystemExit, self.changed_source_error):
            self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)


class WholeFileFixtureBindingTests(FixtureBindingTests):
    expected_displays = 39
    changed_source_error = "exact current source occurrence"
    # Independent whole-file sizes, digests and every displayed function offset.
    sources = {
        "flash-attention": (
            "gfx942-flash-attention", "flash_attention_general_v1", 34804,
            "72fd0148eb5a29c0747ec6b96616eeb4aed15939b365890846c49e1c1d0fbc50",
            [1174, 1480, 1786, 2428, 2517, 3320, 19160, 19354, 20201, 20526,
             21019, 21799, 23944, 26314, 27580, 29032, 30182, 30982, 31514,
             32759, 33363, 33897, 34062]),
        "gemm-autoresearch": (
            "gfx942-gemm-autoresearch", "gemm_autoresearch_v1", 5194,
            "abae10d3a6657c89358a5fecf72359d120b917d0664f7df51de3f7b61b1b4f54",
            [719, 1383, 4843]),
        "gemm-tiling": (
            "gfx942-tiled-gemm", "tiled_gemm_general_v1", 7111,
            "ef3bbea31da5311bfb5bfb9edb9f43569e602501c079bebb16eb5f0c6b5973b4",
            [797, 1526, 6760]),
        "moe-expert-compute": (
            "gfx942-grouped-expert-moe", "moe_grouped_expert_general_v1", 11320,
            "9e31980d48320d2978823af76465f882a99263137ba828261f605e74afc110a7",
            [915, 1145, 1806, 7767, 7961, 8272, 8691, 9841, 10641, 11173]),
    }

    def document(self):
        document = copy.deepcopy(self.original)
        fixtures = {source[0] for source in self.sources.values()}
        ids = {f"fixture:{source[0]}:{source[1]}" for source in self.sources.values()}
        document["compilerFixtures"] = [
            row for row in document["compilerFixtures"] if row["fixtureId"] in fixtures
        ]
        self.assertEqual(len(document["compilerFixtures"]), 4)
        for fixture in document["compilerFixtures"]:
            self.assertIs(fixture["compilerInput"]["defaultFeatures"], True)
            self.assertEqual(fixture["compilerInput"]["features"], [])
        inventory = document["kernelInventory"]
        inventory["kernels"] = [row for row in inventory["kernels"] if row["kernelId"] in ids]
        inventory["negativeCases"] = []
        inventory["displayItems"] = [
            row for row in inventory["displayItems"]
            if row["lessonId"] in self.sources and row["tabOrdinal"] == 0
        ]
        self.assertEqual(len(inventory["kernels"]), self.expected_identities)
        self.assertEqual(len(inventory["displayItems"]), self.expected_displays)
        helpers = [row for row in inventory["displayItems"] if row["classification"] == "helper"]
        self.assertEqual(len(helpers), 35)
        self.assertTrue(all(row["bindingStatus"] == "not-applicable" and row["kernelIds"] == []
                            for row in helpers))
        document["curriculum"]["lessons"] = [
            row for row in document["curriculum"]["lessons"] if row["lessonId"] in self.sources
        ]
        runtime = {"schema": self.parent.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": []}
        for lesson in document["curriculum"]["lessons"]:
            fixture, symbol, size, digest, offsets = self.sources[lesson["lessonId"]]
            lesson["codeTabs"] = lesson["codeTabs"][:1]
            tab = lesson["codeTabs"][0]
            self.assertEqual(tab["sourcePath"], f"examples/{symbol}/src/kernel.rs")
            self.assertEqual(tab["sourceDigestScope"], "file")
            self.assertIsNone(tab["sourceFragmentsSha256"])
            physical = (ROOT / tab["sourcePath"]).read_bytes()
            self.assertEqual(len(physical), size)
            self.assertEqual(hashlib.sha256(physical).hexdigest(), digest)
            rows = [row for row in inventory["displayItems"] if row["lessonId"] == lesson["lessonId"]]
            if lesson["lessonId"] == "gemm-autoresearch":
                # Current-source test projection; the published snapshot stays historical.
                self.assertEqual(tab["displayedUtf8Bytes"], 4842)
                self.assertEqual(tab["sourceSha256"],
                                 "3199202452896fa59e0e3f7f5e7f6e656636af039d4bf5b0718febc5edf8f9d0")
                tab.update(displayedUtf8Bytes=size, displayedSha256=digest, sourceSha256=digest)
                for row, offset in zip(rows, offsets, strict=True):
                    row["functionUtf8Offset"] = offset
                    if row["classification"] == "kernel":
                        self.assertEqual(row["bindingStatus"], "pending")
                        row["bindingStatus"] = "fixture-source-contract"
            if lesson["lessonId"] == "gemm-tiling":
                # The published current view is source-only; this test projection
                # exercises its exact physical fixture without qualifying the lesson.
                self.assertEqual(tab["displayedUtf8Bytes"], size)
                self.assertEqual(tab["displayedSha256"], digest)
                self.assertEqual(tab["sourceSha256"], digest)
                for row, offset in zip(rows, offsets, strict=True):
                    self.assertEqual(row["functionUtf8Offset"], offset)
                    if row["classification"] == "kernel":
                        self.assertEqual(row["bindingStatus"], "pending")
                        row["bindingStatus"] = "fixture-source-contract"
            self.assertEqual([row["functionUtf8Offset"] for row in rows], offsets)
            kernels = [row for row in rows if row["classification"] == "kernel"]
            self.assertEqual(len(kernels), 1)
            self.assertEqual(kernels[0]["kernelIds"], [f"fixture:{fixture}:{symbol}"])
            self.assertEqual(kernels[0]["bindingStatus"], "fixture-source-contract")
            runtime["lessons"].append({"id": lesson["lessonId"], "codeTabs": [
                {**tab, "displayedCode": physical.decode("utf-8"), "sourceFragments": None},
            ]})
        return document, runtime

    def historical_autoresearch_document(self):
        document = copy.deepcopy(self.original)
        fixture_id = "gfx942-gemm-autoresearch"
        identity = "fixture:gfx942-gemm-autoresearch:gemm_autoresearch_v1"
        document["compilerFixtures"] = [row for row in document["compilerFixtures"]
                                        if row["fixtureId"] == fixture_id]
        inventory = document["kernelInventory"]
        inventory["kernels"] = [row for row in inventory["kernels"] if row["kernelId"] == identity]
        inventory["negativeCases"] = []
        inventory["displayItems"] = [row for row in inventory["displayItems"]
                                    if row["lessonId"] == "gemm-autoresearch" and row["tabOrdinal"] == 0]
        document["curriculum"]["lessons"] = [row for row in document["curriculum"]["lessons"]
                                           if row["lessonId"] == "gemm-autoresearch"]
        lesson = document["curriculum"]["lessons"][0]
        lesson["codeTabs"] = lesson["codeTabs"][:1]
        tab = lesson["codeTabs"][0]
        historical = (ROOT / "scripts/tests/fixtures/gemm-autoresearch-winning-0e.rs.txt").read_bytes()
        self.assertEqual(len(historical), 4842)
        self.assertEqual(hashlib.sha256(historical).hexdigest(),
                         "3199202452896fa59e0e3f7f5e7f6e656636af039d4bf5b0718febc5edf8f9d0")
        runtime = {"schema": self.parent.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": [
                       {"id": "gemm-autoresearch", "codeTabs": [
                           {**tab, "displayedCode": historical.decode("utf-8"), "sourceFragments": None}]}]}
        self.parent.validate_site_inventory(document["curriculum"], runtime)
        return document, runtime

    def test_historical_autoresearch_pending_preserves_source_and_identity(self):
        document, runtime = self.historical_autoresearch_document()
        tab = document["curriculum"]["lessons"][0]["codeTabs"][0]
        self.assertEqual(tab["sourceCommit"], "9006001157e2c3062e44088634e467b0f8963ee0")
        self.assertIsNone(tab["sourceItem"])
        self.assertEqual(tab["sourceItemStatus"], "pending")
        rows = document["kernelInventory"]["displayItems"]
        self.assertEqual([row["functionUtf8Offset"] for row in rows], [712, 1380, 4491])
        kernel = next(row for row in rows if row["classification"] == "kernel")
        self.assertEqual(kernel["kernelIds"], ["fixture:gfx942-gemm-autoresearch:gemm_autoresearch_v1"])
        self.assertEqual(kernel["bindingStatus"], "pending")
        self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)

    def test_historical_autoresearch_cannot_claim_current_fixture(self):
        document, runtime = self.historical_autoresearch_document()
        kernel = next(row for row in document["kernelInventory"]["displayItems"]
                      if row["classification"] == "kernel")
        kernel["bindingStatus"] = "fixture-source-contract"
        with self.assertRaisesRegex(SystemExit, "exact current source occurrence"):
            self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)

    def test_historical_autoresearch_modified_bytes_do_not_reuse_old_digest(self):
        document, runtime = self.historical_autoresearch_document()
        runtime["lessons"][0]["codeTabs"][0]["displayedCode"] += "\n"
        with self.assertRaises(SystemExit):
            self.parent.validate_site_inventory(document["curriculum"], runtime)

    def test_different_compilation_feature_rejects(self):
        document, runtime = self.document()
        fixture = next(row for row in document["compilerFixtures"]
                       if row["fixtureId"] == "gfx942-tiled-gemm")
        fixture["compilerInput"]["features"] = ["kernel-simt-gemm-general"]
        fixture["compilerInput"]["contractSha256"] = self.parent.fixture_input_contract_sha256(fixture)
        with self.assertRaisesRegex(SystemExit, "feature-selected fixture kernel roster differs"):
            self.check(document, runtime)


class SystemsFixtureBindingTests(FixtureBindingTests):
    expected_displays = 14
    expected_identities = 11
    # Independent physical spans from the current source, not symbol searches.
    spans = {
        "advanced-moe": [(1177, 14201), (14203, 27794), (27796, 29869)],
        "speculative-mtp-verification": [(29871, 37997)],
        "ngram-embedding-gather": [(37999, 43380)],
        "muon-optimizer": [(43382, 46252), (46254, 52659)],
        "moe-route-performance-lab": [(1177, 14201)],
        "moe-expert-performance-lab": [(14203, 27794)],
        "expert-combine-performance-lab": [(27796, 29869)],
        "speculative-performance-lab": [(29871, 37997)],
        "ngram-performance-lab": [(37999, 43380)],
        "gradient-stage-performance-lab": [(43382, 46252)],
        "muon-performance-lab": [(46254, 52659)],
    }
    coordinates = {
        "gfx950_moe_route_fp4_t16_e4_k2_v1": 1586,
        "gfx950_moe_expert_rank_fp4_fp8_v1": 14796,
        "gfx950_combine_expert_ranks_v1": 28347,
        "gfx950_speculative_transaction_v1": 30537,
        "gfx950_qwen_ngram_gather_v1": 38579,
        "gfx950_stage_gradient_shard_v1": 43942,
        "gfx950_muon_update_4x4_v1": 46810,
    }
    fixture_names = {
        "combine-expert-ranks", "moe-expert-rank-expert-serial", "moe-expert-rank",
        "moe-route", "muon-update-broadcast16", "muon-update",
        "qwen-ngram-gather-reverse-probe", "qwen-ngram-gather",
        "speculative-transaction-recompute-prefix", "speculative-transaction",
        "stage-gradient-shard",
    }

    def document(self):
        """A real-source systems projection; the full Vite census is separate."""
        document = copy.deepcopy(self.original)
        lessons = {"gfx950-" + name for name in self.spans}
        fixtures = {"gfx950-" + name for name in self.fixture_names}
        document["compilerFixtures"] = [
            row for row in document["compilerFixtures"] if row["fixtureId"] in fixtures
        ]
        inventory = document["kernelInventory"]
        inventory["kernels"] = [
            row for row in inventory["kernels"] if any(
                ref["kind"] == "fixture" and ref["fixtureId"] in fixtures for ref in row["selections"])
        ]
        # This eleven-selection display projection intentionally excludes paired
        # counterpart identities. Its source-pair bindings are tested by the full
        # manifest suite; retain this projection's independent pending role.
        for kernel in inventory["kernels"]:
            for variant in kernel["variants"]:
                variant.update(status="pending", source=None)
        inventory["negativeCases"] = []
        inventory["displayItems"] = [
            row for row in inventory["displayItems"] if row["lessonId"] in lessons and row["tabOrdinal"] == 0
        ]
        self.assertEqual(len(inventory["kernels"]), 11)
        self.assertEqual(len(inventory["displayItems"]), 14)
        self.assertTrue(all(row["bindingStatus"] == "fixture-source-contract" for row in inventory["displayItems"]))
        document["curriculum"]["lessons"] = [
            row for row in document["curriculum"]["lessons"] if row["lessonId"] in lessons
        ]
        runtime = {"schema": self.parent.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": []}
        source_path = "examples/gfx950_advanced_systems/src/kernel.rs"
        physical = (ROOT / source_path).read_bytes()
        self.assertEqual(len(physical), 60951)
        self.assertEqual(hashlib.sha256(physical).hexdigest(),
                         "45b9a7e5198bf274f6814438a230ec22405fde713b39f080bb4045c303f671ab")
        for lesson in document["curriculum"]["lessons"]:
            lesson["codeTabs"] = lesson["codeTabs"][:1]
            tab = lesson["codeTabs"][0]
            self.assertEqual(tab["sourcePath"], source_path)
            spans = self.spans[lesson["lessonId"].removeprefix("gfx950-")]
            parts = [physical[start:end].decode("utf-8") for start, end in spans]
            # Current-source test projection; historical checked-in lesson and
            # display records remain unchanged by this fixture-only refresh.
            joined = "\n\n".join(parts).encode("utf-8")
            digest = hashlib.sha256(joined).hexdigest()
            tab.update(
                sourceCommit="1da2f6206b1edc579b7d6f222b6794da4e7af4ff",
                displayedUtf8Bytes=len(joined), displayedSha256=digest,
                sourceSha256=digest, sourceDigestScope="displayed",
                sourceFragmentsSha256=[
                    hashlib.sha256(part.encode("utf-8")).hexdigest() for part in parts
                ],
            )
            for row in inventory["displayItems"]:
                if row["lessonId"] != lesson["lessonId"]:
                    continue
                physical_offset = self.coordinates[row["kernelSymbol"]]
                displayed_offset = 0
                for lower, upper in spans:
                    if lower <= physical_offset < upper:
                        row["functionUtf8Offset"] = displayed_offset + physical_offset - lower
                        break
                    displayed_offset += upper - lower + 2
                else:
                    self.fail("canonical function coordinate is outside its exact physical span")
            for row in inventory["displayItems"]:
                if row["lessonId"] != lesson["lessonId"]:
                    continue
                offset = row["functionUtf8Offset"]
                physical_offsets = []
                for start, end in spans:
                    if 0 <= offset < end - start:
                        physical_offsets.append(start + offset)
                    offset -= end - start + 2
                self.assertEqual(physical_offsets, [self.coordinates[row["kernelSymbol"]]])
            runtime["lessons"].append({"id": lesson["lessonId"], "codeTabs": [
                {**tab, "displayedCode": "\n\n".join(parts),
                 "sourceFragments": parts if tab["sourceFragmentsSha256"] is not None else None},
            ]})
        return document, runtime

    def test_feature_alternatives_remain_distinct_despite_identical_physical_function(self):
        document, runtime = self.document()
        result = self.check(document, runtime)
        by_symbol = {}
        for kernel in result["kernelIdentities"]:
            symbol = kernel["selections"][0]["kernelSymbol"]
            by_symbol.setdefault(symbol, []).append(kernel["selectionSha256"])
        self.assertEqual(sorted(map(len, by_symbol.values())), [1, 1, 1, 2, 2, 2, 2])
        self.assertTrue(all(len(values) == len(set(values)) for values in by_symbol.values()))


class AttentionCfgSelectionTests(unittest.TestCase):
    """Bind physical source declarations, never Rust macro expansion or execution."""

    base_features = (
        "kernel-kda-decode", "kernel-kda-prefill", "kernel-content-sparse-attention",
        "kernel-deepseek-sparse-attention", "kernel-compressed-hybrid-attention",
        "kernel-attnres-aggregate", "kernel-four-branch-residual", "kernel-mhc-sinkhorn-mix",
    )
    # Independent physical coordinates include feature-distinct same-name entries.
    coordinates = {
        "gfx950-attnres-aggregate": ("kernel.rs", 66487),
        "gfx950-attnres-aggregate-simt": ("kernel.rs", 66487),
        "gfx950-attnres-aggregate-tile": ("kernel_tile_mixing.rs", 760),
        "gfx950-four-branch-residual-simt": ("kernel.rs", 70361),
        "gfx950-four-branch-residual-tile": ("kernel_tile_mixing.rs", 3060),
        "gfx950-mhc-sinkhorn-mix-simt": ("kernel.rs", 73476),
        "gfx950-mhc-sinkhorn-mix-tile": ("kernel_tile_mixing.rs", 4836),
        "gfx950-attnres-aggregate-explicit-reuse": ("ablation.rs", 1111),
        "gfx950-compressed-hybrid-attention": ("kernel.rs", 55707),
        "gfx950-compressed-hybrid-attention-division-baseline": ("kernel.rs", 55707),
        "gfx950-content-sparse-attention": ("kernel.rs", 24166),
        "gfx950-content-sparse-attention-reciprocal-reuse": ("kernel.rs", 24166),
        "gfx950-deepseek-sparse-attention": ("kernel.rs", 42341),
        "gfx950-four-branch-residual": ("kernel.rs", 70361),
        "gfx950-four-branch-residual-explicit": ("ablation.rs", 3730),
        "gfx950-kda-decode": ("kernel.rs", 10351),
        "gfx950-kda-decode-baseline": ("kda_baseline.rs", 1610),
        "gfx950-kda-prefill": ("kernel.rs", 15199),
        "gfx950-kda-prefill-baseline": ("kda_baseline.rs", 4908),
        "gfx950-mhc-sinkhorn-mix": ("kernel.rs", 73476),
        "gfx950-mhc-sinkhorn-mix-scalar": ("ablation.rs", 5895),
        "gfx950-compressed-hybrid-attention-simt": ("kernel_simt_compressed_hybrid.rs", 537),
        "gfx950-compressed-hybrid-attention-tile": ("kernel.rs", 55707),
        "gfx950-content-sparse-attention-simt": ("kernel_simt_content_sparse.rs", 557),
        "gfx950-content-sparse-attention-tile": ("kernel.rs", 24166),
        "gfx950-deepseek-sparse-attention-simt": ("kernel.rs", 42341),
        "gfx950-deepseek-sparse-attention-tile": ("kernel.rs", 88905),
        "gfx950-kda-decode-simt": ("kernel.rs", 10351),
        "gfx950-kda-decode-tile": ("kernel.rs", 80355),
        "gfx950-kda-prefill-simt": ("kernel.rs", 15199),
        "gfx950-kda-prefill-tile": ("kernel.rs", 84152),
    }

    @classmethod
    def setUpClass(cls):
        for name, filename in (("parent", "validate-tutorial-kernel-manifest.py"),
                               ("identities", "tutorial_kernel_identities.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            setattr(cls, name, module)
        package = ROOT / "examples/gfx950_advanced_attention"
        cls.cargo = tomllib.loads((package / "Cargo.toml").read_text())
        cls.library = (package / "src/lib.rs").read_text()
        cls.kernel = (package / "src/kernel.rs").read_text()
        cls.dimensions = (package / "src/dimensions.rs").read_text()
        cls.original = cls.parent.load_manifest(ROOT / "config/tutorial-kernel-manifest-v1.json")

    def declarations(self, source, direct=(), budget=None):
        enabled = self.parent.cargo_feature_closure(self.cargo, list(direct), False, "attention cfg test")
        code = self.parent._rust_code_without_comments_and_literals(source)
        pairs = self.parent._rust_delimiters(code)
        return self.identities._fixture_declarations(
            source, set(enabled), self.parent.ordinary_rust_function_items,
            lambda _: (code, pairs), budget if budget is not None else self.identities._Budget(4096))

    def test_real_library_selects_each_single_base_feature(self):
        for feature in self.base_features:
            with self.subTest(feature=feature):
                self.assertEqual(self.declarations(self.library, [feature]),
                             ([], [("kernel", None), ("dimensions", None)]))
                self.assertEqual(self.declarations(self.dimensions, [feature]), ([], []))

    def test_real_library_rejects_zero_and_each_pair_of_base_features(self):
        cases = [(), *itertools.combinations(self.base_features, 2), self.base_features]
        for features in cases:
            with self.subTest(features=features), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "unsupported fixture item or module selection"):
                self.declarations(self.library, features)

    def test_real_library_uses_transitive_feature_aliases(self):
        aliases = {
            "kernel-kda-decode-tile": None,
            "kernel-kda-prefill-tile": None,
            "kernel-deepseek-sparse-attention-tile": None,
            "kernel-content-sparse-attention-simt": "kernel_simt_content_sparse",
            "kernel-compressed-hybrid-attention-simt": "kernel_simt_compressed_hybrid",
            "kernel-kda-decode-baseline-v1": "kda_baseline",
            "kernel-kda-prefill-baseline-v1": "kda_baseline",
            "kernel-content-sparse-attention-reciprocal-reuse-v1": "ablation",
            "kernel-deepseek-sparse-attention-leader-exp-v1": None,
            "kernel-compressed-hybrid-attention-division-baseline-v1": "ablation",
            "kernel-attnres-aggregate-explicit-reuse-v1": "ablation",
            "kernel-attnres-aggregate-simt": None,
            "kernel-attnres-aggregate-tile": "kernel_tile_mixing",
            "kernel-four-branch-residual-simt": None,
            "kernel-four-branch-residual-tile": "kernel_tile_mixing",
            "kernel-mhc-sinkhorn-mix-simt": None,
            "kernel-mhc-sinkhorn-mix-tile": "kernel_tile_mixing",
            "kernel-four-branch-residual-explicit-v1": "ablation",
            "kernel-mhc-sinkhorn-mix-scalar-v1": "ablation",
        }
        for alias, module in aliases.items():
            expected = [module, "kernel", "dimensions"] if module else ["kernel", "dimensions"]
            if alias in ("kernel-content-sparse-attention-simt", "kernel-compressed-hybrid-attention-simt"):
                expected = ["kernel", module, "dimensions"]
            base = self.cargo["features"][alias]
            for direct in ([alias], [alias, *base]):
                with self.subTest(direct=direct):
                    self.assertEqual(self.declarations(self.library, direct),
                                 ([], [(module, None) for module in expected]))
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "unsupported fixture item"):
            self.declarations(self.library, ["kernel-kda-decode-baseline-v1",
                                             "kernel-kda-prefill-baseline-v1"])

    def test_mixing_modes_reject_ambiguous_feature_pairs(self):
        for name, ablation in [('attnres-aggregate', 'explicit-reuse-v1'), ('four-branch-residual', 'explicit-v1'), ('mhc-sinkhorn-mix', 'scalar-v1')]:
            for first, second in (("simt", "tile"), ("simt", ablation), ("tile", ablation)):
                with self.subTest(kernel=name, modes=(first, second)), self.assertRaisesRegex(
                        self.identities.KernelInventoryError, "unsupported fixture item"):
                    self.declarations(self.library, ["kernel-" + name + "-" + first, "kernel-" + name + "-" + second])

    def test_real_kernel_macro_definitions_preserve_selected_functions(self):
        for feature in self.base_features:
            with self.subTest(feature=feature):
                functions, modules = self.declarations(self.kernel, [feature])
                fixture_id = "gfx950-" + feature.removeprefix("kernel-")
                self.assertEqual(len(functions), 1)
                self.assertEqual(functions[0]["functionUtf8Offset"], self.coordinates[fixture_id][1])
                self.assertEqual(modules, [])

    def source_document(self, fixture_id):
        document = copy.deepcopy(self.original)
        fixture = next(row for row in document["compilerFixtures"] if row["fixtureId"] == fixture_id)
        symbol, = fixture["compilerInput"]["kernelSymbols"]
        selection = {"kind": "fixture", "fixtureId": fixture_id, "kernelSymbol": symbol}
        kernel = next(row for row in document["kernelInventory"]["kernels"]
                      if selection in row["selections"])
        filename, offset = self.coordinates[fixture_id]
        path = "examples/gfx950_advanced_attention/src/" + filename
        binding = {
            "implementationKernelId": kernel["kernelId"], "selection": kernel["selections"][0],
            "selectionSha256": self.identities._selection_identity(
                fixture["compilerInput"], symbol, self.identities._Budget(4096)),
            "sourcePath": path, "sourceSha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "functionUtf8Offset": offset,
        }
        kernel["variants"][1 if fixture_id.endswith("-tile") else 0].update(status="source-bound", source=binding)
        return document, fixture, binding

    def test_all_registered_attention_selections_use_the_authenticated_parent(self):
        registered = {row["fixtureId"] for row in self.original["compilerFixtures"]
                      if row["compilerInput"]["packageManifest"]
                      == "examples/gfx950_advanced_attention/Cargo.toml"}
        self.assertEqual(registered, self.coordinates.keys())
        identities = {}
        for fixture_id in sorted(registered):
            with self.subTest(fixture_id=fixture_id):
                document, fixture, binding = self.source_document(fixture_id)
                before = copy.deepcopy(document)
                result = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
                self.assertEqual(document, before)
                expected = sum(variant["source"] is not None
                               for row in document["kernelInventory"]["kernels"] for variant in row["variants"])
                self.assertEqual(result["sourceBoundVariantCount"], expected)
                self.assertEqual(result["sourceBoundPairCount"], 31)
                self.assertFalse(result["inventoryComplete"])
                self.assertIsNone(result["requiredPairCount"])
                symbol, = fixture["compilerInput"]["kernelSymbols"]
                identities.setdefault(symbol, []).append(binding["selectionSha256"])
        self.assertEqual(sorted(map(len, identities.values())), [3, 4, 4, 4, 4, 4, 4, 4])
        alias_symbols = {
            "gfx950_kda_decode", "gfx950_kda_chunkwise_prefill",
            "gfx950_content_sparse_attention", "gfx950_deepseek_sparse_attention",
            "gfx950_compressed_hybrid_attention",
        }
        for symbol, values in identities.items():
            self.assertEqual(len(values) - len(set(values)), int(symbol in alias_symbols))
        for stem in ("kda-decode", "kda-prefill", "content-sparse-attention",
                     "deepseek-sparse-attention", "compressed-hybrid-attention"):
            simt = self.source_document("gfx950-" + stem + "-simt")[2]
            tile = self.source_document("gfx950-" + stem + "-tile")[2]
            self.assertNotEqual(simt["selectionSha256"], tile["selectionSha256"])

    def test_same_named_host_fallback_cannot_replace_the_selected_source(self):
        document, _, binding = self.source_document("gfx950-attnres-aggregate")
        binding["functionUtf8Offset"] = 68441
        with self.assertRaisesRegex(SystemExit, "exact current source occurrence"):
            self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)

    def test_cached_macro_and_helper_bytes_must_match_the_physical_closure(self):
        for before, after in (("let bits = $value;", "let bits = $value ^ 1;"),
                              ("let mut result = 1.0_f32;", "let mut result = 2.0_f32;")):
            with self.subTest(before=before):
                document, fixture, _ = self.source_document("gfx950-attnres-aggregate")
                cache = {}
                self.parent.validate_compiler_input(ROOT, fixture, "macro closure test", cache)
                package = cache[fixture["compilerInput"]["packageManifest"]]
                sources = package["packageSources"]
                index = next(index for index, (path, _) in enumerate(sources)
                             if path.as_posix().endswith("gfx950_advanced_attention/src/kernel.rs"))
                path, source = sources[index]
                self.assertIn(before, source)
                package["packageSources"] = [*sources[:index], (path, source.replace(before, after)),
                                             *sources[index + 1:]]
                with self.assertRaisesRegex(SystemExit, "physical source differs from its validated closure"):
                    self.parent.validate_kernel_inventory(document, None, repo_root=ROOT, package_cache=cache)

    def test_macro_bodies_are_inert_even_with_apparent_nested_kernel_items(self):
        for definition in (
            "macro_rules! local { () => { #[kernel] fn visible() {} }; }",
            "macro_rules /* outer /* inner */ end */ ! /* gap */ local {\n"
            " () => { mod nested { #[kernel] fn fake() {} } }; }",
            "macro_rules! local { ($value:expr) => {{ other!($value); helper($value) }}; };",
        ):
            source = "// UTF-8: \u03bb\n" + definition + "\n#[kernel] fn visible() { local!(helper()); }"
            with self.subTest(definition=definition):
                functions, modules = self.declarations(source)
                self.assertEqual(functions, [{"kernelSymbol": "visible", "attributedKernel": True,
                                             "functionUtf8Offset": source.encode().rindex(b"visible")}])
                self.assertEqual(modules, [])

    def test_macro_definitions_preserve_cfg_without_selecting_body_tokens(self):
        definition = "macro_rules! local { () => { #[kernel] fn fake() {} }; }"
        for cfg in ("test", "not(test)", 'feature = "kernel-kda-decode"'):
            for features in ((), ("kernel-kda-decode",)):
                source = f"#[cfg({cfg})] {definition}\n#[kernel] fn visible() {{}}"
                with self.subTest(cfg=cfg, features=features):
                    functions, modules = self.declarations(source, features)
                    self.assertEqual([row["kernelSymbol"] for row in functions], ["visible"])
                    self.assertEqual(modules, [])

    def test_kernel_attributes_on_macro_definitions_reject_even_when_cfg_inactive(self):
        for attributes in ("#[kernel]", "#[cfg(test)] #[kernel]",
                           "#[cfg_attr(not(test), kernel)]", "#[kernel] #[kernel]"):
            with self.subTest(attributes=attributes), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "kernel attribute on an inert fixture macro"):
                self.declarations(attributes + " macro_rules! local { () => {} }")

    def test_only_exact_braced_definition_headers_are_inert(self):
        for item in (
            'macro_rules! "hidden" local {}', 'macro_rules! local "hidden" {}',
            'macro_rules "hidden" ! local {}', 'macro_rules! r#"hidden"# local {}',
            "macro_rules! local() {};", "macro_rules! local [() => {}];",
            "macro_rules! local (() => {});", "pub macro_rules! local {}",
            "macro_rules! {}", "macro_rulesX! local {}",
        ):
            with self.subTest(item=item), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "unsupported fixture item"):
                self.declarations(item + "\n#[kernel] fn visible() {}")
        for item in ("macro_rules! r#local {}", "macro_rules! local extra {}", "macro_rules! local;"):
            with self.subTest(item=item), self.assertRaisesRegex(SystemExit, "unterminated macro_rules body"):
                self.declarations(item + "\n#[kernel] fn visible() {}")

    def test_item_calls_includes_derives_and_actual_nested_kernels_stay_unsupported(self):
        definition = "macro_rules! local { () => {}; }\n"
        for item, error in (
            ("local!();", "unsupported fixture item"),
            ("local! { #[kernel] fn fake() {} }", "nested fixture kernel"),
            ("crate::local!();", "unsupported fixture item"),
            ('include!("body.rs");', "unsupported fixture item"),
            ("#[derive(Custom)] struct Value;", "builtin derive"),
            ("#[macro_export] macro_rules! other {}", "selection attribute"),
            ("#[unknown] macro_rules! other {}", "selection attribute"),
            ("#[cfg(unknown)] macro_rules! other {}", "cfg predicate"),
            ("fn outer() { #[kernel] fn fake() {} }", "nested fixture kernel"),
        ):
            with self.subTest(item=item), self.assertRaisesRegex(self.identities.KernelInventoryError, error):
                self.declarations(definition + item)

    def test_inert_definitions_keep_exact_record_and_source_byte_budgets(self):
        source = "macro_rules! local { () => { #[kernel] fn fake() {} }; }\n#[kernel] fn visible() {}"
        budget = self.identities._Budget(3)  # One scanned function and two physical items.
        self.declarations(source, budget=budget)
        self.assertEqual(budget.used, 3)
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "record bound"):
            self.declarations(source, budget=self.identities._Budget(2))
        for byte_limit, succeeds in ((len(source.encode()), True), (len(source.encode()) - 1, False)):
            with patch.object(self.identities, "MAX_TEXT_BYTES", byte_limit):
                if succeeds:
                    self.declarations(source)
                else:
                    with self.assertRaisesRegex(self.identities.KernelInventoryError, "byte bound"):
                        self.declarations(source)


class LiteralIncludeSelectionTests(unittest.TestCase):
    """Authenticate inert includes without expanding their tokens into declarations."""

    library = "pkg/src/lib.rs"
    definition = "macro_rules! body { () => { #[kernel] fn fake() {} }; }\n"
    kernel = "#[kernel] fn visible() { body!(); }\n"

    @classmethod
    def setUpClass(cls):
        for name, filename in (("parent", "validate-tutorial-kernel-manifest.py"),
                               ("identities", "tutorial_kernel_identities.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            setattr(cls, name, module)
        cls.original = cls.parent.load_manifest(ROOT / "config/tutorial-kernel-manifest-v1.json")

    def select(self, sources, *, budget=None):
        def syntax(source):
            code = self.parent._rust_code_without_comments_and_literals(source)
            return code, self.parent._rust_delimiters(code)
        return self.identities._fixture_selection(
            {"compilerInput": {"kernelSymbols": ["visible"]}},
            lambda _: (self.library, sources, []), self.parent.ordinary_rust_function_items,
            syntax, budget if budget is not None else self.identities._Budget(4096))

    def test_catalog_attribute_capacity_has_an_exact_finite_boundary(self):
        self.assertEqual(self.identities.MAX_ATTRIBUTE_VISITS, 8192)
        budget = self.identities._Budget(4096, attribute_limit=self.identities.MAX_ATTRIBUTE_VISITS)
        for _ in range(8192):
            budget.visit_attribute()
        self.assertEqual(budget.attribute_visits, 8192)
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "aggregate visit bound"):
            budget.visit_attribute()
        self.assertEqual(budget.attribute_visits, 8192)
        self.assertEqual(budget.maximum, 4096)
        self.assertEqual(self.identities.MAX_SOURCE_ITEM_VISITS, 8192)
        self.assertEqual(self.identities.MAX_RUNTIME_BYTES, 16 * 1024 * 1024)

    def sources(self, prefix='include!("body.rs");\n', body=None):
        return {self.library: prefix + self.kernel,
                "pkg/src/body.rs": self.definition if body is None else body}

    def document(self):
        document = copy.deepcopy(self.original)
        fixture = next(row for row in document["compilerFixtures"]
                       if row["fixtureId"] == "gfx942-typed-vecadd-source")
        kernel = next(row for row in document["kernelInventory"]["kernels"]
                      if row["kernelId"] == "fixture:gfx942-typed-vecadd-source:vecadd")
        binding = {
            "implementationKernelId": kernel["kernelId"], "selection": kernel["selections"][0],
            "selectionSha256": self.identities._selection_identity(
                fixture["compilerInput"], "vecadd", self.identities._Budget(4096)),
            "sourcePath": "examples/vecadd/src/lib.rs",
            "sourceSha256": "96a6d5774c0d36661540829025194c918100abcce53fed91cae3c909aea67345",
            "functionUtf8Offset": 299,
        }
        kernel["variants"][0].update(status="source-bound", source=binding)
        return document, fixture, binding

    def test_real_vecadd_binds_only_the_physical_kernel_through_authenticated_parent(self):
        document, _, binding = self.document()
        before = copy.deepcopy(document)
        result = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
        self.assertEqual(document, before)
        expected = sum(variant["source"] is not None
                       for row in document["kernelInventory"]["kernels"] for variant in row["variants"])
        self.assertEqual(result["sourceBoundVariantCount"], expected)
        self.assertEqual(result["sourceBoundPairCount"], 31)
        self.assertFalse(result["inventoryComplete"])
        self.assertIsNone(result["requiredPairCount"])
        self.assertEqual((ROOT / binding["sourcePath"]).read_bytes()[299:305], b"vecadd")

    def test_real_vecadd_included_bytes_remain_part_of_authenticated_closure(self):
        document, fixture, _ = self.document()
        cache = {}
        self.parent.validate_compiler_input(ROOT, fixture, "include closure test", cache)
        package = cache[fixture["compilerInput"]["packageManifest"]]
        sources = package["packageSources"]
        index = next(i for i, (path, _) in enumerate(sources) if path.name == "vecadd_body.rs")
        path, source = sources[index]
        self.assertIn("$add!($a[i], $b[i])", source)
        package["packageSources"] = [
            *sources[:index], (path, source.replace("$add!($a[i], $b[i])", "$a[i]")), *sources[index + 1:]]
        with self.assertRaisesRegex(SystemExit, "physical source differs from its validated closure"):
            self.parent.validate_kernel_inventory(document, None, repo_root=ROOT, package_cache=cache)

    def test_exact_quoted_and_raw_literals_preserve_physical_offsets(self):
        for literal in ('"body.rs"', 'r"body.rs"', 'r###"body.rs"###'):
            sources = self.sources(f"// UTF-8: \u03bb\ninclude /* gap */ !({literal});\n")
            with self.subTest(literal=literal):
                selected = self.select(sources)
                self.assertEqual(set(selected), {"visible"})
                path, offset, source = selected["visible"]
                self.assertEqual(path, self.library)
                self.assertEqual(source.encode()[offset:offset + 7], b"visible")
                self.assertEqual(source, sources[self.library])

    def test_raw_literal_stops_at_first_delimiter_despite_authenticated_filename(self):
        for hashes in ("", "#", "##"):
            argument = f'r{hashes}"first.rs"{hashes} r{hashes}"second.rs"{hashes}'
            swallowed = f'first.rs"{hashes} r{hashes}"second.rs'
            sources = self.sources(f"include!({argument});\n")
            sources["pkg/src/" + swallowed] = self.definition
            sources["pkg/src/first.rs"] = self.definition
            with self.subTest(hashes=hashes), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "trailing tokens after its literal path"):
                self.select(sources)

    def test_raw_literal_can_name_authenticated_paths_with_embedded_quotes(self):
        for hashes in ("#", "##"):
            source = f'include!(r{hashes}"body"quote.rs"{hashes} /* trailing trivia */);\n' + self.kernel
            sources = {self.library: source, 'pkg/src/body"quote.rs': self.definition}
            with self.subTest(hashes=hashes):
                self.assertEqual(self.select(sources)["visible"][0], self.library)

    def test_nested_includes_are_relative_to_containing_file_not_module_context(self):
        child = 'include!("shared/body.rs");\n' + self.kernel
        sources = {self.library: "mod child;", "pkg/src/child.rs": child,
                   "pkg/src/shared/body.rs": 'include!("leaf.rs");',
                   "pkg/src/shared/leaf.rs": self.definition,
                   "pkg/src/leaf.rs": "fn wrong_physical_parent() {}"}
        self.assertEqual(self.select(sources)["visible"][0], "pkg/src/child.rs")
        del sources["pkg/src/shared/leaf.rs"]
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "missing fixture module"):
            self.select(sources)

    def test_cycles_repeats_missing_and_ambiguous_module_paths_reject(self):
        cases = [
            self.sources('include!("lib.rs");\n'),
            self.sources('include!("missing.rs");\n'),
            self.sources('include!("body.rs"); include!("body.rs");\n'),
            self.sources(body='include!("body.rs");'),
            {**self.sources(body='include!("other.rs");'), "pkg/src/other.rs": 'include!("body.rs");'},
            {self.library: "mod child;", "pkg/src/child.rs": self.kernel,
             "pkg/src/child/mod.rs": self.kernel},
        ]
        for sources in cases:
            with self.subTest(sources=sources), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "ambiguous or missing"):
                self.select(sources)

    def test_nonliteral_noncanonical_qualified_and_generated_paths_reject(self):
        for argument in ('concat!("body", ".rs")', 'env!("OUT_DIR")', '"/body.rs"', '"../body.rs"',
                         '"./body.rs"', '"sub//body.rs"', '""', '"body\\x2ers"', '"body.rs",'):
            with self.subTest(argument=argument), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "fixture include"):
                self.select(self.sources(f"include!({argument});\n"))
        for directive in ('std::include!("body.rs");', 'include!{"body.rs"}', 'include!["body.rs"];'):
            with self.subTest(directive=directive), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "fixture"):
                self.select(self.sources(directive))
        # The authenticated package closure never supplies generated target/ members.
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "missing fixture module"):
            self.select(self.sources('include!("target/generated.rs");\n'))

    def test_included_non_attributed_helpers_do_not_add_kernel_selections(self):
        for body in ("fn helper() {}", "pub fn helper(value: u32) -> u32 { value + 1 }",
                     "#[cfg(test)] fn disabled() {}", "#[allow(dead_code)] pub(crate) fn helper() {}"):
            with self.subTest(body=body):
                selected = self.select(self.sources(body=body))
                self.assertEqual(set(selected), {"visible"})
                self.assertEqual(selected["visible"][0], self.library)

    def test_real_included_helper_bytes_remain_part_of_authenticated_closure(self):
        fixture = next(row for row in self.original["compilerFixtures"]
                       if row["fixtureId"] == "gfx942-fill-simulation")
        cache = {}
        self.parent.validate_compiler_input(ROOT, fixture, "helper closure test", cache)
        package = cache[fixture["compilerInput"]["packageManifest"]]
        sources = package["packageSources"]
        path, source = next(row for row in sources if row[0].name == "reference.rs")
        self.assertIn("*out = 42.5;", source)
        changed = [(p, text.replace("*out = 42.5;", "*out = 41.5;") if p == path else text)
                   for p, text in sources]
        self.assertNotEqual(self.parent.package_source_closure_sha256(ROOT, changed),
                            fixture["compilerInput"]["sourceClosureSha256"])
        with patch.object(self.parent, "package_rust_sources", return_value=changed):
            with self.assertRaisesRegex(SystemExit, "sourceClosureSha256 is stale"):
                self.parent.validate_compiler_input(ROOT, fixture, "helper closure test")
        package["packageSources"] = changed
        with self.assertRaisesRegex(SystemExit, "physical source differs from its validated closure"):
            self.parent.validate_kernel_inventory(self.original, None, repo_root=ROOT, package_cache=cache)

    def test_included_items_never_add_kernels_modules_or_imports(self):
        for body in ("#[kernel] fn injected() {}", "mod child;",
                     "mod child {}", "use crate::helper;", "const VALUE: u32 = 0;",
                     "body!();", '"masked" macro_rules! body {}', "#![no_std]\n" + self.definition,
                     '#[allow(dead_code)] "masked" macro_rules! body {}',
                     "#[cfg(test)] #[kernel] fn disabled() {}"):
            with self.subTest(body=body), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "included fixture source"):
                self.select(self.sources(body=body))

        for body in ("#[unknown_attribute] fn helper() {}", "#[cfg(unknown)] fn helper() {}",
                     "fn helper() { #[kernel] fn nested() {} }"):
            with self.subTest(body=body), self.assertRaises(self.identities.KernelInventoryError):
                self.select(self.sources(body=body))

    def test_parent_macro_shadowing_cannot_fake_builtin_include_in_a_child(self):
        definition = "macro_rules! include { ($path:expr) => {}; }\n"
        for root in (definition + "mod child;", "mod child;\n" + definition):
            sources = {self.library: root, "pkg/src/child.rs": 'include!("body.rs");\n' + self.kernel,
                       "pkg/src/body.rs": self.definition}
            with self.subTest(root=root), self.assertRaisesRegex(
                    self.identities.KernelInventoryError, "ambiguous macro or import scope"):
                self.select(sources)

    def test_duplicate_macros_and_import_aliases_or_globs_reject_in_include_closure(self):
        scope_error = "ambiguous macro or import scope"
        item_error = "unsupported fixture item or module selection"
        # Plain include names already reject in the legacy item check; braced
        # imports and other aliases/globs exercise the new closure-scope check.
        for hazard, error in (
                ("macro_rules! include { () => {}; }", scope_error), (self.definition, scope_error),
                ("use other::include;", item_error), ("use other::thing as include;", item_error),
                ("use other::{include};", scope_error), ("use other::{thing as include};", scope_error),
                ("use other::thing as alias;", scope_error), ("use other::*;", scope_error),
                ("use other::{thing, nested::*};", scope_error)):
            for prefix in (hazard + '\ninclude!("body.rs");\n', 'include!("body.rs");\n' + hazard):
                with self.subTest(prefix=prefix), self.assertRaisesRegex(
                        self.identities.KernelInventoryError, error):
                    self.select(self.sources(prefix))
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "ambiguous macro or import scope"):
            self.select(self.sources(body="macro_rules! include { () => {}; }"))

    def test_inactive_include_is_not_traversed_but_unknown_cfg_rejects(self):
        sources = {self.library: '#[cfg(test)] include!("missing.rs");\n' + self.kernel}
        self.assertEqual(set(self.select(sources)), {"visible"})
        sources[self.library] = '#[cfg(unknown)] include!("missing.rs");\n' + self.kernel
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "cfg predicate"):
            self.select(sources)

    def test_include_records_source_bytes_and_depth_have_exact_boundaries(self):
        sources = self.sources()
        budget = self.identities._Budget(4)  # One scanned function and three physical items.
        self.select(sources, budget=budget)
        self.assertEqual(budget.used, 4)
        with self.assertRaisesRegex(self.identities.KernelInventoryError, "record bound"):
            self.select(sources, budget=self.identities._Budget(3))
        total = sum(len(source.encode()) for source in sources.values())
        for limit, succeeds in ((total, True), (total - 1, False)):
            with patch.object(self.identities, "MAX_RUNTIME_BYTES", limit):
                if succeeds:
                    self.select(sources)
                else:
                    with self.assertRaisesRegex(self.identities.KernelInventoryError, "aggregate byte bound"):
                        self.select(sources)
        for depth in (32, 33):
            chain = {self.library: 'include!("0.rs");\n' + self.kernel}
            chain.update({f"pkg/src/{index}.rs": f'include!("{index + 1}.rs");'
                          for index in range(depth - 1)})
            chain[f"pkg/src/{depth - 1}.rs"] = self.definition
            if depth == 32:
                self.select(chain)
            else:
                with self.assertRaisesRegex(self.identities.KernelInventoryError, "nesting bound"):
                    self.select(chain)


class RowSourceBindingTests(unittest.TestCase):
    lesson_id = "cpu-semantic-simulation"
    kernel_id = "source-driver:cpu-semantic-simulation:6:row_affine_sum_u32_v1"

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "row_binding_parent", ROOT / "scripts/validate-tutorial-kernel-manifest.py")
        cls.parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.parent)
        cls.original = cls.parent.load_manifest(ROOT / "config/tutorial-kernel-manifest-v1.json")

    def row(self, document):
        lesson = next(row for row in document["curriculum"]["lessons"]
                      if row["lessonId"] == self.lesson_id)
        kernel = next(row for row in document["kernelInventory"]["kernels"]
                      if row["kernelId"] == self.kernel_id)
        return lesson["codeTabs"][6], kernel

    def test_registered_row_binds_exact_source_but_does_not_qualify_a_pair(self):
        document = copy.deepcopy(self.original)
        before = copy.deepcopy(document)
        tab, kernel = self.row(document)
        self.parent.validate_curriculum_tab(tab, 6, self.lesson_id, "executable", source_items=True)
        self.parent.validate_source_item(ROOT, self.lesson_id, tab, {})
        report = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
        self.assertEqual(document, before)
        item = tab["sourceItem"]
        self.assertEqual(item["cases"], [{
            "features": ["row-affine-u32-kernel"], "kernelSymbol": "row_affine_sum_u32_v1",
            "target": "gfx942", "displayedFragmentOrdinal": 0,
            "testFunction": "ordinary_row_affine_source_matches_oracle_and_replay",
            "expectation": {"kind": "verified-bundle-export", "bundleVersion": 5},
        }])
        self.assertFalse(item["compilerInput"]["defaultFeatures"])
        self.assertEqual(item["driver"]["path"],
                         "crates/rustc-codegen-fe2o3/tests/production_neutral_workgroup_reduce_driver_v1.rs")
        self.assertEqual(item["sourceRanges"], [{"byteOffset": 0, "byteLength": 2668}])
        self.assertEqual(tab["sourcePath"], "examples/workgroup_sync_v1/src/kernel_row_affine_u32.rs")
        raw = (ROOT / tab["sourcePath"]).read_bytes()
        self.assertEqual(len(raw), 2668)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "07adc0c50f24e51cb3d6c6bcb6cc1c8c6ff2a772c45ee2153435601eca2f39df")
        self.assertEqual(raw.index(b"row_affine_sum_u32_v1"), 885)
        self.assertEqual([(row["kind"], row["status"]) for row in kernel["variants"]],
                         [("simt", "source-bound"), ("tile", "pending"), ("mixed", "pending")])
        binding = kernel["variants"][0]["source"]
        self.assertEqual(binding["selection"], kernel["selections"][0])
        self.assertEqual(binding["implementationKernelId"], self.kernel_id)
        self.assertEqual(binding["functionUtf8Offset"], 885)
        self.assertEqual(binding["sourceSha256"], tab["sourceSha256"])
        for variant in kernel["variants"]:
            self.assertIn("gfx942/mi300x and gfx950/mi350", variant["blocker"]["reason"])
        self.assertEqual(report["knownKernelIdentityCount"], 101)
        self.assertEqual(report["sourceBoundVariantCount"], 76)
        self.assertEqual(report["sourceBoundPairCount"], 31)
        self.assertFalse(report["inventoryComplete"])
        self.assertIsNone(report["requiredPairCount"])
        self.assertEqual([(row["target"], row["deterministicSemanticRunnerAvailability"])
                          for row in document["qualification"]["hardwareTargets"]],
                         [("gfx942", "unavailable"), ("gfx950", "unavailable")])

    def component(self):
        """One actual row component, not the full-site runtime census."""
        document = copy.deepcopy(self.original)
        tab, kernel = self.row(document)
        lesson = next(row for row in document["curriculum"]["lessons"]
                      if row["lessonId"] == self.lesson_id)
        lesson["codeTabs"] = [tab]
        tab["ordinal"] = 0
        tab["sourceItem"]["contractSha256"] = self.parent.source_item_contract_sha256(self.lesson_id, tab)
        kernel["selections"][0]["tabOrdinal"] = 0
        kernel["variants"][0]["source"]["selection"]["tabOrdinal"] = 0
        display = next(row for row in document["kernelInventory"]["displayItems"]
                       if row["lessonId"] == self.lesson_id and row["tabOrdinal"] == 6)
        display["tabOrdinal"] = 0
        document["compilerFixtures"] = []
        document["curriculum"]["lessons"] = [lesson]
        document["kernelInventory"].update(kernels=[kernel], negativeCases=[], displayItems=[display])
        binding = kernel["variants"][0]["source"]
        kernel["variants"][0].update(status="pending", source=None)
        report = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
        binding["selectionSha256"] = report["kernelIdentities"][0]["selectionSha256"]
        kernel["variants"][0].update(status="source-bound", source=binding)
        runtime = {"schema": self.parent.SITE_INVENTORY_SCHEMA,
                   "site": document["curriculum"]["site"], "lessons": [{
                       "id": self.lesson_id, "codeTabs": [{
                           **tab, "displayedCode": (ROOT / tab["sourcePath"]).read_text(),
                           "sourceFragments": None,
                       }],
                   }]}
        return document, runtime, tab, kernel, display

    def check(self, document, runtime):
        self.parent.validate_site_inventory(document["curriculum"], runtime)
        return self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)

    def test_row_component_checks_exact_runtime_bytes_and_association(self):
        document, runtime, _, _, _ = self.component()
        before = copy.deepcopy((document, runtime))
        report = self.check(document, runtime)
        self.assertEqual(report["knownKernelIdentityCount"], 1)
        self.assertTrue(report["runtimeCensusValidated"])
        self.assertEqual(report["displayItemCount"], 1)
        self.assertEqual(report["sourceBoundVariantCount"], 1)
        self.assertEqual(report["sourceBoundPairCount"], 0)
        self.assertEqual((document, runtime), before)

    def test_row_source_and_display_substitutions_fail_closed(self):
        for field, value in (
            ("selectionSha256", "0" * 64), ("sourceSha256", "0" * 64),
            ("sourcePath", "examples/workgroup_sync_v1/src/lib.rs"),
            ("functionUtf8Offset", 886),
        ):
            document, runtime, _, kernel, _ = self.component()
            kernel["variants"][0]["source"][field] = value
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.check(document, runtime)
        for mutation in ("missing-identity", "duplicate-identity", "missing-display", "wrong-case",
                         "changed-display", "wrong-display-offset", "qualified", "extra-qualification"):
            document, runtime, _, kernel, display = self.component()
            if mutation == "missing-identity":
                document["kernelInventory"]["kernels"] = []
            elif mutation == "duplicate-identity":
                document["kernelInventory"]["kernels"].append(copy.deepcopy(kernel))
            elif mutation == "missing-display":
                document["kernelInventory"]["displayItems"] = []
            elif mutation == "wrong-case":
                kernel["variants"][0]["source"]["selection"]["caseOrdinal"] = 1
            elif mutation == "changed-display":
                runtime["lessons"][0]["codeTabs"][0]["displayedCode"] += "\n"
            elif mutation == "wrong-display-offset":
                display["functionUtf8Offset"] += 1
            elif mutation == "qualified":
                kernel["variants"][0]["status"] = "qualified"
            else:
                kernel["variants"][0]["qualified"] = True
            with self.subTest(mutation=mutation), self.assertRaises(SystemExit):
                self.check(document, runtime)

    def test_rehashed_wrong_feature_and_defaults_reject_physical_binding(self):
        for defaults in (False, True):
            document, _, tab, kernel, _ = self.component()
            item = tab["sourceItem"]
            if defaults:
                item["compilerInput"]["defaultFeatures"] = True
            else:
                item["cases"][0]["features"] = ["lds-kernel"]
            item["contractSha256"] = self.parent.source_item_contract_sha256(self.lesson_id, tab)
            binding = kernel["variants"][0]["source"]
            kernel["variants"][0].update(status="pending", source=None)
            report = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
            binding["selectionSha256"] = report["kernelIdentities"][0]["selectionSha256"]
            kernel["variants"][0].update(status="source-bound", source=binding)
            with self.subTest(defaults=defaults), self.assertRaisesRegex(
                    SystemExit, "feature-selected fixture kernel roster differs|selected source"):
                self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)

    def test_rehashed_source_item_still_requires_real_feature_test_and_whole_file(self):
        for mutation in ("feature", "test", "range", "lock", "closure"):
            document = copy.deepcopy(self.original)
            tab, _ = self.row(document)
            item = tab["sourceItem"]
            if mutation == "feature":
                item["cases"][0]["features"] = ["missing-row-feature"]
            elif mutation == "test":
                item["cases"][0]["testFunction"] = "row_affine_source_matches_oracle_and_replay"
            elif mutation == "range":
                item["sourceRanges"][0]["byteLength"] -= 1
            elif mutation == "lock":
                item["compilerInput"]["cargoLockSha256"] = "0" * 64
            else:
                item["compilerInput"]["sourceClosureSha256"] = "0" * 64
            item["contractSha256"] = self.parent.source_item_contract_sha256(self.lesson_id, tab)
            with self.subTest(mutation=mutation), self.assertRaises(SystemExit):
                self.parent.validate_source_item(ROOT, self.lesson_id, tab, {})


class SelectedFeatureFixtureBindingTests(unittest.TestCase):
    """Real file/contract joins; these source associations grant no execution authority."""

    @classmethod
    def setUpClass(cls):
        for name, filename in (("parent", "validate-tutorial-kernel-manifest.py"),
                               ("identities", "tutorial_kernel_identities.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            setattr(cls, name, module)

    def document(self, features, direct=(), defaults=True, build=None, build_file=False):
        temporary = tempfile.TemporaryDirectory(prefix="fe2o3-feature-binding-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        package = root / "pkg"
        (package / "src").mkdir(parents=True)
        (package / "native/src").mkdir(parents=True)
        source = "#[kernel]\npub fn visible() {}\n"
        (package / "src/lib.rs").write_text(source)
        build_setting = "" if build is None else (
            "build = false\n" if build is False else f"build = {build!r}\n")
        (package / "Cargo.toml").write_text(
            '[package]\nname = "feature-fixture"\nversion = "0.1.0"\nedition = "2024"\n'
            + build_setting
            + '[lib]\nname = "feature_fixture"\n'
            + '[dependencies]\nnative = { path = "native", optional = true }\n'
            + '[features]\n'
            + "".join(f"{name} = {members!r}\n" for name, members in features.items()))
        (package / "native/Cargo.toml").write_text(
            '[package]\nname = "native"\nversion = "0.1.0"\nedition = "2024"\n'
            '[features]\nunsafe-launch = []\n')
        (package / "native/src/lib.rs").write_text("pub fn helper() {}\n")
        if build_file:
            (package / ("custom.rs" if isinstance(build, str) else "build.rs")).write_text(
                "fn main() {}\n")
        lock = package / "Cargo.lock"
        lock.write_text(
            'version = 4\n\n[[package]]\nname = "feature-fixture"\nversion = "0.1.0"\n'
            'dependencies = ["native"]\n\n[[package]]\nname = "native"\nversion = "0.1.0"\n')
        inputs = {
            "packageManifest": "pkg/Cargo.toml",
            "packageManifestSha256": hashlib.sha256((package / "Cargo.toml").read_bytes()).hexdigest(),
            "cargoLockPath": "pkg/Cargo.lock",
            "cargoLockSha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "sourcePaths": ["pkg/src/lib.rs"],
            "sourceClosureSha256": self.parent.package_source_closure_sha256(
                root, self.parent.package_rust_sources(root, "pkg/Cargo.toml", "feature fixture")),
            "cargoTarget": {"kind": "lib", "name": "feature_fixture", "sourcePath": "src/lib.rs"},
            "defaultFeatures": defaults, "features": sorted(direct), "kernelSymbols": ["visible"],
        }
        fixture = {
            "fixtureId": "feature-alias", "target": "gfx942",
            "matrix": {"caseId": "feature-alias", "runnerPath": "run.sh",
                       "artifactName": "visible.hsaco", "runnerArguments": [], "environment": []},
            "compilerInput": inputs,
        }
        inputs["contractSha256"] = self.parent.fixture_input_contract_sha256(fixture)
        reference = {"kind": "fixture", "fixtureId": "feature-alias", "kernelSymbol": "visible"}
        kernel_id = "fixture:feature-alias:visible"
        binding = {
            "implementationKernelId": kernel_id, "selection": reference,
            "selectionSha256": self.identities._selection_identity(
                inputs, "visible", self.identities._Budget(4096)),
            "sourcePath": "pkg/src/lib.rs", "sourceSha256": hashlib.sha256(source.encode()).hexdigest(),
            "functionUtf8Offset": source.index("visible"),
        }
        blocker = {"owner": "fixture-tests", "issue": "https://github.com/harsh-nod/fe2o3/issues/275",
                   "reason": "Source association only; no execution or qualification."}
        document = {
            "curriculum": {"schema": "fe2o3-tutorial-curriculum-obligations-v2", "lessons": []},
            "compilerFixtures": [fixture],
            "kernelInventory": {
                "schema": self.identities.SCHEMA, "negativeCases": [], "displayItems": [],
                "kernels": [{
                    "kernelId": kernel_id, "selections": [reference],
                    "variants": [
                        {"kind": "simt", "status": "source-bound", "source": binding, "blocker": blocker},
                        {"kind": "tile", "status": "pending", "source": None, "blocker": blocker},
                    ],
                }],
            },
        }
        return root, document, fixture

    def features(self, member="dep:native"):
        return {"default": ["base"], "base": [], "qualification": [member],
                "nested": ["middle"], "middle": ["qualification"],
                "local-two": ["local-one"], "local-one": ["base"]}

    def check(self, root, document, cache=None):
        return self.parent.validate_kernel_inventory(document, None, repo_root=root,
                                                     package_cache=cache)

    def test_inactive_dependency_aliases_allow_default_local_selection(self):
        for member in ("dep:native", "native/unsafe-launch", "native?/unsafe-launch"):
            with self.subTest(member=member):
                root, document, fixture = self.document(self.features(member))
                checked = self.parent.validate_compiler_input(root, fixture, "inactive alias")
                self.assertEqual(checked["enabledFeatures"], ["base", "default"])
                result = self.check(root, document)
                self.assertEqual(result["sourceBoundVariantCount"], 1)
                self.assertEqual(result["sourceBoundPairCount"], 0)
                self.assertFalse(result["inventoryComplete"])

    def test_nested_local_aliases_use_complete_transitive_closure(self):
        root, document, fixture = self.document(self.features(), ["local-two"], defaults=False)
        checked = self.parent.validate_compiler_input(root, fixture, "nested local")
        self.assertEqual(checked["enabledFeatures"], ["base", "local-one", "local-two"])
        self.assertEqual(self.check(root, document)["sourceBoundVariantCount"], 1)

    def test_direct_dependency_and_unknown_aliases_remain_rejected(self):
        for member in ("dep:native", "native/unsafe-launch", "native?/unsafe-launch", "missing"):
            with self.subTest(member=member):
                root, document, _ = self.document(self.features(member), ["qualification"], False)
                with self.assertRaisesRegex(SystemExit, "requires local features without build-script cfg"):
                    self.check(root, document)

    def test_nested_local_alias_reaching_dependency_alias_remains_rejected(self):
        root, document, fixture = self.document(self.features(), ["nested"], defaults=False)
        checked = self.parent.validate_compiler_input(root, fixture, "nested dependency")
        self.assertEqual(checked["enabledFeatures"], ["middle", "nested", "qualification"])
        with self.assertRaisesRegex(SystemExit, "requires local features without build-script cfg"):
            self.check(root, document)

    def test_default_and_no_default_control_alias_reachability(self):
        features = self.features()
        features["default"] = ["nested"]
        root, document, fixture = self.document(features)
        checked = self.parent.validate_compiler_input(root, fixture, "default dependency")
        self.assertEqual(checked["enabledFeatures"], ["default", "middle", "nested", "qualification"])
        with self.assertRaisesRegex(SystemExit, "requires local features without build-script cfg"):
            self.check(root, document)
        root, document, fixture = self.document(features, defaults=False)
        self.assertEqual(self.parent.validate_compiler_input(
            root, fixture, "no default")["enabledFeatures"], [])
        self.assertEqual(self.check(root, document)["sourceBoundVariantCount"], 1)

    def test_auto_and_explicit_build_scripts_still_reject_with_or_without_defaults(self):
        for build in (None, "custom.rs"):
            for defaults in (False, True):
                with self.subTest(build=build, defaults=defaults):
                    root, document, _ = self.document(
                        self.features(), defaults=defaults, build=build, build_file=True)
                    with self.assertRaisesRegex(SystemExit, "requires local features without build-script cfg"):
                        self.check(root, document)

    def test_explicitly_disabled_build_script_retains_existing_behavior(self):
        root, document, _ = self.document(self.features(), build=False, build_file=True)
        self.assertEqual(self.check(root, document)["sourceBoundVariantCount"], 1)

    def test_cached_local_selection_cannot_admit_later_dependency_alias(self):
        root, document, fixture = self.document(self.features())
        cache = {}
        self.assertEqual(self.check(root, document, cache)["sourceBoundVariantCount"], 1)
        inputs = fixture["compilerInput"]
        inputs["defaultFeatures"] = False
        inputs["features"] = ["nested"]
        inputs["contractSha256"] = self.parent.fixture_input_contract_sha256(fixture)
        document["kernelInventory"]["kernels"][0]["variants"][0]["source"]["selectionSha256"] = (
            self.identities._selection_identity(inputs, "visible", self.identities._Budget(4096)))
        with self.assertRaisesRegex(SystemExit, "requires local features without build-script cfg"):
            self.check(root, document, cache)


def load_tests(loader, tests, pattern):
    child = runpy.run_path(
        str(Path(__file__).with_name("_tutorial_current_entry_binding_tests.py")),
        init_globals={"FixtureBindingTests": FixtureBindingTests, "ROOT": ROOT},
    )["CurrentEntryBindingTests"]
    tests.addTests(child(name) for name in child.__dict__ if name.startswith("test_"))
    derived = runpy.run_path(
        str(Path(__file__).with_name("_tutorial_builtin_derive_binding_tests.py")),
        init_globals={"LiteralIncludeSelectionTests": LiteralIncludeSelectionTests,
                      "FixtureBindingTests": FixtureBindingTests, "ROOT": ROOT},
    )
    for name in ("BuiltinDeriveSelectionTests", "MoeCurrentBindingTests"):
        child = derived[name]
        tests.addTests(child(method) for method in child.__dict__ if method.startswith("test_"))
    return tests


if __name__ == "__main__":
    unittest.main()

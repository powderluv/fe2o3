#!/usr/bin/env python3
"""Pure identity-contract tests; shared Rust scanner tests live in the parent suite."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / "tutorial_kernel_identities.py"
SPEC = importlib.util.spec_from_file_location("tutorial_kernel_identities", PATH)
IDENTITIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITIES)
COUNT_KEYS = ("knownVariantObligationCount", "pendingVariantCount", "unregisteredDisplayItemCount")


def scan(source):
    # This fixture scanner supplies the frozen callback contract, not a second
    # production parser. Its inputs are only the small literal sources below.
    return [{"kernelSymbol": match["name"].removeprefix("r#"),
             "functionUtf8Offset": len(source[:match.start("name")].encode()),
             "attributedKernel": bool(match["attr"])}
            for match in re.finditer(r"(?P<attr>#\[kernel\]\s*)?fn\s+(?P<name>r#\w+|\w+)", source)]


def case_ref(index, tab=0):
    return {"kind": "source-driver-case", "lessonId": "lesson", "tabOrdinal": tab, "caseOrdinal": index}


def variants():
    return [{"kind": kind, "status": "pending", "source": None,
             "blocker": {"owner": "issue275-integration",
                         "issue": "https://github.com/harsh-nod/fe2o3/issues/275",
                         "reason": "Variant implementation is pending."}}
            for kind in ("simt", "tile")]


def selection_digest(inputs, symbol):
    payload = {key: inputs[key] for key in (
        "packageManifest", "packageManifestSha256", "cargoLockPath", "cargoLockSha256",
        "sourcePaths", "sourceClosureSha256", "cargoTarget", "defaultFeatures", "features",
    )}
    payload["kernelSymbol"] = symbol
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"fe2o3-tutorial-kernel-selection-v1\0" + encoded).hexdigest()


class KernelIdentitiesTests(unittest.TestCase):
    def setUp(self):
        self.source = "#[kernel] fn good() {}\n#[kernel] fn bad() {}\nfn helper() {}\n"
        inputs = {"packageManifest": "example/Cargo.toml", "packageManifestSha256": "1" * 64,
                  "cargoLockPath": "Cargo.lock", "cargoLockSha256": "2" * 64,
                  "sourcePaths": ["example/src/lib.rs"], "sourceClosureSha256": "3" * 64,
                  "cargoTarget": {"kind": "lib", "name": "example", "sourcePath": "src/lib.rs"},
                  "defaultFeatures": False, "features": ["good"]}
        self.fixture_ref = {"kind": "fixture", "fixtureId": "first", "kernelSymbol": "good"}
        cases = [{"kernelSymbol": name, "features": [name], "target": "gfx942",
                  "displayedFragmentOrdinal": 0,
                  "expectation": {"kind": kind}}
                 for name, kind in (("good", "verified-bundle-export"), ("bad", "rejected"))]
        tab = self.tab(self.source)
        tab["sourceItem"] = {"compilerInput": {key: value for key, value in inputs.items() if key != "features"},
                             "cases": cases}
        display = []
        for item in scan(self.source):
            symbol = item["kernelSymbol"]
            display.append({"lessonId": "lesson", "tabOrdinal": 0,
                            "functionUtf8Offset": item["functionUtf8Offset"], "kernelSymbol": symbol,
                            "classification": {"good": "kernel", "bad": "required-negative", "helper": "helper"}[symbol],
                            "kernelIds": ["good"] if symbol == "good" else [],
                            "negativeCases": [case_ref(1)] if symbol == "bad" else [],
                            "bindingStatus": "not-applicable" if symbol == "helper" else "source-driver-contract",
                            "reason": "Exact fixture classification."})
        self.manifest = {
            "compilerFixtures": [{"fixtureId": "first", "target": "gfx942",
                                  "compilerInput": {**inputs, "kernelSymbols": ["good"]}}],
            "curriculum": {"schema": "fe2o3-tutorial-curriculum-obligations-v2",
                           "lessons": [{"lessonId": "lesson", "role": "executable", "codeTabs": [tab]}]},
            "kernelInventory": {"schema": IDENTITIES.SCHEMA,
                                "kernels": [{"kernelId": "good", "selections": [self.fixture_ref, case_ref(0)],
                                             "variants": variants()}],
                                "negativeCases": [case_ref(1)], "displayItems": display},
        }
        self.runtime = {"lessons": [{"id": "lesson", "codeTabs": [self.live(self.source)]}]}

    @staticmethod
    def tab(source, kind="kernel"):
        encoded = source.encode()
        return {"language": "rust", "kind": kind, "displayedUtf8Bytes": len(encoded),
                "displayedSha256": hashlib.sha256(encoded).hexdigest(), "sourceItem": None,
                "sourceDigestScope": "displayed", "sourceFragmentsSha256": [hashlib.sha256(encoded).hexdigest()]}

    @staticmethod
    def live(source):
        return {"displayedCode": source, "sourceFragments": [source]}

    def validate(self, runtime=True, **kwargs):
        return IDENTITIES.validate_kernel_inventory(self.manifest, self.runtime if runtime else None, scan, **kwargs)

    def inventory(self):
        return self.manifest["kernelInventory"]

    def assert_refused(self, pattern, runtime=True):
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, pattern):
            self.validate(runtime)

    def add_tab(self, source, kind="host", classification="kernel"):
        ordinal = len(self.manifest["curriculum"]["lessons"][0]["codeTabs"])
        self.manifest["curriculum"]["lessons"][0]["codeTabs"].append(self.tab(source, kind))
        self.runtime["lessons"][0]["codeTabs"].append(self.live(source))
        for item in scan(source):
            if not item["attributedKernel"] and kind != "kernel":
                continue
            self.inventory()["displayItems"].append({
                "lessonId": "lesson", "tabOrdinal": ordinal,
                "functionUtf8Offset": item["functionUtf8Offset"], "kernelSymbol": item["kernelSymbol"],
                "classification": classification, "kernelIds": [], "negativeCases": [],
                "bindingStatus": "pending" if classification == "kernel" else "not-applicable",
                "reason": "Source correspondence is pending."})
        return ordinal

    def test_complete_partition_is_not_variant_qualification_and_does_not_mutate(self):
        before = copy.deepcopy((self.manifest, self.runtime))
        result = self.validate()
        self.assertTrue(result["inventoryComplete"])
        self.assertEqual(result["requiredPairCount"], 1)
        self.assertEqual(result["knownKernelIdentityCount"], 1)
        self.assertEqual([result[key] for key in COUNT_KEYS], [2, 2, 0])
        self.assertEqual(result["kernelIdentities"][0]["selectionSha256"],
                         selection_digest(self.manifest["compilerFixtures"][0]["compilerInput"], "good"))
        self.assertEqual(result["negativeCaseCount"], 1)
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(result["kernelIdentities"][0]["variants"], variants())
        self.assertEqual((self.manifest, self.runtime), before)
        result["kernelIdentities"][0]["variants"][0]["blocker"]["owner"] = "changed"
        self.assertEqual((self.manifest, self.runtime), before)

    def test_without_runtime_never_claims_complete_or_hides_missing_joins(self):
        result = self.validate(False)
        self.assertEqual([result[key] for key in COUNT_KEYS], [2, 2, 0])
        self.assertFalse(result["runtimeCensusValidated"])
        self.assertFalse(result["inventoryComplete"])
        self.assertIsNone(result["requiredPairCount"])
        self.assertTrue(any("selection" in row for row in result["unresolvedBindings"]))

    def test_selection_partition_missing_duplicate_negative_and_foreign(self):
        original = copy.deepcopy(self.manifest)
        for mutation, pattern in (
            (lambda: self.inventory()["kernels"][0]["selections"].pop(), "missing positive"),
            (lambda: self.inventory()["kernels"][0]["selections"].append(copy.deepcopy(self.fixture_ref)), "duplicate positive"),
            (lambda: self.inventory()["kernels"][0]["selections"].append(case_ref(1)), "negative"),
            (lambda: self.inventory()["kernels"][0]["selections"].append(case_ref(99)), "unknown"),
            (lambda: self.inventory()["negativeCases"].clear(), "missing required-negative"),
            (lambda: self.inventory()["negativeCases"].append(case_ref(1)), "duplicate required-negative"),
            (lambda: self.inventory()["negativeCases"].append(case_ref(0)), "positive"),
        ):
            self.manifest = copy.deepcopy(original)
            mutation()
            with self.subTest(pattern=pattern):
                self.assert_refused(pattern)

    def test_features_defaults_source_hashes_and_cargo_target_do_not_merge(self):
        original = copy.deepcopy(self.manifest)
        for key, changed in (("features", ["other"]), ("defaultFeatures", True),
                             ("sourceClosureSha256", "9" * 64), ("cargoLockSha256", "8" * 64),
                             ("sourcePaths", ["example/src/other.rs"]),
                             ("cargoTarget", {"kind": "lib", "name": "other", "sourcePath": "src/lib.rs"})):
            self.manifest = copy.deepcopy(original)
            self.manifest["compilerFixtures"][0]["compilerInput"][key] = changed
            with self.subTest(key=key):
                self.assert_refused("cannot merge different source selections")

    def test_cross_target_identical_source_shares_one_id_not_two(self):
        fixture = copy.deepcopy(self.manifest["compilerFixtures"][0])
        fixture.update(fixtureId="other-target", target="gfx950")
        self.manifest["compilerFixtures"].append(fixture)
        ref = {"kind": "fixture", "fixtureId": "other-target", "kernelSymbol": "good"}
        self.inventory()["kernels"][0]["selections"].append(ref)
        result = self.validate()
        self.assertEqual(result["requiredPairCount"], 1)
        self.assertEqual([result[key] for key in COUNT_KEYS], [2, 2, 0])
        self.inventory()["kernels"][0]["selections"].pop()
        self.inventory()["kernels"].append({"kernelId": "other", "selections": [ref], "variants": variants()})
        self.assert_refused("identical source selections must share")

    def test_unbound_positive_fixture_keeps_count_unknown(self):
        fixture = copy.deepcopy(self.manifest["compilerFixtures"][0])
        fixture.update(fixtureId="different-feature")
        fixture["compilerInput"]["features"] = ["other"]
        self.manifest["compilerFixtures"].append(fixture)
        ref = {"kind": "fixture", "fixtureId": "different-feature", "kernelSymbol": "good"}
        self.inventory()["kernels"].append({"kernelId": "other", "selections": [ref], "variants": variants()})
        result = self.validate()
        self.assertFalse(result["inventoryComplete"])
        self.assertIsNone(result["requiredPairCount"])
        self.assertIn(ref, [row.get("selection") for row in result["unresolvedBindings"]])

    def test_unregistered_occurrences_are_not_stale_bindings_or_new_identities(self):
        self.inventory()["displayItems"][0]["bindingStatus"] = "pending"
        for unregistered in range(3):
            if unregistered:
                self.add_tab("#[kernel] fn unknown() {}")
            for runtime in (False, True):
                with self.subTest(unregistered=unregistered, runtime=runtime):
                    result = self.validate(runtime)
                    self.assertEqual([result[key] for key in COUNT_KEYS], [2, 2, unregistered])
                    self.assertEqual(result["pendingDisplayItemCount"], unregistered + 1)
                    self.assertEqual(result["knownKernelIdentityCount"], 1)
                    self.assertEqual(result["runtimeCensusValidated"], runtime)
                    self.assertFalse(result["inventoryComplete"])
                    self.assertIsNone(result["requiredPairCount"])

    def test_all_tab_kinds_and_bare_kernel_functions_need_occurrences(self):
        self.add_tab("#[kernel] fn first() {}\n#[kernel] fn first() {}\n", "comparison")
        self.add_tab("fn unannotated_entry() {}\n", "kernel")
        result = self.validate()
        self.assertEqual(result["displayItemCount"], 6)
        self.assertEqual(result["pendingDisplayItemCount"], 3)
        self.inventory()["displayItems"].pop()
        self.assert_refused("missing live function census")

    def test_raw_unicode_name_tokens_and_non_kernel_helpers(self):
        self.add_tab("// retained UTF-8: \u00e9\n#[kernel] fn r#type() {}\nfn \u03bb() {}\n", "kernel")
        self.inventory()["displayItems"][-1].update(classification="helper", bindingStatus="not-applicable")
        self.add_tab("fn \u03bb() {}\n", "host")
        result = self.validate()
        self.assertEqual(result["pendingDisplayItemCount"], 1)
        self.assertEqual(result["displayItemCount"], 5)
        self.inventory()["displayItems"][-2]["functionUtf8Offset"] += 2
        self.assert_refused("missing from the live function census")

    def test_no_attributed_helper_or_executable_conceptual_downgrade(self):
        row = self.inventory()["displayItems"][0]
        row.update(kernelIds=[], bindingStatus="not-applicable", classification="conceptual")
        self.assert_refused("cannot detach a source case")
        self.add_tab("#[kernel] fn other() {}\n", "kernel")
        row.update(kernelIds=["good"], bindingStatus="source-driver-contract", classification="kernel")
        self.inventory()["displayItems"][-1].update(classification="helper", bindingStatus="not-applicable")
        self.assert_refused("attributed kernel cannot be classified as a helper")

    def test_negative_cannot_be_reclassified_or_detached(self):
        row = self.inventory()["displayItems"][1]
        row.update(classification="kernel", kernelIds=["good"], negativeCases=[])
        self.assert_refused("negative cannot be converted")

    def test_same_name_in_distinct_positive_negative_fragments(self):
        parts = ["#[kernel] fn good() {}", "#[kernel] fn good() {}"]
        source = "\n\n".join(parts)
        tab = self.manifest["curriculum"]["lessons"][0]["codeTabs"][0]
        item = tab["sourceItem"]
        tab.update(self.tab(source))
        tab["sourceItem"] = item
        tab["sourceFragmentsSha256"] = [hashlib.sha256(part.encode()).hexdigest() for part in parts]
        item["cases"][1].update(kernelSymbol="good", displayedFragmentOrdinal=1)
        self.runtime["lessons"][0]["codeTabs"][0] = {"displayedCode": source, "sourceFragments": parts}
        self.inventory()["displayItems"].pop()
        for row, function in zip(self.inventory()["displayItems"], scan(source), strict=True):
            row.update(kernelSymbol="good", functionUtf8Offset=function["functionUtf8Offset"])
        self.assertFalse(self.validate(False)["inventoryComplete"])
        self.assertTrue(self.validate()["inventoryComplete"])
        self.inventory()["displayItems"][1]["functionUtf8Offset"] -= 1
        self.assert_refused("missing from the live function census")

    def test_case_join_is_not_transferred_to_another_display_occurrence(self):
        tab = copy.deepcopy(self.manifest["curriculum"]["lessons"][0]["codeTabs"][0])
        source = "#[kernel] fn good() {}\n"
        item = tab["sourceItem"]
        item["cases"] = [item["cases"][0]]
        tab.update(self.tab(source))
        tab["sourceItem"] = item
        self.manifest["curriculum"]["lessons"][0]["codeTabs"].append(tab)
        self.runtime["lessons"][0]["codeTabs"].append(self.live(source))
        self.inventory()["kernels"][0]["selections"].append(case_ref(0, 1))
        self.inventory()["displayItems"].append({**copy.deepcopy(self.inventory()["displayItems"][0]),
                                               "tabOrdinal": 1, "bindingStatus": "pending", "kernelIds": [],
                                               "functionUtf8Offset": scan(source)[0]["functionUtf8Offset"]})
        result = self.validate()
        self.assertIn(case_ref(0, 1), [row.get("selection") for row in result["unresolvedBindings"]])
        self.assertFalse(result["inventoryComplete"])

    def test_same_name_helper_in_another_fragment_defers_only_the_occurrence_join(self):
        parts = ["#[kernel] fn good() {}\n#[kernel] fn bad() {}", "mod local { fn good() {} }"]
        source = "\n\n".join(parts)
        tab = self.manifest["curriculum"]["lessons"][0]["codeTabs"][0]
        item = tab["sourceItem"]
        tab.update(self.tab(source))
        tab["sourceItem"] = item
        tab["sourceFragmentsSha256"] = [hashlib.sha256(part.encode()).hexdigest() for part in parts]
        self.runtime["lessons"][0]["codeTabs"][0] = {"displayedCode": source, "sourceFragments": parts}
        for row, function in zip(self.inventory()["displayItems"], scan(source), strict=True):
            row.update(kernelSymbol=function["kernelSymbol"],
                       functionUtf8Offset=function["functionUtf8Offset"])
        self.assertFalse(self.validate(False)["inventoryComplete"])
        self.assertTrue(self.validate()["inventoryComplete"])
        self.inventory()["displayItems"][0].update(
            classification="helper", bindingStatus="not-applicable", kernelIds=[],
        )
        self.assert_refused("cannot detach a source case")

    def test_stale_fragments_bytes_missing_duplicate_and_malformed_fields(self):
        manifest, runtime = copy.deepcopy(self.manifest), copy.deepcopy(self.runtime)
        mutations = [
            (lambda: self.runtime["lessons"][0]["codeTabs"][0].update(displayedCode=self.source + "x"), "displayed bytes"),
            (lambda: self.runtime["lessons"][0]["codeTabs"][0].update(sourceFragments=[self.source + "x"]), "reconstruct"),
            (lambda: self.inventory()["displayItems"].append(copy.deepcopy(self.inventory()["displayItems"][0])), "duplicate display"),
            (lambda: self.inventory()["displayItems"][0].update(classification={}), "must be strings"),
            (lambda: self.inventory()["displayItems"][0].update(bindingStatus=[]), "must be strings"),
            (lambda: self.inventory()["displayItems"][0].update(functionUtf8Offset=True), "nonnegative integer"),
            (lambda: self.inventory()["kernels"][0]["variants"][0].update(status="passed"), "not qualification"),
            (lambda: self.manifest.pop("curriculum"), "existing V2 curriculum"),
            (lambda: self.manifest["curriculum"].update(schema="other"), "existing V2 curriculum"),
        ]
        for mutation, pattern in mutations:
            self.manifest, self.runtime = copy.deepcopy(manifest), copy.deepcopy(runtime)
            mutation()
            with self.subTest(pattern=pattern):
                self.assert_refused(pattern)

    def test_exact_record_and_streamed_identity_byte_limits(self):
        minimum = next(limit for limit in range(1, 100)
                       if self.succeeds_with_limit(limit))
        self.assertTrue(self.validate(max_records=minimum)["inventoryComplete"])
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound"):
            self.validate(max_records=minimum - 1)
        with mock.patch.object(IDENTITIES, "MAX_IDENTITY_BYTES", 1):
            self.assert_refused("identity bytes exceed")
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", 1):
            self.assert_refused("aggregate byte bound")

    def succeeds_with_limit(self, limit):
        try:
            self.validate(max_records=limit)
            return True
        except IDENTITIES.KernelInventoryError:
            return False


class FixtureDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = PATH.with_name("validate-tutorial-kernel-manifest.py")
        spec = importlib.util.spec_from_file_location("fixture_scanner", path)
        cls.scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scanner)

    def setUp(self):
        self.library = "example/src/lib.rs"
        self.path = "example/src/left.rs"
        self.sources = {
            self.library: '#![cfg_attr(target_arch = "amdgpu", no_std)]\n'
                          '#[cfg(feature = "left")] mod left;\n#[cfg(feature = "right")] mod right;\n',
            self.path: '// UTF-8: \u03bb\n#[cfg(any(not(target_arch = "amdgpu"), feature = "left"))]\n#[kernel] fn same() {}\n',
            "example/src/right.rs": '#[cfg(feature = "right")]\n#[kernel] fn same() {}\n',
        }
        source = self.sources[self.path]
        digest = hashlib.sha256(source.encode()).hexdigest()
        self.tab = {**KernelIdentitiesTests.tab(source), "sourcePath": self.path,
                    "sourceDigestScope": "file", "sourceSha256": digest,
                    "sourceFragmentsSha256": None}
        inputs = {"packageManifest": "example/Cargo.toml", "packageManifestSha256": "1" * 64,
                  "cargoLockPath": "Cargo.lock", "cargoLockSha256": "2" * 64,
                  "sourcePaths": [self.path], "sourceClosureSha256": "3" * 64,
                  "cargoTarget": {"kind": "lib", "name": "example", "sourcePath": "src/lib.rs"},
                  "defaultFeatures": False, "features": ["left"], "kernelSymbols": ["same"]}
        self.fixture = {"fixtureId": "left", "target": "gfx950", "compilerInput": inputs}
        function = self.scanner.ordinary_rust_function_items(source)[0]
        self.row = {"lessonId": "lesson", "tabOrdinal": 0, "kernelSymbol": "same",
                    "functionUtf8Offset": function["functionUtf8Offset"], "classification": "kernel",
                    "kernelIds": ["left"], "negativeCases": [], "bindingStatus": "fixture-source-contract",
                    "reason": "Expected fixture/source contract only."}
        self.manifest = {
            "compilerFixtures": [self.fixture],
            "curriculum": {"schema": "fe2o3-tutorial-curriculum-obligations-v2",
                           "lessons": [{"lessonId": "lesson", "role": "executable", "codeTabs": [self.tab]}]},
            "kernelInventory": {"schema": IDENTITIES.SCHEMA,
                                "kernels": [{"kernelId": "left", "variants": variants(), "selections": [
                                    {"kind": "fixture", "fixtureId": "left", "kernelSymbol": "same"}]}],
                                "negativeCases": [], "displayItems": [self.row]},
        }
        self.runtime = {"lessons": [{"id": "lesson", "codeTabs": [
            {"displayedCode": source, "sourceFragments": None}]}]}

    def validate(self, runtime=True, **kwargs):
        def syntax(source):
            code = self.scanner._rust_code_without_comments_and_literals(source)
            return code, self.scanner._rust_delimiters(code)

        return IDENTITIES.validate_kernel_inventory(
            self.manifest, self.runtime if runtime else None, self.scanner.ordinary_rust_function_items,
            load_fixture_sources=kwargs.pop("load_fixture_sources", lambda fixture: (
                self.library, self.sources, fixture["compilerInput"]["features"])),
            rust_syntax=syntax, **kwargs,
        )

    def test_whole_file_contract_requires_live_census_and_keeps_variants_pending(self):
        before = copy.deepcopy((self.manifest, self.runtime, self.sources))
        result = self.validate()
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(result["kernelIdentities"][0]["variants"], variants())
        self.assertEqual((self.manifest, self.runtime, self.sources), before)
        without_runtime = self.validate(False)
        self.assertFalse(without_runtime["inventoryComplete"])
        self.assertIsNone(without_runtime["requiredPairCount"])
        self.assertTrue(any("selection" in row for row in without_runtime["unresolvedBindings"]))
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "physical source validation"):
            IDENTITIES.validate_kernel_inventory(self.manifest, None, self.scanner.ordinary_rust_function_items)

    def test_wrong_file_feature_bytes_and_occurrence_reject(self):
        for field, value in (("sourcePath", "example/src/right.rs"), ("sourceSha256", "0" * 64),
                             ("displayedSha256", "0" * 64), ("displayedUtf8Bytes", 1),
                             ("sourceDigestScope", "unknown"), ("sourceFragmentsSha256", [])):
            self.setUp()
            self.tab[field] = value
            with self.subTest(field=field), self.assertRaises(IDENTITIES.KernelInventoryError):
                self.validate(False)
        self.setUp()
        self.fixture["compilerInput"]["features"] = ["right"]
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "exact current source occurrence"):
            self.validate(False)
        self.setUp()
        self.row["functionUtf8Offset"] += 1
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "exact current source occurrence"):
            self.validate(False)
        self.setUp()
        self.sources[self.path] += "// changed\n"
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "exact current source occurrence"):
            self.validate(False)

    def test_package_membership_does_not_prove_module_reachability(self):
        cases = [
            ('#[cfg(feature = "right")] mod left;', "roster differs"),
            ('mod left; mod left;', "ambiguous"),
            ('#[path = "../left.rs"] mod renamed;', "selection attribute"),
            ('mod left { #[kernel] fn same() {} }', "nested fixture kernel"),
            ('mod r#left;', "unsupported fixture item"),
            ('include!("left.rs");', "included fixture source must contain only inert macro definitions"),
            ('#[cfg_attr(feature = "left", path = "left.rs")] mod left;', "selection attribute"),
            ('#[unknown_attribute] mod left;', "selection attribute"),
            ('#[cfg(all(feature = "left", unknown))] mod left;', "cfg predicate"),
        ]
        for library, message in cases:
            self.setUp()
            self.sources[self.library] = library
            with self.subTest(library=library), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, message):
                self.validate(False)
        self.setUp()
        self.sources["example/src/left/mod.rs"] = self.sources[self.path]
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "ambiguous"):
            self.validate(False)

    def test_non_test_cfg_literal_and_compositions(self):
        for predicate, expected in (
            ("test", False), ("not(test)", True),
            ('any(test, feature = "left")', True),
            ('all(not(test), target_arch = "amdgpu")', True),
            ('all(test, feature = "left")', False),
            ("any(test,)", False), ("not(test,)", True),
        ):
            with self.subTest(predicate=predicate):
                self.assertIs(IDENTITIES._fixture_cfg(predicate, {"left"}), expected)

    def test_non_test_cfg_still_rejects_unknown_and_malformed_predicates(self):
        for predicate in (
            "all(test, unknown)", "any(not(test), unknown)", "not(test, test)",
            'test = "false"', "test()", "test extra", "not(test", "Test",
        ):
            with self.subTest(predicate=predicate), self.assertRaises(IDENTITIES.KernelInventoryError):
                IDENTITIES._fixture_cfg(predicate, {"left"})

    def test_cfg_token_boundary_is_exact_and_checks_inactive_predicates(self):
        predicate = 'any(' + 'feature="left",' * 127 + 'test)'
        self.assertEqual(len(re.findall(r'[A-Za-z_]+|"[A-Za-z_]+"|[(),=]', predicate)), 512)
        oversized = predicate[:-1] + ',)'
        for features, expected in (({"left"}, True), (set(), False)):
            with self.subTest(features=features):
                self.assertIs(IDENTITIES._fixture_cfg(predicate, features), expected)
                self.assertEqual(self.selected_attributes(f"#[cfg_attr({predicate}, kernel)]", features),
                                 int(expected))
                with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "token bound"):
                    IDENTITIES._fixture_cfg(oversized, features)
                with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "token bound"):
                    self.selected_attributes(f"#[cfg_attr(test, cfg({oversized}))]", features)
                unknown = 'any(' + 'feature="left",' * 127 + 'unknown)'
                with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "cfg predicate"):
                    IDENTITIES._fixture_cfg(unknown, features)
                with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "cfg predicate"):
                    self.selected_attributes(f"#[cfg_attr(test, cfg({unknown}))]", features)

    def test_non_test_library_skips_test_only_modules_and_declarations(self):
        self.sources[self.library] = (
            '#[cfg(not(test))] mod left;\n'
            '#[cfg(any(test, feature = "right"))] mod absent;\n'
            '#[cfg(test)] mod tests { #[kernel] fn same() {} }\n'
            '#[cfg(test)] #[kernel] fn same() {}\n'
        )
        result = self.validate()
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(result["sourceBoundPairCount"], 0)

    def test_test_cfg_does_not_admit_conditional_crates_or_transforming_attributes(self):
        for source, error in (
            ("#![cfg(test)]\nmod left;", "conditional fixture crate/module"),
        ):
            self.sources[self.library] = source
            with self.subTest(source=source), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, error):
                self.validate(False)

    def selected_attributes(self, attributes, features=(), budget=None):
        source = "// UTF-8: \u03bb\n" + attributes + "\nfn same() {}\n"
        code = self.scanner._rust_code_without_comments_and_literals(source)
        selected, modules = IDENTITIES._fixture_declarations(
            source, set(features), self.scanner.ordinary_rust_function_items,
            lambda _: (code, self.scanner._rust_delimiters(code)),
            budget if budget is not None else IDENTITIES._Budget(1000))
        self.assertEqual(modules, [])
        for item in selected:
            self.assertEqual(item["kernelSymbol"], "same")
            self.assertEqual(item["functionUtf8Offset"], source.encode().index(b"same"))
        return len(selected)

    def test_conditional_attributes_use_effective_not_lexical_kernel_attribution(self):
        for attributes, count in (
            ("#[cfg_attr(test, kernel)]", 0),
            ("#[cfg_attr(not(test), kernel)]", 1),
            ("#[cfg_attr(test, kernel)] #[kernel]", 1),
            ("#[cfg_attr(not(test), allow(kernel), doc(kernel))]", 0),
            ("#[cfg_attr(test,)]", 0),
            ("#[cfg_attr(not(test), kernel /* outer /* nested */ end */ (), /* trailing */)]", 1),
            ("#[cfg_attr(not(test), inline(/* before */ always /* after */),)]", 0),
            ("#[cfg_attr(not(test), cfg_attr(not(test), kernel),)]", 1),
            ("#[cfg_attr(test, cfg_attr(not(test), kernel))]", 0),
            ("#[cfg_attr(not(test), cfg_attr(test, kernel))]", 0),
            ("#[cfg_attr(not(test), allow(dead_code), kernel(typed, launch(max = [64, 1, 1])),)]", 1),
            ("#[cfg_attr(not(test), cfg(test), kernel)] #[cfg(not(test))]", 0),
            ("#[cfg(test)] #[cfg_attr(not(test), kernel)]", 0),
            ("#[cfg_attr(test, cfg(test))] #[kernel]", 1),
            ("#![cfg_attr(not(test), allow(kernel))] #[kernel]", 1),
        ):
            with self.subTest(attributes=attributes):
                self.assertEqual(self.selected_attributes(attributes), count)
        attributes = ('#[cfg_attr(feature = "left", kernel)]'
                      '#[cfg_attr(not(feature = "left"), kernel)]')
        for features in ((), ("left",)):
            self.assertEqual(self.selected_attributes(attributes, features), 1)

    def test_conditional_attribute_module_selection_and_inactive_kernel_roster(self):
        self.sources[self.library] = '#[cfg_attr(not(test), cfg(feature = "left"))] mod left;'
        self.assertEqual(self.validate()["unresolvedBindings"], [])
        self.sources[self.library] = '#[cfg_attr(not(test), cfg(test))] mod left;'
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "roster differs"):
            self.validate(False)
        self.sources[self.library] = "mod left;"
        self.sources[self.path] = "#[cfg_attr(test, kernel)] fn same() {}"
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "roster differs"):
            self.validate(False)

    def test_conditional_attributes_reject_unsupported_syntax_even_when_inactive(self):
        for predicate in ("test", "not(test)"):
            for child in ('path = "other.rs"', "unknown", "kernel::rewrite", "inline::rewrite(always)",
                          "cfg(unknown)", "cfg_attr(unknown, kernel)", "no_std", "kernel() trailing",
                          'kernel "trailing"', 'inline(always "extra")', ",", "kernel,,inline"):
                attributes = f"#[cfg_attr({predicate}, {child})]"
                with self.subTest(attributes=attributes), self.assertRaises(IDENTITIES.KernelInventoryError):
                    self.selected_attributes(attributes)
        for attributes in (
            "#[cfg_attr()]", "#[cfg_attr(test)]", "#[cfg_attr(, kernel)]",
            "#[cfg_attr(not(test), kernel)] #[kernel]",
            "#[cfg_attr(not(test), kernel, kernel)]",
            "#![cfg_attr(not(test), kernel)]", "#![cfg_attr(not(test), cfg(test))]",
            "#[cfg_attr(not(test), inline(unknown))]",
        ):
            with self.subTest(attributes=attributes), self.assertRaises(IDENTITIES.KernelInventoryError):
                self.selected_attributes(attributes)
        for source in (
            "#[cfg_attr(not(test), allow(unused))] m! { () => {} }",
            "#[cfg_attr(not(test), allow(unused))] include!(\"other.rs\");",
        ):
            with self.subTest(source=source), self.assertRaisesRegex(
                    IDENTITIES.KernelInventoryError, "unsupported fixture item"):
                self.selected_attributes(source)

    def test_conditional_attribute_depth_bytes_children_and_work_are_bounded(self):
        self.assertEqual(self.selected_attributes(
            "#[" + "cfg_attr(not(test), " * 32 + "kernel" + ")" * 32 + "]"), 1)
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "nesting bound"):
            self.selected_attributes("#[" + "cfg_attr(test, " * 33 + "kernel" + ")" * 33 + "]")
        for predicate in ("test", "not(test)"):
            self.assertEqual(self.selected_attributes(
                f"#[cfg_attr({predicate}, " + ", ".join(["inline"] * 64) + ")]"), 0)
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "child bound"):
                self.selected_attributes(f"#[cfg_attr({predicate}, " + ", ".join(["inline"] * 65) + ")]")
        self.assertEqual(self.selected_attributes("#[" + "inline".ljust(8192) + "]"), 0)
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "byte bound"):
            self.selected_attributes("#[" + "inline".ljust(8193) + "]")
        attributes = "#[cfg_attr(test, inline)]"
        exact = IDENTITIES._Budget(1000)
        self.selected_attributes(attributes, budget=exact)
        self.assertEqual(exact.attribute_visits, 2)
        self.selected_attributes(attributes, budget=IDENTITIES._Budget(exact.used))
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound"):
            self.selected_attributes(attributes, budget=IDENTITIES._Budget(exact.used - 1))

    def test_record_and_attribute_quotas_are_independent_and_aggregate(self):
        for reverse in (False, True):
            budget = IDENTITIES._Budget(2)
            actions = [lambda: budget.rows([None, None], "records"),
                       lambda: [budget.visit_attribute() for _ in range(2)]]
            for action in actions[::(-1 if reverse else 1)]:
                action()
            self.assertEqual((budget.used, budget.attribute_visits), (2, 2))
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound"):
                budget.rows([None], "records")
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "aggregate visit bound"):
                budget.visit_attribute()
        for predicate in ("test", "not(test)"):
            attributes = f"#[cfg_attr({predicate}, inline, inline)]"
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "aggregate visit bound"):
                self.selected_attributes(attributes, budget=IDENTITIES._Budget(2))
            budget = IDENTITIES._Budget(6)
            for _ in range(2):
                self.selected_attributes(attributes, budget=budget)
            self.assertEqual((budget.used, budget.attribute_visits), (4, 6))
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "aggregate visit bound"):
                self.selected_attributes(attributes, budget=budget)

    def test_fixture_attributes_reject_qualified_benign_names(self):
        for attribute in ("doc::rewrite", "inline::rewrite", "doc::rewrite(hidden)",
                          "inline :: rewrite(always)", "allow::rewrite(dead_code)",
                          "doc /* before */ :: /* after */ rewrite",
                          "inline /* before */ :: /* after */ rewrite(always)"):
            for location in ("function", "module"):
                self.setUp()
                if location == "function":
                    self.sources[self.path] = f"#[{attribute}]\n#[kernel] fn same() {{}}\n"
                else:
                    self.sources[self.library] = f'#[{attribute}]\n#[cfg(feature = "left")] mod left;\n'
                source = self.sources[self.path]
                digest = hashlib.sha256(source.encode()).hexdigest()
                self.tab.update(sourceSha256=digest, displayedSha256=digest, displayedUtf8Bytes=len(source.encode()))
                self.row["functionUtf8Offset"] = self.scanner.ordinary_rust_function_items(source)[0]["functionUtf8Offset"]
                with self.subTest(attribute=attribute, location=location), self.assertRaisesRegex(
                    IDENTITIES.KernelInventoryError, "unsupported fixture selection attribute",
                ):
                    self.validate(False)

    def test_inner_no_std_preserves_selection_and_pending_qualification(self):
        for runtime in (True, False):
            self.setUp()
            self.sources[self.library] = "#![no_std]\n" + self.sources[self.library].split("\n", 1)[1]
            before = copy.deepcopy((self.manifest, self.runtime, self.sources))
            with self.subTest(runtime=runtime):
                result = self.validate(runtime)
                self.assertEqual(result["kernelIdentities"][0]["variants"], variants())
                self.assertEqual(result["sourceBoundVariantCount"], 0)
                self.assertEqual(result["sourceBoundPairCount"], 0)
                self.assertEqual(result["variantBindingStatus"], "pending")
                self.assertEqual((self.manifest, self.runtime, self.sources), before)

    def test_no_std_extension_rejects_outer_arguments_and_hidden_literals(self):
        for attribute in (
            "#[no_std]", "#![no_std()]", "#![no_std(extra)]",
            '#![no_std = "unexpected"]', '#![no_std "unexpected"]',
            '#![no_std r#"unexpected"#]', "#![no_std::other]", "#![no_main]", "#![unknown]",
            '#![cfg_attr(feature = "left", no_std)]',
        ):
            self.setUp()
            self.sources[self.library] = attribute + "\n" + self.sources[self.library].split("\n", 1)[1]
            with self.subTest(attribute=attribute), self.assertRaisesRegex(
                IDENTITIES.KernelInventoryError, "unsupported fixture selection attribute",
            ):
                self.validate(False)

    def test_fixture_attributes_require_one_supported_body_form(self):
        for attribute in ("doc[hidden]", "inline{always}", "inline(sometimes)",
                          "doc(hidden) (alias)", "allow", "doc", 'doc = "description"'):
            self.setUp()
            self.sources[self.path] = f"#[{attribute}]\n#[kernel] fn same() {{}}\n"
            source = self.sources[self.path]
            digest = hashlib.sha256(source.encode()).hexdigest()
            self.tab.update(sourceSha256=digest, displayedSha256=digest, displayedUtf8Bytes=len(source.encode()))
            self.row["functionUtf8Offset"] = self.scanner.ordinary_rust_function_items(source)[0]["functionUtf8Offset"]
            with self.subTest(attribute=attribute), self.assertRaisesRegex(
                IDENTITIES.KernelInventoryError, "unsupported fixture selection attribute",
            ):
                self.validate(False)

    def test_fixture_attributes_accept_supported_builtin_forms(self):
        for attribute in ("allow(dead_code)", "deny(missing_docs)", "forbid(unsafe_code)",
                          "warn(dead_code)", "doc(hidden)", "inline", "inline(always)",
                          "inline(never)", "inline /* gap */ (always)"):
            for location in (("function",) if attribute.startswith("inline") else ("function", "module")):
                self.setUp()
                if location == "function":
                    self.sources[self.path] = f"#[{attribute}]\n#[kernel] fn same() {{}}\n"
                else:
                    self.sources[self.library] = f'#[{attribute}]\n#[cfg(feature = "left")] mod left;\n'
                source = self.sources[self.path]
                digest = hashlib.sha256(source.encode()).hexdigest()
                self.tab.update(sourceSha256=digest, displayedSha256=digest, displayedUtf8Bytes=len(source.encode()))
                self.row["functionUtf8Offset"] = self.scanner.ordinary_rust_function_items(source)[0]["functionUtf8Offset"]
                self.runtime["lessons"][0]["codeTabs"][0]["displayedCode"] = source
                with self.subTest(attribute=attribute, location=location):
                    result = self.validate()
                    self.assertEqual(result["unresolvedBindings"], [])
                    self.assertEqual(result["kernelIdentities"][0]["variants"], variants())

    def test_duplicate_selected_symbol_and_unknown_cfg_stay_unresolved(self):
        self.fixture["compilerInput"]["features"] = ["left", "right"]
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "ambiguous feature-selected"):
            self.validate(False)
        self.setUp()
        self.sources[self.path] = '#[cfg(unknown)] #[kernel] fn same() {}'
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "cfg predicate"):
            self.validate(False)
        self.row.update(kernelIds=[], bindingStatus="pending")
        result = self.validate(False)
        self.assertEqual(result["pendingDisplayItemCount"], 1)
        self.assertFalse(result["inventoryComplete"])

    def test_fixture_source_work_and_cfg_bounds(self):
        successes = []
        for limit in range(1, 100):
            try:
                self.validate(False, max_records=limit)
                successes.append(limit)
                break
            except IDENTITIES.KernelInventoryError:
                pass
        self.assertTrue(successes)
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound"):
            self.validate(False, max_records=successes[0] - 1)
        source_bytes = sum(len(self.sources[path].encode()) for path in (self.library, self.path))
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", source_bytes):
            self.validate(False)
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", source_bytes - 1):
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "aggregate byte bound"):
                self.validate(False)
        for cfg in ("not(" * 34 + 'feature="left"' + ")" * 34,
                    "all(" + ','.join(['feature="left"'] * 128) + ")", " " * 8193):
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "bound"):
                IDENTITIES._fixture_cfg(cfg, {"left"})

    def test_repeated_displays_use_one_fixture_selection_cache(self):
        with mock.patch.object(IDENTITIES, "_fixture_attribute", wraps=IDENTITIES._fixture_attribute) as visits:
            self.validate()
            expected_visits = visits.call_count
        self.manifest["curriculum"]["lessons"][0]["codeTabs"].append(copy.deepcopy(self.tab))
        self.manifest["kernelInventory"]["displayItems"].append({**self.row, "tabOrdinal": 1})
        self.runtime["lessons"][0]["codeTabs"].append(copy.deepcopy(self.runtime["lessons"][0]["codeTabs"][0]))
        loader = mock.Mock(return_value=(self.library, self.sources, ["left"]))
        with mock.patch.object(IDENTITIES, "_fixture_attribute", wraps=IDENTITIES._fixture_attribute) as visits:
            self.assertEqual(self.validate(load_fixture_sources=loader)["unresolvedBindings"], [])
            self.assertEqual(visits.call_count, expected_visits)
        loader.assert_called_once_with(self.fixture)

    def excerpt(self, parts, explicit=True):
        source = "\n\n".join(parts)
        digest = hashlib.sha256(source.encode()).hexdigest()
        self.tab.update(sourceDigestScope="displayed", sourceSha256=digest,
                        displayedSha256=digest, displayedUtf8Bytes=len(source.encode()),
                        sourceFragmentsSha256=(
                            [hashlib.sha256(part.encode()).hexdigest() for part in parts] if explicit else None))
        self.runtime["lessons"][0]["codeTabs"][0] = {
            "displayedCode": source, "sourceFragments": list(parts) if explicit else None,
        }
        self.row["functionUtf8Offset"] = next(
            item["functionUtf8Offset"] for item in self.scanner.ordinary_rust_function_items(source)
            if item["kernelSymbol"] == "same")

    def test_exact_excerpts_keep_variants_pending_and_require_live_coverage(self):
        for fragment in ("#[kernel] fn same() {}", "fn same() {}"):
            for explicit in (False, True):
                self.setUp()
                self.excerpt([fragment], explicit)
                before = copy.deepcopy((self.manifest, self.runtime, self.sources))
                with self.subTest(fragment=fragment, explicit=explicit):
                    result = self.validate()
                    self.assertEqual(result["unresolvedBindings"], [])
                    self.assertEqual(result["requiredPairCount"], 1)
                    self.assertEqual(result["kernelIdentities"][0]["variants"], variants())
                    self.assertEqual(result["sourceBoundVariantCount"], 0)
                    self.assertEqual(result["sourceBoundPairCount"], 0)
                    pending = self.validate(False)
                    self.assertIsNone(pending["requiredPairCount"])
                    self.assertFalse(pending["inventoryComplete"])
                    self.assertFalse(pending["runtimeCensusValidated"])
                    self.assertTrue(any("selection" in row for row in pending["unresolvedBindings"]))
                    self.assertEqual((self.manifest, self.runtime, self.sources), before)

    def test_multiple_exact_nonoverlapping_fragments_and_adjacent_spans(self):
        parts = ["// before\n", "#[kernel] fn same() {}", "\n// after\n"]
        self.sources[self.path] = "".join(parts)
        for order in (parts, parts[::-1]):
            self.excerpt(order)
            with self.subTest(order=order):
                self.assertEqual(self.validate()["unresolvedBindings"], [])

    def test_excerpt_metadata_and_unmatched_bytes_refuse(self):
        for mutation, message, runtime in (
            (lambda: self.tab.update(sourceSha256="0" * 64), "digests differ", False),
            (lambda: self.tab.update(sourceFragmentsSha256=[]), "empty", False),
            (lambda: self.tab.update(sourceFragmentsSha256=["bad"]), "SHA-256", False),
            (lambda: self.runtime["lessons"][0]["codeTabs"][0].update(sourceFragments=None),
             "fragment coverage", True),
            (lambda: self.runtime["lessons"][0]["codeTabs"][0].update(sourceFragments=[""]),
             "reconstruct", True),
            (lambda: self.tab.update(sourceFragmentsSha256=["0" * 64]), "fragment digest", True),
            (lambda: self.excerpt(["#[kernel] fn same() { }"]), "differs from its physical source", True),
        ):
            self.setUp()
            self.excerpt(["#[kernel] fn same() {}"])
            mutation()
            with self.subTest(message=message), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, message):
                self.validate(runtime)
        self.setUp()
        self.excerpt(["#[kernel] fn same() {}"], explicit=False)
        self.runtime["lessons"][0]["codeTabs"][0]["sourceFragments"] = ["#[kernel] fn same() {}"]
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "unexpected fragments"):
            self.validate()

    def test_exact_excerpt_requires_its_selected_feature_and_physical_function(self):
        left = "#[kernel] fn same() { let value = 1; }"
        right = "#[kernel] fn same() { let value = 2; }"
        self.sources[self.library] = "mod left;"
        self.sources[self.path] = f'#[cfg(feature="left")]\n{left}\n#[cfg(feature="right")]\n{right}\n'
        for feature, selected, foreign in (("left", left, right), ("right", right, left)):
            self.fixture["compilerInput"]["features"] = [feature]
            self.excerpt([selected])
            self.assertEqual(self.validate()["unresolvedBindings"], [])
            self.excerpt([foreign])
            with self.subTest(feature=feature), self.assertRaisesRegex(
                IDENTITIES.KernelInventoryError, "exact selected physical function",
            ):
                self.validate()
        self.sources[self.path] = "fn same() {}"
        self.excerpt(["fn same() {}"])
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "roster differs"):
            self.validate()

    def test_excerpt_raw_names_and_utf8_byte_offsets(self):
        fragment = "// UTF-8: \u03bb\n#[kernel] fn r#same() {}"
        self.sources[self.path] = "// physical prefix\n" + fragment + "\n"
        self.excerpt([fragment])
        self.assertEqual(self.validate()["unresolvedBindings"], [])
        byte_offset = self.row["functionUtf8Offset"]
        physical_offset = self.scanner.ordinary_rust_function_items(self.sources[self.path])[0]["functionUtf8Offset"]
        for wrong, reason in ((fragment.index("r#same"), "live function census"),
                              (physical_offset, "outside a Rust tab"),
                              (byte_offset + 2, "live function census")):
            self.row["functionUtf8Offset"] = wrong
            with self.subTest(wrong=wrong), self.assertRaisesRegex(
                IDENTITIES.KernelInventoryError, reason,
            ):
                self.validate()

    def test_overlapping_repeated_empty_and_ambiguous_physical_fragments_refuse(self):
        for parts, message in (
            (["#[kernel] fn same() {}", "fn same() {}"], "overlap or repeat"),
            (["#[kernel] fn same() {}", "#[kernel] fn same() {}"], "overlap or repeat"),
            (["#[kernel] fn same() {}", ""], "nonempty"),
            (["#[kernel] fn same() {}", "// absent"], "differs from its physical source"),
        ):
            self.setUp()
            self.excerpt(parts)
            with self.subTest(parts=parts), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, message):
                self.validate()
        self.setUp()
        fragment = "#[kernel] fn same() {}"
        self.sources[self.path] = f'#[cfg(feature="left")]\n{fragment}\n#[cfg(feature="right")]\n{fragment}'
        self.excerpt([fragment])
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "ambiguous physical"):
            self.validate()
        self.setUp()
        self.sources[self.path] += "// repeated\n// repeated\n"
        self.excerpt([fragment, "// repeated"])
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "ambiguous physical"):
            self.validate()

    def test_mapping_checks_each_row_and_is_independent_of_row_order(self):
        same, other = "#[kernel] fn same() {}", "#[kernel] fn other() {}"
        self.sources[self.path] = f'#[cfg(feature="left")]\n{same}\n#[cfg(feature="right")]\n{same}\n{other}'
        self.fixture["compilerInput"]["kernelSymbols"].append("other")
        self.manifest["kernelInventory"]["kernels"].append({
            "kernelId": "other", "variants": variants(),
            "selections": [{"kind": "fixture", "fixtureId": "left", "kernelSymbol": "other"}],
        })
        self.excerpt([same, other])
        second = {**self.row, "kernelSymbol": "other", "kernelIds": ["other"],
                  "functionUtf8Offset": self.scanner.ordinary_rust_function_items(
                      self.runtime["lessons"][0]["codeTabs"][0]["displayedCode"])[1]["functionUtf8Offset"]}
        rows = self.manifest["kernelInventory"]["displayItems"]
        rows.append(second)
        for _ in range(2):
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "ambiguous physical"):
                self.validate()
            rows.reverse()
        # A unique full-byte mapping still cannot authorize a second occurrence
        # of the same symbol inside an ordinary nested helper.
        self.setUp()
        fragment = "#[kernel] fn same() { fn same() {} }"
        self.sources[self.path] = fragment
        self.excerpt([fragment])
        functions = self.scanner.ordinary_rust_function_items(fragment)
        self.manifest["kernelInventory"]["displayItems"].append({
            **self.row, "functionUtf8Offset": functions[1]["functionUtf8Offset"],
        })
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "exact selected physical function"):
            self.validate()

    def test_fragment_count_and_exact_matching_and_record_boundaries(self):
        fragment = "#[kernel] fn same() {}"
        parts = [f"// piece-{index:02d}\n" for index in range(63)] + [fragment]
        self.sources[self.path] = "".join(parts)
        self.excerpt(parts)
        self.assertEqual(self.validate()["unresolvedBindings"], [])
        self.excerpt(parts + ["// extra"])
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "fragment metadata"):
            self.validate()
        self.setUp()
        self.sources[self.path] = "// " + "x" * 1024 + "\n" + fragment
        self.excerpt([fragment])
        exact = 2 * len(self.sources[self.path].encode())
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", exact):
            self.assertEqual(self.validate()["unresolvedBindings"], [])
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", exact - 1):
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "aggregate byte-span bound"):
                self.validate()
        minimum = None
        for limit in range(1, 150):
            try:
                self.validate(max_records=limit)
                minimum = limit
                break
            except IDENTITIES.KernelInventoryError:
                pass
        self.assertIsNotNone(minimum)
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound"):
            self.validate(max_records=minimum - 1)

    def test_repeated_excerpt_tabs_reuse_physical_selection_and_encoding(self):
        self.excerpt(["#[kernel] fn same() {}"])
        self.manifest["curriculum"]["lessons"][0]["codeTabs"].append(copy.deepcopy(self.tab))
        self.manifest["kernelInventory"]["displayItems"].append({**self.row, "tabOrdinal": 1})
        self.runtime["lessons"][0]["codeTabs"].append(copy.deepcopy(self.runtime["lessons"][0]["codeTabs"][0]))
        loader = mock.Mock(return_value=(self.library, self.sources, ["left"]))
        with mock.patch.object(IDENTITIES, "_utf8", wraps=IDENTITIES._utf8) as encoded:
            result = self.validate(load_fixture_sources=loader)
            copies = [call for call in encoded.call_args_list if call.args[1] == "fixture matched source"]
        self.assertEqual(result["unresolvedBindings"], [])
        self.assertEqual(len(copies), 1)
        loader.assert_called_once_with(self.fixture)

    def test_shared_fragment_cache_does_not_transfer_feature_selection(self):
        fragment = ('#[cfg(feature="left")]\n#[kernel] fn same() { let left = 1; }\n'
                    '#[cfg(feature="right")]\n#[kernel] fn same() { let right = 2; }')
        self.sources[self.library] = "mod left;"
        self.sources[self.path] = "// physical prefix\n" + fragment
        self.add_right()
        self.manifest["compilerFixtures"][1]["compilerInput"]["sourcePaths"] = [self.path]
        self.excerpt([fragment])
        functions = self.scanner.ordinary_rust_function_items(fragment)
        rows = self.manifest["kernelInventory"]["displayItems"]
        rows.append({**self.row, "kernelIds": ["right"],
                     "functionUtf8Offset": functions[1]["functionUtf8Offset"]})
        expected = copy.deepcopy(rows)
        self.assertEqual(self.validate()["unresolvedBindings"], [])
        for first in (0, 1):
            rows[:] = copy.deepcopy([expected[first], expected[1 - first]])
            rows[1]["kernelIds"] = rows[0]["kernelIds"].copy()
            with self.subTest(first=first), self.assertRaisesRegex(
                IDENTITIES.KernelInventoryError, "exact selected physical function",
            ):
                self.validate()

    def test_multiple_valid_rows_share_one_exact_search_budget(self):
        fragment = "#[kernel] fn same() {}\n#[kernel] fn other() {}"
        self.sources[self.path] = "// " + "x" * 1024 + "\n" + fragment
        self.fixture["compilerInput"]["kernelSymbols"].append("other")
        self.manifest["kernelInventory"]["kernels"].append({
            "kernelId": "other", "variants": variants(),
            "selections": [{"kind": "fixture", "fixtureId": "left", "kernelSymbol": "other"}],
        })
        self.excerpt([fragment])
        functions = self.scanner.ordinary_rust_function_items(fragment)
        rows = self.manifest["kernelInventory"]["displayItems"]
        rows.append({**self.row, "kernelSymbol": "other", "kernelIds": ["other"],
                     "functionUtf8Offset": functions[1]["functionUtf8Offset"]})
        exact = 2 * len(self.sources[self.path].encode())
        for _ in range(2):
            with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", exact):
                self.assertEqual(self.validate()["unresolvedBindings"], [])
            rows.reverse()

    def binding(self, implementation="left", reference=None):
        fixture = next(row for row in self.manifest["compilerFixtures"] if row["fixtureId"] == implementation)
        inputs = fixture["compilerInput"]
        path = inputs["sourcePaths"][0]
        source = self.sources[path]
        return {
            "implementationKernelId": implementation,
            "selection": reference or {"kind": "fixture", "fixtureId": implementation, "kernelSymbol": "same"},
            "selectionSha256": selection_digest(inputs, "same"), "sourcePath": path,
            "sourceSha256": hashlib.sha256(source.encode()).hexdigest(),
            "functionUtf8Offset": self.scanner.ordinary_rust_function_items(source)[0]["functionUtf8Offset"],
        }

    def bind(self, kind="simt", implementation="left"):
        variant = next(row for row in self.manifest["kernelInventory"]["kernels"][0]["variants"]
                       if row["kind"] == kind)
        variant.update(status="source-bound", source=self.binding(implementation))
        return variant["source"]

    def add_right(self):
        fixture = copy.deepcopy(self.fixture)
        fixture["fixtureId"] = "right"
        fixture["compilerInput"].update(features=["right"], sourcePaths=["example/src/right.rs"])
        self.manifest["compilerFixtures"].append(fixture)
        self.manifest["kernelInventory"]["kernels"].append({
            "kernelId": "right", "variants": variants(),
            "selections": [{"kind": "fixture", "fixtureId": "right", "kernelSymbol": "same"}],
        })

    def test_source_bound_variant_is_detached_unqualified_and_not_a_display_join(self):
        binding = self.bind()
        self.row.update(bindingStatus="pending", kernelIds=[])
        before = copy.deepcopy((self.manifest, self.runtime, self.sources))
        result = self.validate()
        self.assertEqual(result["sourceBoundVariantCount"], 1)
        self.assertEqual([result[key] for key in COUNT_KEYS], [2, 1, 1])
        self.assertEqual(result["sourceBoundPairCount"], 0)
        self.assertEqual(result["variantBindingStatus"], "partial")
        self.assertFalse(result["inventoryComplete"])
        self.assertIsNone(result["requiredPairCount"])
        self.assertEqual(result["pendingDisplayItemCount"], 1)
        retained = result["kernelIdentities"][0]["variants"][0]
        self.assertEqual(retained["source"], binding)
        self.assertEqual(retained["status"], "source-bound")
        self.assertTrue(retained["blocker"]["reason"])
        retained["source"]["selection"]["fixtureId"] = "changed"
        self.assertEqual((self.manifest, self.runtime, self.sources), before)

    def test_distinct_file_feature_implementations_associate_without_merging_obligations(self):
        self.add_right()
        simt = self.bind()
        tile = self.bind("tile", "right")
        self.assertNotEqual(simt["sourcePath"], tile["sourcePath"])
        self.assertNotEqual(simt["selectionSha256"], tile["selectionSha256"])
        loader = mock.Mock(side_effect=lambda fixture: (
            self.library, self.sources, fixture["compilerInput"]["features"]))
        result = self.validate(load_fixture_sources=loader)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(result["sourceBoundVariantCount"], 2)
        self.assertEqual(result["sourceBoundPairCount"], 1)
        self.assertEqual([result[key] for key in COUNT_KEYS], [4, 2, 0])
        self.assertEqual(result["knownKernelIdentityCount"], 2)
        self.assertEqual(result["variantBindingStatus"], "partial")
        self.assertIsNone(result["requiredPairCount"])
        self.assertFalse(result["inventoryComplete"])
        self.assertEqual(result["kernelIdentities"][1]["variants"], variants())
        self.assertEqual([row["kernelId"] for row in result["kernelIdentities"]], ["left", "right"])

    def test_mixed_binding_changes_variant_counts_not_pair_counts(self):
        self.manifest["kernelInventory"]["kernels"][0]["variants"].append({**variants()[0], "kind": "mixed"})
        result = self.validate()
        self.assertEqual([result[key] for key in COUNT_KEYS], [3, 3, 0])
        self.bind("simt")
        self.bind("tile")
        for pending in (1, 0):
            if not pending:
                self.bind("mixed")
            result = self.validate()
            self.assertEqual([result[key] for key in COUNT_KEYS], [3, pending, 0])
            self.assertEqual(result["sourceBoundVariantCount"], 3 - pending)
            self.assertEqual(result["sourceBoundPairCount"], 1)

    def test_source_binding_rejects_stale_fields_and_foreign_identity(self):
        mutations = (
            ("implementationKernelId", "missing", "does not belong"),
            ("selection", {"kind": "fixture", "fixtureId": "missing", "kernelSymbol": "same"}, "does not belong"),
            ("selection", case_ref(0), "does not belong"),
            ("selectionSha256", "0" * 64, "selection digest is stale"),
            ("selectionSha256", "F" * 64, "lowercase SHA-256"),
            ("sourcePath", "example/src/right.rs", "not an exact selected source"),
            ("sourceSha256", "0" * 64, "exact current source occurrence"),
            ("sourceSha256", "bad", "lowercase SHA-256"),
            ("functionUtf8Offset", 0, "exact current source occurrence"),
            ("functionUtf8Offset", True, "nonnegative integer"),
        )
        for field, value, message in mutations:
            self.setUp()
            self.bind()[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, message):
                self.validate(False)
        self.setUp()
        self.add_right()
        binding = self.bind("simt", "right")
        binding["implementationKernelId"] = "left"
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "does not belong"):
            self.validate(False)

    def test_source_binding_rejects_feature_revision_and_physical_source_substitution(self):
        for field, value in (("features", ["right"]), ("defaultFeatures", True),
                             ("sourceClosureSha256", "4" * 64), ("cargoLockSha256", "5" * 64),
                             ("packageManifestSha256", "6" * 64),
                             ("cargoTarget", {"kind": "lib", "name": "other", "sourcePath": "src/lib.rs"})):
            self.setUp()
            self.bind()
            self.fixture["compilerInput"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "selection digest is stale"):
                self.validate(False)
        self.setUp()
        binding = self.bind()
        self.fixture["compilerInput"]["features"] = ["right"]
        binding["selectionSha256"] = selection_digest(self.fixture["compilerInput"], "same")
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "exact current source occurrence"):
            self.validate(False)
        self.setUp()
        self.bind()
        self.sources[self.path] += "// changed\n"
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "exact current source occurrence"):
            self.validate(False)

    def test_source_bound_state_requires_source_validation_and_cannot_claim_qualification(self):
        self.bind()
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "physical source validation"):
            self.validate(False, load_fixture_sources=None)
        original = copy.deepcopy(self.manifest)
        for changes, message in (({"status": "qualified"}, "not qualification"),
                                 ({"status": "pending"}, "pending variant cannot"),
                                 ({"source": None}, "variant source has unexpected"),
                                 ({"blocker": None}, "blocker has unexpected")):
            self.manifest = copy.deepcopy(original)
            self.manifest["kernelInventory"]["kernels"][0]["variants"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, message):
                self.validate(False)

    def test_source_driver_implementation_uses_its_own_physical_selection_loader(self):
        # The selected inner library is not the host driver's test harness.
        self.sources[self.library] += (
            '#[cfg(test)] mod tests { #[kernel] fn same() {} }\n'
        )
        binding = self.bind()
        inputs = self.fixture["compilerInput"]
        reference = case_ref(0)
        self.tab["sourceItem"] = {
            "compilerInput": {key: value for key, value in inputs.items() if key not in {"features", "kernelSymbols"}},
            "cases": [{"kernelSymbol": "same", "features": ["left"], "target": "gfx950",
                       "displayedFragmentOrdinal": 0, "expectation": {"kind": "verified-bundle-export"}}],
        }
        self.manifest["compilerFixtures"] = []
        self.manifest["kernelInventory"]["kernels"][0]["selections"] = [reference]
        binding["selection"] = copy.deepcopy(reference)
        self.row.update(bindingStatus="pending", kernelIds=[])
        loader = mock.Mock(return_value=(self.library, self.sources, ["left"]))
        result = self.validate(False, load_source_case_sources=loader)
        self.assertEqual(result["sourceBoundVariantCount"], 1)
        self.assertIsNone(result["requiredPairCount"])
        loader.assert_called_once_with("lesson", self.tab, self.tab["sourceItem"]["cases"][0])
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "source case binding requires"):
            self.validate(False)
        self.tab["sourceItem"]["cases"][0]["expectation"]["kind"] = "rejected"
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "negative"):
            self.validate(False, load_source_case_sources=loader)

    def test_source_binding_record_and_source_byte_limits_are_shared_with_displays(self):
        self.bind()
        self.bind("tile")
        # Binding the same source to two declared modes is not mode verification;
        # this fixture checks only shared traversal and exact accounting.
        self.assertEqual(self.validate(False)["variantBindingStatus"], "source-bound")
        first_success = None
        for limit in range(1, 150):
            try:
                result = self.validate(False, max_records=limit)
                first_success = limit
                break
            except IDENTITIES.KernelInventoryError:
                pass
        self.assertIsNotNone(first_success)
        self.assertEqual(result["sourceBoundVariantCount"], 2)
        self.assertEqual(result["sourceBoundPairCount"], 1)
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound"):
            self.validate(False, max_records=first_success - 1)
        byte_count = sum(len(self.sources[path].encode()) for path in (self.library, self.path))
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", byte_count):
            self.assertEqual(self.validate(False)["sourceBoundPairCount"], 1)
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", byte_count - 1):
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "aggregate byte bound"):
                self.validate(False)


class SiblingModulePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "path_fixture_scanner", PATH.with_name("validate-tutorial-kernel-manifest.py"))
        cls.scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scanner)

    def select(self, sources, features=(), *, library="example/src/lib.rs", symbols=("same",),
               budget=None, caches=None):
        def syntax(source):
            code = self.scanner._rust_code_without_comments_and_literals(source)
            return code, self.scanner._rust_delimiters(code)

        return IDENTITIES._fixture_selection(
            {"compilerInput": {"kernelSymbols": list(symbols)}},
            lambda _: (library, sources, features), self.scanner.ordinary_rust_function_items,
            syntax, budget if budget is not None else IDENTITIES._Budget(4096),
            **(caches or {}))

    def sources(self, declaration):
        return {"example/src/lib.rs": declaration,
                "example/src/selected.rs": "// UTF-8: λ\n#[kernel] pub fn same() {}\n"}

    def test_real_wave_features_select_exact_physical_function_with_warm_caches(self):
        root = PATH.parents[1]
        directory = root / "examples/wave64_collectives_v1/src"
        sources = {p.relative_to(root).as_posix(): p.read_text() for p in directory.glob("*.rs")}
        library = "examples/wave64_collectives_v1/src/lib.rs"
        caches = {"function_cache": {}, "item_cache": {}}
        for features, filename in (((), "kernel.rs"), (("tile-kernel",), "tiled_kernel.rs"),
                                   ((), "kernel.rs"), (("tile-kernel",), "tiled_kernel.rs")):
            with self.subTest(features=features):
                result = self.select(sources, features, library=library,
                                     symbols=("wave64_collectives_v1",), caches=caches)
                path = f"examples/wave64_collectives_v1/src/{filename}"
                source = sources[path]
                offset = source.encode().index(b"wave64_collectives_v1(")
                self.assertEqual(result, {"wave64_collectives_v1": (path, offset, source)})

    def test_literal_path_overrides_default_filename_without_renaming_module(self):
        sources = self.sources('#[path /* selected */ = "selected.rs"] pub mod kernel;')
        sources["example/src/kernel.rs"] = "#[kernel] fn wrong_default() {}"
        sources["example/src/kernel/mod.rs"] = "#[kernel] fn wrong_directory() {}"
        result = self.select(sources)
        source = sources["example/src/selected.rs"]
        self.assertEqual(result["same"], ("example/src/selected.rs",
                                        source.encode().index(b"same"), source))
        # Two physical files cannot declare the same active Rust module name.
        sources["example/src/lib.rs"] += '\n#[path = "other.rs"] mod kernel;'
        sources["example/src/other.rs"] = "#[kernel] fn other() {}"
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "ambiguous"):
            self.select(sources, symbols=("same", "other"))

    def test_inactive_literal_is_validated_but_not_resolved(self):
        sources = self.sources('#[cfg(test)] #[path = "absent.rs"] mod unused;\n'
                               '#[path = "selected.rs"] mod kernel;')
        self.assertEqual(self.select(sources)["same"][0], "example/src/selected.rs")
        sources["example/src/lib.rs"] = sources["example/src/lib.rs"].replace("cfg(test)", "cfg(not(test))")
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "missing"):
            self.select(sources)

    def test_unsafe_or_nonliteral_paths_fail_even_when_inactive(self):
        for literal in ('"../selected.rs"', '"/selected.rs"', '"dir/selected.rs"',
                        '"./selected.rs"', '"selected/../selected.rs"', '"selected.rs/"',
                        '"selected.txt"', '""', 'r"selected.rs"', 'concat!("selected", ".rs")',
                        r'"selected\x2ers"', r'"dir\selected.rs"', '"selected.rs" "extra"'):
            for cfg in ("test", "not(test)"):
                with self.subTest(literal=literal, cfg=cfg), self.assertRaises(IDENTITIES.KernelInventoryError):
                    self.select(self.sources(f'#[cfg({cfg})] #[path = {literal}] mod kernel;'))

    def test_path_attributes_require_one_direct_outer_module_item(self):
        for item in ('fn helper() {}', 'struct Data;', 'const VALUE: u32 = 1;',
                     'use crate::selected;', 'mod kernel {}', 'include!("selected.rs");', ''):
            for cfg in ("test", "not(test)"):
                with self.subTest(item=item, cfg=cfg), self.assertRaises(IDENTITIES.KernelInventoryError):
                    self.select(self.sources(f'#[cfg({cfg})] #[path = "selected.rs"] {item}'))
        for attributes in ('#![path = "selected.rs"]',
                           '#[path = "selected.rs"] #[path = "selected.rs"]',
                           '#[path = "selected.rs"] #[kernel]',
                           '#[cfg_attr(test, path = "selected.rs")]',
                           '#[cfg_attr(not(test), path = "selected.rs")]',
                           '#[cfg_attr(test, cfg_attr(test, path = "selected.rs"))]'):
            for cfg in ("test", "not(test)"):
                with self.subTest(attributes=attributes, cfg=cfg), self.assertRaises(IDENTITIES.KernelInventoryError):
                    self.select(self.sources(f'#[cfg({cfg})] {attributes} mod kernel;'))

    def test_missing_repeated_and_nested_physical_paths_fail(self):
        for declaration in ('#[path = "absent.rs"] mod kernel;',
                            '#[path = "lib.rs"] mod kernel;',
                            '#[path = "selected.rs"] mod first; #[path = "selected.rs"] mod second;'):
            with self.subTest(declaration=declaration), self.assertRaisesRegex(
                    IDENTITIES.KernelInventoryError, "ambiguous|missing"):
                self.select(self.sources(declaration))
        for cfg in ("test", "not(test)"):
            sources = self.sources('mod selected;')
            sources["example/src/selected.rs"] = f'#[cfg({cfg})] #[path = "nested.rs"] mod child;'
            sources["example/src/nested.rs"] = "#[kernel] fn same() {}"
            with self.subTest(cfg=cfg), self.assertRaises(IDENTITIES.KernelInventoryError):
                self.select(sources)

    def test_literal_selection_preserves_byte_record_and_attribute_budgets(self):
        sources = self.sources('#[path = "selected.rs"] mod kernel;')
        total = sum(len(source.encode()) for source in sources.values())
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", total):
            self.assertEqual(set(self.select(sources)), {"same"})
        with mock.patch.object(IDENTITIES, "MAX_RUNTIME_BYTES", total - 1), self.assertRaisesRegex(
                IDENTITIES.KernelInventoryError, "aggregate byte bound"):
            self.select(sources)
        with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "record bound|visit bound"):
            self.select(sources, budget=IDENTITIES._Budget(1))
        for cfg in ("test", "not(test)"):
            attributes = '#[cfg(' + cfg + ')]' + '#[path = "selected.rs"]'
            with self.subTest(cfg=cfg), self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "visit bound"):
                self.select(self.sources(attributes + ' mod kernel;'), budget=IDENTITIES._Budget(1))
            with self.assertRaisesRegex(IDENTITIES.KernelInventoryError, "byte bound"):
                self.select(self.sources('#[cfg(' + cfg + ')] #[path' + ' ' * 8192 + '= "selected.rs"] mod kernel;'))


load_tests = runpy.run_path(str(Path(__file__).with_name("_tutorial_kernel_function_cache_tests.py")))["make_loader"](FixtureDisplayTests, IDENTITIES)
load_tests = runpy.run_path(str(Path(__file__).with_name("_tutorial_kernel_item_cache_tests.py")))["make_loader"](FixtureDisplayTests, IDENTITIES, load_tests)
if __name__ == "__main__":
    unittest.main()

"""Actual current source associations must not transfer historical execution evidence."""

import copy
import hashlib
import json


class CurrentEntryBindingTests(FixtureBindingTests):
    cases = (
        ("typed-vecadd", "gfx942-typed-vecadd-source", "vecadd",
         "examples/vecadd/src/lib.rs", 859, 299,
         "96a6d5774c0d36661540829025194c918100abcce53fed91cae3c909aea67345"),
        ("lds-barriers-atomics", "gfx942-workgroup-collectives",
         "lds_publish_read_reduce_i32_v1", "examples/workgroup_sync_v1/src/kernel.rs",
         2626, 1194, "b0074b426ef8ad0b9eea91e933e76dd03240852ce4ce976ccc89c1f2c7f1b515"),
    )

    def component(self, case):
        lesson_id, fixture_id, symbol, path, size, offset, digest = case
        kernel_id = f"fixture:{fixture_id}:{symbol}"
        lesson = copy.deepcopy(next(row for row in self.original["curriculum"]["lessons"]
                                    if row["lessonId"] == lesson_id))
        ordinal = 0 if lesson_id == "lds-barriers-atomics" else 5
        self.assertEqual(len(lesson["codeTabs"]), 5 if ordinal == 0 else 6)
        tab = lesson["codeTabs"][ordinal]
        tab["ordinal"] = 0
        lesson["codeTabs"] = [tab]
        fixture = copy.deepcopy(next(row for row in self.original["compilerFixtures"]
                                     if row["fixtureId"] == fixture_id))
        kernel = copy.deepcopy(next(row for row in self.original["kernelInventory"]["kernels"]
                                    if row["kernelId"] == kernel_id))
        displays = copy.deepcopy([row for row in self.original["kernelInventory"]["displayItems"]
                                   if row["lessonId"] == lesson_id and row["tabOrdinal"] == ordinal])
        for row in displays:
            row["tabOrdinal"] = 0
        display = next(row for row in displays if row["bindingStatus"] == "fixture-source-contract")
        document = {
            **self.original,
            "compilerFixtures": [fixture],
            "curriculum": {**self.original["curriculum"], "lessons": [lesson]},
            "kernelInventory": {
                **self.original["kernelInventory"], "kernels": [kernel],
                "negativeCases": [], "displayItems": displays,
            },
        }
        physical = (ROOT / path).read_bytes()
        self.assertEqual(len(physical), size)
        self.assertEqual(hashlib.sha256(physical).hexdigest(), digest)
        self.assertEqual(tab["sourcePath"], path)
        self.assertEqual(tab["sourceSha256"], digest)
        self.assertEqual(display["functionUtf8Offset"], offset)
        self.assertEqual(physical[offset:offset + len(symbol)], symbol.encode())
        runtime = {
            "schema": self.parent.SITE_INVENTORY_SCHEMA,
            "site": document["curriculum"]["site"],
            "lessons": [{"id": lesson_id, "codeTabs": [
                {**tab, "displayedCode": physical.decode("utf-8"), "sourceFragments": None},
            ]}],
        }
        return document, runtime, tab, kernel, display

    def test_current_entries_bind_without_qualifying_pairs(self):
        for case in self.cases:
            with self.subTest(lesson=case[0]):
                document, runtime, _, kernel, _ = self.component(case)
                before = copy.deepcopy((document, runtime))
                result = self.check(document, runtime)
                self.assertTrue(result["runtimeCensusValidated"])
                if case[0] == "typed-vecadd":
                    self.assertEqual(
                        [{key: row[key] for key in
                          ("lessonId", "tabOrdinal", "functionUtf8Offset", "kernelSymbol")}
                         for row in result["unresolvedBindings"]],
                        [{"lessonId": "typed-vecadd", "tabOrdinal": 0,
                          "functionUtf8Offset": 567, "kernelSymbol": "vecadd"}],
                    )
                    self.assertTrue(result["unresolvedBindings"][0]["reason"])
                else:
                    self.assertEqual(result["unresolvedBindings"], [])
                self.assertEqual(result["knownKernelIdentityCount"], 1)
                self.assertEqual(result["displayItemCount"], 2 if case[0] == "typed-vecadd" else 1)
                self.assertEqual(result["sourceBoundVariantCount"], 1)
                self.assertEqual(result["sourceBoundPairCount"], 0)
                self.assertEqual([(row["kind"], row["status"]) for row in kernel["variants"]],
                                 [("simt", "source-bound"), ("tile", "pending")])
                self.assertEqual((document, runtime), before)

    def test_changed_binding_occurrence_and_false_qualification_reject(self):
        mutations = ("sourceSha256", "selectionSha256", "sourcePath", "functionUtf8Offset",
                     "fixture", "missing", "display-offset", "status", "authority")
        for case in self.cases:
            for mutation in mutations:
                document, runtime, _, kernel, display = self.component(case)
                variant = kernel["variants"][0]
                binding = variant["source"]
                if mutation in ("sourceSha256", "selectionSha256"):
                    binding[mutation] = "0" * 64
                elif mutation == "sourcePath":
                    binding[mutation] = "examples/fill/src/lib.rs"
                elif mutation == "functionUtf8Offset":
                    binding[mutation] += 1
                elif mutation == "fixture":
                    binding["selection"]["fixtureId"] = "gfx942-fill-simulation"
                elif mutation == "missing":
                    document["kernelInventory"]["displayItems"] = []
                elif mutation == "display-offset":
                    display["functionUtf8Offset"] += 1
                elif mutation == "status":
                    variant["status"] = "qualified"
                else:
                    variant["qualified"] = True
                with self.subTest(lesson=case[0], mutation=mutation), self.assertRaises(SystemExit):
                    self.check(document, runtime)

    def test_rehashed_display_still_requires_the_actual_physical_source(self):
        for case in self.cases:
            document, runtime, tab, kernel, _ = self.component(case)
            live = runtime["lessons"][0]["codeTabs"][0]
            live["displayedCode"] += "\n// not present in the selected physical source\n"
            payload = live["displayedCode"].encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            for row in (tab, live):
                row.update(displayedUtf8Bytes=len(payload), displayedSha256=digest,
                           sourceSha256=digest)
            kernel["variants"][0]["source"]["sourceSha256"] = digest
            self.parent.validate_site_inventory(document["curriculum"], runtime)
            with self.subTest(lesson=case[0]), self.assertRaisesRegex(
                    SystemExit, "exact current source occurrence"):
                self.parent.validate_kernel_inventory(document, runtime, repo_root=ROOT)

    def test_rehashed_workgroup_features_cannot_select_a_different_entry(self):
        for defaults in (False, True):
            document, _, _, kernel, display = self.component(self.cases[1])
            fixture = document["compilerFixtures"][0]
            if defaults:
                fixture["compilerInput"]["defaultFeatures"] = True
            else:
                fixture["compilerInput"]["features"] = ["scoped-atomic-kernel"]
            fixture["compilerInput"]["contractSha256"] = self.parent.fixture_input_contract_sha256(fixture)
            binding = kernel["variants"][0]["source"]
            kernel["variants"][0].update(status="pending", source=None)
            display_binding = display["bindingStatus"], display["kernelIds"]
            display.update(bindingStatus="pending", kernelIds=[])
            report = self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)
            binding["selectionSha256"] = report["kernelIdentities"][0]["selectionSha256"]
            kernel["variants"][0].update(status="source-bound", source=binding)
            display["bindingStatus"], display["kernelIds"] = display_binding
            with self.subTest(defaults=defaults), self.assertRaisesRegex(
                    SystemExit, "feature-selected fixture kernel roster differs|selected source"):
                self.parent.validate_kernel_inventory(document, None, repo_root=ROOT)

    def test_historical_vecadd_and_workgroup_archive_metadata_stay_separate(self):
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()

        typed = next(row for row in self.original["curriculum"]["lessons"]
                     if row["lessonId"] == "typed-vecadd")
        self.assertEqual(len(typed["codeTabs"]), 6)
        self.assertEqual(digest(typed["codeTabs"][:5]),
                         "8684d655640e34b39b11d3f5a945882839079fbf2ba782580bcfa14af5e7458a")
        rows = [row for row in self.original["kernelInventory"]["displayItems"]
                if row["lessonId"] == "typed-vecadd" and row["tabOrdinal"] < 5]
        self.assertEqual(len(rows), 2)
        self.assertEqual(digest(rows), "3f804ff440f95096651083bd005309245e4e57c4f662cbcee2039bf291ed388d")
        self.assertTrue(all(row["bindingStatus"] == "pending" and row["kernelIds"] == [] for row in rows))
        lds = next(row for row in self.original["curriculum"]["lessons"]
                   if row["lessonId"] == "lds-barriers-atomics")
        self.assertEqual(len(lds["codeTabs"]), 5)
        self.assertEqual(digest(lds["codeTabs"][1:4]), "80b14f1bdeb34eb73f805558e0b904b739eff39658bbbd24fe9b934e026f76e4")
        self.assertEqual(lds["codeTabs"][0]["evidenceId"], "workgroup-sync-current-source-v1")
        self.assertEqual(lds["codeTabs"][0]["sourceSha256"],
                         "b0074b426ef8ad0b9eea91e933e76dd03240852ce4ce976ccc89c1f2c7f1b515")
        self.assertEqual(lds["codeTabs"][0]["sourceItemStatus"], "pending")
        rows = [row for row in self.original["kernelInventory"]["displayItems"]
                if row["lessonId"] == "lds-barriers-atomics"]
        self.assertEqual([(row["tabOrdinal"], row["functionUtf8Offset"], row["bindingStatus"])
                          for row in rows], [(0, 1194, "fixture-source-contract")])
        self.assertEqual(rows[0]["kernelIds"],
                         ["fixture:gfx942-workgroup-collectives:lds_publish_read_reduce_i32_v1"])

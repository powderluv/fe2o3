"""Neutral item-span cache controls sharing the physical fixture harness."""

import copy
from unittest import mock


def make_loader(fixture_tests, identities, previous=None):
    class ItemCacheTests(fixture_tests):
        def syntax(self, source):
            code = self.scanner._rust_code_without_comments_and_literals(source)
            return code, self.scanner._rust_delimiters(code)

        def declarations(self, source, budget, cache, functions, **kwargs):
            return identities._fixture_declarations(
                source, kwargs.pop("features", set()), self.scanner.ordinary_rust_function_items,
                kwargs.pop("rust_syntax", self.syntax), budget,
                function_cache=functions, item_cache=cache, **kwargs)

        def attribute_heavy_selection(self):
            self.sources[self.library] = (
                "#[allow(dead_code)]\n" * 100 + "fn helper() {}\nmod left;\n")
            self.bind()

        def test_reserved_report_rows_do_not_reduce_attribute_capacity(self):
            self.attribute_heavy_selection()
            budgets = []
            visit = identities._Budget.visit_attribute

            def counted(budget):
                if not budgets:
                    budgets.append(budget)
                return visit(budget)

            # Report rows consume record capacity, not attribute traversal work.
            with mock.patch.object(identities._Budget, "visit_attribute", counted):
                result = self.validate(False, max_records=100)
            self.assertEqual(result["sourceBoundVariantCount"], 1)
            self.assertFalse(result["inventoryComplete"])
            self.assertEqual(budgets[0].attribute_visits, 102)
            self.assertLess(budgets[0].used, 100)

        def test_inventory_attribute_limit_has_exact_and_one_short_boundaries(self):
            self.attribute_heavy_selection()
            for limit in (102, 101):
                with self.subTest(limit=limit), mock.patch.object(identities, "MAX_ATTRIBUTE_VISITS", limit):
                    if limit == 102:
                        self.assertEqual(self.validate(False, max_records=100)["sourceBoundVariantCount"], 1)
                    else:
                        with self.assertRaisesRegex(identities.KernelInventoryError, "attributes.*visit bound"):
                            self.validate(False, max_records=100)
            with self.assertRaisesRegex(identities.KernelInventoryError, "record bound"):
                self.validate(False, max_records=1)

        def test_attribute_allowance_keeps_inactive_attribute_validation(self):
            self.attribute_heavy_selection()
            self.sources[self.library] += '#[cfg(feature = "never")] #[unknown] fn hidden() {}\n'
            with self.assertRaisesRegex(identities.KernelInventoryError, "unsupported fixture selection attribute"):
                self.validate(False, max_records=100)

        def test_separate_attribute_limit_charges_replays_and_keeps_item_limit(self):
            source = "#[inline] fn helper() {}\n"
            for explicit_limit, successful_visits in ((None, 2), (3, 3)):
                with self.subTest(explicit_limit=explicit_limit):
                    cache, functions = {}, {}
                    budget = identities._Budget(2, source_item_limit=4, attribute_limit=explicit_limit)
                    for _ in range(successful_visits):
                        self.declarations(source, budget, cache, functions)
                    self.assertEqual((budget.used, budget.attribute_visits), (2, successful_visits))
                    with self.assertRaisesRegex(identities.KernelInventoryError, "attributes.*visit bound"):
                        self.declarations(source, budget, cache, functions)
            cache, functions = {}, {}
            budget = identities._Budget(2, source_item_limit=1, attribute_limit=3)
            self.declarations(source, budget, cache, functions)
            with self.assertRaisesRegex(identities.KernelInventoryError, "source items.*visit bound"):
                self.declarations(source, budget, cache, functions)

        def test_separate_attribute_limit_refuses_invalid_values_and_zero_work(self):
            for value in (-1, True, 1.5, "3"):
                with self.subTest(value=value), self.assertRaisesRegex(
                        identities.KernelInventoryError, "attribute visit limit must be a nonnegative integer"):
                    identities._Budget(2, attribute_limit=value)
            budget = identities._Budget(2, attribute_limit=0)
            with self.assertRaisesRegex(identities.KernelInventoryError, "attributes.*visit bound"):
                self.declarations("#[inline] fn helper() {}\n", budget, {}, {})
            self.assertEqual(budget.attribute_visits, 0)

        def large_feature_catalog(self):
            # Distinct features replay one shared library with many ordinary
            # items. Coordinates are reusable; feature decisions and visits are not.
            self.sources[self.library] = "".join(
                f"const VALUE_{index}: u32 = {index};\n" for index in range(2050)
            ) + self.sources[self.library]
            self.add_right()
            self.bind()
            self.bind("tile", "right")

        def test_inventory_feature_replays_can_exceed_materialized_record_ceiling(self):
            self.large_feature_catalog()
            budgets = []
            visit = identities._Budget.visit_source_item

            def counted(budget):
                if not budgets:
                    budgets.append(budget)
                return visit(budget)

            with mock.patch.object(identities._Budget, "visit_source_item", counted):
                result = self.validate(False)
            self.assertEqual(result["sourceBoundVariantCount"], 2)
            self.assertFalse(result["inventoryComplete"])
            # 2050 constants and two module items per feature, plus one kernel.
            self.assertEqual(budgets[0].source_item_visits, 4106)
            self.assertLess(budgets[0].used, 4096)
            self.assertLess(budgets[0].attribute_visits, 4096)

        def test_inventory_item_visit_ceiling_has_exact_and_one_short_boundaries(self):
            self.large_feature_catalog()
            for ceiling in (4106, 4105):
                with self.subTest(ceiling=ceiling), mock.patch.object(
                        identities, "MAX_SOURCE_ITEM_VISITS", ceiling):
                    if ceiling == 4106:
                        self.assertEqual(self.validate(False)["sourceBoundVariantCount"], 2)
                    else:
                        with self.assertRaisesRegex(identities.KernelInventoryError,
                                                    "fixture source items exceed their aggregate visit bound"):
                            self.validate(False)

        def test_larger_visit_allowance_keeps_record_and_inactive_feature_refusals(self):
            self.large_feature_catalog()
            with self.assertRaisesRegex(identities.KernelInventoryError, "record bound"):
                self.validate(False, max_records=32)
            self.sources[self.library] += (
                '#[cfg(feature = "never")] #[cfg(unknown)] fn hidden() {}\n')
            with self.assertRaisesRegex(identities.KernelInventoryError, "cfg predicate"):
                self.validate(False)

        def test_separate_item_visit_limit_still_charges_cached_replays(self):
            source = "fn helper() {}\n"
            for explicit_limit, successful_visits in ((None, 2), (3, 3)):
                with self.subTest(explicit_limit=explicit_limit):
                    cache, functions = {}, {}
                    budget = identities._Budget(2, source_item_limit=explicit_limit)
                    for _ in range(successful_visits):
                        self.declarations(source, budget, cache, functions)
                    self.assertEqual((budget.used, budget.source_item_visits), (2, successful_visits))
                    with self.assertRaisesRegex(identities.KernelInventoryError, "aggregate visit bound"):
                        self.declarations(source, budget, cache, functions)
            with self.assertRaisesRegex(identities.KernelInventoryError, "record bound"):
                self.declarations(source, identities._Budget(1, source_item_limit=3), {}, {})
            cache, functions, budget = {}, {}, identities._Budget(2, source_item_limit=3)
            attributed = "#[inline] #[allow(dead_code)] fn helper() {}\n"
            self.declarations(attributed, budget, cache, functions)
            with self.assertRaisesRegex(identities.KernelInventoryError, "attributes.*visit bound"):
                self.declarations(attributed, budget, cache, functions)

        def test_separate_item_visit_limit_refuses_invalid_values_and_zero_work(self):
            for value in (-1, True, 1.5, "3"):
                with self.subTest(value=value), self.assertRaisesRegex(
                        identities.KernelInventoryError, "source item visit limit must be a nonnegative integer"):
                    identities._Budget(2, source_item_limit=value)
            budget = identities._Budget(2, source_item_limit=0)
            with self.assertRaisesRegex(identities.KernelInventoryError, "aggregate visit bound"):
                self.declarations("fn helper() {}\n", budget, {}, {})
            self.assertEqual(budget.source_item_visits, 0)

        def test_item_cache_reuses_actual_utf8_raw_identifier_body_spans(self):
            source = "// \u03bb\n#[kernel]\nfn r#type() {}\n"
            cache, functions, budget = {}, {}, identities._Budget(20)
            syntax = mock.Mock(wraps=self.syntax)
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                first = self.declarations(source, budget, cache, functions, rust_syntax=syntax)
                retained = cache[source]
                second = self.declarations(source, budget, cache, functions, rust_syntax=syntax)
                self.assertEqual(parse.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first[0][0]["functionUtf8Offset"], source.encode().index(b"r#type"))
            start, boundary, end = source.index("fn "), source.index("{"), source.index("}") + 1
            body = (start, boundary, end, len(source[:start].encode()),
                    len(source[:boundary].encode()), len(source[:end].encode()))
            self.assertEqual(retained, ((((False, source.index("["), source.index("]") + 1),), body),))
            self.assertIs(cache[source], retained)
            self.assertEqual(syntax.call_args_list, [mock.call(source), mock.call(source)])
            self.assertEqual((budget.used, budget.source_item_visits, budget.attribute_visits), (2, 2, 2))

        def test_empty_and_terminal_attribute_cache_entries_are_replayed(self):
            for source in ("", "// empty item roster\n", "#![no_std]\n"):
                with self.subTest(source=source):
                    cache, functions, budget = {}, {}, identities._Budget(10)
                    syntax = mock.Mock(wraps=self.syntax)
                    with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                        for _ in range(2):
                            self.assertEqual(self.declarations(source, budget, cache, functions,
                                                               rust_syntax=syntax), ([], []))
                        parse.assert_not_called()
                    self.assertIsInstance(cache[source], tuple)
                    self.assertEqual(len(cache[source]), int(source.startswith("#!")))
                    if source.startswith("#!"):
                        self.assertEqual(cache[source][0], (((True, 2, source.index("]") + 1),), None))
                        self.assertEqual((budget.used, budget.source_item_visits, budget.attribute_visits), (1, 2, 2))
                    else:
                        self.assertEqual((budget.used, budget.source_item_visits, budget.attribute_visits), (0, 0, 0))
                    self.assertEqual(syntax.call_count, 2)

        def test_cached_spans_do_not_cache_feature_eligibility(self):
            source = '#[cfg(feature = "left")] #[kernel] fn same() {}\n' \
                     '#[cfg(feature = "right")] #[kernel] fn same() {}\n'
            cache, functions, budget = {}, {}, identities._Budget(20)
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                left = self.declarations(source, budget, cache, functions, features={"left"})
                right = self.declarations(source, budget, cache, functions, features={"right"})
                self.assertEqual(parse.call_count, 2)
            self.assertEqual([item["functionUtf8Offset"] for item in left[0]], [source.encode().index(b"same")])
            self.assertEqual([item["functionUtf8Offset"] for item in right[0]], [source.encode().rindex(b"same")])
            self.assertEqual((budget.used, budget.source_item_visits, budget.attribute_visits), (4, 4, 8))

        def test_changed_source_bytes_receive_fresh_spans_and_coordinates(self):
            original = "#[kernel] fn same() {}\n"
            changed = "// \u03bb shifts physical bytes\n" + original
            cache, functions, budget = {}, {}, identities._Budget(20)
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                first = self.declarations(original, budget, cache, functions)
                second = self.declarations(changed, budget, cache, functions)
                self.assertEqual([call.args[0] for call in parse.call_args_list], [original, changed])
            self.assertEqual(set(cache), {original, changed})
            self.assertNotEqual(cache[original], cache[changed])
            self.assertEqual(first[0][0]["functionUtf8Offset"], original.encode().index(b"same"))
            self.assertEqual(second[0][0]["functionUtf8Offset"], changed.encode().index(b"same"))

        def test_failed_traversal_never_publishes_partial_spans_even_for_inactive_items(self):
            source = 'fn helper() {}\n#[cfg(feature = "left")] #[cfg(unknown)] fn other() {}\n'
            cache, functions, budget = {}, {}, identities._Budget(100)
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                for _ in range(2):
                    with self.assertRaisesRegex(identities.KernelInventoryError, "cfg predicate"):
                        self.declarations(source, budget, cache, functions, features={"right"})
                    self.assertNotIn(source, cache)
                self.assertEqual(parse.call_count, 2)

        def test_new_item_span_records_have_exact_and_one_short_bounds(self):
            source = "fn helper() {}\n"
            for limit in (2, 1):
                with self.subTest(limit=limit):
                    cache, functions, budget = {}, {}, identities._Budget(limit)
                    if limit == 2:
                        self.assertEqual(self.declarations(source, budget, cache, functions), ([], []))
                        self.assertEqual((budget.used, budget.source_item_visits), (2, 1))
                        self.assertIn(source, cache)
                    else:
                        with self.assertRaisesRegex(identities.KernelInventoryError, "fixture source items.*record bound"):
                            self.declarations(source, budget, cache, functions)
                        self.assertNotIn(source, cache)
                        self.assertEqual(budget.used, 1)

        def test_replayed_item_visits_have_separate_cumulative_exact_and_short_bounds(self):
            source = "fn helper() {}\n"
            for limit in (3, 2):
                with self.subTest(limit=limit):
                    cache, functions, budget = {}, {}, identities._Budget(limit)
                    for _ in range(2):
                        self.assertEqual(self.declarations(source, budget, cache, functions), ([], []))
                    if limit == 3:
                        self.assertEqual(self.declarations(source, budget, cache, functions), ([], []))
                    else:
                        with self.assertRaisesRegex(identities.KernelInventoryError, "aggregate visit bound"):
                            self.declarations(source, budget, cache, functions)
                    self.assertEqual((budget.used, budget.source_item_visits), (2, limit))

        def test_cached_spans_still_apply_macros_only_context(self):
            for source in ("const VALUE: u32 = 0;\n", "#![no_std]\n"):
                with self.subTest(source=source):
                    cache, functions, budget = {}, {}, identities._Budget(20)
                    self.declarations(source, budget, cache, functions)
                    retained = cache[source]
                    with self.assertRaisesRegex(identities.KernelInventoryError, "included fixture source"):
                        self.declarations(source, budget, cache, functions, macros_only=True)
                    self.assertIs(cache[source], retained)

        def test_cached_spans_refuse_newly_active_inline_module(self):
            source = '#[cfg(feature = "left")] mod nested {}\n'
            cache, functions, budget = {}, {}, identities._Budget(20)
            self.assertEqual(self.declarations(source, budget, cache, functions, features={"right"}), ([], []))
            self.assertIn(source, cache)
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                with self.assertRaisesRegex(identities.KernelInventoryError, "inline fixture modules"):
                    self.declarations(source, budget, cache, functions, features={"left"})
                parse.assert_not_called()

        def test_cached_item_selection_retains_source_byte_and_attribute_limits(self):
            exact_bytes = 2 * sum(len(self.sources[path].encode()) for path in (self.library, self.path))
            for byte_limit, record_limit, error in ((exact_bytes, 20, None),
                                                     (exact_bytes - 1, 20, "aggregate byte bound"),
                                                     (exact_bytes, 9, "aggregate visit bound")):
                with self.subTest(byte_limit=byte_limit, record_limit=record_limit):
                    cache, functions, budget = {}, {}, identities._Budget(record_limit)
                    loader = mock.Mock(return_value=(self.library, self.sources, ["left"]))

                    def select():
                        return identities._fixture_selection(
                            self.fixture, loader, self.scanner.ordinary_rust_function_items, self.syntax,
                            budget, function_cache=functions, item_cache=cache)

                    with mock.patch.object(identities, "MAX_RUNTIME_BYTES", byte_limit):
                        select()
                        with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                            if error is None:
                                select()
                            else:
                                with self.assertRaisesRegex(identities.KernelInventoryError, error):
                                    select()
                            parse.assert_not_called()
                    self.assertEqual(budget.source_bytes, exact_bytes)
                    if record_limit == 9:
                        self.assertEqual((budget.used, budget.source_item_visits, budget.attribute_visits), (4, 6, 9))
                    self.assertEqual(loader.call_args_list, [mock.call(self.fixture)] * 2)

        def test_cached_includes_preserve_current_closure_and_scope_shadowing_refusals(self):
            self.sources[self.library] = 'include!("macros.rs");\nmod left;\n'
            macros = "example/src/macros.rs"
            self.sources[macros] = "macro_rules! inert { () => {} }\n"
            cache, functions, budget = {}, {}, identities._Budget(100)
            loader = mock.Mock(side_effect=lambda fixture: (
                self.library, self.sources, fixture["compilerInput"]["features"]))

            def select():
                return identities._fixture_selection(
                    self.fixture, loader, self.scanner.ordinary_rust_function_items, self.syntax,
                    budget, function_cache=functions, item_cache=cache)

            select()
            shadow = "macro_rules! include { () => {} }\n"
            self.declarations(shadow, budget, cache, functions)
            self.sources[macros] = shadow
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                with self.assertRaisesRegex(identities.KernelInventoryError, "ambiguous macro or import scope"):
                    select()
                parse.assert_not_called()
            del self.sources[macros]
            with self.assertRaisesRegex(identities.KernelInventoryError, "ambiguous or missing fixture module"):
                select()
            self.assertEqual(loader.call_args_list, [mock.call(self.fixture)] * 3)

        def test_public_validation_item_cache_is_local_and_still_reloads_selections(self):
            self.add_right()
            self.bind()
            self.bind("tile", "right")
            before = copy.deepcopy((self.manifest, self.sources))
            loader = mock.Mock(side_effect=lambda fixture: (
                self.library, self.sources, fixture["compilerInput"]["features"]))
            with mock.patch.object(identities, "_fixture_item_span", wraps=identities._fixture_item_span) as parse:
                for ordinal in (1, 2):
                    result = self.validate(False, load_fixture_sources=loader)
                    self.assertEqual(result["sourceBoundVariantCount"], 2)
                    self.assertFalse(result["inventoryComplete"])
                    root_calls = [call for call in parse.call_args_list if call.args[0] == self.sources[self.library]]
                    self.assertEqual(len(root_calls), 2 * ordinal)
            self.assertEqual(loader.call_args_list,
                             [mock.call(fixture) for fixture in self.manifest["compilerFixtures"]] * 2)
            self.assertEqual((self.manifest, self.sources), before)

    def load_tests(loader, tests, pattern):
        if previous is not None:
            tests = previous(loader, tests, pattern)
        for name in sorted(ItemCacheTests.__dict__):
            if name.startswith("test_"):
                tests.addTest(ItemCacheTests(name))
        return tests

    return load_tests

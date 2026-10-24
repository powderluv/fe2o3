#!/usr/bin/env python3
"""Component-only CPU adapter tests; doubles never qualify a tutorial suite."""

from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "tutorial_cpu_reference.py"
SPEC = importlib.util.spec_from_file_location("tutorial_cpu_reference", HELPER)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


# Ordered original 14 suites, then the 18 GPT and four row/scalar mode additions.
# These expectations must not be derived from the manifest under test.
CPU_SUITE_CONFIGURATIONS = {
    "cpu-reference-flash-attention": (True, []),
    "cpu-reference-gemm-autoresearch": (True, []),
    "cpu-reference-gfx950-advanced-attention": (True, []),
    "cpu-reference-gfx950-advanced-systems": (True, []),
    "cpu-reference-gfx950-gpt-oss-decode": (True, []),
    "cpu-reference-gfx950-low-precision": (True, []),
    "cpu-reference-moe-grouped-expert": (True, []),
    "cpu-reference-moe-top2": (True, []),
    "cpu-reference-row-softmax": (True, []),
    "cpu-reference-tiled-gemm-paired-default": (True, []),
    "cpu-reference-tiled-gemm-paired-simt": (False, ["kernel-simt-gemm-general"]),
    "cpu-reference-tiled-gemm": (True, []),
    "cpu-reference-typed-vecadd": (True, []),
    "cpu-reference-scalar-gemm": (True, []),
    "cpu-reference-gfx950-gpt-oss-decode-simt": (False, ["kernel-gpt-oss-decode", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-decode-tile": (False, ["kernel-gpt-oss-decode", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-held-fragments-simt": (False, ["kernel-gpt-oss-decode-held-fragments", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-held-fragments-tile": (False, ["kernel-gpt-oss-decode-held-fragments", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-interleaved-stores-simt": (False, ["kernel-gpt-oss-decode-interleaved-stores", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-interleaved-stores-tile": (False, ["kernel-gpt-oss-decode-interleaved-stores", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-materialized-attention-simt": (False, ["kernel-gpt-oss-attention-component", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-materialized-attention-tile": (False, ["kernel-gpt-oss-attention-component", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-materialized-expert-simt": (False, ["kernel-gpt-oss-expert-component", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-materialized-expert-tile": (False, ["kernel-gpt-oss-expert-component", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-materialized-router-simt": (False, ["kernel-gpt-oss-mode-simt", "kernel-gpt-oss-router-component"]),
    "cpu-reference-gfx950-gpt-oss-materialized-router-tile": (False, ["kernel-gpt-oss-mode-tile", "kernel-gpt-oss-router-component"]),
    "cpu-reference-gfx950-gpt-oss-pipelined-attention-simt": (False, ["kernel-gpt-oss-decode-pipelined-attention", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-pipelined-attention-tile": (False, ["kernel-gpt-oss-decode-pipelined-attention", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-scalar-attention-simt": (False, ["kernel-gpt-oss-decode-scalar-attention", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-scalar-attention-tile": (False, ["kernel-gpt-oss-decode-scalar-attention", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-gpt-oss-serial-router-simt": (False, ["kernel-gpt-oss-decode-router-serial", "kernel-gpt-oss-mode-simt"]),
    "cpu-reference-gfx950-gpt-oss-serial-router-tile": (False, ["kernel-gpt-oss-decode-router-serial", "kernel-gpt-oss-mode-tile"]),
    "cpu-reference-gfx950-row-softmax-simt": (False, ["gfx950-target", "qualification-legacy-hip"]),
    "cpu-reference-gfx950-row-softmax-tile": (False, ["gfx950-target", "qualification-legacy-hip", "structured-tile"]),
    "cpu-reference-gfx950-scalar-gemm-simt": (False, ["gfx950-target", "qualification-legacy-hip"]),
    "cpu-reference-gfx950-scalar-gemm-tile": (False, ["gfx950-target", "qualification-legacy-hip", "structured-tile"]),
}

def source_manifest(arguments: list[str]) -> dict:
    return {
        "qualification": {"suites": [{
            "suiteId": "cpu-reference-fixture", "gate": "cpu-reference",
            "command": {"executable": adapter.WRAPPER, "arguments": arguments,
                        "workingDirectory": ".", "environment": [], "timeoutSeconds": 1200},
            "coverage": [{"lessonId": "fixture", "fixtureIds": ["fixture-a", "fixture-b"]}],
        }]},
        "compilerFixtures": [
            {"fixtureId": "fixture-a", "compilerInput": {"packageManifest": arguments[0], "features": ["kernel-a"]}},
            {"fixtureId": "fixture-b", "compilerInput": {"packageManifest": arguments[0], "features": ["kernel-b"]}},
        ],
    }


class AdapterComponents(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cpu-reference-components-")
        self.root = Path(self.temporary.name).resolve()
        self.target = self.root / "build"
        self.target.mkdir()
        self.executable = self.target / "fixture"
        self.executable.write_bytes(b"component artifact; not executed")
        self.executable.chmod(0o700)
        self.source = self.root / "src/lib.rs"
        self.source.parent.mkdir()
        self.source.write_text("#[test] fn component() {}\n")
        self.manifest = self.root / "Cargo.toml"
        self.manifest.write_text('[package]\nname="fixture"\nversion="0.1.0"\n')
        self.expected = {"packageId": "fixture#0.1.0", "name": "fixture", "kind": ["lib"],
                         "crateTypes": ["lib"], "features": [], "source": str(self.source)}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def events(self, states: list[str] | None = None) -> list[dict]:
        states = ["ok"] if states is None else states
        events = [{"reason": "compiler-artifact", "package_id": self.expected["packageId"],
                   "target": {"name": "fixture", "kind": ["lib"], "crate_types": ["lib"],
                              "src_path": str(self.source)}, "features": [], "profile": {"test": True},
                   "executable": str(self.executable)},
                  {"reason": "build-finished", "success": True},
                  {"type": "suite", "event": "started", "test_count": len(states)}]
        for index, state in enumerate(states):
            events.append({"type": "test", "event": "started", "name": f"test_{index}"})
            events.append({"type": "test", "event": state, "name": f"test_{index}"})
        events.append({"type": "suite", "event": "failed" if "failed" in states else "ok",
                       "passed": states.count("ok"), "failed": states.count("failed"),
                       "ignored": states.count("ignored"), "measured": 0, "filtered_out": 0})
        return events

    def observe(self, events: list[dict]) -> dict:
        payload = b"\n".join(json.dumps(event).encode() for event in events) + b"\n"
        return adapter.test_observation(payload, self.expected, self.target)

    def metadata(self, kind: str = "lib") -> dict:
        return {"packages": [{"id": self.expected["packageId"], "manifest_path": str(self.manifest),
                              "targets": [{"name": "fixture", "kind": [kind], "crate_types": [kind],
                                           "src_path": str(self.source), "test": True}]}],
                "resolve": {"nodes": [{"id": self.expected["packageId"], "features": []}]}}

    def test_literal_argument_planning_preserves_default_host_features(self) -> None:
        for arguments in (["examples/fixture/Cargo.toml", "lib"],
                          ["examples/fixture/Cargo.toml", "test", "reference"]):
            manifest = source_manifest(arguments)
            plan = adapter.select_suite(manifest, arguments)
            self.assertEqual(plan["hostConfiguration"]["features"], [])
            self.assertTrue(plan["hostConfiguration"]["defaultFeatures"])
            self.assertEqual(plan["hostConfiguration"]["selector"], arguments[1:])
            self.assertEqual(len(plan["fixtureContracts"]), 2)
            self.assertEqual(manifest, source_manifest(arguments))

    def test_explicit_host_feature_selection_is_canonical_and_fixture_independent(self) -> None:
        cases = [([], True, []), (["--no-default-features"], False, []),
                 (["--features", "a,b"], True, ["a", "b"]),
                 (["--no-default-features", "--features", "a,b"], False, ["a", "b"]),
                 (["--features", "a,b", "--no-default-features"], False, ["a", "b"])]
        for selector in (["lib"], ["test", "reference"]):
            for options, defaults, features in cases:
                arguments = ["examples/fixture/Cargo.toml", *selector, *options]
                manifest = source_manifest(arguments)
                with self.subTest(arguments=arguments):
                    plan = adapter.select_suite(manifest, arguments)
                    configuration = {"defaultFeatures": defaults, "features": features,
                                     "profile": "test", "selector": selector}
                    self.assertEqual(plan["hostConfiguration"], configuration)
                    self.assertEqual(adapter.feature_arguments(configuration),
                                     ([] if defaults else ["--no-default-features"]) +
                                     (["--features", ",".join(features)] if features else []))
                    self.assertEqual(plan["fixtureContracts"][0]["compilerInput"]["features"], ["kernel-a"])
                    self.assertEqual(manifest, source_manifest(arguments))

    def test_host_feature_grammar_refuses_noncanonical_and_nonlocal_names(self) -> None:
        bad = [["--no-default-features", "--no-default-features"], ["--features"],
               ["--features", "a", "--features", "b"], ["--all-features"],
               ["--features=a"], ["--unknown"], ["--"], ["extra"]]
        bad += [["--features", names] for names in
                ("", "a,a", "b,a", ",a", "a,", "a,,b", " a", "a b", "a,b ",
                 "dep/a", "dep:a", "dep?/a", "--a", ".a", "+a", "*")]
        for selector in (["lib"], ["test", "reference"]):
            for options in bad:
                arguments = ["Cargo.toml", *selector, *options]
                with self.subTest(arguments=arguments), self.assertRaises(adapter.ObservationError):
                    adapter.select_suite(source_manifest(arguments), arguments)
        configuration = adapter.parse_selection(["Cargo.toml", "lib", "--features", "a+b,a-b,a.b,a_b"])
        self.assertEqual(configuration["features"], ["a+b", "a-b", "a.b", "a_b"])

    def test_host_features_are_declared_in_parsed_cargo_toml(self) -> None:
        configuration = adapter.parse_selection(["Cargo.toml", "lib", "--features", "a"])
        for document in ('[features]\na=[]\n', '[features]\na=["helper"]\nhelper=[]\n'):
            self.manifest.write_text(document)
            adapter.validate_features(self.manifest, configuration)
        for document in ('', '[features]\nb=[]\n', 'features=[]\n',
                         '[dependencies]\na={version="1",optional=true}\n'):
            self.manifest.write_text(document)
            with self.subTest(document=document), self.assertRaisesRegex(adapter.ObservationError, "undeclared"):
                adapter.validate_features(self.manifest, configuration)
        self.manifest.write_text('[features\n')
        with self.assertRaisesRegex(adapter.ObservationError, "invalid host package manifest"):
            adapter.validate_features(self.manifest, configuration)

    def test_all_declared_cpu_suites_have_exact_component_plans(self) -> None:
        source = Path(__file__).resolve().parents[2] / "config/tutorial-kernel-manifest-v1.json"
        payload = source.read_bytes()
        manifest = adapter.decode_json(payload)
        suites = [suite for suite in manifest["qualification"]["suites"] if suite["gate"] == "cpu-reference"]
        self.assertEqual([suite["suiteId"] for suite in suites], list(CPU_SUITE_CONFIGURATIONS))
        featureful = []
        for suite in suites:
            arguments = suite["command"]["arguments"]
            plan = adapter.select_suite(manifest, arguments)
            self.assertEqual(plan["suiteId"], suite["suiteId"])
            defaults, features = CPU_SUITE_CONFIGURATIONS[suite["suiteId"]]
            self.assertEqual(plan["hostConfiguration"]["features"], features)
            self.assertEqual(plan["hostConfiguration"]["defaultFeatures"], defaults)
            if features:
                featureful.append(suite["suiteId"])
                # A featureful suite must not silently enable defaults or accept
                # an extra feature while retaining the same advertised identity.
                changed_defaults = [arg for arg in arguments if arg != "--no-default-features"]
                changed_features = arguments.copy()
                feature_index = changed_features.index("--features") + 1
                changed_features[feature_index] = ",".join(sorted([*features, "unexpected-feature"]))
                for changed in (changed_defaults, changed_features):
                    with self.subTest(suite=suite["suiteId"], arguments=changed), \
                            self.assertRaises(adapter.ObservationError):
                        adapter.select_suite(manifest, changed)
        self.assertEqual(featureful, [identity for identity, (_, features)
                                     in CPU_SUITE_CONFIGURATIONS.items() if features])
        self.assertEqual(len(featureful), 23)
        self.assertEqual(source.read_bytes(), payload)

    def test_source_loader_uses_the_existing_validator_before_selecting(self) -> None:
        scripts = self.root / "scripts"
        scripts.mkdir()
        config = self.root / "config"
        config.mkdir()
        arguments = ["Cargo.toml", "lib"]
        (config / "tutorial-kernel-manifest-v1.json").write_text(json.dumps(source_manifest(arguments)))
        validator = scripts / "validate-tutorial-kernel-manifest.py"
        validator.write_text("def validate_manifest(root, manifest):\n    raise SystemExit('fixture refuses')\n")
        with self.assertRaisesRegex(adapter.ObservationError, "fixture refuses"):
            adapter.load_plan(self.root, arguments)
        validator.write_text("def validate_manifest(root, manifest):\n    assert manifest['qualification']['suites']\n")
        plan, _ = adapter.load_plan(self.root, arguments)
        self.assertEqual(plan["suiteId"], "cpu-reference-fixture")
        self.assertIn("validatorSha256", plan)
        arguments.extend(["--features", "missing"])
        (config / "tutorial-kernel-manifest-v1.json").write_text(json.dumps(source_manifest(arguments)))
        with self.assertRaisesRegex(adapter.ObservationError, "undeclared local host feature"):
            adapter.load_plan(self.root, arguments)

    def test_wrong_ambiguous_injected_and_foreign_suite_refuse(self) -> None:
        arguments = ["examples/fixture/Cargo.toml", "lib"]
        for mutation in ("duplicate", "executable", "gate", "environment", "directory", "timeout", "foreign"):
            manifest = source_manifest(arguments)
            suite = manifest["qualification"]["suites"][0]
            if mutation == "duplicate":
                manifest["qualification"]["suites"].append(copy.deepcopy(suite))
            elif mutation == "foreign":
                manifest["compilerFixtures"][0]["compilerInput"]["packageManifest"] = "other/Cargo.toml"
            elif mutation == "gate":
                suite["gate"] = "semantic-simulation"
            else:
                key, value = {"executable": ("executable", "sh"), "environment": ("environment", ["RUSTFLAGS=hostile"]),
                              "directory": ("workingDirectory", ".."), "timeout": ("timeoutSeconds", True)}[mutation]
                suite["command"][key] = value
            with self.subTest(mutation=mutation), self.assertRaises(adapter.ObservationError):
                adapter.select_suite(manifest, arguments)
        for extra in (["--ignored"], [";", "touch", "marker"], ["test", "*"]):
            with self.assertRaises(adapter.ObservationError):
                adapter.select_suite(source_manifest(arguments), arguments + extra)

    def test_standard_library_and_auto_integration_harness_identity(self) -> None:
        selected = adapter.selected_target(self.root, self.metadata(), self.manifest, ["lib"])
        self.assertEqual(selected["source"], str(self.source))
        self.source = self.root / "tests/fixture.rs"
        self.source.parent.mkdir()
        self.source.write_text("#[test] fn reference() {}")
        selected = adapter.selected_target(self.root, self.metadata("test"), self.manifest, ["test", "fixture"])
        self.assertEqual(selected["kind"], ["test"])

    def test_custom_disabled_wrong_and_unbound_target_harnesses_refuse(self) -> None:
        for text in ('[lib]\nharness=false\n', '[lib]\ntest=false\n'):
            self.manifest.write_text('[package]\nname="fixture"\nversion="0.1.0"\n' + text)
            with self.assertRaises(adapter.ObservationError):
                adapter.selected_target(self.root, self.metadata(), self.manifest, ["lib"])
        self.manifest.write_text('[package]\nname="fixture"\nversion="0.1.0"\n')
        for field, value in (("src_path", "/proc/self/fd/7/lib.rs"), ("test", False), ("kind", ["bin"])):
            metadata = self.metadata()
            metadata["packages"][0]["targets"][0][field] = value
            with self.assertRaises((adapter.ObservationError, OSError)):
                adapter.selected_target(self.root, metadata, self.manifest, ["lib"])

    def test_pass_failure_ignore_and_empty_remain_distinct(self) -> None:
        for states, result in ((["ok", "ok"], "passed"), (["failed"], "failed"),
                               (["ok", "ignored"], "partial"), ([], "empty")):
            self.assertEqual(self.observe(self.events(states))["outcome"], result)
        self.assertTrue(all(value is False for value in adapter.NO_AUTHORITY.values()))
        observed = self.observe(self.events())["executable"]
        self.assertEqual(observed["digestScope"], "post-execution-on-disk-artifact-only")
        self.assertEqual(set(observed), {"path", "observedSha256", "digestScope"})

    def test_artifact_features_equal_resolved_closure_not_requested_subset(self) -> None:
        self.expected["features"] = ["feature-a", "helper"]
        events = self.events()
        events[0]["features"] = ["feature-a", "helper"]
        self.assertEqual(self.observe(events)["outcome"], "passed")
        for features in ([], ["feature-a"], ["feature-a", "helper", "unexpected"]):
            events[0]["features"] = features
            with self.subTest(features=features), self.assertRaisesRegex(
                    adapter.ObservationError, "wrong harness artifact identity"):
                self.observe(events)

    def test_artifact_observation_uses_ordinary_host_runner_bound_not_data_bounds(self) -> None:
        self.assertEqual(adapter.MAX_EXECUTABLE_BYTES, 1024 * 1024 * 1024)
        self.assertEqual(adapter.ARTIFACT_HASH_CHUNK, 64 * 1024)
        self.assertEqual(adapter.MAX_FILE, 16 * 1024 * 1024)
        self.assertEqual(adapter.MAX_INPUT_BYTES, 256 * 1024 * 1024)
        self.assertEqual(adapter.MAX_STREAM, 16 * 1024 * 1024)
        # Scale only the constants in this component: no large allocation or
        # real test executable is needed to exercise the distinct bound.
        payload = b"artifact above both data limits"
        self.executable.write_bytes(payload)
        with patch.object(adapter, "MAX_EXECUTABLE_BYTES", 32), \
                patch.object(adapter, "ARTIFACT_HASH_CHUNK", 7), \
                patch.object(adapter, "MAX_INPUT_BYTES", 16), \
                patch.object(adapter, "MAX_FILE", 8):
            for limit in (adapter.MAX_FILE, adapter.MAX_INPUT_BYTES):
                with self.assertRaisesRegex(adapter.ObservationError, "invalid/big file"):
                    adapter.read_regular(self.executable, limit)
            observed = self.observe(self.events())
        self.assertEqual(observed["outcome"], "passed")
        self.assertEqual(observed["executable"]["observedSha256"], adapter.digest(payload))
        self.assertEqual(observed["executable"]["digestScope"], "post-execution-on-disk-artifact-only")
        self.assertTrue(all(value is False for value in adapter.NO_AUTHORITY.values()))

    def test_artifact_hash_empty_exact_and_over_limit_are_bounded(self) -> None:
        real_open = Path.open
        requests = []

        @contextlib.contextmanager
        def opened(path, *arguments, **keywords):
            with real_open(path, *arguments, **keywords) as stream:
                def read(size):
                    requests.append(size)
                    return stream.read(size)
                yield SimpleNamespace(read=read, fileno=stream.fileno)

        with patch.object(adapter, "MAX_EXECUTABLE_BYTES", 32), \
                patch.object(adapter, "ARTIFACT_HASH_CHUNK", 7):
            for size in (1, 31, 32):
                payload = b"x" * size
                self.executable.write_bytes(payload)
                requests.clear()
                with patch.object(Path, "open", opened):
                    observed = adapter.artifact_digest(self.executable)
                self.assertEqual(observed, adapter.digest(payload))
                self.assertTrue(requests)
                self.assertTrue(all(0 < size <= 7 for size in requests))
                self.assertLessEqual(sum(requests), len(payload) + 14)
            for size in (0, 33):
                self.executable.write_bytes(b"x" * size)
                with patch.object(Path, "open", side_effect=AssertionError("invalid file opened")), \
                        self.assertRaisesRegex(adapter.ObservationError, "invalid/big artifact"):
                    adapter.artifact_digest(self.executable)

    def test_artifact_hash_refuses_noncanonical_nonregular_and_substituted_open(self) -> None:
        alias = self.target / "alias"
        alias.symlink_to(self.executable)
        for path in (alias, Path("relative"), Path("/proc/self/fd/9"), self.target):
            with self.subTest(path=path), self.assertRaises(adapter.ObservationError):
                adapter.artifact_digest(path)
        other = self.target / "other"
        other.write_bytes(self.executable.read_bytes())
        real_open = Path.open
        opened = real_open(other, "rb")
        with patch.object(Path, "open", return_value=opened), \
                self.assertRaisesRegex(adapter.ObservationError, "changed while opening"):
            adapter.artifact_digest(self.executable)
        self.assertTrue(opened.closed)

    def test_artifact_hash_checks_every_opened_and_final_descriptor_field(self) -> None:
        real_fstat = os.fstat
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        for changed_call in (1, 2):
            for field in fields:
                calls = 0

                def altered(fd):
                    nonlocal calls
                    calls += 1
                    actual = real_fstat(fd)
                    values = {name: getattr(actual, name) for name in fields}
                    if calls == changed_call:
                        values[field] += 1
                    return SimpleNamespace(**values)

                with self.subTest(call=changed_call, field=field), \
                        patch.object(adapter.os, "fstat", side_effect=altered), \
                        self.assertRaisesRegex(adapter.ObservationError, "artifact changed"):
                    adapter.artifact_digest(self.executable)

    def test_artifact_hash_rejects_short_read_growth_mutation_and_path_drift(self) -> None:
        real_open = Path.open
        for mode in ("short", "growth", "overwrite", "replace", "read-error"):
            self.executable.write_bytes(b"abcdefgh")
            retained = []

            @contextlib.contextmanager
            def opened(path, *arguments, **keywords):
                with real_open(path, *arguments, **keywords) as stream:
                    retained.append(stream)
                    first = True

                    def read(size):
                        nonlocal first
                        if not first:
                            return stream.read(size)
                        first = False
                        if mode == "short":
                            return b""
                        if mode == "read-error":
                            raise OSError("injected artifact read failure")
                        data = stream.read(size)
                        if mode == "growth":
                            with real_open(path, "ab") as writer:
                                writer.write(b"extra")
                        elif mode == "overwrite":
                            with real_open(path, "r+b") as writer:
                                writer.write(b"X")
                            current = path.stat()
                            os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
                        else:
                            path.unlink()
                            with real_open(path, "wb") as writer:
                                writer.write(b"abcdefgh")
                        return data

                    yield SimpleNamespace(read=read, fileno=stream.fileno)

            with self.subTest(mode=mode), patch.object(adapter, "MAX_EXECUTABLE_BYTES", 8), \
                    patch.object(adapter, "ARTIFACT_HASH_CHUNK", 3), \
                    patch.object(Path, "open", opened), \
                    self.assertRaises((adapter.ObservationError, OSError)):
                adapter.artifact_digest(self.executable)
            self.assertTrue(retained and all(stream.closed for stream in retained))

    def test_named_integration_accepts_only_exact_metadata_companion_bins(self) -> None:
        self.source = self.root / "tests/fixture.rs"
        self.source.parent.mkdir()
        self.source.write_text("#[test] fn reference() {}")
        companion_source = self.root / "src/main.rs"
        companion_source.write_text("fn main() {}")
        metadata = self.metadata("test")
        metadata["packages"][0]["targets"][0]["crate_types"] = ["bin"]
        target = {"name": "fixture-tool", "kind": ["bin"], "crate_types": ["bin"],
                  "src_path": str(companion_source), "test": True}
        metadata["packages"][0]["targets"].append(target)
        self.expected = adapter.selected_target(self.root, metadata, self.manifest, ["test", "fixture"])
        binary = self.target / "fixture-tool"
        binary.write_bytes(b"unexecuted companion artifact")
        binary.chmod(0o700)
        companion = {"reason": "compiler-artifact", "package_id": self.expected["packageId"],
                     "target": target, "features": [], "profile": {"test": False}, "executable": str(binary)}

        def events_with_companion():
            events = self.events()
            events[0]["target"].update(kind=["test"], crate_types=["bin"])
            events.insert(1, copy.deepcopy(companion))
            return events

        observed = self.observe(events_with_companion())
        self.assertEqual(observed["outcome"], "passed")
        self.assertEqual(observed["counts"]["passed"], 1)
        self.assertEqual(observed["companionArtifacts"]["fixture-tool"]["digestScope"],
                         "post-execution-on-disk-artifact-only")
        for mutation in ("package", "source", "name", "kind", "crate", "features", "profile", "duplicate", "overlap"):
            events = events_with_companion()
            item = events[1]
            if mutation == "package": item["package_id"] = "unrelated"
            elif mutation == "source": item["target"]["src_path"] = str(self.source)
            elif mutation == "name": item["target"]["name"] = "unrelated"
            elif mutation == "kind": item["target"]["kind"] = ["example"]
            elif mutation == "crate": item["target"]["crate_types"] = ["lib"]
            elif mutation == "features": item["features"] = ["extra"]
            elif mutation == "profile": item["profile"]["test"] = True
            elif mutation == "overlap": item["executable"] = str(self.executable)
            else: events.insert(2, copy.deepcopy(item))
            with self.subTest(mutation=mutation), self.assertRaises(adapter.ObservationError):
                self.observe(events)

    def test_all_artifact_identity_fields_and_extra_harnesses_are_checked(self) -> None:
        for mutation in ("package", "source", "kind", "crate", "name", "features", "profile", "duplicate", "alias"):
            events = self.events()
            artifact = events[0]
            if mutation == "package": artifact["package_id"] = "other"
            elif mutation == "source": artifact["target"]["src_path"] = str(self.root / "other.rs")
            elif mutation == "kind": artifact["target"]["kind"] = ["test"]
            elif mutation == "crate": artifact["target"]["crate_types"] = ["bin"]
            elif mutation == "name": artifact["target"]["name"] = "other"
            elif mutation == "features": artifact["features"] = ["unexpected"]
            elif mutation == "profile": artifact["profile"]["test"] = False
            elif mutation == "alias": artifact["executable"] = "/proc/self/fd/8/test"
            else: events.insert(1, copy.deepcopy(artifact))
            with self.subTest(mutation=mutation), self.assertRaises(adapter.ObservationError):
                self.observe(events)

    def test_event_completion_counts_and_selection_are_not_optional(self) -> None:
        for mutation in ("start", "terminal", "bool", "ignored", "filtered", "measured", "trailing", "extra-suite", "duplicate-test", "build"):
            events = self.events()
            if mutation == "start": del events[2]
            elif mutation == "terminal": events.pop()
            elif mutation == "bool": events[-1]["passed"] = True
            elif mutation == "ignored": events[-1]["ignored"] = 1
            elif mutation == "filtered": events[-1]["filtered_out"] = 1
            elif mutation == "measured": events[-1]["measured"] = 1
            elif mutation == "trailing": events.append(copy.deepcopy(events[-1]))
            elif mutation == "extra-suite": events.insert(3, copy.deepcopy(events[2]))
            elif mutation == "duplicate-test": events.insert(4, copy.deepcopy(events[3]))
            else: events[1]["success"] = False
            with self.subTest(mutation=mutation), self.assertRaises(adapter.ObservationError):
                self.observe(events)

    def test_strict_json_and_exact_stream_event_limits(self) -> None:
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'not json', b'[]'):
            with self.assertRaises(adapter.ObservationError):
                adapter.test_observation(data, self.expected, self.target)
        payload = b"\n".join(json.dumps(event).encode() for event in self.events()) + b"\n"
        with patch.object(adapter, "MAX_STREAM", len(payload)):
            self.assertEqual(adapter.test_observation(payload, self.expected, self.target)["outcome"], "passed")
        with patch.object(adapter, "MAX_STREAM", len(payload) - 1), self.assertRaises(adapter.ObservationError):
            adapter.test_observation(payload, self.expected, self.target)
        with patch.object(adapter, "MAX_EVENTS", len(self.events()) - 1), self.assertRaises(adapter.ObservationError):
            self.observe(self.events())

    def test_json_exponent_overflow_cannot_enter_observation_details(self) -> None:
        events = self.events()
        events[-2]["exec_time"] = 0.125
        payload = b"\n".join(json.dumps(event).encode() for event in events) + b"\n"
        observed = adapter.test_observation(payload, self.expected, self.target)
        self.assertEqual(observed["outcome"], "passed")
        self.assertEqual(observed["tests"][0]["details"]["exec_time"], 0.125)
        json.dumps(observed, allow_nan=False)
        for number in (b"1e999", b"-1e999", b"1.7976931348623159e308"):
            with self.subTest(number=number):
                with self.assertRaises(adapter.ObservationError):
                    adapter.decode_json(b'{"nested":[' + number + b']}')
                malformed = payload.replace(b'"exec_time": 0.125', b'"exec_time": ' + number)
                self.assertNotEqual(malformed, payload)
                with self.assertRaises(adapter.ObservationError):
                    adapter.test_observation(malformed, self.expected, self.target)

    def test_input_identity_change_symlink_and_exact_file_bound(self) -> None:
        before = adapter.footprint([], [self.source], time.monotonic() + 5)
        self.source.write_text("changed")
        self.assertNotEqual(before, adapter.footprint([], [self.source], time.monotonic() + 5))
        with patch.object(adapter, "MAX_FILES", 1):
            self.assertEqual(len(adapter.footprint([], [self.source], time.monotonic() + 5)), 1)
        with patch.object(adapter, "MAX_FILES", 0), self.assertRaises(adapter.ObservationError):
            adapter.footprint([], [self.source], time.monotonic() + 5)
        alias = self.root / "alias.rs"
        alias.symlink_to(self.source)
        with self.assertRaises(adapter.ObservationError):
            adapter.read_regular(alias)
        with self.assertRaises(adapter.ObservationError):
            adapter.relative_file(self.root, "../outside")

    def test_controlled_environment_does_not_accept_inherited_selection(self) -> None:
        for name in ("RUSTC", "RUSTC_BOOTSTRAP", "CARGO_PROFILE_TEST_OPT_LEVEL", "CARGO_TARGET_X_RUNNER",
                     "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_JOBS", "FE2O3_EXAMPLE_CARGO_ARGS",
                     "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "DYLD_LIBRARY_PATH"):
            with patch.dict(os.environ, {name: "hostile"}, clear=True), self.assertRaises(adapter.ObservationError):
                adapter.environment()
        with patch.dict(os.environ, {"PATH": "/usr/bin", "FE2O3_TUTORIAL_CPU_OUTPUT_ROOT": str(self.root)}, clear=True):
            env = adapter.environment()
            self.assertNotIn("FE2O3_TUTORIAL_CPU_OUTPUT_ROOT", env)
            self.assertEqual(env["CARGO_NET_OFFLINE"], "true")
            self.assertEqual(env["CARGO_BUILD_JOBS"], "1")

    def test_component_child_status_cap_deadline_and_cleanup(self) -> None:
        env = {"PATH": os.environ.get("PATH", "")}
        phases: list[dict] = []
        output = adapter.child([sys.executable, "-c", "print('component'); raise SystemExit(7)"],
                               self.root, env, self.root, "exit", time.monotonic() + 5, phases)
        self.assertEqual(output, b"component\n")
        self.assertEqual(phases[-1]["returncode"], 7)
        with patch.object(adapter, "MAX_STREAM", 3), self.assertRaises(adapter.ObservationError):
            adapter.child([sys.executable, "-c", "print('too much')"], self.root, env,
                          self.root, "cap", time.monotonic() + 5, phases)
        with self.assertRaises(adapter.ObservationError):
            adapter.child([sys.executable, "-c", "import time; time.sleep(10)"], self.root, env,
                          self.root, "deadline", time.monotonic() + 0.1, phases)
        self.assertIsNotNone(phases[-1]["returncode"])
        with self.assertRaises(FileExistsError):
            adapter.child([sys.executable, "-c", "print('no overwrite')"], self.root, env,
                          self.root, "exit", time.monotonic() + 5, phases)

    def run_orchestration_double(self, mutate_source: bool = False, interval: str | None = None,
                                 config_change: tuple[str, str] | None = None,
                                 host: str = "x86_64-unknown-linux-gnu", feature_options=(),
                                 resolved_features=()) -> tuple[dict, list[list[str]]]:
        # Every tool response here is a protocol double, never compiler evidence.
        def write(relative: str, value: str = "fixture") -> Path:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
            return path

        app_manifest = write("examples/fixture/Cargo.toml",
                             '[package]\nname="fixture"\nversion="0.1.0"\n'
                             '[features]\nfeature-a=["helper"]\nhelper=[]\n')
        app_source = write("examples/fixture/src/lib.rs", "#[test] fn component() {}")
        driver_manifest = write("crates/cargo-fe2o3/Cargo.toml")
        driver_source = write("crates/cargo-fe2o3/src/main.rs")
        for relative in ("config/tutorial-kernel-manifest-v1.json", "scripts/validate-tutorial-kernel-manifest.py",
                         adapter.WRAPPER, "scripts/tutorial_cpu_reference.py", "Cargo.lock"):
            write(relative)
        write("rust-toolchain.toml", '[toolchain]\nchannel="nightly-2026-04-03"\n')
        (self.root / ".cargo").mkdir(exist_ok=True)
        cargo_home = self.root / "cargo-home"
        cargo_home.mkdir(exist_ok=True)
        for base in (self.root / ".cargo", cargo_home):
            for name in ("config", "config.toml"):
                (base / name).unlink(missing_ok=True)
        config_path = None
        if config_change is not None:
            location, change = config_change
            config_path = (self.root / ".cargo" if location == "ancestor" else cargo_home) / "config.toml"
            if change == "remove":
                config_path.write_text("# original configuration\n")
            elif change == "precedence":
                config_path.write_text("# original config.toml\n")
                config_path = config_path.with_name("config")
        toolpaths = {name: write("tools/bin/" + name) for name in ("cargo", "rustc", "rustup")}
        write("tools/lib/librustc_driver-fixture.so", "pinned runtime")
        output = Path(tempfile.mkdtemp(prefix="observation-", dir=self.root))
        arguments = ["examples/fixture/Cargo.toml", "lib", *feature_options]
        declaration = source_manifest(arguments)
        tutorial_manifest = write("config/tutorial-kernel-manifest-v1.json", json.dumps(declaration))
        plan = adapter.select_suite(declaration, arguments)
        plan["manifestSha256"] = adapter.digest(tutorial_manifest.read_bytes())
        validator_path = self.root / "scripts/validate-tutorial-kernel-manifest.py"
        plan["validatorSha256"] = adapter.digest(validator_path.read_bytes())
        contracts = {path: adapter.digest(path.read_bytes()) for path in
                     (app_manifest, app_source, self.root / "Cargo.lock")}
        observation = {**adapter.NO_AUTHORITY, "phases": [], "errors": []}
        target = {"name": "fixture", "kind": ["lib"], "crate_types": ["lib"],
                  "src_path": str(app_source), "test": True}
        metadata = {"packages": [{"id": "app", "manifest_path": str(app_manifest), "source": None,
                                  "targets": [target]}],
                    "resolve": {"nodes": [{"id": "app", "features": list(resolved_features)}]}}
        driver_metadata = {"packages": [{"id": "driver", "manifest_path": str(driver_manifest), "source": None}]}
        commands: list[list[str]] = []

        def child(argv, cwd, env, directory, label, deadline, phases):
            commands.append(argv)
            phases.append({"label": label, "argv": argv, "returncode": 0})
            self.assertEqual(env["CARGO_BUILD_JOBS"], "2" if label == "driver-build" else "1")
            if label.startswith("locate-"):
                return (str(toolpaths[label[7:]]) + "\n").encode()
            if label == "rustc-version":
                return f"rustc fixture\nhost: {host}\n".encode()
            if label in {"driver-metadata", "suite-metadata"}:
                self.assertEqual(argv[argv.index("--filter-platform") + 1], host)
                if label == "suite-metadata" and interval not in {None, "baseline"}:
                    changed = {"source": app_source, "manifest": app_manifest, "lock": self.root / "Cargo.lock",
                               "tutorial": tutorial_manifest, "validator": validator_path,
                               "cargo": toolpaths["cargo"], "rustc": toolpaths["rustc"],
                               "config": cargo_home / "config"}[interval]
                    with changed.open("a") as stream:
                        stream.write("\n# metadata interval mutation\n")
                return json.dumps(driver_metadata if label == "driver-metadata" else metadata).encode()
            if label == "driver-build":
                self.assertEqual(env["LD_LIBRARY_PATH"], str(toolpaths["rustc"].parent.parent / "lib"))
                self.assertEqual(argv, [str(toolpaths["cargo"]), "build", "--locked", "--offline",
                                        "-p", "cargo-fe2o3", "--bin", "cargo-fe2o3", "--target", host,
                                        "--message-format=json"])
                for name in ("RUSTC", "CARGO_BUILD_RUSTC"):
                    self.assertEqual(env[name], str(toolpaths["rustc"]))
                for name in ("RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER",
                             "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"):
                    self.assertEqual(env[name], "")
                self.assertEqual(plan["driverBuildJobs"], 2)
                self.assertEqual(env["CARGO_INCREMENTAL"], "0")
                self.assertEqual(env["CARGO_TARGET_DIR"], str(output / "build"))
                binary = output / "build" / host / "debug/cargo-fe2o3"
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"unexecuted driver double")
                binary.chmod(0o700)
                self.assertEqual(env["FE2O3_HIP_SYS_DISABLE"], "1")
                return (json.dumps({"reason": "compiler-artifact", "package_id": "driver",
                                    "target": {"name": "cargo-fe2o3", "kind": ["bin"], "crate_types": ["bin"],
                                               "src_path": str(driver_source)}, "profile": {"test": False, "opt_level": "0"},
                                    "executable": str(binary)}) + "\n" +
                        json.dumps({"reason": "build-finished", "success": True}) + "\n").encode()
            self.assertEqual(label, "suite")
            self.assertEqual(argv[argv.index("--manifest-path") + 1], str(app_manifest))
            self.assertIn("--lib", argv)
            self.assertNotIn("--all-targets", argv)
            self.assertNotIn("RUSTC", env)
            self.assertNotIn("FE2O3_HIP_SYS_DISABLE", env)
            executable = output / "build/test"
            executable.write_bytes(b"unexecuted test double")
            executable.chmod(0o700)
            events = self.events(["failed"] if mutate_source else ["ok"])
            events[0].update(package_id="app", target=target, executable=str(executable),
                             features=list(resolved_features))
            if mutate_source:
                app_source.write_text("changed after fake execution")
                phases[-1]["returncode"] = 101
            if config_change is not None:
                assert config_path is not None
                if config_change[1] == "remove":
                    config_path.unlink()
                else:
                    config_path.write_text("# appeared while suite ran\n")
            return b"\n".join(json.dumps(event).encode() for event in events) + b"\n"

        validations = []

        def validate_manifest(root, manifest):
            # A contract-checking double tests the adapter's temporal binding,
            # not the real validator or a compiler result.
            self.assertEqual(manifest, declaration)
            validations.append(True)
            for path, expected in contracts.items():
                if adapter.digest(path.read_bytes()) != expected:
                    raise SystemExit("fixture source/manifest/lock contract is stale")
            if interval == "baseline":
                app_source.write_text("changed during contract validation")

        validator = SimpleNamespace(effective_cargo_lock=lambda root, package, label: root / "Cargo.lock",
                                    validate_manifest=validate_manifest)
        with patch.dict(os.environ, {"HOME": str(self.root), "PATH": "/usr/bin", "CARGO_HOME": str(cargo_home)}, clear=True), \
                patch.object(adapter, "load_plan", return_value=(plan, validator)), \
                patch.object(adapter.shutil, "which", return_value=str(toolpaths["rustup"])), \
                patch.object(adapter, "child", side_effect=child):
            if mutate_source or interval is not None or config_change is not None:
                expected_error = ("input drift" if mutate_source else
                                  "configuration input drift" if config_change is not None else
                                  {"source": "bound source contract refused", "manifest": "bound source contract refused",
                                   "lock": "bound source contract refused", "tutorial": "original tutorial manifest changed",
                                   "validator": "source validator changed", "cargo": "tool changed before baseline",
                                   "rustc": "tool changed before baseline", "config": "configuration drift during metadata",
                                   "baseline": "input drift during baseline validation"}[interval])
                with self.assertRaisesRegex(adapter.ObservationError, expected_error):
                    adapter.execute(self.root, arguments, output, time.monotonic(), observation)
            else:
                adapter.execute(self.root, arguments, output, time.monotonic(), observation)
                self.assertEqual(len(validations), 2)
        if interval is not None:
            self.assertNotIn("driver-build", [phase["label"] for phase in observation["phases"]])
        return observation, commands

    def test_orchestration_double_binds_absolute_selected_source_and_keeps_authority_false(self) -> None:
        observation, commands = self.run_orchestration_double(False)
        self.assertEqual(observation["outcome"], "passed")
        self.assertTrue(observation["inputsUnchanged"])
        self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))
        self.assertEqual(len(commands), 7)

    def test_single_suite_features_bind_metadata_and_test_without_bootstrap_leakage(self) -> None:
        options = ["--no-default-features", "--features", "feature-a"]
        observation, _ = self.run_orchestration_double(
            feature_options=options, resolved_features=["feature-a", "helper"])
        self.assertEqual(observation["outcome"], "passed")
        self.assertEqual(observation["plan"]["hostConfiguration"]["features"], ["feature-a"])
        self.assertEqual(observation["plan"]["selectedTarget"]["features"], ["feature-a", "helper"])
        for phase in observation["phases"]:
            argv = phase["argv"]
            if phase["label"] in {"suite-metadata", "suite"}:
                self.assertEqual(argv.count("--no-default-features"), 1)
                self.assertEqual(argv.count("--features"), 1)
                self.assertEqual(argv[argv.index("--features") + 1], "feature-a")
            else:
                self.assertNotIn("--no-default-features", argv)
                self.assertNotIn("--features", argv)
        self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))

    def test_bootstrap_target_is_exact_discovered_host_with_nested_driver_artifact(self) -> None:
        for host in ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"):
            with self.subTest(host=host):
                observation, commands = self.run_orchestration_double(host=host)
                bootstrap = [command for command in commands if "build" in command]
                self.assertEqual(len(bootstrap), 1)
                self.assertEqual(bootstrap[0].count("--target"), 1)
                self.assertEqual(bootstrap[0][bootstrap[0].index("--target") + 1], host)
                managed = commands[-1]
                self.assertEqual(managed[1], "test")
                self.assertIn("--lib", managed)
                self.assertNotIn("--target", managed)
                self.assertEqual(observation["plan"]["host"], host)
                self.assertEqual(observation["plan"]["declaredCommand"]["timeoutSeconds"], 1200)
                self.assertEqual(observation["outcome"], "passed")
                self.assertTrue(observation["inputsUnchanged"])
                self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))

    def test_postflight_drift_does_not_erase_failed_test_or_process_history(self) -> None:
        observation, _ = self.run_orchestration_double(True)
        self.assertFalse(observation["inputsUnchanged"])
        self.assertEqual(observation["tests"]["outcome"], "failed")
        self.assertEqual(observation["phases"][-1]["returncode"], 101)

    def test_metadata_cannot_refresh_source_contract_or_tool_baselines(self) -> None:
        for interval in ("source", "manifest", "lock", "tutorial", "validator", "cargo", "rustc", "config", "baseline"):
            with self.subTest(interval=interval):
                observation, _ = self.run_orchestration_double(interval=interval)
                self.assertNotEqual(observation.get("outcome"), "passed")

    def test_config_presence_and_precedence_are_rescanned_outside_local_roots(self) -> None:
        for location in ("ancestor", "cargo-home"):
            for change in ("create", "remove", "precedence"):
                with self.subTest(location=location, change=change):
                    observation, _ = self.run_orchestration_double(config_change=(location, change))
                    self.assertFalse(observation["configurationInputsUnchanged"])
                    self.assertEqual(observation["tests"]["outcome"], "passed")
                    self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))

    def test_main_signal_and_deadline_cleanup_are_nonpassing_component_records(self) -> None:
        source_root = self.root / "source-checkout"
        source_root.mkdir(mode=0o700)
        for mode in ("terminate", "deadline"):
            def execute(root, arguments, output, started, observation):
                self.assertEqual(root, source_root)
                self.assertFalse(output.is_relative_to(root))
                code = ("import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(5)"
                        if mode == "terminate" else "import time; time.sleep(5)")
                adapter.child([sys.executable, "-c", code], self.root, {}, output, "component-signal",
                              time.monotonic() + 10, observation["phases"])
            captured = io.StringIO()
            with patch.dict(os.environ, {"FE2O3_TUTORIAL_CPU_OUTPUT_ROOT": str(self.root)}, clear=True), \
                    patch.object(adapter, "__file__", str(source_root / "scripts" / HELPER.name)), \
                    patch.object(adapter, "execute", side_effect=execute), \
                    patch.object(adapter, "MAX_SECONDS", 0.1 if mode == "deadline" else 5), \
                    contextlib.redirect_stdout(captured):
                result = adapter.main(["fixture", "lib"])
            observation = json.loads(captured.getvalue())
            self.assertNotEqual(result, 0)
            self.assertNotEqual(observation["outcome"], "passed")
            self.assertTrue(observation["phases"], observation)
            self.assertIsNotNone(observation["phases"][-1]["returncode"])
            self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))

    def test_main_rejects_output_inside_its_source_before_either_executor(self) -> None:
        source_root = self.root / "source-checkout"
        source_root.mkdir(mode=0o700)
        output_parent = source_root / "private-output"
        output_parent.mkdir(mode=0o700)
        previous = {number: adapter.signal.getsignal(number) for number in
                    (adapter.signal.SIGINT, adapter.signal.SIGTERM, adapter.signal.SIGALRM)}
        for arguments in (["fixture", "lib"], ["--batch"]):
            captured = io.StringIO()
            with patch.dict(os.environ, {"FE2O3_TUTORIAL_CPU_OUTPUT_ROOT": str(output_parent)}, clear=True), \
                    patch.object(adapter, "__file__", str(source_root / "scripts" / HELPER.name)), \
                    patch.object(adapter, "execute") as standalone, \
                    patch.object(adapter, "execute_batch") as batch, \
                    patch.object(adapter.tempfile, "mkdtemp") as allocate, \
                    contextlib.redirect_stdout(captured):
                result = adapter.main(arguments)
            observation = json.loads(captured.getvalue())
            self.assertEqual(result, 1)
            self.assertEqual(observation["outcome"], "invalid")
            self.assertEqual(observation["errors"], ["output directory must be outside the source checkout"])
            self.assertEqual(observation["phases"], [])
            standalone.assert_not_called()
            batch.assert_not_called()
            allocate.assert_not_called()
            self.assertEqual(list(output_parent.iterdir()), [])
            self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))
            self.assertEqual(adapter.signal.getitimer(adapter.signal.ITIMER_REAL), (0.0, 0.0))
            self.assertEqual({number: adapter.signal.getsignal(number) for number in previous}, previous)


class BatchComponents(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cpu-batch-components-")
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "source"
        self.root.mkdir()
        self.clock = 100.0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, relative: str, payload: str = "component input") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return path

    def prepare(self, count: int = 3, feature_options=None, resolved_features=None) -> None:
        self.apps = []
        self.feature_options = [(feature_options or {}).get(index, []) for index in range(count)]
        self.declaration = {"qualification": {"suites": []}, "compilerFixtures": []}
        for index in range(count):
            package = f"arbitrary_package_{index}"
            manifest = self.write(f"examples/{package}/Cargo.toml",
                                  f'[package]\nname="{package}"\nversion="0.1.0"\n'
                                  '[features]\nfeature-a=["helper"]\nhelper=[]\n')
            kind = "lib" if index % 2 == 0 else "test"
            source = self.write(f"examples/{package}/" + ("src/lib.rs" if kind == "lib" else "tests/reference.rs"))
            arguments = [str(manifest.relative_to(self.root)), kind, *([] if kind == "lib" else ["reference"])]
            arguments.extend(self.feature_options[index])
            item = source_manifest(arguments)
            item["qualification"]["suites"][0]["suiteId"] = f"cpu-{index}"
            for fixture in item["compilerFixtures"]:
                fixture["fixtureId"] += f"-{index}"
            item["qualification"]["suites"][0]["coverage"][0]["fixtureIds"] = [
                fixture["fixtureId"] for fixture in item["compilerFixtures"]]
            self.declaration["qualification"]["suites"].extend(item["qualification"]["suites"])
            self.declaration["compilerFixtures"].extend(item["compilerFixtures"])
            target = {"name": package if kind == "lib" else "reference", "kind": [kind], "crate_types": [kind],
                      "src_path": str(source), "test": True}
            metadata = {"packages": [{"id": package, "manifest_path": str(manifest), "source": None,
                                      "targets": [target]}],
                        "resolve": {"nodes": [{"id": package, "features": (resolved_features or {}).get(index, [])}]}}
            self.apps.append((manifest, source, target, metadata))
        self.tutorial = self.write("config/tutorial-kernel-manifest-v1.json", json.dumps(self.declaration))
        self.validator_path = self.write("scripts/validate-tutorial-kernel-manifest.py")
        self.write(adapter.WRAPPER)
        self.write("scripts/tutorial_cpu_reference.py")
        self.write("Cargo.toml")
        self.lock = self.write("Cargo.lock")
        self.write("rust-toolchain.toml", '[toolchain]\nchannel="nightly-2026-04-03"\n')
        self.driver_manifest = self.write("crates/cargo-fe2o3/Cargo.toml")
        self.driver_source = self.write("crates/cargo-fe2o3/src/main.rs")
        self.tools = {name: self.write("tools/bin/" + name) for name in ("cargo", "rustc", "rustup")}
        self.runtime = self.write("tools/lib/librustc_driver-fixture.so", "pinned runtime")
        self.cargo_home = self.base / "cargo-home"
        self.cargo_home.mkdir()
        self.output = self.base / "evidence"
        self.output.mkdir(mode=0o700)
        self.contracts = {path: adapter.digest(path.read_bytes()) for path in
                          [self.lock, *[path for app in self.apps for path in app[:2]]]}
        self.calls = []
        self.alarms = []

    def plans(self) -> list[dict]:
        plans = adapter.batch_plans(self.declaration)
        for plan in plans:
            plan.update(manifestSha256=adapter.digest(self.tutorial.read_bytes()),
                        validatorSha256=adapter.digest(self.validator_path.read_bytes()))
        return plans

    def run_batch(self, states: dict[int, list[str]] | None = None, mutate=None,
                  persist=None, costs: dict[str, float] | None = None,
                  cleanup=None, candidate=None, setup_delay: float = 0.0) -> dict:
        states, costs = states or {}, costs or {}
        plans = self.plans()
        original_rmtree = adapter.shutil.rmtree
        original_save = adapter.save_observation
        original_candidates = adapter.configuration_candidates
        original_mkdir = Path.mkdir
        started = self.clock

        def validate(root, manifest):
            self.assertEqual(manifest, self.declaration)
            for path, expected in self.contracts.items():
                if adapter.digest(path.read_bytes()) != expected:
                    raise SystemExit("component declared source/lock contract changed")

        validator = SimpleNamespace(validate_manifest=validate,
                                    effective_cargo_lock=lambda *args: self.lock)

        def child(argv, cwd, env, directory, label, deadline, phases):
            self.calls.append((label, list(argv), dict(env), deadline, directory))
            self.assertEqual(env["CARGO_BUILD_JOBS"], "2" if label == "driver-build" else "1")
            self.assertLess(self.clock, deadline)
            phase = {"label": label, "argv": argv, "returncode": 0, "completeLogs": True, "logs": []}
            phases.append(phase)
            index = int(directory.name[-2:]) if directory.name.startswith("suite-") else None
            self.clock += costs.get(label, 1.0)
            if label.startswith("locate-"):
                data = (str(self.tools[label[7:]]) + "\n").encode()
            elif label == "rustc-version":
                data = b"rustc component\nhost: x86_64-unknown-linux-gnu\n"
            elif label == "driver-metadata":
                data = json.dumps({"packages": [{"id": "driver", "manifest_path": str(self.driver_manifest),
                                                  "source": None}]}).encode()
            elif label == "suite-metadata":
                data = json.dumps(self.apps[index][3]).encode()
                self.assertEqual(argv[argv.index("--manifest-path") + 1], str(self.apps[index][0]))
                self.assertEqual(argv, [str(self.tools["cargo"]), "metadata", "--locked", "--offline",
                                       "--format-version", "1", "--manifest-path", str(self.apps[index][0]),
                                       "--filter-platform", "x86_64-unknown-linux-gnu", *self.feature_options[index]])
            elif label == "driver-build":
                self.assertEqual(argv, [str(self.tools["cargo"]), "build", "--locked", "--offline", "-p",
                                       "cargo-fe2o3", "--bin", "cargo-fe2o3", "--target",
                                       "x86_64-unknown-linux-gnu", "--message-format=json"])
                self.assertEqual(env["RUSTC"], str(self.tools["rustc"]))
                self.assertEqual(env["RUSTC_WORKSPACE_WRAPPER"], "")
                binary = Path(env["CARGO_TARGET_DIR"]) / "x86_64-unknown-linux-gnu/debug/cargo-fe2o3"
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"unexecuted driver component")
                binary.chmod(0o700)
                data = (json.dumps({"reason": "compiler-artifact", "package_id": "driver",
                                   "target": {"name": "cargo-fe2o3", "kind": ["bin"], "crate_types": ["bin"],
                                              "src_path": str(self.driver_source)},
                                   "profile": {"test": False, "opt_level": "0"}, "executable": str(binary)}) + "\n" +
                        json.dumps({"reason": "build-finished", "success": True}) + "\n").encode()
            else:
                self.assertEqual(label, "suite")
                manifest, _, target, metadata = self.apps[index]
                selector = ["--lib"] if target["kind"] == ["lib"] else ["--test", "reference"]
                self.assertEqual(argv, [str(self.output / "common/driver/cargo-fe2o3"), "test", "--locked", "--offline",
                                       "--manifest-path", str(manifest), *self.feature_options[index],
                                       *selector, "--message-format=json", "--",
                                       "-Z", "unstable-options", "--format=json", "--test-threads=1"])
                self.assertNotIn("RUSTC", env)
                self.assertNotIn("FE2O3_HIP_SYS_DISABLE", env)
                self.assertEqual(env["CARGO_BUILD_JOBS"], "1")
                self.assertEqual(env["CARGO_INCREMENTAL"], "0")
                executable = Path(env["CARGO_TARGET_DIR"]) / "test-component"
                executable.write_bytes(b"unexecuted test component")
                executable.chmod(0o700)
                outcomes = states.get(index, ["ok"])
                events = [{"reason": "compiler-artifact", "package_id": metadata["packages"][0]["id"],
                           "target": target, "features": metadata["resolve"]["nodes"][0]["features"],
                           "profile": {"test": True}, "executable": str(executable)},
                          {"reason": "build-finished", "success": True},
                          {"type": "suite", "event": "started", "test_count": len(outcomes)}]
                for ordinal, result in enumerate(outcomes):
                    events.extend([{"type": "test", "event": "started", "name": f"case-{ordinal}"},
                                   {"type": "test", "event": result, "name": f"case-{ordinal}"}])
                events.append({"type": "suite", "event": "failed" if "failed" in outcomes else "ok",
                               "passed": outcomes.count("ok"), "failed": outcomes.count("failed"),
                               "ignored": outcomes.count("ignored"), "measured": 0, "filtered_out": 0})
                phase["returncode"] = 101 if "failed" in outcomes else 0
                data = b"\n".join(json.dumps(event).encode() for event in events) + b"\n"
            if mutate is not None:
                changed = mutate(label, index, data, phase)
                if changed is not None:
                    data = changed
            return data

        def save(directory, record):
            result = original_save(directory, record)
            if persist is not None:
                persist(directory, record)
            return result

        def remove(path):
            if cleanup is not None:
                cleanup(path)
            return original_rmtree(path)

        def candidates(root, env):
            result = original_candidates(root, env)
            return candidate(result) if candidate is not None else result

        def mkdir(path, *args, **kwargs):
            if path.name.startswith("suite-"):
                self.clock += setup_delay
            return original_mkdir(path, *args, **kwargs)

        observation = {"schema": "fe2o3-tutorial-cpu-reference-batch-observation-v1", **adapter.NO_AUTHORITY,
                       "outcome": "invalid", "phases": [], "errors": []}
        with patch.dict(os.environ, {"HOME": str(self.base), "PATH": "/usr/bin", "CARGO_HOME": str(self.cargo_home)}, clear=True), \
                patch.object(adapter, "load_batch_plans", return_value=(plans, validator)), \
                patch.object(adapter.shutil, "which", return_value=str(self.tools["rustup"])), \
                patch.object(adapter, "child", side_effect=child), \
                patch.object(adapter.time, "monotonic", side_effect=lambda: self.clock), \
                patch.object(adapter.signal, "setitimer", side_effect=lambda kind, delay: self.alarms.append(delay)), \
                patch.object(adapter, "save_observation", side_effect=save), \
                patch.object(adapter.shutil, "rmtree", side_effect=remove), \
                patch.object(Path, "mkdir", new=mkdir), \
                patch.object(adapter, "configuration_candidates", side_effect=candidates):
            adapter.execute_batch(self.root, self.output, started, observation)
        return observation

    def records(self, observation) -> list[dict]:
        result = []
        for item in observation["suites"]:
            payload = Path(item["observation"]["path"]).read_bytes()
            self.assertEqual(adapter.digest(payload), item["observation"]["sha256"])
            result.append(json.loads(payload))
        return result

    def test_pinned_runtime_reaches_cargo_build_scripts_and_suites(self) -> None:
        self.prepare(1)
        observation = self.run_batch()
        self.assertEqual(observation["outcome"], "passed")
        for label, _, env, *_ in self.calls:
            if label in {"driver-metadata", "driver-build", "suite-metadata", "suite"}:
                self.assertEqual(env.get("LD_LIBRARY_PATH"), str(self.runtime.parent), label)
        common = json.loads(Path(observation["common"]["path"]).read_bytes())
        self.assertEqual(common["plan"]["tools"]["runtime:" + self.runtime.name],
                         {"path": str(self.runtime), "sha256": adapter.digest(self.runtime.read_bytes())})

    def test_pinned_runtime_mutation_or_roster_drift_refuses_before_reuse(self) -> None:
        for phase in ("driver-metadata", "driver-build", "suite"):
            for change in ("bytes", "add", "remove", "symlink"):
                with self.subTest(phase=phase, change=change):
                    self.temporary.cleanup()
                    self.setUp()
                    self.prepare(2)
                    def mutate(label, index, data, record):
                        if label != phase or (label == "suite" and index != 0):
                            return
                        if change == "bytes":
                            self.runtime.write_text("changed runtime")
                        elif change == "add":
                            self.runtime.with_name("libextra.so").write_text("extra")
                        else:
                            self.runtime.unlink()
                            if change == "symlink":
                                self.runtime.symlink_to(self.tools["rustc"])
                    observation = self.run_batch(mutate=mutate)
                    self.assertNotEqual(observation["outcome"], "passed")
                    self.assertFalse(self.records(observation)[1]["attempted"])
                    if phase == "driver-metadata":
                        self.assertNotIn("driver-build", [call[0] for call in self.calls])

    def test_pinned_runtime_missing_or_escaping_directory_refuses_discovery(self) -> None:
        for change in ("missing", "symlink", "foreign-cargo"):
            with self.subTest(change=change):
                self.temporary.cleanup()
                self.setUp()
                self.prepare(1)
                if change == "foreign-cargo":
                    self.tools["cargo"] = self.write("foreign/bin/cargo")
                else:
                    directory = self.runtime.parent
                    self.runtime.unlink()
                    directory.rmdir()
                    if change == "symlink":
                        other = self.root / "foreign-lib"
                        other.mkdir()
                        directory.symlink_to(other, target_is_directory=True)
                observation = self.run_batch()
                self.assertNotEqual(observation["outcome"], "passed")
                self.assertNotIn("driver-build", [call[0] for call in self.calls])

    def test_generic_twelve_suite_batch_builds_one_driver_and_fresh_suite_dirs(self) -> None:
        self.prepare(12)
        observation = self.run_batch()
        self.assertEqual(observation["outcome"], "passed")
        self.assertEqual(len(self.calls), 29)
        self.assertEqual([call[0] for call in self.calls].count("driver-build"), 1)
        self.assertEqual([call[0] for call in self.calls].count("suite-metadata"), 12)
        common = json.loads(Path(observation["common"]["path"]).read_bytes())
        self.assertEqual(common["plan"]["driverBuildJobs"], 2)
        suites = [call for call in self.calls if call[0] == "suite"]
        self.assertEqual(len({call[2]["CARGO_TARGET_DIR"] for call in suites}), 12)
        self.assertTrue(all(call[2]["CARGO_TARGET_DIR"] != str(self.output / "common/build") for call in suites))
        records = self.records(observation)
        for record in records:
            self.assertEqual(record["commonElapsedSeconds"], 5.0)
            self.assertEqual(record["ownElapsedSeconds"], 2.0)
            self.assertEqual(record["chargedElapsedSeconds"], 7.0)
            self.assertEqual(record["executionMode"], "batch-shared-private-driver")
            self.assertEqual(record["commonObservation"], observation["common"])
            self.assertEqual(record["tests"]["executable"]["digestScope"], "post-execution-on-disk-artifact-only")
            self.assertTrue(all(record[key] is False for key in adapter.NO_AUTHORITY))
            self.assertFalse((Path(record["evidenceDirectory"]) / "build").exists())
        self.assertEqual(len({call[3] - (105 + 2 * index) for index, call in enumerate(suites)}), 1)
        self.assertFalse((self.output / "common/build").exists())
        self.assertFalse((self.output / "common/driver").exists())
        self.assertTrue(observation["finalSharedInputsUnchanged"])
        self.assertEqual(self.alarms[-1], 0)

    def test_batch_feature_selections_do_not_leak_between_suites_or_bootstrap(self) -> None:
        options = {1: ["--features", "feature-a"],
                   2: ["--no-default-features", "--features", "feature-a"]}
        resolved = {1: ["feature-a", "helper"], 2: ["feature-a", "helper"]}
        self.prepare(4, options, resolved)
        observation = self.run_batch()
        self.assertEqual(observation["outcome"], "passed")
        for index, record in enumerate(self.records(observation)):
            self.assertEqual(record["plan"]["hostConfiguration"]["features"], ["feature-a"] if index in options else [])
            self.assertEqual(record["plan"]["hostConfiguration"]["defaultFeatures"], index != 2)
            self.assertEqual(record["plan"]["selectedTarget"]["features"], resolved.get(index, []))
            self.assertTrue(all(record[key] is False for key in adapter.NO_AUTHORITY))
        for label, argv, *_ in self.calls:
            if label in {"driver-metadata", "driver-build"}:
                self.assertNotIn("--features", argv)
                self.assertNotIn("--no-default-features", argv)

    def test_original_fourteen_unique_suites_retain_their_batch_contract(self) -> None:
        self.prepare(14)
        observation = self.run_batch()
        self.assertEqual(observation["outcome"], "passed")
        self.assertEqual(len(self.records(observation)), 14)
        self.assertEqual(observation["schedulerLimitSeconds"], 16800)
        self.assertEqual(len(self.calls), 33)
        self.assertEqual([call[0] for call in self.calls].count("driver-build"), 1)
        suites = [call for call in self.calls if call[0] == "suite"]
        self.assertEqual(len({call[2]["CARGO_TARGET_DIR"] for call in suites}), 14)

    def test_thirty_six_unique_suites_are_the_exact_batch_limit(self) -> None:
        self.prepare(36)
        self.assertEqual(adapter.MAX_BATCH_SUITES, 36)
        observation = self.run_batch()
        self.assertEqual(observation["outcome"], "passed")
        self.assertEqual(observation["declaredSuiteCount"], 36)
        self.assertEqual(observation["schedulerLimitSeconds"], 43200)
        self.assertEqual(len(self.calls), 77)
        self.assertEqual([call[0] for call in self.calls].count("driver-build"), 1)
        suites = [call for call in self.calls if call[0] == "suite"]
        self.assertEqual(len({call[2]["CARGO_TARGET_DIR"] for call in suites}), 36)
        records = self.records(observation)
        self.assertEqual([record["plan"]["suiteId"] for record in records],
                         [f"cpu-{index}" for index in range(36)])
        for record in records:
            self.assertEqual(record["plan"]["declaredCommand"]["timeoutSeconds"], 1200)
            self.assertEqual(record["commonElapsedSeconds"], 5.0)
            self.assertEqual(record["ownElapsedSeconds"], 2.0)
            self.assertEqual(record["chargedElapsedSeconds"], 7.0)
            self.assertTrue(all(record[key] is False for key in adapter.NO_AUTHORITY))
        self.assertTrue(observation["finalSharedInputsUnchanged"])

    def test_thirty_seven_unique_suites_refuse_the_limit_before_execution(self) -> None:
        self.prepare(37)
        with self.assertRaisesRegex(adapter.ObservationError, "between 1 and 36"):
            adapter.batch_plans(self.declaration)
        self.assertEqual(self.calls, [])

    def test_empty_duplicate_overcap_and_foreign_batch_plans_refuse(self) -> None:
        self.prepare(1)
        for mutation in ("empty", "duplicate-id", "duplicate-command", "overcap", "foreign", "environment"):
            manifest = copy.deepcopy(self.declaration)
            suite = manifest["qualification"]["suites"][0]
            if mutation == "empty":
                manifest["qualification"]["suites"] = []
            elif mutation == "overcap":
                manifest["qualification"]["suites"] *= 37
            elif mutation.startswith("duplicate"):
                extra = copy.deepcopy(suite)
                if mutation == "duplicate-command":
                    extra["suiteId"] = "different"
                manifest["qualification"]["suites"].append(extra)
            elif mutation == "foreign":
                suite["command"]["executable"] = "cargo"
            else:
                suite["command"]["environment"] = ["RUSTFLAGS=override"]
            with self.subTest(mutation=mutation), self.assertRaises(adapter.ObservationError):
                adapter.batch_plans(manifest)

    def test_suite_directory_setup_time_is_part_of_its_own_budget(self) -> None:
        self.prepare(2)
        observation = self.run_batch(setup_delay=3.0)
        self.assertEqual(observation["outcome"], "passed")
        self.assertEqual([record["ownElapsedSeconds"] for record in self.records(observation)], [5.0, 5.0])
        self.assertEqual([record["chargedElapsedSeconds"] for record in self.records(observation)], [10.0, 10.0])

    def test_failed_ignored_and_empty_suites_are_retained_not_aggregate_pass(self) -> None:
        self.prepare(3)
        observation = self.run_batch(states={0: ["failed"], 1: ["ignored"], 2: []})
        self.assertEqual(observation["outcome"], "failed")
        self.assertEqual([record["outcome"] for record in self.records(observation)], ["failed", "partial", "empty"])
        self.assertEqual(len(self.calls), 11)

    def test_bad_suite_events_abort_reuse_and_preserve_remaining_ids(self) -> None:
        self.prepare(3)
        observation = self.run_batch(mutate=lambda label, index, data, phase: b"{}\n" if label == "suite" else None)
        records = self.records(observation)
        self.assertEqual([record["plan"]["suiteId"] for record in records], ["cpu-0", "cpu-1", "cpu-2"])
        self.assertEqual([record["outcome"] for record in records], ["invalid", "unavailable", "unavailable"])
        self.assertEqual([record["attempted"] for record in records], [True, False, False])
        self.assertEqual(len([call for call in self.calls if call[0] == "suite"]), 1)

    def test_common_budget_is_fully_charged_and_later_suites_do_not_pay_previous_runs(self) -> None:
        self.assertEqual(adapter.suite_deadline(10, 7, 12, 100), 15)
        self.assertEqual(adapter.suite_deadline(40, 7, 12, 100), 45)
        self.assertEqual(adapter.suite_deadline(40, 7, 12, 43), 43)
        for common in (12, 13):
            with self.assertRaisesRegex(adapter.ObservationError, "common setup exhausted"):
                adapter.suite_deadline(10, common, 12, 100)
        self.prepare(2)
        observation = self.run_batch(costs={"driver-build": 1196})
        self.assertNotEqual(observation["outcome"], "passed")
        self.assertTrue(all(not record["attempted"] for record in self.records(observation)))
        self.assertNotIn("suite-metadata", [call[0] for call in self.calls])

    def test_suite_exact_deadline_is_not_late_success(self) -> None:
        self.prepare(2)
        observation = self.run_batch(costs={"suite": 1194})
        records = self.records(observation)
        self.assertEqual(records[0]["chargedElapsedSeconds"], 1200)
        self.assertNotEqual(records[0]["outcome"], "passed")
        self.assertFalse(records[1]["attempted"])

    def test_source_contract_mutation_during_metadata_does_not_become_baseline(self) -> None:
        self.prepare(2)
        def mutate(label, index, data, phase):
            if label == "suite-metadata":
                self.apps[0][1].write_text("changed during metadata")
        observation = self.run_batch(mutate=mutate)
        self.assertNotEqual(observation["outcome"], "passed")
        self.assertNotIn("suite", [call[0] for call in self.calls])
        self.assertFalse(self.records(observation)[1]["attempted"])

    def test_between_suite_driver_drift_aborts_before_next_metadata(self) -> None:
        self.prepare(3)
        def persist(directory, record):
            if directory.name == "suite-00":
                driver = self.output / "common/driver/cargo-fe2o3"
                driver.chmod(0o700)
                driver.write_bytes(b"changed copied driver")
        observation = self.run_batch(persist=persist)
        self.assertNotEqual(observation["outcome"], "passed")
        self.assertEqual([record["attempted"] for record in self.records(observation)], [True, False, False])
        self.assertEqual(len([call for call in self.calls if call[0] == "suite-metadata"]), 1)
        self.assertFalse(observation["finalSharedInputsUnchanged"])

    def test_final_configuration_creation_is_not_hidden_by_prior_success(self) -> None:
        self.prepare(1)
        def persist(directory, record):
            if directory.name == "suite-00":
                (self.cargo_home / "config").write_text("# introduced after suite postflight")
        observation = self.run_batch(persist=persist)
        self.assertEqual(self.records(observation)[0]["outcome"], "passed")
        self.assertEqual(observation["outcome"], "invalid")
        self.assertFalse(observation["finalSharedInputsUnchanged"])

    def test_cleanup_failure_poison_and_interrupt_leave_explicit_unavailable_records(self) -> None:
        self.prepare(2)
        def cleanup(path):
            if path.parent.name == "suite-00":
                raise OSError("component cleanup denied")
        observation = self.run_batch(cleanup=cleanup)
        self.assertEqual([record["outcome"] for record in self.records(observation)], ["invalid", "unavailable"])
        self.assertFalse((self.output / "common/driver").exists())

    def test_interruption_retains_phases_and_cleans_private_scratch(self) -> None:
        self.prepare(2)
        def mutate(label, index, data, phase):
            if label == "suite":
                phase["returncode"] = -9
                raise KeyboardInterrupt
        observation = self.run_batch(mutate=mutate)
        records = self.records(observation)
        self.assertEqual(records[0]["outcome"], "interrupted")
        self.assertEqual(records[0]["phases"][-1]["returncode"], -9)
        self.assertEqual(records[1]["outcome"], "unavailable")
        self.assertFalse((self.output / "common/driver").exists())

    def test_original_common_files_and_tools_cannot_be_refreshed_after_metadata(self) -> None:
        for name in ("tutorial", "validator", "lock", "toolchain", "cargo", "rustc"):
            with self.subTest(name=name):
                self.temporary.cleanup()
                self.setUp()
                self.prepare(2)
                path = {"tutorial": self.tutorial, "validator": self.validator_path, "lock": self.lock,
                        "toolchain": self.root / "rust-toolchain.toml", **self.tools}[name]
                def mutate(label, index, data, phase):
                    if label == "driver-metadata":
                        with path.open("a") as stream:
                            stream.write("\n# drift\n")
                observation = self.run_batch(mutate=mutate)
                self.assertNotIn("driver-build", [call[0] for call in self.calls])
                self.assertTrue(all(not record["attempted"] for record in self.records(observation)))
                self.assertNotEqual(observation["outcome"], "passed")

    def test_shared_bootstrap_config_presence_precedence_and_candidates_are_rechecked(self) -> None:
        for location in ("ancestor", "cargo-home"):
            for change in ("create", "remove", "precedence"):
                with self.subTest(location=location, change=change):
                    self.temporary.cleanup()
                    self.setUp()
                    self.prepare(2)
                    folder = self.root / ".cargo" if location == "ancestor" else self.cargo_home
                    folder.mkdir(exist_ok=True)
                    target = folder / "config.toml"
                    if change != "create":
                        target.write_text("# original\n")
                    if change == "precedence":
                        target = folder / "config"
                    def mutate(label, index, data, phase):
                        if label == "driver-build":
                            if change == "remove":
                                target.unlink()
                            else:
                                target.write_text("# changed\n")
                    observation = self.run_batch(mutate=mutate)
                    self.assertTrue(all(not record["attempted"] for record in self.records(observation)))
                    self.assertNotEqual(observation["outcome"], "passed")
                    self.assertFalse((self.output / "common/driver").exists())

    def test_configuration_candidate_roster_drift_refuses_before_bootstrap(self) -> None:
        self.prepare(2)
        visits = 0
        def candidate(paths):
            nonlocal visits
            visits += 1
            return paths if visits == 1 else [*paths, self.base / "new-config"]
        observation = self.run_batch(candidate=candidate)
        self.assertNotIn("driver-build", [call[0] for call in self.calls])
        self.assertNotEqual(observation["outcome"], "passed")

    def test_wrong_bootstrap_package_source_or_profile_never_enters_any_suite(self) -> None:
        for field in ("package", "source", "profile"):
            with self.subTest(field=field):
                self.temporary.cleanup()
                self.setUp()
                self.prepare(2)
                def mutate(label, index, data, phase):
                    if label != "driver-build":
                        return None
                    events = [json.loads(line) for line in data.splitlines()]
                    if field == "package":
                        events[0]["package_id"] = "foreign"
                    elif field == "source":
                        events[0]["target"]["src_path"] = "/proc/self/fd/9/main.rs"
                    else:
                        events[0]["profile"]["test"] = True
                    return b"\n".join(json.dumps(event).encode() for event in events) + b"\n"
                observation = self.run_batch(mutate=mutate)
                self.assertNotIn("suite-metadata", [call[0] for call in self.calls])
                self.assertNotEqual(observation["outcome"], "passed")

    def test_one_unit_before_deadline_passes_without_discounting_common_cost(self) -> None:
        self.prepare(2)
        observation = self.run_batch(costs={"suite": 1193})
        self.assertEqual(observation["outcome"], "passed")
        self.assertEqual([record["chargedElapsedSeconds"] for record in self.records(observation)], [1199, 1199])
        self.assertEqual(observation["schedulerLimitSeconds"], 2400)

    def test_overall_scheduler_and_child_timeout_never_create_passed_aggregate(self) -> None:
        self.prepare(2)
        def timeout(label, index, data, phase):
            if label == "suite":
                phase.update(returncode=-9, completeLogs=False)
                raise adapter.ObservationError("component child deadline", "unavailable")
        observation = self.run_batch(mutate=timeout)
        self.assertEqual([record["outcome"] for record in self.records(observation)], ["unavailable", "unavailable"])
        self.assertEqual(self.records(observation)[0]["phases"][-1]["returncode"], -9)
        self.temporary.cleanup()
        self.setUp()
        self.prepare(1)
        def persist(directory, record):
            if directory.name == "suite-00":
                self.clock = 1300
        observation = self.run_batch(persist=persist)
        self.assertEqual(observation["outcome"], "unavailable")
        self.assertIn("batch scheduler deadline exhausted", observation["errors"])

    def test_batch_loader_and_current_manifest_enumeration_remain_generic(self) -> None:
        self.prepare(2)
        self.validator_path.write_text("def validate_manifest(root, manifest):\n    assert manifest['qualification']['suites']\n")
        plans, _ = adapter.load_batch_plans(self.root)
        self.assertEqual([plan["suiteId"] for plan in plans], ["cpu-0", "cpu-1"])
        self.assertTrue(all("manifestSha256" in plan and "validatorSha256" in plan for plan in plans))
        source = Path(__file__).resolve().parents[2] / "config/tutorial-kernel-manifest-v1.json"
        manifest = adapter.decode_json(source.read_bytes())
        expected = list(CPU_SUITE_CONFIGURATIONS)
        self.assertEqual([suite["suiteId"] for suite in manifest["qualification"]["suites"]
                          if suite["gate"] == "cpu-reference"], expected)
        self.assertEqual([plan["suiteId"] for plan in adapter.batch_plans(manifest)], expected)
        self.declaration["qualification"]["suites"][0]["command"]["arguments"].extend(["--features", "missing"])
        self.tutorial.write_text(json.dumps(self.declaration))
        with self.assertRaisesRegex(adapter.ObservationError, "undeclared local host feature"):
            adapter.load_batch_plans(self.root)

    def test_main_batch_selector_is_exact_and_restores_handlers_and_timer(self) -> None:
        previous = {number: adapter.signal.getsignal(number) for number in
                    (adapter.signal.SIGINT, adapter.signal.SIGTERM, adapter.signal.SIGALRM)}
        for arguments in (["--batch", "external-driver"], ["--batch"]):
            def execute(root, output, started, observation):
                self.assertEqual(root, self.root)
                self.assertFalse(output.is_relative_to(root))
                observation["outcome"] = "passed"
            captured = io.StringIO()
            with patch.dict(os.environ, {"FE2O3_TUTORIAL_CPU_OUTPUT_ROOT": str(self.base)}, clear=True), \
                    patch.object(adapter, "__file__", str(self.root / "scripts" / HELPER.name)), \
                    patch.object(adapter, "execute_batch", side_effect=execute) as batch, \
                    patch.object(adapter, "execute") as standalone, \
                    contextlib.redirect_stdout(captured):
                result = adapter.main(arguments)
            observation = json.loads(captured.getvalue())
            self.assertEqual(result, 0 if arguments == ["--batch"] else 1)
            self.assertEqual(batch.call_count, int(arguments == ["--batch"]))
            standalone.assert_not_called()
            self.assertEqual(observation["schema"], "fe2o3-tutorial-cpu-reference-batch-observation-v1")
            self.assertTrue(all(observation[key] is False for key in adapter.NO_AUTHORITY))
            self.assertEqual(adapter.signal.getitimer(adapter.signal.ITIMER_REAL), (0.0, 0.0))
            self.assertEqual({number: adapter.signal.getsignal(number) for number in previous}, previous)


if __name__ == "__main__":
    unittest.main()

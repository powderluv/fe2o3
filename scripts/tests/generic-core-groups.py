#!/usr/bin/env python3
"""Exercise the production core dispatcher without building or skipping a group."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CI_LOCAL = ROOT / "scripts/ci-local.sh"
POLICY = [
    "run_workspace_dependency_policy",
    "run_standalone_lockfiles",
    "run_runtime_pure_rust_policy",
    "example-manifest",
    "bounded-moe-docs",
    "run_shard_policy",
    "run_parity_matrix_checks",
    "run_format",
    "run_check",
    "run_backend_build",
    "compiler-map-evidence-tests",
    "simulation-expectation-tests",
    "tutorial-scalar-gemm-corpus-tests",
    "tutorial-default-cargo-harness-tests",
    "tutorial-production-census-tests",
    "tutorial-current-simulation-tests",
    "quickstart-shell-tests",
    "kernel-compile-matrix-shell-tests",
    "tutorial-cpu-reference-tests",
    "no-gpu-source-quickstart",
    "kir-sim-capability-matrix",
    "kir-sim-scalar-differential",
    "kir-sim-semantic-differential",
    "kir-sim-f32-differential",
    "ci-local-test-gate",
    "generic-core-group-tests",
    "generic-cpu-group-tests",
    "finalizer-test-shard-tests",
    "runtime-production-proof-pipeline-tests",
]
GROUPS = {
    "policy": POLICY,
    "cpu": ["run_cpu_tests"],
    "auxiliary": ["run_rustc_codegen_lib_tests", "run_auxiliary_tests"],
}
ALL = sum(GROUPS.values(), [])
LEAVES = [name for name in ALL if name.startswith("run_")]
HARNESS = r'''
set -Eeuo pipefail
source "$1"
shift
observe() {
  printf '%s\n' "$1"
  if [[ "$1" == "${FAIL_AT:-}" ]]; then
    return 29
  fi
}
run_step() { observe "$1"; }
''' + "\n".join(
    f"{name}() {{ observe {name}; }}"
    for name in LEAVES + ["run_all_rustc_codegen_shards"]
) + '\n"$@"\n'


class GenericCoreGroupTests(unittest.TestCase):
    def invoke(self, *arguments: str, failure: str = "",
               harness: str = HARNESS) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="fe2o3-core-group-test-") as directory:
            environment = {
                "PATH": os.environ["PATH"],
                "HOME": os.environ["HOME"],
                "CARGO_TARGET_DIR": str(Path(directory) / "target"),
                "CI_LOG_DIR": str(Path(directory) / "logs"),
                "FAIL_AT": failure,
            }
            return subprocess.run(
                ["bash", "-c", harness, "bash", str(CI_LOCAL), *arguments],
                cwd=ROOT, env=environment, text=True, capture_output=True,
                check=False, timeout=30,
            )

    def assert_trace(self, result: subprocess.CompletedProcess[str], expected: list[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), expected)

    def test_default_and_explicit_all_retain_every_existing_group(self) -> None:
        self.assert_trace(self.invoke("run_generic_core"), ALL)
        self.assert_trace(self.invoke("run_generic_core", "all"), ALL)
        self.assert_trace(self.invoke("main", "generic-core"), ALL)

    def test_legacy_groups_partition_the_same_ordered_work_exactly_once(self) -> None:
        actual = []
        for group, expected in GROUPS.items():
            with self.subTest(group=group):
                result = self.invoke("main", "generic-core", group)
                self.assert_trace(result, expected)
                actual.extend(result.stdout.splitlines())
        self.assertEqual(actual, ALL)
        self.assertEqual(len(actual), len(set(actual)))

    def test_generic_still_includes_all_core_and_codegen_shards(self) -> None:
        self.assert_trace(self.invoke("main", "generic"), ALL + ["run_all_rustc_codegen_shards"])

    def test_compiler_map_diagnostics_dispatch_exact_command_once(self) -> None:
        harness = HARNESS.replace('run_step() { observe "$1"; }', r'''
run_step() {
  printf '%s' "$1"
  shift
  printf '\t%s' "$@"
  printf '\n'
}
''')
        expected = ["compiler-map-evidence-tests", "python3", "-I", "-B",
                    "scripts/tests/compiler_map_evidence_test.py"]
        for arguments in (("main", "generic-core", "policy"),
                          ("run_generic_core",), ("main", "generic")):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments, harness=harness)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [line.split("\t") for line in result.stdout.splitlines()]
                self.assertEqual([row for row in rows if row[0] == expected[0]], [expected])

    def test_unknown_empty_or_extra_group_fails_before_any_work(self) -> None:
        for arguments in [("unknown",), ("",), ("--cpu",), ("policy", "cpu"), ("all", "all")]:
            for entry in [("run_generic_core",), ("main", "generic-core")]:
                with self.subTest(arguments=arguments, entry=entry):
                    result = self.invoke(*entry, *arguments)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")

    def test_each_group_propagates_first_failure_and_stops(self) -> None:
        for group, expected in GROUPS.items():
            with self.subTest(group=group):
                result = self.invoke("main", "generic-core", group, failure=expected[0])
                self.assertEqual(result.returncode, 29, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected[:1])

    def test_full_core_stops_before_later_groups_after_failure(self) -> None:
        for failed in ["compiler-map-evidence-tests",
                       "tutorial-default-cargo-harness-tests", "tutorial-production-census-tests",
                       "tutorial-current-simulation-tests", POLICY[-1], "run_cpu_tests",
                       "run_rustc_codegen_lib_tests"]:
            with self.subTest(failed=failed):
                result = self.invoke("main", "generic-core", failure=failed)
                self.assertEqual(result.returncode, 29, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ALL[:ALL.index(failed) + 1])




BOOTSTRAP_HARNESS = r'''
set -Eeuo pipefail
source "$1"
run_step() {
  printf 'stage:%s\n' "$1"
  shift
  "$@"
}
run_rustc_codegen_lib_tests
'''
TOOL_STUB = r'''#!{python}
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
with Path(os.environ["BOOTSTRAP_CALLS"]).open("a") as output:
    output.write(json.dumps({{"command": name, "arguments": sys.argv[1:]}}) + "\n")
if name == "rustc":
    status = int(os.environ.get("BOOTSTRAP_RUSTC_STATUS", "0"))
    if status == 0:
        print(os.environ["BOOTSTRAP_SYSROOT"])
    sys.exit(status)
if sys.argv[1:2] == ["fetch"]:
    sys.exit(int(os.environ.get("BOOTSTRAP_FETCH_STATUS", "0")))
if sys.argv[1:2] == ["test"]:
    failure = os.environ.get("BOOTSTRAP_TEST_FAILURE_SUFFIX", "")
    if failure and any(argument.endswith(failure) for argument in sys.argv[2:]):
        sys.exit(29)
    sys.exit(0)
sys.exit(97)
'''


class RustcSysrootBootstrapTests(unittest.TestCase):
    def invoke(self, *, rustc_status: int = 0, fetch_status: int = 0,
               missing: str = "", malformed: str | None = None,
               test_failure: str = ""):
        with tempfile.TemporaryDirectory(prefix="fe2o3-sysroot-bootstrap-") as directory:
            root = Path(directory)
            selected = root / "selected toolchain"
            library = selected / "lib/rustlib/src/rust/library"
            library.mkdir(parents=True)
            for filename in ("Cargo.toml", "Cargo.lock"):
                if filename != missing:
                    (library / filename).write_text("# fixture\n")
            binary = root / "bin"
            binary.mkdir()
            for tool in ("rustc", "cargo"):
                path = binary / tool
                path.write_text(TOOL_STUB.format(python=sys.executable))
                path.chmod(0o700)
            calls = root / "calls.jsonl"
            environment = {
                "PATH": f"{binary}:/usr/bin:/bin",
                "HOME": os.environ["HOME"],
                "CARGO_TARGET_DIR": str(root / "target"),
                "CI_LOG_DIR": str(root / "logs"),
                "BOOTSTRAP_CALLS": str(calls),
                "BOOTSTRAP_SYSROOT": str(selected) if malformed is None else malformed,
                "BOOTSTRAP_RUSTC_STATUS": str(rustc_status),
                "BOOTSTRAP_FETCH_STATUS": str(fetch_status),
                "BOOTSTRAP_TEST_FAILURE_SUFFIX": test_failure,
            }
            result = subprocess.run(
                ["bash", "-c", BOOTSTRAP_HARNESS, "bash", str(CI_LOCAL)],
                cwd=ROOT, env=environment, text=True, capture_output=True,
                check=False, timeout=30,
            )
            observed = [json.loads(line) for line in calls.read_text().splitlines()]
            return result, observed, str(library / "Cargo.toml")

    def test_cold_auxiliary_fetches_selected_sysroot_before_all_backend_stages(self) -> None:
        result, calls, manifest = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "stage:rustc-codegen-sysroot-dependencies",
            "stage:rustc-codegen-lib-tests",
            "stage:rustc-codegen-expanded-source-tests",
            "stage:rustc-codegen-expanded-model-tests",
            "stage:rustc-codegen-product-frame-tests",
            "stage:rustc-codegen-tile-census-source-tests",
            "stage:rustc-codegen-extractor-bin-tests",
            "stage:rustc-codegen-exporter-bin-tests",
        ])
        self.assertEqual(calls, [
            {"command": "rustc", "arguments": ["--print", "sysroot"]},
            {"command": "cargo", "arguments": [
                "fetch", "--locked", "--manifest-path", manifest,
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3", "--lib",
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3", "--lib",
                "production_rustc_driver_v1::checked_output_source_v1_tests::"
                "context_source_v29_tests::pending_source_tests::expanded_source_tests::"
                "actual_rustc_source_retains_original_neutral_and_expanded_owners",
                "--", "--ignored", "--exact", "--test-threads=1",
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3", "--lib",
                "production_rustc_driver_v1::checked_output_source_v1_tests::"
                "context_source_v29_tests::pending_source_tests::expanded_source_tests::"
                "expanded_model_tests::actual_rustc_expanded_support_model_covers_complete_roots_and_runtime_width_boundaries",
                "--", "--ignored", "--exact", "--test-threads=1",
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3", "--lib",
                "production_rustc_driver_v1::checked_output_source_v1_tests::"
                "context_source_v29_tests::pending_source_tests::expanded_source_tests::"
                "expanded_model_tests::product_frames::actual_rustc_product_carriers_retain_current_and_suspended_source_demands_v283",
                "--", "--ignored", "--exact", "--test-threads=1",
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3", "--lib",
                "production_rustc_driver_v1::checked_output_source_v1_tests::"
                "context_source_v29_tests::pending_source_tests::expanded_source_tests::"
                "tile_census_tests::actual_rustc_collected_tile_census_preserves_shared_body_counts",
                "--", "--ignored", "--exact", "--test-threads=1",
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3",
                "--bin", "fe2o3-rustc-extract",
            ]},
            {"command": "cargo", "arguments": [
                "test", "--locked", "-p", "rustc-codegen-fe2o3",
                "--bin", "fe2o3-export-sim",
            ]},
        ])

    def test_expanded_model_failure_stops_before_later_backend_stages(self) -> None:
        suffix = "actual_rustc_expanded_support_model_covers_complete_roots_and_runtime_width_boundaries"
        result, calls, _ = self.invoke(test_failure=suffix)
        self.assertEqual(result.returncode, 29, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "stage:rustc-codegen-sysroot-dependencies",
            "stage:rustc-codegen-lib-tests",
            "stage:rustc-codegen-expanded-source-tests",
            "stage:rustc-codegen-expanded-model-tests",
        ])
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[-1]["command"], "cargo")
        self.assertTrue(calls[-1]["arguments"][5].endswith("::" + suffix))

    def test_product_frame_failure_stops_before_later_backend_stages(self) -> None:
        suffix = "actual_rustc_product_carriers_retain_current_and_suspended_source_demands_v283"
        result, calls, _ = self.invoke(test_failure=suffix)
        self.assertEqual(result.returncode, 29, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "stage:rustc-codegen-sysroot-dependencies",
            "stage:rustc-codegen-lib-tests",
            "stage:rustc-codegen-expanded-source-tests",
            "stage:rustc-codegen-expanded-model-tests",
            "stage:rustc-codegen-product-frame-tests",
        ])
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[-1]["command"], "cargo")
        self.assertTrue(calls[-1]["arguments"][5].endswith("::" + suffix))

    def test_sysroot_resolution_failure_stops_before_fetch_and_tests(self) -> None:
        result, calls, _ = self.invoke(rustc_status=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(calls, [{"command": "rustc", "arguments": ["--print", "sysroot"]}])

    def test_missing_or_malformed_sysroot_stops_before_fetch_and_tests(self) -> None:
        for fault in [
            {"missing": "Cargo.toml"}, {"missing": "Cargo.lock"},
            {"malformed": ""}, {"malformed": "relative"},
            {"malformed": "/first\n/second"}, {"malformed": "/first\r/second"},
        ]:
            with self.subTest(fault=fault):
                result, calls, _ = self.invoke(**fault)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(calls, [
                    {"command": "rustc", "arguments": ["--print", "sysroot"]},
                ])

    def test_sysroot_fetch_failure_stops_before_offline_consumers(self) -> None:
        result, calls, manifest = self.invoke(fetch_status=29)
        self.assertEqual(result.returncode, 29, result.stderr)
        self.assertEqual(calls, [
            {"command": "rustc", "arguments": ["--print", "sysroot"]},
            {"command": "cargo", "arguments": [
                "fetch", "--locked", "--manifest-path", manifest,
            ]},
        ])



CPU_BOOTSTRAP_HARNESS = r'''
set -Eeuo pipefail
source "$1"
shift
run_step() {
  local name="$1"
  shift
  printf 'stage:%s\n' "$name"
  case "$name" in
    cpu-workspace-dependencies | auxiliary-workspace-dependencies) "$@" ;;
    standalone-lockfiles) return 0 ;;  # The real inventory is covered by generic-cpu-groups.
    *) return 43 ;;
  esac
}
run_host_reference_tests() { printf 'host-reference-tests\n'; }
ensure_production_cargo_fe2o3_driver() { printf 'driver-bootstrap:%s\n' "$1"; }
load_example_packages() {
  local -n output="$2"
  output=()
}
"$@"
'''


class WorkspaceDependencyBootstrapTests(unittest.TestCase):
    def invoke(self, entry: str, *, fetch_status: int = 0):
        with tempfile.TemporaryDirectory(prefix="fe2o3-workspace-bootstrap-") as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            cargo = binary / "cargo"
            cargo.write_text(TOOL_STUB.format(python=sys.executable))
            cargo.chmod(0o700)
            cargo_home = root / "empty cargo home"
            cargo_home.mkdir()
            self.assertEqual(list(cargo_home.iterdir()), [])
            calls = root / "calls.jsonl"
            environment = {
                "PATH": f"{binary}:/usr/bin:/bin",
                "HOME": os.environ["HOME"],
                "CARGO_HOME": str(cargo_home),
                "CARGO_TARGET_DIR": str(root / "target"),
                "CI_LOG_DIR": str(root / "logs"),
                "BOOTSTRAP_CALLS": str(calls),
                "BOOTSTRAP_FETCH_STATUS": str(fetch_status),
            }
            result = subprocess.run(
                ["bash", "-c", CPU_BOOTSTRAP_HARNESS, "bash", str(CI_LOCAL), entry],
                cwd=ROOT, env=environment, text=True, capture_output=True,
                check=False, timeout=30,
            )
            observed = [json.loads(line) for line in calls.read_text().splitlines()]
            return result, observed

    def test_isolated_cpu_and_auxiliary_fetch_workspace_before_offline_test_consumers(self) -> None:
        for entry, prefix, after_fetch in [
            ("run_cpu_tests", "cpu", [
                "stage:standalone-lockfiles",
                "host-reference-tests", "driver-bootstrap:cpu-tests", "stage:cargo-fe2o3-production-bins",
            ]),
            ("run_auxiliary_tests", "auxiliary", ["stage:core-doc-tests"]),
        ]:
            with self.subTest(entry=entry):
                result, calls = self.invoke(entry)
                # The first post-bootstrap build/test stage is the mocked stop boundary.
                self.assertEqual(result.returncode, 43, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [
                    f"stage:{prefix}-workspace-dependencies", *after_fetch,
                ])
                self.assertEqual(calls, [{
                    "command": "cargo",
                    "arguments": ["fetch", "--locked", "--manifest-path", str(ROOT / "Cargo.toml")],
                }])

    def test_workspace_fetch_failure_stops_both_lanes_before_build_or_tests(self) -> None:
        for entry, prefix in [("run_cpu_tests", "cpu"), ("run_auxiliary_tests", "auxiliary")]:
            with self.subTest(entry=entry):
                result, calls = self.invoke(entry, fetch_status=29)
                self.assertEqual(result.returncode, 29, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [f"stage:{prefix}-workspace-dependencies"])
                self.assertEqual(calls, [{
                    "command": "cargo",
                    "arguments": ["fetch", "--locked", "--manifest-path", str(ROOT / "Cargo.toml")],
                }])

if __name__ == "__main__":
    unittest.main()

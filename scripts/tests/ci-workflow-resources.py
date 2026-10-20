#!/usr/bin/env python3
"""Focused hosted CI resource-profile checks; never compile or invoke CI."""

from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CI_PATH = ROOT / ".github/workflows/ci.yml"
ROW_PATH = ROOT / ".github/workflows/row-softmax-v1.yml"
PREPARE_NAME = "Prepare external private temporary directory"
CORE_PREFIX = (
    "policy",
    "cpu-foundation",
    "cpu-analysis",
    "cpu-lowering",
    "cpu-pliron",
)
CORE_SUFFIX = ("cpu-integration", "auxiliary")
FINALIZER_SHARDS = tuple(f"cpu-finalize-{index}" for index in range(4))
CORE_GROUPS = (*CORE_PREFIX, "cpu-finalize", *CORE_SUFFIX)
SHARDED_CORE_GROUPS = (*CORE_PREFIX, *FINALIZER_SHARDS, *CORE_SUFFIX)
PROFILE_LINES = (
    "      CARGO_PROFILE_DEV_DEBUG: '1'",
    "      CARGO_PROFILE_TEST_DEBUG: '1'",
    "      CARGO_INCREMENTAL: '0'",
)
ROW_PROFILE_LINES = (
    "      CARGO_PROFILE_DEV_DEBUG: '0'",
    "      CARGO_PROFILE_TEST_DEBUG: '0'",
    "      CARGO_INCREMENTAL: '0'",
)


def job(source: str, name: str) -> str:
    matches = re.findall(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z][A-Za-z0-9_-]*:\n|\Z)",
        source,
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one job {name!r}")
    return matches[0]


def prepare_script(source: str) -> str:
    owner = job(source, "generic-core")
    marker = f"      - name: {PREPARE_NAME}\n"
    if owner.count(marker) != 1:
        raise ValueError("expected exactly one private temporary-directory step")
    step = owner.split(marker, 1)[1].split("\n      - name:", 1)[0]
    prefix = "        shell: bash\n        run: |\n"
    if not step.startswith(prefix):
        raise ValueError("temporary-directory step must use an unconditional Bash block")
    lines = step[len(prefix):].splitlines()
    while lines and not lines[-1]:
        lines.pop()
    if not lines or any(not line.startswith("          ") for line in lines):
        raise ValueError("unexpected temporary-directory shell indentation")
    return "\n".join(line[10:] for line in lines) + "\n"


def profiles(owner: str, expected: tuple[str, ...] = PROFILE_LINES) -> None:
    for line in expected:
        if owner.splitlines().count(line) != 1:
            raise ValueError(f"missing, duplicate, or changed resource setting: {line}")
    settings = re.findall(r"(?m)^      CARGO_(?:PROFILE_|INCREMENTAL:).*$", owner)
    if settings != list(expected):
        raise ValueError("unexpected extra Cargo profile setting")


def core_groups(source: str) -> tuple[str, ...]:
    owner = job(source, "generic-core")
    strategy = (
        "    strategy:\n"
        "      fail-fast: false\n"
        "      matrix:\n"
        "        group:\n"
    )
    matched = [
        groups for groups in (CORE_GROUPS, SHARDED_CORE_GROUPS)
        if owner.count(strategy + "".join(f"          - {group}\n" for group in groups)
                       + "    env:\n") == 1
    ]
    if owner.count("    strategy:\n") != 1 or len(matched) != 1:
        raise ValueError("core matrix must run each required group exactly once")
    for required in (
        'run: scripts/ci-local.sh generic-core "${{ matrix.group }}"',
        "CARGO_TARGET_DIR: ${{ github.workspace }}/target/ci/generic-core-${{ matrix.group }}",
        "CI_LOG_DIR: ${{ github.workspace }}/target/ci-logs/generic-core-${{ matrix.group }}",
        "name: generic-core-${{ matrix.group }}-logs-${{ github.run_attempt }}",
    ):
        if owner.count(required) != 1:
            raise ValueError(f"missing or duplicated core group routing: {required}")
    if "continue-on-error" in owner or "    if:" in owner.split("    steps:", 1)[0]:
        raise ValueError("required core groups cannot be optional")
    return matched[0]



ROW_MODES = ("debug", "release", "clippy")
ROW_ROUTES = (
    ("Run debug tests", "debug",
     "target/debug/cargo-fe2o3 test --locked --all-targets "
     "--manifest-path examples/row_softmax_v1/Cargo.toml"),
    ("Run release tests", "release",
     "target/debug/cargo-fe2o3 test --release --locked --all-targets "
     "--manifest-path examples/row_softmax_v1/Cargo.toml"),
    ("Run strict Clippy", "clippy",
     "target/debug/cargo-fe2o3 clippy --locked --all-targets --all-features "
     "--manifest-path examples/row_softmax_v1/Cargo.toml -- -D warnings"),
)
ROW_SHARED_STEPS = (
    ("Check out repository",
     "        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n"
     "        with:\n"
     "          fetch-depth: 0"),
    ("Check hosted CI resource contracts",
     "        run: python3 -I -B scripts/tests/ci-workflow-resources.py"),
    ("Install pinned Rust toolchain",
     "        run: rustup show active-toolchain"),
    ("Check formatting",
     "        run: >-\n"
     "          cargo fmt --manifest-path examples/row_softmax_v1/Cargo.toml\n"
     "          --package fe2o3-row-softmax-v1 -- --check"),
    ("Build binding-aware Cargo driver",
     "        run: cargo build --locked -p cargo-fe2o3"),
)


def row_modes(source: str) -> tuple[str, ...]:
    owner = job(source, "host-contract")
    strategy = (
        "    strategy:\n"
        "      fail-fast: false\n"
        "      matrix:\n"
        "        mode:\n"
        + "".join(f"          - {mode}\n" for mode in ROW_MODES)
        + "    env:\n"
    )
    if owner.count("    strategy:\n") != 1 or owner.count(strategy) != 1:
        raise ValueError("host matrix must run debug, release, and clippy exactly once")
    for field, expected in (
        ("name", "Host contract and fail-closed source checks (${{ matrix.mode }})"),
        ("runs-on", "ubuntu-24.04"),
        ("timeout-minutes", "15"),
    ):
        if re.findall(rf"(?m)^    {field}: (.+)$", owner) != [expected]:
            raise ValueError(f"host matrix changed {field}")
    profiles(owner, ROW_PROFILE_LINES)
    header = owner.split("    steps:\n", 1)[0]
    if "continue-on-error" in owner or "if:" in header:
        raise ValueError("all host modes must be mandatory")
    steps = re.findall(
        r"(?ms)^      - name: ([^\n]+)\n(.*?)(?=^      - name: |\Z)", owner
    )
    names = tuple(name for name, _ in steps)
    expected_names = tuple(name for name, _ in ROW_SHARED_STEPS) + tuple(
        name for name, _, _ in ROW_ROUTES
    )
    if owner.count("    steps:\n") != 1 or names != expected_names:
        raise ValueError("host modes must retain all shared and routed steps in order")
    for (name, body), (_, expected) in zip(steps, ROW_SHARED_STEPS):
        if body.rstrip("\n") != expected:
            raise ValueError(f"host matrix changed shared step {name}")
    for (name, body), (_, mode, command) in zip(steps[len(ROW_SHARED_STEPS):], ROW_ROUTES):
        route = re.fullmatch(
            rf"        if: matrix\.mode == '{mode}'\n"
            r"        run: >-\n((?:          [^\n]+\n)+)",
            body.rstrip("\n") + "\n",
        )
        if route is None:
            raise ValueError(f"host matrix changed conditional route {name}")
        folded = " ".join(line.strip() for line in route[1].splitlines())
        if shlex.split(folded) != shlex.split(command):
            raise ValueError(f"host matrix changed validation command {name}")
    return ROW_MODES


class WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ci = CI_PATH.read_text(encoding="utf-8")
        cls.row = ROW_PATH.read_text(encoding="utf-8")

    def test_resource_profiles_are_scoped_to_host_jobs(self) -> None:
        profiles(job(self.ci, "generic-core"))
        profiles(job(self.row, "host-contract"), ROW_PROFILE_LINES)
        for source, name in (
            (self.ci, "parity-policy"),
            (self.ci, "rustc-codegen-shards"),
            (self.ci, "generic-validation"),
            (self.row, "proof-contract"),
        ):
            self.assertNotIn("CARGO_PROFILE_", job(source, name))
            self.assertNotIn("CARGO_INCREMENTAL:", job(source, name))
        self.assertNotIn("CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS", self.row)
        host = job(self.row, "host-contract")
        command = "run: python3 -I -B scripts/tests/ci-workflow-resources.py"
        self.assertEqual(host.count(command), 1)
        self.assertLess(host.index(command), host.index("Build binding-aware Cargo driver"))

    def test_profile_mutations_fail_instead_of_silently_widening(self) -> None:
        for source, name, expected in (
            (self.ci, "generic-core", PROFILE_LINES),
            (self.row, "host-contract", ROW_PROFILE_LINES),
        ):
            owner = job(source, name)
            for old in expected:
                changed_value = old.replace("'1'", "'0'") if "'1'" in old else old.replace("'0'", "'1'")
                for replacement in ("", old + "\n" + old, changed_value):
                    with self.subTest(job=name, old=old, replacement=replacement):
                        with self.assertRaises(ValueError):
                            profiles(owner.replace(old, replacement, 1), expected)
            for setting in (
                "CARGO_PROFILE_TEST_DEBUG_ASSERTIONS: 'false'",
                "CARGO_PROFILE_DEV_OPT_LEVEL: '1'",
                "CARGO_PROFILE_TEST_OPT_LEVEL: '1'",
                "CARGO_PROFILE_TEST_OVERFLOW_CHECKS: 'false'",
            ):
                with self.subTest(job=name, setting=setting):
                    with self.assertRaises(ValueError):
                        profiles(owner + f"\n      {setting}\n", expected)

    def test_existing_mandatory_commands_and_limits_remain(self) -> None:
        generic = job(self.ci, "generic-core")
        host = job(self.row, "host-contract")
        core_groups(self.ci)
        self.assertIn("    timeout-minutes: 90\n", generic)
        self.assertIn("    timeout-minutes: 15\n", host)
        proof = job(self.row, "proof-contract")
        for command in ("run: examples/row_softmax_v1/run-verus.sh",
                        "run: sh examples/qwen3_rope_kv_v1/run-verus.sh"):
            self.assertEqual(proof.count(command), 1)
        self.assertNotIn("continue-on-error:", proof)
        self.assertNotIn("if:", proof)
        for command in (
            "cargo fmt --manifest-path examples/row_softmax_v1/Cargo.toml",
            "run: cargo build --locked -p cargo-fe2o3",
            "target/debug/cargo-fe2o3 test --locked --all-targets",
            "target/debug/cargo-fe2o3 test --release --locked --all-targets",
            "target/debug/cargo-fe2o3 clippy --locked --all-targets --all-features",
            "--manifest-path examples/row_softmax_v1/Cargo.toml -- -D warnings",
        ):
            self.assertEqual(host.count(command), 1, command)
        for owner in (generic, host):
            self.assertNotIn("continue-on-error", owner)
        executable_source = (
            ROOT / "crates/cargo-fe2o3/src/rustc_wrapper/pinned_executable.rs"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "pub(crate) const MAX_EXECUTABLE_BYTES: u64 = 512 * 1024 * 1024;",
            executable_source,
        )


    def test_row_modes_are_mandatory_independent_and_exactly_routed(self) -> None:
        self.assertEqual(row_modes(self.row), ROW_MODES)
        self.assertEqual(len(set(ROW_MODES)), 3)

    def test_row_matrix_cannot_omit_duplicate_or_replace_a_mode(self) -> None:
        for mode in ROW_MODES:
            entry = f"          - {mode}\n"
            for replacement in ("", entry + entry, "          - unknown\n"):
                with self.subTest(mode=mode, replacement=replacement):
                    changed = self.row.replace(entry, replacement, 1)
                    self.assertNotEqual(changed, self.row)
                    with self.assertRaises(ValueError):
                        row_modes(changed)
        for old, new in (
            ("      fail-fast: false\n", "      fail-fast: true\n"),
            ("          - clippy\n", "          - clippy\n        exclude: []\n"),
            ("    steps:\n", "    if: false\n    steps:\n"),
            ("    steps:\n", "    continue-on-error: true\n    steps:\n"),
        ):
            with self.subTest(old=old):
                host = job(self.row, "host-contract")
                changed = host.replace(old, new, 1)
                self.assertNotEqual(changed, host)
                with self.assertRaises(ValueError):
                    row_modes(self.row.replace(host, changed, 1))

    def test_row_routes_cannot_skip_swap_or_weaken_commands(self) -> None:
        for _, mode, _ in ROW_ROUTES:
            route = f"        if: matrix.mode == '{mode}'"
            for replacement in ("", route.replace(mode, "unknown"), route + " && false"):
                with self.subTest(mode=mode, replacement=replacement):
                    changed = self.row.replace(route, replacement, 1)
                    self.assertNotEqual(changed, self.row)
                    with self.assertRaises(ValueError):
                        row_modes(changed)
        for old, new in (
            ("test --release --locked", "test --locked"),
            ("clippy --locked --all-targets --all-features", "clippy --locked --all-targets"),
            ("--manifest-path examples/row_softmax_v1/Cargo.toml -- -D warnings",
             "--manifest-path examples/row_softmax_v1/Cargo.toml"),
            ("run: cargo build --locked -p cargo-fe2o3", "run: cargo build -p cargo-fe2o3"),
            ("          fetch-depth: 0", "          fetch-depth: 1"),
        ):
            with self.subTest(old=old):
                changed = self.row.replace(old, new, 1)
                self.assertNotEqual(changed, self.row)
                with self.assertRaises(ValueError):
                    row_modes(changed)

    def test_row_modes_cannot_widen_resources_or_substitute_shared_steps(self) -> None:
        for old, new in (
            ("    timeout-minutes: 15", "    timeout-minutes: 30"),
            ("    runs-on: ubuntu-24.04", "    runs-on: self-hosted"),
            ("      CARGO_PROFILE_TEST_DEBUG: '0'", "      CARGO_PROFILE_TEST_DEBUG: '1'"),
            ("run: rustup show active-toolchain", "run: true"),
            ("run: python3 -I -B scripts/tests/ci-workflow-resources.py", "run: true"),
        ):
            with self.subTest(old=old):
                host = job(self.row, "host-contract")
                changed = host.replace(old, new, 1)
                self.assertNotEqual(changed, host)
                with self.assertRaises(ValueError):
                    row_modes(self.row.replace(host, changed, 1))


    def test_preparation_is_exactly_once_before_the_unchanged_gate(self) -> None:
        generic = job(self.ci, "generic-core")
        self.assertLess(generic.index(PREPARE_NAME), generic.index("Run generic core validation"))
        self.assertIn("umask 077\n", prepare_script(self.ci))
        self.assertNotIn("rm ", prepare_script(self.ci))
        self.assertNotIn("sudo ", prepare_script(self.ci))
        marker = f"      - name: {PREPARE_NAME}\n"
        for changed in (self.ci.replace(marker, "", 1), self.ci.replace(marker, marker + marker, 1)):
            with self.assertRaises(ValueError):
                prepare_script(changed)

    def test_core_group_mutations_cannot_omit_or_duplicate_work(self) -> None:
        current = "".join(f"          - {group}\n" for group in core_groups(self.ci))
        for groups in (CORE_GROUPS, SHARDED_CORE_GROUPS):
            entries = "".join(f"          - {group}\n" for group in groups)
            source = self.ci.replace(current, entries, 1)
            self.assertEqual(core_groups(source), groups)
            for group in groups:
                entry = f"          - {group}\n"
                self.assertEqual(job(source, "generic-core").count(entry), 1)
                for replacement in ("", entry + entry):
                    with self.subTest(groups=groups, group=group, replacement=replacement):
                        changed = source.replace(entry, replacement, 1)
                        self.assertNotEqual(changed, source)
                        with self.assertRaises(ValueError):
                            core_groups(changed)
        for old, new in (
            ("          - policy\n", "          - all\n"),
            ("      fail-fast: false\n", "      fail-fast: true\n"),
            ("          - auxiliary\n", "          - auxiliary\n        exclude: []\n"),
            ('generic-core "${{ matrix.group }}"', "generic-core policy"),
            ("generic-core-${{ matrix.group }}", "generic-core"),
            ("name: Generic core (${{ matrix.group }})\n", "name: Generic core\n    continue-on-error: true\n"),
            ("name: Generic core (${{ matrix.group }})\n", "name: Generic core\n    if: false\n"),
        ):
            with self.subTest(old=old, new=new):
                changed = self.ci.replace(old, new, 1)
                self.assertNotEqual(changed, self.ci)
                with self.assertRaises(ValueError):
                    core_groups(changed)

    def test_finalizer_layouts_cannot_mix_or_add_unknown_groups(self) -> None:
        current = "".join(f"          - {group}\n" for group in core_groups(self.ci))
        for groups in (CORE_GROUPS, SHARDED_CORE_GROUPS):
            entries = "".join(f"          - {group}\n" for group in groups)
            source = self.ci.replace(current, entries, 1)
            extras = ("cpu-finalize", *FINALIZER_SHARDS, "cpu-finalize-4", "all")
            for extra in extras:
                with self.subTest(groups=groups, extra=extra):
                    changed = source.replace(entries, entries + f"          - {extra}\n", 1)
                    with self.assertRaises(ValueError):
                        core_groups(changed)


class TemporaryDirectoryShellTests(unittest.TestCase):
    """Run the extracted shell only on fresh test-owned fake runner paths."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = prepare_script(CI_PATH.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="fe2o3-ci-resource-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "checkout"
        self.runner_temp = self.root / "runner-temp"
        self.workspace.mkdir(mode=0o700)
        self.runner_temp.mkdir(mode=0o700)
        self.env_file = self.root / "github-env"
        self.env_file.write_text("EXISTING=retained\n", encoding="utf-8")
        self.sentinel = self.runner_temp / "unrelated-retained"
        self.sentinel.write_bytes(b"keep this unrelated fixture\n")
        self.environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "RUNNER_TEMP": str(self.runner_temp),
            "GITHUB_WORKSPACE": str(self.workspace),
            "GITHUB_ENV": str(self.env_file),
        }

    def invoke(
        self,
        *,
        overrides: dict[str, str | None] | None = None,
        prefix: str = "",
    ) -> subprocess.CompletedProcess[str]:
        environment = self.environment.copy()
        for name, value in (overrides or {}).items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", prefix + "\n" + self.script],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(self.sentinel.read_bytes(), b"keep this unrelated fixture\n")
        return result

    def rejected(self, diagnostic: str, **kwargs: object) -> None:
        before = self.env_file.read_bytes()
        result = self.invoke(**kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(diagnostic, result.stderr)
        self.assertEqual(self.env_file.read_bytes(), before)

    def exported_directories(self) -> list[Path]:
        lines = self.env_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "EXISTING=retained")
        directories = []
        for line in lines[1:]:
            self.assertTrue(line.startswith("TMPDIR="))
            directory = Path(line.removeprefix("TMPDIR="))
            self.assertEqual(directory.parent, self.runner_temp)
            self.assertEqual(directory.resolve(), directory)
            self.assertFalse(directory.is_relative_to(self.workspace))
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(directory.stat().st_uid, os.getuid())
            directories.append(directory)
        return directories

    def test_creates_a_fresh_private_external_directory_each_time(self) -> None:
        for _ in range(2):
            result = self.invoke()
            self.assertEqual(result.returncode, 0, result.stderr)
        directories = self.exported_directories()
        self.assertEqual(len(directories), 2)
        self.assertNotEqual(directories[0], directories[1])

    def test_existing_parent_metadata_is_not_a_new_policy(self) -> None:
        self.runner_temp.chmod(0o777)
        self.env_file.chmod(0o664)
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.exported_directories()), 1)

    def test_existing_parent_spellings_are_canonicalized(self) -> None:
        alias = self.root / "runner-alias"
        alias.symlink_to(self.runner_temp, target_is_directory=True)
        for value in (str(alias), str(self.runner_temp) + "/."):
            result = self.invoke(overrides={"RUNNER_TEMP": value})
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.exported_directories()), 2)

    def test_missing_environment_rejects_before_export(self) -> None:
        for name in ("RUNNER_TEMP", "GITHUB_WORKSPACE", "GITHUB_ENV"):
            with self.subTest(name=name):
                self.rejected(f"{name} is required", overrides={name: None})

    def test_missing_or_empty_runner_temp_rejects(self) -> None:
        self.rejected("RUNNER_TEMP is required", overrides={"RUNNER_TEMP": ""})
        self.rejected(
            "RUNNER_TEMP cannot be canonicalized",
            overrides={"RUNNER_TEMP": str(self.root / "missing")},
        )

    def test_existing_non_directory_rejects(self) -> None:
        self.rejected(
            "runner temporary root and checkout must be directories",
            overrides={"RUNNER_TEMP": str(self.sentinel)},
        )

    def test_checkout_and_nested_runner_temp_reject_before_mktemp(self) -> None:
        nested = self.workspace / "temporary"
        nested.mkdir(mode=0o700)
        workspace_alias = self.root / "checkout-alias"
        workspace_alias.symlink_to(self.workspace, target_is_directory=True)
        for directory in (self.workspace, nested):
            with self.subTest(directory=directory):
                self.rejected(
                    "RUNNER_TEMP must be outside the checkout",
                    overrides={
                        "RUNNER_TEMP": str(directory),
                        "GITHUB_WORKSPACE": str(workspace_alias),
                    },
                )
                self.assertEqual(list(directory.glob("fe2o3-generic-core.*")), [])

    def test_canonicalization_error_rejects(self) -> None:
        self.rejected("cannot be canonicalized", prefix="realpath() { return 73; }\n")

    def test_mktemp_failure_does_not_export(self) -> None:
        self.rejected(
            "cannot create fresh temporary directory",
            prefix="mktemp() { return 74; }\n",
        )

    def test_created_directory_mode_is_rechecked(self) -> None:
        prefix = r"""
mktemp() {
  local test_created
  test_created="$(command mktemp "$@")" || return
  chmod 755 -- "$test_created"
  printf '%s\n' "$test_created"
}
"""
        self.rejected("lost private external custody", prefix=prefix)

    def test_reported_foreign_child_owner_rejects(self) -> None:
        # Child metadata fault injection only: no chown or foreign-directory reads.
        prefix = r"""
stat() {
  if [[ "$1" == -c && "$2" == %u ]]; then
    printf '%s\n' "$(( $(id -u) + 1 ))"
  else
    command stat "$@"
  fi
}
"""
        self.rejected("lost private external custody", prefix=prefix)

    def test_created_path_must_be_single_line(self) -> None:
        self.rejected(
            "not a real single-line path",
            prefix=r"""mktemp() { printf '%s\nFAKE=value\n' "$RUNNER_TEMP"; }""",
        )

    def test_created_path_cannot_be_substituted_with_checkout(self) -> None:
        self.rejected(
            "lost private external custody",
            prefix=r"""mktemp() { printf '%s\n' "$GITHUB_WORKSPACE"; }""",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

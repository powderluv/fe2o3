#!/usr/bin/env python3
"""Observe manifest-selected host suites; never grant tutorial qualification.

Workspace tests and Cargo configuration remain trusted, as in cargo-fe2o3's
binding-only host path. Input-after checks are not a hostile-code sandbox or
an atomic snapshot of every external file a test can open.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any


MAX_STREAM = 16 * 1024 * 1024
MAX_EVENTS = 100_000
MAX_FILES = 65_536
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_FILE = 16 * 1024 * 1024
MAX_MANIFEST = 1024 * 1024
MAX_SECONDS = 1200
MAX_BATCH_SUITES = 36
# Match the binding-only host runner's pinned_executable::MAX_BINDING_HOST_TEST_BYTES.
# The shared tool/authority pin limit remains 512 MiB.
# This is an on-disk observation limit, not executed-image provenance.
MAX_EXECUTABLE_BYTES = 1024 * 1024 * 1024
ARTIFACT_HASH_CHUNK = 64 * 1024
WRAPPER = "scripts/run-tutorial-cpu-reference.sh"
NO_AUTHORITY = {
    "qualification": False,
    "compilerQualified": False,
    "simulationQualified": False,
    "policyQualified": False,
    "hardwareExecuted": False,
}


class ObservationError(Exception):
    def __init__(self, message: str, outcome: str = "invalid") -> None:
        super().__init__(message)
        self.outcome = outcome


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationError(message)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(payload: bytes) -> Any:
    def nonfinite(value: str) -> None:
        raise ObservationError(f"nonfinite JSON number: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number

    try:
        return json.loads(payload, object_pairs_hook=unique_object, parse_constant=nonfinite,
                          parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ObservationError(f"invalid JSON: {error}") from error


def read_regular(path: Path, limit: int = MAX_FILE) -> bytes:
    require(path.is_absolute() and "/proc/" not in str(path), "unresolved process path")
    require(path.resolve(strict=True) == path, f"noncanonical path: {path}")
    before = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, f"invalid/big file: {path}")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(len(payload) <= limit and all(getattr(before, field) == getattr(after, field) for field in fields)
            and len(payload) == before.st_size,
            f"file changed/grew while reading: {path}")
    return payload


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relative_file(root: Path, spelling: Any) -> Path:
    require(isinstance(spelling, str), "path must be a string")
    path = PurePosixPath(spelling)
    require(not path.is_absolute() and path.as_posix() == spelling
            and all(part not in {"", ".", ".."} for part in path.parts), "invalid relative path")
    result = root.joinpath(*path.parts)
    read_regular(result)
    require(result.is_relative_to(root), "source escapes repository")
    return result


def parse_selection(arguments: list[str]) -> dict[str, Any]:
    require(2 <= len(arguments) <= 6 and all(isinstance(item, str) for item in arguments),
            "invalid host suite arguments")
    require(arguments[1] == "lib"
            or (len(arguments) >= 3 and arguments[1] == "test"
                and re.fullmatch(r"[A-Za-z0-9_-]+", arguments[2]) is not None),
            "expected <Cargo.toml> lib or <Cargo.toml> test <literal target> before feature options")
    offset = 2 if arguments[1] == "lib" else 3
    selection = {"defaultFeatures": True, "features": [], "profile": "test",
                 "selector": arguments[1:offset]}
    seen: set[str] = set()
    while offset < len(arguments):
        option = arguments[offset]
        require(option in {"--no-default-features", "--features"} and option not in seen,
                "duplicate or unsupported host feature option")
        seen.add(option)
        offset += 1
        if option == "--no-default-features":
            selection["defaultFeatures"] = False
        else:
            require(offset < len(arguments), "missing host feature names")
            features = arguments[offset].split(",")
            require(all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", item) is not None
                        for item in features) and features == sorted(set(features)),
                    "host features must be nonempty sorted unique local names")
            selection["features"] = features
            offset += 1
    return selection


def validate_features(manifest: Path, configuration: dict[str, Any]) -> None:
    try:
        document = tomllib.loads(read_regular(manifest).decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ObservationError(f"invalid host package manifest: {error}") from error
    declared = document.get("features", {})
    require(isinstance(declared, dict) and all(name in declared for name in configuration["features"]),
            "undeclared local host feature")


def feature_arguments(configuration: dict[str, Any]) -> list[str]:
    result = [] if configuration["defaultFeatures"] else ["--no-default-features"]
    if configuration["features"]:
        result.extend(["--features", ",".join(configuration["features"])])
    return result


def select_suite(manifest: dict[str, Any], arguments: list[str]) -> dict[str, Any]:
    configuration = parse_selection(arguments)
    matches = [suite for suite in manifest["qualification"]["suites"]
               if suite["gate"] == "cpu-reference"
               and suite["command"]["executable"] == WRAPPER
               and suite["command"]["arguments"] == arguments]
    require(len(matches) == 1, "expected one exact declared CPU-reference suite")
    suite = matches[0]
    command = suite["command"]
    require(command["workingDirectory"] == "." and command["environment"] == [],
            "unsupported suite working directory/environment")
    require(type(command["timeoutSeconds"]) is int and 0 < command["timeoutSeconds"] <= MAX_SECONDS,
            "unsupported suite timeout")
    fixtures = {fixture["fixtureId"]: fixture for fixture in manifest["compilerFixtures"]}
    covered = sorted({item for coverage in suite["coverage"] for item in coverage["fixtureIds"]})
    require(bool(covered), "suite has no fixture coverage")
    for item in covered:
        require(item in fixtures and fixtures[item]["compilerInput"]["packageManifest"] == arguments[0],
                "CPU suite coverage belongs to another source package")
    return {
        "suiteId": suite["suiteId"], "declaredCommand": command, "coverage": suite["coverage"],
        "fixtureContracts": [{"fixtureId": item, "compilerInput": fixtures[item]["compilerInput"]}
                             for item in covered],
        "hostConfiguration": configuration,
    }


def load_plan(root: Path, arguments: list[str]) -> tuple[dict[str, Any], Any]:
    validator_path = root / "scripts/validate-tutorial-kernel-manifest.py"
    read_regular(validator_path)
    spec = importlib.util.spec_from_file_location("tutorial_source_contract", validator_path)
    require(spec is not None and spec.loader is not None, "missing source validator")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    manifest_path = root / "config/tutorial-kernel-manifest-v1.json"
    payload = read_regular(manifest_path, MAX_MANIFEST)
    manifest = decode_json(payload)
    try:
        validator.validate_manifest(root, manifest)
    except SystemExit as error:
        raise ObservationError(f"source contract refused: {error}") from error
    require(read_regular(manifest_path, MAX_MANIFEST) == payload, "manifest changed during validation")
    plan = select_suite(manifest, arguments)
    plan["manifestSha256"] = digest(payload)
    plan["validatorSha256"] = digest(read_regular(validator_path))
    validate_features(relative_file(root, arguments[0]), plan["hostConfiguration"])
    return plan, validator


def selected_target(root: Path, metadata: dict[str, Any], manifest: Path,
                    selector: list[str]) -> dict[str, Any]:
    packages = [package for package in metadata["packages"]
                if package["manifest_path"] == str(manifest)]
    require(len(packages) == 1, "metadata does not bind the exact package manifest")
    package = packages[0]
    document = tomllib.loads(read_regular(manifest).decode("utf-8"))
    kind = "lib" if selector == ["lib"] else "test"
    targets = [target for target in package["targets"] if target["kind"] == [kind]
               and (kind == "lib" or target["name"] == selector[1])]
    require(len(targets) == 1, "expected one exact selected Cargo target")
    target = targets[0]
    source = Path(target["src_path"])
    read_regular(source)
    require(source.is_relative_to(manifest.parent) and source.is_relative_to(root), "target source escapes package")
    require(target.get("test") is True, "selected target is not test-enabled")
    if kind == "lib":
        declaration = document.get("lib", {})
    else:
        declarations = [item for item in document.get("test", []) if item.get("name") == target["name"]]
        require(len(declarations) <= 1, "duplicate test target declaration")
        if declarations:
            declaration = declarations[0]
        else:
            require(document["package"].get("autotests", True) is True
                    and source in {manifest.parent / "tests" / (target["name"] + ".rs"),
                                   manifest.parent / "tests" / target["name"] / "main.rs"},
                    "unbound auto integration-test target")
            declaration = {}
    require(declaration.get("harness", True) is True and declaration.get("test", True) is True,
            "custom/disabled test harness is unsupported")
    nodes = [node for node in metadata["resolve"]["nodes"] if node["id"] == package["id"]]
    require(len(nodes) == 1 and isinstance(nodes[0]["features"], list)
            and all(isinstance(item, str) for item in nodes[0]["features"]), "missing resolved host features")
    companions = []
    if kind == "test":
        for companion in package["targets"]:
            if companion["kind"] != ["bin"]:
                continue
            companion_source = Path(companion["src_path"])
            read_regular(companion_source)
            require(companion_source.is_relative_to(manifest.parent)
                    and companion_source.is_relative_to(root)
                    and companion["crate_types"] == ["bin"], "unbound companion binary target")
            identity = {"name": companion["name"], "kind": ["bin"], "crateTypes": ["bin"],
                        "source": str(companion_source)}
            require(all(item["name"] != identity["name"] for item in companions), "duplicate companion target")
            companions.append(identity)
    return {"packageId": package["id"], "name": target["name"], "kind": [kind],
            "crateTypes": target["crate_types"], "features": sorted(nodes[0]["features"]),
            "source": str(source), "manifest": str(manifest), "companionTargets": companions}


def batch_plans(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    suites = [suite for suite in manifest["qualification"]["suites"] if suite["gate"] == "cpu-reference"]
    require(1 <= len(suites) <= MAX_BATCH_SUITES,
            f"batch requires between 1 and {MAX_BATCH_SUITES} declared CPU suites")
    plans = []
    identities: set[str] = set()
    commands: set[tuple[str, ...]] = set()
    for suite in suites:
        arguments = suite["command"]["arguments"]
        plan = select_suite(manifest, arguments)
        require(isinstance(plan["suiteId"], str) and plan["suiteId"] not in identities
                and tuple(arguments) not in commands, "duplicate batch suite or command")
        identities.add(plan["suiteId"])
        commands.add(tuple(arguments))
        plans.append(plan)
    return plans


def load_batch_plans(root: Path) -> tuple[list[dict[str, Any]], Any]:
    validator_path = root / "scripts/validate-tutorial-kernel-manifest.py"
    read_regular(validator_path)
    spec = importlib.util.spec_from_file_location("tutorial_source_contract", validator_path)
    require(spec is not None and spec.loader is not None, "missing source validator")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    manifest_path = root / "config/tutorial-kernel-manifest-v1.json"
    payload = read_regular(manifest_path, MAX_MANIFEST)
    manifest = decode_json(payload)
    try:
        validator.validate_manifest(root, manifest)
    except SystemExit as error:
        raise ObservationError(f"source contract refused: {error}") from error
    require(read_regular(manifest_path, MAX_MANIFEST) == payload, "manifest changed during validation")
    plans = batch_plans(manifest)
    validator_hash = digest(read_regular(validator_path))
    for plan in plans:
        plan.update(manifestSha256=digest(payload), validatorSha256=validator_hash)
        validate_features(relative_file(root, plan["declaredCommand"]["arguments"][0]),
                          plan["hostConfiguration"])
    return plans, validator


def count(value: Any, label: str) -> int:
    require(type(value) is int and 0 <= value <= MAX_EVENTS, f"invalid count: {label}")
    return value


def artifact_digest(path: Path) -> str:
    require(path.is_absolute() and "/proc/" not in str(path), "unresolved process path")
    require(path.resolve(strict=True) == path, f"noncanonical artifact path: {path}")
    before = path.stat(follow_symlinks=False)
    limit = MAX_EXECUTABLE_BYTES
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            f"invalid/big artifact: {path}")
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")

    def unchanged(info: Any) -> bool:
        return all(getattr(before, field) == getattr(info, field) for field in fields)

    state = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        require(unchanged(os.fstat(stream.fileno())), f"artifact changed while opening: {path}")
        while True:
            chunk = stream.read(min(ARTIFACT_HASH_CHUNK, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= limit and total <= before.st_size, f"artifact grew while hashing: {path}")
            state.update(chunk)
        require(total == before.st_size and unchanged(os.fstat(stream.fileno()))
                and unchanged(path.stat(follow_symlinks=False)),
                f"artifact changed/short read while hashing: {path}")
    return state.hexdigest()


def observed_executable(executable: Any, target_root: Path) -> dict[str, str]:
    require(isinstance(executable, str), "missing executable path")
    path = Path(executable)
    require(path.is_absolute() and "/proc/" not in str(path)
            and path.is_relative_to(target_root), "unbound executable path")
    require(path.resolve(strict=True) == path and path.is_file()
            and os.access(path, os.X_OK), "invalid executable")
    return {"path": str(path), "observedSha256": artifact_digest(path),
            "digestScope": "post-execution-on-disk-artifact-only"}


def test_observation(payload: bytes, expected: dict[str, Any], target_root: Path) -> dict[str, Any]:
    require(len(payload) <= MAX_STREAM, "test output exceeds bound")
    tests: dict[str, str] = {}
    details: dict[str, dict[str, Any]] = {}
    artifacts = 0
    built = False
    started: int | None = None
    terminal: dict[str, Any] | None = None
    executable_identity: dict[str, Any] | None = None
    companions: dict[str, dict[str, Any]] = {}
    for ordinal, line in enumerate(payload.splitlines()):
        require(ordinal < MAX_EVENTS and bool(line.strip()), "too many/empty event lines")
        event = decode_json(line)
        require(isinstance(event, dict), "event is not an object")
        require(terminal is None, "event after suite terminal")
        reason = event.get("reason")
        if reason == "compiler-artifact":
            require(not built and started is None, "artifact after build completion")
            profile = event.get("profile", {})
            executable = event.get("executable")
            if executable is not None or profile.get("test") is True:
                target = event.get("target", {})
                if profile.get("test") is False and executable is not None:
                    matches = [item for item in expected.get("companionTargets", [])
                               if target.get("name") == item["name"]
                               and target.get("kind") == item["kind"]
                               and target.get("crate_types") == item["crateTypes"]
                               and target.get("src_path") == item["source"]]
                    require(expected["kind"] == ["test"] and len(matches) == 1
                            and event.get("package_id") == expected["packageId"]
                            and event.get("features") == expected["features"]
                            and matches[0]["name"] not in companions, "unexpected companion artifact")
                    companions[matches[0]["name"]] = observed_executable(executable, target_root)
                    continue
                require(profile.get("test") is True and isinstance(executable, str),
                        "unexpected executable or incomplete harness artifact")
                require(event.get("package_id") == expected["packageId"]
                        and target.get("name") == expected["name"]
                        and target.get("kind") == expected["kind"]
                        and target.get("crate_types") == expected["crateTypes"]
                        and event.get("features") == expected["features"]
                        and target.get("src_path") == expected["source"], "wrong harness artifact identity")
                executable_identity = observed_executable(executable, target_root)
                artifacts += 1
                require(artifacts == 1, "duplicate harness artifact")
        elif reason == "build-finished":
            require(not built and artifacts == 1 and event.get("success") is True, "unsuccessful/duplicate build")
            built = True
        elif reason in {"compiler-message", "build-script-executed"}:
            require(not built, "build diagnostic after build completion")
        elif event.get("type") == "suite":
            require(built, "suite precedes successful build")
            if event.get("event") == "started":
                require(started is None, "duplicate suite start")
                started = count(event.get("test_count"), "test_count")
            else:
                require(started is not None and event.get("event") in {"ok", "failed"}, "invalid suite terminal")
                terminal = event
        elif event.get("type") == "test":
            require(started is not None, "test precedes suite start")
            name, outcome = event.get("name"), event.get("event")
            require(isinstance(name, str) and 0 < len(name) <= 4096, "invalid test name")
            if outcome == "started":
                require(name not in tests, "duplicate test start")
            elif outcome == "ignored":
                require(tests.get(name) in {None, "started"}, "duplicate ignored test")
            else:
                require(outcome in {"ok", "failed"} and tests.get(name) == "started", "unmatched test terminal")
            tests[name] = outcome
            if outcome != "started":
                details[name] = {key: value for key, value in event.items() if key not in {"type", "event", "name"}}
        else:
            raise ObservationError("unknown Cargo/libtest event")
    require(terminal is not None and started is not None, "missing completed test suite")
    totals = {key: count(terminal.get(key), key)
              for key in ("passed", "failed", "ignored", "measured", "filtered_out")}
    observed = {"passed": list(tests.values()).count("ok"), "failed": list(tests.values()).count("failed"),
                "ignored": list(tests.values()).count("ignored")}
    require(all(state in {"ok", "failed", "ignored"} for state in tests.values())
            and len(tests) == started and all(totals[key] == value for key, value in observed.items())
            and totals["measured"] == 0 and totals["filtered_out"] == 0,
            "test counts, completion or selection do not reconcile")
    require((terminal["event"] == "ok") == (totals["failed"] == 0), "inconsistent suite status")
    outcome = "failed" if totals["failed"] else "partial" if totals["ignored"] else "empty" if started == 0 else "passed"
    paths = [item["path"] for item in companions.values()] + [executable_identity["path"]]
    require(len(paths) == len(set(paths)), "harness/companion executable paths overlap")
    return {"outcome": outcome, "counts": totals, "executable": executable_identity,
            "companionArtifacts": companions,
            "tests": [{"name": name, "status": state, "details": details[name]}
            for name, state in tests.items()]}


def footprint(roots: list[Path], files: list[Path], deadline: float) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    entries = 0

    def add(path: Path) -> None:
        nonlocal total
        require(time.monotonic() < deadline, "input scan deadline exceeded")
        if str(path) in result:
            return
        payload = read_regular(path)
        total += len(payload)
        require(len(result) < MAX_FILES and total <= MAX_INPUT_BYTES, "input footprint exceeds bound")
        result[str(path)] = digest(payload)

    for path in files:
        add(path)
    for root in roots:
        require(root.resolve(strict=True) == root and root.is_dir(), "invalid input root")
        for directory, directories, names in os.walk(root, followlinks=False):
            require(time.monotonic() < deadline, "input scan deadline exceeded")
            entries += 1 + len(directories) + len(names)
            require(entries <= MAX_FILES, "input traversal exceeds entry bound")
            directories.sort()
            names.sort()
            for name in directories:
                require(not (Path(directory) / name).is_symlink(), "symlink in input tree")
            directories[:] = [name for name in directories if name not in {"target", ".git", "__pycache__"}]
            for name in names:
                add(Path(directory) / name)
    return result


def configuration_candidates(root: Path, env: dict[str, str]) -> list[Path]:
    roots = {base / ".cargo" for base in (root, *root.parents)}
    roots.add(Path(env.get("CARGO_HOME", str(Path.home() / ".cargo"))).resolve(strict=True))
    return sorted(base / name for base in roots for name in ("config", "config.toml"))


def configuration_snapshot(paths: list[Path], deadline: float) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    total = 0
    require(len(paths) <= MAX_FILES, "too many configuration candidates")
    for path in paths:
        require(time.monotonic() < deadline, "configuration scan deadline exceeded")
        try:
            path.lstat()
        except FileNotFoundError:
            result[str(path)] = None
            continue
        payload = read_regular(path)
        total += len(payload)
        require(total <= MAX_INPUT_BYTES, "configuration footprint exceeds bound")
        result[str(path)] = digest(payload)
    return result


def revalidate_plan(root: Path, arguments: list[str], plan: dict[str, Any], validator: Any) -> None:
    manifest_path = root / "config/tutorial-kernel-manifest-v1.json"
    validator_path = root / "scripts/validate-tutorial-kernel-manifest.py"
    payload = read_regular(manifest_path, MAX_MANIFEST)
    require(digest(payload) == plan["manifestSha256"], "original tutorial manifest changed")
    require(digest(read_regular(validator_path)) == plan["validatorSha256"], "source validator changed")
    manifest = decode_json(payload)
    try:
        # validate_manifest constructs a fresh package cache on each call.
        validator.validate_manifest(root, manifest)
    except SystemExit as error:
        raise ObservationError(f"bound source contract refused: {error}") from error
    checked = select_suite(manifest, arguments)
    validate_features(relative_file(root, arguments[0]), checked["hostConfiguration"])
    require(all(plan[key] == value for key, value in checked.items()), "selected contract changed")
    require(read_regular(manifest_path, MAX_MANIFEST) == payload
            and digest(read_regular(validator_path)) == plan["validatorSha256"],
            "source contract inputs changed during validation")


def child(argv: list[str], cwd: Path, env: dict[str, str], directory: Path,
          label: str, deadline: float, phases: list[dict[str, Any]]) -> bytes:
    record: dict[str, Any] = {"label": label, "argv": argv, "returncode": None}
    phases.append(record)
    require(time.monotonic() < deadline, "execution deadline exceeded")
    paths = [directory / (label + suffix) for suffix in (".stdout", ".stderr")]
    counts = [0, 0]
    process: subprocess.Popen[bytes] | None = None
    streams = []
    failure: BaseException | None = None
    try:
        streams = [path.open("xb") for path in paths]
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        with selectors.DefaultSelector() as selector:
            for index, pipe in enumerate((process.stdout, process.stderr)):
                require(pipe is not None, "missing child pipe")
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, index)
            while selector.get_map():
                require(time.monotonic() < deadline, "execution deadline exceeded")
                for key, _ in selector.select(min(0.1, max(0, deadline - time.monotonic()))):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    index = key.data
                    counts[index] += len(data)
                    require(counts[index] <= MAX_STREAM, "child output exceeds bound")
                    streams[index].write(data)
        # Keep the group leader unreaped until group cleanup, so its PID cannot
        # be reused for an unrelated process group before killpg.
        while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            require(time.monotonic() < deadline, "execution deadline exceeded")
            time.sleep(0.01)
    except BaseException as error:
        failure = error
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                failure = ObservationError("child cleanup failed")
            record["returncode"] = process.returncode
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        for stream in streams:
            stream.close()
        record["logs"] = [{"path": str(path), "bytes": path.stat().st_size,
                           "sha256": digest(path.read_bytes())} for path in paths if path.exists()]
        record["observedStreamBytes"] = counts
        record["completeLogs"] = failure is None
    if failure is not None:
        record["error"] = str(failure)
        raise failure
    return read_regular(paths[0], MAX_STREAM)


def environment() -> dict[str, str]:
    forbidden = {"RUSTC", "RUSTDOC", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                 "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CLIPPY_ARGS", "CARGO", "RUSTUP_TOOLCHAIN",
                 "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTDOC", "CARGO_TARGET_DIR",
                 "LD_PRELOAD", "LD_AUDIT", "GLIBC_TUNABLES", "RUST_TEST_THREADS", "RUST_TEST_NOCAPTURE",
                 "RUSTC_BOOTSTRAP", "CARGO_ENCODED_RUSTDOCFLAGS"}
    for name in os.environ:
        require(name not in forbidden and not name.startswith(("CARGO_TARGET_", "CARGO_PROFILE_", "CARGO_BUILD_", "DYLD_", "LD_"))
                and not (name.startswith("FE2O3_") and name != "FE2O3_TUTORIAL_CPU_OUTPUT_ROOT"),
                f"inherited input override: {name}")
    result = dict(os.environ)
    result.pop("FE2O3_TUTORIAL_CPU_OUTPUT_ROOT", None)
    result.update({"CARGO_NET_OFFLINE": "true", "CARGO_BUILD_JOBS": "1", "CARGO_INCREMENTAL": "0",
                   "RUSTUP_AUTO_INSTALL": "0"})
    return result


def require_child_success(phases: list[dict[str, Any]], label: str) -> None:
    if phases[-1]["returncode"] != 0:
        raise ObservationError(f"{label} failed; see retained process status/logs", "unavailable")


def toolchain_runtime(rustc: Path) -> tuple[Path, dict[str, Path]]:
    require(rustc.parent.name == "bin", "unexpected pinned rustc layout")
    directory = rustc.parent.parent / "lib"
    require(directory.resolve(strict=True) == directory and directory.is_dir(),
            "noncanonical pinned runtime directory")
    libraries = sorted(directory.glob("*.so*"))
    require(1 <= len(libraries) <= 64, "invalid pinned runtime library count")
    return directory, {"runtime:" + path.name: path for path in libraries}


def check_tools(tools: dict[str, Path], plan: dict[str, Any], message: str) -> None:
    directory, libraries = toolchain_runtime(tools["rustc"])
    require(str(directory) == plan["runtimeLibraryPath"]
            and libraries == {name: path for name, path in tools.items() if name.startswith("runtime:")},
            "pinned runtime library roster changed")
    for name, path in tools.items():
        require(digest(read_regular(path, MAX_INPUT_BYTES)) == plan["tools"][name]["sha256"], message)


def discover_tools(root: Path, env: dict[str, str], output: Path, deadline: float,
                   phases: list[dict[str, Any]], plan: dict[str, Any]) -> tuple[dict[str, Path], str]:
    toolchain = tomllib.loads(read_regular(root / "rust-toolchain.toml").decode("utf-8"))["toolchain"]["channel"]
    require(re.fullmatch(r"nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}", toolchain) is not None, "unsupported toolchain pin")
    rustup = shutil.which("rustup", path=env.get("PATH"))
    require(rustup is not None, "rustup unavailable")
    rustup_path = Path(rustup).resolve(strict=True)
    tools: dict[str, Path] = {}
    plan["tools"] = {}
    for name in ("cargo", "rustc"):
        data = child([str(rustup_path), "which", "--toolchain", toolchain, name], root, env,
                     output, "locate-" + name, deadline, phases)
        require_child_success(phases, "installed toolchain lookup")
        path = Path(data.decode("utf-8").strip())
        payload = read_regular(path, MAX_INPUT_BYTES)
        tools[name] = path
        plan["tools"][name] = {"path": str(path), "sha256": digest(payload)}
    require(tools["cargo"].parent == tools["rustc"].parent, "pinned toolchain directories differ")
    runtime, libraries = toolchain_runtime(tools["rustc"])
    for name, path in libraries.items():
        plan["tools"][name] = {"path": str(path), "sha256": digest(read_regular(path, MAX_INPUT_BYTES))}
    tools.update(libraries)
    plan["runtimeLibraryPath"] = str(runtime)
    # Caller LD_* overrides remain forbidden. Raw Cargo and its build-script
    # descendants need the same pinned runtime directory that rustup supplies.
    env["LD_LIBRARY_PATH"] = str(runtime)
    versions = child([str(tools["rustc"]), "-vV"], root, env, output, "rustc-version", deadline, phases)
    require_child_success(phases, "rustc version")
    hosts = [line[6:] for line in versions.decode("utf-8").splitlines() if line.startswith("host: ")]
    require(len(hosts) == 1, "missing pinned rustc host")
    require(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", hosts[0]) is not None, "invalid host triple")
    env["PATH"] = str(tools["cargo"].parent) + os.pathsep + env.get("PATH", "")
    return tools, hosts[0]


def bootstrap_driver(root: Path, env: dict[str, str], output: Path, target_dir: Path,
                     tools: dict[str, Path], driver_metadata: dict[str, Any], deadline: float,
                     phases: list[dict[str, Any]], plan: dict[str, Any]) -> Path:
    packages = [package for package in driver_metadata["packages"]
                if package["manifest_path"] == str(root / "crates/cargo-fe2o3/Cargo.toml")]
    require(len(packages) == 1, "driver package is not exact source")
    bootstrap_env = dict(env)
    bootstrap_env.update({"RUSTC": str(tools["rustc"]), "CARGO_BUILD_RUSTC": str(tools["rustc"]),
                          "RUSTC_WRAPPER": "", "CARGO_BUILD_RUSTC_WRAPPER": "",
                          "RUSTC_WORKSPACE_WRAPPER": "", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER": "",
                          "FE2O3_HIP_SYS_DISABLE": "1", "CARGO_BUILD_JOBS": "2"})
    plan["driverBuildJobs"] = 2
    data = child([str(tools["cargo"]), "build", "--locked", "--offline", "-p", "cargo-fe2o3",
                  "--bin", "cargo-fe2o3", "--target", plan["host"], "--message-format=json"], root, bootstrap_env,
                 output, "driver-build", deadline, phases)
    require_child_success(phases, "driver bootstrap")
    artifacts = []
    build_ok = 0
    for ordinal, line in enumerate(data.splitlines()):
        require(ordinal < MAX_EVENTS, "driver build event limit exceeded")
        event = decode_json(line)
        require(isinstance(event, dict), "driver event is not an object")
        if event.get("reason") == "build-finished":
            require(event.get("success") is True, "driver build failed")
            build_ok += 1
        if event.get("reason") == "compiler-artifact" and event.get("executable") is not None:
            require(event.get("package_id") == packages[0]["id"]
                    and event.get("target", {}).get("name") == "cargo-fe2o3"
                    and event["target"].get("kind") == ["bin"]
                    and event["target"].get("crate_types") == ["bin"]
                    and event["target"].get("src_path") == str(root / "crates/cargo-fe2o3/src/main.rs")
                    and event.get("profile", {}).get("test") is False
                    and event["profile"].get("opt_level") == "0", "wrong bootstrap artifact")
            artifacts.append(Path(event["executable"]))
    require(build_ok == 1 and len(artifacts) == 1, "driver bootstrap receipt is incomplete")
    built = artifacts[0]
    require(built.is_relative_to(target_dir), "driver escaped private build directory")
    driver_bytes = read_regular(built, MAX_INPUT_BYTES)
    driver_dir = output / "driver"
    driver_dir.mkdir(mode=0o700)
    driver = driver_dir / "cargo-fe2o3"
    with driver.open("xb") as stream:
        stream.write(driver_bytes)
    driver.chmod(0o500)
    driver_dir.chmod(0o500)
    plan["driverSha256"] = digest(driver_bytes)
    return driver


def execute(root: Path, arguments: list[str], output: Path, started: float,
            observation: dict[str, Any]) -> None:
    plan, validator = load_plan(root, arguments)
    observation["plan"] = plan
    deadline = started + plan["declaredCommand"]["timeoutSeconds"]
    env = environment()
    config_paths = configuration_candidates(root, env)
    configs = configuration_snapshot(config_paths, deadline)
    observation["configurationInputs"] = configs
    target_dir = output / "build"
    target_dir.mkdir(mode=0o700)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["TMPDIR"] = str(output)
    phases = observation["phases"]
    tools, host = discover_tools(root, env, output, deadline, phases, plan)
    manifest = relative_file(root, arguments[0])
    configuration = plan["hostConfiguration"]
    features = feature_arguments(configuration)
    metadata_records = []
    for label, path in (("driver", root / "Cargo.toml"), ("suite", manifest)):
        data = child([str(tools["cargo"]), "metadata", "--locked", "--offline", "--format-version", "1",
                      "--manifest-path", str(path), "--filter-platform", host,
                      *(features if label == "suite" else [])],
                     root, env, output, label + "-metadata", deadline, phases)
        require_child_success(phases, label + " metadata")
        metadata_records.append(decode_json(data))
    driver_metadata, suite_metadata = metadata_records
    expected = selected_target(root, suite_metadata, manifest, configuration["selector"])
    plan["selectedTarget"] = expected
    plan["host"] = host
    check_tools(tools, plan, "tool changed before baseline")
    local_roots = sorted({Path(package["manifest_path"]).parent for metadata in metadata_records
                          for package in metadata["packages"] if package.get("source") is None})
    require(all(path.is_relative_to(root) for path in local_roots), "local dependency escapes repository")
    files = [root / "config/tutorial-kernel-manifest-v1.json", root / "rust-toolchain.toml",
             root / "scripts/validate-tutorial-kernel-manifest.py", root / WRAPPER,
             root / "scripts/tutorial_cpu_reference.py", root / "Cargo.toml", root / "Cargo.lock"]
    files.append(validator.effective_cargo_lock(root, manifest.parent, "CPU reference"))
    files.extend(Path(path) for path, value in configs.items() if value is not None)
    require(configuration_snapshot(config_paths, deadline) == configs, "configuration drift during metadata")
    before = footprint(local_roots, files, deadline)
    revalidate_plan(root, arguments, plan, validator)
    require(configuration_snapshot(config_paths, deadline) == configs
            and footprint(local_roots, files, deadline) == before, "input drift during baseline validation")
    observation["inputs"] = before
    driver = bootstrap_driver(root, env, output, target_dir, tools, driver_metadata,
                              deadline, phases, plan)
    selected = configuration["selector"]
    selector = ["--lib"] if selected == ["lib"] else ["--test", selected[1]]
    command = [str(driver), "test", "--locked", "--offline", "--manifest-path", str(manifest),
               *features, *selector, "--message-format=json", "--", "-Z", "unstable-options", "--format=json", "--test-threads=1"]
    error: BaseException | None = None
    try:
        data = child(command, root, env, output, "suite", deadline, phases)
        observation["tests"] = test_observation(data, expected, target_dir)
        observation["outcome"] = observation["tests"]["outcome"] if phases[-1]["returncode"] == 0 else "failed"
    except BaseException as caught:
        observation["errors"].append(str(caught))
        error = caught
    finally:
        observation["configurationInputsUnchanged"] = configuration_snapshot(config_paths, deadline) == configs
        require(observation["configurationInputsUnchanged"], "configuration input drift")
        after = footprint(local_roots, files, deadline)
        observation["inputsUnchanged"] = before == after
        require(before == after, "source/config input drift")
        revalidate_plan(root, arguments, plan, validator)
        require(configuration_snapshot(config_paths, deadline) == configs
                and footprint(local_roots, files, deadline) == before, "input drift during postflight validation")
        require(digest(read_regular(driver, MAX_INPUT_BYTES)) == plan["driverSha256"], "driver custody changed")
        check_tools(tools, plan, "tool changed")
    if error is not None:
        raise error


def arm_deadline(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ObservationError("batch execution deadline exceeded", "unavailable")
    signal.setitimer(signal.ITIMER_REAL, remaining)


def suite_deadline(started: float, common: float, limit: int, overall: float) -> float:
    if common >= limit:
        raise ObservationError("common setup exhausted declared suite budget", "unavailable")
    return min(overall, started + limit - common)


def remove_scratch(directory: Path, observation: dict[str, Any]) -> None:
    for name in ("build", "driver"):
        path = directory / name
        if path.exists() or path.is_symlink():
            try:
                require(not path.is_symlink() and path.resolve(strict=True) == path and path.is_dir(),
                        "scratch directory identity changed")
                path.chmod(0o700)
                shutil.rmtree(path)
            except (OSError, ObservationError) as error:
                observation["errors"].append(f"scratch cleanup: {error}")
                observation["outcome"] = "invalid"


def save_observation(directory: Path, observation: dict[str, Any]) -> dict[str, str]:
    path = directory / "observation.json"
    payload = (json.dumps(observation, sort_keys=True, indent=2) + "\n").encode("utf-8")
    require(len(payload) <= MAX_INPUT_BYTES, "observation exceeds bound")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"path": str(path), "sha256": digest(payload)}


def execute_batch(root: Path, output: Path, started: float, observation: dict[str, Any]) -> None:
    plans, validator = load_batch_plans(root)
    limits = [plan["declaredCommand"]["timeoutSeconds"] for plan in plans]
    overall = started + sum(limits)
    deadline = min(overall, started + min(limits))
    observation.update(executionMode="batch-shared-private-driver", suites=[],
                       declaredSuiteCount=len(plans), schedulerLimitSeconds=sum(limits))
    common_dir = output / "common"
    common_dir.mkdir(mode=0o700)
    common_observation: dict[str, Any] = {"schema": "fe2o3-tutorial-cpu-reference-common-observation-v1",
        **NO_AUTHORITY, "outcome": "invalid", "phases": [], "errors": [], "plan": {}}
    common_reference: dict[str, str] | None = None
    common_seconds = 0.0
    abort: BaseException | None = None
    final_shared_check = None
    try:
        arm_deadline(deadline)
        base_env = environment()
        env = dict(base_env)
        config_paths = configuration_candidates(root, base_env)
        configs = configuration_snapshot(config_paths, deadline)
        common_observation["configurationInputs"] = configs
        files = [root / "config/tutorial-kernel-manifest-v1.json", root / "rust-toolchain.toml",
                 root / "scripts/validate-tutorial-kernel-manifest.py", root / WRAPPER,
                 root / "scripts/tutorial_cpu_reference.py", root / "Cargo.toml", root / "Cargo.lock"]
        files.extend(Path(path) for path, value in configs.items() if value is not None)
        initial_files = footprint([], files, deadline)
        target_dir = common_dir / "build"
        target_dir.mkdir(mode=0o700)
        env.update(CARGO_TARGET_DIR=str(target_dir), TMPDIR=str(common_dir))
        shared_plan = common_observation["plan"]
        tools, host = discover_tools(root, env, common_dir, deadline, common_observation["phases"], shared_plan)
        shared_plan["host"] = host
        data = child([str(tools["cargo"]), "metadata", "--locked", "--offline", "--format-version", "1",
                      "--manifest-path", str(root / "Cargo.toml"), "--filter-platform", host],
                     root, env, common_dir, "driver-metadata", deadline, common_observation["phases"])
        require_child_success(common_observation["phases"], "driver metadata")
        driver_metadata = decode_json(data)
        common_roots = sorted({Path(package["manifest_path"]).parent for package in driver_metadata["packages"]
                               if package.get("source") is None})
        require(all(path.is_relative_to(root) for path in common_roots), "local dependency escapes repository")

        def check_configuration(until: float) -> None:
            require(environment() == base_env, "batch environment changed")
            require(configuration_candidates(root, base_env) == config_paths,
                    "configuration candidate paths changed")
            require(configuration_snapshot(config_paths, until) == configs, "configuration input drift")
            check_tools(tools, shared_plan, "tool changed")

        def check_plans() -> None:
            # One full fresh validator pass covers all fixture contracts. The
            # other selections are compared against those same original bytes.
            first = plans[0]
            revalidate_plan(root, first["declaredCommand"]["arguments"], first, validator)
            payload = read_regular(root / "config/tutorial-kernel-manifest-v1.json", MAX_MANIFEST)
            require(digest(payload) == first["manifestSha256"], "original tutorial manifest changed")
            checked = batch_plans(decode_json(payload))
            require(len(checked) == len(plans) and all(
                all(plan[key] == value for key, value in item.items()) for plan, item in zip(plans, checked)),
                "batch contract roster changed")

        check_configuration(deadline)
        require(footprint([], files, deadline) == initial_files, "common inputs changed during discovery/metadata")
        common_before = footprint(common_roots, files, deadline)
        check_plans()
        check_configuration(deadline)
        require(footprint(common_roots, files, deadline) == common_before, "input drift during batch baseline")
        common_observation["inputs"] = common_before
        driver = bootstrap_driver(root, env, common_dir, target_dir, tools, driver_metadata,
                                  deadline, common_observation["phases"], shared_plan)

        def check_shared(until: float) -> None:
            arm_deadline(until)
            check_configuration(until)
            require(footprint(common_roots, files, until) == common_before, "shared source/config input drift")
            check_plans()
            check_configuration(until)
            require(footprint(common_roots, files, until) == common_before, "shared input drift during validation")
            require(digest(read_regular(driver, MAX_INPUT_BYTES)) == shared_plan["driverSha256"],
                    "driver custody changed")
            require(time.monotonic() < until, "shared postflight deadline exceeded")

        final_shared_check = check_shared
        check_shared(deadline)
        common_observation.update(outcome="passed", inputsUnchanged=True, configurationInputsUnchanged=True,
                                  evidenceScope="common-setup-and-post-bootstrap-custody-only")
        common_reference = save_observation(common_dir, common_observation)
        common_seconds = time.monotonic() - started
        observation["common"] = common_reference
        observation["commonElapsedSeconds"] = common_seconds
        for current, original_plan in enumerate(plans):
            own_started = time.monotonic()
            suite_dir = output / f"suite-{current:02d}"
            suite_dir.mkdir(mode=0o700)
            plan = dict(original_plan)
            plan.update(tools=shared_plan["tools"], host=host, driverSha256=shared_plan["driverSha256"],
                        runtimeLibraryPath=shared_plan["runtimeLibraryPath"])
            record: dict[str, Any] = {"schema": "fe2o3-tutorial-cpu-reference-observation-v1",
                **NO_AUTHORITY, "executionMode": "batch-shared-private-driver", "plan": plan,
                "commonObservation": common_reference, "commonElapsedSeconds": common_seconds,
                "outcome": "invalid", "phases": [], "errors": [], "attempted": False}
            suite_error: BaseException | None = None
            try:
                until = suite_deadline(own_started, common_seconds, limits[current], overall)
                check_shared(until)
                suite_target = suite_dir / "build"
                suite_target.mkdir(mode=0o700)
                suite_env = dict(env)
                suite_env.update(CARGO_TARGET_DIR=str(suite_target), TMPDIR=str(suite_dir))
                arguments = plan["declaredCommand"]["arguments"]
                manifest = relative_file(root, arguments[0])
                configuration = plan["hostConfiguration"]
                features = feature_arguments(configuration)
                record["attempted"] = True
                data = child([str(tools["cargo"]), "metadata", "--locked", "--offline", "--format-version", "1",
                              "--manifest-path", str(manifest), "--filter-platform", host, *features],
                             root, suite_env, suite_dir, "suite-metadata", until, record["phases"])
                require_child_success(record["phases"], "suite metadata")
                metadata = decode_json(data)
                expected = selected_target(root, metadata, manifest, configuration["selector"])
                plan["selectedTarget"] = expected
                local_roots = sorted({Path(package["manifest_path"]).parent for package in metadata["packages"]
                                      if package.get("source") is None})
                require(all(path.is_relative_to(root) for path in local_roots), "local dependency escapes repository")
                suite_files = [*files, validator.effective_cargo_lock(root, manifest.parent, "CPU reference")]
                check_shared(until)
                before = footprint(local_roots, suite_files, until)
                require(all(common_before[path] == value for path, value in before.items() if path in common_before),
                        "suite baseline disagrees with common inputs")
                revalidate_plan(root, arguments, plan, validator)
                check_configuration(until)
                require(footprint(local_roots, suite_files, until) == before, "input drift during baseline validation")
                record.update(inputs=before, configurationInputs=configs)
                selected = configuration["selector"]
                selector = ["--lib"] if selected == ["lib"] else ["--test", selected[1]]
                command = [str(driver), "test", "--locked", "--offline", "--manifest-path", str(manifest),
                           *features, *selector, "--message-format=json", "--", "-Z", "unstable-options", "--format=json", "--test-threads=1"]
                try:
                    data = child(command, root, suite_env, suite_dir, "suite", until, record["phases"])
                    record["tests"] = test_observation(data, expected, suite_target)
                    record["outcome"] = record["tests"]["outcome"] if record["phases"][-1]["returncode"] == 0 else "failed"
                except BaseException as error:
                    record["errors"].append(str(error))
                    raise
                finally:
                    check_configuration(until)
                    record["configurationInputsUnchanged"] = True
                    record["inputsUnchanged"] = footprint(local_roots, suite_files, until) == before
                    require(record["inputsUnchanged"], "source/config input drift")
                    revalidate_plan(root, arguments, plan, validator)
                    check_configuration(until)
                    require(footprint(local_roots, suite_files, until) == before, "input drift during postflight validation")
                    check_shared(until)
            except BaseException as error:
                suite_error = error
                record["outcome"] = (error.outcome if isinstance(error, ObservationError) else
                                     "interrupted" if isinstance(error, KeyboardInterrupt) else "invalid")
                if str(error) not in record["errors"]:
                    record["errors"].append(str(error) or "interrupted")
            finally:
                # Cleanup still runs after an expired/interrupting alarm. It
                # cannot turn an exhausted suite into successful evidence.
                signal.setitimer(signal.ITIMER_REAL, 0)
                remove_scratch(suite_dir, record)
                own_seconds = time.monotonic() - own_started
                record.update(ownElapsedSeconds=own_seconds, chargedElapsedSeconds=common_seconds + own_seconds,
                              evidenceDirectory=str(suite_dir))
                if common_seconds + own_seconds >= limits[current]:
                    record["outcome"] = "unavailable"
                    record["errors"].append("charged suite deadline exhausted")
                if record["errors"] and suite_error is None:
                    suite_error = ObservationError("suite cleanup or deadline failed")
                reference = save_observation(suite_dir, record)
                observation["suites"].append({"suiteId": plan["suiteId"], "outcome": record["outcome"],
                                              "observation": reference})
            if suite_error is not None:
                raise suite_error
        observation["outcome"] = "passed" if all(item["outcome"] == "passed" for item in observation["suites"]) else "failed"
    except BaseException as error:
        abort = error
        observation["outcome"] = (error.outcome if isinstance(error, ObservationError) else
                                  "interrupted" if isinstance(error, KeyboardInterrupt) else "invalid")
        observation["errors"].append(str(error) or "interrupted")
    finally:
        try:
            if final_shared_check is not None:
                try:
                    final_shared_check(overall)
                    observation["finalSharedInputsUnchanged"] = True
                except BaseException as error:
                    observation["finalSharedInputsUnchanged"] = False
                    observation["outcome"] = "invalid"
                    observation["errors"].append(f"final shared postflight: {error}")
            signal.setitimer(signal.ITIMER_REAL, 0)
            if common_reference is None:
                common_observation["errors"].append(str(abort) if abort is not None else "common setup incomplete")
                common_seconds = time.monotonic() - started
                common_observation["elapsedSeconds"] = common_seconds
                observation["common"] = save_observation(common_dir, common_observation)
                observation["commonElapsedSeconds"] = common_seconds
            for original_plan in plans[len(observation["suites"]):]:
                index = len(observation["suites"])
                suite_dir = output / f"suite-{index:02d}"
                suite_dir.mkdir(mode=0o700, exist_ok=True)
                record = {"schema": "fe2o3-tutorial-cpu-reference-observation-v1", **NO_AUTHORITY,
                          "executionMode": "batch-shared-private-driver", "outcome": "unavailable", "attempted": False,
                          "plan": original_plan, "commonObservation": observation["common"], "phases": [],
                          "commonElapsedSeconds": common_seconds, "ownElapsedSeconds": 0.0,
                          "chargedElapsedSeconds": common_seconds,
                          "errors": ["not started: batch setup/reuse aborted"]}
                reference = save_observation(suite_dir, record)
                observation["suites"].append({"suiteId": original_plan["suiteId"], "outcome": "unavailable",
                                              "observation": reference})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            remove_scratch(common_dir, observation)
            if time.monotonic() >= overall:
                observation["errors"].append("batch scheduler deadline exhausted")
                observation["outcome"] = "unavailable"


def main(arguments: list[str] | None = None) -> int:
    started = time.monotonic()
    interrupted = False

    def interrupt(_number: int, _frame: Any) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    def expired(_number: int, _frame: Any) -> None:
        raise ObservationError("adapter deadline exceeded")

    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    root = Path(__file__).resolve().parents[1]
    observation: dict[str, Any] = {"schema": "fe2o3-tutorial-cpu-reference-observation-v1",
                                  **NO_AUTHORITY, "outcome": "invalid", "phases": [], "errors": []}
    output: Path | None = None
    try:
        parent = Path(os.environ.get("FE2O3_TUTORIAL_CPU_OUTPUT_ROOT", tempfile.gettempdir()))
        require(parent.is_absolute() and parent.resolve(strict=True) == parent and parent.is_dir(),
                "output parent must be an existing canonical absolute directory")
        if "FE2O3_TUTORIAL_CPU_OUTPUT_ROOT" in os.environ:
            info = parent.stat()
            require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
                    "explicit output parent must be owner-private")
        require(not parent.is_relative_to(root), "output directory must be outside the source checkout")
        output = Path(tempfile.mkdtemp(prefix="fe2o3-tutorial-cpu-", dir=parent))
        output.chmod(0o700)
        selected = list(sys.argv[1:] if arguments is None else arguments)
        if selected and selected[0] == "--batch":
            observation["schema"] = "fe2o3-tutorial-cpu-reference-batch-observation-v1"
            require(selected == ["--batch"], "--batch accepts no trailing arguments")
            execute_batch(root, output, started, observation)
        else:
            execute(root, selected, output, started, observation)
    except (ObservationError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        observation["outcome"] = error.outcome if isinstance(error, ObservationError) else "invalid"
        if str(error) not in observation["errors"]:
            observation["errors"].append(str(error))
    except KeyboardInterrupt:
        observation["outcome"] = "interrupted"
        observation["errors"].append("interrupted")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if output is not None:
            for name in ("build", "driver"):
                path = output / name
                if path.exists():
                    try:
                        require(not path.is_symlink() and path.resolve(strict=True) == path and path.is_dir(),
                                "scratch directory identity changed")
                        path.chmod(0o700)
                        shutil.rmtree(path)
                    except (OSError, ObservationError) as error:
                        observation["errors"].append(f"scratch cleanup: {error}")
                        observation["outcome"] = "invalid"
            observation["elapsedSeconds"] = time.monotonic() - started
            observation["evidenceDirectory"] = str(output)
            (output / "observation.json").write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(json.dumps(observation, sort_keys=True))
    return 0 if observation["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate expected ordinary-source corpus contracts, never qualification receipts.

The source scanner is the recovered donor inventory scanner, not rustc semantic
analysis. Feature reachability, observed compiler inputs, optimized-graph policy
checks, simulation and hardware remain separate qualification obligations.

The optional curriculum extension binds every displayed lesson/tab to an
immutable site snapshot and records pending SIMT/tile/mixed and source-item
obligations. It references existing source entries instead of cloning fixtures.
Consumers must use --require-curriculum; --site-inventory additionally checks
the current runtime projection without requiring an unchanged site Git HEAD.
Neither option upgrades a pending obligation to execution or launch evidence.

--emit-kernel-pairs reports existing input selections and exact source cases,
not completed pairs. The optional kernelInventory extension reconciles physical
display occurrences and exact source selections without inferring implementations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import importlib.util
import math
import json
import os
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Any


TOP_LEVEL_KEYS = {
    "schema",
    "roadmapIssue",
    "baseline",
    "recovery",
    "productionContract",
    "qualification",
    "compilerFixtures",
    "entries",
}
CURRICULUM_SCHEMA = "fe2o3-tutorial-curriculum-obligations-v1"
CURRICULUM_SCHEMA_V2 = "fe2o3-tutorial-curriculum-obligations-v2"
SITE_INVENTORY_SCHEMA = "fe2o3-tutorial-runtime-projection-v1"
CURRICULUM_KEYS = {"schema", "site", "status", "lessons"}
CURRICULUM_LESSON_KEYS = {
    "lessonId", "role", "roleReason", "sourceEntryIds", "sourceBindingGap",
    "variants", "codeTabs",
}
CURRICULUM_TAB_FIELDS = (
    "ordinal", "kind", "label", "language", "displayedUtf8Bytes", "displayedSha256",
    "sourcePath", "sourceCommit", "sourceSha256", "sourceDigestScope",
    "sourceFragmentsSha256", "explanatory", "evidenceId",
)
CURRICULUM_TAB_KEYS = set(CURRICULUM_TAB_FIELDS) | {"sourceItem", "sourceItemStatus"}
MAX_SITE_INVENTORY_BYTES = 16 * 1024 * 1024
MAX_CURRICULUM_TABS = 1024
MAX_KERNEL_PAIR_RECORDS = 4096
ENTRY_KEYS = {
    "lessonId",
    "siteEvidenceKind",
    "classification",
    "packageManifest",
    "sourcePaths",
    "compilerFixtureIds",
    "requiredGates",
}
FIXTURE_KEYS = {
    "fixtureId",
    "testId",
    "testPath",
    "target",
    "matrix",
    "compilerInput",
    "simulation",
}
SIMULATION_KEYS = {
    "requestPath",
    "requestSha256",
    "requestBytes",
    "expectationPath",
    "expectationSha256",
    "expectationBytes",
    "bundleVersion",
    "canonicalKirVersion",
    "status",
}
COMPILER_INPUT_KEYS = {
    "packageManifest",
    "packageManifestSha256",
    "cargoLockPath",
    "cargoLockSha256",
    "sourcePaths",
    "sourceClosureSha256",
    "cargoTarget",
    "defaultFeatures",
    "features",
    "kernelSymbols",
    "contractSha256",
}
CARGO_TARGET_KEYS = {"kind", "name", "sourcePath"}
MATRIX_KEYS = {
    "caseId",
    "runnerPath",
    "artifactName",
    "runnerArguments",
    "environment",
}
QUALIFICATION_KEYS = {"status", "evidenceSchemaPath", "hardwareTargets", "suites"}
HARDWARE_TARGET_KEYS = {
    "target",
    "deterministicSemanticRunnerAvailability",
    "reason",
}
SUITE_KEYS = {
    "suiteId",
    "gate",
    "availability",
    "command",
    "unavailableReason",
    "coverage",
}
SUITE_COMMAND_KEYS = {
    "executable",
    "arguments",
    "environment",
    "workingDirectory",
    "timeoutSeconds",
}
SUITE_COVERAGE_KEYS = {"lessonId", "fixtureIds"}
COMPILER_EVIDENCE = {
    "runnable-now",
    "compiler-checked",
    "compiler-hsaco-observed",
    "gpu-observed",
}
ALLOWED_EVIDENCE = COMPILER_EVIDENCE | {
    "source-example",
    "source-tested",
    "source-model-verified",
    "design-only",
}
ALLOWED_TARGETS = {"gfx942", "gfx950"}
ALLOWED_GATES = {
    "production-compile",
    "semantic-simulation",
    "cpu-reference",
    "hardware",
}
MAX_SUITE_COMMAND_ARGUMENTS = 24
MAX_SUITE_COMMAND_ENVIRONMENT = 16
MAX_ATTRIBUTED_SOURCE_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_SOURCE_FILES = 4096
MAX_PACKAGE_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CARGO_MANIFEST_BYTES = 1024 * 1024
MAX_CARGO_LOCK_BYTES = 8 * 1024 * 1024
MAX_SIMULATION_DOCUMENT_BYTES = 16 * 1024 * 1024
IGNORED_PACKAGE_DIRECTORIES = {"target"}
CORPUS_CONTRACT_DIGEST_DOMAIN = b"fe2o3-tutorial-kernel-corpus-contract-v1\0"
SOURCE_CLOSURE_DIGEST_DOMAIN = b"fe2o3-tutorial-package-rust-source-closure-v1\0"
FIXTURE_INPUT_DIGEST_DOMAIN = b"fe2o3-tutorial-fixture-compiler-input-v1\0"
SOURCE_ITEM_DIGEST_DOMAIN = b"fe2o3-tutorial-displayed-source-contract-v1\0"
ATTRIBUTE_START = re.compile(r"#[ \t]*\[")
MACRO_RULES_START = re.compile(r"\bmacro_rules\s*!")
SOURCE_INCLUDE_START = re.compile(r"\binclude\s*!")
RUST_DECLARATION = re.compile(
    r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:(?:const|async|unsafe)\s+)*(?:extern\s+(?:\"[^\"]*\"\s+)?)?fn\b"
)
RUST_FUNCTION_NAME = re.compile(
    r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:(?:const|async|unsafe)\s+)*(?:extern\s+(?:\"[^\"]*\"\s+)?)?"
    r"fn\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
RUST_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TARGET_SECTIONS = ("lib", "bin", "example", "test", "bench")


def fail(message: str) -> None:
    raise SystemExit(f"tutorial kernel manifest: {message}")


def validate_kernel_inventory(
    manifest: dict[str, Any], inventory: dict[str, Any] | None,
    *, max_records: int = MAX_KERNEL_PAIR_RECORDS,
    repo_root: Path | None = None, package_cache: dict[str, dict[str, Any]] | None = None,
    observe_sources: Any = None,
) -> dict[str, Any] | None:
    if "kernelInventory" not in manifest:
        return None
    path = Path(__file__).with_name("tutorial_kernel_identities.py")
    specification = importlib.util.spec_from_file_location("tutorial_kernel_identities", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    root = Path(__file__).resolve().parents[1] if repo_root is None else repo_root
    cache = {} if package_cache is None else package_cache

    def physical_sources(inputs: dict[str, Any], checked: dict[str, Any]) -> tuple[str, dict[str, str], list[str]]:
        package = cache[inputs["packageManifest"]]
        cargo = package["cargo"]
        if (cargo["package"].get("build") not in (None, False)
                or (cargo["package"].get("build") is not False and (package["packageRoot"] / "build.rs").exists())
                or any(member not in cargo.get("features", {})
                       for feature in checked["enabledFeatures"]
                       for member in cargo.get("features", {})[feature])):
            fail("direct fixture binding requires local features without build-script cfg")
        sources = {}
        for path, source in package["packageSources"]:
            if sha256_file(path, MAX_ATTRIBUTED_SOURCE_BYTES, "direct fixture source") != hashlib.sha256(source.encode("utf-8")).hexdigest():
                fail("direct fixture physical source differs from its validated closure")
            sources[path.relative_to(root).as_posix()] = source
        library = (PurePosixPath(inputs["packageManifest"]).parent / inputs["cargoTarget"]["sourcePath"]).as_posix()
        return library, sources, checked["enabledFeatures"]

    def fixture_sources(fixture: dict[str, Any]) -> tuple[str, dict[str, str], list[str]]:
        checked = validate_compiler_input(root, fixture, "direct fixture binding", cache)
        return physical_sources(fixture["compilerInput"], checked)

    def source_case_sources(lesson: str, tab: dict[str, Any], case: dict[str, Any]) -> tuple[str, dict[str, str], list[str]]:
        inputs = {**tab["sourceItem"]["compilerInput"], "features": case["features"],
                  "kernelSymbols": [case["kernelSymbol"]]}
        checked = validate_compiler_input_data(
            root, inputs, f"{lesson} variant source case binding", cache, feature_scoped_includes=True)
        return physical_sources(inputs, checked)

    def rust_syntax(source: str) -> tuple[str, dict[int, int]]:
        code = _rust_code_without_comments_and_literals(source)
        return code, _rust_delimiters(code)

    try:
        return module.validate_kernel_inventory(
            manifest, inventory, ordinary_rust_function_items, max_records=max_records,
            load_fixture_sources=fixture_sources, rust_syntax=rust_syntax,
            load_source_case_sources=source_case_sources,
            observe_sources=observe_sources,
        )
    except module.KernelInventoryError as error:
        fail(str(error))


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        fail(
            f"{label} keys differ: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a nonempty string")
    return value


def require_target(value: Any, label: str) -> str:
    target = require_string(value, label)
    if target not in ALLOWED_TARGETS:
        fail(f"{label} is unsupported: {target}")
    return target


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        fail(f"{label} must be a nonempty array")
    result = [require_string(item, f"{label}[]") for item in value]
    if len(result) != len(set(result)):
        fail(f"{label} must be duplicate-free")
    return result


def require_string_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    result = [require_string(item, f"{label}[]") for item in value]
    if len(result) != len(set(result)):
        fail(f"{label} must be duplicate-free")
    return result


def checked_path(repo_root: Path, value: Any, label: str) -> str:
    text = require_string(value, label)
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != text:
        fail(f"{label} is not a canonical repository-relative path: {text!r}")
    path = repo_root.joinpath(*relative.parts)
    if (
        any(
            repo_root.joinpath(*relative.parts[:index]).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        )
        or not path.is_file()
    ):
        fail(f"{label} does not name a regular repository input: {text}")
    return text


def checked_directory(repo_root: Path, value: Any, label: str) -> str:
    text = require_string(value, label)
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != text:
        fail(f"{label} is not a canonical repository-relative directory: {text!r}")
    path = repo_root.joinpath(*relative.parts)
    if (
        any(
            repo_root.joinpath(*relative.parts[:index]).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        )
        or not path.is_dir()
    ):
        fail(f"{label} does not name a regular repository directory: {text}")
    return text


def _rust_code_without_comments_and_literals(source: str) -> str:
    """Preserve source coordinates while removing non-code Rust text."""
    output = list(source)
    index = 0
    block_depth = 0
    character_literal = re.compile(r"'(?:\\.|[^'\\\n])'")
    raw_literal = re.compile(r"(?:br|cr|r)(?P<hashes>#{0,255})\"")
    while index < len(source):
        if block_depth:
            if source.startswith("/*", index):
                output[index : index + 2] = "  "
                block_depth += 1
                index += 2
            elif source.startswith("*/", index):
                output[index : index + 2] = "  "
                block_depth -= 1
                index += 2
            else:
                if source[index] != "\n":
                    output[index] = " "
                index += 1
            continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = len(source) if end < 0 else end
            output[index:end] = " " * (end - index)
            index = end
            continue
        if source.startswith("/*", index):
            output[index : index + 2] = "  "
            block_depth = 1
            index += 2
            continue

        character = character_literal.match(source, index)
        if character is not None:
            end = character.end()
            output[index:end] = " " * (end - index)
            index = end
            continue

        raw = raw_literal.match(source, index)
        if raw is not None:
            delimiter = '"' + raw.group("hashes")
            end = source.find(delimiter, raw.end())
            if end < 0:
                fail("Rust source contains an unterminated raw string literal")
            end += len(delimiter)
            for offset in range(index, end):
                if source[offset] != "\n":
                    output[offset] = " "
            index = end
            continue

        prefix = 1 if source.startswith(('b"', 'c"'), index) else 0
        if source[index + prefix : index + prefix + 1] == '"':
            cursor = index + prefix + 1
            escaped = False
            closed = False
            while cursor < len(source):
                char = source[cursor]
                cursor += 1
                if char == '"' and not escaped:
                    closed = True
                    break
                escaped = char == "\\" and not escaped
                if char != "\\":
                    escaped = False
            if not closed:
                fail("Rust source contains an unterminated string literal")
            for offset in range(index, cursor):
                if source[offset] != "\n":
                    output[offset] = " "
            index = cursor
            continue
        index += 1
    if block_depth:
        fail("Rust source contains an unterminated block comment")
    return "".join(output)


def _rust_delimiters(code: str) -> dict[int, int]:
    """Return balanced delimiter pairs for code with literals/comments masked."""
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[tuple[str, int]] = []
    pairs: dict[int, int] = {}
    for index, char in enumerate(code):
        if char in opening:
            stack.append((char, index))
        elif char in closing:
            if not stack or stack[-1][0] != closing[char]:
                fail(f"Rust source contains an unmatched delimiter at byte {index}")
            _, start = stack.pop()
            pairs[start] = index + 1
    if stack:
        fail(f"Rust source contains an unterminated {stack[-1][0]} delimiter")
    return pairs


def _rust_attribute(code: str, start: int, pairs: dict[int, int]) -> tuple[int, str]:
    opening = code.find("[", start)
    if opening < 0 or opening not in pairs:
        fail("Rust source contains an unterminated attribute")
    closing = pairs[opening]
    return closing, code[opening + 1 : closing - 1]


def _attribute_has_kernel_name(body: str) -> bool:
    name = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\b", body)
    if name is None:
        return False
    if name.group(1) == "kernel":
        return True
    if name.group(1) != "cfg_attr":
        return False
    # Strings and comments have already been masked. Any kernel token in a
    # cfg_attr is conservatively treated as active because features are not
    # part of this manifest's lexical contract.
    return re.search(r"\bkernel\b", body[name.end() :]) is not None


def _macro_rule_bodies(code: str, pairs: dict[int, int]) -> list[tuple[int, int]]:
    bodies = []
    for match in MACRO_RULES_START.finditer(code):
        cursor = match.end()
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        # macro_rules! name { ... }; has one identifier before its body.
        identifier = RUST_IDENTIFIER.match(code, cursor)
        if identifier is not None:
            cursor = identifier.end()
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        if cursor >= len(code) or code[cursor] not in "([{" or cursor not in pairs:
            fail("Rust source contains an unterminated macro_rules body")
        bodies.append((cursor, pairs[cursor]))
    return bodies


def _attribute_has_qualified_kernel_name(body: str) -> bool:
    cursor = 0
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if body.startswith("::", cursor):
        cursor += 2
    while True:
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        identifier = _rust_function_identifier(body, cursor)
        if identifier is None:
            return False
        cursor, name = identifier
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if not body.startswith("::", cursor):
            break
        cursor += 2
    return name == "kernel" or (
        name == "cfg_attr" and re.search(r"\bkernel\b", body[cursor:]) is not None
    )


def _rust_function_identifier(code: str, start: int) -> tuple[int, str] | None:
    """Read a Rust identifier using Python's Unicode XID character predicates."""
    cursor = start + 2 if code.startswith("r#", start) else start
    first = cursor
    if cursor >= len(code) or not code[cursor].isidentifier():
        return None
    cursor += 1
    while cursor < len(code) and ("a" + code[cursor]).isidentifier():
        cursor += 1
    return cursor, code[first:cursor]


def ordinary_rust_function_items(source: str) -> list[dict[str, Any]]:
    """Return physical fn-name occurrences, not expanded or executable kernels.

    Offsets refer to the original UTF-8 function-name token, including r# for
    raw identifiers; kernelSymbol omits that lexical prefix. Attributes are
    classified conservatively without evaluating cfg. Historical name-list
    helpers intentionally retain their existing behavior.
    """
    if not isinstance(source, str) or len(source) > MAX_ATTRIBUTED_SOURCE_BYTES:
        fail("Rust source exceeds the function occurrence source bound")
    utf8_bytes = 0
    for character in source:
        value = ord(character)
        if 0xD800 <= value <= 0xDFFF:
            fail("Rust source contains invalid Unicode")
        utf8_bytes += 1 if value < 0x80 else 2 if value < 0x800 else 3 if value < 0x10000 else 4
        if utf8_bytes > MAX_ATTRIBUTED_SOURCE_BYTES:
            fail("Rust source exceeds the function occurrence source bound")
    code = _rust_code_without_comments_and_literals(source)
    pairs = _rust_delimiters(code)
    macros = _macro_rule_bodies(code, pairs)
    events = re.compile(r"(?P<attribute>#\s*!?\s*\[)|(?<![\w#])fn(?=\s)")
    following_attribute = re.compile(r"\s*#\s*\[")
    declaration = re.compile(
        r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
        r"(?:(?:const|async|unsafe|safe|default)\s+)*(?:extern\s+)?fn\s+"
    )
    records: list[dict[str, Any]] = []
    cursor = macro_index = previous_character = previous_byte = 0

    def append(name_start: int, name: str, attributed: bool) -> None:
        nonlocal previous_character, previous_byte
        if len(records) >= MAX_KERNEL_PAIR_RECORDS:
            fail("Rust source exceeds the function occurrence count bound")
        # These disjoint slices encode each prefix byte at most once.
        previous_byte += len(source[previous_character:name_start].encode("utf-8"))
        previous_character = name_start
        records.append({
            "kernelSymbol": name,
            "functionUtf8Offset": previous_byte,
            "attributedKernel": attributed,
        })

    while match := events.search(code, cursor):
        start = match.start()
        if match.group("attribute") is None and start > 0 and ("a" + code[start - 1]).isidentifier():
            cursor = match.end()
            continue
        while macro_index < len(macros) and macros[macro_index][1] <= start:
            macro_index += 1
        if macro_index < len(macros) and macros[macro_index][0] <= start < macros[macro_index][1]:
            cursor = macros[macro_index][1]
            continue
        attributed = False
        if match.group("attribute") is not None:
            cursor, body = _rust_attribute(code, start, pairs)
            # Inner attributes attach to the enclosing module/crate, not a fn.
            if "!" in match.group("attribute"):
                continue
            attributed = _attribute_has_qualified_kernel_name(body)
            while following := following_attribute.match(code, cursor):
                cursor, body = _rust_attribute(code, following.start(), pairs)
                attributed |= _attribute_has_qualified_kernel_name(body)
            function = declaration.match(code, cursor)
            if function is None:
                continue
            name_start = function.end()
        else:
            name_start = match.end()
            while name_start < len(code) and code[name_start].isspace():
                name_start += 1
        identifier = _rust_function_identifier(code, name_start)
        if identifier is None:
            cursor = name_start
            continue
        cursor, name = identifier
        append(name_start, name, attributed)
    return records


def source_contains_ordinary_attributed_kernel(source: str) -> bool:
    return bool(ordinary_attributed_kernel_names(source))


def ordinary_attributed_kernel_names(source: str) -> list[str]:
    return ordinary_attributed_function_names(source, _attribute_has_kernel_name)


def ordinary_attributed_function_names(
    source: str, has_attribute: Callable[[str], bool],
) -> list[str]:
    code = _rust_code_without_comments_and_literals(source)
    pairs = _rust_delimiters(code)
    macro_bodies = _macro_rule_bodies(code, pairs)
    names: list[str] = []

    def in_macro_body(index: int) -> bool:
        return any(start < index < end for start, end in macro_bodies)

    for attribute in ATTRIBUTE_START.finditer(code):
        if in_macro_body(attribute.start()):
            continue
        # Inner attributes apply to a module/crate, never to a kernel item.
        if attribute.start() > 0 and code[attribute.start() - 1] == "!":
            continue
        closing, body = _rust_attribute(code, attribute.start(), pairs)
        if not has_attribute(body):
            continue
        cursor = closing
        while True:
            following = re.match(r"\s*#\s*\[", code[cursor:])
            if following is None:
                break
            nested_start = cursor + following.start()
            nested_close, _ = _rust_attribute(code, nested_start, pairs)
            cursor = nested_close
        declaration = RUST_FUNCTION_NAME.match(code[cursor:])
        if declaration is not None:
            names.append(declaration.group(1))
    return names


def _package_path(package_root: Path, value: Any, label: str) -> Path:
    text = require_string(value, label)
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != text:
        fail(f"{label} must stay within its package root: {text!r}")
    if relative.parts and relative.parts[0] in IGNORED_PACKAGE_DIRECTORIES:
        fail(f"{label} points into a generated package directory: {text!r}")
    path = package_root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        fail(f"{label} does not name a regular package file: {text}")
    return path


def _validate_package_source_path(
    repo_root: Path, package_root: Path, value: Any, label: str
) -> Path:
    text = checked_path(repo_root, value, label)
    path = repo_root.joinpath(*PurePosixPath(text).parts)
    try:
        relative = path.relative_to(package_root)
    except ValueError:
        fail(f"{label} escapes its package root: {text}")
    if relative.parts and relative.parts[0] in IGNORED_PACKAGE_DIRECTORIES:
        fail(f"{label} points into a generated package directory: {text}")
    return path


def _validate_package_targets(package_root: Path, manifest_path: Path) -> None:
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        fail(f"cannot read package manifest: {error}")
    if len(payload) > MAX_CARGO_MANIFEST_BYTES:
        fail("package manifest exceeds the source bound")
    try:
        cargo = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot parse package manifest: {error}")
    for section in TARGET_SECTIONS:
        targets = cargo.get(section, []) if section != "lib" else [cargo.get("lib", {})]
        if isinstance(targets, dict):
            targets = [targets]
        if not isinstance(targets, list):
            fail(f"package manifest target section {section} is malformed")
        for index, target in enumerate(targets):
            if not isinstance(target, dict) or "path" not in target:
                continue
            _package_path(
                package_root, target["path"], f"Cargo [{section}][{index}].path"
            )


def package_rust_sources(
    repo_root: Path, package_manifest: str, label: str
) -> list[tuple[Path, str]]:
    manifest_path = repo_root.joinpath(*PurePosixPath(package_manifest).parts)
    package_root = manifest_path.parent
    _validate_package_targets(package_root, manifest_path)
    sources: list[tuple[Path, str]] = []
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(
        package_root, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink():
                fail(
                    f"{label} package contains a symlink: {path.relative_to(repo_root)}"
                )
        directory_names[:] = [
            name for name in directory_names if name not in IGNORED_PACKAGE_DIRECTORIES
        ]
        for name in file_names:
            path = directory_path / name
            if path.is_symlink():
                fail(
                    f"{label} package contains a symlink: {path.relative_to(repo_root)}"
                )
            if path.suffix != ".rs":
                continue
            if not path.is_file():
                fail(
                    f"{label} source is not a regular file: "
                    f"{path.relative_to(repo_root)}"
                )
            try:
                size = path.stat().st_size
                if size > MAX_ATTRIBUTED_SOURCE_BYTES:
                    fail(
                        f"{label} source exceeds the source bound: "
                        f"{path.relative_to(repo_root)}"
                    )
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                fail(f"cannot read {label} package source {path}: {error}")
            total_bytes += size
            if total_bytes > MAX_PACKAGE_SOURCE_BYTES:
                fail(f"{label} package source closure exceeds the byte bound")
            sources.append((path, text))
            if len(sources) > MAX_PACKAGE_SOURCE_FILES:
                fail(f"{label} package source closure exceeds the file bound")
    if not sources:
        fail(f"{label} package has no Rust source files")
    return sources


def sha256_file(path: Path, maximum_bytes: int, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is not a regular file")
    try:
        size = path.stat().st_size
        if size > maximum_bytes:
            fail(f"{label} exceeds its byte bound")
        payload = path.read_bytes()
    except OSError as error:
        fail(f"cannot read {label}: {error}")
    if len(payload) != size:
        fail(f"{label} changed while it was read")
    return hashlib.sha256(payload).hexdigest()


def package_source_closure_sha256(
    repo_root: Path, sources: list[tuple[Path, str]]
) -> str:
    digest = hashlib.sha256(SOURCE_CLOSURE_DIGEST_DOMAIN)
    for path, text in sources:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        payload = text.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def parse_package_identity(
    manifest_path: Path, label: str
) -> tuple[dict[str, Any], str, str]:
    try:
        payload = manifest_path.read_bytes()
        if len(payload) > MAX_CARGO_MANIFEST_BYTES:
            fail(f"{label} exceeds the manifest byte bound")
        cargo = tomllib.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot parse {label}: {error}")
    package = require_object(cargo.get("package"), f"{label} [package]")
    package_name = require_string(package.get("name"), f"{label} package.name")
    package_version = require_string(package.get("version"), f"{label} package.version")
    return cargo, package_name, package_version


def effective_cargo_lock(repo_root: Path, package_root: Path, label: str) -> Path:
    directory = package_root
    while directory == repo_root or repo_root in directory.parents:
        candidate = directory / "Cargo.lock"
        if candidate.exists() or candidate.is_symlink():
            relative = candidate.relative_to(repo_root).as_posix()
            checked_path(repo_root, relative, f"{label}.compilerInput.cargoLockPath")
            return candidate
        manifest = directory / "Cargo.toml"
        if manifest.is_file():
            try:
                cargo = tomllib.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
                fail(f"cannot locate {label} Cargo.lock: {error}")
            if "workspace" in cargo:
                fail(f"{label} workspace root has no Cargo.lock: {directory}")
        if directory == repo_root:
            break
        directory = directory.parent
    fail(f"{label} package has no effective Cargo.lock within the repository")


def cargo_feature_closure(
    cargo: dict[str, Any], direct: list[str], default_features: bool, label: str
) -> list[str]:
    raw_features = cargo.get("features", {})
    features = require_object(raw_features, f"{label} [features]")
    pending = list(direct)
    if default_features and "default" in features:
        pending.append("default")
    enabled: set[str] = set()
    while pending:
        feature = pending.pop()
        if feature in enabled:
            continue
        if feature not in features:
            fail(f"{label} selects unknown Cargo feature: {feature}")
        enabled.add(feature)
        members = features[feature]
        if not isinstance(members, list):
            fail(f"{label} feature {feature} must be an array")
        for member in members:
            member = require_string(member, f"{label} feature {feature}[]")
            local = member.partition("?")[0]
            if local.startswith("dep:") or "/" in local:
                continue
            if local in features:
                pending.append(local)
    return sorted(enabled)


def fixture_input_contract_sha256(fixture: dict[str, Any]) -> str:
    compiler_input = fixture["compilerInput"]
    contract = {
        "fixtureId": fixture["fixtureId"],
        "target": fixture["target"],
        "matrix": fixture["matrix"],
        "compilerInput": {
            key: value
            for key, value in compiler_input.items()
            if key != "contractSha256"
        },
    }
    payload = json.dumps(
        contract,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(FIXTURE_INPUT_DIGEST_DOMAIN + payload).hexdigest()


def validate_compiler_input(
    repo_root: Path,
    fixture: dict[str, Any],
    label: str,
    package_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    compiler_input = require_object(fixture["compilerInput"], f"{label}.compilerInput")
    require_exact_keys(compiler_input, COMPILER_INPUT_KEYS, f"{label}.compilerInput")
    result = validate_compiler_input_data(
        repo_root, {key: value for key, value in compiler_input.items() if key != "contractSha256"},
        label, package_cache,
    )
    contract_sha256 = require_string(
        compiler_input["contractSha256"], f"{label}.compilerInput.contractSha256"
    )
    if contract_sha256 != fixture_input_contract_sha256(fixture):
        fail(f"{label}.compilerInput.contractSha256 is stale")
    return result


def validate_compiler_input_data(
    repo_root: Path,
    compiler_input: dict[str, Any],
    label: str,
    package_cache: dict[str, dict[str, Any]] | None = None,
    *, feature_scoped_includes: bool = False,
) -> dict[str, Any]:
    require_exact_keys(
        require_object(compiler_input, f"{label}.compilerInput"),
        COMPILER_INPUT_KEYS - {"contractSha256"}, f"{label}.compilerInput",
    )
    package_manifest = checked_path(
        repo_root,
        compiler_input["packageManifest"],
        f"{label}.compilerInput.packageManifest",
    )
    if not package_manifest.endswith("/Cargo.toml"):
        fail(f"{label}.compilerInput.packageManifest must name a Cargo.toml")
    cached = None if package_cache is None else package_cache.get(package_manifest)
    if cached is None:
        manifest_path = repo_root.joinpath(*PurePosixPath(package_manifest).parts)
        package_root = manifest_path.parent
        cargo, package_name, package_version = parse_package_identity(
            manifest_path, f"{label}.compilerInput.packageManifest"
        )
        package_sources = package_rust_sources(repo_root, package_manifest, label)
        for path, text in package_sources:
            validate_rust_path_attributes(text, path, package_root, label)
        cached = {
            "manifestPath": manifest_path,
            "packageRoot": package_root,
            "cargo": cargo,
            "packageName": package_name,
            "packageVersion": package_version,
            "manifestSha256": sha256_file(
                manifest_path, MAX_CARGO_MANIFEST_BYTES, f"{label} package manifest"
            ),
            "cargoLockPath": effective_cargo_lock(repo_root, package_root, label),
            "packageSources": package_sources,
            "attributedNames": {},
            "sourceClosureSha256": package_source_closure_sha256(
                repo_root, package_sources
            ),
        }
        if package_cache is not None:
            package_cache[package_manifest] = cached
    manifest_path = cached["manifestPath"]
    package_root = cached["packageRoot"]
    cargo = cached["cargo"]
    package_name = cached["packageName"]
    package_version = cached["packageVersion"]
    manifest_sha256 = require_string(
        compiler_input["packageManifestSha256"],
        f"{label}.compilerInput.packageManifestSha256",
    )
    if LOWER_SHA256.fullmatch(manifest_sha256) is None:
        fail(f"{label}.compilerInput.packageManifestSha256 is not lowercase SHA-256")
    if manifest_sha256 != cached["manifestSha256"]:
        fail(f"{label}.compilerInput.packageManifestSha256 is stale")

    cargo_lock_path = checked_path(
        repo_root,
        compiler_input["cargoLockPath"],
        f"{label}.compilerInput.cargoLockPath",
    )
    expected_cargo_lock_path = cached["cargoLockPath"].relative_to(repo_root).as_posix()
    if cargo_lock_path != expected_cargo_lock_path:
        fail(f"{label}.compilerInput.cargoLockPath is not the effective Cargo.lock")
    cargo_lock_sha256 = require_string(
        compiler_input["cargoLockSha256"],
        f"{label}.compilerInput.cargoLockSha256",
    )
    if LOWER_SHA256.fullmatch(cargo_lock_sha256) is None:
        fail(f"{label}.compilerInput.cargoLockSha256 is not lowercase SHA-256")
    if cargo_lock_sha256 != sha256_file(
        cached["cargoLockPath"], MAX_CARGO_LOCK_BYTES, f"{label} effective Cargo.lock"
    ):
        fail(f"{label}.compilerInput.cargoLockSha256 is stale")

    closure_sha256 = require_string(
        compiler_input["sourceClosureSha256"],
        f"{label}.compilerInput.sourceClosureSha256",
    )
    if LOWER_SHA256.fullmatch(closure_sha256) is None:
        fail(f"{label}.compilerInput.sourceClosureSha256 is not lowercase SHA-256")
    if closure_sha256 != cached["sourceClosureSha256"]:
        fail(f"{label}.compilerInput.sourceClosureSha256 is stale")

    source_paths = require_string_list(
        compiler_input["sourcePaths"], f"{label}.compilerInput.sourcePaths"
    )
    attributed_names: set[str] = set()
    for index, source_path in enumerate(source_paths):
        path = _validate_package_source_path(
            repo_root,
            package_root,
            source_path,
            f"{label}.compilerInput.sourcePaths[{index}]",
        )
        source_key = path.relative_to(package_root).as_posix()
        names = cached["attributedNames"].get(source_key)
        if names is None:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                fail(f"cannot read {label} compiler source: {error}")
            names = ordinary_attributed_kernel_names(source)
            cached["attributedNames"][source_key] = names
        attributed_names.update(names)

    cargo_target = require_object(
        compiler_input["cargoTarget"], f"{label}.compilerInput.cargoTarget"
    )
    require_exact_keys(
        cargo_target, CARGO_TARGET_KEYS, f"{label}.compilerInput.cargoTarget"
    )
    if cargo_target["kind"] != "lib":
        fail(f"{label}.compilerInput.cargoTarget.kind must be lib")
    lib = cargo.get("lib", {})
    if not isinstance(lib, dict):
        fail(f"{label} Cargo [lib] must be a table")
    expected_name = lib.get("name", package_name.replace("-", "_"))
    expected_source = lib.get("path", "src/lib.rs")
    if cargo_target["name"] != expected_name:
        fail(f"{label}.compilerInput.cargoTarget.name differs from Cargo [lib]")
    if cargo_target["sourcePath"] != expected_source:
        fail(f"{label}.compilerInput.cargoTarget.sourcePath differs from Cargo [lib]")
    _package_path(package_root, expected_source, f"{label} Cargo [lib].path")

    if not isinstance(compiler_input["defaultFeatures"], bool):
        fail(f"{label}.compilerInput.defaultFeatures must be boolean")
    direct_features = require_string_array(
        compiler_input["features"], f"{label}.compilerInput.features"
    )
    if direct_features != sorted(direct_features):
        fail(f"{label}.compilerInput.features must be sorted")
    enabled_features = cargo_feature_closure(
        cargo, direct_features, compiler_input["defaultFeatures"], label
    )
    build = cargo["package"].get("build")
    declared_features = cargo.get("features", {})
    only_local_features = all(
        isinstance(values, list) and all(
            isinstance(value, str) and "/" not in value and not value.startswith("dep:")
            for value in values
        ) for values in declared_features.values()
    )
    can_scope_includes = feature_scoped_includes and (
        build is False or (build is None and not (package_root / "build.rs").exists())
    ) and only_local_features
    include_selection = tuple(enabled_features) if can_scope_includes else None
    checked_includes = cached.setdefault("checkedIncludes", set())
    if include_selection not in checked_includes:
        for path, text in cached["packageSources"]:
            validate_rust_source_includes(
                text, path, package_root, label,
                inactive_features=frozenset(declared_features) - frozenset(enabled_features) if can_scope_includes else None,
            )
        checked_includes.add(include_selection)
    kernel_symbols = require_string_list(
        compiler_input["kernelSymbols"], f"{label}.compilerInput.kernelSymbols"
    )
    if kernel_symbols != sorted(kernel_symbols):
        fail(f"{label}.compilerInput.kernelSymbols must be sorted")
    for symbol in kernel_symbols:
        if RUST_IDENTIFIER.fullmatch(symbol) is None:
            fail(f"{label}.compilerInput.kernelSymbols contains an invalid Rust symbol")
        if symbol not in attributed_names:
            fail(
                f"{label}.compilerInput.kernelSymbols is not attributed in sourcePaths: {symbol}"
            )
    return {
        "packageName": package_name,
        "packageVersion": package_version,
        "manifestSha256": manifest_sha256,
        "cargoLockSha256": cargo_lock_sha256,
        "sourceClosureSha256": closure_sha256,
        "enabledFeatures": enabled_features,
        "primarySourceSha256": sha256_file(
            package_root / expected_source,
            MAX_ATTRIBUTED_SOURCE_BYTES,
            f"{label} primary source",
        ),
    }


def validate_rust_path_attributes(
    source: str, source_path: Path, package_root: Path, label: str
) -> None:
    code = _rust_code_without_comments_and_literals(source)
    pairs = _rust_delimiters(code)
    for attribute in ATTRIBUTE_START.finditer(code):
        closing, _ = _rust_attribute(code, attribute.start(), pairs)
        original_opening = code.find("[", attribute.start(), closing)
        body = source[original_opening + 1 : closing - 1]
        match = re.match(
            r"\s*path\s*=\s*(?:\"(?P<quoted>(?:\\.|[^\"\\])*)\"|"
            r"r(?P<hashes>#{0,255})\"(?P<raw>.*?)\"(?P=hashes))",
            body,
            re.DOTALL,
        )
        if match is None:
            continue
        value = match.group("quoted") or match.group("raw") or ""
        relative = PurePosixPath(value.replace("\\", ""))
        if relative.is_absolute():
            fail(f"{label} has a module path escaping its package root: {value!r}")
        candidate = source_path.parent.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(package_root)
        except ValueError:
            fail(f"{label} has a module path escaping its package root: {value!r}")
        package_relative = resolved.relative_to(package_root)
        if (
            package_relative.parts
            and package_relative.parts[0] in IGNORED_PACKAGE_DIRECTORIES
        ):
            fail(f"{label} references a generated package path: {value!r}")
        if not candidate.is_file() or candidate.is_symlink():
            fail(f"{label} module path does not name a regular file: {value!r}")


def inactive_feature_modules(
    source: str, code: str, pairs: dict[int, int], inactive_features: frozenset[str],
) -> list[tuple[int, int]]:
    """Only a literal false cfg on a top-level inline module excludes its body."""
    inactive = []
    for attribute in ATTRIBUTE_START.finditer(code):
        start = attribute.start()
        if (start > 0 and code[start - 1] == "!") or any(
            opening < start < closing for opening, closing in pairs.items()
        ):
            continue
        closing, _ = _rust_attribute(code, start, pairs)
        opening = code.find("[", start, closing)
        condition = re.fullmatch(
            r'\s*cfg\s*\(\s*feature\s*=\s*"([A-Za-z0-9_-]+)"\s*\)\s*',
            source[opening + 1:closing - 1],
        )
        if condition is None or condition[1] not in inactive_features:
            continue
        cursor = closing
        while re.match(r"\s*#\s*\[", code[cursor:]):
            cursor, _ = _rust_attribute(code, cursor + len(code[cursor:]) - len(code[cursor:].lstrip()), pairs)
        module = re.match(r"\s*(?:pub(?:\([^()]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", code[cursor:])
        if module is not None:
            body = cursor + module.end() - 1
            if body in pairs:
                inactive.append((body, pairs[body]))
    return inactive


def validate_rust_source_includes(
    source: str, source_path: Path, package_root: Path, label: str,
    *, inactive_features: frozenset[str] | None = None,
) -> None:
    code = _rust_code_without_comments_and_literals(source)
    pairs = _rust_delimiters(code)
    inactive = [] if inactive_features is None else inactive_feature_modules(source, code, pairs, inactive_features)
    for include in SOURCE_INCLUDE_START.finditer(code):
        if any(start < include.start() < end for start, end in inactive):
            continue
        cursor = include.end()
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        if cursor >= len(code) or code[cursor] != "(" or cursor not in pairs:
            fail(f"{label} has an unterminated include! invocation")
        closing = pairs[cursor]
        body = source[cursor + 1 : closing - 1]
        literal = re.fullmatch(
            r"\s*(?:\"(?P<quoted>(?:\\.|[^\"\\])*)\"|"
            r"r(?P<hashes>#{0,255})\"(?P<raw>.*?)\"(?P=hashes))\s*",
            body,
            re.DOTALL,
        )
        if literal is None:
            fail(f"{label} has a non-literal include! path")
        value = literal.group("quoted") or literal.group("raw") or ""
        value = value.replace("\\\\", "\\").replace('\\"', '"')
        relative = PurePosixPath(value)
        if relative.is_absolute():
            fail(f"{label} has an include! path escaping its package root: {value!r}")
        candidate = source_path.parent.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=False)
            package_relative = resolved.relative_to(package_root)
        except ValueError:
            fail(f"{label} has an include! path escaping its package root: {value!r}")
        if (
            package_relative.parts
            and package_relative.parts[0] in IGNORED_PACKAGE_DIRECTORIES
        ):
            fail(f"{label} references a generated package path: {value!r}")
        if not candidate.is_file():
            source_relative = source_path.relative_to(package_root)
            if source_relative.parts[:2] != ("tests", "fixtures"):
                fail(f"{label} include! path is not a regular file: {value!r}")
        if candidate.is_symlink():
            fail(f"{label} include! path names a symlink: {value!r}")


def require_bounded_command_string(value: Any, label: str) -> str:
    text = require_string(value, label)
    if len(text) > 256 or "\0" in text or "\n" in text or "\r" in text:
        fail(f"{label} exceeds the command string bounds")
    if ".." in PurePosixPath(text).parts:
        fail(f"{label} contains path traversal")
    return text



def bounded_list(value: Any, label: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        fail(f"{label} must contain 1..{maximum} records")
    return value


def expected_path(value: Any, label: str) -> str:
    """A pending obligation can name an absent file, but not escape the repo."""
    text = require_bounded_command_string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text or text == ".":
        fail(f"{label} is not a canonical repository-relative file")
    return text


def require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")


def validate_pending_simulation(value: Any, label: str) -> None:
    simulation = require_object(value, label)
    require_exact_keys(simulation, SIMULATION_KEYS, label)
    status = simulation["status"]
    if status not in {"pending-reconciliation", "pending-design"}:
        fail(f"{label} must remain explicitly pending")
    if simulation["bundleVersion"] != 7 or simulation["canonicalKirVersion"] != 12:
        fail(f"{label} must retain the Bundle V7 / KIR V12 obligation")
    for kind in ("request", "expectation"):
        expected_path(simulation[f"{kind}Path"], f"{label}.{kind}Path")
        digest, size = simulation[f"{kind}Sha256"], simulation[f"{kind}Bytes"]
        if status == "pending-design":
            if digest is not None or size is not None:
                fail(f"{label} undesigned oracle cannot claim content pins")
        else:
            require_digest(digest, f"{label}.{kind}Sha256")
            if type(size) is not int or not 0 < size <= MAX_SIMULATION_DOCUMENT_BYTES:
                fail(f"{label}.{kind}Bytes exceeds the pending document bound")


def matrix_record(fixture: dict[str, Any]) -> str:
    matrix = fixture["matrix"]
    return "|".join(
        [
            matrix["caseId"],
            matrix["runnerPath"],
            matrix["artifactName"],
            next(iter(matrix["runnerArguments"]), ""),
            next(iter(matrix["environment"]), "").partition("=")[2],
            fixture["fixtureId"],
        ]
    )


def validate_matrix(repo_root: Path, fixture: dict[str, Any], label: str) -> None:
    matrix = require_object(fixture["matrix"], f"{label}.matrix")
    require_exact_keys(matrix, MATRIX_KEYS, f"{label}.matrix")
    for key in ("caseId", "artifactName"):
        if not isinstance(matrix[key], str) or re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", matrix[key]
        ) is None:
            fail(f"{label}.matrix.{key} is not a bounded identifier")
    if not matrix["artifactName"].endswith(".hsaco"):
        fail(f"{label} must name an expected HSACO artifact")
    runner = checked_path(repo_root, matrix["runnerPath"], f"{label}.matrix.runnerPath")
    if not runner.startswith("examples/") or not runner.endswith(".sh"):
        fail(f"{label} must use an ordinary-source example runner")
    arguments = require_string_array(matrix["runnerArguments"], f"{label}.arguments")
    if len(arguments) > 1 or any(
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", arg) is None
        for arg in arguments
    ):
        fail(f"{label} runner arguments exceed the matrix record contract")
    environment = require_string_array(matrix["environment"], f"{label}.environment")
    if len(environment) > 1 or any(
        re.fullmatch(
            r"FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT="
            r"(expert-serial|speculative-recompute-prefix|ngram-reverse-probe|muon-broadcast16)",
            item,
        ) is None
        for item in environment
    ):
        fail(f"{label} environment is not a supported ablation selection")
    if environment and fixture["target"] != "gfx950":
        fail(f"{label} gfx950 ablation selection used for another target")
    if "|" in runner or "\n" in runner or "\r" in runner:
        fail(f"{label} runner cannot be represented in a matrix record")


def validate_qualification(
    value: Any, entries: dict[str, Any], fixtures: dict[str, Any]
) -> None:
    qualification = require_object(value, "qualification")
    require_exact_keys(qualification, QUALIFICATION_KEYS, "qualification")
    if qualification["status"] != "pending":
        fail("source contracts cannot claim qualification")
    expected_path(qualification["evidenceSchemaPath"], "qualification.evidenceSchemaPath")
    targets = set()
    for hardware in bounded_list(qualification["hardwareTargets"], "hardwareTargets", 2):
        require_exact_keys(
            require_object(hardware, "hardware target"), HARDWARE_TARGET_KEYS, "hardware target"
        )
        target = require_target(hardware["target"], "hardware target")
        if target in targets:
            fail("duplicate hardware target")
        targets.add(target)
        if hardware["deterministicSemanticRunnerAvailability"] != "unavailable":
            fail("hardware qualification must remain unavailable")
        require_string(hardware["reason"], "hardware reason")
    if targets != ALLOWED_TARGETS:
        fail("both hardware targets must retain explicit obligations")
    suites = set()
    covered: set[tuple[str, str, str]] = set()
    for suite in bounded_list(qualification["suites"], "qualification.suites", 512):
        require_exact_keys(require_object(suite, "suite"), SUITE_KEYS, "suite")
        suite_id = require_string(suite["suiteId"], "suiteId")
        if suite_id in suites:
            fail("duplicate suiteId")
        suites.add(suite_id)
        gate = suite["gate"]
        if gate not in {"semantic-simulation", "cpu-reference"}:
            fail("unsupported pending suite gate")
        if suite["availability"] != "pending":
            fail("a retained suite command is not an available or passed adapter")
        require_string(suite["unavailableReason"], "suite pending reason")
        command = require_object(suite["command"], "suite.command")
        require_exact_keys(command, SUITE_COMMAND_KEYS, "suite.command")
        expected_path(command["executable"], "suite.command.executable")
        for key, maximum in (
            ("arguments", MAX_SUITE_COMMAND_ARGUMENTS),
            ("environment", MAX_SUITE_COMMAND_ENVIRONMENT),
        ):
            values = require_string_array(command[key], f"suite.command.{key}")
            if len(values) > maximum:
                fail(f"suite.command.{key} exceeds its bound")
            for item in values:
                require_bounded_command_string(item, f"suite.command.{key}[]")
        if command["workingDirectory"] != ".":
            expected_path(command["workingDirectory"], "suite.command.workingDirectory")
        timeout = command["timeoutSeconds"]
        if type(timeout) is not int or not 0 < timeout <= 86400:
            fail("suite.command.timeoutSeconds exceeds its bound")
        local_coverage = set()
        for coverage in bounded_list(suite["coverage"], "suite.coverage", 256):
            require_exact_keys(
                require_object(coverage, "suite coverage"), SUITE_COVERAGE_KEYS, "suite coverage"
            )
            lesson = coverage["lessonId"]
            if lesson not in entries:
                fail("suite references an unknown lesson")
            ids = require_string_list(coverage["fixtureIds"], "suite fixtureIds")
            for fixture_id in ids:
                if fixture_id not in fixtures or fixture_id not in entries[lesson]["compilerFixtureIds"]:
                    fail("suite fixture is not bound to its lesson")
                if gate not in entries[lesson]["requiredGates"]:
                    fail("suite does not correspond to a required lesson gate")
                key = (lesson, fixture_id, gate)
                if key in local_coverage:
                    fail("duplicate suite coverage")
                local_coverage.add(key)
                covered.add(key)
    for lesson, entry in entries.items():
        for fixture_id in entry["compilerFixtureIds"]:
            for gate in set(entry["requiredGates"]) & {"semantic-simulation", "cpu-reference"}:
                if (lesson, fixture_id, gate) not in covered:
                    fail(f"missing pending {gate} obligation for {lesson}/{fixture_id}")


def require_digest(value: Any, label: str, length: int = 64) -> None:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        fail(f"{label} must be an exact lowercase {length}-digit digest")


def source_item_contract_sha256(lesson_id: str, tab: dict[str, Any]) -> str:
    payload = {
        "curriculumSchema": CURRICULUM_SCHEMA_V2,
        "lessonId": lesson_id,
        "tab": {key: tab[key] for key in CURRICULUM_TAB_FIELDS},
        "sourceItem": {key: value for key, value in tab["sourceItem"].items() if key != "contractSha256"},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(SOURCE_ITEM_DIGEST_DOMAIN + encoded).hexdigest()


def source_item_fragments(repo_root: Path, tab: dict[str, Any], label: str) -> list[str]:
    item = tab["sourceItem"]
    if tab["sourceDigestScope"] == "file":
        if tab["sourceFragmentsSha256"] is not None:
            fail(f"{label} whole-file source cannot carry fragment digests")
        require_digest(tab["sourceCommit"], f"{label}.sourceCommit", 40)
        require_digest(tab["sourceSha256"], f"{label}.sourceSha256")
        require_digest(tab["displayedSha256"], f"{label}.displayedSha256")
        path = checked_path(repo_root, tab["sourcePath"], f"{label}.sourcePath")
        source_path = repo_root / path
        if source_path.stat().st_size > MAX_ATTRIBUTED_SOURCE_BYTES:
            fail(f"{label} attributed source exceeds its bound")
        source = source_path.read_bytes()
        ranges = bounded_list(item["sourceRanges"], f"{label}.sourceRanges", 64)
        if len(ranges) != 1:
            fail(f"{label} whole-file source requires exactly one full byte range")
        selected = require_object(ranges[0], "source range")
        require_exact_keys(selected, {"byteOffset", "byteLength"}, "source range")
        offset, size = selected["byteOffset"], selected["byteLength"]
        if type(offset) is not int or type(size) is not int or offset != 0 or size != len(source):
            fail(f"{label} whole-file range must cover every source byte")
        if type(tab["displayedUtf8Bytes"]) is not int or tab["displayedUtf8Bytes"] != len(source):
            fail(f"{label} whole-file displayed byte count differs")
        try:
            text = source.decode("utf-8")
        except UnicodeError:
            fail(f"{label} whole-file source is not valid UTF-8")
        digest = hashlib.sha256(source).hexdigest()
        if tab["sourceSha256"] != digest or tab["displayedSha256"] != digest:
            fail(f"{label} whole-file source or displayed digest is stale")
        return [text]
    if tab["sourceDigestScope"] != "displayed" or not tab["sourceFragmentsSha256"]:
        fail(f"{label} requires exact displayed fragments")
    path = checked_path(repo_root, tab["sourcePath"], f"{label}.sourcePath")
    source_path = repo_root / path
    if source_path.stat().st_size > MAX_ATTRIBUTED_SOURCE_BYTES:
        fail(f"{label} attributed source exceeds its bound")
    source = source_path.read_bytes()
    ranges = bounded_list(item["sourceRanges"], f"{label}.sourceRanges", 64)
    if len(ranges) != len(tab["sourceFragmentsSha256"]):
        fail(f"{label} fragment range coverage differs")
    intervals: list[tuple[int, int]] = []
    fragments = []
    for index, selected in enumerate(ranges):
        require_exact_keys(require_object(selected, "source range"), {"byteOffset", "byteLength"}, "source range")
        offset, size = selected["byteOffset"], selected["byteLength"]
        if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0 or offset + size > len(source):
            fail(f"{label} source range is outside its byte bounds")
        end = offset + size
        if any(offset < stop and start < end for start, stop in intervals):
            fail(f"{label} source ranges overlap")
        intervals.append((offset, end))
        encoded = source[offset:end]
        try:
            fragments.append(encoded.decode("utf-8"))
        except UnicodeError:
            fail(f"{label} source range is not valid UTF-8")
        if hashlib.sha256(encoded).hexdigest() != tab["sourceFragmentsSha256"][index]:
            fail(f"{label} source fragment digest is stale")
    return fragments


def source_driver_test_names(source: str) -> list[str]:
    code = _rust_code_without_comments_and_literals(source)
    pairs = _rust_delimiters(code)

    def top_level(start: int) -> bool:
        return not any(opening < start < closing for opening, closing in pairs.items())

    def harmless(body: str) -> bool:
        return re.match(r"\s*(?:test|ignore|doc|allow|warn|deny|forbid|expect)\b(?!\s*::)", body) is not None

    for attribute in re.finditer(r"#\s*!\s*\[", code):
        if top_level(attribute.start()) and not harmless(_rust_attribute(code, attribute.start(), pairs)[1]):
            return []
    attributes = [(match.start(), *_rust_attribute(code, match.start(), pairs))
                  for match in ATTRIBUTE_START.finditer(code)]
    admitted = set()
    for index, (start, _, body) in enumerate(attributes):
        if body.strip() != "test" or not top_level(start):
            continue
        first = last = index
        while first > 0 and not code[attributes[first - 1][1]:attributes[first][0]].strip():
            first -= 1
        while last + 1 < len(attributes) and not code[attributes[last][1]:attributes[last + 1][0]].strip():
            last += 1
        bodies = [entry[2] for entry in attributes[first:last + 1]]
        # The production gate runs --ignored --exact; conditional or custom
        # attributes cannot establish that this ordinary test will be registered.
        if not all(harmless(value) for value in bodies) or not any(re.match(r"\s*ignore\b", value) for value in bodies):
            continue
        declaration = RUST_FUNCTION_NAME.match(code[attributes[last][1]:])
        if declaration is not None:
            admitted.add(declaration[1])
    return [name for name in ordinary_attributed_function_names(source, lambda body: body.strip() == "test")
            if name in admitted]


def validate_source_item(
    repo_root: Path, lesson_id: str, tab: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> str:
    label = f"curriculum {lesson_id} tab {tab['ordinal']} source item"
    item = require_object(tab["sourceItem"], label)
    require_exact_keys(item, {"kind", "compilerInput", "driver", "sourceRanges", "cases", "contractSha256"}, label)
    if item["kind"] != "source-driver":
        fail(f"{label} has an unsupported kind")
    shared = require_object(item["compilerInput"], f"{label}.compilerInput")
    require_exact_keys(shared, COMPILER_INPUT_KEYS - {"features", "kernelSymbols", "contractSha256"}, f"{label}.compilerInput")
    if shared["sourcePaths"] != [tab["sourcePath"]]:
        fail(f"{label} must bind its exact displayed source path")
    driver = require_object(item["driver"], f"{label}.driver")
    require_exact_keys(driver, {"package", "target", "path"}, f"{label}.driver")
    package = require_string(driver["package"], f"{label}.driver.package")
    target = require_string(driver["target"], f"{label}.driver.target")
    if re.fullmatch(r"[a-z][a-z0-9-]*", package) is None or RUST_IDENTIFIER.fullmatch(target) is None:
        fail(f"{label} requires an exact integration-test driver")
    if driver["path"] != f"crates/{package}/tests/{target}.rs":
        fail(f"{label} requires an exact integration-test driver path")
    driver_path = checked_path(repo_root, driver["path"], f"{label}.driver.path")
    driver_manifest = checked_path(repo_root, f"crates/{package}/Cargo.toml", f"{label}.driver manifest")
    driver_cargo, actual_package, _ = parse_package_identity(repo_root / driver_manifest, label)
    if actual_package != package or (repo_root / driver_path).stat().st_size > MAX_ATTRIBUTED_SOURCE_BYTES:
        fail(f"{label} driver package or byte bound differs")
    tests = driver_cargo.get("test", [])
    if not isinstance(tests, list) or any(not isinstance(test, dict) for test in tests):
        fail(f"{label} driver Cargo test targets are malformed")
    selected = [test for test in tests if test.get("name") == target]
    if any(test.get("path") == f"tests/{target}.rs" and test.get("name") != target for test in tests):
        fail(f"{label} driver Cargo test path has a different target name")
    if len(selected) > 1:
        fail(f"{label} driver has duplicate Cargo test targets")
    if selected:
        test = selected[0]
        if (test.get("path", f"tests/{target}.rs") != f"tests/{target}.rs"
                or test.get("harness", True) is not True or test.get("required-features", []) != []):
            fail(f"{label} driver Cargo test target differs")
    elif driver_cargo["package"].get("autotests", True) is not True:
        fail(f"{label} driver Cargo test target is disabled")
    driver_names = source_driver_test_names((repo_root / driver_path).read_text(encoding="utf-8"))
    fragments = source_item_fragments(repo_root, tab, label)
    expected = [(ordinal, symbol) for ordinal, fragment in enumerate(fragments)
                for symbol in ordinary_attributed_kernel_names(fragment)]
    if not expected or len(set(symbol for _, symbol in expected)) != len(expected):
        fail(f"{label} displayed kernel declarations must be nonempty and unique")
    observed = []
    for row in bounded_list(item["cases"], f"{label}.cases", 256):
        require_exact_keys(require_object(row, "source case"),
                           {"features", "kernelSymbol", "target", "displayedFragmentOrdinal", "testFunction", "expectation"}, "source case")
        symbol = require_string(row["kernelSymbol"], f"{label}.kernelSymbol")
        ordinal = row["displayedFragmentOrdinal"]
        if type(ordinal) is not int or not 0 <= ordinal < len(fragments):
            fail(f"{label} displayedFragmentOrdinal is outside its bounds")
        observed.append((ordinal, symbol))
        require_target(row["target"], f"{label}.target")
        name = require_string(row["testFunction"], f"{label}.testFunction")
        if RUST_IDENTIFIER.fullmatch(name) is None or driver_names.count(name) != 1:
            fail(f"{label} must identify one existing driver test function")
        expectation = require_object(row["expectation"], f"{label}.expectation")
        kind = require_string(expectation.get("kind"), f"{label}.expectation.kind")
        if kind not in {"verified-bundle-export", "rejected"}:
            fail(f"{label} has an unsupported expectation")
        keys = {"kind", "bundleVersion"} | ({"diagnosticContains", "outputArtifact"} if kind == "rejected" else set())
        require_exact_keys(expectation, keys, f"{label}.expectation")
        version = expectation["bundleVersion"]
        if type(version) is not int or not 1 <= version <= 6:
            fail(f"{label} has an unsupported bundle version")
        if kind == "rejected":
            diagnostic = require_string(expectation["diagnosticContains"], f"{label}.diagnosticContains")
            if len(diagnostic) > 512 or expectation["outputArtifact"] != "absent":
                fail(f"{label} requires an exact refusal and absent artifact")
        validate_compiler_input_data(
            repo_root, {**shared, "features": row["features"], "kernelSymbols": [symbol]}, label, cache,
            feature_scoped_includes=True,
        )
    if observed != expected:
        fail(f"{label} must cover every displayed declaration in exact fragment/source order")
    require_digest(item["contractSha256"], f"{label}.contractSha256")
    if item["contractSha256"] != source_item_contract_sha256(lesson_id, tab):
        fail(f"{label} contract digest is stale")
    return tab["sourcePath"]


def validate_curriculum_tab(tab: Any, ordinal: int, lesson: str, role: str, source_items: bool = False) -> None:
    label = f"curriculum {lesson} tab {ordinal}"
    require_exact_keys(require_object(tab, label), CURRICULUM_TAB_KEYS, label)
    if type(tab["ordinal"]) is not int or tab["ordinal"] != ordinal:
        fail(f"{label} must retain its ordered ordinal")
    kind = require_string(tab["kind"], f"{label}.kind")
    if kind not in {"kernel", "reference", "spec", "verus", "comparison", "host", "result", "performance"}:
        fail(f"{label} has an unknown kind")
    language = require_string(tab["language"], f"{label}.language")
    if language not in {"rust", "bash", "cpp", "python", "text"}:
        fail(f"{label} has an unknown language")
    if len(require_string(tab["label"], f"{label}.label")) > 256:
        fail(f"{label} label exceeds its bound")
    size = tab["displayedUtf8Bytes"]
    if type(size) is not int or not 0 <= size <= MAX_ATTRIBUTED_SOURCE_BYTES:
        fail(f"{label} displayed bytes exceed their bound")
    require_digest(tab["displayedSha256"], f"{label}.displayedSha256")
    for key, length in (("sourceCommit", 40), ("sourceSha256", 64)):
        if tab[key] is not None:
            require_digest(tab[key], f"{label}.{key}", length)
    if tab["sourcePath"] is not None:
        expected_path(tab["sourcePath"], f"{label}.sourcePath")
    scope = tab["sourceDigestScope"]
    if scope is not None and require_string(scope, f"{label}.sourceDigestScope") not in {"file", "displayed"}:
        fail(f"{label} has an unknown source digest scope")
    if tab["sourceDigestScope"] == "file" and any(
        tab[key] is None for key in ("sourcePath", "sourceCommit", "sourceSha256")
    ):
        fail(f"{label} has incomplete whole-file source metadata")
    fragments = tab["sourceFragmentsSha256"]
    if fragments is not None:
        for digest in bounded_list(fragments, f"{label}.sourceFragmentsSha256", 64):
            require_digest(digest, f"{label}.sourceFragmentsSha256")
    if tab["explanatory"] is not None and type(tab["explanatory"]) is not bool:
        fail(f"{label}.explanatory must be Boolean or null")
    if tab["evidenceId"] is not None:
        require_string(tab["evidenceId"], f"{label}.evidenceId")
    requires_item = role == "executable" and tab["kind"] == "kernel" and tab["language"] == "rust"
    if source_items and requires_item and tab["sourceItem"] is not None:
        if tab["sourceItemStatus"] != "contract-bound":
            fail(f"{label} source item is a contract, not qualification")
        return
    expected_status = "pending" if requires_item else "not-applicable"
    if tab["sourceItem"] is not None or tab["sourceItemStatus"] != expected_status:
        fail(f"{label} must retain its {expected_status} source-item obligation; metadata is not compiler custody")


def validate_curriculum(
    curriculum: Any, entries: dict[str, Any], fixtures: dict[str, Any],
    repo_root: Path, cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    require_exact_keys(require_object(curriculum, "curriculum"), CURRICULUM_KEYS, "curriculum")
    schema = require_string(curriculum["schema"], "curriculum.schema")
    if schema not in {CURRICULUM_SCHEMA, CURRICULUM_SCHEMA_V2} or curriculum["status"] != "pending":
        fail("curriculum must use the pending obligation schema, not qualification")
    source_items = schema == CURRICULUM_SCHEMA_V2
    if cache is None:
        cache = {}
    site = require_object(curriculum["site"], "curriculum.site")
    require_exact_keys(site, {"repository", "commit", "tree"}, "curriculum.site")
    if site["repository"] != "harsh-nod/fe2o3-kernels":
        fail("curriculum.site must identify the tutorial repository")
    require_digest(site["commit"], "curriculum.site.commit", 40)
    require_digest(site["tree"], "curriculum.site.tree", 40)
    seen: set[str] = set()
    associated: set[str] = set()
    tabs = 0
    mixed = False
    gaps: dict[str, list[str]] = {}
    source_paths = {
        entry_id: set(entry["sourcePaths"]) | {
            path for fixture_id in entry["compilerFixtureIds"]
            for path in fixtures[fixture_id]["compilerInput"]["sourcePaths"]
        }
        for entry_id, entry in entries.items()
    }
    for lesson in bounded_list(curriculum["lessons"], "curriculum.lessons", 256):
        require_exact_keys(require_object(lesson, "curriculum lesson"), CURRICULUM_LESSON_KEYS, "curriculum lesson")
        lesson_id = require_string(lesson["lessonId"], "curriculum.lessonId")
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", lesson_id) is None or lesson_id in seen:
            fail("duplicate or invalid curriculum lessonId")
        seen.add(lesson_id)
        role = require_string(lesson["role"], f"{lesson_id}.role")
        if role not in {"executable", "conceptual"}:
            fail(f"{lesson_id} has an unknown curriculum role")
        require_string(lesson["roleReason"], f"{lesson_id}.roleReason")
        ids = require_string_array(lesson["sourceEntryIds"], f"{lesson_id}.sourceEntryIds")
        if len(ids) != len(set(ids)) or not set(ids) <= entries.keys():
            fail(f"{lesson_id} has duplicate or unknown sourceEntryIds")
        associated.update(ids)
        if lesson_id in entries and (role != "executable" or lesson_id not in ids):
            fail(f"{lesson_id} cannot downgrade or detach its existing compiler source entry")
        variants = lesson["variants"]
        if not isinstance(variants, list) or len(variants) > 3:
            fail(f"{lesson_id} variants must be a bounded array")
        kinds = []
        for variant in variants:
            require_exact_keys(require_object(variant, "variant"), {"kind", "status", "sourceItems", "reason"}, "variant")
            kinds.append(variant["kind"])
            if variant["status"] != "pending" or variant["sourceItems"] != []:
                fail(f"{lesson_id} variants are pending obligations, not source implementations or qualification")
            require_string(variant["reason"], f"{lesson_id}.variant.reason")
        if role == "conceptual":
            if ids or variants or lesson["sourceBindingGap"] is not None:
                fail(f"{lesson_id} conceptual lesson cannot carry runnable obligations")
        else:
            if kinds not in (["simt", "tile"], ["simt", "tile", "mixed"]):
                fail(f"{lesson_id} must retain ordered SIMT/tile and any mixed obligations")
            mixed |= "mixed" in kinds
        kernel_paths: set[str] = set()
        bound_paths: set[str] = set()
        for ordinal, tab in enumerate(bounded_list(lesson["codeTabs"], f"{lesson_id}.codeTabs", 64)):
            tabs += 1
            if tabs > MAX_CURRICULUM_TABS:
                fail("curriculum code-tab count exceeds its bound")
            validate_curriculum_tab(tab, ordinal, lesson_id, role, source_items)
            if source_items and tab["sourceItem"] is not None:
                bound_paths.add(validate_source_item(repo_root, lesson_id, tab, cache))
            if tab["kind"] == "kernel" and tab["language"] == "rust" and tab["sourcePath"] is not None:
                kernel_paths.add(tab["sourcePath"])
        if role == "executable":
            # Retain old same-ID obligations even when historical displayed paths
            # differ. New aliases need an actual selected-source path association;
            # package membership alone does not establish feature/symbol coverage.
            for entry_id in ids:
                if entry_id != lesson_id and not kernel_paths & source_paths[entry_id]:
                    fail(f"{lesson_id} sourceEntryId {entry_id} has no matching pinned kernel source path")
            linked_paths = bound_paths.union(*(source_paths[entry_id] for entry_id in ids))
            missing_paths = kernel_paths - linked_paths
            if (not ids and not bound_paths) or missing_paths:
                require_string(
                    lesson["sourceBindingGap"],
                    f"{lesson_id}.sourceBindingGap for unmatched paths {sorted(missing_paths)}",
                )
                gaps[lesson_id] = sorted(missing_paths)
            elif lesson["sourceBindingGap"] is not None:
                fail(f"{lesson_id} fully linked paths cannot have a missing-binding claim")
    if associated != entries.keys():
        fail("curriculum must retain every existing compiler source entry")
    if not mixed:
        fail("curriculum must retain a mixed SIMT/tile showcase obligation")
    return gaps


def validate_site_inventory(curriculum: Any, inventory: Any) -> None:
    """Compare actual ordered display data; site-only CI changes need no repin."""
    require_object(inventory, "site inventory")
    if inventory.get("schema") != SITE_INVENTORY_SCHEMA:
        fail("site inventory has an unsupported projection schema")
    site = require_object(inventory.get("site"), "site inventory.site")
    if site.get("repository") != curriculum["site"]["repository"]:
        fail("site inventory repository differs from the curriculum")
    for key in ("commit", "tree"):
        require_digest(site.get(key), f"site inventory.{key}", 40)
    lessons = bounded_list(inventory.get("lessons"), "site inventory.lessons", 256)
    if [lesson.get("id") for lesson in lessons if isinstance(lesson, dict)] != [
        lesson["lessonId"] for lesson in curriculum["lessons"]
    ] or not all(isinstance(lesson, dict) for lesson in lessons):
        fail("site inventory ordered lesson coverage differs from the curriculum")
    for expected, actual in zip(curriculum["lessons"], lessons, strict=True):
        tabs = bounded_list(actual.get("codeTabs"), "site inventory.codeTabs", 64)
        if len(tabs) != len(expected["codeTabs"]):
            fail(f"site inventory {expected['lessonId']} code-tab coverage differs")
        for retained, tab in zip(expected["codeTabs"], tabs, strict=True):
            require_object(tab, "site inventory tab")
            required = set(CURRICULUM_TAB_FIELDS) - {"sourceFragmentsSha256"}
            if not required <= tab.keys() or "sourceFragments" not in tab:
                fail("site inventory tab is missing explicit source/display fields")
            code = tab.get("displayedCode")
            if not isinstance(code, str) or len(code) > MAX_ATTRIBUTED_SOURCE_BYTES:
                fail("site inventory displayed code is missing or exceeds its bound")
            try:
                encoded = code.encode("utf-8")
            except UnicodeError:
                fail("site inventory displayed code is not valid UTF-8")
            if len(encoded) > MAX_ATTRIBUTED_SOURCE_BYTES:
                fail("site inventory displayed UTF-8 exceeds its bound")
            if type(tab["displayedUtf8Bytes"]) is not int or len(encoded) != tab["displayedUtf8Bytes"] or hashlib.sha256(encoded).hexdigest() != tab["displayedSha256"]:
                fail("site inventory displayed bytes do not match their digest/length")
            fragments = tab["sourceFragments"]
            if fragments is not None:
                fragments = bounded_list(fragments, "site inventory.sourceFragments", 64)
                if any(not isinstance(value, str) for value in fragments):
                    fail("site inventory source fragments must be strings")
                try:
                    fragment_bytes = [value.encode("utf-8") for value in fragments]
                except UnicodeError:
                    fail("site inventory source fragments are not valid UTF-8")
                if any(len(value) > MAX_ATTRIBUTED_SOURCE_BYTES for value in fragment_bytes):
                    fail("site inventory source fragments exceed their bound")
            projection = {key: tab[key] for key in required}
            projection["sourceFragmentsSha256"] = None if fragments is None else [
                hashlib.sha256(value).hexdigest() for value in fragment_bytes
            ]
            # JSON comparison retains Boolean/integer distinctions unlike Python equality.
            if json.dumps(projection, sort_keys=True) != json.dumps(
                {key: retained[key] for key in CURRICULUM_TAB_FIELDS}, sort_keys=True
            ):
                fail(f"site inventory {expected['lessonId']} tab {retained['ordinal']} source/display binding differs")


def validate_manifest(
    repo_root: Path, manifest: Any, *, curriculum_gaps: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    require_object(manifest, "manifest")
    keys = TOP_LEVEL_KEYS | ({"curriculum", "kernelInventory"} & manifest.keys())
    require_exact_keys(manifest, keys, "manifest")
    if manifest["schema"] != "fe2o3-tutorial-kernel-source-contract-v1":
        fail("expected a source-contract schema, not a release or evidence manifest")
    if manifest["roadmapIssue"] != "https://github.com/harsh-nod/fe2o3/issues/271":
        fail("unexpected roadmap issue")
    if manifest["baseline"] != {
        "compilerCommit": None,
        "compilerTree": None,
        "status": "unqualified-source-contract",
    }:
        fail("source-content pins cannot set an accepted compiler baseline")
    recovery = require_object(manifest["recovery"], "recovery")
    require_exact_keys(
        recovery,
        {"donorCompilerCommit", "donorManifestSha256", "sourceBaseCommit", "sourceState"},
        "recovery",
    )
    for key in ("donorCompilerCommit", "sourceBaseCommit"):
        if not isinstance(recovery[key], str) or re.fullmatch(r"[0-9a-f]{40}", recovery[key]) is None:
            fail(f"recovery.{key} is not a commit name")
    require_digest(recovery["donorManifestSha256"], "recovery.donorManifestSha256")
    if recovery["sourceState"] != "dirty-proposal-not-qualified":
        fail("recovery is provenance, not a qualification receipt")
    if manifest["productionContract"] != {
        "pipelineEntry": "rustc-codegen-fe2o3::production_pipeline",
        "requiredPolicyVersion": 4,
        "requiresFinalOptimizedGraphVerification": True,
        "allowsPipelineSelection": False,
        "allowsFallback": False,
    }:
        fail("the sole production pipeline, policy and final-verification obligations are fixed")
    fixtures: dict[str, Any] = {}
    cases = set()
    cache: dict[str, dict[str, Any]] = {}
    for fixture in bounded_list(manifest["compilerFixtures"], "compilerFixtures", 256):
        require_exact_keys(require_object(fixture, "fixture"), FIXTURE_KEYS, "fixture")
        fixture_id = require_string(fixture["fixtureId"], "fixtureId")
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", fixture_id) is None or fixture_id in fixtures:
            fail("duplicate or invalid fixtureId")
        target = require_target(fixture["target"], "fixture target")
        validate_matrix(repo_root, fixture, fixture_id)
        case = (target, fixture["matrix"]["caseId"])
        if case in cases:
            fail("duplicate target/caseId")
        cases.add(case)
        if fixture["testId"] != f"kernel-compile-matrix/{target}/{case[1]}":
            fail(f"{fixture_id} testId does not identify its matrix case")
        if fixture["testPath"] != "scripts/tests/kernel-compile-matrix.sh":
            fail(f"{fixture_id} must retain the compiler-owned matrix harness")
        checked_path(repo_root, fixture["testPath"], "fixture.testPath")
        validate_compiler_input(repo_root, fixture, fixture_id, cache)
        validate_pending_simulation(fixture["simulation"], f"{fixture_id}.simulation")
        fixtures[fixture_id] = fixture
    entries: dict[str, Any] = {}
    associated = set()
    package_kernels: dict[str, bool] = {}
    for entry in bounded_list(manifest["entries"], "entries", 256):
        require_exact_keys(require_object(entry, "entry"), ENTRY_KEYS, "entry")
        lesson = require_string(entry["lessonId"], "lessonId")
        if lesson in entries:
            fail("duplicate lessonId")
        evidence = entry["siteEvidenceKind"]
        if evidence not in ALLOWED_EVIDENCE:
            fail(f"{lesson} has an unknown site evidence kind")
        classification = entry["classification"]
        if classification not in {"compiler-produced", "simulator-only", "design-only", "external-baseline"}:
            fail(f"{lesson} has an unknown classification")
        if evidence in COMPILER_EVIDENCE and classification != "compiler-produced":
            fail(f"{lesson} runnable compiler obligations cannot be downgraded")
        package = checked_path(repo_root, entry["packageManifest"], f"{lesson}.packageManifest")
        sources = require_string_list(entry["sourcePaths"], f"{lesson}.sourcePaths")
        has_kernel = False
        for source in sources:
            path = _validate_package_source_path(repo_root, (repo_root / package).parent, source, lesson)
            if path.stat().st_size > MAX_ATTRIBUTED_SOURCE_BYTES:
                fail(f"{lesson} attributed source exceeds its bound")
            has_kernel |= source_contains_ordinary_attributed_kernel(path.read_text(encoding="utf-8"))
        has_kernel |= _package_has_ordinary_kernel(
            repo_root, package, lesson, cache, package_kernels
        )
        if has_kernel and classification != "compiler-produced":
            fail(f"{lesson} ordinary kernel cannot be downgraded")
        ids = require_string_array(entry["compilerFixtureIds"], f"{lesson}.compilerFixtureIds")
        gates = require_string_list(entry["requiredGates"], f"{lesson}.requiredGates")
        if not set(gates) <= ALLOWED_GATES:
            fail(f"{lesson} has an unknown gate")
        if classification == "compiler-produced":
            if not ids or not {"production-compile", "semantic-simulation"} <= set(gates):
                fail(f"{lesson} must retain compile and simulation obligations")
            if not any(fixtures.get(i, {}).get("compilerInput", {}).get("packageManifest") == package for i in ids):
                fail(f"{lesson} has no fixture for its ordinary-source package")
        for fixture_id in ids:
            if fixture_id not in fixtures:
                fail(f"{lesson} references an unknown fixture")
            associated.add(fixture_id)
        entries[lesson] = entry
    if associated != set(fixtures):
        fail("every compiler fixture must retain a lesson/scope association")
    validate_qualification(manifest["qualification"], entries, fixtures)
    if "curriculum" in manifest:
        gaps = validate_curriculum(manifest["curriculum"], entries, fixtures, repo_root, cache)
        if curriculum_gaps is not None:
            curriculum_gaps.update(gaps)
    validate_kernel_inventory(manifest, None, repo_root=repo_root, package_cache=cache)
    return fixtures


def _package_has_ordinary_kernel(
    repo_root: Path,
    package: str,
    label: str,
    compiler_packages: dict[str, dict[str, Any]],
    package_kernels: dict[str, bool],
) -> bool:
    # One invocation observes one package snapshot. Entry sourcePaths are still
    # independently checked and scanned; this is not atomic filesystem admission.
    if package in package_kernels:
        return package_kernels[package]
    cached = compiler_packages.get(package)
    package_root = (repo_root / package).parent
    if cached is None:
        sources = package_rust_sources(repo_root, package, label)
        attributed_names: dict[str, list[str]] = {}
    else:
        sources = cached["packageSources"]
        attributed_names = cached["attributedNames"]
    has_kernel = False
    for path, text in sources:
        key = path.relative_to(package_root).as_posix()
        names = attributed_names.get(key)
        if names is None:
            names = ordinary_attributed_kernel_names(text)
            attributed_names[key] = names
        if names:
            has_kernel = True
            break
    package_kernels[package] = has_kernel
    return has_kernel


def reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        fail(f"non-finite JSON number: {value}")
    return number


def load_manifest(
    path: Path, maximum_bytes: int = MAX_CARGO_MANIFEST_BYTES, *, with_sha256: bool = False,
) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        fail("manifest must be a bounded regular file")
    try:
        with path.open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes:
            fail("manifest exceeds byte bound")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_json_object,
            parse_float=finite_json_float,
            parse_constant=lambda value: fail(f"non-finite JSON constant: {value}"),
        )
        return (value, hashlib.sha256(raw).hexdigest()) if with_sha256 else value
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        fail(f"cannot read manifest: {error}")


def _kernel_pair_report(
    manifest: dict[str, Any], fixtures: dict[str, Any],
    gaps: dict[str, list[str]], inventory: dict[str, Any] | None,
    *, repo_root: Path | None = None,
) -> dict[str, Any]:
    """Project already validated contracts without inferring variant custody."""
    curriculum = manifest.get("curriculum")
    if curriculum is None or curriculum["schema"] != CURRICULUM_SCHEMA_V2:
        fail("kernel-pair reporting requires the V2 source-item curriculum")
    entries = {entry["lessonId"]: entry for entry in manifest["entries"]}
    lessons = curriculum["lessons"]
    fixture_entries: dict[str, list[str]] = {key: [] for key in fixtures}
    for entry in entries.values():
        for fixture_id in entry["compilerFixtureIds"]:
            fixture_entries[fixture_id].append(entry["lessonId"])

    def selection(inputs: dict[str, Any], symbol: str, target: str) -> dict[str, Any]:
        return {
            **{key: inputs[key] for key in (
                "packageManifest", "cargoTarget", "defaultFeatures", "features", "sourcePaths",
            )},
            "kernelSymbol": symbol, "target": target,
        }

    records = 0

    def append(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
        nonlocal records
        if records >= MAX_KERNEL_PAIR_RECORDS:
            fail("kernel-pair report exceeds its record bound")
        records += 1
        rows.append(row)

    fixture_rows: list[dict[str, Any]] = []
    for fixture_id, fixture in sorted(fixtures.items()):
        inputs = fixture["compilerInput"]
        scope_entries = sorted(fixture_entries[fixture_id])
        scope_lessons = sorted(
            lesson["lessonId"] for lesson in lessons
            if set(lesson["sourceEntryIds"]) & set(scope_entries)
        )
        for symbol in inputs["kernelSymbols"]:
            append(fixture_rows, {
                "fixtureId": fixture_id,
                "compilerInputContractSha256": inputs["contractSha256"],
                "selection": selection(inputs, symbol, fixture["target"]),
                "sourceEntryIds": scope_entries,
                "scopeLessonIds": scope_lessons,
            })

    runtime = {} if inventory is None else {
        lesson["id"]: lesson for lesson in inventory["lessons"]
    }
    source_cases: list[dict[str, Any]] = []
    displays: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    lexical_count = 0
    for lesson in lessons:
        if lesson["role"] != "executable":
            continue
        lesson_id = lesson["lessonId"]
        requirements.append({
            "lessonId": lesson_id,
            "modes": [variant["kind"] for variant in lesson["variants"]],
            "status": "pending",
        })
        for tab in lesson["codeTabs"]:
            if tab["kind"] != "kernel" or tab["language"] != "rust":
                continue
            names = None
            if inventory is not None:
                names = ordinary_attributed_kernel_names(
                    runtime[lesson_id]["codeTabs"][tab["ordinal"]]["displayedCode"]
                )
                lexical_count += len(names)
                if lexical_count > MAX_KERNEL_PAIR_RECORDS:
                    fail("kernel-pair report exceeds its lexical declaration bound")
            append(displays, {
                "lessonId": lesson_id, "tabOrdinal": tab["ordinal"],
                "sourcePath": tab["sourcePath"], "displayedSha256": tab["displayedSha256"],
                "sourceItemStatus": tab["sourceItemStatus"], "lexicalKernelNames": names,
            })
            item = tab["sourceItem"]
            if item is None:
                continue
            for ordinal, case in enumerate(item["cases"]):
                append(source_cases, {
                    "lessonId": lesson_id, "tabOrdinal": tab["ordinal"], "caseOrdinal": ordinal,
                    "sourceItemContractSha256": item["contractSha256"],
                    "selection": selection(
                        {**item["compilerInput"], "features": case["features"]},
                        case["kernelSymbol"], case["target"],
                    ),
                    "displayedFragmentOrdinal": case["displayedFragmentOrdinal"],
                    "driver": item["driver"], "testFunction": case["testFunction"],
                    "expectation": case["expectation"],
                })
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    report = {
        "schema": "fe2o3-tutorial-kernel-pair-obligations-v1",
        "sourceContractSha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
        "qualified": False, "inventoryComplete": False,
        "requiredPairCount": None, "qualifiedPairCount": 0,
        "requiredModes": ["simt", "tile"],
        "variantBindingStatus": "pending", "stageStatus": "not-evaluated",
        "missingBindings": ["per-kernel-variant-sources", "per-variant-target-evidence", "exhaustive-kernel-identity"],
        "productionContract": manifest["productionContract"],
        "curriculumSite": curriculum["site"],
        "runtimeProjectionSite": None if inventory is None else {
            key: inventory["site"][key] for key in ("repository", "commit", "tree")
        },
        "lessonRequirements": requirements,
        "fixtureSelections": fixture_rows,
        "sourceDriverCases": source_cases,
        "displayObservations": displays,
        "sourceBindingGaps": gaps,
    }
    identities = validate_kernel_inventory(
        manifest, inventory, max_records=MAX_KERNEL_PAIR_RECORDS - records, repo_root=repo_root,
    )
    if identities is not None:
        report.update(
            schema="fe2o3-tutorial-kernel-pair-obligations-v2",
            inventoryComplete=identities["inventoryComplete"],
            requiredPairCount=identities["requiredPairCount"],
            knownVariantObligationCount=identities["knownVariantObligationCount"],
            pendingVariantCount=identities["pendingVariantCount"],
            unregisteredDisplayItemCount=identities["unregisteredDisplayItemCount"],
            variantBindingStatus=identities["variantBindingStatus"],
            sourceBoundVariantCount=identities["sourceBoundVariantCount"],
            sourceBoundPairCount=identities["sourceBoundPairCount"],
            kernelInventory=identities,
        )
        if identities["variantBindingStatus"] == "source-bound":
            report["missingBindings"].remove("per-kernel-variant-sources")
        if identities["inventoryComplete"]:
            report["missingBindings"].remove("exhaustive-kernel-identity")
    return report


def _encode_kernel_pair_report(report: dict[str, Any]) -> str:
    chunks = []
    size = 0
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False).iterencode(report):
        size += len(chunk)
        if size > MAX_SITE_INVENTORY_BYTES:
            fail("kernel-pair report exceeds its output byte bound")
        chunks.append(chunk)
    return "".join(chunks)


def _bind_ordinary_source_report(
    projection: dict[str, Any], fixtures: dict[str, Any], corpus: Any, manifest_sha256: str,
) -> None:
    """Bind diagnostic fixture observations, never kernel qualification or authority."""
    def canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, allow_nan=False)

    corpus = require_object(corpus, "ordinary-source report")
    header = {
        "schema": "fe2o3-ordinary-source-policy4-extraction-corpus-v1",
        "manifest_sha256": manifest_sha256,
        "configurations": len(fixtures),
        "distinct_expected_roots": len({
            symbol for fixture in fixtures.values()
            for symbol in fixture["compilerInput"]["kernelSymbols"]
        }),
        "default_pipeline_activated": False,
        "grants_artifact_or_launch_authority": False,
    }
    require_exact_keys(corpus, set(header) | {"cases", "all_checked_output_passed"}, "ordinary-source report")
    if any(canonical(corpus[key]) != canonical(value) for key, value in header.items()):
        fail("ordinary-source report header differs from manifest or diagnostic scope")
    cases = corpus["cases"]
    if not isinstance(cases, list) or len(cases) != len(fixtures):
        fail("ordinary-source report requires the complete fixture roster")
    indices = {}
    for index, value in enumerate(cases):
        case = require_object(value, "ordinary-source case")
        actual = require_object(case.get("fixture"), "ordinary-source fixture")
        key = require_string(actual.get("fixtureId"), "ordinary-source fixtureId")
        if key in indices or key not in fixtures:
            fail("duplicate or unknown ordinary-source fixture")
        expected = {
            "fixtureId": key, "target": fixtures[key]["target"],
            "compilerInput": {k: v for k, v in fixtures[key]["compilerInput"].items() if k != "contractSha256"},
        }
        if canonical(actual) != canonical(expected):
            fail("ordinary-source fixture/input/target differs from manifest")
        if not {"status", "observation", "refusal", "compiler_artifacts"} <= case.keys():
            fail("ordinary-source case is incomplete")
        if not isinstance(case["compiler_artifacts"], list) or not all(
            isinstance(path, str) for path in case["compiler_artifacts"]
        ):
            fail("ordinary-source artifact observations must be strings")
        for field in ("observation", "refusal", "callback_progress"):
            if case.get(field) is not None and not isinstance(case[field], dict):
                fail(f"ordinary-source {field} must be an object or null")
        passed = (isinstance(case["observation"], dict)
                  and case["refusal"] is None and case["compiler_artifacts"] == [])
        if (case["status"] not in ("blocked", "checked-output-pass")
                or (case["status"] == "checked-output-pass") != passed
                or (case["status"] == "blocked" and not isinstance(case["refusal"], dict))):
            fail("ordinary-source case status contradicts its outcome")
        indices[key] = index
    if corpus["all_checked_output_passed"] is not all(case["status"] == "checked-output-pass" for case in cases):
        fail("ordinary-source aggregate contradicts its cases")
    for row in projection["fixtureSelections"]:
        row["ordinarySourceCaseIndex"] = indices[row["fixtureId"]]
    projection.update(
        stageStatus="fixture-source-observations-bound",
        ordinarySourceObservations={"diagnosticOnly": True, "report": corpus},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--emit-matrix", choices=sorted(ALLOWED_TARGETS))
    output.add_argument("--emit-reproduction-plan", choices=sorted(ALLOWED_TARGETS),
                        help="emit a static target-specific view of existing fixture and suite contracts; never execute")
    output.add_argument("--emit-kernel-pairs", action="store_true")
    output.add_argument("--emit-source-input-snapshot", type=Path,
                        help="snapshot one exact registered invocation for the ordinary-source corpus runner")
    parser.add_argument("--require-qualified", action="store_true")
    parser.add_argument("--require-curriculum", action="store_true")
    parser.add_argument("--site-inventory", type=Path)
    parser.add_argument("--ordinary-source-report", type=Path)
    parser.add_argument("--source-driver-report", type=Path)
    arguments = parser.parse_args()
    if arguments.ordinary_source_report and not arguments.emit_kernel_pairs:
        fail("--ordinary-source-report requires --emit-kernel-pairs")
    if arguments.source_driver_report and not arguments.emit_kernel_pairs:
        fail("--source-driver-report requires --emit-kernel-pairs")
    census_module = None
    if arguments.emit_source_input_snapshot or arguments.ordinary_source_report or arguments.source_driver_report:
        specification = importlib.util.spec_from_file_location(
            "tutorial_source_census", Path(__file__).with_name("tutorial_source_census.py"))
        assert specification is not None and specification.loader is not None
        census_module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(census_module)
    root = arguments.repo_root.resolve()
    manifest, manifest_sha256 = load_manifest(
        arguments.manifest or root / "config/tutorial-kernel-manifest-v1.json", with_sha256=True,
    )
    curriculum_gaps: dict[str, list[str]] = {}
    fixtures = validate_manifest(root, manifest, curriculum_gaps=curriculum_gaps)
    if arguments.require_curriculum or arguments.site_inventory or arguments.emit_kernel_pairs:
        if "curriculum" not in manifest:
            fail("the exhaustive curriculum extension is required")
    inventory = None
    if arguments.site_inventory:
        inventory = load_manifest(arguments.site_inventory, MAX_SITE_INVENTORY_BYTES)
        validate_site_inventory(manifest["curriculum"], inventory)
        if not arguments.emit_kernel_pairs:
            validate_kernel_inventory(manifest, inventory, repo_root=root)
    if arguments.require_qualified:
        fail("qualification receipts and policy/final-graph evidence are not implemented by source contracts")
    if arguments.emit_source_input_snapshot:
        import sys
        try:
            snapshot = census_module.input_snapshot(
                sys.modules[__name__], root, manifest, manifest_sha256,
                load_manifest(arguments.emit_source_input_snapshot))
        except census_module.CensusError as error:
            fail(str(error))
        if sha256_file(arguments.manifest or root / "config/tutorial-kernel-manifest-v1.json",
                       MAX_CARGO_MANIFEST_BYTES, "source census manifest") != manifest_sha256:
            fail("source census manifest changed during input validation")
        print(_encode_kernel_pair_report(snapshot))
    elif arguments.emit_kernel_pairs:
        report = _kernel_pair_report(manifest, fixtures, curriculum_gaps, inventory, repo_root=root)
        corpus = None
        if arguments.ordinary_source_report:
            corpus = load_manifest(arguments.ordinary_source_report, MAX_SITE_INVENTORY_BYTES)
            _bind_ordinary_source_report(report, fixtures, corpus, manifest_sha256)
        if arguments.source_driver_report or (corpus is not None and any(
                "source_census" in case for case in corpus["cases"])):
            import sys
            source_drivers = (load_manifest(arguments.source_driver_report, MAX_SITE_INVENTORY_BYTES)
                              if arguments.source_driver_report else None)
            if inventory is None or "kernelInventory" not in report:
                fail("source census corpus joins require --site-inventory and the identity extension")
            try:
                report["sourceCensusObservations"] = census_module.join(
                    sys.modules[__name__], root, manifest, manifest_sha256, inventory,
                    report["kernelInventory"], [corpus, source_drivers])
            except census_module.CensusError as error:
                fail(str(error))
        print(_encode_kernel_pair_report(report))
    elif arguments.emit_reproduction_plan:
        specification = importlib.util.spec_from_file_location(
            "tutorial_reproduction_plan", Path(__file__).with_name("tutorial_reproduction_plan.py"))
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        try:
            plan = module.build_plan(manifest, fixtures, arguments.emit_reproduction_plan, manifest_sha256)
        except ValueError as error:
            fail(str(error))
        print(_encode_kernel_pair_report(plan))
    elif arguments.emit_matrix:
        records = [fixture for fixture in fixtures.values() if fixture["target"] == arguments.emit_matrix]
        if not records:
            fail("selected target has no compiler fixture obligations")
        for fixture in sorted(records, key=lambda fixture: fixture["matrix"]["caseId"]):
            print(matrix_record(fixture))
    else:
        print(
            f"SOURCE CONTRACT VALID fixtures={len(fixtures)} qualified=false "
            "compiler_inputs=expected semantic_oracles=pending policy_verification=pending"
        )
        for lesson_id, paths in sorted(curriculum_gaps.items()):
            print(f"CURRICULUM SOURCE BINDING PENDING lesson={lesson_id} unmatchedPaths={json.dumps(paths)}")


if __name__ == "__main__":
    main()

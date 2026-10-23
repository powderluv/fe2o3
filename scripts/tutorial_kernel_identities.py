"""Bounded identity reconciliation for an already validated tutorial manifest.

This module owns no source inventory, execution evidence, or file I/O. The caller
validates the existing manifest/runtime projection first and supplies its bounded
Rust scanner and lexer. Restricted fixture selection uses those same coordinates.
References retain those input contracts; a lexical occurrence is never compiler
or qualification evidence. A source-bound variant declares an association with
an independently registered implementation identity, not semantic equivalence
or verification of its programming mode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Callable


SCHEMA = "fe2o3-tutorial-kernel-identities-v1"
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_BYTES = 16 * 1024 * 1024
MAX_IDENTITY_BYTES = 16 * 1024 * 1024
MAX_CFG_TOKENS = 512
MAX_INCLUDE_DEPTH = 32
# A catalog can revisit immutable item coordinates under many distinct features.
# Bound that work separately from newly materialized records and attributes:
# the current 68-fixture catalog needs 4,324 visits but only 1,651 records.
MAX_SOURCE_ITEM_VISITS = 8192
# Exported report rows reduce record capacity, not physical attribute work.
# The complete 95-fixture catalog exceeds 4,096 attribute visits. Keep a finite
# independent work bound without changing record, source-byte or item limits.
MAX_ATTRIBUTE_VISITS = 8192
BUILTIN_DERIVES = frozenset({"Clone", "Copy", "Debug", "Default", "Eq", "PartialEq", "Ord", "PartialOrd", "Hash"})
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
KERNEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
ISSUE_URL = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
INPUT_FIELDS = {
    "packageManifest", "packageManifestSha256", "cargoLockPath", "cargoLockSha256",
    "sourcePaths", "sourceClosureSha256", "cargoTarget", "defaultFeatures", "features",
}
DISPLAY_FIELDS = {
    "lessonId", "tabOrdinal", "functionUtf8Offset", "kernelSymbol", "classification",
    "kernelIds", "negativeCases", "bindingStatus", "reason",
}


class KernelInventoryError(ValueError):
    """An identity contract is malformed, stale, or incompletely partitioned."""


def _fail(message: str) -> None:
    raise KernelInventoryError(f"kernel inventory: {message}")


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{label} has unexpected or missing fields")
    return value


def _text(value: Any, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _fail(f"{label} must be bounded nonblank text")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _symbol(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 256 or IDENTIFIER.fullmatch(value) is None:
        _fail("kernelSymbol must be an ordinary Rust identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        _fail(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def _display_symbol(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 256 or not value.isidentifier():
        _fail("display kernelSymbol must be a Rust name without its raw prefix")
    return value


def _name_token_matches(encoded: bytes, offset: int, symbol: str) -> bool:
    token = symbol.encode("utf-8")
    return encoded[offset:offset + len(token)] == token or encoded[offset:offset + len(token) + 2] == b"r#" + token


class _Budget:
    def __init__(self, maximum: int, *, source_item_limit: int | None = None,
                 attribute_limit: int | None = None):
        if type(maximum) is not int or maximum < 0:
            _fail("record limit must be a nonnegative integer")
        if source_item_limit is not None and (type(source_item_limit) is not int or source_item_limit < 0):
            _fail("source item visit limit must be a nonnegative integer")
        if attribute_limit is not None and (type(attribute_limit) is not int or attribute_limit < 0):
            _fail("attribute visit limit must be a nonnegative integer")
        self.maximum = maximum
        self.attribute_limit = attribute_limit
        self.source_item_limit = source_item_limit
        self.used = 0
        self.attribute_visits = 0
        self.source_item_visits = 0
        self.identity_bytes = 0
        self.source_bytes = 0
        self.fragment_match_bytes = 0

    def rows(self, value: Any, label: str) -> list[Any]:
        if not isinstance(value, list) or len(value) > self.maximum - self.used:
            _fail(f"{label} exceeds the record bound or is not an array")
        self.used += len(value)
        return value

    def fragment_search(self, source_bytes: int) -> None:
        # Account for both full-source searches, including overlapping matches.
        amount = 2 * source_bytes
        if amount > MAX_RUNTIME_BYTES - self.fragment_match_bytes:
            _fail("fixture fragment matching exceeds its aggregate byte-span bound")
        self.fragment_match_bytes += amount

    def visit_attribute(self) -> None:
        limit = self.maximum if self.attribute_limit is None else self.attribute_limit
        if self.attribute_visits >= limit:
            _fail("fixture selection attributes exceed their aggregate visit bound")
        self.attribute_visits += 1

    def visit_source_item(self) -> None:
        limit = self.maximum if self.source_item_limit is None else self.source_item_limit
        if self.source_item_visits >= limit:
            _fail("fixture source items exceed their aggregate visit bound")
        self.source_item_visits += 1


def _reference(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not isinstance(value, dict):
        _fail("selection reference must be an object")
    kind = value.get("kind")
    if kind == "fixture":
        _object(value, {"kind", "fixtureId", "kernelSymbol"}, "fixture reference")
        fixture = _text(value["fixtureId"], "fixtureId")
        symbol = _symbol(value["kernelSymbol"])
        return (kind, fixture, symbol), {"kind": kind, "fixtureId": fixture, "kernelSymbol": symbol}
    if kind == "source-driver-case":
        _object(value, {"kind", "lessonId", "tabOrdinal", "caseOrdinal"}, "source case reference")
        lesson = _text(value["lessonId"], "lessonId")
        tab = _integer(value["tabOrdinal"], "tabOrdinal")
        case = _integer(value["caseOrdinal"], "caseOrdinal")
        return (kind, lesson, tab, case), {
            "kind": kind, "lessonId": lesson, "tabOrdinal": tab, "caseOrdinal": case,
        }
    _fail("unknown selection reference kind")


def _selection_identity(inputs: dict[str, Any], symbol: str, budget: _Budget) -> str:
    # Only the GPU target and reference location may differ in one identity.
    # Source revisions, closure/lock hashes, feature selections and Cargo target
    # remain significant even when two symbols or rendered bodies are equal.
    try:
        payload = {key: inputs[key] for key in INPUT_FIELDS}
        payload["kernelSymbol"] = symbol
        digest = hashlib.sha256(b"fe2o3-tutorial-kernel-selection-v1\0")
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        for chunk in encoder.iterencode(payload):
            budget.identity_bytes += len(chunk)
            if budget.identity_bytes > MAX_IDENTITY_BYTES:
                _fail("selection identity bytes exceed their aggregate bound")
            digest.update(chunk.encode("ascii"))
        return digest.hexdigest()
    except KernelInventoryError:
        raise
    except (KeyError, TypeError, ValueError):
        _fail("selection lacks a valid existing compiler input contract")


def _utf8(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_TEXT_BYTES:
        _fail(f"{label} is missing or exceeds its byte bound")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        _fail(f"{label} is not valid UTF-8")
    if len(encoded) > MAX_TEXT_BYTES:
        _fail(f"{label} exceeds its byte bound")
    return encoded


def _fragment_intervals(tab: dict[str, Any], runtime: dict[str, Any], encoded: bytes) -> list[tuple[int, int]]:
    if tab["sourceDigestScope"] == "file":
        if runtime.get("sourceFragments") is not None or tab["sourceFragmentsSha256"] is not None:
            _fail("whole-file source case has fragment metadata")
        if tab["sourceSha256"] != hashlib.sha256(encoded).hexdigest():
            _fail("whole-file source case digest differs")
        return [(0, len(encoded))]
    fragments = runtime.get("sourceFragments")
    digests = tab["sourceFragmentsSha256"]
    if not isinstance(fragments, list) or not isinstance(digests, list) or not 0 < len(fragments) == len(digests) <= 64:
        _fail("source case fragment coverage differs")
    parts = [_utf8(part, "source fragment") for part in fragments]
    if sum(map(len, parts)) + 2 * (len(parts) - 1) > MAX_TEXT_BYTES:
        _fail("source case fragment bytes exceed their bound")
    if b"\n\n".join(parts) != encoded:
        _fail("source case fragments do not reconstruct the displayed bytes")
    intervals = []
    offset = 0
    for part, digest in zip(parts, digests, strict=True):
        if hashlib.sha256(part).hexdigest() != digest:
            _fail("source case fragment digest differs")
        intervals.append((offset, offset + len(part)))
        offset += len(part) + 2
    return intervals


def _fixture_display_declaration(tab: dict[str, Any]) -> None:
    scope = tab["sourceDigestScope"]
    digests = tab["sourceFragmentsSha256"]
    if scope == "file":
        if digests is not None:
            _fail("whole-file fixture display cannot carry fragment metadata")
    elif scope == "displayed":
        if _digest(tab["sourceSha256"], "fixture display source digest") != tab["displayedSha256"]:
            _fail("fixture display source and displayed digests differ")
        if digests is not None:
            if not isinstance(digests, list) or not 0 < len(digests) <= 64:
                _fail("fixture display fragment metadata exceeds its bound or is empty")
            for digest in digests:
                _digest(digest, "fixture display fragment digest")
    else:
        _fail("unsupported fixture display digest scope")


class _FixtureDisplayIndex:
    """Exact excerpt/source correspondence, separate from fixture selection."""

    def __init__(self, budget: _Budget):
        self.budget = budget
        self.parts = {}
        self.sources = {}
        self.mappings = {}

    def match(self, location: tuple[str, int], tab: dict[str, Any], live: dict[str, Any],
              path: str, source: str, digest: str, offset: int, symbol: str,
              source_offset: int) -> None:
        if location not in self.parts:
            displayed = _utf8(live.get("displayedCode"), "fixture displayed code")
            if tab["sourceFragmentsSha256"] is None:
                if live.get("sourceFragments") is not None:
                    _fail("contiguous fixture display has unexpected fragments")
                intervals = [(0, len(displayed))]
            else:
                intervals = _fragment_intervals(tab, live, displayed)
            self.budget.rows(intervals, "fixture display fragments")
            parts = [(start, end, displayed[start:end]) for start, end in intervals]
            if any(not part for _, _, part in parts):
                _fail("fixture display fragments must be nonempty")
            self.parts[location] = parts
        parts = self.parts[location]
        source_key = (path, digest)
        if source_key not in self.sources:
            self.sources[source_key] = _utf8(source, "fixture matched source")
        encoded = self.sources[source_key]
        mapping_key = (location, path, digest)
        if mapping_key not in self.mappings:
            self.budget.rows(parts, "fixture physical fragment mappings")
            starts = []
            for _, _, part in parts:
                if len(part) > len(encoded):
                    _fail("fixture display fragment exceeds its physical source")
                self.budget.fragment_search(len(encoded))
                start = encoded.find(part)
                if start < 0:
                    _fail("fixture display fragment differs from its physical source")
                if encoded.find(part, start + 1) >= 0:
                    _fail("fixture display fragment has ambiguous physical occurrences")
                starts.append(start)
            spans = sorted((start, start + len(part))
                           for start, (_, _, part) in zip(starts, parts, strict=True))
            if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
                _fail("fixture display fragments overlap or repeat in physical source")
            self.mappings[mapping_key] = starts
        token = symbol.encode("utf-8")
        matches = []
        for index, (start, end, part) in enumerate(parts):
            relative = offset - start
            if not start <= offset < end:
                continue
            width = len(token) + (2 if part[relative:relative + 2] == b"r#" else 0)
            if offset + width <= end and _name_token_matches(part, relative, symbol):
                matches.append(self.mappings[mapping_key][index] + relative)
        if matches != [source_offset]:
            _fail("fixture display does not map to the exact selected physical function")


def _fixture_cfg(body: str, features: set[str]) -> bool:
    """Evaluate the declared non-test AMDGPU library context, without expansion."""
    if len(body) > 8192 or len(body.encode("utf-8")) > 8192:
        _fail("fixture cfg exceeds its byte bound")
    tokens = re.findall(r'\s+|[A-Za-z_][A-Za-z0-9_]*|"[A-Za-z0-9_-]+"|[(),=]', body)
    if "".join(tokens) != body:
        _fail("unsupported fixture cfg syntax")
    tokens = [token for token in tokens if not token.isspace()]
    if len(tokens) > MAX_CFG_TOKENS:
        _fail("fixture cfg exceeds its token bound")
    cursor = 0

    def take() -> str:
        nonlocal cursor
        if cursor == len(tokens):
            _fail("incomplete fixture cfg")
        token = tokens[cursor]
        cursor += 1
        return token

    def expression(depth: int) -> bool:
        if depth > 32:
            _fail("fixture cfg exceeds its nesting bound")
        name = take()
        if name in {"all", "any", "not"}:
            if take() != "(":
                _fail("unsupported fixture cfg syntax")
            values = []
            while cursor < len(tokens) and tokens[cursor] != ")":
                values.append(expression(depth + 1))
                if cursor < len(tokens) and tokens[cursor] == ",":
                    take()
                elif cursor == len(tokens) or tokens[cursor] != ")":
                    _fail("unsupported fixture cfg syntax")
            if take() != ")" or (name == "not" and len(values) != 1):
                _fail("unsupported fixture cfg arity")
            return all(values) if name == "all" else any(values) if name == "any" else not values[0]
        if name == "test":
            return False
        if name not in {"feature", "target_arch"} or take() != "=":
            _fail("unsupported fixture cfg predicate")
        value = take()
        if not value.startswith('"'):
            _fail("fixture cfg requires a literal value")
        value = value[1:-1]
        return value in features if name == "feature" else value == "amdgpu"

    result = expression(0)
    if cursor != len(tokens):
        _fail("trailing fixture cfg syntax")
    return result


def _fixture_trivia_end(source: str, start: int, end: int) -> int:
    """Skip only comments/whitespace, never literals hidden by the census lexer."""
    cursor = start
    while cursor < end:
        if source[cursor].isspace():
            cursor += 1
        elif source.startswith("//", cursor):
            newline = source.find("\n", cursor, end)
            cursor = end if newline < 0 else newline + 1
        elif source.startswith("/*", cursor):
            depth = 1
            cursor += 2
            while cursor < end and depth:
                if source.startswith(("/*", "*/"), cursor):
                    depth += 1 if source.startswith("/*", cursor) else -1
                    cursor += 2
                else:
                    cursor += 1
            if depth:
                _fail("unterminated fixture attribute comment")
        else:
            break
    return cursor


def _fixture_attribute(source: str, code: str, pairs: dict[int, int], start: int, end: int,
                       features: set[str], budget: _Budget, inner: bool,
                       depth: int = 0, derives: set[str] | None = None,
                       module_paths: list[str] | None = None) -> tuple[bool, int]:
    """Evaluate a bounded attribute subset, separately from the lexical census."""
    if depth > 32:
        _fail("fixture selection attribute exceeds its nesting bound")
    budget.visit_attribute()
    body = source[start:end]
    if len(body) > 8192 or len(body.encode("utf-8")) > 8192:
        _fail("fixture selection attribute exceeds its byte bound")
    match = IDENTIFIER.match(source, _fixture_trivia_end(source, start, end), end)
    if match is None:
        _fail("unsupported fixture selection attribute")
    name = match[0]
    cursor = _fixture_trivia_end(source, match.end(), end)
    if name == "path":
        # Only a direct outer attribute on a library-root module is supported.
        # Never resolve cfg_attr paths, escapes, directories or computed strings.
        if inner or depth or module_paths is None:
            _fail("unsupported fixture selection attribute")
        if module_paths:
            _fail("duplicate fixture module path")
        if cursor == end or source[cursor] != "=":
            _fail("unsupported fixture selection attribute")
        cursor = _fixture_trivia_end(source, cursor + 1, end)
        literal = re.compile(r'"([A-Za-z0-9_][A-Za-z0-9_-]*\.rs)"').match(source, cursor, end)
        if literal is None or _fixture_trivia_end(source, literal.end(), end) != end:
            _fail("unsupported fixture selection attribute: path requires one sibling .rs literal")
        module_paths.append(literal[1])
        return True, 0
    arguments = None
    if cursor < end:
        if source[cursor] != "(" or pairs.get(cursor, end + 1) > end:
            _fail("unsupported fixture selection attribute")
        closing = pairs[cursor] - 1
        if _fixture_trivia_end(source, closing + 1, end) != end:
            _fail("unsupported fixture selection attribute")
        arguments = (cursor + 1, closing)
    if name == "cfg" and arguments is not None:
        return _fixture_cfg(source[arguments[0]:arguments[1]], features), 0
    if name == "cfg_attr" and arguments is not None:
        # Preserve the existing crate-level no_std spelling; other no_std
        # transformations remain outside this restricted source selector.
        if inner and depth == 0 and re.fullmatch(
                r'\s*cfg_attr\s*\(\s*target_arch\s*=\s*"amdgpu"\s*,\s*no_std\s*\)\s*', body):
            return True, 0
        spans = []
        first = cursor = arguments[0]
        while cursor < arguments[1]:
            if code[cursor] in "([{":
                cursor = pairs[cursor]
                continue
            if code[cursor] == ",":
                spans.append((first, cursor))
                first = cursor + 1
            cursor += 1
        if not spans:
            _fail("fixture cfg_attr requires a predicate and comma")
        if _fixture_trivia_end(source, first, arguments[1]) != arguments[1]:
            spans.append((first, arguments[1]))
        if len(spans) > 65:
            _fail("fixture cfg_attr exceeds its child bound")
        active = _fixture_cfg(source[spans[0][0]:spans[0][1]], features)
        enabled, kernels = True, 0
        for first, last in spans[1:]:
            # Validate inactive branches too. Unsupported transformations must
            # not disappear simply because this particular feature is absent.
            child_enabled, child_kernels = _fixture_attribute(
                source, code, pairs, first, last, features, budget, inner, depth + 1, derives)
            enabled = enabled and child_enabled
            kernels += child_kernels
        return (enabled, kernels) if active else (True, 0)
    if name == "derive":
        if inner or arguments is None or derives is None:
            _fail("unsupported fixture selection attribute")
        cursor, last = arguments
        names = set()
        while True:
            cursor = _fixture_trivia_end(source, cursor, last)
            token = IDENTIFIER.match(source, cursor, last)
            if token is None or token[0] not in BUILTIN_DERIVES or token[0] in names:
                _fail("unsupported fixture builtin derive")
            names.add(token[0])
            cursor = _fixture_trivia_end(source, token.end(), last)
            if cursor == last:
                break
            if source[cursor] != ",":
                _fail("unsupported fixture builtin derive")
            cursor = _fixture_trivia_end(source, cursor + 1, last)
            if cursor == last:
                break
        if derives.intersection(names):
            _fail("duplicate fixture builtin derive")
        derives.update(names)
        return True, 0
    if name == "no_std" and inner and depth == 0 and arguments is None:
        return True, 0
    if name not in {"allow", "deny", "forbid", "warn", "doc", "inline", "kernel"}:
        _fail("unsupported fixture selection attribute")
    if arguments is None and name not in {"inline", "kernel"}:
        _fail("unsupported fixture selection attribute")
    if name == "inline" and arguments is not None:
        inline = IDENTIFIER.match(source, _fixture_trivia_end(source, *arguments), arguments[1])
        if (inline is None or inline[0] not in {"always", "never"}
                or _fixture_trivia_end(source, inline.end(), arguments[1]) != arguments[1]):
            _fail("unsupported fixture selection attribute")
    if name == "kernel" and inner:
        _fail("unsupported fixture selection attribute")
    return True, int(name == "kernel")


def _fixture_macro_definition(source: str, start: int, end: int) -> str | None:
    """Recognize an inert definition header without accepting masked literals."""
    if not source.startswith("macro_rules", start):
        return None
    cursor = _fixture_trivia_end(source, start + len("macro_rules"), end)
    if cursor == end or source[cursor] != "!":
        return None
    cursor = _fixture_trivia_end(source, cursor + 1, end)
    name = IDENTIFIER.match(source, cursor, end)
    return name[0] if name is not None and _fixture_trivia_end(source, name.end(), end) == end else None


class _FixtureIncludeScope:
    """Conservatively exclude textual/import shadowing across the selected closure."""

    def __init__(self):
        self.names: set[str] = set()
        self.ambiguous = False
        self.used = False
        self.derives: set[str] = set()
        self.derive_ambiguous = False

    def define(self, name: str) -> None:
        self.ambiguous |= name == "include" or name in self.names
        self.names.add(name)
        self.derive_ambiguous |= name in BUILTIN_DERIVES

    def imported(self, item: str) -> None:
        self.ambiguous |= re.search(r"\b(?:include|as)\b|\*", item) is not None
        self.derive_ambiguous |= (re.search(r"\bas\b|\*", item) is not None
                                  or any(token[0] in BUILTIN_DERIVES for token in IDENTIFIER.finditer(item)))

    def validate(self) -> None:
        if self.derives and self.derive_ambiguous:
            _fail("fixture builtin derive has ambiguous macro or import scope")
        if self.used and self.ambiguous:
            _fail("literal fixture include has ambiguous macro or import scope")


def _fixture_include_path(source: str, start: int, end: int, pairs: dict[int, int]) -> str:
    cursor = _fixture_trivia_end(source, start + len("include"), end)
    if cursor == end or source[cursor] != "!":
        _fail("unsupported literal fixture include")
    cursor = _fixture_trivia_end(source, cursor + 1, end)
    closing = pairs.get(cursor)
    if (cursor == end or source[cursor] != "(" or closing is None or closing > end
            or _fixture_trivia_end(source, closing, end) != end):
        _fail("unsupported literal fixture include")
    start = _fixture_trivia_end(source, cursor + 1, closing - 1)
    raw = re.compile(r'r(#{0,255})"').match(source, start, closing - 1)
    if raw is not None:
        delimiter = '"' + raw[1]
        end = source.find(delimiter, raw.end(), closing - 1)
        if end < 0:
            _fail("fixture include requires one terminated raw literal path")
        value = source[raw.end():end]
        end += len(delimiter)
    else:
        literal = re.compile(r'"([^"\\\r\n]*)"').match(source, start, closing - 1)
        if literal is None:
            _fail("fixture include requires one unescaped literal path")
        value, end = literal[1], literal.end()
    if _fixture_trivia_end(source, end, closing - 1) != closing - 1:
        _fail("fixture include has trailing tokens after its literal path")
    relative = PurePosixPath(value)
    if (not value or not relative.parts or len(value) > 256 or relative.is_absolute() or str(relative) != value
            or ".." in relative.parts or "\\" in value or any(ord(char) < 32 for char in value)):
        _fail("fixture include requires a canonical relative path")
    return value


def _fixture_data_item(code: str, start: int, boundary: int, end: int) -> bool:
    """Admit inert non-generic data declarations, never item-producing tokens."""
    head = code[start:boundary]
    match = re.fullmatch(
        r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?(struct|enum)\s+[A-Za-z_][A-Za-z0-9_]*"
        r"\s*(\([^{};]*\)\s*)?", head)
    if match is None:
        return False
    if ((match[1] == "enum" and (code[boundary] != "{" or match[2] is not None))
            or (code[boundary] == "{" and match[2] is not None)):
        _fail("unsupported fixture data declaration")
    if re.search(r"[!#]|\b(?:fn|mod|impl|extern|use|macro_rules|struct|enum|union|type)\b",
                 code[start + head.index(match[1]) + len(match[1]):end]):
        _fail("fixture data declaration contains unsupported transformation or item")
    return True


_FixtureItem = tuple[tuple[tuple[bool, int, int], ...], tuple[int, int, int, int, int, int] | None]


def _fixture_item_span(source: str, code: str, pairs: dict[int, int], start: int,
                       previous_character: int, previous_byte: int) -> tuple[int, int, int, int, int, int]:
    cursor = start
    while cursor < len(code) and code[cursor] not in "{;":
        if code[cursor] in "([":
            cursor = pairs[cursor]
        else:
            cursor += 1
    if cursor == len(code):
        _fail("unsupported fixture item boundary")
    boundary = cursor
    cursor = pairs[cursor] if code[cursor] == "{" else cursor + 1
    first_byte = previous_byte + len(source[previous_character:start].encode("utf-8"))
    last_byte = first_byte + len(source[start:boundary].encode("utf-8"))
    end_byte = last_byte + len(source[boundary:cursor].encode("utf-8"))
    return start, boundary, cursor, first_byte, last_byte, end_byte


def _fixture_declarations(
    source: str, features: set[str], scan_functions: Callable,
    rust_syntax: Callable, budget: _Budget,
    *, include_paths: list[str] | None = None, include_scope: _FixtureIncludeScope | None = None,
    macros_only: bool = False, allow_module_path: bool = False,
    function_cache: dict[str, list[dict[str, Any]]] | None = None,
    item_cache: dict[str, tuple[_FixtureItem, ...]] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str | None]]]:
    """Select physical functions/modules; macro definitions are never expanded."""
    _utf8(source, "fixture source")
    # Coordinates depend only on source bytes, not the selected Cargo features.
    functions = None if function_cache is None else function_cache.get(source)
    if functions is None:
        functions = budget.rows(scan_functions(source), "fixture function items")
        if function_cache is not None:
            function_cache[source] = functions
    code, pairs = rust_syntax(source)
    # Cache only immutable coordinates after the complete traversal succeeds;
    # lexer output and every feature/context-dependent decision remain fresh.
    cached_items = None if item_cache is None else item_cache.get(source)
    parsed_items = []
    item_index = 0
    cursor = 0
    function_index = previous_character = previous_byte = 0
    selected = []
    modules = []
    attribute = re.compile(r"#\s*(!?)\s*\[")
    while True:
        if cached_items is None:
            cursor = _fixture_trivia_end(source, cursor, len(code))
            if cursor == len(code):
                break
            if code[cursor] == ";":
                cursor += 1
                continue
            attributes = []
            body = None
        else:
            if item_index == len(cached_items):
                break
            attributes, body = cached_items[item_index]
            item_index += 1
        # Syntax rows bound newly materialized records; a separate cumulative
        # visit budget bounds every feature/context replay, including cache hits.
        budget.visit_source_item()
        if cached_items is None:
            budget.rows([None], "fixture source items")
        enabled = True
        kernels = 0
        derives: set[str] = set()
        module_paths: list[str] = []
        attribute_index = 0
        while True:
            if cached_items is None:
                match = attribute.match(code, cursor)
                if match is None:
                    break
                inner, opening = bool(match[1]), match.end() - 1
                end = pairs[opening]
                attributes.append((inner, opening, end))
            elif attribute_index < len(attributes):
                inner, opening, end = attributes[attribute_index]
                attribute_index += 1
            else:
                break
            if macros_only and inner:
                _fail("included fixture source cannot contain inner attributes")
            attr_enabled, attr_kernels = _fixture_attribute(
                source, code, pairs, opening + 1, end - 1, features, budget, inner, derives=derives,
                module_paths=module_paths if allow_module_path else None)
            enabled = enabled and attr_enabled
            kernels += attr_kernels
            if inner and not enabled:
                _fail("conditional fixture crate/module is unsupported")
            if cached_items is None:
                cursor = _fixture_trivia_end(source, end, len(code))
        if cached_items is None:
            if cursor != len(code):
                body = _fixture_item_span(source, code, pairs, cursor, previous_character, previous_byte)
            if item_cache is not None:
                parsed_items.append((tuple(attributes), body))
        if body is None:
            if module_paths:
                _fail("fixture path requires an ordinary root module declaration")
            break
        start, boundary, cursor, first_byte, last_byte, end_byte = body
        head = code[start:boundary]
        module = re.fullmatch(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*", head)
        # Validate the attributed item even when cfg removes it from selection.
        if module_paths and (module is None or code[boundary] != ";" or kernels or derives):
            _fail("fixture path requires an ordinary root module declaration")
        data_item = _fixture_data_item(code, start, boundary, cursor)
        if derives:
            if not data_item or include_scope is None:
                _fail("fixture derive requires an ordinary data item and checked macro scope")
            include_scope.derives.update(derives)
        if data_item and kernels:
            _fail("kernel attribute on a fixture data declaration")
        previous_character, previous_byte = cursor, end_byte
        inert_macro = code[boundary] == "{" and _fixture_macro_definition(source, start, boundary)
        if inert_macro and kernels:
            _fail("kernel attribute on an inert fixture macro definition")
        include_item = include_paths is not None and re.match(r"include\b", head) is not None
        included_helper = (code[boundary] == "{" and not kernels
                           and re.match(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*\b", head))
        if macros_only and not inert_macro and not include_item and not included_helper:
            _fail("included fixture source must contain only inert macro definitions, non-attributed free functions, or literal includes")
        if inert_macro and enabled and include_scope is not None:
            include_scope.define(inert_macro)
        item_functions = []
        while function_index < len(functions) and functions[function_index]["functionUtf8Offset"] < end_byte:
            function = functions[function_index]
            function_index += 1
            if enabled and not inert_macro and function["attributedKernel"]:
                if not first_byte <= function["functionUtf8Offset"] < last_byte:
                    _fail("nested fixture kernel declaration is unsupported")
                item_functions.append(function)
        # The scanner and item budgets still include the complete definition;
        # apparent function tokens in its body are not physical declarations.
        if not enabled or inert_macro:
            continue
        if kernels > 1:
            _fail("duplicate active fixture kernel attributes")
        if data_item:
            continue
        if include_item:
            if kernels or code[boundary] != ";" or include_scope is None:
                _fail("unsupported literal fixture include")
            include_paths.append(_fixture_include_path(source, start, boundary, pairs))
            include_scope.used = True
            continue
        if include_scope is not None and re.match(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?use\b", head):
            include_scope.imported(code[start:cursor])
        if module is not None:
            if code[boundary] != ";":
                _fail("inline fixture modules are unsupported")
            modules.append((module[1], module_paths[0] if module_paths else None))
            continue
        if (re.search(r"\b(?:macro_rules|include)\b|!", head)
                or re.match(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:use|const|static|type|fn)\b", head) is None):
            _fail("unsupported fixture item or module selection")
        if kernels:
            selected.extend(item_functions)
    if item_cache is not None and cached_items is None:
        item_cache[source] = tuple(parsed_items)
    return selected, modules


def _fixture_selection(fixture: dict[str, Any], load_sources: Callable, scan_functions: Callable,
                       rust_syntax: Callable, budget: _Budget, *,
                       function_cache: dict[str, list[dict[str, Any]]] | None = None,
                       item_cache: dict[str, tuple[_FixtureItem, ...]] | None = None,
                       ) -> dict[str, tuple[str, int, str]]:
    library, sources, enabled = load_sources(fixture)
    # The loader authenticates the current physical package closure and Cargo
    # inputs. Traversal establishes only this bounded source selection, not rustc
    # acceptance or any executable outcome.
    # Physical include paths and Rust module context are different coordinates.
    pending = [(library, (), False, 0)]
    visited = set()
    selected = {}
    include_scope = _FixtureIncludeScope()
    while pending:
        path, module_context, included, include_depth = pending.pop()
        if include_depth > MAX_INCLUDE_DEPTH:
            _fail("fixture include exceeds its nesting bound")
        if path in visited or path not in sources:
            _fail("ambiguous or missing fixture module selection or repeated include")
        visited.add(path)
        source = sources[path]
        budget.source_bytes += len(_utf8(source, "fixture source"))
        if budget.source_bytes > MAX_RUNTIME_BYTES:
            _fail("selected fixture source exceeds its aggregate byte bound")
        includes = []
        functions, modules = _fixture_declarations(
            source, set(enabled), scan_functions, rust_syntax, budget,
            include_paths=includes, include_scope=include_scope, macros_only=included,
            allow_module_path=path == library and not module_context and not included,
            function_cache=function_cache, item_cache=item_cache)
        for function in functions:
            symbol = function["kernelSymbol"]
            if symbol in selected:
                _fail("ambiguous feature-selected fixture kernel")
            selected[symbol] = (path, function["functionUtf8Offset"], source)
        # Deliberately bounded to ordinary sibling modules of a library root.
        # Explicit sibling filenames change the physical file, not the Rust
        # module name. Nested, inline and macro-selected modules stay unsupported.
        if modules and (path != library or module_context):
            _fail("nested fixture modules are unsupported")
        module_names = set()
        for module, filename in modules:
            if module in module_names:
                _fail("ambiguous fixture module name")
            module_names.add(module)
            parent = path.rsplit("/", 1)[0]
            possible = ((f"{parent}/{filename}",) if filename is not None else
                        (f"{parent}/{module}.rs", f"{parent}/{module}/mod.rs"))
            candidates = [candidate for candidate in possible if candidate in sources]
            if len(candidates) != 1:
                _fail("ambiguous or missing fixture module selection")
            pending.append((candidates[0], (module,), False, 0))
        for relative in includes:
            # Membership uses only the loader-authenticated packageSources closure.
            target = (PurePosixPath(path).parent / relative).as_posix()
            pending.append((target, module_context, True, include_depth + 1))
    include_scope.validate()
    if set(selected) != set(fixture["compilerInput"]["kernelSymbols"]):
        _fail("feature-selected fixture kernel roster differs")
    return selected


def validate_kernel_inventory(
    manifest: dict[str, Any], runtime_inventory: dict[str, Any] | None,
    scan_functions: Callable[[str], list[dict[str, Any]]], *, max_records: int = 4096,
    load_fixture_sources: Callable | None = None, rust_syntax: Callable | None = None,
    load_source_case_sources: Callable | None = None,
    observe_sources: Callable | None = None,
) -> dict[str, Any]:
    """Validate the optional sibling and return only bounded validated fields.

    The input manifest and optional runtime projection must already pass their
    existing validators. With no runtime, selection coverage is checked but
    occurrence coverage and source/display joins remain explicitly unvalidated.
    """
    budget = _Budget(max_records, source_item_limit=MAX_SOURCE_ITEM_VISITS,
                     attribute_limit=MAX_ATTRIBUTE_VISITS)
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("curriculum"), dict)
            or manifest["curriculum"].get("schema") != "fe2o3-tutorial-curriculum-obligations-v2"):
        _fail("the identity extension requires the existing V2 curriculum")
    inventory = _object(manifest.get("kernelInventory"),
                        {"schema", "kernels", "negativeCases", "displayItems"}, "root")
    if inventory["schema"] != SCHEMA:
        _fail("unsupported schema")
    tabs: dict[tuple[str, int], tuple[dict[str, Any], str]] = {}
    selections: dict[tuple[Any, ...], dict[str, Any]] = {}
    expected_negative = set()

    def add_selection(key: tuple[Any, ...], value: dict[str, Any]) -> None:
        if key in selections:
            _fail("duplicate existing source selection")
        selections[key] = value

    for fixture in budget.rows(manifest["compilerFixtures"], "fixtures"):
        inputs = fixture["compilerInput"]
        for symbol in budget.rows(inputs["kernelSymbols"], "fixture symbols"):
            symbol = _symbol(symbol)
            key = ("fixture", fixture["fixtureId"], symbol)
            add_selection(key, {"identity": _selection_identity(inputs, symbol, budget), "symbol": symbol,
                                "fixture": fixture})
    for lesson in budget.rows(manifest["curriculum"]["lessons"], "lessons"):
        lesson_id = lesson["lessonId"]
        for ordinal, tab in enumerate(budget.rows(lesson["codeTabs"], "tabs")):
            tabs[(lesson_id, ordinal)] = (tab, lesson["role"])
            item = tab["sourceItem"]
            if item is None:
                continue
            for index, case in enumerate(budget.rows(item["cases"], "source cases")):
                key = ("source-driver-case", lesson_id, ordinal, index)
                inputs = {**item["compilerInput"], "features": case["features"]}
                add_selection(key, {"identity": _selection_identity(inputs, case["kernelSymbol"], budget),
                                    "symbol": case["kernelSymbol"], "case": case, "tab": tab})
                if case["expectation"]["kind"] == "rejected":
                    expected_negative.add(key)
    expected_positive = selections.keys() - expected_negative
    consumed = set()
    kernels = []
    by_id: dict[str, set[tuple[Any, ...]]] = {}
    identity_ids = {}
    for kernel in budget.rows(inventory["kernels"], "kernels"):
        _object(kernel, {"kernelId", "selections", "variants"}, "kernel")
        kernel_id = kernel["kernelId"]
        if not isinstance(kernel_id, str) or KERNEL_ID.fullmatch(kernel_id) is None or kernel_id in by_id:
            _fail("duplicate or invalid kernelId")
        refs = []
        keys = set()
        identity = None
        for value in budget.rows(kernel["selections"], "kernel selections"):
            key, ref = _reference(value)
            if key not in expected_positive or key in consumed:
                _fail("unknown, negative, or duplicate positive selection")
            current = selections[key]["identity"]
            if identity is not None and identity != current:
                _fail("one kernelId cannot merge different source selections")
            identity = current
            consumed.add(key)
            keys.add(key)
            refs.append(ref)
        if not refs:
            _fail("kernel identity requires an existing positive source selection")
        if identity in identity_ids:
            _fail("identical source selections must share one kernelId across targets")
        identity_ids[identity] = kernel_id
        variants = []
        for variant in budget.rows(kernel["variants"], "variants"):
            _object(variant, {"kind", "status", "source", "blocker"}, "variant")
            status = variant["status"]
            if status not in ("pending", "source-bound"):
                _fail("variant status must be pending or source-bound, not qualification")
            source = None
            if status == "pending":
                if variant["source"] is not None:
                    _fail("pending variant cannot carry a source binding")
            else:
                binding = _object(variant["source"], {
                    "implementationKernelId", "selection", "selectionSha256", "sourcePath",
                    "sourceSha256", "functionUtf8Offset",
                }, "variant source")
                budget.rows([binding], "variant sources")
                implementation = _text(binding["implementationKernelId"], "implementationKernelId")
                if KERNEL_ID.fullmatch(implementation) is None:
                    _fail("invalid implementationKernelId")
                _, reference = _reference(binding["selection"])
                source = {
                    "implementationKernelId": implementation, "selection": reference,
                    "selectionSha256": _digest(binding["selectionSha256"], "selectionSha256"),
                    "sourcePath": _text(binding["sourcePath"], "sourcePath", 4096),
                    "sourceSha256": _digest(binding["sourceSha256"], "sourceSha256"),
                    "functionUtf8Offset": _integer(binding["functionUtf8Offset"], "functionUtf8Offset"),
                }
            blocker = _object(variant["blocker"], {"owner", "issue", "reason"}, "blocker")
            owner = _text(blocker["owner"], "blocker owner")
            issue = _text(blocker["issue"], "blocker issue", 1024)
            reason = _text(blocker["reason"], "blocker reason", 2048)
            if ISSUE_URL.fullmatch(issue) is None:
                _fail("blocker issue must be an exact GitHub issue URL")
            variants.append({"kind": variant["kind"], "status": status, "source": source,
                             "blocker": {"owner": owner, "issue": issue, "reason": reason}})
        if [row["kind"] for row in variants] not in (["simt", "tile"], ["simt", "tile", "mixed"]):
            _fail("kernel must retain ordered SIMT/tile and optional mixed variants")
        by_id[kernel_id] = keys
        kernels.append({"kernelId": kernel_id, "selectionSha256": identity,
                        "selections": refs, "variants": variants})
    if consumed != expected_positive:
        _fail("missing positive source selection")

    negative_refs = []
    negative_keys = set()
    for value in budget.rows(inventory["negativeCases"], "negative cases"):
        key, ref = _reference(value)
        if key not in expected_negative or key in negative_keys:
            _fail("unknown, positive, or duplicate required-negative case")
        negative_keys.add(key)
        negative_refs.append(ref)
    if negative_keys != expected_negative:
        _fail("missing required-negative case")

    selected_sources = {}
    source_digests = {}
    function_cache: dict[str, list[dict[str, Any]]] = {}
    item_cache: dict[str, tuple[_FixtureItem, ...]] = {}

    def selected_source(key: tuple[Any, ...]) -> tuple[str, int, str, str]:
        selection = selections[key]
        cache_key = key[:2] if key[0] == "fixture" else key
        if cache_key not in selected_sources:
            if rust_syntax is None:
                _fail("source binding requires current physical source validation")
            if key[0] == "fixture":
                if load_fixture_sources is None:
                    _fail("fixture source binding requires current physical source validation")
                fixture, loader = selection["fixture"], load_fixture_sources
            else:
                if load_source_case_sources is None:
                    _fail("source case binding requires current physical source validation")
                tab, case = selection["tab"], selection["case"]
                fixture = {"compilerInput": {**tab["sourceItem"]["compilerInput"],
                           "features": case["features"], "kernelSymbols": [case["kernelSymbol"]]}}
                loader = lambda _: load_source_case_sources(key[1], tab, case)
            selected_sources[cache_key] = _fixture_selection(
                fixture, loader, scan_functions, rust_syntax, budget,
                function_cache=function_cache, item_cache=item_cache)
        path, offset, source = selected_sources[cache_key][selection["symbol"]]
        digest_key = (cache_key, path)
        if digest_key not in source_digests:
            source_digests[digest_key] = hashlib.sha256(_utf8(source, "selected source")).hexdigest()
        return path, offset, source, source_digests[digest_key]

    # Resolve after collecting all identities: implementations may be declared
    # later and may differ from the obligation's original source selection.
    source_bound_variants = source_bound_pairs = variant_count = 0
    for kernel in kernels:
        for variant in kernel["variants"]:
            variant_count += 1
            binding = variant["source"]
            if binding is None:
                continue
            key, _ = _reference(binding["selection"])
            if key not in by_id.get(binding["implementationKernelId"], set()):
                _fail("variant selection does not belong to its implementation kernel")
            selection = selections[key]
            if binding["selectionSha256"] != selection["identity"]:
                _fail("variant selection digest is stale or differs from its implementation")
            inputs = (selection["fixture"]["compilerInput"] if key[0] == "fixture"
                      else selection["tab"]["sourceItem"]["compilerInput"])
            if binding["sourcePath"] not in inputs["sourcePaths"]:
                _fail("variant source path is not an exact selected source")
            path, offset, _, digest = selected_source(key)
            if (binding["sourcePath"] != path or binding["functionUtf8Offset"] != offset
                    or binding["sourceSha256"] != digest):
                _fail("variant source differs from the exact current source occurrence")
            source_bound_variants += 1
        source_bound_pairs += all(row["status"] == "source-bound" for row in kernel["variants"][:2])

    runtime_tabs = {}
    candidates = {}
    case_intervals = {}
    if runtime_inventory is not None:
        total_bytes = 0
        for lesson in budget.rows(runtime_inventory["lessons"], "runtime lessons"):
            for ordinal, live in enumerate(budget.rows(lesson["codeTabs"], "runtime tabs")):
                location = (lesson["id"], ordinal)
                if location not in tabs or location in runtime_tabs:
                    _fail("runtime tab coverage differs")
                runtime_tabs[location] = live
                tab, _ = tabs[location]
                if tab["language"] != "rust":
                    continue
                encoded = _utf8(live.get("displayedCode"), "displayed code")
                total_bytes += len(encoded)
                if total_bytes > MAX_RUNTIME_BYTES:
                    _fail("runtime Rust source exceeds its aggregate byte bound")
                if (type(tab["displayedUtf8Bytes"]) is not int
                        or len(encoded) != tab["displayedUtf8Bytes"]
                        or hashlib.sha256(encoded).hexdigest() != tab["displayedSha256"]):
                    _fail("displayed bytes differ from the retained tab")
                items = budget.rows(scan_functions(live["displayedCode"]), "scanned function items")
                offsets = set()
                for function in items:
                    _object(function, {"kernelSymbol", "functionUtf8Offset", "attributedKernel"}, "scanned item")
                    symbol = _display_symbol(function["kernelSymbol"])
                    offset = _integer(function["functionUtf8Offset"], "functionUtf8Offset")
                    attributed = function["attributedKernel"]
                    if type(attributed) is not bool or offset in offsets or not _name_token_matches(encoded, offset, symbol):
                        _fail("scanner returned a duplicate or invalid function coordinate")
                    offsets.add(offset)
                    if attributed or tab["kind"] == "kernel":
                        candidates[(*location, offset, symbol)] = attributed
                if tab["sourceItem"] is not None:
                    case_intervals[location] = _fragment_intervals(tab, live, encoded)
        if runtime_tabs.keys() != tabs.keys():
            _fail("runtime tab coverage differs")

    displays = []
    seen = set()
    negative_display_keys = set()
    bound_positive = set()
    unresolved = []
    fixture_displays = _FixtureDisplayIndex(budget)
    for row in budget.rows(inventory["displayItems"], "display items"):
        _object(row, DISPLAY_FIELDS, "display item")
        lesson = _text(row["lessonId"], "lessonId")
        ordinal = _integer(row["tabOrdinal"], "tabOrdinal")
        offset = _integer(row["functionUtf8Offset"], "functionUtf8Offset")
        symbol = _display_symbol(row["kernelSymbol"])
        location = (lesson, ordinal)
        coordinate = (*location, offset, symbol)
        if location not in tabs or coordinate in seen:
            _fail("unknown or duplicate display occurrence")
        seen.add(coordinate)
        tab, role = tabs[location]
        if tab["language"] != "rust" or offset + len(symbol.encode("utf-8")) > tab["displayedUtf8Bytes"]:
            _fail("display occurrence is outside a Rust tab")
        if runtime_inventory is not None and coordinate not in candidates:
            _fail("display occurrence is missing from the live function census")
        classification = row["classification"]
        status = row["bindingStatus"]
        if not isinstance(classification, str) or not isinstance(status, str):
            _fail("display classification and binding status must be strings")
        reason = _text(row["reason"], "display reason", 2048)
        ids = budget.rows(row["kernelIds"], "display kernel IDs")
        if any(not isinstance(value, str) or value not in by_id for value in ids) or len(set(ids)) != len(ids):
            _fail("display has duplicate or unknown kernel IDs")
        refs = []
        refs_keys = set()
        for value in budget.rows(row["negativeCases"], "display negative cases"):
            key, ref = _reference(value)
            if key not in expected_negative or key in refs_keys:
                _fail("display has duplicate or unknown negative cases")
            refs_keys.add(key)
            refs.append(ref)

        matching_cases = {key for key, value in selections.items()
                          if key[:3] == ("source-driver-case", lesson, ordinal) and value["symbol"] == symbol}
        if runtime_inventory is not None:
            intervals = case_intervals.get(location, [])
            matching_cases = {key for key in matching_cases
                              if type(selections[key]["case"]["displayedFragmentOrdinal"]) is int
                              and 0 <= selections[key]["case"]["displayedFragmentOrdinal"] < len(intervals)
                              and intervals[selections[key]["case"]["displayedFragmentOrdinal"]][0] <= offset
                              < intervals[selections[key]["case"]["displayedFragmentOrdinal"]][1]}
        if classification in {"conceptual", "helper"}:
            if ids or refs or status != "not-applicable" or (runtime_inventory is not None and matching_cases):
                _fail("non-kernel classification cannot detach a source case or identity")
            if classification == "conceptual" and role != "conceptual":
                _fail("executable lesson cannot be downgraded to conceptual")
            if classification == "helper" and (tab["kind"] != "kernel" or candidates.get(coordinate, False)):
                _fail("attributed kernel cannot be classified as a helper")
        elif classification in {"kernel", "required-negative"}:
            if role != "executable" or status not in {"pending", "source-driver-contract", "fixture-source-contract"}:
                _fail("kernel display has an invalid role or binding status")
            if classification == "kernel" and (refs or (runtime_inventory is not None and matching_cases & expected_negative)):
                _fail("required negative cannot be converted into a positive kernel")
            if classification == "required-negative":
                if (ids or not refs or not refs_keys <= matching_cases or not refs_keys <= expected_negative
                        or (runtime_inventory is not None and refs_keys != matching_cases)):
                    _fail("required-negative display must retain its exact source cases")
                if negative_display_keys & refs_keys:
                    _fail("required-negative case has duplicate display bindings")
                negative_display_keys.update(refs_keys)
            if status == "fixture-source-contract":
                if (classification != "kernel" or not ids or refs or matching_cases
                        or tab["kind"] != "kernel" or tab["sourceItem"] is not None):
                    _fail("fixture display requires an exclusive positive binding")
                _fixture_display_declaration(tab)
                if load_fixture_sources is None or rust_syntax is None:
                    _fail("fixture display requires current physical source validation")
                for kernel_id in ids:
                    keys = by_id[kernel_id]
                    if any(key[0] != "fixture" or selections[key]["symbol"] != symbol for key in keys):
                        _fail("fixture display identity has a different selection")
                    for key in sorted(keys):
                        fixture = selections[key]["fixture"]
                        if tab["sourcePath"] not in fixture["compilerInput"]["sourcePaths"]:
                            _fail("fixture display path is not an exact selected source")
                        path, source_offset, source, digest = selected_source(key)
                        if path != tab["sourcePath"]:
                            _fail("fixture display differs from the exact current source occurrence")
                        if tab["sourceDigestScope"] == "file":
                            encoded = _utf8(source, "fixture source")
                            if (source_offset != offset or len(encoded) != tab["displayedUtf8Bytes"]
                                    or digest != tab["sourceSha256"] or digest != tab["displayedSha256"]):
                                _fail("fixture display differs from the exact current source occurrence")
                        if runtime_inventory is not None:
                            live = runtime_tabs[location]
                            if tab["sourceDigestScope"] == "file":
                                if (live.get("sourceFragments") is not None
                                        or live["displayedCode"] != source or not candidates[coordinate]):
                                    _fail("fixture display differs from the live whole-file occurrence")
                            else:
                                fixture_displays.match(location, tab, live, path, source, digest,
                                                       offset, symbol, source_offset)
                            bound_positive.add(key)
            elif status == "pending":
                unresolved.append({"lessonId": lesson, "tabOrdinal": ordinal,
                                   "functionUtf8Offset": offset, "kernelSymbol": symbol, "reason": reason})
            else:
                bound_keys = refs_keys if classification == "required-negative" else set()
                if classification == "kernel":
                    if not ids:
                        _fail("source-bound display requires a kernel identity")
                    for kernel_id in ids:
                        joined = by_id[kernel_id] & matching_cases
                        if not joined:
                            _fail("display kernel identity has no exact same-tab source case")
                        bound_keys.update(joined)
                if runtime_inventory is not None:
                    for key in bound_keys:
                        case = selections[key]["case"]
                        fragments = case_intervals.get(location, [])
                        fragment = case["displayedFragmentOrdinal"]
                        if type(fragment) is not int or not 0 <= fragment < len(fragments):
                            _fail("source case fragment ordinal differs")
                        start, end = fragments[fragment]
                        if not candidates[coordinate] or not start <= offset < end:
                            _fail("source case does not bind this exact displayed function")
                        matches = [key for key in candidates if key[:2] == location and key[3] == symbol and start <= key[2] < end]
                        if len(matches) != 1:
                            _fail("source case displayed function is ambiguous")
                    if classification == "kernel":
                        for kernel_id in ids:
                            bound_positive.update(key for key in by_id[kernel_id]
                                                  if key[0] == "fixture" or key in bound_keys)
        else:
            _fail("unknown display classification")
        displays.append({"lessonId": lesson, "tabOrdinal": ordinal,
                         "functionUtf8Offset": offset, "kernelSymbol": symbol,
                         "classification": classification, "kernelIds": list(ids),
                         "negativeCases": refs, "bindingStatus": status, "reason": reason})
    if negative_display_keys != expected_negative:
        _fail("required-negative cases need complete direct display bindings")
    if runtime_inventory is not None and seen != candidates.keys():
        _fail("missing live function census occurrences")
    pending_displays = len(unresolved)
    for key in sorted(expected_positive - bound_positive):
        if key[0] == "fixture":
            reference = {"kind": key[0], "fixtureId": key[1], "kernelSymbol": key[2]}
        else:
            reference = {"kind": key[0], "lessonId": key[1], "tabOrdinal": key[2], "caseOrdinal": key[3]}
        unresolved.append({"selection": reference, "reason": "No exact live displayed source binding for this positive selection."})
    if runtime_inventory is None:
        unresolved.append({"reason": "The live runtime function census and source/display joins have not been validated."})
    complete = runtime_inventory is not None and not unresolved
    result = {
        "inventoryComplete": complete, "requiredPairCount": len(kernels) if complete else None,
        "knownKernelIdentityCount": len(kernels), "displayItemCount": len(displays),
        "knownVariantObligationCount": variant_count,
        "pendingVariantCount": variant_count - source_bound_variants,
        "sourceBoundVariantCount": source_bound_variants, "sourceBoundPairCount": source_bound_pairs,
        "variantBindingStatus": ("pending" if not source_bound_variants else
                                 "source-bound" if source_bound_variants == variant_count else "partial"),
        "pendingDisplayItemCount": pending_displays, "negativeCaseCount": len(negative_refs),
        "unregisteredDisplayItemCount": sum(
            row["classification"] == "kernel" and not row["kernelIds"] for row in displays),
        "runtimeCensusValidated": runtime_inventory is not None,
        "unresolvedBindings": unresolved, "kernelIdentities": kernels,
        "negativeCases": negative_refs, "displayItems": displays,
    }
    if observe_sources is not None:
        observe_sources(selections, selected_source, result)
    return result

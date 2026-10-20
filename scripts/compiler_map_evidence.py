#!/usr/bin/env python3
"""Offline passive map diagnostics. No content custody or admission authority."""

import argparse
import json
from pathlib import Path
import re
import sys

MAX_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 65536
MAX_LINE = 8192
U64 = (1 << 64) - 1
MAP = re.compile(
    r"([0-9a-f]+)-([0-9a-f]+) ([r-][w-][x-][ps]) +"
    r"([0-9a-f]+) +([0-9a-f]+):([0-9a-f]+) +([0-9]+)(?: +(.*))?"
)


class EvidenceError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate JSON field")
        result[key] = value
    return result


def _integer(value, positive=False):
    if type(value) is not int or not (int(positive) <= value <= U64):
        raise EvidenceError("missing or invalid integer identity")
    return value


def parse_map(line):
    """Parse one supplied proc map row, preserving deleted and unnamed mappings."""
    if not isinstance(line, str) or len(line) > MAX_LINE:
        raise EvidenceError("map row missing or too long")
    match = MAP.fullmatch(line)
    if not match:
        raise EvidenceError("malformed map row")
    start, end, permissions, offset, major, minor, inode, path = match.groups()
    numbers = [int(x, 16) for x in (start, end, offset, major, minor)]
    if any(x > U64 for x in numbers) or numbers[0] >= numbers[1]:
        raise EvidenceError("invalid map range or numeric overflow")
    if len(inode) > 20:
        raise EvidenceError("map inode overflow")
    inode = _integer(int(inode))
    return dict(start=numbers[0], end=numbers[1], permissions=permissions,
                offset=numbers[2], device=[numbers[3], numbers[4]], inode=inode,
                path=path or "", deleted=bool(path and path.endswith(" (deleted)")),
                raw=line)


def summarize(document, max_records=MAX_RECORDS):
    """Classify observations only; supplied custody claims remain unverified claims."""
    if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
        raise EvidenceError("record budget outside fixed limit")
    result = dict(
        schema="CompilerMapEvidenceDiagnosticV1", objects=[], unknown_records=[],
        issues=[], contexts=[], recorded_custody_claims=[], recorded_late_hash_claims=[],
        input_consistent=True,
        observation_coverage_complete=False,
        complete_executable_closure=False, validated_content_pins=0,
        authority=False, approval=False, profile=False, proof=False,
        native_execution=False, immutable_custody=False,
        scope="Supplied passive observations only; omitted execution is unknown.",
    )
    groups, births, paths = {}, {}, {}
    remaining = max_records
    object_paths = {}

    def charge():
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise EvidenceError("record budget exhausted; report is incomplete")

    def issue(reason):
        result["input_consistent"] = False
        if reason not in result["issues"]:
            result["issues"].append(reason)

    if not isinstance(document, dict) or not isinstance(document.get("observations"), list):
        raise EvidenceError("missing observations array")
    boot = document.get("boot")
    if not isinstance(boot, str) or not 1 <= len(boot) <= 128:
        issue("missing document boot identity")
    if document.get("errors") != []:
        issue("missing or nonempty observer errors")
    if document.get("truncated") is not False:
        issue("observation truncation not explicitly excluded")
    observations = document["observations"]
    if not observations:
        issue("no observations")
    try:
        for index, event in enumerate(observations):
            charge()
            if not isinstance(event, dict):
                result["unknown_records"].append(dict(observation=index, record=event))
                issue("malformed observation")
                continue
            event_boot = event.get("boot")
            process = event.get("process")
            valid_identity = True
            try:
                if (not isinstance(event_boot, str) or not 1 <= len(event_boot) <= 128
                        or event_boot != boot):
                    raise EvidenceError("boot identity missing or differs")
                if not isinstance(process, dict):
                    raise EvidenceError("missing process identity")
                pid = _integer(process.get("pid"), True)
                birth = _integer(process.get("start_ticks"), True)
                key = (event_boot, pid)
                if key in births and births[key] != birth:
                    raise EvidenceError("PID reuse in supplied observations")
                births[key] = birth
            except EvidenceError as error:
                issue(str(error))
                valid_identity = False
            object_boot = (event_boot if isinstance(event_boot, str)
                           and 1 <= len(event_boot) <= 128 else None)
            valid_exe = (isinstance(event.get("exe"), str)
                         and 1 <= len(event["exe"]) <= MAX_LINE)
            if not valid_exe:
                issue("missing or invalid executable context")
                valid_identity = False
            ancestry = event.get("command_ancestry")
            if not isinstance(ancestry, list) or not all(isinstance(x, str) for x in ancestry):
                issue("missing or invalid ancestry")
                valid_identity = False
            result["contexts"].append(dict(
                observation=index, process=process, identity_consistent=valid_identity,
                exe=event.get("exe"), command_ancestry=ancestry,
                boot=event_boot, mount_namespace=event.get("mount_namespace"),
                observer_mount_namespace=event.get("observer_mount_namespace")))
            mappings = event.get("mapped_files")
            if not isinstance(mappings, list):
                result["unknown_records"].append(dict(observation=index, record=event))
                issue("missing mapped_files array")
                continue
            for mapping in mappings:
                charge()
                if not isinstance(mapping, dict) or not isinstance(mapping.get("raw_lines"), list):
                    result["unknown_records"].append(dict(observation=index, record=mapping))
                    issue("missing raw map rows")
                    continue
                if not mapping["raw_lines"]:
                    result["unknown_records"].append(dict(observation=index, record=mapping))
                    issue("empty raw map rows")
                claims = {field: mapping[field] for field in (
                    "custody", "sha256", "stable_post_observation",
                    "writer_excluded", "immutable") if field in mapping}
                if claims:
                    result["recorded_custody_claims"].append(dict(
                        observation=index, path=mapping.get("path"),
                        claims=claims, verified=False))
                for line in mapping["raw_lines"]:
                    charge()
                    try:
                        row = parse_map(line)
                    except EvidenceError as error:
                        issue(str(error))
                        result["unknown_records"].append(dict(observation=index, record=line))
                        continue
                    try:
                        device = mapping.get("map_device")
                        if (not isinstance(device, str) or len(device) > 33
                                or not re.fullmatch(
                                    r"[0-9a-f]+:[0-9a-f]+", device)):
                            raise EvidenceError("missing or malformed recorded device")
                        if ([int(x, 16) for x in device.split(":")] != row["device"]
                                or _integer(mapping.get("map_inode")) != row["inode"]
                                or mapping.get("path") != row["path"]):
                            raise EvidenceError("raw and structured mapping identity conflict")
                    except EvidenceError as error:
                        issue(str(error))
                    # Inode zero has no file-object identity. Keep each occurrence separate.
                    key = (object_boot, *row["device"], row["inode"])
                    if not row["inode"]:
                        key += (index, remaining)
                    group = groups.setdefault(key, dict(
                        boot=object_boot, device=row["device"], inode=row["inode"],
                        paths=[], observations=[], classification="nonexec-unqualified",
                        validated_content_pin=False,
                    ))
                    object_paths.setdefault(key, set()).add(row["path"])
                    path_key = (object_boot, row["path"])
                    if row["path"]:
                        identity = (tuple(row["device"]), row["inode"])
                        if path_key in paths and paths[path_key] != identity:
                            issue("same recorded path has different object identities")
                        paths[path_key] = identity
                    group["observations"].append(dict(
                        observation=index, segment=row,
                    ))
                    if "x" in row["permissions"]:
                        group["classification"] = "observed-exec-unqualified"
                    elif (group["classification"] != "observed-exec-unqualified"
                          and "w" in row["permissions"]
                          and valid_exe
                          and event["exe"].rsplit("/", 1)[-1] == "rust-lld"):
                        group["classification"] = "nonexec-writable-output-candidate"
        pins = document.get("file_pins", [])
        if not isinstance(pins, list):
            issue("malformed recorded late hash claims")
            pins = []
        for pin in pins:
            charge()
            result["recorded_late_hash_claims"].append(pin)
    except EvidenceError as error:
        issue(str(error))
        result["unprocessed_records_possible"] = True
    for key, group in groups.items():
        group["paths"] = sorted(object_paths[key])
    result["objects"] = list(groups.values())
    result["custody"] = "offline-metadata-and-unverified-claims-only"
    result["custody_states"] = {
        "metadata-only": "Recorded identities, not retained live descriptors.",
        "recorded-late-hash-claims": "Unverified post-observation assertions, not historical bytes.",
        "live-retained-mutable": "Requires a live owner; writable aliases still defeat immutability.",
        "writer-excluded-content-pin": "Cannot be established by offline input, hashes or copies.",
    }
    return result


def read_report(stream, max_records=MAX_RECORDS):
    raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise EvidenceError("input byte budget exhausted")
    try:
        document = json.loads(raw, object_pairs_hook=_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(
                                  EvidenceError("nonfinite JSON number")))
    except (ValueError, RecursionError) as error:
        raise EvidenceError("invalid or truncated JSON: " + str(error)) from error
    return summarize(document, max_records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as stream:
            report = read_report(stream, args.max_records)
    except (OSError, EvidenceError) as error:
        report = dict(schema="CompilerMapEvidenceDiagnosticV1",
                      input_consistent=False, complete_executable_closure=False,
                      authority=False, validated_content_pins=0, issues=[str(error)])
    print(json.dumps(report, sort_keys=True))
    return 0 if report["input_consistent"] else 1


if __name__ == "__main__":
    sys.exit(main())

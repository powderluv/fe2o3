#!/usr/bin/env python3
"""CPU-only controls: offline observations never confer content or execution authority."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_map_evidence as evidence


def mapping(perms="rw-s", path="/out/plugin.so.tmp1", inode=23, offset="0"):
    return dict(path=path, map_device="fc:01", map_inode=inode,
                raw_lines=[f"1000-2000 {perms} {offset} fc:01 {inode} {path}"])


def observation(rows=None):
    return dict(boot="boot-a", process=dict(pid=42, start_ticks=100),
                exe="/tool/rust-lld", command_ancestry=["build-started.json"],
                mount_namespace="mnt:[1]", observer_mount_namespace="mnt:[1]",
                mapped_files=rows if rows is not None else [mapping()])


def document(events=None):
    return dict(boot="boot-a", errors=[], truncated=False,
                observations=events if events is not None else [observation()])


class MapEvidenceTests(unittest.TestCase):
    def report(self, value=None):
        return evidence.summarize(value if value is not None else document())

    def assert_no_authority(self, report):
        for field in ("complete_executable_closure", "authority", "approval",
                      "profile", "proof", "native_execution", "immutable_custody",
                      "observation_coverage_complete"):
            self.assertIs(report[field], False, field)
        self.assertEqual(report["validated_content_pins"], 0)
        for obj in report["objects"]:
            self.assertIs(obj["validated_content_pin"], False)

    def test_linker_output_is_candidate_not_proc_macro(self):
        report = self.report()
        self.assertTrue(report["input_consistent"])
        self.assertEqual(report["objects"][0]["classification"],
                         "nonexec-writable-output-candidate")
        segment = report["objects"][0]["observations"][0]["segment"]
        self.assertEqual((segment["permissions"], segment["offset"], segment["device"],
                          segment["inode"]), ("rw-s", 0, [252, 1], 23))
        self.assertEqual(report["contexts"][0]["process"]["start_ticks"], 100)
        self.assert_no_authority(report)

    def test_any_executable_segment_dominates_in_both_orders(self):
        for path in ("/out/plugin.so.tmp1", "/out/not-code.data"):
            for perms in (["rw-s", "r-xp"], ["r-xp", "rw-s"]):
                with self.subTest(path=path, perms=perms):
                    events = [observation([mapping(p, path)]) for p in perms]
                    report = self.report(document(events))
                    self.assertEqual(len(report["objects"]), 1)
                    self.assertEqual(report["objects"][0]["classification"],
                                     "observed-exec-unqualified")
                    self.assertEqual(len(report["objects"][0]["observations"]), 2)
                    self.assert_no_authority(report)

    def test_all_segments_in_one_snapshot_are_examined(self):
        row = mapping()
        row["raw_lines"].append("2000-3000 r-xp 1000 fc:01 23 /out/plugin.so.tmp1")
        report = self.report(document([observation([row])]))
        self.assertEqual(report["objects"][0]["classification"], "observed-exec-unqualified")
        self.assertEqual(report["objects"][0]["observations"][1]["segment"]["offset"], 4096)

    def test_deleted_executable_and_unnamed_mapping_are_not_dropped(self):
        deleted = mapping("r-xp", "/gone/lib (deleted)")
        anonymous = dict(path="", map_device="00:00", map_inode=0,
                         raw_lines=["3000-4000 rw-p 0 00:00 0"])
        report = self.report(document([observation([deleted, anonymous])]))
        self.assertEqual(len(report["objects"]), 2)
        self.assertTrue(report["objects"][0]["observations"][0]["segment"]["deleted"])
        self.assertEqual(report["objects"][0]["classification"], "observed-exec-unqualified")
        self.assert_no_authority(report)

    def test_claims_never_become_content_pins(self):
        row = mapping()
        row.update(custody="writer-excluded-content-pin", writer_excluded=True,
                   immutable=True, stable_post_observation=True, sha256="a" * 64)
        doc = document([observation([row])])
        doc["file_pins"] = [dict(path=row["path"], sha256="b" * 64,
                                 stable_post_observation=True)]
        doc.update(authority=True, complete_executable_closure=True)
        report = self.report(doc)
        self.assertEqual(report["recorded_late_hash_claims"], doc["file_pins"])
        self.assertFalse(report["recorded_custody_claims"][0]["verified"])
        self.assert_no_authority(report)

    def test_alias_and_rename_do_not_prove_content(self):
        rows = [mapping(path="/old"), mapping(path="/new (deleted)")]
        report = self.report(document([observation(rows)]))
        self.assertEqual(report["objects"][0]["paths"], ["/new (deleted)", "/old"])
        self.assert_no_authority(report)

    def test_same_path_replacement_does_not_substitute_historical_identity(self):
        rows = [mapping(inode=23), mapping(inode=24)]
        report = self.report(document([observation(rows)]))
        self.assertEqual(len(report["objects"]), 2)
        self.assertFalse(report["input_consistent"])
        self.assert_no_authority(report)

    def test_identity_conflicts_and_missing_metadata_are_incomplete(self):
        for mutate in (
            lambda d: d["observations"][0].update(boot="other-boot"),
            lambda d: d["observations"][0]["process"].pop("start_ticks"),
            lambda d: d["observations"][0]["process"].update(pid=True),
            lambda d: d["observations"][0].pop("command_ancestry"),
            lambda d: d["observations"][0].pop("exe"),
            lambda d: d["observations"][0]["mapped_files"][0].update(map_inode=99),
            lambda d: d["observations"][0]["mapped_files"][0].update(map_device="00:01"),
            lambda d: d["observations"][0]["mapped_files"][0].update(path="/replacement"),
            lambda d: d.update(errors=["lost observation"]),
            lambda d: d.update(truncated=True),
            lambda d: d.pop("truncated"),
        ):
            with self.subTest(mutation=mutate):
                doc = document()
                mutate(doc)
                report = self.report(doc)
                self.assertFalse(report["input_consistent"])
                self.assertEqual(len(report["objects"]), 1)
                self.assert_no_authority(report)

    def test_pid_reuse_is_incomplete(self):
        events = [observation(), observation()]
        events[1]["process"]["start_ticks"] = 101
        report = self.report(document(events))
        self.assertIn("PID reuse", " ".join(report["issues"]))
        self.assertFalse(report["input_consistent"])

    def test_unknown_rows_are_retained_not_filtered_by_suffix(self):
        row = mapping()
        row["raw_lines"] += ["bad row", "1000-1000 rw-s 0 fc:01 23 /bad"]
        report = self.report(document([observation([row, None])]))
        self.assertFalse(report["input_consistent"])
        self.assertEqual(len(report["unknown_records"]), 3)

    def test_numeric_limits_and_malformed_permissions(self):
        for line in (
            "1000-2000 rwxz 0 fc:01 23 /bad",
            "2000-1000 r-xp 0 fc:01 23 /bad",
            "1000-2000 r-xp 0 fc:01 " + "9" * 5000 + " /bad",
            "1000-2000 r-xp 10000000000000000 fc:01 23 /bad",
            "x" * (evidence.MAX_LINE + 1),
        ):
            with self.subTest(line=line[:60]):
                with self.assertRaises(evidence.EvidenceError):
                    evidence.parse_map(line)

    def test_omitted_executable_segment_never_proves_absence(self):
        report = self.report()
        self.assert_no_authority(report)
        self.assertNotIn("never_executed", report["objects"][0])

    def test_budget_exhaustion_retains_partial_evidence_and_incompleteness(self):
        doc = document([observation(), observation([mapping("r-xp")])])
        report = evidence.summarize(doc, max_records=3)
        self.assertEqual(len(report["objects"]), 1)
        self.assertTrue(report["unprocessed_records_possible"])
        self.assertFalse(report["input_consistent"])
        self.assert_no_authority(report)
        for limit in (0, True, evidence.MAX_RECORDS + 1):
            with self.assertRaises(evidence.EvidenceError):
                evidence.summarize(doc, limit)

    def test_json_duplicate_truncation_depth_and_byte_bounds(self):
        for raw in (b'{"observations": [], "observations":[]}', b'{"observations":',
                    b'{"observations": NaN}', b"[" * 1500 + b"]" * 1500):
            with self.subTest(raw=raw[:50]):
                with self.assertRaises(evidence.EvidenceError):
                    evidence.read_report(io.BytesIO(raw))
        with patch.object(evidence, "MAX_BYTES", 4):
            with self.assertRaisesRegex(evidence.EvidenceError, "byte budget"):
                evidence.read_report(io.BytesIO(b"12345"))

    def test_offline_helper_does_not_reopen_any_recorded_path(self):
        doc = document()
        with patch.object(Path, "open", side_effect=AssertionError("path reopened")):
            report = evidence.read_report(io.BytesIO(json.dumps(doc).encode()))
        self.assert_no_authority(report)

    def test_context_and_claims_are_not_amplified_for_each_segment(self):
        row = mapping()
        row["raw_lines"] *= 100
        row["custody"] = "live-retained-mutable"
        report = self.report(document([observation([row])]))
        self.assertEqual(len(report["contexts"]), 1)
        self.assertEqual(len(report["recorded_custody_claims"]), 1)
        self.assertEqual(len(report["objects"][0]["observations"]), 100)

    def test_invalid_boot_shapes_do_not_crash_or_amplify_output(self):
        for boot in ([], {}, "x" * 100000):
            with self.subTest(boot_type=type(boot).__name__):
                doc = document()
                doc["observations"][0]["boot"] = boot
                report = self.report(doc)
                self.assertFalse(report["input_consistent"])
                self.assertIsNone(report["objects"][0]["boot"])
                self.assert_no_authority(report)

    def test_oversized_context_fields_are_incomplete_without_repeated_parsing(self):
        doc = document()
        doc["observations"][0]["exe"] = "x" * 100000
        doc["observations"][0]["mapped_files"][0]["map_device"] = "f" * 100000
        report = self.report(doc)
        self.assertFalse(report["input_consistent"])
        self.assertEqual(report["objects"][0]["classification"], "nonexec-unqualified")
        self.assert_no_authority(report)

    def test_nonlinker_writable_mapping_is_not_assumed_to_be_output(self):
        doc = document()
        doc["observations"][0]["exe"] = "/tool/unfamiliar"
        report = self.report(doc)
        self.assertEqual(report["objects"][0]["classification"], "nonexec-unqualified")
        self.assert_no_authority(report)

    def test_cli_consistent_input_still_denies_completeness(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "input.json"
            path.write_text(json.dumps(document()))
            run = subprocess.run(
                [sys.executable, "-I", "-B", evidence.__file__, str(path)],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assert_no_authority(json.loads(run.stdout))

    def test_cli_returns_machine_readable_failure_for_bad_input(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "input.json"
            path.write_text('{"observations":')
            run = subprocess.run(
                [sys.executable, "-I", "-B", evidence.__file__, str(path)],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 1)
            report = json.loads(run.stdout)
            self.assertFalse(report["authority"])
            self.assertFalse(report["complete_executable_closure"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Check exact standalone host-reference selection and shell failure propagation."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    "rmsnorm_residual_v1",
    "qwen3_gqa_prefill_v1",
    "qwen3_paged_gqa_decode_v1",
    "qwen3_swiglu_v1",
    "qwen3_logits_compact_v1",
    "qwen3_rope_kv_v1",
    "qwen3_linear_reference_v1",
)
HARNESS = r'''
set -Eeuo pipefail
source "$1"
failure="$2"
run_step() {
  printf '%s' "$1"
  shift
  printf '\t%s' "$@"
  printf '\n'
  if [[ "$failure" == "$((++step_count))" ]]; then
    return 37
  fi
}
step_count=0
main "$3"
'''


class HostReferenceCiTests(unittest.TestCase):
    def test_vecadd_summary_matches_declared_fixture_cases(self):
        lines = (ROOT / "examples/verus_vecadd/run-verus.sh").read_text().splitlines()
        positives = sum(line.startswith("run_pass ") for line in lines)
        negatives = sum(line.startswith(("run_rejected ", "run_rejected_exact "))
                        for line in lines)
        self.assertGreater(positives, 0)
        self.assertGreater(negatives, 0)
        summary = (f"Verus fixture run passed: {positives} proof harnesses, "
                   f"{negatives} expected rejections")
        self.assertEqual(
            [line for line in lines if line.startswith("printf 'Verus fixture run passed:")],
            [f"printf '{summary}\\n'"],
        )

    def test_rope_proof_is_in_the_pinned_production_proof_job(self):
        workflow = (ROOT / ".github/workflows/row-softmax-v1.yml").read_text()
        proof_job = workflow.split("\n  proof-contract:", 1)[1].split("\n  host-contract:", 1)[0]
        self.assertEqual(proof_job.count("run: sh examples/qwen3_rope_kv_v1/run-verus.sh"), 1)
        self.assertLess(proof_job.index("run: examples/row_softmax_v1/run-verus.sh"),
                        proof_job.index("run: sh examples/qwen3_rope_kv_v1/run-verus.sh"))

    def run_lane(self, root, failure=0, lane="host-reference"):
        environment = dict(os.environ)
        environment.update(
            CARGO_TARGET_DIR=str(root / "target with spaces"),
            CI_LOG_DIR=str(root / "logs"),
            VERUS="/opt/verus/reference",
            FE2O3_RUNTIME_MODEL_VERUS="/opt/verus/runtime-model",
        )
        return subprocess.run(
            ["bash", "-c", HARNESS, "bash", str(ROOT / "scripts/ci-local.sh"), str(failure), lane],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15,
        )

    def expected(self, root):
        target = str(root / "target with spaces" / "host-reference")
        result = []
        for example in EXAMPLES:
            manifest = f"examples/{example}/Cargo.toml"
            base = ["--manifest-path", manifest, "--target-dir", target]
            commands = (
                ("format", ["cargo", "fmt", "--manifest-path", manifest, "--", "--check"]),
                ("clippy", ["cargo", "clippy", "--locked", *base,
                            "--all-targets", "--all-features", "--", "-D", "warnings"]),
                ("test", ["cargo", "test", "--locked", *base,
                          "--all-targets", "--all-features", "--", "--test-threads=1"]),
                ("release", ["cargo", "test", "--locked", "--release", *base,
                             "--all-targets", "--all-features", "--", "--test-threads=1"]),
                ("doc", ["env", "RUSTDOCFLAGS=-D warnings", "cargo", "doc",
                         "--locked", *base, "--no-deps"]),
            )
            result.extend([[f"host-reference-{example}-{kind}", *args]
                           for kind, args in commands])
        return result

    def test_exact_manifest_commands_and_shared_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_lane(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split("\t") for line in result.stdout.splitlines()],
                             self.expected(root))

    def test_every_failing_step_stops_before_the_next_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.expected(root)
            for failure in range(1, len(expected) + 1):
                with self.subTest(failure=failure):
                    result = self.run_lane(root, failure)
                    self.assertEqual(result.returncode, 37, result.stderr)
                    self.assertEqual([line.split("\t") for line in result.stdout.splitlines()],
                                     expected[:failure])

    def test_verus_exact_selection_and_every_failure_stops_the_lane(self):
        expected = [
            ["runtime-model-verus", "env", "VERUS=/opt/verus/runtime-model",
             str(ROOT / "crates/fe2o3-runtime-model/verus/verify-verus.sh")],
            ["verus-fixtures", "env", "VERUS=/opt/verus/reference",
             str(ROOT / "examples/verus_vecadd/run-verus.sh"), "--require"],
            ["scalar-gemm-verus", "env", "VERUS=/opt/verus/reference",
             str(ROOT / "examples/scalar_gemm_v1/run-verus.sh"), "--require"],
            ["mir-pliron-per-compilation-verus", "env", "VERUS=/opt/verus/reference",
             str(ROOT / "scripts/test-mir-pliron-per-compilation-verus.sh")],
            ["qwen3-rope-kv-verus", "env", "VERUS=/opt/verus/reference", "sh",
             str(ROOT / "examples/qwen3_rope_kv_v1/run-verus.sh")],
        ]
        with tempfile.TemporaryDirectory() as directory:
            for failure in range(len(expected) + 1):
                with self.subTest(failure=failure):
                    result = self.run_lane(Path(directory), failure, "verus")
                    self.assertEqual(result.returncode, 37 if failure else 0, result.stderr)
                    self.assertEqual([line.split("\t") for line in result.stdout.splitlines()],
                                     expected[:failure] if failure else expected)


if __name__ == "__main__":
    unittest.main()

"""Pure parser controls: these synthetic strings are not GPU evidence."""
import unittest
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import evaluate

def record(mode="tile", target="gfx950"):
    kernel = "gemm_autoresearch_simt_v1" if mode == "simt" else "gemm_autoresearch_v1"
    text = "\n".join(f"PASS {kernel} case={name} target={target}: 19x21x23, 4 workgroups, max_abs_error=0.000001"
                     for name in ["strided-tails", "nonquarter", "cancellation", "fractional-full"])
    text += "\n" + "\n".join(f"AUTORESEARCH_BENCH size={size} median_us=2.000 p10_us=1.000 p90_us=3.000 gflops=5.00"
                                  for size in [256, 512, 1024])
    return text + "\nAUTORESEARCH_SCORE geometric_mean_gflops=5.00\n"

class EvaluatorTests(unittest.TestCase):
    def test_matching_tile_and_simt_complete_records(self):
        for mode in ["tile", "simt"]:
            result = evaluate.parse_run(record(mode), target="gfx950", mode=mode)
            self.assertEqual([b["size"] for b in result["benchmarks"]], [256, 512, 1024])
            self.assertEqual(len(result["correctness_cases"]), 4)
    def test_wrong_target_and_mode_refuse(self):
        for text in [record(target="gfx942"), record(mode="simt")]:
            with self.assertRaises(RuntimeError): evaluate.parse_run(text, target="gfx950")
    def test_missing_or_duplicate_oracle_refuses(self):
        for text in [record().replace("case=nonquarter", "case=strided-tails"), record().replace("case=cancellation", "case=unknown")]:
            with self.assertRaises(RuntimeError): evaluate.parse_run(text, target="gfx950")
    def test_missing_size_or_duplicate_score_refuses(self):
        for text in [record().replace("size=1024", "size=512"), record()+"AUTORESEARCH_SCORE geometric_mean_gflops=5.00\n"]:
            with self.assertRaises(RuntimeError): evaluate.parse_run(text, target="gfx950")
    def test_nonfinite_or_nonpositive_or_unordered_timings_refuse(self):
        for old,new in [("median_us=2.000", "median_us=0.000"), ("p10_us=1.000", "p10_us=4.000"), ("geometric_mean_gflops=5.00", "geometric_mean_gflops=0.00")]:
            with self.assertRaises(RuntimeError): evaluate.parse_run(record().replace(old,new), target="gfx950")

class RunnerTests(unittest.TestCase):
    def test_already_reaped_runner_is_never_signalled_by_group_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(evaluate.subprocess, "Popen") as popen:
                process = popen.return_value
                process.returncode = 0
                process.wait.side_effect = KeyboardInterrupt
                with patch.object(evaluate.os, "killpg") as group_signal:
                    with self.assertRaises(KeyboardInterrupt):
                        evaluate.run_logged(["inert"], cwd=root, env={}, timeout=1,
                                            raw_path=root / "interrupt.log")
                    group_signal.assert_not_called()

    def test_success_and_nonzero_exit_retain_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for status in [0, 7]:
                raw = root / f"exit-{status}.log"
                code = f"import sys; print('output', flush=True); print('diagnostic', file=sys.stderr, flush=True); sys.exit({status})"
                result = evaluate.run_logged([sys.executable, "-c", code], cwd=root,
                                             env=os.environ.copy(), timeout=10, raw_path=raw)
                self.assertEqual(result.returncode, status)
                self.assertEqual(result.stdout, raw.read_text())
                self.assertIn("output", result.stdout)
                self.assertIn("diagnostic", result.stdout)

    def test_timeout_retains_partial_log_and_stops_owned_descendant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descendant = "import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path('descendant.pid').write_text(str(os.getpid())); print('descendant-ready',flush=True); time.sleep(30)"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{descendant!r}]); print('runner-ready',flush=True); time.sleep(30)"
            raw = root / "partial.log"
            with self.assertRaisesRegex(RuntimeError, "partial log"):
                evaluate.run_logged([sys.executable, "-c", parent], cwd=root,
                                    env=os.environ.copy(), timeout=2, raw_path=raw)
            # Setup assertions must not satisfy the expected runner-error oracle.
            self.assertIn("runner-ready", raw.read_text())
            self.assertIn("descendant-ready", raw.read_text())
            pid = int((root / "descendant.pid").read_text())
            process = Path(f"/proc/{pid}/stat")
            deadline = time.monotonic() + 2
            while True:
                try:
                    state = process.read_text().rsplit(")", 1)[1].split()[0]
                except FileNotFoundError:
                    state = "absent"
                if state in ["absent", "Z"] or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            self.assertIn(state, ["absent", "Z"])

    def test_nonpositive_timeout_refuses_before_any_runner(self):
        for value in ["0", "-1"]:
            with patch.object(sys, "argv", ["evaluate.py", "--label", "invalid", "--timeout-seconds", value]):
                with patch.object(evaluate, "run_logged") as runner:
                    with self.assertRaises(SystemExit) as result:
                        evaluate.main()
                    self.assertEqual(result.exception.code, 2)
                    runner.assert_not_called()

if __name__ == "__main__": unittest.main()

"""Exercise approved historical identities without forging Git objects."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "dco_checker", Path(__file__).resolve().parents[1] / "check-dco-range.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
APPROVED = (
    "3abb7b18ebe5e3cbc32002738bed0f7d72d4f745",
    "5ed3840a90db3f03a2cded9becffc0459b737f36",
    "7dfbe5cc453f12c1b8b32eeb49f08778876bf6ce",
    "d10f49bfedc26848285d20ec1399c193b3340f47",
    "3d473f9ffc850a3a762efee8cd4f210d363f3354",
    "2b4e21c3d730e2da7e62c7e44f81630290bd6a16",
    "817f3f7a4c69a2a0d95ce1833a3757c303ab1b01",
    "dadf99b4a219f33eb09e92df4e56665ee184744e",
    "ed05d16d1747428d20c791cd90e5f582397cf237",
    "01873d46188a9489a2c61e6d2d10f91b76f2240d",
    "1505e6fec9f6f2e0a88ea5328bd8423616db5e41",
    "5661e49e9001b2b204ef559f00080f43a1d61c65",
    "8bc97cc1b89e6df79479c7ffbb6588d615d581d7",
    "6849ca6ea395c9b8e26f1bbc7efc93e9832fec0d",
    "522d4aea1fa41aa8fb07dec5d0ea9b64daead082",
    "3e7a8e885814101a5096f7909027015a5d69e57a",
    "8c27e0029db793a56974ca8974df11fa678fd39e",
    "f445a37eb41700c6eee6f68da607dc598314fd89",
    "3173c1bc77b9286a11b85d0a844e2ccc4ab0a7b6",
    "188f388b5879dabd740c7b4f0c46588d4bb7da3d",
    "54d04795b3bd6c11b14b40dd5a23fafbf9708570",
    "1e9ca6251bfd95a4bd90fdfe4df8e63530ecb7c0",
)
UNSIGNED = "a" * 40
SIGNED = "b" * 40


class InheritedExceptions(unittest.TestCase):
    def test_registry_contains_only_the_exact_approved_identities(self):
        self.assertEqual(CHECKER.INHERITED_EXCEPTION_COMMITS, frozenset(APPROVED))
        self.assertEqual(
            CHECKER.INHERITED_EXCEPTION_REPOSITORIES,
            frozenset({"harsh-nod/fe2o3", "powderluv/fe2o3"}),
        )

    def run_check(self, commits, repo, signed=()):
        def identity(commit):
            body = "Signed-off-by: Test User <test@example.com>" if commit in signed else "unsigned"
            return "Test User", "test@example.com", body

        output = io.StringIO()
        with patch.object(CHECKER, "commits_in_range", return_value=list(commits)), patch.object(
            CHECKER, "commit_identity", side_effect=identity
        ), patch.object(CHECKER, "verified_dependabot", return_value=False), contextlib.redirect_stdout(output):
            count = CHECKER.check("c" * 40, "d" * 40, repo)
        self.assertEqual(count, len(commits))
        return output.getvalue()

    def test_exact_approved_commits_in_both_repositories(self):
        for repo in ("harsh-nod/fe2o3", "powderluv/fe2o3"):
            for commits in (*((commit,) for commit in APPROVED), APPROVED):
                with self.subTest(repo=repo, commits=commits):
                    output = self.run_check(commits, repo)
                    self.assertIn(f"0 commit(s); {len(commits)} approved inherited exception(s)", output)
                    for commit in commits:
                        self.assertIn(f"{repo}@{commit}", output)

    def test_other_repositories_and_lookalikes_reject(self):
        for repo in ("owner/repository", "harsh-nod/fe2o3-kernels", "powderluv/fe2o3-other", "other/fe2o3"):
            for commit in APPROVED:
                with self.subTest(repo=repo, commit=commit), self.assertRaisesRegex(CHECKER.DcoError, commit):
                    self.run_check((commit,), repo)

    def test_changed_identity_and_additional_unsigned_commit_reject(self):
        for repo in ("harsh-nod/fe2o3", "powderluv/fe2o3"):
            for commit in (
                UNSIGNED,
                *(("1" if approved[0] == "0" else "0") + approved[1:]
                  for approved in APPROVED),
            ):
                with self.subTest(repo=repo, commit=commit), self.assertRaisesRegex(CHECKER.DcoError, commit):
                    self.run_check((*APPROVED, commit), repo)

    def test_signed_and_excepted_counts_are_distinct(self):
        output = self.run_check((*APPROVED, SIGNED), "harsh-nod/fe2o3", (SIGNED,))
        self.assertIn(f"1 commit(s); {len(APPROVED)} approved inherited exception(s)", output)
        self.assertEqual(output.count("DCO approved inherited exception:"), len(APPROVED))

    def test_normal_signoff_takes_precedence(self):
        output = self.run_check(APPROVED, "powderluv/fe2o3", APPROVED)
        self.assertIn(f"{len(APPROVED)} commit(s); 0 approved inherited exception(s)", output)
        self.assertNotIn("DCO approved inherited exception:", output)


if __name__ == "__main__":
    unittest.main()

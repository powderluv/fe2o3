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


class CanonicalMergeProvenance(unittest.TestCase):
    """Mock API attestations test policy; they are not real signature verification."""

    def fixture(self):
        tree, left, right, base = (value * 40 for value in "1234")
        header = (f"tree {tree}\nparent {left}\nparent {right}\n"
                  "author Author <author@example.com> 1 +0000\n"
                  "committer GitHub <noreply@github.com> 1 +0000")
        payload = header + "\n\nmerge\n"
        raw = (header + "\ngpgsig fixture-signature\n\nmerge\n").encode()
        commit = CHECKER.hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        document = {"sha": commit, "committer": {"login": "web-flow"},
                    "parents": [{"sha": left}, {"sha": right}], "commit": {
                        "committer": {"name": "GitHub", "email": "noreply@github.com"},
                        "tree": {"sha": tree}, "verification": {
                            "verified": True, "reason": "valid", "payload": payload,
                            "signature": "fixture-signature"}}}
        pr = {"number": 7, "merged": True, "state": "closed", "merged_at": "2026-10-23T00:00:00Z",
              "merge_commit_sha": commit, "head": {"sha": right},
              "base": {"sha": base, "ref": "main", "repo": {"full_name": "harsh-nod/fe2o3"}}}
        associated = [{"number": 7, "merge_commit_sha": commit}]
        return commit, raw, document, pr, associated

    def evaluate(self, fixture, *, repo="powderluv/fe2o3", ancestor=True, tree="1" * 40):
        commit, raw, document, pr, associated = fixture

        def api(path):
            return {f"commits/{commit}": document,
                    f"commits/{commit}/pulls?per_page=100": associated, "pulls/7": pr}[path]

        def run(*arguments):
            if arguments == ("git", "--no-replace-objects", "cat-file", "commit", commit):
                return raw
            self.assertEqual(arguments, ("git", "--no-replace-objects", "merge-base",
                                         "--is-ancestor", pr["base"]["sha"], "2" * 40))
            if not ancestor:
                raise CHECKER.DcoError("PR base is not an ancestor")
            return b""

        with patch.object(CHECKER, "canonical_api", side_effect=api), patch.object(
            CHECKER, "run", side_effect=run
        ), patch.object(CHECKER, "clean_merge_tree", return_value=tree):
            return CHECKER.verified_canonical_merge(commit, repo)

    def test_exact_object_pr_and_tree_accept_in_both_destinations(self):
        for repo in ("harsh-nod/fe2o3", "powderluv/fe2o3"):
            self.assertTrue(self.evaluate(self.fixture(), repo=repo))

    def test_old_pr_base_is_allowed_only_as_actual_first_parent_ancestor(self):
        self.assertTrue(self.evaluate(self.fixture()))
        with self.assertRaises(CHECKER.DcoError):
            self.evaluate(self.fixture(), ancestor=False)

    def test_payload_signature_object_and_pr_mutations_reject(self):
        # Each mutation changes one independently required link.
        cases = [
            (2, ("sha",), "a" * 40), (2, ("committer", "login"), "other"),
            (2, ("commit", "committer", "name"), "Other"),
            (2, ("commit", "committer", "email"), "other@example.com"),
            (2, ("commit", "tree", "sha"), "a" * 40),
            (2, ("parents",), [{"sha": "3" * 40}, {"sha": "2" * 40}]),
            (2, ("commit", "verification", "verified"), False),
            (2, ("commit", "verification", "verified"), 1),
            (2, ("commit", "verification", "reason"), "unknown_key"),
            (2, ("commit", "verification", "payload"), "different signed bytes"),
            (2, ("commit", "verification", "signature"), "different signature"),
            (3, ("merged",), False), (3, ("state",), "open"),
            (3, ("merge_commit_sha",), "a" * 40), (3, ("head", "sha"), "2" * 40),
            (3, ("base", "repo", "full_name"), "other/fe2o3"),
            (3, ("base", "ref"), "feature"), (3, ("base", "sha"), "not-a-sha"),
            (3, ("number",), True),
        ]
        for index, path, value in cases:
            with self.subTest(path=path, value=value):
                fixture = self.fixture()
                target = fixture[index]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(self.evaluate(fixture))
        self.assertFalse(self.evaluate(self.fixture(), tree="a" * 40))
        self.assertFalse(self.evaluate(self.fixture(), repo="other/fe2o3"))

    def test_missing_ambiguous_or_malformed_pr_response_refuses(self):
        fixture = self.fixture()
        fixture[4].clear()
        self.assertFalse(self.evaluate(fixture))
        fixture = self.fixture()
        fixture[4].append(dict(fixture[4][0]))
        self.assertFalse(self.evaluate(fixture))
        fixture = self.fixture()
        del fixture[3]["merged"]
        with self.assertRaises(CHECKER.DcoError):
            self.evaluate(fixture)

    def test_object_bytes_cannot_be_replaced_under_same_sha(self):
        fixture = list(self.fixture())
        fixture[1] += b"extra"
        with self.assertRaisesRegex(CHECKER.DcoError, "local commit bytes"):
            self.evaluate(fixture)

    def test_unavailable_api_fails_closed(self):
        commit, raw, *_ = self.fixture()
        with patch.object(CHECKER, "run", return_value=raw), patch.object(
            CHECKER, "canonical_api", side_effect=CHECKER.DcoError("API unavailable")
        ), self.assertRaisesRegex(CHECKER.DcoError, "API unavailable"):
            CHECKER.verified_canonical_merge(commit, "powderluv/fe2o3")

    def test_integration_metadata_does_not_excuse_new_unsigned_contribution(self):
        commit, *_ = self.fixture()
        unsigned = "f" * 40
        with patch.object(CHECKER, "commits_in_range", return_value=[commit, unsigned]), patch.object(
            CHECKER, "commit_identity", return_value=("Author", "author@example.com", "unsigned")
        ), patch.object(CHECKER, "verified_dependabot", return_value=False), patch.object(
            CHECKER, "verified_canonical_merge", side_effect=lambda sha, repo: sha == commit
        ), self.assertRaisesRegex(CHECKER.DcoError, unsigned):
            CHECKER.check("a" * 40, "b" * 40, "powderluv/fe2o3")


    def test_ordinary_manual_octopus_and_ambiguous_headers_never_query_api(self):
        _, original, *_ = self.fixture()
        variants = [
            original.replace(b"parent " + b"3" * 40 + b"\n", b""),
            original.replace(b"\ngpgsig fixture-signature", b""),
            original.replace(b"committer GitHub <noreply@github.com>", b"committer Manual <manual@example.com>"),
            original.replace(b"\nauthor ", b"\nparent " + b"5" * 40 + b"\nauthor "),
            original.replace(b"\ngpgsig ", b"\ngpgsig duplicate\ngpgsig "),
            original.replace(b"\ngpgsig ", b"\nunknown header\ngpgsig "),
            original.replace(b"parent " + b"3" * 40, b"parent " + b"2" * 40),
        ]
        for raw in variants:
            commit = CHECKER.hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            with self.subTest(raw=raw), patch.object(CHECKER, "run", return_value=raw), patch.object(
                CHECKER, "canonical_api", side_effect=AssertionError("ineligible merge queried API")
            ):
                self.assertFalse(CHECKER.verified_canonical_merge(commit, "powderluv/fe2o3"))

    def test_integration_provenance_is_not_counted_as_author_signoff(self):
        commit, *_ = self.fixture()
        output = io.StringIO()
        with patch.object(CHECKER, "commits_in_range", return_value=[commit]), patch.object(
            CHECKER, "commit_identity", return_value=("Author", "author@example.com", "unsigned")
        ), patch.object(CHECKER, "verified_dependabot", return_value=False), patch.object(
            CHECKER, "verified_canonical_merge", return_value=True
        ), contextlib.redirect_stdout(output):
            self.assertEqual(CHECKER.check("a" * 40, "b" * 40, "powderluv/fe2o3"), 1)
        self.assertIn("0 commit(s); 0 approved inherited exception(s); 1 verified canonical", output.getvalue())
        self.assertIn("not author signoff", output.getvalue())

    def test_api_uses_fixed_host_repository_and_merge_sha_schema(self):
        with patch.object(CHECKER.subprocess, "check_output", return_value=b"{}") as request:
            self.assertEqual(CHECKER.canonical_api("commits/" + "a" * 40), {})
        argv = request.call_args.args[0]
        self.assertEqual(argv[:4], ["gh", "api", "--hostname", "github.com"])
        self.assertIn("X-GitHub-Api-Version: 2022-11-28", argv)
        self.assertEqual(argv[-1], "repos/harsh-nod/fe2o3/commits/" + "a" * 40)
        self.assertEqual(request.call_args.kwargs["timeout"], 30)
        fixture = self.fixture()
        del fixture[4][0]["merge_commit_sha"]
        with self.assertRaises(CHECKER.DcoError):
            self.evaluate(fixture)

    def test_malformed_api_json_fails_closed(self):
        with patch.object(CHECKER.subprocess, "check_output", return_value=b"{broken"), self.assertRaises(
            CHECKER.DcoError
        ):
            CHECKER.canonical_api("commits/" + "a" * 40)


class RealCanonicalMergeTree(unittest.TestCase):
    def git(self, repo, *args, input=None):
        return CHECKER.subprocess.check_output(
            ["git", "-C", str(repo), *args], input=input, stderr=CHECKER.subprocess.PIPE).strip()

    def fixture(self, root, conflict=False):
        repo = Path(root) / "source"
        self.git(Path(root), "init", "-q", str(repo))
        self.git(repo, "config", "user.name", "Fixture Author")
        self.git(repo, "config", "user.email", "fixture@example.com")
        (repo / "common").write_text("base\n")
        (repo / ".gitattributes").write_text("common merge=custom\n")
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-qm", "base")
        base = self.git(repo, "rev-parse", "HEAD").decode()
        for name in ("left", "right"):
            self.git(repo, "checkout", "-q", "-B", name, base)
            (repo / ("common" if conflict else name)).write_text(name + "\n")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", name)
        parents = [self.git(repo, "rev-parse", name).decode() for name in ("left", "right")]
        marker = Path(root) / "driver-ran"
        self.git(repo, "config", "merge.custom.driver", f"touch '{marker}'; exit 0")
        return repo, parents, marker

    def snapshot(self, repo):
        return (self.git(repo, "rev-parse", "HEAD"), self.git(repo, "status", "--porcelain"),
                (repo / ".git/config").read_bytes(), (repo / ".git/index").read_bytes(),
                sorted(str(p.relative_to(repo)) for p in (repo / ".git/objects").rglob("*") if p.is_file()))

    def invoke(self, repo, parents):
        previous = CHECKER.os.getcwd()
        try:
            CHECKER.os.chdir(repo)
            # The helper must not inherit arbitrary Git config injection.
            with patch.dict(CHECKER.os.environ, {"GIT_CONFIG_COUNT": "1",
                            "GIT_CONFIG_KEY_0": "merge.custom.driver",
                            "GIT_CONFIG_VALUE_0": "exit 0"}):
                return CHECKER.clean_merge_tree(parents)
        finally:
            CHECKER.os.chdir(previous)

    def test_real_clean_merge_has_exact_combined_tree_and_preserves_source(self):
        with CHECKER.tempfile.TemporaryDirectory() as temporary:
            repo, parents, marker = self.fixture(temporary)
            entries = {}
            for parent in parents:
                for line in self.git(repo, "ls-tree", parent).splitlines():
                    entries[line.split(b"\t", 1)[1]] = line
            expected = self.git(repo, "mktree", input=b"\n".join(entries[k] for k in sorted(entries)) + b"\n").decode()
            before = self.snapshot(repo)
            self.assertEqual(self.invoke(repo, parents), expected)
            self.assertEqual(self.snapshot(repo), before)
            self.assertFalse(marker.exists())
            # An extra-content merge tree cannot equal the exact combined tree.
            (repo / "extra").write_text("not contributed by either parent\n")
            self.git(repo, "add", "extra")
            extra_tree = self.git(repo, "write-tree").decode()
            self.assertNotEqual(self.invoke(repo, parents), extra_tree)

    def test_real_conflict_refuses_without_parent_driver_or_checkout_changes(self):
        with CHECKER.tempfile.TemporaryDirectory() as temporary:
            repo, parents, marker = self.fixture(temporary, conflict=True)
            before = self.snapshot(repo)
            with self.assertRaisesRegex(CHECKER.DcoError, "cannot be reproduced"):
                self.invoke(repo, parents)
            self.assertEqual(self.snapshot(repo), before)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Require exact DCO sign-offs across a bounded Git commit range."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any


COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
DEPENDABOT_NAME = "dependabot[bot]"
DEPENDABOT_EMAIL = "49699333+dependabot[bot]@users.noreply.github.com"
DEPENDABOT_SIGNOFF = "Signed-off-by: dependabot[bot] <support@github.com>"
MAX_COMMITS = 1024
# Exact inherited exceptions approved by the maintainer on 2026-09-18 and 2026-09-23.
# See docs/dco-inherited-exceptions.md; no caller-supplied waiver is supported.
INHERITED_EXCEPTION_REPOSITORIES = frozenset({"harsh-nod/fe2o3", "powderluv/fe2o3"})
INHERITED_EXCEPTION_COMMITS = frozenset({
    "3abb7b18ebe5e3cbc32002738bed0f7d72d4f745",
    "5ed3840a90db3f03a2cded9becffc0459b737f36",
    "7dfbe5cc453f12c1b8b32eeb49f08778876bf6ce",
    "d10f49bfedc26848285d20ec1399c193b3340f47",
    "3d473f9ffc850a3a762efee8cd4f210d363f3354",
    # Additional exact inherited exceptions approved on 2026-10-20.
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
})


class DcoError(Exception):
    pass


def run(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(arguments, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise DcoError(f"command failed: {' '.join(arguments)}: {error}") from error


def mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DcoError(f"Dependabot API response lacks {label}")
    return value


def verified_dependabot(commit: str, author: str, email: str, body: str, repo: str) -> bool:
    if (
        author != DEPENDABOT_NAME
        or email != DEPENDABOT_EMAIL
        or DEPENDABOT_SIGNOFF not in body.splitlines()
    ):
        return False
    try:
        document = json.loads(
            run(
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                "-H",
                "X-GitHub-Api-Version: 2026-03-10",
                f"repos/{repo}/commits/{commit}",
            )
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DcoError(f"{commit}: invalid Dependabot API response: {error}") from error
    document = mapping(document, "document")
    commit_record = mapping(document.get("commit"), "commit record")
    api_author = mapping(document.get("author"), "author account")
    api_committer = mapping(document.get("committer"), "committer account")
    git_author = mapping(commit_record.get("author"), "Git author")
    git_committer = mapping(commit_record.get("committer"), "Git committer")
    verification = mapping(commit_record.get("verification"), "verification")
    return (
        document.get("sha") == commit
        and api_author.get("login") == DEPENDABOT_NAME
        and api_committer.get("login") == "web-flow"
        and git_author.get("name") == DEPENDABOT_NAME
        and git_author.get("email") == DEPENDABOT_EMAIL
        and git_committer.get("name") == "GitHub"
        and git_committer.get("email") == "noreply@github.com"
        and verification.get("verified") is True
        and verification.get("reason") == "valid"
    )


def commits_in_range(base: str, head: str) -> list[str]:
    raw = run("git", "rev-list", "--reverse", f"{base}..{head}")
    try:
        commits = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise DcoError("commit range is not ASCII") from error
    if not commits:
        raise DcoError("pull request has an empty commit range")
    if len(commits) > MAX_COMMITS:
        raise DcoError(f"pull request exceeds the {MAX_COMMITS}-commit bound")
    if any(COMMIT_PATTERN.fullmatch(commit) is None for commit in commits):
        raise DcoError("commit range contains a noncanonical identity")
    return commits


def commit_identity(commit: str) -> tuple[str, str, str]:
    raw = run("git", "show", "-s", "--format=%an%x00%ae%x00%B", commit)
    try:
        return tuple(raw.decode("utf-8").split("\0", 2))  # type: ignore[return-value]
    except (UnicodeDecodeError, ValueError) as error:
        raise DcoError(f"{commit}: commit identity is not canonical UTF-8") from error


def check(base: str, head: str, repo: str) -> int:
    if COMMIT_PATTERN.fullmatch(base) is None or COMMIT_PATTERN.fullmatch(head) is None:
        raise DcoError("invalid pull-request commit identity")
    if REPOSITORY_PATTERN.fullmatch(repo) is None:
        raise DcoError("invalid GitHub repository identity")
    missing = []
    excepted = []
    signed = 0
    commits = commits_in_range(base, head)
    for commit in commits:
        author, email, body = commit_identity(commit)
        expected = f"Signed-off-by: {author} <{email}>"
        if expected in body.splitlines():
            signed += 1
            continue
        if verified_dependabot(commit, author, email, body, repo):
            signed += 1
            continue
        if repo in INHERITED_EXCEPTION_REPOSITORIES and commit in INHERITED_EXCEPTION_COMMITS:
            excepted.append(commit)
            continue
        missing.append(f"{commit}: missing exact trailer {expected}")
    if missing:
        raise DcoError("\n".join(missing))
    for commit in excepted:
        print(f"DCO approved inherited exception: {repo}@{commit}")
    print(f"DCO sign-off present on {signed} commit(s); {len(excepted)} approved inherited exception(s)")
    return len(commits)


def main() -> None:
    if not sys.flags.isolated:
        raise DcoError("checker must run in Python isolated mode (-I)")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    check(args.base, args.head, args.repo)


if __name__ == "__main__":
    try:
        main()
    except DcoError as error:
        print(f"DCO check: {error}", file=sys.stderr)
        raise SystemExit(1)

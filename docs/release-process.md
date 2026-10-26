# Developer-preview release process

The planned first fe2o3 developer preview is a source-only release from the
canonical repository. It will not publish crates to crates.io or produce
prebuilt compiler, runtime, or GPU binaries.

The first developer-preview release is currently blocked. `powderluv` has not
publicly accepted the proposed protected-evidence and release-review duties,
and the required historical licensing consent is not yet recorded. GitHub
reviewer and CODEOWNERS configuration does not substitute for acceptance.

Canonical source, issues, and releases live in `harsh-nod/fe2o3`.
`powderluv/fe2o3` is the code mirror and the protected parity protocol's
design-bound repository. The protocol is not deployed or qualified: both
repositories are user-owned, GitHub merge queues are unavailable there, and no
protected parity publisher environment is configured. The canonical source
release environment described below does not grant parity authority.

## Version and artifact policy

- Preview tags use **vMAJOR.MINOR.PATCH-dev.N**.
- Tags are annotated and point to a commit reachable from canonical **main**.
- The custom source archive is the release artifact. GitHub's automatically
  generated source links are convenience copies and are not covered by the
  published checksum file.
- Each release includes a SHA-256 checksum manifest, an SPDX 2.3 source SBOM,
  release metadata, and a GitHub build-provenance attestation.
- All workspace packages use **publish = false**. Enabling crates.io publication
  requires a separate review of names, metadata, semver ownership, licenses,
  and the complete non-path dependency closure.

## Prepare

1. Start from a clean checkout of canonical **main**.
2. Confirm generic CI passed for the exact commit. Run any hardware,
   compile-only, or formal qualification required by the changes and link exact
   evidence in the release pull request. Do not update the parity-status ledger:
   generic CI rejects it because neither repository can currently acquire or
   import a protected receipt. Canonical release automation is not a substitute
   for that unavailable authority.
3. Move user-visible entries from **Unreleased** in CHANGELOG.md to a dated
   heading such as **[0.1.0-dev.1] - YYYY-MM-DD**. Add comparison links at the
   bottom of the changelog.
4. Confirm public documentation identifies unsupported targets and
   distinguishes generic, simulated, compile-only, and hardware results.
5. Merge the release-preparation pull request through the protected branch.
6. Synchronize the exact release commit to the mirror's **main** branch. From
   the canonical checkout, verify both the commit and its exact tree are
   reachable from the mirror before creating a tag:

       release_commit="$(git rev-parse HEAD)"
       release_tree="$(git rev-parse 'HEAD^{tree}')"
       bash scripts/check-release-mirror.sh \
         --commit "${release_commit}" \
         --tree "${release_tree}"

   This check fetches `powderluv/fe2o3` directly and fails closed on an absent,
   divergent, or unavailable mirror **main**. Do not tag or dispatch the
   release workflow until it passes.

## Canonical release controls

Before the first `v*` tag, install the canonical release controls. The bootstrap
pins the GitHub user IDs, requires that no `v*` tag already exists, creates a
temporary no-bypass creation/update/deletion guard, configures the reviewed
**release** environment with administrator bypass disabled and an exact
**main** deployment policy, admits only `harsh-nod` for tag creation, and then
reduces the guard to permanent no-bypass update/deletion protection.

    bash scripts/canonical-release-controls.sh bootstrap \
      --repo harsh-nod/fe2o3 \
      --release-user-id 3144552 \
      --reviewer-user-id 74956

An interrupted bootstrap is fail-closed and may be resumed only while no `v*`
tag exists. The tool verifies exact recognized state instead of replacing a
divergent control. Ruleset administrators can still edit or disable repository
rules, so verify all four API controls immediately before every tag:

    bash scripts/canonical-release-controls.sh verify \
      --repo harsh-nod/fe2o3 \
      --release-user-id 3144552 \
      --reviewer-user-id 74956

The environment requires `powderluv` approval and prevents self-review.
`harsh-nod` must dispatch the release workflow from canonical **main** and
`powderluv` must approve it. A `powderluv` dispatch cannot be self-approved.
That review duty remains pending explicit acceptance, so releases remain
blocked even though the fail-closed environment control is installed.

## Tag

Create and push an annotated tag only after the exact **main** commit passes the
full CI workflow, the mirror check in **Prepare** passes, and the pending
maintainer and licensing approvals are recorded:

    git fetch origin main
    git switch --detach <qualified-commit>
    git tag -a v0.1.0-dev.1 -m "fe2o3 0.1.0-dev.1 developer preview"
    git push origin refs/tags/v0.1.0-dev.1

Do not move or reuse a published tag. Correct a release with a new preview
sequence number.

## Build the draft

In the canonical repository, select **main** and run **Developer preview source
release** with the exact tag. The workflow:

1. validates the tag syntax, annotated tag object, canonical-main ancestry,
   exact commit/tree reachability from mirror **main**, and dated changelog
   entry;
2. requires a successful full CI workflow for the exact commit;
3. validates locked Cargo metadata without publishing any package;
4. creates a deterministic gzip-compressed git archive;
5. inventories tracked source files in an SPDX 2.3 SBOM;
6. emits checksums and release metadata;
7. records the exact successful CI run and Generic validation job;
8. revalidates repository, branch, tag, actor, triggering actor, and mirror
   reachability immediately before provenance attestation;
9. attests those artifacts with GitHub artifact provenance; and
10. repeats the authorization and mirror checks immediately before creating a
    draft prerelease.

The release job uses the **release** GitHub environment installed by
`canonical-release-controls.sh`.

## Canonical branch rules

The release baseline requires a user-owned-compatible ruleset for the canonical
default branch. This is a release prerequisite, not a claim that the ruleset is
deployed. On 2026-10-26, the canonical `main` branch API reported
`protected: false`; the active repository rulesets covered release-tag creation
and immutability, not branch protection. Reviewed source publication and passing
CI do not satisfy this missing release control. Issue
[#267](https://github.com/harsh-nod/fe2o3/issues/267) remains open.

The proposed baseline has no bypass actors, rejects deletion and
non-fast-forward updates, requires squash-only pull requests, and requires one
approval with CODEOWNERS, stale-review dismissal, last-push approval, and thread
resolution. Its strict required checks are pinned to the GitHub Actions app:

- **Fork-safe preflight**
- **Generic parity policy gate**
- **Generic validation**

An administrator can review and create the ruleset once with:

    actions_app_id="$(gh api --hostname github.com apps/github-actions --jq .id)"
    bash scripts/canonical-repository-rules.sh render \
      --actions-integration-id "${actions_app_id}" | jq .
    bash scripts/canonical-repository-rules.sh bootstrap \
      --repo harsh-nod/fe2o3 \
      --actions-integration-id "${actions_app_id}"

`bootstrap` refuses to update or replace an existing ruleset and refuses any
repository other than `harsh-nod/fe2o3`. Verify the live policy after creation
and before tagging every release:

    actions_app_id="$(gh api --hostname github.com apps/github-actions --jq .id)"
    bash scripts/canonical-repository-rules.sh verify \
      --repo harsh-nod/fe2o3 \
      --actions-integration-id "${actions_app_id}"

Do not infer that this baseline is active from the checked-in renderer or a
successful source integration. The live `verify` command must pass before a
release. Installing or changing repository policy is a separate administrator
operation, not part of a documentation update.

This proposed canonical baseline intentionally omits merge queues,
required-workflow rules, and the protected signed-evidence check. It does not
deploy or grant the separate protected parity publication authority described
above.

## Review and publish

Before publishing the draft:

- download the assets and run **sha256sum --check SHA256SUMS**;
- verify the attestation with GitHub CLI:

      gh attestation verify fe2o3-0.1.0-dev.1.tar.gz --repo harsh-nod/fe2o3

- confirm the archive expands under one versioned directory;
- compare the release commit and SBOM namespace to the tag;
- rerun `scripts/check-release-mirror.sh` for the recorded commit and tree;
- verify release notes do not claim unrun hardware or protected evidence; and
- perform a clean-checkout getting-started run from the release archive.

Publish the draft only after those checks pass. The exact commit must already
be reachable from **powderluv/fe2o3** **main** before the canonical tag and draft
are created; a post-publication commit sync is too late. Mirror the canonical
tag after publication. Do not create an independent mirror release or change
the canonical source/release URLs in release metadata. No current repository
may publish new protected parity evidence; designed but undeployed workflow
paths must not be relabeled as authority.

## Recovery

If workflow validation or artifact review fails, leave the release unpublished,
fix the problem on **main**, and issue a new tag. Never replace assets underneath
a published release.

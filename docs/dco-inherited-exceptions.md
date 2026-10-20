# Approved inherited DCO exceptions

On 2026-09-18 the maintainer explicitly accepted these two inherited commits
as documented DCO exceptions for publication to `harsh-nod/fe2o3` and
`powderluv/fe2o3`:

| Exact commit | Existing subject |
| --- | --- |
| `3abb7b18ebe5e3cbc32002738bed0f7d72d4f745` | perf(kfd): borrow engineering dispatch payload validators [skip ci] |
| `5ed3840a90db3f03a2cded9becffc0459b737f36` | Amortize CPU-only ordered dispatch preparation fences [skip ci] |

The approval was: "Accept these two inherited exceptions". It was recorded in
[#271](https://github.com/harsh-nod/fe2o3/issues/271#issuecomment-5737431573).
Both commits already existed on canonical main. Their history is preserved;
this record does not add signoff trailers or claim their missing trailers exist.

On 2026-09-23 the maintainer explicitly approved three additional inherited
canonical-main commits for publication to the same two repositories:

| Exact commit | Existing subject |
| --- | --- |
| `7dfbe5cc453f12c1b8b32eeb49f08778876bf6ce` | Refresh exact vendored SDK manifest and target-roster regression [skip ci] |
| `d10f49bfedc26848285d20ec1399c193b3340f47` | feat(kfd): add opt-in ordered64 engineering batches [skip ci] |
| `3d473f9ffc850a3a762efee8cd4f210d363f3354` | feat(kfd): add opt-in ordered64 dispatch timestamps [skip ci] |

The approval was: "Accept these three additional exceptions", recorded in
[#275](https://github.com/harsh-nod/fe2o3/issues/275#issuecomment-5798687623).
Their existing history and missing-trailer status are likewise preserved.

On 2026-10-20 the maintainer explicitly approved 17 additional inherited
commits for publication to the same two repositories:

| Exact commit | Existing subject |
| --- | --- |
| `2b4e21c3d730e2da7e62c7e44f81630290bd6a16` | compiler-bound-explicit-reference-enrollment-requests |
| `817f3f7a4c69a2a0d95ce1833a3757c303ab1b01` | test: exercise admitted reference enrollment loan flows [skip ci] |
| `dadf99b4a219f33eb09e92df4e56665ee184744e` | test: pin loan fixture sysroot and publication accounting [skip ci] |
| `ed05d16d1747428d20c791cd90e5f582397cf237` | refactor: share inert reference enrollment decoder [skip ci] |
| `01873d46188a9489a2c61e6d2d10f91b76f2240d` | test: cover conditional CPU origin capture projection [skip ci] |
| `1505e6fec9f6f2e0a88ea5328bd8423616db5e41` | feat: quote inert enrollment retained storage [skip ci] |
| `5661e49e9001b2b204ef559f00080f43a1d61c65` | test(compiler): follow enrollment delegation in production route guards |
| `8bc97cc1b89e6df79479c7ffbb6588d615d581d7` | feat: add inert rustc enrollment inventory codec |
| `6849ca6ea395c9b8e26f1bbc7efc93e9832fec0d` | fix(custodian): retain one owned account and join raw subjects before recovery |
| `522d4aea1fa41aa8fb07dec5d0ea9b64daead082` | refactor(invocation): parse enrollment with a bounded fixed-schema visitor |
| `3e7a8e885814101a5096f7909027015a5d69e57a` | Merge qualified runtime main into the M1 integration candidate |
| `8c27e0029db793a56974ca8974df11fa678fd39e` | docs: record M1 joined source qualification for publication [skip ci] |
| `f445a37eb41700c6eee6f68da607dc598314fd89` | Merge PR #298: runtime integrity and reproducible CPU foundation |
| `3173c1bc77b9286a11b85d0a844e2ccc4ab0a7b6` | Fix store retirement ordering, Rustls dependency and bounded host CI shards (#299) |
| `188f388b5879dabd740c7b4f0c46588d4bb7da3d` | Add passive compiler-map diagnostics to the CPU policy gate (#300) |
| `54d04795b3bd6c11b14b40dd5a23fafbf9708570` | Merge pull request #301: bind simulator schedules and document CPU replay |
| `1e9ca6251bfd95a4bd90fdfe4df8e63530ecb7c0` | Merge pull request #302 from harsh-nod/release/readiness-owned-audit-20261019-c6ce87a |

The approval was: "yes approve it and keep going", recorded in
[mirror PR #30](https://github.com/powderluv/fe2o3/pull/30#issuecomment-6293576380).
It applies only to these exact 17 commit identities in `harsh-nod/fe2o3` and
`powderluv/fe2o3`. Their historical missing trailers remain missing; this
record neither adds trailers nor forges an author attestation or rewrites history.
Together with the five earlier approvals, the registry contains 22 exceptions.

`scripts/check-dco-range.py` recognizes only these full commit identities in
the two named repositories and reports exceptions separately from signoffs.
There is no command-line, environment, prefix, author-wide or ancestry-based
waiver. Every other commit, including merges and future KFD work, still needs
its exact author signoff or the existing independently verified Dependabot rule.
The exceptions do not waive tests, artifact checks or any publication requirement
other than these 22 missing DCO trailers.

# Source-use and borrowed-local component checkpoint — 2026-09-26

This is a private compiler-component checkpoint for #280/#282, not completed
nominal-helper compilation. Accepted broad exits remain
**M1/V1/V2/U1/U2/U3 (6/18)**. It adds no public assembly API, operation stream,
LLVM continuation, GPU execution or debugger capture.

## Implemented

The source-use selector identifies supported place occurrences by source block,
statement or terminator, and operand ordinal. It derives the actual access kind,
atomic contract and source provenance. A private factory joins that occurrence
to the retained-input owner and existing checked-reference origins on the
original work ledger. It does not manufacture a semantic operation index.

Borrowed local contracts expose exact source-local decisions without rebuilding
owned copies of the rich tables. The immutable-local decision preserves the
existing four terms: non-entry local, exactly one definition, an available
assignment, and no escape. Allocation/provenance fallback order and the ordinary
checked-reference path remain unchanged. The rich-source ledger pair is used
only for identity equality inside the existing loan; it is not dereferenced or
exported.

The two factories each extend a single-visit pending assembly. They are separate
component paths, not sequential visits to the same started owner. Accepted
partial storage remains retained until the existing outer postflight/refund
boundary. Wrapper frames and query work are prepaid through that original
ledger. A foreign ledger, invalid coordinate or unsupported call/effect boundary
cannot manufacture a successful source-use result.

Paid emit/bind helpers retain partial storage, but are not wired into a completed
actual-source operation stream. A missing reference origin is data, not proof
that an access is checked. Calls, tail calls and drops remain incomplete at this
boundary. Guard semantic sites remain unassigned.

## Qualification

Independent source reviews covered both components and the exact integration.

- 331 model tests and 2,723 backend tests passed; 189 backend tests remain ignored.
- Backend/extractor builds, diff checks and 17 existing telemetry controls passed.
- The 40 additional component controls cover source-use, local-contract and rich-ledger behavior.
- The ordinary ladder passed all 36 composition and two tiled observations.
- Independent lossless comparison found all 38 observation bodies and 52 artifacts byte-identical to the published source-origin checkpoint.

The component controls do not enter the two new actual-source factories.
A separate real-source observer and independent occurrence/scalar oracle remain
pending. Their results must not be inferred from the ordinary-path parity run.

The unchanged source snapshot tested here is
`ee0acb52803dd67c614aac350f071debb81808ab123af635812adefbf892bfff`.
Retained qualification records:

| Record | SHA-256 |
| --- | --- |
| Component regression | `5425d43b6fc3cfca6e625e9bd04b83102d7a9baf53d55b816a7f2eb11593796f` |
| Ordinary ladder | `841e2202810a80be75fb38cd4c9bea44996e6860a24f8e26293b81c80560f941` |
| Ordinary lossless readback | `016005cc76a6aee210dcbb2f998bcc0fd057b03908f95fc0164d45e35611a403` |

## Parallel debugger work

The private commit-site diagnostic debugger build and actual static inspection
passed. Measured native adapter size is 8,728 bytes; its complete logical
reservation is 14,488 bytes within the unchanged 65,536-byte allowance.
The corrected startup evidence binder and all 153 captured-evidence decoder
controls passed. The binder observed 78 generated data files and matched 54
Python files against the actual source; no debugger or target was executed.

These results qualify source/build/evidence prerequisites only. A fresh helper
profile, full operational control graph, startup/loaded-closure observation and
bounded native attempt are still required. The prior failed attempts remain
retained; no public capture gate is enabled.

## Next implementation boundary

The independent real-source observer must first qualify the new factories and
their actual per-scope accounting. Remaining production work includes an
authenticated operation cursor, complete block emission, defined-call routing,
source/ranked correspondence, mandatory verification and normal compilation
continuation. These steps cannot be replaced by successful data queries or by
marking a pending assembly ready.

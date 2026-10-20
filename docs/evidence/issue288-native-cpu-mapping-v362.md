# V362 Mapped CPU Recovery

## Scope

This source change adds an explicit, content-only mapping route to conditional
source, source-through-F, native V5, and fixed policy-file recovery. It does not
authenticate compiler capture, an installed policy, a carriage, currentness,
protected custody, a proof run, or a GPU execution. All existing registration-only
and ordered origin-only public APIs remain available and select their old codecs.

The dependency is the compiler-owned `RustcEnrollmentInventoryRefV1` codec. No
parallel codec, closed source-packet extension, inferred origin, or fallback from
malformed mapped inventory is introduced here.

## Public Inputs

`NativeConditionalCpuMappingExpectationV1` contains independently obtained
invocation identity, native-policy identity, policy generation, and original
descriptor enrollment-binding count. These public values are inert expectations,
not acceptance authority. The installed-policy/native owners must derive them
from their authenticated original invocation and retained policy selection.

Lower source/final entrypoints additionally borrow an already decoded inventory
through `NativeConditionalCpuMappingContextV1`. Their external input is explicitly
content-only. Native and policy-file entrypoints accept only the independent
coordinates and decode the actual retained
`handoff.capsule().rustc_identity_inventory().canonical_preimage()` internally.
An external same-header inventory cannot replace that field through these APIs.

Do not interchange the raw full-wrapper SHA256, the codec's domain-separated
framing trailer, or the existing domain-separated inventory receipt identity.
The native-policy identity is also not the semantic policy-file hash.

## Checks

The existing reconstructed source visitor performs the projection; there is no
second MIR decoder. The complete canonical KernelEntry subset must match actual
semantic function IDs, KernelRoot roles, canonical kernel Instance identities and
kernel-binding identities. The projection then follows actual source/packet order
and checks logical-name byte length/hash and launch binding. Kernel IDs, packet
positions and descriptor ordinals are not interchangeable.

The shared codec checks canonical tags and the complete enrollment ordinal census.
The consumer independently compares all header coordinates, retains the existing
256-binding CPU enrollment limit, and selects registration V1 or enrollment V2
before decoding a CPU leaf. After that single strict decode and the existing
source/root/name association, both CPU kernel and CPU reference canonical function
identities must match the projected mapping. Existing full CPU correspondence,
recipe, genuine Request continuation, formula import, and enclosing postchecks
still run. A reference Instance does not have to be a GPU root. Existing unsupported
FFI behavior is unchanged.

## Accounting

All work uses the original budget. Callers retain and prepay actual backing
capacities, not only selected slice lengths. Native entry pays the shared codec's
published working set and live borrowed view/context headers before decoding its
actual inventory. The projection pays its Vec header, actual allocation capacity,
row/hash/result/callback working values, every row walk, and every name/identity
comparison. This is a logical owned/scratch quote, not a theorem about compiler
stack frames or aggregate process RSS.

The lower borrowed mapping context exposes only its canonical wire byte length.
Its backing check is a visible minimum, not evidence of the allocation's actual
capacity. Lower callers still retain and prepay their actual owner; native recovery
separately retains the original handoff's actual capacity rather than replacing
that quote with the inventory length.

Projection owners drop before successful release of their known temporary extent.
Wrong ledger, changed storage account, or underfloor observations override callback
results. Through-F, native, and policy-file failures/unwinds keep terminal charges;
the existing source-only classified-refund protocol is unchanged. Native original-
account overlap includes expectation backing and its final exact-floor checks
include mapped working storage. Old enum/Option representations and old branch
bills are retained explicitly.

Mapped projection refuses prior sticky work/storage denials before allocation and
checks denial history again after authenticating the callback's ledger/account
and floor. A callback cannot suppress a denial and return an accepted owner: the
returned value drops before refusal and known temporary charges remain terminal.
This mapped-only gate does not change legacy prior-denial behavior or bills.

## Qualification Boundary

Authored coverage consists of twelve projection/identity/resource component tests,
four native accounting component tests, two compile-only call-shape helpers and
two compile-fail documentation cases. It covers noncontiguous/permuted roots,
mixed origins, descriptor ordinals permuted independently of canonical row order,
independent reference identity, coherent header/kernel/name/binding
mutants, missing/extra/duplicate roots, legacy bills, exact/one-short resource
thresholds, nonzero floors, refusal, unwind, wrong ledger, underfloor checks,
swallowed work/storage denials and prior-denial refusal before any callback.

The projection fixture uses public inert semantic declarations and the actual
private projection helper. It does not manufacture a ReplayedNativeSource,
Request, imported proof or native owner, and is not an end-to-end positive recovery
demo. Rust compilation/tests, documentation tests and an authenticated original-
capture/native recovery integration remain required. No new solver or GPU run is
part of this source task. Dependency qualification reported by its owner does not
qualify this consumer change.

Qualification must include the existing legacy exact-billing/resource suites and
the complete affected verifier regression suites, not only these sixteen new
tests. Source review and standalone formatting do not establish those results.

The compiler dependency was imported as `b47b242cb1326aca78fcb8a56e52a51e95656dbe`
from owner commit `ca12fd4bf628fe7c3b1fca3a579ee04f1b58a159`. Its exact three-file
format-patch was SHA256
`5475c79ef94748763814ce177f01d766a7291c5726b8d7f3c898899438be74e1`.
The import changes commit metadata/provenance, not the compiler-owned codec bytes.

Final source formatting used the pinned standalone nightly-2026-04-03 rustfmt
on the thirteen owned Rust files, restricted to CPUs 2,3 with a 30-second fuse
and `skip_children=true`. It completed with exit 0, empty output and a successful
`git diff --check`. The retained remote receipt is
`/home/harsh/f288-v362-mapping-rustfmt-final-20261008.json`, SHA256
`b0eb1e23e90b84bc87bb25b2ff9e368fbd4ff6c59b5b3fbb6014af92a6d4c5f5`.
This evidence is source formatting only, not Rust typechecking or test execution.

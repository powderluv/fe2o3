# fe2o3

`fe2o3` is an experimental single-source Rust GPU stack for AMD GPUs.

The current architecture keeps the working AMD runtime while incrementally
replacing the elementwise MIR recognizer with a target-neutral compiler
pipeline and adding source-level Verus contracts. The general form remains
incomplete, while bounded `gfx942` vertical slices exercise the compiler,
artifact, runtime, and proof boundaries described below. See the
[living v2 architecture](docs/architecture-v2.md),
[production compiler convergence design](docs/production-pipeline-convergence-v1.md),
[compiler execution subject V1](docs/compiler-execution-subject-v1.md),
[compiler execution attestation protocol V1](docs/compiler-execution-attestation-v1.md),
[receipt-bearing Worker V3 load envelope V2](docs/worker-v3-receipt-bearing-load-envelope-v2.md),
[gfx942 production LDS reduction](docs/gfx942-production-lds-reduction-v1.md),
[workspace ownership policy](docs/workspace-layers-and-ownership.md),
[Pliron Wave 0 architecture](docs/pliron-wave0-architecture.md),
[cuda-oxide parity matrix](docs/cuda-oxide-parity-matrix.md),
[evidence-backed parity dashboard](docs/generated/cuda-oxide-parity-dashboard.md),
[verification model](docs/verification-model.md),
[GPU safety contract v1](docs/gpu-safety-contract-v1.md), and
[implementation roadmap](docs/implementation-roadmap-v2.md). The
[testing guide](docs/testing.md) defines the generic, Verus, ROCm compile, and
hardware execution lanes.

CPU-only source extraction is available through the explicit, authority-free
[`fe2o3-export-sim` bundle workflow](docs/simulation-bundle-v1.md). It reuses
the production source/MIR/KIR stages under extraction-only custody, records
that no compiler-execution subject is available, and never silently falls back
from hardware execution.

The same CPU-only path can publish and replay a strict bounded
[`fe2o3-simulation-schedule-v1`](docs/semantic-schedule-v1.md) decision record
through `fe2o3-kir-sim`, including exact bundle/request custody and debugger
capture. This is deterministic simulator replay, not GPU scheduling evidence.

On a KFD-capable Linux host, `fe2o3-debug live-kfd` binds that exact V2
bundle/request reference to separately inspected HSACO bytes and an exact
launch-owned executable. Its agent-native V3 JSONL protocol combines
cooperative target declarations with independently observed KFD device, queue,
event, and control facts without upgrading declarations into execution claims.
The current milestone validates real MI300X queue observation and
suspend/resume. Hardware wave/lane state, PC, registers, target memory,
stepping, and breakpoints remain explicitly unavailable. See the
[`fe2o3-debug` contract](crates/fe2o3-debug-cli/README.md#exact-bound-live-kfd-protocol-v3).

`fe2o3-debug live-rocgdb` adds a bounded, agent-facing GDB/MI3 JSONL route for
authorized launch or attach, structured stopped-thread admission, relative
PC/memory inspection and audited execution control. Native process, path,
descriptor and address authorities are never returned. Generic
dispatch/workgroup hierarchy remains explicitly unsupported, and the current
validation covers deterministic end-to-end MI plus installed ROCgdb capability
discovery rather than a live pure-KFD GPU-wave stop. See the
[`live ROCgdb contract`](crates/fe2o3-debug-cli/README.md#structured-live-rocgdb-protocol-v3).

Feature-free production has one compiler transaction and no pipeline selector.
`cargo fe2o3 build` and `cargo fe2o3 run` first build the selected crate graph
for the fixed AMDGPU target through fe2o3, commit the exact device artifact
generation, and then build or run the same selection for the pinned host target
with ordinary rustc. Callers do not pass `--target`. Remaining V1/V2/V3 names
identify frozen records and protocols, not implementations. The target
architecture's default/production surface has exactly one executable compiler,
publication, load, and launch route. Temporary qualification implementations
are deletion backlog: their
useful evidence must move to differential fixtures or offline tools, after
which their features and code are removed rather than retained as alternate
in-tree pipelines. Default `cargo-fe2o3` tests now compile the same production
route as the feature-free binary. The backend is feature-invariant in normal
production builds: the remaining backend qualification feature can compile
only inert unit-test fixtures, and neither `cfg(test)` nor an environment
variable can select an alternate rustc route. The explicit
`fe2o3-core/qualification-unsafe-launch` feature is currently enabled only by
the checked-in standalone external-HSACO numerical examples; it grants no
Worker V3 publication, protected execution, or artifact-currentness authority.

The feature-independent extraction driver now enters the same production
typestate transaction under a real `amdgcn-amd-amdhsa` rustc session. Its
analysis-only custody has passed collection, dynamic ranked-bounds, reference
binding, and loop-carried BF16/F32 MFMA GEMM lanes on gfx942 through semantic
MIR, ranked PLIRON, verified Kernel IR, composed memory checks, and
deterministic LLVM extraction. It cannot publish, finalize HSACO, load, or
launch, so this checkpoint changes no parity status.

The production workgroup slice now goes farther through that same transaction.
The ordinary attributed `lds_publish_read_reduce_i32_v1` Rust kernel retains
its exact `64x1x1` launch and 256-byte static-LDS contract through semantic MIR,
ranked PLIRON, verified Kernel IR, composed memory admission, upstream-LLVM
AMDGPU lowering, a compiler-bound handoff, the measured target-machine and
in-process LLD worker, and COV6 descriptor inspection. Deterministic replay
produces identical HSACO bytes with an exact 256-byte static-LDS contract. This
route selects no workload profile and uses neither COMGR nor a shell linker. It
does not grant load or launch authority. The same SHA-pinned HSACO now also
completes one measured-only pure-Rust KFD diagnostic on the MI300X through the
runtime's invocation-specific authority gate: all 64 inputs reduce to `2080`,
allocation canaries remain intact, and queue/completion teardown succeeds. The
diagnostic supplies an explicitly unsafe, manually asserted authority rather
than a production Worker V3 decision. It establishes native KFD/AQL mechanics
and the runtime identity join only; production executable evidence must still
enter through the Worker V3 application and verifier.

The generated host boundary now joins an authenticated Worker V3 executable,
macro-generated address-free arguments, exact current artifact custody,
geometry, and one checked KFD device into a private move-only invocation. On
MI300X, the exact scalar-GEMM lane executes this joined path and passes its CPU
oracle, completion writeback, and canaries. That test intentionally uses a
synthetic verifier and externally injected HSACO, so it validates composition
and hardware behavior without claiming production proof authority.

The production semantic-lineage handoff now preserves the exact aggregate
MIR-to-live-PLIRON Verus execution instead of reducing it to a digest. The
compiler emits a V4 proof association inside the frozen V3 capsule envelope;
that association binds all five compiler-stage identities and the canonical
signed receipt, embedded verifying key, binding, toolchain, and execution
identities. Worker V3 independently decodes every stage, reimports the signed
receipt, cross-checks its PLIRON identity against the exact middle-end V5
evidence, and retains the move-only result beside signed compiler-currentness
evidence through authentication, load, dispatch, and unload. The embedded key
establishes receipt consistency, not trusted producer origin; protected
compiler currentness supplies the separate origin join. This milestone does
not establish LLVM/ISA refinement, final-machine effects, dynamic launch
preconditions, or runtime authority.

The exact scalar-GEMM production auditor now also joins an independently
authenticated upstream-LLVM machine analysis to its retained Verus input. The
generated source binds the Worker V3 challenge, lineage and host contract; the
compiler/KIR identities; the exact final HSACO and descriptors; all 19 reviewed
machine-effect sites; and the lossless machine request, evidence and receipt.
Pinned Verus accepts the exact source with `94 verified, 0 errors` and rejects
one-at-a-time HSACO, descriptor, effect-offset, effect-kind, and effect-width
substitutions. This closes only the bounded `ProofExecutableBinding`
obligation. Six compiler provenance, semantic refinement, floating-point,
machine refinement, ABI, and end-to-end effect obligations remain open, so the
auditor still cannot authorize publication, loading, or launch.

The production compiler now retains move-only protected-rustc custody through
semantic-lineage construction and repeats complete live process admission
immediately before strict V3 publication. After that publication, the sole
backend derives a fixed 690-byte
`InertCompilerExecutionSubjectV1` from the durable build attempt, exact V3
transaction, complete compiler closure, rustc invocation, inventory and
preflight, semantic capsule, final module commitment, and exact outer handoff.
Strict consumption independently reconstructs identical bytes. A shared V1
protocol now fixes the caller-pinned issuer policy, subject-bound challenge,
complete request, Ed25519 receipt, and rollback transition as exact canonical
records. This removes ambiguity at the future service boundary, but all of
these values remain authority-free. The protected authority service now has a
shared Linux admission primitive and an issuer-admission owner that requires an
irreversibly hardened process, a loader-independent static executable matching
the caller policy, and the canonical move-only service-owned signing-key
capability. That capability enforces an anonymous exact mode-0400, read-only,
sealed key image, zeroizes admitted seed buffers, and exposes neither descriptor
nor signing authority. The issuer retains and remeasures the exact executable
and key at every continuity boundary. A signed fixed-width singleton journal now durably commits fresh
subject-bound challenges and exact receipts, recovers only immediate legal
successors, re-emits prepared/issued outputs after crashes, and advances its
rollback chain idempotently. Its signing APIs require a live move-only compiler
occurrence. The backend and protected service share one authority-free validator
for exact V3 argv, canonical
cwd, complete environment, target, backend and artifact-directory capability
paths, compiler-closure pins, and measured rustc/backend bytes. The service
independently binds the admitted pid/start identity, procfs process directory,
sealed invocation fd 199, backend fd 198, and artifact-directory fd 197; it
retains and repeats every observation without exposing a raw descriptor or
granting authority. The service now derives the exact producer and managed
attempt from that sealed invocation through the same canonical producer
constructor used by Cargo and the backend, reacquires the current production-slot V3
publication, reconstructs its canonical subject under lock, and retains both
custody values through every issuer use. The issuer accepts no caller-selected
occurrence, and its private guard keeps publication currentness locked through
request comparison, signing, and durable commit. A canonical fixed-width packet
codec and allocation-free bounded `SOCK_SEQPACKET` loop now expose only Inspect,
Prepare, Issue, Publish, Cancel, exact-subject Recover, and exact-carriage
VerifyCurrent over the already
admitted peer. Recover strictly reacquires the current Worker record and returns
its complete receipt carriage, or reports nonterminal absence only when no
canonical Worker record exists; a different or damaged current record fails
closed. VerifyCurrent independently reacquires that record under the protected
policy, byte-compares the complete expected carriage, and returns canonical
policy/Worker-ledger, external-anchor commit, and fresh signed current-head
evidence in one terminal exchange. Its 2,250-byte request includes a fresh
client challenge; its 1,784-byte response contains a 1,624-byte Ed25519 V3
attestation over that challenge and the complete 1,440-byte V3 record. The
record embeds the exact 528-byte signed advance receipt retained by the
reacquired Worker V2 record and a fresh 528-byte signed recovery receipt. The
external recovery nonce is derived from the client challenge, exact carriage
identity, and retained commit-receipt identity. The client requires both keys to
equal the pinned policy, every record coordinate to equal its original expected
carriage, the retained receipt to be an exact proposed-position advance, and
the fresh receipt to report the same proposed head under the exact recovery
challenge. This cryptographically authenticates commit and current-head claims;
independent administration, monotonic persistence, and protected anchor-key
custody remain deployment requirements. That same
policy now pins a distinct external-anchor Ed25519 key; weak or equal issuer and
anchor keys fail closed. A staged protected boundary now also admits one
supervisor-provisioned nonblocking unnamed seqpacket and exact live service
pidfd under a separately pinned non-root UID/GID. Its fixed-deadline transport
sends only the canonical anchor challenge and accepts only one exact,
ancillary-free signed observation verified under the policy key. The production
supervisor now retains that exact endpoint and pidfd, binds their admitted
service identity to the launch manifest, and transfers both through the static
launcher for independent issuer admission at FDs 10 and 11. Receipt publication
now durably prepares the exact anchor challenge, exchanges it over that retained
endpoint, persists an exact signed proposed-position observation, and only then
commits and reacquires the Worker record and returns its publication ACK. Restart
re-emits the same prepared challenge, finishes an already anchor-committed local
publication without a second exchange, and durably aborts on an exact signed
prior-position observation. These issuer-side checks alone do not prove that
the external service is independently operated. The external-anchor service
now supplies exclusive single-writer state, exact crash-durable
write/fsync/rename/fsync transitions, idempotent recovery, and an ancillary-free
connected `SOCK_SEQPACKET` loop. Its sealed deployment manifest pins the
dedicated UID/GID, public key, exact supervisor deployment, and bounded SHA-256
measurement of the daemon executable. A separate role-tagged signing-key
capability binds the private seed to that complete deployment identity, requires
anonymous service-owned read-only custody, and can release an in-memory key only
after revalidation. The descriptor-only daemon admits the exact locked profile,
then requires `/proc/self/exe` to be the measured service-owned anonymous
mode-`0555` executable under complete content, execution, and seal locking before
reading that key. It opens only existing state, retains root and peer at private
FDs 256 and 257, closes every other descriptor including stdio, and enters the
bounded packet loop. Its pinned musl release now shares the issuer's syscall-only
pre-runtime entrypoint and passes the exact static-ELF, no-loader, no-undefined-
symbol, and silent fail-closed gates. The measured unprivileged provisioning
helper now reissues root key custody, atomically opens or initializes state under
one retained lock, creates the service socketpair, transfers one exact endpoint,
and crosses into the daemon through `execveat` with an empty environment. The
privileged coordinator now implements measured preparation, atomic pidfd launch,
profile gating, endpoint admission, transfer cloning, and exact kill/reap custody.
Its supervisor-construction wiring, authoritative root-only distinct-UID
qualification, and refinement evidence join remain open, so the result remains
authority-free.
The issuer's direct transition
methods are private. One shared bounded
client now recovers first and resumes Ready, Prepared, or Issued under one
absolute deadline, including exact issued-request reconstruction. Its child
channel now creates the peer after fork inside the selected child, installs fd
195, transfers only the service endpoint to the direct parent, and binds it to
exact child credentials, child-reported direct-parent identity, and a live
pidfd. A
canonical outer handoff binds that exact direct parent to the sealed launch
manifest, and the client can transfer its service peer and pidfd over one
authenticated `SOCK_SEQPACKET` connection to the distinct-UID supervisor. The
supervisor repeats direct-parent `SO_PEERCRED`, policy, rustc peer credentials,
pidfd target/liveness, descriptor identity, and alias checks. It also retains a
separately admitted external-anchor endpoint and exact service pidfd. It now
consumes one accepted handoff into an exact twelve-source prepared launch for
destination FDs `0..=11`, with distinct isolated stdio/readiness pipes, cloned
authority and anchor descriptors, and a canonical 704-byte manifest bound to
the supervisor PID and procfs start time. That manifest is an anonymous read-only mode-`0400` memfd
with complete content seals; every object, byte, access mode, capability, live
client, and non-aliasing relation is revalidated without exposing descriptor
custody. The two process-pidfd entries use a dedicated manifest object class
because Linux pidfds may share one anonymous-inode `fstat` key; the launcher
validates the class while the receiving services bind each pidfd to its exact
process target. A gated `clone3(CLONE_PIDFD | CLONE_CLEAR_SIGHAND)` child now
self-checks the inherited locked service profile and parent-death containment.
Before release, the parent independently rechecks procfs profile fields, every
unchanged namespace, live rustc, and all authority objects. The bootstrap
isolates stdio, installs the manifest and issuer at FDs 198 and 199 and the
twelve sources at FDs 200 through 211, and executes only the authenticated static launcher. The launch
manifest binds the exact rustc PID/UID/GID to the pinned policy, and the
descriptor-only musl-static issuer enters through a syscall-only shim that
restores nondumpability before musl or Rust startup and then consumes fixed FDs
3 through 11. After complete
admission and durable recovery it emits one canonical PID/manifest/policy-bound
readiness record through an atomic nonblocking pipe. Its build gate rejects
dynamic-loader edges, undefined symbols, a displaced secure entry point, and
non-fail-closed startup. A dedicated supervisor crate now authenticates the
provisioned launcher against a service release measurement and the issuer
against that sealed policy before copying both into distinct read-only
mode-0555 memfds with complete content and executable seals. The resulting
move-only program is now consumed into one prepared supervisor together with
the canonical signing-key capability, dedicated non-root UID/GID profile, and
an exact service-owned mode-0700 root. Prepared, launched, ready, and serving
states are move-only and expose no descriptor, key, or signing operation.
Readiness must be one exact PID/manifest/policy record followed by EOF while
the same pidfd child is live. The supervisor then publishes those exact bytes
once over the authenticated Cargo control connection, drops that connection,
and retains pidfd serving custody; cancellation and drop use pidfd signaling
plus exactly-once bounded reaping. The pinned policy has an
immutable sealed memfd capability reserved at rustc fd 202. The protected
release contract now admits and binds the sole fixed root-owned client profile.
The Cargo wrapper transports that profile through its authenticated broker.
For rustc it creates the service endpoint at fd 195 after fork, installs the
exact policy at fd 202, transfers the live endpoint and pidfd only to the fixed
distinct-UID supervisor socket, and gates fresh Worker V3 publication on
canonical readiness while retaining exact invocation custody. The application
runner now creates a separate child-bound fd 195 and reaches the same fixed
supervisor before waiting for the application ACK; fd 202 remains parent-only.
It validates and exposes only the connected `SOCK_SEQPACKET` endpoint through
the seccomp boundary and contains the complete application process group on a
handoff failure. `fe2o3-host` owns a one-use auditor that consumes fd 195,
checks the fresh signed response, and returns move-only authority-free evidence.
Backend receipt acquisition, lossless V2 carriage, exact protected
Worker-record verification, receipt-bearing V3 VerifyCurrent response, and the
receipt-complete Worker V3 promotion boundary are implemented.
Deployed distinct-UID service provisioning, the concrete protected verifier,
and external monotonic anchor remain absent. A fixed
receipt sidecar and publication ACK now
carry the exact journal, occurrence, receipt, Worker record, sequence, and
advanced rollback anchor without granting authority from wire bytes. The
journal V2 retains that complete ACK across every later state, accepts no raw
receipt-digest acknowledgment, signs the occurrence identity, and rejects a
subject-equivalent replacement after restart. A separate 2,218-byte Worker V2
record retains the complete request, sidecar, and exact 528-byte signed
proposed-position external-anchor receipt. It verifies both receipts against
their pinned keys and exact transaction, commits only an immediate successor,
reacquires the exact canonical bytes, and is the sole producer of the move-only
ACK capability. The local anchor journal and Worker record must name the same
receipt; legacy V1 files and record-without-journal state fail closed. A
successor preparation cannot displace the current record's retained receipt.
Issuer admission accepts only the three legal cross-journal crash positions.
Consequently
`CompilerExecutionProvenance` remains open. See
[protected issuer admission V1](docs/compiler-execution-issuer-admission-v1.md)
and [durable issuer state V2](docs/compiler-execution-issuer-durable-v2.md).
The transport records are specified in
[receipt publication V1](docs/compiler-execution-receipt-publication-v1.md).
The durable consumer is specified in
[Worker receipt ledger V1](docs/compiler-execution-worker-ledger-v1.md).
The bounded transport is specified in
[compiler execution service V1](docs/compiler-execution-service-v1.md).

## CUDA-Oxide status

Against the pinned cuda-oxide baseline, the evidence ledger currently records
`0 Complete / 82 Partial / 0 Missing / 12 N/A` normative rows and
`0 Complete / 15 Partial / 0 Missing` supplemental rows. Zero Missing means
every in-scope row now has at least one bounded, tested implementation slice;
it does not mean that any row satisfies its full acceptance contract or that
fe2o3 has reached cuda-oxide parity.

The newest `gfx942` slices cover trusted memory operations, bounded closures
and control flow, cross-crate device roots, typed groups and collectives,
managed barriers and standard atomics, static proof-carrying tiles, launch
policies, FP8/MX and MFMA/LDS contracts, composite O0 debug metadata, and a
closed diagnostic/assembly surface. Device-library and tile interop are narrow:
the former demonstrates one directly linked OCML operation, and the latter one
BF16 XOR4 tile/stream contract. The bounded
[gfx942 wave/LDS V2 slice](docs/gfx942-wave-lds-v2.md) additionally carries one
masked `u32` wave64 reduction and one 256-thread static-LDS reduction through
exact compiler/LLVM checks, Verus proofs, and direct LLVM/LLD MI300X execution.
V2 admits only the full canonical `gfx942:xnack-` target, persists that binding
in Kernel IR, and checks it against the Worker V2 envelope. The older V2
compiler-created fixture remains separate, but the production `i32` WG64 LDS
reduction now joins genuine Rust source to reproducible inspected HSACO through
the one compiler transaction. Exact-artifact KFD execution through an unsafe
diagnostic authority passes, while production Worker V3-authorized KFD
execution remains open. The
dashboard records the exact commits, tests, target lanes, evidence strengths,
and limitations for each Partial row.

The 2026-08-19 [#134](https://github.com/harsh-nod/fe2o3/issues/134)
checkpoint now includes additional fail-closed ownership boundaries. The rustc
frontend retains one non-cloneable, same-session typed MIR/CFG graph with exact
item, instance, source, MIR, ABI, import, and Pliron-graph identities; only the
return-only subset is admitted, and it grants no compiler authority. General
GEMM collection also enforces one aggregate 512-call and 32-trusted-terminal
budget. The closed
gfx942 General GEMM structural route retains its live Pliron LLVM graph,
compiler machine, Worker V2 execution owner, finalized bytes, and post-link
inspection for both schedules. Its late graph, worker, and finalizer axes are
freshly derived from those retained owners. Build-policy admission is not
worker-measurement authentication, and the axes grant no authority. Production
extraction now accepts one safe dynamic General GEMM and emits its deterministic
gfx942 LLVM through the general semantic pipeline. Protected General GEMM
publication remains disabled until #106 is consumed by the owner-carrying #174
receipt and the rustc-owned final authority join consumes that receipt together
with the #173 late-machine binding.
`pliron-llvm` has default features disabled, and no COMGR,
`llvm-sys`, or subprocess compiler/linker has artifact authority on this route.

The closure landed in `fd6520d88` (exact Worker machine effects), `70f9c5ad7`
(structural ELF, descriptor, and decoded-machine inspection), `e016833d3`
(measured-HSACO gate), `c9e8ca702` (move-only Worker execution evidence),
`62efd243e` (repository policy, finalizer join, and sealed one-shot HSA
consumer), and `228c88ed9` (descriptor-versus-runtime kernarg alignment). The
code target is exactly `gfx942:xnack-`; the qualifying MI300X reported
`gfx942:sramecc+:xnack-`. The repository pins Worker executable SHA-256
`12c06e0da5d812c1db6f33450f99a8d70087c585eec552f7f8616077704361fd`,
HSACO SHA-256
`011671a80384051232fb684c90afadd9b5e9d81c13d216238f15af55dd3880b1`,
and ROCr HSA 1.18 image SHA-256
`7010eba894569c044749b71b63ff782080c4a91e19ff24d6dc93e857045ab37e`.
The COV6 descriptor requires 280 kernarg bytes aligned to 8; the observed HSA
kernel requires the same 280 bytes in runtime storage aligned to 16.

The successful run consumed the finalized bytes through the sole typed,
move-only runtime transition, produced bit-exact `3.75f32`, preserved the input
and all allocation canaries, and reached terminal unload. Its
`FE2O3_REPOSITORY_SCALAR_ADD_V1_MI300X_OK` marker is a canonically serialized,
self-consistent record of the bounded policy, artifact, runtime image, device,
dispatch, result, canary, and unload observations. It is not a signature or CI
attestation; process-local runtime, agent, executable, dispatch, and kernarg
identities may differ between runs. Likewise, the compile-time checkout policy
is repository/build provenance, not an externally signed or separately
authenticated approval.
This checkpoint changes no CUDA-Oxide parity row or count and proves neither
general memory safety nor race freedom; it also does not establish general
GEMM, attention, or MoE support.

The Wave64 and workgroup-synchronization slices now start from ordinary
`#[kernel(typed)]` Rust sources rather than explanatory pseudocode. They include
deterministic CPU oracles, hostile source tests, and bounded Verus models for a
masked Wave64 reduction/scan and an LDS/barrier/scoped-atomic profile. The typed
device ABI preserves mutable global address-space pointers and exposes a linear,
compiler-only exact-LDS capability. The Wave64 compiler profile now authenticates
the exact attributed source, FnAbi, trusted definitions, complete reachable MIR,
mask semantics, ordered collectives, and output ownership before selecting a
closed semantic Kernel IR sidecar. The two workgroup profiles likewise
authenticate their exact source, ABI, trusted provider terminals, and complete
reachable MIR closures before selecting closed semantic profiles. Separate
configured finalizer tests use the pinned upstream LLVM target-machine and
in-process LLD worker, and separately scoped protected `gfx942` hardware lanes
exist. Those lanes remain ignored behind exact measured prerequisites and do
not establish source-to-machine refinement, production artifact or launch
authority, or generalized memory/race safety. Their evidence is tracked in
[#117](https://github.com/harsh-nod/fe2o3/issues/117) and
[#118](https://github.com/harsh-nod/fe2o3/issues/118).

The fixed 64-element row-softmax slice uses one shared numerical oracle and an
inert deterministic certificate that binds its exact Rust source, reviewed MIR
profile, Kernel IR and LLVM identities, numerical policy, and Verus/Z3 closure.
The exact compiler/finalizer lane still checks the direct upstream LLVM/LLD
worker exchange, OCML closure, artifact, descriptor, ABI, geometry, and resource
identities. Its workload-specific host token, typed HSA lifecycle, hardware
launcher, and Cargo `legacy-hsa-runtime` switch are deleted. Row-softmax can
return to hardware only through the generic Worker V3 application path.

The row profile also has a host-specific compiler/code-object release gate. By
protocol, implementation Commit A contains the gate but deliberately contains
no release manifest. Only a subsequent manifest-only Commit B directly above A
may select an independently reviewed SHA-256, pinned upstream LLVM 22.1.8
source and package closure, in-process LLD, the exact OCML/device-library
closure, Cargo/rustc and their offline source/sysroot closures, runtime DSOs,
Worker and layout-probe ELFs, and the retained HSACO. Both C++ and Rust require
the measured metadata exactly, including four explicit and nineteen hidden
arguments, presence or absence of
optional fields, register/spill values, resources, symbols, and target. Release
evidence can be claimed only when a compliant B and two runs from distinct fresh
build and Cargo directories reproduce the same caller-supplied manifest digest
and byte-identical outputs. That combination establishes only operator-selected
reviewed integrity, not origin authentication or GPU evidence.

W0/P0 is now accepted as a bounded host-link prerequisite. Its dedicated,
genuinely static `fe2o3-host-lld` is built from pinned upstream LLVM/LLD
archives. `HostLinkClosureV1` supplies descriptor-sealed inputs, launches the
exact approved executable with `execveat`, and returns a receiver-owned sealed
output. Landlock enforces the filesystem boundary, while seccomp denies network
and descriptor-transfer operations. Two fresh MI300X builds produced the same
85,597,472-byte tool with SHA-256
`7c1a7429e93896393eb743ed54ead78ec6d492e3ed887183e67737b3872d7bf9`.
The registered secure-protocol CTest and a real `HostLinkClosureV1` link slice
also passed in separate executions.

This build evidence is measured/no-authority. W0 provides no protected
publication, broker or durable artifact handoff, runtime, load, launch, or GPU
evidence. It proves neither memory safety nor race freedom and provides no
source-to-machine or Verus-to-machine refinement. W1/P0 Broker V4 is the next
production blocker. The parity counts remain `0/82/0/12` normative,
`0/15/0` supplemental, and `0/97/0/12` combined. The direct GPU link path
remains separate and pinned to upstream LLVM 22 with in-process `lld::lldMain`,
without COMGR or a shell GPU linker. The replacement Worker V3 row-softmax
slice remains tracked under [#120](https://github.com/harsh-nod/fe2o3/issues/120).
The subsequent fixed FlashAttention and top-2 MoE vertical slices are tracked by
[#122](https://github.com/harsh-nod/fe2o3/issues/122) through
[#125](https://github.com/harsh-nod/fe2o3/issues/125).

The exact [MoE expert-compute source slice](examples/moe_expert_v1/README.md)
extends the fixed T8/E4/K2/C4 router with two ordinary attributed Rust kernels:
one host-selected `16x16x16` BF16/F32 expert GEMM and one deterministic top-2
weighted combine. Its executable CPU schedule and independent oracle still
provide source/CPU evidence, while the original pinned Verus model verifies 15
logical expert-memory obligations and rejects six named mutations.

The [bounded MoE V1 checkpoint](docs/bounded-moe-v1.md) retains a standalone
`E4/C4/routes16/width16/tile256` compact-plan example that verifies 19 Verus
obligations and rejects seven mutations. It is not a MIR-to-KIR refinement proof
or an authority-bearing proof receipt.

The former MoE V1/V2 host bridges, generated adapters, exact top-2 lifecycle,
and workload-specific HSA launcher were non-production qualification
alternatives. They have been removed so MoE execution cannot bypass the sole
Worker V3 application, descriptor, argument, HSA, and unload lifecycle. The
ordinary Rust kernels, compact-plan Verus example evidence, and independent
source/oracle tests remain. MoE hardware execution through Worker V3 is still pending and no parity promotion is claimed.

The current [general tiled GEMM](examples/tiled_gemm_general_v1/README.md) is an ordinary safe Rust `#[kernel]` example on the shared MIR -> ranked PLIRON -> KIR pipeline. Its dynamic dimensions, strides, tails, MFMA layout, and epilogue exercise workload-neutral compiler checks; the retired fixed Slice 1 standalone composition and its separate observation records have been removed.

The source/IR groundwork landed under
[#85](https://github.com/harsh-nod/fe2o3/issues/85),
[#86](https://github.com/harsh-nod/fe2o3/issues/86), and
[#93](https://github.com/harsh-nod/fe2o3/issues/93). The shared integration
epic [#94](https://github.com/harsh-nod/fe2o3/issues/94) and its exact-profile,
finalizer, host-adapter, and lifecycle children
[#96](https://github.com/harsh-nod/fe2o3/issues/96),
[#97](https://github.com/harsh-nod/fe2o3/issues/97),
[#99](https://github.com/harsh-nod/fe2o3/issues/99), and
[#100](https://github.com/harsh-nod/fe2o3/issues/100) are closed. Production
certificate consumption [#91](https://github.com/harsh-nod/fe2o3/issues/91),
refinement [#106](https://github.com/harsh-nod/fe2o3/issues/106) and
[#107](https://github.com/harsh-nod/fe2o3/issues/107), and the other Slice 2-4
issues remain open.

The intended end state is:

```text
Rust host + #[kernel] device code
        |
        v
rustc frontend and MIR
        |
        +--> native host binary
        |
        +--> fe2o3 device backend -> AMDGPU LLVM IR -> HSACO
                                                |
                                                v
                                  typed HSA / HIP load/launch
```

## Architecture

The 2026-08-18 ownership refactor splits representation, compiler composition,
target lowering, and host execution into explicit ownership boundaries:

- Canonical contracts and models: `fe2o3-mir-model` owns the
  Pliron-independent MIR schema and transformations; `fe2o3-compiler-api`
  owns target-neutral compiler request/result contracts;
  `fe2o3-proof-contracts` owns solver-neutral property records;
  `fe2o3-host-api` owns inert compile/admit/load/dispatch/wait records; and
  `fe2o3-service-model` owns executable-free persistent-service semantics.
  These records validate representation and consistency. They do not prove a
  claim, compile a kernel, execute a service, or grant artifact/runtime
  authority.
- Compiler composition: `cargo-fe2o3` drives the sole production rustc backend through one managed Worker V3 transaction. The production backend has no selector or fallback slot; inspection remains an observation of that transaction, not another compiler implementation. `FE2O3_QUALIFICATION_ORACLE_V1` is only a rejected legacy sentinel; no workload oracle is compiled.
- General kernel checks: `fe2o3-kernel-analysis` owns the fixed pre-lowering
  Kernel IR sequence for structure, control flow, bounds obligations, race
  freedom, barrier convergence, and workgroup-memory initialization/reuse.
  The production MIR-to-Kernel-IR boundary runs the sequence for every kernel
  and rejects concrete failures before transformation. `Incomplete` and
  `Clean` reports remain non-authoritative; see the
  [V1 pipeline contract](docs/general-kernel-check-pipeline-v1.md).
- Pliron framework: `fe2o3-pliron` is a bounded D0 context, registration,
  context-identity, pass-planning, and owner-held textual bridge over Pliron
  v0.17.0 at reviewed fork commit
  `5bdf861bf03e7f20242b25717fb653336d02e487`, a strict descendant of the
  upstream v0.17.0 commit `2610651306ea3ba670f68d5d8b1e1159bcd521ed`.
  The bridge recursively verifies imported operations and enforces bounded
  owner/session accounting, but arbitrary registered `Parsable` implementations
  remain trusted parser code and the bridge grants no compiler authority. Seven
  target-neutral representation shells exist for `kernel.*`, `schedule.*`, `tile.*`,
  `gpu.*`, `proof.*`, `dispatch.*`, and `autotune.*`. `dialect-mir` remains a
  compatibility facade over `fe2o3-mir-model` and additionally exposes a
  bounded `mir.*` Pliron shell only with its non-default `pliron` feature.
  These crates construct and verify in-memory representations; they do not
  form a production MIR-to-HSACO pipeline.
- Retained MIR projection support: `fe2o3-lower-mir-kernel` remains a narrow,
  terminally fail-closed, context-bound `mir.*`-to-`kernel.*` conformance
  service. The rustc integration owns the production semantic MIR to ranked
  Pliron to canonical KIR transaction; stale handles fail with typed errors.
  The former detached KIR-envelope, kernel-to-GPU, and parallel
  AMDGPU-to-LLVM shells were removed because they were not production routes.
  No workload selector or alternate lowering fallback remains.
- Target model and facades: `fe2o3-amd-target` owns canonical AMD target
  contracts. The existing strict AMDGPU lowering implementation moved to
  `fe2o3-amdgcn-model`; `dialect-amdgcn` now preserves the historical crate API
  by re-exporting that model and is not yet an `amdgcn.*` Pliron dialect.
- Host and service boundaries: `fe2o3-core`, `fe2o3-host`,
  `fe2o3-hsa-runtime`, and `fe2o3-hip-sys` own the existing executable runtime.
  `fe2o3-service-host` is a `no_std` typestate adapter over
  `fe2o3-service-model` and `fe2o3-host-api`; it retains storage borrows and
  checks lifecycle descriptions. On Linux x86_64 it also owns a generic,
  addressless composition of checked KFD allocations and one long-lived fixed
  dispatch queue through linear publish, completion, recycle, detach, rebind,
  destruction, and release transitions.
- Pure-Rust runtime foundation: `fe2o3-kfd-uapi`, `fe2o3-kfd`, and
  `fe2o3-runtime-model` provide reviewed KFD 1.18 encodings, fail-closed device
  observation, and Verus-backed lifecycle modeling. They do not yet replace
  the existing HIP/HSA execution path or establish persistent GPU execution.
- Artifact, build, proof, and evidence boundaries remain in
  `fe2o3-artifacts`, `fe2o3-kernel-descriptor`, `fe2o3-hsaco`,
  `fe2o3-hsaco-finalize`, `fe2o3-artifact-transaction`,
  `fe2o3-compiler-execution-protocol`, `fe2o3-runtime-protocol`,
  `fe2o3-compiler-execution-client`, `fe2o3-compiler-execution-issuer`,
  `fe2o3-build-authority`,
  `fe2o3-host-link-closure`, `fe2o3-broker-authority-service`,
  `fe2o3-external-anchor-protocol`, `fe2o3-process-identity`,
  `fe2o3-protected-publisher`, `fe2o3-verifier`, and
  `fe2o3-differential`. The retired Worker V2 bundle and standalone compiler routes are absent from the workspace; frozen V2 wire names remain only where Worker V3 still consumes their protocol representation.

The machine-checked layer policy forbids dependencies that invert these
ownership directions. The production-directed GPU finalizer continues to use
an isolated worker built against pinned upstream LLVM target-machine APIs and
in-process LLD library APIs. COMGR is not part of the architecture, and shell
`clang`/`ld.lld` use belongs only to the historical compatibility path.

[#134](https://github.com/harsh-nod/fe2o3/issues/134) and
[#135](https://github.com/harsh-nod/fe2o3/issues/135) are both still open. The
landed crates make parallel implementation possible and enforce representation
boundaries; they do not mean that the Pliron production compiler or persistent
GPU service exists. No parity row or count is promoted by this refactor.

Safe buffer element types and their limits are documented in the
[device memory safety contract](docs/device-memory-safety.md). `DeviceCopy`
establishes structural host-side byte validity only. Safe device interpretation
also requires manifest-derived type and ABI identity, provenance/address-space,
and capability evidence.
Safe ownership of resources used by asynchronous copies is documented in
[device operations](docs/device-operations.md).

## Current Status

### Working end to end

- `cargo-fe2o3 build` builds and loads the custom backend, delegates host
  codegen to `rustc_codegen_llvm`, discovers strict versioned registrations
  emitted by `#[kernel]`, collects device-reachable MIR, and emits HSACO
  sidecars. Registration identifies compiler semantics; it is not package or
  artifact authentication.
- For a public kernel, `#[kernel]` emits a public, doc-hidden marker with the
  deterministic symbol `__fe2o3_kernel_marker_<function>` and an unsafe
  `KernelMarkerV1` implementation tied to the exact Rust function type and V1
  registration. The marker does not authenticate an executable or establish
  its full packed ABI and semantics; generated binding remains an unsafe
  compiler/runtime boundary.
- `#[kernel(typed)]` emits a generic Worker V3 marker and typed argument plan.
  The generated surface retains slice lifetimes, mutability, aliasing, and
  canonical rustc-derived type/layout identities, but exposes no load or launch
  method and no embedded artifact bytes.
- The backend binds each kernel occurrence, target, descriptor, finalized
  payload, proof records, and generated argument contract into the canonical
  Worker V3 envelope. Production applications may dispatch only after the V3
  verifier authenticates that complete graph; examples without that verifier
  fail closed before runtime dispatch.
- Production compilation is the only unselected compiler route and never
  falls back to a workload-specific implementation. Historical emitters and
  exact workload paths remain only as migration evidence until equivalent
  production coverage permits their deletion.
- The selected production rustc wrapper canonicalizes every kernel compile to
  exactly one `-Coverflow-checks=on`; explicit disabled or conflicting settings
  fail before the in-process driver starts. This fixed compiler policy is not a
  crate-namespace axis, while the exact protected rustc invocation still
  retains the canonical flag. The scalar gfx942 vertical slice derives one
  identity-bound semantic induction certificate for its checked `u32` loop
  increment while preserving the LLVM overflow guard. The current proof-input
  validator strictly decodes canonical KIR V8, deterministically replays the
  exact induction report, and requires that certificate's precise MIR span to
  contain exactly one checked KIR addition. Correspondence and formal-memory
  evidence bind the same versioned KIR digest and byte length. This evidence
  remains inert and does not establish KIR-to-LLVM or LLVM-to-machine
  refinement or authorize removing the guard.
- `fe2o3-core` provides HIP-backed contexts, streams, device buffers, pinned
  host buffers, events, synchronous transfers, and event-backed borrowed and
  owned asynchronous transfers. Its default/production surface exports no raw
  module, function, parameter pack, launch configuration, or launch function.
- The current generated application route enters the authenticated Worker V3
  transaction, compiler-generated typed arguments, and reviewed HSA adapter.
  That HSA-backed implementation is migration debt, not the permanent runtime:
  production is converging on the invocation-specific pure-KFD gate in
  `fe2o3-runtime`. The former host `launch!` macro and selectable raw-HIP core
  production feature are deleted; raw HIP module/launch mechanics remain
  private unit-test implementation details on the default surface. The explicit
  `qualification-unsafe-launch` feature is currently enabled only by the
  checked-in standalone external-HSACO numerical examples and grants no Worker
  V3 publication, protected execution, or artifact-currentness authority.
- `DeviceCopy` and its derive macro restrict safe byte transfers to supported
  layouts and have compile-pass/compile-fail coverage.

Historical `gfx1151` and `gfx942` runs generated, inspected, and executed the
then-runnable qualification paths. The current hardware lane runs focused
runtime checks and fill/vecadd compiler qualification tests. Example binaries
do not provide production Worker V3 execution until their verifier integration
is complete, and the recorded runs grant no current production authority.

### Implemented foundations

- The structured MIR importer lowers the vecadd-shaped subset, including
  scalar control flow, helper calls, and slice memory operations, into the
  target-neutral `fe2o3-kernel-ir`. Its verifier checks types, SSA uses,
  control-flow edges, memory accesses, launch axes, capabilities, barriers, and
  atomics. The IR has a bounded canonical V1 wire format. The G1 lowering now
  owned by `fe2o3-amdgcn-model` and re-exported through the historical
  `dialect-amdgcn` facade lowers the verified 1D fill and vecadd subset to
  deterministic AMDGPU LLVM and is connected to the opt-in `kernel-ir-v1` fill
  and vecadd paths above; it is not yet general or the default. For its
  modeled effects, Kernel IR derives formal allocation identities, affine byte
  regions, bounds requirements, runtime-alias requirements, and
  inter-invocation race obligations. Unsupported index widths, arithmetic,
  calls, or memory effects make the analysis incomplete rather than silently
  granting authority. Even a complete result is conditional on an explicit
  launch extent; the extent and mappings from formal parameters to runtime
  allocations remain unauthenticated and grant no proof or launch authority.
- A bounded rustc-front record models canonical collected function signatures,
  source locations, and CFG edges, and the rustc backend can construct those
  records for monomorphized functions. Reducible-CFG analysis and
  block-argument-to-LLVM-phi lowering are implemented and tested, including
  loop-shaped Kernel IR. The production device pipeline still does not consume
  these records generally, and most Rust MIR operations remain absent.
- The G2 type foundation records validated semantic scalars, pointers,
  references, slices, tuples, arrays, structs, direct and niche enums, padding,
  and rustc ABI facts. A bounded rustc-private extractor obtains those facts
  for fully monomorphized types. Separate bounded foundations now model
  semantic constants/data relocations and manifest-driven scalar/slice packing,
  but neither is a general rustc-to-artifact integration. Dedicated fixtures
  make the current generic, const-generic, aggregate, integer-match, loop, and
  cross-crate collection/lowering frontiers explicit; generic registered kernel
  roots remain unsupported.
- Versioned artifact manifests, ABI layouts, launch contracts, bounded
  containers, payload digests, native-kernel selection, and proof records have
  canonical encoders, decoders, and adversarial tests. The bounded
  `Gfx942TwoKernelBundleV1` profile admits exactly two canonically ordered
  kernels backed by one digest-validated native payload and requires a separate
  proof binding for each kernel. Duplicate proof keys, shared-payload
  substitution, stale ABI/effect/launch identities, and cross-kernel proof
  swaps fail closed. These proof records remain descriptive evidence and grant
  neither load nor launch authority.
- G3 adds a canonical multi-kernel bundle index, validated compiler-generated
  argument-packing plans, and explicit asynchronous operation lifecycle
  records. These are bounded data and typestate foundations; no general
  manifest-to-host-code generator or composable cancellation API consumes all
  of them yet.
- Canonical AMD target IDs, HIP-observed device properties, HSACO metadata and
  descriptor inspection, kernel-descriptor binding, and bounded post-link
  finalization are implemented as separate validation layers.
- The G4 model includes capability tables for supported AMD targets, branded
  3D invocation and wave-lane witnesses, canonical Kernel IR for static and
  dynamic LDS, scoped atomics, fences, and convergence-bearing workgroup
  barriers. The experimental AMD lowering emits LDS, scoped integer atomics,
  fences, workgroup barriers, and explicit wave32/wave64 lane, ballot, vote,
  and bounded shuffle operations. The exact gfx942 wave/LDS V2 path adds an
  authenticated Rust-facing wave64 active-mask reduction and non-forgeable
  1,024-byte static-LDS reduction capability with fail-closed canonical
  `gfx942:xnack-` Kernel IR and Worker target binding. Its independently constructed
  Kernel IR has passed numerical MI300X execution, but the genuine Rust fixture
  reaches only verified Kernel IR. Dynamic-LDS launch-byte plumbing, broad
  atomics and collectives, general source-to-HSACO finalization, and compiler
  refinement remain fail-closed gaps.
- `fe2o3-host` exposes one Worker V3 migration graph. It consumes a recovered
  pinned descriptor, authenticates compiler and verification evidence, grants
  one exact HSA load authorization, resolves the selected kernel, validates
  generated arguments and geometry against the admitted descriptor, packs the
  complete COV6 kernarg, admits aliases, and dispatches through the reviewed HSA
  adapter. The old HIP module/function loader, raw `KernelParams`, `launch!`,
  cooperative-launch bridge, embedded-artifact contract, and profile-specific
  vecadd host route are deleted. `PreparedLaunch<K>` and argument admission are
  inert validation foundations; neither can load or dispatch an executable.
- `fe2o3-runtime` owns the permanent pure-KFD execution boundary. Its safe
  consuming transition matches one exact HSACO, selected kernel, complete
  address-free invocation contract, and checked KFD GPU unique ID against an
  unsafe Worker V3 authority implementation. The LDS diagnostic exercises this
  gate. `#[kernel(typed)]` now also emits a type-sealed, borrow-retaining KFD
  argument implementation: it encodes host scalars and slices into owned
  runtime buffers, emits zero pointer placeholders plus descriptor-derived KFD
  fixups, and applies mutable results only after checked completion.
  `AuthenticatedWorkerV3ExecutableV1::prepare_generated_kfd_invocation` now
  consumes authenticated evidence, those generated arguments, current artifact
  custody, geometry, and one checked device into a private move-only authority;
  no raw request or authority can be extracted. The scalar-GEMM hardware lane
  passes through this joined path only under the explicit
  `worker-v3-verifier-test-support` feature with a synthetic verifier. Default
  builds seal `WorkerV3VerifierV1` against downstream implementations and keep
  `WorkerV3VerificationDecisionV1` construction crate-private, so callers
  cannot manufacture verifier or dispatch authority. The public
  `prepare_inherited_worker_v3_kfd_application_v1` transition now consumes the
  inherited Cargo handoff, derives the selected kernel from its generated type,
  authenticates it, and returns only that joined invocation. The production
  verifier and an ordinary inherited hardware run without external HSACO
  injection remain open.
- `DeviceBuffer::view`, `view_mut`, and `split_at_mut` produce checked,
  borrow-typed contiguous regions while retaining the parent allocation
  identity, context, base address, full extent, and selected region. Splitting
  creates two simultaneously live exclusive views with exact disjoint
  allocation-relative byte regions; nested splits preserve those identities.
  Range, size, address, zero-sized-type, overflow, and null-allocation failures
  are explicit. Rust borrowing enforces exclusivity, but there is not yet a
  mechanical Verus proof of the split implementation. These views are a host
  provenance foundation, not launch authority.
- The bounded general typed V3 foundation accepts by-value `i8`/`u8` through
  `i64`/`u64`, `f32`/`f64`, shared slices, and genuine trusted
  `DisjointSlice<T, Index1D>` arguments. The macro emits an expectation-only V3
  registration while rustc independently reconstructs semantic types, layouts,
  effects, and physical ABI. The ordinary host build has no custom-backend
  object or private semantic-witness symbol dependency. Exact single-source
  typed Rust kernels named `alpha` and `zeta` form the first General-V3 vertical
  slice. Their source roles and argument names are authenticated as part of the
  ABI identity rather than inferred positionally: alpha binds
  `scale/input/output`, and zeta binds `a/b/bias/output`. The corresponding
  descriptors have explicit/complete COV6 kernarg sizes `40/296` and `56/312`.
  Exact role, name, signature, mutability, or layout substitutions fail closed.

  The macro generates one signature-specific `Arguments` family for the
  workload-neutral Worker V3 host contract. Its storage capability parameters
  admit exactly the generated HSA migration capabilities or the generated
  host-memory KFD capabilities; compile-fail tests reject substitution between
  the routes. Both retain source borrows and reconstruct the same named ABI.
  The HSA specialization still provides migration dispatch. The KFD
  specialization produces descriptor-validated address-free arguments,
  buffers, fixups, effects, and completion custody. The joined invocation
  transition promotes these only after an authenticated Worker V3 executable,
  current publication, exact artifact, geometry, and checked device all match.
  A reviewed production verifier is still needed to replace the test verifier
  and make this sole runtime path reachable by ordinary applications.

  The rustc path recognizes only the exact alpha/zeta MIR shapes and lowers
  their trusted thread index, `Option`-guarded `DisjointSlice::get_mut`, slice
  loads, multiply/add operations, bounds control flow, and 256-thread launch
  contract through canonical Kernel IR. Unsupported targets, float policy,
  names, signatures, branches, or payload provenance fail closed. This is an
  exact lowering profile, not general Rust GPU lowering.
- The recovered Worker V2 host descriptor, launch-metadata bridge, synchronous
  HSA handoff, and Scalar GEMM Worker V2 hardware harness are deleted. The
  reviewed HSA adapter and physical observations remain shared mechanics for
  the production Worker V3 lifecycle. Qualification-only compiler publication
  records remain as isolated evidence inputs, but the default/production
  configuration has no Worker V2 or raw HIP host execution authority. The
  explicit unsafe qualification feature is currently enabled only by the
  checked-in standalone external-HSACO numerical examples and provides no
  Worker V3 publication, protected execution, or currentness authority.
- Compiler artifact publication is transactional and generation-owned. Typed
  generation results contain bounded immutable IR and HSACO snapshots captured
  through exact staged file descriptors and validated after publication while
  the transaction lock is still held. Later publication or pathname
  replacement cannot mix generations. Build-attempt and canonical rustc
  invocation descriptors are versioned and bounded. Worker V2 raw/final
  publication intent is derived by `fe2o3-hsaco-finalize` from sealed lineage;
  Cargo no longer duplicates its domain hashes. Completed publications produce
  a canonical inert `DurablePublishedHsacoClaimV1`, from which the transaction
  can reacquire a fresh non-clone currentness lease after revalidating the
  attempt, plan, receipt, generation, directory and file identities, digest,
  and publication lock. A bounded canonical compiler-transaction capsule binds
  source, dependencies, features, invocation, caller-measured compiler/backend
  identities, semantic and Kernel IR identities, Worker V2 request/response,
  target, raw and finalized HSACO, and artifact identity. The capsule is inert
  caller-measured evidence: it does not authenticate the compiler or establish
  source-to-machine-code refinement. None of these values grants load or launch
  authority.
  The protected required-envelope Cargo route now carries its compiler handoff,
  publication intent, receipt, completion, and restart state through the
  closure-bound V2 schema with no V1 fallback. Its cleanup escrow and exact
  successor lease make predecessor retirement restartable across every durable
  boundary, including a crash after a newer `Ready` marker is published. The
  ordinary compatibility route remains schema-separated on V1. This is durable
  provenance and crash recovery, not the still-missing production proof
  authenticator or host/HSA dispatch-authority bridge.
  Strict Worker V3 finalization now has a separate bounded restart route: a
  move-only compact transcript retains the slot and transaction axes, durable
  storage owns each unique outer/provider/finalized component, and fresh
  persistence and process recovery share one validator that reconstructs both
  worker exchanges, rederives the complete semantic binding, re-inspects raw
  HSACO, and requires byte-identical canonical re-finalization. The recovered
  owner crosses one audited, move-only publication-authority boundary; safe
  transaction callers cannot construct that authority from hashes. The
  transaction revalidates the complete durable intent under its publication
  lock, commits version-separated pending/final V3 receipts, returns a bounded
  canonical V3 claim and currentness lease, and reconstructs the same move-only
  production result after either backend-claimed or completed process restart.
  That result now transfers into the receipt-bearing, move-only
  `WorkerV3LoadEnvelopeV2`. V2 losslessly nests the complete V1 replay codec and
  the complete compiler receipt carriage, strictly reconstructs and compares
  the signed compiler subject, and uses the same schema-neutral durable custody
  and restart mechanism. The backend acquires the carriage from the protected
  issuer after durable V3 handoff publication and stores it in an exact
  subject-bound sidecar. Cargo observes the handoff and sidecar under one
  currentness lock, verifies the carriage against the sealed client profile,
  and carries it through fresh execution, finalized-HSACO recovery, V2
  persistence, and ready-state recovery. Cargo application transfer and
  `fe2o3-host` decode and recover only top-level V2; V1 remains only the nested
  replay codec and is not a production route. Exact terminal custody authorizes
  retirement of the duplicate current-generation replay intent, while
  registry-rooted scavenging removes only superseded custody. The envelope
  still grants no compiler, semantic, HSA readiness, load, or launch authority.
  Host admission reconstructs the exact compiler subject and binds it together
  with the complete carriage bytes into the Worker V3 lineage challenge. The
  verifier request lends both canonical records without projection; promotion
  compares every policy, occurrence, receipt, publication, Worker-ledger,
  sequence, and rollback coordinate and rejects missing protected-policy,
  Worker-ledger, or external rollback verification identities before HSA load.
  Production transfers only the canonical V3 envelope and artifact-directory
  descriptors to an identity-pinned sealed application. A
  fresh occurrence binds those descriptors and the ACK channel; Cargo checks
  the challenge-bound ACK and retains the current-publication lease through
  application exit. Cargo has no Worker V2 application transfer branch in
  either production or qualification builds; stale V2 envelope names are
  recognized only so they can be rejected before application spawn.
  Feature-free `fe2o3-host` builds export only the Worker V3 application,
  admission, verification, canonical pure-KFD inherited-application transition,
  and current HSA-backed generated migration route.
  Worker V2 application recovery, bundle admission, prerequisite
  authentication, HSA lifecycle, launch metadata, and workload-specific host
  adapters have been deleted rather than retained behind a qualification
  feature. General
  `#[kernel(typed)]` expansion, including the exact `f32` vecadd signature,
  emits only the generic Worker V3 adapter. The old
  `qualification_worker_v2` macro option, embedded vecadd artifact contract,
  generated `Kernel`/`Prepared` API, and example feature have been deleted.
  The receipt-complete Worker V3 promotion boundary is implemented, including
  exhaustive substitution checks. Its verifier trait is sealed in production,
  its decision constructor is private, and synthetic construction is available
  only under the explicit integration-test feature. A concrete protected
  verifier remains open. The protected service can now independently reacquire
  current policy and exact Worker-ledger state for one complete carriage. The
  production application receives the child-created endpoint, and the host
  exports a one-use auditor that verifies a fresh signed response under the
  pinned key without granting authority. The final verifier must still join
  protected key custody, the external monotonic rollback authority, and the
  owned compiler/proof/machine evidence. Recovery and
  verification admission are device-independent; the canonical KFD transition
  binds one checked physical device only when it joins generated host-memory
  packing to the exact current artifact, geometry, effects, and authenticated
  verifier decision in a private, move-only invocation authority. The MI300X
  scalar-GEMM test uses an explicitly synthetic verifier, so the HSA-backed
  migration route cannot be deleted and the application pipeline is not yet
  production-complete.
- Linux-only rustc and codegen-backend primitives use descriptor-backed procfs
  paths. The external Cargo path copies the backend into a rehashed, immutable
  sealed memfd and installs it after a compile-shaped managed wrapper
  invocation. The caller-selected compiler executable is not authenticated as
  rustc. This protects the measured bytes from pathname substitution; it is not
  a sandbox for hostile build scripts or procedural macros, which remain
  trusted inputs.
- `examples/regression-manifest-v2.txt` is the authoritative package/source-artifact
  inventory for ordinary checks and explicit artifact qualification. The route is
  data, never inferred from a package name; only `fe2o3-fill` currently selects the
  bounded `kernel-ir-v1` oracle. The manifest grants no production or GPU-execution
  authority.
- The Verus vecadd, fill, active-wave, LDS, and exact gfx942 wave/LDS
  harnesses prove bounded source-model properties under documented
  assumptions. The exact control,
  index, guarded memory access, and write body of the production `f32` vecadd
  kernel is mechanically shared with Verus through explicit thread and
  arithmetic adapters. Positive harnesses and paired expected-rejection
  mutations run in the required proof lane; the three real-body mutations
  additionally require one exact primary diagnostic and failed source clause.
  Verus uses a total model arithmetic adapter and does not prove that
  production IEEE `f32` addition,
  compiler output, HSACO, or GPU execution refines that model. Proof-record
  matching rejects incomplete or mismatched identities, but the records remain
  synthetic evidence rather than authenticated compiler-refinement evidence.
  `PersistentlyFreshMultiKernelProofAdmissionV1` additionally consumes
  non-clone per-kernel bindings from one exact local ledger history and requires
  unique kernels and generations, one finalized executable, and identical
  measured verifier policy and tool identities. It is local persistent
  consistency evidence, not rollback resistance, compiler refinement,
  prerequisite authentication, or load/launch authority.
  A separate bounded canonical pre-envelope proof capsule binds proof policy,
  execution and result records, target and payload identity, and the complete
  persistent-ledger ancestry used for freshness. It checks the finalized digest
  against the persistent executable binding and has bounded process-local
  duplicate detection. It does not provide durable single-use enforcement,
  compiler refinement, prerequisite authentication, or runtime authority.
- The G5 contracts now describe bounded independent-thread reads and writes,
  allocation provenance, bounds, injective writes, and deterministic proof
  obligations. Paired copy, gather, and affine elementwise bodies have positive
  and negative Verus harnesses. `fe2o3-verifier` canonicalizes bounded tool,
  policy, invocation, and result records, has a bounded shell-free process
  executor, and can convert validated results into descriptive proof records.
  The sole supported `gfx942` physical-machine analysis path executes the
  upstream-LLVM Object/MC worker from a sealed image under an immutable,
  policy-pinned runtime closure. It returns one canonical bundle containing
  closed static effects and a complete byte-exact instruction/CFG trace for up
  to two arbitrary requested entry symbols. All code locations are validated
  HSACO file offsets, the trace binds the exact effect record, and the
  authenticated receipt binds the complete bundle. A bounded inert analysis of
  that trace additionally derives block dominators, post-dominators, exact
  reaching definitions, and canonical natural loops with their exit edges; it
  rejects CFGs whose blocks cannot reach an exit. A separate bounded EXEC
  control analysis joins exact `S_CBRANCH_EXECZ`/`EXECNZ` sites to their taken
  and fallthrough blocks, unique two-half EXEC reaching definition, immediate
  post-dominator, canonical scalar mask operands, and a matching saved-mask OR
  site when one is structurally present. These are structural extractor
  facts, not compiler, source, Verus, address-safety, race-freedom, machine
  semantics, hardware reconvergence, a proof that any mask is empty, termination,
  or launch authority.
  The Worker V3 semantic-refinement join remains open.
- G6/G7 includes canonical multi-input AMDGPU link plans and a standalone
  direct LLVM/LLD worker with bounded Rust/C++ protocols. Device FFI macros and
  compiler validation bind import/export symbols, physical ABI, address spaces,
  effects, target, and code-object version. Cooperative and peer capabilities
  retain exact contexts, streams, and cleanup ownership. The opt-in
  `kernel-ir-worker-v2` flow now carries a real Cargo crate containing two Rust
  kernel roots and one shared internal helper through rustc collection,
  canonical helper-call resolution, verified kernel IR, an attempt-scoped
  textual LLVM handoff, exact compiler-produced symbol-role manifests, Cargo
  wrapper consumption, and byte-identical GenericLink/V2 execution in a
  measured direct LLVM/LLD worker for `gfx942`. Internal calls resolve by their
  canonical Rust source identity to one collected helper definition and its
  exact predeclared signature and export symbol; ambiguous, uncollected, or
  signature-incompatible callees fail closed. The worker links both kernels and
  the helper into one HSACO using LLVM and LLD library APIs directly, without
  COMGR or command-line linking. Cargo independently checks the exact two-kernel
  symbol set and the returned raw HSACO. Descriptor-free COV5 remains a raw
  compatibility publication; descriptor-bearing COV6 is canonically finalized
  downstream and the exact finalized bytes are durably published with an
  attempt-bound provenance receipt. The durable transaction has adversarial,
  legacy-marker migration, and raw/finalized process crash-recovery coverage.
  Recovery revalidates the exact journal, plan, admission, route, receipt, and
  completed attempt before clearing durable state. Fault exits are available
  only under the non-default `worker-v2-fault-injection-test-only` feature.

  At the worker boundary, COV6 is protocol version 6, LLVM module flag 600, and
  AMDHSA ELF ABI version 4. Native tests preserve two metadata entries, both
  `.kd` symbols, and one shared helper in a single deterministic output. The
  worker does not authenticate `.fe2o3.kd.v1` or construct an
  `ArtifactContainerV1`; those remain downstream responsibilities.

  For this exact profile, the worker canonicalizes every COV5/COV6 kernel to
  the complete 256-byte implicit-argument contract after optimization. The
  finalizer accepts the AMDHSA metadata's explicit-prefix size only for the
  authenticated General-V3 `gfx942:xnack-` COV6 producer and reconciles it with
  the descriptor's complete size. All other size or profile mismatches remain
  rejected.

  At commit `daf0b459ced07a25376670c83b1474eaebcd1a68`, the ignored native
  integration test builds the exact alpha/zeta Rust fixture, lowers both MIR
  bodies through Kernel IR, validates both backend witnesses, links with the
  direct LLVM Worker V2, independently inspects and canonically finalizes one
  two-entry COV6 artifact, and exports exact bytes with SHA-256
  `3a916cdabca05ac74d340889aab2067221d6d1252a7cde13e61c1786252565c4`.
  A feature-gated MI300X HSA run then loaded that digest-pinned artifact once on
  `gfx942:xnack-`, resolved distinct raw `alpha` and `zeta` symbols, ran both
  kernels for lengths `1`, `255`, `256`, `257`, and `1023`, checked independent
  CPU oracles and prefix/suffix canaries, and unloaded the executable once.
  The hardware harness deliberately uses the reviewed unsafe raw HSA boundary;
  it is evidence for the generated artifact's code, ABI, and behavior, not an
  execution of the production generated adapter or a general safety proof.

  At commit `dc9738e367c392f7716eacb8459ca73fa32abbbb`, a now-retired ignored
  MI300X test passed the same digest and boundary-length matrix through the
  generated alpha/zeta argument capabilities, selected-kernel preparation,
  reviewed load/resolve/dispatch/unload lifecycle, and safe `dispatch` SPI.
  It uses an explicitly fake prerequisite authenticator and test-only semantic
  witnesses, so it validates runtime composition and hardware behavior but is
  not production authentication, Verus evidence, or a machine-code safety
  proof. That Worker V2 host harness and its workload-specific adapters are no
  longer part of the source tree; the observation remains historical evidence.

  Compiler identity and origin are not authenticated, no Verus result or
  compiler/machine-code refinement proof is authenticated and bound to this
  executable, and the publication receipt grants no HSA load or launch
  authority. On the MI300X `gfx942:xnack-` lane, the ignored real-Cargo
  alpha/zeta Worker V2 publication test and the digest-pinned HSA hardware
  tests passed alongside the earlier direct-source and external-bitcode-provider
  publication tests. The retired runs do not constitute current production
  execution coverage. A production Worker V3 verifier must authenticate and
  join compiler, proof, and effect evidence before the sole safe SPI can run.
- G8 adds deterministic model generation/reduction and a bounded conformance
  harness that executes fill, vecadd, and affine kernels against an independent
  HIP/CPU oracle. `cargo fe2o3 inspect` performs bounded read-only decoding.
  `sanitize` and `debug` retain plan mode and can execute descriptor-pinned
  native ROCgdb under bounded supervision. ROCgdb precise-memory diagnostics
  are not a race, API, initialization, synchronization, or safety proof.
  The historical S09 source-debug pilot built one exact General V3 `alpha` profile
  at O0 into an alpha-only COV6 HSACO for `gfx942:xnack-`. It binds inert
  semantic and build identity records to the physical `alpha`/`alpha.kd` pair,
  verifies linked DWARF, executes a dedicated controller over lengths 1, 255,
  256, 257, and 1023 with CPU-oracle and canary checks, and uses native ROCgdb
  to inspect scalar and aggregate arguments, a reference value, physical slice
  fields, and local `i`; tuple and array locals also carry located DWARF. The
  archived lane produced only `local-capability-v2` evidence: it does not
  authenticate the compiler or runner, install production trust, materialize
  tuple/array runtime values at the fixed stop, cover optimized or general
  debugging, or provide a safety proof. Rows 45 and 46 and supplemental row S09
  are therefore `Partial`; only the bounded transcript checker is retained.

### Not yet integrated

- General MIR to kernel IR to AMDGPU lowering is not complete. `kernel-ir-v1`
  accepts the exact fill and vecadd shapes, and `kernel-ir-worker-v2` additionally
  accepts only the exact alpha/zeta General-V3 shapes on `gfx942:xnack-`; the
  elementwise recognizer remains the default emitter.
- The production semantic-MIR route now has one mandatory target-neutral
  ranked-PLIRON verifier sequence before Kernel IR lowering: memory bounds,
  global race freedom, barrier convergence, workgroup-memory initialization
  and publication, and declared semantic refinement. The passes are generic
  dialect checks rather than workload recognizers and share a bounded sparse
  index/dataflow analysis. Rust projection is complete for static ranked
  accesses and the checked `ThreadIndex`/`DisjointSlice` dynamic-access
  contract. Other dynamic pointer provenance, semantic CFG projection for
  barriers/workgroup memory, and source-declared equivalence still fail closed;
  the PLIRON passes and textual lit coverage do not by themselves establish
  source-to-machine correspondence.
- General V3 lexical registration, rustc-semantic
  scalar/shared-slice/`DisjointSlice` reconstruction, variable COV6 descriptor
  generation, safe value binding, checked buffer regions, lifetime-retaining
  packing primitives, backend witness emission, and signature-specific
  `Arguments` are implemented as source/unit foundations. At `d509ca5`, their
  generated slice capabilities can consume checked shared/exclusive subregions,
  retain allocation-relative region identity, and feed the existing alias and
  packing foundations. The macro emits the generic Worker V3 argument and
  preparation contract for every accepted signature; exact alpha/zeta host
  adapters have been deleted.
  Aggregates, return values, and arbitrary rustc layouts also remain outside V3.
  In required-envelope mode, the Cargo production path consumes a measured
  upstream canonical envelope-input capsule rather than synthesizing direct-link
  or proof evidence. It binds that input to the build attempt, durably stages
  it, publishes the exact canonical Worker V3 load-readiness envelope, and
  reconstructs the same envelope from durable input and HSACO claims across
  restart. The envelope retains the artifact container, bundle index,
  direct-link evidence,
  descriptor lineage, per-kernel proof records, raw HSACO, finalized payload,
  and canonical reacquirable publication claim. Cargo validates transport,
  canonical encoding, identities, and restart closure; it does not authenticate
  the supplied compiler, proof, or machine-effect claims. A bounded cooperative
  application handoff now transfers pinned envelope and artifact-directory
  descriptors to an identity-pinned sealed application while Cargo retains a
  fresh current-publication lease. The V3 occurrence binds the envelope,
  artifact directory, and ACK descriptors; the host revalidates them before
  returning an inert descriptor. This is the sole protected production
  descriptor handoff, but it grants no prerequisite, load, or launch authority.

  No production Worker V3 verifier yet promotes the retained compiler origin,
  signed MIR-to-live-PLIRON execution, Rust ABI, and machine-effect evidence
  into safe dispatch. Default decisions require the exact move-only V4 proof
  inputs and compiler-currentness evidence, but default builds expose no
  downstream implementation or decision-construction route; they therefore
  fail closed until the crate-owned protected verifier exists.
  Retired Worker V2 test
  authority is not an alternate route and cannot be selected in any build.
- Checked mutable views now support simultaneously live disjoint subviews via
  `split_at_mut`, with exclusivity enforced by Rust borrowing. The mechanical
  Verus proof of that split and its allocation-relative region theorem remains
  open.
- The generated contract identity authenticates compiler declarations and the
  exact payload bytes. The authenticated physical-machine worker extracts one
  indivisible effect-and-instruction bundle from exact `gfx942` HSACO bytes,
  but Worker V3 does not yet consume a semantic refinement receipt relating
  that trace to KIR. The fixed lowering, Kernel IR checks, host alias admission,
  and tests provide separate defenses. The generic ranked-PLIRON bounds and
  race analyses reject unsupported or conflicting compiler IR before lowering,
  but end-to-end source-to-machine safety still requires complete frontend
  projection plus authenticated compiler-refinement evidence. Trusted rustc
  diagnostic-item classification also remains part of the compiler TCB.
- Generated Worker V3 arguments retain the resources required for dispatch,
  but generalized asynchronous application APIs, cancellation, and composition
  remain incomplete.
- Worker V3 marker-to-artifact association remains part of the trusted
  compiler/linker contract and does not itself prove executable semantics.
- `cargo fe2o3 verify` and `build --require-proof` are roadmap commands. The
  current required Verus CI lane is invoked separately and does not prove the
  ordinary Rust function, compiler, ROCm, driver, or machine-code refinement.
  Verus proof identity/refinement is not authenticated into the generated
  vecadd artifact or required by its safe loader and launch API. The exact
  alpha/zeta source models have mechanical Verus proofs and bounded proof-record
  schemas, but no reviewed Rust-semantics refinement or authenticated
  source-to-Kernel-IR-to-machine-code refinement.
- The fail-closed rustc wrapper classifies and preserves approved bootstrap
  invocations, and the external Cargo path now composes compile-shaped managed
  invocations with the descriptor-pinned rustc executable and sealed backend
  snapshot. The selected executable is still not authenticated as rustc;
  rustc-descendant descriptor lifetime, dynamic loading, transitive shared
  libraries, and non-Linux execution remain unresolved.
- General Rust language support, frontend-to-layout integration, broad atomic
  and collective coverage, production direct-link integration, general
  device FFI, occupancy-complete cooperative launch, multi-device memory
  semantics, full sanitizer/debugger coverage, broad differential fuzzing, and
  authenticated Verus refinement remain parity work. The alpha/zeta hardware
  result covers only MI300X `gfx942:xnack-`; architecture-family breadth is
  absent. LDS, atomics, waves, collectives, fences, and barriers have bounded
  source/compiler paths. The exact gfx942 wave/LDS V2 Kernel IR also has one
  numerical MI300X result, but it is not joined to the genuine Rust artifact.
  These facilities are not yet broadly available from ordinary Rust kernels or
  validated across the full operation, type, target, and hardware matrix.

The evidence-gated comparison with cuda-oxide is tracked in the
[parity matrix](docs/cuda-oxide-parity-matrix.md) and the generated
[evidence dashboard](docs/generated/cuda-oxide-parity-dashboard.md). The
dashboard pins a status floor and records qualifying per-row evidence at that
commit or a landed descendant; it is not a claim that every change at the
current repository HEAD has qualifying parity evidence. fe2o3 is not yet at
parity.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the original
compiler/runtime plan and
[docs/implementation-roadmap-v2.md](docs/implementation-roadmap-v2.md) for the
current staged roadmap.

## Commands

Run diagnostics:

```bash
cargo run -p cargo-fe2o3 -- doctor
```

Inspect a bounded fe2o3 artifact without loading it, or print a normalized
ROCgdb execution plan:

```bash
cargo run -p cargo-fe2o3 -- inspect target/fe2o3/kernel.hsaco
cargo run -p cargo-fe2o3 -- sanitize -- ./target/debug/application
cargo run -p cargo-fe2o3 -- debug -- ./target/debug/application
```

Execution is explicit and bounded. Debug execution requires an explicit batch
or interactive mode:

```bash
cargo run -p cargo-fe2o3 -- sanitize --execute -- ./target/debug/application
cargo run -p cargo-fe2o3 -- debug --execute --batch -- ./target/debug/application
```

Sanitize fails when requested precise-memory coverage is unavailable. Race and
API coverage are reported as unsupported rather than inferred from a clean run.

Preview or remove only fe2o3-generated artifacts under `target/fe2o3`:

```bash
cargo run -p cargo-fe2o3 -- clean --dry-run
cargo run -p cargo-fe2o3 -- clean
```

The clean command discovers the enclosing Cargo project or workspace and
preserves the rest of its target directory. Planning opens and retains the
canonical project-root capability. Each successful no-follow component open is
authoritative: substitution completed before that open selects the current
ordinary directory, while substitution after it cannot redirect later access.
Metadata is used only after an open failure to produce a fail-closed diagnostic.

Destructive cleanup is supported on Unix, where the opened `target/fe2o3`
directory is passed to capability-relative opened-directory removal. With the
pinned capability implementation, Windows removal is pathname-based, so fe2o3
fails closed there; `--dry-run` remains available. Unix opened-directory removal
is not atomic against every concurrent rename and can fail after partially
removing the opened directory's contents.

This is intentionally narrower than pinned cuda-oxide's clean command, which
removes the project's full target directory. External-project build and run
orchestration are now supported, but local-clean parity remains partial because
fe2o3 deliberately removes only its guarded `target/fe2o3` output.

If `FE2O3_TARGET` is not set, `cargo-fe2o3` tries to infer the target from
`rocminfo` and falls back to `gfx1100`.

Each external build uses a generation identity that binds the selected target,
backend, Worker V2 configuration, and effective Cargo configuration. A changed
or failed generation receives fresh Cargo fingerprint state; successful
incremental builds republish the exact generated snapshot.

Validate the authoritative example manifest and list a lane:

```bash
cargo run --locked -p cargo-fe2o3 -- examples check
cargo run --quiet --locked -p cargo-fe2o3 -- examples list artifact-qualification
cargo run --quiet --locked -p cargo-fe2o3 -- examples list artifact-kernel-ir-v1
cargo run --quiet --locked -p cargo-fe2o3 -- examples list cpu-test-raw
cargo run --quiet --locked -p cargo-fe2o3 -- examples list cpu-test-wrapper-managed
cargo fe2o3 test --locked --all-targets -p fe2o3-vecadd
```

The two CPU-test queries form a sorted, disjoint, exhaustive partition of
manifest packages selected for Rust checks but not artifact qualification. The
partition is computed from the exact structural wrapper-managed projection, so
generic CI runs namespace-free typed-kernel tests through the mandatory
`cargo fe2o3 test --all-targets` host path and leaves ordinary packages on raw
Cargo without package-name rules. CI revalidates both complete lists and the
full structural projection after tests.

This binding-only path executes trusted workspace source, Cargo configuration,
build scripts, procedural macros, linkers, and test bodies. It rejects a caller
`--target`, `--config`, Cargo-side `-Z`, `--doc`, and `--no-run`, and rejects
ambient compiler, rustdoc, protected fe2o3, and test-runner selection. Configured
compiler, protected fe2o3, loader, and test-runner selection is rejected;
configured rustdoc is overridden with the disabled selection, and ambient loader
variables are scrubbed. The fixed runner retains and hashes Cargo's original test
executable. While Cargo's path remains stable, this preserves ordinary
`current_exe` and `$ORIGIN` behavior and prevents directory-entry substitution
between pin and execution; the runner checks the retained object again afterward.
That is not same-inode immutability, origin authentication, a sandbox, or an
atomic Cargo-configuration snapshot. The pre/post protected-configuration scans
are diagnostic checks against persistent changes, not a TOCTOU proof.

The command produces ordinary Cargo host-test files but grants no fe2o3 backend,
HSACO, publication, or immutable-artifact authority. It neither requests nor
establishes GPU evidence and performs no performance prediction. Trusted test
code is not confined from files, the network, or device nodes.

Run the repository validation lanes:

```bash
scripts/ci-local.sh generic
scripts/ci-local.sh generic-core
scripts/ci-local.sh shard-policy
scripts/ci-local.sh rustc-codegen-shard 04-memory-math-gemm
scripts/ci-local.sh workspace-test
VERUS=/absolute/path/to/verus scripts/ci-local.sh verus
FE2O3_TARGET=gfx942 scripts/ci-local.sh rocm-compile
FE2O3_ALLOW_GPU_SMOKE=1 FE2O3_TARGET=gfx942 scripts/ci-local.sh hardware-smoke
```

`generic` remains the complete serial generic gate. Hosted CI runs
`generic-core` once and executes every selector-free production integration
target in the checked-in shard manifest. `shard-policy` derives the complete
test-target set from locked Cargo metadata, requires an exact active-or-retired
partition, and rejects overlap, missing, duplicate, renamed, unknown,
malformed, empty, or newly unassigned targets. Retired targets are migration
inventory, not executed evidence; their oracle logic runs only in library tests.
Each hosted core or shard job uses separate Cargo and log directories; the
stable `Generic validation` check succeeds only after the core and all shards
succeed.

The historical S09 hardware entry point is retired because its debug HSACO
generator used the removed Worker V2 selector and HSA runtime. Its offline
DWARF/transcript checkers remain tested; hardware evidence must be regenerated
through a production Worker V3 compiler path and KFD runtime before the lane
can be enabled again.

`workspace-test` is the comprehensive local test gate. It runs all workspace
test targets except `rustc-codegen-fe2o3`, then tests that package in a separate
Cargo process. Do not replace it with one `cargo test --workspace --all-targets`
invocation; the codegen backend's `rlib` and unversioned `dylib` outputs can
collide across build variants. This lane can link ROCm libraries. The ROCm and
hardware lanes require a matching AMD GPU and ROCm installation.
The release-evidence collector requires a complete archive-relative row-link
map and records Git, rustc, LLVM, ROCm, driver, target, and stable lane
identities without changing matrix status:

```bash
scripts/parity-evidence.sh collect \
  --rows rows.tsv --hardware-lane mi300x-gfx942-release > evidence.tsv
scripts/parity-evidence.sh validate evidence.tsv
scripts/tests/parity-evidence.sh
scripts/parity-dashboard.sh check
scripts/tests/parity-dashboard.sh
```

`scripts/parity-dashboard.sh update` rewrites only the deterministic generated
Markdown and TSV dashboard. The check rejects stale paths, missing claims,
unsupported status upgrades, target/evidence mismatches, and generated drift.

To build or run one package directly:

```bash
cargo run --locked -p cargo-fe2o3 -- build -p fe2o3-vecadd
cargo run --locked -p cargo-fe2o3 -- run -p fe2o3-vecadd
```

Both commands enter the sole production route. The current vecadd
application fails closed before dispatch because its Worker V3 application
verifier has not yet been wired to the generated `Arguments`; it never falls
back to the embedded V2 artifact. `FE2O3_CODEGEN_PIPELINE` is no longer
accepted, and a production backend rejects `FE2O3_QUALIFICATION_ORACLE_V1`.

The retired manifest smoke command is not a production or direct-KFD path and
is no longer exposed. Hardware execution must enter an explicit runtime-owned
gate; artifact qualification never implies load or dispatch authority.

# Unsafe Code Policy

The target is safe public APIs with a small, reviewed hardware/OS implementation
boundary, not zero `unsafe` keywords. KFD MMIO, mappings, custom ioctls, raw device
memory, ABI exports, and some process setup require explicit contracts. Moving
these operations into generated strings, C, dependencies, or larger unsafe blocks
does not discharge those contracts.

## Source Inventory Gate

The normal `cargo-fe2o3` test suite, already run by `scripts/ci-local.sh`, checks
`scripts/unsafe-source-baseline.json`. The check tokenizes every tracked and
non-ignored untracked Rust source and compares per-file, per-kind counts against
the reviewed baseline. Comments and literals are excluded; test code,
feature-disabled code, and macro templates are included. Templates are counted
once where written, not once per expansion. Unreadable or untokenizable files
fail the check.

The inventory requires a Git checkout; it is not a source-archive test. Run the gate:

```sh
cargo test --locked -p cargo-fe2o3 --test unsafe_source_policy
```

After reviewing a deliberate addition or removal, refresh and review the diff:

```sh
cargo test --locked -p cargo-fe2o3 --test unsafe_source_policy \
  refresh_reviewed_unsafe_inventory -- --ignored --exact
git diff -- scripts/unsafe-source-baseline.json
cargo test --locked -p cargo-fe2o3 --test unsafe_source_policy
```

Do not increase a baseline merely to make CI pass. A new unsafe operation needs
an explanation of why safe APIs cannot express it and a reviewed invariant.
Reductions also update the baseline, preventing an old allowance from remaining
available. Counts are a review aid, not a proof: replacing an operation without
changing its kind/count still needs normal code review. Generated Rust strings,
doctests, dependency implementations, and native C/C++/assembly are outside this
source inventory and remain part of the system's trust boundary.

## Implementation Rules

### Softmax and Scalar GEMM Qualification Inventory Review (2026-10-23)

Reviewed the unchanged qualification sources at public commit
`2739349298149fbaa13409da24d7a494043fdb49`:

- `examples/row_softmax_general_v1/src/qualification.rs`, blob
  `982f298ec20f3a047a9a37c12147cb21513a68ed`: two unsafe blocks.
  The raw launch uses two disjoint guarded F32 allocations, eight owned
  pointer/length/dimension/stride parameters and the 48-byte explicit ABI.
  Its mode-specific rounded grid and 64-thread block are checked against
  the observed device limits before enqueue.
- `examples/scalar_gemm_v1/src/qualification.rs`, blob
  `e7caabe906e0b90971342f495c42b08ee59060c9`: two unsafe blocks.
  The raw launch uses three disjoint guarded F32 allocations, nine owned
  pointer/length/dimension parameters and the 60-byte explicit ABI. Checked
  shapes and the rounded 256-thread grid precede enqueue and device-limit
  checks; input views are shared and the output view is exclusive.

Each file's other block loads an external HSACO through the explicitly
opted-in `qualification-legacy-hip` escape hatch. The runner checks the
artifact's symbol, target, Wave64, kernarg ABI, workgroup metadata and hashes;
this path still trusts the external object's behavior, including load/unload
code. It does not provide authenticated production loading. The existing
safe launch API cannot express this external-object qualification contract.

Both adapters retain module, argument, allocation and stream owners through
unconditional stream synchronization, including enqueue errors. Ambiguous
completion aborts before those owners can drop. Full input readback and output
oracle/canary checks remain unchanged. Only these two inventory entries are
added; all existing allowances and the tokenizer's negative controls remain.
This review grants no protected execution, proof or GPU qualification authority
and does not modify the adapters, kernels, or runtime.

### Vecadd Qualification Inventory Review (2026-10-23)

Reviewed the following unchanged sources at public commit
`5a03850ab2e10db0250412de03e0c269cce4c09b`:

- `crates/fe2o3-hsa-runtime/tests/gfx950_advanced_hardware.rs`, blob
  `91cb62414eb976d70214de9aea489bcb50d492aa`: 18 blocks and two functions.
  The added unsafe forwarding block and private `dispatch_with_grid` preserve
  the original dispatch boundary; existing callers still use `case.grid_x()`.
  Vecadd supplies a nonzero, checked rounded grid. Exact symbol/ABI/alignment
  checks, owned aligned kernarg storage, unchanged explicit bytes, queue binding
  and the synchronous completion check remain before storage is released.
- `crates/fe2o3-hsa-runtime/tests/gfx950_advanced_hardware/vecadd.rs`, blob
  `ac8d1060d092486b0d1da117c15a50623c2c42ab`: four blocks for load, resolve,
  dispatch and unload. Input extents and rounded grid are checked before GPU
  initialization. Digest-pinned, inspected bytes and three distinct guarded
  allocations remain live through dispatch. The kernel set borrows its
  executable inside the closure and drops before the sole consuming unload,
  including ordinary error returns. Readback follows synchronous quiescence;
  the existing adapter aborts without unwinding if publication cannot quiesce.
- `examples/vecadd/src/qualification.rs`, blob
  `e0a18a5bc83b987dfbab251101bd8c5f18d26485`: two blocks for the explicitly
  opted-in legacy HIP external-module load and raw launch. This escape hatch
  trusts the self-contained object's behavior, including load/unload code;
  it is not authenticated production loading. Observed target/launch limits,
  the 48-byte ABI, six owned pointer/length fields and disjoint checked views
  precede enqueue. Module, argument and allocation owners survive unconditional
  stream synchronization even when enqueue fails; ambiguous completion aborts
  before those owners can drop.

These native ABI/lifecycle operations are not expressible by the existing safe
launch APIs; the review preserves their explicit qualification trust boundary.
Only these three inventory entries change. The source counter, its negative
controls, runtime/source/proof gates, kernels and oracles are unchanged. This
inventory enrollment is not a memory-safety proof, protected admission or a new
GPU execution claim; no historical inventory allowance is refreshed.

### Owned-Input Descriptor Probe Review (2026-10-19)

The compiler owner and integrating reviewer examined
`crates/fe2o3-host/src/compiler_execution_current_record_audit/tests/owned_input.rs`
at commit `3296c5112f2fc13511652e7a4f535697f0b2c161`, Git blob
`d085412309ee35bd3d95e4753f5ece1a2c414f6a`. This records exactly one
test-only unsafe block; the reviewed Rust source is unchanged.

The private `descriptor_flags` helper calls `libc::fcntl(raw, libc::F_GETFD)`
only to query scalar descriptor flags, including `EBADF` for retired numeric
slots. A retired slot cannot support the valid `BorrowedFd` required by a
safe `AsFd` interface; constructing such a borrow would assert the validity
these tests deliberately check. The fixed query takes no pointer and neither
adopts ownership nor closes or mutates the descriptor. Negative results capture
`last_os_error` immediately. Callers use it only for test-owned slot-retirement,
CLOEXEC and no-stale-close observations.

This is a contemporaneous slot observation, not evidence of immutable identity,
open-file-description exclusivity, inheritance provenance or production authority.
It is not a memory-safety proof, native qualification or permission to admit other
unsafe constructs. Only this file's `{"block": 1}` inventory entry is added;
all existing entries, the inventory checker and production guards are unchanged.

### Reference Enrollment Inventory Review (2026-10-15)

The three files below were reviewed at `622f671266926639173cb0b7fac82481f8ae966c`
and are unchanged since `8c27e0029db793a56974ca8974df11fa678fd39e`.
Only their stale inventory entries change; no unsafe source, checker or
unrelated allowance changes.

The native application `worker.rs` now records four blocks and two functions.
The sole unsafe entry forwards the original exclusive fixed-descriptor contract
to its private helper. The original owned budget outlives the complete callback;
the fcntl, from_raw_fd and prctl obligations and retained custody are unchanged.

The test-only `reference_enrollment_count_no_alloc.rs` observer records four
blocks, four functions and one implementation. Pointer/Layout obligations pass unchanged to System. Its const
Cell TLS observer does not allocate; the materializing-decode positive control
checks that allocation observation is active.

The isolated `protected_reference_enrollment_loan_v1_flow_fixture.rs` records
three blocks. Its child-only pre_exec remap uses close/dup2 with owned sources above all fixed slots. The sole
child adopts FD198 once for the image-check lifetime, and F_GETFD only observes
the consumed invocation slot. Bounded wait/kill/reap cleanup remains unchanged.
These reviews and affected CPU controls are not a memory-safety proof,
protected compiler admission, native qualification or GPU execution.

### Native Timestamp Inventory Review (2026-10-08)

Two reviewers examined the three timestamp-diagnostic files omitted from the
inventory at `32d7d27c496e6979e0e4c3b5a56d73ffe3fae724`. This adds exactly four
blocks for `engineering_gfx950_native_packet_diagnostic.rs`, three blocks and
two functions for `memory_linux_program_timestamps.rs`, and seven test blocks
for `memory_linux_program_timestamps_tests.rs`. Their implementations are
unchanged; no unrelated allowance or inventory gate changes.

The private clear/observe operations require the original retained signal arena.
Checked count, page-rounded extent, active accessible mapping, slot bounds and
64-byte alignment precede access to the aligned timestamp words. Clearing
requires idle exclusive ownership before publication. Observation requires the
original signal kind and an Acquire-loaded completed value before volatile
reads. The caller binds device, queue and frontier identity, checks currentness
and idleness, and restores profiling properties only after all original signals
retire and closing checks pass. Failure poisons the diagnostic path instead of
reusing its queue or arena. Test blocks use owned aligned inert storage; invalid
extents and misalignment refuse before dereference or mutation.

Volatile access and CPU tests do not prove the firmware signal ABI, device
coherence, immutable currentness, or native qualification. The maintained
tokenizing inventory and affected tests must pass on the integrated successor.

### A1-A3 Integration Review (2026-10-07)

The runtime integration review reconciles the 129 source entries reported by
the candidate `727f32888ff27391223e0c2b2546d867fdc0e453` hosted inventory gate,
plus the successor Registry4 and replica/graph boundaries. Four reviewers
divided the actual implementations and callers by ownership, not by counts:
host/macros/completion, runtime/KFD, process identity/custodian/spawn, and the
compiler/Cargo/proof boundary. Exact reviewed source hashes and raw review notes
are retained with the qualification evidence. This inventory update does not
qualify an unexecuted native path or promote a milestone to complete.

The integrating review found two refusal-cleanup defects. The seccomp-listener
receiver validated an SCM_RIGHTS envelope before owning its descriptors. It now
drains received rights into RAII before any envelope/cardinality rejection;
surplus rights are dropped and truncated messages are still refused. The
compiler-proof inherited-input entry admitted its first source before claiming
the second. It now claims both original, distinct slots once before either
flag or endpoint validation. Its unsafe contract still requires exclusive,
valid inherited ownership. Pipe-EOF tests cover refusal cleanup; isolated
subprocesses exclude unrelated test forks retaining transient descriptor aliases.

The reviewed obligations remain explicit:

- Generated ABI/effect adapters implement an unsafe correspondence contract;
  parsed layouts and exact retained artifact/currentness owners are required.
  Registry4 uses a separate completion contract. Early copied results neither
  retire its common source debit nor authorize scalar graph completion.
- Graph success follows original operation release and version commitment.
  Definite unpublished failure follows actual original disposal, never a
  classification string. Ambiguous custody remains retained and terminal.
  Replica test authorities and deny-only examples are not native providers.
- KFD ioctl buffers, nested event records, aligned atomics and volatile mappings
  retain their original owners. Queue destruction precedes event/mapping
  teardown. Exact-layout Box conversions preserve one allocation and one drop.
  None of these source reviews proves firmware, device coherence or the kernel.
- Process identity queries borrow original pidfds and initialized ABI records.
  Protected clone/exec keeps original cleanup owners before fallible admission,
  uses async-signal-safe child operations, and requires the existing dedicated
  creator, exclusive reaping and external whole-domain custodian contracts.
  Observed PIDs and readiness bytes do not reconstruct process authority.
- Inherited compiler/application inputs are consumed once under their explicit
  startup contracts. Registration, ACK and proof sources remain distinct.
  Test-only descriptor/environment mutation runs in owned isolated helpers;
  root-only fixtures remain opt-in. Unsafe parsed macro fixtures remain counted
  but do not supply a production implementation.

The maintained tokenizing inventory remains unchanged and fail-closed. A local
standalone build of that exact test source is only the mechanical refresh tool;
the pinned-toolchain Cargo gate and affected implementation tests must still
pass on the final integrated candidate. Counts are not executable refinement,
memory-safety proof, hardware evidence, or permission to bypass admission.

The separately reviewed two-cycle Registry4 successor adds one marker trait,
its private host implementation, and an inert test implementation. Both copied
result frames are reserved from the original result account before adoption;
the first result may remain alive while the second uses distinct storage and
its own one-shot gate. Rearm checks all four original recycled/copied generation
owners before mutation, retains monotone counters, and refuses a third cycle.
Common native/source custody survives both decodes and any refusal. This is
eight bounded publications with at most four outstanding, not rolling admission,
thousands in flight, device overlap or an out-of-order hardware result.

The private V53 consuming execution adapter adds one unsafe backend trait and
one private implementation of the existing gfx942 runtime authority. A safe
content callback cannot establish protected compiler/proof provenance or native
semantics, so the extension contract requires an independently approved provider
for the exact retained source, graph, artifact and actual invocation. No provider
is installed. The private implementation is constructed only after that contract,
opaque current-record binding, complete V53 readmission, exact evidence-subject
comparison and pre-/post-verification currentness checks. It retains the same
prepared request, checked device, publication, proof artifacts and output borrows
through the existing runtime completion boundary. Borrowed KFD inspection adds
no unsafe operation or authority. Root and independent source review found no
remaining scoped issue; content/accounting fixtures are not native qualification.
Only this file's `impl: 1, trait: 1` entry is added to the inventory. The ordinary
inventory gate and adapter compilation remain separate required qualification;
this scoped update neither refreshes nor claims to audit unrelated allowances.

The 2026-10-05 clone-compatibility follow-up reviews four inventory entries from
`65468c727` and `0921d0091`; it changes no runtime operation. The closed Linux
x86-64 ABI helper adds nine unsafe blocks and two functions. Its initialized
signal-mask and clone records stay live across each syscall, and the returned
pidfd is adopted exactly once. Only ENOSYS selects the unmapped legacy clone;
mapped or cgroup-placed children keep their clone3-only requirements. The legacy
child inherits blocked handlers, resets dispositions before unmasking, and must
exec or exit without allocation or Rust cleanup. The parent adopts all cleanup
owners before restoring its exact mask through a non-Send, non-Sync token.

The supervisor loses one block as its duplicate clone ABI moves into the helper;
the shared spawn implementation adds one block for that helper call while
retaining the original clone3 path. Fifteen test-only blocks use initialized
capability and signal records, self-installed seccomp filters in disposable
subprocesses, atomically owned pidfds, and exclusive terminal waits. The real
restricted-capability test has separately passed, but neither this inventory nor
that diagnostic establishes compiler/proof admission or production activation.
No unrelated inventory allowance is refreshed.

The 2026-09-30 native-custody audit reconciles seventeen omitted or stale
inventory entries. All seventeen source files are unchanged from the published
`edd71e6762252d1990d10d84d10cc53b666d09b7` checkpoint; this is a review record,
not new runtime behavior. The primary reviewed issuer fixed-slot ownership and
the coordinator publication fixture; two independent workers reviewed broker
publication/channel custody and retained-child cleanup respectively. No blocking
safety defect was identified by these static reviews.

- The six broker files contain sixteen blocks and three trait implementations.
  Test credential changes occur only in explicit disposable subprocesses;
  closed-descriptor probes never construct a borrowed owner. Production payload
  acquisition installs each actual owner before later fallible accounting.
  Retirement retains the independently owned barrier across destruction,
  without holding a coordinator mutex or forging `Send`. OFD-lock probes use a
  fully initialized native `flock` and a live, separately opened file.
- The seven spawn files contain fifty-one constructs. Late operations require
  the original account, creator thread, reserved slot and holder identity.
  Prepared tokens retain mutex custody and trace-bound lifetimes. Deferral keeps
  both payload and funding; the temporary `Arc` is dropped before terminal
  publication. Test implementations hold inert counters or empty payloads;
  fake descriptor-free children cannot signal scalar PIDs. The real traced-child
  fixture is an observation test, not protected compiler admission.
- The issuer V3 reader's three blocks inspect inherited flags and duplicate
  descriptors above the entire inherited table before transferring unique
  ownership to `File`. Its two test files contain seven and four blocks;
  fixed-slot mutations and `dup2` run only in isolated children with retained
  high-numbered sources. Raw borrowed slots remain live for each use.
- The coordinator publication fixture's three blocks use a genuine stopped
  trace and its original cleanup lifetime, plus initialized nonblocking OFD-lock
  probes. They neither resume a compiler nor establish production admission.

The inventory gate must still be rerun in an isolated worktree build. These
reviews do not establish native execution, protected proofs or GPU qualification.

The paired native V3 coordinator and provisioner mains each add one reviewed
unsafe call block. These dedicated binaries call their existing unsafe entrypoint
exactly once, before creating threads, handlers, environment borrows or Rust
owners for activation FDs. The coordinator transfers unique FDs 3..16 and retains
the external whole-cgroup termination contract; the provisioner accepts no
activation. Both own argv/environment mutation until exit. Return or unwind
terminates the process, with no retry or alternate entrypoint. These are ordinary
root Rust mains, not protected-child secure-entry shims. The inventory entries
record this source contract only: effective installed unit/LSM/cgroup custody and
direct-root intake startup still require qualification. No other inventory
allowance is changed by these two entry calls.

The dedicated proof-helper entry owns raw FD 3 before the first fallible check,
validates it before opening other descriptors, duplicates it with CLOEXEC and
closes the original exactly once. Its six mechanical unsafe blocks inspect raw
flags, borrow that live descriptor, close it, close unrelated descriptor ranges,
and poll one initialized parent pidfd. The two unsafe mechanical functions and
the public consuming entry require a sole dedicated secure-start process; these
are not shared-application APIs. The binary has one call block and the entry has
two. Four test blocks operate only in disposable subprocesses. Numeric parent
root and startup acknowledgments do not supply host-root provenance or proof
authority; the actual creator and outside whole-domain custodian remain required.
The private root-side helper adapter has one unsafe consuming launch function
and three blocks calling existing sealed staging, retained namespace spawning,
and exec confirmation. The complete backing enters the existing cleanup slot
before clone. Independent exec-status EOF precedes credentialed readiness.
Its staging test has one call block and an unsafe-function signature assertion;
neither constructs a positive runtime or deployment. The shared send test's two
blocks reset/unblock SIGPIPE only in a disposable subprocess, checking that the
real send uses NOSIGNAL; no application or host signal policy is changed.

The private root-to-V3-issuer composition has two unsafe functions and four call
blocks: delegation with the original confirmed compiler trace, sealed staging,
retained spawning, and exec confirmation. These calls retain the actual prepared
owners and compiler backing in the funded cleanup slot before clone. Final
descriptor checks, profile validation, independent exec EOF, exact readiness plus
EOF, and liveness checks do not remove the caller's creator-thread, exclusive-wait,
bounded-Drop and inherited-lock-alias obligations. Its one test-only unsafe block
stages inert files to check the fixed descriptor map; it never spawns or admits
them. The opt-in isolated root startup matrix adds four reviewed call blocks:
compiler staging, retained clone, held-exec confirmation, and actual V3 issuer
launch. It constructs genuine prepared inputs, closes all inherited lock aliases,
keeps the compiler at its first exec stop, and drains the independently funded
cleanup pool before its creator exits, including during unwinding. These tests
do not resume a compiler or admit a deployment. The scoped root observation view
uses safe pidfd/procfs wrappers; its original thread/account and unreaped custody
checks grant neither compiler admission nor tree supervision.
The shared pipe framer and socket-mode normalization use safe descriptor
APIs. This inventory does not establish installed production integration, an
authorized compiler occurrence, protected proof execution or GPU qualification.

The gated namespace collector in `child_namespace_report.rs` uses six unsafe
blocks and one private syscall function because post-clone observation cannot
use allocating or TLS-dependent wrappers. Its x86-64 stat buffer layout and
syscall clobbers are explicit; static NUL-terminated paths, exclusive writable
outputs and single-close descriptor ownership bound the raw accesses. Every
syscall failure is tested. The report is inert without private-pipe provenance,
EOF, retained pidfd custody and the exec gate.

The synthetic native consuming fixture has seven process-bootstrap blocks and
sixteen static-issuer I/O blocks. Bootstrap drops credentials/capabilities only
in the disposable child. I/O adopts fixed inherited descriptors once, checks
initialization before reading syscall outputs, uses finite nonblocking operations
over live buffers, and never retries close. The profile integration fixture's
fifteen blocks cover the same isolated credential setup plus a non-leader UTS
namespace and raw clone3 child, with exclusive pipe/pidfd custody and bounded
reaping. These are test-only privileges and operations, not production authority.
The four consuming cases passed under the exact locked profile on MI350; they
do not establish signing, durable recovery, protected proofs or GPU execution.

The inherited broker test that closed a still-owned descriptor before borrowing
it again has been removed: `forget` after validation cannot repair that I/O
ownership violation. Existing descriptor-substitution and flag-mutation tests
keep every owner valid while checking continuity refusal. The audit inventory
also accounts for moved socket/pidfd inspection sites without treating relocation
as removal of their safety obligations.

The shared x86-64 secure entrypoint restores confinement before inspecting the
kernel's initial argument table. It requires one nonempty, bounded argv0 and an
empty initial environment, then writes a private atomic once before libc startup.
No public setter exists. Rust admission requires that observation and rechecks
bounded command bytes; it never restores dumpability to read root-owned procfs
environment data. This extends the existing reviewed assembly boundary without
adding an unsafe Rust site or granting service identity or proof authority.

The native profile and compiler-execution subject test allocators forward all
pointer/layout operations unchanged to `System`. Their thread-local counters
use saturating increments so exhaustion cannot unwind through `GlobalAlloc`;
deterministic maximum-count tests check saturation and restoration. Ancillary
custody retains separate ownership for rustix's SCM_RIGHTS descriptors and the
guard's SCM_PIDFD descriptors. Its initialized aligned backing, checked record
bounds, one-shot adoption and unwind cleanup remain required despite extraction
into shared modules. KFD debug-metadata publication has a separate retained
no-queue transaction and native ownership contract; its registration does not
grant queue or dispatch authority. See
[no-queue metadata](gfx950-debug-metadata-noqueue-v1.md).

The no-queue review covers one example call block, one consuming unsafe function,
three retained metadata-store blocks, and two native ioctl blocks. Its transport
uses initialized exact-layout inputs and retains all native custody after any
possible exposure. Fresh-process and no-foreign-runtime obligations remain with
the caller; observational isolation checks do not prove them. The transaction
test's single remaining block reads the freshly supplied root synchronously,
before mutation. Later assertions consume copied transition snapshots, never a
saved shared-derived pointer across mutable access. This reconciliation grants
no GPU qualification or production launch authority.

The separate debugger-observation sibling adds one call to that same consuming
no-queue unsafe function, under the unchanged reviewed-process lifetime and
no-foreign-runtime contract. Its three unique host-only rendezvous exports use
`unsafe(no_mangle)` solely to retain fixed breakpoint names; they expose no
pointer or GPU capability and perform only host atomic operations. The first
marker precedes the sibling's KFD/VM work. Marker presence and entry snapshots
do not prove startup or lifetime isolation. The controller must retain owned
process custody and an independent outer family-cleanup contract. These source
inventory additions do not qualify debugger acceptance or GPU execution.

The source-safety fixture `owned_unsafe_closure.rs` deliberately contains one
empty unsafe block inside an owned `FnOnce` closure. It exercises rooted source
rejection even when MIR optimization removes the empty block. There is no unsafe
operation or runtime permission to replace with a safe API: removing the syntax
would remove the negative case. The inventory records this test-only block;
`production_collector_rejects_reachable_unsafe_rust_with_rooted_diagnostics`
must continue rejecting it before artifact export. This reconciles the fixture
introduced in `669713204edf8f6685b79f604bd5f05708cedcf2`, without changing its
source or the inventory gate.

The `workgroup_unsafe_callback.rs` and `workgroup_external_unsafe_callback.rs`
source-safety fixtures likewise each contain one deliberate empty unsafe block.
They ensure that traversing an authenticated `with_workgroup` provider still
checks user callbacks, including an external callback factory under optimized
MIR. These are negative-test syntax, not unsafe runtime operations. Each block
is counted separately, and the rooted source-safety test must reject both
before artifact export; provider authentication grants no callback exemption.

The Wave64 source-capture fixture introduced in ec9a526c55f82825e17c7a507507d6cb1debf7ce
contains one deliberately unsafe call in a macro template and three unsafe
lookalike functions. The call is a negative case: the collector must reject the
user helper's unsafe block before semantic capture. The three unreachable
lookalikes have the provider's scalar signatures but different local definition
identities; the callback mutation checks must reject them as trusted terminals.
Their bodies only return an input. Replacing these declarations with safe
signatures, or removing the unsafe call syntax, would remove the intended
authentication/source-safety negatives. The inventory counts the template
once, not once per scalar expansion. This reconciliation changes no fixture,
source-admission rule, trusted-provider definition or runtime behavior.
See production_rustc_driver_wave64_capture_source_v1_tests.rs and
collector/production_wave64_shuffle_terminal_v1_tests.rs for the actual
callback assertions; an inventory pass alone does not execute those callbacks.

- Keep pure compiler, simulator, debugger-engine, and profiler-analysis modules
  unsafe-free, using `forbid(unsafe_code)` where the crate boundary supports it.
- Prefer existing safe typed OS APIs. Keep descriptor ownership in `OwnedFd`,
  `BorrowedFd`, `File`, or another owner; do not scatter `borrow_raw` conversions.
- Preserve exact descriptor identity, symlink policy, CLOEXEC, publication,
  locking, error, and teardown behavior during replacements.
- Keep hardware and process boundaries narrow and document why each unsafe
  operation's lifetime, aliasing, initialization, ABI, and concurrency
  requirements hold. Post-fork callbacks need async-signal-safety review even
  when the functions they call are individually safe Rust.
- Do not remove unsafe traits or raw-launch preconditions without replacing
  their guarantees. Extend checked/generated APIs to reduce caller obligations.
- Retire HIP/HSA qualification code only after its users and test coverage have
  replacements. Default production execution remains direct KFD.
- Test ownership and failure behavior, including compile-fail tests where
  applicable. Host tests supplement but do not replace GPU/MMIO validation.

### Nullary Typed Kernel Rejection Fixture

The test-only `fe2o3-macros/src/zero_argument_typed_v1_tests.rs` contains one
`unsafe fn` inside `parse_quote!`. This is parsed syntax supplied to
`validate_typed_kernel_profile_v1`, which must reject it even when the kernel
has no arguments. The declaration is never compiled as a function or called;
there is no unsafe memory, OS, FFI or GPU operation. Replacing its signature
with safe Rust would remove the source-safety negative being tested.

The inventory counts macro-template syntax once, so this file receives exactly
one `function` entry. The tokenizer, validator, fixture and every other
inventory entry are unchanged. Run the nullary macro tests and the complete
source-inventory gate. This reconciliation grants no runtime or launch authority.

## Engineering Dispatch Review

The engineering dispatch preparation and sequence changes (`7abce5c16`), peer
ownership (`2804bf6b6`), and peer sequences (`902fef6e1`) retain explicit unsafe
entry points because argument validation cannot prove that unauthenticated
machine code honors its ABI, buffer access, bounds, and termination contracts.
The operator must supply those guarantees and use the documented disposable,
exclusive process. Every error is terminal; uncertain native owners are retained
until process teardown rather than freed speculatively.

Dispatch preparation and consumption occur under the owning context or group's
mutable borrow, without an intervening free, map, or load. Complete sequences
are prevalidated before the first packet is published. Serial completion and
idle checks prevent argument storage from being reused while a prior dispatch
is outstanding. Peer tokens bind the exact group incarnation and allocation;
participant checks bracket dispatches, and errors quarantine the group. Peer
read-only access is an argument-binding policy, not hardware-enforced memory
protection. The caller still must trust every kernel's actual accesses.

The reviewed inventory counts for `engineering_gfx950.rs`,
`engineering_gfx950_peer.rs`, and `engineering_gfx950_peer_performance.rs` are
respectively four blocks/four unsafe functions, one/two, and one/one. Host tests
cover pointer bindings, bounded sequence ordering, injected failures, partial
mapping cleanup, quarantine, and ownership. Run them and the inventory gate with:

```sh
cargo test --locked -p fe2o3-kfd --features engineering-gfx950 --lib
cargo test --locked -p fe2o3-kfd --features engineering-gfx950 --doc
cargo test --locked -p cargo-fe2o3 --test unsafe_source_policy
```

These checks do not qualify native queue/MMIO execution, authenticate kernel
semantics, or grant protected runtime authority. No runtime behavior changes
as part of this inventory reconciliation.

### Independent-Rank Engineering Rounds

The additive round API separates private publication and completion observation
under the same exclusive owner. Its extra private unsafe publication boundary
does not manufacture a safe launch API: unauthenticated machine code still needs
the trusted-kernel obligations documented in
`gfx950-engineering-peer-round-v1.md`. Cross-command write conflicts reject,
all commands prepare before publication, and each distinct queue retains its
own kernarg and signal. Full group entry/exit checks remain, with live checks
during overlap. Errors retain every uncertain native owner until process exit.

The revised inventory records five blocks/five unsafe functions in
`engineering_gfx950.rs`, one block/one unsafe function in the new round module,
and one test-only block exercising empty-round rejection without native owners.
The peer owner and serial-sequence inventory entries are unchanged. The new
test block contains no GPU operation. Host fault tests and inventory checks are
required but do not substitute for independently scheduled native qualification.

### Ordered Engineering Batches

The ordered-batch additions reviewed against `21682228486f` retain the same
exclusive disposable-process and trusted-machine-code obligations. The worker
call in `engineering_gfx950.rs` enters the private unsafe
`Context::dispatch_ordered_batch` boundary; neither site establishes safe
arbitrary-kernel execution or production gfx950 admission.

All dispatches are checked before exposure. The context retains kernels,
buffers, and separate kernarg and signal slots through ordered publication and
completion. Observing the final signal is insufficient: every retained signal,
queue identity and completion frontier is checked before reuse. Any error
poisons the ordered path; the worker retains uncertain native ownership until
the caller terminates the process. Existing host regressions cover incomplete
publication, deadlines, signal failures, identity checks, and queue rollover.
They do not discharge the trusted-code or native queue/MMIO obligations.

This review changes only the inventory to six blocks/five unsafe functions in
`engineering_gfx950.rs` and one unsafe function in
`engineering_gfx950_ordered_batch.rs`. The library, doctest and inventory
commands above remain required; no runtime behavior is changed.

### Engineering Dispatch Timestamp Diagnostics

The explicit ordered64 profiling command adds one unsafe worker call and one
private unsafe entry. The entry preserves the same trusted-machine-code and
disposable-process obligations as ordinary ordered64; no safe launch authority
is created. Five orchestration blocks delegate existing ordered execution and
the four new raw-memory helpers under the exclusively borrowed queue owner.

The raw-memory module has four unsafe functions and seven unsafe blocks. Safe
byte slices cannot represent concurrently GPU-visible control/signal storage.
These operations access only the reviewed aligned property word and signal
timestamp words: enable/restore use CPU-owned properties only while idle;
clearing requires no outstanding packet; reading follows acquired completion
and forbids signal reuse. Every pointer retains extent/alignment checks. Any
error poisons the owner; only fully successful capture restores properties.
Ten test-only blocks exercise those same helpers on inert aligned owned bytes,
including pending/wrong-kind signals and a rejected misaligned pointer. They
perform no GPU operation. The orchestration/wire test module has no unsafe sites.

The reviewed final inventory is eight blocks/five functions in
`engineering_gfx950.rs`, five/one in its new timestamp module, seven/four in
`memory_linux_dispatch_timestamps.rs`, and ten blocks in its test module.
Of the engineering entry's change from the old baseline six to eight blocks,
one already exists in d10f49bf's ordered64 worker route; only one is new here.
Separately, d10f49bf's ordered-batch module already contains two unsafe blocks
and three unsafe functions, while its older baseline lists only one function.
Those existing ordered64 wrappers retain identical trusted-code obligations;
this patch reconciles their inventory without changing their implementation.
No unrelated inventory entries are refreshed.

Run the existing full source-inventory gate and the engineering library,
doctest and strict Clippy checks remotely. Inventory review and CPU tests do
not qualify the profiling ABI on installed firmware, turn raw GPU clock ticks
into nanoseconds, or establish shader-only execution time.

### Packet-Incapable Debug Empty-Queue Retirement

The opt-in gfx950 empty-queue successor adds one consuming unsafe function,
`Gfx950DebugExecutionPreparationV1::begin_empty_queue_runtime`. The caller
must retain an isolated disposable process, exclude foreign KFD/ROCr runtimes,
queues, code injection and concurrent runtime actors through completion or
process termination (also after failure or Drop), and externally bound/reap
blocking native calls. Neither static preparation nor the in-library gate
proves these lifetime obligations. The returned move-only owners expose no
packet publication, doorbell write, dispatch or sampling operation.

Two new blocks in `runtime_debug_empty_transport_v1.rs` perform the fixed
runtime-disable and trap-clear ioctls. Initialized 16-byte input/output and
24-byte input-only ABI records respectively borrow the actual retained
Context/device fd exclusively; mode-zero output must remain exact. Two blocks
in `queue_linux.rs` destroy the same retained event with an owned 8-byte setter
record and unmap the exact privately owned doorbell VMA. Event poisoning and
doorbell deactivation precede their native attempts, preventing retry after
an error or ambiguous return; no raw pointer or numeric-owner constructor
escapes. These kernel ABI and VMA operations cannot be replaced by ordinary
safe Rust memory access.

The private cursor marks each step Attempting before its action and never
advances on error or unwind. Actual queue destruction precedes event destruction,
metadata withdrawal, runtime-disable return, trap clear and doorbell unmap;
allocation GPU/CPU unmap, handle free, VA release and accounting follow.
Unresolved backing and original descriptors remain in cold custody on failure,
panic or Drop, with no Drop ioctl or speculative cleanup. Only the private
same-owner terminal witness releases retention and closes descriptors last.
Fork checks precede native work and inherited gate-lock access. Existing
ambiguous-VA cleanup retains its abort policy. These are local retirement
invariants, not debugger ACK, sampler exclusion, physical capture or dispatch
qualification.

The exact inventory delta is `queue_linux.rs` 45 to 47 blocks (its one extern
block unchanged), two blocks in `runtime_debug_empty_transport_v1.rs`, and
one function in `engineering_gfx950_debug_execution_v1.rs`. Failure-order,
ABI and compile-fail controls supplement review; the source-inventory gate
and feature tests remain required. This reconciliation changes no runtime
implementation or existing plain/noqueue/gfx942 behavior.

### Fixed One-Stop Debug Target

The separately reviewed one-stop sibling adds two consuming unsafe functions:
`Gfx950DebugExecutionPreparationV1::prepare_fixed_one_stop` and
`Gfx950DebugOneStopPreparedV1::publish_at_owned_checkpoint_once`.
The first retains the original isolated disposable-process, no-foreign-runtime,
no-injection and exclusive native-actor obligations for the entire lifetime,
including failure and process teardown. The second additionally requires the
actual SAME native debugger client to join this exact owned process, source
checkpoint, executable, object, queue, packet and metadata before the first
publication effect. Actual attach-before-runtime, LoadedSuccess, a sole
successful event_processed acknowledgment, reviewed trap/CWSR/TTMP setup,
sampling exclusion and an exact one-shot pre-resume gate are caller obligations.
The client and supervisor must retain control through the unique debug stop,
separately qualified completion-only resume and bounded local retirement.
A JSON record, source digest, runtime-enable return or breakpoint hit cannot
prove those conditions. The source checkpoint itself does not enforce them.

One new unsafe attribute, `#[unsafe(no_mangle)]`, gives the host-only
`fe2o3_gfx950_one_stop_prepublication_checkpoint_v1` rendezvous its fixed,
unique symbol. The safe marker merely passes a borrowed record pointer to
`black_box`; it never dereferences a caller pointer, reads an acceptance
flag or grants native authority. The record stays in the original private Box;
all object addresses, packet bytes, signal BASE and separate VALUE+8 come from
actual retained owners. ABI/symbol collision and native debugger interaction
remain reviewed external boundaries, not properties established by the marker.

There are no new raw-memory or ioctl blocks in this donor. It reuses the existing
bounded memory/AQL/native primitives under retained custody, with INVALID body
before the single release header and one doorbell store. Completion comes from
the actual initialized signal and queue frontiers, not a report. Every failure
or unwind retains uncertain resources; all nine allocations retire before the
distinct same-owner terminal witness, and owned descriptors close last.
The original deadline is rechecked after descriptor Drop; late refusal preserves
the fact that those descriptors closed without attempting to close them again.

The exact inventory delta is two functions (one preexisting plus one new) in
`engineering_gfx950_debug_execution_v1.rs`, one function in
`engineering_gfx950_debug_one_stop_owner_v1.rs`, and one attribute in
`engineering_gfx950_debug_one_stop_checkpoint_v1.rs`.
No other inventory allowance changes. Comments and compile-fail doc examples
are excluded by the actual tokenizer; the checkpoint attribute is included.

This policy review does not qualify one-stop queue publication, a debug stop,
same-client acknowledgment or physical sampling. Earlier packet-incapable
EmptyQueue qualification and static fixture/startup results do not transfer
to this new owner or any future debugger build. Run the source-inventory,
engineering library, doctest and strict Clippy gates before any separately
reviewed native qualification. No native invocation was performed for this
inventory reconciliation.

### Opt-In Engineering Wait and Token-Program Entries

The entries added in `807f0bef7` preserve the expert disposable-process,
selected-device consent, trusted machine-code and terminal-error obligations.
The legacy entry now forwards through one private unsafe policy selector; two
separate public unsafe entries select bounded active polling or token programs.
These are the three additional unsafe functions and three forwarding blocks
in `engineering_gfx950.rs`, taking its inventory from eight/five to eleven/eight.
Safe argument validation still cannot establish the supplied machine code's
actual memory accesses, ABI compliance or termination, so the unsafe boundary
must remain explicit. No pointer, mapping, ioctl or kernel-trust obligation is
removed or discharged by these wrappers.

Root reviewed the three exact forwarding blocks, the shared worker lifecycle,
the retained token-program owner and the ordered wait/publication delegation.
Policy is selected before Ready and cannot change during the worker lifetime.
The default remains 50us sleep with token programs disabled. Active polling
keeps the original completion/currentness checks and aggregate deadline, uses
a bounded 10ms window, and rejects timestamp profiling. Token registration
retains descriptions rather than queue packets; each execution validates all
arguments before any group publishes and shares one aggregate deadline.
Resource mutation requires releasing the registered program. Partial execution
is terminal, poisons the ordered path and retains uncertain native owners until
disposable-process teardown; it is never a retry or safe-launch guarantee.

This reconciliation changes only this one inventory entry and review records.
Run the source-inventory gate, engineering host tests and compile-fail doctests;
CPU evidence does not qualify native GPU execution or protected authority.

### Whole-Program Engineering and Startup Reconciliation

The issue #289 review covers six inventory rows on `166670ab7`, not a blanket
refresh. The affected implementations, tokenizer and baseline are unchanged from
the recorded failing `12bf5e7a` inventory run. The following source contracts
were reviewed independently of test execution; the inventory gate and affected
CPU tests remain required after this correction.

The engineering additions in `eded9474b`, `dcf345452` and `55c1a9b6` add three
public unsafe entrypoints and one private unsafe policy delegate, with four
forwarding blocks. Native whole-program submission and the explicit 512-slot
family preserve the existing device-consent, trusted-machine-code, dedicated
disposable-process and terminal-error obligations. The boundary-fence entry
additionally excludes host, peer and other-queue communication during execution;
all host inputs precede publication, and only final system completion plus
every retained signal's retirement permits data access or reuse. Safe metadata
checks cannot establish those machine-code and external-actor obligations.
Policy is fixed before Ready. Ordinary entries retain their original policy;
legacy commands retain their 256-slot limit even in the opt-in 512-slot worker.

The two new `memory_linux.rs` blocks implement operations ordinary byte slices
cannot safely express for retained GPU-visible storage. Signal initialization
checks the complete rounded arena extent before writing distinct aligned
64-byte signal records, including their initialized atomic values. Its private
production caller checks queue idleness and retains both allocation owners
before initialization. Existing exact-layout reuse does not reinitialize live
signals. Closed-program header publication borrows a checked live, aligned
AtomicU32 slot; INVALID state and the typed program position/count/setup must
match before a release store preserves the setup halfword and publishes the
header. The original private reservation supplies the slot and all INVALID
bodies are written before any header. No raw pointer escapes either helper.

All dynamic dispatch validation precedes publication. Native completion checks
the final signal, queue identity/frontier and every retained signal before
advancing the completed frontier. Errors poison the path and the worker retains
uncertain native resources until disposable-process termination; they never
permit retry or speculative release. Normal close and queue rollover destroy
the queue before releasing program arenas. Host tests cover extent/alignment,
wrong-position and old-route refusal, all-signal retirement, partial-publication
faults and slot-policy separation. They do not qualify native GPU execution.

The `dbd70fd41` host change adds one unsafe function/block for the exact V89
inherited handoff. Its startup contract excludes concurrent environment and FD
mutation and transfers every inherited descriptor exactly once, including on
refusal or unwind. It uses the existing one-shot claim, schema/alias refusal and
owned-descriptor recovery, then the same retained-currentness/acknowledgment
sequence. The shared mixed-application function/block moves from the V53 file
to `production_mixed_application_family.rs`; it is counted once where written,
not twice for its includes. Concrete V53/V89 aliases and auditors remain
distinct. Preparation and current-record checks grant no proof or load authority.

The `d5441e0e4` native loader test adds one block adopting two fresh live fixed
descriptors. The exact-test subprocess's installer relinquishes them with
`into_raw_fd`; the block creates their sole owners. Startup captures only private
CLOEXEC duplicates, and the original owners remain live until their explicit
drop. This is isolated FD-lifetime coverage, not native compiler admission.

The exact changes are: handoff blocks/functions 14/4 to 15/5; the mixed-family
move 1/1 with the old V53 row removed; engineering 11/8 to 15/12; memory blocks
38 to 40 with its extern block unchanged; and native loader test blocks 2 to 3.
No other row, unsafe implementation, tokenizer, assertion or admission policy
changes. Run the full source-inventory gate, engineering library/worker tests
and doctests, host handoff/V89 tests and doctests, and the native loader's
`native_loader_copies_enter_the_same_account_without_consuming_caller_slots`
test. This reconciliation grants no installed-runtime, proof or GPU authority.

### Conditional Invocation Binding Comparison Fixture

Peer commit 5e1f9c3441066f13f5d501a95349a8a99fe886be adds one test-only unsafe
implementation in conditional_authority_tests.rs. The private
BindingComparisonFixture is compiled only as a child of authorized_execution
under cfg(test). Its constructors and all uses remain in three comparison
tests; none supplies it to an execution entrypoint or constructs a checked
device, prepared dispatch, telemetry token, or native request.

An unsafe impl is necessary to exercise the actual generic
validate_authority_bindings_v1 helper without duplicating its behavior or
weakening its unsafe trait bound. The fixture returns inert identity values
and updates a local Cell counter; it performs no raw memory, OS, FFI or device
operation. Tests check that family/payload mismatch precedes currentness,
that matching values still require currentness, and that contract/device
mismatches still refuse. This retains the existing private preflight-test
pattern, not a new execution-authority constructor.

These fixture values do not authenticate a Worker V3 publication or satisfy
the trait contract for native execution. Safety depends on reviewed private
test-only confinement and the complete call-site census, not the hashes,
fixture name, a test-configuration exemption, or the inventory itself.
Any future escape or execution-entry use requires a new safety review even
if its count remains one. Real authority still requires the unchanged
verifier, invocation, device and retained-currentness obligations.

Only this file receives impl: 1 in the inventory. No unsafe syntax, production
implementation, tokenizer, gate or other allowance changes. After integration,
run the conditional-authority and existing identity-comparison tests, runtime
compile-fail doctests, and the full source-inventory gate. This reconciliation
grants no runtime, native, GPU or protected authority.

### Fallible CPU Codec Node Allocation

The private `Scope::boxed` helper in
`fe2o3-verifier/src/portable_reference_v1/codec.rs` adds one unsafe block.
The codec must prepay storage and report allocation failure on the original
budget. It uses the stable fallible Vec allocator rather than adding an unstable
allocator API requirement or making an infallible Box allocation.

The helper rejects zero-sized types, reserves exactly one element, verifies the
returned capacity, and initializes exactly one value. The resulting boxed slice
and that value have identical allocation size, alignment, and global allocator.
The raw cast removes only slice-length metadata; ownership transfers once into
`Box<T>`, with no surviving alias or duplicate destructor. Failure drops the
original value and allocation before the enclosing scratch reservation is
released. The code comment records these obligations at the conversion.

Review covers the complete private helper and its recursive-expression callers.
Tests cover zero-sized refusal, over-alignment, a single destructor on normal,
error and unwind paths, and budget cleanup. Run the CPU codec tests and the
source-inventory gate after integration. This allocation primitive authenticates
no CPU source, proof, compiler artifact, or GPU launch.

### Shared Compiler-Execution Input Slots

The selected compiler admission and the native V3 preparation share one private
guard for inherited policy slot 202 and service slot 195. Both must be live and
non-CLOEXEC before policy duplication: otherwise the duplicate could occupy a
missing service slot and acquire a second closer. Admission exclusively consumes
these protocol slots. The two production unsafe blocks inspect scalar F_GETFD
flags and close an owned slot once; neither fabricates a BorrowedFd or OwnedFd
for a potentially absent input. Ownership is cleared before closing or passing
the service slot to its consuming client, and close is never retried. The native
path still charges its original account before inspecting either descriptor.

Three test-only unsafe blocks cover isolated-child pre-exec cleanup, F_GETFD
absence checks, and adoption of one installed live descriptor after disarming
the guard. Pre-exec performs only scalar close/error handling on child copies.
Each child has a finite deadline and is reaped on timeout or polling failure.
All descriptor duplication uses the existing rustix safe API without replacing
live owners. The native test now shares those helpers and has no unsafe blocks.
Tests cover missing/CLOEXEC inputs, policy and client refusal, the actual
descriptor-allocation collision, unwind cleanup and transferred ownership.

The inventory adds only the three reviewed test blocks. The selected production
file remains at two; the native production and test files now contain none.
Unrelated inventory mismatches are not approved by this entry. These refusal
tests grant no protected execution, native activation or GPU qualification.

### Native Process Boundaries

The native compiler-execution and external-anchor startup families need raw
descriptor operations at their isolated-process entrypoints. Their callers must
transfer exclusive ownership of the fixed input slots; inspecting a descriptor
or authenticating its bytes does not establish Rust I/O ownership. Validate the
whole input table before opening or duplicating files. Duplication leaves the
source owned until the guard is disarmed immediately before its consuming close.
Close errors never permit a retry, including during unwinding. Resource refusal
must obey the same ownership contract as successful admission.

Rustc's dynamic loader does not provide that transfer contract: it can construct
multiple backends after initializing threads. The safe backend factory therefore
only duplicates slots into new close-on-exec owners, without adopting or closing
the originals. Scalar `fcntl` errors do not fabricate borrowed descriptors. The
original slots remain process/caller-owned until compiler exit; successful
capture marks both close-on-exec. Failed capture can leave originals inheritable
and does not assume their cleanup obligations. Successful child completion is
required before finalization. An owned, mutex-protected input is consumed once
per backend instance by the later safe callback, not once per process.
Capture errors retain no vacancy reservation, and repeated loading never closes
another backend's descriptors. Consuming client/host intake APIs remain unsafe
and require a real exclusive transfer. This distinction adds no execution or
signer-policy authority.

Sealed-image inheritance and the Cargo binding-wrapper hooks retain their source
owners through spawn. Child hooks use descriptor syscalls and scalar error
handling without allocation or locking. Coordinator argv/environment intake
requires stable readable backing; NSS calls use matching buffer lengths, check
both status and returned-pointer identity, and copy only scalar account IDs.
Signal-mask storage is initialized and thread-affine, with restoration following
successful cleanup.

Native supervisor startup destructively closes unrelated descriptors only under
its dedicated-process contract, preserving exactly slots `3..12` and `220`.
Readiness transport uses initialized, aligned control storage and bounded payload
lengths. Received `SCM_RIGHTS` and rejected `SCM_PIDFD` descriptors acquire distinct
owners immediately and are dropped on refusal. GNU/musl ancillary fixtures zero
initialize the pinned integer-field header, including musl padding, before
bounded writes. Synthetic ancillary tests do not establish actual `SCM_PIDFD`
reception on a deployed kernel.

The protected-service spawn and supervisor cleanup bridges reserve custody
before clone and adopt the returned PID, pidfd and spawn lease before fallible
parent work. Post-clone callbacks are finite and allocation-free; credential
changes re-arm and check parent-death signaling. Exec confirmation releases only
the spawn lease. Pending or quarantined children retain dependencies; only a
consuming terminal wait permits direct-child retirement. Fresh-domain custody
additionally requires observed aggregate emptiness and successful removal of
the exact retained cgroup. `WNOWAIT` and `ECHILD` are not terminal
evidence. See [cleanup custody](compiler-execution-cleanup-custody.md).

The compiler-only pre-exec restriction module retains one unsafe function/block.
It consumes the actual child's pinned personality observation before three direct
`prctl` calls install/check the immutable 57-instruction filter and its live
native ABI header. The kernel copies the header and filter synchronously; no
allocation, lock, callback or borrowed pointer escapes the child call.

The private `native_compiler_personality.rs` module has nine blocks and six
unsafe functions. After required namespace mapping, before credential and
dumpability changes, it opens verified procfs directories and the calling child's
personality inode using bounded, no-symlink paths. The proc thread-self coordinates
must match actual getpid/gettid. The retained descriptor survives the profile
transition; a bounded read must contain exactly eight hexadecimal digits and a
newline, followed by EOF, with READ_IMPLIES_EXEC absent. Identity, filesystem,
read, format and close failures refuse; there is no personality-syscall query
fallback, supplied descriptor or parent snapshot. Each private descriptor is
consumed once, including abort paths, without retrying close. The caller excludes
foreign FD and mount mutation and retains the original child identity.

The original parent prepays `native_compiler_personality::{WORK, SCRATCH}`
through `native_work::COMPILER_RESTRICTION_WORK`, the compiler restriction
scratch constant and `StagedProtectedServiceExecV2::SPAWN_SCRATCH` before clone.
These are logical work/storage charges, not process RSS or syscall deadlines.
Acquisition refusal is stage 15, post-profile observation refusal is stage 16,
and compiler-filter installation refusal is stage 13; each precedes READY and
exec and retains the original cleanup owner. Ordinary service stages do not
install the compiler filter. This observes pre-exec state only, not personality
established by ELF exec, complete W^X, immutable backing, descendant confinement,
source/output enforcement or runtime admission.

The additional `native_compiler_trace_filter.rs` installer has one unsafe
function and one unsafe block. It synchronously installs a fixed immutable
classic-BPF program only after the original compiler gate is released; the
existing pre-READY filters remain in force. The consuming staged mode prepays
the complete installation work and scratch before clone. Seventeen syscall
checkpoints include executable-memory operations and descriptor imports;
`creat`, `openat2`, and replacement `seccomp` calls are denied. Only the same
retained parent trace may mediate those checkpoints. Staging and filter
installation alone do not establish runtime enforcement or authorize a gate
release. The gated production attempt selects this mode, but retains its closed
gate; original-trace arming and the complete runtime guard are still required.

The original-trace runtime adapter consumes the retained root owner instead of
creating another tracer or accepting a PID. Its one unsafe constructor requires
the original closed gate, exclusive same-thread waits, original account and
absolute deadline, no pre-existing descendants, a dedicated process and an
independent outside-domain custodian. Before releasing that gate the caller must
authenticate the actual typed-filter stage and complete runtime policy. Four
blocks in `native_runtime_trace.rs` initialize the fixed integer register record,
issue bounded ptrace requests to owned tasks, fail-stop an unresolved dedicated
process, and drop the original owner once after complete foreground retirement.
The original root and retained-resource bridges each add one unsafe function and
block to forward that same consuming contract. There is no public PID constructor,
second seize, policy-success token or enforcement guard.

Four unsafe functions/blocks in `native_runtime_trace_custody.rs` forward the
original late-custody and exec-confirmation contracts without exposing mutable
legacy trace access. Exact root exec observation remains required. The one
block in `native_runtime_task_observation.rs` compares actual kernel file-table
sharing with `KCMP_FILES`; unsupported or denied comparisons refuse. Scoped
observations bind the original task, held stop, census generation, account and
thread before and after bounded descriptor, maps and personality reads. Returned
files and bytes are inert, fully charged data, not mapping or compiler admission.

Every acquired child is retained before fallible birth handling. Terminal exit
and fatal-signal kinds remain distinct; descendant slots are reusable only after
a consuming terminal wait is explicitly acknowledged. Cancellation remains a
funded foreground operation until every known task has a terminal wait. A
budgetless cancellation marker performs no syscall or cleanup-success claim;
unresolved drop terminates the dedicated process rather than handing traced
descendants to root-only background cleanup. This does not replace the outside
custodian's separate whole-domain cleanup obligation.

Three test blocks exercise original takeover in bounded disposable processes.
Three standalone fixture blocks install NNP and the exact namespace/checkpoint
filters before bounded thread creation. The fixture is separately compiled and
explicitly selected, not an admitted compiler image. Component lifecycle evidence
does not qualify compiler source/output policy, runtime enforcement, deployment,
proof execution or GPU production activation.

The private compiler channel's consuming runtime transition adds one unsafe
function and block. It preserves the same channel, original backing and receive
deadline, and accepts only the original gated trace's held interrupt. Its caller
must keep the native gate closed and preserve the dedicated process and outside
custodian contract. The private trace-owner enum adds one unsafe confirmation
function and block, forwarding the existing exact-exec/alias-closure obligations
to its actual owner. A runtime event from a descendant cannot confirm root exec.
Budgetless runtime cancellation only marks refusal and retains foreground
custody; funded cancellation drives the same task tree to terminal waits before
reporting the original cleanup disposition. No default selection or compiler
resume permission follows from these transitions.

Publication forwarding adds four unsafe function sites (two private declarations
and two macro definitions) and two blocks. Only the original retained trace and
its consuming runtime successor implement the private trait. Both use the same
concrete artifact quote, install-before-acquire sequence, Acquire/Validate
builders and original cleanup slot. No arbitrary observation provider, payload
owner or mutable root-only trace is exposed. Observation quotes include the
runtime view's additional entry charge. One additional isolated channel-test
block exercises closed-gate takeover and funded cancellation; it never executes
a compiler and does not qualify runtime admission.

Native image-policy preparation uses those same held-task views for bounded
read-only auxv and memory copies; no proc descriptor escapes. Kernel-image
metadata is captured only at an actual exec stop from AT_SYSINFO_EHDR, the
complete RX mapping, its mapped ELF ranges and a SHA256 digest. Later policy
checks compare actual layout and bytes. The native maps predicate does not
exempt mappings by `[vdso]` or `[vsyscall]` labels; unsupported legacy vsyscall
layouts refuse. Original compiler range visits check the actual retained FD
identities and original account before and after, while full source revalidation
remains mandatory at ownership transitions. Continuous immutable-backing and
write-exclusion obligations remain with the owning controller. These private
policy results are not enforcement tokens or permission to release the gate.

Native descriptor checkpoint preparation likewise returns only bounded data.
Open results are inspected through the original task view; pre-open pathname,
creation, truncation and device effects remain a separate controller obligation.
Credential-only receives validate all payload/header/control non-overlap and
reject rights, unknown controls and truncation before any task resumes. A late
ancillary-copy error is not treated as proof that no descriptor was installed.
The selected IPC role audit and remaining qualification are documented in
`docs/compiler/native-descriptor-boundary-v161.md`. These additions introduce no
new unsafe blocks or independent task/descriptor acquisition authority.

Twelve diagnostic blocks in `native_compiler_restrictions_exec_tests.rs`
stage/clone retained inert inputs, query/set/restore isolated creator state,
inspect initialized scalar buffers and install outer test-only denial filters.
Restorable controls arm their guard before mutation; dirty locked controls
terminate their disposable process without clearing the state. Exact-test
subprocesses isolate irreversible filters and dumpability changes. The controls
distinguish acquisition, profile, record/EOF and installation failures; neither
an earlier refusal nor a filter supplied by the host qualifies the intended
boundary. Each native selector requires its stated isolated deployment and
static `-pthread` fixture. Execution evidence remains separate from this review.

The unsafe sites in eleven reconciled inventory entries are inherited from
public `8ecfb2a5e3af052fbd36034c73b51ab0fca7804f`. The review also repaired the
root-request subprocess test to prepay `CreatorScope::CONTROL_WORK` before
scope entry, as the unsafe API requires. The production entrypoint already
prepays this control allowance before constructing its creator. In the
coordinator, the reviewed entries are
`native_compiler_attempt.rs`, `native_compiler_attempt_tests.rs`,
`native_entrypoint.rs`, `native_root_request.rs`,
`native_root_request_preexec_process_tests.rs` and
`native_root_request_process_tests.rs`. Original request ownership is installed
before fallible preparation; helper/child backing enters the original funded
cleanup slot before clone. Original work lifetimes, received objects, exclusive
consuming waits and outside whole-domain custody remain required. The compiler
exec gate stays closed. The attempt test's unsafe-function site is a signature
assertion, not an unsafe call.

The other five entries, in protected-service spawn, are
`native_compiler_personality.rs`, `native_compiler_restrictions_exec_tests.rs`,
`native_compiler_spawn_tests.rs`, `native_namespace_restrictions.rs` and
`native_cgroup_mount_tests.rs`. The namespace filter uses one unsafe function
and block with a synchronous immutable native ABI program; compiler and actually
mapped children install it before READY, with stage-14 failure, while unmapped
service creators retain their existing behavior. Inert staging tests launch no
child. Mount controls use live C strings and change only a disposable process's
unshared, private-propagation mount namespace. Counts add 32 blocks and ten
function syntax sites; no production behavior or test gate changes. This static
review does not execute or qualify the ignored native controls, establish
deployment provenance, or grant compiler, proof or GPU authority.

The optional native fresh-domain spawn extends that same clone path with
`CLONE_INTO_CGROUP`. It reserves rollback custody before mkdir and transfers the
actual domain, atomic pidfd, spawn lease and original cleanup slot before any
fallible parent check. Its unsafe caller must exclude privileged cgroup/mount
writers and must not export cgroup controls or allow child migration/delegation.
No caller path or descriptor constructs a domain owner. Root numeric credentials
and cgroup placement alone establish neither trusted deployment origin nor
proof-process isolation. The child remains nondumpable; production proof gates
are unchanged. One additional unsafe mechanical entry point and two ignored
root-diagnostic call sites are reviewed here, not qualified by their counts.

Fixed-slot, credential and process-transition fixtures run in isolated children.
Their source descriptors stay above the destination slots; raw adoption happens
once after successful creation. Polling failures and timeouts must kill and reap
the test child. Descriptor-free cleanup fixtures exercise only the documented
inert exception. The shared issuer launch-input tests retain the same obligations
after their V2-specific filename is removed. The compiler image-budget fixture
uses safe `rustix` memfd creation and needs no raw-descriptor adoption.

Inventory reconciliation records moved implementation sites, shrinking the old
sealed-image, client, provisioner and service allowances alongside their new
modules. Macro/include sites are counted once where written. This review covers
OS ownership and unsafe-call contracts, not protected execution, proof-policy
admission, native production activation, or GPU qualification.

## Retained Proof Controller

The gated proof launcher performs raw fork only after preparing every descriptor,
C string and argument pointer. Its child performs syscall-only setup and exec;
it must not allocate, unwind, acquire inherited locks or run Rust destructors.
Successful pipe creation transfers each descriptor into exactly one owner.
Current-thread observations use argument-free `gettid`; retained task identifiers
are not authority to act after their terminal wait has been consumed.

Stable inspection initializes register storage before ptrace writes it and
checks status before interpreting the result. Task custody is published before
fallible parent work; unresolved creation, cleanup or unwind remains retained
in the original process. This does not establish external-writer isolation.
The artifact spawn lease must separately survive the entire pre-exec interval,
including failure quarantine: a safe pointer/descriptor inventory cannot replace
that lifecycle obligation.

Socket, mapping, stable-controller and quarantine fixtures run their raw process
operations in owned diagnostic domains. Ancillary buffers are initialized and
bounded. The memory-policy fixture owns one private, writable, non-executable
mapping, mutates one valid byte, and unmaps it once; it never executes that data.
Fixture kill/reap and raw syscall operations provide test observations, not
production launch authority. The reviewed eight-file inventory contains 53
unsafe blocks and eight extern blocks, including 29 test-only blocks. Its old
controller allowance shrinks from 17 blocks to 15 as operations move into the
private spawn and inspection modules. Counts do not qualify protected execution.

## Native Compiler Exec and Root Trace

The compiler mode in `fe2o3-protected-service-spawn` uses the existing raw
clone/profile/gate/exec path. One unsafe staging function requires authenticated
input custody and full overlapping charges. Owned CString buffers and frozen
pointer tables survive source drops and stage moves. The existing child syscall
block adds `fchdir` and exact argv/environment selection; no second launcher or
unsafe Send implementation is introduced. Standard destinations preserve shared
open-file status and offsets; `Some` clears descriptor CLOEXEC. The caller must
map captured absent or original-CLOEXEC streams to `None` for post-exec absence.

Root tracing adds one scalar ptrace block and one unsafe exec-confirmation
function/block. The retained wrapper adds one forwarding function/block. The
controller consumes the existing child, pidfd, cleanup slot, artifact lease and
backing; it binds operations to the original thread, Budget address and ledger.
Its fixed TRACEEXEC/EXITKILL options introduce no descendant or TRACEEXIT stops.
Only the same owner's consuming terminal wait notifies cleanup. Exec observation
alone does not release the lease: confirmation additionally requires the caller
to authenticate native exec and closure of every inherited artifact-lock alias.
It does not establish process-tree isolation or compiler authority.

`fe2o3-process-identity/native_capture.rs` uses three blocks for scalar
standard-slot probes, CLOEXEC duplication above FD2, and adoption of newly owned
descriptors. Capture is safe, inert and non-atomic: concurrent slot replacement
can mix observations, while shared flags, offsets and contents remain mutable.
It neither creates a borrowed lifetime for a raw standard slot nor grants
invocation provenance. The authenticated native stage's obligations are separate.
Cwd access borrows the existing pin without reopening
its pathname. Inode/flag comparison is not evidence of OFD provenance.

Reviewed test-only additions are 12 capture blocks, two compiler staging blocks,
two native-exec blocks, and nine trace-confirmation blocks. Isolated subprocess
tests cover absent/CLOEXEC streams, offsets, partial capture failure, actual
trace waits, short funding and retained cleanup. The explicit root diagnostic
uses a static C fixture through the production clone path, with no compiler,
proof or GPU authority. The C fixture checks inputs and descriptors and exits;
its libc boundary remains part of the diagnostic, outside the Rust inventory.
No unrelated inventory allowances change.

## Wrapper Input Custody

The cwd hook now owns its CLOEXEC duplicate through Command's lifetime, including
reuse and failure, instead of capturing a borrowed raw descriptor. The existing
unsafe block still performs only pre-exec fstat/fchdir; registration can refuse
before changing Command. Six test-only blocks cover descriptor probes and an
isolated soft-limit guard.

The binding wrapper captures actual runtime-sanitized entry stdio before opening
files, and retains it with protected parent invocation custody. Two installation
blocks perform raw standard-slot probes and register a closure owning high-FD
duplicates. Only bounded descriptor operations run before exec. Absent or
original-CLOEXEC slots close; other streams preserve shared offsets/status flags.
The wrapper keeps current parent slots open so pinning and Command's error pipe
cannot occupy a slot the final hook replaces. Capture and sampled revalidation
do not prove immutable aliases, pre-runtime inheritance or compiler authority.

The child-channel transfer now requires kernel message credentials matching both
the declared child PID and the service socket's creation credentials. No new
production unsafe block is needed. Its relay test registers one existing scalar
send helper in a child-only pre-exec hook. The original same-UID/GID relationship,
pidfd liveness and protected-runtime refusals remain unchanged.

## Compiler Trace And Key Transfer

The private compiler-channel trace adds one unsafe confirmation function and one
forwarding block. It preserves the existing original-thread/account trace and
requires the caller to establish exact native exec-status EOF and closure of all
inherited artifact-lock aliases, including any in untraced descendants. Its
diagnostic adds one call block while the non-forking native stage's first exec
is held. Confirmation releases only the spawn lease, not compiler authority.

The fresh service-owned V3 key transfer adds no production unsafe site. Its
isolated receiver test adds two scalar libc blocks to drop real/effective/saved
GID then UID before exercising the unchanged receiver admission. The test requires
an exact single-test invocation in a disposable process with CHOWN/SETGID/SETUID;
it is not a shared-process credential API. The root template remains unchanged,
and a separate read-only sealed image is assigned the deployment's credentials.

## Original Request And Mixed Handoff Review

The 2026-10-01 static review reconciles fifteen specifically reviewed inventory
entries, accounting for fifty-one constructs omitted from the previous baseline.
It changes neither the inventory gate nor the reviewed source implementations.
The compiler qualification worker reviewed all fifteen files; the integrating
reviewer also examined the coordinator call boundaries, personality observation,
namespace filter and mixed host adapter. No blocking safety defect was identified
under the existing caller contracts. Unreviewed paths receive no new allowance.

The original-request coordinator retains the complete receiver and helper backing
before fallible launch continuations, uses the original independently funded
cleanup pool, and keeps the compiler exec gate closed. Dedicated creator-thread,
exclusive-wait, mutation-exclusion and outside whole-domain custodian obligations
remain with its unsafe caller. Its tests either check an unsafe function signature
without calling it or operate through isolated re-exec roles with retained child
and descriptor ownership.

The child personality reader uses initialized, bounded syscall buffers and
privately owned procfs descriptors. It binds the same calling task across the
credential transition, requires an exact record and EOF, and closes each descriptor
once on both success and failure. The namespace filter passes a fixed native BPF
program through a live header after child setup; installation failure is terminal.
Neither reader nor filter authenticates deployment provenance, cgroup exclusion,
or post-exec compiler behavior. Native test mutations occur only in disposable
processes with the existing cleanup and restoration contracts. Inert staging tests
do not create children or grant compiler roles.

The mixed application wrappers preserve the existing one-shot inherited-FD and
single-threaded startup contract. Their safe continuations prepare a checked
roster and audit its current record; they grant no native proof, loading or GPU
dispatch authority. Three private test trait implementations exercise typed
preparation and malformed-layout refusal without constructing an executable
receiver. The extra Wave64 unsafe call and lookalike signature are deliberate
source-rejection fixtures, counted once in the macro template.

This review is not native execution or formal verification. The isolated inventory
gate must be rerun after integration; protected runtime, installed deployment,
compiler gate release, GPU execution and end-to-end production qualification remain
separate obligations.

## Original Request Routing

The added block in `native_root_request.rs` forwards its existing unsafe
dedicated-creator contract to the private execution state machine. The one
unsafe function and three blocks in `native_root_request_runtime.rs` delegate
only original runtime takeover, confirmed first-exec status and consuming issuer
launch. Each call uses the Attempt already owned by that request; the issuer
receives its exact original Prepared and cleanup pool. These sites introduce no
raw syscall, alternate owner or authority constructed from wire data.

The request keeps its original work ledger, budget address, captured input
owners and absolute deadline through all transitions. Errors leave the request
failed and retain the original foreground cancellation obligation. Completion
encoding follows both actual root terminal wait and trace retirement; it grants
no publication or execution authority. Startup funding, downstream admission,
live native qualification and default activation remain separate open gates.
The inventory changes record only these reviewed sites and do not refresh the
unrelated pre-existing GPU/memory inventory differences.

## Original Root Publication Retirement

The private unsafe function and forwarding block in
`compiler_execution_root_publication.rs` prepare retirement of the actual
publication holder in its original runtime cleanup slot. The sole caller block
in `compiler_execution_root_publication_service.rs` first authenticates the
original measured issuer, connection binding and runtime identity. That issuer
sends Retire only after its closed durable Worker/anchor/Ready/ACK join. The
root then matches the complete canonical policy, Subject and occurrence against
its still-locked publication. Decoding a carriage alone cannot reach retirement.

The prepared token holds custody on Drop. Complete reply/tombstone preparation
and the root service's accounting scope precede its infallible commit. The
session keeps the full carriage independently of the replaceable connection;
reply sends and exact retirement replays never reacquire locks. Partial
acquisition, refusal and unwind retain the original funded cleanup obligation.
Production pumps this channel before runtime steps and requires the same
retirement tombstone at held exit, rather than allocating a second late holder.
Native end-to-end qualification and default activation remain separate gates.

## Initial Reduction

The three unsafe functions and three blocks in
`native_compiler_attempt_runtime.rs` delegate consuming takeover, exec
confirmation, and issuer launch to the existing original-trace APIs. The private
Attempt retains the same inventory, controller, helper, deadline, creator and
cleanup obligations. Exec confirmation closes the full Stage and both gate
aliases before observing native status EOF. No application instruction is resumed
before first-exec policy validation and actual issuer readiness. Consuming failure
with unresolved runtime custody requires the established dedicated-process
fail-stop; it cannot silently defer descendant waits to root-only cleanup. These
sites add no raw syscall or PID-based authority constructor. The production gate
still refuses; live native qualification remains separate.

The initial audit of `d9f6bbcd0` found 1,924 source sites in 288 Rust files:
1,026 in non-test crate source across all configurations, 874 in tests/fixtures,
and 24 in examples/benchmarks. This includes legacy qualification code, not just
the default production build. The checked-in baseline records the current
post-reduction counts without requiring a GPU to reproduce them.

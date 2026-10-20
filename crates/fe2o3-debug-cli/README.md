# fe2o3-debug

`fe2o3-debug sim --kir-v7 KERNEL.kir --request REQUEST.json` opens a bounded
JSONL debugger over an exact deterministic CPU simulation transcript. Requests
arrive on standard input and one `fe2o3-debug-response-v1` line is written for
each request. `--protocol jsonl` is accepted explicitly and is the only V1
transport.

On Linux, `--kir-v7-fd FD --request-fd FD` is a paired alternative used for
already captured inputs. Both descriptors must resolve to distinct underlying
anonymous mode-0400 regular-file memfd objects, be read-only and bounded, and
carry the exact immutable seals. Distinct descriptor numbers duplicated from
one `(device, inode)` object are rejected. The debugger marks the inherited
descriptors close-on-exec before duplicating and reading them, then uses the
same KIR/request admission and identity path as the file form. Descriptor
numbers are transport state, not debug evidence, and JSONL remains on standard
input/output.

## Raw diagnostic ordered-region V16

```sh
fe2o3-debug sim --diagnostic-kir-v16 KERNEL.kir --request REQUEST.json \
  --wave-width 64 --protocol jsonl
```

This explicit path uses the simulator CLI's shared hardened V16 loader, retains
the exact immutable verified module, and captures bounded logical CPU execution.
It never projects V16 to V7 or admits a simulation bundle. Its configuration
identity binds the exact module, request, CPU target, wave width and limits;
existing input-mode identity domains remain unchanged. The closed ordered-region
profile requires gfx942:xnack-, wave64 and a `64x1x1` workgroup. Wave32, source-map
overrides, persisted schedule replay and mixed input selectors are refused.
There is no new V16 file-descriptor input mode or persisted schedule format.

The [source walkthrough](../../../docs/ordered-region-authoring-v1.md#use-the-ordinary-diagnostic-tools)
provides the final normal library/binary/example build, real Rust export and
complete strict request. Use `inspect_diagnostic_ordered_region_v16 KIR_PATH
REQUEST_PATH` from the built examples directory to obtain current-owner canonical
identity, function/block/operation roster ordinals and input/result SSA IDs.
Its separate `raw_block_id` is not the protocol block ordinal. The example
preflights but does not execute; its declared source IDs authenticate no source
and supply no map. Re-inspect after every export rather than copying historical
fixture coordinates. The [debugger walkthrough](../../../docs/ordered-region-debugger-v1.md)
explains current revision/anchor selection and lossless wave64 JSON integers.

Forward/reverse navigation, logical SSA values and allocation-relative memory
and resource queries remain available. The entire XOR/ADD region is one atomic
logical operation, including when its result is unused. There is no observation
between its two instructions. Authored register numbers describe a plan, not
physical VGPR/SGPR/AGPR, EXEC or scratch contents, register lifetimes or a final
artifact mapping. Source-variable/span features remain typed unavailable.

`fe2o3-debug-diagnosis-request-v2` returns `unsupported_schema` for this route:
its frozen required input evidence describes canonical V7 and cannot honestly
represent raw V16. Separate resource-query V1 remains available. Raw bytes confer
no compiler source-owner custody, source authentication, proof, artifact, launch
or production-resume authority. Ordinary debugger qualification passed 30 sessions
and 1,020 commands, with selected-lane logical navigation/value checks and all
64 lanes' output memory/write history checked. See the [exact evidence scope](../../../docs/ordered-region-authoring-v1.md#evidence-scope-and-further-work)
for receipts, retained failed client attempts and exclusions. The private
same-owner tests remain distinct from this ordinary CLI qualification.

## Existing bundle and source-map routes

`fe2o3-debug sim --bundle KERNEL.fe2sim --request REQUEST.json` securely
decodes and revalidates the authority-free compiler simulation bundle, then
uses only its exact embedded KIR V7 and target. `--bundle` and `--kir-v7` are
mutually exclusive. An embedded debug map is admitted from those same captured
bundle bytes and labeled `compiler_bundle_bound`; callers cannot override it
with `--source-map`. This label proves exact bundle content association only,
not compiler execution, source authorship, hardware observation, or authority.
When a bundle has no map, source features remain typed `unavailable`.
Compiler-exported bundles normally contain a map derived from the same rustc
and semantic-KIR transaction. V1 exposes resolved call-site spans only;
synthetic KIR operations, source variables, and macro expansion stacks remain
typed unavailable rather than being inferred.

`fe2o3-debug sim --bundle-v2 KERNEL.fe2sim --request REQUEST.json` is a
separate, strict route for `VerifiedSimulationBundleV2`. The V2 envelope
contains one exact authority-free V1 execution bundle with no nested V1 map,
plus one canonical `fe2o3-debug-source-map-v2` payload. V1 decoders continue
to reject these bytes. V2 map records can name bounded lexical scopes, stable
variable identities, storage generations, and exact KIR SSA values at exact
block checkpoints. The production compiler exporter emits V1 by default and
can opt into this route with `--bundle-version 2`.
That route captures rustc lexical scopes in the same extraction session and
binds only exact one-to-one admitted kernel parameters whose entry value is
never moved, dropped, storage-reset, mutated, or mutably aliased in MIR to KIR
function values.
Other locals, projected/constant debug values, and non-one-to-one composite ABI
cases remain typed `Unrepresented`; no SSA lifetime is inferred. V1 remains
the byte-compatible default. Neither route authenticates compiler execution,
source refinement, or hardware behavior. The production regression is
`ordinary_kernel_sources_export_and_query_exact_v2_source_variables`.

`fe2o3-export-sim --bundle-version 3` adds independently bound exact
production semantic-MIR bytes and a canonical semantic-storage map. The map
joins source argument/local/type identities and ownership to exact KIR
parameter/value identities, or records a typed `unavailable`, `ambiguous`, or
`opaque_flattened` state. `fe2o3-debug sim --bundle-v3 ...` runs the unchanged
deterministic debugger against its nested V2 execution/source bundle.

`fe2o3-debug typed-layout --bundle-v3 KERNEL.fe2sim --request REQUEST.json`
emits one bounded JSON object for agents and tools. It re-decodes current
production semantic MIR, cross-checks every referenced root, body, local,
source type, KIR function, parameter, and value, then reports rustc sizes,
alignment, field source/memory order, explicit padding, direct/niche enum
encoding, and variant layouts. Request arguments additionally report exact
scalar bits or allocation-relative regions, byte initialization ranges,
alignment, access, request-local provenance, and exact byte-range overlap for
shared-backing arguments.
Substituted layout, local, KIR, or source-map bindings fail closed. The query is
observational and grants no compiler or execution authority.

Bundle V4 retains the complete V3 payload byte-for-byte and adds a separately
content-bound one-to-many component map with explicit physical kernarg size,
alignment, and slots. The production V4 exporter derives that map from the
sole rustc-to-KIR lowering correspondence. `fe2o3-debug sim --bundle-v4 ...`
runs the nested canonical KIR with the flattened compiler parameter order; no
GPU, KFD, or physical-host ABI authority is implied. Use
`fe2o3-debug typed-layout --bundle-v4 KERNEL.fe2sim --request REQUEST.json` to
associate each observed KIR argument with its source argument, semantic type,
and nested struct, tuple, array, or enum projection path. This emits
`fe2o3-debug-typed-layout-v2`; V3 continues to emit the unchanged V1 response.

KIR V7 still has no by-value aggregate value type. V4 reconstructs components
only when the compiler supplies the exact semantic projection for every
retained scalar KIR parameter; it never infers flattening from scalar names.
Aggregate construction/execution outside the current production lowering
remains a typed compiler rejection, and storage without exact correspondence
remains typed unavailable.

`fe2o3-export-sim --bundle-version 5` emits a separate self-contained Bundle
V5. It binds the original production KIR V8 or V9 identity and an exact
same-module KIR V10 encoding together with Source Map V2, semantic MIR, and
both storage maps. `fe2o3-debug sim --bundle-v5 ...` executes that V10 custody
directly, so compiler-produced gfx950 f32 wave collectives can use the public
CPU debugger without a lossy V7 projection. The route grants no compiler,
artifact, load, launch, or hardware authority and never falls back to a GPU.
Production source lowering for V10-only memory intrinsics remains unavailable;
the exporter reports that boundary rather than synthesizing such a kernel.

Source-variable inspection uses the separate
`fe2o3-debug-source-variable-request-v2` schema. Callers select all variables,
one exact stable identity, or a bounded inert name. Name lookup chooses the
deepest active lexical scope and returns typed ambiguity when exact records do
not distinguish one binding. Values are existing typed simulator observations,
including allocation-relative pointers; native addresses and reconstructed
registers are never emitted. Missing V2 data, an empty V2 variable section,
out-of-scope names, unavailable frames/checkpoints, uninitialized generations,
lifetime kills, optimized-out values, unrepresented values, and truncated
captures remain distinct typed states. `all` queries use admitted per-function
indices and apply page bounds before value materialization.

On schedule-supported input routes (not diagnostic V16),
`--replay-schedule SCHEDULE.json` securely admits the canonical persisted
semantic schedule and requires its exact raw-KIR/bundle route, KIR, complete
bundle identity and subject, request bytes, target, and limits to match this
debugger input before capture. The debugger configuration identity additionally
binds the schedule context, transcript, and exact decision-record integrity.
Protocol revisions still govern interactive navigation of the resulting
immutable transcript; they are not simulator schedule revisions. This is
deterministic CPU semantic replay, not GPU scheduling, timing, race-freedom,
equivalence, or performance evidence.

The simulator exposes work-item, logical wave, workgroup, KIR operation, SSA,
allocation-relative ordinary and integer-atomic memory, fence order points,
barrier, and committed-memory observations. Atomic watchpoints match one
indivisible atomic record; read/write watchpoints include atomic records with
the corresponding effect. Reverse navigation is deterministic transcript
replay; forward stepping includes
frame-aware over/out. It is not GPU reverse execution;
logical waves are visualization partitions, not hardware wave observations.
Captured F16/BF16/F32/F64 scalar values retain their exact software-IEEE bits;
the debugger does not render or recompute them with host floating-point
arithmetic.

For input routes represented by its frozen evidence schema (not raw diagnostic
V16), `fe2o3-debug-diagnosis-request-v2` adds a read-only, page-bounded diagnosis
query to the same simulator JSONL stream without changing the closed V1
operation set. For a retained out-of-bounds failure it returns the exact
trace-local backing allocation, requested range, and narrower legal pointer or
slice view plus the admitted allocation and uniquely joined kernel-ABI
argument contract. Aliased views retain both their common backing identity and
their distinct ABI ordinal/range, so an access can be outside one view while
remaining inside the backing allocation. ABI-required access, request
buffer/view access, and backing-allocation access remain distinct and are
joined with the simulator's monotonic capability rule. For retained
workgroup-barrier divergence it returns the exact phase, declared scope,
ordering and address-space semantics, a phase-derived LDS epoch, complete
arrived/waiting/exited local participant inventories, the observed arrival
count when the transcript is complete, and the expected participant inventory
derived from admitted launch geometry. An LDS epoch after a barrier that did
not release remains unavailable.

Every diagnosis also binds the stable simulator configuration, a
domain-separated identity of the exact request and canonical KIR dispatch
input, and the exact KIR operation when retained. A Source Map V2 can add the
exact source operation and span with its existing caller- or bundle-bound
provenance. Verified bundle envelopes add exact envelope, inner subject,
production-KIR, ABI, and source-lineage receipt references. Those references
are content association only: the debugger does not relabel lineage receipts
as property proofs, and finalized artifact/property fields remain typed
unavailable without independent authority.

Each material fact carries a claim identity tied to a retained input,
simulator-terminal, simulator-transcript, KIR-operation, ABI, source-map,
derived, or availability record identity. The evidence manifest also binds
session revision, configuration, completeness, finding sequence/class, exact
admitted input, and canonical bounded terminal/transcript content. Hashes and
wire decoding prove content integrity, not producer identity. Before returning
a production Bundle V2 diagnosis, the CLI independently rederives the expected
capture binding from its owned deterministic simulator result and exact Bundle
envelope/subject, full input manifest, session/completeness, and response
wrapper (request, operation, and pagination cursor), then exact-compares it
with the response. Source spans include a bounded membership proof against the
exact admitted Source Map V2 operation inventory; changing a span to another
range in the same map fails validation.

Dispatch geometry is `declared`; terminal invocation, KIR site, dynamic range,
phase, and arrived/waiting/exited local participants are CPU-semantic
`observed` facts; expected locals, global participant coordinates, logical
wave/lane partitions, logical element indices, legal-bounds results, and the
current LDS epoch are `inferred` with their derivation; absent facts remain
typed `unavailable`. This diagnoses any
admitted KIR kernel that reaches those simulator error classes and contains no
kernel-name or fixture-specific rule.

The diagnosis schema validates only CPU-simulator sessions. Live KFD and
ROCgdb keep their own separately versioned capability contracts: stopped wave,
lane, register, PC, source, and target-memory facts remain unsupported or not
captured there. A simulator diagnosis may be correlated by an agent as
reference evidence, but it is never upgraded to native hardware state or used
as evidence that the GPU executed the same path.

Barrier residency is replayed from semantic records. A lane is
`barrier_blocked` from its arrival through the record before the matching
workgroup release. A wave or workgroup is `barrier_blocked` only when every
active lane in that aggregate is waiting; a partial wait is `runnable`.
Dispatch aggregation follows the currently scheduled workgroup. The release
record clears residency before scope state is reported, so its representative
lane is `running` and other released lanes are `runnable` until their next
record.

On supported raw-map routes (not diagnostic V16),
`--source-map MAP --source-bundle-subject SUBJECT` admits a strict, bounded
canonical `fe2o3-debug-source-map-v1` or `fe2o3-debug-source-map-v2` sidecar.
Both options are required together. The map binds the canonical KIR digest and length plus a non-circular
compiler-bundle subject identity. The wire map binds these compile-time facts
only. Admission derives the complete runtime simulation configuration from the
admitted request and wave width, then the backend rechecks that internal
binding before use. Bundle sessions additionally bind the verified bundle
subject, which commits the exact target. A reusable compiler map must not claim
a future request or wave configuration.
Files are content identities and display paths only; the debugger never opens
a source path from the map. KIR/source resolution, source breakpoints, and
source stepping return distinct absent, eliminated, and many-to-one states.

This command-line pair is a low-level/test consistency boundary. Because the
caller supplies both documents, it is labeled `caller_bound`, not compiler
provenance. A loose V2 map can resolve diagnosis source operations and V2
source-variable records, but its provenance is still caller-bound. Production compiler integration must call
`run_admitted_jsonl_with_compiler_source_map_v1` with exact map bytes, the
verified bundle subject, and the bundle-committed map identity obtained from the
same compiler extraction/bundle decode transaction. Only that path emits
`compiler_bundle_bound`. This proves exact bundle content association, not
protected compiler execution. The bundle subject excludes the map payload
to avoid circular identity; the enclosing bundle identity commits the map's
domain-separated identity and exact length. `debug_source_map_identity_v1`
implements the bundle's exact map-identity algorithm.

`inspect_stack` pages captured simulator call frames, including function/block
ordinals, the next operation, and typed captured/unavailable value state.
Frames are not reconstructed from names or UI fixtures. Source variables are
available only through the exact V2 bundle route above; V1 and hardware V2
sessions keep them typed unavailable. Hardware registers, hardware wave state,
and KFD dispatch control remain typed `unavailable`; no value is fabricated.
KIR, request, and sidecar files use the hardened regular-file capture boundary
shared with `fe2o3-kir-sim`.

## Fresh reference client

`fe2o3-agent-reference-client --workflow WORKFLOW.json` is a deterministic,
LLM-independent acceptance client. It opens each bounded regular evidence file
once with no-follow semantics, rejects symlinks and hard links, and revalidates
the same descriptor's device, inode, type/mode, size, and modification time,
requires its link count not to increase, and exactly re-reads its content after
the bounded read. A link-count decrease and change-time update caused by rename
or unlink are accepted for legacy descriptor custody; the already opened object
and exact bytes remain authoritative. Archive custody additionally requires
unchanged link and change-time metadata plus persistent path identity. The
client copies simulator KIR/request bytes into distinct read-only sealed
memfds, passes only their explicit descriptor numbers to `fe2o3-debug`, and
then communicates with the debugger and profiler services only through
documented JSONL stdin/stdout. The launch-time workflow names trusted installed
debugger/profiler executables and hostile evidence paths. Each executable is
also opened once with no-follow/close-on-exec semantics, required to be a
bounded singly-linked executable regular file, hashed from that descriptor,
and retained for every launch through `/proc/self/fd/N`. Descriptor metadata
and content are revalidated around each child, and the exact executable byte
identities are included in the report. A later path replacement therefore
cannot select the producer of accepted evidence. Evidence paths never enter
the protocol or final report. Simulator results must name the exact preloaded
request and canonical KIR identities. No named simulator snapshot or temporary
directory participates in the workflow.

The production archive route removes the source-checkout and loose evidence
path requirements:

```text
fe2o3-agent-reference-client \
  --archive evidence.fe2archive \
  --archive-sha256 EXPECTED_LOWER_HEX_SHA256 \
  --debugger ./fe2o3-debug \
  --profiler-service ./fe2o3-agent-profiler-service
```

`fe2o3_debug_cli::reference_archive_v1` encodes and admits this fixed-role
canonical archive. The client securely reads the singly linked archive and
requires its path to resolve to the same admitted object after the read,
requires the caller-pinned digest of its exact bytes, verifies every member
digest and the canonical complete member set, and then supplies member bytes
directly to the existing debugger, Bundle V4, Variant V1 and V2, diagnosis V2,
and Agent Profiler V1 admissions. Members are never extracted or interpreted as
filesystem paths. The archive route therefore has no member symlink, hardlink,
or traversal surface. The archive report preserves the complete existing V1
workflow report and adds the archive plus ordered member content identities.
Before launching, it streams each admitted debugger/profiler executable into
an executable sealed memfd, verifies the copied byte identity, and executes
only the immutable descriptor image. Archive-mode children inherit an empty
environment: loader, locale, path, temporary-directory, ROCm, sanitizer, and
project variables cannot redirect the selected executable or its inputs. The
legacy workflow retains its original executable-descriptor and environment
behavior.
The SHA-256 pin establishes exact caller-selected content for this invocation;
it is not a signature or producer identity.

Every child session has one compiled deadline. Dedicated bounded readers drain
stdout and stderr concurrently, retain at most the documented limit plus one
overflow byte, and reject oversized, unterminated, unsolicited, or trailing
output. The client observes a successful leader exit without reaping it, makes
one process-group termination attempt while the PID/PGID remains pinned, then
always boundedly reaps the direct child and checks group absence. Signal
authority is permanently revoked before that attempt, including failure paths,
so cleanup never signals a numeric PGID again after the leader was reaped.

One workflow discovers capabilities and then performs four read-only tasks:

- diagnose a retained simulator memory out-of-bounds failure;
- diagnose a retained simulator workgroup-barrier divergence;
- compare a seeded schedule/resource Variant V1 regression;
- independently compare the same exact treatments through Variant V2, binding
  both schedules and preserving absent PC/ATT sessions, the profiler-KIR bridge,
  and causal attribution as typed unavailable facts; and
- page the exact dispatch set and ask Agent Profiler V1 for the minimum capture
  that distinguishes scheduling delay from resource pressure.

The client decodes simulator diagnoses with the full evidence-manifest
validator, validates both serialized Variant response identities, and requires
the capability, continued, and diagnosis sessions to carry one configuration
identity, with the diagnosis session exactly equal to the continued session.
Treatment files consume one decreasing aggregate admission budget before any
read past that budget, and exact byte inputs are emitted with fallible bounded
streaming hex encoding. The returned typed Variant V1 comparison is independently
recomputed and decoded with the production exact-input comparator. The V2
comparison is independently reproduced from the same retained bytes and must
equal the complete service result, including its request, artifact and schedule
bindings, evidence IDs, and unavailable facts. Its bounded
report retains each full
authenticated diagnosis with every material citation identity and the capture
plan's exact Agent V1 evidence/origins. Agent Profiler V1 responses must match
the issued schema, request ID, response revision, status, and result kind.
Dispatch pagination additionally requires exact capture binding, a progressing
content-bound cursor, distinct dispatch identities, and second-page exhaustion.
The client independently opens the retained bundle with
`ProfilerQuerySessionV4` and requires full context, cursor, dispatch order, and
evidence arrays to equal the two locally derived pages.
The report otherwise contains only inert
content identities, truth classifications, cited claims, typed unavailable
states, and pagination counts. It has no launch, attach,
pause, scheduling, KFD, ROCgdb-control, rocprofv3-collection, or recapture
operation. `fe2o3-agent-profiler-service` is a small companion executable that
exposes the unchanged Agent Profiler V1 JSONL mode and separate Variant V1 and
V2 modes for this process-isolated workflow; it is not an MCP adapter. Neither
Variant mode accepts paths or grants execution, replay, file, network, patch,
decoder, attach, scheduling, collection, or launch authority.

## KFD hardware protocol V2

`fe2o3-debug hardware -- PROGRAM [ARG...]` launches that exact argument vector
as a ptraced child and exposes a separate bounded
`fe2o3-hardware-debug-request-v2` JSONL protocol on standard input. The
coordinator retains a pidfd for the directly launched process leader, owns the
KFD debug-trap session on its spawning task, installs leader-only
`PTRACE_O_EXITKILL`, and on EOF, protocol/output failure, or `terminate`
finishes KFD state before sending `SIGKILL` to and boundedly reaping that
direct child. It does not create or contain a process group or session, adopt
descendants, or claim descendant cleanup. It invokes no shell, HIP, or HSA.

V2 provides redacted device and queue snapshots, bounded exception events,
and queue suspend/resume after the target enables its KFD debug runtime.
Runtime transition events are recorded and acknowledged internally so the
target-side KFD handshake can proceed. Device and queue identifiers are
session-local generation/ordinal pairs; the wire never contains a PID,
descriptor, native KFD identifier, target virtual address, or target argv.
Control revision and asynchronous observation sequence are independent.

The hardware protocol does not provide wave or lane state, register or CWSR
decode, stack/source/KIR sites, stepping, replay, breakpoints, values, target
memory, semantic trace, address watch, or dispatch submission. These
capabilities are explicitly reported unavailable and are never inferred from
the CPU simulator. The hardware path observes KFD state only; it does not
claim timing, performance prediction, race freedom, or GPU scheduling detail.

## Exact-bound live KFD protocol V3

`fe2o3-debug live-kfd --bundle-v2 KERNEL.fe2sim --request REQUEST.json
--hsaco KERNEL.hsaco -- PROGRAM [ARG...]` composes the deterministic CPU
reference inputs with the direct-KFD lifecycle without conflating their truth
classes. Admission captures and revalidates four distinct regular files,
rejects symlinks, hard links, role aliases, changing inputs, invalid HSACO, and
an unexecutable host. The coordinator launches through the retained executable
descriptor and upgrades that content binding only after its owned exec stop.
It never claims that declared HSACO bytes were loaded or executed. Its process
ownership is likewise limited to the directly launched leader; V3 does not
contain or clean up target descendants.

The target may call
`admit_inherited_kfd_target_debug_telemetry_v1` and send fixed-size cooperative
records. The inherited endpoint is private, send-only, credential-bound, and
contains no KFD authority, native identifier, pointer, address, path, or file
descriptor. Target artifact, dispatch, allocation, and diagnostic facts remain
labeled `declared`. Independently observed devices, queues, runtime events, and
queue-control effects remain KFD observations. A matching target code-object
declaration does not upgrade `execution_code_object`, which stays typed
`not_observed` until an independent execution observation exists.

V3 exposes exact bundle/request/KIR/source-map/HSACO correlation, bounded
target-telemetry summaries, redacted KFD device and queue snapshots, exception
events, queue suspend/resume, and termination over one agent-friendly JSONL
stream. After a successful `suspend_queues`, a client can capture the exact
session-owned queue envelope without granting new authority:

```json
{"schema":"fe2o3-live-gpu-debug-request-v3","operation":"capture_stopped_queue_envelope","request_id":3,"expected_revision":1,"queue":{"generation":1,"ordinal":1}}
```

The response deliberately keeps the overall session `running`; only the named
logical queue has retained session-owned suspension. It contains address-free
queue/device/save-area/header identities, exception bits, ring and queue shape,
gfx target, XCC count, relative header ranges, error-binding presence, and an
`opaque_checkpoint` status. A complete checkpoint exposes only its scoped
checkpoint/content correlation identities, byte and segment counts, and
`private_bytes_exposed: false`; truncation exposes only the required and
configured byte bounds and retains no prefix. The outer session separately
carries its declared artifact binding, which is not observed execution. It contains no PID,
native queue/device ID, descriptor, virtual address, PC, register, checkpoint
bytes, source path, or target-memory value. The response
identities use a fresh private random per-session correlation scope, not the
public artifact binding, so repeated sessions are not intentionally linkable.
The scope itself is never serialized or logged. The response
sets `resume_required: true`; capture never resumes the queue, and the existing
revision-checked `resume_queues` operation is required. Suspension ownership is
bound to the exact queue and device snapshot. Native-ID reuse or binding
substitution invalidates the logical generation, terminally poisons the public
session, and immediately enters the existing KFD session-finish cleanup; the
facade never discards suspension ownership while the KFD engine still owns it.

The CPU-visible capture is bounded, sequential, and non-atomic: direct-KFD
queue/device snapshots bracket eight XCC header reads; each header-bounded
control-stack and wave-state segment is read twice adjacently; and all headers
are reread before the queue/device binding checks. A content change inside an
adjacent pair, partial/denied read, header change, or queue/device substitution
fails closed. `complete` means every announced extent was captured, not that
all segments describe one coherent instant. Runtime and suspension are local
session invariants here, not capture-time hardware reobservations. The private
Rust checkpoint segments are zeroized on drop and cannot appear through
`Debug` or the agent protocol. This is identity-bound opaque capture, not
decoded or authenticated hardware state.
Wave records, lane state, registers, PC, source, and memory remain separately
typed unavailable because KFD 1.18 publishes no stable inner gfx942 decoder.
The generic stopped anchor remains `session_not_stopped`, and stopped
dispatch/workgroup/wave/lane queries remain `unsupported`. The MI300X
live-validation test creates a real target KFD queue, observes it, suspends it,
captures the bounded envelope, explicitly resumes it, then finishes
debugger-side KFD state and sends `SIGKILL` to and reaps the directly launched
leader. The existing live case proves only a complete zero-byte idle
checkpoint; non-empty hardware-written segment capture remains unqualified.
This is forced leader teardown, not graceful target queue/runtime shutdown or
descendant containment; the test deliberately does not load or execute its
declared fixture HSACO.

## Qualification assessment

`fe2o3-debug qualification --manifest /absolute/path/to/qualification.json`
strictly admits one bounded, single-link regular file and emits one
`fe2o3-debug-qualification-assessment-v1` JSON line. The response includes the
complete component and capture-mode evidence matrix, exact manifest and
environment identities, per-mode policy assessments, and an overall
`incomplete`, `failed`, or `caller_bound_policies_satisfied` disposition.
Input is capped at 256 KiB, and the complete response including its newline is
capped at 512 KiB. Symlinks, hard links, changing path identities, and
malformed or oversized input are rejected before stdout is written.

This mode executes no collector or target and grants no observation or
qualification authority. Its two authority fields are always false. An agent
can therefore explain missing evidence and select the next bounded capture
without scraping prose or converting the archived caller-bound manifest into
live GPU truth.

## Structured live ROCgdb protocol V3

`fe2o3-debug live-rocgdb --rocgdb /usr/bin/rocgdb --authorization ID --
PROGRAM [ARG...]` launches the exact argument vector through ROCgdb's GDB/MI3
interpreter. `--attach PID` selects an authorized attach instead. The PID,
ROCgdb path, target path and arguments are bootstrap authorities and never
appear in JSONL responses. The coordinator invokes no shell and uses no HIP or
HSA runtime.

The `fe2o3-rocgdb-mi-request-v3` JSONL protocol exposes structured capability
discovery, asynchronous events, caller-selected generic thread admission from
`-thread-info` tuple ordinals, and audited breakpoint/continue/pause/step
control. Native code-object and allocation addresses and source paths occur
only in admission requests; their responses contain content, allocation, or
source-span identities. Request lines, response lines, command counts, MI
records, nesting, strings and waits are bounded. Duplicate request IDs, stale
revisions, authorization mismatch, unknown fields, aliases and malformed MI
are rejected without screen scraping.

An admitted MI thread remains a generic logical debugger thread. Host, unknown,
and GPU-looking `target-id` text is not GPU classification evidence. Stopped
dispatch/workgroup/wave/lane, relative PC/source, register/value/expression,
and allocation-relative memory capabilities remain typed `unsupported` until a
separate trusted correlation source can authenticate a GPU thread binding; no
wave or topology coordinate is synthesized. ROCgdb and direct-KFD debug-trap
control are mutually selected session backends, although direct-KFD runtime
telemetry may still be correlated when it does not claim a second debug-trap
owner.

The deterministic MI fixture validates the currently available JSONL workflow
in generic CI. Installed ROCgdb capability discovery is validated on MI300X. A
live pure-KFD GPU wave stop with PC, `exec`, registers and resume has not yet
been validated, so this command does not claim that milestone.

## Native ROCgdb/KFD correlation V4

`fe2o3-debug live-rocgdb-kfd-v4` is a separate one-shot JSONL launcher; it does
not change the V3 protocol. It provisions a credential-bound V2 telemetry
endpoint into the exact ROCgdb-launched target, admits one explicitly selected
direct-KFD device, and queries only structured `-agent-info`, `-queue-info`,
`-dispatch-info`, `-thread-info`, and `-lane-info` result tuples at a ROCgdb
stop. The exact target-side KFD publication observation must match the selected
device, target process instance, queue occurrence, AQL packet, artifact,
dispatch generation, and geometry before workgroup, wave, or lane coordinates
are returned. AMD `target-id` values use a strict versioned grammar; MI stream
text and `details` are never evidence.

The structured tuple grammar is pinned to the emitter in the installed
ROCm 7.2.4 `rocm-gdb` source. Queue tuples have no `agent-id`; dispatch tuples
have no queue field, use constant strings for `grid`, `workgroup`, `fence`, and
`address-spaces`, and use zero-padded core addresses. The manual bundled with
that release contains a stale incompatible example with `queue_id`, plural
`fences`, and list-valued address spaces; V4 rejects that example and all
hybrids. Queue association comes only from the strict target-id hierarchy and
the independent KFD publication join. A capture also pins one exact native stop
generation across all five queries and rejects intervening running, exit, or
new-stop events. Wave coordinates use the target's wave-in-workgroup ordinal
and the actual partial edge-workgroup extents.

Native GPU, queue, packet, address, descriptor, PID, and target-id values remain
private. The response contains only derived identities, logical coordinates,
and capability truth. The code load base is an explicit caller admission and
is labeled as such; kernel entry address and size come from the exact admitted
HSACO descriptor inspection. Relative PC evidence binds all of those fields.
Source, registers, and memory remain typed unavailable. ROCgdb is the
sole stop owner and this path never acquires KFD debug-trap ownership. A target
or platform that cannot produce the exact cooperative stop returns a typed V4
unavailable result. The collector pumps MI events and V2 telemetry together,
so a missed breakpoint or target exit cannot erase an already observed AQL
publication.

On 2026-09-02, installed `/usr/bin/rocgdb` (GDB 16.3,
`rocm-rel-7.2-93`) and direct KFD on MI300X
ran the public command with the SHA-pinned
`lds_publish_read_reduce_i32_v1` diagnostic HSACO
(`ab6bda1e8af05b61c22753382e75dd6a9952db8e598eaac3cb5769863a618ed0`).
The safe V2 target declared the dispatch and emitted its target-side KFD
observation at the real post-AQL-publication point. ROCgdb did not return a
structured GPU stopped hierarchy for that direct-KFD code image, so the exact
result was unavailable rather than inferred:

```json
{"schema":"fe2o3-rocgdb-kfd-native-response-v4","result":{"status":"unavailable","probe":{"structured_mi_commands":true,"direct_kfd_device_admitted":true,"cooperative_v2_declaration":true,"cooperative_v2_publication":true},"reason":"gpu_stopped_state_unavailable"}}
```

That response, including its terminating newline, has SHA-256
`e83ce302728df11f5a496cf3576f2785825815a43fd784780828de190e9f8251`
at compiler commit `308d8fa00fa41e098b2a1a47bbfea1bc29735464`, tree
`aee01674fefa733731db35eae1a1705b3286179e`. A separate diagnostic MI run
showed the kernel breakpoint remaining pending, `amd-dbgapi` failing to read
`global#0`, and empty agent, queue, dispatch, and thread lists after the KFD
dispatch completed correctly. Its raw unredacted transcript was not checked
in; its SHA-256 is
`dd172567ea7311aef647161606769a74ac895c129b4e87ef402a8c18fb658856`.
The [official ROCgdb AMD GPU contract](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/ROCgdb/gdb/doc/gdb/AMD-GPU.html)
requires compatible ROCm runtime metadata, whereas the production fe2o3
direct-KFD runtime intentionally supplies `r_debug=0`. This result is
therefore an installed bridge capability gap, not permission to infer stopped
waves from KFD publication.

## Authenticated native stopped-state inspection V5

`fe2o3-debug live-rocgdb-kfd-v5` preserves the V4 launch, telemetry, artifact,
hierarchy, stop, and relative-PC contract and adds machine-interface register
and simple-local inspection. Registry discovery is reported separately and is
never treated as an observation. Register names and values are obtained with
`-data-list-register-names` and `-data-list-register-values`; locals use
`-stack-list-variables --simple-values`. Every command selects the exact
private current-thread token admitted by V4, and the stop generation is checked
after every result. Published scopes and evidence identities bind the V4
association, redacted stop, redacted thread, and authenticated wave. Counts,
MI records, commands, strings, and values retain the V3 bounds. The scope is
wave-level (`lane` is absent): V4 authenticates the current wave thread but has
no independent selected-lane observation, so locals are not labeled as
lane-specific.

Register bit patterns are captured, except the absolute `pc`/`pc_all` register,
which is redacted because V4 already supplies the authenticated artifact-relative
PC. Local values are captured only for explicitly scalar integral or Boolean
debug types. Pointers, aggregates, floating-point text, unknown types,
optimized-out values, and values outside the MI capture are typed unavailable;
the adapter does not reinterpret debugger text with host arithmetic. Source is
unavailable until an authenticated artifact-relative source map is supplied,
ISA until instruction boundaries and decoded bytes are bound to the artifact,
and memory until the target publishes exact allocation-relative authority.
Native selectors and MI address fields are not serialized or treated as
identity. Register bit patterns remain opaque machine bits and can numerically
resemble an address, but are never admitted as pointer or memory authority;
the known absolute PC register is redacted. Pointer local values are not
serialized.

The installed MI300X ROCgdb reports all five V5 inspection command names. A
2026-09-03 run of the public V5 command with the same SHA-pinned diagnostic,
direct-KFD device, and target used for V4 returned:

```json
{"schema":"fe2o3-rocgdb-kfd-native-response-v5","result":{"status":"unavailable","probe":{"structured_mi_commands":true,"direct_kfd_device_admitted":true,"cooperative_v2_declaration":true,"cooperative_v2_publication":true},"inspection_probe":{"register_names":true,"register_values":true,"simple_locals":true,"disassembly":true,"memory_bytes":true},"reason":"gpu_stopped_state_unavailable"}}
```

The exact JSONL record has SHA-256
`71ea391f489e0f068e524bef93fee1384b2dbf4956aeb33ef854db2b2a3dc5e1`.
Thus the direct-KFD launch still reaches the V4 stopped-state boundary.
This release validates V5 parsing, same-stop collection, redaction, and hostile
substitution with deterministic MI fixtures; it does not claim a physical
register or local capture on MI300X.

## Physical-entry V20 CPU observations

`fe2o3-debug sim --diagnostic-kir-v20 kernel.kir --request request.json --protocol jsonl --wave-width 64`
loads an exact verified V20 physical-entry canonical module and the existing
`fe2o3-simulation-request-v1` JSON request. Linux regular-file safeguards apply.
This explicit selector is not a source-authentication path. A canonical digest
identifies bytes, not their author, frontend lineage, deployment eligibility or
runtime artifact. V21 remains refused by this debugger route.

The adapter executes the ordinary CPU engine once and retains budget-owned
observations. JSONL V1 supports capability discovery, state, unfocused
event-granularity forward/reverse stepping, seek, paginated SSA values for the
selected checkpoint's lane, allocation-relative memory reads up to 256 bytes,
and termination. It does not resume execution when moving the cursor. Source
maps, hardware registers, hardware waves, breakpoint/watchpoint control,
persisted replay, continuation and other step granularities are unavailable.

Symbolic pointer halves and carries remain present as SSA bindings with
`Unavailable/NotRepresented`. The library retains their exact symbolic state;
the unchanged JSON grammar does not manufacture numeric addresses or encode that
private state. The displayed Wave64 active mask describes resident lanes, not
physical EXEC; physical EXEC is a separate scalar SSA observation.

Tooling limits are 128 KiB canonical input, 16 KiB request, 128 invocations in
Wave64 workgroups, 8,192 capture records, 8 KiB command frames, 64 KiB replies and
64 SSA rows per page. Input decode, immutable owner, independent admitted CPU view,
capture, retained snapshots and protocol workspace all continue one owned
cumulative verification ledger. Logical payload allowances are not allocator/RSS
caps. Existing legacy simulator/debugger APIs retain their separate accounting
contract; they cannot be used to admit this typed capture.

State-changing observation responses are checked against output bounds before
the cursor/revision commits. A transport failure ends the session, not a
reversible publication transaction. A terminal library forward-step is a no-op
without navigation cost; CLI command parsing/query/serialization is prepaid even
for terminal no-ops. A truncated capture does not claim a completed cursor.
No new canonical, source or debugger wire IDs are allocated by this adapter.

## Physical-global-copy V21 CPU observations

The separate Linux-only route is:

```sh
fe2o3-debug sim --diagnostic-kir-v21 canonical-v21.bin --request request.json --protocol jsonl --wave-width 64
```

It requires the exact V21 canonical owner and the existing
`fe2o3-simulation-request-v1` request, with two arguments in order: readonly
`u32` input and writable `u32` output. Inline buffers or valid shared-buffer
views use the existing request grammar; both allocations are observed. The
closed tooling launch is at most 128 invocations in [64,1,1] workgroups. The
unguarded full-EXEC global read still requires every launched input lane to be
readable and initialized even when output is empty. Short/uninitialized input,
incorrect access, or input/output on the same backing are refused by the real
CPU engine before a CLI session is exposed; disjoint ranges of one backing do
not bypass the profile's same-backing refusal.

For source-produced bytes, use the normal
[source diagnostic command](../../../docs/physical-global-copy-source-v21.md)
and retain its `diagnostic/canonical-v21.bin` unchanged. This CLI admits
serialized canonical structure, not that producer's source custody or host ABI
conditions. There is no snapshot, transcript, source-map, register-map or
runtime-observation importer on this route. V20 and V21 selectors cannot be
combined and neither route guesses another version after refusal.

The V21 adapter uses the same immutable-session JSONL V1 projections, limits,
prepaid command budget and transactional navigation as the V20 adapter. It
retains the original cumulative ledger through secure input reads, typed owner,
independent CPU view, engine capture and protocol workspace. Configuration
identity includes the canonical identity/length, exact request SHA-256/length,
and fixed limits, under a distinct V21 domain; value pages have a distinct V21
domain as well. Stale revisions, cursor configurations and page tokens refuse
without moving the cursor. Requests have an 8 KiB frame limit, replies 64 KiB,
SSA pages 64 rows and memory reads 256 bytes. The 4,096 command ceiling does not
override cumulative work exhaustion. These are logical payload/work bounds,
not whole-process RSS guarantees.

Pending global-load bindings are present as `Unavailable/NotRepresented`.
Only the existing engine's actual VM wait changes that same SSA binding to
captured scalar bits. Symbolic pointer/carry halves remain nonnumeric. Both
readonly input and writable output are available as allocation-relative
checkpoint observations; reverse navigation selects prior snapshots without
re-execution. A logical Wave64 mask is not physical EXEC or a hardware register
sample. Source-variable maps, physical register maps/capture, hardware waves,
native execution, GPU performance, breakpoints, persistent replay and resumable
execution remain unavailable. The source-authoring register names do not
create debugger register authority.

No canonical or debugger schema version changes are made by this route.
Existing V20 configuration/page domains and response bytes remain unchanged.
These implementation and test descriptions are not successful-run evidence;
actual-source CLI qualification is [recorded separately](../../../docs/physical-global-copy-cpu-debug-qualification-20260924.md).

## Diagnostic V18 CPU Debugging

`fe2o3-debug sim --diagnostic-kir-v18 kernel-v18.kir --request request.json --protocol jsonl --wave-width 64`
uses the ordinary simulator capture and JSONL debugger backend. Existing
supported scalar memory operations support logical stepping,
dispatch/workgroup/wave/lane
inspection and checkpoint memory inspection while retaining exact V18 identity
and inert layout metadata. Configuration identity binds the typed request,
target and capture limits, not merely input file names.

The original admission ledger remains live until the input and backend have
dropped, including stream errors and unwinding. Request, capture and protocol
allocations retain their existing separate bounds; no process-wide accounting
claim is made. Raw V18 carries no authenticated source-variable mapping,
persisted replay or GPU authority. Source maps, runtime-observation mode and
Wave32 are rejected before input IO; diagnosis V2 reports unavailable rather
than presenting V18 as a legacy source-bound graph.

Private scalar storage allocation/read/write executes with exact target-sized
layout rows and the existing memory and lifetime checks. Stepping and memory
events/checkpoint snapshots remain available. A stack containing a tagged
storage pointer is explicitly `NotCaptured`: the legacy debugger value format
cannot represent its row identity, so it is never converted into an ordinary
pointer or silently omitted. V18 scalar pointer/slice exposure to Generic keeps
the concrete allocation, permissions, view bounds and lifetime. Debug values
retain the logical Generic address space; memory events and allocation snapshots
retain the concrete backing. Slice source types remain Global/Constant and root
buffers remain Global-only; Private/Workgroup slices are not newly admitted.
Generic kernel arguments, allocations, atomics and
memory intrinsics remain refused. Aggregate/pointer storage, projections, copies,
discriminants and storage-bearing function parameters/results remain unsupported.

This adds no debugger protocol or KIR format change. Raw canonical bytes do not
authenticate an original Rust source owner or a compiler continuation.

## CPU schedule replay walkthrough

### Route availability and inputs

These routes have different input contracts; a larger wire version does not
make a diagnostic input eligible for semantic schedule replay.

| Route | Required inputs | Availability and scope |
| --- | --- | --- |
| Existing semantic CLI replay (`--replay-schedule`) | A supported canonical KIR or bundle route, its exact request, and a matching schedule document | Existing CPU replay; KIR/bundle, request, target and limits must match. |
| Existing diagnostic CLI selectors (for example V16 or V21) | Canonical bytes and a request for that exact version/profile | Existing closed CPU profiles with their own capabilities and refusals; no new persisted replay or physical-state authority. |
| `schedule_replay_walkthrough` library example | Bundle V6, its exact request, and an independent oracle JSON file | Example source is included here; a bundled tutorial fixture and clean-checkout input qualification remain pending. |

No row establishes source authentication, GPU execution, or browser acceptance.
The example does not supply a compiler export, request, or qualified tutorial
fixture. Existing diagnostic routes are documented above; they are not aliases
for the example's Bundle V6 loader.

### Run against caller-provided inputs

The `schedule_replay_walkthrough` example admits an existing Bundle V6 and
request, records a seeded CPU schedule, round-trips its canonical document, and
captures a debugger transcript while replaying that schedule. It checks the
entire selected output, both initialization masks, initial/final checkpoint
memory, and reverse/forward cursor round trips against a separately supplied
oracle.

```sh
BUNDLE=/absolute/path/to/retained-bundle-v6.fe2sim
REQUEST=/absolute/path/to/matching-request.json
ORACLE_JSON=/absolute/path/to/independent-oracle.json

cargo run --locked --offline -p fe2o3-debug-cli \
  --example schedule_replay_walkthrough -- \
  "$BUNDLE" "$REQUEST" "$ORACLE_JSON" 5
```

Replace all three placeholder paths with caller-provided files. The historical
`loop_helper` oracle below illustrates the expected content; it is not a bundled
input set or evidence that a clean checkout can reproduce that compiler export.

Supply the exact bundle and request from your own retained compiler export.
This example does not compile the source. For the retained `loop_helper`
fixture, the request has seed `0xabcd1234`, salt `0x0f0f55aa`, rounds `1`,
and four active invocations. Its Rust helper computes
`(seed ^ salt) & 0xffff == 0x479e`. The output view spans bytes 4..20 of
shared backing 0; the surrounding words are canaries. Its independent oracle is:

```json
{
  "buffer_argument": 0,
  "shared_backing": 0,
  "allocation": 1,
  "initial_bytes_hex": "efbeaddea5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5bebafeca",
  "initial_initialized_hex": "ffffff",
  "bytes_hex": "efbeadde9e4700009e4700009e4700009e470000bebafeca",
  "initialized_hex": "ffffff"
}
```

For an ordinary buffer argument, use `"shared_backing": null`.
`allocation` is the logical CPU allocation ID from the corresponding debugger
capture. Initialization masks use the low bit first within each byte; unused
padding bits must be zero. The oracle covers the complete buffer/backing,
including canaries, and is limited to 4096 bytes. Derive it from the kernel's
intended behavior rather than copying simulator output. Wrong output,
initialization, or checkpoint expectations fail without a success report.

The JSON result includes exact bundle/request/oracle/schedule identities,
record counts, one admission and schedule-recording duration, and raw
nanosecond samples. After one warmup, each of 1..16 samples measures a complete
replayed capture and 128 reverse/forward checkpoint round trips. Memory checks
also revisit the initial snapshot after visiting the final snapshot.
Navigation selects immutable observations; it does not undo kernel execution.

Record the build profile, toolchain, CPU affinity and input hashes alongside
the JSON when comparing timings. The command above uses Cargo's development
profile. These are CPU library timings, not browser or GPU performance results.
The example requires a complete bounded transcript and retains the existing
schedule refusals for raw diagnostic KIR V16..V19.

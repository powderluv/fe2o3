# Support matrix

fe2o3 is a developer preview. This matrix distinguishes a runnable or qualified
engineering slice from a supported public platform. **No GPU target currently
has a stable public support commitment. This matrix makes no HIP/HSA parity
claim.**

## Status definitions

| Status | Meaning |
| --- | --- |
| Supported | Intended for public use and covered by an explicit compatibility policy |
| Qualified | Passed named tests for an exact revision, target, and environment |
| Experimental | Implemented but incomplete, unstable, or not admitted as a public workflow |
| Unavailable | Not implemented, intentionally rejected, or missing required authority |

Qualification does not imply support. Authority-free evidence does not
authorize compiler publication, loading, dispatch, or a hardware claim.

## Host platforms

| Host | Status | Boundary |
| --- | --- | --- |
| x86-64 Linux | Qualified for generic CI; experimental for users | Primary development host and the only host for hardened Linux file/process/device boundaries |
| Other Linux architectures | Unavailable | Not qualified; several binaries and tests require x86-64 |
| macOS | Unavailable | No KFD; Linux-specific simulator input custody intentionally fails instead of weakening guarantees |
| Windows | Unavailable | No KFD; destructive artifact cleanup fails closed because the required opened-directory semantics are absent |

The repository pins Rust `nightly-2026-04-03` and declares Rust version `1.94`.
Use the pinned toolchain for development-preview qualification.

## GPU targets

| Target | Compile | Direct-KFD execution | Public status |
| --- | --- | --- | --- |
| MI300X, `gfx942:xnack-`, Wave64 | Bounded qualified slices | Bounded diagnostics and exact qualification lanes | Experimental |
| `gfx942` configurations other than the qualified profile | Target-dependent | Not qualified | Unavailable |
| MI350 family, `gfx950:xnack-`, Wave64 | Selected ordinary Rust SIMT/tile examples compile through extraction/KIR to inspected LLVM/HSACO | No qualified production direct-KFD launch path; separate example-specific numerical companions exist | Experimental engineering evaluation, not a supported production runtime surface |
| RDNA targets including `gfx1151` | Partial compile/code-object slices | No qualified production launch path | Experimental compile surface only |
| Other AMD GPUs | Not qualified | Not qualified | Unavailable |
| NVIDIA or other non-AMD GPUs | No backend | No runtime | Unavailable |

fe2o3 does not infer support from a `gfx` name reported by a tool. Target,
features, wave width, KFD topology identity, artifact metadata, and qualification
policy must agree exactly.

The [gfx950 engineering evaluation guide](getting-started.md#gfx950-engineering-evaluation)
links the maintained BF16, FP8/FP4 and vecadd setup and commands. These selected
paths are not a claim that every tutorial variant or every MI350-family
configuration is qualified. Compile-only observations, actual numerical runs,
CPU simulation, historical source-model proofs and protected production
admission remain distinct. In particular, a gfx950 result does not qualify
gfx942, and successful HSA/HIP numerical execution does not establish the
direct-KFD and machine-refinement acceptance required by issue #275.

## Execution modes

| Mode | Input | Status | What it establishes |
| --- | --- | --- | --- |
| CPU KIR simulation | Verified canonical KIR V7 plus typed request | Qualified for the admitted subset | Deterministic semantic execution and observation only |
| CPU source bundle simulation | Rust kernel exported through production source/MIR/KIR stages | Experimental | Exact authority-free bundle content and deterministic KIR execution |
| Direct-KFD diagnostic | Exact admitted artifact and invocation | Bounded qualified lanes | The properties named by that test only; the current runtime child multiplexes at most 65,536 logical streams over exactly two persistent compute lanes with caller-driven FIFO scheduling and disjoint-allocation concurrency |
| Production application dispatch | Ordinary Rust application | Unavailable publicly | Typed Worker V3 refinement-backend/receipt consumption exists, but the protected verifier deployment and concrete issue #214 semantic-machine proof backend/artifact are missing |
| Production HIP execution | N/A | Unavailable | Not a fallback runtime |
| Production HSA execution | N/A | Unavailable | Legacy/qualification code does not define the public runtime |
| gfx950 example numerical evaluation | Exact selected source, inspected HSACO and independent oracle | Experimental, example-specific | Explicit external-HSACO or HSA/HIP companions observe numerical behavior only; no protected publication, direct-KFD application authorization or source-to-machine refinement claim |
| Runtime completion, events, and callbacks | Backend-neutral typed handles | Experimental | Submission/event query, poll, and bounded wait share one typed completion state; callbacks discharge exactly once on conclusive status, and stream query/synchronize returns an aggregate observation under one deadline. The optional async engine provides bounded executor-neutral event futures with stable budgeted completion polling and panic containment. Additive progress mode drives a separate bounded roster of pending streams through `flush_stream`; ordinary construction remains observation-only, and registration/future drop never cancels or releases work |
| Typed atomic/collective launch wrappers | Admitted typed kernel plus explicit contract | Experimental direct/multi KFD SPI; no production authority | Operation, scope, success order, compare-exchange failure order/weak mode, geometry, and collective membership are preserved through additive backend SPIs and the exact Runtime Worker V5 transport. KFD ordinary/qualification constructors remain false; an unsafe semantic-authority constructor may enumerate exact non-System atomic and workgroup-collective profiles, which are retained through scheduling and final invocation authorization. Unlisted/invalid profiles reject before custody; later final-authority denial settles the accepted unpublished submission and releases custody before publication. A concrete production authority, broader profiles, native litmus evidence, and formal refinement remain unavailable |
| Runtime Worker subprocess | Exact V1, V4, or V5 handshake | Experimental bounded transport | V1 requires an explicit immediate-progress marker and cannot host KFD. V4 forwards execution capabilities, flush, same-device async copy, cancellation, and deadline-bounded drain. V5 retains that exact surface and adds typed atomic/collective contract carriage. Direct and multi-device KFD implement the V5 server bound; copy-only native XGMI rejects semantic requests before custody and advertises neither capability. V4 bytes and source contracts remain unchanged, cross-version handshakes reject, and capability records fail closed outside the latest successful enumeration. Real child-process tests exercise V4 ordinary and V5 atomic background-flush completion plus timeout, decoded-terminal, and EOF sealing. The handshake is compatibility negotiation, not worker authentication, and these host tests establish no native progress. |
| Same-host multi-GPU peer copy | Exact admitted KFD devices | Experimental bounded lanes | Generic routing uses bounded host staging; a separate exact two-device copy-only backend retains native XGMI mappings and publishes a deterministic FIFO ready queue in directional prefixes of at most 63. One flush synchronously drains non-final prefixes from its entry snapshot and leaves the final prefix published for host/DMA overlap; poll/wait are observation-only. Opt-in async progress applies only through Send-capable Worker V4/V5 adapters; direct KFD remains thread-affine. Benchmark `outstanding_depth` is queued work on one ordered directional SDMA engine, not engine concurrency. |
| Striped same-device SDMA submission | Exact admitted gfx942 striped queue set | Experimental production custody boundary | One call deterministically balances up to 1,008 requests over an even 2-through-16 queue roster, prepares all shards before the first publication, and advances selection only after complete publication plus closing currentness. Partial failure distinguishes confirmed shards, one optional indeterminate shard, and untouched indexed requests as audit-only process-teardown custody; only no-effect preflight is retryable. It is not rollback-capable or an atomic device transaction. Coordinator fault tests exist, but live native fault-injection and performance evidence are missing. |
| Persistent native-allocation custody | One existing mapped gfx942 device-local allocation | Experimental detached core | A non-Clone, thread-affine KFD owner retains one local mapping or complete canonical two-device peer mapping and tracks at most 64 move-only range uses. Access is derived from a closed operation roster; active writer hazards, stale frontiers, timeout release, and normal extraction while work is live or quarantined fail closed. Peer-mapped operations are classification only. Compute/SDMA queue bindings, native publication/completion, directional XGMI routes, live currentness checks, runtime-facade integration, and concrete/formal refinement remain unavailable. |
| Persistent local SDMA adapter | One promoted device-local buffer and one ordinary host buffer on an exact targeted gfx942 SDMA queue | Experimental single-flight custody boundary | Promotion preserves the queue's existing outstanding-buffer debit and binds queue occurrence, native queue, direction/engine, pool generation, storage identity, and extent. Move-only submit, poll, and bounded wait retain exact allocation, host, range-use, and ticket custody; recoverable prepublication failures restore owners, while retained or later ambiguity is opaque process-teardown custody. Exact quiescent-frontier retirement reclaims bounded ledger history, with 66-cycle and stale/substituted-frontier host coverage. Production transition functions and independent executable/Verus models cover the bounded transition surface. It is not connected to the runtime facade or async progress engine and provides no compute, D2D, peer/XGMI, striped queues, concurrent range leases, hardware evidence, refinement theorem, or performance claim. |
| Directional persistent local SDMA adapter | One promoted pooled or exact-size device buffer plus one ordinary host buffer on the exact gfx942 engine-1 H2D/engine-0 D2H pair | Experimental runtime-integrated single-flight custody | R20 wires the R19 parent/child queue and allocation custody into direct `KfdRuntimeBackendV1` H2D/D2H copies. Every packet is capped at `0x003f_ffe0` bytes and exact completion, settlement, and frontier retirement precede continuation publication by `flush_stream`; poll and wait are observation-only. Zero-progress retryable publication is conclusive failure, while partial-progress retryable failure releases all native and scheduler retains into an exact releasable quiescent marker preserved by events, dependencies, shutdown, and Worker V4/V5. R21 adds a private cfg(test) FIFO seam with driver-bound opaque owners and 16 concrete failure-path tests; it also fixes synchronous multi-chunk D2H classification after earlier host mutation. The independent model has 17 tests and brings Verus totals to 373 obligations and 209 rejected mutations. H2H and D2D reject before facade mutation. There is no batched large-copy, persistent compute/XGMI sharing, cross-thread direct progress, hardware, executable/native refinement, or performance-parity claim. |
| Background stream progress | Send-capable runtime backend with portable flush | Experimental opt-in host scheduler | `spawn_with_progress` owns a bounded registration roster and bounded cyclic flush budget independent of event polling. Ordinary `spawn` remains observation-only; registration drop never cancels, releases, or finally flushes. Retryable failures are retained and terminal ambiguity seals the engine. Direct KFD is intentionally non-Send, and no hardware liveness or parity claim follows. |
| Distributed kernel execution | N/A | Unavailable | General multi-device kernels, synchronization, and overlap contracts remain open; there is no unified native multi-device compute owner |

The simulator does not model GPU time, occupancy, cache behavior, physical wave
scheduling, or performance. It does not predict performance or prove GPU
equivalence or race freedom.

## Kernel semantics

| Surface | Status | Notes |
| --- | --- | --- |
| Fixed-width integers and booleans | Qualified subsets | Fixed-width wrapping and comparison semantics are explicit |
| F16, BF16, F32, F64 scalar operations | Qualified simulator/compiler subsets | Simulator uses pinned software IEEE evaluation; unsupported transcendental operations fail closed |
| 1D/2D/3D launch indices | Qualified subsets | Typed logical indices; target layout is explicit |
| Checked global buffers and views | Qualified subsets | Allocation-relative bounds, access, initialization, and provenance checks |
| Workgroup memory and barriers | Qualified subsets | Static scalar LDS plus one explicitly sized reachable canonical dynamic LDS base, with convergent barriers and exact initialization/publication/lifetime checks within defined limits; multiple bases and `DynamicAtLeast` fail typed |
| Integer atomics and fences | Qualified simulator subsets; typed runtime and unsafe KFD-authority SPI | Exact supported width/operation/ordering/scope combinations only; ordinary KFD advertises none, while an unsafe semantic authority may enumerate exact non-System profiles without supplying a shipped production authority or native proof |
| Wave32/Wave64 collectives | Qualified simulator subsets; typed runtime and unsafe KFD-authority SPI | Logical collective semantics, not physical `EXEC` emulation; ordinary KFD advertises none, while an unsafe semantic authority may enumerate exact workgroup profiles without supplying a shipped production authority or native proof |
| Rust volatile memory bridge | Experimental bounded slice | Authenticated scalar volatile load/store with explicit bounds and access checks; not broad Rust or general device-language support |
| Helpers and structured control flow | Qualified simulator/compiler subsets | Bounded call depth and explicit unsupported diagnostics |
| General Rust `std`, allocation, unwind, dynamic dispatch | Unavailable | Device subset fails closed |
| General inline assembly, external calls, generic address space | Unavailable | No inferred lowering |
| General matrix/MFMA/WMMA and workload libraries | Partial bounded slices | Not a stable general kernel surface |

Refer to the [implementation roadmap](implementation-roadmap-v2.md) for exact
open semantics. A passing example or test does not generalize beyond its stated
contract.

## Debugging

| Capability | CPU simulator | Live KFD | ROCgdb adapter |
| --- | --- | --- | --- |
| Dispatch/workgroup/work-item hierarchy | Qualified logical view | Bounded topology/queue facts | Tool-dependent, not generalized |
| Logical wave/lane visualization | Qualified | Unavailable as hardware state | Tool-dependent |
| KIR operation and SSA inspection | Qualified | Unavailable | Unavailable unless separately correlated |
| Allocation-relative memory | Qualified | Bounded declared/control facts | Bounded relative inspection |
| Breakpoints and watchpoints | Qualified transcript controls | Unavailable | Bounded MI controls |
| Forward/reverse stepping | Qualified deterministic replay | Unavailable | Forward tool control only |
| Source maps and variables | Bounded compiler-bundle paths | Unavailable | Tool-dependent and incomplete |
| Hardware PC/register/wave state | Explicitly unavailable | Unavailable | Bounded by installed ROCgdb capabilities |
| Structured failure diagnosis | Bounds/barrier classes implemented | Unavailable | Not inferred from a clean stop |
| Agent-facing protocol | Versioned JSONL | Versioned JSONL | Bounded GDB/MI-to-JSONL adapter |
| Simulator/direct-KFD differential | Prepared from exact Bundle V4 plus the admitted structural bridge; Bundle V5 and current Bundle V6 are CPU-simulation custody only | Sealed generated Worker V3 completion only; currently blocked on the unwired protected application verifier | Not applicable |

Logical CPU observations are never relabeled as hardware observations. Native
paths, descriptors, addresses, and unverified declarations are not agent
authority.

## Profiling

| Capability | Status | Boundary |
| --- | --- | --- |
| rocprofv3 dispatch JSON/CSV import | Experimental, bounded implementation | Strict reviewed dialects, KFD identity join, bounded collector custody |
| Dry-run collection planning | Experimental | Produces no fabricated collection recipe when prerequisites are unavailable |
| Direct-KFD runtime observation | Experimental, MI300X-qualified slice | Opt-in bounded lifecycle, host staging, queue, AQL publication/completion, and runtime-authenticated process-local host timing. Fresh recorder occurrences distinguish `Instant` epochs. A low-level KFD GPU/CPU/system counter sample supports clock-domain calibration only; neither source is an authenticated GPU start/end timestamp or rocprof correlation claim |
| rocprof wrapper host-wall comparison | Experimental, MI300X-observed slice | Explicitly authorized alternating raw/wrapped process timing with exact identities, bounded outputs, complete per-leg outcomes, and a caller candidate budget. This is wrapper-path wall time, not counter/PC/ATT/debugger or kernel-capture overhead; empty collector inventory keeps capture overhead and loss unavailable |
| Real GPU-dispatch round trip | Unavailable as protected qualification | Current tests use deterministic/fake collector inputs plus real KFD observation where gated |
| Runtime/copy attribution | Incomplete | Direct-KFD logical runtime and host staging are observed. A separate opt-in, canonical semantic sidecar and authenticated timestamp-report V2 expose exact runtime-authorized atomic/collective contract declarations without changing frozen Runtime Profile V1; device copy-engine events, semantic execution history, and full treatment lineage remain unavailable |
| ATT/thread trace import | Experimental, authority-free decoded interchange | Strictly admits a canonical external ROCprofiler SDK 7.2.4 callback export with exact manifest/raw/header/library/exporter identities, bounded paging, and loss/incomplete truth. A separate exact supplied decoded-ATT/HSACO/Characteristic binding maps ELF PCs through authenticated kernel symbols to every matching sparse source/MIR/KIR/LLVM/ISA occurrence without exposing symbol names or addresses; raw decoding, authenticated decoder custody, beta collection, and full-grid coverage remain unavailable |
| Source/IR/ISA causal localization | Incomplete | Exact artifact and source-map associations remain open |
| Performance prediction | Out of scope | CPU simulation and profiler import make no prediction claim |
| Agent-native queries | Experimental | Read-only typed facts with explicit provenance and availability |

Optional ROCgdb, rocprofv3, and ROCProfiler SDK components are tools, not fe2o3
runtime dependencies or fallbacks. Their installed versions and measured bytes
are part of the exact workflow admission where required.

## Stability

| Surface | Stability |
| --- | --- |
| Rust crates and macros | Unstable developer preview |
| `cargo fe2o3` command line | Unstable; production route is singular but installation is not released |
| KIR and simulation bundles | Versioned strict formats; no blanket forward-compatibility promise |
| Debugger/profiler JSONL | Versioned strict protocols; capabilities remain additive only where their contract permits |
| Evidence/receipt protocols | Frozen per document version; not a public authority service guarantee |
| Examples | Regression and qualification inputs; not all are runnable production applications |
| GPU compatibility | Exact qualified profiles only; no family-wide implication |

Before relying on a capability, record the fe2o3 commit, exact target/features,
kernel and artifact identities, driver/ROCm/LLVM versions, and named validation
lane. Open a bug when an advertised qualified lane fails under its documented
environment; open a feature request for unavailable combinations.

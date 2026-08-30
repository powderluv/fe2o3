# Production compiler convergence V1

This document defines the implementation shape for
[#175](https://github.com/harsh-nod/fe2o3/issues/175). It narrows the compiler
work under [#134](https://github.com/harsh-nod/fe2o3/issues/134) to one
production transaction. The former scalar, GEMM, attention, collective, and MoE compiler oracles have been deleted. They are not additional production architectures. Differential evidence may remain as
inert fixtures or offline tools, never as another route in production crates.

## One transaction

The completed convergence target sends every kernel-containing final crate
through one rustc-owned transaction:

```text
authenticated rustc kernel closure
    -> canonical semantic MIR
    -> owner-authenticated executable MIR graph
    -> canonical target-neutral Kernel IR
    -> owner-authenticated executable Kernel/GPU graph
    -> typed AMDGPU legalization
    -> canonical typed LLVM handoff
    -> pinned upstream LLVM and in-process LLD worker
    -> independently inspected AMDHSA artifact
    -> generated typed host interface
```

`cargo fe2o3 build` and `cargo fe2o3 run` realize that transaction with one
fixed orchestration plan. Cargo first performs a device `build` for the fixed
AMDGPU target under the protected compiler closure. Only after the exact
generated-artifact generation commits does a fresh Cargo process build or run
the same package/feature/profile selection for the pinned host target using
ordinary rustc. Run payload arguments are forwarded only to that host process.
The caller cannot pass `--target`, and the host phase receives no fe2o3 backend,
wrapper, broker, device, build-manifest, qualification, or simulation controls.
This is phase separation inside one production build, not two compiler routes.

Compiler provenance is one cross-cutting input to this transaction, not a
second compiler route. The canonical `CompilerClosureV2` commits to six
role-specific SHA-256 pins:

1. Cargo executable;
2. static Cargo binding trampoline;
3. full `cargo-fe2o3` binding wrapper;
4. rustc executable;
5. complete rustc runtime tree; and
6. selected rustc codegen backend.

The closure also commits to the canonical Cargo-to-trampoline-to-wrapper
transition protocol, currently
`CARGO_BINDING_TRANSITION_PROTOCOL_VERSION_V1`, and derives one aggregate
identity from the domain, protocol version, and ordered pins. The aggregate is
validated, not an independently trusted seventh pin.

`RustcInvocationDescriptorV3` is exactly one complete
`RustcInvocationDescriptorV2` process description, including cwd, final argv,
and complete sorted child environment, plus the complete canonical
`CompilerClosureV2` preimage. Construction cross-checks the duplicated rustc
and backend digests.

### Compiler provenance wiring

| Boundary | Current state | Remaining production wiring |
|---|---|---|
| Protected release and Cargo broker | The release contract validates `CompilerClosureV2`; the broker transfers a sealed raw closure capability to the binding wrapper. | Preserve the admitted closure through runtime authorization. |
| Exact rustc invocation | The wrapper constructs and seals V3 for production, installs its immutable image at fd 199, retains parent custody, and the backend revalidates argv, cwd, environment, target, role pins, and closure before V3 publication. Qualification V2 captures receive no fd 199 capability and are not retained as production custody. | Extend archived end-to-end evidence across the final application process boundary. |
| Compiler module handoff | Production has one mandatory protected-custody path and one V3 publication/consumption transaction. Before monomorphization, a device transaction must retain one admission containing both the authenticated gfx942 target and exact managed build attempt; preflight roots and post-monomorphization device work must agree exactly. The attempt and protected rustc invocation then move as one publication custody value, so no optional or late direct-publication branch remains. The ordinary publication branch and runtime schema selector are deleted. | Keep V1/V2 consumers confined to explicit qualification code until their oracles retire. |
| Worker publication restart | `ManagedProductionBuild` has only `Fresh`, `Recovered`, and `Ready` states. It performs strict V3 preflight, one-shot consumption, direct LLVM/LLD execution, independent inspection, durable publication, and load-readiness recovery. Recovered current-publication custody now reaches the private joined KFD invocation after Worker V3 authentication. | Replace synthetic verification and remove external HSACO-path injection from the inherited application hardware lane. |
| Application handoff | Production admits only the canonical Worker V3 load envelope and rejects intermediate runners. Cargo pins the application, binds the envelope, artifact-directory, and ACK descriptors into a fresh occurrence, validates the challenge-bound ACK, and retains current-publication custody through exit. Before ACK it also creates fd 195 inside that exact child, transfers the service endpoint and pidfd through the fixed supervisor, exposes only the connected `SOCK_SEQPACKET` through seccomp, and retains issuer readiness; policy fd 202 is never exposed. `fe2o3-host` exports a one-use move-only auditor that consumes fd 195 and verifies the issuer signature, retained external commit receipt, and fresh client-bound external recovery receipt for the exact Worker record without granting authority. The compiler's V4 proof association losslessly carries all five stage identities plus its exact signed aggregate MIR-to-live-PLIRON receipt inside the frozen V3 capsule envelope. Worker V3 independently reimports and cross-checks that receipt, then retains it beside signed compiler-currentness evidence throughout authentication, HSA load, dispatch, and unload. Its generated KFD transition separately joins authenticated verifier evidence, generated host-memory arguments, current publication, launch geometry, and one checked device into a private invocation authority. Worker V2 application routes and raw HIP authority surfaces remain deleted. | Supply the reviewed crate-owned verifier that joins the retained proof-input owner with protected key custody and independently administered monotonic-anchor deployment; add authenticated LLVM/final-machine refinement and dynamic launch evidence; route an ordinary generated application through inherited KFD without external artifact injection; then remove the HSA migration route. |
| Qualification isolation | Workload oracle features and executable oracle paths are deleted. `FE2O3_QUALIFICATION_ORACLE_V1` remains only as a rejected sentinel. Cargo always completes the same Worker V3 transaction under every feature set. | Keep historical V2 names confined to frozen wire compatibility used by Worker V3. |

The Cargo package resolves its build-configuration API directly
to `PreparedProductionBuildConfig`, with no feature-dependent compatibility
type alias and no no-op qualification conversion methods. It parses only
`FE2O3_PRODUCTION_BUILD_CONFIG_V1`; the profile enum, Worker V2 schema parser,
envelope controls, source-debug controls, and workload fields are deleted. The
release path always uses the production expected-identity namespace and
ordinary compiler-capability profile; route-dependent identity and S09
selection logic no longer exists in Cargo. The binding wrapper admits every
managed kernel root through protected
rustc and requires compiler-closure custody directly; it does not compile the
qualification-oracle predicates or qualification command preparation path.
The Cargo driver also binds the fixed gfx942 target profile and production
semantic-generation identity directly. Its backend preparation context has no
production-route boolean or simulation selection.
Once a compile is selected as a production kernel root, the binding wrapper
requires the concrete production manifest before it can begin an artifact
attempt. Production preparation and completion contain no Worker V1/V2,
in-rustc oracle, simulation, row-softmax, or empty-attempt dispatch. Production
capability intake also releases the broker's one-shot invocation
authority immediately after authenticating the transfer. The release
`CompilerCapabilities` shape has no retained invocation-authority field or
child-inheritance API. The S09 broker profile and pinned-Cargo transfer image
are deleted. Shared closure, backend, Cargo-image, and artifact validation
remains implementation-neutral and runs before production receives custody.

The feature-free rustc backend likewise does not compile
`QualificationSelection`, `SelectedQualificationOracle`, or
`RustcInvocationPolicy`.
It captures a selector-free production environment preflight, enters protected
V3 rustc admission directly, and requires the production device transaction to
complete directly for every discovered kernel. The qualification-feature build
has an optional non-publishing oracle token and an invocation-policy enum for
differential testing, but no compiler-route enum or release implementation
choice.

The `cargo fe2o3 simulate` command and its Cargo-side oracle graph are deleted.
On Linux, the standalone `fe2o3-kir-sim-cli` consumes exact verified canonical
KIR V7 without source, compiler, refinement-proof, artifact, load, launch, GPU,
timing, or performance authority. `fe2o3-kir-sim-trace` independently maps
ephemeral CPU-simulator events into collector-neutral Semantic Trace V1. These
model and differential tools remain authority-free test programs; production
`build` and `run` cannot select either one.

The host-consumer and shared hostile application fixtures accept only V3
inputs. The old V2 consumer binary, input adapter, Cargo feature, and hostile
fixture protocol implementation are deleted. All application-boundary
adversarial coverage now runs against the strict V3 path in generic-core CI;
the Cargo V2 publication/restart vertical and its fixture binaries are deleted.

Version suffixes remain on serialized records, identity domains, receipts, and
external protocol types. Private production methods and states are unversioned
because there is only one implementation. A new production schema must be an
explicit migration of the same transaction, never a selectable pipeline.

The implementation uses one move-only typestate owner, conceptually
`ProductionCompilation<'tcx, Stage>`. A transition consumes the previous
stage and returns the next. The owner retains:

- the active compiler session and #140-authenticated graph handles;
- the canonical record and identity at each completed semantic boundary;
- bounded before/after transformation receipts;
- source, ABI, layout, target, proof-obligation, and diagnostic provenance;
- the exact Worker request, response, finalized bytes, and inspection owner;
- no publication, load, launch, or runtime authority.

The owner may retain several graph handles in one session, but a semantic fact
has one authoritative representation at a stage. Side data is permitted only
when the graph cannot yet represent the fact. The graph and side data are
compared at every boundary, and the side-data field has a named removal issue.

## Entry convergence

The explicit extraction driver and compatibility codegen backend must both call
one importer:

```text
import_rustc_kernel_closure_v1(tcx, collected_roots, limits)
    -> OwnerControlledSemanticMirV1
```

The importer is workload-neutral. It discovers roots from authenticated
`#[kernel]` metadata and rustc identities, traverses the complete reachable
monomorphized device closure, and records typed rustc-independent semantics.
It never branches on an export name, source substring, workload identity, or
exact MIR transcript.

The imported representation must preserve the facts needed by later lowering:

- source spans and expansion/call-site origins;
- item, instance, generic, and const-generic identities;
- layouts, FnAbi modes, calling convention, unwind behavior, and relocations;
- locals, types, places, projections, operands, constants, assertions, drops,
  volatility, atomics, direct calls, tail calls, and control-flow edge meaning;
- pointer provenance, address-space requirements, and source-level capabilities;
- deterministic call chains for unsupported reachable behavior.

Ordinary scalar admission and General GEMM refinement consume this same owner.
They do not run separate importers. The current #174 work is accepted only when
generic capture is independent of ordinary-scalar authentication and scalar
lowering is a separate consuming adapter.

### Rustc and device target custody

The current compatibility backend analyzes the final crate in a host rustc
session while `cargo-fe2o3` separately configures the device compiler for
`gfx942`. These are two different target facts. Host-session layout and FnAbi
must never be relabeled as AMDGPU layout or FnAbi merely because device lowering
was selected.

Production collection therefore retains both the exact rustc layout context
and the fixed `gfx942:xnack-` device profile in one move-only token. The
semantic importer must consume that pair and fail closed on an unsupported
bridge. The intended convergence is for the explicit extraction driver and the
compatibility backend to enter the same importer under an AMDGPU rustc target
session; the compatibility backend may become a thin coordinator for that
session. Existing host-to-gfx942 conservative layout projections remain
qualification inputs and cannot mint production semantic identity. Until the
AMDGPU-session handoff exists, production stops before semantic-MIR admission.

## Canonical and executable IR

`fe2o3-mir-model` and `fe2o3-kernel-ir::Module` remain the canonical semantic
identity boundaries. Pliron operations are transient executable state.

The general Kernel IR module already represents functions, roles, signatures,
blocks, SSA values, control flow, memory effects, address spaces, barriers,
atomics, wave operations, matrix operations, capabilities, and inline assembly.
Profile records may validate or construct regression fixtures, but production
MIR lowering must emit the general module rather than select a profile-specific
replacement.

Conversion between canonical records and executable graph state is checked in
both directions. Identity never includes text rendering, traversal accident,
arena slot, pointer, process ID, or filesystem path. Frozen wire formats remain
byte compatible.

## Transformations

All mutable transformations execute through the sealed #140 service. A pass
receives an owner-authenticated operation handle and a bounded configuration;
it cannot receive or return a raw Pliron pointer.

### Session dependency boundary

The sealed service requires a dependency inversion around the dialect crates.
It must not be implemented as a public callback that receives `&mut Context`:
safe callback code could retain a contextless upstream `Ptr<T>` and recreate
the cross-session confusion that #140 is intended to remove.

The production dependency direction is:

```text
Pliron owner/registration core
    <- fe2o3 dialect definitions and typed constructors
    <- closed production Pliron session and transform adapters
    <- ProductionCompilation typestate transaction
```

The lower owner core contains context identity, bounded dialect-registration
actions, opaque handle mechanics, and fixed diagnostics. Dialect crates depend
only on that core and pinned Pliron APIs. The closed production-session layer
depends on the owner core plus the admitted dialect crates, owns the raw
`Context`, and directly invokes their typed constructors and transformations.
Its raw-context implementation is compiler-internal TCB code; it exposes no
callback, trait implementation point, context, pointer, value, block, type, or
attribute handle to callers.

Construction consumes a bounded canonical MIR or Kernel IR recipe and returns
an opaque root handle only after recursive verification and canonical
cross-checking. Transformation selection is a closed fe2o3-owned operation,
not an arbitrary caller-provided Pliron `Pass`. A transition consumes the
input-stage capability, reserves its complete work and growth budget, mutates
only the authenticated tree, recursively verifies the result, and returns a
new stage capability plus a canonical receipt. Any failure after allocation or
mutation begins poisons and terminally consumes the production session.

The current textual import and detached lowering services remain test and
migration bridges. They cannot be called by `ProductionCompilation`, and
removing their final production callers is part of #140/#178 rather than a
second compiler route.

Each pass performs this transaction:

1. Validate the complete input graph and canonical binding.
2. Reserve bounded work, diagnostics, graph growth, and nesting.
3. Apply one independently specified transformation deterministically.
4. Validate the complete output graph and declared analysis preservation.
5. Emit a receipt binding pass, input, output, resource use, and diagnostics.
6. Poison the session after a failure that may have partially mutated state.

The initial pass order is deliberately conservative:

1. unreachable control-flow removal and branch simplification;
2. eligible local-storage promotion to SSA;
3. constant propagation and folding;
4. dead value, operation, argument, helper, and symbol elimination;
5. equivalent pure-computation reuse;
6. bounded aggregate decomposition and helper integration;
7. loop normalization and explicitly bounded unrolling;
8. address-space refinement and memory-effect analysis;
9. uniformity, divergence, barrier, and synchronization validation;
10. ABI preparation and target-independent call lowering.

No optimization is required for semantic correctness. A pass may reject or
leave code unchanged, but it may not select an old compiler route.

## Proof and verification

Verification does not select compiler implementation. The current MIR-to-KIR
receipt binds the exact semantic MIR identity to the production KIR wire
version, digest, and byte length. It retains complete block, statement,
terminator, synthetic-operation, and parameter correspondence plus the exact
semantic induction report. The formal-memory receipt independently binds the
same versioned KIR identity to the complete canonical obligation receipt and
its structural witness.

For gfx942, the independent proof-input validator strictly decodes canonical
KIR V8, checks complete contiguous operation coverage, replays the induction
analysis deterministically, and requires each admitted induction certificate
to name exactly one checked KIR addition in its retained source span. Hostile
span reassignment, report mutation, parameter rebinding, synthetic-trap drift,
and independently well-formed KIR identity substitution all fail closed. A
workload-specific Verus proof may discharge obligations for that module, but it
cannot replace MIR or Kernel IR or provide artifact authority.

The #106 General GEMM proof is therefore the first substantial producer of a
generic MIR-to-KIR correspondence receipt. The #174 consumer retains that
receipt with the same MIR owner. Later kernels use the same receipt type and
relation vocabulary with different proved obligations.

Source proof, compiler transformation validation, LLVM/ISA correspondence,
machine inspection, hardware observation, and runtime authority remain
separate evidence classes.

## AMDGPU and finalization

One AMDGPU lowering owner centralizes exact target identity, features, wave
policy, address spaces, device libraries, code-object policy, resource bounds,
kernel metadata, calling conventions, and module flags. Textual LLVM is a
bounded Worker transport and inspection form, not a semantic identity boundary.

The production finalizer returns a generic move-only inspected-artifact owner.
It retains and freshly revalidates the exact compiler graph/handoff, Worker,
finalized bytes, descriptor, ELF, metadata, target, and ISA observations. The
current General GEMM owner chain under #173 is a qualification oracle for this
generic boundary. Its three late-machine axes must not become a second
GEMM-only authority path.

Generated host interfaces derive from the canonical Kernel IR ABI and are
checked against the inspected descriptor. They still grant no launch authority;
the runtime separately validates allocation, lifetime, launch geometry, and
device compatibility.

## Selector retirement

Cargo has no selector. It rejects both the obsolete pipeline and qualification
oracle environments rather than interpreting absence as a route choice.
Versioned `V1`/`V2`/`V3` suffixes identify frozen records and protocols, not
selectable implementations. Production build inputs use only
`FE2O3_PRODUCTION_BUILD_CONFIG_V1` with the
`fe2o3-production-build-config-v1` schema. Worker V2 config,
expected-identity, envelope, and source-debug controls are recognized only for
fail-closed rejection.

The Cargo qualification feature and executable branches are deleted. Keeping
a route behind `cfg(feature)` would be isolation, not convergence. Default and
all-feature tests compile the same production route; `cfg(test)` cannot
activate alternate Cargo compiler behavior. Backend-only differential oracles
remain temporary work under the backend deletion ledger below.

### Variant deletion ledger

| Area | State | Required deletion |
|---|---|---|
| Host and HSA launch | Complete | Worker V2 host admission, workload launch adapters, raw HIP loading/packing/launch, and compatibility aliases are deleted. |
| Cargo production tests | Complete | Default tests compile `PreparedProductionBuildConfig`, `ManagedProductionBuild`, Worker V3 application handoff, and the fixed device-then-host plan. |
| Cargo qualification graph | Complete | The feature, simulation command, Worker V2 build/restart modules, S09 routing, workload parsers, fixture binaries, vertical tests, and Worker V2 bundle dependency are deleted. Worker V3 application fault coverage uses a compiler-neutral test feature. |
| Backend qualification graph | In progress | Default and all-feature backend libraries are feature-invariant, selector-free, and enter only `ProductionCompilation`. Exact-profile and Worker V2 modules are restricted to feature-enabled unit-test fixtures and cannot be selected by a rustc invocation. Preserve the minimal canonical differential fixtures, then physically delete the remaining modules and backend feature. |
| Finalizer and Worker execution | Open | Move shared protocol, request, admission, finalization, and executor mechanics to workload-neutral owners; migrate V3 callers; delete V2 executable APIs and workers. |
| Artifact restart compatibility | Open | Retain only versioned records needed for canonical decode, explicit rejection, or migration; delete V2 publication/recovery actions once V3 differential coverage owns their hostile cases. |
| Pliron and workload workers | Open | Replace Worker V2 executable bridges with production handoff fixtures or offline comparisons, then delete the worker crates and profile entry points. |
| Versioned schemas | Retained by design | V1/V2/V3 remains on frozen bytes, receipts, domains, and protocol records only. A suffix must not imply a selectable implementation. |

Migration follows these rules:

1. Add no workload-specific production implementation or selector.
2. Move exact-profile evidence into inert differential fixtures or offline
   tools, then delete the executable entry points from production crates.
3. Migrate a semantic slice only after ordinary attributed Rust passes the
   production transaction and differential tests match its existing oracle.
4. Once a slice migrates, preserve only the smallest authority-free fixture
   needed for differential coverage and delete the old implementation.
5. For a kernel-containing crate, unsupported production behavior is terminal.
   `legacy-v1` and exact-profile selectors are never fallbacks.
6. Host-only Rust code may continue through rustc LLVM; that is not a second
   device compiler implementation.
7. Keep non-authoritative comparisons only in offline qualification tooling. The
   compiler API has no implementation selector, and exact-profile qualification
   oracles retire as their differential coverage migrates.

Production became the sole unselected compiler transaction after the first scalar slice
completed its compile, host-interface, artifact, and hardware gates. It has no
selector. An incomplete production transaction now fails closed instead of silently
entering legacy codegen. Backend workload oracles are absent from default, all-feature, and test builds. `FE2O3_QUALIFICATION_ORACLE_V1` is rejected as backend configuration. Unselected host-only dependency units omit fe2o3's managed rustc
arguments and backend descriptor so rustc uses its built-in LLVM backend
directly.

The 2026-08-20 compiler review made this distinction structural. Production has no qualification table or corresponding
variant or selector. The backend has one protected publication call, Cargo has
one production intake without a schema selector, and production recovery is a
separate state machine from V1/V2 qualification recovery. Frontend-record validation does not weaken the boundary: `ProductionCompilation` receives only the move-only production closure and has no oracle helper. See
`compiler-convergence-review-2026-08-20.md` for the deletion inventory and
remaining complexity bounds.

## Migration order

The vertical slices migrate through the same transaction in this order:

1. fill and vector arithmetic;
2. scalar arithmetic, branches, and structured control flow;
3. loops, helpers, cross-crate generic and const-generic calls;
4. multiple kernels in one final crate;
5. global, private, and workgroup memory;
6. barriers, one wave operation, and scoped atomics;
7. scalar GEMM and parameterized tiled GEMM;
8. reductions, softmax, attention, and MoE.

For each slice, the old implementation becomes a differential oracle. Tests compare
canonical MIR, Kernel IR, ABI, artifact structure, numerical results, canaries,
synchronization behavior, and terminal cleanup before its selector is removed.

## Active issue alignment

| Issue | Role in the one production pipeline |
|---|---|
| #140 | Owner-authenticated graph handles and sealed transformation execution |
| #174 | Workload-neutral same-session MIR owner; General GEMM is its first demanding consumer |
| #106 | First mechanically checked producer of the generic MIR-to-KIR correspondence |
| #145 | Typed general AMDGPU to LLVM construction, not artifact authority |
| #146 | Pinned upstream LLVM and in-process LLD Worker consumer |
| #147 | Differential and hostile qualification for the LLVM/Worker boundary |
| #173 | General GEMM oracle for retained compiler/Worker/finalizer ownership and late-machine binding |
| #175 | Production transaction integration, migration order, and selector retirement |
| #176 | One workload-neutral rustc semantic MIR importer for both entry paths |
| #177 | Canonical semantic MIR to general Kernel IR lowering |
| #178 | Owner-authenticated deterministic middle-end transformations |
| #179 | Generic retained finalization and inspected AMDHSA artifact owner |
| #180 | Typed host-interface generation from canonical KIR and inspected ABI |
| #181 | Differential migration and exact-profile selector retirement |

## Parallel implementation lanes

Work remains parallel only at frozen ownership boundaries:

| Lane | Primary write ownership | Exit criterion |
|---|---|---|
| Session safety | `fe2o3-pliron`, owner-handle tests | #140 sealed pass execution with poisoning and receipts |
| Rust import | `fe2o3-mir-model`, `dialect-mir`, rustc importer module | both rustc entry paths return the same generic MIR owner |
| MIR to Kernel IR | `fe2o3-lower-mir-kernel`, correspondence tests | general `KernelModule` for the first scalar/control-flow slice |
| Kernel/GPU passes | dialect and lowering services | deterministic checked pass sequence over owner handles |
| AMDGPU/LLVM | `fe2o3-amdgcn-model`, production target lowering | complete typed target contract and canonical handoff |
| Worker/finalizer | Worker handoff and `fe2o3-hsaco-finalize` | generic retained inspected-artifact owner |
| Host/runtime | generated host and protected runtime adapters | ABI/descriptor agreement and one-shot checked launch |
| Migration/oracles | integration tests, scripts, evidence docs | each old selector removed after differential hardware gates |

Shared root manifests, exports, selectors, and the production transaction are
owned by the integrator. Lane changes merge only after their canonical records
and hostile fixtures are frozen, preventing parallel work from creating new
routes.

## Critical milestones

The current checkpoint has completed the feature-invariant backend entry and
the first source-authentic workgroup vertical slice. Scalar GEMM reaches
deterministic gfx942 LLVM with checked induction custody. Dynamic tiled GEMM and
attention currently stop earlier because their uniform loop bounds are not yet
admitted as exact total unsigned index expressions. The ordinary attributed
WG64 `i32` LDS reduction continues through the compiler-bound handoff, measured
upstream LLVM target APIs, in-process LLD, and inspected COV6 HSACO. Its exact
256-byte LDS and launch-resource contract survives every compiler and artifact
stage. The same route also requalifies the scoped atomic kernel. Neither path
uses COMGR, a shell linker, or a workload-profile selector, and neither
currently grants load or launch authority.

1. **Compiler middle end, bounded operational:** one importer carries the LDS
   slice through semantic MIR, ranked PLIRON, general Kernel IR, composed
   memory checks, and deterministic AMDGPU LLVM. General pass coverage remains.
2. **First production code-object slice, complete:** attributed LDS-reduction
   Rust reaches reproducible inspected gfx942 HSACO through only the production
   compiler and finalizer transaction.
3. **Safety semantics, in progress:** bounded references, LDS, barriers, and
   scoped atomics use the same transaction with hostile tests; general race,
   alias, convergence, and address-space proofs remain.
4. **Rust and verification:** current gfx942 receipts bind exact semantic MIR,
   canonical KIR V8, complete operation spans, formal obligations, and replayed
   checked-induction anchors. KIR-to-LLVM and machine refinement remain open;
   no profile-selected semantic replacement remains.
5. **Parameterized GEMM:** ordinary attributed Rust GEMM reaches inspected
   HSACO through the production transaction; #173 remains only an oracle.
 6. **Worker V3/KFD execution:** the source-bound artifact enters the sole
    application/verifier graph and pure-Rust KFD packet submission path.
7. **Selector convergence:** all exact-profile production selectors are gone,
   default kernel compilation uses the one transaction, and unsupported code
   fails without fallback.

No milestone changes a parity row until its protected evidence policy and
hardware gates independently qualify that row.

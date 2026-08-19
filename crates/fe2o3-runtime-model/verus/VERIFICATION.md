# Runtime model verification

This directory contains the initial issue #137 Verus specifications. The
authenticated runner proves thirty-seven obligations over finite abstract values
and traces. The materialization input and image sequences are capped at 64 MiB
and its phase trace has exactly four entries. The lifecycle-history sequence
lengths are not bounded by these proofs.

`runtime_lifecycle_v1.rs` proves:

1. a retaining dispatch is bound to the exact VM, physical-device identity,
   and device generation carried by its referenced mapping; and
2. releasing a mapping preserves the runtime invariant when no prepared,
   published, or ambiguous dispatch retains that mapping.

`device_identity_generation_v1.rs` proves:

1. registering a fresh device generation preserves unique active generations;
2. registering a VM preserves its exact active device-generation binding;
3. an active VM cannot be substituted onto another generation of the same
   physical device; and
4. while a current generation is active, that generation or an older one
   cannot be reused as a fresh admission.

`device_projection_refinement_v1.rs` proves the pure boundary introduced for
the executable adapter:

1. the model projection retains every field represented in the formal
   canonical observation, including the literal V1 profile and UAPI-schema
   identities, initial wrapping VRAM-loss counter, and contracted reset-fence
   facts;
2. a canonical record satisfying the explicitly modeled V1 predicates projects
   to a model value satisfying the same exact profile/schema identities and
   contracted currentness facts;
3. the projection preserves the explicit 1-through-16-entry topology
   inventory, its pairwise physical/KFD/render/PCI identity uniqueness, and the
   unique selected-device match without replacing the inventory with an opaque
   hash;
4. appending a later generation preserves its exact predecessor link and the
   single-physical-device history invariant.

`memory_lifecycle_v1.rs` proves the initial R2 pure memory obligations:

1. a mapping retains the exact VM, physical-device generation, allocation ID,
   allocation generation, opaque-handle observation, and canonical bounded
   device set represented in the formal binding;
2. a failed map records exactly the reported successful device prefix;
3. a failed unmap treats `n_success` as an absolute cumulative prefix, assigns
   that value without adding prior progress, and retains the unreported suffix;
4. a failed unmap reporting the full prefix remains ambiguous and retains its
   prior conservative range;
5. a substituted device set produces no map state; and
6. any non-released mapping or live publication blocks allocation free.

`queue_lifecycle_v1.rs` proves the initial R4 compute-AQL queue obligations:

1. one concrete canonical four-resource plan exists, preventing the resource
   predicates from being accepted only vacuously;
2. ring, control, end-of-pipe, and context-save resources retain their exact
   composite VM, allocation ID, allocation generation, mapping ID, and
   publication identities and are pairwise distinct;
3. a successful create transition retains the exact plan, queue generation,
   resource sequence, configuration, and opaque returned queue ID, including
   queue ID zero;
4. CREATE status, unchanged/caller-labeled sentinel, returned `u32::MAX`, and
   known-ID collision conditions select Active versus fail-closed Ambiguous
   outcomes exactly; indeterminate CREATE retains a classified returned ID;
5. indeterminate update, disable, and destroy observations become Ambiguous only
   from their matching legal pending phases and retain the exact resources,
   configuration, and prior queue ID; CREATE is excluded from this generic
   relation and is covered only by the field/status-aware relation above;
6. CancelledBeforeCreate and Destroyed are exactly the two non-retaining
   terminals, reached by plan cancellation and successful destroy respectively;
7. generic memory release cannot discharge a live publication structurally
   owned by an exact queue VM, instance, and generation;
8. every retaining queue blocks release of each exact composite mapping in its
   plan;
9. an ambiguous known queue ID remains reserved, while CreatePending or any
   number of Ambiguous states with no known ID poison process-level future
   CREATE; only CreatePending itself is globally single-flight; and
10. appending a history event preserves the exact prior sequence as a prefix
    and places the new event at the next index.

`load_plan_v1.rs` proves the initial R3 abstract load-plan relation:

1. every admitted segment retains the exact 4 KiB page-rounding equations,
   checked `u64` file, memory, and mapping ranges, containing mapping range,
   and the plan retains a checked image span no larger than 64 MiB;
2. the three segments are in canonical increasing virtual-address order, are
   pairwise disjoint in file, memory, and page-rounded mapping ranges, and have
   exactly one each of read-only, read-execute, and read-write permissions; and
3. an admitted descriptor has the same file-to-virtual-address delta within
   exactly one same-permission containing `PT_LOAD` segment.

`materialization_v1.rs` proves the next R3 abstract materialization operation:

1. the three canonical source and destination ranges use checked offset/end
   arithmetic, remain within the 64 MiB input and image bounds, and have
   pairwise-disjoint destinations;
2. the deterministic full-zero transition creates an image of the requested
   length with every byte equal to zero;
3. the deterministic copy-range transition writes the corresponding exact
   source byte at every destination index and preserves every byte outside the
   destination range;
4. for every canonical bounded three-segment plan and exact-length input, the
   defined zero/copy-first/copy-second/copy-third execution has all four states
   at the exact image length and its final byte at every index follows the
   corresponding deterministic transition;
5. the completed execution therefore places every byte from each of the three
   exact input source ranges at its checked destination;
6. every checked range disjoint from all three copy destinations remains zero;
   and
7. the modeled mapping prefixes, in-memory suffixes including BSS, modeled
   mapping tails, and inter-segment gaps satisfy that derived zero-preservation
   property; and
8. one concrete canonical three-segment plan is inhabited, and its constructed
   final image contains both a nonzero copied byte and an uncopied zero byte.

The materialization model receives already-formed mapping ranges. No theorem in
this file composes `MaterializationPlanV1` with
`load_plan_v1::canonical_load_plan_v1`, imports the separate load-plan
invariants, or proves that its mapping starts and sizes use 4096-byte rounding.
The 4 KiB and page-rounded properties listed above remain claims of the separate
`load_plan_v1.rs` proof only.

Run the proofs and all expected-negative mutations with the exact Verus
release whose executable, complete release closure, version, proof sources,
source checker, transcript, and mutations are pinned under `verus/pins`:

```sh
VERUS=/absolute/path/to/verus \
  crates/fe2o3-runtime-model/verus/verify-verus.sh
```

`scripts/ci-local.sh verus` invokes the same authenticated runner. The
`runtime-model-verus.yml` pull-request workflow downloads the named release and
then relies on this runner's executable and complete-closure pins before any
proof result is accepted.

The mutations must fail at their named postconditions: release while retained,
VM generation substitution, stale generation reuse, topology/render PCI
substitution, dropped DRM schema identity, lost history predecessor, mixed
cross-source identity, a dropped final reset-fence observation, allocation free
while a partial mapping remains, cumulative unmap progress incorrectly added to
prior progress, a failed full-prefix unmap treated as releasable, a queue
resource substituted across roles, an indeterminate queue destroy treated as
releasable, queue history overwritten instead of appended, a caller-labeled
returned CREATE sentinel accepted as a queue ID, load segments whose memory
bytes are disjoint but whose rounded pages overlap, generic release ignoring a
queue publication owner, ambiguous queue-ID reuse, allocation-generation
substitution in a mapping identity, ambiguity entered from a non-pending
operation, generic indeterminate handling incorrectly admitting CREATE, a
second CREATE beginning while the first ID is unresolved,
cancellation incorrectly retaining resources, descriptor
containment that substitutes a different file-to-virtual-address delta, the
production-shaped copy transition substituting another source byte, and the
production-shaped zero-first transition omitting the first zero byte. The launcher
rejects source substitution, lexically audits all proof files for trusted
constructs, clears the environment, bounds execution time, pins Z3 through the
authenticated Verus release closure, and rechecks the authenticated inputs after
verification.

The projection proof establishes the mathematical relation implemented by the
pure canonical-record mapping; it is not a proof that the executable Rust
implements that relation, nor that the adapter observed truthful kernel data.
The lifecycle, memory, and queue files prove abstract transition relations, not
refinement of `src/model.rs`, `src/device_identity.rs`,
`src/memory_lifecycle.rs`, or `src/queue_lifecycle.rs`. All
receipts remain model-only and are not production device authority. A later
sealed adapter refinement must
authenticate the KFD topology, DRM render, partition, schema, and process XNACK
observations, bind the dynamically allocated KFD device node to the opened file
descriptor and sysfs device, and connect concrete ioctl/sysfs results to the
canonical record. `DeviceGenerationV1` is a software admission
incarnation for stale-token rejection; topology correlation does not detect or
attest a GPU reset. The reset booleans and wrapping VRAM-loss value are retained
contracted observations only; these proofs do not establish an all-reset
generation, ABA freedom, or correctness of the KFD event stream. Firmware
execution, hardware completion, progress, liveness, coherency, performance, and
absence of kernel/firmware defects remain outside this proof boundary. The R2
proofs do not establish executable-Rust refinement, VA reservation or native
allocation success, KFD `n_success` truth, syscall rollback, CPU/GPU coherence,
page-table state, or quiescence. An adapter must turn malformed or uncertain
side-effecting results into the model's unreleasable ambiguous state. The model
also does not prove that a copied R1 admission token is still active in a
separately evolved `DeviceIdentityStateV1`; that state-composition refinement is
required before a production adapter can consume the memory transitions.

The R4 model is bounded to sixteen queue incarnations, 256 append-only history
entries, and exactly four mapped resource roles. Per-role memory kind,
coherence, and access expectations are explicit plan inputs checked against the
R2 memory state; this proof does not assert that those policy choices are KFD,
gfx942, or ROCr hardware truths. The unchanged CREATE output sentinel and a
caller-labeled returned sentinel are both rejected, while zero is a valid
returned ID. Executable publications carry structural Generic versus exact
ComputeAqlQueue ownership; the generic public memory transition cannot release
the latter. CreatePending and unknown ambiguous CREATE results poison later
CREATE for the process lifetime of this model, and known ambiguous IDs remain
collision-retaining. No model value creates or owns a
native KFD queue. The proof does not establish executable-Rust
refinement, ioctl truth or atomicity, queue-ID ownership, target resource sizes,
doorbell mapping or arithmetic, AQL packet publication, completion, quiescence,
firmware execution, liveness, or performance. Those remain adapter, target
profile, dispatch, and hardware-refinement obligations.

The R3 load-plan and materialization proofs establish only the stated
mathematical relations over already-formed abstract values. They do not prove
that `fe2o3-amdhsa-loader::plan` implements those relations or that its
untrusted ELF byte parser constructs the modeled records. Given a canonical
abstract plan and exact-length mathematical input sequence, the materialization
proof constructs the zero and three copy states rather than assuming a final
image relation. Those sequences and transitions are still mathematical values,
not evidence that executable Rust performed them. In particular, the proof does
not refine `ValidatedEnvelope`, `MaterializationPlan`, slice identity, a CPU or
GPU copy, allocation identity, or any syscall to the abstract operation. It does
not decode or verify metadata or symbols, execute relocations, authenticate
content, establish W^X transitions, or prove materialization on a GPU. Separate
executable parser/copy/syscall refinement and loaded-image proofs remain
required before any `loader_refined` authority claim. There is also no theorem
connecting the materialization plan to the separately proved load-plan profile,
so the materialization proof alone establishes neither 4 KiB alignment nor page
rounding.

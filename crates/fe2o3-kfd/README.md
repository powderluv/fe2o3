# fe2o3-kfd

Owned syscall adapters for the direct-KFD fe2o3 runtime. The initial slice is
deliberately limited to opening `/dev/kfd`, querying its UAPI version, and
producing checked admission evidence for the exact reviewed schema in
`fe2o3-kfd-uapi`. The R1 topology slice additionally provides strict,
read-only discovery of the kernel-owned KFD sysfs tree for the initial `gfx942`
profile. It bounds every read, node/property count, and parsed field; rejects
symlinks, non-regular inputs, duplicate identities, malformed values, and
unknown property keys; and records topology generation plus filesystem and
platform provenance. Default-host discovery additionally records a strictly
parsed boot UUID, bounded kernel release, optional `amdgpu` module version
identities, and opaque KFD firmware-version observations, then correlates every
render minor to a kernel sysfs link below `/sys/devices`.
KFD unique ID, PCI domain/location, vendor/device ID, PCI revision, render
device number, and typed compute/memory partition observations are captured or
must agree. The initial admission layer can require the exported `SPX/NPS1` V1
partition constant without losing the observed values. The fixed
kernel-owned render and PCI symlinks are resolved deliberately; symlinks in
the KFD topology tree and regular-file inputs remain prohibited.

The public safe API does not expose file descriptors or raw ioctl arguments.
The R1 composition path consumes an explicitly selected unique ID and returns a
non-cloneable `CheckedGfx942XnackMinusDevice`. It retains `/dev/kfd` and the
exact correlated render descriptor plus a prospective KFD whole-GPU reset-event
descriptor, owns a process-global fe2o3 admission lease, requires KFD 1.18 and
AMDGPU DRM 3.64.0, compares the DRM identity prefix with topology/sysfs,
establishes a disabled-XNACK no-queue barrier, checks the complete bounded
process-aperture inventory, and repeats process, descriptor, topology, XNACK,
aperture, and reset-event observations before committing the token. The
`DEVICE_ADMISSION_PROFILE_MANIFEST_V1` digest binds the exact checked profile
and claim boundary. Retired model history is retained across admissions in the
same process and observation domain; a poisoned history fails closed. Each
successful admission also retains a solver-neutral `DeviceProjectionRecordV1`
covering platform, module-filesystem, and process provenance, both descriptors
and UAPI schemas, the selected topology/DRM profile fields, the explicit
bounded full-GPU identity inventory, firmware and selected capacity
observations, the initial wrapping VRAM-loss counter, the complete process
aperture inventory, and explicit reset-subscription, event-mask, `CLOEXEC`,
post-subscription DRM equality, and initial/final clear-fence facts. These are
contracted currentness observations, not an all-reset generation or proof.
Projection history is
updated atomically with identity history and links each admission generation to
its exact predecessor. R1 deliberately retains, rather than compacts, at most
`MAX_MODEL_DEVICE_ADMISSIONS_V1` admissions for the process lifetime. After the
first admission, any observation-domain change fails closed with
`ModelDomainChangedWithActiveHistory` or
`ModelDomainChangedWithRetainedHistory`; it never replaces retained history.
The sixty-fifth bind fails with `ProjectionHistoryExhausted`. Restarting the
process is the only supported way to create an empty history. This reviewed
availability bound avoids silently discarding substitution evidence. The
`kfd-device-identity` example performs this no-queue admission.

`check_observable_currentness(&mut self)` sandwiches a complete reobservation
between checks of the retained reset-event descriptor. Any event or error
permanently poisons later checks. It also compares the wrapping DRM
`VRAM_LOST_COUNTER`, but never treats that counter or KFD topology generation as
an all-reset generation. Under the pinned driver contract this detects
subscribed whole-GPU resets, VRAM-loss resets, and all changes visible through
the admitted identity, process, descriptor, XNACK, aperture, and topology
queries.

This crate checks userspace schema admission and encapsulates descriptor
ownership. Verus proves the pure canonical-record projection and abstract
generation/history relations. The executable validator checks the same record,
but there is not yet a Verus proof of the Rust implementation or a syscall-to-record
refinement proof, nor a
`ProductionDeviceAuthorityV1` implementation. No R1 API grants VM, allocation,
mapping, queue, event, code, or dispatch authority. It does not enumerate
cache, memory-bank, or link subtrees or prove their reported counts. The
process-global lease excludes other fe2o3 R1 admissions, not arbitrary raw KFD
users in the process. Ancestor traversal, mount-namespace integrity, sysfs
truth, cross-file snapshot semantics, KFD/DRM ioctl behavior, firmware meaning,
and absence of an ABA reset remain named external contracts. KFD does not expose
a sequence snapshot for the prospective subscription, does not report every
engine/per-queue reset through that stream, and creates its anonymous event fd
with an empty mask and without an atomic `CLOEXEC` option. A reset can therefore
occur between descriptor creation and mask enablement. The adapter sandwiches
that enablement between DRM identity/VRAM-counter observations, sets `CLOEXEC`
immediately, and never drains the complete event after detecting its first byte,
but a VRAM-preserving reset in the enablement gap can remain unobservable. It
also cannot close the concurrent
fork/exec inheritance window or exclude interference from arbitrary raw KFD
users in the process. A retained-device, nonwrapping counter incremented for
every reset class plus an atomic create/mask/CLOEXEC operation, or an atomic
generation-snapshot/event handshake, is required for an all-reset currentness
proof. Successful kernel responses and node metadata are checked or Contracted
observations, not proof of the kernel or hardware implementation.

The separate `scripts/runtime-identity-oracle.sh` hardware lane compares the
`kfd-device-identity --all` evidence with bounded output from an isolated
`/opt/rocm/bin/rocminfo` subprocess. A match is recorded only as `Measured` with
`authority=none`; oracle output is never passed to this crate and cannot create
device, VM, memory, queue, dispatch, or proof authority. The exact comparison,
evidence schema, CI separation, and limitations are documented in
`docs/runtime-identity-oracle-v1.md`.
The evidence marks contracted currentness and the VRAM-loss counter as
pure-Rust-only observations; neither is represented as an HSA differential
match.

## R2 host-visible memory slice

CheckedGfx942XnackMinusDevice::acquire_host_visible_memory_session consumes
the selected device and makes one irreversible ACQUIRE_VM attempt for the
process. A successful HostVisibleMemorySession owns the retained KFD/render
files and admits one ordinary, single-device, host-visible coherent GTT
allocation. The adapter rounds a nonzero requested length to a checked
4096-byte footprint, obtains a temporary anonymous address reservation, checks
the entire half-open interval against the selected process GPUVM aperture, and
passes that fixed GPU VA to ALLOC_MEMORY_OF_GPU. It rejects any mutation of
the input fields, zero handle/offset, unaligned offset, overflow, or profile
flag mismatch.

GPU VA, the opaque allocation handle, and the CPU VMA remain separate private
authorities. After successful ALLOC, the temporary address reservation is
unmapped. The BO is then mapped through the retained selected render file at a
kernel-selected CPU address with MAP_SHARED and PROT_NONE. MADV_DONTFORK must
succeed before mprotect enables read/write access and before any safe
closure-scoped byte borrow can be formed. Failed madvise or mprotect setup is
synchronously unmapped; failed cleanup is process-fatal rather than returning
an ambiguously inheritable VMA.

The mmap-to-DONTFORK step is not atomic. Absence of an external raw fork or
clone during that interval is Contracted; this API does not claim atomic
no-inheritance. Every borrow requires mutable session authority and checks the
opener PID and observable currentness before and after the closure. A reset
concurrent with the closure remains Contracted. CPU borrows are unavailable
while the BO is mapped to the GPU. Native KFD handles, GPU virtual addresses,
and descriptors remain private. Safe byte borrows cannot escape their closure,
but safe code can observe and retain a raw address derived from a slice;
dereferencing it outside the borrow requires unsafe code and is an external
contract.

MAP and UNMAP always use an immutable one-element `[selected_gpu_id]` array.
The returned n_success is cumulative and must satisfy old <= new <= 1.
Only ioctl success plus the full prefix commits a phase transition. An errno
with n_success == 1, malformed output, or a failed currentness check after
any mutation permanently quarantines the session. Cleanup requires successful
UNMAP, then CPU munmap, then exactly one FREE attempt. Any FREE error is
terminal because the pinned driver removes validation-list state before all
interruptible failure points. Drop performs no memory ioctl, munmap, FREE, or
retry; normal Rust ownership still closes the retained descriptors and invokes
driver process teardown.

HOST_VISIBLE_MEMORY_PROFILE_MANIFEST_V1 composes the frozen KFD memory schema
with the R1 device profile, active module digest, 4096-byte page profile, and
the reviewed transitive driver-source closure. The source-to-loaded-binary
relationship and kernel behavior remain Contracted. The completion-only model
journal records only fully committed adapter transitions. It is not a history
of every concrete syscall side effect: a
quarantined ALLOC, MAP, or UNMAP path can have unmodeled kernel effects. The
journal is model-only evidence, not production authority or a Verus/concrete
refinement proof.

AQL, executable, kernarg, VRAM, USERPTR, peer-device mapping, multiple
allocations, retry, queues, and dispatch are rejected or absent. The
default-feature `kfd-host-visible-memory-policy` example links and reaches the
complete production memory adapter without enabling process/fork support. CI
builds and ELF-audits that executable under the pure-Rust runtime policy so
dead-code elimination cannot hide the production syscall closure.

The `live-validation` feature is non-production only. It enables the
`kfd-host-visible-memory` example and a single-threaded fork/mincore negative
that verifies the DONTFORK VMA is absent in the child. The example always
launches the selected-GPU transaction in an isolated subprocess and creates no
queue or reset.

## R4 queue-resource observations

plan_gfx942_aql_queue_resources turns one selected, correlated topology
observation into bounded resource geometry for the exact gfx942,
SPX/NPS1 topology profile. It checks every topology field used by the active
KFD/ROCr CWSR formula, the 4096-byte host page, a conservative
ROCr-compatible power-of-two ring range, exact EOP and context-save sizes,
counter mapping geometry, and non-MES doorbell geometry. The plan requires
read-only module-parameter observations mes=0, sched_policy=0, and
cwsr_enable=1; missing or changed values fail closed. Queue ID zero is
explicitly valid: the pinned KFD process queue manager allocates the first zero
bit from a zero-initialized 1024-slot bitmap.

The plan also names the exact reviewed ROCr 7.2.4 backing-policy expressions.
On the reviewed branches, ring and control produce fine-grained USERPTR
profiles, EOP produces executable coarse VRAM, and CWSR requests anonymous host
SVM attributes with a USERPTR fallback. The manifest pins the queue call sites,
runtime allocator dispatch, KFD driver flag translation, KMT allocation
translation, the header definitions of page and huge-page alignment, and
CWSR/EOP expressions needed to derive those values. This is an exact
expression set, not a transitive ROCr policy implementation closure or
evidence that an invocation selected a particular branch. These observations
are not allocations accepted by the current fe2o3 memory authority. USERPTR,
VRAM, SVM, queue creation, doorbell mmap and doorbell stores remain
unsupported. The topology does not export CWSR sizes on the admitted host, so
the plan uses and tests the exact pinned fallback formula. The read-only
kfd-queue-resources example validates the topology-derived facts on every
visible MI300X without opening /dev/kfd or creating a queue.
This topology-only result does not observe process XNACK mode. Its embedded R1
device-profile digest is only a compositional prerequisite identifier, not
evidence that R1 admission occurred. A future queue authority must pair the
plan with a live checked device token that establishes XNACK-disabled
admission and currentness.

# fe2o3 KIR debugger

`fe2o3-kir-debugger` records bounded, execution-derived snapshots from the
deterministic CPU KIR simulator. It supports thread, logical-wave, workgroup,
and dispatch scopes; operation breakpoints; ordinary and atomic memory watchpoints;
forward and reverse transcript navigation; and typed stack, SSA, and memory
inspection.

An integer atomic contributes one allocation-relative memory record, including
whether it was read-only, write-only, or a committed read-modify-write. Atomic
watchpoints therefore stop once per semantic operation; ordinary read/write
watchpoints also include atomics with the corresponding effect. Fences are
captured as scoped memory-order points, not execution barriers.

Reverse navigation moves over an immutable deterministic transcript. It does
not invert writes or claim physical GPU scheduling. Wave32 and Wave64 are
explicit visualization profiles over the canonical local-work-item order.
Debugger capture can replay an admitted persisted semantic schedule through the
same scheduled simulator API. The captured transcript remains immutable;
interactive debugger revisions navigate that transcript and do not replace or
weaken simulator context, decision, coverage, or transcript validation.

Source locations are optional sidecar claims bound to the exact canonical KIR
digest and byte length. The debugger rejects a source catalog whose identity
does not exactly match the simulated module.

## Typed physical-entry CPU observations

`PhysicalEntryDebugSessionV20::capture` accepts the exact admitted simulator
module, its immutable `VerifiedCanonicalKernelIrModuleV20` owner, a simulation
request, explicit capture limits, and the caller's existing owned verification
resource budget. It runs the same CPU Engine; there is no packed-program or
physical-register interpreter.

The move-only result retains the original cumulative budget and every captured
SSA binding, including symbolic kernarg/output pointer halves, scaled offsets,
address chains, and carries. These symbolic values have a kind but **no numerical
GPU address or carry bits**. Numerical EXEC/VCC values are visible only when the
actual interpreter produces numerical SSA values. Allocation-relative pointers,
memory bytes, initialization state, canonical sites, and invocation coordinates
are observations, not native addresses or source authentication.

`seek`, `step_forward`, and `step_reverse` select existing observations. They
never restore an interpreter, undo memory, or resume execution. Borrowed record
views cannot escape as a cloneable owned transcript or mutable snapshot. Destroy
the session with `into_budget` to release its retained capture charges while
preserving the caller's original storage floor, cumulative work, peak and
first-denial history.

Preflight/Engine storage envelopes are prepaid before execution allocations.
Snapshot vector capacities, temporary ordering storage, symbolic bindings, copied
memory/initialization bytes, and a fixed optional cursor header stay accounted on
the same ledger. Capture traversal/sort/copy and navigation work is cumulative;
execution retains its existing simulator step limits. This is logical payload/
capacity accounting, not an allocator or process-RSS limit. Any allocator-reported
capacity mismatch is refused before storage is populated or retained.
A record/snapshot/work/storage cutoff stops observation, not CPU execution; a
completed execution with a stopped capture is not a complete transcript.

This initial V20 typed API does not add a CLI flag, source catalog, breakpoint
predicate path, persisted schedule replay, detailed execution diagnosis,
hardware-register capture, or protected deployment. Legacy capture/transcript
APIs remain refused for the physical-entry profile; they do not acquire this
budgeted admission. This V20 API does not admit V21 global-copy/pending-read input.

The separate [`PhysicalGlobalCopyDebugSessionV21`](src/physical_global_copy_v21.rs)
API captures that V21 CPU profile. Its matching Linux CLI route and explicit
capability limits are documented under
[Physical-global-copy V21 CPU observations](../fe2o3-debug-cli/README.md#physical-global-copy-v21-cpu-observations).
Use that versioned API/route rather than passing V21 input to the V20 API; it
does not provide physical-register capture, native execution, or GPU authority.

The existing serialized debugger value grammar is unchanged. Short-output and
zero-EXEC CPU controls do not discharge the separate production/formal launch
bounds or imply a deployable short-slice kernel.

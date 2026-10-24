# Verus GPU kernel source models

The primary milestone in this directory verifies the executable body of the
real `f32` `#[kernel]` in `examples/vecadd`. The operation, index extraction,
guarded `DisjointSlice::get_mut`, two input indexes, addition, and output write
live once in `examples/vecadd/src/vecadd_body.rs`, with arithmetic supplied by
an explicit adapter. Both the GPU kernel and `verus/vecadd.rs` mechanically
expand that control/index/read/write fragment.

The fragment has two explicit adapter boundaries. The GPU expansion calls
`fe2o3_device::thread::index_1d()` with no argument, while the Verus expansion
passes a modeled launch witness to `model_gpu_thread::index_1d`. The adapter
returns that same identity witness and introduces no `external_body`. Proving
that the target intrinsic returns the corresponding launch witness remains a
backend-refinement obligation. The production arithmetic macro expands to the
exact expression `lhs + rhs`; a source-shape test and the `kernel-ir-v1`
compile lane enforce that fact. Verus instead expands the same arithmetic
adapter position to a total operation over local `ModelFloat` tokens.

For the real shared body, Verus establishes:

- an arbitrary rounded-up thread performs no input index or output write when
  `DisjointSlice::get_mut` rejects its identity witness;
- an in-range witness is in bounds for the output and both equal-length inputs;
- arbitrary in-range model operands require no arithmetic-domain premise;
- `ThreadIndex::get` and the consuming output access select the same index;
- distinct identity witnesses select distinct output elements;
- the guarded write changes no other modeled output element;
- symbolic input-read and exclusive-output regions are compatible; and
- every modeled four-byte element address ends within both its allocation's
  address space and `usize::MAX`.

The `ModelGpuDisjointSlice` adapter owns a `Vec<ModelFloat>` so Verus can reason
about the shared body's mutation and frame behavior for arbitrary values. A
model add XORs opaque tokens to provide a total executable operation, but no
contract exposes that result. Neither `ModelFloat` nor the model add is claimed
to refine IEEE `f32` or production addition. `ModelGpuDisjointSlice` is also not
a refinement of the raw pointer and length stored by
`fe2o3_device::DisjointSlice`. Allocation IDs, base addresses, extents, and
permissions remain caller-supplied ghost facts; the harness does not
authenticate them against Rust references or a launch.

`real_kernel_arbitrary_in_range_operands_are_memory_safe` exposes the in-range
bounds, ownership, and frame guarantees without constraining model operand
values. `real_kernel_rounded_tail_is_noop` composes for an out-of-range thread
without arithmetic or region-evidence premises. No claim is made about the
stored sum, and memory-safety proofs do not depend on the arithmetic result.
The real-body model adds no `assume`, `admit`, or `external_body`.

## Reference proofs

The older `u32` CPU/reference vecadd remains separate in
`src/vecadd_body.rs`, `src/lib.rs`, and the first part of `verus/vecadd.rs`. It
proves an exact per-thread integer result under an explicit no-overflow
precondition. It is not the GPU kernel and is retained as a stronger
target-neutral functional example, not as evidence for `f32` semantics.

The fill harness additionally proves identity indexing, modeled address
representability, disjoint writes, frame behavior, and its launch-level fill
postcondition. Its `hardware_thread_id` model is the one existing
`#[verifier::external_body]`; the real vecadd source model does not call it.

## Copy, affine-map, and gather pairs

`src/elementwise_bodies.rs` contains three more single-source executable
fragments. Ordinary Rust expands them through `src/lib.rs`, and
`verus/elementwise.rs` expands the same macros through verified adapters:

- copy proves an identity read, identity write, and exact frame update;
- affine-map proves `value * scale + bias` and its exact frame update; and
- gather proves the selected input index is in bounds before an identity write.

The affine source intentionally accepts `i16` values and scale, an `i32` bias,
and writes `i64`. Widening makes the mathematical integer operation total and
exact for every value in those input types, without an overflow premise. This
is not an IEEE floating-point theorem: it says nothing about `f32`/`f64`
rounding, NaNs, infinities, signed zero, fused operations, or target contraction.
The real vecadd `f32` adapter therefore remains opaque to functional proofs.

Each positive elementwise theorem has exactly one paired mutation under
`verus/negative/`: copy reads element zero, affine-map adds one to the bias,
and gather reads element zero. The runner requires each mutated theorem to
reach Verus, emit one primary proof error, and fail at its exact postcondition;
a parser error or an unrelated failure does not count as a successful negative.

## General typed two-kernel slice

`src/two_kernel_bodies.rs` adds two ordinary-Rust/shared macro bodies used as
source models for the bounded general typed two-kernel slice:

- `alpha(scale: f32, input: &[f32], output: DisjointSlice<f32>)` computes
  `output[i] = scale * input[i]`; and
- `zeta(a: &[f32], b: &[f32], bias: f32, output: DisjointSlice<f32>)` computes
  `output[i] = a[i] + b[i] + bias`.

The ordinary CPU test model expands those fragments over actual `f32` slices.
They are not `#[kernel]` entry points, are not lowered to AMDGPU, and are not
evidence about a generated gfx942 code object. The positive Verus harness in
`verus/two_kernel.rs` expands the same guarded index/read/write fragments over a
bounded integer arithmetic abstraction. It imports the reusable, axiom-free
allocation, byte-region, permission, and identity-index lemmas from
`verus/permission_core.rs` and proves, for both modeled signatures:

- the output guard dominates every input access, including rounded launch
  tails;
- every modeled four-byte access is within its allocation and ends at or below
  `usize::MAX`;
- input capabilities are initialized shared reads;
- the owned output element is initialized by the exclusive write;
- different active thread identities have disjoint modeled output regions; and
- the resulting element and frame equal the exact bounded mathematical alpha
  or zeta expression.

The bounded arithmetic abstraction is deliberately not an IEEE-754 theorem.
Proving that production `f32` multiply and add refine it, including rounding,
NaNs, infinities, signed zero, contraction, and operation order, remains an
authenticated compiler/backend refinement obligation.

The exact `gfx942:xnack-` identity belongs to the sealed review capsule and its
artifact join, not to these CPU/shared bodies. Separate exact attributed
alpha/zeta kernel profiles have compiler-derived ABI, Kernel IR, Worker V2, and
bounded artifact/runtime evidence. None of that evidence proves that those
kernels refine these shared bodies or this Verus model, and the ignored
hardware lanes do not turn this proof into machine-code, memory/race-safety, or
GPU-execution evidence.

The input-initialization premise and four-byte address-representability result
are named explicitly in `alpha_input_initialization_assumptions`,
`zeta_input_initialization_assumptions`, and
`f32_access_address_is_representable`. Initialization is consumed as a ghost
assumption; it is not derived from a Rust reference or a runtime allocation.

Five paired negative fixtures require Verus to reject an alpha arithmetic
mutation, an input read moved before the output guard, duplicate exclusive
ownership of one output element, and a shared-read capability whose explicit
initialization state is false. A fifth mutation claims that a four-byte element
starting at the maximum `usize` address has a representable end address.

## Running the checks

Run the ordinary Rust tests with:

```text
cargo +stable test --manifest-path examples/verus_vecadd/Cargo.toml
cargo test -p fe2o3-vecadd
```

Run all seven positive Verus harnesses and all twenty-eight expected proof
rejections with:

```text
VERUS=/absolute/path/to/verus examples/verus_vecadd/run-verus.sh --require
```

The historical runner still prints the stale summary `6 proof harnesses, 26
expected rejections`; the individual results cover seven harnesses and 28
intended rejections. The source guards and proof obligations are unchanged.

Run only the exact gfx942 alpha/zeta source-model proof and its five property
mutations with:

```text
PATH=/path/to/rustup/bin:$PATH \
VERUS=/absolute/path/to/verus examples/verus_vecadd/run-alpha-zeta-verus.sh
```

Source sharing and positive/negative pairing can be checked without Verus:

```text
examples/verus_vecadd/run-verus.sh --source-only
```

Without `--require`, the runner always performs those source-shape checks and
then reports a Verus skip when the executable is unavailable. Ordinary Rust
tests remain a separate command and exercise successful results, mismatched
domains, gather bounds failures, extreme affine inputs, both `f32` signatures,
and rounded-tail no-op behavior.

The real-kernel negative mutations independently reject an input read moved
ahead of the output guard, a real shared-body expansion through a constant-zero
thread adapter, and output/input allocation aliasing. For these fixtures the
runner requires exactly one primary Verus error, exactly one verification
summary reporting one error, the exact error class and failed source clause,
and a stable marker. Parser and unrelated proof failures do not pass.

## Remaining refinement gap

This is source-model evidence, not machine-code verification. It does not yet
prove that the model thread witness is the value returned by the AMDGPU
intrinsic, that `ModelGpuDisjointSlice` refines the actual raw device pointer,
that production `f32 +` refines the total model arithmetic adapter, that ghost
allocation facts came from admitted runtime arguments, or that the shared Rust
expansion refines canonical Kernel IR, LLVM, HSACO, and execution. It does not
create or upgrade runtime `Verified` authority. Closing that gap requires
authenticated compiler-generated proof bindings and a refinement chain from
the real types, intrinsic, and arithmetic adapter through the loaded artifact.

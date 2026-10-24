# Exact causal FlashAttention Phase A

This standalone crate contains ordinary attributed Rust `#[kernel]` source for
one fixed causal `B1/H1/N8/D16` FP32 profile, an independent two-pass host
oracle, executable proof-facing contracts, and a real machine-checked
mathematical model.

The bounded compiler profile authenticates the exact attributed source, kernel
root and `FnAbi`, reviewed provider-terminal manifest, and complete reachable
portable-MIR closure. It consumes that source authority to select the closed
causal Kernel IR profile and publishes an inert COV6/Wave64/WG64 Worker V2
handoff with an unresolved `__ocml_exp_f32` import. This is reviewed
source-to-profile correspondence, not a terminal-body or compiler-refinement
proof.

A configured finalizer test is ignored with the exact prerequisite
`requires the measured direct LLVM/LLD worker built for gfx942`. It consumes a
compiler handoff and uses the pinned upstream LLVM target-machine and in-process
LLD worker to produce a reproducible opaque finalization receipt; it grants no
publication, load, launch, or hardware authority. The hardware gate remains
ignored with `requires the production static wrapper, exact measured pins,
protected linear receipt injection, and MI300X` and deliberately fails closed
before load until that wrapper can deliver the linear receipt in-process.

## Machine-checked mathematical model

`verus/flash_attention_v1.rs` is a pinned Verus proof over exact mathematical
integers and rational outputs represented by numerator/denominator pairs. The
pinned run verifies 25 obligations covering the exact `B1/H1/N8/D16` shape,
inclusive lower-triangular causal domain, nonempty causal prefixes, running-max
update bounds, conditional online frame rescaling, exact running denominator
and numerator invariants, positive denominators, correspondence with the
modeled causal reference, tensor indexing and bounds, and total injective
one-writer ownership of all 128 outputs. Ten independently pinned mutations
must each fail its named postcondition.

The retained proof is bound to public source
`af0fd523e3b774377a9c5192cf0511e34fa19735`, not the changed current kernel.
From the repository root, create a new detached worktree and use the pinned
Verus `0.2026.08.02.b677dd5` release closure:

```sh
git worktree add --detach ../fe2o3-proof-flash-af0fd af0fd523e3b774377a9c5192cf0511e34fa19735
(
  cd ../fe2o3-proof-flash-af0fd
  VERUS=/absolute/path/to/pinned/verus examples/flash_attention_v1/run-verus.sh
)
```

The neighboring worktree must not already exist. Expect 25 verified obligations
and ten intended mutation rejections. The current kernel fails the retained
source hash guard; changing that guard would not prove the changed source.

The runner authenticates the exact kernel and proof bytes, the fixed profile
and model-schema identities, every expected-negative source, the Verus
executable, and the complete 190-file release closure before accepting the
verification result.

## Exponential boundary

`exp_weight_v1` is the model's sole uninterpreted transcendental abstraction.
An admitted frame must supply positive exact weights and the pointwise and
aggregate frame-rescaling relations used by the online recurrence. Verus proves
the causal recurrence and rational correspondence conditional on those explicit
premises; it proves no exponential law.

This proof evidence is not an IEEE-754 `f32` or OCML numerical refinement, a
refinement of `src/kernel.rs`, a compiler/Kernel-IR/LLVM/ISA refinement, a
machine-safety proof, a GPU data-race-freedom proof, or a GPU execution result.
The compiler and finalizer evidence above does not supply those missing joins
either.

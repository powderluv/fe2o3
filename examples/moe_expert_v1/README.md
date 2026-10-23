# Exact host-scheduled MoE expert compute V1

This standalone crate starts the bounded expert-compute slice for the public
`T8/E4/K2/C4` router. It consumes the router's route IDs, stable permutation,
inverse map, expert offsets, capacity, and drop sentinel. The routing ABI does
not produce gating weights, so this slice makes them an explicit finite,
nonnegative input in token-major/rank-minor route-ID order.

The host compacts accepted token activations into four zero-padded `16x16`
BF16 tiles, schedules four independent exact `16x16x16` BF16/F32 GEMMs, packs
active expert rows back into routing-slot order, and combines each token's two
route rows in rank order. Dropped routes contribute zero without
renormalization.

`src/pipeline.rs` is an executable host schedule for that exact plan.
`src/oracle.rs` is independent: it evaluates accepted routes directly in
route-ID order instead of replaying the compaction and per-tile loops. Tests
compare every active and padded expert output, every inverse-permuted compact
row, every route-order weight contribution, and every final token output. They
cover empty experts, capacity drops, lower-expert ties, balanced and patterned
data, input immutability, and adjacent canaries.

`verus/moe_expert_memory_v1.rs` is a fixed logical-source model for exact
index bounds, inverse-slot admission, zero-padding separation, disjoint expert,
compact, and combined write owners, and host phase order. The pinned runner
verifies 15 obligations and requires six named mutations to fail at their
postconditions. Its expected evidence is copyable and inert and cannot mint or
join an authenticated receipt. It proves no Rust-source, MIR, Kernel-IR,
LLVM/ISA, logical-address, artifact, machine-memory, generalized race-freedom,
numerical, or GPU-execution result.

This retained model is bound to the original public source at
`e3c7018f428192c350c9c425df8e4c031c788272`, not the current kernel. From a
repository-root checkout, create a new detached worktree and use the pinned
Verus `0.2026.08.02.b677dd5` release closure:

```sh
git worktree add --detach ../fe2o3-proof-moe-e3c701 e3c7018f428192c350c9c425df8e4c031c788272
(
  cd ../fe2o3-proof-moe-e3c701
  VERUS=/absolute/path/to/pinned/verus examples/moe_expert_v1/run-verus.sh
)
```

The neighboring worktree must not already exist. Expect 15 verified obligations
and six intended mutation rejections. The current kernel deliberately fails the
original source hash guard; do not change that guard to claim a new proof.

`src/kernel.rs` contains ordinary attributed Rust `#[kernel]` definitions for
the expert GEMM and deterministic combine. It contains no `macro_rules!`
kernel facade.

## Production integration status

The former MoE V1/V2 host bridges and generated workload-specific adapters were
qualification-only alternatives and have been removed. This crate remains the
ordinary attributed Rust kernel, independent host schedule/oracle, and Verus
memory-model source for the fixed expert GEMM and combine algorithms.

Production integration must emit a normal Worker V3 descriptor and use the
single generic application handoff, generated argument packing, alias
admission, HSA load/resolve/dispatch/unload lifecycle, and physical-resource
checks. That Worker V3 MoE hardware slice is not complete, so this example
currently grants no artifact, runtime, GPU-execution, or parity authority.
Direct Cargo compilation is intentionally not a supported substitute: the
typed router dependency requires the per-crate binding issued by the fe2o3
wrapper. The rustc-codegen integration test below owns that compiler boundary.

Run the retained checks from the repository root:

```sh
python3 scripts/test-bounded-moe-docs.py
VERUS=/absolute/path/to/pinned/verus \
  ./scripts/test-moe-expert-compact-plan-verus.sh
cargo test --locked -p rustc-codegen-fe2o3 \
```

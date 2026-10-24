# GEMM autoresearch program

Your goal is to improve the geometric-mean BF16/F32 GEMM throughput reported by
`evaluate.py` on one explicitly selected supported wave64 GPU without changing the operation it computes.

## Scope

- For TILE, edit only `src/kernel.rs`; for gfx950 SIMT, edit only `src/kernel_simt.rs`.
- Select `--target gfx950 --mode tile|simt` explicitly for gfx950 evaluations.
- Do not edit `evaluate.py`, `src/main.rs`, `src/reference.rs`, `run-gfx942.sh`, `run-gfx950.sh`,
  `Cargo.toml`, or the repository compiler/runtime.
- Preserve the selected symbol (`gemm_autoresearch_v1` for TILE or
  `gemm_autoresearch_simt_v1` for SIMT), its ABI, dynamic dimensions and strides,
  alpha/beta epilogue, edge behavior, output ownership, and wave64 target.
- Never skip the production fe2o3 extraction or independent numerical check.
- Do not specialize on zero benchmark inputs or the published benchmark sizes.

## Loop

1. Read the current kernel and form one concrete performance hypothesis.
2. Make one attributable change in the selected mode's source file.
3. Run `ROCR_VISIBLE_DEVICES=<one-id> python3 evaluate.py --target gfx950 --mode <tile-or-simt> --label <short-name>`.
4. Reject any compile, verifier, lowering, ISA, launch, or numerical failure.
5. Compare the median score across three fresh processes. Keep the edit only if
   the score improves and no size shows a material regression outside noise.
6. Record what changed and why in the untracked `results.tsv` ledger.
7. Continue until the time budget expires. Re-run the winner from a clean build
   before publishing a performance claim.

Useful hypotheses include removing redundant staging, changing reuse geometry,
unrolling the K loop, reducing live fragments, changing workgroup tiling, and
separating edge and interior paths. Every change must remain expressible through
the supported safe fe2o3 device API and pass its generic compiler checks.

The score is a search signal, not a state-of-the-art claim. Shared-host load,
clock state, thermal state, input distribution, and a narrow shape set can all
change the result.

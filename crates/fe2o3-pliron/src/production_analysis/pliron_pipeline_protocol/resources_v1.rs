fn pipeline_resource_overflow_v1() -> ProductionAnalysisResourceLimitV1 {
    ProductionAnalysisResourceLimitV1 {
        phase: ProductionAnalysisResourcePhaseV1::PipelineProtocol,
        resource: "pipeline protocol resource upper bound",
    }
}

fn checked_pipeline_sum_v1(values: &[usize]) -> Result<usize, ProductionAnalysisResourceLimitV1> {
    values.iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(pipeline_resource_overflow_v1)
    })
}

fn checked_pipeline_product_v1(
    lhs: usize,
    rhs: usize,
) -> Result<usize, ProductionAnalysisResourceLimitV1> {
    lhs.checked_mul(rhs)
        .ok_or_else(pipeline_resource_overflow_v1)
}

fn pipeline_equivalence_query_upper_bound_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<usize, ProductionAnalysisResourceLimitV1> {
    if census.pipeline_creates == 0 {
        return Ok(0);
    }
    let accesses_squared =
        checked_pipeline_product_v1(census.ranked_accesses, census.ranked_accesses)?;
    // Coordinate comparison is bidirectional for every create. Aliased
    // creations can each reuse the same access slice.
    // Dynamic vectors retain all occurrences: a canonical window of C rows
    // compares with windows totaling at most A rows, costing at most 2*C*A*M
    // queries, where C<=A and M is the maximum arity. Deduplication is not a
    // premise of this bound, including failed and memo-hit comparisons.
    // A concrete create instead compares each read with source-ordered writes
    // in the same epoch, using at most A^2*arity metered queries (including
    // failed matches). Concrete/dynamic routes are exclusive per create, so
    // the same 2*P*A^2*arity bound covers both without changing any limits.
    let coordinate_queries = checked_pipeline_product_v1(
        checked_pipeline_product_v1(
            checked_pipeline_product_v1(census.pipeline_creates, accesses_squared)?,
            census.max_operation_arity,
        )?,
        2,
    )?;
    let optional_read_queries = if census.index_lt_branch_candidates == 0 {
        0
    } else {
        checked_pipeline_product_v1(
            checked_pipeline_product_v1(census.pipeline_creates, accesses_squared)?,
            census.max_operation_arity,
        )?
    };
    // Every candidate header can perform B fixed-point rounds over all edge
    // operands while propagating its induction value.
    let loop_queries = checked_pipeline_product_v1(
        checked_pipeline_product_v1(census.index_lt_branch_candidates, census.blocks)?,
        census.operands,
    )?;
    let loop_scalar_queries = checked_pipeline_product_v1(census.index_lt_branch_candidates, 3)?;
    let event_queries = checked_pipeline_product_v1(census.pipeline_events, 4)?;
    let dynamic_queries = checked_pipeline_product_v1(
        checked_pipeline_product_v1(census.pipeline_creates, census.ranked_accesses)?,
        7,
    )?;
    checked_pipeline_sum_v1(&[
        coordinate_queries,
        loop_queries,
        loop_scalar_queries,
        // Child and containing-loop invariants each recheck every edge once.
        checked_pipeline_product_v1(
            checked_pipeline_product_v1(census.index_lt_branch_candidates, census.operands)?,
            2,
        )?,
        // Optional reads query initialized coordinates without establishing coverage.
        optional_read_queries,
        event_queries,
        dynamic_queries,
    ])
}

fn pipeline_equivalence_unique_pair_upper_bound_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<usize, ProductionAnalysisResourceLimitV1> {
    let queries = pipeline_equivalence_query_upper_bound_v1(census)?;
    let query_expansions = checked_pipeline_product_v1(queries, MAX_EQUIVALENCE_WORK_V1)?;
    let values = census.results.saturating_add(census.block_arguments);
    Ok(values.saturating_mul(values).min(query_expansions))
}

fn pipeline_concrete_fact_query_work_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<usize, ProductionAnalysisResourceLimitV1> {
    if census.pipeline_creates == 0 {
        return Ok(0);
    }
    let queries = checked_pipeline_sum_v1(&[
        checked_pipeline_product_v1(census.pipeline_events, 2)?,
        census.ranked_accesses,
    ])?;
    // Per query: preflight literal scan, runtime literal read, borrowed fact
    // lookup, at most MAX_RANK coefficients, and one constant remainder. One
    // fixed cache decision per create. Aliased creates can revisit access rows.
    // Cache construction/retention is separately admitted as SparseIndex.
    let each = checked_pipeline_sum_v1(&[
        1,
        checked_pipeline_product_v1(queries, dialect_kernel::MAX_RANKED_MEMORY_RANK + 4)?,
    ])?;
    checked_pipeline_product_v1(census.pipeline_creates, each)
}

// The census cannot yet distinguish dynamic, same-block and cross-block
// creates, so every create reserves the bounded concrete CFG attempt. No cap
// changes, certificate storage or per-pipeline retained graph is introduced.
fn pipeline_concrete_cfg_resource_delta_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<(usize, usize), ProductionAnalysisResourceLimitV1> {
    if census.pipeline_creates == 0 {
        return Ok((0, 0));
    }
    let sites = checked_pipeline_sum_v1(&[1, census.pipeline_events, census.ranked_accesses])?;
    // One ordinal-control/reachability/topology walk, then at most one dense
    // site registration, order construction and cursor pass per creation.
    let shared = checked_pipeline_product_v1(
        checked_pipeline_sum_v1(&[census.blocks, census.successors, 1])?,
        32,
    )?;
    let each = checked_pipeline_product_v1(
        checked_pipeline_sum_v1(&[
            census.blocks,
            census.successors,
            census.operations,
            sites,
            1,
        ])?,
        32,
    )?;
    let work = checked_pipeline_sum_v1(&[
        shared,
        checked_pipeline_product_v1(census.pipeline_creates, each)?,
    ])?;
    // 16B+E+32 includes the retained discovery successor roster and shared
    // CFG arrays/frontiers. O+2R+B+16 includes one-word sentinel ordinals,
    // two-word borrowed action rows, incoming cursors and vector headers.
    let temporary = checked_pipeline_sum_v1(&[
        checked_pipeline_product_v1(census.blocks, 17)?,
        census.successors,
        census.operations,
        checked_pipeline_product_v1(sites, 2)?,
        48,
    ])?;
    Ok((work, temporary))
}

/// Preflights CFG loop discovery, dominance fixed points, per-pipeline
/// schedule correlation, memoized uniformity proofs, and bounded
/// expression-equivalence searches.
pub(crate) fn preflight_pipeline_protocol_resource_upper_bound_v1(
    census: ProductionAnalysisInputCensusV1,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<ProductionAnalysisResourceUpperBoundV1, ProductionAnalysisResourceLimitV1> {
    if census.pipeline_creates == 0 {
        let findings = census
            .pipeline_events
            .min(MAX_PLIRON_PIPELINE_FINDINGS_V1 + 1);
        let work = checked_pipeline_sum_v1(&[
            checked_pipeline_product_v1(census.operations, 4)?,
            checked_pipeline_product_v1(census.pipeline_events, 2)?,
            1,
        ])?;
        let retained =
            checked_pipeline_product_v1(findings, MAX_PLIRON_PIPELINE_DIAGNOSTIC_BYTES_V1 + 32)?;
        let bound = ProductionAnalysisResourceUpperBoundV1::checked_phase(
            ProductionAnalysisResourcePhaseV1::PipelineProtocol,
            work,
            retained,
            0,
        )?;
        return limits.require(ProductionAnalysisResourcePhaseV1::PipelineProtocol, bound);
    }
    let (concrete_work, concrete_temporary) = pipeline_concrete_cfg_resource_delta_v1(census)?;
    let concrete_fact_work = pipeline_concrete_fact_query_work_v1(census)?;
    let blocks_squared = checked_pipeline_product_v1(census.blocks, census.blocks)?;
    let blocks_cubed = checked_pipeline_product_v1(blocks_squared, census.blocks.max(1))?;
    let equivalence_queries = pipeline_equivalence_query_upper_bound_v1(census)?;
    let equivalence_unique_pairs = pipeline_equivalence_unique_pair_upper_bound_v1(census)?;
    // Every query charges setup plus a root lookup. A pass-wide memo ensures
    // each ordered value pair schedules at most four children and four
    // completion cursors once. The final unit is the checked rejection.
    let equivalence_work = checked_pipeline_sum_v1(&[
        checked_pipeline_product_v1(equivalence_queries, 2)?,
        checked_pipeline_product_v1(equivalence_unique_pairs, 8)?,
        1,
    ])?;
    // Each possible pipeline performs at most one memoized visit per defining
    // operation, plus the rejecting visit that proves the bound was exceeded.
    // Eight units per definition cover root/memo/active hash operations,
    // defining-op inspection, and completion. Eight units per dependency cover
    // operand collection, the explicit-frame push/pop, memo lookup, and the
    // completion scan.
    let uniformity_work_per_pipeline = checked_pipeline_sum_v1(&[
        checked_pipeline_product_v1(
            census
                .operations
                .checked_add(1)
                .ok_or_else(pipeline_resource_overflow_v1)?,
            8,
        )?,
        checked_pipeline_product_v1(census.operands, 8)?,
    ])?;
    let uniformity_work =
        checked_pipeline_product_v1(census.pipeline_creates, uniformity_work_per_pipeline)?;
    let work = checked_pipeline_sum_v1(&[
        checked_pipeline_product_v1(census.operations, 8)?,
        checked_pipeline_product_v1(census.block_arguments, 2)?,
        checked_pipeline_product_v1(census.successors, census.blocks.max(1))?,
        blocks_cubed,
        // Bottom-up child containment is at most C²*B for C candidate
        // headers. Exactly one existing topology pass runs per candidate.
        // The once-per-candidate effect
        // and all-edge invariant scans; no trip-count-dependent work.
        checked_pipeline_product_v1(
            checked_pipeline_product_v1(
                checked_pipeline_product_v1(
                    census.index_lt_branch_candidates,
                    census.index_lt_branch_candidates,
                )?,
                census.blocks,
            )?,
            16,
        )?,
        checked_pipeline_product_v1(
            checked_pipeline_product_v1(
                census.index_lt_branch_candidates,
                checked_pipeline_sum_v1(&[
                    census.blocks,
                    census.successors,
                    census.operations,
                    census.operands,
                    1,
                ])?,
            )?,
            16,
        )?,
        equivalence_work,
        uniformity_work,
        concrete_work,
        concrete_fact_work,
    ])?;
    let certificates = census.pipeline_creates;
    let findings = census
        .operations
        .checked_mul(2)
        .ok_or_else(pipeline_resource_overflow_v1)?
        .min(MAX_PLIRON_PIPELINE_FINDINGS_V1 + 1);
    let retained = checked_pipeline_sum_v1(&[
        // Dominators and every retained loop summary: prologue/body/drain,
        // body membership, induction key/value pairs, and fixed metadata.
        checked_pipeline_product_v1(blocks_squared, 8)?,
        if census.index_lt_branch_candidates == 0 {
            0
        } else {
            checked_pipeline_product_v1(blocks_squared, 2)?
        },
        checked_pipeline_product_v1(
            certificates,
            census
                .blocks
                .checked_mul(3)
                .and_then(|items| items.checked_add(16))
                .ok_or_else(pipeline_resource_overflow_v1)?,
        )?,
        checked_pipeline_product_v1(
            certificates,
            (MAX_PLIRON_PIPELINE_VALUE_NAME_BYTES_V1 + 3) * 2,
        )?,
        checked_pipeline_product_v1(findings, MAX_PLIRON_PIPELINE_DIAGNOSTIC_BYTES_V1 + 32)?,
    ])?;
    let temporary = checked_pipeline_sum_v1(&[
        concrete_temporary,
        if census.index_lt_branch_candidates == 0 {
            0
        } else {
            checked_pipeline_sum_v1(&[
                checked_pipeline_product_v1(census.blocks, 16)?,
                checked_pipeline_product_v1(census.successors, 2)?,
            ])?
        },
        // Dominator fixed points overlap CFG inventories and the current loop
        // candidate's predecessor/member/frontier construction.
        checked_pipeline_product_v1(blocks_squared, 6)?,
        checked_pipeline_product_v1(census.blocks, 12)?,
        checked_pipeline_product_v1(census.successors, 4)?,
        census.block_arguments,
        checked_pipeline_product_v1(
            census.operations,
            census
                .max_operation_arity
                .checked_mul(3)
                .and_then(|items| items.checked_add(32))
                .ok_or_else(pipeline_resource_overflow_v1)?,
        )?,
        // Memo result, active-set entry, and explicit DFS frame per operation,
        // plus every operand roster owned by simultaneously live DFS frames.
        checked_pipeline_sum_v1(&[
            checked_pipeline_product_v1(census.operations, 3)?,
            census.operands,
        ])?,
        // Three scalar slots per ordered-pair memo entry plus one live query's
        // largest staged binary cursors and active pair keys.
        checked_pipeline_product_v1(equivalence_unique_pairs, 3)?,
        checked_pipeline_product_v1(EQUIVALENCE_REJECTING_ATTEMPTS_V1, 16)?,
        1,
    ])?;
    let bound = ProductionAnalysisResourceUpperBoundV1::checked_phase(
        ProductionAnalysisResourcePhaseV1::PipelineProtocol,
        work,
        retained,
        temporary,
    )?;
    limits.require(ProductionAnalysisResourcePhaseV1::PipelineProtocol, bound)
}

// Supplemental reservation only when the existing pointwise route needs the
// exact guarded full-workgroup domain. Concrete/no-new-route budgets stay exact.
fn pipeline_full_writer_domain_resource_bound_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<ProductionAnalysisResourceUpperBoundV1, ProductionAnalysisResourceLimitV1> {
    let phase = ProductionAnalysisResourcePhaseV1::PipelineProtocol;
    let base = preflight_pipeline_protocol_resource_upper_bound_v1(
        census,
        ProductionAnalysisResourceLimitsV1::production_hard_ceiling(),
    )?;
    let live = base
        .peak_storage_upper_bound()
        .checked_sub(base.retained_storage_upper_bound())
        .ok_or_else(pipeline_resource_overflow_v1)?;
    // Geometry/owner scans, at most two uniform queries per preheader block,
    // one existing topology pass, and all incoming edge/payload visits per read.
    let op_inputs = checked_pipeline_sum_v1(&[census.operations, census.operands, 1])?;
    let operations = checked_pipeline_sum_v1(&[census.operations, 1])?;
    let blocks = checked_pipeline_sum_v1(&[census.blocks, 1])?;
    let (queries, pairs) = pipeline_full_writer_domain_equivalence_limits_v1(census)?;
    let work = checked_pipeline_sum_v1(&[
        checked_pipeline_product_v1(queries, 2)?,
        checked_pipeline_product_v1(pairs, 8)?,
        1,
        checked_pipeline_product_v1(operations, 32)?,
        // Finite-child replay may query each child argument's type. A block
        // argument type lookup searches that block's argument roster.
        checked_pipeline_product_v1(
            census.index_lt_branch_candidates,
            checked_pipeline_product_v1(
                checked_pipeline_product_v1(census.block_arguments, census.block_arguments)?,
                8,
            )?,
        )?,
        checked_pipeline_product_v1(
            census.blocks,
            checked_pipeline_sum_v1(&[
                checked_pipeline_product_v1(op_inputs, 32)?,
                checked_pipeline_product_v1(census.blocks, blocks)?,
            ])?,
        )?,
        // Outgoing edge payloads can repeat values whose defining result or
        // argument roster must be searched by each child replay's type query.
        checked_pipeline_product_v1(
            census.index_lt_branch_candidates,
            checked_pipeline_product_v1(
                census.operands,
                checked_pipeline_product_v1(
                    checked_pipeline_sum_v1(&[census.results, census.block_arguments])?,
                    8,
                )?,
            )?,
        )?,
        checked_pipeline_product_v1(
            census.ranked_accesses,
            checked_pipeline_sum_v1(&[
                census.blocks,
                census.successors,
                census.operands,
                census.results,
                census.block_arguments,
                1,
            ])?,
        )?
        .checked_mul(32)
        .ok_or_else(pipeline_resource_overflow_v1)?,
    ])?;
    let temporary = checked_pipeline_sum_v1(&[
        live,
        checked_pipeline_product_v1(census.blocks, 16)?,
        checked_pipeline_product_v1(pairs, 3)?,
        checked_pipeline_product_v1(EQUIVALENCE_REJECTING_ATTEMPTS_V1, 16)?,
        checked_pipeline_product_v1(census.successors, 2)?,
        checked_pipeline_product_v1(op_inputs, 8)?,
        checked_pipeline_product_v1(census.ranked_accesses, 8)?,
    ])?;
    ProductionAnalysisResourceUpperBoundV1::checked_phase(phase, work, 0, temporary)
}

fn pipeline_full_writer_domain_equivalence_limits_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<(usize, usize), ProductionAnalysisResourceLimitV1> {
    let queries = checked_pipeline_product_v1(
        checked_pipeline_product_v1(
            census.index_lt_branch_candidates,
            checked_pipeline_sum_v1(&[census.operands, 3])?,
        )?,
        2,
    )?;
    let values = checked_pipeline_sum_v1(&[census.results, census.block_arguments])?;
    let pairs = checked_pipeline_product_v1(values, values)?.min(checked_pipeline_product_v1(
        queries,
        MAX_EQUIVALENCE_WORK_V1,
    )?);
    Ok((queries, pairs))
}

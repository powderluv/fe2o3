// Private demand solver over one verified, immutable function. An obligation
// describes a relation at block entry, never a global equality of SSA values.
struct BoundsEdgeTransportV1<'a> {
    context: &'a Context,
    blocks: &'a [pliron::context::Ptr<pliron::basic_block::BasicBlock>],
    predecessors: &'a [Vec<PredecessorEdge>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundsTransportObligationV1 {
    block: usize,
    fact: LessThanFact,
}

// Sixteen word-sized items conservatively cover the fixed tagged record on
// the pinned 64-bit host. This is not an allocator-byte accounting contract.
const BOUNDS_TRANSPORT_ROW_ITEMS_V1: usize = 16;
const BOUNDS_TRANSPORT_CANONICAL_WORK_V1: usize = 32 + MAX_RANKED_MEMORY_RANK;
// No allocation follows a failed admission. This includes the largest paid
// roster scan, visited-row scan, or requested-capacity relocation admission.
const BOUNDS_TRANSPORT_DENIED_WORK_V1: usize = MAX_RANKED_BOUNDS_OPERATION_ITEMS
    + MAX_RANKED_BOUNDS_STORAGE_ITEMS * 2
    + BOUNDS_TRANSPORT_CANONICAL_WORK_V1 * 2
    + 64;

fn bounds_transport_resource_bound_v1(
    census: ProductionAnalysisInputCensusV1,
) -> Result<(usize, usize), ProductionAnalysisResourceLimitV1> {
    let error = "memory-bounds transport upper bound";
    let queries = census.operands.min(checked_ranked_bounds_product_v1(
        census.ranked_accesses,
        MAX_RANKED_MEMORY_RANK,
        error,
    )?);
    if census.block_arguments == 0 || queries == 0 {
        return Ok((0, 0));
    }
    let values = checked_ranked_bounds_sum_v1(&[census.results, census.block_arguments], error)?;
    // Value IDs, constant results, static view extents and unresolved view
    // dimensions form a finite atom universe. Each view contributes at most
    // MAX_RANKED_MEMORY_RANK of either kind. Publication also caps the closure.
    let atoms = checked_ranked_bounds_product_v1(values, MAX_RANKED_MEMORY_RANK * 2 + 2, error)?;
    let states = checked_ranked_bounds_product_v1(
        census.blocks,
        checked_ranked_bounds_product_v1(atoms, atoms, error)?,
        error,
    )?
    .min(MAX_RANKED_BOUNDS_FACTS);
    let edge_work = checked_ranked_bounds_sum_v1(
        &[
            75,
            checked_ranked_bounds_product_v1(census.block_arguments, 4, error)?,
            checked_ranked_bounds_product_v1(values, 6, error)?,
            BOUNDS_TRANSPORT_CANONICAL_WORK_V1 * 4,
            checked_ranked_bounds_product_v1(states, BOUNDS_TRANSPORT_ROW_ITEMS_V1 + 1, error)?,
        ],
        error,
    )?;
    let query_work = checked_ranked_bounds_sum_v1(
        &[
            19,
            checked_ranked_bounds_product_v1(states, 59, error)?,
            checked_ranked_bounds_product_v1(
                checked_ranked_bounds_product_v1(states, census.successors, error)?,
                edge_work,
                error,
            )?,
        ],
        error,
    )?;
    let work = checked_ranked_bounds_product_v1(queries, query_work, error)?
        .min(MAX_RANKED_BOUNDS_WORK_UNITS + BOUNDS_TRANSPORT_DENIED_WORK_V1);
    let query_storage = checked_ranked_bounds_sum_v1(
        &[
            3,
            checked_ranked_bounds_product_v1(states, BOUNDS_TRANSPORT_ROW_ITEMS_V1 * 4, error)?,
        ],
        error,
    )?;
    let storage = checked_ranked_bounds_product_v1(queries, query_storage, error)?
        .min(MAX_RANKED_BOUNDS_STORAGE_ITEMS);
    Ok((work, storage))
}

fn bounds_transport_failure_v1(
    resource: &'static str,
    limit: usize,
    actual: usize,
) -> RankedBoundsFindingV1 {
    RankedBoundsFindingV1::ResourceLimitExceeded {
        resource,
        limit,
        actual,
    }
}

fn bounds_transport_product_v1(lhs: usize, rhs: usize) -> Result<usize, RankedBoundsFindingV1> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        bounds_transport_failure_v1(
            "analysis work unit",
            MAX_RANKED_BOUNDS_WORK_UNITS,
            usize::MAX,
        )
    })
}

fn bounds_transport_sum_v1(lhs: usize, rhs: usize) -> Result<usize, RankedBoundsFindingV1> {
    lhs.checked_add(rhs).ok_or_else(|| {
        bounds_transport_failure_v1(
            "analysis work unit",
            MAX_RANKED_BOUNDS_WORK_UNITS,
            usize::MAX,
        )
    })
}

fn bounds_transport_enqueue_v1(
    visited: &mut Vec<BoundsTransportObligationV1>,
    obligation: BoundsTransportObligationV1,
    budget: &mut RankedBoundsBudget,
) -> Result<(), RankedBoundsFindingV1> {
    let scan = bounds_transport_product_v1(visited.len(), BOUNDS_TRANSPORT_ROW_ITEMS_V1 + 1)?;
    budget.work(bounds_transport_sum_v1(scan, 1)?)?;
    if visited.contains(&obligation) {
        return Ok(());
    }
    if visited.len() == MAX_RANKED_BOUNDS_FACTS {
        return Err(bounds_transport_failure_v1(
            "transport obligation",
            MAX_RANKED_BOUNDS_FACTS,
            MAX_RANKED_BOUNDS_FACTS + 1,
        ));
    }
    if visited.len() == visited.capacity() {
        let capacity =
            bounds_transport_product_v1(visited.capacity(), 2)?.clamp(4, MAX_RANKED_BOUNDS_FACTS);
        let storage = bounds_transport_product_v1(capacity, BOUNDS_TRANSPORT_ROW_ITEMS_V1)?;
        budget.storage(storage)?;
        let relocation = bounds_transport_product_v1(visited.len(), BOUNDS_TRANSPORT_ROW_ITEMS_V1)?;
        budget.work(bounds_transport_sum_v1(
            bounds_transport_sum_v1(relocation, capacity)?,
            1,
        )?)?;
        visited
            .try_reserve_exact(capacity - visited.len())
            .map_err(|_| {
                bounds_transport_failure_v1(
                    "analysis storage item",
                    MAX_RANKED_BOUNDS_STORAGE_ITEMS,
                    usize::MAX,
                )
            })?;
    }
    budget.work(BOUNDS_TRANSPORT_ROW_ITEMS_V1 + 1)?;
    visited.push(obligation);
    Ok(())
}

impl BoundsEdgeTransportV1<'_> {
    fn roster_len(&self, value: Value) -> Result<usize, RankedBoundsFindingV1> {
        if let Some(operation) = value.defining_op() {
            Ok(operation
                .try_deref(self.context)
                .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?
                .get_num_results())
        } else {
            let block = value
                .defining_block()
                .ok_or(RankedBoundsFindingV1::StructuralVerificationFailed)?;
            Ok(block
                .try_deref(self.context)
                .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?
                .get_num_arguments())
        }
    }

    fn canonical(
        &self,
        value: Value,
        budget: &mut RankedBoundsBudget,
    ) -> Result<IndexExpr, RankedBoundsFindingV1> {
        budget.work(BOUNDS_TRANSPORT_CANONICAL_WORK_V1)?;
        if let Some(operation) = value.defining_op() {
            let operation = Operation::get_op_dyn(operation, self.context);
            if let Some(dimension) = operation.downcast_ref::<DimensionOp>() {
                // ranked_view_type calls Value::get_type, which searches this
                // exact definition roster even when its type is already known.
                budget.work(self.roster_len(dimension.view(self.context))?)?;
            }
        }
        Ok(canonical_index_expr(value, self.context))
    }

    fn proves(
        &self,
        block: usize,
        fact: LessThanFact,
        budget: &mut RankedBoundsBudget,
    ) -> Result<bool, RankedBoundsFindingV1> {
        budget.work(1)?;
        if block >= self.blocks.len() || self.predecessors.len() != self.blocks.len() {
            return Err(RankedBoundsFindingV1::StructuralVerificationFailed);
        }
        budget.storage(3)?;
        let mut visited = Vec::new();
        bounds_transport_enqueue_v1(
            &mut visited,
            BoundsTransportObligationV1 { block, fact },
            budget,
        )?;
        let mut cursor = 0;
        while cursor < visited.len() {
            budget.work(BOUNDS_TRANSPORT_ROW_ITEMS_V1 + 2)?;
            let obligation = visited[cursor];
            cursor += 1;
            let LessThanFact { lhs, rhs } = obligation.fact;
            if matches!((lhs, rhs), (IndexExpr::Constant(a), IndexExpr::Constant(b)) if a < b) {
                continue;
            }
            // A static operation ID is re-executed on each loop iteration.
            // Incoming facts cannot certify its newly computed result.
            if self.defined_in_block(lhs, obligation.block, budget)?
                || self.defined_in_block(rhs, obligation.block, budget)?
            {
                return Ok(false);
            }
            if obligation.block == 0 || self.predecessors[obligation.block].is_empty() {
                return Ok(false);
            }
            for edge in &self.predecessors[obligation.block] {
                budget.work(1)?;
                let (fact, guard) = self.pull_back(obligation, edge, budget)?;
                budget.work(BOUNDS_TRANSPORT_ROW_ITEMS_V1)?;
                if guard.is_some_and(|guard| {
                    guard.lhs == fact.lhs
                        && (guard.rhs == fact.rhs
                            || matches!((guard.rhs, fact.rhs),
                                (IndexExpr::Constant(upper), IndexExpr::Constant(extent))
                                    if upper <= extent))
                })
                    || matches!((fact.lhs, fact.rhs), (IndexExpr::Constant(a), IndexExpr::Constant(b)) if a < b)
                {
                    continue;
                }
                bounds_transport_enqueue_v1(
                    &mut visited,
                    BoundsTransportObligationV1 {
                        block: edge.block,
                        fact,
                    },
                    budget,
                )?;
            }
        }
        // Every finite predecessor path ends at a matching current-edge guard
        // or a constant proof. Revisited obligations represent a safety cycle,
        // not evidence that distinct loop-carried values are equal.
        Ok(true)
    }

    fn defined_in_block(
        &self,
        expression: IndexExpr,
        block: usize,
        budget: &mut RankedBoundsBudget,
    ) -> Result<bool, RankedBoundsFindingV1> {
        budget.work(2)?;
        let value = match expression {
            IndexExpr::Constant(_) => return Ok(false),
            IndexExpr::Value(value) | IndexExpr::Dimension { view: value, .. } => value,
        };
        match value.defining_op() {
            Some(operation) => Ok(operation
                .try_deref(self.context)
                .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?
                .get_parent_block()
                == Some(self.blocks[block])),
            None => Ok(false),
        }
    }

    fn pull_back(
        &self,
        obligation: BoundsTransportObligationV1,
        edge: &PredecessorEdge,
        budget: &mut RankedBoundsBudget,
    ) -> Result<(LessThanFact, Option<LessThanFact>), RankedBoundsFindingV1> {
        budget.work(16)?;
        let target = self.blocks[obligation.block];
        let terminator = self.blocks[edge.block]
            .deref(self.context)
            .get_terminator(self.context)
            .ok_or(RankedBoundsFindingV1::StructuralVerificationFailed)?;
        let control = ControlViewV1::observe(self.context, terminator)
            .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?;
        let edge = control.edge(edge.successor)
            .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?;
        if edge.target() != target {
            return Err(RankedBoundsFindingV1::StructuralVerificationFailed);
        }
        let map = |expression, budget: &mut RankedBoundsBudget| {
            self.pull_back_expression(expression, &edge, budget)
        };
        let fact = LessThanFact {
            lhs: map(obligation.fact.lhs, budget)?,
            rhs: map(obligation.fact.rhs, budget)?,
        };
        let guard = if let Some((lhs, rhs)) = edge.index_less_than_guard() {
            Some(LessThanFact {
                lhs: self.canonical(lhs, budget)?,
                rhs: self.canonical(rhs, budget)?,
            })
        } else {
            None
        };
        Ok((fact, guard))
    }

    fn pull_back_expression(
        &self,
        expression: IndexExpr,
        edge: &EdgeViewV1<'_, '_>,
        budget: &mut RankedBoundsBudget,
    ) -> Result<IndexExpr, RankedBoundsFindingV1> {
        budget.work(4)?;
        let target = edge.target();
        let value = match expression {
            IndexExpr::Constant(_) => return Ok(expression),
            IndexExpr::Value(value) | IndexExpr::Dimension { view: value, .. } => value,
        };
        if value.defining_block() != Some(target) {
            return Ok(expression);
        }
        let count = target.deref(self.context).get_num_arguments();
        budget.work(bounds_transport_sum_v1(count, 4)?)?;
        let index = value
            .try_find_index(self.context)
            .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?;
        let (incoming, parameter) = edge.argument_at(index)
            .map_err(|_| RankedBoundsFindingV1::StructuralVerificationFailed)?;
        if parameter != value {
            return Err(RankedBoundsFindingV1::StructuralVerificationFailed);
        }
        budget.work(bounds_transport_sum_v1(
            bounds_transport_sum_v1(count, self.roster_len(incoming)?)?,
            4,
        )?)?;
        if incoming.get_type(self.context) != value.get_type(self.context) {
            return Err(RankedBoundsFindingV1::StructuralVerificationFailed);
        }
        Ok(match expression {
            IndexExpr::Value(_) => self.canonical(incoming, budget)?,
            IndexExpr::Dimension { dimension, .. } => {
                budget.work(bounds_transport_sum_v1(
                    BOUNDS_TRANSPORT_CANONICAL_WORK_V1,
                    self.roster_len(incoming)?,
                )?)?;
                let ty = ranked_view_type(incoming, self.context)
                    .ok_or(RankedBoundsFindingV1::StructuralVerificationFailed)?;
                let ty = ty.deref(self.context);
                if dimension >= ty.shape().len() {
                    return Err(RankedBoundsFindingV1::StructuralVerificationFailed);
                }
                extent_expr(incoming, &ty, dimension, self.context)
            }
            IndexExpr::Constant(_) => unreachable!(),
        })
    }
}

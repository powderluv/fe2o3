//! Workload-neutral safe-Rust reference binding and effect IR.
//!
//! This IR is deliberately bounded and fail-closed. It retains exact rustc
//! identities and represents only semantics translated without workload names.
//! A point reference may add up to three leading `usize` coordinate arguments;
//! the remaining arguments map to the kernel ABI. Its extracted write events
//! describe coordinates, path predicates, and RHS expressions independently
//! of the GPU projection. Matching those events proves only per-effect partial
//! correctness, not total coverage of a dynamic output domain.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use rustc_abi::ExternAbi;
use rustc_hir::{Mutability, Safety};
use rustc_middle::mir::{
    AssertMessage, BinOp, Body, CastKind, Operand, Place, ProjectionElem, Rvalue, StatementKind,
    TerminatorKind, UnOp, UnwindAction,
};
use rustc_middle::ty::{Instance, Ty, TyCtxt, TyKind, TypingEnv};
use rustc_span::Spanned;
use sha2::{Digest as _, Sha256};

#[path = "reference_binding_auth_v1.rs"]
mod reference_binding_auth_v1;
pub(crate) use reference_binding_auth_v1::{
    authenticate_reference_binding_v1, authenticate_reference_binding_with_origin_v1,
    instantiated_signature,
};
use reference_binding_auth_v1::{function_identity_v1, scalar_type_v1};

#[path = "reference_extraction_work_v1.rs"]
mod reference_extraction_work_v1;
#[path = "reference_loop_work_v1.rs"]
mod reference_loop_work_v1;
#[path = "reference_raw_source_work_v1.rs"]
mod reference_raw_source_work_v1;
use crate::rustc_semantic_plan_v1::SourceClosureWorkV1;
use reference_extraction_work_v1::ReferenceExtractionWorkV1;
#[path = "production_reference_effect_join_v2_replay.rs"]
mod production_reference_effect_join_v2_replay;
use reference_raw_source_work_v1::{
    authenticate_safe_local_reference_v1, charge_reference_source_v1,
};

#[path = "reference_signature_preimage_v1.rs"]
pub(crate) mod reference_signature_preimage_v1;

#[path = "reference_binding_census_v1.rs"]
mod reference_binding_census_v1;
pub(crate) use reference_binding_census_v1::equivalent_bindings_v1;

pub(crate) use reference_signature_preimage_v1::ReferenceLogicalSignaturePreimageV1;
use reference_signature_preimage_v1::{
    ReferenceCarrierV1, ReferencePointeeV1, ReferenceRegionV1, ReferenceReturnShapeV1,
    ReferenceSignatureErrorV1, ReferenceSignatureInputV1,
};

use crate::rustc_semantic_adapter_v1::{
    canonical_function_identities_v1, rustc_mir_body_sha256_v1,
};
use crate::trusted_device_items::{self, TrustedDeviceItem};

use fe2o3_verifier::portable_reference_v1::extraction::*;
pub(crate) use fe2o3_verifier::portable_reference_v1::*;
#[derive(Clone, Debug, Eq, PartialEq)]
enum ReferenceSymbolicValueV2 {
    Scalar(ReferenceEffectExpressionV1),
    CheckedPair {
        value: ReferenceEffectExpressionV1,
        overflowed: Option<bool>,
    },
}

type ReferenceSymbolicEnvironmentV2 = BTreeMap<u32, ReferenceSymbolicValueV2>;

#[derive(Clone, Debug)]
struct ReferenceLoopTraceV2 {
    header: u32,
    latch: u32,
    exit: Option<u32>,
    initial: ReferenceSymbolicEnvironmentV2,
    transitions: Vec<ReferenceSymbolicEnvironmentV2>,
    variants: Vec<ReferenceEffectExpressionV1>,
    exact_iterations: Option<u64>,
    maximum_iterations: Option<u64>,
}

#[derive(Clone, Debug)]
struct ReferenceSymbolicStateV2 {
    block: u32,
    environment: ReferenceSymbolicEnvironmentV2,
    guard: ReferencePathPredicateV1,
    traces: BTreeMap<(u32, u32), ReferenceLoopTraceV2>,
}

#[derive(Default)]
struct ReferenceSymbolicWorkBudgetV2 {
    charged_nodes: usize,
}

impl ReferenceSymbolicWorkBudgetV2 {
    fn charge_v2(&mut self, nodes: usize) -> Result<(), ReferenceBindingErrorV1> {
        self.charged_nodes = self.charged_nodes.checked_add(nodes).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference symbolic work-node accounting overflowed")
        })?;
        if self.charged_nodes > MAX_REFERENCE_SYMBOLIC_WORK_NODES_V2 {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference symbolic execution exceeds {MAX_REFERENCE_SYMBOLIC_WORK_NODES_V2} cumulative expression work nodes",
            )));
        }
        Ok(())
    }

    fn charge_expression_v2(
        &mut self,
        meter: &ReferenceExtractionWorkV1<'_>,
        expression: &ReferenceEffectExpressionV1,
    ) -> Result<(), ReferenceBindingErrorV1> {
        meter.clone_expression(expression)?;
        self.charge_v2(symbolic_expression_nodes_v2(meter, expression)?)
    }

    fn charge_environment_v2(
        &mut self,
        meter: &ReferenceExtractionWorkV1<'_>,
        environment: &ReferenceSymbolicEnvironmentV2,
    ) -> Result<(), ReferenceBindingErrorV1> {
        meter.charge(meter.environment_units(environment)?)?;
        self.charge_v2(symbolic_environment_nodes_v2(meter, environment)?)
    }

    fn charge_predicate_v2(
        &mut self,
        meter: &ReferenceExtractionWorkV1<'_>,
        predicate: &ReferencePathPredicateV1,
    ) -> Result<(), ReferenceBindingErrorV1> {
        meter.clone_predicate(predicate)?;
        self.charge_v2(symbolic_predicate_nodes_v2(meter, predicate)?)
    }

    fn charge_state_clone_v2(
        &mut self,
        meter: &ReferenceExtractionWorkV1<'_>,
        state: &ReferenceSymbolicStateV2,
    ) -> Result<(), ReferenceBindingErrorV1> {
        meter.charge(meter.state_units(state)?)?;
        self.charge_v2(symbolic_state_nodes_v2(meter, state)?)
    }
}

// Loop execution retains its original backend-local symbolic state and meter.
trait ReferenceEffectLoopExtV2 {
    fn observable_output_writes_with_loops_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        backedges: &BTreeSet<(u32, u32)>,
    ) -> Result<(Vec<ReferenceOutputWriteV1>, Vec<ReferenceLoopSummaryV2>), ReferenceBindingErrorV1>;

    fn output_relation_for_local_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        local: u32,
    ) -> Result<Option<(u32, bool)>, ReferenceBindingErrorV1>;

    fn summarize_dynamic_counted_loop_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        header: &ReferenceBlockV1,
        state: &ReferenceSymbolicStateV2,
        loop_nodes: &BTreeMap<(u32, u32), BTreeSet<u32>>,
    ) -> Result<Option<DynamicReferenceLoopSummaryV2>, ReferenceBindingErrorV1>;

    fn symbolic_output_coordinate_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        environment: &ReferenceSymbolicEnvironmentV2,
        destination: &ReferencePlaceV1,
        coordinate_output: bool,
    ) -> Result<ReferenceOutputCoordinateV1, ReferenceBindingErrorV1>;
}

impl ReferenceEffectLoopExtV2 for ReferenceEffectIrV1 {
    fn observable_output_writes_with_loops_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        backedges: &BTreeSet<(u32, u32)>,
    ) -> Result<(Vec<ReferenceOutputWriteV1>, Vec<ReferenceLoopSummaryV2>), ReferenceBindingErrorV1>
    {
        let loop_nodes = reference_natural_loop_nodes_v2(meter, self, backedges)?;
        meter.product(backedges.len(), backedges.len())?;
        meter.rows::<u32>(backedges.len())?;
        let loop_headers = backedges
            .iter()
            .map(|(_, header)| *header)
            .collect::<BTreeSet<_>>();
        let mut initial_environment = BTreeMap::new();
        meter.charge(self.relations.len())?;
        let point_count = self.point_coordinate_count_v1()?;
        for relation in &self.relations {
            meter.tree::<(u32, ReferenceSymbolicValueV2)>(initial_environment.len())?;
            match relation {
                ReferenceArgumentRelationV1::PointCoordinate {
                    reference_argument,
                    axis,
                } => {
                    initial_environment.insert(
                        reference_argument + 1,
                        ReferenceSymbolicValueV2::Scalar(
                            ReferenceEffectExpressionV1::PointCoordinate { axis: *axis },
                        ),
                    );
                }
                ReferenceArgumentRelationV1::ScalarInput { argument, .. } => {
                    let reference_argument =
                        point_count.checked_add(*argument).ok_or_else(|| {
                            ReferenceBindingErrorV1::new(
                                "reference scalar argument local index overflowed",
                            )
                        })?;
                    initial_environment.insert(
                        reference_argument + 1,
                        ReferenceSymbolicValueV2::Scalar(
                            ReferenceEffectExpressionV1::KernelScalarArgument {
                                argument: *argument,
                            },
                        ),
                    );
                }
                ReferenceArgumentRelationV1::SharedSliceInput { .. }
                | ReferenceArgumentRelationV1::DisjointOutputSlice { .. }
                | ReferenceArgumentRelationV1::DisjointOutputCoordinate { .. } => {}
            }
        }
        meter.rows::<ReferenceSymbolicStateV2>(1)?;
        meter.rows::<ReferenceGuardClauseV1>(1)?;
        let mut pending = VecDeque::from([ReferenceSymbolicStateV2 {
            block: 0,
            environment: initial_environment,
            guard: ReferencePathPredicateV1::unconditional_v1(),
            traces: BTreeMap::new(),
        }]);
        let mut writes = Vec::new();
        let mut completed_traces = Vec::new();
        let mut steps = 0_usize;
        let mut work_budget = ReferenceSymbolicWorkBudgetV2::default();
        work_budget.charge_environment_v2(meter, &pending[0].environment)?;
        while let Some(mut state) = pending.pop_front() {
            meter.charge(1)?;
            meter.tree::<u32>(loop_headers.len())?;
            steps = steps.checked_add(1).ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference symbolic execution step count overflowed")
            })?;
            if steps > MAX_REFERENCE_SYMBOLIC_STEPS_V2 {
                return Err(ReferenceBindingErrorV1::new(format!(
                    "reference symbolic execution exceeds {MAX_REFERENCE_SYMBOLIC_STEPS_V2} steps",
                )));
            }
            if loop_headers.contains(&state.block) {
                for (latch, header) in backedges {
                    meter.tree::<((u32, u32), ReferenceLoopTraceV2)>(state.traces.len())?;
                    if *header == state.block && !state.traces.contains_key(&(*header, *latch)) {
                        work_budget.charge_environment_v2(meter, &state.environment)?;
                        state.traces.insert(
                            (*header, *latch),
                            ReferenceLoopTraceV2 {
                                header: *header,
                                latch: *latch,
                                exit: None,
                                initial: state.environment.clone(),
                                transitions: Vec::new(),
                                variants: Vec::new(),
                                exact_iterations: None,
                                maximum_iterations: None,
                            },
                        );
                    }
                }
            }
            let block = self.blocks.get(state.block as usize).ok_or_else(|| {
                ReferenceBindingErrorV1::new(format!(
                    "reference symbolic execution reached missing block {}",
                    state.block,
                ))
            })?;
            for assignment in &block.assignments {
                meter.charge(1)?;
                if let Some((argument, coordinate_output)) =
                    self.output_relation_for_local_v2(meter, assignment.destination.local)?
                {
                    let coordinate = self.symbolic_output_coordinate_v2(
                        meter,
                        &state.environment,
                        &assignment.destination,
                        coordinate_output,
                    )?;
                    let rhs = symbolic_scalar_v2(symbolic_value_v2(
                        meter,
                        &state.environment,
                        &assignment.value,
                    )?)?;
                    require_symbolic_expression_budget_v2(meter, &rhs)?;
                    work_budget.charge_expression_v2(meter, &rhs)?;
                    work_budget.charge_predicate_v2(meter, &state.guard)?;
                    meter.charge(meter.value(&assignment.value)?)?;
                    meter.grow::<ReferenceOutputWriteV1>(writes.len())?;
                    if writes.len() >= MAX_REFERENCE_STATEMENTS_V1 {
                        return Err(ReferenceBindingErrorV1::new(format!(
                            "reference symbolic execution exceeds {MAX_REFERENCE_STATEMENTS_V1} retained output writes",
                        )));
                    }
                    writes.push(ReferenceOutputWriteV1 {
                        argument,
                        block: block.block,
                        statement: assignment.statement,
                        coordinate,
                        guard: state.guard.clone(),
                        rhs,
                        value: assignment.value.clone(),
                    });
                    continue;
                }
                if !assignment.destination.projection.is_empty() {
                    return Err(ReferenceBindingErrorV1::new(format!(
                        "loop-carried reference assignment to _{} uses unsupported projection {:?}",
                        assignment.destination.local, assignment.destination.projection,
                    )));
                }
                let value = symbolic_value_v2(meter, &state.environment, &assignment.value)?;
                require_symbolic_value_budget_v2(meter, &value)?;
                work_budget.charge_v2(symbolic_value_nodes_v2(meter, &value)?)?;
                meter.tree::<(u32, ReferenceSymbolicValueV2)>(state.environment.len())?;
                state
                    .environment
                    .insert(assignment.destination.local, value);
            }
            match &block.terminator {
                ReferenceTerminatorV1::Return => {
                    for trace in state.traces.into_values() {
                        meter.grow::<ReferenceLoopTraceV2>(completed_traces.len())?;
                        completed_traces.push(trace);
                    }
                }
                ReferenceTerminatorV1::Goto { target } => {
                    dispatch_symbolic_edge_v2(
                        meter,
                        &mut pending,
                        state,
                        (block.block, *target),
                        backedges,
                        &loop_nodes,
                        &mut work_budget,
                    )?;
                }
                ReferenceTerminatorV1::Assert {
                    condition,
                    expected,
                    success,
                    bounds_check,
                } => {
                    if bounds_check.is_some() {
                        dispatch_symbolic_edge_v2(
                            meter,
                            &mut pending,
                            state,
                            (block.block, *success),
                            backedges,
                            &loop_nodes,
                            &mut work_budget,
                        )?;
                        continue;
                    }
                    let expression = symbolic_scalar_v2(symbolic_operand_v2(
                        meter,
                        &state.environment,
                        condition,
                    )?)?;
                    let actual = reference_constant_bits_v2(meter, &expression)?
                        .filter(|(scalar, _)| *scalar == ReferenceScalarTypeV1::Bool)
                        .map(|(_, bits)| bits != 0)
                        .ok_or_else(|| {
                            ReferenceBindingErrorV1::new(format!(
                                "loop-carried reference assertion in block {} is not statically proved; checked arithmetic requires a range fact",
                                block.block,
                            ))
                        })?;
                    if actual != *expected {
                        return Err(ReferenceBindingErrorV1::new(format!(
                            "loop-carried reference assertion in block {} is statically false",
                            block.block,
                        )));
                    }
                    dispatch_symbolic_edge_v2(
                        meter,
                        &mut pending,
                        state,
                        (block.block, *success),
                        backedges,
                        &loop_nodes,
                        &mut work_budget,
                    )?;
                }
                ReferenceTerminatorV1::Switch {
                    discriminant,
                    values,
                    otherwise,
                } => {
                    let expression = symbolic_scalar_v2(symbolic_operand_v2(
                        meter,
                        &state.environment,
                        discriminant,
                    )?)?;
                    meter.tree::<u32>(loop_headers.len())?;
                    if loop_headers.contains(&block.block) {
                        meter.charge(state.traces.len())?;
                        for trace in state
                            .traces
                            .values_mut()
                            .filter(|trace| trace.header == block.block)
                        {
                            work_budget.charge_expression_v2(meter, &expression)?;
                            meter.grow::<ReferenceEffectExpressionV1>(trace.variants.len())?;
                            trace.variants.push(expression.clone());
                        }
                    }
                    if let Some((_, bits)) = reference_constant_bits_v2(meter, &expression)? {
                        meter.charge(values.len())?;
                        let target = values
                            .iter()
                            .find_map(|(value, target)| (*value == bits).then_some(*target))
                            .unwrap_or(*otherwise);
                        dispatch_symbolic_edge_v2(
                            meter,
                            &mut pending,
                            state,
                            (block.block, target),
                            backedges,
                            &loop_nodes,
                            &mut work_budget,
                        )?;
                    } else {
                        meter.tree::<u32>(loop_headers.len())?;
                        if loop_headers.contains(&block.block) {
                            if let Some(summary) = self.summarize_dynamic_counted_loop_v2(
                                meter,
                                block,
                                &state,
                                &loop_nodes,
                            )? {
                                meter.tree::<(u32, ReferenceSymbolicValueV2)>(
                                    state.environment.len(),
                                )?;
                                meter.clone_expression(&summary.final_induction)?;
                                state.environment.insert(
                                    summary.induction_local,
                                    ReferenceSymbolicValueV2::Scalar(
                                        summary.final_induction.clone(),
                                    ),
                                );
                                meter.tree::<((u32, u32), ReferenceLoopTraceV2)>(
                                    state.traces.len(),
                                )?;
                                let trace = state
                                    .traces
                                    .get_mut(&(block.block, summary.latch))
                                    .ok_or_else(|| {
                                        ReferenceBindingErrorV1::new(
                                            "dynamic reference loop lost its compiler trace",
                                        )
                                    })?;
                                trace.exit = Some(summary.exit);
                                trace.exact_iterations = None;
                                trace.maximum_iterations = Some(summary.maximum_iterations);
                                work_budget.charge_environment_v2(meter, &state.environment)?;
                                meter.grow::<ReferenceSymbolicEnvironmentV2>(
                                    trace.transitions.len(),
                                )?;
                                meter.grow::<ReferenceEffectExpressionV1>(trace.variants.len())?;
                                trace.transitions.push(state.environment.clone());
                                trace.variants.push(expression);
                                state.block = summary.exit;
                                work_budget.charge_state_clone_v2(meter, &state)?;
                                meter.grow::<ReferenceSymbolicStateV2>(pending.len())?;
                                pending.push_back(state);
                                continue;
                            }
                            return Err(ReferenceBindingErrorV1::new(format!(
                                "dynamic reference loop at header {} has no authenticated finite maximum and canonical unit-step variant",
                                block.block,
                            )));
                        }
                        let mut by_target = BTreeMap::<u32, Vec<u128>>::new();
                        meter.rows::<u128>(values.len())?;
                        let mut all_values = Vec::with_capacity(values.len());
                        for (value, target) in values {
                            meter.tree::<(u32, Vec<u128>)>(by_target.len())?;
                            let accepted = by_target.entry(*target).or_default();
                            meter.grow::<u128>(accepted.len())?;
                            accepted.push(*value);
                            all_values.push(*value);
                        }
                        for (target, accepted) in by_target {
                            work_budget.charge_state_clone_v2(meter, &state)?;
                            meter.clone_expression(&expression)?;
                            let mut branch = state.clone();
                            branch.guard = reference_predicate_and_atom_v1(
                                meter,
                                &branch.guard,
                                ReferenceGuardAtomV1::SwitchValueSet {
                                    discriminant: expression.clone(),
                                    values: accepted.into_boxed_slice(),
                                    inside_set: true,
                                },
                            )?;
                            work_budget.charge_predicate_v2(meter, &branch.guard)?;
                            dispatch_symbolic_edge_v2(
                                meter,
                                &mut pending,
                                branch,
                                (block.block, target),
                                backedges,
                                &loop_nodes,
                                &mut work_budget,
                            )?;
                        }
                        state.guard = reference_predicate_and_atom_v1(
                            meter,
                            &state.guard,
                            ReferenceGuardAtomV1::SwitchValueSet {
                                discriminant: expression,
                                values: all_values.into_boxed_slice(),
                                inside_set: false,
                            },
                        )?;
                        work_budget.charge_predicate_v2(meter, &state.guard)?;
                        dispatch_symbolic_edge_v2(
                            meter,
                            &mut pending,
                            state,
                            (block.block, *otherwise),
                            backedges,
                            &loop_nodes,
                            &mut work_budget,
                        )?;
                    }
                }
            }
        }
        if completed_traces.is_empty() {
            return Err(ReferenceBindingErrorV1::new(
                "bounded reference loop has no successful return path",
            ));
        }
        meter.rows::<ReferenceLoopSummaryV2>(completed_traces.len())?;
        let mut summaries = completed_traces
            .iter()
            .map(|trace| reference_loop_summary_v2(meter, trace))
            .collect::<Result<Vec<_>, _>>()?;
        let mut summary_units = 0_usize;
        for summary in &summaries {
            meter.charge(1)?;
            summary_units = reference_extraction_work_v1::add(
                summary_units,
                std::mem::size_of::<ReferenceLoopSummaryV2>(),
            )?;
            summary_units = reference_extraction_work_v1::add(
                summary_units,
                summary
                    .carried_locals
                    .len()
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| {
                        ReferenceBindingErrorV1::new("reference loop summary work overflow")
                    })?,
            )?;
        }
        meter.sort(summaries.len(), summary_units)?;
        summaries.sort();
        summaries.dedup();
        meter.sort(writes.len(), meter.effects(&writes)?)?;
        writes.sort_by(|lhs, rhs| {
            (
                lhs.argument,
                lhs.block,
                lhs.statement,
                &lhs.coordinate,
                &lhs.guard,
                &lhs.rhs,
            )
                .cmp(&(
                    rhs.argument,
                    rhs.block,
                    rhs.statement,
                    &rhs.coordinate,
                    &rhs.guard,
                    &rhs.rhs,
                ))
        });
        writes.dedup_by(|lhs, rhs| {
            lhs.argument == rhs.argument
                && lhs.block == rhs.block
                && lhs.statement == rhs.statement
                && lhs.coordinate == rhs.coordinate
                && lhs.guard == rhs.guard
                && lhs.rhs == rhs.rhs
        });
        Ok((writes, summaries))
    }

    fn output_relation_for_local_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        local: u32,
    ) -> Result<Option<(u32, bool)>, ReferenceBindingErrorV1> {
        meter.product(self.relations.len(), 2)?;
        let point_count = self.point_coordinate_count_v1()?;
        Ok(self.relations.iter().find_map(|relation| {
            let (argument, coordinate_output) = match relation {
                ReferenceArgumentRelationV1::DisjointOutputSlice { argument, .. } => {
                    (*argument, false)
                }
                ReferenceArgumentRelationV1::DisjointOutputCoordinate { argument, .. } => {
                    (*argument, true)
                }
                _ => return None,
            };
            let reference_argument = point_count.checked_add(argument)?;
            (local == reference_argument + 1).then_some((argument, coordinate_output))
        }))
    }

    fn summarize_dynamic_counted_loop_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        header: &ReferenceBlockV1,
        state: &ReferenceSymbolicStateV2,
        loop_nodes: &BTreeMap<(u32, u32), BTreeSet<u32>>,
    ) -> Result<Option<DynamicReferenceLoopSummaryV2>, ReferenceBindingErrorV1> {
        meter.charge(1)?;
        let ReferenceTerminatorV1::Switch {
            discriminant,
            values,
            otherwise,
        } = &header.terminator
        else {
            return Ok(None);
        };
        let (ReferenceOperandV1::Copy(discriminant) | ReferenceOperandV1::Move(discriminant)) =
            discriminant
        else {
            return Ok(None);
        };
        if !discriminant.projection.is_empty() {
            return Ok(None);
        }
        meter.rows::<&ReferenceAssignmentV1>(header.assignments.len())?;
        let assignments = header
            .assignments
            .iter()
            .filter(|assignment| {
                assignment.destination.local == discriminant.local
                    && assignment.destination.projection.is_empty()
            })
            .collect::<Vec<_>>();
        let [assignment] = assignments.as_slice() else {
            return Ok(None);
        };
        let ReferenceValueV1::Binary {
            operation: ReferenceBinaryOpV1::LessThan,
            lhs,
            rhs: bound,
            checked: false,
        } = &assignment.value
        else {
            return Ok(None);
        };
        let (ReferenceOperandV1::Copy(induction) | ReferenceOperandV1::Move(induction)) = lhs
        else {
            return Ok(None);
        };
        if !induction.projection.is_empty() {
            return Ok(None);
        }
        meter.charge(meter.place(induction)?)?;
        let mut induction = induction.clone();
        for _ in 0..fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 {
            meter.rows::<&ReferenceAssignmentV1>(header.assignments.len())?;
            let aliases = header
                .assignments
                .iter()
                .filter(|assignment| {
                    assignment.destination.local == induction.local
                        && assignment.destination.projection.is_empty()
                })
                .collect::<Vec<_>>();
            let [alias] = aliases.as_slice() else {
                break;
            };
            let ReferenceValueV1::Use(
                ReferenceOperandV1::Copy(source) | ReferenceOperandV1::Move(source),
            ) = &alias.value
            else {
                break;
            };
            if !source.projection.is_empty() || source.local == induction.local {
                break;
            }
            meter.charge(meter.place(source)?)?;
            induction = source.clone();
        }
        let [(0, exit)] = values.as_ref() else {
            return Ok(None);
        };
        for nodes in loop_nodes.values() {
            meter.charge(1)?;
            meter.tree::<u32>(nodes.len())?;
            meter.tree::<u32>(nodes.len())?;
        }
        let mut matching_loops = loop_nodes.iter().filter(|((_, loop_header), nodes)| {
            *loop_header == header.block && nodes.contains(otherwise) && !nodes.contains(exit)
        });
        let Some((&(latch, _), nodes)) = matching_loops.next() else {
            return Ok(None);
        };
        if matching_loops.next().is_some() {
            return Ok(None);
        }
        meter.tree::<(u32, ReferenceSymbolicValueV2)>(state.environment.len())?;
        let initial = state.environment.get(&induction.local);
        let Some(ReferenceSymbolicValueV2::Scalar(initial)) = initial else {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop has no scalar initial induction value",
            ));
        };
        let Some((initial_scalar, 0)) = reference_constant_bits_v2(meter, initial)? else {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop induction is not initialized to unsigned zero",
            ));
        };
        let final_induction =
            symbolic_scalar_v2(symbolic_operand_v2(meter, &state.environment, bound)?)?;
        let maximum_iterations = match &final_induction {
            ReferenceEffectExpressionV1::KernelScalarArgument { argument } => {
                meter.charge(self.relations.len())?;
                let scalar = self.relations.iter().find_map(|relation| match relation {
                    ReferenceArgumentRelationV1::ScalarInput {
                        argument: actual,
                        scalar,
                    } if actual == argument => Some(*scalar),
                    _ => None,
                });
                let Some(scalar) = scalar else {
                    return Err(ReferenceBindingErrorV1::new(
                        "dynamic counted loop bound has no scalar ABI relation",
                    ));
                };
                let Some(maximum) = unsigned_reference_scalar_maximum_v2(scalar) else {
                    return Err(ReferenceBindingErrorV1::new(
                        "dynamic counted loop bound is not an unsigned finite machine scalar",
                    ));
                };
                if scalar != initial_scalar {
                    return Err(ReferenceBindingErrorV1::new(
                        "dynamic counted loop induction and bound types differ",
                    ));
                }
                maximum
            }
            _ => return Ok(None),
        };
        let mut induction_assignments = Vec::new();
        for node in nodes {
            // This also prepays the later increment and overflow scans.
            meter.charge(2)?;
            for assignment in &self.blocks[*node as usize].assignments {
                meter.charge(2)?;
                if !assignment.destination.projection.is_empty() {
                    return Err(ReferenceBindingErrorV1::new(
                        "dynamic counted loop contains a projected memory effect",
                    ));
                }
                meter.tree::<((u32, u32), ReferenceLoopTraceV2)>(state.traces.len())?;
                let carried = if let Some(trace) = state.traces.get(&(header.block, latch)) {
                    meter.tree::<(u32, ReferenceSymbolicValueV2)>(trace.initial.len())?;
                    trace.initial.contains_key(&assignment.destination.local)
                } else {
                    false
                };
                if carried && assignment.destination.local != induction.local {
                    return Err(ReferenceBindingErrorV1::new(
                        "dynamic counted loop mutates another loop-carried local",
                    ));
                }
                if assignment.destination.local == induction.local {
                    meter.grow::<&ReferenceAssignmentV1>(induction_assignments.len())?;
                    induction_assignments.push(assignment);
                }
            }
        }
        let [induction_assignment] = induction_assignments.as_slice() else {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop does not have one induction assignment",
            ));
        };
        let ReferenceValueV1::Use(
            ReferenceOperandV1::Copy(increment_value) | ReferenceOperandV1::Move(increment_value),
        ) = &induction_assignment.value
        else {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop induction is not assigned from checked arithmetic",
            ));
        };
        let [ReferencePlaceProjectionV1::Field(0)] = increment_value.projection.as_ref() else {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop induction does not select the checked value field",
            ));
        };
        let increment_pair = increment_value.local;
        let increment = nodes
            .iter()
            .flat_map(|node| self.blocks[*node as usize].assignments.iter())
            .find(|assignment| {
                assignment.destination.local == increment_pair
                    && assignment.destination.projection.is_empty()
            });
        let Some(ReferenceAssignmentV1 {
            value:
                ReferenceValueV1::Binary {
                    operation: ReferenceBinaryOpV1::Add,
                    lhs: increment_lhs,
                    rhs: increment_rhs,
                    checked: true,
                },
            ..
        }) = increment
        else {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop has no checked additive transition",
            ));
        };
        let increment_matches = [
            (increment_lhs, increment_rhs),
            (increment_rhs, increment_lhs),
        ]
        .into_iter()
        .any(|(candidate_induction, candidate_step)| {
            matches!(
                candidate_induction,
                ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place)
                    if place.local == induction.local && place.projection.is_empty()
            ) && matches!(
                candidate_step,
                ReferenceOperandV1::Constant(ReferenceConstantV1::Scalar { bits: 1, .. })
            )
        });
        if !increment_matches {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop transition is not induction plus one",
            ));
        }
        let overflow_checked = nodes.iter().any(|node| {
            matches!(
                &self.blocks[*node as usize].terminator,
                ReferenceTerminatorV1::Assert {
                    condition: ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place),
                    expected: false,
                    ..
                } if place.local == increment_pair
                    && matches!(place.projection.as_ref(), [ReferencePlaceProjectionV1::Field(1)])
            )
        });
        if !overflow_checked {
            return Err(ReferenceBindingErrorV1::new(
                "dynamic counted loop does not retain its overflow assertion",
            ));
        }
        Ok(Some(DynamicReferenceLoopSummaryV2 {
            latch,
            exit: *exit,
            induction_local: induction.local,
            final_induction,
            maximum_iterations,
        }))
    }

    fn symbolic_output_coordinate_v2(
        &self,
        meter: &ReferenceExtractionWorkV1<'_>,
        environment: &ReferenceSymbolicEnvironmentV2,
        destination: &ReferencePlaceV1,
        coordinate_output: bool,
    ) -> Result<ReferenceOutputCoordinateV1, ReferenceBindingErrorV1> {
        meter.charge(1)?;
        match destination.projection.as_ref() {
            [ReferencePlaceProjectionV1::Dereference] if coordinate_output => {
                meter.rows::<ReferenceEffectExpressionV1>(self.relations.len())?;
                let axes = self
                    .relations
                    .iter()
                    .filter_map(|relation| match relation {
                        ReferenceArgumentRelationV1::PointCoordinate { axis, .. } => {
                            Some(ReferenceEffectExpressionV1::PointCoordinate { axis: *axis })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                if axes.is_empty() {
                    Ok(ReferenceOutputCoordinateV1::SingleCoordinate)
                } else {
                    Ok(ReferenceOutputCoordinateV1::LogicalPoint(axes))
                }
            }
            [
                ReferencePlaceProjectionV1::Dereference,
                ReferencePlaceProjectionV1::Index(index),
            ] if !coordinate_output => {
                meter.tree::<(u32, ReferenceSymbolicValueV2)>(environment.len())?;
                let value = environment.get(index).ok_or_else(|| {
                    ReferenceBindingErrorV1::new(format!(
                        "reference output index local _{index} has no loop-carried value",
                    ))
                })?;
                Ok(ReferenceOutputCoordinateV1::Dynamic(symbolic_scalar_v2(
                    meter.clone_symbolic(value)?,
                )?))
            }
            [
                ReferencePlaceProjectionV1::Dereference,
                ReferencePlaceProjectionV1::ConstantIndex {
                    offset,
                    minimum_length,
                    from_end,
                },
            ] if !coordinate_output => Ok(ReferenceOutputCoordinateV1::Constant {
                offset: *offset,
                minimum_length: *minimum_length,
                from_end: *from_end,
            }),
            projection => Err(ReferenceBindingErrorV1::new(format!(
                "observable loop output uses unsupported projection {projection:?}",
            ))),
        }
    }
}

struct DynamicReferenceLoopSummaryV2 {
    latch: u32,
    exit: u32,
    induction_local: u32,
    final_induction: ReferenceEffectExpressionV1,
    maximum_iterations: u64,
}

fn unsigned_reference_scalar_maximum_v2(scalar: ReferenceScalarTypeV1) -> Option<u64> {
    Some(match scalar {
        ReferenceScalarTypeV1::U8 => u64::from(u8::MAX),
        ReferenceScalarTypeV1::U16 => u64::from(u16::MAX),
        ReferenceScalarTypeV1::U32 => u64::from(u32::MAX),
        ReferenceScalarTypeV1::U64 | ReferenceScalarTypeV1::Usize => u64::MAX,
        _ => return None,
    })
}

fn symbolic_value_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    environment: &ReferenceSymbolicEnvironmentV2,
    value: &ReferenceValueV1,
) -> Result<ReferenceSymbolicValueV2, ReferenceBindingErrorV1> {
    meter.rows::<ReferenceEffectExpressionV1>(2)?;
    match value {
        ReferenceValueV1::Use(operand) => symbolic_operand_v2(meter, environment, operand),
        ReferenceValueV1::Binary {
            operation,
            lhs,
            rhs,
            checked,
        } => {
            let lhs = symbolic_scalar_v2(symbolic_operand_v2(meter, environment, lhs)?)?;
            let rhs = symbolic_scalar_v2(symbolic_operand_v2(meter, environment, rhs)?)?;
            let overflowed = if *checked {
                reference_checked_overflow_v2(meter, *operation, &lhs, &rhs)?
            } else {
                None
            };
            let expression = ReferenceEffectExpressionV1::Binary {
                operation: *operation,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                checked: *checked,
            };
            let folded = reference_fold_constant_v2(meter, &expression)?.unwrap_or(expression);
            if *checked {
                Ok(ReferenceSymbolicValueV2::CheckedPair {
                    value: folded,
                    overflowed,
                })
            } else {
                Ok(ReferenceSymbolicValueV2::Scalar(folded))
            }
        }
        ReferenceValueV1::Unary { operation, operand } => {
            let operand = symbolic_scalar_v2(symbolic_operand_v2(meter, environment, operand)?)?;
            let expression = ReferenceEffectExpressionV1::Unary {
                operation: *operation,
                operand: Box::new(operand),
            };
            Ok(ReferenceSymbolicValueV2::Scalar(
                reference_fold_constant_v2(meter, &expression)?.unwrap_or(expression),
            ))
        }
        ReferenceValueV1::Cast {
            kind,
            source,
            target,
            operand,
        } => {
            let operand = symbolic_scalar_v2(symbolic_operand_v2(meter, environment, operand)?)?;
            Ok(ReferenceSymbolicValueV2::Scalar(
                ReferenceEffectExpressionV1::Cast {
                    kind: *kind,
                    source: *source,
                    target: *target,
                    operand: Box::new(operand),
                },
            ))
        }
        ReferenceValueV1::InputLength { reference_argument } => Ok(
            ReferenceSymbolicValueV2::Scalar(ReferenceEffectExpressionV1::InputLength {
                reference_argument: *reference_argument,
            }),
        ),
        ReferenceValueV1::SafeHelperCall {
            parameters,
            arguments,
            summary,
            ..
        } => {
            if parameters.len() != arguments.len() {
                return Err(ReferenceBindingErrorV1::new(
                    "authenticated helper summary argument count changed",
                ));
            }
            meter.rows::<ReferenceEffectExpressionV1>(arguments.len())?;
            let arguments = arguments
                .iter()
                .map(|argument| {
                    symbolic_scalar_v2(symbolic_operand_v2(meter, environment, argument)?)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut work = 0;
            let expression =
                substitute_helper_summary_v2(meter, summary, &arguments, &mut work, 0)?;
            Ok(ReferenceSymbolicValueV2::Scalar(
                reference_fold_constant_v2(meter, &expression)?.unwrap_or(expression),
            ))
        }
    }
}

fn symbolic_operand_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    environment: &ReferenceSymbolicEnvironmentV2,
    operand: &ReferenceOperandV1,
) -> Result<ReferenceSymbolicValueV2, ReferenceBindingErrorV1> {
    meter.charge(1)?;
    meter.tree::<(u32, ReferenceSymbolicValueV2)>(environment.len())?;
    match operand {
        ReferenceOperandV1::Constant(constant) => Ok(ReferenceSymbolicValueV2::Scalar(
            ReferenceEffectExpressionV1::Constant(constant.clone()),
        )),
        ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place)
            if place.projection.is_empty() =>
        {
            let value = environment.get(&place.local).ok_or_else(|| {
                ReferenceBindingErrorV1::new(format!(
                    "loop-carried reference local _{} has no symbolic value",
                    place.local,
                ))
            })?;
            meter.clone_symbolic(value)
        }
        ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place)
            if matches!(
                place.projection.as_ref(),
                [ReferencePlaceProjectionV1::Field(0)]
            ) =>
        {
            match environment.get(&place.local) {
                Some(ReferenceSymbolicValueV2::CheckedPair { value, .. }) => {
                    meter.clone_expression(value)?;
                    Ok(ReferenceSymbolicValueV2::Scalar(value.clone()))
                }
                _ => Err(ReferenceBindingErrorV1::new(format!(
                    "reference _{}.0 is not one checked scalar value",
                    place.local,
                ))),
            }
        }
        ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place)
            if matches!(
                place.projection.as_ref(),
                [
                    ReferencePlaceProjectionV1::Dereference,
                    ReferencePlaceProjectionV1::Index(_)
                ]
            ) =>
        {
            let [
                ReferencePlaceProjectionV1::Dereference,
                ReferencePlaceProjectionV1::Index(index),
            ] = place.projection.as_ref()
            else {
                unreachable!()
            };
            let reference_argument = place.local.checked_sub(1).ok_or_else(|| {
                ReferenceBindingErrorV1::new(
                    "safe reference load uses the return-place local as its base",
                )
            })?;
            let index = environment.get(index).ok_or_else(|| {
                ReferenceBindingErrorV1::new(format!(
                    "safe reference load index _{index} has no symbolic value",
                ))
            })?;
            let index = meter.clone_symbolic(index)?;
            meter.rows::<ReferenceEffectExpressionV1>(1)?;
            Ok(ReferenceSymbolicValueV2::Scalar(
                ReferenceEffectExpressionV1::InputLoad {
                    reference_argument,
                    index: Box::new(symbolic_scalar_v2(index)?),
                },
            ))
        }
        ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place)
            if matches!(
                place.projection.as_ref(),
                [ReferencePlaceProjectionV1::Field(1)]
            ) =>
        {
            match environment.get(&place.local) {
                Some(ReferenceSymbolicValueV2::CheckedPair {
                    overflowed: Some(overflowed),
                    ..
                }) => Ok(ReferenceSymbolicValueV2::Scalar(
                    ReferenceEffectExpressionV1::Constant(ReferenceConstantV1::Scalar {
                        scalar: ReferenceScalarTypeV1::Bool,
                        bits: u128::from(*overflowed),
                    }),
                )),
                Some(ReferenceSymbolicValueV2::CheckedPair {
                    overflowed: None, ..
                }) => Err(ReferenceBindingErrorV1::new(
                    "checked loop arithmetic overflow is not statically proved; provide an authenticated range fact",
                )),
                _ => Err(ReferenceBindingErrorV1::new(format!(
                    "reference _{}.1 is not one checked scalar overflow flag",
                    place.local,
                ))),
            }
        }
        ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place) => {
            Err(ReferenceBindingErrorV1::new(format!(
                "reference scalar expression reads projection {:?}; slice and pointer reads require an independently bound GPU load symbol",
                place.projection,
            )))
        }
    }
}

fn symbolic_scalar_v2(
    value: ReferenceSymbolicValueV2,
) -> Result<ReferenceEffectExpressionV1, ReferenceBindingErrorV1> {
    match value {
        ReferenceSymbolicValueV2::Scalar(expression) => Ok(expression),
        ReferenceSymbolicValueV2::CheckedPair { .. } => Err(ReferenceBindingErrorV1::new(
            "checked arithmetic pair is used without selecting its value or overflow field",
        )),
    }
}

fn require_symbolic_value_budget_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    value: &ReferenceSymbolicValueV2,
) -> Result<(), ReferenceBindingErrorV1> {
    match value {
        ReferenceSymbolicValueV2::Scalar(expression)
        | ReferenceSymbolicValueV2::CheckedPair {
            value: expression, ..
        } => require_symbolic_expression_budget_v2(meter, expression),
    }
}

fn require_symbolic_expression_budget_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    expression: &ReferenceEffectExpressionV1,
) -> Result<(), ReferenceBindingErrorV1> {
    symbolic_expression_nodes_v2(meter, expression).map(|_| ())
}

fn symbolic_expression_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    expression: &ReferenceEffectExpressionV1,
) -> Result<usize, ReferenceBindingErrorV1> {
    meter.rows::<(&ReferenceEffectExpressionV1, usize)>(1)?;
    let mut pending = vec![(expression, 0_usize)];
    let mut nodes = 0_usize;
    while let Some((expression, depth)) = pending.pop() {
        meter.charge(1)?;
        if depth > fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference symbolic expression exceeds depth {}",
                fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2,
            )));
        }
        nodes = nodes.checked_add(1).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference symbolic expression node count overflowed")
        })?;
        if nodes > MAX_REFERENCE_EXPRESSION_NODES_V1 {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference symbolic expression exceeds {MAX_REFERENCE_EXPRESSION_NODES_V1} nodes",
            )));
        }
        match expression {
            ReferenceEffectExpressionV1::Binary { lhs, rhs, .. } => {
                meter.grow::<(&ReferenceEffectExpressionV1, usize)>(pending.len())?;
                pending.push((lhs, depth + 1));
                pending.push((rhs, depth + 1));
            }
            ReferenceEffectExpressionV1::Unary { operand, .. }
            | ReferenceEffectExpressionV1::Cast { operand, .. }
            | ReferenceEffectExpressionV1::InputLoad { index: operand, .. } => {
                meter.grow::<(&ReferenceEffectExpressionV1, usize)>(pending.len())?;
                pending.push((operand, depth + 1));
            }
            ReferenceEffectExpressionV1::PointCoordinate { .. }
            | ReferenceEffectExpressionV1::KernelScalarArgument { .. }
            | ReferenceEffectExpressionV1::InputLength { .. }
            | ReferenceEffectExpressionV1::Constant(_) => {}
        }
    }
    Ok(nodes)
}

fn symbolic_value_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    value: &ReferenceSymbolicValueV2,
) -> Result<usize, ReferenceBindingErrorV1> {
    match value {
        ReferenceSymbolicValueV2::Scalar(expression)
        | ReferenceSymbolicValueV2::CheckedPair {
            value: expression, ..
        } => symbolic_expression_nodes_v2(meter, expression),
    }
}

fn symbolic_environment_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    environment: &ReferenceSymbolicEnvironmentV2,
) -> Result<usize, ReferenceBindingErrorV1> {
    meter.charge(environment.len())?;
    environment.values().try_fold(0_usize, |total, value| {
        total
            .checked_add(symbolic_value_nodes_v2(meter, value)?)
            .ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference symbolic environment node count overflowed")
            })
    })
}

fn symbolic_predicate_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    predicate: &ReferencePathPredicateV1,
) -> Result<usize, ReferenceBindingErrorV1> {
    meter.charge(predicate.clauses.len())?;
    predicate.clauses.iter().try_fold(0_usize, |total, clause| {
        meter.charge(clause.atoms.len())?;
        clause.atoms.iter().try_fold(total, |total, atom| {
            let expression = match atom {
                ReferenceGuardAtomV1::SwitchValueSet { discriminant, .. } => discriminant,
                ReferenceGuardAtomV1::Assert { condition, .. } => condition,
            };
            total
                .checked_add(symbolic_expression_nodes_v2(meter, expression)?)
                .ok_or_else(|| {
                    ReferenceBindingErrorV1::new(
                        "reference symbolic predicate node count overflowed",
                    )
                })
        })
    })
}

fn symbolic_trace_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    trace: &ReferenceLoopTraceV2,
) -> Result<usize, ReferenceBindingErrorV1> {
    meter.charge(trace.transitions.len())?;
    meter.charge(trace.variants.len())?;
    let mut nodes = symbolic_environment_nodes_v2(meter, &trace.initial)?;
    for environment in &trace.transitions {
        nodes = nodes
            .checked_add(symbolic_environment_nodes_v2(meter, environment)?)
            .ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference symbolic trace node count overflowed")
            })?;
    }
    for expression in &trace.variants {
        nodes = nodes
            .checked_add(symbolic_expression_nodes_v2(meter, expression)?)
            .ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference symbolic trace node count overflowed")
            })?;
    }
    Ok(nodes)
}

fn symbolic_state_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    state: &ReferenceSymbolicStateV2,
) -> Result<usize, ReferenceBindingErrorV1> {
    meter.charge(state.traces.len())?;
    let mut nodes = symbolic_environment_nodes_v2(meter, &state.environment)?
        .checked_add(symbolic_predicate_nodes_v2(meter, &state.guard)?)
        .ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference symbolic state node count overflowed")
        })?;
    for trace in state.traces.values() {
        nodes = nodes
            .checked_add(symbolic_trace_nodes_v2(meter, trace)?)
            .ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference symbolic state node count overflowed")
            })?;
    }
    Ok(nodes)
}

fn dispatch_symbolic_edge_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    pending: &mut VecDeque<ReferenceSymbolicStateV2>,
    mut state: ReferenceSymbolicStateV2,
    (source, target): (u32, u32),
    backedges: &BTreeSet<(u32, u32)>,
    loop_nodes: &BTreeMap<(u32, u32), BTreeSet<u32>>,
    work_budget: &mut ReferenceSymbolicWorkBudgetV2,
) -> Result<(), ReferenceBindingErrorV1> {
    meter.tree::<(u32, u32)>(backedges.len())?;
    if backedges.contains(&(source, target)) {
        meter.tree::<((u32, u32), ReferenceLoopTraceV2)>(state.traces.len())?;
        let trace = state.traces.get_mut(&(target, source)).ok_or_else(|| {
            ReferenceBindingErrorV1::new(format!(
                "reference backedge {source}->{target} has no canonical loop trace",
            ))
        })?;
        work_budget.charge_environment_v2(meter, &state.environment)?;
        meter.grow::<ReferenceSymbolicEnvironmentV2>(trace.transitions.len())?;
        trace.transitions.push(state.environment.clone());
        if trace.transitions.len() > MAX_REFERENCE_LOOP_ITERATIONS_V2 {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference loop <header={target}, latch={source}> exceeds {MAX_REFERENCE_LOOP_ITERATIONS_V2} iterations",
            )));
        }
    }
    meter.charge(state.traces.len())?;
    if let Some(trace) = state
        .traces
        .values_mut()
        .find(|trace| trace.header == source)
    {
        meter.tree::<((u32, u32), BTreeSet<u32>)>(loop_nodes.len())?;
        let nodes = loop_nodes
            .get(&(trace.latch, trace.header))
            .ok_or_else(|| ReferenceBindingErrorV1::new("reference natural-loop nodes vanished"))?;
        meter.tree::<u32>(nodes.len())?;
        if !nodes.contains(&target) {
            trace.exit = Some(target);
        }
    }
    state.block = target;
    meter.grow::<ReferenceSymbolicStateV2>(pending.len())?;
    pending.push_back(state);
    Ok(())
}

fn reference_cfg_backedges_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    effect_ir: &ReferenceEffectIrV1,
) -> Result<BTreeSet<(u32, u32)>, ReferenceBindingErrorV1> {
    let count = effect_ir.blocks.len();
    if count == 0 {
        return Err(ReferenceBindingErrorV1::new(
            "reference effect IR has no entry block",
        ));
    }
    meter.rows::<usize>(count)?;
    meter.rows::<usize>(
        count
            .checked_mul(count)
            .ok_or_else(|| ReferenceBindingErrorV1::new("reference dominator storage overflow"))?,
    )?;
    meter.rows::<BTreeSet<usize>>(count)?;
    meter.rows::<Vec<usize>>(count)?;
    let all = (0..count).collect::<BTreeSet<_>>();
    let mut dominators = vec![all.clone(); count];
    dominators[0] = BTreeSet::from([0]);
    let mut predecessors = vec![Vec::new(); count];
    for block in &effect_ir.blocks {
        meter.charge(1)?;
        for successor in reference_successors_v1(&block.terminator) {
            meter.charge(1)?;
            let successor = successor as usize;
            if successor >= count {
                return Err(ReferenceBindingErrorV1::new(
                    "reference CFG successor is outside the block table",
                ));
            }
            meter.grow::<usize>(predecessors[successor].len())?;
            predecessors[successor].push(block.block as usize);
        }
    }
    loop {
        meter.charge(1)?;
        let mut changed = false;
        for block in 1..count {
            meter.charge(1)?;
            let mut next = if let Some(first) = predecessors[block].first() {
                meter.rows::<usize>(dominators[*first].len())?;
                dominators[*first].clone()
            } else {
                BTreeSet::new()
            };
            for predecessor in predecessors[block].iter().skip(1) {
                meter.product(
                    reference_extraction_work_v1::add(next.len(), 1)?,
                    reference_extraction_work_v1::add(dominators[*predecessor].len(), 1)?,
                )?;
                meter.rows::<usize>(next.len().min(dominators[*predecessor].len()))?;
                next = next
                    .intersection(&dominators[*predecessor])
                    .copied()
                    .collect();
            }
            meter.tree::<usize>(next.len())?;
            next.insert(block);
            meter.charge(next.len())?;
            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut backedges = BTreeSet::new();
    for block in &effect_ir.blocks {
        meter.charge(1)?;
        for successor in reference_successors_v1(&block.terminator) {
            meter.tree::<usize>(dominators[block.block as usize].len())?;
            if dominators[block.block as usize].contains(&(successor as usize)) {
                meter.tree::<(u32, u32)>(backedges.len())?;
                backedges.insert((block.block, successor));
            }
        }
    }
    Ok(backedges)
}

fn reference_natural_loop_nodes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    effect_ir: &ReferenceEffectIrV1,
    backedges: &BTreeSet<(u32, u32)>,
) -> Result<BTreeMap<(u32, u32), BTreeSet<u32>>, ReferenceBindingErrorV1> {
    meter.rows::<Vec<u32>>(effect_ir.blocks.len())?;
    let mut predecessors = vec![Vec::new(); effect_ir.blocks.len()];
    for block in &effect_ir.blocks {
        meter.charge(1)?;
        for successor in reference_successors_v1(&block.terminator) {
            meter.charge(1)?;
            let incoming = predecessors.get_mut(successor as usize).ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference loop successor is outside the block table")
            })?;
            meter.grow::<u32>(incoming.len())?;
            incoming.push(block.block);
        }
    }
    let mut result = BTreeMap::new();
    for (latch, header) in backedges {
        meter.rows::<u32>(3)?;
        let mut nodes = BTreeSet::from([*header, *latch]);
        let mut pending = vec![*latch];
        while let Some(block) = pending.pop() {
            meter.charge(1)?;
            for predecessor in &predecessors[block as usize] {
                meter.tree::<u32>(nodes.len())?;
                if nodes.insert(*predecessor) && *predecessor != *header {
                    meter.grow::<u32>(pending.len())?;
                    pending.push(*predecessor);
                }
            }
        }
        meter.tree::<((u32, u32), BTreeSet<u32>)>(result.len())?;
        result.insert((*latch, *header), nodes);
    }
    Ok(result)
}

fn validate_reference_loop_shapes_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    effect_ir: &ReferenceEffectIrV1,
    backedges: &BTreeSet<(u32, u32)>,
) -> Result<(), ReferenceBindingErrorV1> {
    let mut headers = BTreeSet::new();
    let loop_nodes = reference_natural_loop_nodes_v2(meter, effect_ir, backedges)?;
    reject_overlapping_reference_loops_v2(meter, &loop_nodes)?;
    for (latch, header) in backedges {
        meter.tree::<u32>(headers.len())?;
        if !headers.insert(*header) {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference loop header {header} has multiple latches; only one canonical recurrence is supported",
            )));
        }
        let header_block = effect_ir.blocks.get(*header as usize).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference loop header is outside the block table")
        })?;
        if !matches!(
            header_block.terminator,
            ReferenceTerminatorV1::Switch { .. }
        ) {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference loop header {header} does not end in one canonical condition switch",
            )));
        }
        let latch_block = effect_ir.blocks.get(*latch as usize).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference loop latch is outside the block table")
        })?;
        if !matches!(
            latch_block.terminator,
            ReferenceTerminatorV1::Goto { target } if target == *header
        ) {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference loop latch {latch} is not one unconditional edge to header {header}",
            )));
        }
        meter.tree::<((u32, u32), BTreeSet<u32>)>(loop_nodes.len())?;
        let nodes = &loop_nodes[&(*latch, *header)];
        let mut exits = BTreeSet::new();
        for node in nodes {
            meter.charge(1)?;
            for target in reference_successors_v1(&effect_ir.blocks[*node as usize].terminator) {
                meter.tree::<u32>(nodes.len())?;
                if !nodes.contains(&target) {
                    meter.tree::<(u32, u32)>(exits.len())?;
                    exits.insert((*node, target));
                }
            }
        }
        let Some((exit_source, _)) = exits.iter().next().copied() else {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference loop <header={header}, latch={latch}> has no finite exit",
            )));
        };
        if exits.len() != 1 || exit_source != *header {
            return Err(ReferenceBindingErrorV1::new(format!(
                "reference loop <header={header}, latch={latch}> must have exactly one header exit",
            )));
        }
    }
    Ok(())
}

fn reject_overlapping_reference_loops_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    loop_nodes: &BTreeMap<(u32, u32), BTreeSet<u32>>,
) -> Result<(), ReferenceBindingErrorV1> {
    meter.rows::<(&(u32, u32), &BTreeSet<u32>)>(loop_nodes.len())?;
    let loops = loop_nodes.iter().collect::<Vec<_>>();
    for (left_index, (left_identity, left_nodes)) in loops.iter().enumerate() {
        meter.charge(1)?;
        for (right_identity, right_nodes) in loops.iter().skip(left_index + 1) {
            meter.product(
                reference_extraction_work_v1::add(left_nodes.len(), 1)?,
                reference_extraction_work_v1::add(right_nodes.len(), 1)?,
            )?;
            if !left_nodes.is_disjoint(right_nodes) {
                return Err(ReferenceBindingErrorV1::new(format!(
                    "reference loops <header={}, latch={}> and <header={}, latch={}> overlap or nest; activation-specific recurrence summaries are not implemented",
                    left_identity.1, left_identity.0, right_identity.1, right_identity.0,
                )));
            }
        }
    }
    Ok(())
}

fn reference_loop_summary_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    trace: &ReferenceLoopTraceV2,
) -> Result<ReferenceLoopSummaryV2, ReferenceBindingErrorV1> {
    let units = meter.trace_units(trace)?;
    let passes = trace
        .initial
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| ReferenceBindingErrorV1::new("reference loop summary work overflow"))?;
    meter.product(passes, units)?;
    meter.rows::<u32>(trace.initial.len())?;
    let exit = trace.exit.ok_or_else(|| {
        ReferenceBindingErrorV1::new(format!(
            "reference loop <header={}, latch={}> has no authenticated exit",
            trace.header, trace.latch,
        ))
    })?;
    // Filtering BTreeMap keys preserves the canonical carried-local order.
    let carried_locals = trace
        .initial
        .iter()
        .filter_map(|(local, initial)| {
            trace
                .transitions
                .iter()
                .filter_map(|transition| transition.get(local))
                .any(|next| next != initial)
                .then_some(*local)
        })
        .collect::<Vec<_>>();
    let mut initial_digest = Sha256::new();
    initial_digest.update(b"fe2o3/reference-loop-initial/v2\0");
    let mut transition_digest = Sha256::new();
    transition_digest.update(b"fe2o3/reference-loop-transition/v2\0");
    for local in &carried_locals {
        initial_digest.update(local.to_le_bytes());
        digest_symbolic_value_v2(
            &mut initial_digest,
            trace.initial.get(local).ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference carried local has no initial value")
            })?,
        );
        transition_digest.update(local.to_le_bytes());
        for transition in &trace.transitions {
            digest_symbolic_value_v2(
                &mut transition_digest,
                transition.get(local).ok_or_else(|| {
                    ReferenceBindingErrorV1::new("reference carried local has no transition value")
                })?,
            );
        }
    }
    let mut variant_digest = Sha256::new();
    variant_digest.update(b"fe2o3/reference-loop-variant/v2\0");
    for variant in &trace.variants {
        digest_effect_expression_v1(&mut variant_digest, variant);
    }
    let iterations = u64::try_from(trace.transitions.len())
        .map_err(|_| ReferenceBindingErrorV1::new("reference loop iteration count exceeds u64"))?;
    let exact_iterations = if trace.maximum_iterations.is_some() {
        trace.exact_iterations
    } else {
        Some(iterations)
    };
    let maximum_iterations = trace.maximum_iterations.unwrap_or(iterations);
    Ok(ReferenceLoopSummaryV2 {
        header: trace.header,
        latch: trace.latch,
        exit,
        exact_iterations,
        maximum_iterations,
        carried_locals: carried_locals.into_boxed_slice(),
        initial_state_sha256: initial_digest.finalize().into(),
        transition_sha256: transition_digest.finalize().into(),
        variant_sha256: variant_digest.finalize().into(),
    })
}

fn digest_symbolic_value_v2(digest: &mut Sha256, value: &ReferenceSymbolicValueV2) {
    match value {
        ReferenceSymbolicValueV2::Scalar(expression) => {
            digest.update([0]);
            digest_effect_expression_v1(digest, expression);
        }
        ReferenceSymbolicValueV2::CheckedPair { value, overflowed } => {
            digest.update([1, overflowed.map(u8::from).unwrap_or(2)]);
            digest_effect_expression_v1(digest, value);
        }
    }
}

fn reference_constant_bits_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    expression: &ReferenceEffectExpressionV1,
) -> Result<Option<(ReferenceScalarTypeV1, u128)>, ReferenceBindingErrorV1> {
    meter.product(4, meter.expression(expression)?)?;
    Ok(reference_constant_bits_prepaid_v2(expression))
}

fn reference_fold_constant_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    expression: &ReferenceEffectExpressionV1,
) -> Result<Option<ReferenceEffectExpressionV1>, ReferenceBindingErrorV1> {
    // Bound recursion and prepay all folding visits before evaluating the tree.
    meter.product(4, meter.expression(expression)?)?;
    Ok(reference_fold_constant_prepaid_v2(expression))
}

fn reference_checked_overflow_v2(
    meter: &ReferenceExtractionWorkV1<'_>,
    operation: ReferenceBinaryOpV1,
    lhs: &ReferenceEffectExpressionV1,
    rhs: &ReferenceEffectExpressionV1,
) -> Result<Option<bool>, ReferenceBindingErrorV1> {
    meter.product(4, meter.expression(lhs)?)?;
    meter.product(4, meter.expression(rhs)?)?;
    Ok(reference_checked_overflow_prepaid_v2(operation, lhs, rhs))
}

fn reference_constant_bits_prepaid_v2(
    expression: &ReferenceEffectExpressionV1,
) -> Option<(ReferenceScalarTypeV1, u128)> {
    let folded = reference_fold_constant_prepaid_v2(expression)?;
    let ReferenceEffectExpressionV1::Constant(ReferenceConstantV1::Scalar { scalar, bits }) =
        folded
    else {
        return None;
    };
    Some((scalar, bits))
}

fn reference_fold_constant_prepaid_v2(
    expression: &ReferenceEffectExpressionV1,
) -> Option<ReferenceEffectExpressionV1> {
    match expression {
        ReferenceEffectExpressionV1::Constant(_) => Some(expression.clone()),
        ReferenceEffectExpressionV1::Binary {
            operation,
            lhs,
            rhs,
            ..
        } => {
            let (lhs_scalar, lhs) = reference_constant_bits_prepaid_v2(lhs)?;
            let (rhs_scalar, rhs) = reference_constant_bits_prepaid_v2(rhs)?;
            if lhs_scalar != rhs_scalar {
                return None;
            }
            if !matches!(
                lhs_scalar,
                ReferenceScalarTypeV1::Bool
                    | ReferenceScalarTypeV1::U8
                    | ReferenceScalarTypeV1::U16
                    | ReferenceScalarTypeV1::U32
                    | ReferenceScalarTypeV1::U64
                    | ReferenceScalarTypeV1::Usize
            ) {
                return None;
            }
            let comparison = matches!(
                operation,
                ReferenceBinaryOpV1::Equal
                    | ReferenceBinaryOpV1::LessThan
                    | ReferenceBinaryOpV1::LessEqual
                    | ReferenceBinaryOpV1::NotEqual
                    | ReferenceBinaryOpV1::GreaterEqual
                    | ReferenceBinaryOpV1::GreaterThan
            );
            let bits = match operation {
                ReferenceBinaryOpV1::Add => lhs.wrapping_add(rhs),
                ReferenceBinaryOpV1::Subtract => lhs.wrapping_sub(rhs),
                ReferenceBinaryOpV1::Multiply => lhs.wrapping_mul(rhs),
                ReferenceBinaryOpV1::Divide if rhs != 0 => lhs / rhs,
                ReferenceBinaryOpV1::Remainder if rhs != 0 => lhs % rhs,
                ReferenceBinaryOpV1::BitXor => lhs ^ rhs,
                ReferenceBinaryOpV1::BitAnd => lhs & rhs,
                ReferenceBinaryOpV1::BitOr => lhs | rhs,
                ReferenceBinaryOpV1::ShiftLeft if rhs < 128 => lhs << rhs,
                ReferenceBinaryOpV1::ShiftRight if rhs < 128 => lhs >> rhs,
                ReferenceBinaryOpV1::Equal => u128::from(lhs == rhs),
                ReferenceBinaryOpV1::LessThan => u128::from(lhs < rhs),
                ReferenceBinaryOpV1::LessEqual => u128::from(lhs <= rhs),
                ReferenceBinaryOpV1::NotEqual => u128::from(lhs != rhs),
                ReferenceBinaryOpV1::GreaterEqual => u128::from(lhs >= rhs),
                ReferenceBinaryOpV1::GreaterThan => u128::from(lhs > rhs),
                _ => return None,
            };
            let scalar = if comparison {
                ReferenceScalarTypeV1::Bool
            } else {
                lhs_scalar
            };
            let bits = bits & reference_scalar_mask_v2(scalar)?;
            Some(ReferenceEffectExpressionV1::Constant(
                ReferenceConstantV1::Scalar { scalar, bits },
            ))
        }
        ReferenceEffectExpressionV1::Unary { operation, operand } => {
            let (scalar, bits) = reference_constant_bits_prepaid_v2(operand)?;
            let mask = reference_scalar_mask_v2(scalar)?;
            let bits = match operation {
                ReferenceUnaryOpV1::Not => (!bits) & mask,
                ReferenceUnaryOpV1::Negate => (!bits).wrapping_add(1) & mask,
            };
            Some(ReferenceEffectExpressionV1::Constant(
                ReferenceConstantV1::Scalar { scalar, bits },
            ))
        }
        ReferenceEffectExpressionV1::PointCoordinate { .. }
        | ReferenceEffectExpressionV1::KernelScalarArgument { .. }
        | ReferenceEffectExpressionV1::InputLoad { .. }
        | ReferenceEffectExpressionV1::InputLength { .. }
        | ReferenceEffectExpressionV1::Cast { .. } => None,
    }
}

fn reference_checked_overflow_prepaid_v2(
    operation: ReferenceBinaryOpV1,
    lhs: &ReferenceEffectExpressionV1,
    rhs: &ReferenceEffectExpressionV1,
) -> Option<bool> {
    let (lhs_scalar, lhs) = reference_constant_bits_prepaid_v2(lhs)?;
    let (rhs_scalar, rhs) = reference_constant_bits_prepaid_v2(rhs)?;
    if lhs_scalar != rhs_scalar {
        return None;
    }
    if !matches!(
        lhs_scalar,
        ReferenceScalarTypeV1::U8
            | ReferenceScalarTypeV1::U16
            | ReferenceScalarTypeV1::U32
            | ReferenceScalarTypeV1::U64
            | ReferenceScalarTypeV1::Usize
    ) {
        return None;
    }
    let mask = reference_scalar_mask_v2(lhs_scalar)?;
    match operation {
        ReferenceBinaryOpV1::Add => Some(lhs.checked_add(rhs).is_none_or(|value| value > mask)),
        ReferenceBinaryOpV1::Subtract => Some(lhs < rhs),
        ReferenceBinaryOpV1::Multiply => {
            Some(lhs.checked_mul(rhs).is_none_or(|value| value > mask))
        }
        _ => None,
    }
}

fn reference_scalar_mask_v2(scalar: ReferenceScalarTypeV1) -> Option<u128> {
    let bits = match scalar {
        ReferenceScalarTypeV1::Bool => 1,
        ReferenceScalarTypeV1::U8 | ReferenceScalarTypeV1::I8 => 8,
        ReferenceScalarTypeV1::U16 | ReferenceScalarTypeV1::I16 => 16,
        ReferenceScalarTypeV1::U32 | ReferenceScalarTypeV1::I32 | ReferenceScalarTypeV1::F32 => 32,
        ReferenceScalarTypeV1::U64
        | ReferenceScalarTypeV1::Usize
        | ReferenceScalarTypeV1::I64
        | ReferenceScalarTypeV1::Isize
        | ReferenceScalarTypeV1::F64 => 64,
    };
    Some((1_u128 << bits) - 1)
}

/// Descriptive provenance only. Neither variant carries a live admission token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReferenceBindingOriginV1 {
    SourceRegistration(String),
    ReferenceEnrollment(fe2o3_verifier::portable_reference_v1::codec::ReferenceEnrollmentOriginV1),
}

impl ReferenceBindingOriginV1 {
    pub(crate) fn source_registration_v1(&self) -> Option<&str> {
        match self {
            Self::SourceRegistration(path) => Some(path),
            Self::ReferenceEnrollment(_) => None,
        }
    }

    #[cfg(test)]
    fn source_registration_mut_v1(&mut self) -> Option<&mut String> {
        match self {
            Self::SourceRegistration(path) => Some(path),
            Self::ReferenceEnrollment(_) => None,
        }
    }

    pub(crate) fn retained_payload_bytes_v1(&self) -> usize {
        match self {
            Self::SourceRegistration(path) => path.len(),
            Self::ReferenceEnrollment(_) => 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AuthenticatedReferenceEffectBindingV1 {
    pub(crate) origin: ReferenceBindingOriginV1,
    pub(crate) logical_kernel_name: String,
    pub(crate) kernel: ReferenceFunctionIdentityV1,
    pub(crate) reference: ReferenceFunctionIdentityV1,
    pub(crate) signature_preimage: ReferenceLogicalSignaturePreimageV1,
    pub(crate) effect_ir_sha256: [u8; 32],
    pub(crate) effect_ir: ReferenceEffectIrV1,
    pub(crate) observable_output_writes: Box<[ReferenceOutputWriteV1]>,
}

#[derive(Debug, Default)]
pub(crate) struct AuthenticatedReferenceEffectBindingsV1 {
    bindings: Box<[AuthenticatedReferenceEffectBindingV1]>,
}

impl AuthenticatedReferenceEffectBindingsV1 {
    pub(crate) fn new(bindings: Vec<AuthenticatedReferenceEffectBindingV1>) -> Self {
        Self {
            bindings: bindings.into_boxed_slice(),
        }
    }

    pub(crate) fn as_slice(&self) -> &[AuthenticatedReferenceEffectBindingV1] {
        &self.bindings
    }
}

trait ReferenceBindingDiagnosticV1 {
    fn at(tcx: TyCtxt<'_>, body: &Body<'_>, block: usize, reason: impl fmt::Display) -> Self;
}

impl ReferenceBindingDiagnosticV1 for ReferenceBindingErrorV1 {
    fn at(tcx: TyCtxt<'_>, body: &Body<'_>, block: usize, reason: impl fmt::Display) -> Self {
        let span = body.basic_blocks[rustc_middle::mir::BasicBlock::from_usize(block)]
            .terminator()
            .source_info
            .span;
        Self::new(format!(
            "unsupported safe Rust reference MIR at {}: {reason}",
            tcx.sess.source_map().span_to_diagnostic_string(span),
        ))
    }
}

fn lower_reference_effect_ir_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    reference: Instance<'tcx>,
    relations: Vec<ReferenceArgumentRelationV1>,
) -> Result<ReferenceEffectIrV1, ReferenceBindingErrorV1> {
    let body = tcx.instance_mir(reference.def);
    if body.basic_blocks.len() > MAX_REFERENCE_BLOCKS_V1 {
        return Err(ReferenceBindingErrorV1::new(format!(
            "safe Rust reference has {} MIR blocks; maximum is {MAX_REFERENCE_BLOCKS_V1}",
            body.basic_blocks.len(),
        )));
    }
    meter.charge(body.basic_blocks.len())?;
    let statement_count = body
        .basic_blocks
        .iter()
        .map(|block| block.statements.len())
        .try_fold(0_usize, usize::checked_add)
        .ok_or_else(|| ReferenceBindingErrorV1::new("reference statement count overflowed"))?;
    if statement_count > MAX_REFERENCE_STATEMENTS_V1 {
        return Err(ReferenceBindingErrorV1::new(format!(
            "safe Rust reference has {statement_count} MIR statements; maximum is {MAX_REFERENCE_STATEMENTS_V1}",
        )));
    }
    for (local, declaration) in body.local_decls.iter_enumerated() {
        meter.charge(1)?;
        if let TyKind::Tuple(fields) = declaration.ty.kind() {
            meter.charge(fields.len())?;
        }
        if !supported_local_type_v1(declaration.ty) {
            return Err(ReferenceBindingErrorV1::new(format!(
                "unsupported safe Rust reference local _{} type '{}'; V1 accepts unit, scalar values, scalar tuples, and references or slices of scalars",
                local.as_usize(),
                declaration.ty,
            )));
        }
    }
    meter.rows::<ReferenceBlockV1>(body.basic_blocks.len())?;
    let mut blocks = Vec::with_capacity(body.basic_blocks.len());
    for (block_id, block) in body.basic_blocks.iter_enumerated() {
        meter.charge(1)?;
        let block_index = block_id.as_usize();
        let mut assignments = Vec::new();
        for (statement_index, statement) in block.statements.iter().enumerate() {
            meter.charge(1)?;
            match &statement.kind {
                StatementKind::Assign(assignment) => {
                    let (destination, value) = &**assignment;
                    meter.grow::<ReferenceAssignmentV1>(assignments.len())?;
                    assignments.push(ReferenceAssignmentV1 {
                        statement: u32::try_from(statement_index).map_err(|_| {
                            ReferenceBindingErrorV1::new("reference statement index exceeds u32")
                        })?,
                        destination: lower_place_v1(meter, tcx, body, *destination, block_index)?,
                        value: lower_rvalue_v1(meter, tcx, body, value, block_index)?,
                    });
                }
                StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Nop => {}
                unsupported => {
                    return Err(ReferenceBindingErrorV1::at(
                        tcx,
                        body,
                        block_index,
                        format_args!(
                            "statement operation '{unsupported:?}' is outside reference-effect V1"
                        ),
                    ));
                }
            }
        }
        meter.charge(1)?;
        let terminator = match &block.terminator().kind {
            TerminatorKind::Return => ReferenceTerminatorV1::Return,
            TerminatorKind::Goto { target } => ReferenceTerminatorV1::Goto {
                target: target.as_u32(),
            },
            TerminatorKind::SwitchInt { discr, targets } => {
                meter.rows::<(u128, u32)>(targets.iter().len())?;
                ReferenceTerminatorV1::Switch {
                    discriminant: lower_operand_v1(meter, tcx, body, discr, block_index)?,
                    values: targets
                        .iter()
                        .map(|(value, target)| (value, target.as_u32()))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    otherwise: targets.otherwise().as_u32(),
                }
            }
            TerminatorKind::Assert {
                cond,
                expected,
                target,
                unwind: UnwindAction::Unreachable,
                msg,
            } => {
                let bounds_check = match &**msg {
                    AssertMessage::BoundsCheck { len, index } => Some(ReferenceBoundsCheckV1 {
                        index: lower_operand_v1(meter, tcx, body, index, block_index)?,
                        length: lower_operand_v1(meter, tcx, body, len, block_index)?,
                    }),
                    _ => None,
                };
                ReferenceTerminatorV1::Assert {
                    condition: lower_operand_v1(meter, tcx, body, cond, block_index)?,
                    expected: *expected,
                    success: target.as_u32(),
                    bounds_check,
                }
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(target),
                unwind: UnwindAction::Continue | UnwindAction::Unreachable,
                ..
            } => {
                meter.grow::<ReferenceAssignmentV1>(assignments.len())?;
                assignments.push(lower_safe_scalar_helper_call_v2(
                    meter,
                    tcx,
                    reference,
                    ReferenceHelperCallSiteV2 {
                        body,
                        block: block_index,
                        statement: u32::try_from(block.statements.len()).map_err(|_| {
                            ReferenceBindingErrorV1::new(
                                "synthetic helper-call statement index exceeds u32",
                            )
                        })?,
                        destination: *destination,
                    },
                    func,
                    args,
                )?);
                ReferenceTerminatorV1::Goto {
                    target: target.as_u32(),
                }
            }
            TerminatorKind::Call { .. } => {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block_index,
                    "only a returning direct call to one safe local pure scalar helper is supported",
                ));
            }
            unsupported => {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block_index,
                    format_args!("terminator '{unsupported:?}' is outside reference-effect V1"),
                ));
            }
        };
        blocks.push(ReferenceBlockV1 {
            block: block_id.as_u32(),
            assignments: assignments.into_boxed_slice(),
            terminator,
        });
    }
    let mut effect_ir = ReferenceEffectIrV1 {
        argument_count: u32::try_from(body.arg_count)
            .map_err(|_| ReferenceBindingErrorV1::new("reference argument count exceeds u32"))?,
        local_count: u32::try_from(body.local_decls.len())
            .map_err(|_| ReferenceBindingErrorV1::new("reference local count exceeds u32"))?,
        relations: relations.into_boxed_slice(),
        blocks: blocks.into_boxed_slice(),
        loop_summaries: Box::default(),
        observable_output_effects: Box::default(),
    };
    let backedges = reference_cfg_backedges_v2(meter, &effect_ir)?;
    if backedges.is_empty() {
        effect_ir.observable_output_effects = effect_ir
            .observable_output_writes_v1(meter)?
            .into_boxed_slice();
    } else {
        validate_reference_loop_shapes_v2(meter, &effect_ir, &backedges)?;
        let (effects, summaries) =
            effect_ir.observable_output_writes_with_loops_v2(meter, &backedges)?;
        effect_ir.observable_output_effects = effects.into_boxed_slice();
        effect_ir.loop_summaries = summaries.into_boxed_slice();
    }
    Ok(effect_ir)
}

struct ReferenceHelperCallSiteV2<'a, 'tcx> {
    body: &'a Body<'tcx>,
    block: usize,
    statement: u32,
    destination: Place<'tcx>,
}

fn lower_safe_scalar_helper_call_v2<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    caller: Instance<'tcx>,
    site: ReferenceHelperCallSiteV2<'_, 'tcx>,
    function: &Operand<'tcx>,
    arguments: &[Spanned<Operand<'tcx>>],
) -> Result<ReferenceAssignmentV1, ReferenceBindingErrorV1> {
    let ReferenceHelperCallSiteV2 {
        body,
        block,
        statement,
        destination,
    } = site;
    let Operand::Constant(function) = function else {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "indirect helper calls are outside authenticated reference semantics",
        ));
    };
    let TyKind::FnDef(definition, generic_arguments) = function.const_.ty().kind() else {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "helper call target is not one statically resolved function item",
        ));
    };
    let helper = Instance::try_resolve(
        tcx,
        TypingEnv::fully_monomorphized(),
        *definition,
        generic_arguments,
    )
    .map_err(|_| {
        ReferenceBindingErrorV1::at(tcx, body, block, "safe helper instance resolution failed")
    })?
    .ok_or_else(|| {
        ReferenceBindingErrorV1::at(tcx, body, block, "safe helper instance is not monomorphic")
    })?;
    if helper == caller {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "recursive safe reference helpers are unsupported",
        ));
    }
    authenticate_safe_local_reference_v1(meter, tcx, helper)?;
    let signature = instantiated_signature(tcx, helper);
    if signature.inputs().len() > MAX_REFERENCE_HELPER_ARGUMENTS_V2 {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            format_args!(
                "safe helper has {} arguments; maximum is {MAX_REFERENCE_HELPER_ARGUMENTS_V2}",
                signature.inputs().len(),
            ),
        ));
    }
    if signature.inputs().len() != arguments.len() {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "safe helper MIR argument count disagrees with its instantiated signature",
        ));
    }
    meter.rows::<ReferenceScalarTypeV1>(arguments.len())?;
    let parameters = signature
        .inputs()
        .iter()
        .copied()
        .zip(arguments)
        .enumerate()
        .map(|(index, (expected, argument))| {
            meter.charge(1)?;
            let scalar = scalar_type_v1(expected).ok_or_else(|| {
                ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    format_args!(
                        "safe helper argument {} type '{expected}' is not a supported scalar",
                        index + 1,
                    ),
                )
            })?;
            let actual = argument.node.ty(body, tcx);
            if actual != expected {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    format_args!(
                        "safe helper argument {} has type '{actual}', expected '{expected}'",
                        index + 1,
                    ),
                ));
            }
            Ok(scalar)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result_ty = signature.output();
    let result = scalar_type_v1(result_ty).ok_or_else(|| {
        ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            format_args!("safe helper result type '{result_ty}' is not a supported scalar"),
        )
    })?;
    if destination.ty(body, tcx).ty != result_ty {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "safe helper return destination type disagrees with its instantiated signature",
        ));
    }
    let summary = lower_safe_scalar_helper_summary_v2(meter, tcx, helper, &parameters)?;
    meter.rows::<ReferenceOperandV1>(arguments.len())?;
    meter.rows::<ReferenceEffectExpressionV1>(1)?;
    Ok(ReferenceAssignmentV1 {
        statement,
        destination: lower_place_v1(meter, tcx, body, destination, block)?,
        value: ReferenceValueV1::SafeHelperCall {
            helper: function_identity_v1(meter, tcx, helper)?,
            parameters: parameters.into_boxed_slice(),
            result,
            arguments: arguments
                .iter()
                .map(|argument| lower_operand_v1(meter, tcx, body, &argument.node, block))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
            summary: Box::new(summary),
        },
    })
}

fn lower_safe_scalar_helper_summary_v2<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    helper: Instance<'tcx>,
    parameters: &[ReferenceScalarTypeV1],
) -> Result<ReferenceEffectExpressionV1, ReferenceBindingErrorV1> {
    let body = tcx.instance_mir(helper.def);
    if body.basic_blocks.len() > MAX_REFERENCE_BLOCKS_V1 {
        return Err(ReferenceBindingErrorV1::new(format!(
            "safe helper '{}' has {} MIR blocks; maximum is {MAX_REFERENCE_BLOCKS_V1}",
            tcx.def_path_str(helper.def_id()),
            body.basic_blocks.len(),
        )));
    }
    // Preserve shared-work exhaustion instead of relabeling it as a loop refusal.
    reject_cycles_v1(meter, tcx, body)?;
    meter.charge(body.basic_blocks.len())?;
    let statement_count = body
        .basic_blocks
        .iter()
        .map(|block| block.statements.len())
        .try_fold(0_usize, usize::checked_add)
        .ok_or_else(|| ReferenceBindingErrorV1::new("safe helper statement count overflowed"))?;
    if statement_count > MAX_REFERENCE_STATEMENTS_V1 {
        return Err(ReferenceBindingErrorV1::new(format!(
            "safe helper '{}' has {statement_count} MIR statements; maximum is {MAX_REFERENCE_STATEMENTS_V1}",
            tcx.def_path_str(helper.def_id()),
        )));
    }
    for (local, declaration) in body.local_decls.iter_enumerated() {
        meter.charge(1)?;
        if let TyKind::Tuple(fields) = declaration.ty.kind() {
            meter.charge(fields.len())?;
        }
        if !supported_local_type_v1(declaration.ty) {
            return Err(ReferenceBindingErrorV1::new(format!(
                "safe helper '{}' local _{} type '{}' is unsupported",
                tcx.def_path_str(helper.def_id()),
                local.as_usize(),
                declaration.ty,
            )));
        }
    }
    meter.rows::<ReferenceBlockV1>(body.basic_blocks.len())?;
    let mut blocks = Vec::with_capacity(body.basic_blocks.len());
    for (block_id, block) in body.basic_blocks.iter_enumerated() {
        meter.charge(1)?;
        let block_index = block_id.as_usize();
        let mut assignments = Vec::new();
        for (statement_index, statement) in block.statements.iter().enumerate() {
            meter.charge(1)?;
            match &statement.kind {
                StatementKind::Assign(assignment) => {
                    let (destination, value) = &**assignment;
                    meter.grow::<ReferenceAssignmentV1>(assignments.len())?;
                    assignments.push(ReferenceAssignmentV1 {
                        statement: u32::try_from(statement_index).map_err(|_| {
                            ReferenceBindingErrorV1::new("safe helper statement index exceeds u32")
                        })?,
                        destination: lower_place_v1(meter, tcx, body, *destination, block_index)?,
                        value: lower_rvalue_v1(meter, tcx, body, value, block_index)?,
                    });
                }
                StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Nop => {}
                unsupported => {
                    return Err(ReferenceBindingErrorV1::at(
                        tcx,
                        body,
                        block_index,
                        format_args!("safe helper statement '{unsupported:?}' is unsupported"),
                    ));
                }
            }
        }
        meter.charge(1)?;
        let terminator = match &block.terminator().kind {
            TerminatorKind::Return => ReferenceTerminatorV1::Return,
            TerminatorKind::Goto { target } => ReferenceTerminatorV1::Goto {
                target: target.as_u32(),
            },
            TerminatorKind::Assert {
                cond,
                expected,
                target,
                unwind,
                ..
            } if matches!(unwind, UnwindAction::Continue | UnwindAction::Unreachable)
                && !*expected
                && matches!(
                    cond,
                    Operand::Copy(place) | Operand::Move(place)
                        if matches!(
                            place.projection.as_ref(),
                            [ProjectionElem::Field(field, _)] if field.as_u32() == 1
                        )
                ) =>
            {
                ReferenceTerminatorV1::Assert {
                    condition: lower_operand_v1(meter, tcx, body, cond, block_index)?,
                    expected: *expected,
                    success: target.as_u32(),
                    bounds_check: None,
                }
            }
            TerminatorKind::SwitchInt { .. } | TerminatorKind::Assert { .. } => {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block_index,
                    "safe scalar helper control flow is unsupported; only compiler checked-overflow assertions may be summarized",
                ));
            }
            TerminatorKind::Call { func, .. } => {
                if let Operand::Constant(function) = func
                    && let TyKind::FnDef(definition, generic_arguments) =
                        function.const_.ty().kind()
                    && let Ok(Some(callee)) = Instance::try_resolve(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        *definition,
                        generic_arguments,
                    )
                    && callee == helper
                {
                    return Err(ReferenceBindingErrorV1::at(
                        tcx,
                        body,
                        block_index,
                        "recursive safe scalar helper is unsupported",
                    ));
                }
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block_index,
                    "nested safe helper calls are unsupported; make the helper one pure scalar expression",
                ));
            }
            unsupported => {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block_index,
                    format_args!("safe helper terminator '{unsupported:?}' is unsupported"),
                ));
            }
        };
        blocks.push(ReferenceBlockV1 {
            block: block_id.as_u32(),
            assignments: assignments.into_boxed_slice(),
            terminator,
        });
    }
    meter.rows::<ReferenceArgumentRelationV1>(parameters.len())?;
    let relations = parameters
        .iter()
        .copied()
        .enumerate()
        .map(
            |(argument, scalar)| ReferenceArgumentRelationV1::ScalarInput {
                argument: u32::try_from(argument).unwrap_or(u32::MAX),
                scalar,
            },
        )
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let effect_ir = ReferenceEffectIrV1 {
        argument_count: u32::try_from(body.arg_count)
            .map_err(|_| ReferenceBindingErrorV1::new("safe helper argument count exceeds u32"))?,
        local_count: u32::try_from(body.local_decls.len())
            .map_err(|_| ReferenceBindingErrorV1::new("safe helper local count exceeds u32"))?,
        relations,
        blocks: blocks.into_boxed_slice(),
        loop_summaries: Box::default(),
        observable_output_effects: Box::default(),
    };
    ReferenceExpressionResolverV1::new(meter, &effect_ir)?.resolve_local_v1(meter, 0)
}

fn reject_cycles_v1(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'_>,
    body: &Body<'_>,
) -> Result<(), ReferenceBindingErrorV1> {
    meter.rows::<usize>(body.basic_blocks.len())?;
    let mut indegree = vec![0_usize; body.basic_blocks.len()];
    for block in body.basic_blocks.iter() {
        meter.charge(1)?;
        for successor in block.terminator().successors() {
            meter.charge(1)?;
            indegree[successor.as_usize()] = indegree[successor.as_usize()]
                .checked_add(1)
                .ok_or_else(|| ReferenceBindingErrorV1::new("reference CFG indegree overflowed"))?;
        }
    }
    meter.rows::<usize>(indegree.len())?;
    let mut pending = indegree
        .iter()
        .enumerate()
        .filter_map(|(block, degree)| (*degree == 0).then_some(block))
        .collect::<VecDeque<_>>();
    let mut visited = 0_usize;
    while let Some(block) = pending.pop_front() {
        meter.charge(1)?;
        visited += 1;
        for successor in body.basic_blocks[rustc_middle::mir::BasicBlock::from_usize(block)]
            .terminator()
            .successors()
        {
            meter.charge(1)?;
            let degree = &mut indegree[successor.as_usize()];
            *degree -= 1;
            if *degree == 0 {
                meter.grow::<usize>(pending.len())?;
                pending.push_back(successor.as_usize());
            }
        }
    }
    if visited != body.basic_blocks.len() {
        meter.charge(indegree.len())?;
        let cyclic = indegree
            .iter()
            .enumerate()
            .find_map(|(block, degree)| (*degree != 0).then_some(block))
            .unwrap_or(0);
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            cyclic,
            "cyclic control flow is outside the metered acyclic reference/helper extraction boundary",
        ));
    }
    Ok(())
}

fn supported_local_type_v1(ty: Ty<'_>) -> bool {
    // Rustc can retain unused never temporaries for ordinary loops. They carry
    // no value; lower_place_v1 rejects every executable access to such a local.
    if ty.is_unit() || ty.is_never() || scalar_type_v1(ty).is_some() {
        return true;
    }
    match *ty.kind() {
        TyKind::Ref(_, pointee, _) => match *pointee.kind() {
            TyKind::Slice(element) => scalar_type_v1(element).is_some(),
            _ => scalar_type_v1(pointee).is_some(),
        },
        TyKind::Tuple(fields) => fields.iter().all(|field| scalar_type_v1(field).is_some()),
        _ => false,
    }
}

fn lower_place_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
    block: usize,
) -> Result<ReferencePlaceV1, ReferenceBindingErrorV1> {
    meter.charge(1)?;
    let declaration = body
        .local_decls
        .get(place.local)
        .ok_or_else(|| ReferenceBindingErrorV1::new("reference place local is absent"))?;
    if declaration.ty.is_never() {
        return Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "reference value access to an uninhabited local is unsupported",
        ));
    }
    meter.rows::<ReferencePlaceProjectionV1>(place.projection.len())?;
    let mut projection = Vec::with_capacity(place.projection.len());
    for element in place.projection {
        meter.charge(1)?;
        projection.push(match element {
            ProjectionElem::Deref => ReferencePlaceProjectionV1::Dereference,
            ProjectionElem::Field(field, _) => ReferencePlaceProjectionV1::Field(field.as_u32()),
            ProjectionElem::Index(index) => ReferencePlaceProjectionV1::Index(index.as_u32()),
            ProjectionElem::ConstantIndex {
                offset,
                min_length,
                from_end,
            } => ReferencePlaceProjectionV1::ConstantIndex {
                offset,
                minimum_length: min_length,
                from_end,
            },
            unsupported => {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    format_args!(
                        "place projection '{unsupported:?}' is outside reference-effect V1"
                    ),
                ));
            }
        });
    }
    Ok(ReferencePlaceV1 {
        local: place.local.as_u32(),
        projection: projection.into_boxed_slice(),
    })
}

fn lower_operand_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
    block: usize,
) -> Result<ReferenceOperandV1, ReferenceBindingErrorV1> {
    meter.rows::<ReferenceOperandV1>(1)?;
    match operand {
        Operand::Copy(place) => Ok(ReferenceOperandV1::Copy(lower_place_v1(
            meter, tcx, body, *place, block,
        )?)),
        Operand::Move(place) => Ok(ReferenceOperandV1::Move(lower_place_v1(
            meter, tcx, body, *place, block,
        )?)),
        Operand::Constant(constant) => {
            let ty = constant.const_.ty();
            if ty.is_unit() {
                return Ok(ReferenceOperandV1::Constant(ReferenceConstantV1::ZeroSized));
            }
            let Some(scalar) = scalar_type_v1(ty) else {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    format_args!("constant type '{ty}' is outside reference-effect V1"),
                ));
            };
            let bits = constant
                .const_
                .try_eval_bits(tcx, TypingEnv::fully_monomorphized())
                .ok_or_else(|| {
                    ReferenceBindingErrorV1::at(tcx, body, block, "constant is not an exact scalar")
                })?;
            Ok(ReferenceOperandV1::Constant(ReferenceConstantV1::Scalar {
                scalar,
                bits,
            }))
        }
        Operand::RuntimeChecks(_) => Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "runtime-check pseudo-operands are outside reference-effect V1",
        )),
    }
}

fn lower_rvalue_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    value: &Rvalue<'tcx>,
    block: usize,
) -> Result<ReferenceValueV1, ReferenceBindingErrorV1> {
    meter.rows::<ReferenceValueV1>(1)?;
    match value {
        Rvalue::Use(operand) => Ok(ReferenceValueV1::Use(lower_operand_v1(
            meter, tcx, body, operand, block,
        )?)),
        Rvalue::BinaryOp(operation, operands) => {
            let checked = matches!(
                operation,
                BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
            );
            let (lhs, rhs) = &**operands;
            Ok(ReferenceValueV1::Binary {
                operation: lower_binary_op_v1(*operation).ok_or_else(|| {
                    ReferenceBindingErrorV1::at(
                        tcx,
                        body,
                        block,
                        format_args!(
                            "binary operation '{operation:?}' is outside reference-effect V1"
                        ),
                    )
                })?,
                lhs: lower_operand_v1(meter, tcx, body, lhs, block)?,
                rhs: lower_operand_v1(meter, tcx, body, rhs, block)?,
                checked,
            })
        }
        Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
            let (Operand::Copy(place) | Operand::Move(place)) = operand else {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    "slice metadata source is not one reference argument",
                ));
            };
            if !place.projection.is_empty() || place.local.as_u32() == 0 {
                return Err(ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    "slice metadata source is not one direct reference argument",
                ));
            }
            Ok(ReferenceValueV1::InputLength {
                reference_argument: place.local.as_u32() - 1,
            })
        }
        Rvalue::UnaryOp(operation, operand) => Ok(ReferenceValueV1::Unary {
            operation: match operation {
                UnOp::Not => ReferenceUnaryOpV1::Not,
                UnOp::Neg => ReferenceUnaryOpV1::Negate,
                UnOp::PtrMetadata => unreachable!(),
            },
            operand: lower_operand_v1(meter, tcx, body, operand, block)?,
        }),
        Rvalue::Cast(kind, operand, target) => {
            let source = scalar_type_v1(operand.ty(body, tcx)).ok_or_else(|| {
                ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    "cast source is outside reference-effect scalar semantics",
                )
            })?;
            let target = scalar_type_v1(*target).ok_or_else(|| {
                ReferenceBindingErrorV1::at(
                    tcx,
                    body,
                    block,
                    "cast target is outside reference-effect scalar semantics",
                )
            })?;
            let kind = match kind {
                CastKind::IntToInt => ReferenceCastKindV1::Integer,
                CastKind::IntToFloat => ReferenceCastKindV1::IntegerToFloat,
                CastKind::FloatToFloat => ReferenceCastKindV1::FloatToFloat,
                CastKind::FloatToInt => ReferenceCastKindV1::FloatToIntegerSaturating,
                _ => {
                    return Err(ReferenceBindingErrorV1::at(
                        tcx,
                        body,
                        block,
                        format_args!(
                            "cast operation '{kind:?}' is outside reference-effect V2 numeric semantics"
                        ),
                    ));
                }
            };
            Ok(ReferenceValueV1::Cast {
                kind,
                source,
                target,
                operand: lower_operand_v1(meter, tcx, body, operand, block)?,
            })
        }
        Rvalue::Ref(..) => Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            "references constructed for helper memory access are outside pure scalar helper summaries",
        )),
        unsupported => Err(ReferenceBindingErrorV1::at(
            tcx,
            body,
            block,
            format_args!("rvalue '{unsupported:?}' is outside reference-effect V1"),
        )),
    }
}

fn lower_binary_op_v1(operation: BinOp) -> Option<ReferenceBinaryOpV1> {
    Some(match operation {
        BinOp::Add | BinOp::AddWithOverflow => ReferenceBinaryOpV1::Add,
        BinOp::Sub | BinOp::SubWithOverflow => ReferenceBinaryOpV1::Subtract,
        BinOp::Mul | BinOp::MulWithOverflow => ReferenceBinaryOpV1::Multiply,
        BinOp::Div => ReferenceBinaryOpV1::Divide,
        BinOp::Rem => ReferenceBinaryOpV1::Remainder,
        BinOp::BitXor => ReferenceBinaryOpV1::BitXor,
        BinOp::BitAnd => ReferenceBinaryOpV1::BitAnd,
        BinOp::BitOr => ReferenceBinaryOpV1::BitOr,
        BinOp::Shl => ReferenceBinaryOpV1::ShiftLeft,
        BinOp::Shr => ReferenceBinaryOpV1::ShiftRight,
        BinOp::Eq => ReferenceBinaryOpV1::Equal,
        BinOp::Lt => ReferenceBinaryOpV1::LessThan,
        BinOp::Le => ReferenceBinaryOpV1::LessEqual,
        BinOp::Ne => ReferenceBinaryOpV1::NotEqual,
        BinOp::Ge => ReferenceBinaryOpV1::GreaterEqual,
        BinOp::Gt => ReferenceBinaryOpV1::GreaterThan,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_constant(bits: u128) -> ReferenceConstantV1 {
        ReferenceConstantV1::Scalar {
            scalar: ReferenceScalarTypeV1::U32,
            bits,
        }
    }

    fn scalar_operand(bits: u128) -> ReferenceOperandV1 {
        ReferenceOperandV1::Constant(scalar_constant(bits))
    }

    fn output_assignment(projection: Vec<ReferencePlaceProjectionV1>) -> ReferenceAssignmentV1 {
        ReferenceAssignmentV1 {
            statement: 0,
            destination: ReferencePlaceV1 {
                local: 3,
                projection: projection.into_boxed_slice(),
            },
            value: ReferenceValueV1::Use(scalar_operand(17)),
        }
    }

    fn guarded_point_reference_ir(
        output_projection: Vec<ReferencePlaceProjectionV1>,
    ) -> ReferenceEffectIrV1 {
        ReferenceEffectIrV1 {
            argument_count: 3,
            local_count: 4,
            relations: vec![
                ReferenceArgumentRelationV1::PointCoordinate {
                    reference_argument: 0,
                    axis: 0,
                },
                ReferenceArgumentRelationV1::ScalarInput {
                    argument: 0,
                    scalar: ReferenceScalarTypeV1::Bool,
                },
                ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                    argument: 1,
                    element: ReferenceScalarTypeV1::U32,
                },
            ]
            .into_boxed_slice(),
            blocks: vec![
                ReferenceBlockV1 {
                    block: 0,
                    assignments: Box::default(),
                    terminator: ReferenceTerminatorV1::Switch {
                        discriminant: ReferenceOperandV1::Copy(ReferencePlaceV1 {
                            local: 2,
                            projection: Box::default(),
                        }),
                        values: vec![(0, 2)].into_boxed_slice(),
                        otherwise: 1,
                    },
                },
                ReferenceBlockV1 {
                    block: 1,
                    assignments: vec![output_assignment(output_projection)].into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Return,
                },
                ReferenceBlockV1 {
                    block: 2,
                    assignments: Box::default(),
                    terminator: ReferenceTerminatorV1::Return,
                },
            ]
            .into_boxed_slice(),
            loop_summaries: Box::default(),
            observable_output_effects: Box::default(),
        }
    }

    #[test]
    fn derives_point_coordinate_guard_and_rhs_from_reference_ir() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let effect_ir = guarded_point_reference_ir(vec![ReferencePlaceProjectionV1::Dereference]);
        let writes = effect_ir.observable_output_writes_v1(meter).unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(
            writes[0].coordinate,
            ReferenceOutputCoordinateV1::LogicalPoint(
                vec![ReferenceEffectExpressionV1::PointCoordinate { axis: 0 }].into_boxed_slice(),
            )
        );
        assert_eq!(
            writes[0].rhs,
            ReferenceEffectExpressionV1::Constant(scalar_constant(17))
        );
        assert_eq!(
            writes[0].guard,
            ReferencePathPredicateV1 {
                clauses: vec![ReferenceGuardClauseV1 {
                    atoms: vec![ReferenceGuardAtomV1::SwitchValueSet {
                        discriminant: ReferenceEffectExpressionV1::KernelScalarArgument {
                            argument: 0,
                        },
                        values: vec![0].into_boxed_slice(),
                        inside_set: false,
                    }]
                    .into_boxed_slice(),
                }]
                .into_boxed_slice(),
            }
        );
    }

    #[test]
    fn target_only_output_relations_preserve_constant_writes() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let coordinate = guarded_point_reference_ir(vec![ReferencePlaceProjectionV1::Dereference]);
        let mut slice = guarded_point_reference_ir(vec![
            ReferencePlaceProjectionV1::Dereference,
            ReferencePlaceProjectionV1::Index(1),
        ]);
        slice.relations[2] = ReferenceArgumentRelationV1::DisjointOutputSlice {
            argument: 1,
            element: ReferenceScalarTypeV1::U32,
        };
        for effect_ir in [coordinate, slice] {
            let writes = effect_ir.observable_output_writes_v1(meter).unwrap();
            assert_eq!(writes.len(), 1);
            assert_eq!(writes[0].argument, 1);
            assert_eq!(
                writes[0].rhs,
                ReferenceEffectExpressionV1::Constant(scalar_constant(17)),
            );
        }
    }

    #[test]
    fn target_only_output_coordinate_rejects_reference_reads() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut effect_ir =
            guarded_point_reference_ir(vec![ReferencePlaceProjectionV1::Dereference]);
        effect_ir.blocks[1].assignments[0].value =
            ReferenceValueV1::Use(ReferenceOperandV1::Copy(ReferencePlaceV1 {
                local: 3,
                projection: vec![ReferencePlaceProjectionV1::Dereference].into_boxed_slice(),
            }));
        let error = effect_ir.observable_output_writes_v1(meter).unwrap_err();
        assert!(error.to_string().contains("unsupported place projection"));
    }

    #[test]
    fn target_only_output_slice_rejects_reference_reads() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let projection = vec![
            ReferencePlaceProjectionV1::Dereference,
            ReferencePlaceProjectionV1::Index(1),
        ];
        let mut effect_ir = guarded_point_reference_ir(projection.clone());
        effect_ir.relations[2] = ReferenceArgumentRelationV1::DisjointOutputSlice {
            argument: 1,
            element: ReferenceScalarTypeV1::U32,
        };
        effect_ir.blocks[1].assignments[0].value =
            ReferenceValueV1::Use(ReferenceOperandV1::Copy(ReferencePlaceV1 {
                local: 3,
                projection: projection.into_boxed_slice(),
            }));
        let error = effect_ir.observable_output_writes_v1(meter).unwrap_err();
        assert!(error.to_string().contains("not a shared-slice input"));
    }

    #[test]
    fn refuses_to_omit_an_unsupported_observable_output_projection() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let effect_ir = guarded_point_reference_ir(vec![
            ReferencePlaceProjectionV1::Dereference,
            ReferencePlaceProjectionV1::Field(0),
        ]);
        let error = effect_ir.observable_output_writes_v1(meter).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot omit a global output write")
        );
    }

    #[test]
    fn derives_multiple_observable_output_effects_without_collapsing_arguments() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let effect_ir = ReferenceEffectIrV1 {
            argument_count: 3,
            local_count: 4,
            relations: vec![
                ReferenceArgumentRelationV1::PointCoordinate {
                    reference_argument: 0,
                    axis: 0,
                },
                ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                    argument: 0,
                    element: ReferenceScalarTypeV1::U32,
                },
                ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                    argument: 1,
                    element: ReferenceScalarTypeV1::U32,
                },
            ]
            .into_boxed_slice(),
            blocks: vec![ReferenceBlockV1 {
                block: 0,
                assignments: vec![
                    ReferenceAssignmentV1 {
                        statement: 0,
                        destination: ReferencePlaceV1 {
                            local: 2,
                            projection: vec![ReferencePlaceProjectionV1::Dereference]
                                .into_boxed_slice(),
                        },
                        value: ReferenceValueV1::Use(scalar_operand(17)),
                    },
                    ReferenceAssignmentV1 {
                        statement: 1,
                        destination: ReferencePlaceV1 {
                            local: 3,
                            projection: vec![ReferencePlaceProjectionV1::Dereference]
                                .into_boxed_slice(),
                        },
                        value: ReferenceValueV1::Use(scalar_operand(23)),
                    },
                ]
                .into_boxed_slice(),
                terminator: ReferenceTerminatorV1::Return,
            }]
            .into_boxed_slice(),
            loop_summaries: Box::default(),
            observable_output_effects: Box::default(),
        };
        let writes = effect_ir.observable_output_writes_v1(meter).unwrap();
        assert_eq!(writes.len(), 2);
        assert_eq!(
            writes
                .iter()
                .map(|write| write.argument)
                .collect::<Vec<_>>(),
            [0, 1],
        );
        assert!(writes.iter().all(|write| matches!(
            write.coordinate,
            ReferenceOutputCoordinateV1::LogicalPoint(_)
        )));
    }

    fn alias_chain_reference_ir(local_count: u32) -> ReferenceEffectIrV1 {
        let assignments = (1..=local_count)
            .map(|local| ReferenceAssignmentV1 {
                statement: local - 1,
                destination: ReferencePlaceV1 {
                    local,
                    projection: Box::default(),
                },
                value: if local == local_count {
                    ReferenceValueV1::Use(scalar_operand(17))
                } else {
                    ReferenceValueV1::Use(ReferenceOperandV1::Copy(ReferencePlaceV1 {
                        local: local + 1,
                        projection: Box::default(),
                    }))
                },
            })
            .collect::<Vec<_>>();
        ReferenceEffectIrV1 {
            argument_count: 0,
            local_count: local_count + 1,
            relations: Box::default(),
            blocks: vec![ReferenceBlockV1 {
                block: 0,
                assignments: assignments.into_boxed_slice(),
                terminator: ReferenceTerminatorV1::Return,
            }]
            .into_boxed_slice(),
            loop_summaries: Box::default(),
            observable_output_effects: Box::default(),
        }
    }

    #[test]
    fn reference_expression_resolution_enforces_128_level_depth_budget() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let boundary = alias_chain_reference_ir(128);
        assert_eq!(
            ReferenceExpressionResolverV1::new(meter, &boundary)
                .unwrap()
                .resolve_local_v1(meter, 1)
                .unwrap(),
            ReferenceEffectExpressionV1::Constant(scalar_constant(17)),
        );
        for depth in [129, 4_096] {
            let overdeep = alias_chain_reference_ir(depth);
            let error = ReferenceExpressionResolverV1::new(meter, &overdeep)
                .unwrap()
                .resolve_local_v1(meter, 1)
                .unwrap_err();
            assert!(error.to_string().contains("exceeds 128 resolution levels"));
        }
    }

    fn local(local: u32) -> ReferenceOperandV1 {
        ReferenceOperandV1::Copy(ReferencePlaceV1 {
            local,
            projection: Box::default(),
        })
    }

    fn checked_field(local: u32, field: u32) -> ReferenceOperandV1 {
        ReferenceOperandV1::Move(ReferencePlaceV1 {
            local,
            projection: vec![ReferencePlaceProjectionV1::Field(field)].into_boxed_slice(),
        })
    }

    fn assign(local: u32, statement: u32, value: ReferenceValueV1) -> ReferenceAssignmentV1 {
        ReferenceAssignmentV1 {
            statement,
            destination: ReferencePlaceV1 {
                local,
                projection: Box::default(),
            },
            value,
        }
    }

    fn counted_loop_reference_ir(upper_bound: ReferenceOperandV1) -> ReferenceEffectIrV1 {
        ReferenceEffectIrV1 {
            argument_count: if matches!(upper_bound, ReferenceOperandV1::Copy(_)) {
                2
            } else {
                1
            },
            local_count: 8,
            relations: if matches!(upper_bound, ReferenceOperandV1::Copy(_)) {
                vec![
                    ReferenceArgumentRelationV1::ScalarInput {
                        argument: 0,
                        scalar: ReferenceScalarTypeV1::U32,
                    },
                    ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                        argument: 1,
                        element: ReferenceScalarTypeV1::U32,
                    },
                ]
            } else {
                vec![ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                    argument: 0,
                    element: ReferenceScalarTypeV1::U32,
                }]
            }
            .into_boxed_slice(),
            blocks: vec![
                ReferenceBlockV1 {
                    block: 0,
                    assignments: vec![
                        assign(3, 0, ReferenceValueV1::Use(scalar_operand(0))),
                        assign(4, 1, ReferenceValueV1::Use(scalar_operand(11))),
                    ]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Goto { target: 1 },
                },
                ReferenceBlockV1 {
                    block: 1,
                    assignments: vec![assign(
                        5,
                        0,
                        ReferenceValueV1::Binary {
                            operation: ReferenceBinaryOpV1::LessThan,
                            lhs: local(3),
                            rhs: upper_bound.clone(),
                            checked: false,
                        },
                    )]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Switch {
                        discriminant: local(5),
                        values: vec![(0, 5)].into_boxed_slice(),
                        otherwise: 2,
                    },
                },
                ReferenceBlockV1 {
                    block: 2,
                    assignments: vec![assign(
                        6,
                        0,
                        ReferenceValueV1::Binary {
                            operation: ReferenceBinaryOpV1::Add,
                            lhs: local(4),
                            rhs: local(3),
                            checked: true,
                        },
                    )]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Assert {
                        condition: checked_field(6, 1),
                        expected: false,
                        success: 3,
                        bounds_check: None,
                    },
                },
                ReferenceBlockV1 {
                    block: 3,
                    assignments: vec![
                        assign(4, 0, ReferenceValueV1::Use(checked_field(6, 0))),
                        assign(
                            7,
                            1,
                            ReferenceValueV1::Binary {
                                operation: ReferenceBinaryOpV1::Add,
                                lhs: local(3),
                                rhs: scalar_operand(1),
                                checked: true,
                            },
                        ),
                    ]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Assert {
                        condition: checked_field(7, 1),
                        expected: false,
                        success: 4,
                        bounds_check: None,
                    },
                },
                ReferenceBlockV1 {
                    block: 4,
                    assignments: vec![assign(3, 0, ReferenceValueV1::Use(checked_field(7, 0)))]
                        .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Goto { target: 1 },
                },
                ReferenceBlockV1 {
                    block: 5,
                    assignments: vec![ReferenceAssignmentV1 {
                        statement: 0,
                        destination: ReferencePlaceV1 {
                            local: if matches!(upper_bound, ReferenceOperandV1::Copy(_)) {
                                2
                            } else {
                                1
                            },
                            projection: vec![ReferencePlaceProjectionV1::Dereference]
                                .into_boxed_slice(),
                        },
                        value: ReferenceValueV1::Use(local(4)),
                    }]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Return,
                },
            ]
            .into_boxed_slice(),
            loop_summaries: Box::default(),
            observable_output_effects: Box::default(),
        }
    }

    #[test]
    fn counted_loop_uses_inherited_work_without_changing_effects() {
        use fe2o3_mir_model::semantic_mir_v1::HARD_MAX_VALIDATION_WORK_V1;
        let ir = counted_loop_reference_ir(scalar_operand(4));
        let derive = |meter: &ReferenceExtractionWorkV1<'_>| {
            let backedges = reference_cfg_backedges_v2(meter, &ir)?;
            validate_reference_loop_shapes_v2(meter, &ir, &backedges)?;
            ir.observable_output_writes_with_loops_v2(meter, &backedges)
        };
        let expected = derive(&ReferenceExtractionWorkV1::Inspection).unwrap();
        let mut measured = SourceClosureWorkV1::default();
        measured.charge(17).unwrap();
        assert_eq!(
            derive(&ReferenceExtractionWorkV1::borrowed(&mut measured)).unwrap(),
            expected
        );
        let cost = measured.validation_work_for_test() - 17;
        assert!(cost > 0 && cost < HARD_MAX_VALIDATION_WORK_V1);
        for remaining in [cost, cost - 1, 0] {
            let mut work = SourceClosureWorkV1::default();
            work.charge((HARD_MAX_VALIDATION_WORK_V1 - remaining) as usize)
                .unwrap();
            let result = derive(&ReferenceExtractionWorkV1::borrowed(&mut work));
            if remaining == cost {
                assert_eq!(result.unwrap(), expected);
                assert_eq!(work.validation_work_for_test(), HARD_MAX_VALIDATION_WORK_V1);
            } else {
                assert!(result.unwrap_err().to_string().contains("ValidationWork"));
            }
        }
    }

    #[test]
    fn exact_counted_loop_derives_loop_carried_value_and_recurrence_identity() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let effect_ir = counted_loop_reference_ir(scalar_operand(4));
        let backedges = reference_cfg_backedges_v2(meter, &effect_ir).unwrap();
        assert_eq!(backedges, BTreeSet::from([(4, 1)]));
        let (writes, summaries) = effect_ir
            .observable_output_writes_with_loops_v2(meter, &backedges)
            .unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(
            writes[0].rhs,
            ReferenceEffectExpressionV1::Constant(scalar_constant(17)),
        );
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].exact_iterations, Some(4));
        assert_eq!(summaries[0].maximum_iterations, 4);
        assert_eq!(summaries[0].carried_locals.as_ref(), &[3, 4]);
        assert_ne!(summaries[0].transition_sha256, [0; 32]);
        assert_ne!(summaries[0].variant_sha256, [0; 32]);
    }

    #[test]
    fn dynamic_loop_with_additional_carried_state_fails_closed() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let effect_ir = counted_loop_reference_ir(local(1));
        let backedges = reference_cfg_backedges_v2(meter, &effect_ir).unwrap();
        let error = effect_ir
            .observable_output_writes_with_loops_v2(meter, &backedges)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("mutates another loop-carried local"),
            "{error}"
        );
    }

    #[test]
    fn dynamic_induction_only_loop_uses_the_unsigned_type_bound() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let effect_ir = ReferenceEffectIrV1 {
            argument_count: 2,
            local_count: 6,
            relations: vec![
                ReferenceArgumentRelationV1::ScalarInput {
                    argument: 0,
                    scalar: ReferenceScalarTypeV1::U32,
                },
                ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                    argument: 1,
                    element: ReferenceScalarTypeV1::U32,
                },
            ]
            .into_boxed_slice(),
            blocks: vec![
                ReferenceBlockV1 {
                    block: 0,
                    assignments: vec![assign(3, 0, ReferenceValueV1::Use(scalar_operand(0)))]
                        .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Goto { target: 1 },
                },
                ReferenceBlockV1 {
                    block: 1,
                    assignments: vec![assign(
                        4,
                        0,
                        ReferenceValueV1::Binary {
                            operation: ReferenceBinaryOpV1::LessThan,
                            lhs: local(3),
                            rhs: local(1),
                            checked: false,
                        },
                    )]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Switch {
                        discriminant: local(4),
                        values: vec![(0, 4)].into_boxed_slice(),
                        otherwise: 2,
                    },
                },
                ReferenceBlockV1 {
                    block: 2,
                    assignments: vec![assign(
                        5,
                        0,
                        ReferenceValueV1::Binary {
                            operation: ReferenceBinaryOpV1::Add,
                            lhs: local(3),
                            rhs: scalar_operand(1),
                            checked: true,
                        },
                    )]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Assert {
                        condition: checked_field(5, 1),
                        expected: false,
                        success: 3,
                        bounds_check: None,
                    },
                },
                ReferenceBlockV1 {
                    block: 3,
                    assignments: vec![assign(3, 0, ReferenceValueV1::Use(checked_field(5, 0)))]
                        .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Goto { target: 1 },
                },
                ReferenceBlockV1 {
                    block: 4,
                    assignments: vec![ReferenceAssignmentV1 {
                        statement: 0,
                        destination: ReferencePlaceV1 {
                            local: 2,
                            projection: vec![ReferencePlaceProjectionV1::Dereference]
                                .into_boxed_slice(),
                        },
                        value: ReferenceValueV1::Use(local(3)),
                    }]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Return,
                },
            ]
            .into_boxed_slice(),
            loop_summaries: Box::default(),
            observable_output_effects: Box::default(),
        };
        let backedges = reference_cfg_backedges_v2(meter, &effect_ir).unwrap();
        let (writes, summaries) = effect_ir
            .observable_output_writes_with_loops_v2(meter, &backedges)
            .unwrap();
        assert_eq!(
            writes[0].rhs,
            ReferenceEffectExpressionV1::KernelScalarArgument { argument: 0 }
        );
        assert_eq!(summaries[0].exact_iterations, None);
        assert_eq!(summaries[0].maximum_iterations, u64::from(u32::MAX));
    }

    #[test]
    fn helper_summary_substitutes_exact_call_arguments() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut effect_ir = alias_chain_reference_ir(2);
        effect_ir.blocks[0].assignments[0].value = ReferenceValueV1::SafeHelperCall {
            helper: ReferenceFunctionIdentityV1 {
                def_path_hash: [1; 16],
                function_sha256: [2; 32],
                item_definition_sha256: [3; 32],
                monomorphization_sha256: [4; 32],
                generic_type_arguments_sha256: [5; 32],
                const_generic_arguments_sha256: [6; 32],
                rustc_mir_body_sha256: [7; 32],
            },
            parameters: vec![ReferenceScalarTypeV1::U32].into_boxed_slice(),
            result: ReferenceScalarTypeV1::U32,
            arguments: vec![scalar_operand(16)].into_boxed_slice(),
            summary: Box::new(ReferenceEffectExpressionV1::Binary {
                operation: ReferenceBinaryOpV1::Add,
                lhs: Box::new(ReferenceEffectExpressionV1::KernelScalarArgument { argument: 0 }),
                rhs: Box::new(ReferenceEffectExpressionV1::Constant(scalar_constant(1))),
                checked: true,
            }),
        };
        assert_eq!(
            ReferenceExpressionResolverV1::new(meter, &effect_ir)
                .unwrap()
                .resolve_local_v1(meter, 1)
                .unwrap(),
            ReferenceEffectExpressionV1::Binary {
                operation: ReferenceBinaryOpV1::Add,
                lhs: Box::new(ReferenceEffectExpressionV1::Constant(scalar_constant(16))),
                rhs: Box::new(ReferenceEffectExpressionV1::Constant(scalar_constant(1))),
                checked: true,
            },
        );
    }

    #[test]
    fn projected_slice_read_retains_its_exact_argument_and_index() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let environment = BTreeMap::from([(
            2,
            ReferenceSymbolicValueV2::Scalar(ReferenceEffectExpressionV1::PointCoordinate {
                axis: 0,
            }),
        )]);
        let value = symbolic_operand_v2(
            meter,
            &environment,
            &ReferenceOperandV1::Copy(ReferencePlaceV1 {
                local: 1,
                projection: vec![
                    ReferencePlaceProjectionV1::Dereference,
                    ReferencePlaceProjectionV1::Index(2),
                ]
                .into_boxed_slice(),
            }),
        )
        .unwrap();
        assert_eq!(
            value,
            ReferenceSymbolicValueV2::Scalar(ReferenceEffectExpressionV1::InputLoad {
                reference_argument: 0,
                index: Box::new(ReferenceEffectExpressionV1::PointCoordinate { axis: 0 }),
            })
        );
    }

    #[test]
    fn unproved_checked_overflow_is_rejected() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut environment = BTreeMap::new();
        environment.insert(
            3,
            ReferenceSymbolicValueV2::CheckedPair {
                value: ReferenceEffectExpressionV1::KernelScalarArgument { argument: 0 },
                overflowed: None,
            },
        );
        let error = symbolic_operand_v2(meter, &environment, &checked_field(3, 1)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("provide an authenticated range fact")
        );
    }

    #[test]
    fn loop_iteration_resource_bound_fails_closed() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut state = ReferenceSymbolicStateV2 {
            block: 4,
            environment: BTreeMap::new(),
            guard: ReferencePathPredicateV1::unconditional_v1(),
            traces: BTreeMap::from([(
                (1, 4),
                ReferenceLoopTraceV2 {
                    header: 1,
                    latch: 4,
                    exit: None,
                    initial: BTreeMap::new(),
                    transitions: vec![BTreeMap::new(); MAX_REFERENCE_LOOP_ITERATIONS_V2],
                    variants: Vec::new(),
                    exact_iterations: None,
                    maximum_iterations: None,
                },
            )]),
        };
        let error = dispatch_symbolic_edge_v2(
            meter,
            &mut VecDeque::new(),
            state.clone(),
            (4, 1),
            &BTreeSet::from([(4, 1)]),
            &BTreeMap::from([((4, 1), BTreeSet::from([1, 4]))]),
            &mut ReferenceSymbolicWorkBudgetV2::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds 4096 iterations"));
        state.traces.clear();
    }

    #[test]
    fn overlapping_or_nested_loop_regions_fail_before_symbolic_execution() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut overlapping = BTreeMap::new();
        overlapping.insert((2, 1), BTreeSet::from([1, 2, 3]));
        overlapping.insert((4, 3), BTreeSet::from([3, 4]));
        let error = reject_overlapping_reference_loops_v2(meter, &overlapping).unwrap_err();
        assert!(error.to_string().contains("overlap or nest"));

        let disjoint = BTreeMap::from([
            ((2, 1), BTreeSet::from([1, 2])),
            ((4, 3), BTreeSet::from([3, 4])),
        ]);
        reject_overlapping_reference_loops_v2(meter, &disjoint).unwrap();
    }

    #[test]
    fn symbolic_unrolling_checks_depth_before_recursive_hashing() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut expression = ReferenceEffectExpressionV1::Constant(ReferenceConstantV1::Scalar {
            scalar: ReferenceScalarTypeV1::U64,
            bits: 1,
        });
        for _ in 0..=fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 {
            expression = ReferenceEffectExpressionV1::Unary {
                operation: ReferenceUnaryOpV1::Not,
                operand: Box::new(expression),
            };
        }
        let error = require_symbolic_expression_budget_v2(meter, &expression).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("symbolic expression exceeds depth")
        );
    }

    #[test]
    fn symbolic_execution_has_one_cumulative_expression_work_budget() {
        let meter = &ReferenceExtractionWorkV1::Inspection;
        let mut budget = ReferenceSymbolicWorkBudgetV2 {
            charged_nodes: MAX_REFERENCE_SYMBOLIC_WORK_NODES_V2,
        };
        let error = budget
            .charge_expression_v2(
                meter,
                &ReferenceEffectExpressionV1::Constant(scalar_constant(1)),
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cumulative expression work nodes")
        );
    }
}
#[test]
fn kernel_scalar_symbols_cannot_enter_the_ranked_load_namespace() {
    assert_eq!(
        kernel_scalar_symbol_v2(0),
        Some(fe2o3_pliron::PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2),
    );
    assert_eq!(
        kernel_scalar_symbol_v2(
            fe2o3_pliron::PRODUCTION_SEMANTIC_LOAD_SYMBOL_BASE_V2
                - fe2o3_pliron::PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2
                - 1,
        ),
        Some(fe2o3_pliron::PRODUCTION_SEMANTIC_LOAD_SYMBOL_BASE_V2 - 1),
    );
    assert_eq!(
        kernel_scalar_symbol_v2(
            fe2o3_pliron::PRODUCTION_SEMANTIC_LOAD_SYMBOL_BASE_V2
                - fe2o3_pliron::PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2,
        ),
        None,
    );
    assert_eq!(kernel_scalar_symbol_v2(u32::MAX), None);
}

#[path = "reference_effect_retained_storage_v1.rs"]
mod retained_storage_v1;

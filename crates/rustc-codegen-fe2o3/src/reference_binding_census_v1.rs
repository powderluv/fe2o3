//! Bounded admission of complete reference-binding equality, not source authority.

use super::*;
use crate::rustc_semantic_plan_v1::SourceClosureWorkV1;
use std::mem::size_of;

const EXPRESSION_STACK_V1: usize = fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 + 1;
type ExpressionFrameV1<'a> = Option<(&'a ReferenceEffectExpressionV1, usize)>;
type CensusResultV1<T> = Result<T, ReferenceBindingErrorV1>;

fn charge(work: &mut SourceClosureWorkV1, amount: usize) -> CensusResultV1<()> {
    work.charge(amount)
        .map_err(|error| ReferenceBindingErrorV1::new(error.to_string()))
}

/// Censuses both complete operands before admitting the existing derived `Eq`.
///
/// Each visited record and flat payload is charged by its in-memory byte size
/// (at least one unit), then the sum for BOTH operands is charged again before
/// equality. Inline fields may be counted twice; this is conservative logical
/// work, not machine instructions or a live/peak-storage receipt. The sole
/// scratch array is fixed-size, initialized once and reused for every expression.
/// No graph, signature array, string or expression is cloned or allocated here;
/// refusals use the existing diagnostic-string error type.
///
/// Existing expression depth/node limits apply independently to every tree.
/// All other traversal accumulates on the supplied source-work meter, including
/// refusals. Equal hashes, pointer equality and early field mismatches do not
/// bypass either census. Extraction and already-owned input storage remain the
/// caller's responsibility; successful equality does not authenticate source.
pub(crate) fn equivalent_bindings_v1(
    old: &AuthenticatedReferenceEffectBindingV1,
    fresh: &AuthenticatedReferenceEffectBindingV1,
    work: &mut SourceClosureWorkV1,
) -> Result<bool, ReferenceBindingErrorV1> {
    charge(
        work,
        size_of::<[ExpressionFrameV1<'_>; EXPRESSION_STACK_V1]>(),
    )?;
    let mut census = BindingCensusV1 {
        work,
        comparison_work: 0,
        expressions: [None; EXPRESSION_STACK_V1],
    };
    census.binding(old)?;
    census.binding(fresh)?;
    charge(census.work, census.comparison_work)?;
    Ok(old == fresh)
}

struct BindingCensusV1<'binding, 'work> {
    work: &'work mut SourceClosureWorkV1,
    comparison_work: usize,
    expressions: [ExpressionFrameV1<'binding>; EXPRESSION_STACK_V1],
}

impl<'binding> BindingCensusV1<'binding, '_> {
    fn bytes(&mut self, bytes: usize) -> CensusResultV1<()> {
        let amount = bytes.max(1);
        charge(self.work, amount)?;
        self.comparison_work = self.comparison_work.checked_add(amount).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference binding comparison work overflow")
        })?;
        Ok(())
    }

    fn record<T>(&mut self) -> CensusResultV1<()> {
        self.bytes(size_of::<T>())
    }

    // Only pointer-free records use this shortcut; nested owners are walked below.
    fn flat<T>(&mut self, values: &[T]) -> CensusResultV1<()> {
        let bytes = values.len().checked_mul(size_of::<T>()).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference binding payload size overflow")
        })?;
        self.bytes(bytes)
    }

    fn binding(
        &mut self,
        binding: &'binding AuthenticatedReferenceEffectBindingV1,
    ) -> CensusResultV1<()> {
        self.record::<AuthenticatedReferenceEffectBindingV1>()?;
        let AuthenticatedReferenceEffectBindingV1 {
            origin,
            logical_kernel_name,
            kernel: _,
            reference: _,
            signature_preimage,
            effect_ir_sha256: _,
            effect_ir,
            observable_output_writes,
        } = binding;
        self.bytes(origin.retained_payload_bytes_v1())?;
        self.bytes(logical_kernel_name.len())?;
        self.record::<ReferenceLogicalSignaturePreimageV1>()?;
        self.flat(signature_preimage.kernel_inputs())?;
        self.flat(signature_preimage.reference_inputs())?;
        self.effect_ir(effect_ir)?;
        self.writes(observable_output_writes)
    }

    fn effect_ir(&mut self, ir: &'binding ReferenceEffectIrV1) -> CensusResultV1<()> {
        self.record::<ReferenceEffectIrV1>()?;
        let ReferenceEffectIrV1 {
            argument_count: _,
            local_count: _,
            relations,
            blocks,
            loop_summaries,
            observable_output_effects,
        } = ir;
        self.flat(relations)?;
        for block in blocks {
            self.record::<ReferenceBlockV1>()?;
            let ReferenceBlockV1 {
                block: _,
                assignments,
                terminator,
            } = block;
            for assignment in assignments {
                self.record::<ReferenceAssignmentV1>()?;
                let ReferenceAssignmentV1 {
                    statement: _,
                    destination,
                    value,
                } = assignment;
                self.place(destination)?;
                self.value(value)?;
            }
            self.terminator(terminator)?;
        }
        for summary in loop_summaries {
            self.record::<ReferenceLoopSummaryV2>()?;
            let ReferenceLoopSummaryV2 {
                header: _,
                latch: _,
                exit: _,
                exact_iterations: _,
                maximum_iterations: _,
                carried_locals,
                initial_state_sha256: _,
                transition_sha256: _,
                variant_sha256: _,
            } = summary;
            self.flat(carried_locals)?;
        }
        self.writes(observable_output_effects)
    }

    fn place(&mut self, place: &ReferencePlaceV1) -> CensusResultV1<()> {
        self.record::<ReferencePlaceV1>()?;
        let ReferencePlaceV1 {
            local: _,
            projection,
        } = place;
        self.flat(projection)
    }

    fn operand(&mut self, operand: &ReferenceOperandV1) -> CensusResultV1<()> {
        self.record::<ReferenceOperandV1>()?;
        match operand {
            ReferenceOperandV1::Copy(place) | ReferenceOperandV1::Move(place) => self.place(place),
            ReferenceOperandV1::Constant(_) => Ok(()),
        }
    }

    fn value(&mut self, value: &'binding ReferenceValueV1) -> CensusResultV1<()> {
        self.record::<ReferenceValueV1>()?;
        match value {
            ReferenceValueV1::Use(operand)
            | ReferenceValueV1::Unary {
                operation: _,
                operand,
            }
            | ReferenceValueV1::Cast {
                kind: _,
                source: _,
                target: _,
                operand,
            } => self.operand(operand),
            ReferenceValueV1::Binary {
                operation: _,
                lhs,
                rhs,
                checked: _,
            } => {
                self.operand(lhs)?;
                self.operand(rhs)
            }
            ReferenceValueV1::InputLength {
                reference_argument: _,
            } => Ok(()),
            ReferenceValueV1::SafeHelperCall {
                helper: _,
                parameters,
                result: _,
                arguments,
                summary,
            } => {
                self.flat(parameters)?;
                for argument in arguments {
                    self.operand(argument)?;
                }
                self.expression(summary)
            }
        }
    }

    fn terminator(&mut self, term: &ReferenceTerminatorV1) -> CensusResultV1<()> {
        self.record::<ReferenceTerminatorV1>()?;
        match term {
            ReferenceTerminatorV1::Return | ReferenceTerminatorV1::Goto { target: _ } => Ok(()),
            ReferenceTerminatorV1::Switch {
                discriminant,
                values,
                otherwise: _,
            } => {
                self.operand(discriminant)?;
                self.flat(values)
            }
            ReferenceTerminatorV1::Assert {
                condition,
                expected: _,
                success: _,
                bounds_check,
            } => {
                self.operand(condition)?;
                if let Some(bounds) = bounds_check {
                    self.record::<ReferenceBoundsCheckV1>()?;
                    let ReferenceBoundsCheckV1 { index, length } = bounds;
                    self.operand(index)?;
                    self.operand(length)?;
                }
                Ok(())
            }
        }
    }

    fn writes(&mut self, writes: &'binding [ReferenceOutputWriteV1]) -> CensusResultV1<()> {
        for write in writes {
            self.record::<ReferenceOutputWriteV1>()?;
            let ReferenceOutputWriteV1 {
                argument: _,
                block: _,
                statement: _,
                coordinate,
                guard,
                rhs,
                value,
            } = write;
            self.record::<ReferenceOutputCoordinateV1>()?;
            match coordinate {
                ReferenceOutputCoordinateV1::LogicalPoint(expressions) => {
                    for expression in expressions {
                        self.expression(expression)?;
                    }
                }
                ReferenceOutputCoordinateV1::Dynamic(expression) => self.expression(expression)?,
                ReferenceOutputCoordinateV1::SingleCoordinate
                | ReferenceOutputCoordinateV1::Constant {
                    offset: _,
                    minimum_length: _,
                    from_end: _,
                } => {}
            }
            self.predicate(guard)?;
            self.expression(rhs)?;
            self.value(value)?;
        }
        Ok(())
    }

    fn predicate(&mut self, predicate: &'binding ReferencePathPredicateV1) -> CensusResultV1<()> {
        self.record::<ReferencePathPredicateV1>()?;
        let ReferencePathPredicateV1 { clauses } = predicate;
        for clause in clauses {
            self.record::<ReferenceGuardClauseV1>()?;
            let ReferenceGuardClauseV1 { atoms } = clause;
            for atom in atoms {
                self.record::<ReferenceGuardAtomV1>()?;
                match atom {
                    ReferenceGuardAtomV1::SwitchValueSet {
                        discriminant,
                        values,
                        inside_set: _,
                    } => {
                        self.flat(values)?;
                        self.expression(discriminant)?;
                    }
                    ReferenceGuardAtomV1::Assert {
                        condition,
                        expected: _,
                    } => self.expression(condition)?,
                }
            }
        }
        Ok(())
    }

    fn expression(&mut self, root: &'binding ReferenceEffectExpressionV1) -> CensusResultV1<()> {
        let mut length = 0;
        let mut nodes = 0_usize;
        self.push_expression(&mut length, root, 0)?;
        while length != 0 {
            self.record::<ReferenceEffectExpressionV1>()?;
            length -= 1;
            let (expression, depth) = self.expressions[length].take().ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference binding expression stack is empty")
            })?;
            nodes = nodes.checked_add(1).ok_or_else(|| {
                ReferenceBindingErrorV1::new("reference expression work overflowed")
            })?;
            if nodes > MAX_REFERENCE_EXPRESSION_NODES_V1 {
                return Err(ReferenceBindingErrorV1::new(format!(
                    "reference effect expression exceeds {MAX_REFERENCE_EXPRESSION_NODES_V1} nodes",
                )));
            }
            match expression {
                ReferenceEffectExpressionV1::Binary {
                    operation: _,
                    lhs,
                    rhs,
                    checked: _,
                } => {
                    self.push_expression(&mut length, rhs, depth + 1)?;
                    self.push_expression(&mut length, lhs, depth + 1)?;
                }
                ReferenceEffectExpressionV1::Unary {
                    operation: _,
                    operand,
                }
                | ReferenceEffectExpressionV1::Cast {
                    kind: _,
                    source: _,
                    target: _,
                    operand,
                }
                | ReferenceEffectExpressionV1::InputLoad {
                    reference_argument: _,
                    index: operand,
                } => {
                    self.push_expression(&mut length, operand, depth + 1)?;
                }
                ReferenceEffectExpressionV1::PointCoordinate { axis: _ }
                | ReferenceEffectExpressionV1::KernelScalarArgument { argument: _ }
                | ReferenceEffectExpressionV1::InputLength {
                    reference_argument: _,
                }
                | ReferenceEffectExpressionV1::Constant(_) => {}
            }
        }
        Ok(())
    }

    fn push_expression(
        &mut self,
        length: &mut usize,
        expression: &'binding ReferenceEffectExpressionV1,
        depth: usize,
    ) -> CensusResultV1<()> {
        ReferenceExpressionResolverV1::require_depth_v1(depth)?;
        let slot = self.expressions.get_mut(*length).ok_or_else(|| {
            ReferenceBindingErrorV1::new("reference binding expression stack capacity exceeded")
        })?;
        *slot = Some((expression, depth));
        *length += 1;
        Ok(())
    }
}

#[cfg(test)]
#[path = "reference_binding_census_v1_tests.rs"]
mod tests;

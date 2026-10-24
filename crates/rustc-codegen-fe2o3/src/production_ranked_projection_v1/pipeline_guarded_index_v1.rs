// Preserve a source LessThan guard that the uniform-only switch projector cannot
// represent. This is an association of an existing true edge with an exact
// unsigned value, not an inferred range or a newly inserted runtime check.
fn project_pipeline_guarded_index_v1(
    projector: &mut PipelineScalarProjectorV1<'_, '_, '_, '_, '_>,
    use_block: usize,
    operand: &SemanticOperandV1,
    effects: &mut [Option<ProjectedPipelineEffectV1>],
) -> Result<ProjectedPipelineScalarV1, ProductionRankedProjectionErrorV1> {
    let use_site = ScalarAssignmentSiteV1 {
        block: use_block,
        statement: projector.function.blocks()[use_block].statements().len(),
    };
    let bounds = projector
        .assertion_proofs
        .authenticated_strict_unsigned_bounds_v1(operand, use_site, use_site)?;
    for (_, site) in bounds {
        let block = &projector.function.blocks()[site.block];
        let SemanticStatementKindV1::Assign(assignment) = block.statements()[site.statement].kind()
        else {
            continue;
        };
        let SemanticRvalueKindV1::Binary {
            operation: SemanticBinaryOpV1::LessThan,
            left,
            right,
        } = assignment.value().kind()
        else {
            continue;
        };
        let local = assignment.destination().local().index() as usize;
        let SemanticTerminatorKindV1::SwitchInt {
            discriminant,
            targets,
        } = block.terminator().kind()
        else {
            continue;
        };
        // The demonstrated contract is one direct true edge into one read/write
        // block. Do not manufacture transport through unrelated paths or loops.
        if !assignment.destination().projections().is_empty()
            || simple_operand_local(discriminant).map(|v| v.index() as usize) != Some(local)
            || projector
                .assertion_proofs
                .definition_counts
                .get(local)
                .copied()
                != Some(1)
            || projector
                .assertion_proofs
                .address_escaped
                .get(local)
                .copied()
                != Some(false)
            || targets.otherwise().target().index() as usize != use_block
            || projector.assertion_proofs.graph.predecessors[use_block].as_slice() != [site.block]
            || !matches!(
                projector
                    .types
                    .get(discriminant.ty().index() as usize)
                    .map(SemanticTypeDeclV1::shape),
                Some(SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool))
            )
            || !projector
                .assertion_proofs
                .comparison_edge_authenticates_each_dynamic_use_v1(
                    site.block, use_block, use_block,
                )?
            || effects[site.block].is_some()
        {
            continue;
        }
        // The source SSA capture records operands at the comparison statement,
        // not at this block's terminator (which uses only the boolean result).
        projector.work = 0;
        let index = projector.project_operand(left, site, &mut HashSet::new())?;
        if !matches!(index, ProjectedPipelineScalarV1::Binary { .. }) {
            continue;
        }
        projector.work = 0;
        let ProjectedPipelineScalarV1::Value(bound) =
            projector.project_operand(right, site, &mut HashSet::new())?
        else {
            continue;
        };
        effects[site.block] = Some(ProjectedPipelineEffectV1::Guard {
            index,
            bound,
            access_block: use_block,
        });
        return Ok(ProjectedPipelineScalarV1::GuardedIndex);
    }
    projector.project_root(use_block, operand)
}

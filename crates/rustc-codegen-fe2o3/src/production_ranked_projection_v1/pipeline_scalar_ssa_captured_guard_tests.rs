#[test]
fn actual_use_ssa_pipeline_guard_requires_the_captured_comparison_statement() {
    let boolean = SemanticTypeIdV1::from_index(3);
    let blocks = vec![
        block(
            60,
            vec![typed_assignment(
                2,
                WORD,
                SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::Remainder,
                    left: copy(1),
                    right: literal(512),
                },
            )],
            goto(1),
        ),
        block(
            61,
            vec![typed_assignment(
                3,
                boolean,
                SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::LessThan,
                    left: copy(2),
                    right: literal(256),
                },
            )],
            zero_switch(3, boolean, 3, 2),
        ),
        // A captured operand use at this direct successor's terminator. The
        // pipeline caller separately supplies the actual read intrinsic.
        block(62, vec![], zero_switch(2, WORD, 3, 3)),
        block(63, vec![], SemanticTerminatorKindV1::Return),
    ];
    let mut owner = make_uncaptured_owner_with_guard(blocks, true).unwrap();
    let mut capture_work = Work::new(10_000_000);
    let mut capture_budget = Budget::new(&mut capture_work, 10_000_000);
    owner
        .try_capture_occurrences_with_budget_v1(&mut capture_budget)
        .unwrap();
    assert_eq!(capture_budget.storage(), 0);
    let function = function(&owner);
    let types = owner.source_semantic().types();
    let mut work = Work::new(1_000_000);
    let mut budget = Budget::new(&mut work, 1_000_000);
    with_index(
        source(&owner),
        function,
        &mut Facts(&mut budget),
        |index, facts| {
            let indices = vec![None; function.locals().len()];
            let mut arguments = vec![None; function.locals().len()];
            let mut next_argument = 0;
            let mut operations = Vec::new();
            let mut next_value = 0;
            let mut effects = vec![None; function.blocks().len()];
            let mut projector = PipelineScalarProjectorV1 {
                types,
                function,
                index_values: &indices,
                runtime_index_arguments: &mut arguments,
                next_runtime_argument: &mut next_argument,
                uniform_inductions: &[],
                entry_operations: &mut operations,
                next_value: &mut next_value,
                assertion_proofs: SemanticAssertProofsV1::new(types, function)?,
                scalar_ssa: Some((index, facts)),
                work: 0,
            };
            assert!(matches!(
                project_pipeline_guarded_index_v1(&mut projector, 2, &copy(2), &mut effects)?,
                ProjectedPipelineScalarV1::GuardedIndex
            ));
            assert!(matches!(
                effects[1],
                Some(ProjectedPipelineEffectV1::Guard {
                    access_block: 2,
                    ..
                })
            ));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(budget.storage(), 0);
    let wrong_site = with_index(
        source(&owner),
        function,
        &mut Facts(&mut budget),
        |index, facts| project(index, facts, &owner, 2, site(1, 1)).map(|_| ()),
    );
    incomplete(
        wrong_site,
        "a pipeline scalar has no captured use at its exact source site",
    );
    assert_eq!(budget.storage(), 0);
    owner.verify_replay().unwrap();
}

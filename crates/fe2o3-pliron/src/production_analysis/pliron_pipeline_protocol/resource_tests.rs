#[cfg(test)]
mod resource_upper_bound_tests {
    use super::*;
    use crate::production_analysis::pliron_ir_identity::LivePlironStructuralIdentityProviderV1;
    use crate::production_analysis::pliron_pass_contract::begin_production_pliron_pass_contract_session_v1;
    use dialect_kernel::{
        BranchArgsOp, DIALECT_NAME, IndexLessThanBranchArgsOp, IndexType, PipelineEventKindAttr,
        PipelineType, RankedViewType, ReturnOp, SemanticSymbolOp, register_dialect,
    };
    use pliron::{
        builtin::{op_interfaces::OneRegionInterface, types::FunctionType},
        dialect::DialectName,
        op::Op,
        r#type::TypeHandle,
    };

    fn append_test_op_v1<O: Op>(context: &Context, block: Ptr<BasicBlock>, operation: &O) {
        operation.get_operation().insert_at_back(block, context);
    }

    fn test_index_block_v1(
        context: &mut Context,
        function: &FuncOp,
        name: &str,
        arguments: usize,
    ) -> (Ptr<BasicBlock>, Vec<Value>) {
        let index: TypeHandle = IndexType::get(context).into();
        let block = BasicBlock::new(
            context,
            Some(name.try_into().unwrap()),
            vec![index; arguments],
        );
        let values = block.deref(context).arguments().collect();
        block.insert_at_back(function.get_region(context), context);
        (block, values)
    }

    #[test]
    fn pipeline_bound_has_exact_and_one_under_admission() {
        let census = ProductionAnalysisInputCensusV1 {
            blocks: 7,
            operations: 29,
            operands: 174,
            results: 40,
            successors: 9,
            block_arguments: 13,
            identifier_bytes: 80,
            max_operation_arity: 6,
            pipeline_creates: 3,
            pipeline_events: 11,
            ranked_accesses: 7,
            index_lt_branch_candidates: 2,
            ..ProductionAnalysisInputCensusV1::default()
        };
        let exact = preflight_pipeline_protocol_resource_upper_bound_v1(
            census,
            ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
        )
        .unwrap();
        // Q=5975 includes 2436 propagation, 696 all-edge invariant and
        // 882 optional membership queries; U=(40+13)^2=2809.
        // U = (40+13)^2 = 2,809; equivalence work is 2Q+8U+1.
        // Concrete CFG reserve: 32*(7+9+1) + 3*32*(7+9+29+19+1) = 6,784.
        // Concrete fact queries: 3*(1+(2*11+7)*(8+4))=1047.
        assert_eq!(exact.work_upper_bound(), 55_302);
        assert_eq!(exact.retained_storage_upper_bound(), 33_611);
        // One-live-creation scratch delta: 17*7+9+29+2*19+48 = 243.
        assert_eq!(exact.peak_storage_upper_bound(), 48_662);
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                census,
                ProductionAnalysisResourceLimitsV1::new(
                    exact.work_upper_bound() - 1,
                    exact.peak_storage_upper_bound(),
                ),
            ),
            Err(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                resource: "work upper bound",
            })
        );
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                census,
                ProductionAnalysisResourceLimitsV1::new(
                    exact.work_upper_bound(),
                    exact.peak_storage_upper_bound() - 1,
                ),
            ),
            Err(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                resource: "peak storage upper bound",
            })
        );
    }

    #[test]
    fn pipeline_bound_rejects_overflow_before_analysis() {
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                ProductionAnalysisInputCensusV1 {
                    operations: usize::MAX,
                    ..ProductionAnalysisInputCensusV1::default()
                },
                ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
            ),
            Err(pipeline_resource_overflow_v1())
        );
    }

    #[test]
    fn duplicated_depth_sixty_four_uniformity_dag_is_memoized_at_the_exact_visit_limit() {
        const DEPTH: usize = 64;
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let constant = IndexConstantOp::new(&mut context, 1);
        let mut value = constant.result(&context);
        for _ in 0..DEPTH {
            let binary = IndexBinaryOp::new(&mut context, IndexBinaryKindAttr::Add, value, value);
            value = binary.result(&context);
        }
        let uniform_roots = HashSet::new();
        assert_eq!(
            is_uniform_value(&context, value, &uniform_roots, DEPTH + 1),
            Ok(true)
        );
        assert_eq!(
            is_uniform_value(&context, value, &uniform_roots, DEPTH),
            Err(UniformityVisitLimitV1)
        );
    }

    #[test]
    fn deep_cast_binary_pair_has_exact_and_one_under_equivalence_admission() {
        const DEPTH: usize = 85;
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let left_root = IndexConstantOp::new(&mut context, 1);
        let right_root = IndexConstantOp::new(&mut context, 1);
        let mut left = left_root.result(&context);
        let mut right = right_root.result(&context);
        for level in 0..DEPTH {
            let left_rhs = IndexConstantOp::new(&mut context, level as u64 + 2);
            let right_rhs = IndexConstantOp::new(&mut context, level as u64 + 2);
            let left_rhs = left_rhs.result(&context);
            let right_rhs = right_rhs.result(&context);
            let left_binary =
                IndexBinaryOp::new(&mut context, IndexBinaryKindAttr::Remainder, left, left_rhs);
            let right_binary = IndexBinaryOp::new(
                &mut context,
                IndexBinaryKindAttr::Remainder,
                right,
                right_rhs,
            );
            let left_binary = left_binary.result(&context);
            let right_binary = right_binary.result(&context);
            let left_cast = IndexUnsignedCastOp::new(&mut context, left_binary, 32);
            let right_cast = IndexUnsignedCastOp::new(&mut context, right_binary, 32);
            left = left_cast.result(&context);
            right = right_cast.result(&context);
        }

        // Each layer visits cast, binary, and its RHS constant pair; the two
        // roots contribute the final visit. Opaque joins cannot prove equality.
        assert_eq!(DEPTH * 3 + 1, MAX_EQUIVALENCE_WORK_V1);
        let mut exact_resources =
            EquivalenceResourceMeterV1::new(2, MAX_EQUIVALENCE_WORK_V1).unwrap();
        assert_eq!(
            evaluate_index_equivalence_v1(
                &context,
                left,
                right,
                MAX_EQUIVALENCE_WORK_V1,
                &mut exact_resources,
            ),
            Ok(true)
        );
        assert!(!exact_resources.exhausted());
        let expanded_pairs = exact_resources.expanded_pairs;
        let cursor_steps = exact_resources.cursor_steps;
        assert_eq!(
            evaluate_index_equivalence_v1(
                &context,
                left,
                right,
                MAX_EQUIVALENCE_WORK_V1,
                &mut exact_resources,
            ),
            Ok(true)
        );
        assert_eq!(exact_resources.expanded_pairs, expanded_pairs);
        assert_eq!(exact_resources.cursor_steps, cursor_steps + 1);

        let mut one_under_resources =
            EquivalenceResourceMeterV1::new(1, MAX_EQUIVALENCE_WORK_V1).unwrap();
        assert_eq!(
            evaluate_index_equivalence_v1(
                &context,
                left,
                right,
                MAX_EQUIVALENCE_WORK_V1 - 1,
                &mut one_under_resources,
            ),
            Err(EquivalenceVisitLimitV1)
        );
        assert!(one_under_resources.exhausted());
    }

    #[test]
    fn wide_bidirectional_coordinates_and_loop_operands_are_charged() {
        const ARITY: usize = 32;
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let value = IndexConstantOp::new(&mut context, 7).result(&context);
        let left = vec![vec![value; ARITY]];
        let right = left.clone();
        let mut exact_resources = EquivalenceResourceMeterV1::new(ARITY * 2, 0).unwrap();
        assert!(coordinate_sets_equivalent_v1(
            &context,
            &left,
            &right,
            &mut exact_resources,
        ));
        assert!(!exact_resources.exhausted());
        let mut one_under_resources = EquivalenceResourceMeterV1::new(ARITY * 2 - 1, 0).unwrap();
        assert!(!coordinate_sets_equivalent_v1(
            &context,
            &left,
            &right,
            &mut one_under_resources,
        ));
        assert!(one_under_resources.exhausted());

        let census = ProductionAnalysisInputCensusV1 {
            blocks: 11,
            operations: 13,
            operands: 143,
            max_operation_arity: ARITY,
            pipeline_creates: 3,
            pipeline_events: 4,
            ranked_accesses: 5,
            index_lt_branch_candidates: 2,
            ..ProductionAnalysisInputCensusV1::default()
        };
        let expected_coordinates = 2 * 3 * 5_usize.pow(2) * ARITY;
        let expected_loop_edges = 2 * 11 * 143;
        assert_eq!(
            pipeline_equivalence_query_upper_bound_v1(census).unwrap(),
            expected_coordinates
                + expected_coordinates / 2
                + expected_loop_edges
                + 2 * 2 * 143
                + 3 * 2
                + 4 * 4
                + 7 * 3 * 5,
        );
        let overflowed_value_space = ProductionAnalysisInputCensusV1 {
            operations: 2,
            results: usize::MAX,
            block_arguments: 1,
            pipeline_creates: 1,
            pipeline_events: 1,
            ..ProductionAnalysisInputCensusV1::default()
        };
        assert_eq!(
            pipeline_equivalence_unique_pair_upper_bound_v1(overflowed_value_space).unwrap(),
            4 * MAX_EQUIVALENCE_WORK_V1,
        );
    }

    #[test]
    fn wide_entry_arguments_are_present_in_real_uniform_root_accounting() {
        const ARGUMENTS: usize = 128;
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let index: TypeHandle = IndexType::get(&context).into();
        let signature = FunctionType::get(&context, vec![index; ARGUMENTS], vec![]);
        let function = FuncOp::new(
            &mut context,
            "wide_uniform_roots".try_into().unwrap(),
            signature,
        );
        let entry = function.get_entry_block(&context);
        assert_eq!(entry.deref(&context).get_num_arguments(), ARGUMENTS);
        let view_type = RankedViewType::new(&context, 16, true, vec![2, 64]).unwrap();
        let view = RankedViewOp::new_in_space(
            &mut context,
            view_type,
            vec![],
            dialect_kernel::MemorySpaceAttr::Workgroup,
        )
        .unwrap();
        let view_result = view.result(&context);
        let create = PipelineCreateOp::new(&mut context, view_result, 2, 1).unwrap();
        let return_op = ReturnOp::new(&mut context);
        append_test_op_v1(&context, entry, &view);
        append_test_op_v1(&context, entry, &create);
        append_test_op_v1(&context, entry, &return_op);

        let preservation = begin_production_pliron_pass_contract_session_v1(
            LivePlironStructuralIdentityProviderV1::new(&context, &function),
        )
        .unwrap();
        let census = preservation.input_census_v1();
        assert_eq!(census.block_arguments, ARGUMENTS);
        assert_eq!(census.pipeline_creates, 1);
        assert_eq!(census.pipeline_events, 0);
        assert_eq!(census.ranked_accesses, 0);
        assert_eq!(census.index_lt_branch_candidates, 0);
        let without_arguments = ProductionAnalysisInputCensusV1 {
            block_arguments: 0,
            ..census
        };
        let exact = preflight_pipeline_protocol_resource_upper_bound_v1(
            census,
            ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
        )
        .unwrap();
        let baseline = preflight_pipeline_protocol_resource_upper_bound_v1(
            without_arguments,
            ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
        )
        .unwrap();
        assert_eq!(
            exact.work_upper_bound() - baseline.work_upper_bound(),
            ARGUMENTS * 2
        );
        assert_eq!(
            exact.peak_storage_upper_bound() - baseline.peak_storage_upper_bound(),
            ARGUMENTS
        );
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                census,
                ProductionAnalysisResourceLimitsV1::new(
                    exact.work_upper_bound() - 1,
                    exact.peak_storage_upper_bound(),
                ),
            ),
            Err(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                resource: "work upper bound",
            })
        );
        assert_eq!(
            run_pliron_pipeline_protocol_check_v1(&context, &function).status(),
            KernelCheckStatusV1::Rejected
        );
    }

    #[test]
    fn large_pipeline_free_function_takes_linear_fast_path() {
        const CONSTANTS: usize = 1_100;
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let signature = FunctionType::get(&context, vec![], vec![]);
        let function = FuncOp::new(
            &mut context,
            "large_pipeline_free".try_into().unwrap(),
            signature,
        );
        let entry = function.get_entry_block(&context);
        for value in 0..CONSTANTS {
            let constant = IndexConstantOp::new(&mut context, value as u64);
            append_test_op_v1(&context, entry, &constant);
        }
        let return_op = ReturnOp::new(&mut context);
        append_test_op_v1(&context, entry, &return_op);
        let preservation = begin_production_pliron_pass_contract_session_v1(
            LivePlironStructuralIdentityProviderV1::new(&context, &function),
        )
        .unwrap();
        let census = preservation.input_census_v1();
        assert_eq!(census.pipeline_creates, 0);
        assert_eq!(census.operations, CONSTANTS + 1);
        let exact = preflight_pipeline_protocol_resource_upper_bound_v1(
            census,
            ProductionAnalysisResourceLimitsV1::production_hard_ceiling(),
        )
        .unwrap();
        assert_eq!(exact.work_upper_bound(), census.operations * 4 + 1);
        assert_eq!(exact.peak_storage_upper_bound(), 0);
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                census,
                ProductionAnalysisResourceLimitsV1::new(
                    exact.work_upper_bound() - 1,
                    exact.peak_storage_upper_bound(),
                ),
            ),
            Err(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                resource: "work upper bound",
            })
        );
        assert!(run_pliron_pipeline_protocol_check_v1(&context, &function).is_clean());
    }

    #[test]
    fn pipeline_free_fast_path_retains_orphans_and_uses_the_finding_limit_marker() {
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let pipeline: TypeHandle = PipelineType::new(&context, 2, 1).unwrap().into();
        let index: TypeHandle = IndexType::get(&context).into();
        let signature = FunctionType::get(&context, vec![pipeline, index, index], vec![]);
        let function = FuncOp::new(
            &mut context,
            "orphan_pipeline_events".try_into().unwrap(),
            signature,
        );
        let entry = function.get_entry_block(&context);
        let arguments = entry.deref(&context).arguments().collect::<Vec<_>>();
        let event = PipelineEventOp::new(
            &mut context,
            arguments[0],
            arguments[1],
            arguments[2],
            PipelineEventKindAttr::Stage,
        )
        .unwrap();
        append_test_op_v1(&context, entry, &event);
        let return_op = ReturnOp::new(&mut context);
        append_test_op_v1(&context, entry, &return_op);

        let preservation = begin_production_pliron_pass_contract_session_v1(
            LivePlironStructuralIdentityProviderV1::new(&context, &function),
        )
        .unwrap();
        let census = preservation.input_census_v1();
        assert_eq!(census.pipeline_creates, 0);
        assert_eq!(census.pipeline_events, 1);
        let exact = preflight_pipeline_protocol_resource_upper_bound_v1(
            census,
            ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
        )
        .unwrap();
        assert_eq!(
            exact.retained_storage_upper_bound(),
            MAX_PLIRON_PIPELINE_DIAGNOSTIC_BYTES_V1 + 32
        );
        let report = run_pliron_pipeline_protocol_check_v1(&context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
        assert!(matches!(
            report.findings(),
            [PlironPipelineProtocolFindingV1::OrphanEvent { .. }]
        ));

        let mut bounded_findings = Vec::new();
        for operation in 0..=MAX_PLIRON_PIPELINE_FINDINGS_V1 {
            push_finding(
                &mut bounded_findings,
                PlironPipelineProtocolFindingV1::OrphanEvent {
                    block: 0,
                    operation,
                },
            );
        }
        assert_eq!(bounded_findings.len(), MAX_PLIRON_PIPELINE_FINDINGS_V1 + 1);
        assert!(matches!(
            bounded_findings.last(),
            Some(PlironPipelineProtocolFindingV1::FindingLimitExceeded)
        ));
    }

    #[test]
    fn authenticated_protocol_census_counts_each_operation_class_exactly() {
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        dialect_proof::register_dialect(&mut context).unwrap();
        let signature = FunctionType::get(&context, vec![], vec![]);
        let function = FuncOp::new(
            &mut context,
            "protocol_census".try_into().unwrap(),
            signature,
        );
        let entry = function.get_entry_block(&context);
        let view_type = RankedViewType::new(&context, 16, true, vec![2, 64]).unwrap();
        let view = RankedViewOp::new_in_space(
            &mut context,
            view_type,
            vec![],
            dialect_kernel::MemorySpaceAttr::Workgroup,
        )
        .unwrap();
        let view_result = view.result(&context);
        let create = PipelineCreateOp::new(&mut context, view_result, 2, 1).unwrap();
        let pipeline = create.pipeline(&context);
        let zero = IndexConstantOp::new(&mut context, 0);
        let zero_value = zero.result(&context);
        let event = PipelineEventOp::new(
            &mut context,
            pipeline,
            zero_value,
            zero_value,
            PipelineEventKindAttr::Stage,
        )
        .unwrap();
        let access = RankedAccessOp::new(
            &mut context,
            AccessKindAttr::Write,
            view_result,
            vec![zero_value, zero_value],
        )
        .unwrap();
        let semantic = SemanticSymbolOp::new(&mut context, 0);
        let semantic_value = semantic.result(&context);
        let effect_contract = RequireEffectRefinementOp::new(
            &mut context,
            dialect_proof::ProofIdAttr::new([1, 2, 3, 4]),
            view_result,
            vec![zero_value, zero_value],
            vec![semantic_value, semantic_value],
            vec![semantic_value, semantic_value],
            semantic_value,
            semantic_value,
            semantic_value,
            semantic_value,
            semantic_value,
            semantic_value,
        );
        let return_op = ReturnOp::new(&mut context);
        append_test_op_v1(&context, entry, &view);
        append_test_op_v1(&context, entry, &create);
        append_test_op_v1(&context, entry, &zero);
        append_test_op_v1(&context, entry, &event);
        append_test_op_v1(&context, entry, &access);
        append_test_op_v1(&context, entry, &semantic);
        append_test_op_v1(&context, entry, &effect_contract);
        append_test_op_v1(&context, entry, &return_op);

        let preservation = begin_production_pliron_pass_contract_session_v1(
            LivePlironStructuralIdentityProviderV1::new(&context, &function),
        )
        .unwrap();
        let census = preservation.input_census_v1();
        assert_eq!(census.pipeline_creates, 1);
        assert_eq!(census.pipeline_events, 1);
        assert_eq!(census.ranked_accesses, 1);
        assert_eq!(census.effect_refinement_contracts, 1);
        assert_eq!(census.index_lt_branch_candidates, 0);
        let mut analyses = PlironAnalysisManagerV1::new(&function);
        analyses.prepare_function_inventory(&context, &function);
        let inventory = analyses.function_inventory_handle().unwrap();
        let fallback = pipeline_protocol_inventory_census_v1(&context, &inventory).unwrap();
        assert_eq!(fallback.max_operation_arity, census.max_operation_arity);
        assert_eq!(fallback.effect_refinement_contracts, 1);
    }

    #[test]
    fn real_nested_cfg_loop_metadata_fits_authenticated_quadratic_storage() {
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        let signature = FunctionType::get(&context, vec![], vec![]);
        let function = FuncOp::new(
            &mut context,
            "nested_loop_metadata".try_into().unwrap(),
            signature,
        );
        let entry = function.get_entry_block(&context);
        let (outer_header, outer) = test_index_block_v1(&mut context, &function, "outer_header", 1);
        let (inner_header, inner) = test_index_block_v1(&mut context, &function, "inner_header", 2);
        let (inner_body, body) = test_index_block_v1(&mut context, &function, "inner_body", 2);
        let (outer_latch, latch) = test_index_block_v1(&mut context, &function, "outer_latch", 1);
        let (exit, _) = test_index_block_v1(&mut context, &function, "exit", 0);
        let zero = IndexConstantOp::new(&mut context, 0);
        let one = IndexConstantOp::new(&mut context, 1);
        let outer_bound = IndexConstantOp::new(&mut context, 4);
        let inner_bound = IndexConstantOp::new(&mut context, 8);
        let zero_value = zero.result(&context);
        let one_value = one.result(&context);
        let outer_bound_value = outer_bound.result(&context);
        let inner_bound_value = inner_bound.result(&context);
        let view_type = RankedViewType::new(&context, 16, true, vec![2, 64]).unwrap();
        let view = RankedViewOp::new_in_space(
            &mut context,
            view_type,
            vec![],
            dialect_kernel::MemorySpaceAttr::Workgroup,
        )
        .unwrap();
        let view_result = view.result(&context);
        let create = PipelineCreateOp::new(&mut context, view_result, 2, 1).unwrap();
        let enter_outer = BranchArgsOp::new(&mut context, vec![zero_value], outer_header);
        let outer_condition = IndexLessThanBranchArgsOp::new(
            &mut context,
            outer[0],
            outer_bound_value,
            vec![outer[0], zero_value],
            vec![],
            inner_header,
            exit,
        );
        let inner_condition = IndexLessThanBranchArgsOp::new(
            &mut context,
            inner[1],
            inner_bound_value,
            inner.clone(),
            vec![inner[0]],
            inner_body,
            outer_latch,
        );
        let next_inner =
            IndexBinaryOp::new(&mut context, IndexBinaryKindAttr::Add, body[1], one_value);
        let next_inner_value = next_inner.result(&context);
        let repeat_inner =
            BranchArgsOp::new(&mut context, vec![body[0], next_inner_value], inner_header);
        let next_outer =
            IndexBinaryOp::new(&mut context, IndexBinaryKindAttr::Add, latch[0], one_value);
        let next_outer_value = next_outer.result(&context);
        let repeat_outer = BranchArgsOp::new(&mut context, vec![next_outer_value], outer_header);
        for operation in [&zero, &one, &outer_bound, &inner_bound] {
            append_test_op_v1(&context, entry, operation);
        }
        append_test_op_v1(&context, entry, &view);
        append_test_op_v1(&context, entry, &create);
        append_test_op_v1(&context, entry, &enter_outer);
        append_test_op_v1(&context, outer_header, &outer_condition);
        append_test_op_v1(&context, inner_header, &inner_condition);
        append_test_op_v1(&context, inner_body, &next_inner);
        append_test_op_v1(&context, inner_body, &repeat_inner);
        append_test_op_v1(&context, outer_latch, &next_outer);
        append_test_op_v1(&context, outer_latch, &repeat_outer);
        let return_op = ReturnOp::new(&mut context);
        append_test_op_v1(&context, exit, &return_op);

        let preservation = begin_production_pliron_pass_contract_session_v1(
            LivePlironStructuralIdentityProviderV1::new(&context, &function),
        )
        .unwrap();
        let census = preservation.input_census_v1();
        assert_eq!(census.pipeline_creates, 1);
        assert_eq!(census.pipeline_events, 0);
        assert_eq!(census.ranked_accesses, 0);
        assert_eq!(census.index_lt_branch_candidates, 2);
        let mut analyses = PlironAnalysisManagerV1::new(&function);
        analyses.prepare_function_inventory(&context, &function);
        let inventory = analyses.function_inventory_handle().unwrap();
        let query_limit = pipeline_equivalence_query_upper_bound_v1(census).unwrap();
        let unique_pair_limit = pipeline_equivalence_unique_pair_upper_bound_v1(census).unwrap();
        let mut resources =
            EquivalenceResourceMeterV1::new(query_limit, unique_pair_limit).unwrap();
        let discovery = discover_epoch_loops(&context, &inventory, &mut resources);
        assert!(!resources.exhausted());
        // Both canonical loops are retained; the child is finite and read/index-only.
        assert_eq!(discovery.loops.len(), 2);
        let observed_items = discovery.dominators.iter().map(HashSet::len).sum::<usize>()
            + discovery
                .loops
                .iter()
                .map(|summary| {
                    summary.prologue.len()
                        + summary.body.len()
                        + summary.body_members.len()
                        + summary.drain.len()
                        + summary.inductions.len() * 2
                        + summary.repeated_blocks.len()
                        + summary.optional_blocks.len()
                })
                .sum::<usize>();
        let authenticated_loop_storage = census.blocks.pow(2) * 10;
        assert!(observed_items <= authenticated_loop_storage);

        let exact = preflight_pipeline_protocol_resource_upper_bound_v1(
            census,
            ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
        )
        .unwrap();
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                census,
                ProductionAnalysisResourceLimitsV1::new(
                    exact.work_upper_bound(),
                    exact.peak_storage_upper_bound() - 1,
                ),
            ),
            Err(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                resource: "peak storage upper bound",
            })
        );
    }
}

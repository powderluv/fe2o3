#[cfg(test)]
mod block_state_resource_tests {
    use super::*;
    use crate::production_analysis::pliron_ir_identity::LivePlironStructuralIdentityProviderV1;
    use crate::production_analysis::pliron_pass_contract::PlironStructuralIdentityProviderV1;
    use crate::production_analysis::pliron_sparse_index::analyze_pliron_sparse_indices_v1;
    use dialect_kernel::{DIALECT_NAME, IndexConstantOp, IndexType, register_dialect};
    use pliron::{
        builtin::{op_interfaces::OneRegionInterface, ops::FuncOp, types::FunctionType},
        dialect::DialectName,
        r#type::TypeHandle,
    };

    fn unlimited() -> ProductionAnalysisResourceLimitsV1 {
        ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX)
    }

    fn setup() -> Context {
        let mut context = Context::new();
        register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
        context
    }

    fn function(context: &mut Context, name: &str, widths: &[usize]) -> FuncOp {
        let index: TypeHandle = IndexType::get(context).into();
        let function = FuncOp::new(
            context,
            name.try_into().unwrap(),
            FunctionType::get(context, vec![index; widths[0]], vec![]),
        );
        let entry = function.get_entry_block(context);
        ReturnOp::new(context)
            .get_operation()
            .insert_at_back(entry, context);
        for width in &widths[1..] {
            let block = BasicBlock::new(context, None, vec![index; *width]);
            block.insert_at_back(function.get_region(context), context);
            ReturnOp::new(context)
                .get_operation()
                .insert_at_back(block, context);
        }
        function
    }

    fn authenticated_inputs(
        context: &Context,
        function: &FuncOp,
    ) -> (
        ProductionAnalysisInputCensusV1,
        BoundedPlironFunctionInventoryV1,
    ) {
        let census = LivePlironStructuralIdentityProviderV1::new(context, function)
            .capture_with_resource_limits_v1(unlimited())
            .ok()
            .expect("the actual function passes structural identity verification")
            .input_census;
        let inventory = BoundedPlironFunctionInventoryV1::collect(context, function).unwrap();
        (census, inventory)
    }

    #[test]
    fn trace_block_state_census_includes_entry_and_unreachable_widths() {
        for (widths, expected) in [([3, 0, 2], 3), ([2, 0, 3], 3), ([0, 0, 0], 0)] {
            let context = &mut setup();
            let function = function(context, "trace_width", &widths);
            let (census, inventory) = authenticated_inputs(context, &function);
            assert_eq!(
                (census.blocks, census.block_arguments),
                (3, widths.iter().sum())
            );
            let result = collect_trace_block_state_census_v1(
                context,
                &inventory,
                census,
                ProductionAnalysisResourceLimitsV1::new(4, 3),
            )
            .unwrap();
            assert_eq!((result.max_block_arguments, result.work), (expected, 4));
        }
    }

    #[test]
    fn trace_block_state_census_prepays_and_rejects_count_mismatches() {
        let context = &mut setup();
        let function = function(context, "trace_width_prefix", &[2, 0, 3]);
        let (census, inventory) = authenticated_inputs(context, &function);
        for (limits, resource) in [
            (
                ProductionAnalysisResourceLimitsV1::new(3, 3),
                "work upper bound",
            ),
            (
                ProductionAnalysisResourceLimitsV1::new(4, 2),
                "peak storage upper bound",
            ),
        ] {
            let error = collect_trace_block_state_census_v1(context, &inventory, census, limits)
                .err()
                .expect("the complete census is prepaid");
            assert_eq!(
                error,
                ProductionAnalysisResourceLimitV1 {
                    phase: ProductionAnalysisResourcePhaseV1::InvocationTrace,
                    resource,
                }
            );
        }
        for bad in [
            ProductionAnalysisInputCensusV1 {
                blocks: 2,
                ..census
            },
            ProductionAnalysisInputCensusV1 {
                blocks: 4,
                ..census
            },
            ProductionAnalysisInputCensusV1 {
                block_arguments: 4,
                ..census
            },
            ProductionAnalysisInputCensusV1 {
                block_arguments: 6,
                ..census
            },
        ] {
            assert_eq!(
                collect_trace_block_state_census_v1(context, &inventory, bad, unlimited()).err(),
                Some(trace_block_state_census_error_v1()),
            );
        }
        let overflow = ProductionAnalysisInputCensusV1 {
            blocks: usize::MAX,
            ..census
        };
        assert_eq!(
            collect_trace_block_state_census_v1(context, &inventory, overflow, unlimited()).err(),
            Some(trace_resource_overflow_v1()),
        );
    }

    #[test]
    fn trace_block_state_bound_has_independent_literal_limits() {
        let context = &mut setup();
        let function = function(context, "trace_width_bound", &[2, 0, 3]);
        let (census, inventory) = authenticated_inputs(context, &function);
        let sparse = analyze_pliron_sparse_indices_v1(context, &function).unwrap();
        assert_eq!(sparse.launch_extents(), &[1]);
        assert_eq!(
            static_invocation_shape_for_resource_v1(&sparse, None),
            Some((1, 1))
        );
        assert_eq!(
            (
                census.blocks,
                census.operations,
                census.results,
                census.block_arguments
            ),
            (3, 3, 0, 5)
        );
        let preflight = preflight_invocation_trace_resource_upper_bound_v1(
            context,
            Some(&inventory),
            census,
            Some(&sparse),
            None,
            unlimited(),
        )
        .unwrap();
        let bound = preflight.attempt_upper_bound();
        // 4*O + B + (B+1) census + one coordinate decode
        // + (MAX+1)*(step1 + value32 + setup3).
        const WORK: usize = 37_748_792;
        // One invocation header and coordinate plus MAX rank-8 event rows.
        const RETAINED: usize = 17_825_794;
        // Block map6 + global environment15 + MAX*(max-width3+key2)
        // + launch extent1 + evaluator scratch1836. The census ends first.
        const PEAK: usize = 23_070_532;
        assert_eq!(
            WORK,
            12 + 3 + 4 + 1 + (MAX_PLIRON_TRACE_TOTAL_STEPS_V1 + 1) * 36
        );
        assert_eq!(RETAINED, 2 + MAX_PLIRON_TRACE_TOTAL_STEPS_V1 * 17);
        assert_eq!(
            PEAK,
            RETAINED + 6 + 15 + MAX_PLIRON_TRACE_TOTAL_STEPS_V1 * 5 + 1 + 1836
        );
        assert_eq!(bound.work_upper_bound(), WORK);
        assert_eq!(bound.retained_storage_upper_bound(), RETAINED);
        assert_eq!(bound.peak_storage_upper_bound(), PEAK);
        assert_eq!(
            preflight_invocation_trace_resource_upper_bound_v1(
                context,
                Some(&inventory),
                census,
                Some(&sparse),
                None,
                ProductionAnalysisResourceLimitsV1::new(WORK, PEAK),
            )
            .unwrap(),
            preflight,
        );
        for (work, storage, resource) in [
            (WORK - 1, PEAK, "work upper bound"),
            (WORK, PEAK - 1, "peak storage upper bound"),
        ] {
            assert_eq!(
                preflight_invocation_trace_resource_upper_bound_v1(
                    context,
                    Some(&inventory),
                    census,
                    Some(&sparse),
                    None,
                    ProductionAnalysisResourceLimitsV1::new(work, storage),
                ),
                Err(ProductionAnalysisResourceLimitV1 {
                    phase: ProductionAnalysisResourcePhaseV1::InvocationTrace,
                    resource,
                }),
            );
        }
        let coarse =
            invocation_trace_resource_upper_bound_for_shape_v1(census, 1, 1, unlimited()).unwrap();
        assert_eq!(
            bound.work_upper_bound(),
            coarse.upper_bound().work_upper_bound() + 4
        );
        assert_eq!(
            coarse.upper_bound().peak_storage_upper_bound() - PEAK,
            2 * MAX_PLIRON_TRACE_TOTAL_STEPS_V1
        );
    }

    #[test]
    fn trace_block_state_equal_aggregates_do_not_supply_the_width() {
        let context = &mut setup();
        let first = function(context, "trace_width_first", &[0, 1, 4]);
        let second = function(context, "trace_width_second", &[0, 2, 3]);
        let mut peaks = Vec::new();
        for (function, width) in [(&first, 4), (&second, 3)] {
            let (census, inventory) = authenticated_inputs(context, function);
            assert_eq!(
                (
                    census.blocks,
                    census.operations,
                    census.results,
                    census.block_arguments
                ),
                (3, 3, 0, 5)
            );
            let actual =
                collect_trace_block_state_census_v1(context, &inventory, census, unlimited())
                    .unwrap();
            assert_eq!(actual.max_block_arguments, width);
            let sparse = analyze_pliron_sparse_indices_v1(context, function).unwrap();
            let result = preflight_invocation_trace_resource_upper_bound_v1(
                context,
                Some(&inventory),
                census,
                Some(&sparse),
                None,
                unlimited(),
            )
            .unwrap();
            peaks.push(result.attempt_upper_bound().peak_storage_upper_bound());
        }
        assert_eq!(peaks[0] - peaks[1], MAX_PLIRON_TRACE_TOTAL_STEPS_V1);
    }

    #[test]
    fn trace_block_state_static_and_rejected_launch_inventory_order_is_preserved() {
        let context = &mut setup();
        let function = function(context, "trace_width_launch", &[0]);
        let (census, inventory) = authenticated_inputs(context, &function);
        let sparse = analyze_pliron_sparse_indices_v1(context, &function).unwrap();
        assert_eq!(
            preflight_invocation_trace_resource_upper_bound_v1(
                context,
                None,
                census,
                Some(&sparse),
                None,
                unlimited(),
            ),
            Err(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::InvocationTrace,
                resource: "invocation trace inventory unavailable",
            }),
        );
        let layout = PlironExecutionLayoutV1 {
            grid: 0,
            global_extents: [1, 1, 1],
            workgroup_extents: [1, 1, 1],
            subgroup_size: 1,
            execution_domain: ExecutionDomainAttr::FullPhysicalWorkgroups,
        };
        for supplied_layout in [None, Some(layout)] {
            let admitted = preflight_invocation_trace_resource_upper_bound_v1(
                context,
                Some(&inventory),
                census,
                Some(&sparse),
                supplied_layout,
                unlimited(),
            )
            .unwrap();
            assert!(admitted.exact_admission.is_some());
            let traces = trace_pliron_invocations_with_inputs_v1(
                context,
                &inventory,
                &sparse,
                supplied_layout,
            )
            .unwrap();
            assert_eq!(traces.len(), 1);
            assert!(admitted.exact_admission(Ok(&traces)).unwrap().is_some());
        }
        // A rejected/dynamic attempt never inspects an unavailable inventory
        // or the irrelevant block totals. Layout presence alone is not sparse authority.
        let bad_census = ProductionAnalysisInputCensusV1 {
            blocks: usize::MAX,
            block_arguments: usize::MAX,
            ..census
        };
        for (sparse, supplied_layout, work, peak) in [
            (None, None, 33, 16),
            (None, Some(layout), 23, 11),
            (
                Some(&sparse),
                Some(PlironExecutionLayoutV1 {
                    global_extents: [0, 1, 1],
                    ..layout
                }),
                23,
                11,
            ),
        ] {
            let result = preflight_invocation_trace_resource_upper_bound_v1(
                context,
                None,
                bad_census,
                sparse,
                supplied_layout,
                ProductionAnalysisResourceLimitsV1::new(work, peak),
            )
            .unwrap();
            assert!(result.exact_admission.is_none());
            assert_eq!(result.attempt_upper_bound().work_upper_bound(), work);
            assert_eq!(
                result.attempt_upper_bound().peak_storage_upper_bound(),
                peak
            );
        }
    }

    #[test]
    fn trace_block_state_repeated_keys_and_rejecting_step_keep_the_same_boundary() {
        for width in [0, 1] {
            let context = &mut setup();
            let function = FuncOp::new(
                context,
                "trace_width_cycle".try_into().unwrap(),
                FunctionType::get(context, vec![], vec![]),
            );
            let entry = function.get_entry_block(context);
            let index: TypeHandle = IndexType::get(context).into();
            let cycle = BasicBlock::new(context, None, vec![index; width]);
            cycle.insert_at_back(function.get_region(context), context);
            let mut inputs = Vec::new();
            if width != 0 {
                let zero = IndexConstantOp::new(context, 0);
                zero.get_operation().insert_at_back(entry, context);
                inputs.push(zero.result(context));
            }
            BranchArgsOp::new(context, inputs, cycle)
                .get_operation()
                .insert_at_back(entry, context);
            let arguments = (0..width)
                .map(|ordinal| cycle.deref(context).get_argument(ordinal))
                .collect();
            BranchArgsOp::new(context, arguments, cycle)
                .get_operation()
                .insert_at_back(cycle, context);
            let (census, inventory) = authenticated_inputs(context, &function);
            let sparse = analyze_pliron_sparse_indices_v1(context, &function).unwrap();
            let admitted = preflight_invocation_trace_resource_upper_bound_v1(
                context,
                Some(&inventory),
                census,
                Some(&sparse),
                None,
                unlimited(),
            )
            .unwrap();
            let failure =
                trace_pliron_invocations_with_inputs_v1(context, &inventory, &sparse, None)
                    .unwrap_err();
            assert_eq!(
                failure,
                PlironTraceFailureV1::CyclicControlFlow { block: 1 }
            );
            assert_eq!(admitted.exact_admission(Err(failure)), Ok(None));

            // A duplicate candidate coexists with old keys but consumes its
            // own admitted step. The next rejected step allocates no key.
            let mut visited = HashSet::new();
            assert!(visited.insert((1_usize, vec![Some(0_u64); width])));
            let candidate = (1_usize, vec![Some(0_u64); width]);
            assert_eq!(
                visited.len() * (width + 2) + candidate.1.len() + 2,
                2 * (width + 2)
            );
            assert!(!visited.insert(candidate));
            let mut steps = MAX_PLIRON_TRACE_TOTAL_STEPS_V1 - 1;
            assert_eq!(charge_trace_work_v1(&mut steps, 1), Ok(()));
            let mut allocated = false;
            let denied = charge_trace_work_v1(&mut steps, 1).map(|()| {
                allocated = true;
                vec![None::<u64>; width]
            });
            assert_eq!(denied, Err(PlironTraceFailureV1::ResourceLimit));
            assert!(!allocated);
            assert_eq!(steps, MAX_PLIRON_TRACE_TOTAL_STEPS_V1 + 1);
        }
    }

    // Same actual 1024/256/64 geometry as the frozen tutorial; the entry's
    // runtime parameters remain immutable source inputs, not launch hints.
    fn entry_key_fixture(context: &mut Context, control: u8) -> FuncOp {
        dialect_gpu::register_dialect(context).unwrap();
        let index: TypeHandle = IndexType::get(context).into();
        let function = FuncOp::new(
            context,
            "entry_key_lifetime".try_into().unwrap(),
            FunctionType::get(context, vec![index; 55], vec![]),
        );
        let entry = function.get_entry_block(context);
        let child = BasicBlock::new(context, None, vec![index; 3]);
        child.insert_at_back(function.get_region(context), context);
        dialect_gpu::ExecutionLayoutOp::new(context, 71, [1024, 1, 1], [256, 1, 1], 64)
            .get_operation()
            .insert_at_back(entry, context);
        let zero = IndexConstantOp::new(context, 0);
        zero.get_operation().insert_at_back(entry, context);
        let z = zero.result(context);
        if control == 2 {
            dialect_kernel::IndexLessThanBranchArgsOp::new(
                context,
                z,
                z,
                vec![z; 55],
                vec![z; 55],
                entry,
                entry,
            )
            .get_operation()
            .insert_at_back(entry, context);
        } else {
            BranchArgsOp::new(context, vec![z; 3], child)
                .get_operation()
                .insert_at_back(entry, context);
        }
        if control == 1 {
            BranchArgsOp::new(context, vec![z; 55], entry)
                .get_operation()
                .insert_at_back(child, context);
        } else {
            ReturnOp::new(context)
                .get_operation()
                .insert_at_back(child, context);
        }
        if control == 3 {
            // Both occurrences are unreachable but still forbid the discount.
            let unreachable = BasicBlock::new(context, None, vec![]);
            unreachable.insert_at_back(function.get_region(context), context);
            let zero = IndexConstantOp::new(context, 0);
            zero.get_operation().insert_at_back(unreachable, context);
            let z = zero.result(context);
            dialect_kernel::IndexLessThanBranchArgsOp::new(
                context,
                z,
                z,
                vec![z; 55],
                vec![z; 55],
                entry,
                entry,
            )
            .get_operation()
            .insert_at_back(unreachable, context);
        }
        if control == 4 {
            let wide = BasicBlock::new(context, None, vec![index; 55]);
            wide.insert_at_back(function.get_region(context), context);
            ReturnOp::new(context)
                .get_operation()
                .insert_at_back(wide, context);
        }
        function
    }

    #[test]
    fn wide_entry_key_has_one_invocation_local_lifetime_with_real_geometry() {
        let context = &mut setup();
        let function = entry_key_fixture(context, 0);
        let (census, inventory) = authenticated_inputs(context, &function);
        assert_eq!(
            (
                census.blocks,
                census.operations,
                census.results,
                census.block_arguments,
                census.successors
            ),
            (2, 4, 1, 58, 1)
        );
        let sparse = analyze_pliron_sparse_indices_v1(context, &function).unwrap();
        let layout = pliron_execution_layout_with_inventory_v1(context, &inventory)
            .unwrap()
            .unwrap();
        assert_eq!(layout.global_extents, [1024, 1, 1]);
        assert_eq!(layout.workgroup_extents, [256, 1, 1]);
        let exact = preflight_invocation_trace_resource_upper_bound_v1(
            context,
            Some(&inventory),
            census,
            Some(&sparse),
            Some(layout),
            unlimited(),
        )
        .unwrap();
        let mut original =
            collect_trace_block_state_census_v1(context, &inventory, census, unlimited()).unwrap();
        let old_work = original.work;
        let refined = preflight_single_entry_key_storage_v1(
            context,
            &inventory,
            census,
            &mut original,
            1024,
            unlimited(),
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(refined, 5_242_932); // M*(3+2)+(55-3)
        assert_eq!(original.work - old_work, 2 + 4 + 1 + 1);
        let coarse = invocation_trace_resource_upper_bound_for_block_state_v1(
            census,
            InvocationTraceBlockStateCensusV1 {
                max_block_arguments: 55,
                work: old_work,
            },
            1024,
            3,
            unlimited(),
        )
        .unwrap()
        .upper_bound();
        let bound = exact.attempt_upper_bound();
        assert_eq!(
            coarse.peak_storage_upper_bound() - bound.peak_storage_upper_bound(),
            (55 - 3) * (MAX_PLIRON_TRACE_TOTAL_STEPS_V1 - 1)
        );
        assert_eq!(bound.work_upper_bound(), 37_751_873);
        assert_eq!(bound.peak_storage_upper_bound(), 23_074_838);
        let traces =
            trace_pliron_invocations_with_inputs_v1(context, &inventory, &sparse, Some(layout))
                .unwrap();
        assert_eq!(traces.len(), 1024);
        assert!(traces.iter().all(|trace| trace.events.is_empty()));
        assert_eq!(
            exact
                .exact_admission(Ok(&traces))
                .unwrap()
                .unwrap()
                .invocation_count(),
            1024
        );
        eprintln!(
            "wide-entry actual census B={} O={} A={} E={}, requested work={} retained={} peak={}",
            census.blocks,
            census.operations,
            census.block_arguments,
            census.successors,
            bound.work_upper_bound(),
            bound.retained_storage_upper_bound(),
            bound.peak_storage_upper_bound()
        );
    }

    #[test]
    fn wide_entry_preflight_exact_and_one_under_keep_real_manager_prefix() {
        use crate::production_analysis::pliron_analysis_manager::PlironAnalysisManagerV1;
        let context = &mut setup();
        let function = entry_key_fixture(context, 0);
        let (census, inventory) = authenticated_inputs(context, &function);
        let sparse = analyze_pliron_sparse_indices_v1(context, &function).unwrap();
        let layout = pliron_execution_layout_with_inventory_v1(context, &inventory).unwrap();
        let phase = ProductionAnalysisResourcePhaseV1::InvocationTrace;
        let setup = ProductionAnalysisResourceUpperBoundV1::checked_phase(
            ProductionAnalysisResourcePhaseV1::StructuralIdentity,
            7,
            11,
            13,
        )
        .unwrap();
        let make = |limits| {
            PlironAnalysisManagerV1::new_with_resource_contract(&function, census, setup, 0, limits)
                .unwrap()
        };
        let mut probe = make(unlimited());
        probe.prepare_function_inventory(context, &function);
        let plan = preflight_invocation_trace_resource_upper_bound_v1(
            context,
            Some(&inventory),
            census,
            Some(&sparse),
            layout,
            probe.remaining_resource_limits(phase).unwrap(),
        )
        .unwrap();
        probe
            .admit_retained_resource_upper_bound(phase, plan.attempt_upper_bound())
            .unwrap();
        let exact = probe.resource_upper_bound();
        for (work, peak, accepted) in [
            (
                exact.work_upper_bound(),
                exact.peak_storage_upper_bound(),
                true,
            ),
            (
                exact.work_upper_bound() - 1,
                exact.peak_storage_upper_bound(),
                false,
            ),
            (
                exact.work_upper_bound(),
                exact.peak_storage_upper_bound() - 1,
                false,
            ),
        ] {
            let mut manager = make(ProductionAnalysisResourceLimitsV1::new(work, peak));
            manager.prepare_function_inventory(context, &function);
            let before = manager.resource_upper_bound();
            let plan = preflight_invocation_trace_resource_upper_bound_v1(
                context,
                Some(&inventory),
                census,
                Some(&sparse),
                layout,
                manager.remaining_resource_limits(phase).unwrap(),
            );
            assert_eq!(plan.is_ok(), accepted);
            if accepted {
                manager
                    .admit_retained_resource_upper_bound(phase, plan.unwrap().attempt_upper_bound())
                    .unwrap();
                assert_eq!(manager.resource_upper_bound(), exact);
                manager.prepare_exact_trace(context, &function);
                assert_eq!(manager.exact_trace().unwrap().len(), 1024);
            } else {
                assert_eq!(manager.resource_upper_bound(), before);
                assert!(!manager.has_successful_exact_trace_v1());
            }
        }
        eprintln!(
            "actual manager cumulative work={} peak={}, nonzero prefix preserved",
            exact.work_upper_bound(),
            exact.peak_storage_upper_bound()
        );
    }

    #[test]
    fn wide_entry_reentry_unreachable_and_duplicate_occurrences_keep_coarse_storage() {
        for control in [1, 2, 3, 4] {
            let context = &mut setup();
            let function = entry_key_fixture(context, control);
            let (census, inventory) = authenticated_inputs(context, &function);
            let mut state =
                collect_trace_block_state_census_v1(context, &inventory, census, unlimited())
                    .unwrap();
            assert_eq!(state.max_block_arguments, 55);
            assert_eq!(
                preflight_single_entry_key_storage_v1(
                    context,
                    &inventory,
                    census,
                    &mut state,
                    1024,
                    unlimited(),
                    None
                )
                .unwrap(),
                None,
                "control {control}"
            );
            let sparse = analyze_pliron_sparse_indices_v1(context, &function);
            if control <= 3 {
                assert!(
                    matches!(
                        sparse,
                        Err(crate::SparseIndexFailureV1::MalformedControlFlow { .. })
                    ),
                    "the genuine source CFG rejects an incoming entry operand: {control}"
                );
                continue;
            }
            let sparse = sparse.unwrap();
            let layout = pliron_execution_layout_with_inventory_v1(context, &inventory).unwrap();
            let bound = preflight_invocation_trace_resource_upper_bound_v1(
                context,
                Some(&inventory),
                census,
                Some(&sparse),
                layout,
                unlimited(),
            )
            .unwrap()
            .attempt_upper_bound();
            // The additional scan is paid, and every key may still be wide.
            let coarse = invocation_trace_resource_upper_bound_for_block_state_v1(
                census,
                state,
                1024,
                3,
                unlimited(),
            )
            .unwrap()
            .upper_bound();
            assert_eq!(bound, coarse, "control {control}");
        }
    }

    #[test]
    fn wide_entry_scan_is_prepaid_count_checked_and_zero_route_is_unchanged() {
        let context = &mut setup();
        let function = entry_key_fixture(context, 0);
        let (census, inventory) = authenticated_inputs(context, &function);
        for bad in [
            ProductionAnalysisInputCensusV1 {
                successors: 0,
                ..census
            },
            ProductionAnalysisInputCensusV1 {
                successors: 2,
                ..census
            },
            ProductionAnalysisInputCensusV1 {
                operations: 3,
                ..census
            },
        ] {
            let mut state =
                collect_trace_block_state_census_v1(context, &inventory, census, unlimited())
                    .unwrap();
            assert_eq!(
                preflight_single_entry_key_storage_v1(
                    context,
                    &inventory,
                    bad,
                    &mut state,
                    1024,
                    unlimited(),
                    None
                ),
                Err(trace_block_state_census_error_v1())
            );
        }
        for (work, peak, expected) in [
            (10, 12, "work upper bound"),
            (11, 11, "peak storage upper bound"),
        ] {
            let mut state =
                collect_trace_block_state_census_v1(context, &inventory, census, unlimited())
                    .unwrap();
            let before = state.work;
            let error = preflight_single_entry_key_storage_v1(
                context,
                &inventory,
                census,
                &mut state,
                1024,
                ProductionAnalysisResourceLimitsV1::new(work, peak),
                None,
            )
            .unwrap_err();
            assert_eq!(error.resource, expected);
            assert_eq!(state.work, before);
        }
        let mut state =
            collect_trace_block_state_census_v1(context, &inventory, census, unlimited()).unwrap();
        let before = state.work;
        assert_eq!(
            preflight_single_entry_key_storage_v1(
                context,
                &inventory,
                census,
                &mut state,
                0,
                ProductionAnalysisResourceLimitsV1::new(0, 0),
                None
            )
            .unwrap(),
            None
        );
        assert_eq!(state.work, before);
    }
    #[test]
    fn one_block_entry_and_overflow_keep_finite_storage_and_step_limits() {
        let context = &mut setup();
        let function = function(context, "single_wide_entry", &[55]);
        let (census, inventory) = authenticated_inputs(context, &function);
        let mut state =
            collect_trace_block_state_census_v1(context, &inventory, census, unlimited()).unwrap();
        assert_eq!(
            preflight_single_entry_key_storage_v1(
                context,
                &inventory,
                census,
                &mut state,
                1,
                unlimited(),
                None
            )
            .unwrap(),
            Some(MAX_PLIRON_TRACE_TOTAL_STEPS_V1 * 2 + 55)
        );
        let sparse = analyze_pliron_sparse_indices_v1(context, &function).unwrap();
        let preflight = preflight_invocation_trace_resource_upper_bound_v1(
            context,
            Some(&inventory),
            census,
            Some(&sparse),
            None,
            unlimited(),
        )
        .unwrap();
        let traces =
            trace_pliron_invocations_with_inputs_v1(context, &inventory, &sparse, None).unwrap();
        assert_eq!(traces.len(), 1);
        assert!(preflight.exact_admission(Ok(&traces)).unwrap().is_some());
        let mut state =
            collect_trace_block_state_census_v1(context, &inventory, census, unlimited()).unwrap();
        for bad in [
            ProductionAnalysisInputCensusV1 {
                operations: usize::MAX,
                ..census
            },
            ProductionAnalysisInputCensusV1 {
                successors: usize::MAX,
                ..census
            },
        ] {
            assert_eq!(
                preflight_single_entry_key_storage_v1(
                    context,
                    &inventory,
                    bad,
                    &mut state,
                    1,
                    unlimited(),
                    None
                ),
                Err(trace_resource_overflow_v1())
            );
        }
    }
}

    #[test]
    fn assignment_and_atomic_result_stores_follow_operand_reads() {
        let assignment = SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            ranked_place(0),
            SemanticRvalueV1::new(
                SCALAR_TYPE,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(ranked_place(1))),
            ),
        ));
        let (operations, sources, _) = audit_statements(vec![statement(assignment)]).unwrap();
        assert_eq!(access_kinds(&operations), vec![AccessKindAttr::Read, AccessKindAttr::Write]);
        assert_eq!(sources.len(), 2);
        let atomic = SemanticStatementKindV1::AtomicRmw(SemanticAtomicRmwV1::new(
            ranked_place(2),
            ranked_place(0),
            SemanticOperandV1::Copy(ranked_place(1)),
            SemanticAtomicRmwOpV1::Add,
            atomic_access(),
        ));
        let (operations, sources, _) = audit_statements(vec![statement(atomic)]).unwrap();
        assert_eq!(access_kinds(&operations), vec![
            AccessKindAttr::Read, AccessKindAttr::AtomicReadModifyWrite, AccessKindAttr::Write,
        ]);
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn regular_and_atomic_stores_project_destination_and_value_footprints() {
        for atomic in [None, Some(atomic_access())] {
            let (operations, sources, _) = audit_statements(vec![statement(
                SemanticStatementKindV1::Store(SemanticMemoryStoreV1::new(
                    ranked_place(0),
                    SemanticOperandV1::Copy(ranked_place(1)),
                    SemanticVolatilityV1::NonVolatile,
                    atomic,
                )),
            )])
            .unwrap();
            assert_eq!(
                access_kinds(&operations),
                vec![
                    AccessKindAttr::Read,
                    if atomic.is_some() {
                        AccessKindAttr::AtomicWrite
                    } else {
                        AccessKindAttr::Write
                    },
                ]
            );
            assert_eq!(sources.len(), 2);
            assert_eq!(
                operations
                    .iter()
                    .filter(|operation| matches!(
                        operation,
                        ProductionRankedOperationV1::ViewInSpace { .. }
                    ))
                    .count(),
                1,
                "two effects on one semantic allocation created different PLIRON views",
            );
        }
    }

    #[test]
    fn checked_binary_projects_copy_and_move_operand_reads_in_order() {
        let checked = SemanticRvalueKindV1::CheckedBinary(SemanticCheckedBinaryRvalueV1::new(
            SemanticCheckedBinaryOpV1::Multiply,
            SemanticOperandV1::Copy(ranked_place(0)),
            SemanticOperandV1::Move(ranked_place(1)),
        ));
        let function = projection_function(vec![block(30, vec![], SemanticTerminatorKindV1::Return)]);
        let types = projection_types();
        let mut operations = Vec::new();
        let mut sources = Vec::new();
        let mut guarded_sites = Vec::new();
        let mut projected_views = ProjectedViewsV1::new(function.locals().len(), None);
        let mut next_value = 0;
        let mut ranked_ir = String::new();
        let local_contracts = synthetic_local_contracts(&function);

        project_rvalue_reads(
            &types,
            &function,
            0,
            &[],
            &checked,
            SemanticSourceProvenanceV1::unavailable(),
            &[None; 4],
            &local_contracts,
            &[],
            &mut guarded_sites,
            &mut projected_views,
            &mut operations,
            &mut sources,
            &mut next_value,
            &mut ranked_ir,
        )
        .unwrap();

        assert_eq!(
            access_kinds(&operations),
            vec![AccessKindAttr::Read, AccessKindAttr::Read]
        );
        assert_eq!(sources.len(), 2);
        assert_eq!(
            ranked_ir.matches("kernel.access").count(),
            2,
            "both checked operands must survive ranked projection"
        );
    }

    #[test]
    fn guarded_disjoint_access_is_ordinary_clean_pliron_cfg() {
        let invocation = ProductionRankedValueIdV1::new(0);
        let view = ProductionRankedValueIdV1::new(1);
        let entry = vec![
            ProductionRankedOperationV1::InvocationIndex {
                result: invocation,
                dimension: 0,
                launch_extent: 64,
            },
            ProductionRankedOperationV1::ViewInSpace {
                result: view,
                element_width: 32,
                writable: true,
                shape: vec![DYNAMIC_EXTENT],
                dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
                memory_space: MemorySpaceAttr::Global,
                allocation_origin: 0,
                noalias_class: 0,
            },
        ];
        let guarded = GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view,
            indices: vec![ProductionRankedValueV1::Local(invocation)],
            checked_success: None,
            comparisons: vec![(
                ProductionRankedValueV1::Local(invocation),
                ProductionRankedValueV1::Argument(0),
            )],
            access: AccessKindAttr::Write,
            memory_space: MemorySpaceAttr::Global,
            source: SemanticSourceProvenanceV1::unavailable(),
            semantic_site: None,
            output_extent: None,
        };
        let (blocks, sources, ranked_ir) = single_guarded_cfg(entry, guarded);
        assert_eq!(blocks.len(), 5);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].block, 2);
        assert!(matches!(
            blocks[3].terminator(),
            ProductionRankedTerminatorV1::Trap
        ));
        let kernel = ProductionRankedKernelV1::new("generic_checked_access", 1, blocks).unwrap();
        let construction =
            ProductionConstructionV1::ranked_kernel("checked_access_module", kernel).unwrap();
        let lowering =
            compile_ranked_kernel_for_lowering_v1(construction, ProductionSessionLimitsV1::default())
                .unwrap();
        assert!(lowering.bounds_report().is_clean());
        assert!(lowering.race_report().is_clean());
        assert!(ranked_ir.contains("kernel.cond_br") && ranked_ir.contains("kernel.access"));
        assert!(ranked_ir.contains("kernel.br ^bb4"));
    }

    #[test]
    fn checked_extent_provenance_survives_cfg_relocation_and_access_retention() {
        // Inert component custody only: no canonical/source authentication.
        let view = ProductionRankedValueIdV1::new(1);
        let index = ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(0));
        let extent = ProductionRankedValueV1::Argument(0);
        let proposal = ProductionRankedOutputExtentSourceV1::new(
            7, ProductionRankedValueV1::Local(view), extent, index,
        );
        let guarded = GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view,
            indices: vec![index],
            checked_success: None,
            comparisons: vec![(index, extent)],
            access: AccessKindAttr::Write,
            memory_space: MemorySpaceAttr::Global,
            source: SemanticSourceProvenanceV1::unavailable(),
            semantic_site: Some(ProjectedSemanticAccessSiteV1 { block: 0, statement: None }),
            output_extent: Some(proposal),
        };
        let (blocks, sources, _) = single_guarded_cfg(vec![], guarded);
        assert_eq!(sources[0].block, 2);
        assert_eq!(sources[0].output_extent, Some(proposal));
        let function = projection_function(vec![block(29, vec![], SemanticTerminatorKindV1::Return)]);
        let retained = production_access_sources(
            &projection_types(), &function, &blocks, &sources,
            &mut ComponentDynamicAssertionFactsV1,
        ).unwrap();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].ranked_block(), 2);
        assert_eq!(retained[0].semantic_access_ordinal(), 0);
        assert_eq!(retained[0].output_extent(), Some(proposal));
        assert_eq!(proposal.source_argument(), 7);
    }

    #[test]
    fn rust_bounds_check_projects_only_the_exact_index_less_than_length_guard() {
        let function = bounds_check_function(SemanticBinaryOpV1::LessThan, true, false, false);
        let mut operations = Vec::new();
        let mut next_value = 0;
        let projected =
            project_rust_bounds_checks(&function, 3, &[], &mut operations, &mut next_value).unwrap();

        assert_eq!(projected.argument_count, 3);
        assert_eq!(next_value, 2);
        assert!(matches!(
            operations.as_slice(),
            [
                ProductionRankedOperationV1::IndexUnknown { result: first },
                ProductionRankedOperationV1::IndexUnknown { result: second },
            ] if first.get() == 0 && second.get() == 1
        ));
        assert_eq!(projected.checks.len(), 1);
        assert_eq!(projected.checks[0].access_block, 1);
        assert_eq!(
            projected.checks[0].extent_source,
            ProjectedBoundsExtentSourceV1::Slice(SemanticLocalIdV1::from_index(1))
        );
        assert_eq!(projected.checks[0].index_local.index(), 4);
        assert_eq!(
            projected.checks[0].index,
            ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(0))
        );
        assert_eq!(
            projected.checks[0].extent,
            ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(1))
        );
    }

    #[test]
    fn rust_bounds_check_reuses_an_exact_compiler_intrinsic_index_fact() {
        let function = bounds_check_function(SemanticBinaryOpV1::LessThan, true, false, false);
        let exact = ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(99));
        let mut known = vec![None; function.locals().len()];
        known[4] = Some(ProjectedDisjointIndexV1 {
            value: exact,
            mapping: SemanticDisjointIndexSpaceV1::Index1d,
            precondition: None,
            availability: None,
        });
        let mut operations = Vec::new();
        let mut next_value = 0;
        let projected =
            project_rust_bounds_checks(&function, 3, &known, &mut operations, &mut next_value).unwrap();

        assert_eq!(projected.checks[0].index, exact);
        assert_eq!(next_value, 1);
        assert!(matches!(
            operations.as_slice(),
            [ProductionRankedOperationV1::IndexUnknown { result }] if result.get() == 0
        ));
    }

    #[test]
    fn repeated_bounds_checks_on_one_stable_slice_share_the_exact_extent() {
        let function = repeated_bounds_check_function(true, false);
        let mut operations = Vec::new();
        let mut next_value = 0;
        let projected =
            project_rust_bounds_checks(&function, 0, &[], &mut operations, &mut next_value).unwrap();

        assert_eq!(projected.checks.len(), 2);
        assert_eq!(
            projected.checks[0].extent_source,
            projected.checks[1].extent_source
        );
        assert_eq!(projected.checks[0].extent, projected.checks[1].extent);
        assert_ne!(projected.checks[0].index, projected.checks[1].index);
        assert_eq!(operations.len(), 3);
        assert_eq!(next_value, 3);
    }

    #[test]
    fn repeated_bounds_extent_identity_does_not_cross_slice_or_mutation_boundaries() {
        let different_slices = repeated_bounds_check_function(false, false);
        let mut operations = Vec::new();
        let mut next_value = 0;
        let projected =
            project_rust_bounds_checks(&different_slices, 0, &[], &mut operations, &mut next_value)
                .unwrap();
        assert_ne!(
            projected.checks[0].extent_source,
            projected.checks[1].extent_source
        );
        assert_ne!(projected.checks[0].extent, projected.checks[1].extent);
        assert_eq!(operations.len(), 4);
        assert_eq!(next_value, 4);

        assert!(matches!(
            project_test_bounds_checks(&repeated_bounds_check_function(true, true), 0),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "repeated Rust bounds checks derive an extent from a mutable slice local"
            ))
        ));
    }

    #[test]
    fn literal_array_bounds_check_does_not_manufacture_dynamic_authorization() {
        for function in [
            literal_bounds_check_function(63, 64),
            literal_bounds_check_function(64, 64),
        ] {
            let mut operations = Vec::new();
            let mut next_value = 0;
            let projected =
                project_rust_bounds_checks(&function, 0, &[], &mut operations, &mut next_value)
                    .unwrap();
            assert!(projected.checks.is_empty());
            assert!(operations.is_empty());
            assert_eq!(next_value, 0);
        }
    }

    #[test]
    fn forged_rust_bounds_messages_and_conditions_fail_closed() {
        assert!(matches!(
            project_test_bounds_checks(
                &bounds_check_function(SemanticBinaryOpV1::GreaterThan, true, false, false),
                0,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a Rust bounds-check message not backed by its exact index < length condition"
            ))
        ));
        assert!(matches!(
            project_test_bounds_checks(
                &bounds_check_function(SemanticBinaryOpV1::LessThan, true, true, false),
                0,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a Rust bounds-check length not derived from one exact slice"
            ))
        ));
        assert!(matches!(
            project_test_bounds_checks(
                &bounds_check_function(SemanticBinaryOpV1::LessThan, false, false, false),
                0,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a Rust bounds check without the canonical success/unreachable shape"
            ))
        ));
    }

    #[test]
    fn rust_bounds_check_cannot_authorize_another_slice_or_a_bypass_edge() {
        let function = bounds_check_function(SemanticBinaryOpV1::LessThan, true, false, false);
        let projected = project_test_bounds_checks(&function, 0).unwrap();
        assert!(matches!(
            projected_bounds_check(
                &projected.checks,
                1,
                ProjectedBoundsExtentSourceV1::Slice(SemanticLocalIdV1::from_index(3)),
                SemanticLocalIdV1::from_index(4),
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a dynamic slice access without its exact Rust bounds-check predecessor"
            ))
        ));

        assert!(matches!(
            project_test_bounds_checks(
                &bounds_check_function(SemanticBinaryOpV1::LessThan, true, false, true),
                0,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a Rust bounds-check success block not uniquely controlled by that check"
            ))
        ));
    }

    #[test]
    fn branch_form_rust_bounds_checks_accept_both_canonical_boolean_encodings() {
        for switch_values in [vec![0], vec![1]] {
            let mut operations = Vec::new();
            let mut next_value = 0;
            let projected = project_rust_bounds_checks(
                &branch_bounds_check_function(BranchBoundsCheckOptionsV1 {
                    switch_values,
                    ..BranchBoundsCheckOptionsV1::default()
                }),
                7,
                &[],
                &mut operations,
                &mut next_value,
            )
            .unwrap();

            assert_eq!(projected.argument_count, 7);
            assert_eq!(projected.checks.len(), 1);
            assert_eq!(projected.checks[0].access_block, 1);
            assert_eq!(
                projected.checks[0].extent_source,
                ProjectedBoundsExtentSourceV1::Slice(SemanticLocalIdV1::from_index(1))
            );
            assert_eq!(projected.checks[0].index_local.index(), 4);
            assert!(!projected.checks[0].must_authorize_access);
            assert_eq!(operations.len(), 2);
            assert_eq!(next_value, 2);
        }
    }

    #[test]
    fn unrelated_branch_comparisons_do_not_become_bounds_authority() {
        for options in [
            BranchBoundsCheckOptionsV1 {
                operation: SemanticBinaryOpV1::GreaterThan,
                ..BranchBoundsCheckOptionsV1::default()
            },
            BranchBoundsCheckOptionsV1 {
                swap_comparison_operands: true,
                ..BranchBoundsCheckOptionsV1::default()
            },
            BranchBoundsCheckOptionsV1 {
                length_from_slice: false,
                ..BranchBoundsCheckOptionsV1::default()
            },
        ] {
            let projected =
                project_test_bounds_checks(&branch_bounds_check_function(options), 0).unwrap();
            assert!(projected.checks.is_empty());
        }
    }

    #[test]
    fn malformed_or_mutable_branch_bounds_evidence_fails_closed() {
        for (options, detail) in [
            (
                BranchBoundsCheckOptionsV1 {
                    switch_values: vec![0, 1],
                    ..BranchBoundsCheckOptionsV1::default()
                },
                "a branch-form Rust bounds check without one exact boolean target",
            ),
            (
                BranchBoundsCheckOptionsV1 {
                    switch_values: vec![2],
                    ..BranchBoundsCheckOptionsV1::default()
                },
                "a branch-form Rust bounds check with a non-boolean switch value",
            ),
            (
                BranchBoundsCheckOptionsV1 {
                    alternate_predecessor: true,
                    ..BranchBoundsCheckOptionsV1::default()
                },
                "a Rust bounds-check success block not uniquely controlled by that check",
            ),
            (
                BranchBoundsCheckOptionsV1 {
                    duplicate_condition: true,
                    ..BranchBoundsCheckOptionsV1::default()
                },
                "a Rust bounds check whose condition, index, or length is not stable",
            ),
            (
                BranchBoundsCheckOptionsV1 {
                    duplicate_index: true,
                    ..BranchBoundsCheckOptionsV1::default()
                },
                "a Rust bounds check whose condition, index, or length is not stable",
            ),
            (
                BranchBoundsCheckOptionsV1 {
                    duplicate_length: true,
                    ..BranchBoundsCheckOptionsV1::default()
                },
                "a Rust bounds check whose condition, index, or length is not stable",
            ),
        ] {
            assert!(matches!(
                project_test_bounds_checks(&branch_bounds_check_function(options), 0),
                Err(ProductionRankedProjectionErrorV1::Incomplete(actual)) if actual == detail
            ));
        }
    }

    #[test]
    fn safe_syncthreads_reaches_mandatory_barrier_and_workgroup_checks() {
        let kernel = ProductionRankedKernelV1::new(
            "safe_syncthreads",
            0,
            vec![ProductionRankedBlockV1::new(
                vec![
                    ProductionRankedOperationV1::ExecutionLayout {
                        grid_identity: 1,
                        global_extents: [64, 1, 1],
                        workgroup_extents: [64, 1, 1],
                        subgroup_size: 64,
                        full_physical_workgroups: true,
                    },
                    ProductionRankedOperationV1::Barrier {
                        execution_scope: HierarchyAttr::Workgroup,
                        memory_scope: MemoryScopeAttr::Workgroup,
                        address_space: AddressSpaceAttr::Workgroup,
                        order: MemoryOrderAttr::AcquireRelease,
                    },
                ],
                ProductionRankedTerminatorV1::Return,
            )],
        )
        .unwrap();
        let construction =
            ProductionConstructionV1::ranked_kernel("safe_syncthreads_module", kernel).unwrap();
        let lowering =
            compile_ranked_kernel_for_lowering_v1(construction, ProductionSessionLimitsV1::default())
                .unwrap();
        assert!(lowering.barrier_report().is_clean());
        assert!(lowering.workgroup_report().is_clean());
    }

    #[test]
    fn implicit_publish_and_explicit_barrier_share_one_ranked_contract() {
        assert!(requires_ranked_workgroup_barrier_v1(
            &SemanticCompilerIntrinsicOperationV1::WorkgroupBarrier,
        ));
        assert!(!requires_ranked_workgroup_barrier_v1(
            &SemanticCompilerIntrinsicOperationV1::WorkgroupReduceSum {
                workgroup: SCALAR_TYPE,
                context: SCALAR_TYPE,
                scratch: SCALAR_TYPE,
                element: SCALAR_TYPE,
            },
        ));
        assert!(requires_ranked_workgroup_barrier_v1(
            &SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposePublish {
                input_tile: SCALAR_TYPE,
                output_tile: SCALAR_TYPE,
                format: SemanticGfx950LdsTransposeFormatV1::Fp8E4M3,
            },
        ));
        assert!(!requires_ranked_workgroup_barrier_v1(
            &SemanticCompilerIntrinsicOperationV1::Gfx950LdsTransposeStage {
                input_tile: SCALAR_TYPE,
                output_tile: SCALAR_TYPE,
                view: SCALAR_TYPE,
                format: SemanticGfx950LdsTransposeFormatV1::Fp8E4M3,
            },
        ));
        assert!(requires_ranked_workgroup_barrier_v1(
            &SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineEvent {
                pipeline: SCALAR_TYPE,
                event: SemanticWorkgroupPipelineEventV1::Wait,
            },
        ));
    }

    #[test]
    fn guarded_access_and_barrier_retain_semantic_source_order() {
        let guarded = || {
            ProjectedBlockItemV1::Guarded(GuardedRankedAccessV1 {
                failure: GuardedAccessFailureV1::Trap,
                view: ProductionRankedValueIdV1::new(0),
                indices: vec![ProductionRankedValueV1::Argument(0)],
                checked_success: None,
                comparisons: vec![(
                    ProductionRankedValueV1::Argument(0),
                    ProductionRankedValueV1::Argument(1),
                )],
                access: AccessKindAttr::Write,
                memory_space: MemorySpaceAttr::Global,
                source: SemanticSourceProvenanceV1::unavailable(),
                output_extent: None,
                semantic_site: None,
            })
        };
        let barrier = || ProjectedBlockItemV1::Effect {
            operation: ProductionRankedOperationV1::Barrier {
                execution_scope: HierarchyAttr::Workgroup,
                memory_scope: MemoryScopeAttr::Workgroup,
                address_space: AddressSpaceAttr::Workgroup,
                order: MemoryOrderAttr::AcquireRelease,
            },
            source: None,
        };
        let function = projection_function(vec![
            block(70, vec![], barrier_call(Some(1))),
            block(71, vec![], SemanticTerminatorKindV1::Return),
        ]);

        let (after_blocks, _, _) = build_ranked_cfg(
            &projection_types(),
            &function,
            &[],
            &vec![None; function.locals().len()],
            &vec![None; function.blocks().len()],
            &[],
            vec![],
            vec![
                ProjectedSemanticBlockV1 {
                    items: vec![guarded(), barrier()],
                },
                ProjectedSemanticBlockV1 { items: vec![] },
            ],
        )
        .unwrap();
        assert!(matches!(
            after_blocks[2].operations(),
            [ProductionRankedOperationV1::Access { .. }]
        ));
        assert!(matches!(
            after_blocks[4].operations(),
            [ProductionRankedOperationV1::Barrier { .. }]
        ));

        let (before_blocks, _, _) = build_ranked_cfg(
            &projection_types(),
            &function,
            &[],
            &vec![None; function.locals().len()],
            &vec![None; function.blocks().len()],
            &[],
            vec![],
            vec![
                ProjectedSemanticBlockV1 {
                    items: vec![barrier()],
                },
                ProjectedSemanticBlockV1 {
                    items: vec![guarded()],
                },
            ],
        )
        .unwrap();
        assert!(matches!(
            before_blocks[1].operations(),
            [ProductionRankedOperationV1::Barrier { .. }]
        ));
        assert!(matches!(
            before_blocks[3].operations(),
            [ProductionRankedOperationV1::Access { .. }]
        ));
    }

    #[test]
    fn shifted_disjoint_access_retains_overflow_and_extent_guards() {
        let invocation = ProductionRankedValueIdV1::new(0);
        let offset = ProductionRankedValueIdV1::new(1);
        let shifted = ProductionRankedValueIdV1::new(2);
        let upper = ProductionRankedValueIdV1::new(3);
        let view = ProductionRankedValueIdV1::new(4);
        let entry = vec![
            ProductionRankedOperationV1::InvocationIndex {
                result: invocation,
                dimension: 0,
                launch_extent: 64,
            },
            ProductionRankedOperationV1::IndexConstant {
                result: offset,
                value: 4,
            },
            ProductionRankedOperationV1::IndexBinary {
                result: shifted,
                kind: IndexBinaryKindAttr::Add,
                lhs: ProductionRankedValueV1::Local(invocation),
                rhs: ProductionRankedValueV1::Local(offset),
            },
            ProductionRankedOperationV1::IndexConstant {
                result: upper,
                value: u64::MAX - 3,
            },
            ProductionRankedOperationV1::ViewInSpace {
                result: view,
                element_width: 32,
                writable: true,
                shape: vec![DYNAMIC_EXTENT],
                dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
                memory_space: MemorySpaceAttr::Global,
                allocation_origin: 0,
                noalias_class: 0,
            },
        ];
        let guarded = GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view,
            indices: vec![ProductionRankedValueV1::Local(shifted)],
            checked_success: None,
            comparisons: vec![
                (
                    ProductionRankedValueV1::Local(invocation),
                    ProductionRankedValueV1::Local(upper),
                ),
                (
                    ProductionRankedValueV1::Local(shifted),
                    ProductionRankedValueV1::Argument(0),
                ),
            ],
            access: AccessKindAttr::Write,
            memory_space: MemorySpaceAttr::Global,
            source: SemanticSourceProvenanceV1::unavailable(),
            semantic_site: None,
            output_extent: None,
        };
        let (blocks, sources, ranked_ir) = single_guarded_cfg(entry, guarded);
        assert_eq!(blocks.len(), 6);
        assert_eq!(sources[0].block, 3);
        assert!(matches!(
            blocks[4].terminator(),
            ProductionRankedTerminatorV1::Trap
        ));
        assert!(ranked_ir.contains("kernel.br ^bb5"));
        assert!(ranked_ir.contains("^bb3:"));

        let kernel = ProductionRankedKernelV1::new("shifted_checked_access", 1, blocks).unwrap();
        let construction =
            ProductionConstructionV1::ranked_kernel("shifted_access_module", kernel).unwrap();
        let lowering =
            compile_ranked_kernel_for_lowering_v1(construction, ProductionSessionLimitsV1::default())
                .unwrap();
        assert!(lowering.bounds_report().is_clean());
        assert!(lowering.race_report().is_clean());
    }

    #[test]
    fn shared_option_dominance_scales_with_cfg_and_producer_count() {
        let (small_function, small_producers) = option_dominance_chain(16);
        let small = SemanticOptionDominanceV1::analyze(&small_function, &small_producers).unwrap();
        let (large_function, large_producers) = option_dominance_chain(64);
        let large = SemanticOptionDominanceV1::analyze(&large_function, &large_producers).unwrap();

        assert!(large.work_units() <= small.work_units() * 5);
        for producer in large_producers {
            assert!(large.availability(producer.option_local()).is_some());
        }
    }

    #[test]
    fn enum_payload_dominance_tracks_only_the_exact_variant_branch() {
        let carrier = SemanticLocalIdV1::from_index(1);
        let discriminator = SemanticLocalIdV1::from_index(2);
        let discriminator_place = SemanticPlaceV1::new(discriminator, vec![], SCALAR_TYPE).unwrap();
        let function = projection_function_with_locals(
            vec![
                block(
                    80,
                    vec![
                        enum_definition(carrier, 0),
                        enum_discriminant(carrier, discriminator),
                    ],
                    SemanticTerminatorKindV1::SwitchInt {
                        discriminant: SemanticOperandV1::Copy(discriminator_place),
                        targets: SemanticSwitchTargetsV1::new(
                            vec![SemanticSwitchTargetV1::new(
                                0,
                                cfg_edge(SemanticEdgeRoleV1::SwitchValue, 1),
                            )],
                            cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                        )
                        .unwrap(),
                    },
                ),
                block(
                    81,
                    vec![],
                    SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 3)),
                ),
                block(82, vec![], SemanticTerminatorKindV1::Return),
                block(83, vec![], SemanticTerminatorKindV1::Return),
            ],
            vec![
                local(20, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(21, ENUM_TYPE, SemanticLocalRoleV1::Temporary),
                local(22, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            ],
        );
        let dominance =
            SemanticEnumPayloadDominanceV1::analyze(&function, &projection_types_with_enum()).unwrap();
        let zero = dominance.availability(carrier, 0).unwrap();
        let one = dominance.availability(carrier, 1).unwrap();

        assert!(dominance.allows(zero, SemanticBlockIdV1::from_index(1)));
        assert!(dominance.allows(zero, SemanticBlockIdV1::from_index(3)));
        assert!(!dominance.allows(zero, SemanticBlockIdV1::from_index(2)));
        assert!(dominance.allows(one, SemanticBlockIdV1::from_index(2)));
        assert!(!dominance.grants_authority());
    }

    #[test]
    fn enum_payload_branch_with_an_alternate_predecessor_is_not_authenticated() {
        let carrier = SemanticLocalIdV1::from_index(1);
        let discriminator = SemanticLocalIdV1::from_index(2);
        let function = projection_function_with_locals(
            vec![
                block(
                    84,
                    vec![
                        enum_definition(carrier, 0),
                        enum_discriminant(carrier, discriminator),
                    ],
                    SemanticTerminatorKindV1::SwitchInt {
                        discriminant: SemanticOperandV1::Copy(
                            SemanticPlaceV1::new(discriminator, vec![], SCALAR_TYPE).unwrap(),
                        ),
                        targets: SemanticSwitchTargetsV1::new(
                            vec![SemanticSwitchTargetV1::new(
                                0,
                                cfg_edge(SemanticEdgeRoleV1::SwitchValue, 1),
                            )],
                            cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                        )
                        .unwrap(),
                    },
                ),
                block(85, vec![], SemanticTerminatorKindV1::Return),
                block(
                    86,
                    vec![],
                    SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 1)),
                ),
            ],
            vec![
                local(20, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(21, ENUM_TYPE, SemanticLocalRoleV1::Temporary),
                local(22, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            ],
        );
        let dominance =
            SemanticEnumPayloadDominanceV1::analyze(&function, &projection_types_with_enum()).unwrap();

        assert!(dominance.availability(carrier, 0).is_none());
        assert!(dominance.availability(carrier, 1).is_some());
    }

    #[test]
    fn multiply_defined_enum_carrier_cannot_authenticate_a_payload() {
        let carrier = SemanticLocalIdV1::from_index(1);
        let discriminator = SemanticLocalIdV1::from_index(2);
        let function = projection_function_with_locals(
            vec![
                block(
                    87,
                    vec![
                        enum_definition(carrier, 0),
                        enum_definition(carrier, 1),
                        enum_discriminant(carrier, discriminator),
                    ],
                    SemanticTerminatorKindV1::SwitchInt {
                        discriminant: SemanticOperandV1::Copy(
                            SemanticPlaceV1::new(discriminator, vec![], SCALAR_TYPE).unwrap(),
                        ),
                        targets: SemanticSwitchTargetsV1::new(
                            vec![SemanticSwitchTargetV1::new(
                                0,
                                cfg_edge(SemanticEdgeRoleV1::SwitchValue, 1),
                            )],
                            cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                        )
                        .unwrap(),
                    },
                ),
                block(88, vec![], SemanticTerminatorKindV1::Return),
                block(89, vec![], SemanticTerminatorKindV1::Return),
            ],
            vec![
                local(20, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(21, ENUM_TYPE, SemanticLocalRoleV1::Temporary),
                local(22, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            ],
        );
        let dominance =
            SemanticEnumPayloadDominanceV1::analyze(&function, &projection_types_with_enum()).unwrap();

        assert!(dominance.availability(carrier, 0).is_none());
        assert!(dominance.availability(carrier, 1).is_none());
    }

    #[test]
    fn private_address_space_five_remains_in_the_generic_memory_model() {
        assert_eq!(memory_space(5).unwrap(), MemorySpaceAttr::Private);
    }

    #[test]
    fn reassigned_option_discriminator_cannot_mint_payload_authority() {
        let option_local = SemanticLocalIdV1::from_index(3);
        let discriminator_local = SemanticLocalIdV1::from_index(2);
        let option_place = SemanticPlaceV1::new(option_local, vec![], POINTER_TYPE).unwrap();
        let discriminator_place =
            SemanticPlaceV1::new(discriminator_local, vec![], SCALAR_TYPE).unwrap();
        let discriminant = statement(SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            discriminator_place.clone(),
            SemanticRvalueV1::new(
                SCALAR_TYPE,
                SemanticRvalueKindV1::Discriminant(option_place.clone()),
            ),
        )));
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![],
            Some(SemanticCallDestinationV1::new(
                option_place,
                cfg_edge(SemanticEdgeRoleV1::CallReturn, 1),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let function = projection_function(vec![
            block(50, vec![], SemanticTerminatorKindV1::Call(call.clone())),
            block(
                51,
                vec![
                    discriminant,
                    statement(SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                        discriminator_place.clone(),
                        SemanticRvalueV1::new(SCALAR_TYPE, SemanticRvalueKindV1::Use(constant(1))),
                    ))),
                ],
                SemanticTerminatorKindV1::SwitchInt {
                    discriminant: SemanticOperandV1::Copy(discriminator_place),
                    targets: SemanticSwitchTargetsV1::new(
                        vec![SemanticSwitchTargetV1::new(
                            0,
                            cfg_edge(SemanticEdgeRoleV1::SwitchValue, 3),
                        )],
                        cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                    )
                    .unwrap(),
                },
            ),
            block(52, vec![], SemanticTerminatorKindV1::Return),
            block(53, vec![], SemanticTerminatorKindV1::Return),
        ]);
        let producer = SemanticOptionProducerV1::new(option_local, SemanticBlockIdV1::from_index(1));
        let error = SemanticOptionDominanceV1::analyze(&function, &[producer]).unwrap_err();

        assert_eq!(
            error.detail(),
            "an Option capability discriminator does not have one exact definition"
        );
    }
    #[test]
    fn unrelated_switch_cannot_authenticate_option_payload() {
        let option_local = SemanticLocalIdV1::from_index(3);
        let discriminator_local = SemanticLocalIdV1::from_index(2);
        let option_place = SemanticPlaceV1::new(option_local, vec![], POINTER_TYPE).unwrap();
        let discriminator_place =
            SemanticPlaceV1::new(discriminator_local, vec![], SCALAR_TYPE).unwrap();
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![],
            Some(SemanticCallDestinationV1::new(
                option_place.clone(),
                cfg_edge(SemanticEdgeRoleV1::CallReturn, 1),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let function = projection_function(vec![
            block(60, vec![], SemanticTerminatorKindV1::Call(call)),
            block(
                61,
                vec![statement(SemanticStatementKindV1::Assign(
                    SemanticAssignmentV1::new(
                        discriminator_place,
                        SemanticRvalueV1::new(
                            SCALAR_TYPE,
                            SemanticRvalueKindV1::Discriminant(option_place),
                        ),
                    ),
                ))],
                SemanticTerminatorKindV1::SwitchInt {
                    discriminant: constant(1),
                    targets: SemanticSwitchTargetsV1::new(
                        vec![SemanticSwitchTargetV1::new(
                            0,
                            cfg_edge(SemanticEdgeRoleV1::SwitchValue, 3),
                        )],
                        cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                    )
                    .unwrap(),
                },
            ),
            block(62, vec![], SemanticTerminatorKindV1::Return),
            block(63, vec![], SemanticTerminatorKindV1::Return),
        ]);
        let producer = SemanticOptionProducerV1::new(option_local, SemanticBlockIdV1::from_index(1));
        let error = SemanticOptionDominanceV1::analyze(&function, &[producer]).unwrap_err();

        assert_eq!(
            error.detail(),
            "an Option capability switch is not bound to its unique discriminator"
        );
    }

    #[test]
    fn alternate_predecessor_cannot_enter_an_authenticated_some_target() {
        let option_local = SemanticLocalIdV1::from_index(3);
        let discriminator_local = SemanticLocalIdV1::from_index(2);
        let option_place = SemanticPlaceV1::new(option_local, vec![], POINTER_TYPE).unwrap();
        let discriminator_place =
            SemanticPlaceV1::new(discriminator_local, vec![], SCALAR_TYPE).unwrap();
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![],
            Some(SemanticCallDestinationV1::new(
                option_place.clone(),
                cfg_edge(SemanticEdgeRoleV1::CallReturn, 1),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let function = projection_function(vec![
            block(70, vec![], SemanticTerminatorKindV1::Call(call)),
            block(
                71,
                vec![statement(SemanticStatementKindV1::Assign(
                    SemanticAssignmentV1::new(
                        discriminator_place.clone(),
                        SemanticRvalueV1::new(
                            SCALAR_TYPE,
                            SemanticRvalueKindV1::Discriminant(option_place),
                        ),
                    ),
                ))],
                SemanticTerminatorKindV1::SwitchInt {
                    discriminant: SemanticOperandV1::Copy(discriminator_place),
                    targets: SemanticSwitchTargetsV1::new(
                        vec![SemanticSwitchTargetV1::new(
                            0,
                            cfg_edge(SemanticEdgeRoleV1::SwitchValue, 3),
                        )],
                        cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                    )
                    .unwrap(),
                },
            ),
            block(72, vec![], SemanticTerminatorKindV1::Return),
            block(
                73,
                vec![],
                SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 2)),
            ),
        ]);
        let producer = SemanticOptionProducerV1::new(option_local, SemanticBlockIdV1::from_index(1));
        let error = SemanticOptionDominanceV1::analyze(&function, &[producer]).unwrap_err();

        assert_eq!(
            error.detail(),
            "an Option capability Some target is not uniquely controlled by its exact branch"
        );
    }

    #[test]
    fn option_payload_availability_excludes_the_none_merge() {
        let option_local = SemanticLocalIdV1::from_index(3);
        let discriminator_local = SemanticLocalIdV1::from_index(2);
        let option_place = SemanticPlaceV1::new(option_local, vec![], POINTER_TYPE).unwrap();
        let discriminator_place =
            SemanticPlaceV1::new(discriminator_local, vec![], SCALAR_TYPE).unwrap();
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![],
            Some(SemanticCallDestinationV1::new(
                option_place.clone(),
                cfg_edge(SemanticEdgeRoleV1::CallReturn, 1),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let function = projection_function(vec![
            block(40, vec![], SemanticTerminatorKindV1::Call(call.clone())),
            block(
                41,
                vec![statement(SemanticStatementKindV1::Assign(
                    SemanticAssignmentV1::new(
                        discriminator_place.clone(),
                        SemanticRvalueV1::new(
                            SCALAR_TYPE,
                            SemanticRvalueKindV1::Discriminant(option_place),
                        ),
                    ),
                ))],
                SemanticTerminatorKindV1::SwitchInt {
                    discriminant: SemanticOperandV1::Copy(discriminator_place),
                    targets: SemanticSwitchTargetsV1::new(
                        vec![
                            SemanticSwitchTargetV1::new(
                                0,
                                cfg_edge(SemanticEdgeRoleV1::SwitchValue, 3),
                            ),
                            SemanticSwitchTargetV1::new(
                                1,
                                cfg_edge(SemanticEdgeRoleV1::SwitchValue, 2),
                            ),
                        ],
                        cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 3),
                    )
                    .unwrap(),
                },
            ),
            block(
                42,
                vec![],
                SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 4)),
            ),
            block(
                43,
                vec![],
                SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 4)),
            ),
            block(44, vec![], SemanticTerminatorKindV1::Return),
        ]);
        let producer = SemanticOptionProducerV1::new(option_local, SemanticBlockIdV1::from_index(1));
        let dominance = SemanticOptionDominanceV1::analyze(&function, &[producer]).unwrap();
        let authority = dominance.availability(option_local).unwrap();

        assert!(dominance.allows(authority, SemanticBlockIdV1::from_index(2)));
        assert!(!dominance.allows(authority, SemanticBlockIdV1::from_index(1)));
        assert!(!dominance.allows(authority, SemanticBlockIdV1::from_index(3)));
        assert!(
            !dominance.allows(authority, SemanticBlockIdV1::from_index(4)),
            "the merge is reachable from None and must not inherit payload authority",
        );
    }

    #[test]
    fn capability_alias_worklist_processes_each_charged_edge_once() {
        let mut edges = vec![Vec::new(); 4];
        let mut charged = 0;
        for source in 0..3 {
            push_capability_edge(
                &mut edges,
                &mut charged,
                source,
                CapabilityEdgeV1 {
                    destination: source + 1,
                    use_block: 0,
                    kind: CapabilityEdgeKindV1::Alias,
                },
            )
            .unwrap();
        }
        let seed = ProjectedDisjointIndexV1 {
            value: ProductionRankedValueV1::Argument(0),
            mapping: SemanticDisjointIndexSpaceV1::Index1d,
            precondition: None,
            availability: None,
        };
        let mut values = vec![None; 4];
        let grid = vec![None; 4];
        let mut worklist = VecDeque::from([0]);
        values[0] = Some(seed);
        let mut processed = 0;
        while let Some(source) = worklist.pop_front() {
            for edge in &edges[source] {
                processed += 1;
                assert_eq!(edge.kind, CapabilityEdgeKindV1::Alias);
                assign_index_capability(
                    edge.destination,
                    values[source].unwrap(),
                    &mut values,
                    &grid,
                    &mut worklist,
                )
                .unwrap();
            }
        }

        assert_eq!(charged, 3);
        assert_eq!(processed, charged);
        assert!(values.iter().all(|value| *value == Some(seed)));
    }

    #[test]
    fn tiled_geometry_requires_finite_equal_products() {
        for ([lanes, rows, columns, elements], expected) in [
            ([4, 2, 4, 2], true),
            ([8, 2, 4, 1], true),
            ([64, 16, 16, 4], true),
            ([u64::MAX, 1, u64::MAX, 1], true),
            ([1, u64::MAX, 1, u64::MAX], true),
            ([u64::MAX, u64::MAX, 1, 1], true),
            ([(1_u64 << 63) - 1, 2, (1_u64 << 63) - 1, 2], true),
            ([1_u64 << 63, 2, 1_u64 << 63, 2], false),
            ([1_u64 << 63, 4, 1_u64 << 62, 2], false),
            ([u64::MAX, 2, u64::MAX, 2], false),
            ([2, u64::MAX, 2, u64::MAX], false),
            ([u64::MAX, 1, u64::MAX, 2], false),
            ([1, u64::MAX, 2, 1], false),
            ([4, 3, 4, 2], false),
            ([3, 2, 2, 1], false),
            ([0, 2, 4, 2], false),
            ([4, 0, 4, 2], false),
            ([4, 2, 0, 2], false),
            ([4, 2, 4, 0], false),
        ] {
            assert_eq!(
                tiled_2d_geometry_valid_v1(lanes, rows, columns, elements),
                expected,
                "geometry {:?}",
                [lanes, rows, columns, elements],
            );
        }
    }

    #[test]
    fn statement_projection_stops_at_the_ranked_operation_bound() {
        let function =
            projection_function(vec![block(50, vec![], SemanticTerminatorKindV1::Return)]);
        let types = projection_types();
        let semantic_statement =
            statement(SemanticStatementKindV1::Store(SemanticMemoryStoreV1::new(
                ranked_place(0),
                constant(1),
                SemanticVolatilityV1::NonVolatile,
                None,
            )));
        let mut operations = vec![
            ProductionRankedOperationV1::IndexConstant {
                result: ProductionRankedValueIdV1::new(0),
                value: 0,
            };
            MAX_PROJECTED_OPERATIONS_V1 - 2
        ];
        let original = operations.len();
        let mut sources = Vec::new();
        let mut projected_views = ProjectedViewsV1::new(function.locals().len(), None);
        let mut guarded_sites = Vec::new();
        let mut next_value = 0;
        let mut ranked_ir = String::new();
        let local_contracts = synthetic_local_contracts(&function);
        let error = project_statement_accesses(
            &types,
            &function,
            0,
            &[],
            &semantic_statement,
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
        .unwrap_err();
        assert!(matches!(
            error,
            ProductionRankedProjectionErrorV1::Unsupported(
                "a semantic statement projection exceeding the ranked operation limit"
            )
        ));
        assert_eq!(operations.len(), original);
        assert!(sources.is_empty());
        assert!(ranked_ir.is_empty());
    }

    #[test]
    fn constant_aliases_resolve_once_in_linear_time() {
        let definitions = [
            ConstantDefinitionV1::Direct(64),
            ConstantDefinitionV1::Alias(SemanticLocalIdV1::from_index(0)),
            ConstantDefinitionV1::Alias(SemanticLocalIdV1::from_index(1)),
        ];
        let mut states = [0; 3];
        let mut values = [None; 3];
        let mut path = Vec::new();
        assert_eq!(
            resolve_constant_iterative(2, &definitions, &mut states, &mut values, &mut path,),
            Some(64),
        );
        assert_eq!(values, [Some(64); 3]);
        assert_eq!(states, [2; 3]);
    }

    #[test]
    fn cyclic_or_multiply_defined_indices_are_not_constants() {
        let cycle = [
            ConstantDefinitionV1::Alias(SemanticLocalIdV1::from_index(1)),
            ConstantDefinitionV1::Alias(SemanticLocalIdV1::from_index(0)),
        ];
        let mut states = [0; 2];
        let mut values = [None; 2];
        let mut path = Vec::new();
        assert_eq!(
            resolve_constant_iterative(0, &cycle, &mut states, &mut values, &mut path),
            None,
        );

        let mut definitions = [ConstantDefinitionV1::Missing];
        record_constant_definition(
            &mut definitions,
            SemanticLocalIdV1::from_index(0),
            ConstantDefinitionV1::Direct(63),
        );
        record_constant_definition(
            &mut definitions,
            SemanticLocalIdV1::from_index(0),
            ConstantDefinitionV1::Direct(64),
        );
        assert!(matches!(definitions[0], ConstantDefinitionV1::Invalid));
    }

    #[test]
    fn source_label_is_explicit_when_unavailable() {
        assert_eq!(
            source_label(SemanticSourceProvenanceV1::unavailable()),
            "Rust source location unavailable",
        );
    }

    #[test]
    fn unresolved_callable_diagnostic_retains_available_source_provenance() {
        let origin = SemanticSourceOriginV1::new(
            SemanticSourceFileIdentityV1::from_sha256(bytes(0xab)),
            100,
            120,
            37,
            11,
            37,
            31,
        )
        .unwrap();
        let error = ProductionRankedProjectionErrorV1::UnresolvedCallableEffect {
            block: 19,
            source: Box::new(SemanticSourceProvenanceV1::new(Some(origin), Some(origin))),
            callee: 23,
            tail: false,
        };
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("semantic block bb19"));
        assert!(diagnostic.contains("Rust source abababababab:37:11"));
        assert!(diagnostic.contains("callable 23"));
    }

    fn tensor_operand(local: u32) -> SemanticOperandV1 {
        SemanticOperandV1::Copy(
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], SCALAR_TYPE)
                .unwrap(),
        )
    }

    fn tensor_payload(carrier: u32, variant: u32) -> SemanticOperandV1 {
        SemanticOperandV1::Move(
            SemanticPlaceV1::new(
                SemanticLocalIdV1::from_index(carrier),
                vec![
                    SemanticProjectionV1::new(
                        SemanticProjectionKindV1::Downcast(variant),
                        ENUM_TYPE,
                    )
                    .unwrap(),
                    SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), SCALAR_TYPE)
                        .unwrap(),
                ],
                SCALAR_TYPE,
            )
            .unwrap(),
        )
    }

    fn tensor_test_call() -> SemanticDirectCallV1 {
        SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![
                tensor_operand(0),
                tensor_operand(1),
                tensor_operand(2),
                tensor_operand(3),
            ],
            Some(SemanticCallDestinationV1::new(
                SemanticPlaceV1::new(SemanticLocalIdV1::from_index(4), vec![], SCALAR_TYPE)
                    .unwrap(),
                cfg_edge(SemanticEdgeRoleV1::CallReturn, 0),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap()
    }

    fn mfma_operand_contract(role: SemanticMfmaOperandRoleV1) -> SemanticMfmaOperandContractV1 {
        SemanticMfmaOperandContractV1 {
            role,
            profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
            register_distribution: SemanticMfmaRegisterDistributionV1::Tile16x16,
            wave_width: 64,
        }
    }

    fn mfma_accumulator_contract() -> SemanticMfmaAccumulatorContractV1 {
        SemanticMfmaAccumulatorContractV1 {
            profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
            distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
            wave_width: 64,
        }
    }

    fn gfx950_mfma_operand_contract(
        profile: SemanticMfmaProfileV1,
        role: SemanticMfmaOperandRoleV1,
    ) -> SemanticMfmaOperandContractV1 {
        SemanticMfmaOperandContractV1 {
            role,
            profile,
            register_distribution: SemanticMfmaRegisterDistributionV1::Gfx950M16N16K128,
            wave_width: 64,
        }
    }

    fn gfx950_mfma_accumulator_contract(
        profile: SemanticMfmaProfileV1,
    ) -> SemanticMfmaAccumulatorContractV1 {
        SemanticMfmaAccumulatorContractV1 {
            profile,
            distribution: SemanticMfmaAccumulatorDistributionV1::RowMajor,
            wave_width: 64,
        }
    }

    fn tensor_test_allocation() -> AllocationContractV1 {
        AllocationContractV1 {
            allocation_origin: 1,
            noalias_class: 1,
            writable: false,
            singleton_object: false,
        }
    }

    fn zero_filled_tensor_load_callable() -> SemanticCallableDeclV1 {
        compiler_intrinsic_callable(
            SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoadZeroFilledV2 {
                fragment: SCALAR_TYPE,
                view: SCALAR_TYPE,
                lane: SCALAR_TYPE,
                contract: mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
                storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
            },
        )
    }

    fn legacy_tensor_load_callable() -> SemanticCallableDeclV1 {
        compiler_intrinsic_callable(SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoad {
            option_fragment: ENUM_TYPE,
            fragment: SCALAR_TYPE,
            view: SCALAR_TYPE,
            lane: SCALAR_TYPE,
            contract: mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
            storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
        })
    }

    fn tensor_load_function(destination: Option<SemanticPlaceV1>) -> SemanticFunctionDeclV1 {
        let destination = destination.map(|place| {
            SemanticCallDestinationV1::new(place, cfg_edge(SemanticEdgeRoleV1::CallReturn, 0))
        });
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![
                tensor_operand(0),
                tensor_operand(1),
                tensor_operand(2),
                tensor_operand(3),
            ],
            destination,
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        projection_function_with_locals(
            vec![block(133, vec![], SemanticTerminatorKindV1::Call(call))],
            vec![
                local(133, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(134, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
                local(135, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
                local(136, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
                local(137, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            ],
        )
    }

    fn authenticated_tensor_load_state() -> ProjectedCapabilityStateV1 {
        HashMap::from([
            (
                0,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::View(
                    ProjectedMfmaViewV1 {
                        role: SemanticMfmaOperandRoleV1::A,
                        profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
                        storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                        allocation: tensor_test_allocation(),
                    },
                )),
            ),
            (
                1,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Lane {
                    root: 20,
                    wave_width: 64,
                }),
            ),
        ])
    }

    fn strided_read_view() -> ProjectedReadViewV1 {
        ProjectedReadViewV1 {
            root: 41,
            element: SCALAR_TYPE,
            allocation: AllocationContractV1 {
                allocation_origin: 3,
                noalias_class: 1,
                writable: false,
                singleton_object: false,
            },
            rows: ProjectedReadValueV1::Constant(7),
            columns: ProjectedReadValueV1::Local(SemanticLocalIdV1::from_index(2)),
        }
    }

    fn strided_read_call_function(destination: Option<SemanticPlaceV1>) -> SemanticFunctionDeclV1 {
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![
                tensor_operand(0),
                tensor_operand(1),
                tensor_operand(2),
                tensor_operand(3),
            ],
            destination.map(|place| {
                SemanticCallDestinationV1::new(place, cfg_edge(SemanticEdgeRoleV1::CallReturn, 0))
            }),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        projection_function_with_locals(
            vec![block(132, vec![], SemanticTerminatorKindV1::Call(call))],
            vec![
                local(132, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(133, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
                local(134, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
                local(135, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
                local(136, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            ],
        )
    }

    fn strided_read_callable() -> SemanticCallableDeclV1 {
        compiler_intrinsic_callable(
            SemanticCompilerIntrinsicOperationV1::StridedReadView2DLoadOr {
                view: SCALAR_TYPE,
                element: SCALAR_TYPE,
            },
        )
    }

    fn strided_read_constructor_function() -> SemanticFunctionDeclV1 {
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![
                tensor_operand(1),
                constant(0),
                constant(1),
                constant(3),
                constant(3),
            ],
            Some(SemanticCallDestinationV1::new(
                SemanticPlaceV1::new(SemanticLocalIdV1::from_index(2), vec![], ENUM_TYPE).unwrap(),
                cfg_edge(SemanticEdgeRoleV1::CallReturn, 0),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        projection_function_with_locals(
            vec![block(132, vec![], SemanticTerminatorKindV1::Call(call))],
            vec![
                local(132, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(133, SCALAR_TYPE, SemanticLocalRoleV1::Argument(0)),
                local(134, ENUM_TYPE, SemanticLocalRoleV1::Temporary),
            ],
        )
    }

    #[test]
    fn checked_shared_read_constructor_attenuates_writable_shared_abi_authority() {
        let function = strided_read_constructor_function();
        let callable = compiler_intrinsic_callable(
            SemanticCompilerIntrinsicOperationV1::StridedReadView2DFromSharedSlice {
                result: ENUM_TYPE,
                view: POINTER_TYPE,
                error: SCALAR_TYPE,
                element: SCALAR_TYPE,
            },
        );
        let source = AllocationContractV1 {
            allocation_origin: 1,
            noalias_class: 1,
            writable: true,
            singleton_object: false,
        };
        let mut state = HashMap::new();
        transfer_capability_terminator_v1(
            &[callable.clone()],
            &function,
            0,
            &mut state,
            &[None, Some(source), None],
            &[None; 3],
            &[],
            &HashMap::new(),
            true,
        )
        .unwrap();
        assert!(matches!(
            state.get(&2),
            Some(ProjectedCapabilityValueV1::Known(
                ProjectedCapabilityOriginV1::ReadViewResult(ProjectedReadViewV1 {
                    allocation: AllocationContractV1 {
                        allocation_origin: 1,
                        noalias_class: 1,
                        writable: false,
                        singleton_object: false,
                    },
                    ..
                })
            ))
        ));

        for noalias_class in [0, 2] {
            let mut rejected = HashMap::new();
            transfer_capability_terminator_v1(
                &[callable.clone()],
                &function,
                0,
                &mut rejected,
                &[
                    None,
                    Some(AllocationContractV1 {
                        noalias_class,
                        ..source
                    }),
                    None,
                ],
                &[None; 3],
                &[],
                &HashMap::new(),
                true,
            )
            .unwrap();
            assert_eq!(rejected.get(&2), Some(&ProjectedCapabilityValueV1::Invalid));
        }
    }

    #[test]
    fn strided_read_requires_exact_dominating_view_and_records_discarded_loads() {
        let function = strided_read_call_function(None);
        let view = strided_read_view();
        let mut state = HashMap::from([(
            0,
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::ReadView(view)),
        )]);
        let effects = transfer_capability_terminator_v1(
            &[strided_read_callable()],
            &function,
            0,
            &mut state,
            &[None; 5],
            &[None; 5],
            &[],
            &HashMap::new(),
            true,
        )
        .unwrap();
        assert_eq!(
            effects.read_view,
            Some(ProjectedReadViewAccessV1 {
                view,
                row: ProjectedReadValueV1::Local(SemanticLocalIdV1::from_index(1)),
                column: ProjectedReadValueV1::Local(SemanticLocalIdV1::from_index(2)),
            })
        );
        assert_eq!(
            state.get(&0),
            Some(&ProjectedCapabilityValueV1::Known(
                ProjectedCapabilityOriginV1::ReadView(view)
            ))
        );

        let mut invalid = HashMap::from([(0, ProjectedCapabilityValueV1::Invalid)]);
        assert!(matches!(
            transfer_capability_terminator_v1(
                &[strided_read_callable()],
                &function,
                0,
                &mut invalid,
                &[None; 5],
                &[None; 5],
                &[],
                &HashMap::new(),
                true,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a strided read without one dominating checked view payload and exact scalar operands"
            ))
        ));
    }

    #[test]
    fn strided_read_projects_rank_two_guarded_access_without_fabricated_indices() {
        let function =
            projection_function(vec![block(131, vec![], SemanticTerminatorKindV1::Return)]);
        let effect = ProjectedReadViewAccessV1 {
            view: strided_read_view(),
            row: ProjectedReadValueV1::Local(SemanticLocalIdV1::from_index(1)),
            column: ProjectedReadValueV1::Constant(4),
        };
        let mut arguments = vec![None; function.locals().len()];
        let mut next_argument = 1;
        let mut operations = Vec::new();
        let mut next_value = 0;
        let projected = project_strided_read_effects_v1(
            &projection_types(),
            &function,
            &[Some(effect)],
            &vec![None; function.locals().len()],
            &mut arguments,
            &mut next_argument,
            &mut operations,
            &mut next_value,
        )
        .unwrap();
        assert!(matches!(
            operations.as_slice(),
            [
                ProductionRankedOperationV1::IndexConstant { value: 7, .. },
                ProductionRankedOperationV1::ViewInSpace {
                    writable: false,
                    shape,
                    memory_space: MemorySpaceAttr::Global,
                    allocation_origin: 3,
                    ..
                },
                ProductionRankedOperationV1::IndexConstant { value: 4, .. }
            ] if shape == &[DYNAMIC_EXTENT, DYNAMIC_EXTENT]
        ));
        let access = projected[0].as_ref().unwrap();
        assert_eq!(access.access, AccessKindAttr::Read);
        assert_eq!(access.indices.len(), 2);
        assert_eq!(access.comparisons.len(), 2);
    }

    #[test]
    fn every_authenticated_global_fragment_load_emits_a_read_even_when_unused() {
        let function = tensor_load_function(Some(
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(4), vec![], SCALAR_TYPE).unwrap(),
        ));
        let mut state = authenticated_tensor_load_state();
        let effects = transfer_capability_terminator_v1(
            &[zero_filled_tensor_load_callable()],
            &function,
            0,
            &mut state,
            &[None; 5],
            &[],
            &[],
            &HashMap::new(),
            true,
        )
        .unwrap();

        assert_eq!(effects.global_read, Some(tensor_test_allocation()));
        assert!(effects.layout.is_none());
        assert!(matches!(
            state.get(&4),
            Some(ProjectedCapabilityValueV1::Known(
                ProjectedCapabilityOriginV1::Operand(_)
            ))
        ));
    }

    #[test]
    fn operand_a_and_b_reads_remain_distinct_effects_in_their_mir_call_blocks() {
        let function = projection_function(vec![
            block(138, vec![], SemanticTerminatorKindV1::Return),
            block(139, vec![], SemanticTerminatorKindV1::Return),
            block(140, vec![], SemanticTerminatorKindV1::Return),
        ]);
        let operand_a = tensor_test_allocation();
        let operand_b = AllocationContractV1 {
            allocation_origin: 2,
            noalias_class: 1,
            writable: false,
            singleton_object: false,
        };
        let effects = bind_capability_read_effects_to_call_blocks_v1(
            &function,
            &[Some(operand_a), None, Some(operand_b)],
        )
        .unwrap();

        assert_eq!(effects.len(), 3);
        assert_eq!(effects[0].map(|effect| effect.allocation), Some(operand_a));
        assert_eq!(effects[1], None);
        assert_eq!(effects[2].map(|effect| effect.allocation), Some(operand_b));
        assert_ne!(
            effects[0].unwrap().allocation.allocation_origin,
            effects[2].unwrap().allocation.allocation_origin
        );
        assert!(
            bind_capability_read_effects_to_call_blocks_v1(&function, &[Some(operand_a)]).is_err()
        );
    }

    #[test]
    fn global_fragment_loads_fail_closed_before_later_mfma_consumption() {
        let direct_destination =
            SemanticPlaceV1::new(SemanticLocalIdV1::from_index(4), vec![], SCALAR_TYPE).unwrap();
        let function = tensor_load_function(Some(direct_destination));
        let mut merged_state = authenticated_tensor_load_state();
        assert!(merge_capability_states_v1(&mut merged_state, &HashMap::new()).unwrap());
        let error = transfer_capability_terminator_v1(
            &[zero_filled_tensor_load_callable()],
            &function,
            0,
            &mut merged_state,
            &[None; 5],
            &[],
            &[],
            &HashMap::new(),
            true,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProductionRankedProjectionErrorV1::Incomplete(
                "a typed global fragment load without exact authenticated view, lane, allocation, and result provenance"
            )
        ));

        let mut invalid_lane = authenticated_tensor_load_state();
        invalid_lane.insert(1, ProjectedCapabilityValueV1::Invalid);
        assert!(matches!(
            transfer_capability_terminator_v1(
                &[zero_filled_tensor_load_callable()],
                &function,
                0,
                &mut invalid_lane,
                &[None; 5],
                &[],
                &[],
                &HashMap::new(),
                false,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a typed global fragment load without exact authenticated view, lane, allocation, and result provenance"
            ))
        ));
    }

    #[test]
    fn global_fragment_loads_cannot_discard_or_project_their_result() {
        let function = tensor_load_function(None);
        assert!(matches!(
            transfer_capability_terminator_v1(
                &[zero_filled_tensor_load_callable()],
                &function,
                0,
                &mut authenticated_tensor_load_state(),
                &[None; 5],
                &[],
                &[],
                &HashMap::new(),
                true,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a typed global fragment load without one direct local result"
            ))
        ));

        let projected = SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(4),
            vec![
                SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), SCALAR_TYPE).unwrap(),
            ],
            SCALAR_TYPE,
        )
        .unwrap();
        let function = tensor_load_function(Some(projected));
        assert!(matches!(
            transfer_capability_terminator_v1(
                &[zero_filled_tensor_load_callable()],
                &function,
                0,
                &mut authenticated_tensor_load_state(),
                &[None; 5],
                &[],
                &[],
                &HashMap::new(),
                true,
            ),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a typed global fragment load into a projected destination"
            ))
        ));
    }

    #[test]
    fn zero_filled_v2_load_is_a_direct_authenticated_operand() {
        let call = tensor_test_call();
        let contract = mfma_operand_contract(SemanticMfmaOperandRoleV1::A);
        let state = HashMap::from([
            (
                0,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::View(
                    ProjectedMfmaViewV1 {
                        role: SemanticMfmaOperandRoleV1::A,
                        profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
                        storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                        allocation: tensor_test_allocation(),
                    },
                )),
            ),
            (
                1,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Lane {
                    root: 20,
                    wave_width: 64,
                }),
            ),
        ]);

        assert!(matches!(
            project_tensor_load_origin_v1(
                &call,
                &state,
                SCALAR_TYPE,
                SCALAR_TYPE,
                contract,
                SemanticMfmaStorageLayoutV1::RowMajor,
            ),
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(_))
        ));
        assert_eq!(
            project_tensor_load_origin_v1(
                &call,
                &state,
                ARRAY_TYPE,
                SCALAR_TYPE,
                contract,
                SemanticMfmaStorageLayoutV1::RowMajor,
            ),
            ProjectedCapabilityValueV1::Invalid
        );
        assert_eq!(
            project_tensor_load_origin_v1(
                &call,
                &state,
                SCALAR_TYPE,
                SCALAR_TYPE,
                mfma_operand_contract(SemanticMfmaOperandRoleV1::B),
                SemanticMfmaStorageLayoutV1::RowMajor,
            ),
            ProjectedCapabilityValueV1::Invalid
        );
    }

    #[test]
    fn production_ranked_projection_rejects_the_retired_option_load_before_analysis() {
        assert!(matches!(
            reject_retired_production_intrinsics_v1(&[legacy_tensor_load_callable()]),
            Err(ProductionRankedProjectionErrorV1::Unsupported(
                "the retired Option-returning BF16 matrix load; use Bf16MatrixLoadZeroFilledV2"
            ))
        ));
        assert!(
            reject_retired_production_intrinsics_v1(&[zero_filled_tensor_load_callable()]).is_ok()
        );

        let function = tensor_load_function(None);
        assert!(matches!(
            transfer_capability_terminator_v1(
                &[legacy_tensor_load_callable()],
                &function,
                0,
                &mut authenticated_tensor_load_state(),
                &[None; 5],
                &[],
                &[],
                &HashMap::new(),
                false,
            ),
            Err(ProductionRankedProjectionErrorV1::Unsupported(
                "the retired Option-returning BF16 matrix load; use Bf16MatrixLoadZeroFilledV2"
            ))
        ));
    }

    fn authenticated_tensor_state(
        lhs_storage: SemanticMfmaStorageLayoutV1,
        rhs_storage: SemanticMfmaStorageLayoutV1,
    ) -> ProjectedCapabilityStateV1 {
        HashMap::from([
            (
                0,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::MatrixContext {
                    root: 10,
                }),
            ),
            (
                1,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(
                    ProjectedMfmaOperandV1 {
                        contract: mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
                        storage_layout: lhs_storage,
                        lane_root: 20,
                        allocation: tensor_test_allocation(),
                    },
                )),
            ),
            (
                2,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(
                    ProjectedMfmaOperandV1 {
                        contract: mfma_operand_contract(SemanticMfmaOperandRoleV1::B),
                        storage_layout: rhs_storage,
                        lane_root: 20,
                        allocation: tensor_test_allocation(),
                    },
                )),
            ),
            (
                3,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Accumulator(
                    ProjectedMfmaAccumulatorV1 {
                        contract: mfma_accumulator_contract(),
                        lane_root: 20,
                        value_root: 30,
                        flow_root: 30,
                    },
                )),
            ),
        ])
    }

    #[test]
    fn column_major_b_load_requires_its_exact_view_role_layout_and_wave() {
        let call = tensor_test_call();
        let contract = mfma_operand_contract(SemanticMfmaOperandRoleV1::B);
        let view = ProjectedMfmaViewV1 {
            role: SemanticMfmaOperandRoleV1::B,
            profile: SemanticMfmaProfileV1::Bf16F32M16N16K16,
            storage_layout: SemanticMfmaStorageLayoutV1::ColumnMajor,
            allocation: tensor_test_allocation(),
        };
        let mut state = authenticated_tensor_load_state();
        state.insert(
            0,
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::View(view)),
        );
        let actual = authenticate_tensor_load_v1(
            &call,
            &state,
            contract,
            SemanticMfmaStorageLayoutV1::ColumnMajor,
        )
        .unwrap();
        assert_eq!(
            actual.storage_layout,
            SemanticMfmaStorageLayoutV1::ColumnMajor
        );
        assert!(
            authenticate_tensor_load_v1(
                &call,
                &state,
                contract,
                SemanticMfmaStorageLayoutV1::RowMajor
            )
            .is_none()
        );
        assert!(
            authenticate_tensor_load_v1(
                &call,
                &state,
                mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
                SemanticMfmaStorageLayoutV1::ColumnMajor
            )
            .is_none()
        );
        state.insert(
            0,
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::View(
                ProjectedMfmaViewV1 {
                    storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                    ..view
                },
            )),
        );
        assert!(
            authenticate_tensor_load_v1(
                &call,
                &state,
                contract,
                SemanticMfmaStorageLayoutV1::ColumnMajor
            )
            .is_none()
        );
        state.insert(
            0,
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::View(view)),
        );
        state.insert(
            1,
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Lane {
                root: 20,
                wave_width: 32,
            }),
        );
        assert!(
            authenticate_tensor_load_v1(
                &call,
                &state,
                contract,
                SemanticMfmaStorageLayoutV1::ColumnMajor
            )
            .is_none()
        );
    }

    #[test]
    fn column_major_b_preserves_register_layout_but_has_distinct_source_identity() {
        let call = tensor_test_call();
        let authenticate = |lhs, rhs| {
            authenticate_tensor_instruction_v1(
                &call,
                &authenticated_tensor_state(lhs, rhs),
                mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
                mfma_operand_contract(SemanticMfmaOperandRoleV1::B),
                mfma_accumulator_contract(),
            )
        };
        let row = authenticate(
            SemanticMfmaStorageLayoutV1::RowMajor,
            SemanticMfmaStorageLayoutV1::RowMajor,
        )
        .unwrap();
        let column = authenticate(
            SemanticMfmaStorageLayoutV1::RowMajor,
            SemanticMfmaStorageLayoutV1::ColumnMajor,
        )
        .unwrap();
        assert_eq!(row.contract, column.contract);
        assert_ne!(
            tensor_operand_root_v1(row.rhs),
            tensor_operand_root_v1(column.rhs)
        );
        assert!(
            authenticate(
                SemanticMfmaStorageLayoutV1::ColumnMajor,
                SemanticMfmaStorageLayoutV1::RowMajor
            )
            .is_err()
        );
        let mut state = authenticated_tensor_state(
            SemanticMfmaStorageLayoutV1::RowMajor,
            SemanticMfmaStorageLayoutV1::ColumnMajor,
        );
        let Some(ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(rhs))) =
            state.get_mut(&2)
        else {
            panic!("checked B producer")
        };
        rhs.lane_root += 1;
        assert!(
            authenticate_tensor_instruction_v1(
                &call,
                &state,
                mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
                mfma_operand_contract(SemanticMfmaOperandRoleV1::B),
                mfma_accumulator_contract()
            )
            .is_err()
        );
    }

    #[test]
    fn authenticated_mfma_producers_derive_independent_storage_and_zero_fill() {
        let call = tensor_test_call();
        let state = authenticated_tensor_state(
            SemanticMfmaStorageLayoutV1::LdsXor4,
            SemanticMfmaStorageLayoutV1::RowMajor,
        );
        let authenticated = authenticate_tensor_instruction_v1(
            &call,
            &state,
            mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
            mfma_operand_contract(SemanticMfmaOperandRoleV1::B),
            mfma_accumulator_contract(),
        )
        .unwrap();
        let contract = authenticated.contract;

        assert_eq!(
            contract.a.lds_swizzle,
            fe2o3_kernel_ir::TensorLdsSwizzleV1::Xor4
        );
        assert_eq!(
            contract.b.lds_swizzle,
            fe2o3_kernel_ir::TensorLdsSwizzleV1::None
        );
        assert_eq!(
            contract.tail_mask,
            fe2o3_kernel_ir::TensorTailMaskV1::ZeroFilledPredicateInputs
        );
        assert_eq!(authenticated.context_root, 10);
        assert_eq!(authenticated.accumulator.value_root, 30);
        assert_ne!(
            tensor_operand_root_v1(authenticated.lhs),
            tensor_operand_root_v1(authenticated.rhs),
            "operand roles must remain part of otherwise identical allocation roots",
        );
        let mut substituted = authenticated.lhs;
        substituted.allocation.noalias_class += 1;
        assert_ne!(
            tensor_operand_root_v1(authenticated.lhs),
            tensor_operand_root_v1(substituted),
            "allocation provenance substitution must change the retained root",
        );
        let binding = tensor_site_binding_v1(authenticated, 40, 4).unwrap();
        assert_eq!(binding.argument_count(), 4);
        assert_eq!(
            binding.accumulator_root(),
            tensor_capability_root_v1(6, &[30])
        );
        assert_eq!(binding.result_root(), tensor_capability_root_v1(6, &[40]));
        assert_ne!(binding.accumulator_root(), binding.result_root());
        let chained = AuthenticatedTensorInstructionV1 {
            accumulator: ProjectedMfmaAccumulatorV1 {
                flow_root: 40,
                ..authenticated.accumulator
            },
            ..authenticated
        };
        let chained_binding = tensor_site_binding_v1(chained, 50, 4).unwrap();
        assert_eq!(
            chained_binding.accumulator_root(),
            binding.result_root(),
            "the next MFMA must consume the exact prior-result layout root",
        );
        assert_eq!(
            chained.accumulator.value_root, 30,
            "layout flow must not rewrite the semantic loop-carried value root",
        );
        assert!(tensor_site_binding_v1(authenticated, 40, 0).is_none());
    }

    #[test]
    fn authenticated_gfx950_fp4_mfma_derives_the_exact_ranked_tensor_layout() {
        let profile = SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128;
        let lhs_contract = gfx950_mfma_operand_contract(profile, SemanticMfmaOperandRoleV1::A);
        let rhs_contract = gfx950_mfma_operand_contract(profile, SemanticMfmaOperandRoleV1::B);
        let accumulator_contract = gfx950_mfma_accumulator_contract(profile);
        let state = HashMap::from([
            (
                0,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::MatrixContext {
                    root: 10,
                }),
            ),
            (
                1,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(
                    ProjectedMfmaOperandV1 {
                        contract: lhs_contract,
                        storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                        lane_root: 20,
                        allocation: tensor_test_allocation(),
                    },
                )),
            ),
            (
                2,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(
                    ProjectedMfmaOperandV1 {
                        contract: rhs_contract,
                        storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
                        lane_root: 20,
                        allocation: tensor_test_allocation(),
                    },
                )),
            ),
            (
                3,
                ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Accumulator(
                    ProjectedMfmaAccumulatorV1 {
                        contract: accumulator_contract,
                        lane_root: 20,
                        value_root: 30,
                        flow_root: 30,
                    },
                )),
            ),
        ]);

        let authenticated = authenticate_tensor_instruction_v1(
            &tensor_test_call(),
            &state,
            lhs_contract,
            rhs_contract,
            accumulator_contract,
        )
        .unwrap();
        let expected = fe2o3_kernel_ir::TensorLayoutContractV1::
            gfx950_scaled_mfma_fp4_e2m1_f32_m16n16k128_wave64();
        assert_eq!(authenticated.contract, expected);
        assert_eq!(
            authenticated.contract.profile,
            fe2o3_kernel_ir::TensorInstructionProfileV1::Gfx950ScaledMfmaFp4E2M1F32M16N16K128Wave64,
        );
        assert_eq!(
            authenticated.contract.a.packing,
            fe2o3_kernel_ir::TensorElementPackingV1::Fp4EightInI32,
        );
        assert_eq!(
            authenticated.contract.b.packing,
            fe2o3_kernel_ir::TensorElementPackingV1::Fp4EightInI32,
        );
    }

    #[path = "authenticated_load_or_v1_tests.rs"]
    mod authenticated_load_or_v1_tests;

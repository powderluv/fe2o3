fn fixture(mutation: Mutation) -> Result<ProductionSemanticSsaOwnerV1, String> {
    let source = SemanticSourceProvenanceV1::unavailable();
    let shift_counter = matches!(
        mutation,
        Mutation::ShiftCounter
            | Mutation::ShiftCounterZero
            | Mutation::ShiftCounterOverwide
            | Mutation::ShiftCounterZeroShift
            | Mutation::ShiftCounterWideShift
            | Mutation::ShiftCounterDynamicShift
            | Mutation::ShiftCounterReassign
            | Mutation::ShiftCounterInitReentry
            | Mutation::ShiftCounterBorrow
            | Mutation::ShiftCounterRawAddress
            | Mutation::ShiftCounterSigned
    );
    let raw = shift_counter
        || matches!(
            mutation,
            Mutation::RawShuffle
                | Mutation::RawShuffleTerminalDeadSnapshot
                | Mutation::RawShuffleHolderOrderedStorage
        );
    let id = SemanticTypeIdV1::from_index;
    let unit = id(0);
    let word = id(1);
    let float = id(2);
    let lane = id(3);
    let context = id(4);
    let snapshot = id(5);
    let snapshot_ref = id(6);
    let context_ref = id(7);
    let word_ref = id(8);
    let snapshot_mut_ref = id(9);
    let raw_snapshot_ptr = id(10);
    let raw_context_ptr = id(11);
    let types = fixture_types(shift_counter);
    let signed_word = id(12);
    let raw_word_pointer = id(13);
    let abi = fixture_abi;
    let callables = fixture_callables(raw);
    let place =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let assignment = |local, ty, value| {
        SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                place(local, ty),
                SemanticRvalueV1::new(ty, value),
            )),
        )
    };
    let constant = |ty, value| SemanticOperandV1::Constant(SemanticConstantV1::new(ty, value));
    let number = |n| {
        constant(
            word,
            SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(n, 4).unwrap()),
        )
    };
    let zst = || constant(unit, SemanticConstantValueV1::ZeroSized);
    let snapshot_value = |operand| {
        SemanticRvalueKindV1::Aggregate(
            SemanticAggregateRvalueV1::new(
                SemanticAggregateKindV1::Aggregate,
                vec![operand, zst()],
            )
            .unwrap(),
        )
    };
    let call = |callee, arguments, destination, ty, next| {
        SemanticTerminatorKindV1::Call(
            SemanticDirectCallV1::new_callable(
                SemanticCallableIdV1::from_index(callee),
                arguments,
                Some(SemanticCallDestinationV1::new(
                    place(destination, ty),
                    SemanticControlFlowEdgeV1::new(
                        SemanticEdgeRoleV1::CallReturn,
                        SemanticBlockIdV1::from_index(next),
                    ),
                )),
                SemanticUnwindActionV1::Unreachable,
            )
            .unwrap(),
        )
    };
    let block = |tag, statements, terminator| {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([tag; 32]),
            source,
            statements,
            SemanticTerminatorV1::new(source, terminator),
        )
        .unwrap()
    };
    let lane_field = SemanticOperandV1::Copy(
        SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(1),
            vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), word).unwrap()],
            word,
        )
        .unwrap(),
    );
    let mut statements = vec![
        assignment(
            3,
            snapshot,
            snapshot_value(if matches!(mutation, Mutation::WrongLaneField) {
                number(7)
            } else {
                lane_field
            }),
        ),
        assignment(
            4,
            snapshot_ref,
            if matches!(mutation, Mutation::RawBorrow) {
                SemanticRvalueKindV1::AddressOf {
                    mutability: SemanticMutabilityV1::Immutable,
                    place: place(3, snapshot),
                }
            } else {
                SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Shared,
                    place: place(3, snapshot),
                }
            },
        ),
        assignment(
            5,
            context_ref,
            SemanticRvalueKindV1::Borrow {
                kind: SemanticBorrowKindV1::Shared,
                place: place(2, context),
            },
        ),
    ];
    match mutation {
        Mutation::ReassignSnapshotReference => statements.push(assignment(
            4,
            snapshot_ref,
            SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(4, snapshot_ref))),
        )),
        Mutation::ReassignContextReference => statements.push(assignment(
            5,
            context_ref,
            SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(5, context_ref))),
        )),
        Mutation::RawSnapshotReference | Mutation::HolderAliasInitialLive => {
            statements.push(assignment(
                14,
                raw_snapshot_ptr,
                SemanticRvalueKindV1::Cast {
                    kind: SemanticCastKindV1::Pointer,
                    operand: SemanticOperandV1::Copy(place(4, snapshot_ref)),
                },
            ))
        }
        Mutation::RawContextReference => statements.push(assignment(
            15,
            raw_context_ptr,
            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Pointer,
                operand: SemanticOperandV1::Copy(place(5, context_ref)),
            },
        )),
        _ => {}
    }
    if raw {
        // The real defined-call input is an original SSA use, with distinct
        // real source values available for coherent typed graph mutations.
        statements.push(assignment(
            6,
            float,
            SemanticRvalueKindV1::Use(constant(
                float,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(0x3f800000, 4).unwrap()),
            )),
        ));
        statements.push(assignment(16, word, SemanticRvalueKindV1::Use(number(7))));
        statements.push(assignment(
            17,
            float,
            SemanticRvalueKindV1::Use(constant(
                float,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(0x40000000, 4).unwrap()),
            )),
        ));
    }
    if matches!(mutation, Mutation::UnissuedLane) {
        statements.insert(
            0,
            assignment(
                1,
                lane,
                SemanticRvalueKindV1::Aggregate(
                    SemanticAggregateRvalueV1::new(
                        SemanticAggregateKindV1::Aggregate,
                        vec![number(3)],
                    )
                    .unwrap(),
                ),
            ),
        );
    }
    if matches!(mutation, Mutation::UnissuedContext) {
        statements.insert(
            0,
            assignment(
                2,
                context,
                SemanticRvalueKindV1::Aggregate(
                    SemanticAggregateRvalueV1::new(SemanticAggregateKindV1::Aggregate, vec![])
                        .unwrap(),
                ),
            ),
        );
    }
    if matches!(
        mutation,
        Mutation::StorageLiveSnapshot | Mutation::DeadLiveSnapshot
    ) {
        if matches!(mutation, Mutation::DeadLiveSnapshot) {
            statements.push(SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(3)),
            ));
        }
        statements.push(SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(3)),
        ));
    }
    if matches!(
        mutation,
        Mutation::StorageLiveContext | Mutation::DeadLiveContext
    ) {
        if matches!(mutation, Mutation::DeadLiveContext) {
            statements.push(SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(2)),
            ));
        }
        statements.push(SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(2)),
        ));
    }
    if matches!(mutation, Mutation::StorageDeadSnapshot) {
        statements.push(SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(3)),
        ));
    }
    if matches!(mutation, Mutation::ProjectedExposure) {
        statements.push(assignment(
            12,
            word_ref,
            SemanticRvalueKindV1::Borrow {
                kind: SemanticBorrowKindV1::Shared,
                place: SemanticPlaceV1::new(
                    SemanticLocalIdV1::from_index(3),
                    vec![
                        SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), word)
                            .unwrap(),
                    ],
                    word,
                )
                .unwrap(),
            },
        ));
    }
    if matches!(mutation, Mutation::MutableExposure) {
        statements.push(assignment(
            13,
            snapshot_mut_ref,
            SemanticRvalueKindV1::Borrow {
                kind: SemanticBorrowKindV1::Mutable,
                place: place(3, snapshot),
            },
        ));
    }
    if matches!(mutation, Mutation::ReassignSnapshot) {
        statements.push(assignment(3, snapshot, snapshot_value(number(9))));
    }
    if matches!(mutation, Mutation::TerminalDeadSnapshotAliasCopy) {
        statements.push(assignment(
            11,
            snapshot_ref,
            SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(4, snapshot_ref))),
        ));
    }
    if matches!(mutation, Mutation::TerminalDeadSnapshotRawAlias) {
        statements.push(assignment(
            14,
            raw_snapshot_ptr,
            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Pointer,
                operand: SemanticOperandV1::Copy(place(4, snapshot_ref)),
            },
        ));
    }
    if matches!(mutation, Mutation::HolderDeadBeforeCall) {
        // The original reference holder dies after its real Borrow but
        // before the original helper's CallArgument(0) reads it.
        statements.push(SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(4)),
        ));
    }
    if matches!(
        mutation,
        Mutation::HolderLocalLive | Mutation::HolderLiveBackedge
    ) {
        let index = statements.iter().position(|statement| matches!(statement.kind(),
            SemanticStatementKindV1::Assign(assignment) if assignment.destination().local().index() == 4)).unwrap();
        statements.insert(
            index,
            SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(4)),
            ),
        );
    }
    let mut root_blocks = vec![
        // Keep both genuine compiler declarations in the root closure even
        // when the selected source value is independently overwritten.
        block(30, vec![], call(2, vec![], 1, lane, 1)),
        block(31, vec![], call(3, vec![], 2, context, 2)),
        block(
            32,
            statements,
            call(
                1,
                vec![
                    if matches!(mutation, Mutation::Rank) {
                        SemanticOperandV1::Copy(
                            SemanticPlaceV1::new(
                                SemanticLocalIdV1::from_index(4),
                                vec![
                                    SemanticProjectionV1::new(
                                        SemanticProjectionKindV1::Dereference,
                                        snapshot,
                                    )
                                    .unwrap(),
                                    SemanticProjectionV1::new(
                                        SemanticProjectionKindV1::Field(0),
                                        word,
                                    )
                                    .unwrap(),
                                ],
                                word,
                            )
                            .unwrap(),
                        )
                    } else {
                        SemanticOperandV1::Copy(place(4, snapshot_ref))
                    },
                    SemanticOperandV1::Copy(place(5, context_ref)),
                    if raw {
                        SemanticOperandV1::Copy(place(6, float))
                    } else {
                        constant(
                            float,
                            SemanticConstantValueV1::Scalar(
                                SemanticScalarValueV1::new(0, 4).unwrap(),
                            ),
                        )
                    },
                ],
                6,
                float,
                3,
            ),
        ),
        block(33, vec![], SemanticTerminatorKindV1::Return),
    ];
    if raw {
        root_blocks[3] = block(
            33,
            vec![],
            SemanticTerminatorKindV1::Goto(SemanticControlFlowEdgeV1::new(
                SemanticEdgeRoleV1::Goto,
                SemanticBlockIdV1::from_index(4),
            )),
        );
        root_blocks.push(block(34, vec![], SemanticTerminatorKindV1::Return));
    }
    if matches!(
        mutation,
        Mutation::TerminalDeadSnapshot
            | Mutation::TerminalDeadSnapshotAliasCopy
            | Mutation::TerminalDeadSnapshotRawAlias
            | Mutation::TerminalDeadSnapshotFormalRawAlias
            | Mutation::DeadSnapshotFutureRead
            | Mutation::DeadSnapshotBackedge
            | Mutation::DeadSnapshotUnreachableIncoming
            | Mutation::RawShuffleTerminalDeadSnapshot
    ) {
        let dead = SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(3)),
        );
        let terminal = if matches!(mutation, Mutation::RawShuffleTerminalDeadSnapshot) {
            4
        } else {
            3
        };
        let mut suffix = vec![dead];
        if matches!(
            mutation,
            Mutation::DeadSnapshotFutureRead | Mutation::TerminalDeadSnapshotAliasCopy
        ) {
            let read_holder = if matches!(mutation, Mutation::TerminalDeadSnapshotAliasCopy) {
                11
            } else {
                4
            };
            suffix.push(assignment(
                16,
                word,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(
                    SemanticPlaceV1::new(
                        SemanticLocalIdV1::from_index(read_holder),
                        vec![
                            SemanticProjectionV1::new(
                                SemanticProjectionKindV1::Dereference,
                                snapshot,
                            )
                            .unwrap(),
                            SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), word)
                                .unwrap(),
                        ],
                        word,
                    )
                    .unwrap(),
                )),
            ));
        }
        let tail = if matches!(mutation, Mutation::DeadSnapshotBackedge) {
            SemanticTerminatorKindV1::Goto(SemanticControlFlowEdgeV1::new(
                SemanticEdgeRoleV1::Goto,
                SemanticBlockIdV1::from_index(2),
            ))
        } else {
            SemanticTerminatorKindV1::Return
        };
        root_blocks[terminal] = block(33 + terminal as u8, suffix, tail);
        if matches!(mutation, Mutation::TerminalDeadSnapshotAliasCopy) {
            root_blocks[terminal] = block(
                36,
                vec![SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(3)),
                )],
                call(
                    1,
                    vec![
                        SemanticOperandV1::Copy(place(11, snapshot_ref)),
                        SemanticOperandV1::Copy(place(5, context_ref)),
                        constant(
                            float,
                            SemanticConstantValueV1::Scalar(
                                SemanticScalarValueV1::new(0, 4).unwrap(),
                            ),
                        ),
                    ],
                    6,
                    float,
                    4,
                ),
            );
            root_blocks.push(block(39, vec![], SemanticTerminatorKindV1::Return));
        }
        if matches!(mutation, Mutation::DeadSnapshotUnreachableIncoming) {
            root_blocks.push(block(
                37,
                vec![],
                call(
                    1,
                    vec![
                        SemanticOperandV1::Copy(place(4, snapshot_ref)),
                        SemanticOperandV1::Copy(place(5, context_ref)),
                        constant(
                            float,
                            SemanticConstantValueV1::Scalar(
                                SemanticScalarValueV1::new(0, 4).unwrap(),
                            ),
                        ),
                    ],
                    6,
                    float,
                    5,
                ),
            ));
            root_blocks.push(block(38, vec![], SemanticTerminatorKindV1::Return));
        }
    }
    if matches!(
        mutation,
        Mutation::HolderEntryLive
            | Mutation::HolderOrderedStorage
            | Mutation::RawShuffleHolderOrderedStorage
            | Mutation::HolderRestart
            | Mutation::HolderAliasInitialLive
    ) {
        root_blocks[0] = block(
            30,
            vec![
                SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(4)),
                ),
                SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(5)),
                ),
            ],
            call(2, vec![], 1, lane, 1),
        );
    }
    if matches!(
        mutation,
        Mutation::HolderOrderedStorage | Mutation::RawShuffleHolderOrderedStorage
    ) {
        let terminal = if matches!(mutation, Mutation::RawShuffleHolderOrderedStorage) {
            4
        } else {
            3
        };
        root_blocks[terminal] = block(
            81,
            vec![
                SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(4)),
                ),
                SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(5)),
                ),
            ],
            SemanticTerminatorKindV1::Return,
        );
    }
    if matches!(mutation, Mutation::HolderLateLive | Mutation::HolderRestart) {
        root_blocks[3] = block(
            81,
            vec![SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(4)),
            )],
            SemanticTerminatorKindV1::Return,
        );
    }
    if matches!(mutation, Mutation::HolderUnreachableLive) {
        root_blocks.push(block(
            82,
            vec![SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(4)),
            )],
            SemanticTerminatorKindV1::Return,
        ));
    }
    if matches!(
        mutation,
        Mutation::HolderLiveBackedge | Mutation::HolderDeadBackedge
    ) {
        let statements = if matches!(mutation, Mutation::HolderDeadBackedge) {
            vec![SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(4)),
            )]
        } else {
            vec![]
        };
        root_blocks[3] = block(
            81,
            statements,
            SemanticTerminatorKindV1::Goto(SemanticControlFlowEdgeV1::new(
                SemanticEdgeRoleV1::Goto,
                SemanticBlockIdV1::from_index(2),
            )),
        );
    }
    if matches!(mutation, Mutation::InitialStorageLive) {
        root_blocks[0] = block(
            30,
            vec![
                SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(2)),
                ),
                SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(3)),
                ),
            ],
            call(2, vec![], 1, lane, 1),
        );
    }
    if matches!(mutation, Mutation::ForeignIncoming) {
        // A genuine first call must not excuse a second unissued context.
        root_blocks[3] = block(
            33,
            vec![
                assignment(
                    7,
                    context,
                    SemanticRvalueKindV1::Aggregate(
                        SemanticAggregateRvalueV1::new(SemanticAggregateKindV1::Aggregate, vec![])
                            .unwrap(),
                    ),
                ),
                assignment(
                    8,
                    context_ref,
                    SemanticRvalueKindV1::Borrow {
                        kind: SemanticBorrowKindV1::Shared,
                        place: place(7, context),
                    },
                ),
            ],
            call(
                1,
                vec![
                    SemanticOperandV1::Copy(place(4, snapshot_ref)),
                    SemanticOperandV1::Copy(place(8, context_ref)),
                    constant(
                        float,
                        SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(0, 4).unwrap()),
                    ),
                ],
                6,
                float,
                4,
            ),
        );
        root_blocks.push(block(34, vec![], SemanticTerminatorKindV1::Return));
    }
    if matches!(mutation, Mutation::SameIncoming) {
        root_blocks[3] = block(
            33,
            vec![],
            call(
                1,
                vec![
                    SemanticOperandV1::Copy(place(4, snapshot_ref)),
                    SemanticOperandV1::Copy(place(5, context_ref)),
                    constant(
                        float,
                        SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(0, 4).unwrap()),
                    ),
                ],
                6,
                float,
                4,
            ),
        );
        root_blocks.push(block(34, vec![], SemanticTerminatorKindV1::Return));
    }
    if matches!(
        mutation,
        Mutation::ForeignIncomingLane | Mutation::DistinctIssuedLane
    ) {
        let new_lane = SemanticOperandV1::Copy(
            SemanticPlaceV1::new(
                SemanticLocalIdV1::from_index(9),
                vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), word).unwrap()],
                word,
            )
            .unwrap(),
        );
        let mut second_statements = vec![
            assignment(10, snapshot, snapshot_value(new_lane)),
            assignment(
                11,
                snapshot_ref,
                SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Shared,
                    place: place(10, snapshot),
                },
            ),
        ];
        let next = if matches!(mutation, Mutation::DistinctIssuedLane) {
            5
        } else {
            4
        };
        let second_call = call(
            1,
            vec![
                SemanticOperandV1::Copy(place(11, snapshot_ref)),
                SemanticOperandV1::Copy(place(5, context_ref)),
                constant(
                    float,
                    SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(0, 4).unwrap()),
                ),
            ],
            6,
            float,
            next,
        );
        if matches!(mutation, Mutation::DistinctIssuedLane) {
            // A real second issuer is still a distinct original source SSA key.
            root_blocks[3] = block(33, vec![], call(2, vec![], 9, lane, 4));
            root_blocks.push(block(34, second_statements, second_call));
            root_blocks.push(block(35, vec![], SemanticTerminatorKindV1::Return));
        } else {
            second_statements.insert(
                0,
                assignment(
                    9,
                    lane,
                    SemanticRvalueKindV1::Aggregate(
                        SemanticAggregateRvalueV1::new(
                            SemanticAggregateKindV1::Aggregate,
                            vec![number(7)],
                        )
                        .unwrap(),
                    ),
                ),
            );
            root_blocks[3] = block(33, second_statements, second_call);
            root_blocks.push(block(34, vec![], SemanticTerminatorKindV1::Return));
        }
    }
    if matches!(mutation, Mutation::ContextCallOverwrite) {
        root_blocks[3] = block(33, vec![], call(3, vec![], 2, context, 4));
        root_blocks.push(block(34, vec![], SemanticTerminatorKindV1::Return));
    }
    let function = fixture_function;
    let dimensions = SemanticWorkgroupDimensionsV1::new([64, 1, 1]).unwrap();
    let root = function(
        40,
        true,
        vec![
            (unit, SemanticLocalRoleV1::Return),
            (lane, SemanticLocalRoleV1::Temporary),
            (context, SemanticLocalRoleV1::Temporary),
            (snapshot, SemanticLocalRoleV1::Temporary),
            (snapshot_ref, SemanticLocalRoleV1::Temporary),
            (context_ref, SemanticLocalRoleV1::Temporary),
            (float, SemanticLocalRoleV1::Temporary),
            (context, SemanticLocalRoleV1::Temporary),
            (context_ref, SemanticLocalRoleV1::Temporary),
            (lane, SemanticLocalRoleV1::Temporary),
            (snapshot, SemanticLocalRoleV1::Temporary),
            (snapshot_ref, SemanticLocalRoleV1::Temporary),
            (word_ref, SemanticLocalRoleV1::Temporary),
            (snapshot_mut_ref, SemanticLocalRoleV1::Temporary),
            (raw_snapshot_ptr, SemanticLocalRoleV1::Temporary),
            (raw_context_ptr, SemanticLocalRoleV1::Temporary),
            (word, SemanticLocalRoleV1::Temporary),
            (float, SemanticLocalRoleV1::Temporary),
        ],
        abi(40, vec![], unit, vec![], true),
        root_blocks,
    )
    .with_kernel_entry(SemanticKernelEntryV1::new(
        SemanticLinkSymbolV1::new(b"synthetic_reference_relation".to_vec()).unwrap(),
        SemanticKernelBindingIdentityV1::from_sha256([50; 32]),
        SemanticKernelSourceContractV1::new(
            Some(
                SemanticKernelLaunchBoundsV1::new(Some(dimensions), Some(dimensions), None)
                    .unwrap(),
            ),
            None,
            None,
        )
        .unwrap(),
    ));
    let mut helper_statements = vec![assignment(
        0,
        float,
        SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(3, float))),
    )];
    match mutation {
        Mutation::OverwriteSnapshotFormal => helper_statements.insert(
            0,
            assignment(
                1,
                snapshot_ref,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(1, snapshot_ref))),
            ),
        ),
        Mutation::OverwriteContextFormal => helper_statements.insert(
            0,
            assignment(
                2,
                context_ref,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(2, context_ref))),
            ),
        ),
        Mutation::RawSnapshotFormal | Mutation::TerminalDeadSnapshotFormalRawAlias => {
            helper_statements.insert(
                0,
                assignment(
                    4,
                    raw_snapshot_ptr,
                    SemanticRvalueKindV1::Cast {
                        kind: SemanticCastKindV1::Pointer,
                        operand: SemanticOperandV1::Copy(place(1, snapshot_ref)),
                    },
                ),
            )
        }
        Mutation::RawContextFormal => helper_statements.insert(
            0,
            assignment(
                5,
                raw_context_ptr,
                SemanticRvalueKindV1::Cast {
                    kind: SemanticCastKindV1::Pointer,
                    operand: SemanticOperandV1::Copy(place(2, context_ref)),
                },
            ),
        ),
        _ => {}
    }
    if matches!(
        mutation,
        Mutation::HolderFormalLive | Mutation::HolderFormalDead
    ) {
        let kind = if matches!(mutation, Mutation::HolderFormalLive) {
            SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(1))
        } else {
            SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(1))
        };
        helper_statements.insert(0, SemanticStatementV1::new(source, kind));
    }
    let counter_ty = if matches!(mutation, Mutation::ShiftCounterSigned) {
        signed_word
    } else {
        word
    };
    let mut helper_locals = vec![
        (float, SemanticLocalRoleV1::Return),
        (
            if matches!(mutation, Mutation::Rank) {
                word
            } else {
                snapshot_ref
            },
            SemanticLocalRoleV1::Argument(0),
        ),
        (context_ref, SemanticLocalRoleV1::Argument(1)),
        (float, SemanticLocalRoleV1::Argument(2)),
        (raw_snapshot_ptr, SemanticLocalRoleV1::Temporary),
        (raw_context_ptr, SemanticLocalRoleV1::Temporary),
        (word, SemanticLocalRoleV1::Temporary),
    ];
    if shift_counter {
        helper_locals.extend([
            (counter_ty, SemanticLocalRoleV1::Temporary),
            (word, SemanticLocalRoleV1::Temporary),
            (word, SemanticLocalRoleV1::Temporary),
            (word_ref, SemanticLocalRoleV1::Temporary),
            (raw_word_pointer, SemanticLocalRoleV1::Temporary),
        ]);
        if matches!(mutation, Mutation::ShiftCounterDynamicShift) {
            helper_locals.push((signed_word, SemanticLocalRoleV1::Temporary));
        }
    }
    let helper_blocks = if shift_counter {
        let edge = |role, target| {
            SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target))
        };
        let init = if matches!(mutation, Mutation::ShiftCounterZero) {
            0
        } else if matches!(mutation, Mutation::ShiftCounterOverwide) {
            64
        } else {
            32
        };
        let literal = |ty, value| {
            constant(
                ty,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, 4).unwrap()),
            )
        };
        let mut initialize = vec![
            SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::StorageLive(SemanticLocalIdV1::from_index(7)),
            ),
            assignment(
                7,
                counter_ty,
                SemanticRvalueKindV1::Use(literal(counter_ty, init)),
            ),
        ];
        if matches!(mutation, Mutation::ShiftCounterBorrow) {
            initialize.push(assignment(
                10,
                word_ref,
                SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Shared,
                    place: place(7, word),
                },
            ));
        }
        if matches!(mutation, Mutation::ShiftCounterRawAddress) {
            initialize.push(assignment(
                11,
                raw_word_pointer,
                SemanticRvalueKindV1::AddressOf {
                    mutability: SemanticMutabilityV1::Mutable,
                    place: place(7, word),
                },
            ));
        }
        let shift = if matches!(mutation, Mutation::ShiftCounterZeroShift) {
            0
        } else if matches!(mutation, Mutation::ShiftCounterWideShift) {
            32
        } else {
            1
        };
        let right = if matches!(mutation, Mutation::ShiftCounterDynamicShift) {
            SemanticOperandV1::Copy(place(12, signed_word))
        } else {
            literal(signed_word, shift)
        };
        let mut step = vec![assignment(
            7,
            counter_ty,
            SemanticRvalueKindV1::Binary {
                operation: SemanticBinaryOpV1::ShiftRight,
                left: SemanticOperandV1::Copy(place(7, counter_ty)),
                right,
            },
        )];
        if matches!(mutation, Mutation::ShiftCounterDynamicShift) {
            step.insert(
                0,
                assignment(
                    12,
                    signed_word,
                    SemanticRvalueKindV1::Cast {
                        kind: SemanticCastKindV1::Integer,
                        operand: SemanticOperandV1::Copy(place(9, word)),
                    },
                ),
            );
        }
        if matches!(mutation, Mutation::ShiftCounterReassign) {
            step.push(assignment(
                7,
                counter_ty,
                SemanticRvalueKindV1::Use(literal(counter_ty, 64)),
            ));
        }
        vec![
            block(
                83,
                vec![assignment(
                    0,
                    float,
                    SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(3, float))),
                )],
                SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 5)),
            ),
            block(
                84,
                vec![],
                SemanticTerminatorKindV1::SwitchInt {
                    discriminant: SemanticOperandV1::Copy(place(7, counter_ty)),
                    targets: SemanticSwitchTargetsV1::new(
                        vec![SemanticSwitchTargetV1::new(
                            0,
                            edge(SemanticEdgeRoleV1::SwitchValue, 4),
                        )],
                        edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                    )
                    .unwrap(),
                },
            ),
            block(
                85,
                vec![
                    assignment(
                        6,
                        word,
                        SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(
                            SemanticPlaceV1::new(
                                SemanticLocalIdV1::from_index(1),
                                vec![
                                    SemanticProjectionV1::new(
                                        SemanticProjectionKindV1::Dereference,
                                        snapshot,
                                    )
                                    .unwrap(),
                                    SemanticProjectionV1::new(
                                        SemanticProjectionKindV1::Field(0),
                                        word,
                                    )
                                    .unwrap(),
                                ],
                                word,
                            )
                            .unwrap(),
                        )),
                    ),
                    assignment(
                        9,
                        word,
                        if matches!(mutation, Mutation::ShiftCounterSigned) {
                            SemanticRvalueKindV1::Cast {
                                kind: SemanticCastKindV1::Integer,
                                operand: SemanticOperandV1::Copy(place(7, signed_word)),
                            }
                        } else {
                            SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(7, word)))
                        },
                    ),
                    assignment(
                        8,
                        word,
                        SemanticRvalueKindV1::Binary {
                            operation: SemanticBinaryOpV1::BitXor,
                            left: SemanticOperandV1::Copy(place(6, word)),
                            right: SemanticOperandV1::Copy(place(9, word)),
                        },
                    ),
                ],
                call(
                    4,
                    vec![
                        SemanticOperandV1::Copy(place(2, context_ref)),
                        SemanticOperandV1::Copy(place(0, float)),
                        SemanticOperandV1::Copy(place(8, word)),
                    ],
                    0,
                    float,
                    3,
                ),
            ),
            block(
                86,
                step,
                SemanticTerminatorKindV1::Goto(edge(
                    SemanticEdgeRoleV1::Goto,
                    if matches!(mutation, Mutation::ShiftCounterInitReentry) {
                        5
                    } else {
                        1
                    },
                )),
            ),
            block(
                87,
                vec![SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(7)),
                )],
                SemanticTerminatorKindV1::Return,
            ),
            block(
                88,
                initialize,
                SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1)),
            ),
        ]
    } else if raw {
        vec![
            block(
                70,
                vec![assignment(
                    6,
                    word,
                    SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(
                        SemanticPlaceV1::new(
                            SemanticLocalIdV1::from_index(1),
                            vec![
                                SemanticProjectionV1::new(
                                    SemanticProjectionKindV1::Dereference,
                                    snapshot,
                                )
                                .unwrap(),
                                SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), word)
                                    .unwrap(),
                            ],
                            word,
                        )
                        .unwrap(),
                    )),
                )],
                call(
                    4,
                    vec![
                        SemanticOperandV1::Copy(place(2, context_ref)),
                        SemanticOperandV1::Copy(place(3, float)),
                        SemanticOperandV1::Copy(place(6, word)),
                    ],
                    0,
                    float,
                    1,
                ),
            ),
            block(71, vec![], SemanticTerminatorKindV1::Return),
        ]
    } else {
        vec![block(
            70,
            helper_statements,
            SemanticTerminatorKindV1::Return,
        )]
    };
    let helper = function(
        60,
        false,
        helper_locals,
        abi(
            60,
            vec![
                if matches!(mutation, Mutation::Rank) {
                    word
                } else {
                    snapshot_ref
                },
                context_ref,
                float,
            ],
            float,
            vec![
                if matches!(mutation, Mutation::Rank) {
                    SemanticSourceArgumentOwnershipV1::ByValue
                } else {
                    SemanticSourceArgumentOwnershipV1::SharedBorrow
                },
                SemanticSourceArgumentOwnershipV1::SharedBorrow,
                SemanticSourceArgumentOwnershipV1::ByValue,
            ],
            false,
        ),
        helper_blocks,
    );
    let admitted = InertSemanticMirRequestV1::new_with_callables(
        SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256([250; 32])),
        types,
        vec![],
        vec![],
        vec![],
        vec![root, helper],
        callables,
        vec![SemanticFunctionIdV1::from_index(0)],
    )
    .map_err(|e| format!("{e:?}"))?
    .admit_current_production(SemanticMirLimitsV1::default())
    .map_err(|e| format!("{e:?}"))?;
    let semantic =
        ProductionSemanticMirOwnerV1::try_new(admitted, ProductionSemanticMirLimitsV1::default())
            .map_err(|e| format!("{e:?}"))?;
    let mut owner =
        ProductionSemanticSsaOwnerV1::try_new(semantic, ProductionSemanticSsaLimitsV1::default())
            .map_err(|e| format!("{e:?}"))?;
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    owner
        .try_capture_occurrences_with_budget_v1(&mut budget)
        .map_err(|e| format!("{e:?}"))?;
    Ok(owner)
}

#[test]
fn issued_lane_field_is_plain_arithmetic_and_refuses_foreign_shapes() {
    let owner = fixture(Mutation::None).expect("admitted source owner");
    let semantic = owner.source_semantic();
    let types = semantic.types();
    let carrier = SemanticTypeIdV1::from_index(3);
    let word = SemanticTypeIdV1::from_index(1);
    let float = SemanticTypeIdV1::from_index(2);
    let snapshot = SemanticTypeIdV1::from_index(5);
    let projection = SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), word).unwrap();
    let lane = SemanticValueBindingV1::WaveLane {
        value: ValueId(42),
        wave: SemanticCurrentWaveV1::new(64),
    };
    let mut work = Work::new(1_000_000);
    let mut budget = Budget::new(&mut work, 1_000_000);
    let issued = super::super::compiler_issued_ssa_bindings_v1(
        types,
        semantic.callables(),
        &semantic.functions()[0],
        SemanticFunctionIdV1::from_index(0),
        None,
        Some(&mut budget),
    )
    .unwrap();
    let arithmetic = super::super::wave_lane_arithmetic_field_v1(
        types,
        &issued,
        carrier,
        &projection,
        &lane,
        &mut budget,
    )
    .unwrap()
    .expect("exact source lane field");
    assert!(matches!(
        arithmetic,
        SemanticValueBindingV1::Value {
            id: ValueId(42),
            ty: Type::Scalar(ScalarType::U32)
        }
    ));
    assert!(
        super::super::require_current_wave_lane(
            SemanticBlockIdV1::from_index(0),
            arithmetic,
            64,
            "arithmetic observation is not lane authority"
        )
        .is_err()
    );
    for fault in 0..9 {
        let mut bindings = issued.clone();
        let mut source_types = types.to_vec();
        let mut source_carrier = carrier;
        let mut field = projection.clone();
        let mut source_binding = lane.clone();
        match fault {
            0 => bindings.clear(),
            1 => {
                bindings.insert(carrier, SemanticPromotedBindingV1::Ordinary);
            }
            2 => {
                bindings.insert(
                    carrier,
                    SemanticPromotedBindingV1::WaveLane { wave_width: 32 },
                );
            }
            3 => {
                source_binding = SemanticValueBindingV1::WaveLane {
                    value: ValueId(42),
                    wave: SemanticCurrentWaveV1::new(32),
                }
            }
            4 => source_carrier = snapshot,
            5 => {
                field = SemanticProjectionV1::new(SemanticProjectionKindV1::Field(1), word).unwrap()
            }
            6 => {
                field =
                    SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), float).unwrap()
            }
            7 => {
                source_binding = SemanticValueBindingV1::Value {
                    id: ValueId(42),
                    ty: Type::Scalar(ScalarType::U32),
                }
            }
            8 => source_types[word.index() as usize] = types[float.index() as usize].clone(),
            _ => unreachable!(),
        }
        assert!(
            super::super::wave_lane_arithmetic_field_v1(
                &source_types,
                &bindings,
                source_carrier,
                &field,
                &source_binding,
                &mut budget
            )
            .unwrap()
            .is_none(),
            "field mutation {fault}"
        );
    }
    let mut denied_work = Work::new(0);
    let mut denied = Budget::new(&mut denied_work, 0);
    assert!(
        super::super::wave_lane_arithmetic_field_v1(
            types,
            &issued,
            carrier,
            &projection,
            &lane,
            &mut denied
        )
        .is_err()
    );
    assert_eq!(denied.failed_work(), Some(4));
    assert_eq!(denied.storage(), 0);
}

#[test]
fn snapshot_storage_end_requires_a_terminal_suffix_after_every_original_use() {
    for mutation in [
        Mutation::TerminalDeadSnapshot,
        Mutation::RawShuffleTerminalDeadSnapshot,
    ] {
        let owner = fixture(mutation).unwrap_or_else(|e| panic!("{mutation:?}: {e}"));
        check(&owner).unwrap_or_else(|e| panic!("{mutation:?}: {e:?}"));
        if matches!(mutation, Mutation::RawShuffleTerminalDeadSnapshot) {
            let semantic = owner.into_source_owner().expect("original source replay");
            let whole = ProductionSemanticKirOwnerV1::try_lower(
                semantic,
                ProductionSemanticKirLimitsV1::default(),
            )
            .expect("normal whole-owner construction with terminal snapshot storage end");
            whole
                .verify_equivalence()
                .expect("whole-owner replay with terminal snapshot storage end");
            assert!(
                whole
                    .module()
                    .functions
                    .iter()
                    .flat_map(|function| function.body.iter())
                    .flat_map(|body| &body.blocks)
                    .flat_map(|block| &block.operations)
                    .any(|operation| matches!(
                        operation.kind,
                        OperationKind::Wave(WaveOperation {
                            kind: WaveOperationKind::ShuffleIndex { .. },
                            ..
                        })
                    ))
            );
        }
    }
    for mutation in [
        Mutation::StorageDeadSnapshot,
        Mutation::DeadSnapshotFutureRead,
        Mutation::DeadSnapshotBackedge,
        Mutation::DeadLiveSnapshot,
        Mutation::TerminalDeadSnapshotAliasCopy,
        Mutation::TerminalDeadSnapshotRawAlias,
        Mutation::TerminalDeadSnapshotFormalRawAlias,
    ] {
        let owner = fixture(mutation).unwrap_or_else(|e| panic!("{mutation:?}: {e}"));
        assert!(
            matches!(check(&owner), Err(ReferenceErrorV1::Unavailable(_))),
            "{mutation:?}"
        );
    }
    let unreachable =
        fixture(Mutation::DeadSnapshotUnreachableIncoming).expect("unreachable call source admits");
    check(&unreachable).expect("first reachable call is genuine");
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, 10_000_000);
    budget
        .reserve_storage(unreachable.occurrence_storage().unwrap().retained_storage())
        .unwrap();
    assert!(matches!(
        with_checked_reference_helper_v1(
            &unreachable,
            SemanticFunctionIdV1::from_index(1),
            &mut budget,
            |_, _| Ok(())
        ),
        Err(ReferenceErrorV1::Unavailable(_))
    ));
    let owner = fixture(Mutation::TerminalDeadSnapshot).unwrap();
    let resolver = Resolver {
        owner: &owner,
        root: SemanticFunctionIdV1::from_index(0),
        visits: 0,
    };
    let mut zero_work = Work::new(0);
    let mut denied = Budget::new(&mut zero_work, 0);
    assert!(matches!(
        resolver.require_snapshot_dead_suffix(
            SemanticFunctionIdV1::from_index(0),
            SemanticLocalIdV1::from_index(3),
            3,
            0,
            &mut denied
        ),
        Err(ReferenceErrorV1::Resource(Resource::Work(_)))
    ));
    assert_eq!(denied.failed_work(), Some(1));
    assert_eq!(denied.storage(), 0);
}

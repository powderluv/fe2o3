use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_pliron::{
    ProductionSemanticMirLimitsV1, ProductionSemanticMirOwnerV1, ProductionSemanticSsaLimitsV1,
};

#[derive(Clone, Copy, Debug)]
enum Mutation {
    None,
    Rank,
    UnissuedLane,
    UnissuedContext,
    WrongLaneField,
    ReassignSnapshot,
    ForeignIncoming,
    SameIncoming,
    ForeignIncomingLane,
    DistinctIssuedLane,
    StorageDeadSnapshot,
    TerminalDeadSnapshot,
    TerminalDeadSnapshotAliasCopy,
    TerminalDeadSnapshotRawAlias,
    TerminalDeadSnapshotFormalRawAlias,
    DeadSnapshotFutureRead,
    DeadSnapshotBackedge,
    DeadSnapshotUnreachableIncoming,
    RawShuffleTerminalDeadSnapshot,
    InitialStorageLive,
    StorageLiveSnapshot,
    StorageLiveContext,
    DeadLiveSnapshot,
    DeadLiveContext,
    ProjectedExposure,
    MutableExposure,
    ContextCallOverwrite,
    ReassignSnapshotReference,
    ReassignContextReference,
    OverwriteSnapshotFormal,
    OverwriteContextFormal,
    RawSnapshotReference,
    RawContextReference,
    RawSnapshotFormal,
    RawContextFormal,
    RawBorrow,
    RawShuffle,
    HolderEntryLive,
    HolderLocalLive,
    HolderOrderedStorage,
    RawShuffleHolderOrderedStorage,
    HolderLateLive,
    HolderRestart,
    HolderUnreachableLive,
    HolderLiveBackedge,
    HolderDeadBackedge,
    HolderDeadBeforeCall,
    HolderAliasInitialLive,
    HolderFormalLive,
    HolderFormalDead,
    ShiftCounter,
    ShiftCounterZero,
    ShiftCounterOverwide,
    ShiftCounterZeroShift,
    ShiftCounterWideShift,
    ShiftCounterDynamicShift,
    ShiftCounterReassign,
    ShiftCounterInitReentry,
    ShiftCounterBorrow,
    ShiftCounterRawAddress,
    ShiftCounterSigned,
}

fn check(owner: &ProductionSemanticSsaOwnerV1) -> ResultV1<()> {
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, 10_000_000);
    budget.reserve_storage(owner.occurrence_storage().unwrap().retained_storage())?;
    with_checked_reference_call_v1(
        owner,
        SemanticFunctionIdV1::from_index(0),
        SemanticBlockIdV1::from_index(2),
        &mut budget,
        |view, _| {
            assert!(std::ptr::eq(view.owner(), owner));
            assert_eq!(view.helper(), SemanticFunctionIdV1::from_index(1));
            assert!(matches!(
                view.argument(0),
                Some(ArgumentV1::Snapshot { lane_field: 0, .. } | ArgumentV1::Rank { .. })
            ));
            assert!(matches!(view.argument(1), Some(ArgumentV1::Context { .. })));
            assert!(matches!(
                view.meanings[0].origin,
                SourceOrigin::RootIssuedSsa { .. }
            ));
            assert!(
                matches!(view.meanings[1].origin, SourceOrigin::RootRetainedCall {
                function, block, local, context_type,
            } if function.index() == 0 && block.index() == 1 && local.index() == 2 && context_type.index() == 4)
            );
            Ok(())
        },
    )
}
include!("tests/fixture_builders.rs");
include!("tests/fixture.rs");
include!("tests/call_relation_tests.rs");
include!("tests/borrow_tests.rs");
include!("tests/whole_owner_tests.rs");
include!("tests/lane_and_suffix_tests.rs");
include!("tests/holder_and_counter_tests.rs");

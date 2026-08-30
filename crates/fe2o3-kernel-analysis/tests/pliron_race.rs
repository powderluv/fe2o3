use dialect_gpu::{AddressSpaceAttr, ExecutionLayoutOp, FenceOp, MemoryOrderAttr, MemoryScopeAttr};
use dialect_kernel::{
    AccessKindAttr, AllocationEffectOp, AtomicOrderingAttr, AtomicScopeAttr, BranchOp,
    CheckedRowStripedIndex2DOp, CheckedTiledIndex2DOp, DIALECT_NAME,
    GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
    GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1, IndexBinaryKindAttr, IndexBinaryOp,
    IndexConstantOp, IndexEqualBranchOp, IndexLessThanBranchOp, IndexType, IndexUnknownOp,
    InvocationIndexOp, MemorySpaceAttr, RankedAccessOp, RankedViewOp, RankedViewType, ReturnOp,
    register_dialect,
};
use fe2o3_kernel_analysis::{
    KernelCheckPassKindV1, KernelCheckStatusV1, RankedRaceFindingV1, RankedRaceReportV1,
    require_pliron_ranked_race_freedom_before_lowering_v1, run_pliron_ranked_race_check_v1,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{op_interfaces::OneRegionInterface, ops::FuncOp, types::FunctionType},
    context::{Context, Ptr},
    dialect::DialectName,
    op::Op,
    r#type::TypeHandle,
    value::Value,
};

fn setup() -> Context {
    let mut context = Context::new();
    register_dialect(
        &mut context,
        &DialectName::try_new(DIALECT_NAME).expect("valid dialect"),
    )
    .expect("register kernel dialect");
    dialect_gpu::register_dialect(&mut context).expect("register gpu dialect");
    context
}

fn function(context: &mut Context, name: &str) -> FuncOp {
    let function_type = FunctionType::get(context, vec![], vec![]);
    FuncOp::new(
        context,
        name.try_into().expect("valid function"),
        function_type,
    )
}

fn function_with_index_arguments(
    context: &mut Context,
    name: &str,
    arguments: usize,
) -> (FuncOp, Vec<Value>) {
    let index: TypeHandle = IndexType::get(context).into();
    let function = FuncOp::new(
        context,
        name.try_into().expect("valid function"),
        FunctionType::get(context, vec![index; arguments], vec![]),
    );
    let entry = function.get_entry_block(context);
    let arguments = (0..arguments)
        .map(|ordinal| entry.deref(context).get_argument(ordinal))
        .collect();
    (function, arguments)
}

fn append<O: Op>(context: &Context, block: Ptr<BasicBlock>, operation: &O) {
    operation.get_operation().insert_at_back(block, context);
}

fn block(context: &mut Context, function: &FuncOp, name: &str) -> Ptr<BasicBlock> {
    let block = BasicBlock::new(
        context,
        Some(name.try_into().expect("valid block name")),
        vec![],
    );
    block.insert_at_back(function.get_region(context), context);
    block
}

fn view(context: &mut Context, shape: Vec<u64>, memory_space: MemorySpaceAttr) -> RankedViewOp {
    let view_type = RankedViewType::new(context, 32, true, shape).expect("ranked view type");
    RankedViewOp::new_in_space(context, view_type, vec![], memory_space).expect("ranked view")
}

fn view_with_contract(
    context: &mut Context,
    shape: Vec<u64>,
    memory_space: MemorySpaceAttr,
    allocation_origin: u64,
    noalias_class: u64,
) -> RankedViewOp {
    let view_type = RankedViewType::new(context, 32, true, shape).expect("ranked view type");
    RankedViewOp::new_in_space_with_allocation_contract(
        context,
        view_type,
        vec![],
        memory_space,
        allocation_origin,
        noalias_class,
    )
    .expect("ranked view")
}

fn access(
    context: &mut Context,
    kind: AccessKindAttr,
    view: Value,
    index: Value,
) -> RankedAccessOp {
    let atomic_ordering = match kind {
        AccessKindAttr::AtomicRead => Some(AtomicOrderingAttr::Acquire),
        AccessKindAttr::AtomicWrite => Some(AtomicOrderingAttr::Release),
        AccessKindAttr::AtomicReadModifyWrite => Some(AtomicOrderingAttr::AcquireRelease),
        AccessKindAttr::Read | AccessKindAttr::Write => None,
    };
    match atomic_ordering {
        Some(ordering) => RankedAccessOp::new_atomic(
            context,
            kind,
            ordering,
            AtomicScopeAttr::Device,
            view,
            vec![index],
        ),
        None => RankedAccessOp::new(context, kind, view, vec![index]),
    }
    .unwrap()
}

fn large_affine_store_family(formulas: &[(u64, u64)]) -> RankedRaceReportV1 {
    const LAUNCH: u64 = 65_537;

    let context = &mut setup();
    let function = function(context, "large_affine_store_family");
    let entry = function.get_entry_block(context);
    let maximum = formulas
        .iter()
        .map(|(stride, offset)| stride * (LAUNCH - 1) + offset)
        .max()
        .unwrap_or(0);
    let memory = view(context, vec![maximum + 1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, LAUNCH);
    append(context, entry, &memory);
    append(context, entry, &invocation);
    for &(stride, offset) in formulas {
        let stride = IndexConstantOp::new(context, stride);
        let base = IndexBinaryOp::new(
            context,
            IndexBinaryKindAttr::Multiply,
            invocation.result(context),
            stride.result(context),
        );
        let offset = IndexConstantOp::new(context, offset);
        let index = IndexBinaryOp::new(
            context,
            IndexBinaryKindAttr::Add,
            base.result(context),
            offset.result(context),
        );
        let write = access(
            context,
            AccessKindAttr::Write,
            memory.result(context),
            index.result(context),
        );
        append(context, entry, &stride);
        append(context, entry, &base);
        append(context, entry, &offset);
        append(context, entry, &index);
        append(context, entry, &write);
    }
    let ret = ReturnOp::new(context);
    append(context, entry, &ret);
    run_pliron_ranked_race_check_v1(context, &function)
}

#[test]
fn identity_write_is_injective_for_every_static_invocation() {
    let context = &mut setup();
    let function = function(context, "identity_write");
    let entry = function.get_entry_block(context);
    let output = view(context, vec![64], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 64);
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        output.result(context),
        vec![invocation.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &write);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.pass(), KernelCheckPassKindV1::RaceFreedom);
    assert_eq!(report.status(), KernelCheckStatusV1::Clean);
    assert!(report.findings().is_empty());
    assert!(!report.grants_compiler_refinement_authority());
    assert!(!report.grants_artifact_or_launch_authority());
}

#[test]
fn constant_output_coordinate_reports_two_exact_invocations() {
    let context = &mut setup();
    let function = function(context, "duplicate_output");
    let entry = function.get_entry_block(context);
    let output = view(context, vec![64], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 64);
    let zero = IndexConstantOp::new(context, 0);
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        output.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &zero);
    append(context, entry, &write);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    let [
        RankedRaceFindingV1::ConflictingEffects {
            indices,
            first,
            second,
            ..
        },
    ] = report.findings()
    else {
        panic!("unexpected findings: {:?}", report.findings());
    };
    assert_eq!(indices, &[0]);
    assert_eq!(first.invocation(), &[0]);
    assert_eq!(second.invocation(), &[1]);
    assert_eq!(first.access(), AccessKindAttr::Write);
    assert_eq!(second.access(), AccessKindAttr::Write);
    let error = require_pliron_ranked_race_freedom_before_lowering_v1(context, &function)
        .expect_err("duplicate write must stop lowering")
        .to_string();
    assert!(error.contains("error[FE2O3-RACE-001]"));
    assert!(error.contains("distinct concurrent invocations"));
    assert!(error.contains("invocation [0]"));
    assert!(error.contains("invocation [1]"));
}

#[test]
fn read_read_sharing_is_clean_but_read_write_and_write_write_are_rejected() {
    for (first_kind, second_kind, rejected) in [
        (AccessKindAttr::Read, AccessKindAttr::Read, false),
        (AccessKindAttr::Read, AccessKindAttr::Write, true),
        (AccessKindAttr::Write, AccessKindAttr::Read, true),
        (AccessKindAttr::Write, AccessKindAttr::Write, true),
    ] {
        let context = &mut setup();
        let function = function(context, "effect_pair");
        let entry = function.get_entry_block(context);
        let memory = view(context, vec![1], MemorySpaceAttr::Global);
        let invocation = InvocationIndexOp::new(context, 0, 2);
        let zero = IndexConstantOp::new(context, 0);
        let first = RankedAccessOp::new(
            context,
            first_kind,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        let second = RankedAccessOp::new(
            context,
            second_kind,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        let ret = ReturnOp::new(context);
        append(context, entry, &memory);
        append(context, entry, &invocation);
        append(context, entry, &zero);
        append(context, entry, &first);
        append(context, entry, &second);
        append(context, entry, &ret);
        assert_eq!(
            run_pliron_ranked_race_check_v1(context, &function).status()
                == KernelCheckStatusV1::Rejected,
            rejected,
            "unexpected result for {first_kind:?}/{second_kind:?}",
        );
    }
}

#[test]
fn atomics_order_with_atomics_but_not_with_plain_reads_or_writes() {
    for (other, rejected) in [
        (AccessKindAttr::AtomicRead, false),
        (AccessKindAttr::Read, true),
        (AccessKindAttr::Write, true),
    ] {
        let context = &mut setup();
        let function = function(context, "atomic_pair");
        let entry = function.get_entry_block(context);
        let memory = view(context, vec![1], MemorySpaceAttr::Global);
        let invocation = InvocationIndexOp::new(context, 0, 4);
        let zero = IndexConstantOp::new(context, 0);
        let atomic = RankedAccessOp::new_atomic(
            context,
            AccessKindAttr::AtomicReadModifyWrite,
            AtomicOrderingAttr::AcquireRelease,
            AtomicScopeAttr::Device,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        let other = access(context, other, memory.result(context), zero.result(context));
        let ret = ReturnOp::new(context);
        append(context, entry, &memory);
        append(context, entry, &invocation);
        append(context, entry, &zero);
        append(context, entry, &atomic);
        append(context, entry, &other);
        append(context, entry, &ret);
        assert_eq!(
            run_pliron_ranked_race_check_v1(context, &function).status()
                == KernelCheckStatusV1::Rejected,
            rejected,
        );
    }
}

#[test]
fn atomic_reads_share_with_plain_reads_and_all_atomic_effects() {
    for other in [
        AccessKindAttr::Read,
        AccessKindAttr::AtomicRead,
        AccessKindAttr::AtomicWrite,
        AccessKindAttr::AtomicReadModifyWrite,
    ] {
        let context = &mut setup();
        let function = function(context, "atomic_read_pair");
        let entry = function.get_entry_block(context);
        let memory = view(context, vec![1], MemorySpaceAttr::Global);
        let invocation = InvocationIndexOp::new(context, 0, 4);
        let zero = IndexConstantOp::new(context, 0);
        let atomic_read = RankedAccessOp::new_atomic(
            context,
            AccessKindAttr::AtomicRead,
            AtomicOrderingAttr::Acquire,
            AtomicScopeAttr::Device,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        let other = access(context, other, memory.result(context), zero.result(context));
        let ret = ReturnOp::new(context);
        append(context, entry, &memory);
        append(context, entry, &invocation);
        append(context, entry, &zero);
        append(context, entry, &atomic_read);
        append(context, entry, &other);
        append(context, entry, &ret);
        assert_eq!(
            run_pliron_ranked_race_check_v1(context, &function).status(),
            KernelCheckStatusV1::Clean,
        );
    }
}

#[test]
fn cross_workgroup_atomic_overlap_requires_agent_or_wider_scope() {
    for (scope, expected) in [
        (AtomicScopeAttr::Workgroup, KernelCheckStatusV1::Rejected),
        (AtomicScopeAttr::Agent, KernelCheckStatusV1::Clean),
        (AtomicScopeAttr::Device, KernelCheckStatusV1::Clean),
    ] {
        let context = &mut setup();
        let function = function(context, "cross_workgroup_atomic");
        let entry = function.get_entry_block(context);
        let layout = ExecutionLayoutOp::new(context, 41, [128, 1, 1], [64, 1, 1], 64);
        let memory = view(context, vec![1], MemorySpaceAttr::Global);
        let invocation = InvocationIndexOp::new(context, 0, 128);
        let zero = IndexConstantOp::new(context, 0);
        let atomic = RankedAccessOp::new_atomic(
            context,
            AccessKindAttr::AtomicReadModifyWrite,
            AtomicOrderingAttr::AcquireRelease,
            scope,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        let ret = ReturnOp::new(context);
        append(context, entry, &layout);
        append(context, entry, &memory);
        append(context, entry, &invocation);
        append(context, entry, &zero);
        append(context, entry, &atomic);
        append(context, entry, &ret);
        let report = run_pliron_ranked_race_check_v1(context, &function);
        assert_eq!(report.status(), expected, "unexpected status for {scope:?}");
        if scope == AtomicScopeAttr::Workgroup {
            assert!(matches!(
                report.findings(),
                [RankedRaceFindingV1::InsufficientAtomicScope { first, second, .. }]
                    if first.workgroup() == Some(0) && second.workgroup() == Some(1)
            ));
        }
    }
}

#[test]
fn narrow_atomic_overlap_without_layout_is_incomplete() {
    let context = &mut setup();
    let function = function(context, "unresolved_atomic_scope");
    let entry = function.get_entry_block(context);
    let memory = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 2);
    let zero = IndexConstantOp::new(context, 0);
    let atomic = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicReadModifyWrite,
        AtomicOrderingAttr::AcquireRelease,
        AtomicScopeAttr::Workgroup,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &memory);
    append(context, entry, &invocation);
    append(context, entry, &zero);
    append(context, entry, &atomic);
    append(context, entry, &ret);
    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::ExecutionLayoutUnavailable { .. }]
    ));
}

#[test]
fn fence_only_publication_is_incomplete_without_synchronizes_with() {
    let context = &mut setup();
    let function = function(context, "plain_grid_fence");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 42, [128, 1, 1], [64, 1, 1], 64);
    let memory = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 128);
    let zero = IndexConstantOp::new(context, 0);
    let first = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let fence = FenceOp::new(
        context,
        MemoryScopeAttr::Device,
        AddressSpaceAttr::Global,
        MemoryOrderAttr::AcquireRelease,
    );
    let second = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &memory);
    append(context, entry, &invocation);
    append(context, entry, &zero);
    append(context, entry, &first);
    append(context, entry, &fence);
    append(context, entry, &second);
    append(context, entry, &ret);
    assert_eq!(
        run_pliron_ranked_race_check_v1(context, &function).status(),
        KernelCheckStatusV1::Incomplete
    );
    assert!(
        run_pliron_ranked_race_check_v1(context, &function)
            .findings()
            .iter()
            .any(|finding| matches!(
                finding,
                RankedRaceFindingV1::HappensBeforeIncomplete { detail, .. }
                    if detail.contains("fence alone")
            ))
    );
}

#[test]
fn release_store_acquire_load_signaling_needs_a_read_from_proof() {
    let context = &mut setup();
    let function = function(context, "atomic_signal_publication");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 43, [128, 1, 1], [64, 1, 1], 64);
    let data = view_with_contract(context, vec![1], MemorySpaceAttr::Global, 431, 431);
    let signal = view_with_contract(context, vec![1], MemorySpaceAttr::Global, 432, 432);
    let invocation = InvocationIndexOp::new(context, 0, 128);
    let zero = IndexConstantOp::new(context, 0);
    let data_write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        data.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let signal_release = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicWrite,
        AtomicOrderingAttr::Release,
        AtomicScopeAttr::Agent,
        signal.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let signal_acquire = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicRead,
        AtomicOrderingAttr::Acquire,
        AtomicScopeAttr::Agent,
        signal.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let data_read = RankedAccessOp::new(
        context,
        AccessKindAttr::Read,
        data.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &data);
    append(context, entry, &signal);
    append(context, entry, &invocation);
    append(context, entry, &zero);
    append(context, entry, &data_write);
    append(context, entry, &signal_release);
    append(context, entry, &signal_acquire);
    append(context, entry, &data_read);
    append(context, entry, &ret);
    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(report.findings().iter().any(|finding| matches!(
        finding,
        RankedRaceFindingV1::HappensBeforeIncomplete { detail, .. }
            if detail.contains("authenticated read-from relation")
    )));
}

#[test]
fn affine_stride_and_offset_remain_injective() {
    let context = &mut setup();
    let function = function(context, "strided_output");
    let entry = function.get_entry_block(context);
    let output = view(context, vec![128], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 64);
    let two = IndexConstantOp::new(context, 2);
    let one = IndexConstantOp::new(context, 1);
    let scaled = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Multiply,
        invocation.result(context),
        two.result(context),
    );
    let offset = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        scaled.result(context),
        one.result(context),
    );
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        output.result(context),
        vec![offset.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &two);
    append(context, entry, &one);
    append(context, entry, &scaled);
    append(context, entry, &offset);
    append(context, entry, &write);
    append(context, entry, &ret);
    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

#[test]
fn guarded_overflowing_affine_multiply_is_not_proved_injective() {
    let context = &mut setup();
    let function = function(context, "guarded_overflowing_multiply");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 3);
    let factor = IndexConstantOp::new(context, 1_u64 << 63);
    let index = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Multiply,
        invocation.result(context),
        factor.result(context),
    );
    let extent = IndexConstantOp::new(context, 1);
    let guard = IndexLessThanBranchOp::new(
        context,
        index.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        index.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &factor);
    append(context, entry, &index);
    append(context, entry, &extent);
    append(context, entry, &guard);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert_eq!(
        report.findings(),
        &[RankedRaceFindingV1::BoundsPrerequisiteRejected]
    );
}

#[test]
fn guarded_overflowing_affine_add_is_not_proved_injective() {
    let context = &mut setup();
    let function = function(context, "guarded_overflowing_add");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 2);
    let maximum = IndexConstantOp::new(context, u64::MAX);
    let index = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        invocation.result(context),
        maximum.result(context),
    );
    let extent = IndexConstantOp::new(context, 1);
    let guard = IndexLessThanBranchOp::new(
        context,
        index.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        index.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &maximum);
    append(context, entry, &index);
    append(context, entry, &extent);
    append(context, entry, &guard);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert_eq!(
        report.findings(),
        &[RankedRaceFindingV1::BoundsPrerequisiteRejected]
    );
}

#[test]
fn checked_tiled_overflowing_invocation_is_not_proved_injective() {
    let context = &mut setup();
    let function = function(context, "checked_tiled_overflowing_invocation");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 3);
    let factor = IndexConstantOp::new(context, 1_u64 << 63);
    let tile_invocation = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Multiply,
        invocation.result(context),
        factor.result(context),
    );
    let zero = IndexConstantOp::new(context, 0);
    let sixteen = IndexConstantOp::new(context, 16);
    let tiled = CheckedTiledIndex2DOp::new(
        context,
        tile_invocation.result(context),
        zero.result(context),
        sixteen.result(context),
        sixteen.result(context),
        sixteen.result(context),
        [64, 16, 16, 4],
    );
    let extent = IndexConstantOp::new(context, 1);
    let guard = IndexLessThanBranchOp::new(
        context,
        tiled.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        tiled.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        factor.get_operation(),
        tile_invocation.get_operation(),
        zero.get_operation(),
        sixteen.get_operation(),
        tiled.get_operation(),
        extent.get_operation(),
        guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(
        report
            .findings()
            .iter()
            .any(|finding| matches!(finding, RankedRaceFindingV1::UnresolvedIndex { .. }))
    );
}

#[test]
fn checked_tiled_raw_marker_cannot_authorize_a_dynamic_launch() {
    let context = &mut setup();
    let function = function(context, "checked_tiled_dynamic_raw_invocation");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let zero = IndexConstantOp::new(context, 0);
    let sixteen = IndexConstantOp::new(context, 16);
    let tiled = CheckedTiledIndex2DOp::new(
        context,
        invocation.result(context),
        zero.result(context),
        sixteen.result(context),
        sixteen.result(context),
        sixteen.result(context),
        [64, 16, 16, 4],
    );
    let extent = IndexConstantOp::new(context, 1);
    let guard = IndexLessThanBranchOp::new(
        context,
        tiled.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        tiled.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        zero.get_operation(),
        sixteen.get_operation(),
        tiled.get_operation(),
        extent.get_operation(),
        guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
    ));
}

#[test]
fn checked_tiled_marker_shape_cannot_replace_validity_and_success_proofs() {
    #[derive(Clone, Copy, Debug)]
    enum HostileMarker {
        MissingSuccessEdge,
        ComponentEqualsElements,
        ComponentOutOfRange,
        DynamicComponent,
        TooSmallStride,
    }

    for hostile in [
        HostileMarker::MissingSuccessEdge,
        HostileMarker::ComponentEqualsElements,
        HostileMarker::ComponentOutOfRange,
        HostileMarker::DynamicComponent,
        HostileMarker::TooSmallStride,
    ] {
        let context = &mut setup();
        let function = function(context, "hostile_checked_tiled_marker");
        let entry = function.get_entry_block(context);
        let access_block = block(context, &function, "access");
        let exit = block(context, &function, "exit");
        let output = view(context, vec![4096], MemorySpaceAttr::Global);
        let invocation = InvocationIndexOp::new(context, 0, 2);
        let component_constant = IndexConstantOp::new(
            context,
            match hostile {
                HostileMarker::MissingSuccessEdge
                | HostileMarker::DynamicComponent
                | HostileMarker::TooSmallStride => 0,
                HostileMarker::ComponentEqualsElements => 4,
                HostileMarker::ComponentOutOfRange => 5,
            },
        );
        let rows = IndexConstantOp::new(context, 16);
        let columns = IndexConstantOp::new(context, 16);
        let stride = IndexConstantOp::new(
            context,
            if matches!(hostile, HostileMarker::TooSmallStride) {
                1
            } else {
                16
            },
        );
        let component = if matches!(hostile, HostileMarker::DynamicComponent) {
            invocation.result(context)
        } else {
            component_constant.result(context)
        };
        let tiled = CheckedTiledIndex2DOp::new(
            context,
            invocation.result(context),
            component,
            rows.result(context),
            columns.result(context),
            stride.result(context),
            [64, 16, 16, 4],
        );
        let extent = IndexConstantOp::new(context, 4096);
        let guard = IndexLessThanBranchOp::new(
            context,
            tiled.result(context),
            extent.result(context),
            access_block,
            exit,
        );
        let write = access(
            context,
            AccessKindAttr::Write,
            output.result(context),
            tiled.result(context),
        );
        let to_exit = BranchOp::new(context, exit);
        let ret = ReturnOp::new(context);
        for operation in [
            output.get_operation(),
            invocation.get_operation(),
            component_constant.get_operation(),
            rows.get_operation(),
            columns.get_operation(),
            stride.get_operation(),
            tiled.get_operation(),
            extent.get_operation(),
            guard.get_operation(),
        ] {
            operation.insert_at_back(entry, context);
        }
        append(context, access_block, &write);
        append(context, access_block, &to_exit);
        append(context, exit, &ret);

        let report = run_pliron_ranked_race_check_v1(context, &function);
        assert_eq!(
            report.status(),
            KernelCheckStatusV1::Incomplete,
            "raw marker {hostile:?} must not grant a proof: {report:?}"
        );
        assert!(matches!(
            report.findings(),
            [RankedRaceFindingV1::UnresolvedIndex { .. }]
        ));
        assert!(
            report.findings()[0]
                .to_string()
                .contains("checked structured index markers are currently incomplete")
        );
    }
}

#[test]
fn checked_tiled_equivalent_layout_markers_do_not_supply_success_semantics() {
    let context = &mut setup();
    let function = function(context, "checked_tiled_distinct_layout_constants");
    let entry = function.get_entry_block(context);
    let second_access = block(context, &function, "second_access");
    let third_access = block(context, &function, "third_access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![256], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let component_zero = IndexConstantOp::new(context, 0);
    let component_one = IndexConstantOp::new(context, 1);
    let first_rows = IndexConstantOp::new(context, 16);
    let first_columns = IndexConstantOp::new(context, 16);
    let first_stride = IndexConstantOp::new(context, 16);
    let second_rows = IndexConstantOp::new(context, 16);
    let second_columns = IndexConstantOp::new(context, 16);
    let second_stride = IndexConstantOp::new(context, 16);
    let first = CheckedTiledIndex2DOp::new(
        context,
        invocation.result(context),
        component_zero.result(context),
        first_rows.result(context),
        first_columns.result(context),
        first_stride.result(context),
        [64, 16, 16, 4],
    );
    let second = CheckedTiledIndex2DOp::new(
        context,
        invocation.result(context),
        component_one.result(context),
        second_rows.result(context),
        second_columns.result(context),
        second_stride.result(context),
        [64, 16, 16, 4],
    );
    let extent = IndexConstantOp::new(context, 256);
    let first_guard = IndexLessThanBranchOp::new(
        context,
        first.result(context),
        extent.result(context),
        second_access,
        exit,
    );
    let first_write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        first.result(context),
    );
    let second_guard = IndexLessThanBranchOp::new(
        context,
        second.result(context),
        extent.result(context),
        third_access,
        exit,
    );
    let second_write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        second.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        component_zero.get_operation(),
        component_one.get_operation(),
        first_rows.get_operation(),
        first_columns.get_operation(),
        first_stride.get_operation(),
        second_rows.get_operation(),
        second_columns.get_operation(),
        second_stride.get_operation(),
        first.get_operation(),
        second.get_operation(),
        extent.get_operation(),
        first_guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, second_access, &first_write);
    append(context, second_access, &second_guard);
    append(context, third_access, &second_write);
    append(context, third_access, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
    ));
}

#[test]
fn checked_tiled_dynamic_layout_never_authorizes_a_raw_marker() {
    #[derive(Clone, Copy, Debug)]
    enum LayoutCase {
        SharedEntryArguments,
        DifferentEntryArguments,
        InvocationVarying,
    }

    for case in [
        LayoutCase::SharedEntryArguments,
        LayoutCase::DifferentEntryArguments,
        LayoutCase::InvocationVarying,
    ] {
        let context = &mut setup();
        let argument_count = match case {
            LayoutCase::SharedEntryArguments => 3,
            LayoutCase::DifferentEntryArguments => 4,
            LayoutCase::InvocationVarying => 0,
        };
        let (function, arguments) =
            function_with_index_arguments(context, "checked_tiled_dynamic_layout", argument_count);
        let entry = function.get_entry_block(context);
        let second_access = block(context, &function, "second_access");
        let third_access = block(context, &function, "third_access");
        let exit = block(context, &function, "exit");
        let output = view(context, vec![4096], MemorySpaceAttr::Global);
        let invocation = InvocationIndexOp::new(context, 0, 0);
        let component_zero = IndexConstantOp::new(context, 0);
        let component_one = IndexConstantOp::new(context, 1);
        let fallback_layout = IndexConstantOp::new(context, 16);
        let extent = IndexConstantOp::new(context, 4096);
        let (first_layout, second_layout) = match case {
            LayoutCase::SharedEntryArguments => (
                [arguments[0], arguments[1], arguments[2]],
                [arguments[0], arguments[1], arguments[2]],
            ),
            LayoutCase::DifferentEntryArguments => (
                [arguments[0], arguments[1], arguments[2]],
                [arguments[0], arguments[1], arguments[3]],
            ),
            LayoutCase::InvocationVarying => (
                [
                    invocation.result(context),
                    fallback_layout.result(context),
                    fallback_layout.result(context),
                ],
                [
                    invocation.result(context),
                    fallback_layout.result(context),
                    fallback_layout.result(context),
                ],
            ),
        };
        let first = CheckedTiledIndex2DOp::new(
            context,
            invocation.result(context),
            component_zero.result(context),
            first_layout[0],
            first_layout[1],
            first_layout[2],
            [64, 16, 16, 4],
        );
        let second = CheckedTiledIndex2DOp::new(
            context,
            invocation.result(context),
            component_one.result(context),
            second_layout[0],
            second_layout[1],
            second_layout[2],
            [64, 16, 16, 4],
        );
        let first_guard = IndexLessThanBranchOp::new(
            context,
            first.result(context),
            extent.result(context),
            second_access,
            exit,
        );
        let first_write = access(
            context,
            AccessKindAttr::Write,
            output.result(context),
            first.result(context),
        );
        let second_guard = IndexLessThanBranchOp::new(
            context,
            second.result(context),
            extent.result(context),
            third_access,
            exit,
        );
        let second_write = access(
            context,
            AccessKindAttr::Write,
            output.result(context),
            second.result(context),
        );
        let to_exit = BranchOp::new(context, exit);
        let ret = ReturnOp::new(context);
        for operation in [
            output.get_operation(),
            invocation.get_operation(),
            component_zero.get_operation(),
            component_one.get_operation(),
            fallback_layout.get_operation(),
            extent.get_operation(),
            first.get_operation(),
            second.get_operation(),
            first_guard.get_operation(),
        ] {
            operation.insert_at_back(entry, context);
        }
        append(context, second_access, &first_write);
        append(context, second_access, &second_guard);
        append(context, third_access, &second_write);
        append(context, third_access, &to_exit);
        append(context, exit, &ret);

        let report = run_pliron_ranked_race_check_v1(context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
        assert!(matches!(
            report.findings(),
            [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
        ));
    }
}

#[test]
fn predicated_checked_access_proves_only_race_freedom() {
    for launch_extent in [0, 64] {
        for access_uses in [1, 2] {
            let context = &mut setup();
            let (function, arguments) =
                function_with_index_arguments(context, "raw_predicated_checked_access", 4);
            let entry = function.get_entry_block(context);
            let access_block = block(context, &function, "access");
            let exit = block(context, &function, "exit");
            let view_type =
                RankedViewType::new(context, 32, true, vec![dialect_kernel::DYNAMIC_EXTENT])
                    .unwrap();
            let output = RankedViewOp::new(context, view_type, vec![arguments[0]]).unwrap();
            let invocation = InvocationIndexOp::new(context, 0, launch_extent);
            let component = IndexConstantOp::new(context, 0);
            let checked = CheckedTiledIndex2DOp::new_predicated(
                context,
                invocation.result(context),
                component.result(context),
                arguments[1],
                arguments[2],
                arguments[3],
                arguments[0],
                [64, 16, 16, 4],
            );
            let guard = IndexLessThanBranchOp::new(
                context,
                checked.result(context),
                arguments[0],
                access_block,
                exit,
            );
            for operation in [
                output.get_operation(),
                invocation.get_operation(),
                component.get_operation(),
                checked.get_operation(),
                guard.get_operation(),
            ] {
                operation.insert_at_back(entry, context);
            }
            for _ in 0..access_uses {
                let write = RankedAccessOp::new_predicated(
                    context,
                    AccessKindAttr::Write,
                    output.result(context),
                    checked.result(context),
                    checked.success(context).unwrap(),
                )
                .unwrap();
                append(context, access_block, &write);
            }
            let to_exit = BranchOp::new(context, exit);
            let ret = ReturnOp::new(context);
            append(context, access_block, &to_exit);
            append(context, exit, &ret);

            let report = run_pliron_ranked_race_check_v1(context, &function);
            assert_eq!(report.status(), KernelCheckStatusV1::Clean);
            assert!(!report.grants_compiler_refinement_authority());
            assert!(!report.grants_artifact_or_launch_authority());
            let admitted =
                require_pliron_ranked_race_freedom_before_lowering_v1(context, &function)
                    .expect("the exact successful checked mapping is injective");
            assert!(!admitted.grants_compiler_refinement_authority());
            assert!(!admitted.grants_artifact_or_launch_authority());
        }
    }
}

#[test]
fn predicated_checked_access_rejects_invocation_varying_runtime_layout() {
    let context = &mut setup();
    let (function, arguments) =
        function_with_index_arguments(context, "varying_predicated_layout", 4);
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let view_type =
        RankedViewType::new(context, 32, true, vec![dialect_kernel::DYNAMIC_EXTENT]).unwrap();
    let output = RankedViewOp::new(context, view_type, vec![arguments[0]]).unwrap();
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let component = IndexConstantOp::new(context, 0);
    let checked = CheckedTiledIndex2DOp::new_predicated(
        context,
        invocation.result(context),
        component.result(context),
        invocation.result(context),
        arguments[2],
        arguments[3],
        arguments[0],
        [64, 16, 16, 4],
    );
    let guard = IndexLessThanBranchOp::new(
        context,
        checked.result(context),
        arguments[0],
        access_block,
        exit,
    );
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        component.get_operation(),
        checked.get_operation(),
        guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    let write = RankedAccessOp::new_predicated(
        context,
        AccessKindAttr::Write,
        output.result(context),
        checked.result(context),
        checked.success(context).unwrap(),
    )
    .unwrap();
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
    ));
}

#[test]
fn dynamic_multiaxis_mapping_requires_every_active_axis() {
    for drop_y in [false, true] {
        let context = &mut setup();
        let function = function(context, "dynamic_multiaxis");
        let entry = function.get_entry_block(context);
        let y_guard = block(context, &function, "y_guard");
        let access_block = block(context, &function, "access");
        let exit = block(context, &function, "exit");
        let shape = if drop_y { vec![1024] } else { vec![1024, 1024] };
        let output = view(context, shape, MemorySpaceAttr::Global);
        let x = InvocationIndexOp::new(context, 0, 0);
        let y = InvocationIndexOp::new(context, 1, 0);
        let extent = IndexConstantOp::new(context, 1024);
        let x_branch = IndexLessThanBranchOp::new(
            context,
            x.result(context),
            extent.result(context),
            y_guard,
            exit,
        );
        let y_branch = IndexLessThanBranchOp::new(
            context,
            y.result(context),
            extent.result(context),
            access_block,
            exit,
        );
        let indices = if drop_y {
            vec![x.result(context)]
        } else {
            vec![x.result(context), y.result(context)]
        };
        let write = RankedAccessOp::new(
            context,
            AccessKindAttr::Write,
            output.result(context),
            indices,
        )
        .unwrap();
        let to_exit = BranchOp::new(context, exit);
        let ret = ReturnOp::new(context);
        for operation in [
            output.get_operation(),
            x.get_operation(),
            y.get_operation(),
            extent.get_operation(),
            x_branch.get_operation(),
        ] {
            operation.insert_at_back(entry, context);
        }
        append(context, y_guard, &y_branch);
        append(context, access_block, &write);
        append(context, access_block, &to_exit);
        append(context, exit, &ret);

        let report = run_pliron_ranked_race_check_v1(context, &function);
        assert_eq!(
            report.status(),
            if drop_y {
                KernelCheckStatusV1::Incomplete
            } else {
                KernelCheckStatusV1::Clean
            }
        );
    }
}

#[test]
fn nonlinear_dynamic_mapping_remains_incomplete_even_with_finite_guards() {
    let context = &mut setup();
    let function = function(context, "nonlinear_dynamic_mapping");
    let entry = function.get_entry_block(context);
    let index_guard = block(context, &function, "index_guard");
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![16], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let square = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Multiply,
        invocation.result(context),
        invocation.result(context),
    );
    let invocation_extent = IndexConstantOp::new(context, 4);
    let output_extent = IndexConstantOp::new(context, 16);
    let invocation_guard = IndexLessThanBranchOp::new(
        context,
        invocation.result(context),
        invocation_extent.result(context),
        index_guard,
        exit,
    );
    let bounds_guard = IndexLessThanBranchOp::new(
        context,
        square.result(context),
        output_extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        square.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        square.get_operation(),
        invocation_extent.get_operation(),
        output_extent.get_operation(),
        invocation_guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, index_guard, &bounds_guard);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
    ));
}

#[test]
fn checked_row_striped_raw_marker_cannot_authorize_a_dynamic_launch() {
    let context = &mut setup();
    let function = function(context, "checked_row_striped_dynamic_raw_invocation");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let zero = IndexConstantOp::new(context, 0);
    let rows = IndexConstantOp::new(context, 7);
    let columns = IndexConstantOp::new(context, 257);
    let stride = IndexConstantOp::new(context, 269);
    let striped = CheckedRowStripedIndex2DOp::new(
        context,
        invocation.result(context),
        zero.result(context),
        rows.result(context),
        columns.result(context),
        stride.result(context),
        [64, 64],
    );
    let extent = IndexConstantOp::new(context, 1);
    let guard = IndexLessThanBranchOp::new(
        context,
        striped.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        striped.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        zero.get_operation(),
        rows.get_operation(),
        columns.get_operation(),
        stride.get_operation(),
        striped.get_operation(),
        extent.get_operation(),
        guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);
    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
    ));
}

#[test]
fn checked_row_striped_shared_dynamic_layout_marker_is_not_authority() {
    let context = &mut setup();
    let (function, layout) =
        function_with_index_arguments(context, "checked_row_striped_dynamic_layout", 3);
    let entry = function.get_entry_block(context);
    let second_access = block(context, &function, "second_access");
    let third_access = block(context, &function, "third_access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![4096], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let component_zero = IndexConstantOp::new(context, 0);
    let component_one = IndexConstantOp::new(context, 1);
    let extent = IndexConstantOp::new(context, 4096);
    let first = CheckedRowStripedIndex2DOp::new(
        context,
        invocation.result(context),
        component_zero.result(context),
        layout[0],
        layout[1],
        layout[2],
        [64, 4],
    );
    let second = CheckedRowStripedIndex2DOp::new(
        context,
        invocation.result(context),
        component_one.result(context),
        layout[0],
        layout[1],
        layout[2],
        [64, 4],
    );
    let first_guard = IndexLessThanBranchOp::new(
        context,
        first.result(context),
        extent.result(context),
        second_access,
        exit,
    );
    let first_write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        first.result(context),
    );
    let second_guard = IndexLessThanBranchOp::new(
        context,
        second.result(context),
        extent.result(context),
        third_access,
        exit,
    );
    let second_write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        second.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        component_zero.get_operation(),
        component_one.get_operation(),
        extent.get_operation(),
        first.get_operation(),
        second.get_operation(),
        first_guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, second_access, &first_write);
    append(context, second_access, &second_guard);
    append(context, third_access, &second_write);
    append(context, third_access, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }]
    ));
}

#[test]
fn checked_row_striped_overflowing_invocation_is_not_proved_injective() {
    let context = &mut setup();
    let function = function(context, "checked_row_striped_overflowing_invocation");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 3);
    let factor = IndexConstantOp::new(context, 1_u64 << 63);
    let mapped_invocation = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Multiply,
        invocation.result(context),
        factor.result(context),
    );
    let zero = IndexConstantOp::new(context, 0);
    let rows = IndexConstantOp::new(context, 7);
    let columns = IndexConstantOp::new(context, 257);
    let stride = IndexConstantOp::new(context, 269);
    let striped = CheckedRowStripedIndex2DOp::new(
        context,
        mapped_invocation.result(context),
        zero.result(context),
        rows.result(context),
        columns.result(context),
        stride.result(context),
        [64, 64],
    );
    let extent = IndexConstantOp::new(context, 1);
    let guard = IndexLessThanBranchOp::new(
        context,
        striped.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        striped.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        factor.get_operation(),
        mapped_invocation.get_operation(),
        zero.get_operation(),
        rows.get_operation(),
        columns.get_operation(),
        stride.get_operation(),
        striped.get_operation(),
        extent.get_operation(),
        guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);
    assert_eq!(
        run_pliron_ranked_race_check_v1(context, &function).status(),
        KernelCheckStatusV1::Incomplete
    );
}

#[test]
fn dynamic_launch_identity_is_symbolically_disjoint_after_a_bounds_guard() {
    let context = &mut setup();
    let function = function(context, "dynamic_identity");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1024], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let extent = IndexConstantOp::new(context, 1024);
    let branch = IndexLessThanBranchOp::new(
        context,
        invocation.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        output.result(context),
        vec![invocation.result(context)],
    )
    .unwrap();
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &extent);
    append(context, entry, &branch);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

#[test]
fn dynamic_launch_shift_is_symbolically_disjoint_after_a_bounds_guard() {
    let context = &mut setup();
    let function = function(context, "dynamic_shift");
    let entry = function.get_entry_block(context);
    let bounds_block = block(context, &function, "bounds");
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1028], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let offset = IndexConstantOp::new(context, 4);
    let shifted = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        invocation.result(context),
        offset.result(context),
    );
    let extent = IndexConstantOp::new(context, 1024);
    let branch = IndexLessThanBranchOp::new(
        context,
        invocation.result(context),
        extent.result(context),
        bounds_block,
        exit,
    );
    let output_extent = IndexConstantOp::new(context, 1028);
    let bounds_branch = IndexLessThanBranchOp::new(
        context,
        shifted.result(context),
        output_extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        shifted.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        offset.get_operation(),
        shifted.get_operation(),
        extent.get_operation(),
        branch.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, bounds_block, &output_extent);
    append(context, bounds_block, &bounds_branch);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

#[test]
fn dynamic_launch_bound_is_discarded_at_an_unguarded_merge() {
    let context = &mut setup();
    let function = function(context, "unguarded_merge");
    let entry = function.get_entry_block(context);
    let guarded = block(context, &function, "guarded");
    let unguarded = block(context, &function, "unguarded");
    let merge = block(context, &function, "merge");
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1028], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let offset = IndexConstantOp::new(context, 4);
    let shifted = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        invocation.result(context),
        offset.result(context),
    );
    let extent = IndexConstantOp::new(context, 1024);
    let branch = IndexLessThanBranchOp::new(
        context,
        invocation.result(context),
        extent.result(context),
        guarded,
        unguarded,
    );
    let guarded_to_merge = BranchOp::new(context, merge);
    let unguarded_to_merge = BranchOp::new(context, merge);
    let output_extent = IndexConstantOp::new(context, 1028);
    let bounds_branch = IndexLessThanBranchOp::new(
        context,
        shifted.result(context),
        output_extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        shifted.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        offset.get_operation(),
        shifted.get_operation(),
        extent.get_operation(),
        branch.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, guarded, &guarded_to_merge);
    append(context, unguarded, &unguarded_to_merge);
    append(context, merge, &output_extent);
    append(context, merge, &bounds_branch);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(report.findings().iter().any(|finding| matches!(
        finding,
        RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 }
    )));
}

#[test]
fn dynamic_launch_equality_zero_proves_a_single_writer() {
    let context = &mut setup();
    let function = function(context, "single_writer");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let zero = IndexConstantOp::new(context, 0);
    let branch = IndexEqualBranchOp::new(
        context,
        invocation.result(context),
        zero.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        zero.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        output.get_operation(),
        invocation.get_operation(),
        zero.get_operation(),
        branch.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

#[test]
fn remainder_mapping_reports_wraparound_collision() {
    let context = &mut setup();
    let function = function(context, "wrapped_output");
    let entry = function.get_entry_block(context);
    let output = view(context, vec![32], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 64);
    let modulus = IndexConstantOp::new(context, 32);
    let wrapped = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Remainder,
        invocation.result(context),
        modulus.result(context),
    );
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        output.result(context),
        vec![wrapped.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &modulus);
    append(context, entry, &wrapped);
    append(context, entry, &write);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    let finding = report
        .findings()
        .iter()
        .find_map(|finding| match finding {
            RankedRaceFindingV1::ConflictingEffects { first, second, .. } => Some((first, second)),
            _ => None,
        })
        .expect("wraparound conflict");
    assert_eq!(finding.0.invocation(), &[0]);
    assert_eq!(finding.1.invocation(), &[32]);
}

#[test]
fn guarded_quotient_mapping_uses_exact_fallback_and_reports_a_collision() {
    let context = &mut setup();
    let function = function(context, "quotient_output");
    let entry = function.get_entry_block(context);
    let access_block = block(context, &function, "access");
    let exit = block(context, &function, "exit");
    let output = view(context, vec![4], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 64);
    let divisor = IndexConstantOp::new(context, 16);
    let quotient = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Divide,
        invocation.result(context),
        divisor.result(context),
    );
    let extent = IndexConstantOp::new(context, 4);
    let guard = IndexLessThanBranchOp::new(
        context,
        quotient.result(context),
        extent.result(context),
        access_block,
        exit,
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        quotient.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &divisor);
    append(context, entry, &quotient);
    append(context, entry, &extent);
    append(context, entry, &guard);
    append(context, access_block, &write);
    append(context, access_block, &to_exit);
    append(context, exit, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    let (first, second) = report
        .findings()
        .iter()
        .find_map(|finding| match finding {
            RankedRaceFindingV1::ConflictingEffects { first, second, .. } => Some((first, second)),
            _ => None,
        })
        .expect("quotient collision");
    assert_eq!(first.invocation(), &[0]);
    assert_eq!(second.invocation(), &[1]);
}

#[test]
fn multidimensional_identity_is_clean_and_dropped_dimension_collides() {
    for drop_y in [false, true] {
        let context = &mut setup();
        let function = function(context, "image_output");
        let entry = function.get_entry_block(context);
        let output = view(context, vec![3, 4], MemorySpaceAttr::Global);
        let x = InvocationIndexOp::new(context, 0, 4);
        let y = InvocationIndexOp::new(context, 1, 3);
        let zero = IndexConstantOp::new(context, 0);
        let write = RankedAccessOp::new(
            context,
            AccessKindAttr::Write,
            output.result(context),
            vec![
                if drop_y {
                    zero.result(context)
                } else {
                    y.result(context)
                },
                x.result(context),
            ],
        )
        .unwrap();
        let ret = ReturnOp::new(context);
        append(context, entry, &output);
        append(context, entry, &x);
        append(context, entry, &y);
        append(context, entry, &zero);
        append(context, entry, &write);
        append(context, entry, &ret);
        assert_eq!(
            run_pliron_ranked_race_check_v1(context, &function).status(),
            if drop_y {
                KernelCheckStatusV1::Rejected
            } else {
                KernelCheckStatusV1::Clean
            },
        );
    }
}

#[test]
fn private_memory_and_single_invocation_do_not_create_inter_invocation_races() {
    for (space, extent) in [(MemorySpaceAttr::Private, 64), (MemorySpaceAttr::Global, 1)] {
        let context = &mut setup();
        let function = function(context, "nonconcurrent_constant");
        let entry = function.get_entry_block(context);
        let memory = view(context, vec![1], space);
        let invocation = InvocationIndexOp::new(context, 0, extent);
        let zero = IndexConstantOp::new(context, 0);
        let write = RankedAccessOp::new(
            context,
            AccessKindAttr::Write,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        let ret = ReturnOp::new(context);
        append(context, entry, &memory);
        append(context, entry, &invocation);
        append(context, entry, &zero);
        append(context, entry, &write);
        append(context, entry, &ret);
        assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
    }
}

#[test]
fn dynamic_global_launch_needs_symbolic_disjointness_and_workgroup_effects_defer() {
    let context = &mut setup();
    let global_function = function(context, "unresolved_domain");
    let entry = global_function.get_entry_block(context);
    let memory = view(context, vec![1], MemorySpaceAttr::Global);
    let invocation = InvocationIndexOp::new(context, 0, 0);
    let zero = IndexConstantOp::new(context, 0);
    let read = RankedAccessOp::new(
        context,
        AccessKindAttr::Read,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &memory);
    append(context, entry, &invocation);
    append(context, entry, &zero);
    append(context, entry, &read);
    append(context, entry, &ret);
    assert!(run_pliron_ranked_race_check_v1(context, &global_function).is_clean());

    let constant_write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    constant_write
        .get_operation()
        .insert_before(context, ret.get_operation());
    let report = run_pliron_ranked_race_check_v1(context, &global_function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(
        report.findings()[0]
            .to_string()
            .contains("dynamic launch dimension")
    );

    let context = &mut setup();
    let function = function(context, "workgroup_deferred");
    let entry = function.get_entry_block(context);
    let memory = view(context, vec![2], MemorySpaceAttr::Workgroup);
    let invocation = InvocationIndexOp::new(context, 0, 2);
    let access = RankedAccessOp::new(
        context,
        AccessKindAttr::Read,
        memory.result(context),
        vec![invocation.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &memory);
    append(context, entry, &invocation);
    append(context, entry, &access);
    append(context, entry, &ret);
    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

#[test]
fn oversized_static_launch_is_rejected_before_effect_enumeration() {
    let context = &mut setup();
    let function = function(context, "oversized_launch");
    let entry = function.get_entry_block(context);
    let invocation = InvocationIndexOp::new(context, 0, 65_537);
    let memory = view(context, vec![1], MemorySpaceAttr::Global);
    let zero = IndexConstantOp::new(context, 0);
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &invocation);
    append(context, entry, &memory);
    append(context, entry, &zero);
    append(context, entry, &write);
    append(context, entry, &ret);
    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::LaunchDomainTooLarge {
            invocations: 65_537,
            ..
        }]
    ));
}

#[test]
fn presburger_relations_prove_disjoint_effects_beyond_the_trace_limit() {
    let context = &mut setup();
    let function = function(context, "presburger_large_disjoint");
    let entry = function.get_entry_block(context);
    let invocation = InvocationIndexOp::new(context, 0, 65_537);
    let memory = view(context, vec![131_074], MemorySpaceAttr::Global);
    let two = IndexConstantOp::new(context, 2);
    let one = IndexConstantOp::new(context, 1);
    let even = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Multiply,
        invocation.result(context),
        two.result(context),
    );
    let odd = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        even.result(context),
        one.result(context),
    );
    let write = access(
        context,
        AccessKindAttr::Write,
        memory.result(context),
        even.result(context),
    );
    let read = access(
        context,
        AccessKindAttr::Read,
        memory.result(context),
        odd.result(context),
    );
    let ret = ReturnOp::new(context);
    for operation in [
        invocation.get_operation(),
        memory.get_operation(),
        two.get_operation(),
        one.get_operation(),
        even.get_operation(),
        odd.get_operation(),
        write.get_operation(),
        read.get_operation(),
        ret.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert!(report.is_clean(), "{:#?}", report.findings());
}

#[test]
fn affine_residue_family_proves_eight_blocked_stores_beyond_the_trace_limit() {
    let formulas = (0..8).map(|offset| (8, offset)).collect::<Vec<_>>();
    let report = large_affine_store_family(&formulas);
    assert!(report.is_clean(), "{:#?}", report.findings());
}

#[test]
fn affine_residue_family_checks_every_pair_and_rejects_collisions() {
    for formulas in [
        vec![(8, 0), (8, 1), (8, 9)],
        vec![(8, 0), (8, 8)],
        vec![(8, 0), (4, 0)],
    ] {
        let report = large_affine_store_family(&formulas);
        assert!(!report.is_clean(), "hostile formulas passed: {formulas:?}");
    }
}

#[test]
fn oversized_presburger_pair_inventory_fails_closed_before_pair_enumeration() {
    let context = &mut setup();
    let function = function(context, "presburger_pair_budget");
    let entry = function.get_entry_block(context);
    let invocation = InvocationIndexOp::new(context, 0, 65_537);
    let memory = view(context, vec![1], MemorySpaceAttr::Global);
    let zero = IndexConstantOp::new(context, 0);
    append(context, entry, &invocation);
    append(context, entry, &memory);
    append(context, entry, &zero);
    for _ in 0..1_448 {
        let write = RankedAccessOp::new(
            context,
            AccessKindAttr::Write,
            memory.result(context),
            vec![zero.result(context)],
        )
        .unwrap();
        append(context, entry, &write);
    }
    let ret = ReturnOp::new(context);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::LaunchDomainTooLarge {
            invocations: 65_537,
            ..
        }]
    ));
}

#[test]
fn declared_layout_checks_constant_effect_without_invocation_index() {
    let context = &mut setup();
    let function = function(context, "constant_without_index");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 50, [64, 1, 1], [64, 1, 1], 64);
    let memory = view_with_contract(context, vec![1], MemorySpaceAttr::Global, 50, 50);
    let zero = IndexConstantOp::new(context, 0);
    let write = RankedAccessOp::new(
        context,
        AccessKindAttr::Write,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &memory);
    append(context, entry, &zero);
    append(context, entry, &write);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::ConflictingEffects { first, second, .. }]
            if first.invocation() == [0, 0, 0] && second.invocation() == [1, 0, 0]
    ));
}

#[test]
fn overlapping_atomics_require_both_scopes_to_cover_the_pair() {
    let context = &mut setup();
    let function = function(context, "mixed_atomic_scopes");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 51, [128, 1, 1], [64, 1, 1], 64);
    let memory = view_with_contract(context, vec![1], MemorySpaceAttr::Global, 51, 51);
    let zero = IndexConstantOp::new(context, 0);
    let agent_write = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicWrite,
        AtomicOrderingAttr::Release,
        AtomicScopeAttr::Agent,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let workgroup_read = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicRead,
        AtomicOrderingAttr::Acquire,
        AtomicScopeAttr::Workgroup,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &memory);
    append(context, entry, &zero);
    append(context, entry, &agent_write);
    append(context, entry, &workgroup_read);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(report.findings().iter().any(|finding| matches!(
        finding,
        RankedRaceFindingV1::InsufficientAtomicScope { first, second, .. }
            if first.workgroup() == Some(0) && second.workgroup() == Some(1)
    )));
}

fn cross_view_alias_report(
    first_origin: u64,
    first_class: u64,
    second_origin: u64,
    second_class: u64,
) -> fe2o3_kernel_analysis::RankedRaceReportV1 {
    let context = &mut setup();
    let function = function(context, "cross_view_alias");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 52, [2, 1, 1], [2, 1, 1], 2);
    let first = view_with_contract(
        context,
        vec![3],
        MemorySpaceAttr::Global,
        first_origin,
        first_class,
    );
    let second = view_with_contract(
        context,
        vec![3],
        MemorySpaceAttr::Global,
        second_origin,
        second_class,
    );
    let invocation = InvocationIndexOp::new(context, 0, 2);
    let one = IndexConstantOp::new(context, 1);
    let shifted = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        invocation.result(context),
        one.result(context),
    );
    let first_write = access(
        context,
        AccessKindAttr::Write,
        first.result(context),
        invocation.result(context),
    );
    let second_write = access(
        context,
        AccessKindAttr::Write,
        second.result(context),
        shifted.result(context),
    );
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &first);
    append(context, entry, &second);
    append(context, entry, &invocation);
    append(context, entry, &one);
    append(context, entry, &shifted);
    append(context, entry, &first_write);
    append(context, entry, &second_write);
    append(context, entry, &ret);
    run_pliron_ranked_race_check_v1(context, &function)
}

#[test]
fn same_noalias_class_without_relative_offsets_fails_closed() {
    let report = cross_view_alias_report(521, 53, 522, 53);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(report.findings().iter().any(|finding| matches!(
        finding,
        RankedRaceFindingV1::AllocationContractUnavailable { detail }
            if detail.contains("relative base offset")
    )));
}

#[test]
fn unknown_alias_views_without_relative_offsets_fail_closed() {
    let report = cross_view_alias_report(0, 0, 0, 0);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(report.findings().iter().any(|finding| matches!(
        finding,
        RankedRaceFindingV1::AllocationContractUnavailable { detail }
            if detail.contains("relative base offset")
    )));
}

#[test]
fn distinct_authenticated_noalias_classes_are_disjoint() {
    assert_eq!(
        cross_view_alias_report(521, 54, 522, 55).status(),
        KernelCheckStatusV1::Clean
    );
}

fn allocation_read_and_write_report(
    invocation_count: u64,
    read_origin: u64,
    read_class: u64,
    write_origin: u64,
    write_class: u64,
) -> fe2o3_kernel_analysis::RankedRaceReportV1 {
    let context = &mut setup();
    let function = function(context, "allocation_read_and_write");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(
        context,
        58,
        [invocation_count, 1, 1],
        [invocation_count, 1, 1],
        invocation_count,
    );
    let read = AllocationEffectOp::new(
        context,
        AccessKindAttr::Read,
        MemorySpaceAttr::Global,
        read_origin,
        read_class,
    )
    .unwrap();
    let output = view_with_contract(
        context,
        vec![invocation_count],
        MemorySpaceAttr::Global,
        write_origin,
        write_class,
    );
    let invocation = InvocationIndexOp::new(context, 0, invocation_count);
    let write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        invocation.result(context),
    );
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &read);
    append(context, entry, &output);
    append(context, entry, &invocation);
    append(context, entry, &write);
    append(context, entry, &ret);
    run_pliron_ranked_race_check_v1(context, &function)
}

#[test]
fn whole_allocation_read_is_safe_with_a_distinct_exclusive_output() {
    assert_eq!(
        allocation_read_and_write_report(64, 581, 58, 582, 59).status(),
        KernelCheckStatusV1::Clean
    );
}

#[test]
fn whole_allocation_read_and_same_class_write_are_safe_for_one_invocation() {
    assert_eq!(
        allocation_read_and_write_report(1, 581, 58, 581, 58).status(),
        KernelCheckStatusV1::Clean
    );
}

#[test]
fn whole_allocation_read_and_same_class_write_fail_closed_when_concurrent() {
    let report = allocation_read_and_write_report(64, 581, 58, 581, 58);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::AllocationContractUnavailable { detail }]
            if detail.contains("whole-allocation read")
                && detail.contains("concurrent invocations")
                && !detail.contains("[0]")
                && !detail.contains("coordinate")
    ));
}

#[test]
fn whole_allocation_unknown_alias_read_fails_closed_against_an_output() {
    let report = allocation_read_and_write_report(64, 0, 0, 582, 59);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::AllocationContractUnavailable { detail }]
            if detail.contains("unknown-alias")
    ));
}

#[test]
fn reserved_gfx950_transpose_effect_is_not_a_global_race() {
    let context = &mut setup();
    let function = function(context, "reserved_gfx950_transpose_effect");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 59, [64, 1, 1], [64, 1, 1], 64);
    let write = AllocationEffectOp::new(
        context,
        AccessKindAttr::Write,
        MemorySpaceAttr::Workgroup,
        GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
        GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
    )
    .expect("reserved transpose write");
    let read = AllocationEffectOp::new(
        context,
        AccessKindAttr::Read,
        MemorySpaceAttr::Workgroup,
        GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
        GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
    )
    .expect("reserved transpose read");
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &write);
    append(context, entry, &read);
    append(context, entry, &ret);

    assert_eq!(
        run_pliron_ranked_race_check_v1(context, &function).status(),
        KernelCheckStatusV1::Clean
    );
}

#[test]
fn malformed_non_global_allocation_effect_cannot_fail_open() {
    let context = &mut setup();
    let function = function(context, "malformed_non_global_allocation_effect");
    let entry = function.get_entry_block(context);
    let effect = AllocationEffectOp::new(
        context,
        AccessKindAttr::Read,
        MemorySpaceAttr::Global,
        581,
        58,
    )
    .expect("valid allocation effect before hostile mutation");
    effect.set_attr_kernel_allocation_effect_memory_space(context, MemorySpaceAttr::Workgroup);
    let ret = ReturnOp::new(context);
    append(context, entry, &effect);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::BoundsPrerequisiteRejected]
    ));
}

#[test]
fn incompatible_potentially_aliasing_view_signatures_fail_closed() {
    let context = &mut setup();
    let function = function(context, "incompatible_alias_views");
    let entry = function.get_entry_block(context);
    let first = view(context, vec![2], MemorySpaceAttr::Global);
    let second_type = RankedViewType::new(context, 64, true, vec![2]).expect("ranked view type");
    let second = RankedViewOp::new_in_space(context, second_type, vec![], MemorySpaceAttr::Global)
        .expect("ranked view");
    let zero = IndexConstantOp::new(context, 0);
    let first_write = access(
        context,
        AccessKindAttr::Write,
        first.result(context),
        zero.result(context),
    );
    let second_write = access(
        context,
        AccessKindAttr::Write,
        second.result(context),
        zero.result(context),
    );
    let ret = ReturnOp::new(context);
    append(context, entry, &first);
    append(context, entry, &second);
    append(context, entry, &zero);
    append(context, entry, &first_write);
    append(context, entry, &second_write);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::AllocationContractUnavailable { .. }]
    ));
}

#[test]
fn heterogeneous_read_only_views_in_one_alias_class_are_clean() {
    let context = &mut setup();
    let function = function(context, "heterogeneous_shared_inputs");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 57, [64, 1, 1], [64, 1, 1], 64);
    let first_type = RankedViewType::new(context, 16, false, vec![4]).expect("ranked view type");
    let first = RankedViewOp::new_in_space_with_allocation_contract(
        context,
        first_type,
        vec![],
        MemorySpaceAttr::Global,
        571,
        57,
    )
    .expect("ranked view");
    let second_type = RankedViewType::new(context, 32, false, vec![8]).expect("ranked view type");
    let second = RankedViewOp::new_in_space_with_allocation_contract(
        context,
        second_type,
        vec![],
        MemorySpaceAttr::Global,
        572,
        57,
    )
    .expect("ranked view");
    let zero = IndexConstantOp::new(context, 0);
    let first_read = access(
        context,
        AccessKindAttr::Read,
        first.result(context),
        zero.result(context),
    );
    let second_read = access(
        context,
        AccessKindAttr::Read,
        second.result(context),
        zero.result(context),
    );
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &first);
    append(context, entry, &second);
    append(context, entry, &zero);
    append(context, entry, &first_read);
    append(context, entry, &second_read);
    append(context, entry, &ret);

    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

fn guarded_unknown_read_with_two_disjoint_writes(
    context: &mut Context,
    read_noalias_class: u64,
) -> FuncOp {
    let function = function(context, "guarded_unknown_read_with_two_disjoint_writes");
    let entry = function.get_entry_block(context);
    let read_block = block(context, &function, "read");
    let exit = block(context, &function, "exit");
    let input_type = RankedViewType::new(context, 32, true, vec![128]).expect("input type");
    let input = RankedViewOp::new_in_space_with_allocation_contract(
        context,
        input_type,
        vec![],
        MemorySpaceAttr::Global,
        read_noalias_class,
        read_noalias_class,
    )
    .expect("input view");
    let output = view_with_contract(context, vec![128], MemorySpaceAttr::Global, 702, 702);
    let invocation = InvocationIndexOp::new(context, 0, 64);
    let offset = IndexConstantOp::new(context, 64);
    let input_extent = IndexConstantOp::new(context, 128);
    let second_index = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        invocation.result(context),
        offset.result(context),
    );
    let unknown = IndexUnknownOp::new(context);
    let guard = IndexLessThanBranchOp::new(
        context,
        unknown.result(context),
        input_extent.result(context),
        read_block,
        exit,
    );
    let first_write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        invocation.result(context),
    );
    let second_write = access(
        context,
        AccessKindAttr::Write,
        output.result(context),
        second_index.result(context),
    );
    let read = access(
        context,
        AccessKindAttr::Read,
        input.result(context),
        unknown.result(context),
    );
    let to_exit = BranchOp::new(context, exit);
    let ret = ReturnOp::new(context);
    for operation in [
        input.get_operation(),
        output.get_operation(),
        invocation.get_operation(),
        offset.get_operation(),
        input_extent.get_operation(),
        second_index.get_operation(),
        unknown.get_operation(),
        first_write.get_operation(),
        second_write.get_operation(),
        guard.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, read_block, &read);
    append(context, read_block, &to_exit);
    append(context, exit, &ret);
    function
}

#[test]
fn unresolved_read_only_class_does_not_block_disjoint_write_proof() {
    let context = &mut setup();
    let function = guarded_unknown_read_with_two_disjoint_writes(context, 701);

    assert!(run_pliron_ranked_race_check_v1(context, &function).is_clean());
}

#[test]
fn unresolved_read_in_writable_alias_class_still_fails_closed() {
    let context = &mut setup();
    let function = guarded_unknown_read_with_two_disjoint_writes(context, 702);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(
        matches!(
            report.findings(),
            [RankedRaceFindingV1::UnresolvedIndex { .. }]
        ),
        "{:#?}",
        report.findings()
    );
}

#[test]
fn multidimensional_workgroup_identity_is_componentwise() {
    let context = &mut setup();
    let function = function(context, "multidimensional_scope");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 56, [16, 16, 1], [8, 8, 1], 64);
    let memory = view_with_contract(context, vec![1], MemorySpaceAttr::Global, 56, 56);
    let x = InvocationIndexOp::new(context, 0, 16);
    let y = InvocationIndexOp::new(context, 1, 16);
    let zero = IndexConstantOp::new(context, 0);
    let atomic = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicReadModifyWrite,
        AtomicOrderingAttr::AcquireRelease,
        AtomicScopeAttr::Workgroup,
        memory.result(context),
        vec![zero.result(context)],
    )
    .unwrap();
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &memory);
    append(context, entry, &x);
    append(context, entry, &y);
    append(context, entry, &zero);
    append(context, entry, &atomic);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::InsufficientAtomicScope { first, second, .. }]
            if first.invocation() == [0, 0, 0]
                && second.invocation() == [8, 0, 0]
                && first.workgroup() == Some(0)
                && second.workgroup() == Some(1)
    ));
}

#[test]
fn invocation_axis_outside_retained_layout_fails_closed() {
    let context = &mut setup();
    let function = function(context, "unsupported_fourth_axis");
    let entry = function.get_entry_block(context);
    let layout = ExecutionLayoutOp::new(context, 58, [1, 1, 1], [1, 1, 1], 1);
    let memory = view_with_contract(context, vec![1], MemorySpaceAttr::Global, 58, 58);
    let fourth_axis = InvocationIndexOp::new(context, 3, 0);
    let zero = IndexConstantOp::new(context, 0);
    let read = access(
        context,
        AccessKindAttr::Read,
        memory.result(context),
        zero.result(context),
    );
    let ret = ReturnOp::new(context);
    append(context, entry, &layout);
    append(context, entry, &memory);
    append(context, entry, &fourth_axis);
    append(context, entry, &zero);
    append(context, entry, &read);
    append(context, entry, &ret);

    let report = run_pliron_ranked_race_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [RankedRaceFindingV1::ExecutionLayoutUnavailable { detail }]
            if detail.contains("axis 3")
                && detail.contains("outside the three-dimensional gpu.execution_layout")
    ));
}

#[test]
fn dialect_index_type_is_still_the_only_function_index_type() {
    let context = &mut setup();
    let index: TypeHandle = dialect_kernel::IndexType::get(context).into();
    assert!(index.deref(context).is::<dialect_kernel::IndexType>());
}

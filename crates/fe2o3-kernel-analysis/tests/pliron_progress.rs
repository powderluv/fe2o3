use dialect_kernel::{
    BranchArgsOp, BranchOp, DIALECT_NAME, IndexBinaryKindAttr, IndexBinaryOp, IndexConstantOp,
    IndexLessThanBranchArgsOp, IndexLessThanBranchOp, IndexType, IndexUnsignedCastOp,
    InvocationIndexOp, ReturnOp, register_dialect,
};
use fe2o3_kernel_analysis::{
    KernelCheckStatusV1, MAX_PLIRON_PROGRESS_ATTRIBUTES_V1, MAX_PLIRON_PROGRESS_BLOCK_ARGUMENTS_V1,
    MAX_PLIRON_PROGRESS_BLOCKS_V1, MAX_PLIRON_PROGRESS_EDGES_V1,
    MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1, MAX_PLIRON_PROGRESS_OPERANDS_V1,
    MAX_PLIRON_PROGRESS_OPERATIONS_V1, MAX_PLIRON_PROGRESS_REGIONS_V1,
    MAX_PLIRON_PROGRESS_RESULTS_V1, MAX_PLIRON_PROGRESS_WORK_UNITS_V1, PlironProgressFindingV1,
    PlironProgressReportV1, run_pliron_progress_check_v1,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::UnitAttr, op_interfaces::OneRegionInterface, ops::FuncOp, types::FunctionType,
    },
    context::{Context, Ptr},
    dialect::DialectName,
    op::Op,
    operation::verify_operation,
    r#type::TypeHandle,
    value::Value,
};

fn setup() -> Context {
    let mut context = Context::new();
    register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
    context
}

fn make_function(context: &mut Context, name: &str, arguments: usize) -> (FuncOp, Vec<Value>) {
    let index: TypeHandle = IndexType::get(context).into();
    let function = FuncOp::new(
        context,
        name.try_into().unwrap(),
        FunctionType::get(context, vec![index; arguments], vec![]),
    );
    let values = (0..arguments)
        .map(|ordinal| {
            function
                .get_entry_block(context)
                .deref(context)
                .get_argument(ordinal)
        })
        .collect();
    (function, values)
}

fn append<O: Op>(context: &Context, block: Ptr<BasicBlock>, operation: &O) {
    operation.get_operation().insert_at_back(block, context);
}

fn block(context: &mut Context, function: &FuncOp, name: &str) -> Ptr<BasicBlock> {
    let block = BasicBlock::new(context, Some(name.try_into().unwrap()), vec![]);
    block.insert_at_back(function.get_region(context), context);
    block
}

fn index_block(context: &mut Context, function: &FuncOp, name: &str) -> (Ptr<BasicBlock>, Value) {
    let index: TypeHandle = IndexType::get(context).into();
    let block = BasicBlock::new(context, Some(name.try_into().unwrap()), vec![index]);
    let argument = block.deref(context).get_argument(0);
    block.insert_at_back(function.get_region(context), context);
    (block, argument)
}

fn index_block_n(
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
    let values = (0..arguments)
        .map(|ordinal| block.deref(context).get_argument(ordinal))
        .collect();
    block.insert_at_back(function.get_region(context), context);
    (block, values)
}

#[derive(Clone, Copy)]
enum NestedLoopCase {
    Canonical,
    ZeroInnerStep,
    NonzeroInnerStart,
    LoopLocalInnerBound,
}

fn nested_loop(context: &mut Context, case: NestedLoopCase) -> FuncOp {
    let (function, _) = make_function(context, "nested_loop", 0);
    let entry = function.get_entry_block(context);
    let (outer_header, outer) = index_block_n(context, &function, "outer_header", 2);
    let (inner_header, inner) = index_block_n(context, &function, "inner_header", 3);
    let (inner_body, body) = index_block_n(context, &function, "inner_body", 3);
    let (outer_latch, latch) = index_block_n(context, &function, "outer_latch", 2);
    let exit = block(context, &function, "exit");
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let carried = IndexConstantOp::new(context, 9);
    let outer_bound = IndexConstantOp::new(context, 3);
    let inner_bound = IndexConstantOp::new(context, 4);
    for operation in [
        zero.get_operation(),
        one.get_operation(),
        carried.get_operation(),
        outer_bound.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    if matches!(case, NestedLoopCase::LoopLocalInnerBound) {
        append(context, inner_header, &inner_bound);
    } else {
        append(context, entry, &inner_bound);
    }
    let enter = BranchArgsOp::new(
        context,
        vec![zero.result(context), carried.result(context)],
        outer_header,
    );
    append(context, entry, &enter);
    let inner_start = if matches!(case, NestedLoopCase::NonzeroInnerStart) {
        one.result(context)
    } else {
        zero.result(context)
    };
    let outer_condition = IndexLessThanBranchArgsOp::new(
        context,
        outer[0],
        outer_bound.result(context),
        vec![inner_start, outer[0], outer[1]],
        vec![],
        inner_header,
        exit,
    );
    append(context, outer_header, &outer_condition);
    let inner_condition = IndexLessThanBranchArgsOp::new(
        context,
        inner[0],
        inner_bound.result(context),
        vec![inner[0], inner[1], inner[2]],
        vec![inner[1], inner[2]],
        inner_body,
        outer_latch,
    );
    append(context, inner_header, &inner_condition);
    let inner_step = if matches!(case, NestedLoopCase::ZeroInnerStep) {
        zero.result(context)
    } else {
        one.result(context)
    };
    let inner_next = IndexBinaryOp::new(context, IndexBinaryKindAttr::Add, body[0], inner_step);
    let inner_repeat = BranchArgsOp::new(
        context,
        vec![inner_next.result(context), body[1], body[2]],
        inner_header,
    );
    append(context, inner_body, &inner_next);
    append(context, inner_body, &inner_repeat);
    let outer_next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        latch[0],
        one.result(context),
    );
    let outer_repeat = BranchArgsOp::new(
        context,
        vec![outer_next.result(context), latch[1]],
        outer_header,
    );
    append(context, outer_latch, &outer_next);
    append(context, outer_latch, &outer_repeat);
    let ret = ReturnOp::new(context);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    function
}

fn constant_loop(context: &mut Context, start: u64, bound: u64, step: u64) -> FuncOp {
    let (function, _) = make_function(context, "constant_loop", 0);
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (body, body_induction) = index_block(context, &function, "body");
    let exit = block(context, &function, "exit");
    let start = IndexConstantOp::new(context, start);
    let bound = IndexConstantOp::new(context, bound);
    let step = IndexConstantOp::new(context, step);
    let enter = BranchArgsOp::new(context, vec![start.result(context)], header);
    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        bound.result(context),
        vec![induction],
        vec![],
        body,
        exit,
    );
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        body_induction,
        step.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let ret = ReturnOp::new(context);
    for operation in [
        start.get_operation(),
        bound.get_operation(),
        step.get_operation(),
        enter.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, header, &condition);
    append(context, body, &next);
    append(context, body, &repeat);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    function
}

#[derive(Clone, Copy)]
enum MultiBlockCase {
    Canonical,
    MutatedForwarding,
    GuardedForwarding,
    GuardedMutatedForwarding,
    GuardedInternalFork,
    GuardedResetFork,
    ExternalIntermediateEntry,
    MultipleHeaderEntries,
    InvocationLatchUpdate,
}

#[derive(Clone, Copy)]
enum RangeCase {
    None,
    Bound(&'static [u64]),
    Mismatched(u64),
    NonEntry(u64),
}

fn multi_block_loop(
    context: &mut Context,
    static_bound: Option<u64>,
    step_value: u64,
    case: MultiBlockCase,
    range_case: RangeCase,
) -> FuncOp {
    let (function, arguments) = make_function(
        context,
        "multi_block_loop",
        usize::from(static_bound.is_none()),
    );
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (first, first_induction) = index_block(context, &function, "first");
    let (second, second_induction) = index_block(context, &function, "second");
    let (latch, latch_induction) = index_block(context, &function, "latch");
    let exit = block(context, &function, "exit");
    let start = IndexConstantOp::new(context, 0);
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let step = IndexConstantOp::new(context, step_value);
    let bound_constant = static_bound.map(|bound| IndexConstantOp::new(context, bound));
    let mut bound = bound_constant.as_ref().map_or_else(
        || arguments.first().copied().expect("symbolic bound"),
        |bound| bound.result(context),
    );
    let invocation = matches!(case, MultiBlockCase::InvocationLatchUpdate)
        .then(|| InvocationIndexOp::new(context, 0, 8));

    for operation in [
        start.get_operation(),
        zero.get_operation(),
        one.get_operation(),
        step.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    if let Some(bound) = &bound_constant {
        append(context, entry, bound);
    }
    if let Some(invocation) = &invocation {
        append(context, entry, invocation);
    }
    match range_case {
        RangeCase::None | RangeCase::NonEntry(_) => {}
        RangeCase::Bound(widths) => {
            for width in widths {
                let cast = IndexUnsignedCastOp::new(context, bound, *width);
                append(context, entry, &cast);
                bound = cast.result(context);
            }
        }
        RangeCase::Mismatched(width) => {
            let cast = IndexUnsignedCastOp::new(context, start.result(context), width);
            append(context, entry, &cast);
        }
    }
    if matches!(case, MultiBlockCase::MultipleHeaderEntries) {
        let first_entry = block(context, &function, "first_entry");
        let second_entry = block(context, &function, "second_entry");
        let split = IndexLessThanBranchArgsOp::new(
            context,
            zero.result(context),
            one.result(context),
            vec![],
            vec![],
            first_entry,
            second_entry,
        );
        let first_enter = BranchArgsOp::new(context, vec![start.result(context)], header);
        let second_enter = BranchArgsOp::new(context, vec![start.result(context)], header);
        append(context, entry, &split);
        append(context, first_entry, &first_enter);
        append(context, second_entry, &second_enter);
    } else if matches!(case, MultiBlockCase::ExternalIntermediateEntry) {
        let enter = IndexLessThanBranchArgsOp::new(
            context,
            zero.result(context),
            one.result(context),
            vec![start.result(context)],
            vec![start.result(context)],
            header,
            second,
        );
        append(context, entry, &enter);
    } else {
        let enter = BranchArgsOp::new(context, vec![start.result(context)], header);
        append(context, entry, &enter);
    }

    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        bound,
        vec![induction],
        vec![],
        first,
        exit,
    );
    append(context, header, &condition);

    if matches!(case, MultiBlockCase::MutatedForwarding) {
        let changed = IndexBinaryOp::new(
            context,
            IndexBinaryKindAttr::Add,
            first_induction,
            one.result(context),
        );
        let forward = BranchArgsOp::new(context, vec![changed.result(context)], second);
        append(context, first, &changed);
        append(context, first, &forward);
    } else if matches!(
        case,
        MultiBlockCase::GuardedForwarding
            | MultiBlockCase::GuardedMutatedForwarding
            | MultiBlockCase::GuardedInternalFork
            | MultiBlockCase::GuardedResetFork
    ) {
        let payload = if matches!(case, MultiBlockCase::GuardedMutatedForwarding) {
            let changed = IndexBinaryOp::new(
                context,
                IndexBinaryKindAttr::Add,
                first_induction,
                one.result(context),
            );
            append(context, first, &changed);
            changed.result(context)
        } else {
            first_induction
        };
        let internal_fork = matches!(
            case,
            MultiBlockCase::GuardedInternalFork | MultiBlockCase::GuardedResetFork
        );
        let reset_fork = matches!(case, MultiBlockCase::GuardedResetFork);
        let guard = IndexLessThanBranchArgsOp::new(
            context,
            if reset_fork {
                first_induction
            } else {
                zero.result(context)
            },
            one.result(context),
            vec![payload],
            internal_fork
                .then_some(if reset_fork {
                    zero.result(context)
                } else {
                    first_induction
                })
                .into_iter()
                .collect(),
            second,
            if internal_fork { latch } else { exit },
        );
        append(context, first, &guard);
    } else {
        if let RangeCase::NonEntry(width) = range_case {
            let cast = IndexUnsignedCastOp::new(context, bound, width);
            append(context, first, &cast);
        }
        let forward = BranchArgsOp::new(context, vec![first_induction], second);
        append(context, first, &forward);
    }
    let forward = BranchArgsOp::new(context, vec![second_induction], latch);
    append(context, second, &forward);

    let latch_base = invocation
        .as_ref()
        .map_or(latch_induction, |invocation| invocation.result(context));
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        latch_base,
        step.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let ret = ReturnOp::new(context);
    append(context, latch, &next);
    append(context, latch, &repeat);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    function
}

#[derive(Clone, Copy)]
enum BranchyLoopCase {
    Canonical,
    MutatedArm,
    BypassUpdate,
}

fn branchy_loop(context: &mut Context, case: BranchyLoopCase) -> FuncOp {
    let (function, _) = make_function(context, "branchy_loop", 0);
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (split, split_induction) = index_block(context, &function, "split");
    let (left, left_induction) = index_block(context, &function, "left");
    let (right, right_induction) = index_block(context, &function, "right");
    let (merge, merge_induction) = index_block(context, &function, "merge");
    let (latch, latch_induction) = index_block(context, &function, "latch");
    let exit = block(context, &function, "exit");
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let bound = IndexConstantOp::new(context, 8);
    let enter = BranchArgsOp::new(context, vec![zero.result(context)], header);
    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        bound.result(context),
        vec![induction],
        vec![],
        split,
        exit,
    );
    let fork = IndexLessThanBranchArgsOp::new(
        context,
        zero.result(context),
        one.result(context),
        vec![split_induction],
        vec![split_induction],
        left,
        right,
    );
    let left_target = if matches!(case, BranchyLoopCase::BypassUpdate) {
        split
    } else {
        merge
    };
    let left_forward = BranchArgsOp::new(context, vec![left_induction], left_target);
    let changed = matches!(case, BranchyLoopCase::MutatedArm).then(|| {
        IndexBinaryOp::new(
            context,
            IndexBinaryKindAttr::Add,
            right_induction,
            one.result(context),
        )
    });
    let right_value = changed
        .as_ref()
        .map_or(right_induction, |operation| operation.result(context));
    let right_forward = BranchArgsOp::new(context, vec![right_value], merge);
    let to_latch = BranchArgsOp::new(context, vec![merge_induction], latch);
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        latch_induction,
        one.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let ret = ReturnOp::new(context);

    for operation in [
        zero.get_operation(),
        one.get_operation(),
        bound.get_operation(),
        enter.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, header, &condition);
    append(context, split, &fork);
    append(context, left, &left_forward);
    if let Some(changed) = &changed {
        append(context, right, changed);
    }
    append(context, right, &right_forward);
    append(context, merge, &to_latch);
    append(context, latch, &next);
    append(context, latch, &repeat);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    function
}

#[test]
fn branchy_recurrence_reconverges_before_one_authenticated_update() {
    let context = &mut setup();
    let function = branchy_loop(context, BranchyLoopCase::Canonical);
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.certificates().len(), 1);
}

#[test]
fn branchy_recurrence_rejects_mutated_arm_and_update_bypass_cycle() {
    for (case, expected) in [
        (
            BranchyLoopCase::MutatedArm,
            "forward the induction value unchanged",
        ),
        (BranchyLoopCase::BypassUpdate, "bypass the induction update"),
    ] {
        let context = &mut setup();
        let function = branchy_loop(context, case);
        let report = run_pliron_progress_check_v1(context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
        assert!(matches!(
            report.findings(),
            [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
                if reason.contains(expected)
        ));
    }
}

#[test]
fn malformed_branch_arguments_fail_structural_verification_before_progress() {
    let context = &mut setup();
    let (function, _) = make_function(context, "malformed_progress_subject", 0);
    let entry = function.get_entry_block(context);
    let (target, _) = index_block(context, &function, "target");
    let malformed = BranchArgsOp::new(context, vec![], target);
    let ret = ReturnOp::new(context);
    append(context, entry, &malformed);
    append(context, target, &ret);

    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::StructuralPrerequisiteRejected { reason }]
            if reason.contains("verifier rejected") && reason.contains("requires 1 operands")
    ));
}

#[test]
fn missing_terminator_is_a_structural_rejection_with_verifier_detail() {
    let context = &mut setup();
    let (function, _) = make_function(context, "missing_terminator", 0);

    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::StructuralPrerequisiteRejected { reason }]
            if reason.contains("verifier rejected") && reason.contains("terminator")
    ));
}

#[test]
fn foreign_successor_is_a_structural_rejection_not_a_progress_gap() {
    let context = &mut setup();
    let (function, _) = make_function(context, "foreign_successor", 0);
    let (other, _) = make_function(context, "other_function", 0);
    let entry = function.get_entry_block(context);
    let branch = BranchOp::new(context, other.get_entry_block(context));
    append(context, entry, &branch);

    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::StructuralPrerequisiteRejected { reason }]
            if reason.contains("successor outside the function")
    ));
}

#[test]
fn cross_context_pointer_panic_is_contained_as_a_structural_rejection() {
    let owner = &mut setup();
    let (function, _) = make_function(owner, "foreign_context", 0);
    let context = setup();

    let report = run_pliron_progress_check_v1(&context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::StructuralPrerequisiteRejected { reason }]
            if reason.contains("cannot be borrowed from the supplied context")
    ));
}

#[test]
fn operation_count_rejects_exactly_one_over_the_limit_before_verification() {
    let context = &mut setup();
    let (function, _) = make_function(context, "operation_limit", 0);
    let entry = function.get_entry_block(context);
    for value in 0..MAX_PLIRON_PROGRESS_OPERATIONS_V1 {
        let constant = IndexConstantOp::new(context, value as u64);
        append(context, entry, &constant);
    }
    let ret = ReturnOp::new(context);
    append(context, entry, &ret);

    let report = run_pliron_progress_check_v1(context, &function);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ResourceLimitExceeded {
            resource: "operations",
            actual,
            limit: MAX_PLIRON_PROGRESS_OPERATIONS_V1,
        }] if *actual == MAX_PLIRON_PROGRESS_OPERATIONS_V1 + 1
    ));
}

#[test]
fn large_scc_uses_linear_accounting_and_does_not_regress_to_a_quadratic_rejection() {
    const CYCLE_BLOCKS: usize = 512;

    let context = &mut setup();
    let (function, _) = make_function(context, "work_limit", 0);
    let entry = function.get_entry_block(context);
    let cycle_tail = (1..CYCLE_BLOCKS)
        .map(|index| block(context, &function, &format!("cycle_{index}")))
        .collect::<Vec<_>>();
    let exit = block(context, &function, "exit");
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let enter = IndexLessThanBranchOp::new(
        context,
        zero.result(context),
        one.result(context),
        cycle_tail[0],
        exit,
    );
    append(context, entry, &zero);
    append(context, entry, &one);
    append(context, entry, &enter);
    for (index, current) in cycle_tail.iter().copied().enumerate() {
        let successor = cycle_tail.get(index + 1).copied().unwrap_or(entry);
        let repeat = BranchOp::new(context, successor);
        append(context, current, &repeat);
    }
    let ret = ReturnOp::new(context);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();

    let report = run_pliron_progress_check_v1(context, &function);
    assert!(!report.findings().iter().any(|finding| matches!(
        finding,
        PlironProgressFindingV1::ResourceLimitExceeded {
            resource: "work units",
            ..
        }
    )));
}

fn assert_not_limited_by(report: &PlironProgressReportV1, resource: &str) {
    assert!(
        !report.findings().iter().any(|finding| matches!(
            finding,
            PlironProgressFindingV1::ResourceLimitExceeded {
                resource: found,
                ..
            } if *found == resource
        )),
        "exact boundary unexpectedly rejected: {:?}",
        report.findings()
    );
}

fn assert_one_over(report: &PlironProgressReportV1, resource: &'static str, limit: usize) {
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ResourceLimitExceeded {
            resource: found,
            actual,
            limit: found_limit,
        }] if *found == resource && *actual == limit + 1 && *found_limit == limit
    ));
}

#[test]
fn operation_limit_accepts_exactly_the_boundary() {
    let context = &mut setup();
    let (function, _) = make_function(context, "operation_limit_exact", 0);
    let entry = function.get_entry_block(context);
    for value in 0..MAX_PLIRON_PROGRESS_OPERATIONS_V1 - 1 {
        let constant = IndexConstantOp::new(context, value as u64);
        append(context, entry, &constant);
    }
    let ret = ReturnOp::new(context);
    append(context, entry, &ret);

    let report = run_pliron_progress_check_v1(context, &function);
    assert_not_limited_by(&report, "operations");
}

#[test]
fn block_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_BLOCKS_V1, false),
        (MAX_PLIRON_PROGRESS_BLOCKS_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "block_limit", 0);
        for index in 1..count {
            block(context, &function, &format!("limit_{index}"));
        }
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(&report, "basic blocks", MAX_PLIRON_PROGRESS_BLOCKS_V1);
        } else {
            assert_not_limited_by(&report, "basic blocks");
        }
    }
}

#[test]
fn edge_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_EDGES_V1, false),
        (MAX_PLIRON_PROGRESS_EDGES_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "edge_limit", 0);
        let entry = function.get_entry_block(context);
        let ret = ReturnOp::new(context);
        for _ in 0..count {
            pliron::operation::Operation::push_successor(ret.get_operation(), context, entry);
        }
        append(context, entry, &ret);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(&report, "CFG edges", MAX_PLIRON_PROGRESS_EDGES_V1);
        } else {
            assert_not_limited_by(&report, "CFG edges");
        }
    }
}

#[test]
fn operand_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_OPERANDS_V1, false),
        (MAX_PLIRON_PROGRESS_OPERANDS_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "operand_limit", 0);
        let entry = function.get_entry_block(context);
        let constant = IndexConstantOp::new(context, 0);
        let ret = ReturnOp::new(context);
        append(context, entry, &constant);
        for _ in 0..count {
            pliron::operation::Operation::push_operand(
                ret.get_operation(),
                context,
                constant.result(context),
            );
        }
        append(context, entry, &ret);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(&report, "operands", MAX_PLIRON_PROGRESS_OPERANDS_V1);
        } else {
            assert_not_limited_by(&report, "operands");
        }
    }
}

#[test]
fn result_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_RESULTS_V1, false),
        (MAX_PLIRON_PROGRESS_RESULTS_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "result_limit", 0);
        let entry = function.get_entry_block(context);
        let ret = ReturnOp::new(context);
        let index: TypeHandle = IndexType::get(context).into();
        for _ in 0..count {
            pliron::operation::Operation::push_result(ret.get_operation(), context, index);
        }
        append(context, entry, &ret);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(&report, "results", MAX_PLIRON_PROGRESS_RESULTS_V1);
        } else {
            assert_not_limited_by(&report, "results");
        }
    }
}

#[test]
fn attribute_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_ATTRIBUTES_V1, false),
        (MAX_PLIRON_PROGRESS_ATTRIBUTES_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "attribute_limit", 0);
        let entry = function.get_entry_block(context);
        let ret = ReturnOp::new(context);
        let existing = function.get_operation().deref(context).attributes.0.len()
            + entry.deref(context).attributes.0.len();
        for index in 0..count - existing {
            ret.get_operation()
                .deref_mut(context)
                .attributes
                .set(format!("limit_{index}").try_into().unwrap(), UnitAttr);
        }
        append(context, entry, &ret);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(&report, "attributes", MAX_PLIRON_PROGRESS_ATTRIBUTES_V1);
        } else {
            assert_not_limited_by(&report, "attributes");
        }
    }
}

#[test]
fn block_argument_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_BLOCK_ARGUMENTS_V1, false),
        (MAX_PLIRON_PROGRESS_BLOCK_ARGUMENTS_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "argument_limit", 0);
        let index: TypeHandle = IndexType::get(context).into();
        let argument_block = BasicBlock::new(context, None, vec![index; count]);
        argument_block.insert_at_back(function.get_region(context), context);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(
                &report,
                "block arguments",
                MAX_PLIRON_PROGRESS_BLOCK_ARGUMENTS_V1,
            );
        } else {
            assert_not_limited_by(&report, "block arguments");
        }
    }
}

#[test]
fn region_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (count, one_over) in [
        (MAX_PLIRON_PROGRESS_REGIONS_V1, false),
        (MAX_PLIRON_PROGRESS_REGIONS_V1 + 1, true),
    ] {
        let context = &mut setup();
        let (function, _) = make_function(context, "region_limit", 0);
        let entry = function.get_entry_block(context);
        let ret = ReturnOp::new(context);
        for _ in 1..count {
            pliron::operation::Operation::add_region(ret.get_operation(), context);
        }
        append(context, entry, &ret);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(&report, "regions", MAX_PLIRON_PROGRESS_REGIONS_V1);
        } else {
            assert_not_limited_by(&report, "regions");
        }
    }
}

fn nested_function_chain(context: &mut Context, nested_functions: usize) -> FuncOp {
    let (root, _) = make_function(context, "nesting_root", 0);
    let mut parent = root;
    for depth in 0..nested_functions {
        let (child, _) = make_function(context, &format!("nested_{depth}"), 0);
        child
            .get_operation()
            .insert_at_back(parent.get_entry_block(context), context);
        parent = child;
    }
    let entry = parent.get_entry_block(context);
    let ret = ReturnOp::new(context);
    append(context, entry, &ret);
    root
}

#[test]
fn nesting_depth_accepts_exactly_the_boundary_and_rejects_one_more() {
    for (nested, one_over) in [
        (MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1 - 1, false),
        (MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1, true),
    ] {
        let context = &mut setup();
        let function = nested_function_chain(context, nested);
        let report = run_pliron_progress_check_v1(context, &function);
        if one_over {
            assert_one_over(
                &report,
                "operation nesting depth",
                MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1,
            );
        } else {
            assert_not_limited_by(&report, "operation nesting depth");
        }
    }
}

fn work_limit_function(context: &mut Context, arguments: usize) -> FuncOp {
    const CASTS: usize = 65_529;
    let (function, _) = make_function(context, "work_limit_boundary", arguments);
    let entry = function.get_entry_block(context);
    let zero = IndexConstantOp::new(context, 0);
    append(context, entry, &zero);
    for _ in 0..CASTS {
        let cast = IndexUnsignedCastOp::new(context, zero.result(context), 8);
        append(context, entry, &cast);
    }
    let ret = ReturnOp::new(context);
    append(context, entry, &ret);
    function
}

#[test]
fn cumulative_work_limit_accepts_exactly_the_boundary_and_rejects_one_more() {
    let context = &mut setup();
    let function = work_limit_function(context, 8);
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(
        report.is_clean(),
        "exact {MAX_PLIRON_PROGRESS_WORK_UNITS_V1}-unit boundary failed: {:?}",
        report.findings()
    );

    let context = &mut setup();
    let function = work_limit_function(context, 9);
    let report = run_pliron_progress_check_v1(context, &function);
    assert_one_over(&report, "work units", MAX_PLIRON_PROGRESS_WORK_UNITS_V1);
}

#[test]
fn canonical_unit_induction_proves_machine_finite_progress() {
    let context = &mut setup();
    let function = constant_loop(context, 0, 8, 1);
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean());
    assert_eq!(report.certificates().len(), 1);
    assert_eq!(report.certificates()[0].step(), 1);
}

#[test]
fn nested_positive_inductions_with_unrelated_carried_state_are_proved() {
    let context = &mut setup();
    let function = nested_loop(context, NestedLoopCase::Canonical);
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.certificates().len(), 2);
    assert!(
        report
            .certificates()
            .iter()
            .all(|certificate| certificate.step() == 1)
    );
}

#[test]
fn malformed_nested_inductions_fail_closed() {
    for case in [
        NestedLoopCase::ZeroInnerStep,
        NestedLoopCase::NonzeroInnerStart,
        NestedLoopCase::LoopLocalInnerBound,
    ] {
        let context = &mut setup();
        let function = nested_loop(context, case);
        let report = run_pliron_progress_check_v1(context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
        assert!(report.certificates().is_empty());
    }
}

#[test]
fn static_positive_nonunit_step_is_proved_only_when_its_update_cannot_overflow() {
    let context = &mut setup();
    let function = constant_loop(context, 0, 64, 16);
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean());
    assert_eq!(report.certificates()[0].step(), 16);

    let function = constant_loop(context, 0, u64::MAX, 16);
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
            if reason.contains("overflow")
    ));
}

#[test]
fn canonical_multiblock_recurrence_forwards_one_induction_value() {
    let context = &mut setup();
    let function = multi_block_loop(
        context,
        Some(64),
        16,
        MultiBlockCase::Canonical,
        RangeCase::None,
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.certificates().len(), 1);
    assert_eq!(report.certificates()[0].step(), 16);
}

#[test]
fn guarded_multiblock_recurrence_forwards_one_induction_value() {
    let context = &mut setup();
    let function = multi_block_loop(
        context,
        Some(64),
        1,
        MultiBlockCase::GuardedForwarding,
        RangeCase::None,
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.certificates().len(), 1);
}

#[test]
fn guarded_multiblock_recurrence_rejects_mutation_and_reset_forks() {
    for case in [
        MultiBlockCase::GuardedMutatedForwarding,
        MultiBlockCase::GuardedResetFork,
    ] {
        let context = &mut setup();
        let function = multi_block_loop(context, Some(64), 1, case, RangeCase::None);
        let report = run_pliron_progress_check_v1(context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    }
}

#[test]
fn induction_preserving_internal_fork_is_proved_by_the_general_loop_checker() {
    let context = &mut setup();
    let function = multi_block_loop(
        context,
        Some(64),
        1,
        MultiBlockCase::GuardedInternalFork,
        RangeCase::None,
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.certificates().len(), 1);
}

#[test]
fn multiblock_symbolic_bound_is_supported_only_for_a_unit_step() {
    let context = &mut setup();
    let function = multi_block_loop(context, None, 1, MultiBlockCase::Canonical, RangeCase::None);
    assert!(run_pliron_progress_check_v1(context, &function).is_clean());

    let function = multi_block_loop(
        context,
        None,
        16,
        MultiBlockCase::Canonical,
        RangeCase::None,
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
            if reason.contains("symbolic bound") && reason.contains("no-wrap")
    ));
}

#[test]
fn compiler_derived_unsigned_widths_prove_symbolic_nonunit_no_wrap() {
    for widths in [&[8_u64][..], &[16_u64][..], &[32_u64][..]] {
        let context = &mut setup();
        let function = multi_block_loop(
            context,
            None,
            16,
            MultiBlockCase::Canonical,
            RangeCase::Bound(widths),
        );
        let report = run_pliron_progress_check_v1(context, &function);
        assert!(report.is_clean(), "widths={widths:?}: {report:?}");
        assert_eq!(report.certificates()[0].step(), 16);
    }
}

#[test]
fn u64_range_does_not_hide_a_nonunit_update_overflow() {
    let context = &mut setup();
    let function = multi_block_loop(
        context,
        None,
        16,
        MultiBlockCase::Canonical,
        RangeCase::Bound(&[64]),
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
            if reason.contains("overflow")
    ));
}

#[test]
fn unrelated_and_unused_unsigned_casts_do_not_discharge_the_bound() {
    for (range_case, expected) in [
        (RangeCase::Mismatched(32), "symbolic bound"),
        (RangeCase::NonEntry(32), "symbolic bound"),
    ] {
        let context = &mut setup();
        let function = multi_block_loop(context, None, 16, MultiBlockCase::Canonical, range_case);
        let report = run_pliron_progress_check_v1(context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
        assert!(matches!(
            report.findings(),
            [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
                if reason.contains(expected)
        ));
    }
}

#[test]
fn unsigned_cast_width_is_verifier_closed_and_non_authoritative() {
    let context = &mut setup();
    let (function, arguments) = make_function(context, "malformed_range", 1);
    let cast = IndexUnsignedCastOp::new(context, arguments[0], 24);
    assert!(verify_operation(cast.get_operation(), context).is_err());
    assert!(!cast.grants_compiler_refinement_authority());
    assert!(!cast.grants_artifact_or_launch_authority());
    let ret = ReturnOp::new(context);
    append(context, function.get_entry_block(context), &ret);
}

#[test]
fn multiblock_static_bound_must_cover_the_final_update_without_wrap() {
    let context = &mut setup();
    let function = multi_block_loop(
        context,
        Some(u64::MAX),
        16,
        MultiBlockCase::Canonical,
        RangeCase::None,
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
            if reason.contains("overflow")
    ));
}

#[test]
fn multiblock_recurrence_rejects_mutation_and_alternate_entry() {
    for case in [
        MultiBlockCase::MutatedForwarding,
        MultiBlockCase::ExternalIntermediateEntry,
        MultiBlockCase::InvocationLatchUpdate,
    ] {
        let context = &mut setup();
        let function = multi_block_loop(context, Some(64), 1, case, RangeCase::None);
        let report = run_pliron_progress_check_v1(context, &function);
        assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    }
}

#[test]
fn multiblock_two_external_header_entries_are_not_single_entry() {
    let context = &mut setup();
    let function = multi_block_loop(
        context,
        Some(64),
        1,
        MultiBlockCase::MultipleHeaderEntries,
        RangeCase::None,
    );
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
            if reason.contains("exactly one external entry")
    ));
}

#[test]
fn external_body_predecessor_invalidates_the_canonical_recurrence() {
    let context = &mut setup();
    let (function, _) = make_function(context, "external_body_predecessor", 0);
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (body, body_induction) = index_block(context, &function, "body");
    let exit = block(context, &function, "exit");
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let bound = IndexConstantOp::new(context, 8);
    let hostile = IndexConstantOp::new(context, u64::MAX);
    let enter = IndexLessThanBranchArgsOp::new(
        context,
        zero.result(context),
        one.result(context),
        vec![hostile.result(context)],
        vec![zero.result(context)],
        body,
        header,
    );
    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        bound.result(context),
        vec![induction],
        vec![],
        body,
        exit,
    );
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        body_induction,
        one.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let ret = ReturnOp::new(context);
    for operation in [
        zero.get_operation(),
        one.get_operation(),
        bound.get_operation(),
        hostile.get_operation(),
        enter.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, header, &condition);
    append(context, body, &next);
    append(context, body, &repeat);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { reason, .. }]
            if reason.contains("bypasses the guarded header")
    ));
}

#[test]
fn feasible_zero_step_has_a_live_counterexample() {
    let context = &mut setup();
    let function = constant_loop(context, 0, 8, 0);
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::NonTerminatingCycle { counterexample, .. }]
            if counterexample.contains("i = 0") && counterexample.contains("bound = 8")
    ));
}

#[test]
fn infeasible_zero_step_true_edge_is_not_a_false_rejection() {
    let context = &mut setup();
    let function = constant_loop(context, 8, 8, 0);
    let report = run_pliron_progress_check_v1(context, &function);
    assert!(report.is_clean());
    assert!(report.certificates().is_empty());
}

#[test]
fn symbolic_zero_step_fails_closed_without_inventing_a_witness() {
    let context = &mut setup();
    let (function, arguments) = make_function(context, "symbolic_zero_step", 1);
    let bound = arguments[0];
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (body, body_induction) = index_block(context, &function, "body");
    let exit = block(context, &function, "exit");
    let start = IndexConstantOp::new(context, 0);
    let zero = IndexConstantOp::new(context, 0);
    let enter = BranchArgsOp::new(context, vec![start.result(context)], header);
    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        bound,
        vec![induction],
        vec![],
        body,
        exit,
    );
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        body_induction,
        zero.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let ret = ReturnOp::new(context);
    for operation in [
        start.get_operation(),
        zero.get_operation(),
        enter.get_operation(),
    ] {
        operation.insert_at_back(entry, context);
    }
    append(context, header, &condition);
    append(context, body, &next);
    append(context, body, &repeat);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Incomplete);
    assert!(matches!(
        report.findings(),
        [PlironProgressFindingV1::ProgressIncomplete { .. }]
    ));
}

#[test]
fn entry_self_cycle_is_rejected_but_unreachable_cycle_is_ignored() {
    let context = &mut setup();
    let (function, _) = make_function(context, "self_cycle", 0);
    let entry = function.get_entry_block(context);
    let repeat = BranchOp::new(context, entry);
    append(context, entry, &repeat);
    let report = run_pliron_progress_check_v1(context, &function);
    assert_eq!(report.status(), KernelCheckStatusV1::Rejected);

    let (function, _) = make_function(context, "unreachable_cycle", 0);
    let entry = function.get_entry_block(context);
    let cycle = block(context, &function, "cycle");
    let ret = ReturnOp::new(context);
    let repeat = BranchOp::new(context, cycle);
    append(context, entry, &ret);
    append(context, cycle, &repeat);
    assert!(run_pliron_progress_check_v1(context, &function).is_clean());
}

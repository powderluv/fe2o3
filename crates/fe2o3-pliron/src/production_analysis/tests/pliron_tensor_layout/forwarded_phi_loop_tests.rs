#[derive(Clone, Copy)]
enum ForwardedLoopSelectorV1 {
    Opaque,
    Lane,
}

fn forwarded_loop_phi_report_v1(
    selector: ForwardedLoopSelectorV1,
    identical_payload: bool,
) -> crate::PlironTensorLayoutReportV1 {
    let context = &mut setup();
    let (function, arguments) = function(context, "forwarded_loop_phi", 1);
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (body, body_induction) = index_block(context, &function, "body");
    let (left, left_induction) = index_block(context, &function, "left");
    let (right, right_induction) = index_block(context, &function, "right");
    let (join, joined_induction) = index_block(context, &function, "join");
    let exit = block(context, &function, "exit");
    let execution = layout(context, 0, 64, 64);
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let cutoff = IndexConstantOp::new(context, 32);
    let lane = InvocationIndexOp::new(context, 0, 0);
    let enter = BranchArgsOp::new(context, vec![zero.result(context)], header);
    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        arguments[0],
        vec![induction],
        vec![],
        body,
        exit,
    );
    let matrix = tensor(context, 64);
    let right_payload = if identical_payload {
        body_induction
    } else {
        one.result(context)
    };
    assert_ne!(left_induction, right_induction);
    let left_join = BranchArgsOp::new(context, vec![left_induction], join);
    let right_join = BranchArgsOp::new(context, vec![right_induction], join);
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        joined_induction,
        one.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let ret = ReturnOp::new(context);
    append(context, entry, &execution);
    append(context, entry, &zero);
    append(context, entry, &one);
    append(context, entry, &cutoff);
    append(context, entry, &lane);
    append(context, entry, &enter);
    append(context, header, &condition);
    append(context, body, &matrix);
    match selector {
        ForwardedLoopSelectorV1::Opaque => {
            let split = AnalysisSplitOp::new_with_control_and_arguments(
                context,
                vec![],
                vec![body_induction],
                vec![right_payload],
                left,
                right,
            );
            append(context, body, &split);
        }
        ForwardedLoopSelectorV1::Lane => {
            let split = IndexLessThanBranchArgsOp::new(
                context,
                lane.result(context),
                cutoff.result(context),
                vec![body_induction],
                vec![right_payload],
                left,
                right,
            );
            append(context, body, &split);
        }
    }
    append(context, left, &left_join);
    append(context, right, &right_join);
    append(context, join, &next);
    append(context, join, &repeat);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    run_pliron_tensor_layout_check_v1(context, &function)
}

#[test]
fn forwarded_equal_loop_phi_does_not_inherit_opaque_selection() {
    let report = forwarded_loop_phi_report_v1(ForwardedLoopSelectorV1::Opaque, true);
    assert!(report.is_clean(), "{:?}", report.findings());
}

#[test]
fn forwarded_equal_loop_phi_does_not_inherit_lane_selection() {
    let report = forwarded_loop_phi_report_v1(ForwardedLoopSelectorV1::Lane, true);
    assert!(report.is_clean(), "{:?}", report.findings());
}

#[test]
fn forwarded_unequal_loop_phi_keeps_selector_dependence() {
    let opaque = forwarded_loop_phi_report_v1(ForwardedLoopSelectorV1::Opaque, false);
    assert_eq!(opaque.status(), KernelCheckStatusV1::Incomplete);
    assert!(opaque.findings().iter().any(|finding| matches!(
        finding,
        PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail }
            if detail.contains("unresolved branch block 1")
    )));
    let varying = forwarded_loop_phi_report_v1(ForwardedLoopSelectorV1::Lane, false);
    assert_eq!(varying.status(), KernelCheckStatusV1::Rejected);
    assert!(varying.findings().iter().any(|finding| matches!(
        finding,
        PlironTensorLayoutFindingV1::DivergentSubgroupControl { controller: 1, .. }
    )));
}

#[test]
fn forwarded_phi_cycles_terminate_without_changing_control() {
    for forwarding_blocks in [1, 2, 9] {
        let context = &mut setup();
        let (function, arguments) = function(context, "forwarded_phi_cycle", 1);
        let entry = function.get_entry_block(context);
        let (header, induction) = index_block(context, &function, "header");
        let forwarding = (0..forwarding_blocks)
            .map(|ordinal| index_block(context, &function, &format!("forward_{ordinal}")))
            .collect::<Vec<_>>();
        let exit = block(context, &function, "exit");
        let execution = layout(context, 0, 64, 64);
        let zero = IndexConstantOp::new(context, 0);
        let one = IndexConstantOp::new(context, 1);
        let enter = BranchArgsOp::new(context, vec![zero.result(context)], header);
        let condition = IndexLessThanBranchArgsOp::new(
            context,
            arguments[0],
            one.result(context),
            vec![induction],
            vec![],
            forwarding[0].0,
            exit,
        );
        append(context, entry, &execution);
        append(context, entry, &zero);
        append(context, entry, &one);
        append(context, entry, &enter);
        append(context, header, &condition);
        for (ordinal, (source, value)) in forwarding.iter().copied().enumerate() {
            let target = forwarding.get(ordinal + 1).map_or(header, |next| next.0);
            let edge = BranchArgsOp::new(context, vec![value], target);
            append(context, source, &edge);
        }
        let matrix = tensor(context, 64);
        let ret = ReturnOp::new(context);
        append(context, exit, &matrix);
        append(context, exit, &ret);
        verify_operation(function.get_operation(), context).unwrap();
        let report = run_pliron_tensor_layout_check_v1(context, &function);
        assert!(
            report.is_clean(),
            "forwarding blocks={forwarding_blocks}: {:?}",
            report.findings()
        );
    }
}

fn nested_forwarded_outer_phi_report_v1(
    selector: ForwardedLoopSelectorV1,
    identical_payload: bool,
) -> crate::PlironTensorLayoutReportV1 {
    let context = &mut setup();
    let (function, arguments) = function(context, "nested_forwarded_outer_phi", 2);
    let entry = function.get_entry_block(context);
    let (outer, key) = index_block(context, &function, "outer");
    let (inner, inner_args) = index_block_n(context, &function, "inner", 2);
    let (body, body_args) = index_block_n(context, &function, "body", 2);
    let (left, left_args) = index_block_n(context, &function, "left", 2);
    let (right, right_args) = index_block_n(context, &function, "right", 2);
    let (join, join_args) = index_block_n(context, &function, "join", 2);
    let (latch, latch_key) = index_block(context, &function, "latch");
    let exit = block(context, &function, "exit");
    let execution = layout(context, 0, 64, 64);
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let step = IndexConstantOp::new(context, 16);
    let cutoff = IndexConstantOp::new(context, 32);
    let lane = InvocationIndexOp::new(context, 0, 0);
    let enter = BranchArgsOp::new(context, vec![zero.result(context)], outer);
    let outer_condition = IndexLessThanBranchArgsOp::new(
        context,
        key,
        arguments[0],
        vec![key, zero.result(context)],
        vec![],
        inner,
        exit,
    );
    let inner_condition = IndexLessThanBranchArgsOp::new(
        context,
        inner_args[1],
        arguments[1],
        inner_args.clone(),
        vec![inner_args[0]],
        body,
        latch,
    );
    let matrix = tensor(context, 64);
    let right_key = if identical_payload {
        body_args[0]
    } else {
        one.result(context)
    };
    let left_join = BranchArgsOp::new(context, left_args, join);
    let right_join = BranchArgsOp::new(context, right_args, join);
    let next_inner = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        join_args[1],
        one.result(context),
    );
    let repeat_inner = BranchArgsOp::new(
        context,
        vec![join_args[0], next_inner.result(context)],
        inner,
    );
    let next_outer = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        latch_key,
        step.result(context),
    );
    let repeat_outer = BranchArgsOp::new(context, vec![next_outer.result(context)], outer);
    let ret = ReturnOp::new(context);
    append(context, entry, &execution);
    append(context, entry, &zero);
    append(context, entry, &one);
    append(context, entry, &step);
    append(context, entry, &cutoff);
    append(context, entry, &lane);
    append(context, entry, &enter);
    append(context, outer, &outer_condition);
    append(context, inner, &inner_condition);
    append(context, body, &matrix);
    match selector {
        ForwardedLoopSelectorV1::Opaque => {
            let split = AnalysisSplitOp::new_with_control_and_arguments(
                context,
                vec![],
                body_args.clone(),
                vec![right_key, body_args[1]],
                left,
                right,
            );
            append(context, body, &split);
        }
        ForwardedLoopSelectorV1::Lane => {
            let split = IndexLessThanBranchArgsOp::new(
                context,
                lane.result(context),
                cutoff.result(context),
                body_args.clone(),
                vec![right_key, body_args[1]],
                left,
                right,
            );
            append(context, body, &split);
        }
    }
    append(context, left, &left_join);
    append(context, right, &right_join);
    append(context, join, &next_inner);
    append(context, join, &repeat_inner);
    append(context, latch, &next_outer);
    append(context, latch, &repeat_outer);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    run_pliron_tensor_layout_check_v1(context, &function)
}

#[test]
fn nested_unchanged_outer_phi_does_not_inherit_opaque_selection() {
    let report = nested_forwarded_outer_phi_report_v1(ForwardedLoopSelectorV1::Opaque, true);
    assert!(report.is_clean(), "{:?}", report.findings());
}

#[test]
fn nested_unchanged_outer_phi_does_not_inherit_lane_selection() {
    let report = nested_forwarded_outer_phi_report_v1(ForwardedLoopSelectorV1::Lane, true);
    assert!(report.is_clean(), "{:?}", report.findings());
}

#[test]
fn nested_changed_outer_phi_retains_unknown_and_divergent_control() {
    let unknown = nested_forwarded_outer_phi_report_v1(ForwardedLoopSelectorV1::Opaque, false);
    assert_eq!(unknown.status(), KernelCheckStatusV1::Incomplete);
    let varying = nested_forwarded_outer_phi_report_v1(ForwardedLoopSelectorV1::Lane, false);
    assert_eq!(varying.status(), KernelCheckStatusV1::Rejected);
}

fn guarded_forwarded_loop_report_v1(
    selector: ForwardedLoopSelectorV1,
    can_trap: bool,
) -> crate::PlironTensorLayoutReportV1 {
    let context = &mut setup();
    let (function, arguments) = function(context, "guarded_forwarded_loop", 3);
    let entry = function.get_entry_block(context);
    let (header, induction) = index_block(context, &function, "header");
    let (body, carried) = index_block(context, &function, "body");
    let (bypass, bypass_value) = index_block(context, &function, "bypass");
    let (checked, checked_value) = index_block(context, &function, "checked");
    let (join, joined) = index_block(context, &function, "join");
    let trap = block(context, &function, "trap");
    let exit = block(context, &function, "exit");
    let execution = layout(context, 0, 64, 64);
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let cutoff = IndexConstantOp::new(context, 32);
    let lane = InvocationIndexOp::new(context, 0, 0);
    let enter = BranchArgsOp::new(context, vec![zero.result(context)], header);
    let condition = IndexLessThanBranchArgsOp::new(
        context,
        induction,
        arguments[0],
        vec![induction],
        vec![],
        body,
        exit,
    );
    let matrix = tensor(context, 64);
    let bypass_join = BranchArgsOp::new(context, vec![bypass_value], join);
    // The guard is subgroup-uniform, but only the selected lanes execute it.
    // Without a proof that it succeeds, identical carried values do not
    // establish that every lane reaches the latch (real bb180/161/164 shape).
    let checked_join = IndexLessThanBranchArgsOp::new(
        context,
        arguments[1],
        arguments[2],
        vec![checked_value],
        if can_trap {
            vec![]
        } else {
            vec![checked_value]
        },
        join,
        if can_trap { trap } else { join },
    );
    let next = IndexBinaryOp::new(
        context,
        IndexBinaryKindAttr::Add,
        joined,
        one.result(context),
    );
    let repeat = BranchArgsOp::new(context, vec![next.result(context)], header);
    let abort = dialect_kernel::TrapOp::new(context);
    let ret = ReturnOp::new(context);
    append(context, entry, &execution);
    append(context, entry, &zero);
    append(context, entry, &one);
    append(context, entry, &cutoff);
    append(context, entry, &lane);
    append(context, entry, &enter);
    append(context, header, &condition);
    append(context, body, &matrix);
    match selector {
        ForwardedLoopSelectorV1::Opaque => {
            let split = AnalysisSplitOp::new_with_control_and_arguments(
                context,
                vec![],
                vec![carried],
                vec![carried],
                bypass,
                checked,
            );
            append(context, body, &split);
        }
        ForwardedLoopSelectorV1::Lane => {
            let split = IndexLessThanBranchArgsOp::new(
                context,
                lane.result(context),
                cutoff.result(context),
                vec![carried],
                vec![carried],
                bypass,
                checked,
            );
            append(context, body, &split);
        }
    }
    append(context, bypass, &bypass_join);
    append(context, checked, &checked_join);
    append(context, join, &next);
    append(context, join, &repeat);
    append(context, trap, &abort);
    append(context, exit, &ret);
    verify_operation(function.get_operation(), context).unwrap();
    run_pliron_tensor_layout_check_v1(context, &function)
}

#[test]
fn identical_loop_payload_with_guarded_trap_retains_participation_obligation() {
    let unknown = guarded_forwarded_loop_report_v1(ForwardedLoopSelectorV1::Opaque, true);
    assert_eq!(unknown.status(), KernelCheckStatusV1::Incomplete);
    assert!(unknown.findings().iter().any(|finding| matches!(
        finding,
        PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail }
            if detail.contains("unresolved branch block 1")
    )));
    let varying = guarded_forwarded_loop_report_v1(ForwardedLoopSelectorV1::Lane, true);
    assert_eq!(varying.status(), KernelCheckStatusV1::Rejected);
    assert!(varying.findings().iter().any(|finding| matches!(
        finding,
        PlironTensorLayoutFindingV1::DivergentSubgroupControl { controller: 1, .. }
    )));
}

#[test]
fn identical_loop_payload_with_reconverging_uniform_guard_remains_uniform() {
    for selector in [
        ForwardedLoopSelectorV1::Opaque,
        ForwardedLoopSelectorV1::Lane,
    ] {
        let report = guarded_forwarded_loop_report_v1(selector, false);
        assert!(report.is_clean(), "{:?}", report.findings());
    }
}

use super::*;
use dialect_kernel::{DIALECT_NAME, register_dialect};
use pliron::op::Op;
use pliron::{dialect::DialectName, operation::verify_operation, parsable::parse_from_str};
const BASE: &str = include_str!("../tests/lit/pipeline_dynamic_double_buffer.pliron");

fn parse(source: &str) -> (Context, FuncOp) {
    let mut context = Context::new();
    register_dialect(&mut context, &DialectName::try_new(DIALECT_NAME).unwrap()).unwrap();
    dialect_gpu::register_dialect(&mut context).unwrap();
    dialect_proof::register_dialect(&mut context).unwrap();
    let ir = source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let op = parse_from_str(Operation::top_level_parser(), &mut context, &ir).unwrap();
    verify_operation(op, &context).unwrap();
    (context, FuncOp::from_operation(op))
}

fn fixture(trip: u64) -> String {
    let begin = BASE
        .find("    kernel.access (v0, current_slot_v10,")
        .unwrap();
    let end = BASE.find("  ^exit_block4v1").unwrap();
    let region = r#"
    kernel.br_args (body_i_v7, zero_v2) [^depth_header] []: <(kernel.index, kernel.index) -> ()>
  ^depth_header(depth_phase: kernel.index, depth_iv: kernel.index):
    kernel.index_lt_br_args (depth_iv, child_bound, depth_phase, depth_iv, depth_phase) [^depth_body, ^outer_latch] []: <(kernel.index, kernel.index, kernel.index, kernel.index, kernel.index) -> ()>
  ^depth_body(read_phase: kernel.index, read_depth: kernel.index):
    read_slot = kernel.index_binary (read_phase, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.access (v0, read_slot, zero_v2) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <16,true,[2,256]>, kernel.index, kernel.index) -> ()>;
    kernel.br_args (read_phase, read_depth, zero_v2) [^component_header] []: <(kernel.index, kernel.index, kernel.index) -> ()>
  ^component_header(component_phase: kernel.index, component_depth: kernel.index, component_iv: kernel.index):
    kernel.index_lt_br_args (component_iv, child_bound, component_phase, component_depth, component_iv, component_phase, component_depth) [^component_body, ^depth_latch] []: <(kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index) -> ()>
  ^component_body(math_phase: kernel.index, math_depth: kernel.index, math_iv: kernel.index):
    math_value = kernel.index_binary (math_iv, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Multiply]: <(kernel.index, kernel.index) -> (kernel.index)>;
    next_component = kernel.index_binary (math_iv, one_v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.br_args (math_phase, math_depth, next_component) [^component_header] []: <(kernel.index, kernel.index, kernel.index) -> ()>
  ^depth_latch(latch_phase: kernel.index, latch_depth: kernel.index):
    next_depth = kernel.index_binary (latch_depth, one_v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.br_args (latch_phase, next_depth) [^depth_header] []: <(kernel.index, kernel.index) -> ()>
  ^outer_latch(release_phase: kernel.index):
    release_slot = kernel.index_binary (release_phase, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.pipeline_event (p1, release_phase, release_slot) [] [kernel_pipeline_event_kind: kernel.pipeline_event_kind Release]: <(kernel.pipeline <2,1>, kernel.index, kernel.index) -> ()>;
    next_outer = kernel.index_binary (release_phase, one_v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.br_args (next_outer) [^header_block2v1] []: <(kernel.index) -> ()>

"#;
    let source = format!("{}{}{}", &BASE[..begin], region, &BASE[end..]);
    source.replace("    p1 = ", &format!("    child_bound = kernel.index_constant () [] [kernel_index_value: kernel.index_value {trip}]: <() -> (kernel.index)>;\n    p1 = "))
}

fn check(source: &str, accepted: bool) {
    let (context, function) = parse(source);
    let report = run_pliron_pipeline_protocol_check_v1(&context, &function);
    assert_eq!(report.is_clean(), accepted, "{report:?}");
    if accepted {
        assert_eq!(report.certificates().len(), 1);
        assert!(report.certificates()[0].dynamic_loop().is_some());
        assert!(report.certificates()[0].access_refinement_proven());
    } else {
        assert!(report.certificates().is_empty());
    }
}

#[test]
fn actual_two_and_three_argument_child_shape_zero_one_four() {
    for trip in [0, 1, 4] {
        check(&fixture(trip), true);
    }
}

#[test]
fn nested_authority_effects_and_writes_cannot_repeat() {
    let source = fixture(4);
    let marker = "    next_component = ";
    for extra in [
        "kernel.pipeline_event (p1, math_phase, zero_v2) [] [kernel_pipeline_event_kind: kernel.pipeline_event_kind Release]: <(kernel.pipeline <2,1>, kernel.index, kernel.index) -> ()>;",
        "kernel.pipeline_event (p1, math_phase, zero_v2) [] [kernel_pipeline_event_kind: kernel.pipeline_event_kind Stage]: <(kernel.pipeline <2,1>, kernel.index, kernel.index) -> ()>;",
        "gpu.barrier Workgroup Workgroup Workgroup AcquireRelease;",
        "hidden_pipeline = kernel.pipeline_create (v0) [] []: <(kernel.ranked_view <16,true,[2,256]>) -> (kernel.pipeline <2,1>)>;",
        "kernel.access (v0, zero_v2, zero_v2) [] [kernel_access_kind: kernel.access_kind Write]: <(kernel.ranked_view <16,true,[2,256]>, kernel.index, kernel.index) -> ()>;",
    ] {
        check(
            &source.replace(marker, &format!("    {extra}\n{marker}")),
            false,
        );
    }
}

#[test]
fn child_trip_and_outer_phase_must_be_exact_on_every_edge() {
    let source = fixture(4);
    for bad in [
        source.replace("(component_iv, child_bound,", "(component_iv, bound_v5,"),
        source.replace("(math_iv, one_v3)", "(math_iv, two_v4)"),
        source.replace("(latch_depth, one_v3)", "(latch_depth, zero_v2)"),
        source.replace("(latch_phase, next_depth)", "(next_depth, next_depth)"),
        source.replace(
            "(math_phase, math_depth, next_component)",
            "(math_depth, math_depth, next_component)",
        ),
        source.replace(
            "(read_phase, read_depth, zero_v2)",
            "(read_phase, read_depth, one_v3)",
        ),
    ] {
        check(&bad, false);
    }
}

#[test]
fn nested_read_requires_held_epoch_and_initialized_coordinate() {
    let source = fixture(4);
    for bad in [
        source.replace("(read_phase, two_v4)", "(read_depth, two_v4)"),
        source.replace("(v0, read_slot, zero_v2)", "(v0, read_slot, one_v3)"),
        source.replace(
            "kernel.pipeline_event_kind Consume",
            "kernel.pipeline_event_kind Discard",
        ),
    ] {
        check(&bad, false);
    }
    // A dead child cannot invent a newly initialized coordinate.
    check(
        &fixture(0).replace("(v0, read_slot, zero_v2)", "(v0, read_slot, one_v3)"),
        false,
    );
}

#[test]
fn child_side_exit_and_reentry_fail() {
    let source = fixture(4);
    let edge = "kernel.br_args (math_phase, math_depth, next_component) [^component_header] []: <(kernel.index, kernel.index, kernel.index) -> ()>";
    assert!(source.contains(edge));
    check(&source.replace(edge,
        "kernel.br_args (math_phase, math_depth) [^depth_body] []: <(kernel.index, kernel.index) -> ()>"), false);
    check(&source.replace(edge,
        "kernel.index_lt_br_args (zero_v2, two_v4, math_phase, math_depth, next_component, math_phase) [^component_header, ^outer_latch] []: <(kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index) -> ()>"), false);
}

fn guarded() -> String {
    let mut source = fixture(4);
    let begin = source.find("  ^depth_body").unwrap();
    let end = source.find("  ^component_header").unwrap();
    source.replace_range(begin..end, r#"
  ^depth_body(read_phase: kernel.index, read_depth: kernel.index):
    kernel.index_lt_br_args (zero_v2, two_v4, read_phase, read_depth, read_phase, read_depth) [^guarded_read, ^after_read] []: <(kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index) -> ()>
  ^guarded_read(guard_phase: kernel.index, guard_depth: kernel.index):
    read_slot = kernel.index_binary (guard_phase, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.access (v0, read_slot, zero_v2) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <16,true,[2,256]>, kernel.index, kernel.index) -> ()>;
    kernel.br_args (guard_phase, guard_depth) [^after_read] []: <(kernel.index, kernel.index) -> ()>
  ^after_read(after_phase: kernel.index, after_depth: kernel.index):
    kernel.br_args (after_phase, after_depth, zero_v2) [^component_header] []: <(kernel.index, kernel.index, kernel.index) -> ()>
"#);
    source
}

#[test]
fn partial_child_reads_require_initialization_without_inventing_coverage() {
    let source = guarded();
    check(&source, true);
    let no_steady_write = source
        .lines()
        .filter(|line| !line.contains("kernel.access (v0, future_slot_v9,"))
        .collect::<Vec<_>>()
        .join("\n");
    check(&no_steady_write, false);
    check(
        &source.replace("(v0, read_slot, zero_v2)", "(v0, read_slot, one_v3)"),
        false,
    );
    // The complete positive child can also be bypassed before entry.
    let source = fixture(4).replace(
        "kernel.br_args (body_i_v7, zero_v2) [^depth_header] []: <(kernel.index, kernel.index) -> ()>",
        "kernel.index_lt_br_args (zero_v2, two_v4, body_i_v7, zero_v2, body_i_v7) [^depth_header, ^outer_latch] []: <(kernel.index, kernel.index, kernel.index, kernel.index, kernel.index) -> ()>"
    ).replace("[^depth_body, ^outer_latch]", "[^depth_body, ^child_exit]").replace(
        "  ^outer_latch(release_phase:",
        "  ^child_exit(child_exit_phase: kernel.index):\n    kernel.br_args (child_exit_phase) [^outer_latch] []: <(kernel.index) -> ()>\n  ^outer_latch(release_phase:"
    );
    check(&source, true);
    let no_write = source
        .lines()
        .filter(|line| !line.contains("kernel.access (v0, future_slot_v9,"))
        .collect::<Vec<_>>()
        .join("\n");
    check(&no_write, false);
}

#[test]
fn same_target_false_edge_cannot_mutate_outer_phase() {
    let source = fixture(4);
    let edge = "kernel.br_args (math_phase, math_depth, next_component) [^component_header] []: <(kernel.index, kernel.index, kernel.index) -> ()>";
    let replacement = r#"kernel.index_lt_br_args (zero_v2, two_v4, math_phase, math_depth, math_iv, math_depth, math_depth, math_iv) [^bridge, ^bridge] []: <(kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index, kernel.index) -> ()>
  ^bridge(bridge_phase: kernel.index, bridge_depth: kernel.index, bridge_iv: kernel.index):
    next_bridge = kernel.index_binary (bridge_iv, one_v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index, kernel.index) -> (kernel.index)>;
    kernel.br_args (bridge_phase, bridge_depth, next_bridge) [^component_header] []: <(kernel.index, kernel.index, kernel.index) -> ()>"#;
    check(&source.replace(edge, replacement), false);
}

#[test]
fn nonuniform_trip_and_lifetime_effects_fail() {
    let source = fixture(4);
    let source=source.replace(
        "child_bound = kernel.index_constant () [] [kernel_index_value: kernel.index_value 4]: <() -> (kernel.index)>;",
        "child_bound = kernel.invocation_index () [] [kernel_invocation_dimension: kernel.invocation_dimension 0, kernel_launch_extent: kernel.launch_extent 64]: <() -> (kernel.index)>;");
    check(&source, false);
    let effect = "kernel.allocation_effect () [] [kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_origin: kernel.allocation_origin 0, kernel_allocation_effect_noalias_class: kernel.noalias_class 0]: <() -> ()>;";
    check(
        &fixture(4).replace(
            "    next_component = ",
            &format!("    {effect}\n    next_component = "),
        ),
        false,
    );
}

#[test]
fn nested_actual_census_resource_exact_and_one_under() {
    let (context, function) = parse(&guarded());
    let mut analyses = PlironAnalysisManagerV1::new(&function);
    analyses.prepare_function_inventory(&context, &function);
    let inventory = analyses.function_inventory_handle().unwrap();
    let census = pipeline_protocol_inventory_census_v1(&context, &inventory).unwrap();
    let bound = preflight_pipeline_protocol_resource_upper_bound_v1(
        census,
        ProductionAnalysisResourceLimitsV1::new(usize::MAX, usize::MAX),
    )
    .unwrap();
    for (work, storage, accepted) in [
        (
            bound.work_upper_bound(),
            bound.peak_storage_upper_bound(),
            true,
        ),
        (
            bound.work_upper_bound() - 1,
            bound.peak_storage_upper_bound(),
            false,
        ),
        (
            bound.work_upper_bound(),
            bound.peak_storage_upper_bound() - 1,
            false,
        ),
    ] {
        assert_eq!(
            preflight_pipeline_protocol_resource_upper_bound_v1(
                census,
                ProductionAnalysisResourceLimitsV1::new(work, storage)
            )
            .is_ok(),
            accepted
        );
    }
    let mut meter = EquivalenceResourceMeterV1::new(
        pipeline_equivalence_query_upper_bound_v1(census).unwrap(),
        pipeline_equivalence_unique_pair_upper_bound_v1(census).unwrap(),
    )
    .unwrap();
    let discovery = discover_epoch_loops(&context, &inventory, &mut meter);
    assert_eq!(discovery.loops.len(), 3);
    assert!(!meter.exhausted());
    check(&guarded(), true);
    eprintln!(
        "nested exact work={} peak={} queries={} expanded={}",
        bound.work_upper_bound(),
        bound.peak_storage_upper_bound(),
        meter.queries,
        meter.expanded_pairs
    );
}

#[test]
fn bypassable_staging_write_cannot_initialize_child_reads() {
    let source = fixture(4);
    let write = source
        .lines()
        .find(|line| line.contains("kernel.access (v0, future_slot_v9,"))
        .unwrap();
    let replacement = format!(
        r#"
    kernel.index_lt_br_args (two_v4, zero_v2, body_i_v7, body_i_v7) [^conditional_write, ^write_join] []: <(kernel.index, kernel.index, kernel.index, kernel.index) -> ()>
  ^conditional_write(write_phase: kernel.index):
{write}
    kernel.br_args (write_phase) [^write_join] []: <(kernel.index) -> ()>
  ^write_join(join_phase: kernel.index):
"#
    );
    let source = source.replace(write, &replacement);
    let begin = source.find("  ^write_join").unwrap();
    let mut tail = source[begin..]
        .replace("body_i_v7", "join_phase")
        .replace("future_v8", "join_future")
        .replace("future_slot_v9", "join_future_slot");
    tail=tail.replace("  ^write_join(join_phase: kernel.index):",
        "  ^write_join(join_phase: kernel.index):\n    join_future = kernel.index_binary (join_phase, one_v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index, kernel.index) -> (kernel.index)>;\n    join_future_slot = kernel.index_binary (join_future, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index, kernel.index) -> (kernel.index)>;");
    let source = format!("{}{}", &source[..begin], tail);

    let (context, function) = parse(&source);
    let report = run_pliron_pipeline_protocol_check_v1(&context, &function);
    assert!(
        matches!(report.findings(),
        [PlironPipelineProtocolFindingV1::InvalidSchedule{detail,..}]
        if detail=="nested epoch has a bypassable staging write"),
        "{report:?}"
    );
    assert!(report.certificates().is_empty());
}

#[test]
fn nested_division_requires_a_total_constant_divisor() {
    let source = fixture(4);
    for kind in ["Divide", "Remainder"] {
        check(
            &source.replace(
                "kernel.index_binary_kind Multiply",
                &format!("kernel.index_binary_kind {kind}"),
            ),
            true,
        );
        for divisor in ["zero_v2", "bound_v5"] {
            let bad = source
                .replace("(math_iv, two_v4)", &format!("(math_iv, {divisor})"))
                .replace(
                    "kernel.index_binary_kind Multiply",
                    &format!("kernel.index_binary_kind {kind}"),
                );
            check(&bad, false);
        }
    }
}

#[test]
fn nested_write_to_other_storage_is_not_a_read_only_child() {
    let source=fixture(4).replace("    p1 = ",
        "    other_view = kernel.ranked_view () [] [kernel_memory_space: kernel.memory_space Global]: <() -> (kernel.ranked_view <16,true,[2,256]>)>;\n    p1 = ");
    let write = "kernel.access (other_view, zero_v2, zero_v2) [] [kernel_access_kind: kernel.access_kind Write]: <(kernel.ranked_view <16,true,[2,256]>, kernel.index, kernel.index) -> ()>;";
    check(
        &source.replace(
            "    next_component = ",
            &format!("    {write}\n    next_component = "),
        ),
        false,
    );
}

#[cfg(test)]
mod full_writer_domain {
    include!("pipeline_domain_v1_tests.rs");
}

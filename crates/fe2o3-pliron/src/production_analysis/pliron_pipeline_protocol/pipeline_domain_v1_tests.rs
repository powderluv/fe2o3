use super::*;

fn domain_fixture(trip: u64) -> String {
    let source = fixture(trip).replace("ranked_view <16,true", "ranked_view <64,true");
    let source=source.replace("    p1 = ", r#"
    gpu.execution_layout () [] [gpu_execution_grid_identity: gpu.grid_identity 71, gpu_execution_global_x: gpu.execution_extent 1024, gpu_execution_global_y: gpu.execution_extent 1, gpu_execution_global_z: gpu.execution_extent 1, gpu_execution_workgroup_x: gpu.execution_extent 256, gpu_execution_workgroup_y: gpu.execution_extent 1, gpu_execution_workgroup_z: gpu.execution_extent 1, gpu_execution_subgroup_size: gpu.subgroup_size 64, gpu_execution_domain: gpu.execution_domain FullPhysicalWorkgroups]: <() -> ()>;
    invocation = kernel.invocation_index () [] [kernel_invocation_dimension: kernel.invocation_dimension 0, kernel_launch_extent: kernel.launch_extent 0]: <() -> (kernel.index)>;
    extent = kernel.index_constant () [] [kernel_index_value: kernel.index_value 256]: <() -> (kernel.index)>;
    wave = kernel.index_constant () [] [kernel_index_value: kernel.index_value 64]: <() -> (kernel.index)>;
    sixteen = kernel.index_constant () [] [kernel_index_value: kernel.index_value 16]: <() -> (kernel.index)>;
    four = kernel.index_constant () [] [kernel_index_value: kernel.index_value 4]: <() -> (kernel.index)>;
    writer = kernel.index_binary (invocation, extent) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index,kernel.index) -> (kernel.index)>;
    p1 = "#);
    let mut source = source
        .replace("(v0, zero_v2, zero_v2)", "(v0, zero_v2, writer)")
        .replace(
            "(v0, future_slot_v9, zero_v2)",
            "(v0, future_slot_v9, writer)",
        );
    let begin = source.find("  ^depth_body").unwrap();
    let end = source.find("  ^component_header").unwrap();
    source.replace_range(begin..end,r#"
  ^depth_body(read_phase: kernel.index, read_depth: kernel.index):
    wave_id = kernel.index_binary (writer, wave) [] [kernel_index_binary_kind: kernel.index_binary_kind Divide]: <(kernel.index,kernel.index) -> (kernel.index)>;
    wave_base = kernel.index_binary (wave_id, wave) [] [kernel_index_binary_kind: kernel.index_binary_kind Multiply]: <(kernel.index,kernel.index) -> (kernel.index)>;
    depth_offset = kernel.index_binary (read_depth, sixteen) [] [kernel_index_binary_kind: kernel.index_binary_kind Multiply]: <(kernel.index,kernel.index) -> (kernel.index)>;
    read_base = kernel.index_binary (wave_base, depth_offset) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index,kernel.index) -> (kernel.index)>;
    local_lane = kernel.index_binary (invocation, wave) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index,kernel.index) -> (kernel.index)>;
    row_group = kernel.index_binary (local_lane, sixteen) [] [kernel_index_binary_kind: kernel.index_binary_kind Divide]: <(kernel.index,kernel.index) -> (kernel.index)>;
    output_row = kernel.index_binary (row_group, four) [] [kernel_index_binary_kind: kernel.index_binary_kind Multiply]: <(kernel.index,kernel.index) -> (kernel.index)>;
    gather = kernel.index_binary (read_base, output_row) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index,kernel.index) -> (kernel.index)>;
    kernel.index_lt_br_args (gather, extent, read_phase, read_depth, gather, read_phase, read_depth) [^guarded_read, ^after_read] []: <(kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index) -> ()>
  ^guarded_read(guard_phase: kernel.index, guard_depth: kernel.index, guard_index: kernel.index):
    read_slot = kernel.index_binary (guard_phase, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index,kernel.index) -> (kernel.index)>;
    kernel.access (v0, read_slot, guard_index) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <64,true,[2,256]>,kernel.index,kernel.index) -> ()>;
    kernel.br_args (guard_phase, guard_depth) [^after_read] []: <(kernel.index,kernel.index) -> ()>
  ^after_read(after_phase: kernel.index, after_depth: kernel.index):
    kernel.br_args (after_phase, after_depth, zero_v2) [^component_header] []: <(kernel.index,kernel.index,kernel.index) -> ()>
"#);
    source
}

#[test]
fn exact_full_writer_domain_covers_guarded_gather_zero_one_four() {
    for trip in [0, 1, 4] {
        check(&domain_fixture(trip), true);
    }
}

#[test]
fn full_geometry_and_actual_modulus_must_match() {
    let source = domain_fixture(4);
    for (before, after) in [
        (
            "gpu_execution_workgroup_x: gpu.execution_extent 256",
            "gpu_execution_workgroup_x: gpu.execution_extent 128",
        ),
        (
            "gpu_execution_workgroup_x: gpu.execution_extent 256, gpu_execution_workgroup_y: gpu.execution_extent 1",
            "gpu_execution_workgroup_x: gpu.execution_extent 128, gpu_execution_workgroup_y: gpu.execution_extent 2",
        ),
        ("(invocation, extent)", "(invocation, wave)"),
        (
            "kernel_launch_extent: kernel.launch_extent 0",
            "kernel_launch_extent: kernel.launch_extent 256",
        ),
        (
            "kernel_invocation_dimension: kernel.invocation_dimension 0",
            "kernel_invocation_dimension: kernel.invocation_dimension 1",
        ),
        (
            "gpu.execution_domain FullPhysicalWorkgroups",
            "gpu.execution_domain PotentiallyPartial",
        ),
    ] {
        let mut bad = source.replace(before, after);
        if after.contains("gpu_execution_workgroup_y: gpu.execution_extent 2") {
            bad = bad.replace(
                "gpu_execution_global_y: gpu.execution_extent 1",
                "gpu_execution_global_y: gpu.execution_extent 2",
            );
        }
        assert!(bad != source, "mutation must change fixture: {before}");
        check(&bad, false);
    }
}

#[test]
fn every_publication_window_needs_actual_complete_writers() {
    for marker in [
        "kernel.access (v0, zero_v2, writer)",
        "kernel.access (v0, future_slot_v9, writer)",
    ] {
        let bad = domain_fixture(4)
            .lines()
            .filter(|line| !line.contains(marker))
            .collect::<Vec<_>>()
            .join("\n");
        check(&bad, false);
    }
    let source = domain_fixture(4);
    let initial = source
        .lines()
        .find(|line| line.contains("kernel.access (v0, zero_v2, writer)"))
        .unwrap();
    let conditional_initial = format!(
        r#"
    kernel.index_lt_br_args (invocation, wave) [^partial_initial, ^initial_join] []: <(kernel.index,kernel.index) -> ()>
  ^partial_initial():
{initial}
    kernel.br_args () [^initial_join] []: <() -> ()>
  ^initial_join():
"#
    );
    check(&source.replace(initial, &conditional_initial), false);
    let write = source
        .lines()
        .find(|line| line.contains("kernel.access (v0, future_slot_v9,"))
        .unwrap();
    let replacement = format!(
        r#"
    kernel.index_lt_br_args (two_v4, zero_v2, body_i_v7, body_i_v7) [^conditional_write, ^write_join] []: <(kernel.index,kernel.index,kernel.index,kernel.index) -> ()>
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
        "  ^write_join(join_phase: kernel.index):\n    join_future = kernel.index_binary (join_phase, one_v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index,kernel.index) -> (kernel.index)>;\n    join_future_slot = kernel.index_binary (join_future, two_v4) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index,kernel.index) -> (kernel.index)>;");
    check(&format!("{}{}", &source[..begin], tail), false);
}

#[test]
fn guarded_read_cannot_borrow_wrong_false_edge_or_coordinate() {
    let source = domain_fixture(4);
    for (before, after) in [
        (
            "(gather, extent, read_phase, read_depth, gather,",
            "(gather, extent, read_phase, read_depth, writer,",
        ),
        (
            "(gather, extent, read_phase, read_depth, gather,",
            "(writer, extent, read_phase, read_depth, gather,",
        ),
        (
            "kernel.access (v0, read_slot, guard_index)",
            "kernel.access (v0, read_slot, extent)",
        ),
        ("(guard_phase, two_v4)", "(guard_depth, two_v4)"),
        (
            "kernel.pipeline_event_kind Consume",
            "kernel.pipeline_event_kind Discard",
        ),
        ("kernel.index_value 256]:", "kernel.index_value 257]:"),
    ] {
        let mut bad = source.replace(before, after);
        if after.contains("gpu_execution_workgroup_y: gpu.execution_extent 2") {
            bad = bad.replace(
                "gpu_execution_global_y: gpu.execution_extent 1",
                "gpu_execution_global_y: gpu.execution_extent 2",
            );
        }
        assert!(bad != source, "mutation must change fixture: {before}");
        check(&bad, false);
    }
    let duplicate = source.replace(
        "kernel.index_lt_br_args (gather, extent, read_phase, read_depth, gather, read_phase, read_depth) [^guarded_read, ^after_read] []: <(kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index)",
        "kernel.index_lt_br_args (gather, extent, read_phase, read_depth, gather, read_phase, read_depth, gather) [^guarded_read, ^guarded_read] []: <(kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index,kernel.index)");
    assert!(duplicate != source);
    check(&duplicate, false);
    let source=source.replace(
        "kernel.index_lt_br_args (gather, extent, read_phase, read_depth, gather, read_phase, read_depth) [^guarded_read, ^after_read]",
        "kernel.index_lt_br_args (gather, extent, read_phase, read_depth, read_phase, read_depth, gather) [^after_read, ^guarded_read]");
    check(&source, false);
}

fn early_exit(source: &str, uniform: bool) -> String {
    let edge = "kernel.br_args (zero_v2) [^header_block2v1] []: <(kernel.index ) -> ()>";
    // Gate before BOTH initial staging events, not after publication.
    let begin = source
        .find("    kernel.pipeline_event (p1, zero_v2, zero_v2)")
        .unwrap();
    let mut source = source.to_owned();
    let gate = format!(
        r#"
    kernel.index_lt_br_args ({}, two_v4) [^begin_staging, ^early_return] []: <(kernel.index,kernel.index) -> ()>
  ^early_return():
    kernel.return () [] []: <() -> ()>
  ^begin_staging():
"#,
        if uniform { "bound_v5" } else { "invocation" }
    );
    source.insert_str(begin, &gate);
    assert!(source.contains(edge));
    source
}

#[test]
fn full_participation_requires_uniform_early_exit_and_finite_prefix() {
    check(&early_exit(&domain_fixture(4), true), true);
    check(&early_exit(&domain_fixture(4), false), false);
}

#[test]
fn domain_supplement_has_exact_and_one_under_resources() {
    let (context, function) = parse(&domain_fixture(4));
    let inventory = BoundedPlironFunctionInventoryV1::collect(&context, &function).unwrap();
    let census = pipeline_protocol_inventory_census_v1(&context, &inventory).unwrap();
    let setup = ProductionAnalysisResourceUpperBoundV1::checked_phase(
        ProductionAnalysisResourcePhaseV1::StructuralIdentity,
        7,
        11,
        13,
    )
    .unwrap();
    let mut probe = PlironAnalysisManagerV1::new_with_resource_contract(
        &function,
        census,
        setup,
        0,
        ProductionAnalysisResourceLimitsV1::production_hard_ceiling(),
    )
    .unwrap();
    let report =
        run_pliron_pipeline_protocol_check_with_analyses_v1(&context, &function, &mut probe);
    assert!(report.is_clean(), "{report:?}");
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
        let mut manager = PlironAnalysisManagerV1::new_with_resource_contract(
            &function,
            census,
            setup,
            0,
            ProductionAnalysisResourceLimitsV1::new(work, peak),
        )
        .unwrap();
        let report =
            run_pliron_pipeline_protocol_check_with_analyses_v1(&context, &function, &mut manager);
        assert_eq!(report.is_clean(), accepted, "{report:?}");
        if accepted {
            assert_eq!(manager.resource_upper_bound(), exact);
            assert_eq!(report.certificates().len(), 1);
        } else {
            assert!(report.certificates().is_empty());
            assert!(
                report.findings().iter().any(|finding| matches!(finding,
                PlironPipelineProtocolFindingV1::AnalysisIncomplete { detail }
                if detail.contains("writer-domain refinement"))),
                "{report:?}"
            );
            let before = manager.resource_upper_bound();
            // A refusal leaves the supplementary reservation uncommitted.
            assert!(before.work_upper_bound() < exact.work_upper_bound());
            assert!(before.peak_storage_upper_bound() < exact.peak_storage_upper_bound());
        }
    }
    eprintln!(
        "actual domain ledger work={} peak={}",
        exact.work_upper_bound(),
        exact.peak_storage_upper_bound()
    );
}

#[test]
fn coverage_cannot_cross_storage_owner_or_epoch() {
    let source = domain_fixture(4);
    let source=source.replace("    p1 = ",
        "    other = kernel.ranked_view () [] [kernel_memory_space: kernel.memory_space Workgroup]: <() -> (kernel.ranked_view <64,true,[2,256]>)>;\n    p1 = ");
    for bad in [
        source.replace(
            "kernel.access (v0, zero_v2, writer)",
            "kernel.access (other, zero_v2, writer)",
        ),
        source.replace(
            "kernel.access (v0, future_slot_v9, writer)",
            "kernel.access (other, future_slot_v9, writer)",
        ),
        source
            .replace(
                "kernel.pipeline_create (v0)",
                "kernel.pipeline_create (other)",
            )
            .replace(
                "kernel.access (v0, read_slot, guard_index)",
                "kernel.access (other, read_slot, guard_index)",
            ),
        source.replace(
            "(p1, future_v8, future_slot_v9)",
            "(p1, body_i_v7, future_slot_v9)",
        ),
        source.replace(
            "(p1, body_i_v7, current_slot_v10)",
            "(p1, future_v8, current_slot_v10)",
        ),
    ] {
        assert!(bad != source);
        check(&bad, false);
    }
}

fn finite_prelude(source: &str, step: &str) -> String {
    let source=source.replace("    p1 = ",
        "    init_bound = kernel.index_constant () [] [kernel_index_value: kernel.index_value 2880]: <() -> (kernel.index)>;\n    p1 = ");
    let begin = source
        .find("    kernel.pipeline_event (p1, zero_v2, zero_v2)")
        .unwrap();
    let loop_source = format!(
        r#"
    kernel.br_args (zero_v2) [^init_header] []: <(kernel.index) -> ()>
  ^init_header(init_iv: kernel.index):
    kernel.index_lt_br_args (init_iv, init_bound, init_iv) [^init_body, ^begin_staging] []: <(kernel.index,kernel.index,kernel.index) -> ()>
  ^init_body(body_iv: kernel.index):
    init_next = kernel.index_binary (body_iv, {step}) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index,kernel.index) -> (kernel.index)>;
    kernel.br_args (init_next) [^init_header] []: <(kernel.index) -> ()>
  ^begin_staging():
"#
    );
    let mut source = source;
    source.insert_str(begin, &loop_source);
    source
}

#[test]
fn existing_finite_read_only_prefix_loop_is_required() {
    check(&finite_prelude(&domain_fixture(4), "one_v3"), true);
    check(&finite_prelude(&domain_fixture(4), "zero_v2"), false);
}

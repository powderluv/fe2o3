// Include only as a #[cfg(test)] child of the private Wave64 context module.
// These are actual KIR/context/LLVM-construction controls, not Rust/ISA receipts.
use super::*;
use fe2o3_kernel_ir::{IntrinsicOperation, ValueDef, gfx950_xnack_minus_target_capability};

fn value(id: u32, ty: Type, kind: OperationKind) -> Operation {
    Operation::effect_free(ValueDef::new(ValueId(id), ty), kind)
}

fn block(id: u32, operations: Vec<Operation>, terminator: Terminator) -> BasicBlock {
    let mut block = BasicBlock::new(BlockId(id));
    block.operations = operations;
    block.terminator = Some(terminator);
    block
}

fn call(id: u32, callee: &str, arguments: &[u32]) -> Operation {
    value(
        id,
        Type::F32,
        OperationKind::Call {
            callee: FunctionId::new(callee),
            arguments: arguments.iter().copied().map(ValueId).collect(),
        },
    )
}

fn finish(module: &mut Module) {
    let target = gfx950_xnack_minus_target_capability();
    let mut all = BTreeSet::from([target.clone()]);
    for function in &mut module.functions {
        // Preserve exact reserved declaration capabilities from its catalog.
        if function.body.is_some() {
            function.required_capabilities = function.derived_capabilities();
            function.required_capabilities.insert(target.clone());
        }
        all.extend(function.required_capabilities.iter().cloned());
    }
    module.required_capabilities = all.clone();
    for kernel in &mut module.kernels {
        kernel.required_capabilities = all.clone();
    }
}

fn module() -> Module {
    let pure = Function::internal_helper(
        "pure_add",
        Signature::new(vec![Type::F32, Type::F32], vec![Type::F32]),
        vec![ValueId(0), ValueId(1)],
        vec![block(
            0,
            vec![value(
                2,
                Type::F32,
                OperationKind::Binary {
                    op: BinaryOp::Add,
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
            )],
            Terminator::Return {
                values: vec![ValueId(2)],
            },
        )],
    );
    let leaf = Function::internal_helper(
        "wave_leaf",
        Signature::new(
            vec![Type::F32, Type::Scalar(ScalarType::U32)],
            vec![Type::F32],
        ),
        vec![ValueId(0), ValueId(1)],
        vec![block(
            0,
            vec![
                value(
                    2,
                    Type::Scalar(ScalarType::U32),
                    OperationKind::Cast {
                        kind: CastKind::Bitcast,
                        value: ValueId(0),
                        to: Type::Scalar(ScalarType::U32),
                    },
                ),
                value(
                    3,
                    Type::Scalar(ScalarType::U32),
                    OperationKind::Wave(WaveOperation::full(
                        WaveOperationKind::ShuffleIndex {
                            value: ValueId(2),
                            source_lane: ValueId(1),
                            tile_width: 64,
                        },
                        WaveWidth::Wave64,
                    )),
                ),
                value(
                    4,
                    Type::F32,
                    OperationKind::Cast {
                        kind: CastKind::Bitcast,
                        value: ValueId(3),
                        to: Type::F32,
                    },
                ),
            ],
            Terminator::Return {
                values: vec![ValueId(4)],
            },
        )],
    );
    let entry = block(
        0,
        vec![
            value(
                2,
                Type::Scalar(ScalarType::U32),
                OperationKind::Constant(Constant::U32(32)),
            ),
            value(
                3,
                Type::BOOL,
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThan,
                    lhs: ValueId(1),
                    rhs: ValueId(2),
                },
            ),
        ],
        Terminator::ConditionalBranch {
            condition: ValueId(3),
            then_target: BlockId(1),
            then_arguments: vec![],
            else_target: BlockId(2),
            else_arguments: vec![],
        },
    );
    let varying_add = block(
        1,
        vec![call(4, "pure_add", &[0, 0])],
        Terminator::Branch {
            target: BlockId(3),
            arguments: vec![ValueId(4)],
        },
    );
    let other = block(
        2,
        vec![],
        Terminator::Branch {
            target: BlockId(3),
            arguments: vec![ValueId(0)],
        },
    );
    let mut joined = block(
        3,
        vec![call(6, "wave_leaf", &[5, 1])],
        Terminator::Return {
            values: vec![ValueId(6)],
        },
    );
    joined.parameters.push(ValueDef::new(ValueId(5), Type::F32));
    let dispatch = Function::internal_helper(
        "wave_dispatch",
        Signature::new(
            vec![Type::F32, Type::Scalar(ScalarType::U32)],
            vec![Type::F32],
        ),
        vec![ValueId(0), ValueId(1)],
        vec![entry, varying_add, other, joined],
    );
    let root = Function::kernel_entry(
        "entry",
        Signature::new(vec![Type::F32], vec![]),
        vec![ValueId(0)],
        vec![block(
            0,
            vec![
                value(
                    1,
                    Type::Scalar(ScalarType::U32),
                    OperationKind::Wave(WaveOperation::full(
                        WaveOperationKind::LaneId,
                        WaveWidth::Wave64,
                    )),
                ),
                call(2, "wave_dispatch", &[0, 1]),
            ],
            Terminator::Return { values: vec![] },
        )],
    );
    let mut kernel = Kernel::new(
        "wave_kernel",
        "entry",
        LaunchDomain::D1 {
            x: LaunchExtent::Static(64),
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("wave64_closed_helper_controls");
    module.functions = vec![root, dispatch, leaf, pure];
    module.kernels.push(kernel);
    finish(&mut module);
    module
}

fn checked_context(module: &Module) -> Result<Option<Context<'_>>, LoweringErrors> {
    verify_module(module).expect("control must reach the actual context with verified KIR");
    Context::check(module, LoweringTarget::Gfx950XnackMinusV1)
}

fn assert_refused(module: &Module) {
    assert!(
        checked_context(module).is_err(),
        "context accepted hostile verified KIR"
    );
    assert!(
        lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(module).is_err(),
        "final LLVM route accepted hostile verified KIR"
    );
}

#[test]
fn closed_wave64_helpers_accept_varying_pure_add_before_uniform_shuffle() {
    let module = module();
    let context = checked_context(&module)
        .unwrap()
        .expect("actual closed helper context");
    let dispatch = &module.functions[1];
    let report = context.report(&module, dispatch).unwrap();
    assert_eq!(
        report.block_control(BlockId(1)),
        fe2o3_kernel_analysis::Variation::Varying
    );
    assert!(
        report
            .block_control(BlockId(3))
            .is_uniform_for(SynchronizationScope::Subgroup)
    );
    assert_eq!(
        report.value(ValueId(6)),
        fe2o3_kernel_analysis::Variation::Varying
    );
    let llvm = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap();
    eprintln!(
        "FE2O3_WAVE64_HELPER_LLVM_CONTROL_BEGIN_V1\n{llvm}FE2O3_WAVE64_HELPER_LLVM_CONTROL_END_V1"
    );
    assert!(llvm.contains("@llvm.amdgcn.ds.bpermute"));
    assert!(llvm.contains("bitcast float") && llvm.contains("to float"));
    for name in ["wave_leaf", "wave_dispatch"] {
        let definition = llvm
            .lines()
            .find(|line| line.starts_with("define ") && line.contains(&format!("@{name}(")))
            .expect("actual helper definition");
        assert!(definition.contains("convergent"), "{definition}");
        let calls: Vec<_> = llvm
            .lines()
            .filter(|line| line.contains("call ") && line.contains(&format!("@{name}(")))
            .collect();
        assert!(!calls.is_empty());
        assert!(calls.iter().all(|line| line.contains("convergent")));
    }
    for line in llvm.lines().filter(|line| {
        (line.starts_with("define ") || line.contains("call ")) && line.contains("@pure_add(")
    }) {
        assert!(
            !line.contains("convergent"),
            "pure scalar add was marked: {line}"
        );
    }
}

#[test]
fn closed_wave64_helpers_refuse_varying_helper_formal_wave_call() {
    let mut module = module();
    module.functions[1].body.as_mut().unwrap().blocks[1]
        .operations
        .push(call(7, "wave_leaf", &[0, 1]));
    finish(&mut module);
    assert_refused(&module);
}

// A varying physical lane controls entry into the effectful call. The branch
// either rejoins without a call, exits the invocation, or terminates in Trap.
fn root_control(module: &mut Module, early_exit: bool, trap: bool, varying: bool) {
    let body = module.functions[0].body.as_mut().unwrap();
    let effectful = body.blocks[0].operations.pop().unwrap();
    body.blocks[0].operations.extend([
        value(
            3,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(32)),
        ),
        value(
            4,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: if varying { ValueId(1) } else { ValueId(3) },
                rhs: ValueId(3),
            },
        ),
    ]);
    body.blocks[0].terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut first = block(
        1,
        vec![],
        if early_exit {
            Terminator::Return { values: vec![] }
        } else {
            Terminator::Branch {
                target: BlockId(3),
                arguments: vec![],
            }
        },
    );
    if trap {
        first
            .operations
            .push(AmdGpuDiagnosticOperation::Trap.operation(None));
        first.terminator = Some(Terminator::Unreachable);
    }
    let second = block(
        2,
        vec![effectful],
        if early_exit || trap {
            Terminator::Return { values: vec![] }
        } else {
            Terminator::Branch {
                target: BlockId(3),
                arguments: vec![],
            }
        },
    );
    body.blocks.extend([first, second]);
    if !early_exit && !trap {
        body.blocks
            .push(block(3, vec![], Terminator::Return { values: vec![] }));
    }
    if trap {
        module
            .functions
            .push(AmdGpuDiagnosticOperation::Trap.declaration());
    }
    finish(module);
}

#[test]
fn closed_wave64_helpers_refuse_partial_callsite_early_return_and_final_trap() {
    for (early_exit, trap) in [(false, false), (true, false), (true, true)] {
        let mut module = module();
        root_control(&mut module, early_exit, trap, true);
        assert_refused(&module);
    }
}

#[test]
fn closed_wave64_helpers_accept_uniform_final_trap_guard() {
    let mut module = module();
    root_control(&mut module, true, true, false);
    assert!(checked_context(&module).unwrap().is_some());
    lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap();
}

#[test]
fn closed_wave64_helpers_refuse_unavailable_recursive_unused_and_foreign_callees() {
    for case in 0..4 {
        let mut module = module();
        match case {
            0 => {
                let OperationKind::Call { callee, .. } =
                    &mut module.functions[1].body.as_mut().unwrap().blocks[1].operations[0].kind
                else {
                    panic!("pure call");
                };
                *callee = FunctionId::new("unavailable");
                module.functions.push(Function::declaration(
                    "unavailable",
                    Signature::new(vec![Type::F32, Type::F32], vec![Type::F32]),
                ));
            }
            1 => module.functions[3].body.as_mut().unwrap().blocks[0]
                .operations
                .extend([
                    value(
                        4,
                        Type::Scalar(ScalarType::U32),
                        OperationKind::Constant(Constant::U32(0)),
                    ),
                    call(3, "wave_dispatch", &[0, 4]),
                ]),
            2 => module.functions.push(Function::internal_helper(
                "unused",
                Signature::new(vec![], vec![]),
                vec![],
                vec![block(0, vec![], Terminator::Return { values: vec![] })],
            )),
            3 => module.functions.push(Function::device_ffi_export(
                "foreign",
                Signature::new(vec![], vec![]),
                vec![],
                vec![block(0, vec![], Terminator::Return { values: vec![] })],
            )),
            _ => unreachable!(),
        }
        finish(&mut module);
        assert_refused(&module);
    }
}

#[test]
fn closed_wave64_helpers_refuse_workgroup32_and_preserve_lane_only_helper_refusal() {
    let mut partial = module();
    partial.kernels[0].workgroup_size = Some(WorkgroupSize::new(32, 1, 1));
    assert_refused(&partial);
    let mut lane_only = module();
    // Keep the root free of Wave operations so the legacy helper-context gate,
    // rather than an earlier root CallWithoutSummary diagnostic, is exercised.
    lane_only.functions[0].body.as_mut().unwrap().blocks[0].operations[0].kind =
        OperationKind::Constant(Constant::U32(0));
    lane_only.functions[2].body.as_mut().unwrap().blocks[0].operations[1].kind =
        OperationKind::Wave(WaveOperation::full(
            WaveOperationKind::LaneId,
            WaveWidth::Wave64,
        ));
    finish(&mut lane_only);
    assert!(checked_context(&lane_only).unwrap().is_none());
    let error = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&lane_only).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("compiler-module helpers cannot contain kernel-context wave operations")
    );
}

#[test]
fn closed_wave64_helpers_never_bypass_malformed_trap_verification() {
    let mut module = module();
    module.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(0, AmdGpuDiagnosticOperation::Trap.operation(None));
    module
        .functions
        .push(AmdGpuDiagnosticOperation::Trap.declaration());
    finish(&mut module);
    assert!(verify_module(&module).is_err());
    let error = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap_err();
    assert!(error.diagnostics().iter().any(|diagnostic| diagnostic.code
        == LoweringDiagnosticCode::InputVerification(
            fe2o3_kernel_ir::DiagnosticCode::InvalidAmdGpuDiagnosticOperation
        )));
}

#[test]
fn closed_wave64_helpers_function_envelope_accepts_1024_and_refuses_1025() {
    assert_eq!(MAX_COMPILER_MODULE_GRAPH_FUNCTIONS, 1_024);
    let mut module = module();
    assert_eq!(module.functions.len(), 4);
    let extra = MAX_COMPILER_MODULE_GRAPH_FUNCTIONS - module.functions.len();
    let forwarding = |name: &str, callee: &str| {
        Function::internal_helper(
            name,
            Signature::new(
                vec![Type::F32, Type::Scalar(ScalarType::U32)],
                vec![Type::F32],
            ),
            vec![ValueId(0), ValueId(1)],
            vec![block(
                0,
                vec![call(2, callee, &[0, 1])],
                Terminator::Return {
                    values: vec![ValueId(2)],
                },
            )],
        )
    };
    // All 1,020 new nodes belong to one rooted acyclic chain. The final node
    // reaches the existing dispatcher, shuffle leaf, and varying pure-add arm.
    for index in 0..extra {
        let name = format!("cap_node_{index:04}");
        let callee = if index + 1 == extra {
            "wave_dispatch".to_owned()
        } else {
            format!("cap_node_{:04}", index + 1)
        };
        module.functions.push(forwarding(&name, &callee));
    }
    let OperationKind::Call { callee, .. } =
        &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[1].kind
    else {
        panic!("root call");
    };
    *callee = FunctionId::new("cap_node_0000");
    finish(&mut module);
    assert_eq!(module.functions.len(), MAX_COMPILER_MODULE_GRAPH_FUNCTIONS);
    assert!(checked_context(&module).unwrap().is_some());

    // Preserve complete reachability while increasing only the function count.
    module
        .functions
        .push(forwarding("cap_extra", "cap_node_0000"));
    let OperationKind::Call { callee, .. } =
        &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[1].kind
    else {
        panic!("root call");
    };
    *callee = FunctionId::new("cap_extra");
    finish(&mut module);
    assert_eq!(
        module.functions.len(),
        MAX_COMPILER_MODULE_GRAPH_FUNCTIONS + 1
    );
    let error = match checked_context(&module) {
        Err(error) => error,
        Ok(_) => panic!("function envelope accepted one additional rooted helper"),
    };
    assert!(
        error
            .to_string()
            .contains("one bounded actual kernel context")
    );
}

#[test]
fn closed_wave64_helpers_row_envelope_accepts_131072_and_refuses_131073() {
    assert_eq!(MAX_COMPILER_MODULE_CALL_EDGES, 131_072);
    let mut module = module();
    let rows = |module: &Module| -> usize {
        module
            .functions
            .iter()
            .filter_map(|function| function.body.as_ref())
            .flat_map(|body| &body.blocks)
            .map(|block| block.operations.len() + 1)
            .sum()
    };
    let baseline = rows(&module);
    assert!(baseline < MAX_COMPILER_MODULE_CALL_EDGES);
    let padding = MAX_COMPILER_MODULE_CALL_EDGES - baseline;
    // Root ids 0/1/2 are already in use. Each unused scalar constant has a
    // distinct SSA result and leaves the actual call graph/control unchanged.
    for ordinal in 0..padding {
        let id = u32::try_from(ordinal + 3).unwrap();
        module.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .push(value(
                id,
                Type::Scalar(ScalarType::U32),
                OperationKind::Constant(Constant::U32(0)),
            ));
    }
    finish(&mut module);
    assert_eq!(rows(&module), MAX_COMPILER_MODULE_CALL_EDGES);
    assert!(checked_context(&module).unwrap().is_some());

    let id = u32::try_from(padding + 3).unwrap();
    module.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .push(value(
            id,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(0)),
        ));
    finish(&mut module);
    assert_eq!(rows(&module), MAX_COMPILER_MODULE_CALL_EDGES + 1);
    let error = match checked_context(&module) {
        Err(error) => error,
        Ok(_) => panic!("row envelope accepted one additional scalar operation"),
    };
    assert!(
        error
            .to_string()
            .contains("exceeds the existing compiler-module graph envelope")
    );
}

// The same public physical-launch route used by production must compose with
// actual closed Wave64 helpers; these remain KIR/LLVM controls, not Rust receipts.
fn physical_context(module: &Module) -> CompilerPhysicalLaunchV2<'_> {
    CompilerPhysicalLaunchV2::new(
        module,
        fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx950,
        &[(&module.kernels[0], [1, 1, 1])],
    )
    .unwrap()
}

fn physical_identity(module: &Module) -> ProductionSemanticAnchorKirIdentityV1 {
    let owner = VerifiedCanonicalKernelIrV8::from_module(module.clone()).unwrap();
    ProductionSemanticAnchorKirIdentityV1::from_v8(&owner)
}

fn lower_physical(module: &Module) -> Result<String, LoweringErrors> {
    verify_module(module).expect("control must reach the actual physical/context route");
    lower_compiler_module_to_xnack_minus_llvm_ir_with_physical_launch_v2(
        module,
        physical_identity(module),
        &physical_context(module),
    )
}

#[test]
fn closed_wave64_helpers_compose_actual_physical_root_without_uniform_helper_formals() {
    let module = module();
    let physical = physical_context(&module);
    let context = Context::check_with_physical_v1(
        &module,
        LoweringTarget::Gfx950XnackMinusV1,
        Some(&physical),
    )
    .unwrap()
    .expect("same-owner physical and closed helper context");
    let root = physical.entry(&module.kernels[0]).unwrap().unwrap();
    assert!(context.matches_physical_input_v1(&module, &module.functions[0], Some(root)));
    assert!(!context.matches_physical_input_v1(&module, &module.functions[0], None));
    assert!(!context.matches_physical_input_v1(&module, &module.functions[1], Some(root)));
    assert!(context.matches_physical_input_v1(&module, &module.functions[1], None));
    // A separately reconstructed, equal-byte input is not the retained root.
    let detached = physical_context(&module);
    assert!(!context.matches_physical_input_v1(
        &module,
        &module.functions[0],
        detached.entry(&module.kernels[0]).unwrap(),
    ));
    let report = context.report(&module, &module.functions[1]).unwrap();
    assert_eq!(
        report.value(ValueId(0)),
        fe2o3_kernel_analysis::Variation::Varying
    );
    assert_eq!(
        report.block_control(BlockId(1)),
        fe2o3_kernel_analysis::Variation::Varying
    );
    assert!(
        report
            .block_control(BlockId(3))
            .is_uniform_for(SynchronizationScope::Subgroup)
    );
    let llvm = lower_compiler_module_to_xnack_minus_llvm_ir_with_physical_launch_v2(
        &module,
        physical_identity(&module),
        &physical,
    )
    .unwrap();
    assert!(llvm.contains("@llvm.amdgcn.ds.bpermute"));
    for name in ["wave_leaf", "wave_dispatch"] {
        let definition = llvm
            .lines()
            .find(|line| line.starts_with("define ") && line.contains(&format!("@{name}(")))
            .expect("actual convergent helper definition");
        assert!(definition.contains("convergent"));
        let calls: Vec<_> = llvm
            .lines()
            .filter(|line| line.contains("call ") && line.contains(&format!("@{name}(")))
            .collect();
        assert!(!calls.is_empty());
        assert!(calls.iter().all(|line| line.contains("convergent")));
    }
    for line in llvm.lines().filter(|line| {
        (line.starts_with("define ") || line.contains("call ")) && line.contains("@pure_add(")
    }) {
        assert!(!line.contains("convergent"));
    }
}

#[test]
fn closed_wave64_helpers_physical_route_refuses_foreign_roster_and_exact_target_mismatch() {
    let module = module();
    let copy = module.clone();
    assert!(
        CompilerPhysicalLaunchV2::new(
            &module,
            fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx950,
            &[(&copy.kernels[0], [1, 1, 1])],
        )
        .is_err()
    );
    let physical = physical_context(&module);
    let foreign = lower_compiler_module_to_xnack_minus_llvm_ir_with_physical_launch_v2(
        &copy,
        physical_identity(&copy),
        &physical,
    )
    .unwrap_err();
    assert!(foreign.contains(LoweringDiagnosticCode::InvalidLaunchPolicy));
    let wrong_target = CompilerPhysicalLaunchV2::new(
        &module,
        fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942,
        &[(&module.kernels[0], [1, 1, 1])],
    )
    .unwrap();
    let error = match Context::check_with_physical_v1(
        &module,
        LoweringTarget::Gfx950XnackMinusV1,
        Some(&wrong_target),
    ) {
        Err(error) => error,
        Ok(_) => panic!("foreign physical target reached closed helper admission"),
    };
    assert!(error.contains(LoweringDiagnosticCode::InvalidLaunchPolicy));
    assert!(
        lower_compiler_module_to_xnack_minus_llvm_ir_with_physical_launch_v2(
            &module,
            physical_identity(&module),
            &wrong_target,
        )
        .is_err()
    );
}

#[test]
fn closed_wave64_helpers_physical_route_keeps_divergent_partial_and_trap_refusals() {
    for (early_exit, trap) in [(false, false), (true, false), (true, true)] {
        let mut module = module();
        root_control(&mut module, early_exit, trap, true);
        assert!(lower_physical(&module).is_err());
    }
    let mut divergent = module();
    divergent.functions[1].body.as_mut().unwrap().blocks[1]
        .operations
        .push(call(7, "wave_leaf", &[0, 1]));
    finish(&mut divergent);
    assert!(lower_physical(&divergent).is_err());
    let mut partial = module();
    partial.kernels[0].workgroup_size = Some(WorkgroupSize::new(32, 1, 1));
    assert!(lower_physical(&partial).is_err());
    let mut uniform_trap = module();
    root_control(&mut uniform_trap, true, true, false);
    lower_physical(&uniform_trap).unwrap();
}

#[test]
fn closed_wave64_helpers_physical_refusal_names_actual_helper_and_analysis_diagnostic() {
    let mut module = module();
    // Verified but non-returning helper CFG: the refusal must retain the actual
    // analysis diagnostic, not collapse it into a generic helper predicate.
    module.functions[2].body.as_mut().unwrap().blocks[0].terminator = Some(Terminator::Branch {
        target: BlockId(0),
        arguments: vec![],
    });
    finish(&mut module);
    let error = lower_physical(&module).unwrap_err();
    assert!(error.contains(LoweringDiagnosticCode::UnprovenBarrierConvergence));
    assert!(error.to_string().contains("PostdominanceUnavailable"));
    assert!(error.diagnostics().iter().any(|diagnostic| {
        diagnostic.location.function.as_ref() == Some(&module.functions[2].id)
    }));
}

#[test]
fn closed_wave64_helpers_physical_root_uses_actual_grid_bound() {
    let mut module = module();
    root_control(&mut module, true, false, false);
    let root = &mut module.functions[0].body.as_mut().unwrap().blocks[0];
    root.operations.splice(
        2..2,
        [
            value(
                5,
                Type::INDEX,
                OperationKind::Intrinsic(IntrinsicOperation::new(
                    IntrinsicKind::InvocationIndex {
                        kind: IndexKind::Global,
                        axis: Axis::X,
                    },
                    Type::INDEX,
                )),
            ),
            value(6, Type::INDEX, OperationKind::Constant(Constant::Index(65))),
        ],
    );
    root.operations.last_mut().unwrap().kind = OperationKind::Compare {
        predicate: ComparePredicate::GreaterThanOrEqual,
        lhs: ValueId(5),
        rhs: ValueId(6),
    };
    finish(&mut module);
    verify_module(&module).expect("actual typed global-index guard");
    // Cutoff65 splits a later wave at global64/65. Static logical extent64
    // does not exclude that dispatch; the exact one-group [0,63] input does.
    assert!(checked_context(&module).is_err());
    lower_physical(&module).unwrap();
}

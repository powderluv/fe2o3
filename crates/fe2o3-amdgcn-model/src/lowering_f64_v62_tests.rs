use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget, CanonicalKernelIrWorkBudgetV1 as Work,
    MemoryAccess, StorageLayoutLimitsV1, ValueDef, VerifiedCanonicalKernelIrModuleV18 as Owner,
    WorkgroupMemory,
};

const TARGETS: [LoweringTarget; 3] = [
    LoweringTarget::Gfx942StrictFloatV1,
    LoweringTarget::Gfx942XnackMinusV1,
    LoweringTarget::Gfx950XnackMinusV1,
];

fn verify_llvm_if_configured(text: &str) {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let Some(opt) = std::env::var_os("FE2O3_OPT") else {
        return;
    };
    let mut child = Command::new(opt)
        .args(["-passes=verify", "-disable-output", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("FE2O3_OPT must name an executable LLVM verifier");
    let written = child.stdin.take().unwrap().write_all(text.as_bytes());
    let output = child.wait_with_output().expect("reap LLVM verifier");
    assert!(
        output.status.success(),
        "LLVM verification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    written.expect("write complete LLVM text to verifier");
}

fn value(id: u32, ty: Type, kind: OperationKind) -> Operation {
    Operation::new(vec![ValueDef::new(ValueId(id), ty)], kind)
}

fn pointer(space: KernelAddressSpace) -> Type {
    Type::pointer(Type::F64, space, AccessMode::ReadWrite)
}

fn kernel(parameters: Vec<Type>, operations: Vec<Operation>) -> Module {
    let ids = (0..parameters.len()).map(|i| ValueId(i as u32)).collect();
    let mut block = BasicBlock::new(BlockId(0));
    block.operations = operations;
    block.terminator = Some(Terminator::Return { values: vec![] });
    let mut module = Module::new("f64_basic");
    module.functions.push(Function::kernel_entry(
        "entry",
        Signature::new(parameters, vec![]),
        ids,
        vec![block],
    ));
    let mut kernel = Kernel::new(
        "kernel",
        "entry",
        LaunchDomain::D1 {
            x: LaunchExtent::Static(64),
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    module.kernels.push(kernel);
    module
}

fn bind(mut module: Module, target: LoweringTarget) -> Module {
    if let Some(name) = target.exact_target_binding() {
        let capability = TargetCapability::Extension {
            namespace: AMDGPU_EXACT_TARGET_CAPABILITY_NAMESPACE.to_owned(),
            name: name.to_owned(),
        };
        module.required_capabilities.insert(capability.clone());
        for kernel in &mut module.kernels {
            kernel.required_capabilities.insert(capability.clone());
        }
        for function in &mut module.functions {
            function.required_capabilities.insert(capability.clone());
        }
    }
    module
}

fn emit(module: &Module, target: LoweringTarget) -> Result<String, LoweringErrors> {
    lower_kernel_to_llvm_ir_for_target(
        module,
        &KernelId::new("kernel"),
        target,
        None,
        MAX_PRODUCTION_LEGACY_REPLAY_LLVM_TEXT_BYTES_V1,
    )
}

fn store(pointer: u32, value: u32, space: KernelAddressSpace) -> Operation {
    Operation::new(
        vec![],
        OperationKind::Store {
            pointer: ValueId(pointer),
            value: ValueId(value),
            access: MemoryAccess::new(space, 8),
        },
    )
}

fn scalar_module() -> Module {
    let mut operations = Vec::new();
    for (i, op) in [
        BinaryOp::Add,
        BinaryOp::Subtract,
        BinaryOp::Multiply,
        BinaryOp::Divide,
    ]
    .into_iter()
    .enumerate()
    {
        operations.push(value(
            4 + i as u32,
            Type::F64,
            OperationKind::Binary {
                op,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ));
    }
    operations.push(value(
        8,
        Type::F64,
        OperationKind::Unary {
            op: UnaryOp::Negate,
            operand: ValueId(0),
        },
    ));
    for (i, predicate) in [
        ComparePredicate::Equal,
        ComparePredicate::NotEqual,
        ComparePredicate::LessThan,
        ComparePredicate::LessThanOrEqual,
        ComparePredicate::GreaterThan,
        ComparePredicate::GreaterThanOrEqual,
    ]
    .into_iter()
    .enumerate()
    {
        operations.push(value(
            9 + i as u32,
            Type::BOOL,
            OperationKind::Compare {
                predicate,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ));
    }
    operations.push(value(
        15,
        Type::F64,
        OperationKind::Select {
            condition: ValueId(2),
            true_value: ValueId(4),
            false_value: ValueId(8),
        },
    ));
    operations.push(store(3, 15, KernelAddressSpace::Global));
    kernel(
        vec![
            Type::F64,
            Type::F64,
            Type::BOOL,
            pointer(KernelAddressSpace::Global),
        ],
        operations,
    )
}

fn memory_module() -> Module {
    let mut module = kernel(
        vec![
            Type::F64,
            Type::slice(Type::F64, KernelAddressSpace::Global, AccessMode::ReadWrite),
        ],
        vec![
            value(
                2,
                pointer(KernelAddressSpace::Global),
                OperationKind::SliceData { slice: ValueId(1) },
            ),
            value(3, Type::INDEX, OperationKind::Constant(Constant::Index(2))),
            value(
                4,
                pointer(KernelAddressSpace::Global),
                OperationKind::GetElementPointer {
                    base: ValueId(2),
                    offset: ValueId(3),
                },
            ),
            value(
                5,
                Type::F64,
                OperationKind::Load {
                    pointer: ValueId(4),
                    access: MemoryAccess::new(KernelAddressSpace::Global, 8),
                },
            ),
            value(
                6,
                pointer(KernelAddressSpace::Private),
                OperationKind::Alloca {
                    element: Type::F64,
                    count: None,
                    address_space: KernelAddressSpace::Private,
                    alignment: 8,
                },
            ),
            value(
                7,
                pointer(KernelAddressSpace::Private),
                OperationKind::Alloca {
                    element: Type::F64,
                    count: Some(ValueId(3)),
                    address_space: KernelAddressSpace::Private,
                    alignment: 8,
                },
            ),
            value(
                8,
                pointer(KernelAddressSpace::Workgroup),
                OperationKind::WorkgroupMemory(WorkgroupMemory {
                    element: Type::F64,
                    extent: WorkgroupMemoryExtent::Static(3),
                    alignment: 8,
                }),
            ),
            value(
                9,
                pointer(KernelAddressSpace::Workgroup),
                OperationKind::WorkgroupMemory(WorkgroupMemory {
                    element: Type::F64,
                    extent: WorkgroupMemoryExtent::Dynamic,
                    alignment: 8,
                }),
            ),
            store(6, 5, KernelAddressSpace::Private),
            value(
                10,
                Type::F64,
                OperationKind::Load {
                    pointer: ValueId(6),
                    access: MemoryAccess::new(KernelAddressSpace::Private, 8),
                },
            ),
            store(7, 10, KernelAddressSpace::Private),
            store(8, 10, KernelAddressSpace::Workgroup),
            value(
                11,
                Type::F64,
                OperationKind::Load {
                    pointer: ValueId(8),
                    access: MemoryAccess::new(KernelAddressSpace::Workgroup, 8),
                },
            ),
            store(4, 11, KernelAddressSpace::Global),
        ],
    );
    module.required_capabilities.extend([
        TargetCapability::WorkgroupMemory,
        TargetCapability::DynamicWorkgroupMemory,
    ]);
    module
}

#[test]
fn f64_basic_scalar_emission_is_ieee_typed_on_all_production_targets() {
    for target in TARGETS {
        let text = emit(&bind(scalar_module(), target), target).unwrap();
        for instruction in [
            "%v4 = fadd double %arg0, %arg1",
            "%v5 = fsub double %arg0, %arg1",
            "%v6 = fmul double %arg0, %arg1",
            "%v7 = fdiv double %arg0, %arg1",
            "%v8 = fneg double %arg0",
            "%v15 = select i1 %arg2, double %v4, double %v8",
            "store double %v15, ptr addrspace(1) %arg3, align 8",
        ] {
            assert!(
                text.contains(instruction),
                "{target:?}: {instruction}\n{text}"
            );
        }
        for (i, predicate) in ["oeq", "une", "olt", "ole", "ogt", "oge"]
            .into_iter()
            .enumerate()
        {
            assert!(text.contains(&format!(
                "%v{} = fcmp {predicate} double %arg0, %arg1",
                9 + i
            )));
        }
        for forbidden in [
            "fadd fast",
            "fdiv fast",
            "fcmp fast",
            "fneg fast",
            "nsz",
            "nnan",
            "ninf",
        ] {
            assert!(!text.contains(forbidden), "{forbidden}: {text}");
        }
        assert!(text.contains("\"unsafe-fp-math\"=\"false\""));
        assert!(text.contains("\"no-signed-zeros-fp-math\"=\"false\""));
        verify_llvm_if_configured(&text);
    }
}

#[test]
fn f64_global_private_counted_and_lds_memory_use_eight_byte_elements() {
    assert_eq!(amdgpu_private_element_alignment(&Type::F64), Some(8));
    assert_eq!(amdgpu_lds_element_bytes(&Type::F64), Some(8));
    for target in TARGETS {
        let text = emit(&bind(memory_module(), target), target).unwrap();
        for instruction in [
            "%v2 = getelementptr i8, ptr addrspace(1) %arg1.data, i64 0",
            "%v4 = getelementptr double, ptr addrspace(1) %v2, i64 2",
            "load double, ptr addrspace(1) %v4, align 8",
            "alloca double, align 8, addrspace(5)",
            "alloca double, i64 2, align 8, addrspace(5)",
            "internal addrspace(3) global [3 x double] undef, align 8",
            "external addrspace(3) global [0 x double], align 8",
            "store double %v5, ptr addrspace(5) %v6, align 8",
            "load double, ptr addrspace(5) %v6, align 8",
            "store double %v10, ptr addrspace(3) %v8, align 8",
            "load double, ptr addrspace(3) %v8, align 8",
        ] {
            assert!(
                text.contains(instruction),
                "{target:?}: {instruction}\n{text}"
            );
        }
        verify_llvm_if_configured(&text);
        for result in [6, 7, 8, 9] {
            let mut module = bind(memory_module(), target);
            let operation = module.functions[0].body.as_mut().unwrap().blocks[0]
                .operations
                .iter_mut()
                .find(|operation| {
                    operation
                        .results
                        .first()
                        .is_some_and(|v| v.id == ValueId(result))
                })
                .unwrap();
            let expected = match &mut operation.kind {
                OperationKind::Alloca { alignment, .. } => {
                    *alignment = 4;
                    LoweringDiagnosticCode::UnsupportedOperation
                }
                OperationKind::WorkgroupMemory(memory) => {
                    memory.alignment = 4;
                    LoweringDiagnosticCode::UnsupportedWorkgroupMemory
                }
                _ => unreachable!(),
            };
            assert!(emit(&module, target).unwrap_err().contains(expected));
        }
    }
}

#[test]
fn f64_constants_preserve_all_ieee_bit_classes_without_f32_widening() {
    let cases = [
        (0x0000_0000_0000_0000, "0x0000000000000000"),
        (0x8000_0000_0000_0000, "0x8000000000000000"),
        (0x0000_0000_0000_0001, "0x0000000000000001"),
        (0x0010_0000_0000_0000, "0x0010000000000000"),
        (0x3ff0_0000_0000_0001, "0x3FF0000000000001"),
        (0x7fef_ffff_ffff_ffff, "0x7FEFFFFFFFFFFFFF"),
        (0x7ff0_0000_0000_0000, "0x7FF0000000000000"),
        (0xfff0_0000_0000_0000, "0xFFF0000000000000"),
        (0x7ff8_0000_0000_0042, "0x7FF8000000000042"),
        (0xfff0_0000_0000_0042, "0xFFF0000000000042"),
    ];
    let mut operations = Vec::new();
    for (i, (bits, expected)) in cases.into_iter().enumerate() {
        assert_eq!(expected.len(), 18, "0x prefix plus sixteen F64 hex digits");
        let constant = Constant::F64Bits(bits);
        assert_eq!(constant_value(&constant).as_deref(), Some(expected));
        assert!(validate_constant(&constant, LoweringTarget::Baseline).is_err());
        for target in TARGETS {
            assert!(validate_constant(&constant, target).is_ok());
        }
        let id = 1 + i as u32;
        operations.push(value(id, Type::F64, OperationKind::Constant(constant)));
        operations.push(store(0, id, KernelAddressSpace::Global));
    }
    for target in TARGETS {
        let text = emit(
            &bind(
                kernel(
                    vec![pointer(KernelAddressSpace::Global)],
                    operations.clone(),
                ),
                target,
            ),
            target,
        )
        .unwrap();
        for (_, expected) in cases {
            assert!(text.contains(&format!(
                "store double {expected}, ptr addrspace(1) %arg0, align 8"
            )));
        }
        verify_llvm_if_configured(&text);
    }
    assert_eq!(
        constant_value(&Constant::F32Bits(1.0f32.to_bits())).as_deref(),
        Some("0x3FF0000000000000")
    );
    assert_eq!(
        constant_value(&Constant::F32Bits(0x7fc0_0042)).as_deref(),
        Some("bitcast (i32 2143289410 to float)")
    );
}

#[test]
fn f64_capabilities_are_target_gated_without_changing_baseline_rejections() {
    for target in TARGETS {
        for owner in 0..3 {
            let mut module = bind(scalar_module(), target);
            let capabilities = match owner {
                0 => &mut module.required_capabilities,
                1 => &mut module.kernels[0].required_capabilities,
                _ => &mut module.functions[0].required_capabilities,
            };
            capabilities.insert(TargetCapability::Float64);
            emit(&module, target).unwrap();
        }
    }
    for owner in 0..3 {
        let mut module = scalar_module();
        let capabilities = match owner {
            0 => &mut module.required_capabilities,
            1 => &mut module.kernels[0].required_capabilities,
            _ => &mut module.functions[0].required_capabilities,
        };
        capabilities.insert(TargetCapability::Float64);
        assert!(
            emit(&module, LoweringTarget::Baseline)
                .unwrap_err()
                .contains(LoweringDiagnosticCode::UnsupportedCapability)
        );
    }
    for ty in [
        Type::F64,
        pointer(KernelAddressSpace::Global),
        Type::slice(Type::F64, KernelAddressSpace::Global, AccessMode::ReadOnly),
    ] {
        let module = kernel(vec![ty], vec![]);
        assert!(
            emit(&module, LoweringTarget::Baseline)
                .unwrap_err()
                .contains(LoweringDiagnosticCode::UnsupportedType)
        );
    }
}

#[test]
fn f64_casts_remainder_and_non_arithmetic_operators_remain_closed() {
    for target in TARGETS {
        for op in [
            BinaryOp::Remainder,
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::ShiftLeft,
            BinaryOp::ShiftRight,
            BinaryOp::Checked(CheckedBinaryOperator::Add),
        ] {
            assert!(!supported_binary(op, &Type::F64, target));
        }
        assert!(!supported_unary(UnaryOp::Not, &Type::F64, target));
        let remainder = kernel(
            vec![Type::F64, Type::F64],
            vec![value(
                2,
                Type::F64,
                OperationKind::Binary {
                    op: BinaryOp::Remainder,
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
            )],
        );
        assert!(
            emit(&bind(remainder, target), target)
                .unwrap_err()
                .contains(LoweringDiagnosticCode::UnsupportedOperation)
        );
        for (from, to, kind) in [
            (
                Type::F64,
                Type::Scalar(ScalarType::I64),
                CastKind::FloatToInteger,
            ),
            (
                Type::Scalar(ScalarType::U64),
                Type::F64,
                CastKind::IntegerToFloat,
            ),
            (Type::F32, Type::F64, CastKind::FloatExtend),
            (Type::F64, Type::F32, CastKind::FloatTruncate),
            (Type::F64, Type::Scalar(ScalarType::U64), CastKind::Bitcast),
            (Type::Scalar(ScalarType::U64), Type::F64, CastKind::Bitcast),
        ] {
            assert!(validate_cast(kind, &from, &to, target).is_err());
            let module = kernel(
                vec![from],
                vec![value(
                    1,
                    to.clone(),
                    OperationKind::Cast {
                        kind,
                        value: ValueId(0),
                        to,
                    },
                )],
            );
            assert!(
                emit(&bind(module, target), target)
                    .unwrap_err()
                    .contains(LoweringDiagnosticCode::UnsupportedCast)
            );
        }
        for function in [
            F32MathFunction::Sqrt,
            F32MathFunction::Abs,
            F32MathFunction::Sin,
        ] {
            let intrinsic = FloatOperation::F32Math {
                function,
                implementation: function.required_implementation(),
                arguments: vec![ValueId(0)],
            };
            let mut module = bind(
                kernel(vec![Type::F64], vec![intrinsic.operation(ValueId(1))]),
                target,
            );
            module.functions.push(intrinsic.declaration());
            assert!(emit(&module, target).unwrap_err().contains(
                LoweringDiagnosticCode::InputVerification(VerificationDiagnosticCode::TypeMismatch)
            ));
        }
    }
}

#[test]
fn f64_helper_results_phi_and_guarded_memory_keep_double_types() {
    let mut module = kernel(
        vec![
            Type::F64,
            Type::F64,
            Type::BOOL,
            pointer(KernelAddressSpace::Global),
        ],
        vec![
            value(
                4,
                Type::F64,
                OperationKind::Call {
                    callee: FunctionId::new("helper"),
                    arguments: vec![ValueId(0), ValueId(1), ValueId(2)],
                },
            ),
            value(
                5,
                Type::F64,
                OperationKind::GuardedLoad {
                    pointer: ValueId(3),
                    predicate: ValueId(2),
                    fallback: ValueId(4),
                    access: MemoryAccess::new(KernelAddressSpace::Global, 8),
                },
            ),
            Operation::new(
                vec![],
                OperationKind::GuardedStore {
                    pointer: ValueId(3),
                    predicate: ValueId(2),
                    value: ValueId(5),
                    access: MemoryAccess::new(KernelAddressSpace::Global, 8),
                },
            ),
        ],
    );
    let mut entry = BasicBlock::new(BlockId(0));
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut left = BasicBlock::new(BlockId(1));
    left.operations.push(value(
        3,
        Type::F64,
        OperationKind::Unary {
            op: UnaryOp::Negate,
            operand: ValueId(0),
        },
    ));
    left.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(3)],
    });
    let mut right = BasicBlock::new(BlockId(2));
    right.operations.push(value(
        4,
        Type::F64,
        OperationKind::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
        },
    ));
    right.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(4)],
    });
    let mut merge = BasicBlock::new(BlockId(3));
    merge.parameters.push(ValueDef::new(ValueId(5), Type::F64));
    merge.terminator = Some(Terminator::Return {
        values: vec![ValueId(5)],
    });
    module.functions.push(Function::internal_helper(
        "helper",
        Signature::new(vec![Type::F64, Type::F64, Type::BOOL], vec![Type::F64]),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        vec![entry, left, right, merge],
    ));
    module
        .required_capabilities
        .insert(TargetCapability::WaveWidth(WaveWidth::Wave64));
    for target in TARGETS {
        let text = lower_compiler_module_to_llvm_ir_for_target(
            &bind(module.clone(), target),
            target,
            None,
            None,
            true,
        )
        .unwrap();
        for instruction in [
            "define internal double @helper(double %arg0, double %arg1, i1 %arg2)",
            "call double @helper(double %arg0, double %arg1, i1 %arg2)",
            "%v5 = phi double",
            "ret double %v5",
            "%v5.loaded = load double, ptr addrspace(1) %arg3, align 8",
            "store double %v5, ptr addrspace(1) %arg3, align 8",
        ] {
            assert!(
                text.contains(instruction),
                "{target:?}: {instruction}\n{text}"
            );
        }
        assert_eq!(text.matches("%v5 = phi double").count(), 2);
        verify_llvm_if_configured(&text);
    }
}

#[test]
fn f64_native_v18_owner_uses_the_same_scalar_emitter_and_exact_target_layout() {
    for target in [
        LoweringTarget::Gfx942XnackMinusV1,
        LoweringTarget::Gfx950XnackMinusV1,
    ] {
        let module = bind(scalar_module(), target);
        let mut work = Work::new(1_000_000_000);
        let mut budget = Budget::new(&mut work, 1_000_000_000);
        let (owner, receipt) = Owner::from_module_ref_with_verification_budget_v18(
            &module,
            StorageLayoutLimitsV1 {
                rows: 64,
                edges: 256,
                containment_depth: 32,
                object_bytes: 4096,
            },
            &mut budget,
        )
        .unwrap();
        let retained = receipt.retained_storage();
        budget.reserve_storage(retained).unwrap();
        let before = (budget.work(), budget.storage());
        let text = lower_compiler_module_to_llvm_ir_for_target(
            owner.module(),
            target,
            None,
            Some(SemanticAnchorInputV1::NativeV18(&owner)),
            true,
        )
        .unwrap();
        assert!(text.contains("fadd double %arg0, %arg1"));
        assert!(text.contains("kir-version:18"));
        let expected = format!(
            "target datalayout = \"{}\"",
            fe2o3_amd_target::PRODUCTION_AMDHSA_LLVM22_WORKER_DATA_LAYOUT_V1
        );
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("target datalayout"))
                .collect::<Vec<_>>(),
            [expected.as_str()]
        );
        verify_llvm_if_configured(&text);
        assert_eq!((budget.work(), budget.storage()), before);
        drop(owner);
        budget.release_storage(retained).unwrap();
        assert_eq!(budget.storage(), 0);
    }
}

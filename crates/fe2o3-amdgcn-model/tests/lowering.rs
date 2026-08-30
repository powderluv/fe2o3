use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use fe2o3_amdgcn_model::{
    LoweringDiagnosticCode, lower_compiler_module_to_gfx950_xnack_minus_llvm_ir,
    lower_kernel_to_gfx942_xnack_minus_llvm_ir, lower_kernel_to_gfx950_xnack_minus_llvm_ir,
    lower_kernel_to_llvm_ir,
};
use fe2o3_kernel_ir::*;

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "fe2o3-amdgcn-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn canonical_test_target(text: &str) -> Result<&str, String> {
    let mut components = text.split(':');
    let processor = components.next().unwrap_or_default();
    let suffix = processor.strip_prefix("gfx").unwrap_or_default();
    if suffix.len() < 3 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("invalid AMDGPU processor {processor:?}"));
    }

    let mut last_order = 0;
    for feature in components {
        let order = match feature {
            "sramecc+" | "sramecc-" => 1,
            "xnack+" | "xnack-" => 2,
            _ => return Err(format!("invalid AMDGPU target feature {feature:?}")),
        };
        if order <= last_order {
            return Err(format!("non-canonical AMDGPU target feature {feature:?}"));
        }
        last_order = order;
    }
    Ok(processor)
}

fn assert_occurrences(text: &str, needle: &str, expected: usize) {
    assert_eq!(
        text.matches(needle).count(),
        expected,
        "expected {expected} occurrence(s) of {needle:?} in:\n{text}"
    );
}

fn symbol_blocks(report: &str) -> Vec<&str> {
    report.split("\n  Symbol {\n").skip(1).collect()
}

fn symbol_block<'a>(report: &'a str, name: &str) -> &'a str {
    let needle = format!("    Name: {name} (");
    let matches = symbol_blocks(report)
        .into_iter()
        .filter(|block| block.contains(&needle))
        .collect::<Vec<_>>();
    let [block] = matches.as_slice() else {
        panic!(
            "expected exactly one symbol named {name:?}, found {}",
            matches.len()
        )
    };
    block
}

fn metadata(report: &str) -> &str {
    let marker = "        AMDGPU Metadata: ---\n";
    let matches = report.split(marker).skip(1).collect::<Vec<_>>();
    let [metadata] = matches.as_slice() else {
        panic!(
            "expected exactly one decoded AMDGPU metadata note, found {}",
            matches.len()
        )
    };
    metadata
        .split("\n...\n")
        .next()
        .expect("metadata terminator must follow the decoded note")
}

fn assert_metadata_argument(
    metadata: &str,
    name: &str,
    offset: u64,
    size: u64,
    value_kind: &str,
    address_space: Option<&str>,
) {
    let name_line = format!(".name:           {name}");
    let blocks = metadata
        .split("\n      - ")
        .filter(|block| block.contains(&name_line))
        .collect::<Vec<_>>();
    let [block] = blocks.as_slice() else {
        panic!(
            "expected exactly one metadata argument named {name:?}, found {}",
            blocks.len()
        )
    };
    assert!(block.contains(&format!("        .offset:         {offset}\n")));
    assert!(block.contains(&format!("        .size:           {size}\n")));
    assert!(block.contains(&format!("        .value_kind:     {value_kind}")));
    match address_space {
        Some(space) => assert!(block.contains(&format!(".address_space:  {space}\n"))),
        None => assert!(!block.contains(".address_space:")),
    }
}

fn global_slice(access: AccessMode) -> Type {
    Type::slice(Type::F32, AddressSpace::Global, access)
}

fn global_pointer(access: AccessMode) -> Type {
    Type::pointer(Type::F32, AddressSpace::Global, access)
}

fn op(result: u32, ty: Type, kind: OperationKind) -> Operation {
    Operation::effect_free(ValueDef::new(ValueId(result), ty), kind)
}

fn wave_module(width: WaveWidth) -> Module {
    let wave_op = |result, ty, kind| {
        op(
            result,
            ty,
            OperationKind::Wave(WaveOperation::full(kind, width)),
        )
    };
    let mask_type = match width {
        WaveWidth::Wave32 => Type::Scalar(ScalarType::U32),
        WaveWidth::Wave64 => Type::Scalar(ScalarType::U64),
    };
    let mut block = BasicBlock::new(BlockId(0));
    block.operations = vec![
        wave_op(3, Type::Scalar(ScalarType::U32), WaveOperationKind::LaneId),
        wave_op(
            4,
            mask_type,
            WaveOperationKind::Ballot {
                predicate: ValueId(0),
            },
        ),
        wave_op(
            5,
            Type::BOOL,
            WaveOperationKind::Any {
                predicate: ValueId(0),
            },
        ),
        wave_op(
            6,
            Type::BOOL,
            WaveOperationKind::All {
                predicate: ValueId(0),
            },
        ),
        wave_op(
            7,
            Type::Scalar(ScalarType::I32),
            WaveOperationKind::ShuffleIndex {
                value: ValueId(1),
                source_lane: ValueId(2),
                tile_width: width.lanes() / 2,
            },
        ),
    ];
    block.terminator = Some(Terminator::Return { values: vec![] });
    let mut function = Function::kernel_entry(
        "wave_impl",
        Signature::new(
            vec![
                Type::BOOL,
                Type::Scalar(ScalarType::I32),
                Type::Scalar(ScalarType::U32),
            ],
            vec![],
        ),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        vec![block],
    );
    function
        .required_capabilities
        .extend(WaveOperation::full(WaveOperationKind::LaneId, width).required_capabilities());
    let mut kernel = Kernel::new(
        "wave_kernel",
        "wave_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("tests::wave");
    module.functions.push(function);
    module.kernels.push(kernel);
    module
}

fn fill_module() -> Module {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations = vec![
        op(
            2,
            Type::INDEX,
            OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
        ),
        op(
            3,
            Type::INDEX,
            OperationKind::SliceLength { slice: ValueId(0) },
        ),
        op(
            4,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(2),
                rhs: ValueId(3),
            },
        ),
    ];
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });

    let mut body = BasicBlock::new(BlockId(1));
    body.operations = vec![
        op(
            5,
            global_pointer(AccessMode::ReadWrite),
            OperationKind::SliceData { slice: ValueId(0) },
        ),
        op(
            6,
            global_pointer(AccessMode::ReadWrite),
            OperationKind::GetElementPointer {
                base: ValueId(5),
                offset: ValueId(2),
            },
        ),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(6),
                value: ValueId(1),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
    ];
    body.terminator = Some(Terminator::Branch {
        target: BlockId(2),
        arguments: vec![],
    });

    let mut exit = BasicBlock::new(BlockId(2));
    exit.terminator = Some(Terminator::Return { values: vec![] });

    let function = Function::kernel_entry(
        "fill_impl",
        Signature::new(vec![global_slice(AccessMode::ReadWrite), Type::F32], vec![]),
        vec![ValueId(0), ValueId(1)],
        vec![entry, body, exit],
    );
    let mut kernel = Kernel::new(
        "fill",
        "fill_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));

    let mut module = Module::new("tests::fill");
    module.functions.push(function);
    module.kernels.push(kernel);
    module
}

fn private_pointer_slot_module() -> Module {
    let global_pointer = global_pointer(AccessMode::ReadOnly);
    let private_slot = Type::pointer(
        global_pointer.clone(),
        AddressSpace::Private,
        AccessMode::ReadWrite,
    );
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations = vec![
        op(
            1,
            global_pointer.clone(),
            OperationKind::SliceData { slice: ValueId(0) },
        ),
        op(
            2,
            private_slot,
            OperationKind::Alloca {
                element: global_pointer.clone(),
                count: None,
                address_space: AddressSpace::Private,
                alignment: 8,
            },
        ),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(2),
                value: ValueId(1),
                access: MemoryAccess::new(AddressSpace::Private, 8),
            },
        ),
        op(
            3,
            global_pointer.clone(),
            OperationKind::Load {
                pointer: ValueId(2),
                access: MemoryAccess::new(AddressSpace::Private, 8),
            },
        ),
        op(4, Type::INDEX, OperationKind::Constant(Constant::Index(0))),
        op(
            5,
            global_pointer,
            OperationKind::GetElementPointer {
                base: ValueId(3),
                offset: ValueId(4),
            },
        ),
        op(
            6,
            Type::F32,
            OperationKind::Load {
                pointer: ValueId(5),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
    ];
    entry.terminator = Some(Terminator::Return { values: vec![] });
    let function = Function::kernel_entry(
        "private_pointer_slot_impl",
        Signature::new(vec![global_slice(AccessMode::ReadOnly)], vec![]),
        vec![ValueId(0)],
        vec![entry],
    );
    let mut kernel = Kernel::new(
        "private_pointer_slot",
        "private_pointer_slot_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("tests::private_pointer_slot");
    module.functions.push(function);
    module.kernels.push(kernel);
    exact_gfx942_xnack_minus(module)
}

fn exact_gfx942_xnack_minus(mut module: Module) -> Module {
    let target = gfx942_xnack_minus_target_capability();
    module.required_capabilities.insert(target.clone());
    module.functions[0]
        .required_capabilities
        .insert(target.clone());
    module.kernels[0].required_capabilities.insert(target);
    module
}

fn exact_gfx950_xnack_minus(mut module: Module) -> Module {
    let target = gfx950_xnack_minus_target_capability();
    let wave = TargetCapability::WaveWidth(WaveWidth::Wave64);
    module.required_capabilities.insert(target.clone());
    module.required_capabilities.insert(wave.clone());
    module.functions[0]
        .required_capabilities
        .insert(target.clone());
    module.functions[0]
        .required_capabilities
        .insert(wave.clone());
    module.kernels[0].required_capabilities.insert(target);
    module.kernels[0].required_capabilities.insert(wave);
    module
}

fn gfx950_bf16_mfma_module() -> Module {
    let parameters = [
        vec![global_pointer(AccessMode::ReadWrite)],
        vec![Type::Scalar(ScalarType::Bf16); 8],
        vec![Type::F32; 4],
    ]
    .concat();
    let matrix = MatrixOperation::multiply_accumulate(
        [ValueId(1), ValueId(2), ValueId(3), ValueId(4)],
        [ValueId(5), ValueId(6), ValueId(7), ValueId(8)],
        [ValueId(9), ValueId(10), ValueId(11), ValueId(12)],
    )
    .with_declared_tensor_layout(TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64());
    let operation = Operation::new(
        (13..17)
            .map(|id| ValueDef::new(ValueId(id), Type::F32))
            .collect(),
        OperationKind::Matrix(matrix),
    );
    let mut operations = vec![operation];
    for index in 0..4_u32 {
        let offset = ValueId(17 + index * 2);
        let pointer = ValueId(18 + index * 2);
        operations.push(op(
            offset.0,
            Type::INDEX,
            OperationKind::Constant(Constant::Index(u64::from(index))),
        ));
        operations.push(op(
            pointer.0,
            global_pointer(AccessMode::ReadWrite),
            OperationKind::GetElementPointer {
                base: ValueId(0),
                offset,
            },
        ));
        operations.push(Operation::new(
            vec![],
            OperationKind::Store {
                pointer,
                value: ValueId(13 + index),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ));
    }
    let mut function = Function::kernel_entry(
        "gfx950_bf16_mfma_impl",
        Signature::new(parameters, vec![]),
        (0..13).map(ValueId).collect(),
        vec![BasicBlock {
            id: BlockId(0),
            parameters: vec![],
            operations,
            terminator: Some(Terminator::Return { values: vec![] }),
        }],
    );
    function.required_capabilities = function.derived_capabilities();

    let mut kernel = Kernel::new(
        "gfx950_bf16_mfma",
        "gfx950_bf16_mfma_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));

    let mut module = Module::new("tests::gfx950_bf16_mfma");
    module.functions.push(function);
    module.kernels.push(kernel);
    exact_gfx950_xnack_minus(module)
}

fn gfx950_scaled_mfma_module() -> Module {
    let parameters = [vec![Type::Scalar(ScalarType::U32); 16], vec![Type::F32; 4]].concat();
    let matrix = MatrixOperation::scaled_multiply_accumulate_fp8_e4m3(
        std::array::from_fn(|index| ValueId(index as u32)),
        std::array::from_fn(|index| ValueId(8 + index as u32)),
        std::array::from_fn(|index| ValueId(16 + index as u32)),
    )
    .with_declared_tensor_layout(
        TensorLayoutContractV1::gfx950_scaled_mfma_fp8_e4m3_f32_m16n16k128_wave64(),
    );
    let operation = Operation::new(
        (20..24)
            .map(|id| ValueDef::new(ValueId(id), Type::F32))
            .collect(),
        OperationKind::Matrix(matrix),
    );
    let mut function = Function::kernel_entry(
        "gfx950_scaled_mfma_impl",
        Signature::new(parameters, vec![]),
        (0..20).map(ValueId).collect(),
        vec![BasicBlock {
            id: BlockId(0),
            parameters: vec![],
            operations: vec![operation],
            terminator: Some(Terminator::Return { values: vec![] }),
        }],
    );
    function.required_capabilities = function.derived_capabilities();

    let mut kernel = Kernel::new(
        "gfx950_scaled_mfma",
        "gfx950_scaled_mfma_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));

    let mut module = Module::new("tests::gfx950_scaled_mfma");
    module.functions.push(function);
    module.kernels.push(kernel);
    exact_gfx950_xnack_minus(module)
}

fn gfx950_scaled_fp4_mfma_module() -> Module {
    let mut module = gfx950_scaled_mfma_module();
    let operation = &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[0];
    let OperationKind::Matrix(matrix) = &mut operation.kind else {
        panic!("expected scaled matrix operation")
    };
    matrix.kind = MatrixOperationKind::ScaledMultiplyAccumulate {
        lhs: std::array::from_fn(|index| ValueId(index as u32)),
        rhs: std::array::from_fn(|index| ValueId(8 + index as u32)),
        accumulator: std::array::from_fn(|index| ValueId(16 + index as u32)),
        profile: MatrixMultiplyProfile::fp4_e2m1_f32_m16n16k128_wave64(),
    };
    matrix.tensor_layout =
        Some(TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_f32_m16n16k128_wave64());
    module.functions[0].required_capabilities = module.functions[0].derived_capabilities();
    exact_gfx950_xnack_minus(module)
}

fn gfx950_scaled_mixed_fp4_fp8_mfma_module() -> Module {
    let mut module = gfx950_scaled_fp4_mfma_module();
    let operation = &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[0];
    let OperationKind::Matrix(matrix) = &mut operation.kind else {
        panic!("expected scaled matrix operation")
    };
    matrix.tensor_layout =
        Some(TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_fp8_e4m3_f32_m16n16k128_wave64());
    module.functions[0].required_capabilities = module.functions[0].derived_capabilities();
    exact_gfx950_xnack_minus(module)
}

fn gfx950_diagnostic_module(diagnostic: AmdGpuDiagnosticOperation) -> Module {
    let mut block = BasicBlock::new(BlockId(0));
    block.operations.push(diagnostic.operation(None));
    block.terminator = Some(Terminator::Unreachable);

    let mut function = Function::kernel_entry(
        "gfx950_diagnostic_impl",
        Signature::new(vec![], vec![]),
        vec![],
        vec![block],
    );
    function.required_capabilities = function.derived_capabilities();
    let diagnostic_capabilities = function.required_capabilities.clone();

    let mut kernel = Kernel::new(
        "gfx950_diagnostic",
        "gfx950_diagnostic_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    kernel
        .required_capabilities
        .extend(diagnostic_capabilities.iter().cloned());

    let declaration = diagnostic.declaration();
    let mut module = Module::new("tests::gfx950_diagnostic");
    module.required_capabilities.extend(diagnostic_capabilities);
    module.functions.push(function);
    module.functions.push(declaration);
    module.kernels.push(kernel);
    exact_gfx950_xnack_minus(module)
}

fn sqrt_module() -> Module {
    let sqrt = FloatOperation::F32Math {
        function: F32MathFunction::Sqrt,
        implementation: F32MathImplementation::IeeeSqrtRoundTiesEvenIgnoreExceptionsV1,
        arguments: vec![ValueId(0)],
    };
    let mut block = BasicBlock::new(BlockId(0));
    block.operations.push(sqrt.operation(ValueId(1)));
    block.terminator = Some(Terminator::Return { values: vec![] });

    let mut function = Function::kernel_entry(
        "sqrt_impl",
        Signature::new(vec![Type::F32], vec![]),
        vec![ValueId(0)],
        vec![block],
    );
    function.required_capabilities = function.derived_capabilities();
    let capabilities = function.required_capabilities.clone();

    let mut kernel = Kernel::new(
        "sqrt_kernel",
        "sqrt_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    kernel
        .required_capabilities
        .extend(capabilities.iter().cloned());

    let mut module = Module::new("tests::sqrt");
    module.required_capabilities.extend(capabilities);
    module.functions.push(function);
    module.functions.push(sqrt.declaration());
    module.kernels.push(kernel);
    module
}

fn phi_loop_module() -> Module {
    let slice = global_slice(AccessMode::ReadOnly);
    let pointer = global_pointer(AccessMode::ReadOnly);

    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.push(op(
        2,
        pointer.clone(),
        OperationKind::SliceData { slice: ValueId(0) },
    ));
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(1), ValueId(0), ValueId(2)],
    });

    let mut loop_header = BasicBlock::new(BlockId(1));
    loop_header.parameters = vec![
        ValueDef::new(ValueId(10), Type::INDEX),
        ValueDef::new(ValueId(11), slice.clone()),
        ValueDef::new(ValueId(12), pointer),
    ];
    loop_header.operations = vec![
        op(
            13,
            Type::INDEX,
            OperationKind::SliceLength { slice: ValueId(11) },
        ),
        op(14, Type::INDEX, OperationKind::Constant(Constant::Index(1))),
        op(
            15,
            Type::INDEX,
            OperationKind::Binary {
                op: BinaryOp::Add,
                lhs: ValueId(10),
                rhs: ValueId(14),
            },
        ),
        op(
            16,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(15),
                rhs: ValueId(13),
            },
        ),
    ];
    loop_header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(16),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(15), ValueId(11), ValueId(12)],
        else_target: BlockId(2),
        else_arguments: vec![],
    });

    let mut exit = BasicBlock::new(BlockId(2));
    exit.terminator = Some(Terminator::Return { values: vec![] });

    let function = Function::kernel_entry(
        "phi_loop_impl",
        Signature::new(vec![slice, Type::INDEX], vec![]),
        vec![ValueId(0), ValueId(1)],
        vec![entry, loop_header, exit],
    );
    let mut kernel = Kernel::new(
        "phi_loop",
        "phi_loop_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));

    let mut module = Module::new("tests::phi_loop");
    module.functions.push(function);
    module.kernels.push(kernel);
    module
}

fn vecadd_module() -> Module {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations = vec![
        op(
            3,
            Type::INDEX,
            OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
        ),
        op(4, Type::INDEX, OperationKind::Constant(Constant::Index(0))),
        op(
            5,
            Type::INDEX,
            OperationKind::Binary {
                op: BinaryOp::Add,
                lhs: ValueId(3),
                rhs: ValueId(4),
            },
        ),
        op(
            6,
            Type::INDEX,
            OperationKind::SliceLength { slice: ValueId(2) },
        ),
        op(
            7,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(3),
                rhs: ValueId(6),
            },
        ),
        op(
            8,
            global_pointer(AccessMode::ReadWrite),
            OperationKind::SliceData { slice: ValueId(2) },
        ),
        op(
            9,
            global_pointer(AccessMode::ReadWrite),
            OperationKind::GetElementPointer {
                base: ValueId(8),
                offset: ValueId(3),
            },
        ),
    ];
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(7),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(4),
        else_arguments: vec![],
    });

    let mut first_bounds = BasicBlock::new(BlockId(1));
    first_bounds.operations = vec![
        op(
            10,
            Type::INDEX,
            OperationKind::SliceLength { slice: ValueId(0) },
        ),
        op(
            11,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(5),
                rhs: ValueId(10),
            },
        ),
    ];
    first_bounds.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(11),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(5),
        else_arguments: vec![],
    });

    let mut second_bounds = BasicBlock::new(BlockId(2));
    second_bounds.operations = vec![
        op(
            12,
            global_pointer(AccessMode::ReadOnly),
            OperationKind::SliceData { slice: ValueId(0) },
        ),
        op(
            13,
            global_pointer(AccessMode::ReadOnly),
            OperationKind::GetElementPointer {
                base: ValueId(12),
                offset: ValueId(5),
            },
        ),
        op(
            14,
            Type::F32,
            OperationKind::Load {
                pointer: ValueId(13),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
        op(
            15,
            Type::INDEX,
            OperationKind::SliceLength { slice: ValueId(1) },
        ),
        op(
            16,
            Type::BOOL,
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(5),
                rhs: ValueId(15),
            },
        ),
    ];
    second_bounds.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(16),
        then_target: BlockId(3),
        then_arguments: vec![],
        else_target: BlockId(5),
        else_arguments: vec![],
    });

    let mut compute = BasicBlock::new(BlockId(3));
    compute.operations = vec![
        op(
            17,
            global_pointer(AccessMode::ReadOnly),
            OperationKind::SliceData { slice: ValueId(1) },
        ),
        op(
            18,
            global_pointer(AccessMode::ReadOnly),
            OperationKind::GetElementPointer {
                base: ValueId(17),
                offset: ValueId(5),
            },
        ),
        op(
            19,
            Type::F32,
            OperationKind::Load {
                pointer: ValueId(18),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
        op(
            20,
            Type::F32,
            OperationKind::Binary {
                op: BinaryOp::Add,
                lhs: ValueId(14),
                rhs: ValueId(19),
            },
        ),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(9),
                value: ValueId(20),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
    ];
    compute.terminator = Some(Terminator::Branch {
        target: BlockId(4),
        arguments: vec![],
    });

    let mut exit = BasicBlock::new(BlockId(4));
    exit.terminator = Some(Terminator::Return { values: vec![] });
    let mut trap = BasicBlock::new(BlockId(5));
    trap.terminator = Some(Terminator::Unreachable);

    let function = Function::kernel_entry(
        "vecadd_impl",
        Signature::new(
            vec![
                global_slice(AccessMode::ReadOnly),
                global_slice(AccessMode::ReadOnly),
                global_slice(AccessMode::ReadWrite),
            ],
            vec![],
        ),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        vec![entry, first_bounds, second_bounds, compute, exit, trap],
    );
    let mut kernel = Kernel::new(
        "vecadd",
        "vecadd_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(256, 1, 1));

    let mut module = Module::new("tests::vecadd");
    module.functions.push(function);
    module.kernels.push(kernel);
    module
}

fn first_code(module: &Module, kernel: &str) -> LoweringDiagnosticCode {
    lower_kernel_to_llvm_ir(module, &KernelId::new(kernel))
        .unwrap_err()
        .diagnostics()[0]
        .code
}

fn g4_synchronization_module() -> Module {
    let mut module = fill_module();
    module.required_capabilities.extend([
        TargetCapability::WorkgroupMemory,
        TargetCapability::DynamicWorkgroupMemory,
        TargetCapability::WorkgroupBarrier,
        TargetCapability::WaveWidth(WaveWidth::Wave64),
    ]);
    let operations = &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations;
    operations.splice(
        0..0,
        [
            op(
                30,
                Type::pointer(
                    Type::Scalar(ScalarType::U32),
                    AddressSpace::Workgroup,
                    AccessMode::ReadWrite,
                ),
                OperationKind::WorkgroupMemory(WorkgroupMemory {
                    element: Type::Scalar(ScalarType::U32),
                    extent: WorkgroupMemoryExtent::Static(64),
                    alignment: 16,
                }),
            ),
            op(
                31,
                Type::pointer(Type::F32, AddressSpace::Workgroup, AccessMode::ReadWrite),
                OperationKind::WorkgroupMemory(WorkgroupMemory {
                    element: Type::F32,
                    extent: WorkgroupMemoryExtent::Dynamic,
                    alignment: 16,
                }),
            ),
            op(
                32,
                Type::Scalar(ScalarType::U32),
                OperationKind::Constant(Constant::U32(7)),
            ),
            Operation::new(
                vec![],
                OperationKind::Store {
                    pointer: ValueId(30),
                    value: ValueId(32),
                    access: MemoryAccess::new(AddressSpace::Workgroup, 4),
                },
            ),
            Operation::new(
                vec![],
                OperationKind::Fence(Fence {
                    memory_scope: SynchronizationScope::Device,
                    semantics: BarrierSemantics::new(
                        MemoryOrdering::Release,
                        [AddressSpace::Global],
                    ),
                }),
            ),
            Operation::new(
                vec![],
                OperationKind::WorkgroupBarrier(WorkgroupBarrier {
                    memory_scope: SynchronizationScope::Workgroup,
                    semantics: BarrierSemantics::new(
                        MemoryOrdering::AcquireRelease,
                        [AddressSpace::Workgroup],
                    ),
                    convergence: Convergence::uniform(SynchronizationScope::Workgroup),
                }),
            ),
        ],
    );
    module
}

fn workgroup_barrier(
    memory_scope: SynchronizationScope,
    ordering: MemoryOrdering,
    address_spaces: impl IntoIterator<Item = AddressSpace>,
) -> Operation {
    Operation::new(
        vec![],
        OperationKind::WorkgroupBarrier(WorkgroupBarrier {
            memory_scope,
            semantics: BarrierSemantics::new(ordering, address_spaces),
            convergence: Convergence::uniform(SynchronizationScope::Workgroup),
        }),
    )
}

fn barrier_only_module(memory_scope: SynchronizationScope, ordering: MemoryOrdering) -> Module {
    let mut module = fill_module();
    module
        .required_capabilities
        .insert(TargetCapability::WorkgroupBarrier);
    module.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            workgroup_barrier(memory_scope, ordering, [AddressSpace::Global]),
        );
    module
}

fn atomic(
    kind: AtomicKind,
    pointer: ValueId,
    address_space: AddressSpace,
    scope: SynchronizationScope,
    ordering: MemoryOrdering,
) -> Atomic {
    Atomic {
        kind,
        pointer,
        value: (kind != AtomicKind::Load).then_some(ValueId(1)),
        compare: (kind == AtomicKind::CompareExchange).then_some(ValueId(2)),
        access: MemoryAccess::new(address_space, 4),
        scope,
        ordering,
        failure_ordering: (kind == AtomicKind::CompareExchange).then_some(MemoryOrdering::Acquire),
    }
}

fn atomic_result(result: u32, atomic: Atomic) -> Operation {
    op(
        result,
        Type::Scalar(ScalarType::U32),
        OperationKind::Atomic(atomic),
    )
}

fn scoped_atomics_module() -> Module {
    let u32_type = Type::Scalar(ScalarType::U32);
    let global_slice = Type::slice(
        u32_type.clone(),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let global_pointer = Type::pointer(
        u32_type.clone(),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let workgroup_pointer = Type::pointer(
        u32_type.clone(),
        AddressSpace::Workgroup,
        AccessMode::ReadWrite,
    );

    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations = vec![
        op(
            3,
            global_pointer,
            OperationKind::SliceData { slice: ValueId(0) },
        ),
        op(
            4,
            workgroup_pointer,
            OperationKind::WorkgroupMemory(WorkgroupMemory {
                element: u32_type.clone(),
                extent: WorkgroupMemoryExtent::Static(1),
                alignment: 4,
            }),
        ),
        atomic_result(
            5,
            atomic(
                AtomicKind::Load,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Workgroup,
                MemoryOrdering::Relaxed,
            ),
        ),
        Operation::new(
            vec![],
            OperationKind::Atomic(atomic(
                AtomicKind::Store,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Device,
                MemoryOrdering::Release,
            )),
        ),
        atomic_result(
            6,
            atomic(
                AtomicKind::Exchange,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::System,
                MemoryOrdering::SequentiallyConsistent,
            ),
        ),
        Operation::new(
            vec![
                ValueDef::new(ValueId(7), u32_type.clone()),
                ValueDef::new(ValueId(8), Type::BOOL),
            ],
            OperationKind::Atomic(atomic(
                AtomicKind::CompareExchange,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Device,
                MemoryOrdering::AcquireRelease,
            )),
        ),
        atomic_result(
            9,
            atomic(
                AtomicKind::Add,
                ValueId(4),
                AddressSpace::Workgroup,
                SynchronizationScope::Workgroup,
                MemoryOrdering::Relaxed,
            ),
        ),
        atomic_result(
            10,
            atomic(
                AtomicKind::Subtract,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Workgroup,
                MemoryOrdering::Relaxed,
            ),
        ),
        atomic_result(
            11,
            atomic(
                AtomicKind::Min,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Device,
                MemoryOrdering::Acquire,
            ),
        ),
        atomic_result(
            12,
            atomic(
                AtomicKind::Max,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Device,
                MemoryOrdering::Release,
            ),
        ),
        atomic_result(
            13,
            atomic(
                AtomicKind::BitAnd,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::System,
                MemoryOrdering::AcquireRelease,
            ),
        ),
        atomic_result(
            14,
            atomic(
                AtomicKind::BitOr,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::System,
                MemoryOrdering::SequentiallyConsistent,
            ),
        ),
        atomic_result(
            15,
            atomic(
                AtomicKind::BitXor,
                ValueId(3),
                AddressSpace::Global,
                SynchronizationScope::Workgroup,
                MemoryOrdering::Relaxed,
            ),
        ),
    ];
    entry.terminator = Some(Terminator::Return { values: vec![] });

    let function = Function::kernel_entry(
        "scoped_atomics_impl",
        Signature::new(vec![global_slice, u32_type.clone(), u32_type], vec![]),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        vec![entry],
    );
    let mut kernel = Kernel::new(
        "scoped_atomics",
        "scoped_atomics_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));

    let mut module = Module::new("tests::scoped_atomics");
    module.required_capabilities.extend([
        TargetCapability::WorkgroupMemory,
        TargetCapability::Atomic {
            width_bits: 32,
            address_space: AddressSpace::Workgroup,
            max_scope: SynchronizationScope::Workgroup,
        },
        TargetCapability::Atomic {
            width_bits: 32,
            address_space: AddressSpace::Global,
            max_scope: SynchronizationScope::System,
        },
    ]);
    module.functions.push(function);
    module.kernels.push(kernel);
    module
}

fn single_global_atomic_module(
    scalar: ScalarType,
    kind: AtomicKind,
    scope: SynchronizationScope,
    ordering: MemoryOrdering,
    failure_ordering: Option<MemoryOrdering>,
) -> Module {
    let scalar_type = Type::Scalar(scalar);
    let slice = Type::slice(
        scalar_type.clone(),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let pointer = Type::pointer(
        scalar_type.clone(),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let results = match kind {
        AtomicKind::Store => vec![],
        AtomicKind::CompareExchange => vec![
            ValueDef::new(ValueId(4), scalar_type.clone()),
            ValueDef::new(ValueId(5), Type::BOOL),
        ],
        _ => vec![ValueDef::new(ValueId(4), scalar_type.clone())],
    };

    let mut entry = BasicBlock::new(BlockId(0));
    let mut atomic = atomic(kind, ValueId(3), AddressSpace::Global, scope, ordering);
    atomic.failure_ordering = failure_ordering;
    entry.operations = vec![
        op(3, pointer, OperationKind::SliceData { slice: ValueId(0) }),
        Operation::new(results, OperationKind::Atomic(atomic)),
    ];
    entry.terminator = Some(Terminator::Return { values: vec![] });

    let function = Function::kernel_entry(
        "single_atomic_impl",
        Signature::new(vec![slice, scalar_type.clone(), scalar_type], vec![]),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        vec![entry],
    );
    let mut kernel = Kernel::new(
        "single_atomic",
        "single_atomic_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));

    let mut module = Module::new("tests::single_atomic");
    module.functions.push(function);
    module.kernels.push(kernel);
    module
}

fn direct_global_pointer_atomic_module() -> Module {
    let mut module = single_global_atomic_module(
        ScalarType::U32,
        AtomicKind::Add,
        SynchronizationScope::System,
        MemoryOrdering::Relaxed,
        None,
    );
    module.functions[0].signature.parameters[0] = Type::pointer(
        Type::Scalar(ScalarType::U32),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let operations = &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations;
    operations.remove(0);
    let OperationKind::Atomic(atomic) = &mut operations[0].kind else {
        panic!("atomic expected")
    };
    atomic.pointer = ValueId(0);
    module
}

#[test]
fn lowers_fences_convergent_barriers_static_dynamic_lds_and_wave_width() {
    let llvm =
        lower_kernel_to_llvm_ir(&g4_synchronization_module(), &KernelId::new("fill")).unwrap();

    for expected in [
        "@__fe2o3_lds_fill_30 = internal addrspace(3) global [64 x i32] undef, align 16",
        "@__fe2o3_lds_fill_31 = external addrspace(3) global [0 x float], align 16",
        "%v30 = getelementptr [64 x i32], ptr addrspace(3) @__fe2o3_lds_fill_30, i32 0, i32 0",
        "store i32 7, ptr addrspace(3) %v30, align 4",
        "fence syncscope(\"agent\") release",
        "fence syncscope(\"workgroup\") release",
        "call void @llvm.amdgcn.s.barrier()",
        "fence syncscope(\"workgroup\") acquire",
        "attributes #2 = { convergent nounwind }",
        "\"target-features\"=\"-wavefrontsize32,+wavefrontsize64\"",
    ] {
        assert!(llvm.contains(expected), "missing {expected:?} in:\n{llvm}");
    }
}

#[test]
fn lowers_each_workgroup_barrier_order_and_memory_scope_exactly() {
    let acquire = lower_kernel_to_llvm_ir(
        &barrier_only_module(SynchronizationScope::Workgroup, MemoryOrdering::Acquire),
        &KernelId::new("fill"),
    )
    .unwrap();
    assert!(
        acquire.contains(
            "call void @llvm.amdgcn.s.barrier()\n  fence syncscope(\"workgroup\") acquire"
        )
    );
    assert!(!acquire.contains("fence syncscope(\"workgroup\") release"));

    let release = lower_kernel_to_llvm_ir(
        &barrier_only_module(SynchronizationScope::Device, MemoryOrdering::Release),
        &KernelId::new("fill"),
    )
    .unwrap();
    assert!(
        release
            .contains("fence syncscope(\"agent\") release\n  call void @llvm.amdgcn.s.barrier()")
    );
    assert!(!release.contains("fence syncscope(\"agent\") acquire"));

    let acquire_release = lower_kernel_to_llvm_ir(
        &barrier_only_module(SynchronizationScope::Device, MemoryOrdering::AcquireRelease),
        &KernelId::new("fill"),
    )
    .unwrap();
    assert!(acquire_release.contains("fence syncscope(\"agent\") release"));
    assert!(acquire_release.contains("fence syncscope(\"agent\") acquire"));

    let sequential = lower_kernel_to_llvm_ir(
        &barrier_only_module(
            SynchronizationScope::System,
            MemoryOrdering::SequentiallyConsistent,
        ),
        &KernelId::new("fill"),
    )
    .unwrap();
    assert_occurrences(&sequential, "  fence seq_cst", 2);
    assert!(!sequential.contains("fence syncscope"));
}

#[test]
fn gfx942_barriers_use_one_workload_neutral_physical_policy() {
    let module = exact_gfx942_xnack_minus(barrier_only_module(
        SynchronizationScope::Workgroup,
        MemoryOrdering::AcquireRelease,
    ));
    let llvm = lower_kernel_to_gfx942_xnack_minus_llvm_ir(&module, &KernelId::new("fill")).unwrap();
    assert_eq!(
        llvm.matches("call void asm sideeffect \"s_barrier\", \"\"()")
            .count(),
        1
    );
    assert!(!llvm.contains("call void @llvm.amdgcn.s.barrier()"));
}

#[test]
fn gfx950_exact_lowering_binds_cpu_features_layout_and_physical_barrier() {
    let module = exact_gfx950_xnack_minus(barrier_only_module(
        SynchronizationScope::Workgroup,
        MemoryOrdering::AcquireRelease,
    ));
    let llvm = lower_kernel_to_gfx950_xnack_minus_llvm_ir(&module, &KernelId::new("fill")).unwrap();
    assert!(llvm.contains("target datalayout = \"e-m:e-p:64:64-p1:64:64"));
    assert!(llvm.contains("\"target-cpu\"=\"gfx950\""));
    assert!(llvm.contains("\"target-features\"=\"-wavefrontsize32,+wavefrontsize64,-xnack\""));
    assert_eq!(
        llvm.matches("call void asm sideeffect \"s_barrier\", \"\"()")
            .count(),
        1
    );

    let missing_entry = {
        let mut missing = module.clone();
        missing.functions[0]
            .required_capabilities
            .retain(|capability| {
                !matches!(
                    capability,
                    TargetCapability::Extension { namespace, name }
                        if namespace == AMDGPU_EXACT_TARGET_CAPABILITY_NAMESPACE
                            && name == "gfx950:xnack-"
                )
            });
        missing
    };
    assert!(
        lower_kernel_to_gfx950_xnack_minus_llvm_ir(&missing_entry, &KernelId::new("fill")).is_err()
    );
    assert!(lower_kernel_to_gfx942_xnack_minus_llvm_ir(&module, &KernelId::new("fill")).is_err());
}

#[test]
fn gfx950_sqrt_uses_the_native_llvm_intrinsic_accepted_by_rocm() {
    let gfx950 = exact_gfx950_xnack_minus(sqrt_module());
    let llvm =
        lower_kernel_to_gfx950_xnack_minus_llvm_ir(&gfx950, &KernelId::new("sqrt_kernel")).unwrap();
    assert_occurrences(&llvm, "declare float @llvm.sqrt.f32(float)", 1);
    assert_occurrences(&llvm, "call float @llvm.sqrt.f32(float %arg0)", 1);
    assert!(!llvm.contains("llvm.experimental.constrained.sqrt.f32"));
    assert!(llvm.contains("\"unsafe-fp-math\"=\"false\""));
    assert!(llvm.contains("\"approx-func-fp-math\"=\"false\""));
    assert!(llvm.contains("\"denormal-fp-math-f32\"=\"ieee,ieee\""));
    assert!(!llvm.contains("call fast float @llvm.sqrt.f32"));

    let gfx942 = exact_gfx942_xnack_minus(sqrt_module());
    let llvm =
        lower_kernel_to_gfx942_xnack_minus_llvm_ir(&gfx942, &KernelId::new("sqrt_kernel")).unwrap();
    assert_occurrences(
        &llvm,
        "declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)",
        1,
    );
    assert_occurrences(
        &llvm,
        "call float @llvm.experimental.constrained.sqrt.f32(float %arg0, metadata !\"round.tonearest\", metadata !\"fpexcept.ignore\")",
        1,
    );
}

#[test]
fn gfx950_full_module_lowers_exact_bf16_mfma_profile() {
    let module = gfx950_bf16_mfma_module();
    verify_module(&module).unwrap();
    let llvm = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap();

    let intrinsic = "llvm.amdgcn.mfma.f32.16x16x16bf16.1k";
    assert_eq!(llvm.matches(intrinsic).count(), 2, "{llvm}");
    assert!(llvm.contains(
        "<4 x i16> %matrix.0.0.lhs.3, <4 x i16> %matrix.0.0.rhs.3, <4 x float> %matrix.0.0.acc.3, i32 0, i32 0, i32 0"
    ));
    assert!(llvm.contains("\"target-cpu\"=\"gfx950\""));
    assert!(llvm.contains("\"target-features\"=\"-wavefrontsize32,+wavefrontsize64,-xnack\""));
    assert!(!llvm.contains("llvm.amdgcn.mfma.scale"));
}

#[test]
fn gfx950_bf16_admission_does_not_open_other_matrix_capabilities() {
    for capability_name in [
        format!("{BF16_F32_M16N16K16_CAPABILITY}.lookalike"),
        LDS_TILE_16X16_XOR4_CAPABILITY.to_owned(),
    ] {
        let mut module = gfx950_bf16_mfma_module();
        let capability = TargetCapability::Extension {
            namespace: MATRIX_CAPABILITY_NAMESPACE.to_owned(),
            name: capability_name.clone(),
        };
        module.required_capabilities.insert(capability);
        let error = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module)
            .expect_err("gfx950 must reject every non-allowlisted matrix capability");
        assert!(error.contains(LoweringDiagnosticCode::UnsupportedCapability));
        assert!(error.to_string().contains(&capability_name), "{error}");
    }
}

#[test]
fn gfx950_bf16_mfma_rejects_non_wave64_activity() {
    let mut module = gfx950_bf16_mfma_module();
    let OperationKind::Matrix(matrix) =
        &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[0].kind
    else {
        panic!("expected matrix operation")
    };
    matrix.active_lanes = 32;
    assert!(lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).is_err());
}

#[test]
fn gfx950_full_module_lowers_scaled_fp8_mfma_with_exact_intrinsic_abi() {
    let module = gfx950_scaled_mfma_module();
    let llvm = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap();

    let intrinsic = "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32";
    assert_eq!(llvm.matches(intrinsic).count(), 2, "{llvm}");
    assert!(llvm.contains(
        "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32)"
    ));
    assert!(llvm.contains(
        "<8 x i32> %matrix.0.0.lhs.7, <8 x i32> %matrix.0.0.rhs.7, <4 x float> %matrix.0.0.acc.3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0"
    ));
    assert_eq!(
        llvm.matches("extractelement <4 x float> %matrix.0.0.mfma")
            .count(),
        4
    );
    assert!(llvm.contains("\"target-cpu\"=\"gfx950\""));
    assert!(llvm.contains("\"target-features\"=\"-wavefrontsize32,+wavefrontsize64,-xnack\""));
    assert!(
        lower_kernel_to_gfx942_xnack_minus_llvm_ir(&module, &KernelId::new("gfx950_scaled_mfma"))
            .is_err()
    );
}

#[test]
fn gfx950_full_module_lowers_scaled_fp4_mfma_with_format_selectors_4_4() {
    let module = gfx950_scaled_fp4_mfma_module();
    let llvm = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap();

    assert!(llvm.contains(
        "<8 x i32> %matrix.0.0.lhs.7, <8 x i32> %matrix.0.0.rhs.7, <4 x float> %matrix.0.0.acc.3, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0"
    ));
    assert!(
        !llvm.contains("<4 x float> %matrix.0.0.acc.3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0")
    );
    assert!(
        lower_kernel_to_gfx942_xnack_minus_llvm_ir(&module, &KernelId::new("gfx950_scaled_mfma"))
            .is_err()
    );
}

#[test]
fn gfx950_full_module_lowers_mixed_fp4_by_fp8_mfma_with_format_selectors_4_0() {
    let module = gfx950_scaled_mixed_fp4_fp8_mfma_module();
    let llvm = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&module).unwrap();

    assert!(llvm.contains(
        "<8 x i32> %matrix.0.0.lhs.7, <8 x i32> %matrix.0.0.rhs.7, <4 x float> %matrix.0.0.acc.3, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0"
    ));
    assert!(!llvm.contains("<4 x float> %matrix.0.0.acc.3, i32 0, i32 4"));
    assert!(
        lower_kernel_to_gfx942_xnack_minus_llvm_ir(&module, &KernelId::new("gfx950_scaled_mfma"))
            .is_err()
    );
}

#[test]
fn gfx950_exact_diagnostics_admit_only_terminating_traps() {
    let trap = gfx950_diagnostic_module(AmdGpuDiagnosticOperation::Trap);
    let llvm = lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&trap).unwrap();
    assert_eq!(llvm.matches("declare void @llvm.trap()").count(), 1);
    assert_eq!(llvm.matches("call void @llvm.trap()").count(), 1);
    assert!(!llvm.contains(AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAME));

    let debug_trap = gfx950_diagnostic_module(AmdGpuDiagnosticOperation::DebugTrap);
    assert!(
        lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&debug_trap)
            .unwrap_err()
            .contains(LoweringDiagnosticCode::UnsupportedDiagnosticOperation)
    );
}

#[test]
fn accepts_only_structurally_convergent_workgroup_barrier_placement() {
    let mut unconditional = barrier_only_module(
        SynchronizationScope::Workgroup,
        MemoryOrdering::AcquireRelease,
    );
    let barrier = unconditional.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .remove(0);
    unconditional.functions[0].body.as_mut().unwrap().blocks[0].terminator =
        Some(Terminator::Branch {
            target: BlockId(1),
            arguments: vec![],
        });
    unconditional.functions[0].body.as_mut().unwrap().blocks[1]
        .operations
        .insert(0, barrier);
    lower_kernel_to_llvm_ir(&unconditional, &KernelId::new("fill"))
        .expect("an acyclic unconditional entry chain is convergent by construction");

    let mut divergent = barrier_only_module(
        SynchronizationScope::Workgroup,
        MemoryOrdering::AcquireRelease,
    );
    let barrier = divergent.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .remove(0);
    divergent.functions[0].body.as_mut().unwrap().blocks[1]
        .operations
        .insert(0, barrier);
    let error = lower_kernel_to_llvm_ir(&divergent, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnprovenBarrierConvergence
    );
    assert_eq!(error.diagnostics()[0].location.block, Some(BlockId(1)));
    assert!(
        error.diagnostics()[0]
            .message
            .contains("convergent operation requires Workgroup uniform control")
    );

    let mut cyclic = phi_loop_module();
    cyclic
        .required_capabilities
        .insert(TargetCapability::WorkgroupBarrier);
    cyclic.functions[0].body.as_mut().unwrap().blocks[1]
        .operations
        .insert(
            0,
            workgroup_barrier(
                SynchronizationScope::Workgroup,
                MemoryOrdering::AcquireRelease,
                [AddressSpace::Global],
            ),
        );
    let llvm = lower_kernel_to_llvm_ir(&cyclic, &KernelId::new("phi_loop"))
        .expect("a grid-uniform exiting loop has the same barrier count in every work-item");
    assert!(llvm.contains("call void @llvm.amdgcn.s.barrier()"));

    let mut duplicate_successor = barrier_only_module(
        SynchronizationScope::Workgroup,
        MemoryOrdering::AcquireRelease,
    );
    let barrier = duplicate_successor.functions[0]
        .body
        .as_mut()
        .unwrap()
        .blocks[0]
        .operations
        .remove(0);
    duplicate_successor.functions[0]
        .body
        .as_mut()
        .unwrap()
        .blocks[1]
        .operations
        .insert(0, barrier);
    duplicate_successor.functions[0]
        .body
        .as_mut()
        .unwrap()
        .blocks[0]
        .terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(1),
        else_arguments: vec![],
    });
    let result = std::panic::catch_unwind(|| {
        lower_kernel_to_llvm_ir(&duplicate_successor, &KernelId::new("fill"))
    });
    assert!(result.is_ok(), "duplicate CFG successors must not panic");
    assert!(
        result.unwrap().is_ok(),
        "identical successors reconverge independently of the branch value"
    );
}

#[test]
fn lds_lowering_requires_declared_capabilities_alignment_and_addressability() {
    let mut missing_dynamic = g4_synchronization_module();
    missing_dynamic
        .required_capabilities
        .remove(&TargetCapability::DynamicWorkgroupMemory);
    let error = lower_kernel_to_llvm_ir(&missing_dynamic, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedCapability
    );
    assert!(
        error.diagnostics()[0]
            .message
            .contains("DynamicWorkgroupMemory")
    );

    for (extent, expected_fragment) in [
        (WorkgroupMemoryExtent::Static(1), "requires alignment 8"),
        (
            WorkgroupMemoryExtent::Static(u32::MAX),
            "exceeding the AMDGPU 32-bit LDS address space",
        ),
    ] {
        let mut module = fill_module();
        module
            .required_capabilities
            .insert(TargetCapability::WorkgroupMemory);
        module.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .insert(
                0,
                op(
                    30,
                    Type::pointer(
                        Type::Scalar(ScalarType::U64),
                        AddressSpace::Workgroup,
                        AccessMode::ReadWrite,
                    ),
                    OperationKind::WorkgroupMemory(WorkgroupMemory {
                        element: Type::Scalar(ScalarType::U64),
                        extent,
                        alignment: if matches!(extent, WorkgroupMemoryExtent::Static(1)) {
                            4
                        } else {
                            8
                        },
                    }),
                ),
            );
        let error = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap_err();
        assert_eq!(
            error.diagnostics()[0].code,
            LoweringDiagnosticCode::UnsupportedWorkgroupMemory
        );
        assert!(
            error.diagnostics()[0].message.contains(expected_fragment),
            "unexpected diagnostic: {}",
            error.diagnostics()[0]
        );
    }
}

#[test]
fn scoped_integer_atomics_match_the_exact_golden() {
    let llvm = lower_kernel_to_llvm_ir(&scoped_atomics_module(), &KernelId::new("scoped_atomics"))
        .unwrap();

    assert_eq!(llvm, include_str!("fixtures/scoped_atomics.ll"));
    assert!(llvm.contains("syncscope(\"workgroup\") monotonic"));
    assert!(llvm.contains("syncscope(\"agent\") release"));
    assert!(llvm.contains("atomicrmw xchg ptr addrspace(1) %v3, i32 %arg1 seq_cst"));
    assert!(llvm.contains("%v7.cmpxchg = cmpxchg"));
    assert!(llvm.contains("%v8 = extractvalue { i32, i1 } %v7.cmpxchg, 1"));
    assert!(llvm.contains("atomicrmw add ptr addrspace(3) %v4"));
    assert!(llvm.contains("atomicrmw umin"));
    assert!(llvm.contains("atomicrmw umax"));
}

#[test]
fn signed_integer_min_and_max_select_signed_llvm_operations() {
    for (kind, opcode) in [(AtomicKind::Min, "min"), (AtomicKind::Max, "max")] {
        let module = single_global_atomic_module(
            ScalarType::I32,
            kind,
            SynchronizationScope::Device,
            MemoryOrdering::Relaxed,
            None,
        );
        let llvm = lower_kernel_to_llvm_ir(&module, &KernelId::new("single_atomic")).unwrap();
        assert!(
            llvm.contains(&format!(
                "atomicrmw {opcode} ptr addrspace(1) %v3, i32 %arg1 syncscope(\"agent\") monotonic"
            )),
            "wrong signed atomic operation in:\n{llvm}"
        );
    }
}

#[test]
fn global_pointer_kernel_parameter_reaches_atomic_llvm_exactly() {
    let module = direct_global_pointer_atomic_module();
    let llvm = lower_kernel_to_llvm_ir(&module, &KernelId::new("single_atomic")).unwrap();
    assert!(
        llvm.contains("ptr addrspace(1) %arg0, i32 %arg1, i32 %arg2"),
        "global pointer parameter missing from:\n{llvm}"
    );
    assert!(
        llvm.contains("%v4 = atomicrmw add ptr addrspace(1) %arg0, i32 %arg1 monotonic, align 4")
    );
}

#[test]
fn non_global_pointer_kernel_parameter_fails_closed() {
    let mut module = direct_global_pointer_atomic_module();
    let Type::Pointer(pointer) = &mut module.functions[0].signature.parameters[0] else {
        panic!("pointer expected")
    };
    pointer.address_space = AddressSpace::Workgroup;
    let OperationKind::Atomic(atomic) =
        &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[0].kind
    else {
        panic!("atomic expected")
    };
    atomic.access.address_space = AddressSpace::Workgroup;
    atomic.scope = SynchronizationScope::Workgroup;
    assert_eq!(
        first_code(&module, "single_atomic"),
        LoweringDiagnosticCode::UnsupportedAddressSpace
    );
}

#[test]
fn lowers_aligned_64_bit_integer_atomics() {
    let mut module = single_global_atomic_module(
        ScalarType::U64,
        AtomicKind::Add,
        SynchronizationScope::System,
        MemoryOrdering::SequentiallyConsistent,
        None,
    );
    let OperationKind::Atomic(atomic) =
        &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[1].kind
    else {
        panic!("atomic expected")
    };
    atomic.access.alignment = 8;

    let llvm = lower_kernel_to_llvm_ir(&module, &KernelId::new("single_atomic")).unwrap();
    assert!(llvm.contains("%v4 = atomicrmw add ptr addrspace(1) %v3, i64 %arg1 seq_cst, align 8"));
}

#[test]
fn rust_atomic_ordering_legality_is_rejected_before_lowering() {
    let cases = [
        single_global_atomic_module(
            ScalarType::U32,
            AtomicKind::Load,
            SynchronizationScope::Device,
            MemoryOrdering::Release,
            None,
        ),
        single_global_atomic_module(
            ScalarType::U32,
            AtomicKind::Store,
            SynchronizationScope::Device,
            MemoryOrdering::Acquire,
            None,
        ),
        single_global_atomic_module(
            ScalarType::U32,
            AtomicKind::CompareExchange,
            SynchronizationScope::Device,
            MemoryOrdering::AcquireRelease,
            Some(MemoryOrdering::Release),
        ),
    ];

    for module in cases {
        let errors = lower_kernel_to_llvm_ir(&module, &KernelId::new("single_atomic")).unwrap_err();
        assert_eq!(
            errors.diagnostics()[0].code,
            LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidAtomic)
        );
        assert!(errors.diagnostics()[0].message.contains("malformed"));
    }
}

#[test]
fn unsupported_atomic_type_scope_and_volatile_modes_fail_closed() {
    let float = single_global_atomic_module(
        ScalarType::F32,
        AtomicKind::Add,
        SynchronizationScope::Device,
        MemoryOrdering::Relaxed,
        None,
    );
    let float_error = lower_kernel_to_llvm_ir(&float, &KernelId::new("single_atomic")).unwrap_err();
    assert_eq!(
        float_error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedAtomic
    );
    assert_eq!(
        float_error.diagnostics()[0].message,
        "AMDGPU atomic lowering supports only 32-bit and 64-bit integers, found F32"
    );

    let subword = single_global_atomic_module(
        ScalarType::U16,
        AtomicKind::BitXor,
        SynchronizationScope::Device,
        MemoryOrdering::Relaxed,
        None,
    );
    let subword_error =
        lower_kernel_to_llvm_ir(&subword, &KernelId::new("single_atomic")).unwrap_err();
    assert_eq!(
        subword_error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedAtomic
    );
    assert_eq!(
        subword_error.diagnostics()[0].message,
        "AMDGPU atomic lowering supports only 32-bit and 64-bit integers, found U16"
    );

    let subgroup = single_global_atomic_module(
        ScalarType::U32,
        AtomicKind::Add,
        SynchronizationScope::Subgroup,
        MemoryOrdering::Relaxed,
        None,
    );
    let subgroup_error =
        lower_kernel_to_llvm_ir(&subgroup, &KernelId::new("single_atomic")).unwrap_err();
    assert_eq!(
        subgroup_error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedAtomic
    );
    assert_eq!(
        subgroup_error.diagnostics()[0].message,
        "AMDGPU atomic lowering does not support Global memory at Subgroup scope"
    );

    let mut volatile = single_global_atomic_module(
        ScalarType::U32,
        AtomicKind::Add,
        SynchronizationScope::Device,
        MemoryOrdering::Relaxed,
        None,
    );
    let OperationKind::Atomic(atomic) =
        &mut volatile.functions[0].body.as_mut().unwrap().blocks[0].operations[1].kind
    else {
        panic!("atomic expected")
    };
    atomic.access.volatile = true;
    let volatile_error =
        lower_kernel_to_llvm_ir(&volatile, &KernelId::new("single_atomic")).unwrap_err();
    assert_eq!(
        volatile_error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedAtomic
    );
    assert_eq!(
        volatile_error.diagnostics()[0].message,
        "volatile scoped atomics are outside the supported AMDGPU subset"
    );
}

#[test]
fn unsupported_atomic_capabilities_and_address_spaces_are_rejected() {
    let mut unsupported_capability = single_global_atomic_module(
        ScalarType::U32,
        AtomicKind::Add,
        SynchronizationScope::Device,
        MemoryOrdering::Relaxed,
        None,
    );
    unsupported_capability
        .required_capabilities
        .insert(TargetCapability::Atomic {
            width_bits: 16,
            address_space: AddressSpace::Global,
            max_scope: SynchronizationScope::Device,
        });
    assert_eq!(
        first_code(&unsupported_capability, "single_atomic"),
        LoweringDiagnosticCode::UnsupportedCapability
    );

    let mut generic = single_global_atomic_module(
        ScalarType::U32,
        AtomicKind::Add,
        SynchronizationScope::System,
        MemoryOrdering::Relaxed,
        None,
    );
    let Type::Slice(slice) = &mut generic.functions[0].signature.parameters[0] else {
        panic!("slice expected")
    };
    slice.address_space = AddressSpace::Generic;
    let Type::Pointer(pointer) =
        &mut generic.functions[0].body.as_mut().unwrap().blocks[0].operations[0].results[0].ty
    else {
        panic!("pointer expected")
    };
    pointer.address_space = AddressSpace::Generic;
    let OperationKind::Atomic(atomic) =
        &mut generic.functions[0].body.as_mut().unwrap().blocks[0].operations[1].kind
    else {
        panic!("atomic expected")
    };
    atomic.access.address_space = AddressSpace::Generic;
    assert_eq!(
        first_code(&generic, "single_atomic"),
        LoweringDiagnosticCode::UnsupportedAddressSpace
    );

    let mut invalid_system_lds = scoped_atomics_module();
    let OperationKind::Atomic(atomic) = &mut invalid_system_lds.functions[0]
        .body
        .as_mut()
        .unwrap()
        .blocks[0]
        .operations[6]
        .kind
    else {
        panic!("atomic expected")
    };
    atomic.scope = SynchronizationScope::System;
    assert_eq!(
        first_code(&invalid_system_lds, "scoped_atomics"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidAtomic)
    );
}

#[test]
fn atomic_lowering_diagnostics_are_deterministic() {
    let module = single_global_atomic_module(
        ScalarType::F32,
        AtomicKind::Exchange,
        SynchronizationScope::System,
        MemoryOrdering::SequentiallyConsistent,
        None,
    );
    let diagnostics = (0..32)
        .map(|_| {
            lower_kernel_to_llvm_ir(&module, &KernelId::new("single_atomic"))
                .unwrap_err()
                .into_diagnostics()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(diagnostics.len(), 1);
}

#[test]
fn synchronization_and_lds_lowering_fail_closed_with_specific_diagnostics() {
    let mut atomic = fill_module();
    atomic.functions[0].body.as_mut().unwrap().blocks[1]
        .operations
        .push(op(
            30,
            Type::F32,
            OperationKind::Atomic(Atomic {
                kind: AtomicKind::Add,
                pointer: ValueId(5),
                value: Some(ValueId(1)),
                compare: None,
                access: MemoryAccess::new(AddressSpace::Global, 4),
                scope: SynchronizationScope::Device,
                ordering: MemoryOrdering::Relaxed,
                failure_ordering: None,
            }),
        ));
    assert_eq!(
        first_code(&atomic, "fill"),
        LoweringDiagnosticCode::UnsupportedAtomic
    );

    let mut legacy_barrier = fill_module();
    legacy_barrier.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            Operation::new(
                vec![],
                OperationKind::Barrier(Barrier {
                    execution_scope: SynchronizationScope::Workgroup,
                    memory_scope: SynchronizationScope::Workgroup,
                    semantics: BarrierSemantics::new(
                        MemoryOrdering::AcquireRelease,
                        [AddressSpace::Workgroup],
                    ),
                }),
            ),
        );
    assert_eq!(
        first_code(&legacy_barrier, "fill"),
        LoweringDiagnosticCode::UnsupportedBarrier
    );

    let mut ambiguous_lds = fill_module();
    ambiguous_lds.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            op(
                30,
                Type::pointer(Type::F32, AddressSpace::Workgroup, AccessMode::ReadWrite),
                OperationKind::Alloca {
                    element: Type::F32,
                    count: None,
                    address_space: AddressSpace::Workgroup,
                    alignment: 4,
                },
            ),
        );
    assert_eq!(
        first_code(&ambiguous_lds, "fill"),
        LoweringDiagnosticCode::UnsupportedWorkgroupMemory
    );

    let mut invalid_fence = fill_module();
    invalid_fence.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            Operation::new(
                vec![],
                OperationKind::Fence(Fence {
                    memory_scope: SynchronizationScope::Device,
                    semantics: BarrierSemantics::new(
                        MemoryOrdering::Relaxed,
                        [AddressSpace::Global],
                    ),
                }),
            ),
        );
    assert_eq!(
        first_code(&invalid_fence, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidFence)
    );
}

#[test]
fn dynamic_1d_fill_matches_the_exact_golden() {
    let output = lower_kernel_to_llvm_ir(&fill_module(), &KernelId::new("fill")).unwrap();
    assert_eq!(output, include_str!("fixtures/fill_g1.ll"));
    assert!(output.contains("mul i64 %v2.group, 64"));
    assert!(!output.contains("256"));
    assert!(!output.contains("getelementptr inbounds"));
}

#[test]
fn gfx942_private_pointer_slots_lower_with_exact_address_spaces() {
    let output = lower_kernel_to_gfx942_xnack_minus_llvm_ir(
        &private_pointer_slot_module(),
        &KernelId::new("private_pointer_slot"),
    )
    .unwrap();

    assert!(output.contains("%v2 = alloca ptr addrspace(1), align 8, addrspace(5)"));
    assert!(output.contains("store ptr addrspace(1) %v1, ptr addrspace(5) %v2, align 8"));
    assert!(output.contains("%v3 = load ptr addrspace(1), ptr addrspace(5) %v2, align 8"));
    assert!(output.contains("%v5 = getelementptr float, ptr addrspace(1) %v3, i64 0"));
}

#[test]
fn vecadd_three_slice_abi_and_cfg_match_the_exact_golden() {
    let output = lower_kernel_to_llvm_ir(&vecadd_module(), &KernelId::new("vecadd")).unwrap();
    assert_eq!(output, include_str!("fixtures/vecadd_g1.ll"));
    assert!(output.contains(
        "@vecadd(ptr addrspace(1) %arg0.data, i64 %arg0.len, ptr addrspace(1) %arg1.data, i64 %arg1.len, ptr addrspace(1) %arg2.data, i64 %arg2.len)"
    ));
    assert_occurrences(&output, "load float", 2);
    assert_occurrences(&output, "store float", 1);
    assert!(output.contains("%v20 = fadd float %v14, %v19"));
}

#[test]
fn loop_block_arguments_materialize_as_exact_phi_golden() {
    let output = lower_kernel_to_llvm_ir(&phi_loop_module(), &KernelId::new("phi_loop")).unwrap();
    assert_eq!(output, include_str!("fixtures/phi_loop_g1.ll"));
    assert!(output.contains("%v10 = phi i64 [ %arg1, %bb0 ], [ %v15, %edge_bb1_0_bb1 ]"));
    assert!(output.contains(
        "%v11.data = phi ptr addrspace(1) [ %arg0.data, %bb0 ], [ %v11.data, %edge_bb1_0_bb1 ]"
    ));
    assert!(
        output.contains("%v12 = phi ptr addrspace(1) [ %v2, %bb0 ], [ %v12, %edge_bb1_0_bb1 ]")
    );
}

#[test]
fn static_and_dynamic_1d_extents_lower_identically() {
    let dynamic = fill_module();
    let mut static_extent = dynamic.clone();
    static_extent.kernels[0].domain = LaunchDomain::D1 {
        x: LaunchExtent::Static(4096),
    };

    assert_eq!(
        lower_kernel_to_llvm_ir(&dynamic, &KernelId::new("fill")).unwrap(),
        lower_kernel_to_llvm_ir(&static_extent, &KernelId::new("fill")).unwrap()
    );
}

#[test]
fn lowering_is_deterministic() {
    let module = fill_module();
    let outputs = (0..32)
        .map(|_| lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(outputs.len(), 1);
}

#[test]
fn extended_subset_lowers_constants_casts_arithmetic_loads_and_volatile_access() {
    let mut module = fill_module();
    let function = &mut module.functions[0];
    let blocks = &mut function.body.as_mut().unwrap().blocks;
    blocks[0].operations.insert(
        1,
        op(
            7,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(0)),
        ),
    );
    blocks[0].operations.insert(
        2,
        op(
            8,
            Type::INDEX,
            OperationKind::Cast {
                kind: CastKind::ZeroExtend,
                value: ValueId(7),
                to: Type::INDEX,
            },
        ),
    );
    blocks[0].operations.insert(
        3,
        op(
            9,
            Type::INDEX,
            OperationKind::Binary {
                op: BinaryOp::Add,
                lhs: ValueId(2),
                rhs: ValueId(8),
            },
        ),
    );
    if let OperationKind::Compare { lhs, .. } = &mut blocks[0].operations[5].kind {
        *lhs = ValueId(9);
    }
    if let OperationKind::GetElementPointer { offset, .. } = &mut blocks[1].operations[1].kind {
        *offset = ValueId(9);
    }
    let mut volatile_load = MemoryAccess::new(AddressSpace::Global, 8);
    volatile_load.volatile = true;
    blocks[1].operations.insert(
        2,
        op(
            10,
            Type::F32,
            OperationKind::Load {
                pointer: ValueId(6),
                access: volatile_load,
            },
        ),
    );
    blocks[1].operations.insert(
        3,
        op(
            11,
            Type::F32,
            OperationKind::Binary {
                op: BinaryOp::Add,
                lhs: ValueId(10),
                rhs: ValueId(1),
            },
        ),
    );
    let OperationKind::Store { value, access, .. } = &mut blocks[1].operations[4].kind else {
        panic!("store expected")
    };
    *value = ValueId(11);
    let mut volatile_store = MemoryAccess::new(AddressSpace::Global, 16);
    volatile_store.volatile = true;
    *access = volatile_store;
    let mut dead = BasicBlock::new(BlockId(3));
    dead.terminator = Some(Terminator::Unreachable);
    blocks.push(dead);

    let output = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap();
    assert!(output.contains("%v8 = zext i32 0 to i64"));
    assert!(output.contains("%v9 = add i64 %v2, %v8"));
    assert!(output.contains("load volatile float, ptr addrspace(1) %v6, align 8"));
    assert!(output.contains("store volatile float %v11, ptr addrspace(1) %v6, align 16"));
    assert!(output.contains("bb3:\n  unreachable"));
}

#[test]
fn verifier_runs_before_lowering_and_ambiguous_ids_fail_closed() {
    let mut malformed = fill_module();
    let OperationKind::Compare { lhs, .. } =
        &mut malformed.functions[0].body.as_mut().unwrap().blocks[0].operations[2].kind
    else {
        panic!("compare expected")
    };
    *lhs = ValueId(999);
    assert_eq!(
        first_code(&malformed, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::UndefinedValue)
    );

    let mut ambiguous = fill_module();
    ambiguous.kernels.push(ambiguous.kernels[0].clone());
    assert_eq!(
        first_code(&ambiguous, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::DuplicateKernel)
    );
}

#[test]
fn kernel_selection_and_symbol_names_are_fail_closed() {
    let module = fill_module();
    assert_eq!(
        first_code(&module, "missing"),
        LoweringDiagnosticCode::MissingKernel
    );

    let mut unsafe_name = fill_module();
    unsafe_name.kernels[0].id = KernelId::new("fill\nret_void");
    assert_eq!(
        first_code(&unsafe_name, "fill\nret_void"),
        LoweringDiagnosticCode::UnsafeSymbolName
    );
}

#[test]
fn kernel_selection_requires_an_exact_identity() {
    let mut module = fill_module();
    module.kernels[0].id = KernelId::new("fill_extra");

    for alias in ["fill", "fill_impl", "extra"] {
        assert_eq!(
            first_code(&module, alias),
            LoweringDiagnosticCode::MissingKernel
        );
    }
    let output = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill_extra")).unwrap();
    assert!(output.contains("define amdgpu_kernel void @fill_extra("));
    assert!(!output.contains("define amdgpu_kernel void @fill("));
}

#[test]
fn dynamic_higher_rank_domains_preserve_xyz_workgroups_and_workgroup_size_is_mandatory() {
    let mut missing_size = fill_module();
    missing_size.kernels[0].workgroup_size = None;
    assert_eq!(
        first_code(&missing_size, "fill"),
        LoweringDiagnosticCode::MissingWorkgroupSize
    );

    for (domain, workgroup, flat) in [
        (
            LaunchDomain::D2 {
                x: LaunchExtent::Dynamic,
                y: LaunchExtent::Dynamic,
            },
            WorkgroupSize::new(64, 1, 1),
            64,
        ),
        (
            LaunchDomain::D2 {
                x: LaunchExtent::Dynamic,
                y: LaunchExtent::Dynamic,
            },
            WorkgroupSize::new(16, 16, 1),
            256,
        ),
        (
            LaunchDomain::D3 {
                x: LaunchExtent::Dynamic,
                y: LaunchExtent::Dynamic,
                z: LaunchExtent::Dynamic,
            },
            WorkgroupSize::new(4, 4, 4),
            64,
        ),
    ] {
        let mut module = fill_module();
        let rank = domain.rank();
        module.kernels[0].domain = domain;
        module.kernels[0].workgroup_size = Some(workgroup);
        let llvm = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap();
        assert!(llvm.contains(&format!(
            "\"amdgpu-flat-work-group-size\"=\"{flat},{flat}\""
        )));
        assert!(llvm.contains(&format!(
            "!0 = !{{i32 {}, i32 {}, i32 {}}}",
            workgroup.x, workgroup.y, workgroup.z
        )));
        assert!(llvm.contains("call i32 @llvm.amdgcn.workitem.id.y()"));
        assert!(llvm.contains("call i32 @llvm.amdgcn.workgroup.id.y()"));
        assert!(llvm.contains("call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()"));
        assert!(llvm.contains("getelementptr inbounds i8, ptr addrspace(4)"));
        assert!(llvm.contains("i64 12"));
        if rank == 2 {
            assert!(llvm.contains(".row = mul i64"));
            assert!(!llvm.contains("i64 16"));
        } else {
            assert!(llvm.contains("call i32 @llvm.amdgcn.workitem.id.z()"));
            assert!(llvm.contains("call i32 @llvm.amdgcn.workgroup.id.z()"));
            assert!(llvm.contains("i64 16"));
            assert!(llvm.contains(".plane_row_scaled = mul i64"));
        }
    }
}

#[test]
fn workgroup_count_and_size_use_dispatch_and_authenticated_descriptor_geometry() {
    let mut module = fill_module();
    let entry = &mut module.functions[0].body.as_mut().unwrap().blocks[0];
    entry.operations.push(op(
        20,
        Type::INDEX,
        OperationKind::Intrinsic(IntrinsicOperation::new(
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::WorkgroupCount,
                axis: Axis::X,
            },
            Type::INDEX,
        )),
    ));
    entry.operations.push(op(
        21,
        Type::INDEX,
        OperationKind::Intrinsic(IntrinsicOperation::new(
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::WorkgroupSize,
                axis: Axis::X,
            },
            Type::INDEX,
        )),
    ));

    let llvm = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap();
    assert!(llvm.contains("%v20.dispatch = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()"));
    assert!(llvm.contains(
        "%v20.grid.ptr = getelementptr inbounds i8, ptr addrspace(4) %v20.dispatch, i64 12"
    ));
    assert!(llvm.contains("%v20.grid.i32 = load i32, ptr addrspace(4) %v20.grid.ptr, align 4"));
    assert!(llvm.contains("%v20.grid = zext i32 %v20.grid.i32 to i64"));
    assert!(llvm.contains("%v20.rounded = add i64 %v20.grid, 63"));
    assert!(llvm.contains("%v20 = udiv i64 %v20.rounded, 64"));
    assert!(llvm.contains("%v21 = add i64 64, 0"));
    assert_occurrences(
        &llvm,
        "declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()",
        1,
    );

    let OperationKind::Intrinsic(intrinsic) =
        &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[3].kind
    else {
        panic!("workgroup-count intrinsic expected")
    };
    intrinsic.kind = IntrinsicKind::InvocationIndex {
        kind: IndexKind::WorkgroupCount,
        axis: Axis::Y,
    };
    assert_eq!(
        first_code(&module, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidLaunchDomain)
    );
}

#[test]
fn invalid_or_unbounded_workgroup_geometry_is_rejected() {
    let mut zero = fill_module();
    zero.kernels[0].workgroup_size = Some(WorkgroupSize::new(0, 1, 1));
    assert_eq!(
        first_code(&zero, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidWorkgroupSize)
    );

    let mut zero_extent = fill_module();
    zero_extent.kernels[0].domain = LaunchDomain::D1 {
        x: LaunchExtent::Static(0),
    };
    assert_eq!(
        first_code(&zero_extent, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidLaunchDomain)
    );

    let mut oversized = fill_module();
    oversized.kernels[0].workgroup_size = Some(WorkgroupSize::new(1025, 1, 1));
    assert_eq!(
        first_code(&oversized, "fill"),
        LoweringDiagnosticCode::UnsupportedWorkgroupSize
    );

    let mut overflow = fill_module();
    overflow.kernels[0].domain = LaunchDomain::D2 {
        x: LaunchExtent::Static(1),
        y: LaunchExtent::Static(1),
    };
    overflow.kernels[0].workgroup_size = Some(WorkgroupSize::new(u32::MAX, 2, 1));
    let error = lower_kernel_to_llvm_ir(&overflow, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedWorkgroupSize
    );
    assert_eq!(
        error.diagnostics()[0].message,
        "workgroup dimensions overflow the flat workgroup size"
    );

    for size in [WorkgroupSize::new(64, 2, 1), WorkgroupSize::new(64, 1, 2)] {
        let mut inactive_axis = fill_module();
        inactive_axis.kernels[0].workgroup_size = Some(size);
        assert_eq!(
            first_code(&inactive_axis, "fill"),
            LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidWorkgroupSize)
        );
    }

    let mut rank2_inactive_axis = fill_module();
    rank2_inactive_axis.kernels[0].domain = LaunchDomain::D2 {
        x: LaunchExtent::Dynamic,
        y: LaunchExtent::Dynamic,
    };
    rank2_inactive_axis.kernels[0].workgroup_size = Some(WorkgroupSize::new(64, 1, 2));
    assert_eq!(
        first_code(&rank2_inactive_axis, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidWorkgroupSize)
    );

    let mut rank2_zero_extent = fill_module();
    rank2_zero_extent.kernels[0].domain = LaunchDomain::D2 {
        x: LaunchExtent::Dynamic,
        y: LaunchExtent::Static(0),
    };
    assert_eq!(
        first_code(&rank2_zero_extent, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidLaunchDomain)
    );
}

#[test]
fn static_multidimensional_workgroups_preserve_exact_and_flat_geometry() {
    let mut module = fill_module();
    module.kernels[0].domain = LaunchDomain::D2 {
        x: LaunchExtent::Static(8),
        y: LaunchExtent::Static(4),
    };
    module.kernels[0].workgroup_size = Some(WorkgroupSize::new(32, 2, 1));

    let llvm = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap();
    assert!(llvm.contains("\"amdgpu-flat-work-group-size\"=\"64,64\""));
    assert!(llvm.contains("!0 = !{i32 32, i32 2, i32 1}"));
    assert!(llvm.contains(".x.base = mul i64 %v2.x.group, 32"));
    assert!(llvm.contains(".y.base = mul i64 %v2.y.group, 2"));
    assert!(llvm.contains("%v2.row = mul i64 %v2.y, %v2.grid.x"));

    let mut unsupported_axis = module;
    let OperationKind::Intrinsic(intrinsic) =
        &mut unsupported_axis.functions[0].body.as_mut().unwrap().blocks[0].operations[0].kind
    else {
        panic!("global invocation index expected")
    };
    intrinsic.kind = IntrinsicKind::InvocationIndex {
        kind: IndexKind::Global,
        axis: Axis::Y,
    };
    assert_eq!(
        first_code(&unsupported_axis, "fill"),
        LoweringDiagnosticCode::UnsupportedOperation
    );
}

#[test]
fn declarations_and_kernel_results_are_rejected_by_input_verification() {
    let mut declaration = fill_module();
    declaration.functions[0].body = None;
    assert_eq!(
        first_code(&declaration, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::KernelEntryDeclaration)
    );

    let mut result = fill_module();
    result.functions[0].signature.results.push(Type::F32);
    result.functions[0].body.as_mut().unwrap().blocks[2].terminator = Some(Terminator::Return {
        values: vec![ValueId(1)],
    });
    assert_eq!(
        first_code(&result, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::KernelReturnsValue)
    );
}

#[test]
fn every_capability_owner_is_rejected_at_its_location() {
    for owner in 0..3 {
        let mut module = fill_module();
        match owner {
            0 => {
                module
                    .required_capabilities
                    .insert(TargetCapability::Float64);
            }
            1 => {
                module.kernels[0]
                    .required_capabilities
                    .insert(TargetCapability::Float64);
            }
            _ => {
                module.functions[0]
                    .required_capabilities
                    .insert(TargetCapability::Float64);
            }
        }
        let errors = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap_err();
        assert_eq!(
            errors.diagnostics()[0].code,
            LoweringDiagnosticCode::UnsupportedCapability
        );
        assert_eq!(
            errors.diagnostics()[0].location.function.is_some(),
            owner == 2
        );
        assert_eq!(
            errors.diagnostics()[0].location.kernel.is_some(),
            owner != 0
        );
    }
}

#[test]
fn unsupported_parameter_types_and_address_spaces_are_rejected() {
    let cases = [
        (Type::F64, LoweringDiagnosticCode::UnsupportedType),
        (
            Type::slice(Type::F32, AddressSpace::Workgroup, AccessMode::ReadWrite),
            LoweringDiagnosticCode::UnsupportedAddressSpace,
        ),
        (
            Type::pointer(Type::Unit, AddressSpace::Global, AccessMode::ReadWrite),
            LoweringDiagnosticCode::UnsupportedType,
        ),
        (Type::Unit, LoweringDiagnosticCode::UnsupportedParameter),
    ];
    for (parameter, expected) in cases {
        let mut module = fill_module();
        module.functions[0].signature.parameters.push(parameter);
        module.functions[0]
            .body
            .as_mut()
            .unwrap()
            .parameters
            .push(ValueId(20));
        assert_eq!(first_code(&module, "fill"), expected);
    }
}

#[test]
fn read_only_and_cross_address_space_stores_fail_before_emission() {
    let mut read_only = fill_module();
    read_only.functions[0].signature.parameters[0] = global_slice(AccessMode::ReadOnly);
    let body = read_only.functions[0].body.as_mut().unwrap();
    body.blocks[1].operations[0].results[0].ty = global_pointer(AccessMode::ReadOnly);
    body.blocks[1].operations[1].results[0].ty = global_pointer(AccessMode::ReadOnly);
    assert_eq!(
        first_code(&read_only, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidMemoryAccess)
    );

    let mut mismatched_space = fill_module();
    let OperationKind::Store { access, .. } =
        &mut mismatched_space.functions[0].body.as_mut().unwrap().blocks[1].operations[2].kind
    else {
        panic!("store expected")
    };
    access.address_space = AddressSpace::Workgroup;
    assert_eq!(
        first_code(&mismatched_space, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidMemoryAccess)
    );
}

#[test]
fn lowers_wave32_and_wave64_to_exact_width_bound_llvm() {
    let wave32 = lower_kernel_to_llvm_ir(
        &wave_module(WaveWidth::Wave32),
        &KernelId::new("wave_kernel"),
    )
    .unwrap();
    let wave64 = lower_kernel_to_llvm_ir(
        &wave_module(WaveWidth::Wave64),
        &KernelId::new("wave_kernel"),
    )
    .unwrap();
    assert_eq!(wave32, include_str!("fixtures/wave32.ll"));
    assert_eq!(wave64, include_str!("fixtures/wave64.ll"));
    assert!(wave32.contains("\"target-features\"=\"+wavefrontsize32,-wavefrontsize64\""));
    assert!(wave64.contains("\"target-features\"=\"-wavefrontsize32,+wavefrontsize64\""));
    assert!(wave32.contains("@llvm.amdgcn.ballot.i32"));
    assert!(!wave32.contains("@llvm.amdgcn.mbcnt.hi"));
    assert!(wave64.contains("@llvm.amdgcn.ballot.i64"));
    assert!(wave64.contains("@llvm.amdgcn.mbcnt.hi"));
}

#[test]
fn rejects_missing_mismatched_and_partial_wave_execution() {
    let mut missing = wave_module(WaveWidth::Wave64);
    missing.functions[0].required_capabilities.clear();
    assert_eq!(
        first_code(&missing, "wave_kernel"),
        LoweringDiagnosticCode::UnsupportedWaveOperation
    );

    let mut mismatch = wave_module(WaveWidth::Wave32);
    mismatch.functions[0].required_capabilities = BTreeSet::from([
        TargetCapability::Subgroups,
        TargetCapability::SubgroupSize(64),
        TargetCapability::WaveWidth(WaveWidth::Wave64),
    ]);
    assert_eq!(
        first_code(&mismatch, "wave_kernel"),
        LoweringDiagnosticCode::UnsupportedWaveOperation
    );

    let mut multidimensional = wave_module(WaveWidth::Wave64);
    multidimensional.kernels[0].domain = LaunchDomain::D2 {
        x: LaunchExtent::Static(1),
        y: LaunchExtent::Static(1),
    };
    multidimensional.kernels[0].workgroup_size = Some(WorkgroupSize::new(32, 2, 1));
    let llvm = lower_kernel_to_llvm_ir(&multidimensional, &KernelId::new("wave_kernel")).unwrap();
    assert!(llvm.contains("\"amdgpu-flat-work-group-size\"=\"64,64\""));
    assert!(llvm.contains("!0 = !{i32 32, i32 2, i32 1}"));

    let mut partial = wave_module(WaveWidth::Wave64);
    partial.kernels[0].domain = LaunchDomain::D2 {
        x: LaunchExtent::Static(1),
        y: LaunchExtent::Static(1),
    };
    partial.kernels[0].workgroup_size = Some(WorkgroupSize::new(32, 3, 1));
    let error = lower_kernel_to_llvm_ir(&partial, &KernelId::new("wave_kernel")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedWaveOperation
    );
    assert_eq!(
        error.diagnostics()[0].message,
        "full-wave execution requires flat workgroup size 96 to be a multiple of 64"
    );
}

#[test]
fn invalid_wave_types_tiles_and_convergence_fail_before_emission() {
    let mut tile = wave_module(WaveWidth::Wave32);
    let OperationKind::Wave(WaveOperation {
        kind: WaveOperationKind::ShuffleIndex { tile_width, .. },
        ..
    }) = &mut tile.functions[0].body.as_mut().unwrap().blocks[0].operations[4].kind
    else {
        panic!("shuffle expected")
    };
    *tile_width = 3;
    assert_eq!(
        first_code(&tile, "wave_kernel"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidWaveOperation)
    );

    let mut convergence = wave_module(WaveWidth::Wave64);
    let OperationKind::Wave(wave) =
        &mut convergence.functions[0].body.as_mut().unwrap().blocks[0].operations[0].kind
    else {
        panic!("wave operation expected")
    };
    wave.convergence = Convergence::uniform(SynchronizationScope::Workgroup);
    assert_eq!(
        first_code(&convergence, "wave_kernel"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidConvergence)
    );
}

#[test]
fn rocm_test_targets_are_exact_canonical_amd_target_ids() {
    for target in ["gfx1151", "gfx942:sramecc+:xnack-"] {
        assert_eq!(
            canonical_test_target(target).unwrap(),
            target.split(':').next().unwrap()
        );
    }
    for target in [
        "",
        " gfx1151",
        "gfx1151 ",
        "gfx942:xnack-:sramecc+",
        "gfx942:sramecc+:sramecc-",
        "gfx942:xnack",
        "gfx9-generic",
        "--help",
        "gfx1151\n-mcpu=gfx942",
    ] {
        assert!(
            canonical_test_target(target).is_err(),
            "accepted adversarial target {target:?}"
        );
    }
}

#[test]
fn excluded_operations_constants_casts_and_comparisons_have_located_errors() {
    let mut divide = fill_module();
    divide.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            op(
                20,
                Type::F32,
                OperationKind::Binary {
                    op: BinaryOp::Divide,
                    lhs: ValueId(1),
                    rhs: ValueId(1),
                },
            ),
        );
    let error = lower_kernel_to_llvm_ir(&divide, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedOperation
    );
    assert_eq!(error.diagnostics()[0].location.block, Some(BlockId(0)));
    assert_eq!(error.diagnostics()[0].location.operation, Some(0));
    let exact_divide = exact_gfx942_xnack_minus(divide);
    let exact_divide =
        lower_kernel_to_gfx942_xnack_minus_llvm_ir(&exact_divide, &KernelId::new("fill")).unwrap();
    assert!(exact_divide.contains("%v20 = fdiv float %arg1, %arg1"));

    let mut nan = fill_module();
    nan.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            op(
                20,
                Type::F32,
                OperationKind::Constant(Constant::F32Bits(f32::NAN.to_bits())),
            ),
        );
    assert_eq!(
        first_code(&nan, "fill"),
        LoweringDiagnosticCode::UnsupportedConstant
    );

    let mut bad_cast = fill_module();
    let operations = &mut bad_cast.functions[0].body.as_mut().unwrap().blocks[0].operations;
    operations.insert(
        0,
        op(
            20,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(7)),
        ),
    );
    operations.insert(
        1,
        op(
            21,
            Type::Scalar(ScalarType::U64),
            OperationKind::Cast {
                kind: CastKind::SignExtend,
                value: ValueId(20),
                to: Type::Scalar(ScalarType::U64),
            },
        ),
    );
    assert_eq!(
        first_code(&bad_cast, "fill"),
        LoweringDiagnosticCode::InputVerification(DiagnosticCode::InvalidCast)
    );

    let mut float_compare = fill_module();
    let OperationKind::Compare { lhs, rhs, .. } =
        &mut float_compare.functions[0].body.as_mut().unwrap().blocks[0].operations[2].kind
    else {
        panic!("compare expected")
    };
    *lhs = ValueId(1);
    *rhs = ValueId(1);
    assert_eq!(
        first_code(&float_compare, "fill"),
        LoweringDiagnosticCode::UnsupportedOperation
    );
    let exact_float_compare = exact_gfx942_xnack_minus(float_compare);
    let exact_float_compare =
        lower_kernel_to_gfx942_xnack_minus_llvm_ir(&exact_float_compare, &KernelId::new("fill"))
            .unwrap();
    assert!(exact_float_compare.contains("%v4 = fcmp olt float %arg1, %arg1"));
}

#[test]
fn integer_switches_lower_deterministically() {
    let mut switch = fill_module();
    switch.functions[0].body.as_mut().unwrap().blocks[0].terminator = Some(Terminator::Switch {
        selector: ValueId(2),
        cases: vec![SwitchCase {
            value: 7,
            target: BlockId(1),
            arguments: vec![],
        }],
        default_target: BlockId(2),
        default_arguments: vec![],
    });
    let legacy = lower_kernel_to_llvm_ir(&switch, &KernelId::new("fill")).unwrap();
    assert!(legacy.contains("switch i64 %v2, label %bb2 [\n    i64 7, label %bb1\n  ]"));

    switch.functions[0].body.as_mut().unwrap().blocks[0].terminator =
        Some(Terminator::IntegerSwitch {
            selector: ValueId(2),
            cases: vec![IntegerSwitchCase {
                value: Constant::Index(42),
                target: BlockId(1),
                arguments: vec![],
            }],
            default_target: BlockId(2),
            default_arguments: vec![],
        });
    let typed = lower_kernel_to_llvm_ir(&switch, &KernelId::new("fill")).unwrap();
    assert!(typed.contains("switch i64 %v2, label %bb2 [\n    i64 42, label %bb1\n  ]"));
    assert_eq!(
        typed,
        lower_kernel_to_llvm_ir(&switch, &KernelId::new("fill")).unwrap()
    );
}

#[test]
fn irreducible_control_flow_is_rejected_deterministically() {
    let mut module = fill_module();
    let blocks = &mut module.functions[0].body.as_mut().unwrap().blocks;
    blocks[1].terminator = Some(Terminator::Branch {
        target: BlockId(2),
        arguments: vec![],
    });
    blocks[2].terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });

    let first = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap_err();
    let second = lower_kernel_to_llvm_ir(&module, &KernelId::new("fill")).unwrap_err();
    assert_eq!(first, second);
    assert!(first.contains(LoweringDiagnosticCode::IrreducibleControlFlow));
    assert_eq!(first.diagnostics()[0].location.block, Some(BlockId(1)));
    assert!(first.to_string().contains("bb1, bb2"));
}

#[test]
fn unrepresentable_phi_cfgs_fail_closed_with_located_errors() {
    let mut entry_parameter = fill_module();
    entry_parameter.functions[0].body.as_mut().unwrap().blocks[0]
        .parameters
        .push(ValueDef::new(ValueId(20), Type::F32));
    let error = lower_kernel_to_llvm_ir(&entry_parameter, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedBlockArguments
    );
    assert_eq!(error.diagnostics()[0].location.block, Some(BlockId(0)));
    assert!(error.diagnostics()[0].message.contains("entry-block"));

    let mut predecessorless = fill_module();
    let mut dead = BasicBlock::new(BlockId(3));
    dead.parameters.push(ValueDef::new(ValueId(20), Type::F32));
    dead.terminator = Some(Terminator::Unreachable);
    predecessorless.functions[0]
        .body
        .as_mut()
        .unwrap()
        .blocks
        .push(dead);
    let error = lower_kernel_to_llvm_ir(&predecessorless, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedBlockArguments
    );
    assert_eq!(error.diagnostics()[0].location.block, Some(BlockId(3)));
    assert!(
        error.diagnostics()[0]
            .message
            .contains("without an incoming CFG edge")
    );

    let mut duplicate_edges = fill_module();
    let blocks = &mut duplicate_edges.functions[0].body.as_mut().unwrap().blocks;
    blocks[1]
        .parameters
        .push(ValueDef::new(ValueId(20), Type::F32));
    let Terminator::ConditionalBranch {
        then_arguments,
        else_target,
        else_arguments,
        ..
    } = blocks[0].terminator.as_mut().unwrap()
    else {
        panic!("conditional branch expected")
    };
    then_arguments.push(ValueId(1));
    *else_target = BlockId(1);
    else_arguments.push(ValueId(1));
    let llvm = lower_kernel_to_llvm_ir(&duplicate_edges, &KernelId::new("fill")).unwrap();
    assert!(llvm.contains("br i1 %v4, label %edge_bb0_0_bb1, label %edge_bb0_1_bb1"));
    assert!(
        llvm.contains("%v20 = phi float [ %arg1, %edge_bb0_0_bb1 ], [ %arg1, %edge_bb0_1_bb1 ]")
    );
    assert!(llvm.contains("edge_bb0_0_bb1:\n  br label %bb1"));
    assert!(llvm.contains("edge_bb0_1_bb1:\n  br label %bb1"));
}

#[test]
fn unsupported_phi_types_and_address_spaces_fail_before_emission() {
    let mut unsupported_type = fill_module();
    let blocks = &mut unsupported_type.functions[0].body.as_mut().unwrap().blocks;
    blocks[0].operations.insert(
        0,
        op(
            20,
            Type::F64,
            OperationKind::Constant(Constant::F64Bits(1.0f64.to_bits())),
        ),
    );
    blocks[1]
        .parameters
        .push(ValueDef::new(ValueId(21), Type::F64));
    let Terminator::ConditionalBranch { then_arguments, .. } =
        blocks[0].terminator.as_mut().unwrap()
    else {
        panic!("conditional branch expected")
    };
    then_arguments.push(ValueId(20));
    let error = lower_kernel_to_llvm_ir(&unsupported_type, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedType
    );
    assert_eq!(error.diagnostics()[0].location.block, Some(BlockId(1)));

    let mut unsupported_address_space = fill_module();
    let blocks = &mut unsupported_address_space.functions[0]
        .body
        .as_mut()
        .unwrap()
        .blocks;
    let pointer = Type::pointer(Type::F32, AddressSpace::Private, AccessMode::ReadWrite);
    blocks[0].operations.insert(
        0,
        op(
            20,
            pointer.clone(),
            OperationKind::Alloca {
                element: Type::F32,
                count: None,
                address_space: AddressSpace::Private,
                alignment: 4,
            },
        ),
    );
    blocks[1]
        .parameters
        .push(ValueDef::new(ValueId(21), pointer));
    let Terminator::ConditionalBranch { then_arguments, .. } =
        blocks[0].terminator.as_mut().unwrap()
    else {
        panic!("conditional branch expected")
    };
    then_arguments.push(ValueId(20));
    let error =
        lower_kernel_to_llvm_ir(&unsupported_address_space, &KernelId::new("fill")).unwrap_err();
    assert_eq!(
        error.diagnostics()[0].code,
        LoweringDiagnosticCode::UnsupportedAddressSpace
    );
    assert_eq!(error.diagnostics()[0].location.block, Some(BlockId(1)));
}

fn compile_scoped_atomics_for_target(target: &str) {
    canonical_test_target(target).unwrap();
    let clang = std::env::var_os("FE2O3_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/llvm/bin/clang"));
    let directory = TemporaryDirectory::new(&format!("g4-atomics-{target}"));
    let input = directory.join("scoped_atomics.ll");
    let output = directory.join("scoped_atomics.hsaco");
    let llvm_ir =
        lower_kernel_to_llvm_ir(&scoped_atomics_module(), &KernelId::new("scoped_atomics"))
            .unwrap();
    assert_eq!(llvm_ir, include_str!("fixtures/scoped_atomics.ll"));
    fs::write(&input, llvm_ir).unwrap();

    let result = Command::new(clang)
        .arg("--target=amdgcn-amd-amdhsa")
        .arg(format!("-mcpu={target}"))
        .arg("-nogpulib")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "clang failed for {target}:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fs::metadata(output).unwrap().len() > 64);
}

fn compile_wave_for_target(target: &str, width: WaveWidth) {
    canonical_test_target(target).unwrap();
    let clang = std::env::var_os("FE2O3_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/llvm/bin/clang"));
    let directory = TemporaryDirectory::new(&format!("g4-wave-{target}"));
    let input = directory.join("wave.ll");
    let output = directory.join("wave.hsaco");
    let llvm_ir =
        lower_kernel_to_llvm_ir(&wave_module(width), &KernelId::new("wave_kernel")).unwrap();
    fs::write(&input, llvm_ir).unwrap();

    let result = Command::new(clang)
        .arg("--target=amdgcn-amd-amdhsa")
        .arg(format!("-mcpu={target}"))
        .arg("-nogpulib")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "clang failed for {target} {width:?}:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fs::metadata(output).unwrap().len() > 64);
}

#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx1151 support"]
fn rocm_compiles_wave32_for_gfx1151() {
    compile_wave_for_target("gfx1151", WaveWidth::Wave32);
}

#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx942 support"]
fn rocm_compiles_wave64_for_gfx942() {
    compile_wave_for_target("gfx942", WaveWidth::Wave64);
}

#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx950 support"]
fn rocm_compiles_exact_gfx950_bf16_mfma_to_code_object() {
    let clang = std::env::var_os("FE2O3_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/llvm/bin/clang"));
    let directory = TemporaryDirectory::new("gfx950-bf16-mfma");
    let input = directory.join("gfx950_bf16_mfma.ll");
    let output = directory.join("gfx950_bf16_mfma.hsaco");
    let llvm =
        lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(&gfx950_bf16_mfma_module()).unwrap();
    fs::write(&input, llvm).unwrap();

    let result = Command::new(&clang)
        .arg("--target=amdgcn-amd-amdhsa")
        .arg("-mcpu=gfx950")
        .arg("-nogpulib")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "clang failed for gfx950 BF16 MFMA:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fs::metadata(&output).unwrap().len() > 64);

    let objdump = std::env::var_os("FE2O3_ROCM_LLVM_OBJDUMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut tool = clang;
            tool.set_file_name("llvm-objdump");
            tool
        });
    let result = Command::new(objdump)
        .arg("--disassemble")
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success());
    let disassembly = String::from_utf8(result.stdout).unwrap();
    assert!(
        disassembly.contains("v_mfma_f32_16x16x16_bf16"),
        "{disassembly}"
    );
}
#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx950 support"]
fn rocm_compiles_wave64_for_gfx950() {
    compile_wave_for_target("gfx950", WaveWidth::Wave64);
}

#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx1151 support"]
fn rocm_compiles_scoped_atomics_for_gfx1151() {
    compile_scoped_atomics_for_target("gfx1151");
}

#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx942 support"]
fn rocm_compiles_scoped_atomics_for_gfx942() {
    compile_scoped_atomics_for_target("gfx942");
}

#[test]
#[ignore = "requires a ROCm LLVM toolchain with gfx950 support"]
fn rocm_compiles_scoped_atomics_for_gfx950() {
    compile_scoped_atomics_for_target("gfx950");
}

#[test]
#[ignore = "requires the ROCm LLVM toolchain"]
fn rocm_compiles_phi_golden_to_an_amdgpu_code_object() {
    let clang = std::env::var_os("FE2O3_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/llvm/bin/clang"));
    let target_text = std::env::var("FE2O3_TARGET")
        .expect("FE2O3_TARGET must name an exact canonical AMDGPU target");
    canonical_test_target(&target_text).unwrap();
    let directory = TemporaryDirectory::new("g1-phi");
    let input = directory.join("phi_loop.ll");
    let output = directory.join("phi_loop.hsaco");
    let llvm_ir = lower_kernel_to_llvm_ir(&phi_loop_module(), &KernelId::new("phi_loop")).unwrap();
    assert_eq!(llvm_ir, include_str!("fixtures/phi_loop_g1.ll"));
    fs::write(&input, llvm_ir).unwrap();

    let result = Command::new(clang)
        .arg("--target=amdgcn-amd-amdhsa")
        .arg(format!("-mcpu={target_text}"))
        .arg("-nogpulib")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fs::metadata(output).unwrap().len() > 64);
}

#[test]
#[ignore = "requires the ROCm LLVM toolchain"]
fn rocm_compiles_g4_synchronization_and_lds() {
    let clang = std::env::var_os("FE2O3_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/llvm/bin/clang"));
    let target_text = std::env::var("FE2O3_TARGET")
        .expect("FE2O3_TARGET must name an exact canonical AMDGPU target");
    canonical_test_target(&target_text).unwrap();
    let directory = TemporaryDirectory::new("g4-sync-lds");
    let input = directory.join("g4_sync_lds.ll");
    let output = directory.join("g4_sync_lds.hsaco");
    let llvm_ir =
        lower_kernel_to_llvm_ir(&g4_synchronization_module(), &KernelId::new("fill")).unwrap();
    fs::write(&input, llvm_ir).unwrap();

    let result = Command::new(&clang)
        .arg("--target=amdgcn-amd-amdhsa")
        .arg(format!("-mcpu={target_text}"))
        .arg("-nogpulib")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fs::metadata(&output).unwrap().len() > 64);

    let readobj = std::env::var_os("FE2O3_ROCM_LLVM_READOBJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut tool = clang;
            tool.set_file_name("llvm-readobj");
            tool
        });
    let result = Command::new(readobj)
        .arg("--notes")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "llvm-readobj failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = String::from_utf8(result.stdout).unwrap();
    let metadata = metadata(&report);
    assert!(
        metadata.contains("    .group_segment_fixed_size: 256\n"),
        "static LDS was not reflected in AMDGPU metadata:\n{metadata}"
    );
}

#[test]
#[ignore = "requires the ROCm LLVM toolchain"]
fn rocm_compiles_the_golden_to_an_amdgpu_code_object() {
    let clang = std::env::var_os("FE2O3_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/llvm/bin/clang"));
    let target_text = std::env::var("FE2O3_TARGET")
        .expect("FE2O3_TARGET must name an exact canonical AMDGPU target");
    let processor = canonical_test_target(&target_text).unwrap();
    let directory = TemporaryDirectory::new("g1");
    let input = directory.join("fill.ll");
    let output = directory.join("fill.hsaco");
    let llvm_ir = lower_kernel_to_llvm_ir(&fill_module(), &KernelId::new("fill")).unwrap();
    assert_eq!(llvm_ir, include_str!("fixtures/fill_g1.ll"));
    assert_eq!(
        llvm_ir,
        lower_kernel_to_llvm_ir(&fill_module(), &KernelId::new("fill")).unwrap()
    );
    fs::write(&input, &llvm_ir).unwrap();
    assert_eq!(fs::read_to_string(&input).unwrap(), llvm_ir);

    let result = Command::new(&clang)
        .arg("--target=amdgcn-amd-amdhsa")
        .arg(format!("-mcpu={target_text}"))
        .arg("-nogpulib")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "clang failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let object = fs::read(&output).unwrap();
    assert!(object.len() > 64);

    let readobj = std::env::var_os("FE2O3_ROCM_LLVM_READOBJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut tool = clang;
            tool.set_file_name("llvm-readobj");
            tool
        });
    let result = Command::new(readobj)
        .args(["--file-header", "--symbols", "--notes"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "llvm-readobj failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = String::from_utf8(result.stdout).unwrap();

    for invariant in [
        "Format: elf64-amdgpu",
        "Arch: amdgcn",
        "Class: 64-bit (0x2)",
        "DataEncoding: LittleEndian (0x1)",
        "OS/ABI: AMDGPU_HSA (0x40)",
        "ABIVersion: 4",
        "Type: SharedObject (0x3)",
        "Machine: EM_AMDGPU (0xE0)",
    ] {
        assert!(
            report.contains(invariant),
            "missing {invariant:?} in:\n{report}"
        );
    }
    let processor_flag = format!("EF_AMDGPU_MACH_AMDGCN_{}", processor.to_ascii_uppercase());
    assert_occurrences(&report, &processor_flag, 1);
    for feature in target_text.split(':').skip(1) {
        let feature_flag = match feature {
            "sramecc+" => "EF_AMDGPU_FEATURE_SRAMECC_ON_V4",
            "sramecc-" => "EF_AMDGPU_FEATURE_SRAMECC_OFF_V4",
            "xnack+" => "EF_AMDGPU_FEATURE_XNACK_ON_V4",
            "xnack-" => "EF_AMDGPU_FEATURE_XNACK_OFF_V4",
            _ => unreachable!("canonical_test_target rejected unknown features"),
        };
        assert_occurrences(&report, feature_flag, 1);
    }

    let entry = symbol_block(&report, "fill");
    assert!(entry.contains("    Binding: Global (0x1)"));
    assert!(entry.contains("    Type: Function (0x2)"));
    assert!(entry.contains("      STV_PROTECTED (0x3)"));
    assert!(entry.contains("    Section: .text"));
    let descriptor = symbol_block(&report, "fill.kd");
    assert!(descriptor.contains("    Size: 64"));
    assert!(descriptor.contains("    Binding: Global (0x1)"));
    assert!(descriptor.contains("    Type: Object (0x1)"));
    assert!(descriptor.contains("    Section: .rodata"));
    let global_functions = symbol_blocks(&report)
        .into_iter()
        .filter(|block| {
            block.contains("    Binding: Global (0x1)")
                && block.contains("    Type: Function (0x2)")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        global_functions.len(),
        1,
        "unexpected global kernel symbols"
    );

    assert_occurrences(&report, "Type: NT_AMDGPU_METADATA (AMDGPU Metadata)", 1);
    let metadata = metadata(&report);
    assert_occurrences(metadata, "    .symbol:         fill.kd", 1);
    assert_occurrences(metadata, "    .name:           fill", 1);
    assert!(metadata.contains("    .group_segment_fixed_size: 0\n"));
    assert!(metadata.contains("    .kernarg_segment_align: 8\n"));
    assert!(metadata.contains("    .max_flat_workgroup_size: 64\n"));
    assert!(metadata.contains("    .reqd_workgroup_size:\n      - 64\n      - 1\n      - 1\n"));
    assert!(metadata.contains("    .uses_dynamic_stack: false\n"));
    assert_metadata_argument(metadata, "arg0.data", 0, 8, "global_buffer", Some("global"));
    assert_metadata_argument(metadata, "arg0.len", 8, 8, "by_value", None);
    assert_metadata_argument(metadata, "arg1", 16, 4, "by_value", None);
    assert_occurrences(metadata, ".name:", 4);
    assert_occurrences(metadata, "amdhsa.version:\n  - 1\n  - 2", 1);
    let expected_target = if target_text.contains(':') {
        format!("amdhsa.target:   'amdgcn-amd-amdhsa--{target_text}'")
    } else {
        format!("amdhsa.target:   amdgcn-amd-amdhsa--{target_text}")
    };
    assert_occurrences(metadata, &expected_target, 1);
}

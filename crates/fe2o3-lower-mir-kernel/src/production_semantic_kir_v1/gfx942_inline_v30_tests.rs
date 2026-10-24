use fe2o3_kernel_ir::{
    AssemblyConstraint, AssemblyOperandKind, AssemblyOption, InlineAssemblyTarget,
    validate_gfx942_inline_assembly_v1,
};
use fe2o3_mir_model::semantic_mir_v1::*;

const ISA_KINDS: [SemanticGfx942InlineInstructionV30; 6] = [
    SemanticGfx942InlineInstructionV30::VMovB32,
    SemanticGfx942InlineInstructionV30::VAddU32,
    SemanticGfx942InlineInstructionV30::VSubU32,
    SemanticGfx942InlineInstructionV30::VAndB32,
    SemanticGfx942InlineInstructionV30::VOrB32,
    SemanticGfx942InlineInstructionV30::VXorB32,
];
const ISA_BLOCK: SemanticBlockIdV1 = SemanticBlockIdV1::from_index(0);
const ISA_TYPE: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(0);

#[path = "gfx942_inline_v30_pre_ranked_tests.rs"]
mod pre_ranked_tests;

fn u32_abi_value() -> SemanticAbiValueV1 {
    // Same initialized-scalar/noundef ABI profile as the admitted helper in
    // production_slice_call_composition_v1_tests.rs.
    SemanticAbiValueV1::new(
        ISA_TYPE,
        SemanticAbiPassModeV1::Direct(
            SemanticAbiValueAttributesV1::new(
                SemanticAbiRegularAttributesV1::new(false, None, false, false, false, true),
                SemanticAbiExtensionV1::None,
                0,
                None,
            )
            .unwrap(),
        ),
    )
}

struct IsaFixture {
    types: Vec<SemanticTypeDeclV1>,
    callables: Vec<SemanticCallableDeclV1>,
    function: SemanticFunctionDeclV1,
    call: SemanticDirectCallV1,
}

impl IsaFixture {
    fn new(kind: SemanticGfx942InlineInstructionV30) -> Self {
        Self::with_intrinsic_callee(kind, SemanticCallableIdV1::from_index(0))
    }

    fn with_intrinsic_callee(
        kind: SemanticGfx942InlineInstructionV30,
        callee: SemanticCallableIdV1,
    ) -> Self {
        let source = SemanticSourceProvenanceV1::unavailable();
        let types = vec![SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([1; 32]),
            SemanticLayoutIdentityV1::from_sha256([2; 32]),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(4),
                4,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::integer(false, 32, 4),
                    SemanticScalarValidityRangeV1::new(0, u32::MAX.into()),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 32,
            }),
        )];
        let value = u32_abi_value();
        let abi = |count| {
            SemanticFunctionAbiV1::new(
                SemanticAbiIdentityV1::from_sha256([3; 32]),
                SemanticLayoutIdentityV1::from_sha256([4; 32]),
                SemanticCanonAbiV1::Rust,
                false,
                false,
                vec![value.clone(); count],
                value.clone(),
            )
            .unwrap()
        };
        let function_identity = SemanticFunctionIdentityV1::from_sha256([10; 32]);
        let callables = vec![SemanticCallableDeclV1::CompilerIntrinsic {
            binding: SemanticNonBodyCallableBindingV1::new(
                SemanticFunctionIdentityV1::from_sha256([20; 32]),
                SemanticItemDefinitionIdentityV1::from_sha256([21; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([22; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([23; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([24; 32]),
                source,
                abi(kind.input_count()),
            ),
            operation: SemanticCompilerIntrinsicOperationV1::Gfx942InlineU32(
                SemanticGfx942InlineU32V30::new(kind, 1).unwrap(),
            ),
            operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([25; 32]),
        }];
        let call = SemanticDirectCallV1::new_callable(
            callee,
            (0..kind.input_count())
                .map(|index| {
                    SemanticOperandV1::Copy(
                        SemanticPlaceV1::new(
                            SemanticLocalIdV1::from_index(index as u32 + 1),
                            vec![],
                            ISA_TYPE,
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            Some(SemanticCallDestinationV1::new(
                SemanticPlaceV1::new(SemanticLocalIdV1::from_index(0), vec![], ISA_TYPE).unwrap(),
                SemanticControlFlowEdgeV1::new(
                    SemanticEdgeRoleV1::CallReturn,
                    SemanticBlockIdV1::from_index(1),
                ),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap()
        .with_inline_assembly_source_v30(
            SemanticInlineAssemblySourceV30::new([30; 32], function_identity, [31; 32], [32; 32])
                .unwrap(),
        );
        let local = |tag, role| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([tag; 32]),
                ISA_TYPE,
                role,
                source,
            )
        };
        let function = SemanticFunctionDeclV1::new(
            function_identity,
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([11; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([12; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([13; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([14; 32]),
            source,
            abi(2),
            vec![
                local(40, SemanticLocalRoleV1::Return),
                local(41, SemanticLocalRoleV1::Argument(0)),
                local(42, SemanticLocalRoleV1::Argument(1)),
            ],
            ISA_BLOCK,
            vec![
                SemanticBasicBlockV1::new(
                    SemanticBlockIdentityV1::from_sha256([50; 32]),
                    source,
                    vec![],
                    SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Call(call.clone())),
                )
                .unwrap(),
                SemanticBasicBlockV1::new(
                    SemanticBlockIdentityV1::from_sha256([51; 32]),
                    source,
                    vec![],
                    SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        Self {
            types,
            callables,
            function,
            call,
        }
    }

    fn lowering<'a, 'service, 'work>(
        &'a self,
        maximum_operations: usize,
        budget: &'service mut ArgumentBudgetV1<'work>,
    ) -> SemanticFunctionLoweringV1<'a, 'service> {
        let mut lowering = SemanticFunctionLoweringV1::new(
            &self.types,
            &self.callables,
            &self.function,
            SemanticParameterBindingsV1 {
                declarations: &[(0, 1, ISA_TYPE), (1, 2, ISA_TYPE)],
                values: &[ValueId(0), ValueId(1)],
                types: &[Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::U32)],
                local_bindings: None,
            },
            None,
            None,
            BTreeSet::new(),
            1,
            false,
            64,
            budget,
        )
        .unwrap();
        // The constructor also meters source analysis with this limit. Isolate
        // the instruction emission boundary after constructing the SSA plan.
        lowering.max_operations = maximum_operations;
        lowering
    }
}

#[test]
fn all_six_calls_preserve_exact_isa_operands_source_and_options() {
    for kind in ISA_KINDS {
        let fixture = IsaFixture::new(kind);
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(1, &mut lowering_budget);
        let mut target = BasicBlock::new(BlockId(0));
        lowering.begin_block(ISA_BLOCK, &mut target).unwrap();
        let terminator = lowering
            .lower_call(ISA_BLOCK, &fixture.call, &mut target.operations)
            .unwrap();
        assert!(matches!(
            terminator,
            Terminator::Branch {
                target: BlockId(1),
                ..
            }
        ));
        assert_eq!(target.operations.len(), 1);
        let operation = &target.operations[0];
        let OperationKind::InlineAssembly(assembly) = &operation.kind else {
            panic!("ISA became an ordinary operation")
        };
        assert_eq!(assembly.mnemonic, kind.mnemonic());
        assert_eq!(assembly.target, InlineAssemblyTarget::AmdGpuGfx942);
        assert_eq!(assembly.options, BTreeSet::from([AssemblyOption::NoMemory]));
        assert!(assembly.declared_effects.is_empty());
        let source = fixture.call.inline_assembly_source_v30().unwrap();
        assert_eq!(assembly.source.frontend_unit, source.frontend_unit());
        assert_eq!(
            assembly.source.function,
            *fixture.function.identity().as_bytes()
        );
        assert_eq!(assembly.source.contract, source.contract());
        assert_eq!(assembly.source.statement, source.statement());
        assert_eq!(
            assembly.operands[0].kind,
            AssemblyOperandKind::Output { result_index: 0 }
        );
        assert_eq!(assembly.operands.len(), kind.input_count() + 1);
        for (index, operand) in assembly.operands[1..].iter().enumerate() {
            assert_eq!(
                operand.kind,
                AssemblyOperandKind::Input(ValueId(index as u32))
            );
        }
        assert!(
            assembly
                .operands
                .iter()
                .all(|operand| operand.constraint == AssemblyConstraint::Vgpr32)
        );
        let checked = validate_gfx942_inline_assembly_v1(operation, |value| {
            (value.0 < 2).then_some(ScalarType::U32)
        })
        .unwrap();
        assert_eq!(checked.instruction().mnemonic(), kind.mnemonic());
        assert_eq!(
            operation.required_capabilities(),
            assembly.required_capabilities()
        );
        assert!(!operation.required_capabilities().is_empty());
        // This emission slice must not grant final effect/refinement authority.
        assert!(!operation.has_complete_effect_summary());
        assert!(lowering.emitted_unsigned_constants.is_empty());
        assert!(lowering.emitted_u32_bitand_masks.is_empty());
    }
}

#[test]
fn isa_occurrence_caller_substitution_and_absence_are_rejected_before_emission() {
    let fixture = IsaFixture::new(ISA_KINDS[1]);
    let bare = SemanticDirectCallV1::new_callable(
        fixture.call.callee(),
        fixture.call.arguments().to_vec(),
        fixture.call.destination().cloned(),
        fixture.call.unwind(),
    )
    .unwrap();
    let wrong = bare.clone().with_inline_assembly_source_v30(
        SemanticInlineAssemblySourceV30::new(
            [30; 32],
            SemanticFunctionIdentityV1::from_sha256([99; 32]),
            [31; 32],
            [32; 32],
        )
        .unwrap(),
    );
    for call in [bare, wrong] {
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(1, &mut lowering_budget);
        let mut operations = Vec::new();
        assert!(
            lowering
                .lower_call(ISA_BLOCK, &call, &mut operations)
                .is_err()
        );
        assert!(operations.is_empty());
        assert_eq!(lowering.emitted_operations, 0);
    }
}

#[test]
fn isa_arity_scalar_contract_and_operation_budgets_are_exact() {
    let fixture = IsaFixture::new(ISA_KINDS[1]);
    let source = fixture.call.inline_assembly_source_v30().unwrap();
    for arguments in [
        vec![],
        fixture.call.arguments()[..1].to_vec(),
        vec![fixture.call.arguments()[0].clone(); 3],
    ] {
        let call = SemanticDirectCallV1::new_callable(
            fixture.call.callee(),
            arguments,
            fixture.call.destination().cloned(),
            fixture.call.unwind(),
        )
        .unwrap()
        .with_inline_assembly_source_v30(source);
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(1, &mut lowering_budget);
        let mut operations = Vec::new();
        assert!(
            lowering
                .lower_call(ISA_BLOCK, &call, &mut operations)
                .is_err()
        );
        assert!(operations.is_empty());
    }
    let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
    let mut lowering = fixture.lowering(0, &mut lowering_budget);
    let mut operations = Vec::new();
    assert!(matches!(
        lowering.lower_call(ISA_BLOCK, &fixture.call, &mut operations),
        Err(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::Operations,
            actual: 1,
            limit: 0
        })
    ));
    assert!(operations.is_empty());
    let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
    let mut lowering = fixture.lowering(1, &mut lowering_budget);
    lowering.locals[1] = Some(SemanticValueBindingV1::Value {
        id: ValueId(0),
        ty: Type::Scalar(ScalarType::I32),
    });
    assert!(
        lowering
            .lower_gfx942_inline_u32_v30(
                ISA_BLOCK,
                &fixture.call,
                SemanticGfx942InlineU32V30::new(ISA_KINDS[1], 1).unwrap(),
                &mut operations
            )
            .is_err()
    );
    assert!(operations.is_empty());
}

use dialect_kernel::{
    AccessKindAttr, AlgorithmOp, AlgorithmType, AllocationEffectOp, AnalysisSplitControlCountAttr,
    AnalysisSplitOp, AtomicOrderingAttr, AtomicScopeAttr, BranchArgsOp, BranchOp,
    CheckedRowStripedIndex2DOp, CheckedTiledIndex2DOp, DIALECT_NAME, DYNAMIC_EXTENT,
    DeterministicJoinOp, DimensionAttr, DimensionOp,
    GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
    GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
    GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
    GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1, ITERATION_DOMAIN_ATTR_KEY, IndexConstantOp,
    IndexEqualBranchArgsOp, IndexEqualBranchOp, IndexLessThanBranchArgsOp, IndexLessThanBranchOp,
    IndexType, IndexValueAttr, IterationDomainAttr, KernelError, MAX_DETERMINISTIC_JOIN_INPUTS_V1,
    MAX_ITERATION_RANK, MAX_RANKED_MEMORY_RANK, MemorySpaceAttr, OwnershipContractOp,
    OwnershipCoverageAttr, OwnershipPartitionAttr, RankedAccessOp, RankedMemoryError, RankedViewOp,
    RankedViewType, RegistrationError, RegistrationOutcome, SemanticExceptionalValueAttr,
    SemanticIeeeRoundingAttr, SemanticNumericalPolicyAttr, SemanticOverflowAttr, SemanticOwner,
    SemanticScalarKindAttr, SemanticTypedBinaryKindAttr, SemanticTypedBinaryOp,
    SemanticTypedConstantOp, SemanticTypedExpressionRootOp, SemanticTypedScalarV1,
    SemanticTypedSymbolOp, StructuredAlgorithmOp, TensorConvergenceAttr, TensorLayoutOp,
    register_dialect,
};

#[test]
fn typed_semantic_verifiers_reject_bad_width_arity_and_ieee_policy() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let u32_scalar =
        SemanticTypedScalarV1::new(SemanticScalarKindAttr::UnsignedInteger, 32).unwrap();
    let symbol = SemanticTypedSymbolOp::new(context, 1, u32_scalar);
    let oversized = SemanticTypedConstantOp::new(context, 1_u64 << 32, u32_scalar);
    assert!(verify_op(&oversized, context).is_err());

    let constant = SemanticTypedConstantOp::new(context, 4, u32_scalar);
    let binary = SemanticTypedBinaryOp::new(
        context,
        SemanticTypedBinaryKindAttr::Add,
        SemanticOverflowAttr::Wrapping,
        u32_scalar,
        symbol.result(context),
        constant.result(context),
    );
    Operation::remove_operand(binary.get_operation(), context, 1);
    assert!(verify_op(&binary, context).is_err());

    let bad_policy = SemanticTypedExpressionRootOp::new(
        context,
        symbol.result(context),
        SemanticNumericalPolicyAttr::ExactIeeeNearestTiesToEvenPreserveBits,
        SemanticIeeeRoundingAttr::TowardZero,
        SemanticExceptionalValueAttr::PreserveExactBits,
        [1, 2, 3, 4],
    );
    assert!(verify_op(&bad_policy, context).is_err());
}
use fe2o3_kernel_ir::{TensorInstructionProfileV1, TensorLayoutContractV1, TensorSymbolicMapV1};
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::{
        attributes::BytesAttr, op_interfaces::SingleBlockRegionInterface, ops::ModuleOp,
        types::UnitType,
    },
    common_traits::Verify,
    context::Context,
    dialect::DialectName,
    identifier::Identifier,
    op::{Op, op_cast, verify_op},
    operation::{Operation, verify_operation},
    parsable::{Parsable, parse_from_str},
    printable::Printable,
    r#type::{TypeHandle, Typed},
};

#[test]
fn branches_require_exact_successor_arguments() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let index: TypeHandle = IndexType::get(context).into();
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let argument_target = BasicBlock::new(context, None, vec![index]);
    let empty_target = BasicBlock::new(context, None, vec![]);

    assert!(verify_op(&BranchOp::new(context, argument_target), context).is_err());
    assert!(
        verify_op(
            &IndexLessThanBranchOp::new(
                context,
                zero.result(context),
                one.result(context),
                argument_target,
                empty_target,
            ),
            context,
        )
        .is_err()
    );
    assert!(
        verify_op(
            &BranchArgsOp::new(context, vec![], argument_target),
            context
        )
        .is_err()
    );
    assert!(
        verify_op(
            &IndexLessThanBranchArgsOp::new(
                context,
                zero.result(context),
                one.result(context),
                vec![],
                vec![],
                argument_target,
                empty_target,
            ),
            context,
        )
        .is_err()
    );
    assert!(
        verify_op(
            &IndexEqualBranchOp::new(
                context,
                zero.result(context),
                one.result(context),
                argument_target,
                empty_target,
            ),
            context,
        )
        .is_err()
    );
    assert!(
        verify_op(
            &IndexEqualBranchArgsOp::new(
                context,
                zero.result(context),
                one.result(context),
                vec![],
                vec![],
                argument_target,
                empty_target,
            ),
            context,
        )
        .is_err()
    );

    let attributed = IndexEqualBranchOp::new(
        context,
        zero.result(context),
        one.result(context),
        empty_target,
        empty_target,
    );
    attributed
        .get_operation()
        .deref_mut(context)
        .attributes
        .0
        .insert(
            "kernel_index_value".try_into().expect("valid key"),
            Box::new(IndexValueAttr(0)),
        );
    assert!(verify_op(&attributed, context).is_err());

    let foreign = AlgorithmOp::new(context, 1).unwrap();
    let foreign_result = foreign.get_operation().deref(context).get_result(0);
    assert!(
        verify_op(
            &IndexEqualBranchArgsOp::new(
                context,
                zero.result(context),
                one.result(context),
                vec![foreign_result],
                vec![],
                argument_target,
                empty_target,
            ),
            context,
        )
        .is_err()
    );
}

#[test]
fn analysis_split_binds_control_and_successor_operand_segments_exactly() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let index: TypeHandle = IndexType::get(context).into();
    let first = BasicBlock::new(context, None, vec![index.clone()]);
    let second = BasicBlock::new(context, None, vec![index]);
    let zero = IndexConstantOp::new(context, 0);
    let one = IndexConstantOp::new(context, 1);
    let valid = AnalysisSplitOp::new_with_control_and_arguments(
        context,
        vec![zero.result(context)],
        vec![zero.result(context)],
        vec![one.result(context)],
        first,
        second,
    );
    verify_op(&valid, context).unwrap();
    assert_eq!(
        valid.control_dependencies(context),
        vec![zero.result(context)]
    );
    assert_eq!(valid.first_arguments(context), vec![zero.result(context)]);
    assert_eq!(valid.second_arguments(context), vec![one.result(context)]);

    valid.set_attr_kernel_analysis_split_control_count(context, AnalysisSplitControlCountAttr(2));
    assert!(verify_op(&valid, context).is_err());
    valid.set_attr_kernel_analysis_split_control_count(context, AnalysisSplitControlCountAttr(1));
    Operation::pop_operand(valid.get_operation(), context);
    assert!(verify_op(&valid, context).is_err());

    let foreign = AlgorithmOp::new(context, 1).unwrap();
    let foreign_result = foreign.get_operation().deref(context).get_result(0);
    let wrong_control = AnalysisSplitOp::new_with_control_and_arguments(
        context,
        vec![foreign_result],
        vec![zero.result(context)],
        vec![one.result(context)],
        first,
        second,
    );
    assert!(verify_op(&wrong_control, context).is_err());
}

#[test]
fn deterministic_join_is_bounded_and_carries_no_authority() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let dependency = IndexConstantOp::new(context, 7);
    let join = DeterministicJoinOp::new(context, vec![dependency.result(context)]);
    verify_op(&join, context).expect("one exact index dependency");
    assert!(!join.grants_compiler_refinement_authority());
    assert!(!join.grants_artifact_or_launch_authority());

    let empty = DeterministicJoinOp::new(context, vec![]);
    assert!(verify_op(&empty, context).is_err());
    let oversized = DeterministicJoinOp::new(
        context,
        vec![dependency.result(context); MAX_DETERMINISTIC_JOIN_INPUTS_V1 + 1],
    );
    assert!(verify_op(&oversized, context).is_err());

    let foreign = AlgorithmOp::new(context, 1).unwrap();
    let foreign_result = foreign.get_operation().deref(context).get_result(0);
    let malformed = DeterministicJoinOp::new(context, vec![foreign_result]);
    assert!(verify_op(&malformed, context).is_err());
}

fn kernel_name() -> DialectName {
    DialectName::try_new(DIALECT_NAME).expect("valid dialect")
}

#[test]
fn tensor_layout_round_trips_full_opaque_identities_without_aliasing() {
    let mut context = Context::new();
    register_dialect(&mut context, &kernel_name()).unwrap();
    let mut contract = TensorLayoutContractV1::gfx942_mfma_bf16_f32_m16n16k16_wave64();
    contract.profile = TensorInstructionProfileV1::Opaque(u32::MAX);
    contract.a.mapping = TensorSymbolicMapV1::Opaque(u32::MAX);
    let op = TensorLayoutOp::new(
        &mut context,
        &contract,
        TensorConvergenceAttr::UniformSubgroup,
        64,
    );
    verify_op(&op, &context).expect("locally well-formed opaque contract");
    assert_eq!(op.contract(&context).unwrap(), contract);
}

#[test]
fn tensor_layout_round_trips_exact_gfx950_fp4_profile() {
    let mut context = Context::new();
    register_dialect(&mut context, &kernel_name()).unwrap();
    let contract = TensorLayoutContractV1::gfx950_scaled_mfma_fp4_e2m1_f32_m16n16k128_wave64();
    let op = TensorLayoutOp::new(
        &mut context,
        &contract,
        TensorConvergenceAttr::UniformSubgroup,
        64,
    );
    verify_op(&op, &context).expect("locally well-formed gfx950 FP4 contract");
    assert_eq!(op.contract(&context).unwrap(), contract);
}

#[test]
fn registration_is_real_idempotent_and_collision_safe() {
    let context = &mut Context::new();
    assert_eq!(
        register_dialect(context, &kernel_name()),
        Ok(RegistrationOutcome::Registered)
    );
    assert_eq!(
        register_dialect(context, &kernel_name()),
        Ok(RegistrationOutcome::AlreadyRegistered)
    );
    assert_eq!(
        register_dialect(
            context,
            &DialectName::try_new("schedule").expect("valid dialect")
        ),
        Err(RegistrationError::WrongDialect)
    );

    let parsed = parse_from_str(TypeHandle::parser(()), context, "kernel.algorithm <2>")
        .expect("registered type parses");
    assert!(parsed.verify(context).is_ok());
    let parsed = parse_from_str(
        <Box<dyn Attribute>>::parser(()),
        context,
        "kernel.iteration_domain <2>",
    )
    .expect("registered attribute parses");
    assert!(parsed.verify(context).is_ok());

    let operation = AlgorithmOp::new(context, 2).expect("bounded algorithm");
    let module = ModuleOp::new(context, "registration".try_into().expect("valid name"));
    module.append_operation(context, operation.get_operation(), 0);
    let printed = module.get_operation().disp(context).to_string();
    let parsed = parse_from_str(Operation::top_level_parser(), context, &printed)
        .expect("registered operation parses");
    verify_operation(parsed, context).expect("parsed operation verifies");
}

#[test]
fn hostile_registration_marker_is_rejected() {
    let context = &mut Context::new();
    let key: Identifier = "fe2o3_dialect_kernel_registration_v1"
        .try_into()
        .expect("valid key");
    let hostile = context.aux_data.insert(Box::new(17_u32));
    context.aux_data_map.insert(key, hostile);
    assert_eq!(
        register_dialect(context, &kernel_name()),
        Err(RegistrationError::MarkerCollision)
    );
    context.aux_data.remove(hostile);
    assert_eq!(
        register_dialect(context, &kernel_name()),
        Err(RegistrationError::CorruptMarker)
    );
}

#[test]
fn constructors_and_parsed_values_enforce_rank_bounds() {
    let context = &mut Context::new();
    assert_eq!(
        AlgorithmType::new(context, 0).unwrap_err(),
        KernelError::IterationRankOutOfBounds(0)
    );
    assert_eq!(
        IterationDomainAttr::new(MAX_ITERATION_RANK + 1).unwrap_err(),
        KernelError::IterationRankOutOfBounds(MAX_ITERATION_RANK + 1)
    );

    let parsed = parse_from_str(TypeHandle::parser(()), context, "kernel.algorithm <0>")
        .expect("syntax is valid before semantic verification");
    assert!(parsed.verify(context).is_err());
    let parsed = parse_from_str(
        <Box<dyn Attribute>>::parser(()),
        context,
        "kernel.iteration_domain <4294967295>",
    )
    .expect("bounded scalar syntax parses");
    assert!(parsed.verify(context).is_err());

    let parsed = parse_from_str(
        TypeHandle::parser(()),
        context,
        "kernel.ranked_view <32, false, [0, 64]>",
    )
    .expect("ranked dynamic type parses");
    assert!(parsed.verify(context).is_ok());
}

#[test]
fn ranked_view_constructors_enforce_rank_width_and_dynamic_extent_count() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    assert_eq!(
        RankedViewType::new(context, 32, false, vec![]).unwrap_err(),
        RankedMemoryError::RankOutOfBounds(0),
    );
    assert_eq!(
        RankedViewType::new(context, 32, false, vec![1; MAX_RANKED_MEMORY_RANK + 1]).unwrap_err(),
        RankedMemoryError::RankOutOfBounds(MAX_RANKED_MEMORY_RANK + 1),
    );
    assert_eq!(
        RankedViewType::new(context, 24, false, vec![16]).unwrap_err(),
        RankedMemoryError::UnsupportedElementWidth(24),
    );
    assert!(
        RankedViewType::new(context, 128, true, vec![1; MAX_RANKED_MEMORY_RANK]).is_ok(),
        "the documented maximum rank must remain admitted",
    );

    let first_extent = IndexConstantOp::new(context, 17);
    let second_extent = IndexConstantOp::new(context, 19);
    let dynamic =
        RankedViewType::new(context, 32, false, vec![DYNAMIC_EXTENT, 64, DYNAMIC_EXTENT]).unwrap();
    assert_eq!(
        RankedViewOp::new(context, dynamic, vec![]).err().unwrap(),
        RankedMemoryError::DynamicExtentCountMismatch {
            expected: 2,
            actual: 0,
        },
    );
    let dynamic =
        RankedViewType::new(context, 32, false, vec![DYNAMIC_EXTENT, 64, DYNAMIC_EXTENT]).unwrap();
    let view = RankedViewOp::new(
        context,
        dynamic,
        vec![first_extent.result(context), second_extent.result(context)],
    )
    .unwrap();
    assert_eq!(
        view.dynamic_extent(context, 0),
        Some(first_extent.result(context))
    );
    assert_eq!(view.dynamic_extent(context, 1), None);
    assert_eq!(
        view.dynamic_extent(context, 2),
        Some(second_extent.result(context))
    );
    assert_eq!(view.dynamic_extent(context, 3), None);

    let foreign = AlgorithmOp::new(context, 1).unwrap();
    let foreign_result = foreign.get_operation().deref(context).get_result(0);
    let foreign_extent_type =
        RankedViewType::new(context, 32, false, vec![DYNAMIC_EXTENT]).unwrap();
    let foreign_extent =
        RankedViewOp::new(context, foreign_extent_type, vec![foreign_result]).unwrap();
    assert!(verify_op(&foreign_extent, context).is_err());
}

#[test]
fn ranked_view_allocation_contract_is_explicit_and_fail_closed() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let ty = RankedViewType::new(context, 32, true, vec![16]).unwrap();
    let authenticated = RankedViewOp::new_in_space_with_allocation_contract(
        context,
        ty,
        vec![],
        MemorySpaceAttr::Global,
        17,
        23,
    )
    .unwrap();
    verify_op(&authenticated, context).unwrap();
    assert_eq!(authenticated.allocation_origin(context), Some(17));
    assert_eq!(authenticated.noalias_class(context), Some(23));

    let ty = RankedViewType::new(context, 32, true, vec![16]).unwrap();
    let forged = RankedViewOp::new_in_space_with_allocation_contract(
        context,
        ty,
        vec![],
        MemorySpaceAttr::Global,
        0,
        23,
    )
    .unwrap();
    assert!(verify_op(&forged, context).is_err());
}

#[test]
fn allocation_effect_is_global_non_atomic_and_authenticated() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let effect = AllocationEffectOp::new(
        context,
        AccessKindAttr::Read,
        MemorySpaceAttr::Global,
        17,
        23,
    )
    .unwrap();
    verify_op(&effect, context).unwrap();
    assert_eq!(effect.kind(context), Some(AccessKindAttr::Read));
    assert_eq!(effect.memory_space(context), Some(MemorySpaceAttr::Global));
    assert_eq!(effect.allocation_origin(context), Some(17));
    assert_eq!(effect.noalias_class(context), Some(23));

    for (origin, noalias) in [
        (
            GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
            GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
        ),
        (
            GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
            GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1,
        ),
    ] {
        for kind in [AccessKindAttr::Read, AccessKindAttr::Write] {
            let transpose =
                AllocationEffectOp::new(context, kind, MemorySpaceAttr::Workgroup, origin, noalias)
                    .expect("reserved gfx950 transpose effect");
            verify_op(&transpose, context).expect("reserved transpose effect verifies");
        }
    }

    assert!(
        AllocationEffectOp::new(
            context,
            AccessKindAttr::Read,
            MemorySpaceAttr::Workgroup,
            17,
            23,
        )
        .is_err()
    );
    assert!(
        AllocationEffectOp::new(
            context,
            AccessKindAttr::AtomicRead,
            MemorySpaceAttr::Global,
            17,
            23,
        )
        .is_err()
    );
    assert!(
        AllocationEffectOp::new(
            context,
            AccessKindAttr::Read,
            MemorySpaceAttr::Global,
            0,
            23,
        )
        .is_err()
    );
}

#[test]
fn parsed_allocation_effect_payloads_fail_closed_before_analysis() {
    let in_function = |name: &str, operation: &str| {
        format!(
            "builtin.func @{name}: builtin.function <() -> ()>\n{{\n  ^entry_block1v1():\n    {operation};\n    kernel.return () [] []: <() -> ()>\n}}"
        )
    };
    let cases = [
        "kernel_allocation_effect_access_kind: kernel.access_kind Write, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_origin: kernel.allocation_origin 17, kernel_allocation_effect_noalias_class: kernel.noalias_class 23",
        "kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Workgroup, kernel_allocation_effect_origin: kernel.allocation_origin 17, kernel_allocation_effect_noalias_class: kernel.noalias_class 23",
        "kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_origin: kernel.allocation_origin 0, kernel_allocation_effect_noalias_class: kernel.noalias_class 23",
        "kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_noalias_class: kernel.noalias_class 23",
        "kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_origin: kernel.noalias_class 17, kernel_allocation_effect_noalias_class: kernel.noalias_class 23",
        "kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_origin: kernel.allocation_origin 17, kernel_allocation_effect_noalias_class: kernel.noalias_class 23, kernel_index_value: kernel.index_value 0",
    ];
    for attributes in cases {
        let context = &mut Context::new();
        register_dialect(context, &kernel_name()).unwrap();
        let source = in_function(
            "payload",
            &format!("kernel.allocation_effect () [] [{attributes}]: <() -> ()>"),
        );
        let operation = parse_from_str(Operation::top_level_parser(), context, &source)
            .expect("malformed payload remains syntactically parseable");
        assert!(
            verify_operation(operation, context).is_err(),
            "malformed parsed effect verified: {source}"
        );
    }

    let attributes = "kernel_allocation_effect_access_kind: kernel.access_kind Read, kernel_allocation_effect_memory_space: kernel.memory_space Global, kernel_allocation_effect_origin: kernel.allocation_origin 17, kernel_allocation_effect_noalias_class: kernel.noalias_class 23";
    let structured_cases = [
        in_function(
            "result",
            &format!("v0 = kernel.allocation_effect () [] [{attributes}]: <() -> (kernel.index )>"),
        ),
        format!(
            "builtin.func @operand: builtin.function <() -> ()>\n{{\n  ^entry_block1v1():\n    v0 = kernel.index_constant () [] [kernel_index_value: kernel.index_value 0]: <() -> (kernel.index )>;\n    kernel.allocation_effect (v0) [] [{attributes}]: <(kernel.index ) -> ()>;\n    kernel.return () [] []: <() -> ()>\n}}"
        ),
        format!(
            "builtin.func @successor: builtin.function <() -> ()>\n{{\n  ^entry_block1v1():\n    kernel.allocation_effect () [^exit_block1v1] [{attributes}]: <() -> ()>;\n    kernel.return () [] []: <() -> ()>\n  ^exit_block1v1():\n    kernel.return () [] []: <() -> ()>\n}}"
        ),
        in_function(
            "region",
            &format!(
                "kernel.allocation_effect () [] [{attributes}]: <() -> ()>\n    {{\n      ^nested_block1v1():\n        kernel.return () [] []: <() -> ()>\n    }}"
            ),
        ),
    ];
    for source in structured_cases {
        let context = &mut Context::new();
        register_dialect(context, &kernel_name()).unwrap();
        let operation = parse_from_str(Operation::top_level_parser(), context, &source)
            .expect("invalid structural payload remains syntactically parseable");
        assert!(
            verify_operation(operation, context).is_err(),
            "structurally malformed parsed effect verified: {source}"
        );
    }
}

#[test]
fn ranked_memory_local_verifiers_reject_foreign_indices_rank_mismatch_and_writes() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let extent = IndexConstantOp::new(context, 32);
    verify_op(&extent, context).unwrap();
    let view_type = RankedViewType::new(context, 32, false, vec![DYNAMIC_EXTENT, 64]).unwrap();
    let view = RankedViewOp::new(context, view_type, vec![extent.result(context)]).unwrap();
    verify_op(&view, context).unwrap();

    let index = IndexConstantOp::new(context, 4);
    let column = IndexConstantOp::new(context, 7);
    let read = RankedAccessOp::new(
        context,
        AccessKindAttr::Read,
        view.result(context),
        vec![index.result(context), column.result(context)],
    )
    .unwrap();
    verify_op(&read, context).unwrap();
    assert_eq!(
        RankedAccessOp::new(
            context,
            AccessKindAttr::AtomicRead,
            view.result(context),
            vec![index.result(context), column.result(context)],
        )
        .err()
        .unwrap(),
        RankedMemoryError::MissingAtomicContract,
    );
    let atomic_read = RankedAccessOp::new_atomic(
        context,
        AccessKindAttr::AtomicRead,
        AtomicOrderingAttr::Acquire,
        AtomicScopeAttr::Device,
        view.result(context),
        vec![index.result(context), column.result(context)],
    )
    .unwrap();
    verify_op(&atomic_read, context).unwrap();
    assert_eq!(
        RankedAccessOp::new(
            context,
            AccessKindAttr::Read,
            view.result(context),
            vec![index.result(context)],
        )
        .err()
        .unwrap(),
        RankedMemoryError::OperandCountMismatch {
            expected: 2,
            actual: 1,
        },
    );
    assert_eq!(
        RankedAccessOp::new(
            context,
            AccessKindAttr::Write,
            view.result(context),
            vec![index.result(context), column.result(context)],
        )
        .err()
        .unwrap(),
        RankedMemoryError::WriteThroughReadOnlyView,
    );

    let foreign = AlgorithmOp::new(context, 1).unwrap();
    let foreign_result = foreign.get_operation().deref(context).get_result(0);
    let raw = Operation::new(
        context,
        RankedAccessOp::get_concrete_op_info(),
        vec![],
        vec![view.result(context), foreign_result, column.result(context)],
        vec![],
        0,
    );
    let hostile = RankedAccessOp::from_operation(raw);
    hostile.set_attr_kernel_access_kind(context, AccessKindAttr::Read);
    assert!(verify_op(&hostile, context).is_err());
}

#[test]
fn ownership_contract_local_verifier_rejects_forged_view_and_payload_claims() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let writable_type = RankedViewType::new(context, 32, true, vec![8]).unwrap();
    let global =
        RankedViewOp::new_in_space(context, writable_type, vec![], MemorySpaceAttr::Global)
            .unwrap();
    let valid = OwnershipContractOp::new(
        context,
        global.result(context),
        OwnershipCoverageAttr::ExactView,
        OwnershipPartitionAttr::DenseRectangles,
    )
    .unwrap();
    verify_op(&valid, context).unwrap();

    let readonly_type = RankedViewType::new(context, 32, false, vec![8]).unwrap();
    let readonly = RankedViewOp::new(context, readonly_type, vec![]).unwrap();
    assert!(matches!(
        OwnershipContractOp::new(
            context,
            readonly.result(context),
            OwnershipCoverageAttr::ExactView,
            OwnershipPartitionAttr::ExactSets,
        ),
        Err(RankedMemoryError::WriteThroughReadOnlyView)
    ));
    let workgroup =
        RankedViewOp::new_in_space(context, writable_type, vec![], MemorySpaceAttr::Workgroup)
            .unwrap();
    assert!(
        OwnershipContractOp::new(
            context,
            workgroup.result(context),
            OwnershipCoverageAttr::ExactView,
            OwnershipPartitionAttr::ExactSets,
        )
        .is_err()
    );

    let attributes = "kernel_ownership_coverage: kernel.ownership_coverage ExactView, kernel_ownership_partition: kernel.ownership_partition ExactSets";
    let source = |view_type: &str, view_space: &str, contract_attributes: &str| {
        format!(
            "builtin.func @ownership_payload: builtin.function <() -> ()>\n{{\n  ^entry_block1v1():\n    v0 = kernel.ranked_view () [] [kernel_memory_space: kernel.memory_space {view_space}]: <() -> ({view_type})>;\n    kernel.ownership_contract (v0) [] [{contract_attributes}]: <({view_type}) -> ()>;\n    kernel.return () [] []: <() -> ()>\n}}"
        )
    };
    let valid_source = source("kernel.ranked_view <32,true,[8]>", "Global", attributes);
    let parsed = parse_from_str(Operation::top_level_parser(), context, &valid_source).unwrap();
    verify_operation(parsed, context).unwrap();

    for hostile in [
        source("kernel.ranked_view <32,false,[8]>", "Global", attributes),
        source("kernel.ranked_view <32,true,[8]>", "Workgroup", attributes),
        source(
            "kernel.ranked_view <32,true,[8]>",
            "Global",
            "kernel_ownership_coverage: kernel.ownership_coverage ExactView",
        ),
        source(
            "kernel.ranked_view <32,true,[8]>",
            "Global",
            &format!("{attributes}, kernel_index_value: kernel.index_value 0"),
        ),
    ] {
        let operation = parse_from_str(Operation::top_level_parser(), context, &hostile).unwrap();
        assert!(
            verify_operation(operation, context).is_err(),
            "forged ownership payload verified: {hostile}"
        );
    }
}

#[test]
fn dimension_verifier_binds_selector_to_the_same_view_rank() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let view_type = RankedViewType::new(context, 16, true, vec![8, 16]).unwrap();
    let view = RankedViewOp::new(context, view_type, vec![]).unwrap();
    let dimension = DimensionOp::new(context, view.result(context), 1).unwrap();
    verify_op(&dimension, context).unwrap();
    assert_eq!(
        DimensionOp::new(context, view.result(context), 2)
            .err()
            .unwrap(),
        RankedMemoryError::DimensionOutOfBounds {
            dimension: 2,
            rank: 2,
        },
    );

    dimension.set_attr_kernel_dimension(context, DimensionAttr(2));
    assert!(verify_op(&dimension, context).is_err());
    assert!(
        dimension
            .result(context)
            .get_type(context)
            .deref(context)
            .is::<IndexType>()
    );
}

#[test]
fn checked_tiled_index_verifier_rejects_malformed_geometry_and_payload() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let values = (0..5)
        .map(|value| IndexConstantOp::new(context, value))
        .collect::<Vec<_>>();
    let valid = CheckedTiledIndex2DOp::new(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        [64, 16, 16, 4],
    );
    verify_op(&valid, context).unwrap();

    valid.set_attr_kernel_tile_rows(context, IndexValueAttr(15));
    assert!(verify_op(&valid, context).is_err());

    let raw = Operation::new(
        context,
        CheckedTiledIndex2DOp::get_concrete_op_info(),
        vec![IndexType::get(context).into()],
        values[..4]
            .iter()
            .map(|value| value.result(context))
            .collect(),
        vec![],
        0,
    );
    let missing_operand = CheckedTiledIndex2DOp::from_operation(raw);
    missing_operand.set_attr_kernel_lanes_per_tile(context, IndexValueAttr(64));
    missing_operand.set_attr_kernel_tile_rows(context, IndexValueAttr(16));
    missing_operand.set_attr_kernel_tile_columns(context, IndexValueAttr(16));
    missing_operand.set_attr_kernel_elements_per_lane(context, IndexValueAttr(4));
    assert!(verify_op(&missing_operand, context).is_err());
}

#[test]
fn predicated_checked_access_binds_index_success_and_physical_extent() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let values = (0..6)
        .map(|value| IndexConstantOp::new(context, value + 1))
        .collect::<Vec<_>>();
    let extent = values[5].result(context);
    let view_type = RankedViewType::new(context, 32, true, vec![DYNAMIC_EXTENT]).unwrap();
    let view = RankedViewOp::new(context, view_type, vec![extent]).unwrap();
    let tiled = CheckedTiledIndex2DOp::new_predicated(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        extent,
        [64, 16, 16, 4],
    );
    verify_op(&tiled, context).unwrap();
    let access = RankedAccessOp::new_predicated(
        context,
        AccessKindAttr::Write,
        view.result(context),
        tiled.result(context),
        tiled.success(context).unwrap(),
    )
    .unwrap();
    verify_op(&access, context).unwrap();
    assert_eq!(access.indices(context), [tiled.result(context)]);
    assert_eq!(access.checked_success(context), tiled.success(context));

    let other = CheckedRowStripedIndex2DOp::new_predicated(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        extent,
        [64, 4],
    );
    verify_op(&other, context).unwrap();
    assert!(
        RankedAccessOp::new_predicated(
            context,
            AccessKindAttr::Write,
            view.result(context),
            tiled.result(context),
            other.success(context).unwrap(),
        )
        .is_err()
    );

    let different_extent = IndexConstantOp::new(context, 99);
    let different_view_type = RankedViewType::new(context, 32, true, vec![DYNAMIC_EXTENT]).unwrap();
    let different_view = RankedViewOp::new(
        context,
        different_view_type,
        vec![different_extent.result(context)],
    )
    .unwrap();
    assert!(
        RankedAccessOp::new_predicated(
            context,
            AccessKindAttr::Write,
            different_view.result(context),
            tiled.result(context),
            tiled.success(context).unwrap(),
        )
        .is_err()
    );

    let read_only_type = RankedViewType::new(context, 32, false, vec![DYNAMIC_EXTENT]).unwrap();
    let read_only = RankedViewOp::new(context, read_only_type, vec![extent]).unwrap();
    assert!(
        RankedAccessOp::new_predicated(
            context,
            AccessKindAttr::Write,
            read_only.result(context),
            tiled.result(context),
            tiled.success(context).unwrap(),
        )
        .is_err()
    );
    let read = RankedAccessOp::new_predicated(
        context,
        AccessKindAttr::Read,
        read_only.result(context),
        tiled.result(context),
        tiled.success(context).unwrap(),
    )
    .unwrap();
    verify_op(&read, context).unwrap();

    let rank_two_type =
        RankedViewType::new(context, 32, true, vec![DYNAMIC_EXTENT, DYNAMIC_EXTENT]).unwrap();
    let rank_two = RankedViewOp::new(context, rank_two_type, vec![extent, extent]).unwrap();
    assert!(
        RankedAccessOp::new_predicated(
            context,
            AccessKindAttr::Write,
            rank_two.result(context),
            tiled.result(context),
            tiled.success(context).unwrap(),
        )
        .is_err()
    );

    let wrong_kind = RankedAccessOp::new_predicated(
        context,
        AccessKindAttr::Write,
        view.result(context),
        tiled.result(context),
        tiled.success(context).unwrap(),
    )
    .unwrap();
    wrong_kind.set_attr_kernel_access_kind(context, AccessKindAttr::AtomicRead);
    assert!(verify_op(&wrong_kind, context).is_err());

    let malformed_count = CheckedTiledIndex2DOp::new_predicated(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        extent,
        [64, 16, 16, 4],
    );
    Operation::pop_result(malformed_count.get_operation(), context);
    assert!(verify_op(&malformed_count, context).is_err());

    let wrong_success_type = CheckedTiledIndex2DOp::new_predicated(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        extent,
        [64, 16, 16, 4],
    );
    wrong_success_type
        .success(context)
        .unwrap()
        .set_type(context, IndexType::get(context).into());
    assert!(verify_op(&wrong_success_type, context).is_err());
    assert!(
        RankedAccessOp::new_predicated(
            context,
            AccessKindAttr::Write,
            view.result(context),
            wrong_success_type.result(context),
            wrong_success_type.success(context).unwrap(),
        )
        .is_err()
    );

    // Missing or repeated uses remain structurally valid. This dialect shape
    // carries an obligation only; production source correspondence is what
    // may later prove an exact use relation.
    let unused = CheckedRowStripedIndex2DOp::new_predicated(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        extent,
        [64, 4],
    );
    verify_op(&unused, context).unwrap();
    let reused_first = RankedAccessOp::new_predicated(
        context,
        AccessKindAttr::Read,
        view.result(context),
        unused.result(context),
        unused.success(context).unwrap(),
    )
    .unwrap();
    let reused_second = RankedAccessOp::new_predicated(
        context,
        AccessKindAttr::Write,
        view.result(context),
        unused.result(context),
        unused.success(context).unwrap(),
    )
    .unwrap();
    verify_op(&reused_first, context).unwrap();
    verify_op(&reused_second, context).unwrap();
}

#[test]
fn checked_row_striped_index_verifier_rejects_zero_overflow_and_missing_operands() {
    let context = &mut Context::new();
    register_dialect(context, &kernel_name()).unwrap();
    let values = (0..5)
        .map(|value| IndexConstantOp::new(context, value))
        .collect::<Vec<_>>();
    let valid = CheckedRowStripedIndex2DOp::new(
        context,
        values[0].result(context),
        values[1].result(context),
        values[2].result(context),
        values[3].result(context),
        values[4].result(context),
        [64, 64],
    );
    verify_op(&valid, context).unwrap();
    valid.set_attr_kernel_lanes_per_row(context, IndexValueAttr(0));
    assert!(verify_op(&valid, context).is_err());
    valid.set_attr_kernel_lanes_per_row(context, IndexValueAttr(u64::MAX));
    valid.set_attr_kernel_row_striped_elements_per_lane(context, IndexValueAttr(2));
    assert!(verify_op(&valid, context).is_err());

    let raw = Operation::new(
        context,
        CheckedRowStripedIndex2DOp::get_concrete_op_info(),
        vec![IndexType::get(context).into()],
        values[..4]
            .iter()
            .map(|value| value.result(context))
            .collect(),
        vec![],
        0,
    );
    let missing = CheckedRowStripedIndex2DOp::from_operation(raw);
    missing.set_attr_kernel_lanes_per_row(context, IndexValueAttr(64));
    missing.set_attr_kernel_row_striped_elements_per_lane(context, IndexValueAttr(64));
    assert!(verify_op(&missing, context).is_err());
}

#[test]
fn op_interface_reports_only_target_neutral_kernel_ownership() {
    let context = &mut Context::new();
    let algorithm = AlgorithmOp::new(context, 3).expect("bounded algorithm");
    verify_op(&algorithm, context).expect("valid algorithm");

    let interface = op_cast::<dyn StructuredAlgorithmOp>(&algorithm).expect("interface present");
    assert_eq!(interface.semantic_owner(), SemanticOwner::Kernel);
    assert!(interface.is_target_neutral());
}

#[test]
fn hostile_operation_shapes_and_metadata_fail_verification() {
    let context = &mut Context::new();

    let wrong_type = UnitType::get(context);
    let raw = Operation::new(
        context,
        AlgorithmOp::get_concrete_op_info(),
        vec![wrong_type.into()],
        vec![],
        vec![],
        0,
    );
    let wrong_type_op = AlgorithmOp::from_operation(raw);
    wrong_type_op.set_iteration_domain(
        context,
        IterationDomainAttr::new(2).expect("bounded domain"),
    );
    assert!(verify_op(&wrong_type_op, context).is_err());

    let bounded_type = AlgorithmType::new(context, 2).expect("bounded type").into();
    let missing = Operation::new(
        context,
        AlgorithmOp::get_concrete_op_info(),
        vec![bounded_type],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&AlgorithmOp::from_operation(missing), context).is_err());

    let mismatched = AlgorithmOp::new(context, 2).expect("bounded algorithm");
    mismatched
        .get_operation()
        .deref_mut(context)
        .attributes
        .0
        .insert(
            ITERATION_DOMAIN_ATTR_KEY.try_into().expect("valid key"),
            Box::new(IterationDomainAttr::new(3).expect("bounded domain")),
        );
    assert!(verify_op(&mismatched, context).is_err());

    let extra = AlgorithmOp::new(context, 2).expect("bounded algorithm");
    extra
        .get_operation()
        .deref_mut(context)
        .attributes
        .0
        .insert(
            "kernel_hostile_extra".try_into().expect("valid key"),
            Box::new(BytesAttr::new(vec![0xde, 0xad, 0xbe, 0xef])),
        );
    assert!(verify_op(&extra, context).is_err());
}

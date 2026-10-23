use super::*;
use fe2o3_mir_model::semantic_mir_v1::{SemanticSourceFileIdentityV1, SemanticSourceOriginV1};

const UNIT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(0);
const ELEMENT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(1);
const ARRAY: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(2);
const INDEX: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(3);
const BLOCK: SemanticBlockIdV1 = SemanticBlockIdV1::from_index(0);
const FUNCTION: SemanticFunctionIdV1 = SemanticFunctionIdV1::from_index(7);

fn origin(tag: u8, line: u32, column: u32) -> SemanticSourceOriginV1 {
    SemanticSourceOriginV1::new(
        SemanticSourceFileIdentityV1::from_sha256([tag; 32]),
        10,
        11,
        line,
        column,
        line,
        column + 1,
    )
    .unwrap()
}

struct Fixture {
    types: Vec<SemanticTypeDeclV1>,
    function: SemanticFunctionDeclV1,
    statement_source: SemanticSourceProvenanceV1,
    terminator_source: SemanticSourceProvenanceV1,
}

impl Fixture {
    fn new() -> Self {
        let statement_source =
            SemanticSourceProvenanceV1::new(Some(origin(0xaa, 3, 4)), Some(origin(0xbb, 17, 20)));
        let terminator_source = SemanticSourceProvenanceV1::new(Some(origin(0xcc, 23, 8)), None);
        let source = SemanticSourceProvenanceV1::unavailable();
        let array = SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([123; 32]),
            SemanticLayoutIdentityV1::from_sha256([124; 32]),
            SemanticTypeLayoutV1::with_exact_rustc_layout(
                16,
                4,
                SemanticFieldsShapeV1::Array {
                    stride_bytes: 4,
                    count: 4,
                },
                SemanticRustcVariantsV1::Single { index: 0 },
                SemanticBackendReprV1::memory(true),
                None,
                false,
                None,
                4,
                0,
                SemanticTypeLayoutDetailsV1::None,
            )
            .unwrap(),
            SemanticTypeShapeV1::Array {
                element: ELEMENT,
                length: 4,
            },
        );
        let types = vec![
            unit_type(),
            integer_type(121, false, 32),
            array,
            integer_type(125, false, 64),
        ];
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([127; 32]),
            SemanticLayoutIdentityV1::from_sha256([128; 32]),
            SemanticCanonAbiV1::Rust,
            SemanticExternAbiV1::Rust,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(UNIT, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let local = |tag, ty, role| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([tag; 32]),
                ty,
                role,
                source,
            )
        };
        let block = SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([132; 32]),
            source,
            vec![SemanticStatementV1::new(
                statement_source,
                SemanticStatementKindV1::Nop,
            )],
            SemanticTerminatorV1::new(terminator_source, SemanticTerminatorKindV1::Return),
        )
        .unwrap();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([133; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([134; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([135; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([136; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([137; 32]),
            source,
            abi,
            vec![
                local(129, UNIT, SemanticLocalRoleV1::Return),
                local(130, ARRAY, SemanticLocalRoleV1::Temporary),
                local(131, INDEX, SemanticLocalRoleV1::Temporary),
            ],
            BLOCK,
            vec![block],
        )
        .unwrap();
        Self {
            types,
            function,
            statement_source,
            terminator_source,
        }
    }

    fn lowering<'a, 'service, 'work>(
        &'a self,
        budget: &'service mut ArgumentBudgetV1<'work>,
    ) -> SemanticFunctionLoweringV1<'a, 'service> {
        let plan = plan_semantic_function_ssa_with_module_v1(
            FUNCTION,
            &self.function,
            &self.types,
            &[],
            ProductionSemanticSsaLimitsV1::default(),
        )
        .unwrap();
        SemanticFunctionLoweringV1::new_interprocedural(
            &self.types,
            &[],
            &self.function,
            &plan,
            FUNCTION,
            FUNCTION,
            BTreeMap::new(),
            BTreeMap::new(),
            vec![],
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
                local_bindings: None,
            },
            None,
            None,
            BTreeSet::new().into(),
            1,
            false,
            64,
            PrivateArrayRecorderWorkV1::Owned(PrivateArrayLazyBudgetV1::new(1, 64)),
            None,
            CallReturnBufferV1::for_function(&self.function, &[], &BTreeMap::new(), 0, &mut *budget)
                .unwrap(),
            Some(budget),
            SemanticEmissionPlacementV1::default(),
            None,
            None,
            None,
)
        .unwrap()
    }
}

fn place(kind: SemanticProjectionKindV1) -> SemanticPlaceV1 {
    SemanticPlaceV1::new(
        SemanticLocalIdV1::from_index(1),
        vec![SemanticProjectionV1::new(kind, ELEMENT).unwrap()],
        ELEMENT,
    )
    .unwrap()
}

fn fill_array(
    lowering: &mut SemanticFunctionLoweringV1<'_, '_>,
    operations: &mut Vec<Operation>,
) -> Vec<ValueId> {
    let fields = (0..4)
        .map(|index| {
            lowering
                .emit(
                    operations,
                    Type::Scalar(ScalarType::U32),
                    OperationKind::Constant(Constant::U32(100 + index)),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let values = fields
        .iter()
        .map(|field| field.value().unwrap().0)
        .collect();
    lowering.locals[1] = Some(SemanticValueBindingV1::Aggregate(fields));
    values
}

fn assert_oob(
    error: ProductionSemanticKirErrorV1,
    block: u32,
    statement: Option<u32>,
    index: u128,
    length: u64,
    from_end: bool,
    expected_source: SemanticSourceProvenanceV1,
) -> String {
    let rendered = error.to_string();
    let ProductionSemanticKirErrorV1::FixedArrayIndexOutOfBounds {
        function,
        block: actual_block,
        statement: actual_statement,
        local,
        index: actual_index,
        length: actual_length,
        from_end: actual_from_end,
        source,
    } = error
    else {
        panic!("expected a structured fixed-array bounds error");
    };
    assert_eq!(
        (
            function,
            actual_block,
            actual_statement,
            local,
            actual_index,
            actual_length,
            actual_from_end
        ),
        (7, block, statement, 1, index, length, from_end),
    );
    assert_eq!(*source, expected_source);
    assert!(rendered.contains("error[FE2O3-BOUNDS-001]"));
    assert!(rendered.contains("lowering stopped before target IR or artifact emission"));
    rendered
}

#[test]
fn exact_bounds_helper_preserves_wide_indices_and_from_end_offsets() {
    let fixture = Fixture::new();
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
    let lowering = fixture.lowering(&mut budget);
    let place = place(SemanticProjectionKindV1::Index(
        SemanticLocalIdV1::from_index(2),
    ));
    for (index, from_end, expected) in [(0, false, 0), (3, false, 3), (1, true, 3), (4, true, 0)] {
        assert_eq!(
            lowering
                .checked_by_value_array_index(BLOCK, Some(0), &place, index, 4, from_end)
                .unwrap(),
            expected,
        );
    }
    for (index, from_end) in [
        (4, false),
        (u128::MAX, false),
        (0, true),
        (5, true),
        (u128::MAX, true),
    ] {
        let error = lowering
            .checked_by_value_array_index(BLOCK, Some(0), &place, index, 4, from_end)
            .unwrap_err();
        let rendered = assert_oob(
            error,
            0,
            Some(0),
            index,
            4,
            from_end,
            fixture.statement_source,
        );
        assert!(rendered.contains("Rust source bbbbbbbbbbbb:17:20"));
        assert!(!rendered.contains("Rust source aaaaaaaaaaaa:3:4"));
        let witness = if from_end {
            format!("required: 0 < {index} <= 4 (from-end offset)")
        } else {
            format!("required: {index} < 4")
        };
        assert!(rendered.contains(&witness));
    }
    for from_end in [false, true] {
        assert_oob(
            lowering
                .checked_by_value_array_index(BLOCK, Some(0), &place, 0, 0, from_end)
                .unwrap_err(),
            0,
            Some(0),
            0,
            0,
            from_end,
            fixture.statement_source,
        );
    }
}

#[test]
fn bounds_error_uses_terminator_or_explicitly_unavailable_provenance() {
    let fixture = Fixture::new();
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
    let lowering = fixture.lowering(&mut budget);
    let place = place(SemanticProjectionKindV1::Index(
        SemanticLocalIdV1::from_index(2),
    ));
    for (block, statement, source) in [
        (0, None, fixture.terminator_source),
        (0, Some(99), SemanticSourceProvenanceV1::unavailable()),
        (99, None, SemanticSourceProvenanceV1::unavailable()),
    ] {
        let error = lowering
            .checked_by_value_array_index(
                SemanticBlockIdV1::from_index(block),
                statement,
                &place,
                4,
                4,
                false,
            )
            .unwrap_err();
        let rendered = assert_oob(error, block, statement, 4, 4, false, source);
        if source == fixture.terminator_source {
            assert!(rendered.contains("Rust source cccccccccccc:23:8"));
        } else {
            assert!(rendered.contains("Rust source location unavailable"));
        }
    }
}

#[test]
fn helper_effect_diagnostic_labels_declaration_provenance_without_inventing_a_caller() {
    let fixture = Fixture::new();
    for (source, location) in [
        (fixture.statement_source, "Rust source bbbbbbbbbbbb:17:20"),
        (fixture.terminator_source, "Rust source cccccccccccc:23:8"),
        (
            SemanticSourceProvenanceV1::unavailable(),
            "Rust source location unavailable",
        ),
    ] {
        let error = ProductionSemanticKirErrorV1::HelperEffectsUnavailable {
            function: 7,
            declaration_source: Box::new(source),
        };
        let rendered = error.to_string();
        assert!(rendered.contains("rejected function 7"));
        assert!(rendered.contains("not interprocedurally complete and pure"));
        assert!(rendered.contains(&format!("helper declaration at {location}")));
        assert!(rendered.contains("lowering stopped before target IR or artifact emission"));
        assert!(!rendered.contains("caller"));
        assert!(std::error::Error::source(&error).is_none());
    }
}

#[test]
fn indexed_place_uses_actual_emitted_constants_without_host_truncation() {
    let fixture = Fixture::new();
    let place = place(SemanticProjectionKindV1::Index(
        SemanticLocalIdV1::from_index(2),
    ));
    for index in [0, 3, 4, u64::MAX] {
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(&mut budget);
        let mut operations = vec![];
        let values = fill_array(&mut lowering, &mut operations);
        let binding = lowering
            .emit(
                &mut operations,
                Type::Scalar(ScalarType::U64),
                OperationKind::Constant(Constant::U64(index)),
            )
            .unwrap();
        lowering.locals[2] = Some(binding);
        let before = (
            operations.len(),
            lowering.emitted_operations,
            lowering.next_value,
        );
        let result = lowering.lower_indexed_place_address(BLOCK, Some(0), &place, &mut operations);
        if index < 4 {
            assert_eq!(
                result.unwrap().value().unwrap(),
                (values[index as usize], Type::Scalar(ScalarType::U32)),
            );
        } else {
            assert_oob(
                result.unwrap_err(),
                0,
                Some(0),
                u128::from(index),
                4,
                false,
                fixture.statement_source,
            );
        }
        assert_eq!(
            (
                operations.len(),
                lowering.emitted_operations,
                lowering.next_value
            ),
            before
        );
    }
}

#[test]
fn constant_index_place_resolves_forward_and_from_end_boundaries() {
    let fixture = Fixture::new();
    assert!(matches!(
        SemanticProjectionV1::new(
            SemanticProjectionKindV1::ConstantIndex {
                offset: 0,
                minimum_length: 4,
                from_end: true,
            },
            ELEMENT,
        ),
        Err(fe2o3_mir_model::semantic_mir_v1::SemanticMirErrorV1::InvalidProjectionShape)
    ));
    for (offset, minimum_length, from_end, expected) in [
        (3, 4, false, Some(3)),
        (1, 4, true, Some(3)),
        (3, 4, true, Some(1)),
        (5, 5, true, None),
    ] {
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(&mut budget);
        let mut operations = vec![];
        let values = fill_array(&mut lowering, &mut operations);
        let place = place(SemanticProjectionKindV1::ConstantIndex {
            offset,
            minimum_length,
            from_end,
        });
        let before = (
            operations.len(),
            lowering.emitted_operations,
            lowering.next_value,
        );
        let result = lowering.lower_indexed_place_address(BLOCK, None, &place, &mut operations);
        if let Some(expected) = expected {
            assert_eq!(
                result.unwrap().value().unwrap(),
                (values[expected], Type::Scalar(ScalarType::U32)),
            );
        } else {
            assert_oob(
                result.unwrap_err(),
                0,
                None,
                u128::from(offset),
                4,
                true,
                fixture.terminator_source,
            );
        }
        assert_eq!(
            (
                operations.len(),
                lowering.emitted_operations,
                lowering.next_value
            ),
            before
        );
    }
}

#[test]
fn malformed_array_binding_remains_a_structural_rejection() {
    let fixture = Fixture::new();
    for kind in [
        SemanticProjectionKindV1::Index(SemanticLocalIdV1::from_index(2)),
        SemanticProjectionKindV1::ConstantIndex {
            offset: 0,
            minimum_length: 4,
            from_end: false,
        },
    ] {
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(&mut budget);
        let mut operations = vec![];
        fill_array(&mut lowering, &mut operations);
        let Some(SemanticValueBindingV1::Aggregate(fields)) = &mut lowering.locals[1] else {
            unreachable!();
        };
        fields.pop();
        let error = lowering
            .lower_indexed_place_address(BLOCK, Some(0), &place(kind), &mut operations)
            .unwrap_err();
        assert!(matches!(
            error,
            ProductionSemanticKirErrorV1::Unsupported {
                detail: "indexed array binding differs from its semantic length",
                ..
            }
        ));
    }
}

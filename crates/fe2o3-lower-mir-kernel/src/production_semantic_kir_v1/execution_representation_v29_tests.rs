use super::*;
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticDirectEnumEncodingV1, SemanticEnumLayoutV1, SemanticEnumVariantLayoutV1,
    SemanticExecutionRoleV29, SemanticRustTypeKindV1,
};

const UNIT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(0);
const U64: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(1);
const U8: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(2);
const MARKER: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(3);
const EPOCH: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(4);
const CONTEXT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(5);
const PLAIN_CONTEXT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(6);
const WORKGROUP: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(7);
const PLAIN_WORKGROUP: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(8);
const REFUSAL: &str =
    "execution roles require occurrence-bound transport, not an ordinary Rust representation";

fn aggregate(
    tag: u8,
    fields: Vec<SemanticTypeIdV1>,
    offsets: Vec<u64>,
    size: u64,
    alignment: u64,
    tuple: bool,
) -> SemanticTypeDeclV1 {
    let fields = SemanticAggregateTypeV1::new(fields).unwrap();
    SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([tag; 32]),
        SemanticLayoutIdentityV1::from_sha256([tag; 32]),
        SemanticTypeLayoutV1::aggregate(
            Some(size),
            alignment,
            SemanticAggregateLayoutV1::new(offsets, vec![]).unwrap(),
        )
        .unwrap(),
        if tuple {
            SemanticTypeShapeV1::Tuple(fields)
        } else {
            SemanticTypeShapeV1::Aggregate(fields)
        },
    )
}

fn types() -> Vec<SemanticTypeDeclV1> {
    let context = aggregate(50, vec![MARKER; 5], vec![0; 5], 0, 1, false);
    let workgroup = aggregate(
        52,
        vec![U64, U64, EPOCH, MARKER],
        vec![0, 8, 16, 16],
        16,
        8,
        false,
    );
    vec![
        unit_type(),
        integer_type(41, false, 64),
        integer_type(43, false, 8),
        aggregate(46, vec![], vec![], 0, 1, false),
        aggregate(48, vec![MARKER; 3], vec![0; 3], 0, 1, false),
        context
            .clone()
            .with_rust_type_kind(SemanticRustTypeKindV1::Execution(
                SemanticExecutionRoleV29::KernelContext,
            )),
        context,
        workgroup
            .clone()
            .with_rust_type_kind(SemanticRustTypeKindV1::Execution(
                SemanticExecutionRoleV29::Workgroup,
            )),
        workgroup,
    ]
}

fn array(
    tag: u8,
    element: SemanticTypeIdV1,
    length: u64,
    stride: u64,
    alignment: u64,
) -> SemanticTypeDeclV1 {
    SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([tag; 32]),
        SemanticLayoutIdentityV1::from_sha256([tag; 32]),
        SemanticTypeLayoutV1::with_exact_rustc_layout(
            stride.checked_mul(length).unwrap(),
            alignment,
            SemanticFieldsShapeV1::array(stride, length),
            SemanticRustcVariantsV1::Single { index: 0 },
            SemanticBackendReprV1::memory(true),
            None,
            false,
            None,
            alignment,
            0,
            SemanticTypeLayoutDetailsV1::None,
        )
        .unwrap(),
        SemanticTypeShapeV1::Array { element, length },
    )
}

fn optional_context(tag: u8, payload: SemanticTypeIdV1) -> SemanticTypeDeclV1 {
    let variants = [vec![], vec![1]]
        .into_iter()
        .enumerate()
        .map(|(index, offsets)| {
            SemanticEnumVariantLayoutV1::from_rustc(
                index as u32,
                1,
                1,
                SemanticFieldsShapeV1::arbitrary(
                    offsets.clone(),
                    (0..offsets.len() as u32).collect(),
                )
                .unwrap(),
                SemanticBackendReprV1::memory(true),
                None,
                false,
                None,
                1,
                0,
                SemanticAggregateLayoutV1::new(offsets, vec![]).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let layout = SemanticEnumLayoutV1::new(
        variants,
        SemanticEnumEncodingV1::Direct(SemanticDirectEnumEncodingV1::new(
            0,
            0,
            SemanticBackendScalarV1::initialized(
                SemanticBackendPrimitiveV1::integer(false, 8, 1),
                SemanticScalarValidityRangeV1::new(0, 1),
            ),
        )),
    )
    .unwrap();
    SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([tag; 32]),
        SemanticLayoutIdentityV1::from_sha256([tag; 32]),
        SemanticTypeLayoutV1::enum_layout(1, 1, layout).unwrap(),
        SemanticTypeShapeV1::Enum {
            discriminant: U8,
            variants: vec![
                SemanticEnumVariantV1::new(0, SemanticAggregateTypeV1::new(vec![]).unwrap()),
                SemanticEnumVariantV1::new(1, SemanticAggregateTypeV1::new(vec![payload]).unwrap()),
            ]
            .into_boxed_slice(),
        },
    )
}

fn push(types: &mut Vec<SemanticTypeDeclV1>, declaration: SemanticTypeDeclV1) -> SemanticTypeIdV1 {
    let ty = SemanticTypeIdV1::from_index(types.len() as u32);
    types.push(declaration);
    ty
}

fn refused<T>(result: Result<T, ProductionSemanticKirErrorV1>) {
    match result {
        Err(ProductionSemanticKirErrorV1::Unsupported {
            detail: REFUSAL, ..
        }) => {}
        Err(error) => panic!("expected nominal execution-role refusal, got {error:?}"),
        Ok(_) => panic!("ordinary representation accepted a nominal execution role"),
    }
}

fn constant(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
    let source = SemanticSourceProvenanceV1::unavailable();
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([201; 32]),
        SemanticLayoutIdentityV1::from_sha256([202; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(UNIT, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([203; 32]),
        SemanticFunctionRoleV1::InternalHelper,
        SemanticItemDefinitionIdentityV1::from_sha256([204; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([205; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([206; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([207; 32]),
        source,
        abi,
        vec![SemanticLocalDeclV1::new(
            SemanticLocalIdentityV1::from_sha256([208; 32]),
            UNIT,
            SemanticLocalRoleV1::Return,
            source,
        )],
        SemanticBlockIdV1::from_index(0),
        vec![
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([209; 32]),
                source,
                vec![],
                SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
    let mut lowering = SemanticFunctionLoweringV1::new(
        types,
        &[],
        &function,
        SemanticParameterBindingsV1 {
            declarations: &[],
            values: &[],
            types: &[],
            local_bindings: None,
        },
        None,
        None,
        BTreeSet::new(),
        1,
        false,
        128,
        &mut lowering_budget,
    )?;
    let bytes = vec![0; types[ty.index() as usize].layout().size_bytes().unwrap() as usize];
    lowering.lower_constant_bytes(
        SemanticBlockIdV1::from_index(0),
        Some(0),
        ty,
        &bytes,
        0,
        &mut 0,
        &mut Vec::new(),
    )
}

fn check_pair(types: &[SemanticTypeDeclV1], nominal: SemanticTypeIdV1, ordinary: SemanticTypeIdV1) {
    assert_eq!(
        types[nominal.index() as usize].layout(),
        types[ordinary.index() as usize].layout()
    );
    let components = lower_ssa_value_components_v1(types, ordinary).unwrap();
    let values = components
        .iter()
        .enumerate()
        .map(|(index, (_, ty))| ValueDef::new(ValueId(index as u32), ty.clone()))
        .collect::<Vec<_>>();
    refused(lower_ssa_value_components_v1(types, nominal));
    if matches!(
        types[nominal.index() as usize].rust_type_kind(),
        SemanticRustTypeKindV1::Execution(_)
    ) {
        refused(lower_parameter_type(types, &[], nominal));
    }
    for validate_types in [true, false] {
        refused(binding_from_value_defs_with_validation(
            types,
            nominal,
            &values,
            validate_types,
        ));
        assert!(
            binding_from_value_defs_with_validation(types, ordinary, &values, validate_types)
                .is_ok()
        );
    }
    refused(binding_from_value_defs(types, nominal, &values));
    refused(constant(types, nominal));
    assert!(constant(types, ordinary).is_ok());
}

fn check_erased_pair(
    types: &[SemanticTypeDeclV1],
    nominal: SemanticTypeIdV1,
    ordinary: SemanticTypeIdV1,
) {
    assert_eq!(
        types[nominal.index() as usize].layout(),
        types[ordinary.index() as usize].layout()
    );
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(types.len());
    let mut budget = ArgumentBudgetV1::new(&mut work, 0);
    refused(require_execution_free_types_v29(types, &mut budget));

    // A separate inert ordinary-only table is the positive control. Raw helpers
    // may erase absent fields only behind this whole-owner invariant; a future
    // mixed-V29 materializer needs an explicit checked type view before activation.
    let ordinary_types = types
        .iter()
        .cloned()
        .map(|declaration| declaration.with_rust_type_kind(SemanticRustTypeKindV1::Ordinary))
        .collect::<Vec<_>>();
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(ordinary_types.len());
    let mut budget = ArgumentBudgetV1::new(&mut work, 0);
    require_execution_free_types_v29(&ordinary_types, &mut budget).unwrap();
    for ty in [nominal, ordinary] {
        let components = lower_ssa_value_components_v1(&ordinary_types, ty).unwrap();
        let values = components
            .into_iter()
            .enumerate()
            .map(|(index, (_, ty))| ValueDef::new(ValueId(index as u32), ty))
            .collect::<Vec<_>>();
        for validate_types in [true, false] {
            binding_from_value_defs_with_validation(&ordinary_types, ty, &values, validate_types)
                .unwrap();
        }
        constant(&ordinary_types, ty).unwrap();
    }
}

#[test]
fn nominal_execution_roles_do_not_use_their_physical_aggregate_layout() {
    let types = types();
    check_pair(&types, CONTEXT, PLAIN_CONTEXT);
    check_pair(&types, WORKGROUP, PLAIN_WORKGROUP);
    assert!(
        lower_ssa_value_components_v1(&types, PLAIN_CONTEXT)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        lower_ssa_value_components_v1(&types, PLAIN_WORKGROUP).unwrap(),
        vec![
            (U64, Type::Scalar(ScalarType::U64)),
            (U64, Type::Scalar(ScalarType::U64)),
        ]
    );
    assert_eq!(
        lower_parameter_type(&types, &[], U64).unwrap(),
        Type::Scalar(ScalarType::U64)
    );
}

#[test]
fn nested_tuple_and_nonempty_array_cannot_erase_nominal_execution_roles() {
    let mut types = types();
    let nominal_array = push(&mut types, array(60, WORKGROUP, 1, 16, 8));
    let ordinary_array = push(&mut types, array(61, PLAIN_WORKGROUP, 1, 16, 8));
    check_pair(&types, nominal_array, ordinary_array);
    let nominal = push(
        &mut types,
        aggregate(62, vec![CONTEXT, nominal_array], vec![0, 0], 16, 8, true),
    );
    let ordinary = push(
        &mut types,
        aggregate(
            63,
            vec![PLAIN_CONTEXT, ordinary_array],
            vec![0, 0],
            16,
            8,
            true,
        ),
    );
    check_pair(&types, nominal, ordinary);
}

#[test]
fn zero_length_arrays_still_require_an_ordinary_element_type() {
    let mut types = types();
    let nominal_array = push(&mut types, array(70, CONTEXT, 0, 0, 1));
    let ordinary_array = push(&mut types, array(71, PLAIN_CONTEXT, 0, 0, 1));
    check_erased_pair(&types, nominal_array, ordinary_array);
    let nominal = push(
        &mut types,
        aggregate(72, vec![nominal_array], vec![0], 0, 1, true),
    );
    let ordinary = push(
        &mut types,
        aggregate(73, vec![ordinary_array], vec![0], 0, 1, true),
    );
    check_erased_pair(&types, nominal, ordinary);
}

#[test]
fn enum_discriminant_and_empty_variant_do_not_hide_execution_payloads() {
    let mut types = types();
    let nominal = push(&mut types, optional_context(80, CONTEXT));
    let ordinary = push(&mut types, optional_context(81, PLAIN_CONTEXT));
    // The all-zero bytes select the payload-free variant. Its sibling still
    // prevents this type from entering ordinary enum/discriminant transport.
    check_erased_pair(&types, nominal, ordinary);
}

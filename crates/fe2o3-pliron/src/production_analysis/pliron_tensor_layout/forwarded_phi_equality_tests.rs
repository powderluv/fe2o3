use super::*;
use pliron::{
    builtin::{op_interfaces::OneRegionInterface, types::FunctionType},
    op::Op,
    r#type::TypeHandle,
};

struct Fixture {
    context: Context,
    entry: Ptr<BasicBlock>,
    values: Vec<Value>,
    zero: Value,
    one: Value,
}

fn fixture(count: usize) -> Fixture {
    let mut context = Context::new();
    dialect_kernel::register_dialect(
        &mut context,
        &pliron::dialect::DialectName::try_new(dialect_kernel::DIALECT_NAME).unwrap(),
    )
    .unwrap();
    let index: TypeHandle = dialect_kernel::IndexType::get(&context).into();
    let signature = FunctionType::get(&context, vec![index], vec![]);
    let function = FuncOp::new(
        &mut context,
        "forwarding_resources".try_into().unwrap(),
        signature,
    );
    let entry = function.get_entry_block(&context);
    let zero = IndexConstantOp::new(&mut context, 0);
    let one = IndexConstantOp::new(&mut context, 1);
    zero.get_operation().insert_at_back(entry, &context);
    one.get_operation().insert_at_back(entry, &context);
    let ret = ReturnOp::new(&mut context);
    ret.get_operation().insert_at_back(entry, &context);
    let values = (0..count)
        .map(|ordinal| {
            let block = BasicBlock::new(
                &mut context,
                Some(format!("phi_{ordinal}").try_into().unwrap()),
                vec![index],
            );
            block.insert_at_back(function.get_region(&context), &context);
            let ret = ReturnOp::new(&mut context);
            ret.get_operation().insert_at_back(block, &context);
            block.deref(&context).get_argument(0)
        })
        .collect();
    pliron::operation::verify_operation(function.get_operation(), &context).unwrap();
    Fixture {
        zero: zero.result(&context),
        one: one.result(&context),
        context,
        entry,
        values,
    }
}

fn index<'a>(fixture: &Fixture, inputs: &'a [Vec<Value>]) -> SubgroupForwardingIndexV1<'a> {
    assert_eq!(fixture.values.len(), inputs.len());
    let mut work = 0;
    let mut index = SubgroupForwardingIndexV1::new(fixture.entry, inputs.len(), &mut work).unwrap();
    for (&value, inputs) in fixture.values.iter().zip(inputs) {
        index.push(value, inputs, &mut work).unwrap();
    }
    index
}

fn memo_index(fixture: &Fixture, targets: &[Option<usize>]) -> SubgroupForwardingIndexV1<'static> {
    assert_eq!(fixture.values.len(), targets.len());
    let mut work = 0;
    let mut index =
        SubgroupForwardingIndexV1::new(fixture.entry, targets.len(), &mut work).unwrap();
    for (&value, target) in fixture.values.iter().zip(targets) {
        index.push(value, &[], &mut work).unwrap();
        index.rows.last_mut().unwrap().target = match target {
            Some(ordinal) => {
                assert!(*ordinal < targets.len());
                SubgroupForwardingTargetV1::Phi(*ordinal)
            }
            None => SubgroupForwardingTargetV1::Terminal(fixture.zero),
        };
    }
    index
}

fn assert_resolved_to(result: &ResolvedSubgroupForwardingV1<'_>, expected: Value) {
    assert!(result.0.path.is_empty());
    for row in &result.0.rows {
        assert!(
            matches!(row.state, SubgroupForwardingStateV1::Resolved(value) if value == expected)
        );
    }
}

#[test]
fn forwarded_identity_uses_only_phi_rows_and_exact_typed_inputs() {
    let fixture = fixture(4);
    let inputs = vec![
        vec![fixture.zero],
        vec![fixture.zero, fixture.zero],
        vec![fixture.values[0], fixture.values[1]],
        vec![fixture.zero, fixture.one],
    ];
    let result = index(&fixture, &inputs)
        .finish(&fixture.context, &mut 0)
        .unwrap();
    assert!(result.inputs_equal(2, &mut 0).unwrap());
    assert!(!result.inputs_equal(3, &mut 0).unwrap());
    assert_eq!(
        result.representative(fixture.values[0], &mut 0).unwrap(),
        fixture.zero
    );
    assert_eq!(
        result.representative(fixture.values[1], &mut 0).unwrap(),
        fixture.zero
    );
    // Transitive equal incoming values retain their exact typed identity.
    assert_eq!(
        result.representative(fixture.values[2], &mut 0).unwrap(),
        fixture.zero
    );
}

#[test]
fn forwarded_identity_never_treats_arithmetic_or_cast_dependencies_as_copies() {
    let mut fixture = fixture(2);
    let cast = IndexUnsignedCastOp::new(&mut fixture.context, fixture.zero, 32);
    let add = IndexBinaryOp::new(
        &mut fixture.context,
        dialect_kernel::IndexBinaryKindAttr::Add,
        fixture.zero,
        fixture.zero,
    );
    let cast_value = cast.result(&fixture.context);
    let add_value = add.result(&fixture.context);
    let inputs = vec![vec![cast_value], vec![add_value]];
    let result = index(&fixture, &inputs)
        .finish(&fixture.context, &mut 0)
        .unwrap();
    assert_eq!(
        result.representative(fixture.values[0], &mut 0).unwrap(),
        cast_value
    );
    assert_eq!(
        result.representative(fixture.values[1], &mut 0).unwrap(),
        add_value
    );
    assert_ne!(cast_value, fixture.zero);
    assert_ne!(add_value, fixture.zero);
    let mut builder = SubgroupForwardingIndexV1::new(fixture.entry, 1, &mut 0).unwrap();
    let source = [fixture.zero];
    assert!(matches!(
        builder.push(cast_value, &source, &mut 0),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { .. })
    ));
}

#[test]
fn forwarded_identity_rejects_entry_duplicates_missing_rows_types_and_foreign_owners() {
    let mut fixture = fixture(2);
    let entry_argument = fixture.entry.deref(&fixture.context).get_argument(0);
    let mut builder = SubgroupForwardingIndexV1::new(fixture.entry, 2, &mut 0).unwrap();
    assert!(builder.push(entry_argument, &[], &mut 0).is_err());
    builder.push(fixture.values[0], &[], &mut 0).unwrap();
    assert!(builder.push(fixture.values[0], &[], &mut 0).is_err());
    assert!(builder.finish(&fixture.context, &mut 0).is_err());

    let integer: TypeHandle = pliron::builtin::types::IntegerType::get(
        &fixture.context,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    )
    .into();
    let wrong = BasicBlock::new(&mut fixture.context, None, vec![integer]);
    let wrong = wrong.deref(&fixture.context).get_argument(0);
    let inputs = vec![vec![wrong], vec![]];
    assert!(matches!(
        index(&fixture, &inputs).finish(&fixture.context, &mut 0),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
            if detail == "forwarding identity has unequal edge types"
    ));
    let foreign = self::fixture(1);
    let inputs = vec![vec![foreign.zero], vec![]];
    assert!(matches!(
        index(&fixture, &inputs).finish(&fixture.context, &mut 0),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
            if detail == "forwarded value has a foreign or missing owner"
    ));
    let mut foreign_row = SubgroupForwardingIndexV1::new(fixture.entry, 1, &mut 0).unwrap();
    foreign_row.push(foreign.values[0], &[], &mut 0).unwrap();
    assert!(matches!(
        foreign_row.finish(&fixture.context, &mut 0),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
            if detail == "forwarded value has a foreign or missing owner"
    ));
    let index_type: TypeHandle = dialect_kernel::IndexType::get(&fixture.context).into();
    let stale_owner = BasicBlock::new(&mut fixture.context, None, vec![index_type]);
    let stale = stale_owner.deref(&fixture.context).get_argument(0);
    BasicBlock::remove_argument(stale_owner, &fixture.context, 0);
    let inputs = vec![vec![stale], vec![]];
    assert!(matches!(
        index(&fixture, &inputs).finish(&fixture.context, &mut 0),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
            if detail == "forwarded value is absent from its defining roster"
    ));
    let mut stale_row = SubgroupForwardingIndexV1::new(fixture.entry, 1, &mut 0).unwrap();
    stale_row.push(stale, &[], &mut 0).unwrap();
    assert!(matches!(
        stale_row.finish(&fixture.context, &mut 0),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
            if detail == "forwarded value is absent from its defining roster"
    ));
}

#[test]
fn forwarded_hash_collisions_preserve_exact_hits_duplicates_and_paid_misses() {
    let fixture = fixture(3);
    let mut index = SubgroupForwardingIndexV1::new(fixture.entry, 3, &mut 0).unwrap();
    index
        .push_hashed(fixture.values[0], &[], 7, &mut 0)
        .unwrap();
    index
        .push_hashed(fixture.values[1], &[], 7, &mut 0)
        .unwrap();
    assert_eq!(index.buckets.len(), 1);
    // Capacity three gives 8 * bit_length(4) = 24 ordered comparisons.
    // Each collision row pays one exact opaque-Value width plus two row steps.
    const VALUE: usize = std::mem::size_of::<Value>();
    let cases = [
        (fixture.values[1], 7, Some(1), 27 + VALUE),
        (fixture.values[0], 7, Some(0), 29 + 2 * VALUE),
        (fixture.values[2], 7, None, 29 + 2 * VALUE),
        (fixture.values[2], 8, None, 25),
    ];
    for (value, hash, expected, expected_work) in cases {
        let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - expected_work;
        assert_eq!(
            index.lookup_hashed(value, hash, &mut work).unwrap(),
            expected
        );
        assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
        let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - expected_work + 1;
        assert!(matches!(
            index.lookup_hashed(value, hash, &mut work),
            Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
        ));
        assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    }
    assert!(
        index
            .push_hashed(fixture.values[0], &[], 7, &mut 0)
            .is_err()
    );
    assert_eq!(index.rows.len(), 2);
    assert_eq!(index.buckets.len(), 1);
}

#[test]
fn forwarded_memo_exact_work_covers_chains_shared_paths_and_rejecting_prefixes() {
    for (targets, expected_work) in [
        (vec![None], 9),
        (vec![Some(1), None], 18),
        (vec![None, Some(0)], 21),
        (vec![Some(2), Some(2), Some(3), None], 39),
    ] {
        let fixture = fixture(targets.len());
        let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - expected_work;
        let result = memo_index(&fixture, &targets).resolve(&mut work).unwrap();
        assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
        assert_resolved_to(&result, fixture.zero);
        let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - expected_work + 1;
        assert!(matches!(
            memo_index(&fixture, &targets).resolve(&mut work),
            Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
        ));
        assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    }
}

#[test]
fn forwarded_memo_cycles_have_deterministic_representatives_and_literal_work() {
    for (targets, expected_work, representative) in [
        (vec![Some(0)], 14, 0),
        (vec![Some(1), Some(0)], 24, 0),
        (vec![Some(1), Some(2), Some(0)], 34, 0),
        (vec![Some(1), Some(2), Some(1), Some(0)], 45, 1),
    ] {
        let fixture = fixture(targets.len());
        let mut work = 0;
        let result = memo_index(&fixture, &targets).resolve(&mut work).unwrap();
        assert_eq!(work, expected_work);
        assert_resolved_to(&result, fixture.values[representative]);
        let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - expected_work + 1;
        assert!(memo_index(&fixture, &targets).resolve(&mut work).is_err());
        assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    }
}

#[test]
fn forwarded_memo_deep_and_shared_paths_are_linear_and_nonrecursive() {
    for (count, chain_work, shared_work, cycle_work) in
        [(257, 2_313, 3_078, 2_574), (1_024, 9_216, 12_282, 10_244)]
    {
        let fixture = fixture(count);
        let chain = (0..count)
            .map(|index| (index + 1 < count).then_some(index + 1))
            .collect::<Vec<_>>();
        let mut work = 0;
        let result = memo_index(&fixture, &chain).resolve(&mut work).unwrap();
        assert_eq!(work, chain_work);
        assert_resolved_to(&result, fixture.zero);
        let mut shared = vec![Some(count - 1); count];
        shared[count - 1] = None;
        let mut work = 0;
        let result = memo_index(&fixture, &shared).resolve(&mut work).unwrap();
        assert_eq!(work, shared_work);
        assert_resolved_to(&result, fixture.zero);
        let cycle = (0..count)
            .map(|index| Some((index + 1) % count))
            .collect::<Vec<_>>();
        let mut work = 0;
        let result = memo_index(&fixture, &cycle).resolve(&mut work).unwrap();
        assert_eq!(work, cycle_work);
        assert_resolved_to(&result, fixture.values[0]);
    }
}

#[test]
fn forwarded_storage_and_empty_rosters_preserve_caps_and_attempted_work() {
    let fixture = fixture(0);
    let mut work = 0;
    let empty = SubgroupForwardingIndexV1::new(fixture.entry, 0, &mut work).unwrap();
    assert_eq!(work, 0);
    assert!(
        empty
            .finish(&fixture.context, &mut work)
            .unwrap()
            .0
            .rows
            .is_empty()
    );
    assert_eq!(work, 1);
    assert_eq!(subgroup_forwarding_storage_items_v1(0), Some(0));
    assert_eq!(subgroup_forwarding_storage_items_v1(usize::MAX), None);
    let mut work = 0;
    assert!(
        SubgroupForwardingIndexV1::new(
            fixture.entry,
            MAX_PLIRON_TENSOR_UNIFORMITY_VALUES_V1 + 1,
            &mut work,
        )
        .is_err()
    );
    assert_eq!(work, 0);
    let storage = std::mem::size_of::<SubgroupForwardingRowV1<'static>>()
        .div_ceil(std::mem::size_of::<usize>())
        + 1
        + 64
        + std::mem::size_of::<SubgroupForwardingIndexV1<'static>>()
            .div_ceil(std::mem::size_of::<usize>());
    assert_eq!(subgroup_forwarding_storage_items_v1(1), Some(storage));
    let setup_work = storage + 64 + 8;
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - setup_work;
    assert!(SubgroupForwardingIndexV1::new(fixture.entry, 1, &mut work).is_ok());
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - setup_work + 1;
    assert!(SubgroupForwardingIndexV1::new(fixture.entry, 1, &mut work).is_err());
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    let mut work = usize::MAX;
    assert!(SubgroupForwardingIndexV1::new(fixture.entry, 1, &mut work).is_err());
    assert_eq!(work, usize::MAX);
}

#[test]
fn forwarded_type_observation_has_literal_valid_stale_and_denied_prefixes() {
    let mut fixture = fixture(1);
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - 7;
    subgroup_forwarding_type_v1(&fixture.context, fixture.values[0], &mut work).unwrap();
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - 6;
    assert!(matches!(
        subgroup_forwarding_type_v1(&fixture.context, fixture.values[0], &mut work),
        Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
    ));
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    let index_type: TypeHandle = dialect_kernel::IndexType::get(&fixture.context).into();
    let owner = BasicBlock::new(&mut fixture.context, None, vec![index_type]);
    let stale = owner.deref(&fixture.context).get_argument(0);
    BasicBlock::remove_argument(owner, &fixture.context, 0);
    let mut work = 0;
    assert!(matches!(
        subgroup_forwarding_type_v1(&fixture.context, stale, &mut work),
        Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { .. })
    ));
    assert_eq!(work, 5);
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - 4;
    assert!(matches!(
        subgroup_forwarding_type_v1(&fixture.context, stale, &mut work),
        Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
    ));
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    let foreign = self::fixture(0);
    let mut work = 0;
    assert!(subgroup_forwarding_type_v1(&fixture.context, foreign.zero, &mut work).is_err());
    assert_eq!(work, 2);
}

#[test]
fn forwarded_hash_and_bucket_publication_pay_before_observation_or_mutation() {
    let fixture = fixture(3);
    let hash_work = 16 + 2 * std::mem::size_of::<Value>();
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - hash_work;
    let hash = subgroup_forwarding_hash_v1(fixture.values[0], &mut work).unwrap();
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    assert_eq!(
        subgroup_forwarding_hash_v1(fixture.values[0], &mut 0).unwrap(),
        hash
    );
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - hash_work + 1;
    assert!(subgroup_forwarding_hash_v1(fixture.values[0], &mut work).is_err());
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);

    // Two admission observations, a 25-unit empty query, then 27 for insert/push.
    let mut exact = SubgroupForwardingIndexV1::new(fixture.entry, 3, &mut 0).unwrap();
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - 54;
    exact
        .push_hashed(fixture.values[0], &[], 7, &mut work)
        .unwrap();
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    assert_eq!(exact.rows.len(), 1);
    assert_eq!(exact.buckets.get(&7), Some(&0));
    let mut denied = SubgroupForwardingIndexV1::new(fixture.entry, 3, &mut 0).unwrap();
    let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - 53;
    assert!(
        denied
            .push_hashed(fixture.values[0], &[], 7, &mut work)
            .is_err()
    );
    assert_eq!(work, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 + 1);
    assert!(denied.rows.is_empty());
    assert!(denied.buckets.is_empty());
}

#[test]
fn anchored_self_phi_and_transitive_joins_preserve_the_original_value() {
    let fixture = fixture(6);
    let v = &fixture.values;
    let inputs = vec![
        vec![fixture.zero, v[0]],
        vec![v[0], v[0]],
        vec![v[1], v[0]],
        vec![v[2], v[3]],
        vec![v[3], v[0]],
        vec![fixture.zero, fixture.one],
    ];
    let result = index(&fixture, &inputs)
        .finish(&fixture.context, &mut 0)
        .unwrap();
    for ordinal in 0..5 {
        assert_eq!(
            result.representative(v[ordinal], &mut 0).unwrap(),
            fixture.zero
        );
        assert!(result.inputs_equal(ordinal, &mut 0).unwrap());
    }
    assert_eq!(result.representative(v[5], &mut 0).unwrap(), v[5]);
    assert!(!result.inputs_equal(5, &mut 0).unwrap());
}

#[test]
fn unanchored_cycles_and_distinct_external_values_are_not_collapsed() {
    let fixture = fixture(4);
    let v = &fixture.values;
    let inputs = vec![
        vec![v[0], v[0]],
        vec![fixture.zero, fixture.one, v[1]],
        vec![v[1], v[2]],
        vec![v[0], v[3]],
    ];
    let result = index(&fixture, &inputs)
        .finish(&fixture.context, &mut 0)
        .unwrap();
    assert_eq!(result.representative(v[0], &mut 0).unwrap(), v[0]);
    assert_eq!(result.representative(v[1], &mut 0).unwrap(), v[1]);
    assert_ne!(result.representative(v[2], &mut 0).unwrap(), fixture.zero);
    assert_ne!(result.representative(v[3], &mut 0).unwrap(), fixture.zero);
    assert!(!result.inputs_equal(1, &mut 0).unwrap());
}

#[test]
fn unknown_anchor_is_preserved_as_an_unknown_value_not_a_constant() {
    let mut fixture = fixture(2);
    let unknown =
        dialect_kernel::IndexUnknownOp::new(&mut fixture.context).result(&fixture.context);
    let inputs = vec![
        vec![unknown, fixture.values[0]],
        vec![fixture.values[0], unknown],
    ];
    let result = index(&fixture, &inputs)
        .finish(&fixture.context, &mut 0)
        .unwrap();
    for value in &fixture.values {
        assert_eq!(result.representative(*value, &mut 0).unwrap(), unknown);
        assert_ne!(result.representative(*value, &mut 0).unwrap(), fixture.zero);
    }
}

#[test]
fn refined_forwarding_pays_before_each_visit_and_rejects_one_unit_short() {
    let fixture = fixture(3);
    let inputs = vec![
        vec![fixture.zero, fixture.values[0]],
        vec![fixture.values[0], fixture.values[1]],
        vec![fixture.values[1], fixture.zero],
    ];
    let mut used = 0;
    index(&fixture, &inputs)
        .finish(&fixture.context, &mut used)
        .unwrap();
    assert!(used > 0 && used < MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    let mut exact = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - used;
    assert!(
        index(&fixture, &inputs)
            .finish(&fixture.context, &mut exact)
            .is_ok()
    );
    assert_eq!(exact, MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    let mut short = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 - used + 1;
    assert!(matches!(
        index(&fixture, &inputs).finish(&fixture.context, &mut short),
        Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
    ));
}

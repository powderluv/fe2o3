use super::*;

#[test]
fn numerical_mode_cli_is_explicit_closed_and_never_replays_under_changed_semantics() {
    let parse = |args: &[&str]| parse_options(args.iter().map(OsString::from));
    let base = ["--bundle-v6", "actual.bundle", "--request", "actual.json"];
    assert_eq!(
        parse(&base).unwrap().matrix_bf16_numerics,
        fe2o3_kir_sim::MatrixBf16NumericsV1::ExactV1
    );
    let mut finite = base.to_vec();
    finite.extend(["--matrix-bf16-numerics", "finite-sequential-fma-v1"]);
    assert_eq!(
        parse(&finite).unwrap().matrix_bf16_numerics,
        fe2o3_kir_sim::MatrixBf16NumericsV1::FiniteSequentialFmaV1
    );
    for extra in [
        vec!["--matrix-bf16-numerics", "exact-v1"],
        vec!["--record-canonical-schedule", "x"],
        vec!["--replay-schedule", "x"],
        vec!["--reduce-failure"],
    ] {
        let mut invalid = finite.clone();
        invalid.extend(extra);
        assert!(parse(&invalid).is_err());
    }
    let mut unknown = base.to_vec();
    unknown.extend(["--matrix-bf16-numerics", "unknown"]);
    assert!(parse(&unknown).is_err());
    let mut raw = finite.clone();
    raw[0] = "--kir-v12";
    assert!(parse(&raw).is_err());
}

#[test]
fn exact_matrix_domain_error_has_closed_positional_details_only() {
    for (role, name) in [
        (fe2o3_kir_sim::MatrixInputRoleV1::A, "a"),
        (fe2o3_kir_sim::MatrixInputRoleV1::B, "b"),
        (fe2o3_kir_sim::MatrixInputRoleV1::Accumulator, "accumulator"),
    ] {
        let error = SimulationExecutionErrorKindV1::UnsupportedMatrixInputDomain {
            role,
            lane: 63,
            component: 3,
        };
        assert_eq!(
            serde_json::to_value(execution_kind(&error)).unwrap(),
            "execution_unsupported_matrix_input_domain"
        );
        assert_eq!(
            serde_json::to_value(execution_detail(&error)).unwrap(),
            serde_json::json!({
                "kind": "unsupported_matrix_input_domain", "role": name, "lane": 63, "component": 3,
            })
        );
    }
}

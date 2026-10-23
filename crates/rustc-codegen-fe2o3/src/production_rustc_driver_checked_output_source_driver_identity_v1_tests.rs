use super::*;

fn original() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../config/tutorial-kernel-manifest-v1.json"
    ))
    .unwrap()
}

fn selected(value: &serde_json::Value) -> Result<Vec<Selected>, SourceFailure> {
    source_driver_selections(&serde_json::to_vec(value).unwrap())
}

fn item(value: &mut serde_json::Value) -> &mut serde_json::Value {
    &mut value["curriculum"]["lessons"][4]["codeTabs"][0]["sourceItem"]
}

#[test]
fn source_driver_census_preserves_all_eleven_exact_inputs_and_three_negatives() {
    let document = original();
    let rows = selected(&document).unwrap();
    assert_eq!(rows.len(), POSITIVES);
    let expected = [
        (
            0,
            0,
            "barrier_before_access",
            "gfx942",
            "barrier_before_access",
        ),
        (
            0,
            1,
            "aggregate_pair_struct",
            "gfx942",
            "aggregate_pair_struct",
        ),
        (
            0,
            2,
            "aggregate_pair_tuple",
            "gfx942",
            "aggregate_pair_tuple",
        ),
        (
            0,
            3,
            "aggregate_pair_array",
            "gfx942",
            "aggregate_pair_array",
        ),
        (0, 4, "aggregate_zst", "gfx942", "aggregate_zst"),
        (0, 5, "aggregate_nested", "gfx942", "aggregate_nested"),
        (0, 9, "wave_reduce_f32", "gfx950", "wave_reduce_f32"),
        (
            0,
            10,
            "workgroup_reduce_u32",
            "gfx942",
            "workgroup_reduce_u32",
        ),
        (
            0,
            11,
            "workgroup_reduce_i32",
            "gfx942",
            "workgroup_reduce_i32",
        ),
        (
            0,
            12,
            "workgroup_reduce_f32",
            "gfx942",
            "workgroup_reduce_f32",
        ),
        (
            6,
            0,
            "row_affine_sum_u32_v1",
            "gfx942",
            "row-affine-u32-kernel",
        ),
    ];
    for (row, (tab, case, symbol, target, feature)) in rows.iter().zip(expected) {
        assert_eq!(row.selection, source_selection(tab, case));
        assert_eq!(row.fixture.target, target);
        assert_eq!(row.fixture.compiler_input.kernel_symbols, [symbol]);
        assert_eq!(row.fixture.compiler_input.features, [feature]);
        assert_eq!(row.fixture.compiler_input.default_features, tab == 0);
        assert!(
            row.fixture
                .fixture_id
                .starts_with("source-driver-cpu-semantic-simulation-tab")
        );
        let source = &document["curriculum"]["lessons"][4]["codeTabs"][tab]["sourceItem"];
        let mut expected_input = source["compilerInput"].clone();
        expected_input["features"] = source["cases"][case]["features"].clone();
        expected_input["kernelSymbols"] = serde_json::json!([symbol]);
        assert_eq!(
            serde_json::to_value(&row.fixture.compiler_input).unwrap(),
            expected_input
        );
        assert_eq!(
            serde_json::to_value(&row.original_driver).unwrap(),
            source["driver"]
        );
        assert_eq!(
            row.original_driver_test,
            source["cases"][case]["testFunction"].as_str().unwrap()
        );
        assert_eq!(
            row.source_item_contract_sha256,
            source["contractSha256"].as_str().unwrap()
        );
        assert_eq!(
            serde_json::to_value(&row.original_expectation).unwrap(),
            source["cases"][case]["expectation"]
        );
    }
    assert_eq!(
        document["kernelInventory"]["negativeCases"]
            .as_array()
            .unwrap()
            .len(),
        NEGATIVES
    );
    let bytes = serde_json::to_vec(&document).unwrap();
    assert_eq!(fixtures(&bytes).unwrap().len(), CONFIGURATIONS);
    assert_eq!(
        document,
        original(),
        "selection must not rewrite the source contract"
    );
}

#[test]
fn source_driver_census_rejects_missing_extra_and_swapped_contract_cases() {
    for change in 0..8 {
        let mut document = original();
        match change {
            0 => {
                item(&mut document)["cases"]
                    .as_array_mut()
                    .unwrap()
                    .remove(0);
            }
            1 => {
                let duplicate = item(&mut document)["cases"][0].clone();
                item(&mut document)["cases"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            2 => {
                item(&mut document)["cases"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
            }
            3 => {
                document["curriculum"]["lessons"][4]["codeTabs"][6]["sourceItem"] =
                    serde_json::Value::Null;
            }
            4 => {
                let extra = document["curriculum"]["lessons"][4].clone();
                document["curriculum"]["lessons"]
                    .as_array_mut()
                    .unwrap()
                    .push(extra);
            }
            5 => {
                document["curriculum"]["lessons"][4]["codeTabs"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
            }
            6 => {
                item(&mut document)["cases"][0]["kernelSymbol"] =
                    serde_json::json!("different_root");
            }
            7 => {
                item(&mut document)["cases"][9]["target"] = serde_json::json!("gfx942");
            }
            _ => unreachable!(),
        }
        assert!(
            selected(&document).is_err(),
            "accepted census mutant {change}"
        );
    }
}

#[test]
fn source_driver_census_accepts_exact_fixture_aliases_but_refuses_changed_inputs() {
    let document = original();
    let aliases: Vec<_> = document["kernelInventory"]["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|kernel| kernel["selections"].as_array().unwrap().len() > 1)
        .collect();
    assert_eq!(aliases.len(), 5);
    assert_eq!(selected(&document).unwrap().len(), POSITIVES);
    for kernel in aliases {
        assert_eq!(kernel["selections"].as_array().unwrap().len(), 2);
        let id = kernel["selections"][1]["fixtureId"].as_str().unwrap();
        // Keep the global 11/84 profile totals unchanged so this exercises
        // alias target identity, not the earlier roster-count refusal.
        let mut modified = document.clone();
        let fixtures = modified["compilerFixtures"].as_array_mut().unwrap();
        fixtures
            .iter_mut()
            .find(|row| row["fixtureId"] == id)
            .unwrap()["target"] = "gfx942".into();
        fixtures[0]["target"] = "gfx950".into();
        assert_eq!(
            selected(&modified).unwrap_err().detail,
            "duplicate or ambiguous inventory kernel identity"
        );
        // Every source-bearing field remains significant even for a real
        // canonical alias. Preserve roster, counts and valid field syntax.
        for field in [
            "packageManifest",
            "packageManifestSha256",
            "cargoLockPath",
            "cargoLockSha256",
            "sourcePaths",
            "sourceClosureSha256",
            "cargoTarget",
            "defaultFeatures",
            "features",
        ] {
            let mut modified = document.clone();
            let fixture = modified["compilerFixtures"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|row| row["fixtureId"] == id)
                .unwrap();
            let input = &mut fixture["compilerInput"];
            match field {
                "defaultFeatures" => input[field] = (!input[field].as_bool().unwrap()).into(),
                "features" => input[field]
                    .as_array_mut()
                    .unwrap()
                    .push("different-feature".into()),
                "sourcePaths" => input[field]
                    .as_array_mut()
                    .unwrap()
                    .push("different.rs".into()),
                "cargoTarget" => input[field]["name"] = "different_target".into(),
                field if field.ends_with("Sha256") => input[field] = "0".repeat(64).into(),
                _ => input[field] = "different/path".into(),
            }
            let error = selected(&modified).unwrap_err();
            assert_eq!(
                error.detail, "duplicate or ambiguous inventory kernel identity",
                "{id} {field}"
            );
        }
    }
}

#[test]
fn source_driver_census_rejects_inventory_aliases_and_incomplete_identity_joins() {
    for change in 0..8 {
        let mut document = original();
        let kernels = document["kernelInventory"]["kernels"]
            .as_array_mut()
            .unwrap();
        let driver = kernels
            .iter()
            .position(|row| row["selections"][0]["kind"] == "source-driver-case")
            .unwrap();
        match change {
            0 => {
                kernels.remove(driver);
            }
            1 => {
                kernels.push(kernels[driver].clone());
            }
            2 => {
                kernels[driver]["kernelId"] = serde_json::json!("source-driver:wrong-root");
            }
            3 => {
                kernels[driver]["selections"][0]["lessonId"] = serde_json::json!("other-lesson");
            }
            4 => {
                let duplicate = kernels[driver]["selections"][0].clone();
                kernels[driver]["selections"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            5 => {
                kernels[driver]["selections"][0]["kind"] = serde_json::json!("unknown-source");
            }
            6 => {
                kernels[0]["selections"][0]["kernelSymbol"] = serde_json::json!("different_root");
            }
            7 => {
                document["kernelInventory"]["negativeCases"][0] =
                    document["kernelInventory"]["negativeCases"][1].clone();
            }
            _ => unreachable!(),
        }
        assert!(
            selected(&document).is_err(),
            "accepted inventory mutant {change}"
        );
    }
}

#[test]
fn source_driver_census_rejects_boolean_integer_and_unknown_field_confusion() {
    for change in 0..10 {
        let mut document = original();
        match change {
            0 => {
                item(&mut document)["compilerInput"]["defaultFeatures"] = serde_json::json!(0);
            }
            1 => {
                document["curriculum"]["lessons"][4]["codeTabs"][0]["ordinal"] =
                    serde_json::json!(false);
            }
            2 => {
                item(&mut document)["cases"][0]["displayedFragmentOrdinal"] =
                    serde_json::json!(false);
            }
            3 => {
                item(&mut document)["cases"][0]["expectation"]["bundleVersion"] =
                    serde_json::json!(true);
            }
            4 => {
                item(&mut document)["sourceRanges"][0]["byteOffset"] = serde_json::json!(false);
            }
            5 => {
                document["kernelInventory"]["negativeCases"][0]["caseOrdinal"] =
                    serde_json::json!(true);
            }
            6 => {
                item(&mut document)["cases"][0]["features"] =
                    serde_json::json!("barrier_before_access");
            }
            7 => {
                item(&mut document)["cases"][0]["expectation"]["outputArtifact"] =
                    serde_json::json!("absent");
            }
            8 => {
                item(&mut document)["compilerInput"]["features"] = serde_json::json!([]);
            }
            9 => {
                item(&mut document)["compilerInput"]["cargoTarget"]["extra"] =
                    serde_json::json!(true);
            }
            _ => unreachable!(),
        }
        assert!(
            selected(&document).is_err(),
            "accepted typed mutant {change}"
        );
    }
}

#[test]
fn source_driver_census_rejects_malformed_paths_hashes_ranges_and_features() {
    for change in 0..8 {
        let mut document = original();
        match change {
            0 => {
                item(&mut document)["compilerInput"]["packageManifest"] =
                    serde_json::json!("../outside/Cargo.toml");
            }
            1 => {
                item(&mut document)["compilerInput"]["cargoLockSha256"] =
                    serde_json::json!("a".repeat(63));
            }
            2 => {
                item(&mut document)["contractSha256"] = serde_json::json!("A".repeat(64));
            }
            3 => {
                item(&mut document)["cases"][0]["features"] = serde_json::json!(["same", "same"]);
            }
            4 => {
                item(&mut document)["sourceRanges"][0]["byteLength"] = serde_json::json!(0);
            }
            5 => {
                item(&mut document)["cases"][0]["displayedFragmentOrdinal"] = serde_json::json!(64);
            }
            6 => {
                item(&mut document)["driver"]["path"] = serde_json::json!("different/test.rs");
            }
            7 => {
                item(&mut document)["compilerInput"]["sourcePaths"] =
                    serde_json::json!(["examples/other.rs"]);
            }
            _ => unreachable!(),
        }
        assert!(
            selected(&document).is_err(),
            "accepted shape mutant {change}"
        );
    }
    assert!(source_driver_selections(&vec![b' '; MAX_MANIFEST_BYTES + 1]).is_err());
}

#[test]
fn source_driver_census_never_turns_required_negatives_into_positive_or_ignored_cases() {
    for change in 0..5 {
        let mut document = original();
        match change {
            0 => {
                item(&mut document)["cases"][6]["expectation"] =
                    serde_json::json!({"kind":"verified-bundle-export","bundleVersion":4});
            }
            1 => {
                item(&mut document)["cases"][0]["expectation"] =
                    item(&mut document)["cases"][6]["expectation"].clone();
            }
            2 => {
                item(&mut document)["cases"][6]["expectation"]["diagnosticContains"] =
                    serde_json::json!("");
            }
            3 => {
                item(&mut document)["cases"][6]["expectation"]["outputArtifact"] =
                    serde_json::json!("present");
            }
            4 => {
                document["kernelInventory"]["negativeCases"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            _ => unreachable!(),
        }
        assert!(
            selected(&document).is_err(),
            "accepted negative mutant {change}"
        );
    }
}

#[test]
fn source_driver_corpus_report_preserves_refusal_without_artifact_authority() {
    let mut rows = selected(&original()).unwrap();
    let row = rows.remove(0);
    let blocked = CaseReport::blocked(&row.fixture, invalid("source validation refused"));
    assert!(!blocked.passed());
    let report = DriverReport {
        schema: "fe2o3-ordinary-source-policy4-source-driver-corpus-v1",
        manifest_sha256: "1".repeat(64),
        configurations: 1,
        strict_negative_drivers_unchanged: NEGATIVES,
        all_checked_output_passed: false,
        default_pipeline_activated: false,
        grants_artifact_or_launch_authority: false,
        cases: vec![DriverCaseReport {
            source: row,
            result: blocked,
        }],
    };
    let json = serde_json::to_value(report).unwrap();
    assert_eq!(
        json["cases"][0]["source"]["selection"]["kind"],
        "source-driver-case"
    );
    assert_eq!(json["cases"][0]["result"]["status"], "blocked");
    assert_eq!(json["all_checked_output_passed"], false);
    assert_eq!(json["default_pipeline_activated"], false);
    assert_eq!(json["grants_artifact_or_launch_authority"], false);
}

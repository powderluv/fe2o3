//! Additional source-driver selections, never aliases for the ordinary fixture cases.
use super::*;
use std::collections::BTreeMap;

const SOURCE_DRIVER_REPORT: &str = "FE2O3_TEST_CHECKED_OUTPUT_SOURCE_DRIVER_REPORT_V1";
const POSITIVES: usize = 11;
const NEGATIVES: usize = 3;
const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum Selection {
    Fixture {
        #[serde(rename = "fixtureId")]
        fixture_id: String,
        #[serde(rename = "kernelSymbol")]
        kernel_symbol: String,
    },
    SourceDriverCase {
        #[serde(rename = "lessonId")]
        lesson_id: String,
        #[serde(rename = "tabOrdinal")]
        tab_ordinal: usize,
        #[serde(rename = "caseOrdinal")]
        case_ordinal: usize,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Driver {
    package: String,
    target: String,
    path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum Expectation {
    VerifiedBundleExport {
        #[serde(rename = "bundleVersion")]
        bundle_version: u8,
    },
    Rejected {
        #[serde(rename = "bundleVersion")]
        bundle_version: u8,
        #[serde(rename = "diagnosticContains")]
        diagnostic_contains: String,
        #[serde(rename = "outputArtifact")]
        output_artifact: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DriverCase {
    features: Vec<String>,
    kernel_symbol: String,
    target: String,
    displayed_fragment_ordinal: usize,
    test_function: String,
    expectation: Expectation,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SourceRange {
    byte_offset: usize,
    byte_length: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SourceItem {
    kind: String,
    compiler_input: serde_json::Value,
    driver: Driver,
    source_ranges: Vec<SourceRange>,
    cases: Vec<DriverCase>,
    contract_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Tab {
    ordinal: usize,
    kind: String,
    language: String,
    source_path: Option<String>,
    source_item_status: String,
    source_item: Option<SourceItem>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Lesson {
    lesson_id: String,
    role: String,
    code_tabs: Vec<Tab>,
}

#[derive(Debug, Deserialize)]
struct Curriculum {
    schema: String,
    lessons: Vec<Lesson>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InventoryKernel {
    kernel_id: String,
    selections: Vec<Selection>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Inventory {
    schema: String,
    kernels: Vec<InventoryKernel>,
    negative_cases: Vec<Selection>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Document {
    curriculum: Curriculum,
    kernel_inventory: Inventory,
}

#[derive(Debug, Serialize)]
struct Selected {
    selection: Selection,
    source_item_contract_sha256: String,
    original_driver: Driver,
    original_driver_test: String,
    original_expectation: Expectation,
    fixture: Fixture,
}

fn invalid(detail: &str) -> SourceFailure {
    fail(SourceStage::Manifest, detail)
}

fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn source_selection(tab: usize, case: usize) -> Selection {
    Selection::SourceDriverCase {
        lesson_id: "cpu-semantic-simulation".to_owned(),
        tab_ordinal: tab,
        case_ordinal: case,
    }
}

fn expected_selections(positive: bool) -> BTreeSet<Selection> {
    // Explicit source contract coordinates, not a search by kernel name.
    if positive {
        [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]
            .into_iter()
            .map(|case| source_selection(0, case))
            .chain(std::iter::once(source_selection(6, 0)))
            .collect()
    } else {
        [6, 7, 8]
            .into_iter()
            .map(|case| source_selection(0, case))
            .collect()
    }
}

fn input_for_case(
    shared: &serde_json::Value,
    source_path: Option<&str>,
    case: &DriverCase,
) -> Result<CompilerInput, SourceFailure> {
    let mut object = shared
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("source input object"))?;
    let keys: BTreeSet<_> = object.keys().map(String::as_str).collect();
    if keys
        != BTreeSet::from([
            "packageManifest",
            "packageManifestSha256",
            "cargoLockPath",
            "cargoLockSha256",
            "sourcePaths",
            "sourceClosureSha256",
            "cargoTarget",
            "defaultFeatures",
        ])
    {
        return Err(invalid("source-driver shared input keys differ"));
    }
    let target = object["cargoTarget"]
        .as_object()
        .ok_or_else(|| invalid("Cargo target object"))?;
    if target.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from(["kind", "name", "sourcePath"])
    {
        return Err(invalid("source-driver Cargo target keys differ"));
    }
    object.insert("features".to_owned(), serde_json::json!(case.features));
    object.insert(
        "kernelSymbols".to_owned(),
        serde_json::json!([case.kernel_symbol]),
    );
    let input: CompilerInput = serde_json::from_value(object.into())
        .map_err(|error| fail(SourceStage::Manifest, error))?;
    if !relative(&input.package_manifest)
        || !relative(&input.cargo_lock_path)
        || !relative(&input.cargo_target.source_path)
        || input.cargo_target.kind != "lib"
        || !identifier(&input.cargo_target.name)
        || input.source_paths.len() != 1
        || input.source_paths.first().map(String::as_str) != source_path
        || !input.source_paths.iter().all(|path| relative(path))
        || !identifier(&case.kernel_symbol)
        || !identifier(&case.test_function)
        || case.features.len() > 256
        || !case.features.iter().all(|feature| identifier(feature))
        || case.features.iter().collect::<BTreeSet<_>>().len() != case.features.len()
        || ![
            &input.package_manifest_sha256,
            &input.cargo_lock_sha256,
            &input.source_closure_sha256,
        ]
        .into_iter()
        .all(|value| hash(value))
    {
        return Err(invalid("source-driver input identity differs"));
    }
    Ok(input)
}

// Ordinary aliases follow the manifest validator's source-selection identity:
// all compiler input fields and the selected symbol must agree. Per-fixture
// contractSha256 includes the fixture reference and is deliberately not part
// of CompilerInput, just as in tutorial_kernel_identities.py::INPUT_FIELDS.
// This corpus admits only same-target aliases; source-driver cases retain
// their separate, single-selection identities.
fn fixture_aliases_match(kernel: &InventoryKernel, ordinary: &[Fixture]) -> bool {
    if kernel.selections.len() == 1 {
        return true;
    }
    if kernel.selections.is_empty() || kernel.selections.len() > CONFIGURATIONS {
        return false;
    }
    let Selection::Fixture {
        fixture_id,
        kernel_symbol,
    } = &kernel.selections[0]
    else {
        return false;
    };
    let Some(first) = ordinary.iter().find(|row| &row.fixture_id == fixture_id) else {
        return false;
    };
    if kernel.kernel_id != format!("fixture:{fixture_id}:{kernel_symbol}") {
        return false;
    }
    kernel.selections.iter().skip(1).all(|selection| {
        let Selection::Fixture {
            fixture_id,
            kernel_symbol: symbol,
        } = selection
        else {
            return false;
        };
        ordinary.iter().any(|row| {
            &row.fixture_id == fixture_id
                && symbol == kernel_symbol
                && row.target == first.target
                && row.compiler_input == first.compiler_input
        })
    })
}

fn source_driver_selections(bytes: &[u8]) -> Result<Vec<Selected>, SourceFailure> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(invalid("manifest exceeds byte bound"));
    }
    let ordinary = fixtures(bytes)?;
    let document: Document =
        serde_json::from_slice(bytes).map_err(|error| fail(SourceStage::Manifest, error))?;
    if document.curriculum.schema != "fe2o3-tutorial-curriculum-obligations-v2"
        || document.kernel_inventory.schema != "fe2o3-tutorial-kernel-identities-v1"
        || document.curriculum.lessons.is_empty()
        || document.curriculum.lessons.len() > 256
    {
        return Err(invalid("source-driver curriculum/inventory schema"));
    }
    let mut lessons = BTreeSet::new();
    let mut selected = Vec::new();
    let mut positives = BTreeMap::new();
    let mut negatives = BTreeSet::new();
    let mut profiles = [0; 2];
    let mut items = 0;
    for lesson in document.curriculum.lessons {
        if !identifier(&lesson.lesson_id)
            || !lessons.insert(lesson.lesson_id.clone())
            || lesson.code_tabs.len() > 64
        {
            return Err(invalid("duplicate/malformed lesson identity"));
        }
        for (ordinal, tab) in lesson.code_tabs.into_iter().enumerate() {
            if tab.ordinal != ordinal {
                return Err(invalid("tab ordinal differs from original order"));
            }
            let Some(item) = tab.source_item else {
                continue;
            };
            items += 1;
            if items > 2
                || lesson.role != "executable"
                || tab.kind != "kernel"
                || tab.language != "rust"
                || tab.source_item_status != "contract-bound"
                || item.kind != "source-driver"
                || !hash(&item.contract_sha256)
                || item.cases.is_empty()
                || item.cases.len() > 256
                || item.source_ranges.is_empty()
                || item.source_ranges.len() > 64
                || item.source_ranges.iter().any(|range| {
                    range.byte_length == 0
                        || range.byte_offset.checked_add(range.byte_length).is_none()
                })
                || item.driver.package != "rustc-codegen-fe2o3"
                || !identifier(&item.driver.target)
                || item.driver.path
                    != format!(
                        "crates/{}/tests/{}.rs",
                        item.driver.package, item.driver.target
                    )
            {
                return Err(invalid("source-driver contract shape differs"));
            }
            for (case_ordinal, case) in item.cases.into_iter().enumerate() {
                if case.displayed_fragment_ordinal >= item.source_ranges.len() {
                    return Err(invalid("source-driver fragment ordinal differs"));
                }
                let input =
                    input_for_case(&item.compiler_input, tab.source_path.as_deref(), &case)?;
                let selection = Selection::SourceDriverCase {
                    lesson_id: lesson.lesson_id.clone(),
                    tab_ordinal: ordinal,
                    case_ordinal,
                };
                let profile = match case.target.as_str() {
                    "gfx942" => 0,
                    "gfx950" => 1,
                    _ => return Err(invalid("source-driver target profile differs")),
                };
                match &case.expectation {
                    Expectation::VerifiedBundleExport { bundle_version }
                        if (1..=6).contains(bundle_version) => {}
                    Expectation::Rejected {
                        bundle_version,
                        diagnostic_contains,
                        output_artifact,
                    } if (1..=6).contains(bundle_version)
                        && !diagnostic_contains.is_empty()
                        && diagnostic_contains.len() <= 512
                        && output_artifact == "absent" =>
                    {
                        if !negatives.insert(selection) {
                            return Err(invalid("duplicate negative selection"));
                        }
                        continue;
                    }
                    _ => return Err(invalid("source-driver expectation differs")),
                }
                let kernel_id = format!(
                    "source-driver:{}:{ordinal}:{}",
                    lesson.lesson_id, case.kernel_symbol
                );
                if positives.insert(selection.clone(), kernel_id).is_some() {
                    return Err(invalid("duplicate positive selection"));
                }
                profiles[profile] += 1;
                selected.push(Selected {
                    selection,
                    source_item_contract_sha256: item.contract_sha256.clone(),
                    original_driver: item.driver.clone(),
                    original_driver_test: case.test_function,
                    original_expectation: case.expectation,
                    fixture: Fixture {
                        fixture_id: format!(
                            "source-driver-{}-tab{ordinal}-case{case_ordinal}",
                            lesson.lesson_id
                        ),
                        target: case.target,
                        compiler_input: input,
                    },
                });
            }
        }
    }
    if items != 2
        || positives.keys().cloned().collect::<BTreeSet<_>>() != expected_selections(true)
        || negatives != expected_selections(false)
        || profiles != [10, 1]
        || selected
            .iter()
            .flat_map(|row| &row.fixture.compiler_input.kernel_symbols)
            .collect::<BTreeSet<_>>()
            .len()
            != POSITIVES
    {
        return Err(invalid(
            "exact eleven-positive/three-negative source selection census changed",
        ));
    }

    let mut inventory_positive = BTreeMap::new();
    let mut inventory_fixture = BTreeSet::new();
    let mut kernel_ids = BTreeSet::new();
    for kernel in document.kernel_inventory.kernels {
        if !kernel_ids.insert(kernel.kernel_id.clone())
            || !fixture_aliases_match(&kernel, &ordinary)
        {
            return Err(invalid("duplicate or ambiguous inventory kernel identity"));
        }
        for selection in kernel.selections {
            match &selection {
                Selection::Fixture { .. } => {
                    if !inventory_fixture.insert(selection) {
                        return Err(invalid("duplicate inventory fixture"));
                    }
                }
                Selection::SourceDriverCase { .. } => {
                    if inventory_positive
                        .insert(selection, kernel.kernel_id.clone())
                        .is_some()
                    {
                        return Err(invalid("duplicate inventory source selection"));
                    }
                }
            }
        }
    }
    let expected_fixtures: BTreeSet<_> = ordinary
        .iter()
        .flat_map(|row| {
            row.compiler_input
                .kernel_symbols
                .iter()
                .map(|symbol| Selection::Fixture {
                    fixture_id: row.fixture_id.clone(),
                    kernel_symbol: symbol.clone(),
                })
        })
        .collect();
    let inventory_negative: BTreeSet<_> = document
        .kernel_inventory
        .negative_cases
        .iter()
        .cloned()
        .collect();
    if inventory_positive != positives
        || inventory_fixture != expected_fixtures
        || inventory_negative != negatives
        || document.kernel_inventory.negative_cases.len() != NEGATIVES
    {
        return Err(invalid(
            "complete source inventory does not rejoin exact selections",
        ));
    }
    Ok(selected)
}

fn validate_source_contract(workspace: &Path, bytes: &[u8]) -> Result<(), SourceFailure> {
    let path = workspace.join(MANIFEST);
    if std::fs::read(&path).map_err(|e| fail(SourceStage::Manifest, e))? != bytes {
        return Err(invalid("manifest changed before source validation"));
    }
    let output = Command::new("python3")
        .arg("-B")
        .current_dir(workspace)
        .arg(workspace.join("scripts/validate-tutorial-kernel-manifest.py"))
        .arg("--repo-root")
        .arg(workspace)
        .arg("--manifest")
        .arg(&path)
        .arg("--require-curriculum")
        .output()
        .map_err(|e| fail(SourceStage::Manifest, e))?;
    if !output.status.success() {
        return Err(fail(
            SourceStage::Manifest,
            corpus_cargo::diagnostics(&output),
        ));
    }
    if std::fs::read(path).map_err(|e| fail(SourceStage::Manifest, e))? != bytes {
        return Err(invalid("manifest changed during source validation"));
    }
    Ok(())
}

#[derive(Serialize)]
struct DriverCaseReport {
    source: Selected,
    result: CaseReport,
}

#[derive(Serialize)]
struct DriverReport {
    schema: &'static str,
    manifest_sha256: String,
    configurations: usize,
    strict_negative_drivers_unchanged: usize,
    all_checked_output_passed: bool,
    default_pipeline_activated: bool,
    grants_artifact_or_launch_authority: bool,
    cases: Vec<DriverCaseReport>,
}

#[test]
#[ignore = "additional eleven real Cargo/rustc P4 source-driver positives; requires pinned rust-src/dependencies; not the full tutorial corpus"]
fn ordinary_tutorial_source_driver_positives_require_checked_policy4_output() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let bytes = std::fs::read(workspace.join(MANIFEST)).unwrap();
    let selections = source_driver_selections(&bytes).expect("exact source-driver identity census");
    validate_source_contract(&workspace, &bytes)
        .expect("compiler-owned source/driver/closure validation");
    let scratch = crate::test_temp_dir::TestTempDir::create("fe2o3-policy4-source-drivers");
    let mut cases = Vec::with_capacity(POSITIVES);
    for selection in selections {
        let fixture = &selection.fixture;
        eprintln!(
            "P4 SOURCE DRIVER {} target={} status=RUNNING",
            fixture.fixture_id, fixture.target
        );
        let case = scratch.path().join(&fixture.fixture_id);
        let target = scratch.path().join("dependencies").join(&fixture.target);
        let observed = run_case(&workspace, fixture, &case, &target);
        eprintln!(
            "P4 SOURCE DRIVER {} status={} refusal={:?}",
            fixture.fixture_id, observed.status, observed.refusal
        );
        cases.push(DriverCaseReport {
            source: selection,
            result: observed,
        });
    }
    let passed = cases.len() == POSITIVES && cases.iter().all(|case| case.result.passed());
    let report = DriverReport {
        schema: "fe2o3-ordinary-source-policy4-source-driver-corpus-v1",
        manifest_sha256: digest(&bytes),
        configurations: cases.len(),
        strict_negative_drivers_unchanged: NEGATIVES,
        all_checked_output_passed: passed,
        default_pipeline_activated: false,
        grants_artifact_or_launch_authority: false,
        cases,
    };
    let encoded = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(path) = env::var_os(SOURCE_DRIVER_REPORT) {
        assert!(
            Path::new(&path).is_absolute(),
            "source-driver report path must be absolute"
        );
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create explicitly requested fresh source-driver report");
        std::io::Write::write_all(&mut output, &encoded).expect("write source-driver report");
    }
    eprintln!(
        "P4 SOURCE DRIVER REPORT\n{}",
        String::from_utf8(encoded).unwrap()
    );
    assert!(
        passed,
        "not qualified: one or more of the eleven source-driver positives is blocked"
    );
}

#[path = "production_rustc_driver_checked_output_source_driver_identity_v1_tests.rs"]
mod identity_tests;

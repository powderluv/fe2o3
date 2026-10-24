//! Full ordinary-source P4 extraction gate, not default/protected publication,
//! simulator, numerical, hardware, or trusted proof-execution qualification.
use super::*;
use std::collections::BTreeSet;
use std::path::Component;

const MANIFEST: &str = "config/tutorial-kernel-manifest-v1.json";
const REPORT: &str = "FE2O3_TEST_CHECKED_OUTPUT_CORPUS_REPORT_V1";
const CONFIGURATIONS: usize = 95;
const ROOTS: usize = 34;

// Original 50 configurations in order, followed by the 45 reviewed gfx950
// counterparts. A same-count replacement must not silently drop an obligation.
const FIXTURE_IDS: &[&str] = &[
    "gfx942-fill-simulation",
    "gfx942-flash-attention",
    "gfx942-gemm-autoresearch",
    "gfx942-grouped-expert-moe",
    "gfx942-moe-top2",
    "gfx942-row-softmax",
    "gfx942-scalar-gemm",
    "gfx942-tiled-gemm",
    "gfx942-typed-vecadd-source",
    "gfx942-wave64-collectives",
    "gfx942-workgroup-collectives",
    "gfx950-attnres-aggregate",
    "gfx950-attnres-aggregate-explicit-reuse",
    "gfx950-combine-expert-ranks",
    "gfx950-compressed-hybrid-attention",
    "gfx950-compressed-hybrid-attention-division-baseline",
    "gfx950-content-sparse-attention",
    "gfx950-content-sparse-attention-reciprocal-reuse",
    "gfx950-deepseek-sparse-attention",
    "gfx950-four-branch-residual",
    "gfx950-four-branch-residual-explicit",
    "gfx950-fp4-attention",
    "gfx950-fp4-gemm",
    "gfx950-fp8-attention",
    "gfx950-fp8-gemm",
    "gfx950-gpt-oss-decode",
    "gfx950-gpt-oss-held-fragments",
    "gfx950-gpt-oss-interleaved-stores",
    "gfx950-gpt-oss-materialized-attention",
    "gfx950-gpt-oss-materialized-expert",
    "gfx950-gpt-oss-materialized-router",
    "gfx950-gpt-oss-pipelined-attention",
    "gfx950-gpt-oss-scalar-attention",
    "gfx950-gpt-oss-serial-router",
    "gfx950-kda-decode",
    "gfx950-kda-decode-baseline",
    "gfx950-kda-prefill",
    "gfx950-kda-prefill-baseline",
    "gfx950-mhc-sinkhorn-mix",
    "gfx950-mhc-sinkhorn-mix-scalar",
    "gfx950-moe-expert-rank",
    "gfx950-moe-expert-rank-expert-serial",
    "gfx950-moe-route",
    "gfx950-muon-update",
    "gfx950-muon-update-broadcast16",
    "gfx950-qwen-ngram-gather",
    "gfx950-qwen-ngram-gather-reverse-probe",
    "gfx950-speculative-transaction",
    "gfx950-speculative-transaction-recompute-prefix",
    "gfx950-stage-gradient-shard",
    "gfx950-gpt-oss-decode-simt",
    "gfx950-gpt-oss-decode-tile",
    "gfx950-gpt-oss-held-fragments-simt",
    "gfx950-gpt-oss-held-fragments-tile",
    "gfx950-gpt-oss-interleaved-stores-simt",
    "gfx950-gpt-oss-interleaved-stores-tile",
    "gfx950-gpt-oss-materialized-attention-simt",
    "gfx950-gpt-oss-materialized-attention-tile",
    "gfx950-gpt-oss-materialized-expert-simt",
    "gfx950-gpt-oss-materialized-expert-tile",
    "gfx950-gpt-oss-materialized-router-simt",
    "gfx950-gpt-oss-materialized-router-tile",
    "gfx950-gpt-oss-pipelined-attention-simt",
    "gfx950-gpt-oss-pipelined-attention-tile",
    "gfx950-gpt-oss-scalar-attention-simt",
    "gfx950-gpt-oss-scalar-attention-tile",
    "gfx950-gpt-oss-serial-router-simt",
    "gfx950-gpt-oss-serial-router-tile",
    "gfx950-row-softmax-simt",
    "gfx950-row-softmax-tile",
    "gfx950-scalar-gemm-simt",
    "gfx950-scalar-gemm-tile",
    "gfx950-attnres-aggregate-simt",
    "gfx950-attnres-aggregate-tile",
    "gfx950-four-branch-residual-simt",
    "gfx950-four-branch-residual-tile",
    "gfx950-mhc-sinkhorn-mix-simt",
    "gfx950-mhc-sinkhorn-mix-tile",
    "gfx950-kda-decode-simt",
    "gfx950-kda-decode-tile",
    "gfx950-kda-prefill-simt",
    "gfx950-kda-prefill-tile",
    "gfx950-content-sparse-attention-simt",
    "gfx950-content-sparse-attention-tile",
    "gfx950-deepseek-sparse-attention-simt",
    "gfx950-deepseek-sparse-attention-tile",
    "gfx950-compressed-hybrid-attention-simt",
    "gfx950-compressed-hybrid-attention-tile",
    "gfx950-moe-expert-rank-simt",
    "gfx950-moe-route-tile",
    "gfx950-combine-expert-ranks-tile",
    "gfx950-speculative-transaction-tile",
    "gfx950-qwen-ngram-gather-tile",
    "gfx950-stage-gradient-shard-tile",
    "gfx950-muon-update-tile",
];

#[path = "production_rustc_driver_checked_output_source_driver_corpus_v1_tests.rs"]
mod source_driver_corpus;

#[path = "production_rustc_driver_checked_output_source_census_v1_tests.rs"]
mod source_census;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) struct CargoTarget {
    pub kind: String,
    pub name: String,
    pub source_path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompilerInput {
    pub package_manifest: String,
    pub package_manifest_sha256: String,
    pub cargo_lock_path: String,
    pub cargo_lock_sha256: String,
    pub source_paths: Vec<String>,
    pub source_closure_sha256: String,
    pub cargo_target: CargoTarget,
    pub default_features: bool,
    #[serde(default)]
    pub features: Vec<String>,
    pub kernel_symbols: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Fixture {
    pub fixture_id: String,
    pub target: String,
    pub compiler_input: CompilerInput,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    fixture: Fixture,
    status: &'static str,
    observation: Option<Observation>,
    refusal: Option<SourceFailure>,
    compiler_artifacts: Vec<String>,
    actual_cfg: Vec<String>,
    cargo_diagnostics: String,
    rustc_diagnostics: String,
    case_elapsed_millis: u64,
    callback_progress: Option<progress::Snapshot>,
    callback_progress_error: Option<String>,
    source_census: Option<serde_json::Value>,
    source_census_error: Option<String>,
}

impl CaseReport {
    fn blocked(fixture: &Fixture, error: SourceFailure) -> Self {
        Self {
            fixture: fixture.clone(),
            status: "blocked",
            observation: None,
            refusal: Some(error),
            compiler_artifacts: Vec::new(),
            actual_cfg: Vec::new(),
            cargo_diagnostics: String::new(),
            rustc_diagnostics: String::new(),
            case_elapsed_millis: 0,
            callback_progress: None,
            callback_progress_error: None,
            source_census: None,
            source_census_error: None,
        }
    }

    fn passed(&self) -> bool {
        self.status == "checked-output-pass"
            && self.observation.is_some()
            && self.refusal.is_none()
            && self.compiler_artifacts.is_empty()
    }
}

#[derive(Serialize)]
struct CorpusReport {
    schema: &'static str,
    manifest_sha256: String,
    configurations: usize,
    distinct_expected_roots: usize,
    all_checked_output_passed: bool,
    default_pipeline_activated: bool,
    grants_artifact_or_launch_authority: bool,
    cases: Vec<CaseReport>,
}

fn fail(stage: SourceStage, detail: impl std::fmt::Display) -> SourceFailure {
    SourceFailure::new(stage, detail)
}

fn relative(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

fn digest(bytes: &[u8]) -> String {
    crate::encode_hex(&Sha256::digest(bytes))
}

fn fixtures(bytes: &[u8]) -> Result<Vec<Fixture>, SourceFailure> {
    let manifest: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| fail(SourceStage::Manifest, e))?;
    let contract = &manifest["productionContract"];
    if manifest["schema"] != "fe2o3-tutorial-kernel-source-contract-v1"
        || contract["pipelineEntry"] != "rustc-codegen-fe2o3::production_pipeline"
        || contract["requiredPolicyVersion"] != 4
        || contract["requiresFinalOptimizedGraphVerification"] != true
        || contract["allowsPipelineSelection"] != false
        || contract["allowsFallback"] != false
    {
        return Err(fail(
            SourceStage::Manifest,
            "expected the fixed Policy4/no-fallback source contract",
        ));
    }
    let fixtures: Vec<Fixture> = serde_json::from_value(manifest["compilerFixtures"].clone())
        .map_err(|e| fail(SourceStage::Manifest, e))?;
    let mut ids = BTreeSet::new();
    let mut roots = BTreeSet::new();
    let mut profiles = [0; 2];
    for fixture in &fixtures {
        let input = &fixture.compiler_input;
        if !identifier(&fixture.fixture_id)
            || !ids.insert(fixture.fixture_id.clone())
            || !relative(&input.package_manifest)
            || !relative(&input.cargo_lock_path)
            || !relative(&input.cargo_target.source_path)
            || input.cargo_target.kind != "lib"
            || !identifier(&input.cargo_target.name)
            || input.source_paths.is_empty()
            || !input.source_paths.iter().all(|path| relative(path))
            || input.kernel_symbols.is_empty()
            || !input.kernel_symbols.iter().all(|name| identifier(name))
            || input.kernel_symbols.iter().collect::<BTreeSet<_>>().len()
                != input.kernel_symbols.len()
            || !input.features.iter().all(|name| identifier(name))
            || input.features.iter().collect::<BTreeSet<_>>().len() != input.features.len()
        {
            return Err(fail(
                SourceStage::Manifest,
                format!("invalid or duplicate fixture {}", fixture.fixture_id),
            ));
        }
        for hash in [
            &input.package_manifest_sha256,
            &input.cargo_lock_sha256,
            &input.source_closure_sha256,
        ] {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(fail(SourceStage::Manifest, "invalid expected input hash"));
            }
        }
        match fixture.target.as_str() {
            "gfx942" => profiles[0] += 1,
            "gfx950" => profiles[1] += 1,
            _ => {
                return Err(fail(
                    SourceStage::Manifest,
                    "unsupported exact target profile",
                ));
            }
        }
        roots.extend(input.kernel_symbols.iter().cloned());
    }
    if fixtures.len() != CONFIGURATIONS
        || roots.len() != ROOTS
        || profiles != [11, 84]
        || fixtures
            .iter()
            .map(|row| row.fixture_id.as_str())
            .ne(FIXTURE_IDS.iter().copied())
    {
        return Err(fail(
            SourceStage::Manifest,
            "complete 95-configuration/34-root roster changed; review coverage rather than silently shrinking it",
        ));
    }
    Ok(fixtures)
}

fn check_input_files(workspace: &Path, fixture: &Fixture) -> Result<(), SourceFailure> {
    for (path, expected) in [
        (
            &fixture.compiler_input.package_manifest,
            &fixture.compiler_input.package_manifest_sha256,
        ),
        (
            &fixture.compiler_input.cargo_lock_path,
            &fixture.compiler_input.cargo_lock_sha256,
        ),
    ] {
        let bytes =
            std::fs::read(workspace.join(path)).map_err(|e| fail(SourceStage::Manifest, e))?;
        if digest(&bytes) != *expected {
            return Err(fail(
                SourceStage::Manifest,
                format!("current input differs from manifest pin: {path}"),
            ));
        }
    }
    // Source closure scanning remains the compiler-owned manifest validator's
    // contract. This test compiles actual files and reports the exact manifest,
    // not a second, weaker source scanner or a qualification receipt.
    for path in &fixture.compiler_input.source_paths {
        let actual = workspace
            .join(path)
            .canonicalize()
            .map_err(|e| fail(SourceStage::Manifest, e))?;
        if !actual.starts_with(workspace) || !actual.is_file() {
            return Err(fail(
                SourceStage::Manifest,
                "source file escaped the workspace",
            ));
        }
    }
    Ok(())
}

fn artifacts(directory: &Path) -> Result<Vec<String>, SourceFailure> {
    let mut result = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|e| fail(SourceStage::Observation, e))? {
        let entry = entry.map_err(|e| fail(SourceStage::Observation, e))?;
        // A dependency-info file is not a compiled/native artifact. All other
        // files or directories are unexpected after Compilation::Stop.
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "d")
            && entry.path().is_file()
        {
            continue;
        }
        result.push(entry.path().display().to_string());
    }
    result.sort();
    Ok(result)
}

fn check_observation(fixture: &Fixture, observed: &Observation) -> Result<(), SourceFailure> {
    runtime_domains::check_observation(observed)?;
    let actual: BTreeSet<_> = observed.roots.iter().collect();
    let expected: BTreeSet<_> = fixture.compiler_input.kernel_symbols.iter().collect();
    if actual != expected
        || actual.len() != observed.roots.len()
        || observed.policy != 4
        || observed.output_digest == [0; 32]
        || observed.llvm_bytes == 0
        || observed.descriptor_roots != expected.len()
        || observed.missing_proof_refused
        || observed.formal_accesses != observed.global_reads + observed.global_writes
        || observed.reads != observed.global_reads + observed.private_reads + observed.other_reads
        || observed.writes
            != observed.global_writes + observed.private_writes + observed.other_writes
        || observed.other_reads != 0
        || observed.other_writes != 0
    {
        return Err(fail(
            SourceStage::Observation,
            "complete roots/P4/actual-O/native/formal access observations disagree",
        ));
    }
    Ok(())
}

fn run_case(workspace: &Path, fixture: &Fixture, case: &Path, target: &Path) -> CaseReport {
    run_case_with_simulation(workspace, fixture, case, target, None)
}

fn run_case_with_simulation(
    workspace: &Path,
    fixture: &Fixture,
    case: &Path,
    target: &Path,
    simulation_case: Option<simulation::Case>,
) -> CaseReport {
    let started = std::time::Instant::now();
    let prepared = (|| {
        check_input_files(workspace, fixture)?;
        std::fs::create_dir_all(case).map_err(|e| fail(SourceStage::Invocation, e))?;
        let census = source_census::Pending::prepare(workspace, fixture, case)?;
        let captured = corpus_cargo::capture(workspace, fixture, case, target)?;
        Ok((captured, census))
    })();
    let (captured, census) = match prepared {
        Ok(captured) => captured,
        Err(error) => {
            let mut report = CaseReport::blocked(fixture, error);
            report.case_elapsed_millis = progress::elapsed_millis(started);
            return report;
        }
    };
    let mut report =
        CaseReport::blocked(fixture, fail(SourceStage::Invocation, "child not invoked"));
    report.actual_cfg = captured.cfg;
    report.cargo_diagnostics = captured.cargo_diagnostics;
    let callback_progress = case.join("callback-progress.json");
    let run = (|| {
        let request = case.join("callback-args.json");
        let response = case.join("callback-result.json");
        std::fs::write(
            &request,
            serde_json::to_vec(&captured.args).map_err(|e| fail(SourceStage::Invocation, e))?,
        )
        .map_err(|e| fail(SourceStage::Invocation, e))?;
        let mut command =
            Command::new(env::current_exe().map_err(|e| fail(SourceStage::Invocation, e))?);
        command
            .env_clear()
            .envs(captured.environment)
            .current_dir(&captured.cwd)
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .env_remove(CHILD_PROOF_PROBE)
            .env(CHILD_ARGS, request)
            .env(CHILD_RESULT, &response)
            .env(progress::CHILD_PROGRESS, &callback_progress)
            .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"]);
        progress::clear_inherited_jobserver(&mut command);
        const CUSTODY_TRACE: &str = "FE2O3_TRACE_RANKED_CUSTODY_V1";
        if let Some(value) = env::var_os(CUSTODY_TRACE) {
            command.env(CUSTODY_TRACE, value);
        }
        simulation::configure_child(&mut command, simulation_case);
        census.configure(&mut command);
        snapshots::configure_child(&mut command, &fixture.fixture_id);
        eprintln!(
            "P4 CORPUS {} callback-progress={}",
            fixture.fixture_id,
            callback_progress.display()
        );
        let output = command.output().map_err(|e| fail(SourceStage::Rustc, e))?;
        report.rustc_diagnostics = corpus_cargo::diagnostics(&output);
        match census.finish(workspace, fixture, &captured.args, &captured.cwd) {
            Ok(observation) => report.source_census = Some(observation),
            Err(error) => {
                report.source_census_error = Some(error.detail.clone());
                return Err(error);
            }
        }
        let bytes = std::fs::read(response).map_err(|e| {
            fail(
                SourceStage::Rustc,
                format!(
                    "no structured callback response ({e}); {}",
                    report.rustc_diagnostics
                ),
            )
        })?;
        let result: Result<Observation, SourceFailure> =
            serde_json::from_slice(&bytes).map_err(|e| fail(SourceStage::Rustc, e))?;
        match result {
            Ok(observed) => {
                if !output.status.success() {
                    return Err(fail(
                        SourceStage::Rustc,
                        "successful callback response but failed compiler subprocess",
                    ));
                }
                check_observation(fixture, &observed)?;
                if let Some(case) = simulation_case {
                    simulation::check_observation(&observed, case)?;
                } else if observed.simulation.is_some() {
                    return Err(fail(
                        SourceStage::Simulation,
                        "unrequested simulation observation",
                    ));
                }
                Ok(observed)
            }
            Err(error) => Err(error),
        }
    })();
    match run {
        Ok(observed) => {
            report.observation = Some(observed);
            report.refusal = None;
        }
        Err(error) => report.refusal = Some(error),
    }
    match std::fs::read(&callback_progress)
        .map_err(|e| e.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|e| e.to_string()))
    {
        Ok(snapshot) => report.callback_progress = Some(snapshot),
        Err(error) => report.callback_progress_error = Some(error),
    }
    match artifacts(&case.join("compiler-output")) {
        Ok(paths) => {
            report.compiler_artifacts = paths;
            if !report.compiler_artifacts.is_empty() && report.refusal.is_none() {
                report.refusal = Some(fail(
                    SourceStage::Observation,
                    "callback unexpectedly emitted compiler artifacts",
                ));
            }
        }
        Err(error) => {
            if report.refusal.is_none() {
                report.refusal = Some(error);
            }
        }
    }
    if report.observation.is_some()
        && report.refusal.is_none()
        && report.compiler_artifacts.is_empty()
    {
        report.status = "checked-output-pass";
    }
    report.case_elapsed_millis = progress::elapsed_millis(started);
    report
}

#[test]
#[ignore = "full 95-case real Cargo/rustc P4 extraction gate; blocked cases fail, requires pinned rust-src and dependencies"]
fn ordinary_tutorial_corpus_requires_every_checked_policy4_output() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let manifest = std::fs::read(workspace.join(MANIFEST)).unwrap();
    let fixtures = fixtures(&manifest).expect("complete fixed-P4 tutorial source contract");
    let scratch = crate::test_temp_dir::TestTempDir::create("fe2o3-policy4-corpus");
    let mut cases = Vec::with_capacity(fixtures.len());
    for fixture in &fixtures {
        eprintln!(
            "P4 CORPUS {} target={} status=RUNNING",
            fixture.fixture_id, fixture.target
        );
        let case = scratch.path().join(&fixture.fixture_id);
        let target = scratch.path().join("dependencies").join(&fixture.target);
        let report = run_case(&workspace, fixture, &case, &target);
        eprintln!(
            "P4 CORPUS {} status={} refusal={:?}",
            fixture.fixture_id, report.status, report.refusal
        );
        cases.push(report);
    }
    let all_checked_output_passed =
        cases.len() == CONFIGURATIONS && cases.iter().all(CaseReport::passed);
    let report = CorpusReport {
        schema: "fe2o3-ordinary-source-policy4-extraction-corpus-v1",
        manifest_sha256: digest(&manifest),
        configurations: cases.len(),
        distinct_expected_roots: ROOTS,
        all_checked_output_passed,
        default_pipeline_activated: false,
        grants_artifact_or_launch_authority: false,
        cases,
    };
    let encoded = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(path) = env::var_os(REPORT) {
        std::fs::write(path, &encoded).expect("write explicitly requested corpus report");
    }
    eprintln!("P4 CORPUS REPORT\n{}", String::from_utf8(encoded).unwrap());
    assert!(
        all_checked_output_passed,
        "not qualified: at least one of all 95 ordinary-source configurations is blocked; see exact stage refusals"
    );
}

#[test]
#[ignore = "focused ordinary-source arithmetic regression; requires pinned rust-src and dependencies, not full-corpus qualification"]
fn ordinary_scalar_gemm_requires_checked_policy4_output() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let manifest = std::fs::read(workspace.join(MANIFEST)).unwrap();
    let fixtures = fixtures(&manifest).expect("complete fixed-P4 tutorial source contract");
    let fixture = fixtures
        .iter()
        .find(|fixture| fixture.fixture_id == "gfx942-scalar-gemm")
        .expect("the declared ordinary-source scalar arithmetic regression");
    let scratch = crate::test_temp_dir::TestTempDir::create("fe2o3-policy4-scalar-gemm");
    let report = run_case_with_simulation(
        &workspace,
        fixture,
        &scratch.path().join("case"),
        &scratch.path().join("dependencies"),
        Some(simulation::Case::ScalarGemm),
    );
    eprintln!(
        "P4 FOCUSED REPORT\n{}",
        serde_json::to_string_pretty(&report).unwrap()
    );
    assert!(report.passed(), "ordinary scalar GEMM remains unqualified");
    let runtime = report
        .observation
        .as_ref()
        .unwrap()
        .runtime_domains
        .as_ref()
        .expect("actual-O runtime-domain observation");
    assert!(runtime.reads >= 1, "{runtime:?}");
    assert_eq!(runtime.writes, 0);
    assert!(runtime.kernels >= 1);
    assert_eq!(runtime.v4_policy3_receipts, runtime.kernels);
}

#[test]
#[ignore = "focused ordinary-source P4 diagnostic; requires pinned rust-src and dependencies, not full-corpus qualification"]
fn ordinary_scalar_attention_requires_checked_policy4_output() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let manifest = std::fs::read(workspace.join(MANIFEST)).unwrap();
    let fixtures = fixtures(&manifest).expect("complete fixed-P4 source contract");
    let fixture = fixtures
        .iter()
        .find(|fixture| fixture.fixture_id == "gfx950-gpt-oss-scalar-attention")
        .expect("declared scalar-attention fixture");
    assert_eq!(fixture.target, "gfx950");
    let scratch = crate::test_temp_dir::TestTempDir::create("fe2o3-policy4-scalar-attention");
    let report = run_case(
        &workspace,
        fixture,
        &scratch.path().join("case"),
        &scratch.path().join("dependencies"),
    );
    eprintln!(
        "P4 FOCUSED REPORT\n{}",
        serde_json::to_string_pretty(&report).unwrap()
    );
    assert!(
        report.passed(),
        "ordinary scalar attention remains unqualified"
    );
}

#[test]
fn manifest_retains_all_feature_configurations_and_distinct_roots() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../",
        "config/tutorial-kernel-manifest-v1.json"
    ));
    let fixtures = fixtures(bytes).unwrap();
    assert_eq!(fixtures.len(), CONFIGURATIONS);
    assert_eq!(
        fixtures
            .iter()
            .flat_map(|f| &f.compiler_input.kernel_symbols)
            .collect::<BTreeSet<_>>()
            .len(),
        34
    );
    assert!(
        fixtures
            .windows(2)
            .any(|rows| rows[0].compiler_input.kernel_symbols
                == rows[1].compiler_input.kernel_symbols
                && rows[0].compiler_input.features != rows[1].compiler_input.features)
    );
    for (id, feature, source) in [
        (
            "gfx950-gpt-oss-pipelined-attention",
            "kernel-gpt-oss-decode-pipelined-attention",
            "examples/gfx950_gpt_oss_decode/src/kernel_pipelined_attention.rs",
        ),
        (
            "gfx950-gpt-oss-scalar-attention",
            "kernel-gpt-oss-decode-scalar-attention",
            "examples/gfx950_gpt_oss_decode/src/kernel_scalar_attention.rs",
        ),
    ] {
        let fixture = fixtures.iter().find(|row| row.fixture_id == id).unwrap();
        assert_eq!(fixture.target, "gfx950");
        assert!(!fixture.compiler_input.default_features);
        assert_eq!(fixture.compiler_input.features, [feature]);
        assert_eq!(fixture.compiler_input.source_paths, [source]);
        assert_eq!(
            fixture.compiler_input.kernel_symbols,
            ["gfx950_gpt_oss_120b_decode_megakernel_v1"]
        );
    }
    let mut modified: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    modified["compilerFixtures"].as_array_mut().unwrap().pop();
    assert!(super::corpus::fixtures(&serde_json::to_vec(&modified).unwrap()).is_err());
    let mut modified: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    modified["productionContract"]["allowsFallback"] = true.into();
    assert!(super::corpus::fixtures(&serde_json::to_vec(&modified).unwrap()).is_err());
}

#[test]
fn manifest_refuses_lost_original_or_counterpart_obligations_at_the_same_count() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/tutorial-kernel-manifest-v1.json"
    ));
    // Exercise both the retained prefix and new counterparts, preserving
    // counts, roots, targets, features and all valid input hashes.
    for index in [0, 49, 50, 94] {
        let mut modified: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        modified["compilerFixtures"][index]["fixtureId"] = "unreviewed-replacement".into();
        assert!(fixtures(&serde_json::to_vec(&modified).unwrap()).is_err());
    }
    let mut modified: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    modified["compilerFixtures"][50]["target"] = "gfx942".into();
    assert!(fixtures(&serde_json::to_vec(&modified).unwrap()).is_err());
}

#[test]
fn private_memory_is_not_counted_as_global_formal_evidence() {
    let fixture = fixtures(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/tutorial-kernel-manifest-v1.json"
    )))
    .unwrap()
    .remove(0);
    let mut observed = Observation {
        source_route: None,
        roots: fixture.compiler_input.kernel_symbols.clone(),
        transparent_result_wrappers: None,
        internal_helpers: 0,
        helper_calls: 0,
        reads: 3,
        writes: 4,
        global_reads: 1,
        global_writes: 1,
        private_reads: 2,
        private_writes: 3,
        other_reads: 0,
        other_writes: 0,
        formal_accesses: 2,
        runtime_domains: Some(runtime_domains::RuntimeDomainObservation::default()),
        simulation: None,
        constant_shift: None,
        masked_shift: None,
        policy: 4,
        output_digest: [1; 32],
        llvm_bytes: 100,
        descriptor_roots: 1,
        missing_proof_refused: false,
    };
    check_observation(&fixture, &observed).unwrap();
    observed.formal_accesses = observed.reads + observed.writes;
    assert!(check_observation(&fixture, &observed).is_err());
    let blocked = CaseReport::blocked(&fixture, fail(SourceStage::Policy4, "exact refusal"));
    assert!(!blocked.passed());
    assert_eq!(blocked.status, "blocked");
    assert!(blocked.observation.is_none());
}

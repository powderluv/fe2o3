//! Actual safe wrapper source import and checked general verification.
//! Historical refusals remain explicit; ranked checks grant no artifact or launch authority.

use super::*;
use crate::rustc_semantic_adapter_v1::canonical_function_identities_v1;
use crate::trusted_device_items::{self, TrustedDeviceItem, Wave64ShuffleScalarV1};
use fe2o3_mir_model::semantic_mir_v1::{
    AdmittedInertSemanticMirV1, SemanticCallableDeclV1, SemanticCompilerIntrinsicOperationV1,
    SemanticMirLimitsV1, SemanticMirWireVersionV1, SemanticTerminatorKindV1,
};
use rustc_middle::ty::{self, Instance, TypingEnv};

const CHILD: &str = "production_rustc_driver_v1::checked_output_source_v1_tests::wave64_capture_source::wave64_capture_source_child";
const CASE_ENV: &str = "FE2O3_TEST_WAVE64_CAPTURE_CASE_V1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum Scalar {
    U32,
    I32,
    F32,
}
impl Scalar {
    fn provider(self) -> Wave64ShuffleScalarV1 {
        match self {
            Self::U32 => Wave64ShuffleScalarV1::U32,
            Self::I32 => Wave64ShuffleScalarV1::I32,
            Self::F32 => Wave64ShuffleScalarV1::F32,
        }
    }
    fn feature(self) -> &'static str {
        match self {
            Self::U32 => "wave64-capture-u32",
            Self::I32 => "wave64-capture-i32",
            Self::F32 => "wave64-capture-f32",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum Expectation {
    InertCapture,
    UserUnsafe,
    UserUnsafeScan,
    WrongWavefront,
    XnackPlus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum Operation {
    Reduce,
    Inclusive,
    Exclusive,
}
impl Operation {
    fn wrapper(self) -> TrustedDeviceItem {
        match self {
            Self::Reduce => TrustedDeviceItem::Gfx942Wave64ReduceSum,
            Self::Inclusive => TrustedDeviceItem::Gfx942Wave64InclusiveScanSum,
            Self::Exclusive => TrustedDeviceItem::Gfx942Wave64ExclusiveScanSum,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct Case {
    gfx950: bool,
    scalar: Scalar,
    operation: Operation,
    retained: bool,
    expectation: Expectation,
}
impl Case {
    fn permits_ranked_only(self) -> bool {
        self.expectation == Expectation::InertCapture
            && self.scalar == Scalar::F32
            && matches!(
                self.operation,
                Operation::Reduce | Operation::Inclusive | Operation::Exclusive
            )
    }

    fn requires_ranked_only(self) -> bool {
        // Only these exact non-retained gfx942 rows have been observed reaching
        // ranked verification. Every other permitted row remains conditional.
        self.permits_ranked_only()
            && !self.gfx950
            && matches!(self.operation, Operation::Reduce | Operation::Inclusive)
            && !self.retained
    }

    fn hostile_features(self) -> Option<&'static str> {
        match self.expectation {
            Expectation::WrongWavefront => Some("+wavefrontsize32,-wavefrontsize64,-xnack"),
            Expectation::XnackPlus => Some("-wavefrontsize32,+wavefrontsize64,+xnack"),
            _ => None,
        }
    }

    fn target(self) -> &'static str {
        if self.gfx950 { "gfx950" } else { "gfx942" }
    }
    fn feature(self) -> &'static str {
        match self.expectation {
            Expectation::UserUnsafe => "wave64-capture-direct-unsafe",
            Expectation::UserUnsafeScan => "wave64-capture-forged-scan",
            _ => match self.operation {
                Operation::Reduce => self.scalar.feature(),
                Operation::Inclusive => "wave64-capture-inclusive",
                Operation::Exclusive => "wave64-capture-exclusive",
            },
        }
    }
    fn features(self) -> Vec<String> {
        let mut features = vec![self.scalar.feature().into()];
        if self.feature() != self.scalar.feature() {
            features.push(self.feature().into());
        }
        features
    }
}

// General ranked checks carry no artifact or launch authority. Keep this
// outcome separate from the historical materialization/projection refusals.
#[derive(Debug, Serialize, Deserialize)]
struct RankedOnlyReport {
    roots: usize,
    functions: usize,
    callables: usize,
    retained_bindings: usize,
    bounds_clean: bool,
    checks_clean: bool,
    grants_artifact_or_launch_authority: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Report {
    case: Case,
    primitive_calls: usize,
    retained_wrapper_functions: usize,
    retained_scan_helpers: usize,
    actual_can_unwind: Option<bool>,
    semantic_bytes: usize,
    refusal: String,
    ranked_only: Option<RankedOnlyReport>,
}
impl Report {
    fn validate(&self, case: Case) -> Result<(), String> {
        if self.case != case || self.refusal.is_empty() != self.ranked_only.is_some() {
            return Err("wave capture report lost its exact case or distinct outcome".into());
        }
        if case.requires_ranked_only() && self.ranked_only.is_none() {
            return Err(
                "observed gfx942 non-retained f32 ranked witness regressed to refusal".into(),
            );
        }
        if let Some(ranked) = &self.ranked_only {
            if !case.permits_ranked_only()
                || ranked.roots != 1
                || ranked.functions == 0
                || ranked.callables == 0
                || ranked.retained_bindings != 6
                || !ranked.bounds_clean
                || !ranked.checks_clean
                || ranked.grants_artifact_or_launch_authority
            {
                return Err(
                    "actual ranked-only outcome changed checks, custody or authority".into(),
                );
            }
        }
        if case.expectation == Expectation::InertCapture {
            if self.primitive_calls == 0
                || self.actual_can_unwind.is_none()
                || self.semantic_bytes == 0
                || (case.retained && self.retained_wrapper_functions != 1)
                || (case.retained
                    && case.operation != Operation::Reduce
                    && self.retained_scan_helpers != 1)
                || (case.operation == Operation::Reduce && self.retained_scan_helpers != 0)
            {
                return Err("source report lacks actual inert primitive/body/ABI evidence".into());
            }
        } else if self.primitive_calls != 0
            || self.retained_wrapper_functions != 0
            || self.retained_scan_helpers != 0
            || self.actual_can_unwind.is_some()
            || self.semantic_bytes != 0
        {
            return Err("source safety refusal was substituted with descriptor evidence".into());
        }
        Ok(())
    }
}

fn observe_imported(
    tcx: TyCtxt<'_>,
    case: Case,
    semantic: &AdmittedInertSemanticMirV1,
) -> Result<Report, String> {
    if semantic.wire_version() != SemanticMirWireVersionV1::V33 || semantic.roots().len() != 1 {
        return Err("actual ordinary wrapper did not import exact V33 with one root".into());
    }
    let scalar = case.scalar.provider();
    let primitive =
        trusted_device_items::definition(tcx, TrustedDeviceItem::Gfx942Wave64Shuffle(scalar))
            .ok_or("exact primitive DefId is unavailable")?;
    trusted_device_items::check_actual_sealed_trait_chain_paths_v1(tcx, primitive, scalar);
    trusted_device_items::check_actual_scan_instances_v1(tcx);
    let instance = Instance::mono(tcx, primitive);
    let identity = canonical_function_identities_v1(tcx, instance).function();
    let actual_fn_abi = tcx
        .fn_abi_of_instance(
            TypingEnv::fully_monomorphized().as_query_input((instance, ty::List::empty())),
        )
        .map_err(|error| format!("actual FnAbi: {error:?}"))?;
    let mut primitive_ids = Vec::new();
    for (index, callable) in semantic.callables().iter().enumerate() {
        if let SemanticCallableDeclV1::CompilerIntrinsic {
            binding,
            operation: SemanticCompilerIntrinsicOperationV1::Gfx942Wave64ShuffleIndex { .. },
            ..
        } = callable
        {
            if binding.identity() != identity
                || binding.abi().can_unwind() != actual_fn_abi.can_unwind
            {
                return Err(
                    "primitive descriptor substituted the actual instance or unwind ABI".into(),
                );
            }
            crate::collector::check_wave64_descriptor_mutations_v1(
                tcx,
                instance,
                scalar,
                binding.abi(),
                semantic.types(),
            );
            primitive_ids.push(index);
        }
    }
    if primitive_ids.len() != 1 {
        return Err(format!(
            "expected one exact primitive declaration, got {}",
            primitive_ids.len()
        ));
    }
    let primitive_calls = semantic.functions().iter().flat_map(|function| function.blocks())
        .filter(|block| matches!(block.terminator().kind(), SemanticTerminatorKindV1::Call(call)
            if primitive_ids.contains(&(call.callee().index() as usize)) && call.arguments().len() == 3))
        .count();
    let wrapper = trusted_device_items::definition(tcx, case.operation.wrapper())
        .ok_or("reviewed safe collective wrapper unavailable")?;
    let wrapper_identity = canonical_function_identities_v1(
        tcx,
        Instance::new_raw(wrapper, ty::GenericArgs::identity_for_item(tcx, wrapper)),
    )
    .item_definition();
    let retained_wrapper_functions = semantic
        .functions()
        .iter()
        .filter(|function| function.item_definition_identity() == wrapper_identity)
        .count();
    let helper =
        trusted_device_items::definition(tcx, TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper)
            .ok_or("reviewed scan helper unavailable")?;
    let helper_identity = canonical_function_identities_v1(
        tcx,
        Instance::new_raw(helper, ty::GenericArgs::identity_for_item(tcx, helper)),
    )
    .item_definition();
    let retained_scan_helpers = semantic
        .functions()
        .iter()
        .filter(|function| function.item_definition_identity() == helper_identity)
        .count();
    let decoded = AdmittedInertSemanticMirV1::decode_exact_v33_canonical(
        semantic.canonical_encoding(),
        SemanticMirLimitsV1::default(),
    )
    .map_err(|error| format!("actual V33 decode: {error:?}"))?;
    if decoded.canonical_encoding() != semantic.canonical_encoding()
        || decoded.functions() != semantic.functions()
        || decoded.callables() != semantic.callables()
    {
        return Err("actual source V33 roundtrip changed body or primitive binding".into());
    }
    Ok(Report {
        case,
        primitive_calls,
        retained_wrapper_functions,
        retained_scan_helpers,
        actual_can_unwind: Some(actual_fn_abi.can_unwind),
        semantic_bytes: semantic.canonical_encoding().len(),
        refusal: String::new(),
        ranked_only: None,
    })
}

fn observe_capture(tcx: TyCtxt<'_>, case: Case) -> Result<Report, String> {
    use crate::collector::semantic_import_observation_v1_tests::with_observer;
    use crate::production_pipeline::ProductionPipelineError as Pipeline;
    use std::cell::RefCell;
    use std::rc::Rc;
    let transaction = transaction_in_active_session_v1(
        tcx,
        crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2::Disabled,
    )?;
    let observed = Rc::new(RefCell::new((0_usize, None)));
    let callback_state = Rc::clone(&observed);
    let result = with_observer(
        Box::new(move |tcx, semantic| {
            let mut state = callback_state.borrow_mut();
            state.0 += 1;
            state.1 = Some(observe_imported(tcx, case, semantic));
        }),
        || transaction.verify_general_kernel_checks(),
    );
    let (count, report) = Rc::try_unwrap(observed)
        .expect("observer was removed")
        .into_inner();
    if count != 1 {
        return Err(format!(
            "expected one actual admitted-owner observation, got {count}; result: {:?}",
            result.err()
        ));
    }
    let mut report = report.ok_or("missing actual source observation")??;
    report.refusal = match result {
        Err(error @ (Pipeline::PreRankedMaterialization(_) | Pipeline::RankedProjection(_))) => {
            format!("{error:?}")
        }
        Err(error) => return Err(format!("unexpected pre-materialization refusal: {error:?}")),
        Ok(ranked) => {
            if !case.permits_ranked_only() {
                return Err(
                    "source outside the reviewed f32 collectives reached ranked checks".into(),
                );
            }
            // Read the actual returned stage; no target lowering, artifact
            // publication or launch is performed by this source-capture test.
            report.ranked_only = Some(RankedOnlyReport {
                roots: ranked.ranked_root_count(),
                functions: ranked.semantic_function_count(),
                callables: ranked.semantic_callable_count(),
                retained_bindings: ranked.retained_identity_and_transaction_binding_count(),
                bounds_clean: ranked.bounds_are_clean(),
                checks_clean: ranked.all_kernel_checks_are_clean(),
                grants_artifact_or_launch_authority: ranked.grants_artifact_or_launch_authority(),
            });
            report.validate(case)?;
            return Ok(report);
        }
    };
    report.validate(case)?;
    Ok(report)
}

fn observe_refusal(tcx: TyCtxt<'_>, case: Case) -> Result<Report, String> {
    use crate::collector::semantic_import_observation_v1_tests::with_observer;
    let observations = std::rc::Rc::new(std::cell::Cell::new(0_usize));
    let callback_observations = std::rc::Rc::clone(&observations);
    let result = with_observer(
        Box::new(move |_, _| callback_observations.set(callback_observations.get() + 1)),
        || {
            transaction_in_active_session_v1(
                tcx,
                crate::rustc_semantic_plan_v1::DebugSourceCaptureRequestV2::Disabled,
            )
        },
    );
    if observations.get() != 0 {
        return Err("rejected source/profile reached the semantic import observer".into());
    }
    let refusal = match result {
        Err(error) => error,
        Ok(_) => {
            return Err("ordinary source acquired an unauthorized unsafe/target closure".into());
        }
    };
    let expected = match case.expectation {
        Expectation::UserUnsafe => refusal.contains("user-provided unsafe block"),
        Expectation::UserUnsafeScan => {
            refusal.contains("reaches unsafe function instance") && refusal.contains("forged_scan")
        }
        Expectation::WrongWavefront | Expectation::XnackPlus => {
            refusal.contains("production extraction target authentication failed before monomorphization: production compilation requires authoritative rustc active target features")
                && refusal.contains("-wavefrontsize32,+wavefrontsize64,-xnack")
                && refusal.contains(&format!("found {:?}", case.hostile_features().unwrap()))
        }
        Expectation::InertCapture => false,
    };
    if !expected {
        return Err(format!(
            "wrong source boundary for {:?}: {refusal}",
            case.expectation
        ));
    }
    let report = Report {
        case,
        primitive_calls: 0,
        retained_wrapper_functions: 0,
        retained_scan_helpers: 0,
        actual_can_unwind: None,
        semantic_bytes: 0,
        refusal,
        ranked_only: None,
    };
    report.validate(case)?;
    Ok(report)
}

struct CaptureCallbacks {
    case: Case,
    result: Option<Result<Report, String>>,
}
impl Callbacks for CaptureCallbacks {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        self.result = Some(if self.case.expectation == Expectation::InertCapture {
            observe_capture(tcx, self.case)
        } else {
            observe_refusal(tcx, self.case)
        });
        Compilation::Stop
    }
}

#[test]
#[ignore = "subprocess helper; parent supplies exact Cargo capture and source boundary"]
fn wave64_capture_source_child() {
    let Some(path) = env::var_os(CHILD_ARGS) else {
        return;
    };
    let args: Vec<String> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let case: Case = serde_json::from_str(&env::var(CASE_ENV).unwrap()).unwrap();
    let mut callbacks = CaptureCallbacks { case, result: None };
    let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rustc_driver::run_compiler(&args, &mut callbacks);
    }));
    let result = if completed.is_err() {
        Err("rustc or wave capture callback panicked".to_owned())
    } else {
        callbacks
            .result
            .unwrap_or_else(|| Err("source callback did not execute".into()))
    };
    std::fs::write(
        env::var_os(CHILD_RESULT).unwrap(),
        serde_json::to_vec(&result).unwrap(),
    )
    .unwrap();
    assert!(result.is_ok(), "actual wave source capture: {result:?}");
}

fn fixture(workspace: &Path, case: Case) -> corpus::Fixture {
    let base = "crates/rustc-codegen-fe2o3/tests/fixtures/production-extraction-device";
    let manifest = format!("{base}/Cargo.toml");
    let hash = |path: &str| {
        Sha256::digest(std::fs::read(workspace.join(path)).unwrap())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    corpus::Fixture {
        fixture_id: format!(
            "{}-{}-{}-retained-{}-{:?}",
            case.target(),
            case.feature(),
            case.scalar.feature(),
            case.retained,
            case.expectation
        ),
        target: case.target().into(),
        compiler_input: corpus::CompilerInput {
            package_manifest_sha256: hash(&manifest),
            package_manifest: manifest,
            cargo_lock_path: "Cargo.lock".into(),
            cargo_lock_sha256: hash("Cargo.lock"),
            source_paths: vec![
                format!("{base}/src/lib.rs"),
                format!("{base}/src/wave64_capture.rs"),
            ],
            source_closure_sha256: String::new(),
            cargo_target: corpus::CargoTarget {
                kind: "lib".into(),
                name: "fe2o3_production_extraction_fixture".into(),
                source_path: "src/lib.rs".into(),
            },
            default_features: false,
            features: case.features(),
            kernel_symbols: vec!["wave64_capture".into()],
        },
    }
}

fn capture_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for gfx950 in [false, true] {
        for operation in [
            Operation::Reduce,
            Operation::Inclusive,
            Operation::Exclusive,
        ] {
            for scalar in [Scalar::U32, Scalar::I32, Scalar::F32] {
                for retained in [false, true] {
                    cases.push(Case {
                        gfx950,
                        scalar,
                        operation,
                        retained,
                        expectation: Expectation::InertCapture,
                    });
                }
            }
        }
        for expectation in [
            Expectation::UserUnsafe,
            Expectation::UserUnsafeScan,
            Expectation::WrongWavefront,
            Expectation::XnackPlus,
        ] {
            for retained in [false, true] {
                cases.push(Case {
                    gfx950,
                    scalar: Scalar::U32,
                    operation: Operation::Reduce,
                    retained,
                    expectation,
                });
            }
        }
    }
    cases
}

// These controls validate only the report contract. Actual source observation
// and the returned ranked stage are still required by the subprocess campaign.
fn ranked_report_control() -> Report {
    Report {
        case: Case {
            gfx950: false,
            scalar: Scalar::F32,
            operation: Operation::Inclusive,
            retained: false,
            expectation: Expectation::InertCapture,
        },
        primitive_calls: 1,
        retained_wrapper_functions: 0,
        retained_scan_helpers: 1,
        actual_can_unwind: Some(false),
        semantic_bytes: 1,
        refusal: String::new(),
        ranked_only: Some(RankedOnlyReport {
            roots: 1,
            functions: 2,
            callables: 3,
            retained_bindings: 6,
            bounds_clean: true,
            checks_clean: true,
            grants_artifact_or_launch_authority: false,
        }),
    }
}

#[test]
fn ranked_only_report_rejects_authority_dirty_checks_counts_and_mixed_outcomes() {
    let report = ranked_report_control();
    report.validate(report.case).unwrap();
    for fault in 0..10 {
        let mut report = ranked_report_control();
        let ranked = report.ranked_only.as_mut().unwrap();
        match fault {
            0 => ranked.grants_artifact_or_launch_authority = true,
            1 => ranked.bounds_clean = false,
            2 => ranked.checks_clean = false,
            3 => ranked.roots = 0,
            4 => ranked.functions = 0,
            5 => ranked.callables = 0,
            6 => ranked.retained_bindings = 7,
            7 => report.refusal = "simultaneous refusal is invalid".into(),
            8 => report.case.expectation = Expectation::UserUnsafe,
            9 => report.case.scalar = Scalar::U32,
            _ => unreachable!(),
        }
        assert!(
            report.validate(report.case).is_err(),
            "report fault {fault} accepted"
        );
    }
}

#[test]
fn observed_ranked_witness_cannot_regress_to_refusal_without_requiring_gfx950_success() {
    let mut report = ranked_report_control();
    report.ranked_only = None;
    report.refusal = "historical pre-ranked refusal control".into();
    assert!(report.validate(report.case).is_err());
    report.case.gfx950 = true;
    report.validate(report.case).unwrap();
    report.case.gfx950 = false;
    report.case.operation = Operation::Exclusive;
    report.validate(report.case).unwrap();
}

#[test]
fn observed_gfx942_nonretained_f32_reduction_keeps_actual_ranked_stage_requirement() {
    // Report-contract control only. The actual source subprocess must still
    // return and validate its real ranked stage; this is no native receipt.
    let mut report = ranked_report_control();
    report.case.operation = Operation::Reduce;
    report.retained_scan_helpers = 0;
    assert!(report.case.permits_ranked_only());
    assert!(report.case.requires_ranked_only());
    report.validate(report.case).unwrap();
    report
        .ranked_only
        .as_mut()
        .unwrap()
        .grants_artifact_or_launch_authority = true;
    assert!(report.validate(report.case).is_err());
    report
        .ranked_only
        .as_mut()
        .unwrap()
        .grants_artifact_or_launch_authority = false;
    for scalar in [Scalar::U32, Scalar::I32] {
        report.case.scalar = scalar;
        assert!(!report.case.permits_ranked_only());
        assert!(report.validate(report.case).is_err());
    }
    report.case.scalar = Scalar::F32;
    report.ranked_only = None;
    report.refusal = "historical pre-ranked refusal control".into();
    assert!(report.validate(report.case).is_err());
    // No unobserved gfx950 or retained reduction is made mandatory.
    for (gfx950, retained) in [(true, false), (false, true), (true, true)] {
        report.case.gfx950 = gfx950;
        report.case.retained = retained;
        report.retained_wrapper_functions = usize::from(retained);
        assert!(report.case.permits_ranked_only());
        assert!(!report.case.requires_ranked_only());
        report.validate(report.case).unwrap();
    }
}

#[test]
fn wave64_capture_cases_keep_scalar_scan_and_hostile_boundaries_distinct() {
    let cases = capture_cases();
    assert_eq!(cases.len(), 52);
    assert_eq!(
        cases
            .iter()
            .filter(|case| case.permits_ranked_only())
            .count(),
        12
    );
    assert_eq!(
        cases
            .iter()
            .filter(|case| case.requires_ranked_only())
            .count(),
        2
    );
    assert_eq!(
        cases
            .iter()
            .filter(|case| case.expectation != Expectation::InertCapture)
            .count(),
        16
    );
    let mut identities = std::collections::BTreeSet::new();
    for case in cases {
        assert!(identities.insert(serde_json::to_string(&case).unwrap()));
        assert!(case.features().contains(&case.scalar.feature().to_owned()));
        assert_eq!(case.target(), if case.gfx950 { "gfx950" } else { "gfx942" });
    }
}

fn run_capture_cases(cases: impl IntoIterator<Item = Case>) {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let scratch = crate::test_temp_dir::TestTempDir::create("fe2o3-wave64-capture");
    for case in cases {
        let fixture = fixture(&workspace, case);
        let directory = scratch.path().join(&fixture.fixture_id);
        std::fs::create_dir(&directory).unwrap();
        let target_dir = scratch.path().join(case.target());
        let mut captured =
            corpus_cargo::capture(&workspace, &fixture, &directory, &target_dir).unwrap();
        if let Some(features) = case.hostile_features() {
            // The validated dependency capture remains normal. Replace the one
            // actual root flag so the live rustc owner presents a hostile profile.
            let mut flags = captured
                .args
                .iter_mut()
                .filter(|arg| arg.starts_with("-Ctarget-feature="));
            *flags.next().expect("one captured target-feature argument") =
                format!("-Ctarget-feature={features}");
            assert!(
                flags.next().is_none(),
                "duplicate captured feature argument"
            );
        }
        if case.retained {
            captured.args.push("-Zinline-mir=no".into());
            captured.args.push("-Zmir-opt-level=0".into());
        }
        let request = directory.join("args.json");
        let response = directory.join("result.json");
        std::fs::write(&request, serde_json::to_vec(&captured.args).unwrap()).unwrap();
        let mut command = Command::new(env::current_exe().unwrap());
        command
            .env_clear()
            .envs(captured.environment)
            .current_dir(captured.cwd)
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .env_remove(CHILD_PROOF_PROBE)
            .env(CHILD_ARGS, &request)
            .env(CHILD_RESULT, &response)
            .env(CASE_ENV, serde_json::to_string(&case).unwrap())
            .args(["--exact", CHILD, "--ignored", "--nocapture"]);
        progress::clear_inherited_jobserver(&mut command);
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}: {}",
            fixture.fixture_id,
            corpus_cargo::diagnostics(&output)
        );
        let report: Result<Report, String> =
            serde_json::from_slice(&std::fs::read(response).unwrap()).unwrap();
        let report = report.unwrap();
        report.validate(case).unwrap();
        eprintln!(
            "WAVE64 SOURCE {}: {} primitive calls, {} retained wrappers, {} scan helpers; refusal={:?}; ranked_only={:?}",
            fixture.fixture_id,
            report.primitive_calls,
            report.retained_wrapper_functions,
            report.retained_scan_helpers,
            report.refusal,
            report.ranked_only
        );
    }
}

#[test]
#[ignore = "genuine Cargo/AMD rustc source capture; requires pinned nightly toolchain"]
fn safe_wave64_wrappers_capture_exact_primitives_without_execution_authority() {
    run_capture_cases(capture_cases());
}
#[test]
#[ignore = "genuine live AMD rustc target profiles; requires pinned nightly toolchain"]
fn live_invalid_wave64_profiles_refused_before_collection() {
    let cases: Vec<_> = capture_cases()
        .into_iter()
        .filter(|case| case.hostile_features().is_some())
        .collect();
    assert_eq!(cases.len(), 8);
    run_capture_cases(cases);
}

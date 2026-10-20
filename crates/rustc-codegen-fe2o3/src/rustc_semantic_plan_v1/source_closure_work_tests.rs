use super::*;
use crate::rustc_semantic_adapter_v1::canonical_target_layout_v1;
use crate::semantic_layout_bridge::rustc_semantic_layout_target_v1;
use crate::test_temp_dir::TestTempDir;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;

fn work_with_limit(maximum: u64) -> SourceClosureWorkV1 {
    SourceClosureWorkV1 {
        limits: SemanticMirLimitsV1::default()
            .with_limit(SemanticMirResourceV1::ValidationWork, maximum)
            .unwrap(),
        ..SourceClosureWorkV1::default()
    }
}

#[test]
fn source_closure_work_defaults_to_the_existing_raw_budget() {
    let mut work = SourceClosureWorkV1::default();
    let limits = SemanticMirLimitsV1::default();
    let mut ordinary = RawMirPreflightCountsV1::default();
    assert_eq!(work.limits, limits);
    assert_eq!(work.counts, ordinary);
    assert!(work.identity.is_none());

    for amount in [0, 2, 3] {
        work.charge(amount).unwrap();
        ordinary
            .charge(SemanticMirResourceV1::ValidationWork, amount, limits)
            .unwrap();
        assert_eq!(work.counts, ordinary);
    }
    assert_eq!(
        work.counts.digest_fields(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
    );
}

#[test]
fn source_closure_work_accepts_exact_and_rejects_one_short() {
    let mut exact = work_with_limit(5);
    exact.charge(2).unwrap();
    exact.charge(3).unwrap();
    exact.charge(0).unwrap();
    assert_eq!(exact.counts.validation_work, 5);

    let mut one_short = work_with_limit(4);
    one_short.charge(2).unwrap();
    assert!(matches!(
        one_short.charge(3),
        Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::ValidationWork,
            actual: 5,
            maximum: 4,
        })
    ));
    assert_eq!(one_short.counts.validation_work, 5);
}

#[test]
fn source_closure_work_reuses_checked_overflow_accounting() {
    let mut work = SourceClosureWorkV1::default();
    work.counts.validation_work = u64::MAX;
    let before = work.counts;
    assert!(matches!(
        work.charge(1),
        Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
            resource: SemanticMirResourceV1::ValidationWork,
            actual: u64::MAX,
            maximum,
        }) if maximum == work.limits.limit(SemanticMirResourceV1::ValidationWork)
    ));
    assert_eq!(work.counts, before);
}

#[test]
fn source_closure_identity_follows_moves_not_reconstructed_counters() {
    let mut original = SourceClosureWorkV1::default();
    original.charge(17).unwrap();
    let stamp = original.retain_identity_v1().unwrap();
    let mut moved = original;
    assert!(moved.matches_identity_v1(&stamp).unwrap());

    let mut replacement = SourceClosureWorkV1 {
        limits: moved.limits,
        counts: moved.counts,
        ..SourceClosureWorkV1::default()
    };
    assert!(!replacement.matches_identity_v1(&stamp).unwrap());
    let replacement_stamp = replacement.retain_identity_v1().unwrap();
    assert!(!moved.matches_identity_v1(&replacement_stamp).unwrap());
    assert!(!replacement.matches_identity_v1(&stamp).unwrap());
    assert!(moved.matches_identity_v1(&stamp).unwrap());
    assert!(replacement.matches_identity_v1(&replacement_stamp).unwrap());
}

#[test]
fn source_closure_identity_stamp_does_not_keep_a_dropped_account_alive() {
    let stamp = {
        let mut original = SourceClosureWorkV1::default();
        original.retain_identity_v1().unwrap()
    };
    assert!(stamp.identity.upgrade().is_none());
    let mut replacement = SourceClosureWorkV1::default();
    let _replacement_stamp = replacement.retain_identity_v1().unwrap();
    assert!(!replacement.matches_identity_v1(&stamp).unwrap());
}

#[test]
fn source_closure_identity_creation_is_paid_before_installation() {
    let inherited = 17;
    let mut measured = SourceClosureWorkV1::default();
    measured.charge(inherited).unwrap();
    let _stamp = measured.retain_identity_v1().unwrap();
    let total = measured.counts.validation_work;
    assert!(total > inherited as u64);

    let mut exact = work_with_limit(total);
    exact.charge(inherited).unwrap();
    let _stamp = exact.retain_identity_v1().unwrap();
    assert_eq!(exact.counts.validation_work, total);

    let mut short = work_with_limit(total - 1);
    short.charge(inherited).unwrap();
    assert!(short.retain_identity_v1().is_err());
    assert!(short.identity.is_none());
    assert_eq!(short.counts.validation_work, total);
    assert!(short.retain_identity_v1().is_err());
    assert!(short.identity.is_none());
    assert!(short.counts.validation_work > total);

    let mut zero = work_with_limit(0);
    assert!(zero.retain_identity_v1().is_err());
    assert!(zero.identity.is_none());
}

#[test]
fn source_closure_identity_rechecks_debit_the_original_account() {
    let mut work = SourceClosureWorkV1::default();
    let stamp = work.retain_identity_v1().unwrap();
    let paid = work.counts.validation_work;
    work.limits = work
        .limits
        .with_limit(SemanticMirResourceV1::ValidationWork, paid + 1)
        .unwrap();
    assert!(work.matches_identity_v1(&stamp).unwrap());
    assert_eq!(work.counts.validation_work, paid + 1);
    assert!(work.matches_identity_v1(&stamp).is_err());
    assert_eq!(work.counts.validation_work, paid + 2);
    assert!(work.matches_identity_v1(&stamp).is_err());
    assert_eq!(work.counts.validation_work, paid + 3);
    assert!(work.identity.is_some());
}

fn transcript_counts(plan: &ProductionSemanticPreflightPlanV1<'_>) -> [u64; 11] {
    let fields = transcript_fields_v5(plan.canonical_transcript());
    assert_eq!(fields[0], PREFLIGHT_PLAN_DOMAIN_V5);
    assert_eq!(fields[1], PREFLIGHT_PLAN_DOMAIN_V4);
    // Four identity fields and twelve cardinalities precede the raw counters.
    std::array::from_fn(|index| u64::from_le_bytes(fields[16 + index].try_into().unwrap()))
}

#[derive(Default)]
struct WorkCallbacks {
    completed: bool,
}

impl Callbacks for WorkCallbacks {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        let instance = Instance::mono(
            tcx,
            tcx.hir_body_owners()
                .filter(|id| tcx.def_kind(id.to_def_id()) == rustc_hir::def::DefKind::Fn)
                .find(|id| tcx.item_name(id.to_def_id()).as_str() == "root")
                .unwrap()
                .to_def_id(),
        );
        let functions = vec![RetainedSemanticFunctionProducerV1 {
            identities: canonical_function_identities_v1(tcx, instance),
            instance,
            role: CollectedFunctionRole::InternalHelper,
            export_name: None,
            kernel_binding: None,
            generated_host_contract_identity: None,
            frontend_contract: None,
        }]
        .into_boxed_slice();
        let roots = vec![SemanticFunctionIdV1::from_index(0)].into_boxed_slice();
        let target = canonical_target_layout_v1(&rustc_semantic_layout_target_v1(tcx).unwrap());
        let ordinary = build_production_semantic_preflight_plan_v1(
            tcx,
            target,
            functions.clone(),
            roots.clone(),
            [7; 32],
            DebugSourceCaptureRequestV2::Disabled,
            None,
        )
        .unwrap();
        let build = |work| {
            build_production_semantic_preflight_plan_with_work_v1(
                tcx,
                target,
                functions.clone(),
                roots.clone(),
                [7; 32],
                DebugSourceCaptureRequestV2::Disabled,
                (None, work),
            )
        };
        let default = build(SourceClosureWorkV1::default()).unwrap();
        assert_eq!(
            default.canonical_transcript(),
            ordinary.canonical_transcript()
        );
        assert_eq!(default.sha256, ordinary.sha256);
        let ordinary_counts = transcript_counts(&ordinary);
        let preflight_work = ordinary_counts[10];
        assert!(preflight_work > 0);

        let collection_work = 2;
        let reobservation_work = 3;
        let prior_work = u64::try_from(collection_work + reobservation_work).unwrap();
        let total = prior_work + preflight_work;
        let sequential_work = |maximum| {
            let mut work = work_with_limit(maximum);
            work.charge(collection_work).unwrap();
            work.charge(reobservation_work).unwrap();
            assert_eq!(work.counts.validation_work, prior_work);
            work
        };
        let exact = build(sequential_work(total)).unwrap();
        let mut expected_counts = ordinary_counts;
        expected_counts[10] = total;
        assert_eq!(transcript_counts(&exact), expected_counts);
        assert!(matches!(
            build(sequential_work(total - 1)),
            Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
                resource: SemanticMirResourceV1::ValidationWork,
                actual,
                maximum,
            }) if actual == total && maximum == total - 1
        ));
        assert!(matches!(
            build(sequential_work(prior_work)),
            Err(ProductionSemanticPreflightErrorV1::LimitExceeded {
                resource: SemanticMirResourceV1::ValidationWork,
                actual,
                maximum,
            }) if actual > prior_work && maximum == prior_work
        ));
        self.completed = true;
        Compilation::Stop
    }
}

#[test]
fn collection_and_reobservation_work_survives_into_semantic_preflight() {
    let directory = TestTempDir::create("fe2o3-source-closure-work");
    let source = directory.path().join("fixture.rs");
    std::fs::write(&source, "pub fn root(input: u32) -> u32 { input ^ 1 }").unwrap();
    let output = std::process::Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let sysroot = String::from_utf8(output.stdout).unwrap();
    let args = vec![
        "rustc".into(),
        "--crate-name=fe2o3_source_closure_work".into(),
        "--crate-type=lib".into(),
        "--edition=2024".into(),
        "--emit=metadata".into(),
        "-Zmir-opt-level=0".into(),
        "-Cpanic=abort".into(),
        "--sysroot".into(),
        sysroot.trim().into(),
        "-o".into(),
        directory.path().join("fixture.rmeta").display().to_string(),
        source.display().to_string(),
    ];
    let mut callbacks = WorkCallbacks::default();
    rustc_driver::run_compiler(&args, &mut callbacks);
    assert!(callbacks.completed);
}

//! Opt-in probes borrowing a genuine production policy-origin replay.
//! No request, retained proof, enrollment owner, issuer or runtime is invented.
//! Shared input consistency is not an enrollment issuer/currentness check.
use super::{Binding, Budget, Origin, Request, native_cpu_input_v1, native_cpu_policy_input_v2};
use fe2o3_verifier::portable_reference_v1::codec::{
    NativeCpuOriginV2, NativeCpuPolicyInputV2, ReferenceEnrollmentOriginV1,
};
use fe2o3_verifier::{
    ProductionConditionalFormulaErrorV2 as FormulaError,
    RetainedProductionConditionalFormulaV2 as Retained,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Mutex,
};

const SCHEMA: &str = "fe2o3-test-conditional-policy-v2-genuine-v1";
const CALLBACK_ERROR: &str = "genuine policy V2 callback refusal";
const CALLBACK_PANIC: &str = "genuine policy V2 callback unwind";
const ORIGIN_REFUSAL: &str = "conditional V2 CPU input substitution";
const ROWS: [&str; 9] = [
    "original-typed-policy-input",
    "registration-input-refused",
    "retained-same-policy-input",
    "origin-rustc-invocation",
    "origin-native-policy",
    "origin-policy-generation",
    "origin-mapping-ordinal",
    "callback-error",
    "callback-unwind",
];
static SERIAL: Mutex<()> = Mutex::new(());
static ACTIVE: Mutex<Option<State>> = Mutex::new(None);

struct State {
    seen: usize,
    report: Option<Report>,
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        ACTIVE
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    name: String,
    callback_entered: bool,
    work: usize,
    floor_before: usize,
    floor_after: usize,
    refusal: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema: String,
    observations: usize,
    semantic_mir_sha256: [u8; 32],
    semantic_root: u32,
    rustc_invocation_sha256: [u8; 32],
    native_policy_sha256: [u8; 32],
    policy_generation: u64,
    mapping_ordinal: u32,
    statement: [u8; 32],
    cpu_input_commitment: [u8; 32],
    original_subjects_borrowed: bool,
    original_floor_before: usize,
    original_floor_after: usize,
    original_work_before: usize,
    original_work_after: usize,
    original_account_preserved: bool,
    original_denials_preserved: bool,
    rows: Vec<Row>,
    outer_postchecks_completed: bool,
    live_enrollment_issuer_checked: bool,
    live_enrollment_currentness_checked: bool,
    decoded_policy_recovery_covered: bool,
    native_output_emitted: bool,
    qualification_credit: bool,
    grants_artifact_or_launch_authority: bool,
}

pub(crate) fn observe(child: &str, run: impl FnOnce()) -> Json {
    let args: Vec<_> = std::env::args().collect();
    assert!(!child.is_empty());
    assert!(args.iter().any(|arg| arg == "--exact") && args.iter().any(|arg| arg == child));
    assert!(args.iter().any(|arg| arg == "--test-threads=1"));
    let _serial = SERIAL.lock().unwrap();
    {
        let mut state = ACTIVE.lock().unwrap();
        assert!(state.is_none());
        *state = Some(State {
            seen: 0,
            report: None,
        });
    }
    let _reset = Reset;
    run();
    let mut state = ACTIVE.lock().unwrap();
    let state = state.as_mut().unwrap();
    assert_eq!(
        state.seen, 1,
        "one actual enrolled production replay required"
    );
    let mut report = state.report.take().expect("complete genuine policy probes");
    report.outer_postchecks_completed = true;
    let value = serde_json::to_value(report).unwrap();
    check_report(&value);
    value
}

pub(crate) fn check_report(value: &Json) {
    let report: Report = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(report.schema, SCHEMA);
    assert_eq!(report.observations, 1);
    assert_ne!(report.statement, [0; 32]);
    assert_ne!(report.cpu_input_commitment, [0; 32]);
    assert!(report.original_subjects_borrowed);
    assert_eq!(report.original_floor_before, report.original_floor_after);
    assert!(report.original_work_after > report.original_work_before);
    assert!(report.original_account_preserved && report.original_denials_preserved);
    assert!(report.outer_postchecks_completed);
    assert!(
        !report.live_enrollment_issuer_checked
            && !report.live_enrollment_currentness_checked
            && !report.decoded_policy_recovery_covered
            && !report.native_output_emitted
            && !report.qualification_credit
            && !report.grants_artifact_or_launch_authority
    );
    assert_eq!(report.rows.len(), ROWS.len());
    for (index, (row, expected)) in report.rows.iter().zip(ROWS).enumerate() {
        assert_eq!(row.name, expected);
        assert_eq!(row.floor_before, row.floor_after);
        assert_eq!(row.callback_entered, matches!(index, 2 | 7 | 8));
        if index < 2 {
            assert_eq!(row.work, 0);
        } else {
            assert!(row.work > 0);
        }
        match index {
            0 | 2 => assert!(row.refusal.is_none()),
            1 => assert_eq!(
                row.refusal.as_deref(),
                Some("policy enrollment requires the V2 CPU execution path")
            ),
            3..=6 => assert_eq!(row.refusal.as_deref(), Some(ORIGIN_REFUSAL)),
            7 => assert_eq!(row.refusal.as_deref(), Some(CALLBACK_ERROR)),
            8 => assert_eq!(row.refusal.as_deref(), Some(CALLBACK_PANIC)),
            _ => unreachable!(),
        }
    }
}

fn typed_input<'a>(
    request: &Request<'_>,
    binding: &'a Binding,
    root: u32,
) -> NativeCpuPolicyInputV2<'a> {
    let Origin::ReferenceEnrollment(origin) = &binding.origin else {
        panic!("policy probe received a registration binding");
    };
    let input = native_cpu_policy_input_v2(request, binding, root).unwrap();
    assert_eq!(
        input.association.semantic_mir_sha256,
        *request
            .source()
            .semantic_ssa()
            .source_semantic()
            .semantic_sha256()
            .as_bytes()
    );
    assert_eq!(input.association.semantic_root, root);
    assert_eq!(input.association.origin, *origin);
    assert_eq!(
        input.origin_v2(),
        NativeCpuOriginV2::AdmittedPolicy(*origin)
    );
    assert!(std::ptr::eq(
        input.association.logical_kernel_name,
        binding.logical_kernel_name.as_str()
    ));
    assert!(std::ptr::eq(input.kernel, &binding.kernel));
    assert!(std::ptr::eq(input.reference, &binding.reference));
    assert!(std::ptr::eq(
        input.replay.signature_preimage,
        &binding.signature_preimage
    ));
    assert!(std::ptr::eq(input.replay.effect_ir, &binding.effect_ir));
    assert_eq!(input.replay.effect_ir_sha256, binding.effect_ir_sha256);
    assert!(std::ptr::eq(
        input.replay.observable_output_writes,
        binding.observable_output_writes.as_ref()
    ));
    input
}

fn row(
    name: &str,
    before: (usize, usize),
    budget: &Budget<'_>,
    entered: bool,
    refusal: Option<&str>,
) -> Row {
    assert_eq!(budget.storage(), before.1);
    Row {
        name: name.into(),
        callback_entered: entered,
        work: budget.work() - before.0,
        floor_before: before.1,
        floor_after: budget.storage(),
        refusal: refusal.map(str::to_owned),
    }
}

fn callbacks(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Row>,
) {
    for unwind in [false, true] {
        let before = (budget.work(), budget.storage());
        let account = budget.work_ledger_identity_v1();
        let mut entered = false;
        let result = catch_unwind(AssertUnwindSafe(|| {
            retained.with_replayed_policy_request_v2(
                request,
                typed_input(request, binding, root),
                budget,
                |execution, budget| -> Result<(), FormulaError> {
                    entered = true;
                    assert_eq!(execution.report(), retained.report());
                    assert!(budget.work_ledger_identity_v1() == account);
                    budget.reserve_storage(7)?;
                    if unwind {
                        panic!("{CALLBACK_PANIC}");
                    }
                    Err(FormulaError::Subject(CALLBACK_ERROR))
                },
            )
        }));
        assert!(entered && budget.work_ledger_identity_v1() == account);
        if unwind {
            let panic = result.expect_err("policy callback must unwind");
            assert!(
                panic
                    .downcast_ref::<String>()
                    .is_some_and(|value| value == CALLBACK_PANIC)
                    || panic.downcast_ref::<&str>() == Some(&CALLBACK_PANIC)
            );
        } else {
            assert!(matches!(
                result.unwrap(),
                Err(FormulaError::Subject(CALLBACK_ERROR))
            ));
        }
        assert_eq!(budget.storage(), before.1 + 7);
        budget.release_storage(7).unwrap();
        rows.push(row(
            if unwind { ROWS[8] } else { ROWS[7] },
            before,
            budget,
            entered,
            Some(if unwind {
                CALLBACK_PANIC
            } else {
                CALLBACK_ERROR
            }),
        ));
    }
}

pub(super) fn on_replay(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
) {
    {
        let mut state = ACTIVE.lock().unwrap();
        let Some(state) = state.as_mut() else {
            return;
        };
        state.seen += 1;
        assert_eq!(state.seen, 1);
    }
    let before = (budget.work(), budget.storage());
    let account = budget.work_ledger_identity_v1();
    let denials = (budget.failed_work(), budget.failed_storage());
    let input = typed_input(request, binding, root);
    let semantic_mir_sha256 = input.association.semantic_mir_sha256;
    let origin: ReferenceEnrollmentOriginV1 = input.association.origin;
    let mut rows = Vec::with_capacity(ROWS.len());
    rows.push(row(ROWS[0], before, budget, false, None));
    let Err(super::Error::UnsupportedReference(reason)) =
        native_cpu_input_v1(request, binding, root)
    else {
        panic!("policy enrollment must refuse the V1 constructor");
    };
    assert_eq!(
        reason,
        "policy enrollment requires the V2 CPU execution path"
    );
    rows.push(row(ROWS[1], before, budget, false, Some(reason)));

    let mut entered = false;
    retained
        .with_replayed_policy_request_v2(request, input, budget, |execution, budget| {
            entered = true;
            assert!(budget.work_ledger_identity_v1() == account);
            assert_eq!(execution.report(), retained.report());
            Ok(())
        })
        .unwrap();
    assert!(entered);
    rows.push(row(ROWS[2], before, budget, entered, None));

    for field in 0..4 {
        let before = (budget.work(), budget.storage());
        let mut input = typed_input(request, binding, root);
        match field {
            0 => input.association.origin.rustc_invocation_sha256[31] ^= 1,
            1 => input.association.origin.native_policy_sha256[31] ^= 1,
            2 => input.association.origin.policy_generation ^= 1,
            3 => input.association.origin.mapping_ordinal ^= 1,
            _ => unreachable!(),
        }
        assert_ne!(input.association.origin, origin);
        let mut entered = false;
        let result = retained.with_replayed_policy_request_v2(request, input, budget, |_, _| {
            entered = true;
            Ok(())
        });
        assert!(!entered);
        assert!(matches!(result, Err(FormulaError::Subject(ORIGIN_REFUSAL))));
        assert!(budget.work_ledger_identity_v1() == account);
        rows.push(row(
            ROWS[field + 3],
            before,
            budget,
            entered,
            Some(ORIGIN_REFUSAL),
        ));
    }
    callbacks(retained, request, binding, root, budget, &mut rows);
    assert_eq!(budget.storage(), before.1);
    assert!(budget.work_ledger_identity_v1() == account);
    assert_eq!((budget.failed_work(), budget.failed_storage()), denials);
    let unchanged = typed_input(request, binding, root);
    assert_eq!(unchanged.association.origin, origin);
    ACTIVE.lock().unwrap().as_mut().unwrap().report = Some(Report {
        schema: SCHEMA.into(),
        observations: 1,
        semantic_mir_sha256,
        semantic_root: root,
        rustc_invocation_sha256: origin.rustc_invocation_sha256,
        native_policy_sha256: origin.native_policy_sha256,
        policy_generation: origin.policy_generation,
        mapping_ordinal: origin.mapping_ordinal,
        statement: *retained.report().statement_identity().as_bytes(),
        cpu_input_commitment: *retained.report().cpu_input_commitment().as_bytes(),
        original_subjects_borrowed: true,
        original_floor_before: before.1,
        original_floor_after: budget.storage(),
        original_work_before: before.0,
        original_work_after: budget.work(),
        original_account_preserved: true,
        original_denials_preserved: true,
        rows,
        outer_postchecks_completed: false,
        live_enrollment_issuer_checked: false,
        live_enrollment_currentness_checked: false,
        decoded_policy_recovery_covered: false,
        native_output_emitted: false,
        qualification_credit: false,
        grants_artifact_or_launch_authority: false,
    });
}

//! Opt-in genuine tests inside the real lower continuation, not a Request constructor.
//! Unexecuted protected coverage until the genuine prepared parent is run.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_verifier::portable_reference_v1::{self as cpu, extraction::ReferenceWorkV1};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

const CHILD: &str = "production_rustc_driver_v1::checked_output_source_v1_tests::conditional_bound_source::vecadd::conditional_reference_composition_source_child";
const SCHEMA: &str = "fe2o3-test-conditional-reference-composition-v1";
const CALLBACK_ERROR: &str = "composition nested callback refusal";
const CALLBACK_PANIC: &str = "composition nested callback unwind";
const PROBE_CAP: usize = 32 * 1024 * 1024;
static SERIAL: Mutex<()> = Mutex::new(());
static STATE: Mutex<State> = Mutex::new(State {
    enabled: false,
    seen: 0,
    origin: None,
    report: None,
});

struct State {
    enabled: bool,
    seen: usize,
    origin: Option<std::thread::ThreadId>,
    report: Option<Report>,
}
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        state.enabled = false;
        state.report = None;
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema: String,
    observations: usize,
    semantic_root: u32,
    callback_on_other_thread: bool,
    source_adjusted: Vec<(u32, u32)>,
    nonidentity_source_adjusted_covered: bool,
    subject_debit: usize,
    shared_work: usize,
    adapter_work: usize,
    floor_before: usize,
    floor_after: usize,
    work_before: usize,
    work_after: usize,
    original_account_preserved: bool,
    original_denials_preserved: bool,
    rows: Vec<Json>,
    outer_postchecks_completed: bool,
    native_output_emitted: bool,
    qualification_credit: bool,
    grants_artifact_or_launch_authority: bool,
}

/// The prepared parent selects one exact child. Global state is scoped because
/// run_compiler may execute its analysis callback on a different thread; TLS
/// would silently miss it. No environment mutation, thread spawn or unsafe code.
pub(crate) fn observe(run: impl FnOnce()) -> Json {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.iter().any(|a| a == "--exact") && args.iter().any(|a| a == CHILD));
    assert!(args.iter().any(|a| a == "--test-threads=1"));
    let _serial = SERIAL.lock().unwrap();
    {
        let mut state = STATE.lock().unwrap();
        assert!(!state.enabled);
        *state = State {
            enabled: true,
            seen: 0,
            origin: Some(std::thread::current().id()),
            report: None,
        };
    }
    let _reset = Reset;
    run(); // The existing child must still check genuine proof and final refusal.
    let mut state = STATE.lock().unwrap();
    assert_eq!(
        state.seen, 1,
        "one actual retained-formula lower request required"
    );
    let mut report = state.report.take().expect("complete composition probes");
    report.outer_postchecks_completed = true;
    let value = serde_json::to_value(report).unwrap();
    check_report(&value);
    value
}

pub(crate) fn check_report(value: &Json) {
    let report: Report = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(report.schema, SCHEMA);
    assert_eq!(report.observations, 1);
    assert_eq!(report.source_adjusted, [(0, 0), (1, 1), (2, 2)]);
    assert!(!report.nonidentity_source_adjusted_covered);
    assert_eq!(report.subject_debit, 256);
    assert_eq!(report.shared_work, report.adapter_work);
    assert!(report.shared_work > 256 && report.work_after > report.work_before);
    assert_eq!(report.floor_before, report.floor_after);
    assert!(
        report.original_account_preserved
            && report.original_denials_preserved
            && report.outer_postchecks_completed
    );
    assert!(
        !report.native_output_emitted
            && !report.qualification_credit
            && !report.grants_artifact_or_launch_authority
    );
    let mut expected = vec!["shared-success".to_owned(), "adapter-success".to_owned()];
    for route in ["shared", "adapter"] {
        expected.extend([
            format!("{route}-callback-error"),
            format!("{route}-callback-unwind"),
        ]);
    }
    expected.push("wrong-root".into());
    expected.extend((0..10).map(|n| format!("identity-{n}")));
    expected.extend(
        [
            "output-occurrence",
            "same-input-read",
            "source-input-permutation",
        ]
        .map(str::to_owned),
    );
    for source in [0, 1] {
        expected.extend([
            format!("origin-source-{source}"),
            format!("origin-adjusted-{source}"),
        ]);
    }
    for route in ["shared", "adapter"] {
        for case in [
            "exact",
            "join-work-short",
            "subject-work-short",
            "storage-short",
        ] {
            expected.push(format!("component-{route}-{case}"));
        }
    }
    assert_eq!(
        report
            .rows
            .iter()
            .map(|row| row["case"].as_str().unwrap())
            .collect::<Vec<_>>(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert!(report.rows.iter().all(|row| row["checked"] == true));
}

fn join<R>(
    adapter: bool,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    consume: impl for<'a> FnOnce(
        &'a SourceBoundCpuCorrespondenceV1<'_>,
        &mut Budget<'_>,
    ) -> Result<R, Error>,
) -> Result<R, Error> {
    if adapter {
        with_cpu_binding(request, binding, root, budget, consume)
    } else {
        portable::with_source_bound_cpu_correspondence_v1(
            request,
            borrowed_input(binding),
            root,
            budget,
            consume,
        )
        .map_err(Error::from)?
    }
}

fn success(
    adapter: bool,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Json>,
) -> usize {
    let account = budget.work_ledger_identity_v1();
    let (floor, before) = (budget.storage(), budget.work());
    let mut called = false;
    join(adapter, request, binding, root, budget, |checked, b| {
        called = true;
        assert!(std::ptr::eq(checked.request(), request));
        assert!(b.storage() > floor && b.work_ledger_identity_v1() == account);
        let before = b.work();
        checked.require_subjects(b).map_err(Error::from)?;
        assert_eq!(b.work() - before, 256);
        b.reserve_storage(7).unwrap();
        Ok(())
    })
    .unwrap();
    assert!(called && budget.work_ledger_identity_v1() == account);
    assert_eq!(budget.storage(), floor + 7);
    budget.release_storage(7).unwrap(); // Only this test callback's reservation.
    let cost = budget.work() - before;
    rows.push(
        json!({"case": if adapter {"adapter-success"} else {"shared-success"},
        "checked": true, "work": cost, "subject_debit": 256, "callback_storage_retained": 7}),
    );
    cost
}

fn callbacks(
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Json>,
) {
    for adapter in [false, true] {
        let route = if adapter { "adapter" } else { "shared" };
        for unwind in [false, true] {
            let (floor, before) = (budget.storage(), budget.work());
            let account = budget.work_ledger_identity_v1();
            let mut entered = false;
            let result = catch_unwind(AssertUnwindSafe(|| {
                let consume = |cpu: &SourceBoundCpuCorrespondenceV1<'_>,
                               b: &mut Budget<'_>|
                 -> Result<(), Error> {
                    entered = true;
                    cpu.require_subjects(b).map_err(Error::from)?;
                    b.reserve_storage(7).unwrap();
                    if unwind {
                        panic!("{CALLBACK_PANIC}");
                    }
                    Err(Error::UnsupportedReference(CALLBACK_ERROR))
                };
                if adapter {
                    assert!(matches!(
                        with_cpu_binding(request, binding, root, budget, consume),
                        Err(Error::UnsupportedReference(CALLBACK_ERROR))
                    ));
                } else {
                    // Public join retains the callback error as its nested R.
                    let nested = portable::with_source_bound_cpu_correspondence_v1(
                        request,
                        borrowed_input(binding),
                        root,
                        budget,
                        consume,
                    )
                    .unwrap();
                    assert!(matches!(
                        nested,
                        Err(Error::UnsupportedReference(CALLBACK_ERROR))
                    ));
                }
            }));
            assert!(entered, "a pre-callback failure is not a callback control");
            if unwind {
                let panic = result.expect_err("callback unwind propagated");
                assert!(
                    panic
                        .downcast_ref::<String>()
                        .is_some_and(|v| v == CALLBACK_PANIC)
                        || panic.downcast_ref::<&str>() == Some(&CALLBACK_PANIC)
                );
            } else {
                result.unwrap();
            }
            assert!(budget.work_ledger_identity_v1() == account && budget.work() > before);
            assert_eq!(budget.storage(), floor + 7);
            budget.release_storage(7).unwrap();
            rows.push(
                json!({"case": format!("{route}-callback-{}", if unwind {"unwind"} else {"error"}),
                "checked": true, "work": budget.work() - before, "callback_storage_retained": 7}),
            );
        }
    }
}

fn rejection(
    request: &Request<'_>,
    input: ConditionalReferenceInputV1<'_>,
    root: u32,
    budget: &mut Budget<'_>,
    name: &str,
    expected: &'static str,
    rows: &mut Vec<Json>,
) {
    let floor = budget.storage();
    let account = budget.work_ledger_identity_v1();
    let before = budget.work();
    let mut entered = false;
    let error =
        portable::with_source_bound_cpu_correspondence_v1(request, input, root, budget, |_, _| {
            entered = true;
        })
        .expect_err("changed input reached source join callback");
    assert!(!entered && budget.work_ledger_identity_v1() == account);
    assert_eq!(budget.storage(), floor);
    assert!(
        matches!(
            &error,
            portable::ConditionalReferenceErrorV1::UnsupportedReference(actual) if *actual == expected
        ),
        "{name}: expected {expected}, got {error}"
    );
    rows.push(json!({"case": name, "checked": true, "refusal": error.to_string(), "work": budget.work() - before}));
}

fn identities(
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Json>,
) {
    rejection(
        request,
        borrowed_input(binding),
        u32::MAX,
        budget,
        "wrong-root",
        "conditional CPU semantic root",
        rows,
    );
    for case in 0..10 {
        let mut kernel = binding.kernel;
        let mut reference = binding.reference;
        match case {
            0 => kernel = binding.reference,
            1 => reference = binding.kernel,
            2 => kernel.function_sha256[0] ^= 1,
            3 => kernel.item_definition_sha256[0] ^= 1,
            4 => kernel.monomorphization_sha256[0] ^= 1,
            5 => kernel.generic_type_arguments_sha256[0] ^= 1,
            6 => kernel.const_generic_arguments_sha256[0] ^= 1,
            7 => kernel.rustc_mir_body_sha256[0] ^= 1,
            8 => reference.function_sha256[0] ^= 1,
            _ => reference.rustc_mir_body_sha256[0] ^= 1,
        }
        let mut input = borrowed_input(binding);
        input.kernel = &kernel;
        input.reference = &reference;
        rejection(
            request,
            input,
            root,
            budget,
            &format!("identity-{case}"),
            if matches!(case, 0 | 2..=6) {
                "conditional CPU kernel identity"
            } else {
                "conditional CPU reference subject substitution"
            },
            rows,
        );
    }
}

struct FixtureMeter<'a, 'w>(RefCell<&'a mut Budget<'w>>);
impl ReferenceWorkV1 for FixtureMeter<'_, '_> {
    fn charge(&self, amount: usize) -> Result<(), cpu::ReferenceBindingErrorV1> {
        self.0
            .borrow_mut()
            .charge_work(amount)
            .map_err(|e| cpu::ReferenceBindingErrorV1::new(e.to_string()))
    }
}

// This visitor only changes inert fixture operands; the existing CPU resolver
// below, not this visitor, rederives the expressions and both retained claims.
fn remap_inputs(ir: &mut cpu::ReferenceEffectIrV1, a: u32, b: u32, swap: bool) {
    let raw = |value: &mut u32| {
        if *value == b {
            *value = a;
        } else if swap && *value == a {
            *value = b;
        }
    };
    let place = |p: &mut cpu::ReferencePlaceV1| {
        let mut value = p.local.checked_sub(1).expect("read input local");
        raw(&mut value);
        p.local = value + 1;
    };
    let operand = |op: &mut cpu::ReferenceOperandV1| match op {
        cpu::ReferenceOperandV1::Copy(p) | cpu::ReferenceOperandV1::Move(p) => {
            if p.local != 0 {
                place(p);
            }
        }
        cpu::ReferenceOperandV1::Constant(_) => (),
    };
    for block in &mut ir.blocks {
        for assignment in &mut block.assignments {
            match &mut assignment.value {
                cpu::ReferenceValueV1::Use(op)
                | cpu::ReferenceValueV1::Unary { operand: op, .. }
                | cpu::ReferenceValueV1::Cast { operand: op, .. } => operand(op),
                cpu::ReferenceValueV1::Binary { lhs, rhs, .. } => {
                    operand(lhs);
                    operand(rhs);
                }
                cpu::ReferenceValueV1::InputLength { reference_argument } => {
                    raw(reference_argument)
                }
                cpu::ReferenceValueV1::SafeHelperCall { .. } => panic!("unexpected Vecadd helper"),
            }
        }
        match &mut block.terminator {
            cpu::ReferenceTerminatorV1::Switch { discriminant, .. } => operand(discriminant),
            cpu::ReferenceTerminatorV1::Assert {
                condition,
                bounds_check,
                ..
            } => {
                operand(condition);
                if let Some(bounds) = bounds_check {
                    operand(&mut bounds.index);
                    operand(&mut bounds.length);
                }
            }
            cpu::ReferenceTerminatorV1::Goto { .. } | cpu::ReferenceTerminatorV1::Return => (),
        }
    }
}

fn coherent_inputs(
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Json>,
) {
    assert!(binding.effect_ir.blocks.len() <= 64 && binding.effect_ir.local_count <= 4096);
    assert!(
        binding
            .effect_ir
            .blocks
            .iter()
            .map(|b| b.assignments.len())
            .sum::<usize>()
            <= 512
    );
    assert!(binding.effect_ir.loop_summaries.is_empty());
    let [write] = binding.observable_output_writes.as_ref() else {
        panic!("one genuine output");
    };
    let derived = binding.signature_preimage.derive_relations_v1().unwrap();
    let a = derived
        .reference_argument_for_kernel_argument_v1(0)
        .unwrap();
    let b = derived
        .reference_argument_for_kernel_argument_v1(1)
        .unwrap();
    for case in [
        "output-occurrence",
        "same-input-read",
        "source-input-permutation",
    ] {
        // Test scaffold is bounded by the genuine small fixture above. This is
        // an inert IR copy, never a cloned authenticated Binding or Request.
        let mut ir = binding.effect_ir.clone();
        if case == "output-occurrence" {
            let assignments = &mut ir.blocks[write.block as usize].assignments;
            assert!(assignments.iter().any(|a| a.statement == write.statement));
            for assignment in assignments
                .iter_mut()
                .filter(|a| a.statement >= write.statement)
            {
                assignment.statement = assignment.statement.checked_add(1).unwrap();
            }
        } else {
            remap_inputs(&mut ir, a, b, case == "source-input-permutation");
        }
        let writes = ir
            .observable_output_writes_v1(&FixtureMeter(RefCell::new(&mut *budget)))
            .unwrap();
        assert_eq!(writes.len(), 1);
        if case == "output-occurrence" {
            assert_ne!(writes[0].statement, write.statement);
            assert_eq!(writes[0].rhs, write.rhs);
        } else {
            assert_eq!(writes[0].statement, write.statement);
            assert_ne!(
                writes[0].rhs, write.rhs,
                "read mutation must reach the output"
            );
        }
        ir.observable_output_effects = writes.clone().into_boxed_slice();
        let digest = ir.canonical_sha256_v1();
        assert_ne!(digest, binding.effect_ir_sha256);
        let input = || ConditionalReferenceInputV1 {
            kernel: &binding.kernel,
            reference: &binding.reference,
            replay: ReferenceReplayInputV1 {
                signature_preimage: &binding.signature_preimage,
                effect_ir: &ir,
                effect_ir_sha256: digest,
                observable_output_writes: &writes,
            },
        };
        cpu::with_replayed_output_writes_v1(input().replay, budget, |actual, _| {
            assert_eq!(actual.writes, writes)
        })
        .unwrap();
        rejection(
            request,
            input(),
            root,
            budget,
            case,
            if case == "output-occurrence" {
                "conditional CPU output occurrence"
            } else {
                "conditional CPU MIR value expression substitution"
            },
            rows,
        );
    }
}

fn loads<'a>(
    expression: &'a fe2o3_pliron::ProductionSemanticExpressionV2,
    out: &mut Vec<&'a fe2o3_pliron::ProductionSemanticLoadV2>,
    depth: usize,
    nodes: &mut usize,
) {
    use fe2o3_pliron::ProductionSemanticExpressionV2 as E;
    *nodes += 1;
    assert!(depth < 128 && *nodes <= 8192 && out.len() <= 2);
    match expression {
        E::Load(load) => out.push(load),
        E::Unary { operand, .. } | E::Cast { operand, .. } => loads(operand, out, depth + 1, nodes),
        E::Binary { lhs, rhs, .. } | E::Compare { lhs, rhs, .. } => {
            loads(lhs, out, depth + 1, nodes);
            loads(rhs, out, depth + 1, nodes);
        }
        E::Select {
            condition,
            when_true,
            when_false,
            ..
        } => {
            loads(condition, out, depth + 1, nodes);
            loads(when_true, out, depth + 1, nodes);
            loads(when_false, out, depth + 1, nodes);
        }
        E::Constant { .. } | E::Symbol { .. } | E::GlobalInvocation1d { .. } => (),
    }
}

fn origins(
    request: &Request<'_>,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Json>,
) -> Vec<(u32, u32)> {
    let input = request.pliron_input();
    let mut mapping: Vec<_> = request
        .arguments()
        .iter()
        .map(|a| (a.source_argument(), a.adjusted_argument()))
        .collect();
    mapping.sort_unstable();
    mapping.dedup();
    // The preserved Vecadd capture has IDENTITY mapping, not a shifted ABI.
    assert_eq!(mapping, [(0, 0), (1, 1), (2, 2)]);
    let [output] = input.outputs() else {
        panic!("genuine output roster");
    };
    let value = input.effect_contract(output).unwrap().gpu_value();
    let mut found = None;
    for block in input.kernel().blocks() {
        for operation in block.operations() {
            if let fe2o3_pliron::ProductionRankedOperationV1::SemanticExpression {
                result,
                expression,
                ..
            } = operation
                && value == fe2o3_pliron::ProductionRankedValueV1::Local(*result)
            {
                assert!(found.replace(expression).is_none());
            }
        }
    }
    let mut actual = Vec::new();
    loads(found.unwrap(), &mut actual, 0, &mut 0);
    assert_eq!(actual.len(), 2);
    assert_eq!(input.reads().len(), 2);
    for source in [0, 1] {
        let row = request
            .arguments()
            .iter()
            .find(|a| a.source_argument() == source)
            .unwrap();
        let read = input
            .reads()
            .iter()
            .find(|r| r.canonical().parameter() == row.canonical_parameter())
            .unwrap();
        let load = actual
            .iter()
            .find(|l| (l.block, l.operation) == (read.site().block, read.site().operation))
            .unwrap();
        portable::require_read_origins(
            input.kernel().blocks(),
            load,
            source,
            row.adjusted_argument(),
            budget,
        )
        .unwrap();
        for adjusted in [false, true] {
            let error = portable::require_read_origins(
                input.kernel().blocks(),
                load,
                source + u32::from(!adjusted),
                row.adjusted_argument() + u32::from(adjusted),
                budget,
            )
            .unwrap_err();
            let expected = if adjusted {
                "conditional CPU read expression origin substitution"
            } else {
                "conditional CPU read recipe origin substitution"
            };
            assert!(
                matches!(&error,
                    portable::ConditionalReferenceErrorV1::UnsupportedReference(actual) if *actual == expected
                ),
                "expected {expected}, got {error}"
            );
            rows.push(json!({"case": format!("origin-{}-{source}", if adjusted {"adjusted"} else {"source"}),
                "checked": true, "refusal": error.to_string(), "component_only": true,
                "actual_source": source, "actual_adjusted": row.adjusted_argument()}));
        }
    }
    mapping
}

fn component_resources(request: &Request<'_>, binding: &Binding, root: u32, rows: &mut Vec<Json>) {
    // These are bounded component CALLER accounts, not new production custody
    // of the borrowed lower owners and never replacements for the source ledger.
    for adapter in [false, true] {
        let route = if adapter { "adapter" } else { "shared" };
        let mut work = Work::new(PROBE_CAP);
        let mut b = Budget::new(&mut work, PROBE_CAP);
        b.charge_work(17).unwrap();
        b.reserve_storage(31).unwrap();
        join(adapter, request, binding, root, &mut b, |_, _| Ok(())).unwrap();
        assert_eq!(b.storage(), 31);
        let (plain, peak) = (b.work(), b.peak_storage());
        for (case, limit, storage, entered_expected) in [
            ("exact", plain + 256, peak, true),
            ("join-work-short", plain - 1, peak, false),
            ("subject-work-short", plain + 255, peak, true),
            ("storage-short", plain + 256, peak - 1, false),
        ] {
            let mut work = Work::new(limit);
            let mut b = Budget::new(&mut work, storage);
            b.charge_work(17).unwrap();
            b.reserve_storage(31).unwrap();
            let account = b.work_ledger_identity_v1();
            let mut entered = false;
            let result = join(adapter, request, binding, root, &mut b, |cpu, b| {
                entered = true;
                let before = b.work();
                let result = cpu.require_subjects(b).map_err(Error::from);
                if result.is_ok() {
                    assert_eq!(b.work() - before, 256);
                }
                result
            });
            assert_eq!(entered, entered_expected);
            assert_eq!(result.is_ok(), case == "exact");
            assert!(b.work_ledger_identity_v1() == account);
            assert_eq!(b.storage(), 31);
            if case == "exact" {
                assert_eq!(b.work(), plain + 256);
                assert_eq!(b.peak_storage(), peak);
                assert_eq!((b.failed_work(), b.failed_storage()), (None, None));
            } else if case == "storage-short" {
                assert_eq!(b.failed_storage(), Some(peak));
                assert_eq!(b.failed_work(), None);
            } else {
                let rejected = if case == "subject-work-short" {
                    assert_eq!(b.work(), plain);
                    plain + 256
                } else {
                    assert!(b.work() < plain);
                    plain
                };
                assert_eq!(b.failed_work(), Some(rejected));
                assert_eq!(b.failed_storage(), None);
            }
            rows.push(json!({"case": format!("component-{route}-{case}"), "checked": true,
                "component_only": true, "callback_entered": entered, "work": b.work(),
                "peak": b.peak_storage(), "failed_work": b.failed_work(), "failed_storage": b.failed_storage()}));
        }
    }
}

pub(super) fn on_replay(
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
) {
    let other_thread = {
        let mut state = STATE.lock().unwrap();
        if !state.enabled {
            return;
        }
        state.seen += 1;
        assert_eq!(state.seen, 1);
        state.origin != Some(std::thread::current().id())
    };
    // Exercise the live V2 assembler against the actual authenticated row. This
    // borrows only; it adds no codec/JOIN replay or report-schema field.
    let input = native_cpu_input_v1(request, binding, root).unwrap();
    assert_eq!(input.association.semantic_root, root);
    assert_eq!(
        input.association.semantic_mir_sha256,
        *request
            .source()
            .semantic_ssa()
            .source_semantic()
            .semantic_sha256()
            .as_bytes()
    );
    assert_eq!(
        input.association.semantic_mir_sha256,
        *request.pliron_input().source_semantic_identity().as_bytes()
    );
    assert_eq!(
        input.association.registration_path,
        binding.origin.source_registration_v1().unwrap()
    );
    assert_eq!(
        input.association.logical_kernel_name,
        binding.logical_kernel_name
    );
    assert!(std::ptr::eq(
        input.association.registration_path,
        binding.origin.source_registration_v1().unwrap()
    ));
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
    let (floor, work_before) = (budget.storage(), budget.work());
    let account = budget.work_ledger_identity_v1();
    let denials = (budget.failed_work(), budget.failed_storage());
    let mut rows = Vec::new();
    let shared_work = success(false, request, binding, root, budget, &mut rows);
    let adapter_work = success(true, request, binding, root, budget, &mut rows);
    assert_eq!(shared_work, adapter_work);
    callbacks(request, binding, root, budget, &mut rows);
    identities(request, binding, root, budget, &mut rows);
    coherent_inputs(request, binding, root, budget, &mut rows);
    let source_adjusted = origins(request, budget, &mut rows);
    let before_components = (budget.work(), budget.storage());
    component_resources(request, binding, root, &mut rows);
    assert_eq!((budget.work(), budget.storage()), before_components);
    assert!(budget.work_ledger_identity_v1() == account);
    assert_eq!(budget.storage(), floor);
    assert_eq!((budget.failed_work(), budget.failed_storage()), denials);
    let report = Report {
        schema: SCHEMA.into(),
        observations: 1,
        semantic_root: root,
        callback_on_other_thread: other_thread,
        source_adjusted,
        nonidentity_source_adjusted_covered: false,
        subject_debit: 256,
        shared_work,
        adapter_work,
        floor_before: floor,
        floor_after: budget.storage(),
        work_before,
        work_after: budget.work(),
        original_account_preserved: true,
        original_denials_preserved: true,
        rows,
        outer_postchecks_completed: false,
        native_output_emitted: false,
        qualification_credit: false,
        grants_artifact_or_launch_authority: false,
    };
    STATE.lock().unwrap().report = Some(report);
}

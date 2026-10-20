//! Opt-in probes of actual source/retained V2 custody. No Request, Execution or
//! success proof is fabricated. Signature mutations start from genuine wire;
//! no probe runs another protected proof.
use super::{Binding, Budget, Request, native_cpu_input_v1};
use cpu::codec::{
    NativeCpuCodecErrorV1 as CodecError, NativeCpuInputV1,
    with_decoded_native_cpu_input_v1 as decode, with_encoded_native_cpu_input_v1 as encode,
};
use fe2o3_functional_proof::{
    FunctionalRefinementBoundaryV2 as Boundary, FunctionalRefinementImportPolicyV2 as Policy,
    VerusToolchainIdentityV2,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceErrorV1 as Resource, CanonicalKernelIrWorkBudgetV1 as Work,
};
use fe2o3_proof_contracts::DigestV1;
use fe2o3_verifier::conditional_reference_v1::ConditionalReferenceErrorV1 as JoinError;
use fe2o3_verifier::portable_reference_v1::{self as cpu, extraction::ReferenceWorkV1};
use fe2o3_verifier::{
    InertFunctionalRefinementReceiptSignatureV2 as Signature,
    ProductionConditionalFormulaErrorV1 as FormulaError,
    ProductionConditionalFormulaErrorV2 as Error,
    ProductionConditionalFormulaReportV2 as FormulaReport,
    RetainedProductionConditionalFormulaV2 as Retained,
    import_and_retain_conditional_ranked_formula_v2 as import,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::{
    cell::{Cell, RefCell},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Mutex,
};

pub(crate) const CHILD: &str = "production_rustc_driver_v1::checked_output_source_v1_tests::conditional_bound_source::vecadd::conditional_formula_v2_source_child";
const SCHEMA: &str = "fe2o3-test-conditional-formula-v2-genuine-v1";
const CAP: usize = 32 * 1024 * 1024;
const PREFIX: usize = 17;
const CALLBACK_ERROR: &str = "genuine V2 callback refusal";
const CALLBACK_PANIC: &str = "genuine V2 callback unwind";
static SERIAL: Mutex<()> = Mutex::new(());
static ACTIVE: Mutex<Option<State>> = Mutex::new(None);
struct State {
    seen: usize,
    report: Option<Report>,
}
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    name: String,
    component_only: bool,
    callback_entered: bool,
    work: usize,
    floor_before: usize,
    floor_after: usize,
    provisional_drops: usize,
    refusal: Option<String>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema: String,
    observations: usize,
    semantic_root: u32,
    nonroot_function_index: Option<u32>,
    statement: [u8; 32],
    cpu_input_commitment: [u8; 32],
    original_floor_before: usize,
    original_floor_after: usize,
    original_work_before: usize,
    original_work_after: usize,
    original_account_preserved: bool,
    original_denials_preserved: bool,
    rows: Vec<Row>,
    outer_postchecks_completed: bool,
    foreign_account_replacement_covered: bool,
    native_output_emitted: bool,
    qualification_credit: bool,
    grants_artifact_or_launch_authority: bool,
}

pub(crate) fn observe(run: impl FnOnce()) -> Json {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.iter().any(|a| a == "--exact") && args.iter().any(|a| a == CHILD));
    assert!(args.iter().any(|a| a == "--test-threads=1"));
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
    run(); // Requires the real rustc callback, genuine proof and finalizer refusal.
    let mut state = ACTIVE.lock().unwrap();
    let state = state.as_mut().unwrap();
    assert_eq!(state.seen, 1);
    let mut report = state.report.take().expect("complete V2 genuine probes");
    report.outer_postchecks_completed = true;
    let value = serde_json::to_value(report).unwrap();
    check_report(&value);
    value
}

fn expected_rows() -> Vec<String> {
    let mut names = [
        "retained-same-input",
        "imported-same-input",
        "callback-error",
        "callback-unwind",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut mutations = [
        "source-hash",
        "root-not-in-roster",
        "registration",
        "logical-name",
    ]
    .map(str::to_owned)
    .to_vec();
    for subject in ["kernel", "reference"] {
        for field in 0..7 {
            mutations.push(format!("{subject}-identity-{field}"));
        }
    }
    mutations.push("coherent-cpu-operand".into());
    for mutation in mutations {
        names.extend([
            format!("retained-{mutation}"),
            format!("imported-{mutation}"),
        ]);
    }
    names.push("legacy-value-omission".into());
    names.extend(
        [
            "foreign-policy",
            "foreign-transport",
            "foreign-policy-and-transport",
            "wrong-boundary",
            "wrong-toolchain",
            "damaged-signature",
            "exact-resources",
            "work-short",
            "storage-short",
            "replay-damaged-floor",
            "decode-late-cleanup",
        ]
        .map(str::to_owned),
    );
    names
}

pub(crate) fn check_report(value: &Json) {
    let report: Report = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(report.schema, SCHEMA);
    assert_eq!(report.observations, 1);
    assert_ne!(report.statement, [0; 32]);
    assert_ne!(report.cpu_input_commitment, [0; 32]);
    assert_eq!(report.original_floor_before, report.original_floor_after);
    assert!(report.original_work_after > report.original_work_before);
    assert!(
        report.original_account_preserved
            && report.original_denials_preserved
            && report.outer_postchecks_completed
    );
    assert!(
        !report.foreign_account_replacement_covered
            && !report.native_output_emitted
            && !report.qualification_credit
            && !report.grants_artifact_or_launch_authority
    );
    assert_eq!(
        report
            .rows
            .iter()
            .map(|r| r.name.clone())
            .collect::<Vec<_>>(),
        expected_rows()
    );
    for (i, row) in report.rows.iter().enumerate() {
        assert_eq!(row.component_only, i >= 4);
        assert!(row.work > 0);
        if row.name != "replay-damaged-floor" && row.name != "decode-late-cleanup" {
            assert_eq!(row.floor_after, row.floor_before);
        } else {
            assert!(row.callback_entered && row.refusal.is_some());
            assert_eq!(row.provisional_drops, 1);
        }
        if [
            "retained-same-input",
            "imported-same-input",
            "exact-resources",
        ]
        .contains(&row.name.as_str())
        {
            assert!(row.callback_entered && row.refusal.is_none());
        } else {
            assert!(row.refusal.is_some());
        }
    }
}

fn row(
    name: impl Into<String>,
    component_only: bool,
    before: (usize, usize),
    b: &Budget<'_>,
    entered: bool,
    drops: usize,
    refusal: Option<String>,
) -> Row {
    Row {
        name: name.into(),
        component_only,
        callback_entered: entered,
        work: b.work() - before.0,
        floor_before: before.1,
        floor_after: b.storage(),
        provisional_drops: drops,
        refusal,
    }
}
fn component<R>(retained: &Retained, run: impl FnOnce(&mut Budget<'_>) -> R) -> R {
    let mut work = Work::new(CAP);
    let mut budget = Budget::new(&mut work, CAP);
    budget.charge_work(PREFIX).unwrap();
    // Test account for extra reimport scratch, not a substitute source owner.
    budget
        .reserve_storage(retained.retained_storage_v2() + 31)
        .unwrap();
    run(&mut budget)
}

// The accepted policy comes from retained protected-runtime custody, never the
// transported key. Both strict import and replay borrow THIS decoded owner.
fn imported(
    input: NativeCpuInputV1<'_>,
    request: &Request<'_>,
    signature: &Signature,
    accepted: &Policy,
    budget: &mut Budget<'_>,
    entered: &mut bool,
    must_refuse_import: bool,
) -> Result<FormulaReport, Error> {
    encode(input, budget, |bytes, _, budget| {
        decode(bytes, budget, |decoded, budget| {
            let result = import(request, decoded, signature, accepted, budget);
            if must_refuse_import {
                assert!(
                    result.is_err(),
                    "strict import returned an owner for a negative input"
                );
            }
            let retained = result?;
            let storage = retained.retained_storage_v2();
            let result = retained.with_replayed_decoded_request_v2(
                request,
                decoded,
                budget,
                |execution, _| {
                    *entered = true;
                    assert_eq!(execution.signed_receipt_wire(), signature.wire());
                    assert_eq!(execution.receipt_verifying_key(), signature.verifying_key());
                    Ok(execution.report())
                },
            );
            drop(retained);
            budget.release_storage(storage)?;
            result
        })
        .map_err(Error::Codec)?
    })
    .map_err(Error::Codec)?
}

fn successes(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    budget: &mut Budget<'_>,
    rows: &mut Vec<Row>,
) -> Signature {
    let before = (budget.work(), budget.storage());
    let account = budget.work_ledger_identity_v1();
    let signature = retained
        .with_replayed_request_v2(
            request,
            native_cpu_input_v1(request, binding, root).unwrap(),
            budget,
            |execution, b| {
                assert!(b.work_ledger_identity_v1() == account);
                assert_eq!(execution.report(), retained.report());
                Ok(Signature::from_untrusted_parts(
                    execution.signed_receipt_wire().try_into().unwrap(),
                    *execution.receipt_verifying_key(),
                ))
            },
        )
        .unwrap();
    assert!(budget.work_ledger_identity_v1() == account);
    assert_eq!(budget.storage(), before.1);
    rows.push(row(
        "retained-same-input",
        false,
        before,
        budget,
        true,
        0,
        None,
    ));
    let before = (budget.work(), budget.storage());
    let accepted = retained.import_policy_v2();
    let unchanged = accepted.clone();
    let mut entered = false;
    assert_eq!(
        imported(
            native_cpu_input_v1(request, binding, root).unwrap(),
            request,
            &signature,
            accepted,
            budget,
            &mut entered,
            false
        )
        .unwrap(),
        retained.report()
    );
    assert!(entered && budget.work_ledger_identity_v1() == account);
    assert_eq!(budget.storage(), before.1);
    assert_eq!(accepted, &unchanged);
    rows.push(row(
        "imported-same-input",
        false,
        before,
        budget,
        entered,
        0,
        None,
    ));
    signature
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
            retained.with_replayed_request_v2(
                request,
                native_cpu_input_v1(request, binding, root).unwrap(),
                budget,
                |_, b| -> Result<(), Error> {
                    entered = true;
                    b.reserve_storage(7)?;
                    if unwind {
                        panic!("{CALLBACK_PANIC}");
                    }
                    Err(Error::Subject(CALLBACK_ERROR))
                },
            )
        }));
        assert!(entered && budget.work_ledger_identity_v1() == account);
        if unwind {
            let panic = result.expect_err("callback must unwind");
            assert!(
                panic
                    .downcast_ref::<String>()
                    .is_some_and(|v| v == CALLBACK_PANIC)
                    || panic.downcast_ref::<&str>() == Some(&CALLBACK_PANIC)
            );
        } else {
            assert!(matches!(
                result.unwrap(),
                Err(Error::Subject(CALLBACK_ERROR))
            ));
        }
        assert_eq!(budget.storage(), before.1 + 7);
        budget.release_storage(7).unwrap();
        rows.push(row(
            if unwind {
                "callback-unwind"
            } else {
                "callback-error"
            },
            false,
            before,
            budget,
            entered,
            0,
            Some(
                if unwind {
                    CALLBACK_PANIC
                } else {
                    CALLBACK_ERROR
                }
                .into(),
            ),
        ));
    }
}

fn semantic_refusal(error: &Error) {
    assert!(
        matches!(
            error,
            Error::Subject(_) | Error::Correspondence(JoinError::UnsupportedReference(_))
        ),
        "expected semantic/signature refusal, not a codec/resource failure: {error}"
    );
}

fn reject_input<'a>(
    name: &str,
    make: impl Fn() -> NativeCpuInputV1<'a>,
    expected: Option<&str>,
    retained: &Retained,
    request: &Request<'_>,
    signature: &Signature,
    rows: &mut Vec<Row>,
) {
    // Establish codec acceptance before either negative; malformed transport is
    // not a substitute for testing B2, commitment replay or strict signature import.
    component(retained, |b| {
        encode(make(), b, |_, commitment, _| {
            assert_ne!(
                &commitment,
                retained.report().cpu_input_commitment().as_bytes()
            );
        })
        .unwrap();
    });
    for imported_route in [false, true] {
        component(retained, |b| {
            let before = (b.work(), b.storage());
            let account = b.work_ledger_identity_v1();
            let mut entered = false;
            let result = if imported_route {
                imported(
                    make(),
                    request,
                    signature,
                    retained.import_policy_v2(),
                    b,
                    &mut entered,
                    true,
                )
            } else {
                retained.with_replayed_request_v2(request, make(), b, |execution, _| {
                    entered = true;
                    Ok(execution.report())
                })
            };
            let error = result.expect_err("changed CPU input must not return a replayed report");
            assert!(!entered && b.work_ledger_identity_v1() == account);
            assert_eq!(b.storage(), before.1);
            assert_eq!((b.failed_work(), b.failed_storage()), (None, None));
            semantic_refusal(&error);
            // These fields are intentionally not selectors in B2. They must
            // fail the full-frame V2 commitment/signature, not an earlier join.
            if matches!(
                name,
                "registration"
                    | "logical-name"
                    | "kernel-identity-0"
                    | "reference-identity-0"
                    | "reference-identity-2"
                    | "reference-identity-3"
                    | "reference-identity-4"
                    | "reference-identity-5"
            ) {
                let reason = if imported_route {
                    "conditional V2 signature/policy"
                } else {
                    "conditional V2 CPU input substitution"
                };
                assert!(
                    matches!(&error, Error::Subject(actual) if *actual == reason),
                    "{error}"
                );
            }
            if let Some(expected) = expected {
                assert!(
                    matches!(&error, Error::Subject(actual) if *actual == expected),
                    "{error}"
                );
            }
            rows.push(row(
                format!(
                    "{}-{name}",
                    if imported_route {
                        "imported"
                    } else {
                        "retained"
                    }
                ),
                true,
                before,
                b,
                entered,
                0,
                Some(error.to_string()),
            ));
        });
    }
}

fn mutations(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    signature: &Signature,
    rows: &mut Vec<Row>,
) -> Option<u32> {
    let semantic = request.source().semantic_ssa().source_semantic();
    let is_root = |index| semantic.roots().iter().any(|r| r.index() == index);
    let nonroot = (0..u32::try_from(semantic.functions().len()).unwrap()).find(|i| !is_root(*i));
    // Prefer an actual function index outside the complete root roster. If the
    // fixture has none, record that gap and still test a codec-valid absent root.
    let absent = nonroot.unwrap_or_else(|| {
        (0..u32::try_from(fe2o3_mir_model::semantic_mir_v1::HARD_MAX_FUNCTIONS_V1).unwrap())
            .find(|i| !is_root(*i))
            .unwrap()
    });
    let registration = format!(
        "{}::v2_probe",
        binding.origin.source_registration_v1().unwrap()
    );
    let logical = format!("{}_v2_probe", binding.logical_kernel_name);
    for (case, name) in [
        "source-hash",
        "root-not-in-roster",
        "registration",
        "logical-name",
    ]
    .iter()
    .enumerate()
    {
        reject_input(
            name,
            || {
                let mut input = native_cpu_input_v1(request, binding, root).unwrap();
                match case {
                    0 => input.association.semantic_mir_sha256[0] ^= 1,
                    1 => input.association.semantic_root = absent,
                    2 => input.association.registration_path = &registration,
                    3 => input.association.logical_kernel_name = &logical,
                    _ => unreachable!(),
                }
                input
            },
            match case {
                0 => Some("conditional V2 CPU source association"),
                1 => Some("conditional V2 CPU root association"),
                _ => None,
            },
            retained,
            request,
            signature,
            rows,
        );
    }
    for reference in [false, true] {
        for field in 0..7 {
            let mut identity = if reference {
                binding.reference
            } else {
                binding.kernel
            };
            match field {
                0 => identity.def_path_hash[0] ^= 1,
                1 => identity.function_sha256[0] ^= 1,
                2 => identity.item_definition_sha256[0] ^= 1,
                3 => identity.monomorphization_sha256[0] ^= 1,
                4 => identity.generic_type_arguments_sha256[0] ^= 1,
                5 => identity.const_generic_arguments_sha256[0] ^= 1,
                6 => identity.rustc_mir_body_sha256[0] ^= 1,
                _ => unreachable!(),
            }
            reject_input(
                &format!(
                    "{}-identity-{field}",
                    if reference { "reference" } else { "kernel" }
                ),
                || {
                    let mut input = native_cpu_input_v1(request, binding, root).unwrap();
                    if reference {
                        input.reference = &identity;
                    } else {
                        input.kernel = &identity;
                    }
                    input
                },
                None,
                retained,
                request,
                signature,
                rows,
            );
        }
    }
    coherent_operand(retained, request, binding, root, signature, rows);
    nonroot
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

fn coherent_operand(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    signature: &Signature,
    rows: &mut Vec<Row>,
) {
    // Only inert CPU fixture data is cloned; source and retained proof stay borrowed.
    let mut ir = binding.effect_ir.clone();
    let mut changed = false;
    'blocks: for block in ir.blocks.iter_mut().rev() {
        for assignment in block.assignments.iter_mut().rev() {
            if let cpu::ReferenceValueV1::Binary {
                operation: cpu::ReferenceBinaryOpV1::Add,
                lhs,
                rhs,
                ..
            } = &mut assignment.value
            {
                if lhs != rhs {
                    *rhs = lhs.clone();
                    changed = true;
                    break 'blocks;
                }
            }
        }
    }
    assert!(
        changed,
        "genuine Vecadd must contain an addition with distinct operands"
    );
    let writes = component(retained, |b| {
        ir.observable_output_writes_v1(&FixtureMeter(RefCell::new(b)))
            .unwrap()
    });
    let [write] = writes.as_slice() else {
        panic!("one mutated output required");
    };
    let [original] = binding.observable_output_writes.as_ref() else {
        panic!("one actual output required");
    };
    assert_eq!(
        (write.block, write.statement),
        (original.block, original.statement)
    );
    assert_ne!(
        write.rhs, original.rhs,
        "the changed operand must reach the actual output"
    );
    ir.observable_output_effects = writes.clone().into_boxed_slice();
    let digest = ir.canonical_sha256_v1();
    assert_ne!(digest, binding.effect_ir_sha256);
    reject_input(
        "coherent-cpu-operand",
        || {
            let mut input = native_cpu_input_v1(request, binding, root).unwrap();
            input.replay.effect_ir = &ir;
            input.replay.effect_ir_sha256 = digest;
            input.replay.observable_output_writes = &writes;
            input
        },
        None,
        retained,
        request,
        signature,
        rows,
    );
}

fn legacy_omission(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    signature: &Signature,
    rows: &mut Vec<Row>,
) {
    let mut ir = binding.effect_ir.clone();
    assert_eq!(ir.observable_output_effects.len(), 1);
    let changed = cpu::ReferenceValueV1::Use(cpu::ReferenceOperandV1::Constant(
        cpu::ReferenceConstantV1::Scalar {
            scalar: cpu::ReferenceScalarTypeV1::F32,
            bits: 0,
        },
    ));
    assert_ne!(ir.observable_output_effects[0].value, changed);
    ir.observable_output_effects[0].value = changed;
    assert_eq!(ir.canonical_sha256_v1(), binding.effect_ir_sha256);
    let make = || {
        let mut input = native_cpu_input_v1(request, binding, root).unwrap();
        input.replay.effect_ir = &ir;
        input.replay.observable_output_writes = &ir.observable_output_effects;
        input
    };
    component(retained, |b| {
        let before = (b.work(), b.storage());
        let account = b.work_ledger_identity_v1();
        let mut entered = false;
        for imported_route in [false, true] {
            let result = if imported_route {
                imported(
                    make(),
                    request,
                    signature,
                    retained.import_policy_v2(),
                    b,
                    &mut entered,
                    true,
                )
            } else {
                retained.with_replayed_request_v2(request, make(), b, |execution, _| {
                    entered = true;
                    Ok(execution.report())
                })
            };
            assert!(matches!(
                result,
                Err(Error::Codec(CodecError::Wire("effect assignment value")))
            ));
            assert_eq!(b.storage(), before.1);
        }
        assert!(!entered && b.work_ledger_identity_v1() == account);
        assert_eq!((b.failed_work(), b.failed_storage()), (None, None));
        rows.push(row(
            "legacy-value-omission",
            true,
            before,
            b,
            entered,
            0,
            Some("codec: effect assignment value (both routes)".into()),
        ));
    });
}

fn policies(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    signature: &Signature,
    rows: &mut Vec<Row>,
) {
    let accepted = retained.import_policy_v2();
    let unchanged = accepted.clone();
    // Compressed Ed25519 base point, only a foreign public key for refusal.
    // No signing key, signed stand-in or accepted policy from transported bytes.
    let mut foreign_key = [0x66; 32];
    foreign_key[0] = 0x58;
    let foreign = Policy::new(foreign_key, accepted.toolchain(), accepted.boundary()).unwrap();
    assert_ne!(foreign.signer_identity(), accepted.signer_identity());
    let toolchain = accepted.toolchain();
    let mut closure = *toolchain.runtime_closure().as_bytes();
    closure[0] ^= 1;
    let wrong_toolchain = VerusToolchainIdentityV2::new(
        toolchain.verus_executable(),
        toolchain.verus_configuration(),
        toolchain.solver_executable(),
        toolchain.solver_configuration(),
        DigestV1::from_untrusted_bytes(closure),
    )
    .unwrap();
    for name in [
        "foreign-policy",
        "foreign-transport",
        "foreign-policy-and-transport",
        "wrong-boundary",
        "wrong-toolchain",
        "damaged-signature",
    ] {
        let policy = match name {
            "foreign-policy" | "foreign-policy-and-transport" => foreign.clone(),
            "wrong-boundary" => Policy::new(
                *signature.verifying_key(),
                accepted.toolchain(),
                Boundary::SafeReferenceMirToKernelMir,
            )
            .unwrap(),
            "wrong-toolchain" => Policy::new(
                *signature.verifying_key(),
                wrong_toolchain,
                accepted.boundary(),
            )
            .unwrap(),
            _ => accepted.clone(),
        };
        let mut wire = *signature.wire();
        if name == "damaged-signature" {
            let last = wire.len() - 1;
            wire[last] ^= 1;
        }
        let key = if matches!(name, "foreign-transport" | "foreign-policy-and-transport") {
            foreign_key
        } else {
            *signature.verifying_key()
        };
        let transported = Signature::from_untrusted_parts(wire, key);
        component(retained, |b| {
            let before = (b.work(), b.storage());
            let account = b.work_ledger_identity_v1();
            let mut entered = false;
            let error = imported(
                native_cpu_input_v1(request, binding, root).unwrap(),
                request,
                &transported,
                &policy,
                b,
                &mut entered,
                true,
            )
            .expect_err("changed policy/signature must refuse");
            let expected = if matches!(name, "wrong-boundary" | "wrong-toolchain") {
                "conditional V2 accepted boundary/toolchain"
            } else {
                "conditional V2 signature/policy"
            };
            assert!(
                matches!(&error, Error::Subject(actual) if *actual == expected),
                "{error}"
            );
            assert!(!entered && b.work_ledger_identity_v1() == account);
            assert_eq!(b.storage(), before.1);
            assert_eq!((b.failed_work(), b.failed_storage()), (None, None));
            rows.push(row(
                name,
                true,
                before,
                b,
                entered,
                0,
                Some(error.to_string()),
            ));
        });
        assert_eq!(accepted, &unchanged);
    }
}

#[derive(Debug)]
struct Provisional<'a>(&'a Cell<usize>);
impl Drop for Provisional<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

fn resources(
    retained: &Retained,
    request: &Request<'_>,
    binding: &Binding,
    root: u32,
    signature: &Signature,
    rows: &mut Vec<Row>,
) {
    let (exact_work, peak, floor) = component(retained, |b| {
        let floor = b.storage();
        retained
            .with_replayed_request_v2(
                request,
                native_cpu_input_v1(request, binding, root).unwrap(),
                b,
                |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(b.storage(), floor);
        (b.work(), b.peak_storage(), floor)
    });
    for (name, work, storage) in [
        ("exact-resources", exact_work, peak),
        ("work-short", exact_work - 1, peak),
        ("storage-short", exact_work, peak - 1),
    ] {
        let mut meter = Work::new(work);
        let mut b = Budget::new(&mut meter, storage);
        b.charge_work(PREFIX).unwrap();
        b.reserve_storage(floor).unwrap();
        let before = (b.work(), b.storage());
        let account = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let mut entered = false;
        let result = retained.with_replayed_request_v2(
            request,
            native_cpu_input_v1(request, binding, root).unwrap(),
            &mut b,
            |_, _| {
                entered = true;
                Ok(Provisional(&drops))
            },
        );
        let refusal = if name == "exact-resources" {
            drop(result.unwrap());
            assert!(entered);
            assert_eq!((b.work(), b.peak_storage()), (exact_work, peak));
            assert_eq!((b.failed_work(), b.failed_storage()), (None, None));
            None
        } else {
            let error = result.expect_err("short account must never return a provisional success");
            if name == "work-short" {
                assert!(b.failed_work().is_some());
            } else {
                assert!(b.failed_storage().is_some());
            }
            let denials = (b.failed_work(), b.failed_storage());
            b.charge_work(0).unwrap();
            assert_eq!((b.failed_work(), b.failed_storage()), denials);
            Some(error.to_string())
        };
        assert_eq!(drops.get(), usize::from(entered));
        assert_eq!(b.storage(), floor);
        assert!(b.work_ledger_identity_v1() == account && b.work() <= work);
        rows.push(row(name, true, before, &b, entered, drops.get(), refusal));
    }
    component(retained, |b| {
        let before = (b.work(), b.storage());
        let account = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let mut entered = false;
        let result = retained.with_replayed_request_v2(
            request,
            native_cpu_input_v1(request, binding, root).unwrap(),
            b,
            |_, b| {
                entered = true;
                b.release_storage(1)?;
                Ok(Provisional(&drops))
            },
        );
        let error = result.expect_err("damaged replay floor must override callback success");
        assert!(matches!(
            &error,
            Error::Formula(FormulaError::Resource(Resource::Accounting))
        ));
        assert!(entered && b.work_ledger_identity_v1() == account);
        assert_eq!(drops.get(), 1);
        // Fail-closed scratch stays charged on this terminal test account.
        assert!(b.storage() > before.1);
        rows.push(row(
            "replay-damaged-floor",
            true,
            before,
            b,
            entered,
            drops.get(),
            Some(error.to_string()),
        ));
    });
    component(retained, |b| {
        let before = (b.work(), b.storage());
        let account = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let mut entered = false;
        let result: Result<Provisional<'_>, Error> = (|| {
            encode(
                native_cpu_input_v1(request, binding, root).unwrap(),
                b,
                |bytes, _, b| {
                    decode(bytes, b, |decoded, b| {
                        let imported =
                            import(request, decoded, signature, retained.import_policy_v2(), b)?;
                        assert_eq!(imported.report(), retained.report());
                        let storage = imported.retained_storage_v2();
                        drop(imported);
                        b.release_storage(storage)?;
                        entered = true;
                        b.release_storage(1)?;
                        Ok(Provisional(&drops))
                    })
                    .map_err(Error::Codec)?
                },
            )
            .map_err(Error::Codec)?
        })();
        let error =
            result.expect_err("successful import is not success after codec cleanup failure");
        assert!(matches!(
            &error,
            Error::Codec(CodecError::Resource(Resource::Accounting))
        ));
        assert!(entered && b.work_ledger_identity_v1() == account);
        assert_eq!(drops.get(), 1);
        assert!(b.storage() > before.1);
        rows.push(row(
            "decode-late-cleanup",
            true,
            before,
            b,
            entered,
            drops.get(),
            Some(error.to_string()),
        ));
    });
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
    // Bound every inert test clone before any mutation/allocation traversal.
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
    assert_eq!(binding.observable_output_writes.len(), 1);
    let before = (budget.work(), budget.storage());
    let account = budget.work_ledger_identity_v1();
    let denials = (budget.failed_work(), budget.failed_storage());
    let mut rows = Vec::new();
    let signature = successes(retained, request, binding, root, budget, &mut rows);
    callbacks(retained, request, binding, root, budget, &mut rows);
    let before_components = (budget.work(), budget.storage());
    let nonroot = mutations(retained, request, binding, root, &signature, &mut rows);
    legacy_omission(retained, request, binding, root, &signature, &mut rows);
    policies(retained, request, binding, root, &signature, &mut rows);
    resources(retained, request, binding, root, &signature, &mut rows);
    assert_eq!((budget.work(), budget.storage()), before_components);
    assert_eq!(budget.storage(), before.1);
    assert!(budget.work_ledger_identity_v1() == account);
    assert_eq!((budget.failed_work(), budget.failed_storage()), denials);
    ACTIVE.lock().unwrap().as_mut().unwrap().report = Some(Report {
        schema: SCHEMA.into(),
        observations: 1,
        semantic_root: root,
        nonroot_function_index: nonroot,
        statement: *retained.report().statement_identity().as_bytes(),
        cpu_input_commitment: *retained.report().cpu_input_commitment().as_bytes(),
        original_floor_before: before.1,
        original_floor_after: budget.storage(),
        original_work_before: before.0,
        original_work_after: budget.work(),
        original_account_preserved: true,
        original_denials_preserved: true,
        rows,
        outer_postchecks_completed: false,
        foreign_account_replacement_covered: false,
        native_output_emitted: false,
        qualification_credit: false,
        grants_artifact_or_launch_authority: false,
    });
}

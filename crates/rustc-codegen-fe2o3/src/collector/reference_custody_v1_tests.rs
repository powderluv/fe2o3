//! Source252 same-session custody checks using actual rustc source and instances.
//! These scalar fixtures establish no GPU execution, output coverage, or proof authority.

use super::*;
use crate::reference_effect_v1::authenticate_reference_binding_v1;
use crate::reference_effect_v1::{ReferenceArgumentRelationV1, ReferenceScalarTypeV1};
use crate::test_temp_dir::TestTempDir;
use fe2o3_mir_model::semantic_mir_v1::{SemanticMirLimitsV1, SemanticMirResourceV1};
use rustc_driver::{Callbacks, Compilation};
use rustc_hir::def::DefKind;
use rustc_interface::interface::Compiler;

// Plain mutable kernel references have no logical ABI relation. Scalars allow
// real compiler extraction without introducing a trusted device output carrier.
const SOURCE: &str = r#"
#![allow(dead_code)]
pub fn kernel_a(seed: u32, mask: u64) { let _value = seed ^ mask as u32; }
pub fn reference_a(seed: u32, mask: u64) { let _value = seed ^ mask as u32; }
pub fn alternate_reference(seed: u32, mask: u64) { let _value = seed & mask as u32; }
pub fn kernel_b(flag: bool, value: i32) { let _value = value ^ flag as i32; }
pub fn reference_b(point: usize, flag: bool, value: i32) {
    let _value = value ^ point as i32 ^ flag as i32;
}
pub fn kernel_with_return(seed: u32, mask: u64) -> u32 { seed ^ mask as u32 }
pub fn helper(value: u32) -> u32 { value ^ 7 }
pub fn loop_kernel(seed: u32) { let _value = seed; }
pub fn loop_reference(seed: u32) {
    let mut value = seed;
    let mut index = 0_u32;
    while index < 3 {
        value ^= index;
        index += 1;
    }
    let _value = value;
}
"#;

struct CheckCallbacks(for<'tcx> fn(TyCtxt<'tcx>), bool);

impl Callbacks for CheckCallbacks {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        (self.0)(tcx);
        self.1 = true;
        Compilation::Stop
    }
}

fn with_source(check: for<'tcx> fn(TyCtxt<'tcx>)) {
    let directory = TestTempDir::create("fe2o3-reference-custody");
    let source = directory.path().join("fixture.rs");
    std::fs::write(&source, SOURCE).unwrap();
    let output = std::process::Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let sysroot = String::from_utf8(output.stdout).unwrap();
    let args = vec![
        "rustc".into(),
        "--crate-name=fe2o3_reference_custody".into(),
        "--crate-type=lib".into(),
        "--edition=2024".into(),
        "--emit=metadata".into(),
        "-Zmir-opt-level=0".into(),
        "-Copt-level=0".into(),
        "-Coverflow-checks=off".into(),
        "-Cpanic=abort".into(),
        "--sysroot".into(),
        sysroot.trim().into(),
        "-o".into(),
        directory.path().join("fixture.rmeta").display().to_string(),
        source.display().to_string(),
    ];
    let mut callbacks = CheckCallbacks(check, false);
    rustc_driver::run_compiler(&args, &mut callbacks);
    assert!(callbacks.1, "source custody callback did not run");
}

fn local<'tcx>(tcx: TyCtxt<'tcx>, name: &str) -> Instance<'tcx> {
    let definition = tcx
        .hir_body_owners()
        .find(|id| {
            tcx.def_kind(id.to_def_id()) == DefKind::Fn
                && tcx.item_name(id.to_def_id()).as_str() == name
        })
        .unwrap_or_else(|| panic!("missing fixture function {name}"));
    Instance::mono(tcx, definition.to_def_id())
}

fn unbound<'tcx>(tcx: TyCtxt<'tcx>, name: &str) -> CollectedFunction<'tcx> {
    CollectedFunction {
        instance: local(tcx, name),
        role: CollectedFunctionRole::InternalHelper,
        export_name: name.into(),
        logical_name: None,
        generated_host_contract_identity: None,
        kernel_binding: None,
        frontend_contract: None,
        reference_effect_binding: None,
        reference_instance: None,
        dead_branches: None,
        closure_observation: None,
        kernel_context_contract: None,
    }
}

fn functions<'tcx>(
    tcx: TyCtxt<'tcx>,
    work: &mut SourceClosureWorkV1,
) -> Result<Vec<CollectedFunction<'tcx>>, ReferenceBindingErrorV1> {
    let mut functions = Vec::new();
    for (kernel, reference) in [("kernel_a", "reference_a"), ("kernel_b", "reference_b")] {
        let mut function = unbound(tcx, kernel);
        let reference = local(tcx, reference);
        function.role = CollectedFunctionRole::KernelEntry;
        function.logical_name = Some(kernel.into());
        function.reference_instance = Some(reference);
        function.reference_effect_binding = Some(authenticate_reference_binding_v1(
            tcx,
            format!("source252::{kernel}_registration"),
            kernel.into(),
            function.instance,
            reference,
            work,
        )?);
        functions.push(function);
    }
    functions.insert(1, unbound(tcx, "helper"));
    Ok(functions)
}

fn assert_rejected<T>(result: Result<T, ReferenceBindingErrorV1>, reason: &str) {
    let Err(error) = result else {
        panic!("expected rejection containing {reason:?}");
    };
    assert!(error.to_string().contains(reason), "{error}");
}

#[test]
fn source_cyclic_reference_keeps_bounded_loop_support() {
    with_source(|tcx| {
        let mut function = unbound(tcx, "loop_kernel");
        let mut work = SourceClosureWorkV1::default();
        function.role = CollectedFunctionRole::KernelEntry;
        function.logical_name = Some("loop_kernel".into());
        function.reference_instance = Some(local(tcx, "loop_reference"));
        function.reference_effect_binding = Some(
            authenticate_reference_binding_v1(
                tcx,
                "source252::loop".into(),
                "loop_kernel".into(),
                function.instance,
                function.reference_instance.unwrap(),
                &mut work,
            )
            .unwrap(),
        );
        let functions = [function];
        let retained = RetainedReferenceInputsV1::capture(tcx, &functions, &mut work).unwrap();
        let fresh = retained.rederive(tcx, &functions, &mut work).unwrap();
        let summaries = &fresh.as_slice()[0].effect_ir.loop_summaries;
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].exact_iterations, Some(3));
        assert!(work.validation_work_for_test() > 0);
    });
}

#[test]
fn source_round_trip_retains_raw_signatures_and_ordered_occurrences() {
    with_source(|tcx| {
        let mut work = SourceClosureWorkV1::default();
        let functions = functions(tcx, &mut work).unwrap();
        let retained = RetainedReferenceInputsV1::capture(tcx, &functions, &mut work).unwrap();
        assert_eq!(retained.function_count, 3);
        assert_eq!(
            retained
                .inputs
                .iter()
                .map(|input| input.function)
                .collect::<Vec<_>>(),
            [0, 2]
        );
        for input in &retained.inputs {
            let function = &functions[input.function];
            let binding = function.reference_effect_binding.as_ref().unwrap();
            assert_eq!(input.kernel, function.instance);
            assert_eq!(Some(input.reference), function.reference_instance);
            assert_eq!(
                input.kernel_signature,
                instantiated_signature(tcx, input.kernel)
            );
            assert_eq!(
                input.reference_signature,
                instantiated_signature(tcx, input.reference)
            );
            assert_eq!(input.origin, binding.origin);
            assert_eq!(input.logical_kernel_name, binding.logical_kernel_name);
            assert_eq!(input.kernel_signature.output(), tcx.types.unit);
            assert_eq!(input.reference_signature.output(), tcx.types.unit);
        }
        assert_eq!(
            retained.inputs[0].kernel_signature.inputs(),
            &[tcx.types.u32, tcx.types.u64]
        );
        assert_eq!(
            retained.inputs[0].reference_signature.inputs(),
            &[tcx.types.u32, tcx.types.u64]
        );
        assert_eq!(
            retained.inputs[1].kernel_signature.inputs(),
            &[tcx.types.bool, tcx.types.i32]
        );
        assert_eq!(
            retained.inputs[1].reference_signature.inputs(),
            &[tcx.types.usize, tcx.types.bool, tcx.types.i32]
        );

        let rederived = retained.rederive(tcx, &functions, &mut work).unwrap();
        let expected = functions
            .iter()
            .filter_map(|function| function.reference_effect_binding.clone())
            .collect::<Vec<_>>();
        assert_eq!(rederived.as_slice(), expected.as_slice());
        assert_eq!(
            rederived.as_slice()[1].effect_ir.relations.as_ref(),
            &[
                ReferenceArgumentRelationV1::PointCoordinate {
                    reference_argument: 0,
                    axis: 0
                },
                ReferenceArgumentRelationV1::ScalarInput {
                    argument: 0,
                    scalar: ReferenceScalarTypeV1::Bool
                },
                ReferenceArgumentRelationV1::ScalarInput {
                    argument: 1,
                    scalar: ReferenceScalarTypeV1::I32
                },
            ]
        );
        assert!(
            rederived
                .as_slice()
                .iter()
                .all(|binding| binding.observable_output_writes.is_empty())
        );
    });
}

#[test]
fn source_empty_reference_rosters_round_trip() {
    with_source(|tcx| {
        for functions in [Vec::new(), vec![unbound(tcx, "helper")]] {
            let mut work = SourceClosureWorkV1::default();
            let retained = RetainedReferenceInputsV1::capture(tcx, &functions, &mut work).unwrap();
            assert_eq!(retained.function_count, functions.len());
            assert!(retained.inputs.is_empty());
            assert!(
                retained
                    .rederive(tcx, &functions, &mut work)
                    .unwrap()
                    .as_slice()
                    .is_empty()
            );
        }
    });
}

#[test]
fn source_capture_rejects_partial_presence_and_non_kernel_roles() {
    with_source(|tcx| {
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        for case in ["binding only", "instance only", "helper", "ffi"] {
            let mut changed = functions.clone();
            match case {
                "binding only" => changed[0].reference_instance = None,
                "instance only" => changed[0].reference_effect_binding = None,
                "helper" => changed[0].role = CollectedFunctionRole::InternalHelper,
                "ffi" => changed[0].role = CollectedFunctionRole::DeviceFfiExport,
                _ => unreachable!(),
            }
            assert_rejected(
                RetainedReferenceInputsV1::capture(
                    tcx,
                    &changed,
                    &mut SourceClosureWorkV1::default(),
                ),
                "presence or kernel role",
            );
        }
    });
}

#[test]
fn source_rederive_rejects_missing_extra_and_moved_binding_rows() {
    with_source(|tcx| {
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        let retained = RetainedReferenceInputsV1::capture(
            tcx,
            &functions,
            &mut SourceClosureWorkV1::default(),
        )
        .unwrap();
        for (binding, instance) in [(false, false), (true, false), (false, true)] {
            let mut missing = functions.clone();
            if !binding {
                missing[0].reference_effect_binding = None;
            }
            if !instance {
                missing[0].reference_instance = None;
            }
            assert_rejected(
                retained.rederive(tcx, &missing, &mut SourceClosureWorkV1::default()),
                "roster changed",
            );
        }
        for (binding, instance) in [(true, true), (true, false), (false, true)] {
            let mut extra = functions.clone();
            extra[1].role = CollectedFunctionRole::KernelEntry;
            if binding {
                extra[1].reference_effect_binding = functions[0].reference_effect_binding.clone();
            }
            if instance {
                extra[1].reference_instance = functions[0].reference_instance;
            }
            assert_rejected(
                retained.rederive(tcx, &extra, &mut SourceClosureWorkV1::default()),
                "roster changed",
            );
        }
        for additional in [false, true] {
            let mut changed = functions.clone();
            if additional {
                changed.push(unbound(tcx, "helper"));
            } else {
                changed.pop();
            }
            assert_rejected(
                retained.rederive(tcx, &changed, &mut SourceClosureWorkV1::default()),
                "function roster changed",
            );
        }
        for other in [1, 2] {
            let mut changed = functions.clone();
            changed.swap(0, other);
            assert_rejected(
                retained.rederive(tcx, &changed, &mut SourceClosureWorkV1::default()),
                "changed",
            );
        }
    });
}

#[test]
fn source_rederive_rejects_mutated_retained_occurrence_order() {
    with_source(|tcx| {
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        for case in [
            "missing first",
            "missing last",
            "extra",
            "reordered",
            "duplicate ordinal",
            "helper ordinal",
            "past end",
        ] {
            let mut retained = RetainedReferenceInputsV1::capture(
                tcx,
                &functions,
                &mut SourceClosureWorkV1::default(),
            )
            .unwrap();
            let mut inputs = retained.inputs.into_vec();
            match case {
                "missing first" => {
                    inputs.remove(0);
                }
                "missing last" => {
                    inputs.pop();
                }
                "extra" => {
                    let input = &inputs[0];
                    inputs.push(ReferenceInputV1 {
                        function: functions.len(),
                        kernel: input.kernel,
                        reference: input.reference,
                        kernel_signature: input.kernel_signature,
                        reference_signature: input.reference_signature,
                        origin: input.origin.clone(),
                        logical_kernel_name: input.logical_kernel_name.clone(),
                    });
                }
                "reordered" => inputs.swap(0, 1),
                "duplicate ordinal" => inputs[1].function = inputs[0].function,
                "helper ordinal" => inputs[0].function = 1,
                "past end" => inputs[1].function = functions.len(),
                _ => unreachable!(),
            }
            retained.inputs = inputs.into_boxed_slice();
            assert_rejected(
                retained.rederive(tcx, &functions, &mut SourceClosureWorkV1::default()),
                if case == "extra" {
                    "occurrence is missing"
                } else {
                    "roster changed"
                },
            );
        }
    });
}

#[test]
fn source_rederive_rejects_instance_role_registration_and_name_mutations() {
    with_source(|tcx| {
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        let retained = RetainedReferenceInputsV1::capture(
            tcx,
            &functions,
            &mut SourceClosureWorkV1::default(),
        )
        .unwrap();
        for case in [
            "kernel",
            "reference",
            "helper",
            "ffi",
            "registration",
            "binding name",
            "logical name",
            "missing name",
            "both names",
        ] {
            let mut changed = functions.clone();
            let function = &mut changed[0];
            match case {
                "kernel" => function.instance = local(tcx, "kernel_with_return"),
                "reference" => {
                    function.reference_instance = Some(local(tcx, "alternate_reference"))
                }
                "helper" => function.role = CollectedFunctionRole::InternalHelper,
                "ffi" => function.role = CollectedFunctionRole::DeviceFfiExport,
                "registration" => {
                    let ReferenceBindingOriginV1::SourceRegistration(path) =
                        &mut function.reference_effect_binding.as_mut().unwrap().origin
                    else {
                        panic!("fixture must retain source registration");
                    };
                    path.push_str("::changed");
                }
                "binding name" => function
                    .reference_effect_binding
                    .as_mut()
                    .unwrap()
                    .logical_kernel_name
                    .push_str("_changed"),
                "logical name" => function.logical_name = Some("changed".into()),
                "missing name" => function.logical_name = None,
                "both names" => {
                    function.logical_name = Some("changed".into());
                    function
                        .reference_effect_binding
                        .as_mut()
                        .unwrap()
                        .logical_kernel_name = "changed".into();
                }
                _ => unreachable!(),
            }
            assert_rejected(
                retained.rederive(tcx, &changed, &mut SourceClosureWorkV1::default()),
                "instance, role or registration changed",
            );
        }
    });
}

#[test]
fn source_rederive_checks_raw_signatures_beyond_logical_projection() {
    with_source(|tcx| {
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        let different_return = local(tcx, "kernel_with_return");
        let projected = authenticate_reference_binding_v1(
            tcx,
            "source252::return_shape".into(),
            "return_shape".into(),
            different_return,
            local(tcx, "reference_a"),
            &mut SourceClosureWorkV1::default(),
        )
        .unwrap();
        assert_eq!(
            projected.signature_preimage,
            functions[0]
                .reference_effect_binding
                .as_ref()
                .unwrap()
                .signature_preimage
        );

        for kernel in [true, false] {
            let mut retained = RetainedReferenceInputsV1::capture(
                tcx,
                &functions,
                &mut SourceClosureWorkV1::default(),
            )
            .unwrap();
            if kernel {
                let signature = instantiated_signature(tcx, different_return);
                assert_eq!(
                    signature.inputs(),
                    retained.inputs[0].kernel_signature.inputs()
                );
                assert_ne!(
                    signature.output(),
                    retained.inputs[0].kernel_signature.output()
                );
                retained.inputs[0].kernel_signature = signature;
            } else {
                retained.inputs[0].reference_signature =
                    instantiated_signature(tcx, local(tcx, "reference_b"));
            }
            assert_rejected(
                retained.rederive(tcx, &functions, &mut SourceClosureWorkV1::default()),
                "instantiated signature changed",
            );
        }
    });
}

#[test]
fn source_rederive_rejects_stale_compiler_binding_fields() {
    with_source(|tcx| {
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        let retained = RetainedReferenceInputsV1::capture(
            tcx,
            &functions,
            &mut SourceClosureWorkV1::default(),
        )
        .unwrap();
        for case in [
            "kernel identity",
            "reference identity",
            "preimage",
            "effect digest",
            "effect body",
            "relation order",
        ] {
            let mut changed = functions.clone();
            let binding = changed[0].reference_effect_binding.as_mut().unwrap();
            match case {
                "kernel identity" => binding.kernel.function_sha256[0] ^= 1,
                "reference identity" => binding.reference.rustc_mir_body_sha256[0] ^= 1,
                "preimage" => {
                    binding.signature_preimage = functions[2]
                        .reference_effect_binding
                        .as_ref()
                        .unwrap()
                        .signature_preimage
                        .clone()
                }
                "effect digest" => binding.effect_ir_sha256[0] ^= 1,
                "effect body" => binding.effect_ir.argument_count += 1,
                "relation order" => binding.effect_ir.relations.swap(0, 1),
                _ => unreachable!(),
            }
            assert_rejected(
                retained.rederive(tcx, &changed, &mut SourceClosureWorkV1::default()),
                "differs from fresh compiler extraction",
            );
        }
    });
}

fn check_work_boundaries<T>(
    operation: impl Fn(&mut SourceClosureWorkV1) -> Result<T, ReferenceBindingErrorV1>,
) {
    let mut measured = SourceClosureWorkV1::default();
    operation(&mut measured).unwrap_or_else(|error| panic!("measuring source work: {error}"));
    let cost = measured.validation_work_for_test();
    assert!(cost > 0);
    let limit = SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::ValidationWork);
    for inherited in [0, 17] {
        let mut work = SourceClosureWorkV1::default();
        work.charge(inherited).unwrap();
        operation(&mut work).unwrap_or_else(|error| panic!("inherited source work: {error}"));
        assert_eq!(work.validation_work_for_test(), inherited as u64 + cost);
    }
    for remaining in [cost, cost - 1, 0] {
        let mut work = SourceClosureWorkV1::default();
        work.charge(17).unwrap();
        work.charge(usize::try_from(limit - work.validation_work_for_test() - remaining).unwrap())
            .unwrap();
        let result = operation(&mut work);
        if remaining == cost {
            result.unwrap_or_else(|error| panic!("exact source work: {error}"));
            assert_eq!(work.validation_work_for_test(), limit);
            assert!(work.charge(1).is_err());
        } else {
            assert_rejected(result, "ValidationWork");
        }
    }
}

#[test]
fn source_authentication_capture_and_rederive_share_exact_cumulative_work() {
    with_source(|tcx| {
        check_work_boundaries(|work| functions(tcx, work));
        let functions = functions(tcx, &mut SourceClosureWorkV1::default()).unwrap();
        check_work_boundaries(|work| RetainedReferenceInputsV1::capture(tcx, &functions, work));
        let retained = RetainedReferenceInputsV1::capture(
            tcx,
            &functions,
            &mut SourceClosureWorkV1::default(),
        )
        .unwrap();
        check_work_boundaries(|work| retained.rederive(tcx, &functions, work));
        check_work_boundaries(|work| {
            let functions = self::functions(tcx, work)?;
            let retained = RetainedReferenceInputsV1::capture(tcx, &functions, work)?;
            let first = retained.rederive(tcx, &functions, work)?;
            let before = work.validation_work_for_test();
            let second = retained.rederive(tcx, &functions, work)?;
            assert_eq!(first.as_slice(), second.as_slice());
            assert!(work.validation_work_for_test() > before);
            Ok(second)
        });
    });
}

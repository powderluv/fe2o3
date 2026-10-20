//! Live scalar rustc instances under the existing original-loan process fixture.
//! This stops at preflight: it does not authenticate an AMDGPU production target,
//! collect registration statics, install a backend DSO, or publish a recipe.
use super::*;
use crate::collector::reference_custody_v1::RetainedReferenceInputsV1;
use crate::collector::{CollectedFunction, KernelRoot, reference_enrollment_v1};
use crate::protected_compiler_execution::native_v3::{Admitted, Error};
use crate::protected_rustc_invocation::AdmittedProtectedRustcInvocationV1 as Invocation;
use crate::reference_effect_v1::authenticate_reference_binding_v1;
use crate::semantic_layout_bridge::rustc_semantic_layout_target_v1;
use crate::test_temp_dir::TestTempDir;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_middle::mir::mono::MonoItem;
use sha2::{Digest, Sha256};

// The GPU FnAbi query needs the real AMD target. This no-core identity fixture
// does not replace authentic device dependencies or production source admission.
const SOURCE: &str = r#"
#![feature(no_core, lang_items)]
#![no_core]
#![allow(dead_code, internal_features)]
#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}
#[inline(never)]
pub fn generic_kernel<const N: u32>() {}
#[inline(never)]
pub fn generic_reference<const N: u32>() {}
pub fn anchor() {
    generic_kernel::<3>();
    generic_kernel::<5>();
    generic_reference::<3>();
    generic_reference::<5>();
}
"#;

struct Call<F> {
    check: F,
    calls: usize,
}
impl<F: for<'tcx> FnMut(TyCtxt<'tcx>)> Callbacks for Call<F> {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        self.calls += 1;
        (self.check)(tcx);
        Compilation::Stop
    }
}

fn with_source(check: impl for<'tcx> FnMut(TyCtxt<'tcx>) + Send) {
    let directory = TestTempDir::create("fe2o3-original-root-inventory");
    let source = directory.path().join("fixture.rs");
    let mut source_text = SOURCE.to_owned();
    for index in 0..16 {
        for prefix in ["kernel", "reference"] {
            source_text.push_str(&format!("\npub fn {prefix}_{index:02}() {{}}\n"));
        }
    }
    std::fs::write(&source, source_text).unwrap();
    // This also runs before the child invocation is sealed to choose a genuinely
    // reversed semantic order. Descriptor order remains lexical as required.
    // Use the linked driver, not PATH/rustup.
    let executable = std::path::PathBuf::from(std::env::args_os().next().unwrap());
    assert!(executable.is_absolute());
    assert_eq!(executable, executable.canonicalize().unwrap());
    let sysroot = rustc_session::config::Sysroot::new(None);
    let args = vec![
        "rustc".into(),
        "--crate-name=fe2o3_original_root_inventory".into(),
        "--crate-type=lib".into(),
        "--edition=2024".into(),
        "--target=amdgcn-amd-amdhsa".into(),
        "-Ctarget-cpu=gfx942".into(),
        "-Ctarget-feature=-xnack,+wavefrontsize64,-wavefrontsize32".into(),
        "--emit=obj".into(),
        "-Ccodegen-units=1".into(),
        "-Zmir-opt-level=0".into(),
        "-Copt-level=0".into(),
        "-Coverflow-checks=off".into(),
        "-Cpanic=abort".into(),
        "--sysroot".into(),
        sysroot.path().display().to_string(),
        "-o".into(),
        directory.path().join("fixture.o").display().to_string(),
        source.display().to_string(),
    ];
    let mut callbacks = Call { check, calls: 0 };
    rustc_driver::run_compiler(&args, &mut callbacks);
    assert_eq!(callbacks.calls, 1);
    assert!(!directory.path().join("fixture.o").exists());
}

fn local<'tcx>(tcx: TyCtxt<'tcx>, name: &str) -> Instance<'tcx> {
    let definition = tcx
        .hir_body_owners()
        .find(|id| {
            tcx.def_kind(id.to_def_id()) == rustc_hir::def::DefKind::Fn
                && tcx.item_name(id.to_def_id()).as_str() == name
        })
        .unwrap();
    assert_eq!(tcx.generics_of(definition).count(), 0);
    Instance::mono(tcx, definition.to_def_id())
}

fn selected_pairs(tcx: TyCtxt<'_>) -> [(String, String); 2] {
    let candidates = (0..16)
        .map(|index| {
            let name = format!("kernel_{index:02}");
            let identity = canonical_function_identities_v1(tcx, local(tcx, &name)).function();
            (name, format!("reference_{index:02}"), identity)
        })
        .collect::<Vec<_>>();
    for (index, left) in candidates.iter().enumerate() {
        for right in &candidates[index + 1..] {
            if left.2 > right.2 {
                assert!(left.0 < right.0);
                return [
                    (left.0.clone(), left.1.clone()),
                    (right.0.clone(), right.1.clone()),
                ];
            }
        }
    }
    panic!("bounded real fixture roster has no lexical/canonical inversion");
}

pub(crate) fn reversed_original_root_inventory_request() -> String {
    let mut request = None;
    with_source(|tcx| {
        let pairs = selected_pairs(tcx);
        let bindings = pairs
            .iter()
            .map(|(kernel, reference)| {
                serde_json::json!({
                    "kernel": format!("fe2o3_original_root_inventory::{kernel}"),
                    "reference": format!("fe2o3_original_root_inventory::{reference}")
                })
            })
            .collect::<Vec<_>>();
        request = Some(serde_json::json!({"version": 1, "bindings": bindings}).to_string());
    });
    request.unwrap()
}

fn root<'tcx>(instance: Instance<'tcx>, name: String) -> KernelRoot<Instance<'tcx>> {
    KernelRoot {
        target: instance,
        logical_name: name.clone(),
        export_name: name,
        generated_host_contract_identity: None,
        // Binding IDs are inert carrier metadata; actual identity and custody
        // come from the live instances and the existing authentication calls.
        kernel_binding: Some(reserved_fe2o3_symbols::KernelBindingIdV1::from_bytes(
            [29; 32],
        )),
        frontend_contract: None,
        kernel_context_contract: None,
        reference_effect_binding: None,
        reference_target: None,
    }
}

fn collected<'tcx>(root: KernelRoot<Instance<'tcx>>) -> CollectedFunction<'tcx> {
    CollectedFunction {
        instance: root.target,
        role: CollectedFunctionRole::KernelEntry,
        export_name: root.export_name,
        logical_name: Some(root.logical_name),
        generated_host_contract_identity: root.generated_host_contract_identity,
        kernel_binding: root.kernel_binding,
        frontend_contract: root.frontend_contract,
        reference_effect_binding: root.reference_effect_binding,
        reference_instance: root.reference_target,
        dead_branches: None,
        closure_observation: None,
        kernel_context_contract: None,
    }
}

// Independent, fixed V2 framing for this no-frontend-contract scalar fixture.
// Do not call the production field visitor: this protects the old byte contract.
fn legacy_golden(
    target: SemanticTargetDataLayoutV1,
    functions: &[RetainedSemanticFunctionProducerV1<'_>],
    roots: &[SemanticFunctionIdV1],
) -> Vec<u8> {
    fn field(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }
    let mut out = Vec::new();
    field(&mut out, b"fe2o3/semantic-mir/rustc-identity-inventory/v2");
    field(&mut out, target.identity().as_bytes());
    for function in functions {
        for identity in [
            function.identities.function().as_bytes(),
            function.identities.item_definition().as_bytes(),
            function.identities.monomorphization().as_bytes(),
            function.identities.generic_type_arguments().as_bytes(),
            function.identities.const_generic_arguments().as_bytes(),
        ] {
            field(&mut out, identity);
        }
        assert_eq!(function.role, CollectedFunctionRole::KernelEntry);
        field(&mut out, &[0]);
        field(&mut out, &[1]);
        field(&mut out, function.export_name.as_ref().unwrap().as_bytes());
        field(&mut out, &[1]);
        field(&mut out, &function.kernel_binding.unwrap().as_bytes());
        assert!(function.generated_host_contract_identity.is_none());
        assert!(function.frontend_contract.is_none());
        field(&mut out, &[0]);
        field(&mut out, &[0]);
    }
    for root in roots {
        field(&mut out, &root.index().to_le_bytes());
    }
    out
}

pub(crate) fn check_original_root_inventory_flow(
    native: Admitted<'_, '_>,
    invocation: &mut Invocation,
    enrolled: bool,
) {
    let mut native = Some(native);
    with_source(|tcx| {
        let cgus = tcx.collect_and_partition_mono_items(()).codegen_units;
        let mut source = SourceClosureWorkV1::default();
        source.charge(17).unwrap();
        let mut registered = Vec::new();
        for cgu in cgus {
            for item in cgu.items().keys() {
                let MonoItem::Fn(instance) = item else {
                    continue;
                };
                if tcx.item_name(instance.def_id()).as_str() == "generic_kernel"
                    && !registered.contains(instance)
                {
                    registered.push(*instance);
                }
            }
        }
        assert_eq!(registered.len(), 2);
        assert_eq!(registered[0].def_id(), registered[1].def_id());
        assert_ne!(registered[0].args, registered[1].args);
        let mut roots = Vec::new();
        for (index, instance) in registered.iter().enumerate() {
            let reference = cgus
                .iter()
                .flat_map(|cgu| cgu.items().keys())
                .filter_map(|item| match item {
                    MonoItem::Fn(reference)
                        if tcx.item_name(reference.def_id()).as_str() == "generic_reference"
                            && reference.args == instance.args =>
                    {
                        Some(*reference)
                    }
                    _ => None,
                })
                .next()
                .unwrap();
            let mut registered_root = root(*instance, format!("registered_{index}"));
            registered_root.reference_effect_binding = Some(
                authenticate_reference_binding_v1(
                    tcx,
                    format!("fixture::registration_{index}"),
                    registered_root.logical_name.clone(),
                    *instance,
                    reference,
                    &mut source,
                )
                .unwrap(),
            );
            registered_root.reference_target = Some(reference);
            roots.push(registered_root);
        }
        for (name, reference_name) in selected_pairs(tcx) {
            let mut scalar = root(local(tcx, &name), name.clone());
            if !enrolled {
                let reference = local(tcx, &reference_name);
                scalar.reference_effect_binding = Some(
                    authenticate_reference_binding_v1(
                        tcx,
                        format!("fixture::{name}_registration"),
                        name.clone(),
                        scalar.target,
                        reference,
                        &mut source,
                    )
                    .unwrap(),
                );
                scalar.reference_target = Some(reference);
            }
            roots.push(scalar);
        }
        let (session, (functions, retained)) = native
            .take()
            .unwrap()
            .with_reference_enrollment::<_, Error>(invocation, |loan| {
                let loan = loan.unwrap();
                let enrollment =
                    reference_enrollment_v1::bind_v1(tcx, cgus, &mut roots, loan, &mut source)?;
                let functions = roots.into_iter().map(collected).collect::<Vec<_>>();
                let retained = RetainedReferenceInputsV1::capture_with_enrollment(
                    tcx,
                    &functions,
                    &mut source,
                    Some(enrollment),
                    Some(loan),
                )?;
                Ok((functions, retained))
            })
            .unwrap();
        let (session, ()) = session
            .with_reference_enrollment::<_, Error>(invocation, |loan| {
                let loan = loan.unwrap();
                // Same binding and definition do not authorize a different
                // actual substitution on either side of the retained pair.
                for change_reference in [false, true] {
                    let mut changed = functions.clone();
                    if change_reference {
                        changed[0].reference_instance = changed[1].reference_instance;
                    } else {
                        changed[0].instance = changed[1].instance;
                    }
                    assert!(
                        retained
                            .rederive_with_inventory_capture(tcx, &changed, &mut source, Some(loan))
                            .is_err()
                    );
                }
                let (_, pending) = retained.rederive_with_inventory_capture(
                    tcx,
                    &functions,
                    &mut source,
                    Some(loan),
                )?;
                let mut canonical = functions
                    .iter()
                    .map(|function| RetainedSemanticFunctionProducerV1 {
                        identities: canonical_function_identities_v1(tcx, function.instance),
                        instance: function.instance,
                        role: function.role,
                        export_name: Some(function.export_name.clone()),
                        kernel_binding: function.kernel_binding,
                        generated_host_contract_identity: None,
                        frontend_contract: None,
                    })
                    .collect::<Vec<_>>();
                canonical.sort_by_key(|function| function.identities.function());
                let semantic_roots = (0..canonical.len())
                    .map(|i| SemanticFunctionIdV1::from_index(i as u32))
                    .collect::<Vec<_>>();
                let target =
                    canonical_target_layout_v1(&rustc_semantic_layout_target_v1(tcx).unwrap());
                let golden = legacy_golden(target, &canonical, &semantic_roots);
                let (legacy_sha, legacy) = identity_inventory_identity_and_transcript_v1(
                    target,
                    &canonical,
                    &semantic_roots,
                );
                assert_eq!(&*legacy, golden);
                assert_eq!(legacy_sha, <[u8; 32]>::from(Sha256::digest(&golden)));
                let (sha, transcript) = if enrolled {
                    let mut associations = pending.unwrap().seal(
                        tcx,
                        loan,
                        &mut source,
                        &canonical,
                        &semantic_roots,
                    )?;
                    assert_eq!(associations.header().kernel_count, 4);
                    assert_eq!(associations.header().enrollment_binding_count, 2);
                    assert_eq!(
                        associations
                            .roots()
                            .iter()
                            .filter(|row| row.origin_tag == 1)
                            .map(|row| row.descriptor_ordinal)
                            .collect::<Vec<_>>(),
                        [1, 0]
                    );
                    for row in associations.roots() {
                        let kernel = &canonical[row.semantic_root as usize];
                        let original = functions
                            .iter()
                            .find(|function| function.instance == kernel.instance)
                            .unwrap();
                        assert_eq!(
                            row.kernel_instance,
                            *canonical_function_identities_v1(tcx, original.instance)
                                .function()
                                .as_bytes()
                        );
                        assert_eq!(
                            row.reference_instance,
                            *canonical_function_identities_v1(
                                tcx,
                                original.reference_instance.unwrap()
                            )
                            .function()
                            .as_bytes()
                        );
                        assert_eq!(row.kernel_binding, [29; 32]);
                    }
                    associations.check_paid_spare_capacity_for_test(loan);
                    let output = enrolled_identity_inventory_transcript_v1(
                        target,
                        &canonical,
                        &semantic_roots,
                        &associations,
                        loan,
                        &mut source,
                    )
                    .unwrap();
                    let decoded = fe2o3_compiler_lineage::read_rustc_enrollment_inventory_v1(
                        &output.1,
                        loan.inventory_storage_limit(),
                        |amount| source.charge(amount),
                    )
                    .unwrap();
                    assert_eq!(decoded.legacy_inventory(), golden);
                    assert_eq!(decoded.full_wrapper_sha256(), &output.0);
                    assert_eq!(output.0, <[u8; 32]>::from(Sha256::digest(&output.1)));
                    assert_ne!(decoded.framing_sha256(), &output.0);
                    output
                } else {
                    assert!(pending.is_none());
                    (legacy_sha, legacy)
                };
                // The same SOURCE account is consumed by the actual preflight.
                let plan = build_production_semantic_preflight_plan_with_work_v1(
                    tcx,
                    target,
                    canonical.into_boxed_slice(),
                    semantic_roots.into_boxed_slice(),
                    sha,
                    DebugSourceCaptureRequestV2::Disabled,
                    (None, source),
                )
                .unwrap();
                let mut preflight = plan.canonical_transcript();
                let mut fields = Vec::new();
                for _ in 0..4 {
                    let size = u64::from_le_bytes(preflight[..8].try_into().unwrap()) as usize;
                    fields.push(&preflight[8..8 + size]);
                    preflight = &preflight[8 + size..];
                }
                assert_eq!(fields[3], sha);
                assert_eq!(sha, <[u8; 32]>::from(Sha256::digest(&transcript)));
                assert_eq!(plan.function_producers().len(), 4);
                Ok(())
            })
            .unwrap();
        drop(session);
    });
    assert!(native.is_none());
}

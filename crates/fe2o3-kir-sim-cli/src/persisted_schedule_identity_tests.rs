//! Synthetic mutation controls for persisted schedule artifact identity.
use super::*;
use fe2o3_kernel_ir::{
    BasicBlock, BlockId, Function, Kernel, LaunchDomain, LaunchExtent, Module, Signature,
    Terminator, VerifiedCanonicalKernelIrV11,
};

fn module(name: &str) -> AdmittedSimulationModuleV1 {
    let mut block = BasicBlock::new(BlockId(0));
    block.terminator = Some(Terminator::Return { values: vec![] });
    let mut module = Module::new(name);
    module.functions.push(Function::kernel_entry(
        "entry",
        Signature::new(vec![], vec![]),
        vec![],
        vec![block],
    ));
    module.kernels.push(Kernel::new(
        "kernel",
        "entry",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    ));
    AdmittedSimulationModuleV1::admit_v11(
        VerifiedCanonicalKernelIrV11::from_module(module).unwrap(),
        SimulationLimitsV1::default(),
    )
    .unwrap()
}

fn input(bundle: bool) -> AdmittedSimulationInputV1 {
    let module = module("original");
    let kir_sha256 = *module.identity().digest();
    let evidence = bundle.then_some(AdmittedSimulationBundleEvidenceV1 {
        envelope_version: 6,
        envelope_identity: [1; 32],
        subject_identity: [2; 32],
        production_kir_version: 11,
        production_kir_sha256: kir_sha256,
        production_kir_bytes: module.identity().canonical_length(),
        kernel_abi_identity: [3; 32],
        identity_inventory_receipt_sha256: [4; 32],
        identity_inventory_receipt_bytes: 1,
        preflight_plan_receipt_sha256: [5; 32],
        preflight_plan_receipt_bytes: 1,
    });
    AdmittedSimulationInputV1 {
        simulation_bundle_kir_identity: bundle.then_some(*module.identity()),
        module,
        request: SimulationRequestV1::new("kernel", [1; 3], [1; 3], vec![]),
        simulation_limits: SimulationLimitsV1::default(),
        simulation_target: SimulationTargetV1::amdgpu_64(),
        simulation_bundle_subject: evidence.map(|e| e.subject_identity),
        simulation_bundle_identity: evidence.map(|e| e.envelope_identity),
        simulation_bundle_evidence: evidence,
        bundle_target_v1: None,
        kir_sha256,
        request_sha256: [6; 32],
        request_bytes: 1,
    }
}

fn refused(input: &AdmittedSimulationInputV1) {
    let error = input.persisted_schedule_binding().unwrap_err();
    assert_eq!(error.stage, "arguments");
    assert_eq!(error.code, "schedule_input_unsupported");
    assert!(error.message.contains("identity"));
}

#[test]
fn unchanged_raw_and_bundle_inputs_keep_exact_bindings() {
    for bundle in [false, true] {
        let input = input(bundle);
        let binding = input.persisted_schedule_binding().unwrap();
        assert_eq!(binding.kir_sha256(), input.kir_sha256);
        assert_eq!(
            binding.kir_canonical_bytes(),
            input.module.identity().canonical_length()
        );
        assert_eq!(binding.request_sha256(), input.request_sha256);
        assert_eq!(binding.kir_wire_version(), 11);
    }
}

#[test]
fn raw_module_replacement_is_refused_before_schedule_file_read() {
    let mut input = input(false);
    input.module = module("replacement");
    assert_ne!(*input.module.identity().digest(), input.kir_sha256);
    refused(&input);
    #[cfg(target_os = "linux")]
    {
        let error = load_debug_simulation_schedule_v1(
            Path::new("/missing-fe2o3-replay-identity-control/schedule.json"),
            &input,
        )
        .unwrap_err();
        assert_eq!(error.stage, "arguments");
        assert!(error.message.contains("identity"));
    }
}

#[test]
fn changed_public_kir_digest_is_refused() {
    let mut input = input(false);
    input.kir_sha256[0] ^= 1;
    refused(&input);
}

#[test]
fn bundle_module_and_public_digest_replacement_cannot_relabel_retained_bundle() {
    let mut input = input(true);
    input.module = module("replacement");
    input.kir_sha256 = *input.module.identity().digest();
    refused(&input);
}

#[test]
fn inconsistent_retained_bundle_identity_references_are_refused() {
    for field in 0..2 {
        let mut input = input(true);
        let evidence = input.simulation_bundle_evidence.as_mut().unwrap();
        match field {
            0 => evidence.envelope_identity[0] ^= 1,
            1 => evidence.subject_identity[0] ^= 1,
            _ => unreachable!(),
        }
        refused(&input);
    }
}

#[test]
fn independently_readmitted_identical_module_preserves_binding() {
    let mut input = input(true);
    let original = input.persisted_schedule_binding().unwrap();
    input.module = module("original");
    assert_eq!(input.persisted_schedule_binding().unwrap(), original);
}

#[test]
fn same_length_bundle_module_substitution_is_refused() {
    let mut input = input(true);
    let original_length = input.module.identity().canonical_length();
    input.module = module("changed!");
    assert_eq!(input.module.identity().canonical_length(), original_length);
    input.kir_sha256 = *input.module.identity().digest();
    assert_ne!(
        input.kir_sha256,
        input
            .simulation_bundle_evidence
            .unwrap()
            .production_kir_sha256
    );
    refused(&input);
}

#[test]
fn explicitly_rebound_raw_module_keeps_its_new_exact_identity() {
    let mut input = input(false);
    input.module = module("changed!");
    input.kir_sha256 = *input.module.identity().digest();
    let binding = input.persisted_schedule_binding().unwrap();
    assert_eq!(binding.kir_sha256(), input.kir_sha256);
    assert_eq!(
        binding.artifact(),
        PersistedSimulationScheduleArtifactV1::CanonicalKirV11
    );
}

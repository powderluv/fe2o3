//! Public inert framing only. Adapted from the maintained FFI V5 fixture.
//! No proof admission, publication, lease, compiler owner or currentness authority.
use super::*;
use fe2o3_build_authority::CompilerClosureV2;
use fe2o3_compiler_ffi::*;
use fe2o3_compiler_lineage::*;
use fe2o3_rustc_invocation::{
    CompileEnvironmentV2, RustcInvocationDescriptorV2, RustcInvocationDescriptorV3, RustcUnitV2,
};

pub(super) fn handoff() -> Handoff {
    handoff_with_enrollment(None, |_| b"host-inert-inventory".to_vec())
}

pub(super) fn handoff_with_enrollment(
    enrollment: Option<&str>,
    inventory: impl FnOnce(&[u8]) -> Vec<u8>,
) -> Handoff {
    let profile = fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942;
    let target = DeviceTargetV1::parse("gfx942:xnack-").unwrap();
    let pins = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32]];
    let closure =
        CompilerClosureV2::new(pins[0], pins[1], pins[2], pins[3], pins[4], pins[5]).unwrap();
    let rustc = RustcUnitV2::new(
        "/workspace/fe2o3",
        vec![
            "/opt/fe2o3/rustc".into(),
            "--crate-name".into(),
            "host_inert_raw_gate".into(),
            "crates/host-inert-fixture/src/lib.rs".into(),
            "--crate-type=lib".into(),
            "--edition=2024".into(),
            "-Zcodegen-backend=/opt/fe2o3/librustc_codegen_fe2o3.so".into(),
        ],
    )
    .unwrap();
    let environment = CompileEnvironmentV2::from_child_environment(
        [
            ("CARGO_CFG_TARGET_ARCH", "amdgcn"),
            ("FE2O3_HSACO_DIR", "/workspace/fe2o3/target/fe2o3"),
            ("FE2O3_TARGET", "gfx942:xnack-"),
            ("FE2O3_VERIFY_KERNEL_IR", "1"),
        ]
        .into_iter()
        .chain(
            enrollment
                .map(|request| (fe2o3_rustc_invocation::REFERENCE_ENROLLMENT_ENV_V1, request)),
        )
        .map(|(key, value)| (key.into(), value.into())),
    )
    .unwrap();
    let invocation = RustcInvocationDescriptorV3::new(
        RustcInvocationDescriptorV2::new(pins[3], pins[5], rustc, environment).unwrap(),
        closure,
    )
    .unwrap();
    let invocation = fe2o3_rustc_invocation::encode_descriptor_v3(&invocation).unwrap();
    let inventory = inventory(&invocation);
    let module = CompilerModuleHandoffV2::new(
        CompilerModuleKindV1::LlvmTextIr,
        target,
        CodeObjectVersion::V6,
        CompilerFfiEnvelopeV1::for_module_without_device_ffi(target, CodeObjectVersion::V6)
            .unwrap(),
        CompilerModuleSymbolManifestV1::new([
            (CompilerModuleSymbolRoleV1::KernelEntry, "kernel"),
            (CompilerModuleSymbolRoleV1::KernelDescriptor, "kernel.kd"),
        ])
        .unwrap(),
        b"define amdgpu_kernel void @kernel() { ret void }\n",
    )
    .unwrap();
    let limit = MAX_NATIVE_CONDITIONAL_STORAGE_V1;
    let output_layout = NativeConditionalOutputLayoutV1::new::<()>(7, 7, 10).unwrap();
    let carrier_layout =
        NativeConditionalCarrierLayoutV1::new::<()>(output_layout.encoded_len(), 6).unwrap();
    let mut carrier = vec![0; carrier_layout.encoded_len()];
    let output = &mut carrier[carrier_layout.output_range()];
    output[output_layout.history_range()].copy_from_slice(b"history");
    output[output_layout.catalog_range()].copy_from_slice(b"catalog");
    output[output_layout.descriptor_range()].copy_from_slice(b"descriptor");
    seal_native_conditional_output_v1(output_layout, output, limit, |_| Ok::<_, ()>(())).unwrap();
    carrier[carrier_layout.source_range()].copy_from_slice(b"source");
    let carrier_id = seal_native_conditional_carrier_v1(
        carrier_layout,
        &mut carrier,
        limit,
        |_| Ok::<_, ()>(()),
    )
    .unwrap();
    let axis = |hash, len| TargetLineageIdentityV3::new(hash, len).unwrap();
    let lowering = InertNativeLoweringAssociationV1::new(NativeLoweringAssociationInputsV1 {
        final_native: InertNativeNeutralSubjectV1::new([1; 32], 7, [2; 32], 7).unwrap(),
        carrier: axis(*carrier_id.sha256(), carrier_id.byte_len()),
        descriptor: axis(Sha256::digest(b"descriptor").into(), 10),
        pre_descriptor_llvm: axis([3; 32], 1),
        final_llvm: axis(
            *module.module_identity().sha256(),
            module.module_identity().byte_len(),
        ),
        module_handoff: axis(*module.identity().sha256(), module.identity().byte_len()),
        profile,
    });
    let target_layout = canonical_semantic_target_layout_transcript_v1(
        profile.rustc_target(),
        "layout",
        64,
        profile.cpu(),
        profile.rustc_features(),
    )
    .unwrap();
    let commitment = InertFinalCompilerModuleCommitmentV3::from_handoff(&module).unwrap();
    let input = NativeConditionalMetadataInputV1 {
        invocation: &invocation,
        rustc_inventory: &inventory,
        rustc_preflight: b"host-inert-preflight",
        semantic_target_layout: &target_layout,
        native_lowering: lowering.canonical_bytes(),
        final_module_commitment: commitment.canonical_bytes(),
    };
    let metadata_layout = NativeConditionalMetadataLayoutV1::new::<()>(input).unwrap();
    let roots = [NativeConditionalPolicyRootInputV1 {
        semantic_root: 9,
        kernel_binding: [1; 32],
        effect_signers: &[[2; 32]],
        effect_toolchain: [[3; 32]; 5],
        formula_verifying_key: [4; 32],
        formula_toolchain: [[5; 32]; 5],
        formula_boundary: 1,
    }];
    let roster = encode_native_conditional_policy_roster_v1(
        NativeConditionalPolicyRosterInputV1 {
            source_packet: b"source",
            roots: &roots,
        },
        limit,
        |_| Ok::<_, ()>(()),
    )
    .unwrap();
    let envelope_layout =
        NativeConditionalMetadataLayoutV2::new::<()>(metadata_layout.encoded_len(), roster.len())
            .unwrap();
    let capsule_layout = InertProductionSemanticCapsuleLayoutV5::new::<()>(
        envelope_layout.encoded_len(),
        carrier.len(),
    )
    .unwrap();
    let mut bytes = vec![0; capsule_layout.encoded_len()];
    let envelope = &mut bytes[capsule_layout.metadata_range()];
    let metadata = &mut envelope[envelope_layout.metadata_range()];
    for (range, payload) in [
        (metadata_layout.invocation_range(), input.invocation),
        (
            metadata_layout.rustc_inventory_range(),
            input.rustc_inventory,
        ),
        (
            metadata_layout.rustc_preflight_range(),
            input.rustc_preflight,
        ),
        (
            metadata_layout.semantic_target_layout_range(),
            input.semantic_target_layout,
        ),
        (
            metadata_layout.native_lowering_range(),
            input.native_lowering,
        ),
        (
            metadata_layout.final_module_commitment_range(),
            input.final_module_commitment,
        ),
    ] {
        metadata[range].copy_from_slice(payload);
    }
    seal_native_conditional_metadata_v1(metadata_layout, metadata, limit, |_| Ok::<_, ()>(()))
        .unwrap();
    envelope[envelope_layout.policy_roster_range()].copy_from_slice(&roster);
    seal_native_conditional_metadata_v2(envelope_layout, envelope, limit, |_| Ok::<_, ()>(()))
        .unwrap();
    bytes[capsule_layout.carrier_range()].copy_from_slice(&carrier);
    seal_inert_production_semantic_capsule_v5(capsule_layout, &mut bytes, limit, |_| {
        Ok::<_, ()>(())
    })
    .unwrap();
    let capsule = InertProductionSemanticCapsuleV5::decode_owned(bytes).unwrap();
    let layout = preflight_inert_semantic_compiler_module_handoff_v5(&capsule, &module).unwrap();
    let mut bytes = vec![0; layout.encoded_len()];
    bytes[layout.capsule_range()].copy_from_slice(capsule.canonical_bytes());
    bytes[layout.module_handoff_range()].copy_from_slice(module.canonical_bytes());
    seal_inert_semantic_compiler_module_handoff_v5(
        layout,
        &mut bytes,
        capsule.identity(),
        module.identity(),
        |_| Ok::<_, ()>(()),
    )
    .unwrap();
    Handoff::decode_owned(bytes).unwrap()
}

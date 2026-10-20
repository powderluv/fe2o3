use super::*;

fn from_hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    (0..text.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&text[offset..offset + 2], 16).unwrap())
        .collect()
}

#[test]
fn registration_v1_matches_independently_assembled_golden() {
    // Assembled from the frozen 5cc V1 grammar and legacy IR preimage,
    // independently of either Rust encoder. This is not native proof evidence.
    let prefix = "463243505531000001000000be0200004013131313131313131313131313131313131313131313131313131313131313130200000011000000666978747572653a3a72656769737465720400000066696c6c";
    let suffix = "000000000100000002000b020000000005010001000b5468a10e3a0e89f44c6c7a7d613178ce7af9e472b7b525591ee1a714c95d787502000000030000000200000004000000000000000003000000000b010000000000000001000000000000000200000001000000000002010b00002a4200000000000000000000000000000000000100000000000000000000000000000000010000000000000000010000000000000002010b00002a420000000000000000000000000002010b00002a42000000000000000000000000";
    let expected = from_hex(&format!(
        "{prefix}{}{suffix}",
        "0b".repeat(208) + &"11".repeat(208)
    ));
    assert_eq!(expected.len(), 702);
    let f = fixture();
    assert_eq!(
        f.digest.as_slice(),
        from_hex("5468a10e3a0e89f44c6c7a7d613178ce7af9e472b7b525591ee1a714c95d7875")
    );
    let (bytes, commitment) = encode(f.input()).unwrap();
    assert_eq!(bytes, expected);
    let wire_sha256: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!(
        wire_sha256.as_slice(),
        from_hex("b520565c839eb05e9e36e8ad4e67d4ae42c8aee5833c3992122d08b778619324")
    );
    assert_eq!(
        commitment.as_slice(),
        from_hex("6a3a9193767c0b7cac6463edc8fcfc2ac850f73c482c1dd8d0974a286866bf8a")
    );
    decode(&expected).unwrap();
}

fn origin() -> ReferenceEnrollmentOriginV1 {
    ReferenceEnrollmentOriginV1 {
        rustc_invocation_sha256: [23; 32],
        native_policy_sha256: [29; 32],
        policy_generation: 37,
        mapping_ordinal: 7,
    }
}

fn input(f: &Fixture) -> NativeCpuPolicyInputV2<'_> {
    NativeCpuPolicyInputV2 {
        association: NativeCpuPolicyAssociationV2 {
            semantic_mir_sha256: [19; 32],
            semantic_root: 2,
            logical_kernel_name: "fill",
            origin: origin(),
        },
        kernel: &f.kernel,
        reference: &f.reference,
        replay: ReferenceReplayInputV1 {
            signature_preimage: &f.signature,
            effect_ir: &f.ir,
            effect_ir_sha256: f.digest,
            observable_output_writes: &f.writes,
        },
    }
}

fn encoded(input: NativeCpuPolicyInputV2<'_>) -> Result<(Vec<u8>, [u8; 32]), Error> {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    with_encoded_native_cpu_policy_input_v2(input, &mut budget, |bytes, hash, _| {
        (bytes.to_vec(), hash)
    })
}

fn decoded(bytes: &[u8]) -> Result<(), Error> {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    let result = with_decoded_native_cpu_policy_input_v2(bytes, &mut budget, |_, _| ());
    assert_eq!(budget.storage(), 0);
    result
}

#[test]
fn canonical_policy_roundtrip_preserves_subject_and_shared_replay() {
    let f = fixture();
    let (bytes, hash) = encoded(input(&f)).unwrap();
    assert_eq!(encoded(input(&f)).unwrap(), (bytes.clone(), hash));
    assert_eq!(&bytes[..12], b"F2CPU2\0\0\x02\0\0\0");
    assert_eq!(bytes[16], 64);
    assert_eq!(&bytes[53..56], &[1, 1, 0]);
    assert_eq!(&bytes[56..88], &[23; 32]);
    assert_eq!(&bytes[88..120], &[29; 32]);
    assert_eq!(&bytes[120..128], &37u64.to_le_bytes());
    assert_eq!(&bytes[128..132], &7u32.to_le_bytes());
    let mut expected = Sha256::new();
    expected.update(b"fe2o3/native-cpu-input/v2\0");
    expected.update(&bytes);
    assert_eq!(hash, <[u8; 32]>::from(expected.finalize()));
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    with_decoded_native_cpu_policy_input_v2(&bytes, &mut budget, |owner, budget| {
        assert_eq!(owner.commitment_v2(), hash);
        let got = owner.input_v2();
        assert_eq!(got.origin_v2(), NativeCpuOriginV2::AdmittedPolicy(origin()));
        assert_eq!(got.kernel, &f.kernel);
        assert_eq!(got.reference, &f.reference);
        assert_eq!(got.replay.signature_preimage, &f.signature);
        assert_eq!(got.replay.effect_ir, &f.ir);
        assert!(std::ptr::eq(
            got.replay.observable_output_writes,
            got.replay.effect_ir.observable_output_effects.as_ref(),
        ));
        with_encoded_native_cpu_policy_input_v2(owner.input_v2(), budget, |again, digest, _| {
            assert_eq!(again, bytes.as_slice());
            assert_eq!(digest, hash);
        })
        .unwrap();
        with_replayed_output_writes_v1(got.replay, budget, |effects, _| {
            assert_eq!(effects.writes.as_slice(), f.writes.as_ref());
        })
        .unwrap();
        budget.reserve_storage(7).unwrap();
    })
    .unwrap();
    assert_eq!(budget.storage(), 38);
}

#[test]
fn registration_and_policy_frames_cannot_cross_decode() {
    let f = fixture();
    assert_eq!(
        f.input().origin_v2(),
        NativeCpuOriginV2::SourceRegistration {
            registration_path: "fixture::register",
        }
    );
    let v1 = encode(f.input()).unwrap();
    let v2 = encoded(input(&f)).unwrap();
    assert_ne!(v1.1, v2.1);
    assert_eq!(decoded(&v1.0), Err(Error::Wire("magic")));
    assert_eq!(decode(&v2.0), Err(Error::Wire("magic")));
    for (mut wire, to_policy) in [(v1.0, true), (v2.0, false)] {
        wire[5] = if to_policy { b'2' } else { b'1' };
        wire[8] = if to_policy { 2 } else { 1 };
        assert!(
            if to_policy {
                decoded(&wire)
            } else {
                decode(&wire)
            }
            .is_err()
        );
    }
}

#[test]
fn all_origin_fields_and_association_fields_affect_commitment() {
    let f = fixture();
    let baseline = encoded(input(&f)).unwrap().1;
    for field in 0..7 {
        let mut changed = input(&f);
        match field {
            0 => changed.association.origin.rustc_invocation_sha256[31] ^= 1,
            1 => changed.association.origin.native_policy_sha256[31] ^= 1,
            2 => changed.association.origin.policy_generation += 1,
            3 => changed.association.origin.mapping_ordinal += 1,
            4 => changed.association.semantic_mir_sha256[31] ^= 1,
            5 => changed.association.semantic_root += 1,
            _ => changed.association.logical_kernel_name = "other",
        }
        let (wire, hash) = encoded(changed).unwrap();
        assert_ne!(hash, baseline, "field {field}");
        decoded(&wire).unwrap();
    }
}

#[test]
fn kernel_and_reference_identities_remain_mandatory_subjects() {
    let f = fixture();
    let baseline = encoded(input(&f)).unwrap().1;
    for kernel in [false, true] {
        for field in 0..7 {
            let mut changed = fixture();
            let id = if kernel {
                &mut changed.kernel
            } else {
                &mut changed.reference
            };
            match field {
                0 => id.def_path_hash[15] ^= 1,
                1 => id.function_sha256[31] ^= 1,
                2 => id.item_definition_sha256[31] ^= 1,
                3 => id.monomorphization_sha256[31] ^= 1,
                4 => id.generic_type_arguments_sha256[31] ^= 1,
                5 => id.const_generic_arguments_sha256[31] ^= 1,
                _ => id.rustc_mir_body_sha256[31] ^= 1,
            }
            assert_eq!(changed.digest, f.digest);
            assert_ne!(encoded(input(&changed)).unwrap().1, baseline);
        }
    }
}

#[test]
fn ordinal_bound_is_descriptive_not_actual_mapping_membership() {
    let f = fixture();
    for ordinal in [0, 255] {
        let mut selected = input(&f);
        selected.association.origin.mapping_ordinal = ordinal;
        decoded(&encoded(selected).unwrap().0).unwrap();
    }
    for ordinal in [256, u32::MAX] {
        let mut selected = input(&f);
        selected.association.origin.mapping_ordinal = ordinal;
        assert_eq!(encoded(selected), Err(Error::Wire("mapping ordinal")));
        let mut wire = encoded(input(&f)).unwrap().0;
        wire[128..132].copy_from_slice(&ordinal.to_le_bytes());
        assert_eq!(decoded(&wire), Err(Error::Wire("mapping ordinal")));
    }
}

#[test]
fn unknown_header_origin_and_domain_tags_reject() {
    let wire = encoded(input(&fixture())).unwrap().0;
    for offset in [0, 8, 9, 10, 11, 16, 53, 54, 55] {
        let mut changed = wire.clone();
        changed[offset] ^= 0x80;
        assert!(decoded(&changed).is_err(), "offset {offset}");
    }
    let mut root = wire.clone();
    root[49..53].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(decoded(&root), Err(Error::Wire("semantic root")));
    let mut length = wire;
    length[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(decoded(&length), Err(Error::Wire("frame length")));
}

#[test]
fn every_policy_frame_prefix_and_trailing_byte_rejects() {
    let (wire, _) = encoded(input(&fixture())).unwrap();
    for end in 0..wire.len() {
        assert!(decoded(&wire[..end]).is_err(), "prefix {end}");
        if end >= 16 {
            let mut shortened = wire[..end].to_vec();
            shortened[12..16].copy_from_slice(&(end as u32).to_le_bytes());
            assert!(decoded(&shortened).is_err(), "inner prefix {end}");
        }
    }
    let mut extra = wire;
    extra.push(0);
    let len = extra.len() as u32;
    extra[12..16].copy_from_slice(&len.to_le_bytes());
    assert_eq!(decoded(&extra), Err(Error::Wire("trailing bytes")));
}

#[test]
fn policy_description_does_not_bypass_body_validation() {
    let mut f = fixture();
    f.digest[0] ^= 1;
    assert_eq!(encoded(input(&f)), Err(Error::Wire("legacy effect digest")));
    let mut f = fixture();
    f.writes[0].rhs = ReferenceEffectExpressionV1::Constant(ReferenceConstantV1::ZeroSized);
    assert!(encoded(input(&f)).is_err());
    let mut f = fixture();
    f.ir.blocks[0].terminator = ReferenceTerminatorV1::Goto { target: 0 };
    f.refresh();
    assert_eq!(encoded(input(&f)), Err(Error::Wire("cyclic CPU body")));
}

#[test]
fn zero_descriptions_remain_inert_without_fabricated_authentication() {
    let f = fixture();
    let mut value = input(&f);
    value.association.origin = ReferenceEnrollmentOriginV1 {
        rustc_invocation_sha256: [0; 32],
        native_policy_sha256: [0; 32],
        policy_generation: 0,
        mapping_ordinal: 0,
    };
    decoded(&encoded(value).unwrap().0).unwrap();
}

#[path = "policy_resource_tests.rs"]
mod resources;

use std::panic::{AssertUnwindSafe, catch_unwind};

use fe2o3_build_authority::{
    CompilerClosureDigestFieldV2, CompilerClosureErrorV2, CompilerClosureV2,
};

use crate::*;

const PINS: [[u8; 32]; 6] = [
    [0x11; 32], [0x22; 32], [0x33; 32], [0x44; 32], [0x55; 32], [0x66; 32],
];
const CLOSURE_OFFSET: usize = 20;
const CLOSURE_PREIMAGE_BYTES: usize = 2 + 6 * 32;
const V2_BODY_OFFSET: usize = CLOSURE_OFFSET + CLOSURE_PREIMAGE_BYTES;

fn closure(pins: [[u8; 32]; 6]) -> CompilerClosureV2 {
    CompilerClosureV2::new(pins[0], pins[1], pins[2], pins[3], pins[4], pins[5])
        .expect("fixture closure pins are nonzero")
}

fn fixture_v2() -> RustcInvocationDescriptorV2 {
    let rustc = RustcUnitV2::new(
        "/workspace/fe2o3",
        vec![
            "/opt/fe2o3/rustc".into(),
            "--crate-name".into(),
            "fe2o3_device".into(),
            "crates/fe2o3-device/src/lib.rs".into(),
            "--crate-type=lib".into(),
            "--edition=2024".into(),
            "-Cmetadata=0123".into(),
            "-Zcodegen-backend=/opt/fe2o3/librustc_codegen_fe2o3.so".into(),
        ],
    )
    .expect("valid rustc unit");
    let environment = CompileEnvironmentV2::from_entries_for_test([
        ("CARGO_CFG_TARGET_ARCH", "amdgcn"),
        ("CARGO_FEATURE_FAST", "1"),
        ("FE2O3_HSACO_DIR", "/workspace/fe2o3/target/fe2o3"),
        ("FE2O3_TARGET", "gfx942:sramecc+:xnack-"),
        ("FE2O3_VERIFY_KERNEL_IR", "1"),
    ])
    .expect("valid environment");
    RustcInvocationDescriptorV2::new(PINS[3], PINS[5], rustc, environment)
        .expect("valid V2 descriptor")
}

fn fixture() -> RustcInvocationDescriptorV3 {
    RustcInvocationDescriptorV3::from_v2_and_compiler_closure(fixture_v2(), closure(PINS))
        .expect("matching V2 descriptor and closure")
}

fn digest(descriptor: &RustcInvocationDescriptorV3) -> InvocationDigestV3 {
    InvocationDigestV3::calculate(descriptor).expect("valid V3 descriptor digest")
}

fn find(bytes: &[u8], needle: &[u8]) -> usize {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("needle occurs in fixture")
}

#[test]
fn explicit_upgrade_round_trips_the_full_closure_and_exact_v2_body() {
    let descriptor_v2 = fixture_v2();
    let encoded_v2 = encode_descriptor_v2(&descriptor_v2).expect("encode V2");
    let compiler_closure = closure(PINS);
    let descriptor = RustcInvocationDescriptorV3::new(descriptor_v2.clone(), compiler_closure)
        .expect("upgrade V2");
    let encoded = encode_descriptor_v3(&descriptor).expect("encode V3");
    let decoded = decode_descriptor_v3(&encoded).expect("decode V3");

    assert_eq!(decoded, descriptor);
    assert_eq!(decoded.descriptor_v2(), &descriptor_v2);
    assert_eq!(decoded.compiler_closure(), &compiler_closure);
    assert_eq!(
        decoded.compiler_closure_identity_sha256(),
        compiler_closure.identity_sha256()
    );
    assert_eq!(decoded.rustc_executable_sha256(), &PINS[3]);
    assert_eq!(decoded.codegen_backend_sha256(), &PINS[5]);
    assert_eq!(decoded.rustc().working_directory(), "/workspace/fe2o3");
    assert_eq!(decoded.compile_environment().entries().len(), 5);
    assert_eq!(decoded.rustc_executable_path(), "/opt/fe2o3/rustc");
    assert_eq!(
        decoded.codegen_backend_path(),
        "/opt/fe2o3/librustc_codegen_fe2o3.so"
    );
    assert_eq!(decoded.amd_target(), "gfx942:sramecc+:xnack-");
    assert_eq!(
        decoded.artifact_output_directory(),
        "/workspace/fe2o3/target/fe2o3"
    );
    assert!(decoded.verification_required());
    assert_eq!(
        &encoded[V2_BODY_OFFSET..],
        &encoded_v2[crate::encode_v2::HEADER_BYTES..]
    );
    assert_eq!(encode_descriptor_v3(&decoded).unwrap(), encoded);
}

#[test]
fn closure_preimage_wire_order_is_exact_and_identity_is_derived() {
    let encoded = encode_descriptor_v3(&fixture()).expect("encode V3");
    assert_eq!(
        &encoded[CLOSURE_OFFSET..CLOSURE_OFFSET + 2],
        &1_u16.to_le_bytes()
    );
    for (index, pin) in PINS.iter().enumerate() {
        let offset = CLOSURE_OFFSET + 2 + index * 32;
        assert_eq!(&encoded[offset..offset + 32], pin, "closure pin {index}");
    }
    assert_eq!(
        encoded.len(),
        encode_descriptor_v2(&fixture_v2()).unwrap().len() + 194
    );
}

#[test]
fn golden_v3_encoding_and_digest_are_stable() {
    const GOLDEN_WIRE_BYTES: usize = 684;
    const GOLDEN_DIGEST_HEX: &str =
        "9061c9a6f41e175d50267b1861b64e052d86ee20ae95fc446395c0212b94f071";
    let encoded = encode_descriptor_v3(&fixture()).expect("encode V3");
    assert_eq!(encoded.len(), GOLDEN_WIRE_BYTES);
    assert_eq!(digest(&fixture()).to_hex(), GOLDEN_DIGEST_HEX);
}

#[test]
fn every_semantic_field_class_and_closure_role_affects_v3_identity() {
    let baseline = fixture();
    let baseline_wire = encode_descriptor_v3(&baseline).unwrap();
    let baseline_digest = digest(&baseline);

    for index in 0..PINS.len() {
        let mut pins = PINS;
        pins[index][index] ^= 0x80;
        let mut descriptor_v2 = fixture_v2();
        if index == 3 {
            descriptor_v2.rustc_executable_sha256 = pins[index];
        } else if index == 5 {
            descriptor_v2.codegen_backend_sha256 = pins[index];
        }
        let changed = RustcInvocationDescriptorV3::new(descriptor_v2, closure(pins)).unwrap();
        assert_ne!(
            encode_descriptor_v3(&changed).unwrap(),
            baseline_wire,
            "closure pin role {index} did not affect the wire identity"
        );
        assert_ne!(
            digest(&changed),
            baseline_digest,
            "closure pin role {index} did not affect the digest"
        );
        assert_ne!(
            changed.compiler_closure_identity_sha256(),
            baseline.compiler_closure_identity_sha256(),
            "closure pin role {index} did not affect the closure identity"
        );
    }

    let mut process_mutations = Vec::new();

    let mut working_directory = fixture_v2();
    working_directory.rustc.working_directory =
        crate::model_v2::AbsolutePath::new("/workspace/other", "rustc working directory").unwrap();
    process_mutations.push(("working directory", working_directory));

    let mut argument = fixture_v2();
    argument.rustc.argv[2] = crate::model_v2::Argument::new("other_crate").unwrap();
    process_mutations.push(("argument", argument));

    let mut environment_key = fixture_v2();
    environment_key.compile_environment.entries[0].key =
        crate::model_v2::Name::new("CARGO_CFG_TARGET_CPU", "compile environment key").unwrap();
    process_mutations.push(("environment key", environment_key));

    let mut environment_value = fixture_v2();
    environment_value.compile_environment.entries[0].value =
        crate::model_v2::EnvironmentValue::new("amdgpu").unwrap();
    process_mutations.push(("environment value", environment_value));

    for (field, descriptor_v2) in process_mutations {
        let changed = RustcInvocationDescriptorV3::new(descriptor_v2, closure(PINS)).unwrap();
        assert_ne!(
            digest(&changed),
            baseline_digest,
            "{field} did not affect digest"
        );
    }
}

#[test]
fn mismatched_rustc_and_backend_roles_fail_at_construction_encoding_and_decode() {
    let mut rustc_mismatch = fixture_v2();
    rustc_mismatch.rustc_executable_sha256[0] ^= 1;
    assert_eq!(
        RustcInvocationDescriptorV3::new(rustc_mismatch, closure(PINS)),
        Err(ValidationError::CompilerClosurePinMismatch {
            field: "rustc executable"
        })
    );

    let mut backend_mismatch = fixture_v2();
    backend_mismatch.codegen_backend_sha256[0] ^= 1;
    assert_eq!(
        RustcInvocationDescriptorV3::new(backend_mismatch, closure(PINS)),
        Err(ValidationError::CompilerClosurePinMismatch {
            field: "codegen backend"
        })
    );

    let mut swapped_pins = PINS;
    swapped_pins.swap(3, 5);
    assert!(matches!(
        RustcInvocationDescriptorV3::new(fixture_v2(), closure(swapped_pins)),
        Err(ValidationError::CompilerClosurePinMismatch { .. })
    ));

    let mut internally_mismatched = fixture();
    internally_mismatched.descriptor_v2.rustc_executable_sha256[0] ^= 1;
    assert!(matches!(
        encode_descriptor_v3(&internally_mismatched),
        Err(ValidationError::CompilerClosurePinMismatch {
            field: "rustc executable"
        })
    ));

    for (offset, field) in [
        (CLOSURE_OFFSET + 2 + 3 * 32, "rustc executable"),
        (CLOSURE_OFFSET + 2 + 5 * 32, "codegen backend"),
        (V2_BODY_OFFSET, "rustc executable"),
        (V2_BODY_OFFSET + 32, "codegen backend"),
    ] {
        let mut mutated = encode_descriptor_v3(&fixture()).unwrap();
        mutated[offset] ^= 1;
        assert!(
            matches!(
                decode_descriptor_v3(&mutated),
                Err(DecodeError::Validation(
                    ValidationError::CompilerClosurePinMismatch { field: actual }
                )) if actual == field
            ),
            "accepted mismatched {field} wire role"
        );
    }
}

#[test]
fn descriptor_and_closure_versions_are_explicit_and_fail_closed() {
    let encoded = encode_descriptor_v3(&fixture()).unwrap();
    for version in [0, 1, 2, 4, u16::MAX] {
        let mut mutated = encoded.clone();
        mutated[8..10].copy_from_slice(&version.to_le_bytes());
        assert_eq!(
            decode_descriptor_v3(&mutated),
            Err(DecodeError::UnknownVersion(version))
        );
    }
    assert_eq!(
        decode_descriptor_v2(&encoded),
        Err(DecodeError::UnknownVersion(3))
    );
    assert_eq!(
        decode_descriptor_v1(&encoded),
        Err(DecodeError::UnknownVersion(3))
    );
    let encoded_v2 = encode_descriptor_v2(&fixture_v2()).unwrap();
    assert_eq!(
        decode_descriptor_v3(&encoded_v2),
        Err(DecodeError::UnknownVersion(2))
    );

    for version in [0, 2, u16::MAX] {
        let mut mutated = encoded.clone();
        mutated[CLOSURE_OFFSET..CLOSURE_OFFSET + 2].copy_from_slice(&version.to_le_bytes());
        assert_eq!(
            decode_descriptor_v3(&mutated),
            Err(DecodeError::CompilerClosure(
                CompilerClosureErrorV2::UnsupportedTransitionProtocolVersion { version }
            ))
        );
    }
}

#[test]
fn noncanonical_headers_closure_pins_and_environment_are_rejected() {
    let encoded = encode_descriptor_v3(&fixture()).unwrap();

    let mut invalid = encoded.clone();
    invalid[10..12].copy_from_slice(&1_u16.to_le_bytes());
    assert_eq!(
        decode_descriptor_v3(&invalid),
        Err(DecodeError::UnsupportedFlags(1))
    );

    let mut invalid = encoded.clone();
    invalid[16] = 1;
    assert_eq!(
        decode_descriptor_v3(&invalid),
        Err(DecodeError::NonzeroReserved {
            field: "descriptor header"
        })
    );

    for (index, field) in [
        CompilerClosureDigestFieldV2::CargoExecutable,
        CompilerClosureDigestFieldV2::CargoBindingTrampoline,
        CompilerClosureDigestFieldV2::CargoFe2o3BindingWrapper,
        CompilerClosureDigestFieldV2::RustcExecutable,
        CompilerClosureDigestFieldV2::RustcRuntimeTree,
        CompilerClosureDigestFieldV2::CodegenBackend,
    ]
    .into_iter()
    .enumerate()
    {
        let mut invalid = encoded.clone();
        let offset = CLOSURE_OFFSET + 2 + index * 32;
        invalid[offset..offset + 32].fill(0);
        assert_eq!(
            decode_descriptor_v3(&invalid),
            Err(DecodeError::CompilerClosure(
                CompilerClosureErrorV2::ZeroDigest { field }
            ))
        );
    }

    let mut invalid = encoded.clone();
    invalid[find(&encoded, b"CARGO_CFG_TARGET_ARCH")] = b'Z';
    assert!(matches!(
        decode_descriptor_v3(&invalid),
        Err(DecodeError::Validation(
            ValidationError::NonCanonicalOrder {
                field: "compile environment"
            }
        ))
    ));

    let mut trailing = encoded;
    trailing.push(0);
    assert_eq!(
        decode_descriptor_v3(&trailing),
        Err(DecodeError::TrailingBytes)
    );
    let trailing_len = trailing.len() as u32;
    trailing[12..16].copy_from_slice(&trailing_len.to_le_bytes());
    assert_eq!(
        decode_descriptor_v3(&trailing),
        Err(DecodeError::TrailingBytes)
    );
}

#[test]
fn every_truncation_and_complete_descriptor_bound_are_rejected() {
    let encoded = encode_descriptor_v3(&fixture()).unwrap();
    for length in 0..encoded.len() {
        assert!(
            decode_descriptor_v3(&encoded[..length]).is_err(),
            "accepted truncation at {length}"
        );
    }

    let mut oversized = vec![0_u8; MAX_DESCRIPTOR_BYTES_V3 + 1];
    oversized[..8].copy_from_slice(&INVOCATION_DESCRIPTOR_MAGIC_V3);
    assert_eq!(
        decode_descriptor_v3(&oversized),
        Err(DecodeError::TooLarge {
            max: MAX_DESCRIPTOR_BYTES_V3
        })
    );
}

#[test]
fn deterministic_exhaustive_byte_mutations_never_bypass_canonical_reencoding() {
    let descriptor = fixture();
    let encoded = encode_descriptor_v3(&descriptor).unwrap();
    let original_digest = digest(&descriptor);

    for index in 0..encoded.len() {
        for mask in [1, 0x80, 0xff] {
            let mut mutated = encoded.clone();
            mutated[index] ^= mask;
            let result = catch_unwind(AssertUnwindSafe(|| decode_descriptor_v3(&mutated)));
            assert!(
                result.is_ok(),
                "decoder panicked at byte {index}, mask {mask:#x}"
            );
            if let Ok(decoded) = result.unwrap() {
                assert_eq!(encode_descriptor_v3(&decoded).unwrap(), mutated);
                assert_ne!(digest(&decoded), original_digest);
            }
        }
    }
}

#[test]
fn digest_bytes_and_hex_are_canonical_nonzero_and_domain_separated() {
    let value = digest(&fixture());
    assert_eq!(
        INVOCATION_DIGEST_DOMAIN_V3,
        b"FE2O3/RUSTC-BUILD-INVOCATION/V3\0"
    );
    assert_ne!(INVOCATION_DIGEST_DOMAIN_V3, INVOCATION_DIGEST_DOMAIN_V2);
    assert_eq!(
        InvocationDigestV3::from_bytes(value.into_bytes()).unwrap(),
        value
    );
    assert_eq!(
        InvocationDigestV3::from_hex(&value.to_hex()).unwrap(),
        value
    );
    assert_eq!(value.to_string(), value.to_hex());
    assert_eq!(value.to_hex().parse::<InvocationDigestV3>().unwrap(), value);
    assert_eq!(
        InvocationDigestV3::from_bytes([0; 32]),
        Err(DigestError::ReservedAllZero)
    );
    assert_eq!(
        InvocationDigestV3::from_hex(&"0".repeat(64)),
        Err(DigestError::ReservedAllZero)
    );
    assert_eq!(
        InvocationDigestV3::from_hex("00"),
        Err(DigestError::InvalidHexLength)
    );
    assert!(matches!(
        InvocationDigestV3::from_hex(&"A".repeat(64)),
        Err(DigestError::InvalidHexCharacter { index: 0 })
    ));
}

#[test]
fn encoded_digest_uses_the_same_domain_length_and_preimage_without_reencoding() {
    use sha2::{Digest, Sha256};
    let descriptor = fixture();
    let bytes = encode_descriptor_v3(&descriptor).unwrap();
    let expected = digest(&descriptor);
    assert_eq!(
        InvocationDigestV3::calculate_encoded(&bytes).unwrap(),
        expected
    );
    let raw: [u8; 32] = Sha256::digest(&bytes).into();
    assert_ne!(raw, expected.into_bytes());
    let mut h = Sha256::new();
    h.update(INVOCATION_DIGEST_DOMAIN_V3);
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(&bytes);
    assert_eq!(<[u8; 32]>::from(h.finalize()), expected.into_bytes());
    let mut changed = bytes.clone();
    changed.push(0);
    assert_ne!(
        InvocationDigestV3::calculate_encoded(&changed).unwrap(),
        expected
    );
    changed.pop();
    changed[0] ^= 1;
    assert_ne!(
        InvocationDigestV3::calculate_encoded(&changed).unwrap(),
        expected
    );
}

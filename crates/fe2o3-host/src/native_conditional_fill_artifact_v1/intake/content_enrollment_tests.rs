//! Actual inert V5 inputs and signed fixture carriage, never protected execution.
use super::*;

// Independent legacy field framing used by the compiler's no-enrollment producer.
// These inert identities exercise compatibility, not recovered source authority.
fn legacy_inventory() -> Vec<u8> {
    fn field(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }
    let mut out = Vec::new();
    field(&mut out, b"fe2o3/semantic-mir/rustc-identity-inventory/v2");
    field(&mut out, &[1; 32]);
    for value in 2..7 {
        field(&mut out, &[value; 32]);
    }
    field(&mut out, &[0]);
    field(&mut out, &[1]);
    field(&mut out, b"fill");
    field(&mut out, &[1]);
    field(&mut out, &[7; 32]);
    field(&mut out, &[0]);
    field(&mut out, &[0]);
    field(&mut out, &0u32.to_le_bytes());
    out
}

#[test]
fn producer_legacy_absence_is_accepted_without_enrollment_fallback() {
    for request in [
        None,
        Some(REQUEST),
        Some(r#"{"version":1,"bindings":[]}"#),
        Some("{"),
    ] {
        let handoff = raw_fixture::handoff_with_enrollment(request, |_| legacy_inventory());
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|b| {
            let (mut input, policy) = input(handoff, b);
            let floor = b.storage();
            assert_eq!(
                input.require_enrollment(&policy, b).is_ok(),
                request.is_none()
            );
            assert_eq!(input.enrollment, None);
            assert_eq!(b.storage(), floor);
        });
    }
}

#[test]
fn synthetic_zero_count_inventory_cannot_replace_original_absence() {
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|b| {
        let (mut input, policy) = input(handoff(None, "zero_wrapper"), b);
        let floor = b.storage();
        assert!(input.require_enrollment(&policy, b).is_err());
        assert_eq!(b.storage(), floor);
    });
}

#[test]
fn original_absence_still_enforces_account_floor_and_terminal_denials() {
    for case in 0..4 {
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|b| {
            let (mut input, policy) = input(handoff(None, ""), b);
            let mut foreign = Owned::new(Work::new(WORK), STORAGE);
            foreign.with_budget(|other| {
                other.reserve_storage(input.floor).unwrap();
                assert!(matches!(
                    input.require_enrollment(&policy, other),
                    Err(Error::Resource(Resource::Accounting))
                ));
                assert_eq!(other.work(), 4);
                assert_eq!(other.storage(), input.floor);
            });
            match case {
                0 => b.release_storage(1).unwrap(),
                1 => assert!(b.charge_work(WORK).is_err()),
                2 => assert!(b.reserve_storage(STORAGE).is_err()),
                _ => b.charge_work(WORK - b.work() - 4).unwrap(),
            }
            let before = (b.work(), b.storage(), b.failed_work(), b.failed_storage());
            assert!(matches!(
                input.require_enrollment(&policy, b),
                Err(Error::Resource(_))
            ));
            assert_eq!(input.enrollment, None);
            assert_eq!(b.storage(), before.1);
            if case == 1 || case == 2 {
                assert_eq!(
                    (b.work(), b.storage(), b.failed_work(), b.failed_storage()),
                    before
                );
            }
        });
    }
}
use ed25519_dalek::SigningKey;
use fe2o3_compiler_execution_protocol::{
    CompilerExecutionAttestationReceiptV3 as Receipt,
    CompilerExecutionAttestationRequestV3 as Request,
    CompilerExecutionAttestationStorageV3 as Charge,
    CompilerExecutionReceiptPublicationAckV3 as Ack,
    CompilerExecutionReceiptPublicationV3 as ReceiptPublication,
};
use fe2o3_compiler_lineage::{
    MAX_NATIVE_CONDITIONAL_STORAGE_V1, RustcEnrollmentInventoryHeaderV1 as Header,
    RustcEnrollmentInventoryInputV1 as Inventory, RustcEnrollmentInventoryRootV1 as Root,
    encode_rustc_enrollment_inventory_v1 as encode_inventory,
};

const REQUEST: &str =
    r#"{"version":1,"bindings":[{"kernel":"demo::kernel","reference":"demo::reference"}]}"#;

fn handoff(request: Option<&str>, mutation: &str) -> Handoff {
    raw_fixture::handoff_with_enrollment(request, |invocation| {
        if request.is_none() && mutation.is_empty() {
            return legacy_inventory();
        }
        if mutation == "legacy" {
            return b"legacy inventory only".to_vec();
        }
        let policy = fixture::policy_wire(3);
        let mut header = Header {
            kernel_count: 1,
            enrollment_binding_count: u32::from(request.is_some()),
            invocation_identity: Sha256::digest(invocation).into(),
            native_policy_identity: policy[184..].try_into().unwrap(),
            native_policy_generation: 7,
        };
        match mutation {
            "invocation" => header.invocation_identity[0] ^= 1,
            "domain" => {
                header.invocation_identity =
                    fe2o3_rustc_invocation::InvocationDigestV3::calculate_encoded(invocation)
                        .unwrap()
                        .into_bytes()
            }
            "policy" => header.native_policy_identity[0] ^= 1,
            "generation" => header.native_policy_generation += 1,
            "count" => header.enrollment_binding_count ^= 1,
            _ => {}
        }
        let roots = [Root {
            semantic_root: 9,
            origin_tag: header.enrollment_binding_count as u8,
            descriptor_ordinal: 0,
            logical_name_len: 6,
            logical_name_sha256: Sha256::digest(b"kernel").into(),
            kernel_binding: [1; 32],
            kernel_instance: [2; 32],
            reference_instance: [if mutation == "root" { 4 } else { 3 }; 32],
        }];
        let mut bytes = encode_inventory(
            Inventory {
                legacy_inventory: b"host-inert-inventory",
                header,
                roots: &roots,
            },
            MAX_NATIVE_CONDITIONAL_STORAGE_V1,
            |_| Ok::<_, Resource>(()),
        )
        .unwrap();
        if mutation == "corrupt" {
            let last = bytes.len() - 1;
            bytes[last] ^= 1;
        }
        if mutation == "truncated" {
            bytes.pop();
        }
        bytes
    })
}

fn retain<T, E: std::fmt::Debug>(
    result: std::result::Result<(T, Charge), E>,
    b: &mut Budget<'_>,
) -> T {
    let (value, charge) = result.unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    value
}

fn input<'work>(handoff: Handoff, b: &mut Budget<'work>) -> (AuthenticatedInput<'work>, Policy) {
    let wire = raw_claim_wire(&handoff, 9);
    let mut request = fixture::request_wire(3);
    request[224..914].copy_from_slice(&wire);
    request[80..112].copy_from_slice(&wire[658..]);
    fixture::seal(&mut request[24..224], "COMPILER-EXECUTION-CHALLENGE", 3);
    fixture::seal(&mut request, "COMPILER-EXECUTION-REQUEST", 3);
    b.reserve_storage(
        handoff.backing_capacity()
            + METADATA
            + request.len()
            + wire.len()
            + size_of::<SigningKey>()
            + size_of::<AuthenticatedInput<'work>>(),
    )
    .unwrap();
    let policy_wire = fixture::policy_wire(3);
    let carried_policy = policy(&policy_wire, b);
    let independent_policy = policy(&policy_wire, b);
    let request = retain(Request::decode_in_original_account_v3(&request, b), b);
    let key = SigningKey::from_bytes(&[0x51; 32]);
    let receipt = retain(Receipt::issue(&carried_policy, &request, &key, b), b);
    let publication = retain(
        ReceiptPublication::new([0x81; 32], [0x82; 32], receipt, b),
        b,
    );
    let ack = retain(Ack::new(&publication, [0x83; 32], b), b);
    let carriage = retain(
        Carriage::new(carried_policy, request, publication, ack, b),
        b,
    );
    require_same_policy(
        &independent_policy,
        carriage.policy(),
        independent_policy.identity().as_bytes(),
        b,
    )
    .unwrap();
    authenticate_raw_subject(&handoff, carriage.request().subject(), b).unwrap();
    let value = AuthenticatedInput {
        handoff,
        carriage,
        enrollment: None,
        ledger: b.work_ledger_identity_v1(),
        account: b.storage_account_identity_v1().unwrap(),
        floor: b.storage(),
        _work: std::marker::PhantomData,
    };
    (value, independent_policy)
}

#[test]
fn actual_capsule_and_signed_subject_accept_original_absence_and_enrollment() {
    for (request, spare) in [(None, false), (Some(REQUEST), false), (Some(REQUEST), true)] {
        let original = handoff(request, "");
        let handoff = if spare {
            let bytes = original.canonical_bytes();
            let mut backing = Vec::with_capacity(bytes.len() + 8192);
            backing.extend_from_slice(&[5; 31]);
            backing.extend_from_slice(bytes);
            backing.extend_from_slice(&[7; 47]);
            Handoff::decode_shared_vec(std::sync::Arc::new(backing), 31..31 + bytes.len()).unwrap()
        } else {
            original
        };
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|b| {
            let (mut input, policy) = input(handoff, b);
            let capsule = input.handoff.capsule();
            let invocation = capsule.invocation_bytes();
            assert_eq!(
                invocation,
                fe2o3_rustc_invocation::encode_descriptor_v3(capsule.invocation()).unwrap()
            );
            let bytes = capsule.canonical_bytes();
            let start = invocation.as_ptr() as usize - bytes.as_ptr() as usize;
            assert_eq!(&bytes[start..start + invocation.len()], invocation);
            assert_ne!(
                <[u8; 32]>::from(Sha256::digest(invocation)),
                capsule.invocation_digest().into_bytes()
            );
            let floor = b.storage();
            let expected = request.map(|_| Expected {
                rustc_invocation_sha256: Sha256::digest(invocation).into(),
                native_policy_sha256: *policy.identity().as_bytes(),
                policy_generation: policy.generation(),
                enrollment_binding_count: 1,
            });
            input.require_enrollment(&policy, b).unwrap();
            assert_eq!(input.enrollment, expected);
            assert_eq!(b.storage(), floor);
            assert_eq!(b.failed_work(), None);
            assert_eq!(b.failed_storage(), None);
            assert!(!input.carriage.grants_launch_authority());
        });
    }
}

#[test]
fn signed_resealed_capsules_reject_inventory_coordinate_and_framing_changes() {
    for request in [None, Some(REQUEST)] {
        for mutation in [
            "invocation",
            "domain",
            "policy",
            "generation",
            "count",
            "legacy",
            "corrupt",
            "truncated",
        ] {
            let handoff = handoff(request, mutation);
            let mut owned = Owned::new(Work::new(WORK), STORAGE);
            owned.with_budget(|b| {
                let (mut input, policy) = input(handoff, b);
                let floor = b.storage();
                assert!(input.require_enrollment(&policy, b).is_err(), "{mutation}");
                assert_eq!(input.enrollment, None);
                assert_eq!(b.storage(), floor);
            });
        }
    }
}

#[test]
fn present_empty_or_malformed_captured_request_never_falls_back_to_absent() {
    for request in [
        r#"{"version":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[{"kernel":"k","reference":"r"}],"extra":1}"#,
        "{",
    ] {
        let handoff = handoff(Some(request), "");
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|b| {
            let (mut input, policy) = input(handoff, b);
            let floor = b.storage();
            assert!(input.require_enrollment(&policy, b).is_err());
            assert_eq!(b.storage(), floor);
        });
    }
}

#[test]
fn matching_headers_do_not_establish_root_or_cpu_source_equivalence() {
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|b| {
        let (mut input, policy) = input(handoff(Some(REQUEST), "root"), b);
        input.require_enrollment(&policy, b).unwrap();
        // Retained coordinates are not a mapped verifier receipt.
        assert!(input.enrollment.is_some());
        assert!(!input.carriage.grants_compiler_authority());
    });
}

#[test]
fn equal_funded_foreign_account_and_short_original_floor_refuse_before_inventory_read() {
    let mut owned = Owned::new(Work::new(WORK), STORAGE);
    owned.with_budget(|b| {
        let (mut input, policy) = input(handoff(Some(REQUEST), ""), b);
        let mut foreign = Owned::new(Work::new(WORK), STORAGE);
        foreign.with_budget(|other| {
            other.reserve_storage(input.floor).unwrap();
            assert!(matches!(
                input.require_enrollment(&policy, other),
                Err(Error::Resource(Resource::Accounting))
            ));
            assert_eq!(other.work(), 4);
            assert_eq!(other.storage(), input.floor);
        });
        b.release_storage(1).unwrap();
        let before = b.work();
        assert!(matches!(
            input.require_enrollment(&policy, b),
            Err(Error::Resource(Resource::Accounting))
        ));
        assert_eq!(b.work(), before + 4);
        assert_eq!(b.storage(), input.floor - 1);
    });
}

#[test]
fn prior_denial_is_terminal_without_refunding_work_or_storage() {
    for storage in [false, true] {
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|b| {
            let (mut input, policy) = input(handoff(Some(REQUEST), ""), b);
            if storage {
                assert!(b.reserve_storage(STORAGE).is_err());
            } else {
                assert!(b.charge_work(WORK).is_err());
            }
            let before = (b.work(), b.storage(), b.failed_work(), b.failed_storage());
            assert!(matches!(
                input.require_enrollment(&policy, b),
                Err(Error::Resource(_))
            ));
            assert_eq!(
                (b.work(), b.storage(), b.failed_work(), b.failed_storage()),
                before
            );
        });
    }
}

#[test]
fn short_inventory_scratch_and_work_preserve_all_original_input_owners() {
    for storage in [false, true] {
        let mut owned = Owned::new(Work::new(WORK), STORAGE);
        owned.with_budget(|b| {
            let (mut input, policy) = input(handoff(Some(REQUEST), ""), b);
            if storage {
                b.reserve_storage(Budget::STORAGE_WINDOW_SCRATCH_V1)
                    .unwrap();
            } else {
                b.charge_work(WORK - b.work() - 4).unwrap();
            }
            let floor = b.storage();
            let result = if storage {
                b.with_additional_storage_window_v1(enrollment::SCRATCH - 1, |b| {
                    input.require_enrollment(&policy, b)
                })
            } else {
                input.require_enrollment(&policy, b)
            };
            assert!(matches!(result, Err(Error::Resource(_))));
            assert_eq!(b.storage(), floor);
            assert!(!input.carriage.grants_launch_authority());
        });
    }
}

#[test]
fn original_signature_policy_and_raw_subject_join_precede_inventory_validation() {
    let source = include_str!("content.rs");
    let authenticate = source
        .split("pub(super) fn authenticate")
        .nth(1)
        .unwrap()
        .split("fn require_same_policy(")
        .next()
        .unwrap();
    before(
        authenticate,
        "require_same_policy(",
        "authenticate_raw_subject(",
    );
    before(
        authenticate,
        "authenticate_raw_subject(",
        "authenticated.require_enrollment(",
    );
    before(
        authenticate,
        "authenticated.require_enrollment(",
        "Ok(authenticated)",
    );
    assert!(
        source
            .contains("recover_native_conditional_handoff_under_policy_file_v1 as recover_source")
    );
    assert!(source.contains(
        "recover_native_conditional_handoff_under_policy_file_with_cpu_mapping_v1 as recover_mapped_source"
    ));
    let recover = source.split("pub(super) fn recover").nth(1).unwrap();
    before(recover, "require_original_account(", "match enrollment");
    assert_eq!(recover.matches("recover_source(").count(), 1);
    assert_eq!(recover.matches("recover_mapped_source(").count(), 1);
    assert!(recover.contains("None => recover_source("));
    assert!(recover.contains("Some(expected) =>"));
    before(
        recover,
        "recover_mapped_source(",
        "let (finalized, charge) = replay(",
    );
    assert!(!recover.contains("or_else("));
}

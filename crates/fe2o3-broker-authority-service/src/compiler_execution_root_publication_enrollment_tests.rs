//! Content/accounting tests; not a live compiler, proof or launch qualification.
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
    let legacy = legacy_inventory();
    assert!(check(&legacy, &None));
    assert!(!check(&legacy, &Some(expected())));
}

#[test]
fn synthetic_zero_count_inventory_cannot_replace_original_absence() {
    assert!(!check(&bytes(expected(), false, 5), &None));
}
use fe2o3_compiler_lineage::{
    MAX_NATIVE_CONDITIONAL_STORAGE_V1, RustcEnrollmentInventoryHeaderV1 as Header,
    RustcEnrollmentInventoryInputV1 as Input, RustcEnrollmentInventoryRootV1 as Root,
    encode_rustc_enrollment_inventory_v1 as encode,
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as WorkBudget;
use sha2::{Digest, Sha256};

fn expected() -> Enrollment {
    Enrollment {
        rustc_invocation_sha256: Sha256::digest(b"original canonical invocation").into(),
        native_policy_sha256: [2; 32],
        policy_generation: 7,
        enrollment_binding_count: 1,
    }
}

fn bytes(expected: Enrollment, enrolled: bool, reference: u8) -> Vec<u8> {
    encode(
        Input {
            legacy_inventory: b"retained legacy inventory",
            header: Header {
                kernel_count: 1,
                enrollment_binding_count: u32::from(enrolled),
                invocation_identity: expected.rustc_invocation_sha256,
                native_policy_identity: expected.native_policy_sha256,
                native_policy_generation: expected.policy_generation,
            },
            roots: &[Root {
                semantic_root: 0,
                origin_tag: u8::from(enrolled),
                descriptor_ordinal: 0,
                logical_name_len: 4,
                logical_name_sha256: Sha256::digest(b"fill").into(),
                kernel_binding: [3; 32],
                kernel_instance: [4; 32],
                reference_instance: [reference; 32],
            }],
        },
        MAX_NATIVE_CONDITIONAL_STORAGE_V1,
        |_| Ok::<_, Resource>(()),
    )
    .unwrap()
}

fn check(bytes: &[u8], expected: &Option<Enrollment>) -> bool {
    let owners = bytes.len() + 4096;
    let floor = owners + size_of::<Option<Enrollment>>();
    let mut work = WorkBudget::new(WORK);
    let mut b = Budget::new(&mut work, floor + SCRATCH);
    b.reserve_storage(floor).unwrap();
    let result = require(bytes, owners, expected, &mut b);
    assert_eq!(b.storage(), floor);
    assert!(b.work() <= WORK);
    result.is_ok()
}

#[test]
fn original_absence_is_distinct_from_zero_or_present_bindings() {
    let original = expected();
    let enrolled = bytes(original, true, 5);
    let registration = legacy_inventory();
    assert!(check(&enrolled, &Some(original)));
    assert!(check(&registration, &None));
    assert!(!check(&enrolled, &None));
    assert!(!check(&registration, &Some(original)));
    let mut zero = original;
    zero.enrollment_binding_count = 0;
    assert!(!check(&registration, &Some(zero)));
}

#[test]
fn retained_inventory_rejects_swapped_policy_generation_count_and_digest_domain() {
    let original = expected();
    for case in 0..5 {
        let mut changed = original;
        match case {
            0 => changed.rustc_invocation_sha256[0] ^= 1,
            1 => {
                changed.rustc_invocation_sha256 =
                    *fe2o3_rustc_invocation::InvocationDigestV3::calculate_encoded(
                        b"original canonical invocation",
                    )
                    .unwrap()
                    .as_bytes()
            }
            2 => changed.native_policy_sha256[0] ^= 1,
            3 => changed.policy_generation += 1,
            _ => changed.enrollment_binding_count += 1,
        }
        // Compare a canonically encoded foreign capsule member, never a header
        // supplied alongside the original member as an alternative authority.
        if case < 4 {
            assert!(!check(&bytes(changed, true, 5), &Some(original)));
        } else {
            assert!(!check(&bytes(original, true, 5), &Some(changed)));
        }
    }
}

#[test]
fn missing_legacy_truncated_or_corrupted_inventory_never_falls_back() {
    let original = expected();
    let mut encoded = bytes(original, true, 5);
    for invalid in [&[][..], &b"legacy only"[..], &encoded[..encoded.len() - 1]] {
        assert!(!check(invalid, &Some(original)));
        assert!(!check(invalid, &None));
    }
    let last = encoded.len() - 1;
    encoded[last] ^= 1;
    assert!(!check(&encoded, &Some(original)));
}

#[test]
fn equal_headers_still_require_later_full_source_and_root_validation() {
    let original = expected();
    let first = bytes(original, true, 5);
    let other = bytes(original, true, 6);
    assert_ne!(first, other);
    assert!(check(&first, &Some(original)));
    assert!(check(&other, &Some(original)));
    // This prerequisite deliberately returns no certificate, mapping or owner.
}

#[test]
fn original_capacity_floor_short_work_and_sticky_denials_refuse() {
    let encoded = bytes(expected(), true, 5);
    let owners = encoded.capacity() + size_of::<Vec<u8>>() + 4096;
    let floor = owners + size_of::<Option<Enrollment>>();
    for case in 0..5 {
        let work_limit = if case == 1 { ENTRY - 1 } else { WORK };
        let storage_limit = floor + SCRATCH - usize::from(case == 2);
        let mut work = WorkBudget::new(work_limit);
        let mut b = Budget::new(&mut work, storage_limit);
        b.reserve_storage(floor - usize::from(case == 0)).unwrap();
        if case == 3 {
            assert!(b.charge_work(WORK + 1).is_err());
        }
        if case == 4 {
            assert!(b.reserve_storage(storage_limit + 1).is_err());
        }
        let before = (b.work(), b.storage(), b.failed_work(), b.failed_storage());
        assert!(require(&encoded, owners, &Some(expected()), &mut b).is_err());
        assert_eq!(b.storage(), before.1);
        if case >= 3 {
            assert_eq!(
                (b.work(), b.storage(), b.failed_work(), b.failed_storage()),
                before
            );
        }
    }
}

#[test]
fn publication_schedule_adds_reader_cost_without_changing_ordinary_custody() {
    let ordinary = RootPublicationCustodyV3::maximum_revalidation_quota(1024 * 1024).unwrap();
    let mapped =
        RootPublicationCustodyV3::maximum_enrollment_revalidation_quota(1024 * 1024).unwrap();
    assert_eq!(mapped.work(), ordinary.work() + WORK);
    assert_eq!(mapped.scratch(), ordinary.scratch() + SCRATCH);
}

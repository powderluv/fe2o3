//! Public verifier entry refusal using the existing inert V5 fixture, never proof execution.
use super::*;
use fe2o3_verifier as verifier;
use std::mem::size_of_val;

fn denied_budget(b: &mut Budget<'_>, storage: bool) {
    if storage {
        assert!(b.reserve_storage(STORAGE).is_err());
    } else {
        assert!(b.charge_work(WORK + 1).is_err());
    }
}
fn state(b: &Budget<'_>) -> (usize, usize, usize, Option<usize>, Option<usize>) {
    (
        b.work(),
        b.storage(),
        b.peak_storage(),
        b.failed_work(),
        b.failed_storage(),
    )
}

#[test]
fn public_native_recovery_routes_refuse_before_reading_inert_handoff() {
    for route in 0..6 {
        for storage in [false, true] {
            let handoff = raw_fixture::handoff();
            let expected = verifier::NativeConditionalCpuMappingExpectationV1 {
                rustc_invocation_sha256: [3; 32],
                native_policy_sha256: [5; 32],
                policy_generation: 7,
                enrollment_binding_count: 1,
            };
            let mut owned = Owned::new(Work::new(WORK), 256 * 1024 * 1024);
            owned.with_budget(|b| {
                b.reserve_storage(handoff.backing_capacity() + METADATA + size_of_val(&expected)).unwrap();
                denied_budget(b, storage);
                let before = state(b);
                let ledger = b.work_ledger_identity_v1();
                let account = b.storage_account_identity_v1();
                let limits = verifier::CanonicalRefinedForwardingHistoryLimitsV1::production_v1();
                let profile = fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942;
                let result = match route {
                    0 => verifier::recover_compiler_conditional_native_semantic_handoff_v5(handoff, &[], limits, profile, b),
                    1 => verifier::recover_compiler_conditional_native_semantic_handoff_in_original_account_v5(handoff, &[], limits, profile, b),
                    2 => verifier::recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_v5(handoff, &[], &[], limits, profile, b),
                    3 => verifier::recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_in_original_account_v5(handoff, &[], &[], limits, profile, b),
                    4 => verifier::recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_v5(handoff, &[], &expected, limits, profile, b),
                    _ => verifier::recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_in_original_account_v5(handoff, &[], &expected, limits, profile, b),
                };
                assert!(result.is_err());
                assert_eq!(state(b), before);
                assert!(b.work_ledger_identity_v1() == ledger);
                assert_eq!(b.storage_account_identity_v1(), account);
            });
        }
    }
}

#[test]
fn public_policy_selected_recovery_preserves_prior_denial_before_policy_decode() {
    for route in 0..3 {
        for storage in [false, true] {
            let handoff = raw_fixture::handoff();
            let expected = verifier::NativeConditionalCpuMappingExpectationV1 {
                rustc_invocation_sha256: [3; 32],
                native_policy_sha256: [5; 32],
                policy_generation: 7,
                enrollment_binding_count: 1,
            };
            let mut owned = Owned::new(Work::new(WORK), STORAGE);
            owned.with_budget(|b| {
                b.reserve_storage(handoff.backing_capacity() + METADATA + size_of_val(&expected)).unwrap();
                denied_budget(b, storage);
                let before = state(b);
                // Invalid policy is deliberate: original denial must win before reading it.
                let result = match route {
                    0 => verifier::recover_native_conditional_handoff_under_policy_file_v1(&[], handoff, b),
                    1 => verifier::recover_native_conditional_handoff_under_policy_file_with_cpu_origins_v1(&[], handoff, &[], b),
                    _ => verifier::recover_native_conditional_handoff_under_policy_file_with_cpu_mapping_v1(&[], handoff, &expected, b),
                };
                let error = match result { Err(e) => e, Ok(_) => panic!("denied entry succeeded") };
                let resource = error.get_ref().and_then(|e| e.downcast_ref::<Resource>()).unwrap();
                assert!(matches!((storage, resource),
                    (true, Resource::Storage(_)) | (false, Resource::Work(_))));
                assert_eq!(state(b), before);
            });
        }
    }
}

//! Inert roster/codec refusal tests; no Request, Loan or proof is fabricated.
use super::*;
use crate::portable_reference_v1::codec::{
    NativeCpuPolicyAssociationV2, NativeCpuPolicyInputV2, ReferenceEnrollmentOriginV1,
    with_decoded_native_cpu_input_v1, with_encoded_native_cpu_policy_input_v2,
};
use cpu_origin::{CpuReplayMode, DecodedCpu};
use std::cell::Cell;

fn origin() -> ReferenceEnrollmentOriginV1 {
    ReferenceEnrollmentOriginV1 {
        rustc_invocation_sha256: [23; 32],
        native_policy_sha256: [29; 32],
        policy_generation: 37,
        mapping_ordinal: 7,
    }
}

fn expectations() -> [NativeConditionalCpuExpectationV1; 2] {
    [
        NativeConditionalCpuExpectationV1 {
            semantic_root: 9,
            origin: NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
        },
        NativeConditionalCpuExpectationV1 {
            semantic_root: 4,
            origin: NativeConditionalCpuOriginExpectationV1::ReferenceEnrollmentV1(origin()),
        },
    ]
}

// The existing independently assembled V1 golden is inert codec input only.
fn registration_wire() -> Vec<u8> {
    let prefix = "463243505531000001000000be0200004013131313131313131313131313131313131313131313131313131313131313130200000011000000666978747572653a3a72656769737465720400000066696c6c";
    let suffix = "000000000100000002000b020000000005010001000b5468a10e3a0e89f44c6c7a7d613178ce7af9e472b7b525591ee1a714c95d787502000000030000000200000004000000000000000003000000000b010000000000000001000000000000000200000001000000000002010b00002a4200000000000000000000000000000000000100000000000000000000000000000000010000000000000000010000000000000002010b00002a420000000000000000000000000002010b00002a42000000000000000000000000";
    let text = format!("{prefix}{}{suffix}", "0b".repeat(208) + &"11".repeat(208));
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn policy_wire(origin: ReferenceEnrollmentOriginV1) -> Vec<u8> {
    budgeted(|budget| {
        with_decoded_native_cpu_input_v1(&registration_wire(), budget, |owner, budget| {
            let input = owner.input_v1();
            let policy = NativeCpuPolicyInputV2 {
                association: NativeCpuPolicyAssociationV2 {
                    semantic_mir_sha256: input.association.semantic_mir_sha256,
                    semantic_root: input.association.semantic_root,
                    logical_kernel_name: input.association.logical_kernel_name,
                    origin,
                },
                kernel: input.kernel,
                reference: input.reference,
                replay: input.replay,
            };
            with_encoded_native_cpu_policy_input_v2(policy, budget, |bytes, _, _| bytes.to_vec())
                .unwrap()
        })
        .unwrap()
    })
}

#[test]
fn complete_mixed_expectations_follow_source_order_not_numeric_root_order() {
    fixture(|packet, accepted| {
        budgeted(|b| {
            let expected = expectations();
            let mode = CpuReplayMode::Expected(&expected);
            mode.require_roster(packet.roots, accepted, b).unwrap();
            assert_eq!(b.work(), 10);
            for (ordinal, row) in expected.iter().enumerate() {
                assert_eq!(mode.at(ordinal, row.semantic_root, b).unwrap(), row.origin);
            }
            assert_eq!(b.work(), 14);
            assert_eq!(b.storage(), FLOOR);
        })
    });
}

#[test]
fn missing_extra_reordered_wrong_and_duplicate_expectations_refuse() {
    fixture(|packet, accepted| {
        let exact = expectations();
        let extra = [exact[0], exact[1], exact[0]];
        let reordered = [exact[1], exact[0]];
        let duplicate = [exact[0], exact[0]];
        let wrong = [
            NativeConditionalCpuExpectationV1 {
                semantic_root: 8,
                ..exact[0]
            },
            exact[1],
        ];
        for expected in [&[][..], &exact[..1], &extra, &reordered, &duplicate, &wrong] {
            budgeted(|b| {
                assert!(
                    CpuReplayMode::Expected(expected)
                        .require_roster(packet.roots, accepted, b)
                        .is_err()
                )
            });
        }
        budgeted(|b| {
            assert!(
                CpuReplayMode::Expected(&exact)
                    .require_roster(packet.roots, &accepted[..1], b)
                    .is_err()
            )
        });
        // Even mutually matching duplicate rosters must not pass this boundary.
        let roots = [packet.roots[0], packet.roots[0]];
        let policies = [
            NativeConditionalRootPolicyV2 { ..accepted[0] },
            NativeConditionalRootPolicyV2 { ..accepted[0] },
        ];
        budgeted(|b| {
            assert!(matches!(
                CpuReplayMode::Expected(&duplicate).require_roster(&roots, &policies, b),
                Err(E(Cause::Invalid(
                    "duplicate independent CPU origin expectation root"
                )))
            ))
        });
    });
}

#[test]
fn selected_ordinal_cannot_be_missing_or_rebound() {
    let rows = expectations();
    for (ordinal, root) in [(2, 4), (0, 4), (1, 9)] {
        budgeted(|b| {
            assert!(CpuReplayMode::Expected(&rows).at(ordinal, root, b).is_err());
            assert_eq!(b.work(), 2);
        });
    }
}

#[test]
fn empty_rosters_and_independently_changed_policy_root_refuse() {
    budgeted(|b| {
        assert!(
            CpuReplayMode::Expected(&[])
                .require_roster(&[], &[], b)
                .is_err()
        );
    });
    fixture(|packet, accepted| {
        budgeted(|b| {
            let expected = expectations();
            let changed = [
                NativeConditionalRootPolicyV2 {
                    semantic_root: 8,
                    ..accepted[0]
                },
                NativeConditionalRootPolicyV2 { ..accepted[1] },
            ];
            assert!(
                CpuReplayMode::Expected(&expected)
                    .require_roster(packet.roots, &changed, b)
                    .is_err()
            );
        })
    });
}

#[test]
fn expectation_backing_and_roster_work_have_exact_thresholds() {
    let rows = expectations();
    let mode = CpuReplayMode::Expected(&rows);
    let bytes = std::mem::size_of_val(&rows);
    assert_eq!(mode.backing_bytes(), bytes);
    for paid in [bytes - 1, bytes] {
        let mut work = Work::new(1);
        let mut b = Budget::new(&mut work, paid);
        b.reserve_storage(paid).unwrap();
        let result = mode.require_backing(&mut b);
        if paid == bytes {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(E(Cause::Resource(Resource::Accounting)))
            ));
        }
        assert_eq!(b.storage(), paid);
        assert_eq!(b.work(), 1);
    }
    fixture(|packet, accepted| {
        for limit in [9, 10] {
            let mut work = Work::new(limit);
            let mut b = Budget::new(&mut work, usize::MAX);
            assert_eq!(
                mode.require_roster(packet.roots, accepted, &mut b).is_ok(),
                limit == 10
            );
            assert_eq!(b.failed_work().is_some(), limit == 9);
        }
    });
}

#[test]
fn legacy_selection_keeps_zero_extra_roster_work_and_storage() {
    fixture(|packet, accepted| {
        budgeted(|b| {
            let mode = CpuReplayMode::RegistrationOnly;
            mode.require_backing(b).unwrap();
            mode.require_roster(packet.roots, accepted, b).unwrap();
            assert_eq!(
                mode.at(0, 9, b).unwrap(),
                NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1
            );
            assert_eq!(mode.backing_bytes(), 0);
            assert_eq!(b.work(), 0);
            assert_eq!(b.storage(), FLOOR);
        })
    });
}

#[test]
fn explicit_selection_rejects_opposite_codec_without_callback_or_fallback() {
    let registration = registration_wire();
    let policy = policy_wire(origin());
    for (bytes, selection) in [
        (
            &registration,
            NativeConditionalCpuOriginExpectationV1::ReferenceEnrollmentV1(origin()),
        ),
        (
            &policy,
            NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
        ),
    ] {
        budgeted(|b| {
            let calls = Cell::new(0);
            let result: Result<(), E> = cpu_origin::with_decoded(bytes, selection, b, |_, _| {
                calls.set(calls.get() + 1);
                Ok(())
            });
            assert!(matches!(result, Err(E(Cause::Cpu(_)))));
            assert_eq!(calls.get(), 0);
            assert_eq!(b.storage(), FLOOR);
        });
    }
}

#[test]
fn every_origin_field_is_checked_before_decoded_consumer() {
    let wire = policy_wire(origin());
    for field in 0..5 {
        let mut expected = origin();
        match field {
            0 => expected.rustc_invocation_sha256[31] ^= 1,
            1 => expected.native_policy_sha256[31] ^= 1,
            2 => expected.policy_generation += 1,
            3 => expected.mapping_ordinal += 1,
            _ => (),
        }
        budgeted(|b| {
            let calls = Cell::new(0);
            let result: Result<(), E> = cpu_origin::with_decoded(
                &wire,
                NativeConditionalCpuOriginExpectationV1::ReferenceEnrollmentV1(expected),
                b,
                |decoded, _| {
                    calls.set(calls.get() + 1);
                    assert!(matches!(decoded, DecodedCpu::Policy(_)));
                    assert_eq!(decoded.subjects().semantic_root, 2);
                    Ok(())
                },
            );
            if field == 4 {
                result.unwrap();
            } else {
                assert!(matches!(
                    result,
                    Err(E(Cause::Invalid(
                        "independent CPU enrollment-origin expectation"
                    )))
                ));
            }
            assert_eq!(calls.get(), usize::from(field == 4));
            assert_eq!(b.storage(), FLOOR);
        });
    }
}

#[test]
fn registration_dispatch_preserves_direct_codec_metering_and_subjects() {
    let bytes = registration_wire();
    let direct = budgeted(|b| {
        with_decoded_native_cpu_input_v1(&bytes, b, |_, _| ()).unwrap();
        b.work()
    });
    budgeted(|b| {
        let result: Result<(), E> = cpu_origin::with_decoded(
            &bytes,
            NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
            b,
            |decoded, _| {
                assert!(matches!(decoded, DecodedCpu::Registration(_)));
                let subject = decoded.subjects();
                assert_eq!(subject.semantic_mir_sha256, [19; 32]);
                assert_eq!(subject.semantic_root, 2);
                assert_eq!(subject.logical_kernel_name, "fill");
                assert_eq!(subject.kernel.def_path_hash, [11; 16]);
                assert_eq!(subject.reference.def_path_hash, [17; 16]);
                Ok(())
            },
        );
        result.unwrap();
        assert_eq!(b.work(), direct);
        assert_eq!(b.storage(), FLOOR);
    });
}

#[test]
fn public_opt_in_refuses_unpaid_expectations_before_packet_decode() {
    let expected = expectations();
    fixture(|_, accepted| {
        budgeted(|b| {
            assert!(std::mem::size_of_val(&expected) > FLOOR);
            assert!(matches!(
                validate_native_conditional_source_packet_with_cpu_origins_v2(
                    b"", accepted, &expected, b
                ),
                Err(E(Cause::Resource(Resource::Accounting)))
            ));
            assert_eq!(b.storage(), FLOOR);
        })
    });
}

#[test]
fn policy_comparison_work_exact_and_one_short_preserves_original_account() {
    let wire = policy_wire(origin());
    let selected = NativeConditionalCpuOriginExpectationV1::ReferenceEnrollmentV1(origin());
    let required = budgeted(|b| {
        let result: Result<(), E> = cpu_origin::with_decoded(&wire, selected, b, |_, _| Ok(()));
        result.unwrap();
        b.work()
    });
    for limit in [required - 1, required] {
        let mut work = Work::new(limit);
        let mut b = Budget::new(&mut work, usize::MAX);
        b.reserve_storage(FLOOR).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let calls = Cell::new(0);
        let result: Result<(), E> = cpu_origin::with_decoded(&wire, selected, &mut b, |_, _| {
            calls.set(calls.get() + 1);
            Ok(())
        });
        assert_eq!(result.is_ok(), limit == required);
        assert_eq!(calls.get(), usize::from(limit == required));
        assert_eq!(b.failed_work().is_some(), limit != required);
        assert_eq!(b.storage(), FLOOR);
        assert!(b.work_ledger_identity_v1() == ledger);
    }
}

#[test]
fn selected_decode_account_failure_overrides_refundable_consumer_error() {
    use crate::portable_reference_v1::codec::NativeCpuCodecErrorV1 as CodecError;
    let wire = policy_wire(origin());
    for foreign in [false, true] {
        budgeted(|b| {
            let ledger = b.work_ledger_identity_v1();
            let protected = Cell::new(0);
            let result: Result<(), E> = cpu_origin::with_decoded(
                &wire,
                NativeConditionalCpuOriginExpectationV1::ReferenceEnrollmentV1(origin()),
                b,
                |_, b| {
                    protected.set(b.storage());
                    if foreign {
                        // Inert foreign test meter outlives the callback's budget lifetime.
                        *b = Budget::new(Box::leak(Box::new(Work::new(usize::MAX))), usize::MAX);
                        b.reserve_storage(protected.get()).unwrap();
                    } else {
                        b.release_storage(1).unwrap();
                    }
                    Err(E::invalid("inert consumer refusal"))
                },
            );
            assert!(matches!(
                result,
                Err(E(Cause::Cpu(CodecError::Resource(Resource::Accounting))))
            ));
            assert_eq!(b.storage(), protected.get() - usize::from(!foreign));
            assert_eq!(b.work_ledger_identity_v1() != ledger, foreign);
        });
    }
}

// Type-checks the public API without constructing a proof, Request or final owner.
#[allow(dead_code)]
fn public_source_and_final_call_shapes(
    bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected: &[NativeConditionalCpuExpectationV1],
    inputs: crate::NativeConditionalFinalInputsV2<'_, '_, '_>,
    budget: &mut Budget<'_>,
) {
    let _ = crate::validate_native_conditional_source_packet_with_cpu_origins_v2(
        bytes, accepted, expected, budget,
    );
    let _ = crate::validate_native_conditional_source_through_f_with_cpu_origins_v2(
        bytes, accepted, expected, inputs, budget,
    );
}

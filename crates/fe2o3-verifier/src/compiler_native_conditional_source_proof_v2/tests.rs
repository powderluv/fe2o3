//! Inert component/refusal cases only: no Request, signed success or execution.
use super::*;
use crate::InertFunctionalRefinementReceiptSignatureV2 as Signature;
use crate::NativeCompilerStagingCommitmentV1 as Commitment;
use fe2o3_functional_proof::{
    FUNCTIONAL_REFINEMENT_RECEIPT_WIRE_BYTES_V2, FunctionalRefinementBoundaryV2 as Boundary,
    FunctionalRefinementImportPolicyV2 as Policy, VerusToolchainIdentityV2 as Toolchain,
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_lower_mir_kernel::{
    ProductionSourceLaunchInputV1 as Geometry, ProductionSourceLaunchRootInputV1 as Launch,
};
use fe2o3_pliron::ProductionRefinementStagingPolicyV2 as EffectPolicy;
use fe2o3_proof_contracts::DigestV1;

#[path = "tests/resources_tests.rs"]
mod resources_tests;
#[path = "tests/rows.rs"]
mod rows;
#[path = "tests/selector.rs"]
mod selector;

#[path = "tests/cpu_origins.rs"]
mod cpu_origins;

const FLOOR: usize = 43;
// RFC 8032 public key. No secret key, signature generation or successful proof.
const KEY: [u8; 32] = [
    0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
    0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a,
];
fn digest(n: u8) -> DigestV1 {
    DigestV1::from_untrusted_bytes([n; 32])
}
fn toolchain() -> Toolchain {
    Toolchain::new(digest(1), digest(2), digest(3), digest(4), digest(5)).unwrap()
}
fn policy() -> Policy {
    Policy::new(KEY, toolchain(), Boundary::SafeReferenceMirToKernelMir).unwrap()
}
fn signature() -> Signature {
    Signature::from_untrusted_parts([0; FUNCTIONAL_REFINEMENT_RECEIPT_WIRE_BYTES_V2], KEY)
}
fn commitment() -> Commitment {
    Commitment {
        receipt: [1; 32],
        effect: [2; 32],
        signer: *policy().signer_identity().as_bytes(),
        execution: [3; 32],
        toolchain: [[4; 32]; 5],
    }
}
pub(super) fn fixture<R>(
    run: impl FnOnce(
        NativeConditionalSourcePacketInputV2<'_>,
        &[NativeConditionalRootPolicyV2<'_>],
    ) -> R,
) -> R {
    let formula = policy();
    let effects = EffectPolicy::new([formula.signer_identity()], toolchain()).unwrap();
    let signatures = [signature(), signature()];
    // Equal inert bytes and repeated signers are NOT an additional uniqueness rule.
    let commitments = [commitment(), commitment()];
    let roots = [9, 4].map(|semantic_root| NativeConditionalSourceRootV2 {
        semantic_root,
        launch_rank: 1,
        launch: Launch::new(
            if semantic_root == 9 {
                "first"
            } else {
                "second"
            },
            [semantic_root as u8; 32],
            Geometry::new(1, Some([64, 1, 1]), [8, 1, 1]),
        ),
        induction_bytes: b"inert induction",
        recipe_bytes: b"inert recipe",
        source_rows_bytes: b"inert rows",
        ranked_ir: "diagnostic",
        cpu_input_bytes: b"inert CPU",
        staging_commitments: &commitments,
        effect_receipts: &signatures,
        formula_receipt: &signatures[0],
    });
    let accepted = [9, 4].map(|semantic_root| NativeConditionalRootPolicyV2 {
        semantic_root,
        effects: &effects,
        formula: &formula,
    });
    run(
        NativeConditionalSourcePacketInputV2 {
            semantic_mir: b"inert MIR",
            native_module: b"inert N",
            canonical_kernel_order: &[1, 0],
            roots: &roots,
        },
        &accepted,
    )
}
fn budgeted<R>(run: impl FnOnce(&mut Budget<'_>) -> R) -> R {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(FLOOR).unwrap();
    run(&mut budget)
}

#[test]
fn complete_policy_order_and_actual_kernel_id_order_not_arbitrary_permutation() {
    fixture(|packet, accepted| {
        budgeted(|b| {
            reconstruct::roster(&packet, accepted, b).unwrap();
            let before = b.work();
            let wrong = NativeConditionalSourcePacketInputV2 {
                canonical_kernel_order: &[0, 1],
                ..packet
            };
            assert!(matches!(
                reconstruct::roster(&wrong, accepted, b),
                Err(E(Cause::Invalid("actual full-binding KernelId order")))
            ));
            assert!(b.work() > before);
            assert_eq!(b.storage(), FLOOR);
        })
    });
}

#[test]
fn omissions_duplicate_ordinals_and_policy_substitution_refuse_without_deduplication() {
    for order in [&[0][..], &[1, 1][..], &[0, 2][..]] {
        fixture(|packet, accepted| {
            budgeted(|b| {
                let packet = NativeConditionalSourcePacketInputV2 {
                    canonical_kernel_order: order,
                    ..packet
                };
                assert!(reconstruct::roster(&packet, accepted, b).is_err());
            })
        });
    }
    fixture(|packet, accepted| {
        budgeted(|b| {
            assert!(reconstruct::roster(&packet, &accepted[..1], b).is_err());
            let reordered = [
                NativeConditionalRootPolicyV2 {
                    semantic_root: 4,
                    ..accepted[0]
                },
                NativeConditionalRootPolicyV2 {
                    semantic_root: 9,
                    ..accepted[1]
                },
            ];
            assert!(reconstruct::roster(&packet, &reordered, b).is_err());
            let duplicate = [
                NativeConditionalRootPolicyV2 {
                    semantic_root: 9,
                    ..accepted[0]
                },
                NativeConditionalRootPolicyV2 {
                    semantic_root: 9,
                    ..accepted[1]
                },
            ];
            assert!(reconstruct::roster(&packet, &duplicate, b).is_err());
        })
    });
}

#[test]
fn cpu_association_checks_source_root_and_logical_name_not_registration_origin() {
    use crate::portable_reference_v1::codec::NativeCpuAssociationV1;
    fixture(|packet, _| {
        budgeted(|b| {
            for mutation in 0..5 {
                let cpu = NativeCpuAssociationV1 {
                    semantic_mir_sha256: if mutation == 1 { [8; 32] } else { [7; 32] },
                    semantic_root: if mutation == 2 { 4 } else { 9 },
                    logical_kernel_name: if mutation == 3 { "other" } else { "first" },
                    registration_path: if mutation == 4 {
                        "untrusted/replacement"
                    } else {
                        "untrusted/original"
                    },
                };
                let result = root::association(cpu, [7; 32], &packet.roots[0], b);
                assert_eq!(result.is_ok(), mutation == 0 || mutation == 4);
            }
            assert_eq!(b.storage(), FLOOR);
        })
    });
}

#[test]
fn accepted_repeated_signer_is_membership_not_proof_or_unique_receipt_rule() {
    let policy = policy();
    let accepted = EffectPolicy::new(
        [policy.signer_identity(), policy.signer_identity()],
        toolchain(),
    )
    .unwrap();
    budgeted(|b| {
        for _ in 0..2 {
            root::accepted_effect(&signature(), &commitment(), &accepted, |n| b.charge_work(n))
                .unwrap();
        }
        assert_eq!(b.work(), 130);
        let foreign = EffectPolicy::new([digest(90)], toolchain()).unwrap();
        assert!(matches!(
            root::accepted_effect(&signature(), &commitment(), &foreign, |n| b.charge_work(n)),
            Err(crate::NativeCompilerSourceProofErrorV1::Mismatch(_))
        ));
        let mut changed = commitment();
        changed.signer = [90; 32];
        assert!(
            root::accepted_effect(&signature(), &changed, &accepted, |n| b.charge_work(n)).is_err()
        );
    });
}

#[test]
fn public_consumer_rejects_inert_native_source_and_v1_frames_without_owner() {
    use crate::compiler_native_conditional_source_packet_v2::encode_native_conditional_source_packet_v2 as encode;
    fixture(|packet, accepted| {
        budgeted(|b| {
            let (bytes, storage) = encode(packet, b).unwrap();
            b.reserve_storage(storage.retained_storage()).unwrap();
            let protected = b.storage();
            let result = validate_native_conditional_source_packet_v2(&bytes, accepted, b);
            assert!(matches!(result, Err(E(Cause::Native(_)))));
            // Unknown inherited leaf errors retain terminal reservations, but this
            // refusal occurs before source allocation. Codec metadata is released.
            assert_eq!(b.storage(), protected);
            for bytes in [b"F2NSRC1\0".as_slice(), b"".as_slice()] {
                assert!(matches!(
                    validate_native_conditional_source_packet_v2(bytes, accepted, b),
                    Err(E(Cause::Packet(_)))
                ));
            }
            assert_eq!(b.storage(), protected);
        })
    });
}

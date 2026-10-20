//! Inert projection/accounting components. No source owner, Request, compiler
//! capture, authenticated carriage, imported proof or execution is fabricated.
use super::*;
use fe2o3_compiler_lineage::{
    MAX_NATIVE_CONDITIONAL_STORAGE_V1 as LIMIT,
    RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1 as CODEC_WORKING,
    RustcEnrollmentInventoryHeaderV1, RustcEnrollmentInventoryInputV1,
    encode_rustc_enrollment_inventory_v1, read_rustc_enrollment_inventory_v1,
};
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_mir_model::semantic_mir_v1::*;
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

const FLOOR: usize = 37;
fn expectation() -> NativeConditionalCpuMappingExpectationV1 {
    NativeConditionalCpuMappingExpectationV1 {
        rustc_invocation_sha256: [11; 32],
        native_policy_sha256: [13; 32],
        policy_generation: 17,
        enrollment_binding_count: 1,
    }
}
fn rows() -> [RustcEnrollmentInventoryRootV1; 2] {
    [(4, "second", 1), (9, "first", 0)].map(|(root, name, tag)| RustcEnrollmentInventoryRootV1 {
        semantic_root: root,
        origin_tag: tag,
        descriptor_ordinal: 0,
        logical_name_len: name.len() as u32,
        logical_name_sha256: Sha256::digest(name.as_bytes()).into(),
        kernel_binding: [root as u8; 32],
        kernel_instance: [root as u8; 32],
        reference_instance: [root as u8 + 100; 32],
    })
}
fn wire(
    rows: &[RustcEnrollmentInventoryRootV1],
    expected: &NativeConditionalCpuMappingExpectationV1,
) -> Vec<u8> {
    // Ordinary inert fixture construction, outside the measured consumer scope.
    encode_rustc_enrollment_inventory_v1(
        RustcEnrollmentInventoryInputV1 {
            legacy_inventory: b"unchanged inert legacy inventory",
            header: RustcEnrollmentInventoryHeaderV1 {
                kernel_count: rows.len() as u32,
                enrollment_binding_count: expected.enrollment_binding_count,
                invocation_identity: expected.rustc_invocation_sha256,
                native_policy_identity: expected.native_policy_sha256,
                native_policy_generation: expected.policy_generation,
            },
            roots: rows,
        },
        LIMIT,
        |_| Ok::<_, Resource>(()),
    )
    .unwrap()
}
fn with_mapping<R>(
    rows: &[RustcEnrollmentInventoryRootV1],
    run: impl FnOnce(&NativeConditionalCpuMappingContextV1<'_>, &mut Budget<'_>) -> R,
) -> R {
    with_expected_mapping(rows, &expectation(), run)
}
fn with_expected_mapping<R>(
    rows: &[RustcEnrollmentInventoryRootV1],
    expected: &NativeConditionalCpuMappingExpectationV1,
    run: impl FnOnce(&NativeConditionalCpuMappingContextV1<'_>, &mut Budget<'_>) -> R,
) -> R {
    let bytes = wire(rows, expected);
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, LIMIT);
    budget
        .reserve_storage(
            FLOOR
                + bytes.capacity()
                + size_of::<Vec<u8>>()
                + CODEC_WORKING
                + size_of::<NativeConditionalCpuMappingExpectationV1>()
                + size_of::<NativeConditionalCpuMappingContextV1<'_>>()
                + std::mem::size_of_val(rows),
        )
        .unwrap();
    let decoded =
        read_rustc_enrollment_inventory_v1(&bytes, LIMIT, |n| budget.charge_work(n)).unwrap();
    let context = NativeConditionalCpuMappingContextV1 {
        inventory: &decoded,
        expected,
    };
    run(&context, &mut budget)
}

// These ordinary declarations exercise the same private projection used with
// the actual reconstructed source. They are NOT an admitted semantic owner.
fn declarations() -> Vec<SemanticFunctionDeclV1> {
    (0..10u8)
        .map(|index| {
            let is_kernel = index == 4 || index == 9;
            let function = SemanticFunctionDeclV1::new(
                SemanticFunctionIdentityV1::from_sha256([index; 32]),
                if is_kernel {
                    SemanticFunctionRoleV1::KernelRoot
                } else {
                    SemanticFunctionRoleV1::InternalHelper
                },
                SemanticItemDefinitionIdentityV1::from_sha256([index; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([index; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([index; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([index; 32]),
                SemanticSourceProvenanceV1::unavailable(),
                SemanticFunctionAbiV1::new(
                    SemanticAbiIdentityV1::from_sha256([index; 32]),
                    SemanticLayoutIdentityV1::from_sha256([31; 32]),
                    SemanticCanonAbiV1::Rust,
                    false,
                    false,
                    vec![],
                    SemanticAbiValueV1::new(
                        SemanticTypeIdV1::from_index(0),
                        SemanticAbiPassModeV1::Ignore,
                    ),
                )
                .unwrap(),
                vec![],
                SemanticBlockIdV1::from_index(0),
                vec![],
            )
            .unwrap();
            if is_kernel {
                function.with_kernel_entry(SemanticKernelEntryV1::new(
                    SemanticLinkSymbolV1::new(format!("kernel_{index}").into_bytes()).unwrap(),
                    SemanticKernelBindingIdentityV1::from_sha256([index; 32]),
                    SemanticKernelSourceContractV1::new(None, None, None).unwrap(),
                ))
            } else {
                function
            }
        })
        .collect()
}
fn source_roots() -> [SemanticFunctionIdV1; 2] {
    [9, 4].map(SemanticFunctionIdV1::from_index)
}

#[test]
fn mixed_mapping_projects_noncontiguous_packet_order_and_independent_reference() {
    with_mapping(&rows(), |context, budget| {
        super::super::tests::fixture(|packet, accepted| {
            let selection = ReplaySelection::Mapping(context);
            selection.require_backing(budget).unwrap();
            selection
                .require_roster(packet.roots, accepted, budget)
                .unwrap();
            let functions = declarations();
            let before = budget.storage();
            let ledger = budget.work_ledger_identity_v1();
            projected_scope(
                2,
                budget,
                |out, b| project_components(context, &functions, &source_roots(), &packet, out, b),
                |projected, b| -> Result<(), E> {
                    let first = projected.at(0, 9, b)?;
                    assert_eq!(first.origin, Origin::SourceRegistrationV1);
                    let second = projected.at(1, 4, b)?;
                    assert_eq!(
                        second.origin,
                        Origin::ReferenceEnrollmentV1(ReferenceEnrollmentOriginV1 {
                            rustc_invocation_sha256: [11; 32],
                            native_policy_sha256: [13; 32],
                            policy_generation: 17,
                            mapping_ordinal: 0,
                        })
                    );
                    assert_eq!(second.functions.unwrap().reference, [104; 32]);
                    assert!(
                        functions
                            .iter()
                            .all(|f| f.identity().as_bytes() != &[104; 32])
                    );
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(budget.storage(), before);
            assert!(budget.work_ledger_identity_v1() == ledger);
        })
    });
}

#[test]
fn enrollment_ordinals_follow_mapping_not_canonical_function_order() {
    let mut values = rows();
    for (row, descriptor_ordinal) in values.iter_mut().zip([1, 0]) {
        row.origin_tag = 1;
        row.descriptor_ordinal = descriptor_ordinal;
    }
    let expected = NativeConditionalCpuMappingExpectationV1 {
        enrollment_binding_count: 2,
        ..expectation()
    };
    with_expected_mapping(&values, &expected, |context, budget| {
        super::super::tests::fixture(|packet, accepted| {
            ReplaySelection::Mapping(context)
                .require_roster(packet.roots, accepted, budget)
                .unwrap();
            projected_scope(
                2,
                budget,
                |out, b| {
                    project_components(context, &declarations(), &source_roots(), &packet, out, b)
                },
                |projected, b| -> Result<(), E> {
                    for (packet_ordinal, root, descriptor_ordinal) in [(0, 9, 0), (1, 4, 1)] {
                        let actual = projected.at(packet_ordinal, root, b)?;
                        assert_eq!(
                            actual.origin,
                            Origin::ReferenceEnrollmentV1(ReferenceEnrollmentOriginV1 {
                                rustc_invocation_sha256: expected.rustc_invocation_sha256,
                                native_policy_sha256: expected.native_policy_sha256,
                                policy_generation: expected.policy_generation,
                                mapping_ordinal: descriptor_ordinal,
                            })
                        );
                    }
                    Ok(())
                },
            )
            .unwrap();
        });
    });
}

#[test]
fn every_independent_header_axis_refuses_even_with_a_valid_coherent_frame() {
    with_mapping(&rows(), |context, b| {
        for mutation in 0..5 {
            let mut expected = *context.expected;
            match mutation {
                0 => expected.rustc_invocation_sha256[0] ^= 1,
                1 => expected.native_policy_sha256[0] ^= 1,
                2 => expected.policy_generation += 1,
                3 => expected.enrollment_binding_count = 0,
                _ => expected.enrollment_binding_count = MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 + 1,
            }
            let foreign = NativeConditionalCpuMappingContextV1 {
                inventory: context.inventory,
                expected: &expected,
            };
            assert!(matches!(
                require_header(&foreign, b),
                Err(E(Cause::Invalid(_)))
            ));
        }
    });
}

#[test]
fn coherent_kernel_identity_binding_name_and_membership_mutants_refuse() {
    for mutation in 0..8 {
        let mut values = rows();
        match mutation {
            0 => values[0].kernel_instance[0] ^= 1,
            1 => values[0].kernel_binding[0] ^= 1,
            2 => values[0].logical_name_len += 1,
            3 => values[0].logical_name_sha256[0] ^= 1,
            4 => values[0].semantic_root = 3,
            _ => {}
        }
        with_mapping(&values, |context, b| {
            super::super::tests::fixture(|packet, _| {
                let mut functions = declarations();
                if mutation == 5 {
                    functions[4] = functions[4]
                        .clone()
                        .with_role(SemanticFunctionRoleV1::InternalHelper);
                }
                if mutation == 6 {
                    functions[4] = functions[3].clone();
                }
                let roots = if mutation == 7 {
                    [4, 9].map(SemanticFunctionIdV1::from_index)
                } else {
                    source_roots()
                };
                let result: Result<(), E> = projected_scope(
                    2,
                    b,
                    |out, b| project_components(context, &functions, &roots, &packet, out, b),
                    |_, _| panic!("invalid mapping reached CPU projection callback"),
                );
                assert!(matches!(result, Err(E(Cause::Invalid(_)))));
            })
        });
    }
}

#[test]
fn missing_extra_and_duplicate_packet_roots_refuse_complete_projection() {
    with_mapping(&rows(), |context, b| {
        super::super::tests::fixture(|packet, accepted| {
            assert!(
                ReplaySelection::Mapping(context)
                    .require_roster(&packet.roots[..1], &accepted[..1], b)
                    .is_err()
            );
            let duplicate = [packet.roots[0], packet.roots[0]];
            let changed = NativeConditionalSourcePacketInputV2 {
                roots: &duplicate,
                ..packet
            };
            let result: Result<(), E> = projected_scope(
                2,
                b,
                |out, b| {
                    project_components(
                        context,
                        &declarations(),
                        &[SemanticFunctionIdV1::from_index(9); 2],
                        &changed,
                        out,
                        b,
                    )
                },
                |_, _| panic!("duplicate root reached callback"),
            );
            assert!(matches!(result, Err(E(Cause::Invalid(_)))));
        })
    });
    let one = [rows()[0]];
    with_mapping(&one, |context, b| {
        super::super::tests::fixture(|packet, _| {
            let result: Result<(), E> = projected_scope(
                2,
                b,
                |out, b| {
                    project_components(context, &declarations(), &source_roots(), &packet, out, b)
                },
                |_, _| panic!("missing mapping reached callback"),
            );
            assert!(matches!(result, Err(E(Cause::Invalid(_)))));
        })
    });
    let extra = [
        rows()[0],
        rows()[1],
        RustcEnrollmentInventoryRootV1 {
            semantic_root: 10,
            ..rows()[1]
        },
    ];
    with_mapping(&extra, |context, b| {
        super::super::tests::fixture(|packet, _| {
            let result: Result<(), E> = projected_scope(
                2,
                b,
                |out, b| {
                    project_components(context, &declarations(), &source_roots(), &packet, out, b)
                },
                |_, _| panic!("extra mapping reached callback"),
            );
            assert!(matches!(result, Err(E(Cause::Invalid(_)))));
        })
    });
}

fn identity(value: u8) -> crate::portable_reference_v1::ReferenceFunctionIdentityV1 {
    crate::portable_reference_v1::ReferenceFunctionIdentityV1 {
        def_path_hash: [value; 16],
        function_sha256: [value; 32],
        item_definition_sha256: [value; 32],
        monomorphization_sha256: [value; 32],
        generic_type_arguments_sha256: [value; 32],
        const_generic_arguments_sha256: [value; 32],
        rustc_mir_body_sha256: [value; 32],
    }
}
#[test]
fn both_cpu_function_axes_must_match_after_origin_selection() {
    let row = ProjectedRoot {
        semantic_root: 4,
        origin: Origin::SourceRegistrationV1,
        kernel: [4; 32],
        reference: [104; 32],
    };
    for mutation in 0..3 {
        let kernel = identity(if mutation == 1 { 5 } else { 4 });
        let reference = identity(if mutation == 2 { 105 } else { 104 });
        let cpu = CpuSubjects {
            semantic_mir_sha256: [0; 32],
            semantic_root: 4,
            logical_kernel_name: "second",
            kernel: &kernel,
            reference: &reference,
        };
        let mut work = Work::new(66);
        let mut budget = Budget::new(&mut work, FLOOR);
        budget.reserve_storage(FLOOR).unwrap();
        let result = RootExpectation {
            origin: row.origin,
            functions: Some(&row),
        }
        .require_subjects(&cpu, &mut budget);
        assert_eq!(result.is_ok(), mutation == 0);
        assert_eq!((budget.work(), budget.storage()), (66, FLOOR));
    }
}

#[test]
fn legacy_selection_preserves_exact_representation_work_and_header_bills() {
    let rows = [NativeConditionalCpuExpectationV1 {
        semantic_root: 9,
        origin: Origin::SourceRegistrationV1,
    }];
    for mode in [
        CpuReplayMode::RegistrationOnly,
        CpuReplayMode::Expected(&rows),
    ] {
        let selection = ReplaySelection::Legacy(mode);
        assert_eq!(selection.backing_bytes(), mode.backing_bytes());
        assert_eq!(
            selection.working_header(),
            if matches!(mode, CpuReplayMode::RegistrationOnly) {
                0
            } else {
                size_of::<CpuReplayMode<'_>>()
            }
        );
        let mut work = Work::new(100);
        let mut b = Budget::new(&mut work, 1024);
        b.reserve_storage(1024).unwrap();
        assert_eq!(
            ProjectedSelection::Legacy(mode)
                .at(0, 9, &mut b)
                .unwrap()
                .origin,
            Origin::SourceRegistrationV1
        );
        assert_eq!(
            b.work(),
            if matches!(mode, CpuReplayMode::RegistrationOnly) {
                0
            } else {
                2
            }
        );
        assert_eq!(b.storage(), 1024);
    }
}

fn measured_scope(budget: &mut Budget<'_>, mode: u8, drops: &Cell<usize>) -> Result<(), E> {
    struct DropProbe<'a>(&'a Cell<usize>);
    impl Drop for DropProbe<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    projected_scope(
        1,
        budget,
        |rows, b| {
            b.charge_work(3)?;
            rows.push(ProjectedRoot {
                semantic_root: 4,
                origin: Origin::SourceRegistrationV1,
                kernel: [4; 32],
                reference: [104; 32],
            });
            Ok(())
        },
        |selection, b| {
            let _owner = DropProbe(drops);
            selection.at(0, 4, b)?;
            if mode == 1 {
                return Err(E::invalid("inert component callback refusal"));
            }
            if mode == 2 {
                panic!("inert component callback unwind");
            }
            if mode == 3 {
                b.release_storage(1)?;
                return Err(E::invalid("original floor damage must override this error"));
            }
            if mode == 4 {
                let stored = b.storage();
                // One bounded adversarial test ledger, never a production reset.
                *b = Budget::new(Box::leak(Box::new(Work::new(100))), LIMIT);
                b.reserve_storage(stored)?;
            }
            b.charge_work(7)?;
            Ok(())
        },
    )
}

#[test]
fn projection_actual_capacity_exact_one_short_work_and_storage_preserve_floor() {
    let drops = Cell::new(0);
    let mut work = Work::new(1_000_000);
    let mut budget = Budget::new(&mut work, LIMIT);
    budget.reserve_storage(FLOOR).unwrap();
    measured_scope(&mut budget, 0, &drops).unwrap();
    let (used, peak) = (budget.work(), budget.peak_storage());
    assert_eq!(budget.storage(), FLOOR);
    assert!(peak >= FLOOR + size_of::<ProjectedRoot>());
    for (work_limit, storage_limit, success) in [
        (used, peak, true),
        (used - 1, peak, false),
        (used, peak - 1, false),
    ] {
        let mut work = Work::new(work_limit);
        let mut b = Budget::new(&mut work, storage_limit);
        b.reserve_storage(FLOOR).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let storage_account = b.storage_account_identity_v1();
        let result = measured_scope(&mut b, 0, &drops);
        assert_eq!(result.is_ok(), success);
        assert!(b.work_ledger_identity_v1() == ledger);
        assert_eq!(b.storage_account_identity_v1(), storage_account);
        if success {
            assert_eq!(
                (b.work(), b.peak_storage(), b.storage()),
                (used, peak, FLOOR)
            );
        } else {
            assert!(b.storage() > FLOOR);
        }
    }
}

#[test]
fn projection_refusal_unwind_and_floor_damage_are_terminal_after_owned_drop() {
    for mode in 1..=4 {
        let drops = Cell::new(0);
        let mut work = Work::new(1_000_000);
        let mut b = Budget::new(&mut work, LIMIT);
        b.reserve_storage(FLOOR).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| measured_scope(&mut b, mode, &drops)));
        assert_eq!(drops.get(), 1);
        assert!(b.storage() > FLOOR);
        match mode {
            1 => assert!(matches!(result, Ok(Err(E(Cause::Invalid(_)))))),
            2 => assert!(result.is_err()),
            3 | 4 => assert!(matches!(
                result,
                Ok(Err(E(Cause::Resource(Resource::Accounting))))
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn projection_swallowed_denials_drop_returned_owner_and_retain_charges() {
    struct ReturnedOwner<'a>(&'a Cell<usize>);
    impl Drop for ReturnedOwner<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    for storage_denial in [false, true] {
        let drops = Cell::new(0);
        let mut work = Work::new(100);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let account = budget.storage_account_identity_v1();
        let result: Result<ReturnedOwner<'_>, E> = projected_scope(
            0,
            &mut budget,
            |_, b| {
                b.charge_work(1)?;
                Ok(())
            },
            |_, b| {
                if storage_denial {
                    b.reserve_storage(LIMIT + 1).unwrap_err();
                } else {
                    b.charge_work(101).unwrap_err();
                }
                Ok(ReturnedOwner(&drops))
            },
        );
        assert_eq!(drops.get(), 1);
        assert!(budget.storage() > FLOOR);
        assert!(budget.work_ledger_identity_v1() == ledger);
        assert_eq!(budget.storage_account_identity_v1(), account);
        match result {
            Err(E(Cause::Resource(Resource::Storage(_)))) => assert!(storage_denial),
            Err(E(Cause::Resource(Resource::Work(_)))) => assert!(!storage_denial),
            _ => panic!("swallowed original denial must remain terminal"),
        }
    }
}

#[test]
fn projection_prior_denials_refuse_before_build_without_new_charges() {
    for storage_denial in [false, true] {
        let mut work = Work::new(100);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        if storage_denial {
            budget.reserve_storage(LIMIT + 1).unwrap_err();
        } else {
            budget.charge_work(101).unwrap_err();
        }
        let prior = (budget.work(), budget.failed_work(), budget.failed_storage());
        let builds = Cell::new(0);
        let callbacks = Cell::new(0);
        let result: Result<(), E> = projected_scope(
            0,
            &mut budget,
            |_, _| {
                builds.set(builds.get() + 1);
                Ok(())
            },
            |_, _| {
                callbacks.set(callbacks.get() + 1);
                Ok(())
            },
        );
        assert_eq!((builds.get(), callbacks.get()), (0, 0));
        assert_eq!(budget.storage(), FLOOR);
        assert_eq!(
            (budget.work(), budget.failed_work(), budget.failed_storage()),
            prior
        );
        match result {
            Err(E(Cause::Resource(Resource::Storage(_)))) => assert!(storage_denial),
            Err(E(Cause::Resource(Resource::Work(_)))) => assert!(!storage_denial),
            _ => panic!("prior original denial must refuse projection"),
        }
    }
}

#[test]
fn mapped_backing_minimum_and_subject_work_thresholds_are_exact() {
    with_mapping(&rows(), |context, _| {
        let selection = ReplaySelection::Mapping(context);
        for paid in [selection.backing_bytes() - 1, selection.backing_bytes()] {
            let mut work = Work::new(1);
            let mut b = Budget::new(&mut work, paid);
            b.reserve_storage(paid).unwrap();
            assert_eq!(
                selection.require_backing(&mut b).is_ok(),
                paid == selection.backing_bytes()
            );
            assert_eq!((b.work(), b.storage()), (1, paid));
        }
    });
    let row = ProjectedRoot {
        semantic_root: 4,
        origin: Origin::SourceRegistrationV1,
        kernel: [4; 32],
        reference: [104; 32],
    };
    let (kernel, reference) = (identity(4), identity(104));
    let cpu = CpuSubjects {
        semantic_mir_sha256: [0; 32],
        semantic_root: 4,
        logical_kernel_name: "second",
        kernel: &kernel,
        reference: &reference,
    };
    let mut work = Work::new(65);
    let mut budget = Budget::new(&mut work, FLOOR);
    budget.reserve_storage(FLOOR).unwrap();
    assert!(matches!(
        RootExpectation {
            origin: row.origin,
            functions: Some(&row)
        }
        .require_subjects(&cpu, &mut budget),
        Err(E(Cause::Resource(Resource::Work(_))))
    ));
    assert_eq!(budget.work(), 0);
    assert_eq!(budget.storage(), FLOOR);
}

#[allow(dead_code)]
fn mapped_public_call_shapes(
    packet: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    context: &NativeConditionalCpuMappingContextV1<'_>,
    inputs: crate::NativeConditionalFinalInputsV2<'_, '_, '_>,
    budget: &mut Budget<'_>,
) {
    let _ = crate::validate_native_conditional_source_packet_with_cpu_mapping_v2(
        packet, accepted, context, budget,
    );
    let _ = crate::validate_native_conditional_source_through_f_with_cpu_mapping_v2(
        packet, accepted, context, inputs, budget,
    );
}

// Included in the existing synthetic component fixture module. None of these
// tests constructs an authenticated source owner or grants continuation.
mod rich_source_preparation_tests {
    use super::super::super::bf16_nominal_source_preparation_v1::{
        measure_preparation_core_for_test_v1, rich_frame_for_test_v1, with_rich_tables_for_test_v1,
    };
    use super::*;
    use fe2o3_lower_mir_kernel::Bf16NominalCallQueryErrorV1 as Error;
    const LIMIT: usize = 16_000_000;
    const FLOOR: usize = 37;

    #[test]
    fn rich_immutable_tables_match_existing_shared_algorithms() {
        for (escape, redefine) in [(false, false), (true, false), (false, true)] {
            let function = fixture(escape, redefine);
            let types = projection_types();
            let (scalar, provenance, allocations, constants) =
                prepare_component(&function, &types, &mut PreparationResourcesV1::unmetered())
                    .unwrap();
            let dominance = SemanticEnumPayloadDominanceV1::analyze(&function, &types).unwrap();
            let option_producers =
                fe2o3_mir_model::semantic_option_producers_v1(&function, &[]).unwrap();
            let option_dominance =
                fe2o3_mir_model::SemanticOptionDominanceV1::analyze(&function, &option_producers)
                    .unwrap();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(FLOOR).unwrap();
            with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |rich, budget| {
                budget.charge_work(512)?;
                assert!(std::ptr::eq(rich.function(), &function));
                assert_eq!(rich.scalar_counts(), scalar.counts);
                assert_eq!(rich.scalar_blocks(), scalar.blocks);
                assert_eq!(rich.scalar_assignments(), scalar.assignments);
                assert_eq!(rich.address_escaped(), scalar.address_escaped);
                assert_eq!(
                    rich.stable_argument_origins(),
                    provenance.stable_argument_origins
                );
                assert_eq!(rich.allocation_origins(), provenance.allocation_origins);
                assert_eq!(
                    rich.allocation_provenance(),
                    provenance.allocation_provenance
                );
                assert_eq!(rich.allocations(), allocations);
                assert_eq!(rich.constants(), constants);
                assert_eq!(rich.enum_payload_dominance(), &dominance);
                assert_eq!(rich.option_producers(), option_producers);
                assert_eq!(rich.option_dominance(), &option_dominance);
                assert!(!rich.option_dominance().grants_authority());
                // The existing four-input borrow remains available without any
                // scalar/provenance clone or ownership escape.
                let _inputs = rich.dense_inputs();
                assert!(budget.storage() > FLOOR);
                Ok(())
            })
            .unwrap();
            assert_eq!(budget.storage(), FLOOR);
            assert_eq!(
                (budget.failed_work(), budget.failed_storage()),
                (None, None)
            );
        }
    }

    #[test]
    fn retaining_tables_keeps_existing_algorithm_charges_exact() {
        let function = fixture(true, false);
        let types = projection_types();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let existing =
            measure_preparation_core_for_test_v1(&[], &types, &function, false, &mut budget)
                .unwrap();
        let retained =
            measure_preparation_core_for_test_v1(&[], &types, &function, true, &mut budget)
                .unwrap();
        assert_eq!(retained, existing);
        assert_eq!(budget.storage(), FLOOR);
    }

    fn probe(work_limit: usize, storage_limit: usize) -> (bool, usize, usize, bool, bool) {
        let function = fixture(true, false);
        let types = projection_types();
        let mut work = Work::new(work_limit);
        let mut budget = Budget::new(&mut work, storage_limit);
        budget.reserve_storage(FLOOR).unwrap();
        let result =
            with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |rich, budget| {
                budget.charge_work(9)?;
                assert_eq!(rich.constants()[2], None);
                Ok(())
            });
        let result = (
            result.is_ok(),
            budget.work(),
            budget.peak_storage(),
            budget.failed_work().is_some(),
            budget.failed_storage().is_some(),
        );
        assert_eq!(budget.storage(), FLOOR);
        result
    }
    #[test]
    fn rich_exact_work_storage_and_one_short_are_fail_closed() {
        let measured = probe(LIMIT, LIMIT);
        assert!(measured.0);
        assert_eq!(probe(measured.1, measured.2), measured);
        let work = probe(measured.1 - 1, measured.2);
        assert!(!work.0 && work.3 && !work.4);
        let storage = probe(measured.1, measured.2 - 1);
        assert!(!storage.0 && !storage.3 && storage.4);
    }

    #[test]
    fn rich_success_error_and_panic_preserve_callback_surplus() {
        for mode in 0..3 {
            let function = fixture(false, false);
            let types = projection_types();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(FLOOR).unwrap();
            let result = with_rich_tables_for_test_v1(
                &[],
                &types,
                &function,
                &mut budget,
                |rich, budget| -> std::result::Result<(), Error> {
                    assert!(!rich.scalar_counts().is_empty());
                    budget.reserve_storage(23)?;
                    budget.charge_work(17)?;
                    match mode {
                        0 => Ok(()),
                        1 => Err(Error::Unavailable("synthetic rich callback error")),
                        _ => panic!("synthetic rich callback panic"),
                    }
                },
            );
            match mode {
                0 => assert_eq!(result, Ok(())),
                1 => assert_eq!(
                    result,
                    Err(Error::Unavailable("synthetic rich callback error"))
                ),
                _ => assert_eq!(result, Err(Error::CallbackPanicked)),
            }
            assert_eq!(budget.storage(), FLOOR + 23);
            budget.release_storage(23).unwrap();
            assert_eq!(budget.storage(), FLOOR);
        }
    }

    #[test]
    fn rich_ignored_denials_cannot_report_success() {
        for storage in [false, true] {
            let function = fixture(false, false);
            let types = projection_types();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(FLOOR).unwrap();
            let result =
                with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |_, budget| {
                    if storage {
                        let _ = budget.reserve_storage(LIMIT + 1);
                    } else {
                        let _ = budget.charge_work(LIMIT + 1);
                    }
                    Ok(())
                });
            assert_eq!(result, Err(Error::Resource(Resource::Accounting)));
            assert_eq!(budget.storage(), FLOOR);
            assert_eq!(budget.failed_storage().is_some(), storage);
            assert_eq!(budget.failed_work().is_some(), !storage);
        }
    }

    #[test]
    fn rich_floor_corruption_is_not_repaired() {
        let function = fixture(false, false);
        let types = projection_types();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let mut seen = 0;
        let result =
            with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |_, budget| {
                seen = budget.storage();
                budget.release_storage(1)?;
                Ok(())
            });
        assert_eq!(result, Err(Error::Resource(Resource::Accounting)));
        assert!(seen > FLOOR);
        assert_eq!(budget.storage(), seen - 1);
    }

    #[test]
    fn rich_replaced_original_ledger_is_not_repaired() {
        let function = fixture(false, false);
        let types = projection_types();
        let mut original = Work::new(LIMIT);
        let mut foreign = Work::new(LIMIT);
        let replacement = Budget::new(&mut foreign, LIMIT);
        let mut budget = Budget::new(&mut original, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let identity = budget.work_ledger_identity_v1();
        let result =
            with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |_, budget| {
                *budget = replacement;
                Ok(())
            });
        assert_eq!(result, Err(Error::Resource(Resource::Accounting)));
        assert!(budget.work_ledger_identity_v1() != identity);
        assert_eq!(
            (budget.storage(), budget.work(), budget.peak_storage()),
            (0, 0, 0)
        );
    }

    #[test]
    fn rich_preexisting_denial_prevents_all_preparation() {
        let function = fixture(false, false);
        let types = projection_types();
        for storage in [false, true] {
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            if storage {
                assert!(budget.reserve_storage(LIMIT + 1).is_err());
            } else {
                assert!(budget.charge_work(LIMIT + 1).is_err());
            }
            let result = with_rich_tables_for_test_v1(
                &[],
                &types,
                &function,
                &mut budget,
                |_, _| -> std::result::Result<(), Error> {
                    panic!("denied entry");
                },
            );
            assert_eq!(result, Err(Error::Resource(Resource::Accounting)));
            assert_eq!((budget.storage(), budget.work()), (0, 0));
        }
    }

    #[test]
    fn rich_large_callback_and_copy_result_are_in_frame_before_preparation() {
        let bytes = rich_frame_for_test_v1::<[u8; 8192]>(8192).unwrap();
        assert!(bytes >= 4 * 8192);
        assert_eq!(
            rich_frame_for_test_v1::<()>(usize::MAX),
            Err(Error::Resource(Resource::Arithmetic))
        );
        let function = fixture(false, false);
        let types = projection_types();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, bytes - 1);
        let large = [7u8; 8192];
        let result =
            with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, move |_, _| {
                assert_eq!(large[8191], 7);
                Ok(large)
            });
        assert!(matches!(result, Err(Error::Resource(Resource::Storage(_)))));
        assert_eq!(budget.storage(), 0);
        assert_eq!(budget.work(), 32);
        assert!(budget.failed_storage().is_some());
    }

    #[test]
    fn rich_panic_payload_drops_before_refund() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let function = fixture(false, false);
        let types = projection_types();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let dropped = Arc::new(AtomicBool::new(false));
        let result = with_rich_tables_for_test_v1(
            &[],
            &types,
            &function,
            &mut budget,
            |_, _| -> std::result::Result<(), Error> {
                std::panic::panic_any(DropFlag(dropped.clone()))
            },
        );
        assert_eq!(result, Err(Error::CallbackPanicked));
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(budget.storage(), 0);
    }

    #[test]
    fn rich_lexical_ledger_pair_matches_original_and_refuses_foreign_same_source() {
        let function = fixture(false, false);
        let types = projection_types();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |rich, budget| {
            let original = (
                budget as *const Budget<'_> as usize,
                budget.work_ledger_identity_v1(),
            );
            assert!(std::ptr::eq(rich.function(), &function));
            assert!(rich.belongs_to_original_ledger_v1(original));
            let mut other_work = Work::new(LIMIT);
            let other = Budget::new(&mut other_work, LIMIT);
            let foreign = (
                &other as *const Budget<'_> as usize,
                other.work_ledger_identity_v1(),
            );
            assert!(!rich.belongs_to_original_ledger_v1(foreign));
            assert!(!rich.belongs_to_original_ledger_v1((foreign.0, original.1)));
            assert!(!rich.belongs_to_original_ledger_v1((original.0, foreign.1)));
            assert!(rich.belongs_to_original_ledger_v1(original));
            Ok(())
        })
        .unwrap();
        assert_eq!(budget.storage(), 0);
    }

    #[test]
    fn rich_lexical_ledger_pair_refuses_replacement_before_existing_postflight() {
        let function = fixture(false, false);
        let types = projection_types();
        let mut original_work = Work::new(LIMIT);
        let mut replacement_work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut original_work, LIMIT);
        let replacement = Budget::new(&mut replacement_work, LIMIT);
        let result =
            with_rich_tables_for_test_v1(&[], &types, &function, &mut budget, |rich, budget| {
                let before = (
                    budget as *const Budget<'_> as usize,
                    budget.work_ledger_identity_v1(),
                );
                assert!(rich.belongs_to_original_ledger_v1(before));
                *budget = replacement;
                let after = (
                    budget as *const Budget<'_> as usize,
                    budget.work_ledger_identity_v1(),
                );
                assert!(before.0 == after.0 && before.1 != after.1);
                assert!(!rich.belongs_to_original_ledger_v1(after));
                Ok(())
            });
        assert_eq!(result, Err(Error::Resource(Resource::Accounting)));
        assert_eq!(
            (budget.storage(), budget.work(), budget.peak_storage()),
            (0, 0, 0)
        );
    }
}

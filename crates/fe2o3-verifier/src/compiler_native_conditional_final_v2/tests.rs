//! Component accounting/limit tests only, never genuine import/F success.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use std::{
    cell::Cell,
    error::Error as _,
    panic::{AssertUnwindSafe, catch_unwind},
};

const FLOOR: usize = 19;

#[path = "hook_tests.rs"]
mod hook;

fn budgeted<T>(run: impl FnOnce(&mut Budget<'_>) -> T) -> T {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(FLOOR).unwrap();
    run(&mut budget)
}

struct Provisional<'a>(&'a Cell<usize>);
impl Drop for Provisional<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn conditional_final_component_success_transfers_only_after_all_scopes() {
    budgeted(|b| {
        b.charge_work(7).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let (owner, storage) = account::transfer(b, |b| {
            account::temporary(b, 13, |b| {
                b.reserve_storage(17)?;
                account::temporary(b, 23, |b| {
                    b.charge_work(5)?;
                    Ok(())
                })?;
                // The nested scope releases itself, not this live owner.
                assert_eq!(b.storage(), FLOOR + 13 + 17);
                Ok((Provisional(&drops), NativeConditionalSourceStorageV2(17)))
            })
        })
        .unwrap();
        assert_eq!(drops.get(), 0);
        assert_eq!(storage.retained_storage(), 17);
        assert_eq!(b.storage(), FLOOR);
        assert_eq!(b.work(), 7 + 8 + 1 + 1 + 5);
        assert!(b.work_ledger_identity_v1() == ledger);
        drop(owner);
        assert_eq!(drops.get(), 1);
    });
}

#[test]
fn conditional_final_component_exact_limits_and_denial_history() {
    let run = |b: &mut Budget<'_>| {
        account::transfer(b, |b| {
            account::temporary(b, 13, |b| {
                b.reserve_storage(17)?;
                b.charge_work(5)?;
                Ok(((), NativeConditionalSourceStorageV2(17)))
            })
        })
    };
    for (work_limit, storage_limit, outcome) in [
        (14, FLOOR + 30, 0),
        (13, FLOOR + 30, 1),
        (14, FLOOR + 29, 2),
    ] {
        let mut work = Work::new(work_limit);
        let mut b = Budget::new(&mut work, storage_limit);
        b.reserve_storage(FLOOR).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let result = run(&mut b);
        match outcome {
            0 => {
                result.unwrap();
                assert_eq!(b.storage(), FLOOR);
            }
            1 => {
                assert!(matches!(
                    result,
                    Err(Error(Cause::Resource(Resource::Work(_))))
                ));
                assert!(b.failed_work().is_some());
                assert_eq!(b.storage(), FLOOR + 30);
            }
            _ => {
                assert!(matches!(
                    result,
                    Err(Error(Cause::Resource(Resource::Storage(_))))
                ));
                assert!(b.failed_storage().is_some());
                assert_eq!(b.storage(), FLOOR + 13);
            }
        }
        assert!(b.work_ledger_identity_v1() == ledger);
    }
    budgeted(|b| {
        b.charge_work(7).unwrap();
        assert!(b.charge_work(usize::MAX).is_err());
        assert!(b.reserve_storage(usize::MAX).is_err());
        let first = (b.failed_work(), b.failed_storage());
        assert!(matches!(
            run(b),
            Err(Error(Cause::Resource(Resource::Work(_))))
        ));
        assert_eq!(b.work(), 7);
        assert_eq!((b.failed_work(), b.failed_storage()), first);
        assert_eq!(b.storage(), FLOOR);
    });
}

#[test]
fn conditional_final_component_five_live_accounts_have_independent_peak_threshold() {
    use fe2o3_amdgcn_model::ReplayedNativeV12TextDescriptorRelationV5;
    use fe2o3_kernel_descriptor::DESCRIPTOR_QUERY_STORAGE_V5;

    // Independent of HEADER: these are the five simultaneously live guards in
    // the explicit nesting below, not five sequential uses of one temporary.
    let required_header = size_of::<Inputs<'static, 'static, 'static>>()
        + size_of::<Result<Output, Error>>()
        + size_of::<[account::Account; 5]>();
    assert_eq!(HEADER, required_header);
    // Inert receipt extents model history/coordinate storage; no proof or
    // history owner is constructed. The two real leaf header sizes are disjoint.
    const HISTORY: usize = 17;
    const COORDINATES: usize = 23;
    for leaf in [
        DESCRIPTOR_QUERY_STORAGE_V5,
        size_of::<ReplayedNativeV12TextDescriptorRelationV5<'_, '_, '_, '_, '_>>(),
    ] {
        assert!(leaf >= size_of::<account::Account>());
        let peak = FLOOR + required_header + HISTORY + COORDINATES + leaf;
        for short in [0, 1, size_of::<account::Account>()] {
            let mut work = Work::new(usize::MAX);
            let mut b = Budget::new(&mut work, peak - short);
            b.reserve_storage(FLOOR).unwrap();
            b.charge_work(7).unwrap();
            let ledger = b.work_ledger_identity_v1();
            let reached_leaf = Cell::new(false);
            let result = account::transfer(&mut b, |b| {
                account::temporary(b, HEADER, |b| {
                    account::temporary(b, HISTORY, |b| {
                        account::temporary(b, COORDINATES, |b| {
                            account::temporary(b, leaf, |b| {
                                reached_leaf.set(true);
                                assert_eq!(b.storage(), peak);
                                Ok(())
                            })
                        })
                    })?;
                    Ok(((), NativeConditionalSourceStorageV2(0)))
                })
            });
            if short == 0 {
                result.unwrap();
                assert!(reached_leaf.get());
                assert_eq!(b.peak_storage(), peak);
                assert_eq!(b.storage(), FLOOR);
                assert!(b.failed_storage().is_none());
            } else {
                assert!(matches!(
                    result,
                    Err(Error(Cause::Resource(Resource::Storage(_))))
                ));
                assert!(!reached_leaf.get());
                assert!(b.failed_storage().is_some());
                // Earlier live reservations stay terminal after the last reserve
                // fails; neither enclosing success nor wholesale refund occurs.
                assert_eq!(b.storage(), FLOOR + required_header + HISTORY + COORDINATES);
            }
            assert_eq!(b.work(), 7 + 8 + 4);
            assert!(b.work_ledger_identity_v1() == ledger);
        }
    }
}

#[test]
fn conditional_final_component_rejection_and_unwind_never_refund() {
    for unwind in [false, true] {
        budgeted(|b| {
            let drops = Cell::new(0);
            let result = catch_unwind(AssertUnwindSafe(|| {
                account::transfer::<()>(b, |b| {
                    account::temporary(b, 13, |b| {
                        b.reserve_storage(17)?;
                        let _owner = Provisional(&drops);
                        // Model a later-root/callback failure, not a proof result.
                        if unwind {
                            panic!("component-only late failure");
                        }
                        Err(Error::mismatch("component-only late root"))
                    })
                })
            }));
            assert_eq!(drops.get(), 1);
            assert_eq!(b.storage(), FLOOR + 30);
            assert_eq!(b.work(), 9);
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
        });
    }
}

#[test]
fn conditional_final_component_all_error_categories_are_terminal_leaves() {
    for error in [
        Error::mismatch("late row"),
        Error::from(SourceError::invalid("old source-only refundable rejection")),
        Error::from(Resource::Accounting),
        Error::from(Resource::Arithmetic),
    ] {
        assert!(error.source().is_none());
        budgeted(|b| {
            let result = account::transfer::<()>(b, |b| {
                b.reserve_storage(17)?;
                Err(error)
            });
            assert!(result.is_err());
            assert_eq!(b.storage(), FLOOR + 17);
        });
    }
}

#[test]
fn conditional_final_component_underpaid_owner_is_dropped_before_return() {
    budgeted(|b| {
        let drops = Cell::new(0);
        let result = account::transfer(b, |b| {
            b.reserve_storage(16)?;
            Ok((Provisional(&drops), NativeConditionalSourceStorageV2(17)))
        });
        assert!(matches!(
            result,
            Err(Error(Cause::Resource(Resource::Accounting)))
        ));
        assert_eq!(drops.get(), 1);
        assert_eq!(b.storage(), FLOOR + 16);
    });
}

#[test]
fn conditional_final_component_floor_damage_overrides_success_and_error() {
    for success in [false, true] {
        budgeted(|b| {
            let drops = Cell::new(0);
            let result = account::transfer(b, |b| {
                account::temporary(b, 13, |b| {
                    // Floor+scope must remain, even if a callback reports failure.
                    b.release_storage(1)?;
                    if success {
                        Ok((Provisional(&drops), NativeConditionalSourceStorageV2(0)))
                    } else {
                        Err(Error::mismatch("earlier callback rejection"))
                    }
                })
            });
            assert!(matches!(
                result,
                Err(Error(Cause::Resource(Resource::Accounting)))
            ));
            assert_eq!(drops.get(), usize::from(success));
            assert_eq!(b.storage(), FLOOR + 12);
        });
    }
}

#[test]
fn conditional_final_component_foreign_ledger_cannot_transfer_or_refund() {
    budgeted(|b| {
        let original = b.work_ledger_identity_v1();
        let drops = Cell::new(0);
        let result = account::transfer(b, |b| {
            account::temporary(b, 13, |b| {
                // Test meter outlives the callback's arbitrary budget lifetime.
                *b = Budget::new(Box::leak(Box::new(Work::new(usize::MAX))), usize::MAX);
                b.reserve_storage(FLOOR + 13)?;
                Ok((Provisional(&drops), NativeConditionalSourceStorageV2(0)))
            })
        });
        assert!(matches!(
            result,
            Err(Error(Cause::Resource(Resource::Accounting)))
        ));
        assert_eq!(drops.get(), 1);
        assert!(b.work_ledger_identity_v1() != original);
        assert_eq!(b.storage(), FLOOR + 13);
    });
}

#[test]
fn conditional_final_component_complete_external_limits_not_packet_selected() {
    let expected = Limits {
        refinement: Default::default(),
        forwarding: Default::default(),
    };
    budgeted(|b| require_limits(expected, expected, b).unwrap());
    let mutations: &[fn(&mut Limits)] = &[
        |v| v.refinement.functions += 1,
        |v| v.refinement.blocks += 1,
        |v| v.refinement.edges += 1,
        |v| v.refinement.definitions += 1,
        |v| v.refinement.operations += 1,
        |v| v.refinement.loops += 1,
        |v| v.refinement.rows += 1,
        |v| v.forwarding.memory.functions += 1,
        |v| v.forwarding.memory.blocks += 1,
        |v| v.forwarding.memory.operations += 1,
        |v| v.forwarding.memory.effects += 1,
        |v| v.forwarding.memory.edges += 1,
        |v| v.forwarding.control_flow.blocks += 1,
        |v| v.forwarding.control_flow.edges += 1,
        |v| v.forwarding.control_flow.edge_arguments += 1,
        |v| v.forwarding.control_flow.phi_inputs += 1,
        |v| v.forwarding.control_flow.analysis_work += 1,
    ];
    for mutate in mutations {
        let mut actual = expected;
        mutate(&mut actual);
        budgeted(|b| {
            let ledger = b.work_ledger_identity_v1();
            assert!(matches!(
                require_limits(actual, expected, b),
                Err(Error(Cause::Mismatch(_)))
            ));
            assert_eq!(b.work(), size_of::<Limits>() + 1);
            assert_eq!(b.storage(), FLOOR);
            assert!(b.work_ledger_identity_v1() == ledger);
        });
    }
}

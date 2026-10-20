//! Ordered CPU-origin descriptions projected from the retained binding owner.
use super::{AuthenticatedProductionBindings, Budget, Error, Resource, Root, policy_capture};
use crate::reference_effect_v1::ReferenceBindingOriginV1 as BindingOrigin;
use fe2o3_compiler_lineage::{
    RustcEnrollmentInventoryHeaderV1 as InventoryHeader,
    RustcEnrollmentInventoryRootV1 as InventoryRoot,
};
use fe2o3_verifier::{
    NativeConditionalCpuExpectationV1 as Expectation,
    NativeConditionalCpuOriginExpectationV1 as Origin,
};
use sha2::{Digest, Sha256};
use std::mem::size_of;

/// The caller retains the original source and invocation beside these inert rows.
/// No constructor from CPU packet bytes or public provenance token is provided.
pub(super) struct RetainedCpuOriginRosterV1 {
    rows: Vec<Expectation>,
}

impl RetainedCpuOriginRosterV1 {
    pub(super) fn rows(&self) -> &[Expectation] {
        &self.rows
    }

    pub(super) fn retained_storage(&self) -> Result<usize, Resource> {
        self.rows
            .capacity()
            .checked_mul(size_of::<Expectation>())
            .and_then(|bytes| bytes.checked_add(size_of::<Self>()))
            .ok_or(Resource::Arithmetic)
    }
}

/// Called inside the original producer visit, after the packet assembler checks
/// the complete root/descriptor/source order. Terminal failures earn no refund.
pub(super) fn capture(
    bindings: &AuthenticatedProductionBindings,
    roots: &[Root],
    budget: &mut Budget<'_>,
) -> Result<RetainedCpuOriginRosterV1, Error> {
    if let Some(original) = bindings
        .rustc_identity_inventory
        .original_root_associations()
    {
        return project_enrolled(
            original.header(),
            original.roots(),
            roots.iter().map(|root| {
                (
                    root.semantic_root().index(),
                    root.logical_name(),
                    *root.semantic_root_identity().as_bytes(),
                    *root.kernel_binding(),
                )
            }),
            budget,
        );
    }
    budget.charge_work(bindings.reference_effect_bindings.as_slice().len())?;
    if bindings
        .reference_effect_bindings
        .as_slice()
        .iter()
        .any(|binding| matches!(binding.origin, BindingOrigin::ReferenceEnrollment(_)))
    {
        return Err(Error::Mismatch(
            "enrollment requires original root inventory",
        ));
    }
    project(
        roots
            .iter()
            .map(|root| (root.semantic_root().index(), root.logical_name())),
        bindings
            .reference_effect_bindings
            .as_slice()
            .iter()
            .map(|binding| (binding.logical_kernel_name.as_str(), &binding.origin)),
        budget,
    )
}

// Canonical IDs select original rows; packet order is independently preserved.
// The full reference Instance identity stays in the retained private inventory
// and its hashed wrapper. This origin projection does not replace that evidence.
fn project_enrolled<'a>(
    header: InventoryHeader,
    original: &[InventoryRoot],
    roots: impl ExactSizeIterator<Item = (u32, &'a str, [u8; 32], [u8; 32])>,
    budget: &mut Budget<'_>,
) -> Result<RetainedCpuOriginRosterV1, Error> {
    budget.charge_work(3)?;
    if original.is_empty()
        || roots.len() != original.len()
        || original.len() != header.kernel_count as usize
        || original.len() > fe2o3_compiler_lineage::MAX_NATIVE_CONDITIONAL_POLICY_ROSTER_ROOTS_V1
    {
        return Err(Error::Mismatch(
            "complete original root inventory projection",
        ));
    }
    budget.charge_work(original.len())?;
    if original
        .windows(2)
        .any(|pair| pair[0].semantic_root >= pair[1].semantic_root)
    {
        return Err(Error::Mismatch("canonical original root inventory order"));
    }
    let mut seen = policy_capture::vector(original.len(), budget)?;
    seen.resize(original.len(), false);
    let mut rows = policy_capture::vector(original.len(), budget)?;
    let comparisons = original.len().ilog2() as usize + 2;
    for (semantic_root, name, kernel_instance, kernel_binding) in roots {
        budget.charge_work(
            comparisons
                .checked_add(name.len())
                .and_then(|n| n.checked_add(size_of::<InventoryRoot>()))
                .ok_or(Resource::Arithmetic)?,
        )?;
        let index = original
            .binary_search_by_key(&semantic_root, |row| row.semantic_root)
            .map_err(|_| Error::Mismatch("packet root missing from original inventory"))?;
        let row = &original[index];
        if seen[index]
            || row.logical_name_len as usize != name.len()
            || row.logical_name_sha256 != <[u8; 32]>::from(Sha256::digest(name.as_bytes()))
            || row.kernel_instance != kernel_instance
            || row.kernel_binding != kernel_binding
        {
            return Err(Error::Mismatch(
                "packet root differs from original inventory",
            ));
        }
        seen[index] = true;
        let origin = match row.origin_tag {
            0 if row.descriptor_ordinal == 0 => Origin::SourceRegistrationV1,
            1 if row.descriptor_ordinal < header.enrollment_binding_count => {
                Origin::ReferenceEnrollmentV1(
                    fe2o3_verifier::portable_reference_v1::codec::ReferenceEnrollmentOriginV1 {
                        rustc_invocation_sha256: header.invocation_identity,
                        native_policy_sha256: header.native_policy_identity,
                        policy_generation: header.native_policy_generation,
                        mapping_ordinal: row.descriptor_ordinal,
                    },
                )
            }
            _ => {
                return Err(Error::Mismatch(
                    "original inventory origin or ordinal changed",
                ));
            }
        };
        budget.charge_work(size_of::<Expectation>())?;
        rows.push(Expectation {
            semantic_root,
            origin,
        });
    }
    let seen_storage = seen
        .capacity()
        .checked_mul(size_of::<bool>())
        .and_then(|n| n.checked_add(size_of::<Vec<bool>>()))
        .ok_or(Resource::Arithmetic)?;
    drop(seen);
    budget.release_storage(seen_storage)?;
    Ok(RetainedCpuOriginRosterV1 { rows })
}

// One projection body for the authenticated adapter and inert algorithm tests.
// Borrowed iterators add no root vector/index or independent authority surface.
fn project<'a>(
    roots: impl ExactSizeIterator<Item = (u32, &'a str)>,
    references: impl ExactSizeIterator<Item = (&'a str, &'a BindingOrigin)> + Clone,
    budget: &mut Budget<'_>,
) -> Result<RetainedCpuOriginRosterV1, Error> {
    budget.charge_work(3)?;
    if roots.len() == 0
        || roots.len() > fe2o3_compiler_lineage::MAX_NATIVE_CONDITIONAL_POLICY_ROSTER_ROOTS_V1
        || roots.len() != references.len()
    {
        return Err(Error::Mismatch(
            "complete original CPU origin binding roster",
        ));
    }
    let mut rows = policy_capture::vector(roots.len(), budget)?;
    for (semantic_root, name) in roots {
        let origin = select(name, references.clone(), budget)?;
        budget.charge_work(size_of::<Expectation>())?;
        rows.push(Expectation {
            semantic_root,
            origin,
        });
    }
    Ok(RetainedCpuOriginRosterV1 { rows })
}

// Names select among already retained bindings, never among decoded CPU leaves.
// The bounded scan charges every comparison and rejects ambiguous source rows.
fn select<'a>(
    name: &str,
    references: impl IntoIterator<Item = (&'a str, &'a BindingOrigin)>,
    budget: &mut Budget<'_>,
) -> Result<Origin, Error> {
    let mut selected = None;
    for (candidate, origin) in references {
        budget.charge_work(
            name.len()
                .checked_add(candidate.len())
                .and_then(|bytes| bytes.checked_add(1))
                .ok_or(Resource::Arithmetic)?,
        )?;
        if name != candidate {
            continue;
        }
        if selected.is_some() {
            return Err(Error::Mismatch("ambiguous original CPU origin binding"));
        }
        budget.charge_work(size_of::<Origin>() + 1)?;
        selected = Some(match origin {
            BindingOrigin::SourceRegistration(_) => Origin::SourceRegistrationV1,
            BindingOrigin::ReferenceEnrollment(origin) => Origin::ReferenceEnrollmentV1(*origin),
        });
    }
    selected.ok_or(Error::Mismatch("missing original CPU origin binding"))
}

#[cfg(test)]
#[path = "production_pipeline_conditional_cpu_inventory_v362_tests.rs"]
mod inventory_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
    use fe2o3_verifier::portable_reference_v1::codec::ReferenceEnrollmentOriginV1;

    fn enrolled() -> BindingOrigin {
        BindingOrigin::ReferenceEnrollment(ReferenceEnrollmentOriginV1 {
            rustc_invocation_sha256: [1; 32],
            native_policy_sha256: [2; 32],
            policy_generation: 17,
            mapping_ordinal: 9,
        })
    }

    // These are inert projection checks, not fabricated issuer/admission tests.
    #[test]
    fn registration_uses_only_the_registration_codec() {
        let origin = BindingOrigin::SourceRegistration("crate::reference".into());
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, 31);
        budget.reserve_storage(31).unwrap();
        assert_eq!(
            select("kernel", [("kernel", &origin)], &mut budget).unwrap(),
            Origin::SourceRegistrationV1
        );
        assert_eq!(budget.storage(), 31);
    }

    #[test]
    fn enrollment_preserves_all_original_fields() {
        let origin = enrolled();
        let BindingOrigin::ReferenceEnrollment(expected) = origin else {
            unreachable!()
        };
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, 0);
        assert_eq!(
            select("kernel", [("kernel", &origin)], &mut budget).unwrap(),
            Origin::ReferenceEnrollmentV1(expected)
        );
    }

    #[test]
    fn mixed_bindings_select_by_exact_original_name() {
        let registration = BindingOrigin::SourceRegistration("crate::reference".into());
        let enrollment = enrolled();
        let rows = [("a", &registration), ("b", &enrollment)];
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, 0);
        assert!(matches!(
            select("b", rows, &mut budget).unwrap(),
            Origin::ReferenceEnrollmentV1(_)
        ));
        assert_eq!(
            select("a", rows, &mut budget).unwrap(),
            Origin::SourceRegistrationV1
        );
    }

    #[test]
    fn missing_binding_has_no_registration_fallback() {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, 0);
        assert!(matches!(
            select("kernel", [], &mut budget),
            Err(Error::Mismatch("missing original CPU origin binding"))
        ));
    }

    #[test]
    fn equal_duplicate_bindings_are_still_ambiguous() {
        let origin = enrolled();
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, 0);
        assert!(matches!(
            select(
                "kernel",
                [("kernel", &origin), ("kernel", &origin)],
                &mut budget
            ),
            Err(Error::Mismatch("ambiguous original CPU origin binding"))
        ));
    }

    #[test]
    fn name_prefixes_cannot_select_a_binding() {
        let origin = enrolled();
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, 0);
        assert!(matches!(
            select("kernel", [("kernel_extra", &origin)], &mut budget),
            Err(Error::Mismatch("missing original CPU origin binding"))
        ));
        assert_eq!(budget.work(), "kernel".len() + "kernel_extra".len() + 1);
    }

    #[test]
    fn exact_projection_work_preserves_the_original_account_and_floor() {
        let origin = enrolled();
        let cost = "kernel".len() * 2 + 1 + size_of::<Origin>() + 1;
        let mut work = Work::new(7 + cost);
        let mut budget = Budget::new(&mut work, 31);
        budget.charge_work(7).unwrap();
        budget.reserve_storage(31).unwrap();
        let account = budget.work_ledger_identity_v1();
        select("kernel", [("kernel", &origin)], &mut budget).unwrap();
        assert_eq!(budget.work(), 7 + cost);
        assert_eq!(budget.storage(), 31);
        assert!(budget.work_ledger_identity_v1() == account);
    }

    #[test]
    fn repeated_projection_keeps_prefix_work_and_first_refusal() {
        let origin = enrolled();
        let comparison = "kernel".len() * 2 + 1;
        let cost = comparison + size_of::<Origin>() + 1;
        let mut work = Work::new(7 + cost * 2 - 1);
        let mut budget = Budget::new(&mut work, 31);
        budget.charge_work(7).unwrap();
        budget.reserve_storage(31).unwrap();
        let account = budget.work_ledger_identity_v1();
        select("kernel", [("kernel", &origin)], &mut budget).unwrap();
        assert!(matches!(
            select("kernel", [("kernel", &origin)], &mut budget),
            Err(Error::Resource(_))
        ));
        assert_eq!(budget.work(), 7 + cost + comparison);
        let first_refusal = budget.failed_work();
        assert!(first_refusal.is_some());
        assert!(select("kernel", [("kernel", &origin)], &mut budget).is_err());
        assert_eq!(budget.failed_work(), first_refusal);
        assert_eq!(budget.storage(), 31);
        assert!(budget.work_ledger_identity_v1() == account);
    }

    // These exercise capture's exact projection body, not authenticated binding
    // construction, genuine issuer admission or the same-visit bridge caller.
    const FLOOR: usize = 31;
    const PREFIX: usize = 7;

    fn two_root_quote(origin: &BindingOrigin) -> usize {
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        // Measure the same-shaped projection's actual allocation, not an
        // assumption that Vec capacity always equals its requested length.
        project(
            [(2, "a"), (3, "b")].into_iter(),
            [("a", origin), ("b", origin)].into_iter(),
            &mut budget,
        )
        .unwrap()
        .retained_storage()
        .unwrap()
    }

    #[test]
    fn capture_projection_rejects_cardinality_before_iteration_or_allocation() {
        let oversized = fe2o3_compiler_lineage::MAX_NATIVE_CONDITIONAL_POLICY_ROSTER_ROOTS_V1 + 1;
        for (root_count, binding_count) in [
            (0, 0),
            (0, 1),
            (1, 0),
            (2, 1),
            (1, 2),
            (oversized, oversized),
        ] {
            let mut work = Work::new(usize::MAX);
            let mut budget = Budget::new(&mut work, FLOOR);
            budget.charge_work(PREFIX).unwrap();
            budget.reserve_storage(FLOOR).unwrap();
            let account = budget.work_ledger_identity_v1();
            let roots = (0..root_count)
                .map(|_| -> (u32, &'static str) { panic!("invalid roots iterated") });
            let bindings = (0..binding_count).map(|_| -> (&'static str, &'static BindingOrigin) {
                panic!("invalid bindings iterated")
            });
            assert!(matches!(
                project(roots, bindings, &mut budget),
                Err(Error::Mismatch(
                    "complete original CPU origin binding roster"
                ))
            ));
            assert_eq!(
                (budget.work(), budget.storage(), budget.peak_storage()),
                (PREFIX + 3, FLOOR, FLOOR)
            );
            assert_eq!(
                (budget.failed_work(), budget.failed_storage()),
                (None, None)
            );
            assert!(budget.work_ledger_identity_v1() == account);
        }
    }

    #[test]
    fn capture_projection_preserves_mixed_origins_in_original_root_order() {
        let registration = BindingOrigin::SourceRegistration("crate::reference".into());
        let first = ReferenceEnrollmentOriginV1 {
            rustc_invocation_sha256: [11; 32],
            native_policy_sha256: [12; 32],
            policy_generation: 13,
            mapping_ordinal: 14,
        };
        let second = ReferenceEnrollmentOriginV1 {
            rustc_invocation_sha256: [21; 32],
            native_policy_sha256: [22; 32],
            policy_generation: 23,
            mapping_ordinal: 24,
        };
        let a_binding = BindingOrigin::ReferenceEnrollment(first);
        let b_binding = BindingOrigin::ReferenceEnrollment(second);
        let bindings = [("b", &b_binding), ("c", &registration), ("a", &a_binding)];
        let a = Expectation {
            semantic_root: 41,
            origin: Origin::ReferenceEnrollmentV1(first),
        };
        let b = Expectation {
            semantic_root: 7,
            origin: Origin::ReferenceEnrollmentV1(second),
        };
        let c = Expectation {
            semantic_root: 90,
            origin: Origin::SourceRegistrationV1,
        };
        for (roots, expected) in [
            ([(90, "c"), (41, "a"), (7, "b")], [c, a, b]),
            ([(7, "b"), (90, "c"), (41, "a")], [b, c, a]),
        ] {
            let mut work = Work::new(usize::MAX);
            let mut budget = Budget::new(&mut work, usize::MAX);
            budget.charge_work(PREFIX).unwrap();
            budget.reserve_storage(FLOOR).unwrap();
            let account = budget.work_ledger_identity_v1();
            let roster = project(roots.into_iter(), bindings.into_iter(), &mut budget).unwrap();
            assert_eq!(roster.rows(), &expected);
            let quote = roster.retained_storage().unwrap();
            assert_eq!(
                quote,
                size_of::<RetainedCpuOriginRosterV1>()
                    + roster.rows.capacity() * size_of::<Expectation>()
            );
            assert_eq!(budget.storage(), FLOOR + quote);
            assert_eq!(budget.peak_storage(), FLOOR + quote);
            assert_eq!(
                budget.work(),
                PREFIX + 4 + 9 * 3 + 3 * (size_of::<Origin>() + 1 + size_of::<Expectation>())
            );
            assert!(budget.work_ledger_identity_v1() == account);
            drop(roster);
            assert_eq!(budget.storage(), FLOOR + quote);
            assert_eq!(
                (budget.failed_work(), budget.failed_storage()),
                (None, None)
            );
        }
    }

    #[test]
    fn capture_projection_missing_and_ambiguous_names_keep_terminal_charges() {
        let origin = enrolled();
        let quote = two_root_quote(&origin);
        let roots = [(2, "a"), (3, "b")];
        for (bindings, reason, accepted) in [
            (
                [("a", &origin), ("c", &origin)],
                "missing original CPU origin binding",
                4 + 12 + size_of::<Origin>() + 1 + size_of::<Expectation>(),
            ),
            (
                [("a", &origin), ("a", &origin)],
                "ambiguous original CPU origin binding",
                4 + 6 + size_of::<Origin>() + 1,
            ),
        ] {
            let mut work = Work::new(usize::MAX);
            let mut budget = Budget::new(&mut work, usize::MAX);
            budget.charge_work(PREFIX).unwrap();
            budget.reserve_storage(FLOOR).unwrap();
            let account = budget.work_ledger_identity_v1();
            assert!(
                matches!(project(roots.into_iter(), bindings.into_iter(), &mut budget), Err(Error::Mismatch(actual)) if actual == reason)
            );
            let terminal = budget.storage();
            assert_eq!(terminal, FLOOR + quote);
            assert_eq!(budget.peak_storage(), terminal);
            assert_eq!(budget.work(), PREFIX + accepted);
            assert_eq!(
                (budget.failed_work(), budget.failed_storage()),
                (None, None)
            );
            assert!(budget.work_ledger_identity_v1() == account);
        }
    }

    #[test]
    fn capture_projection_exact_and_one_short_limits_preserve_floor_and_denials() {
        let origin = enrolled();
        let cost = 4 + 3 + size_of::<Origin>() + 1 + size_of::<Expectation>();
        let requested = size_of::<Vec<Expectation>>() + size_of::<Expectation>();
        let run = |work_limit, storage_limit, prior_denials| {
            let mut work = Work::new(work_limit);
            let mut budget = Budget::new(&mut work, storage_limit);
            budget.charge_work(PREFIX).unwrap();
            budget.reserve_storage(FLOOR).unwrap();
            let account = budget.work_ledger_identity_v1();
            if prior_denials {
                assert!(budget.charge_work(usize::MAX).is_err());
                assert!(budget.reserve_storage(usize::MAX).is_err());
            }
            let result = project(
                [(5, "a")].into_iter(),
                [("a", &origin)].into_iter(),
                &mut budget,
            );
            assert!(budget.work_ledger_identity_v1() == account);
            assert_eq!(budget.storage(), budget.peak_storage());
            (
                result,
                budget.work(),
                budget.storage(),
                budget.failed_work(),
                budget.failed_storage(),
            )
        };
        let (result, used_work, used_storage, _, _) = run(usize::MAX, usize::MAX, false);
        let roster = result.unwrap();
        let quote = roster.retained_storage().unwrap();
        assert_eq!(used_work, PREFIX + cost);
        assert_eq!(used_storage, FLOOR + quote);
        let (result, work, storage, failed_work, failed_storage) =
            run(PREFIX + cost, FLOOR + quote, false);
        assert_eq!(result.unwrap().rows(), roster.rows());
        assert_eq!((work, storage), (PREFIX + cost, FLOOR + quote));
        assert_eq!((failed_work, failed_storage), (None, None));
        for prior in [false, true] {
            let history = prior.then_some(usize::MAX);
            let (result, work, storage, failed_work, failed_storage) =
                run(PREFIX + cost - 1, FLOOR + quote, prior);
            assert!(
                matches!(result, Err(Error::Resource(Resource::Work(error))) if error.actual() == PREFIX + cost)
            );
            assert_eq!(work, PREFIX + cost - size_of::<Expectation>());
            assert_eq!(storage, FLOOR + quote);
            assert_eq!(failed_work, history.or(Some(PREFIX + cost)));
            assert_eq!(failed_storage, history);
            let (result, work, storage, failed_work, failed_storage) =
                run(PREFIX + cost, FLOOR + quote - 1, prior);
            assert!(
                matches!(result, Err(Error::Resource(Resource::Storage(error))) if error.actual() == FLOOR + quote)
            );
            // If the allocator reports spare capacity, the requested reservation
            // remains charged when the later excess-capacity reservation fails.
            let (accepted_work, accepted_storage) = if quote == requested {
                (3, FLOOR)
            } else {
                (4, FLOOR + requested)
            };
            assert_eq!((work, storage), (PREFIX + accepted_work, accepted_storage));
            assert_eq!(failed_work, history);
            assert_eq!(failed_storage, history.or(Some(FLOOR + quote)));
        }
    }

    #[test]
    fn retained_roster_quote_counts_actual_spare_capacity() {
        let mut rows = Vec::with_capacity(5);
        rows.push(Expectation {
            semantic_root: 17,
            origin: Origin::SourceRegistrationV1,
        });
        let roster = RetainedCpuOriginRosterV1 { rows };
        assert!(roster.rows.capacity() > roster.rows().len());
        assert_eq!(
            roster.retained_storage().unwrap(),
            size_of::<RetainedCpuOriginRosterV1>()
                + roster.rows.capacity() * size_of::<Expectation>()
        );
        assert!(
            roster.retained_storage().unwrap()
                > size_of::<RetainedCpuOriginRosterV1>() + std::mem::size_of_val(roster.rows())
        );
    }

    #[test]
    fn capture_projection_unwind_keeps_allocated_storage_and_prefix_work() {
        let origin = enrolled();
        let quote = two_root_quote(&origin);
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.charge_work(PREFIX).unwrap();
        budget.reserve_storage(FLOOR).unwrap();
        let account = budget.work_ledger_identity_v1();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let roots = [(2, "a"), (3, "b")].into_iter().map(|row| {
                if row.0 == 3 {
                    std::panic::panic_any("stop after first projected row");
                }
                row
            });
            project(
                roots,
                [("a", &origin), ("b", &origin)].into_iter(),
                &mut budget,
            )
        }));
        let Err(payload) = result else {
            panic!("projection did not unwind after its first row")
        };
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"stop after first projected row")
        );
        assert_eq!(budget.storage(), FLOOR + quote);
        assert_eq!(budget.peak_storage(), budget.storage());
        assert_eq!(
            budget.work(),
            PREFIX + 4 + 6 + size_of::<Origin>() + 1 + size_of::<Expectation>()
        );
        assert_eq!(
            (budget.failed_work(), budget.failed_storage()),
            (None, None)
        );
        assert!(budget.work_ledger_identity_v1() == account);
    }
}

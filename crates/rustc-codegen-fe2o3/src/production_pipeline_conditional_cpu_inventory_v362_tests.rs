//! Inert projection tests, not original compiler-owner construction or admission.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

fn header() -> InventoryHeader {
    InventoryHeader {
        kernel_count: 3,
        enrollment_binding_count: 2,
        invocation_identity: [11; 32],
        native_policy_identity: [12; 32],
        native_policy_generation: 19,
    }
}

fn original() -> [InventoryRoot; 3] {
    [
        (2, "first", 1, 1),
        (5, "registered", 0, 0),
        (17, "last", 1, 0),
    ]
    .map(|(id, name, tag, ordinal)| InventoryRoot {
        semantic_root: id,
        origin_tag: tag,
        descriptor_ordinal: ordinal,
        logical_name_len: name.len() as u32,
        logical_name_sha256: Sha256::digest(name.as_bytes()).into(),
        kernel_binding: [id as u8; 32],
        kernel_instance: [id as u8 + 1; 32],
        reference_instance: [id as u8 + 2; 32],
    })
}

fn packet(id: u32, name: &str) -> (u32, &str, [u8; 32], [u8; 32]) {
    (id, name, [id as u8 + 1; 32], [id as u8; 32])
}

#[test]
fn noncontiguous_canonical_roots_project_complete_mixed_packet_permutation() {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.charge_work(7).unwrap();
    budget.reserve_storage(31).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    let observed = project_enrolled(
        header(),
        &original(),
        [
            packet(17, "last"),
            packet(2, "first"),
            packet(5, "registered"),
        ]
        .into_iter(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(
        observed
            .rows()
            .iter()
            .map(|row| row.semantic_root)
            .collect::<Vec<_>>(),
        [17, 2, 5]
    );
    for (index, ordinal) in [(0, 0), (1, 1)] {
        let Origin::ReferenceEnrollmentV1(origin) = observed.rows()[index].origin else {
            panic!("enrolled root changed");
        };
        assert_eq!(origin.mapping_ordinal, ordinal);
        assert_eq!(origin.rustc_invocation_sha256, header().invocation_identity);
        assert_eq!(origin.native_policy_sha256, header().native_policy_identity);
        assert_eq!(origin.policy_generation, 19);
    }
    assert_eq!(observed.rows()[2].origin, Origin::SourceRegistrationV1);
    assert!(budget.work_ledger_identity_v1() == ledger);
    assert_eq!(budget.storage(), 31 + observed.retained_storage().unwrap());
    assert!(budget.peak_storage() > budget.storage());
}

#[test]
fn original_identity_binding_name_and_packet_bijection_are_all_required() {
    for mutation in 0..8 {
        let mut roots = [
            packet(2, "first"),
            packet(5, "registered"),
            packet(17, "last"),
        ];
        match mutation {
            0 => roots[0].0 = 3,
            1 => roots[0].1 = "First",
            2 => roots[0].1 = "first-prefix",
            3 => roots[0].2[31] ^= 1,
            4 => roots[0].3[31] ^= 1,
            5 => roots[1] = roots[0],
            6 => roots[2] = roots[0],
            7 => roots.swap(0, 1),
            _ => unreachable!(),
        }
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        let result = project_enrolled(header(), &original(), roots.into_iter(), &mut budget);
        assert_eq!(result.is_ok(), mutation == 7, "mutation {mutation}");
    }
}

#[test]
fn incomplete_noncanonical_and_invalid_origin_rosters_refuse() {
    for mutation in 0..7 {
        let mut h = header();
        let mut original = original();
        match mutation {
            0 => h.kernel_count -= 1,
            1 => original.swap(0, 1),
            2 => original[1].semantic_root = original[0].semantic_root,
            3 => original[0].origin_tag = 2,
            4 => original[0].descriptor_ordinal = 2,
            5 => original[1].descriptor_ordinal = 1,
            6 => original[0].logical_name_len += 1,
            _ => unreachable!(),
        }
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        assert!(
            project_enrolled(
                h,
                &original,
                [
                    packet(2, "first"),
                    packet(5, "registered"),
                    packet(17, "last")
                ]
                .into_iter(),
                &mut budget
            )
            .is_err(),
            "mutation {mutation}"
        );
    }
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    assert!(
        project_enrolled(
            header(),
            &original(),
            [packet(2, "first")].into_iter(),
            &mut budget
        )
        .is_err()
    );
    assert!(project_enrolled(header(), &[], [].into_iter(), &mut budget).is_err());
}

#[test]
fn original_projection_exact_work_and_one_short_keep_prefix_account() {
    let roots = [
        packet(2, "first"),
        packet(5, "registered"),
        packet(17, "last"),
    ];
    let mut measured_work = Work::new(usize::MAX);
    let mut measured = Budget::new(&mut measured_work, usize::MAX);
    project_enrolled(header(), &original(), roots.into_iter(), &mut measured).unwrap();
    let cost = measured.work();
    let peak = measured.peak_storage();
    for (work_limit, storage_limit, success) in [
        (cost + 7, peak + 31, true),
        (cost + 6, peak + 31, false),
        (cost + 7, peak + 30, false),
    ] {
        let mut work = Work::new(work_limit);
        let mut budget = Budget::new(&mut work, storage_limit);
        budget.charge_work(7).unwrap();
        budget.reserve_storage(31).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let result = project_enrolled(header(), &original(), roots.into_iter(), &mut budget);
        assert_eq!(result.is_ok(), success);
        assert!(budget.work_ledger_identity_v1() == ledger);
        assert!(budget.storage() >= 31);
        if success {
            assert_eq!(budget.work(), 7 + cost);
        } else {
            assert!(budget.failed_work().is_some() || budget.failed_storage().is_some());
        }
    }
}

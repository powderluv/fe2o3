//! Shape-only unit owners: these fixtures do not model authenticated source
//! import, transcript validity, live rustc custody, or executable authority.
use super::*;

fn inventory(bytes: &[u8], spare: usize) -> AuthenticatedRustcIdentityInventoryV3 {
    let mut data = Vec::with_capacity(spare);
    data.extend_from_slice(bytes);
    AuthenticatedRustcIdentityInventoryV3 {
        sha256: [7; 32],
        canonical_transcript: data.into_boxed_slice(),
        original_root_associations: None,
    }
}

fn plan(bytes: &[u8], spare: usize) -> AuthenticatedRustcPreflightPlanV3 {
    let mut data = Vec::with_capacity(spare);
    data.extend_from_slice(bytes);
    AuthenticatedRustcPreflightPlanV3 {
        sha256: [8; 32],
        rustc_identity_inventory_sha256: [7; 32],
        canonical_transcript: data.into_boxed_slice(),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Refused {
    Arithmetic,
    Bytes,
    Visits,
}

fn charge(
    state: &mut (usize, usize),
    count: usize,
    width: usize,
    max_bytes: usize,
    max_visits: usize,
) -> Result<(), Refused> {
    let bytes = count.checked_mul(width).ok_or(Refused::Arithmetic)?;
    let next = (
        state.0.checked_add(bytes).ok_or(Refused::Arithmetic)?,
        state.1.checked_add(1).ok_or(Refused::Arithmetic)?,
    );
    if next.0 > max_bytes {
        return Err(Refused::Bytes);
    }
    if next.1 > max_visits {
        return Err(Refused::Visits);
    }
    *state = next;
    Ok(())
}

#[test]
fn inventory_reports_original_box_without_header_or_digest_change() {
    let owner = inventory(b"identity", 0);
    let before = owner.canonical_transcript.as_ptr();
    let mut calls = Vec::new();
    owner
        .visit_retained_heap_storage_v1(|n, w| {
            calls.push((n, w));
            Ok::<_, Refused>(())
        })
        .unwrap();
    assert_eq!(calls, [(8, 1)]);
    assert_eq!(owner.sha256(), [7; 32]);
    assert_eq!(owner.canonical_transcript(), b"identity");
    assert_eq!(owner.canonical_transcript.as_ptr(), before);
}

#[test]
fn preflight_reports_one_box_not_an_inventory_alias() {
    let owner = plan(b"preflight", 0);
    let before = owner.canonical_transcript.as_ptr();
    let mut calls = Vec::new();
    owner
        .visit_retained_heap_storage_v1(|n, w| {
            calls.push((n, w));
            Ok::<_, Refused>(())
        })
        .unwrap();
    assert_eq!(calls, [(9, 1)]);
    assert_eq!(owner.sha256(), [8; 32]);
    assert_eq!(owner.rustc_identity_inventory_sha256(), [7; 32]);
    assert_eq!(owner.canonical_transcript(), b"preflight");
    assert_eq!(owner.canonical_transcript.as_ptr(), before);
}

#[test]
fn empty_owners_each_require_a_callback_visit() {
    let mut state = (0, 0);
    inventory(b"", 0)
        .visit_retained_heap_storage_v1(|n, w| charge(&mut state, n, w, 0, 2))
        .unwrap();
    plan(b"", 0)
        .visit_retained_heap_storage_v1(|n, w| charge(&mut state, n, w, 0, 2))
        .unwrap();
    assert_eq!(state, (0, 2));
}

#[test]
fn consumed_vector_capacity_is_not_retained_by_a_box() {
    for spare in [0, 64, 4096] {
        let mut state = (0, 0);
        inventory(b"abc", spare)
            .visit_retained_heap_storage_v1(|n, w| charge(&mut state, n, w, 6, 2))
            .unwrap();
        plan(b"abc", spare)
            .visit_retained_heap_storage_v1(|n, w| charge(&mut state, n, w, 6, 2))
            .unwrap();
        assert_eq!(state, (6, 2));
    }
}

#[test]
fn equal_bytes_in_distinct_owners_are_both_counted() {
    let left = inventory(b"same", 0);
    let right = plan(b"same", 0);
    assert_ne!(
        left.canonical_transcript.as_ptr(),
        right.canonical_transcript.as_ptr()
    );
    let mut state = (0, 0);
    left.visit_retained_heap_storage_v1(|n, w| charge(&mut state, n, w, 8, 2))
        .unwrap();
    right
        .visit_retained_heap_storage_v1(|n, w| charge(&mut state, n, w, 8, 2))
        .unwrap();
    assert_eq!(state, (8, 2));
}

#[test]
fn same_ledger_exact_and_one_short_limits_preserve_incomplete_prefix() {
    let left = inventory(b"abc", 0);
    let right = plan(b"12345", 0);
    let prefix = 13
        + size_of::<AuthenticatedRustcIdentityInventoryV3>()
        + size_of::<AuthenticatedRustcPreflightPlanV3>();
    let mut exact = (prefix, 2);
    left.visit_retained_heap_storage_v1(|n, w| charge(&mut exact, n, w, prefix + 8, 4))
        .unwrap();
    right
        .visit_retained_heap_storage_v1(|n, w| charge(&mut exact, n, w, prefix + 8, 4))
        .unwrap();
    assert_eq!(exact, (prefix + 8, 4));
    for (max_bytes, max_visits, error) in [
        (prefix + 7, 4, Refused::Bytes),
        (prefix + 8, 3, Refused::Visits),
    ] {
        let mut partial = (prefix, 2);
        left.visit_retained_heap_storage_v1(|n, w| {
            charge(&mut partial, n, w, max_bytes, max_visits)
        })
        .unwrap();
        assert_eq!(
            right.visit_retained_heap_storage_v1(|n, w| {
                charge(&mut partial, n, w, max_bytes, max_visits)
            }),
            Err(error)
        );
        assert_eq!(partial, (prefix + 3, 3));
    }
}

#[test]
fn each_first_error_is_returned_unchanged_with_one_callback() {
    let mut calls = 0;
    assert_eq!(
        inventory(b"x", 0).visit_retained_heap_storage_v1(|_, _| {
            calls += 1;
            Err::<(), _>("inventory sentinel")
        }),
        Err("inventory sentinel")
    );
    assert_eq!(calls, 1);
    calls = 0;
    assert_eq!(
        plan(b"y", 0).visit_retained_heap_storage_v1(|_, _| {
            calls += 1;
            Err::<(), _>("preflight sentinel")
        }),
        Err("preflight sentinel")
    );
    assert_eq!(calls, 1);
}

#[test]
fn checked_addition_and_multiplication_never_saturate() {
    for initial in [(usize::MAX, 0), (0, usize::MAX)] {
        let mut state = initial;
        assert_eq!(
            inventory(b"x", 0).visit_retained_heap_storage_v1(|n, w| {
                charge(&mut state, n, w, usize::MAX, usize::MAX)
            }),
            Err(Refused::Arithmetic)
        );
        assert_eq!(state, initial);
        assert_eq!(
            plan(b"x", 0).visit_retained_heap_storage_v1(|n, w| {
                charge(&mut state, n, w, usize::MAX, usize::MAX)
            }),
            Err(Refused::Arithmetic)
        );
        assert_eq!(state, initial);
    }
    assert_eq!(
        charge(&mut (0, 0), usize::MAX, 2, usize::MAX, usize::MAX),
        Err(Refused::Arithmetic)
    );
}

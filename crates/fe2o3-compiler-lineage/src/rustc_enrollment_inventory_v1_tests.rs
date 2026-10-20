use super::*;
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

const LIMIT: usize = MAX_NATIVE_CONDITIONAL_STORAGE_V1;

#[test]
fn legacy_discriminator_recognizes_only_the_original_domain_and_target_prefix() {
    let domain = b"fe2o3/semantic-mir/rustc-identity-inventory/v2";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&32u64.to_le_bytes());
    bytes.extend_from_slice(&[7; 32]);
    let prefix = bytes.len();
    bytes.extend_from_slice(&4u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    assert!(has_rustc_identity_inventory_prefix_v2(&bytes));
    for end in 0..=prefix {
        assert!(!has_rustc_identity_inventory_prefix_v2(&bytes[..end]));
    }
    for offset in [0, 8, 8 + domain.len()] {
        let mut wrong = bytes.clone();
        wrong[offset] ^= 1;
        assert!(!has_rustc_identity_inventory_prefix_v2(&wrong));
    }
    assert!(!has_rustc_identity_inventory_prefix_v2(&fixture()));
    bytes.resize(MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3 + 1, 0);
    assert!(!has_rustc_identity_inventory_prefix_v2(&bytes));
    assert!(
        RUSTC_IDENTITY_INVENTORY_PREFIX_WORK_V2
            <= NativeConditionalCpuMappingExpectationV1::HEADER_MATCH_WORK
    );
}

fn independent_mapping_expectation() -> NativeConditionalCpuMappingExpectationV1 {
    NativeConditionalCpuMappingExpectationV1 {
        rustc_invocation_sha256: [0x11; 32],
        native_policy_sha256: [0x22; 32],
        policy_generation: 0x0807_0605_0403_0201,
        enrollment_binding_count: 1,
    }
}

#[test]
fn mapping_coordinates_reject_canonically_decoded_foreign_inventory_headers() {
    let expected = independent_mapping_expectation();
    for case in 0..5 {
        let mut claimed = header(1, 1);
        let mut root = row(2, 1, 0);
        match case {
            1 => claimed.invocation_identity[0] ^= 1,
            2 => claimed.native_policy_identity[0] ^= 1,
            3 => claimed.native_policy_generation += 1,
            4 => {
                claimed.enrollment_binding_count = 0;
                root.origin_tag = 0;
            }
            _ => {}
        }
        let bytes = encode(input(b"old", claimed, &[root]));
        let inventory = read(&bytes).unwrap();
        assert_eq!(expected.matches_header(&inventory.header()), case == 0);
    }
    assert_eq!(
        NativeConditionalCpuMappingExpectationV1::HEADER_MATCH_WORK,
        80 + size_of::<NativeConditionalCpuMappingExpectationV1>()
    );
}

#[test]
fn equal_headers_do_not_make_different_root_inventories_equivalent() {
    let expected = independent_mapping_expectation();
    let first = encode(input(b"old", header(1, 1), &[row(2, 1, 0)]));
    let mut changed = row(2, 1, 0);
    changed.reference_instance[0] ^= 1;
    let second = encode(input(b"old", header(1, 1), &[changed]));
    let first = read(&first).unwrap();
    let second = read(&second).unwrap();
    assert!(expected.matches_header(&first.header()));
    assert!(expected.matches_header(&second.header()));
    assert_ne!(first.full_wrapper_sha256(), second.full_wrapper_sha256());
    assert_ne!(
        first.roots().next().unwrap().value(),
        second.roots().next().unwrap().value()
    );
    assert!(!first.grants_authority() && !second.grants_authority());
}

#[test]
fn missing_truncated_or_legacy_inventory_cannot_supply_a_mapping_header() {
    let bytes = fixture();
    for missing in [&[][..], &b"old"[..], &bytes[..bytes.len() - 1]] {
        assert!(read(missing).is_err());
    }
}

#[test]
fn mapping_count_bound_is_checked_even_when_the_claimed_header_agrees() {
    let mut expected = independent_mapping_expectation();
    let count = fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 as u32 + 1;
    expected.enrollment_binding_count = count;
    assert!(!expected.matches_header(&header(count, count)));
}

fn header(count: u32, enrolled: u32) -> RustcEnrollmentInventoryHeaderV1 {
    RustcEnrollmentInventoryHeaderV1 {
        kernel_count: count,
        enrollment_binding_count: enrolled,
        invocation_identity: [0x11; 32],
        native_policy_identity: [0x22; 32],
        native_policy_generation: 0x0807_0605_0403_0201,
    }
}
fn row(id: u32, origin: u8, ordinal: u32) -> RustcEnrollmentInventoryRootV1 {
    RustcEnrollmentInventoryRootV1 {
        semantic_root: id,
        origin_tag: origin,
        descriptor_ordinal: ordinal,
        logical_name_len: 7,
        logical_name_sha256: [0x33; 32],
        kernel_binding: [0x44; 32],
        kernel_instance: [0x55; 32],
        reference_instance: [0x66; 32],
    }
}
fn input<'a>(
    legacy: &'a [u8],
    header: RustcEnrollmentInventoryHeaderV1,
    roots: &'a [RustcEnrollmentInventoryRootV1],
) -> RustcEnrollmentInventoryInputV1<'a> {
    RustcEnrollmentInventoryInputV1 {
        legacy_inventory: legacy,
        header,
        roots,
    }
}
fn encode(input: RustcEnrollmentInventoryInputV1<'_>) -> Vec<u8> {
    encode_rustc_enrollment_inventory_v1(input, LIMIT, |_| Ok::<_, ()>(())).unwrap()
}
fn read(bytes: &[u8]) -> Result<RustcEnrollmentInventoryRefV1<'_>, Error<()>> {
    read_rustc_enrollment_inventory_v1(bytes, LIMIT, |_| Ok::<_, ()>(()))
}
fn fixture() -> Vec<u8> {
    encode(input(
        b"old",
        header(3, 2),
        &[row(2, 1, 1), row(7, 0, 0), row(90, 1, 0)],
    ))
}
fn association_start(bytes: &[u8]) -> usize {
    48 + u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize
}
fn rehash(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut hash = Sha256::new();
    hash.update(b"FE2O3/RUSTC-ENROLLMENT-INVENTORY/V1\0");
    hash.update((end as u64).to_le_bytes());
    hash.update(&bytes[..end]);
    let digest: [u8; 32] = hash.finalize().into();
    bytes[end..].copy_from_slice(&digest);
}
fn reject_rehashed(mut bytes: Vec<u8>, expected: Error<()>) {
    rehash(&mut bytes);
    assert_eq!(read(&bytes).err(), Some(expected));
}

#[test]
fn exact_golden_wire_uses_nul_domain_and_full_wrapper_identity() {
    let roots = [row(17, 1, 0)];
    let bytes = encode(input(b"old", header(1, 1), &roots));
    let mut golden = Vec::new();
    golden.extend_from_slice(b"F2RINV1\0");
    golden.extend_from_slice(&[1, 0, 1, 0]);
    golden.extend_from_slice(&48_u32.to_le_bytes());
    golden.extend_from_slice(&307_u64.to_le_bytes());
    golden.extend_from_slice(&3_u64.to_le_bytes());
    golden.extend_from_slice(&224_u64.to_le_bytes());
    golden.extend_from_slice(&[0; 8]);
    golden.extend_from_slice(b"old");
    golden.extend_from_slice(&1_u32.to_le_bytes());
    golden.extend_from_slice(&1_u32.to_le_bytes());
    golden.extend_from_slice(&[0x11; 32]);
    golden.extend_from_slice(&[0x22; 32]);
    golden.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    golden.extend_from_slice(&17_u32.to_le_bytes());
    golden.extend_from_slice(&[1, 0, 0, 0]);
    golden.extend_from_slice(&0_u32.to_le_bytes());
    golden.extend_from_slice(&7_u32.to_le_bytes());
    golden.extend_from_slice(&[0x33; 32]);
    golden.extend_from_slice(&[0x44; 32]);
    golden.extend_from_slice(&[0x55; 32]);
    golden.extend_from_slice(&[0x66; 32]);
    assert_eq!(golden.len(), 275);
    let mut digest = Sha256::new();
    digest.update(b"FE2O3/RUSTC-ENROLLMENT-INVENTORY/V1\0");
    digest.update(275_u64.to_le_bytes());
    digest.update(&golden);
    golden.extend_from_slice(&digest.finalize());
    assert_eq!(bytes, golden);
    let decoded = read(&bytes).unwrap();
    assert_eq!(decoded.canonical_bytes(), bytes);
    assert_eq!(decoded.legacy_inventory(), b"old");
    assert_eq!(decoded.header(), header(1, 1));
    assert_eq!(decoded.header_ref().canonical_bytes(), &bytes[51..131]);
    assert_eq!(decoded.roots().next().unwrap().value(), roots[0]);
    assert_eq!(
        decoded.roots().next().unwrap().canonical_bytes(),
        &bytes[131..275]
    );
    assert_eq!(decoded.byte_len(), 307);
    let raw: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!(*decoded.full_wrapper_sha256(), raw);
    assert_eq!(decoded.framing_sha256().as_slice(), &bytes[275..]);
    // Independently calculated from the explicit wire above, not this codec.
    assert_eq!(
        *decoded.framing_sha256(),
        [
            0x1e, 0x23, 0x2e, 0x3e, 0x4e, 0x42, 0xd5, 0xd8, 0x6d, 0x3e, 0x32, 0xef, 0xd0, 0xd5,
            0xe4, 0xb5, 0x24, 0xf4, 0xc4, 0x39, 0x75, 0x48, 0xaf, 0xdd, 0x2f, 0xf9, 0x56, 0xea,
            0xfe, 0x7b, 0x33, 0xdb,
        ]
    );
    assert_eq!(
        *decoded.full_wrapper_sha256(),
        [
            0xe0, 0x2e, 0xca, 0x08, 0x39, 0x9d, 0x6f, 0x2e, 0xaf, 0x5b, 0xbc, 0x56, 0x94, 0x86,
            0x4a, 0x76, 0x53, 0xc5, 0xdf, 0x77, 0xb7, 0x00, 0x76, 0x98, 0x5c, 0xc7, 0xe3, 0x34,
            0x9e, 0xef, 0x6c, 0xeb,
        ]
    );
    assert_ne!(decoded.full_wrapper_sha256(), decoded.framing_sha256());
    assert!(!decoded.grants_authority());
}

#[test]
fn noncontiguous_canonical_roots_and_independent_descriptor_permutation_round_trip() {
    let roots = [
        row(2, 1, 2),
        row(7, 0, 0),
        row(100, 1, 0),
        row(u32::MAX, 1, 1),
    ];
    let bytes = encode(input(b"unchanged legacy", header(4, 3), &roots));
    let decoded = read(&bytes).unwrap();
    assert_eq!(decoded.root_count(), 4);
    assert_eq!(decoded.roots().len(), 4);
    assert_eq!(
        decoded.roots().map(|r| r.value()).collect::<Vec<_>>(),
        roots
    );
}

#[test]
fn empty_complete_subset_remains_inert_and_nonempty_legacy_is_preserved() {
    let bytes = encode(input(b"legacy ffi-only inventory", header(0, 0), &[]));
    let decoded = read(&bytes).unwrap();
    assert_eq!(decoded.root_count(), 0);
    assert_eq!(decoded.roots().len(), 0);
    assert!(!decoded.grants_authority());
}

#[test]
fn codec_does_not_parse_or_reinterpret_legacy_member() {
    let legacy = [0xff, 0, 1, 2, 0x80];
    let bytes = encode(input(&legacy, header(0, 0), &[]));
    assert_eq!(read(&bytes).unwrap().legacy_inventory(), legacy);
}

#[test]
fn in_place_encoding_initializes_all_reserved_bytes() {
    let roots = [row(3, 0, 0)];
    let input = input(b"old", header(1, 0), &roots);
    let expected = encode(input);
    let mut output = vec![0xa5; expected.len()];
    encode_rustc_enrollment_inventory_into_v1(input, &mut output, LIMIT, |_| Ok::<_, ()>(()))
        .unwrap();
    assert_eq!(output, expected);
}

#[test]
fn layout_rejects_empty_legacy_and_checked_arithmetic_overflow() {
    assert_eq!(
        RustcEnrollmentInventoryLayoutV1::new::<()>(0, 0),
        Err(Error::LegacyLength)
    );
    assert_eq!(
        RustcEnrollmentInventoryLayoutV1::new::<()>(1, usize::MAX),
        Err(Error::Arithmetic)
    );
    assert_eq!(
        RustcEnrollmentInventoryLayoutV1::new::<()>(usize::MAX, 0),
        Err(Error::LegacyLength)
    );
}

#[test]
fn aggregate_ceiling_is_not_a_separate_allowance_for_each_member() {
    assert_eq!(
        RustcEnrollmentInventoryLayoutV1::new::<()>(MAX, 0),
        Err(Error::Length)
    );
    assert!(RustcEnrollmentInventoryLayoutV1::new::<()>(MAX / 2, MAX / (2 * ROW)).is_err());
    assert_eq!(
        RustcEnrollmentInventoryLayoutV1::new::<()>(MAX - 160, 0)
            .unwrap()
            .encoded_len(),
        MAX
    );
    assert_eq!(
        RustcEnrollmentInventoryLayoutV1::new::<()>(MAX - 159, 0),
        Err(Error::Length)
    );
}

#[test]
fn aggregate_exact_cap_round_trip_and_one_byte_extra_refused() {
    let legacy = vec![0x5a; MAX - 160];
    let bytes = encode(input(&legacy, header(0, 0), &[]));
    assert_eq!(bytes.len(), MAX);
    assert_eq!(bytes.capacity(), MAX);
    assert_eq!(read(&bytes).unwrap().legacy_inventory(), legacy);
    let mut oversized = bytes;
    oversized.push(0);
    assert_eq!(read(&oversized).err(), Some(Error::Length));
}

#[test]
fn maximum_derived_census_handles_reverse_descriptor_order_linearly() {
    let roots = (0..MAX_ROOTS)
        .map(|i| row(i as u32, 1, (MAX_ROOTS - i - 1) as u32))
        .collect::<Vec<_>>();
    let bytes = encode(input(
        b"x",
        header(MAX_ROOTS as u32, MAX_ROOTS as u32),
        &roots,
    ));
    let decoded = read(&bytes).unwrap();
    assert_eq!(decoded.roots().count(), MAX_ROOTS);
    assert!(RustcEnrollmentInventoryLayoutV1::new::<()>(1, MAX_ROOTS + 1).is_err());
    assert!(bytes.len() <= MAX);
}

#[test]
fn owned_storage_quote_is_exact_and_independent_of_input_backing() {
    let layout = RustcEnrollmentInventoryLayoutV1::new::<()>(3, 1).unwrap();
    assert_eq!(layout.encoded_len(), 307);
    assert_eq!(layout.root_count(), 1);
    assert_eq!(
        layout.owned_storage_bytes::<()>().unwrap(),
        307 + size_of::<Vec<u8>>()
    );
    let roots = [row(1, 0, 0)];
    let input = input(b"old", header(1, 0), &roots);
    let required =
        layout.owned_storage_bytes::<()>().unwrap() + RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1;
    assert_eq!(
        encode_rustc_enrollment_inventory_v1(input, required - 1, |_| Ok::<_, ()>(())).unwrap_err(),
        Error::StorageLimit
    );
    let bytes = encode_rustc_enrollment_inventory_v1(input, required, |_| Ok::<_, ()>(())).unwrap();
    assert_eq!(bytes.capacity(), layout.encoded_len());
    assert_eq!(
        encode_rustc_enrollment_inventory_v1(input, LIMIT + 1, |_| Ok::<_, ()>(())).unwrap_err(),
        Error::StorageLimit
    );
}

#[test]
fn borrowed_decode_and_in_place_storage_exact_and_one_short() {
    let roots = [row(1, 0, 0)];
    let input = input(b"x", header(1, 0), &roots);
    let bytes = encode(input);
    let required = RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1;
    assert!(read_rustc_enrollment_inventory_v1(&bytes, required, |_| Ok::<_, ()>(())).is_ok());
    assert_eq!(
        read_rustc_enrollment_inventory_v1(&bytes, required - 1, |_| Ok::<_, ()>(())).err(),
        Some(Error::StorageLimit)
    );
    assert_eq!(
        read_rustc_enrollment_inventory_v1(&bytes, LIMIT + 1, |_| Ok::<_, ()>(())).err(),
        Some(Error::StorageLimit)
    );
    let mut dest = vec![0x5a; bytes.len()];
    let original = dest.clone();
    assert_eq!(
        encode_rustc_enrollment_inventory_into_v1(input, &mut dest, required - 1, |_| Ok::<_, ()>(
            ()
        )),
        Err(Error::StorageLimit)
    );
    assert_eq!(dest, original);
    encode_rustc_enrollment_inventory_into_v1(input, &mut dest, required, |_| Ok::<_, ()>(()))
        .unwrap();
    assert_eq!(dest, bytes);
}

struct Meter {
    remaining: usize,
    floor: usize,
    denied: bool,
}
impl Meter {
    fn charge(&mut self, n: usize) -> Result<(), &'static str> {
        if self.denied {
            return Err("first denial");
        }
        if n > self.remaining.saturating_sub(self.floor) {
            self.denied = true;
            return Err("first denial");
        }
        self.remaining -= n;
        Ok(())
    }
}

#[test]
fn encode_work_exact_one_short_zero_and_inherited_nonzero_floor() {
    let roots = [row(1, 0, 0)];
    let input = input(b"x", header(1, 0), &roots);
    let mut work = 0;
    encode_rustc_enrollment_inventory_v1(input, LIMIT, |n| {
        work += n;
        Ok::<_, &str>(())
    })
    .unwrap();
    assert!(work > 0);
    for credit in [0, work - 1, work] {
        let mut meter = Meter {
            remaining: 97 + credit,
            floor: 97,
            denied: false,
        };
        let encoded = encode_rustc_enrollment_inventory_v1(input, LIMIT, |n| meter.charge(n));
        assert_eq!(encoded.is_ok(), credit == work);
        assert!(meter.remaining >= 97);
        if credit == work {
            assert_eq!(meter.remaining, 97);
        } else {
            assert_eq!(encoded.unwrap_err(), Error::Charge("first denial"));
            assert_eq!(meter.charge(0), Err("first denial"));
        }
    }
}

#[test]
fn read_work_exact_one_short_zero_and_inherited_nonzero_floor() {
    let bytes = fixture();
    let mut work = 0;
    read_rustc_enrollment_inventory_v1(&bytes, LIMIT, |n| {
        work += n;
        Ok::<_, &str>(())
    })
    .unwrap();
    for credit in [0, work - 1, work] {
        let mut meter = Meter {
            remaining: 113 + credit,
            floor: 113,
            denied: false,
        };
        let result = read_rustc_enrollment_inventory_v1(&bytes, LIMIT, |n| meter.charge(n));
        assert_eq!(result.is_ok(), credit == work);
        assert!(meter.remaining >= 113);
        if credit == work {
            assert_eq!(meter.remaining, 113);
        } else {
            assert_eq!(result.err(), Some(Error::Charge("first denial")));
            assert_eq!(meter.charge(0), Err("first denial"));
        }
    }
}

#[test]
fn every_reader_callback_denial_retains_original_error() {
    let bytes = fixture();
    let mut total_calls = 0;
    read_rustc_enrollment_inventory_v1(&bytes, LIMIT, |_| {
        total_calls += 1;
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(total_calls, 3);
    for denied_call in 0..total_calls {
        let mut call = 0;
        let result = read_rustc_enrollment_inventory_v1(&bytes, LIMIT, |_| {
            let current = call;
            call += 1;
            if current == denied_call {
                Err(denied_call)
            } else {
                Ok(())
            }
        });
        assert_eq!(result.err(), Some(Error::Charge(denied_call)));
        assert_eq!(call, denied_call + 1);
    }
}

#[test]
fn write_denial_never_mutates_destination() {
    let roots = [row(1, 0, 0)];
    let input = input(b"x", header(1, 0), &roots);
    let mut bytes = vec![0x5a; 305];
    let original = bytes.clone();
    assert_eq!(
        encode_rustc_enrollment_inventory_into_v1(input, &mut bytes, LIMIT, |_| Err("original")),
        Err(Error::Charge("original"))
    );
    assert_eq!(bytes, original);
}

struct DropCallback<'a> {
    dropped: &'a Cell<bool>,
    panic: bool,
}
impl DropCallback<'_> {
    fn touch(&self) {}
}
impl Drop for DropCallback<'_> {
    fn drop(&mut self) {
        self.dropped.set(true);
        assert!(!self.panic, "callback drop");
    }
}

#[test]
fn callback_destructor_unwind_precedes_destination_mutation() {
    let roots = [row(1, 0, 0)];
    let input = input(b"x", header(1, 0), &roots);
    let dropped = Cell::new(false);
    let captured = DropCallback {
        dropped: &dropped,
        panic: true,
    };
    let mut dest = vec![0xa5; 305];
    let original = dest.clone();
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = encode_rustc_enrollment_inventory_into_v1(input, &mut dest, LIMIT, move |_| {
            captured.touch();
            Ok::<_, ()>(())
        });
    }));
    assert!(panic.is_err());
    assert!(dropped.get());
    assert_eq!(dest, original);
}

#[test]
fn callback_destructor_runs_before_owned_result_or_borrowed_view_returns() {
    let roots = [row(1, 0, 0)];
    let dropped = Cell::new(false);
    let captured = DropCallback {
        dropped: &dropped,
        panic: false,
    };
    let bytes =
        encode_rustc_enrollment_inventory_v1(input(b"x", header(1, 0), &roots), LIMIT, move |_| {
            captured.touch();
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(dropped.replace(false));
    let captured = DropCallback {
        dropped: &dropped,
        panic: false,
    };
    let decoded = read_rustc_enrollment_inventory_v1(&bytes, LIMIT, move |_| {
        captured.touch();
        Ok::<_, ()>(())
    })
    .unwrap();
    assert!(dropped.get());
    assert!(!decoded.grants_authority());
}

#[test]
fn reader_callback_destructor_unwind_cannot_return_provisional_view() {
    let bytes = fixture();
    let dropped = Cell::new(false);
    let captured = DropCallback {
        dropped: &dropped,
        panic: true,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        read_rustc_enrollment_inventory_v1(&bytes, LIMIT, move |_| {
            captured.touch();
            Ok::<_, ()>(())
        })
    }));
    assert!(result.is_err());
    assert!(dropped.get());
}

#[test]
fn untrusted_outer_magic_version_policy_and_header_size_rejected_after_rehash() {
    for index in [0, 7, 8, 9, 10, 11, 12, 15] {
        let mut bytes = fixture();
        bytes[index] ^= 1;
        reject_rehashed(bytes, Error::Header);
    }
}

#[test]
fn all_outer_and_row_reserved_bytes_rejected_after_rehash() {
    for index in 40..48 {
        let mut bytes = fixture();
        bytes[index] = 1;
        reject_rehashed(bytes, Error::Reserved);
    }
    for row in 0..3 {
        for reserved in 5..8 {
            let mut bytes = fixture();
            let index = association_start(&bytes) + HEADER + row * ROW + reserved;
            bytes[index] = 1;
            reject_rehashed(bytes, Error::Reserved);
        }
    }
}

#[test]
fn every_truncation_and_trailing_byte_rejected() {
    let bytes = fixture();
    for len in 0..bytes.len() {
        assert!(read(&bytes[..len]).is_err(), "accepted truncation {len}");
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert!(read(&trailing).is_err());
}

#[test]
fn every_unrehashed_byte_mutation_is_rejected() {
    let bytes = fixture();
    for index in 0..bytes.len() {
        let mut altered = bytes.clone();
        altered[index] ^= 1;
        assert!(read(&altered).is_err(), "accepted mutation {index}");
    }
}

#[test]
fn outer_total_and_member_lengths_are_exact_not_prefix_accepting() {
    for at in [16, 24, 32] {
        let mut bytes = fixture();
        bytes[at] ^= 1;
        rehash(&mut bytes);
        assert!(read(&bytes).is_err());
    }
    let mut bytes = fixture();
    bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    rehash(&mut bytes);
    assert!(read(&bytes).is_err());
}

#[test]
fn second_member_requires_exact_header_and_fixed_rows() {
    for association in [vec![0; 79], vec![0; 81], vec![0; 223]] {
        let layout = pair::Layout::new::<()>(&POLICY, 1, association.len()).unwrap();
        let mut bytes = vec![0; layout.encoded_len()];
        bytes[48] = b'x';
        bytes[layout.second_range()].copy_from_slice(&association);
        pair::seal(&POLICY, layout, &mut bytes, LIMIT, |_| Ok::<_, ()>(())).unwrap();
        assert_eq!(read(&bytes).err(), Some(Error::AssociationLength));
    }
}

#[test]
fn header_complete_and_enrollment_counts_are_independent_exact_censuses() {
    for (offset, value, expected) in [
        (0, 2_u32, Error::Count),
        (0, 4, Error::Count),
        (4, 4, Error::Count),
        (4, 3, Error::EnrollmentOrdinal),
        (4, 1, Error::EnrollmentOrdinal),
    ] {
        let mut bytes = fixture();
        let at = association_start(&bytes) + offset;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        reject_rehashed(bytes, expected);
    }
}

#[test]
fn duplicate_and_reversed_semantic_ids_rejected_without_sorting() {
    for id in [0_u32, 2] {
        let mut bytes = fixture();
        let at = association_start(&bytes) + HEADER + ROW;
        bytes[at..at + 4].copy_from_slice(&id.to_le_bytes());
        reject_rehashed(bytes, Error::RootOrder);
    }
}

#[test]
fn unsupported_tags_and_registration_nonzero_ordinal_rejected() {
    for tag in [2_u8, 3, 255] {
        let mut bytes = fixture();
        let at = association_start(&bytes) + HEADER;
        bytes[at + 4] = tag;
        reject_rehashed(bytes, Error::Origin);
    }
    let mut bytes = fixture();
    let at = association_start(&bytes) + HEADER + ROW;
    bytes[at + 8..at + 12].copy_from_slice(&1_u32.to_le_bytes());
    reject_rehashed(bytes, Error::Origin);
}

#[test]
fn missing_duplicate_and_out_of_range_descriptor_ordinals_rejected() {
    for ordinal in [1_u32, 2, u32::MAX] {
        let mut bytes = fixture();
        let at = association_start(&bytes) + HEADER + 2 * ROW + 8;
        bytes[at..at + 4].copy_from_slice(&ordinal.to_le_bytes());
        reject_rehashed(bytes, Error::EnrollmentOrdinal);
    }
    let mut bytes = fixture();
    let at = association_start(&bytes) + HEADER + 2 * ROW;
    bytes[at + 4] = 0;
    reject_rehashed(bytes, Error::EnrollmentOrdinal);
}

#[test]
fn original_reference_identity_is_preserved_for_both_origin_variants() {
    let mut roots = [row(1, 0, 0), row(4, 1, 0)];
    roots[0].reference_instance = [0xa1; 32];
    roots[1].reference_instance = [0xb2; 32];
    let bytes = encode(input(b"legacy", header(2, 1), &roots));
    assert_eq!(
        read(&bytes)
            .unwrap()
            .roots()
            .map(|r| r.value().reference_instance)
            .collect::<Vec<_>>(),
        vec![[0xa1; 32], [0xb2; 32]]
    );
}

#[test]
fn coherently_rehashed_claim_changes_are_inert_not_original_provenance() {
    // These are legal framing changes, NOT authenticated admission positives.
    // Actual source/owner/policy joins must reject substituted original axes.
    for offset in [
        8,
        40,
        72,
        HEADER + 12,
        HEADER + 16,
        HEADER + 48,
        HEADER + 80,
        HEADER + 112,
    ] {
        let mut bytes = fixture();
        let at = association_start(&bytes) + offset;
        bytes[at] ^= 0x80;
        rehash(&mut bytes);
        let decoded = read(&bytes).unwrap();
        assert!(!decoded.grants_authority());
        assert_eq!(decoded.canonical_bytes(), bytes);
    }
}

#[test]
fn every_invalid_input_axis_is_refused_before_destination_mutation() {
    let valid = [row(2, 1, 1), row(7, 0, 0), row(90, 1, 0)];
    let cases = [
        (header(2, 2), valid, Error::Count),
        (header(3, 4), valid, Error::Count),
        (header(3, 3), valid, Error::EnrollmentOrdinal),
        (
            header(3, 2),
            [row(2, 1, 1), row(2, 0, 0), row(90, 1, 0)],
            Error::RootOrder,
        ),
        (
            header(3, 2),
            [row(2, 1, 1), row(7, 2, 0), row(90, 1, 0)],
            Error::Origin,
        ),
        (
            header(3, 2),
            [row(2, 1, 1), row(7, 0, 1), row(90, 1, 0)],
            Error::Origin,
        ),
        (
            header(3, 2),
            [row(2, 1, 1), row(7, 0, 0), row(90, 1, 1)],
            Error::EnrollmentOrdinal,
        ),
    ];
    for (header, roots, error) in cases {
        let mut bytes = vec![0xa5; 595];
        let original = bytes.clone();
        let result = encode_rustc_enrollment_inventory_into_v1(
            input(b"old", header, &roots),
            &mut bytes,
            LIMIT,
            |_| Ok::<_, ()>(()),
        );
        assert_eq!(result, Err(error));
        assert_eq!(bytes, original);
    }
}

#[test]
fn wrong_destination_extent_is_refused_before_callbacks_or_writes() {
    let roots = [row(1, 0, 0)];
    for length in [0, 304, 306] {
        let mut bytes = vec![0xa5; length];
        let original = bytes.clone();
        let result = encode_rustc_enrollment_inventory_into_v1(
            input(b"x", header(1, 0), &roots),
            &mut bytes,
            LIMIT,
            |_| -> Result<(), ()> { panic!("callback on impossible extent") },
        );
        assert_eq!(result, Err(Error::Length));
        assert_eq!(bytes, original);
    }
}

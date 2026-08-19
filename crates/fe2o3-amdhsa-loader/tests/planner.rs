use std::{env, fs};

use fe2o3_amdhsa_loader::{
    AdmittedProfile, ImageRange, InterSegmentGapOrdinal, LOAD_SEGMENT_COUNT, PlanError,
    SegmentOrdinal, SegmentPermissions, plan, validate,
};

const PHOFF: usize = 64;
const PHENT: usize = 56;
const PHNUM: usize = 8;
const SHOFF: usize = 0x3000;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_NOTE: u32 = 4;
const PT_PHDR: u32 = 6;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PT_GNU_RELRO: u32 = 0x6474_e552;

#[test]
fn builds_a_canonical_inert_plan() {
    let bytes = fixture();
    let plan = plan(&bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    assert_eq!(plan.profile().target(), "gfx942:xnack-");
    assert_eq!(plan.input_len(), bytes.len() as u64);
    assert_eq!(plan.image_start(), 0);
    assert_eq!(plan.image_end(), 0x5000);
    assert_eq!(plan.segments().len(), LOAD_SEGMENT_COUNT);
    assert_eq!(
        plan.segments()
            .iter()
            .map(|segment| segment.permissions())
            .collect::<Vec<_>>(),
        [
            SegmentPermissions::ReadOnly,
            SegmentPermissions::ReadExecute,
            SegmentPermissions::ReadWrite,
        ]
    );
    assert_eq!(plan.segments()[1].mapping_address(), 0x2000);
    assert_eq!(plan.segments()[1].mapping_prefix_size(), 0);
    assert_eq!(plan.segments()[2].zero_fill_size(), 0xf80);
    assert_eq!(plan.metadata_note().file_offset(), 0x214);
    assert_eq!(plan.metadata_note().byte_len(), 1);
}

#[test]
fn binds_sources_metadata_and_zero_then_copy_instructions() {
    let bytes = fixture();
    let envelope = validate(&bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    let inert = plan(&bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    assert_eq!(*envelope.plan(), inert);
    assert_eq!(envelope.input_len(), bytes.len() as u64);
    assert_eq!(envelope.metadata_descriptor(), &[0x80]);

    let first = envelope.segment_source(SegmentOrdinal::First);
    let second = envelope.segment_source(SegmentOrdinal::Second);
    let third = envelope.segment_source(SegmentOrdinal::Third);
    assert_eq!((first.file_offset(), first.byte_len()), (0, 0x300));
    assert_eq!((second.file_offset(), second.byte_len()), (0x1000, 0x100));
    assert_eq!((third.file_offset(), third.byte_len()), (0x2000, 0x80));
    assert_eq!(first.bytes(), &bytes[..0x300]);
    assert_eq!(second.bytes(), &bytes[0x1000..0x1100]);
    assert_eq!(third.bytes(), &bytes[0x2000..0x2080]);

    let materialization = envelope.materialization();
    assert_eq!(materialization.image_len(), 0x5000);
    let zero = materialization.zero_phase();
    assert_range(zero.mapping(SegmentOrdinal::First).destination(), 0, 0x1000);
    assert_range(
        zero.inter_segment_gap(InterSegmentGapOrdinal::FirstToSecond)
            .destination(),
        0x1000,
        0x1000,
    );
    assert_range(
        zero.mapping(SegmentOrdinal::Second).destination(),
        0x2000,
        0x1000,
    );
    assert_range(
        zero.inter_segment_gap(InterSegmentGapOrdinal::SecondToThird)
            .destination(),
        0x3000,
        0x1000,
    );
    assert_range(
        zero.mapping(SegmentOrdinal::Third).destination(),
        0x4000,
        0x1000,
    );

    let copy = materialization.copy_phase();
    assert_range(copy.segment(SegmentOrdinal::First).destination(), 0, 0x300);
    assert_range(
        copy.segment(SegmentOrdinal::Second).destination(),
        0x2000,
        0x100,
    );
    assert_range(
        copy.segment(SegmentOrdinal::Third).destination(),
        0x4000,
        0x80,
    );
    let third_zero = envelope.segment_zero_fill(SegmentOrdinal::Third);
    assert_range(third_zero.mapping_prefix().destination(), 0x4000, 0);
    assert_range(third_zero.memory_suffix().destination(), 0x4080, 0xf80);
    assert_range(third_zero.mapping_tail().destination(), 0x5000, 0);
}

#[test]
fn copy_destination_accounts_for_a_nonzero_mapping_prefix() {
    let mut bytes = fixture();
    write_phdr_u64(&mut bytes, 2, 8, 0x1100);
    write_phdr_u64(&mut bytes, 2, 16, 0x2100);
    write_phdr_u64(&mut bytes, 2, 24, 0x2100);
    bytes[0x1100] = 0xa5;

    let envelope = validate(&bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    let source = envelope.segment_source(SegmentOrdinal::Second);
    assert_eq!(source.file_offset(), 0x1100);
    assert_eq!(source.bytes()[0], 0xa5);
    assert_range(
        envelope
            .materialization()
            .zero_phase()
            .mapping(SegmentOrdinal::Second)
            .destination(),
        0x2000,
        0x1000,
    );
    assert_range(
        envelope
            .materialization()
            .copy_phase()
            .segment(SegmentOrdinal::Second)
            .destination(),
        0x2100,
        0x100,
    );
    let zero_fill = envelope.segment_zero_fill(SegmentOrdinal::Second);
    assert_range(zero_fill.mapping_prefix().destination(), 0x2000, 0x100);
    assert_range(zero_fill.memory_suffix().destination(), 0x2200, 0);
    assert_range(zero_fill.mapping_tail().destination(), 0x2200, 0xe00);
}

#[test]
fn validated_envelopes_do_not_substitute_sources_between_objects() {
    let mut first_bytes = fixture();
    let mut second_bytes = fixture();
    first_bytes[0x1000] = 0xaa;
    second_bytes[0x1000] = 0xbb;
    first_bytes[0x214] = 0x80;
    second_bytes[0x214] = 0x81;

    let first = validate(&first_bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    let second = validate(&second_bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    let first_source = first.segment_source(SegmentOrdinal::Second);
    let second_source = second.segment_source(SegmentOrdinal::Second);
    assert_eq!(first_source.bytes()[0], 0xaa);
    assert_eq!(second_source.bytes()[0], 0xbb);
    assert_eq!(
        first_source.bytes().as_ptr(),
        first_bytes[0x1000..].as_ptr()
    );
    assert_eq!(
        second_source.bytes().as_ptr(),
        second_bytes[0x1000..].as_ptr()
    );
    assert_ne!(
        first_source.bytes().as_ptr(),
        second_source.bytes().as_ptr()
    );
    assert_eq!(first.metadata_descriptor(), &[0x80]);
    assert_eq!(second.metadata_descriptor(), &[0x81]);
    assert_eq!(
        first
            .materialization()
            .copy_phase()
            .segment(SegmentOrdinal::Second)
            .source()
            .bytes()
            .as_ptr(),
        first_source.bytes().as_ptr()
    );
}

#[test]
fn validated_path_rejects_an_out_of_bounds_segment_source() {
    let mut bytes = fixture();
    write_phdr_u64(&mut bytes, 2, 8, SHOFF as u64);
    assert!(matches!(
        validate(&bytes, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::FileRangeOutOfBounds { index: 2 })
    ));
}

#[test]
fn canonicalizes_program_header_order() {
    let canonical = fixture();
    let mut permuted = canonical.clone();
    swap_program_headers(&mut permuted, 1, 3);
    assert_eq!(
        plan(&canonical, AdmittedProfile::Gfx942XnackOffCov6),
        plan(&permuted, AdmittedProfile::Gfx942XnackOffCov6)
    );
}

#[test]
fn rejects_every_truncated_prefix() {
    let bytes = fixture();
    for end in 0..bytes.len() {
        assert!(
            plan(&bytes[..end], AdmittedProfile::Gfx942XnackOffCov6).is_err(),
            "accepted prefix of length {end}"
        );
    }
}

#[test]
fn rejects_wrong_identity_and_unknown_target_features() {
    for (offset, value) in [(4, 1), (5, 2), (6, 2), (7, 0), (8, 3), (9, 1)] {
        let mut bytes = fixture();
        bytes[offset] = value;
        assert!(plan(&bytes, AdmittedProfile::Gfx942XnackOffCov6).is_err());
    }

    let mut wrong_type = fixture();
    write_u16(&mut wrong_type, 16, 2);
    assert!(matches!(
        plan(&wrong_type, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedElfType(2))
    ));

    let mut wrong_machine = fixture();
    write_u16(&mut wrong_machine, 18, 62);
    assert!(matches!(
        plan(&wrong_machine, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedMachine(62))
    ));

    let mut unknown_flags = fixture();
    write_u32(&mut unknown_flags, 48, 0x0100_064c);
    assert!(matches!(
        plan(&unknown_flags, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedElfFlags(0x0100_064c))
    ));
}

#[test]
fn rejects_table_and_range_overflow() {
    let mut table = fixture();
    write_u64(&mut table, 32, u64::MAX - 8);
    assert!(matches!(
        plan(&table, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::TableRangeOverflow("program-header table"))
    ));

    let mut file = fixture();
    write_phdr_u64(&mut file, 1, 8, u64::MAX - 7);
    write_phdr_u64(&mut file, 1, 16, 0xff8);
    write_phdr_u64(&mut file, 1, 24, 0xff8);
    assert!(matches!(
        plan(&file, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::FileRangeOverflow { index: 1 })
    ));

    let mut memory = fixture();
    write_phdr_u64(&mut memory, 2, 8, 0x1f80);
    write_phdr_u64(&mut memory, 2, 16, u64::MAX - 0x7f);
    write_phdr_u64(&mut memory, 2, 24, u64::MAX - 0x7f);
    assert!(matches!(
        plan(&memory, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::MemoryRangeOverflow { index: 2 })
    ));
}

#[test]
fn rejects_bad_alignment_and_congruence() {
    let mut non_power_of_two = fixture();
    write_phdr_u64(&mut non_power_of_two, 2, 48, 3);
    assert!(matches!(
        plan(&non_power_of_two, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::InvalidProgramHeaderAlignment {
            index: 2,
            alignment: 3
        })
    ));

    let mut incongruent = fixture();
    write_phdr_u64(&mut incongruent, 2, 16, 0x2100);
    write_phdr_u64(&mut incongruent, 2, 24, 0x2100);
    assert!(matches!(
        plan(&incongruent, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::MisalignedProgramHeader { index: 2 })
    ));
}

#[test]
fn rejects_descriptor_file_to_virtual_translation_mismatch() {
    let mut note_virtual_address = fixture();
    write_phdr_u64(&mut note_virtual_address, 7, 16, 0x204);
    write_phdr_u64(&mut note_virtual_address, 7, 24, 0x204);
    assert_eq!(
        plan(&note_virtual_address, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::DescriptorLoadMappingMismatch { index: 7 })
    );

    let mut dynamic_virtual_address = fixture();
    write_phdr_u64(&mut dynamic_virtual_address, 4, 16, 0x4008);
    write_phdr_u64(&mut dynamic_virtual_address, 4, 24, 0x4008);
    assert_eq!(
        plan(
            &dynamic_virtual_address,
            AdmittedProfile::Gfx942XnackOffCov6
        ),
        Err(PlanError::DescriptorLoadMappingMismatch { index: 4 })
    );

    let mut dynamic_file_offset = fixture();
    write_phdr_u64(&mut dynamic_file_offset, 4, 8, 0x2008);
    assert_eq!(
        plan(&dynamic_file_offset, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::DescriptorLoadMappingMismatch { index: 4 })
    );
}

#[test]
fn rejects_file_memory_and_page_mapping_overlap() {
    let mut file_overlap = fixture();
    write_phdr_u64(&mut file_overlap, 2, 8, 0x200);
    write_phdr_u64(&mut file_overlap, 2, 16, 0x2200);
    write_phdr_u64(&mut file_overlap, 2, 24, 0x2200);
    assert_eq!(
        plan(&file_overlap, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::LoadFileOverlap)
    );

    let mut memory_overlap = fixture();
    write_phdr_u64(&mut memory_overlap, 2, 16, 0);
    write_phdr_u64(&mut memory_overlap, 2, 24, 0);
    assert_eq!(
        plan(&memory_overlap, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::LoadMemoryOverlap)
    );

    let mut mapping_overlap = fixture();
    write_phdr_u64(&mut mapping_overlap, 2, 8, 0x1900);
    write_phdr_u64(&mut mapping_overlap, 2, 16, 0x900);
    write_phdr_u64(&mut mapping_overlap, 2, 24, 0x900);
    assert_eq!(
        plan(&mapping_overlap, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::LoadMappingOverlap)
    );
}

#[test]
fn rejects_unknown_or_writable_executable_load_flags() {
    let mut unknown = fixture();
    write_phdr_u32(&mut unknown, 2, 4, 0xc);
    assert!(matches!(
        plan(&unknown, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedProgramHeaderFlags {
            index: 2,
            flags: 0xc
        })
    ));

    let mut writable_executable = fixture();
    write_phdr_u32(&mut writable_executable, 2, 4, 7);
    assert!(matches!(
        plan(&writable_executable, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::WritableExecutableSegment { index: 2 })
    ));
}

#[test]
fn rejects_unknown_program_and_section_features() {
    let mut program = fixture();
    write_phdr_u32(&mut program, 6, 0, 0xdead_beef);
    assert!(matches!(
        plan(&program, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedProgramHeaderType {
            index: 6,
            header_type: 0xdead_beef
        })
    ));

    let mut section = fixture();
    write_u32(&mut section, SHOFF + 4, 0x7000_0001);
    assert!(matches!(
        plan(&section, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedSectionType {
            index: 0,
            section_type: 0x7000_0001
        })
    ));

    let mut relocation = fixture();
    write_u32(&mut relocation, SHOFF + 4, 4);
    assert!(matches!(
        plan(&relocation, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedRelocationSection {
            index: 0,
            section_type: 4
        })
    ));
}

#[test]
fn rejects_unknown_or_malformed_metadata_notes() {
    let mut owner = fixture();
    owner[0x20c] = b'X';
    assert_eq!(
        plan(&owner, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedNote)
    );

    let mut note_type = fixture();
    write_u32(&mut note_type, 0x208, 33);
    assert_eq!(
        plan(&note_type, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedNote)
    );

    let mut descriptor_size = fixture();
    write_u32(&mut descriptor_size, 0x204, 4 * 1024 * 1024 + 1);
    assert_eq!(
        plan(&descriptor_size, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::MetadataNoteTooLarge)
    );
}

#[test]
fn rejects_relocation_and_unknown_dynamic_tags() {
    let mut relocation = fixture();
    write_u64(&mut relocation, 0x2000, 7);
    assert_eq!(
        plan(&relocation, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedRelocationTag(7))
    );

    let mut constructor = fixture();
    write_u64(&mut constructor, 0x2000, 12);
    assert_eq!(
        plan(&constructor, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedDynamicFeature(12))
    );

    let mut unknown = fixture();
    write_u64(&mut unknown, 0x2000, 0x1234_5678);
    assert_eq!(
        plan(&unknown, AdmittedProfile::Gfx942XnackOffCov6),
        Err(PlanError::UnsupportedDynamicTag(0x1234_5678))
    );
}

#[test]
#[ignore = "requires FE2O3_TEST_COV6 to name a pinned finalizer artifact"]
fn plans_a_real_pinned_finalizer_artifact() {
    let path = env::var("FE2O3_TEST_COV6").expect("set FE2O3_TEST_COV6");
    let bytes = fs::read(path).unwrap();
    let plan = plan(&bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    assert_eq!(plan.segments().len(), 3);
    assert!(plan.metadata_note().byte_len() > 0);
    let envelope = validate(&bytes, AdmittedProfile::Gfx942XnackOffCov6).unwrap();
    assert_eq!(*envelope.plan(), plan);
    assert_eq!(
        envelope.metadata_descriptor().len() as u64,
        plan.metadata_note().byte_len()
    );
    for ordinal in [
        SegmentOrdinal::First,
        SegmentOrdinal::Second,
        SegmentOrdinal::Third,
    ] {
        let source = envelope.segment_source(ordinal);
        let copy = envelope.materialization().copy_phase().segment(ordinal);
        assert_eq!(copy.source().bytes().as_ptr(), source.bytes().as_ptr());
        assert_eq!(copy.destination().byte_len(), source.byte_len());
    }
}

fn fixture() -> Vec<u8> {
    let mut bytes = vec![0u8; SHOFF + 64];
    bytes[..16].copy_from_slice(b"\x7fELF\x02\x01\x01\x40\x04\0\0\0\0\0\0\0");
    write_u16(&mut bytes, 16, 3);
    write_u16(&mut bytes, 18, 224);
    write_u32(&mut bytes, 20, 1);
    write_u64(&mut bytes, 24, 0);
    write_u64(&mut bytes, 32, PHOFF as u64);
    write_u64(&mut bytes, 40, SHOFF as u64);
    write_u32(&mut bytes, 48, 0x64c);
    write_u16(&mut bytes, 52, 64);
    write_u16(&mut bytes, 54, PHENT as u16);
    write_u16(&mut bytes, 56, PHNUM as u16);
    write_u16(&mut bytes, 58, 64);
    write_u16(&mut bytes, 60, 1);
    write_u16(&mut bytes, 62, 0);

    phdr(&mut bytes, 0, PT_PHDR, 4, 0x40, 0x40, 0x1c0, 0x1c0, 8);
    phdr(&mut bytes, 1, PT_LOAD, 4, 0, 0, 0x300, 0x300, 0x1000);
    phdr(
        &mut bytes, 2, PT_LOAD, 5, 0x1000, 0x2000, 0x100, 0x100, 0x1000,
    );
    phdr(
        &mut bytes, 3, PT_LOAD, 6, 0x2000, 0x4000, 0x80, 0x1000, 0x1000,
    );
    phdr(&mut bytes, 4, PT_DYNAMIC, 6, 0x2000, 0x4000, 0x70, 0x70, 8);
    phdr(
        &mut bytes,
        5,
        PT_GNU_RELRO,
        4,
        0x2000,
        0x4000,
        0x80,
        0x1000,
        1,
    );
    phdr(&mut bytes, 6, PT_GNU_STACK, 6, 0, 0, 0, 0, 0);
    phdr(&mut bytes, 7, PT_NOTE, 4, 0x200, 0x200, 0x18, 0x18, 4);

    write_u32(&mut bytes, 0x200, 7);
    write_u32(&mut bytes, 0x204, 1);
    write_u32(&mut bytes, 0x208, 32);
    bytes[0x20c..0x213].copy_from_slice(b"AMDGPU\0");
    bytes[0x214] = 0x80;

    for (index, (tag, value)) in [
        (6, 0x220),
        (11, 24),
        (5, 0x240),
        (10, 16),
        (0x6fff_fef5, 0x260),
        (4, 0x280),
        (0, 0),
    ]
    .into_iter()
    .enumerate()
    {
        write_u64(&mut bytes, 0x2000 + index * 16, tag);
        write_u64(&mut bytes, 0x2008 + index * 16, value);
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
fn phdr(
    bytes: &mut [u8],
    index: usize,
    header_type: u32,
    flags: u32,
    offset: u64,
    virtual_address: u64,
    file_size: u64,
    memory_size: u64,
    alignment: u64,
) {
    let base = PHOFF + index * PHENT;
    write_u32(bytes, base, header_type);
    write_u32(bytes, base + 4, flags);
    write_u64(bytes, base + 8, offset);
    write_u64(bytes, base + 16, virtual_address);
    write_u64(bytes, base + 24, virtual_address);
    write_u64(bytes, base + 32, file_size);
    write_u64(bytes, base + 40, memory_size);
    write_u64(bytes, base + 48, alignment);
}

fn swap_program_headers(bytes: &mut [u8], first: usize, second: usize) {
    let first = PHOFF + first * PHENT;
    let second = PHOFF + second * PHENT;
    for offset in 0..PHENT {
        bytes.swap(first + offset, second + offset);
    }
}

fn assert_range(range: ImageRange, offset_from_image_start: u64, byte_len: u64) {
    assert_eq!(range.offset_from_image_start(), offset_from_image_start);
    assert_eq!(range.byte_len(), byte_len);
}

fn write_phdr_u32(bytes: &mut [u8], index: usize, field: usize, value: u32) {
    write_u32(bytes, PHOFF + index * PHENT + field, value);
}

fn write_phdr_u64(bytes: &mut [u8], index: usize, field: usize, value: u64) {
    write_u64(bytes, PHOFF + index * PHENT + field, value);
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

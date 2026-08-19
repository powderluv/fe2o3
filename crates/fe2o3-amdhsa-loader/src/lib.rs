#![no_std]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

const ELF_HEADER_BYTES: usize = 64;
const PROGRAM_HEADER_BYTES: usize = 56;
const SECTION_HEADER_BYTES: usize = 64;
const DYNAMIC_ENTRY_BYTES: u64 = 16;
const ELF_CLASS_64: u8 = 2;
const ELF_DATA_LITTLE_ENDIAN: u8 = 1;
const ELF_VERSION_CURRENT: u8 = 1;
const ELF_OSABI_AMDGPU_HSA: u8 = 64;
const ELF_ABI_VERSION_COV6: u8 = 4;
const ELF_TYPE_DYNAMIC: u16 = 3;
const ELF_MACHINE_AMDGPU: u16 = 224;
const ELF_FLAGS_GFX942_XNACK_OFF: u32 = 0x64c;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_NOTE: u32 = 4;
const PT_PHDR: u32 = 6;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PT_GNU_RELRO: u32 = 0x6474_e552;
const PF_EXECUTE: u32 = 1;
const PF_WRITE: u32 = 2;
const PF_READ: u32 = 4;
const PF_READ_EXECUTE: u32 = PF_READ | PF_EXECUTE;
const PF_READ_WRITE: u32 = PF_READ | PF_WRITE;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_HASH: u32 = 5;
const SHT_DYNAMIC: u32 = 6;
const SHT_NOTE: u32 = 7;
const SHT_NOBITS: u32 = 8;
const SHT_REL: u32 = 9;
const SHT_DYNSYM: u32 = 11;
const SHT_GNU_HASH: u32 = 0x6fff_fff6;

const DT_NULL: u64 = 0;
const DT_PLTRELSZ: u64 = 2;
const DT_HASH: u64 = 4;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_RELAENT: u64 = 9;
const DT_STRSZ: u64 = 10;
const DT_SYMENT: u64 = 11;
const DT_INIT: u64 = 12;
const DT_FINI: u64 = 13;
const DT_REL: u64 = 17;
const DT_RELSZ: u64 = 18;
const DT_RELENT: u64 = 19;
const DT_PLTREL: u64 = 20;
const DT_JMPREL: u64 = 23;
const DT_INIT_ARRAY: u64 = 25;
const DT_FINI_ARRAY: u64 = 26;
const DT_INIT_ARRAYSZ: u64 = 27;
const DT_FINI_ARRAYSZ: u64 = 28;
const DT_FLAGS: u64 = 30;
const DT_GNU_HASH: u64 = 0x6fff_fef5;
const DT_RELACOUNT: u64 = 0x6fff_fff9;
const DT_RELCOUNT: u64 = 0x6fff_fffa;
const DF_SYMBOLIC: u64 = 2;

const NT_AMDGPU_METADATA: u32 = 32;
const AMDGPU_NOTE_NAME: &[u8] = b"AMDGPU\0";

/// Stable name of this data-only parser/planner profile.
pub const LOADER_PROFILE_ID: &str = "fe2o3-amdhsa-cov6-gfx942-xnack-off-envelope-v1";
/// Largest byte slice accepted by the parser (64 MiB).
pub const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
/// Largest number of ELF program headers accepted before profile checks.
pub const MAX_PROGRAM_HEADERS: usize = 16;
/// Exact number of load segments in the admitted finalizer profile.
pub const LOAD_SEGMENT_COUNT: usize = 3;
/// Largest number of ELF section headers inspected for unsupported features.
pub const MAX_SECTION_HEADERS: usize = 256;
/// Largest AMDGPU metadata descriptor retained as a range (4 MiB).
pub const MAX_METADATA_BYTES: u64 = 4 * 1024 * 1024;
/// Exact load-segment alignment in the admitted finalizer profile.
pub const LOAD_ALIGNMENT: u64 = 4096;
/// Largest total virtual span covered by the admitted load plan (64 MiB).
pub const MAX_IMAGE_SPAN_BYTES: u64 = 64 * 1024 * 1024;

/// The only code-object envelope admitted by this foundation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmittedProfile {
    /// AMDHSA COV6 for `gfx942:xnack-`, with SRAM-ECC left unspecified.
    Gfx942XnackOffCov6,
}

impl AdmittedProfile {
    /// Returns the canonical target spelling represented by the ELF flags.
    pub const fn target(self) -> &'static str {
        match self {
            Self::Gfx942XnackOffCov6 => "gfx942:xnack-",
        }
    }

    /// Returns the exact admitted ELF `e_flags` word.
    pub const fn elf_flags(self) -> u32 {
        match self {
            Self::Gfx942XnackOffCov6 => ELF_FLAGS_GFX942_XNACK_OFF,
        }
    }
}

/// Final permissions requested by one canonical load segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentPermissions {
    /// Read-only, non-executable bytes.
    ReadOnly,
    /// Read-only executable bytes.
    ReadExecute,
    /// Read-write, non-executable bytes used by the finalized data segment.
    ReadWrite,
}

/// One checked, canonical `PT_LOAD` plan entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadSegment {
    file_offset: u64,
    file_size: u64,
    virtual_address: u64,
    memory_size: u64,
    mapping_address: u64,
    mapping_size: u64,
    permissions: SegmentPermissions,
}

impl LoadSegment {
    const EMPTY: Self = Self {
        file_offset: 0,
        file_size: 0,
        virtual_address: 0,
        memory_size: 0,
        mapping_address: 0,
        mapping_size: 0,
        permissions: SegmentPermissions::ReadOnly,
    };

    /// Byte offset of the file-backed prefix.
    pub const fn file_offset(self) -> u64 {
        self.file_offset
    }

    /// Number of bytes copied from the input.
    pub const fn file_size(self) -> u64 {
        self.file_size
    }

    /// Link-time virtual address of the first segment byte.
    pub const fn virtual_address(self) -> u64 {
        self.virtual_address
    }

    /// Total in-memory length, including the zero-filled suffix.
    pub const fn memory_size(self) -> u64 {
        self.memory_size
    }

    /// Number of zero bytes following the file-backed prefix.
    pub const fn zero_fill_size(self) -> u64 {
        self.memory_size - self.file_size
    }

    /// Page-rounded virtual address at which a later mapper starts.
    pub const fn mapping_address(self) -> u64 {
        self.mapping_address
    }

    /// Page-rounded mapping length required by this segment.
    pub const fn mapping_size(self) -> u64 {
        self.mapping_size
    }

    /// Offset of the first segment byte within the rounded mapping.
    pub const fn mapping_prefix_size(self) -> u64 {
        self.virtual_address - self.mapping_address
    }

    /// Checked final permissions for this segment.
    pub const fn permissions(self) -> SegmentPermissions {
        self.permissions
    }

    fn file_end(self) -> u64 {
        self.file_offset + self.file_size
    }

    fn memory_end(self) -> u64 {
        self.virtual_address + self.memory_size
    }

    fn mapping_end(self) -> u64 {
        self.mapping_address + self.mapping_size
    }
}

/// File range of the single admitted AMDGPU metadata descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetadataNote {
    file_offset: u64,
    byte_len: u64,
}

impl MetadataNote {
    /// Byte offset of the MessagePack descriptor in the input.
    pub const fn file_offset(self) -> u64 {
        self.file_offset
    }

    /// Descriptor length in bytes.
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// Fully checked, allocation-free load-envelope plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadPlan {
    profile: AdmittedProfile,
    input_len: u64,
    segments: [LoadSegment; LOAD_SEGMENT_COUNT],
    metadata_note: MetadataNote,
    image_start: u64,
    image_end: u64,
}

impl LoadPlan {
    /// Exact profile against which the input was admitted.
    pub const fn profile(self) -> AdmittedProfile {
        self.profile
    }

    /// Length of the byte slice validated by this plan.
    pub const fn input_len(self) -> u64 {
        self.input_len
    }

    /// Canonical load segments, sorted by virtual address.
    pub fn segments(&self) -> &[LoadSegment] {
        &self.segments
    }

    /// Range of the AMDGPU metadata note descriptor.
    pub const fn metadata_note(self) -> MetadataNote {
        self.metadata_note
    }

    /// Lowest page-rounded address in the plan.
    pub const fn image_start(self) -> u64 {
        self.image_start
    }

    /// Exclusive highest page-rounded address in the plan.
    pub const fn image_end(self) -> u64 {
        self.image_end
    }
}

/// Ordinal of a canonical load segment, in increasing virtual-address order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentOrdinal {
    First,
    Second,
    Third,
}

impl SegmentOrdinal {
    const fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
            Self::Third => 2,
        }
    }
}

/// Ordinal of an unmapped gap between adjacent canonical load mappings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterSegmentGapOrdinal {
    FirstToSecond,
    SecondToThird,
}

impl InterSegmentGapOrdinal {
    const fn index(self) -> usize {
        match self {
            Self::FirstToSecond => 0,
            Self::SecondToThird => 1,
        }
    }
}

/// A checked range relative to [`LoadPlan::image_start`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageRange {
    offset_from_image_start: u64,
    byte_len: u64,
}

impl ImageRange {
    /// Offset from the plan's lowest page-rounded mapping address.
    pub const fn offset_from_image_start(self) -> u64 {
        self.offset_from_image_start
    }

    /// Length of the range in bytes.
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// Exact file-backed bytes for one validated canonical segment.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct SegmentSource<'a> {
    ordinal: SegmentOrdinal,
    file_offset: u64,
    bytes: &'a [u8],
}

impl<'a> SegmentSource<'a> {
    /// Canonical segment to which these bytes belong.
    pub const fn ordinal(self) -> SegmentOrdinal {
        self.ordinal
    }

    /// Offset of the borrowed bytes in the validated input object.
    pub const fn file_offset(self) -> u64 {
        self.file_offset
    }

    /// Number of borrowed source bytes.
    pub const fn byte_len(self) -> u64 {
        self.bytes.len() as u64
    }

    /// Exact source bytes borrowed from the object passed to [`validate`].
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// Instruction to zero a checked image-relative range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeroInstruction {
    destination: ImageRange,
}

impl ZeroInstruction {
    /// Image-relative range that must be zeroed.
    pub const fn destination(self) -> ImageRange {
        self.destination
    }
}

/// Explicit zero-filled portions associated with one segment mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentZeroFill {
    mapping_prefix: ZeroInstruction,
    memory_suffix: ZeroInstruction,
    mapping_tail: ZeroInstruction,
}

impl SegmentZeroFill {
    /// Page prefix before the segment's virtual address.
    pub const fn mapping_prefix(self) -> ZeroInstruction {
        self.mapping_prefix
    }

    /// In-memory suffix after the file-backed bytes, including BSS.
    pub const fn memory_suffix(self) -> ZeroInstruction {
        self.memory_suffix
    }

    /// Page-rounding tail after the segment's in-memory range.
    pub const fn mapping_tail(self) -> ZeroInstruction {
        self.mapping_tail
    }
}

/// Instruction to copy one exact borrowed segment source into the image.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct CopyInstruction<'a> {
    source: SegmentSource<'a>,
    destination: ImageRange,
}

impl<'a> CopyInstruction<'a> {
    /// Exact bytes borrowed from the validated object.
    pub const fn source(self) -> SegmentSource<'a> {
        self.source
    }

    /// Prefix-adjusted image-relative copy destination.
    pub const fn destination(self) -> ImageRange {
        self.destination
    }
}

/// First materialization phase: zero the complete planned image span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeroPhase {
    mappings: [ZeroInstruction; LOAD_SEGMENT_COUNT],
    gaps: [ZeroInstruction; LOAD_SEGMENT_COUNT - 1],
    segments: [SegmentZeroFill; LOAD_SEGMENT_COUNT],
}

impl ZeroPhase {
    /// Complete page-rounded mapping to zero for a segment.
    pub const fn mapping(&self, ordinal: SegmentOrdinal) -> ZeroInstruction {
        self.mappings[ordinal.index()]
    }

    /// Complete gap to zero between two adjacent mappings.
    pub const fn inter_segment_gap(&self, ordinal: InterSegmentGapOrdinal) -> ZeroInstruction {
        self.gaps[ordinal.index()]
    }

    /// Descriptions of the zero-preserved portions of one segment mapping.
    pub const fn segment(&self, ordinal: SegmentOrdinal) -> SegmentZeroFill {
        self.segments[ordinal.index()]
    }
}

/// Second materialization phase: copy exact file-backed segment bytes.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct CopyPhase<'a> {
    segments: [CopyInstruction<'a>; LOAD_SEGMENT_COUNT],
}

impl<'a> CopyPhase<'a> {
    /// Copy instruction for a canonical segment.
    pub const fn segment(&self, ordinal: SegmentOrdinal) -> CopyInstruction<'a> {
        self.segments[ordinal.index()]
    }
}

/// Checked, descriptive materialization instructions for a validated object.
///
/// A later adapter must complete every [`ZeroPhase`] instruction before any
/// [`CopyPhase`] instruction. This type performs neither phase and grants no
/// allocation, mapping, copying, permission-transition, or execution authority.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct MaterializationPlan<'a> {
    image_len: u64,
    zero_phase: ZeroPhase,
    copy_phase: CopyPhase<'a>,
}

impl<'a> MaterializationPlan<'a> {
    /// Length of the complete image span described by the instructions.
    pub const fn image_len(&self) -> u64 {
        self.image_len
    }

    /// Complete-mapping and inter-mapping-gap zero instructions.
    pub const fn zero_phase(&self) -> &ZeroPhase {
        &self.zero_phase
    }

    /// Exact borrowed-source copy instructions.
    pub const fn copy_phase(&self) -> &CopyPhase<'a> {
        &self.copy_phase
    }
}

/// Fail-closed caller-buffer materialization error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterializationError {
    /// The caller-provided image does not exactly match the checked image span.
    DestinationLengthMismatch { expected: u64, actual: u64 },
    /// A private, previously checked copy range was not representable.
    CopyRangeUnavailable { ordinal: SegmentOrdinal },
    /// A private copy destination no longer matched its exact borrowed source.
    CopyLengthMismatch { ordinal: SegmentOrdinal },
}

/// A validated envelope permanently associated with its exact borrowed bytes.
///
/// This type has no public constructor. In particular, a caller cannot combine
/// an inert [`LoadPlan`] with a different byte slice to construct one:
///
/// ```compile_fail
/// use fe2o3_amdhsa_loader::{LoadPlan, ValidatedEnvelope};
///
/// fn substitute<'a>(plan: LoadPlan, unrelated: &'a [u8]) -> ValidatedEnvelope<'a> {
///     ValidatedEnvelope { bytes: unrelated, plan }
/// }
/// ```
///
/// The object retains validated data and can materialize its checked bytes into
/// an exact caller-provided, exclusively borrowed image. It is not loaded-code
/// or launch authority.
pub struct ValidatedEnvelope<'a> {
    bytes: &'a [u8],
    plan: LoadPlan,
    metadata_descriptor: &'a [u8],
    materialization: MaterializationPlan<'a>,
}

impl<'a> ValidatedEnvelope<'a> {
    /// Length of the exact input borrow retained by this envelope.
    pub const fn input_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Inert canonical plan derived from the retained input bytes.
    pub const fn plan(&self) -> &LoadPlan {
        &self.plan
    }

    /// Exact borrowed source bytes for a canonical segment.
    pub const fn segment_source(&self, ordinal: SegmentOrdinal) -> SegmentSource<'a> {
        self.materialization.copy_phase.segment(ordinal).source()
    }

    /// Zero-preserved portions associated with a canonical segment.
    pub const fn segment_zero_fill(&self, ordinal: SegmentOrdinal) -> SegmentZeroFill {
        self.materialization.zero_phase.segment(ordinal)
    }

    /// Exact metadata descriptor borrowed from the validated input object.
    ///
    /// The bytes are not decoded and convey no metadata or kernel authority.
    pub const fn metadata_descriptor(&self) -> &'a [u8] {
        self.metadata_descriptor
    }

    /// Checked zero-then-copy instructions tied to the retained input bytes.
    pub const fn materialization(&self) -> &MaterializationPlan<'a> {
        &self.materialization
    }

    /// Deterministically materializes the checked image into caller-provided,
    /// exclusively borrowed bytes.
    ///
    /// The destination must have exactly [`MaterializationPlan::image_len`]
    /// bytes. After all private copy ranges are rechecked, the complete image is
    /// zeroed once and the exact borrowed `PT_LOAD` sources are copied in
    /// canonical virtual-address order. Prefixes, BSS suffixes, mapping tails,
    /// and inter-mapping gaps are therefore left zero.
    ///
    /// This operation allocates nothing and grants no GPU mapping, permission,
    /// relocation, symbol, kernel, loaded-image, or execution authority.
    pub fn materialize_into(&self, destination: &mut [u8]) -> Result<(), MaterializationError> {
        let expected = self.materialization.image_len;
        if destination.len() as u64 != expected {
            return Err(MaterializationError::DestinationLengthMismatch {
                expected,
                actual: destination.len() as u64,
            });
        }

        let copy = self.materialization.copy_phase;
        let instructions = [
            copy.segment(SegmentOrdinal::First),
            copy.segment(SegmentOrdinal::Second),
            copy.segment(SegmentOrdinal::Third),
        ];
        let mut bounds = [(0usize, 0usize); LOAD_SEGMENT_COUNT];
        for (index, instruction) in instructions.iter().copied().enumerate() {
            bounds[index] = checked_copy_bounds(destination.len(), instruction)?;
        }

        destination.fill(0);
        for (instruction, (start, end)) in instructions.into_iter().zip(bounds) {
            destination[start..end].copy_from_slice(instruction.source.bytes);
        }
        Ok(())
    }
}

fn checked_copy_bounds(
    image_len: usize,
    instruction: CopyInstruction<'_>,
) -> Result<(usize, usize), MaterializationError> {
    let ordinal = instruction.source.ordinal;
    let start = usize::try_from(instruction.destination.offset_from_image_start)
        .map_err(|_| MaterializationError::CopyRangeUnavailable { ordinal })?;
    let byte_len = usize::try_from(instruction.destination.byte_len)
        .map_err(|_| MaterializationError::CopyRangeUnavailable { ordinal })?;
    let end = start
        .checked_add(byte_len)
        .filter(|end| *end <= image_len)
        .ok_or(MaterializationError::CopyRangeUnavailable { ordinal })?;
    if byte_len != instruction.source.bytes.len() {
        return Err(MaterializationError::CopyLengthMismatch { ordinal });
    }
    Ok((start, end))
}

/// Fail-closed parser or profile-admission error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    InputTooLarge,
    Truncated(&'static str),
    InvalidMagic,
    UnsupportedClass(u8),
    UnsupportedEndianness(u8),
    UnsupportedIdentVersion(u8),
    UnsupportedOsAbi(u8),
    UnsupportedAbiVersion(u8),
    UnsupportedIdentPadding,
    UnsupportedElfType(u16),
    UnsupportedMachine(u16),
    UnsupportedElfVersion(u32),
    UnsupportedEntryPoint(u64),
    UnsupportedElfFlags(u32),
    UnsupportedHeaderSize(u16),
    UnsupportedProgramHeaderEntrySize(u16),
    UnsupportedSectionHeaderEntrySize(u16),
    MissingProgramHeaders,
    TooManyProgramHeaders,
    TooManySectionHeaders,
    ExtendedSectionNumbering,
    InvalidSectionStringTableIndex,
    TableRangeOverflow(&'static str),
    TableOutOfBounds(&'static str),
    UnsupportedSectionType { index: usize, section_type: u32 },
    UnsupportedRelocationSection { index: usize, section_type: u32 },
    InvalidSectionAlignment { index: usize, alignment: u64 },
    SectionRangeOverflow { index: usize },
    SectionOutOfBounds { index: usize },
    UnsupportedProgramHeaderType { index: usize, header_type: u32 },
    DuplicateProgramHeader(u32),
    UnsupportedProgramHeaderFlags { index: usize, flags: u32 },
    WritableExecutableSegment { index: usize },
    InvalidProgramHeaderAlignment { index: usize, alignment: u64 },
    UnsupportedLoadAlignment { index: usize, alignment: u64 },
    MisalignedProgramHeader { index: usize },
    FileSizeExceedsMemorySize { index: usize },
    FileRangeOverflow { index: usize },
    FileRangeOutOfBounds { index: usize },
    MemoryRangeOverflow { index: usize },
    PhysicalAddressMismatch { index: usize },
    TooManyLoadSegments,
    UnsupportedProgramHeaderProfile,
    LoadFileOverlap,
    LoadMemoryOverlap,
    LoadMappingOverlap,
    MappingRangeOverflow,
    ImageSpanTooLarge,
    ProgramHeaderDescriptorMismatch,
    ProgramHeaderNotReadOnlyLoaded,
    DynamicSegmentNotWritableLoaded,
    DescriptorLoadMappingMismatch { index: usize },
    AmbiguousDescriptorLoadMapping { index: usize },
    RelroSegmentMismatch,
    InvalidStackDescriptor,
    MetadataNoteNotReadOnlyLoaded,
    InvalidNoteRecord,
    UnsupportedNote,
    MetadataNoteTooLarge,
    MissingMetadataNote,
    DuplicateMetadataNote,
    TooManyDynamicEntries,
    MissingDynamicTerminator,
    DuplicateDynamicTag(u64),
    MissingDynamicTag(u64),
    InvalidDynamicValue { tag: u64, value: u64 },
    UnsupportedRelocationTag(u64),
    UnsupportedDynamicFeature(u64),
    UnsupportedDynamicTag(u64),
    DynamicReferenceOutOfBounds(u64),
    ValidatedRangeUnavailable,
    MaterializationRangeInvalid,
    MaterializationZeroCoverageMismatch,
}

#[derive(Clone, Copy)]
struct Header {
    program_offset: u64,
    program_count: usize,
    section_offset: u64,
    section_count: usize,
    section_string_index: usize,
}

#[derive(Clone, Copy)]
struct ProgramHeader {
    index: usize,
    header_type: u32,
    flags: u32,
    offset: u64,
    virtual_address: u64,
    physical_address: u64,
    file_size: u64,
    memory_size: u64,
    alignment: u64,
}

/// Validates and canonically plans the admitted AMDHSA COV6 load envelope.
pub fn plan(bytes: &[u8], profile: AdmittedProfile) -> Result<LoadPlan, PlanError> {
    let header = validate_header(bytes, profile)?;
    validate_sections(bytes, header)?;
    let input_len = bytes.len() as u64;

    let mut segments = [LoadSegment::EMPTY; LOAD_SEGMENT_COUNT];
    let mut load_count = 0usize;
    let mut program_descriptor = None;
    let mut dynamic = None;
    let mut note = None;
    let mut stack = None;
    let mut relro = None;

    for index in 0..header.program_count {
        let raw = read_program_header(bytes, header.program_offset, index)?;
        validate_program_header_range(raw, input_len)?;
        match raw.header_type {
            PT_LOAD => {
                if load_count == LOAD_SEGMENT_COUNT {
                    return Err(PlanError::TooManyLoadSegments);
                }
                segments[load_count] = plan_load_segment(raw)?;
                load_count += 1;
            }
            PT_PHDR => set_unique(&mut program_descriptor, raw)?,
            PT_DYNAMIC => set_unique(&mut dynamic, raw)?,
            PT_NOTE => set_unique(&mut note, raw)?,
            PT_GNU_STACK => set_unique(&mut stack, raw)?,
            PT_GNU_RELRO => set_unique(&mut relro, raw)?,
            header_type => {
                return Err(PlanError::UnsupportedProgramHeaderType { index, header_type });
            }
        }
    }

    let (Some(program_descriptor), Some(dynamic), Some(note), Some(stack), Some(relro)) =
        (program_descriptor, dynamic, note, stack, relro)
    else {
        return Err(PlanError::UnsupportedProgramHeaderProfile);
    };
    if load_count != LOAD_SEGMENT_COUNT {
        return Err(PlanError::UnsupportedProgramHeaderProfile);
    }

    segments.sort_unstable_by_key(|segment| segment.virtual_address);
    validate_load_profile(&segments)?;
    validate_program_descriptor(program_descriptor, header, &segments)?;
    validate_dynamic_descriptor(dynamic, &segments)?;
    validate_relro_descriptor(relro, &segments)?;
    validate_stack_descriptor(stack)?;
    validate_note_descriptor(note, &segments)?;
    let metadata_note = parse_metadata_note(bytes, note)?;
    validate_dynamic_entries(bytes, dynamic, &segments)?;

    let image_start = segments[0].mapping_address;
    let image_end = segments[LOAD_SEGMENT_COUNT - 1].mapping_end();
    let span = image_end
        .checked_sub(image_start)
        .ok_or(PlanError::MappingRangeOverflow)?;
    if span > MAX_IMAGE_SPAN_BYTES {
        return Err(PlanError::ImageSpanTooLarge);
    }

    Ok(LoadPlan {
        profile,
        input_len,
        segments,
        metadata_note,
        image_start,
        image_end,
    })
}

/// Validates an object and binds its canonical plan to the exact input borrow.
///
/// Unlike [`plan`], the returned envelope retains the source association needed
/// to describe checked zero-then-copy materialization. It still performs no
/// allocation, copying, mapping, permission transition, or execution.
pub fn validate<'a>(
    bytes: &'a [u8],
    profile: AdmittedProfile,
) -> Result<ValidatedEnvelope<'a>, PlanError> {
    let plan = plan(bytes, profile)?;
    let metadata = plan.metadata_note;
    let metadata_descriptor = validated_slice(bytes, metadata.file_offset, metadata.byte_len)?;
    let materialization = build_materialization(bytes, plan)?;

    Ok(ValidatedEnvelope {
        bytes,
        plan,
        metadata_descriptor,
        materialization,
    })
}

fn build_materialization<'a>(
    bytes: &'a [u8],
    plan: LoadPlan,
) -> Result<MaterializationPlan<'a>, PlanError> {
    let first = build_segment_instructions(bytes, plan, SegmentOrdinal::First, plan.segments[0])?;
    let second = build_segment_instructions(bytes, plan, SegmentOrdinal::Second, plan.segments[1])?;
    let third = build_segment_instructions(bytes, plan, SegmentOrdinal::Third, plan.segments[2])?;
    let mappings = [first.0, second.0, third.0];
    let gaps = [
        build_inter_segment_gap(plan, plan.segments[0], plan.segments[1])?,
        build_inter_segment_gap(plan, plan.segments[1], plan.segments[2])?,
    ];
    let image_len = plan
        .image_end
        .checked_sub(plan.image_start)
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    validate_zero_coverage(mappings, gaps, image_len)?;

    Ok(MaterializationPlan {
        image_len,
        zero_phase: ZeroPhase {
            mappings,
            gaps,
            segments: [first.1, second.1, third.1],
        },
        copy_phase: CopyPhase {
            segments: [first.2, second.2, third.2],
        },
    })
}

fn build_segment_instructions<'a>(
    bytes: &'a [u8],
    plan: LoadPlan,
    ordinal: SegmentOrdinal,
    segment: LoadSegment,
) -> Result<(ZeroInstruction, SegmentZeroFill, CopyInstruction<'a>), PlanError> {
    let mapping = ZeroInstruction {
        destination: checked_image_range(plan, segment.mapping_address, segment.mapping_size)?,
    };
    let prefix = ZeroInstruction {
        destination: checked_image_range(
            plan,
            segment.mapping_address,
            segment.mapping_prefix_size(),
        )?,
    };
    let memory_suffix_start = segment
        .virtual_address
        .checked_add(segment.file_size)
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    let memory_suffix = ZeroInstruction {
        destination: checked_image_range(plan, memory_suffix_start, segment.zero_fill_size())?,
    };
    let mapping_tail_len = segment
        .mapping_end()
        .checked_sub(segment.memory_end())
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    let mapping_tail = ZeroInstruction {
        destination: checked_image_range(plan, segment.memory_end(), mapping_tail_len)?,
    };

    let source_bytes = validated_slice(bytes, segment.file_offset, segment.file_size)?;
    let source = SegmentSource {
        ordinal,
        file_offset: segment.file_offset,
        bytes: source_bytes,
    };
    let destination = checked_image_range(plan, segment.virtual_address, segment.file_size)?;
    if destination.byte_len != source.byte_len() {
        return Err(PlanError::MaterializationRangeInvalid);
    }

    Ok((
        mapping,
        SegmentZeroFill {
            mapping_prefix: prefix,
            memory_suffix,
            mapping_tail,
        },
        CopyInstruction {
            source,
            destination,
        },
    ))
}

fn build_inter_segment_gap(
    plan: LoadPlan,
    first: LoadSegment,
    second: LoadSegment,
) -> Result<ZeroInstruction, PlanError> {
    let start = first.mapping_end();
    let byte_len = second
        .mapping_address
        .checked_sub(start)
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    Ok(ZeroInstruction {
        destination: checked_image_range(plan, start, byte_len)?,
    })
}

fn checked_image_range(
    plan: LoadPlan,
    absolute_start: u64,
    byte_len: u64,
) -> Result<ImageRange, PlanError> {
    let image_len = plan
        .image_end
        .checked_sub(plan.image_start)
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    let offset_from_image_start = absolute_start
        .checked_sub(plan.image_start)
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    let end = offset_from_image_start
        .checked_add(byte_len)
        .ok_or(PlanError::MaterializationRangeInvalid)?;
    if end > image_len {
        return Err(PlanError::MaterializationRangeInvalid);
    }
    Ok(ImageRange {
        offset_from_image_start,
        byte_len,
    })
}

fn validate_zero_coverage(
    mappings: [ZeroInstruction; LOAD_SEGMENT_COUNT],
    gaps: [ZeroInstruction; LOAD_SEGMENT_COUNT - 1],
    image_len: u64,
) -> Result<(), PlanError> {
    let ordered = [mappings[0], gaps[0], mappings[1], gaps[1], mappings[2]];
    let mut expected_start = 0;
    for instruction in ordered {
        let range = instruction.destination;
        if range.offset_from_image_start != expected_start {
            return Err(PlanError::MaterializationZeroCoverageMismatch);
        }
        expected_start = expected_start
            .checked_add(range.byte_len)
            .ok_or(PlanError::MaterializationZeroCoverageMismatch)?;
    }
    if expected_start != image_len {
        return Err(PlanError::MaterializationZeroCoverageMismatch);
    }
    Ok(())
}

fn validated_slice(bytes: &[u8], offset: u64, byte_len: u64) -> Result<&[u8], PlanError> {
    let end = offset
        .checked_add(byte_len)
        .ok_or(PlanError::ValidatedRangeUnavailable)?;
    let start = usize::try_from(offset).map_err(|_| PlanError::ValidatedRangeUnavailable)?;
    let end = usize::try_from(end).map_err(|_| PlanError::ValidatedRangeUnavailable)?;
    bytes
        .get(start..end)
        .ok_or(PlanError::ValidatedRangeUnavailable)
}

fn validate_header(bytes: &[u8], profile: AdmittedProfile) -> Result<Header, PlanError> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(PlanError::InputTooLarge);
    }
    if bytes.len() < ELF_HEADER_BYTES {
        return Err(PlanError::Truncated("ELF header"));
    }
    if bytes[..4] != *b"\x7fELF" {
        return Err(PlanError::InvalidMagic);
    }
    check_byte(bytes[4], ELF_CLASS_64, PlanError::UnsupportedClass)?;
    check_byte(
        bytes[5],
        ELF_DATA_LITTLE_ENDIAN,
        PlanError::UnsupportedEndianness,
    )?;
    check_byte(
        bytes[6],
        ELF_VERSION_CURRENT,
        PlanError::UnsupportedIdentVersion,
    )?;
    check_byte(bytes[7], ELF_OSABI_AMDGPU_HSA, PlanError::UnsupportedOsAbi)?;
    check_byte(
        bytes[8],
        ELF_ABI_VERSION_COV6,
        PlanError::UnsupportedAbiVersion,
    )?;
    if bytes[9..16].iter().any(|byte| *byte != 0) {
        return Err(PlanError::UnsupportedIdentPadding);
    }

    let elf_type = read_u16(bytes, 16, "ELF type")?;
    if elf_type != ELF_TYPE_DYNAMIC {
        return Err(PlanError::UnsupportedElfType(elf_type));
    }
    let machine = read_u16(bytes, 18, "ELF machine")?;
    if machine != ELF_MACHINE_AMDGPU {
        return Err(PlanError::UnsupportedMachine(machine));
    }
    let version = read_u32(bytes, 20, "ELF version")?;
    if version != u32::from(ELF_VERSION_CURRENT) {
        return Err(PlanError::UnsupportedElfVersion(version));
    }
    let entry = read_u64(bytes, 24, "ELF entry point")?;
    if entry != 0 {
        return Err(PlanError::UnsupportedEntryPoint(entry));
    }
    let flags = read_u32(bytes, 48, "ELF flags")?;
    if flags != profile.elf_flags() {
        return Err(PlanError::UnsupportedElfFlags(flags));
    }
    let header_size = read_u16(bytes, 52, "ELF header size")?;
    if usize::from(header_size) != ELF_HEADER_BYTES {
        return Err(PlanError::UnsupportedHeaderSize(header_size));
    }

    let program_offset = read_u64(bytes, 32, "program-header offset")?;
    let program_entry_size = read_u16(bytes, 54, "program-header entry size")?;
    let program_count = usize::from(read_u16(bytes, 56, "program-header count")?);
    if program_count == 0 {
        return Err(PlanError::MissingProgramHeaders);
    }
    if program_count > MAX_PROGRAM_HEADERS {
        return Err(PlanError::TooManyProgramHeaders);
    }
    if usize::from(program_entry_size) != PROGRAM_HEADER_BYTES {
        return Err(PlanError::UnsupportedProgramHeaderEntrySize(
            program_entry_size,
        ));
    }
    validate_table_range(
        bytes.len() as u64,
        program_offset,
        PROGRAM_HEADER_BYTES as u64,
        program_count,
        "program-header table",
    )?;

    let section_offset = read_u64(bytes, 40, "section-header offset")?;
    let section_entry_size = read_u16(bytes, 58, "section-header entry size")?;
    let section_count = usize::from(read_u16(bytes, 60, "section-header count")?);
    let section_string_index = usize::from(read_u16(bytes, 62, "section-string index")?);
    if section_count > MAX_SECTION_HEADERS {
        return Err(PlanError::TooManySectionHeaders);
    }
    if section_count == 0 {
        if section_offset != 0 || section_entry_size != 0 || section_string_index != 0 {
            return Err(PlanError::ExtendedSectionNumbering);
        }
    } else {
        if usize::from(section_entry_size) != SECTION_HEADER_BYTES {
            return Err(PlanError::UnsupportedSectionHeaderEntrySize(
                section_entry_size,
            ));
        }
        validate_table_range(
            bytes.len() as u64,
            section_offset,
            SECTION_HEADER_BYTES as u64,
            section_count,
            "section-header table",
        )?;
        if section_string_index >= section_count {
            return Err(PlanError::InvalidSectionStringTableIndex);
        }
    }

    Ok(Header {
        program_offset,
        program_count,
        section_offset,
        section_count,
        section_string_index,
    })
}

fn check_byte(found: u8, expected: u8, error: fn(u8) -> PlanError) -> Result<(), PlanError> {
    if found == expected {
        Ok(())
    } else {
        Err(error(found))
    }
}

fn validate_table_range(
    input_len: u64,
    offset: u64,
    entry_size: u64,
    count: usize,
    context: &'static str,
) -> Result<(), PlanError> {
    let table_size = entry_size
        .checked_mul(count as u64)
        .ok_or(PlanError::TableRangeOverflow(context))?;
    let end = offset
        .checked_add(table_size)
        .ok_or(PlanError::TableRangeOverflow(context))?;
    if end > input_len {
        return Err(PlanError::TableOutOfBounds(context));
    }
    Ok(())
}

fn validate_sections(bytes: &[u8], header: Header) -> Result<(), PlanError> {
    if header.section_count == 0 {
        return Ok(());
    }
    let _ = header.section_string_index;
    for index in 0..header.section_count {
        let base = table_entry_offset(
            header.section_offset,
            SECTION_HEADER_BYTES,
            index,
            "section header",
        )?;
        let section_type = read_u32(bytes, base + 4, "section type")?;
        match section_type {
            SHT_RELA | SHT_REL => {
                return Err(PlanError::UnsupportedRelocationSection {
                    index,
                    section_type,
                });
            }
            SHT_NULL | SHT_PROGBITS | SHT_SYMTAB | SHT_STRTAB | SHT_HASH | SHT_DYNAMIC
            | SHT_NOTE | SHT_NOBITS | SHT_DYNSYM | SHT_GNU_HASH => {}
            section_type => {
                return Err(PlanError::UnsupportedSectionType {
                    index,
                    section_type,
                });
            }
        }
        if index == 0 && section_type != SHT_NULL {
            return Err(PlanError::UnsupportedSectionType {
                index,
                section_type,
            });
        }
        let alignment = read_u64(bytes, base + 48, "section alignment")?;
        if !valid_alignment(alignment) {
            return Err(PlanError::InvalidSectionAlignment { index, alignment });
        }
        if section_type != SHT_NOBITS {
            let offset = read_u64(bytes, base + 24, "section offset")?;
            let size = read_u64(bytes, base + 32, "section size")?;
            let end = offset
                .checked_add(size)
                .ok_or(PlanError::SectionRangeOverflow { index })?;
            if end > bytes.len() as u64 {
                return Err(PlanError::SectionOutOfBounds { index });
            }
        }
    }
    Ok(())
}

fn read_program_header(
    bytes: &[u8],
    table_offset: u64,
    index: usize,
) -> Result<ProgramHeader, PlanError> {
    let base = table_entry_offset(table_offset, PROGRAM_HEADER_BYTES, index, "program header")?;
    Ok(ProgramHeader {
        index,
        header_type: read_u32(bytes, base, "program-header type")?,
        flags: read_u32(bytes, base + 4, "program-header flags")?,
        offset: read_u64(bytes, base + 8, "program-header offset")?,
        virtual_address: read_u64(bytes, base + 16, "program-header virtual address")?,
        physical_address: read_u64(bytes, base + 24, "program-header physical address")?,
        file_size: read_u64(bytes, base + 32, "program-header file size")?,
        memory_size: read_u64(bytes, base + 40, "program-header memory size")?,
        alignment: read_u64(bytes, base + 48, "program-header alignment")?,
    })
}

fn validate_program_header_range(raw: ProgramHeader, input_len: u64) -> Result<(), PlanError> {
    if !valid_alignment(raw.alignment) {
        return Err(PlanError::InvalidProgramHeaderAlignment {
            index: raw.index,
            alignment: raw.alignment,
        });
    }
    if raw.alignment > 1 && raw.offset % raw.alignment != raw.virtual_address % raw.alignment {
        return Err(PlanError::MisalignedProgramHeader { index: raw.index });
    }
    if raw.file_size > raw.memory_size {
        return Err(PlanError::FileSizeExceedsMemorySize { index: raw.index });
    }
    let file_end = raw
        .offset
        .checked_add(raw.file_size)
        .ok_or(PlanError::FileRangeOverflow { index: raw.index })?;
    if file_end > input_len {
        return Err(PlanError::FileRangeOutOfBounds { index: raw.index });
    }
    raw.virtual_address
        .checked_add(raw.memory_size)
        .ok_or(PlanError::MemoryRangeOverflow { index: raw.index })?;
    if raw.header_type != PT_GNU_STACK && raw.physical_address != raw.virtual_address {
        return Err(PlanError::PhysicalAddressMismatch { index: raw.index });
    }
    Ok(())
}

fn valid_alignment(alignment: u64) -> bool {
    alignment == 0 || alignment.is_power_of_two()
}

fn set_unique(slot: &mut Option<ProgramHeader>, raw: ProgramHeader) -> Result<(), PlanError> {
    if slot.replace(raw).is_some() {
        Err(PlanError::DuplicateProgramHeader(raw.header_type))
    } else {
        Ok(())
    }
}

fn plan_load_segment(raw: ProgramHeader) -> Result<LoadSegment, PlanError> {
    if raw.flags & PF_EXECUTE != 0 && raw.flags & PF_WRITE != 0 {
        return Err(PlanError::WritableExecutableSegment { index: raw.index });
    }
    let permissions = match raw.flags {
        PF_READ => SegmentPermissions::ReadOnly,
        PF_READ_EXECUTE => SegmentPermissions::ReadExecute,
        PF_READ_WRITE => SegmentPermissions::ReadWrite,
        flags => {
            return Err(PlanError::UnsupportedProgramHeaderFlags {
                index: raw.index,
                flags,
            });
        }
    };
    if raw.alignment != LOAD_ALIGNMENT {
        return Err(PlanError::UnsupportedLoadAlignment {
            index: raw.index,
            alignment: raw.alignment,
        });
    }
    if raw.file_size == 0 || raw.memory_size == 0 {
        return Err(PlanError::UnsupportedProgramHeaderProfile);
    }

    let mapping_address = align_down(raw.virtual_address, LOAD_ALIGNMENT);
    let prefix = raw.virtual_address - mapping_address;
    let unrounded = prefix
        .checked_add(raw.memory_size)
        .ok_or(PlanError::MappingRangeOverflow)?;
    let mapping_size =
        align_up(unrounded, LOAD_ALIGNMENT).ok_or(PlanError::MappingRangeOverflow)?;
    mapping_address
        .checked_add(mapping_size)
        .ok_or(PlanError::MappingRangeOverflow)?;

    Ok(LoadSegment {
        file_offset: raw.offset,
        file_size: raw.file_size,
        virtual_address: raw.virtual_address,
        memory_size: raw.memory_size,
        mapping_address,
        mapping_size,
        permissions,
    })
}

fn validate_load_profile(segments: &[LoadSegment; LOAD_SEGMENT_COUNT]) -> Result<(), PlanError> {
    let mut read_only = 0;
    let mut read_execute = 0;
    let mut read_write = 0;
    for segment in segments {
        match segment.permissions {
            SegmentPermissions::ReadOnly => read_only += 1,
            SegmentPermissions::ReadExecute => read_execute += 1,
            SegmentPermissions::ReadWrite => read_write += 1,
        }
    }
    if (read_only, read_execute, read_write) != (1, 1, 1) {
        return Err(PlanError::UnsupportedProgramHeaderProfile);
    }

    for (index, first) in segments.iter().enumerate() {
        for second in &segments[index + 1..] {
            if ranges_overlap(
                first.file_offset,
                first.file_end(),
                second.file_offset,
                second.file_end(),
            ) {
                return Err(PlanError::LoadFileOverlap);
            }
        }
    }
    for pair in segments.windows(2) {
        let first = pair[0];
        let second = pair[1];
        if ranges_overlap(
            first.virtual_address,
            first.memory_end(),
            second.virtual_address,
            second.memory_end(),
        ) {
            return Err(PlanError::LoadMemoryOverlap);
        }
        if ranges_overlap(
            first.mapping_address,
            first.mapping_end(),
            second.mapping_address,
            second.mapping_end(),
        ) {
            return Err(PlanError::LoadMappingOverlap);
        }
    }
    Ok(())
}

fn ranges_overlap(first_start: u64, first_end: u64, second_start: u64, second_end: u64) -> bool {
    first_start < second_end && second_start < first_end
}

fn validate_program_descriptor(
    raw: ProgramHeader,
    header: Header,
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
) -> Result<(), PlanError> {
    let table_size = (header.program_count as u64)
        .checked_mul(PROGRAM_HEADER_BYTES as u64)
        .ok_or(PlanError::ProgramHeaderDescriptorMismatch)?;
    if raw.flags != PF_READ
        || raw.alignment != 8
        || raw.offset != header.program_offset
        || raw.virtual_address != header.program_offset
        || raw.file_size != table_size
        || raw.memory_size != table_size
    {
        return Err(PlanError::ProgramHeaderDescriptorMismatch);
    }
    validate_descriptor_load_mapping(
        raw,
        segments,
        SegmentPermissions::ReadOnly,
        PlanError::ProgramHeaderNotReadOnlyLoaded,
    )
}

fn validate_dynamic_descriptor(
    raw: ProgramHeader,
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
) -> Result<(), PlanError> {
    if raw.flags != PF_READ_WRITE
        || raw.alignment != 8
        || raw.file_size == 0
        || raw.file_size != raw.memory_size
        || !raw.file_size.is_multiple_of(DYNAMIC_ENTRY_BYTES)
    {
        return Err(PlanError::UnsupportedProgramHeaderProfile);
    }
    validate_descriptor_load_mapping(
        raw,
        segments,
        SegmentPermissions::ReadWrite,
        PlanError::DynamicSegmentNotWritableLoaded,
    )
}

fn validate_relro_descriptor(
    raw: ProgramHeader,
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
) -> Result<(), PlanError> {
    let writable = segments
        .iter()
        .find(|segment| segment.permissions == SegmentPermissions::ReadWrite)
        .ok_or(PlanError::RelroSegmentMismatch)?;
    if raw.flags != PF_READ
        || raw.alignment != 1
        || raw.offset != writable.file_offset
        || raw.virtual_address != writable.virtual_address
        || raw.file_size != writable.file_size
        || raw.memory_size != writable.memory_size
    {
        return Err(PlanError::RelroSegmentMismatch);
    }
    validate_descriptor_load_mapping(
        raw,
        segments,
        SegmentPermissions::ReadWrite,
        PlanError::RelroSegmentMismatch,
    )
}

fn validate_stack_descriptor(raw: ProgramHeader) -> Result<(), PlanError> {
    if raw.flags != PF_READ_WRITE
        || raw.offset != 0
        || raw.virtual_address != 0
        || raw.physical_address != 0
        || raw.file_size != 0
        || raw.memory_size != 0
        || raw.alignment != 0
    {
        return Err(PlanError::InvalidStackDescriptor);
    }
    Ok(())
}

fn validate_note_descriptor(
    raw: ProgramHeader,
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
) -> Result<(), PlanError> {
    if raw.flags != PF_READ
        || raw.alignment != 4
        || raw.file_size == 0
        || raw.file_size != raw.memory_size
    {
        return Err(PlanError::UnsupportedProgramHeaderProfile);
    }
    validate_descriptor_load_mapping(
        raw,
        segments,
        SegmentPermissions::ReadOnly,
        PlanError::MetadataNoteNotReadOnlyLoaded,
    )
}

fn validate_descriptor_load_mapping(
    raw: ProgramHeader,
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
    permissions: SegmentPermissions,
    not_contained: PlanError,
) -> Result<(), PlanError> {
    let mut file_segment = None;
    let mut memory_segment = None;
    for (index, segment) in segments.iter().enumerate() {
        if segment.permissions != permissions {
            continue;
        }
        if range_contains(
            segment.virtual_address,
            segment.memory_size,
            raw.virtual_address,
            raw.memory_size,
        ) && memory_segment.replace(index).is_some()
        {
            return Err(PlanError::AmbiguousDescriptorLoadMapping { index: raw.index });
        }
        if range_contains(
            segment.file_offset,
            segment.file_size,
            raw.offset,
            raw.file_size,
        ) && file_segment.replace(index).is_some()
        {
            return Err(PlanError::AmbiguousDescriptorLoadMapping { index: raw.index });
        }
    }

    let (Some(file_segment), Some(memory_segment)) = (file_segment, memory_segment) else {
        return Err(not_contained);
    };
    if file_segment != memory_segment {
        return Err(PlanError::DescriptorLoadMappingMismatch { index: raw.index });
    }

    let segment = segments[file_segment];
    let file_delta = raw
        .offset
        .checked_sub(segment.file_offset)
        .ok_or(PlanError::DescriptorLoadMappingMismatch { index: raw.index })?;
    let memory_delta = raw
        .virtual_address
        .checked_sub(segment.virtual_address)
        .ok_or(PlanError::DescriptorLoadMappingMismatch { index: raw.index })?;
    if file_delta != memory_delta {
        return Err(PlanError::DescriptorLoadMappingMismatch { index: raw.index });
    }
    Ok(())
}

fn range_contains(outer_start: u64, outer_len: u64, inner_start: u64, inner_len: u64) -> bool {
    let Some(outer_end) = outer_start.checked_add(outer_len) else {
        return false;
    };
    let Some(inner_end) = inner_start.checked_add(inner_len) else {
        return false;
    };
    inner_start >= outer_start && inner_end <= outer_end
}

fn parse_metadata_note(bytes: &[u8], note: ProgramHeader) -> Result<MetadataNote, PlanError> {
    let start = usize::try_from(note.offset).map_err(|_| PlanError::InvalidNoteRecord)?;
    let end_u64 = note
        .offset
        .checked_add(note.file_size)
        .ok_or(PlanError::InvalidNoteRecord)?;
    let end = usize::try_from(end_u64).map_err(|_| PlanError::InvalidNoteRecord)?;
    let name_size = u64::from(read_u32(bytes, start, "note name size")?);
    let descriptor_size = u64::from(read_u32(bytes, start + 4, "note descriptor size")?);
    let note_type = read_u32(bytes, start + 8, "note type")?;
    if name_size != AMDGPU_NOTE_NAME.len() as u64 || note_type != NT_AMDGPU_METADATA {
        return Err(PlanError::UnsupportedNote);
    }
    if descriptor_size == 0 || descriptor_size > MAX_METADATA_BYTES {
        return Err(PlanError::MetadataNoteTooLarge);
    }

    let name_start = start.checked_add(12).ok_or(PlanError::InvalidNoteRecord)?;
    let name_end = name_start
        .checked_add(name_size as usize)
        .ok_or(PlanError::InvalidNoteRecord)?;
    if bytes.get(name_start..name_end) != Some(AMDGPU_NOTE_NAME) {
        return Err(PlanError::UnsupportedNote);
    }
    let descriptor_start = align_up(name_end as u64, 4)
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or(PlanError::InvalidNoteRecord)?;
    let descriptor_end = descriptor_start
        .checked_add(descriptor_size as usize)
        .ok_or(PlanError::InvalidNoteRecord)?;
    let record_end = align_up(descriptor_end as u64, 4)
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or(PlanError::InvalidNoteRecord)?;
    if descriptor_end > end || record_end != end {
        return Err(PlanError::InvalidNoteRecord);
    }

    Ok(MetadataNote {
        file_offset: descriptor_start as u64,
        byte_len: descriptor_size,
    })
}

fn validate_dynamic_entries(
    bytes: &[u8],
    dynamic: ProgramHeader,
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
) -> Result<(), PlanError> {
    let entry_count = usize::try_from(dynamic.file_size / DYNAMIC_ENTRY_BYTES)
        .map_err(|_| PlanError::TooManyDynamicEntries)?;
    if entry_count > 16 {
        return Err(PlanError::TooManyDynamicEntries);
    }
    let base = usize::try_from(dynamic.offset).map_err(|_| PlanError::FileRangeOverflow {
        index: dynamic.index,
    })?;

    let mut seen = 0u16;
    let mut symtab = 0;
    let mut strtab = 0;
    let mut strsz = 0;
    let mut hash = 0;
    let mut gnu_hash = 0;
    let mut terminated = false;
    for index in 0..entry_count {
        let offset = base
            .checked_add(index * DYNAMIC_ENTRY_BYTES as usize)
            .ok_or(PlanError::FileRangeOverflow {
                index: dynamic.index,
            })?;
        let tag = read_u64(bytes, offset, "dynamic tag")?;
        let value = read_u64(bytes, offset + 8, "dynamic value")?;
        if tag == DT_NULL {
            if value != 0 || index + 1 != entry_count {
                return Err(PlanError::InvalidDynamicValue { tag, value });
            }
            terminated = true;
            break;
        }
        let (bit, destination) = match tag {
            DT_SYMTAB => (1u16 << 0, Some(&mut symtab)),
            DT_SYMENT => {
                if value != 24 {
                    return Err(PlanError::InvalidDynamicValue { tag, value });
                }
                (1u16 << 1, None)
            }
            DT_STRTAB => (1u16 << 2, Some(&mut strtab)),
            DT_STRSZ => {
                if value == 0 {
                    return Err(PlanError::InvalidDynamicValue { tag, value });
                }
                strsz = value;
                (1u16 << 3, None)
            }
            DT_HASH => (1u16 << 4, Some(&mut hash)),
            DT_GNU_HASH => (1u16 << 5, Some(&mut gnu_hash)),
            DT_FLAGS => {
                if value != DF_SYMBOLIC {
                    return Err(PlanError::InvalidDynamicValue { tag, value });
                }
                (1u16 << 6, None)
            }
            DT_PLTRELSZ | DT_RELA | DT_RELASZ | DT_RELAENT | DT_REL | DT_RELSZ | DT_RELENT
            | DT_PLTREL | DT_JMPREL | DT_RELACOUNT | DT_RELCOUNT => {
                return Err(PlanError::UnsupportedRelocationTag(tag));
            }
            DT_INIT | DT_FINI | DT_INIT_ARRAY | DT_FINI_ARRAY | DT_INIT_ARRAYSZ
            | DT_FINI_ARRAYSZ => return Err(PlanError::UnsupportedDynamicFeature(tag)),
            tag => return Err(PlanError::UnsupportedDynamicTag(tag)),
        };
        if seen & bit != 0 {
            return Err(PlanError::DuplicateDynamicTag(tag));
        }
        seen |= bit;
        if let Some(destination) = destination {
            *destination = value;
        }
    }
    if !terminated {
        return Err(PlanError::MissingDynamicTerminator);
    }
    const REQUIRED: u16 = (1 << 6) - 1;
    if seen & REQUIRED != REQUIRED {
        for (bit, tag) in [
            (1u16 << 0, DT_SYMTAB),
            (1u16 << 1, DT_SYMENT),
            (1u16 << 2, DT_STRTAB),
            (1u16 << 3, DT_STRSZ),
            (1u16 << 4, DT_HASH),
            (1u16 << 5, DT_GNU_HASH),
        ] {
            if seen & bit == 0 {
                return Err(PlanError::MissingDynamicTag(tag));
            }
        }
    }

    validate_read_only_reference(segments, symtab, 24, DT_SYMTAB)?;
    validate_read_only_reference(segments, strtab, strsz, DT_STRTAB)?;
    validate_read_only_reference(segments, hash, 8, DT_HASH)?;
    validate_read_only_reference(segments, gnu_hash, 16, DT_GNU_HASH)?;
    Ok(())
}

fn validate_read_only_reference(
    segments: &[LoadSegment; LOAD_SEGMENT_COUNT],
    address: u64,
    len: u64,
    tag: u64,
) -> Result<(), PlanError> {
    let valid = segments.iter().any(|segment| {
        segment.permissions == SegmentPermissions::ReadOnly
            && range_contains(segment.virtual_address, segment.file_size, address, len)
    });
    if valid {
        Ok(())
    } else {
        Err(PlanError::DynamicReferenceOutOfBounds(tag))
    }
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| align_down(sum, alignment))
}

fn table_entry_offset(
    table_offset: u64,
    entry_size: usize,
    index: usize,
    context: &'static str,
) -> Result<usize, PlanError> {
    let relative = (entry_size as u64)
        .checked_mul(index as u64)
        .ok_or(PlanError::TableRangeOverflow(context))?;
    let offset = table_offset
        .checked_add(relative)
        .ok_or(PlanError::TableRangeOverflow(context))?;
    usize::try_from(offset).map_err(|_| PlanError::TableRangeOverflow(context))
}

fn read_u16(bytes: &[u8], offset: usize, context: &'static str) -> Result<u16, PlanError> {
    Ok(u16::from_le_bytes(read_array(bytes, offset, context)?))
}

fn read_u32(bytes: &[u8], offset: usize, context: &'static str) -> Result<u32, PlanError> {
    Ok(u32::from_le_bytes(read_array(bytes, offset, context)?))
}

fn read_u64(bytes: &[u8], offset: usize, context: &'static str) -> Result<u64, PlanError> {
    Ok(u64::from_le_bytes(read_array(bytes, offset, context)?))
}

fn read_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
    context: &'static str,
) -> Result<[u8; N], PlanError> {
    let end = offset.checked_add(N).ok_or(PlanError::Truncated(context))?;
    bytes
        .get(offset..end)
        .ok_or(PlanError::Truncated(context))?
        .try_into()
        .map_err(|_| PlanError::Truncated(context))
}

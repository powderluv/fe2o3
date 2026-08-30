use crate::advanced_model::AdvancedCapabilityStatus;
use crate::atomic_legalizability::{
    AtomicLegalizability, StandardAtomicQuery, evaluate_gfx942_atomic_query,
};
use crate::capabilities::{AtomicScopes, WavefrontWidth};

/// One coordinate axis in a three-dimensional workgroup.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WorkgroupAxis {
    X,
    Y,
    Z,
}

/// Reviewed target limits for one workgroup.
///
/// A dimension is admissible only when every extent is nonzero, each extent
/// fits its axis limit, and their product fits the total work-item limit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorkgroupLimits {
    max_workitems: u32,
    max_extents: [u32; 3],
}

impl WorkgroupLimits {
    pub(crate) const GFX942: Self = Self {
        max_workitems: 1024,
        max_extents: [1024, 1024, 1024],
    };

    // Reviewed against rocminfo on an MI350X (gfx950), ROCm 7.2.1 / HSA
    // runtime 1.18: total and per-dimension workgroup limits are all 1024.
    pub(crate) const GFX950_MI350_ROCM_7_2_REVIEWED: Self = Self {
        max_workitems: 1024,
        max_extents: [1024, 1024, 1024],
    };

    /// Maximum total work-items in one workgroup.
    pub const fn max_workitems(self) -> u32 {
        self.max_workitems
    }

    /// Maximum extent along `axis`, subject to the total work-item limit.
    pub const fn max_extent(self, axis: WorkgroupAxis) -> u32 {
        match axis {
            WorkgroupAxis::X => self.max_extents[0],
            WorkgroupAxis::Y => self.max_extents[1],
            WorkgroupAxis::Z => self.max_extents[2],
        }
    }

    /// Returns whether the exact three-dimensional dimensions fit this target.
    pub const fn supports_dimensions(self, x: u32, y: u32, z: u32) -> bool {
        if x == 0
            || y == 0
            || z == 0
            || x > self.max_extents[0]
            || y > self.max_extents[1]
            || z > self.max_extents[2]
        {
            return false;
        }
        let Some(xy) = x.checked_mul(y) else {
            return false;
        };
        let Some(xyz) = xy.checked_mul(z) else {
            return false;
        };
        xyz <= self.max_workitems
    }

    pub(crate) const fn max_wavefronts(self, width: WavefrontWidth) -> u32 {
        let lanes = match width {
            WavefrontWidth::Wave32 => 32,
            WavefrontWidth::Wave64 => 64,
        };
        self.max_workitems.div_ceil(lanes)
    }
}

/// A storage width supported by the standard Rust atomic source contract.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AtomicWidth {
    Bits8,
    Bits16,
    Bits32,
    Bits64,
    Bits128,
}

/// A compact set of standard Rust atomic storage widths.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AtomicWidths(u8);

impl AtomicWidths {
    const BITS8: u8 = 1 << 0;
    const BITS16: u8 = 1 << 1;
    const BITS32: u8 = 1 << 2;
    const BITS64: u8 = 1 << 3;
    const BITS128: u8 = 1 << 4;

    pub(crate) const NONE: Self = Self(0);
    pub(crate) const GFX942_LEGALIZABLE: Self =
        Self(Self::BITS8 | Self::BITS16 | Self::BITS32 | Self::BITS64);

    /// Returns whether `width` occurs in at least one reviewed legalizable
    /// standard-atomic combination.
    ///
    /// This coarse projection must not be combined independently with the
    /// scope or ordering sets. Use the complete atomic query for admission.
    pub const fn contains(self, width: AtomicWidth) -> bool {
        let flag = match width {
            AtomicWidth::Bits8 => Self::BITS8,
            AtomicWidth::Bits16 => Self::BITS16,
            AtomicWidth::Bits32 => Self::BITS32,
            AtomicWidth::Bits64 => Self::BITS64,
            AtomicWidth::Bits128 => Self::BITS128,
        };
        self.0 & flag != 0
    }

    /// Returns whether no standard atomic width has been reviewed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A Rust atomic ordering supported by target lowering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

/// A compact set of Rust atomic orderings.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AtomicOrderings(u8);

impl AtomicOrderings {
    const RELAXED: u8 = 1 << 0;
    const ACQUIRE: u8 = 1 << 1;
    const RELEASE: u8 = 1 << 2;
    const ACQUIRE_RELEASE: u8 = 1 << 3;
    const SEQUENTIALLY_CONSISTENT: u8 = 1 << 4;

    pub(crate) const NONE: Self = Self(0);
    pub(crate) const ALL: Self = Self(
        Self::RELAXED
            | Self::ACQUIRE
            | Self::RELEASE
            | Self::ACQUIRE_RELEASE
            | Self::SEQUENTIALLY_CONSISTENT,
    );

    /// Returns whether target lowering preserves `ordering`.
    ///
    /// Operation-specific legality remains a separate verifier obligation.
    pub const fn contains(self, ordering: AtomicOrdering) -> bool {
        let flag = match ordering {
            AtomicOrdering::Relaxed => Self::RELAXED,
            AtomicOrdering::Acquire => Self::ACQUIRE,
            AtomicOrdering::Release => Self::RELEASE,
            AtomicOrdering::AcquireRelease => Self::ACQUIRE_RELEASE,
            AtomicOrdering::SequentiallyConsistent => Self::SEQUENTIALLY_CONSISTENT,
        };
        self.0 & flag != 0
    }

    /// Returns whether no standard atomic ordering has been reviewed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A scalar eight-bit floating-point encoding.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Fp8Format {
    E4M3Fnuz,
    E5M2Fnuz,
    E4M3Ocp,
    E5M2Ocp,
}

/// A compact set of scalar FP8 encodings.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Fp8Formats(u8);

impl Fp8Formats {
    const E4M3_FNUZ: u8 = 1 << 0;
    const E5M2_FNUZ: u8 = 1 << 1;
    const E4M3_OCP: u8 = 1 << 2;
    const E5M2_OCP: u8 = 1 << 3;

    pub(crate) const NONE: Self = Self(0);
    pub(crate) const FNUZ: Self = Self(Self::E4M3_FNUZ | Self::E5M2_FNUZ);
    pub(crate) const OCP: Self = Self(Self::E4M3_OCP | Self::E5M2_OCP);

    /// Returns whether the exact scalar encoding is reviewed for the target.
    ///
    /// This does not by itself authorize a conversion or matrix operation.
    pub const fn contains(self, format: Fp8Format) -> bool {
        let flag = match format {
            Fp8Format::E4M3Fnuz => Self::E4M3_FNUZ,
            Fp8Format::E5M2Fnuz => Self::E5M2_FNUZ,
            Fp8Format::E4M3Ocp => Self::E4M3_OCP,
            Fp8Format::E5M2Ocp => Self::E5M2_OCP,
        };
        self.0 & flag != 0
    }

    /// Returns whether no scalar FP8 encoding has been reviewed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A block-scaled Open Compute Project microscaling format.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MxFormat {
    Fp8,
    Bf8,
    Fp4,
}

/// A compact set of block-scaled microscaling formats.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MxFormats(u8);

impl MxFormats {
    const FP8: u8 = 1 << 0;
    const BF8: u8 = 1 << 1;
    const FP4: u8 = 1 << 2;

    pub(crate) const NONE: Self = Self(0);
    pub(crate) const GFX950_REVIEWED: Self = Self(Self::FP8 | Self::BF8 | Self::FP4);

    /// Returns whether the target has a reviewed native MX format contract.
    pub const fn contains(self, format: MxFormat) -> bool {
        let flag = match format {
            MxFormat::Fp8 => Self::FP8,
            MxFormat::Bf8 => Self::BF8,
            MxFormat::Fp4 => Self::FP4,
        };
        self.0 & flag != 0
    }

    /// Returns whether no MX format has been reviewed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A numerical input/accumulator family for AMD MFMA instructions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MfmaFamily {
    F32FromF16,
    F32FromBf16,
    F32FromFp8Fnuz,
    F32FromBf8Fnuz,
    F32FromFp8Ocp,
    F32FromBf8Ocp,
    F32FromFp4Ocp,
    F64FromF64,
    I32FromI8,
}

/// A compact set of reviewed MFMA numerical families.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MfmaFamilies(u16);

impl MfmaFamilies {
    const F32_FROM_F16: u16 = 1 << 0;
    const F32_FROM_BF16: u16 = 1 << 1;
    const F32_FROM_FP8_FNUZ: u16 = 1 << 2;
    const F32_FROM_BF8_FNUZ: u16 = 1 << 3;
    const F64_FROM_F64: u16 = 1 << 4;
    const I32_FROM_I8: u16 = 1 << 5;
    const F32_FROM_FP8_OCP: u16 = 1 << 6;
    const F32_FROM_BF8_OCP: u16 = 1 << 7;
    const F32_FROM_FP4_OCP: u16 = 1 << 8;

    pub(crate) const NONE: Self = Self(0);
    pub(crate) const GFX942_REVIEWED: Self = Self(
        Self::F32_FROM_F16
            | Self::F32_FROM_BF16
            | Self::F32_FROM_FP8_FNUZ
            | Self::F32_FROM_BF8_FNUZ,
    );
    pub(crate) const GFX950_REVIEWED: Self = Self(
        Self::F32_FROM_BF16
            | Self::F32_FROM_FP8_OCP
            | Self::F32_FROM_BF8_OCP
            | Self::F32_FROM_FP4_OCP,
    );

    /// Returns whether the numerical family has a reviewed target mapping.
    ///
    /// Shapes, controls, wave width, and operand layout require separate
    /// capability checks.
    pub const fn contains(self, family: MfmaFamily) -> bool {
        let flag = match family {
            MfmaFamily::F32FromF16 => Self::F32_FROM_F16,
            MfmaFamily::F32FromBf16 => Self::F32_FROM_BF16,
            MfmaFamily::F32FromFp8Fnuz => Self::F32_FROM_FP8_FNUZ,
            MfmaFamily::F32FromBf8Fnuz => Self::F32_FROM_BF8_FNUZ,
            MfmaFamily::F32FromFp8Ocp => Self::F32_FROM_FP8_OCP,
            MfmaFamily::F32FromBf8Ocp => Self::F32_FROM_BF8_OCP,
            MfmaFamily::F32FromFp4Ocp => Self::F32_FROM_FP4_OCP,
            MfmaFamily::F64FromF64 => Self::F64_FROM_F64,
            MfmaFamily::I32FromI8 => Self::I32_FROM_I8,
        };
        self.0 & flag != 0
    }

    /// Returns whether no MFMA numerical family has been reviewed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// One gfx950 LDS transpose-load instruction.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LdsTransposeInstruction {
    DsReadTr4B64,
    DsReadTr8B64,
    DsReadTr16B64,
}

/// A compact set of reviewed LDS transpose-load instructions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LdsTransposeInstructions(u8);

impl LdsTransposeInstructions {
    const TR4_B64: u8 = 1 << 0;
    const TR8_B64: u8 = 1 << 1;
    const TR16_B64: u8 = 1 << 2;

    pub(crate) const NONE: Self = Self(0);
    pub(crate) const GFX950_REVIEWED: Self = Self(Self::TR4_B64 | Self::TR8_B64 | Self::TR16_B64);

    /// Returns whether the exact transpose-load instruction is reviewed.
    pub const fn contains(self, instruction: LdsTransposeInstruction) -> bool {
        let flag = match instruction {
            LdsTransposeInstruction::DsReadTr4B64 => Self::TR4_B64,
            LdsTransposeInstruction::DsReadTr8B64 => Self::TR8_B64,
            LdsTransposeInstruction::DsReadTr16B64 => Self::TR16_B64,
        };
        self.0 & flag != 0
    }

    /// Returns whether no LDS transpose-load instruction has been reviewed.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A target-facing diagnostic or observation facility used by device code.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceDiagnosticFeature {
    Printf,
    Trap,
    DebugTrap,
    ClockCounter,
    ProfilingMarker,
}

/// One source-level launch-bounds field requiring target metadata support.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LaunchBoundsField {
    MaxWorkgroupSize,
    MinWorkgroupsPerComputeUnit,
}

/// One target metadata form related to source-level launch bounds.
///
/// Metadata availability does not establish that a source-level field has a
/// semantics-preserving translation. In particular, waves-per-EU metadata is
/// not itself CUDA-style minimum-workgroups-per-compute-unit admission.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LaunchBoundsMetadata {
    FlatWorkgroupSize,
    WavesPerExecutionUnit,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AdvancedTargetCapabilities {
    pub(crate) profile_status: AdvancedCapabilityStatus,
    pub(crate) workgroup_limits: Option<WorkgroupLimits>,
    pub(crate) workgroup_limits_status: AdvancedCapabilityStatus,
    pub(crate) standard_atomic_widths: AtomicWidths,
    pub(crate) standard_atomic_scopes: AtomicScopes,
    pub(crate) standard_atomic_orderings: AtomicOrderings,
    pub(crate) standard_atomics_status: AdvancedCapabilityStatus,
    pub(crate) native_split_barriers: AdvancedCapabilityStatus,
    pub(crate) fp8_formats: Fp8Formats,
    pub(crate) fp8_formats_status: AdvancedCapabilityStatus,
    pub(crate) fp8_formats_nonmember_status: AdvancedCapabilityStatus,
    pub(crate) mx_formats: MxFormats,
    pub(crate) mx_formats_status: AdvancedCapabilityStatus,
    pub(crate) mx_formats_nonmember_status: AdvancedCapabilityStatus,
    pub(crate) mfma_families: MfmaFamilies,
    pub(crate) mfma_families_status: AdvancedCapabilityStatus,
    pub(crate) mfma_families_nonmember_status: AdvancedCapabilityStatus,
    pub(crate) lds_transpose_instructions: LdsTransposeInstructions,
    pub(crate) lds_transpose_instructions_status: AdvancedCapabilityStatus,
    pub(crate) lds_transpose_instructions_nonmember_status: AdvancedCapabilityStatus,
    pub(crate) device_printf: AdvancedCapabilityStatus,
    pub(crate) device_trap: AdvancedCapabilityStatus,
    pub(crate) device_debug_trap: AdvancedCapabilityStatus,
    pub(crate) device_clock_counter: AdvancedCapabilityStatus,
    pub(crate) device_profiling_marker: AdvancedCapabilityStatus,
    pub(crate) max_workgroup_size: AdvancedCapabilityStatus,
    pub(crate) min_workgroups_per_compute_unit: AdvancedCapabilityStatus,
    pub(crate) flat_workgroup_size_metadata: AdvancedCapabilityStatus,
    pub(crate) waves_per_execution_unit_metadata: AdvancedCapabilityStatus,
}

impl AdvancedTargetCapabilities {
    pub(crate) fn for_processor(processor: &str) -> Self {
        if processor == "gfx942" {
            Self {
                profile_status: AdvancedCapabilityStatus::Supported,
                workgroup_limits: Some(WorkgroupLimits::GFX942),
                workgroup_limits_status: AdvancedCapabilityStatus::Supported,
                standard_atomic_widths: AtomicWidths::GFX942_LEGALIZABLE,
                standard_atomic_scopes: AtomicScopes::ALL,
                standard_atomic_orderings: AtomicOrderings::ALL,
                standard_atomics_status: AdvancedCapabilityStatus::Supported,
                native_split_barriers: AdvancedCapabilityStatus::Unsupported,
                fp8_formats: Fp8Formats::FNUZ,
                fp8_formats_status: AdvancedCapabilityStatus::Supported,
                fp8_formats_nonmember_status: AdvancedCapabilityStatus::Unsupported,
                mx_formats: MxFormats::NONE,
                mx_formats_status: AdvancedCapabilityStatus::Supported,
                mx_formats_nonmember_status: AdvancedCapabilityStatus::Unsupported,
                mfma_families: MfmaFamilies::GFX942_REVIEWED,
                mfma_families_status: AdvancedCapabilityStatus::Supported,
                mfma_families_nonmember_status: AdvancedCapabilityStatus::Unsupported,
                lds_transpose_instructions: LdsTransposeInstructions::NONE,
                lds_transpose_instructions_status: AdvancedCapabilityStatus::Supported,
                lds_transpose_instructions_nonmember_status: AdvancedCapabilityStatus::Unsupported,
                // Device printf additionally needs an allowlisted device
                // library ABI and compatible host runtime.
                device_printf: AdvancedCapabilityStatus::RequiresRuntimeEvidence,
                device_trap: AdvancedCapabilityStatus::Supported,
                // Debug-trap observation depends on the driver/debugger path.
                device_debug_trap: AdvancedCapabilityStatus::RequiresRuntimeEvidence,
                device_clock_counter: AdvancedCapabilityStatus::Supported,
                device_profiling_marker: AdvancedCapabilityStatus::Unsupported,
                max_workgroup_size: AdvancedCapabilityStatus::Supported,
                // A CUDA-style minimum-blocks hint needs a reviewed gfx942
                // occupancy translation before it can be admitted.
                min_workgroups_per_compute_unit: AdvancedCapabilityStatus::Unsupported,
                flat_workgroup_size_metadata: AdvancedCapabilityStatus::Supported,
                waves_per_execution_unit_metadata: AdvancedCapabilityStatus::Supported,
            }
        } else if processor == "gfx950" {
            Self {
                profile_status: AdvancedCapabilityStatus::Supported,
                workgroup_limits: Some(WorkgroupLimits::GFX950_MI350_ROCM_7_2_REVIEWED),
                workgroup_limits_status: AdvancedCapabilityStatus::Supported,
                standard_atomic_widths: AtomicWidths::NONE,
                standard_atomic_scopes: AtomicScopes::NONE,
                standard_atomic_orderings: AtomicOrderings::NONE,
                standard_atomics_status: AdvancedCapabilityStatus::Unreviewed,
                native_split_barriers: AdvancedCapabilityStatus::Unreviewed,
                fp8_formats: Fp8Formats::OCP,
                fp8_formats_status: AdvancedCapabilityStatus::Supported,
                fp8_formats_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
                mx_formats: MxFormats::GFX950_REVIEWED,
                mx_formats_status: AdvancedCapabilityStatus::Supported,
                mx_formats_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
                mfma_families: MfmaFamilies::GFX950_REVIEWED,
                mfma_families_status: AdvancedCapabilityStatus::Supported,
                mfma_families_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
                lds_transpose_instructions: LdsTransposeInstructions::GFX950_REVIEWED,
                lds_transpose_instructions_status: AdvancedCapabilityStatus::Supported,
                lds_transpose_instructions_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
                device_printf: AdvancedCapabilityStatus::Unreviewed,
                device_trap: AdvancedCapabilityStatus::Unreviewed,
                device_debug_trap: AdvancedCapabilityStatus::Unreviewed,
                device_clock_counter: AdvancedCapabilityStatus::Unreviewed,
                device_profiling_marker: AdvancedCapabilityStatus::Unreviewed,
                max_workgroup_size: AdvancedCapabilityStatus::Unreviewed,
                min_workgroups_per_compute_unit: AdvancedCapabilityStatus::Unreviewed,
                flat_workgroup_size_metadata: AdvancedCapabilityStatus::Unreviewed,
                waves_per_execution_unit_metadata: AdvancedCapabilityStatus::Unreviewed,
            }
        } else {
            Self::unreviewed()
        }
    }

    const fn unreviewed() -> Self {
        Self {
            profile_status: AdvancedCapabilityStatus::Unreviewed,
            workgroup_limits: None,
            workgroup_limits_status: AdvancedCapabilityStatus::Unreviewed,
            standard_atomic_widths: AtomicWidths::NONE,
            standard_atomic_scopes: AtomicScopes::NONE,
            standard_atomic_orderings: AtomicOrderings::NONE,
            standard_atomics_status: AdvancedCapabilityStatus::Unreviewed,
            native_split_barriers: AdvancedCapabilityStatus::Unreviewed,
            fp8_formats: Fp8Formats::NONE,
            fp8_formats_status: AdvancedCapabilityStatus::Unreviewed,
            fp8_formats_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
            mx_formats: MxFormats::NONE,
            mx_formats_status: AdvancedCapabilityStatus::Unreviewed,
            mx_formats_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
            mfma_families: MfmaFamilies::NONE,
            mfma_families_status: AdvancedCapabilityStatus::Unreviewed,
            mfma_families_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
            lds_transpose_instructions: LdsTransposeInstructions::NONE,
            lds_transpose_instructions_status: AdvancedCapabilityStatus::Unreviewed,
            lds_transpose_instructions_nonmember_status: AdvancedCapabilityStatus::Unreviewed,
            device_printf: AdvancedCapabilityStatus::Unreviewed,
            device_trap: AdvancedCapabilityStatus::Unreviewed,
            device_debug_trap: AdvancedCapabilityStatus::Unreviewed,
            device_clock_counter: AdvancedCapabilityStatus::Unreviewed,
            device_profiling_marker: AdvancedCapabilityStatus::Unreviewed,
            max_workgroup_size: AdvancedCapabilityStatus::Unreviewed,
            min_workgroups_per_compute_unit: AdvancedCapabilityStatus::Unreviewed,
            flat_workgroup_size_metadata: AdvancedCapabilityStatus::Unreviewed,
            waves_per_execution_unit_metadata: AdvancedCapabilityStatus::Unreviewed,
        }
    }

    pub(crate) const fn reviewed_set_member(
        member_status: AdvancedCapabilityStatus,
        nonmember_status: AdvancedCapabilityStatus,
        contains: bool,
    ) -> AdvancedCapabilityStatus {
        if contains {
            member_status
        } else {
            nonmember_status
        }
    }

    pub(crate) const fn atomic_legalizability(
        self,
        query: StandardAtomicQuery,
    ) -> AtomicLegalizability {
        evaluate_gfx942_atomic_query(self.standard_atomics_status, query)
    }
}

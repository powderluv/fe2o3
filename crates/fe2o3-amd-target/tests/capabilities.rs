use fe2o3_amd_target::{
    ADVANCED_CAPABILITY_MODEL_REVISION, AdvancedCapabilityModelIdentity,
    AdvancedCapabilityModelRevision, AdvancedCapabilityStatus, AmdTargetFeature, AmdTargetId,
    AsyncCopyInstructionSet, AtomicAddressSpace, AtomicLegalizability, AtomicOperation,
    AtomicOrdering, AtomicScope, AtomicWidth, CapabilityDerivationError, CapabilitySupport,
    DeviceDiagnosticFeature, Fp8Format, KNOWN_PROCESSORS, LaunchBoundsField, LaunchBoundsMetadata,
    LdsTransposeInstruction, MatrixInstructionSet, MfmaFamily, MxFormat, ParseAmdTargetIdError,
    StandardAtomicQuery, WavefrontWidth, WorkgroupAxis,
};

const ATOMIC_OPERATIONS: [AtomicOperation; 14] = [
    AtomicOperation::Load,
    AtomicOperation::Store,
    AtomicOperation::Swap,
    AtomicOperation::CompareExchange,
    AtomicOperation::FetchAdd,
    AtomicOperation::FetchSub,
    AtomicOperation::FetchAnd,
    AtomicOperation::FetchNand,
    AtomicOperation::FetchOr,
    AtomicOperation::FetchXor,
    AtomicOperation::FetchMinSigned,
    AtomicOperation::FetchMinUnsigned,
    AtomicOperation::FetchMaxSigned,
    AtomicOperation::FetchMaxUnsigned,
];
const ATOMIC_WIDTHS: [AtomicWidth; 5] = [
    AtomicWidth::Bits8,
    AtomicWidth::Bits16,
    AtomicWidth::Bits32,
    AtomicWidth::Bits64,
    AtomicWidth::Bits128,
];
const ATOMIC_ADDRESS_SPACES: [AtomicAddressSpace; 5] = [
    AtomicAddressSpace::Global,
    AtomicAddressSpace::Workgroup,
    AtomicAddressSpace::Generic,
    AtomicAddressSpace::Private,
    AtomicAddressSpace::Constant,
];
const ATOMIC_SCOPES: [AtomicScope; 3] = [
    AtomicScope::Workgroup,
    AtomicScope::Device,
    AtomicScope::System,
];
const ATOMIC_ORDERINGS: [AtomicOrdering; 5] = [
    AtomicOrdering::Relaxed,
    AtomicOrdering::Acquire,
    AtomicOrdering::Release,
    AtomicOrdering::AcquireRelease,
    AtomicOrdering::SequentiallyConsistent,
];

#[test]
fn every_accepted_processor_has_a_canonical_capability_profile() {
    for &processor in KNOWN_PROCESSORS {
        let target = AmdTargetId::parse(processor).unwrap();
        let capabilities = target.capabilities().unwrap();
        assert_eq!(capabilities.target(), target, "{processor}");
        assert!(
            capabilities
                .wavefront_widths()
                .contains(capabilities.default_wavefront_width()),
            "{processor}"
        );
        assert!(
            capabilities.max_lds_bytes_per_workgroup() > 0,
            "{processor}"
        );
        for scope in [
            AtomicScope::Workgroup,
            AtomicScope::Device,
            AtomicScope::System,
        ] {
            assert!(capabilities.atomic_scopes().contains(scope), "{processor}");
        }
        assert_eq!(
            capabilities.cooperative_launch(),
            CapabilitySupport::RequiresRuntimeEvidence,
            "{processor}"
        );
    }
}

#[test]
fn representative_wave_and_lds_profiles_match_llvm_subtarget_facts() {
    let gfx600 = capabilities("gfx600");
    assert_eq!(gfx600.default_wavefront_width(), WavefrontWidth::Wave64);
    assert!(!gfx600.wavefront_widths().contains(WavefrontWidth::Wave32));
    assert!(gfx600.wavefront_widths().contains(WavefrontWidth::Wave64));
    assert_eq!(gfx600.max_lds_bytes_per_workgroup(), 32 * 1024);

    let gfx942 = capabilities("gfx942");
    assert_eq!(gfx942.default_wavefront_width(), WavefrontWidth::Wave64);
    assert!(!gfx942.wavefront_widths().contains(WavefrontWidth::Wave32));
    assert_eq!(gfx942.max_lds_bytes_per_workgroup(), 64 * 1024);

    let gfx1151 = capabilities("gfx1151");
    assert_eq!(gfx1151.default_wavefront_width(), WavefrontWidth::Wave32);
    assert!(gfx1151.wavefront_widths().contains(WavefrontWidth::Wave32));
    assert!(gfx1151.wavefront_widths().contains(WavefrontWidth::Wave64));
    assert_eq!(gfx1151.max_lds_bytes_per_workgroup(), 64 * 1024);

    let gfx950 = capabilities("gfx950");
    assert_eq!(gfx950.max_lds_bytes_per_workgroup(), 160 * 1024);

    let gfx1250 = capabilities("gfx1250");
    assert_eq!(gfx1250.default_wavefront_width(), WavefrontWidth::Wave32);
    assert!(gfx1250.wavefront_widths().contains(WavefrontWidth::Wave32));
    assert!(!gfx1250.wavefront_widths().contains(WavefrontWidth::Wave64));
    assert_eq!(gfx1250.max_lds_bytes_per_workgroup(), 320 * 1024);

    let gfx1310 = capabilities("gfx1310");
    assert!(gfx1310.wavefront_widths().contains(WavefrontWidth::Wave32));
    assert!(gfx1310.wavefront_widths().contains(WavefrontWidth::Wave64));
}

#[test]
fn representative_matrix_families_are_exact_and_fail_closed() {
    let gfx942 = capabilities("gfx942").matrix_instruction_sets();
    assert!(gfx942.contains(MatrixInstructionSet::Mfma));
    assert!(!gfx942.contains(MatrixInstructionSet::ScaledMfmaF8F6F4));
    assert!(!gfx942.contains(MatrixInstructionSet::Wmma128));

    let gfx950 = capabilities("gfx950").matrix_instruction_sets();
    assert!(gfx950.contains(MatrixInstructionSet::Mfma));
    assert!(gfx950.contains(MatrixInstructionSet::ScaledMfmaF8F6F4));

    let gfx1151 = capabilities("gfx1151").matrix_instruction_sets();
    assert!(gfx1151.contains(MatrixInstructionSet::Wmma256));
    assert!(!gfx1151.contains(MatrixInstructionSet::Mfma));

    let gfx1170 = capabilities("gfx1170").matrix_instruction_sets();
    assert!(gfx1170.contains(MatrixInstructionSet::Wmma128));
    assert!(gfx1170.contains(MatrixInstructionSet::Swmma));
    assert!(!gfx1170.contains(MatrixInstructionSet::Wmma256));

    let gfx1251 = capabilities("gfx1251").matrix_instruction_sets();
    assert!(gfx1251.contains(MatrixInstructionSet::Swmma));
    assert!(!gfx1251.contains(MatrixInstructionSet::Wmma128));

    assert!(capabilities("gfx906").matrix_instruction_sets().is_empty());
    assert!(capabilities("gfx1310").matrix_instruction_sets().is_empty());
}

#[test]
fn representative_async_families_are_exact_and_fail_closed() {
    let gfx942 = capabilities("gfx942").async_copy_instruction_sets();
    assert!(gfx942.contains(AsyncCopyInstructionSet::VmemToLds));
    assert!(!gfx942.contains(AsyncCopyInstructionSet::AsyncLoadToLds));

    assert!(
        capabilities("gfx1151")
            .async_copy_instruction_sets()
            .is_empty()
    );

    let gfx1250 = capabilities("gfx1250").async_copy_instruction_sets();
    assert!(gfx1250.contains(AsyncCopyInstructionSet::AsyncLoadToLds));
    assert!(gfx1250.contains(AsyncCopyInstructionSet::AsyncStoreFromLds));
    assert!(!gfx1250.contains(AsyncCopyInstructionSet::VmemToLds));

    let gfx1310 = capabilities("gfx1310").async_copy_instruction_sets();
    assert!(gfx1310.contains(AsyncCopyInstructionSet::AsyncLoadToLds));
    assert!(!gfx1310.contains(AsyncCopyInstructionSet::AsyncStoreFromLds));
}

#[test]
fn gfx942_advanced_capabilities_are_exact_and_conservative() {
    let gfx942 = capabilities("gfx942");
    assert_eq!(
        gfx942.advanced_profile_status(),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(
        gfx942.workgroup_limits_support(),
        AdvancedCapabilityStatus::Supported
    );
    let limits = gfx942.workgroup_limits().unwrap();
    assert_eq!(limits.max_workitems(), 1024);
    for axis in [WorkgroupAxis::X, WorkgroupAxis::Y, WorkgroupAxis::Z] {
        assert_eq!(limits.max_extent(axis), 1024);
    }
    assert!(limits.supports_dimensions(1024, 1, 1));
    assert!(limits.supports_dimensions(32, 8, 4));
    assert!(!limits.supports_dimensions(0, 1, 1));
    assert!(!limits.supports_dimensions(1024, 2, 1));
    assert!(!limits.supports_dimensions(u32::MAX, u32::MAX, 1));
    assert_eq!(
        gfx942.max_wavefronts_per_workgroup(WavefrontWidth::Wave64),
        Some(16)
    );
    assert_eq!(
        gfx942.max_wavefronts_per_workgroup(WavefrontWidth::Wave32),
        None
    );

    for width in [
        AtomicWidth::Bits8,
        AtomicWidth::Bits16,
        AtomicWidth::Bits32,
        AtomicWidth::Bits64,
    ] {
        assert!(gfx942.standard_atomic_widths().contains(width));
    }
    assert!(
        !gfx942
            .standard_atomic_widths()
            .contains(AtomicWidth::Bits128)
    );
    for scope in [
        AtomicScope::Workgroup,
        AtomicScope::Device,
        AtomicScope::System,
    ] {
        assert!(gfx942.standard_atomic_scopes().contains(scope));
    }
    for ordering in [
        AtomicOrdering::Relaxed,
        AtomicOrdering::Acquire,
        AtomicOrdering::Release,
        AtomicOrdering::AcquireRelease,
        AtomicOrdering::SequentiallyConsistent,
    ] {
        assert!(gfx942.standard_atomic_orderings().contains(ordering));
    }
    assert_eq!(
        gfx942.native_split_barriers(),
        AdvancedCapabilityStatus::Unsupported
    );

    for format in [Fp8Format::E4M3Fnuz, Fp8Format::E5M2Fnuz] {
        assert!(gfx942.fp8_formats().contains(format));
        assert_eq!(
            gfx942.fp8_format_support(format),
            AdvancedCapabilityStatus::Supported
        );
    }
    for format in [Fp8Format::E4M3Ocp, Fp8Format::E5M2Ocp] {
        assert!(!gfx942.fp8_formats().contains(format));
        assert_eq!(
            gfx942.fp8_format_support(format),
            AdvancedCapabilityStatus::Unsupported
        );
    }
    for format in [MxFormat::Fp8, MxFormat::Bf8] {
        assert!(!gfx942.mx_formats().contains(format));
        assert_eq!(
            gfx942.mx_format_support(format),
            AdvancedCapabilityStatus::Unsupported
        );
    }

    for family in [
        MfmaFamily::F32FromF16,
        MfmaFamily::F32FromBf16,
        MfmaFamily::F32FromFp8Fnuz,
        MfmaFamily::F32FromBf8Fnuz,
    ] {
        assert!(gfx942.mfma_families().contains(family));
        assert_eq!(
            gfx942.mfma_family_support(family),
            AdvancedCapabilityStatus::Supported
        );
    }
    for family in [MfmaFamily::F64FromF64, MfmaFamily::I32FromI8] {
        assert!(!gfx942.mfma_families().contains(family));
        assert_eq!(
            gfx942.mfma_family_support(family),
            AdvancedCapabilityStatus::Unsupported
        );
    }

    assert_eq!(
        gfx942.device_diagnostic_support(DeviceDiagnosticFeature::Printf),
        AdvancedCapabilityStatus::RequiresRuntimeEvidence
    );
    assert_eq!(
        gfx942.device_diagnostic_support(DeviceDiagnosticFeature::Trap),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(
        gfx942.device_diagnostic_support(DeviceDiagnosticFeature::DebugTrap),
        AdvancedCapabilityStatus::RequiresRuntimeEvidence
    );
    assert_eq!(
        gfx942.device_diagnostic_support(DeviceDiagnosticFeature::ClockCounter),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(
        gfx942.device_diagnostic_support(DeviceDiagnosticFeature::ProfilingMarker),
        AdvancedCapabilityStatus::Unsupported
    );
    assert_eq!(
        gfx942.launch_bounds_support(LaunchBoundsField::MaxWorkgroupSize),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(
        gfx942.launch_bounds_support(LaunchBoundsField::MinWorkgroupsPerComputeUnit),
        AdvancedCapabilityStatus::Unsupported
    );
    assert_eq!(
        gfx942.launch_bounds_metadata_support(LaunchBoundsMetadata::FlatWorkgroupSize),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(
        gfx942.launch_bounds_metadata_support(LaunchBoundsMetadata::WavesPerExecutionUnit),
        AdvancedCapabilityStatus::Supported
    );
}

#[test]
fn gfx942_min_workgroups_is_not_implied_by_waves_per_eu_metadata() {
    let gfx942 = capabilities("gfx942");
    assert_eq!(
        gfx942.launch_bounds_metadata_support(LaunchBoundsMetadata::WavesPerExecutionUnit),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(
        gfx942.launch_bounds_support(LaunchBoundsField::MinWorkgroupsPerComputeUnit),
        AdvancedCapabilityStatus::Unsupported
    );
}

#[test]
fn gfx950_low_precision_and_transpose_capabilities_are_exact() {
    let gfx950 = capabilities("gfx950");
    assert_eq!(
        gfx950.advanced_profile_status(),
        AdvancedCapabilityStatus::Supported
    );
    assert_eq!(gfx950.max_lds_bytes_per_workgroup(), 160 * 1024);
    assert_eq!(
        gfx950.workgroup_limits_support(),
        AdvancedCapabilityStatus::Supported
    );
    let limits = gfx950.workgroup_limits().unwrap();
    assert_eq!(limits.max_workitems(), 1024);
    for axis in [WorkgroupAxis::X, WorkgroupAxis::Y, WorkgroupAxis::Z] {
        assert_eq!(limits.max_extent(axis), 1024);
    }
    assert!(limits.supports_dimensions(1024, 1, 1));
    assert!(limits.supports_dimensions(16, 8, 8));
    assert!(!limits.supports_dimensions(1024, 2, 1));
    assert_eq!(
        gfx950.max_wavefronts_per_workgroup(WavefrontWidth::Wave64),
        Some(16)
    );
    assert!(gfx950.standard_atomic_widths().is_empty());
    assert_eq!(
        gfx950.native_split_barriers(),
        AdvancedCapabilityStatus::Unreviewed
    );

    for format in [Fp8Format::E4M3Ocp, Fp8Format::E5M2Ocp] {
        assert!(gfx950.fp8_formats().contains(format));
        assert_eq!(
            gfx950.fp8_format_support(format),
            AdvancedCapabilityStatus::Supported
        );
    }
    for format in [Fp8Format::E4M3Fnuz, Fp8Format::E5M2Fnuz] {
        assert!(!gfx950.fp8_formats().contains(format));
        assert_eq!(
            gfx950.fp8_format_support(format),
            AdvancedCapabilityStatus::Unreviewed
        );
    }
    for format in [MxFormat::Fp8, MxFormat::Bf8, MxFormat::Fp4] {
        assert!(gfx950.mx_formats().contains(format));
        assert_eq!(
            gfx950.mx_format_support(format),
            AdvancedCapabilityStatus::Supported
        );
    }
    for family in [
        MfmaFamily::F32FromBf16,
        MfmaFamily::F32FromFp8Ocp,
        MfmaFamily::F32FromBf8Ocp,
        MfmaFamily::F32FromFp4Ocp,
    ] {
        assert!(gfx950.mfma_families().contains(family));
        assert_eq!(
            gfx950.mfma_family_support(family),
            AdvancedCapabilityStatus::Supported
        );
    }
    for family in [MfmaFamily::F32FromFp8Fnuz, MfmaFamily::F32FromBf8Fnuz] {
        assert_eq!(
            gfx950.mfma_family_support(family),
            AdvancedCapabilityStatus::Unreviewed
        );
    }

    for instruction in [
        LdsTransposeInstruction::DsReadTr4B64,
        LdsTransposeInstruction::DsReadTr8B64,
        LdsTransposeInstruction::DsReadTr16B64,
    ] {
        assert!(gfx950.lds_transpose_instructions().contains(instruction));
        assert_eq!(
            gfx950.lds_transpose_instruction_support(instruction),
            AdvancedCapabilityStatus::Supported
        );
        assert_eq!(
            capabilities("gfx942").lds_transpose_instruction_support(instruction),
            AdvancedCapabilityStatus::Unsupported
        );
    }
}

#[test]
fn unreviewed_advanced_capability_profiles_fail_closed() {
    for &processor in KNOWN_PROCESSORS {
        if matches!(processor, "gfx942" | "gfx950") {
            continue;
        }
        let capabilities = capabilities(processor);
        assert_eq!(
            capabilities.advanced_profile_status(),
            AdvancedCapabilityStatus::Unreviewed,
            "{processor}"
        );
        assert_eq!(
            capabilities.workgroup_limits_support(),
            AdvancedCapabilityStatus::Unreviewed,
            "{processor}"
        );
        assert!(capabilities.workgroup_limits().is_none(), "{processor}");
        assert!(
            capabilities.standard_atomic_widths().is_empty(),
            "{processor}"
        );
        assert!(
            capabilities.standard_atomic_scopes().is_empty(),
            "{processor}"
        );
        assert!(
            capabilities.standard_atomic_orderings().is_empty(),
            "{processor}"
        );
        assert_eq!(
            capabilities.native_split_barriers(),
            AdvancedCapabilityStatus::Unreviewed,
            "{processor}"
        );
        assert!(capabilities.fp8_formats().is_empty(), "{processor}");
        assert!(capabilities.mx_formats().is_empty(), "{processor}");
        assert!(capabilities.mfma_families().is_empty(), "{processor}");
        assert_eq!(
            capabilities.fp8_format_support(Fp8Format::E4M3Fnuz),
            AdvancedCapabilityStatus::Unreviewed,
            "{processor}"
        );
        assert_eq!(
            capabilities.mx_format_support(MxFormat::Fp8),
            AdvancedCapabilityStatus::Unreviewed,
            "{processor}"
        );
        assert_eq!(
            capabilities.mfma_family_support(MfmaFamily::F32FromF16),
            AdvancedCapabilityStatus::Unreviewed,
            "{processor}"
        );
        for feature in [
            DeviceDiagnosticFeature::Printf,
            DeviceDiagnosticFeature::Trap,
            DeviceDiagnosticFeature::DebugTrap,
            DeviceDiagnosticFeature::ClockCounter,
            DeviceDiagnosticFeature::ProfilingMarker,
        ] {
            assert_eq!(
                capabilities.device_diagnostic_support(feature),
                AdvancedCapabilityStatus::Unreviewed,
                "{processor}: {feature:?}"
            );
        }
        for field in [
            LaunchBoundsField::MaxWorkgroupSize,
            LaunchBoundsField::MinWorkgroupsPerComputeUnit,
        ] {
            assert_eq!(
                capabilities.launch_bounds_support(field),
                AdvancedCapabilityStatus::Unreviewed,
                "{processor}: {field:?}"
            );
        }
        for metadata in [
            LaunchBoundsMetadata::FlatWorkgroupSize,
            LaunchBoundsMetadata::WavesPerExecutionUnit,
        ] {
            assert_eq!(
                capabilities.launch_bounds_metadata_support(metadata),
                AdvancedCapabilityStatus::Unreviewed,
                "{processor}: {metadata:?}"
            );
        }
    }
}

#[test]
fn atomic_non_compare_exchange_matrix_is_exhaustive_and_evidence_correct() {
    let gfx942 = capabilities("gfx942");
    let unreviewed = capabilities("gfx906");
    let mut legalizable = 0;
    let mut runtime_evidence = 0;
    let mut unsupported = 0;
    let mut invalid = 0;
    let mut unreviewed_count = 0;

    for operation in ATOMIC_OPERATIONS {
        if matches!(operation, AtomicOperation::CompareExchange) {
            continue;
        }
        for width in ATOMIC_WIDTHS {
            for address_space in ATOMIC_ADDRESS_SPACES {
                for scope in ATOMIC_SCOPES {
                    for ordering in ATOMIC_ORDERINGS {
                        let query = StandardAtomicQuery::new(
                            operation,
                            width,
                            address_space,
                            scope,
                            ordering,
                        )
                        .expect("only compare-exchange requires a failure ordering");
                        let ordering_is_valid =
                            non_compare_exchange_ordering_is_valid(operation, ordering);
                        let expected_gfx942 = if ordering_is_valid {
                            reviewed_tuple_status(width, address_space, scope)
                        } else {
                            AtomicLegalizability::Invalid
                        };
                        assert_eq!(
                            gfx942.standard_atomic_legalizability(query),
                            expected_gfx942,
                            "gfx942 {operation:?} {width:?} {address_space:?} {scope:?} {ordering:?}"
                        );
                        match expected_gfx942 {
                            AtomicLegalizability::Legalizable => legalizable += 1,
                            AtomicLegalizability::LegalizableWithRuntimeEvidence => {
                                runtime_evidence += 1
                            }
                            AtomicLegalizability::Unsupported => unsupported += 1,
                            AtomicLegalizability::Invalid => invalid += 1,
                            _ => panic!("reviewed gfx942 tuple returned an unexpected status"),
                        }

                        let expected_unreviewed = if ordering_is_valid {
                            AtomicLegalizability::Unreviewed
                        } else {
                            AtomicLegalizability::Invalid
                        };
                        assert_eq!(
                            unreviewed.standard_atomic_legalizability(query),
                            expected_unreviewed,
                            "unreviewed {operation:?} {width:?} {address_space:?} {scope:?} {ordering:?}"
                        );
                        if matches!(expected_unreviewed, AtomicLegalizability::Unreviewed) {
                            unreviewed_count += 1;
                        }
                    }
                }
            }
        }
    }

    assert!(legalizable > 0);
    assert!(runtime_evidence > 0);
    assert!(unsupported > 0);
    assert!(invalid > 0);
    assert!(unreviewed_count > 0);
}

#[test]
fn atomic_compare_exchange_matrix_covers_every_success_failure_pair() {
    let gfx942 = capabilities("gfx942");
    let unreviewed = capabilities("gfx906");
    let mut valid_pairs = 0;
    let mut invalid_pairs = 0;
    let mut legalizable = 0;
    let mut runtime_evidence = 0;
    let mut unsupported = 0;
    let mut unreviewed_count = 0;

    for width in ATOMIC_WIDTHS {
        for address_space in ATOMIC_ADDRESS_SPACES {
            for scope in ATOMIC_SCOPES {
                for success in ATOMIC_ORDERINGS {
                    for failure in ATOMIC_ORDERINGS {
                        let query = StandardAtomicQuery::compare_exchange(
                            width,
                            address_space,
                            scope,
                            success,
                            failure,
                        );
                        let pair_is_valid =
                            compare_exchange_ordering_pair_is_valid(success, failure);
                        let expected_gfx942 = if pair_is_valid {
                            valid_pairs += 1;
                            reviewed_tuple_status(width, address_space, scope)
                        } else {
                            invalid_pairs += 1;
                            AtomicLegalizability::Invalid
                        };
                        assert_eq!(
                            gfx942.standard_atomic_legalizability(query),
                            expected_gfx942,
                            "gfx942 cmpxchg {width:?} {address_space:?} {scope:?} {success:?}/{failure:?}"
                        );
                        if pair_is_valid {
                            match expected_gfx942 {
                                AtomicLegalizability::Legalizable => legalizable += 1,
                                AtomicLegalizability::LegalizableWithRuntimeEvidence => {
                                    runtime_evidence += 1
                                }
                                AtomicLegalizability::Unsupported => unsupported += 1,
                                _ => panic!("valid reviewed CAS tuple had an unexpected status"),
                            }
                        }
                        let expected_unreviewed = if pair_is_valid {
                            unreviewed_count += 1;
                            AtomicLegalizability::Unreviewed
                        } else {
                            AtomicLegalizability::Invalid
                        };
                        assert_eq!(
                            unreviewed.standard_atomic_legalizability(query),
                            expected_unreviewed,
                            "unreviewed cmpxchg {width:?} {address_space:?} {scope:?} {success:?}/{failure:?}"
                        );
                    }
                }
            }
        }
    }

    assert!(valid_pairs > 0);
    assert!(invalid_pairs > 0);
    assert!(legalizable > 0);
    assert!(runtime_evidence > 0);
    assert!(unsupported > 0);
    assert!(unreviewed_count > 0);
    assert!(
        StandardAtomicQuery::new(
            AtomicOperation::CompareExchange,
            AtomicWidth::Bits32,
            AtomicAddressSpace::Global,
            AtomicScope::Device,
            AtomicOrdering::AcquireRelease,
        )
        .is_none()
    );
}

#[test]
fn subword_generic_and_signedness_cases_are_explicit() {
    let gfx942 = capabilities("gfx942");
    for operation in [
        AtomicOperation::FetchNand,
        AtomicOperation::FetchMinSigned,
        AtomicOperation::FetchMinUnsigned,
        AtomicOperation::FetchMaxSigned,
        AtomicOperation::FetchMaxUnsigned,
    ] {
        for width in [AtomicWidth::Bits8, AtomicWidth::Bits16] {
            let query = StandardAtomicQuery::new(
                operation,
                width,
                AtomicAddressSpace::Generic,
                AtomicScope::Device,
                AtomicOrdering::AcquireRelease,
            )
            .unwrap();
            assert_eq!(
                gfx942.standard_atomic_legalizability(query),
                AtomicLegalizability::Legalizable,
                "{operation:?} {width:?}"
            );
        }
    }

    assert_eq!(
        AmdTargetId::parse("gfx999"),
        Err(ParseAmdTargetIdError::UnknownProcessor)
    );
}

#[test]
fn advanced_model_revision_and_identity_are_explicit_without_changing_v1() {
    assert_eq!(
        ADVANCED_CAPABILITY_MODEL_REVISION,
        AdvancedCapabilityModelRevision::V3
    );
    assert_eq!(ADVANCED_CAPABILITY_MODEL_REVISION.get(), 3);
    assert_ne!(
        AdvancedCapabilityModelRevision::V1,
        AdvancedCapabilityModelRevision::V2
    );

    let target = AmdTargetId::parse("gfx942:xnack-").unwrap();
    let pre_change_identity =
        AdvancedCapabilityModelIdentity::for_revision(AdvancedCapabilityModelRevision::V1, target);
    let capabilities = target.capabilities().unwrap();
    assert_eq!(
        capabilities.advanced_model_revision(),
        AdvancedCapabilityModelRevision::V3
    );
    let post_change_identity = capabilities.advanced_model_identity();
    assert_ne!(pre_change_identity, post_change_identity);
    assert_eq!(
        pre_change_identity.revision(),
        AdvancedCapabilityModelRevision::V1
    );
    assert_eq!(
        post_change_identity.revision(),
        AdvancedCapabilityModelRevision::V3
    );
    assert_eq!(pre_change_identity.target(), target);
    assert_eq!(post_change_identity.target(), target);
    assert_eq!(
        pre_change_identity.to_string(),
        "amd-advanced-capability-model-v1{target=gfx942:xnack-}"
    );
    assert_eq!(
        post_change_identity.to_string(),
        "amd-advanced-capability-model-v3{target=gfx942:xnack-}"
    );
    assert_eq!(
        capabilities.to_string(),
        "amd-target-capabilities-v1{target=gfx942:xnack-;default-wave=wave64;waves=[wave64];max-lds-per-workgroup=65536;atomic-scopes=[workgroup,device,system];cooperative-launch=runtime-evidence;matrix=[mfma];async-copy=[vmem-to-lds]}"
    );
}

#[test]
fn explicit_target_id_features_are_bound_into_the_canonical_encoding() {
    let target = AmdTargetId::parse("gfx942:xnack-:sramecc+").unwrap();
    let capabilities = target.capabilities().unwrap();
    let expected = "amd-target-capabilities-v1{target=gfx942:sramecc+:xnack-;default-wave=wave64;waves=[wave64];max-lds-per-workgroup=65536;atomic-scopes=[workgroup,device,system];cooperative-launch=runtime-evidence;matrix=[mfma];async-copy=[vmem-to-lds]}";
    assert_eq!(capabilities.to_string(), expected);

    let mut encoded = String::new();
    capabilities.encode_canonical(&mut encoded).unwrap();
    assert_eq!(encoded, expected);
}

#[test]
fn unknown_unsupported_and_contradictory_features_never_reach_derivation() {
    assert_eq!(
        AmdTargetId::parse("gfx999"),
        Err(ParseAmdTargetIdError::UnknownProcessor)
    );
    assert_eq!(
        AmdTargetId::parse("gfx1151:xnack+"),
        Err(ParseAmdTargetIdError::UnsupportedFeature(
            AmdTargetFeature::Xnack
        ))
    );
    assert_eq!(
        AmdTargetId::parse("gfx942:xnack+:xnack-"),
        Err(ParseAmdTargetIdError::DuplicateFeature(
            AmdTargetFeature::Xnack
        ))
    );
    assert_eq!(
        AmdTargetId::parse("gfx942:wavefrontsize32+"),
        Err(ParseAmdTargetIdError::UnknownFeature)
    );
}

#[test]
fn capability_derivation_error_text_is_stable() {
    assert_eq!(
        CapabilityDerivationError::UnknownProcessor.to_string(),
        "no canonical capabilities exist for this processor"
    );
    assert_eq!(
        CapabilityDerivationError::ContradictoryWavefrontProfile.to_string(),
        "processor has a contradictory wavefront profile"
    );
}

fn non_compare_exchange_ordering_is_valid(
    operation: AtomicOperation,
    ordering: AtomicOrdering,
) -> bool {
    match operation {
        AtomicOperation::Load => matches!(
            ordering,
            AtomicOrdering::Relaxed
                | AtomicOrdering::Acquire
                | AtomicOrdering::SequentiallyConsistent
        ),
        AtomicOperation::Store => matches!(
            ordering,
            AtomicOrdering::Relaxed
                | AtomicOrdering::Release
                | AtomicOrdering::SequentiallyConsistent
        ),
        AtomicOperation::Swap
        | AtomicOperation::FetchAdd
        | AtomicOperation::FetchSub
        | AtomicOperation::FetchAnd
        | AtomicOperation::FetchNand
        | AtomicOperation::FetchOr
        | AtomicOperation::FetchXor
        | AtomicOperation::FetchMinSigned
        | AtomicOperation::FetchMinUnsigned
        | AtomicOperation::FetchMaxSigned
        | AtomicOperation::FetchMaxUnsigned => true,
        AtomicOperation::CompareExchange => false,
        _ => false,
    }
}

fn compare_exchange_ordering_pair_is_valid(
    success: AtomicOrdering,
    failure: AtomicOrdering,
) -> bool {
    match success {
        AtomicOrdering::Relaxed => matches!(failure, AtomicOrdering::Relaxed),
        AtomicOrdering::Acquire => {
            matches!(failure, AtomicOrdering::Relaxed | AtomicOrdering::Acquire)
        }
        AtomicOrdering::Release => matches!(failure, AtomicOrdering::Relaxed),
        AtomicOrdering::AcquireRelease => {
            matches!(failure, AtomicOrdering::Relaxed | AtomicOrdering::Acquire)
        }
        AtomicOrdering::SequentiallyConsistent => matches!(
            failure,
            AtomicOrdering::Relaxed
                | AtomicOrdering::Acquire
                | AtomicOrdering::SequentiallyConsistent
        ),
        _ => false,
    }
}

fn reviewed_tuple_status(
    width: AtomicWidth,
    address_space: AtomicAddressSpace,
    scope: AtomicScope,
) -> AtomicLegalizability {
    if matches!(width, AtomicWidth::Bits128)
        || matches!(
            address_space,
            AtomicAddressSpace::Private | AtomicAddressSpace::Constant
        )
    {
        AtomicLegalizability::Unsupported
    } else if matches!(
        address_space,
        AtomicAddressSpace::Global | AtomicAddressSpace::Generic
    ) && matches!(scope, AtomicScope::System)
    {
        AtomicLegalizability::LegalizableWithRuntimeEvidence
    } else {
        AtomicLegalizability::Legalizable
    }
}

fn capabilities(target: &str) -> fe2o3_amd_target::AmdTargetCapabilities {
    AmdTargetId::parse(target).unwrap().capabilities().unwrap()
}

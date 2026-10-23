const CARGO_VENDOR_DEVICE_MANIFEST_V1: &[u8] =
    include_bytes!("fixtures/fe2o3-device-cargo-vendor-v1.toml");

#[test]
fn cargo_vendor_fixture_matches_actual_sdk_cargo_test_target_roster() {
    let package_root = Path::new(super::REVIEWED_FE2O3_DEVICE_PACKAGE_ROOT)
        .canonicalize()
        .unwrap();
    // Auto-discovered targets can change the vendor manifest without changing Cargo.toml.
    let output =
        std::process::Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
            .args([
                "metadata",
                "--locked",
                "--offline",
                "--no-deps",
                "--format-version=1",
            ])
            .arg("--manifest-path")
            .arg(package_root.join("Cargo.toml"))
            .output()
            .unwrap();
    assert!(
        output.status.success(),
        "Cargo target discovery failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let package = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["name"] == "fe2o3-device")
        .unwrap();
    assert_eq!(
        Path::new(package["manifest_path"].as_str().unwrap())
            .canonicalize()
            .unwrap(),
        package_root.join("Cargo.toml")
    );
    let mut stanzas = Vec::new();
    for target in package["targets"].as_array().unwrap() {
        if !target["kind"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kind| kind == "test")
        {
            continue;
        }
        let path = Path::new(target["src_path"].as_str().unwrap())
            .strip_prefix(&package_root)
            .unwrap()
            .to_str()
            .unwrap();
        stanzas.push(format!(
            "[[test]]\nname = {}\npath = {}\n",
            serde_json::to_string(target["name"].as_str().unwrap()).unwrap(),
            serde_json::to_string(path).unwrap()
        ));
    }
    assert!(!stanzas.is_empty());
    let matches_roster = |manifest: &str| {
        manifest.lines().filter(|line| *line == "[[test]]").count() == stanzas.len()
            && stanzas
                .iter()
                .all(|stanza| manifest.matches(stanza.as_str()).count() == 1)
    };
    let manifest = std::str::from_utf8(CARGO_VENDOR_DEVICE_MANIFEST_V1).unwrap();
    assert!(
        matches_roster(manifest),
        "review the exact Cargo-vendored manifest and source closure after SDK target changes"
    );
    for stanza in &stanzas {
        assert!(!matches_roster(&manifest.replacen(stanza.as_str(), "", 1)));
    }
    assert!(!matches_roster(&format!("{manifest}\n{}", stanzas[0])));
}

fn reviewed_materialization_fixture(vendored: bool) -> ProviderPackageFixture {
    let fixture = ProviderPackageFixture::new();
    fs::remove_dir_all(fixture.source_root()).unwrap();
    let original = Path::new(super::REVIEWED_FE2O3_DEVICE_PACKAGE_ROOT);
    let mut files = Vec::new();
    super::collect_reviewed_source_files(&original.join("src"), &mut files).unwrap();
    assert_eq!(files.len(), 35);
    for file in files {
        let target = fixture.root.join(file.strip_prefix(original).unwrap());
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(file, target).unwrap();
    }
    let manifest = if vendored {
        CARGO_VENDOR_DEVICE_MANIFEST_V1.to_vec()
    } else {
        fs::read(original.join("Cargo.toml")).unwrap()
    };
    fs::write(fixture.root.join("Cargo.toml"), manifest).unwrap();
    fixture
}

fn admit_reviewed_materialization(
    fixture: &ProviderPackageFixture,
) -> Result<super::ReviewedProviderSourceClosureV1, String> {
    reviewed_provider_source_closure_from_definition(
        &fixture.definition(),
        WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        &super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1,
    )
}

// Restore only the two lossless-widening spellings in a historical fixture.
// This does not add either historical closure to the current admission policy.
fn restore_subgroup_tile_from_widening(fixture: &ProviderPackageFixture) {
    let path = fixture.source_root().join("group.rs");
    let mut source = fs::read_to_string(&path).unwrap();
    for (current, previous) in [
        (
            "fn size(&self) -> u64 {\n        N as u64\n    }",
            "fn size(&self) -> u64 {\n        u64::from(N)\n    }",
        ),
        (
            "fn thread_rank(&self) -> u64 {\n        (self.lane % N) as u64\n    }",
            "fn thread_rank(&self) -> u64 {\n        u64::from(self.lane % N)\n    }",
        ),
    ] {
        assert_eq!(source.matches(current).count(), 1);
        source = source.replacen(current, previous, 1);
    }
    fs::write(path, source).unwrap();
}

#[test]
fn subgroup_widening_refresh_replaces_both_previous_materializations() {
    assert_eq!(super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1.len(), 2);
    for (vendored, previous) in [
        (
            false,
            "cc33bc488a6f9db3d07e2078d4026ef57c8190466f9a8bc508ba983bb4238352",
        ),
        (
            true,
            "ab247e81428067d8567b9f69eca5ec4967a4bab8b359a5416ab066034486f24d",
        ),
    ] {
        let fixture = reviewed_materialization_fixture(vendored);
        admit_reviewed_materialization(&fixture).unwrap();
        restore_subgroup_tile_from_widening(&fixture);
        let identity = reviewed_provider_source_closure_identity(
            &fixture.root,
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        )
        .unwrap();
        assert_eq!(identity, digest(previous));
        assert!(!super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1.contains(&identity));
        assert!(admit_reviewed_materialization(&fixture).is_err());
    }
}

// Reconstruct the exact historical package; never admit it as a fallback.
fn remove_fp4_materialization_delta(fixture: &ProviderPackageFixture, vendored: bool) {
    restore_subgroup_tile_from_widening(fixture);
    fs::remove_file(fixture.source_root().join("fp4.rs")).unwrap();
    let path = fixture.source_root().join("lib.rs");
    let mut source = fs::read_to_string(&path).unwrap();
    for addition in [
        "pub mod fp4;\n",
        "pub use fp4::{Fp4E2M1Error, Fp4E2M1Ocp, Fp4E2M1Ocpx8};\n",
    ] {
        assert_eq!(source.matches(addition).count(), 1);
        source = source.replacen(addition, "", 1);
    }
    fs::write(path, source).unwrap();
    if vendored {
        let path = fixture.root.join("Cargo.toml");
        let manifest = fs::read_to_string(&path).unwrap();
        let stanza = "[[test]]\nname = \"fp4_api\"\npath = \"tests/fp4_api.rs\"\n\n";
        assert_eq!(manifest.matches(stanza).count(), 1);
        fs::write(path, manifest.replacen(stanza, "", 1)).unwrap();
    }
}

#[test]
fn fp4_refresh_replaces_both_previous_materializations_and_preserves_atomic_provider() {
    assert_eq!(super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1.len(), 2);
    for (vendored, previous) in [
        (
            false,
            "b1f821474cc4f1dabd6735e86e152c61a72655e4dd599e881c5308599b6b031d",
        ),
        (
            true,
            "ad5a497278b75847aa48ccc7369c51e713540cb3952372bb525a5f0a019a914b",
        ),
    ] {
        let fixture = reviewed_materialization_fixture(vendored);
        let current = admit_reviewed_materialization(&fixture).unwrap().identity;
        let atomic = reviewed_provider_source_closure_from_definition(
            &fixture.source_root().join("atomic.rs"),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            &super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1,
        )
        .unwrap();
        assert_eq!(atomic.identity, current);
        assert!(!super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1.contains(&digest(previous)));
        assert!(
            reviewed_provider_source_closure_from_definition(
                &fixture.source_root().join("atomic.rs"),
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
                &[digest(previous)],
            )
            .is_err()
        );
        remove_fp4_materialization_delta(&fixture, vendored);
        let observed = reviewed_provider_source_closure_identity(
            &fixture.root,
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        )
        .unwrap();
        assert_eq!(observed, digest(previous));
        assert_ne!(observed, current);
        assert!(admit_reviewed_materialization(&fixture).is_err());
    }
}

#[test]
fn execution_documentation_refresh_replaces_both_previous_materializations() {
    for (vendored, previous) in [
        (
            false,
            "09d8b1702b59f7d2a6d11f2c687c4bd5e4d172efde7a051e01c8e8ad7fddb01f",
        ),
        (
            true,
            "542cf2ba9747a6800e23c413fd0c6000ac788e13d40541b9bd6f21cc630cb529",
        ),
    ] {
        let fixture = reviewed_materialization_fixture(vendored);
        let current = admit_reviewed_materialization(&fixture).unwrap().identity;
        // Keep the old execution-comment regression tied to its exact historical tree.
        let current_fixture = fixture;
        let fixture = reviewed_materialization_fixture(vendored);
        remove_fp4_materialization_delta(&fixture, vendored);
        let source_path = fixture.source_root().join("execution.rs");
        let source = fs::read_to_string(&source_path).unwrap();
        let mut historical = source.clone();
        for comment in [
            "    // Authenticated issuance records the total number of work-items here.\n",
            "    // X-fastest linear work-item rank within this workgroup, matching\n",
            "    // group::Workgroup::thread_rank; not launch dimensionality or group index.\n",
        ] {
            assert_eq!(historical.matches(comment).count(), 1);
            historical = historical.replacen(comment, "", 1);
        }
        fs::write(&source_path, historical).unwrap();
        let observed = reviewed_provider_source_closure_identity(
            &fixture.root,
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
        )
        .unwrap();
        assert_eq!(observed, digest(previous));
        assert_ne!(observed, current);
        assert!(admit_reviewed_materialization(&fixture).is_err());
        fs::write(&source_path, source).unwrap();
        assert!(admit_reviewed_materialization(&fixture).is_err());
        assert_eq!(
            admit_reviewed_materialization(&current_fixture)
                .unwrap()
                .identity,
            current
        );
    }
}

#[test]
fn canonical_and_cargo_vendor_materializations_preserve_actual_identities() {
    use sha2::{Digest as _, Sha256};

    let manifest_sha: [u8; 32] = Sha256::digest(CARGO_VENDOR_DEVICE_MANIFEST_V1).into();
    assert_eq!(
        manifest_sha,
        digest("7a4843e05c6b5bb09ad2da1a0439b4ad1fb5e7c0a17b3e07a043d366112b45bf")
    );
    let item = TrustedDeviceItem::WriteOnlyDisjointSliceLen;
    let path = exact_provider_compiler_definition_path_v1(item)
        .unwrap()
        .strip_prefix("fe2o3_device::")
        .unwrap();
    let mut definitions = Vec::new();
    for (vendored, expected) in [
        (false, super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1),
        (
            true,
            super::REVIEWED_SAFE_EXECUTION_CARGO_VENDOR_SOURCE_CLOSURE_V1,
        ),
    ] {
        let fixture = reviewed_materialization_fixture(vendored);
        let admitted = admit_reviewed_materialization(&fixture).unwrap();
        assert_eq!(admitted.identity, expected);
        let other = if vendored {
            super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1
        } else {
            super::REVIEWED_SAFE_EXECUTION_CARGO_VENDOR_SOURCE_CLOSURE_V1
        };
        assert!(
            reviewed_provider_source_closure_from_definition(
                &fixture.definition(),
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
                &[other],
            )
            .is_err()
        );
        assert_eq!(
            admitted.source_root,
            fs::canonicalize(fixture.source_root()).unwrap()
        );
        assert!(!fixture.root.join("Cargo.toml.orig").exists());
        let definition = semantic_definition(path, admitted.identity, [6; 32]);
        validate_reviewed_fe2o3_device_provider_definition_v1(item, &definition).unwrap();
        let mut changed = definition.clone();
        changed.source_closure_identity[0] ^= 1;
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &changed).is_err());
        changed = definition.clone();
        changed.provider.crate_name = "impostor".into();
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &changed).is_err());
        let wrong_path = semantic_definition("wrong::len", admitted.identity, [6; 32]);
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &wrong_path).is_err());
        definitions.push(definition);
    }
    assert_ne!(
        definitions[0].source_closure_identity,
        definitions[1].source_closure_identity
    );
    assert_ne!(
        definitions[0]
            .durable_semantic_identity(item.canonical_path())
            .unwrap(),
        definitions[1]
            .durable_semantic_identity(item.canonical_path())
            .unwrap()
    );
}

#[test]
fn reviewed_materializations_reject_manifest_and_source_mutations() {
    for vendored in [false, true] {
        for mutation in 0..22 {
            let fixture = reviewed_materialization_fixture(vendored);
            admit_reviewed_materialization(&fixture).unwrap();
            match mutation {
                0 => {
                    let path = fixture.root.join("Cargo.toml");
                    let mut bytes = fs::read(&path).unwrap();
                    bytes.push(b'\n');
                    fs::write(path, bytes).unwrap();
                }
                1 => {
                    let path = fixture.root.join("Cargo.toml");
                    let manifest = fs::read_to_string(&path).unwrap();
                    fs::write(path, manifest.replace("fe2o3-macros", "unreviewed-macros")).unwrap();
                }
                2 => fs::write(fixture.definition(), b"// substituted source\n").unwrap(),
                3 => fs::write(fixture.source_root().join("extra.rs"), b"// extra\n").unwrap(),
                4 => fs::remove_file(fixture.source_root().join("thread.rs")).unwrap(),
                5 => fs::rename(
                    fixture.source_root().join("thread.rs"),
                    fixture.source_root().join("renamed.rs"),
                )
                .unwrap(),
                6 => fs::write(fixture.root.join("build.rs"), b"fn main() {}\n").unwrap(),
                7 => {
                    let path = fixture.root.join("Cargo.toml");
                    let manifest = fs::read_to_string(&path).unwrap();
                    fs::write(
                        path,
                        manifest.replace("name = \"fe2o3_device\"", "name = \"unreviewed\""),
                    )
                    .unwrap();
                }
                8 => {
                    let path = fixture.root.join("Cargo.toml");
                    fs::copy(&path, fixture.root.join("Cargo.toml.orig")).unwrap();
                    fs::write(path, b"[package]\nname = 'unreviewed'\n").unwrap();
                }
                9 => fs::write(
                    fixture.source_root().join("physical_entry_v1.rs"),
                    b"// substituted physical provider\n",
                )
                .unwrap(),
                10 => fs::remove_file(fixture.source_root().join("physical_entry_v1.rs")).unwrap(),
                11 => fs::rename(
                    fixture.source_root().join("physical_entry_v1.rs"),
                    fixture.source_root().join("physical_entry_renamed.rs"),
                )
                .unwrap(),
                12 => fs::write(
                    fixture.source_root().join("physical_global_copy_v1.rs"),
                    b"// substituted global-copy provider\n",
                )
                .unwrap(),
                13 => fs::remove_file(fixture.source_root().join("physical_global_copy_v1.rs"))
                    .unwrap(),
                14 => fs::rename(
                    fixture.source_root().join("physical_global_copy_v1.rs"),
                    fixture
                        .source_root()
                        .join("physical_global_copy_renamed.rs"),
                )
                .unwrap(),
                15 => {
                    let path = fixture.source_root().join("context.rs");
                    let mut bytes = fs::read(&path).unwrap();
                    bytes.extend_from_slice(b"\n// unreviewed documentation change\n");
                    fs::write(path, bytes).unwrap();
                }
                16 => fs::remove_file(fixture.source_root().join("context.rs")).unwrap(),
                17 => fs::rename(
                    fixture.source_root().join("context.rs"),
                    fixture.source_root().join("context_renamed.rs"),
                )
                .unwrap(),
                18 => fs::write(
                    fixture.source_root().join("fp4.rs"),
                    b"// substituted FP4 storage source\n",
                )
                .unwrap(),
                19 => fs::remove_file(fixture.source_root().join("fp4.rs")).unwrap(),
                20 => fs::rename(
                    fixture.source_root().join("fp4.rs"),
                    fixture.source_root().join("fp4_renamed.rs"),
                )
                .unwrap(),
                21 => {
                    let path = fixture.source_root().join("lib.rs");
                    let source = fs::read_to_string(&path).unwrap();
                    let registration = "pub mod fp4;\n";
                    assert_eq!(source.matches(registration).count(), 1);
                    fs::write(path, source.replacen(registration, "", 1)).unwrap();
                }
                _ => unreachable!(),
            }
            assert!(
                admit_reviewed_materialization(&fixture).is_err(),
                "accepted mutation {mutation} for vendored={vendored}"
            );
            let identity = reviewed_provider_source_closure_identity(
                &fixture.root,
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            )
            .unwrap();
            let definition = semantic_definition("wave::{impl#4}::current", identity, [6; 32]);
            assert!(super::validate_safe_execution_provider_definition_v1(&definition).is_err());
        }
    }
}

#[test]
fn physical_entry_provider_uses_the_exact_reviewed_materialization() {
    for vendored in [false, true] {
        let fixture = reviewed_materialization_fixture(vendored);
        let admitted = reviewed_provider_source_closure_from_definition(
            &fixture.source_root().join("physical_entry_v1.rs"),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            &super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1,
        )
        .unwrap();
        assert_eq!(
            admitted.identity,
            admit_reviewed_materialization(&fixture).unwrap().identity
        );
        for item in [
            TrustedDeviceItem::AmdGpuPhysicalEntryBeginGfx942,
            TrustedDeviceItem::AmdGpuPhysicalEntryLabelGfx942,
            TrustedDeviceItem::AmdGpuPhysicalEntryStepGfx942,
        ] {
            let path = exact_provider_compiler_definition_path_v1(item)
                .unwrap()
                .strip_prefix("fe2o3_device::")
                .unwrap();
            assert!(path.starts_with("physical_entry_v1::"));
            let definition = semantic_definition(path, admitted.identity, [6; 32]);
            validate_reviewed_fe2o3_device_provider_definition_v1(item, &definition).unwrap();
        }
    }
}

#[test]
fn physical_entry_refresh_preserves_the_exact_diagnostics_leaf() {
    use sha2::{Digest as _, Sha256};
    let source =
        fs::read(Path::new(super::REVIEWED_FE2O3_DEVICE_SOURCE_ROOT).join("diagnostics.rs"))
            .unwrap();
    assert_eq!(source.len(), 7490);
    let actual: [u8; 32] = Sha256::digest(source).into();
    assert_eq!(
        actual,
        digest("803b2178d789c18875b8d3a816af355abfd6a3f8f29cbe7b3262c1ba6e75111d")
    );
}

#[test]
fn materialization_policy_rejects_empty_or_zero_identities() {
    let fixture = reviewed_materialization_fixture(true);
    for policy in [
        &[][..],
        &[[0; 32]][..],
        &[super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURE_V1, [0; 32]][..],
    ] {
        assert!(
            reviewed_provider_source_closure_from_definition(
                &fixture.definition(),
                WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
                policy,
            )
            .is_err()
        );
    }
}
#[test]
fn physical_global_copy_provider_uses_the_exact_reviewed_materialization() {
    for vendored in [false, true] {
        let fixture = reviewed_materialization_fixture(vendored);
        let admitted = reviewed_provider_source_closure_from_definition(
            &fixture.source_root().join("physical_global_copy_v1.rs"),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            &super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1,
        )
        .unwrap();
        assert_eq!(
            admitted.identity,
            admit_reviewed_materialization(&fixture).unwrap().identity
        );
        for item in [
            TrustedDeviceItem::AmdGpuPhysicalGlobalCopyBeginGfx942,
            TrustedDeviceItem::AmdGpuPhysicalGlobalCopyLabelGfx942,
            TrustedDeviceItem::AmdGpuPhysicalGlobalCopyStepGfx942,
        ] {
            let path = exact_provider_compiler_definition_path_v1(item)
                .unwrap()
                .strip_prefix("fe2o3_device::")
                .unwrap();
            assert!(path.starts_with("physical_global_copy_v1::"));
            let definition = semantic_definition(path, admitted.identity, [6; 32]);
            validate_reviewed_fe2o3_device_provider_definition_v1(item, &definition).unwrap();
        }
    }
}

#[test]
fn physical_lds_exchange_provider_uses_the_exact_reviewed_materialization() {
    for vendored in [false, true] {
        let fixture = reviewed_materialization_fixture(vendored);
        let admitted = reviewed_provider_source_closure_from_definition(
            &fixture.source_root().join("physical_lds_exchange_v1.rs"),
            WORKGROUP_SYNC_PROVIDER_SOURCE_CLOSURE_DOMAIN_V1,
            &super::REVIEWED_SAFE_EXECUTION_SOURCE_CLOSURES_V1,
        )
        .unwrap();
        assert_eq!(
            admitted.identity,
            admit_reviewed_materialization(&fixture).unwrap().identity
        );
        for item in [
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeBeginGfx942,
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeLabelGfx942,
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeStepGfx942,
        ] {
            let path = exact_provider_compiler_definition_path_v1(item)
                .unwrap()
                .strip_prefix("fe2o3_device::")
                .unwrap();
            assert!(path.starts_with("physical_lds_exchange_v1::"));
            let definition = semantic_definition(path, admitted.identity, [6; 32]);
            validate_reviewed_fe2o3_device_provider_definition_v1(item, &definition).unwrap();
        }
    }
}

#[test]
fn physical_lds_exchange_provider_rejects_foreign_path_and_stale_source_closure() {
    let fixture = reviewed_materialization_fixture(false);
    let admitted = admit_reviewed_materialization(&fixture).unwrap();
    for (item, foreign) in [
        (
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeBeginGfx942,
            "physical_global_copy_v1::__amdgpu_physical_global_copy_begin_gfx942_v1",
        ),
        (
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeLabelGfx942,
            "physical_global_copy_v1::__amdgpu_physical_global_copy_label_gfx942_v1",
        ),
        (
            TrustedDeviceItem::AmdGpuPhysicalLdsExchangeStepGfx942,
            "physical_global_copy_v1::__amdgpu_physical_global_copy_step_gfx942_v1",
        ),
    ] {
        let foreign = semantic_definition(foreign, admitted.identity, [6; 32]);
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &foreign).is_err());
        let path = exact_provider_compiler_definition_path_v1(item)
            .unwrap()
            .strip_prefix("fe2o3_device::")
            .unwrap();
        let mut stale = semantic_definition(path, admitted.identity, [6; 32]);
        stale.source_closure_identity[0] ^= 1;
        assert!(validate_reviewed_fe2o3_device_provider_definition_v1(item, &stale).is_err());
    }
}

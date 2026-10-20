//! Synthetic inert packaging fixtures, never ELF, root approval or execution.
use super::*;
use fe2o3_build_authority::{
    COMPILER_RUNTIME_MANIFEST_ENTRY_BYTES_V1 as ENTRY,
    COMPILER_RUNTIME_MANIFEST_HEADER_BYTES_V1 as HEADER,
    COMPILER_RUNTIME_MANIFEST_IDENTITY_DOMAIN_V1,
    COMPILER_RUNTIME_MANIFEST_MAX_TOTAL_BYTES_V1 as MAX_TOTAL_BYTES, CompilerClosureV2,
    CompilerRuntimeEntryV1, CompilerRuntimeRoleV1 as Role,
};
use std::{
    fs,
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, PermissionsExt, symlink},
    },
    panic::{AssertUnwindSafe, catch_unwind},
};

pub(crate) struct Fixture {
    _temporary: tempfile::TempDir,
    parent: PathBuf,
    pub(crate) bundle: PathBuf,
    pub(crate) policy: CompilerApprovalPolicyV2,
    pub(crate) manifest: CompilerRuntimeManifestV1,
    contents: Vec<(&'static str, Role, Vec<u8>)>,
}
impl Fixture {
    pub(crate) fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("parent");
        let bundle = parent.join("bundle");
        for path in [
            &parent,
            &bundle,
            &bundle.join("runtime"),
            &bundle.join("runtime/bin"),
            &bundle.join("runtime/lib"),
        ] {
            fs::create_dir(path).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let contents = vec![
            (
                "bin/helper",
                Role::ProofExecutorHelper,
                b"inert helper".to_vec(),
            ),
            ("bin/rustc", Role::Rustc, b"inert rustc".to_vec()),
            ("lib/ld.so", Role::ElfInterpreter, b"inert loader".to_vec()),
            ("lib/libc.so", Role::SharedLibrary, vec![0x5a; CHUNK + 17]),
            (
                "lib/libfe2o3.so",
                Role::CodegenBackend,
                b"inert backend".to_vec(),
            ),
            (
                "lib/libfe2o3_macros.so",
                Role::Fe2o3ProcMacro,
                b"inert macro".to_vec(),
            ),
        ];
        let closure = CompilerClosureV2::new(
            [1; 32],
            [2; 32],
            [3; 32],
            digest(&contents[1].2),
            [5; 32],
            digest(&contents[4].2),
        )
        .unwrap();
        let entries: Vec<_> = contents
            .iter()
            .map(|(path, role, bytes)| CompilerRuntimeEntryV1 {
                role: *role,
                path,
                length: bytes.len() as u64,
                sha256: digest(bytes),
            })
            .collect();
        let manifest =
            CompilerRuntimeManifestV1::new(closure, [9; 32], &entries, |_| Ok::<_, Infallible>(()))
                .unwrap();
        let policy = policy(closure, *manifest.identity());
        for (path, role, bytes) in &contents {
            write(
                &bundle.join("runtime").join(path),
                bytes,
                role.protected_mode(),
            );
        }
        write(&bundle.join(POLICY), policy.canonical_bytes(), 0o444);
        write(&bundle.join(MANIFEST), manifest.canonical_bytes(), 0o444);
        Self {
            _temporary: temporary,
            parent,
            bundle,
            policy,
            manifest,
            contents,
        }
    }

    pub(crate) fn pins(&self) -> ([u8; 32], [u8; 32]) {
        (
            digest(self.policy.canonical_bytes()),
            digest(self.manifest.canonical_bytes()),
        )
    }

    pub(crate) fn verify(&self) -> Result<VerifiedCompilerRuntimeDeploymentV1> {
        let (policy, manifest) = self.pins();
        verify_compiler_runtime_deployment_v1(&self.bundle, policy, manifest)
    }

    pub(crate) fn bind_profile(&mut self, identity: [u8; 32], uid: u32, gid: u32) {
        self.policy = CompilerApprovalPolicyV2::new(
            self.manifest.compiler_closure(),
            identity,
            *self.manifest.identity(),
            1,
            uid,
            gid,
            |_| Ok::<_, Infallible>(()),
        )
        .unwrap();
        write(&self.path(POLICY), self.policy.canonical_bytes(), 0o444);
    }

    fn using(
        &self,
        hook: &mut impl FnMut(&File) -> Result<()>,
    ) -> Result<VerifiedCompilerRuntimeDeploymentV1> {
        let (policy, manifest) = self.pins();
        verify_using(&self.bundle, policy, manifest, hook)
    }

    pub(crate) fn path(&self, path: &str) -> PathBuf {
        self.bundle.join(path)
    }

    fn tracked_origins(&self) -> Vec<(u64, u64)> {
        let mut paths = vec![
            self.parent.clone(),
            self.bundle.clone(),
            self.path("runtime"),
            self.path("runtime/bin"),
            self.path("runtime/lib"),
            self.path(POLICY),
            self.path(MANIFEST),
        ];
        paths.extend(
            self.contents
                .iter()
                .map(|(path, _, _)| self.path(&format!("runtime/{path}"))),
        );
        paths
            .iter()
            .map(|path| identity(&fs::metadata(path).unwrap()))
            .collect()
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn policy(closure: CompilerClosureV2, manifest: [u8; 32]) -> CompilerApprovalPolicyV2 {
    CompilerApprovalPolicyV2::new(closure, [8; 32], manifest, 1, 60001, 60002, |_| {
        Ok::<_, Infallible>(())
    })
    .unwrap()
}
fn write(path: &Path, bytes: &[u8], mode: u32) {
    if path.exists() {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}
fn kind(result: Result<VerifiedCompilerRuntimeDeploymentV1>) -> Kind {
    result.err().expect("fixture must refuse").kind()
}
fn identity(metadata: &fs::Metadata) -> (u64, u64) {
    (metadata.dev(), metadata.ino())
}
fn references(id: (u64, u64)) -> usize {
    let mut count = 0;
    for (index, entry) in fs::read_dir("/proc/self/fd").unwrap().enumerate() {
        assert!(index < 4096, "bounded diagnostic FD census");
        if let Ok(metadata) = fs::metadata(entry.unwrap().path()) {
            count += usize::from(identity(&metadata) == id);
        }
    }
    count
}

#[test]
fn canonical_inert_bundle_retains_exact_sealed_manifest_order() {
    let fixture = Fixture::new();
    let verified = fixture.verify().unwrap();
    assert_eq!(verified.policy(), &fixture.policy);
    assert_eq!(verified.manifest(), &fixture.manifest);
    assert!(!verified.policy().grants_authority());
    assert!(!verified.manifest().grants_authority());
    assert_eq!(verified.files.len(), 6);
    for (file, entry) in verified.files.iter().zip(verified.manifest().entries()) {
        validate_sealed(
            file,
            entry.role.protected_mode(),
            entry.length,
            entry.sha256,
        )
        .unwrap();
        assert!(rustix::io::pwrite(file, b"x", 0).is_err());
        assert!(file.set_len(entry.length - 1).is_err());
        assert!(file.set_len(entry.length + 1).is_err());
    }
    validate_sealed(
        &verified.policy_file,
        0o444,
        POLICY_BYTES as u64,
        fixture.pins().0,
    )
    .unwrap();
    validate_sealed(
        &verified.manifest_file,
        0o444,
        fixture.manifest.canonical_bytes().len() as u64,
        fixture.pins().1,
    )
    .unwrap();
}

#[test]
fn shared_fixture_rebinds_only_inert_profile_and_helper_configuration() {
    let mut fixture = Fixture::new();
    let old = fixture.pins();
    fixture.bind_profile([97; 32], 61001, 61002);
    assert_ne!(fixture.pins().0, old.0);
    assert_eq!(fixture.pins().1, old.1);
    let verified = fixture.verify().unwrap();
    assert_eq!(verified.policy().client_profile_identity(), &[97; 32]);
    assert_eq!(verified.policy().proof_helper_uid(), 61001);
    assert_eq!(verified.policy().proof_helper_gid(), 61002);
    assert!(!verified.policy().grants_authority());
}

#[test]
fn sealed_copies_survive_source_mutation_and_tree_removal() {
    let fixture = Fixture::new();
    let verified = fixture.verify().unwrap();
    for (path, role, bytes) in &fixture.contents {
        write(
            &fixture.path(&format!("runtime/{path}")),
            &vec![0; bytes.len()],
            role.protected_mode(),
        );
    }
    write(&fixture.path(POLICY), &[0; POLICY_BYTES], 0o444);
    fs::remove_dir_all(&fixture.bundle).unwrap();
    for (file, entry) in verified.files.iter().zip(verified.manifest().entries()) {
        validate_sealed(
            file,
            entry.role.protected_mode(),
            entry.length,
            entry.sha256,
        )
        .unwrap();
    }
    validate_sealed(
        &verified.policy_file,
        0o444,
        POLICY_BYTES as u64,
        fixture.pins().0,
    )
    .unwrap();
    validate_sealed(
        &verified.manifest_file,
        0o444,
        fixture.manifest.canonical_bytes().len() as u64,
        fixture.pins().1,
    )
    .unwrap();
}

#[test]
fn external_pins_are_raw_sha256_not_record_identities() {
    let fixture = Fixture::new();
    let (p, m) = fixture.pins();
    for (p, m) in [
        (digest(b"wrong"), m),
        (p, digest(b"wrong")),
        (*fixture.policy.identity(), m),
        (p, *fixture.manifest.identity()),
    ] {
        assert_eq!(
            kind(verify_compiler_runtime_deployment_v1(&fixture.bundle, p, m)),
            Kind::ManifestMismatch
        );
    }
}

#[test]
fn externally_repinned_malformed_records_still_fail_shared_codecs() {
    for (name, offset) in [
        (POLICY, 0),
        (POLICY, 28),
        (MANIFEST, 0),
        (MANIFEST, 22),
        (MANIFEST, HEADER + 2),
    ] {
        let fixture = Fixture::new();
        let mut bytes = fs::read(fixture.path(name)).unwrap();
        bytes[offset] ^= 0xff;
        write(&fixture.path(name), &bytes, 0o444);
        let (mut p, mut m) = fixture.pins();
        if name == POLICY {
            p = digest(&bytes);
        } else {
            m = digest(&bytes);
        }
        assert_eq!(
            kind(verify_compiler_runtime_deployment_v1(&fixture.bundle, p, m)),
            Kind::InvalidManifest
        );
    }
}

#[test]
fn policy_binds_manifest_identity_and_full_closure_not_only_rustc_backend() {
    let fixture = Fixture::new();
    let original = fixture.manifest.compiler_closure();
    let different = CompilerClosureV2::new(
        original.cargo_executable_sha256(),
        original.cargo_binding_trampoline_sha256(),
        original.cargo_fe2o3_binding_wrapper_sha256(),
        original.rustc_executable_sha256(),
        [23; 32],
        original.codegen_backend_sha256(),
    )
    .unwrap();
    for policy in [
        policy(original, [42; 32]),
        policy(different, *fixture.manifest.identity()),
    ] {
        write(&fixture.path(POLICY), policy.canonical_bytes(), 0o444);
        assert_eq!(
            kind(verify_compiler_runtime_deployment_v1(
                &fixture.bundle,
                digest(policy.canonical_bytes()),
                fixture.pins().1
            )),
            Kind::ContentMismatch
        );
    }
}

#[test]
fn exact_file_hashes_lengths_modes_and_single_links_are_required() {
    for change in 0..6 {
        let fixture = Fixture::new();
        let path = fixture.path("runtime/lib/libc.so");
        let mut bytes = fs::read(&path).unwrap();
        match change {
            0 => {
                bytes[CHUNK] ^= 1;
                write(&path, &bytes, 0o444);
            }
            1 => {
                bytes.pop();
                write(&path, &bytes, 0o444);
            }
            2 => {
                bytes.push(1);
                write(&path, &bytes, 0o444);
            }
            3 => fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap(),
            4 => fs::set_permissions(&path, fs::Permissions::from_mode(0o555)).unwrap(),
            5 => fs::hard_link(&path, fixture.parent.join("alias")).unwrap(),
            _ => unreachable!(),
        }
        assert_eq!(
            kind(fixture.verify()),
            if change == 0 {
                Kind::ContentMismatch
            } else {
                Kind::InvalidMetadata
            }
        );
    }
    for name in [POLICY, MANIFEST] {
        let fixture = Fixture::new();
        fs::hard_link(fixture.path(name), fixture.parent.join("alias")).unwrap();
        assert_eq!(kind(fixture.verify()), Kind::InvalidMetadata);
    }
}

#[test]
fn exact_inventory_rejects_missing_and_extra_files_and_directories() {
    for name in [POLICY, MANIFEST, "runtime/bin/rustc"] {
        let fixture = Fixture::new();
        fs::remove_file(fixture.path(name)).unwrap();
        assert!(fixture.verify().is_err());
    }
    for name in ["runtime", "runtime/lib"] {
        let fixture = Fixture::new();
        fs::remove_dir_all(fixture.path(name)).unwrap();
        assert!(fixture.verify().is_err());
    }
    for (name, directory) in [
        ("extra", false),
        ("extra", true),
        ("runtime/empty", true),
        ("runtime/lib/extra", false),
    ] {
        let fixture = Fixture::new();
        if directory {
            fs::create_dir(fixture.path(name)).unwrap();
        } else {
            write(&fixture.path(name), b"extra", 0o444);
        }
        assert_eq!(kind(fixture.verify()), Kind::InvalidInventory);
    }
}

#[test]
fn canonical_paths_reject_symlinks_at_root_parent_directory_and_file() {
    for name in ["runtime/lib", "runtime/bin/rustc", POLICY] {
        let fixture = Fixture::new();
        let path = fixture.path(name);
        let moved = fixture.parent.join("moved");
        fs::rename(&path, &moved).unwrap();
        symlink(&moved, &path).unwrap();
        assert!(fixture.verify().is_err());
    }
    let fixture = Fixture::new();
    let alias = fixture.parent.join("alias");
    symlink(&fixture.bundle, &alias).unwrap();
    assert!(
        verify_compiler_runtime_deployment_v1(&alias, fixture.pins().0, fixture.pins().1).is_err()
    );
    let alias = fixture._temporary.path().join("parent-alias");
    symlink(&fixture.parent, &alias).unwrap();
    assert!(
        verify_compiler_runtime_deployment_v1(
            &alias.join("bundle"),
            fixture.pins().0,
            fixture.pins().1
        )
        .is_err()
    );
    for path in [
        fixture.parent.join("../parent/bundle"),
        fixture.parent.join("./bundle"),
        fixture.bundle.join(""),
    ] {
        assert!(
            verify_compiler_runtime_deployment_v1(&path, fixture.pins().0, fixture.pins().1)
                .is_err(),
            "{path:?}"
        );
    }
}

#[test]
fn root_relative_ownership_directory_modes_and_xattrs_are_enforced() {
    for name in ["", "runtime", "runtime/lib", POLICY, MANIFEST] {
        let fixture = Fixture::new();
        fs::set_permissions(fixture.path(name), fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(kind(fixture.verify()), Kind::InvalidMetadata);
    }
    for name in ["", "runtime/lib", POLICY, MANIFEST, "runtime/bin/rustc"] {
        let fixture = Fixture::new();
        let path = fixture.path(name);
        let mode = fs::metadata(&path).unwrap().mode() & 0o7777;
        if path.is_file() {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        let file = File::open(&path).unwrap();
        rustix::fs::fsetxattr(
            &file,
            "user.fe2o3-runtime-test",
            b"x",
            rustix::fs::XattrFlags::empty(),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
        assert_eq!(kind(fixture.verify()), Kind::ForbiddenAttributes);
    }
    let fixture = Fixture::new();
    let mut root = Root::open(&fixture.bundle).unwrap();
    let file = open_beneath(&root.file, POLICY, false).unwrap();
    // Negative metadata control only; no alternate successful owner constructor.
    root.initial.uid ^= 1;
    assert!(
        source_metadata(
            &file,
            &root,
            0o444,
            POLICY_BYTES as u64,
            Some(POLICY_BYTES as u64)
        )
        .is_err()
    );
}

#[test]
fn file_directory_prefix_collision_is_not_an_additional_roster() {
    let fixture = Fixture::new();
    let mut entries: Vec<_> = fixture.manifest.entries().collect();
    entries.push(CompilerRuntimeEntryV1 {
        role: Role::SharedLibrary,
        path: "lib/libc.so/child",
        length: 1,
        sha256: [17; 32],
    });
    entries.sort_by_key(|e| e.path);
    let manifest = CompilerRuntimeManifestV1::new(
        fixture.manifest.compiler_closure(),
        [9; 32],
        &entries,
        |_| Ok::<_, Infallible>(()),
    )
    .unwrap();
    assert!(matches!(roster(&manifest), Err(e) if e.kind() == Kind::InvalidInventory));
    let policy = policy(manifest.compiler_closure(), *manifest.identity());
    write(&fixture.path(MANIFEST), manifest.canonical_bytes(), 0o444);
    write(&fixture.path(POLICY), policy.canonical_bytes(), 0o444);
    assert_eq!(
        kind(verify_compiler_runtime_deployment_v1(
            &fixture.bundle,
            digest(policy.canonical_bytes()),
            digest(manifest.canonical_bytes())
        )),
        Kind::InvalidInventory
    );
}

#[test]
fn final_source_content_revalidation_refuses_equal_metadata_snapshot() {
    let fixture = Fixture::new();
    let root = Root::open(&fixture.bundle).unwrap();
    let original = b"inert rustc";
    let (sealed, mut source) = copy_source(
        &root,
        "runtime/bin/rustc",
        0o555,
        Some(original.len() as u64),
        MAX_FILE_BYTES,
        digest(original),
        Kind::ContentMismatch,
    )
    .unwrap();
    write(&fixture.path("runtime/bin/rustc"), b"other rustc", 0o555);
    // Model a coarse timestamp tick deterministically, without sleeping or
    // constructing a public verified owner from the mutated source.
    source.initial = file_snapshot(&source.file).unwrap();
    source.revalidate(&root).unwrap();
    assert!(source.revalidate_content(&root).is_err());
    validate_sealed(&sealed, 0o555, original.len() as u64, digest(original)).unwrap();
}

#[test]
fn source_path_and_directory_replacement_after_sealing_refuse() {
    for change in 0..5 {
        let fixture = Fixture::new();
        let mut turns = 0;
        let result = fixture.using(&mut |_| {
            turns += 1;
            if turns == fixture.contents.len() + 2 {
                match change {
                    0 => write(&fixture.path("runtime/bin/rustc"), b"other rustc", 0o555),
                    1 => {
                        let path = fixture.path("runtime/bin/rustc");
                        let bytes = fs::read(&path).unwrap();
                        fs::remove_file(&path).unwrap();
                        write(&path, &bytes, 0o555);
                    }
                    2 => {
                        fs::rename(fixture.path("runtime/lib"), fixture.parent.join("old-lib"))
                            .unwrap();
                        fs::create_dir(fixture.path("runtime/lib")).unwrap();
                    }
                    3 => {
                        fs::rename(&fixture.bundle, fixture.parent.join("old-bundle")).unwrap();
                        fs::create_dir(&fixture.bundle).unwrap();
                    }
                    4 => {
                        fs::rename(
                            &fixture.parent,
                            fixture._temporary.path().join("old-parent"),
                        )
                        .unwrap();
                        fs::create_dir(&fixture.parent).unwrap();
                    }
                    _ => unreachable!(),
                }
            }
            Ok(())
        });
        assert_eq!(turns, fixture.contents.len() + 2);
        assert!(result.is_err(), "mutation {change}");
    }
}

#[test]
fn every_sealed_copy_checkpoint_closes_all_custody_on_error_and_unwind() {
    for unwind in [false, true] {
        for stop in 1..=8 {
            let fixture = Fixture::new();
            let originals = fixture.tracked_origins();
            let baseline: Vec<_> = originals.iter().map(|id| references(*id)).collect();
            let mut sealed = Vec::new();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                fixture.using(&mut |file| {
                    sealed.push((file.as_raw_fd(), identity(&file.metadata().unwrap())));
                    if sealed.len() == stop {
                        if unwind {
                            panic!("runtime bundle copy checkpoint");
                        }
                        return Err(invalid(
                            Kind::InjectedFailure,
                            "runtime bundle copy checkpoint",
                        ));
                    }
                    Ok(())
                })
            }));
            assert_eq!(sealed.len(), stop);
            if unwind {
                assert_eq!(
                    outcome.err().unwrap().downcast_ref::<&str>(),
                    Some(&"runtime bundle copy checkpoint")
                );
            } else {
                assert_eq!(kind(outcome.unwrap()), Kind::InjectedFailure);
            }
            for (fd, id) in sealed {
                assert_eq!(references(id), 0, "leaked sealed source fd {fd}");
            }
            for (id, count) in originals.into_iter().zip(baseline) {
                assert_eq!(references(id), count);
            }
            assert!(
                fixture.verify().is_ok(),
                "source bundle is not consumed on refusal"
            );
        }
    }
}

#[test]
fn short_or_oversized_records_and_code_refuse_without_large_allocation() {
    for (name, size) in [
        (POLICY, POLICY_BYTES as u64 - 1),
        (POLICY, POLICY_BYTES as u64 + 1),
        (MANIFEST, 0),
        (MANIFEST, MANIFEST_BYTES as u64 + 1),
        ("runtime/bin/rustc", MAX_FILE_BYTES + 1),
    ] {
        let fixture = Fixture::new();
        let path = fixture.path(name);
        let mode = fs::metadata(&path).unwrap().mode() & 0o7777;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(size)
            .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
        assert!(fixture.verify().is_err());
    }
    let fixture = Fixture::new();
    let file = File::open(fixture.path("runtime/bin/rustc")).unwrap();
    let n = file.metadata().unwrap().len();
    assert!(stream_hash(&file, n + 1, None).is_err());
    assert!(stream_hash(&file, n - 1, None).is_err());
    assert!(Root::open(Path::new(&"a".repeat(MAX_BUNDLE_PATH + 1))).is_err());
}

fn reseal_manifest(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut hash = Sha256::new();
    hash.update(COMPILER_RUNTIME_MANIFEST_IDENTITY_DOMAIN_V1);
    hash.update((end as u64).to_le_bytes());
    hash.update(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash.finalize());
}

#[test]
fn shared_decoder_refuses_resealed_duplicate_traversal_and_excessive_lengths() {
    for change in 0..4 {
        let fixture = Fixture::new();
        let mut bytes = fixture.manifest.canonical_bytes().to_vec();
        match change {
            0 => {
                let first = bytes[HEADER..HEADER + ENTRY].to_vec();
                bytes[HEADER + ENTRY..HEADER + 2 * ENTRY].copy_from_slice(&first);
            }
            1 => {
                bytes[HEADER + 4..HEADER + 6].copy_from_slice(&9u16.to_le_bytes());
                bytes[HEADER + 48..HEADER + ENTRY].fill(0);
                bytes[HEADER + 48..HEADER + 57].copy_from_slice(b"../helper");
            }
            2 => {
                bytes[HEADER + 8..HEADER + 16].copy_from_slice(&(MAX_FILE_BYTES + 1).to_le_bytes())
            }
            3 => {
                for slot in bytes[HEADER..HEADER + 6 * ENTRY].chunks_exact_mut(ENTRY) {
                    slot[8..16].copy_from_slice(&MAX_FILE_BYTES.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        reseal_manifest(&mut bytes);
        write(&fixture.path(MANIFEST), &bytes, 0o444);
        assert_eq!(
            kind(verify_compiler_runtime_deployment_v1(
                &fixture.bundle,
                fixture.pins().0,
                digest(&bytes)
            )),
            Kind::InvalidManifest
        );
    }
}

#[test]
fn maximum_manifest_roster_is_bounded_and_enumeration_refuses_one_extra() {
    let fixture = Fixture::new();
    let mut entries: Vec<_> = fixture.manifest.entries().collect();
    let paths: Vec<_> = (entries.len()..MAX_FILES)
        .map(|i| format!("z{i:03}/{}file", "d/".repeat(MAX_COMPONENTS - 2)))
        .collect();
    entries.extend(paths.iter().map(|path| CompilerRuntimeEntryV1 {
        role: Role::SharedLibrary,
        path,
        length: 1,
        sha256: [55; 32],
    }));
    let manifest = CompilerRuntimeManifestV1::new(
        fixture.manifest.compiler_closure(),
        [9; 32],
        &entries,
        |_| Ok::<_, Infallible>(()),
    )
    .unwrap();
    assert_eq!(manifest.canonical_bytes().len(), MANIFEST_BYTES);
    let expected = roster(&manifest).unwrap();
    assert!(expected.values().map(BTreeSet::len).sum::<usize>() <= MAX_OBJECTS);
    assert!(expected.len() <= MAX_OBJECTS);
    assert_eq!(MAX_FILE_BYTES, 1024 * 1024 * 1024);
    assert_eq!(MAX_TOTAL_BYTES, 4 * MAX_FILE_BYTES);
    assert_eq!(CHUNK, 65536);
    let root = Root::open(&fixture.bundle).unwrap();
    assert!(check_children(&root.file, &vec!["x"; MAX_OBJECTS + 1]).is_err());
    fs::create_dir(fixture.path("extra")).unwrap();
    assert!(check_children(&root.file, &[MANIFEST, POLICY, "runtime"]).is_err());
}

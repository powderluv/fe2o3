//! Pinned compiler input packaging, not root approval or executable admission.
use crate::{
    DeploymentVerificationErrorKindV1 as Kind, DeploymentVerificationErrorV1 as Error, FileSpecV1,
    ObjectSnapshotV1, changed, invalid, io_error, open_beneath, open_bundle_root, snapshot,
    validate_directory, validate_source_metadata,
};
use fe2o3_build_authority::{
    COMPILER_APPROVAL_POLICY_BYTES_V2 as POLICY_BYTES,
    COMPILER_RUNTIME_MANIFEST_MAX_BYTES_V1 as MANIFEST_BYTES,
    COMPILER_RUNTIME_MANIFEST_MAX_ENTRIES_V1 as MAX_FILES,
    COMPILER_RUNTIME_MANIFEST_MAX_FILE_BYTES_V1 as MAX_FILE_BYTES,
    COMPILER_RUNTIME_MANIFEST_MAX_PATH_COMPONENTS_V1 as MAX_COMPONENTS, CompilerApprovalPolicyV2,
    CompilerRuntimeManifestV1,
};
use rustix::fs::{
    FileType, MemfdFlags, Mode, OFlags, SealFlags, fcntl_add_seals, fcntl_get_seals, fstat,
    memfd_create, openat,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    fs::File,
    path::{Component, Path, PathBuf},
};

const POLICY: &str = "policy-v2";
const MANIFEST: &str = "compiler-runtime-manifest-v1";
const CHUNK: usize = 64 * 1024;
// Two records, runtime/, and at most sixteen path components per code file.
const MAX_OBJECTS: usize = 3 + MAX_FILES * MAX_COMPONENTS;
const MAX_BUNDLE_PATH: usize = 4096;
const SEALS: SealFlags = SealFlags::WRITE
    .union(SealFlags::GROW)
    .union(SealFlags::SHRINK)
    .union(SealFlags::SEAL);
type Result<T> = std::result::Result<T, Error>;
type Roster = BTreeMap<String, BTreeSet<String>>;

/// Move-only sealed copies of a caller-pinned compiler input bundle.
///
/// This value grants no compiler, installation or execution authority. Role
/// labels and hashes do not validate ELF files, resolve their dependencies, or
/// establish approval of a release. The installer must independently establish
/// root ownership, immutable backing and the exact production profile binding.
/// No public constructor imports a descriptor and no public API exposes one.
/// Original source paths are not needed after verification returns successfully.
///
/// ```compile_fail
/// use fe2o3_compiler_execution_deployment::VerifiedCompilerRuntimeDeploymentV1;
/// fn duplicate(value: VerifiedCompilerRuntimeDeploymentV1) { let _ = value.clone(); }
/// ```
/// ```compile_fail
/// use fe2o3_compiler_execution_deployment::VerifiedCompilerRuntimeDeploymentV1;
/// fn descriptor<T: std::os::fd::AsFd>() {}
/// descriptor::<VerifiedCompilerRuntimeDeploymentV1>();
/// ```
pub struct VerifiedCompilerRuntimeDeploymentV1 {
    pub(super) policy: CompilerApprovalPolicyV2,
    pub(super) manifest: CompilerRuntimeManifestV1,
    pub(super) policy_file: File,
    pub(super) manifest_file: File,
    /// Sealed code files in the exact canonical manifest order.
    pub(super) files: Vec<File>,
}

impl VerifiedCompilerRuntimeDeploymentV1 {
    /// Borrow inert policy data; this is not fixed-origin compiler approval.
    pub const fn policy(&self) -> &CompilerApprovalPolicyV2 {
        &self.policy
    }

    /// Borrow inert inventory data, not proof of a complete executable closure.
    pub const fn manifest(&self) -> &CompilerRuntimeManifestV1 {
        &self.manifest
    }
}

/// Verify exactly policy-v2, compiler-runtime-manifest-v1 and runtime/{paths}.
///
/// Both external pins are raw SHA256 of complete record files, not the records'
/// domain-separated identities. The existing codecs enforce their schema bounds;
/// the complete compiler closure and manifest identity must agree across records.
/// Directories are mode 0700; records are 0444 and code has its closed role mode.
/// All contents share the bundle root's UID/GID, with no xattrs, symlinks,
/// hardlinks, extra objects or subordinate mounts. Offline sources need not be
/// owned by UID 0 or already immutable: only sealed copies escape this verifier.
///
/// The bundle pathname must be canonical, non-root and at most 4096 bytes. Its
/// containing directory and the complete source tree must remain stable during
/// verification. Descriptor/path rechecks detect observed changes, not hostile
/// privileged filesystem manipulation. Reads/writes are bounded single attempts
/// per 64 KiB chunk, with no whole-code-file heap buffer. The largest successful
/// sealed code set is the manifest's existing 4 GiB cap, plus the two records;
/// this API neither allocates a resource account nor grants execution authority.
pub fn verify_compiler_runtime_deployment_v1(
    bundle: &Path,
    policy_sha256: [u8; 32],
    manifest_sha256: [u8; 32],
) -> Result<VerifiedCompilerRuntimeDeploymentV1> {
    verify_using(bundle, policy_sha256, manifest_sha256, &mut |_| Ok(()))
}

// The sole production call supplies a no-op. Private tests interrupt the same
// engine after a copy is sealed, without alternate validation or source owners.
fn verify_using(
    bundle: &Path,
    policy_sha256: [u8; 32],
    manifest_sha256: [u8; 32],
    checkpoint: &mut impl FnMut(&File) -> Result<()>,
) -> Result<VerifiedCompilerRuntimeDeploymentV1> {
    let root = Root::open(bundle)?;
    check_children(&root.file, &[MANIFEST, POLICY, "runtime"])?;
    let mut sources = Vec::with_capacity(MAX_FILES + 2);
    let (policy_file, source) = copy_source(
        &root,
        POLICY,
        0o444,
        Some(POLICY_BYTES as u64),
        POLICY_BYTES as u64,
        policy_sha256,
        Kind::ManifestMismatch,
    )?;
    sources.push(source);
    checkpoint(&policy_file)?;
    let mut policy_bytes = [0; POLICY_BYTES];
    read_exact(&policy_file, &mut policy_bytes, 0)?;
    let policy = CompilerApprovalPolicyV2::decode(&policy_bytes, |_| Ok::<_, Infallible>(()))
        .map_err(|e| invalid(Kind::InvalidManifest, format!("compiler policy: {e}")))?;

    let (manifest_file, source) = copy_source(
        &root,
        MANIFEST,
        0o444,
        None,
        MANIFEST_BYTES as u64,
        manifest_sha256,
        Kind::ManifestMismatch,
    )?;
    let length = source.initial.byte_len as usize;
    sources.push(source);
    checkpoint(&manifest_file)?;
    let mut manifest_bytes = [0; MANIFEST_BYTES];
    read_exact(&manifest_file, &mut manifest_bytes[..length], 0)?;
    let manifest =
        CompilerRuntimeManifestV1::decode(&manifest_bytes[..length], |_| Ok::<_, Infallible>(()))
            .map_err(|e| {
            invalid(
                Kind::InvalidManifest,
                format!("compiler runtime manifest: {e}"),
            )
        })?;
    if policy.compiler_closure() != manifest.compiler_closure()
        || policy.runtime_manifest_identity() != manifest.identity()
    {
        return Err(invalid(
            Kind::ContentMismatch,
            "compiler policy/inventory binding differs",
        ));
    }

    let roster = roster(&manifest)?;
    let directories = retain_directories(&root, &roster)?;
    let mut files = Vec::with_capacity(manifest.entries().len());
    for entry in manifest.entries() {
        let path = format!("runtime/{}", entry.path);
        let (file, source) = copy_source(
            &root,
            &path,
            entry.role.protected_mode(),
            Some(entry.length),
            MAX_FILE_BYTES,
            entry.sha256,
            Kind::ContentMismatch,
        )?;
        sources.push(source);
        checkpoint(&file)?;
        files.push(file);
    }
    for source in &sources {
        source.revalidate_content(&root)?;
    }
    for directory in &directories {
        directory.revalidate(&root, &roster)?;
    }
    check_children(&root.file, &[MANIFEST, POLICY, "runtime"])?;
    root.revalidate()?;
    Ok(VerifiedCompilerRuntimeDeploymentV1 {
        policy,
        manifest,
        policy_file,
        manifest_file,
        files,
    })
}

struct Root {
    path: PathBuf,
    parent_path: PathBuf,
    parent: File,
    parent_initial: ObjectSnapshotV1,
    file: File,
    initial: ObjectSnapshotV1,
}
impl Root {
    fn open(path: &Path) -> Result<Self> {
        if path.as_os_str().len() > MAX_BUNDLE_PATH
            || path.file_name().is_none()
            || path
                .components()
                .any(|c| matches!(c, Component::CurDir | Component::ParentDir))
            || path.components().collect::<PathBuf>().as_os_str() != path.as_os_str()
        {
            return Err(invalid(
                Kind::InvalidMetadata,
                "noncanonical compiler bundle path",
            ));
        }
        let parent_path = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = open_bundle_root(parent_path)?;
        let parent_initial = file_snapshot(&parent)?;
        let file = open_bundle_root(path)?;
        let initial = validate_directory(&file, None, "compiler bundle root")?;
        Ok(Self {
            path: path.to_owned(),
            parent_path: parent_path.to_owned(),
            parent,
            parent_initial,
            file,
            initial,
        })
    }

    fn revalidate(&self) -> Result<()> {
        let owner = Some((self.initial.uid, self.initial.gid));
        if validate_directory(&self.file, owner, "retained compiler bundle root")? != self.initial
            || file_snapshot(&self.parent)? != self.parent_initial
            || file_snapshot(&open_bundle_root(&self.parent_path)?)? != self.parent_initial
            || validate_directory(
                &open_bundle_root(&self.path)?,
                owner,
                "current compiler bundle root",
            )? != self.initial
        {
            return Err(changed(
                "compiler bundle root or containing directory changed",
            ));
        }
        Ok(())
    }
}

fn file_snapshot(file: &File) -> Result<ObjectSnapshotV1> {
    fstat(file)
        .map(|stat| snapshot(&stat))
        .map_err(|e| io_error("inspect compiler bundle object", e))
}

fn source_metadata(
    file: &File,
    root: &Root,
    mode: u32,
    maximum: u64,
    length: Option<u64>,
) -> Result<ObjectSnapshotV1> {
    let metadata = validate_source_metadata(
        file,
        root.initial,
        FileSpecV1 {
            source: "compiler runtime input",
            install: "",
            mode,
            max_bytes: maximum,
        },
        None,
    )?;
    if length.is_some_and(|n| metadata.byte_len != n) {
        return Err(invalid(
            Kind::InvalidMetadata,
            "compiler bundle file length differs",
        ));
    }
    Ok(metadata)
}

struct Source {
    path: String,
    file: File,
    initial: ObjectSnapshotV1,
    sha256: [u8; 32],
}
impl Source {
    fn revalidate_content(&self, root: &Root) -> Result<()> {
        self.revalidate(root)?;
        // Same-length writes can share an observed filesystem timestamp tick.
        if stream_hash(&self.file, self.initial.byte_len, None)? != self.sha256 {
            return Err(changed("compiler bundle source content changed"));
        }
        self.revalidate(root)
    }

    fn revalidate(&self, root: &Root) -> Result<()> {
        let mode = self.initial.mode & 0o7777;
        let length = self.initial.byte_len;
        if source_metadata(&self.file, root, mode, length, Some(length))? != self.initial
            || source_metadata(
                &open_beneath(&root.file, &self.path, false)?,
                root,
                mode,
                length,
                Some(length),
            )? != self.initial
        {
            return Err(changed("compiler bundle source or pathname changed"));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_source(
    root: &Root,
    path: &str,
    mode: u32,
    length: Option<u64>,
    maximum: u64,
    digest: [u8; 32],
    mismatch: Kind,
) -> Result<(File, Source)> {
    let file = open_beneath(&root.file, path, false)?;
    let initial = source_metadata(&file, root, mode, maximum, length)?;
    let source = Source {
        path: path.to_owned(),
        file,
        initial,
        sha256: digest,
    };
    let sealed = File::from(
        memfd_create(
            c"fe2o3-compiler-runtime-source-v1",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .map_err(|e| io_error("create compiler runtime sealed source", e))?,
    );
    if stream_hash(&source.file, initial.byte_len, Some(&sealed))? != digest {
        return Err(invalid(
            mismatch,
            "compiler bundle raw SHA256 differs from pin",
        ));
    }
    source.revalidate(root)?;
    rustix::fs::fchmod(&sealed, Mode::from_raw_mode(mode))
        .map_err(|e| io_error("set compiler runtime sealed source mode", e))?;
    fcntl_add_seals(&sealed, SEALS).map_err(|e| io_error("seal compiler runtime source", e))?;
    validate_sealed(&sealed, mode, initial.byte_len, digest)?;
    Ok((sealed, source))
}

fn read_exact(file: &File, bytes: &mut [u8], offset: u64) -> Result<()> {
    let n = rustix::io::pread(file, &mut *bytes, offset)
        .map_err(|e| io_error("read compiler runtime bytes", e))?;
    if n != bytes.len() {
        return Err(changed("short compiler runtime read"));
    }
    Ok(())
}

fn stream_hash(file: &File, length: u64, copy: Option<&File>) -> Result<[u8; 32]> {
    let mut buffer = [0; CHUNK];
    let mut hash = Sha256::new();
    let mut offset = 0;
    while offset < length {
        let n = (length - offset).min(CHUNK as u64) as usize;
        read_exact(file, &mut buffer[..n], offset)?;
        if let Some(copy) = copy {
            let written = rustix::io::pwrite(copy, &buffer[..n], offset)
                .map_err(|e| io_error("copy compiler runtime bytes", e))?;
            if written != n {
                return Err(changed("short compiler runtime copy"));
            }
        }
        hash.update(&buffer[..n]);
        offset += n as u64;
    }
    if rustix::io::pread(file, &mut buffer[..1], length)
        .map_err(|e| io_error("probe compiler runtime extent", e))?
        != 0
    {
        return Err(changed(
            "compiler runtime source grew beyond declared extent",
        ));
    }
    Ok(hash.finalize().into())
}

fn validate_sealed(file: &File, mode: u32, length: u64, digest: [u8; 32]) -> Result<()> {
    let metadata = file_snapshot(file)?;
    if FileType::from_raw_mode(metadata.mode) != FileType::RegularFile
        || metadata.mode & 0o7777 != mode
        || metadata.byte_len != length
        || rustix::io::fcntl_getfd(file)
            .map_err(|e| io_error("inspect compiler source flags", e))?
            != rustix::io::FdFlags::CLOEXEC
        || fcntl_get_seals(file).map_err(|e| io_error("inspect compiler source seals", e))? != SEALS
        || stream_hash(file, length, None)? != digest
    {
        return Err(changed("sealed compiler runtime source differs"));
    }
    Ok(())
}

fn roster(manifest: &CompilerRuntimeManifestV1) -> Result<Roster> {
    let mut paths = BTreeSet::from([POLICY.to_owned(), MANIFEST.to_owned()]);
    for entry in manifest.entries() {
        if !paths.insert(format!("runtime/{}", entry.path)) {
            return Err(invalid(
                Kind::InvalidInventory,
                "duplicate compiler bundle file",
            ));
        }
    }
    let mut roster = Roster::new();
    let mut objects = 0usize;
    for path in &paths {
        let mut parent = String::new();
        for component in path.split('/') {
            if paths.contains(&parent) {
                return Err(invalid(
                    Kind::InvalidInventory,
                    "compiler file is also a directory",
                ));
            }
            if roster
                .entry(parent.clone())
                .or_default()
                .insert(component.to_owned())
            {
                objects += 1;
                if objects > MAX_OBJECTS {
                    return Err(invalid(
                        Kind::InvalidInventory,
                        "compiler bundle roster exceeds bound",
                    ));
                }
            }
            if !parent.is_empty() {
                parent.push('/');
            }
            parent.push_str(component);
        }
    }
    Ok(roster)
}

struct Directory {
    path: String,
    file: File,
    initial: ObjectSnapshotV1,
}
impl Directory {
    fn revalidate(&self, root: &Root, roster: &Roster) -> Result<()> {
        let owner = Some((root.initial.uid, root.initial.gid));
        let expected: Vec<&str> = roster[&self.path].iter().map(String::as_str).collect();
        check_children(&self.file, &expected)?;
        if validate_directory(&self.file, owner, "retained runtime directory")? != self.initial
            || validate_directory(
                &open_beneath(&root.file, &self.path, true)?,
                owner,
                "current runtime directory",
            )? != self.initial
        {
            return Err(changed("compiler runtime directory changed"));
        }
        Ok(())
    }
}

fn retain_directories(root: &Root, roster: &Roster) -> Result<Vec<Directory>> {
    let mut directories = Vec::with_capacity(roster.len());
    for (path, children) in roster {
        if path.is_empty() {
            continue;
        }
        let file = open_beneath(&root.file, path, true)?;
        let initial = validate_directory(
            &file,
            Some((root.initial.uid, root.initial.gid)),
            "runtime bundle directory",
        )?;
        let expected: Vec<&str> = children.iter().map(String::as_str).collect();
        check_children(&file, &expected)?;
        let directory = Directory {
            path: path.clone(),
            file,
            initial,
        };
        directory.revalidate(root, roster)?;
        directories.push(directory);
    }
    Ok(directories)
}

fn check_children(directory: &File, expected: &[&str]) -> Result<()> {
    if expected.len() > MAX_OBJECTS {
        return Err(invalid(
            Kind::InvalidInventory,
            "compiler directory roster exceeds bound",
        ));
    }
    let scan = openat(
        directory,
        ".",
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|e| io_error("retain compiler directory for enumeration", e))?;
    let mut entries = rustix::fs::Dir::read_from(&scan)
        .map_err(|e| io_error("enumerate compiler directory", e))?;
    let mut seen = vec![false; expected.len()];
    // Bound every returned entry, including dot entries, before processing names.
    for (turn, entry) in (&mut entries).enumerate() {
        if turn >= expected.len() + 2 {
            return Err(invalid(
                Kind::InvalidInventory,
                "compiler directory enumeration exceeds bound",
            ));
        }
        let entry = entry.map_err(|e| io_error("read compiler directory entry", e))?;
        let name = entry.file_name().to_bytes();
        if matches!(name, b"." | b"..") {
            continue;
        }
        let Some(index) = expected.iter().position(|s| s.as_bytes() == name) else {
            return Err(invalid(
                Kind::InvalidInventory,
                "extra compiler bundle object",
            ));
        };
        if std::mem::replace(&mut seen[index], true) {
            return Err(invalid(
                Kind::InvalidInventory,
                "duplicate compiler directory entry",
            ));
        }
    }
    if seen.iter().any(|seen| !seen) {
        return Err(invalid(
            Kind::InvalidInventory,
            "missing compiler bundle object",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "compiler_runtime_bundle_tests.rs"]
pub(super) mod tests;

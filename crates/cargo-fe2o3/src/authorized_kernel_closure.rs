//! Pre-Cargo host-code policy for authority-bearing kernel compilations.

use std::collections::{BTreeMap, BTreeSet};
#[cfg(target_os = "linux")]
use std::ffi::{CStr, CString, OsStr};
use std::fs::{self, Metadata, OpenOptions};
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::pinned_executable::PinnedExecutable;
use crate::project::CargoProject;

const MAX_METADATA_BYTES: usize = 32 * 1024 * 1024;
const MAX_LOCK_BYTES: usize = 8 * 1024 * 1024;
const MAX_SOURCE_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SOURCE_TREE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_SOURCE_TREE_FILES: usize = 100_000;
const CRATES_IO_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";
const TRUSTED_FE2O3_EXTERNAL_SOURCE: &str = concat!(
    "git+https://github.com/harsh-nod/fe2o3.git?rev=",
    "d955209099c7b434dfceb69e1152d948dab76b22#",
    "d955209099c7b434dfceb69e1152d948dab76b22",
);
const TRUSTED_REGISTRY_BUILD_SCRIPTS: [(&str, &str, &str); 25] = [
    (
        "cap-primitives",
        "4.0.2",
        "9c5c6262db1c26d16dc7bc175a2785b15d8e5e0c02825ffd1be9e20a4bff50f1",
    ),
    (
        "const_fn",
        "0.4.12",
        "12719e3056fd7d108dce37f1802f2ab7d4e57c1ebbd28bf263c2dde74a4977f8",
    ),
    (
        "curve25519-dalek",
        "4.1.3",
        "d6ce8e9f5bcd25566d94e0086de692a8f5049baca759b54ec2fcb04fcc6ad157",
    ),
    (
        "generic-array",
        "0.14.7",
        "4342744f404f2087772e53283fbedaa581be1b1cea9e5ba0d538b9e66dfbb948",
    ),
    (
        "io-extras",
        "0.19.0",
        "41ca4460f88bdeb51e5236f79540483047e935c6386db8ee5242cde7572075d4",
    ),
    (
        "io-lifetimes",
        "2.0.4",
        "41e3c9aefd8e587fce5aaa6b326b7f72babb8509fd97fec0b85923f46d244d7e",
    ),
    (
        "io-lifetimes",
        "3.0.1",
        "f992fc3ac492e187e4a5c5b4d6501e6fcfee4a8ab3ee1983fb1291cdea4d4108",
    ),
    (
        "libc",
        "0.2.189",
        "54f1477836437a81c25bfb0774a700a3043d80a3f20c395429f47f66db34ace3",
    ),
    (
        "linkme",
        "0.3.37",
        "fc0248a176fa41d6cbb679f98e38621b4361e5f0edf56788ed527fdf84396c36",
    ),
    (
        "linkme-impl",
        "0.3.37",
        "af6739c0caf779cbb89368a70ea31d1c36251543c66440f36f14566dfc28e6aa",
    ),
    (
        "num-traits",
        "0.2.19",
        "a70b98cd31b6d7ed51cb8c0d25cbe86b0d61bccd3fcb8775b5369aaeb5f18a7e",
    ),
    (
        "object",
        "0.39.1",
        "5007f85062b9881599bd080205a7bce0e2fffbcc024ea9d6cc2d35c5ab58dc4f",
    ),
    (
        "proc-macro2",
        "1.0.106",
        "09dc7aa3070a182d1a247cb6876af476bcc0db2390908facd93edbf3ede8a03b",
    ),
    (
        "proc-macro2",
        "1.0.107",
        "369ed937912de48187d2b6b9706b3c76128b3c9ca75b4404abf79306ac6db9f1",
    ),
    (
        "quote",
        "1.0.45",
        "8d6c868c0e133b6a257d426993ff53ba648a8fa4cfc805fe480b6c0c89e56638",
    ),
    (
        "quote",
        "1.0.47",
        "600f7d275e1f5f809dd2c6670acf7a4966733eb66983d8650c4d6b1d51184f29",
    ),
    (
        "rustc_apfloat",
        "0.2.3+llvm-462a31f5a5ab",
        "a014a9ad4bf9561d5fc1f2622245293b8213fea3ac714776a1bcda6c3f95711c",
    ),
    (
        "rustix",
        "1.1.4",
        "62dd547337499a696f957e605c76c0419a2d34d915fcf06554827f955b93994a",
    ),
    (
        "rustversion",
        "1.0.23",
        "e61d0c17536142c100d3cd417564742345b6f78db5ef37cf053eb396bd9c6987",
    ),
    (
        "serde",
        "1.0.229",
        "c270b89adc556d39cd1cb5943cb564e56c09d2fe10797143ec89e7b02800758c",
    ),
    (
        "serde_core",
        "1.0.229",
        "cfc95168c497e78cae0bf1dcb1c18302358c3b1831dcd5fbdda49fc51975def1",
    ),
    (
        "serde_json",
        "1.0.151",
        "9fb6b972f5ef5eaf17be5c50854f9a328c5c0c8f9a7b462f51119ec92001f682",
    ),
    (
        "slotmap",
        "1.1.1",
        "44aa0e37cc10b3306d0035a8c94c229a24de8602ae1a88a2cc27b229bb24f9f4",
    ),
    (
        "thiserror",
        "2.0.20",
        "681133a62937d3660c49de30899f07ccdb1b4ef53426f9014fefd59cf78ab670",
    ),
    (
        "zmij",
        "1.0.23",
        "ee4ed4bdafb98dc92c5a51095290212137f81ffc6cdfae77e9cb540373fb4c11",
    ),
];
const TRUSTED_REGISTRY_PROC_MACROS: [(&str, &str, &str); 9] = [
    (
        "awint_macros",
        "0.18.1",
        "a48c5475a5c4adf80066644a0cbbc3ed565e38eb5a9dd061cf8953450ba8e3b5",
    ),
    (
        "awint_macros",
        "0.19.0",
        "ee1c3c771747ccebec28a74521447163a7d3088d68b64f64b26241a6e32b8725",
    ),
    (
        "const_fn",
        "0.4.12",
        "12719e3056fd7d108dce37f1802f2ab7d4e57c1ebbd28bf263c2dde74a4977f8",
    ),
    (
        "curve25519-dalek-derive",
        "0.1.1",
        "95a234384a3fb6a73a7addf7543e85d47e7a1d175b138bd4617a1d0487c6b6b9",
    ),
    (
        "linkme-impl",
        "0.3.37",
        "af6739c0caf779cbb89368a70ea31d1c36251543c66440f36f14566dfc28e6aa",
    ),
    (
        "rustversion",
        "1.0.23",
        "e61d0c17536142c100d3cd417564742345b6f78db5ef37cf053eb396bd9c6987",
    ),
    (
        "serde_derive",
        "1.0.229",
        "d685313d886c74b7780135bebc48dfa0d5df91d67f1008ad0113d788b359aa32",
    ),
    (
        "thiserror-impl",
        "2.0.16",
        "3aa02dfbd6d84d88f75ea9f799e51903be846c41685bd3d15988a2bbfd15455b",
    ),
    (
        "thiserror-impl",
        "2.0.20",
        "2b03916e618b74694c727ea32ae011c0d183faf846fd6d11bd6917aae0dc13f4",
    ),
];
// These exact revisions have identical reviewed pliron-derive trees. Keep the
// prior admission while admitting the workspace pin; neither allows Git drift.
const TRUSTED_GIT_PROC_MACROS: [(&str, &str, &str, &str); 2] = [
    (
        "pliron-derive",
        "0.17.0",
        "git+https://github.com/harsh-nod/pliron.git?rev=e054e5b2e53c7330470f9202c35c8c0e4e102092#e054e5b2e53c7330470f9202c35c8c0e4e102092",
        "18abd61218886753068c38294931388e53e90f17b85cd8660b83ec93280bc824",
    ),
    (
        "pliron-derive",
        "0.17.0",
        "git+https://github.com/harsh-nod/pliron.git?rev=90ef1c11647cc33bbadc83c29d21525cbd8e4c2c#90ef1c11647cc33bbadc83c29d21525cbd8e4c2c",
        "18abd61218886753068c38294931388e53e90f17b85cd8660b83ec93280bc824",
    ),
];
// The complete reviewed workspace tree includes macro source, the manifest and
// nested fixture lockfiles. Source changes require review and a fresh tree pin;
// they do not change the independent external-source admission below.
const TRUSTED_FE2O3_MACROS_TREE: &str =
    "9e3ab9723b1ea729730a9353f0da5e9b972ddef1a4f6518ecb44b9a960633dda";
// This digest belongs to TRUSTED_FE2O3_EXTERNAL_SOURCE and is intentionally
// independent of the workspace-local macros tree.
const TRUSTED_FE2O3_EXTERNAL_MACROS_TREE: &str =
    "a6ae6e79c48ee48389411aee2db6b438599e525009e45f5773d5e0d3ba57efcc";

pub(crate) struct AuthorizedKernelClosureV1 {
    snapshot: Vec<u8>,
    source_trees: Vec<ObservedSourceTree>,
    lockfile: ObservedFile,
    mutation_journal: MutationJournal,
}

struct ObservedSourceTree {
    package_name: String,
    root: PathBuf,
    excluded: Option<PathBuf>,
    digest: [u8; 32],
}

impl ObservedSourceTree {
    fn capture(
        package_name: String,
        root: PathBuf,
        excluded: Option<PathBuf>,
        mutation_journal: Option<&MutationJournal>,
    ) -> Result<Self, String> {
        let digest = canonical_tree_digest_monitored(&root, excluded.as_deref(), mutation_journal)?;
        Ok(Self {
            package_name,
            root,
            excluded,
            digest,
        })
    }

    fn revalidate(&self) -> Result<(), String> {
        let current = canonical_tree_digest(&self.root, self.excluded.as_deref())?;
        if current != self.digest {
            return Err(format!(
                "authoritative source closure changed after preflight under {}",
                self.root.display()
            ));
        }
        Ok(())
    }
}

struct ObservedFile {
    path: PathBuf,
    digest: [u8; 32],
    size: u64,
}

#[cfg(target_os = "linux")]
struct MutationJournal {
    descriptor: OwnedFd,
    watch_by_path: std::sync::Mutex<BTreeMap<PathBuf, i32>>,
    path_by_watch: std::sync::Mutex<BTreeMap<i32, PathBuf>>,
    excluded: Vec<PathBuf>,
}

#[cfg(target_os = "linux")]
impl MutationJournal {
    fn new(excluded: Vec<PathBuf>) -> Result<Self, String> {
        let descriptor = unsafe { libc::inotify_init1(libc::IN_CLOEXEC | libc::IN_NONBLOCK) };
        if descriptor < 0 {
            return Err(format!(
                "cannot start authoritative source mutation journal: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self {
            descriptor: unsafe { OwnedFd::from_raw_fd(descriptor) },
            watch_by_path: std::sync::Mutex::new(BTreeMap::new()),
            path_by_watch: std::sync::Mutex::new(BTreeMap::new()),
            excluded,
        })
    }

    fn watch_directory(&self, directory: &Path) -> Result<(), String> {
        let mut watch_by_path = self
            .watch_by_path
            .lock()
            .map_err(|_| "authoritative mutation journal path map was poisoned".to_owned())?;
        if watch_by_path.contains_key(directory) {
            return Ok(());
        }
        let path = CString::new(directory.as_os_str().as_bytes()).map_err(|_| {
            format!(
                "authoritative source directory contains a NUL byte: {}",
                directory.display()
            )
        })?;
        let mask = libc::IN_ATTRIB
            | libc::IN_CLOSE_WRITE
            | libc::IN_CREATE
            | libc::IN_DELETE
            | libc::IN_DELETE_SELF
            | libc::IN_MODIFY
            | libc::IN_MOVE_SELF
            | libc::IN_MOVED_FROM
            | libc::IN_MOVED_TO
            | libc::IN_DONT_FOLLOW
            | libc::IN_EXCL_UNLINK
            | libc::IN_ONLYDIR;
        let watch =
            unsafe { libc::inotify_add_watch(self.descriptor.as_raw_fd(), path.as_ptr(), mask) };
        if watch < 0 {
            return Err(format!(
                "cannot journal authoritative source directory {}: {}",
                directory.display(),
                std::io::Error::last_os_error()
            ));
        }
        self.path_by_watch
            .lock()
            .map_err(|_| "authoritative mutation journal watch map was poisoned".to_owned())?
            .insert(watch, directory.to_path_buf());
        watch_by_path.insert(directory.to_path_buf(), watch);
        Ok(())
    }

    fn ensure_quiet(&self) -> Result<(), String> {
        let mut storage = [0_u64; 1024];
        loop {
            let bytes = unsafe {
                libc::read(
                    self.descriptor.as_raw_fd(),
                    storage.as_mut_ptr().cast(),
                    std::mem::size_of_val(&storage),
                )
            };
            if bytes < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::WouldBlock {
                    return Ok(());
                }
                return Err(format!(
                    "cannot read authoritative source mutation journal: {error}"
                ));
            }
            if bytes == 0 {
                return Err("authoritative source mutation journal closed unexpectedly".to_owned());
            }
            self.reject_events(storage.as_ptr().cast(), bytes as usize)?;
        }
    }

    fn reject_events(&self, bytes: *const u8, length: usize) -> Result<(), String> {
        let paths = self
            .path_by_watch
            .lock()
            .map_err(|_| "authoritative mutation journal watch map was poisoned".to_owned())?;
        let mut offset = 0_usize;
        while offset < length {
            if length - offset < std::mem::size_of::<libc::inotify_event>() {
                return Err(
                    "authoritative source mutation journal returned a partial event".to_owned(),
                );
            }
            let event = unsafe {
                std::ptr::read_unaligned(bytes.add(offset).cast::<libc::inotify_event>())
            };
            let record_length = std::mem::size_of::<libc::inotify_event>()
                .checked_add(event.len as usize)
                .ok_or_else(|| {
                    "authoritative source mutation journal event length overflowed".to_owned()
                })?;
            if record_length > length - offset {
                return Err(
                    "authoritative source mutation journal returned a truncated event".to_owned(),
                );
            }
            if event.mask & libc::IN_Q_OVERFLOW != 0 {
                return Err(
                    "authoritative source mutation journal overflowed; closure authority is denied"
                        .to_owned(),
                );
            }
            let base = paths.get(&event.wd).ok_or_else(|| {
                format!(
                    "authoritative source mutation journal reported unknown watch {}",
                    event.wd
                )
            })?;
            let path = if event.len == 0 {
                base.clone()
            } else {
                let name = unsafe {
                    CStr::from_ptr(
                        bytes
                            .add(offset + std::mem::size_of::<libc::inotify_event>())
                            .cast(),
                    )
                };
                base.join(OsStr::from_bytes(name.to_bytes()))
            };
            if !self
                .excluded
                .iter()
                .any(|excluded| path == *excluded || path.starts_with(excluded))
            {
                return Err(format!(
                    "authoritative source closure mutated after preflight: {} (inotify mask 0x{:x})",
                    path.display(),
                    event.mask
                ));
            }
            offset += record_length;
        }
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
struct MutationJournal;

#[cfg(not(target_os = "linux"))]
impl MutationJournal {
    fn new(_excluded: Vec<PathBuf>) -> Result<Self, String> {
        Err("authoritative source mutation journaling requires Linux".to_owned())
    }

    fn watch_directory(&self, _directory: &Path) -> Result<(), String> {
        Err("authoritative source mutation journaling requires Linux".to_owned())
    }

    fn ensure_quiet(&self) -> Result<(), String> {
        Err("authoritative source mutation journaling requires Linux".to_owned())
    }
}

impl ObservedFile {
    fn capture(path: PathBuf, limit: u64) -> Result<Self, String> {
        let (digest, size) = hash_regular_file(&path, limit)?;
        Ok(Self { path, digest, size })
    }

    fn revalidate(&self, limit: u64) -> Result<(), String> {
        let (digest, size) = hash_regular_file(&self.path, limit)?;
        if digest != self.digest || size != self.size {
            return Err(format!(
                "authoritative input changed after preflight: {}",
                self.path.display()
            ));
        }
        Ok(())
    }
}

impl AuthorizedKernelClosureV1 {
    pub(crate) fn observe(
        project: &CargoProject,
        args: &[std::ffi::OsString],
        cargo: &PinnedExecutable,
        rustc: &crate::PinnedRustc,
    ) -> Result<Self, String> {
        let mut command = cargo
            .command()
            .map_err(|error| format!("failed to prepare pinned Cargo metadata: {error}"))?;
        command
            .as_command_mut()
            .args(["metadata", "--format-version", "1"])
            .args(project.authority_metadata_args(args)?)
            .args(["--frozen", "--offline"])
            .current_dir(project.invocation_dir().child_path());
        crate::configure_authority_cargo_child(command.as_command_mut(), rustc)?;
        let output = command
            .output()
            .map_err(|error| format!("failed to run pinned Cargo metadata: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "could not resolve authoritative Cargo closure: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        if output.stdout.is_empty() || output.stdout.len() > MAX_METADATA_BYTES {
            return Err(format!(
                "authoritative Cargo metadata must contain 1 through {MAX_METADATA_BYTES} bytes"
            ));
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("failed to parse authoritative Cargo metadata: {error}"))?;
        Self::from_metadata(&metadata, args, cargo.sha256())
    }

    pub(crate) fn snapshot(&self) -> &[u8] {
        &self.snapshot
    }

    pub(crate) fn package_name_for_root(&self, root: &Path) -> Result<&str, String> {
        let mut matches = self.source_trees.iter().filter(|tree| tree.root == root);
        let owner = matches.next().ok_or_else(|| {
            format!(
                "host binding source has no retained package owner: {}",
                root.display()
            )
        })?;
        if matches.next().is_some() {
            return Err("host binding source has multiple retained package owners".to_owned());
        }
        Ok(&owner.package_name)
    }

    pub(crate) fn revalidate(&self) -> Result<(), String> {
        self.mutation_journal.ensure_quiet()?;
        for tree in &self.source_trees {
            tree.revalidate()?;
        }
        self.lockfile.revalidate(MAX_LOCK_BYTES as u64)?;
        self.mutation_journal.ensure_quiet()
    }

    fn from_metadata(
        metadata: &Value,
        args: &[std::ffi::OsString],
        cargo_digest: &[u8; 32],
    ) -> Result<Self, String> {
        let packages = metadata
            .get("packages")
            .and_then(Value::as_array)
            .ok_or_else(|| "authoritative Cargo metadata has no package array".to_owned())?;
        let package_by_id = packages
            .iter()
            .map(|package| Ok((required_string(package, "id")?.to_owned(), package)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let resolve = metadata
            .get("resolve")
            .and_then(Value::as_object)
            .ok_or_else(|| "authoritative Cargo metadata has no resolved graph".to_owned())?;
        let nodes = resolve
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| "authoritative Cargo metadata has no resolved nodes".to_owned())?;
        let dependencies = nodes
            .iter()
            .map(|node| {
                let id = required_string(node, "id")?.to_owned();
                Ok((id, build_dependency_ids(node)?))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let roots = selected_roots(metadata, resolve, args, &package_by_id)?;
        let mut pending = roots;
        let mut closure = BTreeSet::new();
        while let Some(id) = pending.pop() {
            if !closure.insert(id.clone()) {
                continue;
            }
            let next = dependencies
                .get(&id)
                .ok_or_else(|| format!("selected package {id:?} has no resolved node"))?;
            pending.extend(next.iter().cloned());
        }

        let target_directory = metadata
            .get("target_directory")
            .and_then(Value::as_str)
            .map(PathBuf::from);
        let mutation_journal = MutationJournal::new(target_directory.iter().cloned().collect())?;
        let mut snapshot = b"fe2o3-authorized-kernel-closure-content-v2\0".to_vec();
        append_field(&mut snapshot, cargo_digest);
        let mut source_trees = Vec::with_capacity(closure.len());
        for id in &closure {
            let package = package_by_id
                .get(id)
                .ok_or_else(|| format!("resolved package {id:?} has no metadata record"))?;
            let manifest = PathBuf::from(required_string(package, "manifest_path")?);
            let root = manifest.parent().ok_or_else(|| {
                format!(
                    "package {id:?} manifest has no parent: {}",
                    manifest.display()
                )
            })?;
            let observed_tree = ObservedSourceTree::capture(
                required_string(package, "name")?.to_owned(),
                root.to_path_buf(),
                target_directory
                    .as_ref()
                    .filter(|target| target.starts_with(root))
                    .cloned(),
                Some(&mutation_journal),
            )?;
            let tree_digest = observed_tree.digest;
            validate_host_code_package(package, &tree_digest)?;
            append_field(&mut snapshot, id.as_bytes());
            for field in ["name", "version", "source", "checksum", "manifest_path"] {
                append_field(
                    &mut snapshot,
                    package
                        .get(field)
                        .and_then(Value::as_str)
                        .unwrap_or("-")
                        .as_bytes(),
                );
            }
            let mut next = dependencies
                .get(id)
                .cloned()
                .ok_or_else(|| format!("resolved package {id:?} has no dependency record"))?;
            next.sort();
            for dependency in next {
                append_field(&mut snapshot, dependency.as_bytes());
            }
            append_field(&mut snapshot, &tree_digest);
            source_trees.push(observed_tree);
        }

        let workspace_root = metadata
            .get("workspace_root")
            .and_then(Value::as_str)
            .ok_or_else(|| "authoritative Cargo metadata has no workspace root".to_owned())?;
        let lock_path = Path::new(workspace_root).join("Cargo.lock");
        mutation_journal.watch_directory(Path::new(workspace_root))?;
        let lockfile = ObservedFile::capture(lock_path, MAX_LOCK_BYTES as u64)?;
        if lockfile.size == 0 {
            return Err(format!(
                "authoritative lockfile must contain 1 through {MAX_LOCK_BYTES} bytes"
            ));
        }
        append_field(&mut snapshot, &lockfile.digest);
        mutation_journal.ensure_quiet()?;
        Ok(Self {
            snapshot,
            source_trees,
            lockfile,
            mutation_journal,
        })
    }
}

fn build_dependency_ids(node: &Value) -> Result<Vec<String>, String> {
    let id = required_string(node, "id")?;
    let deps = node
        .get("deps")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("resolved package {id:?} has no structured dependency array"))?;
    let mut included = Vec::new();
    for dependency in deps {
        let package = required_string(dependency, "pkg")?;
        let kinds = dependency
            .get("dep_kinds")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!("resolved dependency {package:?} of package {id:?} has no kind array")
            })?;
        if kinds.is_empty() {
            return Err(format!(
                "resolved dependency {package:?} of package {id:?} has no dependency kind"
            ));
        }
        let mut used_by_build = false;
        for kind in kinds {
            match kind.get("kind") {
                Some(Value::Null) => used_by_build = true,
                Some(Value::String(value)) if value == "build" => used_by_build = true,
                Some(Value::String(value)) if value == "dev" => {}
                Some(Value::String(value)) => {
                    return Err(format!(
                        "resolved dependency {package:?} of package {id:?} has unknown kind {value:?}"
                    ));
                }
                _ => {
                    return Err(format!(
                        "resolved dependency {package:?} of package {id:?} has a malformed kind"
                    ));
                }
            }
        }
        if used_by_build {
            included.push(package.to_owned());
        }
    }
    included.sort();
    included.dedup();
    Ok(included)
}

fn selected_roots(
    metadata: &Value,
    resolve: &serde_json::Map<String, Value>,
    args: &[std::ffi::OsString],
    packages: &BTreeMap<String, &Value>,
) -> Result<Vec<String>, String> {
    let selected = selected_package_names(args)?;
    if !selected.is_empty() {
        let workspace_members = string_array(metadata, "workspace_members")?
            .into_iter()
            .collect::<BTreeSet<_>>();
        let mut roots = Vec::new();
        for selected in selected {
            let matches = packages
                .iter()
                .filter(|(id, package)| {
                    workspace_members.contains(*id)
                        && package.get("name").and_then(Value::as_str) == Some(&selected)
                })
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(format!(
                    "authoritative package selection {selected:?} matched {} workspace packages",
                    matches.len()
                ));
            }
            roots.push(matches[0].clone());
        }
        return Ok(roots);
    }
    if args.iter().any(|argument| argument == "--workspace") {
        return string_array(metadata, "workspace_members");
    }
    if let Some(root) = resolve.get("root").and_then(Value::as_str) {
        return Ok(vec![root.to_owned()]);
    }
    string_array(metadata, "workspace_default_members")
}

fn selected_package_names(args: &[std::ffi::OsString]) -> Result<Vec<String>, String> {
    let mut selected = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let value = args[index]
            .to_str()
            .ok_or_else(|| "authoritative Cargo arguments must be UTF-8".to_owned())?;
        if value == "-p" || value == "--package" {
            index += 1;
            let package = args
                .get(index)
                .and_then(|value| value.to_str())
                .ok_or_else(|| "authoritative --package requires one UTF-8 name".to_owned())?;
            selected.push(package.to_owned());
        } else if let Some(package) = value.strip_prefix("--package=") {
            if package.is_empty() {
                return Err("authoritative --package requires a non-empty name".to_owned());
            }
            selected.push(package.to_owned());
        }
        index += 1;
    }
    selected.sort();
    selected.dedup();
    Ok(selected)
}

fn validate_host_code_package(package: &Value, tree_digest: &[u8; 32]) -> Result<(), String> {
    let name = required_string(package, "name")?;
    if package.get("links").is_some_and(|value| !value.is_null()) {
        return Err(format!(
            "authoritative kernel closure rejects native links package {name:?}"
        ));
    }
    let targets = package
        .get("targets")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("package {name:?} has no target array"))?;
    let kinds = targets
        .iter()
        .flat_map(|target| {
            target
                .get("kind")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
        })
        .collect::<BTreeSet<_>>();
    if kinds.contains("custom-build") && !is_reviewed_registry_build_script(package, tree_digest)? {
        return Err(format!(
            "authoritative kernel closure rejects unreviewed custom-build package {name:?}"
        ));
    }
    if kinds.contains("proc-macro") {
        validate_reviewed_proc_macro(package, tree_digest)?;
    }
    Ok(())
}

fn is_reviewed_registry_build_script(
    package: &Value,
    tree_digest: &[u8; 32],
) -> Result<bool, String> {
    validate_registry_build_script_against(package, tree_digest, &TRUSTED_REGISTRY_BUILD_SCRIPTS)
}

fn validate_registry_build_script_against(
    package: &Value,
    tree_digest: &[u8; 32],
    trusted: &[(&str, &str, &str)],
) -> Result<bool, String> {
    let Some(name) = package.get("name").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some(version) = package.get("version").and_then(Value::as_str) else {
        return Ok(false);
    };
    if package.get("source").and_then(Value::as_str) != Some(CRATES_IO_SOURCE) {
        return Ok(false);
    }
    let Some((_, _, expected)) = trusted.iter().find(|(trusted_name, trusted_version, _)| {
        name == *trusted_name && version == *trusted_version
    }) else {
        return Ok(false);
    };
    validate_expected_tree(
        tree_digest,
        expected,
        Path::new(required_string(package, "manifest_path")?)
            .parent()
            .expect("manifest path was already required"),
        "registry build-script",
    )?;
    Ok(true)
}

fn validate_reviewed_proc_macro(package: &Value, tree_digest: &[u8; 32]) -> Result<(), String> {
    if package.get("name").and_then(Value::as_str) == Some("fe2o3-macros") {
        return validate_reviewed_fe2o3_macros(package, tree_digest);
    }
    if validate_registry_proc_macro_against(package, tree_digest, &TRUSTED_REGISTRY_PROC_MACROS)?
        || validate_git_proc_macro_against(package, tree_digest, &TRUSTED_GIT_PROC_MACROS)?
    {
        return Ok(());
    }
    Err(unreviewed_proc_macro(package))
}

fn unreviewed_proc_macro(package: &Value) -> String {
    let name = package
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("<missing-name>");
    let version = package
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("<missing-version>");
    let source = match package.get("source") {
        Some(Value::Null) => "local",
        Some(Value::String(source)) => source,
        _ => "<missing-source>",
    };
    format!(
        "authoritative kernel closure rejects an unreviewed procedural macro {name:?} version {version:?} from {source:?}"
    )
}

fn validate_registry_proc_macro_against(
    package: &Value,
    tree_digest: &[u8; 32],
    trusted: &[(&str, &str, &str)],
) -> Result<bool, String> {
    let Some(name) = package.get("name").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some(version) = package.get("version").and_then(Value::as_str) else {
        return Ok(false);
    };
    if package.get("source").and_then(Value::as_str) != Some(CRATES_IO_SOURCE) {
        return Ok(false);
    }
    let Some((_, _, expected)) = trusted.iter().find(|(trusted_name, trusted_version, _)| {
        name == *trusted_name && version == *trusted_version
    }) else {
        return Ok(false);
    };
    validate_expected_tree(
        tree_digest,
        expected,
        Path::new(required_string(package, "manifest_path")?)
            .parent()
            .expect("manifest path was already required"),
        "registry proc-macro",
    )?;
    Ok(true)
}

fn validate_git_proc_macro_against(
    package: &Value,
    tree_digest: &[u8; 32],
    trusted: &[(&str, &str, &str, &str)],
) -> Result<bool, String> {
    let Some(name) = package.get("name").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some(version) = package.get("version").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some(source) = package.get("source").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some((_, _, _, expected)) =
        trusted
            .iter()
            .find(|(trusted_name, trusted_version, trusted_source, _)| {
                name == *trusted_name && version == *trusted_version && source == *trusted_source
            })
    else {
        return Ok(false);
    };
    validate_expected_tree(
        tree_digest,
        expected,
        Path::new(required_string(package, "manifest_path")?)
            .parent()
            .expect("manifest path was already required"),
        "git proc-macro",
    )?;
    Ok(true)
}

fn validate_reviewed_fe2o3_macros(package: &Value, tree_digest: &[u8; 32]) -> Result<(), String> {
    if package.get("name").and_then(Value::as_str) != Some("fe2o3-macros")
        || package.get("version").and_then(Value::as_str) != Some("0.1.0")
    {
        return Err(unreviewed_proc_macro(package));
    }
    let expected = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("cargo-fe2o3 has a workspace crates directory")
        .join("fe2o3-macros");
    let (observed_root, external) =
        validate_reviewed_workspace_package_source(package, &expected, "fe2o3-macros")
            .map_err(|_| unreviewed_proc_macro(package))?;
    validate_expected_tree(
        tree_digest,
        if external {
            TRUSTED_FE2O3_EXTERNAL_MACROS_TREE
        } else {
            TRUSTED_FE2O3_MACROS_TREE
        },
        &observed_root,
        "proc-macro",
    )
}

fn validate_reviewed_workspace_package_source(
    package: &Value,
    expected_local_root: &Path,
    package_name: &str,
) -> Result<(PathBuf, bool), String> {
    let manifest = PathBuf::from(required_string(package, "manifest_path")?);
    match package.get("source") {
        Some(Value::Null) if manifest == expected_local_root.join("Cargo.toml") => {
            Ok((expected_local_root.to_path_buf(), false))
        }
        Some(Value::String(source))
            if source == TRUSTED_FE2O3_EXTERNAL_SOURCE
                && manifest.is_absolute()
                && manifest.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml")
                && manifest
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|name| name.to_str())
                    == Some(package_name)
                && manifest
                    .parent()
                    .and_then(Path::parent)
                    .and_then(Path::file_name)
                    .and_then(|name| name.to_str())
                    == Some("crates") =>
        {
            Ok((
                manifest
                    .parent()
                    .expect("reviewed git manifest has a package parent")
                    .to_path_buf(),
                true,
            ))
        }
        _ => Err(format!(
            "authoritative kernel closure rejects {package_name} from {}",
            manifest.display()
        )),
    }
}

fn validate_expected_tree(
    observed: &[u8; 32],
    expected: &str,
    root: &Path,
    kind: &str,
) -> Result<(), String> {
    if hex(observed) != expected {
        return Err(format!(
            "reviewed {kind} closure content changed under {}",
            root.display()
        ));
    }
    Ok(())
}

fn canonical_tree_digest(root: &Path, excluded: Option<&Path>) -> Result<[u8; 32], String> {
    canonical_tree_digest_monitored(root, excluded, None)
}

fn canonical_tree_digest_monitored(
    root: &Path,
    excluded: Option<&Path>,
    mutation_journal: Option<&MutationJournal>,
) -> Result<[u8; 32], String> {
    let mut files = Vec::new();
    collect_tree_files(root, root, excluded, mutation_journal, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    if files.len() > MAX_SOURCE_TREE_FILES {
        return Err(format!(
            "authoritative source tree {} contains more than {MAX_SOURCE_TREE_FILES} files",
            root.display()
        ));
    }

    let mut total = 0_u64;
    let mut tree = Sha256::new();
    tree.update(b"fe2o3-canonical-source-tree-v1\0");
    for (relative, path) in files {
        let (digest, size) = hash_regular_file(&path, MAX_SOURCE_FILE_BYTES)?;
        total = total
            .checked_add(size)
            .ok_or_else(|| format!("authoritative source tree {} is too large", root.display()))?;
        if total > MAX_SOURCE_TREE_BYTES {
            return Err(format!(
                "authoritative source tree {} exceeds {MAX_SOURCE_TREE_BYTES} bytes",
                root.display()
            ));
        }
        tree.update(hex(&digest).as_bytes());
        tree.update(b"  ");
        tree.update(relative.as_bytes());
        tree.update(b"\n");
    }
    Ok(tree.finalize().into())
}

fn collect_tree_files(
    root: &Path,
    directory: &Path,
    excluded: Option<&Path>,
    mutation_journal: Option<&MutationJournal>,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    if excluded.is_some_and(|excluded| directory == excluded || directory.starts_with(excluded)) {
        return Ok(());
    }
    let initial = fs::symlink_metadata(directory).map_err(|error| {
        format!(
            "cannot inspect authoritative source directory {}: {error}",
            directory.display()
        )
    })?;
    if !initial.is_dir() || initial.file_type().is_symlink() {
        return Err(format!(
            "authoritative source directory must be a real directory: {}",
            directory.display()
        ));
    }
    if let Some(mutation_journal) = mutation_journal {
        mutation_journal.watch_directory(directory)?;
    }
    let entries = fs::read_dir(directory).map_err(|error| {
        format!(
            "cannot enumerate authoritative source directory {}: {error}",
            directory.display()
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "cannot enumerate authoritative source directory {}: {error}",
                directory.display()
            )
        })?;
        let path = entry.path();
        if excluded.is_some_and(|excluded| path == excluded || path.starts_with(excluded)) {
            continue;
        }
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            format!(
                "cannot inspect authoritative source {}: {error}",
                path.display()
            )
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "authoritative source closure rejects symbolic link {}",
                path.display()
            ));
        }
        if metadata.is_dir() {
            collect_tree_files(root, &path, excluded, mutation_journal, files)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .expect("recursive source path remains under its root")
                .to_str()
                .ok_or_else(|| {
                    format!("authoritative source path is not UTF-8: {}", path.display())
                })?;
            if relative.contains(['\n', '\r']) {
                return Err(format!(
                    "authoritative source path contains a line break: {}",
                    path.display()
                ));
            }
            files.push((relative.replace(std::path::MAIN_SEPARATOR, "/"), path));
        } else {
            return Err(format!(
                "authoritative source closure rejects special file {}",
                path.display()
            ));
        }
    }
    let final_metadata = fs::symlink_metadata(directory).map_err(|error| {
        format!(
            "cannot re-inspect authoritative source directory {}: {error}",
            directory.display()
        )
    })?;
    if !same_observed_object(&initial, &final_metadata) {
        return Err(format!(
            "authoritative source directory changed while observed: {}",
            directory.display()
        ));
    }
    Ok(())
}

fn hash_regular_file(path: &Path, limit: u64) -> Result<([u8; 32], u64), String> {
    let initial = fs::symlink_metadata(path).map_err(|error| {
        format!(
            "cannot inspect authoritative input {}: {error}",
            path.display()
        )
    })?;
    if !initial.is_file() || initial.file_type().is_symlink() {
        return Err(format!(
            "authoritative input must be a regular file: {}",
            path.display()
        ));
    }
    if initial.len() > limit {
        return Err(format!(
            "authoritative input {} must contain at most {limit} bytes",
            path.display()
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options.open(path).map_err(|error| {
        format!(
            "cannot open authoritative input {}: {error}",
            path.display()
        )
    })?;
    let opened = file.metadata().map_err(|error| {
        format!(
            "cannot inspect authoritative input {}: {error}",
            path.display()
        )
    })?;
    if !same_observed_object(&initial, &opened) {
        return Err(format!(
            "authoritative input changed before it was opened: {}",
            path.display()
        ));
    }
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            format!(
                "cannot read authoritative input {}: {error}",
                path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .ok_or_else(|| format!("authoritative input is too large: {}", path.display()))?;
        if size > limit {
            return Err(format!(
                "authoritative input {} exceeds {limit} bytes",
                path.display()
            ));
        }
        digest.update(&buffer[..read]);
    }
    let final_metadata = file.metadata().map_err(|error| {
        format!(
            "cannot re-inspect authoritative input {}: {error}",
            path.display()
        )
    })?;
    if size != initial.len() || !same_observed_object(&initial, &final_metadata) {
        return Err(format!(
            "authoritative input changed while it was read: {}",
            path.display()
        ));
    }
    Ok((digest.finalize().into(), size))
}

#[cfg(unix)]
fn same_observed_object(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.mode() == right.mode()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}

#[cfg(not(unix))]
fn same_observed_object(left: &Metadata, right: &Metadata) -> bool {
    left.file_type() == right.file_type()
        && left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Cargo metadata record has no string {field:?}"))
}

fn string_array(value: &Value, field: &str) -> Result<Vec<String>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("Cargo metadata has no {field:?} array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("Cargo metadata {field:?} contains a non-string"))
        })
        .collect()
}

fn append_field(snapshot: &mut Vec<u8>, field: &[u8]) {
    snapshot.extend_from_slice(&(field.len() as u64).to_le_bytes());
    snapshot.extend_from_slice(field);
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[(byte >> 4) as usize]));
        encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinned_executable_test_directory::TestDirectory;
    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;

    fn metadata(target_kind: &str, package_name: &str, links: Value) -> Value {
        serde_json::json!({
            "packages": [{
                "id": "path+file:///fixture#0.1.0",
                "name": package_name,
                "version": "0.1.0",
                "source": null,
                "checksum": null,
                "manifest_path": "/fixture/Cargo.toml",
                "links": links,
                "targets": [{"kind": [target_kind]}]
            }],
            "resolve": {
                "root": "path+file:///fixture#0.1.0",
                "nodes": [{
                    "id": "path+file:///fixture#0.1.0",
                    "dependencies": [],
                    "deps": []
                }]
            },
            "workspace_members": ["path+file:///fixture#0.1.0"],
            "workspace_default_members": ["path+file:///fixture#0.1.0"],
            "workspace_root": "/fixture"
        })
    }

    fn decode_digest(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64);
        let mut output = [0_u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            output[index] = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
        }
        output
    }

    #[test]
    fn build_closure_excludes_dev_only_dependencies() {
        let node = serde_json::json!({
            "id": "root",
            "dependencies": ["normal", "build", "dev", "mixed"],
            "deps": [
                {"pkg": "dev", "dep_kinds": [{"kind": "dev", "target": null}]},
                {"pkg": "normal", "dep_kinds": [{"kind": null, "target": null}]},
                {"pkg": "build", "dep_kinds": [{"kind": "build", "target": null}]},
                {"pkg": "mixed", "dep_kinds": [
                    {"kind": "dev", "target": null},
                    {"kind": null, "target": null}
                ]}
            ]
        });
        assert_eq!(
            build_dependency_ids(&node).unwrap(),
            vec!["build".to_owned(), "mixed".to_owned(), "normal".to_owned()]
        );
    }

    #[test]
    fn build_closure_rejects_unknown_or_malformed_dependency_kinds() {
        for dep_kinds in [
            serde_json::json!([]),
            serde_json::json!([{}]),
            serde_json::json!([{"kind": "future-kind", "target": null}]),
        ] {
            let node = serde_json::json!({
                "id": "root",
                "deps": [{"pkg": "dependency", "dep_kinds": dep_kinds}]
            });
            assert!(build_dependency_ids(&node).is_err());
        }
    }

    #[test]
    fn package_selection_is_exact_and_rejects_non_utf8_arguments() {
        assert_eq!(
            selected_package_names(&[
                "--package=beta".into(),
                "-p".into(),
                "alpha".into(),
                "--package".into(),
                "beta".into(),
            ])
            .unwrap(),
            vec!["alpha".to_owned(), "beta".to_owned()]
        );

        #[cfg(unix)]
        for argument in [
            std::ffi::OsString::from_vec(b"--package=alias-\xff".to_vec()),
            std::ffi::OsString::from_vec(b"irrelevant-\xff".to_vec()),
        ] {
            assert_eq!(
                selected_package_names(&[argument]).unwrap_err(),
                "authoritative Cargo arguments must be UTF-8"
            );
        }

        #[cfg(unix)]
        assert_eq!(
            selected_package_names(&[
                "--package".into(),
                std::ffi::OsString::from_vec(b"alias-\xff".to_vec()),
            ])
            .unwrap_err(),
            "authoritative --package requires one UTF-8 name"
        );
    }

    #[test]
    fn host_code_policy_rejects_custom_build_proc_macro_and_native_links() {
        for (kind, name, links, expected) in [
            ("custom-build", "hostile-build", Value::Null, "custom-build"),
            (
                "proc-macro",
                "hostile-macro",
                Value::Null,
                "procedural macro",
            ),
            (
                "lib",
                "hostile-links",
                Value::String("native".into()),
                "native links",
            ),
        ] {
            let record = metadata(kind, name, links);
            let package = &record["packages"][0];
            let error = validate_host_code_package(package, &[0_u8; 32]).unwrap_err();
            assert!(error.contains(expected));
            assert!(error.contains(name));
        }
    }

    #[test]
    fn tampered_registry_build_script_changes_the_reviewed_content_identity() {
        let directory = TestDirectory::new();
        fs::write(
            directory.path().join("Cargo.toml"),
            b"[package]\nname='reviewed-registry-crate'\nversion='1.0.0'\n",
        )
        .unwrap();
        let build_script = directory.path().join("build.rs");
        fs::write(&build_script, b"fn main() {}\n").unwrap();
        let reviewed = canonical_tree_digest(directory.path(), None).unwrap();
        let reviewed_hex = hex(&reviewed);
        let package = serde_json::json!({
            "name": "reviewed-registry-crate",
            "version": "1.0.0",
            "source": CRATES_IO_SOURCE,
            "manifest_path": directory.path().join("Cargo.toml"),
        });
        let trusted = [("reviewed-registry-crate", "1.0.0", reviewed_hex.as_str())];
        assert!(validate_registry_build_script_against(&package, &reviewed, &trusted,).unwrap());

        fs::write(&build_script, b"fn main() { panic!(\"tampered\") }\n").unwrap();
        let tampered = canonical_tree_digest(directory.path(), None).unwrap();
        let error =
            validate_registry_build_script_against(&package, &tampered, &trusted).unwrap_err();
        assert!(error.contains("registry build-script closure content changed"));
    }

    #[test]
    fn reviewed_external_proc_macros_require_exact_source_and_content_identity() {
        let directory = TestDirectory::new();
        fs::write(
            directory.path().join("Cargo.toml"),
            b"[package]\nname='reviewed-macro'\nversion='1.0.0'\n",
        )
        .unwrap();
        let library = directory.path().join("lib.rs");
        fs::write(&library, b"pub fn reviewed() {}\n").unwrap();
        let reviewed = canonical_tree_digest(directory.path(), None).unwrap();
        let reviewed_hex = hex(&reviewed);

        let registry = serde_json::json!({
            "name": "reviewed-macro",
            "version": "1.0.0",
            "source": CRATES_IO_SOURCE,
            "manifest_path": directory.path().join("Cargo.toml"),
        });
        let registry_trusted = [("reviewed-macro", "1.0.0", reviewed_hex.as_str())];
        assert!(
            validate_registry_proc_macro_against(&registry, &reviewed, &registry_trusted).unwrap()
        );
        let mut local_substitution = registry.clone();
        local_substitution["source"] = Value::Null;
        assert!(
            !validate_registry_proc_macro_against(
                &local_substitution,
                &reviewed,
                &registry_trusted,
            )
            .unwrap()
        );

        let git_source = "git+https://example.invalid/reviewed.git?rev=0123#0123";
        let git = serde_json::json!({
            "name": "reviewed-macro",
            "version": "1.0.0",
            "source": git_source,
            "manifest_path": directory.path().join("Cargo.toml"),
        });
        let git_trusted = [("reviewed-macro", "1.0.0", git_source, reviewed_hex.as_str())];
        assert!(validate_git_proc_macro_against(&git, &reviewed, &git_trusted).unwrap());
        let mut moving_git_reference = git.clone();
        moving_git_reference["source"] =
            Value::String("git+https://example.invalid/reviewed.git?branch=main#0123".into());
        assert!(
            !validate_git_proc_macro_against(&moving_git_reference, &reviewed, &git_trusted)
                .unwrap()
        );

        fs::write(&library, b"pub fn substituted() {}\n").unwrap();
        let tampered = canonical_tree_digest(directory.path(), None).unwrap();
        assert!(
            validate_registry_proc_macro_against(&registry, &tampered, &registry_trusted)
                .unwrap_err()
                .contains("registry proc-macro closure content changed")
        );
        assert!(
            validate_git_proc_macro_against(&git, &tampered, &git_trusted)
                .unwrap_err()
                .contains("git proc-macro closure content changed")
        );
    }

    #[test]
    fn reviewed_awint_macros_0_19_requires_exact_registry_identity() {
        let package = serde_json::json!({
            "name": "awint_macros",
            "version": "0.19.0",
            "source": CRATES_IO_SOURCE,
            "manifest_path": "/cargo/registry/awint_macros-0.19.0/Cargo.toml",
        });
        let reviewed =
            decode_digest("ee1c3c771747ccebec28a74521447163a7d3088d68b64f64b26241a6e32b8725");
        assert!(
            validate_registry_proc_macro_against(
                &package,
                &reviewed,
                &TRUSTED_REGISTRY_PROC_MACROS,
            )
            .unwrap()
        );

        let mut wrong_version = package.clone();
        wrong_version["version"] = Value::String("0.19.1".into());
        assert!(
            !validate_registry_proc_macro_against(
                &wrong_version,
                &reviewed,
                &TRUSTED_REGISTRY_PROC_MACROS,
            )
            .unwrap()
        );
        assert!(
            validate_registry_proc_macro_against(
                &package,
                &[0_u8; 32],
                &TRUSTED_REGISTRY_PROC_MACROS,
            )
            .unwrap_err()
            .contains("registry proc-macro closure content changed")
        );
    }

    #[test]
    fn reviewed_workspace_pliron_macro_requires_exact_pinned_source_and_content() {
        let manifest: toml::Value = toml::from_str(include_str!("../../../Cargo.toml")).unwrap();
        let dependency = &manifest["workspace"]["dependencies"]["pliron"];
        let repository = dependency["git"].as_str().unwrap();
        let revision = dependency["rev"].as_str().unwrap();
        let source = format!("git+{repository}?rev={revision}#{revision}");
        let lock: toml::Value = toml::from_str(include_str!("../../../Cargo.lock")).unwrap();
        let macros: Vec<_> = lock["package"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|package| package["name"].as_str() == Some("pliron-derive"))
            .collect();
        assert_eq!(macros.len(), 1);
        assert_eq!(macros[0]["source"].as_str(), Some(source.as_str()));
        assert_eq!(macros[0]["version"].as_str(), Some("0.17.0"));

        let reviewed =
            decode_digest("18abd61218886753068c38294931388e53e90f17b85cd8660b83ec93280bc824");
        let package = serde_json::json!({
            "name": "pliron-derive",
            "version": "0.17.0",
            "source": source,
            "links": Value::Null,
            "manifest_path": "/cargo/git/checkouts/pliron/revision/pliron-derive/Cargo.toml",
            "targets": [{"kind": ["proc-macro"]}],
        });
        validate_host_code_package(&package, &reviewed).unwrap();

        for (field, value) in [
            ("name", serde_json::json!("substituted-macro")),
            ("version", serde_json::json!("0.17.1")),
            ("source", Value::Null),
            (
                "source",
                serde_json::json!(format!("git+{repository}?branch=main#{revision}")),
            ),
            (
                "source",
                serde_json::json!(format!(
                    "git+{repository}?rev={revision}#{}",
                    "0".repeat(40)
                )),
            ),
        ] {
            let mut substituted = package.clone();
            substituted[field] = value;
            assert!(validate_host_code_package(&substituted, &reviewed).is_err());
        }
        assert!(
            validate_host_code_package(&package, &[0_u8; 32])
                .unwrap_err()
                .contains("git proc-macro closure content changed")
        );
    }

    #[test]
    fn reviewed_workspace_host_code_requires_exact_local_or_git_revision_identity() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cargo-fe2o3 has a workspace crates directory")
            .join("fe2o3-macros");
        let digest = canonical_tree_digest(&root, None).unwrap();
        let local = serde_json::json!({
            "name": "fe2o3-macros",
            "version": "0.1.0",
            "source": Value::Null,
            "manifest_path": root.join("Cargo.toml"),
        });
        validate_reviewed_fe2o3_macros(&local, &digest).unwrap();

        let git = serde_json::json!({
            "name": "fe2o3-macros",
            "version": "0.1.0",
            "source": TRUSTED_FE2O3_EXTERNAL_SOURCE,
            "links": Value::Null,
            "manifest_path": format!(
                "/cargo/git/checkouts/fe2o3/revision/crates/fe2o3-macros/Cargo.toml"
            ),
            "targets": [{"kind": ["proc-macro"]}],
        });
        let external_macro_digest = decode_digest(TRUSTED_FE2O3_EXTERNAL_MACROS_TREE);
        validate_host_code_package(&git, &external_macro_digest).unwrap();

        for substituted_digest in [
            [0_u8; 32],
            decode_digest("0456f54f422af97e3292bc14622ab618026bf0627ec9a80c794fe8a5b354d052"),
            decode_digest("7486446880845fec8c9d9db54c0fb4239b4561357414c7c1d10d6b92a10092a5"),
            decode_digest("65aa1e5e26c40735f10562a847d46dcff5835cb43ad9355002d8a83c93eb38e2"),
            decode_digest("52bc00864c6a96e00ce68a40af13402ce492bc2e99127c74ab062ccae80fbbfa"),
            decode_digest("7486446880845fec8c9d9db54c0fb4239b4561357414c7c1d10d6b92a10092a5"),
            decode_digest("607b499eff7e8d8475af6a1c526b8d7da2ad415152086b16f582b9ba555e9b8c"),
            external_macro_digest,
        ] {
            assert!(
                validate_reviewed_fe2o3_macros(&local, &substituted_digest)
                    .unwrap_err()
                    .contains("reviewed proc-macro closure content changed")
            );
        }
        assert!(
            validate_host_code_package(&git, &digest)
                .unwrap_err()
                .contains("reviewed proc-macro closure content changed")
        );
        let mut relocated_local = local.clone();
        relocated_local["manifest_path"] = serde_json::json!(root.join("other/Cargo.toml"));
        assert!(
            validate_reviewed_fe2o3_macros(&relocated_local, &digest)
                .unwrap_err()
                .contains("unreviewed procedural macro")
        );

        let hip = serde_json::json!({
            "name": "fe2o3-hip-sys",
            "version": "0.1.0",
            "source": git["source"].clone(),
            "links": "amdhip64",
            "manifest_path":
                "/cargo/git/checkouts/fe2o3/revision/crates/fe2o3-hip-sys/Cargo.toml",
            "targets": [{"kind": ["custom-build"]}],
        });
        assert!(
            validate_host_code_package(&hip, &[0_u8; 32])
                .unwrap_err()
                .contains("rejects native links package")
        );

        for (field, value) in [
            ("version", Value::String("0.1.1".into())),
            ("source", Value::String(CRATES_IO_SOURCE.into())),
        ] {
            let mut substituted = local.clone();
            substituted[field] = value;
            assert!(
                validate_reviewed_fe2o3_macros(&substituted, &digest)
                    .unwrap_err()
                    .contains("unreviewed procedural macro")
            );
        }

        for source in [
            "git+https://github.com/harsh-nod/fe2o3.git?rev=d955209099c7b434dfceb69e1152d948dab76b22#ffffffffffffffffffffffffffffffffffffffff",
            "git+https://example.invalid/fe2o3.git?rev=d955209099c7b434dfceb69e1152d948dab76b22#d955209099c7b434dfceb69e1152d948dab76b22",
            "git+https://github.com/harsh-nod/fe2o3.git?branch=main#d955209099c7b434dfceb69e1152d948dab76b22",
            "git+https://github.com/harsh-nod/fe2o3.git?rev=0123456789abcdef0123456789abcdef01234567#0123456789abcdef0123456789abcdef01234567",
        ] {
            let mut substituted = git.clone();
            substituted["source"] = Value::String(source.into());
            assert!(validate_host_code_package(&substituted, &external_macro_digest).is_err());
        }

        {
            let mut missing_source = git.clone();
            missing_source
                .as_object_mut()
                .unwrap()
                .remove("source")
                .unwrap();
            assert!(validate_host_code_package(&missing_source, &external_macro_digest).is_err());

            let mut relative_manifest = git.clone();
            relative_manifest["manifest_path"] =
                Value::String("crates/fe2o3-macros/Cargo.toml".to_owned());
            assert!(
                validate_host_code_package(&relative_manifest, &external_macro_digest).is_err()
            );

            let mut wrong_layout = git.clone();
            wrong_layout["manifest_path"] = Value::String(
                "/cargo/git/checkouts/fe2o3/revision/fe2o3-macros/Cargo.toml".to_owned(),
            );
            assert!(validate_host_code_package(&wrong_layout, &external_macro_digest).is_err());

            assert!(validate_host_code_package(&git, &[0_u8; 32]).is_err());
        }
    }

    #[test]
    fn canonical_closure_identity_covers_manifests_and_nested_sources() {
        let directory = TestDirectory::new();
        let source = directory.path().join("src");
        fs::create_dir(&source).unwrap();
        let manifest = directory.path().join("Cargo.toml");
        let library = source.join("lib.rs");
        let fixture = directory.path().join("tests/fixtures/worker");
        fs::create_dir_all(&fixture).unwrap();
        let fixture_lock = fixture.join("Cargo.lock");
        fs::write(&fixture_lock, b"version = 4\n").unwrap();
        fs::write(&manifest, b"[package]\nname='fixture'\nversion='0.1.0'\n").unwrap();
        fs::write(&library, b"pub fn reviewed() {}\n").unwrap();
        let reviewed = canonical_tree_digest(directory.path(), None).unwrap();

        fs::write(&manifest, b"[package]\nname='fixture'\nversion='0.1.1'\n").unwrap();
        let manifest_drift = canonical_tree_digest(directory.path(), None).unwrap();
        assert_ne!(manifest_drift, reviewed);

        fs::write(&manifest, b"[package]\nname='fixture'\nversion='0.1.0'\n").unwrap();
        fs::write(&library, b"pub fn substituted() {}\n").unwrap();
        let source_drift = canonical_tree_digest(directory.path(), None).unwrap();
        assert_ne!(source_drift, reviewed);

        fs::write(
            &fixture_lock,
            b"version = 4\n[[package]]\nname = 'dependency'\nversion = '0.1.0'\n",
        )
        .unwrap();
        let lock_drift = canonical_tree_digest(directory.path(), None).unwrap();
        assert_ne!(lock_drift, source_drift);
        let pinned_before_lock = hex(&source_drift);
        validate_expected_tree(
            &source_drift,
            &pinned_before_lock,
            directory.path(),
            "fixture",
        )
        .unwrap();
        assert!(
            validate_expected_tree(
                &lock_drift,
                &pinned_before_lock,
                directory.path(),
                "fixture",
            )
            .unwrap_err()
            .contains("reviewed fixture closure content changed")
        );
    }

    #[test]
    fn reviewed_workspace_host_code_trees_match_their_complete_pins() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cargo-fe2o3 has a workspace crates directory");
        let root = crates.join("fe2o3-macros");
        assert_eq!(
            hex(&canonical_tree_digest(&root, None).unwrap()),
            TRUSTED_FE2O3_MACROS_TREE
        );
        assert_eq!(
            TRUSTED_FE2O3_EXTERNAL_MACROS_TREE,
            "a6ae6e79c48ee48389411aee2db6b438599e525009e45f5773d5e0d3ba57efcc"
        );
    }

    #[test]
    fn proc_macro_mutation_after_preflight_fails_revalidation() {
        let directory = TestDirectory::new();
        let source = directory.path().join("src");
        fs::create_dir(&source).unwrap();
        fs::write(
            directory.path().join("Cargo.toml"),
            b"[lib]\nproc-macro=true\n",
        )
        .unwrap();
        let library = source.join("lib.rs");
        fs::write(&library, b"pub fn reviewed() {}\n").unwrap();
        let observed = ObservedSourceTree::capture(
            "test".to_owned(),
            directory.path().to_path_buf(),
            None,
            None,
        )
        .unwrap();

        fs::write(&library, b"pub fn injected_after_preflight() {}\n").unwrap();
        let error = observed.revalidate().unwrap_err();
        assert!(error.contains("source closure changed after preflight"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn restored_proc_macro_mutation_remains_in_the_journal() {
        let directory = TestDirectory::new();
        let source = directory.path().join("src");
        fs::create_dir(&source).unwrap();
        let library = source.join("lib.rs");
        let reviewed = b"pub fn reviewed() {}\n";
        fs::write(&library, reviewed).unwrap();
        let journal = MutationJournal::new(Vec::new()).unwrap();
        let observed = ObservedSourceTree::capture(
            "test".to_owned(),
            directory.path().to_path_buf(),
            None,
            Some(&journal),
        )
        .unwrap();
        journal.ensure_quiet().unwrap();

        fs::write(&library, b"pub fn injected() {}\n").unwrap();
        fs::write(&library, reviewed).unwrap();
        observed.revalidate().unwrap();
        let error = journal.ensure_quiet().unwrap_err();
        assert!(error.contains("source closure mutated after preflight"));
    }
}

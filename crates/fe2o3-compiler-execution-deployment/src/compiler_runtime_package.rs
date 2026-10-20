//! Administrative record assembly. Measured bytes and generated hashes are not approval.
use std::{
    collections::BTreeSet,
    fmt,
    fs::File,
    mem::size_of,
    path::{Component, Path},
};

use super::{
    DeploymentVerificationErrorKindV1 as Kind, DeploymentVerificationErrorV1 as Deployment,
    FileSpecV1, ObjectSnapshotV1 as Snapshot, changed, invalid, io_error, open_beneath,
    open_bundle_root, snapshot, validate_directory, validate_directory_mode,
    validate_source_metadata, verify_compiler_runtime_deployment_v1,
};
use fe2o3_build_authority::{
    COMPILER_APPROVAL_POLICY_RUNTIME_ENFORCEMENT_VERSION_V2 as ENFORCEMENT,
    COMPILER_APPROVAL_POLICY_WORK_V2 as POLICY_WORK,
    COMPILER_RUNTIME_MANIFEST_MAX_ENTRIES_V1 as MAX_FILES,
    COMPILER_RUNTIME_MANIFEST_STORAGE_V1 as MANIFEST_STORAGE,
    COMPILER_RUNTIME_MANIFEST_WORK_V1 as MANIFEST_WORK, CompilerApprovalPolicyV2 as Policy,
    CompilerClosureV2 as Closure, CompilerRuntimeEntryV1 as Entry,
    CompilerRuntimeManifestV1 as Manifest, CompilerRuntimeRoleV1 as Role,
};
use fe2o3_compiler_execution_protocol::{
    COMPILER_EXECUTION_CLIENT_PROFILE_BYTES_V3 as PROFILE_BYTES,
    COMPILER_EXECUTION_CLIENT_PROFILE_STORAGE_V3 as PROFILE_STORAGE,
    COMPILER_EXECUTION_CLIENT_PROFILE_WORK_V3 as PROFILE_WORK,
    CompilerExecutionClientProfileV3 as Profile,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use rustix::fs::{Mode, OFlags, ResolveFlags, fchmod, fstat, fsync, mkdirat, openat, openat2};
use sha2::{Digest, Sha256};

const MAX_RECIPE: usize = 64 * 1024;
const MAX_PATH: usize = 4096;
const CHUNK: usize = 64 * 1024;
const PROFILE: &str = "etc/fe2o3/compiler-execution/client-profile-v3";
const HEADER: &str = "fe2o3-compiler-runtime-package-input-v1";
type Result<T> = std::result::Result<T, CompilerRuntimePackageErrorV1>;

/// Refusal preserves the original account's accepted work, peak and denials.
#[derive(Debug)]
pub enum CompilerRuntimePackageErrorV1 {
    /// The original budget refused work, storage, arithmetic or an owner floor.
    Resource(Resource),
    /// Recipe, filesystem, canonical record or completed-bundle verification failed.
    Deployment(Deployment),
}
impl From<Resource> for CompilerRuntimePackageErrorV1 {
    fn from(e: Resource) -> Self {
        Self::Resource(e)
    }
}
impl From<Deployment> for CompilerRuntimePackageErrorV1 {
    fn from(e: Deployment) -> Self {
        Self::Deployment(e)
    }
}
impl fmt::Display for CompilerRuntimePackageErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(e) => e.fmt(f),
            Self::Deployment(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for CompilerRuntimePackageErrorV1 {}
fn refuse(message: &'static str) -> CompilerRuntimePackageErrorV1 {
    invalid(Kind::InvalidManifest, message).into()
}

/// Operator-supplied UNAPPROVED inputs, never an approved closure or runtime owner.
/// Per-file expected hashes must be selected through the operator's independent
/// release review. Neither parsing them nor matching bytes proves that review.
/// The recipe grammar is documented in docs/compiler-runtime-deployment.md.
pub struct CompilerRuntimePackagePlanV1 {
    manifest: Manifest,
    helper_uid: u32,
    helper_gid: u32,
}
impl CompilerRuntimePackagePlanV1 {
    /// Complete read/parse work, including the existing canonical manifest codec.
    pub const READ_WORK: usize = 8 * 1024 * 1024 + MANIFEST_WORK;
    /// Read/parse scratch including the bounded recipe and canonical codec frames.
    pub const READ_STORAGE: usize = 1024 * 1024;
    /// Unreserved returned owner charge. Reserve before retaining/using a plan.
    pub const STORAGE: usize = size_of::<Self>();

    /// Reads one single-link mode0444 recipe without following any symlink.
    /// One original budget is used throughout. Entry storage is restored on exit.
    /// Keep the recipe and its parent path stable for the read. On success reserve
    /// `Self::STORAGE` on this same account before retaining the returned plan;
    /// that charge is not included in the restored read/parse scratch reservation.
    pub fn read(path: &Path, b: &mut Budget<'_>) -> Result<Self> {
        b.with_prepaid_scope(
            0,
            8,
            Self::READ_WORK - MANIFEST_WORK,
            Self::READ_STORAGE,
            |b| {
                canonical_path(path)?;
                let file = File::from(
                    openat2(
                        rustix::fs::CWD,
                        path,
                        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NONBLOCK,
                        Mode::empty(),
                        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
                    )
                    .map_err(|e| io_error("open runtime package recipe", e))?,
                );
                let initial = inspect(&file)?;
                let initial = metadata(&file, initial, 0o444, initial.byte_len)?;
                let n = usize::try_from(initial.byte_len).map_err(|_| Resource::Arithmetic)?;
                if n == 0 || n > MAX_RECIPE {
                    return Err(refuse("recipe exceeds its nonempty bound"));
                }
                let mut bytes = [0; MAX_RECIPE];
                read_at(&file, &mut bytes[..n], 0)?;
                if inspect(&file)? != initial {
                    return Err(changed("package recipe changed").into());
                }
                parse(&bytes[..n], b)
            },
        )
    }

    /// Conservative complete schedule, not an execution quota or account waiver.
    pub fn quota(&self) -> Result<CompilerRuntimePackageQuotaV1> {
        let bytes =
            usize::try_from(self.manifest.total_file_bytes()).map_err(|_| Resource::Arithmetic)?;
        // Covers copying, full verifier reads/seals, all descriptor/path rechecks,
        // final origin-content revalidation and bounded retirement. Eight extra
        // work units per code byte fund that additional streaming read/hash.
        // Four full backings include verifier overlap; hashing adds no backing.
        let local_work = bytes
            .checked_mul(72)
            .and_then(|n| n.checked_add(64 * 1024 * 1024))
            .ok_or(Resource::Arithmetic)?;
        let local_storage = bytes
            .checked_mul(4)
            .and_then(|n| n.checked_add(16 * 1024 * 1024))
            .ok_or(Resource::Arithmetic)?;
        Ok(CompilerRuntimePackageQuotaV1 {
            work: local_work
                .checked_add(PROFILE_WORK)
                .and_then(|n| n.checked_add(POLICY_WORK))
                .ok_or(Resource::Arithmetic)?,
            storage: local_storage
                .checked_add(PROFILE_STORAGE)
                .ok_or(Resource::Arithmetic)?,
        })
    }
}

const _: () = assert!(
    CompilerRuntimePackagePlanV1::READ_STORAGE
        >= MAX_RECIPE
            + MANIFEST_STORAGE
            + CompilerRuntimePackagePlanV1::STORAGE
            + size_of::<[Entry<'static>; MAX_FILES]>()
            + 8 * MAX_PATH
);

/// Additional peak storage over the original caller floor, and total operation work.
#[derive(Clone, Copy, Debug)]
pub struct CompilerRuntimePackageQuotaV1 {
    work: usize,
    storage: usize,
}
impl CompilerRuntimePackageQuotaV1 {
    /// Complete operation work, including the same-ledger nested codecs.
    pub const fn work(self) -> usize {
        self.work
    }
    /// Additional peak storage; retain the original plan/caller floor separately.
    pub const fn storage(self) -> usize {
        self.storage
    }
}

/// UNAPPROVED raw record hashes. Independent release review must occur separately.
/// This scalar report cannot admit a compiler, install files or execute anything.
#[derive(Clone, Copy, Debug)]
pub struct PackagedCompilerRuntimeDeploymentV1 {
    policy: [u8; 32],
    manifest: [u8; 32],
    count: usize,
}
impl PackagedCompilerRuntimeDeploymentV1 {
    /// Raw SHA-256 of the emitted policy; UNAPPROVED until independent review.
    pub const fn policy_sha256(self) -> [u8; 32] {
        self.policy
    }
    /// Raw SHA-256 of the emitted manifest; not its domain-separated identity.
    pub const fn manifest_sha256(self) -> [u8; 32] {
        self.manifest
    }
    /// Number of measured code files, excluding the two canonical records.
    pub const fn file_count(self) -> usize {
        self.count
    }
}

fn parse(bytes: &[u8], b: &mut Budget<'_>) -> Result<CompilerRuntimePackagePlanV1> {
    if bytes.len() > MAX_RECIPE
        || !bytes.is_ascii()
        || !bytes.ends_with(b"\n")
        || bytes.contains(&b'\r')
    {
        return Err(refuse(
            "recipe must be bounded canonical ASCII with a final newline",
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| refuse("invalid recipe text"))?;
    let mut lines = text[..text.len() - 1].split('\n');
    if lines.next() != Some(HEADER) {
        return Err(refuse("wrong package recipe header"));
    }
    let mut field = |name: &str| -> Result<&str> {
        let (key, value) = lines
            .next()
            .and_then(|s| s.split_once('='))
            .ok_or_else(|| refuse("missing recipe field"))?;
        if key != name {
            return Err(refuse("recipe fields differ from canonical order"));
        }
        Ok(value)
    };
    let closure = Closure::new(
        hex(field("cargo_sha256")?)?,
        hex(field("trampoline_sha256")?)?,
        hex(field("wrapper_sha256")?)?,
        hex(field("rustc_sha256")?)?,
        hex(field("rustc_tree_sha256")?)?,
        hex(field("backend_sha256")?)?,
    )
    .map_err(|_| refuse("invalid complete compiler closure"))?;
    let proof = hex(field("proof_runtime_identity")?)?;
    let uid =
        u32::try_from(decimal(field("helper_uid")?)?).map_err(|_| refuse("helper UID overflow"))?;
    let gid =
        u32::try_from(decimal(field("helper_gid")?)?).map_err(|_| refuse("helper GID overflow"))?;
    if [uid, gid].iter().any(|n| *n == 0 || *n == u32::MAX) {
        return Err(refuse("invalid helper credentials"));
    }
    let empty = Entry {
        role: Role::SharedLibrary,
        path: "",
        length: 0,
        sha256: [0; 32],
    };
    let mut entries = [empty; MAX_FILES];
    let mut count = 0;
    for line in lines {
        if count == MAX_FILES {
            return Err(refuse("too many package entries"));
        }
        let mut parts = line.split(' ');
        let role = match parts.next() {
            Some("rustc") => Role::Rustc,
            Some("backend") => Role::CodegenBackend,
            Some("proc-macro") => Role::Fe2o3ProcMacro,
            Some("interpreter") => Role::ElfInterpreter,
            Some("proof-helper") => Role::ProofExecutorHelper,
            Some("shared-library") => Role::SharedLibrary,
            _ => return Err(refuse("unknown compiler runtime role")),
        };
        let length = decimal(parts.next().ok_or_else(|| refuse("missing entry length"))?)?;
        let sha256 = hex(parts.next().ok_or_else(|| refuse("missing entry digest"))?)?;
        let path = parts.next().ok_or_else(|| refuse("missing entry path"))?;
        if parts.next().is_some() {
            return Err(refuse("extra entry fields"));
        }
        entries[count] = Entry {
            role,
            path,
            length,
            sha256,
        };
        count += 1;
    }
    let manifest = Manifest::new(closure, proof, &entries[..count], |w| b.charge_work(w)).map_err(
        |e| match e {
            fe2o3_build_authority::CompilerRuntimeManifestErrorV1::Charge(e) => e.into(),
            _ => refuse("invalid canonical compiler runtime inventory"),
        },
    )?;
    // Canonical ordering alone permits a file that is an ancestor of another.
    for (i, entry) in entries[..count].iter().enumerate() {
        if entries[..i].iter().any(|prior| {
            entry
                .path
                .strip_prefix(prior.path)
                .is_some_and(|s| s.starts_with('/'))
        }) {
            return Err(refuse("runtime file is also a directory"));
        }
    }
    Ok(CompilerRuntimePackagePlanV1 {
        manifest,
        helper_uid: uid,
        helper_gid: gid,
    })
}
fn decimal(value: &str) -> Result<u64> {
    if value.is_empty() || value.len() > 20 || value.starts_with('0') {
        return Err(refuse("noncanonical positive decimal"));
    }
    value.bytes().try_fold(0u64, |n, c| {
        if !c.is_ascii_digit() {
            return Err(refuse("nondecimal recipe value"));
        }
        n.checked_mul(10)
            .and_then(|n| n.checked_add(u64::from(c - b'0')))
            .ok_or_else(|| refuse("decimal overflow"))
    })
}
fn hex(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(refuse("digest must be 64 lowercase hexadecimal characters"));
    }
    let mut bytes = [0; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[2 * i..2 * i + 2], 16)
            .map_err(|_| refuse("invalid digest"))?;
    }
    Ok(bytes)
}

/// Creates a NEW unapproved bundle, never an installation or approved runtime.
///
/// Source paths are relative to a mode0700 tree owned by the invoking UID/GID;
/// directories are 0700 and files have their exact role mode, one link, no xattrs.
/// The profile is read only at the fixed V3 suffix beneath a private root-owned
/// mode0700 offline root. Its parents must be root-owned mode0755.
/// All paths are absolute/canonical. Destination's private parent must exist;
/// destination must not, and must lie outside the source/profile trees. The
/// original source/profile roots, every source directory/file, their path edges
/// and the destination parent must remain stable throughout the call. Exclude
/// administrative concurrent writers; custody rechecks are point-in-time checks,
/// not protection against privileged filesystem manipulation.
///
/// Retain the plan's STORAGE on the original budget. This operation restores
/// entry storage, including on unwind, and does not refund work/denial history.
/// Nested profile/record codecs charge that same ledger. Generated output may
/// remain after failure and is explicitly unapproved: no rollback, overwrite or
/// reuse. Single-attempt I/O does not bound filesystem latency or allocator RSS.
/// The returned fixed-size scalar report belongs to the caller's frame. The
/// compiler closure and proof-runtime pins are supplied inputs, not independently
/// remeasured Cargo/toolchain/proof-runtime evidence from this packaging call.
pub fn package_compiler_runtime_deployment_v1(
    plan: &CompilerRuntimePackagePlanV1,
    source: &Path,
    profile_root: &Path,
    destination: &Path,
    b: &mut Budget<'_>,
) -> Result<PackagedCompilerRuntimeDeploymentV1> {
    package_using(
        plan,
        source,
        profile_root,
        destination,
        (0, 0),
        b,
        &mut |_| Ok(()),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Copied,
    Verified,
}
fn package_using(
    plan: &CompilerRuntimePackagePlanV1,
    source_path: &Path,
    profile_path: &Path,
    destination: &Path,
    profile_owner: (u32, u32),
    b: &mut Budget<'_>,
    hook: &mut impl FnMut(Step) -> Result<()>,
) -> Result<PackagedCompilerRuntimeDeploymentV1> {
    let quota = plan.quota()?;
    let mut created = false;
    let result = b.with_prepaid_scope(CompilerRuntimePackagePlanV1::STORAGE, 8,
        quota.work - PROFILE_WORK - POLICY_WORK, quota.storage - PROFILE_STORAGE, |b| {
        // Check the complete nested peak before I/O; do not narrow or replace
        // the account. The nested decoder then uses this same available space.
        b.reserve_storage(PROFILE_STORAGE)?;
        b.release_storage(PROFILE_STORAGE)?;
        for path in [source_path, profile_path, destination] { canonical_path(path)?; }
        let owner = (rustix::process::geteuid().as_raw(), rustix::process::getegid().as_raw());
        let source = Directory::open(source_path, owner)?;
        let profile_root = Directory::open(profile_path, profile_owner)?;
        let profile_dirs = ["etc", "etc/fe2o3", "etc/fe2o3/compiler-execution"]
            .map(|p| Directory::beneath(&profile_root.file, p, profile_owner, 0o755))
            .into_iter().collect::<Result<Vec<_>>>()?;
        let profile_file = open_beneath(&profile_root.file, PROFILE, false)?;
        let profile_snapshot = metadata(&profile_file, profile_root.initial, 0o444, PROFILE_BYTES as u64)?;
        let mut profile_bytes = [0; PROFILE_BYTES];
        read_at(&profile_file, &mut profile_bytes, 0)?;
        if inspect(&profile_file)? != profile_snapshot { return Err(changed("V3 profile changed during read").into()); }
        let (profile, charge) = Profile::decode(&profile_bytes, b)
            .map_err(|e| -> CompilerRuntimePackageErrorV1 { match e {
                fe2o3_compiler_execution_protocol::CompilerExecutionClientProfileErrorV3::Resource(e) => e.into(),
                e => invalid(Kind::InvalidManifest, format!("V3 profile: {e}")).into(),
            } })?;
        b.reserve_storage(charge.additional_storage())?;
        let anchor = profile.external_anchor_service();
        if [profile.supervisor_uid(), anchor.uid()].contains(&plan.helper_uid)
            || [profile.supervisor_gid(), anchor.gid()].contains(&plan.helper_gid) {
            return Err(refuse("proof-helper credentials alias a V3 service"));
        }
        let policy = Policy::new(plan.manifest.compiler_closure(), *profile.identity().as_bytes(),
            *plan.manifest.identity(), ENFORCEMENT, plan.helper_uid, plan.helper_gid, |w| b.charge_work(w))
            .map_err(|e| -> CompilerRuntimePackageErrorV1 { match e {
                fe2o3_build_authority::CompilerApprovalPolicyErrorV2::Framing(
                    fe2o3_build_authority::CompilerApprovalPolicyErrorV1::Charge(e)) => e.into(),
                e => invalid(Kind::InvalidManifest, format!("compiler policy: {e}")).into(),
            } })?;
        let mut paths = BTreeSet::new();
        for entry in plan.manifest.entries() {
            for (i, _) in entry.path.match_indices('/') { paths.insert(&entry.path[..i]); }
        }
        let mut directories = Vec::with_capacity(paths.len());
        for path in &paths { directories.push(Directory::beneath(&source.file, path, owner, 0o700)?); }
        let mut inputs = Vec::with_capacity(plan.manifest.entries().len());
        for entry in plan.manifest.entries() {
            let file = open_beneath(&source.file, entry.path, false)?;
            let initial = metadata(&file, source.initial, entry.role.protected_mode(), entry.length)?;
            inputs.push((file, initial));
        }
        let parent_path = destination.parent().ok_or_else(|| refuse("destination has no parent"))?;
        if destination.starts_with(source_path) || destination.starts_with(profile_path) {
            return Err(refuse("destination must be outside the retained input trees"));
        }
        let parent = Directory::open(parent_path, owner)?;
        let name = destination.file_name().ok_or_else(|| refuse("destination has no name"))?;
        mkdirat(&parent.file, name, Mode::from_raw_mode(0o700)).map_err(|e| io_error("create new unapproved bundle", e))?;
        created = true;
        let output = File::from(openat2(&parent.file, name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty(),
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS | ResolveFlags::NO_XDEV)
            .map_err(|e| io_error("retain newly created bundle", e))?);
        fchmod(&output, Mode::from_raw_mode(0o700)).map_err(|e| io_error("set private package mode", e))?;
        let output_owner = validate_directory(&output, Some(owner), "new package root")?;
        let runtime = create_directory(&output, "runtime")?;
        for path in &paths { create_directory(&runtime, path)?; }
        let mut outputs = Vec::with_capacity(inputs.len() + 2);
        for ((file, _), entry) in inputs.iter().zip(plan.manifest.entries()) {
            outputs.push(copy(file, &runtime, entry)?);
        }
        let report = PackagedCompilerRuntimeDeploymentV1 {
            policy: Sha256::digest(policy.canonical_bytes()).into(),
            manifest: Sha256::digest(plan.manifest.canonical_bytes()).into(), count: inputs.len(),
        };
        outputs.push(write_record(&output, "policy-v2", policy.canonical_bytes())?);
        outputs.push(write_record(&output, "compiler-runtime-manifest-v1", plan.manifest.canonical_bytes())?);
        for path in paths.iter().rev() { sync(&open_beneath(&runtime, path, true)?)?; }
        sync(&runtime)?; sync(&output)?; sync(&parent.file)?;
        let completed = inspect(&output)?;
        let mut completed_dirs = Vec::with_capacity(paths.len() + 1);
        completed_dirs.push(Directory::beneath(&output, "runtime", owner, 0o700)?);
        for path in &paths {
            completed_dirs.push(Directory::beneath(&output, &format!("runtime/{path}"), owner, 0o700)?);
        }
        hook(Step::Copied)?;
        let verified = verify_compiler_runtime_deployment_v1(destination, report.policy, report.manifest)?;
        hook(Step::Verified)?;
        if verified.manifest() != &plan.manifest || verified.policy() != &policy {
            return Err(changed("verified package differs from measured records").into());
        }
        for (file, initial) in &outputs {
            if inspect(file)? != *initial { return Err(changed("created package file changed").into()); }
        }
        for directory in &completed_dirs { directory.revalidate_beneath(&output, owner, 0o700)?; }
        for ((file, initial), entry) in inputs.iter().zip(plan.manifest.entries()) {
            if metadata(file, source.initial, entry.role.protected_mode(), entry.length)? != *initial
                || metadata(&open_beneath(&source.file, entry.path, false)?, source.initial,
                    entry.role.protected_mode(), entry.length)? != *initial {
                return Err(changed("original package source changed or was replaced").into());
            }
        }
        for directory in &directories { directory.revalidate_beneath(&source.file, owner, 0o700)?; }
        for directory in &profile_dirs { directory.revalidate_beneath(&profile_root.file, profile_owner, 0o755)?; }
        let current_profile = open_beneath(&profile_root.file, PROFILE, false)?;
        let mut current_bytes = [0; PROFILE_BYTES];
        read_at(&profile_file, &mut current_bytes, 0)?;
        if current_bytes != profile_bytes || metadata(&profile_file, profile_root.initial, 0o444, PROFILE_BYTES as u64)? != profile_snapshot
            || metadata(&current_profile, profile_root.initial, 0o444, PROFILE_BYTES as u64)? != profile_snapshot {
            return Err(changed("original profile changed or was replaced").into());
        }
        source.revalidate(source_path, owner)?;
        profile_root.revalidate(profile_path, profile_owner)?;
        // Creating the bundle changes the parent's timestamps/link count. Its
        // ownership and original inode still must match the canonical parent.
        let named_parent = validate_directory(&open_bundle_root(parent_path)?, Some(owner), "package parent")?;
        if (named_parent.device, named_parent.inode) != (parent.initial.device, parent.initial.inode)
            || inspect(&output)? != completed || inspect(&open_bundle_root(destination)?)? != completed
            || (completed.device, completed.inode) != (output_owner.device, output_owner.inode) {
            return Err(changed("package root or parent was displaced").into());
        }
        Ok(report)
    });
    result.map_err(|e| match e {
        CompilerRuntimePackageErrorV1::Deployment(e) if created => invalid(
            e.kind(),
            format!(
                "unapproved partial package may remain at {}; no rollback or reuse: {e}",
                destination.display()
            ),
        )
        .into(),
        e => e,
    })
}

struct Directory {
    file: File,
    initial: Snapshot,
    path: String,
}
impl Directory {
    fn open(path: &Path, owner: (u32, u32)) -> Result<Self> {
        let file = open_bundle_root(path)?;
        let initial = validate_directory(&file, Some(owner), "package root")?;
        Ok(Self {
            file,
            initial,
            path: String::new(),
        })
    }
    fn beneath(root: &File, path: &str, owner: (u32, u32), mode: u32) -> Result<Self> {
        let file = open_beneath(root, path, true)?;
        let initial = validate_directory_mode(&file, Some(owner), mode, "package input directory")?;
        Ok(Self {
            file,
            initial,
            path: path.to_owned(),
        })
    }
    fn revalidate(&self, path: &Path, owner: (u32, u32)) -> Result<()> {
        if validate_directory(&self.file, Some(owner), "retained package root")? != self.initial
            || validate_directory(&open_bundle_root(path)?, Some(owner), "named package root")?
                != self.initial
        {
            return Err(changed("package input root changed").into());
        }
        Ok(())
    }
    fn revalidate_beneath(&self, root: &File, owner: (u32, u32), mode: u32) -> Result<()> {
        if validate_directory_mode(&self.file, Some(owner), mode, "retained input directory")?
            != self.initial
            || validate_directory_mode(
                &open_beneath(root, &self.path, true)?,
                Some(owner),
                mode,
                "named input directory",
            )? != self.initial
        {
            return Err(changed("package input directory changed").into());
        }
        Ok(())
    }
}
fn canonical_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path.as_os_str().len() > MAX_PATH
        || path.file_name().is_none()
        || path
            .components()
            .any(|p| matches!(p, Component::CurDir | Component::ParentDir))
        || path
            .components()
            .collect::<std::path::PathBuf>()
            .as_os_str()
            != path.as_os_str()
    {
        return Err(refuse(
            "package path must be bounded absolute and canonical",
        ));
    }
    Ok(())
}
fn inspect(file: &File) -> Result<Snapshot> {
    Ok(snapshot(
        &fstat(file).map_err(|e| io_error("inspect package object", e))?,
    ))
}
fn metadata(file: &File, owner: Snapshot, mode: u32, length: u64) -> Result<Snapshot> {
    let value = validate_source_metadata(
        file,
        owner,
        FileSpecV1 {
            source: "runtime package input",
            install: "",
            mode,
            max_bytes: length,
        },
        None,
    )?;
    if value.byte_len != length {
        return Err(refuse("package source length differs"));
    }
    Ok(value)
}
fn create_directory(root: &File, path: &str) -> Result<File> {
    let (base, name) = split(root, path)?;
    mkdirat(&base, name, Mode::from_raw_mode(0o700))
        .map_err(|e| io_error("create package directory", e))?;
    let file = open_beneath(&base, name, true)?;
    fchmod(&file, Mode::from_raw_mode(0o700))
        .map_err(|e| io_error("set package directory mode", e))?;
    validate_directory(&file, None, "created package directory")?;
    Ok(file)
}
fn split<'a>(root: &File, path: &'a str) -> Result<(File, &'a str)> {
    let (parent, name) = path.rsplit_once('/').unwrap_or((".", path));
    Ok((open_beneath(root, parent, true)?, name))
}
fn output(root: &File, path: &str) -> Result<File> {
    let (parent, name) = split(root, path)?;
    Ok(File::from(
        openat(
            &parent,
            name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )
        .map_err(|e| io_error("create new package file", e))?,
    ))
}
fn finish(file: File, mode: u32) -> Result<(File, Snapshot)> {
    fchmod(&file, Mode::from_raw_mode(mode)).map_err(|e| io_error("set package file mode", e))?;
    sync(&file)?;
    let initial = inspect(&file)?;
    Ok((file, initial))
}
fn copy(source: &File, root: &File, entry: Entry<'_>) -> Result<(File, Snapshot)> {
    let file = output(root, entry.path)?;
    let mut bytes = [0; CHUNK];
    let mut digest = Sha256::new();
    let mut offset = 0;
    while offset < entry.length {
        let n = (entry.length - offset).min(CHUNK as u64) as usize;
        read_at(source, &mut bytes[..n], offset)?;
        write_at(&file, &bytes[..n], offset)?;
        digest.update(&bytes[..n]);
        offset += n as u64;
    }
    if <[u8; 32]>::from(digest.finalize()) != entry.sha256 {
        return Err(invalid(
            Kind::ContentMismatch,
            "measured source differs from expected digest",
        )
        .into());
    }
    finish(file, entry.role.protected_mode())
}
fn write_record(root: &File, name: &str, bytes: &[u8]) -> Result<(File, Snapshot)> {
    let file = output(root, name)?;
    write_at(&file, bytes, 0)?;
    finish(file, 0o444)
}
fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> Result<()> {
    let n = bytes.len();
    if rustix::io::pread(file, bytes, offset).map_err(|e| io_error("read package bytes", e))? != n {
        return Err(changed("short package read").into());
    }
    Ok(())
}
fn write_at(file: &File, bytes: &[u8], offset: u64) -> Result<()> {
    if rustix::io::pwrite(file, bytes, offset).map_err(|e| io_error("write package bytes", e))?
        != bytes.len()
    {
        return Err(changed("short package write").into());
    }
    Ok(())
}
fn sync(file: &File) -> Result<()> {
    fsync(file).map_err(|e| io_error("sync package object", e).into())
}

#[cfg(test)]
#[path = "compiler_runtime_package_tests.rs"]
mod tests;

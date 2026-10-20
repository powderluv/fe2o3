//! Optional inert bytes from the live V90 observation, never a compiler input.

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1 as Budget;
use rustix::fs::{Mode, OFlags, openat};
use sha2::{Digest, Sha256};

pub(crate) const DIRECTORY_ENV: &str = "FE2O3_TUTORIAL_GRAPH_CAPTURE_V92";
const MAX_GRAPH: usize = 64 * 1024 * 1024;
const MAX_DIRECTORY: u64 = 256 * 1024 * 1024;
const MAX_FILES: usize = 256;
const SUFFIX: &str = ".kir-v18";
// Logical bounded IO scratch, including the fixed comparison buffer and names.
const SCRATCH: usize = 64 * 1024;

struct CaptureError(String);
impl From<fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1> for CaptureError {
    fn from(value: fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1) -> Self {
        Self(value.to_string())
    }
}

fn error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn identity(bytes: &[u8]) -> [u8; 32] {
    let domain = fe2o3_kernel_ir::VERIFIED_CANONICAL_KERNEL_IR_V18_IDENTITY_DOMAIN_V1;
    let mut hash = Sha256::new();
    hash.update((domain.len() as u32).to_le_bytes());
    hash.update(domain);
    hash.update(fe2o3_kernel_ir::VERIFIED_CANONICAL_KERNEL_IR_V18_IDENTITY_POLICY_V1.to_le_bytes());
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    hash.finalize().into()
}

// CLOEXEC closes only at exec; fork/dup aliases can otherwise outlive this call.
// Release the acquired OFD lock at the lexical boundary, before closing root.
struct CaptureDirectoryLock<'a> {
    root: &'a File,
    held: bool,
}

impl<'a> CaptureDirectoryLock<'a> {
    fn acquire(root: &'a File) -> Result<Self, String> {
        rustix::fs::flock(root, rustix::fs::FlockOperation::NonBlockingLockExclusive)
            .map_err(error)?;
        Ok(Self { root, held: true })
    }

    fn finish(mut self) -> Result<(), String> {
        rustix::fs::flock(self.root, rustix::fs::FlockOperation::Unlock).map_err(error)?;
        self.held = false;
        Ok(())
    }
}

impl Drop for CaptureDirectoryLock<'_> {
    fn drop(&mut self) {
        if self.held {
            let _ = rustix::fs::flock(self.root, rustix::fs::FlockOperation::Unlock);
        }
    }
}

fn directory_identity(value: &std::fs::Metadata) -> (u64, u64, u32, u32, u32) {
    (
        value.dev(),
        value.ino(),
        value.uid(),
        value.gid(),
        value.mode(),
    )
}

fn file_identity(
    value: &std::fs::Metadata,
) -> (u64, u64, u32, u32, u32, u64, u64, i64, i64, i64, i64) {
    (
        value.dev(),
        value.ino(),
        value.uid(),
        value.gid(),
        value.mode(),
        value.nlink(),
        value.len(),
        value.mtime(),
        value.mtime_nsec(),
        value.ctime(),
        value.ctime_nsec(),
    )
}

fn require_file(metadata: &std::fs::Metadata) -> Result<(), String> {
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.mode() & 0o7777 != 0o600
        || metadata.uid() != rustix::process::getuid().as_raw()
    {
        return Err("noncanonical captured graph file".to_owned());
    }
    Ok(())
}

pub(crate) fn capture_if_requested(
    graph: &[u8],
    expected: ([u8; 32], u64),
    budget: &mut Budget<'_>,
) -> Result<(), String> {
    let Some(directory) = std::env::var_os(DIRECTORY_ENV) else {
        return Ok(());
    };
    capture(Path::new(&directory), graph, expected, budget)
}

fn capture(
    directory: &Path,
    graph: &[u8],
    expected: ([u8; 32], u64),
    budget: &mut Budget<'_>,
) -> Result<(), String> {
    let floor = budget.storage();
    budget
        .with_prepaid_scope(floor, 0, 0, SCRATCH, |budget| {
            capture_scoped(directory, graph, expected, budget).map_err(CaptureError)
        })
        .map_err(|error: CaptureError| error.0)
}

fn capture_scoped(
    directory: &Path,
    graph: &[u8],
    expected: ([u8; 32], u64),
    budget: &mut Budget<'_>,
) -> Result<(), String> {
    if !directory.is_absolute()
        || graph.is_empty()
        || graph.len() > MAX_GRAPH
        || graph.len() as u64 != expected.1
    {
        return Err("V92 graph capture bound or path".to_owned());
    }
    budget
        .charge_work(
            graph
                .len()
                .checked_add(128)
                .ok_or("capture work overflow")?,
        )
        .map_err(error)?;
    if identity(graph) != expected.0 {
        return Err("V92 captured graph differs from live typed receipt".to_owned());
    }
    // The caller supplies an existing private directory. Never create parents,
    // follow a final symlink, or overwrite an existing observation.
    let root = File::from(
        openat(
            rustix::fs::CWD,
            directory,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(error)?,
    );
    let initial = root.metadata().map_err(error)?;
    if !initial.is_dir()
        || initial.mode() & 0o7777 != 0o700
        || initial.uid() != rustix::process::getuid().as_raw()
        || std::fs::canonicalize(directory).map_err(error)? != directory
    {
        return Err("V92 capture directory is not private and canonical".to_owned());
    }
    let lock = CaptureDirectoryLock::acquire(&root)?;
    let name = expected
        .0
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        + SUFFIX;
    let mut count = 0usize;
    let mut total = 0u64;
    let mut exists = false;
    for entry in std::fs::read_dir(format!("/proc/self/fd/{}", root.as_raw_fd())).map_err(error)? {
        budget.charge_work(64).map_err(error)?;
        count = count.checked_add(1).ok_or("capture file count overflow")?;
        if count > MAX_FILES {
            return Err("V92 capture file count bound".to_owned());
        }
        let entry = entry.map_err(error)?;
        let entry_name = entry.file_name();
        let text = entry_name.to_str().ok_or("non-UTF8 capture name")?;
        let stem = text
            .strip_suffix(SUFFIX)
            .ok_or("unexpected capture directory entry")?;
        if stem.len() != 64
            || !stem
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("unexpected captured graph name".to_owned());
        }
        let file = File::from(
            openat(
                &root,
                text,
                OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(error)?,
        );
        let metadata = file.metadata().map_err(error)?;
        require_file(&metadata)?;
        if metadata.len() == 0 || metadata.len() > MAX_GRAPH as u64 {
            return Err("captured graph size bound".to_owned());
        }
        total = total
            .checked_add(metadata.len())
            .ok_or("capture byte count overflow")?;
        if total > MAX_DIRECTORY {
            return Err("V92 capture directory byte bound".to_owned());
        }
        exists |= text == name;
    }
    if !exists
        && (count == MAX_FILES
            || total
                .checked_add(graph.len() as u64)
                .is_none_or(|n| n > MAX_DIRECTORY))
    {
        return Err("V92 capture directory admission bound".to_owned());
    }
    let flags = if exists {
        OFlags::RDONLY
    } else {
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL
    };
    let mut file = File::from(
        openat(
            &root,
            name.as_str(),
            flags | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_bits_truncate(0o600),
        )
        .map_err(error)?,
    );
    if exists {
        let before = file.metadata().map_err(error)?;
        require_file(&before)?;
        if before.len() != graph.len() as u64 {
            return Err("captured graph length changed".to_owned());
        }
        let mut buffer = [0u8; 16 * 1024];
        budget.charge_work(graph.len()).map_err(error)?;
        for chunk in graph.chunks(buffer.len()) {
            file.read_exact(&mut buffer[..chunk.len()]).map_err(error)?;
            if &buffer[..chunk.len()] != chunk {
                return Err("captured graph bytes changed".to_owned());
            }
        }
        if file.read(&mut buffer[..1]).map_err(error)? != 0
            || file_identity(&before) != file_identity(&file.metadata().map_err(error)?)
        {
            return Err("captured graph changed during bounded read".to_owned());
        }
    } else {
        file.write_all(graph).map_err(error)?;
        file.sync_all().map_err(error)?;
        let metadata = file.metadata().map_err(error)?;
        require_file(&metadata)?;
        if metadata.len() != graph.len() as u64 {
            return Err("captured graph write length".to_owned());
        }
    }
    let named = File::from(
        openat(
            &root,
            name.as_str(),
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(error)?,
    );
    if file_identity(&file.metadata().map_err(error)?)
        != file_identity(&named.metadata().map_err(error)?)
        || directory_identity(&initial) != directory_identity(&root.metadata().map_err(error)?)
        || directory_identity(&initial)
            != directory_identity(&std::fs::symlink_metadata(directory).map_err(error)?)
    {
        return Err("V92 capture path custody changed".to_owned());
    }
    lock.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temporary(std::path::PathBuf);
    impl Temporary {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "fe2o3-graph-v92-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&path)
                .unwrap();
            Self(std::fs::canonicalize(path).unwrap())
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn call(path: &Path, bytes: &[u8], expected: ([u8; 32], u64)) -> Result<(), String> {
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut budget = Budget::new(&mut work, SCRATCH + 37);
        budget.reserve_storage(37).unwrap();
        let result = capture(path, bytes, expected, &mut budget);
        assert_eq!(budget.storage(), 37);
        result
    }
    #[test]
    fn captured_graph_exact_content_replay_and_substitution_refusal() {
        let tmp = Temporary::new();
        let bytes = b"inert test bytes, not an admitted graph";
        let expected = (identity(bytes), bytes.len() as u64);
        call(&tmp.0, bytes, expected).unwrap();
        call(&tmp.0, bytes, expected).unwrap();
        let path = std::fs::read_dir(&tmp.0)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        std::fs::write(&path, vec![0; bytes.len()]).unwrap();
        assert!(
            call(&tmp.0, bytes, expected)
                .unwrap_err()
                .contains("bytes changed")
        );
        assert!(
            call(&tmp.0, bytes, ([0; 32], expected.1))
                .unwrap_err()
                .contains("typed receipt")
        );
    }
    #[test]
    fn capture_rejects_nonprivate_directory_and_foreign_entries() {
        let tmp = Temporary::new();
        let bytes = b"bounded";
        let expected = (identity(bytes), bytes.len() as u64);
        std::fs::set_permissions(&tmp.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(call(&tmp.0, bytes, expected).is_err());
        std::fs::set_permissions(&tmp.0, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(tmp.0.join("unexpected"), bytes).unwrap();
        assert!(call(&tmp.0, bytes, expected).unwrap_err().contains("entry"));
    }
    #[test]
    fn capture_rejects_symlink_and_restores_short_storage_floor() {
        let tmp = Temporary::new();
        let link = tmp.0.join("link");
        std::os::unix::fs::symlink(&tmp.0, &link).unwrap();
        let bytes = b"bounded";
        assert!(call(&link, bytes, (identity(bytes), bytes.len() as u64)).is_err());
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut budget = Budget::new(&mut work, SCRATCH - 1);
        assert!(
            capture(
                &tmp.0,
                bytes,
                (identity(bytes), bytes.len() as u64),
                &mut budget
            )
            .is_err()
        );
        assert_eq!(budget.storage(), 0);
        assert_eq!(budget.work(), 0);
    }
    #[test]
    fn capture_refuses_fifo_and_contended_directory_without_waiting() {
        let tmp = Temporary::new();
        let bytes = b"bounded";
        let expected = (identity(bytes), bytes.len() as u64);
        let name = expected
            .0
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            + SUFFIX;
        rustix::fs::mkfifoat(rustix::fs::CWD, tmp.0.join(&name), Mode::RUSR | Mode::WUSR).unwrap();
        assert!(
            call(&tmp.0, bytes, expected)
                .unwrap_err()
                .contains("noncanonical")
        );
        std::fs::remove_file(tmp.0.join(name)).unwrap();
        let held = File::open(&tmp.0).unwrap();
        rustix::fs::flock(&held, rustix::fs::FlockOperation::NonBlockingLockExclusive).unwrap();
        assert!(call(&tmp.0, bytes, expected).is_err());
        assert_eq!(std::fs::read_dir(&tmp.0).unwrap().count(), 0);
        drop(held);
        call(&tmp.0, bytes, expected).unwrap();
    }

    #[test]
    fn capture_lock_finish_releases_retained_ofd_alias() {
        let tmp = Temporary::new();
        let root = File::open(&tmp.0).unwrap();
        let lock = CaptureDirectoryLock::acquire(&root).unwrap();
        let alias = root.try_clone().unwrap();
        lock.finish().unwrap();
        let bytes = b"bounded";
        call(&tmp.0, bytes, (identity(bytes), bytes.len() as u64)).unwrap();
        assert!(alias.metadata().unwrap().is_dir());
    }

    #[test]
    fn failed_capture_lock_acquisition_preserves_independent_holder() {
        let tmp = Temporary::new();
        let root = File::open(&tmp.0).unwrap();
        let lock = CaptureDirectoryLock::acquire(&root).unwrap();
        let contender = File::open(&tmp.0).unwrap();
        let Err(refusal) = CaptureDirectoryLock::acquire(&contender) else {
            panic!("independent capture lock must be refused");
        };
        assert_eq!(refusal, error(rustix::io::Errno::WOULDBLOCK));
        assert_eq!(
            rustix::fs::flock(
                &contender,
                rustix::fs::FlockOperation::NonBlockingLockExclusive
            ),
            Err(rustix::io::Errno::WOULDBLOCK)
        );
        let bytes = b"bounded";
        assert_eq!(
            call(&tmp.0, bytes, (identity(bytes), bytes.len() as u64)),
            Err(error(rustix::io::Errno::WOULDBLOCK))
        );
        assert_eq!(std::fs::read_dir(&tmp.0).unwrap().count(), 0);
        lock.finish().unwrap();
        call(&tmp.0, bytes, (identity(bytes), bytes.len() as u64)).unwrap();
    }

    #[test]
    fn capture_lock_error_and_unwind_release_retained_ofd_alias() {
        let tmp = Temporary::new();
        let root = File::open(&tmp.0).unwrap();
        let alias = root.try_clone().unwrap();
        let result = (|| {
            let _lock = CaptureDirectoryLock::acquire(&root).unwrap();
            Err::<(), _>("injected capture error")
        })();
        assert_eq!(result, Err("injected capture error"));
        let bytes = b"bounded";
        let expected = (identity(bytes), bytes.len() as u64);
        call(&tmp.0, bytes, expected).unwrap();
        let unwind = std::panic::catch_unwind(|| {
            let _lock = CaptureDirectoryLock::acquire(&root).unwrap();
            panic!("injected capture unwind");
        });
        assert!(unwind.is_err());
        call(&tmp.0, bytes, expected).unwrap();
        assert!(alias.metadata().unwrap().is_dir());
    }
}

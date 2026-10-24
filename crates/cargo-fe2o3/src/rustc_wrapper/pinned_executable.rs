//! Fail-closed pinning for the native `rustc` executable.
//!
//! Linux is the only supported platform in this increment. The implementation opens the final
//! path component with `O_NOFOLLOW`, except for an exact, fully sealed `/proc/self/fd/N` input,
//! hashes through that opened descriptor, and executes through a validated `/proc/self/fd`
//! reference while retaining the descriptor. Other platforms return
//! [`PinExecutableError::UnsupportedPlatform`]; they must grow an equivalent descriptor-based
//! execution primitive rather than falling back to reopening the input pathname.
//!
//! This is an inert object/byte consistency primitive, not a trust-chain or execution-history
//! authenticator. Parent-directory symlinks, mutations of the opened inode by another writer, the
//! ELF interpreter, shared libraries, the codegen backend dynamic library, and the kernel/`procfs`
//! implementation remain outside this boundary. A protected supervisor is required to grant
//! authority over pre-backend loader history.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use fe2o3_runtime_protocol::WorkerV3ApplicationHandoffProtocolErrorV1;

/// A deliberately bounded read prevents a selected tool path from causing unbounded hashing work.
pub(crate) const MAX_EXECUTABLE_BYTES: u64 = 512 * 1024 * 1024;

/// Ordinary binding-only Cargo test images retain default debug information and can exceed
/// the tool/authority limit (the MoE oracle measured 909,946,760 bytes). This separate finite
/// allowance grants no authority and is selected only by the binding-only host-test runner.
pub(crate) const MAX_BINDING_HOST_TEST_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
pub(crate) enum PinExecutableError {
    #[allow(dead_code)] // Constructed by the non-Linux fail-closed implementation.
    UnsupportedPlatform,
    Open {
        path: PathBuf,
        source: io::Error,
    },
    Inspect {
        path: PathBuf,
        source: io::Error,
    },
    NotRegular {
        path: PathBuf,
    },
    NotExecutable {
        path: PathBuf,
        mode: u32,
    },
    Empty {
        path: PathBuf,
    },
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    Read {
        path: PathBuf,
        source: io::Error,
    },
    UnexpectedEof {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    GrewDuringRead {
        path: PathBuf,
        expected: u64,
    },
    ChangedDuringRead {
        path: PathBuf,
    },
    Rewind {
        path: PathBuf,
        source: io::Error,
    },
    ExecutionStrategy {
        path: PathBuf,
        source: io::Error,
    },
    ExecutionObjectChanged {
        path: PathBuf,
    },
    SnapshotDigestMismatch {
        path: PathBuf,
    },
    SnapshotSealsChanged {
        path: PathBuf,
    },
    InvalidV3ApplicationIdentity {
        path: PathBuf,
        source: WorkerV3ApplicationHandoffProtocolErrorV1,
    },
}

impl fmt::Display for PinExecutableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => formatter
                .write_str("pinned executable preparation is supported only on Linux with procfs"),
            Self::Open { path, source } => {
                write!(
                    formatter,
                    "failed to open executable {}: {source}",
                    path.display()
                )
            }
            Self::Inspect { path, source } => write!(
                formatter,
                "failed to inspect opened executable {}: {source}",
                path.display()
            ),
            Self::NotRegular { path } => {
                write!(
                    formatter,
                    "executable is not a regular file: {}",
                    path.display()
                )
            }
            Self::NotExecutable { path, mode } => write!(
                formatter,
                "executable has no execute permission bits: {} (mode {mode:#o})",
                path.display()
            ),
            Self::Empty { path } => {
                write!(formatter, "executable is empty: {}", path.display())
            }
            Self::TooLarge { path, size, limit } => write!(
                formatter,
                "executable {} is {size} bytes, exceeding the {limit}-byte hashing limit",
                path.display()
            ),
            Self::Read { path, source } => write!(
                formatter,
                "failed while hashing executable {}: {source}",
                path.display()
            ),
            Self::UnexpectedEof {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "executable {} became shorter while hashing: expected {expected} bytes, read {actual}",
                path.display()
            ),
            Self::GrewDuringRead { path, expected } => write!(
                formatter,
                "executable {} grew beyond its initial {expected}-byte size while hashing",
                path.display()
            ),
            Self::ChangedDuringRead { path } => write!(
                formatter,
                "executable metadata changed while hashing: {}",
                path.display()
            ),
            Self::Rewind { path, source } => write!(
                formatter,
                "failed to rewind opened executable {} after hashing: {source}",
                path.display()
            ),
            Self::ExecutionStrategy { path, source } => write!(
                formatter,
                "fd-backed execution is unavailable for {}: {source}",
                path.display()
            ),
            Self::ExecutionObjectChanged { path } => write!(
                formatter,
                "pinned executable changed after hashing: {}",
                path.display()
            ),
            Self::SnapshotDigestMismatch { path } => write!(
                formatter,
                "sealed executable snapshot does not match the pinned bytes from {}",
                path.display()
            ),
            Self::SnapshotSealsChanged { path } => write!(
                formatter,
                "sealed executable snapshot has missing or unexpected seals for {}",
                path.display()
            ),
            Self::InvalidV3ApplicationIdentity { path, source } => write!(
                formatter,
                "application {} has no valid Worker V3 application identity: {source}",
                path.display()
            ),
        }
    }
}

impl Error for PinExecutableError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Open { source, .. }
            | Self::Inspect { source, .. }
            | Self::Read { source, .. }
            | Self::Rewind { source, .. }
            | Self::ExecutionStrategy { source, .. } => Some(source),
            Self::InvalidV3ApplicationIdentity { source, .. } => Some(source),
            Self::UnsupportedPlatform
            | Self::NotRegular { .. }
            | Self::NotExecutable { .. }
            | Self::Empty { .. }
            | Self::TooLarge { .. }
            | Self::UnexpectedEof { .. }
            | Self::GrewDuringRead { .. }
            | Self::ChangedDuringRead { .. }
            | Self::ExecutionObjectChanged { .. }
            | Self::SnapshotDigestMismatch { .. }
            | Self::SnapshotSealsChanged { .. } => None,
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{
        MAX_BINDING_HOST_TEST_BYTES, MAX_EXECUTABLE_BYTES, Path, PathBuf, PinExecutableError,
    };
    use fe2o3_process_identity::LinuxObjectIdentityV3;
    use rustix::fs::{Access, MemfdFlags, Mode, OFlags, SealFlags};
    use sha2::{Digest, Sha256};
    use std::ffi::OsStr;
    use std::fs::{File, Metadata};
    use std::io::{self, Read, Seek, SeekFrom, Write};
    use std::os::fd::{AsRawFd, BorrowedFd, RawFd};
    use std::os::unix::fs::FileExt;
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command, ExitStatus, Output};

    const HASH_CHUNK_BYTES: usize = 64 * 1024;
    const REQUIRED_INHERITED_SEALS: SealFlags = SealFlags::WRITE
        .union(SealFlags::GROW)
        .union(SealFlags::SHRINK)
        .union(SealFlags::SEAL);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ObjectSnapshot {
        device: u64,
        inode: u64,
        mode: u32,
        size: u64,
        modified_seconds: i64,
        modified_nanoseconds: i64,
        changed_seconds: i64,
        changed_nanoseconds: i64,
    }

    impl ObjectSnapshot {
        fn from_metadata(metadata: &Metadata) -> Self {
            Self {
                device: metadata.dev(),
                inode: metadata.ino(),
                mode: metadata.mode(),
                size: metadata.len(),
                modified_seconds: metadata.mtime(),
                modified_nanoseconds: metadata.mtime_nsec(),
                changed_seconds: metadata.ctime(),
                changed_nanoseconds: metadata.ctime_nsec(),
            }
        }

        fn same_execution_object(self, current: Self) -> bool {
            self.device == current.device
                && self.inode == current.inode
                && self.mode == current.mode
                && self.size == current.size
                && self.modified_seconds == current.modified_seconds
                && self.modified_nanoseconds == current.modified_nanoseconds
                && self.changed_seconds == current.changed_seconds
                && self.changed_nanoseconds == current.changed_nanoseconds
        }

        fn matches_stat(self, stat: &rustix::fs::Stat) -> bool {
            self.device == stat.st_dev
                && self.inode == stat.st_ino
                && self.mode == stat.st_mode
                && u64::try_from(stat.st_size).ok() == Some(self.size)
                && stat.st_mtime == self.modified_seconds
                && i64::try_from(stat.st_mtime_nsec).ok() == Some(self.modified_nanoseconds)
                && stat.st_ctime == self.changed_seconds
                && i64::try_from(stat.st_ctime_nsec).ok() == Some(self.changed_nanoseconds)
        }
    }

    /// A native executable held open from validation through command execution.
    pub(crate) struct PinnedExecutable {
        file: File,
        display_path: PathBuf,
        execution_path: PathBuf,
        snapshot: ObjectSnapshot,
        sha256: [u8; 32],
    }

    /// An exact, immutable application image captured from a validated executable descriptor.
    pub(crate) struct SealedStaticApplication {
        file: File,
        display_path: PathBuf,
        execution_path: PathBuf,
        snapshot: ObjectSnapshot,
        seals: SealFlags,
        identity_v3: fe2o3_runtime_protocol::WorkerV3ApplicationIdentityV1,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum StaticSnapshotBarrier {
        SourceValidated,
        SourceCopied,
        SnapshotSealed,
        SnapshotBound,
    }

    impl PinnedExecutable {
        pub(crate) fn open(path: &Path) -> Result<Self, PinExecutableError> {
            Self::open_with_limit(path, MAX_EXECUTABLE_BYTES)
        }

        pub(crate) fn open_binding_host_test(path: &Path) -> Result<Self, PinExecutableError> {
            Self::open_with_limit(path, MAX_BINDING_HOST_TEST_BYTES)
        }

        fn open_with_limit(path: &Path, limit: u64) -> Result<Self, PinExecutableError> {
            let display_path = path.to_path_buf();
            let retained_descriptor = retained_descriptor_number(path);
            let mut open_flags = OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC;
            if retained_descriptor.is_none() {
                open_flags |= OFlags::NOFOLLOW;
            }
            let fd = rustix::fs::open(path, open_flags, Mode::empty()).map_err(|source| {
                PinExecutableError::Open {
                    path: display_path.clone(),
                    source: source.into(),
                }
            })?;
            let file = File::from(fd);
            if retained_descriptor.is_some()
                && rustix::fs::fcntl_get_seals(&file).map_err(|source| {
                    PinExecutableError::ExecutionStrategy {
                        path: display_path.clone(),
                        source: source.into(),
                    }
                })? != REQUIRED_INHERITED_SEALS
            {
                return Err(PinExecutableError::ExecutionStrategy {
                    path: display_path,
                    source: io::Error::new(
                        io::ErrorKind::InvalidData,
                        "inherited executable descriptor is not fully sealed",
                    ),
                });
            }
            Self::from_open_file(file, display_path, limit)
        }

        pub(crate) fn from_transferred_file(
            file: File,
            display_path: PathBuf,
        ) -> Result<Self, PinExecutableError> {
            let flags = rustix::io::fcntl_getfd(&file).map_err(|source| {
                PinExecutableError::ExecutionStrategy {
                    path: display_path.clone(),
                    source: source.into(),
                }
            })?;
            if !flags.contains(rustix::io::FdFlags::CLOEXEC) {
                return Err(PinExecutableError::ExecutionStrategy {
                    path: display_path,
                    source: io::Error::new(
                        io::ErrorKind::InvalidData,
                        "transferred executable observation is not close-on-exec",
                    ),
                });
            }
            Self::from_open_file(file, display_path, MAX_EXECUTABLE_BYTES)
        }

        pub(crate) fn try_clone_for_transfer(&self) -> Result<File, PinExecutableError> {
            self.file
                .try_clone()
                .map_err(|source| PinExecutableError::Inspect {
                    path: self.display_path.clone(),
                    source,
                })
        }

        fn from_open_file(
            mut file: File,
            display_path: PathBuf,
            limit: u64,
        ) -> Result<Self, PinExecutableError> {
            let initial_metadata =
                file.metadata()
                    .map_err(|source| PinExecutableError::Inspect {
                        path: display_path.clone(),
                        source,
                    })?;
            if !initial_metadata.is_file() {
                return Err(PinExecutableError::NotRegular { path: display_path });
            }

            let initial = ObjectSnapshot::from_metadata(&initial_metadata);
            if initial.mode & 0o111 == 0 {
                return Err(PinExecutableError::NotExecutable {
                    path: display_path,
                    mode: initial.mode,
                });
            }
            if initial.size == 0 {
                return Err(PinExecutableError::Empty { path: display_path });
            }
            if initial.size > limit {
                return Err(PinExecutableError::TooLarge {
                    path: display_path,
                    size: initial.size,
                    limit,
                });
            }

            let sha256 = hash_exact(&mut file, &display_path, initial.size)?;
            let final_metadata = file
                .metadata()
                .map_err(|source| PinExecutableError::Inspect {
                    path: display_path.clone(),
                    source,
                })?;
            let snapshot = ObjectSnapshot::from_metadata(&final_metadata);
            if snapshot != initial {
                return Err(PinExecutableError::ChangedDuringRead { path: display_path });
            }
            file.seek(SeekFrom::Start(0))
                .map_err(|source| PinExecutableError::Rewind {
                    path: display_path.clone(),
                    source,
                })?;

            let execution_path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
            validate_execution_path(&file, &execution_path, snapshot, &display_path)?;

            Ok(Self {
                file,
                display_path,
                execution_path,
                snapshot,
                sha256,
            })
        }

        pub(crate) const fn sha256(&self) -> &[u8; 32] {
            &self.sha256
        }

        pub(crate) fn require_sealed_executable_image(&self) -> Result<(), PinExecutableError> {
            require_exact_seals(&self.file, REQUIRED_INHERITED_SEALS, &self.display_path)
        }

        pub(crate) fn fixed_child_path(
            &self,
            target_fd: RawFd,
        ) -> Result<PathBuf, PinExecutableError> {
            self.require_sealed_executable_image()?;
            require_unused_child_descriptor(target_fd, &self.display_path)?;
            Ok(PathBuf::from(format!("/proc/self/fd/{target_fd}")))
        }

        pub(crate) fn inherit_for_child_at(
            &self,
            command: &mut Command,
            target_fd: RawFd,
        ) -> Result<(), PinExecutableError> {
            self.require_sealed_executable_image()?;
            require_unused_child_descriptor(target_fd, &self.display_path)?;
            let installed =
                rustix::io::fcntl_dupfd_cloexec(&self.file, target_fd).map_err(|source| {
                    PinExecutableError::ExecutionStrategy {
                        path: self.display_path.clone(),
                        source: source.into(),
                    }
                })?;
            if installed.as_raw_fd() != target_fd {
                return Err(PinExecutableError::ExecutionStrategy {
                    path: self.display_path.clone(),
                    source: io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        format!("fixed executable descriptor {target_fd} raced with another user"),
                    ),
                });
            }
            let snapshot = self.snapshot;
            let stat = rustix::fs::fstat(&installed).map_err(|source| {
                PinExecutableError::ExecutionStrategy {
                    path: self.display_path.clone(),
                    source: source.into(),
                }
            })?;
            if !snapshot.matches_stat(&stat) {
                return Err(PinExecutableError::ExecutionObjectChanged {
                    path: self.display_path.clone(),
                });
            }
            // SAFETY: `installed` reserves the exact target in the parent through spawn. The
            // callback only revalidates that descriptor and clears CLOEXEC in the child.
            unsafe {
                command.pre_exec(move || {
                    let stat = rustix::fs::fstat(&installed).map_err(io::Error::from)?;
                    let seals = rustix::fs::fcntl_get_seals(&installed).map_err(io::Error::from)?;
                    let status = rustix::fs::fcntl_getfl(&installed).map_err(io::Error::from)?;
                    if !snapshot.matches_stat(&stat)
                        || seals != REQUIRED_INHERITED_SEALS
                        || status & OFlags::ACCMODE != OFlags::RDONLY
                    {
                        return Err(io::Error::from_raw_os_error(
                            rustix::io::Errno::STALE.raw_os_error(),
                        ));
                    }
                    rustix::io::fcntl_setfd(&installed, rustix::io::FdFlags::empty())
                        .map_err(io::Error::from)?;
                    // `Command` retains this callback through the immediately following exec.
                    // The child does not unwind or drop it between this return and that exec.
                    Ok(())
                });
            }
            Ok(())
        }

        pub(crate) fn seal_executable_image(&self) -> Result<Self, PinExecutableError> {
            let initial = self
                .file
                .metadata()
                .map_err(|source| PinExecutableError::Inspect {
                    path: self.display_path.clone(),
                    source,
                })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&initial))
            {
                return Err(PinExecutableError::ExecutionObjectChanged {
                    path: self.display_path.clone(),
                });
            }
            let image_fd = rustix::fs::memfd_create(
                "fe2o3-sealed-executable",
                MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
            )
            .map_err(|source| PinExecutableError::ExecutionStrategy {
                path: self.display_path.clone(),
                source: source.into(),
            })?;
            let mut image = File::from(image_fd);
            let mut source = self
                .file
                .try_clone()
                .map_err(|source| PinExecutableError::Read {
                    path: self.display_path.clone(),
                    source,
                })?;
            source
                .seek(SeekFrom::Start(0))
                .map_err(|source| PinExecutableError::Rewind {
                    path: self.display_path.clone(),
                    source,
                })?;
            let captured = copy_exact(
                &mut source,
                &mut image,
                &self.display_path,
                self.snapshot.size,
            )?;
            if captured != self.sha256 {
                return Err(PinExecutableError::SnapshotDigestMismatch {
                    path: self.display_path.clone(),
                });
            }
            let final_source =
                self.file
                    .metadata()
                    .map_err(|source| PinExecutableError::Inspect {
                        path: self.display_path.clone(),
                        source,
                    })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&final_source))
            {
                return Err(PinExecutableError::ChangedDuringRead {
                    path: self.display_path.clone(),
                });
            }
            rustix::fs::fchmod(&image, Mode::RUSR | Mode::XUSR).map_err(|source| {
                PinExecutableError::ExecutionStrategy {
                    path: self.display_path.clone(),
                    source: source.into(),
                }
            })?;
            seal_immutable_image(&image, &self.display_path)?;
            let seals = REQUIRED_INHERITED_SEALS;
            let writable_path = PathBuf::from(format!("/proc/self/fd/{}", image.as_raw_fd()));
            let read_only_fd = rustix::fs::open(
                &writable_path,
                OFlags::RDONLY | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|source| PinExecutableError::ExecutionStrategy {
                path: self.display_path.clone(),
                source: source.into(),
            })?;
            let file = File::from(read_only_fd);
            require_exact_seals(&file, seals, &self.display_path)?;
            drop(image);
            let sealed = Self::from_transferred_file(file, self.display_path.clone())?;
            if sealed.sha256 != self.sha256 {
                return Err(PinExecutableError::SnapshotDigestMismatch {
                    path: self.display_path.clone(),
                });
            }
            Ok(sealed)
        }

        pub(crate) fn authenticated_bytes(&self) -> Result<Vec<u8>, PinExecutableError> {
            let initial = self
                .file
                .metadata()
                .map_err(|source| PinExecutableError::Inspect {
                    path: self.display_path.clone(),
                    source,
                })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&initial))
            {
                return Err(PinExecutableError::ExecutionObjectChanged {
                    path: self.display_path.clone(),
                });
            }
            let size = usize::try_from(self.snapshot.size).expect("executable size is bounded");
            let mut bytes = vec![0_u8; size];
            let mut offset = 0;
            while offset != bytes.len() {
                let read = self
                    .file
                    .read_at(&mut bytes[offset..], offset as u64)
                    .map_err(|source| PinExecutableError::Read {
                        path: self.display_path.clone(),
                        source,
                    })?;
                if read == 0 {
                    return Err(PinExecutableError::UnexpectedEof {
                        path: self.display_path.clone(),
                        expected: self.snapshot.size,
                        actual: offset as u64,
                    });
                }
                offset += read;
            }
            let final_metadata =
                self.file
                    .metadata()
                    .map_err(|source| PinExecutableError::Inspect {
                        path: self.display_path.clone(),
                        source,
                    })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&final_metadata))
            {
                return Err(PinExecutableError::ChangedDuringRead {
                    path: self.display_path.clone(),
                });
            }
            if <[u8; 32]>::from(Sha256::digest(&bytes)) != self.sha256 {
                return Err(PinExecutableError::SnapshotDigestMismatch {
                    path: self.display_path.clone(),
                });
            }
            Ok(bytes)
        }

        pub(crate) fn seal_static_application(
            &self,
        ) -> Result<SealedStaticApplication, PinExecutableError> {
            self.seal_static_application_with_barrier(|_| {})
        }

        fn seal_static_application_with_barrier(
            &self,
            mut barrier: impl FnMut(StaticSnapshotBarrier),
        ) -> Result<SealedStaticApplication, PinExecutableError> {
            let initial = self
                .file
                .metadata()
                .map_err(|source| PinExecutableError::Inspect {
                    path: self.display_path.clone(),
                    source,
                })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&initial))
            {
                return Err(PinExecutableError::ExecutionObjectChanged {
                    path: self.display_path.clone(),
                });
            }
            barrier(StaticSnapshotBarrier::SourceValidated);

            let image_fd = rustix::fs::memfd_create(
                "fe2o3-static-application",
                MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
            )
            .map_err(|source| PinExecutableError::ExecutionStrategy {
                path: self.display_path.clone(),
                source: source.into(),
            })?;
            let mut image = File::from(image_fd);
            let mut source = self
                .file
                .try_clone()
                .map_err(|source| PinExecutableError::Read {
                    path: self.display_path.clone(),
                    source,
                })?;
            source
                .seek(SeekFrom::Start(0))
                .map_err(|source| PinExecutableError::Rewind {
                    path: self.display_path.clone(),
                    source,
                })?;
            let captured = copy_exact(
                &mut source,
                &mut image,
                &self.display_path,
                self.snapshot.size,
            )?;
            barrier(StaticSnapshotBarrier::SourceCopied);
            if captured != self.sha256 {
                return Err(PinExecutableError::SnapshotDigestMismatch {
                    path: self.display_path.clone(),
                });
            }
            let final_source =
                self.file
                    .metadata()
                    .map_err(|source| PinExecutableError::Inspect {
                        path: self.display_path.clone(),
                        source,
                    })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&final_source))
            {
                return Err(PinExecutableError::ChangedDuringRead {
                    path: self.display_path.clone(),
                });
            }

            rustix::fs::fchmod(&image, Mode::RUSR | Mode::XUSR).map_err(|source| {
                PinExecutableError::ExecutionStrategy {
                    path: self.display_path.clone(),
                    source: source.into(),
                }
            })?;
            seal_immutable_image(&image, &self.display_path)?;
            let seals = REQUIRED_INHERITED_SEALS;
            barrier(StaticSnapshotBarrier::SnapshotSealed);

            image
                .seek(SeekFrom::Start(0))
                .map_err(|source| PinExecutableError::Rewind {
                    path: self.display_path.clone(),
                    source,
                })?;
            let size = usize::try_from(self.snapshot.size).expect("executable size is bounded");
            let mut bytes = Vec::with_capacity(size.saturating_add(1));
            Read::by_ref(&mut image)
                .take(self.snapshot.size + 1)
                .read_to_end(&mut bytes)
                .map_err(|source| PinExecutableError::Read {
                    path: self.display_path.clone(),
                    source,
                })?;
            if bytes.len() != size || <[u8; 32]>::from(Sha256::digest(&bytes)) != captured {
                return Err(PinExecutableError::SnapshotDigestMismatch {
                    path: self.display_path.clone(),
                });
            }
            let identity_v3 =
                fe2o3_runtime_protocol::WorkerV3ApplicationIdentityV1::from_sealed_static_elf_v1(
                    &bytes,
                )
                .map_err(|source| {
                    PinExecutableError::InvalidV3ApplicationIdentity {
                        path: self.display_path.clone(),
                        source,
                    }
                })?;

            let writable_path = PathBuf::from(format!("/proc/self/fd/{}", image.as_raw_fd()));
            let read_only_fd = rustix::fs::open(
                &writable_path,
                OFlags::RDONLY | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|source| PinExecutableError::ExecutionStrategy {
                path: self.display_path.clone(),
                source: source.into(),
            })?;
            let file = File::from(read_only_fd);
            let snapshot = ObjectSnapshot::from_metadata(&file.metadata().map_err(|source| {
                PinExecutableError::Inspect {
                    path: self.display_path.clone(),
                    source,
                }
            })?);
            require_exact_seals(&file, seals, &self.display_path)?;
            let execution_path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
            validate_execution_path(&file, &execution_path, snapshot, &self.display_path)?;
            barrier(StaticSnapshotBarrier::SnapshotBound);
            drop(image);

            Ok(SealedStaticApplication {
                file,
                display_path: self.display_path.clone(),
                execution_path,
                snapshot,
                seals,
                identity_v3,
            })
        }

        pub(crate) const fn size(&self) -> u64 {
            self.snapshot.size
        }

        pub(crate) const fn object_identity(&self) -> LinuxObjectIdentityV3 {
            LinuxObjectIdentityV3::from_linux_stat(
                self.snapshot.device,
                self.snapshot.inode,
                self.snapshot.mode,
            )
        }

        pub(crate) fn command(&self) -> Result<PinnedCommand<'_>, PinExecutableError> {
            let current = self
                .file
                .metadata()
                .map_err(|source| PinExecutableError::Inspect {
                    path: self.display_path.clone(),
                    source,
                })?;
            if !self
                .snapshot
                .same_execution_object(ObjectSnapshot::from_metadata(&current))
            {
                return Err(PinExecutableError::ExecutionObjectChanged {
                    path: self.display_path.clone(),
                });
            }
            validate_execution_path(
                &self.file,
                &self.execution_path,
                self.snapshot,
                &self.display_path,
            )?;

            let mut command = Command::new(&self.execution_path);
            command.arg0(&self.display_path);
            Ok(PinnedCommand {
                _executable: &self.file,
                command,
                configured_argv0: self.display_path.as_os_str(),
            })
        }

        #[cfg(test)]
        pub(crate) fn raw_fd(&self) -> RawFd {
            self.file.as_raw_fd()
        }

        #[cfg(test)]
        fn execution_path(&self) -> &Path {
            &self.execution_path
        }
    }

    fn retained_descriptor_number(path: &Path) -> Option<i32> {
        let text = path.to_str()?;
        let suffix = text.strip_prefix("/proc/self/fd/")?;
        if suffix.is_empty()
            || !suffix.bytes().all(|byte| byte.is_ascii_digit())
            || (suffix.len() > 1 && suffix.starts_with('0'))
        {
            return None;
        }
        let descriptor = suffix.parse::<i32>().ok()?;
        (descriptor >= 3 && text == format!("/proc/self/fd/{descriptor}")).then_some(descriptor)
    }

    impl SealedStaticApplication {
        pub(crate) const fn identity_v3(
            &self,
        ) -> fe2o3_runtime_protocol::WorkerV3ApplicationIdentityV1 {
            self.identity_v3
        }

        pub(crate) fn command(&self) -> Result<PinnedCommand<'_>, PinExecutableError> {
            require_exact_seals(&self.file, self.seals, &self.display_path)?;
            validate_execution_path(
                &self.file,
                &self.execution_path,
                self.snapshot,
                &self.display_path,
            )?;
            let status = rustix::fs::fcntl_getfl(&self.file).map_err(|source| {
                PinExecutableError::ExecutionStrategy {
                    path: self.display_path.clone(),
                    source: source.into(),
                }
            })?;
            if status & OFlags::ACCMODE != OFlags::RDONLY {
                return Err(PinExecutableError::ExecutionObjectChanged {
                    path: self.display_path.clone(),
                });
            }
            let mut command = Command::new(&self.execution_path);
            command.arg0(&self.display_path);
            Ok(PinnedCommand {
                _executable: &self.file,
                command,
                configured_argv0: self.display_path.as_os_str(),
            })
        }

        #[cfg(test)]
        fn bytes(&self) -> Vec<u8> {
            std::fs::read(&self.execution_path).unwrap()
        }
    }

    /// A command that cannot outlive the descriptor used by its executable pathname.
    pub(crate) struct PinnedCommand<'executable> {
        _executable: &'executable File,
        command: Command,
        configured_argv0: &'executable OsStr,
    }

    impl PinnedCommand<'_> {
        pub(crate) const fn as_command(&self) -> &Command {
            &self.command
        }

        pub(crate) fn as_command_mut(&mut self) -> &mut Command {
            &mut self.command
        }

        pub(crate) const fn configured_argv0(&self) -> &OsStr {
            self.configured_argv0
        }

        pub(crate) fn args<I, S>(&mut self, args: I) -> &mut Self
        where
            I: IntoIterator<Item = S>,
            S: AsRef<OsStr>,
        {
            self.command.args(args);
            self
        }

        pub(crate) fn status(&mut self) -> io::Result<ExitStatus> {
            crate::process_execution::status(&mut self.command)
        }

        pub(crate) fn spawn(&mut self) -> io::Result<Child> {
            crate::process_execution::spawn(&mut self.command)
        }

        pub(crate) fn spawn_with_compiler_proof(
            self,
            proof: fe2o3_verifier::PendingCompilerProofDelegationV1,
        ) -> io::Result<(Child, fe2o3_verifier::SpawnedCompilerProofDelegationV1)> {
            proof.spawn(self.command)
        }

        pub(crate) fn output(&mut self) -> io::Result<Output> {
            crate::process_execution::capture_output(&mut self.command)
        }
    }

    fn hash_exact<R: Read>(
        reader: &mut R,
        display_path: &Path,
        expected_size: u64,
    ) -> Result<[u8; 32], PinExecutableError> {
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; HASH_CHUNK_BYTES];
        let mut total = 0_u64;

        while total < expected_size {
            let remaining = expected_size - total;
            let requested = usize::try_from(remaining.min(HASH_CHUNK_BYTES as u64))
                .expect("hash chunk length fits usize");
            let read = read_retry(reader, &mut buffer[..requested]).map_err(|source| {
                PinExecutableError::Read {
                    path: display_path.to_path_buf(),
                    source,
                }
            })?;
            if read == 0 {
                return Err(PinExecutableError::UnexpectedEof {
                    path: display_path.to_path_buf(),
                    expected: expected_size,
                    actual: total,
                });
            }
            hasher.update(&buffer[..read]);
            total += read as u64;
        }

        if read_retry(reader, &mut buffer[..1]).map_err(|source| PinExecutableError::Read {
            path: display_path.to_path_buf(),
            source,
        })? != 0
        {
            return Err(PinExecutableError::GrewDuringRead {
                path: display_path.to_path_buf(),
                expected: expected_size,
            });
        }

        Ok(hasher.finalize().into())
    }

    fn copy_exact<R: Read, W: Write>(
        reader: &mut R,
        writer: &mut W,
        display_path: &Path,
        expected_size: u64,
    ) -> Result<[u8; 32], PinExecutableError> {
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; HASH_CHUNK_BYTES];
        let mut total = 0_u64;
        while total < expected_size {
            let remaining = expected_size - total;
            let requested = usize::try_from(remaining.min(HASH_CHUNK_BYTES as u64))
                .expect("copy chunk length fits usize");
            let read = read_retry(reader, &mut buffer[..requested]).map_err(|source| {
                PinExecutableError::Read {
                    path: display_path.to_path_buf(),
                    source,
                }
            })?;
            if read == 0 {
                return Err(PinExecutableError::UnexpectedEof {
                    path: display_path.to_path_buf(),
                    expected: expected_size,
                    actual: total,
                });
            }
            writer.write_all(&buffer[..read]).map_err(|source| {
                PinExecutableError::ExecutionStrategy {
                    path: display_path.to_path_buf(),
                    source,
                }
            })?;
            hasher.update(&buffer[..read]);
            total += read as u64;
        }
        if read_retry(reader, &mut buffer[..1]).map_err(|source| PinExecutableError::Read {
            path: display_path.to_path_buf(),
            source,
        })? != 0
        {
            return Err(PinExecutableError::GrewDuringRead {
                path: display_path.to_path_buf(),
                expected: expected_size,
            });
        }
        Ok(hasher.finalize().into())
    }

    fn require_exact_seals(
        file: &File,
        expected: SealFlags,
        display_path: &Path,
    ) -> Result<(), PinExecutableError> {
        let actual = rustix::fs::fcntl_get_seals(file).map_err(|source| {
            PinExecutableError::ExecutionStrategy {
                path: display_path.to_path_buf(),
                source: source.into(),
            }
        })?;
        if actual != expected {
            return Err(PinExecutableError::SnapshotSealsChanged {
                path: display_path.to_path_buf(),
            });
        }
        Ok(())
    }

    fn seal_immutable_image(file: &File, display_path: &Path) -> Result<(), PinExecutableError> {
        // A fresh descriptor is still reachable by same-UID `/proc` observers. Waiting is safe
        // here because both callers rehash the exact sealed image before granting authority.
        fe2o3_process_identity::seal_immutable_memfd_v1(
            file,
            fe2o3_process_identity::ImmutableMemfdBusyPolicyV1::BoundedExternalObserverQuiescence,
        )
        .map_err(|source| PinExecutableError::ExecutionStrategy {
            path: display_path.to_path_buf(),
            source: source.into(),
        })
    }

    fn require_unused_child_descriptor(
        target_fd: RawFd,
        display_path: &Path,
    ) -> Result<(), PinExecutableError> {
        if target_fd < 3 {
            return Err(PinExecutableError::ExecutionStrategy {
                path: display_path.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "fixed executable descriptor would replace a standard stream",
                ),
            });
        }
        match rustix::fs::fstat(unsafe { BorrowedFd::borrow_raw(target_fd) }) {
            Err(rustix::io::Errno::BADF) => Ok(()),
            Err(source) => Err(PinExecutableError::ExecutionStrategy {
                path: display_path.to_path_buf(),
                source: source.into(),
            }),
            Ok(_) => Err(PinExecutableError::ExecutionStrategy {
                path: display_path.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("fixed executable descriptor {target_fd} is already in use"),
                ),
            }),
        }
    }

    fn read_retry<R: Read>(reader: &mut R, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            match reader.read(buffer) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => return result,
            }
        }
    }

    fn validate_execution_path(
        file: &File,
        execution_path: &Path,
        expected: ObjectSnapshot,
        display_path: &Path,
    ) -> Result<(), PinExecutableError> {
        let descriptor_metadata =
            file.metadata()
                .map_err(|source| PinExecutableError::Inspect {
                    path: display_path.to_path_buf(),
                    source,
                })?;
        let descriptor = ObjectSnapshot::from_metadata(&descriptor_metadata);
        if !expected.same_execution_object(descriptor) {
            return Err(PinExecutableError::ExecutionObjectChanged {
                path: display_path.to_path_buf(),
            });
        }

        let execution_metadata = std::fs::metadata(execution_path).map_err(|source| {
            PinExecutableError::ExecutionStrategy {
                path: display_path.to_path_buf(),
                source,
            }
        })?;
        let execution = ObjectSnapshot::from_metadata(&execution_metadata);
        if descriptor.device != execution.device || descriptor.inode != execution.inode {
            return Err(PinExecutableError::ExecutionObjectChanged {
                path: display_path.to_path_buf(),
            });
        }
        rustix::fs::access(execution_path, Access::EXEC_OK).map_err(|source| {
            PinExecutableError::ExecutionStrategy {
                path: display_path.to_path_buf(),
                source: source.into(),
            }
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::pinned_executable_test_directory::TestDirectory;
        use std::fs::{self, FileTimes, OpenOptions};
        use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
        use std::time::{Duration, Instant};

        fn write_executable(path: &Path, contents: &[u8]) {
            fs::write(path, contents).expect("write executable fixture");
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(path, permissions).expect("make fixture executable");
        }

        fn copy_executable(source: &Path, destination: &Path) {
            fs::copy(source, destination).expect("copy executable fixture");
            let mut permissions = fs::metadata(destination).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(destination, permissions).expect("make copy executable");
        }

        fn sealed_static_elf() -> Vec<u8> {
            const HEADER: usize = 64;
            const PROGRAM: usize = 56;
            const PROGRAMS: usize = 4;
            const CODE_OFFSET: usize = 0x1000;
            let mut bytes = vec![0_u8; CODE_OFFSET + 1];
            bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
            bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
            bytes[18..20].copy_from_slice(&62_u16.to_le_bytes());
            bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
            bytes[24..32].copy_from_slice(&0x401000_u64.to_le_bytes());
            bytes[32..40].copy_from_slice(&(HEADER as u64).to_le_bytes());
            bytes[52..54].copy_from_slice(&(HEADER as u16).to_le_bytes());
            bytes[54..56].copy_from_slice(&(PROGRAM as u16).to_le_bytes());
            bytes[56..58].copy_from_slice(&(PROGRAMS as u16).to_le_bytes());

            let mut program = |index: usize,
                               kind: u32,
                               flags: u32,
                               offset: u64,
                               address: u64,
                               file_size: u64,
                               memory_size: u64,
                               alignment: u64| {
                let start = HEADER + index * PROGRAM;
                bytes[start..start + 4].copy_from_slice(&kind.to_le_bytes());
                bytes[start + 4..start + 8].copy_from_slice(&flags.to_le_bytes());
                bytes[start + 8..start + 16].copy_from_slice(&offset.to_le_bytes());
                bytes[start + 16..start + 24].copy_from_slice(&address.to_le_bytes());
                bytes[start + 32..start + 40].copy_from_slice(&file_size.to_le_bytes());
                bytes[start + 40..start + 48].copy_from_slice(&memory_size.to_le_bytes());
                bytes[start + 48..start + 56].copy_from_slice(&alignment.to_le_bytes());
            };
            let table_size = (PROGRAM * PROGRAMS) as u64;
            program(0, 6, 4, HEADER as u64, 0x400040, table_size, table_size, 8);
            program(
                1,
                1,
                4,
                0,
                0x400000,
                (HEADER as u64) + table_size,
                (HEADER as u64) + table_size,
                0x1000,
            );
            program(2, 1, 5, CODE_OFFSET as u64, 0x401000, 1, 1, 0x1000);
            program(3, 0x6474_e551, 6, 0, 0, 0, 0, 16);
            bytes[CODE_OFFSET] = 0xc3;
            bytes
        }

        #[test]
        fn digest_is_computed_from_the_opened_object() {
            let root = TestDirectory::new();
            let path = root.path().join("tool");
            write_executable(&path, b"abc");

            let pinned = PinnedExecutable::open(&path).unwrap();

            assert_eq!(pinned.size(), 3);
            assert_eq!(
                pinned.sha256(),
                &[
                    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d,
                    0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10,
                    0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
                ]
            );
        }

        #[test]
        fn sealed_static_v3_identity_matches_the_shared_derivation() {
            let root = TestDirectory::new();
            let path = root.path().join("static-app");
            let exact = sealed_static_elf();
            write_executable(&path, &exact);
            let pinned = PinnedExecutable::open(&path).unwrap();
            let sealed = pinned.seal_static_application().unwrap();

            assert_eq!(
                sealed.identity_v3(),
                fe2o3_runtime_protocol::WorkerV3ApplicationIdentityV1::from_sealed_static_elf_v1(
                    &exact,
                )
                .unwrap()
            );
        }

        #[test]
        fn sealed_static_snapshot_is_exact_immutable_and_source_independent() {
            let root = TestDirectory::new();
            let path = root.path().join("static-app");
            let original = sealed_static_elf();
            write_executable(&path, &original);
            let pinned = PinnedExecutable::open(&path).unwrap();
            let sealed = pinned.seal_static_application().unwrap();
            let identity = sealed.identity_v3();

            let mut first_mutation = original.clone();
            *first_mutation.last_mut().unwrap() ^= 0xff;
            fs::write(&path, &first_mutation).unwrap();
            let command = sealed.command().unwrap();
            assert_eq!(sealed.bytes(), original);
            assert_eq!(sealed.identity_v3(), identity);

            let mut second_mutation = first_mutation;
            second_mutation[0] ^= 0xff;
            fs::write(&path, &second_mutation).unwrap();
            assert_eq!(sealed.bytes(), original);
            assert_eq!(sealed.identity_v3(), identity);
            assert_eq!(command.as_command().get_program(), sealed.execution_path);
        }

        #[test]
        fn sealed_executable_image_is_independent_of_later_source_mutation() {
            let root = TestDirectory::new();
            let path = root.path().join("cargo-image");
            write_executable(&path, b"#!/bin/sh\nexit 0\n");
            let source = PinnedExecutable::open(&path).unwrap();
            assert!(source.require_sealed_executable_image().is_err());
            let expected = *source.sha256();
            let sealed = source.seal_executable_image().unwrap();
            sealed.require_sealed_executable_image().unwrap();

            write_executable(&path, b"#!/bin/sh\nexit 41\n");
            assert_eq!(sealed.sha256(), &expected);
            assert_ne!(PinnedExecutable::open(&path).unwrap().sha256(), &expected);
            sealed
                .command()
                .expect("sealed Cargo image remains executable");
        }

        #[test]
        fn sealed_executable_is_inherited_at_one_verified_fixed_child_descriptor() {
            // The unit-test binary shares one descriptor table across hundreds of parallel
            // tests. Keep this probe above their working set; production 191/192 are exercised
            // by the process-isolated release integration tests.
            const CHILD_FD: RawFd = 511;
            let root = TestDirectory::new();
            let path = root.path().join("wrapper-image");
            write_executable(&path, b"#!/bin/sh\nexit 0\n");
            let source = PinnedExecutable::open(&path).unwrap();
            let sealed = source.seal_executable_image().unwrap();
            assert_eq!(
                sealed.fixed_child_path(CHILD_FD).unwrap(),
                PathBuf::from("/proc/self/fd/511")
            );

            let mut command = Command::new("/bin/sh");
            command
                .arg("-c")
                .arg("test -r /proc/self/fd/511 && test \"$(stat -c %a /proc/self/fd/511)\" = 500");
            sealed.inherit_for_child_at(&mut command, CHILD_FD).unwrap();
            assert!(
                crate::process_execution::status(&mut command)
                    .unwrap()
                    .success()
            );
        }

        #[test]
        fn same_inode_mutation_is_fail_closed_at_every_snapshot_barrier() {
            for target in [
                StaticSnapshotBarrier::SourceValidated,
                StaticSnapshotBarrier::SourceCopied,
                StaticSnapshotBarrier::SnapshotSealed,
                StaticSnapshotBarrier::SnapshotBound,
            ] {
                let root = TestDirectory::new();
                let path = root.path().join("static-app");
                let original = sealed_static_elf();
                write_executable(&path, &original);
                let pinned = PinnedExecutable::open(&path).unwrap();
                let mut mutated = original.clone();
                *mutated.last_mut().unwrap() ^= 0xff;
                let result = pinned.seal_static_application_with_barrier(|barrier| {
                    if barrier == target {
                        fs::write(&path, &mutated).unwrap();
                    }
                });

                match (target, result) {
                    (StaticSnapshotBarrier::SourceValidated, Err(_)) => {}
                    (StaticSnapshotBarrier::SourceValidated, Ok(_)) => {
                        panic!("source mutation escaped before capture")
                    }
                    (StaticSnapshotBarrier::SourceCopied, Err(_)) => {}
                    (_, Ok(sealed)) => {
                        assert_eq!(sealed.bytes(), original, "snapshot changed at {target:?}");
                        sealed.command().unwrap();
                    }
                    (_, Err(error)) => {
                        panic!("sealed snapshot changed after {target:?}: {error}")
                    }
                }
            }
        }

        #[test]
        fn sealed_static_snapshot_rejects_write_resize_and_new_seals() {
            let root = TestDirectory::new();
            let path = root.path().join("static-app");
            let original = sealed_static_elf();
            write_executable(&path, &original);
            let pinned = PinnedExecutable::open(&path).unwrap();
            let sealed = pinned.seal_static_application().unwrap();

            assert_eq!(
                rustix::fs::fcntl_get_seals(&sealed.file).unwrap(),
                SealFlags::WRITE | SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL
            );
            if let Ok(mut writable) = OpenOptions::new().write(true).open(&sealed.execution_path) {
                assert!(writable.write_all(b"attacker").is_err());
                assert!(writable.set_len(1).is_err());
                assert!(writable.set_len(original.len() as u64 + 1).is_err());
                assert!(rustix::fs::fcntl_add_seals(&writable, SealFlags::EXEC).is_err());
            }
            assert_eq!(sealed.bytes(), original);
        }

        #[test]
        fn sealed_static_command_rejects_descriptor_identity_substitution() {
            let root = TestDirectory::new();
            let first_path = root.path().join("first");
            let second_path = root.path().join("second");
            write_executable(&first_path, &sealed_static_elf());
            let mut second_bytes = sealed_static_elf();
            second_bytes.push(0);
            write_executable(&second_path, &second_bytes);
            let first = PinnedExecutable::open(&first_path)
                .unwrap()
                .seal_static_application()
                .unwrap();
            let second = PinnedExecutable::open(&second_path)
                .unwrap()
                .seal_static_application()
                .unwrap();
            let mut substituted = first;
            substituted.execution_path = second.execution_path.clone();

            assert!(matches!(
                substituted.command(),
                Err(PinExecutableError::ExecutionObjectChanged { .. })
            ));
        }

        #[test]
        fn exact_hash_policy_rejects_short_and_growing_streams() {
            let path = Path::new("fixture");
            let mut short = &b"abc"[..];
            assert!(matches!(
                hash_exact(&mut short, path, 4),
                Err(PinExecutableError::UnexpectedEof {
                    expected: 4,
                    actual: 3,
                    ..
                })
            ));

            let mut growing = &b"abc"[..];
            assert!(matches!(
                hash_exact(&mut growing, path, 2),
                Err(PinExecutableError::GrewDuringRead { expected: 2, .. })
            ));

            let mut short = &b"abc"[..];
            let mut snapshot = Vec::new();
            assert!(matches!(
                copy_exact(&mut short, &mut snapshot, path, 4),
                Err(PinExecutableError::UnexpectedEof {
                    expected: 4,
                    actual: 3,
                    ..
                })
            ));
        }

        #[test]
        fn final_symlink_is_rejected_without_following_it() {
            let root = TestDirectory::new();
            let target = root.path().join("target");
            let link = root.path().join("tool");
            write_executable(&target, b"abc");
            symlink(&target, &link).unwrap();

            assert!(matches!(
                PinnedExecutable::open(&link),
                Err(PinExecutableError::Open { .. })
            ));
        }

        fn inherited_image(sealed: bool) -> (File, PathBuf) {
            use std::io::Write;

            let bytes = fs::read("/bin/true").unwrap();
            let fd = rustix::fs::memfd_create(
                "fe2o3-inherited-rustc-test",
                rustix::fs::MemfdFlags::ALLOW_SEALING,
            )
            .unwrap();
            let mut image = File::from(fd);
            image.write_all(&bytes).unwrap();
            image
                .set_permissions(fs::Permissions::from_mode(0o555))
                .unwrap();
            if sealed {
                rustix::fs::fcntl_add_seals(
                    &image,
                    SealFlags::WRITE | SealFlags::GROW | SealFlags::SHRINK,
                )
                .and_then(|()| rustix::fs::fcntl_add_seals(&image, SealFlags::SEAL))
                .unwrap();
            }
            let path = PathBuf::from(format!("/proc/self/fd/{}", image.as_raw_fd()));
            (image, path)
        }

        #[test]
        fn accepts_only_fully_sealed_inherited_descriptors() {
            let (_image, path) = inherited_image(true);
            assert!(PinnedExecutable::open(&path).is_ok());

            let (_image, path) = inherited_image(false);
            assert!(matches!(
                PinnedExecutable::open(&path),
                Err(PinExecutableError::ExecutionStrategy { .. })
            ));
        }

        #[test]
        fn non_regular_object_is_rejected() {
            let root = TestDirectory::new();

            assert!(matches!(
                PinnedExecutable::open(root.path()),
                Err(PinExecutableError::NotRegular { .. })
            ));
        }

        #[test]
        fn fifo_is_rejected_without_waiting_for_a_writer() {
            let root = TestDirectory::new();
            let fifo = root.path().join("fifo");
            rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, Mode::RUSR | Mode::WUSR | Mode::XUSR)
                .unwrap();

            let writer_path = fifo.clone();
            let delayed_writer = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(400));
                let _ = rustix::fs::open(
                    &writer_path,
                    OFlags::WRONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                );
            });
            let started = Instant::now();
            let result = PinnedExecutable::open(&fifo);
            let elapsed = started.elapsed();
            delayed_writer.join().unwrap();

            assert!(matches!(result, Err(PinExecutableError::NotRegular { .. })));
            assert!(
                elapsed < Duration::from_millis(250),
                "opening a FIFO blocked for {elapsed:?}"
            );
        }

        #[test]
        fn regular_file_without_execute_bits_is_rejected() {
            let root = TestDirectory::new();
            let path = root.path().join("tool");
            fs::write(&path, b"abc").unwrap();
            let mut permissions = fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o600);
            fs::set_permissions(&path, permissions).unwrap();

            assert!(matches!(
                PinnedExecutable::open(&path),
                Err(PinExecutableError::NotExecutable { .. })
            ));
        }

        #[test]
        fn empty_executable_is_rejected() {
            let root = TestDirectory::new();
            let path = root.path().join("tool");
            write_executable(&path, b"");

            assert!(matches!(
                PinnedExecutable::open(&path),
                Err(PinExecutableError::Empty { .. })
            ));
        }

        #[test]
        fn oversized_file_is_rejected_before_it_is_read() {
            let root = TestDirectory::new();
            let path = root.path().join("tool");
            let file = File::create(&path).unwrap();
            file.set_len(MAX_EXECUTABLE_BYTES + 1).unwrap();
            drop(file);
            let mut permissions = fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&path, permissions).unwrap();

            assert!(matches!(
                PinnedExecutable::open(&path),
                Err(PinExecutableError::TooLarge { .. })
            ));
        }

        #[test]
        fn ordinary_host_test_limit_preserves_default_and_transfer_boundaries() {
            assert_eq!(MAX_EXECUTABLE_BYTES, 512 * 1024 * 1024);
            assert_eq!(MAX_BINDING_HOST_TEST_BYTES, 1024 * 1024 * 1024);
            let root = TestDirectory::new();
            let path = root.path().join("host-test");
            write_executable(&path, b"x");
            let resize = |size| {
                OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len(size)
                    .unwrap();
            };
            resize(MAX_EXECUTABLE_BYTES);
            assert!(PinnedExecutable::open(&path).is_ok());
            resize(MAX_EXECUTABLE_BYTES + 1);
            assert!(matches!(
                PinnedExecutable::open(&path),
                Err(PinExecutableError::TooLarge {
                    limit: MAX_EXECUTABLE_BYTES,
                    ..
                })
            ));
            assert!(matches!(
                PinnedExecutable::from_transferred_file(File::open(&path).unwrap(), path.clone()),
                Err(PinExecutableError::TooLarge {
                    limit: MAX_EXECUTABLE_BYTES,
                    ..
                })
            ));
            assert!(PinnedExecutable::open_binding_host_test(&path).is_ok());
            resize(MAX_BINDING_HOST_TEST_BYTES);
            assert!(PinnedExecutable::open_binding_host_test(&path).is_ok());
            resize(MAX_BINDING_HOST_TEST_BYTES + 1);
            assert!(matches!(
                PinnedExecutable::open_binding_host_test(&path),
                Err(PinExecutableError::TooLarge {
                    limit: MAX_BINDING_HOST_TEST_BYTES,
                    ..
                })
            ));
        }

        #[test]
        fn ordinary_host_test_open_retains_file_and_post_pin_refusals() {
            let root = TestDirectory::new();
            let path = root.path().join("host-test");
            write_executable(&path, b"");
            assert!(matches!(
                PinnedExecutable::open_binding_host_test(&path),
                Err(PinExecutableError::Empty { .. })
            ));
            write_executable(&path, b"abc");
            let alias = root.path().join("alias");
            symlink(&path, &alias).unwrap();
            assert!(matches!(
                PinnedExecutable::open_binding_host_test(&alias),
                Err(PinExecutableError::Open { .. })
            ));
            assert!(matches!(
                PinnedExecutable::open_binding_host_test(root.path()),
                Err(PinExecutableError::NotRegular { .. })
            ));
            let pinned = PinnedExecutable::open_binding_host_test(&path).unwrap();
            fs::write(&path, b"changed").unwrap();
            assert!(matches!(
                pinned.command(),
                Err(PinExecutableError::ExecutionObjectChanged { .. })
            ));
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            assert!(matches!(
                PinnedExecutable::open_binding_host_test(&path),
                Err(PinExecutableError::NotExecutable { .. })
            ));
        }

        #[test]
        fn retained_descriptor_prevents_path_trampoline() {
            let root = TestDirectory::new();
            let selected = root.path().join("rustc");
            let replacement = root.path().join("replacement");
            copy_executable(Path::new("/bin/true"), &selected);
            let pinned = PinnedExecutable::open(&selected).unwrap();
            let selected_digest = *pinned.sha256();

            copy_executable(Path::new("/bin/false"), &replacement);
            fs::rename(&replacement, &selected).expect("atomically replace selected pathname");

            let pinned_result = pinned.command();
            match pinned_result {
                Ok(command) => {
                    assert_eq!(command.as_command().get_program(), pinned.execution_path());
                    assert_eq!(command.configured_argv0(), selected.as_os_str());
                    drop(command);
                    assert!(
                        crate::process_execution::retry_transient_executable_busy(|| {
                            pinned.command().unwrap().status()
                        })
                        .unwrap()
                        .success(),
                        "pinned command reopened replacement"
                    );
                }
                Err(PinExecutableError::ExecutionObjectChanged { .. }) => {}
                Err(error) => panic!("unexpected substitution result: {error}"),
            }
            assert_eq!(pinned.sha256(), &selected_digest);
            let replacement_bytes = fs::read(&selected).unwrap();
            assert_eq!(replacement_bytes, fs::read("/bin/false").unwrap());
            let replacement_digest: [u8; 32] = Sha256::digest(replacement_bytes).into();
            assert_ne!(pinned.sha256(), &replacement_digest);
        }

        #[test]
        fn same_size_mutation_with_restored_mtime_is_rejected_by_ctime() {
            let root = TestDirectory::new();
            let selected = root.path().join("rustc");
            copy_executable(Path::new("/bin/true"), &selected);
            let pinned = PinnedExecutable::open(&selected).unwrap();
            let original_metadata = fs::metadata(&selected).unwrap();
            let original_modified = original_metadata.modified().unwrap();
            let original_ctime = (original_metadata.ctime(), original_metadata.ctime_nsec());

            let mut bytes = fs::read(&selected).unwrap();
            bytes[0] ^= 0xff;
            let deadline = Instant::now() + Duration::from_secs(2);
            let changed_metadata = loop {
                fs::write(&selected, &bytes).unwrap();
                File::options()
                    .write(true)
                    .open(&selected)
                    .unwrap()
                    .set_times(FileTimes::new().set_modified(original_modified))
                    .unwrap();
                let metadata = fs::metadata(&selected).unwrap();
                if (metadata.ctime(), metadata.ctime_nsec()) != original_ctime {
                    break metadata;
                }
                assert!(
                    Instant::now() < deadline,
                    "fixture filesystem did not expose a ctime change"
                );
                std::thread::sleep(Duration::from_millis(20));
            };
            assert_eq!(changed_metadata.len(), original_metadata.len());
            assert_eq!(changed_metadata.mode(), original_metadata.mode());
            assert_eq!(changed_metadata.modified().unwrap(), original_modified);

            assert!(matches!(
                pinned.command(),
                Err(PinExecutableError::ExecutionObjectChanged { .. })
            ));
        }

        #[test]
        fn descriptor_lives_through_command_and_closes_with_pin() {
            let root = TestDirectory::new();
            let selected = root.path().join("rustc");
            copy_executable(Path::new("/bin/true"), &selected);
            let pinned = PinnedExecutable::open(&selected).unwrap();
            let descriptor_path = pinned.execution_path().to_path_buf();
            let descriptor = pinned.raw_fd();
            let pinned_metadata = fs::metadata(&descriptor_path).unwrap();

            {
                let command = pinned.command().unwrap();
                assert_eq!(
                    command.as_command().get_program(),
                    descriptor_path.as_os_str()
                );
                assert_eq!(command.configured_argv0(), selected.as_os_str());
                let during_command = fs::metadata(&descriptor_path).unwrap();
                assert_eq!(pinned_metadata.dev(), during_command.dev());
                assert_eq!(pinned_metadata.ino(), during_command.ino());
            }
            let after_command = fs::metadata(&descriptor_path).unwrap();
            assert_eq!(pinned_metadata.dev(), after_command.dev());
            assert_eq!(pinned_metadata.ino(), after_command.ino());

            drop(pinned);
            if let Ok(reused) = fs::metadata(format!("/proc/self/fd/{descriptor}")) {
                assert!(
                    reused.dev() != pinned_metadata.dev() || reused.ino() != pinned_metadata.ino(),
                    "dropping the pin left its executable descriptor open"
                );
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[allow(unused_imports)] // Used by the parent when compile execution is activated.
pub(crate) use platform::PinnedExecutable;

#[cfg(not(target_os = "linux"))]
pub(crate) struct PinnedExecutable;

#[cfg(not(target_os = "linux"))]
impl PinnedExecutable {
    pub(crate) fn open(_path: &Path) -> Result<Self, PinExecutableError> {
        Err(PinExecutableError::UnsupportedPlatform)
    }

    pub(crate) fn open_binding_host_test(_path: &Path) -> Result<Self, PinExecutableError> {
        Err(PinExecutableError::UnsupportedPlatform)
    }
}

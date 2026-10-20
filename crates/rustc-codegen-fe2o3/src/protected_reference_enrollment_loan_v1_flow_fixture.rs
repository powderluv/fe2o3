//! Real process-consistency admission, not a protected production compiler.
//!
//! The parent seals the child's exact argv/cwd/environment and measured test
//! executable. FD198 holds that same real image, not an installed codegen DSO.
//! All admission and loan revalidation use the production entry points. These
//! fixtures establish no compiler provenance, receipt, proof or native pass.

#![cfg(test)]

use super::super::{Budget, Invocation, Policy};
use crate::protected_compiler_execution::{
    OwnedExecutionInputs,
    native_v3::{Admitted, Client},
    tests::install,
};
use fe2o3_build_authority::CompilerClosureV2;
use fe2o3_compiler_closure_capability::{
    RUSTC_INVOCATION_CHILD_FD_V1, RustcInvocationCapabilityV1,
};
use fe2o3_compiler_execution_protocol::{
    CompilerExecutionIssuerMeasurementV1 as Measurement, CompilerExecutionIssuerPolicyV3,
};
use fe2o3_process_identity::{
    CODEGEN_BACKEND_BUILD_OBSERVATION_ENV_V2, CompilerImageRoleV1,
    EXPECTED_COMPILER_CLOSURE_SHA256_ENV_V1, measure_compiler_image_sha256_v1,
};
use fe2o3_rustc_invocation::{
    CompileEnvironmentV2, RustcInvocationDescriptorV2, RustcInvocationDescriptorV3, RustcUnitV2,
};
use rustix::net::{AddressFamily, SocketFlags, SocketType, socketpair};
use std::{
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::process::CommandExt,
    },
    process::Command,
    time::{Duration, Instant},
};

const CHILD: &str = "FE2O3_REFERENCE_LOAN_FLOW_CHILD";
const BACKEND_FD: i32 = 198;

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

const REPORT_DIAGNOSTIC_LIMIT: usize = 16 * 1024;
const PROC_DIAGNOSTIC_LIMIT: usize = 4 * 1024;

fn bounded_diagnostic(path: &std::path::Path, limit: usize) -> String {
    use std::io::Read;
    let read = || -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)?;
        Ok(bytes)
    };
    match read() {
        Ok(bytes) => {
            let retained = bytes.len().min(limit);
            let suffix = if bytes.len() > limit {
                " [truncated]"
            } else {
                ""
            };
            format!(
                "{retained} bytes{suffix}: {:?}",
                String::from_utf8_lossy(&bytes[..retained])
            )
        }
        Err(error) => format!("unavailable: {error}"),
    }
}

fn child_diagnostics(report: &std::path::Path, pid: u32, elapsed: Duration) -> String {
    let mut output = format!(
        "owned child pid={pid} elapsed={elapsed:?}\npartial libtest report: {}\n",
        bounded_diagnostic(report, REPORT_DIAGNOSTIC_LIMIT)
    );
    for name in ["status", "stat", "io", "wchan"] {
        let path = std::path::PathBuf::from(format!("/proc/{pid}/{name}"));
        output.push_str(&format!(
            "child /proc/{name}: {}\n",
            bounded_diagnostic(&path, PROC_DIAGNOSTIC_LIMIT)
        ));
    }
    output
}

#[test]
fn diagnostic_report_format_is_bounded_and_escaped() {
    let directory = crate::test_temp_dir::TestTempDir::create("fe2o3-loan-diagnostic");
    let report = directory.path().join("partial.jsonl");
    std::fs::write(&report, b"abcdmore").unwrap();
    assert_eq!(
        bounded_diagnostic(&report, 4),
        "4 bytes [truncated]: \"abcd\""
    );
    std::fs::write(&report, b"a\n\xff").unwrap();
    assert_eq!(
        bounded_diagnostic(&report, 3),
        format!("3 bytes: {:?}", String::from_utf8_lossy(b"a\n\xff"))
    );
    std::fs::write(&report, b"").unwrap();
    assert_eq!(bounded_diagnostic(&report, 4), "0 bytes: \"\"");
}

#[test]
fn diagnostic_missing_report_and_process_are_explicit() {
    let directory = crate::test_temp_dir::TestTempDir::create("fe2o3-loan-diagnostic");
    let report = directory.path().join("missing.jsonl");
    assert!(bounded_diagnostic(&report, REPORT_DIAGNOSTIC_LIMIT).starts_with("unavailable: "));
    // PID zero has no proc directory; this exercises diagnostics without a child.
    let output = child_diagnostics(&report, 0, Duration::from_secs(7));
    assert!(output.starts_with("owned child pid=0 elapsed=7s\n"));
    assert!(output.contains("partial libtest report: unavailable: "));
    for name in ["status", "stat", "io", "wchan"] {
        assert!(output.contains(&format!("child /proc/{name}: unavailable: ")));
    }
}

/// Mirrors the existing isolated fixed-slot harness. The extra libtest skip
/// value is the required final descriptor argument; it selects no extra test.
pub(super) fn isolated(full_name: &str, request: Option<&str>) -> bool {
    let name = full_name.split_once("::").unwrap().1;
    if let Some(value) = std::env::var_os(CHILD) {
        assert_eq!(value, name);
        assert_eq!(
            std::env::var(crate::reference_enrollment_policy_v1::ENV)
                .ok()
                .as_deref(),
            request
        );
        return false;
    }
    let directory = crate::test_temp_dir::TestTempDir::create("fe2o3-loan-flow");
    let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
    let digest =
        measure_compiler_image_sha256_v1(&executable, CompilerImageRoleV1::Executable, |_| {
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
    // These descriptive closure entries intentionally identify the test image.
    // An independent protected compiler/supervisor has not approved them.
    let closure = CompilerClosureV2::new(digest, digest, digest, digest, digest, digest).unwrap();
    let report = directory.path().join("libtest.jsonl");
    let mut command = Command::new(&executable);
    command
        .args([
            "--exact",
            name,
            "--test-threads=1",
            "--format=json",
            "-Zunstable-options",
            "--nocapture",
            "--skip",
        ])
        .arg(format!(
            "-Zcodegen-backend={}",
            fe2o3_artifact_transaction::BROKERED_CODEGEN_BACKEND_PATH_V1
        ))
        .current_dir(directory.path())
        .stdout(File::create_new(&report).unwrap())
        .env_clear()
        .env(CHILD, name)
        .env("LANG", "C.UTF-8")
        .env(
            "FE2O3_TARGET",
            fe2o3_amd_target::PRODUCTION_GFX942_DEVICE_TARGET_V1,
        )
        .env(
            "FE2O3_HSACO_DIR",
            fe2o3_artifact_transaction::BROKERED_ARTIFACT_DIRECTORY_PATH_V1,
        )
        .env(
            EXPECTED_COMPILER_CLOSURE_SHA256_ENV_V1,
            hex(closure.identity_sha256()),
        )
        .env(CODEGEN_BACKEND_BUILD_OBSERVATION_ENV_V2, hex(digest));
    // Preserve only dynamic-library loading. The collector resolves its linked
    // driver sysroot in process, without PATH, HOME or a rustup proxy selection.
    if let Some(value) = std::env::var_os("LD_LIBRARY_PATH") {
        command.env("LD_LIBRARY_PATH", value);
    }
    if let Some(request) = request {
        command.env(crate::reference_enrollment_policy_v1::ENV, request);
    }
    let argv = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|arg| arg.to_str().unwrap().to_owned())
        .collect();
    let environment = CompileEnvironmentV2::from_child_environment(
        command
            .get_envs()
            .map(|(key, value)| (key.to_owned(), value.unwrap().to_owned())),
    )
    .unwrap();
    let descriptor = RustcInvocationDescriptorV3::new(
        RustcInvocationDescriptorV2::new(
            digest,
            digest,
            RustcUnitV2::new(
                directory.path().canonicalize().unwrap().to_str().unwrap(),
                argv,
            )
            .unwrap(),
            environment,
        )
        .unwrap(),
        closure,
    )
    .unwrap();
    let sealed = RustcInvocationCapabilityV1::create(descriptor).unwrap();
    let transfer = sealed.try_clone_for_transfer().unwrap();
    let backend = File::open(&executable).unwrap();
    // Keep sources above all fixed slots so the child remap cannot alias them.
    let backend = rustix::io::fcntl_dupfd_cloexec(&backend, 256).unwrap();
    let transfer = rustix::io::fcntl_dupfd_cloexec(&transfer, 256).unwrap();
    let backend_fd = backend.as_raw_fd();
    let transfer_fd = transfer.as_raw_fd();
    // SAFETY: the post-fork closure uses only scalar fd operations on the
    // child's copies. Parent descriptors and protocol sessions stay untouched.
    unsafe {
        command.pre_exec(move || {
            for fd in [195, 202] {
                if libc::close(fd) != 0
                    && io::Error::last_os_error().raw_os_error() != Some(libc::EBADF)
                {
                    return Err(io::Error::last_os_error());
                }
            }
            for (source, target) in [
                (backend_fd, BACKEND_FD),
                (transfer_fd, RUSTC_INVOCATION_CHILD_FD_V1),
            ] {
                if libc::dup2(source, target) != target {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    // This is the fixture's sole child. The CHILD branch cannot re-exec, and
    // the collector's fixed-source callback runs rustc in process with no
    // helper, proc macro or linker. Every wait failure/deadline kills and reaps
    // this owned child; successful exits are reaped before report assertions.
    let mut child = command.spawn().unwrap();
    let started = Instant::now();
    let deadline = started + Duration::from_secs(60);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = std::fs::read_to_string(&report).unwrap();
                eprint!("{output}");
                assert!(status.success(), "isolated loan flow: {status}");
                assert_child_report(&output, name);
                return true;
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            result => {
                let diagnostics = child_diagnostics(&report, child.id(), started.elapsed());
                let killed = child.kill();
                let waited = child.wait();
                panic!(
                    "isolated loan flow timed out or failed: {result:?}\n{diagnostics}owned child kill={killed:?} wait={waited:?}"
                );
            }
        }
    }
}

fn assert_child_report(output: &str, name: &str) {
    let events: Vec<serde_json::Value> = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("child libtest JSON"))
        .collect();
    let [start, entered, passed, finish] = events.as_slice() else {
        panic!("expected one child suite and one executed test: {output}")
    };
    assert_eq!(start["type"], "suite");
    assert_eq!(start["event"], "started");
    assert_eq!(start["test_count"], 1);
    for (event, expected) in [(entered, "started"), (passed, "ok")] {
        assert_eq!(event["type"], "test");
        assert_eq!(event["name"], name);
        assert_eq!(event["event"], expected);
    }
    assert_eq!(finish["type"], "suite");
    assert_eq!(finish["event"], "ok");
    assert_eq!(finish["passed"], 1);
    assert_eq!(finish["failed"], 0);
    assert_eq!(finish["ignored"], 0);
    assert_eq!(finish["measured"], 0);
}

pub(super) fn admit() -> (Invocation, OwnedFd) {
    assert!(std::env::var_os(CHILD).is_some());
    // SAFETY: this isolated exec inherited FD198 from our remap, without a Rust
    // owner. This sole owner outlives every live image revalidation in the test.
    let backend = unsafe { OwnedFd::from_raw_fd(BACKEND_FD) };
    rustix::io::fcntl_setfd(&backend, rustix::io::FdFlags::CLOEXEC).unwrap();
    let invocation = crate::protected_rustc_invocation::admit_for_production_codegen()
        .unwrap()
        .expect("actual process admission");
    assert_invocation_slot_consumed();
    (invocation, backend)
}

pub(super) fn equal_invocation(original: &Invocation) -> Invocation {
    let capability = RustcInvocationCapabilityV1::create(original.descriptor().clone()).unwrap();
    let transfer = capability.try_clone_for_transfer().unwrap();
    install(&transfer, RUSTC_INVOCATION_CHILD_FD_V1, false);
    let other = crate::protected_rustc_invocation::admit_for_production_codegen()
        .unwrap()
        .expect("second actual process admission");
    assert_invocation_slot_consumed();
    assert_eq!(original.descriptor(), other.descriptor());
    assert_eq!(
        capability.canonical_bytes(),
        fe2o3_rustc_invocation::encode_descriptor_v3(other.descriptor()).unwrap()
    );
    other
}

fn assert_invocation_slot_consumed() {
    // SAFETY: F_GETFD observes only this isolated child's fixed descriptor slot.
    assert_eq!(
        unsafe { libc::fcntl(RUSTC_INVOCATION_CHILD_FD_V1, libc::F_GETFD) },
        -1
    );
    assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EBADF));
}

pub(super) fn policy(budget: &mut Budget<'_>) -> Policy {
    // Same inert public points as the existing native session fixtures. No
    // signing key or simulated receipt service is created.
    let mut key = [0x66; 32];
    key[0] = 0x58;
    let mut anchor = key;
    anchor[31] ^= 0x80;
    let (policy, charge) = CompilerExecutionIssuerPolicyV3::new(
        1,
        Measurement::new([1; 32], 1).unwrap(),
        Measurement::new([2; 32], 1).unwrap(),
        key,
        anchor,
        budget,
    )
    .unwrap();
    budget.reserve_storage(charge.additional_storage()).unwrap();
    let (policy, charge) = Policy::create(policy, budget).unwrap();
    budget.reserve_storage(charge.additional_storage()).unwrap();
    policy
}

pub(super) fn session<'b, 'w>(
    policy: &Policy,
    budget: &'b mut Budget<'w>,
) -> (Admitted<'b, 'w>, OwnedFd) {
    let (file, charge) = policy.try_clone_for_transfer(budget).unwrap();
    budget
        .reserve_storage(charge.additional_storage() + Client::PEER_STORAGE)
        .unwrap();
    let (client, server) = socketpair(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    let session = Admitted::from_owned(
        OwnedExecutionInputs {
            policy: file.into(),
            service: client,
        },
        budget,
    )
    .unwrap();
    (session, server)
}

pub(super) struct ChangedDirectory {
    original: std::path::PathBuf,
    _directory: crate::test_temp_dir::TestTempDir,
}

impl ChangedDirectory {
    pub(super) fn enter() -> Self {
        let original = std::env::current_dir().unwrap();
        let directory = crate::test_temp_dir::TestTempDir::create("fe2o3-loan-stale-cwd");
        std::env::set_current_dir(directory.path()).unwrap();
        Self {
            original,
            _directory: directory,
        }
    }
}

impl Drop for ChangedDirectory {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.original).unwrap();
    }
}

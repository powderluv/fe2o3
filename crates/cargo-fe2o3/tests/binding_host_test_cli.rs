use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestWorkspace(PathBuf);

impl TestWorkspace {
    fn new() -> Self {
        loop {
            let suffix = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "fe2o3-binding-host-test-cli-{}-{suffix}",
                std::process::id()
            ));
            match std::fs::create_dir(&root) {
                Ok(()) => return Self(root),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("failed to create test workspace: {error}"),
            }
        }
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture parent");
    std::fs::write(path, contents).expect("write fixture");
}

fn write_executable(root: &Path, relative: &str, contents: &str) -> PathBuf {
    write(root, relative, contents);
    let path = root.join(relative);
    let mut permissions = std::fs::metadata(&path)
        .expect("inspect executable fixture")
        .permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&path, permissions).expect("make fixture executable");
    path
}

fn cargo() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn host_target() -> String {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let output = Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("query rustc host target");
    assert!(output.status.success(), "rustc host query failed");
    String::from_utf8(output.stdout)
        .expect("UTF-8 rustc version")
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
        .expect("rustc host target")
}

fn fixture() -> TestWorkspace {
    let workspace = TestWorkspace::new();
    write(
        &workspace.0,
        "Cargo.toml",
        "[workspace]\nresolver = \"2\"\nmembers = [\"managed\"]\n",
    );
    write(
        &workspace.0,
        "managed/Cargo.toml",
        "[package]\nname = \"managed\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    );
    write(
        &workspace.0,
        "managed/src/lib.rs",
        r#"#[cfg(any())] #[kernel(typed)] pub fn managed() {}
const _: () = assert!(option_env!("FE2O3_CRATE_BINDING_ID_V1").is_some());

#[cfg(test)]
mod tests {
    #[test]
    fn binding_is_present_and_runner_custody_is_closed() {
        assert!(option_env!("FE2O3_CRATE_BINDING_ID_V1").is_some());
        if let Some(marker) = std::env::var_os("BINDING_TEST_EXECUTION_MARKER") {
            std::fs::write(marker, b"executed").expect("write host-test execution marker");
        }
        if let Some(output) = std::env::var_os("BINDING_TEST_VALUE_OUTPUT") {
            std::fs::write(output, env!("FE2O3_CRATE_BINDING_ID_V1"))
                .expect("write embedded crate-binding value");
        }
        for (name, value) in std::env::vars_os() {
            let name = name.to_string_lossy();
            assert!(!name.starts_with("FE2O3_"), "protected variable {name}");
            if matches!(name.as_ref(), "RUSTC" | "CARGO_BUILD_RUSTC") {
                assert_eq!(value, "/proc/self/fd/194", "unpinned compiler {name}");
                continue;
            }
            assert!(
                !matches!(
                    name.as_ref(),
                    "RUSTC_WRAPPER"
                        | "CARGO_BUILD_RUSTC_WRAPPER"
                        | "RUSTC_WORKSPACE_WRAPPER"
                        | "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"
                        | "RUSTDOC"
                        | "CARGO_BUILD_RUSTDOC"
                        | "RUSTDOCFLAGS"
                        | "RUSTFLAGS"
                        | "CARGO_ENCODED_RUSTFLAGS"
                        | "LD_PRELOAD"
                        | "LD_AUDIT"
                        | "GLIBC_TUNABLES"
                ),
                "tool or loader variable {name}"
            );
            assert!(!name.starts_with("DYLD_"), "loader variable {name}");
            assert!(name == "LD_LIBRARY_PATH" || !name.starts_with("LD_"), "loader variable {name}");
            assert!(!name.starts_with("CARGO_TARGET_") || !name.ends_with("_RUNNER"), "runner variable {name}");
        }

        unsafe extern "C" {
            fn fcntl(fd: i32, command: i32, ...) -> i32;
        }
        for fd in [191, 192, 195, 196, 197, 198, 199, 200, 201] {
            // SAFETY: F_GETFD only queries whether this integer descriptor is open.
            assert_eq!(unsafe { fcntl(fd, 1) }, -1, "fixed descriptor {fd} remained open");
            assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(9));
        }
        for fd in [193, 194] {
            // SAFETY: F_GETFD only queries whether this integer descriptor is open.
            let flags = unsafe { fcntl(fd, 1) };
            assert!(flags >= 0, "pinned compiler descriptor {fd} was closed");
            assert_eq!(flags & 1, 0, "pinned compiler descriptor {fd} is close-on-exec");
        }
        let loader_path = std::env::var_os("LD_LIBRARY_PATH")
            .expect("pinned rustc library environment");
        assert!(
            std::env::split_paths(&loader_path)
                .any(|component| component == std::path::Path::new("/proc/self/fd/193")),
            "pinned rustc library descriptor is absent from LD_LIBRARY_PATH"
        );
        assert!(std::fs::metadata("/proc/self/fd/193").expect("inspect rustc lib tree").is_dir());
        assert!(std::fs::metadata("/proc/self/fd/194").expect("inspect rustc image").is_file());
        let rustc = std::process::Command::new(
            std::env::var_os("RUSTC").expect("pinned nested rustc environment"),
        )
        .arg("-vV")
        .output()
        .expect("execute pinned nested rustc");
        assert!(rustc.status.success(), "pinned nested rustc failed");

        let executable = std::env::current_exe().expect("resolve original test executable");
        assert!(executable.exists(), "current test executable was detached from its Cargo path");
        assert!(!executable.to_string_lossy().contains("memfd:"));
        let relaunch = std::process::Command::new(executable)
            .arg("--list")
            .output()
            .expect("relaunch Cargo test executable by current_exe path");
        assert!(relaunch.status.success(), "current_exe relaunch failed");
    }
}
"#,
    );
    let status = Command::new(cargo())
        .args(["generate-lockfile", "--offline"])
        .current_dir(&workspace.0)
        .status()
        .expect("generate fixture lockfile");
    assert!(status.success(), "fixture lockfile generation failed");
    workspace
}

const COMPILER_SELECTION_ENVIRONMENT: &[&str] = &[
    "RUSTC",
    "CARGO_BUILD_RUSTC",
    "RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
];

const COMPILER_FLAGS_ENVIRONMENT: &[&str] = &["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"];

fn binding_command(workspace: &TestWorkspace) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cargo-fe2o3"));
    // Keep outer-lane compiler and linker settings out of the independent fixture.
    for name in COMPILER_SELECTION_ENVIRONMENT
        .iter()
        .chain(COMPILER_FLAGS_ENVIRONMENT)
    {
        command.env_remove(name);
    }
    command.env("CARGO", cargo()).current_dir(&workspace.0);
    command
}

fn binding_test(workspace: &TestWorkspace) -> Command {
    let mut command = binding_command(workspace);
    command.args(["test", "--locked", "--all-targets", "-p", "managed"]);
    command
}

fn binding_clippy(workspace: &TestWorkspace) -> Command {
    let mut command = binding_command(workspace);
    command.args(["clippy", "--locked", "--all-targets", "-p", "managed"]);
    command
}

#[test]
fn binding_fixture_removes_inherited_compiler_selection_without_erasing_explicit_overrides() {
    let workspace = TestWorkspace::new();
    for mut command in [
        binding_test(&workspace),
        binding_clippy(&workspace),
        binding_selected_test(&workspace, &["--lib"]),
    ] {
        for name in COMPILER_SELECTION_ENVIRONMENT {
            assert!(
                command
                    .get_envs()
                    .any(|(key, value)| key == std::ffi::OsStr::new(name) && value.is_none())
            );
        }
        assert_eq!(command.get_current_dir(), Some(workspace.0.as_path()));
        for name in COMPILER_SELECTION_ENVIRONMENT {
            command.env(name, "/explicit/test/compiler");
            assert!(command.get_envs().any(|(key, value)| {
                key == std::ffi::OsStr::new(name)
                    && value == Some(std::ffi::OsStr::new("/explicit/test/compiler"))
            }));
        }
    }
}

#[test]
fn binding_fixture_isolates_outer_flags_and_preserves_explicit_flag_overrides() {
    let workspace = TestWorkspace::new();
    for mut command in [
        binding_test(&workspace),
        binding_clippy(&workspace),
        binding_selected_test(&workspace, &["--lib"]),
    ] {
        for name in COMPILER_FLAGS_ENVIRONMENT {
            let key = std::ffi::OsStr::new(name);
            assert!(
                command
                    .get_envs()
                    .any(|(name, value)| name == key && value.is_none())
            );
            command.env(name, "-Copt-level=1");
            assert!(command.get_envs().any(|(name, value)| {
                name == key && value == Some(std::ffi::OsStr::new("-Copt-level=1"))
            }));
        }
    }
}

#[test]
fn binding_fixture_explicit_compiler_selection_still_reaches_production_refusal() {
    let workspace = TestWorkspace::new();
    for name in COMPILER_SELECTION_ENVIRONMENT {
        let output = binding_test(&workspace)
            .env(name, "/does/not/exist/explicit-compiler")
            .output()
            .expect("run explicit compiler selection refusal");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "explicit {name} was accepted");
        let expected = if name.ends_with("WRAPPER") {
            format!("with preexisting {name}=")
        } else {
            format!("rejects preexisting compiler selection {name}=")
        };
        assert!(stderr.contains(&expected), "{name}: {stderr}");
    }
}

#[test]
fn binding_host_clippy_runs_lints_with_the_compiler_derived_binding() {
    let workspace = fixture();
    write(
        &workspace.0,
        "managed/src/lib.rs",
        r#"#[cfg(any())] #[kernel(typed)] pub fn managed() {}
const _: () = assert!(option_env!("FE2O3_CRATE_BINDING_ID_V1").is_some());

#[deny(clippy::identity_op)]
pub fn clippy_probe(value: usize) -> usize {
    value + 0
}
"#,
    );
    let output = binding_clippy(&workspace)
        .env(
            "CARGO_TARGET_DIR",
            workspace.0.join("target-binding-clippy"),
        )
        .output()
        .expect("run binding-aware host clippy");
    assert!(
        !output.status.success(),
        "binding-aware Clippy did not run its lint driver"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("clippy::identity_op"),
        "binding-aware Clippy failed for the wrong reason: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    write(
        &workspace.0,
        "managed/src/lib.rs",
        r#"#[cfg(any())] #[kernel(typed)] pub fn managed() {}
const _: () = assert!(option_env!("FE2O3_CRATE_BINDING_ID_V1").is_some());

pub fn clippy_probe(value: usize) -> usize {
    value
}
"#,
    );
    let output = binding_clippy(&workspace)
        .env(
            "CARGO_TARGET_DIR",
            workspace.0.join("target-binding-clippy-clean"),
        )
        .output()
        .expect("run clean binding-aware host clippy");
    assert!(
        output.status.success(),
        "clean binding-aware Clippy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn close_binding_toolchain_descriptors(command: &mut Command) {
    // SAFETY: the child callback closes only the two fixed descriptors whose absence this
    // subprocess regression validates.
    unsafe {
        command.pre_exec(|| {
            libc::close(193);
            libc::close(194);
            Ok(())
        });
    }
}

#[test]
fn internal_binding_host_runner_rejects_missing_pinned_toolchain_descriptors() {
    let workspace = TestWorkspace::new();
    let marker = workspace.0.join("unexpected-test-execution");
    let test = write_executable(
        &workspace.0,
        "test.sh",
        &format!("#!/bin/sh\nprintf executed > '{}'\n", marker.display()),
    );
    let mut command = Command::new(env!("CARGO_BIN_EXE_cargo-fe2o3"));
    command
        .arg("__fe2o3-binding-host-test-runner-v1")
        .arg(test)
        .env("FE2O3_BINDING_CHECK_WRAPPER_MODE_V1", "1")
        .env("LD_LIBRARY_PATH", "/proc/self/fd/193");
    close_binding_toolchain_descriptors(&mut command);
    let output = command
        .output()
        .expect("run internal binding-only host-test runner");
    assert!(
        !output.status.success(),
        "runner accepted missing descriptors"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("cannot inspect pinned rustc library descriptor 193"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !marker.exists(),
        "test executable ran before descriptor validation"
    );
}

#[test]
fn large_host_test_still_requires_custody_and_obeys_its_finite_bound() {
    let workspace = TestWorkspace::new();
    let image = workspace.0.join("large-host-test");
    std::fs::copy("/bin/true", &image).unwrap();
    std::fs::set_permissions(&image, std::fs::Permissions::from_mode(0o700)).unwrap();
    for (size, marker, error) in [
        (
            512 * 1024 * 1024 + 1,
            false,
            "missing exact binding-only wrapper custody marker",
        ),
        (
            512 * 1024 * 1024 + 1,
            true,
            "cannot inspect pinned rustc library descriptor 193",
        ),
        (
            1024 * 1024 * 1024 + 1,
            true,
            "exceeding the 1073741824-byte hashing limit",
        ),
    ] {
        std::fs::OpenOptions::new()
            .write(true)
            .open(&image)
            .unwrap()
            .set_len(size)
            .unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_cargo-fe2o3"));
        command
            .arg("__fe2o3-binding-host-test-runner-v1")
            .arg(&image)
            .env_remove("FE2O3_BINDING_CHECK_WRAPPER_MODE_V1");
        if marker {
            command.env("FE2O3_BINDING_CHECK_WRAPPER_MODE_V1", "1");
        }
        close_binding_toolchain_descriptors(&mut command);
        let output = command.output().unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(error),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn literal_binding_host_test_executes_only_through_the_pinned_runner() {
    let workspace = fixture();
    let rustdoc_marker = workspace.0.join("hostile-rustdoc-invoked");
    let rustdoc = write_executable(
        &workspace.0,
        "hostile-rustdoc.sh",
        &format!(
            "#!/bin/sh\nprintf invoked > '{}'\nexit 98\n",
            rustdoc_marker.display()
        ),
    );
    write(
        &workspace.0,
        ".cargo/config.toml",
        &format!(
            "[env]\nRUSTDOC = {{ value = '{}', force = true }}\n",
            rustdoc.display()
        ),
    );
    let target = workspace.0.join("target-binding-host");
    let output = binding_test(&workspace)
        .env("CARGO_TARGET_DIR", target)
        .output()
        .expect("run literal binding-only host test");
    assert!(
        output.status.success(),
        "binding-only host test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("binding_is_present_and_runner_custody_is_closed"),
        "managed host unit test did not execute: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(!rustdoc_marker.exists(), "hostile rustdoc executed");
}

#[test]
fn binding_is_portable_across_absolute_workspace_roots() {
    let first = fixture();
    let relocated = fixture();
    let first_output = first.0.join("embedded-binding");
    let relocated_output = relocated.0.join("embedded-binding");

    for (workspace, output) in [(&first, &first_output), (&relocated, &relocated_output)] {
        let result = binding_test(workspace)
            .env("CARGO_TARGET_DIR", workspace.0.join("target-binding-host"))
            .env("BINDING_TEST_VALUE_OUTPUT", output)
            .output()
            .expect("run portable binding-only host test");
        assert!(
            result.status.success(),
            "binding-only host test failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    let first_binding = std::fs::read_to_string(first_output).expect("read first binding");
    let relocated_binding =
        std::fs::read_to_string(relocated_output).expect("read relocated binding");
    assert_eq!(first_binding.len(), 64);
    assert_eq!(first_binding, relocated_binding);
}

#[test]
fn binding_host_test_rejects_every_runner_and_unstable_config_channel() {
    let workspace = fixture();
    let host = host_target();
    let marker = workspace.0.join("hostile-runner-invoked");
    let runner = write_executable(
        &workspace.0,
        "hostile-runner.sh",
        &format!(
            "#!/bin/sh\nprintf invoked > '{}'\nexit 97\n",
            marker.display()
        ),
    );

    for args in [
        vec![
            "test",
            "--all-targets",
            "--config",
            "target.hostile.runner='/bin/false'",
            "-p",
            "managed",
        ],
        vec![
            "test",
            "--all-targets",
            "-Zconfig-include=hostile.toml",
            "-p",
            "managed",
        ],
    ] {
        let output = binding_command(&workspace)
            .args(args)
            .output()
            .expect("run caller config rejection");
        assert!(!output.status.success());
    }

    let ambient = binding_test(&workspace)
        .env(
            format!(
                "CARGO_TARGET_{}_RUNNER",
                host.to_ascii_uppercase().replace('-', "_")
            ),
            &runner,
        )
        .output()
        .expect("run ambient runner rejection");
    assert!(!ambient.status.success());
    assert!(
        String::from_utf8_lossy(&ambient.stderr).contains("ambient Cargo runner selection"),
        "{}",
        String::from_utf8_lossy(&ambient.stderr)
    );

    let rustdoc_marker = workspace.0.join("ambient-rustdoc-invoked");
    let rustdoc = write_executable(
        &workspace.0,
        "ambient-rustdoc.sh",
        &format!(
            "#!/bin/sh\nprintf invoked > '{}'\nexit 96\n",
            rustdoc_marker.display()
        ),
    );
    let ambient_rustdoc = binding_test(&workspace)
        .env("RUSTDOC", &rustdoc)
        .output()
        .expect("run ambient rustdoc rejection");
    assert!(!ambient_rustdoc.status.success());
    assert!(
        String::from_utf8_lossy(&ambient_rustdoc.stderr)
            .contains("ambient rustdoc selection RUSTDOC="),
        "{}",
        String::from_utf8_lossy(&ambient_rustdoc.stderr)
    );
    assert!(!rustdoc_marker.exists(), "ambient rustdoc executed");

    for (label, configuration, expected) in [
        (
            "cfg",
            format!("[target.'cfg(unix)']\nrunner = ['{}']\n", runner.display()),
            "configured target.cfg(unix).runner".to_owned(),
        ),
        (
            "exact",
            format!("[target.{host}]\nrunner = ['{}']\n", runner.display()),
            format!("configured target.{host}.runner"),
        ),
        (
            "environment",
            format!(
                "[env]\nCARGO_TARGET_{}_RUNNER = {{ value = '{}', force = true }}\n",
                host.to_ascii_uppercase().replace('-', "_"),
                runner.display()
            ),
            "configured runner environment".to_owned(),
        ),
        (
            "protected",
            "[env]\nFE2O3_TARGET = { value = 'hostile', force = true }\n".to_owned(),
            "configured protected environment env.FE2O3_TARGET".to_owned(),
        ),
    ] {
        write(&workspace.0, ".cargo/config.toml", &configuration);
        let output = binding_test(&workspace)
            .output()
            .unwrap_or_else(|error| panic!("run {label} config rejection: {error}"));
        assert!(!output.status.success(), "{label} config was accepted");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(&expected),
            "{label}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!marker.exists(), "{label} runner executed");
    }

    std::fs::remove_file(workspace.0.join(".cargo/config.toml"))
        .expect("remove workspace Cargo config");
    let cargo_home = workspace.0.join("hostile-cargo-home");
    write(
        &cargo_home,
        "config.toml",
        &format!("[target.{host}]\nrunner = ['{}']\n", runner.display()),
    );
    let global = binding_test(&workspace)
        .env("CARGO_HOME", cargo_home)
        .output()
        .expect("run global Cargo-home runner rejection");
    assert!(!global.status.success());
    assert!(
        String::from_utf8_lossy(&global.stderr)
            .contains(&format!("configured target.{host}.runner")),
        "{}",
        String::from_utf8_lossy(&global.stderr)
    );
    assert!(!marker.exists(), "global Cargo-home runner executed");
}

#[test]
fn binding_host_test_rejects_nonexecuting_cargo_modes_before_test_spawn() {
    let workspace = fixture();
    let host = host_target();
    for (label, args, expected) in [
        (
            "missing-all-targets",
            vec!["test", "-p", "managed"],
            "requires exact --all-targets",
        ),
        (
            "doc",
            vec!["test", "--all-targets", "--doc", "-p", "managed"],
            "does not admit rustdoc targets",
        ),
        (
            "no-run",
            vec!["test", "--all-targets", "--no-run", "-p", "managed"],
            "must execute the selected host tests",
        ),
        (
            "target",
            vec![
                "test",
                "--all-targets",
                "--target",
                host.as_str(),
                "-p",
                "managed",
            ],
            "--target is not admitted",
        ),
    ] {
        let marker = workspace.0.join(format!("{label}-test-executed"));
        let output = binding_command(&workspace)
            .args(args)
            .env("BINDING_TEST_EXECUTION_MARKER", &marker)
            .output()
            .unwrap_or_else(|error| panic!("run {label} rejection: {error}"));
        assert!(!output.status.success(), "{label} was accepted");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{label}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!marker.exists(), "{label} executed the host test");
    }
}

#[test]
fn binding_host_test_rejects_forced_loader_configuration_before_construction() {
    let workspace = fixture();
    let marker = workspace.0.join("hostile-constructor-invoked");
    write(
        &workspace.0,
        "hostile-constructor.c",
        &format!(
            "#include <stdio.h>\n__attribute__((constructor)) static void hostile(void) {{ FILE *f = fopen(\"{}\", \"w\"); if (f) {{ fputs(\"invoked\", f); fclose(f); }} }}\n",
            marker.display()
        ),
    );
    let shared_object = workspace.0.join("hostile-constructor.so");
    let compile = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(workspace.0.join("hostile-constructor.c"))
        .arg("-o")
        .arg(&shared_object)
        .status()
        .expect("compile hostile constructor fixture");
    assert!(compile.success(), "hostile constructor fixture failed");
    write(
        &workspace.0,
        ".cargo/config.toml",
        &format!(
            "[env]\nLD_PRELOAD = {{ value = '{}', force = true }}\n",
            shared_object.display()
        ),
    );
    let output = binding_test(&workspace)
        .output()
        .expect("run configured loader rejection");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("configured dynamic-loader environment env.LD_PRELOAD"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!marker.exists(), "configured loader constructor executed");
}

#[test]
fn binding_host_test_rejects_persistent_post_query_config_swap() {
    let workspace = fixture();
    let host = host_target();
    let host_environment = host.to_ascii_uppercase().replace('-', "_");
    let runner_marker = workspace.0.join("raced-runner-invoked");
    let rustdoc_marker = workspace.0.join("raced-rustdoc-invoked");
    let runner = write_executable(
        &workspace.0,
        "raced-runner.sh",
        &format!(
            "#!/bin/sh\nprintf invoked > '{}'\nexit 94\n",
            runner_marker.display()
        ),
    );
    let rustdoc = write_executable(
        &workspace.0,
        "raced-rustdoc.sh",
        &format!(
            "#!/bin/sh\nprintf invoked > '{}'\nexit 95\n",
            rustdoc_marker.display()
        ),
    );
    write(
        &workspace.0,
        "delegating-cargo.c",
        r#"#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int run_real_cargo(const char *real, int argc, char **argv) {
    char **child = calloc((size_t)argc + 1, sizeof(char *));
    if (child == NULL) return 125;
    child[0] = (char *)real;
    for (int i = 1; i < argc; ++i) child[i] = argv[i];
    pid_t pid = fork();
    if (pid == 0) {
        execvp(real, child);
        _exit(126);
    }
    if (pid < 0) return 125;
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return 125;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 125;
}

int main(int argc, char **argv) {
    const char *real = getenv("FE2O3_TEST_REAL_CARGO");
    if (real == NULL) return 125;
    const char *trace_path = getenv("FE2O3_TEST_CARGO_PROXY_TRACE");
    if (trace_path == NULL) return 125;
    FILE *trace = fopen(trace_path, "a");
    if (trace == NULL) return 125;
    for (int i = 1; i < argc; ++i) fprintf(trace, "%s%s", i == 1 ? "" : " ", argv[i]);
    fputc('\n', trace);
    fclose(trace);
    int status = run_real_cargo(real, argc, argv);
    if (argc >= 3 && strcmp(argv[1], "config") == 0 &&
        strcmp(argv[2], "get") == 0 && strcmp(argv[argc - 1], "env") == 0) {
        const char *count_path = getenv("FE2O3_TEST_CONFIG_QUERY_COUNT");
        const char *config_path = getenv("FE2O3_TEST_SWAP_CONFIG_PATH");
        const char *configuration = getenv("FE2O3_TEST_SWAP_CONFIG_BYTES");
        if (count_path == NULL || config_path == NULL || configuration == NULL) return 125;
        unsigned count = 0;
        FILE *count_in = fopen(count_path, "r");
        if (count_in != NULL) {
            if (fscanf(count_in, "%u", &count) != 1) return 125;
            fclose(count_in);
        }
        ++count;
        FILE *count_out = fopen(count_path, "w");
        if (count_out == NULL) return 125;
        fprintf(count_out, "%u\n", count);
        fclose(count_out);
        if (count == 2) {
            FILE *config = fopen(config_path, "w");
            if (config == NULL) return 125;
            if (fputs(configuration, config) == EOF || fclose(config) != 0) return 125;
        }
    }
    return status;
}
"#,
    );
    let proxy = workspace.0.join("delegating-cargo");
    let compile = Command::new("cc")
        .arg(workspace.0.join("delegating-cargo.c"))
        .arg("-o")
        .arg(&proxy)
        .status()
        .expect("compile delegating Cargo fixture");
    assert!(compile.success(), "delegating Cargo fixture failed");

    let configuration = format!(
        "[target.{host}]\nrunner = ['{}']\n[env]\nCARGO_TARGET_{host_environment}_RUNNER = {{ value = '{}', force = true }}\nLD_TRACE_LOADED_OBJECTS = {{ value = '1', force = true }}\nRUSTDOC = {{ value = '{}', force = true }}\n",
        runner.display(),
        runner.display(),
        rustdoc.display()
    );
    let count = workspace.0.join("config-env-query-count");
    let trace = workspace.0.join("delegating-cargo-trace");
    let config_path = workspace.0.join(".cargo/config.toml");
    std::fs::create_dir(workspace.0.join(".cargo")).expect("create raced Cargo config directory");
    let cargo_home = workspace.0.join("isolated-cargo-home");
    std::fs::create_dir(&cargo_home).expect("create isolated Cargo home");
    let target = workspace.0.join("target-raced-custody");
    let output = binding_test(&workspace)
        .env("CARGO", &proxy)
        .env("CARGO_HOME", cargo_home)
        .env("CARGO_TARGET_DIR", target)
        .env("FE2O3_TEST_REAL_CARGO", cargo())
        .env("FE2O3_TEST_CONFIG_QUERY_COUNT", &count)
        .env("FE2O3_TEST_CARGO_PROXY_TRACE", &trace)
        .env("FE2O3_TEST_SWAP_CONFIG_PATH", &config_path)
        .env("FE2O3_TEST_SWAP_CONFIG_BYTES", &configuration)
        .output()
        .expect("run post-query Cargo configuration swap");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(
            "Cargo test configuration revalidation also failed: cargo fe2o3 rejects configured dynamic-loader environment env.LD_TRACE_LOADED_OBJECTS"
        ),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        config_path.exists(),
        "delegating Cargo did not swap configuration; trace:\n{}",
        std::fs::read_to_string(&trace).unwrap_or_else(|error| format!("<unavailable: {error}>"))
    );
    assert!(
        std::fs::read_to_string(count)
            .expect("read config query count")
            .trim()
            .parse::<u32>()
            .expect("parse config query count")
            >= 2
    );
    assert!(!runner_marker.exists(), "raced runner executed");
    assert!(!rustdoc_marker.exists(), "raced rustdoc executed");
}

fn binding_selected_test(workspace: &TestWorkspace, selector: &[&str]) -> Command {
    let mut command = binding_command(workspace);
    command
        .args(["test", "--locked", "--manifest-path"])
        .arg(workspace.0.join("managed/Cargo.toml"))
        .args(selector);
    command
}

#[test]
fn binding_host_library_selection_keeps_custody_and_does_not_run_sibling_targets() {
    let workspace = fixture();
    write(
        &workspace.0,
        "managed/tests/sibling.rs",
        "#[test] fn sibling_must_fail_if_selected() { panic!(\"sibling was selected\"); }\n",
    );
    let output = binding_selected_test(&workspace, &["--lib"])
        .env("CARGO_TARGET_DIR", workspace.0.join("target-selected-lib"))
        .output()
        .expect("run exact host library suite");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("binding_is_present_and_runner_custody_is_closed")
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("sibling_must_fail_if_selected"));

    let all = binding_test(&workspace)
        .env("CARGO_TARGET_DIR", workspace.0.join("target-selected-lib"))
        .output()
        .expect("run unchanged all-target host scope");
    assert!(!all.status.success());
    assert!(String::from_utf8_lossy(&all.stdout).contains("sibling_must_fail_if_selected"));
}

#[test]
fn binding_host_named_integration_selection_uses_the_same_closed_runner() {
    let workspace = fixture();
    write(
        &workspace.0,
        "managed/tests/reference.rs",
        r#"#[test]
fn exact_reference_target_has_compiler_binding() {
    assert!(option_env!("FE2O3_CRATE_BINDING_ID_V1").is_some());
    for (name, _) in std::env::vars_os() {
        assert!(!name.to_string_lossy().starts_with("FE2O3_"));
    }
}
"#,
    );
    write(
        &workspace.0,
        "managed/tests/sibling.rs",
        "#[test] fn sibling_must_not_run() { panic!(\"unexpected sibling\"); }\n",
    );
    let output = binding_selected_test(&workspace, &["--test", "reference"])
        .env(
            "CARGO_TARGET_DIR",
            workspace.0.join("target-selected-reference"),
        )
        .output()
        .expect("run exact host integration suite");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("exact_reference_target_has_compiler_binding"));
    assert!(!stdout.contains("sibling_must_not_run"));
    assert!(!stdout.contains("binding_is_present_and_runner_custody_is_closed"));

    let missing = binding_selected_test(&workspace, &["--test", "missing"])
        .env(
            "CARGO_TARGET_DIR",
            workspace.0.join("target-selected-reference"),
        )
        .output()
        .expect("refuse missing integration target");
    assert!(!missing.status.success());
}

#[test]
fn binding_host_selected_scopes_keep_config_and_selector_refusals() {
    let workspace = fixture();
    for selector in [
        vec!["--lib", "--config=target.host.runner='hostile'"],
        vec!["--lib", "--test", "reference"],
        vec!["--test", "*"],
        vec!["--lib", "--no-run"],
        vec!["--test", "reference", "--doc"],
        vec!["--lib", "-Zunstable-options"],
    ] {
        let output = binding_selected_test(&workspace, &selector)
            .env(
                "CARGO_TARGET_DIR",
                workspace.0.join("target-selected-refusal"),
            )
            .output()
            .expect("refuse malformed selected host scope");
        assert!(!output.status.success(), "{selector:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("binding-only"));
    }
}

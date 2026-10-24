mod application_exec;
mod application_handoff;
mod application_sandbox;
mod application_supervisor;
mod authority_release;
mod authorized_kernel_closure;
mod binding_check_projection;
mod binding_check_wrapper;
mod binding_wrapper;
mod build_config;
mod capability_broker;
mod cargo_binding_trampoline;
mod cargo_invocation_boundary;
mod clean;
mod compiler_execution_boundary;
mod compiler_toolchain;
mod doctor;
mod engineering_hsaco;
mod engineering_rustc_runtime;
mod example_manifest;
mod generation;
mod inert_rustc_invocation_capture;
mod inspect;
mod native_policy_export;
mod non_production_reproduction;
mod observer_telemetry;
#[allow(dead_code)]
#[path = "rustc_wrapper/pinned_codegen_backend.rs"]
mod pinned_codegen_backend;
#[path = "rustc_wrapper/pinned_executable.rs"]
mod pinned_executable;
#[cfg(test)]
mod pinned_executable_test_directory;
mod process_execution;
mod production_cargo_plan;
mod production_census_v91;
mod production_graph_capture_v92;
mod production_host_binding;
mod profile_command;
mod profile_dispatch_import_v1;
mod profile_live_qualification_v1;
mod profile_wrapper_overhead_v1;
mod project;
mod protected_compiler_handoff_v3;
#[path = "rustc_runtime.rs"]
mod rustc_lib_tree;
mod source_isa_observation;
mod tool_commands;

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::{AsRawFd, BorrowedFd, IntoRawFd};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};

const TARGET_ENV: &str = "FE2O3_TARGET";
const BACKEND_ENV: &str = "FE2O3_BACKEND";
const HSACO_DIR_ENV: &str = "FE2O3_HSACO_DIR";
const OBSOLETE_CODEGEN_PIPELINE_ENV: &str = "FE2O3_CODEGEN_PIPELINE";
const BINDING_WRAPPER_MODE_ENV: &str = "FE2O3_BINDING_WRAPPER_MODE_V1";
const MANAGED_RUSTC_ARGS_ENV: &str = "FE2O3_MANAGED_RUSTC_ARGS_V1";
const BUILD_SESSION_ENV: &str = "FE2O3_BUILD_SESSION_V1";
const OBSOLETE_SIMULATION_MODE_ENV: &str = "FE2O3_SIMULATION_MODE_V1";
const OBSOLETE_SIMULATION_ATTEMPT_ENV: &str = "FE2O3_SIMULATION_ATTEMPT_V1";
const EXPECTED_RUSTC_SHA256_ENV: &str = "FE2O3_EXPECTED_RUSTC_SHA256_V1";
const EXPECTED_COMPILER_CLOSURE_SHA256_ENV: &str = "FE2O3_EXPECTED_COMPILER_CLOSURE_SHA256_V1";
const CARGO_PRIMARY_PACKAGE_ENV: &str = "CARGO_PRIMARY_PACKAGE";
const AUTHORITY_CARGO_SHA256_ENV: &str = "FE2O3_AUTHORITY_CARGO_SHA256_V1";
const AUTHORITY_RUSTC_SHA256_ENV: &str = "FE2O3_AUTHORITY_RUSTC_SHA256_V1";
const AUTHORITY_RUSTC_PATH_ENV: &str = "FE2O3_AUTHORITY_RUSTC_PATH_V1";
const AUTHORITY_RUSTC_RUNTIME_SHA256_ENV: &str = "FE2O3_AUTHORITY_RUSTC_RUNTIME_SHA256_V1";
const AUTHORITY_BACKEND_SHA256_ENV: &str = "FE2O3_AUTHORITY_BACKEND_SHA256_V1";
const AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_ENV: &str =
    "FE2O3_AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_V1";
const AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_ENV: &str =
    "FE2O3_AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_V1";
const SOURCE_ISA_COLLECTION_STDERR_PREFIX_V1: &str =
    "[cargo-fe2o3] source-isa-observation-collection-v1";
const SOURCE_ISA_CHARACTERISTIC_BROKER_STDERR_PREFIX_V3: &str =
    "[cargo-fe2o3] source-isa-characteristic-broker-v3";
const MAX_SOURCE_ISA_COLLECTION_STDERR_LINE_BYTES_V1: usize =
    fe2o3_source_isa_observation::wire_v1::MAX_SOURCE_ISA_OBSERVATION_COLLECTION_HEX_BYTES_V1 + 256;
const _: () = assert!(MAX_SOURCE_ISA_COLLECTION_STDERR_LINE_BYTES_V1 <= 2 * 1024 * 1024);
const NON_PRODUCTION_AUTHORITY_VALIDATION_ENV: &str =
    "FE2O3_NON_PRODUCTION_UNPROTECTED_AUTHORITY_VALIDATION_V1";
const INTERNAL_RUNNER_ARG: &str = "__fe2o3-runner-v1";
const BINDING_HOST_TEST_RUNNER_ARG: &str = "__fe2o3-binding-host-test-runner-v1";
const BINDING_HOST_DISABLED_RUSTDOC: &str = "/__fe2o3_binding_host_rustdoc_disabled__";
const CARGO_BINDING_WRAPPER_CHILD_FD: std::os::fd::RawFd = 191;
const CARGO_BINDING_TRAMPOLINE_CHILD_FD: std::os::fd::RawFd = 192;
const BACKEND_BUILD_CHILD_FD: std::os::fd::RawFd = 196;
const CARGO_BINDING_CHECK_WRAPPER_CHILD_FD: std::os::fd::RawFd = 200;
const CARGO_BINDING_CHECK_PROJECTION_CHILD_FD: std::os::fd::RawFd = 201;
const CLIPPY_DRIVER_CHILD_FD: std::os::fd::RawFd = 202;
const RUSTC_LIBRARY_CHILD_FD: std::os::fd::RawFd = 193;
const RUSTC_CHILD_FD: std::os::fd::RawFd = 194;
const RUSTC_INVOCATION_CHILD_FD: std::os::fd::RawFd =
    fe2o3_compiler_closure_capability::RUSTC_INVOCATION_CHILD_FD_V1;
const ARTIFACT_CHILD_FD: std::os::fd::RawFd =
    fe2o3_artifact_transaction::BROKERED_ARTIFACT_DIRECTORY_CHILD_FD_V1;
const BACKEND_CHILD_FD: std::os::fd::RawFd =
    fe2o3_artifact_transaction::BROKERED_CODEGEN_BACKEND_CHILD_FD_V1;
const _: () = assert!(RUSTC_INVOCATION_CHILD_FD != RUSTC_LIBRARY_CHILD_FD);
const _: () = assert!(RUSTC_INVOCATION_CHILD_FD != RUSTC_CHILD_FD);
const _: () = assert!(
    RUSTC_INVOCATION_CHILD_FD
        != fe2o3_artifact_transaction::BROKERED_INVOCATION_AUTHORITY_CHILD_FD_V1
);
const _: () = assert!(RUSTC_INVOCATION_CHILD_FD != ARTIFACT_CHILD_FD);
const _: () = assert!(RUSTC_INVOCATION_CHILD_FD != BACKEND_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD != RUSTC_LIBRARY_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD != RUSTC_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD != RUSTC_INVOCATION_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD != ARTIFACT_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD != BACKEND_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != RUSTC_LIBRARY_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != RUSTC_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != CARGO_BINDING_WRAPPER_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != CARGO_BINDING_TRAMPOLINE_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != BACKEND_BUILD_CHILD_FD);
const _: () =
    assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != CARGO_BINDING_CHECK_WRAPPER_CHILD_FD);
const _: () = assert!(CLIPPY_DRIVER_CHILD_FD != CARGO_BINDING_CHECK_WRAPPER_CHILD_FD);
const _: () = assert!(CLIPPY_DRIVER_CHILD_FD != CARGO_BINDING_CHECK_PROJECTION_CHILD_FD);
const _: () = assert!(CLIPPY_DRIVER_CHILD_FD != RUSTC_LIBRARY_CHILD_FD);
const _: () = assert!(CLIPPY_DRIVER_CHILD_FD != RUSTC_CHILD_FD);
const _: () = assert!(
    CARGO_BINDING_CHECK_PROJECTION_CHILD_FD
        != fe2o3_artifact_transaction::BROKERED_INVOCATION_AUTHORITY_CHILD_FD_V1
);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != RUSTC_INVOCATION_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != ARTIFACT_CHILD_FD);
const _: () = assert!(CARGO_BINDING_CHECK_PROJECTION_CHILD_FD != BACKEND_CHILD_FD);

const COMPILER_SELECTION_ENVIRONMENT: &[&str] = &[
    "RUSTC",
    "CARGO_BUILD_RUSTC",
    "RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
];

fn main() -> ExitCode {
    let raw_args = env::args_os().skip(1).collect::<Vec<_>>();
    if raw_args
        .first()
        .is_some_and(|argument| argument == authority_release::INTERNAL_CHILD_ARG)
    {
        return authority_release::run_child(&raw_args[1..]);
    }
    if raw_args
        .first()
        .is_some_and(|argument| argument == authority_release::INTERNAL_CHILD_ARG_V4)
    {
        return authority_release::run_child_v4(&raw_args[1..]);
    }
    if raw_args
        .first()
        .is_some_and(|argument| argument == application_supervisor::INTERNAL_SUPERVISOR_ARG)
    {
        return run_application_supervisor(&raw_args[1..]);
    }
    if raw_args
        .first()
        .is_some_and(|argument| argument == INTERNAL_RUNNER_ARG)
    {
        return run_application_boundary_frontend(&raw_args[1..]);
    }
    if raw_args
        .first()
        .is_some_and(|argument| argument == BINDING_HOST_TEST_RUNNER_ARG)
    {
        return binding_host_test_runner(&raw_args[1..]);
    }
    if env::var_os(production_host_binding::MODE_ENV).is_some() {
        return match production_host_binding::run(raw_args) {
            Ok(status) => ExitCode::from(binding_check_wrapper::exit_code(status)),
            Err(error) => {
                eprintln!("cargo-fe2o3 production host binding: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if env::var_os(binding_check_wrapper::MODE_ENV_V1).is_some() {
        return match binding_check_wrapper::run(raw_args) {
            Ok(status) => ExitCode::from(binding_check_wrapper::exit_code(status)),
            Err(error) => {
                eprintln!("cargo-fe2o3 binding-check wrapper: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if env::var_os(BINDING_WRAPPER_MODE_ENV).is_some() {
        return match binding_wrapper::run(raw_args) {
            Ok(status) => ExitCode::from(binding_wrapper::exit_code(status)),
            Err(error) => {
                eprintln!("cargo-fe2o3 binding wrapper: {error}");
                ExitCode::FAILURE
            }
        };
    }

    let mut invocation = normalize_invocation(raw_args);
    let mut args = invocation.drain(..);
    let command = args.next().unwrap_or_else(|| OsString::from("help"));
    let rest: Vec<OsString> = args.collect();

    match command.to_str() {
        Some("authority") => authority_release::command(&rest),
        Some("doctor") => with_utf8_args(&rest, doctor::command),
        Some("engineering") => match rest.as_slice() {
            [subcommand, args @ ..] if subcommand == "hsaco" => engineering_hsaco::command(args),
            [subcommand, args @ ..] if subcommand == "rustc-runtime" => {
                engineering_rustc_runtime::command(args)
            }
            _ => {
                eprintln!("cargo fe2o3 engineering requires `hsaco` or `rustc-runtime`");
                ExitCode::FAILURE
            }
        },
        Some("check") => binding_host_command(BindingHostMode::Check, &rest),
        Some("clippy") => binding_host_command(BindingHostMode::Clippy, &rest),
        Some("test") => binding_host_command(BindingHostMode::Test, &rest),
        Some("build") => cargo_with_backend("build", &rest),
        Some("run") => cargo_with_backend("run", &rest),
        Some("examples") => with_utf8_args(&rest, example_manifest::command),
        Some("clean") => clean_command(&rest),
        Some("inspect") => with_utf8_args(&rest, inspect::command),
        Some("sanitize") => with_utf8_args(&rest, |args| {
            tool_report(tool_commands::command(tool_commands::Mode::Sanitize, args))
        }),
        Some("debug") => with_utf8_args(&rest, |args| {
            tool_report(tool_commands::command(tool_commands::Mode::Debug, args))
        }),
        Some("profile") => {
            with_utf8_args(&rest, |args| profile_report(profile_command::command(args)))
        }
        Some("help" | "--help" | "-h") => {
            print_help();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("unknown cargo-fe2o3 command {command:?}");
            print_help();
            ExitCode::FAILURE
        }
    }
}

fn with_utf8_args(args: &[OsString], command: impl FnOnce(&[String]) -> ExitCode) -> ExitCode {
    let args = match args
        .iter()
        .map(|arg| {
            arg.to_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("command argument is not valid UTF-8: {arg:?}"))
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    command(&args)
}

fn tool_report(result: Result<tool_commands::CommandReport, String>) -> ExitCode {
    match result {
        Ok(report) => {
            println!("{}", report.output());
            if report.succeeded() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn profile_report(result: Result<profile_command::CommandReport, String>) -> ExitCode {
    match result {
        Ok(report) => {
            println!("{}", report.output());
            if report.succeeded() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn normalize_invocation(mut args: Vec<OsString>) -> Vec<OsString> {
    if args.first().is_some_and(|arg| arg == OsStr::new("fe2o3")) {
        args.remove(0);
    }
    args
}

fn clean_command(args: &[OsString]) -> ExitCode {
    let options = match clean::parse_project_options(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let project = match project::CargoProject::discover(args, None, None, false) {
        Ok(project) => project,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let plan = match clean::plan_project(project) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    match clean::execute_project(plan, options) {
        Ok(actions) => {
            for action in actions {
                eprintln!("{}", action.diagnostic());
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

/// Host checks, lints, and tests compile trusted workspace code without artifact or GPU authority.
/// The test runner fixes tool selection and child custody; it is not a sandbox for hostile test
/// code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BindingHostMode {
    Check,
    Clippy,
    Test,
}

impl BindingHostMode {
    const fn cargo_command(self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Clippy => "clippy",
            Self::Test => "test",
        }
    }

    const fn executes_tests(self) -> bool {
        matches!(self, Self::Test)
    }
}

fn binding_host_command(mode: BindingHostMode, args: &[OsString]) -> ExitCode {
    match binding_host_result(mode, args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cargo fe2o3 {}: {error}", mode.cargo_command());
            ExitCode::FAILURE
        }
    }
}

fn binding_host_result(mode: BindingHostMode, args: &[OsString]) -> Result<(), String> {
    binding_check_wrapper::reject_prohibited_environment().map_err(|error| error.to_string())?;
    if env::var_os(binding_check_wrapper::CLIPPY_DRIVER_ENV_V1).is_some()
        || env::var_os("CLIPPY_ARGS").is_some()
    {
        return Err("binding-only host commands reject inherited Clippy configuration".to_owned());
    }
    scrub_process_dynamic_loader_environment();
    reject_preexisting_compiler_environment()?;
    if mode.executes_tests() {
        reject_binding_test_invocation_config(args)?;
        reject_ambient_cargo_test_runners()?;
    }
    if selected_run_target(args)?.is_some() {
        return Err(format!(
            "binding-only host {} selects the pinned rustc host target; --target is not admitted",
            mode.cargo_command()
        ));
    }

    let invocation_directory = env::current_dir()
        .map_err(|error| format!("failed to resolve Cargo invocation directory: {error}"))?;
    let cargo_declaration = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let pinned_cargo = pin_default_cargo(&cargo_declaration, &invocation_directory)?;
    let project = project::CargoProject::discover(args, Some(&pinned_cargo), None, false)?;
    reject_configured_compiler_selection(&project, args, &pinned_cargo, None, false)?;
    let pinned_rustc = pin_default_rustc(&project)?;
    let clippy_driver = if mode == BindingHostMode::Clippy {
        Some(pin_default_clippy_driver(
            project.invocation_dir().child_path().as_path(),
        )?)
    } else {
        None
    };
    let host_target = pinned_rustc_host_target(&pinned_rustc)?;
    if mode.executes_tests() {
        reject_configured_cargo_test_runners(&project, args, &pinned_cargo)?;
    }
    let projection = example_manifest::pinned_workspace_binding_projection(
        project.workspace_root().display_path(),
        &pinned_cargo,
    )?;
    let sealed_projection = binding_check_projection::SealedProjection::new(&projection)?;

    let wrapper_path = env::current_exe()
        .map_err(|error| format!("failed to locate cargo-fe2o3 executable: {error}"))?;
    let wrapper_source = pinned_executable::PinnedExecutable::open(&wrapper_path)
        .map_err(|error| format!("failed to pin the binding-check wrapper: {error}"))?;
    let wrapper = wrapper_source
        .seal_executable_image()
        .map_err(|error| format!("failed to seal the binding-check wrapper: {error}"))?;
    let workspace_wrapper = wrapper
        .fixed_child_path(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD)
        .map_err(|error| format!("failed to retain the binding-check wrapper: {error}"))?;

    project.validate_paths()?;
    let mut cargo = pinned_cargo
        .command()
        .map_err(|error| format!("failed to prepare pinned Cargo executable: {error}"))?;
    wrapper
        .inherit_for_child_at(cargo.as_command_mut(), CARGO_BINDING_CHECK_WRAPPER_CHILD_FD)
        .map_err(|error| format!("failed to inherit the binding-check wrapper: {error}"))?;
    sealed_projection.inherit_for_child_at(
        cargo.as_command_mut(),
        CARGO_BINDING_CHECK_PROJECTION_CHILD_FD,
    )?;
    if let Some(driver) = &clippy_driver {
        driver
            .inherit_for_child_at(cargo.as_command_mut(), CLIPPY_DRIVER_CHILD_FD)
            .map_err(|error| format!("failed to inherit the pinned Clippy driver: {error}"))?;
    }
    let (cargo_command, mut forwarded_args, clippy_args) = if mode == BindingHostMode::Clippy {
        let (forwarded, clippy_args) = prepare_binding_clippy_args(args)?;
        ("check", forwarded, Some(clippy_args))
    } else {
        (mode.cargo_command(), args.to_vec(), None)
    };
    let separator = forwarded_args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(forwarded_args.len());
    forwarded_args.splice(
        separator..separator,
        [
            OsString::from("--target"),
            OsString::from(host_target.as_str()),
        ],
    );
    if mode.executes_tests() {
        inject_binding_host_test_custody(&mut forwarded_args, &host_target, &workspace_wrapper)?;
    }
    binding_check_wrapper::clear_prohibited_environment(cargo.as_command_mut());
    clear_inherited_cargo_unit_identity(cargo.as_command_mut());
    cargo
        .as_command_mut()
        .arg(cargo_command)
        .args(&forwarded_args)
        .current_dir(project.invocation_dir().child_path())
        .env("RUSTC_WRAPPER", "")
        .env("CARGO_BUILD_RUSTC_WRAPPER", "")
        .env("RUSTC_WORKSPACE_WRAPPER", workspace_wrapper)
        .env(
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            format!("/proc/self/fd/{CARGO_BINDING_CHECK_WRAPPER_CHILD_FD}"),
        )
        .env("RUSTDOC", BINDING_HOST_DISABLED_RUSTDOC)
        .env("CARGO_BUILD_RUSTDOC", BINDING_HOST_DISABLED_RUSTDOC)
        .env_remove(CARGO_PRIMARY_PACKAGE_ENV)
        .env_remove("CARGO_PKG_NAME")
        .env_remove("CARGO_MANIFEST_DIR")
        .env(binding_check_wrapper::MODE_ENV_V1, "1")
        .env("FE2O3_HIP_SYS_DISABLE", "1");
    if let Some(clippy_args) = clippy_args {
        cargo
            .as_command_mut()
            .env("CLIPPY_ARGS", clippy_args)
            .env(binding_check_wrapper::CLIPPY_DRIVER_ENV_V1, "1");
    } else {
        cargo
            .as_command_mut()
            .env_remove("CLIPPY_ARGS")
            .env_remove(binding_check_wrapper::CLIPPY_DRIVER_ENV_V1);
    }
    remove_dynamic_loader_environment(cargo.as_command_mut());
    configure_pinned_rustc_child(cargo.as_command_mut(), &pinned_rustc)?;
    cargo.as_command_mut().env(
        "LD_LIBRARY_PATH",
        format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"),
    );

    let status = cargo
        .status()
        .map_err(|error| format!("failed to run pinned Cargo: {error}"))?;
    // These before/after scans reject a persistent protected configuration change. They are
    // deliberately not described as a TOCTOU-proof snapshot: Cargo configuration and test code
    // are trusted on this authority-free path, while the fixed runner still closes its own child
    // boundary.
    let post_test_configuration = if mode.executes_tests() {
        aggregate_post_spawn_results(
            reject_configured_compiler_selection(&project, args, &pinned_cargo, None, false),
            [(
                "Cargo test runner configuration revalidation",
                reject_configured_cargo_test_runners(&project, args, &pinned_cargo),
            )],
        )
    } else {
        Ok(())
    };
    aggregate_post_spawn_results(
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "pinned Cargo {} failed with status {status}",
                mode.cargo_command()
            ))
        },
        [
            ("Cargo project path revalidation", project.validate_paths()),
            (
                "rustc toolchain lib-tree revalidation",
                pinned_rustc.revalidate_lib_tree(),
            ),
            (
                "Cargo test configuration revalidation",
                post_test_configuration,
            ),
        ],
    )
}

fn prepare_binding_clippy_args(args: &[OsString]) -> Result<(Vec<OsString>, OsString), String> {
    let separator = args.iter().position(|argument| argument == "--");
    let cargo_end = separator.unwrap_or(args.len());
    if args[..cargo_end].iter().any(|argument| argument == "--fix") {
        return Err(
            "binding-aware Clippy does not support --fix because its source projection is immutable"
                .to_owned(),
        );
    }

    let mut forwarded = Vec::with_capacity(cargo_end);
    let mut clippy = Vec::new();
    for argument in &args[..cargo_end] {
        if argument == "--no-deps" {
            clippy.push(argument.clone());
        } else {
            forwarded.push(argument.clone());
        }
    }
    if let Some(separator) = separator {
        clippy.extend_from_slice(&args[separator + 1..]);
    }

    let mut encoded = OsString::new();
    for argument in clippy {
        let argument = argument
            .to_str()
            .ok_or_else(|| "Clippy arguments must be UTF-8".to_owned())?;
        if argument.contains("__CLIPPY_HACKERY__") {
            return Err("Clippy argument contains its reserved separator".to_owned());
        }
        encoded.push(argument);
        encoded.push("__CLIPPY_HACKERY__");
    }
    Ok((forwarded, encoded))
}

fn reject_binding_test_invocation_config(args: &[OsString]) -> Result<(), String> {
    let cargo_args = args
        .iter()
        .take_while(|argument| *argument != "--")
        .collect::<Vec<_>>();
    if cargo_args
        .iter()
        .any(|argument| **argument == "--config" || os_bytes(argument).starts_with(b"--config="))
    {
        return Err(
            "binding-only host test rejects caller-supplied --config before execution".to_owned(),
        );
    }
    if cargo_args
        .iter()
        .any(|argument| os_bytes(argument).starts_with(b"-Z"))
    {
        return Err(
            "binding-only host test rejects every caller-supplied Cargo -Z option before execution"
                .to_owned(),
        );
    }
    if cargo_args.iter().any(|argument| **argument == "--doc") {
        return Err("binding-only host test does not admit rustdoc targets".to_owned());
    }
    if cargo_args.iter().any(|argument| **argument == "--no-run") {
        return Err("binding-only host test must execute the selected host tests".to_owned());
    }
    if !cargo_args
        .iter()
        .any(|argument| **argument == "--all-targets")
    {
        reject_binding_selected_test_shape(&cargo_args)?;
    }
    Ok(())
}

fn reject_binding_selected_test_shape(args: &[&OsString]) -> Result<(), String> {
    const VALUE_OPTIONS: &[&str] = &[
        "--manifest-path",
        "--package",
        "-p",
        "--target-dir",
        "--message-format",
        "--jobs",
        "-j",
        "--features",
        "-F",
        "--profile",
        "--exclude",
        "--color",
        "--target",
    ];
    let mut selected = false;
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        let text = argument.to_str().ok_or_else(|| {
            "binding-only selected host test requires UTF-8 option names".to_owned()
        })?;
        match text {
            "--lib" | "--test" => {
                if selected {
                    return Err(
                        "binding-only selected host test rejects mixed or duplicate selectors"
                            .to_owned(),
                    );
                }
                selected = true;
                if text == "--test" {
                    index += 1;
                    let name = args
                        .get(index)
                        .and_then(|name| name.to_str())
                        .ok_or_else(|| {
                            "binding-only selected host test requires one literal test target"
                                .to_owned()
                        })?;
                    if name.is_empty()
                        || name.starts_with('-')
                        || !name
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                    {
                        return Err(
                            "binding-only selected host test requires one literal test target"
                                .to_owned(),
                        );
                    }
                }
            }
            "--locked"
            | "--offline"
            | "--frozen"
            | "--quiet"
            | "-q"
            | "--verbose"
            | "-v"
            | "--release"
            | "--no-default-features"
            | "--all-features" => {}
            value_option if VALUE_OPTIONS.contains(&value_option) => {
                index += 1;
                let value = args.get(index).ok_or_else(|| {
                    "binding-only selected host test option is missing its value".to_owned()
                })?;
                if value.is_empty() || os_bytes(value).starts_with(b"-") {
                    return Err(
                        "binding-only selected host test option has an invalid value".to_owned(),
                    );
                }
            }
            joined
                if joined.split_once('=').is_some_and(|(key, value)| {
                    key.starts_with("--")
                        && VALUE_OPTIONS.contains(&key)
                        && !value.is_empty()
                        && !value.starts_with('-')
                }) => {}
            _ => {
                return Err(
                    "binding-only selected host test rejects unsupported Cargo arguments"
                        .to_owned(),
                );
            }
        }
        index += 1;
    }
    if !selected {
        return Err(
            "binding-only host test requires exact --all-targets so rustdoc is never selected"
                .to_owned(),
        );
    }
    Ok(())
}

fn inject_binding_host_test_custody(
    args: &mut Vec<OsString>,
    host_target: &str,
    workspace_wrapper: &Path,
) -> Result<(), String> {
    // Config preflight is diagnostic, not an atomic snapshot. These highest-precedence entries
    // stabilize the core compiler, runner, rustdoc, and executable loader-selection channels;
    // workspace config/source, linker, network, build scripts, and tests remain trusted.
    let wrapper = workspace_wrapper.to_str().ok_or_else(|| {
        "binding-only host test requires a UTF-8 sealed wrapper descriptor path".to_owned()
    })?;
    let rustc = format!("/proc/self/fd/{RUSTC_CHILD_FD}");
    let rustc_lib = format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}");
    let host_target = binding_host_target_key(host_target)?;
    let runner = serde_json::to_string(&[wrapper, BINDING_HOST_TEST_RUNNER_ARG])
        .map_err(|error| format!("failed to encode the pinned host-test runner: {error}"))?;

    let mut configs = vec![
        format!("build.rustc={}", cargo_config_string(&rustc)?),
        format!(
            "build.rustc-workspace-wrapper={}",
            cargo_config_string(wrapper)?
        ),
        format!("target.{host_target}.runner={runner}"),
    ];
    for (name, value) in [
        ("RUSTC", rustc.as_str()),
        ("CARGO_BUILD_RUSTC", rustc.as_str()),
        ("RUSTC_WRAPPER", ""),
        ("CARGO_BUILD_RUSTC_WRAPPER", ""),
        ("RUSTC_WORKSPACE_WRAPPER", wrapper),
        ("CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", wrapper),
        ("RUSTDOC", BINDING_HOST_DISABLED_RUSTDOC),
        ("CARGO_BUILD_RUSTDOC", BINDING_HOST_DISABLED_RUSTDOC),
        ("LD_PRELOAD", ""),
        ("LD_AUDIT", ""),
        ("GLIBC_TUNABLES", ""),
        ("LD_LIBRARY_PATH", rustc_lib.as_str()),
    ] {
        configs.extend(forced_cargo_environment(name, value)?);
    }
    for config in configs {
        let insert_at = args
            .iter()
            .position(|argument| argument == "--")
            .unwrap_or(args.len());
        args.insert(insert_at, OsString::from("--config"));
        args.insert(insert_at + 1, OsString::from(config));
    }
    Ok(())
}

fn binding_host_target_key(host_target: &str) -> Result<String, String> {
    if host_target.is_empty()
        || host_target.len() > 128
        || !host_target
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(format!(
            "pinned rustc reported an unsupported host target {host_target:?}"
        ));
    }
    cargo_config_string(host_target)
}

fn cargo_config_string(value: &str) -> Result<String, String> {
    serde_json::to_string(value)
        .map_err(|error| format!("failed to encode trusted Cargo configuration: {error}"))
}

fn forced_cargo_environment(name: &str, value: &str) -> Result<[String; 2], String> {
    Ok([
        format!("env.{name}.value={}", cargo_config_string(value)?),
        format!("env.{name}.force=true"),
    ])
}

fn binding_host_test_runner(args: &[OsString]) -> ExitCode {
    match binding_host_test_runner_result(args) {
        Ok(status) => ExitCode::from(binding_check_wrapper::exit_code(status)),
        Err(error) => {
            eprintln!("cargo-fe2o3 pinned host-test runner: {error}");
            ExitCode::FAILURE
        }
    }
}

fn binding_host_test_runner_result(args: &[OsString]) -> Result<std::process::ExitStatus, String> {
    if env::var_os(binding_check_wrapper::MODE_ENV_V1).as_deref() != Some(OsStr::new("1")) {
        return Err("missing exact binding-only wrapper custody marker".to_owned());
    }
    let (executable, test_args) = args
        .split_first()
        .ok_or_else(|| "Cargo supplied no host-test executable".to_owned())?;
    if executable.is_empty() {
        return Err("Cargo supplied an empty host-test executable".to_owned());
    }
    let source = pinned_executable::PinnedExecutable::open_binding_host_test(Path::new(executable))
        .map_err(|error| format!("failed to pin the Cargo host-test executable: {error}"))?;
    // The retained, hashed original preserves Cargo's current_exe/$ORIGIN behavior. This trusted,
    // non-authoritative test path makes no immutable-publication claim against a same-inode writer.
    let mut command = source
        .command()
        .map_err(|error| format!("failed to prepare the pinned host-test executable: {error}"))?;
    command.args(test_args);
    for (name, _) in env::vars_os() {
        if os_bytes(&name).starts_with(b"FE2O3_")
            || is_cargo_target_runner_environment_name(&name)
            || COMPILER_SELECTION_ENVIRONMENT
                .iter()
                .any(|candidate| name == *candidate)
            || (is_dynamic_loader_environment_name(&name) && name != "LD_LIBRARY_PATH")
            || matches!(
                name.to_str(),
                Some(
                    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"
                        | "CARGO_BUILD_RUSTDOC"
                        | "RUSTDOC"
                        | "RUSTDOCFLAGS"
                        | "RUSTFLAGS"
                        | "CARGO_ENCODED_RUSTFLAGS"
                )
            )
        {
            command.as_command_mut().env_remove(name);
        }
    }
    // Cargo may add target-directory dylib paths to the CLI-forced rustc library path. Retaining
    // that forced/augmented path and the two pinned compiler descriptors keeps nested Cargo tools
    // such as trybuild on the same compiler as their parent test. Test code is trusted here and
    // receives no artifact or GPU authority.
    configure_binding_host_test_toolchain(command.as_command_mut())?;
    let status = command
        .status()
        .map_err(|error| format!("failed to execute the pinned host test: {error}"))?;
    drop(command);
    source
        .command()
        .map_err(|error| format!("host-test executable changed across execution: {error}"))?;
    Ok(status)
}

fn configure_binding_host_test_toolchain(command: &mut Command) -> Result<(), String> {
    let descriptors = BindingHostTestToolchainDescriptors::observe()?;
    let rustc = format!("/proc/self/fd/{RUSTC_CHILD_FD}");
    command.env("RUSTC", &rustc).env("CARGO_BUILD_RUSTC", rustc);
    // SAFETY: this single callback closes the ambient descriptor set, revalidates the two
    // parent-inherited compiler objects, and exposes only those objects to trusted test code.
    unsafe {
        command.pre_exec(move || {
            application_exec::protect_all_nonstdio_descriptors()?;
            descriptors.revalidate()?;
            application_exec::expose_descriptor(RUSTC_LIBRARY_CHILD_FD)?;
            application_exec::expose_descriptor(RUSTC_CHILD_FD)?;
            Ok(())
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct BindingHostTestToolchainDescriptors {
    library_device: u64,
    library_inode: u64,
    library_mode: u32,
    rustc_device: u64,
    rustc_inode: u64,
    rustc_mode: u32,
    rustc_size: i64,
}

impl BindingHostTestToolchainDescriptors {
    const REQUIRED_RUSTC_SEALS: i32 =
        libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

    fn observe() -> Result<Self, String> {
        let library_stat = fixed_descriptor_stat(RUSTC_LIBRARY_CHILD_FD).map_err(|error| {
            format!(
                "binding-only host-test runner cannot inspect pinned rustc library descriptor {RUSTC_LIBRARY_CHILD_FD}: {error}"
            )
        })?;
        let library_status = fixed_descriptor_fcntl(RUSTC_LIBRARY_CHILD_FD, libc::F_GETFL)
            .map_err(|error| {
            format!(
                "binding-only host-test runner cannot inspect pinned rustc library access: {error}"
            )
        })?;
        if library_stat.st_mode & libc::S_IFMT != libc::S_IFDIR
            || library_status & libc::O_ACCMODE != libc::O_RDONLY
        {
            return Err(
                "binding-only host-test runner requires a read-only pinned rustc library directory"
                    .to_owned(),
            );
        }

        let rustc_stat = fixed_descriptor_stat(RUSTC_CHILD_FD).map_err(|error| {
            format!(
                "binding-only host-test runner cannot inspect pinned rustc descriptor {RUSTC_CHILD_FD}: {error}"
            )
        })?;
        let rustc_seals =
            fixed_descriptor_fcntl(RUSTC_CHILD_FD, libc::F_GET_SEALS).map_err(|error| {
                format!("binding-only host-test runner cannot inspect pinned rustc seals: {error}")
            })?;
        if rustc_stat.st_mode & libc::S_IFMT != libc::S_IFREG
            || rustc_stat.st_mode & 0o111 == 0
            || rustc_seals != Self::REQUIRED_RUSTC_SEALS
        {
            return Err(
                "binding-only host-test runner requires an executable fully sealed rustc image"
                    .to_owned(),
            );
        }

        let library_path = PathBuf::from(format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"));
        let loader_path = env::var_os("LD_LIBRARY_PATH")
            .ok_or_else(|| "binding-only host-test runner requires LD_LIBRARY_PATH".to_owned())?;
        if !env::split_paths(&loader_path).any(|component| component == library_path) {
            return Err(
                "binding-only host-test runner requires its pinned rustc library descriptor in LD_LIBRARY_PATH"
                    .to_owned(),
            );
        }

        Ok(Self {
            library_device: library_stat.st_dev,
            library_inode: library_stat.st_ino,
            library_mode: library_stat.st_mode,
            rustc_device: rustc_stat.st_dev,
            rustc_inode: rustc_stat.st_ino,
            rustc_mode: rustc_stat.st_mode,
            rustc_size: rustc_stat.st_size,
        })
    }

    fn revalidate(self) -> std::io::Result<()> {
        let library_stat = fixed_descriptor_stat(RUSTC_LIBRARY_CHILD_FD)?;
        let library_status = fixed_descriptor_fcntl(RUSTC_LIBRARY_CHILD_FD, libc::F_GETFL)?;
        let rustc_stat = fixed_descriptor_stat(RUSTC_CHILD_FD)?;
        let rustc_seals = fixed_descriptor_fcntl(RUSTC_CHILD_FD, libc::F_GET_SEALS)?;
        if (
            library_stat.st_dev,
            library_stat.st_ino,
            library_stat.st_mode,
        ) != (self.library_device, self.library_inode, self.library_mode)
            || library_status & libc::O_ACCMODE != libc::O_RDONLY
            || (
                rustc_stat.st_dev,
                rustc_stat.st_ino,
                rustc_stat.st_mode,
                rustc_stat.st_size,
            ) != (
                self.rustc_device,
                self.rustc_inode,
                self.rustc_mode,
                self.rustc_size,
            )
            || rustc_seals != Self::REQUIRED_RUSTC_SEALS
        {
            return Err(std::io::Error::from_raw_os_error(
                rustix::io::Errno::STALE.raw_os_error(),
            ));
        }
        Ok(())
    }
}

fn fixed_descriptor_stat(descriptor: std::os::fd::RawFd) -> std::io::Result<libc::stat> {
    loop {
        let mut status = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: fstat initializes the supplied stat object on success and accepts invalid raw
        // descriptor numbers by returning EBADF without dereferencing descriptor-owned memory.
        if unsafe { libc::fstat(descriptor, status.as_mut_ptr()) } == 0 {
            // SAFETY: a successful fstat initialized the complete object.
            return Ok(unsafe { status.assume_init() });
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

fn fixed_descriptor_fcntl(
    descriptor: std::os::fd::RawFd,
    command: libc::c_int,
) -> std::io::Result<libc::c_int> {
    loop {
        // SAFETY: F_GETFL and F_GET_SEALS read only the supplied descriptor state; invalid
        // descriptor numbers are reported as EBADF.
        let result = unsafe { libc::fcntl(descriptor, command) };
        if result >= 0 {
            return Ok(result);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

fn is_cargo_target_runner_environment_name(name: &OsStr) -> bool {
    let name = os_bytes(name);
    name.starts_with(b"CARGO_TARGET_")
        && name.ends_with(b"_RUNNER")
        && name.len() > b"CARGO_TARGET__RUNNER".len()
}

fn reject_ambient_cargo_test_runners() -> Result<(), String> {
    let mut runners = env::vars_os()
        .map(|(name, _)| name)
        .filter(|name| is_cargo_target_runner_environment_name(name))
        .collect::<Vec<_>>();
    runners.sort_by(|left, right| os_bytes(left).cmp(os_bytes(right)));
    if let Some(name) = runners.first() {
        return Err(format!(
            "binding-only host test rejects ambient Cargo runner selection {name:?}"
        ));
    }
    for name in ["RUSTDOC", "CARGO_BUILD_RUSTDOC", "RUSTDOCFLAGS"] {
        if let Some(value) = env::var_os(name) {
            return Err(format!(
                "binding-only host test rejects ambient rustdoc selection {name}={value:?}"
            ));
        }
    }
    Ok(())
}

fn reject_configured_cargo_test_runners(
    project: &project::CargoProject,
    args: &[OsString],
    pinned_cargo: &pinned_executable::PinnedExecutable,
) -> Result<(), String> {
    if let Some(value) = project.cargo_config_value(args, "target", pinned_cargo, None)? {
        let serde_json::Value::Object(targets) = value else {
            return Err(
                "binding-only host test cannot inspect configured Cargo target table".to_owned(),
            );
        };
        for (selector, configuration) in targets {
            let serde_json::Value::Object(configuration) = configuration else {
                return Err(format!(
                    "binding-only host test cannot inspect configured target.{selector}"
                ));
            };
            if let Some(runner) = configuration.get("runner") {
                return Err(format!(
                    "binding-only host test rejects configured target.{selector}.runner={runner}"
                ));
            }
        }
    }

    if let Some(value) = project.cargo_config_value(args, "env", pinned_cargo, None)? {
        let serde_json::Value::Object(configured) = value else {
            return Err(
                "binding-only host test cannot inspect configured Cargo env table".to_owned(),
            );
        };
        for (name, value) in configured {
            if is_cargo_target_runner_environment_name(OsStr::new(&name)) {
                return Err(format!(
                    "binding-only host test rejects configured runner environment env.{name}={value}"
                ));
            }
            if name.as_bytes().starts_with(b"FE2O3_") {
                return Err(format!(
                    "binding-only host test rejects configured protected environment env.{name}={value}"
                ));
            }
            if is_dynamic_loader_environment_name(OsStr::new(&name)) {
                return Err(format!(
                    "binding-only host test rejects configured dynamic-loader environment env.{name}={value}"
                ));
            }
        }
    }
    Ok(())
}

fn clear_inherited_cargo_unit_identity(command: &mut Command) {
    clear_cargo_unit_identity_names(command, env::vars_os().map(|(name, _)| name));
}

fn clear_cargo_unit_identity_names(
    command: &mut Command,
    names: impl IntoIterator<Item = OsString>,
) {
    for name in names {
        if name == CARGO_PRIMARY_PACKAGE_ENV
            || name == "CARGO_MANIFEST_DIR"
            || name.as_encoded_bytes().starts_with(b"CARGO_PKG_")
        {
            command.env_remove(name);
        }
    }
}

fn cargo_with_backend(command: &str, args: &[OsString]) -> ExitCode {
    match cargo_with_backend_result(command, args, None) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn cargo_with_protected_release(
    args: &[OsString],
    admission: authority_release::ProtectedReleaseAdmission,
) -> ExitCode {
    let Some(command) = args.first().and_then(|value| value.to_str()) else {
        eprintln!("protected authority release has no UTF-8 Cargo command");
        return ExitCode::FAILURE;
    };
    if !matches!(command, "build" | "run") {
        eprintln!("protected authority release child requires build or run");
        return ExitCode::FAILURE;
    }
    match cargo_with_backend_result(command, &args[1..], Some(&admission)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_application_proof_route(
    command: &str,
    args: &[OsString],
) -> Result<
    (
        Vec<OsString>,
        application_handoff::ApplicationCompilerServiceExposureV1,
    ),
    String,
> {
    use application_handoff::ApplicationCompilerServiceExposureV1 as Service;
    let mut service = Service::Required;
    let mut forwarded = Vec::with_capacity(args.len());
    let mut application_args = false;
    for argument in args {
        if !application_args
            && (argument == "--application-proof-custodian"
                || argument == "--native-application-proof-custodian")
        {
            if command != "run" || service != Service::Required {
                return Err("one explicit application proof-custodian route requires run".into());
            }
            service = if argument == "--native-application-proof-custodian" {
                Service::NativeCustodianRequired
            } else {
                Service::CustodianRequired
            };
        } else {
            application_args |= argument == "--";
            forwarded.push(argument.clone());
        }
    }
    Ok((forwarded, service))
}

fn cargo_with_backend_result(
    command: &str,
    args: &[OsString],
    protected_release: Option<&authority_release::ProtectedReleaseAdmission>,
) -> Result<(), String> {
    let (export_args, policy_export) = native_policy_export::parse(command, args)?;
    native_policy_export::reject_ambient(env::var_os(native_policy_export::ENV).as_deref())?;
    if policy_export.is_some()
        && protected_release
            .and_then(|release| release.compiler_execution_profile_v3())
            .is_none()
    {
        return Err(
            "native policy input export requires the original protected V3 build route".into(),
        );
    }
    let output_parent = policy_export
        .as_ref()
        .map(|request| request.preflight())
        .transpose()?;
    let (forwarded_args, compiler_service) = parse_application_proof_route(command, &export_args)?;
    let args = forwarded_args.as_slice();
    reject_obsolete_codegen_pipeline(env::var_os(OBSOLETE_CODEGEN_PIPELINE_ENV).as_deref())?;
    validate_production_compilation_environment(
        env::var_os(build_config::QUALIFICATION_ORACLE_ENV).as_deref(),
    )?;
    reject_dynamic_loader_environment()?;
    scrub_process_dynamic_loader_environment();
    reject_preexisting_compiler_environment()?;
    validate_production_cargo_inputs(args, env::var_os(TARGET_ENV).as_deref())?;
    let build_config =
        build_config::PreparedProductionBuildConfig::from_environment_for_cargo_setup()
            .map_err(|error| format!("production build configuration setup failed: {error}"))?;
    let requires_authorized_closure = true;
    if requires_authorized_closure {
        require_protected_authority_launch(protected_release)?;
        reject_authority_environment_overrides(args)?;
    }
    let authority_rustc_sha256 = requires_authorized_closure
        .then(authority_rustc_sha256_from_environment)
        .transpose()?;
    let authority_rustc_lib_tree_sha256 = requires_authorized_closure
        .then(authority_rustc_runtime_sha256_from_environment)
        .transpose()?;
    let authority_cargo_sha256 = requires_authorized_closure
        .then(authority_cargo_sha256_from_environment)
        .transpose()?;
    let authority_backend_sha256 = requires_authorized_closure
        .then(authority_backend_sha256_from_environment)
        .transpose()?;
    let authority_backend = authority_backend_sha256
        .map(preflight_declared_authority_backend)
        .transpose()?;
    let invocation_directory = env::current_dir()
        .map_err(|error| format!("failed to resolve Cargo invocation directory: {error}"))?;
    let cargo_declaration = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let source_cargo = if requires_authorized_closure {
        let cargo_path = require_absolute_authority_tool_path(&cargo_declaration, "CARGO")?;
        reject_authority_rustup_proxy(&cargo_path, "Cargo")?;
        pinned_executable::PinnedExecutable::open(&cargo_path)
            .map_err(|error| format!("failed to pin authority Cargo executable: {error}"))?
    } else {
        pin_default_cargo(&cargo_declaration, &invocation_directory)?
    };
    if let Some(expected) = authority_cargo_sha256
        && source_cargo.sha256() != &expected
    {
        return Err(format!(
            "cargo fe2o3 authority Cargo does not match {AUTHORITY_CARGO_SHA256_ENV}"
        ));
    }
    let pinned_cargo = if requires_authorized_closure {
        source_cargo
            .seal_executable_image()
            .map_err(|error| format!("failed to seal authority Cargo executable: {error}"))?
    } else {
        source_cargo
    };
    let authority_rustc = if requires_authorized_closure {
        Some(pin_authority_rustc(
            &invocation_directory,
            authority_rustc_sha256.expect("authority rustc digest parsed"),
            authority_rustc_lib_tree_sha256.expect("authority rustc lib-tree digest parsed"),
        )?)
    } else {
        None
    };
    let project = project::CargoProject::discover(
        args,
        Some(&pinned_cargo),
        authority_rustc.as_ref(),
        requires_authorized_closure,
    )?;
    reject_configured_compiler_selection(
        &project,
        args,
        &pinned_cargo,
        authority_rustc.as_ref(),
        requires_authorized_closure,
    )?;
    if requires_authorized_closure {
        reject_authority_config_overrides(
            &project,
            args,
            &pinned_cargo,
            authority_rustc.as_ref().expect("authority rustc pinned"),
        )?;
    }
    let pinned_rustc = match authority_rustc {
        Some(rustc) => rustc,
        None => pin_default_rustc(&project)?,
    };
    pinned_rustc.assert_lib_tree_unmutated()?;
    let authorized_closure = requires_authorized_closure
        .then(|| {
            authorized_kernel_closure::AuthorizedKernelClosureV1::observe(
                &project,
                args,
                &pinned_cargo,
                &pinned_rustc,
            )
        })
        .transpose()?;
    let protected_binding_wrapper = protected_release
        .map(authority_release::ProtectedReleaseAdmission::pin_binding_wrapper)
        .transpose()?;
    let cargo_binding_trampoline = protected_release
        .map(|_| pin_authority_cargo_binding_trampoline())
        .transpose()?;
    let protected_compiler_closure =
        protected_release.map(authority_release::ProtectedReleaseAdmission::compiler_closure);
    let compiler_execution_profile = protected_release
        .map(authority_release::ProtectedReleaseAdmission::compiler_execution_profile_identity)
        .ok_or_else(|| {
            "production compilation requires the protected compiler-execution client profile"
                .to_owned()
        })?;
    let preparation = BackendRunPreparation {
        project,
        build_config,
        pinned_cargo,
        pinned_rustc,
        authority_backend,
        protected_binding_wrapper,
        cargo_binding_trampoline,
        protected_compiler_closure,
        compiler_execution_profile,
        authorized_closure,
    };
    let mut context = BackendRunContext::prepare(preparation, args)?;
    run_cargo_with_backend(
        &mut context,
        command,
        args,
        protected_release,
        compiler_service,
        policy_export.as_ref(),
    )?;
    if let Some(parent) = output_parent {
        parent.require_bundle()?;
    }
    Ok(())
}

fn validate_production_compilation_environment(
    qualification_oracle: Option<&OsStr>,
) -> Result<(), String> {
    if let Some(value) = qualification_oracle {
        return Err(format!(
            "{} is unavailable in the production cargo-fe2o3 build; production compilation has no selector; found {value:?}",
            build_config::QUALIFICATION_ORACLE_ENV,
        ));
    }
    Ok(())
}

fn reject_obsolete_codegen_pipeline(value: Option<&OsStr>) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    Err(format!(
        "{OBSOLETE_CODEGEN_PIPELINE_ENV} has been removed; production compilation has no selector and temporary test oracles use {}; found {value:?}",
        build_config::QUALIFICATION_ORACLE_ENV,
    ))
}

fn validate_production_cargo_inputs(
    args: &[OsString],
    device_target: Option<&OsStr>,
) -> Result<(), String> {
    production_target_profile(device_target)?;
    reject_caller_target(args)
}

fn production_target_profile(
    device_target: Option<&OsStr>,
) -> Result<fe2o3_amd_target::ProductionAmdTargetProfileV1, String> {
    let profile = device_target
        .and_then(OsStr::to_str)
        .and_then(fe2o3_amd_target::ProductionAmdTargetProfileV1::from_cpu);
    profile.ok_or_else(|| {
        format!("production compilation requires exact {TARGET_ENV}=gfx942 or {TARGET_ENV}=gfx950")
    })
}

fn reject_caller_target(args: &[OsString]) -> Result<(), String> {
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            return Ok(());
        }
        if argument == "--target" {
            if args.get(index + 1).is_none() {
                return Err("--target requires an argument".to_owned());
            }
            return Err(
                "cargo fe2o3 owns device and host target selection; remove caller --target"
                    .to_owned(),
            );
        }
        if os_bytes(argument).starts_with(b"--target=") {
            return Err(
                "cargo fe2o3 owns device and host target selection; remove caller --target"
                    .to_owned(),
            );
        }
        index += 1;
    }
    Ok(())
}

fn preflight_declared_authority_backend(
    expected: [u8; 32],
) -> Result<(PathBuf, pinned_codegen_backend::PinnedCodegenBackend), String> {
    let path = env::var_os(BACKEND_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "cargo fe2o3 authority build requires {BACKEND_ENV} to name an explicit prebuilt codegen backend"
            )
        })?;
    if !path.is_absolute() {
        return Err(format!(
            "cargo fe2o3 authority build requires {BACKEND_ENV} to name an absolute prebuilt codegen backend path"
        ));
    }
    let backend = pinned_codegen_backend::PinnedCodegenBackend::open(&path)
        .map_err(|error| format!("failed to pin declared authority codegen backend: {error}"))?;
    if backend.sha256() != &expected {
        return Err(format!(
            "cargo fe2o3 authority backend does not match {AUTHORITY_BACKEND_SHA256_ENV}"
        ));
    }
    Ok((path, backend))
}

fn pin_authority_cargo_binding_trampoline() -> Result<pinned_executable::PinnedExecutable, String> {
    let declaration = env::var_os(AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_ENV).ok_or_else(|| {
        format!(
            "cargo fe2o3 authority release requires {AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_ENV}"
        )
    })?;
    let path = require_absolute_authority_tool_path(
        &declaration,
        AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_ENV,
    )?;
    let expected =
        authority_sha256_from_environment(AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_ENV)?;
    let source = pinned_executable::PinnedExecutable::open(&path)
        .map_err(|error| format!("failed to pin Cargo binding trampoline: {error}"))?;
    if source.sha256() != &expected {
        return Err(format!(
            "Cargo binding trampoline does not match {AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_ENV}"
        ));
    }
    let bytes = source
        .authenticated_bytes()
        .map_err(|error| format!("failed to authenticate Cargo binding trampoline: {error}"))?;
    cargo_binding_trampoline::validate_v1(&bytes)?;
    source
        .seal_executable_image()
        .map_err(|error| format!("failed to seal Cargo binding trampoline: {error}"))
}

struct BackendRunContext {
    target: String,
    target_profile: fe2o3_amd_target::ProductionAmdTargetProfileV1,
    host_target: String,
    project: project::CargoProject,
    backend: PathBuf,
    pinned_backend: pinned_codegen_backend::PinnedCodegenBackend,
    pinned_cargo: pinned_executable::PinnedExecutable,
    pinned_rustc: PinnedRustc,
    build_config: Option<build_config::PreparedProductionBuildConfig>,
    build_config_identity: Option<build_config::BuildConfigIdentity>,
    compiler_closure_sha256: [u8; 32],
    protected_compiler_closure: Option<fe2o3_build_authority::CompilerClosureV2>,
    compiler_execution_profile: authority_release::CompilerExecutionProfileIdentity,
    target_dir: project::PinnedDirectory,
    generation: generation::PreparedGeneration,
    managed_rustc_args: OsString,
    binding_wrapper_path: PathBuf,
    pinned_binding_wrapper: pinned_executable::PinnedExecutable,
    cargo_binding_trampoline: Option<pinned_executable::PinnedExecutable>,
    build_session: fe2o3_artifact_transaction::BuildSession,
    requires_locked_closure: bool,
    authorized_closure: Option<authorized_kernel_closure::AuthorizedKernelClosureV1>,
}

struct BackendRunPreparation {
    project: project::CargoProject,
    build_config: Option<build_config::PreparedProductionBuildConfig>,
    pinned_cargo: pinned_executable::PinnedExecutable,
    pinned_rustc: PinnedRustc,
    authority_backend: Option<(PathBuf, pinned_codegen_backend::PinnedCodegenBackend)>,
    protected_binding_wrapper: Option<pinned_executable::PinnedExecutable>,
    cargo_binding_trampoline: Option<pinned_executable::PinnedExecutable>,
    protected_compiler_closure: Option<fe2o3_build_authority::CompilerClosureV2>,
    compiler_execution_profile: authority_release::CompilerExecutionProfileIdentity,
    authorized_closure: Option<authorized_kernel_closure::AuthorizedKernelClosureV1>,
}

impl BackendRunContext {
    fn prepare(preparation: BackendRunPreparation, args: &[OsString]) -> Result<Self, String> {
        let BackendRunPreparation {
            project,
            build_config,
            pinned_cargo,
            pinned_rustc,
            authority_backend,
            protected_binding_wrapper,
            cargo_binding_trampoline,
            protected_compiler_closure,
            compiler_execution_profile,
            authorized_closure,
        } = preparation;
        let target_profile = production_target_profile(env::var_os(TARGET_ENV).as_deref())?;
        let target = target_profile.cpu().to_owned();
        let target_dir = project.open_or_create_target()?;
        pinned_rustc.assert_lib_tree_unmutated()?;
        let host_target = pinned_rustc_host_target(&pinned_rustc)?;
        let (backend, pinned_backend) = match authority_backend {
            Some(prebuilt) => prebuilt,
            None => {
                let backend = find_or_build_backend(&target_dir, &pinned_cargo, &pinned_rustc)?;
                let pinned_backend =
                    pinned_codegen_backend::PinnedCodegenBackend::open(&backend)
                        .map_err(|error| format!("failed to pin codegen backend: {error}"))?;
                (backend, pinned_backend)
            }
        };
        pinned_rustc.assert_lib_tree_unmutated()?;
        let binding_wrapper_path = env::current_exe()
            .map_err(|error| format!("failed to locate cargo-fe2o3 executable: {error}"))?;
        let pinned_binding_wrapper = match protected_binding_wrapper {
            Some(wrapper) => wrapper,
            None => pinned_executable::PinnedExecutable::open(&binding_wrapper_path)
                .map_err(|error| format!("failed to pin cargo-fe2o3 wrapper: {error}"))?,
        };
        let protected_closure = match (
            cargo_binding_trampoline.as_ref(),
            protected_compiler_closure,
        ) {
            (Some(trampoline), Some(expected)) => {
                if pinned_cargo.sha256() == trampoline.sha256()
                    || pinned_cargo.sha256() == pinned_binding_wrapper.sha256()
                    || trampoline.sha256() == pinned_binding_wrapper.sha256()
                {
                    return Err(
                        "protected Cargo, binding trampoline, and full wrapper images must be distinct"
                            .to_owned(),
                    );
                }
                let actual = compiler_toolchain::compiler_closure_v2_from_pins(
                    pinned_cargo.sha256(),
                    trampoline.sha256(),
                    pinned_binding_wrapper.sha256(),
                    pinned_rustc.executable.sha256(),
                    pinned_rustc.lib_tree_sha256(),
                    pinned_backend.sha256(),
                )
                .map_err(|error| format!("invalid protected compiler closure: {error}"))?;
                if actual != expected {
                    return Err(
                        "retained protected compiler closure differs from release admission"
                            .to_owned(),
                    );
                }
                Some(actual)
            }
            (None, None) => None,
            _ => {
                return Err(
                    "protected compiler closure and static binding trampoline must be admitted together"
                        .to_owned(),
                );
            }
        };
        let compiler_closure_sha256 = protected_closure.map_or_else(
            || {
                compiler_toolchain::compiler_closure_sha256_v1(
                    pinned_cargo.sha256(),
                    pinned_rustc.executable.sha256(),
                    pinned_rustc.lib_tree_sha256(),
                    pinned_backend.sha256(),
                )
            },
            fe2o3_build_authority::CompilerClosureV2::identity_sha256,
        );
        let build_config_identity = build_config.as_ref().map(|config| config.identity());
        let build_session = random_build_session()?;
        let mut cargo_configuration = project.semantic_configuration(
            args,
            &pinned_cargo,
            authorized_closure.is_some().then_some(&pinned_rustc),
        )?;
        if let Some(authorized_closure) = authorized_closure.as_ref() {
            cargo_configuration.extend_from_slice(b"fe2o3-authorized-kernel-closure-v1\0");
            cargo_configuration
                .extend_from_slice(&(authorized_closure.snapshot().len() as u64).to_le_bytes());
            cargo_configuration.extend_from_slice(authorized_closure.snapshot());
        }
        append_production_target_semantic_configuration(&mut cargo_configuration, target_profile);
        compiler_execution_profile.append_semantic_configuration(&mut cargo_configuration);
        let backend_reference = pinned_backend
            .fixed_child_descriptor_path(BACKEND_CHILD_FD)
            .map_err(|error| format!("failed to retain pinned codegen backend: {error}"))?;
        let semantic = generation::semantic_identity(
            &target,
            &compiler_closure_sha256,
            build_config_identity,
            &cargo_configuration,
        )?;
        let generation = generation::PreparedGeneration::prepare(&target_dir, semantic)?;
        let managed_rustc_args =
            generation::managed_rustc_args(&backend_reference, generation.token())?;
        Ok(Self {
            target,
            target_profile,
            host_target,
            project,
            backend,
            pinned_backend,
            pinned_cargo,
            pinned_rustc,
            build_config,
            build_config_identity,
            compiler_closure_sha256,
            protected_compiler_closure: protected_closure,
            compiler_execution_profile,
            target_dir,
            generation,
            managed_rustc_args,
            binding_wrapper_path,
            pinned_binding_wrapper,
            cargo_binding_trampoline,
            build_session,
            requires_locked_closure: authorized_closure.is_some(),
            authorized_closure,
        })
    }
}

fn append_production_target_semantic_configuration(
    configuration: &mut Vec<u8>,
    profile: fe2o3_amd_target::ProductionAmdTargetProfileV1,
) {
    configuration.extend_from_slice(b"fe2o3-production-target-profile-v1\0");
    configuration.extend_from_slice(profile.rustc_target().as_bytes());
    configuration.push(0);
    configuration.extend_from_slice(profile.cargo_rustflags().as_bytes());
    configuration.push(0);
}

#[cfg(test)]
fn append_compiler_execution_profile_semantic_configuration(
    configuration: &mut Vec<u8>,
    profile: &fe2o3_compiler_execution_protocol::CompilerExecutionClientProfileV1,
) {
    authority_release::CompilerExecutionProfileIdentity::V1(*profile.identity().as_bytes())
        .append_semantic_configuration(configuration);
}

fn run_cargo_with_backend(
    context: &mut BackendRunContext,
    command: &str,
    args: &[OsString],
    protected_release: Option<&authority_release::ProtectedReleaseAdmission>,
    compiler_service: application_handoff::ApplicationCompilerServiceExposureV1,
    policy_export: Option<&native_policy_export::Request>,
) -> Result<(), String> {
    run_cargo_with_backend_inner(
        context,
        command,
        args,
        protected_release,
        compiler_service,
        policy_export,
    )
}

fn run_cargo_with_backend_inner(
    context: &mut BackendRunContext,
    command: &str,
    args: &[OsString],
    protected_release: Option<&authority_release::ProtectedReleaseAdmission>,
    compiler_service: application_handoff::ApplicationCompilerServiceExposureV1,
    policy_export: Option<&native_policy_export::Request>,
) -> Result<(), String> {
    if policy_export.is_some()
        && context.target_profile != fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942
    {
        return Err("native policy input export is closed to gfx942".into());
    }
    context.project.validate_paths()?;
    context.target_dir.validate_path("Cargo target directory")?;
    context.generation.reject_if_substituted()?;
    let mut production_plan = if compiler_service.requires_custodian() {
        if command != "run" {
            return Err("custodian application route requires run".into());
        }
        production_cargo_plan::ProductionCargoPlan::new_for_custodian_run(
            args,
            &context.host_target,
            context.requires_locked_closure,
        )?
    } else {
        production_cargo_plan::ProductionCargoPlan::new(
            command,
            args,
            &context.host_target,
            context.requires_locked_closure,
        )?
    };
    let cargo_command = production_plan.device().command();
    eprintln!(
        "cargo fe2o3 {command}: device phase uses backend {} for target {}",
        context.backend.display(),
        context.target
    );

    let mut cargo = context
        .pinned_cargo
        .command()
        .map_err(|error| format!("failed to prepare pinned Cargo executable: {error}"))?;
    let workspace_wrapper = match context.cargo_binding_trampoline.as_ref() {
        Some(trampoline) => {
            context
                .pinned_binding_wrapper
                .inherit_for_child_at(cargo.as_command_mut(), CARGO_BINDING_WRAPPER_CHILD_FD)
                .map_err(|error| {
                    format!("failed to inherit sealed cargo-fe2o3 wrapper: {error}")
                })?;
            let path = trampoline
                .fixed_child_path(CARGO_BINDING_TRAMPOLINE_CHILD_FD)
                .map_err(|error| format!("failed to retain Cargo binding trampoline: {error}"))?;
            trampoline
                .inherit_for_child_at(cargo.as_command_mut(), CARGO_BINDING_TRAMPOLINE_CHILD_FD)
                .map_err(|error| format!("failed to inherit Cargo binding trampoline: {error}"))?;
            path
        }
        None => context.binding_wrapper_path.clone(),
    };
    let forwarded_args = production_plan.device().args().to_vec();
    let artifact_dir = context.generation.artifact_dir();
    let capability_profile = capability_broker::CapabilityProfileV1::Ordinary;
    let rustc_lib_tree_stat = rustix::fs::fstat(context.pinned_rustc.lib_tree_directory().file())
        .map_err(|error| {
        format!("failed to inspect retained rustc lib-tree directory: {error}")
    })?;
    let retained_object_binding_sha256 = compiler_toolchain::retained_object_binding_sha256_v1(
        &context.compiler_closure_sha256,
        rustc_lib_tree_stat.st_dev,
        rustc_lib_tree_stat.st_ino,
        rustc_lib_tree_stat.st_mode,
    );
    let config_identity = context
        .build_config_identity
        .map(|identity| *identity.as_bytes());
    let source_isa_observer_policy = context
        .build_config
        .as_ref()
        .map(build_config::PreparedProductionBuildConfig::source_isa_observer_policy)
        .transpose()
        .map_err(|error| format!("source/ISA observer policy setup failed: {error}"))?
        .flatten();
    let source_isa_observer_enabled = source_isa_observer_policy.is_some();
    let capability_broker = if let Some(compiler_closure) = context.protected_compiler_closure {
        let protected_release = protected_release.ok_or_else(|| {
            "protected compiler closure has no retained authority release".to_owned()
        })?;
        if protected_release.compiler_execution_profile_identity()
            != context.compiler_execution_profile
        {
            return Err(
                "retained compiler-execution client profile changed after generation preparation"
                    .to_owned(),
            );
        }
        if let Some(profile) = protected_release.compiler_execution_profile_v3() {
            let binding = capability_broker::CapabilityBindingV3::new_protected_v4(
                capability_profile,
                config_identity,
                compiler_closure,
                retained_object_binding_sha256,
                *profile.profile().identity().as_bytes(),
            )?;
            capability_broker::CapabilityBroker::start_protected_v4(
                context.build_session,
                binding,
                compiler_closure,
                profile,
                &context.pinned_backend,
                artifact_dir,
                &context.pinned_cargo,
                source_isa_observer_policy.as_ref(),
            )?
        } else {
            let profile = protected_release
                .compiler_execution_profile_capability()
                .ok_or("protected release has no retained compiler-execution profile")?;
            let binding = capability_broker::CapabilityBindingV3::new_protected(
                capability_profile,
                config_identity,
                compiler_closure,
                retained_object_binding_sha256,
            )?;
            match source_isa_observer_policy.as_ref() {
                Some(policy) => {
                    capability_broker::CapabilityBroker::start_protected_with_source_isa_observer(
                        context.build_session,
                        binding,
                        compiler_closure,
                        profile,
                        &context.pinned_backend,
                        artifact_dir,
                        &context.pinned_cargo,
                        policy,
                    )?
                }
                None => capability_broker::CapabilityBroker::start_protected(
                    context.build_session,
                    binding,
                    compiler_closure,
                    profile,
                    &context.pinned_backend,
                    artifact_dir,
                    &context.pinned_cargo,
                )?,
            }
        }
    } else {
        if source_isa_observer_enabled {
            return Err(
                "source/ISA observation requires the protected compiler closure".to_owned(),
            );
        }
        let binding = capability_broker::CapabilityBindingV3::new(
            capability_profile,
            config_identity,
            context.compiler_closure_sha256,
            *context.pinned_rustc.executable.sha256(),
            retained_object_binding_sha256,
        )?;
        capability_broker::CapabilityBroker::start(
            context.build_session,
            binding,
            &context.pinned_backend,
            artifact_dir,
            &context.pinned_cargo,
        )?
    };
    let capability_broker =
        CapabilityBrokerCompletionV1::new(capability_broker, source_isa_observer_enabled);
    let invocation_authorization = capability_broker.broker().invocation_authorization();
    let pending_invocation_boundary =
        cargo_invocation_boundary::PendingCargoInvocationBoundary::start(
            &context.pinned_cargo,
            &context.pinned_binding_wrapper,
            context.cargo_binding_trampoline.as_ref(),
            invocation_authorization.clone(),
        )?;
    cargo
        .as_command_mut()
        .arg(cargo_command)
        .args(&forwarded_args)
        .current_dir(context.project.invocation_dir().child_path())
        .env_remove(HSACO_DIR_ENV)
        .env(
            capability_broker::CAPABILITY_BROKER_ENV,
            capability_broker.broker().route(),
        )
        .env(TARGET_ENV, &context.target)
        .env("RUSTC_WRAPPER", "")
        .env("CARGO_BUILD_RUSTC_WRAPPER", "")
        .env("RUSTC_WORKSPACE_WRAPPER", workspace_wrapper)
        // Cargo owns this per-unit marker. Do not let the caller preselect
        // dependency units before Cargo computes its unit graph.
        .env_remove(CARGO_PRIMARY_PACKAGE_ENV)
        .env(BINDING_WRAPPER_MODE_ENV, "1")
        .env(MANAGED_RUSTC_ARGS_ENV, &context.managed_rustc_args)
        .env(
            EXPECTED_RUSTC_SHA256_ENV,
            hex_encode(context.pinned_rustc.executable.sha256()),
        )
        .env(
            EXPECTED_COMPILER_CLOSURE_SHA256_ENV,
            hex_encode(&context.compiler_closure_sha256),
        )
        .env(BUILD_SESSION_ENV, context.build_session.to_hex());
    configure_production_cargo_tool_environment(cargo.as_command_mut());
    native_policy_export::configure(cargo.as_command_mut(), policy_export)?;
    scrub_simulation_build_environment(cargo.as_command_mut());
    configure_production_target_environment(cargo.as_command_mut(), context.target_profile);
    if context.requires_locked_closure {
        // Authority builds do not admit unpinned C tools, ROCm headers, or native libraries.
        cargo.as_command_mut().env("FE2O3_HIP_SYS_DISABLE", "1");
    }
    cargo
        .as_command_mut()
        .env_remove(AUTHORITY_CARGO_SHA256_ENV)
        .env_remove(AUTHORITY_RUSTC_SHA256_ENV)
        .env_remove(AUTHORITY_RUSTC_PATH_ENV)
        .env_remove(AUTHORITY_RUSTC_RUNTIME_SHA256_ENV)
        .env_remove(AUTHORITY_BACKEND_SHA256_ENV)
        .env_remove(AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_ENV)
        .env_remove(AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_ENV);
    remove_dynamic_loader_environment(cargo.as_command_mut());
    context.pinned_rustc.assert_lib_tree_unmutated()?;
    configure_pinned_rustc_child(cargo.as_command_mut(), &context.pinned_rustc)?;
    cargo.as_command_mut().env(
        "LD_LIBRARY_PATH",
        format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"),
    );
    cargo
        .as_command_mut()
        .env_remove(build_config::PRODUCTION_BUILD_EXPECTED_ID_ENV)
        .env_remove(build_config::PRODUCTION_BUILD_EXPECTED_ID_V2_ENV)
        .env_remove(build_config::WORKER_V2_EXPECTED_ID_ENV);
    match (context.build_config_identity, context.build_config.as_ref()) {
        (Some(identity), Some(config)) => {
            cargo.as_command_mut().env(
                config.expected_identity_environment_name(),
                identity.to_hex(),
            );
        }
        (None, None) => {}
        _ => unreachable!("production build configuration and identity have matching presence"),
    }
    if let Some(admission) = protected_release {
        admission.configure_descendant(cargo.as_command_mut());
    }
    pending_invocation_boundary.configure_child(cargo.as_command_mut());
    let mut cargo_child = cargo
        .spawn()
        .map_err(|error| format!("failed to run pinned Cargo: {error}"))?;
    let invocation_boundary =
        match pending_invocation_boundary.complete(cargo_child.id(), invocation_authorization) {
            Ok(boundary) => boundary,
            Err(error) => {
                let _ = cargo_child.kill();
                let cleanup_result = cargo_child
                    .wait()
                    .map(|_| ())
                    .map_err(|cleanup| format!("failed to reap rejected Cargo child: {cleanup}"));
                capability_broker.finish();
                let lib_tree_result = context.pinned_rustc.revalidate_lib_tree();
                let closure_result = context
                    .authorized_closure
                    .as_ref()
                    .map_or(Ok(()), |closure| closure.revalidate());
                return aggregate_post_spawn_results(
                    Err(error),
                    [
                        ("Cargo child cleanup", cleanup_result),
                        ("rustc runtime-tree revalidation", lib_tree_result),
                        ("authorized kernel-closure revalidation", closure_result),
                    ],
                );
            }
        };
    let status = cargo_child.wait();
    let boundary_result = invocation_boundary.finish();
    capability_broker.finish();
    let lib_tree_result = context.pinned_rustc.revalidate_lib_tree();
    let closure_result = context
        .authorized_closure
        .as_ref()
        .map_or(Ok(()), |closure| closure.revalidate());
    let cargo_result = match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!(
            "cargo fe2o3 device phase ({cargo_command}) failed with status {status}"
        )),
        Err(error) => Err(format!("failed to run cargo: {error}")),
    };
    aggregate_post_spawn_results(
        cargo_result,
        [
            ("Cargo invocation-boundary finish", boundary_result),
            ("rustc runtime-tree revalidation", lib_tree_result),
            ("authorized kernel-closure revalidation", closure_result),
        ],
    )?;

    context.project.validate_paths()?;
    context.target_dir.validate_path("Cargo target directory")?;
    context.generation.reject_if_substituted()?;
    context.generation.commit()?;
    if command == "run" {
        if !context.requires_locked_closure {
            return Err(
                "production Worker V3 run requires an authorized locked compiler closure"
                    .to_owned(),
            );
        }
        inject_production_application_runner(
            context,
            production_plan.host_mut().args_mut(),
            compiler_service,
        )?;
    }
    run_production_host_cargo(
        context,
        production_plan.host(),
        protected_release,
        compiler_service,
    )?;
    Ok(())
}

struct CapabilityBrokerCompletionV1 {
    broker: ObserverFinishOnDropV1<capability_broker::CapabilityBroker>,
}

impl CapabilityBrokerCompletionV1 {
    fn new(broker: capability_broker::CapabilityBroker, observer_enabled: bool) -> Self {
        Self {
            broker: ObserverFinishOnDropV1::new(
                broker,
                observer_enabled,
                finish_capability_broker_observations,
            ),
        }
    }

    fn broker(&self) -> &capability_broker::CapabilityBroker {
        self.broker.get()
    }

    fn finish(self) {
        self.broker.finish();
    }
}

struct ObserverFinishOnDropV1<Value> {
    value: Option<Value>,
    observer_enabled: bool,
    finish: fn(Value, bool),
}

impl<Value> ObserverFinishOnDropV1<Value> {
    fn new(value: Value, observer_enabled: bool, finish: fn(Value, bool)) -> Self {
        Self {
            value: Some(value),
            observer_enabled,
            finish,
        }
    }

    fn get(&self) -> &Value {
        self.value
            .as_ref()
            .expect("observer completion value is live before completion")
    }

    fn finish(mut self) {
        let value = self
            .value
            .take()
            .expect("observer completion runs exactly once");
        (self.finish)(value, self.observer_enabled);
    }
}

impl<Value> Drop for ObserverFinishOnDropV1<Value> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            (self.finish)(value, self.observer_enabled);
        }
    }
}

fn finish_capability_broker_observations(
    broker: capability_broker::CapabilityBroker,
    observer_enabled: bool,
) {
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    finish_capability_broker_observations_to(broker, observer_enabled, &mut stderr);
}

fn finish_capability_broker_observations_to(
    broker: capability_broker::CapabilityBroker,
    observer_enabled: bool,
    output: &mut (impl std::io::Write + ?Sized),
) {
    if !observer_enabled {
        drop(broker);
        return;
    }
    let completed = match broker.finish_source_isa_observations() {
        Ok(completed) => completed,
        Err(error) => {
            let _ = observer_telemetry::write_line_to(
                output,
                format_args!("[cargo-fe2o3] source/ISA observer collection failed: {error}"),
            );
            return;
        }
    };
    let collection = completed.summary;
    debug_assert!(!collection.grants_compiler_authority());
    debug_assert!(!collection.grants_publication_authority());
    debug_assert!(!collection.grants_runtime_authority());
    match collection.encode_canonical() {
        Ok(encoded) => {
            let decoded =
                match fe2o3_source_isa_observation::wire_v1::SourceIsaObservationCollectionV1::decode_canonical(
                    &encoded,
                ) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        let _ = observer_telemetry::write_line_to(
                            output,
                            format_args!(
                                "[cargo-fe2o3] source/ISA observer self-validation failed: {error}"
                            ),
                        );
                        return;
                    }
                };
            let frames = decoded.frames().len();
            let missing = decoded.missing_units().len();
            let failure = decoded.failure().map_or(0, |failure| failure.code());
            if !completed.census.is_empty() {
                if completed.census.len() != frames || missing != 0 || failure != 0 {
                    let _ = observer_telemetry::write_line_to(
                        output,
                        format_args!("[cargo-fe2o3] incomplete production census refused"),
                    );
                    return;
                }
                let encoded = (|| -> Result<Vec<u8>, String> {
                    let mut rows = Vec::new();
                    rows.try_reserve_exact(frames).map_err(|e| e.to_string())?;
                    for ((unit, bytes), frame) in completed.census.iter().zip(decoded.frames()) {
                        let row = production_census_v91::Census::decode(bytes)?;
                        row.check_frame(frame)?;
                        if row.unit != *unit || row.config != completed.config {
                            return Err("production census collector context differs".to_owned());
                        }
                        rows.push(row);
                    }
                    let bytes = serde_json::to_vec(&rows).map_err(|e| e.to_string())?;
                    if bytes.len() > production_census_v91::MAX_AGGREGATE {
                        return Err("production census aggregate bound".to_owned());
                    }
                    Ok(bytes)
                })();
                match encoded.and_then(|bytes| source_isa_collection_hex(&bytes)) {
                    Ok(bytes) => {
                        let _ = observer_telemetry::write_line_to(
                            output,
                            format_args!(
                                "{} frames={frames} missing=0 failure=0 encoding=hex:{bytes} authority=observation-only",
                                production_census_v91::PREFIX
                            ),
                        );
                    }
                    Err(error) => {
                        let _ = observer_telemetry::write_line_to(
                            output,
                            format_args!(
                                "[cargo-fe2o3] production census encoding failed: {error}"
                            ),
                        );
                    }
                }
                return;
            }
            if let Some((unit, characteristic)) = completed.characteristic {
                if frames != 1 || missing != 0 || failure != 0 {
                    let _ = observer_telemetry::write_line_to(
                        output,
                        format_args!(
                            "[cargo-fe2o3] Source/ISA characteristic export rejected an incomplete summary"
                        ),
                    );
                    return;
                }
                match source_isa_observation::encode_source_isa_characteristic_broker_v3(
                    completed.config,
                    unit,
                    &characteristic,
                )
                .and_then(|cell| source_isa_characteristic_broker_hex(&cell))
                {
                    Ok(encoded) => {
                        let _ = observer_telemetry::write_line_to(
                            output,
                            format_args!(
                                "{SOURCE_ISA_CHARACTERISTIC_BROKER_STDERR_PREFIX_V3} encoding=hex:{encoded} authority=observation-only"
                            ),
                        );
                    }
                    Err(error) => {
                        let _ = observer_telemetry::write_line_to(
                            output,
                            format_args!(
                                "[cargo-fe2o3] Source/ISA characteristic Broker V3 encoding failed: {error}"
                            ),
                        );
                        return;
                    }
                }
            }
            match source_isa_collection_hex(&encoded) {
                Ok(encoded) => {
                    let _ = observer_telemetry::write_line_to(
                        output,
                        format_args!(
                            "{SOURCE_ISA_COLLECTION_STDERR_PREFIX_V1} frames={frames} missing={missing} failure={failure} encoding=hex:{encoded} authority=observation-only"
                        ),
                    );
                }
                Err(error) => {
                    let _ = observer_telemetry::write_line_to(
                        output,
                        format_args!(
                            "[cargo-fe2o3] source/ISA observer hex encoding failed: {error}"
                        ),
                    );
                }
            }
        }
        Err(error) => {
            let _ = observer_telemetry::write_line_to(
                output,
                format_args!("[cargo-fe2o3] source/ISA observer encoding failed: {error}"),
            );
        }
    }
}

fn source_isa_collection_hex(bytes: &[u8]) -> Result<String, String> {
    let encoded_len = source_isa_collection_hex_length(bytes.len())?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::new();
    encoded
        .try_reserve_exact(encoded_len)
        .map_err(|_| "cannot allocate bounded source/ISA collection hex".to_owned())?;
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

fn source_isa_characteristic_broker_hex(bytes: &[u8]) -> Result<String, String> {
    let encoded_len = source_isa_characteristic_broker_hex_length(bytes.len())?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::new();
    encoded.try_reserve_exact(encoded_len).map_err(|_| {
        "cannot allocate bounded Source/ISA characteristic Broker V3 hex".to_owned()
    })?;
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

fn source_isa_characteristic_broker_hex_length(binary_len: usize) -> Result<usize, String> {
    let maximum = fe2o3_source_isa_observation::characteristic_v1::MAX_SOURCE_ISA_CHARACTERISTIC_COLLECTION_BYTES_V1
        .checked_add(256)
        .and_then(|length| length.checked_mul(2))
        .ok_or_else(|| "Source/ISA characteristic Broker V3 hex bound overflow".to_owned())?;
    binary_len
        .checked_mul(2)
        .filter(|length| *length <= maximum)
        .ok_or_else(|| "Source/ISA characteristic Broker V3 hex exceeds its bound".to_owned())
}

fn source_isa_collection_hex_length(binary_len: usize) -> Result<usize, String> {
    binary_len
        .checked_mul(2)
        .filter(|bytes| {
            *bytes <= fe2o3_source_isa_observation::wire_v1::MAX_SOURCE_ISA_OBSERVATION_COLLECTION_HEX_BYTES_V1
        })
        .ok_or_else(|| "source/ISA collection hex exceeds its canonical bound".to_owned())
}

fn run_production_host_cargo(
    context: &BackendRunContext,
    phase: &production_cargo_plan::CargoPhase,
    protected_release: Option<&authority_release::ProtectedReleaseAdmission>,
    compiler_service: application_handoff::ApplicationCompilerServiceExposureV1,
) -> Result<(), String> {
    if compiler_service
        == application_handoff::ApplicationCompilerServiceExposureV1::NativeCustodianRequired
    {
        if phase.command() != "run" {
            return Err("native custodian host binding is restricted to run".into());
        }
        let admission =
            protected_release.ok_or("native custodian run requires protected release")?;
        let profile = admission
            .compiler_execution_profile_v3()
            .ok_or("native custodian run requires the actual native V3 profile")?;
        return profile.with_profile_budget(|capability, budget| {
            let projection =
                production_host_binding::CommittedNativeHostBindingProjection::prepare(
                    context, admission, capability, budget,
                )?;
            let result =
                run_production_host_cargo_inner(context, phase, Some(admission), Some(&projection));
            aggregate_post_spawn_results(
                result,
                [(
                    "native committed host binding revalidation",
                    projection.revalidate(budget),
                )],
            )
        });
    }
    let mut projection = if compiler_service.requires_custodian() {
        if phase.command() != "run" {
            return Err("custodian host binding is restricted to run".into());
        }
        let admission =
            protected_release.ok_or("custodian run requires protected release admission")?;
        Some(production_host_binding::CommittedHostBindingProjection::prepare(context, admission)?)
    } else {
        None
    };
    let result = run_production_host_cargo_inner(
        context,
        phase,
        protected_release,
        projection
            .as_ref()
            .map(|p| p as &dyn production_host_binding::HostBindingProjection),
    );
    aggregate_post_spawn_results(
        result,
        [(
            "committed host binding revalidation",
            projection.as_mut().map_or(
                Ok(()),
                production_host_binding::CommittedHostBindingProjection::revalidate,
            ),
        )],
    )
}

fn run_production_host_cargo_inner(
    context: &BackendRunContext,
    phase: &production_cargo_plan::CargoPhase,
    protected_release: Option<&authority_release::ProtectedReleaseAdmission>,
    projection: Option<&dyn production_host_binding::HostBindingProjection>,
) -> Result<(), String> {
    context.project.validate_paths()?;
    context.target_dir.validate_path("Cargo target directory")?;
    context.generation.reject_if_substituted()?;
    eprintln!(
        "cargo fe2o3 {}: host phase uses ordinary rustc for target {}",
        phase.command(),
        context.host_target
    );

    let mut cargo = context
        .pinned_cargo
        .command()
        .map_err(|error| format!("failed to prepare pinned host Cargo executable: {error}"))?;
    cargo
        .as_command_mut()
        .arg(phase.command())
        .args(phase.args())
        .current_dir(context.project.invocation_dir().child_path())
        .env_remove(HSACO_DIR_ENV)
        .env_remove(BACKEND_ENV)
        .env_remove(OBSOLETE_CODEGEN_PIPELINE_ENV)
        .env_remove(capability_broker::CAPABILITY_BROKER_ENV)
        .env_remove(TARGET_ENV)
        .env("RUSTC_WRAPPER", "")
        .env("CARGO_BUILD_RUSTC_WRAPPER", "")
        .env("RUSTC_WORKSPACE_WRAPPER", "")
        .env_remove(CARGO_PRIMARY_PACKAGE_ENV)
        .env_remove(BINDING_WRAPPER_MODE_ENV)
        .env_remove(production_host_binding::MODE_ENV)
        .env_remove(reserved_fe2o3_symbols::CRATE_BINDING_ID_ENV_V1)
        .env_remove(MANAGED_RUSTC_ARGS_ENV)
        .env_remove(EXPECTED_RUSTC_SHA256_ENV)
        .env_remove(EXPECTED_COMPILER_CLOSURE_SHA256_ENV)
        .env_remove(BUILD_SESSION_ENV)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove(fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_ENV_V1)
        .env_remove(fe2o3_amd_target::PRODUCTION_GFX950_CARGO_RUSTFLAGS_ENV_V1)
        .env_remove(build_config::PRODUCTION_BUILD_EXPECTED_ID_ENV)
        .env_remove(build_config::PRODUCTION_BUILD_EXPECTED_ID_V2_ENV)
        .env_remove(build_config::PRODUCTION_BUILD_CONFIG_ENV)
        .env_remove(build_config::PRODUCTION_BUILD_CONFIG_V2_ENV)
        .env_remove(build_config::WORKER_V2_EXPECTED_ID_ENV)
        .env_remove(build_config::WORKER_V2_CONFIG_ENV)
        .env_remove(build_config::QUALIFICATION_ORACLE_ENV)
        .env_remove(OBSOLETE_SIMULATION_MODE_ENV)
        .env_remove(OBSOLETE_SIMULATION_ATTEMPT_ENV)
        .env_remove(NON_PRODUCTION_AUTHORITY_VALIDATION_ENV)
        .env_remove(AUTHORITY_CARGO_SHA256_ENV)
        .env_remove(AUTHORITY_RUSTC_SHA256_ENV)
        .env_remove(AUTHORITY_RUSTC_PATH_ENV)
        .env_remove(AUTHORITY_RUSTC_RUNTIME_SHA256_ENV)
        .env_remove(AUTHORITY_BACKEND_SHA256_ENV)
        .env_remove(AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_ENV)
        .env_remove(AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_ENV);
    configure_production_cargo_tool_environment(cargo.as_command_mut());
    if context.requires_locked_closure {
        cargo.as_command_mut().env("FE2O3_HIP_SYS_DISABLE", "1");
    }
    remove_dynamic_loader_environment(cargo.as_command_mut());
    context.pinned_rustc.assert_lib_tree_unmutated()?;
    configure_pinned_rustc_child(cargo.as_command_mut(), &context.pinned_rustc)?;
    cargo.as_command_mut().env(
        "LD_LIBRARY_PATH",
        format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"),
    );
    if let Some(admission) = protected_release {
        admission.configure_descendant(cargo.as_command_mut());
    }

    if let Some(projection) = projection {
        projection.configure_child(cargo.as_command_mut(), &context.pinned_binding_wrapper)?;
    } else if phase.command() == "run" {
        context
            .pinned_binding_wrapper
            .inherit_for_child_at(cargo.as_command_mut(), CARGO_BINDING_CHECK_WRAPPER_CHILD_FD)
            .map_err(|error| format!("inherit production application runner: {error}"))?;
    }
    let invocation_authorization =
        cargo_invocation_boundary::InvocationAuthorizationRegistryV1::new();
    // Host run has no compiler broker or permit consumer. Its application installs
    // the one-exec/no-fork listener; Linux rejects nesting another USER_NOTIF listener.
    let pending_invocation_boundary = if phase.command() == "run" {
        None
    } else {
        let pending = cargo_invocation_boundary::PendingCargoInvocationBoundary::start(
            &context.pinned_cargo,
            &context.pinned_binding_wrapper,
            None,
            invocation_authorization.clone(),
        )?;
        pending.configure_child(cargo.as_command_mut());
        Some(pending)
    };
    let mut cargo_child = cargo
        .spawn()
        .map_err(|error| format!("failed to run pinned host Cargo: {error}"))?;
    let invocation_boundary = match pending_invocation_boundary {
        None => None,
        Some(pending) => match pending.complete(cargo_child.id(), invocation_authorization) {
            Ok(boundary) => Some(boundary),
            Err(error) => {
                let _ = cargo_child.kill();
                let cleanup_result = cargo_child.wait().map(|_| ()).map_err(|cleanup| {
                    format!("failed to reap rejected host Cargo child: {cleanup}")
                });
                let lib_tree_result = context.pinned_rustc.revalidate_lib_tree();
                let closure_result = context
                    .authorized_closure
                    .as_ref()
                    .map_or(Ok(()), |closure| closure.revalidate());
                return aggregate_post_spawn_results(
                    Err(error),
                    [
                        ("host Cargo child cleanup", cleanup_result),
                        ("rustc runtime-tree revalidation", lib_tree_result),
                        ("authorized kernel-closure revalidation", closure_result),
                    ],
                );
            }
        },
    };
    let status = cargo_child.wait();
    let boundary_result = invocation_boundary.map_or(Ok(()), |boundary| boundary.finish());
    let lib_tree_result = context.pinned_rustc.revalidate_lib_tree();
    let closure_result = context
        .authorized_closure
        .as_ref()
        .map_or(Ok(()), |closure| closure.revalidate());
    let cargo_result = match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!(
            "cargo fe2o3 host phase ({}) failed with status {status}",
            phase.command()
        )),
        Err(error) => Err(format!("failed to wait for host Cargo: {error}")),
    };
    aggregate_post_spawn_results(
        cargo_result,
        [
            ("host Cargo invocation-boundary finish", boundary_result),
            ("rustc runtime-tree revalidation", lib_tree_result),
            ("authorized kernel-closure revalidation", closure_result),
        ],
    )?;

    context.project.validate_paths()?;
    context.target_dir.validate_path("Cargo target directory")?;
    context.generation.reject_if_substituted()
}

fn configure_production_cargo_tool_environment(command: &mut Command) {
    command.env("PATH", "/usr/bin");
}

fn scrub_simulation_build_environment(command: &mut Command) {
    command
        .env_remove(OBSOLETE_SIMULATION_MODE_ENV)
        .env_remove(OBSOLETE_SIMULATION_ATTEMPT_ENV)
        .env_remove("FE2O3_HIP_SYS_DISABLE");
}

fn configure_production_target_environment(
    command: &mut Command,
    profile: fe2o3_amd_target::ProductionAmdTargetProfileV1,
) {
    command
        .env(TARGET_ENV, profile.device_target())
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove(fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_ENV_V1)
        .env_remove(fe2o3_amd_target::PRODUCTION_GFX950_CARGO_RUSTFLAGS_ENV_V1)
        .env(profile.cargo_rustflags_env(), profile.cargo_rustflags());
}

fn aggregate_post_spawn_results<const N: usize>(
    primary: Result<(), String>,
    checks: [(&str, Result<(), String>); N],
) -> Result<(), String> {
    let mut failure = primary.err();
    for (label, result) in checks {
        if let Err(error) = result {
            match failure.as_mut() {
                Some(primary) => primary.push_str(&format!("; {label} also failed: {error}")),
                None => failure = Some(error),
            }
        }
    }
    failure.map_or(Ok(()), Err)
}

fn resolve_application_runner(
    project: &project::CargoProject,
    pinned_cargo: &pinned_executable::PinnedExecutable,
    pinned_rustc: &PinnedRustc,
    args: &[OsString],
    authority: bool,
) -> Result<(String, Vec<OsString>), String> {
    let target = match selected_run_target(args)? {
        Some(target) => target,
        None => match configured_run_target(
            project,
            pinned_cargo,
            authority.then_some(pinned_rustc),
            args,
        )? {
            Some(target) => target,
            None if authority => {
                return Err(
                    "cargo fe2o3 authority run requires an explicit --target or reviewed build.target"
                        .to_owned(),
                );
            }
            None => host_rustc_target()?,
        },
    };
    if !target
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(format!(
            "unsupported Cargo run target for runner isolation: {target:?}"
        ));
    }
    let original_runner = resolve_original_runner(
        project,
        pinned_cargo,
        authority.then_some(pinned_rustc),
        args,
        &target,
    )?;
    reject_recursive_runner(&original_runner)?;
    Ok((target, original_runner))
}

fn inject_production_application_runner(
    context: &BackendRunContext,
    args: &mut Vec<OsString>,
    compiler_service: application_handoff::ApplicationCompilerServiceExposureV1,
) -> Result<(), String> {
    let (target, original_runner) = resolve_application_runner(
        &context.project,
        &context.pinned_cargo,
        &context.pinned_rustc,
        args,
        true,
    )?;
    if !original_runner.is_empty() {
        return Err(
            "production Worker V3 application handoff does not permit an intermediate Cargo runner"
                .to_owned(),
        );
    }
    let executable = context
        .pinned_binding_wrapper
        .fixed_child_path(CARGO_BINDING_CHECK_WRAPPER_CHILD_FD)
        .map_err(|error| format!("retain production application runner: {error}"))?;
    let artifact_dir = context.generation.artifact_dir();
    let (artifact_device, artifact_inode) = artifact_dir.identity_parts();
    inject_serialized_application_runner_config(
        args,
        &target,
        vec![
            executable.to_string_lossy().into_owned(),
            INTERNAL_RUNNER_ARG.to_string(),
            match compiler_service {
                application_handoff::ApplicationCompilerServiceExposureV1::CustodianRequired => {
                    application_handoff::RUNNER_CUSTODIAN_CONTEXT_VERSION
                }
                application_handoff::ApplicationCompilerServiceExposureV1::NativeCustodianRequired => {
                    application_handoff::RUNNER_NATIVE_CUSTODIAN_CONTEXT_VERSION
                }
                application_handoff::ApplicationCompilerServiceExposureV1::Required => {
                    application_handoff::RUNNER_CONTEXT_VERSION
                }
                #[cfg(any(
                    test,
                    feature = "application-handoff-adversarial-fixture",
                    feature = "application-handoff-fault-injection-test-only"
                ))]
                application_handoff::ApplicationCompilerServiceExposureV1::TestDisabled => {
                    return Err("production runner requires compiler service custody".into());
                }
            }
            .to_string(),
            hex_encode(os_bytes(artifact_dir.display_path().as_os_str())),
            artifact_device.to_string(),
            artifact_inode.to_string(),
            application_handoff::RUNNER_EXPECTS_ENVELOPE.to_string(),
            "0".to_owned(),
        ],
    )
}

fn inject_serialized_application_runner_config(
    args: &mut Vec<OsString>,
    target: &str,
    runner: Vec<String>,
) -> Result<(), String> {
    let runner = serde_json::to_string(&runner)
        .map_err(|error| format!("failed to encode Cargo runner configuration: {error}"))?;
    let config = OsString::from(format!("target.{target}.runner={runner}"));
    let insert_at = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    args.insert(insert_at, OsString::from("--config"));
    args.insert(insert_at + 1, config);
    Ok(())
}

fn configured_run_target(
    project: &project::CargoProject,
    pinned_cargo: &pinned_executable::PinnedExecutable,
    authority_rustc: Option<&PinnedRustc>,
    args: &[OsString],
) -> Result<Option<String>, String> {
    let Some(value) =
        project.cargo_config_value(args, "build.target", pinned_cargo, authority_rustc)?
    else {
        return Ok(None);
    };
    let targets = match value {
        serde_json::Value::String(target) => vec![target],
        serde_json::Value::Array(targets) => targets
            .into_iter()
            .map(|target| {
                target.as_str().map(str::to_string).ok_or_else(|| {
                    "Cargo build.target array contains a non-string value".to_string()
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err("Cargo build.target must be a string or string array".to_string());
        }
    };
    match targets.as_slice() {
        [] => Err("Cargo build.target may not be empty".to_string()),
        [target] if !target.is_empty() => Ok(Some(target.clone())),
        [_] => Err("Cargo build.target may not contain an empty target".to_string()),
        _ => Err("cargo fe2o3 run requires exactly one configured build target".to_string()),
    }
}

fn resolve_original_runner(
    project: &project::CargoProject,
    pinned_cargo: &pinned_executable::PinnedExecutable,
    authority_rustc: Option<&PinnedRustc>,
    args: &[OsString],
    target: &str,
) -> Result<Vec<OsString>, String> {
    let environment_name = target_runner_environment_name(target);
    if let Some(environment_runner) = env::var_os(&environment_name)
        && environment_runner.to_str().is_none()
    {
        if has_invocation_config(args) {
            return Err(format!(
                "cannot determine precedence for non-UTF-8 {environment_name} with --config; refusing to bypass a Cargo runner"
            ));
        }
        return parse_runner_bytes(os_bytes(&environment_runner), &environment_name);
    }

    let key = format!("target.{target}.runner");
    if let Some(value) = project.cargo_config_value(args, &key, pinned_cargo, authority_rustc)? {
        return parse_runner_value(value, &key);
    }

    if let Some(serde_json::Value::Object(targets)) =
        project.cargo_config_value(args, "target", pinned_cargo, authority_rustc)?
        && targets.iter().any(|(selector, value)| {
            selector.starts_with("cfg(")
                && value
                    .as_object()
                    .is_some_and(|table| table.contains_key("runner"))
        })
    {
        return Err(format!(
            "cannot safely resolve cfg-selected Cargo runner for target {target}; configure target.{target}.runner explicitly"
        ));
    }
    Ok(Vec::new())
}

fn target_runner_environment_name(target: &str) -> String {
    let normalized = target
        .bytes()
        .map(|byte| match byte {
            b'a'..=b'z' => (byte - b'a' + b'A') as char,
            b'A'..=b'Z' | b'0'..=b'9' => byte as char,
            _ => '_',
        })
        .collect::<String>();
    format!("CARGO_TARGET_{normalized}_RUNNER")
}

fn has_invocation_config(args: &[OsString]) -> bool {
    args.iter()
        .take_while(|argument| *argument != "--")
        .any(|argument| argument == "--config" || os_bytes(argument).starts_with(b"--config="))
}

fn parse_runner_value(value: serde_json::Value, source: &str) -> Result<Vec<OsString>, String> {
    let runner = match value {
        serde_json::Value::String(value) => value
            .split_whitespace()
            .map(OsString::from)
            .collect::<Vec<_>>(),
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(|value| {
                value.as_str().map(OsString::from).ok_or_else(|| {
                    format!("Cargo runner `{source}` contains a non-string argument")
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(format!(
                "Cargo runner `{source}` must be a string or string array"
            ));
        }
    };
    if runner.is_empty() || runner[0].is_empty() {
        return Err(format!("Cargo runner `{source}` may not be empty"));
    }
    Ok(runner)
}

fn parse_runner_bytes(value: &[u8], source: &str) -> Result<Vec<OsString>, String> {
    let runner = value
        .split(|byte| byte.is_ascii_whitespace())
        .filter(|argument| !argument.is_empty())
        .map(|argument| os_string(argument.to_vec()))
        .collect::<Result<Vec<_>, _>>()?;
    if runner.is_empty() {
        return Err(format!("Cargo runner `{source}` may not be empty"));
    }
    Ok(runner)
}

fn reject_recursive_runner(runner: &[OsString]) -> Result<(), String> {
    let Some(program) = runner.first() else {
        return Ok(());
    };
    let current = env::current_exe()
        .map_err(|error| format!("failed to identify cargo-fe2o3 executable: {error}"))?;
    if program == current.as_os_str()
        || program == current.file_name().unwrap_or_default()
        || resolve_runner_program(program).is_some_and(|path| {
            same_executable_file(&path, &current)
                || std::fs::canonicalize(path).ok() == std::fs::canonicalize(&current).ok()
        })
    {
        return Err("refusing recursive cargo-fe2o3 Cargo runner configuration".to_string());
    }
    Ok(())
}

#[cfg(unix)]
fn same_executable_file(left: &Path, right: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    let Ok(left) = std::fs::metadata(left) else {
        return false;
    };
    let Ok(right) = std::fs::metadata(right) else {
        return false;
    };
    left.is_file() && right.is_file() && left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_executable_file(_left: &Path, _right: &Path) -> bool {
    false
}

fn resolve_runner_program(program: &OsStr) -> Option<PathBuf> {
    let path = Path::new(program);
    if path.is_absolute() || os_bytes(program).contains(&b'/') {
        return Some(if path.is_absolute() {
            path.to_path_buf()
        } else {
            env::current_dir().ok()?.join(path)
        });
    }
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(path))
        .find(|candidate| candidate.is_file())
}

fn selected_run_target(args: &[OsString]) -> Result<Option<String>, String> {
    let mut selected = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            break;
        }
        let value = if argument == "--target" {
            index += 1;
            Some(
                args.get(index)
                    .ok_or_else(|| "--target requires an argument".to_string())?
                    .clone(),
            )
        } else {
            split_joined_os_option(argument, "--target")
        };
        if let Some(value) = value {
            let value = value
                .to_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "Cargo run target must be non-empty UTF-8".to_string())?;
            if selected.replace(value.to_string()).is_some() {
                return Err("--target may be specified only once".to_string());
            }
        }
        index += 1;
    }
    Ok(selected)
}

fn split_joined_os_option(argument: &OsStr, option: &str) -> Option<OsString> {
    let bytes = os_bytes(argument);
    let prefix = option.as_bytes();
    if !bytes.starts_with(prefix) || bytes.get(prefix.len()) != Some(&b'=') {
        return None;
    }
    os_string(bytes[prefix.len() + 1..].to_vec()).ok()
}

fn host_rustc_target() -> Result<String, String> {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let mut command = Command::new(rustc);
    command.arg("-vV");
    let output = process_execution::capture_output(&mut command)
        .map_err(|error| format!("failed to query rustc host target: {error}"))?;
    if !output.status.success() {
        return Err(format!("rustc -vV failed with status {}", output.status));
    }
    let output = String::from_utf8(output.stdout)
        .map_err(|_| "rustc -vV output was not UTF-8".to_string())?;
    output
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_string)
        .ok_or_else(|| "rustc -vV output did not contain a host target".to_string())
}

fn pinned_rustc_host_target(rustc: &PinnedRustc) -> Result<String, String> {
    let mut command = rustc
        .executable
        .command()
        .map_err(|error| format!("failed to prepare pinned rustc host query: {error}"))?;
    command.as_command_mut().arg("-vV");
    remove_dynamic_loader_environment(command.as_command_mut());
    rustc
        .lib_tree_directory()
        .inherit_for_child_at(command.as_command_mut(), RUSTC_LIBRARY_CHILD_FD)?;
    command.as_command_mut().env(
        "LD_LIBRARY_PATH",
        format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"),
    );
    let output = command
        .output()
        .map_err(|error| format!("failed to query pinned rustc host target: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "pinned rustc -vV failed with status {}",
            output.status
        ));
    }
    let output = String::from_utf8(output.stdout)
        .map_err(|_| "pinned rustc -vV output was not UTF-8".to_owned())?;
    output
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
        .ok_or_else(|| "pinned rustc -vV output did not contain a host target".to_owned())
}

fn run_application_boundary_frontend(args: &[OsString]) -> ExitCode {
    match application_supervisor::run_frontend(args) {
        Ok(status) => ExitCode::from(binding_wrapper::exit_code(status)),
        Err(error) => {
            eprintln!("cargo-fe2o3 application runner: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_application_supervisor(args: &[OsString]) -> ExitCode {
    match application_supervisor::run_supervisor(
        args,
        run_application_boundary_result,
        application_handoff::application_cleanup_is_pending,
        application_handoff::finish_application_cleanup_supervisor,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cargo-fe2o3 application supervisor: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_application_boundary_result(args: &[OsString]) -> Result<std::process::ExitStatus, String> {
    if args.len() < 7 {
        return Err(
            "runner requires a generation context, original-runner count, and application"
                .to_string(),
        );
    }
    let (application_timeouts, compiler_service) = application_runner_policy(&args[0])?;
    let artifact_path = PathBuf::from(hex_decode_os(&args[1])?);
    let artifact_device = parse_runner_u64(&args[2], "artifact directory device")?;
    let artifact_inode = parse_runner_u64(&args[3], "artifact directory inode")?;
    let artifact_dir = application_handoff::open_expected_generation(
        artifact_path,
        artifact_device,
        artifact_inode,
    )?;
    if args[4].to_str() != Some(application_handoff::RUNNER_EXPECTS_ENVELOPE) {
        return Err(format!(
            "production application runner requires the Worker V3 envelope marker, got {:?}",
            args[4]
        ));
    }
    let runner_count = args[5]
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| format!("invalid original runner argument count {:?}", args[5]))?;
    let application_index = 6_usize
        .checked_add(runner_count)
        .ok_or_else(|| "original runner argument count overflowed".to_string())?;
    let application = args
        .get(application_index)
        .ok_or_else(|| "runner argument count does not leave an application".to_string())?;
    let original_runner = args[6..application_index]
        .iter()
        .map(|argument| hex_decode_os(argument))
        .collect::<Result<Vec<_>, _>>()?;
    if original_runner
        .first()
        .is_some_and(|program| program.is_empty())
    {
        return Err("original Cargo runner executable may not be empty".to_string());
    }
    reject_recursive_runner(&original_runner)?;

    if runner_count != 0 || !original_runner.is_empty() {
        return Err(
            "production Worker V3 runner does not admit an intermediate Cargo runner".to_owned(),
        );
    }
    if compiler_service
        == application_handoff::ApplicationCompilerServiceExposureV1::NativeCustodianRequired
    {
        return application_handoff::native::run(
            &artifact_dir,
            application,
            &args[application_index + 1..],
        );
    }
    let handoff = application_handoff::PinnedApplicationEnvelope::discover(&artifact_dir)?
        .ok_or_else(|| {
            "production Worker V3 runner requires a canonical load envelope".to_owned()
        })?;
    run_application_with_handoff(
        handoff,
        application,
        &args[application_index + 1..],
        application_timeouts,
        compiler_service,
    )
}

fn application_runner_policy(
    context: &OsStr,
) -> Result<
    (
        application_handoff::ApplicationTimeouts,
        application_handoff::ApplicationCompilerServiceExposureV1,
    ),
    String,
> {
    match context.to_str() {
        Some(application_handoff::RUNNER_NATIVE_CUSTODIAN_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::PRODUCTION,
            application_handoff::ApplicationCompilerServiceExposureV1::NativeCustodianRequired,
        )),
        Some(application_handoff::RUNNER_CUSTODIAN_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::PRODUCTION,
            application_handoff::ApplicationCompilerServiceExposureV1::CustodianRequired,
        )),
        Some(application_handoff::RUNNER_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::PRODUCTION,
            application_handoff::ApplicationCompilerServiceExposureV1::Required,
        )),
        #[cfg(feature = "worker-v3-envelope-integration-test-only")]
        Some(application_handoff::RUNNER_ENVELOPE_ONLY_TEST_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::PRODUCTION,
            application_handoff::ApplicationCompilerServiceExposureV1::TestDisabled,
        )),
        #[cfg(any(test, feature = "application-handoff-fault-injection-test-only"))]
        Some(application_handoff::RUNNER_SHORT_TIMEOUT_TEST_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::TEST_SHORT,
            application_handoff::ApplicationCompilerServiceExposureV1::TestDisabled,
        )),
        #[cfg(feature = "application-handoff-fault-injection-test-only")]
        Some(application_handoff::RUNNER_SCHEDULER_TOLERANT_TEST_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::TEST_SCHEDULER_TOLERANT,
            application_handoff::ApplicationCompilerServiceExposureV1::TestDisabled,
        )),
        #[cfg(feature = "application-handoff-adversarial-fixture")]
        Some(application_handoff::RUNNER_FAST_FAILURE_TEST_CONTEXT_VERSION) => Ok((
            application_handoff::ApplicationTimeouts::TEST_FAST_FAILURES,
            application_handoff::ApplicationCompilerServiceExposureV1::TestDisabled,
        )),
        _ => Err(format!(
            "unsupported application runner context {context:?}"
        )),
    }
}

fn run_application_with_handoff(
    mut handoff: application_handoff::PinnedApplicationEnvelope,
    application: &OsStr,
    application_args: &[OsString],
    application_timeouts: application_handoff::ApplicationTimeouts,
    compiler_service: application_handoff::ApplicationCompilerServiceExposureV1,
) -> Result<ExitStatus, String> {
    let compiler_execution_profile = compiler_service
        .is_required()
        .then(fe2o3_compiler_closure_capability::CompilerExecutionClientProfileCapabilityV1::from_production_profile)
        .transpose()
        .map_err(|error| {
            format!("cannot admit the fixed compiler-execution client profile: {error}")
        })?;
    let current_dir = env::current_dir()
        .map_err(|error| format!("failed to resolve application runner directory: {error}"))?;
    let application_path =
        binding_wrapper::resolve_command_executable(application, &current_dir)
            .map_err(|error| format!("failed to resolve application executable: {error}"))?;
    let pinned_application = pinned_executable::PinnedExecutable::open(&application_path)
        .map_err(|error| format!("failed to pin application executable: {error}"))?;
    let sealed_application = pinned_application
        .seal_static_application()
        .map_err(|error| format!("failed to seal application runtime image: {error}"))?;
    let mut child = sealed_application
        .command()
        .map_err(|error| format!("failed to prepare sealed application: {error}"))?;
    child.args(application_args);
    scrub_application_environment(child.as_command_mut());
    let compiler_execution_boundary = compiler_execution_profile
        .as_ref()
        .map(|profile| {
            compiler_execution_boundary::PreparedCompilerExecutionBoundaryV1::prepare_application_verifier(
                profile,
                child.as_command_mut(),
            )
        })
        .transpose()
        .map_err(|error| error.to_string())?;
    let pending_ack = handoff.configure_child_with_timeouts(
        child.as_command_mut(),
        sealed_application.identity_v3(),
        application_timeouts,
        compiler_service,
    )?;
    let process = process_execution::spawn(child.as_command_mut())
        .map_err(|error| format!("failed to launch pinned Cargo application: {error}"))?;
    let mut spawned_ack = match pending_ack.after_spawn(&process) {
        Ok(spawned_ack) => spawned_ack,
        Err(failure) => {
            let (error, cleanup) = failure.into_parts();
            drop(handoff);
            return terminate_application_with_error(process, cleanup, error);
        }
    };
    let compiler_execution_readiness = match compiler_execution_boundary {
        Some(boundary) => {
            match spawned_ack.take_registration_transfer().and_then(
                |(proof, inputs, selected_service)| {
                    boundary
                        .finish_application(
                            spawned_ack.retained_child(),
                            proof,
                            inputs,
                            spawned_ack.deadline(),
                            selected_service,
                        )
                        .map_err(|error| error.to_string())
                },
            ) {
                Ok(readiness) => Some(readiness),
                Err(error) => {
                    drop(handoff);
                    return terminate_application_with_error(
                        process,
                        spawned_ack.into_cleanup(),
                        error.to_string(),
                    );
                }
            }
        }
        None => None,
    };
    let active_handoff = match spawned_ack.await_ack(&process) {
        Ok(active_handoff) => active_handoff,
        Err(failure) => {
            let (error, cleanup) = failure.into_parts();
            drop(handoff);
            return terminate_application_with_error(process, cleanup, error);
        }
    };
    if let Some(readiness) = compiler_execution_readiness.as_ref()
        && let Err(error) = readiness.revalidate()
    {
        drop(handoff);
        return terminate_application_with_error(
            process,
            active_handoff.into_cleanup(),
            error.to_string(),
        );
    }
    if let Err(error) = application_handoff::wait_for_application_exit_without_reaping(&process) {
        drop(handoff);
        return terminate_application_with_error(process, active_handoff.into_cleanup(), error);
    }
    if let Some(readiness) = compiler_execution_readiness.as_ref()
        && let Err(error) = readiness.revalidate()
    {
        drop(handoff);
        return terminate_application_with_error(
            process,
            active_handoff.into_cleanup(),
            error.to_string(),
        );
    }
    // The application retains its currentness token through all descriptor-dependent work.
    // Observe its exit without reaping before reacquiring the runner's token, avoiding a
    // scheduler race while preserving the leader identity for process-group containment.
    if let Err(error) = handoff.validate_retained_currentness() {
        drop(handoff);
        return terminate_application_with_error(process, active_handoff.into_cleanup(), error);
    }
    let cleanup = active_handoff.into_cleanup();
    drop(handoff);
    application_handoff::wait_and_contain_application_group(process, cleanup)
}

fn terminate_application_with_error(
    process: std::process::Child,
    cleanup: application_handoff::ApplicationCleanup,
    error: String,
) -> Result<ExitStatus, String> {
    match application_handoff::terminate_application_group(process, cleanup) {
        Ok(_) => Err(error),
        Err(containment) => Err(format!(
            "{error}; application containment failed: {containment}"
        )),
    }
}

fn scrub_application_environment(child: &mut Command) {
    // The application boundary has no ambient-environment allowlist. The typed production
    // handoff adds only its explicit descriptor protocol values after this reset.
    child.env_clear();
}

fn parse_runner_u64(value: &OsStr, kind: &str) -> Result<u64, String> {
    value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| format!("invalid {kind} {value:?}"))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn hex_decode_os(value: &OsStr) -> Result<OsString, String> {
    let bytes = os_bytes(value);
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("invalid encoded Cargo runner argument {value:?}"));
    }
    let decoded = bytes
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0])?;
            let low = hex_nibble(pair[1])?;
            Ok((high << 4) | low)
        })
        .collect::<Result<Vec<_>, String>>()?;
    os_string(decoded)
}

fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err("Cargo runner argument contains invalid hexadecimal".to_string()),
    }
}

#[cfg(unix)]
fn os_bytes(value: &OsStr) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes()
}

#[cfg(unix)]
fn os_string(value: Vec<u8>) -> Result<OsString, String> {
    use std::os::unix::ffi::OsStringExt;
    Ok(OsString::from_vec(value))
}

#[cfg(not(unix))]
fn os_bytes(value: &OsStr) -> &[u8] {
    value
        .to_str()
        .expect("environment names are UTF-8 off Unix")
        .as_bytes()
}

#[cfg(not(unix))]
fn os_string(value: Vec<u8>) -> Result<OsString, String> {
    String::from_utf8(value)
        .map(OsString::from)
        .map_err(|_| "Cargo option value is not UTF-8".to_string())
}

fn random_build_session() -> Result<fe2o3_artifact_transaction::BuildSession, String> {
    if non_production_reproduction::enabled() {
        return Ok(fe2o3_artifact_transaction::BuildSession::from_bytes(
            non_production_reproduction::deterministic_16(b"build-session"),
        ));
    }
    for _ in 0..8 {
        let mut bytes = [0_u8; 16];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut source| source.read_exact(&mut bytes))
            .map_err(|error| format!("failed to obtain a build-session nonce: {error}"))?;
        let session = fe2o3_artifact_transaction::BuildSession::from_bytes(bytes);
        if session != fe2o3_artifact_transaction::BuildSession::DIRECT {
            return Ok(session);
        }
    }
    Err("failed to obtain a nonzero build-session nonce".to_string())
}

fn reject_preexisting_compiler_environment() -> Result<(), String> {
    for variable in COMPILER_SELECTION_ENVIRONMENT {
        if let Some(value) = env::var_os(variable) {
            if variable.ends_with("WRAPPER") {
                return Err(format!(
                    "cargo fe2o3 cannot compose its binding-identity wrapper with preexisting {variable}={value:?}"
                ));
            }
            return Err(format!(
                "cargo fe2o3 rejects preexisting compiler selection {variable}={value:?}"
            ));
        }
    }
    Ok(())
}

fn reject_configured_compiler_selection(
    project: &project::CargoProject,
    args: &[OsString],
    pinned_cargo: &pinned_executable::PinnedExecutable,
    authority_rustc: Option<&PinnedRustc>,
    authority: bool,
) -> Result<(), String> {
    let authority_rustc =
        if authority {
            Some(authority_rustc.ok_or_else(|| {
                "authority config query has no independently pinned rustc".to_owned()
            })?)
        } else {
            None
        };
    for key in [
        "build.rustc",
        "build.rustc-wrapper",
        "build.rustc-workspace-wrapper",
    ] {
        if let Some(value) = project.cargo_config_value(args, key, pinned_cargo, authority_rustc)? {
            if key.ends_with("wrapper") {
                return Err(format!(
                    "cargo fe2o3 cannot compose its binding-identity wrapper with configured {key}={value}"
                ));
            }
            return Err(format!(
                "cargo fe2o3 rejects configured compiler selection {key}={value}"
            ));
        }
    }
    if let Some(serde_json::Value::Object(configured)) =
        project.cargo_config_value(args, "env", pinned_cargo, authority_rustc)?
    {
        for name in configured.keys() {
            if is_dynamic_loader_environment_name(OsStr::new(name)) {
                return Err(format!(
                    "cargo fe2o3 rejects configured dynamic-loader environment env.{name}"
                ));
            }
        }
    }
    Ok(())
}

fn require_protected_authority_launch(
    protected_release: Option<&authority_release::ProtectedReleaseAdmission>,
) -> Result<(), String> {
    if protected_release.is_some() {
        return Ok(());
    }
    if cfg!(debug_assertions)
        && env::var_os(NON_PRODUCTION_AUTHORITY_VALIDATION_ENV).as_deref() == Some(OsStr::new("1"))
    {
        eprintln!(
            "cargo fe2o3: non-production unprotected authority validation only; no protected release-launch claim"
        );
        return Ok(());
    }
    Err(
        "cargo fe2o3 authority release requires a protected pre-exec launcher/image contract; this build has no admitted release launcher"
            .to_owned(),
    )
}

fn reject_authority_environment_overrides(args: &[OsString]) -> Result<(), String> {
    if has_invocation_config(args) {
        return Err(
            "cargo fe2o3 authority build rejects command-line --config before admission".to_owned(),
        );
    }
    for (name, value) in env::vars_os() {
        let bytes = os_bytes(&name);
        if bytes.starts_with(b"RUSTUP_") {
            return Err(format!(
                "cargo fe2o3 authority build rejects rustup selection channel {name:?}={value:?}"
            ));
        }
        if bytes.starts_with(b"CARGO_REGISTRIES_")
            || bytes.starts_with(b"CARGO_REGISTRY_")
            || bytes.starts_with(b"CARGO_CREDENTIAL")
            || bytes.starts_with(b"CARGO_HTTP_")
            || bytes.starts_with(b"CARGO_NET_")
        {
            return Err(format!(
                "cargo fe2o3 authority build rejects pre-admission helper/configuration channel {name:?}={value:?}"
            ));
        }
        if is_authority_tool_override_environment_name(&name) {
            return Err(format!(
                "cargo fe2o3 authority build rejects tool override {name:?}={value:?}"
            ));
        }
    }
    Ok(())
}

fn reject_authority_config_overrides(
    project: &project::CargoProject,
    args: &[OsString],
    pinned_cargo: &pinned_executable::PinnedExecutable,
    pinned_rustc: &PinnedRustc,
) -> Result<(), String> {
    for key in [
        "source",
        "registries",
        "registry",
        "credential-alias",
        "net",
        "http",
    ] {
        if let Some(value) =
            project.cargo_config_value(args, key, pinned_cargo, Some(pinned_rustc))?
        {
            return Err(format!(
                "cargo fe2o3 authority build rejects configured pre-admission Cargo {key}={value}"
            ));
        }
    }
    reject_authority_configured_environment(project.cargo_config_value(
        args,
        "env",
        pinned_cargo,
        Some(pinned_rustc),
    )?)?;
    match project.cargo_config_value(args, "target", pinned_cargo, Some(pinned_rustc))? {
        Some(serde_json::Value::Object(targets)) => {
            for (target, configuration) in targets {
                let serde_json::Value::Object(configuration) = configuration else {
                    return Err(format!(
                        "cargo fe2o3 authority build cannot inspect configured target.{target}"
                    ));
                };
                for key in ["linker", "runner"] {
                    if let Some(value) = configuration.get(key) {
                        return Err(format!(
                            "cargo fe2o3 authority build rejects configured target.{target}.{key}={value}"
                        ));
                    }
                }
            }
        }
        Some(_) => {
            return Err(
                "cargo fe2o3 authority build cannot inspect configured target table".to_owned(),
            );
        }
        None => {}
    }
    Ok(())
}

fn reject_authority_configured_environment(
    configured: Option<serde_json::Value>,
) -> Result<(), String> {
    match configured {
        Some(serde_json::Value::Object(environment)) if environment.is_empty() => Ok(()),
        Some(serde_json::Value::Object(environment)) => {
            let (name, value) = environment
                .iter()
                .next()
                .expect("nonempty configured Cargo environment has a first entry");
            Err(format!(
                "cargo fe2o3 authority build rejects configured pre-admission Cargo env.{name}={value}"
            ))
        }
        Some(_) => {
            Err("cargo fe2o3 authority build cannot inspect configured Cargo env table".to_owned())
        }
        None => Ok(()),
    }
}

fn authority_rustc_sha256_from_environment() -> Result<[u8; 32], String> {
    authority_sha256_from_environment(AUTHORITY_RUSTC_SHA256_ENV)
}

fn authority_rustc_runtime_sha256_from_environment() -> Result<[u8; 32], String> {
    authority_sha256_from_environment(AUTHORITY_RUSTC_RUNTIME_SHA256_ENV)
}

fn authority_cargo_sha256_from_environment() -> Result<[u8; 32], String> {
    authority_sha256_from_environment(AUTHORITY_CARGO_SHA256_ENV)
}

fn authority_backend_sha256_from_environment() -> Result<[u8; 32], String> {
    authority_sha256_from_environment(AUTHORITY_BACKEND_SHA256_ENV)
}

fn authority_sha256_from_environment(name: &'static str) -> Result<[u8; 32], String> {
    let value =
        env::var_os(name).ok_or_else(|| format!("cargo fe2o3 authority build requires {name}"))?;
    let encoded = os_bytes(&value);
    if encoded.len() != 64 {
        return Err(format!(
            "{name} must be exactly 64 lowercase hexadecimal bytes"
        ));
    }
    let mut digest = [0_u8; 32];
    for (output, pair) in digest.iter_mut().zip(encoded.chunks_exact(2)) {
        let decode = |byte| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        let Some(high) = decode(pair[0]) else {
            return Err(format!(
                "{name} must be exactly 64 lowercase hexadecimal bytes"
            ));
        };
        let Some(low) = decode(pair[1]) else {
            return Err(format!(
                "{name} must be exactly 64 lowercase hexadecimal bytes"
            ));
        };
        *output = (high << 4) | low;
    }
    if digest == [0; 32] {
        return Err(format!("{name} may not be zero"));
    }
    Ok(digest)
}

fn is_authority_tool_override_environment_name(name: &OsStr) -> bool {
    let name = os_bytes(name);
    name.starts_with(b"CARGO_TARGET_")
        && [b"_LINKER".as_slice(), b"_RUNNER"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

fn reject_dynamic_loader_environment() -> Result<(), String> {
    for (name, value) in env::vars_os() {
        if is_dynamic_loader_environment_name(&name) {
            return Err(format!(
                "cargo fe2o3 rejects dynamic-loader injection variable {name:?}={value:?}"
            ));
        }
    }
    Ok(())
}

fn scrub_process_dynamic_loader_environment() {
    let names = env::vars_os()
        .map(|(name, _)| name)
        .filter(|name| is_dynamic_loader_environment_name(name))
        .collect::<Vec<_>>();
    for name in names {
        // SAFETY: cargo-fe2o3 performs this one-time environment normalization before starting
        // any worker or supervisor threads. The variables stay absent for the process lifetime.
        unsafe { env::remove_var(name) };
    }
}

pub(crate) fn is_dynamic_loader_injection_environment_name(name: &OsStr) -> bool {
    is_dynamic_loader_environment_name(name)
}

pub(crate) fn is_dynamic_loader_environment_name(name: &OsStr) -> bool {
    let name = os_bytes(name);
    name.starts_with(b"LD_") || name.starts_with(b"DYLD_") || name == b"GLIBC_TUNABLES"
}

pub(crate) fn remove_dynamic_loader_environment(command: &mut Command) {
    for (name, _) in env::vars_os() {
        if is_dynamic_loader_environment_name(&name) {
            command.env_remove(name);
        }
    }
    for name in [
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "GLIBC_TUNABLES",
    ] {
        command.env_remove(name);
    }
}

fn find_or_build_backend(
    target_dir: &project::PinnedDirectory,
    pinned_cargo: &pinned_executable::PinnedExecutable,
    pinned_rustc: &PinnedRustc,
) -> Result<PathBuf, String> {
    if let Some(path) = env::var_os(BACKEND_ENV) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "{BACKEND_ENV} points to {}, but that file does not exist",
            path.display()
        ));
    }

    let source_root = fe2o3_source_root()?;
    let cargo_fe2o3_executable = env::current_exe()
        .map_err(|error| format!("failed to locate running cargo-fe2o3 executable: {error}"))?;
    let cargo_fe2o3_sha256 = fe2o3_process_identity::measure_executable_sha256_v3(
        &cargo_fe2o3_executable,
    )
    .map_err(|error| format!("failed to measure running cargo-fe2o3 executable: {error}"))?;
    let backend_target = target_dir.open_or_create_child(
        ".fe2o3-backend-build-v1",
        "isolated codegen-backend build directory",
    )?;
    let backend = dylib_path(backend_target.display_path());
    eprintln!("building rustc-codegen-fe2o3 backend...");
    let mut command = pinned_cargo
        .command()
        .map_err(|error| format!("failed to prepare pinned Cargo executable: {error}"))?;
    command
        .as_command_mut()
        .args(["build", "--manifest-path"])
        .arg(source_root.join("Cargo.toml"))
        .args(["--target-dir"])
        .arg(backend_target.fixed_child_path(BACKEND_BUILD_CHILD_FD)?)
        .args(["-p", "rustc-codegen-fe2o3"])
        .current_dir(&source_root)
        // The backend is an internal compiler plugin, not a debuggable user artifact. Limited
        // line-table information keeps the pinned image and its bounded hashing work stable as
        // target-neutral analysis dependencies grow.
        .env("CARGO_PROFILE_DEV_DEBUG", "1")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env(
            "FE2O3_BUILD_CARGO_FE2O3_EXECUTABLE_SHA256_V1",
            cargo_fe2o3_sha256
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        );
    remove_dynamic_loader_environment(command.as_command_mut());
    for name in [
        TARGET_ENV,
        BACKEND_ENV,
        HSACO_DIR_ENV,
        capability_broker::CAPABILITY_BROKER_ENV,
        BINDING_WRAPPER_MODE_ENV,
        BUILD_SESSION_ENV,
        OBSOLETE_CODEGEN_PIPELINE_ENV,
        build_config::QUALIFICATION_ORACLE_ENV,
        build_config::PRODUCTION_BUILD_CONFIG_ENV,
        build_config::PRODUCTION_BUILD_CONFIG_V2_ENV,
        build_config::PRODUCTION_BUILD_EXPECTED_ID_ENV,
        build_config::PRODUCTION_BUILD_EXPECTED_ID_V2_ENV,
        build_config::WORKER_V2_CONFIG_ENV,
        build_config::WORKER_V2_EXPECTED_ID_ENV,
        AUTHORITY_CARGO_SHA256_ENV,
        AUTHORITY_RUSTC_SHA256_ENV,
        AUTHORITY_RUSTC_RUNTIME_SHA256_ENV,
        AUTHORITY_BACKEND_SHA256_ENV,
    ] {
        command.as_command_mut().env_remove(name);
    }
    configure_pinned_rustc_child(command.as_command_mut(), pinned_rustc)?;
    command.as_command_mut().env(
        "LD_LIBRARY_PATH",
        format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"),
    );
    backend_target.inherit_for_child_at(command.as_command_mut(), BACKEND_BUILD_CHILD_FD)?;
    let status = command
        .status()
        .map_err(|error| format!("failed to build rustc-codegen-fe2o3: {error}"))?;

    if !status.success() {
        return Err("failed to build rustc-codegen-fe2o3".to_string());
    }

    if backend.is_file() {
        Ok(backend)
    } else {
        Err(format!(
            "backend build succeeded, but {} was not produced",
            backend.display()
        ))
    }
}

struct PinnedRustc {
    executable: pinned_executable::PinnedExecutable,
    lib_tree: RustcLibTree,
}

enum RustcLibTree {
    Ordinary(project::PinnedDirectory),
    Authority(rustc_lib_tree::PinnedRustcLibTree),
}

impl PinnedRustc {
    fn lib_tree_directory(&self) -> &project::PinnedDirectory {
        match &self.lib_tree {
            RustcLibTree::Ordinary(directory) => directory,
            RustcLibTree::Authority(lib_tree) => lib_tree.directory(),
        }
    }

    fn lib_tree_sha256(&self) -> &[u8; 32] {
        match &self.lib_tree {
            RustcLibTree::Ordinary(_) => &[0_u8; 32],
            RustcLibTree::Authority(lib_tree) => lib_tree.sha256(),
        }
    }

    fn assert_lib_tree_unmutated(&self) -> Result<(), String> {
        match &self.lib_tree {
            RustcLibTree::Ordinary(_) => Ok(()),
            RustcLibTree::Authority(lib_tree) => lib_tree.assert_unmutated(),
        }
    }

    fn revalidate_lib_tree(&self) -> Result<(), String> {
        match &self.lib_tree {
            RustcLibTree::Ordinary(_) => Ok(()),
            RustcLibTree::Authority(lib_tree) => lib_tree.revalidate(),
        }
    }
}

fn require_absolute_authority_tool_path(value: &OsStr, name: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!(
            "cargo fe2o3 authority build requires {name} to name an absolute executable path"
        ));
    }
    Ok(path)
}

fn reject_authority_rustup_proxy(path: &Path, tool: &str) -> Result<(), String> {
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        format!(
            "failed to inspect authority {tool} executable {}: {error}",
            path.display()
        )
    })?;
    if path.file_name() == Some(OsStr::new("rustup"))
        || canonical.file_name() == Some(OsStr::new("rustup"))
    {
        return Err(format!(
            "cargo fe2o3 authority {tool} path resolves to a rustup proxy; rustup is never executed during authority selection"
        ));
    }
    Ok(())
}

fn pin_authority_rustc(
    invocation_directory: &Path,
    expected_executable_sha256: [u8; 32],
    expected_lib_tree_sha256: [u8; 32],
) -> Result<PinnedRustc, String> {
    let declaration = env::var_os(AUTHORITY_RUSTC_PATH_ENV).ok_or_else(|| {
        format!("cargo fe2o3 authority build requires {AUTHORITY_RUSTC_PATH_ENV}")
    })?;
    let declared = require_absolute_authority_tool_path(&declaration, AUTHORITY_RUSTC_PATH_ENV)?;
    let canonical = std::fs::canonicalize(&declared).map_err(|error| {
        format!(
            "failed to inspect authority rustc executable {}: {error}",
            declared.display()
        )
    })?;
    if canonical.file_name() == Some(OsStr::new("rustup")) {
        return Err(
            "cargo fe2o3 authority rustc path resolves to a rustup proxy; rustup is never executed during authority selection"
                .to_owned(),
        );
    }
    if canonical.parent().is_none() || !invocation_directory.is_absolute() {
        return Err("authority rustc path is not canonicalizable".to_owned());
    }
    let source_executable = pinned_executable::PinnedExecutable::open(&canonical)
        .map_err(|error| format!("failed to pin authority rustc executable: {error}"))?;
    if source_executable.sha256() != &expected_executable_sha256 {
        return Err(format!(
            "cargo fe2o3 authority rustc does not match {AUTHORITY_RUSTC_SHA256_ENV}"
        ));
    }
    let lib_tree_directory = rustc_lib_tree_directory(&canonical)?;
    let lib_tree = rustc_lib_tree::PinnedRustcLibTree::pin(lib_tree_directory)?;
    if lib_tree.sha256() != &expected_lib_tree_sha256 {
        return Err(format!(
            "cargo fe2o3 authority rustc toolchain lib tree does not match {AUTHORITY_RUSTC_RUNTIME_SHA256_ENV}"
        ));
    }
    let executable = source_executable
        .seal_executable_image()
        .map_err(|error| format!("failed to seal authority rustc executable: {error}"))?;
    Ok(PinnedRustc {
        executable,
        lib_tree: RustcLibTree::Authority(lib_tree),
    })
}

fn pin_default_rustc(project: &project::CargoProject) -> Result<PinnedRustc, String> {
    let declared = OsStr::new("rustc");
    let resolved = binding_wrapper::resolve_command_executable(
        declared,
        &project.invocation_dir().child_path(),
    )
    .map_err(|error| format!("failed to resolve default rustc executable: {error}"))?;
    let canonical = std::fs::canonicalize(&resolved)
        .map_err(|error| format!("failed to inspect default rustc executable: {error}"))?;
    let rustc_path = if canonical.file_name() == Some(OsStr::new("rustup")) {
        resolve_rustup_toolchain_rustc(&canonical, project)?
    } else {
        canonical
    };
    let executable = pinned_executable::PinnedExecutable::open(&rustc_path)
        .map_err(|error| format!("failed to pin default rustc executable: {error}"))?;
    let lib_tree = rustc_lib_tree_directory(&rustc_path)?;
    Ok(PinnedRustc {
        executable,
        lib_tree: RustcLibTree::Ordinary(lib_tree),
    })
}

fn pin_default_cargo(
    declaration: &OsStr,
    invocation_directory: &Path,
) -> Result<pinned_executable::PinnedExecutable, String> {
    let resolved = binding_wrapper::resolve_command_executable(declaration, invocation_directory)
        .map_err(|error| format!("failed to resolve Cargo executable: {error}"))?;
    let canonical = std::fs::canonicalize(&resolved)
        .map_err(|error| format!("failed to inspect Cargo executable: {error}"))?;
    let cargo_path = if canonical.file_name() == Some(OsStr::new("rustup")) {
        resolve_rustup_toolchain_tool(&canonical, invocation_directory, "cargo")?
    } else {
        canonical
    };
    pinned_executable::PinnedExecutable::open(&cargo_path)
        .map_err(|error| format!("failed to pin Cargo executable: {error}"))
}

fn pin_default_clippy_driver(
    invocation_directory: &Path,
) -> Result<pinned_executable::PinnedExecutable, String> {
    let resolved = binding_wrapper::resolve_command_executable(
        OsStr::new("clippy-driver"),
        invocation_directory,
    )
    .map_err(|error| format!("failed to resolve clippy-driver executable: {error}"))?;
    let canonical = std::fs::canonicalize(&resolved)
        .map_err(|error| format!("failed to inspect clippy-driver executable: {error}"))?;
    let driver_path = if canonical.file_name() == Some(OsStr::new("rustup")) {
        resolve_rustup_toolchain_tool(&canonical, invocation_directory, "clippy-driver")?
    } else {
        canonical
    };
    if driver_path.file_name() != Some(OsStr::new("clippy-driver")) {
        return Err(format!(
            "resolved Clippy driver has unexpected name: {}",
            driver_path.display()
        ));
    }
    pinned_executable::PinnedExecutable::open(&driver_path)
        .and_then(|driver| driver.seal_executable_image())
        .map_err(|error| format!("failed to pin clippy-driver executable: {error}"))
}

fn rustc_lib_tree_directory(rustc_path: &Path) -> Result<project::PinnedDirectory, String> {
    let executable_directory = rustc_path
        .parent()
        .ok_or_else(|| "default rustc executable has no parent directory".to_owned())?;
    let toolchain_library = executable_directory
        .parent()
        .map(|parent| parent.join("lib"))
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| executable_directory.to_path_buf());
    project::PinnedDirectory::open_existing(
        toolchain_library,
        "pinned rustc toolchain lib-tree directory",
    )
}

fn resolve_rustup_toolchain_rustc(
    rustup_proxy: &Path,
    project: &project::CargoProject,
) -> Result<PathBuf, String> {
    let pinned = pinned_executable::PinnedExecutable::open(rustup_proxy)
        .map_err(|error| format!("failed to pin rustup proxy: {error}"))?;
    let mut command = pinned
        .command()
        .map_err(|error| format!("failed to prepare pinned rustup proxy: {error}"))?;
    command
        .as_command_mut()
        .arg0("rustup")
        .args(["which", "rustc"])
        .current_dir(project.invocation_dir().child_path());
    remove_dynamic_loader_environment(command.as_command_mut());
    let output = command
        .output()
        .map_err(|error| format!("failed to resolve rustup toolchain rustc: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "pinned rustup could not resolve the active rustc: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut path = output.stdout;
    while matches!(path.last(), Some(b'\n' | b'\r')) {
        path.pop();
    }
    if path.is_empty() || path.contains(&b'\n') || path.contains(&0) {
        return Err("pinned rustup returned a noncanonical rustc path".to_owned());
    }
    let path = PathBuf::from(os_string(path)?);
    if !path.is_absolute() {
        return Err("pinned rustup returned a relative rustc path".to_owned());
    }
    Ok(path)
}

fn resolve_rustup_toolchain_tool(
    rustup_proxy: &Path,
    invocation_directory: &Path,
    tool: &str,
) -> Result<PathBuf, String> {
    let pinned = pinned_executable::PinnedExecutable::open(rustup_proxy)
        .map_err(|error| format!("failed to pin rustup proxy: {error}"))?;
    let mut command = pinned
        .command()
        .map_err(|error| format!("failed to prepare pinned rustup proxy: {error}"))?;
    command
        .as_command_mut()
        .arg0("rustup")
        .args(["which", tool])
        .current_dir(invocation_directory);
    remove_dynamic_loader_environment(command.as_command_mut());
    let output = command
        .output()
        .map_err(|error| format!("failed to resolve rustup toolchain {tool}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "pinned rustup could not resolve the active {tool}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let path = parse_rustup_tool_path(&output.stdout, tool)?;
    Ok(path)
}

fn parse_rustup_tool_path(stdout: &[u8], tool: &str) -> Result<PathBuf, String> {
    let mut path = stdout;
    if path.ends_with(b"\n") {
        path = &path[..path.len() - 1];
        if path.ends_with(b"\r") {
            path = &path[..path.len() - 1];
        }
    }
    if path.is_empty() || path.contains(&b'\n') || path.contains(&b'\r') || path.contains(&0) {
        return Err(format!("pinned rustup returned a noncanonical {tool} path"));
    }
    let path = PathBuf::from(os_string(path.to_vec())?);
    if !path.is_absolute() {
        return Err(format!("pinned rustup returned a relative {tool} path"));
    }
    Ok(path)
}

fn configure_pinned_rustc_child(command: &mut Command, rustc: &PinnedRustc) -> Result<(), String> {
    let descriptor_path = PathBuf::from(format!("/proc/self/fd/{RUSTC_CHILD_FD}"));
    match std::fs::symlink_metadata(&descriptor_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(format!(
                "reserved rustc child descriptor {RUSTC_CHILD_FD} is already in use"
            ));
        }
        Err(error) => {
            return Err(format!(
                "failed to inspect reserved rustc child descriptor {RUSTC_CHILD_FD}: {error}"
            ));
        }
    }
    let mut source = rustc
        .executable
        .try_clone_for_transfer()
        .map_err(|error| format!("failed to retain pinned rustc executable: {error}"))?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind pinned rustc executable: {error}"))?;
    let image = rustix::fs::memfd_create(
        "fe2o3-pinned-rustc",
        rustix::fs::MemfdFlags::CLOEXEC | rustix::fs::MemfdFlags::ALLOW_SEALING,
    )
    .map(File::from)
    .map_err(|error| format!("failed to allocate sealed rustc image: {error}"))?;
    let mut image = image;
    let copied = std::io::copy(&mut source, &mut image)
        .map_err(|error| format!("failed to snapshot pinned rustc executable: {error}"))?;
    if copied != rustc.executable.size() {
        return Err(format!(
            "pinned rustc snapshot copied {copied} bytes instead of {}",
            rustc.executable.size()
        ));
    }
    image
        .set_permissions(std::fs::Permissions::from_mode(0o500))
        .map_err(|error| format!("failed to make sealed rustc image executable: {error}"))?;
    // A same-UID observer can transiently retain a writable `/proc` alias. A
    // bounded wait is safe here because the sealed bytes are independently
    // reauthenticated against the parent pin immediately below.
    fe2o3_process_identity::seal_immutable_memfd_v1(
        &image,
        fe2o3_process_identity::ImmutableMemfdBusyPolicyV1::BoundedExternalObserverQuiescence,
    )
    .map_err(|error| format!("failed to seal pinned rustc image: {error}"))?;
    image
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind sealed rustc image: {error}"))?;
    let verified = pinned_executable::PinnedExecutable::from_transferred_file(
        image
            .try_clone()
            .map_err(|error| format!("failed to verify sealed rustc image: {error}"))?,
        PathBuf::from("<sealed parent-pinned rustc image>"),
    )
    .map_err(|error| format!("failed to verify sealed rustc image: {error}"))?;
    if verified.sha256() != rustc.executable.sha256() {
        return Err("sealed rustc image differs from the parent-pinned compiler".to_owned());
    }
    let metadata = image
        .metadata()
        .map_err(|error| format!("failed to inspect sealed rustc image: {error}"))?;
    let expected = (metadata.dev(), metadata.ino(), metadata.mode());
    // SAFETY: the callback performs descriptor-only operations and owns `image` through exec.
    unsafe {
        command.pre_exec(move || {
            let installed = rustix::io::fcntl_dupfd_cloexec(&image, RUSTC_CHILD_FD)
                .map_err(std::io::Error::from)?;
            if installed.as_raw_fd() != RUSTC_CHILD_FD {
                return Err(std::io::Error::from_raw_os_error(
                    rustix::io::Errno::BUSY.raw_os_error(),
                ));
            }
            let descriptor = BorrowedFd::borrow_raw(RUSTC_CHILD_FD);
            let stat = rustix::fs::fstat(descriptor).map_err(std::io::Error::from)?;
            if (stat.st_dev, stat.st_ino, stat.st_mode) != expected {
                return Err(std::io::Error::from_raw_os_error(
                    rustix::io::Errno::STALE.raw_os_error(),
                ));
            }
            rustix::io::fcntl_setfd(descriptor, rustix::io::FdFlags::empty())
                .map_err(std::io::Error::from)?;
            let _ = installed.into_raw_fd();
            Ok(())
        });
    }
    rustc
        .lib_tree_directory()
        .inherit_for_child_at(command, RUSTC_LIBRARY_CHILD_FD)?;
    command.env("RUSTC", descriptor_path);
    Ok(())
}

fn configure_authority_cargo_child(
    command: &mut Command,
    rustc: &PinnedRustc,
) -> Result<(), String> {
    #[cfg(debug_assertions)]
    let fixture_environment = if env::var_os(NON_PRODUCTION_AUTHORITY_VALIDATION_ENV).as_deref()
        == Some(OsStr::new("1"))
    {
        env::vars_os()
            .filter(|(name, _)| os_bytes(name).starts_with(b"FE2O3_TEST_"))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    command.env_clear();
    command.env("LANG", "C");
    #[cfg(debug_assertions)]
    command.envs(fixture_environment);
    configure_pinned_rustc_child(command, rustc)?;
    command.env(
        "LD_LIBRARY_PATH",
        format!("/proc/self/fd/{RUSTC_LIBRARY_CHILD_FD}"),
    );
    Ok(())
}

fn fe2o3_source_root() -> Result<PathBuf, String> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source_root = crate_root.parent().and_then(Path::parent).ok_or_else(|| {
        format!(
            "cargo-fe2o3 was built from an unsupported source layout: {}",
            crate_root.display()
        )
    })?;
    let manifest = source_root.join("Cargo.toml");
    if !manifest.is_file() {
        return Err(format!(
            "fe2o3 source manifest is unavailable at {}; set {BACKEND_ENV} to a built backend",
            manifest.display()
        ));
    }
    std::fs::canonicalize(source_root)
        .map_err(|error| format!("failed to resolve fe2o3 source root: {error}"))
}

fn dylib_path(target_dir: &Path) -> PathBuf {
    target_dir.join("debug/librustc_codegen_fe2o3.so")
}

fn find_workspace_root() -> Result<PathBuf, String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command.args(["locate-project", "--workspace", "--message-format", "json"]);
    let output = process_execution::capture_output(&mut command)
        .map_err(|error| format!("failed to run cargo locate-project: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "could not find Cargo project/workspace root: {}",
            stderr.trim()
        ));
    }

    let record: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("failed to parse cargo locate-project output: {error}"))?;
    let manifest = record
        .get("root")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "cargo locate-project output did not contain a string `root`".to_string())?;
    let root = Path::new(manifest)
        .parent()
        .ok_or_else(|| format!("Cargo manifest has no parent directory: {manifest}"))?;

    std::fs::canonicalize(root)
        .map_err(|error| format!("failed to resolve Cargo project/workspace root: {error}"))
}

fn print_help() {
    eprintln!(
        "usage: cargo fe2o3 <command>\n\ncommands:\n  authority release   run an authority build through the protected self-launch boundary\n  doctor              report direct-KFD runtime, compiler, and optional tool readiness\n  engineering hsaco   emit a non-authoritative measured source-to-HSACO observation\n  check               check host targets with compiler-derived kernel bindings\n  clippy              lint host targets with compiler-derived kernel bindings\n  test --all-targets  run trusted binding-aware host tests; no artifact/GPU authority\n  build               build with the fe2o3 rustc backend\n  run                 run with the fe2o3 rustc backend\n  examples            validate or query the example regression manifest\n  clean [--dry-run]   remove guarded fe2o3-owned target artifacts\n  inspect             inspect bounded artifact, HSACO, or observation metadata\n  sanitize            plan or execute bounded ROCgdb precise-memory diagnostics\n  debug               plan or execute bounded batch/interactive ROCgdb sessions\n  profile             plan or authorize bounded rocprofv3 collection",
    );
    eprintln!(
        "  engineering rustc-runtime --lib-tree <path>  observe the canonical rustc library-tree digest; no authority"
    );
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
#[path = "tests/production_source_isa_unit_matrix_v1.rs"]
mod production_source_isa_unit_matrix_v1;

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
#[path = "tests/production_source_isa_characteristic_matrix_v2.rs"]
mod production_source_isa_characteristic_matrix_v2;

#[cfg(test)]
mod tests {
    mod application_custody;

    use super::{
        BindingHostMode, MAX_SOURCE_ISA_COLLECTION_STDERR_LINE_BYTES_V1, ObserverFinishOnDropV1,
        TARGET_ENV, aggregate_post_spawn_results,
        append_compiler_execution_profile_semantic_configuration, application_runner_policy,
        binding_host_target_key, clear_cargo_unit_identity_names,
        configure_production_cargo_tool_environment, configure_production_target_environment,
        inject_binding_host_test_custody, is_cargo_target_runner_environment_name,
        normalize_invocation, parse_rustup_tool_path, prepare_binding_clippy_args,
        reject_authority_configured_environment, reject_authority_rustup_proxy,
        reject_binding_test_invocation_config, reject_obsolete_codegen_pipeline,
        selected_run_target, source_isa_characteristic_broker_hex,
        source_isa_characteristic_broker_hex_length, source_isa_collection_hex,
        source_isa_collection_hex_length, validate_production_cargo_inputs,
        validate_production_compilation_environment,
    };
    use crate::observer_telemetry;
    use crate::pinned_executable_test_directory::TestDirectory;
    use std::ffi::{OsStr, OsString};
    use std::io::{self, Write};
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static OBSERVER_FINISH_CALLS: AtomicUsize = AtomicUsize::new(0);

    struct FailingWriter {
        bytes_before_failure: usize,
        bytes_written: usize,
    }

    impl Write for FailingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if self.bytes_before_failure == 0 {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "injected"));
            }
            let written = self.bytes_before_failure.min(buffer.len());
            self.bytes_before_failure -= written;
            self.bytes_written += written;
            Ok(written)
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "injected"))
        }
    }

    fn finish_with_failing_observer_writer(_value: u8, _observer_enabled: bool) {
        OBSERVER_FINISH_CALLS.fetch_add(1, Ordering::SeqCst);
        let mut writer = FailingWriter {
            bytes_before_failure: 3,
            bytes_written: 0,
        };
        let _ = observer_telemetry::write_line_to(
            &mut writer,
            format_args!("source-isa-observation-collection-v1"),
        );
    }

    fn command_environment<'command>(
        command: &'command Command,
        name: &str,
    ) -> Option<&'command OsStr> {
        command
            .get_envs()
            .find(|(candidate, _)| *candidate == OsStr::new(name))
            .and_then(|(_, value)| value)
    }

    #[test]
    fn normalizes_direct_and_cargo_subcommand_invocations() {
        for command in [
            "authority",
            "doctor",
            "check",
            "clippy",
            "test",
            "build",
            "run",
            "examples",
            "clean",
            "inspect",
            "sanitize",
            "debug",
            "profile",
        ] {
            let direct = vec![OsString::from(command), OsString::from("argument")];
            let cargo = vec![
                OsString::from("fe2o3"),
                OsString::from(command),
                OsString::from("argument"),
            ];

            assert_eq!(normalize_invocation(direct.clone()), direct);
            assert_eq!(normalize_invocation(cargo), direct);
        }
    }

    #[test]
    fn binding_host_modes_select_only_the_exact_cargo_command() {
        assert_eq!(BindingHostMode::Check.cargo_command(), "check");
        assert!(!BindingHostMode::Check.executes_tests());
        assert_eq!(BindingHostMode::Clippy.cargo_command(), "clippy");
        assert!(!BindingHostMode::Clippy.executes_tests());
        assert_eq!(BindingHostMode::Test.cargo_command(), "test");
        assert!(BindingHostMode::Test.executes_tests());
    }

    #[test]
    fn binding_clippy_arguments_follow_the_clippy_driver_protocol() {
        let args = [
            OsString::from("--locked"),
            OsString::from("--no-deps"),
            OsString::from("-p"),
            OsString::from("managed"),
            OsString::from("--"),
            OsString::from("-D"),
            OsString::from("warnings"),
        ];
        let (cargo, clippy) = prepare_binding_clippy_args(&args).unwrap();
        assert_eq!(cargo, ["--locked", "-p", "managed"].map(OsString::from));
        assert_eq!(
            clippy,
            "--no-deps__CLIPPY_HACKERY__-D__CLIPPY_HACKERY__warnings__CLIPPY_HACKERY__"
        );

        assert!(prepare_binding_clippy_args(&[OsString::from("--fix")]).is_err());
        assert!(
            prepare_binding_clippy_args(&[
                OsString::from("--"),
                OsString::from("bad__CLIPPY_HACKERY__argument"),
            ])
            .is_err()
        );
    }

    #[test]
    fn binding_host_test_requires_the_closed_cargo_argument_profile() {
        for name in [
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
            "CARGO_TARGET_CFG_UNIX_RUNNER",
            "CARGO_TARGET_A_RUNNER",
        ] {
            assert!(is_cargo_target_runner_environment_name(OsStr::new(name)));
        }
        for name in [
            "CARGO_TARGET__RUNNER",
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER",
            "FE2O3_CARGO_TARGET_X_RUNNER",
        ] {
            assert!(!is_cargo_target_runner_environment_name(OsStr::new(name)));
        }

        for args in [
            vec![
                OsString::from("--all-targets"),
                OsString::from("--config=target.host.runner='hostile'"),
            ],
            vec![
                OsString::from("--all-targets"),
                OsString::from("--config"),
                OsString::from("target.host.runner='hostile'"),
            ],
            vec![OsString::from("--all-targets"), OsString::from("-Z")],
            vec![
                OsString::from("--all-targets"),
                OsString::from("-Zconfig-include=hostile.toml"),
            ],
            vec![OsString::from("--all-targets"), OsString::from("--doc")],
            vec![OsString::from("--all-targets"), OsString::from("--no-run")],
            vec![OsString::from("-p"), OsString::from("managed")],
        ] {
            assert!(reject_binding_test_invocation_config(&args).is_err());
        }
        assert!(
            reject_binding_test_invocation_config(&[
                OsString::from("--all-targets"),
                OsString::from("--"),
                OsString::from("--config=ordinary-test-argument"),
                OsString::from("-Zordinary-test-argument"),
            ])
            .is_ok()
        );
    }

    #[test]
    fn binding_host_test_explicit_selectors_are_value_aware_and_component_only() {
        for args in [
            vec!["--locked", "--lib"],
            vec!["--offline", "--test", "reference"],
            vec![
                "--manifest-path",
                "path/Cargo.toml",
                "--lib",
                "--message-format=json",
            ],
            vec![
                "--test",
                "reference",
                "--",
                "-Z",
                "unstable-options",
                "--format=json",
            ],
            vec!["--all-targets", "--lib"],
        ] {
            let args = args.into_iter().map(OsString::from).collect::<Vec<_>>();
            assert!(
                reject_binding_test_invocation_config(&args).is_ok(),
                "{args:?}"
            );
        }
        for args in [
            vec!["--lib", "--test", "reference"],
            vec!["--lib", "--lib"],
            vec!["--test"],
            vec!["--test", ""],
            vec!["--test", "*"],
            vec!["--test", "ref?"],
            vec!["--test", "[a]"],
            vec!["--test=reference"],
            vec!["--tests"],
            vec!["--bin", "main"],
            vec!["--package", "--lib"],
            vec!["--unknown", "--lib"],
            vec!["--", "--lib"],
            vec!["--lib", "substring-filter"],
            vec!["--lib", "--config", "a=b"],
            vec!["--test", "reference", "-Zunstable-options"],
            vec!["--lib", "--doc"],
            vec!["--test", "reference", "--no-run"],
        ] {
            let args = args.into_iter().map(OsString::from).collect::<Vec<_>>();
            assert!(
                reject_binding_test_invocation_config(&args).is_err(),
                "{args:?}"
            );
        }
    }

    #[test]
    fn binding_host_test_custody_precedes_test_binary_arguments() {
        let mut args = vec![
            OsString::from("--all-targets"),
            OsString::from("--"),
            OsString::from("--ignored"),
        ];
        inject_binding_host_test_custody(
            &mut args,
            "x86_64-unknown-linux-gnu",
            std::path::Path::new("/proc/self/fd/200"),
        )
        .unwrap();
        let separator = args.iter().position(|argument| argument == "--").unwrap();
        assert_eq!(args[separator + 1], "--ignored");
        let cargo_side = &args[..separator];
        assert!(cargo_side.windows(2).any(|pair| {
            pair[0] == "--config"
                && pair[1]
                    .to_string_lossy()
                    .starts_with("target.\"x86_64-unknown-linux-gnu\".runner=")
        }));
        for protected in [
            "build.rustc=",
            "build.rustc-workspace-wrapper=",
            "env.RUSTDOC.value=",
            "env.LD_PRELOAD.value=",
            "env.LD_AUDIT.value=",
            "env.GLIBC_TUNABLES.value=",
            "env.LD_LIBRARY_PATH.value=",
        ] {
            assert!(
                cargo_side
                    .iter()
                    .any(|argument| { argument.to_string_lossy().starts_with(protected) })
            );
        }
    }

    #[test]
    fn binding_host_target_is_bounded_and_toml_quoted() {
        assert_eq!(
            binding_host_target_key("x86_64-unknown-linux-gnu").unwrap(),
            "\"x86_64-unknown-linux-gnu\""
        );
        for hostile in [
            "",
            "host.target",
            "host\".runner='hostile'",
            "host\nrunner",
            "host target",
        ] {
            assert!(binding_host_target_key(hostile).is_err(), "{hostile:?}");
        }
        assert!(binding_host_target_key(&"a".repeat(129)).is_err());
    }

    #[test]
    fn rustup_cargo_output_requires_one_absolute_path_line() {
        assert_eq!(
            parse_rustup_tool_path(b"/toolchain/bin/cargo\n", "cargo")
                .expect("absolute LF-terminated path"),
            std::path::PathBuf::from("/toolchain/bin/cargo")
        );
        assert_eq!(
            parse_rustup_tool_path(b"/toolchain/bin/cargo\r\n", "cargo")
                .expect("absolute CRLF-terminated path"),
            std::path::PathBuf::from("/toolchain/bin/cargo")
        );
        for malformed in [
            b"".as_slice(),
            b"cargo\n".as_slice(),
            b"./cargo\n".as_slice(),
            b"/first/cargo\n/second/cargo\n".as_slice(),
            b"/toolchain/bin/cargo\n\n".as_slice(),
            b"/toolchain/bin/car\0go\n".as_slice(),
            b"/toolchain/bin/cargo\r".as_slice(),
        ] {
            assert!(
                parse_rustup_tool_path(malformed, "cargo").is_err(),
                "accepted malformed rustup output: {malformed:?}"
            );
        }
    }

    #[test]
    fn authority_cargo_selection_never_resolves_a_rustup_proxy() {
        let directory = TestDirectory::new();
        let rustup = directory.path().join("rustup");
        std::fs::write(&rustup, b"not executed").expect("write proxy fixture");
        let error = reject_authority_rustup_proxy(&rustup, "Cargo")
            .expect_err("direct rustup proxy must be rejected");
        assert!(error.contains("rustup is never executed"), "{error}");

        let cargo_alias = directory.path().join("cargo");
        std::os::unix::fs::symlink(&rustup, &cargo_alias).expect("create cargo proxy alias");
        let error = reject_authority_rustup_proxy(&cargo_alias, "Cargo")
            .expect_err("cargo alias to rustup must be rejected");
        assert!(error.contains("rustup is never executed"), "{error}");
    }

    #[test]
    fn binding_check_parent_clears_ambient_cargo_unit_identity() {
        let mut command = Command::new("cargo");
        command
            .env("CARGO_PRIMARY_PACKAGE", "attacker")
            .env("CARGO_MANIFEST_DIR", "/attacker")
            .env("CARGO_PKG_NAME", "attacker")
            .env("CARGO_PKG_VERSION", "attacker")
            .env("CARGO_TARGET_DIR", "/retained");
        clear_cargo_unit_identity_names(
            &mut command,
            [
                OsString::from("CARGO_PRIMARY_PACKAGE"),
                OsString::from("CARGO_MANIFEST_DIR"),
                OsString::from("CARGO_PKG_NAME"),
                OsString::from("CARGO_PKG_VERSION"),
            ],
        );
        for name in [
            "CARGO_PRIMARY_PACKAGE",
            "CARGO_MANIFEST_DIR",
            "CARGO_PKG_NAME",
            "CARGO_PKG_VERSION",
        ] {
            assert_eq!(command_environment(&command, name), None);
        }
        assert_eq!(
            command_environment(&command, "CARGO_TARGET_DIR"),
            Some(OsStr::new("/retained"))
        );
    }

    #[test]
    fn production_is_the_unselected_route_and_has_no_selector() {
        assert!(validate_production_compilation_environment(None).is_ok());
        assert!(
            validate_production_compilation_environment(Some(OsStr::new("kernel-ir-v1")))
                .expect_err("qualification selector must fail")
                .contains("production compilation has no selector")
        );
    }

    #[test]
    fn obsolete_pipeline_environment_is_rejected() {
        assert!(reject_obsolete_codegen_pipeline(None).is_ok());
        let reason = reject_obsolete_codegen_pipeline(Some(OsStr::new("production-v1")))
            .expect_err("obsolete pipeline environment must fail");
        assert!(reason.contains("FE2O3_CODEGEN_PIPELINE has been removed"));
        assert!(reason.contains("FE2O3_QUALIFICATION_ORACLE_V1"));
    }

    #[test]
    fn production_target_profile_is_exact_and_parent_owned() {
        let mut command = Command::new("cargo");
        command
            .env("RUSTFLAGS", "-Ctarget-cpu=attacker")
            .env("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-feature=+attacker")
            .env(
                fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_ENV_V1,
                "attacker",
            );
        configure_production_target_environment(
            &mut command,
            fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942,
        );
        assert_eq!(command_environment(&command, "RUSTFLAGS"), None);
        assert_eq!(
            command_environment(&command, TARGET_ENV),
            Some(OsStr::new(
                fe2o3_amd_target::PRODUCTION_GFX942_DEVICE_TARGET_V1
            ))
        );
        assert_eq!(
            command_environment(&command, "CARGO_ENCODED_RUSTFLAGS"),
            None
        );
        assert_eq!(
            command_environment(
                &command,
                fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_ENV_V1,
            ),
            Some(OsStr::new(
                fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_V1,
            ))
        );

        let mut gfx950 = Command::new("cargo");
        gfx950.env(TARGET_ENV, "gfx942:xnack-").env(
            fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_ENV_V1,
            fe2o3_amd_target::PRODUCTION_GFX942_CARGO_RUSTFLAGS_V1,
        );
        configure_production_target_environment(
            &mut gfx950,
            fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx950,
        );
        assert_eq!(
            command_environment(&gfx950, TARGET_ENV),
            Some(OsStr::new(
                fe2o3_amd_target::PRODUCTION_GFX950_DEVICE_TARGET_V1
            ))
        );
        assert_eq!(
            command_environment(
                &gfx950,
                fe2o3_amd_target::PRODUCTION_GFX950_CARGO_RUSTFLAGS_ENV_V1,
            ),
            Some(OsStr::new(
                fe2o3_amd_target::PRODUCTION_GFX950_CARGO_RUSTFLAGS_V1,
            ))
        );
    }

    #[test]
    fn production_cargo_tool_environment_replaces_the_caller_path() {
        let mut command = Command::new("cargo");
        command.env("PATH", "/attacker/bin");

        configure_production_cargo_tool_environment(&mut command);

        assert_eq!(
            command_environment(&command, "PATH"),
            Some(OsStr::new("/usr/bin"))
        );
    }

    #[test]
    fn authority_rejects_configured_cargo_environment_before_child_execution() {
        let forced_path = serde_json::json!({
            "PATH": {"force": true, "value": "/attacker/bin"}
        });
        let error = reject_authority_configured_environment(Some(forced_path))
            .expect_err("configured PATH must fail closed");
        assert!(error.contains("env.PATH"), "{error}");
        assert!(error.contains("/attacker/bin"), "{error}");

        assert!(reject_authority_configured_environment(Some(serde_json::json!({}))).is_ok());
        assert!(reject_authority_configured_environment(None).is_ok());
    }

    #[test]
    fn compiler_execution_profile_identity_is_generation_semantic_input() {
        fn profile(
            seed: u8,
        ) -> fe2o3_compiler_execution_protocol::CompilerExecutionClientProfileV1 {
            use ed25519_dalek::SigningKey;
            use fe2o3_compiler_execution_protocol::{
                CompilerExecutionExternalAnchorServiceIdentityV1,
                CompilerExecutionIssuerMeasurementV1, CompilerExecutionIssuerPolicyV1,
            };

            let policy = CompilerExecutionIssuerPolicyV1::new(
                u64::from(seed),
                CompilerExecutionIssuerMeasurementV1::new([seed + 1; 32], 123).unwrap(),
                CompilerExecutionIssuerMeasurementV1::new([seed + 2; 32], 456).unwrap(),
                SigningKey::from_bytes(&[seed; 32])
                    .verifying_key()
                    .to_bytes(),
                SigningKey::from_bytes(&[seed.wrapping_add(1); 32])
                    .verifying_key()
                    .to_bytes(),
            )
            .unwrap();
            fe2o3_compiler_execution_protocol::CompilerExecutionClientProfileV1::new(
                1_234,
                5_678,
                CompilerExecutionExternalAnchorServiceIdentityV1::new(6_000, 7_000).unwrap(),
                policy,
            )
            .unwrap()
        }

        let first = profile(7);
        let second = profile(8);
        let mut first_configuration = b"existing-semantic-input".to_vec();
        let mut second_configuration = first_configuration.clone();
        append_compiler_execution_profile_semantic_configuration(&mut first_configuration, &first);
        append_compiler_execution_profile_semantic_configuration(
            &mut second_configuration,
            &second,
        );
        assert_ne!(first_configuration, second_configuration);
        assert!(first_configuration.ends_with(first.identity().as_bytes()));
        assert!(second_configuration.ends_with(second.identity().as_bytes()));
    }

    #[test]
    fn production_target_selection_is_owned_by_the_orchestrator() {
        for cpu in [
            fe2o3_amd_target::PRODUCTION_GFX942_DEVICE_CPU_V1,
            fe2o3_amd_target::PRODUCTION_GFX950_DEVICE_CPU_V1,
        ] {
            assert!(
                validate_production_cargo_inputs(&["--lib".into()], Some(OsStr::new(cpu))).is_ok()
            );
        }
        for args in [
            vec![
                "--target".into(),
                fe2o3_amd_target::PRODUCTION_GFX942_RUSTC_TARGET_V1.into(),
            ],
            vec!["--target=x86_64-unknown-linux-gnu".into()],
        ] {
            assert!(
                validate_production_cargo_inputs(
                    &args,
                    Some(OsStr::new(
                        fe2o3_amd_target::PRODUCTION_GFX942_DEVICE_CPU_V1,
                    )),
                )
                .is_err()
            );
        }
        assert!(validate_production_cargo_inputs(&[], Some(OsStr::new("gfx942:xnack-"))).is_err());
        assert!(validate_production_cargo_inputs(&[], Some(OsStr::new("gfx950:xnack-"))).is_err());
        assert!(validate_production_cargo_inputs(&[], Some(OsStr::new("GFX950"))).is_err());
    }

    #[test]
    fn run_target_routing_is_strict_and_stops_at_separator() {
        let args = ["--target=gfx942", "--", "--target", "application"].map(OsString::from);
        assert_eq!(
            selected_run_target(&args).expect("parse target").as_deref(),
            Some("gfx942")
        );
        let duplicate = ["--target", "gfx942", "--target=gfx1100"].map(OsString::from);
        assert!(selected_run_target(&duplicate).is_err());
    }

    #[test]
    fn post_spawn_failures_preserve_primary_and_append_checks_in_order() {
        let error = aggregate_post_spawn_results(
            Err("cargo failed".to_owned()),
            [
                ("boundary", Err("boundary failed".to_owned())),
                ("runtime", Err("runtime changed".to_owned())),
                ("closure", Err("source changed".to_owned())),
            ],
        )
        .expect_err("post-spawn failures must fail");
        assert_eq!(
            error,
            "cargo failed; boundary also failed: boundary failed; runtime also failed: runtime changed; closure also failed: source changed"
        );
    }

    #[test]
    fn first_post_spawn_check_becomes_primary_after_cargo_success() {
        let error = aggregate_post_spawn_results(
            Ok(()),
            [
                ("boundary", Err("boundary failed".to_owned())),
                ("runtime", Err("runtime changed".to_owned())),
            ],
        )
        .expect_err("post-spawn validation failure must fail");
        assert_eq!(
            error,
            "boundary failed; runtime also failed: runtime changed"
        );
    }

    #[test]
    fn source_isa_collection_hex_is_lowercase_bounded_and_fallible() {
        assert_eq!(
            source_isa_collection_hex(&[0x00, 0xab, 0xff]).unwrap(),
            "00abff"
        );
        assert_eq!(
            source_isa_collection_hex_length(
                fe2o3_source_isa_observation::wire_v1::MAX_SOURCE_ISA_OBSERVATION_COLLECTION_HEX_BYTES_V1 / 2
            ),
            Ok(fe2o3_source_isa_observation::wire_v1::MAX_SOURCE_ISA_OBSERVATION_COLLECTION_HEX_BYTES_V1)
        );
        assert!(
            source_isa_collection_hex_length(
                fe2o3_source_isa_observation::wire_v1::MAX_SOURCE_ISA_OBSERVATION_COLLECTION_HEX_BYTES_V1 / 2 + 1
            )
            .is_err()
        );
        assert!(source_isa_collection_hex_length(usize::MAX).is_err());
        assert_eq!(
            source_isa_characteristic_broker_hex(&[0x00, 0xab, 0xff]).unwrap(),
            "00abff"
        );
        let maximum = fe2o3_source_isa_observation::characteristic_v1::MAX_SOURCE_ISA_CHARACTERISTIC_COLLECTION_BYTES_V1
            + 256;
        assert_eq!(
            source_isa_characteristic_broker_hex_length(maximum),
            Ok(maximum * 2)
        );
        assert!(source_isa_characteristic_broker_hex_length(maximum + 1).is_err());
        assert!(source_isa_characteristic_broker_hex_length(usize::MAX).is_err());
    }

    #[test]
    fn max_observer_line_and_raii_finish_ignore_zero_and_partial_writer_failures() {
        let payload = "x".repeat(MAX_SOURCE_ISA_COLLECTION_STDERR_LINE_BYTES_V1 - 1);
        for bytes_before_failure in [0, 31] {
            let mut writer = FailingWriter {
                bytes_before_failure,
                bytes_written: 0,
            };
            assert!(
                observer_telemetry::write_line_to(&mut writer, format_args!("{payload}")).is_err()
            );
            assert_eq!(writer.bytes_written, bytes_before_failure);
        }

        OBSERVER_FINISH_CALLS.store(0, Ordering::SeqCst);
        let primary = std::panic::catch_unwind(|| {
            let _completion =
                ObserverFinishOnDropV1::new(1_u8, true, finish_with_failing_observer_writer);
            Err::<(), _>("authoritative primary failure")
        })
        .expect("observer Drop must not panic");
        assert_eq!(primary, Err("authoritative primary failure"));
        assert_eq!(OBSERVER_FINISH_CALLS.load(Ordering::SeqCst), 1);

        ObserverFinishOnDropV1::new(2_u8, true, finish_with_failing_observer_writer).finish();
        assert_eq!(OBSERVER_FINISH_CALLS.load(Ordering::SeqCst), 2);
    }
}

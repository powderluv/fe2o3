#!/usr/bin/env bash

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
DEFAULT_CARGO_TARGET_ROOT="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
if [[ "${DEFAULT_CARGO_TARGET_ROOT}" != /* ]]; then
  DEFAULT_CARGO_TARGET_ROOT="${REPO_ROOT}/${DEFAULT_CARGO_TARGET_ROOT}"
fi
DEFAULT_CARGO_TARGET_ROOT="$(realpath --canonicalize-missing -- "${DEFAULT_CARGO_TARGET_ROOT}")"
readonly DEFAULT_CARGO_TARGET_ROOT
LOG_DIR="${CI_LOG_DIR:-${DEFAULT_CARGO_TARGET_ROOT}/ci-logs}"
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${REPO_ROOT}/${LOG_DIR}"
fi
LOG_DIR="$(realpath --canonicalize-missing -- "${LOG_DIR}")"
readonly LOG_DIR
readonly RUSTC_CODEGEN_TEST_PACKAGE="rustc-codegen-fe2o3"
readonly CARGO_FE2O3_WORKER_V3_INTEGRATION_FEATURE="worker-v3-envelope-integration-test-only"
readonly RUSTC_CODEGEN_SHARD_POLICY="${REPO_ROOT}/scripts/rustc-codegen-shards.py"
readonly HYGIENE_DELTA_POLICY="${REPO_ROOT}/scripts/hygiene_delta_policy.py"
readonly HYGIENE_DELTA_POLICY_TESTS="${REPO_ROOT}/scripts/tests/hygiene_delta_policy.py"
readonly WORKSPACE_DEPENDENCY_POLICY_CHECKER="${REPO_ROOT}/scripts/workspace_dependency_policy.py"
readonly WORKSPACE_DEPENDENCY_POLICY="${REPO_ROOT}/scripts/workspace-dependency-policy.json"
readonly WORKSPACE_DEPENDENCY_POLICY_TESTS="${REPO_ROOT}/scripts/tests/workspace_dependency_policy.py"
readonly PLIRON_DEPENDENCY_POLICY_CHECKER="${REPO_ROOT}/scripts/pliron_dependency_policy.py"
readonly PLIRON_DEPENDENCY_POLICY_TESTS="${REPO_ROOT}/scripts/tests/pliron_dependency_policy.py"
readonly STANDALONE_LOCKFILE_CHECKER="${REPO_ROOT}/scripts/check-standalone-lockfiles.sh"
readonly RUNTIME_PURE_RUST_AUDITOR="${REPO_ROOT}/scripts/runtime_pure_rust_audit.py"
readonly RUNTIME_PURE_RUST_POLICY="${REPO_ROOT}/scripts/runtime-pure-rust-policy.json"
readonly SIM_RUNTIME_NO_GPU_POLICY="${REPO_ROOT}/scripts/sim-runtime-no-gpu-policy.json"
readonly VIRTUAL_RUNTIME_NO_GPU_POLICY="${REPO_ROOT}/scripts/virtual-runtime-no-gpu-policy.json"
readonly RUNTIME_PURE_RUST_AUDIT_TESTS="${REPO_ROOT}/scripts/tests/runtime_pure_rust_audit.py"
readonly RUNTIME_IDENTITY_ORACLE_TESTS="${REPO_ROOT}/scripts/tests/runtime_identity_oracle.py"
readonly RUNTIME_IDENTITY_ORACLE="${REPO_ROOT}/scripts/runtime-identity-oracle.sh"
readonly RUNTIME_PURE_RUST_TARGET_DIR="${DEFAULT_CARGO_TARGET_ROOT}/runtime-pure-rust-policy"
readonly CI_STEP_TIMEOUT_SECONDS="${FE2O3_CI_STEP_TIMEOUT_SECONDS:-3000}"
readonly CI_STEP_KILL_AFTER_SECONDS="${FE2O3_CI_STEP_KILL_AFTER_SECONDS:-15}"
readonly TEST_DRIVER_BINARY_ENV="FE2O3_TEST_CARGO_FE2O3_BIN"
readonly TEST_DRIVER_SHA256_ENV="FE2O3_TEST_CARGO_FE2O3_SHA256"
readonly -a SOURCE_ISA_PROTECTED_ENVIRONMENT=(
  FE2O3_TEST_CARGO_FE2O3_BIN
  FE2O3_TEST_CARGO_FE2O3_SHA256
  FE2O3_PRODUCTION_BUILD_CONFIG_V2
  FE2O3_AUTHORITY_BACKEND_SHA256_V1
  FE2O3_AUTHORITY_CARGO_SHA256_V1
  FE2O3_AUTHORITY_CARGO_BINDING_TRAMPOLINE_PATH_V1
  FE2O3_AUTHORITY_CARGO_BINDING_TRAMPOLINE_SHA256_V1
  FE2O3_AUTHORITY_RUSTC_PATH_V1
  FE2O3_AUTHORITY_RUSTC_RUNTIME_SHA256_V1
  FE2O3_AUTHORITY_RUSTC_SHA256_V1
  FE2O3_BACKEND
)
CARGO_FE2O3_BINARY=
CARGO_FE2O3_SHA256=
CARGO_FE2O3_DRIVER_ROOT=
CARGO_FE2O3_DRIVER_PROFILE=
CARGO_TARGET_DIRECTORY=
CI_PRIVATE_TMP_ROOT=
STANDALONE_LOCKFILES_CHECKED=0
readonly -a ROCM_TRUSTED_DEVICE_ITEM_PACKAGES=(
  fe2o3-vecadd
  fe2o3-trusted-item-renamed-genuine
  fe2o3-trusted-item-lookalike-type
  fe2o3-trusted-item-lookalike-helper
  fe2o3-trusted-item-lookalike-thread
  fe2o3-trusted-item-external-spoof
  fe2o3-trusted-item-local-marker
  fe2o3-typed-alias-spoof
)
# This proof fixture is also compiled outside the managed wrapper and carries a
# source-identity-pinned fallback namespace. Every other gate source must use
# the compiler-derived binding.
readonly -a ROCM_EXPLICIT_NAMESPACE_FALLBACK_PACKAGES=(
  fe2o3-typed-alias-spoof
)
readonly CPU_TEST_PACKAGES=(
  dialect-amdgcn
  dialect-autotune
  dialect-dispatch
  dialect-gpu
  dialect-kernel
  dialect-mir
  dialect-proof
  dialect-schedule
  dialect-tile
  fe2o3-amd-target
  fe2o3-amdgcn-model
  fe2o3-amdhsa-loader
  fe2o3-aql
  fe2o3-broker-authority-service
  fe2o3-completion
  fe2o3-compiler-api
  fe2o3-compiler-closure-capability
  fe2o3-compiler-execution-client
  fe2o3-compiler-execution-coordinator
  fe2o3-compiler-execution-protocol
  fe2o3-compiler-execution-supervisor
  fe2o3-artifacts
  fe2o3-contracts
  fe2o3-device
  fe2o3-debug-cli
  fe2o3-debug-protocol
  fe2o3-differential
  fe2o3-drm-uapi
  fe2o3-hsaco
  fe2o3-hsaco-finalize
  fe2o3-host
  fe2o3-host-api
  fe2o3-kfd
  fe2o3-kfd-uapi
  fe2o3-kernel-analysis
  fe2o3-kernel-descriptor
  fe2o3-kernel-ir
  fe2o3-kir-debugger
  fe2o3-kir-sim
  fe2o3-kir-sim-cli
  fe2o3-kir-sim-trace
  fe2o3-loop-device
  fe2o3-lower-mir-kernel
  fe2o3-macros
  fe2o3-mir-model
  fe2o3-pliron
  fe2o3-pliron-conformance
  fe2o3-process-identity
  fe2o3-profiler-protocol
  fe2o3-proof-contracts
  fe2o3-proof-custodian
  fe2o3-resource-accounting
  fe2o3-protected-service-profile
  fe2o3-protected-service-spawn
  fe2o3-protected-static-executable
  fe2o3-rustc-front
  fe2o3-rustc-invocation
  fe2o3-service-host
  fe2o3-service-model
  fe2o3-sim-differential
  fe2o3-sim-physical-differential
  fe2o3-sim-runtime
  fe2o3-source-isa-observation
  fe2o3-semantic-import
  fe2o3-semantic-query
  fe2o3-semantic-trace
  fe2o3-runtime
  fe2o3-runtime-machine-adapter
  fe2o3-runtime-model
  fe2o3-target-spec
  fe2o3-virtual-runtime
  fe2o3-virtual-runtime-cli
  fe2o3-verifier
  fe2o3-worker-v3-verification-protocol
  fe2o3-worker-v3-verification-client
  fe2o3-worker-v3-verification-service
  reserved-fe2o3-symbols
)
readonly HOST_REFERENCE_EXAMPLES=(
  rmsnorm_residual_v1
  qwen3_gqa_prefill_v1
  qwen3_paged_gqa_decode_v1
  qwen3_swiglu_v1
  qwen3_logits_compact_v1
  qwen3_rope_kv_v1
  qwen3_linear_reference_v1
)

usage() {
  cat <<'EOF'
Usage: scripts/ci-local.sh <command>

Commands:
  generic         Run all validation suitable for a machine without ROCm/GPU
  generic-core [group]  Run all core validation or policy, cpu, auxiliary, or cpu-<group>
                         CPU groups: foundation, analysis, lowering, pliron, finalize, finalize-0..3, integration
  workspace-policy  Validate workspace ownership and dependency directions
  hygiene-delta <base> <head>  Validate changed production source hygiene
  standalone-locks  Validate every tracked standalone Cargo lockfile
  runtime-policy  Validate the pure-Rust runtime dependency and ELF auditor
  runtime-identity-oracle  Measure MI300X identity against isolated rocminfo; explicit opt-in
  shard-policy    Validate the codegen integration shard assignment
  rustc-codegen-shard <id>  Run one codegen integration shard
  format          Check Rust formatting
  check           Check every workspace target, including example binaries
  test            Run unit tests that do not link or load the HIP runtime
  host-reference  Validate standalone host-only numerical reference models
  workspace-test  Run every workspace test target; may require ROCm libraries
  rustc-codegen-test  Run backend library and integration tests without dylib replacement
  backend         Build the rustc codegen backend dylib
  authority-launcher  Run bounded protected build-authority launcher tests
  source-isa-unit-matrix  Run the opt-in protected source/ISA ordinary-unit matrix
  source-isa-characteristic-contract-v2  Validate the opt-in characteristic matrix contract
  source-isa-characteristic-matrix-v2  Run the opt-in protected ordinary-source 3x2 characteristic matrix
  rustc-trampoline    Run non-integrated static rustc trampoline tests
  parity-evidence Run parity, signed-attestation, and queue shell tests
  parity-production-immutable  Run opt-in root ext4/XFS ingestion test
  verus           Run positive and negative Verus proof fixtures; requires Verus
  rocm-compile    Run bounded production ROCm compiler checks; requires ROCm
  hardware-smoke  Run guarded KFD hardware checks; requires MI300X and opt-in
EOF
}

run_step_with_timeout() {
  local timeout_seconds="$1"
  local name="$2"
  shift 2
  local log_file="${LOG_DIR}/${name}.log"

  if [[ ! "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
    ((timeout_seconds >= 7200)); then
    printf 'step timeout must be an integer from 1 through 7199: %s\n' \
      "${timeout_seconds}" >&2
    return 2
  fi
  if [[ ! "${CI_STEP_KILL_AFTER_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
    ((CI_STEP_KILL_AFTER_SECONDS > 300)); then
    printf '%s\n' \
      'FE2O3_CI_STEP_KILL_AFTER_SECONDS must be an integer from 1 through 300' >&2
    return 2
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    printf '%s\n' 'ci-local requires GNU timeout to supervise each step' >&2
    return 2
  fi

  printf '\n==> %s\n' "${name}"
  printf '   command:'
  printf ' %q' "$@"
  printf '\n   timeout: %ss' "${timeout_seconds}"
  printf '\n   log: %s\n' "${log_file}"

  set +e
  timeout --signal=TERM --kill-after="${CI_STEP_KILL_AFTER_SECONDS}s" \
    "${timeout_seconds}s" "$@" 2>&1 | tee "${log_file}"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  local command_status="${pipeline_status[0]}"
  local tee_status="${pipeline_status[1]}"
  local status
  set -e

  if ((command_status != 0)); then
    status="${command_status}"
  else
    status="${tee_status}"
  fi
  if ((tee_status != 0)); then
    printf 'step %s log write failed with status %d\n' \
      "${name}" "${tee_status}" >&2
  fi
  if ((status != 0)); then
    if ((command_status == 124)); then
      printf 'step %s timed out after %s seconds\n' \
        "${name}" "${timeout_seconds}" >&2
    fi
    printf 'step %s failed with status %d\n' "${name}" "${status}" >&2
    return "${status}"
  fi
}

run_step() {
  if [[ ! "${CI_STEP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
    ((CI_STEP_TIMEOUT_SECONDS >= 3600)); then
    printf '%s\n' \
      'FE2O3_CI_STEP_TIMEOUT_SECONDS must be an integer from 1 through 3599' >&2
    return 2
  fi
  run_step_with_timeout "${CI_STEP_TIMEOUT_SECONDS}" "$@"
}

validate_private_directory() {
  local label="$1"
  local directory="$2"
  local canonical mode owner current_owner

  [[ "${directory}" == /* && -d "${directory}" && ! -L "${directory}" ]] || {
    printf '%s must be an absolute non-symlink directory: %s\n' \
      "${label}" "${directory}" >&2
    return 2
  }
  canonical="$(realpath --canonicalize-existing -- "${directory}")" || return $?
  mode="$(stat -c '%a' -- "${directory}")" || return $?
  owner="$(stat -c '%u' -- "${directory}")" || return $?
  current_owner="$(id -u)" || return $?
  if [[ "${canonical}" != "${directory}" ]] ||
    ((8#${mode} & 8#077)) || [[ "${owner}" != "${current_owner}" ]]; then
    printf '%s must be canonical, owner-held, and private: %s\n' \
      "${label}" "${directory}" >&2
    return 2
  fi
}

resolve_cargo_target_directory() {
  local target_directory canonical mode="${1-existing}"
  case "${mode}" in
    existing | create-private) ;;
    *) printf 'unknown Cargo target preparation mode: %s\n' "${mode}" >&2; return 2 ;;
  esac
  target_directory="$(
    cargo metadata --locked --no-deps --format-version 1 |
      python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
  )" || return $?
  if [[ "${target_directory}" != /* ]] ||
    [[ "${target_directory}" == *$'\n'* ]]; then
    printf 'Cargo reported an invalid target directory: %q\n' \
      "${target_directory}" >&2
    return 2
  fi
  # Cold package lanes have no preceding build to create this metadata-bound root.
  if [[ "${mode}" == create-private && ! -e "${target_directory}" &&
    ! -L "${target_directory}" ]]; then
    canonical="$(realpath --canonicalize-missing -- "${target_directory}")" || return $?
    [[ "${canonical}" == "${target_directory}" ]] || {
      printf 'new Cargo target directory must be canonical: %s\n' \
        "${target_directory}" >&2
      return 2
    }
    mkdir -p -m 700 -- "${target_directory}" || return $?
  fi
  [[ -d "${target_directory}" && ! -L "${target_directory}" ]] || {
    printf 'Cargo target directory is not a real directory: %s\n' \
      "${target_directory}" >&2
    return 2
  }
  canonical="$(realpath --canonicalize-existing -- "${target_directory}")" || return $?
  if [[ "${canonical}" != "${target_directory}" ]]; then
    printf 'Cargo target directory must already be canonical: %s\n' \
      "${target_directory}" >&2
    return 2
  fi
  validate_private_directory 'Cargo target directory' "${canonical}" || return $?
  printf '%s\n' "${canonical}"
}

validate_cargo_fe2o3_driver() {
  local canonical mode owner digest parent parent_mode parent_owner

  [[ -n "${CARGO_FE2O3_BINARY}" && -n "${CARGO_FE2O3_SHA256}" ]] || {
    printf '%s\n' 'cargo-fe2o3 sealed driver is not prepared' >&2
    return 2
  }
  [[ "${CARGO_FE2O3_BINARY}" == /* && -f "${CARGO_FE2O3_BINARY}" &&
    ! -L "${CARGO_FE2O3_BINARY}" && -x "${CARGO_FE2O3_BINARY}" ]] || {
    printf 'cargo-fe2o3 sealed driver is not an absolute executable file: %s\n' \
      "${CARGO_FE2O3_BINARY}" >&2
    return 2
  }
  canonical="$(realpath --canonicalize-existing -- "${CARGO_FE2O3_BINARY}")"
  mode="$(stat -c '%a' -- "${CARGO_FE2O3_BINARY}")"
  owner="$(stat -c '%u' -- "${CARGO_FE2O3_BINARY}")"
  parent="$(dirname -- "${CARGO_FE2O3_BINARY}")"
  parent_mode="$(stat -c '%a' -- "${parent}")"
  parent_owner="$(stat -c '%u' -- "${parent}")"
  digest="$(sha256sum -- "${CARGO_FE2O3_BINARY}")"
  digest="${digest%% *}"
  if [[ "${canonical}" != "${CARGO_FE2O3_BINARY}" ]] ||
    [[ "${mode}" != 500 ]] || [[ "${owner}" != "$(id -u)" ]] ||
    [[ "${parent_mode}" != 500 ]] || [[ "${parent_owner}" != "$(id -u)" ]] ||
    [[ "${digest}" != "${CARGO_FE2O3_SHA256}" ]]; then
    printf '%s\n' \
      'cargo-fe2o3 sealed driver identity or private custody changed' >&2
    return 2
  fi
}

resolve_cargo_fe2o3_artifact() {
  python3 - "$@" <<'PY'
import json
import os
import sys

receipt, expected_package, expected_source, expected_target = sys.argv[1:]
paths = []
with open(receipt, "rb") as records:
    for line in records:
        record = json.loads(line)
        if record.get("reason") != "compiler-artifact" or record.get("target", {}).get("name") != "cargo-fe2o3":
            continue
        target = record.get("target", {})
        profile = record.get("profile", {})
        if (
            record.get("package_id") != expected_package
            or target.get("kind") != ["bin"]
            or target.get("crate_types") != ["bin"]
            or target.get("src_path") != expected_source
            or profile.get("test") is not False
            or profile.get("opt_level") != "0"
        ):
            raise SystemExit("Cargo reported a cargo-fe2o3 artifact with mismatched package, source, target, or profile identity")
        executable = record.get("executable")
        if not executable or not os.path.isabs(executable):
            raise SystemExit("Cargo reported cargo-fe2o3 without an absolute executable")
        canonical = os.path.realpath(executable)
        if canonical != executable or os.path.commonpath([expected_target, canonical]) != expected_target:
            raise SystemExit("Cargo reported cargo-fe2o3 outside the admitted target")
        paths.append(canonical)
if len(paths) != 1:
    raise SystemExit(
        f"expected exactly one cargo-fe2o3 executable artifact; found {len(paths)}"
    )
print(paths[0])
PY
}

prepare_private_tmp_root() {
  if [[ -n "${TMPDIR:-}" ]]; then
    validate_private_directory TMPDIR "${TMPDIR}"
    return
  fi
  validate_private_directory 'Cargo target directory' "${CARGO_TARGET_DIRECTORY}"
  CI_PRIVATE_TMP_ROOT="${CARGO_TARGET_DIRECTORY}/fe2o3-ci-tmp-${BASHPID}"
  [[ ! -e "${CI_PRIVATE_TMP_ROOT}" && ! -L "${CI_PRIVATE_TMP_ROOT}" ]] || {
    printf 'private CI temporary root already exists: %s\n' \
      "${CI_PRIVATE_TMP_ROOT}" >&2
    return 2
  }
  mkdir -m 700 -- "${CI_PRIVATE_TMP_ROOT}"
  [[ "$(realpath --canonicalize-existing -- "${CI_PRIVATE_TMP_ROOT}")" == \
    "${CI_PRIVATE_TMP_ROOT}" ]] || {
    printf 'private CI temporary root is not canonical: %s\n' \
      "${CI_PRIVATE_TMP_ROOT}" >&2
    return 2
  }
  TMPDIR="${CI_PRIVATE_TMP_ROOT}"
  export TMPDIR
  validate_private_directory TMPDIR "${TMPDIR}"
}

prepare_cargo_fe2o3_driver() {
  local step_prefix="$1"
  local driver_profile="$2"
  local target_mode="${3-existing}"
  local metadata_receipt receipt built_binary built_sha256
  local -a driver_identity feature_args=()

  case "${driver_profile}" in
    production) ;;
    *)
      printf 'unknown cargo-fe2o3 driver profile: %s\n' \
        "${driver_profile}" >&2
      return 2
      ;;
  esac
  CARGO_FE2O3_DRIVER_PROFILE=

  CARGO_TARGET_DIRECTORY="$(resolve_cargo_target_directory "${target_mode}")"
  prepare_private_tmp_root
  metadata_receipt="$(mktemp -- "${TMPDIR}/cargo-fe2o3-metadata.XXXXXX.json")"
  cargo metadata --locked --no-deps --format-version 1 >"${metadata_receipt}"
  mapfile -t driver_identity < <(
    python3 - "${metadata_receipt}" "${REPO_ROOT}" "${CARGO_TARGET_DIRECTORY}" <<'PY'
import json
import os
import sys

workspace = os.path.realpath(sys.argv[2])
expected_target = sys.argv[3]
with open(sys.argv[1], "rb") as source:
    metadata = json.load(source)
if metadata.get("target_directory") != expected_target:
    raise SystemExit("Cargo metadata target_directory changed during driver bootstrap")
packages = [package for package in metadata.get("packages", []) if package.get("name") == "cargo-fe2o3"]
if len(packages) != 1:
    raise SystemExit(f"expected exactly one cargo-fe2o3 package; found {len(packages)}")
package = packages[0]
expected_manifest = os.path.join(workspace, "crates/cargo-fe2o3/Cargo.toml")
if os.path.realpath(package.get("manifest_path", "")) != expected_manifest:
    raise SystemExit("cargo-fe2o3 package resolved outside the admitted workspace")
targets = [
    target
    for target in package.get("targets", [])
    if target.get("name") == "cargo-fe2o3" and target.get("kind") == ["bin"]
]
if len(targets) != 1:
    raise SystemExit(f"expected exactly one cargo-fe2o3 binary target; found {len(targets)}")
source = os.path.realpath(targets[0].get("src_path", ""))
if source != os.path.join(workspace, "crates/cargo-fe2o3/src/main.rs"):
    raise SystemExit("cargo-fe2o3 binary resolved to an unexpected source")
print(package["id"])
print(source)
PY
  )
  rm -f -- "${metadata_receipt}"
  ((${#driver_identity[@]} == 2)) || {
    printf '%s\n' 'failed to bind cargo-fe2o3 package identity' >&2
    return 2
  }
  receipt="$(mktemp -- "${TMPDIR}/cargo-fe2o3-artifacts.XXXXXX.json")"
  run_step "${step_prefix}-cargo-fe2o3-bootstrap" \
    bash -c 'set -Eeuo pipefail
      receipt="$1"
      shift
      exec env CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=1 \
        cargo build --locked -p cargo-fe2o3 --bin cargo-fe2o3 \
        "$@" --message-format=json-render-diagnostics >"${receipt}"' \
    cargo-fe2o3-bootstrap "${receipt}" "${feature_args[@]}"
  built_binary="$(resolve_cargo_fe2o3_artifact "${receipt}" \
    "${driver_identity[0]}" "${driver_identity[1]}" \
    "${CARGO_TARGET_DIRECTORY}")"
  rm -f -- "${receipt}"
  [[ "${built_binary}" == /* && -f "${built_binary}" &&
    ! -L "${built_binary}" && -x "${built_binary}" ]] || {
    printf 'Cargo reported an invalid cargo-fe2o3 executable: %s\n' \
      "${built_binary}" >&2
    return 2
  }
  [[ "$(realpath --canonicalize-existing -- "${built_binary}")" == "${built_binary}" ]] || {
    printf 'Cargo reported a noncanonical cargo-fe2o3 executable: %s\n' \
      "${built_binary}" >&2
    return 2
  }
  built_sha256="$(sha256sum -- "${built_binary}")"
  built_sha256="${built_sha256%% *}"
  CARGO_FE2O3_DRIVER_ROOT="${TMPDIR}/fe2o3-ci-driver-${built_sha256}"
  [[ ! -e "${CARGO_FE2O3_DRIVER_ROOT}" && ! -L "${CARGO_FE2O3_DRIVER_ROOT}" ]] || {
    printf 'private sealed driver root already exists: %s\n' \
      "${CARGO_FE2O3_DRIVER_ROOT}" >&2
    return 2
  }
  mkdir -m 700 -- "${CARGO_FE2O3_DRIVER_ROOT}"
  install -m 0500 -- "${built_binary}" \
    "${CARGO_FE2O3_DRIVER_ROOT}/cargo-fe2o3"
  CARGO_FE2O3_BINARY="${CARGO_FE2O3_DRIVER_ROOT}/cargo-fe2o3"
  CARGO_FE2O3_SHA256="${built_sha256}"
  chmod 500 -- "${CARGO_FE2O3_DRIVER_ROOT}"
  validate_cargo_fe2o3_driver
  CARGO_FE2O3_DRIVER_PROFILE="${driver_profile}"
}

ensure_production_cargo_fe2o3_driver() {
  local step_prefix="$1"
  case "${CARGO_FE2O3_DRIVER_PROFILE}" in
    production)
      validate_cargo_fe2o3_driver
      ;;
    "")
      prepare_cargo_fe2o3_driver "${step_prefix}" production "${2-existing}"
      ;;
    *)
      printf 'CPU tests cannot reuse %s cargo-fe2o3 driver as production\n' \
        "${CARGO_FE2O3_DRIVER_PROFILE}" >&2
      return 2
      ;;
  esac
}

retire_cargo_fe2o3_driver() {
  if [[ -n "${CARGO_FE2O3_DRIVER_ROOT}" && -d "${CARGO_FE2O3_DRIVER_ROOT}" &&
    ! -L "${CARGO_FE2O3_DRIVER_ROOT}" ]]; then
    chmod 700 -- "${CARGO_FE2O3_DRIVER_ROOT}" || true
    rm -rf -- "${CARGO_FE2O3_DRIVER_ROOT}"
  fi
  CARGO_FE2O3_BINARY=
  CARGO_FE2O3_SHA256=
  CARGO_FE2O3_DRIVER_ROOT=
  CARGO_FE2O3_DRIVER_PROFILE=
}

cleanup_cargo_fe2o3_driver() {
  retire_cargo_fe2o3_driver
  if [[ -n "${CI_PRIVATE_TMP_ROOT}" && -n "${CARGO_TARGET_DIRECTORY}" &&
    -d "${CI_PRIVATE_TMP_ROOT}" && ! -L "${CI_PRIVATE_TMP_ROOT}" &&
    "$(dirname -- "${CI_PRIVATE_TMP_ROOT}")" == "${CARGO_TARGET_DIRECTORY}" &&
    "$(basename -- "${CI_PRIVATE_TMP_ROOT}")" == "fe2o3-ci-tmp-${BASHPID}" &&
    "$(stat -c '%u' -- "${CI_PRIVATE_TMP_ROOT}")" == "$(id -u)" ]]; then
    rm -rf -- "${CI_PRIVATE_TMP_ROOT}"
  fi
}

trap cleanup_cargo_fe2o3_driver EXIT

load_dynamic_loader_environment_removals() {
  local destination_name="$1"
  local name
  local -n destination="${destination_name}"

  destination=()
  while IFS= read -r name; do
    case "${name}" in
      LD_* | DYLD_* | GLIBC_TUNABLES)
        destination+=(-u "${name}")
        ;;
    esac
  done < <(compgen -e)
}

validate_managed_wrapper_source_namespaces() {
  local package fallback
  local -a managed_packages=() loader_environment_removals
  local -A seen=() fallback_packages=()

  for fallback in "${ROCM_EXPLICIT_NAMESPACE_FALLBACK_PACKAGES[@]}"; do
    fallback_packages["${fallback}"]=1
  done
  for package in "$@"; do
    if [[ -n "${seen[${package}]:-}" ]]; then
      continue
    fi
    seen["${package}"]=1
    if [[ -n "${fallback_packages[${package}]:-}" ]]; then
      continue
    fi
    managed_packages+=("${package}")
  done
  ((${#managed_packages[@]} > 0)) || return 0
  validate_cargo_fe2o3_driver
  load_dynamic_loader_environment_removals loader_environment_removals
  env "${loader_environment_removals[@]}" \
    "${CARGO_FE2O3_BINARY}" examples check-wrapper-namespaces \
    "${managed_packages[@]}"
}

load_example_packages() {
  local lane="$1"
  local destination_name="$2"
  local cargo_fe2o3_binary="${3:-}"
  local output
  local -n destination="${destination_name}"

  if [[ -n "${cargo_fe2o3_binary}" ]]; then
    local -a loader_environment_removals
    load_dynamic_loader_environment_removals loader_environment_removals
    output="$(
      env "${loader_environment_removals[@]}" \
        "${cargo_fe2o3_binary}" examples list "${lane}"
    )"
  else
    output="$(
      cargo run --quiet --locked -p cargo-fe2o3 -- examples list "${lane}"
    )"
  fi
  destination=()
  if [[ -n "${output}" ]]; then
    # shellcheck disable=SC2034  # The destination is written through a nameref.
    mapfile -t destination <<<"${output}"
  fi
}

run_format() {
  run_step format cargo fmt --all -- --check
}

run_check() {
  local -a all_examples rustc_examples wrapper_managed_packages cargo_args
  local -a loader_environment_removals
  local -A rustc_example_set=()
  local package
  run_standalone_lockfiles
  prepare_cargo_fe2o3_driver generic-check production
  validate_cargo_fe2o3_driver
  load_example_packages all all_examples "${CARGO_FE2O3_BINARY}"
  load_example_packages rustc-check rustc_examples "${CARGO_FE2O3_BINARY}"
  load_example_packages wrapper-managed wrapper_managed_packages \
    "${CARGO_FE2O3_BINARY}"
  for package in "${rustc_examples[@]}"; do
    rustc_example_set["${package}"]=1
  done

  cargo_args=(check --workspace --all-targets --locked)
  # These fixtures intentionally require the authenticated cargo-fe2o3 wrapper
  # to supply their compiler-owned crate binding.
  cargo_args+=(
    --exclude fe2o3-production-extraction-fixture
    --exclude fe2o3-production-ranked-bounds-fixture
  )
  for package in "${all_examples[@]}"; do
    if [[ -z "${rustc_example_set[${package}]+selected}" ]]; then
      cargo_args+=(--exclude "${package}")
    fi
  done

  ((${#wrapper_managed_packages[@]} > 0)) || {
    printf '%s\n' 'workspace contains no structurally detected wrapper-managed package' >&2
    return 2
  }
  # The package-aware wrapper sees the complete supported workspace graph. Its sealed
  # target-source projection injects bindings only for directly managed package targets,
  # including transitive managed libraries, and leaves ordinary dependencies unbound.
  load_dynamic_loader_environment_removals loader_environment_removals
  run_step workspace-binding-check \
    env "${loader_environment_removals[@]}" \
    "${CARGO_FE2O3_BINARY}" "${cargo_args[@]}"
  run_step workspace-binding-check-boundary \
    bash scripts/tests/binding-check-boundary.sh \
    "${CARGO_FE2O3_BINARY}" "${wrapper_managed_packages[0]}"
  # This is an authority-free policy rescan, not a retained source snapshot.
  # Recompute the exact set after the nested Cargo checks so projection drift
  # fails this lane before it reports success.
  run_step workspace-binding-projection-revalidation \
    env "${loader_environment_removals[@]}" \
    "${CARGO_FE2O3_BINARY}" examples check-wrapper-managed \
    "${wrapper_managed_packages[@]}"
  run_step standalone-tiled-gemm-general-host-check \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    "${CARGO_FE2O3_BINARY}" check --locked --all-targets \
      --manifest-path examples/tiled_gemm_general_v1/Cargo.toml
  run_step standalone-flash-attention-general-host-check \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    "${CARGO_FE2O3_BINARY}" check --locked --all-targets \
      --manifest-path examples/flash_attention_general_v1/Cargo.toml
  run_step native-conditional-application-binding-check \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    "${CARGO_FE2O3_BINARY}" check --locked --bins \
      --manifest-path crates/cargo-fe2o3/tests/fixtures/conditional-custodian-application/Cargo.toml
}

run_artifact_transaction_tests() {
  # Artifact publication tests intentionally retain descriptor custody. Bound
  # their libtest fanout below the common 1024-descriptor soft limit.
  run_step fe2o3-artifact-transaction-tests \
    env FE2O3_HIP_SYS_DISABLE=1 RUST_TEST_THREADS=8 \
    cargo test --locked -p fe2o3-artifact-transaction
}


run_native_timestamp_cpu_tests() {
  # These three cases use aligned host arrays, never a native device or queue.
  local -a command=(cargo test --locked -p fe2o3-kfd
    --features engineering-native-packet-diagnostics --lib native_timestamp_)
  run_step native-timestamp-cpu-discovery "${command[@]}" -- --list
  run_step native-timestamp-cpu-discovery-check \
    python3 -I -B scripts/check-native-timestamp-cpu.py list \
      "${LOG_DIR}/native-timestamp-cpu-discovery.log"
  run_step native-timestamp-cpu-tests "${command[@]}" -- --test-threads=1
  run_step native-timestamp-cpu-result-check \
    python3 -I -B scripts/check-native-timestamp-cpu.py run \
      "${LOG_DIR}/native-timestamp-cpu-tests.log"
}


run_runtime_release_tests() {
  # These host-only tests intentionally abort child processes and must also
  # exercise bookkeeping with debug assertions disabled.
  ulimit -c 0
  run_step fe2o3-runtime-release-tests \
    env -u FE2O3_TEST_SCRIPTED_SDMA_ABORT_CHILD \
      -u FE2O3_RUNTIME_TELEMETRY_ABORT_CASE \
      CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=false \
    cargo test --locked --release -p fe2o3-runtime \
      --features hardware-qualification --lib -- --test-threads=1
  run_step fe2o3-resource-accounting-release-tests \
    env CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=false \
      cargo test --locked --release -p fe2o3-resource-accounting --lib
  run_step fe2o3-runtime-scale-cpu-tests \
    cargo test --locked -p fe2o3-runtime \
      --features scale-qualification,cpu-runtime-fixtures --lib -- --test-threads=1
  run_step fe2o3-runtime-scoped-ownership-doc-tests \
    cargo test --locked -p fe2o3-runtime \
      --features scale-qualification,cpu-runtime-fixtures --doc generated_scope
}

run_host_reference_tests() {
  local example manifest
  local target="${DEFAULT_CARGO_TARGET_ROOT}/host-reference"
  for example in "${HOST_REFERENCE_EXAMPLES[@]}"; do
    manifest="examples/${example}/Cargo.toml"
    run_step "host-reference-${example}-format" \
      cargo fmt --manifest-path "${manifest}" -- --check
    run_step "host-reference-${example}-clippy" \
      cargo clippy --locked --manifest-path "${manifest}" --target-dir "${target}" \
        --all-targets --all-features -- -D warnings
    run_step "host-reference-${example}-test" \
      cargo test --locked --manifest-path "${manifest}" --target-dir "${target}" \
        --all-targets --all-features -- --test-threads=1
    run_step "host-reference-${example}-release" \
      cargo test --locked --release --manifest-path "${manifest}" --target-dir "${target}" \
        --all-targets --all-features -- --test-threads=1
    run_step "host-reference-${example}-doc" \
      env RUSTDOCFLAGS="-D warnings" \
        cargo doc --locked --manifest-path "${manifest}" --target-dir "${target}" --no-deps
  done
}

run_workspace_dependency_bootstrap() {
  # Offline closure checks include optional workspace packages that the selected
  # test binary need not build. Each isolated lane must fetch their locked input.
  run_step "$1-workspace-dependencies" \
    cargo fetch --locked --manifest-path "${REPO_ROOT}/Cargo.toml"
}

# Keep one package roster. Unlisted ordinary CPU packages enter foundation;
# isolate the existing expensive suites without selecting individual tests.
load_cpu_package_group() {
  if (($# != 2)); then
    printf 'CPU package group requires a group and destination\n' >&2
    return 2
  fi
  local requested="$1" package group
  # shellcheck disable=SC2178  # The destination is a caller-owned array nameref.
  local -n selected="$2"
  local -A seen=()
  case "${requested}" in
    foundation | analysis | lowering | pliron | finalize) ;;
    *) printf 'unknown CPU package group: %s\n' "${requested}" >&2; return 2 ;;
  esac
  selected=()
  for package in "${CPU_TEST_PACKAGES[@]}"; do
    if [[ -z "${package}" ]]; then
      printf 'empty CPU package in the required roster\n' >&2
      return 2
    fi
    if [[ -n "${seen[${package}]+selected}" ]]; then
      printf 'duplicate CPU package: %s\n' "${package}" >&2
      return 2
    fi
    seen["${package}"]=1
    case "${package}" in
      fe2o3-kernel-analysis | fe2o3-kernel-ir | fe2o3-mir-model | fe2o3-verifier)
        group=analysis ;;
      fe2o3-lower-mir-kernel) group=lowering ;;
      fe2o3-pliron | fe2o3-pliron-conformance) group=pliron ;;
      fe2o3-hsaco-finalize) group=finalize ;;
      *) group=foundation ;;
    esac
    [[ "${group}" != "${requested}" ]] || selected+=("${package}")
  done
  if ((${#selected[@]} == 0)); then
    printf 'empty CPU package group: %s\n' "${requested}" >&2
    return 2
  fi
}

run_pliron_default_api_tests() {
  run_step fe2o3-pliron-default-api-ui \
    cargo test --locked -p fe2o3-pliron --no-default-features \
      --test middle_end_evidence_ui default_api_cannot_self_authorize -- --exact
}

run_cpu_package_group() {
  if (($# != 1)); then
    printf 'CPU package tests require exactly one group\n' >&2
    return 2
  fi
  local group="$1" package
  local -a packages cargo_args=(test --locked)
  load_cpu_package_group "${group}" packages
  run_workspace_dependency_bootstrap "cpu-${group}"
  # Each isolated lane must resolve tracked standalone locks before offline fixtures.
  run_standalone_lockfiles
  ensure_production_cargo_fe2o3_driver "cpu-${group}" create-private
  for package in "${packages[@]}"; do
    cargo_args+=(-p "${package}")
  done
  if [[ "${group}" == pliron ]]; then
    run_pliron_default_api_tests
  fi
  if [[ "${group}" == foundation ]]; then
    # Runtime fail-stop can abort during unwinding, before libtest prints its buffer.
    cargo_args+=(-- --nocapture)
  fi
  run_step "cpu-${group}-tests" \
    env FE2O3_HIP_SYS_DISABLE=1 cargo "${cargo_args[@]}"
}

run_cpu_finalizer_shard() {
  if (($# != 1)) || [[ ! "$1" =~ ^[0-3]$ ]]; then
    printf 'finalizer shard requires exactly one id from 0 through 3\n' >&2
    return 2
  fi
  local shard="$1"
  local -a packages
  load_cpu_package_group finalize packages
  if [[ "${packages[*]}" != fe2o3-hsaco-finalize ]]; then
    printf 'finalizer shards require the exact original package group\n' >&2
    return 2
  fi
  run_workspace_dependency_bootstrap "cpu-finalize-${shard}"
  run_standalone_lockfiles
  ensure_production_cargo_fe2o3_driver "cpu-finalize-${shard}" create-private
  run_step "cpu-finalize-${shard}-tests" \
    env FE2O3_HIP_SYS_DISABLE=1 python3 -I -B "${REPO_ROOT}/scripts/finalizer-test-shards.py" \
      --shard "${shard}" --output "${LOG_DIR}/finalizer-shard-${shard}"
}

run_cpu_tests() {
  if (($# > 1)); then
    printf 'CPU tests accept at most one mode\n' >&2
    return 2
  fi
  local mode="${1-all}" package_step=cpu-tests
  case "${mode}" in
    all) ;;
    integration) package_step=cpu-integration-tests ;;
    *) printf 'unknown CPU test mode: %s\n' "${mode}" >&2; return 2 ;;
  esac
  run_workspace_dependency_bootstrap cpu
  run_standalone_lockfiles
  local cargo_args=(test --locked)
  local wrapper_cargo_args=(test --locked --all-targets)
  local -a raw_cpu_examples wrapper_cpu_examples wrapper_managed_packages
  local -a loader_environment_removals
  local -A selected_cpu_examples=()
  local package
  run_host_reference_tests
  ensure_production_cargo_fe2o3_driver cpu-tests
  if [[ "${mode}" == all ]]; then
    for package in "${CPU_TEST_PACKAGES[@]}"; do
      cargo_args+=(-p "${package}")
    done
  fi
  load_example_packages cpu-test-raw raw_cpu_examples "${CARGO_FE2O3_BINARY}"
  load_example_packages cpu-test-wrapper-managed wrapper_cpu_examples \
    "${CARGO_FE2O3_BINARY}"
  load_example_packages wrapper-managed wrapper_managed_packages \
    "${CARGO_FE2O3_BINARY}"
  for package in "${raw_cpu_examples[@]}"; do
    [[ -z "${selected_cpu_examples[${package}]+selected}" ]] || {
      printf 'duplicate raw CPU example package: %s\n' "${package}" >&2
      return 2
    }
    selected_cpu_examples["${package}"]=raw
    cargo_args+=(-p "${package}")
  done
  for package in "${wrapper_cpu_examples[@]}"; do
    [[ -z "${selected_cpu_examples[${package}]+selected}" ]] || {
      printf 'CPU example partition selected package twice: %s\n' "${package}" >&2
      return 2
    }
    selected_cpu_examples["${package}"]=wrapper-managed
    wrapper_cargo_args+=(-p "${package}")
  done
  # Keep the generic test lane independent of whether the host happens to have
  # ROCm installed. The raw HIP crate supplies a fail-closed no-runtime ABI.
  # Unit-test builds expose dev-dependencies to production modules.
  run_step cargo-fe2o3-production-bins env FE2O3_HIP_SYS_DISABLE=1 \
    cargo build --locked --no-default-features -p cargo-fe2o3 \
      --bin cargo-fe2o3 --bin fe2o3-rustc-wrapper
  run_step cargo-fe2o3-tests env FE2O3_HIP_SYS_DISABLE=1 \
    cargo test --locked -p cargo-fe2o3
  run_step cargo-fe2o3-worker-v3-envelope-tests env FE2O3_HIP_SYS_DISABLE=1 \
    cargo test --locked -p cargo-fe2o3 \
      --features "${CARGO_FE2O3_WORKER_V3_INTEGRATION_FEATURE}" \
      --test worker_v3_load_envelope_vertical --test worker_v3_load_envelope_v2 -- --test-threads=1
  if [[ "${mode}" == all ]]; then
    run_pliron_default_api_tests
  fi
  run_artifact_transaction_tests
  # Empty dynamic raw examples must never become workspace-wide Cargo.
  if [[ "${mode}" == all ]] || ((${#raw_cpu_examples[@]} > 0)); then
    run_step "${package_step}" env FE2O3_HIP_SYS_DISABLE=1 cargo "${cargo_args[@]}"
  fi
  run_runtime_release_tests
  run_native_timestamp_cpu_tests
  run_step fe2o3-device-release-tests \
    env CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=false \
      cargo test --locked --release -p fe2o3-device --lib
  load_dynamic_loader_environment_removals loader_environment_removals
  if ((${#wrapper_cpu_examples[@]} > 0)); then
    validate_cargo_fe2o3_driver
    run_step wrapper-managed-cpu-tests \
      env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
      "${CARGO_FE2O3_BINARY}" "${wrapper_cargo_args[@]}"
  fi
  validate_cargo_fe2o3_driver
  run_step native-data-copy-application-binding-check \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    "${CARGO_FE2O3_BINARY}" check --locked --bins --features data-copy-observation \
      --manifest-path crates/cargo-fe2o3/tests/fixtures/conditional-custodian-application/Cargo.toml
  run_step tiled-gemm-capability-ui \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    cargo test --locked --offline \
      --manifest-path examples/tiled_gemm_general_v1/device-api/Cargo.toml --test device_api_ui
  # Keep host test images within the sealed runner's unchanged 512 MiB bound.
  run_step cpu-reference-tiled-gemm-paired-default \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
    "${CARGO_FE2O3_BINARY}" test --locked --offline \
      --manifest-path examples/tiled_gemm_general_v1/Cargo.toml --test paired_contract
  run_step cpu-reference-tiled-gemm-paired-simt \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
    "${CARGO_FE2O3_BINARY}" test --locked --offline \
      --manifest-path examples/tiled_gemm_general_v1/Cargo.toml \
      --no-default-features --features kernel-simt-gemm-general --test paired_contract
  # The queries above and the test command are authority-free policy scans. Recheck
  # both selected lists and the complete structural set so source drift cannot
  # silently reroute or omit a package.
  run_step cpu-test-partition-revalidation \
    env "${loader_environment_removals[@]}" \
    "${CARGO_FE2O3_BINARY}" examples check-cpu-test-partition \
    "${raw_cpu_examples[@]}" -- "${wrapper_cpu_examples[@]}"
  run_step cpu-test-binding-projection-revalidation \
    env "${loader_environment_removals[@]}" \
    "${CARGO_FE2O3_BINARY}" examples check-wrapper-managed \
    "${wrapper_managed_packages[@]}"
  run_step dialect-mir-pliron-tests \
    cargo test --locked -p dialect-mir --features pliron --test pliron_shell
}

run_auxiliary_tests() {
  run_workspace_dependency_bootstrap auxiliary
  # The default core is a DeviceCopy-only contract crate. Deprecated HIP/HSA
  # compatibility is qualified separately and never enters default resolution.
  run_step core-doc-tests cargo test --locked --doc -p fe2o3-core
  run_step device-copy-renamed-dependency \
    cargo check --locked -p device-copy-renamed-dependency
  run_step device-copy-derive-real-trait \
    cargo check --locked -p fe2o3-core --test device_copy_derive_compile
  run_step device-copy-derive-ui \
    cargo test --locked -p fe2o3-core --test device_copy_derive_ui
  run_step core-production-runtime-surface-ui \
    env FE2O3_HIP_SYS_DISABLE=1 \
      cargo test --locked -p fe2o3-core --test production_runtime_surface_ui
  run_step kfd-default-dependency-closure \
    cargo test --locked -p cargo-fe2o3 --test production_dependency_closure
  run_step deprecated-hip-core-qualification \
    env FE2O3_HIP_SYS_DISABLE=1 \
      cargo test --locked -p fe2o3-core \
        --features qualification-legacy-hip-runtime
  run_step deprecated-hip-hsa-host-qualification \
    env FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
      cargo test --locked -p fe2o3-host \
        --features qualification-legacy-hip-hsa
  run_step deprecated-hsa-runtime-qualification \
    env FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
      cargo test --locked -p fe2o3-hsa-runtime \
        --features qualification-legacy-hsa-runtime
  run_step compiler-execution-systemd-contract \
    bash scripts/tests/compiler-execution-systemd.sh
  run_step compiler-execution-deployment-bundle-contract \
    bash scripts/tests/compiler-execution-deployment-bundle.sh
  run_step compiler-execution-qualification-base-contract \
    bash scripts/tests/compiler-execution-qualification-base.sh
  run_step proof-manager-systemd-contract \
    bash scripts/tests/proof-manager-systemd.sh
  run_step native-application-manager-systemd-contract \
    python3 -I -B scripts/tests/native-application-manager-systemd.py
  run_step native-application-startup-observer \
    python3 -I -B scripts/tests/native_application_startup_test.py
  run_step native-manager-packaged-cpu-observer \
    python3 -I -B scripts/tests/qualify_native_manager_cpu_test.py
  run_step native-application-qualification-oracle \
    python3 -I -B scripts/tests/qualify_native_conditional_application_test.py
  run_step native-cpu-chain-proof-oracle \
    python3 -I -B scripts/tests/qualify_native_conditional_cpu_chain_test.py
  run_step native-application-fixture-oracles \
    cargo test --locked -p cargo-fe2o3 \
      --test native_v5_case --test native_v5_roster_case --test native_proof_case
  run_step runtime-planner-input-checker-tests \
    python3 -I -B crates/fe2o3-runtime-model/verus/test-producer-planner-input-composition.py
  run_step runtime-planner-input-source-check \
    python3 -I -B crates/fe2o3-runtime-model/verus/check-producer-planner-input-composition.py
  run_step functional-refinement-qualification-inputs \
    bash scripts/tests/functional-refinement-qualification-inputs.sh
  run_step qualification-input-bundle-tests \
    python3 -I -B scripts/tests/qualification_input_bundle_test.py
  run_step qualification-host-link-tests \
    python3 -I -B scripts/tests/qualification_host_link_test.py
  run_step qualification-host-link-shell-tests \
    python3 -I -B scripts/tests/qualification_host_link_shell_test.py
  run_step two-gpu-mount-contract \
    bash scripts/tests/qualify-two-gpu-mounts.sh
  run_step device-roster-mount-contract \
    bash scripts/tests/qualify-device-roster-mounts.sh
  run_step native-readiness-fixture-builder-contract \
    bash scripts/tests/native-ready-fixture.sh
  run_step s09-debug-checker bash scripts/tests/s09-debug.sh
}

run_shard_policy() {
  run_step rustc-codegen-shard-policy \
    python3 "${RUSTC_CODEGEN_SHARD_POLICY}" check
}

run_workspace_dependency_policy() {
  run_step workspace-dependency-policy-tests \
    python3 "${WORKSPACE_DEPENDENCY_POLICY_TESTS}"
  run_step workspace-dependency-policy \
    python3 "${WORKSPACE_DEPENDENCY_POLICY_CHECKER}" \
      --policy "${WORKSPACE_DEPENDENCY_POLICY}"
  run_step pliron-dependency-policy-tests \
    python3 "${PLIRON_DEPENDENCY_POLICY_TESTS}"
  run_step pliron-dependency-policy \
    python3 "${PLIRON_DEPENDENCY_POLICY_CHECKER}"
}

run_hygiene_delta_policy() {
  local base_sha="$1"
  local head_sha="$2"
  run_step hygiene-delta-policy-tests \
    env PYTHONDONTWRITEBYTECODE=1 python3 "${HYGIENE_DELTA_POLICY_TESTS}"
  run_step hygiene-delta-policy \
    python3 "${HYGIENE_DELTA_POLICY}" --base "${base_sha}" --head "${head_sha}"
}

run_standalone_lockfiles() {
  if ((STANDALONE_LOCKFILES_CHECKED)); then
    return 0
  fi
  run_step standalone-lockfiles bash "${STANDALONE_LOCKFILE_CHECKER}"
  STANDALONE_LOCKFILES_CHECKED=1
}

run_runtime_pure_rust_policy() {
  run_step runtime-pure-rust-audit-tests \
    env PYTHONDONTWRITEBYTECODE=1 python3 "${RUNTIME_PURE_RUST_AUDIT_TESTS}"
  run_step runtime-identity-oracle-parser-tests \
    env PYTHONDONTWRITEBYTECODE=1 python3 "${RUNTIME_IDENTITY_ORACLE_TESTS}"
  run_step virtual-runtime-no-gpu-metadata \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${VIRTUAL_RUNTIME_NO_GPU_POLICY}" metadata --cargo \
      --root fe2o3-virtual-runtime \
      --root fe2o3-virtual-runtime-cli \
      --root fe2o3-sim-differential
  run_step virtual-runtime-no-gpu-build \
    env CARGO_TARGET_DIR="${RUNTIME_PURE_RUST_TARGET_DIR}" \
      cargo build --locked -p fe2o3-virtual-runtime-cli \
        --bin fe2o3-virtual-runtime
  run_step virtual-runtime-no-gpu-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${VIRTUAL_RUNTIME_NO_GPU_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/fe2o3-virtual-runtime"
  run_step sim-differential-no-gpu-build \
    env CARGO_TARGET_DIR="${RUNTIME_PURE_RUST_TARGET_DIR}" \
      cargo build --locked -p fe2o3-sim-differential \
        --bin fe2o3-sim-differential
  run_step sim-differential-no-gpu-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${VIRTUAL_RUNTIME_NO_GPU_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/fe2o3-sim-differential"
  run_step runtime-pure-rust-metadata \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" metadata --cargo \
      --root fe2o3-kfd \
      --root fe2o3-drm-uapi \
      --root fe2o3-kfd-uapi \
      --root fe2o3-amdhsa-loader \
      --root fe2o3-aql \
      --root fe2o3-runtime \
      --root fe2o3-runtime-model \
      --root fe2o3-sim-runtime
  run_step sim-runtime-no-gpu-metadata \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${SIM_RUNTIME_NO_GPU_POLICY}" metadata --cargo \
      --root fe2o3-sim-runtime
  run_step sim-runtime-no-gpu-build \
    env CARGO_TARGET_DIR="${RUNTIME_PURE_RUST_TARGET_DIR}" \
      cargo build --locked -p fe2o3-sim-runtime \
        --example sim-runtime-evidence
  run_step runtime-pure-rust-kfd-examples-build \
    env CARGO_TARGET_DIR="${RUNTIME_PURE_RUST_TARGET_DIR}" \
      cargo build --locked -p fe2o3-kfd \
        --example kfd-version \
        --example kfd-topology \
        --example kfd-device-identity \
        --example kfd-host-visible-memory-policy \
        --example kfd-shared-gtt-memory-policy \
        --example kfd-queue-resources \
        --example kfd-compute-aql-queue-policy
  run_step runtime-pure-rust-dispatch-diagnostic-build \
    env CARGO_TARGET_DIR="${RUNTIME_PURE_RUST_TARGET_DIR}" \
      cargo build --locked -p fe2o3-runtime --features hardware-diagnostic \
        --example gfx942-lds-diagnostic
  run_step runtime-pure-rust-kfd-version-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-version"
  run_step runtime-pure-rust-kfd-topology-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-topology"
  run_step runtime-pure-rust-kfd-device-identity-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-device-identity"
  run_step runtime-pure-rust-dispatch-diagnostic-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/gfx942-lds-diagnostic"
  run_step sim-runtime-no-gpu-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${SIM_RUNTIME_NO_GPU_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/sim-runtime-evidence"
  run_step runtime-pure-rust-kfd-memory-policy-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-host-visible-memory-policy"
  run_step runtime-pure-rust-kfd-shared-memory-policy-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-shared-gtt-memory-policy"
  run_step runtime-pure-rust-kfd-queue-resources-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-queue-resources"
  run_step runtime-pure-rust-kfd-compute-aql-queue-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-compute-aql-queue-policy"
}

run_runtime_identity_oracle() {
  run_step runtime-identity-oracle bash "${RUNTIME_IDENTITY_ORACLE}"
}

load_rustc_codegen_shards() {
  local destination_name="$1"
  local output
  # shellcheck disable=SC2178  # The destination is a caller-owned array nameref.
  local -n destination="${destination_name}"
  if ! output="$(python3 "${RUSTC_CODEGEN_SHARD_POLICY}" list)"; then
    return 2
  fi
  destination=()
  # shellcheck disable=SC2034  # The destination is written through a nameref.
  mapfile -t destination <<<"${output}"
}

load_rustc_codegen_shard_targets() {
  local shard_id="$1"
  local destination_name="$2"
  local output
  # shellcheck disable=SC2178  # The destination is a caller-owned array nameref.
  local -n destination="${destination_name}"
  if ! output="$(python3 "${RUSTC_CODEGEN_SHARD_POLICY}" tests "${shard_id}")"; then
    return 2
  fi
  destination=()
  # shellcheck disable=SC2034  # The destination is written through a nameref.
  mapfile -t destination <<<"${output}"
}

run_rustc_codegen_lib_tests() {
  # Offline build-std tests resolve the selected toolchain's complete sysroot
  # lockfile, which is not part of the ordinary workspace dependency closure.
  run_step rustc-codegen-sysroot-dependencies \
    bash "${SCRIPT_DIR}/fetch-rustc-sysroot.sh"
  # Do not combine this with integration targets: Cargo can emit a test rlib
  # and an unversioned backend dylib with different Rust symbol hashes.
  # Keep the aggregate rustc-private harness bounded like the isolated targets;
  # full debuginfo can exceed the executable identity measurement limit.
  run_step rustc-codegen-lib-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --lib
  run_step rustc-codegen-expanded-source-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 \
      cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --lib \
        production_rustc_driver_v1::checked_output_source_v1_tests::context_source_v29_tests::pending_source_tests::expanded_source_tests::actual_rustc_source_retains_original_neutral_and_expanded_owners -- \
        --ignored --exact --test-threads=1
  run_step rustc-codegen-expanded-model-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 \
      cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --lib \
        production_rustc_driver_v1::checked_output_source_v1_tests::context_source_v29_tests::pending_source_tests::expanded_source_tests::expanded_model_tests::actual_rustc_expanded_support_model_covers_complete_roots_and_runtime_width_boundaries -- \
        --ignored --exact --test-threads=1
  run_step rustc-codegen-product-frame-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 \
      cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --lib \
        production_rustc_driver_v1::checked_output_source_v1_tests::context_source_v29_tests::pending_source_tests::expanded_source_tests::expanded_model_tests::product_frames::actual_rustc_product_carriers_retain_current_and_suspended_source_demands_v283 -- \
        --ignored --exact --test-threads=1
  run_step rustc-codegen-tile-census-source-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 \
      cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --lib \
        production_rustc_driver_v1::checked_output_source_v1_tests::context_source_v29_tests::pending_source_tests::expanded_source_tests::tile_census_tests::actual_rustc_collected_tile_census_preserves_shared_body_counts -- \
        --ignored --exact --test-threads=1
  run_step rustc-codegen-extractor-bin-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --bin fe2o3-rustc-extract
  run_step rustc-codegen-exporter-bin-tests \
    env CARGO_PROFILE_DEV_DEBUG=1 cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --bin fe2o3-export-sim
}

run_rustc_codegen_target() {
  local test_target="$1"
  local -a command=(
    env CARGO_PROFILE_DEV_DEBUG=1
    cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}"
    --test "${test_target}"
  )

  # Cargo can emit a test rlib and an unversioned backend dylib with different
  # Rust symbol hashes during one --all-targets build. Each target-isolated
  # invocation produces the exact backend dylib before running its linked test.
  # Match the limited-debug profile used by the production automatic backend
  # builder so clean and cache-restored shards exercise the same bounded image.
  run_step "rustc-codegen-test-${test_target}" \
    "${command[@]}"
}

run_rustc_codegen_shard_targets() {
  local shard_id="$1"
  local -a test_targets
  local test_target
  load_rustc_codegen_shard_targets "${shard_id}" test_targets
  for test_target in "${test_targets[@]}"; do
    run_rustc_codegen_target "${test_target}"
  done
}

run_all_rustc_codegen_shards() {
  local -a shard_ids
  local shard_id
  load_rustc_codegen_shards shard_ids
  for shard_id in "${shard_ids[@]}"; do
    run_rustc_codegen_shard_targets "${shard_id}"
  done
}

run_rustc_codegen_shard() {
  local shard_id="$1"
  run_shard_policy
  run_rustc_codegen_shard_targets "${shard_id}"
}

run_rustc_codegen_tests() {
  run_shard_policy
  run_rustc_codegen_lib_tests
  run_all_rustc_codegen_shards
}

run_tests() {
  run_cpu_tests
  run_rustc_codegen_tests
  run_auxiliary_tests
}

run_workspace_tests() {
  local -a rustc_examples wrapper_managed_packages wrapper_examples
  local -a loader_environment_removals
  local -a workspace_args=(
    test --locked --workspace --all-targets
    --exclude "${RUSTC_CODEGEN_TEST_PACKAGE}"
    --exclude fe2o3-artifact-transaction
  )
  local -a wrapper_args=(test --locked --all-targets)
  local -A rustc_example_set=()
  local package

  ensure_production_cargo_fe2o3_driver workspace-tests
  load_example_packages rustc-check rustc_examples "${CARGO_FE2O3_BINARY}"
  load_example_packages wrapper-managed wrapper_managed_packages \
    "${CARGO_FE2O3_BINARY}"
  for package in "${rustc_examples[@]}"; do
    rustc_example_set["${package}"]=1
  done
  for package in "${wrapper_managed_packages[@]}"; do
    if [[ -n "${rustc_example_set[${package}]:-}" ]]; then
      wrapper_examples+=("${package}")
      workspace_args+=(--exclude "${package}")
      wrapper_args+=(-p "${package}")
    fi
  done
  ((${#wrapper_examples[@]} > 0)) || {
    printf '%s\n' 'workspace contains no binding-managed test example' >&2
    return 2
  }

  run_step workspace-tests \
    cargo "${workspace_args[@]}"
  run_runtime_release_tests
  load_dynamic_loader_environment_removals loader_environment_removals
  run_step workspace-binding-example-tests \
    env "${loader_environment_removals[@]}" FE2O3_HIP_SYS_DISABLE=1 \
    "${CARGO_FE2O3_BINARY}" "${wrapper_args[@]}"
  run_step workspace-binding-example-revalidation \
    env "${loader_environment_removals[@]}" \
    "${CARGO_FE2O3_BINARY}" examples check-wrapper-managed \
      "${wrapper_managed_packages[@]}"
  run_artifact_transaction_tests
  run_rustc_codegen_tests
}

run_backend_build() {
  run_step backend-check-tests python3 -B scripts/tests/check-rustc-codegen-backend.py
  run_step backend-build python3 -B scripts/check-rustc-codegen-backend.py
  run_step backend-all-features-build \
    python3 -B scripts/check-rustc-codegen-backend.py --all-features
}

run_verus() {
  local default_verus="${VERUS:-verus}"
  local runtime_model_verus="${FE2O3_RUNTIME_MODEL_VERUS:-${default_verus}}"
  run_step runtime-model-verus \
    env VERUS="${runtime_model_verus}" \
      "${REPO_ROOT}/crates/fe2o3-runtime-model/verus/verify-verus.sh"
  run_step verus-fixtures \
    env VERUS="${default_verus}" \
      "${REPO_ROOT}/examples/verus_vecadd/run-verus.sh" --require
  run_step scalar-gemm-verus \
    env VERUS="${default_verus}" \
      "${REPO_ROOT}/examples/scalar_gemm_v1/run-verus.sh" --require
  run_step mir-pliron-per-compilation-verus \
    env VERUS="${default_verus}" \
      "${REPO_ROOT}/scripts/test-mir-pliron-per-compilation-verus.sh"
  run_step qwen3-rope-kv-verus \
    env VERUS="${default_verus}" \
      sh "${REPO_ROOT}/examples/qwen3_rope_kv_v1/run-verus.sh"
}

run_authority_launcher_tests() {
  run_step authority-launcher-tests \
    bash scripts/tests/cargo-fe2o3-authority-launcher.sh
}

require_source_isa_protected_environment() {
  local label="$1"
  local name
  for name in "${SOURCE_ISA_PROTECTED_ENVIRONMENT[@]}"; do
    if [[ -z "${!name:-}" ]]; then
      printf '%s requires %s\n' "${label}" "${name}" >&2
      return 2
    fi
  done
}

run_source_isa_unit_matrix() {
  if [[ "${FE2O3_RUN_SOURCE_ISA_UNIT_MATRIX:-}" != "1" ]]; then
    printf '%s\n' \
      'protected source/ISA unit matrix requires FE2O3_RUN_SOURCE_ISA_UNIT_MATRIX=1' >&2
    return 2
  fi
  local platform_architecture platform_kernel
  platform_kernel="$(uname -s)" || {
    printf '%s\n' 'protected source/ISA unit matrix could not identify the host kernel' >&2
    return 2
  }
  platform_architecture="$(uname -m)" || {
    printf '%s\n' 'protected source/ISA unit matrix could not identify the host architecture' >&2
    return 2
  }
  if [[ "${platform_kernel}" != "Linux" || "${platform_architecture}" != "x86_64" ]]; then
    printf 'protected source/ISA unit matrix requires Linux x86_64, found %s %s\n' \
      "${platform_kernel}" "${platform_architecture}" >&2
    return 2
  fi
  require_source_isa_protected_environment 'protected source/ISA unit matrix' || return $?
  run_step source-isa-unit-matrix \
    cargo test --locked -p cargo-fe2o3 --bin cargo-fe2o3 \
      production_source_isa_unit_matrix_v1::ordinary_source_units_round_trip_through_the_production_observer_on_both_targets -- \
      --ignored --exact --test-threads=1 --nocapture
}

run_source_isa_characteristic_contract_v2() {
  if [[ "${FE2O3_RUN_SOURCE_ISA_CHARACTERISTIC_CONTRACT_V2:-}" != "1" ]]; then
    printf '%s\n' \
      'source/ISA characteristic V2 contract requires FE2O3_RUN_SOURCE_ISA_CHARACTERISTIC_CONTRACT_V2=1' >&2
    return 2
  fi
  local platform_architecture platform_kernel
  platform_kernel="$(uname -s)" || {
    printf '%s\n' 'source/ISA characteristic V2 contract could not identify the host kernel' >&2
    return 2
  }
  platform_architecture="$(uname -m)" || {
    printf '%s\n' 'source/ISA characteristic V2 contract could not identify the host architecture' >&2
    return 2
  }
  if [[ "${platform_kernel}" != "Linux" || "${platform_architecture}" != "x86_64" ]]; then
    printf 'source/ISA characteristic V2 contract requires Linux x86_64, found %s %s\n' \
      "${platform_kernel}" "${platform_architecture}" >&2
    return 2
  fi
  run_step source-isa-characteristic-contract-v2 \
    cargo test --locked -p cargo-fe2o3 --bin cargo-fe2o3 \
      production_source_isa_characteristic_matrix_v2:: -- \
      --test-threads=1
}

run_source_isa_characteristic_matrix_v2() {
  if [[ "${FE2O3_RUN_SOURCE_ISA_CHARACTERISTIC_MATRIX_V2:-}" != "1" ]]; then
    printf '%s\n' \
      'protected source/ISA characteristic V2 matrix requires FE2O3_RUN_SOURCE_ISA_CHARACTERISTIC_MATRIX_V2=1' >&2
    return 2
  fi
  local platform_architecture platform_kernel
  platform_kernel="$(uname -s)" || {
    printf '%s\n' 'protected source/ISA characteristic V2 matrix could not identify the host kernel' >&2
    return 2
  }
  platform_architecture="$(uname -m)" || {
    printf '%s\n' 'protected source/ISA characteristic V2 matrix could not identify the host architecture' >&2
    return 2
  }
  if [[ "${platform_kernel}" != "Linux" || "${platform_architecture}" != "x86_64" ]]; then
    printf 'protected source/ISA characteristic V2 matrix requires Linux x86_64, found %s %s\n' \
      "${platform_kernel}" "${platform_architecture}" >&2
    return 2
  fi
  require_source_isa_protected_environment \
    'protected source/ISA characteristic V2 matrix' || return $?
  run_step source-isa-characteristic-matrix-v2 \
    cargo test --locked -p cargo-fe2o3 --bin cargo-fe2o3 \
      production_source_isa_characteristic_matrix_v2::production_adapter_v2::ordinary_source_units_preserve_characteristic_facts_on_both_targets_v2 -- \
      --ignored --exact --test-threads=1 --nocapture
}

run_rustc_trampoline_tests() {
  run_step rustc-trampoline-tests \
    bash scripts/tests/fe2o3-rustc-trampoline.sh
  run_step cargo-binding-trampoline-tests \
    bash scripts/tests/fe2o3-cargo-binding-trampoline.sh
}

run_parity_matrix_checks() {
  run_step parity-matrix-check bash scripts/parity-matrix.sh check
  run_step parity-matrix-tests bash scripts/tests/parity-matrix.sh
  run_step runtime-parity-report-tests \
    python3 scripts/tests/test_runtime_parity_report.py
  run_step parity-evidence-tests bash scripts/tests/parity-evidence.sh
  run_step parity-snapshot-tests bash scripts/tests/run-parity-snapshot.sh
  run_step parity-oci-executor-tests \
    bash scripts/tests/parity-oci-executor.sh
  run_step parity-oci-operator-tests \
    bash scripts/tests/parity-oci-operator.sh
  run_authority_launcher_tests
  run_rustc_trampoline_tests
  run_step parity-row-evidence-tests \
    bash scripts/tests/parity-row-evidence.sh
  run_step parity-publisher-client-tests \
    python3 scripts/tests/parity-publisher-client.py
  run_step parity-signed-evidence-fd-tests \
    python3 scripts/tests/parity-signed-evidence-fd.py
  run_step parity-repository-rules-tests \
    bash scripts/tests/parity-repository-rules.sh
  run_step canonical-repository-rules-tests \
    bash scripts/tests/canonical-repository-rules.sh
  run_step canonical-release-controls-tests \
    bash scripts/tests/canonical-release-controls.sh
  run_step dco-range-tests bash scripts/tests/check-dco-range.sh
  run_step mi300x-evidence-queue-tests \
    bash scripts/tests/mi300x-evidence-queue.sh
  run_step hosted-parity-ci-tests \
    bash scripts/tests/hosted-parity-ci.sh
}

run_generic_core() {
  if (($# > 1)); then
    printf 'generic-core accepts at most one group\n' >&2
    return 2
  fi
  local group="${1-all}"
  case "${group}" in
    all | policy) ;;
    cpu)
      run_cpu_tests
      return
      ;;
    cpu-foundation | cpu-analysis | cpu-lowering | cpu-pliron | cpu-finalize)
      run_cpu_package_group "${group#cpu-}"
      return
      ;;
    cpu-finalize-[0-3])
      run_cpu_finalizer_shard "${group##*-}"
      return
      ;;
    cpu-integration)
      run_cpu_tests integration
      return
      ;;
    auxiliary)
      run_rustc_codegen_lib_tests
      run_auxiliary_tests
      return
      ;;
    *)
      printf 'unknown generic-core group: %s\n' "${group}" >&2
      return 2
      ;;
  esac
  run_workspace_dependency_policy
  run_standalone_lockfiles
  run_runtime_pure_rust_policy
  run_step example-manifest \
    cargo run --quiet --locked -p cargo-fe2o3 -- examples check
  run_step bounded-moe-docs \
    python3 scripts/test-bounded-moe-docs.py
  run_shard_policy
  run_parity_matrix_checks
  run_format
  run_check
  run_backend_build
  run_step compiler-map-evidence-tests \
    python3 -I -B scripts/tests/compiler_map_evidence_test.py
  run_step simulation-expectation-tests \
    python3 -I -B scripts/tests/simulation_expectation.py
  run_step tutorial-scalar-gemm-corpus-tests \
    python3 -I -B scripts/tests/tutorial_scalar_gemm_corpus.py
  run_step tutorial-default-cargo-harness-tests \
    python3 -I -B scripts/tests/tutorial_default_cargo.py
  run_step tutorial-production-census-tests \
    python3 -I -B scripts/tests/tutorial_production_census_v91.py
  run_step tutorial-current-simulation-tests \
    python3 -I -B scripts/tests/tutorial_simulation_v92.py
  run_step quickstart-shell-tests bash scripts/tests/quickstart.sh
  run_step kernel-compile-matrix-shell-tests \
    bash scripts/tests/kernel-compile-matrix.sh
  run_step tutorial-cpu-reference-tests \
    python3 -B scripts/tests/tutorial_cpu_reference.py
  run_step no-gpu-source-quickstart bash scripts/quickstart.sh no-gpu
  run_step kir-sim-capability-matrix \
    cargo test --locked -p fe2o3-kir-sim --test capability_matrix
  run_step kir-sim-scalar-differential \
    cargo run --quiet --locked -p fe2o3-sim-differential --bin fe2o3-sim-differential -- \
      --seed-start 0 --cases 256
  run_step kir-sim-semantic-differential \
    cargo run --quiet --locked -p fe2o3-sim-differential --bin fe2o3-sim-differential -- \
      semantic-run-v2 --seed 0
  run_step kir-sim-f32-differential \
    cargo run --quiet --locked -p fe2o3-sim-differential --bin fe2o3-sim-differential -- \
      f32-run-v3
  run_step ci-local-test-gate bash scripts/tests/ci-local-test-gate.sh
  run_step generic-core-group-tests \
    python3 -I -B scripts/tests/generic-core-groups.py
  run_step generic-cpu-group-tests \
    python3 -I -B scripts/tests/generic-cpu-groups.py
  run_step finalizer-test-shard-tests \
    python3 -I -B scripts/tests/finalizer-test-shards.py
  run_step runtime-production-proof-pipeline-tests \
    python3 -I -B scripts/tests/runtime-production-proof-pipeline.py
  if [[ "${group}" == all ]]; then
    run_generic_core cpu
    run_generic_core auxiliary
  fi
}

run_generic() {
  run_generic_core
  run_all_rustc_codegen_shards
}

run_rocm_compile() {
  unset FE2O3_WORKER_V2_CONFIG_V2
  export FE2O3_TARGET="${FE2O3_TARGET:-gfx942}"
  [[ "${FE2O3_TARGET}" == gfx942 ]] || {
    printf 'production ROCm compilation requires FE2O3_TARGET=gfx942, got %s\n' \
      "${FE2O3_TARGET}" >&2
    return 2
  }
  local -a loader_environment_removals wrapper_managed_packages

  prepare_cargo_fe2o3_driver rocm production
  load_dynamic_loader_environment_removals loader_environment_removals
  validate_cargo_fe2o3_driver
  load_example_packages wrapper-managed wrapper_managed_packages \
    "${CARGO_FE2O3_BINARY}"
  validate_managed_wrapper_source_namespaces \
    "${wrapper_managed_packages[@]}" \
    "${ROCM_TRUSTED_DEVICE_ITEM_PACKAGES[@]}"
  validate_cargo_fe2o3_driver
  run_step rocm-doctor \
    env "${loader_environment_removals[@]}" \
      "${CARGO_FE2O3_BINARY}" doctor --require-gfx942-and-tools-present
  run_step rocm-production-extraction-safe-kernel \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_extraction_driver_v1 \
        attributed_kernel_is_recollected_inside_a_real_amdgcn_dependency_graph -- \
        --ignored --exact
  run_step rocm-production-extraction-memory-free-kernel \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_extraction_driver_v1 \
        ordinary_memory_free_kernel_reaches_ranked_verification -- \
        --ignored --exact
  run_step rocm-production-extraction-unsafe-rejection \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_extraction_driver_v1 \
        production_collector_rejects_reachable_unsafe_rust_with_rooted_diagnostics -- \
        --ignored --exact
  run_step rocm-production-extraction-inline-source-safety \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_extraction_driver_v1 \
        inlined_source_safety_v1::ordinary_inline_helper_collection_rejects_erased_empty_unsafe_blocks -- \
        --ignored --exact
  run_step rocm-production-general-matrix \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_general_matrix_driver_v1 \
        dynamic_matrix_kernel_reaches_gfx942_llvm -- \
        --ignored --exact
  run_step rocm-production-general-attention \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_general_matrix_driver_v1 \
        dynamic_attention_kernel_reaches_gfx942_llvm -- \
        --ignored --exact
  run_step rocm-production-transaction \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_pipeline \
        attributed_kernel_enters_one_transaction_and_fails_without_fallback -- \
        --ignored --exact
  run_step rocm-production-ranked-bounds \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_rust_bounds_and_production_pliron_pipeline_fail_closed -- \
        --ignored --exact
  run_step rocm-production-write-only-witness-mappings \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        write_only_witness_mappings_retain_exact_ranked_predicates -- \
        --ignored --exact
  run_step rocm-production-barrier-cfg \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        production_barrier_cfg_preserves_order_and_fails_closed -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-gfx942 \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_kernel_source_exports_one_verified_authority_free_simulation_bundle -- \
        --ignored --exact
  run_step rocm-production-explicit-bin-source-dependencies \
    env "${loader_environment_removals[@]}" \
      cargo fetch --locked --manifest-path \
        crates/rustc-codegen-fe2o3/tests/fixtures/production-explicit-bin-device/Cargo.toml
  run_step rocm-production-explicit-bin-source-sim \
    env "${loader_environment_removals[@]}" CARGO_PROFILE_DEV_DEBUG=1 \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        explicit_bin_source_exports_and_simulates_its_own_kernel_not_the_same_named_library -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-gfx950 \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_kernel_source_exports_the_exact_gfx950_simulation_target -- \
        --ignored --exact
  run_step rocm-production-simulation-float-casts \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_float_casts_saturate_in_simulation -- \
        --ignored --exact
  run_step rocm-production-simulation-wrapping-integers \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_wrapping_integers_match_rust_in_simulation -- \
        --ignored --exact
  run_step rocm-production-simulation-launch-wrapping-integers \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_launch_wrapping_integers_match_rust_in_simulation -- \
        --ignored --exact
  run_step rocm-production-simulation-rust-call \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_rust_call_closures_match_rust_in_simulation -- \
        --ignored --exact
  run_step rocm-production-simulation-slice-metadata-arguments \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_slice_metadata_arguments_match_rust_in_simulation -- \
        --ignored --exact
  run_step rocm-production-simulation-dynamic-local-array \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_dynamic_local_array_matches_rust_in_simulation -- \
        --ignored --exact
  run_step rocm-production-generative-provider-refusal \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        staged_generative_providers_reject_without_export_authority -- \
        --ignored --exact
  run_step rocm-production-generative-provider-forgery \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_extraction_driver_v1 \
        forged_capability_markers_cannot_gain_nominal_authority -- \
        --ignored --exact
  run_step rocm-production-simulation-sqrt \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_sqrt_executes_exact_binary32_in_simulation -- \
        --ignored --exact
  run_step rocm-production-integer-reduction-overflow \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_source_workgroup_reductions_wrap_on_overflow -- \
        --ignored --exact
  run_step rocm-production-scalar-casts \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p fe2o3-amdgcn-model \
        --test scalar_v2 \
        rocm_clang_compiles_every_accepted_gfx942_scalar_path -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v2-source-variables \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_kernel_sources_export_and_query_exact_v2_source_variables -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v2-invalid-name \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        v2_rejects_an_overbound_debug_name_without_inspecting_it_on_v1 -- \
        --ignored --exact
  run_step rocm-production-scan-bundle-v5-cpu \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_neutral_workgroup_reduce_driver_v1 \
        ordinary_scan_sources_export_v5_and_execute_every_cpu_observation_path -- \
        --ignored --exact
  run_step rocm-production-row-affine-source-sim \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_neutral_workgroup_reduce_driver_v1 \
        ordinary_row_affine_source_matches_oracle_and_replay -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v3-typed-layouts \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_rust_exports_and_queries_exact_v3_typed_layouts_and_regions -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v4-aggregate-components \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_rust_struct_argument_exports_exact_v4_components -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v5-recursive-aggregates \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_recursive_aggregates_export_and_execute_bundle_v5 -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v4-recursive-refusals \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_recursive_aggregates_export_and_unsafe_shapes_fail_typed -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v5-wave-debugger \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_rust_v9_wave_collective_exports_v5_and_runs_in_public_debugger -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v5-workgroup-cpu \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_ranked_bounds_driver_v1 \
        ordinary_rust_workgroup_reductions_export_v5_and_execute_every_cpu_path -- \
        --ignored --exact
  run_step rocm-production-simulation-bundle-v6-nested-control-flow \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_semantic_conformance_v3 \
        ordinary_rust_nested_control_flow_executes_after_production_ssa_lowering -- \
        --ignored --exact
  run_step rocm-production-integer-scan-overflow \
    env "${loader_environment_removals[@]}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
        --test production_neutral_workgroup_reduce_driver_v1 \
        ordinary_integer_scan_sources_wrap_on_overflow -- \
        --ignored --exact
  run_step rocm-g1-code-object \
    cargo test --locked -p dialect-amdgcn --test lowering \
      rocm_compiles_the_golden_to_an_amdgpu_code_object -- \
      --ignored --exact
  run_step rocm-kernel-compile-matrix \
    bash scripts/kernel-compile-matrix.sh gfx942
}

require_gpu_access() {
  if [[ ! -c /dev/kfd || ! -r /dev/kfd || ! -w /dev/kfd ]]; then
    printf '%s\n' 'GPU smoke requires read/write access to /dev/kfd' >&2
    return 2
  fi
}

run_hardware_smoke() {
  if [[ "${FE2O3_ALLOW_GPU_SMOKE:-}" != "1" ]]; then
    printf '%s\n' \
      'refusing to run GPU smoke without FE2O3_ALLOW_GPU_SMOKE=1' >&2
    return 2
  fi
  if [[ -z "${FE2O3_TARGET:-}" ]]; then
    printf '%s\n' \
      'hardware HSACO inspection requires an explicit FE2O3_TARGET' >&2
    return 2
  fi
  require_gpu_access
  case "${FE2O3_TARGET}" in
    gfx942 | gfx942:xnack-) ;;
    *)
      printf 'KFD hardware smoke requires FE2O3_TARGET=gfx942[:xnack-], got %s\n' \
        "${FE2O3_TARGET}" >&2
      return 2
      ;;
  esac

  run_step hardware-runtime-identity-oracle \
    env FE2O3_ALLOW_RUNTIME_IDENTITY_ORACLE=1 \
      bash "${RUNTIME_IDENTITY_ORACLE}"
  run_step hardware-kfd-device-identity \
    cargo run --locked -p fe2o3-kfd --example kfd-device-identity -- --all
  run_step hardware-kfd-host-visible-memory \
    cargo run --locked -p fe2o3-kfd --features live-validation \
      --example kfd-host-visible-memory -- --all
  run_step hardware-kfd-compute-aql-queue \
    cargo run --locked -p fe2o3-kfd --features live-validation \
      --example kfd-compute-aql-queue -- --all
  run_step hardware-kfd-profiler-smoke \
    bash scripts/kfd-profiler-hardware-smoke.sh
  run_step hardware-kfd-debug-trap-live \
    cargo test --locked -p fe2o3-kfd --features live-validation \
      --test kfd_debug_trap_live -- \
      --exact mi300x_ptrace_runtime_handshake_and_typed_gate \
      --nocapture --test-threads=1
  run_step hardware-kfd-debug-protocol-v2 \
    cargo test --locked -p fe2o3-debug-cli --features live-validation \
      --test hardware_v2_live -- --test-threads=1
  run_step hardware-kfd-live-gpu-debug-v3 \
    cargo test --locked -p fe2o3-debug-cli --features live-validation \
      --test live_kfd_v3_live -- \
      --exact mi300x_live_kfd_v3_binds_observes_controls_and_terminates \
      --nocapture --test-threads=1
  if [[ -n "${FE2O3_TEST_SOURCE_AUTH_LDS_GFX942_HSACO:-}" || \
        -n "${FE2O3_KFD_DIAGNOSTIC_UNIQUE_ID:-}" ]]; then
    if [[ -z "${FE2O3_TEST_SOURCE_AUTH_LDS_GFX942_HSACO:-}" || \
          -z "${FE2O3_KFD_DIAGNOSTIC_UNIQUE_ID:-}" ]]; then
      printf '%s\n' \
        'KFD LDS diagnostic requires both FE2O3_TEST_SOURCE_AUTH_LDS_GFX942_HSACO and FE2O3_KFD_DIAGNOSTIC_UNIQUE_ID' >&2
      return 2
    fi
    run_step hardware-kfd-lds-diagnostic \
      cargo run --locked -p fe2o3-runtime --features hardware-diagnostic \
        --example gfx942-lds-diagnostic -- \
        "${FE2O3_KFD_DIAGNOSTIC_UNIQUE_ID}" \
        "${FE2O3_TEST_SOURCE_AUTH_LDS_GFX942_HSACO}"
  fi
}

run_parity_production_immutable() {
  run_step parity-production-immutable \
    bash scripts/tests/parity-production-immutable-ingest.sh
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}"
  validate_private_directory 'CI log directory' "${LOG_DIR}"

  case "${1:-}" in
    generic) run_generic ;;
    generic-core) run_generic_core "${@:2}" ;;
    workspace-policy) run_workspace_dependency_policy ;;
    hygiene-delta)
      if (($# != 3)); then
        printf '%s\n' 'hygiene-delta requires exactly base and head commit refs' >&2
        return 2
      fi
      run_hygiene_delta_policy "$2" "$3"
      ;;
    standalone-locks) run_standalone_lockfiles ;;
    runtime-policy) run_runtime_pure_rust_policy ;;
    runtime-identity-oracle) run_runtime_identity_oracle ;;
    shard-policy) run_shard_policy ;;
    rustc-codegen-shard)
      if (($# != 2)); then
        printf '%s\n' 'rustc-codegen-shard requires exactly one shard id' >&2
        return 2
      fi
      run_rustc_codegen_shard "$2"
      ;;
    format) run_format ;;
    check) run_check ;;
    test) run_tests ;;
    host-reference) run_host_reference_tests ;;
    workspace-test) run_workspace_tests ;;
    rustc-codegen-test) run_rustc_codegen_tests ;;
    backend) run_backend_build ;;
    authority-launcher) run_authority_launcher_tests ;;
    source-isa-unit-matrix) run_source_isa_unit_matrix ;;
    source-isa-characteristic-contract-v2) run_source_isa_characteristic_contract_v2 ;;
    source-isa-characteristic-matrix-v2) run_source_isa_characteristic_matrix_v2 ;;
    rustc-trampoline) run_rustc_trampoline_tests ;;
    parity-evidence) run_parity_matrix_checks ;;
    parity-production-immutable) run_parity_production_immutable ;;
    verus) run_verus ;;
    rocm-compile) run_rocm_compile ;;
    hardware-smoke) run_hardware_smoke ;;
    -h | --help | help) usage ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi

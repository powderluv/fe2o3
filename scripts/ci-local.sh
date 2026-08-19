#!/usr/bin/env bash

set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly LOG_DIR="${CI_LOG_DIR:-${REPO_ROOT}/target/ci-logs}"
readonly RUSTC_CODEGEN_TEST_PACKAGE="rustc-codegen-fe2o3"
readonly RUSTC_CODEGEN_SHARD_POLICY="${REPO_ROOT}/scripts/rustc-codegen-shards.py"
readonly WORKSPACE_DEPENDENCY_POLICY_CHECKER="${REPO_ROOT}/scripts/workspace_dependency_policy.py"
readonly WORKSPACE_DEPENDENCY_POLICY="${REPO_ROOT}/scripts/workspace-dependency-policy.json"
readonly WORKSPACE_DEPENDENCY_POLICY_TESTS="${REPO_ROOT}/scripts/tests/workspace_dependency_policy.py"
readonly PLIRON_DEPENDENCY_POLICY_CHECKER="${REPO_ROOT}/scripts/pliron_dependency_policy.py"
readonly PLIRON_DEPENDENCY_POLICY_TESTS="${REPO_ROOT}/scripts/tests/pliron_dependency_policy.py"
readonly STANDALONE_LOCKFILE_CHECKER="${REPO_ROOT}/scripts/check-standalone-lockfiles.sh"
readonly RUNTIME_PURE_RUST_AUDITOR="${REPO_ROOT}/scripts/runtime_pure_rust_audit.py"
readonly RUNTIME_PURE_RUST_POLICY="${REPO_ROOT}/scripts/runtime-pure-rust-policy.json"
readonly RUNTIME_PURE_RUST_AUDIT_TESTS="${REPO_ROOT}/scripts/tests/runtime_pure_rust_audit.py"
readonly RUNTIME_IDENTITY_ORACLE_TESTS="${REPO_ROOT}/scripts/tests/runtime_identity_oracle.py"
readonly RUNTIME_IDENTITY_ORACLE="${REPO_ROOT}/scripts/runtime-identity-oracle.sh"
readonly RUNTIME_PURE_RUST_TARGET_DIR="${REPO_ROOT}/target/runtime-pure-rust-policy"
readonly CI_STEP_TIMEOUT_SECONDS="${FE2O3_CI_STEP_TIMEOUT_SECONDS:-3000}"
readonly CI_STEP_KILL_AFTER_SECONDS="${FE2O3_CI_STEP_KILL_AFTER_SECONDS:-15}"

readonly CPU_TEST_PACKAGES=(
  cargo-fe2o3
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
  fe2o3-artifact-transaction
  fe2o3-completion
  fe2o3-compiler-api
  fe2o3-compiler-driver
  fe2o3-artifacts
  fe2o3-contracts
  fe2o3-device
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
  fe2o3-kir-pliron-bridge
  fe2o3-legacy-compiler
  fe2o3-lower-kernel-gpu
  fe2o3-lower-mir-kernel
  fe2o3-macros
  fe2o3-mir-model
  fe2o3-pliron
  fe2o3-pliron-conformance
  fe2o3-proof-contracts
  fe2o3-rustc-front
  fe2o3-rustc-invocation
  fe2o3-service-host
  fe2o3-service-model
  fe2o3-runtime-model
  fe2o3-verifier
  fe2o3-worker-v2-bundle
  reserved-fe2o3-symbols
)

usage() {
  cat <<'EOF'
Usage: scripts/ci-local.sh <command>

Commands:
  generic         Run all validation suitable for a machine without ROCm/GPU
  generic-core    Run generic validation except codegen integration shards
  workspace-policy  Validate workspace ownership and dependency directions
  standalone-locks  Validate every tracked standalone Cargo lockfile
  runtime-policy  Validate the pure-Rust runtime dependency and ELF auditor
  runtime-identity-oracle  Measure MI300X identity against isolated rocminfo; explicit opt-in
  shard-policy    Validate the codegen integration shard assignment
  rustc-codegen-shard <id>  Run one codegen integration shard
  format          Check Rust formatting
  check           Check every workspace target, including example binaries
  test            Run unit tests that do not link or load the HIP runtime
  workspace-test  Run every workspace test target; may require ROCm libraries
  rustc-codegen-test  Run backend library and integration tests without dylib replacement
  backend         Build the rustc codegen backend dylib
  authority-launcher  Run bounded protected build-authority launcher tests
  rustc-trampoline    Run non-integrated static rustc trampoline tests
  parity-evidence Run parity, signed-attestation, and queue shell tests
  parity-production-immutable  Run opt-in root ext4/XFS ingestion test
  verus           Run positive and negative Verus proof fixtures; requires Verus
  rocm-compile    Compile every example to host code and HSACO; requires ROCm
  hardware-smoke  Build and run every example; requires an AMD GPU and opt-in
  s09-debug-hardware  Run exact gfx942 direct-link and ROCgdb evidence; explicit opt-in
EOF
}

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  if [[ ! "${CI_STEP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
    ((CI_STEP_TIMEOUT_SECONDS >= 3600)); then
    printf '%s\n' \
      'FE2O3_CI_STEP_TIMEOUT_SECONDS must be an integer from 1 through 3599' >&2
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
  printf '\n   timeout: %ss' "${CI_STEP_TIMEOUT_SECONDS}"
  printf '\n   log: %s\n' "${log_file}"

  set +e
  timeout --signal=TERM --kill-after="${CI_STEP_KILL_AFTER_SECONDS}s" \
    "${CI_STEP_TIMEOUT_SECONDS}s" "$@" 2>&1 | tee "${log_file}"
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
        "${name}" "${CI_STEP_TIMEOUT_SECONDS}" >&2
    fi
    printf 'step %s failed with status %d\n' "${name}" "${status}" >&2
    return "${status}"
  fi
}

load_example_packages() {
  local lane="$1"
  local destination_name="$2"
  local output
  local -n destination="${destination_name}"

  output="$(
    cargo run --quiet --locked -p cargo-fe2o3 -- examples list "${lane}"
  )"
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
  local -a all_examples rustc_examples cargo_args
  local -A rustc_example_set=()
  local package
  load_example_packages all all_examples
  load_example_packages rustc-check rustc_examples
  for package in "${rustc_examples[@]}"; do
    rustc_example_set["${package}"]=1
  done

  cargo_args=(check --workspace --all-targets --locked)
  for package in "${all_examples[@]}"; do
    if [[ -z "${rustc_example_set[${package}]+selected}" ]]; then
      cargo_args+=(--exclude "${package}")
    fi
  done

  # `cargo check` does not link libamdhip64, so all host-facing examples are safe
  # to validate on a generic runner.
  run_step workspace-check cargo "${cargo_args[@]}"
}

run_cpu_tests() {
  local cargo_args=(test --locked)
  local -a rustc_examples rocm_examples
  local -A rocm_example_set=()
  local package
  for package in "${CPU_TEST_PACKAGES[@]}"; do
    cargo_args+=(-p "${package}")
  done
  load_example_packages rustc-check rustc_examples
  load_example_packages rocm-compile rocm_examples
  for package in "${rocm_examples[@]}"; do
    rocm_example_set["${package}"]=1
  done
  for package in "${rustc_examples[@]}"; do
    if [[ -z "${rocm_example_set[${package}]+selected}" ]]; then
      cargo_args+=(-p "${package}")
    fi
  done
  # Keep the generic test lane independent of whether the host happens to have
  # ROCm installed. The raw HIP crate supplies a fail-closed no-runtime ABI.
  run_step cpu-tests env FE2O3_HIP_SYS_DISABLE=1 cargo "${cargo_args[@]}"
  run_step dialect-mir-pliron-tests \
    cargo test --locked -p dialect-mir --features pliron --test pliron_shell
}

run_auxiliary_tests() {
  # fe2o3-core unit tests link HIP, but its compile-fail doctests do not.
  run_step core-doc-tests cargo test --locked --doc -p fe2o3-core
  run_step device-copy-renamed-dependency \
    cargo check --locked -p device-copy-renamed-dependency
  run_step device-copy-derive-real-trait \
    cargo check --locked -p fe2o3-core --test device_copy_derive_compile
  run_step device-copy-derive-ui \
    cargo test --locked -p fe2o3-core --test device_copy_derive_ui
  run_step s09-debug-checker bash scripts/tests/s09-debug.sh
  run_step s09-debug-ci-guard bash scripts/tests/s09-debug-ci.sh
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

run_standalone_lockfiles() {
  run_step standalone-lockfiles bash "${STANDALONE_LOCKFILE_CHECKER}"
}

run_runtime_pure_rust_policy() {
  run_step runtime-pure-rust-audit-tests \
    env PYTHONDONTWRITEBYTECODE=1 python3 "${RUNTIME_PURE_RUST_AUDIT_TESTS}"
  run_step runtime-identity-oracle-parser-tests \
    env PYTHONDONTWRITEBYTECODE=1 python3 "${RUNTIME_IDENTITY_ORACLE_TESTS}"
  run_step runtime-pure-rust-metadata \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" metadata --cargo \
      --root fe2o3-kfd \
      --root fe2o3-drm-uapi \
      --root fe2o3-kfd-uapi \
      --root fe2o3-amdhsa-loader \
      --root fe2o3-runtime-model
  run_step runtime-pure-rust-kfd-examples-build \
    env CARGO_TARGET_DIR="${RUNTIME_PURE_RUST_TARGET_DIR}" \
      cargo build --locked -p fe2o3-kfd \
        --example kfd-version \
        --example kfd-topology \
        --example kfd-device-identity \
        --example kfd-host-visible-memory-policy \
        --example kfd-queue-resources
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
  run_step runtime-pure-rust-kfd-memory-policy-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-host-visible-memory-policy"
  run_step runtime-pure-rust-kfd-queue-resources-elf \
    python3 "${RUNTIME_PURE_RUST_AUDITOR}" \
      --policy "${RUNTIME_PURE_RUST_POLICY}" elf \
      --input "${RUNTIME_PURE_RUST_TARGET_DIR}/debug/examples/kfd-queue-resources"
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
  # Do not combine this with integration targets: Cargo can emit a test rlib
  # and an unversioned backend dylib with different Rust symbol hashes.
  run_step rustc-codegen-lib-tests \
    cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" --lib
}

run_rustc_codegen_target() {
  local test_target="$1"
  # Cargo can emit a test rlib and an unversioned backend dylib with different
  # Rust symbol hashes during one --all-targets build. This target-isolated Cargo
  # invocation produces the exact backend dylib before running its linked test.
  run_step "rustc-codegen-test-${test_target}" \
    cargo test --locked -p "${RUSTC_CODEGEN_TEST_PACKAGE}" \
      --test "${test_target}"
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
  run_step workspace-tests \
    cargo test --locked --workspace --all-targets \
      --exclude "${RUSTC_CODEGEN_TEST_PACKAGE}"
  run_rustc_codegen_tests
}

run_backend_build() {
  run_step backend-build cargo build --locked -p rustc-codegen-fe2o3
}

run_verus() {
  run_step runtime-model-verus \
    "${REPO_ROOT}/crates/fe2o3-runtime-model/verus/verify-verus.sh"
  run_step verus-fixtures \
    "${REPO_ROOT}/examples/verus_vecadd/run-verus.sh" --require
  run_step scalar-gemm-verus \
    "${REPO_ROOT}/examples/scalar_gemm_v1/run-verus.sh" --require
}

run_authority_launcher_tests() {
  run_step authority-launcher-tests \
    bash scripts/tests/cargo-fe2o3-authority-launcher.sh
}

run_rustc_trampoline_tests() {
  run_step rustc-trampoline-tests \
    bash scripts/tests/fe2o3-rustc-trampoline.sh
}

run_parity_matrix_checks() {
  run_step parity-matrix-check bash scripts/parity-matrix.sh check
  run_step parity-matrix-tests bash scripts/tests/parity-matrix.sh
  run_step parity-evidence-tests bash scripts/tests/parity-evidence.sh
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
  run_step mi300x-evidence-queue-tests \
    bash scripts/tests/mi300x-evidence-queue.sh
  run_step hosted-parity-ci-tests \
    bash scripts/tests/hosted-parity-ci.sh
}

run_generic_core() {
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
  run_step ci-local-test-gate bash scripts/tests/ci-local-test-gate.sh
  run_cpu_tests
  run_rustc_codegen_lib_tests
  run_auxiliary_tests
}

run_generic() {
  run_generic_core
  run_all_rustc_codegen_shards
}

run_rocm_compile() {
  export FE2O3_TARGET="${FE2O3_TARGET:-gfx1100}"
  local -a example_packages
  load_example_packages rocm-compile example_packages
  run_step rocm-doctor cargo run --locked -p cargo-fe2o3 -- doctor
  run_step rocm-trusted-device-items \
    cargo test --locked -p rustc-codegen-fe2o3 \
      --test trusted_device_items \
      genuine_markers_emit_and_local_external_spoofs_fail_closed -- \
      --ignored --exact
  run_step rocm-trusted-device-item-stale-cleanup \
    cargo test --locked -p rustc-codegen-fe2o3 \
      --test trusted_device_items \
      rejected_lookalikes_remove_preseeded_artifacts_atomically -- \
      --ignored --exact
  run_step rocm-cross-crate-typed-binding \
    env FE2O3_TEST_TARGET="${FE2O3_TARGET}" \
      cargo test --locked -p rustc-codegen-fe2o3 \
      --test cross_crate_typed_binding \
      same_logical_name_in_two_rlibs_resolves_distinct_artifacts -- \
      --ignored --exact
  run_step rocm-g1-code-object \
    cargo test --locked -p dialect-amdgcn --test lowering \
      rocm_compiles_the_golden_to_an_amdgpu_code_object -- \
      --ignored --exact
  run_step rocm-kernel-ir-codegen-rejection \
    cargo test --locked -p rustc-codegen-fe2o3 \
      --test kernel_ir_codegen \
      selected_pipeline_rejects_invalid_or_unsupported_inputs_and_cleans_stale_artifacts -- \
      --ignored --exact
  run_step rocm-kernel-ir-vecadd \
    cargo test --locked -p rustc-codegen-fe2o3 \
      --test kernel_ir_codegen \
      opt_in_vecadd_publishes_exact_g1_without_gpu -- \
      --ignored --exact

  local package
  for package in "${example_packages[@]}"; do
    # cargo-fe2o3 rotates its device-artifact generation when codegen
    # semantics change. Rebuild each example so Cargo cannot reuse a host
    # package fingerprint that predates the new generation.
    run_step "rocm-clean-${package}" \
      cargo clean -p "${package}"
    run_step "rocm-build-${package}" \
      cargo run --locked -p cargo-fe2o3 -- build -p "${package}"
    run_step "rocm-artifacts-${package}" \
      cargo run --quiet --locked -p cargo-fe2o3 -- \
        examples check-artifacts "${package}"
  done
  run_step rocm-kernel-ir-verification \
    cargo test --locked -p cargo-fe2o3 --test kernel_ir_verification \
      verification_gate_accepts_rejects_and_remains_opt_in -- --ignored --exact
}

require_gpu_access() {
  local kfd_path="${1:-/dev/kfd}"
  local dxg_path="${2:-/dev/dxg}"
  local device_node

  if [[ -e "${kfd_path}" ]]; then
    device_node="${kfd_path}"
  elif [[ -e "${dxg_path}" ]]; then
    if [[ "${HSA_ENABLE_DXG_DETECTION:-}" != "1" ]]; then
      printf '%s\n' \
        'WSL GPU smoke requires HSA_ENABLE_DXG_DETECTION=1' >&2
      return 2
    fi
    device_node="${dxg_path}"
  else
    printf '%s\n' \
      'GPU smoke requires /dev/kfd (native Linux) or /dev/dxg (WSL)' >&2
    return 2
  fi

  if [[ ! -r "${device_node}" || ! -w "${device_node}" ]]; then
    printf 'GPU smoke requires read/write access to %s\n' \
      "${device_node}" >&2
    return 2
  fi
}

wavefront_for_target() {
  local processor="${1%%:*}"
  case "${processor}" in
    gfx9*) printf '%s\n' 64 ;;
    gfx*) printf '%s\n' 32 ;;
    *)
      printf 'cannot derive wavefront size for FE2O3_TARGET=%s\n' "$1" >&2
      return 2
      ;;
  esac
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
  require_gpu_access /dev/kfd /dev/dxg
  if ! command -v rocminfo >/dev/null 2>&1; then
    printf '%s\n' 'GPU smoke requires rocminfo on PATH' >&2
    return 2
  fi

  run_step hardware-rocminfo rocminfo
  run_step hardware-doctor cargo run --locked -p cargo-fe2o3 -- doctor
  local rocm_path="${ROCM_PATH:-/opt/rocm}"
  local native_test="${REPO_ROOT}/target/fe2o3-hip-device-properties-test"
  run_step hardware-hip-device-properties-build \
    "${CC:-cc}" -std=c11 -Wall -Wextra -Werror -D__HIP_PLATFORM_AMD__ \
      -I "${rocm_path}/include" -I "${REPO_ROOT}/crates/fe2o3-hip-sys/native" \
      "${REPO_ROOT}/crates/fe2o3-hip-sys/native/device_properties_test.c" \
      -L "${rocm_path}/lib" -Wl,-rpath,"${rocm_path}/lib" -lamdhip64 \
      -o "${native_test}"
  run_step hardware-hip-device-properties-test "${native_test}"
  run_step hardware-observed-device-target \
    cargo test --locked -p fe2o3-core --lib \
      device_target::tests::context_observes_a_real_hip_device -- \
      --ignored --exact
  run_step hardware-device-copy-transfer \
    cargo test --locked -p fe2o3-core --test device_copy_derive_hardware -- \
      --ignored --exact derived_struct_bytes_round_trip_through_device_memory
  run_step hardware-kernel-ir-fill \
    cargo test --locked -p rustc-codegen-fe2o3 \
      --test kernel_ir_codegen \
      opt_in_fill_publishes_g1_and_executes_on_the_gpu -- \
      --ignored --exact
  run_step hardware-kernel-ir-vecadd \
    cargo test --locked -p rustc-codegen-fe2o3 \
      --test kernel_ir_codegen \
      opt_in_vecadd_publishes_exact_g1_and_executes_on_the_gpu -- \
      --ignored --exact
  run_step hardware-smoke cargo run --locked -p cargo-fe2o3 -- smoke
  local test_wavefront
  test_wavefront="$(wavefront_for_target "${FE2O3_TARGET}")"
  run_step hardware-hsaco-inspection env \
    FE2O3_TEST_HSACO="${REPO_ROOT}/target/fe2o3/vecadd.hsaco" \
    FE2O3_TEST_TARGET="${FE2O3_TARGET}" \
    FE2O3_TEST_WAVEFRONT="${test_wavefront}" \
    cargo test --locked -p fe2o3-hsaco --test inspection \
      inspects_real_generated_vecadd_hsaco -- --ignored --exact
}

run_s09_debug_hardware() {
  if [[ -z "${FE2O3_S09_EVIDENCE_DIR:-}" ]]; then
    printf '%s\n' 'S09 debug hardware requires FE2O3_S09_EVIDENCE_DIR' >&2
    return 2
  fi
  run_step s09-debug-hardware \
    bash scripts/s09-debug-ci.sh "${FE2O3_S09_EVIDENCE_DIR}"
}

run_parity_production_immutable() {
  run_step parity-production-immutable \
    bash scripts/tests/parity-production-immutable-ingest.sh
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}"

  case "${1:-}" in
    generic) run_generic ;;
    generic-core) run_generic_core ;;
    workspace-policy) run_workspace_dependency_policy ;;
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
    workspace-test) run_workspace_tests ;;
    rustc-codegen-test) run_rustc_codegen_tests ;;
    backend) run_backend_build ;;
    authority-launcher) run_authority_launcher_tests ;;
    rustc-trampoline) run_rustc_trampoline_tests ;;
    parity-evidence) run_parity_matrix_checks ;;
    parity-production-immutable) run_parity_production_immutable ;;
    verus) run_verus ;;
    rocm-compile) run_rocm_compile ;;
    hardware-smoke) run_hardware_smoke ;;
    s09-debug-hardware) run_s09_debug_hardware ;;
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

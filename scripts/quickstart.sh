#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly DEFAULT_MANIFEST="${REPO_ROOT}/examples/vecadd/Cargo.toml"
readonly FILL_REQUEST="${REPO_ROOT}/scripts/quickstart/fill-canary-request.json"
readonly FILL_EXPECTATION="${REPO_ROOT}/scripts/quickstart/fill-canary-expectation.json"
readonly FIXTURE_KIR="${REPO_ROOT}/crates/fe2o3-kir-sim-cli/tutorial/fill-v1/kernel.kir"
readonly FIXTURE_REQUEST="${REPO_ROOT}/crates/fe2o3-kir-sim-cli/tutorial/fill-v1/request.json"
readonly EXPORT_TARGET_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/fe2o3-sim-export"
CARGO_COMMAND="${CARGO:-cargo}"
readonly CARGO_COMMAND
SIMULATE_TEMP_ROOT=""

cleanup_simulate_source() {
  if [[ -n "${SIMULATE_TEMP_ROOT}" ]]; then
    rm -rf -- "${SIMULATE_TEMP_ROOT}"
    SIMULATE_TEMP_ROOT=""
  fi
}

trap cleanup_simulate_source EXIT

usage() {
  cat <<'EOF'
Usage: scripts/quickstart.sh <command> [options]

Commands:
  doctor [OPTIONS]
      Inspect CPU onboarding, direct-KFD topology, compiler tools, and optional
      debugger/profiler tools without linking or loading HIP/HSA runtimes.

  no-gpu
      Export examples/fill from ordinary #[kernel] Rust source and simulate it
      on the CPU, checking exact output and untouched canaries. The bundle is
      extraction-only evidence, not GPU equivalence.

  simulate-source --crate NAME --request FILE [--target gfx942|gfx950]
      [--bundle-version 1|5|6] [--output BUNDLE] [--expectation FILE]
                  [--matrix-bf16-numerics exact-v1|finite-sequential-fma-v1]
      -- <Cargo package/feature/target selection>
      Export and simulate any admitted kernel crate through the same general
      source/MIR/KIR path. A temporary bundle is removed unless --output is set.
      With --expectation, publish result JSON only after exact output comparison.

  source-check [MANIFEST]
      Run binding-only check and host tests for an ordinary fe2o3 Cargo project.

  exact-kir-fixture
      Run the committed exact canonical KIR V7 known-answer fixture.

  gfx942-preflight
      Require a direct-KFD gfx942 Wave64 device, then report the currently
      unavailable Worker V3 application-execution boundary and exit nonzero.
EOF
}

require_prerequisites() {
  command -v "${CARGO_COMMAND}" >/dev/null 2>&1 || {
    printf 'quickstart: Cargo command is unavailable: %s\n' "${CARGO_COMMAND}" >&2
    return 2
  }
  command -v realpath >/dev/null 2>&1 || {
    printf '%s\n' 'quickstart: GNU realpath is required' >&2
    return 2
  }
}

cargo_workspace() {
  (
    cd -- "${REPO_ROOT}"
    FE2O3_HIP_SYS_DISABLE=1 FE2O3_HSA_RUNTIME_DISABLE=1 \
      "${CARGO_COMMAND}" "$@"
  )
}

run_doctor() {
  cargo_workspace run --locked --quiet -p cargo-fe2o3 --bin cargo-fe2o3 -- \
    doctor "$@"
}

run_source_check() {
  local manifest="${1:-${DEFAULT_MANIFEST}}"
  manifest="$(realpath --canonicalize-existing -- "${manifest}")"
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || {
    printf 'quickstart: manifest is not a regular file: %s\n' "${manifest}" >&2
    return 2
  }
  cargo_workspace run --locked --quiet -p cargo-fe2o3 --bin cargo-fe2o3 -- \
    check --manifest-path "${manifest}"
  cargo_workspace run --locked --quiet -p cargo-fe2o3 --bin cargo-fe2o3 -- \
    test --all-targets --manifest-path "${manifest}"
}

run_exact_kir_fixture() {
  cargo_workspace run --locked --quiet -p fe2o3-kir-sim-cli \
    --bin fe2o3-kir-sim -- \
    --kir-v7 "${FIXTURE_KIR}" --request "${FIXTURE_REQUEST}"
}

run_simulate_source() {
  local crate_name='' request='' output='' expectation='' target=gfx942 bundle_version=1
  local matrix_bf16_numerics=''
  local -a cargo_args=() simulator_args=()
  while (($# > 0)); do
    case "$1" in
      --crate | --request | --output | --target | --bundle-version | --expectation | --matrix-bf16-numerics)
        (($# >= 2)) || {
          printf 'quickstart: %s requires a value\n' "$1" >&2
          return 2
        }
        case "$1" in
          --crate) crate_name="$2" ;;
          --request) request="$2" ;;
          --output) output="$2" ;;
          --target) target="$2" ;;
          --bundle-version) bundle_version="$2" ;;
          --matrix-bf16-numerics)
            [[ -n "$2" && -z "${matrix_bf16_numerics}" ]] || {
              printf '%s\n' 'quickstart: --matrix-bf16-numerics requires one nonempty value' >&2
              return 2
            }
            matrix_bf16_numerics="$2"
            ;;
          --expectation)
            [[ -n "$2" && -z "${expectation}" ]] || {
              printf '%s\n' 'quickstart: --expectation requires one nonempty value' >&2
              return 2
            }
            expectation="$2"
            ;;
        esac
        shift 2
        ;;
      --)
        shift
        cargo_args=("$@")
        break
        ;;
      *)
        printf 'quickstart: unknown simulate-source option: %s\n' "$1" >&2
        return 2
        ;;
    esac
  done
  [[ -n "${crate_name}" && -n "${request}" && ${#cargo_args[@]} -gt 0 ]] || {
    printf '%s\n' 'quickstart: simulate-source requires --crate, --request, and Cargo selection after --' >&2
    return 2
  }
  [[ "${target}" == gfx942 || "${target}" == gfx950 ]] || {
    printf '%s\n' 'quickstart: --target must be exactly gfx942 or gfx950' >&2
    return 2
  }
  [[ "${bundle_version}" == 1 || "${bundle_version}" == 5 || "${bundle_version}" == 6 ]] || {
    printf '%s\n' 'quickstart: --bundle-version must be exactly 1, 5, or 6' >&2
    return 2
  }
  if [[ -n "${matrix_bf16_numerics}" ]]; then
    case "${matrix_bf16_numerics}" in
      exact-v1) ;;
      finite-sequential-fma-v1)
        [[ "${bundle_version}" == 6 ]] || {
          printf '%s\n' 'quickstart: nondefault --matrix-bf16-numerics requires --bundle-version 6' >&2
          return 2
        }
        ;;
      *)
        printf '%s\n' 'quickstart: unknown --matrix-bf16-numerics mode' >&2
        return 2
        ;;
    esac
    simulator_args=(--matrix-bf16-numerics "${matrix_bf16_numerics}")
  fi
  request="$(realpath --canonicalize-existing -- "${request}")"
  [[ -f "${request}" && ! -L "${request}" ]] || {
    printf 'quickstart: request is not a regular file: %s\n' "${request}" >&2
    return 2
  }
  if [[ -n "${expectation}" ]]; then
    command -v python3 >/dev/null 2>&1 || {
      printf '%s\n' 'quickstart: python3 is required with --expectation' >&2
      return 2
    }
    python3 -I -B "${SCRIPT_DIR}/check-simulation-expectation.py" \
      --expectation "${expectation}" --validate-only
  fi

  local temporary_root='' bundle='' remove_bundle=0
  if [[ -n "${output}" ]]; then
    local output_parent output_name
    output_parent="$(realpath --canonicalize-existing -- "$(dirname -- "${output}")")"
    output_name="$(basename -- "${output}")"
    [[ "${output_name}" != . && "${output_name}" != .. && -n "${output_name}" ]] || {
      printf '%s\n' 'quickstart: --output must name a file' >&2
      return 2
    }
    bundle="${output_parent}/${output_name}"
    [[ ! -e "${bundle}" && ! -L "${bundle}" ]] || {
      printf 'quickstart: --output already exists: %s\n' "${bundle}" >&2
      return 2
    }
  fi
  if [[ -z "${output}" || -n "${expectation}" ]]; then
    SIMULATE_TEMP_ROOT="$(
      mktemp -d "${TMPDIR:-/tmp}/fe2o3-quickstart.XXXXXXXX"
    )"
    temporary_root="${SIMULATE_TEMP_ROOT}"
    chmod 700 "${temporary_root}"
  fi
  if [[ -z "${output}" ]]; then
    bundle="${temporary_root}/kernel.fe2sim"
    remove_bundle=1
  fi

  cargo_workspace build --locked --quiet -p rustc-codegen-fe2o3 \
    --bin fe2o3-rustc-extract >&2
  local simulator_input=--bundle
  if [[ "${bundle_version}" == 5 ]]; then
    simulator_input=--bundle-v5
  elif [[ "${bundle_version}" == 6 ]]; then
    simulator_input=--bundle-v6
  fi
  cargo_workspace run --locked --quiet -p rustc-codegen-fe2o3 \
    --bin fe2o3-export-sim -- \
    --crate "${crate_name}" --output "${bundle}" --target "${target}" \
    --bundle-version "${bundle_version}" --target-dir "${EXPORT_TARGET_DIR}" -- \
    "${cargo_args[@]}" >&2
  if [[ -n "${expectation}" ]]; then
    local result="${temporary_root}/result.json"
    cargo_workspace run --locked --quiet -p fe2o3-kir-sim-cli \
      --bin fe2o3-kir-sim -- "${simulator_input}" "${bundle}" --request "${request}" "${simulator_args[@]}" \
      >"${result}"
    python3 -I -B "${SCRIPT_DIR}/check-simulation-expectation.py" \
      --expectation "${expectation}" --result "${result}"
    cat -- "${result}"
  else
    cargo_workspace run --locked --quiet -p fe2o3-kir-sim-cli \
      --bin fe2o3-kir-sim -- "${simulator_input}" "${bundle}" --request "${request}" "${simulator_args[@]}"
  fi

  if ((remove_bundle == 0)); then
    printf 'quickstart: retained extraction-only simulation bundle: %s\n' "${bundle}" >&2
  fi
  cleanup_simulate_source
}

run_no_gpu() {
  printf '%s\n' \
    'quickstart: exporting ordinary Rust source under extraction-only custody; this grants no compiler, artifact, GPU, or equivalence authority' >&2
  run_simulate_source \
    --crate fe2o3_fill \
    --request "${FILL_REQUEST}" \
    --expectation "${FILL_EXPECTATION}" \
    --target gfx942 \
    -- --package fe2o3-fill --lib
}

run_gfx942_preflight() {
  run_doctor --require-gfx942
  printf '%s\n' \
    'quickstart: GPU application execution is unavailable: the production Worker V3 application route is not wired' >&2
  return 3
}

main() {
  require_prerequisites
  local command="${1:-}"
  [[ -n "${command}" ]] || {
    usage >&2
    return 2
  }
  shift
  case "${command}" in
    doctor) run_doctor "$@" ;;
    no-gpu)
      (($# == 0)) || { usage >&2; return 2; }
      run_no_gpu
      ;;
    simulate-source) run_simulate_source "$@" ;;
    source-check)
      (($# <= 1)) || { usage >&2; return 2; }
      run_source_check "$@"
      ;;
    exact-kir-fixture)
      (($# == 0)) || { usage >&2; return 2; }
      run_exact_kir_fixture
      ;;
    gfx942-preflight)
      (($# == 0)) || { usage >&2; return 2; }
      run_gfx942_preflight
      ;;
    -h | --help | help)
      (($# == 0)) || { usage >&2; return 2; }
      usage
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"

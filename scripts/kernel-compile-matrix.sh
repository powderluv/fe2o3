#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
if (($# > 1)); then
  printf '%s\n' 'kernel compile matrix: expected at most one target argument' >&2
  exit 2
fi
readonly TARGET="${1:-gfx942}"
declare -a CASES=()

usage() {
  cat <<'EOF'
Usage: scripts/kernel-compile-matrix.sh [gfx942|gfx950]

Compile the manifest-enumerated ordinary-source matrix through the
fe2o3 extractor and ROCm finalizer without executing any kernel.

The gfx950 matrix defaults to the exact manifest-pinned ROCm 7.2.1 Clang, LLD,
and device-library closure. Set FE2O3_GFX950_OCML_MANIFEST explicitly to use
the checked-in ROCm 7.2.4 closure. Neither target covers proof-only examples,
the basic Cargo regression manifest, hardware behavior, or numerical results.
EOF
}

case "${TARGET}" in
  -h | --help | help)
    usage
    exit 0
    ;;
  gfx942 | gfx950)
    :
    ;;
  *)
    printf 'kernel compile matrix: unsupported target: %s\n' "${TARGET}" >&2
    usage >&2
    exit 2
    ;;
esac

# Capture status directly: process substitution would hide validator failure.
matrix_records=$(python3 "${REPO_ROOT}/scripts/validate-tutorial-kernel-manifest.py" \
  --repo-root "${REPO_ROOT}" --emit-matrix "${TARGET}")
mapfile -t CASES <<<"$matrix_records"
if ((${#CASES[@]} == 0)) || [[ -z ${CASES[0]} ]]; then
  printf '%s\n' 'kernel compile matrix: validated matrix is empty' >&2
  exit 1
fi
printf '%s\n' 'MATRIX CONTRACT expected-source-inputs-only qualified=false policy_verification=pending semantic_oracles=pending'

printf 'kernel compile matrix: target=%s mode=compile-only kernels=%d hardware_observed=false\n' \
  "${TARGET}" "${#CASES[@]}"
if [[ ${TARGET} == gfx950 ]]; then
  printf '%s\n' \
    'MATRIX PREREQUISITE target=gfx950 exact manifest-pinned ROCm 7.2.1 or 7.2.4 Clang/LLD/device-library closure required'
  # Check the shared prerequisite before building, independent of case order.
  (
    CLANG=${CLANG:-${ROCM_PATH:-/opt/rocm}/llvm/bin/clang}
    LD_LLD=${LD_LLD:-${ROCM_PATH:-/opt/rocm}/llvm/bin/ld.lld}
    SHA256SUM=${SHA256SUM:-sha256sum}
    FE2O3_GFX950_OCML_MANIFEST=${FE2O3_GFX950_OCML_MANIFEST:-$REPO_ROOT/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.1.manifest}
    source "$REPO_ROOT/examples/gfx950_low_precision/gfx950-ocml-closure.sh"
    validate_gfx950_ocml_closure
  )
fi

MATRIX_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fe2o3-kernel-compile-matrix.XXXXXXXX")"
readonly MATRIX_ROOT
cleanup() {
  rm -rf -- "${MATRIX_ROOT:?}"
}
trap cleanup EXIT

readonly BUILD_TARGET="${MATRIX_ROOT}/cargo-target"
readonly CASE_ROOT="${MATRIX_ROOT}/cases"
mkdir -p -- "${BUILD_TARGET}" "${CASE_ROOT}"

readonly TOOLCHAIN="${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}"
readonly EXTRACTOR="${BUILD_TARGET}/debug/fe2o3-rustc-extract"
CARGO_TARGET_DIR="${BUILD_TARGET}" rustup run "${TOOLCHAIN}" cargo build \
  --locked --manifest-path "${REPO_ROOT}/Cargo.toml" \
  -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract
if [[ ! -f "${EXTRACTOR}" || -L "${EXTRACTOR}" || ! -x "${EXTRACTOR}" ]]; then
  printf 'kernel compile matrix: extractor build did not produce a regular executable: %s\n' \
    "${EXTRACTOR}" >&2
  exit 1
fi

passed=0
for record in "${CASES[@]}"; do
  IFS='|' read -r name runner artifact runner_arg systems_variant fixture_id <<<"${record}"
  runner_args=()
  if [[ -n ${runner_arg} ]]; then
    runner_args=("${runner_arg}")
  fi
  runner_env=()
  if [[ -n ${systems_variant} ]]; then
    runner_env=("FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=${systems_variant}")
  fi
  case_dir="${CASE_ROOT}/${name}"
  output_dir="${case_dir}/artifacts"
  temporary_dir="${case_dir}/tmp"
  runner_path="${REPO_ROOT}/${runner}"
  mkdir -p -- "${output_dir}" "${temporary_dir}"
  if [[ ! -f "${runner_path}" || -L "${runner_path}" ]]; then
    printf 'kernel compile matrix: missing regular runner for %s: %s\n' \
      "${name}" "${runner_path}" >&2
    exit 1
  fi

  printf 'CASE %s target=%s status=RUNNING fixture=%s\n' "${name}" "${TARGET}" "${fixture_id}"
  env \
    -u FE2O3_EXAMPLE_CARGO_ARGS \
    -u FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT \
    -u FE2O3_GFX950_ADVANCED_MODE \
    -u FE2O3_ADVANCED_SUITE \
    -u FE2O3_ADVANCED_SCRIPT_DIR \
    CARGO_TARGET_DIR="${BUILD_TARGET}" \
    FE2O3_EXAMPLE_COMPILE_ONLY=1 \
    FE2O3_ROOT_TARGET_DIR="${BUILD_TARGET}" \
    FE2O3_RUSTC_EXTRACTOR="${EXTRACTOR}" \
    FE2O3_OUTPUT_DIR="${output_dir}" \
    FE2O3_GFX950_FP4_OUTPUT_DIR="${output_dir}" \
    FE2O3_GFX950_FP8_OUTPUT_DIR="${output_dir}" \
    FE2O3_GFX950_FP4_ATTENTION_OUTPUT_DIR="${output_dir}" \
    FE2O3_GFX950_FP8_ATTENTION_OUTPUT_DIR="${output_dir}" \
    FE2O3_GFX950_ADVANCED_OUTPUT_DIR="${output_dir}" \
    FE2O3_GFX950_PRUNE_AMDGPU_TARGET=1 \
    TMPDIR="${temporary_dir}" \
    "${runner_env[@]}" \
    bash "${runner_path}" "${runner_args[@]}"

  artifact_paths=()
  mapfile -d '' -t artifact_paths < <(
    find "${output_dir}" -type f -name "${artifact}" -print0
  )
  if ((${#artifact_paths[@]} != 1)); then
    printf 'kernel compile matrix: %s produced %d valid candidates for expected HSACO %s\n' \
      "${name}" "${#artifact_paths[@]}" "${artifact}" >&2
    exit 1
  fi
  artifact_path=${artifact_paths[0]}
  if [[ -L ${artifact_path} || ! -s ${artifact_path} ]]; then
    printf 'kernel compile matrix: %s produced an invalid expected HSACO: %s\n' \
      "${name}" "${artifact_path}" >&2
    exit 1
  fi
  passed=$((passed + 1))
  printf 'CASE %s target=%s status=PASS hardware_observed=false\n' \
    "${name}" "${TARGET}"
done

printf 'MATRIX PASS target=%s compiled=%d hardware_executed=0 artifacts=temporary\n' \
  "${TARGET}" "${passed}"
if [[ ${TARGET} == gfx942 ]]; then
  printf '%s\n' \
    'MATRIX LIMITATION gfx950 requires its separate exact ROCm 7.2.1 or 7.2.4 matrix; source-model-only, proof-only, and basic Cargo examples are not covered'
else
  printf '%s\n' \
    'MATRIX LIMITATION remaining gfx950 ablations, HIP comparators, hardware behavior, and numerical results are not covered'
fi

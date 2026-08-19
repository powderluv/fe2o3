#!/bin/sh
set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../../.." && pwd)
lifecycle_proof="$script_dir/runtime_lifecycle_v1.rs"
identity_proof="$script_dir/device_identity_generation_v1.rs"
projection_proof="$script_dir/device_projection_refinement_v1.rs"
memory_proof="$script_dir/memory_lifecycle_v1.rs"
load_plan_proof="$script_dir/load_plan_v1.rs"
materialization_proof="$script_dir/materialization_v1.rs"
negative_lifecycle="$script_dir/negative/runtime_lifecycle_v1_release_while_published.rs"
negative_vm="$script_dir/negative/device_identity_generation_v1_vm_substitution.rs"
negative_stale="$script_dir/negative/device_identity_generation_v1_stale_reuse.rs"
negative_render="$script_dir/negative/device_identity_generation_v1_render_substitution.rs"
negative_projection_schema="$script_dir/negative/device_projection_refinement_v1_schema_drop.rs"
negative_projection_history="$script_dir/negative/device_projection_refinement_v1_history_link.rs"
negative_projection_identity="$script_dir/negative/device_projection_refinement_v1_identity_mix.rs"
negative_projection_currentness="$script_dir/negative/device_projection_refinement_v1_currentness_drop.rs"
negative_memory_free="$script_dir/negative/memory_lifecycle_v1_free_while_partial.rs"
negative_memory_unmap="$script_dir/negative/memory_lifecycle_v1_unmap_prefix.rs"
negative_memory_failed_full="$script_dir/negative/memory_lifecycle_v1_failed_full_release.rs"
negative_load_page_overlap="$script_dir/negative/load_plan_v1_page_overlap.rs"
negative_load_descriptor_delta="$script_dir/negative/load_plan_v1_descriptor_delta.rs"
negative_materialization_source="$script_dir/negative/materialization_v1_source_substitution.rs"
negative_materialization_zero="$script_dir/negative/materialization_v1_zero_omission.rs"
pin_dir="$script_dir/pins"
closure_manifest="$pin_dir/VERUS_CLOSURE_MANIFEST"
closure_checker="$repo_root/examples/row_softmax_v1/verify-verus-closure.sh"
source_checker="$repo_root/examples/wave64_collectives_v1/check-proof-source.py"
verus_bin=${VERUS:-verus}

if [ "$#" -ne 0 ]; then
    printf 'usage: %s\n' "$0" >&2
    exit 2
fi

read_pin() {
    value=$(sed -n '1p' "$1")
    case "$value" in
        *[!0-9a-f]*|'') printf 'FAIL: invalid SHA-256 pin in %s\n' "$1" >&2; exit 1 ;;
    esac
    if [ "${#value}" -ne 64 ]; then
        printf 'FAIL: SHA-256 pin in %s must contain 64 hex digits\n' "$1" >&2
        exit 1
    fi
    printf '%s\n' "$value"
}

expected_lifecycle=$(read_pin "$pin_dir/MODEL_SHA256")
expected_identity=$(read_pin "$pin_dir/DEVICE_IDENTITY_MODEL_SHA256")
expected_projection=$(read_pin "$pin_dir/DEVICE_PROJECTION_REFINEMENT_SHA256")
expected_memory=$(read_pin "$pin_dir/MEMORY_LIFECYCLE_SHA256")
expected_load_plan=$(read_pin "$pin_dir/LOAD_PLAN_SHA256")
expected_materialization=$(read_pin "$pin_dir/MATERIALIZATION_SHA256")
expected_negative_lifecycle=$(read_pin "$pin_dir/NEGATIVE_SHA256")
expected_negative_vm=$(read_pin "$pin_dir/NEGATIVE_VM_SUBSTITUTION_SHA256")
expected_negative_stale=$(read_pin "$pin_dir/NEGATIVE_STALE_REUSE_SHA256")
expected_negative_render=$(read_pin "$pin_dir/NEGATIVE_RENDER_SUBSTITUTION_SHA256")
expected_negative_projection_schema=$(read_pin "$pin_dir/NEGATIVE_PROJECTION_SCHEMA_SHA256")
expected_negative_projection_history=$(read_pin "$pin_dir/NEGATIVE_PROJECTION_HISTORY_SHA256")
expected_negative_projection_identity=$(read_pin "$pin_dir/NEGATIVE_PROJECTION_IDENTITY_SHA256")
expected_negative_projection_currentness=$(read_pin "$pin_dir/NEGATIVE_PROJECTION_CURRENTNESS_SHA256")
expected_negative_memory_free=$(read_pin "$pin_dir/NEGATIVE_MEMORY_FREE_SHA256")
expected_negative_memory_unmap=$(read_pin "$pin_dir/NEGATIVE_MEMORY_UNMAP_SHA256")
expected_negative_memory_failed_full=$(read_pin "$pin_dir/NEGATIVE_MEMORY_FAILED_FULL_SHA256")
expected_negative_load_page_overlap=$(read_pin "$pin_dir/NEGATIVE_LOAD_PAGE_OVERLAP_SHA256")
expected_negative_load_descriptor_delta=$(read_pin "$pin_dir/NEGATIVE_LOAD_DESCRIPTOR_DELTA_SHA256")
expected_negative_materialization_source=$(read_pin "$pin_dir/NEGATIVE_MATERIALIZATION_SOURCE_SHA256")
expected_negative_materialization_zero=$(read_pin "$pin_dir/NEGATIVE_MATERIALIZATION_ZERO_SHA256")
expected_verus=$(read_pin "$pin_dir/VERUS_SHA256")
expected_closure=$(read_pin "$pin_dir/VERUS_CLOSURE_MANIFEST_SHA256")
expected_source_checker=$(read_pin "$pin_dir/PROOF_SOURCE_CHECKER_SHA256")
expected_transcript=$(read_pin "$pin_dir/TRANSCRIPT_SHA256")
expected_version=$(sed -n '1p' "$pin_dir/VERUS_VERSION")
case "$expected_version" in
    ''|*[!0-9A-Za-z.-]*) printf 'FAIL: invalid pinned Verus version\n' >&2; exit 1 ;;
esac

sha256_path=$(command -v sha256sum 2>/dev/null || true)
timeout_path=$(command -v timeout 2>/dev/null || true)
readlink_path=$(command -v readlink 2>/dev/null || true)
if [ -z "$sha256_path" ] || [ -z "$timeout_path" ] || [ -z "$readlink_path" ]; then
    printf 'FAIL: sha256sum, timeout, and readlink are required\n' >&2
    exit 1
fi

check_digest() {
    actual=$("$sha256_path" "$2" | awk '{ print $1 }')
    if [ "$actual" != "$1" ]; then
        printf 'FAIL: SHA-256 substitution for %s\n' "$2" >&2
        exit 1
    fi
}

check_sources() {
    check_digest "$expected_lifecycle" "$lifecycle_proof"
    check_digest "$expected_identity" "$identity_proof"
    check_digest "$expected_projection" "$projection_proof"
    check_digest "$expected_memory" "$memory_proof"
    check_digest "$expected_load_plan" "$load_plan_proof"
    check_digest "$expected_materialization" "$materialization_proof"
    check_digest "$expected_negative_lifecycle" "$negative_lifecycle"
    check_digest "$expected_negative_vm" "$negative_vm"
    check_digest "$expected_negative_stale" "$negative_stale"
    check_digest "$expected_negative_render" "$negative_render"
    check_digest "$expected_negative_projection_schema" "$negative_projection_schema"
    check_digest "$expected_negative_projection_history" "$negative_projection_history"
    check_digest "$expected_negative_projection_identity" "$negative_projection_identity"
    check_digest "$expected_negative_projection_currentness" "$negative_projection_currentness"
    check_digest "$expected_negative_memory_free" "$negative_memory_free"
    check_digest "$expected_negative_memory_unmap" "$negative_memory_unmap"
    check_digest "$expected_negative_memory_failed_full" "$negative_memory_failed_full"
    check_digest "$expected_negative_load_page_overlap" "$negative_load_page_overlap"
    check_digest "$expected_negative_load_descriptor_delta" "$negative_load_descriptor_delta"
    check_digest "$expected_negative_materialization_source" "$negative_materialization_source"
    check_digest "$expected_negative_materialization_zero" "$negative_materialization_zero"
    check_digest "$expected_closure" "$closure_manifest"
    check_digest 'c0f5f201dca9ea6b3fa953884cdfaca8ca38413ad2a9de7700b3aaeb3a610d0c' "$closure_checker"
    check_digest "$expected_source_checker" "$source_checker"
}

check_sources
"$source_checker" \
    "$lifecycle_proof" \
    "$identity_proof" \
    "$projection_proof" \
    "$memory_proof" \
    "$load_plan_proof" \
    "$materialization_proof" \
    "$negative_lifecycle" \
    "$negative_vm" \
    "$negative_stale" \
    "$negative_render" \
    "$negative_projection_schema" \
    "$negative_projection_history" \
    "$negative_projection_identity" \
    "$negative_projection_currentness" \
    "$negative_memory_free" \
    "$negative_memory_unmap" \
    "$negative_memory_failed_full" \
    "$negative_load_page_overlap" \
    "$negative_load_descriptor_delta" \
    "$negative_materialization_source" \
    "$negative_materialization_zero"

case "$verus_bin" in
    */*) [ -x "$verus_bin" ] && verus_path=$verus_bin || verus_path= ;;
    *) verus_path=$(command -v "$verus_bin" 2>/dev/null || true) ;;
esac
if [ -z "$verus_path" ]; then
    printf 'FAIL: Verus is unavailable; set VERUS=/absolute/path/to/verus\n' >&2
    exit 1
fi
verus_path=$("$readlink_path" -f "$verus_path")
if [ "$(basename "$verus_path")" != verus ]; then
    printf 'FAIL: pinned Verus executable must be named verus\n' >&2
    exit 1
fi
check_digest "$expected_verus" "$verus_path"
verus_root=$(CDPATH='' cd -- "$(dirname -- "$verus_path")" && pwd)
"$closure_checker" "$verus_root" "$closure_manifest"

runner_home=${HOME:-/nonexistent}
runner_path=${PATH:-/usr/local/bin:/usr/bin:/bin}
runner_rustup_home=${RUSTUP_HOME:-"$runner_home/.rustup"}
runner_cargo_home=${CARGO_HOME:-"$runner_home/.cargo"}
actual_version=$(
    env -i \
        "HOME=$runner_home" \
        "PATH=$runner_path" \
        "RUSTUP_HOME=$runner_rustup_home" \
        "CARGO_HOME=$runner_cargo_home" \
        "VERUS_Z3_PATH=$verus_root/z3" \
        "$verus_path" --version \
        | awk '/^[[:space:]]*Version:/ { print $2; exit }'
)
if [ "$actual_version" != "$expected_version" ]; then
    printf 'FAIL: Verus version does not match the pin\n' >&2
    exit 1
fi

timeout_seconds=${VERUS_TIMEOUT_SECONDS:-120}
case "$timeout_seconds" in
    ''|*[!0-9]*) printf 'FAIL: VERUS_TIMEOUT_SECONDS must be 1 through 300\n' >&2; exit 2 ;;
esac
if [ "$timeout_seconds" -lt 1 ] || [ "$timeout_seconds" -gt 300 ]; then
    printf 'FAIL: VERUS_TIMEOUT_SECONDS must be 1 through 300\n' >&2
    exit 2
fi

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/fe2o3-runtime-model-verus.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

run_verus() {
    "$timeout_path" --foreground --signal=TERM --kill-after=5 "$timeout_seconds" \
        env -i \
        "HOME=$runner_home" \
        "PATH=$runner_path" \
        "RUSTUP_HOME=$runner_rustup_home" \
        "CARGO_HOME=$runner_cargo_home" \
        "VERUS_Z3_PATH=$verus_root/z3" \
        "$verus_path" --crate-type lib --triggers-mode silent "$1"
}

check_positive() {
    source=$1
    expected_summary=$2
    label=$3
    log="$tmp_dir/$label-positive.log"
    if ! run_verus "$source" >"$log" 2>&1; then
        printf 'FAIL: positive proof did not verify: %s\n' "$label" >&2
        cat "$log" >&2
        exit 1
    fi
    if ! grep -Fq "$expected_summary" "$log"; then
        printf 'FAIL: unexpected positive verification summary: %s\n' "$label" >&2
        cat "$log" >&2
        exit 1
    fi
    cat "$log"
}

check_negative() {
    source=$1
    marker=$2
    label=$3
    log="$tmp_dir/$label-negative.log"
    if run_verus "$source" >"$log" 2>&1; then
        printf 'FAIL: expected-negative proof unexpectedly verified: %s\n' "$label" >&2
        cat "$log" >&2
        exit 1
    fi
    if ! grep -Fq "$marker" "$log" \
        || ! grep -Fq 'error: postcondition not satisfied' "$log" \
        || ! grep -Fq 'verification results:: 0 verified, 1 errors' "$log"; then
        printf 'FAIL: mutation failed at an unexpected verification surface: %s\n' "$label" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'expected-negative rejected: %s\n' "$label"
}

check_positive "$lifecycle_proof" 'verification results:: 2 verified, 0 errors' lifecycle
check_positive "$identity_proof" 'verification results:: 4 verified, 0 errors' identity-generation
check_positive "$projection_proof" 'verification results:: 4 verified, 0 errors' device-projection-refinement
check_positive "$memory_proof" 'verification results:: 6 verified, 0 errors' memory-lifecycle
check_positive "$load_plan_proof" 'verification results:: 3 verified, 0 errors' load-plan
check_positive "$materialization_proof" 'verification results:: 8 verified, 0 errors' materialization
check_negative "$negative_lifecycle" mutated_release_while_published_is_safe_v1 release-while-published
check_negative "$negative_vm" mutated_vm_generation_substitution_is_exact_v1 vm-generation-substitution
check_negative "$negative_stale" mutated_stale_generation_reuse_advances_v1 stale-generation-reuse
check_negative "$negative_render" mutated_render_substitution_correlates_v1 render-substitution
check_negative "$negative_projection_schema" mutated_projection_drops_drm_schema_v1 projection-schema-drop
check_negative "$negative_projection_history" mutated_history_forgets_predecessor_v1 projection-history-link
check_negative "$negative_projection_identity" mutated_cross_source_identity_mix_is_equal_v1 projection-identity-mix
check_negative "$negative_projection_currentness" mutated_projection_drops_reset_fence_v1 projection-currentness-drop
check_negative "$negative_memory_free" mutated_free_while_partial_is_safe_v1 memory-free-while-partial
check_negative "$negative_memory_unmap" mutated_unmap_uses_absolute_cumulative_progress_v1 memory-unmap-cumulative
check_negative "$negative_memory_failed_full" mutated_failed_full_unmap_is_unreleasable_v1 memory-unmap-failed-full
check_negative "$negative_load_page_overlap" mutated_memory_only_check_rejects_page_overlap_v1 load-page-overlap
check_negative "$negative_load_descriptor_delta" mutated_descriptor_delta_substitution_is_bound_v1 load-descriptor-delta
check_negative "$negative_materialization_source" mutated_source_substitution_preserves_exact_byte_v1 materialization-source-substitution
check_negative "$negative_materialization_zero" mutated_zero_first_initializes_every_byte_v1 materialization-zero-omission

# Detect source, checker, closure, or executable replacement during the run.
check_sources
check_digest "$expected_verus" "$verus_path"
"$closure_checker" "$verus_root" "$closure_manifest"

transcript='FE2O3_RUNTIME_MODEL_VERUS_OK lifecycle_obligations=2 identity_obligations=4 projection_obligations=4 memory_obligations=6 load_plan_obligations=3 materialization_obligations=8 mutations=15'
actual_transcript=$(printf '%s\n' "$transcript" | "$sha256_path" | awk '{ print $1 }')
if [ "$actual_transcript" != "$expected_transcript" ]; then
    printf 'FAIL: verification transcript does not match the pin\n' >&2
    exit 1
fi
printf '%s\n' "$transcript"

#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
require_verus=0
source_only=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --require) require_verus=1 ;;
        --source-only) source_only=1 ;;
        *)
            printf 'usage: %s [--require] [--source-only]\n' "$0" >&2
            exit 2
            ;;
    esac
    shift
done

source_failures=0
require_source() {
    file=$1
    needle=$2
    if ! grep -Fq "$needle" "$file"; then
        printf 'FAIL:  %s is missing source-shape marker %s\n' "$file" "$needle" >&2
        source_failures=$((source_failures + 1))
    fi
}

forbid_source() {
    file=$1
    needle=$2
    if grep -Fq "$needle" "$file"; then
        printf 'FAIL:  %s contains forbidden proof shortcut %s\n' "$file" "$needle" >&2
        source_failures=$((source_failures + 1))
    fi
}

shared_body="$script_dir/src/elementwise_bodies.rs"
positive="$script_dir/verus/elementwise.rs"
require_source "$positive" 'include!("../src/elementwise_bodies.rs")'
require_source "$shared_body" 'macro_rules! copy_kernel_body'
require_source "$shared_body" 'macro_rules! affine_map_kernel_body'
require_source "$shared_body" 'macro_rules! gather_kernel_body'

for fixture in \
    "$script_dir/verus/negative/copy_wrong_source.rs" \
    "$script_dir/verus/negative/affine_wrong_bias.rs" \
    "$script_dir/verus/negative/gather_wrong_index.rs"
do
    require_source "$fixture" 'include!("../../src/elementwise_bodies.rs")'
done
require_source "$positive" 'pub fn verified_copy_thread'
require_source "$positive" 'pub fn verified_affine_map_thread'
require_source "$positive" 'pub fn verified_gather_thread'
require_source "$script_dir/verus/negative/copy_wrong_source.rs" \
    'pub fn mutated_copy_source'
require_source "$script_dir/verus/negative/affine_wrong_bias.rs" \
    'pub fn mutated_affine'
require_source "$script_dir/verus/negative/gather_wrong_index.rs" \
    'pub fn mutated_gather_source'

two_kernel_body="$script_dir/src/two_kernel_bodies.rs"
two_kernel="$script_dir/verus/two_kernel.rs"
require_source "$two_kernel" 'include!("../src/two_kernel_bodies.rs")'
require_source "$two_kernel" '#[path = "permission_core.rs"]'
forbid_source "$script_dir/verus/permission_core.rs" '#[verifier::external_body]'
forbid_source "$script_dir/verus/permission_core.rs" 'admit('
forbid_source "$script_dir/verus/permission_core.rs" 'assume('
require_source "$two_kernel_body" 'macro_rules! alpha_kernel_body'
require_source "$two_kernel_body" 'macro_rules! zeta_kernel_body'
require_source "$two_kernel_body" 'if let Some(out) = $output.get_mut($thread)'
require_source "$two_kernel" 'pub fn verified_alpha_thread'
require_source "$two_kernel" 'pub fn verified_zeta_thread'
require_source "$two_kernel" 'pub open spec fn alpha_input_initialization_assumptions'
require_source "$two_kernel" 'pub open spec fn zeta_input_initialization_assumptions'
require_source "$two_kernel" 'pub proof fn f32_access_address_is_representable'
require_source "$two_kernel" 'pub proof fn initialized_read_is_bounded'
require_source "$two_kernel" 'pub proof fn exclusive_output_is_bounded_and_initialized_by_write'
require_source "$two_kernel" 'pub proof fn two_kernel_identity_ownership_is_race_free'
forbid_source "$two_kernel" 'admit('
forbid_source "$two_kernel" 'assume(false'
forbid_source "$two_kernel" '#[verifier::external_body]'

require_source "$script_dir/verus/negative/two_kernel_wrong_scalar.rs" \
    'mutated_alpha_uses_wrong_scalar_result'
require_source "$script_dir/verus/negative/two_kernel_wrong_scalar.rs" \
    'include!("../../src/two_kernel_bodies.rs")'
require_source "$script_dir/verus/negative/two_kernel_guard_bypass.rs" \
    'mutated_alpha_bypasses_output_guard'
require_source "$script_dir/verus/negative/two_kernel_overlapping_output.rs" \
    'mutated_overlapping_output_ownership_is_race_free'
require_source "$script_dir/verus/negative/two_kernel_uninitialized_input.rs" \
    'mutated_uninitialized_input_is_readable'
require_source "$script_dir/verus/negative/two_kernel_address_overflow.rs" \
    'mutated_f32_address_overflow_is_representable'

wave_lds="$script_dir/verus/wave_lds.rs"
require_source "$wave_lds" 'include!("vecadd.rs")'
require_source "$wave_lds" 'pub proof fn active_values_determine_reduction'
require_source "$wave_lds" 'pub proof fn distinct_active_lanes_have_disjoint_scan_outputs'
require_source "$wave_lds" 'pub proof fn owned_lds_write_is_in_bounds_and_framed'
require_source "$wave_lds" 'pub proof fn distinct_threads_have_disjoint_lds_writes'
require_source "$wave_lds" 'pub proof fn convergent_barrier_enables_shared_lds_read'
forbid_source "$wave_lds" 'admit('
forbid_source "$wave_lds" 'assume(false'
forbid_source "$wave_lds" '#[verifier::external_body]'

require_source "$script_dir/verus/negative/wave_inactive_lane_contributes.rs" \
    'mutated_inactive_lane_contributes'
require_source "$script_dir/verus/negative/lds_duplicate_writer.rs" \
    'mutated_duplicate_lds_writers_are_race_free'
require_source "$script_dir/verus/negative/lds_read_before_barrier.rs" \
    'mutated_read_before_barrier_is_legal'
require_source "$script_dir/verus/negative/lds_out_of_bounds_read.rs" \
    'mutated_unbounded_lds_read_is_in_bounds'

gfx942_wave_lds="$script_dir/verus/gfx942_wave_lds_v1.rs"
require_source "$gfx942_wave_lds" 'pub open spec fn gfx942_wave64_contract'
require_source "$gfx942_wave_lds" 'pub proof fn gfx942_inactive_values_do_not_change_wave_reduction'
require_source "$gfx942_wave_lds" 'pub open spec fn gfx942_static_lds_u32x256_is_exact'
require_source "$gfx942_wave_lds" 'pub proof fn gfx942_distinct_threads_own_disjoint_lds_slots'
require_source "$gfx942_wave_lds" 'pub proof fn gfx942_barrier_enables_legal_stage_partner_read'
require_source "$gfx942_wave_lds" 'pub open spec fn gfx942_barrier_trace_is_uniform'
require_source "$gfx942_wave_lds" 'Deliberate refinement boundary'
forbid_source "$gfx942_wave_lds" 'admit('
forbid_source "$gfx942_wave_lds" 'assume(false'
forbid_source "$gfx942_wave_lds" '#[verifier::external_body]'
require_source "$script_dir/verus/negative/gfx942_wave_wrong_extent.rs" \
    'mutated_gfx942_wave63_is_wave64'
require_source "$script_dir/verus/negative/gfx942_lds_missing_barrier_lane.rs" \
    'mutated_missing_barrier_lane_is_complete'

if [ "$source_failures" -ne 0 ]; then
    printf 'Source-shape checks failed: %s missing marker(s)\n' "$source_failures" >&2
    exit 1
fi
printf 'PASS:  shared-body, two-kernel, active-wave, and LDS proof source shapes are paired\n'

if [ "$source_only" -eq 1 ]; then
    exit 0
fi

verus_bin=${VERUS:-verus}
case "$verus_bin" in
    */*)
        if [ ! -x "$verus_bin" ]; then
            verus_path=
        else
            verus_path=$verus_bin
        fi
        ;;
    *) verus_path=$(command -v "$verus_bin" 2>/dev/null || true) ;;
esac

if [ -z "$verus_path" ]; then
    printf 'SKIP: Verus is unavailable (set VERUS=/path/to/verus)\n'
    if [ "$require_verus" -eq 1 ]; then
        exit 1
    fi
    exit 0
fi

timeout_path=$(command -v timeout 2>/dev/null || true)
if [ -z "$timeout_path" ]; then
    printf 'FAIL:  timeout is required to bound each Verus invocation\n' >&2
    exit 1
fi

verus_timeout_seconds=${VERUS_TIMEOUT_SECONDS:-60}
case "$verus_timeout_seconds" in
    ''|*[!0-9]*)
        printf 'FAIL:  VERUS_TIMEOUT_SECONDS must be an integer from 1 through 300\n' >&2
        exit 2
        ;;
esac
if [ "$verus_timeout_seconds" -lt 1 ] || [ "$verus_timeout_seconds" -gt 300 ]; then
    printf 'FAIL:  VERUS_TIMEOUT_SECONDS must be an integer from 1 through 300\n' >&2
    exit 2
fi

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/fe2o3-verus-fill.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

failures=0
run_verus() {
    file=$1
    "$timeout_path" --foreground --signal=TERM --kill-after=5 \
        "$verus_timeout_seconds" \
        "$verus_path" --crate-type lib --triggers-mode silent "$file"
}

record_timeout() {
    name=$1
    status=$2
    if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
        printf 'FAIL:  %s exceeded the %ss Verus time limit\n' \
            "$name" "$verus_timeout_seconds" >&2
        failures=$((failures + 1))
        return 0
    fi
    return 1
}

run_pass() {
    name=$1
    file=$2
    log="$tmp_dir/$name.log"
    if run_verus "$file" >"$log" 2>&1; then
        printf 'PASS:  %s verified\n' "$name"
    else
        status=$?
        if ! record_timeout "$name" "$status"; then
            printf 'FAIL:  %s was expected to verify\n' "$name" >&2
            cat "$log" >&2
            failures=$((failures + 1))
        fi
    fi
}

run_rejected() {
    name=$1
    file=$2
    marker=$3
    diagnostic=$4
    log="$tmp_dir/$name.log"
    if run_verus "$file" >"$log" 2>&1; then
        printf 'FAIL:  %s unexpectedly verified\n' "$name" >&2
        failures=$((failures + 1))
        return
    else
        status=$?
    fi
    if record_timeout "$name" "$status"; then
        return
    elif ! grep -Fq "$marker" "$log"; then
        printf 'FAIL:  %s failed without marker %s\n' "$name" "$marker" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    elif grep -Eiq "$diagnostic" "$log"; then
        printf 'XFAIL: %s rejected for the expected proof obligation\n' "$name"
    else
        printf 'FAIL:  %s failed without the expected proof diagnostic\n' "$name" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    fi
}

run_rejected_exact() {
    name=$1
    file=$2
    marker=$3
    diagnostic=$4
    failed_clause=$5
    log="$tmp_dir/$name.log"
    if run_verus "$file" >"$log" 2>&1; then
        printf 'FAIL:  %s unexpectedly verified\n' "$name" >&2
        failures=$((failures + 1))
        return
    else
        status=$?
    fi
    if record_timeout "$name" "$status"; then
        return
    fi

    primary_error_count=$(awk '
        /^error:/ && $0 !~ /^error: aborting due to / { count++ }
        END { print count + 0 }
    ' "$log")
    verification_result_count=$(awk '
        /^verification results::/ { count++ }
        END { print count + 0 }
    ' "$log")
    single_error_result_count=$(awk '
        /^verification results:: [0-9]+ verified, 1 errors$/ { count++ }
        END { print count + 0 }
    ' "$log")

    if [ "$primary_error_count" -ne 1 ]; then
        printf 'FAIL:  %s emitted %s primary Verus errors, expected exactly one\n' \
            "$name" "$primary_error_count" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    elif [ "$verification_result_count" -ne 1 ] || [ "$single_error_result_count" -ne 1 ]; then
        printf 'FAIL:  %s did not emit one verification result reporting one error\n' \
            "$name" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    elif ! grep -Fq "$marker" "$log"; then
        printf 'FAIL:  %s failed without marker %s\n' "$name" "$marker" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    elif ! grep -Fq "$diagnostic" "$log"; then
        printf 'FAIL:  %s failed without exact diagnostic %s\n' "$name" "$diagnostic" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    elif ! grep -Fq "$failed_clause" "$log"; then
        printf 'FAIL:  %s failed without exact clause %s\n' "$name" "$failed_clause" >&2
        cat "$log" >&2
        failures=$((failures + 1))
    else
        printf 'XFAIL: %s rejected at the expected proof clause\n' "$name"
    fi
}

run_pass vecadd "$script_dir/verus/vecadd.rs"
run_pass fill "$script_dir/verus/fill.rs"
run_pass elementwise "$script_dir/verus/elementwise.rs"
run_pass wave_lds "$script_dir/verus/wave_lds.rs"
run_pass gfx942_wave_lds_v1 "$script_dir/verus/gfx942_wave_lds_v1.rs"
run_pass two_kernel "$script_dir/verus/two_kernel.rs"
run_pass reference_refinement_v1 \
    "$script_dir/verus/reference_refinement_v1.rs"
run_rejected reference_refinement_wrong_value \
    "$script_dir/verus/negative/reference_refinement_wrong_value.rs" \
    'mutated_reference_value_is_accepted' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected reference_refinement_duplicate_owner \
    "$script_dir/verus/negative/reference_refinement_duplicate_owner.rs" \
    'mutated_duplicate_owner_is_injective' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected fill_missing_bounds \
    "$script_dir/verus/negative/fill_missing_bounds.rs" \
    'mutated_fill_index_is_in_bounds' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected fill_non_injective \
    "$script_dir/verus/negative/fill_non_injective.rs" \
    'mutated_distinct_threads_have_disjoint_outputs' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected fill_incorrect_postcondition \
    "$script_dir/verus/negative/fill_incorrect_postcondition.rs" \
    'mutated_one_write_fills_every_element' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected permission_overlapping_output_writes \
    "$script_dir/verus/negative/permission_overlapping_output_writes.rs" \
    'mutated_overlapping_output_writes_are_disjoint' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected permission_write_read_alias \
    "$script_dir/verus/negative/permission_write_read_alias.rs" \
    'mutated_write_read_alias_is_compatible' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected permission_out_of_bounds_region \
    "$script_dir/verus/negative/permission_out_of_bounds_region.rs" \
    'mutated_unbounded_region_is_in_bounds' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected same_source_wrong_bounds \
    "$script_dir/verus/negative/same_source_wrong_bounds.rs" \
    'thread.linear < domain.length' \
    'precondition.*not satisfied|precondition failure'
run_rejected same_source_output_alias \
    "$script_dir/verus/negative/same_source_output_alias.rs" \
    'rejects_output_input_alias' \
    'precondition.*not satisfied|precondition failure'
run_rejected same_source_functional_error \
    "$script_dir/verus/negative/same_source_functional_error.rs" \
    'mutated_same_source_claims_wrong_sum' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected_exact real_kernel_guard_bypass \
    "$script_dir/verus/negative/real_kernel_guard_bypass.rs" \
    'real_kernel_guard_bypass_input_index' \
    'error: precondition not met: index in bounds for this access' \
    'let _bypassed_input = a[i]'
run_rejected_exact real_kernel_wrong_index \
    "$script_dir/verus/negative/real_kernel_wrong_index.rs" \
    'real_kernel_wrong_index.rs' \
    'error: postcondition not satisfied' \
    'result.values@[0] == output.values@[0]'
run_rejected_exact real_kernel_output_alias \
    "$script_dir/verus/negative/real_kernel_output_alias.rs" \
    'rejects_real_output_input_alias' \
    'error: precondition not satisfied' \
    'real_vecadd_source_evidence_is_valid('
run_rejected_exact copy_wrong_source \
    "$script_dir/verus/negative/copy_wrong_source.rs" \
    'mutated_copy_claims_identity_source' \
    'error: postcondition not satisfied' \
    'final(output)@ == old(output)@.update'
run_rejected_exact affine_wrong_bias \
    "$script_dir/verus/negative/affine_wrong_bias.rs" \
    'affine_wrong_bias.rs' \
    'error: postcondition not satisfied' \
    'final(output)@ == old(output)@.update'
run_rejected_exact gather_wrong_index \
    "$script_dir/verus/negative/gather_wrong_index.rs" \
    'gather_wrong_index.rs' \
    'error: postcondition not satisfied' \
    'final(output)@ == old(output)@.update'
run_rejected two_kernel_wrong_scalar \
    "$script_dir/verus/negative/two_kernel_wrong_scalar.rs" \
    'mutated_alpha_uses_wrong_scalar_result' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected two_kernel_guard_bypass \
    "$script_dir/verus/negative/two_kernel_guard_bypass.rs" \
    'mutated_alpha_bypasses_output_guard' \
    'precondition not met|precondition.*not satisfied|precondition failure'
run_rejected two_kernel_overlapping_output \
    "$script_dir/verus/negative/two_kernel_overlapping_output.rs" \
    'mutated_overlapping_output_ownership_is_race_free' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected two_kernel_uninitialized_input \
    "$script_dir/verus/negative/two_kernel_uninitialized_input.rs" \
    'mutated_uninitialized_input_is_readable' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected two_kernel_address_overflow \
    "$script_dir/verus/negative/two_kernel_address_overflow.rs" \
    'mutated_f32_address_overflow_is_representable' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected wave_inactive_lane_contributes \
    "$script_dir/verus/negative/wave_inactive_lane_contributes.rs" \
    'mutated_inactive_lane_contributes' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected lds_duplicate_writer \
    "$script_dir/verus/negative/lds_duplicate_writer.rs" \
    'mutated_duplicate_lds_writers_are_race_free' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected lds_read_before_barrier \
    "$script_dir/verus/negative/lds_read_before_barrier.rs" \
    'mutated_read_before_barrier_is_legal' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected lds_out_of_bounds_read \
    "$script_dir/verus/negative/lds_out_of_bounds_read.rs" \
    'mutated_unbounded_lds_read_is_in_bounds' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected gfx942_wave_wrong_extent \
    "$script_dir/verus/negative/gfx942_wave_wrong_extent.rs" \
    'mutated_gfx942_wave63_is_wave64' \
    'postcondition.*not satisfied|postcondition failure'
run_rejected gfx942_lds_missing_barrier_lane \
    "$script_dir/verus/negative/gfx942_lds_missing_barrier_lane.rs" \
    'mutated_missing_barrier_lane_is_complete' \
    'postcondition.*not satisfied|postcondition failure'

if [ "$failures" -ne 0 ]; then
    printf 'Verus fixture run failed: %s unexpected result(s)\n' "$failures" >&2
    exit 1
fi
printf 'Verus fixture run passed: 7 proof harnesses, 28 expected rejections\n'

#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT
FIXTURE="$(mktemp -d "${TMPDIR:-/tmp}/fe2o3-quickstart-test.XXXXXXXX")"
readonly FIXTURE
trap 'rm -rf -- "${FIXTURE}"' EXIT
mkdir "${FIXTURE}/bin" "${FIXTURE}/tmp"
LOG="${FIXTURE}/cargo.log"
readonly LOG

cat >"${FIXTURE}/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
[[ "${FE2O3_HIP_SYS_DISABLE:-}" == 1 ]] || {
  printf '%s\n' 'mock cargo: FE2O3_HIP_SYS_DISABLE is not enforced' >&2
  exit 91
}
[[ "${FE2O3_HSA_RUNTIME_DISABLE:-}" == 1 ]] || {
  printf '%s\n' 'mock cargo: FE2O3_HSA_RUNTIME_DISABLE is not enforced' >&2
  exit 92
}
printf 'cargo' >>"${QUICKSTART_TEST_LOG}"
printf ' %q' "$@" >>"${QUICKSTART_TEST_LOG}"
printf '\n' >>"${QUICKSTART_TEST_LOG}"

output=
request=
simulator=0
doctor=0
for ((index = 1; index <= $#; index++)); do
  argument="${!index}"
  if [[ "${argument}" == --output ]]; then
    next=$((index + 1))
    output="${!next}"
  elif [[ "${argument}" == --request ]]; then
    next=$((index + 1))
    request="${!next}"
  elif [[ "${argument}" == fe2o3-kir-sim ]]; then
    simulator=1
  elif [[ "${argument}" == cargo-fe2o3 ]]; then
    doctor=1
  fi
done
if [[ -n "${output}" ]]; then
  printf 'mock-bundle' >"${output}"
fi
if ((simulator)); then
  [[ -z "${output}" ]] || {
    printf '%s\n' 'mock cargo: simulator must use stdout, not exporter --output' >&2
    exit 93
  }
  python3 -I -B - "${request}" "${QUICKSTART_TEST_MODE:-valid}" <<'PY'
import json
import sys

request, mode = sys.argv[1:]
if mode == "missing-output":
    sys.exit(0)
if mode == "malformed":
    print('{"schema":')
    sys.exit(0)
canary = request.endswith("/fill-canary-request.json")
data = "0x" + "00002a42" * 4 + ("a5a5a5a5deadbeef" if canary else "")
result = {
    "schema": "fe2o3-simulation-result-v1", "status": "ok",
    "authority": "observation_only", "simulated": True,
    "hardware_observed": False, "hardware_validation": False,
    "performance_prediction": False,
    "arguments": [{"kind": "buffer", "value": {
        "element": "f32", "access": "read_write", "alignment": 4,
        "bytes": data, "initialized": "0xffffff" if canary else "0xffff",
    }}],
    "shared_buffers": [],
    "counts": {
        "arguments": 1, "shared_buffers": 0, "invocations_executed": 4,
        "workgroups_visited": 1, "scheduled_slots_visited": 64,
        "steps_executed": 12, "events_emitted": 0,
    },
}
if mode == "wrong":
    result["arguments"][0]["value"]["bytes"] = "0x01002a42" + data[10:]
elif mode == "canary":
    result["arguments"][0]["value"]["bytes"] = data[:-2] + "00"
elif mode == "missing-field":
    del result["arguments"][0]["value"]["initialized"]
elif mode == "wrong-flags":
    result["hardware_observed"] = True
print(json.dumps(result, separators=(",", ":")))
sys.exit(17 if mode == "failed-output" else 0)
PY
fi
if ((doctor)) && [[ " $* " == *" doctor "* ]]; then
  printf '%s\n' \
    'direct-kfd-preflight: ready' \
    'device[0]: node=2 target=gfx942 wave-width=64 render=/dev/dri/renderD128 render-status=ready' \
    'runtime-libraries: HIP/HSA not-required-or-loaded'
fi
EOF
chmod 700 "${FIXTURE}/bin/cargo"

run_quickstart() {
  env \
    CARGO="${FIXTURE}/bin/cargo" \
    QUICKSTART_TEST_LOG="${LOG}" \
    QUICKSTART_TEST_MODE="${QUICKSTART_TEST_MODE:-valid}" \
    TMPDIR="${FIXTURE}/tmp" \
    PATH="${FIXTURE}/bin:/usr/bin:/bin" \
    bash "${REPO_ROOT}/scripts/quickstart.sh" "$@"
}

assert_no_temporary_files() {
  [[ -z "$(find "${FIXTURE}/tmp" -mindepth 1 -print -quit)" ]]
}

checked_fill() {
  run_quickstart simulate-source --crate fe2o3_fill \
    --request "${REPO_ROOT}/scripts/quickstart/fill-request.json" \
    --expectation "${REPO_ROOT}/scripts/quickstart/fill-expectation.json" \
    "$@" -- --package fe2o3-fill --lib
}

run_quickstart no-gpu >"${FIXTURE}/no-gpu.stdout" 2>"${FIXTURE}/no-gpu.stderr"
grep -F -- 'fe2o3-export-sim' "${LOG}" >/dev/null
grep -F -- 'build --locked --quiet -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract' \
  "${LOG}" >/dev/null
grep -F -- '--crate fe2o3_fill' "${LOG}" >/dev/null
grep -F -- '--bundle-version 1' "${LOG}" >/dev/null
grep -F -- '--package fe2o3-fill --lib' "${LOG}" >/dev/null
grep -F -- 'fe2o3-kir-sim' "${LOG}" >/dev/null
grep -F -- '"hardware_observed":false' "${FIXTURE}/no-gpu.stdout" >/dev/null
grep -F -- 'grants no compiler, artifact, GPU, or equivalence authority' \
  "${FIXTURE}/no-gpu.stderr" >/dev/null
grep -F -- 'fill-canary-request.json' "${LOG}" >/dev/null
grep -F -- '00002a4200002a4200002a4200002a42a5a5a5a5deadbeef' \
  "${FIXTURE}/no-gpu.stdout" >/dev/null
[[ "$(wc -l <"${LOG}")" -eq 3 ]]
assert_no_temporary_files
! grep -F -- '--matrix-bf16-numerics' "${LOG}" >/dev/null

: >"${LOG}"
checked_fill --output "${FIXTURE}/retained.fe2sim" \
  >"${FIXTURE}/checked.stdout" 2>"${FIXTURE}/checked.stderr"
[[ "$(cat "${FIXTURE}/retained.fe2sim")" == mock-bundle ]]
grep -F -- 'retained extraction-only simulation bundle:' "${FIXTURE}/checked.stderr" >/dev/null
grep -F -- '"initialized":"0xffff"' "${FIXTURE}/checked.stdout" >/dev/null
[[ "$(wc -l <"${LOG}")" -eq 3 ]]
assert_no_temporary_files

for mode in wrong missing-field malformed missing-output wrong-flags failed-output; do
  : >"${LOG}"
  set +e
  QUICKSTART_TEST_MODE="${mode}" checked_fill \
    >"${FIXTURE}/${mode}.stdout" 2>"${FIXTURE}/${mode}.stderr"
  status=$?
  set -e
  if [[ "${mode}" == failed-output ]]; then
    [[ "${status}" -eq 17 ]]
  else
    [[ "${status}" -eq 1 ]]
    grep -F -- 'simulation expectation:' "${FIXTURE}/${mode}.stderr" >/dev/null
  fi
  [[ ! -s "${FIXTURE}/${mode}.stdout" ]]
  [[ "$(wc -l <"${LOG}")" -eq 3 ]]
  assert_no_temporary_files
done

set +e
QUICKSTART_TEST_MODE=canary run_quickstart no-gpu \
  >"${FIXTURE}/canary.stdout" 2>"${FIXTURE}/canary.stderr"
status=$?
set -e
[[ "${status}" -eq 1 && ! -s "${FIXTURE}/canary.stdout" ]]
grep -F -- 'complete arguments mismatch' "${FIXTURE}/canary.stderr" >/dev/null
assert_no_temporary_files

: >"${LOG}"
set +e
QUICKSTART_TEST_MODE=wrong checked_fill --output "${FIXTURE}/failed-retained.fe2sim" \
  >"${FIXTURE}/failed-retained.stdout" 2>"${FIXTURE}/failed-retained.stderr"
status=$?
set -e
[[ "${status}" -eq 1 && ! -s "${FIXTURE}/failed-retained.stdout" ]]
[[ "$(cat "${FIXTURE}/failed-retained.fe2sim")" == mock-bundle ]]
assert_no_temporary_files

: >"${LOG}"
set +e
checked_fill --output "${FIXTURE}/retained.fe2sim" \
  >"${FIXTURE}/existing.stdout" 2>"${FIXTURE}/existing.stderr"
status=$?
set -e
[[ "${status}" -eq 2 && ! -s "${LOG}" && ! -s "${FIXTURE}/existing.stdout" ]]
[[ "$(cat "${FIXTURE}/retained.fe2sim")" == mock-bundle ]]
grep -F -- '--output already exists:' "${FIXTURE}/existing.stderr" >/dev/null
assert_no_temporary_files

printf '%s\n' '{"schema":"fe2o3-simulation-expectation-v1","arguments":null,"shared_buffers":[]}' \
  >"${FIXTURE}/invalid-expectation.json"
for expectation in "${FIXTURE}/missing-expectation.json" "${FIXTURE}/invalid-expectation.json"; do
  : >"${LOG}"
  set +e
  run_quickstart simulate-source --crate fe2o3_fill \
    --request "${REPO_ROOT}/scripts/quickstart/fill-request.json" \
    --expectation "${expectation}" -- --package fe2o3-fill --lib \
    >"${FIXTURE}/bad-expectation.stdout" 2>"${FIXTURE}/bad-expectation.stderr"
  status=$?
  set -e
  [[ "${status}" -eq 1 && ! -s "${LOG}" && ! -s "${FIXTURE}/bad-expectation.stdout" ]]
  assert_no_temporary_files
done

: >"${LOG}"
set +e
run_quickstart simulate-source --crate fe2o3_fill \
  --request "${REPO_ROOT}/scripts/quickstart/fill-request.json" --expectation \
  >"${FIXTURE}/missing-value.stdout" 2>"${FIXTURE}/missing-value.stderr"
status=$?
set -e
[[ "${status}" -eq 2 && ! -s "${LOG}" && ! -s "${FIXTURE}/missing-value.stdout" ]]
grep -F -- '--expectation requires a value' "${FIXTURE}/missing-value.stderr" >/dev/null
assert_no_temporary_files

: >"${LOG}"
run_quickstart doctor --require-tools-present >"${FIXTURE}/doctor.stdout"
grep -F -- 'doctor --require-tools-present' "${LOG}" >/dev/null
grep -F -- 'runtime-libraries: HIP/HSA not-required-or-loaded' \
  "${FIXTURE}/doctor.stdout" >/dev/null

: >"${LOG}"
run_quickstart source-check "${REPO_ROOT}/examples/vecadd/Cargo.toml"
[[ "$(wc -l <"${LOG}")" -eq 2 ]]
grep -F -- 'cargo-fe2o3 -- check --manifest-path' "${LOG}" >/dev/null
grep -F -- 'cargo-fe2o3 -- test --all-targets --manifest-path' "${LOG}" >/dev/null

: >"${LOG}"
set +e
run_quickstart gfx942-preflight >"${FIXTURE}/gfx942.stdout" 2>"${FIXTURE}/gfx942.stderr"
status=$?
set -e
[[ "${status}" -eq 3 ]]
grep -F -- 'doctor --require-gfx942' "${LOG}" >/dev/null
grep -F -- 'Worker V3 application route is not wired' "${FIXTURE}/gfx942.stderr" >/dev/null

set +e
run_quickstart simulate-source --crate bad --request "${FILL_REQUEST:-/missing}" \
  >"${FIXTURE}/invalid.stdout" 2>"${FIXTURE}/invalid.stderr"
status=$?
set -e
[[ "${status}" -eq 2 ]]
grep -F -- 'requires --crate, --request, and Cargo selection after --' \
  "${FIXTURE}/invalid.stderr" >/dev/null

: >"${LOG}"
run_quickstart simulate-source --crate fe2o3_fill \
  --request "${REPO_ROOT}/scripts/quickstart/fill-request.json" \
  --bundle-version 5 -- --package fe2o3-fill --lib \
  >"${FIXTURE}/v5.stdout" 2>"${FIXTURE}/v5.stderr"
grep -F -- '--bundle-version 5' "${LOG}" >/dev/null
grep -F -- '--bundle-v5' "${LOG}" >/dev/null
grep -F -- '"status":"ok"' "${FIXTURE}/v5.stdout" >/dev/null
assert_no_temporary_files

: >"${LOG}"
checked_fill --bundle-version 5 >"${FIXTURE}/checked-v5.stdout" 2>"${FIXTURE}/checked-v5.stderr"
grep -F -- '--bundle-v5' "${LOG}" >/dev/null
grep -F -- '"status":"ok"' "${FIXTURE}/checked-v5.stdout" >/dev/null
assert_no_temporary_files

set +e
run_quickstart simulate-source --crate fe2o3_fill \
  --request "${REPO_ROOT}/scripts/quickstart/fill-request.json" \
  --bundle-version 4 -- --package fe2o3-fill --lib \
  >"${FIXTURE}/invalid-version.stdout" 2>"${FIXTURE}/invalid-version.stderr"
status=$?
set -e
[[ "${status}" -eq 2 ]]
grep -F -- '--bundle-version must be exactly 1, 5, or 6' \
  "${FIXTURE}/invalid-version.stderr" >/dev/null


# V6 dispatch: explicit modes and omission, checked and unchecked output paths.
for numerics in default exact-v1 finite-sequential-fma-v1; do
  : >"${LOG}"
  options=(--bundle-version 6)
  if [[ "${numerics}" != default ]]; then
    options+=(--matrix-bf16-numerics "${numerics}")
  fi
  checked_fill "${options[@]}" >"${FIXTURE}/v6-${numerics}.stdout" 2>"${FIXTURE}/v6-${numerics}.stderr"
  grep -F -- '--bundle-version 6' "${LOG}" >/dev/null
  grep -F -- '--bundle-v6' "${LOG}" >/dev/null
  if [[ "${numerics}" == default ]]; then
    ! grep -F -- '--matrix-bf16-numerics' "${LOG}" >/dev/null
  else
    grep -F -- "--matrix-bf16-numerics ${numerics}" "${LOG}" >/dev/null
  fi
  [[ "$(wc -l <"${LOG}")" -eq 3 ]]
  assert_no_temporary_files
done

: >"${LOG}"
run_quickstart simulate-source --crate fe2o3_fill \
  --request "${REPO_ROOT}/scripts/quickstart/fill-request.json" \
  --bundle-version 6 --matrix-bf16-numerics finite-sequential-fma-v1 \
  -- --package fe2o3-fill --lib >"${FIXTURE}/v6-unchecked.stdout"
grep -F -- '--bundle-v6' "${LOG}" >/dev/null
grep -F -- '--matrix-bf16-numerics finite-sequential-fma-v1' "${LOG}" >/dev/null
assert_no_temporary_files

for version in 1 5; do
  : >"${LOG}"
  checked_fill --bundle-version "${version}" --matrix-bf16-numerics exact-v1 \
    >"${FIXTURE}/legacy-exact.stdout" 2>"${FIXTURE}/legacy-exact.stderr"
  grep -F -- '--matrix-bf16-numerics exact-v1' "${LOG}" >/dev/null
  : >"${LOG}"
  set +e
  checked_fill --bundle-version "${version}" --matrix-bf16-numerics finite-sequential-fma-v1 \
    >"${FIXTURE}/legacy-finite.stdout" 2>"${FIXTURE}/legacy-finite.stderr"
  status=$?
  set -e
  [[ "${status}" -eq 2 && ! -s "${LOG}" && ! -s "${FIXTURE}/legacy-finite.stdout" ]]
  grep -F -- 'requires --bundle-version 6' "${FIXTURE}/legacy-finite.stderr" >/dev/null
  assert_no_temporary_files
done

for invalid in unknown missing duplicate; do
  : >"${LOG}"
  options=(--bundle-version 6 --matrix-bf16-numerics)
  case "${invalid}" in
    unknown) options+=(unknown-mode) ;;
    missing) ;;
    duplicate) options+=(exact-v1 --matrix-bf16-numerics finite-sequential-fma-v1) ;;
  esac
  set +e
  checked_fill "${options[@]}" >"${FIXTURE}/bad-numerics.stdout" 2>"${FIXTURE}/bad-numerics.stderr"
  status=$?
  set -e
  [[ "${status}" -eq 2 && ! -s "${LOG}" && ! -s "${FIXTURE}/bad-numerics.stdout" ]]
  assert_no_temporary_files
done

printf '%s\n' 'quickstart shell tests passed'

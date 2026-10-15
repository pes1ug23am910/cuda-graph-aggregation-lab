#!/usr/bin/env bash
# Collect diagnostics separately from ordinary benchmark timings.
# See NVIDIA Nsight Systems, Nsight Compute CLI, and Compute Sanitizer manuals.
set -u
set -o pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
executable=${1:-"$script_dir/../build/cuda/graphlab"}
output_directory=${2:-"$script_dir/../results/profiles"}
nodes=${NODES:-4096}
features=${FEATURES:-64}
family=${FAMILY:-skewed}
seed=${SEED:-42}
nsys=${NSYS:-nsys}
ncu=${NCU:-ncu}
sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
if [[ ! -x "$executable" ]]; then printf 'Executable is unavailable: %s\n' "$executable" >&2; exit 2; fi
if ! command -v python3 >/dev/null 2>&1; then printf 'Python 3 is required for JSON command records.\n' >&2; exit 2; fi
run_directory="$output_directory/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p -- "$run_directory" || exit 2
status_file="$run_directory/status.jsonl"
had_failure=0
had_missing=0
last_run_code=0

record_status() {
    local label=$1 status=$2 code=$3 reason=$4 tool=$5
    shift 5
    python3 - "$status_file" "$label" "$status" "$code" "$reason" "$tool" "$@" <<'PY'
import datetime
import json
import sys
record = {
    "type": "diagnostic_command", "label": sys.argv[2], "status": sys.argv[3],
    "exit_code": None if sys.argv[4] == "null" else int(sys.argv[4]),
    "reason": sys.argv[5], "executable": sys.argv[6], "arguments": sys.argv[7:],
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(sys.argv[1], "a", encoding="utf-8") as output:
    print(json.dumps(record, separators=(",", ":")), file=output)
PY
    if [[ $? -ne 0 ]]; then printf 'Could not retain a command status.\n' >&2; exit 2; fi
}
run_diagnostic() {
    local label=$1 tool=$2
    shift 2
    local found
    found=$(command -v "$tool" 2>/dev/null) || found=''
    if [[ -z "$found" ]]; then
        record_status "$label" not_run null 'Tool is unavailable' "$tool" "$@"
        had_missing=1; last_run_code=77; return
    fi
    "$found" "$@" >"$run_directory/$label.log" 2>&1
    last_run_code=$?
    if [[ $last_run_code -eq 0 ]]; then
        record_status "$label" completed "$last_run_code" '' "$found" "$@"
    else
        record_status "$label" failed "$last_run_code" '' "$found" "$@"
        had_failure=1
    fi
}

run_diagnostic environment "$executable" info
run_diagnostic nsys-version "$nsys" --version
run_diagnostic ncu-version "$ncu" --version
run_diagnostic sanitizer-version "$sanitizer" --version
for backend in naive row_split cusparse; do
    app_arguments=(bench --backend "$backend" --nodes "$nodes" --features "$features" --family "$family" --seed "$seed" --groups 1 --calls 10 --warmup 3 --smoke)
    prefix="$run_directory/$backend"
    run_diagnostic "$backend-nsys" "$nsys" profile --trace=cuda --sample=none --cpuctxsw=none --stats=true --force-overwrite=true --output "$prefix-timeline" "$executable" "${app_arguments[@]}"
    # Kernel replay/collection time is diagnostic, never benchmark latency.
    run_diagnostic "$backend-ncu" "$ncu" --set basic --target-processes all --launch-count 3 --force-overwrite --export "$prefix-kernels" "$executable" "${app_arguments[@]}"
    sanitizer_arguments=(bench --backend "$backend" --nodes "$nodes" --features "$features" --family "$family" --seed "$seed" --groups 1 --calls 2 --warmup 1 --smoke)
    run_diagnostic "$backend-memcheck" "$sanitizer" --tool memcheck --error-exitcode 99 --leak-check full "$executable" "${sanitizer_arguments[@]}"
    memcheck_code=$last_run_code
    for check in initcheck synccheck racecheck; do
        if [[ $memcheck_code -eq 0 ]]; then
            run_diagnostic "$backend-$check" "$sanitizer" --tool "$check" --error-exitcode 99 "$executable" "${sanitizer_arguments[@]}"
        else
            record_status "$backend-$check" not_run null 'Memcheck did not complete successfully' "$sanitizer" --tool "$check" --error-exitcode 99 "$executable" "${sanitizer_arguments[@]}"
            had_missing=1
        fi
    done
done
printf 'Diagnostic logs and command statuses: %s\n' "$run_directory"
printf 'Inspect report files and logs before interpreting completed commands as usable evidence.\n'
if [[ $had_failure -ne 0 ]]; then exit 1; fi
if [[ $had_missing -ne 0 ]]; then exit 77; fi
exit 0

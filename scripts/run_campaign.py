"""Run the frozen matrix, alternate backend order, and retain every invocation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import measurement_error


def invoke(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


def safe_invoke(command):
    try:
        return invoke(command)
    except OSError as error:
        return 127, "", str(error)


def decode_output(stdout):
    records = []
    valid = True
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(row, dict):
                raise ValueError("output record is not an object")
            records.append(row)
        except (ValueError, json.JSONDecodeError) as error:
            records.append({"type": "invalid_output", "text": line, "error": str(error)})
            valid = False
    return records, valid


def expected_measurements(fixture, groups, backends, calls, warmup, smoke, group_offset=0):
    expected = []
    for group in range(group_offset, group_offset + groups):
        order = backends if group % 2 == 0 else list(reversed(backends))
        for backend in order:
            expected.append(dict(fixture, backend=backend, group=group, calls=calls, warmup=warmup,
                                 smoke=smoke, status="passed",
                                 resident_boundary="host_validation_allocation_aggregation" if backend == "cpu" else "device_normalized_aggregation"))
    return expected


def validate_invocation(records, expected, fixture_identities):
    """Validate every requested batch, including its position in the emitted order."""
    if len(records) != len(expected):
        return f"expected exactly {len(expected)} measurements, received {len(records)} records"
    identities = dict(fixture_identities)
    for index, (row, requested) in enumerate(zip(records, expected)):
        if row.get("type") != "measurement" or row.get("status") != "passed":
            return f"record {index} is not a passed measurement"
        error = measurement_error(row)
        if error:
            return f"record {index}: {error}"
        if any(row.get(key) != value for key, value in requested.items()):
            return f"record {index} does not match requested batch identity or order"
        fixture_key = tuple(requested[key] for key in ("nodes", "features", "family", "seed"))
        observed = (row["fingerprint"], row.get("edges"), row.get("max_degree"))
        if fixture_key in identities and identities[fixture_key] != observed:
            return f"record {index}: input identity differs across batches or backends"
        identities[fixture_key] = observed
    fixture_identities.update(identities)
    return ""


def source_state(root):
    code, revision, error = safe_invoke(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty_code, dirty_text, dirty_error = safe_invoke(["git", "-C", str(root), "status", "--porcelain"])
    snapshot = {}
    candidates = [root / "CMakeLists.txt"]
    for directory in ("src", "include", "tests", "scripts", "cmake"):
        folder = root / directory
        if folder.exists():
            candidates.extend(path for path in folder.rglob("*") if path.is_file() and path.suffix in {".cpp", ".hpp", ".cu", ".cuh", ".py", ".cmake"})
    for path in sorted(candidates):
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    return {"source_revision": revision.strip() if code == 0 else "unavailable",
            "source_dirty": bool(dirty_text.strip()) if dirty_code == 0 else None,
            "source_revision_error": error if code else "",
            "source_status_error": dirty_error if dirty_code else "",
            "source_tree_sha256": digest, "source_files_sha256": snapshot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--backends", default="naive,row_split,cusparse",
                        choices=["naive,row_split,cusparse"],
                        help="fixed GPU backend order required by the campaign protocol")
    args = parser.parse_args()
    requested_backends = args.backends.split(",")
    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error("--exe must identify an existing executable")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backends = ["cpu"] if args.cpu_only else requested_backends
    nodes = [33] if args.smoke else [4096, 65536]
    widths = [7, 33] if args.smoke else [32, 64, 128]
    families = ["regular8", "regular32", "skewed"]
    seeds = [7] if args.smoke else [7, 42, 2026]
    manifest = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "status": "initializing", "completed_batches": 0,
                "platform": platform.platform(), "python": platform.python_version(),
                "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "cpu_only": args.cpu_only, "smoke": args.smoke, "backends": backends,
                "invocation_strategy": "one CPU batch per process" if args.cpu_only else "all groups and backends per fixture process",
                "telemetry_sampling": "before and after each fixture process; CPU campaigns record no GPU telemetry",
                "groups": 2 if args.smoke else 5, "calls_per_batch": 3 if args.smoke else 100,
                "warmup_calls": 1 if args.smoke else 20,
                "backend_order": "forward on even groups, reversed on odd groups",
                "power_thermal_background_state": "not controlled; inspect telemetry.jsonl",
                "matrix": {"nodes": nodes, "features": widths, "families": families, "seeds": seeds}}
    manifest.update(source_state(Path(__file__).resolve().parents[1]))
    manifest["campaign_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    def write_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def finish(status, code, reason=""):
        manifest.update(status=status, ended_utc=datetime.now(timezone.utc).isoformat(), exit_code=code)
        if reason:
            manifest["failure_reason"] = reason
        manifest["artifact_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in output.iterdir() if path.is_file() and path.name != "manifest.json"}
        write_manifest()
        print(json.dumps({"status": status, "batches": manifest["completed_batches"]}))
        return code

    write_manifest()
    rc, stdout, stderr = safe_invoke([str(exe), "info"])
    manifest.update(environment_stdout=stdout, environment_stderr=stderr, environment_exit_code=rc)
    environment, valid_environment = decode_output(stdout)
    if rc or not valid_environment or len(environment) != 1 or environment[0].get("type") != "environment":
        return finish("failed", rc or 1, "environment inspection failed or returned invalid output")
    manifest["environment_id"] = hashlib.sha256(json.dumps(environment[0], sort_keys=True).encode()).hexdigest()
    verify_cmd = [str(exe), "verify", "--backend", "cpu" if args.cpu_only else "all"]
    code, verification, verify_err = safe_invoke(verify_cmd)
    manifest["verification_exit_code"] = code
    (output / "verification.jsonl").write_text(verification, encoding="utf-8")
    (output / "verification.stderr.txt").write_text(verify_err, encoding="utf-8")
    verification_records, valid_verification = decode_output(verification)
    cases = [row for row in verification_records if row.get("type") == "verification"]
    summaries = [row for row in verification_records if row.get("type") == "verification_summary"]
    verified = (valid_verification and len(summaries) == 1 and summaries[0].get("status") == "passed"
                and type(summaries[0].get("cases")) is int and summaries[0]["cases"] == len(cases)
                and bool(cases) and all(row.get("status") == "passed" for row in cases)
                and len(cases) + 1 == len(verification_records))
    if code or not verified:
        return finish("not_run" if code == 77 else "failed", code or 1, "verification unavailable, failed, or incomplete")
    manifest["status"] = "running"
    write_manifest()
    fixture_identities = {}
    failed = False
    with (output / "raw.jsonl").open("w", encoding="utf-8") as raw, (output / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry, (output / "invocations.jsonl").open("w", encoding="utf-8") as invocations:
        for n, width, family, seed in itertools.product(nodes, widths, families, seeds):
            fixture = {"nodes": n, "features": width, "family": family, "seed": seed}
            schedules = [("cpu", 1, group) for group in range(manifest["groups"])] if args.cpu_only else [("all", manifest["groups"], 0)]
            for backend_option, group_count, group_offset in schedules:
                expected = expected_measurements(fixture, group_count, backends, manifest["calls_per_batch"], manifest["warmup_calls"], args.smoke, group_offset)
                command = [str(exe), "bench", "--backend", backend_option, "--nodes", str(n), "--features", str(width),
                           "--family", family, "--seed", str(seed), "--groups", str(group_count), "--group-offset", str(group_offset),
                           "--calls", str(manifest["calls_per_batch"]), "--warmup", str(manifest["warmup_calls"])]
                if args.smoke:
                    command.append("--smoke")

                def record_telemetry(phase):
                    if args.cpu_only:
                        telemetry_code, t_out, t_err = 0, "CPU-only campaign; GPU telemetry not requested", ""
                    else:
                        telemetry_code, t_out, t_err = safe_invoke(["nvidia-smi", "--query-gpu=name,driver_version,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu", "--format=csv,noheader,nounits"])
                    telemetry.write(json.dumps(dict(fixture, backend=backend_option, first_group=group_offset,
                                                    groups=group_count, phase=phase, utc=datetime.now(timezone.utc).isoformat(),
                                                    code=telemetry_code, stdout=t_out, stderr=t_err)) + "\n")

                record_telemetry("before")
                process_code, stdout, stderr = safe_invoke(command)
                record_telemetry("after")
                invocations.write(json.dumps({"command": command[1:], "expected_batches": len(expected),
                                              "exit_code": process_code, "stdout": stdout, "stderr": stderr}, allow_nan=False) + "\n")
                records, valid = decode_output(stdout)
                reason = validate_invocation(records, expected, fixture_identities) if valid else "invalid JSON output"
                effective_code = process_code or (1 if reason else 0)
                metadata = {"process_exit_code": effective_code, "invocation": command[1:],
                            "campaign_id": manifest["campaign_id"], "campaign_expected_groups": manifest["groups"],
                            "environment_id": manifest["environment_id"], "executable_sha256": manifest["executable_sha256"],
                            "source_tree_sha256": manifest["source_tree_sha256"]}
                for index, record in enumerate(records):
                    requested = expected[min(index, len(expected) - 1)]
                    record.update(metadata)
                    record["invocation_identity"] = {key: requested[key] for key in ("nodes", "features", "family", "seed", "backend", "group")}
                    raw.write(json.dumps(record, allow_nan=False) + "\n")
                if stderr or effective_code:
                    # One diagnostic per backend prevents a failed multi-backend
                    # retry from silently completing a matching summary group.
                    for backend in backends:
                        identity = dict(fixture, backend=backend, group=group_offset)
                        diagnostic = dict(metadata, type="process_diagnostic", invocation_identity=identity,
                                          exit_code=effective_code, process_return_code=process_code,
                                          stderr=stderr, reason=reason, smoke=args.smoke)
                        raw.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                raw.flush()
                invocations.flush()
                telemetry.flush()
                if effective_code:
                    failed = True
                    break
                manifest["completed_batches"] += len(expected)
            if failed:
                break
    return finish("failed" if failed else "completed", 1 if failed else 0,
                  "benchmark invocation failed or did not match its requested input" if failed else "")


if __name__ == "__main__":
    raise SystemExit(main())

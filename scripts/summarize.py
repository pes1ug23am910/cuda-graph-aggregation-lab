"""Summarize complete campaigns while retaining failed and malformed evidence."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def quantile(values, fraction):
    values = sorted(values)
    if not values or not 0 <= fraction <= 1 or any(not math.isfinite(value) for value in values):
        raise ValueError("quantiles require finite values and a fraction in [0,1]")
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def measurement_error(row):
    if row.get("schema") != 1 or row.get("contract") != "csr-mean-v1":
        return "unsupported measurement schema or contract"
    for field, lower in (("nodes", 0), ("features", 1), ("group", 0), ("calls", 1), ("warmup", 0)):
        value = row.get(field)
        if type(value) is not int or value < lower:
            return "invalid " + field
    if "seed" not in row:
        return "missing seed"
    if row.get("seed") is not None and (type(row.get("seed")) is not int or row["seed"] < 0):
        return "invalid seed"
    for field in ("family", "fingerprint", "backend", "resident_boundary"):
        if not isinstance(row.get(field), str) or not row[field]:
            return "missing or invalid " + field
    if type(row.get("smoke")) is not bool:
        return "missing or invalid smoke flag"
    value = row.get("per_call_ms")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return "invalid per-call latency"
    for field in ("campaign_id", "environment_id", "executable_sha256", "source_tree_sha256"):
        if field in row and not isinstance(row[field], str):
            return "invalid " + field
    expected = row.get("campaign_expected_groups")
    if expected is not None and (type(expected) is not int or expected < 1):
        return "invalid expected group count"
    return None


def request_identity(row):
    request = row.get("invocation_identity", row)
    try:
        return (request["nodes"], request["features"], request["family"], request["seed"],
                request["backend"], row.get("campaign_id", ""))
    except (KeyError, TypeError):
        return None


def summarize(records, minimum_groups=5):
    if minimum_groups < 1:
        raise ValueError("minimum_groups must be positive")
    groups = defaultdict(list)
    failures = []
    blocked_requests = set()
    for original in records:
        if not isinstance(original, dict):
            failures.append({"type": "invalid_record", "record": original})
            continue
        row = original
        error = None
        if row.get("type") == "measurement" and row.get("status") == "passed" and not row.get("process_exit_code", 0):
            error = measurement_error(row)
            if error is None and row["smoke"]:
                continue
            if error is None:
                key = tuple(row[field] for field in ("nodes", "features", "family", "seed", "fingerprint", "backend", "resident_boundary")) + tuple(row.get(field, "") for field in ("campaign_id", "environment_id", "executable_sha256", "source_tree_sha256"))
                groups[key].append(row)
                continue
        failure = dict(row)
        if error:
            failure["summary_rejection"] = error
        failures.append(failure)
        identity = request_identity(row)
        if identity is not None and not row.get("smoke", False):
            try:
                blocked_requests.add(identity)
            except TypeError:
                pass
    result = []
    # JSON encoding is a stable sort key even for an external fixture with seed=null.
    for key, rows in sorted(groups.items(), key=lambda item: json.dumps(item[0])):
        identities = [row["group"] for row in rows]
        expected_counts = {row.get("campaign_expected_groups") for row in rows}
        complete = (len(set(identities)) >= minimum_groups and len(set(identities)) == len(identities)
                    and all(row["calls"] >= 100 and row["warmup"] >= 20 for row in rows)
                    and request_identity(rows[0]) not in blocked_requests and len(expected_counts) == 1)
        expected = next(iter(expected_counts))
        if expected is not None:
            complete = complete and set(identities) == set(range(expected))
        times = [row["per_call_ms"] for row in rows]
        result.append(dict(zip(("nodes", "features", "family", "seed", "fingerprint", "backend", "boundary", "campaign_id", "environment_id", "executable_sha256", "source_tree_sha256"), key),
                           groups=len(rows), group_ids=sorted(identities), complete=complete,
                           median_ms=statistics.median(times), q1_ms=quantile(times, 0.25),
                           q3_ms=quantile(times, 0.75), min_ms=min(times), max_ms=max(times)))
    return result, failures


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for number, line in enumerate(args.raw.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
        except (ValueError, json.JSONDecodeError) as error:
            records.append({"type": "invalid_json", "line": number, "text": line, "error": str(error)})
    rows, failures = summarize(records)
    lines = ["# Benchmark results", "", "Each latency is the median of per-call averages from independent batches. Q1 and Q3 use linear interpolation.", "", f"Failed or diagnostic records retained: {len(failures)}.", "", "| N | F | Family | Seed | Backend | Fingerprint | Boundary | Groups | Complete | Median ms | Q1–Q3 ms |", "|---:|---:|---|---:|---|---|---|---:|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['nodes']} | {row['features']} | {row['family']} | {row['seed']} | {row['backend']} | {row['fingerprint']} | {row['boundary']} | {row['groups']} | {row['complete']} | {row['median_ms']:.6g} | {row['q1_ms']:.6g}–{row['q3_ms']:.6g} |")
    if not rows:
        lines += ["", "No eligible campaign measurements. Smoke runs are excluded."]
    lines += ["", "Complete groups require unique batch identities, matching recorded campaign and environment identities, adequate repetitions, and no failed attempts for the same requested input/backend. Failed retries are retained and make that group incomplete. Comparisons also require matching input fingerprints, group identities, execution boundaries, toolchain and device conditions. CPU timings include validation and allocation and cannot be compared as resident GPU timings. No speedup is inferred from incomplete or unmatched groups.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(json_safe({"rows": rows, "diagnostics": failures}), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

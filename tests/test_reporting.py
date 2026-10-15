"""Regression tests for benchmark eligibility, provenance and failed evidence."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_campaign
import summarize


def measurement(group=0, **changes):
    row = {"type": "measurement", "schema": 1, "contract": "csr-mean-v1", "status": "passed",
           "smoke": False, "nodes": 4096, "features": 32, "family": "regular8", "seed": 7,
           "fingerprint": "0123456789abcdef", "backend": "naive", "group": group,
           "calls": 100, "warmup": 20, "resident_boundary": "device_normalized_aggregation",
           "per_call_ms": float(group + 1), "process_exit_code": 0}
    row.update(changes)
    return row


class SummaryTests(unittest.TestCase):
    def test_complete_group_quantiles(self):
        rows, failures = summarize.summarize([measurement(i) for i in range(5)])
        self.assertFalse(failures)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["complete"])
        self.assertEqual(rows[0]["group_ids"], list(range(5)))
        self.assertEqual((rows[0]["median_ms"], rows[0]["q1_ms"], rows[0]["q3_ms"]), (3, 2, 4))

    def test_incomplete_and_duplicate_group_ids(self):
        for identities in ([0, 1, 2, 3], [0, 1, 2, 3, 3]):
            with self.subTest(identities=identities):
                rows, _ = summarize.summarize([measurement(i) for i in identities])
                self.assertFalse(rows[0]["complete"])

    def test_insufficient_repetitions_remain_visible(self):
        for change in ({"calls": 99}, {"warmup": 19}):
            rows, failures = summarize.summarize([measurement(i, **(change if i == 4 else {})) for i in range(5)])
            self.assertFalse(rows[0]["complete"])
            self.assertEqual(rows[0]["groups"], 5)
            self.assertFalse(failures)

    def test_smoke_is_excluded(self):
        rows, failures = summarize.summarize([measurement(i, smoke=True, calls=3, warmup=1) for i in range(5)])
        self.assertEqual(rows, [])
        self.assertEqual(failures, [])

    def test_failed_measurements_are_retained_and_excluded(self):
        for change in ({"status": "failed"}, {"process_exit_code": 3}):
            bad = measurement(4, **change)
            rows, failures = summarize.summarize([measurement(i) for i in range(4)] + [bad])
            self.assertFalse(rows[0]["complete"])
            self.assertEqual(rows[0]["groups"], 4)
            self.assertIn(bad, failures)

    def test_failed_retry_blocks_complete_label(self):
        rows, failures = summarize.summarize([measurement(i) for i in range(5)] + [measurement(2, status="failed")])
        self.assertFalse(rows[0]["complete"])
        self.assertEqual(len(failures), 1)

    def test_diagnostic_for_requested_input_blocks_complete_label(self):
        diagnostic = {"type": "process_diagnostic", "exit_code": 3,
                      "invocation_identity": {key: measurement()[key] for key in ("nodes", "features", "family", "seed", "backend")}}
        rows, failures = summarize.summarize([measurement(i) for i in range(5)] + [diagnostic])
        self.assertFalse(rows[0]["complete"])
        self.assertIn(diagnostic, failures)

    def test_nonfinite_and_negative_latency_cannot_enter_summary(self):
        for invalid in (math.nan, math.inf, -math.inf, -1.0, "1.0", True):
            with self.subTest(value=invalid):
                rows, failures = summarize.summarize([measurement(0, per_call_ms=invalid)])
                self.assertEqual(rows, [])
                self.assertEqual(len(failures), 1)
                self.assertIn("summary_rejection", failures[0])

    def test_missing_identity_field_is_retained_as_diagnostic(self):
        row = measurement()
        del row["seed"]
        rows, failures = summarize.summarize([row])
        self.assertEqual(rows, [])
        self.assertEqual(len(failures), 1)

    def test_mismatched_identity_never_combines_groups(self):
        for field, value in (("fingerprint", "fedcba9876543210"), ("resident_boundary", "host_allocation"),
                             ("seed", 42), ("backend", "cusparse"), ("campaign_id", "different"),
                             ("environment_id", "different"), ("executable_sha256", "different"),
                             ("source_tree_sha256", "different")):
            with self.subTest(field=field):
                records = [measurement(i) for i in range(4)] + [measurement(4, **{field: value})]
                rows, _ = summarize.summarize(records)
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(not row["complete"] for row in rows))

    def test_schema_and_contract_mismatch_are_diagnostics(self):
        for change in ({"schema": 2}, {"contract": "different"}):
            rows, failures = summarize.summarize([measurement(i) for i in range(4)] + [measurement(4, **change)])
            self.assertEqual(len(failures), 1)
            self.assertFalse(rows[0]["complete"])

    def test_expected_campaign_ids_cannot_have_holes(self):
        rows, _ = summarize.summarize([measurement(i, campaign_expected_groups=5) for i in [1, 2, 3, 4, 5]])
        self.assertFalse(rows[0]["complete"])

    def test_quantile_interpolation_and_singleton(self):
        self.assertEqual(summarize.quantile([10, 1, 9, 2], 0.25), 1.75)
        self.assertEqual(summarize.quantile([10, 1, 9, 2], 0.75), 9.25)
        self.assertEqual(summarize.quantile([4], 0.25), 4)
        for values, fraction in (([], 0.5), ([1], -0.1), ([1], 1.1), ([math.nan], 0.5)):
            with self.assertRaises(ValueError):
                summarize.quantile(values, fraction)

    def test_cli_retains_malformed_json_and_writes_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / "raw.jsonl"
            raw.write_text(json.dumps(measurement(0)) + "\nnot json\n" + '{"type":"measurement","per_call_ms":NaN}\n', encoding="utf-8")
            output = folder / "summary.md"
            with patch.object(sys, "argv", ["summarize", str(raw), "--output", str(output)]):
                summarize.main()
            result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"), parse_constant=lambda value: self.fail(value))
            self.assertEqual(len(result["diagnostics"]), 2)
            self.assertFalse(result["rows"][0]["complete"])


class InvocationValidationTests(unittest.TestCase):
    def fixture(self):
        return {"nodes": 33, "features": 7, "family": "skewed", "seed": 7}

    def records(self, expected):
        return [measurement(**requested) for requested in expected]

    def test_all_backends_alternate_within_one_fixture(self):
        expected = run_campaign.expected_measurements(self.fixture(), 5, ["naive", "row_split", "cusparse"], 100, 20, False)
        self.assertEqual(len(expected), 15)
        self.assertEqual([row["backend"] for row in expected[:6]], ["naive", "row_split", "cusparse", "cusparse", "row_split", "naive"])
        known = {}
        self.assertEqual(run_campaign.validate_invocation(self.records(expected), expected, known), "")
        self.assertEqual(len(known), 1)

    def test_missing_extra_duplicate_and_reordered_batches_rejected(self):
        expected = run_campaign.expected_measurements(self.fixture(), 2, ["naive", "row_split", "cusparse"], 3, 1, True)
        valid = self.records(expected)
        for records in (valid[:-1], valid + [valid[-1]], valid[:-1] + [valid[0]], list(reversed(valid))):
            with self.subTest(length=len(records)):
                self.assertTrue(run_campaign.validate_invocation(records, expected, {}))

    def test_identity_mismatch_and_partial_failure_are_rejected(self):
        expected = run_campaign.expected_measurements(self.fixture(), 2, ["naive", "row_split", "cusparse"], 3, 1, True)
        for change in ({"fingerprint": "different"}, {"seed": 42}, {"features": 33}, {"status": "failed"}, {"per_call_ms": math.inf}):
            records = self.records(expected)
            records[-1].update(change)
            known = {}
            with self.subTest(change=change):
                self.assertTrue(run_campaign.validate_invocation(records, expected, known))
                self.assertEqual(known, {})
        records = self.records(expected)[:2] + [{"type": "error", "message": "device failed"}]
        self.assertTrue(run_campaign.validate_invocation(records, expected, {}))

    def test_existing_fixture_identity_cannot_change(self):
        expected = run_campaign.expected_measurements(self.fixture(), 1, ["naive"], 100, 20, False)
        known = {}
        self.assertFalse(run_campaign.validate_invocation(self.records(expected), expected, known))
        changed = self.records(expected)
        changed[0]["fingerprint"] = "different"
        self.assertTrue(run_campaign.validate_invocation(changed, expected, known))


class CampaignTests(unittest.TestCase):
    def test_backend_subsets_and_reordering_rejected_before_side_effects(self):
        for selected in ("naive", "naive,cusparse", "cusparse,row_split,naive"):
            with self.subTest(backends=selected), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                exe = folder / "graphlab.exe"
                exe.write_bytes(b"unused executable fixture")
                output = folder / "campaign"
                argv = ["run_campaign", "--exe", str(exe), "--output", str(output),
                        "--backends", selected]
                errors = io.StringIO()
                with patch.object(sys, "argv", argv), \
                     patch.object(run_campaign, "invoke") as invoke, \
                     contextlib.redirect_stderr(errors), \
                     self.assertRaises(SystemExit) as raised:
                    run_campaign.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("invalid choice", errors.getvalue())
                invoke.assert_not_called()
                self.assertFalse(output.exists())

    def run_fake(self, mode="valid", gpu=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        exe = folder / "graphlab.exe"
        exe.write_bytes(b"test executable identity")
        output = folder / "campaign"
        commands = []
        bench_calls = 0

        def fake_invoke(command):
            nonlocal bench_calls
            commands.append(command)
            if command[0] == "git":
                if mode == "missing_git":
                    raise FileNotFoundError("git unavailable")
                return (0, "a" * 40 + "\n", "") if "rev-parse" in command else (0, " M src/core.cpp\n", "")
            if command[0] == "nvidia-smi":
                return 0, "test GPU telemetry", ""
            if command[1] == "info":
                return 0, json.dumps({"type": "environment", "cuda_build": False, "gpu_available": False}) + "\n", ""
            if command[1] == "verify":
                if mode == "bad_verify":
                    return 0, '{"type":"verification_summary","status":"passed","cases":1}\n', ""
                if mode == "failed_verify":
                    return 77, '{"type":"verification","status":"not_run"}\n', "unavailable"
                return 0, '{"type":"verification","status":"passed"}\n{"type":"verification_summary","status":"passed","cases":1}\n', ""
            self.assertEqual(command[1], "bench")
            bench_calls += 1
            if mode == "empty":
                return 0, "", ""
            if mode == "diagnostic_only":
                return 0, '{"type":"error","message":"failed"}\n', ""
            options = {command[i]: command[i + 1] for i in range(2, len(command) - 1, 2)}
            fixture = {"nodes": int(options["--nodes"]), "features": int(options["--features"]),
                       "family": options["--family"], "seed": int(options["--seed"])}
            selected = ["naive", "row_split", "cusparse"] if options["--backend"] == "all" else [options["--backend"]]
            expected = run_campaign.expected_measurements(fixture, int(options["--groups"]), selected, 3, 1, True, int(options["--group-offset"]))
            records = [measurement(**requested) for requested in expected]
            if mode == "mismatch":
                records[-1]["seed"] = 42
            if mode == "changed_fingerprint" and bench_calls == 2:
                records[-1]["fingerprint"] = "fedcba9876543210"
            if mode == "nonfinite":
                records[-1]["per_call_ms"] = math.nan
            if mode == "partial_gpu":
                records = records[:1] + [{"type": "error", "message": "device failed after partial output"}]
            code = 3 if mode in {"failed_process", "partial_gpu"} else 0
            return code, "".join(json.dumps(record) + "\n" for record in records), "device failed" if code else ""

        argv = ["run_campaign", "--exe", str(exe), "--output", str(output), "--smoke"]
        if not gpu:
            argv.append("--cpu-only")
        with patch.object(sys, "argv", argv), patch.object(run_campaign, "invoke", side_effect=fake_invoke), contextlib.redirect_stdout(io.StringIO()):
            code = run_campaign.main()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        return code, manifest, output, commands

    def test_valid_smoke_campaign_retains_every_batch(self):
        code, manifest, output, commands = self.run_fake()
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["completed_batches"], 12)
        records = [json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 12)
        self.assertTrue(all(row["campaign_id"] == manifest["campaign_id"] for row in records))
        self.assertEqual(summarize.summarize(records), ([], []))
        git_commands = [command for command in commands if command[0] == "git"]
        self.assertTrue(all(command[1:3] == ["-C", str(SCRIPTS.parent)] for command in git_commands))
        self.assertTrue(manifest["source_dirty"])
        self.assertEqual(len(manifest["source_tree_sha256"]), 64)
        self.assertIn("invocations.jsonl", manifest["artifact_sha256"])

    def test_gpu_fixture_uses_one_process_for_all_groups(self):
        code, manifest, output, commands = self.run_fake(gpu=True)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["completed_batches"], 36)
        self.assertEqual(manifest["backends"], ["naive", "row_split", "cusparse"])
        bench = [command for command in commands if len(command) > 1 and command[1] == "bench"]
        self.assertEqual(len(bench), 6)
        self.assertTrue(all(command[command.index("--backend") + 1] == "all" for command in bench))
        self.assertTrue(all(command[command.index("--groups") + 1] == "2" for command in bench))
        records = [json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 36)
        self.assertEqual([row["backend"] for row in records[:6]], ["naive", "row_split", "cusparse", "cusparse", "row_split", "naive"])
        self.assertEqual(len([command for command in commands if command[0] == "nvidia-smi"]), 12)

    def test_gpu_partial_output_is_retained_but_not_counted(self):
        code, manifest, output, _ = self.run_fake("partial_gpu", gpu=True)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["completed_batches"], 0)
        records = [json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["type"], "measurement")
        self.assertEqual(records[0]["process_exit_code"], 3)
        self.assertEqual({row["invocation_identity"]["backend"] for row in records if row["type"] == "process_diagnostic"}, {"naive", "row_split", "cusparse"})
        self.assertEqual(summarize.summarize(records)[0], [])

    def test_missing_git_does_not_prevent_campaign(self):
        code, manifest, _, _ = self.run_fake("missing_git")
        self.assertEqual(code, 0)
        self.assertEqual(manifest["source_revision"], "unavailable")
        self.assertIsNone(manifest["source_dirty"])

    def test_invalid_success_output_is_a_failure(self):
        for mode in ("empty", "diagnostic_only", "mismatch", "nonfinite", "changed_fingerprint"):
            with self.subTest(mode=mode):
                code, manifest, output, _ = self.run_fake(mode)
                self.assertNotEqual(code, 0)
                self.assertEqual(manifest["status"], "failed")
                self.assertLessEqual(manifest["completed_batches"], 1)
                records = [json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertTrue(any(row["type"] == "process_diagnostic" for row in records))
                self.assertTrue((output / "invocations.jsonl").read_text(encoding="utf-8"))

    def test_failed_process_preserves_stdout_and_exit_code(self):
        code, manifest, output, _ = self.run_fake("failed_process")
        self.assertEqual(code, 1)
        self.assertEqual(manifest["completed_batches"], 0)
        invocations = [json.loads(line) for line in (output / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(invocations[0]["exit_code"], 3)
        self.assertEqual(json.loads(invocations[0]["stdout"])["type"], "measurement")

    def test_preflight_failure_finalizes_manifest(self):
        for mode in ("bad_verify", "failed_verify"):
            with self.subTest(mode=mode):
                code, manifest, output, commands = self.run_fake(mode)
                self.assertNotEqual(code, 0)
                self.assertIn(manifest["status"], ("failed", "not_run"))
                self.assertIn("ended_utc", manifest)
                self.assertIn("verification.jsonl", manifest["artifact_sha256"])
                self.assertFalse(any(len(command) > 1 and command[1] == "bench" for command in commands))
                self.assertFalse((output / "raw.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

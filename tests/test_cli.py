"""CLI parser regression tests; set GRAPHLAB_EXE to select a built executable."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def executable():
    explicit = os.environ.get("GRAPHLAB_EXE")
    if explicit:
        selected = Path(explicit).resolve()
        if not selected.is_file():
            raise AssertionError("GRAPHLAB_EXE does not point to an existing executable: " + str(selected))
        return selected
    names = ["graphlab.exe", "graphlab"] if os.name == "nt" else ["graphlab", "graphlab.exe"]
    for directory in (ROOT / "build-cpu", ROOT / "build-cpu" / "Release", ROOT / "build-cpu" / "Debug",
                      ROOT / "build" / "Release", ROOT / "build", ROOT / "build" / "Debug"):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = executable()
        if cls.exe is None:
            raise unittest.SkipTest("No built graphlab executable; set GRAPHLAB_EXE to enable CLI checks")

    def invoke(self, *arguments):
        result = subprocess.run([str(self.exe), *map(str, arguments)], capture_output=True, text=True, timeout=10, check=False)
        self.assertTrue(result.stdout.strip(), result.stderr)
        records = [json.loads(line, parse_constant=lambda value: self.fail("nonstandard JSON: " + value))
                   for line in result.stdout.splitlines() if line.strip()]
        return result.returncode, records

    def validate_text(self, content):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.csr"
            fixture.write_text(content, encoding="utf-8")
            return self.invoke("validate", "--input", fixture)

    def assert_rejected(self, content, expected="invalid_input"):
        code, records = self.validate_text(content)
        self.assertNotEqual(code, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "error")
        self.assertEqual(records[0]["code"], expected)

    def test_hostile_maximum_header_rejected_before_allocation(self):
        # Four times (N*F + N+1 + E) used to wrap uint64 to zero.
        code, records = self.validate_text("graphlab-csr-v1\n2147483647 2147483647 2147483647\n")
        self.assertEqual(code, 2)
        self.assertEqual(records[0]["code"], "unsupported_domain")
        self.assertIn("budget", records[0]["message"])

    def test_large_offset_and_edge_allocation_rejected(self):
        self.assert_rejected("graphlab-csr-v1\n2147483647 1 2147483647\n", "unsupported_domain")

    def test_malformed_headers(self):
        for text in ("", "unknown 1 1 0\n", "graphlab-csr-v1 -1 1 0\n",
                     "graphlab-csr-v1 1 0 0\n", "graphlab-csr-v1 1 1 -1\n",
                     "graphlab-csr-v1 2147483648 1 0\n", "graphlab-csr-v1 1 two 0\n"):
            with self.subTest(text=text):
                self.assert_rejected(text)

    def test_truncated_and_malformed_arrays(self):
        for text in ("graphlab-csr-v1 2 1 1\n0 1\n",
                     "graphlab-csr-v1 2 1 1\n0 1 1\n",
                     "graphlab-csr-v1 2 1 1\n0 1 1\n0\n1\n",
                     "graphlab-csr-v1 1 1 1\n0 nope\n0\n1\n",
                     "graphlab-csr-v1 1 1 1\n0 1\nnope\n1\n",
                     "graphlab-csr-v1 1 1 1\n0 1\n0\nnope\n"):
            with self.subTest(text=text):
                self.assert_rejected(text)

    def test_trailing_data_and_nonfinite_features(self):
        self.assert_rejected("graphlab-csr-v1 1 1 1\n0 1\n0\n1\ntrailing\n")
        for value in ("nan", "inf", "-inf", "1e999"):
            with self.subTest(value=value):
                self.assert_rejected("graphlab-csr-v1 1 1 1\n0 1\n0\n" + value + "\n")

    def test_generated_fixture_survives_exact_write_load(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csr"
            second = Path(directory) / "second.csr"
            code, generated = self.invoke("generate", "--nodes", 33, "--features", 7,
                                          "--family", "skewed", "--seed", 2026, "--output", first)
            self.assertEqual(code, 0)
            code, loaded = self.invoke("validate", "--input", first)
            self.assertEqual(code, 0)
            code, rewritten = self.invoke("generate", "--input", first, "--output", second)
            self.assertEqual(code, 0)
            self.assertRegex(generated[0]["fingerprint"], "^[0-9a-f]{16}$")
            self.assertEqual(generated[0]["fingerprint"], loaded[0]["fingerprint"])
            self.assertEqual(generated[0]["fingerprint"], rewritten[0]["fingerprint"])
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_subnormal_and_signed_zero_fixture_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "tiny.csr"
            second = Path(directory) / "tiny-copy.csr"
            first.write_text("graphlab-csr-v1\n3 1 3\n0 1 2 3\n0 1 2\n1.40129846e-45 -0 1.17549435e-38\n", encoding="utf-8")
            code, before = self.invoke("validate", "--input", first)
            self.assertEqual(code, 0)
            code, saved = self.invoke("generate", "--input", first, "--output", second)
            self.assertEqual(code, 0)
            code, after = self.invoke("validate", "--input", second)
            self.assertEqual(code, 0)
            self.assertEqual(before[0]["fingerprint"], saved[0]["fingerprint"])
            self.assertEqual(before[0]["fingerprint"], after[0]["fingerprint"])

    def test_unknown_backend_is_rejected(self):
        for command in ("bench", "verify"):
            code, records = self.invoke(command, "--backend", "unknown")
            self.assertNotEqual(code, 0)
            self.assertEqual(records[0]["type"], "error")
            self.assertIn("backend", records[0]["message"])

    def test_group_offset_overflow_is_rejected(self):
        code, records = self.invoke("bench", "--backend", "cpu", "--nodes", 1, "--features", 1,
                                    "--smoke", "--calls", 1, "--warmup", 0, "--groups", 2,
                                    "--group-offset", 9223372036854775807)
        self.assertNotEqual(code, 0)
        self.assertEqual(records[0]["type"], "error")
        self.assertIn("repetition", records[0]["message"])

    def test_maximum_safe_group_offset_remains_valid(self):
        code, records = self.invoke("bench", "--backend", "cpu", "--nodes", 1, "--features", 1,
                                    "--smoke", "--calls", 1, "--warmup", 0, "--groups", 1,
                                    "--group-offset", 9223372036854775807)
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["group"], 9223372036854775807)
        self.assertEqual(records[0]["status"], "passed")


if __name__ == "__main__":
    unittest.main()

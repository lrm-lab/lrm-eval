import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "framework" / "run.py"


class GatewayThinkingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        task = root / "tasks" / "example"
        task.mkdir(parents=True)
        (task / "task.toml").touch()
        config = root / "config.toml"
        config.write_text('[datasets.demo]\nname = "demo"\npath = "tasks"\n')
        self.command = [sys.executable, str(RUNNER), "demo", "--config", str(config),
                        "--all", "--batch", "test", "--agent", "pi", "--model", "example"]

    def invoke(self, *args):
        return subprocess.run([*self.command, *args], env={}, capture_output=True, text=True)

    def test_explicit_thinking_rejected_before_gateway_dispatch(self):
        for args in [("--gateway-capture", "--thinking", "high"),
                     ("--thinking=medium", "--gateway-capture", "--plan")]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                message = result.stderr.splitlines()[-1]
                self.assertIn("--thinking", message)
                self.assertIn("--gateway-capture", message)

    def test_gateway_plan_without_override(self):
        result = self.invoke("--gateway-capture", "--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["runner"], "openenv-gateway")

    def test_native_plan_reasoning_effort(self):
        for args, effort in [((), "medium"), (("--thinking", "low"), "low")]:
            with self.subTest(args=args):
                result = self.invoke("--plan", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"reasoning_effort={effort}", shlex.split(result.stdout))

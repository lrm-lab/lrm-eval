import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SELECT = Path(__file__).resolve().parents[1] / "framework" / "synthesize.py"


class PartialRewardSelection(unittest.TestCase):
    def select(self, *options):
        groups = {
            "partial": [0.25, 0.75], "edge": [0.5, 0.75], "flat": [0.5, 0.5],
            "fail": [0, 0], "pass": [1, 1], "mixed": [0, 1],
            "invalid": [True, 2], "missing": [0.5],
        }
        records = [
            dict(task_id=key, trial_id=f"{key}-{i}", model_route="candidate",
                 harness="pi", harness_version="0.84.3", dataset="example-train",
                 status="scored", reward=v, task_checksum=key)
            for key, values in groups.items() for i, v in enumerate(values)
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episodes.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            result = subprocess.run([sys.executable, "-B", str(SELECT), "select", str(path),
                                     "--attempts", "2", "--harness", "pi", *options],
                                    check=True, capture_output=True, text=True)
        return {t["task_id"]: t for t in json.loads(result.stdout)["tasks"]}

    def test_continuous_and_noise_floor(self):
        tasks = self.select("--reward-mode", "continuous", "--min-reward-spread", "0.25")
        expected = {"partial": "grpo", "edge": "review", "flat": "review", "fail": "teacher",
                    "pass": "regression", "mixed": "grpo", "invalid": "review", "missing": "review"}
        self.assertEqual({k: d["lane"] for k, d in tasks.items()}, expected)
        self.assertEqual(tasks["partial"]["min_reward"], 0.25)
        self.assertIsNone(tasks["invalid"]["max_reward"])

    def test_default_is_binary(self):
        tasks = self.select()
        self.assertEqual(tasks["partial"]["lane"], "review")
        self.assertEqual(tasks["partial"]["excluded"], {"invalid_reward": 2})
        self.assertEqual(tasks["mixed"]["lane"], "grpo")

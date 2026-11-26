import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from training_grpo import HarborReward, check_weight, recording_factory


def score_in_worker(factory, score):
    # Mirror the pinned trainer's spawn + per-session thread pool. A factory and
    # callback which carry a closure, an open file, or a thread lock cannot spawn.
    class SessionFactory:
        def __init__(self):
            self.model = "student"

        def create(self, task, seed=None, episode_id=None):
            return SimpleNamespace(_task_index=3, _model=self.model)

    modules = {
        "harbor_env.harness": SimpleNamespace(HarborSessionFactory=SessionFactory),
        "async_grpo_harbor": SimpleNamespace(tool_efficiency=lambda n: 1 - n / 15),
        "trl.experimental.async_grpo.openenv_harness": SimpleNamespace(CaptureContractError=ValueError),
    }
    with patch.dict(sys.modules, modules):
        sessions = factory()

        def run(item):
            index, raw = item
            sessions.create("task", seed=index // 2, episode_id=str(index))
            return score(SimpleNamespace(env_reward=raw, tool_call_count=3))

        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            return list(pool.map(run, enumerate([1, 0, 0.5, None])))


class GrpoTest(unittest.TestCase):
    def test_reward_records_follow_rollouts_across_spawn(self):
        with tempfile.TemporaryDirectory() as root, concurrent.futures.ProcessPoolExecutor(
            1, mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            for weight in (0.0, 0.3):
                path = Path(root) / f"{weight}.jsonl"
                score = HarborReward(path, "tools", "revision", weight)
                rewards = pool.submit(score_in_worker, recording_factory, score).result(timeout=30)
                self.assertEqual(rewards, [1 + 0.8 * weight, 0, 0.5, None])
                rows = sorted(map(json.loads, path.read_text().splitlines()), key=lambda row: row["rollout_id"])
                self.assertEqual([r["group_id"] for r in rows], [0, 0, 1, 1])
                self.assertEqual([r["verifier_reward"] for r in rows], [1, 0, 0.5, None])
                self.assertEqual([r["tool_bonus"] for r in rows], [0.8 * weight, 0, 0, None])
                self.assertEqual([r["training_reward"] for r in rows], rewards)

    def test_invalid_weight(self):
        for weight in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                check_weight(weight)

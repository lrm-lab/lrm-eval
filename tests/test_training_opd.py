from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from training_grpo import verify_grpo_update
from training_opd import Rollout, TokenTurn
from training_opd_models import Student

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch, "OPD reference optimizer requires torch")
class PolicyCheckTest(unittest.TestCase):
    def setUp(self):
        self.previous = torch.nn.Linear(1, 1, bias=False)
        current = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.previous.weight.fill_(-0.5)
            current.weight.fill_(-0.8)
        current.device = torch.device("cpu")
        self.student = Student.__new__(Student)
        self.student.model = current
        self.student.torch = torch
        self.student.options = {}
        self.student.temperature = 1
        self.student.tolerance = 0.01
        self.student.total_tokens = 1
        self.checkpoint = Path("checkpoint-0000")
        self.enterContext(patch("training_opd_models.load_model", return_value=self.previous))
        self.enterContext(patch("training_opd_models.sampled_logprobs",
                                side_effect=lambda model, turn, temp: model.weight.flatten()))

    def rollout(self, logprob):
        turn = TokenTurn((1,), (2, 3), (logprob, -10), (1, 0))
        return Rollout(Path("checkpoint-0001"), 0.0, (turn,))

    def test_current_policy_matches_and_previous_is_excluded(self):
        sample = self.rollout(-0.8)
        evidence = self.student.verify_rollouts(self.checkpoint, [sample])
        self.assertEqual(evidence["changed_parameter"], "weight")
        self.assertGreater(evidence["previous_max_error_per_rollout"][0], self.student.tolerance)
        self.student.accumulate(sample.turns[0], [-0.3])
        self.assertLess(self.student.last_logprob_error, self.student.tolerance)

    def test_one_stale_rollout_in_mixed_batch(self):
        with self.assertRaisesRegex(ValueError, "cannot distinguish"):
            self.student.verify_rollouts(self.checkpoint, [self.rollout(-0.8), self.rollout(-0.5)])

    def test_rollout_matches_neither_checkpoint(self):
        sample = self.rollout(-0.1)
        self.student.verify_rollouts(self.checkpoint, [sample])
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.student.accumulate(sample.turns[0], [-0.3])

    def test_identical_parameters(self):
        self.previous.load_state_dict(self.student.model.state_dict())
        with self.assertRaisesRegex(ValueError, "No student parameter changed"):
            self.student.verify_rollouts(self.checkpoint, [self.rollout(-0.8)])

    def test_grpo_probe_uses_live_endpoint_and_final_weights(self):
        client = Mock()
        task = SimpleNamespace(dataset="tools", task_id="id-7")
        client.get_task.return_value = task
        client.run_rollout.return_value = SimpleNamespace(dataset="tools", task_id="id-7", reward=0,
                                                          trial_name="trial", session_id="session", task_name="task")
        self.enterContext(patch.dict(sys.modules, {"openenv.harbor.client": SimpleNamespace(HarborEnv=lambda **kw: client)}))
        capture = self.enterContext(patch("training_opd_openenv.captured_turns"))
        output = Path("final")
        self.enterContext(patch("training_opd_models.load_model",
                                side_effect=lambda path, options, device: self.student.model if path == output
                                else self.previous))
        args = SimpleNamespace(server="harbor", split="tools", harness="opencode", sandbox="docker", reward_key="",
                               vllm_url="current-server", model="base", agent_timeout=60, agent_step_limit=4,
                               temperature=1)
        capture.return_value = self.rollout(-0.8).turns
        evidence = verify_grpo_update(args, output, 7, 0.01)
        self.assertEqual(evidence["trial_name"], "trial")
        self.assertEqual(evidence["tokens"], 1)
        self.assertEqual(client.run_rollout.call_args.kwargs["llm_url"], args.vllm_url)
        self.assertEqual(client.run_rollout.call_args.kwargs["task_index"], 7)
        client.close.assert_called_once()
        capture.return_value = self.rollout(-0.5).turns
        with self.assertRaisesRegex(ValueError, "differs from final"):
            verify_grpo_update(args, output, 7, 0.01)

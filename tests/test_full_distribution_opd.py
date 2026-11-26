from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from training_opd import TokenTurn
from training_opd_models import ExpertDistribution, FullDistributionExpert, Student

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch, "requires torch")
class ExactDistillationTest(unittest.TestCase):
    def setUp(self):
        class TokenTable(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self, weights):
                super().__init__()
                self.weights = torch.nn.Parameter(weights)

            def forward(self, input_ids, use_cache):
                return SimpleNamespace(logits=self.weights[input_ids])

        torch.manual_seed(107)
        self.student = Student.__new__(Student)
        self.student.torch = torch
        self.student.model = TokenTable(torch.randn(4, 4))
        self.student.temperature = 0.8
        self.student.total_tokens = 5
        self.student.tolerance = 1e-5
        self.expert = FullDistributionExpert.__new__(FullDistributionExpert)
        self.expert.model = TokenTable(torch.randn(4, 4))
        self.expert.temperature = 1.2
        # Only input token 1 -> target 2 and input token 3 -> target 0 contribute.
        with torch.no_grad():
            logp = (self.student.model.weights / self.student.temperature).log_softmax(-1)
        self.turn = TokenTurn((0, 1), (2, 3, 0), (logp[1, 2].item(), -10, logp[3, 0].item()), (1, 0, 1))

    def test_gradient_matches_conditional_kl_and_preserves_sampled_metric(self):
        targets = self.expert.score(self.turn)
        self.assertEqual(targets.logps.shape, (2, 4))
        self.assertFalse(targets.logps.requires_grad)
        observed = self.student.accumulate(self.turn, targets)
        weights = self.student.model.weights.detach().clone().requires_grad_()
        student = (weights[[1, 3]] / self.student.temperature).softmax(-1)
        teacher = (self.expert.model.weights[[1, 3]].detach() / self.expert.temperature).softmax(-1)
        kl = torch.distributions.kl_divergence(torch.distributions.Categorical(student),
                                              torch.distributions.Categorical(teacher)).sum()
        (kl / self.student.total_tokens).backward()
        self.assertTrue(torch.allclose(weights.grad, self.student.model.weights.grad, atol=1e-6))
        self.assertAlmostEqual(self.student.last_exact_kl_sum, kl.item(), places=5)
        expected = (student.log() - teacher.log())[[0, 1], [2, 0]].sum().item()
        self.assertAlmostEqual(observed, expected, places=5)
        self.assertIsNone(self.expert.model.weights.grad)

    def test_rejects_stale_capture_and_misaligned_distribution(self):
        targets = self.expert.score(self.turn)
        stale = TokenTurn(self.turn.prompt, self.turn.completion, (-20, -10, -20), self.turn.loss_mask)
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.student.accumulate(stale, targets)
        with self.assertRaisesRegex(ValueError, "expert logprobs"):
            self.student.accumulate(self.turn, ExpertDistribution(targets.logps[:1]))
        self.assertIsNone(self.student.model.weights.grad)

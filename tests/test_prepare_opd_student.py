import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))
from prepare_opd_student import load_checkpoint, prepare

try:
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
except ImportError:
    torch = None


@unittest.skipUnless(torch, "requires OPD training dependencies")
class PreparingWeights(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.base = self.root / "base"
        self.experts = [self.root / "left", self.root / "right"]
        self.output = self.root / "student"
        config = GPT2Config(vocab_size=4, n_positions=4, n_embd=4, n_layer=1, n_head=1)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]")), unk_token="[UNK]")
        for path, seed in zip([self.base, *self.experts], [1, 2, 3]):
            torch.manual_seed(seed)
            model = GPT2LMHeadModel(config).bfloat16()
            model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def test_exact_mix_preserves_dtype_tying_and_sources(self):
        prepare(self.base, self.experts, self.output)
        model = load_checkpoint(self.output)
        snapshots = [dict(load_checkpoint(path).named_parameters()) for path in [self.base, *self.experts]]
        for name, param in model.named_parameters():
            expected = snapshots[0][name].float().mul(0.5)
            for expert in snapshots[1:]:
                expected.add_(expert[name].float(), alpha=0.25)
            self.assertEqual(param.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(param, expected.bfloat16()), name)
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)
        self.assertEqual(self.tokenizer.get_vocab(),
                         PreTrainedTokenizerFast.from_pretrained(self.output).get_vocab())
        record = json.loads((self.output / "opd-init.json").read_text())
        self.assertEqual(record["experts"], [{"path": str(path), "weight": 0.25} for path in self.experts])
        with self.assertRaises(FileExistsError):
            prepare(self.base, self.experts, self.output)

    def test_rejects_reordered_token_ids(self):
        different = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {"[UNK]": 0, "a": 2, "b": 1, "c": 3}, unk_token="[UNK]")), unk_token="[UNK]")
        different.save_pretrained(self.experts[1])
        with self.assertRaisesRegex(ValueError, "token IDs"):
            prepare(self.base, self.experts, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_different_attention_layout_with_identical_shapes(self):
        config = GPT2Config.from_pretrained(self.experts[0])
        config.n_head = 2
        config.save_pretrained(self.experts[0])
        with self.assertRaisesRegex(ValueError, "architecture"):
            prepare(self.base, self.experts, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_missing_expert_tensor(self):
        path = self.experts[0] / "model.safetensors"
        tensors = load_file(path)
        tensors.pop("transformer.ln_f.bias")
        save_file(tensors, path, metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "missing_keys"):
            prepare(self.base, self.experts, self.output)
        self.assertFalse(self.output.exists())

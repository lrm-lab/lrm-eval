"""Transformers optimizer and exact-token expert scorers for Tianxi OPD."""

import json
import math
import os
from urllib.request import Request, urlopen


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def load_model(model, options, device):
    import torch
    from transformers import AutoModelForCausalLM

    arguments = {"revision": options["revision"]} if "revision" in options else {}
    dtype = options.get("dtype")
    if dtype:
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported model dtype: {dtype}")
        arguments["dtype"] = getattr(torch, dtype)
    return AutoModelForCausalLM.from_pretrained(model, **arguments).to(device).eval()


def sampled_logprobs(model, turn, temperature):
    import torch

    token_ids = torch.tensor([turn.prompt + turn.completion], device=model.device)
    # Logits at position j predict the token at position j+1.
    positions = [len(turn.prompt) + i - 1 for i, bit in enumerate(turn.loss_mask) if bit]
    logits = model(input_ids=token_ids[:, :-1], use_cache=False).logits[0, positions].float()
    selected = token_ids[0, [index + 1 for index in positions]]
    scaled = logits / temperature
    return scaled.gather(1, selected[:, None]).squeeze(1) - scaled.logsumexp(dim=1)


def behavior_logprob_error(scores, turn):
    import torch

    if not torch.isfinite(scores).all():
        raise ValueError("Checkpoint produced non-finite logprobs")
    behavior = scores.new_tensor([p for p, bit in zip(turn.behavior_logps, turn.loss_mask) if bit])
    return (scores.detach() - behavior).abs().max().item()


def check_previous_policy(model, previous_checkpoint, options, samples, temperature, tolerance):
    """Test whether fresh behavior rules out the previous weights at the configured tolerance."""
    import torch

    previous = load_model(previous_checkpoint, options, model.device)
    old_parameters = dict(previous.named_parameters())
    changed_name = next((name for name, parameter in model.named_parameters()
                         if not torch.equal(old_parameters[name], parameter)), None)
    if changed_name is None:
        raise ValueError(f"No student parameter changed since {previous_checkpoint}")
    old_errors = []
    with torch.no_grad():
        for sample in samples:
            old_error = 0.0
            for turn in sample.turns:
                old = sampled_logprobs(previous, turn, temperature)
                old_error = max(old_error, behavior_logprob_error(old, turn))
            old_errors.append(old_error)
    # Both callers also check the current weights against every supervised token.
    if min(old_errors) <= tolerance:
        raise ValueError("A rollout cannot distinguish the two checkpoints at logprob_tolerance")
    return {"previous_checkpoint": str(previous_checkpoint), "changed_parameter": changed_name,
            "previous_max_error_per_rollout": old_errors, "logprob_tolerance": tolerance}


class Student:
    """Single-device reference optimizer. Each sampled batch drives one update."""

    def __init__(self, args, settings):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        torch.manual_seed(args.seed)
        options = settings.get("student", {})
        self.temperature = positive(settings.get("temperature", 1), "OPD temperature")
        self.tolerance = positive(options.get("logprob_tolerance", 0.2), "student logprob_tolerance")
        self.max_grad_norm = positive(options.get("max_grad_norm", 1), "student max_grad_norm")
        device = options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(args.student, revision=options.get("revision"))
        self.options = options
        self.model = load_model(args.student, options, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=positive(options.get("learning_rate", 1e-5), "student learning_rate"),
            weight_decay=0,
        )

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def begin(self, total_tokens):
        self.total_tokens = total_tokens
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate(self, turn, targets):
        current = sampled_logprobs(self.model, turn, self.temperature)
        behavior = current.new_tensor([logp for logp, bit in zip(turn.behavior_logps, turn.loss_mask) if bit])
        teacher = current.new_tensor(targets)
        if teacher.shape != current.shape or not self.torch.isfinite(teacher).all() or (teacher > 0).any():
            raise ValueError("Expert must return one finite log-probability for every unmasked sampled token")
        self.last_logprob_error = (current.detach() - behavior).abs().max().item()
        if not self.torch.isfinite(current).all() or self.last_logprob_error > self.tolerance:
            raise ValueError("Rollout logprobs disagree with the current student checkpoint or sampling policy")
        # A token-level score-function gradient for KL(student || expert), with the
        # behavior logprob and expert logprob held fixed through the update.
        difference = behavior - teacher
        loss = (current * difference).sum() / self.total_tokens
        loss.backward()
        return difference.sum().item()

    def verify_rollouts(self, previous_checkpoint, samples):
        return check_previous_policy(self.model, previous_checkpoint, self.options, samples,
                                     self.temperature, self.tolerance)

    def step(self):
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()


class Expert:
    """Teacher-forced scoring with a local Hugging Face model."""

    def __init__(self, definition, args, settings, student):
        del args, settings
        from transformers import AutoTokenizer

        self.definition = definition
        self.temperature = positive(definition.get("temperature", 1), "expert temperature")
        self.device = definition.get("device", "cpu")
        tokenizer = AutoTokenizer.from_pretrained(definition["model"], revision=definition.get("revision"))
        if tokenizer.get_vocab() != student.tokenizer.get_vocab():
            raise ValueError(f"Expert {definition['model']} must share the student's token IDs")
        self.vocab_size = student.model.get_output_embeddings().weight.shape[0]
        self.model = None

    def score(self, turn):
        import torch

        if self.model is None:
            self.model = load_model(self.definition["model"], self.definition, self.device)
            if self.model.get_output_embeddings().weight.shape[0] != self.vocab_size:
                raise ValueError("Expert and student must share the same output vocabulary size")
        with torch.no_grad():
            return sampled_logprobs(self.model, turn, self.temperature).tolist()


class VLLMExpert:
    """Read token-ID prompt logprobs from an independently served vLLM expert."""

    def __init__(self, definition, args, settings, student):
        del args, settings
        from transformers import AutoTokenizer

        self.model = definition["model"]
        self.endpoint = definition["url"].rstrip("/") + "/completions"
        self.timeout = positive(definition.get("timeout_sec", 180), "expert timeout_sec")
        tokenizer = AutoTokenizer.from_pretrained(definition["tokenizer"], revision=definition.get("revision"))
        if tokenizer.get_vocab() != student.tokenizer.get_vocab():
            raise ValueError(f"Expert {self.model} must share the student's token IDs")
        self.headers = {"Content-Type": "application/json"}
        if key_env := definition.get("api_key_env"):
            self.headers["Authorization"] = f"Bearer {os.environ[key_env]}"
        if definition.get("temperature", 1) != 1:
            raise ValueError("vLLM prompt scoring requires expert temperature 1")

    def score(self, turn):
        token_ids = list(turn.prompt + turn.completion)
        request = Request(self.endpoint, method="POST", headers=self.headers, data=json.dumps({
            "model": self.model, "prompt": token_ids, "max_tokens": 0, "echo": True,
            "prompt_logprobs": 0, "return_token_ids": True, "temperature": 1,
            "add_special_tokens": False,
        }).encode())
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        try:
            choice, = result["choices"]
            scores = choice["prompt_logprobs"]
            if choice["prompt_token_ids"] != token_ids or len(scores) != len(token_ids):
                raise ValueError("vLLM expert returned a different token sequence")
            return [scores[i][str(token_ids[i])]["logprob"]
                    for i, selected in enumerate(turn.loss_mask, len(turn.prompt)) if selected]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("vLLM expert response is missing exact-token logprobs") from error

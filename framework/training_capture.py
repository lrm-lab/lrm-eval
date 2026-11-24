"""Validation shared by training capture publication and readiness checks."""

import hashlib
import json
import math


def load_training_capture(directory, trial_id):
    """Read a trial-bound OpenEnv capture and check the arrays used for training."""
    capture = directory / trial_id
    evidence = json.loads((capture / "capture.json").read_text())
    trace_bytes = (capture / "training_trace.json").read_bytes()
    if (evidence.get("trial_id") != trial_id or evidence.get("purpose") != "train"
            or not isinstance(evidence.get("checkpoint"), str) or not evidence["checkpoint"]
            or evidence.get("trace_sha256") != hashlib.sha256(trace_bytes).hexdigest()):
        raise ValueError(f"Training capture provenance does not match trial {trial_id}")
    trace = json.loads(trace_bytes)
    turns = trace.get("turns")
    if trace.get("schema_version", 1) != 1 or not isinstance(turns, list) or not turns:
        raise ValueError(f"Missing training turns for {trial_id}")
    nodes = set()
    supervised = 0
    for turn in turns:
        node = turn["node_id"]
        prompt, completion = turn["prompt_token_ids"], turn["completion_token_ids"]
        logprobs, mask = turn["per_token_logps"], turn["loss_mask"]
        if (not isinstance(node, str) or not node or node in nodes
                or not all(isinstance(values, list) for values in (prompt, completion, logprobs, mask))
                or len(completion) != len(logprobs) or len(mask) != len(prompt) + len(completion)
                or any(type(token) is not int or token < 0 for token in prompt + completion)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in logprobs)
                or any(type(value) is not int or value not in (0, 1) for value in mask)
                or any(mask[:len(prompt)])):
            raise ValueError(f"Misaligned training turn for {trial_id}")
        nodes.add(node)
        supervised += sum(mask)
    if not supervised:
        raise ValueError(f"No supervised tokens for {trial_id}")
    return evidence, trace_bytes

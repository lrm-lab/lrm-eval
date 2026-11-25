"""Run the pinned TRL Harbor example with per-rollout reward accounting."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from uuid import uuid4


PINNED_TRL = "b2686bf5c300756cdf878b84f83a48f794e2fb94"
RECIPE_DIR = Path("examples/async_grpo_harbor")


def check_recipe(checkout, revision):
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("TRL revision must be a full Git commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if head != revision:
        raise ValueError(f"TRL checkout is at {head}, expected {revision}")
    if not (checkout / RECIPE_DIR / "async_grpo_harbor.py").is_file():
        raise ValueError(f"Missing Harbor recipe in {checkout}")
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(RECIPE_DIR), "trl"], cwd=checkout, text=True,
    )
    if changes:
        raise ValueError("Local changes in the TRL package or Harbor recipe differ from the selected revision")


def check_weight(weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("GRPO tool efficiency weight must be finite and nonnegative")


def check_tolerance(tolerance):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("GRPO update verification requires a finite positive logprob tolerance")


# The trainer pickles its factory and reward callback into a spawned worker.
# In the pinned recipe session.create() and scoring run on the same pool thread.
_episode = threading.local()
_write_lock = threading.Lock()


def recording_factory(*args, **kwargs):
    from harbor_env.harness import HarborSessionFactory

    class RecordingFactory(HarborSessionFactory):
        def create(self, task, seed=None, episode_id=None):
            session = super().create(task, seed=seed, episode_id=episode_id)
            _episode.record = {"rollout_id": episode_id, "group_id": seed,
                               "task_index": session._task_index, "requested_model": session._model}
            return session

    return RecordingFactory(*args, **kwargs)


@dataclass(frozen=True)
class HarborReward:
    path: Path
    split: str
    revision: str
    weight: float

    def __call__(self, outcome):
        from async_grpo_harbor import tool_efficiency
        from trl.experimental.async_grpo.openenv_harness import CaptureContractError

        raw = outcome.env_reward
        efficiency = tool_efficiency(outcome.tool_call_count)
        bonus = self.weight * efficiency if raw is not None and raw >= 1 and efficiency is not None else 0.0
        reward = None if raw is None else raw + bonus
        record = {
            **_episode.record, "split": self.split, "trl_revision": self.revision,
            "verifier_reward": raw, "tool_calls": outcome.tool_call_count, "tool_efficiency": efficiency,
            "tool_efficiency_weight": self.weight, "tool_bonus": None if raw is None else bonus,
            "training_reward": reward,
        }
        try:
            with _write_lock, self.path.open("a", encoding="utf-8") as evidence:
                evidence.write(json.dumps(record, allow_nan=False) + "\n")
        except (OSError, ValueError) as error:
            # TRL would otherwise turn logging and invalid-score errors into silent skips.
            raise CaptureContractError(f"Could not record GRPO reward: {error}") from error
        return reward


def verify_grpo_update(args, output, task_index, tolerance):
    import torch
    from openenv.harbor.client import HarborEnv
    from training_opd import Rollout
    from training_opd_models import behavior_logprob_error, check_previous_policy, load_model, sampled_logprobs
    from training_opd_openenv import captured_turns

    client = HarborEnv(base_url=args.server)
    try:
        task = client.get_task(args.split, task_index)
        result = client.run_rollout(
            split=args.split, task_index=task_index, harness=args.harness, sandbox=args.sandbox,
            reward_key=args.reward_key, llm_url=args.vllm_url, model=args.model, purpose="train",
            agent_timeout_sec=args.agent_timeout, agent_step_limit=args.agent_step_limit,
            sampling={"temperature": args.temperature, "top_p": 1.0, "top_k": -1,
                      "min_p": 0.0, "repetition_penalty": 1.0},
        )
    finally:
        client.close()
    if (result.dataset, result.task_id) != (task.dataset, task.task_id):
        raise ValueError("GRPO verification rollout returned a different task")
    turns = captured_turns(result, args.temperature)
    sample = Rollout(output, result.reward, turns)
    # Run this opt-in small-model acceptance check on CPU so it does not contend
    # with the still-live vLLM service. Match the pinned recipe's training dtype.
    options = {"dtype": "bfloat16"}
    current = load_model(output, options, "cpu")
    with torch.no_grad():
        current_error = max(behavior_logprob_error(sampled_logprobs(current, turn, args.temperature), turn)
                            for turn in turns)
    if current_error > tolerance:
        raise ValueError(f"GRPO rollout differs from final checkpoint by {current_error} (tolerance {tolerance})")
    check = check_previous_policy(current, args.model, options, [sample], args.temperature, tolerance)
    return {**check, "current_checkpoint": str(output), "current_max_behavior_logprob_error": current_error,
            "requested_model": args.model, "session_id": result.session_id, "trial_name": result.trial_name,
            "dataset": result.dataset, "task": result.task_name, "tokens": sum(turn.tokens for turn in turns)}


def run_recipe(recipe, revision, weight, tolerance):
    # main() owns trainer construction. Keep its CLI and replace only the two
    # callables that it passes to HarnessAsyncGRPOTrainer.
    args = recipe.parse_args()
    if tolerance is not None and args.lora_r:
        raise ValueError("--verify-grpo-update requires full-weight GRPO for the base/final comparison")
    run_id = uuid4().hex
    if args.output_dir:
        output = Path(args.output_dir)
    else:
        run_name = args.run_name or f"{args.model.split('/')[-1]}-{args.harness}-{args.max_steps}steps-{run_id[:8]}"
        output = Path("runs/async_grpo_harbor") / run_name
        sys.argv.extend(["--output-dir", str(output)])
        if not args.run_name:
            sys.argv.extend(["--run-name", run_name])
    output.mkdir(parents=True, exist_ok=True)
    recipe.HarborSessionFactory = recording_factory
    recipe.harbor_reward = HarborReward(output.resolve() / f"grpo-rewards-{run_id}.jsonl", args.split, revision, weight)
    recipe.main()
    if tolerance is not None and int(os.environ.get("RANK", 0)) == 0:
        indices = recipe.task_indices(args.task_indices)
        check = verify_grpo_update(args, output.resolve(), indices[0] if indices else 0, tolerance)
        path = output / f"grpo-policy-{run_id}.json"
        path.write_text(json.dumps({**check, "trl_revision": revision}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tool-efficiency-weight", required=True, type=float)
    parser.add_argument("--verify-grpo-update", type=float)
    boundary = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    options = parser.parse_args(sys.argv[1:boundary])
    del sys.argv[1:boundary + 1]
    check_recipe(options.checkout, options.revision)
    check_weight(options.tool_efficiency_weight)
    if options.verify_grpo_update is not None:
        check_tolerance(options.verify_grpo_update)
    sys.path[:0] = [str(options.checkout), str(options.checkout / RECIPE_DIR)]
    import async_grpo_harbor

    run_recipe(async_grpo_harbor, options.revision, options.tool_efficiency_weight, options.verify_grpo_update)

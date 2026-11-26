"""Route Harbor tasks to experts and update one student on fresh rollouts."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from itertools import cycle
import json
import math
import os
from pathlib import Path
import random
import tomllib


@dataclass(frozen=True)
class Task:
    dataset: str
    task: str
    environment: str
    source: str
    openenv_split: str
    reward_key: str


@dataclass(frozen=True)
class TokenTurn:
    prompt: tuple[int, ...]
    completion: tuple[int, ...]
    behavior_logps: tuple[float, ...]
    loss_mask: tuple[int, ...]

    def __post_init__(self):
        if (not self.prompt or not self.completion
                or len(self.completion) != len(self.behavior_logps)
                or len(self.completion) != len(self.loss_mask)
                or any(type(token) is not int or token < 0 for token in self.prompt + self.completion)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value > 0
                       for value in self.behavior_logps)
                or any(type(bit) is not int or bit not in (0, 1) for bit in self.loss_mask)
                or not any(self.loss_mask)):
            raise ValueError("Invalid OPD token turn")

    @property
    def tokens(self):
        return sum(self.loss_mask)


@dataclass(frozen=True)
class Rollout:
    checkpoint: Path
    reward: float
    turns: tuple[TokenTurn, ...]
    requested_model: str | None = None
    trial_name: str | None = None
    session_id: str | None = None


def resolve(target):
    module, function = target.split(":", 1)
    return getattr(import_module(module), function)


def read_tasks(config, config_dir, aliases, selected, options):
    matches = set()
    seen = set()
    for alias in dict.fromkeys(aliases):
        suite = config["datasets"].get(alias)
        if suite is None or suite.get("runner", "harbor") != "harbor":
            raise ValueError(f"OPD requires a registered Harbor dataset: {alias}")
        path_root = suite.get("path_root", "bundle" if "handoff_profile" in suite else "config")
        if path_root not in ("bundle", "config"):
            raise ValueError(f"Invalid path_root for {alias}")
        root = config_dir
        if path_root == "bundle":
            bundle = os.getenv("TIANXI_BENCH_BUNDLE") or config.get("bundle")
            if not bundle:
                raise ValueError(f"OPD dataset {alias} requires a handoff bundle")
            root /= Path(bundle).expanduser()
        folder = (root / suite["path"]).resolve()
        if (folder / "task.toml").is_file():
            files = [folder / "task.toml"]
        else:
            files = sorted(folder.glob("*/task.toml"))
        if not files:
            raise ValueError(f"No Harbor tasks found for {alias}: {folder}")
        for file in files:
            document = tomllib.loads(file.read_text())
            metadata = document.get("metadata", {})
            task = file.parent.name
            name = document.get("task", {}).get("name", task)
            identifiers = {task, name, f"{alias}/{task}"}
            if selected and not (matched := identifiers & selected):
                continue
            if selected:
                matches.update(matched)
            if (file.resolve() in seen
                    or metadata.get("split", suite.get("split")) != "train"
                    or suite.get("split", "train") != "train"
                    or alias.endswith("-holdout")):
                raise ValueError(f"OPD requires distinct train tasks: {alias}/{task}")
            seen.add(file.resolve())
            source = metadata.get("source", "")
            environment = metadata.get("environment", suite.get("environment"))
            if environment is None:
                environment = "offline-files" if source in ("order-reconcile", "log-rollup") else alias
            split = options.get("splits", {}).get(alias, suite.get("name", alias))
            yield Task(alias, task, environment, source, split, suite.get("reward_key", ""))
    if missing := selected - matches:
        raise ValueError(f"Unknown task selection: {', '.join(sorted(missing))}")


MATCH_FIELDS = ("dataset", "environment", "source", "task")


def route(task, settings):
    """Choose one expert by rule priority, then the number of matched selectors."""
    candidates = []
    for rule in settings.get("routes", []):
        if set(rule) - {*MATCH_FIELDS, "expert", "priority"}:
            raise ValueError(f"Unknown OPD route selector in {rule}")
        selectors = {key: rule[key] for key in MATCH_FIELDS if key in rule}
        if all(getattr(task, key) == value for key, value in selectors.items()):
            priority = rule.get("priority", 0)
            if type(priority) is not int:
                raise ValueError("OPD route priority must be an integer")
            candidates.append(((priority, len(selectors)), rule["expert"]))
    if not candidates:
        raise ValueError(f"No expert route for {task.dataset}/{task.task} ({task.environment})")
    best = max(score for score, _ in candidates)
    experts = {expert for score, expert in candidates if score == best}
    if len(experts) != 1:
        raise ValueError(f"Ambiguous expert routes for {task.dataset}/{task.task}: {', '.join(sorted(experts))}")
    return experts.pop()


def schedule(routed, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for task, expert in routed:
        groups[expert].append((task, expert))
    queues = {}
    for expert, tasks in groups.items():
        rng.shuffle(tasks)
        queues[expert] = cycle(tasks)
    for expert in cycle(list(queues)):
        yield next(queues[expert])


def run(args, settings, routed, routing):
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory

    components = settings.get("components", {})
    build_optimizer = resolve(components.get("optimizer", "training_opd_models:Student"))
    build_rollout = resolve(components.get("rollout", "training_opd_openenv:OpenEnvRollout"))
    build_publisher = resolve(components.get("publisher", "training_opd_openenv:VLLMPublisher"))
    schedule_tasks = resolve(components.get("scheduler", "training_opd:schedule"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "routing.json").write_text(json.dumps(routing, indent=2) + "\n")
    tasks = iter(schedule_tasks(routed, args.seed))

    with ExitStack() as stack:
        optimizer = build_optimizer(args, settings)
        definitions = settings["experts"]
        scorers = {
            expert: resolve(definitions[expert].get("scorer", "training_opd_models:Expert"))(
                definitions[expert], args, settings, optimizer
            )
            for expert in dict.fromkeys(expert for _, expert in routed)
        }
        rollout = stack.enter_context(build_rollout(args, settings))
        publisher = build_publisher(args, settings)

        def save(step, metrics):
            checkpoint = output / f"checkpoint-{step:04d}"
            with TemporaryDirectory(prefix=".checkpoint-", dir=output) as stage:
                path = Path(stage) / "weights"
                optimizer.save(path)
                (path / "opd.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
                path.rename(checkpoint)
            return checkpoint

        if args.verify_update and not callable(getattr(optimizer, "verify_rollouts", None)):
            raise ValueError("--verify-update requires an optimizer supporting verify_rollouts")
        previous_checkpoint = None
        checkpoint = save(0, {"student": args.student})
        batch_size = args.rollouts_per_step if args.rollouts_per_step is not None else len(scorers)
        for step in range(1, args.steps + 1):
            batch = []
            with publisher.serve(checkpoint) as endpoint:
                for _ in range(batch_size):
                    task, expert = next(tasks)
                    sample = rollout.sample(task, checkpoint, endpoint)
                    if sample.checkpoint != checkpoint or not sample.turns or not math.isfinite(sample.reward):
                        raise ValueError(f"Invalid rollout for {task.dataset}/{task.task} at {checkpoint.name}")
                    batch.append((task, expert, sample))

            policy_check = None
            if args.verify_update and previous_checkpoint is not None:
                policy_check = optimizer.verify_rollouts(previous_checkpoint, [sample for _, _, sample in batch])
            optimizer.begin(sum(turn.tokens for _, _, sample in batch for turn in sample.turns))
            evidence = []
            for task, expert, sample in batch:
                reverse_kl = 0.0
                tokens = 0
                logprob_errors = []
                exact_kl_sums = []
                for turn in sample.turns:
                    targets = scorers[expert].score(turn)
                    reverse_kl += optimizer.accumulate(turn, targets)
                    tokens += turn.tokens
                    if getattr(optimizer, "last_exact_kl_sum", None) is not None:
                        exact_kl_sums.append(optimizer.last_exact_kl_sum)
                    if getattr(optimizer, "last_logprob_error", None) is not None:
                        logprob_errors.append(optimizer.last_logprob_error)
                evidence.append({"dataset": task.dataset, "task": task.task, "expert": expert,
                                 "requested_model": getattr(sample, "requested_model", None),
                                 "trial_name": getattr(sample, "trial_name", None),
                                 "session_id": getattr(sample, "session_id", None),
                                 "max_behavior_logprob_error": max(logprob_errors, default=None),
                                 "reward": sample.reward, "tokens": tokens,
                                 "sampled_reverse_kl": reverse_kl / tokens})
                if len(exact_kl_sums) == len(sample.turns):
                    evidence[-1]["exact_reverse_kl"] = sum(exact_kl_sums) / tokens
            optimizer.step()
            previous_checkpoint = checkpoint
            metrics = {"step": step, "rollout_checkpoint": checkpoint.name, "rollouts": evidence}
            if policy_check is not None:
                metrics["policy_check"] = policy_check
            checkpoint = save(step, metrics)
            print(f"OPD step {step}: {checkpoint}", flush=True)


def launch(config, config_dir, argv, *, plan=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opd-config", type=Path, help="Optional TOML with top-level [opd] settings")
    parser.add_argument("--dataset", action="append", required=True, help="Registered training dataset. Repeat to mix.")
    parser.add_argument("--task", action="append", default=[], help="Task ID or dataset/task ID. Repeat to mix.")
    parser.add_argument("--student")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--rollouts-per-step", type=int)
    parser.add_argument("--verify-update", action="store_true",
                        help="Verify new rollouts against both current and previous student weights (2+ steps)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.steps < 1 or (args.rollouts_per_step is not None and args.rollouts_per_step < 1):
        parser.error("Steps and rollouts per step must be positive")
    if args.verify_update and args.steps < 2:
        parser.error("--verify-update needs at least two rounds of sampling")
    settings = config.get("training", {}).get("opd", {})
    if args.opd_config:
        settings = {**settings, **tomllib.loads(args.opd_config.read_text())["opd"]}
    router = resolve(settings.get("components", {}).get("router", "training_opd:route"))
    experts = settings.get("experts", {})
    routed = []
    routing = []
    for task in read_tasks(config, config_dir, args.dataset, set(args.task), settings):
        expert = router(task, settings)
        model = experts.get(expert, {}).get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Configure a model for OPD expert {expert}")
        routed.append((task, expert))
        routing.append({"dataset": task.dataset, "task": task.task, "environment": task.environment,
                        "source": task.source, "expert": expert, "model": model})
    if not routed:
        raise ValueError("No OPD tasks selected")
    if plan:
        print(json.dumps({"tasks": routing}, indent=2))
        return 0
    if args.student is None or args.output is None:
        parser.error("Set --student and --output to start OPD")
    run(args, settings, routed, routing)
    return 0

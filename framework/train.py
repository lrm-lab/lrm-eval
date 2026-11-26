#!/usr/bin/env python3
"""Launch the configured trainer with its native arguments after --."""

import argparse
from importlib import import_module
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib


def main():
    launcher_args, trainer_args = sys.argv[1:], []
    if "--" in launcher_args:
        boundary = launcher_args.index("--")
        launcher_args, trainer_args = launcher_args[:boundary], launcher_args[boundary + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("harbor.toml"))
    parser.add_argument("--backend", help="trl, xtuner, or a backend registered in training.backends")
    parser.add_argument("--trl-root", type=Path, help="Local TRL checkout with the official Harbor example")
    parser.add_argument("--trl-revision", help="Expected full Git commit for the Harbor recipe")
    parser.add_argument("--tool-efficiency-weight", type=float, help="GRPO bonus weight for solved tasks")
    parser.add_argument("--verify-grpo-update", type=float, metavar="TOLERANCE",
                        help="After full-weight GRPO, verify a fresh Harbor rollout against final and initial weights")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(launcher_args)
    config = tomllib.loads(args.config.read_text())
    settings = config.get("training", {})
    backend = args.backend or settings.get("backend", "trl")
    registered = {"opd": "training_opd:launch", **settings.get("backends", {})}
    prefix = []
    trl_options = (args.trl_revision, args.tool_efficiency_weight, args.verify_grpo_update)
    if backend != "trl" and any(option is not None for option in trl_options):
        parser.error("TRL launcher options require --backend trl")
    if backend == "trl":
        from training_grpo import PINNED_TRL, check_recipe, check_weight, check_tolerance

        if args.trl_root is None:
            parser.error("Set --trl-root to the TRL checkout")
        checkout = args.trl_root.resolve()
        revision = args.trl_revision or settings.get("trl_revision", PINNED_TRL)
        weight = args.tool_efficiency_weight
        if weight is None:
            weight = settings.get("tool_efficiency_weight", 0.0)
        try:
            check_recipe(checkout, revision)
            check_weight(weight)
            if args.verify_grpo_update is not None:
                check_tolerance(args.verify_grpo_update)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            parser.error(str(error))
        script = Path(__file__).with_name("training_grpo.py").resolve()
        prefix = ["--checkout", str(checkout), "--revision", revision, "--tool-efficiency-weight", str(weight)]
        if args.verify_grpo_update is not None:
            prefix.extend(["--verify-grpo-update", str(args.verify_grpo_update)])
        prefix.append("--")
        harness = settings.get("harness")
        has_harness = any(flag == "--harness" or flag.startswith("--harness=") for flag in trainer_args)
        if harness and not has_harness:
            trainer_args.extend(["--harness", harness])
    elif backend == "xtuner":
        if settings.get("xtuner_enabled") is not True:
            parser.error("XTuner is disabled by training.xtuner_enabled")
        if not settings.get("xtuner_entrypoint"):
            parser.error("Configure training.xtuner_entrypoint with a working OpenEnv/XTuner adapter")
        script = (args.config.resolve().parent / settings["xtuner_entrypoint"]).resolve()
        checkout = script.parent
    elif backend in registered:
        try:
            module, name = registered[backend].split(":", 1)
            launch = getattr(import_module(module), name)
            return launch(config, args.config.resolve().parent, trainer_args, plan=args.plan)
        except (OSError, ValueError, RuntimeError, KeyError, AttributeError, ImportError, subprocess.SubprocessError) as error:
            parser.error(str(error))
    else:
        parser.error(f"Unknown training backend: {backend}")
    if not script.is_file():
        parser.error(f"Missing trainer entrypoint: {script}")
    command = [sys.executable, str(script), *prefix, *trainer_args]
    if args.plan:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=checkout).returncode


if __name__ == "__main__":
    raise SystemExit(main())

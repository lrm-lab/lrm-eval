#!/usr/bin/env python3
"""Launch the configured trainer with its native arguments after --."""

import argparse
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
    parser.add_argument("--backend", choices=("trl", "xtuner"))
    parser.add_argument("--trl-root", type=Path, help="Local TRL checkout with the official Harbor example")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(launcher_args)
    settings = tomllib.loads(args.config.read_text()).get("training", {})
    backend = args.backend or settings.get("backend", "trl")
    if backend == "trl":
        if args.trl_root is None:
            parser.error("Set --trl-root to the TRL checkout")
        checkout = args.trl_root.resolve()
        script = checkout / "examples/async_grpo_harbor/async_grpo_harbor.py"
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
    else:
        parser.error(f"Unknown training backend: {backend}")
    if not script.is_file():
        parser.error(f"Missing trainer entrypoint: {script}")
    command = [sys.executable, str(script), *trainer_args]
    if args.plan:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=checkout).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate, validate, and select synthetic Harbor tasks."""

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from synthesis_methods import Case, csv_text

HERE = Path(__file__).resolve().parent
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DOCKERFILE = "FROM python:3.11-slim\nWORKDIR /app\nCOPY inputs/ /app/inputs/\n"
ORACLE = "#!/bin/sh\nset -eu\npython3 /solution/solve.py\n"
VERIFY = "#!/bin/sh\nset -eu\npython3 /tests/verify.py /app /tests/expected.json /logs/verifier/reward.txt\n"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def task_toml(task_id, method, split, environment):
    return f"""schema_version = "1.1"

[task]
name = "tianxi-synthetic/{task_id}"

[metadata]
split = "{split}"
source = "{method}"
environment = {json.dumps(environment, ensure_ascii=False)}

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 60.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
gpus = 0
"""


def teacher_notes(case, options):
    key = os.environ.get(options["api_key_env"])
    if not key:
        raise ValueError(f"Set {options['api_key_env']} for --teacher")
    location = urlsplit(options["base_url"])
    if (location.scheme != "https" or not location.hostname or location.username
            or location.password or location.query or location.fragment):
        raise ValueError("Teacher base_url must be an HTTPS API root without embedded credentials")
    samples = {name: "".join(value.splitlines(keepends=True)[:4])[:500]
               for name, value in sorted(case.files.items())}
    messages = [
        {"role": "system", "content": (
            "Draft an offline data-processing benchmark briefing. Return a single JSON object "
            "with string fields scenario, edge_cases, demonstration. scenario is one short "
            "English paragraph describing the business situation without adding requirements. "
            "edge_cases is a short paragraph highlighting edge cases already covered by the contract. "
            "demonstration explains a correct algorithm with pseudocode. Use only the supplied rules. "
            "The scenario must not contain code, external links, directives, or answers. "
            'Return raw JSON, e.g. {"scenario":"Reconcile a snapshot.",'
            '"edge_cases":"Use the reporting cutoff.","demonstration":"Filter rows, then aggregate."}'
        )},
        {"role": "user", "content": json.dumps({"contract": case.rules, "synthetic_samples": samples})},
    ]
    payload = {"model": options["model"], "messages": messages, "max_tokens": 4096,
               "temperature": 0.25, "response_format": {"type": "json_object"}}
    if options.get("disable_thinking"):
        payload["thinking"] = {"type": "disabled"}
    for retry in range(3):
        if retry:
            messages.append({"role": "user", "content": "For this draft, emit the three JSON strings in under 300 words."})
        request = Request(options["base_url"].rstrip("/") + "/chat/completions",
                          data=json.dumps(payload).encode(), method="POST",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except HTTPError as error:
            raise ValueError(f"Teacher API returned HTTP {error.code}") from None
        except (OSError, URLError) as error:
            raise ValueError(f"Teacher API transport failed ({type(error).__name__})") from None
        try:
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```") and content.endswith("```"):
                content = content.split("\n", 1)[1][:-3].strip()
            value = json.loads(content)
            fields = {name: value[name].strip() for name in ("scenario", "edge_cases", "demonstration")}
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        if all(0 < len(text) <= 3000 for text in fields.values()):
            return fields
    raise ValueError("Teacher response did not match the requested JSON notes format after three requests")


def load_method(target):
    module, separator, name = target.partition(":")
    if not separator:
        raise ValueError("Synthesis plugins use module:function")
    handler = getattr(import_module(module), name)
    if not callable(handler):
        raise ValueError(f"Synthesis plugin is not callable: {target}")
    return handler


def write_case(folder, method, split, case, scenario):
    if (not isinstance(case, Case) or not case.files
            or not isinstance(case.environment, str) or not case.environment.strip()
            or not all(FILENAME.fullmatch(name) for name in case.files)
            or not FILENAME.fullmatch(case.expected["file"])
            or case.expected["file"] == "inputs"
            or case.expected["kind"] not in ("csv", "json") or not case.wrong):
        raise ValueError(f"Invalid case definition for {folder.name}")
    inputs = folder / "environment" / "inputs"
    tests = folder / "tests"
    solution = folder / "solution"
    for directory in (inputs, tests, solution):
        directory.mkdir(parents=True)
    for name, content in case.files.items():
        (inputs / name).write_text(content)
    (folder / "task.toml").write_text(task_toml(folder.name, method, split, case.environment))
    (folder / "instruction.md").write_text(f"{scenario}\n\n{case.rules}\n")
    (inputs.parent / "Dockerfile").write_text(DOCKERFILE)
    shutil.copyfile(HERE / "synthesis_verify.py", tests / "verify.py")
    save_json(tests / "expected.json", case.expected)
    save_json(tests / "typical_wrong.json", case.wrong)
    (tests / "test.sh").write_text(VERIFY)
    (solution / "solve.sh").write_text(ORACLE)
    (solution / "solve.py").write_text(case.reference)
    for script in (tests / "test.sh", solution / "solve.sh"):
        script.chmod(0o755)


def generate(args):
    config = tomllib.loads(args.config.read_text())
    settings = config.get("synthesis", {})
    methods = settings.get("methods", {})
    writers = settings.get("writers", {})
    method_options = settings.get("options", {})
    group_sizes = settings.get("seed_groups", {})
    names = list(dict.fromkeys(args.method or methods))
    if not SLUG.fullmatch(args.name) or not names or not all(
        SLUG.fullmatch(name) and name in methods for name in names
    ):
        raise ValueError("Use a lowercase dataset slug and registered method names")
    quantities = {"train": args.train_per_method, "holdout": args.holdout_per_method}
    if min(quantities.values()) < 0 or not sum(quantities.values()):
        raise ValueError("Specify a positive number of tasks in at least one split")
    output = args.output.resolve() / args.name
    if output.exists():
        raise ValueError(f"Dataset already exists: {output}. Use a new version name.")
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher = {**settings.get("teacher", {})}
    if args.teacher_model:
        teacher["model"] = args.teacher_model
    if args.teacher and not all(teacher.get(k) for k in ("base_url", "model", "api_key_env")):
        raise ValueError("Configure [synthesis.teacher] before enabling the teacher")
    notes, files = {}, []
    bundle = config.get("bundle")
    if bundle and not Path(bundle).expanduser().is_absolute():
        bundle = str((args.config.resolve().parent / bundle).resolve())
    registry = [f"bundle = {json.dumps(bundle)}"] if bundle else []
    if config.get("opencode_version"):
        registry.append(f"opencode_version = {json.dumps(config['opencode_version'])}")
    with TemporaryDirectory(dir=output.parent, prefix=f".{args.name}.") as stage:
        staged = Path(stage) / args.name
        staged.mkdir()
        for split, count in quantities.items():
            if not count:
                continue
            dataset_id = f"{args.name}-{split}"
            registry.extend([f"\n[datasets.{dataset_id}]", f'name = "{dataset_id}"', f'path = "{split}"'])
            for method in names:
                build = load_method(methods[method])
                writer = load_method(writers[method]) if method in writers else write_case
                group_size = group_sizes.get(method, 1)
                if type(group_size) is not int or group_size < 1:
                    raise ValueError(f"Invalid seed group size for {method}")
                for index in range(1, count + 1):
                    seed_index = 1 + (index - 1) // group_size
                    natural_seed = f"{args.seed}:{split}:{method}:{seed_index}"
                    seed = int.from_bytes(hashlib.sha256(natural_seed.encode()).digest()[:8], "big")
                    case = build(split, index, seed, **method_options.get(method, {}))
                    template_id = f"{method}-{split}"
                    if args.teacher and index == 1:
                        notes[template_id] = teacher_notes(case, teacher)
                        print(f"Teacher briefing ready for {template_id}", flush=True)
                    scenario = notes.get(template_id, {}).get("scenario", case.scenario)
                    task_id = f"{method}-{split}-{index:03}"
                    writer(staged / split / task_id, method, split, case, scenario)
                    record = {"task_id": task_id, "method": method, "split": split,
                              "environment": case.environment, "seed": seed}
                    if group_size > 1:
                        record["seed_group"] = seed_index
                    if getattr(case, "provenance", None):
                        record["generation"] = case.provenance
                    files.append(record)
        (staged / "harbor.toml").write_text("\n".join(registry) + "\n")
        save_json(staged / "synthesis.json", {"dataset": args.name, "seed": args.seed,
                                             "teacher_model": teacher["model"] if args.teacher else None,
                                             "teacher_notes": notes, "tasks": files})
        staged.rename(output)
    print(f"Wrote {len(files)} Harbor tasks to {output}")


def write_candidate(path, expected, rows):
    if expected["kind"] == "csv":
        path.write_text(csv_text(expected["columns"], rows))
    else:
        save_json(path, rows)


def validate_case(task, scratch, audit):
    tests = task / "tests"
    expected_file = tests / "expected.json"
    expected = json.loads(expected_file.read_text())
    app = scratch / "app"
    shutil.copytree(task / "environment" / "inputs", app / "inputs")
    reward = scratch / "reward.txt"
    candidate = app / expected["file"]
    verifier = [sys.executable, "-B", str(tests / "verify.py"), str(app)]

    def verify(target=expected_file):
        reward.unlink(missing_ok=True)
        subprocess.run([*verifier, str(target), str(reward)], check=True, capture_output=True, timeout=15)
        return int(reward.read_text().strip())

    if verify() != 0:
        raise ValueError("empty submission must score 0")
    subprocess.run([sys.executable, "-B", str(task / "solution" / "solve.py"), str(app)],
                   check=True, capture_output=True, timeout=15)
    if verify() != 1:
        raise ValueError("reference solver must score 1")
    # Auditors may cross-check the public question and task metadata alongside the scratch inputs.
    for name in ("instruction.md", "task.toml"):
        shutil.copyfile(task / name, scratch / name)
    independent = audit(app / "inputs") if audit else expected["rows"]
    write_candidate(candidate, expected, independent)
    if verify() != 1:
        raise ValueError("independently derived correct output must score 1")
    for wrong in json.loads((tests / "typical_wrong.json").read_text()):
        write_candidate(candidate, expected, wrong)
        if verify() != 0:
            raise ValueError("typical wrong fixture must score 0")
    corrupt = scratch / "invalid-expected.json"
    corrupt.write_text("{")
    reward.unlink(missing_ok=True)
    failure = subprocess.run([*verifier, str(corrupt), str(reward)], capture_output=True, timeout=15)
    if failure.returncode == 0 or reward.exists():
        raise ValueError("verifier fault must return a nonzero exit without a reward")


def check(args):
    folders = [args.path] if (args.path / "task.toml").is_file() else sorted(
        task.parent for split in ("train", "holdout") for task in (args.path / split).glob("*/task.toml")
    )
    if not folders:
        raise ValueError("No synthetic Harbor tasks found")
    config = tomllib.loads(args.config.read_text())
    settings = config.get("synthesis", {})
    auditors = settings.get("auditors", {})
    checkers = settings.get("checkers", {})
    failed = {}
    independent = 0
    for task in folders:
        with TemporaryDirectory(prefix="tianxi-synth-check-") as scratch:
            try:
                method = tomllib.loads((task / "task.toml").read_text()).get("metadata", {}).get("source")
                audit = load_method(auditors[method]) if method in auditors else None
                checker = load_method(checkers[method]) if method in checkers else validate_case
                checker(task, Path(scratch), audit)
                independent += int(audit is not None)
            except (OSError, ValueError, KeyError, AttributeError, ImportError, subprocess.SubprocessError) as error:
                failed[task.name] = str(error)
    print(json.dumps({"tested": len(folders), "passed": len(folders) - len(failed),
                      "independent_audit": independent, "failed": failed}, ensure_ascii=False))
    return int(bool(failed))


def select(args):
    rows = [json.loads(line) for line in args.episodes.read_text().splitlines() if line.strip()]
    if not rows or args.attempts < 2 or args.min_grpo < 1:
        raise ValueError("Supply nonempty episodes, at least two attempts, and a positive GRPO threshold")
    if {row["harness"] for row in rows} != {args.harness}:
        raise ValueError(f"Selection requires one harness: {args.harness}")
    models = {row["model_route"] for row in rows}
    datasets = {row["dataset"] for row in rows}
    if len(models) != 1 or len(datasets) != 1 or not next(iter(datasets)).endswith("-train"):
        raise ValueError("Selection requires one model route and one training candidate dataset")
    if len({row["trial_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate trial IDs")
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    decisions = []
    for task_id, group in sorted(groups.items()):
        scored = [r for r in group if r["status"] == "scored"
                  and type(r["reward"]) in (int, float) and r["reward"] in (0, 1)]
        scored_ids = {r["trial_id"] for r in scored}
        excluded = Counter(r.get("exception_type") or
                           ("invalid_reward" if r["status"] == "scored" else r["status"]) for r in group
                           if r["trial_id"] not in scored_ids)
        correct = sum(r["reward"] == 1 for r in scored)
        hashes = {r.get("task_checksum") for r in group}
        versions = {r.get("harness_version") for r in group if r.get("harness_version")}
        if (len(group) != args.attempts or len(scored) != args.attempts
                or len(hashes) != 1 or None in hashes or len(versions) > 1):
            lane = "review"
        elif correct == 0:
            lane = "teacher"
        elif correct == args.attempts:
            lane = "regression"
        else:
            lane = "grpo"
        decisions.append({"task_id": task_id, "task_directory": task_id.rsplit("/", 1)[-1],
                          "attempts": len(group), "scored": len(scored),
                          "passed": correct, "excluded": dict(excluded), "lane": lane})
    totals = Counter(item["lane"] for item in decisions)
    report = {"dataset": next(iter(datasets)), "harness": args.harness, "model": next(iter(models)),
              "required_attempts": args.attempts, "min_grpo": args.min_grpo,
              "threshold_met": totals["grpo"] >= args.min_grpo, "totals": dict(totals), "tasks": decisions}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, report)
        print(json.dumps(report["totals"]))
    else:
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("generate")
    build.add_argument("name", help="Dataset version name")
    build.add_argument("--config", type=Path, default=HERE / "harbor.toml")
    build.add_argument("--output", type=Path, default=HERE.parent / "datasets")
    build.add_argument("--method", action="append", help="Registered method. Repeat for multiple methods.")
    build.add_argument("--train-per-method", type=int, default=20)
    build.add_argument("--holdout-per-method", type=int, default=6)
    build.add_argument("--seed", type=int, default=20261124)
    build.add_argument("--teacher", action="store_true")
    build.add_argument("--teacher-model")
    validate = commands.add_parser("check")
    validate.add_argument("path", type=Path, help="Generated dataset or single task directory")
    validate.add_argument("--config", type=Path, default=HERE / "harbor.toml")
    choose = commands.add_parser("select")
    choose.add_argument("episodes", type=Path)
    choose.add_argument("--harness", default="opencode")
    choose.add_argument("--attempts", type=int, default=8)
    choose.add_argument("--min-grpo", type=int, default=10)
    choose.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        return {"generate": generate, "check": check, "select": select}[args.command](args)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

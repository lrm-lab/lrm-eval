#!/usr/bin/env python3
"""Recompute benchmark coverage, scores, and evidence checks from a batch."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import tempfile


def load_json(path):
    return json.loads(path.read_text())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_batch(path):
    root = (path if path.is_dir() else path.parent).resolve()
    index = root / "episodes.jsonl" if path.is_dir() else path.resolve()
    meta_path = root / "manifest.json"
    progress_path = root / "batch-status.json"
    meta = load_json(meta_path) if meta_path.is_file() else {}
    progress = load_json(progress_path) if progress_path.is_file() else {}
    rows = []
    seen = set()
    if index.is_file():
        for line_number, line in enumerate(index.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get("trial_id") if isinstance(row, dict) else None
            if (not isinstance(identifier, str) or not identifier or identifier in seen
                    or not isinstance(row.get("task_id"), str) or not row["task_id"]
                    or not isinstance(row.get("status"), str)):
                raise ValueError(f"Invalid or duplicate episode at line {line_number}")
            rows.append(row)
            seen.add(identifier)
    elif not path.is_dir() or not progress:
        raise ValueError(f"Missing episode index: {index}")
    return root, rows, meta, progress


def count_summary(rows, planned, pass_threshold):
    outcomes = Counter()
    scores = []
    for row in rows:
        status = row["status"]
        if status == "scored":
            if finite(row.get("reward")):
                scores.append(row["reward"])
            else:
                status = "invalid_reward"
        outcomes[status] += 1
    if planned is not None:
        if type(planned) is not int or planned < len(rows) or planned <= 0:
            raise ValueError("Planned attempts must be positive and cover all recorded attempts")
        if planned > len(rows):
            outcomes["unrecorded"] = planned - len(rows)
    total = math.fsum(scores)
    return {
        "planned": planned, "recorded": len(rows), "scored": len(scores),
        "coverage": len(scores) / planned if planned is not None else None,
        "reward_sum": total, "valid_mean": total / len(scores) if scores else None,
        "passed": sum(value >= pass_threshold for value in scores) if pass_threshold is not None else None,
        "pass_threshold": pass_threshold, "outcomes": dict(sorted(outcomes.items())),
    }


def planned_count(override, progress, task_attempts):
    values = [n for n in (override, progress.get("planned"),
                         sum(task_attempts.values()) if task_attempts else None) if n is not None]
    if any(type(n) is not int or n <= 0 for n in values) or len(set(values)) > 1:
        raise ValueError("Planned attempt counts disagree")
    return values[0] if values else None


def read_official(path, dataset):
    item = load_json(path)
    if (not isinstance(item, dict) or item.get("dataset") != dataset
            or not isinstance(item.get("metric"), str) or not item["metric"]
            or not isinstance(item.get("source"), str) or not item["source"]
            or not finite(item.get("score"))
            or not finite(item.get("denominator")) or item["denominator"] <= 0):
        raise ValueError("Official result needs the matching dataset, metric, source, score and positive denominator")
    return {key: item[key] for key in ("dataset", "metric", "source", "score", "denominator")}


def summarize(rows, meta, progress, *, planned=None, attempts_per_task=None,
              pass_threshold=None, official=None):
    if pass_threshold is not None and not finite(pass_threshold):
        raise ValueError("The pass threshold must be finite")
    dimensions = {}
    for target, field in (("dataset", "dataset"), ("model", "model_route"), ("harness", "harness")):
        values = {r[field] for r in rows if r.get(field) is not None}
        if meta.get(target) is not None:
            values.add(meta[target])
        if len(values) > 1:
            raise ValueError(f"Episodes contain multiple {target} values")
        dimensions[target] = next(iter(values), None)

    task_attempts = progress.get("task_attempts") or {}
    if (not isinstance(task_attempts, dict)
            or any(type(n) is not int or n <= 0 for n in task_attempts.values())):
        raise ValueError("Per-task plan must contain positive attempt counts")
    planned = planned_count(planned, progress, task_attempts)
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    if task_attempts and set(groups) - task_attempts.keys():
        raise ValueError("Recorded task IDs are outside this batch's plan")
    if attempts_per_task is not None and (attempts_per_task <= 0 or
            any(n != attempts_per_task for n in task_attempts.values())):
        raise ValueError("The per-task attempt count disagrees with the plan")
    if attempts_per_task and planned is not None and len(groups) * attempts_per_task > planned:
        raise ValueError("Per-task attempt counts exceed the overall plan")
    valid_scores = [r["reward"] for r in rows if r["status"] == "scored" and finite(r.get("reward"))]
    if (pass_threshold is None and valid_scores and all(x in (0, 1) for x in valid_scores)
            and dimensions["dataset"] != "wildclawbench-harbor"):
        pass_threshold = 1
    totals = count_summary(rows, planned, pass_threshold)
    tasks = {
        name: count_summary(groups[name], task_attempts.get(name, attempts_per_task), pass_threshold)
        for name in sorted(groups.keys() | task_attempts.keys())
    }
    usage = {}
    for key in ("input_tokens", "output_tokens", "cost_usd", "duration_seconds"):
        values = [(r.get("usage") or {}).get(key) for r in rows]
        values = [v for v in values if finite(v) and v >= 0]
        usage[key] = {"observed": len(values), "total": math.fsum(values) if values else None}
    return {
        "schema_version": 1, "batch": meta.get("batch"), **dimensions,
        "batch_status": progress.get("status"), "totals": totals, "tasks": tasks, "usage": usage,
        "failure_kinds": dict(sorted(Counter(
            (r.get("failure") or {}).get("kind") or r["exception_type"]
            for r in rows if r.get("exception_type")
        ).items())),
        "trajectories": dict(sorted(Counter(r.get("trajectory_status", "unknown") for r in rows).items())),
        "training": dict(sorted(Counter(r.get("training_status", "not_captured") for r in rows).items())),
        "official": read_official(official, dimensions["dataset"]) if official else None,
    }


def artifact(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Artifact reference must be batch-relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact is outside the batch")
    return path


def has_tool_execution(trajectory):
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
        return False
    calls = set()
    for step in trajectory["steps"]:
        if not isinstance(step, dict):
            continue
        calls.update(call["tool_call_id"] for call in step.get("tool_calls") or []
                     if isinstance(call, dict) and isinstance(call.get("tool_call_id"), str)
                     and isinstance(call.get("function_name"), str) and call["function_name"])
        for result in (step.get("observation") or {}).get("results") or []:
            if not isinstance(result, dict):
                continue
            call = result.get("source_call_id")
            if isinstance(call, str) and call in calls and not result.get("is_error") and (
                result.get("content") is not None or result.get("subagent_trajectory_ref")
            ):
                return True
    return False


def check_smoke(root, rows, summary, minimum_reward):
    if not finite(minimum_reward):
        raise ValueError("The smoke minimum reward must be finite")
    reasons = Counter()
    planned, recorded = summary["totals"]["planned"], len(rows)
    if planned is None or recorded != planned:
        reasons["incomplete_or_unknown_plan"] += 1
    for row in rows:
        if row["status"] != "scored" or not finite(row.get("reward")) or row["reward"] < minimum_reward:
            reasons["reward_check_failed"] += 1
            continue
        try:
            receipt = load_json(artifact(root, row.get("result")))
            trajectory = load_json(artifact(root, row.get("trajectory")))
            rewards = receipt.get("verifier_rewards") or {}
            if receipt.get("id") != row["trial_id"] or rewards.get(receipt.get("reward_key", "reward")) != row["reward"]:
                reasons["receipt_mismatch"] += 1
            elif not has_tool_execution(trajectory):
                reasons["no_matching_tool_observation"] += 1
        except (OSError, ValueError, TypeError, AttributeError):
            reasons["missing_or_invalid_artifact"] += 1
    return {"passed": bool(rows) and not reasons, "minimum_reward": minimum_reward,
            "reasons": dict(sorted(reasons.items()))}


def check_training_captures(root, rows, summary):
    from training_capture import load_training_capture

    reasons = Counter()
    checkpoints = set()
    planned = summary["totals"]["planned"]
    if not rows or planned is None or len(rows) != planned:
        reasons["incomplete_or_unknown_plan"] += 1
    for row in rows:
        if row.get("training_status") != "validated" or row["status"] != "scored" or not finite(row.get("reward")):
            reasons["no_scored_validated_capture"] += 1
            continue
        try:
            trace = artifact(root, row.get("training_trace"))
            if trace.parent.name != row["trial_id"] or trace.name != "training_trace.json":
                raise ValueError("Training trace does not match the trial")
            artifact(root, str(trace.relative_to(root).with_name("capture.json")))
            capture, _ = load_training_capture(trace.parent.parent, row["trial_id"])
            if capture["checkpoint"] != row.get("checkpoint"):
                raise ValueError("Training checkpoint does not match the episode")
            checkpoints.add(capture["checkpoint"])
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            reasons["invalid_capture"] += 1
    if len(checkpoints) > 1:
        reasons["mixed_checkpoints"] += 1
    return {"passed": len(checkpoints) == 1 and not reasons, "reasons": dict(sorted(reasons.items()))}


def write_report(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(contents)
            os.fchmod(stream.fileno(), 0o644)
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def as_markdown(report):
    def label(value):
        return str(value).replace("|", "\\|").replace("\n", " ") if value is not None else "—"

    def cells(name, item):
        planned = item["planned"]
        scored = item["scored"]
        coverage = f"{scored}/{planned} ({item['coverage']:.1%})" if planned else "—"
        passed = f"{item['passed']}/{scored}" if item["passed"] is not None else "—"
        mean = f"{item['valid_mean']:.4f}" if item["valid_mean"] is not None else "—"
        outcomes = ", ".join(f"{label(k)} {v}" for k, v in item["outcomes"].items())
        return f"| {label(name)} | {label(planned)} | {item['recorded']} | {scored} | {passed} | {coverage} | {mean} | {outcomes} |"

    lines = [
        f"# {label(report['batch'] or report['dataset'] or 'Batch report')}", "",
        f"{label(report['dataset'])} · {label(report['harness'])} · {label(report['model'])}", "",
        "| 范围 | 计划 | 已有结果 | 有效评分 | 有效样本通过 | 覆盖率 | 有效均分 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        cells("批次", report["totals"]),
        *(cells(name, summary) for name, summary in report["tasks"].items()),
    ]
    if official := report["official"]:
        lines.extend(["", f"官方结果 **{label(official['metric'])}** {official['score']}，官方分母 "
                           f"{official['denominator']}。来源 {label(official['source'])}"])
    for name, title in (("smoke_check", "冒烟检查"), ("training_capture_check", "训练采集检查")):
        if verdict := report.get(name):
            details = ", ".join(f"{label(k)} {v}" for k, v in verdict["reasons"].items())
            lines.extend(["", f"{title} **{'通过' if verdict['passed'] else '待验收'}** {details}".rstrip()])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Batch directory or episodes.jsonl")
    parser.add_argument("--planned", type=int, help="Planned attempts when the batch has no saved plan")
    parser.add_argument("--attempts-per-task", type=int, help="Per-task denominator for an older batch")
    parser.add_argument("--pass-threshold", type=float, help="Optional pass threshold for graded tasks")
    parser.add_argument("--official-result", type=Path, help="Result from the benchmark's official scorer")
    parser.add_argument("--check-smoke", action="store_true", help="Require a scored result and actual tool observation")
    parser.add_argument("--min-smoke-reward", type=float, default=1)
    parser.add_argument("--check-training-captures", action="store_true", help="Require verified, scored token captures")
    parser.add_argument("--output", type=Path, help="Write JSON to a file")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    try:
        root, rows, meta, progress = read_batch(args.batch)
        result = summarize(rows, meta, progress, planned=args.planned,
                           attempts_per_task=args.attempts_per_task, pass_threshold=args.pass_threshold,
                           official=args.official_result)
        tests = []
        if args.check_smoke:
            result["smoke_check"] = check_smoke(root, rows, result, args.min_smoke_reward)
            tests.append(result["smoke_check"]["passed"])
        if args.check_training_captures:
            result["training_capture_check"] = check_training_captures(root, rows, result)
            tests.append(result["training_capture_check"]["passed"])
        content = as_markdown(result) if args.format == "markdown" else (
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if args.output:
            write_report(args.output, content)
        else:
            print(content, end="")
        return int(not all(tests))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
